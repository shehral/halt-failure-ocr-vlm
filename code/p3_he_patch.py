"""Phase 3 / Job 4 — H-E component-resolved causal patching (FCCT-style).

Implementation note: we use **direct PyTorch forward-hooks** rather than the
pyvene `IntervenableModel` API. Reasons: (1) pyvene's standard pattern is
source-to-base activation patching on a *pair* of forward passes, while we
patch a precomputed source-residual into a single live generation; (2) the
existing project tooling (fix/halt_monitor.py, code/p1_cache_harness.py)
already speaks the hook API on the same modules
(`model.model.language_model.layers[L].self_attn` for MHSA,
`.mlp` for FFN, the layer itself for block_output); (3) avoids adding a
new dependency at this stage. pyvene's RepresentationConfig component names
(mlp_output / attention_output / block_output) are kept as the public
argument labels in this script so the protocol matches the published FCCT
framing.

Per (positive doc, layer in TARGET ∪ OFF-TARGET, component in
{attention_output, mlp_output, block_output}):
  1. Load model with the mandatory lm_head workaround.
  2. Run baseline generation (no patch) to confirm the positive reproduces
     and record baseline_tokens.
  3. Compute SOURCE = mean L_n residual of matched controls at the
     decision-moment position.
  4. Patch SOURCE into the positive's forward at the decision-moment step
     by overwriting the component's output at the LAST sequence position
     of the prefill forward (the position that produces the next token).
  5. Record three negative controls per (layer, component):
       - random_matched_norm   (random direction, same L2 norm as SOURCE)
       - off_layer             (SOURCE drawn from L_(n+4); falls back to L_(n-4))
       - shuffled_values       (SOURCE dims shuffled — preserves norm + marginal)

For each combination, write
  results/p3_he_patch/<doc>/<layer>_<component>_<intervention>.json

B3 — Reverse-direction off-target patch (causal sufficiency test):
  For each (layer, component), ALSO patch the positive's in-loop residual at
  matched gen-position INTO a CONTROL doc's forward, then generate from the
  control. If `tokens_emitted > 2 * baseline_control_tokens`, the patch
  INDUCED a halt failure → L24 is causally SUFFICIENT, not merely necessary.
  Output: <control_doc>/L<layer>_<component>_reverse_from_<positive_doc>.json

B4 — Off-target layer controls:
  L0, L8, L35 are added to the sweep and tagged as "off-target controls"
  via the `layer_role` field in each output record. PASS-strict now ALSO
  requires that no component at L0/L8/L35 reduces failure ≥10%.

B5 — Per-failure-class breakdown:
  _summary.json now contains a top-level `by_class` section keyed by the
  doc's failure class (pulled from results/p1_trigger_v2/<doc>.json's
  failure_class field, with fallback to failure_taxonomy.json). PASS-strict
  ALSO requires the SAME component to dominate across ALL tested classes;
  otherwise CL-12 (unified mechanism) is falsified at this scale.

After the full sweep, write results/p3_he_patch/_summary.json with the
PRE-REGISTERED PASS criterion evaluated inline (PASS-strict / PASS-soft / FAIL).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "nanonets/Nanonets-OCR2-3B"
MODEL_REV = "c3886ff00bb037ce7da24988c9eafaf1fe2bed72"
EOS_TOKEN_ID = 151645  # <|im_end|>

# B4: off-target layer controls. PASS-strict requires no component at these
# layers reduces failure >=10%. If they reduce >=30% too, L24 isn't privileged.
OFF_TARGET_LAYERS = (0, 8, 35)
# B3: reverse-patch INDUCES halt-failure on a control if tokens_emitted exceeds
# this multiplier of the control's natural baseline.
REVERSE_INDUCE_MULT = 2.0
# B3: number of clean-EOS controls to use as the receiver-side of the reverse
# patch test. Limited for compute budget.
N_REVERSE_CONTROLS = 5

ROOT = Path(__file__).resolve().parents[1]
ACT_DIR = ROOT / "results" / "activations"
TRIG_DIR_V2 = ROOT / "results" / "p1_trigger_v2"
# DATA_DIRS now sourced from _constants (canonical, 2026-05-17 refactor); kept here as alias for backward compat
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _constants import DATA_DIRS as _CANONICAL_DATA_DIRS, find_image_path as _canonical_find_image_path
DATA_DIRS = list(_CANONICAL_DATA_DIRS)

USER_PROMPT = (
    "Extract the text from the above document as if you were reading it naturally. "
    "Return the tables in HTML format. Return the equations in LaTeX representation. "
    "If there is an image in the document and image caption is not present, add a small "
    "description of the image inside the <img></img> tag; otherwise, add the image caption "
    "inside <img></img>. Watermarks should be wrapped in brackets. Ex: "
    "<watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. "
    "Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
    "Prefer using ☐ and ☑ for check boxes."
)


# =================================================================
# IO helpers
# =================================================================

def find_doc_image(doc_id: str) -> Path:
    for d in DATA_DIRS:
        m = d / "manifest.json"
        if not m.exists():
            continue
        for entry in json.loads(m.read_text()):
            if entry.get("doc_id") == doc_id:
                return ROOT / entry["path"]
    raise FileNotFoundError(f"No manifest entry for {doc_id} in {DATA_DIRS}")


def load_cached_layer_residual(doc_id: str, layer: int) -> Optional[torch.Tensor]:
    """Return cached (T, hidden) residual for `layer` if present, else None.

    The cache stored by p1_cache_harness.py keeps a subset of layers (every-k);
    here we require the exact layer index to be present.
    """
    d = ACT_DIR / doc_id
    if not (d / "hidden_states.pt").exists():
        return None
    layer_indices = torch.load(d / "layer_indices.pt", weights_only=True).tolist()
    if layer not in layer_indices:
        return None
    li = layer_indices.index(layer)
    hs = torch.load(d / "hidden_states.pt", weights_only=True)
    # hs shape: (num_kept_layers, T, hidden)
    return hs[li].float()  # (T, hidden)


def get_decision_position(doc_id: str, meta: dict) -> int:
    """Decision-moment position in the cached-doc sequence (in PROMPT+GEN index space).

    For a control doc that cleanly halts, the decision moment is the LAST cached
    position — that's where the halt-token probability is high.
    For a positive doc with a loop, p2 analyses use the loop-start position; here
    we use the trigger record's decision_moment_position when available, else the
    last cached gen position.
    """
    trig_path = TRIG_DIR_V2 / f"{doc_id}.json"
    if trig_path.exists():
        trig = json.loads(trig_path.read_text())
        dm = trig.get("decision_moment_position")
        if dm is not None:
            return int(meta["prompt_len"]) + int(dm)
    # Fallback: last cached gen position
    return int(meta["prompt_len"]) + int(meta["gen_len"]) - 1


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load N>=20 manifest_labeled. Schema: list of {doc, label, ...}."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"positives manifest not found at {manifest_path}. "
            f"Run Job 1 (the sweep) first to produce it."
        )
    return json.loads(manifest_path.read_text())


def load_matched_controls(manifest: list[dict], doc_id: str, n: int = 5) -> list[str]:
    """Return up to `n` control doc_ids from the manifest, drawn deterministically.

    Matched-controls semantics: any doc with label=='control'. We take the first
    `n` whose cached activations exist on disk. The choice of `n` is a memory
    knob — larger n gives a smoother SOURCE mean.
    """
    controls = [m["doc"] for m in manifest if m.get("label") == "control"]
    # Stable order: filter by cache availability and take the first n.
    keep = []
    for c in controls:
        if (ACT_DIR / c / "hidden_states.pt").exists():
            keep.append(c)
        if len(keep) >= n:
            break
    return keep


def compute_source_residual(
    control_doc_ids: list[str], layer: int
) -> Optional[torch.Tensor]:
    """SOURCE = mean L_n residual across matched controls at each control's
    decision-moment position.

    Returns a (hidden,) tensor on CPU in float32, or None if no controls usable.
    """
    samples = []
    for cid in control_doc_ids:
        meta_path = ACT_DIR / cid / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        hs = load_cached_layer_residual(cid, layer)
        if hs is None:
            continue
        pos = get_decision_position(cid, meta)
        pos = max(0, min(pos, hs.shape[0] - 1))
        samples.append(hs[pos])
    if not samples:
        return None
    return torch.stack(samples, dim=0).mean(dim=0)  # (hidden,)


# =================================================================
# B3 / B5 helpers — reverse-patch source + failure-class lookup
# =================================================================

def get_in_loop_position(doc_id: str, meta: dict) -> int:
    """Return the in-loop / mid-loop position in PROMPT+GEN index space for a
    POSITIVE doc, used as the source position for the B3 reverse patch.

    Prefers `loop_start_position` from the trigger record, falling back to the
    decision_moment_position, then to a mid-cache position. The reverse patch
    asks: "if we inject the positive's deeply-in-loop residual into a control
    at the matched gen-position, does the control's generation diverge?"
    """
    trig_path = TRIG_DIR_V2 / f"{doc_id}.json"
    if trig_path.exists():
        trig = json.loads(trig_path.read_text())
        for key in ("loop_start_position", "mid_loop_position",
                    "decision_moment_position"):
            v = trig.get(key)
            if v is not None:
                return int(meta["prompt_len"]) + int(v)
    # Fallback: middle of the cached gen.
    return int(meta["prompt_len"]) + int(meta["gen_len"]) // 2


def compute_positive_in_loop_residual(
    positive_doc_id: str, layer: int
) -> Optional[torch.Tensor]:
    """For B3 reverse-patch: pull the positive doc's L_n residual at its
    in-loop (mid-loop) position. Returns (hidden,) on CPU float32 or None.
    """
    meta_path = ACT_DIR / positive_doc_id / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    hs = load_cached_layer_residual(positive_doc_id, layer)
    if hs is None:
        return None
    pos = get_in_loop_position(positive_doc_id, meta)
    pos = max(0, min(pos, hs.shape[0] - 1))
    return hs[pos]  # (hidden,)


def load_failure_class_map() -> dict[str, str]:
    """Build {doc_id -> failure_class} from results/p1_trigger_v2/.

    Priority: each `<doc>.json`'s `failure_class` field. Fallback: the global
    `failure_taxonomy.json` (which maps class -> [doc_ids]).
    """
    out: dict[str, str] = {}
    if TRIG_DIR_V2.exists():
        for p in TRIG_DIR_V2.glob("*.json"):
            if p.name == "failure_taxonomy.json":
                continue
            try:
                d = json.loads(p.read_text())
                fc = d.get("failure_class") or d.get("class") or d.get("label_class")
                if fc:
                    doc_id = d.get("doc_id") or p.stem
                    out[doc_id] = fc
            except Exception:
                pass
    # Fallback: taxonomy index (class -> [docs]).
    taxo_path = TRIG_DIR_V2 / "failure_taxonomy.json"
    if taxo_path.exists():
        try:
            taxo = json.loads(taxo_path.read_text())
            classes = taxo.get("classes", taxo) if isinstance(taxo, dict) else {}
            if isinstance(classes, dict):
                for fc, info in classes.items():
                    docs = info.get("docs") if isinstance(info, dict) else info
                    if isinstance(docs, list):
                        for d in docs:
                            out.setdefault(str(d), fc)
        except Exception:
            pass
    return out


def list_clean_eos_controls(manifest: list[dict], n: int = N_REVERSE_CONTROLS) -> list[str]:
    """For B3 reverse-patch: clean-EOS controls (label=='control') with cached
    activations and a known clean baseline_tokens. Returns up to `n` doc_ids.
    """
    candidates = [m["doc"] for m in manifest if m.get("label") == "control"]
    keep = []
    for c in candidates:
        if (ACT_DIR / c / "hidden_states.pt").exists():
            keep.append(c)
        if len(keep) >= n:
            break
    return keep


# =================================================================
# Model + lm_head workaround
# =================================================================

def load_model(device: str, attn_impl: str = "eager"):
    proc = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REV)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REV,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    # MANDATORY lm_head workaround — see project CLAUDE.md.
    model.config.tie_word_embeddings = True
    model.tie_weights()
    assert (
        model.lm_head.weight.data_ptr()
        == model.model.language_model.embed_tokens.weight.data_ptr()
    ), "lm_head failed to tie — generation will collapse to '!'"
    model = model.to(device)
    model.train(False)
    return proc, model


def prepare_inputs(processor, image_path: Path, device: str):
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


# =================================================================
# Intervention hook
# =================================================================

class _PatchState:
    """Single-shot intervention state: triggers once on the prefill forward
    (the only forward where seq_len > 1), patches the LAST sequence position
    of the targeted component's output, and disarms.

    During HuggingFace .generate() with use_cache=True, the prefill pass sees
    the whole prompt at once (seq_len = prompt_len). Per-token decoding then
    runs seq_len = 1 forwards. We patch the LAST position of the prefill —
    that's the next-token-decision residual.
    """

    def __init__(self, source_vec: torch.Tensor, device: str, dtype: torch.dtype):
        # Cast SOURCE to model dtype + device once.
        self.source = source_vec.to(device=device, dtype=dtype)
        self.fired = False
        self.fire_count = 0  # diagnostic

    def reset(self):
        self.fired = False
        self.fire_count = 0


def make_component_hook(state: _PatchState, component: str):
    """Forward hook that overwrites the LAST-position output of the targeted
    component during the prefill forward.

    `component` selects which module the hook is registered ON in attach_patch:
      - 'mlp_output'       -> hook on layer.mlp
      - 'attention_output' -> hook on layer.self_attn
      - 'block_output'     -> hook on the layer itself

    All three modules return either a Tensor or a tuple whose first element is
    the (B, T, hidden) output we want to patch.
    """

    def hook(_mod, _inp, out):
        # Only patch on the prefill forward (T > 1) and only once.
        if state.fired:
            return out

        if isinstance(out, tuple):
            tensor = out[0]
            rest = out[1:]
        else:
            tensor = out
            rest = None

        # Only patch if this is a multi-token forward (prefill); decode steps
        # have T=1 and we already patched the prefill's last position.
        if tensor.dim() != 3 or tensor.shape[1] <= 1:
            return out

        new_tensor = tensor.clone()
        new_tensor[0, -1, :] = state.source.to(dtype=new_tensor.dtype)
        state.fired = True
        state.fire_count += 1

        if rest is None:
            return new_tensor
        else:
            return (new_tensor, *rest)

    return hook


def attach_patch(model, layer: int, component: str, state: _PatchState):
    """Register a forward hook on the correct sub-module for `component`
    at the requested decoder layer. Returns the handle so the caller can detach.
    """
    layers = model.model.language_model.layers
    target_layer = layers[layer]
    if component == "mlp_output":
        target_mod = target_layer.mlp
    elif component == "attention_output":
        target_mod = target_layer.self_attn
    elif component == "block_output":
        target_mod = target_layer
    else:
        raise ValueError(f"unknown component: {component}")
    return target_mod.register_forward_hook(make_component_hook(state, component))


# =================================================================
# Generation runner
# =================================================================

def run_generation(model, proc, inputs, max_new_tokens: int) -> dict:
    torch.manual_seed(0)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )
    dt = time.time() - t0
    new_ids = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    last_id = new_ids[-1] if new_ids else None
    hit_cap = (len(new_ids) >= max_new_tokens) and (last_id != EOS_TOKEN_ID)
    stop_reason = "max_new_tokens" if hit_cap else ("eos" if last_id == EOS_TOKEN_ID else "other")
    return {
        "tokens_emitted": len(new_ids),
        "stop_reason": stop_reason,
        "elapsed_s": dt,
        "last_token_id": last_id,
    }


# =================================================================
# Sweep over (doc, layer, component, intervention_type)
# =================================================================

INTERVENTIONS = ("real", "random_matched_norm", "off_layer", "shuffled_values")


def build_source_for_intervention(
    *,
    real_source: torch.Tensor,
    intervention: str,
    doc_layer_residuals: dict[int, torch.Tensor],
    layer: int,
    control_doc_ids: list[str],
    seed: int,
) -> Optional[torch.Tensor]:
    """Construct the SOURCE vector to patch in for the given intervention.

    real:                returns real_source as-is.
    random_matched_norm: random unit normal scaled to ||real_source||_2.
    off_layer:           SOURCE computed at L_(n+4); fallback L_(n-4).
    shuffled_values:     real_source dims shuffled with a fixed seed.
    """
    if intervention == "real":
        return real_source
    rng = np.random.default_rng(seed)
    if intervention == "random_matched_norm":
        v = rng.standard_normal(real_source.shape[-1]).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9) * float(real_source.float().norm().item())
        return torch.tensor(v, dtype=real_source.dtype, device=real_source.device)
    if intervention == "off_layer":
        for off_layer in (layer + 4, layer - 4):
            src = compute_source_residual(control_doc_ids, off_layer)
            if src is not None:
                return src.to(dtype=real_source.dtype, device=real_source.device)
        return None
    if intervention == "shuffled_values":
        v = real_source.detach().cpu().float().numpy().copy()
        rng.shuffle(v)
        return torch.tensor(v, dtype=real_source.dtype, device=real_source.device)
    raise ValueError(f"unknown intervention: {intervention}")


def out_record_path(out_dir: Path, doc_id: str, layer: int, component: str, intervention: str) -> Path:
    return out_dir / doc_id / f"L{layer:02d}_{component}_{intervention}.json"


def out_reverse_record_path(out_dir: Path, control_doc: str, layer: int, component: str, positive_doc: str) -> Path:
    return out_dir / control_doc / f"L{layer:02d}_{component}_reverse_from_{positive_doc}.json"


def layer_role_for(layer: int) -> str:
    """B4: tag layer as 'target' or 'off_target_control'."""
    return "off_target_control" if layer in OFF_TARGET_LAYERS else "target"


def run_one_combo(
    *,
    model,
    proc,
    inputs,
    doc_id: str,
    layer: int,
    component: str,
    intervention: str,
    source_vec: torch.Tensor,
    baseline_tokens: int,
    max_new_tokens: int,
    out_dir: Path,
    failure_class: Optional[str] = None,
) -> dict:
    state = _PatchState(source_vec, device=str(source_vec.device), dtype=source_vec.dtype)
    handle = attach_patch(model, layer, component, state)
    try:
        gen = run_generation(model, proc, inputs, max_new_tokens=max_new_tokens)
    finally:
        handle.remove()

    intervention_norm = float(state.source.float().norm().item())
    if baseline_tokens > 0:
        reduction_pct = (1.0 - gen["tokens_emitted"] / baseline_tokens) * 100.0
    else:
        reduction_pct = 0.0

    record = {
        "doc_id": doc_id,
        "layer": layer,
        "layer_role": layer_role_for(layer),
        "component": component,
        "intervention": intervention,
        "tokens_emitted": gen["tokens_emitted"],
        "stop_reason": gen["stop_reason"],
        "baseline_tokens": int(baseline_tokens),
        "reduction_pct": reduction_pct,
        "elapsed_s": gen["elapsed_s"],
        "intervention_norm": intervention_norm,
        "intervention_cosine_to_source": 1.0 if intervention == "real" else None,
        "fired_count": state.fire_count,
        "failure_class": failure_class,
    }
    rp = out_record_path(out_dir, doc_id, layer, component, intervention)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(record, indent=2))
    return record


def run_reverse_patch_combo(
    *,
    model,
    proc,
    control_inputs,
    control_doc: str,
    positive_doc: str,
    layer: int,
    component: str,
    source_vec: torch.Tensor,
    baseline_control_tokens: int,
    max_new_tokens: int,
    out_dir: Path,
) -> dict:
    """B3: patch positive's in-loop residual INTO a control doc's forward.
    If `tokens_emitted_with_patch > REVERSE_INDUCE_MULT * baseline_control_tokens`,
    the patch INDUCED a halt failure → L24 causally SUFFICIENT.
    """
    state = _PatchState(source_vec, device=str(source_vec.device), dtype=source_vec.dtype)
    handle = attach_patch(model, layer, component, state)
    try:
        gen = run_generation(model, proc, control_inputs, max_new_tokens=max_new_tokens)
    finally:
        handle.remove()

    induced = (
        baseline_control_tokens > 0
        and gen["tokens_emitted"] > REVERSE_INDUCE_MULT * baseline_control_tokens
    )

    record = {
        "control_doc": control_doc,
        "positive_doc_source": positive_doc,
        "layer": layer,
        "layer_role": layer_role_for(layer),
        "component": component,
        "intervention": "reverse_patch",
        "tokens_emitted_with_patch": gen["tokens_emitted"],
        "baseline_control_tokens": int(baseline_control_tokens),
        "induced_failure": bool(induced),
        "stop_reason": gen["stop_reason"],
        "elapsed_s": gen["elapsed_s"],
        "intervention_norm": float(state.source.float().norm().item()),
        "fired_count": state.fire_count,
    }
    rp = out_reverse_record_path(out_dir, control_doc, layer, component, positive_doc)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(record, indent=2))
    return record


def write_summary(
    out_dir: Path,
    all_records: list[dict],
    reverse_records: Optional[list[dict]] = None,
) -> dict:
    """Aggregate per-(layer, component, intervention) statistics and evaluate
    the pre-registered pass criterion.

    Additions vs the original criterion:
      B3 — `reverse_patch` aggregation under `reverse_patch_summary`.
      B4 — off-target layer mean-reduction summary (`off_target_layer_summary`)
            and a new sub-clause in PASS-strict: no component at L0/L8/L35
            reduces failure >=10%.
      B5 — `by_class` block, and a new sub-clause in PASS-strict: the SAME
            component dominates across ALL tested failure classes.
    """
    from collections import defaultdict

    reverse_records = reverse_records or []

    bucket: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for r in all_records:
        key = (r["layer"], r["component"], r["intervention"])
        bucket[key].append(float(r["reduction_pct"]))

    agg: dict[str, dict] = {}
    for (layer, comp, interv), vals in bucket.items():
        key = f"L{layer:02d}/{comp}/{interv}"
        agg[key] = {
            "layer": layer,
            "component": comp,
            "intervention": interv,
            "n_docs": len(vals),
            "mean_reduction_pct": float(np.mean(vals)) if vals else 0.0,
            "median_reduction_pct": float(np.median(vals)) if vals else 0.0,
            "std_reduction_pct": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }

    def mean_at(layer: int, comp: str, interv: str) -> float:
        v = agg.get(f"L{layer:02d}/{comp}/{interv}")
        return v["mean_reduction_pct"] if v else float("nan")

    # -----------------------------------------------------------------
    # B5 — per-failure-class breakdown
    # Bucket records by (failure_class, layer, component) over the real
    # intervention only, and compute mean / n_docs / n_pass_30pct.
    # -----------------------------------------------------------------
    class_bucket: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for r in all_records:
        if r.get("intervention") != "real":
            continue
        fc = r.get("failure_class") or "unknown"
        class_bucket[(fc, r["layer"], r["component"])].append(float(r["reduction_pct"]))

    by_class: dict[str, dict] = {}
    for (fc, layer, comp), vals in class_bucket.items():
        L_key = f"L{layer:02d}"
        by_class.setdefault(fc, {}).setdefault(L_key, {})[comp] = {
            "mean_reduction_pct": float(np.mean(vals)) if vals else 0.0,
            "median_reduction_pct": float(np.median(vals)) if vals else 0.0,
            "n_docs": len(vals),
            "n_pass_30pct": int(sum(1 for v in vals if v >= 30.0)),
        }

    # Per-class dominant component at L24 (real intervention, mean reduction).
    L24_class_dominant: dict[str, Optional[str]] = {}
    for fc, layers_dict in by_class.items():
        l24 = layers_dict.get("L24")
        if not l24:
            L24_class_dominant[fc] = None
            continue
        best_comp, best_val = None, -float("inf")
        for comp, stats in l24.items():
            if stats["mean_reduction_pct"] > best_val:
                best_val = stats["mean_reduction_pct"]
                best_comp = comp
        L24_class_dominant[fc] = best_comp

    valid_class_dominants = [c for c in L24_class_dominant.values() if c is not None]
    same_component_across_classes = (
        len(valid_class_dominants) >= 2
        and len(set(valid_class_dominants)) == 1
    )
    unified_component = valid_class_dominants[0] if same_component_across_classes else None

    # -----------------------------------------------------------------
    # B4 — off-target layer summary. PASS-strict requires no component at
    # L0/L8/L35 reduces failure >=10%.
    # -----------------------------------------------------------------
    off_target_layer_summary: dict[str, dict] = {}
    off_target_ok = True
    for L in OFF_TARGET_LAYERS:
        per_comp: dict[str, float] = {}
        for comp in ("mlp_output", "attention_output", "block_output"):
            v = mean_at(L, comp, "real")
            per_comp[comp] = v
            if not np.isnan(v) and v >= 10.0:
                off_target_ok = False
        off_target_layer_summary[f"L{L:02d}"] = per_comp

    # -----------------------------------------------------------------
    # B3 — reverse-patch summary. SUFFICIENT if any (layer, component)
    # induced failure (tokens_emitted > 2 * baseline) on >=1 control.
    # -----------------------------------------------------------------
    reverse_bucket: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in reverse_records:
        reverse_bucket[(r["layer"], r["component"])].append(r)

    reverse_patch_summary: dict[str, dict] = {}
    causally_sufficient = False
    causally_sufficient_at: list[str] = []
    for (layer, comp), recs in reverse_bucket.items():
        n_induced = sum(1 for r in recs if r.get("induced_failure"))
        key = f"L{layer:02d}/{comp}"
        reverse_patch_summary[key] = {
            "n_controls": len(recs),
            "n_induced_failure": n_induced,
            "induce_rate": (n_induced / len(recs)) if recs else 0.0,
            "mean_tokens_emitted": float(np.mean([r["tokens_emitted_with_patch"] for r in recs])) if recs else 0.0,
            "mean_baseline_control_tokens": float(np.mean([r["baseline_control_tokens"] for r in recs])) if recs else 0.0,
        }
        if n_induced >= 1:
            causally_sufficient = True
            causally_sufficient_at.append(key)

    causal_sufficiency_verdict = (
        "SUFFICIENT" if causally_sufficient else "NECESSARY_BUT_NOT_SUFFICIENT"
    )

    # -----------------------------------------------------------------
    # PASS-strict / PASS-soft / FAIL — computed at L24 with B3-B5 add-ons.
    # -----------------------------------------------------------------
    l24_mlp_real = mean_at(24, "mlp_output", "real")
    l24_attn_real = mean_at(24, "attention_output", "real")
    l24_block_real = mean_at(24, "block_output", "real")

    def neg_controls_under_10(comp: str) -> bool:
        return all(
            (not np.isnan(mean_at(24, comp, neg))) and mean_at(24, comp, neg) < 10.0
            for neg in ("random_matched_norm", "off_layer", "shuffled_values")
        )

    pass_strict_base = False
    pass_strict_component: Optional[str] = None
    if not np.isnan(l24_mlp_real) and not np.isnan(l24_attn_real):
        if l24_mlp_real >= 30.0 and l24_attn_real < 10.0 and neg_controls_under_10("mlp_output"):
            pass_strict_base = True
            pass_strict_component = "mlp_output"
        elif l24_attn_real >= 30.0 and l24_mlp_real < 10.0 and neg_controls_under_10("attention_output"):
            pass_strict_base = True
            pass_strict_component = "attention_output"

    # B4 sub-clause: off-target layers must not reduce >=10%.
    # B5 sub-clause: same component dominates across all classes, AND that
    # unified component must equal `pass_strict_component`.
    unified_matches = (
        same_component_across_classes
        and pass_strict_component is not None
        and unified_component == pass_strict_component
    )
    pass_strict = pass_strict_base and off_target_ok and unified_matches

    pass_soft = False
    if not pass_strict and not np.isnan(l24_block_real) and l24_block_real >= 30.0:
        pass_soft = True

    # FAIL: no component at any of L20..L28 exceeds 10% real-direction reduction.
    fail = True
    for L in (20, 22, 24, 26, 28):
        for comp in ("mlp_output", "attention_output", "block_output"):
            v = mean_at(L, comp, "real")
            if not np.isnan(v) and v > 10.0:
                fail = False
                break
        if not fail:
            break

    verdict = "PASS-strict" if pass_strict else ("PASS-soft" if pass_soft else ("FAIL" if fail else "INCONCLUSIVE"))

    summary = {
        "aggregates": agg,
        "by_class": by_class,
        "L24_class_dominant_component": L24_class_dominant,
        "same_component_across_classes": same_component_across_classes,
        "unified_component": unified_component,
        "off_target_layer_summary": off_target_layer_summary,
        "off_target_ok": off_target_ok,
        "reverse_patch_summary": reverse_patch_summary,
        "causal_sufficiency_verdict": causal_sufficiency_verdict,
        "causally_sufficient_at": causally_sufficient_at,
        "pre_registered_criterion": {
            "pass_strict": pass_strict,
            "pass_strict_base": pass_strict_base,
            "pass_strict_component": pass_strict_component,
            "pass_strict_off_target_ok": off_target_ok,
            "pass_strict_unified_component_matches": unified_matches,
            "pass_soft": pass_soft,
            "fail": fail,
            "verdict": verdict,
            "L24_mlp_output_real_mean_pct": l24_mlp_real,
            "L24_attention_output_real_mean_pct": l24_attn_real,
            "L24_block_output_real_mean_pct": l24_block_real,
        },
        "n_records": len(all_records),
        "n_reverse_records": len(reverse_records),
    }
    sp = out_dir / "_summary.json"
    sp.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[summary] wrote {sp}")
    print(f"[summary] verdict: {verdict}")
    print(f"[summary]   L24 mlp_output  real mean reduction = {l24_mlp_real:.1f}%")
    print(f"[summary]   L24 attn_output real mean reduction = {l24_attn_real:.1f}%")
    print(f"[summary]   L24 block_output real mean reduction = {l24_block_real:.1f}%")
    print(f"[summary]   off-target layers ok (no comp >=10%): {off_target_ok}")
    print(f"[summary]   same component across classes: {same_component_across_classes} ({unified_component})")
    print(f"[summary]   causal sufficiency: {causal_sufficiency_verdict}")
    return summary


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--positives-manifest",
        default="results/p1_hpc_n40/manifest_labeled.json",
        help="path to the N>=20 manifest_labeled.json (list of {doc, label, ...})",
    )
    ap.add_argument(
        "--layers",
        default="L0,L8,L20,L22,L24,L26,L28,L35",
        help="comma-separated layer indices (with or without leading 'L'). "
             "L0/L8/L35 are off-target controls — PASS-strict additionally "
             "requires no component reduces failure >=10%% at these layers.",
    )
    ap.add_argument(
        "--components",
        default="mlp_output,attention_output,block_output",
        help="comma-separated pyvene component names",
    )
    ap.add_argument("--max-new-tokens", type=int, default=12000)
    ap.add_argument(
        "--n-docs", type=int, default=None,
        help="cap on number of positive docs to run; default = all",
    )
    ap.add_argument(
        "--n-controls", type=int, default=5,
        help="number of matched controls to average into the SOURCE",
    )
    ap.add_argument(
        "--reverse-patch", type=_str2bool, default=True,
        help="B3: also patch positive's in-loop residual INTO clean-EOS controls. "
             "If `tokens_emitted_with_patch > 2 * baseline_control_tokens`, "
             "L24 is causally SUFFICIENT (not just necessary).",
    )
    ap.add_argument(
        "--n-reverse-controls", type=int, default=N_REVERSE_CONTROLS,
        help="number of clean-EOS controls to use as receivers of the reverse patch",
    )
    ap.add_argument("--out-dir", default="results/p3_he_patch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    started_at = time.time()
    print(f"[p3_he_patch] start {time.strftime('%Y-%m-%d %H:%M:%S')}")

    assert torch.cuda.is_available(), "CUDA required for H-E patching"
    device = "cuda"
    attn_impl = "eager"  # consistent with diagnostic forward pass

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / args.positives_manifest
    manifest = load_manifest(manifest_path)
    positives = [m["doc"] for m in manifest if m.get("label") != "control"]
    if args.n_docs is not None:
        positives = positives[: args.n_docs]
    print(f"[p3_he_patch] {len(positives)} positive docs to sweep")
    if not positives:
        raise SystemExit(
            f"[p3_he_patch] FATAL: 0 positive docs in manifest {manifest_path} "
            f"(empty/all-control cohort; heuristic #80 — exit-1 beats exit-0-with-zero-data)"
        )

    # B5: failure-class lookup table.
    fc_map = load_failure_class_map()
    print(f"[p3_he_patch] failure-class map: {len(fc_map)} entries; "
          f"distinct classes: {sorted(set(fc_map.values()))}")

    # Parse layers / components.
    def parse_layer_tok(tok: str) -> int:
        tok = tok.strip()
        if tok.startswith("L") or tok.startswith("l"):
            tok = tok[1:]
        return int(tok)
    layers = [parse_layer_tok(t) for t in args.layers.split(",") if t.strip()]
    components = [c.strip() for c in args.components.split(",") if c.strip()]
    target_layers = [L for L in layers if L not in OFF_TARGET_LAYERS]
    off_target_layers_present = [L for L in layers if L in OFF_TARGET_LAYERS]
    print(f"[p3_he_patch] layers={layers}  components={components}")
    print(f"[p3_he_patch] target layers={target_layers}; off-target controls={off_target_layers_present}")
    print(f"[p3_he_patch] reverse_patch={args.reverse_patch} "
          f"(n_reverse_controls={args.n_reverse_controls})")

    proc, model = load_model(device, attn_impl=attn_impl)
    print(f"[p3_he_patch] model loaded on {device}, dtype=bfloat16, attn_impl={attn_impl}")

    all_records: list[dict] = []
    reverse_records: list[dict] = []

    # -----------------------------------------------------------------
    # FORWARD sweep (positive doc as base; control mean as source).
    # -----------------------------------------------------------------
    for doc_idx, doc_id in enumerate(positives):
        try:
            image_path = find_doc_image(doc_id)
        except FileNotFoundError as e:
            print(f"[skip] {doc_id}: {e}")
            continue
        print(f"\n[p3_he_patch] === doc {doc_idx + 1}/{len(positives)}: {doc_id} ===")

        inputs = prepare_inputs(proc, image_path, device)

        # Per-doc matched controls (deterministic from manifest).
        control_doc_ids = load_matched_controls(manifest, doc_id, n=args.n_controls)
        if not control_doc_ids:
            print(f"  [warn] no cached matched controls for {doc_id}; skipping")
            continue
        print(f"  matched controls: {control_doc_ids}")

        failure_class = fc_map.get(doc_id, "unknown")
        print(f"  failure_class: {failure_class}")

        # Baseline (no patch). Cache to disk so reruns can reuse it.
        baseline_path = out_dir / doc_id / "_baseline.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
        else:
            print(f"  baseline ...")
            baseline = run_generation(model, proc, inputs, max_new_tokens=args.max_new_tokens)
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps(baseline, indent=2))
        baseline_tokens = baseline["tokens_emitted"]
        print(f"  baseline: tokens={baseline_tokens}  stop={baseline['stop_reason']}")

        if baseline["stop_reason"] != "max_new_tokens":
            print(f"  [warn] {doc_id} did not reproduce the runaway (stopped at {baseline['stop_reason']}); "
                  f"still running interventions, but reduction_pct will be near 0")

        # Precompute SOURCE per layer (real direction).
        real_sources: dict[int, torch.Tensor] = {}
        for layer in layers:
            src = compute_source_residual(control_doc_ids, layer)
            if src is None:
                print(f"  [warn] no cached residual at L{layer} for controls; skipping that layer")
                continue
            real_sources[layer] = src.to(device=device, dtype=torch.bfloat16)

        # Sweep.
        for layer in layers:
            if layer not in real_sources:
                continue
            for component in components:
                for k, intervention in enumerate(INTERVENTIONS):
                    rp = out_record_path(out_dir, doc_id, layer, component, intervention)
                    if rp.exists():
                        # Idempotent: reuse cached record.
                        try:
                            rec = json.loads(rp.read_text())
                            # Backfill new fields onto pre-existing records.
                            rec.setdefault("layer_role", layer_role_for(layer))
                            rec.setdefault("failure_class", failure_class)
                            all_records.append(rec)
                            continue
                        except Exception:
                            pass
                    seed = args.seed + 1000 * layer + 7 * k + hash(doc_id) % 997
                    src_vec = build_source_for_intervention(
                        real_source=real_sources[layer],
                        intervention=intervention,
                        doc_layer_residuals=real_sources,
                        layer=layer,
                        control_doc_ids=control_doc_ids,
                        seed=seed,
                    )
                    if src_vec is None:
                        print(f"  [warn] cannot build source for L{layer}/{component}/{intervention}; skipping")
                        continue
                    src_vec = src_vec.to(device=device, dtype=torch.bfloat16)
                    rec = run_one_combo(
                        model=model,
                        proc=proc,
                        inputs=inputs,
                        doc_id=doc_id,
                        layer=layer,
                        component=component,
                        intervention=intervention,
                        source_vec=src_vec,
                        baseline_tokens=baseline_tokens,
                        max_new_tokens=args.max_new_tokens,
                        out_dir=out_dir,
                        failure_class=failure_class,
                    )
                    print(f"  L{layer:02d}/{component:<18}/{intervention:<20}  "
                          f"tokens={rec['tokens_emitted']:>5d}  "
                          f"reduction={rec['reduction_pct']:>6.1f}%  "
                          f"stop={rec['stop_reason']}  "
                          f"elapsed={rec['elapsed_s']:.1f}s")
                    all_records.append(rec)

        # Empty CUDA cache between docs to keep peak memory tractable.
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # B3 — REVERSE-DIRECTION PATCH sweep.
    # For each (positive_doc, layer, component), patch positive's in-loop
    # residual INTO each clean-EOS control's forward; check if it induces
    # the control to run away.
    # -----------------------------------------------------------------
    if args.reverse_patch:
        reverse_controls = list_clean_eos_controls(manifest, n=args.n_reverse_controls)
        print(f"\n[p3_he_patch] === REVERSE PATCH (B3): {len(reverse_controls)} controls "
              f"× {len(positives)} positives ===")
        if not reverse_controls:
            print("  [warn] no clean-EOS controls with cached activations; skipping reverse patch")
        else:
            # Cache control baselines + inputs.
            control_baselines: dict[str, int] = {}
            control_inputs_cache: dict[str, dict] = {}
            for c_id in reverse_controls:
                try:
                    img = find_doc_image(c_id)
                except FileNotFoundError as e:
                    print(f"  [skip control] {c_id}: {e}")
                    continue
                control_inputs_cache[c_id] = prepare_inputs(proc, img, device)
                cb_path = out_dir / c_id / "_baseline.json"
                if cb_path.exists():
                    cb = json.loads(cb_path.read_text())
                else:
                    print(f"  control baseline {c_id} ...")
                    cb = run_generation(model, proc, control_inputs_cache[c_id],
                                        max_new_tokens=args.max_new_tokens)
                    cb_path.parent.mkdir(parents=True, exist_ok=True)
                    cb_path.write_text(json.dumps(cb, indent=2))
                control_baselines[c_id] = cb["tokens_emitted"]
                print(f"  control baseline: {c_id}  tokens={cb['tokens_emitted']}  stop={cb['stop_reason']}")

            for p_id in positives:
                # Pull positive's in-loop residual at each layer.
                pos_in_loop: dict[int, torch.Tensor] = {}
                for L in layers:
                    v = compute_positive_in_loop_residual(p_id, L)
                    if v is None:
                        continue
                    pos_in_loop[L] = v.to(device=device, dtype=torch.bfloat16)
                if not pos_in_loop:
                    print(f"  [reverse-skip] {p_id}: no cached in-loop residuals")
                    continue
                for c_id in reverse_controls:
                    if c_id not in control_inputs_cache:
                        continue
                    for layer in layers:
                        if layer not in pos_in_loop:
                            continue
                        for component in components:
                            rp = out_reverse_record_path(out_dir, c_id, layer, component, p_id)
                            if rp.exists():
                                try:
                                    reverse_records.append(json.loads(rp.read_text()))
                                    continue
                                except Exception:
                                    pass
                            rec = run_reverse_patch_combo(
                                model=model,
                                proc=proc,
                                control_inputs=control_inputs_cache[c_id],
                                control_doc=c_id,
                                positive_doc=p_id,
                                layer=layer,
                                component=component,
                                source_vec=pos_in_loop[layer],
                                baseline_control_tokens=control_baselines[c_id],
                                max_new_tokens=args.max_new_tokens,
                                out_dir=out_dir,
                            )
                            print(f"  REV  control={c_id:<26} pos={p_id:<26} "
                                  f"L{layer:02d}/{component:<18}  "
                                  f"tokens={rec['tokens_emitted_with_patch']:>5d}  "
                                  f"baseline={rec['baseline_control_tokens']:>5d}  "
                                  f"induced={rec['induced_failure']}  "
                                  f"stop={rec['stop_reason']}")
                            reverse_records.append(rec)
                torch.cuda.empty_cache()

    if not all_records:
        raise SystemExit(
            "[p3_he_patch] FATAL: 0 forward patch records produced — every doc was "
            "skipped (no cached image, or no cached matched-control residuals). "
            "Check find_doc_image / compute_source_residual coverage (heuristic #80)."
        )
    summary = write_summary(out_dir, all_records, reverse_records=reverse_records)

    # Provenance.
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "no-git"
    try:
        import pyvene  # noqa: F401
        pyvene_version = pyvene.__version__ if hasattr(pyvene, "__version__") else "installed"
    except Exception:
        pyvene_version = "not_installed_using_direct_hooks"
    provenance = {
        "seed": args.seed,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REV,
        "dtype": "bfloat16",
        "attn_impl": attn_impl,
        "device": device,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "pyvene_version": pyvene_version,
        "implementation": "direct_pytorch_forward_hooks (pyvene API not used; component names mirror pyvene's)",
        "git_commit": git_commit,
        "started_at": started_at,
        "finished_at": time.time(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "args": vars(args),
        "n_positives": len(positives),
        "layers": layers,
        "target_layers": target_layers,
        "off_target_layers": off_target_layers_present,
        "components": components,
        "interventions": list(INTERVENTIONS),
        "n_records": len(all_records),
        "n_reverse_records": len(reverse_records),
        "reverse_patch_enabled": bool(args.reverse_patch),
        "n_reverse_controls": args.n_reverse_controls,
        "reverse_induce_mult": REVERSE_INDUCE_MULT,
        "summary_verdict": summary["pre_registered_criterion"]["verdict"],
        "causal_sufficiency_verdict": summary.get("causal_sufficiency_verdict"),
        "same_component_across_classes": summary.get("same_component_across_classes"),
        "unified_component": summary.get("unified_component"),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    print(f"[p3_he_patch] wrote provenance.json  verdict={provenance['summary_verdict']}")
    print(f"[p3_he_patch] done {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
