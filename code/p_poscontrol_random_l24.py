"""Positive-control patch — random unit direction at L24 with escalating norm.

Purpose. The 6 converging causal-perturbation nulls (B3, P16, P10b, P19v2,
P12 v3, P22) define mechanism-as-distributed in `CL-35`, BUT every one of
them is equally consistent with "patching protocol under-powered". The
discriminating test, per `docs/positive_control_he_patch_plan.md` §1
Option C, is a positive-control patch on a *known harmful direction* —
a random unit vector scaled to high norm. If the H-E framework delivers
perturbations at all, a random direction at L24 with norm 10x or 100x
the natural residual magnitude must cause SOMETHING — gibberish output,
early EOS, or a substantial token-count change.

Conditions per doc:
  - vanilla            : no patch (baseline)
  - C1_random_10x      : forward-patch L24 residual with random unit vec
                         scaled to 10 x ||residual_norm|| (per-doc per-pos
                         dynamic norm)
  - C2_random_100x     : same, x 100 (escalating perturbation strength)
  - C3_random_0.1x     : same, x 0.1 (negative control on positive control;
                         should be no-op if framework is calibrated)

For each (doc, condition) we record tokens_emitted, stop_reason, elapsed_s,
and first 100 chars of decoded output (for gibberish detection).

Pre-registered interpretation (see plan doc §2):
  - C2 reduces length >50% on >=3/4 docs => protocol DELIVERS; nulls mean
    "mechanism distributed, not at the tested directions" => CL-35 sharpens.
  - C2 produces gibberish but no length change >=3/4 docs => protocol
    delivers but halt state is robust to single-layer perturbations.
  - C2 produces 0 effect on all 4 docs => protocol CANNOT deliver =>
    CL-35 caveats with methodology disclosure.
  - C3 produces non-zero effect => framework over-sensitive; H-E results
    suspect including B3.

NOTES on departures from the plan doc:
  - Plan §1 originally said "scale to 10 * ||resid_norm||". This script
    interprets that as "10 x the L2 norm of the prefill's LAST-position
    residual at L24 BEFORE the patch", computed dynamically per-doc.
    This makes the perturbation magnitude doc-dependent rather than a
    universal constant; the alternative (a single norm number across
    docs) would create per-doc variance in *effective* perturbation
    strength because residual norms differ across docs.
  - We patch ONLY the LAST sequence position of the prefill (same as
    p3_he_patch.py's hook), not "every gen position" as the prompt
    text mentioned. The plan doc itself doesn't actually specify
    "every gen position" — and per-token patching during decode is
    a substantially different intervention shape that would require
    a different hook architecture (hooking the residual STREAM, not
    a layer output). Keeping the existing single-prefill-last-position
    convention preserves protocol parity with the 6 nulls we're trying
    to validate. This is heuristic #67 cleanest-route: change one
    variable (random direction at high norm) at a time, not two
    (random direction AND streaming patch).

Manifest of 4 docs:
  - 3 cap-hit positives from public_corpus
      docvqa_gjhp0000_p1
      docvqa_jqbg0227_p1
      docvqa_srgb0228_p2
  - 1 clean-EOS control from data/synthetic
      table_N005_C04_s05000 (shortest clean-EOS control = 477 tokens
      natural halt per results/p1_trigger/manifest_labeled.json)

Output schema:
  results/p_poscontrol/<doc_id>__<condition>.json
    {
      "doc_id": ...,
      "condition": ...,             # vanilla / C1_random_10x / ...
      "layer": 24,
      "tokens_emitted": int,
      "stop_reason": "max_new_tokens" | "eos" | "other",
      "elapsed_s": float,
      "first_100_chars": str,
      "intervention_norm": float | None,
      "residual_norm_at_patch": float | None,
      "scale": float,               # 10.0, 100.0, 0.1, or 0.0 (vanilla)
      "seed": int,
      "fired_count": int,
    }
  results/p_poscontrol/_summary.json   # aggregate across 4 docs x 4 conditions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Local imports — _provenance + _constants live in code/
sys.path.insert(0, str(Path(__file__).parent))
from _constants import (  # noqa: E402
    DATA_DIRS,
    NANONETS_MODEL_ID,
    NANONETS_REVISION,
    REPO_ROOT,
    find_image_path,
)
from _provenance import write_provenance  # noqa: E402


def resolve_image_path(doc_id: str) -> Optional[Path]:
    """Locate the image for a doc_id. Falls back through three lookup strategies:

      1. _constants.find_image_path (scans <corpus>/images/<doc_id>.<ext>)
      2. Each corpus manifest.json's `path` field (handles synthetic/ which
         stores images at data/synthetic/<doc_id>.png without an images/ subdir)
      3. Direct glob under DATA_DIRS for <doc_id>.<ext>

    Schema-inconsistency note. public_corpus/long_form_corpus use the
    data/<corpus>/images/ convention. synthetic/synthetic_stress/real_loopy
    store images directly under data/<corpus>/. Step 2 reads each corpus's
    manifest.json to get the canonical path.
    """
    # Strategy 1: standard images/ subdir.
    p = find_image_path(doc_id)
    if p is not None:
        return p
    # Strategy 2: manifest direct-path lookup.
    for data_dir in DATA_DIRS:
        manifest = data_dir / "manifest.json"
        if not manifest.exists():
            continue
        try:
            entries = json.loads(manifest.read_text())
        except Exception:
            continue
        for entry in entries:
            if entry.get("doc_id") == doc_id:
                hint = entry.get("path")
                if hint:
                    for candidate in (
                        Path(hint),
                        REPO_ROOT / hint,
                    ):
                        if candidate.exists():
                            return candidate
    # Strategy 3: direct-glob fallback.
    for data_dir in DATA_DIRS:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            cand = data_dir / f"{doc_id}{ext}"
            if cand.exists():
                return cand
    return None

MODEL_ID = NANONETS_MODEL_ID
MODEL_REV = NANONETS_REVISION
EOS_TOKEN_ID = 151645  # <|im_end|>

# Patching target — fixed per plan doc.
TARGET_LAYER = 24

# Conditions (label, scale). Scale is the multiplier on the per-doc
# residual norm at the patch position. Vanilla has scale=0.0 (no patch).
CONDITIONS = (
    ("vanilla", 0.0),
    ("C1_random_10x", 10.0),
    ("C2_random_100x", 100.0),
    ("C3_random_0.1x", 0.1),
)

# Random-direction seed. Fixed across conditions for reproducibility.
RANDOM_SEED = 42

# Default manifest of (doc_id, label).
DEFAULT_MANIFEST = [
    ("docvqa_gjhp0000_p1", "positive"),
    ("docvqa_jqbg0227_p1", "positive"),
    ("docvqa_srgb0228_p2", "positive"),
    ("table_N005_C04_s05000", "control"),
]

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
# Model load (with the MANDATORY lm_head workaround)
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
    # MANDATORY lm_head workaround per project CLAUDE.md. Without this,
    # generation collapses to "!" forever because lm_head stays at
    # meta-init zeros (transformers 5.x reads tie_word_embeddings off
    # the OUTER config, which is False, instead of text_config).
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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


# =================================================================
# Intervention hook — random-direction at L24 last prefill position
# =================================================================

class _RandomPatchState:
    """Single-shot intervention state.

    Generates a fresh random unit vector (seeded by `seed`), scales it to
    `scale * ||residual_at_last_position||`, and overwrites that position.
    Fires ONCE on the prefill forward; subsequent decode forwards (T=1)
    are not patched (KV-cache + use_cache=True means decode never sees
    the block_output again).

    Doc-dynamic norm. residual_norm is captured from the live forward
    BEFORE we overwrite. This makes the perturbation magnitude scale
    with the doc's natural residual scale rather than being a fixed
    absolute number.
    """

    def __init__(self, scale: float, seed: int, hidden_size: int):
        self.scale = float(scale)
        self.seed = int(seed)
        self.hidden_size = int(hidden_size)
        self.fired = False
        self.fire_count = 0
        self.intervention_norm: Optional[float] = None
        self.residual_norm: Optional[float] = None
        # Pre-generate the random direction. Same draw across conditions
        # for the same doc (per heuristic #51 reproducibility).
        rng = np.random.default_rng(self.seed)
        vec = rng.standard_normal(self.hidden_size).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-9)  # unit vector
        self._unit = torch.tensor(vec, dtype=torch.float32)  # (hidden,)

    def reset(self):
        self.fired = False
        self.fire_count = 0
        self.intervention_norm = None
        self.residual_norm = None


def make_random_patch_hook(state: _RandomPatchState):
    """Forward hook on the layer module (block_output).

    The Qwen2.5-VL decoder layer module returns a tuple (hidden_states, ...).
    We patch the LAST sequence position of `hidden_states` ONLY on the
    prefill forward (T>1). Per plan §1 Option C — random direction at
    high norm at L24.
    """

    def hook(_mod, _inp, out):
        if state.fired:
            return out

        if isinstance(out, tuple):
            tensor = out[0]
            rest = out[1:]
        else:
            tensor = out
            rest = None

        # Only patch the prefill (T>1); decode steps are T=1.
        if tensor.dim() != 3 or tensor.shape[1] <= 1:
            return out

        # Vanilla shortcut: scale==0.0 means no-op (just observe + record norm).
        last_pos = tensor[0, -1, :]  # (hidden,)
        res_norm = float(last_pos.float().norm().item())
        state.residual_norm = res_norm

        if state.scale == 0.0:
            # Vanilla — record norm but don't patch.
            state.fired = True
            state.fire_count += 1
            state.intervention_norm = 0.0
            return out

        # Construct the patched vector: unit direction * scale * res_norm.
        device = tensor.device
        dtype = tensor.dtype
        unit_d = state._unit.to(device=device, dtype=dtype)
        patch_vec = unit_d * (state.scale * res_norm)
        state.intervention_norm = float(patch_vec.float().norm().item())

        new_tensor = tensor.clone()
        new_tensor[0, -1, :] = patch_vec
        state.fired = True
        state.fire_count += 1

        if rest is None:
            return new_tensor
        return (new_tensor, *rest)

    return hook


def attach_patch(model, layer: int, state: _RandomPatchState):
    """Register the random-patch hook on `model.model.language_model.layers[layer]`
    (the layer module itself => block_output).
    """
    target_layer = model.model.language_model.layers[layer]
    return target_layer.register_forward_hook(make_random_patch_hook(state))


# =================================================================
# Generation runner
# =================================================================

def run_generation(
    model,
    proc,
    inputs,
    max_new_tokens: int,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
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
    if hit_cap:
        stop_reason = "max_new_tokens"
    elif last_id == EOS_TOKEN_ID:
        stop_reason = "eos"
    else:
        stop_reason = "other"
    # Decode first 100 chars for gibberish-detection.
    decoded = proc.tokenizer.decode(new_ids[:128], skip_special_tokens=False)
    first_100 = decoded[:100]
    return {
        "tokens_emitted": len(new_ids),
        "stop_reason": stop_reason,
        "elapsed_s": dt,
        "last_token_id": last_id,
        "first_100_chars": first_100,
    }


# =================================================================
# Per-doc per-condition runner
# =================================================================

def run_one_combo(
    *,
    model,
    proc,
    inputs,
    doc_id: str,
    label: str,
    condition: str,
    scale: float,
    seed: int,
    layer: int,
    max_new_tokens: int,
    hidden_size: int,
    out_dir: Path,
) -> dict:
    state = _RandomPatchState(scale=scale, seed=seed, hidden_size=hidden_size)
    handle = attach_patch(model, layer, state)
    try:
        gen = run_generation(model, proc, inputs, max_new_tokens=max_new_tokens, seed=0)
    finally:
        handle.remove()

    record = {
        "doc_id": doc_id,
        "label": label,
        "condition": condition,
        "layer": layer,
        "scale": scale,
        "seed": seed,
        "tokens_emitted": gen["tokens_emitted"],
        "stop_reason": gen["stop_reason"],
        "elapsed_s": gen["elapsed_s"],
        "first_100_chars": gen["first_100_chars"],
        "last_token_id": gen["last_token_id"],
        "intervention_norm": state.intervention_norm,
        "residual_norm_at_patch": state.residual_norm,
        "fired_count": state.fire_count,
    }
    rp = out_dir / f"{doc_id}__{condition}.json"
    rp.write_text(json.dumps(record, indent=2))
    return record


# =================================================================
# Manifest loader
# =================================================================

def load_manifest(manifest_path: Optional[Path]) -> list[tuple[str, str]]:
    """Load a (doc_id, label) list. If `manifest_path` is None or absent,
    return DEFAULT_MANIFEST. If it's a JSON file, expect a list of
    {doc_id, label} dicts (label optional, defaults to 'positive').
    """
    if manifest_path is None or not manifest_path.exists():
        return list(DEFAULT_MANIFEST)
    data = json.loads(manifest_path.read_text())
    out = []
    for entry in data:
        if isinstance(entry, dict):
            doc_id = entry.get("doc_id") or entry.get("doc")
            label = entry.get("label", "positive")
            if doc_id:
                out.append((doc_id, label))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
            out.append((str(entry[0]), str(entry[1]) if len(entry) > 1 else "positive"))
    return out or list(DEFAULT_MANIFEST)


# =================================================================
# Main
# =================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default=None,
        help="path to JSON manifest of [{doc_id, label}, ...]; default is "
             "the 4-doc plan-doc manifest (3 positives + 1 clean-EOS control)",
    )
    ap.add_argument(
        "--out",
        default="results/p_poscontrol",
        help="output directory (relative to repo root unless absolute)",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=12000,
        help="max_new_tokens for generation; 12000 captures full runaway",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=TARGET_LAYER,
        help=f"layer to patch (default {TARGET_LAYER})",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="random direction seed (fixed across conditions for repro)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="smoke-test: process 1 doc only (the first in manifest)",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="device override (default: cuda if available else cpu)",
    )
    args = ap.parse_args()

    started_at_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    started_at_t = time.time()
    print(f"[p_poscontrol] start {started_at_iso}")

    # Resolve output dir.
    out_arg = Path(args.out)
    out_dir = out_arg if out_arg.is_absolute() else (REPO_ROOT / out_arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[p_poscontrol] out_dir = {out_dir}")

    # Device.
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    attn_impl = "eager"  # required for interventions (heuristic from CLAUDE.md)
    print(f"[p_poscontrol] device={device}  attn_impl={attn_impl}")

    # Manifest.
    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path and not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    docs = load_manifest(manifest_path)
    if args.smoke:
        docs = docs[:1]
        print(f"[p_poscontrol] SMOKE MODE — {len(docs)} doc only")
    print(f"[p_poscontrol] {len(docs)} docs x {len(CONDITIONS)} conditions = "
          f"{len(docs) * len(CONDITIONS)} runs")

    # Write provenance.json at results-creation time (heuristic #74).
    write_provenance(
        out_dir,
        seed=args.seed,
        attn_impl=attn_impl,
        started_at=started_at_iso,
        extra={
            "experiment": "positive_control_random_l24",
            "layer": args.layer,
            "conditions": [c[0] for c in CONDITIONS],
            "scales": {c[0]: c[1] for c in CONDITIONS},
            "manifest_docs": [{"doc_id": d, "label": l} for d, l in docs],
            "max_new_tokens": args.max_new_tokens,
            "device": device,
            "smoke_mode": bool(args.smoke),
            "rationale_doc": "docs/positive_control_he_patch_plan.md",
            "target_claim": "CL-35",
        },
    )

    # Load model.
    proc, model = load_model(device=device, attn_impl=attn_impl)
    hidden_size = model.config.text_config.hidden_size if hasattr(
        model.config, "text_config"
    ) else model.config.hidden_size
    print(f"[p_poscontrol] model loaded  hidden_size={hidden_size}")

    all_records: list[dict] = []
    for d_idx, (doc_id, label) in enumerate(docs):
        # Resolve image path (handles both images/ subdir + manifest direct-path).
        image_path = resolve_image_path(doc_id)
        if image_path is None:
            print(f"  [skip] {doc_id}: image not found in DATA_DIRS")
            continue
        print(f"\n[p_poscontrol] === doc {d_idx + 1}/{len(docs)}: "
              f"{doc_id} ({label}) image={image_path.name} ===")

        inputs = prepare_inputs(proc, image_path, device)

        for condition, scale in CONDITIONS:
            rp = out_dir / f"{doc_id}__{condition}.json"
            if rp.exists():
                # Idempotent: reuse cached record.
                try:
                    rec = json.loads(rp.read_text())
                    print(f"  [cached] {condition:<18}  tokens={rec.get('tokens_emitted')}  "
                          f"stop={rec.get('stop_reason')}")
                    all_records.append(rec)
                    continue
                except Exception:
                    pass

            print(f"  running {condition:<18} (scale={scale}) ...")
            rec = run_one_combo(
                model=model,
                proc=proc,
                inputs=inputs,
                doc_id=doc_id,
                label=label,
                condition=condition,
                scale=scale,
                seed=args.seed,
                layer=args.layer,
                max_new_tokens=args.max_new_tokens,
                hidden_size=hidden_size,
                out_dir=out_dir,
            )
            iv_n = rec.get("intervention_norm")
            res_n = rec.get("residual_norm_at_patch")
            iv_str = f"{iv_n:.1f}" if iv_n is not None else "None"
            res_str = f"{res_n:.1f}" if res_n is not None else "None"
            print(f"    -> tokens={rec['tokens_emitted']:>5d}  "
                  f"stop={rec['stop_reason']:<14}  "
                  f"elapsed={rec['elapsed_s']:>6.1f}s  "
                  f"iv_norm={iv_str}  res_norm={res_str}")
            print(f"    first_100: {rec['first_100_chars']!r}")
            all_records.append(rec)

        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary aggregation.
    summary = {
        "n_docs_run": len({r["doc_id"] for r in all_records}),
        "n_records": len(all_records),
        "layer": args.layer,
        "conditions": [c[0] for c in CONDITIONS],
        "started_at": started_at_iso,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_total_s": time.time() - started_at_t,
        "by_doc_condition": {},
    }
    for r in all_records:
        key = f"{r['doc_id']}::{r['condition']}"
        summary["by_doc_condition"][key] = {
            "tokens_emitted": r["tokens_emitted"],
            "stop_reason": r["stop_reason"],
            "intervention_norm": r.get("intervention_norm"),
            "residual_norm_at_patch": r.get("residual_norm_at_patch"),
            "first_100_chars": r.get("first_100_chars"),
        }

    # Pre-registered interpretation flags. Per plan §2 pre-reg table.
    # For each doc, compute the C2_random_100x reduction vs vanilla.
    docs_seen = sorted({r["doc_id"] for r in all_records})
    per_doc_summary: dict[str, dict] = {}
    for d in docs_seen:
        van = next(
            (r for r in all_records if r["doc_id"] == d and r["condition"] == "vanilla"),
            None,
        )
        c1 = next(
            (r for r in all_records if r["doc_id"] == d and r["condition"] == "C1_random_10x"),
            None,
        )
        c2 = next(
            (r for r in all_records if r["doc_id"] == d and r["condition"] == "C2_random_100x"),
            None,
        )
        c3 = next(
            (r for r in all_records if r["doc_id"] == d and r["condition"] == "C3_random_0.1x"),
            None,
        )
        per_doc_summary[d] = {
            "vanilla_tokens": van["tokens_emitted"] if van else None,
            "C1_10x_tokens": c1["tokens_emitted"] if c1 else None,
            "C2_100x_tokens": c2["tokens_emitted"] if c2 else None,
            "C3_0.1x_tokens": c3["tokens_emitted"] if c3 else None,
            "C2_reduction_pct": (
                (1.0 - c2["tokens_emitted"] / van["tokens_emitted"]) * 100.0
                if van and c2 and van["tokens_emitted"] > 0
                else None
            ),
        }
    summary["per_doc_summary"] = per_doc_summary

    # Headline metric: does C2 reduce length >=50% on >=3/4 docs?
    docs_with_c2 = [
        d for d, s in per_doc_summary.items()
        if s.get("C2_reduction_pct") is not None
    ]
    n_reduced = sum(
        1 for d in docs_with_c2
        if per_doc_summary[d]["C2_reduction_pct"] >= 50.0
    )
    summary["pre_registered"] = {
        "C2_n_docs_with_data": len(docs_with_c2),
        "C2_n_reduced_50pct": n_reduced,
        "C2_delivers_perturbation": n_reduced >= max(1, int(0.75 * len(docs_with_c2))),
        "note": "PASS if >= 3/4 docs show >= 50% reduction at C2_random_100x. "
                "See docs/positive_control_he_patch_plan.md §2 for full table.",
    }

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[p_poscontrol] wrote {summary_path}")
    print(f"[p_poscontrol] C2 reduced >=50% on {n_reduced}/{len(docs_with_c2)} docs")
    print(f"[p_poscontrol] done {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
