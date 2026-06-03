"""Phase 1.3 — cache harness.

Given (image, prompt, generated_tokens, decision_moment_position), run a single
teacher-forced forward pass over [prompt + generated] and cache per-layer/per-position:
  - residual stream (post-block; len = num_layers + 1 incl. embeddings)
  - attention weights (len = num_layers)
  - residual-stream L2 norm

To control memory: keep a coarse band of layers (every Nth) and stream tensors to disk.
The attentions are the OOM hog: (B, H, T, T) at T=2-4k scales quadratically.

Run (smoke-test):
    .venv/bin/python code/p1_cache_harness.py --doc table_N005_C04_s05000

Outputs:
    results/activations/<doc_id>/
        hidden_states.pt        # (num_kept_layers, T, hidden) bf16
        attn_weights.pt         # (num_kept_layers, H, T, T) bf16 — selective
        residual_norms.pt       # (num_kept_layers, T) float32
        layer_indices.pt        # (num_kept_layers,) int — which layers were saved
        meta.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "nanonets/Nanonets-OCR2-3B"
MODEL_REV = "c3886ff00bb037ce7da24988c9eafaf1fe2bed72"
# Repo root resolution. The original project ran on two hosts (laptop + HPC cluster);
# absolute host paths have been replaced for the public release. We resolve relative
# to this file, which works regardless of where the repo is checked out. Set
# HALT_REPO_ROOT to override.
import os
ROOT = Path(os.environ.get("HALT_REPO_ROOT", Path(__file__).resolve().parents[1]))
TRIGGER_DIR = ROOT / "results" / "p1_trigger"
TRIGGER_DIR_V2 = ROOT / "results" / "p1_trigger_v2"
# DATA_DIRS from canonical _constants module (2026-05-17 refactor; resolves 8-recurrence path-resolution bug family)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _constants import DATA_DIRS as _CANONICAL_DATA_DIRS
DATA_DIRS = list(_CANONICAL_DATA_DIRS)
OUT = ROOT / "results" / "activations"
OUT.mkdir(parents=True, exist_ok=True)


def find_trigger_record(doc_id: str) -> Path:
    """Look in p1_trigger_v2 first (current), fall back to p1_trigger (legacy)."""
    for d in (TRIGGER_DIR_V2, TRIGGER_DIR):
        p = d / f"{doc_id}.json"
        if p.exists():
            return p
    raise FileNotFoundError(f"No trigger record for {doc_id} in {TRIGGER_DIR_V2} or {TRIGGER_DIR}")


def find_doc_meta(doc_id: str) -> tuple[dict, Path]:
    """Search the cross-set manifests for the doc's image + metadata."""
    import json as _json
    for d in DATA_DIRS:
        m = d / "manifest.json"
        if not m.exists():
            continue
        for entry in _json.loads(m.read_text()):
            if entry.get("doc_id") == doc_id:
                return entry, ROOT / entry["path"]
    raise FileNotFoundError(f"No manifest entry for {doc_id} in any of {DATA_DIRS}")

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


def load_model_for_caching(device: str, attn_impl: str = "eager"):
    """attn_impl='eager' exposes attention weights but materializes (T, T) per layer.
    Use 'sdpa' when caching hidden states only (no attention) — order-of-magnitude
    less peak memory at long T."""
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REV)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REV,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    model.config.tie_word_embeddings = True
    model.tie_weights()
    assert (
        model.lm_head.weight.data_ptr()
        == model.model.language_model.embed_tokens.weight.data_ptr()
    ), "lm_head failed to tie"
    model = model.to(device)
    model.train(False)
    return processor, model


def make_inputs(processor, image_path: Path, generated_ids: list[int], device: str):
    """Re-tokenize the FULL [prompt + generated_text] through the chat template
    so Qwen2.5-VL's image-token expansion in get_rope_index stays consistent.

    Manually concatenating input_ids after the processor returns mismatches
    attention_mask vs input_token_type inside get_rope_index. The processor
    must see the whole thing as one sequence.

    Returns inputs dict and the prompt/gen split (recovered by tokenizing the
    bare prompt template alone).
    """
    image = Image.open(image_path).convert("RGB")
    base_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]
    prompt_text = processor.apply_chat_template(base_messages, tokenize=False, add_generation_prompt=True)
    prompt_inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
    prompt_len = int(prompt_inputs["input_ids"].shape[1])

    # Decode the generated tokens and re-tokenize alongside the prompt as one block.
    gen_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
    full_text = prompt_text + gen_text
    full_inputs = processor(text=[full_text], images=[image], padding=True, return_tensors="pt")
    full_len = int(full_inputs["input_ids"].shape[1])

    # Sanity: full_len should equal prompt_len + len(generated_ids), modulo tokenizer
    # quirks around the boundary. If they drift by >2 tokens, the chat template
    # ate or added something we didn't account for.
    drift = full_len - (prompt_len + len(generated_ids))
    inputs = {k: v.to(device) for k, v in full_inputs.items()}
    return inputs, prompt_len, len(generated_ids), drift


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True, help="doc_id from data/synthetic/manifest.json")
    ap.add_argument("--every-k-layers", type=int, default=4,
                    help="coarse band: keep every Kth layer (default 4 → 9 layers of 36)")
    ap.add_argument("--save-attn", action="store_true",
                    help="also save attention weights (memory expensive)")
    ap.add_argument("--attn-only-decision-window", type=int, default=64,
                    help="if save-attn: only save attn rows/cols around decision moment, ±W")
    ap.add_argument("--gen-cutoff", type=int, default=None,
                    help="truncate generated tokens to first N (for long positives that won't fit at full T)")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    attn_impl = "eager" if args.save_attn else "sdpa"
    print(f"[cache] device={device} every_k_layers={args.every_k_layers} save_attn={args.save_attn} attn_impl={attn_impl}")

    trig_path = find_trigger_record(args.doc)
    trig = json.loads(trig_path.read_text())
    # v1 trigger record had new_token_ids inline; v2 saves them only to a .pt file.
    if "new_token_ids" in trig:
        generated_ids = trig["new_token_ids"]
    else:
        toks_path = trig_path.with_suffix(".tokens.pt")
        generated_ids = torch.load(toks_path, weights_only=True).tolist()
    if args.gen_cutoff is not None and len(generated_ids) > args.gen_cutoff:
        print(f"[cache] truncating gen from {len(generated_ids)} to {args.gen_cutoff} (--gen-cutoff)")
        generated_ids = generated_ids[: args.gen_cutoff]
    doc_meta, image_path = find_doc_meta(args.doc)
    print(f"[cache] doc={args.doc} gt_rows={doc_meta.get('n_rows', 0)} gen_len={len(generated_ids)}")
    trig_label = trig.get("trigger") or trig.get("label", "?")
    trig_symptoms = trig.get("symptoms") or trig.get("production_patterns_fired", [])
    print(f"[cache] label={trig_label} symptoms={trig_symptoms} decision_moment={trig.get('decision_moment_position')}")

    processor, model = load_model_for_caching(device, attn_impl=attn_impl)
    inputs, prompt_len, gen_len, drift = make_inputs(processor, image_path, generated_ids, device)
    T = int(inputs["input_ids"].shape[1])
    print(f"[cache] full seq len T={T} (prompt={prompt_len} + gen={gen_len}, drift={drift})")

    num_layers = model.config.text_config.num_hidden_layers
    hidden = model.config.text_config.hidden_size
    heads = model.config.text_config.num_attention_heads
    print(f"[cache] L={num_layers} H={heads} hidden={hidden}")

    keep_layers = list(range(0, num_layers, args.every_k_layers))
    if (num_layers - 1) not in keep_layers:
        keep_layers.append(num_layers - 1)
    print(f"[cache] keeping layers {keep_layers} ({len(keep_layers)}/{num_layers})")

    # Forward hooks: only stash the kept-layer outputs, immediately offloaded to
    # CPU + bf16. Avoid output_hidden_states=True which keeps ALL L+1 layer outputs
    # resident on the device — OOMs at T~14K on 24 GB unified memory.
    decoder_layers = model.model.language_model.layers  # length 36
    cap_hs: dict[int, torch.Tensor] = {}
    cap_attn: dict[int, torch.Tensor] = {}
    kept_set = set(keep_layers)

    def make_hook(layer_idx: int):
        def _hook(_mod, _in, out):
            # decoder block output is a tuple (hidden_states, optional attn, ...)
            hs_tensor = out[0] if isinstance(out, tuple) else out
            cap_hs[layer_idx] = hs_tensor.detach()[0].to(torch.bfloat16).cpu()
        return _hook

    handles = [decoder_layers[i].register_forward_hook(make_hook(i)) for i in keep_layers]

    # Attention extraction (eager mode only) — register hooks on self_attn modules
    # and capture attn_weights from their output tuple.
    if args.save_attn:
        for i in keep_layers:
            attn_mod = decoder_layers[i].self_attn
            def make_attn_hook(layer_idx: int):
                def _hook(_m, _i, out):
                    # qwen2.5-vl self_attn returns (attn_output, attn_weights)
                    if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                        cap_attn[layer_idx] = out[1].detach()[0].to(torch.bfloat16).cpu()
                return _hook
            handles.append(attn_mod.register_forward_hook(make_attn_hook(i)))

    t0 = time.time()
    with torch.inference_mode():
        _ = model(
            **inputs,
            return_dict=True,
            use_cache=False,
        )
    dt = time.time() - t0
    for h in handles:
        h.remove()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print(f"[cache] forward done in {dt:.1f}s | captured {len(cap_hs)} layer-hidden-states"
          + (f" + {len(cap_attn)} attns" if args.save_attn else ""))

    # Stack the kept layers (in keep_layers order) into one bundled tensor.
    kept_hs = torch.stack([cap_hs[i] for i in keep_layers], dim=0)
    print(f"[cache] kept hidden states tensor: {tuple(kept_hs.shape)}  dtype={kept_hs.dtype}")

    norms = torch.stack([cap_hs[i].float().norm(dim=-1) for i in keep_layers], dim=0)
    print(f"[cache] residual L2 norms: {tuple(norms.shape)} (mean={norms.mean():.2f})")

    if args.save_attn and cap_attn:
        # Per-layer attention is (heads, T, T) on CPU. Window it now to a band
        # around the decision moment to keep on-disk size reasonable.
        dm = trig.get("decision_moment_position")
        anchor_in_gen = int(dm) if dm is not None else max(0, gen_len - 1)
        anchor_in_seq = prompt_len + anchor_in_gen
        W = args.attn_only_decision_window
        lo = max(0, anchor_in_seq - W)
        hi = min(T, anchor_in_seq + W + 1)
        print(f"[cache] attn window: anchor_in_seq={anchor_in_seq} W={W} → rows [{lo}:{hi}], cols [0:{T}]")
        kept_attn = torch.stack([cap_attn[i][:, lo:hi, :] for i in keep_layers], dim=0)
        print(f"[cache] kept attn tensor: {tuple(kept_attn.shape)}  ({kept_attn.numel()*2/1e6:.1f} MB)")
    else:
        kept_attn = None

    doc_out = OUT / args.doc
    doc_out.mkdir(parents=True, exist_ok=True)
    torch.save(kept_hs, doc_out / "hidden_states.pt")
    torch.save(norms, doc_out / "residual_norms.pt")
    torch.save(torch.tensor(keep_layers, dtype=torch.long), doc_out / "layer_indices.pt")
    if kept_attn is not None:
        torch.save(kept_attn, doc_out / "attn_weights.pt")

    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = "no-git"

    meta = {
        "doc_id": args.doc,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REV,
        "device": device,
        "dtype": "bfloat16",
        "attn_impl": "eager",
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "T": T,
        "drift_full_vs_prompt_plus_gen": drift,
        "num_layers_total": num_layers,
        "kept_layers": keep_layers,
        "hidden_size": hidden,
        "heads": heads,
        "saved_files": {
            "hidden_states": "hidden_states.pt",
            "residual_norms": "residual_norms.pt",
            "layer_indices": "layer_indices.pt",
            **({"attn_weights": "attn_weights.pt"} if kept_attn is not None else {}),
        },
        "elapsed_forward_s": dt,
        "git_commit": git,
        "trigger_record": str(trig_path.relative_to(ROOT)),
        "started_at": t0,
        "finished_at": time.time(),
    }
    (doc_out / "meta.json").write_text(json.dumps(meta, indent=2))
    (doc_out / "provenance.json").write_text(json.dumps(meta, indent=2))
    print(f"[cache] wrote → {doc_out}")
    print(f"[cache] sizes: hidden={(doc_out/'hidden_states.pt').stat().st_size/1e6:.1f} MB"
          + (f", attn={(doc_out/'attn_weights.pt').stat().st_size/1e6:.1f} MB" if kept_attn is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
