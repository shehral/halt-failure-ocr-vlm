"""B3 reverse-direction patch — discriminate halt mechanism vs repetition readout.

Removes the trained L24 halt-direction component from the L24 residual stream
at EVERY generation position via a project-out forward hook. Then compares
generation behavior against baseline.

Predictions (from convergent mechinterp critic 2026-05-16):
  HALT MECHANISM:    Subtracting the halt direction removes the model's
                     "halt now" signal. Controls that normally halt cleanly
                     should run LONGER (or fail to halt within max_new_tokens),
                     because the load-bearing halt-direction is gone.
  REPETITION READOUT:Subtracting the halt direction removes a passive
                     readout of "repetition density => predict EOS". The
                     model's underlying halt decision was driven elsewhere
                     (e.g., earlier layers, content-end tokens), so controls
                     still halt at approximately the same token count.

The test discriminates which prediction holds by running clean-EOS-halting
controls with the project-out hook and comparing tokens_emitted to baseline.

Usage:
    python code/test_b3_reverse.py <doc_id> <mode:baseline|subtract> <max_new_tokens>

Output: results/p2_pilot/b3_<doc_id>_<mode>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# CODE_DIR is the directory holding this script (and the `fix/` package). ROOT is the
# repo root used for data/results lookups. In this released layout the `fix/` package
# lives alongside this script under `code/`, so we put CODE_DIR on sys.path for the
# `from fix.halt_monitor import ...` to resolve.
CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
from fix.halt_monitor import EOS_TOKEN_ID, DEFAULT_DIRECTION_PATH

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

CORPORA = ("public_corpus", "synthetic_stress", "real_loopy", "long_form_corpus", "supplementary_corpus")


def make_subtract_hook(halt_direction, mean, scale):
    """Forward hook that projects-out the halt-direction component from L24 residuals.

    Standardizes residuals with the LR-training scaler, projects them onto the
    UNIT-NORMED halt direction, subtracts that scalar projection times the
    unit direction, then un-standardizes back to the model's native space.

    Applied to ALL positions in the (batch, seq, hidden) tensor, every forward pass.
    """
    hd = halt_direction.to(torch.float32)
    hd_norm = hd / hd.norm().clamp_min(1e-9)
    mean = mean.to(torch.float32)
    scale = scale.to(torch.float32)

    def hook(_mod, _in, out):
        h = out[0] if isinstance(out, tuple) else out
        dt = h.dtype
        s = (h.to(torch.float32) - mean) / (scale + 1e-9)
        proj = s @ hd_norm  # (batch, seq)
        s_new = s - proj.unsqueeze(-1) * hd_norm.view(1, 1, -1)
        h_new = (s_new * scale + mean).to(dt)
        if isinstance(out, tuple):
            return (h_new,) + out[1:]
        return h_new

    return hook


def find_image_path(doc_id: str) -> Path | None:
    for set_dir in CORPORA:
        m = ROOT / "data" / set_dir / "manifest.json"
        if not m.exists():
            continue
        for e in json.loads(m.read_text()):
            if e.get("doc_id") == doc_id:
                return ROOT / e["path"]
    return None


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: test_b3_reverse.py <doc_id> <mode:baseline|subtract> <max_new_tokens>")
        return 2

    doc_id = sys.argv[1]
    mode = sys.argv[2]
    max_new_tokens = int(sys.argv[3])
    assert mode in ("baseline", "subtract"), f"mode must be baseline or subtract, got {mode}"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    proc = AutoProcessor.from_pretrained(
        "nanonets/Nanonets-OCR2-3B",
        revision="c3886ff00bb037ce7da24988c9eafaf1fe2bed72",
    )
    model = AutoModelForImageTextToText.from_pretrained(
        "nanonets/Nanonets-OCR2-3B",
        revision="c3886ff00bb037ce7da24988c9eafaf1fe2bed72",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    # MANDATORY lm_head re-tie (see CLAUDE.md).
    model.config.tie_word_embeddings = True
    model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr()
    model = model.to(device)
    model.train(False)
    print(f"[{mode}] model loaded on {device}, doc={doc_id}")

    hook_handle = None
    if mode == "subtract":
        payload = torch.load(DEFAULT_DIRECTION_PATH, weights_only=True)
        target_layer = int(payload["target_layer"])
        halt_direction = payload["halt_direction"].to(device)
        mean = payload["scaler_mean"].to(device)
        scale = payload["scaler_scale"].to(device)
        hook = make_subtract_hook(halt_direction, mean, scale)
        hook_handle = model.model.language_model.layers[target_layer].register_forward_hook(hook)
        print(f"[{mode}] subtract-hook attached at L{target_layer}")

    img_path = find_image_path(doc_id)
    if img_path is None or not img_path.exists():
        print(f"ERROR: image not found for doc_id={doc_id}")
        return 1
    img = Image.open(img_path).convert("RGB")
    print(f"[{mode}] image: {img_path}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": USER_PROMPT}]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

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
    if hook_handle is not None:
        hook_handle.remove()

    new_ids = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    last_id = new_ids[-1] if new_ids else -1
    hit_cap = (len(new_ids) >= max_new_tokens) and (last_id != EOS_TOKEN_ID)
    stop_reason = "max_new_tokens" if hit_cap else ("eos" if last_id == EOS_TOKEN_ID else "other")
    tail = proc.tokenizer.decode(new_ids[-80:], skip_special_tokens=False) if new_ids else ""

    print(f"[{mode}] tokens={len(new_ids)}  stop_reason={stop_reason}  elapsed={dt:.1f}s")
    print(f"[{mode}] tail: {tail[-200:]!r}")

    out_path = ROOT / "results" / "p2_pilot" / f"b3_{doc_id}_{mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "doc_id": doc_id,
        "mode": mode,
        "max_new_tokens": max_new_tokens,
        "tokens_emitted": len(new_ids),
        "stop_reason": stop_reason,
        "elapsed_s": dt,
        "last_token_id": last_id,
        "tail": tail[-200:],
        "device": device,
        "cuda_dev": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch_version": torch.__version__,
        "model_revision": "c3886ff00bb037ce7da24988c9eafaf1fe2bed72",
    }, indent=2))
    print(f"[{mode}] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
