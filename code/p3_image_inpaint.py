"""Image-inpaint causal protocol — discovery experiment.

Question: Is the infinite-gen loop driven by VISION attention, or is it text-only?
If the loop persists when the image is replaced with grey noise, the loop is
text-self-reinforcing (text-only). If the loop breaks, vision attention is
load-bearing.

Method: for each CUDA-positive doc, run model.generate TWICE:
  (1) with the original image (baseline — known to cap-hit)
  (2) with the image replaced by grey noise of identical dimensions
Compare token counts + tail signatures. Loop = persists if both cap-hit AND
tail tokens repeat similarly.

Output (results/p3_image_inpaint/<doc>.json):
- per-doc: baseline_tokens, baseline_tail, noise_tokens, noise_tail, persists_under_noise

Usage:
    python code/p3_image_inpaint.py <doc_id> <max_new_tokens>
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
EOS_TOKEN_ID = 151645

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


def find_image_path(doc_id: str) -> Path | None:
    for set_dir in CORPORA:
        m = ROOT / "data" / set_dir / "manifest.json"
        if not m.exists():
            continue
        for e in json.loads(m.read_text()):
            if e.get("doc_id") == doc_id:
                return ROOT / e["path"]
    return None


def make_grey_noise(width: int, height: int, seed: int = 0) -> Image.Image:
    """Grey image at the same dimensions as the doc — preserves vision-encoder input shape
    but contains no semantic content."""
    rng = random.Random(seed)
    pixels = [(rng.randint(110, 145), rng.randint(110, 145), rng.randint(110, 145))
              for _ in range(width * height)]
    img = Image.new("RGB", (width, height))
    img.putdata(pixels)
    return img


def run_one(model, proc, img: Image.Image, max_new_tokens: int, device: str, label: str):
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
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, return_dict_in_generate=True)
    dt = time.time() - t0
    new_ids = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    last_id = new_ids[-1] if new_ids else -1
    hit_cap = (len(new_ids) >= max_new_tokens) and (last_id != EOS_TOKEN_ID)
    stop_reason = "max_new_tokens" if hit_cap else ("eos" if last_id == EOS_TOKEN_ID else "other")
    tail = proc.tokenizer.decode(new_ids[-80:], skip_special_tokens=False) if new_ids else ""
    print(f"[{label}] tokens={len(new_ids)} stop_reason={stop_reason} elapsed={dt:.1f}s")
    print(f"[{label}] tail: {tail[-200:]!r}")
    return {
        "tokens_emitted": len(new_ids),
        "stop_reason": stop_reason,
        "elapsed_s": dt,
        "last_token_id": last_id,
        "tail": tail[-200:],
    }


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: p3_image_inpaint.py <doc_id> <max_new_tokens>")
        return 2
    doc_id = sys.argv[1]
    max_new_tokens = int(sys.argv[2])

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    proc = AutoProcessor.from_pretrained("nanonets/Nanonets-OCR2-3B", revision="c3886ff00bb037ce7da24988c9eafaf1fe2bed72")
    model = AutoModelForImageTextToText.from_pretrained(
        "nanonets/Nanonets-OCR2-3B",
        revision="c3886ff00bb037ce7da24988c9eafaf1fe2bed72",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.tie_word_embeddings = True
    model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr()
    model = model.to(device).train(False)

    img_path = find_image_path(doc_id)
    if img_path is None or not img_path.exists():
        print(f"ERROR: image not found for doc_id={doc_id}")
        return 1
    img = Image.open(img_path).convert("RGB")
    print(f"image: {img_path}  size={img.size}")

    # 1. Baseline with real image
    baseline = run_one(model, proc, img, max_new_tokens, device, "baseline")

    # 2. With image replaced by grey noise of same dimensions
    noise_img = make_grey_noise(img.size[0], img.size[1])
    noise = run_one(model, proc, noise_img, max_new_tokens, device, "noise")

    # Discriminate
    persists = (baseline["stop_reason"] == "max_new_tokens"
                and noise["stop_reason"] == "max_new_tokens"
                and noise["tokens_emitted"] >= 0.5 * baseline["tokens_emitted"])
    if persists:
        verdict = "TEXT_ONLY_LOOP"  # loop runs even without image → text-self-reinforcing
    elif baseline["stop_reason"] == "max_new_tokens" and noise["stop_reason"] == "eos":
        verdict = "VISION_LOAD_BEARING"  # loop breaks when image is replaced → vision matters
    else:
        verdict = "AMBIGUOUS"

    out_path = ROOT / "results" / "p3_image_inpaint" / f"{doc_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "doc_id": doc_id,
        "max_new_tokens": max_new_tokens,
        "baseline": baseline,
        "noise": noise,
        "verdict": verdict,
    }, indent=2))
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
