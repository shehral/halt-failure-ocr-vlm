"""Run a single (doc_id, use_monitor) test of the halt monitor in a fresh
process. Avoids the cumulative MPS memory pressure that OOM'd the combined
test harness.

Usage:
    .venv/bin/python fix/test_halt_monitor.py <doc_id> <use_monitor:0|1> <max_new_tokens>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fix.halt_monitor import HaltMonitorPipeline, EOS_TOKEN_ID, DEFAULT_DIRECTION_PATH

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


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: test_halt_monitor.py <doc_id> <use_monitor:0|1> <max_new_tokens>")
        return 2
    doc_id = sys.argv[1]
    use_monitor = bool(int(sys.argv[2]))
    max_new_tokens = int(sys.argv[3])
    label = "with-monitor" if use_monitor else "BASELINE"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load model.
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
    model = model.to(device)
    model.train(False)
    print(f"[{label}] model loaded on {device}, doc={doc_id}")

    pipeline = None
    if use_monitor:
        pipeline = HaltMonitorPipeline.from_path(
            DEFAULT_DIRECTION_PATH,
            threshold=0.0, consecutive_required=8, boost_db=30.0, verbose=True,
        ).to(device)
        pipeline.attach(model)
        print(f"[{label}] halt-monitor attached at L{pipeline.target_layer}")

    # Find image path.
    for set_dir in ("public_corpus", "synthetic_stress", "real_loopy", "long_form_corpus", "supplementary_corpus"):
        m = ROOT / "data" / set_dir / "manifest.json"
        if m.exists():
            for e in json.loads(m.read_text()):
                if e["doc_id"] == doc_id:
                    img_path = ROOT / e["path"]
                    break
            else:
                continue
            break
    img = Image.open(img_path).convert("RGB")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": USER_PROMPT}]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logits_processors = [pipeline.processor] if use_monitor else []
    torch.manual_seed(0)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            logits_processor=logits_processors,
        )
    dt = time.time() - t0
    new_ids = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    last_id = new_ids[-1]
    hit_cap = (len(new_ids) >= max_new_tokens) and (last_id != EOS_TOKEN_ID)
    stop_reason = "max_new_tokens" if hit_cap else ("eos" if last_id == EOS_TOKEN_ID else "other")
    tail = proc.tokenizer.decode(new_ids[-80:], skip_special_tokens=False)
    halt_scores_tail = pipeline.state.recent_scores[-20:] if use_monitor else []

    print(f"[{label}] tokens={len(new_ids)}  stop_reason={stop_reason}  elapsed={dt:.1f}s")
    print(f"[{label}] tail: {tail[-200:]!r}")
    if use_monitor:
        print(f"[{label}] last 20 halt-scores: {[round(s, 2) for s in halt_scores_tail]}")

    out_path = ROOT / "results" / "p2_pilot" / f"fix_test_{doc_id}_{label.replace('-', '_')}.json"
    out_path.write_text(json.dumps({
        "doc_id": doc_id, "use_monitor": use_monitor,
        "max_new_tokens": max_new_tokens,
        "tokens_emitted": len(new_ids), "stop_reason": stop_reason,
        "elapsed_s": dt, "last_token_id": last_id,
        "tail": tail[-200:],
        "halt_scores_tail": halt_scores_tail,
    }, indent=2))
    print(f"[{label}] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
