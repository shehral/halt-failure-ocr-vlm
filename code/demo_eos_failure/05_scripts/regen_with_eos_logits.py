"""
regen_with_eos_logits.py
========================

THE missing piece for the demo. Every existing cap-hit generation in this repo
records the *output tokens* but throws away the per-step *logit distribution* —
which means we have evidence the model didn't stop, but no direct picture of
*how strongly* EOS was being suppressed at each step.

This script re-runs generation on one or more demo documents with
`output_scores=True` so we can save the full per-step trajectory of:
  - the chosen token id (what the model actually emitted)
  - the EOS-token logit (the raw "stop now" score, pre-softmax)
  - the EOS-token softmax probability (the calibrated "stop now" probability)
  - the top-1 chosen-token logit (so we can see the EOS-vs-chosen *margin*)
  - the top-5 alternates with their logits (for richer visualization)

Output: one .pt + .json sidecar per demo doc, written to ../03_eos_trajectories/

This is the ground-truth data the team needs to *see* the suppression that
makes phantom rows happen.

Runs locally on an M4 Pro (24GB) at ~5-10 tok/sec in eager attention.
Expected wall time per doc:
  - clean control (~200 tok)       :  ~1-2 min
  - phantom-rows positive (3000 tok of max_new_tokens=3000) : ~6-12 min
  - tighter budget controls overall runtime; defaults are demo-sized.

Usage:
  cd <repo_root>
  source .venv/bin/activate    # or whichever Python 3.11 env has torch+transformers
  python demo_eos_failure/05_scripts/regen_with_eos_logits.py \
      --doc docvqa_kshm0227_p6 \
      --max-new-tokens 3000

  # or run the full demo set:
  python demo_eos_failure/05_scripts/regen_with_eos_logits.py --all
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Heuristic #71: on HPC compute nodes the Squid proxy isn't reachable, so any
# silent HF cache miss becomes a network call that hangs or 503s. We honor the
# HF_HUB_OFFLINE env var by passing local_files_only=True to from_pretrained,
# which makes the failure mode loud (FileNotFoundError) instead of silent
# (meta-init zeros and generation collapses to "!" forever).
LOCAL_ONLY = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]                 # <repo_root>
DEMO = REPO / "demo_eos_failure"

MODEL_ID = "nanonets/Nanonets-OCR2-3B"
MODEL_REVISION = "c3886ff00bb037ce7da24988c9eafaf1fe2bed72"   # pinned per CLAUDE.md
PROMPT = (
    "Extract the text from the above document as if you were reading it naturally. "
    "Return the tables in html format. Return the equations in LaTeX representation. "
    "If there is an image in the document and image caption is not present, add a "
    "small description of the image inside the <img></img> tag; otherwise, add the "
    "image caption inside <img></img>. Watermarks should be wrapped in brackets. "
    "Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in "
    "brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
    "Prefer using ☐ and ☑ for check boxes."
)

# Map demo doc id -> the image path on disk
DOCS = {
    "docvqa_kshm0227_p6":   DEMO / "01_documents" / "01_phantom_rows_PRIMARY__docvqa_kshm0227_p6.png",
    "docvqa_srgb0228_p2":   DEMO / "01_documents" / "02_phantom_rows_SECONDARY__docvqa_srgb0228_p2.png",
    "docvqa_fhxn0226_p2":   DEMO / "01_documents" / "04_CLEAN_CONTROL__docvqa_fhxn0226_p2.png",
}

OUT_DIR = DEMO / "03_eos_trajectories"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model loading — with the MANDATORY lm_head tying fix (see CLAUDE.md)
# ---------------------------------------------------------------------------

def load_model_and_processor():
    """Load Nanonets-OCR2-3B with the workaround for the missing lm_head weight.

    Why this is needed:
      The checkpoint relies on tie_word_embeddings (lm_head shares weights with
      the input embedding), but transformers 5.x reads the flag off the outer
      config (which is False) instead of text_config (which is True).
      tie_weights() becomes a no-op and lm_head stays at meta-init zeros, which
      makes generation collapse to "!" forever. We force the tie ourselves.

    This block is identical to what fix/halt_monitor.py and code/p97_run_bakeoff.py
    do; it is non-negotiable.
    """
    # Pick the best device available on this machine.
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16     # MPS supports bf16 in recent torch builds
    else:
        device = "cpu"
        dtype = torch.float32       # bf16 on CPU is too slow to be useful

    print(f"[load] device={device}  dtype={dtype}  local_files_only={LOCAL_ONLY}")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=LOCAL_ONLY,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        local_files_only=LOCAL_ONLY,
    )

    # ---- The lm_head fix ----
    model.config.tie_word_embeddings = True
    model.tie_weights()
    assert (
        model.lm_head.weight.data_ptr()
        == model.model.language_model.embed_tokens.weight.data_ptr()
    ), "lm_head was not actually tied - generation will output '!' forever"

    model = model.to(device)
    model.eval()  # switch to inference mode
    return model, processor, device


# ---------------------------------------------------------------------------
# The actual generation pass with per-step EOS logit capture
# ---------------------------------------------------------------------------

def _resolve_eos_ids(processor, model):
    """Nanonets-OCR2 uses <|im_end|> as the end-of-turn marker.

    We collect *every* token id that the model could legitimately use to stop:
    config.eos_token_id (single int or list) plus the explicit <|im_end|> if it
    isn't already covered. We'll later report the *max* logit across these
    "halting tokens" so we don't undercount EOS-suppression.
    """
    eos_ids = []
    cfg_eos = model.generation_config.eos_token_id
    if isinstance(cfg_eos, int):
        eos_ids.append(cfg_eos)
    elif isinstance(cfg_eos, (list, tuple)):
        eos_ids.extend(cfg_eos)

    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end >= 0 and im_end not in eos_ids:
        eos_ids.append(im_end)
    return eos_ids


@torch.inference_mode()
def generate_with_traces(model, processor, image, max_new_tokens: int):
    """Run greedy generation, returning per-step traces.

    Why greedy (do_sample=False)?
      Reproducibility. Every run on the same image + same model + greedy decode
      gives the same token sequence and the same logit traces. With sampling,
      no two runs are alike and the demo can't be replayed.

    Returns a dict of CPU tensors / lists, all length T = number of generated steps.
    """
    eos_ids = _resolve_eos_ids(processor, model)
    print(f"[gen] halting-token ids = {eos_ids}")

    # Build the chat-format prompt the model was fine-tuned on.
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        # THESE TWO FLAGS ARE THE WHOLE POINT OF THIS SCRIPT
        return_dict_in_generate=True,
        output_scores=True,
        # repetition_penalty and no_repeat_ngram_size left at default (no anti-loop)
        # so the model is free to exhibit the failure mode we're studying.
    )
    elapsed = time.time() - t0
    print(f"[gen] done in {elapsed:.1f}s")

    # out.sequences shape: (batch=1, prompt_len + T)
    # out.scores: tuple of T tensors, each shape (batch=1, vocab_size)
    prompt_len = inputs["input_ids"].shape[1]
    gen_token_ids = out.sequences[0, prompt_len:].cpu().tolist()
    scores = out.scores                       # tuple of length T
    T = len(scores)
    vocab_size = scores[0].shape[-1]

    # Stack into one (T, vocab) tensor so we can vectorize the per-step extracts.
    # bf16 -> float32 here for numerical comfort; this is ~T * 152k * 4 bytes,
    # which for T=3000 is ~1.7GB - manageable on M4 Pro RAM.
    logits = torch.stack([s[0] for s in scores], dim=0).float().cpu()

    # Per-step softmax - needed to convert logit margins into actual probabilities.
    probs = torch.softmax(logits, dim=-1)

    # The "EOS logit" we save is the MAX over all halting-token ids.
    # That is the most generous reading of "how strong was the stop signal at step t."
    eos_idx = torch.tensor(eos_ids, dtype=torch.long)
    eos_logits = logits.index_select(dim=-1, index=eos_idx).max(dim=-1).values    # (T,)
    # Sum prob across halting tokens because they're mutually-exclusive stop events.
    eos_probs  = probs.index_select(dim=-1, index=eos_idx).sum(dim=-1)            # (T,)

    chosen_idx = torch.tensor(gen_token_ids).unsqueeze(-1)
    chosen_logits = logits.gather(dim=-1, index=chosen_idx).squeeze(-1)
    chosen_probs  = probs.gather (dim=-1, index=chosen_idx).squeeze(-1)

    # Top-5 alternates per step - useful when you want to see what the model
    # *would* have said if it didn't pick the loop-token.
    top5 = torch.topk(logits, k=5, dim=-1)
    top5_ids    = top5.indices                              # (T, 5)
    top5_logits = top5.values                               # (T, 5)

    traces = {
        "gen_token_ids":  gen_token_ids,
        "eos_logits":     eos_logits,             # (T,)
        "eos_probs":      eos_probs,              # (T,)
        "chosen_logits":  chosen_logits,          # (T,)
        "chosen_probs":   chosen_probs,           # (T,)
        "top5_token_ids": top5_ids,               # (T, 5)
        "top5_logits":    top5_logits,            # (T, 5)
        "eos_token_ids":  eos_ids,                # config
        "T":              T,
        "vocab_size":     vocab_size,
        "elapsed_s":      elapsed,
        "max_new_tokens": max_new_tokens,
        "hit_max_new_tokens": (T >= max_new_tokens),
    }
    return traces


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_traces(doc_id: str, traces: dict, processor):
    """Save the heavy tensors as a .pt and a slim human-readable JSON sidecar."""
    pt_path   = OUT_DIR / f"{doc_id}_eos_trace.pt"
    json_path = OUT_DIR / f"{doc_id}_eos_trace.json"

    # The .pt file holds the full tensors - the visualization script reads this.
    torch.save({
        "doc_id":          doc_id,
        "gen_token_ids":   torch.tensor(traces["gen_token_ids"], dtype=torch.long),
        "eos_logits":      traces["eos_logits"],
        "eos_probs":       traces["eos_probs"],
        "chosen_logits":   traces["chosen_logits"],
        "chosen_probs":    traces["chosen_probs"],
        "top5_token_ids":  traces["top5_token_ids"],
        "top5_logits":     traces["top5_logits"],
        "eos_token_ids":   traces["eos_token_ids"],
        "model_id":        MODEL_ID,
        "model_revision":  MODEL_REVISION,
    }, pt_path)

    # The JSON sidecar holds summary stats + the first/last 20 steps decoded
    # so a teammate can eyeball it without loading torch.
    tok = processor.tokenizer
    def decode_one(tid):
        try:
            return tok.decode([int(tid)], clean_up_tokenization_spaces=False)
        except Exception:
            return f"<id={int(tid)}>"

    T = traces["T"]
    summary = {
        "doc_id":              doc_id,
        "T_generated":         T,
        "hit_max_new_tokens":  traces["hit_max_new_tokens"],
        "elapsed_s":           traces["elapsed_s"],
        "eos_token_ids":       traces["eos_token_ids"],
        "eos_logit_first10":   traces["eos_logits"][:10].tolist(),
        "eos_logit_last10":    traces["eos_logits"][-10:].tolist(),
        "eos_logit_min":       float(traces["eos_logits"].min()),
        "eos_logit_max":       float(traces["eos_logits"].max()),
        "eos_logit_mean":      float(traces["eos_logits"].mean()),
        "eos_prob_min":        float(traces["eos_probs"].min()),
        "eos_prob_max":        float(traces["eos_probs"].max()),
        "eos_prob_mean":       float(traces["eos_probs"].mean()),
        "first_20_steps_decoded": [
            {"step": i, "tok_id": int(traces["gen_token_ids"][i]),
             "text": decode_one(traces["gen_token_ids"][i]),
             "eos_logit":    float(traces["eos_logits"][i]),
             "eos_prob":     float(traces["eos_probs"][i]),
             "chosen_logit": float(traces["chosen_logits"][i])}
            for i in range(min(20, T))
        ],
        "last_20_steps_decoded": [
            {"step": T - 20 + i,
             "tok_id": int(traces["gen_token_ids"][T - 20 + i]),
             "text": decode_one(traces["gen_token_ids"][T - 20 + i]),
             "eos_logit":    float(traces["eos_logits"][T - 20 + i]),
             "eos_prob":     float(traces["eos_probs"][T - 20 + i]),
             "chosen_logit": float(traces["chosen_logits"][T - 20 + i])}
            for i in range(min(20, T))
        ],
    }
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"[save] {pt_path.name}  +  {json_path.name}")
    return pt_path, json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", choices=list(DOCS.keys()), default=None,
                    help="single doc id to regenerate (default: kshm phantom-rows primary)")
    ap.add_argument("--all", action="store_true",
                    help="regenerate all three demo docs (control + 2 phantom-rows)")
    ap.add_argument("--max-new-tokens", type=int, default=3000,
                    help="cap on generation length (default 3000 - enough to *see* the loop)")
    args = ap.parse_args()

    if args.all:
        targets = list(DOCS.keys())
    elif args.doc:
        targets = [args.doc]
    else:
        targets = ["docvqa_kshm0227_p6"]   # the cleanest phantom-rows exemplar

    model, processor, device = load_model_and_processor()

    for doc_id in targets:
        img_path = DOCS[doc_id]
        if not img_path.exists():
            print(f"[skip] image not found: {img_path}")
            continue
        print(f"\n========== {doc_id} ==========")
        img = Image.open(img_path).convert("RGB")
        # Controls don't need 3000 steps - let them stop naturally well below.
        budget = args.max_new_tokens if "control" not in img_path.name.lower() else 600
        traces = generate_with_traces(model, processor, img, max_new_tokens=budget)
        save_traces(doc_id, traces, processor)

    print("\n[done] All traces saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
