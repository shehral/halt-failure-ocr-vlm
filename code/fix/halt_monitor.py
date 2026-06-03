"""Phase 3 fix-prototype — L24 halt-direction monitor as a HuggingFace LogitsProcessor.

This is a PROTOTYPE, not a production fix. The L24 halt-direction is derived from
N=8 small-N pilot data with confound caveats (see docs/understanding_guide.md
Section 11d). HPC scale-up at N=40 will produce the actual production-ready
halt-direction.

What this module does:
  1. Trains (or loads) a halt-direction at L24: a learned 2048-dim direction in
     the residual stream where positive projection = "in halt-failure mode."
  2. Registers a forward hook on `model.model.language_model.layers[24]` that
     captures the last-position residual at each generation step.
  3. Provides a LogitsProcessor that, at each step, projects the L24 residual
     onto the halt direction and boosts the EOS logit if the projection > threshold
     for ≥ K consecutive steps.

Composable with `repetition_penalty` and `no_repeat_ngram_size` — applied AFTER
those in the LogitsProcessorList.

Usage:
    from fix.halt_monitor import HaltMonitorPipeline
    pipeline = HaltMonitorPipeline.from_pretrained(model, halt_direction_path="fix/halt_direction_L24.pt")
    pipeline.attach(model)
    out = model.generate(**inputs, logits_processor=[pipeline.processor])

Test:
    .venv/bin/python fix/halt_monitor.py --train   # train the halt direction
    .venv/bin/python fix/halt_monitor.py --test     # test on the 3 positives
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    LogitsProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTION_PATH = ROOT / "fix" / "halt_direction_L24.pt"

EOS_TOKEN_ID = 151645  # <|im_end|>
TARGET_LAYER = 24


# =================================================================
# Training the halt direction
# =================================================================

def train_halt_direction(positives, controls, target_layer=TARGET_LAYER,
                          position_step=25, save_path=DEFAULT_DIRECTION_PATH):
    """Train a logistic-regression halt direction at the target layer using
    within-doc gen-position-matched samples (the cleanest confound-controlled setup
    per Section 11d.5 of the understanding guide).

    Inputs: lists of doc_ids; loads cached residuals from results/activations/.
    Output: saves halt_direction tensor + standardizer mean/scale to save_path.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    ACT = ROOT / "results" / "activations"
    TRIG = ROOT / "results" / "p1_trigger_v2"

    proc = AutoProcessor.from_pretrained(
        "nanonets/Nanonets-OCR2-3B",
        revision="c3886ff00bb037ce7da24988c9eafaf1fe2bed72",
    )
    tok = proc.tokenizer

    def char_to_tok(ids, char_off):
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(tok.decode(ids[:mid], skip_special_tokens=False)) < char_off:
                lo = mid + 1
            else:
                hi = mid
        return max(0, lo - 1)

    def loop_start(doc_id, ids, text, trig):
        p = trig.get("production_patterns_fired", [])
        if "empty_rows" in p:
            m = re.search(r"<tr[^>]*>\s*(?:<t[dh][^>]*>\s*</t[dh]>\s*)+</tr>", text, re.I | re.S)
            return char_to_tok(ids, m.start()) if m else None
        if doc_id == "docvqa_pgjw0227_p5":
            m = re.search(r"<signature>", text, re.I)
            return char_to_tok(ids, m.start()) if m else None
        return None

    X_train, y_train = [], []
    for doc_id in positives:
        meta = json.loads((ACT / doc_id / "meta.json").read_text())
        layer_indices = torch.load(ACT / doc_id / "layer_indices.pt", weights_only=True).tolist()
        li = layer_indices.index(target_layer)
        hs = torch.load(ACT / doc_id / "hidden_states.pt", weights_only=True)
        ids = torch.load(TRIG / f"{doc_id}.tokens.pt", weights_only=True).tolist()
        text = (TRIG / f"{doc_id}.txt").read_text()
        trig = json.loads((TRIG / f"{doc_id}.json").read_text())
        ls = loop_start(doc_id, ids, text, trig)
        if ls is None:
            continue
        prompt_len = meta["prompt_len"]
        gen_cached = meta["gen_len"]
        # Pre-loop positions in same doc = label 0 (halt-fires-correctly mode)
        # In-loop positions = label 1 (halt-failure mode)
        pre = list(range(0, ls, position_step))
        inn = list(range(ls, gen_cached, position_step))
        # Balance
        n = min(len(pre), len(inn))
        rng = np.random.default_rng(hash(doc_id) & 0xffff)
        pre = list(rng.choice(pre, size=n, replace=False)) if len(pre) > n else pre
        inn = list(rng.choice(inn, size=n, replace=False)) if len(inn) > n else inn
        for p in pre:
            X_train.append(hs[li, prompt_len + p, :].float().numpy())
            y_train.append(0)
        for p in inn:
            X_train.append(hs[li, prompt_len + p, :].float().numpy())
            y_train.append(1)

    for doc_id in controls:
        meta = json.loads((ACT / doc_id / "meta.json").read_text())
        layer_indices = torch.load(ACT / doc_id / "layer_indices.pt", weights_only=True).tolist()
        li = layer_indices.index(target_layer)
        hs = torch.load(ACT / doc_id / "hidden_states.pt", weights_only=True)
        prompt_len = meta["prompt_len"]
        gen_cached = meta["gen_len"]
        positions = list(range(0, gen_cached, position_step))
        for p in positions:
            X_train.append(hs[li, prompt_len + p, :].float().numpy())
            y_train.append(0)

    X_train = np.stack(X_train); y_train = np.array(y_train)
    print(f"[train] {len(y_train)} samples ({y_train.sum()} pos, {(y_train == 0).sum()} ctl)")

    sc = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X_train), y_train)
    direction = clf.coef_[0]  # (2048,)
    intercept = clf.intercept_[0]
    print(f"[train] direction norm: {np.linalg.norm(direction):.3f}  intercept: {intercept:.3f}")

    torch.save({
        "target_layer": target_layer,
        "halt_direction": torch.tensor(direction, dtype=torch.float32),
        "intercept": float(intercept),
        "scaler_mean": torch.tensor(sc.mean_, dtype=torch.float32),
        "scaler_scale": torch.tensor(sc.scale_, dtype=torch.float32),
        "n_train_samples": int(len(y_train)),
        "n_positive_train": int(y_train.sum()),
    }, save_path)
    print(f"[train] saved to {save_path}")
    return save_path


# =================================================================
# Runtime: forward hook + LogitsProcessor
# =================================================================

class _HaltDirectionState:
    """Shared state between the forward hook and the LogitsProcessor."""
    def __init__(self, halt_direction, intercept, mean, scale, threshold, consecutive_required):
        self.halt_direction = halt_direction  # (hidden,) on device
        self.intercept = float(intercept)
        self.mean = mean       # (hidden,) on device
        self.scale = scale     # (hidden,) on device
        self.threshold = float(threshold)
        self.consecutive_required = int(consecutive_required)
        self.last_residual = None
        self.recent_scores = []
        self.eos_boost_applied = False

    def reset(self):
        self.last_residual = None
        self.recent_scores = []
        self.eos_boost_applied = False

    def update_from_residual(self, residual):
        # residual: (hidden,) on device
        normed = (residual.float() - self.mean) / (self.scale + 1e-9)
        score = float((normed * self.halt_direction).sum() + self.intercept)
        self.recent_scores.append(score)
        if len(self.recent_scores) > 1024:
            self.recent_scores = self.recent_scores[-512:]
        return score


def make_layer_hook(state: _HaltDirectionState):
    """Hook to attach to the target decoder layer. Captures the LAST-position
    residual at each forward pass — that's the next-token-prediction position."""
    def hook(_mod, _in, out):
        hs_tensor = out[0] if isinstance(out, tuple) else out
        # hs_tensor: (batch, seq, hidden). Grab the last position of batch 0.
        last_residual = hs_tensor[0, -1, :].detach()
        state.update_from_residual(last_residual)
    return hook


class HaltMonitorLogitsProcessor(LogitsProcessor):
    """Reads the latest L24 halt-direction score (computed by the forward hook),
    and boosts the EOS logit if the recent K scores are all above threshold."""
    def __init__(self, state: _HaltDirectionState, eos_token_id=EOS_TOKEN_ID, boost_db=30.0, verbose=False):
        self.state = state
        self.eos_token_id = eos_token_id
        # NOTE: `boost_db` is a misnomer. The math at line 222 below adds this value
        # DIRECTLY to the EOS logit (raw logit-add), NOT after a dB-to-log conversion.
        # The variable name persists for API compatibility; semantics are "raw additive
        # boost to EOS logit in logit-space, default +30." Confirmed by audit 2026-05-18:
        # this is the boost that produces CL-17's 83.1% reduction at N=19. Treating
        # this value as actual decibels (i.e., 30 dB → +6.91 logit) gives a far weaker
        # boost (verified by code/p97_run_bakeoff.py:67-72 prior to the alignment fix).
        self.boost_db = float(boost_db)  # raw additive logit-space boost (NOT actual dB)
        self.verbose = verbose

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Check if recent halt-direction scores are above threshold.
        recent = self.state.recent_scores[-self.state.consecutive_required:]
        if len(recent) >= self.state.consecutive_required and all(s > self.state.threshold for s in recent):
            scores = scores.clone()
            scores[:, self.eos_token_id] = scores[:, self.eos_token_id] + self.boost_db
            if self.verbose and not self.state.eos_boost_applied:
                print(f"  [halt-monitor] EOS boost APPLIED (recent K scores all > {self.state.threshold:.2f}; mean: {np.mean(recent):.2f})")
                self.state.eos_boost_applied = True
        return scores


class HaltMonitorPipeline:
    """High-level wrapper: attach hook + provide a LogitsProcessor."""
    def __init__(self, direction_payload, threshold=0.0, consecutive_required=8, boost_db=30.0, verbose=False):
        self.target_layer = int(direction_payload["target_layer"])
        self.state = _HaltDirectionState(
            halt_direction=direction_payload["halt_direction"],
            intercept=direction_payload["intercept"],
            mean=direction_payload["scaler_mean"],
            scale=direction_payload["scaler_scale"],
            threshold=threshold,
            consecutive_required=consecutive_required,
        )
        self.boost_db = boost_db
        self.verbose = verbose
        self.hook_handle = None
        self.processor = HaltMonitorLogitsProcessor(self.state, boost_db=boost_db, verbose=verbose)

    @classmethod
    def from_path(cls, path, threshold=0.0, consecutive_required=8, boost_db=30.0, verbose=False):
        payload = torch.load(path, weights_only=True)
        return cls(payload, threshold=threshold, consecutive_required=consecutive_required, boost_db=boost_db, verbose=verbose)

    def to(self, device):
        self.state.halt_direction = self.state.halt_direction.to(device)
        self.state.mean = self.state.mean.to(device)
        self.state.scale = self.state.scale.to(device)
        return self

    def attach(self, model):
        layer = model.model.language_model.layers[self.target_layer]
        self.hook_handle = layer.register_forward_hook(make_layer_hook(self.state))
        return self

    def detach(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def reset(self):
        self.state.reset()


# =================================================================
# Test harness: run on the 3 positives, compare with/without the monitor
# =================================================================

def test_on_positives():
    """Run each of the 3 positives twice: once with and once without the halt monitor.
    Compare generated-token counts and stop-reasons.
    """
    from PIL import Image

    POSITIVES = ["docvqa_kshm0227_p6", "docvqa_srgb0228_p2", "docvqa_pgjw0227_p5"]
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
    model = model.to(device)
    model.train(False)
    print(f"[test] model loaded on {device}")

    pipeline = HaltMonitorPipeline.from_path(
        DEFAULT_DIRECTION_PATH,
        threshold=0.0, consecutive_required=8, boost_db=30.0, verbose=True,
    ).to(device)
    print(f"[test] halt-direction loaded; target layer L{pipeline.target_layer}")

    manifest = json.loads((ROOT / "data" / "public_corpus" / "manifest.json").read_text())
    results = []
    for doc_id in POSITIVES:
        img_path = next(m["path"] for m in manifest if m["doc_id"] == doc_id)
        img = Image.open(ROOT / img_path).convert("RGB")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": USER_PROMPT}]},
        ]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        max_new_tokens = 6000  # keep modest for local; still much longer than legitimate output

        for use_monitor in (False, True):
            label = "with-monitor" if use_monitor else "BASELINE"
            print(f"\n[test] === {doc_id}  {label} ===")
            if use_monitor:
                pipeline.reset()
                pipeline.attach(model)
                logits_processors = [pipeline.processor]
            else:
                logits_processors = []
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
            if use_monitor:
                pipeline.detach()
            new_ids = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
            last_id = new_ids[-1]
            hit_cap = (len(new_ids) >= max_new_tokens) and (last_id != EOS_TOKEN_ID)
            stop_reason = "max_new_tokens" if hit_cap else ("eos" if last_id == EOS_TOKEN_ID else "other")
            print(f"  tokens={len(new_ids)}  stop_reason={stop_reason}  elapsed={dt:.1f}s")
            tail_text = proc.tokenizer.decode(new_ids[-100:], skip_special_tokens=False)
            print(f"  tail: {tail_text[-200:]!r}")
            results.append({
                "doc_id": doc_id, "use_monitor": use_monitor,
                "tokens_emitted": len(new_ids), "stop_reason": stop_reason,
                "elapsed_s": dt, "tail": tail_text[-200:],
            })

    out_path = ROOT / "results" / "p2_pilot" / "fix_prototype_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[test] wrote {out_path}")

    # Pretty summary
    print(f"\n[test] SUMMARY:")
    print(f"{'doc':<30}  {'baseline tokens':>18}  {'with-monitor tokens':>22}  {'reduction':>11}")
    docs_seen = sorted(set(r["doc_id"] for r in results))
    for doc in docs_seen:
        bl = next(r for r in results if r["doc_id"] == doc and not r["use_monitor"])
        wm = next(r for r in results if r["doc_id"] == doc and r["use_monitor"])
        red = (1 - wm["tokens_emitted"] / max(bl["tokens_emitted"], 1)) * 100
        print(f"{doc:<30}  {bl['tokens_emitted']:>18d}  {wm['tokens_emitted']:>22d}  {red:>10.1f}%")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.train:
        POSITIVES = ["docvqa_kshm0227_p6", "docvqa_srgb0228_p2", "docvqa_pgjw0227_p5"]
        CONTROLS  = ["docvqa_jrcy0227_p98", "docvqa_yghg0065_p1", "docvqa_gyjf0226_p1",
                      "docvqa_kzbn0226_p31", "docvqa_txpp0227_p9"]
        train_halt_direction(POSITIVES, CONTROLS, target_layer=TARGET_LAYER)
    if args.test:
        test_on_positives()
    if not args.train and not args.test:
        ap.print_help()


if __name__ == "__main__":
    main()
