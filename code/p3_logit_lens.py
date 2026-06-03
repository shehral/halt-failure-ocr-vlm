"""Logit lens at every cached layer — discovery experiment.

Question: WHERE in the forward pass does the EOS-logit get suppressed on positive
(infinite-gen) docs vs amplified on control (clean-halt) docs?

Method: classic logit lens — for each cached layer's residual stream, apply the
model's final RMSNorm + lm_head and read out the EOS logit. Track per-layer
per-position EOS logit trajectory for each doc, then aggregate positives vs
controls.

Output (results/p3_logit_lens/_summary.json):
- per_layer_mean_eos_logit_positives: dict[layer -> mean over positions over docs]
- per_layer_mean_eos_logit_controls:  dict[layer -> mean]
- divergence_layer: the layer where the gap (positive - control) is most negative
  (i.e., where EOS gets suppressed *more* on positives than controls)
- per_doc_trajectories: dict[doc_id -> per-layer EOS-logit at last position]

This is uses the model only for lm_head + final norm — light compute (~30s/doc).

Usage:
    python code/p3_logit_lens.py --cache-dir results/activations \
        --out-dir results/p3_logit_lens
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText

ROOT = Path(__file__).resolve().parents[1]
EOS_TOKEN_ID = 151645  # <|im_end|> for Qwen2.5-VL


def load_model_head():
    """Load only the language head components needed for logit lens."""
    model = AutoModelForImageTextToText.from_pretrained(
        "nanonets/Nanonets-OCR2-3B",
        revision="c3886ff00bb037ce7da24988c9eafaf1fe2bed72",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.tie_word_embeddings = True
    model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr()
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.train(False)
    return model


def collect_doc_kinds(cache_dir: Path) -> dict[str, str]:
    """Return doc_id -> 'positive'|'control'|'unknown' map by reading per-doc trigger files."""
    kinds: dict[str, str] = {}
    seen: set[str] = set()
    for dir_name in ("p1_cuda_n40", "p1_cuda_supplementary"):
        d = ROOT / "results" / dir_name
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name.startswith("_"):
                continue
            try:
                r = json.loads(p.read_text())
            except Exception:
                continue
            doc = r.get("doc_id")
            if not doc or doc in seen:
                continue
            seen.add(doc)
            trig = r.get("trigger")
            stop = r.get("stop_reason")
            if trig == "positive" and stop == "max_new_tokens":
                kinds[doc] = "positive"
            elif trig == "control":
                kinds[doc] = "control"
            else:
                kinds[doc] = "unknown"
    return kinds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "results" / "activations"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "p3_logit_lens"))
    ap.add_argument("--last-n-positions", type=int, default=200,
                    help="Per doc, project lens on the last N positions only (decision region)")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cache_dir.is_dir():
        print(f"FATAL: cache-dir {cache_dir} not found — Job 2 must run first.")
        return 2

    kinds = collect_doc_kinds(cache_dir)
    cached_docs = sorted([d for d in cache_dir.iterdir()
                          if d.is_dir() and (d / "hidden_states.pt").is_file()])
    print(f"Found {len(cached_docs)} cached docs, {sum(1 for d in cached_docs if kinds.get(d.name) == 'positive')} positives, {sum(1 for d in cached_docs if kinds.get(d.name) == 'control')} controls")

    print("Loading model (lm_head + final RMSNorm)...")
    model = load_model_head()
    device = next(model.parameters()).device
    lm_head = model.lm_head
    final_norm = model.model.language_model.norm

    # per_layer_eos_logits[layer_idx]['positive' | 'control'] -> list of mean-EOS-logit-per-doc
    per_layer_means: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"positive": [], "control": []})
    per_doc_trajectories: dict[str, dict[str, float]] = {}

    for doc_dir in cached_docs:
        doc = doc_dir.name
        kind = kinds.get(doc, "unknown")
        if kind == "unknown":
            continue
        try:
            hs = torch.load(doc_dir / "hidden_states.pt", map_location=device, weights_only=True)
            meta = json.loads((doc_dir / "meta.json").read_text())
        except Exception as e:
            print(f"  [warn] {doc}: load failed: {e}")
            continue
        kept_layers = meta.get("kept_layers", [])
        if not kept_layers:
            continue
        # hs: (num_kept, T, hidden)
        T = hs.shape[1]
        start = max(0, T - args.last_n_positions)
        per_layer_trajectory: dict[str, float] = {}
        with torch.inference_mode():
            for li, L in enumerate(kept_layers):
                # Apply final norm + lm_head to layer L's residuals at last_n positions
                resid = hs[li, start:].to(torch.bfloat16)
                normed = final_norm(resid)  # (Nlast, hidden)
                logits = lm_head(normed)  # (Nlast, vocab)
                eos_logit = logits[:, EOS_TOKEN_ID].float().cpu().numpy()
                mean_eos = float(eos_logit.mean())
                per_layer_means[L][kind].append(mean_eos)
                per_layer_trajectory[f"L{L:02d}"] = mean_eos
        per_doc_trajectories[doc] = {"kind": kind, "trajectory": per_layer_trajectory}
        print(f"  [{kind:<8}] {doc:<42} L0={per_layer_trajectory.get('L00', 'NA'):>8.3f} L24={per_layer_trajectory.get('L24', 'NA'):>8.3f} L35={per_layer_trajectory.get('L35', 'NA'):>8.3f}")

    # Aggregate per-layer
    per_layer_summary: dict[str, dict[str, float]] = {}
    divergence_layer = None
    max_gap = -float("inf")
    for L in sorted(per_layer_means.keys()):
        pos_vals = per_layer_means[L]["positive"]
        ctl_vals = per_layer_means[L]["control"]
        if not pos_vals or not ctl_vals:
            continue
        mp = float(np.mean(pos_vals))
        mc = float(np.mean(ctl_vals))
        gap = mp - mc  # negative = EOS suppressed more on positives (mechanism evidence)
        per_layer_summary[f"L{L:02d}"] = {
            "mean_positive": round(mp, 4),
            "mean_control": round(mc, 4),
            "gap_pos_minus_ctl": round(gap, 4),
            "n_positives": len(pos_vals),
            "n_controls": len(ctl_vals),
        }
        # Most-suppressive layer = lowest gap (most negative)
        if abs(gap) > abs(max_gap):
            max_gap = gap
            divergence_layer = f"L{L:02d}"

    summary = {
        "experiment": "logit-lens-every-layer",
        "n_positives": sum(1 for d in per_doc_trajectories.values() if d["kind"] == "positive"),
        "n_controls": sum(1 for d in per_doc_trajectories.values() if d["kind"] == "control"),
        "per_layer_summary": per_layer_summary,
        "divergence_layer": divergence_layer,
        "max_gap": round(max_gap, 4),
        "per_doc_trajectories": per_doc_trajectories,
        "interpretation": (
            "gap_pos_minus_ctl negative at layer L means EOS-logit is SUPPRESSED more on "
            "positives than controls when read out at L. The most-negative gap = the layer "
            "where the halt-failure manifests most strongly in EOS logits."
        ),
    }

    out_path = out_dir / "_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print("=" * 60)
    print("LOGIT LENS SUMMARY")
    print("=" * 60)
    print(f"divergence_layer: {divergence_layer}  (max_gap = {max_gap:+.4f})")
    print(f"per-layer gaps (positive - control):")
    for L_key, vals in per_layer_summary.items():
        print(f"  {L_key}: gap={vals['gap_pos_minus_ctl']:+.4f}  pos={vals['mean_positive']:+.4f}  ctl={vals['mean_control']:+.4f}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
