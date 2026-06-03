"""
plot_eos_failure.py
===================

Reads the per-step traces produced by regen_with_eos_logits.py and renders the
visualizations the team will look at during the demo.

Each figure is saved as both PNG (slides) and SVG (zoomable).

Figures produced:
  fig01_eos_logit_trajectory_<doc>.{png,svg}
      Per-step EOS-token logit over the entire generation. The line falls below
      every reasonable threshold during the loop region. This is THE smoking-gun
      picture: "EOS is not being scored highly enough at any step to stop."

  fig02_eos_vs_chosen_margin_<doc>.{png,svg}
      Per-step: chosen-token logit on top, EOS logit on bottom. The *gap*
      between them is how confidently the model is choosing the loop token
      over stopping. When the gap is large and stable, the attractor is deep.

  fig03_eos_softmax_prob_<doc>.{png,svg}
      Same trace but in probability space (after softmax). This is what
      production code would actually see if it looked at output probabilities.

  fig04_caphit_vs_control_overlay.{png,svg}
      Two trajectories overlaid: one phantom-rows cap-hit + one clean control.
      The control's EOS logit climbs as the model approaches end-of-document;
      the cap-hit's stays flat-low forever.

  fig05_token_id_sequence_<doc>.{png,svg}
      Histogram of which token ids dominate the generation. For phantom-rows
      docs you'll see 3-5 token ids accounting for ~80% of all emissions
      ('<', 'tr', '>', '<', 'td'...) - the surface signature of the loop.

Usage:
  python demo_eos_failure/05_scripts/plot_eos_failure.py             # plot all available
  python demo_eos_failure/05_scripts/plot_eos_failure.py --doc docvqa_kshm0227_p6
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "demo_eos_failure"
TRACE_DIR = DEMO / "03_eos_trajectories"
OUT_DIR   = DEMO / "04_visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pretty styling - same scheme across all figures so the team can read them as a set
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

CAPHIT_COLOR  = "#c1272d"   # red - phantom-rows / cap-hit
CONTROL_COLOR = "#4a90e2"   # blue - clean halt
EOS_COLOR     = "#2c8a3d"   # green - EOS logit/prob


def _save(fig, name: str):
    """Write both PNG and SVG so the team can drop into slides or zoom in."""
    png = OUT_DIR / f"{name}.png"
    svg = OUT_DIR / f"{name}.svg"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {png.name}  +  {svg.name}")


def _load_trace(doc_id: str):
    """Return the saved trace tensor dict for one doc, or None if missing."""
    pt = TRACE_DIR / f"{doc_id}_eos_trace.pt"
    if not pt.exists():
        print(f"[skip] missing trace: {pt.name} - run regen_with_eos_logits.py first")
        return None
    return torch.load(pt, weights_only=False)


# ---------------------------------------------------------------------------
# Figure 1: per-step EOS logit
# ---------------------------------------------------------------------------

def plot_eos_logit_trajectory(doc_id: str, trace: dict, is_caphit: bool):
    eos_logits = trace["eos_logits"].numpy()
    T = len(eos_logits)
    fig, ax = plt.subplots(figsize=(11, 4))

    color = CAPHIT_COLOR if is_caphit else CONTROL_COLOR
    label = "phantom-rows cap-hit" if is_caphit else "clean control"

    ax.plot(np.arange(T), eos_logits, color=color, lw=0.8, alpha=0.9, label=label)
    # Mean line for orientation
    ax.axhline(float(eos_logits.mean()), color=color, ls=":", lw=1.0, alpha=0.6,
               label=f"mean = {eos_logits.mean():.2f}")
    ax.set_xlabel("generation step  t")
    ax.set_ylabel("EOS-token logit  (max over halting-token ids)")
    ax.set_title(f"EOS logit per generation step — {doc_id}\n"
                 f"T = {T} steps,  hit_max_new_tokens = {bool(T >= 600)}")
    ax.legend(loc="upper right", framealpha=0.9)

    _save(fig, f"fig01_eos_logit_trajectory__{doc_id}")


# ---------------------------------------------------------------------------
# Figure 2: chosen-token logit vs EOS logit margin
# ---------------------------------------------------------------------------

def plot_eos_vs_chosen(doc_id: str, trace: dict, is_caphit: bool):
    eos_logits    = trace["eos_logits"].numpy()
    chosen_logits = trace["chosen_logits"].numpy()
    T = len(eos_logits)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(chosen_logits, color="#444", lw=0.8, label="chosen-token logit (what the model picked)")
    ax.plot(eos_logits,    color=EOS_COLOR, lw=0.9, label="EOS-token logit (what stopping would need)")

    # Shade the gap to make it visually obvious
    ax.fill_between(np.arange(T), eos_logits, chosen_logits,
                    where=(chosen_logits > eos_logits),
                    color=CAPHIT_COLOR if is_caphit else CONTROL_COLOR,
                    alpha=0.15, label="margin (chosen > EOS)")
    ax.set_xlabel("generation step  t")
    ax.set_ylabel("raw logit")
    title_tag = "cap-hit" if is_caphit else "control"
    ax.set_title(f"Chosen-token logit vs EOS logit — {doc_id}  ({title_tag})")
    ax.legend(loc="best", framealpha=0.9)

    _save(fig, f"fig02_eos_vs_chosen_margin__{doc_id}")


# ---------------------------------------------------------------------------
# Figure 3: EOS softmax probability (production-readable)
# ---------------------------------------------------------------------------

def plot_eos_softmax_prob(doc_id: str, trace: dict, is_caphit: bool):
    eos_probs = trace["eos_probs"].numpy()
    T = len(eos_probs)

    fig, ax = plt.subplots(figsize=(11, 4))
    color = CAPHIT_COLOR if is_caphit else CONTROL_COLOR
    # Log y-axis so probabilities orders-of-magnitude apart stay readable.
    ax.semilogy(np.arange(T), np.clip(eos_probs, 1e-20, 1.0), color=color, lw=0.7)
    ax.axhline(0.5, color="#888", ls="--", lw=0.7, label="P(EOS) = 0.5  (would stop)")
    ax.set_xlabel("generation step  t")
    ax.set_ylabel("P(EOS at step t)  [log scale]")
    ax.set_ylim(1e-10, 1.0)
    ax.set_title(f"EOS softmax probability per step — {doc_id}\n"
                 f"max = {eos_probs.max():.2e},  mean = {eos_probs.mean():.2e}")
    ax.legend()

    _save(fig, f"fig03_eos_softmax_prob__{doc_id}")


# ---------------------------------------------------------------------------
# Figure 4: cap-hit vs control overlay
# ---------------------------------------------------------------------------

def plot_caphit_vs_control(caphit_id: str, control_id: str):
    caphit  = _load_trace(caphit_id)
    control = _load_trace(control_id)
    if caphit is None or control is None:
        return

    eos_c = caphit ["eos_logits"].numpy()
    eos_v = control["eos_logits"].numpy()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(np.arange(len(eos_c)), eos_c, color=CAPHIT_COLOR, lw=0.7, alpha=0.9,
            label=f"phantom-rows: {caphit_id}  (T={len(eos_c)})")
    ax.plot(np.arange(len(eos_v)), eos_v, color=CONTROL_COLOR, lw=1.0, alpha=0.95,
            label=f"clean control: {control_id}  (T={len(eos_v)})")
    # Mark where the control fired EOS (its last step)
    if len(eos_v) > 0:
        ax.axvline(len(eos_v) - 1, color=CONTROL_COLOR, ls=":", lw=1.0, alpha=0.6,
                   label="control emitted EOS here")

    ax.set_xlabel("generation step  t")
    ax.set_ylabel("EOS-token logit")
    ax.set_title("EOS logit trajectory — cap-hit vs. clean control\n"
                 "Look for: cap-hit stays flat-low for thousands of steps; control rises and fires.")
    ax.legend(loc="best", framealpha=0.9)

    _save(fig, "fig04_caphit_vs_control_overlay")


# ---------------------------------------------------------------------------
# Figure 5: which tokens dominate the generation
# ---------------------------------------------------------------------------

def plot_token_dominance(doc_id: str, trace: dict):
    ids = trace["gen_token_ids"].tolist()
    counter = Counter(ids)
    top = counter.most_common(15)
    labels = [f"id={tid}" for tid, _ in top]
    counts = [c for _, c in top]
    total  = sum(counter.values())
    pct = [100.0 * c / total for c in counts]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(range(len(top)), pct[::-1], color=CAPHIT_COLOR, alpha=0.75)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("% of all generated tokens")
    top_share = sum(pct[:5])
    ax.set_title(f"Token-id dominance — {doc_id}\n"
                 f"Top 5 token ids account for {top_share:.1f}% of {total} generated tokens "
                 f"(loop signature)")

    _save(fig, f"fig05_token_dominance__{doc_id}")


# ---------------------------------------------------------------------------
# Bonus: residual-projection trajectory (uses pre-existing tg3 data)
# ---------------------------------------------------------------------------

def plot_residual_projection_proxy(doc_id: str):
    """Plot the cosine-to-halt-centroid trajectory from tg3_eos_manifold/.

    This is a PROXY for EOS suppression - it measures how close the residual
    stream is to the "halt" attractor cluster in 2048-dim space. It's been
    computed for srgb0228_p2 and arxiv_table_000266 already (no model rerun needed).
    """
    src = None
    if doc_id == "docvqa_srgb0228_p2":
        src = TRACE_DIR / "srgb0228_p2_residual_projection_PROXY" / "trajectory.pt"
    elif doc_id == "arxiv_table_000266":
        src = TRACE_DIR / "arxiv_000266_residual_projection_PROXY" / "trajectory.pt"
    if not src or not src.exists():
        return

    traj = torch.load(src, weights_only=False)
    # Schema: dict-like with keys cosine_to_centroid, euclidean_to_centroid, reconstruction_error
    cos = traj.get("cosine_to_centroid")
    if cos is None: return
    cos = cos.numpy() if hasattr(cos, "numpy") else np.asarray(cos)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(np.arange(len(cos)), cos, color="#7a3eb1", lw=0.7)
    ax.set_xlabel("generation step  t")
    ax.set_ylabel("cosine similarity to halt-direction centroid (L24)")
    ax.set_title(f"L24 residual-stream alignment with halt centroid — {doc_id}\n"
                 f"Proxy: high values = 'model is geometrically in the halt-failure state'")

    _save(fig, f"figXX_residual_projection_proxy__{doc_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=None,
                    help="single doc id (default: plot all available traces)")
    args = ap.parse_args()

    available = sorted(p.stem.replace("_eos_trace", "")
                       for p in TRACE_DIR.glob("*_eos_trace.pt"))
    if not available:
        print("No *_eos_trace.pt files found. Run regen_with_eos_logits.py first.")
        return

    targets = [args.doc] if args.doc else available
    print(f"[plot] traces available: {available}")
    print(f"[plot] plotting: {targets}")

    for doc_id in targets:
        trace = _load_trace(doc_id)
        if trace is None: continue
        is_caphit = "fhxn" not in doc_id   # the one control is fhxn; everything else is cap-hit
        plot_eos_logit_trajectory(doc_id, trace, is_caphit)
        plot_eos_vs_chosen      (doc_id, trace, is_caphit)
        plot_eos_softmax_prob   (doc_id, trace, is_caphit)
        if is_caphit:
            plot_token_dominance(doc_id, trace)
        # Optional bonus if pre-existing tg3 proxy exists for this doc
        plot_residual_projection_proxy(doc_id)

    # Overlay figure only if we have both a cap-hit and the control
    has_control = any("fhxn" in d for d in available)
    caphit_options = [d for d in available if "fhxn" not in d]
    if has_control and caphit_options:
        plot_caphit_vs_control(caphit_options[0], "docvqa_fhxn0226_p2")

    print("\n[done] All figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
