"""Train a halt-direction at an arbitrary layer using cached residuals.

Used by Job 5 (adaptive SAE) when Job 4's H-E patching identifies a non-L24
mechanism layer. Loads cached residuals at the target layer for known positives
and matched controls, fits logistic regression to discriminate loop-state from
clean-halt-state, and saves the direction + standardizer params in the same
schema as the existing fix/halt_direction_L24.pt.

Output schema (matches fix/halt_direction_L24.pt for drop-in compatibility):
{
    "target_layer": int,
    "halt_direction": tensor (hidden,),
    "intercept": float,
    "scaler_mean": tensor (hidden,),
    "scaler_scale": tensor (hidden,),
}

Usage:
    python code/p3_train_halt_direction.py --target-layer 20 \\
        --out fix/halt_direction_L20.pt \\
        --cache-dir results/activations
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def collect_doc_ids() -> tuple[list[str], list[str]]:
    """Walk Job 1b + Job 1c output dirs, return (positives, controls) doc_id lists.

    Positive = trigger=='positive' AND stop=='max_new_tokens' (cap-hit infinite gen).
    Control  = trigger=='control'  AND stop=='eos' (clean halt).
    """
    positives: list[str] = []
    controls: list[str] = []
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
                positives.append(doc)
            elif trig == "control" and stop == "eos":
                controls.append(doc)
    return positives, controls


def load_residuals_at_layer(doc_id: str, target_layer: int, cache_dir: Path) -> torch.Tensor | None:
    """Load hidden_states.pt for one doc + slice out the target layer's residuals.

    Returns (T, hidden) tensor in float32, or None if missing.
    """
    h_path = cache_dir / doc_id / "hidden_states.pt"
    m_path = cache_dir / doc_id / "meta.json"
    if not (h_path.is_file() and m_path.is_file()):
        return None
    try:
        hs = torch.load(h_path, map_location="cpu", weights_only=True)  # (num_kept, T, hidden)
        meta = json.loads(m_path.read_text())
    except Exception as e:
        print(f"  [warn] failed to load {doc_id}: {e}")
        return None
    kept_layers = meta.get("kept_layers")
    if kept_layers is None or target_layer not in kept_layers:
        print(f"  [warn] {doc_id}: L{target_layer} not in kept_layers {kept_layers}")
        return None
    li = kept_layers.index(target_layer)
    return hs[li].float()  # (T, hidden)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-layer", type=int, required=True)
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--cache-dir", default=str(ROOT / "results" / "activations"))
    ap.add_argument("--max-positions-per-doc", type=int, default=1000,
                    help="Cap positions per doc to keep training tractable")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        print(f"FATAL: cache-dir {cache_dir} not found — Job 2 must run first.")
        return 2

    positives, controls = collect_doc_ids()
    print(f"Found {len(positives)} positives + {len(controls)} controls")
    if len(positives) < 3 or len(controls) < 3:
        print(f"FATAL: need at least 3 positives + 3 controls, got {len(positives)}+{len(controls)}")
        return 2

    rng = np.random.default_rng(args.seed)

    # Collect residuals — label 1 for positions in positive docs (in-loop state),
    # label 0 for positions in control docs (clean halt state).
    # We use the LAST max_positions_per_doc positions of each doc to focus on the
    # decision-moment region where halt-vs-continue is being decided.
    X_pos: list[np.ndarray] = []
    X_ctl: list[np.ndarray] = []
    n_pos_loaded = 0
    n_ctl_loaded = 0
    for doc in positives:
        h = load_residuals_at_layer(doc, args.target_layer, cache_dir)
        if h is None:
            continue
        # Take the last K positions of the doc (in-loop region).
        K = min(args.max_positions_per_doc, h.shape[0])
        X_pos.append(h[-K:].cpu().numpy())
        n_pos_loaded += 1
    for doc in controls:
        h = load_residuals_at_layer(doc, args.target_layer, cache_dir)
        if h is None:
            continue
        K = min(args.max_positions_per_doc, h.shape[0])
        X_ctl.append(h[-K:].cpu().numpy())
        n_ctl_loaded += 1

    if n_pos_loaded < 3 or n_ctl_loaded < 3:
        print(f"FATAL: loaded only {n_pos_loaded}+{n_ctl_loaded} docs with cached L{args.target_layer} residuals")
        return 2

    X = np.concatenate(X_pos + X_ctl, axis=0)  # (N, hidden)
    y = np.concatenate([
        np.ones(sum(x.shape[0] for x in X_pos), dtype=np.int64),
        np.zeros(sum(x.shape[0] for x in X_ctl), dtype=np.int64),
    ])
    print(f"Training set: X={X.shape}, y_pos={(y == 1).sum()}, y_neg={(y == 0).sum()}")

    # Standardize, then fit LR.
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    clf.fit(Xs, y)
    print(f"Train accuracy: {clf.score(Xs, y):.4f}")

    direction = clf.coef_.ravel().astype(np.float32)
    intercept = float(clf.intercept_[0])
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)

    payload = {
        "target_layer": int(args.target_layer),
        "halt_direction": torch.tensor(direction, dtype=torch.float32),
        "intercept": intercept,
        "scaler_mean": torch.tensor(mean, dtype=torch.float32),
        "scaler_scale": torch.tensor(scale, dtype=torch.float32),
        "n_positive_docs": n_pos_loaded,
        "n_control_docs": n_ctl_loaded,
        "max_positions_per_doc": args.max_positions_per_doc,
        "trained_at": "p3_train_halt_direction.py",
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"\nWrote {out_path}")
    print(f"  direction shape: {tuple(payload['halt_direction'].shape)}")
    print(f"  direction norm:  {torch.linalg.norm(payload['halt_direction']):.4f}")
    print(f"  intercept:       {intercept:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
