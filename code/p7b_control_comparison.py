"""
Phase-7b control comparison — the critic-tick-#48-flagged missing experiment.

For CL-25 to hold, the PCA pre-vs-post-crossover |Cohen's d| effect must be HALT-SPECIFIC,
not just "long-generation state shift". This script runs the same PCA analysis on the 56
clean-EOS-halt CONTROL docs cached during Job 2 v3.

If control |d| is also ~1.3 → CL-25 is artifact (long-generation state shift regardless of halt).
If control |d| << 0.5 → CL-25 stands as halt-specific.

Plus: random-LR control for CL-23. Train LR direction on (random doc, late-gen pos) vs (different
random doc, late-gen pos) — i.e., a halt-irrelevant LR direction — and compute LOPO cosine across
14 random splits. Establishes baseline LOPO cosine for "any small-N LR direction" so we know how
much of CL-23's 0.38-0.44 is halt-specific vs noise-floor.
"""

import json, argparse, os
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# Repo root resolution. Original host paths (HPC allocation under
# /projects/<group>/<user>/ and laptop checkout) replaced with a portable
# default that resolves relative to this file. Set HALT_REPO_ROOT to override.
REPO_ROOT = Path(os.environ.get("HALT_REPO_ROOT", Path(__file__).resolve().parent.parent))
ACT_DIR = REPO_ROOT / "results/activations"
TRIG_DIR = REPO_ROOT / "results/p1_trigger_v2"
TEST_LAYERS = [16, 20, 24]


def categorize_doc(d):
    if d.get("label") == "positive_runaway" or d.get("is_real_runaway") is True:
        return "positive"
    if d.get("hit_max_new_tokens") and d.get("tokens_emitted", 0) >= 12000:
        return "positive"
    if d.get("stop_reason") == "eos" and d.get("tokens_emitted", 0) >= 50:
        return "control"
    return "unknown"


def load_all_cached_docs():
    out = []
    for p in sorted(TRIG_DIR.glob("*.json")):
        if p.name.startswith("_") or p.name == "failure_taxonomy.json" or p.name.endswith(".tokens.pt"):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        doc_id = d.get("doc") or d.get("doc_id") or p.stem
        if not (ACT_DIR / doc_id).is_dir():
            continue
        out.append({
            "doc_id": doc_id,
            "category": categorize_doc(d),
            "tokens_emitted": d.get("tokens_emitted"),
        })
    return out


def load_doc_residuals(doc_id, target_layers):
    doc_dir = ACT_DIR / doc_id
    hs = torch.load(doc_dir / "hidden_states.pt", weights_only=False)
    meta = json.loads((doc_dir / "meta.json").read_text())
    layer_indices = torch.load(doc_dir / "layer_indices.pt", weights_only=False)
    if hasattr(layer_indices, "tolist"):
        layer_indices = layer_indices.tolist()
    cached_layers = list(layer_indices)
    if hs.ndim == 3 and hs.shape[0] < hs.shape[1]:
        hs = hs.permute(1, 0, 2).contiguous()
    if not all(L in cached_layers for L in target_layers):
        return None
    aligned_pos = [cached_layers.index(L) for L in target_layers]
    return hs[:, aligned_pos, :].float(), meta


def run_pca_control(out_dir):
    """PCA pre-vs-post-crossover on CONTROL docs — does the |d| effect persist if there's no halt failure?"""
    print("=== PCA on CONTROL docs (the CL-25 critic-flagged missing control) ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = load_all_cached_docs()
    controls = [d for d in docs if d["category"] == "control"]
    positives = [d for d in docs if d["category"] == "positive"]
    print(f"docs: {len(controls)} controls, {len(positives)} positives")

    results_by_layer = {}
    for L_idx, L in enumerate(TEST_LAYERS):
        print(f"\n--- L{L} ---")
        per_doc_control = []
        per_doc_positive = []
        for category, doc_list, target in [("control", controls, per_doc_control), ("positive", positives, per_doc_positive)]:
            for d in doc_list:
                loaded = load_doc_residuals(d["doc_id"], TEST_LAYERS)
                if loaded is None: continue
                hs, meta = loaded
                prompt_len = meta.get("prompt_len", 0)
                T = hs.shape[0]
                if T - prompt_len < 800: continue
                pre = hs[prompt_len+50:prompt_len+500, L_idx, :].numpy()
                post = hs[prompt_len+700:T, L_idx, :].numpy()
                if len(pre) < 50 or len(post) < 50: continue
                stacked = np.concatenate([pre, post])
                pca = PCA(n_components=1)
                pca.fit(stacked)
                pc1 = pca.components_[0]
                pc1 = pc1 / (np.linalg.norm(pc1) + 1e-9)
                pre_proj = pre @ pc1
                post_proj = post @ pc1
                cohen_d = (post_proj.mean() - pre_proj.mean()) / (np.std(np.concatenate([pre_proj, post_proj])) + 1e-9)
                target.append({
                    "doc_id": d["doc_id"], "category": category,
                    "cohen_d": float(cohen_d),
                    "explained_variance": float(pca.explained_variance_ratio_[0]),
                })

        # Aggregate
        ctl_ds = [abs(d["cohen_d"]) for d in per_doc_control]
        pos_ds = [abs(d["cohen_d"]) for d in per_doc_positive]
        results_by_layer[f"L{L}"] = {
            "n_controls": len(per_doc_control),
            "n_positives": len(per_doc_positive),
            "control_abs_cohen_d_median": float(np.median(ctl_ds)) if ctl_ds else None,
            "control_abs_cohen_d_mean": float(np.mean(ctl_ds)) if ctl_ds else None,
            "control_abs_cohen_d_max": float(np.max(ctl_ds)) if ctl_ds else None,
            "positive_abs_cohen_d_median": float(np.median(pos_ds)) if pos_ds else None,
            "positive_abs_cohen_d_mean": float(np.mean(pos_ds)) if pos_ds else None,
            "ratio_pos_over_ctl_median": (float(np.median(pos_ds)) / float(np.median(ctl_ds))) if ctl_ds and pos_ds else None,
            "per_doc_control": per_doc_control,
            "per_doc_positive": per_doc_positive,
        }
        if ctl_ds and pos_ds:
            print(f"  controls (N={len(ctl_ds)}): |d| median {np.median(ctl_ds):.3f}, mean {np.mean(ctl_ds):.3f}, max {np.max(ctl_ds):.3f}")
            print(f"  positives (N={len(pos_ds)}): |d| median {np.median(pos_ds):.3f}, mean {np.mean(pos_ds):.3f}")
            print(f"  ratio pos/ctl (median): {np.median(pos_ds) / np.median(ctl_ds):.2f}x")

    (out_dir / "_summary.json").write_text(json.dumps({"experiment": "PCA_control_vs_positive", "results_by_layer": results_by_layer}, indent=2))
    print(f"\n[ok] wrote {out_dir / '_summary.json'}")

    print("\n=== HEADLINE ===")
    for L, r in results_by_layer.items():
        ratio = r.get("ratio_pos_over_ctl_median")
        if ratio is not None:
            if ratio > 2.0:
                verdict = "HALT-SPECIFIC effect: CL-25 stands"
            elif ratio > 1.3:
                verdict = "Partial: positives have somewhat larger effect than controls"
            else:
                verdict = "CL-25 ARTIFACT: controls show same magnitude effect — long-gen state shift, not halt-specific"
            print(f"  {L}: pos/ctl ratio {ratio:.2f}x → {verdict}")


def run_random_lr_baseline(out_dir):
    """Random-LR LOPO baseline for CL-23: how stable is ANY small-N LR direction trained on residual streams?"""
    print("\n=== Random-LR LOPO baseline (CL-23 critic-flagged missing control) ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = load_all_cached_docs()
    controls = [d for d in docs if d["category"] == "control"]
    if len(controls) < 14:
        print(f"INSUFFICIENT controls: only {len(controls)}")
        return

    rng = np.random.default_rng(0)
    # Pick 14 random control docs as "synthetic positive" group + 30 different controls as "synthetic negative"
    shuffled = list(range(len(controls)))
    rng.shuffle(shuffled)
    syn_pos_ids = [controls[i]["doc_id"] for i in shuffled[:14]]
    syn_neg_ids = [controls[i]["doc_id"] for i in shuffled[14:44]]
    print(f"synthetic 'positive' group: 14 controls, 'negative' group: {len(syn_neg_ids)} controls")

    results = {}
    for L_idx, L in enumerate(TEST_LAYERS):
        directions = {}
        for holdout in syn_pos_ids:
            X_pos, X_neg = [], []
            for did in syn_pos_ids:
                if did == holdout: continue
                loaded = load_doc_residuals(did, TEST_LAYERS)
                if loaded is None: continue
                hs, meta = loaded
                T = hs.shape[0]; prompt_len = meta.get("prompt_len", 0)
                if T - prompt_len < 500: continue
                # Use last 500 as "positive class" (same protocol as CL-23/CL-25 P5b training)
                pool = hs[max(prompt_len+500, T-500):T, L_idx, :].numpy()
                idx = rng.choice(len(pool), min(200, len(pool)), replace=False)
                X_pos.append(pool[idx])
            for did in syn_neg_ids:
                loaded = load_doc_residuals(did, TEST_LAYERS)
                if loaded is None: continue
                hs, meta = loaded
                T = hs.shape[0]; prompt_len = meta.get("prompt_len", 0)
                if T - prompt_len < 20: continue
                pool = hs[prompt_len:T, L_idx, :].numpy()
                idx = rng.choice(len(pool), min(200, len(pool)), replace=False)
                X_neg.append(pool[idx])
            if not X_pos or not X_neg: continue
            X_pos = np.concatenate(X_pos); X_neg = np.concatenate(X_neg)
            n_min = min(len(X_pos), len(X_neg))
            X_pos = X_pos[rng.choice(len(X_pos), n_min, replace=False)]
            X_neg = X_neg[rng.choice(len(X_neg), n_min, replace=False)]
            X = np.concatenate([X_neg, X_pos]); y = np.concatenate([np.zeros(n_min), np.ones(n_min)])
            clf = LogisticRegression(max_iter=500).fit(X, y)
            direction = clf.coef_[0] / (np.linalg.norm(clf.coef_[0]) + 1e-9)
            directions[holdout] = direction

        ids = list(directions.keys())
        cosines = []
        for i, a in enumerate(ids):
            for j, b in enumerate(ids):
                if i < j:
                    cos = float(directions[a] @ directions[b])
                    cosines.append(cos)
        results[f"L{L}"] = {
            "n_directions": len(ids),
            "cos_mean": float(np.mean(cosines)) if cosines else None,
            "cos_min": float(np.min(cosines)) if cosines else None,
            "cos_max": float(np.max(cosines)) if cosines else None,
        }
        if cosines:
            print(f"  L{L}: random-controls LOPO cos mean={np.mean(cosines):.3f}, range=[{min(cosines):.3f}, {max(cosines):.3f}]")

    (out_dir / "_summary.json").write_text(json.dumps({"experiment": "random_lr_lopo_baseline", "results": results}, indent=2))
    print(f"\n[ok] wrote {out_dir / '_summary.json'}")
    print("\nCompare to CL-23 within-arxiv-positive LOPO mean 0.38-0.44:")
    print("If random-controls baseline is ~0.4 too → CL-23's 'within-class heterogeneity' is noise floor, not signal.")
    print("If random-controls baseline is << 0.2 → CL-23's signal is real (halt-specific stability differs from noise).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default output base. Original used cluster scratch (/scratch/<user>/...);
    # replaced with a repo-relative path. Override with --out-base.
    parser.add_argument("--out-base", type=str, default=str(REPO_ROOT / "results" / "p7b_mirror"))
    args = parser.parse_args()
    out_base = Path(args.out_base)
    run_pca_control(out_base / "PCA_control")
    run_random_lr_baseline(out_base / "random_lr_lopo")
    print("\n=== ALL DONE ===")
