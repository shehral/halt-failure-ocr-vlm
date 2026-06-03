"""q1_combined_probe — Q1 v2 combined design: global widened probe + per-class probes.

Produces CL-42 (per-class halt-direction probe). Tests two claims in one run:
  - Global (broader): is there a "structural-loop token vs content token" direction (any class)?
  - Per-class (finer): does each loop class have its OWN direction? Are they pairwise distinct?

Both fall under the structural-content-decoupling hypothesis. Per heuristic #67 cleanest route,
running BOTH probe modes in the same script has zero compute overhead (same residual loads,
same tokenizer pass) and gives a richer evidence matrix.

`★ Pedagogical aside`: per-class probes are more rigorous than a single global probe because
they let us measure HOW MUCH structural representation is class-specific vs class-general.
If per-class directions are mutually orthogonal (low pairwise cosine), the model has fine-
grained structural representations. If they collapse to one shared direction, only "any
structure vs content" is decodable. Either result is publishable.

Schema (input layout this script reads):
  - results/p1_cuda_supplementary/<doc>.txt           full generated text
  - results/p1_cuda_supplementary/<doc>.tokens.pt     tokens [gen_len]
  - results/p1_cuda_n40/<doc>.txt                     fallback for non-supplementary docs
  - results/p1_trigger_v2/<doc>.json                  has `failure_class` field per doc
  - results/activations/<doc>/hidden_states.pt        residuals [n_layers, T, hidden_dim]
  - results/activations/<doc>/meta.json               has `prompt_len`

Usage:
    python code/q1_combined_probe.py \\
        --cache results/activations \\
        --text-dirs results/p1_cuda_supplementary results/p1_cuda_n40 \\
        --trigger-dir results/p1_trigger_v2 \\
        --pos-list code/pos_doc_list.txt \\
        --halt-dir fix/ \\
        --layers 0 4 8 12 16 20 24 28 32 35 \\
        --out results/q1_combined/

Heuristics applied: #51 pre-registration before run, #67 cleanest-route combined design.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# Multi-class structural-loop patterns. Each entry: (class_name, regex pattern).
# Patterns ordered by specificity — most specific first; first-match wins per token.
LOOP_PATTERNS = [
    # HTML phantom row — 2+ consecutive empty <td></td> rows
    ("html_phantom_row",
     re.compile(r"(?:<tr>\s*(?:<td>\s*</td>\s*)+</tr>\s*){2,}", re.IGNORECASE)),
    # Tag-spam patterns (watermark / page-number / br / signature etc., 2+ consecutive)
    ("tag_spam_structural",
     re.compile(r"(?:<(watermark|page_number|signature|br)>?[^<]*</?\1>?\s*){2,}", re.IGNORECASE)),
    # LaTeX brace loop — 3+ consecutive empty/whitespace { } pairs
    ("latex_brace_loop",
     re.compile(r"(?:\\?[a-zA-Z]*\{\s*\}\s*){3,}")),
    # LaTeX paren-bracket / Greek-bracket — 3+ consecutive bracket sequences
    ("latex_bracket_loop",
     re.compile(r"(?:[\(\[\{]\s*[\)\]\}]\s*){3,}")),
    # LaTeX backslash-escape loop — 3+ consecutive lone backslashes
    ("latex_backslash_loop",
     re.compile(r"(?:\\\\?\s*){4,}")),
    # LaTeX math-cmd loop — 3+ consecutive math commands like \alpha \beta \gamma...
    ("latex_math_cmd_loop",
     re.compile(r"(?:\\[a-zA-Z]+\s*){5,}")),
    # LaTeX ampersand-cellsep — 3+ consecutive empty cells in tabular
    ("latex_ampersand_loop",
     re.compile(r"(?:\&\s*){4,}")),
    # Count-up bullet — 3+ consecutive lines starting with "* N" or "- N" with incrementing N
    ("count_up_bullet",
     re.compile(r"(?:[\*\-]\s*\d+\s*[\n\r]+\s*){3,}")),
    # Filled-cell repeat — same long content token repeated
    ("filled_cell_repeat",
     re.compile(r"(?:([^\s<>]{10,})\s*\1\s*){3,}")),
    # Checkbox spam — 3+ consecutive checkbox symbols
    ("checkbox_spam",
     re.compile(r"(?:[☐☑☒]\s*){4,}")),
    # Bare-word repeat — same word repeated 5+ times
    ("bare_word_repeat",
     re.compile(r"(?:\b([a-zA-Z]{3,})\b[\s,]*){5,}")),
]


def find_loop_spans(text: str) -> list[tuple[int, int, str]]:
    """Find all structural-loop spans across all classes. Returns (start, end, class_name)."""
    spans = []
    for class_name, pattern in LOOP_PATTERNS:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end(), class_name))
    # Sort by start; if multiple classes match overlapping spans, first-match-wins by class order.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    return spans


def find_text_file(doc_id: str, text_dirs: list[Path]) -> Path | None:
    for d in text_dirs:
        p = d / f"{doc_id}.txt"
        if p.exists():
            return p
    return None


def find_tokens_file(doc_id: str, text_dirs: list[Path]) -> Path | None:
    for d in text_dirs:
        p = d / f"{doc_id}.tokens.pt"
        if p.exists():
            return p
    return None


def build_token_offsets(text: str, tokens: torch.Tensor, tokenizer) -> list[tuple[int, int]]:
    """Return per-token (char_start, char_end). Same as v1: decode token-by-token, accumulate."""
    offsets = []
    cur = 0
    for tok_id in tokens.tolist():
        s = tokenizer.decode([tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not s:
            offsets.append((cur, cur))
            continue
        idx = text.find(s, cur)
        if idx == -1 or idx > cur + 5:
            offsets.append((cur, cur + len(s)))
            cur += len(s)
        else:
            offsets.append((idx, idx + len(s)))
            cur = idx + len(s)
    return offsets


def label_tokens(token_offsets: list[tuple[int, int]], loop_spans: list[tuple[int, int, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-token labels.

    Returns:
      labels: [n_tokens] int8 — 0=content, 1=loop (any class)
      class_ids: [n_tokens] int8 — -1=content, 0..K-1=loop class index per LOOP_PATTERNS order
    """
    n = len(token_offsets)
    labels = np.zeros(n, dtype=np.int8)
    class_ids = -np.ones(n, dtype=np.int8)
    class_name_to_idx = {cn: i for i, (cn, _) in enumerate(LOOP_PATTERNS)}
    for i, (tok_s, tok_e) in enumerate(token_offsets):
        for span_s, span_e, class_name in loop_spans:
            if span_s <= tok_s and tok_e <= span_e:
                labels[i] = 1
                class_ids[i] = class_name_to_idx[class_name]
                break  # first-match-wins per token
    return labels, class_ids


def train_probe(X: np.ndarray, y: np.ndarray, doc_arr: np.ndarray) -> tuple[float, list[float], torch.Tensor | None]:
    """Train probe with LODO AUC + return fold means + direction trained on all data.

    Returns: (mean_auc, fold_aucs, direction)
    """
    if X.shape[0] < 20 or y.sum() < 5 or (y == 0).sum() < 5:
        return float("nan"), [], None
    unique_docs = sorted(set(doc_arr))
    fold_aucs = []
    for held_out in unique_docs:
        train_mask = doc_arr != held_out
        test_mask = doc_arr == held_out
        if y[train_mask].sum() < 5 or (y[train_mask] == 0).sum() < 5:
            continue
        if len(set(y[test_mask])) < 2:
            continue
        sc = StandardScaler().fit(X[train_mask])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[train_mask]), y[train_mask])
        test_pred = clf.predict_proba(sc.transform(X[test_mask]))[:, 1]
        fold_aucs.append(roc_auc_score(y[test_mask], test_pred))
    if not fold_aucs:
        return float("nan"), [], None
    # Direction trained on full data.
    sc_full = StandardScaler().fit(X)
    clf_full = LogisticRegression(max_iter=2000, C=1.0).fit(sc_full.transform(X), y)
    direction = torch.from_numpy(clf_full.coef_[0]).float()
    direction = direction / direction.norm()
    return float(np.mean(fold_aucs)), fold_aucs, direction


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--text-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--trigger-dir", type=Path, default=Path("results/p1_trigger_v2"))
    ap.add_argument("--pos-list", type=Path, required=True)
    ap.add_argument("--halt-dir", type=Path, default=Path("fix"))
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 28, 32, 35])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-tokens-per-class", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Heuristic #71 (added 2026-05-17 23:08): compute nodes lack internet for
    # huggingface.co API calls. transformers 5.x's `model_info()` mistral-regex
    # patch check triggers a connection timeout. Force local-only.
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("nanonets/Nanonets-OCR2-3B", local_files_only=True)

    doc_ids = [line.strip() for line in args.pos_list.read_text().splitlines() if line.strip()]
    if args.smoke:
        doc_ids = doc_ids[:3]
    print(f"[q1-combined] processing {len(doc_ids)} positive docs across layers {args.layers}")

    # Accumulators per layer.
    per_layer_data: dict[int, dict] = {layer: {"X": [], "y": [], "class_id": [], "doc": []} for layer in args.layers}
    per_doc_stats: list[dict] = []

    for doc_id in doc_ids:
        text_path = find_text_file(doc_id, args.text_dirs)
        tokens_path = find_tokens_file(doc_id, args.text_dirs)
        hs_path = args.cache / doc_id / "hidden_states.pt"
        li_path = args.cache / doc_id / "layer_indices.pt"
        meta_path = args.cache / doc_id / "meta.json"
        if not all(p and p.exists() for p in [text_path, tokens_path, hs_path, li_path, meta_path]):
            print(f"[q1] SKIP {doc_id} — missing files")
            continue
        text = text_path.read_text(errors="ignore")
        tokens = torch.load(tokens_path, weights_only=True, map_location="cpu")
        meta = json.loads(meta_path.read_text())
        prompt_len = meta["prompt_len"]

        loop_spans = find_loop_spans(text)
        if not loop_spans:
            print(f"[q1] SKIP {doc_id} — no loop spans found across any class")
            continue
        token_offsets = build_token_offsets(text, tokens, tokenizer)
        labels, class_ids = label_tokens(token_offsets, loop_spans)
        n_loop = int(labels.sum())
        n_content = int((labels == 0).sum())
        if n_loop < args.min_tokens_per_class or n_content < args.min_tokens_per_class:
            print(f"[q1] SKIP {doc_id} — insufficient pos/neg: {n_loop} loop / {n_content} content")
            continue
        class_breakdown = {}
        for c_idx in range(len(LOOP_PATTERNS)):
            n = int((class_ids == c_idx).sum())
            if n > 0:
                class_breakdown[LOOP_PATTERNS[c_idx][0]] = n
        per_doc_stats.append({
            "doc_id": doc_id, "n_tokens_gen": len(tokens), "n_loop_tokens": n_loop,
            "n_content_tokens": n_content, "loop_to_content_ratio": round(n_loop / max(1, n_content), 3),
            "n_loop_spans": len(loop_spans), "class_breakdown": class_breakdown,
        })

        # Load residuals.
        hs = torch.load(hs_path, weights_only=True, map_location="cpu")
        li = torch.load(li_path, weights_only=True, map_location="cpu")
        layer_idx_map = {int(l): i for i, l in enumerate(li.tolist() if hasattr(li, "tolist") else li)}
        gen_end = prompt_len + len(tokens)
        for layer in args.layers:
            if layer not in layer_idx_map:
                continue
            gen_resid = hs[layer_idx_map[layer], prompt_len:gen_end, :].float().numpy()
            # Off-by-one safety: hidden_states may be saved at T-1 positions
            # (no next-token-prediction for the last token). Clip the loop.
            n_iter = min(len(tokens), gen_resid.shape[0])
            for tok_i in range(n_iter):
                per_layer_data[layer]["X"].append(gen_resid[tok_i])
                per_layer_data[layer]["y"].append(int(labels[tok_i]))
                per_layer_data[layer]["class_id"].append(int(class_ids[tok_i]))
                per_layer_data[layer]["doc"].append(doc_id)
        print(f"[q1] {doc_id}: n_loop={n_loop} n_content={n_content} classes={list(class_breakdown.keys())}")

    # Per-layer: global probe + per-class probes + cosine matrix.
    per_layer_results = []
    for layer in args.layers:
        ld = per_layer_data[layer]
        if not ld["X"]:
            continue
        X = np.stack(ld["X"])
        y = np.array(ld["y"])
        c = np.array(ld["class_id"])
        doc_arr = np.array(ld["doc"])

        # Global probe: loop (any class) vs content.
        global_auc, global_folds, global_dir = train_probe(X, y, doc_arr)
        # Per-class probes: each class with >=3 docs.
        class_directions = {}
        per_class_aucs = {}
        for c_idx, (class_name, _) in enumerate(LOOP_PATTERNS):
            class_mask = c == c_idx
            # Construct class-specific labels: class_idx token = 1, content tokens = 0.
            y_class = np.zeros(len(X), dtype=int)
            y_class[class_mask] = 1
            content_mask = c == -1
            train_keep = class_mask | content_mask  # exclude OTHER classes
            if class_mask.sum() < 50:
                continue
            unique_docs_in_class = len(set(doc_arr[class_mask]))
            if unique_docs_in_class < 3:
                continue
            auc, _, direction = train_probe(X[train_keep], y_class[train_keep], doc_arr[train_keep])
            if direction is not None:
                class_directions[class_name] = direction
                per_class_aucs[class_name] = round(auc, 4)

        # Halt-direction cosine.
        halt_path = args.halt_dir / f"halt_direction_L{layer}.pt"
        halt_cosine_global = None
        halt_cosine_per_class = {}
        if halt_path.exists():
            halt_dir = torch.load(halt_path, weights_only=True, map_location="cpu").float()
            halt_dir = halt_dir / halt_dir.norm()
            if global_dir is not None:
                halt_cosine_global = round(float((global_dir * halt_dir).sum()), 4)
            for cn, cd in class_directions.items():
                halt_cosine_per_class[cn] = round(float((cd * halt_dir).sum()), 4)

        # Pairwise per-class cosines.
        class_names = sorted(class_directions.keys())
        pairwise_cosine = {}
        for i, cn1 in enumerate(class_names):
            for cn2 in class_names[i+1:]:
                cos = float((class_directions[cn1] * class_directions[cn2]).sum())
                pairwise_cosine[f"{cn1}__{cn2}"] = round(cos, 4)

        # Save directions.
        if global_dir is not None:
            torch.save(global_dir, args.out / f"global_direction_L{layer}.pt")
        for cn, cd in class_directions.items():
            torch.save(cd, args.out / f"class_{cn}_direction_L{layer}.pt")

        result = {
            "layer": layer,
            "n_records": len(X),
            "global_auc": round(global_auc, 4) if not np.isnan(global_auc) else None,
            "global_folds": len(global_folds),
            "global_halt_cosine": halt_cosine_global,
            "per_class_aucs": per_class_aucs,
            "per_class_halt_cosines": halt_cosine_per_class,
            "n_classes_with_probe": len(class_directions),
            "pairwise_cosine_max": max(pairwise_cosine.values()) if pairwise_cosine else None,
            "pairwise_cosine_mean": float(np.mean(list(pairwise_cosine.values()))) if pairwise_cosine else None,
            "pairwise_cosines": pairwise_cosine,
        }
        per_layer_results.append(result)
        print(f"[L{layer}] global_AUC={global_auc:.3f} halt_cos={halt_cosine_global} | per-class: {per_class_aucs}")

    # Verdict.
    n_layers_pass_global = sum(1 for r in per_layer_results
                                if r["global_auc"] is not None and r["global_auc"] >= 0.80
                                and r["global_halt_cosine"] is not None and abs(r["global_halt_cosine"]) <= 0.35)
    n_layers_pass_per_class = sum(1 for r in per_layer_results
                                   if r["n_classes_with_probe"] >= 2
                                   and r["pairwise_cosine_max"] is not None and r["pairwise_cosine_max"] <= 0.50)
    out = {
        "_pre_registration": "docs/q1_phantom_row_preregistration.md (v1 narrow) + this v2 widening as per heuristic #67",
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_docs_in_pool": len(per_doc_stats),
        "per_doc_stats": per_doc_stats,
        "per_layer_results": per_layer_results,
        "verdict": {
            "n_layers_pass_global_decoupling": n_layers_pass_global,
            "n_layers_pass_per_class_distinctness": n_layers_pass_per_class,
            "global_decoupling_passes": n_layers_pass_global >= 2,
            "per_class_distinctness_passes": n_layers_pass_per_class >= 2,
            "overall": "PASS_BOTH" if (n_layers_pass_global >= 2 and n_layers_pass_per_class >= 2)
                       else "PASS_GLOBAL_ONLY" if n_layers_pass_global >= 2
                       else "PASS_PER_CLASS_ONLY" if n_layers_pass_per_class >= 2
                       else "FAIL",
        },
    }
    (args.out / "_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\n[q1] VERDICT: {out['verdict']['overall']}")
    print(f"[q1] {n_layers_pass_global} layers pass global decoupling; {n_layers_pass_per_class} pass per-class distinctness")
    print(f"[q1] wrote {args.out / '_summary.json'}")
    return 0 if "PASS" in out["verdict"]["overall"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
