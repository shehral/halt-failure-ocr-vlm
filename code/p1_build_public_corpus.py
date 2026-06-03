"""Phase 1.1 revised — build a public-corpus trigger set.

Pulls samples from cached HuggingFace datasets + the LoopyLeaderboard adversarial
fixtures, normalizes them to PNG on disk, and emits a unified manifest. All
sources are public and citable.

Sources:
  - lmms-lab/DocVQA (DocVQA config, validation) — business docs w/ GT answers
  - nielsr/funsd (train)                       — forms w/ word + bbox annotations
  - mychen76/invoices-and-receipts_ocr_v2 (valid) — invoices w/ parsed_data
  - data/real_loopy/                            — LoopyLeaderboard adversarial fixtures

Output: data/public_corpus/images/<doc_id>.png + data/public_corpus/manifest.json

The manifest is shaped to be a drop-in replacement for data/synthetic/manifest.json
so p1_trigger_pass_v2.py can consume it via --set real_loopy / a new --set public.

Run:
    .venv/bin/python code/p1_build_public_corpus.py
    .venv/bin/python code/p1_build_public_corpus.py --n-docvqa 60 --n-funsd 30 --n-invoices 30 --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from datasets import load_dataset
from PIL import Image

# Repo root resolution. The original project ran on two hosts (laptop + HPC cluster);
# absolute host paths have been replaced for the public release. We resolve relative
# to this file, which works regardless of where the repo is checked out. Set
# HALT_REPO_ROOT to override.
import os
ROOT = Path(os.environ.get("HALT_REPO_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "data" / "public_corpus"
IMG_DIR = OUT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def _save_png(img: Image.Image, path: Path, max_side: int | None = None) -> tuple[int, int, str]:
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    img.save(path, format="PNG", optimize=False)
    return img.size[0], img.size[1], hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def pull_docvqa(n: int, seed: int) -> list[dict]:
    """Sample n images from DocVQA validation. Each image has one question; we
    record the question + GT answer but our prompt is OCR-the-whole-doc."""
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    rng = random.Random(seed)
    # Sample unique images: many questions per image, dedupe by docId+page.
    seen = set()
    picks: list[int] = []
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    for i in idxs:
        row = ds[i]
        key = (row["ucsf_document_id"], row["ucsf_document_page_no"])
        if key in seen:
            continue
        seen.add(key)
        picks.append(i)
        if len(picks) >= n:
            break
    entries: list[dict] = []
    for j, i in enumerate(picks):
        row = ds[i]
        doc_id = f"docvqa_{row['ucsf_document_id']}_p{row['ucsf_document_page_no']}"
        path = IMG_DIR / f"{doc_id}.png"
        w, h, sha = _save_png(row["image"], path, max_side=1600)
        entries.append({
            "doc_id": doc_id,
            "path": str(path.relative_to(ROOT)),
            "source": "lmms-lab/DocVQA",
            "source_idx": int(i),
            "category": "business_doc",
            "width": w, "height": h, "sha256": sha,
            "questions_present": [row["question"]],
            "answers_present": list(row["answers"]),
            "n_rows": 0, "n_cols": 0,   # not a labeled table
        })
        print(f"  [docvqa] {j+1}/{n} {doc_id} ({w}x{h})", flush=True)
    return entries


def pull_funsd(n: int, seed: int) -> list[dict]:
    ds = load_dataset("nielsr/funsd", split="train")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    entries: list[dict] = []
    for j, i in enumerate(idxs[:n]):
        row = ds[i]
        doc_id = f"funsd_{row['id']:>03}_{i:03d}"
        path = IMG_DIR / f"{doc_id}.png"
        w, h, sha = _save_png(row["image"], path, max_side=1600)
        entries.append({
            "doc_id": doc_id,
            "path": str(path.relative_to(ROOT)),
            "source": "nielsr/funsd",
            "source_idx": int(i),
            "category": "form",
            "width": w, "height": h, "sha256": sha,
            "n_words_gt": len(row["words"]),
            "n_rows": 0, "n_cols": 0,
        })
        print(f"  [funsd] {j+1}/{n} {doc_id} ({w}x{h}, {len(row['words'])} words GT)", flush=True)
    return entries


def pull_invoices(n: int, seed: int) -> list[dict]:
    ds = load_dataset("mychen76/invoices-and-receipts_ocr_v2", split="valid")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    entries: list[dict] = []
    for j, i in enumerate(idxs[:n]):
        row = ds[i]
        doc_id = f"invoice_{i:04d}"
        path = IMG_DIR / f"{doc_id}.png"
        w, h, sha = _save_png(row["image"], path, max_side=1600)
        # parsed_data is a stringified JSON-of-JSON; keep raw for ground-truth audits later.
        entries.append({
            "doc_id": doc_id,
            "path": str(path.relative_to(ROOT)),
            "source": "mychen76/invoices-and-receipts_ocr_v2",
            "source_idx": int(i),
            "category": "invoice",
            "width": w, "height": h, "sha256": sha,
            "parsed_data_preview": row["parsed_data"][:300],
            "n_rows": 0, "n_cols": 0,
        })
        print(f"  [invoices] {j+1}/{n} {doc_id} ({w}x{h})", flush=True)
    return entries


def pull_loopy() -> list[dict]:
    """Anything in data/real_loopy/ that has an existing image file."""
    src_dir = ROOT / "data" / "real_loopy"
    if not src_dir.exists():
        return []
    src_manifest = json.loads((src_dir / "manifest.json").read_text())
    entries: list[dict] = []
    for m in src_manifest:
        src_path = ROOT / m["path"]
        if not src_path.exists():
            continue
        doc_id = m["doc_id"]
        # Copy normalised into images/
        img = Image.open(src_path)
        out_path = IMG_DIR / f"{doc_id}.png"
        w, h, sha = _save_png(img, out_path, max_side=1600)
        entries.append({
            "doc_id": doc_id,
            "path": str(out_path.relative_to(ROOT)),
            "source": "LoopyLeaderboard testing repo",
            "source_idx": 0,
            "category": m.get("condition", "loopy_adversarial"),
            "width": w, "height": h, "sha256": sha,
            "n_rows": m.get("n_rows", 0), "n_cols": m.get("n_cols", 0),
            "notes": m.get("notes", ""),
        })
        print(f"  [loopy] {doc_id} ({w}x{h})", flush=True)
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docvqa", type=int, default=60)
    ap.add_argument("--n-funsd", type=int, default=30)
    ap.add_argument("--n-invoices", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    started = time.time()
    all_entries: list[dict] = []

    print(f"[pub] sampling DocVQA  (n={args.n_docvqa}, seed={args.seed})")
    all_entries.extend(pull_docvqa(args.n_docvqa, args.seed))
    print(f"[pub] sampling FUNSD  (n={args.n_funsd}, seed={args.seed})")
    all_entries.extend(pull_funsd(args.n_funsd, args.seed))
    print(f"[pub] sampling Invoices  (n={args.n_invoices}, seed={args.seed})")
    all_entries.extend(pull_invoices(args.n_invoices, args.seed))
    print(f"[pub] folding LoopyLeaderboard fixtures")
    all_entries.extend(pull_loopy())

    (OUT / "manifest.json").write_text(json.dumps(all_entries, indent=2))
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = "no-git"
    by_cat = {}
    for e in all_entries:
        by_cat[e["category"]] = by_cat.get(e["category"], 0) + 1
    (OUT / "provenance.json").write_text(json.dumps({
        "script": "code/p1_build_public_corpus.py",
        "started_at": started, "finished_at": time.time(),
        "seed": args.seed,
        "args": {"n_docvqa": args.n_docvqa, "n_funsd": args.n_funsd, "n_invoices": args.n_invoices},
        "n_docs": len(all_entries),
        "by_category": by_cat,
        "git_commit": git,
        "sources": [
            "lmms-lab/DocVQA (validation, DocVQA config)",
            "nielsr/funsd (train)",
            "mychen76/invoices-and-receipts_ocr_v2 (valid)",
            "LoopyLeaderboard testing repo (github.com/99991/testing)",
        ],
        "notes": "Skipped: darentang/sroie (broken in datasets ≥ 4.x), Teklia/IAM-line (single-line handwriting; will halt instantly and serve only as control).",
    }, indent=2))
    print(f"\n[pub] wrote {len(all_entries)} docs to {OUT}")
    print(f"[pub] by category: {by_cat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
