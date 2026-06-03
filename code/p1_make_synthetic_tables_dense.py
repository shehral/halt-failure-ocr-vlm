"""Phase 1.1 tier-2 — denser tables to push past clean-halt behavior.

If the base parametric set in p1_make_synthetic_tables.py doesn't reliably trigger
the bug, this generator escalates: more rows (200, 400), more columns (8), longer
cell content, smaller fonts, narrower row spacing, slight noise. Goal: get the
model into the regime the literature documents as still-failing (see
docs/references.md: "cannot trigger the bug").

Outputs to data/synthetic_dense/ so the base set stays intact.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Repo root resolution. The original project ran on two hosts (laptop + HPC cluster);
# absolute host paths have been replaced for the public release. We resolve relative
# to this file, which works regardless of where the repo is checked out. Set
# HALT_REPO_ROOT to override.
import os
ROOT = Path(os.environ.get("HALT_REPO_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "data" / "synthetic_dense"
OUT.mkdir(parents=True, exist_ok=True)

N_BUCKETS = [100, 200, 400]
COLS_CYCLE = [6, 8]
DOCS_PER_BUCKET = 2

# Longer, more variable cell strings to load the model harder.
WORDS = [
    "Aurora", "Basalt", "Cobalt", "Drift", "Ember", "Fjord", "Glyph", "Halo",
    "Iris", "Jade", "Karst", "Lumen", "Mica", "Nimbus", "Onyx", "Pyrite",
    "Quartz", "Reef", "Sable", "Talus", "Umbra", "Vellum", "Wisp", "Xeric",
]
ADJ = ["small", "narrow", "deep", "shallow", "thick", "thin", "soft", "hard"]
UNITS = ["mg", "kg", "lb", "USD", "EUR", "pcs", "%", "m³", "km", "ppm"]


def cell_text(rng: random.Random) -> str:
    style = rng.choice(["adj_noun_num", "code", "money", "phrase", "ratio"])
    if style == "adj_noun_num":
        return f"{rng.choice(ADJ)} {rng.choice(WORDS).lower()} {rng.randint(1, 999)}"
    if style == "code":
        return f"{rng.choice(WORDS)[:3].upper()}-{rng.randint(1000, 99999)}/{rng.choice('ABCDE')}"
    if style == "money":
        return f"${rng.randint(100, 999_999):,}.{rng.randint(0, 99):02d}"
    if style == "ratio":
        return f"{rng.randint(1, 99)}/{rng.randint(1, 99)} {rng.choice(UNITS)}"
    return f"{rng.choice(WORDS)} {rng.randint(1, 999)} {rng.choice(UNITS)}"


def render_table(path: Path, n_rows: int, n_cols: int, seed: int) -> dict:
    rng = random.Random(seed)
    cell_w, cell_h = 130, 22   # narrower rows, tighter
    pad_x, pad_y = 20, 20
    header_h = 24
    W = pad_x * 2 + cell_w * n_cols
    H = pad_y * 2 + header_h + cell_h * n_rows + 4

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 11)
        bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        bold = font

    header = [f"Field {i+1}" for i in range(n_cols)]
    for c, h in enumerate(header):
        x = pad_x + c * cell_w
        y = pad_y
        draw.rectangle([x, y, x + cell_w, y + header_h], outline="black", fill=(230, 230, 230))
        draw.text((x + 5, y + 5), h, fill="black", font=bold)

    cells: list[list[str]] = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            x = pad_x + c * cell_w
            y = pad_y + header_h + r * cell_h
            draw.rectangle([x, y, x + cell_w, y + cell_h], outline="black")
            txt = cell_text(rng)
            row.append(txt)
            draw.text((x + 4, y + 4), txt[:18], fill="black", font=font)
        cells.append(row)

    img.save(path, format="PNG")
    img_bytes = path.read_bytes()
    return {
        "doc_id": path.stem,
        "path": str(path.relative_to(ROOT)),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "seed": seed,
        "width": W,
        "height": H,
        "sha256": hashlib.sha256(img_bytes).hexdigest()[:16],
        "header": header,
        "cells": cells,
        "tier": "dense",
    }


def main() -> int:
    manifest: list[dict] = []
    started = time.time()
    for n in N_BUCKETS:
        for k in range(DOCS_PER_BUCKET):
            cols = COLS_CYCLE[k % len(COLS_CYCLE)]
            seed = n * 1000 + k + 700_000
            name = f"dense_N{n:03d}_C{cols:02d}_s{seed}.png"
            entry = render_table(OUT / name, n_rows=n, n_cols=cols, seed=seed)
            manifest.append(entry)
            print(f"[dense] {name}  N={n} cols={cols}  {entry['width']}x{entry['height']}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = "no-git"
    (OUT / "provenance.json").write_text(json.dumps({
        "script": "code/p1_make_synthetic_tables_dense.py",
        "started_at": started,
        "finished_at": time.time(),
        "n_buckets": N_BUCKETS,
        "docs_per_bucket": DOCS_PER_BUCKET,
        "cols_cycle": COLS_CYCLE,
        "n_docs": len(manifest),
        "git_commit": git,
    }, indent=2))
    print(f"[dense] wrote {len(manifest)} dense tables to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
