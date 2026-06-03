"""Phase 1.1 — parametric table generator.

Renders markdown-style tables to PNG at row counts N in {5, 10, 20, 50, 100}, 4-6
columns, randomized cell content. Ground-truth row count is exact, so phantom-row
count for any generation will be unambiguous: generated_rows - N.

Writes:
    data/synthetic/manifest.json
    data/synthetic/table_N{nnn}_C{cc}_s{seed}.png
    data/synthetic/provenance.json

Run:
    .venv/bin/python code/p1_make_synthetic_tables.py
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
OUT = ROOT / "data" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)

# Buckets per CLAUDE.md / research-brief Phase 1.1.
N_BUCKETS = [5, 10, 20, 50, 100]
DOCS_PER_BUCKET = 3
COLS_CYCLE = [4, 5, 6]  # rotated across the docs in each bucket

# Cell lexicon: mixed alpha-numeric so the model sees real OCR-like load,
# not a single token.
WORDS = [
    "Aurora", "Basalt", "Cobalt", "Drift", "Ember", "Fjord", "Glyph", "Halo",
    "Iris", "Jade", "Karst", "Lumen", "Mica", "Nimbus", "Onyx", "Pyrite",
    "Quartz", "Reef", "Sable", "Talus", "Umbra", "Vellum", "Wisp", "Xeric",
    "Yarrow", "Zephyr", "Cinder", "Dune", "Echo", "Flint",
]
UNITS = ["mg", "kg", "lb", "USD", "EUR", "pcs", "%", "m³", "km"]


def cell_text(rng: random.Random) -> str:
    style = rng.choice(["word", "code", "money", "phrase"])
    if style == "word":
        return rng.choice(WORDS)
    if style == "code":
        return f"{rng.choice(WORDS)[:3].upper()}-{rng.randint(100, 9999)}"
    if style == "money":
        return f"${rng.randint(10, 99999):,}.{rng.randint(0, 99):02d}"
    return f"{rng.choice(WORDS)} {rng.randint(1, 999)} {rng.choice(UNITS)}"


def render_table(path: Path, n_rows: int, n_cols: int, seed: int) -> dict:
    rng = random.Random(seed)
    # Generous cell widths so dense content fits.
    cell_w, cell_h = 170, 34
    pad_x, pad_y = 28, 28
    header_h = cell_h
    W = pad_x * 2 + cell_w * n_cols
    H = pad_y * 2 + header_h + cell_h * n_rows + 8

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
        bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        bold = font

    header = [f"Field {i+1}" for i in range(n_cols)]
    for c, h in enumerate(header):
        x = pad_x + c * cell_w
        y = pad_y
        draw.rectangle([x, y, x + cell_w, y + header_h], outline="black", fill=(240, 240, 240))
        draw.text((x + 8, y + 7), h, fill="black", font=bold)

    cells: list[list[str]] = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            x = pad_x + c * cell_w
            y = pad_y + header_h + r * cell_h
            draw.rectangle([x, y, x + cell_w, y + cell_h], outline="black")
            txt = cell_text(rng)
            row.append(txt)
            # Truncate visually if extreme; rendered text is what the model sees.
            draw.text((x + 8, y + 8), txt[:24], fill="black", font=font)
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
    }


def main() -> int:
    manifest: list[dict] = []
    started = time.time()
    for n in N_BUCKETS:
        for k in range(DOCS_PER_BUCKET):
            cols = COLS_CYCLE[k % len(COLS_CYCLE)]
            seed = n * 1000 + k
            name = f"table_N{n:03d}_C{cols:02d}_s{seed:05d}.png"
            entry = render_table(OUT / name, n_rows=n, n_cols=cols, seed=seed)
            manifest.append(entry)
            print(f"[gen] {name}  N={n} cols={cols}  {entry['width']}x{entry['height']}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = "no-git"
    (OUT / "provenance.json").write_text(json.dumps({
        "script": "code/p1_make_synthetic_tables.py",
        "started_at": started,
        "finished_at": time.time(),
        "n_buckets": N_BUCKETS,
        "docs_per_bucket": DOCS_PER_BUCKET,
        "cols_cycle": COLS_CYCLE,
        "n_docs": len(manifest),
        "git_commit": git,
    }, indent=2))
    print(f"[gen] wrote {len(manifest)} tables and manifest to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
