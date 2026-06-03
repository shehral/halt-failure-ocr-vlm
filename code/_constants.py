"""
Canonical shared constants and helpers for the halt-failure OCR-VLM project.

This module exists to resolve the recurring path-resolution bug family (heuristic #36,
8 recurrences over 5 days). Each sibling script previously maintained its own DATA_DIRS
and `find_image_path` function, leading to schema drift and silent failures whenever
new data subdirs were added.

Import this module at the top of any script that needs to resolve image paths:

    from _constants import DATA_DIRS, find_image_path

Or:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from _constants import find_image_path

The single canonical function `find_image_path(doc_id, hint_path=None)` tries the
hint path first, then scans every known data subdir for the doc by id.

Adding a new data subdir: just append to `DATA_DIR_NAMES`. All consumers will pick it up.
"""

from pathlib import Path
from typing import Optional, Union

# Repo root resolution. The original project ran on two hosts (laptop + HPC cluster);
# absolute host paths have been replaced with placeholders for the public release.
# By default we resolve relative to this file, which works regardless of where the
# repo is checked out. Set HALT_REPO_ROOT to override.
import os

_ENV_ROOT = os.environ.get("HALT_REPO_ROOT")
_CANDIDATE_REPO_ROOTS = [
    Path(_ENV_ROOT) if _ENV_ROOT else None,
    # HPC allocation (placeholder): /projects/<group>/<user>/<repo>
    # Laptop checkout (placeholder): <repo-root>
    Path(__file__).resolve().parent.parent,
]
_CANDIDATE_REPO_ROOTS = [p for p in _CANDIDATE_REPO_ROOTS if p is not None]
REPO_ROOT = next((p for p in _CANDIDATE_REPO_ROOTS if p.exists()), _CANDIDATE_REPO_ROOTS[-1])

# All subdirs under data/ that hold images. APPEND new ones here when adding sub-corpora.
DATA_DIR_NAMES = (
    "public_corpus",
    "supplementary_corpus",
    "long_form_corpus",
    "synthetic_stress",
    "synthetic_dense",
    "real_loopy",
    "synthetic",
)

DATA_DIRS = tuple(REPO_ROOT / "data" / name for name in DATA_DIR_NAMES)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def find_image_path(doc_id: str, hint_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Locate the image file for a doc.

    Resolution order:
      1. If hint_path is a relative or absolute path that exists, use it.
      2. If hint_path is given but doesn't exist, take its basename and scan DATA_DIRS.
      3. If hint_path is None or unusable, scan DATA_DIRS for <doc_id>.<ext> across IMAGE_EXTENSIONS.

    Returns Path on success, None if no candidate exists on disk.
    """
    # Step 1: explicit hint that resolves
    if hint_path:
        for candidate in (Path(hint_path), REPO_ROOT / hint_path, REPO_ROOT / "data" / hint_path):
            try:
                if candidate.exists():
                    return candidate
            except (OSError, ValueError):
                continue
        # Step 2: hint failed; try basename in each data dir
        try:
            name = Path(hint_path).name
            for data_dir in DATA_DIRS:
                c = data_dir / "images" / name
                if c.exists():
                    return c
        except (OSError, ValueError):
            pass

    # Step 3: doc-id-based scan
    if doc_id:
        for ext in IMAGE_EXTENSIONS:
            for data_dir in DATA_DIRS:
                c = data_dir / "images" / f"{doc_id}{ext}"
                if c.exists():
                    return c
    return None


# Canonical layer indices for the Nanonets-OCR2-3B (Qwen2.5-VL-3B-Instruct base) decoder.
# These 10 layers were chosen at Job 2 v3 cache time as the kept set; any consumer reading
# `results/activations/*/hidden_states.pt` must align against this exact list.
# Duplicated in 6+ scripts (p4, p5, p5b, p6, p7, p7b, p8) before this consolidation.
KEPT_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 35)

# Canonical "test layers" — the mid-decoder L16/L20/L24 cluster that CL-21 showed are equivalent
# probe targets (within 0.015 AUC of each other across the 22-doc workshop set). Use these
# when an analysis needs to span the "best probe region" for the halt direction.
TEST_LAYERS = (16, 20, 24)

# Model identifiers / revisions. Pinned in CLAUDE.md as the canonical revision for reproducibility.
NANONETS_MODEL_ID = "nanonets/Nanonets-OCR2-3B"
NANONETS_REVISION = "c3886ff00bb037ce7da24988c9eafaf1fe2bed72"
QWEN_BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Hidden size and architectural constants (Qwen2.5-VL-3B family)
HIDDEN_SIZE = 2048
NUM_DECODER_LAYERS = 36
NUM_ATTENTION_HEADS = 16

# Generation constants used across diagnostic scripts
MAX_NEW_TOKENS_RUNAWAY = 12000  # production-like max for full-runaway detection
MAX_NEW_TOKENS_DIAGNOSTIC = 4000  # diagnostic runs (H-E patching, etc.) — faster, still cap-hits

# Late-gen threshold for "positive class" in position-independent labeling (P5b convention)
LATE_GEN_THRESHOLD = 500  # gen-positions >= this are confidently in-loop for cap-hit docs


__all__ = [
    "REPO_ROOT", "DATA_DIRS", "DATA_DIR_NAMES", "IMAGE_EXTENSIONS", "find_image_path",
    "KEPT_LAYERS", "TEST_LAYERS",
    "NANONETS_MODEL_ID", "NANONETS_REVISION", "QWEN_BASE_MODEL_ID",
    "HIDDEN_SIZE", "NUM_DECODER_LAYERS", "NUM_ATTENTION_HEADS",
    "MAX_NEW_TOKENS_RUNAWAY", "MAX_NEW_TOKENS_DIAGNOSTIC",
    "LATE_GEN_THRESHOLD",
]
