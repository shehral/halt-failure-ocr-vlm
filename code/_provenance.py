"""_provenance — write provenance.json AT results-creation time.

`★ Pedagogical aside`: provenance.json is the per-results-dir reproducibility
receipt. Today's project state is that it's filled in REACTIVELY (after results
already exist) by `backfill_provenance.py`, which means:

  - `args: {}`                  (irrecoverable — argparse state is gone)
  - `script: "unknown"`         (irrecoverable — argv[0] is gone)
  - `started_at/finished_at`:   filesystem mtime (lossy, off by minutes-hours)
  - `attn_impl: "unknown"`      (irrecoverable from disk)
  - `_backfilled: True`         (signals "this is reconstructed")

This module exists so going-forward scripts write a FULL provenance.json at
result-creation time, capturing argv + git-HEAD + ISO-timestamp + project
defaults. Tooling-downstream then trusts `_backfilled == False / absent`
as a "real" provenance row and `_backfilled == True` as a degraded one.

USAGE — drop this at the top of any script after creating the results dir:

    from _provenance import write_provenance
    out_dir = ROOT / "results" / "my_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(out_dir, seed=42)

For scripts that already write a provenance.json with extra experiment-specific
fields (e.g. p2_h2h_runner.py's `docs` + `conditions` arrays), pass those via
the `extra=` kwarg. They get merged into the canonical schema.

ACCEPTANCE — heuristic #74 (call write_provenance once per results dir at
creation time). Pre-commit hook should grep for "from _provenance import" in
any new script under `code/` that creates a `results/<dir>/`.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


# Pinned project defaults (kept in sync with CLAUDE.md "Environment" section).
# If you bump these (e.g. new model revision after refit), the change should
# be intentional and surface in a commit message. Heuristic #51 (preregistration)
# applies here too: the defaults are part of the experimental contract.
MODEL_REVISION_DEFAULT = "c3886ff00bb037ce7da24988c9eafaf1fe2bed72"
DTYPE_DEFAULT = "bfloat16"
ATTN_IMPL_DEFAULT = "eager"
MODEL_ID_DEFAULT = "nanonets/Nanonets-OCR2-3B"

# Schema version. Bump on any non-additive change to provenance.json keys
# (e.g. renaming `script` → `script_path`). Additive changes (new optional keys)
# do not require a version bump.
PROVENANCE_VERSION = 1


def _git_head() -> str:
    """Capture `git rev-parse HEAD` from the repo containing this module.

    Falls back to "no-git" if the working tree isn't a git repo (e.g. when the
    script is run from an exported sandbox) — that's a soft-fail, NOT a hard
    error. We don't want provenance-writing to crash an HPC job.
    """
    module_dir = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=module_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "no-git"


def _capture_argv() -> tuple[str, list[str]]:
    """Capture (script_name, args_list) from sys.argv.

    script_name = basename of argv[0] (or "unknown" if argv is empty, which
    happens in some test harnesses).
    args_list = argv[1:] (the actual argparse-able arguments).
    """
    if not sys.argv:
        return "unknown", []
    argv0 = sys.argv[0] or ""
    script = os.path.basename(argv0) if argv0 else "unknown"
    return script, list(sys.argv[1:])


def write_provenance(
    results_dir: Path,
    seed: Optional[int] = None,
    model_revision: str = MODEL_REVISION_DEFAULT,
    dtype: str = DTYPE_DEFAULT,
    attn_impl: str = ATTN_IMPL_DEFAULT,
    started_at: Optional[str] = None,
    extra: Optional[dict] = None,
) -> Path:
    """Write provenance.json at results-creation time. Call ONCE per results dir.

    Args:
        results_dir: Path to the results subdir. Created (with parents) if absent.
        seed: Random seed for the experiment. None if not applicable.
        model_revision: Pinned HF revision. Defaults to the CLAUDE.md-pinned hash.
        dtype: Working dtype (e.g. "bfloat16"). Project rule: never int4/int8.
        attn_impl: "eager" (diagnostic) or "sdpa" (routine). See CLAUDE.md.
        started_at: ISO8601 start timestamp. Defaults to datetime.now().isoformat().
        extra: Arbitrary additional fields to merge into the provenance dict.
            Use for experiment-specific data (e.g. docs list, conditions, slurm_id).

    Returns:
        Path to the written provenance.json (useful for sanity-asserts in callers).

    Side effects:
        - Creates `results_dir` if it doesn't exist (parents=True).
        - Writes `results_dir / "provenance.json"` with indent=2.
        - Does NOT touch any other file. Idempotent on the same argv (last
          writer wins; if you call it twice with different `extra` payloads,
          you'll lose the first).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    script, args = _capture_argv()

    provenance: dict[str, Any] = {
        "_provenance_version": PROVENANCE_VERSION,
        "model_id": MODEL_ID_DEFAULT,
        "model_revision": model_revision,
        "dtype": dtype,
        "attn_impl": attn_impl,
        "seed": seed,
        "git_commit": _git_head(),
        "started_at": started_at if started_at is not None else dt.datetime.now().isoformat(),
        "script": script,
        "args": args,
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "results_dir": str(results_dir),
    }

    # Merge extra fields LAST so callers can override defaults if they really
    # need to (e.g. overriding `started_at` after the fact). The override-vs-
    # preserve trade-off: we prefer caller-explicit over module-default.
    if extra:
        provenance.update(extra)

    out_path = results_dir / "provenance.json"
    out_path.write_text(json.dumps(provenance, indent=2, default=str))
    return out_path


__all__ = [
    "write_provenance",
    "PROVENANCE_VERSION",
    "MODEL_REVISION_DEFAULT",
    "DTYPE_DEFAULT",
    "ATTN_IMPL_DEFAULT",
    "MODEL_ID_DEFAULT",
]
