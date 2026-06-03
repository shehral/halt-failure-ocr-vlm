"""Driver: run the fix-prototype test as separate subprocesses per (doc, condition)
to avoid cumulative MPS memory pressure. Smaller max_new_tokens (4000) to keep
each run comfortably within memory limits.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POSITIVES = ["docvqa_kshm0227_p6", "docvqa_srgb0228_p2", "docvqa_pgjw0227_p5"]
MAX_NEW = 4000


def main() -> int:
    results = []
    for doc in POSITIVES:
        for use_monitor in (0, 1):
            label = "with-monitor" if use_monitor else "BASELINE"
            print(f"\n=== {doc}  {label} ===", flush=True)
            t0 = time.time()
            out = subprocess.run(
                [str(ROOT / ".venv" / "bin" / "python"),
                 str(ROOT / "fix" / "test_halt_monitor.py"),
                 doc, str(use_monitor), str(MAX_NEW)],
                cwd=ROOT, capture_output=True, text=True, timeout=3600,
            )
            dt = time.time() - t0
            tail = "\n".join(out.stdout.splitlines()[-12:])
            print(tail)
            if out.returncode != 0:
                print(f"  STDERR tail:\n{out.stderr[-800:]}")
            json_path = ROOT / "results" / "p2_pilot" / f"fix_test_{doc}_{label.replace('-', '_')}.json"
            if json_path.exists():
                results.append(json.loads(json_path.read_text()))
            print(f"  total elapsed: {dt:.0f}s")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"{'doc':<32}  {'baseline':>12}  {'with-monitor':>14}  {'reduction':>11}")
    for doc in POSITIVES:
        bl = next((r for r in results if r["doc_id"] == doc and not r["use_monitor"]), None)
        wm = next((r for r in results if r["doc_id"] == doc and r["use_monitor"]), None)
        if bl is None or wm is None:
            print(f"{doc:<32}  {'missing':>12}")
            continue
        red = (1 - wm["tokens_emitted"] / max(bl["tokens_emitted"], 1)) * 100
        bl_str = f"{bl['tokens_emitted']} ({bl['stop_reason']})"
        wm_str = f"{wm['tokens_emitted']} ({wm['stop_reason']})"
        print(f"{doc:<32}  {bl_str:>12}  {wm_str:>14}  {red:>10.1f}%")

    out_path = ROOT / "results" / "p2_pilot" / "fix_test_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
