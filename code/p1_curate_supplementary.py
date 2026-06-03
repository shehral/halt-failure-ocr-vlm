"""Supplementary corpus curator (E-CE / Job 1c) -- corpus expansion for N >= 15.

Goal: pull ~200 fresh document images from 3 public sources targeting diverse
failure-class likelihoods (arXiv tables, SEC-EDGAR-style financial docs,
DocLayNet academic pages). This expands the corpus past the N >= 15 paper
target by adding ~67 docs per source, all of which feed into the greedy
generation / trigger pass (a single `model.generate` call per manifest row,
`max_new_tokens=12000`, `do_sample=False`; the trigger-sweep driver is not
bundled in this public repo).

Sources and rationale:
  1. arXiv tables (`ccdv/arxiv-summarization` filtered for table markers) --
     same source family as long_form_corpus, but filtered to docs that contain
     multi-row tables. These are likely to elicit empty-row or repeat failures.
     Filters on raw text for `|...|...|` markdown tables, `\\begin{tabular}` or
     `\\hline` LaTeX, and HTML `<table>` tags.

  2. SEC EDGAR-style financial documents -- primary path: `JanosAudran/financial-reports-sec`
     (filings_with_section text). Has financial tables with sub-totals, blank
     header rows, year-over-year columns. Fallbacks: `nlpaueb/finer-139`,
     synthetic SEC-balance-sheet generator.

  3. DocLayNet academic pages -- `ds4sd/DocLayNet` (or `cmarkea/doclaynet` or
     `IBM/DocLayNet` fallback). Native PNG/JPG pages with diverse layouts
     (academic, manual, magazine). For native-image sources we skip rendering
     and write the raw image to disk; for text-only sources we render via
     Pillow ImageDraw using the same pattern as `p1_curate_long_form.py`.

Rendering: Pillow ImageDraw text-on-canvas (same as `p1_curate_long_form.py`),
1024 px wide, adaptive height up to 4096 px. Same monospace font fallback
chain. For DocLayNet-style native-image sources we re-encode the raw bytes
into the canonical PNG output path (no re-rendering needed).

Filter constraint: text length in [5000, 30000] characters (or for image
sources, image height in [600, 4096] px) -- the production "trigger-rate
sweet spot" based on prior sweep evidence.

Manifest schema matches `data/public_corpus/manifest.json` (consumable by the
generation/trigger pass via its `--manifest` interface):
    {doc_id, path, source_dataset, source_url, source_idx, category,
     width, height, expected_token_count, produced_by, text_char_count?}

Idempotent: any doc whose output image already exists is skipped.
Honest failure: if a source 404s / auth-gates / mid-stream-errors, we log,
skip, and continue with the next source. The final manifest is non-empty as
long as at least one source survives.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import textwrap
from io import BytesIO
from pathlib import Path

# Repo root resolution. The original project ran on two hosts (laptop + HPC cluster);
# absolute host paths have been replaced for the public release. We resolve relative
# to this file, which works regardless of where the repo is checked out. Set
# HALT_REPO_ROOT to override.
import os
ROOT = Path(os.environ.get("HALT_REPO_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_OUT_DIR = ROOT / "data" / "supplementary_corpus"


# =================================================================
# Source 1: arXiv with table-marker filter
# =================================================================

_TABLE_MARKERS = [
    # LaTeX
    re.compile(r"\\begin\{tabular\}"),
    re.compile(r"\\hline"),
    re.compile(r"\\begin\{table\}"),
    # Markdown / HTML
    re.compile(r"\|[^\n]*\|[^\n]*\|"),  # |a|b|c| style row
    re.compile(r"<table\b", re.IGNORECASE),
    re.compile(r"<tr\b", re.IGNORECASE),
]


def _has_table(text: str) -> bool:
    """Cheap heuristic for whether the raw text contains a multi-row table."""
    for pat in _TABLE_MARKERS:
        if pat.search(text):
            return True
    return False


def load_arxiv_table_docs(
    n_docs: int,
    min_chars: int = 5_000,
    max_chars: int = 30_000,
    stream_cap: int = 20_000,
) -> list[dict]:
    """Pull n_docs arXiv articles containing >= 1 table marker."""
    from datasets import load_dataset

    out: list[dict] = []
    last_exc = None
    for ds_name, split, text_field in [
        ("ccdv/arxiv-summarization", "train", "article"),
        ("scientific_papers", "train", "article"),
    ]:
        try:
            kwargs = dict(split=split, streaming=True)
            if ds_name == "scientific_papers":
                kwargs["name"] = "arxiv"
            print(f"[arxiv-tables] streaming {ds_name} (split={split}); "
                  f"need={n_docs}, char-range=[{min_chars}, {max_chars}]")
            ds = load_dataset(ds_name, **kwargs)
            for i, ex in enumerate(ds):
                if i >= stream_cap:
                    break
                if len(out) >= n_docs:
                    break
                text = ex.get(text_field, "") or ""
                if not (min_chars <= len(text) <= max_chars):
                    continue
                if not _has_table(text):
                    continue
                out.append({
                    "doc_id": f"arxiv_table_{i:06d}",
                    "text": text,
                    "image_bytes": None,
                    "source_dataset": ds_name,
                    "source_url": f"https://huggingface.co/datasets/{ds_name}",
                    "source_idx": i,
                    "category": "arxiv_table",
                })
            if out:
                print(f"[arxiv-tables] retrieved {len(out)} from {ds_name}")
                return out[:n_docs]
        except Exception as exc:
            print(f"[arxiv-tables] {ds_name} FAILED: {type(exc).__name__}: {exc}")
            last_exc = exc
            continue

    if not out:
        print(f"[arxiv-tables] ALL sources failed; last_exc={last_exc}; returning empty list.")
    return out[:n_docs]


# =================================================================
# Source 2: SEC-EDGAR-style financial documents
# =================================================================

def load_sec_docs(
    n_docs: int,
    min_chars: int = 5_000,
    max_chars: int = 30_000,
    stream_cap: int = 20_000,
) -> list[dict]:
    """Pull n_docs SEC-style financial docs. Multiple HF fallbacks + synthetic."""
    from datasets import load_dataset

    out: list[dict] = []
    last_exc = None

    # Candidate datasets in order of preference. Each tuple is
    # (dataset, config, split, text_field, source_url_suffix).
    candidates = [
        ("JanosAudran/financial-reports-sec", "small_lite", "train", "sentence",
         "JanosAudran/financial-reports-sec"),
        ("JanosAudran/financial-reports-sec", "large_lite", "train", "sentence",
         "JanosAudran/financial-reports-sec"),
        ("eloukas/edgar-corpus", "year_2020", "train", "section_1",
         "eloukas/edgar-corpus"),
        ("eloukas/edgar-corpus", "full", "train", "section_1",
         "eloukas/edgar-corpus"),
        ("nlpaueb/finer-139", None, "train", "tokens",
         "nlpaueb/finer-139"),
    ]

    # FIRST try the parquet-revision rescue for eloukas/edgar-corpus.
    # HuggingFace deprecated `.py` loading scripts; many SEC datasets broke. But
    # HF auto-converts each dataset to a parquet revision at `refs/convert/parquet`
    # — this bypasses scripts entirely. Schema is `default` config with section_*
    # fields (real 10-K filing text).
    try:
        print(f"[sec] TRYING parquet-revision rescue: eloukas/edgar-corpus @ refs/convert/parquet")
        ds_pq = load_dataset(
            "eloukas/edgar-corpus", split="train", streaming=True,
            revision="refs/convert/parquet",
        )
        # Schema has section_1, section_1A, ... 15 sections. Concat into doc-shaped text.
        section_fields = [f"section_{i}" for i in (1, "1A", "1B", 2, 3, 4, 5, 6, 7, "7A", 8)]
        for i, ex in enumerate(ds_pq):
            if i >= stream_cap or len(out) >= n_docs:
                break
            chunks = [str(ex.get(f, "") or "") for f in section_fields]
            text = "\n\n".join(c for c in chunks if c.strip()).strip()
            if not (min_chars <= len(text) <= max_chars):
                continue
            out.append({
                "doc_id": f"sec_edgar_{ex.get('cik', i):>09s}_{i:06d}" if isinstance(ex.get('cik'), str) else f"sec_edgar_{i:06d}",
                "text": text,
                "image_bytes": None,
                "source_dataset": "eloukas/edgar-corpus",
                "source_url": "https://huggingface.co/datasets/eloukas/edgar-corpus",
                "source_idx": i,
                "category": "sec_financial",
            })
        if len(out) >= n_docs:
            print(f"[sec] parquet-rescue SUCCESS — retrieved {len(out)} from eloukas/edgar-corpus")
            return out[:n_docs]
        print(f"[sec] parquet-rescue partial ({len(out)}/{n_docs}); falling through to legacy candidates")
    except Exception as exc:
        print(f"[sec] parquet-rescue FAILED: {type(exc).__name__}: {str(exc)[:200]}")
        last_exc = exc

    for ds_name, config, split, text_field, url_suffix in candidates:
        try:
            kwargs = dict(split=split, streaming=True)
            if config is not None:
                kwargs["name"] = config
            print(f"[sec] streaming {ds_name} (config={config}, split={split}); "
                  f"need={n_docs - len(out)}")
            ds = load_dataset(ds_name, **kwargs)

            # For finer-139 (token-list), we have to join tokens into a string and
            # accumulate across rows until we hit the size window. We aggregate
            # in groups of N rows to form per-doc texts.
            if ds_name == "nlpaueb/finer-139":
                buf: list[str] = []
                buf_chars = 0
                doc_idx = 0
                for i, ex in enumerate(ds):
                    if i >= stream_cap or len(out) >= n_docs:
                        break
                    toks = ex.get(text_field, []) or []
                    s = " ".join(toks) if isinstance(toks, list) else str(toks)
                    buf.append(s)
                    buf_chars += len(s) + 1
                    if buf_chars >= min_chars:
                        text = "\n".join(buf)[:max_chars]
                        if len(text) >= min_chars:
                            out.append({
                                "doc_id": f"sec_finer_{doc_idx:06d}",
                                "text": text,
                                "image_bytes": None,
                                "source_dataset": ds_name,
                                "source_url": f"https://huggingface.co/datasets/{url_suffix}",
                                "source_idx": doc_idx,
                                "category": "sec_financial",
                            })
                            doc_idx += 1
                        buf, buf_chars = [], 0
            else:
                for i, ex in enumerate(ds):
                    if i >= stream_cap or len(out) >= n_docs:
                        break
                    text = ex.get(text_field, "") or ""
                    if isinstance(text, list):
                        text = "\n".join(str(x) for x in text)
                    if not (min_chars <= len(text) <= max_chars):
                        continue
                    out.append({
                        "doc_id": f"sec_{ds_name.split('/')[-1]}_{i:06d}",
                        "text": text,
                        "image_bytes": None,
                        "source_dataset": ds_name,
                        "source_url": f"https://huggingface.co/datasets/{url_suffix}",
                        "source_idx": i,
                        "category": "sec_financial",
                    })
            if len(out) >= n_docs:
                print(f"[sec] retrieved {len(out)} from {ds_name}")
                return out[:n_docs]
        except Exception as exc:
            print(f"[sec] {ds_name} ({config}) FAILED: {type(exc).__name__}: {exc}")
            last_exc = exc
            continue

    # If nothing worked, fall back to a synthetic SEC-balance-sheet generator.
    if len(out) < n_docs:
        n_missing = n_docs - len(out)
        print(f"[sec] falling back to SYNTHETIC SEC-balance-sheet generator "
              f"for the remaining {n_missing} doc(s); last_exc={last_exc}")
        out.extend(_synthesize_sec_docs(n_missing))

    return out[:n_docs]


def _synthesize_sec_docs(n_docs: int) -> list[dict]:
    """Last-resort synthesizer for SEC-balance-sheet-style documents."""
    import random

    rng = random.Random(0xBA1ABCE)
    out: list[dict] = []
    line_items = [
        "Cash and cash equivalents", "Short-term investments", "Accounts receivable",
        "Inventories", "Property and equipment, net", "Goodwill", "Intangible assets, net",
        "Other assets", "Total assets", "Accounts payable", "Accrued liabilities",
        "Deferred revenue", "Long-term debt", "Other long-term liabilities",
        "Total liabilities", "Common stock, $0.001 par value", "Additional paid-in capital",
        "Retained earnings", "Accumulated other comprehensive loss",
        "Total stockholders' equity", "Total liabilities and stockholders' equity",
    ]
    for idx in range(n_docs):
        ticker = f"TICK{idx:03d}"
        year = 2020 + (idx % 5)
        lines: list[str] = [
            f"UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
            f"FORM 10-K",
            f"ANNUAL REPORT FOR THE FISCAL YEAR ENDED December 31, {year}",
            f"{ticker} CORPORATION",
            "",
            "CONSOLIDATED BALANCE SHEETS",
            f"(In thousands, except share data)",
            f"{'Item':<48} {str(year):>14} {str(year-1):>14}",
            "-" * 78,
        ]
        for item in line_items:
            v_now = rng.randint(0, 9_999_999)
            v_prev = rng.randint(0, 9_999_999)
            # Add a few all-zero rows to elicit phantom-row failures
            if rng.random() < 0.05:
                v_now = 0
                v_prev = 0
            lines.append(f"{item:<48} ${v_now:>13,} ${v_prev:>13,}")
        # Repeat a notes section to push the text into the [5K, 30K] window.
        notes = [
            "",
            "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS",
            "",
            "1. Summary of Significant Accounting Policies",
            "",
            "The Company prepares its consolidated financial statements in accordance "
            "with United States generally accepted accounting principles. The "
            "preparation of financial statements in conformity with US GAAP requires "
            "management to make estimates and assumptions that affect the reported "
            "amounts of assets and liabilities and the disclosure of contingent assets "
            "and liabilities at the date of the financial statements and the reported "
            "amounts of revenues and expenses during the reporting period. Actual "
            "results could differ from those estimates.",
            "",
            "2. Revenue Recognition",
            "",
            "Revenue is recognized when control of the promised goods or services is "
            "transferred to the customer, in an amount that reflects the consideration "
            "that the Company expects to receive in exchange for those goods or "
            "services. Revenue is measured as the amount of consideration the Company "
            "expects to receive in exchange for the promised goods or services. ",
        ]
        body = "\n".join(lines + notes * 6)
        out.append({
            "doc_id": f"sec_synth_{idx:06d}",
            "text": body,
            "image_bytes": None,
            "source_dataset": "synthetic_sec_balance_sheet",
            "source_url": "local-synthesizer",
            "source_idx": idx,
            "category": "sec_financial",
        })
    return out


# =================================================================
# Source 3: DocLayNet (native PNG pages)
# =================================================================

def load_doclaynet_docs(
    n_docs: int,
    min_height: int = 600,
    max_height: int = 4096,
    stream_cap: int = 20_000,
) -> list[dict]:
    """Pull n_docs PNG pages from DocLayNet (or fallback layout sets)."""
    from datasets import load_dataset

    out: list[dict] = []
    last_exc = None

    # Image-field-name varies by dataset. We try a couple of fields per dataset.
    candidates = [
        ("ds4sd/DocLayNet", None, "train", ["image"], "ds4sd/DocLayNet"),
        ("ds4sd/DocLayNet-v1.1", None, "train", ["image"], "ds4sd/DocLayNet-v1.1"),
        ("cmarkea/doclaynet", None, "train", ["image", "page_image"],
         "cmarkea/doclaynet"),
        ("IBM/DocLayNet", None, "train", ["image"], "IBM/DocLayNet"),
        # PubLayNet-style fallback
        ("HuggingFaceM4/Docmatix", None, "train", ["images"],
         "HuggingFaceM4/Docmatix"),
    ]

    for ds_name, config, split, img_fields, url_suffix in candidates:
        try:
            kwargs = dict(split=split, streaming=True)
            if config is not None:
                kwargs["name"] = config
            print(f"[doclaynet] streaming {ds_name} (split={split}); need={n_docs - len(out)}")
            ds = load_dataset(ds_name, **kwargs)

            for i, ex in enumerate(ds):
                if i >= stream_cap or len(out) >= n_docs:
                    break

                # Find the image field
                img_obj = None
                for field in img_fields:
                    candidate = ex.get(field, None)
                    if candidate is not None:
                        img_obj = candidate
                        break
                if img_obj is None:
                    continue

                # Some datasets return list of images (Docmatix) -- pick first.
                if isinstance(img_obj, list):
                    if not img_obj:
                        continue
                    img_obj = img_obj[0]

                # img_obj is a PIL.Image or similar with .size = (w, h)
                try:
                    w = int(getattr(img_obj, "width", 0)) or int(img_obj.size[0])
                    h = int(getattr(img_obj, "height", 0)) or int(img_obj.size[1])
                except Exception:
                    continue
                if not (min_height <= h <= max_height):
                    continue
                if w < 200:
                    continue

                # Serialize to PNG bytes for later write.
                try:
                    buf = BytesIO()
                    img_rgb = img_obj.convert("RGB") if hasattr(img_obj, "convert") else img_obj
                    img_rgb.save(buf, format="PNG", optimize=True)
                    image_bytes = buf.getvalue()
                except Exception as exc:
                    print(f"[doclaynet] {ds_name} idx={i} serialize fail: {exc}")
                    continue

                out.append({
                    "doc_id": f"doclaynet_{ds_name.split('/')[-1]}_{i:06d}",
                    "text": None,
                    "image_bytes": image_bytes,
                    "image_width": w,
                    "image_height": h,
                    "source_dataset": ds_name,
                    "source_url": f"https://huggingface.co/datasets/{url_suffix}",
                    "source_idx": i,
                    "category": "doclaynet_page",
                })
            if len(out) >= n_docs:
                print(f"[doclaynet] retrieved {len(out)} from {ds_name}")
                return out[:n_docs]
        except Exception as exc:
            print(f"[doclaynet] {ds_name} FAILED: {type(exc).__name__}: {exc}")
            last_exc = exc
            continue

    if not out:
        print(f"[doclaynet] all sources failed; last_exc={last_exc}")
    return out[:n_docs]


# =================================================================
# Rendering -- text to single PNG (Pillow)
# =================================================================

def render_text_to_png(
    text: str,
    out_path: Path,
    width_px: int = 1024,
    margin_px: int = 48,
    font_size: int = 14,
    line_spacing: int = 4,
    max_height_px: int = 4096,
    bg: tuple = (255, 255, 255),
    fg: tuple = (16, 16, 16),
) -> tuple[int, int]:
    """Render plain text onto a single PNG; returns (width, height)."""
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for cand in [
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    ]:
        try:
            font = ImageFont.truetype(cand, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    chars_per_line = max(40, int((width_px - 2 * margin_px) / max(1, font_size * 0.6)))
    wrapped: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(raw_line, width=chars_per_line) or [""])

    line_h = font_size + line_spacing
    max_lines = (max_height_px - 2 * margin_px) // line_h
    if len(wrapped) > max_lines:
        wrapped = wrapped[: max_lines - 1] + ["[... truncated to fit image height ...]"]

    actual_height = 2 * margin_px + len(wrapped) * line_h
    img = Image.new("RGB", (width_px, actual_height), bg)
    draw = ImageDraw.Draw(img)
    y = margin_px
    for line in wrapped:
        draw.text((margin_px, y), line, font=font, fill=fg)
        y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)
    return (width_px, actual_height)


def write_native_image(image_bytes: bytes, out_path: Path) -> tuple[int, int]:
    """Persist native image bytes to PNG. Returns (width, height)."""
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(BytesIO(image_bytes))
    img = img.convert("RGB")
    img.save(out_path, format="PNG", optimize=True)
    return (img.width, img.height)


# =================================================================
# Token-count estimator (shared with p1_curate_long_form.py)
# =================================================================

_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(
            "nanonets/Nanonets-OCR2-3B",
            revision="c3886ff00bb037ce7da24988c9eafaf1fe2bed72",
        )
    return _TOKENIZER


def estimate_token_count(text: str) -> int:
    try:
        tok = _get_tokenizer()
        return int(len(tok.encode(text, add_special_tokens=False)))
    except Exception:
        # Rough fallback: ~4 chars/token.
        return max(0, len(text) // 4)


# =================================================================
# Main
# =================================================================

SOURCE_FNS = {
    "arxiv": load_arxiv_table_docs,
    "sec": load_sec_docs,
    "doclaynet": load_doclaynet_docs,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/supplementary_corpus", type=str,
                    help="Output directory (relative to repo root or absolute)")
    ap.add_argument("--n-per-source", default=67, type=int,
                    help="Target docs per source (~67 * 3 = ~200 total)")
    ap.add_argument(
        "--sources", default="arxiv,sec,doclaynet", type=str,
        help="Comma-separated source list (subset of arxiv,sec,doclaynet)"
    )
    ap.add_argument("--min-text-chars", default=5_000, type=int)
    ap.add_argument("--max-text-chars", default=30_000, type=int)
    ap.add_argument("--width-px", default=1024, type=int)
    ap.add_argument("--max-height-px", default=4096, type=int)
    ap.add_argument("--font-size", default=14, type=int)
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()

    started_at = time.time()

    sources_requested = [s.strip() for s in args.sources.split(",") if s.strip()]
    for s in sources_requested:
        if s not in SOURCE_FNS:
            print(f"[main] WARN: unknown source '{s}'; ignoring")
    sources = [s for s in sources_requested if s in SOURCE_FNS]
    if not sources:
        print("[main] ERROR: no valid sources requested.")
        return 2

    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else (ROOT / args.out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    existing: list[dict] = []
    seen_ids: set[str] = set()
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
            seen_ids = {m["doc_id"] for m in existing}
            print(f"[idempotent] manifest has {len(existing)} existing docs; "
                  f"will skip those.")
        except Exception:
            existing, seen_ids = [], set()

    per_source_summary: dict[str, dict] = {}
    new_entries: list[dict] = []

    for src in sources:
        print("")
        print(f"================ SOURCE: {src} ================")
        n_existing_this_src = sum(
            1 for m in existing if m.get("category", "").startswith(_category_prefix(src))
        )
        n_needed = max(0, args.n_per_source - n_existing_this_src)
        if n_needed == 0:
            print(f"[{src}] manifest already has {n_existing_this_src} >= n_per_source; skipping fetch.")
            per_source_summary[src] = {
                "needed": 0, "fetched": 0, "rendered": 0,
                "existing_at_start": n_existing_this_src,
            }
            continue

        # Over-pull factor: arxiv requires filter so over-pull more.
        over = 3 if src == "arxiv" else 2
        try:
            if src == "arxiv":
                raw_docs = load_arxiv_table_docs(
                    n_docs=n_needed * over,
                    min_chars=args.min_text_chars,
                    max_chars=args.max_text_chars,
                )
            elif src == "sec":
                raw_docs = load_sec_docs(
                    n_docs=n_needed * over,
                    min_chars=args.min_text_chars,
                    max_chars=args.max_text_chars,
                )
            elif src == "doclaynet":
                raw_docs = load_doclaynet_docs(
                    n_docs=n_needed * over,
                    max_height=args.max_height_px,
                )
            else:
                raw_docs = []
        except Exception as exc:
            print(f"[{src}] EXCEPTION during fetch: {type(exc).__name__}: {exc}; skipping source")
            per_source_summary[src] = {
                "needed": n_needed, "fetched": 0, "rendered": 0,
                "error": f"{type(exc).__name__}: {exc}",
                "existing_at_start": n_existing_this_src,
            }
            continue

        print(f"[{src}] retrieved {len(raw_docs)} raw candidates")

        rendered = 0
        total_chars = 0
        total_expected_tokens = 0
        for d in raw_docs:
            if rendered >= n_needed:
                break
            if d["doc_id"] in seen_ids:
                continue

            img_path = img_dir / f"{d['doc_id']}.png"
            try:
                if d.get("image_bytes") is not None:
                    w, h = write_native_image(d["image_bytes"], img_path)
                    char_count = 0
                    tok_count = -1
                else:
                    text = d.get("text") or ""
                    if not text:
                        continue
                    w, h = render_text_to_png(
                        text=text,
                        out_path=img_path,
                        width_px=args.width_px,
                        font_size=args.font_size,
                        max_height_px=args.max_height_px,
                    )
                    char_count = len(text)
                    tok_count = estimate_token_count(text)
            except Exception as exc:
                print(f"[{src}] {d['doc_id']}: render/write FAIL: {exc}; skipping")
                continue

            entry = {
                "doc_id": d["doc_id"],
                "path": str(img_path.relative_to(ROOT)),
                "source": d["source_dataset"],
                "source_dataset": d["source_dataset"],
                "source_url": d.get("source_url", ""),
                "source_idx": int(d.get("source_idx", -1)),
                "category": d.get("category", "supplementary"),
                "width": int(w),
                "height": int(h),
                "expected_token_count": int(tok_count),
                # Per-doc generation budget: 1.8x expected, clamped to [6000, 18000].
                # - Floor 6000: short legit docs still have room to fail at cap if buggy.
                # - Ceiling 18000: bounds compute (200 docs * 18K * 40 tok/s = ~25h max, fits budget).
                # FP-control: long legit arXiv docs get longer budgets without unbounded compute;
                # the length-relative filter in Job 1c step 3 catches any remaining FPs.
                "max_new_tokens": min(max(int(tok_count * 1.8), 6000), 18000),
                "produced_by": "code/p1_curate_supplementary.py",
                "text_char_count": int(char_count),
                # Provide manifest-schema parity with public_corpus (used by trigger
                # detectors): n_rows / n_cols default to 0 for free-form text.
                "n_rows": 0,
                "n_cols": 0,
            }
            new_entries.append(entry)
            seen_ids.add(d["doc_id"])
            rendered += 1
            total_chars += char_count
            total_expected_tokens += max(0, tok_count)
            if rendered <= 5 or rendered % 25 == 0:
                print(f"[{src}] {rendered}/{n_needed} {d['doc_id']}  "
                      f"{w}x{h}  chars={char_count}  ~tok={tok_count}")

        per_source_summary[src] = {
            "needed": n_needed,
            "fetched": len(raw_docs),
            "rendered": rendered,
            "total_chars": total_chars,
            "total_expected_tokens": total_expected_tokens,
            "existing_at_start": n_existing_this_src,
        }
        print(f"[{src}] DONE: rendered={rendered}, total_chars={total_chars}, "
              f"total_expected_tokens={total_expected_tokens}")

    combined = existing + new_entries
    manifest_path.write_text(json.dumps(combined, indent=2))
    print(f"\n[manifest] wrote {manifest_path} ({len(combined)} total docs, "
          f"{len(new_entries)} new this run)")

    finished_at = time.time()
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    provenance = {
        "script": "code/p1_curate_supplementary.py",
        "seed": args.seed,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_s": finished_at - started_at,
        "args": vars(args),
        "git_commit": git_commit,
        "sources_requested": sources_requested,
        "sources_used": sources,
        "per_source_summary": per_source_summary,
        "n_docs_total": len(combined),
        "n_docs_new_this_run": len(new_entries),
        "n_docs_skipped_idempotent": len(existing),
        "notes": (
            "Supplementary corpus expansion -- ~200 fresh docs across arXiv "
            "(table-filtered), SEC-EDGAR-style filings, and DocLayNet pages. "
            "Manifest schema matches data/public_corpus/manifest.json so the "
            "generation/trigger pass can consume it via --manifest. Native "
            "images are passed through; text-only sources are rendered via "
            "Pillow ImageDraw at width=1024, adaptive height up to 4096."
        ),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"[provenance] wrote {out_dir / 'provenance.json'}")

    # Final summary
    print("\n================ SUMMARY ================")
    print(f"{'source':<14} {'rendered':>9} {'chars':>12} {'~tokens':>12}")
    print("-" * 50)
    for src in sources:
        s = per_source_summary.get(src, {})
        print(f"{src:<14} {s.get('rendered', 0):>9} {s.get('total_chars', 0):>12} "
              f"{s.get('total_expected_tokens', 0):>12}")
    print("-" * 50)
    grand_rendered = sum(s.get("rendered", 0) for s in per_source_summary.values())
    grand_chars = sum(s.get("total_chars", 0) for s in per_source_summary.values())
    grand_tokens = sum(s.get("total_expected_tokens", 0) for s in per_source_summary.values())
    print(f"{'TOTAL':<14} {grand_rendered:>9} {grand_chars:>12} {grand_tokens:>12}")
    print(f"\nmanifest size: {len(combined)} docs")
    return 0


def _category_prefix(src: str) -> str:
    return {
        "arxiv": "arxiv_table",
        "sec": "sec_financial",
        "doclaynet": "doclaynet_page",
    }.get(src, "")


if __name__ == "__main__":
    sys.exit(main())
