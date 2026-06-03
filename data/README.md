# Data — eval manifests and public-corpus provenance

This directory holds the **eval manifests** for the halt-failure OCR-VLM study and
documents how to fetch the underlying document images. Only manifests (JSON) are
checked in here — the images themselves are **not redistributed**, because they come
from public datasets that carry their own licenses. Fetch them from the original
sources using the manifest metadata below.

All paths and identifiers in these manifests have been sanitized for public release:
absolute cluster paths, HPC usernames, and scratch paths were replaced with
`<placeholder>` tokens consistent with the published Explorer skill. The public model
revision of `nanonets/Nanonets-OCR2-3B`, the public dataset names, and the claim IDs
(`CL-xx`) are intentionally preserved for reproducibility.

## Files

| File | What it is |
|---|---|
| `public_corpus/manifest.json` | The 123-doc public evaluation corpus. Each row carries `doc_id`, relative `path`, `source` (HF dataset id), `source_idx` (row index in that dataset), `category`, image `width`/`height`, a truncated `sha256`, and the DocVQA `questions_present`/`answers_present`. |
| `public_corpus/provenance.json` | Build provenance: dataset sources + splits, per-category counts, and the corpus-build git commit. |
| `p16_manifest_labeled.json` | Labeled manifest for the P16 component-resolved FCCT patch (CL-19/CL-20): 12 positive + 5 control docs with `label`/`trigger`/`src_tokens`. References `data/public_corpus/` and `data/supplementary_corpus/`. |
| `paper_n_manifest.json` | The 60-doc paper evaluation manifest (positives + controls) with per-doc `tokens_emitted`, `stop_reason`, and the heuristic loop-class label. |
| `supplementary_corpus/images/*.png` | 201 synthetic long-document renders (arXiv table pages, SEC-EDGAR-style filings, DocLayNet pages) that the P16 and paper_n cohorts depend on. **Regenerable** — see below. Checked in here because they are derived renders (not redistributed source datasets), seed-pinned, and small enough to bundle for one-shot reproducibility. |
| `supplementary_corpus/manifest.json` | Per-render metadata: `doc_id`, relative `path`, `source`/`source_dataset` (HF dataset id), `source_url`, `source_idx`, `category` (`arxiv_table` / `sec_financial` / `doclaynet_page`), `width`/`height`, `expected_token_count`, `max_new_tokens`, `produced_by`, `text_char_count`. |
| `supplementary_corpus/provenance.json` | Build provenance for the supplementary corpus: the `code/p1_curate_supplementary.py` invocation, seed, per-source counts, and timing. |
| `../review_corpus/00_INDEX.csv` | Provenance index for the manually reviewed positives: **56 confirmed infinite-generation positives across 12+ surface classes** (plus 1 false positive, 2 boundary EOS-at-cap, 1 synthetic control). Each row carries `doc_id`, `class`, `source_corpus` (HF dataset id), `tokens_emitted`, `max_new_tokens`, `label`, `production_patterns`, `empty_rows_count`, and a short `hint`. This is the "56 positives / 12 classes" provenance behind the paper's failure taxonomy. |

## Public dataset provenance

The corpus is assembled from four **public** sources (see `public_corpus/provenance.json`):

| Source (`source` field) | Split | Category | Count | How to fetch |
|---|---|---|---|---|
| `lmms-lab/DocVQA` | `validation` (DocVQA config) | `business_doc` | 60 | `datasets.load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")`, index by `source_idx`. |
| `nielsr/funsd` | `train` | `form` | 30 | `datasets.load_dataset("nielsr/funsd", split="train")`, index by `source_idx`. |
| `mychen76/invoices-and-receipts_ocr_v2` | `valid` | `invoice` | 30 | `datasets.load_dataset("mychen76/invoices-and-receipts_ocr_v2", split="valid")`, index by `source_idx`. |
| LoopyLeaderboard testing repo (`github.com/99991/testing`) | — | `sequential_numbers` / `repeated` / `real_receipt` | 3 | Clone the repo and use the named test images. |

The `p16` and `paper_n` manifests additionally reference `data/supplementary_corpus/`,
which is **not a fetched dataset** but synthetic long-document renders (arXiv table
pages, SEC-EDGAR-style filings, and DocLayNet pages rendered to wide page images). The
201 renders are bundled here so the repo reproduces one-shot, but they are also fully
**regenerable** from public HuggingFace sources via `code/p1_curate_supplementary.py`
(seed 0, `--sources arxiv,sec,doclaynet`, `--n-per-source 67`):

| Category (`category` field) | Public source(s) | Rendering |
|---|---|---|
| `arxiv_table` | `ccdv/arxiv-summarization` (`train`, streamed; falls back to `scientific_papers`/`arxiv`), filtered to articles containing a multi-row table marker | text rendered to PNG via Pillow ImageDraw, 1024 px wide, adaptive height ≤ 4096 px |
| `sec_financial` | `eloukas/edgar-corpus` (parquet-revision rescue at `refs/convert/parquet`; falls back to `JanosAudran/financial-reports-sec`, `nlpaueb/finer-139`, or a deterministic synthetic balance-sheet generator) | text rendered to PNG as above |
| `doclaynet_page` | `ds4sd/DocLayNet` (falls back to `ds4sd/DocLayNet-v1.1`, `cmarkea/doclaynet`, `IBM/DocLayNet`, `HuggingFaceM4/Docmatix`) | native page images re-encoded to PNG (no re-rendering) |

To regenerate from scratch:

```bash
python code/p1_curate_supplementary.py --sources arxiv,sec,doclaynet --n-per-source 67 --seed 0
```

The build is idempotent (existing renders are skipped) and writes
`data/supplementary_corpus/{manifest.json,provenance.json}` alongside the PNGs.
Because the upstream datasets stream in order, the exact doc set is reproducible at a
given seed; verify regenerated renders against the bundled `manifest.json` `doc_id`s.

## How to fetch the DocVQA (and other) images

```python
from datasets import load_dataset
from pathlib import Path
import json

manifest = json.load(open("data/public_corpus/manifest.json"))
out = Path("data/public_corpus/images"); out.mkdir(parents=True, exist_ok=True)

# DocVQA (validation split, DocVQA config)
docvqa = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
for row in manifest:
    if row["source"] != "lmms-lab/DocVQA":
        continue
    img = docvqa[row["source_idx"]]["image"]   # PIL.Image
    img.convert("RGB").save(out / Path(row["path"]).name)
```

Repeat with the FUNSD and invoice datasets for the `nielsr/funsd` and
`mychen76/invoices-and-receipts_ocr_v2` rows, indexing each by `source_idx`. Verify
each fetched image against the truncated `sha256` in the manifest to confirm you
pulled the same revision.

## Reproducibility pin

All generations/probes in `results/` were produced against
`nanonets/Nanonets-OCR2-3B` at the pinned public revision
`c3886ff00bb037ce7da24988c9eafaf1fe2bed72` (greedy decoding, bf16, seed 0). The
revision is recorded in every `results/**/provenance.json`.
