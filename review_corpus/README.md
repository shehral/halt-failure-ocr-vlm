# Review corpus — cap-hit failures by surface class

This directory holds a browsable corpus of **cap-hit** generations: decodes from
`nanonets/Nanonets-OCR2-3B` that ran to the `max_new_tokens` cap instead of emitting
the end-of-sequence token. These are the halt-failure positives studied in the paper,
sorted by the **surface class** of the runaway output (phantom HTML rows, filled-cell
repeats, LaTeX structure/command loops, dash/punctuation runs, etc.).

The documents are drawn from public sources, including the public **DocVQA** dataset
(UCSF industry documents), FUNSD, DocLayNet, public SEC EDGAR filings, and public arXiv
tables. The displayed model outputs are derived from these public documents.

## viewer.html

`viewer.html` is a self-contained, single-file HTML browser for the corpus. Open it
directly in a web browser (no server, no build step):

```
open review_corpus/viewer.html
```

What it gives you:

- **Sidebar** of every cap-hit `doc_id`, grouped by surface class, with per-class counts.
- **Class filter** checkboxes and a `doc_id` search box (`/` to focus search).
- **Keyboard navigation:** `j` / `k` (or arrow keys) to step through documents.
- For each document: the model's full runaway output, the repeated tail, a per-class
  hint describing the failure pattern, and a collapsible block of run metadata
  (token count, stop reason, seed, dtype, attention impl, model id + revision, source
  dataset). All output text is inlined in the HTML.

### Note on page images

The viewer references the original page image for each document via the relative path
`01_documents/<doc_id>.png`. The model-output text, class labels, tails, and metadata
all render from `viewer.html` alone. To also see the source page images, place the
matching page PNGs in a sibling `01_documents/` directory next to `viewer.html`; if that
directory is absent the image pane simply shows a placeholder and everything else works.

## Surface classes

| # | Class | N docs |
|---|---|---:|
| 01 | html phantom row | 7 |
| 02 | filled cell repeat | 21 |
| 03 | latex math cmd loop | 2 |
| 04 | latex structure loop | 7 |
| 05 | latex ampersand loop | 3 |
| 06 | latex backslash loop | 4 |
| 08 | count up bullet | 1 |
| 09 | tag spam structural | 1 |
| 11 | bare word repeat | 1 |
| 12 | dash run | 4 |
| 13 | punct run | 4 |
| 14 | constant value repeat | 1 |
| 99 | unclassified | 1 |
| -- | boundary (EOS at cap) | 2 |
| -- | false positive (legit long content) | 1 |

**Total: 60 unique cap-hit documents across 15 surface classes** (57 clear
infinite-generation positives, 2 boundary cases that emitted EOS exactly at the cap,
and 1 confirmed false positive where the cap-hit was legitimate long content rather than
a loop).

## 00_INDEX.csv

`00_INDEX.csv` is one row per document — class, token count, label, and per-doc hints —
suitable for sorting or filtering outside the viewer.
