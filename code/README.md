# Reproduction code — halt-failure (infinite generation) in OCR vision-language models

This directory contains the load-bearing experiment code behind the verified claims for the
halt-failure / infinite-generation study on `nanonets/Nanonets-OCR2-3B`. Each script below maps to
a specific claim (`CL-xx`) and the on-disk result file that the verification ledger confirmed.

The model under study is **`nanonets/Nanonets-OCR2-3B`** (a Qwen2.5-VL-3B-Instruct fine-tune,
36 decoder layers, hidden size 2048), pinned at public revision
`c3886ff00bb037ce7da24988c9eafaf1fe2bed72`. All trigger work uses the public **DocVQA** validation
set (plus FUNSD and public receipt fixtures). Greedy decoding (`do_sample=False`), bf16 weights.

## Mandatory loader fix

The checkpoint omits `lm_head.weight` and relies on `tie_word_embeddings`, but recent `transformers`
reads that flag off the outer config (False) instead of `text_config` (True), so `tie_weights()`
silently no-ops and generation collapses to `"!"`. Every loader here applies the fix:

```python
model.config.tie_word_embeddings = True
model.tie_weights()
assert model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr()
```

## Portability / paths

Original host-absolute paths (an HPC allocation and a laptop checkout) have been replaced with a
portable default that resolves relative to each file. To point the scripts at a checkout elsewhere,
set the `HALT_REPO_ROOT` environment variable. Cluster username, SLURM account, conda-env name, and
business-internal references have been replaced with `<placeholder>` tokens (see "Sanitization"
below). The scripts read cached residuals/generations under `results/` and images under `data/`;
those large directories are not bundled here. The EOS-margin demo, by contrast, ships with its
cached trajectory artifacts so it runs end-to-end without a GPU.

---

## Script → claim → result map

### Per-class halt-direction probe — CL-42
- **`q1_combined_probe.py`** — fits a global "structural-loop vs content" probe plus per-class
  logistic-regression halt directions at 10 layers (L0…L35), then measures pairwise cosines to test
  structural-content decoupling.
  - Produces: `results/q1_combined/per_layer_extracted.csv`, `results/q1_combined/_summary.json`.
  - Verified claim **CL-42**: per-class AUC 0.876–0.983 (filled@L8=0.8763 … filled@L24=0.9826);
    peaks latex@L8=0.9785, filled@L24=0.9826, bare@L20=0.9173; decoupling verdict `PASS_BOTH`
    (10/10 distinct directions, max pairwise cos 0.2349 @ L20). Also covers **CL-42-global**
    (global AUC peak L8=0.8571, below per-class everywhere).

### Logit-lens EOS-suppression locus — CL-16
- **`p3_logit_lens.py`** — classic logit lens: applies the final RMSNorm + `lm_head` to each cached
  layer's residual and reads the EOS logit per layer/position, aggregating positives vs controls.
  - Produces: `results/p3_logit_lens/_summary.json`.
  - Verified claim **CL-16**: EOS gap most negative at L24 (−0.419), recovering at L32 (+0.481) and
    L35 (+0.336) (N=4 positives + 9 controls).

### L0 sufficiency null — CL-49
- **`p6_l0_sufficiency.py`** — adds `alpha * g_L0_bare_word_repeat` to the last-position L0 residual
  at every generation step on clean-halt docs; four conditions (vanilla, class-matched,
  class-mismatched, random-norm-matched). Includes a `--smoke` end-to-end wiring check (heuristic
  #75). Imports `_constants.py`.
  - Produces: `results/p6_l0_sufficiency/_summary.json`.
  - Verified claim **CL-49**: injecting the L0 direction does NOT induce halt-failure
    (ΔC1 = 0.0 pp @ α=10 → FAIL against the pre-registered ≥40 pp bar). L0 is a readout, not a
    generator.

### Norm-scaled positive control — CL-36
- **`p_poscontrol_random_l24.py`** — patches a random unit direction at L24 scaled to 0.1×/10×/100×
  the residual norm, to establish the patching protocol's sensitivity floor (does any single-layer
  perturbation cause SOMETHING?). Imports `_constants.py` and `_provenance.py`.
  - Produces: `results/p_poscontrol/_summary.json`.
  - Verified claim **CL-36**: 0/4 docs escape the 12000 cap at any scale; output content visibly
    changes (so the hook fires) but token count does not — single-layer perturbation does not break
    the loop.

### Fine-tune PCA signature (control comparison) — CL-25
- **`p7b_control_comparison.py`** — re-runs the PCA pre-vs-post-crossover |Cohen's d| analysis on the
  clean-EOS control docs (the missing control for CL-25), plus a random-LR LOPO baseline for CL-23.
  - Produces: `results/p7b_mirror/PCA_control/_summary.json`, `.../random_lr_lopo/_summary.json`.
  - Verified claim **CL-25**: Nanonets pos/ctl |d| ratio L16=3.79, L20=2.74, L24=2.78 (N=22 pos /
    52 ctl) — the fine-tune-introduced signature is halt-specific, not a generic long-generation
    shift.

### B3 reverse-direction necessity null — CL-12
- **`test_b3_reverse.py`** — project-out forward hook that removes the trained L24 halt-direction
  component from the L24 residual at every position, then compares generation length vs baseline on
  clean-halt controls + a positive. Imports `fix/halt_monitor.py` (for `EOS_TOKEN_ID` and the
  direction path).
  - Produces: `results/p2_pilot/b3_<doc>_<mode>.json` (aggregated as `b3_reverse_direction_summary.json`).
  - Verified claim **CL-12**: 3/3 controls unchanged (<10%), positive 12000→12000; hook-fired
    confirmed (2/4 docs changed output content). Verdict **READOUT**, not halt circuit.

### Component-resolved FCCT patch (P16) — CL-19 / CL-20
- **`p3_he_patch.py`** — component-resolved (MHSA vs FFN vs block) causal patching at L16/L20/L24 via
  direct PyTorch forward hooks (FCCT-style), with norm-matched real vs control patch cells. Imports
  `_constants.py`.
  - Produces: `results/p16_component_resolved_short/_summary.json`.
  - Verified claim **CL-19/CL-20**: 0/45 real-patch cells nonzero (real-null); 9/134 controls nonzero
    (controls perturb more than the real direction). Verdict **FAIL** — reproduces the original
    0/58-real, 13/174-control real-null.

### Grey-noise image inpaint (vision sufficiency) — CL-11
- **`p3_image_inpaint.py`** — runs each positive twice: original image vs the image replaced by
  grey noise of identical dimensions; compares token counts + tail signatures.
  - Produces: `results/p3_image_inpaint/_summary.json` (+ per-doc JSON).
  - Verified claim **CL-11**: with grey noise the loop still cap-hits (12000) but collapses to 13–18
    tokens with `stop=eos` — the runaway loop is text-self-reinforcing once started; vision is not
    sufficient to sustain it.

### Cross-modal image-attention collapse (H-C baseline) — CL-09e
- **`p2_hc_baseline.py`** — recomputes the text→image-patch attention-mass statistic at L16/L20/L24
  with a proper control-doc baseline (the fix for the original baseline-free 1.6% number).
  - Produces: `results/p2_pilot/hc_baseline_results.json`.
  - Verified claim **CL-09e**: positive/control image-attention at L20 = 0.070 (14.2× lower at L20),
    0.216 @ L24 (4.6×), 0.695 @ L16 (N=1 vs 1).

### EOS-margin demo (relative-not-absolute EOS suppression)
- **`demo_eos_failure/05_scripts/regen_with_eos_logits.py`** — re-runs generation on the demo docs
  with `output_scores=True` + `return_dict_in_generate=True`, capturing the per-step EOS logit, EOS
  softmax prob, chosen-token logit, and EOS-vs-chosen margin. Writes `*_eos_trace.{pt,json}` to
  `demo_eos_failure/03_eos_trajectories/`.
- **`demo_eos_failure/05_scripts/plot_eos_failure.py`** — renders the EOS-failure figures from those
  cached trajectory tensors.
  - Cached evidence shipped here: `demo_eos_failure/03_eos_trajectories/`
    `docvqa_{kshm0227_p6, srgb0228_p2, fhxn0226_p2}_eos_trace.{pt,json}` (two phantom-row positives +
    one clean control). The plot script runs from these without a GPU.
  - Verified claim **EOS-margin**: the suppression is relative, not absolute — the cap-hit doc's mean
    EOS logit (+10.76) is actually *higher* than the control's (+10.07); continuation tokens
    outscore EOS by a chosen−EOS margin of ≈20.6, crushing P(EOS) to 1e-8…1e-10. Source file:
    `docvqa_srgb0228_p2_eos_trace.json`.

### Production fix — the L24 halt monitor
- **`fix/halt_monitor.py`** — the shippable module. A forward hook on decoder layer 24 reads the
  last-position residual at each step, projects it onto the trained halt direction, and a
  `LogitsProcessor` boosts the EOS logit once K=8 consecutive scores exceed threshold. Composable
  with `repetition_penalty` / `no_repeat_ngram_size`. `--train` re-fits the direction; `--test` runs
  the with/without comparison on the cached positives.
  - Behavioral claim **CL-09a / CL-17 family**: 82–98% generation-length reduction on confirmed
    positives with `stop=eos` (see the verification ledger's monitor bake-off entries CL-48/CL-39 for
    the precision-recall trade-off at different boost magnitudes).
- **`fix/halt_direction_L24.pt`** — the trained artifact `halt_monitor.py` and `test_b3_reverse.py`
  load: `{target_layer: 24, halt_direction: (2048,), intercept, scaler_mean, scaler_scale}`,
  fit on 305 within-doc position-matched samples.
- **`fix/test_halt_monitor.py`**, **`fix/run_fix_test_driver.py`** — test harness + driver for the
  monitor.
- **`p3_train_halt_direction.py`** — standalone trainer that fits a halt direction at an arbitrary
  layer and writes it in the same schema as `fix/halt_direction_L24.pt` (used to produce off-L24
  direction variants).

## Corpus build + activation extraction

These scripts build the evaluation corpora and the per-document activation cache that the
probing/patching scripts above read. They are the upstream data-prep stage of the pipeline.

- **`p1_build_public_corpus.py`** — assembles `data/public_corpus/` (images + `manifest.json` +
  `provenance.json`) by sampling public HF datasets — `lmms-lab/DocVQA` (validation), `nielsr/funsd`
  (train), `mychen76/invoices-and-receipts_ocr_v2` (valid) — plus the LoopyLeaderboard adversarial
  fixtures, normalizing each to PNG and recording `source` / `source_idx` / truncated `sha256`.
  - Run: `python code/p1_build_public_corpus.py --n-docvqa 60 --n-funsd 30 --n-invoices 30 --seed 0`.
- **`p1_curate_supplementary.py`** — builds `data/supplementary_corpus/` (201 renders that the P16
  and `paper_n` cohorts depend on) from three public sources: `ccdv/arxiv-summarization`
  (table-filtered), `eloukas/edgar-corpus` (SEC-EDGAR-style, with `JanosAudran/financial-reports-sec`
  / `nlpaueb/finer-139` / synthetic fallbacks), and `ds4sd/DocLayNet` (native pages). Text sources are
  rendered to wide PNG pages via Pillow; native pages are re-encoded. Idempotent; seed-pinned.
  - Run: `python code/p1_curate_supplementary.py --sources arxiv,sec,doclaynet --n-per-source 67 --seed 0`.
  - See `data/README.md` for the per-category source/rendering table and the regeneration recipe.
- **`p1_make_synthetic_tables.py`** — parametric markdown-table render generator at row counts
  N ∈ {5, 10, 20, 50, 100}, 4–6 columns, randomized cell content. Ground-truth row count is exact, so
  phantom-row count for any generation is unambiguous (`generated_rows − N`). Writes
  `data/synthetic/`.
- **`p1_make_synthetic_tables_dense.py`** — tier-2 escalation of the above: more rows (200, 400),
  8 columns, longer cells, smaller fonts, tighter spacing, to push the model past clean-halt behavior.
  Writes `data/synthetic_dense/`.
- **`p1_cache_harness.py`** — the **activation-extraction** stage. Given a trigger record
  (image, prompt, generated tokens, decision-moment position), runs a single teacher-forced forward
  pass over `[prompt + generated]` and caches a coarse band of layers (`--every-k-layers`, default 4 →
  9 of 36 layers) to `results/activations/<doc_id>/`. This is what writes the **`hidden_states.pt`**
  `(num_kept_layers, T, hidden)` and **`layer_indices.pt`** `(num_kept_layers,)` tensors (plus
  `residual_norms.pt`, optional windowed `attn_weights.pt`, and `meta.json`) that
  `q1_combined_probe.py`, `p3_logit_lens.py`, `p3_he_patch.py`, and the other probing/patching scripts
  consume. Applies the mandatory `lm_head` tie fix; bf16; `attn_implementation="eager"` when saving
  attention, `"sdpa"` otherwise.
  - Run (smoke-test): `python code/p1_cache_harness.py --doc table_N005_C04_s05000`.

## Shared helpers

- **`_constants.py`** — canonical model id / revision, layer sets (`KEPT_LAYERS`, `TEST_LAYERS`),
  architectural constants, and `find_image_path()`. Imported by `p6_l0_sufficiency.py`,
  `p_poscontrol_random_l24.py`, and `p3_he_patch.py`.
- **`_provenance.py`** — `write_provenance()` writes a per-results-dir reproducibility receipt
  (model revision, dtype, attn impl, seed, git HEAD, argv) at result-creation time. Imported by
  `p_poscontrol_random_l24.py`.

## Sanitization

This is a publication-bound reproduction repo. The following were replaced with placeholder tokens
matching the published Explorer skill convention, or removed:
- HPC username → `<user>`; SLURM account → `<slurm-account>`; allocation group → `<group>`.
- Cluster absolute paths (`/projects/<group>/<user>/…`, `/scratch/<user>/…`) and personal laptop
  paths → portable file-relative resolution with a `HALT_REPO_ROOT` override.
- Conda env name → `<conda-env>`; cluster login host / scheduler-specific phrasing genericized.
- Business-internal references (internal PR numbers, dashboards, decision logs) removed.

Public/scientific content is preserved verbatim: the model id `nanonets/Nanonets-OCR2-3B` and its
pinned revision, the public DocVQA dataset, all method names, and the `CL-xx` claim IDs (kept for
reproducibility traceability against the verification ledger).
