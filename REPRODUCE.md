# REPRODUCE.md — Halt-Failure Behaves Like a Class-Structured Residual-Stream Attractor in Fine-Tuned OCR-VLMs

This guide lets a stranger reproduce every confirmed result in the paper
*"Halt-Failure Behaves Like a Class-Structured Residual-Stream Attractor in Fine-Tuned OCR
Vision-Language Models"* from scratch, driving an Explorer-style SLURM cluster
(SLURM + H200 GPUs) autonomously from a Claude Code session.

Every number in the paper traces to an on-disk results file. This guide tells you,
per claim, exactly which script to run, which output file it writes, and which
headline number to check against the published ledger. If your regenerated number
disagrees with the ledger value below, the discrepancy is the finding — record it,
don't paper over it.

> **Placeholder convention.** Wherever you see an angle-bracket token like
> `<user>`, `<slurm-account>`, `<group>`, `<conda-env>`, `<org>/<model>`, or
> `<revision>`, substitute your own account's value. These match the
> [Northeastern Explorer autonomy skill](https://github.com/shehral/northeastern-explorer-autonomy-skill)
> "Customization Points" block one-for-one. Nothing in this repo hard-codes a
> personal username, allocation path, or billing account.

---

## 0. The 60-second mental model

Fine-tuned OCR-VLMs sometimes fail to emit EOS and run to `max_new_tokens`
(phantom table rows, repetition loops, hallucinated structural tags). The paper
advances the **hypothesis** that this *behaves like* a **distributed,
class-structured attractor in the residual stream**, not a single broken layer —
while being explicit that the causal nulls do not settle "distributed" against
"under-powered protocol" (the mechanism remains unidentified). The reproduction
splits into four buckets:

1. **Characterization** — build the cap-hit corpus, measure the device-induced
   positive-set shrink, capture per-step EOS logits.
2. **Mechanism (correlational)** — per-class halt-direction probes, logit-lens
   triangulation, FCCT build-vs-read decomposition, calibration rank.
3. **Mechanism (causal)** — a converging series of single-direction / single-layer
   perturbation **nulls**, plus a norm-scaled positive control that bounds (but does
   not eliminate) the "protocol under-powered" objection.
4. **Production + vision** — runtime EOS-boost monitor (precision-recall trade-off)
   and grey-noise image inpainting (vision is causally load-bearing).

---

## 1. Prerequisites

### 1.1 Software / accounts

| Need | Why | Notes |
|---|---|---|
| **Claude Code** | Drives the cluster autonomously (submit, monitor, integrate). | Any recent Claude Code build. The Explorer skill (below) is what makes hands-off operation safe. |
| **The Explorer autonomy skill** | Tiered autonomy, verified `sbatch` path, smoke gate, node-exclude, login-node aggregation. | Install from `https://github.com/shehral/northeastern-explorer-autonomy-skill`. Edit its Customization Points block once for your account. |
| **An Explorer-style SLURM cluster** | All H200 jobs run here. | Needs a `gpu` partition (8 h) + a `gpu-short` partition (≤ 2 h) and `--gres=gpu:h200:1`. Any SLURM cluster with H200s and an offline-compute-node posture works with minor edits. |
| **1× NVIDIA H200 (or H100 80 GB)** | bf16 weights, full KV cache, no quantization. | A 3B VLM at long context (T up to 12000) needs the headroom. |
| **The public model** | The system under test. | `nanonets/Nanonets-OCR2-3B`, pinned revision `c3886ff00bb037ce7da24988c9eafaf1fe2bed72` (a `Qwen/Qwen2.5-VL-3B-Instruct` fine-tune; 36 decoder layers, hidden 2048, 16 attn heads, 2 KV heads). |
| **The public base model** | For the fine-tune-introduced PCA signature (CL-25). | `Qwen/Qwen2.5-VL-3B-Instruct` — THE base; do not substitute other Qwen-VL family checkpoints, which test cross-family generality, not fine-tune introduction. |
| **The public DocVQA dataset** | Source of real cap-hit documents. | Public DocVQA. Many positive doc IDs in this repo are `docvqa_*` pages. |

### 1.2 Hardware budget

The full confirmed-result set is cheap: roughly **a handful of H200-hours** total
(most diagnostics are single-input or N≤56). Budget one day of wall-clock if you
run sequentially through a single `gpu` allocation. Use a soft daily GPU budget
(the skill's G4 guardrail, default `<daily-gpu-budget>` ≈ 24 H200-h/day) so an
autonomous loop can't run away.

### 1.3 Repo layout you'll produce

```
code/        experiment + analysis scripts (one concern per file; phase-prefixed)
results/     all outputs — generations, metrics, cached activations, plots (never hand-edited)
data/        eval-set images, ground-truth labels, manifests
code/fix/    the production module (halt_monitor.py, halt_direction_L24.pt)
code/demo_eos_failure/   per-step EOS-logit capture + plot scripts (05_scripts/)
review_corpus/      cap-hit docs sorted by failure mode + viewer.html
environment/        pinned deps + env build log (this dir already exists)
```

---

## 2. Environment setup

### 2.1 Local (for smoke tests and plotting — optional but recommended)

Smoke-testing locally before any `gpu` submission is the single highest-leverage
habit (≈ 5000:1 cost asymmetry). A 24 GB Apple-Silicon laptop runs the model for
context lengths up to ~2500 tokens with `attn_implementation="eager"`.

```bash
# Python 3.11 in an isolated venv (3.11 avoids an expat ABI mismatch seen on macOS 25.x)
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install \
  torch transformers accelerate pillow numpy scipy scikit-learn \
  matplotlib datasets nnsight
```

Do **not** use TransformerLens — it does not support this VLM. Use `nnsight` with
its `VisionLanguageModel` wrapper (not generic `NNsight()`) for all probing /
patching.

### 2.2 Cluster (the real environment — conda + CUDA)

On the **login node** (never the compute node for setup), build the pinned conda
env. Substitute your own allocation and env-name placeholders:

```bash
# placeholders to fill in:
#   <user>          your Explorer login
#   <group>         your project group
#   <conda-env>     your pinned env name
ALLOC=/projects/<group>/<user>
ENV=$ALLOC/conda_envs/<conda-env>

module load miniconda3/24.11.1 cuda/12.3.0   # pin CUDA 12.3.0 — 12.8 is too new for torch 2.4.1
conda create -p "$ENV" python=3.11 -y
export PATH="$ENV/bin:$PATH"

pip install \
  torch==2.4.1 transformers accelerate pillow numpy scipy \
  scikit-learn matplotlib datasets nnsight

pip freeze > results/env.txt   # record the exact env on every build
```

**Non-negotiable runtime settings** (these are correctness gates, not preferences):

- **bf16 weights.** Never int4/int8. Never quantize the KV cache. Quantization
  perturbs the very logit margins (~20 logits) the paper measures.
- `attn_implementation="eager"` for any attention-extracting diagnostic;
  `"sdpa"` for routine generation.
- Greedy decoding (`do_sample=False`) for all diagnostics. Pin and log every seed.

### 2.3 Pre-cache the model on the LOGIN node (mandatory pair with offline mode)

Compute nodes have **no internet** (no proxy passthrough). A HuggingFace load
without offline mode silently 503s and falls back to **meta-init zeros** — the model
"loads" and then generates garbage forever, burning your whole walltime. So:

```bash
# on the LOGIN node only (it has proxy access):
export HF_HOME=$ALLOC/hf_cache
huggingface-cli download nanonets/Nanonets-OCR2-3B \
  --revision c3886ff00bb037ce7da24988c9eafaf1fe2bed72
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
```

Then on **every** compute-node job set offline mode and pass `local_files_only`:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # in the sbatch
```
```python
AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REV, local_files_only=True)
```

Do **not** load the model on the login node itself — it has a ~8 GB effective RAM
cap and will OOM-kill you. Pre-cache (download) on login; load only on compute.

### 2.4 The MANDATORY `lm_head` tie-fix for this checkpoint

`Nanonets-OCR2-3B` omits `lm_head.weight` and relies on `tie_word_embeddings`.
But transformers 5.x reads that flag off the **outer** config (`False`) instead of
`text_config` (`True`), so `tie_weights()` becomes a no-op, `lm_head` stays at
meta-init zeros, and **generation collapses to `"!"` forever**. Every loader in this
repo must do, immediately after `from_pretrained`:

```python
model.config.tie_word_embeddings = True
model.tie_weights()
assert (model.lm_head.weight.data_ptr()
        == model.model.language_model.embed_tokens.weight.data_ptr())
```

If the assert fails, stop — every downstream number is invalid. (This fix is
checkpoint-specific. If you adapt this guide to a different model, delete or replace
it.)

### 2.5 Canonical sbatch skeleton

Compose every job from the autonomy skill's `assets/job_template.sbatch`. The
load-bearing fragments:

```bash
#SBATCH --account=<slurm-account>
#SBATCH --partition=gpu              # or gpu-short for ≤2h smoke/aggregation
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --export=ALL                 # propagate proxy vars (login-set) but proxy itself is unreachable on compute

module load miniconda3/24.11.1 cuda/12.3.0
ENV=/projects/<group>/<user>/conda_envs/<conda-env>
export PATH="$ENV/bin:$PATH"
PYTHON="$ENV/bin/python"             # always explicit $PYTHON, never plain `python`
export HF_HOME=/projects/<group>/<user>/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$PYTHON" code/<script>.py --seed 0 --revision c3886ff00bb037ce7da24988c9eafaf1fe2bed72 ...
```

**Do NOT use `source activate`** — use the explicit `PATH`-export + explicit
`$PYTHON` pattern above.

---

## 3. How to drive the cluster autonomously

Read the published skill before your first submission:
**https://github.com/shehral/northeastern-explorer-autonomy-skill**

The four mechanics that make hands-off runs safe:

1. **Tiered autonomy (T0–T3).** Reads / smoke-tests / local edits are always
   allowed (T0). An HPC job ≤ a walltime cap auto-submits **only if all five
   guardrails pass** (T1). New experiment designs are done-but-surfaced (T2).
   Destructive ops, external comms, and touching restricted artifacts never
   auto-run (T3). **Start a new project at T0+T2 only**; promote to T1 once you've
   watched the verifier + budget + streak machinery behave across several manual
   cycles.
2. **The verified `sbatch` path.** A PreToolUse hook runs a per-job verifier
   *before* the `sbatch` lands: the script + sbatch exist, `--smoke` ran clean,
   offline flags are set, the pre-registration constant is present, and — the
   end-to-end wiring check — a with-intervention run produces **different output**
   than without. This catches the silent "the hook never actually fired" bug that
   otherwise masquerades as a real (null) result.
3. **The smoke gate (G1).** A 60-second single-input run (local or `gpu-short`)
   must pass before any `gpu` submission.
4. **Node-exclude + login-node aggregation.** Some nodes signal-53 launch-kill jobs
   in a node-specific (not code) way — exclude them with `--exclude`. Long jobs can
   complete their compute grid and then time out on a short partition's cap; the
   final CPU-only aggregation step (e.g. building a `_summary.json`) can run on the
   **login node** to recover the result without re-burning GPU time.

**Persistent Monitor** for any job > 1 h (`persistent: true`); a missing output
file is **not** a dead-monitor signal. Keep an unattended loop alive with
`/loop 30m ...` or a cron cadence at a **1200–1800 s** interval (never 300 s —
cache-window thrash).

---

## 4. Per-claim reproduction

Each entry gives: **what it shows**, the **script**, the **expected output file**,
and the **headline number to check** (the CONFIRMED ledger value). Run the
characterization claims first (they build the corpus everything else indexes).

> All ledger values below are the on-disk CONFIRMED values. Treat them as the
> pass/fail target for your regenerated run.

### 4.1 Characterization

#### CL-02 — Device-induced positive-set shrink (MPS → CUDA)
- **Shows:** MPS-to-CUDA non-determinism *shrinks* the stable positive set, so the
  device-consistent CUDA set is the conservative ground truth.
- **Run:** `code/p1_cuda_n40` re-screen over the 123-document set on CUDA (H200).
- **Output:** `results/p1_cuda_n40/_summary.json`
- **Check:** strict CUDA-positive rate **6/123 = 4.88%**; **8** MPS-positives flip to
  CUDA-control; **0** new CUDA-only positives.

#### CL-02b-corpus — Cap-hit corpus + failure taxonomy
- **Shows:** the confirmed positive set and its surface-class structure.
- **Run:** corpus build → `review_corpus/00_INDEX.csv` (excludes `BOUNDARY`,
  `FALSE_POSITIVE`, `99_UNCLASSIFIED`).
- **Output:** `review_corpus/00_INDEX.csv` (+ `review_corpus/viewer.html`)
- **Check:** **60 cap-hit document IDs** in the index → **56 confirmed positives**
  across **12 populated** surface classes (after excluding 2 `BOUNDARY`, 1
  `FALSE_POSITIVE`, 1 `99_UNCLASSIFIED`). Note: an earlier "14 classes"
  figure is a hardcoded title string — the on-disk truth is 12 populated; do **not**
  cite 14. The "45 strict / 49 outline" and "1,139 runs / 424 pages" counts have no
  on-disk source — do not cite.

#### EOS-relative-margin — EOS suppression is relative, not absolute
- **Shows:** the cap-hit document's EOS logit is *higher* than a clean control's;
  continuation tokens simply out-score EOS by ~20 logits.
- **Run:** `code/demo_eos_failure/05_scripts/regen_with_eos_logits.py` (captures per-step EOS logits
  via `output_scores=True` + `return_dict_in_generate=True`).
- **Output:** `demo_eos_failure/03_eos_trajectories/docvqa_srgb0228_p2_eos_trace.json`
  (the EOS-margin figure is regenerated by `code/demo_eos_failure/05_scripts/plot_eos_failure.py`; it is not checked in).
- **Check:** chosen−EOS margin mean **≈ 20.61** (median ≈ 20.63); P(EOS) spans
  **1e-8 … 1e-10**; cap-hit mean EOS logit **+10.76** > control **+10.07**
  (`docvqa_fhxn0226_p2`).

### 4.2 Mechanism — correlational

#### CL-42 — Per-class halt directions, distinct + content-decoupled
- **Shows:** per-class halt directions are linearly decodable across the decoder and
  decoupled from content (PASS_BOTH).
- **Run:** per-class LOPO/LODO probing over 10 sampled layers
  (L0, L4, L8, L12, L16, L20, L24, L28, L32, L35).
- **Output:** `results/q1_combined/per_layer_extracted.csv` and
  `results/q1_combined/_summary.json`
- **Check:** per-class AUC **0.876–0.983** (filled@L8 = 0.8763 … filled@L24 = 0.9826);
  peaks at **different** layers (latex@L8 = 0.9785, filled@L24 = 0.9826,
  bare@L20 = 0.9173); verdict **PASS_BOTH** (10/10 distinct; 3/10 global at
  L4/L8/L32; max pairwise cosine **0.2349** @ L20).

#### CL-42-global — The single global direction is weaker
- **Output:** same `results/q1_combined/per_layer_extracted.csv`
- **Check:** global AUC peaks **L8 = 0.8571**, troughs **L32 = 0.7702** — below the
  per-class AUC at every layer.

#### CL-50 — Build-vs-read (FCCT block-contribution)
- **Shows:** the filled-cell halt direction is *built* at L20 and *read* at L24.
- **Run:** FCCT block-contribution decomposition.
- **Output:** `results/fcct_refresh/per_class_block_contribution_aggregate.csv`
- **Check:** `mean_cos_block` **+0.042 @ L20** vs **−0.010 @ L24** (filled, N=18).

#### CL-16 — Logit-lens EOS gap arrives late
- **Shows:** the downstream "should-halt" signal becomes legible only in the final
  layers.
- **Run:** apply final RMSNorm + `lm_head` to the cached residual stream per layer;
  read EOS logit, positives vs controls.
- **Output:** `results/p3_logit_lens/_summary.json`
- **Check:** EOS gap most negative **L24 = −0.419**, flips to **+0.481 @ L32**,
  **+0.336 @ L35** (N=4 pos + 9 ctl).

#### CL-22 — EOS genuinely missing at loop onset
- **Shows:** not a near-miss in calibration — EOS is deeply buried.
- **Output:** `results/p6_mirror/B_calib_onset/_summary.json`
- **Check:** EOS **median rank ≈ 10,502**; only **0.9%** of records have EOS in top-5
  (N=110 over 22 docs). Note the framing caveat: programmatic loop-onset detection
  biases late.

#### CL-05e / CL-21 — Cross-doc LODO probe + per-layer distribution
- **Shows:** halt detection is distributed across L12–L24, not localized.
- **Output:** `results/p3_full_probe/full_probe_results.json` (CL-05e),
  `results/p4_mirror/H7_cross_layer/_summary.json` (CL-21)
- **Check (CL-05e):** L24 LODO AUC **0.894**; B=20 block-shuffle **p = 0.0033**,
  which is the **resolution floor** `1/(N_perm+1) = 1/301` at **N_perm = 300** (the
  observed AUC beat all 300 block-shuffled permutations, exceedance count 0) — read it
  as **p <= 0.0033**, not a finer measured separation; a tighter floor below the 0.005
  Bonferroni threshold needs a re-run at larger N_perm. Only **L20 + L24** clear the
  Bonferroni threshold of 0.005 at B=20.
- **Check (CL-21):** per-layer LODO AUC **≥ 0.84 every layer**; L16 = 0.929 edges
  L24 = 0.914 (N=22).

#### CL-25 — Fine-tune-introduced PCA signature (Nanonets)
- **Shows:** the fine-tune introduces a detectable residual-stream PCA geometry
  signature.
- **Run:** PCA pos/ctl |Cohen's d| crossover at L16/L20/L24 on `Nanonets-OCR2-3B`.
- **Output:** `results/p7b_mirror/PCA_control/_summary.json`
- **Check:** |d| ratio **L16 = 3.79, L20 = 2.74, L24 = 2.78** (N=22 pos / 52 ctl) —
  all above the pre-registered 2.0× threshold. The canonical base-model
  `Qwen2.5-VL-3B` triple (0.96/0.89/0.95) is **unverifiable** — do not cite it;
  the on-disk base file shows superseded 0.66–0.80. Direction holds (base < 2.0×);
  the specific base triple does not.

### 4.3 Mechanism — causal (the converging nulls)

> These are first-class results. The point is that single-direction / single-layer
> perturbations **fail** to move token count. Reproduce the nulls *and* the positive
> control that shows the protocol was not inert (it visibly alters content) — but note
> the nulls do not, by themselves, settle "distributed mechanism" against "under-powered
> protocol." The mechanism remains unidentified at this protocol's power (CL-35).

#### CL-49 — Sufficiency null at L0
- **Shows:** injecting the bare-word-repeat direction at L0 does not induce
  halt-failure.
- **Run:** inject class-matched halt direction at L0 at α=10; compare cap-hit rate vs
  vanilla. **Pre-register** the PASS threshold (≥ 40 pp) before running.
- **Output:** `results/p6_l0_sufficiency/_summary.json`
- **Check:** ΔC1 = **0.0 pp** @ α=10 → **FAIL** vs pre-reg ≥ 40 pp. L0 is a readout.

#### CL-36 — Norm-scaled positive control (bounds, but does not eliminate, "protocol under-powered")
- **Shows:** scaling a single-direction L24 patch to 0.1× / 10× / 100× residual norm
  changes content visibly but never escapes the cap.
- **Output:** `results/p_poscontrol/_summary.json` (labels in `provenance.json` →
  `manifest_docs`)
- **Check:** the run held **4 documents but only 3 are cap-hit positives**
  (`manifest_docs` label `positive`: `docvqa_gjhp0000_p1`, `docvqa_jqbg0227_p1`,
  `docvqa_srgb0228_p2`, each vanilla = 12000 tokens). **0/3** of those escape the
  12000-token cap at any scale; content visibly changes (e.g. a prepended Chinese token,
  `**` formatting noise). The protocol delivers content-altering perturbations but cannot
  dislodge the halt state. The 4th doc, `table_N005_C04_s05000`, is labeled `control`
  and is **not** a cap-hit positive — it halts cleanly on its own (vanilla = **477
  tokens, stop_reason = eos**), so it can never "escape" a cap it never hit and is
  excluded from the positive tally. Note it as a caution, not support: under the same
  patch its output **collapsed** (C1_random_10x → **3 tokens**, C3_random_0.1x → **1
  token**, C2_random_100x → **12000 tokens** of degenerate single-character repetition).
  That collapse is a norm-shock signature that *weakens* rather than strengthens the
  protocol-sensitivity rebuttal.
- **What this does and does not establish:** it rebuts only the *strongest* form of the
  "protocol under-powered" objection — that the protocol is inert. It does **not** rebut
  the weaker, more relevant form: that the protocol cannot perturb the *specific*
  halt-relevant direction at the *specific* halt-relevant locus. A random-direction
  patch that visibly reorders surface tokens (or shatters an unrelated clean halt) is not
  evidence that a targeted halt-mechanism patch would have been detectable. Treat it as a
  sensitivity floor, not a clean refutation of under-power. The mechanism remains
  unidentified at this protocol's power.

#### CL-12 — B3 reverse-direction necessity null (READOUT)
- **Shows:** projecting the halt direction *out* of every position does not lengthen
  generation.
- **Run:** `code/<b3_reverse_direction>.py`, pinned revision `c3886ff`, greedy CUDA.
- **Output:** `results/p2_pilot/b3_reverse_direction_summary.json`
- **Check:** **3/3** controls unchanged (<10%: +9.1% / 0% / 0%); positive pinned
  **12000 → 12000**; verdict **READOUT**. End-to-end wiring confirmed: 2/4 docs
  change output (one positive flips markdown→HTML), so the project-out hook genuinely
  fires — a real null, not an inert intervention.

#### CL-19 / CL-20 — Component-resolved FCCT patch grid (real-null)
- **Shows:** MLP / attn / block patches at L16/L20/L24 produce no length reduction.
- **Run:** `code/p3_he_patch.py` over 5 docs × 3 layers × 3 components × 4
  interventions, manifest `data/p16_manifest_labeled.json`. Pre-reg at
  `docs/p16_he_patch_preregistration.md`. Use `--exclude` for signal-53 nodes;
  the final CPU-only aggregation can run on the login node.
- **Output:** `results/p16_component_resolved_short/_summary.json`
- **Check:** **0/45** real-patch cells nonzero (clean real-null); 9/134 controls
  nonzero (controls perturb more); verdict **FAIL** — reproduces the original
  0/58-real, 13/174-control finding.

#### CL-12b — P10b induction-head zeroing null
- **Shows:** zeroing the top induction heads does not move token count.
- **Output:** `results/p10b_head_zero_patching/_summary.json`
- **Check:** `PRE_REG_VERDICT` = **FAIL**; `induction_zero_passes_n` = **0** (0/4
  induction heads pass the pre-reg 30% threshold under zeroing); **N=2** docs. One control
  head (L19h7, 45.76% mean) and one induction head (L17h10, 46.45% mean) move length on a
  single doc each — neither passes the rule.

#### CL-12c — P19 system-prompt counterfactual null
- **Shows:** stop-instruction prompts do not reliably stop the runaway.
- **Output:** `results/p19_system_prompt_counterfactual/_summary.json`
- **Check:** `PRE_REG_VERDICT` = **FAIL**; `passing_prompts` = **[]** (0/6 prompts pass);
  all six aggregate `median_reduction_pct` = **0.0**; **N=5** docs.

#### CL-12d — P12 SAE-feature ablation null
- **Shows:** no single SAE feature carries the halt decision under cross-validation.
- **Output:** `results/p12_sae_ablation/_summary.json`
- **Check:** `PRE_REG_VERDICT` = **FAIL**; `folds_passing_all_criteria` = **0** (0/5 CV
  folds); per-fold top-feature AUC **0.562 / 0.566 / 0.543 / 0.585 / 0.629** (all below the
  0.85 bar); `n_positives` = **22**, `n_controls` = **56**; `saelens_version` = 6.0.0.

#### CL-12e — P22 multi-layer coordinated patch null
- **Shows:** zeroing a coordinated L16/L20/L24 window does not move token count.
- **Output:** `results/p22_multilayer_coordinated/_summary.json`
- **Check:** `PRE_REG_VERDICT` = **FAIL**; the `real_coord_zero_target_window` aggregate
  `mean` = **0.0** (0/2 pass 30%); **N=2** docs. Only the off-target control
  (`ctl_off_target_zero_window`) moves length (99.99%), a degenerate off-target collapse,
  not a circuit hit.

> **Honest count:** all **six** converging perturbation nulls now reproduce from local
> `results/`, each returning its pre-registered FAIL/null verdict, with **N ranging 2-22
> documents**: **B3** (`p2_pilot/b3_reverse_direction_summary.json`, READOUT), **P16**
> (`p16_component_resolved_short/_summary.json`, FAIL, N=5), **P10b**
> (`p10b_head_zero_patching/_summary.json`, FAIL, N=2), **P19**
> (`p19_system_prompt_counterfactual/_summary.json`, FAIL, N=5), **P12**
> (`p12_sae_ablation/_summary.json`, FAIL, N=22), and **P22**
> (`p22_multilayer_coordinated/_summary.json`, FAIL, N=2). The four formerly-on-`/scratch/`
> summaries (P10b, P19v2, P12-v3, P22) were synced here and verified; the paper now states
> the count as "six converging perturbation nulls, all returning the pre-registered
> FAIL/null verdict (N ranging 2-22 docs)." P16 subsumes CL-19/CL-20 (counted once, not
> double-counted). The older "8 converging nulls" figure was prose-only and stays retired.
> The caveat is kept: three of the six rest on **N=2** (B3's positive, P10b, P22), so a
> uniform null at that sample size does **not** by itself separate "distributed mechanism"
> from "under-powered protocol." The nonzero controls are dominated by
> `off_layer` norm-shock, so they bound protocol sensitivity rather than localize a
> mechanism — which is exactly why CL-36's matched-norm positive control is load-bearing.
>
> **The honest bottom line (per the project's own CL-35).** All single- and multi-site
> perturbations returned null at this protocol's power. Whether the mechanism is genuinely
> distributed or the protocol is under-powered at the halt-relevant locus is **not
> resolved** by these data; the mechanism remains unidentified. CL-36 bounds, but does not
> eliminate, the under-powered alternative. The "distributed attractor" reading is a
> labeled hypothesis that fits the full evidence pattern, not a causally-isolated
> mechanism. The discriminating experiment (patching the *same* halt direction at a
> *non-loop* position) is left to future work.

### 4.4 Production fix + vision

#### CL-39 / CL-48 — Runtime EOS-boost monitor (precision-recall trade-off)
- **Shows:** a runtime monitor reading the L24 halt direction trades recall against
  control specificity; there is no free lunch (this supersedes a walked-back
  "83.1% + 0 FP" single-number claim).
- **Run:** `fix/halt_monitor*.py` bake-off across boost magnitudes.
- **Output:** `results/p97_rebakeoff_boost_magnitudes/boost_6p91/_comparison_table.json`
  (CL-39) and `.../boost_30/_comparison_table.json` (CL-48)
- **Check (CL-39, deployable point):** **+6.91** boost → **37.68%** mean length
  reduction, **0/10** control false-positives, 5/13 positive cap-hits remaining.
- **Check (CL-48, catastrophic point):** **+30** boost → **99.58%** reduction, 0
  cap-hits — but **10/10** control false-positives. The two points bracket the
  trade-off; ship the conservative +6.91.

#### CL-11 — Grey-noise inpaint (vision is causally load-bearing)
- **Shows:** replacing the image with grey noise collapses runaways to clean EOS
  halts.
- **Output:** `results/p3_image_inpaint/_summary.json`
- **Check:** baseline 12000-token `max_new_tokens` stops become **13–18 token,
  `stop_reason=eos`** halts under grey-noise inpaint. The on-disk file holds 5/5;
  the paper reports the named subset at 4/4 (Wilson 95% CI [51%, 100%]) —
  cite the conservative 4/4.

#### CL-09e — Text-to-image attention collapse (N=1-vs-N=1, exploratory)
- **Shows:** a correlational vision signal consistent with CL-11.
- **Output:** `results/p2_pilot/hc_baseline_results.json`
- **Check:** pos/ctl text→image attention-mass ratio **0.070 @ L20** (14.2× lower),
  **0.216 @ L24** (4.6× lower), **0.695 @ L16**. Flag explicitly as N=1-vs-N=1;
  reserved for future N=40 work — do not over-claim.

---

## 5. The verification discipline

This is the part that makes the paper reproducible rather than merely runnable.

1. **Every number traces to a file.** No load-bearing figure may live in prose. If
   you cannot point at a `results/.../_summary.json` (or equivalent) record that
   produces it, mark it `[UNVERIFIED - pending re-run]` and do not use it as
   evidence. The paper's verification block enumerates exactly which claims are
   CONFIRMED vs UNVERIFIABLE/STALE.

2. **Sync vs regenerate — know which you did.**
   - **Sync** = the genuine result already exists on the cluster and was never
     pulled local. Pull it; verify the headline number on disk; record provenance.
     (Example: the B3 necessity null existed on the cluster and was synced down, not
     re-run.)
   - **Regenerate** = no trustworthy on-disk artifact exists, so re-run the script
     from a pinned revision and clean output dir. (Example: the component-resolved
     FCCT grid was regenerated from `code/p3_he_patch.py`.)
   - Never silently mix the two. A regenerated number that disagrees with the
     synced/ledger value is a finding to report, not a value to overwrite.

3. **Clean the output dir on resubmit after a script-bug fix.** Skip-existing logic
   silently inherits bug-mode data otherwise.

4. **End-to-end wiring check before any with-module vs vanilla comparison.**
   Generate one token with the module active and assert the EOS-logit delta ≠ 0.
   This catches "the hook isn't actually intervening" — the bug that turns a real
   intervention into a fake null.

5. **Provenance on every results dir.** Each `results/<exp>/` carries a
   `provenance.json`: `{seed, model_revision, dtype, attn_impl, git_commit,
   started_at, finished_at, script, args}`. Backfill before any external share; do
   not auto-fabricate it (a wrong fix to a right finding is still wrong).

6. **Pre-register falsification criteria BEFORE the validating run.** Each causal
   test carries a numeric PASS threshold fixed in advance (e.g. CL-49's ≥ 40 pp).
   Significance for the cross-doc probe uses a block-shuffle null at B = 5, 10, 20
   with Bonferroni correction; B=20 is the autocorrelation-conservative,
   load-bearing block size. The three nulls are persisted at
   `results/p2_pilot/round2_block_shuffle_null_B{5,10,20}.json` (each gives
   layers 20 and 24 p=0.0033). That p=0.0033 is the resolution floor
   `1/(N_perm+1) = 1/301` at N_perm=300 — the observed AUC beat all 300
   permutations (exceedance count 0) — so read it as p<=0.0033, not a measured
   separation; only L20 and L24 clear the Bonferroni threshold of 0.005 at B=20.

---

## 6. Confirmed-result checklist (copy into your run log)

| Claim | Output file | Headline to match |
|---|---|---|
| CL-02 | `results/p1_cuda_n40/_summary.json` | 6/123 = 4.88%; 8 flip to control; 0 new |
| CL-02b-corpus | `review_corpus/00_INDEX.csv` | 56 positives / 12 classes / 60 cap-hits |
| EOS-margin | `demo_eos_failure/03_eos_trajectories/docvqa_srgb0228_p2_eos_trace.json` | margin ≈ 20.61; EOS +10.76 > ctrl +10.07 |
| CL-42 | `results/q1_combined/per_layer_extracted.csv` + `_summary.json` | AUC 0.876–0.983; PASS_BOTH; max cos 0.2349 |
| CL-42-global | `results/q1_combined/per_layer_extracted.csv` | global L8=0.8571, L32=0.7702 |
| CL-50 | `results/fcct_refresh/per_class_block_contribution_aggregate.csv` | +0.042 @ L20 vs −0.010 @ L24 |
| CL-16 | `results/p3_logit_lens/_summary.json` | L24=−0.419 → L32=+0.481, L35=+0.336 |
| CL-22 | `results/p6_mirror/B_calib_onset/_summary.json` | median rank ≈ 10,502; 0.9% top-5 |
| CL-05e | `results/p3_full_probe/full_probe_results.json` | AUC 0.894; B=20 p<=0.0033 (=1/301 floor at N_perm=300); only L20+L24 clear Bonferroni 0.005 |
| CL-21 | `results/p4_mirror/H7_cross_layer/_summary.json` | ≥ 0.84 every layer; L16=0.929 |
| CL-25 | `results/p7b_mirror/PCA_control/_summary.json` | |d| ratio 3.79 / 2.74 / 2.78 |
| CL-49 | `results/p6_l0_sufficiency/_summary.json` | ΔC1 = 0.0 pp → FAIL |
| CL-36 | `results/p_poscontrol/_summary.json` | 0/3 cap-hit positives escape; content changes (4th doc is clean-halt control, collapsed under patch) |
| CL-12 | `results/p2_pilot/b3_reverse_direction_summary.json` | 3/3 ctrl unchanged; READOUT |
| CL-19/20 | `results/p16_component_resolved_short/_summary.json` | 0/45 real nonzero; FAIL |
| CL-39 | `results/p97_rebakeoff_boost_magnitudes/boost_6p91/_comparison_table.json` | 37.68%; 0/10 control FP |
| CL-48 | `results/p97_rebakeoff_boost_magnitudes/boost_30/_comparison_table.json` | 99.58%; 10/10 control FP |
| CL-11 | `results/p3_image_inpaint/_summary.json` | 13–18 tok, stop=eos; report 4/4 |
| CL-09e | `results/p2_pilot/hc_baseline_results.json` | 0.070 @ L20; N=1-vs-N=1 |

---

## 7. Known footguns (read before your first H200 burn)

- **The `"!"` collapse** = you forgot the `lm_head` tie-fix (§2.4). Every output is
  `"!"`. Re-assert the `data_ptr` equality.
- **Garbage-forever generation** = compute node loaded the model online and got
  meta-init zeros. Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` + `local_files_only`,
  and confirm the model was pre-cached on login (§2.3).
- **Login-node OOM-kill** = you loaded the model (or ran a heavy `du`) on the login
  node. Only load on compute; only download on login.
- **Signal-53 launch-kills** = node-specific, not your code. `--exclude` the bad
  nodes and resubmit.
- **A "real" null that's actually an inert hook** = run the end-to-end wiring check
  (§5.4) first.
- **`source activate` failures** = use the explicit `PATH`-export + `$PYTHON`
  pattern (§2.5).

---

*All paths above are repo-relative. Substitute every `<…>` placeholder with your own
account's values per the Explorer skill's Customization Points block. The model id,
its pinned revision, the public DocVQA dataset, the method names, and the CL-xx claim
IDs are public and load-bearing for reproducibility — they are intentionally kept.*
