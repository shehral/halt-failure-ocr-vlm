# Substantiation evidence — what each shipped artifact backs

This file maps the substantiation-evidence batch shipped into `results/` to the specific
claim each artifact backs. It complements `REGENERATE.md` (the checked-in-vs-regenerated
ledger). Every file was sanitized for public release: absolute cluster paths, the HPC
username, the SLURM account, scratch paths, the conda env name, SLURM job/node IDs, and
any internal references were replaced with `<placeholder>` tokens. The public model id
`nanonets/Nanonets-OCR2-3B` (+ its public revision), the public dataset names (DocVQA,
FUNSD, arXiv-table), method names, and the `CL-xx` claim IDs are preserved for traceability.

## The claims these artifacts substantiate

**Base-model comparison (the "fine-tune-introduced signature" claim).** The mechanism
story rests on the fine-tune introducing a *detectable* residual-stream geometry signature
that is absent (or far weaker) in the base checkpoint. `p18_base_archetype/_summary.json`
is the base-model PCA control: it re-runs the per-doc residual-stream PCA archetype
comparison on the **base** `Qwen/Qwen2.5-VL-3B-Instruct` (the sole checkpoint
Nanonets-OCR2-3B was fine-tuned from), measuring the pos/ctl |Cohen's d| ratio at L16/L20/L24.
The base ratios sit at ~0.78–1.49× (vs the 2.7–3.8× reported in the fine-tune), and the base
checkpoint itself cap-hits on the positive docs (7 base cap-hits / 15 base clean-halts) —
substantiating both that the *continuation bias is inherited from the base* and that the
*sharp PCA separation is fine-tune-introduced*, not pre-existing.

**Four further converging causal-perturbation nulls (the "no single-/multi-layer halt circuit;
detection is distributed" claim).** These four join the B3 reverse-direction project-out
(`p2_pilot/b3_reverse_direction_summary.json`, READOUT) and the P16 component-resolved patch
grid (`p16_component_resolved_short/_summary.json`, FAIL, N=5) to make **six converging
perturbation nulls, all now on disk**, N ranging 2-22 docs. Each of these four is a
pre-registered causal perturbation whose pre-registered criterion was NOT met
(`PRE_REG_VERDICT: FAIL`), which is the *intended* evidential role — converging nulls refute a
single-locus causal reading and support the distributed-detection framing. Note the caveat
(CL-35): the nulls do not resolve distributed-vs-under-powered, sharpest at the N=2 nulls
(P10b and P22 below, and B3's positive):

- `p10b_head_zero_patching/_summary.json` — zeroing the top induction heads (and norm-matched
  head controls) does not reduce runaway generation beyond the noninduction control floor
  (induction-zero passes = 0; verdict FAIL). Refutes an "induction-head halt circuit."
- `p19_system_prompt_counterfactual/_summary.json` — 6 stop-instruction system-prompt
  counterfactuals over 5 docs; no prompt reliably averts the cap-hit (passing prompts = [];
  verdict FAIL). Refutes a prompt-level fix and shows the failure is not instruction-surface.
- `p12_sae_ablation/_summary.json` — 5-fold SAE feature search for a "halt-pure" feature;
  0/5 folds meet the pre-registered AUC/separation/stability bar (verdict FAIL). The halt
  mechanism does not decompose into a single interpretable SAE feature.
- `p22_multilayer_coordinated/_summary.json` — coordinated multi-layer (L16/L20/L24) window
  zeroing plus norm-matched and shifted-window controls; the real coordinated intervention
  yields 0% reduction while only the off-target norm-shock control collapses generation
  (verdict FAIL). Refutes the "coordinated multi-layer intervention is the right causal class."

**Bootstrap CIs on the logit-lens EOS gap (guards against a single-layer localization claim).**
`p3_logit_lens/bootstrap_ci.json` reports B=10000 bootstrap 95% CIs for the logit-lens
EOS-gap (positives minus controls) at every sampled layer. At N=4 positives / 9 controls,
**every** layer's CI includes 0, including L24 — so "most negative at L24" is an unreliable
point estimate and cannot support single-layer localization. This is the honest-uncertainty
backstop for any logit-lens readout claim (CL-16 family).

**Permutation-null recompute / resolution-floor disclosure (cross-doc LODO probe
significance, honestly stated).** `p3_full_probe/perm_null_recompute.json` documents that the
published L24 block-shuffle p≈0.0033 is exactly the `1/(N_perm+1)` resolution floor at
N_perm=300 (0/300 exceedances), reports the B=10 and B=20 per-layer block-shuffle nulls,
and records why a true higher-resolution recompute could not be reproduced from cache without
fabrication. It backs the probe-significance claim (CL-05e family) while correctly bounding it
as "p < 0.0033, resolution-floor-limited, survives Bonferroni at B=10 and B=20."

**Cross-family generality evidence (the "halt-failure reproduces across architecture
families" claim, CL-43–CL-47).** `p3_cross_family/` holds the staged cross-family replication:
SmolDocling-256M (Idefics3/SigLIP+SmolLM2), Qwen3-VL-8B, InternVL3-8B, and Qwen3-VL-30B-A3B
(MoE) run over 4 shared positive docs (`docvqa_jqbg0227_p1`, `docvqa_kshm0227_p7`,
`docvqa_srgb0228_p2`, `funsd_105_105`). Per-model per-doc JSONs record `tokens_emitted`,
`stop_reason`, and the production patterns fired (`ngram_repeat`, `empty_rows`); the
`_summary.json` and `_stageN_candidates.json` files carry the staged-selection rationale and
provenance. Together these substantiate that the failure spans 256M→30B params and 3 vendors,
not just the Nanonets fine-tune.

## File index (this batch)

| Claim(s) backed | Subdir / file | Role |
|---|---|---|
| Fine-tune-introduced PCA signature (base control) | `p18_base_archetype/_summary.json` | base-model PCA comparison |
| No induction-head halt circuit (converging null) | `p10b_head_zero_patching/_summary.json` | head-zeroing patch, verdict FAIL |
| No prompt-level halt fix (converging null) | `p19_system_prompt_counterfactual/_summary.json` | system-prompt counterfactual, verdict FAIL |
| No single SAE halt feature (converging null) | `p12_sae_ablation/_summary.json` | SAE feature search, verdict FAIL |
| No coordinated multi-layer halt circuit (converging null) | `p22_multilayer_coordinated/_summary.json` | multi-layer window patch, verdict FAIL |
| Logit-lens EOS gap — no single-layer localization | `p3_logit_lens/bootstrap_ci.json` | B=10000 bootstrap 95% CIs |
| Cross-doc LODO probe significance (resolution-floor honest) | `p3_full_probe/perm_null_recompute.json` | block-shuffle null recompute + disclosure |
| Cross-family generality (CL-43–CL-47) | `p3_cross_family/**` | 4-family staged replication + provenance |

All numbers here were re-checked against disk during the verification pass; see the project
`VERIFICATION.md` for the full ledger.

## Addendum batch — bifurcation, residual-norm null, early-commitment

Three further artifacts were shipped to back the layer-35 EOS bifurcation, the residual-norm
null, and the early-commitment finding. Each was sanitized for public release (the one absolute
cluster path in `norm_vs_projection/_summary.json::_meta.halt_direction_path` was replaced with a
`<placeholder>` token; the other two files contained no cluster identifiers). The public model id,
dataset names, layer indices, and claim IDs are preserved.

**L35 P(EOS) bifurcation (the "EOS is suppressed late, and the gap opens at the final layer"
claim).** `eos_logit_lens/logit_lens_results.json` holds the per-doc logit-lens EOS readout at
layers 16/24/28/35 over three cap-hit docs. It shows the EOS rank/probability stays in the
1e-7 to 1e-8 floor through L16-L28 and only at **L35** does EOS climb into the top-10/top-100 with
mean EOS-prob jumping by 3-4 orders of magnitude on some docs (e.g. `pgjw0227_p5` L35
mean_eos_prob 5.9e-4 and 459/1681 positions with EOS in top-100, vs ~1e-7 at L16-L28). The
bifurcation is **between pre-loop and in-loop** positions: at L35, `srgb0228_p2` carries
pre_loop_eos_prob 1.7e-4 but loop_eos_prob 4.1e-10 — the EOS signal that exists pre-loop is
crushed once the loop engages. This is the source for the L35 P(EOS) bifurcation claim.

**Residual-norm null (the "runaway is repetition-driven, not residual-norm-driven" claim).**
`norm_vs_projection/_summary.json` decomposes, per doc, the correlation of the halt direction
against (a) residual-stream norm and (b) a repetition-density (rd) feature, pre-loop vs in-loop.
Residual norm does **not** grow into the loop (norm_growth_factor 0.90-0.97, i.e. norm slightly
*shrinks*) and norm-vs-halt correlations are weak/negative (corr_norm_halt_full -0.24 to -0.54),
while repetition-density-vs-halt correlations are strong and positive (corr_rd_halt_full
0.74-0.90). Per-doc verdicts are REPETITION-DRIVEN ×2 / MIXED ×1 (modal REPETITION-DRIVEN). This
is the residual-norm null: the halt signal tracks repetition density, not a growing residual norm.

**Early commitment (the "cap-hit decision is detectable within the first ~50 generated tokens"
claim).** `seq_init_probe/_summary.json` is the multi-layer sequence-initiation probe: leave-one-out
AUC for a cap-hit-vs-control probe at 10 layers × 6 generation positions (gen-pos 50-800, N=12-16).
The pre-registered verdict is **PASS** — cap-hit is detectable as early as **gen-pos 50 at L24
(AUC 0.857)**, well before the gen-pos ~600 PCA transition, which is the visible *consequence* not
the cause. This substantiates the early-commitment claim and the "first-50-tokens monitor at L24"
production implication.

### Addendum file index

| Claim backed | Subdir / file | Role |
|---|---|---|
| L35 P(EOS) bifurcation | `eos_logit_lens/logit_lens_results.json` | per-doc logit-lens EOS rank/prob at L16/24/28/35 |
| Residual-norm null (repetition-driven) | `norm_vs_projection/_summary.json` | per-doc norm-vs-rep correlation decomposition |
| Early commitment (detectable by gen-pos 50) | `seq_init_probe/_summary.json` | multi-layer × multi-gen-pos LOO-AUC probe (verdict PASS) |
