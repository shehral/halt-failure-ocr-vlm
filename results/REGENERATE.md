# Results — what is checked in vs. regenerated

This directory holds the **small summary / CSV / provenance evidence files** that the
verification ledger's CONFIRMED claims point to (one subdir per experiment, named to
match the original run). Every file here was sanitized for public release: absolute
cluster paths, the HPC username, the SLURM account, scratch paths, the conda env name,
SLURM job/node IDs, and any internal-dashboard references were replaced with
`<placeholder>` tokens. The public model id `nanonets/Nanonets-OCR2-3B` + its public
revision, the public dataset names, method names, and the `CL-xx` claim IDs are
preserved for reproducibility traceability.

## Intentionally **not** checked in (regenerate via code)

To keep the publication repo lightweight, large binaries are **excluded**. They are
fully regenerable from the pinned model + manifests:

| Excluded artifact | Where it lived | Regenerate via |
|---|---|---|
| Activation `.pt` caches (`results/activations/**/hidden_states.pt`, ~hundreds of MB) | per-doc hidden-state caches | re-run the Job-2 activation-extraction script over `data/*/manifest.json` (layers `0,4,8,12,16,20,24,28,32,35`). |
| Full per-doc generations (`results/p1_*/`, `results/p1_trigger_v2/`, full `.txt`/`.tokens.pt`) | runaway-generation dumps | re-run `code/p1_cuda_resweep.py` / the trigger pass over the public corpus (`max_new_tokens=12000`, greedy, seed 0). |
| EOS-trajectory tensors (`demo_eos_failure/03_eos_trajectories/*.pt`) | per-step logit tensors | re-run `code/demo_eos_failure/05_scripts/regen_with_eos_logits.py` (the `.json` sidecars **are** checked in). |
| P16 per-cell patch tensors (`results/p16_component_resolved_short/<doc>/`) | component-resolved patch grid | re-run `code/p3_he_patch.py` over `data/p16_manifest_labeled.json` (L16/L20/L24 × {mlp,attn,block}); the aggregated `_summary.json` + `provenance.json` are checked in. |
| Model weights | HF cache | `nanonets/Nanonets-OCR2-3B` @ revision `c3886ff00bb037ce7da24988c9eafaf1fe2bed72`. |

## Checked-in evidence (by claim)

| Claim(s) | Subdir | Files |
|---|---|---|
| CL-42 / CL-42-global | `q1_combined/` | `per_layer_extracted.csv`, `_summary.json` |
| CL-16 | `p3_logit_lens/` | `_summary.json`, `provenance.json` |
| CL-49 | `p6_l0_sufficiency/` | `_summary.json`, `provenance.json`, `cohort_snapshot.json` |
| CL-36 | `p_poscontrol/` | `_summary.json`, `provenance.json` |
| CL-25 | `p7b_mirror/PCA_control/` | `_summary.json` |
| CL-05e | `p3_full_probe/` | `full_probe_results.json`, `provenance.json` |
| CL-21 | `p4_mirror/H7_cross_layer/` | `_summary.json` |
| CL-22 | `p6_mirror/B_calib_onset/` | `_summary.json` |
| CL-50 | `fcct_refresh/` | `per_class_block_contribution_aggregate.csv`, `provenance.json` |
| CL-11 | `p3_image_inpaint/` | `_summary.json`, `provenance.json` |
| CL-09e | `p2_pilot/` | `hc_baseline_results.json` |
| CL-48 | `p97_rebakeoff_boost_magnitudes/boost_30/` | `_comparison_table.json`, `provenance.json` |
| CL-39 | `p97_rebakeoff_boost_magnitudes/boost_6p91/` | `_comparison_table.json`, `provenance.json` |
| CL-02 | `p1_cuda_n40/` | `_summary.json`, `_provenance.json` |
| CL-12 | `p2_pilot/` | `b3_reverse_direction_summary.json` + 8 `b3_*_{baseline,subtract}.json` per-doc files |
| CL-19 / CL-20 | `p16_component_resolved_short/` | `_summary.json`, `provenance.json` |
| EOS-margin | `demo_eos_failure/03_eos_trajectories/` | 3 `*_eos_trace.json` sidecars (the `.pt` tensors are regenerated) |

All numbers here were independently re-checked against disk during the verification
pass; see the project `VERIFICATION.md` for the full ledger.
