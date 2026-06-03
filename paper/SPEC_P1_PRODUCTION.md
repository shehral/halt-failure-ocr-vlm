# P1 — Production / Systems Paper: RE-SCOPE SPEC

**Status:** Spec for re-scoping the existing `paper/main.tex` in place.
**Repo:** `<repo-root>`
**Working file:** `<repo-root>/paper/main.tex` (re-scoped in place; do NOT create a new dir)
**Companion:** P3 (deep representational paper) receives the migrated-out content. P3 scaffold lives in the companion repo `<companion-repo>`.
**Evidence map:** `<companion-repo>/docs/EVIDENCE_MAP.md`
**Date:** 2026-06-03

---

## 0. One-paragraph thesis (the re-scoped P1)

In a fine-tuned OCR-VLM, halt-failure (running to the token cap instead of emitting EOS) is
**a missing halt signal, not a corrupted state, not mere repetition, and not a single
steerable layer**. P1 is the applied/systems paper: it characterizes the corpus and trigger
distribution, localizes *where to monitor* via an early-commitment probe, presents a runtime
readout monitor with an **honest precision-recall trade-off** (no free sweet spot), and
treats the **converging perturbation nulls** as the systems-relevant conclusion — *the deficit
is a missing halt signal that single-site interventions cannot install*. The deep
representational geometry that explains *why* (per-class halt directions, bifurcation,
count-manifold, PCA fine-tune signature, build-vs-read, DFC) **migrates out to P3**; P1 keeps
only the minimal mechanistic facts a practitioner needs to justify the monitor.

**Distinct contribution vs the other three papers:**
- vs **P0 (cross-family generality):** P0 owns the 4×4 doc×model matrix as its headline. P1
  cites cross-family reproduction only as a one-paragraph "this isn't one fine-tune's bug"
  motivation, then abstains. P1's product is the monitor + trade-off + corpus, not generality.
- vs **P3 (representational):** P3 owns per-class geometry, bifurcation, count-manifold, PCA
  signature, build-vs-read, DFC. P1 owns the deployed artifact and the honest negative
  ("can't fix it by patching one site").
- vs **P4 (methodology):** P4 owns the audit harness, walk-back ledger, confound case studies.
  P1 inherits the *discipline* (pre-registered thresholds, file-traced numbers) but ships no
  methodology-tooling claims.

---

## 1. Proposed title + framing

**Primary:** *A Runtime Readout Monitor for Halt-Failure in OCR Vision-Language Models: Corpus,
Early-Commitment, and the Honest Precision-Recall Trade-off of a Missing Halt Signal*

**Backups:**
- *Halt-Failure Is a Missing Halt Signal: A Production Monitor and Its Honest Trade-off for OCR-VLMs*
- *When OCR-VLMs Won't Stop: Characterizing, Detecting Early, and Monitoring Runaway Decoding*

**Framing change from current main.tex:** drop "Behaves Like a Class-Structured Residual-Stream
Attractor" from the title — that is the P3 thesis. P1's organizing claim is **"the deficit is a
missing halt signal"**, evidenced by (a) the converging nulls (you can't install halting at one
site), (b) calibration-refuted / relative-suppression (EOS is buried, not narrowly under-promoted),
(c) the coherent-not-corrupted residual (the state is normal, the decision is wrong), and (d)
rep_penalty drift-to-salad if landed. The attractor metaphor is demoted to one hedged sentence
pointing at P3.

---

## 2. Re-scoped P1 section structure (what STAYS)

Each line: section title + the one thing it argues.

1. **Abstract** — Halt-failure is a missing halt signal; we characterize it, detect commitment
   early, ship a monitor with an honest precision-recall trade-off, and show single-site fixes fail.
2. **Introduction** — Production reliability bug; thesis = missing halt signal (not corruption,
   not repetition, not one layer); contributions are corpus + early-commitment localization +
   monitor + honest nulls. Drop the dual "two payoff results" attractor setup.
3. **The phenomenon and corpus** — Failure taxonomy (12 populated surface classes), corpus
   construction (56 confirmed positives from 60 cap-hit IDs), MPS↔CUDA non-determinism shrinks
   (not enlarges) the stable positive set, domain-conditional trigger gradient (arxiv_table high,
   sec_edgar/doclaynet low). **PROMOTE the domain-trigger gradient in from omitted (thread 1b).**
4. **Where to monitor: the runaway is committed early** — Early-commitment probe (L24, AUC 0.857
   at gen-pos ~50, degrading to 0.735 by gen-pos 600). This is the *systems* justification for
   *where and when* the monitor should read. KEEP and elevate to its own top-level section.
5. **The deficit is a missing halt signal** (NEW consolidating section) — fold three converging
   diagnostic facts that a practitioner needs:
   (a) **Calibration refuted / EOS genuinely missing** (median rank ~12,234, only 1/22 in top-5).
   (b) **EOS suppression is relative not absolute** (chosen−EOS margin ~20 logits; EOS logit stays
       positive; N=1-vs-N=1, illustrative).
   (c) **Coherent, not corrupted** (in-loop/pre-loop residual norm 0.90–0.97×, rules out
       attention-sink/extreme-token corruption that needs ≥3×; N=3). Affirmative null.
   (d) (if landed) **rep_penalty drift-to-salad** — escapes the repeat-attractor into off-domain
       vocab without ever emitting EOS; direct evidence the deficit is missing HALT, not repetition.
   Logit-lens "should-halt arrives late" (EOS gap −0.419 @L24, +0.481 @L32) STAYS here as the
   minimal depth fact (N=4+9), framed as motivation for the monitor's read site, NOT as attractor
   geometry.
6. **Single-site fixes do not install halting: a converging series of nulls** — the six
   pre-registered perturbation nulls (B3, P16, P10b, P19, P12, P22) + L0 sufficiency null +
   norm-scaled positive control. KEEP IN FULL — this is P1's load-bearing systems conclusion
   ("you cannot patch one site to make it stop"). Keep the honest under-power hedge and the
   controls-are-suspect caveat.
7. **Production monitor: an honest precision-recall trade-off** — the L24 readout monitor;
   +6.91 boost → 37.68% mean reduction, 0/10 control FP, 5/13 positives still at cap; +30 boost →
   99.58% reduction but 10/10 control FP. No free sweet spot. Plus image-content sufficiency
   (grey-noise inpaint collapses 5/5 named positives to 13–18-token EOS halts) as the
   vision-aware-guardrail motivation. KEEP — this is the deliverable.
8. **Cross-family: motivation only, verdict abstains** — compress to ONE short paragraph:
   reproduces on 10/16 doc-model cells across four families, aggregate verdict abstains
   (INSUFFICIENT_DATA), "not unique to this fine-tune," then point to P0 for the real treatment.
   Do NOT make a generality claim.
9. **Related work** — keep the production-facing threads (degeneration/non-termination,
   OCR/structural hallucination, attention-sink-as-competing-mechanism-we-set-aside, single-
   direction-steering-that-comes-back-null, cross-modal grounding). TRIM the count-manifold and
   crosscoder/DFC paragraphs down to one sentence each pointing to P3.
10. **Limitations and walk-back** — keep the production-relevant walk-backs (under-power vs
    distributed is unresolved; nulls at small N; controls suspect; cross-family abstains;
    vision-attention N=1; loop-onset framing). Move the fine-tune-origin Cohen's-d walk-back to P3.
11. **Conclusion** — missing halt signal; deployable monitor with honest trade-off; single-site
    fixes fail; point representational "why" to P3.

---

## 3. What MIGRATES OUT to P3 (remove from P1)

These are the DEEP representational threads. Remove from `main.tex`; they become P3's load-bearing
results (P3 scaffold already anticipates most). Each line: P1 location → P3 home.

| # | Content to remove from P1 | Current P1 location | P3 home (per p3_iclr_outline.md) | Evidence file |
|---|---|---|---|---|
| R1 | **Per-class halt directions are class-specific & content-decoupled** (AUC 0.876–0.983; distinct peaks latex@L8 / filled@L24 / bare@L20; max cosine 0.235; 3/10 global-decoupling layers) + Fig per_class_auc + Fig null_layer_profile | §Mechanism 5.1 + Figs | P3 §6 Results (HEADLINE, CL-42) | `results/q1_combined/_summary.json`, `per_class_pairwise_cosines.csv` |
| R2 | **Build-vs-read (FCCT)** filled-cell built@L20 read@L24, cosine +0.042 vs −0.010 | §Mechanism 5.2 "Build versus read" | P3 §7 Mechanism story (CL-50 candidate) | `results/fcct_refresh/` |
| R3 | **Halt-direction per-position trajectories** (positives ramp into sustained high-halt regime) + Fig halt_trajectories | §Mechanism 5.2 (figure) | P3 §6 (illustrates attractor reading) | `results/q1_combined/` (LODO trajectories) |
| R4 | **Opposite-sign EOS bifurcation @L35** (empty-row SUPPRESS 16×–4.2e5×; count-manifold AMPLIFY 508×; N=3) | §Mechanism 5.6 "bifurcation" | P3 §7 (the two-regime complication of the attractor; CL-03) | `results/p2_pilot/logit_lens_results.json` |
| R5 | **PCA fine-tune signature / Cohen's-d crossover** (pos/ctl ratio 3.79/2.74/2.78 FT vs 1.12/0.78/0.92 base @L16/L20/L24, 2.0× bar) | abstract + §RelatedWork (DFC para) + §Limitations item 5 | P3 §6 triangulation (CL-25/CL-27) | `results/p2_pilot/hb_pca_results.json` + base comparison (RE-VERIFY scratch) |
| R6 | **DFC crosscoder feature-timing** (2 FT-introduced features peak-fire @crossover; first DFC on a VLM) | §RelatedWork (crosscoder vocab) | P3 §7 / §4 (CL-26, novelty) | `results/p4_mirror/DFC_features/_summary.json` |
| R7 | **Count-manifold geometry** (running count as 1-D residual manifold; LOO R2 0.45@L16, 0.79@L4; walked back from in-sample 0.90) | §RelatedWork (counting-manifold para) | P3 §7 (CL-06, dominant-family geometry) | `results/p2_pilot/hb_pca_results.json` + `hb_pca_nonzero_results.json` |
| R8 | **2-family subspace sub-structure** (bare-word orthogonal everywhere; latex+filled share L12–L28) | not yet in P1 (omitted) — keep omitted, route to P3 | P3 §6 (refines CL-42) | `results/q1_combined/per_class_pairwise_cosines.csv` |
| R9 | **Distributed-detection cross-layer sweep** (AUC≥0.84 every layer; L16=0.929) | implicit in P1's null framing | P3 §6 (CL-21, the explicit geometry) — RE-VERIFY before citing | `<scratch>/H7_cross_layer` (scratch-purged) |
| R10 | **Temporal precursor** (internal halt-shift precedes surface loop by ~130 tok; depth-ordered) | not in P1 — route to P3 | P3 §7 (CL-24/CL-29) — RE-VERIFY | `results/p5b_mirror/_summary.json` (precursor only) |

**Note on the attractor metaphor itself:** the *title-level* "class-structured residual-stream
attractor" framing migrates to P3. P1 keeps at most one hedged sentence: "the representational
geometry that organizes these per-class signals is treated in a companion paper; here we take the
systems view that the deficit is a missing halt signal." Remove the attractor framing from P1's
abstract, intro thesis, and conclusion.

---

## 4. What STAYS in P1 (the keep-list, explicit)

- **Corpus + taxonomy** (12 populated classes; 56/60 positives; CUDA-shrink disclosure) —
  `review_corpus/00_INDEX.csv`, `results/p1_cuda_n40/`, `results/p1_trigger_v2/`. **DISK-OK.**
- **Domain-conditional trigger gradient** (PROMOTE from omitted) — arxiv_table high vs
  sec_edgar/doclaynet low. `results/p1_cuda_n40/` (E-HPC-37). **DISK-OK.**
- **Early-commitment probe** (L24 AUC 0.857 @gen-pos 50 → 0.735 @600; N=16) —
  `results/p21_seq_init_probe/_summary.json`. **DISK-OK / small-N.**
- **Calibration refuted** (EOS rank ~12,234 doc-level; 1/22 in top-5; effective N=22) — in current
  main.tex; Phase-6 Task B retest. **DISK-OK.**
- **EOS relative-suppression** (margin ~20 logits; N=1-vs-1, illustrative) — `demo_eos_failure/`.
  **DISK-OK / N=1.**
- **Coherent-not-corrupted norm null** (0.90–0.97× vs ≥3× bar; N=3) — `results/p2_pilot/hd_pilot_results.json`.
  **DISK-OK / small-N.** PROMOTE to its own subsection under "missing halt signal."
- **Logit-lens depth (minimal)** (EOS gap −0.419@L24 → +0.481@L32; N=4+9) —
  `results/p3_logit_lens/_summary.json`. **DISK-OK / small-N.** Keep as monitor-read-site
  motivation, not as geometry.
- **Six converging perturbation nulls** + L0 sufficiency null + norm-scaled positive control —
  `results/p16_component_resolved_short/_summary.json`, `results/p2_pilot/b3_reverse_direction_summary.json`,
  `results/p6_l0_sufficiency/_summary.json`; (P10b/P19/P22 RE-VERIFY, scratch-purged). **mixed.**
- **Production monitor trade-off** (37.68%/0FP/5-cap @+6.91; 99.58%/10/10FP @+30; N=13/10) —
  `results/p97_rebakeoff_boost_magnitudes/boost_6p91/_comparison_table.json`. **DISK-OK / N=13.**
- **Image-content sufficiency** (grey-noise collapses 5/5 to 13–18-tok EOS) —
  `results/p3_image_inpaint/_summary.json`. **DISK-OK / N=5.**
- **Cross-family (motivation only, abstains)** — `results/p3_cross_family_p2/p2_outcome_matrix.md`
  (strict 10/16; permissive 11/16; aggregate INSUFFICIENT_DATA). **DISK-OK / small-N.**
- (Optional) **rep_penalty drift-to-salad** if re-synced — direct missing-halt evidence (CL-09a).
  **RE-VERIFY (HPC-only).** Add only if synced; otherwise cite qualitatively.

---

## 5. Standalone readiness: STRONG (with honest gaps)

P1 stands alone strongly because every load-bearing systems claim has a local DISK-OK evidence
file: the corpus index, the early-commitment probe, the monitor comparison table, and the
image-inpaint summary are all locally verified at workshop scale. The honest negative (six nulls)
is the conceptual spine and is locally reproducible for the two anchor nulls (B3, P16); the other
four are RE-VERIFY (scratch-purged) and must be re-synced before external share but are recorded
with their pre-registered FAIL verdicts.

**Honest gaps / thin spots to disclose in-paper:**
- Monitor and image-inpaint are workshop-N (13/10 and 5); the N≥600 long-form FP characterization
  (CL-14 gate) is future work — state the FP bound is N=10-controls, not N≥600.
- Four of six nulls are RE-VERIFY (P10b/P19/P22/and P12's full cohort were on scratch). Keep the
  "all verified on disk" language ONLY for B3+P16; soften the rest to "recorded with FAIL verdicts;
  primary files pending re-sync."
- Norm-null, relative-suppression, bifurcation-adjacent facts are N=1–N=3 — keep the illustrative
  framing already in main.tex.
- Under-power vs distributed remains unresolved — this is fine for P1 because P1's claim is the
  systems one ("single-site fix fails"), not the mechanistic one ("distributed"). The mechanistic
  adjudication is explicitly P3/future-work.
- Cross-family count is criterion-dependent (10/16 strict vs 11/16 permissive) — disclose both.

---

## 6. Reproduction scope this repo needs (P1)

To reproduce P1 standalone, the repo needs only the production/systems evidence subset:
- `review_corpus/00_INDEX.csv` + `results/p1_trigger_v2/` + `results/p1_cuda_n40/` (corpus + trigger)
- `results/p21_seq_init_probe/_summary.json` (early-commitment)
- `results/p3_logit_lens/_summary.json` (depth, minimal)
- `results/p2_pilot/hd_pilot_results.json` (norm null) + `demo_eos_failure/` (relative suppression)
- the six-null set: `results/p16_component_resolved_short/`, `results/p2_pilot/b3_reverse_direction_summary.json`,
  `results/p6_l0_sufficiency/`, and (re-sync) p10b/p19/p22/p12 summaries
- `results/p97_rebakeoff_boost_magnitudes/boost_6p91/_comparison_table.json` + `boost_30/` (monitor)
- `results/p3_image_inpaint/_summary.json` (image sufficiency)
- `results/p3_cross_family_p2/p2_outcome_matrix.md` (cross-family, motivation only)
- `fix/halt_monitor*.py` + `fix/halt_direction_L24.pt` (the deployable artifact)

P1 does NOT need: `results/q1_combined/` (per-class), `results/fcct_refresh/` (build-vs-read),
`results/p2_pilot/hb_pca_results.json` (count-manifold/PCA), `results/p4_mirror/DFC_features/`
(DFC), `results/p2_pilot/logit_lens_results.json` (bifurcation) — those go with P3.

---

## 7. Concrete edit checklist on main.tex

1. Retitle (drop "attractor"); rewrite abstract around "missing halt signal" + monitor + nulls.
2. Rewrite Intro thesis (remove dual attractor payoff; state missing-halt thesis + 4 contributions).
3. §Phenomenon: add the domain-trigger-gradient paragraph (PROMOTE thread 1b).
4. Promote §5.4 early-commitment to a standalone top-level "Where to monitor" section.
5. Create consolidating "The deficit is a missing halt signal" section: move in calibration-refuted,
   relative-suppression, coherent-not-corrupted (PROMOTE from omitted), minimal logit-lens depth.
   (optional) add rep_penalty-salad if synced.
6. **Remove** §5.1 per-class geometry + both figures (per_class_auc, null_layer_profile) → P3.
7. **Remove** §5.2 build-vs-read + halt_trajectories figure → P3.
8. **Remove** §5.6 bifurcation → P3.
9. **Remove** PCA/Cohen's-d crossover from abstract, Related Work DFC para, Limitations item 5 → P3.
10. Keep §Causal nulls in full (this is now P1's spine, reframed as "single-site fixes don't install halting").
11. Keep §Production monitor in full.
12. Compress §Cross-family to one paragraph (abstain; point to P0).
13. Related Work: trim count-manifold + crosscoder paras to one sentence each pointing to P3.
14. Limitations: drop fine-tune-origin Cohen's-d item (→P3); keep under-power, small-N nulls,
    controls-suspect, cross-family-abstains, vision-N=1, loop-onset.
15. Conclusion: rewrite around missing-halt + monitor + nulls; one sentence to P3 for the "why."
16. Recompile with `tectonic paper/main.tex`; confirm orphaned figure references (per_class_auc,
    null_layer_profile, halt_trajectories) are removed.
