# A Runtime Readout Monitor for Halt-Failure in OCR Vision-Language Models: Corpus, Early-Commitment, and the Honest Precision-Recall Trade-off of a Missing Halt Signal

Reproducibility repository for the **production / systems paper (P1)**.

> The deep representational-geometry results (per-class halt-direction geometry, the
> two-family bifurcation, the count-manifold, the PCA/crosscoder fine-tune signature, and
> build-vs-read) now live in a **companion representational-geometry paper** (repo
> `halt-failure-mechanism`). This paper keeps only the minimal mechanistic facts a
> practitioner needs to justify the monitor.

## Summary

Fine-tuned OCR vision-language models (OCR-VLMs) sometimes fail to stop: instead of
emitting the end-of-sequence (EOS) token, decoding runs to the `max_new_tokens` cap and
surfaces as phantom blank table rows, runaway token or phrase loops, and hallucinated
structural markers. We take the **systems view** of this *halt-failure* in
`nanonets/Nanonets-OCR2-3B` (a `Qwen2.5-VL-3B-Instruct` fine-tune) and argue the deficit is
best read as a **missing halt signal**: not a corrupted residual state, not mere repetition,
and not a single steerable layer. The contribution is a deployable artifact plus an honest
negative:

1. **Corpus + taxonomy** — 56 confirmed cap-hit positives across 12 populated surface
   classes, a device-induced (MPS↔CUDA) non-determinism that *shrinks* the stable positive
   set, and a domain-conditional trigger gradient (arXiv-table high; SEC-EDGAR / DocLayNet low).
2. **Where to monitor (early-commitment)** — the cap-hit outcome is linearly detectable in
   the first tens of generated tokens (best LOO AUC 0.857 at L24, gen-pos ~50, N=16),
   degrading by gen-pos 600. The monitor should read early at L24.
3. **The deficit is a missing halt signal** — the should-halt signal arrives late
   (logit-lens EOS gap most negative at L24, flips positive by L32; N=4+9); suppression is
   *relative* not absolute (chosen−EOS margin ~20 logits, N=1 illustration); the loop runs
   on a *coherent, not corrupted* residual (in-loop/pre-loop norm 0.90–0.97×, ruling out the
   ≥3× attention-sink/extreme-token inflation; N=3); and EOS is genuinely buried at loop
   onset (per-document median rank ~12,234, only 1/22 in the top 5).
4. **Single-site fixes do not install halting** — six pre-registered perturbation nulls all
   return FAIL/null: you cannot patch one site to make it stop. Whether the deficit is
   genuinely distributed or the protocol is under-powered is **not resolved** by these data,
   and the systems claim does not require resolving it.
5. **The deployable artifact** — a runtime readout monitor with an *honest precision-recall
   trade-off* (no free sweet spot): +6.91 EOS boost → 37.68% mean reduction, 0/10 control
   FP, 5/13 positives still at cap; +30 boost → 99.58% reduction but 10/10 control FP. Plus
   grey-noise image inpaint collapsing 5/5 named positives to 13–18-token EOS halts,
   motivating a vision-aware guardrail.

## What's verified (P1 production scope)

| Component | Status |
|---|---|
| Corpus + taxonomy (56/12; MPS↔CUDA shrink; domain trigger gradient) | verified on disk |
| Early-commitment probe — L24 AUC 0.857 @gen-pos 50 → 0.735 @600 (N=16) | verified on disk |
| Logit-lens EOS gap (minimal depth fact; N=4+9) | verified on disk |
| EOS-rank at loop onset (per-doc median ~12,234; 1/22 in top-5) | verified on disk |
| Coherent-not-corrupted residual norm null (0.90–0.97×; N=3) | verified on disk |
| L0 sufficiency null | verified on disk |
| Norm-scaled positive control | verified on disk |
| Six converging perturbation nulls — B3, P16, P10b, P19, P12, P22 (all FAIL/null) | **B3 + P16 verified on disk; P10b/P19/P12/P22 recorded with FAIL verdicts, primary files pending re-sync.** Nulls do not resolve distributed-vs-under-powered (esp. N=2). |
| Runtime monitor trade-off (37.68%/0-FP/5-cap @+6.91; 99.58%/10-FP @+30; N=13/10) | verified on disk (FP bound is N=10 controls; N≥600 long-form FP is future work) |
| Vision sufficiency / image inpaint (5/5 collapse to EOS; N=5) | verified on disk |
| Cross-family (10-strict / 11-permissive of 16; aggregate INSUFFICIENT_DATA) | verified on disk; **motivation only**, no generality claim |
| Per-class geometry, bifurcation, count-manifold, PCA/DFC fine-tune signature, build-vs-read | **migrated to the companion `halt-failure-mechanism` paper** — not a P1 result |

See [VERIFICATION.md](VERIFICATION.md) for the full claim-to-number-to-result-file ledger,
including the honest walk-backs (stale / pending entries) that are part of the result.

## Repository layout

```
README.md                 # this file
LICENSE                   # MIT
VERIFICATION.md           # verification ledger: claim (CL-xx) -> number -> results/ file
REPRODUCE.md              # step-by-step reproduction instructions (added with paper/)
UNDERSTANDING_GUIDE.md    # narrator / educational companion (added with paper/)
environment/
  requirements.txt        # pinned dependencies + model revision + lm_head tie-fix note
paper/                    # compiled PDF + LaTeX source
code/                     # extraction, probing, patching, and monitor scripts
results/                  # numerical result files referenced by VERIFICATION.md
data/                     # public dataset manifests (DocVQA and others)
review_corpus/            # browsable cap-hit failure corpus + self-contained viewer.html
```

> Note: `paper/` contains the compiled PDF, LaTeX source (main.tex), and references (references.bib).
> The `results/` paths in VERIFICATION.md are the source of truth
> for every load-bearing number in the paper.

## Model

- Model id: `nanonets/Nanonets-OCR2-3B` (a `Qwen2.5-VL-3B-Instruct` fine-tune; 36 decoder
  layers, hidden 2048, 16 attention heads, 2 KV heads).
- Pinned public revision: `c3886ff00bb037ce7da24988c9eafaf1fe2bed72`.
- See `environment/requirements.txt` for the **mandatory `lm_head` tie-fix** that every
  loader must apply (the public checkpoint omits `lm_head.weight` and relies on
  `tie_word_embeddings`).

## Data

The review corpus is drawn from public documents, including the public **DocVQA** dataset
(UCSF industry documents). Dataset manifests live under `data/`.

The cap-hit failures themselves are browsable by surface class under
[`review_corpus/`](review_corpus/README.md): open the self-contained
`review_corpus/viewer.html` in a browser to step through every cap-hit `doc_id`, its
runaway model output, and run metadata, filtered by failure-mode class.

## Autonomous reproduction on HPC

The experiments were run on a SLURM + GPU cluster. For a turnkey way to connect to a
cluster, set up the pinned environment, and submit smoke-tested jobs autonomously, see the
Claude Code Explorer skill:
<https://github.com/shehral/northeastern-explorer-autonomy-skill>.
Cluster-specific values (username, account, paths, environment name) are kept as
`<placeholder>` tokens consistent with that skill.

## License

MIT. See [LICENSE](LICENSE).
