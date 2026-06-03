# Halt-Failure Behaves Like a Class-Structured Residual-Stream Attractor in Fine-Tuned OCR Vision-Language Models

Reproducibility repository for the paper.

## Summary

Fine-tuned OCR vision-language models (OCR-VLMs) sometimes fail to stop: instead of
emitting the end-of-sequence (EOS) token, decoding runs to the `max_new_tokens` cap and
surfaces as phantom blank table rows, runaway token or phrase loops, and hallucinated
structural markers. We study this *halt-failure* phenomenon in `nanonets/Nanonets-OCR2-3B`
(a `Qwen2.5-VL-3B-Instruct` fine-tune) and advance the **hypothesis** that it is best
understood not as a single broken layer but as something that **behaves like a stable,
class-structured attractor in the residual stream**. "Attractor" is a labeled hypothesis
and metaphor for the evidence pattern, not a mechanism we have causally isolated.
Four lines of evidence motivate the reading: (1) per-class halt directions are linearly
decodable across the decoder (per-class AUC 0.876-0.983) and are content-decoupled on a
pre-registered distinctness-plus-decoupling test; (2) the network computes a downstream
"should-halt" signal that arrives too late to override an upstream continuation state
(logit-lens EOS gap most negative at L24, flipping positive by L32), and EOS suppression is
*relative* not absolute (continuation tokens out-score EOS by ~20 logits); (3) the halt
token is genuinely absent at loop onset (median EOS rank ~10,500); and (4) a converging
series of single-direction and single-layer causal-perturbation tests fail to move the token
count. We are explicit that this last line is a **null that does not settle the
mechanism**: all single- and multi-site perturbations we ran returned null at this
protocol's power, and whether the mechanism is genuinely distributed or the protocol is
under-powered at the halt-relevant locus is **not resolved** by these data — the mechanism
remains unidentified. The norm-scaled positive control (0 of 3 cap-hit positives escape
the cap; its 4th doc is a clean-halt control that itself collapsed under the patch, a
norm-shock signature that blunts the rebuttal) rebuts only the strongest
(protocol-is-inert) form of the under-powered objection, not the weaker form that the
protocol cannot perturb the halt-relevant direction at the halt-relevant locus. We close
with an honest production trade-off (a runtime monitor) and causal evidence that vision is
load-bearing.

## What's verified

| Component | Status |
|---|---|
| Per-class halt-direction AUC (CL-42) | verified on disk |
| Logit-lens EOS gap (CL-16) | verified on disk |
| EOS-rank at loop onset (CL-22) | verified on disk |
| L0 sufficiency null (CL-49) | verified on disk |
| Norm-scaled positive control (CL-36) | verified on disk |
| B3 reverse-direction necessity null (CL-12) | verified on disk |
| Component-resolved FCCT patch (CL-19/20) | verified on disk |
| Runtime monitor trade-off (CL-39 / CL-48) | verified on disk |
| Vision sufficiency / inpaint (CL-11) | verified on disk |
| Causal nulls: B3 + P16 verified on disk; P10b/P19v2/P12v3/P22 recorded on cluster, not synced | 2 verified + 4 unsynced; see VERIFICATION.md |
| Cross-family counts (CL-47 / CL-13) | pending re-run; see VERIFICATION.md |

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

## Autonomous reproduction on HPC

The experiments were run on a SLURM + GPU cluster. For a turnkey way to connect to a
cluster, set up the pinned environment, and submit smoke-tested jobs autonomously, see the
Claude Code Explorer skill:
<https://github.com/shehral/northeastern-explorer-autonomy-skill>.
Cluster-specific values (username, account, paths, environment name) are kept as
`<placeholder>` tokens consistent with that skill.

## License

MIT. See [LICENSE](LICENSE).
