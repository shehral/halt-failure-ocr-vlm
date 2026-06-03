# Understanding Guide: Why a Fine-Tuned OCR Vision-Language Model Forgets to Stop

*An educational companion for a strong ML reader who is new to mechanistic interpretability on vision-language models (VLMs).*

This guide walks one project end to end: the problem, the vocabulary you need, how the thinking actually evolved (including the parts that did not work), how to read the methods, **how every load-bearing number was verified against on-disk data**, and a self-check at the end. The goal is not to convince you of a result. It is to give you the mental model to read the paper critically and to see which claims are load-bearing versus preliminary.

**One-sentence thesis (a hypothesis, not a settled mechanism).** In a fine-tuned OCR vision-language model, the failure to halt generation is not a single broken neuron or layer. It *behaves like* a stable, class-structured pattern in the residual stream — an *attractor* — that the network falls into and cannot climb out of, even though it separately computes a "you should stop now" signal that simply arrives too late to win. We label "attractor" a hypothesis and metaphor: our causal nulls are consistent with a genuinely distributed mechanism but equally consistent with a perturbation protocol too weak to reach the halt mechanism, and the data presented here do not distinguish the two (the mechanism remains unidentified).

**Model under study.** `nanonets/Nanonets-OCR2-3B` (a 3B-parameter OCR fine-tune of an instruction-tuned vision-language backbone), pinned to its public revision. Evaluation documents are drawn from the public DocVQA dataset. Reproducibility traceability is carried through claim IDs of the form `CL-xx` throughout, matching the verification ledger.

> **A note on reproducibility tokens.** Cluster-specific details (login user, SLURM billing account, scratch and project paths, conda environment name) have been replaced with `<placeholder>` tokens following the published Explorer skill convention: the project allocation is `/projects/<group>/<user>`, jobs bill to `<slurm-account>`, and the pinned environment lives at `$ALLOC/conda_envs/<conda-env>`. The science (model id, dataset, methods, claim IDs, numeric results) is preserved exactly.

---

## 0. How to use this guide

Read Sections 1 to 3 to build the mental model. Read Section 4 to learn the method vocabulary. Read Sections 5 and 6 to follow the *research arc* and the *verification journey* (which numbers survived, which were caught as stale and excluded, and how two missing results were recovered). Use Section 7 (glossary), Section 8 (decision log), and Section 9 (comprehension questions) as reference and self-check.

The single most important habit this guide tries to instill: **separate correlational claims from causal claims, and separate confirmed-on-disk numbers from preliminary or directional ones.** The project's honesty is its backbone, and reading it well means tracking that line.

---

## 1. The field landscape: why OCR-VLMs over-generate

### What the model is supposed to do

An OCR vision-language model takes a document image and transcribes it into text. Internally it is an autoregressive decoder: it emits one token at a time, and at every step it chooses among tens of thousands of vocabulary tokens which one comes next. Among those tokens is a special end-of-sequence (EOS) token. When the model has finished transcribing the document, the *right* behavior is to make EOS the most probable next token, emit it, and stop.

### The halt decision

The "halt decision" is exactly this moment: the model recognizes "the document content is exhausted, I should emit EOS now." This is a real computation, not a free reflex. The model has to represent something like "I have transcribed everything in the image" and convert that into a high logit for the EOS token.

### How it fails

Sometimes the model never makes that decision. It keeps emitting tokens long after the document content is gone. In practice this is a runaway decode that only stops at a hard token cap (here, 12,000 tokens). This is the **halt-failure** or **infinite-generation** phenomenon. Common surface forms:

- a table cell value repeated forever (filled-cell repeat),
- a LaTeX command or math expression looped over and over (LaTeX loop),
- a single bare word repeated indefinitely (bare-word repeat),
- HTML phantom rows (`<tr><td></td></tr>` emitted endlessly), tag spam, punctuation runs, constant-value repetition.

These look like different bugs on the surface. The project's central question is whether they share an underlying mechanism.

### Why fine-tuning matters

A key framing choice: the base backbone and the OCR fine-tune are *the same architecture*. If the fine-tune halts badly and the base model does not, then halt-failure is something the fine-tuning *introduced*, not an inherent property of the architecture. That distinction shapes the whole investigation: the fix should target what fine-tuning changed, not the pretrained backbone.

### Why this is hard to study

You cannot read off "the model decided to keep going" from the output. The decision lives in the internal activations. Studying it requires tools that inspect and manipulate hidden states as generation happens. That is mechanistic interpretability, and Section 4 gives you the vocabulary.

---

## 2. The full research arc at a glance

Before the details, here is the shape of the whole project. Each stage is expanded later.

1. **Phenomenon.** Confirm runaway generation is real, reproducible, and stable on the target hardware (not a measurement artifact).
2. **Taxonomy and corpus.** Catalog the surface classes of failure and build a confirmed positive set with controls. Disclose that the set is selection-biased and device-sensitive.
3. **Localization.** Use linear probes, logit lens, and block-contribution decomposition to find *where* a halt signal lives, where it is built, and where it is read. This is correlational.
4. **Converging causal nulls.** Try repeatedly to flip the behavior by editing one direction or one layer. Each attempt fails. A norm-scaled positive control shows the protocol was not *inert* (it visibly moved content), but it does **not** prove the protocol could have perturbed the halt-relevant direction at the halt-relevant locus, so the nulls remain ambiguous between "distributed mechanism" and "under-powered protocol."
5. **The attractor *hypothesis*.** The framing that fits the full evidence pattern: halt-failure *behaves like* a *distributed, fine-tune-introduced residual-stream attractor*, not a single-layer bug. This is a labeled hypothesis and metaphor, not a causally-isolated mechanism. The nulls are *consistent with* "distributed" but are equally consistent with an under-powered protocol; the data do not distinguish the two, and the mechanism remains unidentified.
6. **Cross-family reproduction (modest, no generality claim).** Under a strict `stop_reason == max_new_tokens` criterion the bug reproduces on **10 of 16 doc-model cells across 4 families** (per-family 3/4, 1/4, 3/4, 3/4), but the matrix is small and the directory-level aggregate verdict is `INSUFFICIENT_DATA`, so the paper draws **no strong generality claim** from it; a permissive near-cap relabel would shift the count to 11/16.
7. **Production fix trade-off.** Characterize a runtime EOS-boosting monitor as an honest precision-recall curve, and show via grey-noise image inpainting that image content is sufficient to sustain the loop (all 5/5 named positives collapse to short EOS halts), motivating a vision-aware guardrail.

The thesis is unusual in that **its strongest evidence is negative**: a series of causal perturbations that *fail*. In an attractor framing, a converging set of nulls is what you would expect — but it is equally what you would expect from a perturbation protocol too weak to reach the halt mechanism. The guide is careful never to read the nulls as *proof* of distribution; they motivate the attractor hypothesis without settling it.

---

## 3. The central metaphor: a residual-stream attractor

An **attractor** is a stable state that a dynamical system gets pulled into and stays in. The *hypothesis* is that once the model's residual stream enters the halt-failure region, ongoing generation reinforces that state. Each new repeated token nudges the residual stream right back into the loop. A "should-stop" computation does develop downstream, but it cannot overpower the upstream continuation state that is already locked in. Read "attractor" as a metaphor that organizes the evidence, not as a mechanism the project has causally isolated.

Why reach for "attractor" rather than "broken layer"? Because every attempt to push the model out by editing one place fails. If the failure lived in one site, surgically editing that site *might* fix it; it does not. That is **consistent with** a failure that is distributed across the network — but, crucially, it is **equally consistent with** a perturbation protocol that is simply too weak to reach the halt mechanism. A converging set of nulls cannot, on its own, tell those two apart. So the honest statement is: all single- and multi-site perturbations returned null at this protocol's power, and whether the mechanism is genuinely distributed or the protocol is under-powered is **not resolved** by these data — the mechanism remains unidentified. The project is also careful about which of these nulls are fully verified versus recovered late versus preliminary (Sections 5 and 6).

Why "class-structured"? Because the readable halt signal is not one monolithic direction. Of the 12 populated surface classes, only three (LaTeX command loops, filled-cell repetition, bare-word repetition) accumulated the >=50 halt-state tokens (`min_tokens_per_class = 50`) needed to fit reliable per-class probes, so the mechanism analysis is restricted to these three (`n_classes_with_probe = 3`). Across those three it is **three class-conditioned directions**, not one: the direction for a filled-cell loop is not the direction for a bare-word loop, and they peak at different depths.

---

## 4. Key concepts glossary (method vocabulary)

These are the load-bearing terms. Each is defined plainly, then connected to the project.

### Residual stream

In a transformer, every layer reads from and writes to a shared running vector, the **residual stream**. Think of it as the model's working memory: layer 1 writes its contribution, layer 2 adds on top, and so on. By the final layer, the residual stream is decoded into next-token probabilities. Almost everything the model "knows" at a generation step is encoded as a direction (a vector) in this stream.

### Linear probe and halt direction

A **linear probe** is a logistic-regression classifier trained on residual-stream activations. If it separates two conditions with high accuracy, the distinguishing feature is *linearly decodable* at that layer. The probe's weight vector is a **direction** in the residual stream.

A **halt direction** is the probe vector separating *positive* documents (the model ran away) from *control* documents (the model halted cleanly). The project finds these are **per-class**: a separate direction per surface class, each linearly decodable with high accuracy, each peaking at a *different* layer. Only the three classes that cleared the >=50-halt-token floor (LaTeX command loops, filled-cell repetition, bare-word repetition; `n_classes_with_probe = 3`) were probed, so this is a three-direction finding, not all 12 populated classes. A single "global" direction that ignores class is measurably weaker. The takeaway: halt-failure has *class structure*; it is not one signal (claim **CL-42**).

### Activation patching and path patching

**Activation patching** is the workhorse causal test. You take an activation (say the residual stream at layer 20) from one run and splice it into another, then check whether behavior changes. If patching layer L changes the output, layer L is *causally* involved, not just correlated. This moves you from "the probe can read a signal here" to "this signal actually drives the behavior."

**Path patching** is a refinement: instead of patching a whole layer, you patch a specific pathway (the output of one component feeding into one later component) to find *which route* carries the signal.

In this project the patching results are mostly *nulls*: editing the halt direction at layer 0 produces a **0.0 percentage-point** change in cap-hit rate (the pre-registered bar for "this worked" was at least a 40-point change, so this is a clear FAIL for single-direction sufficiency; claim **CL-49**). That null is informative but ambiguous: it is what you expect if the failure is a distributed attractor, *and also* what you expect if the perturbation protocol cannot reach the halt mechanism. The null alone does not pick between those.

### Logit lens

The **logit lens** decodes an *intermediate* layer's residual stream as if it were the final layer, by applying the model's final normalization and output projection early. It answers "what would the model predict if it stopped thinking at layer L?" The project tracks the EOS token's logit across layers. The finding: the gap in EOS logit between positive and control runs is most negative around layer 24 (the model most suppresses EOS there), then flips positive by layer 32. A "should-halt" signal does develop late. It just arrives too late to win (claim **CL-16**).

### Block-contribution decomposition (build vs read)

This decomposes a layer's residual update into block contributions and measures the cosine alignment between a layer's block output and the per-class halt direction. A positive alignment means the layer is *building* the halt representation; a negative or near-zero alignment at a later layer means that layer *reads* (consumes) rather than constructs it. For filled-cell-repeat, the halt direction looks built at layer 20 and read at layer 24 (claim **CL-50**) — but treat this as a **preliminary candidate**. It holds under the `all_docs` aggregation (all 18 cached docs pooled), only 3 of which carry a filled-cell label; under the `doc_argmax_only` scope (N=3) the cosine signs persist but the alignment ratio flips sign (-2.32 vs +1.30 at L20). The block contribution is also cumulative over a four-layer sampled-layer gap (e.g. L16->L20), the magnitudes are tiny (~0.04), and MHSA is not separated from FFN.

### Cross-modal attention mass

A VLM attends across two modalities: text tokens and image tokens. **Cross-modal attention mass** measures how much of the model's attention, at a given layer, points from generated text back to the image. If the model stops looking at the image, it has nothing new to transcribe and may fall into a content-free loop. The project has a *correlational* signal here, but it is **N=1 versus N=1** (claim **CL-09e**) and explicitly reserved for larger-N work. The collapse is also **band-localized rather than monotone**: the same document pair shows the positive with *higher* image attention than the control at layer 8 (1.10x) and layer 35 (1.37x), so the L20-to-L24 dip is suggestive only.

### Image inpainting (grey-noise replacement)

Replace the input image with grey noise and re-decode. If runaway documents collapse to short clean EOS halts, the image *content* is sufficient to sustain the loop. They do: all 5 of 5 named cap-hit positives (`arxiv_table_000266`, `docvqa_jqbg0227_p1`, `docvqa_kshm0227_p7`, `docvqa_srgb0228_p2`, `funsd_105_105`) collapse to a 13- to 18-token EOS-terminated halt (claim **CL-11**, `results/p3_image_inpaint/_summary.json`, `n_vision_load_bearing=5`). With N=5 we report this qualitatively, with no Wilson interval. We say "image content is sufficient to sustain the loop" rather than "vision is causally load-bearing" because grey noise confounds two things — removing visual evidence and destroying readable content — and the discriminating run (swapping in a *different readable* document) is left to future work. This is still the stronger vision-side result; the attention-mass finding above is the weaker one.

### Fine-tune signature

A **fine-tune signature** is a measurable property present in the fine-tuned model but absent or much weaker in the base, letting you attribute a behavior to fine-tuning. Here it is a separation in a principal-component crossover analysis: the ratio of positive-class to control-class effect size (absolute Cohen's d) in the fine-tune is 3.79 at layer 16, 2.74 at layer 20, and 2.78 at layer 24, all above the pre-registered 2.0x threshold (claim **CL-25**). The base-model comparison has now been run and persisted on the same Qwen2.5-VL-3B backbone (claim **CL-25-base**, `p18_base_archetype/_summary.json`): over the base model's own cap-hit documents (N=10 base cap-hits / 15 controls) the ratio triple is 1.12 / 0.78 / 0.92 at layers 16 / 20 / 24, all near 1x and all below the 2.0x bar. So the base backbone shows no separable halt geometry even where it itself runs to the cap, while the fine-tune exceeds the bar at every layer. This is direct evidence that fine-tuning *introduces and sharply amplifies* the separable geometry, not a merely directional read.

### EOS suppression: relative, not absolute

A subtle but central reframe. EOS is **not** suppressed in an absolute sense. The runaway document's raw EOS logit (+10.76) is actually *higher* than a clean control's (+10.07). This is a single illustrative **N=1-vs-N=1** trace (positive `docvqa_srgb0228_p2`, traced to a 3000-token cap rather than the paper-wide 12,000-token cap; control `docvqa_fhxn0226_p2`, which halts on its own at 193 tokens). What kills EOS is that continuation tokens out-score it by roughly 20 logits, crushing P(EOS) to a mean of roughly 1e-7 (ranging from about 1e-5 down to 1e-16 over the logged steps). So it is a *relative-margin* failure, not an "EOS logit went to negative infinity" failure. And calibration is genuinely refuted, not a near-miss. Here we report at the **document level** rather than pseudoreplicating: the 110 onset records are five contiguous records per document, so the effective N is 22 documents. Per document the median EOS rank is roughly 12,234, no document has its median rank inside the top 5, and only 1 of 22 documents ever puts EOS in the top 5 at any logged onset position (the pooled record-level figure is a median rank around 10,500). EOS is truly absent from the top of the distribution (claims **EOS-margin** and **CL-22**).

### Coherent, not corrupted: the residual-norm null

A second "what the failure is / isn't" reframe, and the one that lets us set aside the most actionable competing mechanism. A leading rival account of degenerate generation is the **attention-sink / extreme-token** mechanism (StreamingLLM / DeepMind line): the loop supposedly rides a *corrupted* residual state where a few tokens accumulate anomalously large activations and the residual L2 norm blows up. That account makes one sharp, falsifiable prediction — once the loop is entered, the in-loop residual norm should be **much larger** than the pre-loop norm, with extreme-token reports placing the inflation at **>=3x**. We measure exactly this at the target layer (L24) for the three cap-hit documents and find the opposite. The in-loop/pre-loop mean residual-norm ratio is **0.96x** (kshm: 86.61 in-loop vs 89.97 pre-loop), **0.90x** (srgb: 84.04 vs 93.66), and **0.97x** (pgjw: 84.30 vs 87.30) — in every case the norm *shrinks slightly* on loop entry, far below the >=3.0x corruption regime. Worse for the sink account, the norm does not merely fail to grow with the halt score, it **anti-correlates** with it (full-trace correlation -0.236 / -0.539 / -0.298): a *bigger* norm reads as *more*-halt, not the runaway never-halt inflation a sink-corruption account predicts. So the loop is running on a **normal-looking residual whose halt-relevant projection is miscomputed**, not on a blown-up, information-losing sink state. The failure is a **miscomputed decision on a coherent state, not a corrupted state**, and this is an *affirmative* null — it rules out the attention-sink mechanism the Related Work only mentions in passing, rather than merely declining to find it. **Caveat:** this is N=3, and the halt direction is itself repetition-density-correlated (the modal per-document verdict is **REPETITION-DRIVEN**; the full-trace correlation between repetition density and the halt projection is +0.84 / +0.90 / +0.74), so the anti-correlation with norm is consistent with — and does not escape — the repetition-density confound disclosed elsewhere. The norm result discriminates against *corruption*; it does not by itself identify what the halt projection is tracking (claim **CL-16c**, source `results/norm_vs_projection/_summary.json`).

### Pre-registration and block-shuffle null

**Pre-registration** means the numeric PASS threshold for a test is fixed *before* the validating run (e.g., the L0 sufficiency test pre-registers "PASS = at least a 40 percentage-point cap-hit increase"). This prevents post-hoc goalpost-moving. The **block-shuffle null** is a significance test for autocorrelated sequence data: you shuffle in blocks of size B (here B = 5, 10, 20) to build a null distribution. The conservative B=20 block is the load-bearing one because it respects the longest autocorrelation. All three block sizes are persisted on disk (`results/p2_pilot/round2_block_shuffle_null_B{5,10,20}.json`); each puts layers 20 and 24 at p=0.0033. That p=0.0033 is **not a measured separation** — it is the **resolution floor** of the test, `1/(N_perm+1) = 1/301` at N_perm=300. It means the observed AUC beat *all* 300 block-shuffled permutations (an exceedance count of 0), so the honest reading is **p <= 0.0033**; resolving a tighter value below the 0.005 Bonferroni threshold would need a re-run at larger N_perm. Only layers 20 and 24 clear the Bonferroni threshold of 0.005 for the cross-document probe at B=20 (claim **CL-05e**).

---

## 5. The decision log: how the project pivoted

Research is a path, not a straight line. Here is the actual sequence of moves, including the dead ends, because the dead ends are part of the contribution.

### Trigger: confirm the phenomenon is real and stable

Before mechanism, you need a confirmed positive set. The first move was to re-screen documents and confirm runaway generation reproduces. **Honesty point:** the phenomenon was originally observed on Apple-silicon (MPS) hardware, and moving to CUDA (an H200 cluster) changed the numbers because of cross-backend non-determinism. The strict CUDA hit rate is **6 of 123 documents (4.88%)**. Of the MPS-positive documents that disagreed under CUDA, 8 flipped to control and *zero* new CUDA-only positives appeared. So non-determinism *shrank* the stable set rather than enlarging it; the smaller device-consistent CUDA set is treated as conservative ground truth (claim **CL-02**).

The stable confirmed corpus, per the on-disk index, is **56 positives spanning 12 populated surface classes**. An older write-up cited a 14-class / 45-strict / 49-outline breakdown; those are **stale** (the "14" was a hardcoded title string, and 45/49 appear nowhere on disk). Use 56 / 12. This corpus is also **selection-biased**: it is built from documents already known to trigger the bug, so it is not a random sample and base-rate statements should not be drawn from it.

### Localize: probe, then triangulate

With a confirmed set, *localize* the signal. Linear probes showed halt-failure is linearly decodable at every layer (leave-one-document-out AUC at or above 0.84 at every layer; cross-doc AUC of 0.894 at layer 24, B=20 block-shuffle null p<=0.0033 — the 1/301 resolution floor at N_perm=300, i.e. the observed AUC beat all 300 permutations, not a finer measured separation — only layers 20 and 24 clearing the 0.005 Bonferroni threshold). Logit-lens triangulation located the late "should-halt" signal at layer 24. The block-contribution decomposition showed the filled-cell halt feature is *built* at layer 20 and *read* at layer 24. So far, correlational localization: where the signal is strongest, where it is built and read.

### Walk-backs: the causal tests that failed

This is the heart of the decision log. The project tried repeatedly to find a single causal lever and failed each time. Each failure was logged honestly rather than buried:

- **Sufficiency at layer 0 (CL-49).** Injecting the class-matched halt direction at layer 0 produced a 0.0 pp change (pre-reg needed at least 40 pp). FAIL. Layer 0 is a readout, not a generator.
- **Six converging necessity/perturbation nulls, all now on disk (CL-12, CL-19/20, CL-12b, CL-12c, CL-12d, CL-12e).** Six different ways of removing or zeroing the halt mechanism were each tried and each returned its pre-registered FAIL/null verdict: **B3** reverse-direction project-out (READOUT; 3/3 controls unchanged, positive pinned at the cap, hook confirmed to fire), **P16** component-resolved patch grid (FAIL; 0/45 real-patch cells reduce length, N=5), **P10b** induction-head zeroing (FAIL; 0/4 induction heads pass the 30% threshold, N=2), **P19** system-prompt counterfactual (FAIL; 0/6 stop-instruction prompts pass, median reduction 0%, N=5), **P12** SAE-feature ablation (FAIL; 0/5 CV folds pass, top-feature AUC 0.54-0.63 vs a 0.85 bar, N=22 positives / 56 controls), and **P22** multi-layer coordinated zero-patch (FAIL; the real coordinated patch reduces length 0%, N=2). N ranges 2-22 documents across the six. Crucially, **the convergence does not resolve the mechanism**: a uniform null is equally consistent with a genuinely distributed mechanism and with an under-powered protocol, and this ambiguity is sharpest at the three N=2 nulls (B3's positive, P10b, P22), where the sample is too small to tell the two apart.
- **Norm-scaled positive control (CL-36).** Scaling a single-direction patch at layer 24 by 0.1x, 10x, and 100x of the residual norm moved the *content* visibly (it even prepended foreign-language tokens and inserted stray Markdown) but moved the *token count* not at all on the documents that were actually at the cap. Be precise about the count: the run had **four documents but only three were cap-hit positives** (manifest label `positive`, vanilla = 12,000 tokens), and 0 of those 3 escaped the 12,000 cap at any scale. The fourth document (`table_N005_C04_s05000`, manifest label `control`) was **not** a runaway at all — it halts cleanly on its own at 477 tokens with an EOS stop — so it could never "escape" a cap it never hit, and we do **not** count it as a positive. It is worth flagging because under the *same* perturbation its output **collapsed** (3 tokens at 10x, 1 token at 0.1x, and 12,000 tokens of degenerate single-character repetition at 100x); that collapse is a norm-shock signature that, if anything, *weakens* the "the protocol clearly has sensitivity" reading rather than strengthening it. So the honest framing is: 0 of 3 cap-hit positives escape. This rebuts only the *strongest* form of the easy objection — "your protocol was inert, it did nothing" — because the protocol clearly *did* do something to the output. It does **not** rebut the weaker, more relevant form: that the protocol cannot perturb the *specific* halt direction at the *specific* halt-relevant locus. A random patch that reorders surface tokens (or shatters an unrelated clean halt) is not evidence that a targeted halt-mechanism patch would have been detectable. So this is a sensitivity floor, not a clean refutation of "under-powered."
- **An early optimistic mitigation claim** ("83.1% reduction with zero false positives") did *not* reproduce on the workshop-sized re-bake-off and was walked back. The honest replacement is the precision-recall trade-off (Section 6).

### Chosen framing: the attractor hypothesis

The pattern across all of this is the framing the project adopts — as a *hypothesis*, not a proven mechanism. The signal is strongly *decodable* everywhere and clearly *class-structured*. A genuine "should-halt" signal *develops* late. Yet every attempt to flip behavior by editing one direction or one layer *fails*, while the fine-tune-vs-base separation says fine-tuning introduced the geometry. The framing that *fits* all of it is a **distributed, fine-tune-introduced residual-stream attractor**, not a single-layer bug — but "fits" is the operative word. The converging nulls are equally consistent with an under-powered protocol, and the data here do not adjudicate between "distributed" and "under-powered" (the mechanism remains unidentified). A spine-selection analysis scored attractor / representational-geometry at 0.82 versus attention-collapse at 0.46 and a fused spine at 0.40, precisely because the attractor skeleton is the one whose load-bearing *correlational* claims are overwhelmingly confirmed on disk, while attention-collapse rests on only two confirmed claims (one of them the N=1 attention result). Note that this score reflects which framing the *confirmed* evidence best supports, not a resolution of the distributed-vs-under-powered question.

---

## 6. The verification journey (read this carefully)

This project's credibility rests on a discipline most papers skip: **every load-bearing number cited in the narrative was independently checked against the actual file on disk, not trusted from prose.** The check was run as an adversarial verification pass over 25 claims, with the four headline numbers re-checked by hand. The result is a verification ledger that sorts every claim into one of four verdicts:

- **confirmed** — a matching number was found in a `results/` or `evidence/` data file and quoted exactly;
- **stale** — disk shows a different, superseded value, so the old number must not be cited;
- **unverifiable** — prose-only, no on-disk data file (pending a cluster re-run or never persisted);
- **contradicted** — no supporting file found at all.

The draft is only allowed to cite **confirmed** numbers as fact. This is why several headline-sounding figures were demoted to "future work" or restated.

### Two results were recovered, not abandoned

The two most important verification events were recoveries of genuinely missing results.

**B3 (necessity) was recovered by syncing a real unsynced result.** The strongest necessity test is a *reverse-direction project-out*: it removes the halt direction by projecting it out of the residual stream at every position, and asks whether generation gets longer. The narrative had described this result, but it was not on local disk, so the first ledger pass marked it **unverifiable**. Investigation showed the genuine result had run on the cluster and simply was never synced down. It was pulled to local disk and verified (claim **CL-12**): the verdict is READOUT, the three clean-halting controls are unchanged (within 10%: +9.1%, 0%, 0%), the positive document stays pinned at the cap (12000 to 12000), and crucially the project-out hook is confirmed to *fire* (2 of 4 documents produce materially different output, one flipping its table from Markdown to HTML). That last point matters: it proves the intervention was active, so the null is a genuine null, not a silently inert hook. After syncing, CL-12 moved from unverifiable to **confirmed**, pinned to a specific commit, device=cuda.

**P16 (component-resolved patching) was regenerated after its grid was lost.** The component-resolved FCCT patch grid decomposes the patch into MLP, attention, and block contributions at layers 16, 20, and 24, across documents and intervention types. Its original results file had been lost, so it was also marked **unverifiable**. Rather than cite the remembered numbers, the grid was *regenerated* from scratch on the cluster (45 real-patch cells plus 134 control cells). The regenerated result reproduces the original real-null: **0 of 45 real-patch cells** produce any length reduction (verdict FAIL), while 9 of 134 control cells do (claims **CL-19 / CL-20**). Be precise about those 9 nonzero controls, because the honest reading is sharper than "off-layer norm-shock": per `provenance.json` they break down as random_matched_norm 2, off_layer 4, shuffled_values 3, so **only 4 of the 9 are off-layer — the other 5 fire at the same layer and component as the real patch**. The starkest case is at L16/block_output, where the `random_matched_norm` control (norm 47.354, matched to the real patch's 47.358) collapses one document (`arxiv_table_000401`) by **99.95%** while the **real** halt-direction patch at exactly that site moves length **0%**. A matched-norm *random* vector destroying generation where the real patch does nothing is not "circuit-finding" and not merely "off-layer norm-shock"; it makes **"the controls themselves are suspect"** a live alternative reading of the whole grid. The regeneration even survived an infrastructure detour (node-specific launch failures fixed by excluding the bad nodes, and a wall-time cap that required aggregating the summary on the login node). After regeneration, CL-19/20 moved to **confirmed**, and the unverifiable section of the ledger was emptied.

The lesson: a missing file does **not** become a cited number. It becomes either a sync, a re-run, or a downgrade to future work.

### Stale numbers that were caught and excluded

Equally important is what was *removed*. The ledger caught several plausible-sounding figures that did not survive a disk check, and the paper either corrected or dropped each:

- **"six converging nulls" -> previously walked back to two, now upgraded back to six (all on disk).** The earlier "8" was prose-only. The count was then conservatively walked back to **two** when only B3 and P16 reproduced from local `results/`. As of UP-03 the four formerly-on-scratch summaries have been synced and verified, so all **six** converging perturbation nulls now reproduce from on-disk `_summary.json` files, each returning its pre-registered FAIL/null verdict (N ranging 2-22 docs): **B3** (`p2_pilot/b3_reverse_direction_summary.json`, READOUT), **P16** (`p16_component_resolved_short/_summary.json`, FAIL, N=5), **P10b** (`p10b_head_zero_patching/_summary.json`, FAIL, N=2), **P19** (`p19_system_prompt_counterfactual/_summary.json`, FAIL, N=5), **P12** (`p12_sae_ablation/_summary.json`, FAIL, N=22), and **P22** (`p22_multilayer_coordinated/_summary.json`, FAIL, N=2). The paper now states the count as "six converging perturbation nulls, all returning the pre-registered FAIL/null verdict (N ranging 2-22 docs)" and counts P16 once (it subsumes CL-19/CL-20). The honest caveat is kept: the nulls still do **not** resolve distributed-vs-under-powered, especially at the N=2 nulls (B3's positive, P10b, P22).
- **"83.1% reduction, zero false positives."** Did not reproduce on the workshop-sized re-bake-off; walked back entirely and replaced by the operating curve.
- **The base-model effect-size triple, now superseded by a clean on-disk run (CL-27 -> CL-25-base).** The old prose triple (0.96 / 0.89 / 0.95) was unsynced, and an earlier on-disk file showed a different, superseded triple (0.66 / 0.66 / 0.80). Both are retired. A clean, persisted base-model run now exists (`p18_base_archetype/_summary.json`) on the same Qwen2.5-VL-3B backbone, giving a base triple of **1.12 / 0.78 / 0.92** (N=10 base cap-hits / 15 controls), all near 1x and below 2.0x while the fine-tune scores 2.7 to 3.8x. The fine-tune-origin claim is therefore now substantiated by a hard on-disk number rather than left directional.
- **Off-target control reductions of "96.0 to 99.99%" under norm-shock.** The 99.99% figure has no local source; the 96.0% figure comes from a *different*, separately walked-back confounded experiment. Both are excluded from the load-bearing narrative.
- **Cross-family tallies "11/16" (CL-47) and "8/20 over 4/5 models" (CL-13).** As of UP-13 the cross-family claim is stated with the strict `stop_reason == max_new_tokens` count: **10/16 doc-model cells across 4 families** (per-family 3/4 SmolDocling-256M-preview, 1/4 InternVL3-8B, 3/4 Qwen3-VL-8B-Instruct, 3/4 Qwen3-VL-30B-A3B-Instruct), with the directory-level aggregate verdict `INSUFFICIENT_DATA`. The "11/16" figure is a permissive near-cap relabel (InternVL3-8B/funsd_105 stopped 2 tokens short of the cap). The paper now carries the modest "reproduces on 10 of 16 cells across 4 families, aggregate verdict abstains" framing and makes **no strong generality claim**. CL-13 ("8/20 over 4/5 models") is still the separate, larger sweep where only 4 of 5 models loaded (the fifth load-failed); under strict counting that gives 6 cap-hits over 4 loaded models, and it remains pending reconciliation.

The net effect: the paper's headline numbers are exactly the ones with an on-disk home, and the temptation to round up is visible and resisted in the ledger.

---

## 7. How to read the methodology

A reading guide so the paper's evidence chain makes sense.

**Read the claims as a ladder: correlational, then causal.** Probe AUCs and logit-lens gaps tell you *where* a signal lives and how strong it is. They do not prove the signal *drives* behavior. Patching results do that. When you see a high AUC, do not over-read it; look for the matching causal test.

**Nulls are evidence here, but they are *ambiguous* evidence.** In most papers a null is a disappointment. In an attractor framing, a *converging series* of nulls is what you would expect under distribution — but it is also exactly what you would expect under an under-powered protocol. So the nulls *motivate* the attractor hypothesis without *proving* distribution. The norm-scaled control (0 of 3 cap-hit positives escape; its 4th doc was a clean-halt control that itself collapsed under the patch) shows the protocol was not inert, but it cannot show the protocol could have reached the halt mechanism. Read the nulls as consistent-with-distributed, not as a demonstration of it; the mechanism remains unidentified.

**Watch the confirmed-vs-preliminary line.** After the verification journey, all six converging causal nulls (B3 reverse-direction, P16 component-resolved grid, P10b induction-head zeroing, P19 prompt counterfactual, P12 SAE ablation, P22 multi-layer coordinated patch) are now confirmed on disk, each at its pre-registered FAIL/null verdict, but their residual caveat is *sensitivity, not verification*: in the P16 grid only 4 of the 9 nonzero control patches are off-layer — the other 5 fire at the same layer/component as the real patch, and a norm-matched random vector ($\approx$47.4) collapsed one document 99.95% at L16/block_output where the real patch did 0%, so "the controls are suspect" is a live reading and the nulls establish "single- and multi-site perturbation does not move halting at this protocol's power," not a fully exhaustive causal search. Three of the six rest on only N=2 documents (B3's positive, P10b, P22), so a uniform null at that sample size cannot by itself separate "distributed" from "under-powered." Hold the *strength* of the conclusion accordingly.

**Distinguish per-class from global.** Whenever you see a halt-direction number, ask "is this the per-class direction or the single global one?" Per-class directions are stronger and peak at different layers; the global one is weaker. The class structure is part of the thesis.

**Distinguish relative from absolute EOS suppression.** EOS is not suppressed in an absolute sense (its raw logit +10.76 exceeds a control's +10.07; a single N=1-vs-N=1 trace at a 3000-token cap). It is out-competed by about 20 logits, crushing P(EOS) to a mean of roughly 1e-7 (ranging from about 1e-5 down to 1e-16). It is a relative-margin failure.

**Read cross-family reproduction as modest and not a generality claim; fine-tune-origin is now substantiated.** Under the strict `stop_reason == max_new_tokens` criterion the bug reproduces on 10 of 16 doc-model cells across 4 families (per-family 3/4, 1/4, 3/4, 3/4), but sixteen cells over four families is a thin matrix and the directory-level aggregate verdict abstains (`INSUFFICIENT_DATA`), so read this as "reproduces on 10 of 16 cells" with no strong generality claim, not as proof the bug is universal. The fine-tune signature, by contrast, is no longer merely directional: a matched base-model run on the same Qwen2.5-VL-3B backbone (CL-25-base) shows no separable halt geometry (ratio 1.12 / 0.78 / 0.92, all below 2.0x) even on the base model's own cap-hit docs, so fine-tuning is shown to introduce and sharply amplify the geometry.

**Read the fix as an honest trade-off, not a silver bullet.** The runtime monitor has an operating curve. At a +6.91 logit boost it gives a 37.68% length reduction with 0 of 10 control false positives (clean specificity). At a +30 boost it gives a 99.58% reduction but 10 of 10 control false positives (catastrophic). There is no free lunch; the contribution is characterizing the curve and motivating a vision-aware guardrail (because the grey-noise inpaint result shows image content is sufficient to sustain the loop).

**Stay alert to the two preliminary spots.** The attention-mass collapse (CL-09e) is **N=1 vs N=1** and is the single weakest leg. The corpus is **selection-biased** (built from known-positive documents), so it supports mechanism claims but not base-rate claims.

---

## 8. Compressed claim-to-verdict map

A quick reference tying the headline claims to their verification status. "Confirmed" means a matching number sits in an on-disk data file.

| Claim | What it says | Verdict |
|---|---|---|
| CL-42 | Per-class halt AUC 0.876 to 0.983; per-class peaks at different layers; global direction weaker | confirmed |
| CL-49 | L0 sufficiency injection = 0.0 pp change (FAIL vs 40 pp pre-reg) | confirmed |
| CL-36 | Norm-scaled positive control: 0 of 3 cap-hit positives escape cap at any scale; content visibly changes. (Run's 4th doc is a clean-halt control, not a positive; it collapsed under the patch — a norm-shock signature that weakens the rebuttal.) | confirmed |
| CL-16 | Logit-lens EOS gap most negative at L24 (-0.419), flips positive by L32 (+0.481) | confirmed |
| CL-05e | L24 cross-doc AUC 0.894; B=20 null p<=0.0033 (=1/301 floor at N_perm=300, exceedance count 0, not a measured separation); only L20+L24 clear Bonferroni 0.005 | confirmed |
| CL-22 | EOS genuinely missing at loop onset: document-level (effective N=22) per-doc median rank ~12,234, 0/22 with median rank in top-5, only 1/22 ever in top-5 (pooled record-level median ~10,502) | confirmed |
| CL-50 | Filled-cell halt direction looks built at L20, read at L24 (all_docs N=18) | preliminary candidate (3/18 labeled; alignment ratio sign-flips under doc_argmax_only N=3; cumulative-over-4-layer-gap; MHSA/FFN not separated) |
| CL-11 | Grey-noise inpaint collapses all 5/5 named runaways to 13 to 18 token EOS halts (image content sufficient to sustain loop; grey noise confounds vision-removal with content-removal) | confirmed |
| CL-25 | Fine-tune pos/ctl Cohen's-d ratio 3.79 / 2.74 / 2.78 at L16/L20/L24 (all > 2.0x) | confirmed |
| CL-48 / CL-39 | Monitor: 99.58% reduction at +30 (10/10 FP) vs 37.68% at +6.91 (0/10 FP) | confirmed |
| CL-02 | CUDA re-screen 6/123 = 4.88%; 8 MPS-pos flip to control; 0 new positives | confirmed |
| EOS-margin | Chosen-EOS margin ~20.6; cap-hit EOS logit +10.76 > control +10.07 (single N=1-vs-N=1 trace, 3000-tok cap) | confirmed (illustrative) |
| CL-12 | B3 reverse-direction necessity null (READOUT; hook confirmed to fire) | confirmed (recovered by sync) |
| CL-19/20 | Component-resolved patch: 0/45 real-patch reductions (FAIL) | confirmed (regenerated) |
| CL-12b | P10b induction-head zeroing: 0/4 induction heads pass (FAIL, N=2) | confirmed (`p10b_head_zero_patching/_summary.json`) |
| CL-12c | P19 system-prompt counterfactual: 0/6 stop-prompts pass (FAIL, N=5) | confirmed (`p19_system_prompt_counterfactual/_summary.json`) |
| CL-12d | P12 SAE-feature ablation: 0/5 folds pass; AUC 0.54-0.63 vs 0.85 bar (FAIL, N=22) | confirmed (`p12_sae_ablation/_summary.json`) |
| CL-12e | P22 multi-layer coordinated patch: real patch reduces length 0% (FAIL, N=2) | confirmed (`p22_multilayer_coordinated/_summary.json`) |
| CL-09e | Attention-mass collapse in halt band | preliminary (N=1 vs N=1) |
| CL-25-base | Base-model (Qwen2.5-VL-3B) pos/ctl Cohen's-d ratio 1.12 / 0.78 / 0.92 at L16/L20/L24 (all < 2.0x; N=10 base cap-hits / 15 controls) | confirmed (`p18_base_archetype/_summary.json`) |
| CL-27 | Old base-model triple (0.96/0.89/0.95) | superseded by CL-25-base; retired |
| "six nulls" | Causal-null count | confirmed: all 6 verified on disk (B3 READOUT; P16 FAIL N=5; P10b FAIL N=2; P19 FAIL N=5; P12 FAIL N=22; P22 FAIL N=2), each at its pre-reg FAIL/null verdict; caveat kept that the nulls do not resolve distributed-vs-under-powered (esp. at N=2) |
| 56 / 12 corpus | Positive set size and populated classes | confirmed (the older 14 / 45 / 49 figures are stale) |

---

## 9. Comprehension questions

Try to answer before reading the response.

**Q1. In one sentence, what does it mean to say halt-failure "behaves like" a "residual-stream attractor" rather than a single-layer bug — and why is this a hypothesis?**

It means the runaway state behaves *as if* it were a stable, self-reinforcing pattern distributed across many layers of the model's working memory, so editing any one direction or one layer does not pull the model out of it; we call this a hypothesis because the same nulls are also consistent with a perturbation protocol too weak to reach the mechanism, and the data here do not distinguish "distributed" from "under-powered."

**Q2. The project finds high probe AUC for halt directions but the layer-0 injection produces a 0.0 pp change in cap-hit rate. Why are these two findings not a contradiction?**

Because AUC is correlational (the signal is *readable* at that layer) while injection is causal (does adding the signal *cause* the behavior). A signal can be reliably present and decodable without that single direction being sufficient to drive the behavior on its own. The gap is consistent with the attractor hypothesis (the signal is everywhere but no single lever controls it) — though it is also consistent with the injection protocol being too weak to act on the mechanism, which is why the gap motivates rather than proves the attractor reading.

**Q3. Why is the norm-scaled positive control (0 of 3 cap-hit positives escape the cap) important to the argument, and what does it *not* establish?**

First, the count, honestly: the run held four documents but only three were cap-hit positives (vanilla = 12,000 tokens), and 0 of those 3 escaped the cap at any scale. The fourth doc (`table_N005_C04_s05000`) was a clean-halt control (vanilla 477 tokens, EOS stop), not a runaway, so it is excluded from the positive tally; under the same patch it actually collapsed (down to 1–3 tokens, or 12,000 tokens of degenerate repetition at 100x), a norm-shock signature that *weakens* rather than supports the sensitivity story. With that corrected: the three positives pre-empt the *strongest* form of the objection that the causal nulls just mean the protocol was too weak — the perturbation visibly changed the model's *output content* (even inserting foreign-language tokens) while leaving the *token count* pinned at the 12,000 cap, so the protocol was clearly not inert. But it does **not** establish that the protocol could perturb the specific halt direction at the halt-relevant locus — a random patch reordering surface tokens (or shattering an unrelated clean halt) is not evidence of that. So it bounds, rather than refutes, the "under-powered" alternative; the nulls are meaningful but the mechanism remains unidentified.

**Q4. Is the EOS token "suppressed" in an absolute sense at loop onset? Explain the relative-margin reframe.**

No. The runaway document's raw EOS logit (+10.76) is actually higher than a clean control's (+10.07) — a single N=1-vs-N=1 trace at a 3000-token cap, shown as illustration. The failure is *relative*: continuation tokens out-score EOS by about 20 logits, which crushes P(EOS) to a mean of roughly 1e-7 (ranging from about 1e-5 down to 1e-16). EOS is not pushed down in absolute terms; it is out-competed. Reported at the document level (effective N=22, since the 110 onset records are 5 contiguous records per document), no document has its median EOS rank in the top 5 and only 1 of 22 documents ever puts EOS in the top 5, genuinely missing from the top of the distribution.

**Q5. What is the fine-tune signature, and does the base-model comparison support a fine-tune origin?**

It is a separation in a principal-component crossover analysis: positive-vs-control effect-size ratios of 3.79, 2.74, and 2.78 at layers 16, 20, and 24 in the fine-tuned model, all above the pre-registered 2.0x bar (CL-25, confirmed on disk). The base-model comparison has now been run and persisted on the same Qwen2.5-VL-3B backbone (CL-25-base, `p18_base_archetype/_summary.json`): over the base model's own cap-hit documents (N=10 base cap-hits / 15 controls) the ratio triple is 1.12 / 0.78 / 0.92, all near 1x and below the 2.0x bar. The base backbone shows no separable halt geometry even where it itself runs to the cap, while the fine-tune exceeds the bar at every layer, so the comparison now directly supports that fine-tuning introduces and sharply amplifies the geometry. The retired prose triple (0.96 / 0.89 / 0.95) and the earlier superseded on-disk file (0.66 to 0.80) are no longer cited; the residual caveat is one of scale (10-vs-15 archetype split versus the fine-tune's 22-vs-52).

**Q6. Two of the strongest causal results were initially marked unverifiable. How were they each resolved, and why does the resolution method matter?**

The B3 reverse-direction necessity null was *recovered by syncing*: the genuine result had run on the cluster but was never copied to local disk, so it was pulled down and verified (READOUT verdict, controls unchanged, positive pinned at the cap, project-out hook confirmed to fire). The component-resolved patch grid was *regenerated*: its results file was lost, so the grid was re-run from scratch and reproduced the original real-null (0/45 real-patch reductions). The method matters because in both cases the remembered numbers were not simply cited; they were re-grounded in an on-disk file before being used. A missing file becomes a sync, a re-run, or a downgrade to future work, never a trusted recollection.

**Q7. What does the grey-noise image-inpaint experiment establish, and how does it differ in strength from the cross-modal attention finding?**

Replacing the image with grey noise collapses every tested runaway document — all 5 of 5 named positives — to a short clean EOS halt (about 13 to 18 tokens), showing that the image *content* is sufficient to sustain the loop (CL-11, confirmed; reported qualitatively at N=5 with no Wilson interval). We deliberately say "sufficient" rather than "vision is causally load-bearing" because grey noise confounds removing visual evidence with destroying readable content; the discriminating run (a different *readable* document) is future work. The cross-modal attention-mass finding (about 14x lower text-to-image attention at layer 20 in the runaway case) is only *correlational* and only **N=1 versus N=1** (CL-09e), and it is band-localized rather than monotone — the same document pair shows the positive with *higher* image attention than the control at layer 8 (1.10x) and layer 35 (1.37x) — so it is a suggestive signal reserved for larger-N future work, not an established result.

**Q8. The mitigation once claimed "83.1% reduction with zero false positives." Why was that walked back, and what replaced it?**

It did not reproduce on the workshop-sized re-bake-off, so claiming it would have been dishonest. It was replaced by an explicit precision-recall trade-off: at a +6.91 boost the monitor gives 37.68% reduction with 0 of 10 control false positives (clean specificity), while at a +30 boost it gives 99.58% reduction but 10 of 10 control false positives (catastrophic). The honest contribution is the operating curve, not a single magic number.

**Q9. The count of converging causal nulls moved from "eight" to "six" to "two verified" and back to "six verified." Why all the movement, and what does it tell you about how the project handles its own prose?**

The number tracks exactly what reproduces from on-disk data files, nothing more. An early draft said "eight" (prose-only). It was then trimmed to "six." When the ledger checked disk, only two of the six actually reproduced locally — B3 reverse-direction project-out (`p2_pilot/b3_reverse_direction_summary.json`, READOUT) and the P16 component-resolved grid (`p16_component_resolved_short/_summary.json`, FAIL, 0/45 real-patch) — so the count was conservatively walked back to "two verified plus four unsynced," because the other four ({P10b, P19v2, P12 v3, P22}) lived on `/scratch/...` and could not be verified here. As of UP-03 those four `_summary.json` files were synced into `results/` and each was verified against its pre-registered verdict (P10b 0/4 induction heads, N=2; P19 0/6 stop-prompts, N=5; P12 0/5 folds, N=22; P22 real patch reduces 0%, N=2), so the count is now an honest "six converging perturbation nulls, all verified on disk," with P16 counted once (it subsumes CL-19/CL-20). The movement tells you the project treats its own narrative as something to audit against data: the count goes down when files are missing and back up only when they are on disk and verified — never on recollection. The substantive caveat survives the upgrade unchanged: all six are FAIL/null verdicts, but three rest on N=2 (B3's positive, P10b, P22), so they still do not separate a genuinely distributed mechanism from an under-powered protocol.

**Q10. The corpus is 56 positives across 12 classes and is selection-biased. Why is the selection bias acceptable for the mechanism claims but not for base-rate claims?**

The corpus is deliberately built from documents already known to trigger halt-failure, so it is dense in the phenomenon, which is what you want to *study the mechanism* (you need positives to probe). But because it is not a random sample of documents, you cannot use it to estimate how *often* halt-failure occurs in the wild. The conservative device-consistent base rate comes from a separate re-screen (6 of 123 under CUDA), not from the curated positive set.

---

## 10. The honest takeaway

The strength of this project is not a single dramatic intervention that fixed the bug. It is the disciplined accumulation of evidence, including failed interventions, walked-back claims, and two results that had to be recovered (one synced, one regenerated) before they could be cited. When you read the paper, treat the walk-backs and the verification journey as part of the result, not as weaknesses to skim past. The thesis — that halt-failure *behaves like* a distributed, class-structured, fine-tune-introduced residual-stream attractor — is the *framing* that a converging series of honest nulls, each verified against an on-disk file, best fits. It is a labeled hypothesis, not a settled mechanism: those same nulls are equally consistent with a perturbation protocol too weak to reach the halt mechanism, and the data here do not distinguish "distributed" from "under-powered." The single experiment that would settle it — patching the same halt direction at a non-loop position — is the clearest next step.
