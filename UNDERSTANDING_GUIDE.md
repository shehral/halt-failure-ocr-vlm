# Understanding Guide: Why a Fine-Tuned OCR Vision-Language Model Forgets to Stop

*An educational companion for a strong ML reader who is new to mechanistic interpretability on vision-language models (VLMs).*

This guide walks one project end to end: the problem, the vocabulary you need, how the thinking actually evolved (including the parts that did not work), how to read the methods, **how every load-bearing number was verified against on-disk data**, and a self-check at the end. The goal is not to convince you of a result. It is to give you the mental model to read the paper critically and to see which claims are load-bearing versus preliminary.

**One-sentence thesis.** In a fine-tuned OCR vision-language model, the failure to halt generation is not a single broken neuron or layer. It is a stable, class-structured pattern in the residual stream, an *attractor*, that the network falls into and cannot climb out of, even though it separately computes a "you should stop now" signal that simply arrives too late to win.

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
4. **Converging causal nulls.** Try repeatedly to flip the behavior by editing one direction or one layer. Each attempt fails. A norm-scaled positive control proves the protocol *had* power, so the nulls are informative, not just weak.
5. **The attractor thesis.** The pattern that fits everything: a *distributed, fine-tune-introduced residual-stream attractor*, not a single-layer bug. The nulls are the positive evidence for "distributed."
6. **Cross-family generality.** Show the bug crosses model families and checkpoints (qualitatively, because strict counting and permissive counting disagree).
7. **Production fix trade-off.** Characterize a runtime EOS-boosting monitor as an honest precision-recall curve, and show via image inpainting that vision is causally load-bearing, motivating a vision-aware guardrail.

The thesis is unusual in that **its strongest evidence is negative**: a series of causal perturbations that *fail*. In an attractor framing, a converging set of nulls is exactly what you expect, so they become the positive argument for "distributed."

---

## 3. The central metaphor: a residual-stream attractor

An **attractor** is a stable state that a dynamical system gets pulled into and stays in. The claim is that once the model's residual stream enters the halt-failure region, ongoing generation reinforces that state. Each new repeated token nudges the residual stream right back into the loop. A "should-stop" computation does develop downstream, but it cannot overpower the upstream continuation state that is already locked in.

Why "attractor" rather than "broken layer"? Because every attempt to push the model out by editing one place fails. If the failure lived in one site, surgically editing that site should fix it. It does not. So the failure is **distributed** across the network. The project is careful about which of these nulls are fully verified versus recovered late versus preliminary (Sections 5 and 6).

Why "class-structured"? Because the readable halt signal is not one monolithic direction. It is a *family* of class-conditioned directions: the direction for a filled-cell loop is not the direction for a bare-word loop, and they peak at different depths.

---

## 4. Key concepts glossary (method vocabulary)

These are the load-bearing terms. Each is defined plainly, then connected to the project.

### Residual stream

In a transformer, every layer reads from and writes to a shared running vector, the **residual stream**. Think of it as the model's working memory: layer 1 writes its contribution, layer 2 adds on top, and so on. By the final layer, the residual stream is decoded into next-token probabilities. Almost everything the model "knows" at a generation step is encoded as a direction (a vector) in this stream.

### Linear probe and halt direction

A **linear probe** is a logistic-regression classifier trained on residual-stream activations. If it separates two conditions with high accuracy, the distinguishing feature is *linearly decodable* at that layer. The probe's weight vector is a **direction** in the residual stream.

A **halt direction** is the probe vector separating *positive* documents (the model ran away) from *control* documents (the model halted cleanly). The project finds these are **per-class**: a separate direction per surface class, each linearly decodable with high accuracy, each peaking at a *different* layer. A single "global" direction that ignores class is measurably weaker. The takeaway: halt-failure has *class structure*; it is not one signal (claim **CL-42**).

### Activation patching and path patching

**Activation patching** is the workhorse causal test. You take an activation (say the residual stream at layer 20) from one run and splice it into another, then check whether behavior changes. If patching layer L changes the output, layer L is *causally* involved, not just correlated. This moves you from "the probe can read a signal here" to "this signal actually drives the behavior."

**Path patching** is a refinement: instead of patching a whole layer, you patch a specific pathway (the output of one component feeding into one later component) to find *which route* carries the signal.

In this project the patching results are mostly *nulls*: editing the halt direction at layer 0 produces a **0.0 percentage-point** change in cap-hit rate (the pre-registered bar for "this worked" was at least a 40-point change, so this is a clear FAIL for single-direction sufficiency; claim **CL-49**). That null is informative: it is what you expect if the failure is a distributed attractor.

### Logit lens

The **logit lens** decodes an *intermediate* layer's residual stream as if it were the final layer, by applying the model's final normalization and output projection early. It answers "what would the model predict if it stopped thinking at layer L?" The project tracks the EOS token's logit across layers. The finding: the gap in EOS logit between positive and control runs is most negative around layer 24 (the model most suppresses EOS there), then flips positive by layer 32. A "should-halt" signal does develop late. It just arrives too late to win (claim **CL-16**).

### Block-contribution decomposition (build vs read)

This decomposes a layer's residual update into block contributions and measures the cosine alignment between a layer's block output and the per-class halt direction. A positive alignment means the layer is *building* the halt representation; a negative or near-zero alignment at a later layer means that layer *reads* (consumes) rather than constructs it. For filled-cell-repeat, the halt direction is built at layer 20 and read at layer 24 (claim **CL-50**).

### Cross-modal attention mass

A VLM attends across two modalities: text tokens and image tokens. **Cross-modal attention mass** measures how much of the model's attention, at a given layer, points from generated text back to the image. If the model stops looking at the image, it has nothing new to transcribe and may fall into a content-free loop. The project has a *correlational* signal here, but it is **N=1 versus N=1** (claim **CL-09e**) and explicitly reserved for larger-N work.

### Image inpainting (grey-noise replacement)

Replace the input image with grey noise and re-decode. If runaway documents collapse to short clean EOS halts, vision is *causally* load-bearing for the halt decision. They do: every tested cap-hit positive collapses to a 13- to 18-token EOS-terminated halt (claim **CL-11**). This is the strong vision result; the attention-mass finding above is the weak one.

### Fine-tune signature

A **fine-tune signature** is a measurable property present in the fine-tuned model but absent or much weaker in the base, letting you attribute a behavior to fine-tuning. Here it is a separation in a principal-component crossover analysis: the ratio of positive-class to control-class effect size (absolute Cohen's d) in the fine-tune is 3.79 at layer 16, 2.74 at layer 20, and 2.78 at layer 24, all above the pre-registered 2.0x threshold (claim **CL-25**). The base-model comparison is *directionally* consistent (its ratios fall below 2.0x) but the cleanest base triple is not cleanly persisted, so it is read as directional, not a hard number.

### EOS suppression: relative, not absolute

A subtle but central reframe. EOS is **not** suppressed in an absolute sense. The runaway document's raw EOS logit (+10.76) is actually *higher* than a clean control's (+10.07). What kills EOS is that continuation tokens out-score it by roughly 20 logits, crushing P(EOS) to between 1e-8 and 1e-10. So it is a *relative-margin* failure, not an "EOS logit went to negative infinity" failure. And calibration is genuinely refuted, not a near-miss: at loop onset EOS sits at median rank around 10,500, truly absent from the top of the distribution (claims **EOS-margin** and **CL-22**).

### Pre-registration and block-shuffle null

**Pre-registration** means the numeric PASS threshold for a test is fixed *before* the validating run (e.g., the L0 sufficiency test pre-registers "PASS = at least a 40 percentage-point cap-hit increase"). This prevents post-hoc goalpost-moving. The **block-shuffle null** is a significance test for autocorrelated sequence data: you shuffle in blocks of size B (here B = 5, 10, 20) to build a null distribution. The conservative B=20 block is the load-bearing one because it respects the longest autocorrelation. Only layers 20 and 24 survive Bonferroni correction for the cross-document probe (claim **CL-05e**).

---

## 5. The decision log: how the project pivoted

Research is a path, not a straight line. Here is the actual sequence of moves, including the dead ends, because the dead ends are part of the contribution.

### Trigger: confirm the phenomenon is real and stable

Before mechanism, you need a confirmed positive set. The first move was to re-screen documents and confirm runaway generation reproduces. **Honesty point:** the phenomenon was originally observed on Apple-silicon (MPS) hardware, and moving to CUDA (an H200 cluster) changed the numbers because of cross-backend non-determinism. The strict CUDA hit rate is **6 of 123 documents (4.88%)**. Of the MPS-positive documents that disagreed under CUDA, 8 flipped to control and *zero* new CUDA-only positives appeared. So non-determinism *shrank* the stable set rather than enlarging it; the smaller device-consistent CUDA set is treated as conservative ground truth (claim **CL-02**).

The stable confirmed corpus, per the on-disk index, is **56 positives spanning 12 populated surface classes**. An older write-up cited a 14-class / 45-strict / 49-outline breakdown; those are **stale** (the "14" was a hardcoded title string, and 45/49 appear nowhere on disk). Use 56 / 12. This corpus is also **selection-biased**: it is built from documents already known to trigger the bug, so it is not a random sample and base-rate statements should not be drawn from it.

### Localize: probe, then triangulate

With a confirmed set, *localize* the signal. Linear probes showed halt-failure is linearly decodable at every layer (leave-one-document-out AUC at or above 0.84 at every layer; cross-doc AUC of 0.894 at layer 24, B=20 block-shuffle null p=0.0033, only layers 20 and 24 surviving Bonferroni). Logit-lens triangulation located the late "should-halt" signal at layer 24. The block-contribution decomposition showed the filled-cell halt feature is *built* at layer 20 and *read* at layer 24. So far, correlational localization: where the signal is strongest, where it is built and read.

### Walk-backs: the causal tests that failed

This is the heart of the decision log. The project tried repeatedly to find a single causal lever and failed each time. Each failure was logged honestly rather than buried:

- **Sufficiency at layer 0 (CL-49).** Injecting the class-matched halt direction at layer 0 produced a 0.0 pp change (pre-reg needed at least 40 pp). FAIL. Layer 0 is a readout, not a generator.
- **Norm-scaled positive control (CL-36).** Scaling a single-direction patch at layer 24 by 0.1x, 10x, and 100x of the residual norm moved the *content* visibly (it even prepended foreign-language tokens and inserted stray Markdown) but moved the *token count* not at all: 0 of 4 documents escaped the 12,000 cap. This rebuts the easy objection "your protocol was just too weak to do anything," because the protocol clearly *did* do something to the output, just not to halting.
- **An early optimistic mitigation claim** ("83.1% reduction with zero false positives") did *not* reproduce on the workshop-sized re-bake-off and was walked back. The honest replacement is the precision-recall trade-off (Section 6).

### Chosen spine: the attractor

The pattern across all of this is the spine. The signal is strongly *decodable* everywhere and clearly *class-structured*. A genuine "should-halt" signal *develops* late. Yet every attempt to flip behavior by editing one direction or one layer *fails*, while the fine-tune-vs-base separation says fine-tuning introduced the whole thing. The framing that fits all of it is a **distributed, fine-tune-introduced residual-stream attractor**, not a single-layer bug. A spine-selection analysis scored attractor / representational-geometry at 0.82 versus attention-collapse at 0.46 and a fused spine at 0.40, precisely because the attractor skeleton is the one whose load-bearing claims are overwhelmingly confirmed on disk, while attention-collapse rests on only two confirmed claims (one of them the N=1 attention result).

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

**P16 (component-resolved patching) was regenerated after its grid was lost.** The component-resolved FCCT patch grid decomposes the patch into MLP, attention, and block contributions at layers 16, 20, and 24, across documents and intervention types. Its original results file had been lost, so it was also marked **unverifiable**. Rather than cite the remembered numbers, the grid was *regenerated* from scratch on the cluster (45 real-patch cells plus 134 control cells). The regenerated result reproduces the original real-null: **0 of 45 real-patch cells** produce any length reduction (verdict FAIL), while 9 of 134 control cells do, with the nonzero controls dominated by off-layer norm-shock rather than circuit-finding (claims **CL-19 / CL-20**). The regeneration even survived an infrastructure detour (node-specific launch failures fixed by excluding the bad nodes, and a wall-time cap that required aggregating the summary on the login node). After regeneration, CL-19/20 moved to **confirmed**, and the unverifiable section of the ledger was emptied.

The lesson: a missing file does **not** become a cited number. It becomes either a sync, a re-run, or a downgrade to future work.

### Stale numbers that were caught and excluded

Equally important is what was *removed*. The ledger caught several plausible-sounding figures that did not survive a disk check, and the paper either corrected or dropped each:

- **"8 converging nulls" -> 6.** The figure "8" appeared only in prose; the canonical claims record says **six** ({B3, P16, P10b, P19v2, P12 v3, P22}). The paper cites six and flags eight as unverified.
- **"83.1% reduction, zero false positives."** Did not reproduce on the workshop-sized re-bake-off; walked back entirely and replaced by the operating curve.
- **The base-model effect-size triple (0.96 / 0.89 / 0.95) and "10/22 base also cap-hit" (CL-27).** Prose-only; the cited cluster path was unsynced, and the only on-disk base file shows a different, superseded triple (0.66 / 0.66 / 0.80). The *direction* still holds (base ratios fall below 2.0x and far below the fine-tune's 2.7 to 3.8x), so the paper keeps the directional claim but refuses to cite the specific triple as fact.
- **Off-target control reductions of "96.0 to 99.99%" under norm-shock.** The 99.99% figure has no local source; the 96.0% figure comes from a *different*, separately walked-back confounded experiment. Both are excluded from the load-bearing narrative.
- **Cross-family tallies "11/16" (CL-47) and "8/20 over 4/5 models" (CL-13).** Strict `stop_reason == max_new_tokens` counting gives smaller numbers (10/16, and 6 cap-hits over 4 loaded models; the fifth model failed to load). The 11th cross-family cell was a near-cap relabel that stopped 2 tokens short of the cap. The paper presents cross-family reproduction *qualitatively* and defers a strict count to future work.

The net effect: the paper's headline numbers are exactly the ones with an on-disk home, and the temptation to round up is visible and resisted in the ledger.

---

## 7. How to read the methodology

A reading guide so the paper's evidence chain makes sense.

**Read the claims as a ladder: correlational, then causal.** Probe AUCs and logit-lens gaps tell you *where* a signal lives and how strong it is. They do not prove the signal *drives* behavior. Patching results do that. When you see a high AUC, do not over-read it; look for the matching causal test.

**Nulls are evidence here, not absence of evidence.** In most papers a null is a disappointment. In an attractor framing, a *converging series* of nulls is the positive evidence for distribution. The argument only works if the nulls are real and the positive control shows the protocol had power. Check both: the 0/4 norm-scaled control is what gives the nulls their teeth.

**Watch the confirmed-vs-preliminary line.** After the verification journey, the strongest causal nulls (B3 and the component-resolved grid) are now confirmed on disk, but their residual caveat is *sensitivity, not verification*: the nonzero control patches are mostly off-layer norm-shock, so the nulls establish "single-site perturbation does not move halting at this protocol's power," not a fully exhaustive causal search. Hold the *strength* of the conclusion accordingly.

**Distinguish per-class from global.** Whenever you see a halt-direction number, ask "is this the per-class direction or the single global one?" Per-class directions are stronger and peak at different layers; the global one is weaker. The class structure is part of the thesis.

**Distinguish relative from absolute EOS suppression.** EOS is not suppressed in an absolute sense (its raw logit +10.76 exceeds a control's +10.07). It is out-competed by about 20 logits, crushing P(EOS) to 1e-8 to 1e-10. It is a relative-margin failure.

**Read generality and fine-tune-origin as directional.** Cross-family and cross-checkpoint reproduction is real qualitatively (the bug crosses families), but literal tallies are interpretive. Read "reproduces across families" as the claim, not a specific fraction. Likewise the fine-tune signature is verified but its base-model comparison is directional.

**Read the fix as an honest trade-off, not a silver bullet.** The runtime monitor has an operating curve. At a +6.91 logit boost it gives a 37.68% length reduction with 0 of 10 control false positives (clean specificity). At a +30 boost it gives a 99.58% reduction but 10 of 10 control false positives (catastrophic). There is no free lunch; the contribution is characterizing the curve and motivating a vision-aware guardrail (because the grey-noise inpaint result shows vision is causally load-bearing).

**Stay alert to the two preliminary spots.** The attention-mass collapse (CL-09e) is **N=1 vs N=1** and is the single weakest leg. The corpus is **selection-biased** (built from known-positive documents), so it supports mechanism claims but not base-rate claims.

---

## 8. Compressed claim-to-verdict map

A quick reference tying the headline claims to their verification status. "Confirmed" means a matching number sits in an on-disk data file.

| Claim | What it says | Verdict |
|---|---|---|
| CL-42 | Per-class halt AUC 0.876 to 0.983; per-class peaks at different layers; global direction weaker | confirmed |
| CL-49 | L0 sufficiency injection = 0.0 pp change (FAIL vs 40 pp pre-reg) | confirmed |
| CL-36 | Norm-scaled positive control: 0/4 escape cap at any scale; content visibly changes | confirmed |
| CL-16 | Logit-lens EOS gap most negative at L24 (-0.419), flips positive by L32 (+0.481) | confirmed |
| CL-05e | L24 cross-doc AUC 0.894; B=20 null p=0.0033; only L20+L24 Bonferroni-significant | confirmed |
| CL-22 | EOS median rank ~10,502 at loop onset (genuinely missing) | confirmed |
| CL-50 | Filled-cell halt direction built at L20, read at L24 | confirmed |
| CL-11 | Grey-noise inpaint collapses runaways to 13 to 18 token EOS halts (vision causal) | confirmed |
| CL-25 | Fine-tune pos/ctl Cohen's-d ratio 3.79 / 2.74 / 2.78 at L16/L20/L24 (all > 2.0x) | confirmed |
| CL-48 / CL-39 | Monitor: 99.58% reduction at +30 (10/10 FP) vs 37.68% at +6.91 (0/10 FP) | confirmed |
| CL-02 | CUDA re-screen 6/123 = 4.88%; 8 MPS-pos flip to control; 0 new positives | confirmed |
| EOS-margin | Chosen-EOS margin ~20.6; cap-hit EOS logit +10.76 > control +10.07 | confirmed |
| CL-12 | B3 reverse-direction necessity null (READOUT; hook confirmed to fire) | confirmed (recovered by sync) |
| CL-19/20 | Component-resolved patch: 0/45 real-patch reductions (FAIL) | confirmed (regenerated) |
| CL-09e | Attention-mass collapse in halt band | preliminary (N=1 vs N=1) |
| CL-27 | Base-model effect-size triple | directional only (specific triple not on disk) |
| "8 nulls" | Converging-null count | stale: correct value is 6 |
| 56 / 12 corpus | Positive set size and populated classes | confirmed (the older 14 / 45 / 49 figures are stale) |

---

## 9. Comprehension questions

Try to answer before reading the response.

**Q1. In one sentence, what does it mean to say halt-failure is a "residual-stream attractor" rather than a single-layer bug?**

It means the runaway state is a stable, self-reinforcing pattern distributed across many layers of the model's working memory, so editing any one direction or one layer does not pull the model out of it, rather than a single localized component you could surgically fix.

**Q2. The project finds high probe AUC for halt directions but the layer-0 injection produces a 0.0 pp change in cap-hit rate. Why are these two findings not a contradiction?**

Because AUC is correlational (the signal is *readable* at that layer) while injection is causal (does adding the signal *cause* the behavior). A signal can be reliably present and decodable without that single direction being sufficient to drive the behavior on its own. The gap is exactly what the attractor thesis predicts: the signal is everywhere but no single lever controls it.

**Q3. Why is the norm-scaled positive control (0 of 4 documents escape the cap) important to the argument?**

It pre-empts the objection that the causal nulls just mean the perturbation protocol was too weak. The perturbation visibly changed the model's *output content* (even inserting foreign-language tokens) while leaving the *token count* pinned at the 12,000 cap. So the protocol clearly had power; it just could not move halting. That makes the nulls meaningful rather than uninformative.

**Q4. Is the EOS token "suppressed" in an absolute sense at loop onset? Explain the relative-margin reframe.**

No. The runaway document's raw EOS logit (+10.76) is actually higher than a clean control's (+10.07). The failure is *relative*: continuation tokens out-score EOS by about 20 logits, which crushes P(EOS) to roughly 1e-8 to 1e-10. EOS is not pushed down in absolute terms; it is out-competed, and ends up at median rank around 10,502, genuinely missing from the top of the distribution.

**Q5. What is the fine-tune signature, and what is the honest caveat on the base-model comparison?**

It is a separation in a principal-component crossover analysis: positive-vs-control effect-size ratios of 3.79, 2.74, and 2.78 at layers 16, 20, and 24 in the fine-tuned model, all above the pre-registered 2.0x bar (CL-25, confirmed on disk). The caveat is that the *clean base-model triple* (canonically cited as 0.96 / 0.89 / 0.95) is prose-only and unsynced; the only on-disk base file shows a different superseded triple (0.66 to 0.80). The *direction* holds (base ratios fall below 2.0x), so the paper keeps the directional claim but does not cite the specific base triple as fact.

**Q6. Two of the strongest causal results were initially marked unverifiable. How were they each resolved, and why does the resolution method matter?**

The B3 reverse-direction necessity null was *recovered by syncing*: the genuine result had run on the cluster but was never copied to local disk, so it was pulled down and verified (READOUT verdict, controls unchanged, positive pinned at the cap, project-out hook confirmed to fire). The component-resolved patch grid was *regenerated*: its results file was lost, so the grid was re-run from scratch and reproduced the original real-null (0/45 real-patch reductions). The method matters because in both cases the remembered numbers were not simply cited; they were re-grounded in an on-disk file before being used. A missing file becomes a sync, a re-run, or a downgrade to future work, never a trusted recollection.

**Q7. What does the grey-noise image-inpaint experiment establish, and how does it differ in strength from the cross-modal attention finding?**

Replacing the image with grey noise collapses every tested runaway document to a short clean EOS halt (about 13 to 18 tokens), a *causal* demonstration that vision is load-bearing for the halt decision (CL-11, confirmed). The cross-modal attention-mass finding (about 14x lower text-to-image attention at layer 20 in the runaway case) is only *correlational* and only **N=1 versus N=1** (CL-09e), so it is a suggestive signal reserved for larger-N future work, not an established result.

**Q8. The mitigation once claimed "83.1% reduction with zero false positives." Why was that walked back, and what replaced it?**

It did not reproduce on the workshop-sized re-bake-off, so claiming it would have been dishonest. It was replaced by an explicit precision-recall trade-off: at a +6.91 boost the monitor gives 37.68% reduction with 0 of 10 control false positives (clean specificity), while at a +30 boost it gives 99.58% reduction but 10 of 10 control false positives (catastrophic). The honest contribution is the operating curve, not a single magic number.

**Q9. Why does the project report "six converging nulls" rather than "eight," and what does this tell you about how it handles its own prose?**

Because the figure "eight" appeared only in informal prose, while the canonical claims record lists exactly six nulls ({B3, P16, P10b, P19v2, P12 v3, P22}). The verification pass caught the discrepancy and the paper cites six, flagging eight as unverified. It tells you the project treats its own narrative as something to audit against data, not as a source of truth, which is the same discipline applied to every headline number.

**Q10. The corpus is 56 positives across 12 classes and is selection-biased. Why is the selection bias acceptable for the mechanism claims but not for base-rate claims?**

The corpus is deliberately built from documents already known to trigger halt-failure, so it is dense in the phenomenon, which is what you want to *study the mechanism* (you need positives to probe). But because it is not a random sample of documents, you cannot use it to estimate how *often* halt-failure occurs in the wild. The conservative device-consistent base rate comes from a separate re-screen (6 of 123 under CUDA), not from the curated positive set.

---

## 10. The honest takeaway

The strength of this project is not a single dramatic intervention that fixed the bug. It is the disciplined accumulation of evidence, including failed interventions, walked-back claims, and two results that had to be recovered (one synced, one regenerated) before they could be cited. When you read the paper, treat the walk-backs and the verification journey as part of the result, not as weaknesses to skim past. The thesis, a distributed, class-structured, fine-tune-introduced residual-stream attractor, is exactly the shape that a converging series of honest nulls, each verified against an on-disk file, points to.
