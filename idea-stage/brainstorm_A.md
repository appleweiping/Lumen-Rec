# Brainstorm A — Uncertainty-aware LLM4Rec ideas for SIGIR 2027  [gen: opus-A]

Generator: fresh Claude Opus 5.5 sub-agent ("opus-A"). Date: 2026-10-01.
Inputs read in full: `idea-stage/brainstorm_bundle.md`, `RESEARCH_BRIEF.md`, `docs/sigir/LIT_HOOI_GROUP.md`,
`docs/sigir/LIT_RECENT_UNC_LLM4REC.md`, `Paper/CALIBRA_draft.md`, and the sibling project's
`docs/NEGATIVE_RESULTS.md` and `docs/ADVISOR_MEMO.md`.
Novelty sanity checks used WebSearch/WebFetch only (sources at the end). No existing file was edited.

## Seed-question key (the user's six questions, in the user's order)

| Key | User's question |
|---|---|
| **S1** | The LLM may give the correct output, but is it *sure* of it? |
| **S2** | Is a wrong answer wrong *because* of low confidence, or is it confidently wrong? |
| **S3** | Does high confidence cause an echo-chamber effect? |
| **S4** | How does the yes/no output confidence correlate with ground truth? |
| **S5** | Do popular items get high confidence and niche items low confidence? |
| **S6** | Can we prune training data by uncertainty, because it signals noise? |

## Shared substrate (so each idea below does not repeat it)

- **Labels match the question.** This fixes the CALIBRA flaw. Every calibration claim below uses one of two label types:
  - explicit like/dislike labels: rating ≥ 4 → Yes, rating ≤ 2 → No, drop 3 (TALLRec-style); Steam's "recommended" flag; KuaiRec watch_ratio above a threshold;
  - or a list-normalised probability, i.e. a softmax over a user's candidate logits, scored against "is this the next item".
- **Confidence channel.** Token-level `P(Yes) = softmax(logit_Yes, logit_No)` at the first answer position, read with vLLM prompt logprobs.
  - No generation is needed, and prefix caching shares the user's history across that user's candidates. Expect tens to hundreds of prompts per second on one A100.
  - Lumen's verbalized `p_i` is kept as a second channel where useful.
- **Backbones:**
  - Qwen3-8B (thinking off unless stated);
  - Llama-3.1-8B-Instruct;
  - **OLMo-2-7B-Instruct**, the principled third family: its pretraining corpus is open, so item exposure can be *counted* (Idea 3).
- **Datasets, chosen to span LLM familiarity:**
  - ML-1M: high familiarity.
  - Amazon-2023 Books/Toys/Beauty: medium, brand-heavy.
  - Steam: native binary recommend label.
  - KuaiRec: low familiarity, creator captions in Chinese, fully-observed small matrix.
- **Rigor defaults:**
  - ≥ 3 seeds for anything trained (5 for pruning);
  - a matched-random or placebo control for every intervention;
  - a pre-registered kill gate for every pilot;
  - the test split is touched once.

---

## Idea 1 — MIRROR: ask the question both ways; the acquiescence gap separates "I know this item" from "this user likes it"  [gen: opus-A]

1. **One-sentence summary.**
   - Score every candidate with two mirrored prompts, "Will the user *like* X?" and "Will the user *dislike* X?".
   - The **sum** of the two Yes log-odds measures item-driven acquiescence (yes-saying to recognisable items).
   - The **difference** measures user-conditional preference.
   - Ranking and calibrating on the difference removes the familiarity term with the model's own counterfactual.
2. **Core hypothesis.**
   - Model: `logit_like(u,i) = a(i) + s(u,i)` and `logit_dislike(u,i) = a(i) − s(u,i)`.
   - A negation-coherent model has `a ≈ 0`. We expect `a(i) > 0`, increasing with the item's pretraining exposure or popularity, and expect confident errors to concentrate where `a` is large.
   - Because `a(i)` depends on the item, removing it **changes within-user ranking**. This is a legitimate escape from the N1 rank-preservation negative.
   - It should also cut head over-exposure and improve calibration on tail items.
   - Either outcome matters. If `a ≈ 0`, LLM recommenders are negation-coherent, and familiarity enters confidence only through `s`. That directly contradicts the "confidence = popularity" story and is itself a finding.
3. **Method (one mechanism).**
   - Score: `score_M = (logit_like − logit_dislike)/2`, with `P_M = σ(score_M)`.
   - Fine-tuned setting, "mirror tuning": each training pair appears in both phrasings with flipped targets under the same BCE. The scoring rule stays proper, and the model is trained to be negation-coherent. At test time, use the difference.
   - Identifiability argument: under the additive model, `score_M` is unbiased for `s` whatever `a(i)` is. Any single-prompt score is biased by `a(i)`.
4. **Minimum viable experiment (≈1–1.5 GPU-h).**
   - Setup: zero-shot Qwen3-8B on Amazon Toys + ML-1M; 2k users × 20 rated candidates each.
   - Three prompts: *like*, *dislike*, and a *paraphrased-like* placebo.
   - Report:
     - `corr(a(i), log popularity)`;
     - ECE, AUROC(correct), UAUC and AUC for raw vs MIRROR vs a **paraphrase-ensemble placebo** with identical compute;
     - head share of top-1.
   - **Kill gate:** drop MIRROR as a method (keep it as a finding) if either
     - mean |a| < 0.1 logit, or
     - the confidence interval of UAUC(MIRROR) − UAUC(placebo) contains 0 on both domains.

     The placebo is the key control: it separates acquiescence removal from plain two-prompt ensembling.
5. **Contribution type:** diagnostic (the acquiescence gap) plus a new method (mirror scoring/tuning) with an identifiability argument.
6. **Risk: MEDIUM.**
   - After rec fine-tuning, the dislike prompt is out-of-distribution unless the model is mirror-tuned.
   - Mirror tuning may make the model coherent and collapse `a` toward 0. In that case the gain shows up in calibration rather than ranking.
   - Novelty check: the NLP yes/no-bias literature shows *global* Yes/No answer bias. We found no item-conditional acquiescence analysis for recommendation.

   **Effort:** weeks (pilot 2 days).
7. **Seed questions:** S4, S5, S2, S1 (primary); S3 via exposure.

---

## Idea 2 — Pseudonym knockout: a causal test of whether LLM-rec confidence is name recognition, with name-dropout tuning as the fix  [gen: opus-A]

1. **One-sentence summary.**
   - Replace every real item or brand name in the prompt, in the history and in the candidate, with a **consistent invented pseudonym** that keeps category and attributes.
   - In-prompt collaborative structure survives (brand loyalty, series, repeat purchase), but pretraining knowledge of the name is removed.
   - The confidence shift is therefore the *causal* familiarity effect.
2. **Core hypothesis.**
   - Expected outcome: under pseudonyms, the head > mid > tail over-confidence gradient (CALIBRA RQ2) and the confidently-wrong mass shrink sharply, while within-user discrimination on tail items is unchanged.
     - That would show the popularity skew in confidence is a pretraining-familiarity effect.
     - It would also explain *why* interaction-popularity recalibration (N2) failed: it conditioned on the wrong variable.
   - Equally informative opposite outcome: if confidence is unchanged under pseudonyms, LLM-rec confidence is in-prompt evidence, and N2's null is not about familiarity.
3. **Method (one knob).** "Name-dropout" LoRA tuning.
   - Pseudonymize names in a random 30–50% of training examples, with a map that is consistent within each example.
   - This pushes Yes/No toward history-conditional evidence. Evaluate on real names.
   - Measure head over-confidence, tail UAUC/NDCG, ECE and head over-exposure against standard tuning at matched steps, over 3 seeds.
4. **Minimum viable experiment (≤2 GPU-h).**
   - Setup: zero-shot Qwen3-8B on Amazon Beauty and Toys (brand-dominated), 2k users × 20 rated candidates, real names vs pseudonyms.
   - The pseudonym map is built from the `store`/brand metadata field plus brand tokens in titles, using hash-seeded pronounceable names.
   - Report:
     - Δ mean logit by popularity decile;
     - the ProCal-style confidence-matched Bias Index, before and after;
     - UAUC change, overall and in the tail.
   - **Kill gate:** drop name-dropout if the head-vs-tail difference in Δ logit is < 0.1 and the Bias Index does not move.
5. **Contribution type:** empirical finding (causal familiarity estimate) plus a new method (name-dropout regularizer).
6. **Risk: MEDIUM-HIGH.**
   - In Books and Movies the title *is* the content, so pseudonymizing it destroys information. Restrict to brand tokens or to brand-dominated domains.
   - Name-dropout may cost head accuracy, since KnowSA shows parametric knowledge helps.
   - Novelty check: "Incumbent Advantage" (arXiv 2606.17443) shows real brands beat fictional ones in *zero-shot, non-personalized* choice with identical specs. Nobody uses a consistent pseudonym map as a familiarity knockout for the *confidence/calibration of personalized* recommendations, or as a training regularizer.

   **Effort:** weeks.
7. **Seed questions:** S5 (primary), S2, S3, S1.

---

## Idea 3 — Exposure, not popularity: open-corpus counts explain the CALIBRA sign flip  [gen: opus-A]

1. **One-sentence summary.**
   - Use a backbone with an open pretraining corpus: OLMo-2-7B, with exact title/brand n-gram counts via infini-gram. Add proxy counts for Qwen and Llama.
   - Test whether **pretraining exposure**, rather than interaction popularity, predicts:
     - the confidence level;
     - the confidence-matched calibration gap (ProCal Bias Index).
   - Then test whether exposure explains the domain-dependent sign flip CALIBRA found and could not explain.
2. **Core hypothesis.** Three predictions:
   - (a) The partial Spearman correlation of confidence with exposure given popularity is much larger than that with popularity given exposure.
   - (b) At matched confidence, high-exposure items are *less* accurate (familiarity-inflated confidence) in every domain, so the sign flip disappears once we condition on exposure. The flip itself would come from `corr(popularity, exposure)` differing by domain: high for branded toys and beauty items, low for generic sports and home items.
   - (c) An exposure-informed calibration error (EICE, the PIECE analogue; EICE ≥ ECE by the ProCal argument) exposes over/under-confidence cancellation that ECE hides.
3. **Minimum viable experiment (≈1 GPU-h plus CPU/API).**
   - Score rating-labeled data on 2 domains (Toys, where the head coefficient was negative; Sports, where it was positive) with OLMo-2-7B-Instruct and Qwen3-8B.
   - Get infini-gram counts for about 20k titles, using brand + model-number matching for generic titles.
   - Run partial correlations and the confidence-matched Bias Index by exposure tercile vs popularity tercile.
   - Optional light method: a 2-D (confidence × exposure) bin-mean-shift. It is item-conditional, so it can change ranking. Run it pre-registered head-to-head against the dead PopProCal baseline. If exposure-ProCal wins where PopProCal did not, that isolates the latent variable.
   - **Kill gate:** if exposure adds less than 0.05 partial correlation over popularity in both domains, report it as a negative and drop the method.
4. **Contribution type:** empirical finding plus a metric (EICE); the method is optional.
5. **Risk: MEDIUM.**
   - Counts are noisy for generic titles ("Wireless Mouse").
   - OLMo-2 is a weaker recommender.
   - Infini-gram API rate limits apply.
   - Novelty check: a search snippet indicates prior work has correlated infini-gram item-name frequency with LLM-rec *accuracy* (the paper must be pinned down before writing). The Mozafari SIGIR'26 short covers popularity *judgments*, and "Towards Objective Fine-tuning" (2505.20903) links known data to overconfidence in QA. Our delta is the confidence–accuracy *gap at matched confidence* in personalized recommendation, and the resolution of a measured sign flip.

   **Effort:** 1–3 weeks.
6. **Seed questions:** S5 (primary), S2, S4.

---

## Idea 4 — Calibration is the exchange rate between recommenders: calibrated log-odds pooling of LLM and CF  [gen: opus-A]

1. **One-sentence summary.** Calibration is provably rank-inert for *one* model (N1) but rank-determining for any *combination* of models. So we calibrate LLM P(Yes) and a CF/sequential model to a common probability scale and pool them by log-odds (product of experts with prior correction). This gives per-item adaptive weights with no tuned mixing coefficient.
2. **Core hypothesis.**
   - Under conditional independence given relevance, the Bayes-optimal fused score is `logit P_LLM + logit P_CF − logit π`. Miscalibration of either model is *exactly* a mis-set fusion weight.
   - Expected results:
     - calibrated pooling ≥ the best grid-tuned linear interpolation overall;
     - and strictly better on tail and cold items, where CF log-odds collapse to the prior while the LLM's stay informative;
     - it recovers a visible share of the +21% oracle LLM↔CF router gap.
   - The same theory explains why the cheap uncertainty gate failed (correlation 0.07): routing needs *relative* calibrated confidence, not one model's uncertainty.
3. **Minimum viable experiment (≤1 GPU-h).**
   - Data: one Amazon domain. LLM scores and SASRec scores on identical 101-candidate lists.
   - Calibration: list-normalized temperature calibration of both models on validation users, so the labels match "is this the next item".
   - Compare: LLM alone, SASRec alone, tuned α-interpolation, reciprocal-rank fusion, calibrated product of experts (PoE), and the oracle router (NDCG@10, overall and by popularity tercile).
   - **Kill gate:** in the pilot, calibrated PoE < tuned interpolation − 1 pt NDCG@10 → drop it as a method and keep it as a theory note inside Idea 5.
4. **Contribution type:** theoretical result (calibration ⇔ fusion weight) plus a simple method.
5. **Risk: MEDIUM-HIGH.**
   - Tuned interpolation is a strong, cheap baseline.
   - The conditional-independence assumption is violated, because both models see popularity.
   - Novelty check: RouteRec (2607.09908) is negative on hard selection. "Rethinking Semantic-Collaborative Integration" (SIGIR'26, 2604.22195) quantifies complementarity but fuses representations, with no calibrated-probability fusion (checked against the abstract).

   **Effort:** days to 2 weeks.
6. **Seed questions:** S4, S2 (when the LLM is wrong, who is right?), S1.

---

## Idea 5 — Confidence measures redundancy, not value  [gen: opus-A]

1. **One-sentence summary.** We test the inverted hypothesis that LLM-rec confidence is highest exactly where a CF model is *also* right (head consensus) and lowest where the LLM's *unique* hits live. If true, "correct but unsure" is the most valuable region, and confidence-gated routing is mis-specified by construction.
2. **Core hypothesis.**
   - `P(LLM hit ∧ CF miss | LLM-confidence decile)` is flat, or peaks in the low and mid deciles.
   - `P(LLM hit ∧ CF hit)` dominates the top decile.
   - The LLM's unique hits are tail or semantic matches with low confidence (S1).
   - This explains both the failed gate (correlation 0.07) and RouteRec's negative, and it prescribes routing on *complementarity* (LLM confidence relative to CF confidence on the same items) instead.
3. **Minimum viable experiment (≤1 GPU-h).**
   - One domain; LLM P(Yes) (or verbalized p) plus SASRec on identical candidate lists.
   - Build a 2×2 hit contingency table by LLM-confidence decile, with bootstrap CIs, split by item-popularity tercile.
   - This is pure analysis once the scores exist.
4. **Contribution type:** diagnostic. One figure that reframes selective serving and routing; a natural section inside Idea 4 or the flagship paper.
5. **Risk: LOW.** It is cheap, and both outcomes are informative. **Effort:** days.
6. **Seed questions:** S1, S2, S5.

---

## Idea 6 — A ground-truth echo chamber: closed-loop LLM recommendation on a fully-observed matrix  [gen: opus-A]

1. **One-sentence summary.**
   - Run the feedback loop on KuaiRec's fully-observed small matrix (1,411 users × 3,327 videos, 99.6% dense, with creator captions and tags).
   - Every served recommendation gets its *real recorded response*, so no LLM or user simulator is needed.
   - Test whether confidence-greedy retraining produces an echo chamber that shows up **first as confidence inflation**, before it shows up in accuracy or exposure.
2. **Core hypothesis.**
   - Under confidence-greedy serving plus retraining on served feedback:
     - the model's mean confidence on served items rises faster than its realized hit rate (self-confirmation);
     - the inflation concentrates on already-dominant categories;
     - coverage and Gini degrade.
   - An uncertainty-aware policy slows homogenization at equal cumulative *true* reward. Candidates: Thompson-style sampling from a calibrated Beta whose variance comes from M candidate-order permutations, or ε-exploration at a matched budget.
   - Equally clean negative alternative: on low-familiarity short-video captions, LLM priors are weak and loops do not homogenize. That would be consistent with RecLoop's "generative recommenders show less cocoon", and here it would come with ground truth.
3. **Minimum viable experiment (≤2 GPU-h).**
   - A zero-shot, 2-round loop for 300 small-matrix users. History is updated in-context, no retraining.
   - Confirm that Qwen3 reads the Chinese captions, and measure confidence-vs-hit drift and category concentration between rounds.
   - Full protocol:
     - TALLRec-style LoRA on big-matrix interactions;
     - each round, serve top-K of 100 candidates per small-matrix user, read the true watch_ratio, and add the served feedback to training;
     - 4–5 rounds × 3 policies × 3 seeds.
4. **Contribution type:** empirical finding (first non-simulated-user loop study of LLM-rec confidence) plus an evaluation protocol.
5. **Risk: MEDIUM-HIGH.**
   - About 30–45 short LoRA runs.
   - The small matrix may be too small for homogenization to show.
   - Watch_ratio labels are noisy.
   - Captions are in Chinese, so Llama will be weaker; OLMo is not suitable here.
   - Novelty check: EchoTrace (CIKM'26) and RecLoop (2606.17707) use simulated or LLM user agents with no confidence signal. ABPO uses logged bandit feedback offline.

   **Effort:** weeks to 1 month.
6. **Seed questions:** S3 (primary), S5, S4.

---

## Idea 7 — The MNAR calibration illusion  [gen: opus-A]

1. **One-sentence summary.**
   - LLM-rec P(Yes) is always evaluated, and recalibrated, on items users *chose* to consume.
   - We test whether it stays calibrated on *randomly exposed* items, using the few datasets that have missing-at-random (MAR) test data with item text:
     - KuaiRec small matrix (captions);
     - Coat (attribute text, templated);
     - KuaiRand if text can be joined (unverified).
2. **Core hypothesis.**
   - A fine-tuned yes/no recommender learns "consumed ⇒ liked" positivity. It therefore looks calibrated on MNAR test pairs but is over-confident on MAR exposures.
   - The gap is largest for familiar/popular items, because self-selection correlates with familiarity.
   - Recalibrating on MNAR data cannot fix this, since the shift is item-dependent rather than a monotone map.
   - A tiny MAR slice (1–5%) with doubly-robust calibration should fix most of it.
3. **Minimum viable experiment (≈1.5 GPU-h).**
   - Short LoRA on KuaiRec big-matrix yes/no pairs (MNAR).
   - Evaluate on (a) held-out big-matrix pairs and (b) fully-observed small-matrix pairs.
   - Reliability diagrams and ECE by popularity tercile; compare isotonic fitted on MNAR vs on 2% MAR.
4. **Contribution type:** empirical finding plus diagnostic protocol (and a cheap fix).
5. **Risk: MEDIUM.**
   - Only 2–3 suitable datasets exist.
   - Coat is tiny and has no pretraining familiarity, which makes it a useful control but weak evidence.
   - Novelty check: doubly-calibrated MNAR estimators (2403.00817) calibrate *propensities* for classical debiasing. LLM-as-judge offline evaluation work (2606.22961) uses KuaiRec/Coat for evaluation. Neither measures LLM-rec confidence under MAR exposure.

   **Effort:** 2–3 weeks.
6. **Seed questions:** S4 (primary), S5, S3.

---

## Idea 8 — A natural-noise oracle for "prune by uncertainty": rating–review contradictions  [gen: opus-A]

1. **One-sentence summary.**
   - Use Amazon reviews whose star rating **contradicts their own review text** as silver-standard *real* label noise. An LLM or sentiment judge flags them by reading text the recommender never sees.
   - Then ask, with ≥ 5 seeds, which uncertainty signal finds that noise, and whether pruning or down-weighting it helps yes/no LLM-rec fine-tuning.
2. **Core hypothesis (H3 made testable).**
   - Raw training confidence or loss flags hard and tail positives (difficulty ≠ noise). Its precision on *real* noise is near random, which would replicate N3 and GORACS.
   - A cross-fitted **history-conditional contradiction** signal should do much better than random. It is the out-of-fold model's evidence term (the prior-removed log-odds of Idea 9, or the MIRROR difference of Idea 1) disagreeing with the label.
   - Down-weighting by that signal beats random-prune and RosePO-style smoothing only above a noise-rate threshold.
   - Deliverable: a phase diagram of *when* uncertainty denoising pays, anchored on real noise and extended with an injected-noise ladder.
3. **Minimum viable experiment (≤2 GPU-h).**
   - Judge 50k Beauty reviews for rating–text contradiction, and hand-check 200 to estimate precision.
   - In one GPU pass, score the training examples with zero-shot and one short out-of-fold LoRA, then compute detection AUROC of each signal against the silver noise. No full training yet.
   - **Kill gate:** if the contradiction rate is < 2% or judge precision is < 70%, fall back to injected noise only.
   - Full study:
     - 3 domains × {none, random-prune, confidence-prune, evidence-prune, evidence-weight, RosePO-smooth} × 5 seeds;
     - clean test labels, i.e. test pairs whose rating and text agree.
4. **Contribution type:** empirical finding plus a benchmark protocol (a real-noise oracle) and a small method (evidence-weighted BCE).
5. **Risk: MEDIUM.**
   - Sarcasm and mixed reviews will limit judge precision.
   - Gains may stay within 2σ_seed. That would still be a publishable, rigorous negative.
   - Novelty check: rating–review inconsistency is known as a phenomenon in classical RS. We found no use of it as ground truth for evaluating LLM-rec uncertainty denoising.

   **Effort:** 3–4 weeks.
6. **Seed questions:** S6 (primary), S2.

---

## Idea 9 — Evidence-gated serving: abstain when the confidence is borrowed from the item prior  [gen: opus-A]

1. **One-sentence summary.**
   - Estimate each item's "default yes" by averaging P(Yes) over K random **other users'** histories. This stays in-distribution, unlike empty-history or "N/A" contextual calibration.
   - Decompose confidence into item prior + history evidence.
   - Use the *evidence*, not the confidence, for cross-user decisions: whom to serve, and when to fall back to CF.
2. **Core hypothesis.**
   - Confidently-wrong predictions are prior-dominated: high P with low evidence (S2).
   - Selective serving by evidence beats selective serving by confidence on risk–coverage (AURC) and on tail-user fairness.
   - The reason is that confidence ranks users by how many *familiar* candidates they happened to get, not by how well the model knows them.
   - Caveat from the sibling project: prior-correction used as a *ranking* change was domain-conditional. Here the prior is used for *selection*, which N1 does not cover.
3. **Minimum viable experiment (≈1.5 GPU-h).**
   - Cost: the prior is user-independent, so it takes K = 16 forward passes per candidate item, once. That is roughly one extra scoring run.
   - Setup: one Amazon domain, 2k users × 20 rated candidates (or 101-candidate next-item with list-normalised probabilities).
   - Report AURC for {confidence, evidence, margin, entropy} against matched-random coverage.
   - **Kill gate:** drop the idea if AURC(evidence) is not better than AURC(confidence), with CI.
4. **Contribution type:** new method (decision rule) plus diagnostic.
   - It is a direct competitor estimator to MIRROR's `a(i)`.
   - The two can be compared head-to-head as two estimators of the same latent item prior.
5. **Risk: MEDIUM.**
   - The evidence term may be noisy for short histories.
   - Selective serving is textbook (CALIBRA already conceded this), so the novelty lies entirely in *what* is gated on.

   **Effort:** 1–2 weeks.
6. **Seed questions:** S2, S1, S5, S3.

---

## Idea 10 — Between-user calibration: the only calibration that matters for budgeted decisions  [gen: opus-A]

1. **One-sentence summary.**
   - Formalize that per-user monotone miscalibration is invisible to UAUC/NDCG, yet it fully determines *budgeted cross-user* decisions: "send 10k notifications across all users", global AUC, list truncation.
   - Define **between-user calibration error (BUCE)** as the calibration of user-level mean confidence against user-level like-rate.
   - Show that LLM P(Yes) has large BUCE driven by *prompt artifacts* such as history length, verbosity and the rating-leniency text.
2. **Core hypothesis.**
   - At a fixed true like-rate, mean P(Yes) rises with history length or token count.
   - A one-parameter-per-user offset predicted from cheap prompt features (history length, the user's mean shown rating) removes most BUCE. It raises budgeted precision@B and global AUC while leaving UAUC *exactly* unchanged, which demonstrates the decomposition empirically.
   - Theory:
     - a lemma stating N1 formally: any rule that depends only on within-user order is invariant to per-user monotone maps;
     - its exact complement: the regret of a cross-user thresholding rule is bounded by BUCE + within-user ECE.
3. **Minimum viable experiment (≤1 GPU-h).**
   - Data: ML-1M + Amazon Books, 3k users, rating labels.
   - A *causal* length test: for the same users and the same candidate label set, truncate the history to 5/10/20/40 items and measure the slope of mean logit against length.
   - Then compute BUCE before and after the offset correction.
4. **Contribution type:** theoretical result plus diagnostic plus a tiny method.
5. **Risk: LOW-MEDIUM.**
   - The length effect may be small. BUCE would still be a clean, reusable metric.
   - Novelty check: PerK (WWW'24) owns "calibrated probabilities → per-user K". Our delta is LLM-specific prompt-artifact miscalibration and budgeted decisions across users.

   **Effort:** 1–2 weeks.
6. **Seed questions:** S4 (primary), S1, S2.

---

## Idea 11 — Does thinking make LLM recommenders confidently wrong?  [gen: opus-A]

1. **One-sentence summary.**
   - Use Qwen3-8B's switchable thinking mode (plus an R1-distilled 8B) to measure whether reasoning before the Yes/No answer raises confidence more than accuracy in personalized recommendation.
   - The question is especially about familiar items, where the model can rationalize from brand knowledge.
2. **Core hypothesis.**
   - This is the recommendation analogue of the Hooi-group A6 result on factual QA.
   - Reasoning inflates confidence by confirmation ("X is a well-reviewed brand"). We expect:
     - higher head over-confidence;
     - larger head over-exposure;
     - flat AUC/UAUC;
     - help only for users with long, coherent histories.
   - Equally interesting opposite outcome: reasoning improves tail calibration by surfacing attributes.
3. **Minimum viable experiment (≤2 GPU-h).**
   - Setup: 1k users × 10 rated candidates, Amazon Toys + ML-1M, thinking on vs off (~400 thought tokens), with P(Yes) read at the answer token after the trace.
   - Report ECE and AUROC by popularity decile, and the rate of brand/popularity mentions in traces (regex).
4. **Contribution type:** empirical finding, i.e. one RQ section of the flagship observation study.
5. **Risk: LOW-MEDIUM.** **Effort:** days.
6. **Seed questions:** S1, S2, S5, S3.

---

## Triage table  [gen: opus-A]

| # | Idea | Type | Risk | Effort | Pilot GPU-h | Seeds answered |
|---|---|---|---|---|---|---|
| 1 | MIRROR (acquiescence gap + mirror scoring/tuning) | diagnostic + method | MED | weeks | ~1.5 | S4 S5 S2 S1 (S3) |
| 2 | Pseudonym knockout + name-dropout tuning | finding + method | MED-HIGH | weeks | ≤2 | S5 S2 S3 S1 |
| 3 | Exposure (open-corpus counts) explains the sign flip | finding + metric | MED | 1–3 wk | ~1 | S5 S2 S4 |
| 4 | Calibration as exchange rate: calibrated LLM⊕CF pooling | theory + method | MED-HIGH | days–2 wk | ≤1 | S4 S2 S1 |
| 5 | Confidence measures redundancy, not value | diagnostic | LOW | days | ≤1 | S1 S2 S5 |
| 6 | Ground-truth echo chamber on KuaiRec | finding + protocol | MED-HIGH | weeks–1 mo | ≤2 | S3 S5 S4 |
| 7 | MNAR calibration illusion | finding + diagnostic | MED | 2–3 wk | ~1.5 | S4 S5 S3 |
| 8 | Natural-noise oracle for uncertainty pruning | finding + benchmark | MED | 3–4 wk | ≤2 | S6 S2 |
| 9 | Evidence-gated serving (user-marginalized prior) | method + diagnostic | MED | 1–2 wk | ~1.5 | S2 S1 S5 S3 |
| 10 | Between-user calibration (BUCE) + budgeted decisions | theory + diagnostic | LOW-MED | 1–2 wk | ≤1 | S4 S1 S2 |
| 11 | Thinking mode and confident errors | finding | LOW-MED | days | ≤2 | S1 S2 S5 S3 |

## Suggested paper-carrying composition (one observation study → one method)  [gen: opus-A]

**Working title:** *"Yes Because It Knows the Item, or Because It Knows the User? Disentangling Familiarity from Preference in LLM Recommender Confidence."*

- **§3 Observation (RQ1–RQ4), on rating-labeled yes/no data across 4 datasets × 3 backbones:**
  - RQ1: is the LLM right when it is sure, and sure when it is right? (S1, S2, S4)
  - RQ2: a causal familiarity knockout (Idea 2, diagnostic only) plus exposure counts (Idea 3) show that confidence = familiarity + preference, which resolves the CALIBRA sign flip. (S5)
  - RQ3: the acquiescence gap `a(i)` (Idea 1, diagnostic) concentrates the confident errors. (S2)
  - RQ4: the exposure consequence, static (head over-exposure), plus Idea 11 as a short RQ if space allows. (S3)
- **§4 Method:** MIRROR scoring and mirror tuning (Idea 1). One mechanism, an identifiability argument, a proper-scoring BCE, and an item-dependent correction that legitimately changes ranking.
- **§5 Experiments:**
  - Metrics:
    - UAUC/NDCG for ranking;
    - ECE, Brier and EICE for calibration;
    - tail share and Gini;
    - AURC for selective serving.
  - Controls: the paraphrase-ensemble placebo and ≥ 3 seeds for the tuned variant.
  - Baselines: contextual calibration, PMI with the user-marginalized prior (Idea 9) as a competitor estimator, and PopProCal.
- **Fallback flagship:** Ideas 4 + 5 ("calibration is the exchange rate; confidence measures redundancy"), if MIRROR's kill gate fires.
- **Q6 (S6):** Idea 8 is better as a separate companion study. It needs 5-seed training grids that would crowd a 9-page paper.

## Deliberately not proposed (dead or owned)  [gen: opus-A]

- Uncertainty as a within-list ranking multiplier, or any monotone post-hoc recalibration presented as a ranking method (N1).
- Interaction-popularity-conditioned recalibration without a new latent variable (N2). Idea 3 re-tests it only as a pre-registered baseline.
- Entropy or loss-based hard pruning for generative DPO (N3).
- Plackett–Luce listwise UQ and uncertainty-aware prompting (UQRec), confidence tokens and confident-error RL rewards (UGR), position-level calibration (EviRank), and conformal list truncation or guarantees (owned by sibling TRUCE-Rec, PerK and CRC).

## Sources consulted for novelty sanity checks

Every item below was retrieved in this session. Where only a search-result title or snippet was seen, the entry says so.

- KuaiRec repo: captions file `kuairec_caption_category.csv` (caption, manual_cover_text, topic_tag, categories), added 2024-06-02. [GitHub chongminggao/KuaiRec](https://github.com/chongminggao/KuaiRec) — fetched.
- KuaiRec paper: small matrix 1,411 × 3,327, 99.6% dense. [arXiv 2202.10842](https://arxiv.org/abs/2202.10842) — search result.
- Incumbent Advantage: brand bias in LLM recommendation (zero-shot, fictional vs real brands). [arXiv 2606.17443](https://arxiv.org/abs/2606.17443) — search-result snippet only.
- RouteRec: hard selection below BM25; selective LLM escalation does not help. [arXiv 2607.09908](https://arxiv.org/abs/2607.09908) — abstract fetched.
- Rethinking Semantic Collaborative Integration (SIGIR 2026): complementarity diagnostics, no confidence-based fusion. [arXiv 2604.22195](https://arxiv.org/abs/2604.22195) — abstract fetched.
- The yes-no bias of LLMs reflects answer order and wording. [arXiv 2607.05552](https://arxiv.org/abs/2607.05552) — search-result title and snippet only.
- Towards Objective Fine-tuning: prior knowledge causes poor calibration. [arXiv 2505.20903](https://arxiv.org/abs/2505.20903) — search-result title and snippet only.
- Doubly Calibrated Estimator for recommendation on MNAR data. [arXiv 2403.00817](https://arxiv.org/pdf/2403.00817) — search result only.
- LLM-as-a-Judge for offline evaluation in top-K recommendation (uses KuaiRec/Coat). [arXiv 2606.22961](https://arxiv.org/abs/2606.22961) — search result only.
- LRWorld, "Mental World of LLMs in Recommendation": no confidence or infini-gram analysis in the abstract. [arXiv 2512.17389](https://arxiv.org/abs/2512.17389) — abstract fetched.
- RecLoop closed-loop cocoon simulation. [arXiv 2606.17707](https://arxiv.org/abs/2606.17707) — search result (already covered in LIT_RECENT).
- Pretraining exposure explains popularity judgments (SIGIR'26 short). [arXiv 2605.12382](https://arxiv.org/abs/2605.12382) — search result (already covered in LIT_RECENT).
- Unresolved: one search snippet stated that prior work used infini-gram item-name counts to correlate with LLM-rec *accuracy*. The specific paper was not identified, and it must be found before Idea 3 is written up.
