# Brainstorm B: uncertainty-aware LLM4Rec through a causal-inference and decision-theory lens

Generator: opus-B. Date: 2026-10-01. Input: `brainstorm_bundle.md` and every file it lists (RESEARCH_BRIEF,
LIT_HOOI_GROUP, LIT_RECENT_UNC_LLM4REC, CALIBRA_draft, the UCalRec NEGATIVE_RESULTS and ADVISOR_MEMO).

**Lens.** I approached this as a researcher who works on exposure, counterfactual evaluation, feedback loops,
deferral and selective prediction, and proper scoring rules. That lens reads the brief's four negatives (N1–N4)
as one diagnosis: **the project has been scoring confidence against the logging policy's labels and acting on it
inside a single list.** In causal terms, both of those are the wrong estimand.

**Status of the numbers.** Every number in a "Core hypothesis" field is a guess made before any experiment. None
of them is a measurement.

## The six seed questions (labels used below)

| Tag | Seed question (from the user's own words) |
|---|---|
| S1 | When the LLM is right, is it actually sure? |
| S2 | Are wrong answers wrong *because* confidence was low, or are they confidently wrong? |
| S3 | Does high confidence cause an echo chamber? |
| S4 | How does yes/no confidence correlate with ground truth? |
| S5 | Do popular items get high confidence and niche items low confidence? |
| S6 | Can uncertainty be used to prune noisy training data? |

## Assumptions I deliberately invert

- **INV-1: "ground truth is the yardstick" (S4, S2).** A logged label in a recommender is the output of a
  logging policy, not a measure of preference: an item you never showed cannot be clicked. So a "confidently
  wrong" prediction may in fact be **confidently right about an item that was never labelled**. Calibration
  measured on logs is calibration against the exposure policy. → Ideas 1, 2.
- **INV-2: "high confidence causes the echo chamber" (S3).** Reverse the causal arrow: the feedback loop
  produces the confidence. Retraining on self-exposed logs makes the model *look* well calibrated on what it
  shows, while it becomes *under-confident* on what it hides. The echo chamber runs on unwarranted **low**
  confidence about items that were never exposed, and on **acting on low confidence** by abstaining.
  → Ideas 2, 10.
- **INV-3: "prune what the model is uncertain about" (S6).** In implicit-feedback recommendation, the examples
  that carry no preference information are the ones the model is *already sure of* (prior-congruent,
  high-propensity positives), plus false negatives among the sampled negatives. So prune the familiar, not the
  uncertain. → Idea 4.
- **INV-4: "ask the LLM whether it is sure" (S1, S2).** For a serve-or-fallback decision, the estimand is
  *comparative*: P(LLM right) − P(fallback right). It also has to be computed **before** the LLM is called, so
  the right question is whether the cheap fallback is unsure. → Idea 3.
- **INV-5: "item popularity is the axis" (S5).** Move to the user side: confidence-gated serving may silence
  niche *users*, not just niche items. → Ideas 6, 10.

---

## Idea 1: Confidently wrong, or confidently unlabelled? Exposure-confounded calibration of LLM recommenders  [gen: opus-B]

1. **Summary.** Measure P(Yes) calibration of yes/no LLM recommenders against two kinds of labels: logged MNAR
   labels and missing-at-random (MAR) or fully observed ground truth. Show that a large share of "overconfidence"
   and the popularity–confidence pattern is a property of the *label-generating exposure*, not of the LLM.
2. **Core hypothesis.**
   - On KuaiRec, a sizeable fraction (guess: ≥30%) of top-decile-confidence "false positives" are true positives
     under the fully observed matrix. Logged ECE therefore overstates overconfidence.
   - The head-vs-tail calibration gap measured on logs shrinks or flips sign under MAR labels. This would
     explain CALIBRA's finding that the `is_head` coefficient flips sign across domains: the sign follows each
     domain's exposure concentration (interaction Gini), not the LLM.
   - **"Fine-tuning learns the logging policy."** TALLRec-style LoRA on MNAR logs improves logged ECE and AUC
     but *worsens* MAR ECE relative to zero-shot. The pretraining prior is partly de-confounded from the
     platform's exposure, and fine-tuning re-confounds it.
   - Mechanism: the observed label factorises as y_obs = exposure × preference. Any calibration metric that
     uses y_obs therefore scores the model's ability to predict exposure as well as preference.
3. **Minimum viable experiment (≤2 GPU-h).**
   - Setup: Qwen3-8B zero-shot, prefill-only P(Yes) (softmax over {Yes, No}). Run 300 KuaiRec small-matrix users
     × 200 items, about 60k prompts, roughly 30 minutes with vLLM `prompt_logprobs`.
   - Compute reliability diagrams, ECE and Brier (calibration/refinement split) under:
     - (a) MAR labels from the small matrix;
     - (b) an MNAR "view": labels only on pairs observed in the big matrix, other pairs treated as sampled
       negatives.
   - Report the share of confident false positives that are true positives under (a), plus the ProCal Bias Index
     per popularity tercile under each view.
   - Full study:
     - Coat (MAR test ratings, verbalisable item attributes);
     - MIND (impression logs separate exposed-not-clicked from never-exposed);
     - one Amazon-2023 domain (MNAR only, for scale);
     - LoRA before vs after on each.
   - Practicalities (all to verify):
     - **Pair overlap.** Check whether KuaiRec's big and small matrices share pairs; drop any shared pairs from
       training.
     - **Watch-ratio threshold.** Run a sensitivity check at 1.0, 2.0 and 3.0.
     - **Text quality.** Captions are short Chinese text, which suits Qwen.
4. **Contribution type.** Empirical finding plus diagnostic (an "exposure gap" in ECE: logged ECE − MAR ECE).
5. **Risk: MEDIUM.**
   - Zero-shot AUC on KuaiRec captions may be near 0.5. Mitigation: LoRA on the big matrix, about 1–2 GPU-h.
   - Coat is tiny.
   - KuaiRec's forced exposure may itself change behaviour.
   - Kweon et al. (AAAI'22) already do unbiased calibration under MNAR for classic recommenders. Our novelty is
     the LLM-specific re-confounding by fine-tuning and the confident-error audit.
6. **Effort.** 2–3 weeks.
7. **Seeds answered.** S4, S2 (inverted: confidently *unlabelled*), S5.

## Idea 2: Self-confirming calibration. LLM recommenders retrained on their own exposure become calibrated on what they show and wrong about what they hide  [gen: opus-B]

1. **Summary.**
   - In a closed loop, a yes/no LLM recommender is retrained on feedback to its own top-K. It converges to a
     *self-confirming equilibrium* in the sense of Fudenberg–Levine: beliefs are correct on-path (exposed items)
     and biased off-path (never-exposed items).
   - Logged calibration *improves* while counterfactual (MAR) calibration of the tail degrades.
   - An **exposure-weighted proper score** removes this fixed point. This is one mechanism: score the
     confidence against the counterfactual distribution, not the logged one.
2. **Core hypothesis.**
   - **Proposition (to prove).** Train with log-loss on shown items plus unexposed items sampled as negatives.
     The population minimiser is q(u,i) = P(exposed | u,i) · P(like | u,i). Confidence for rarely exposed items
     then decays toward 0, which reduces their exposure further. The fixed point is self-confirming, and
     log-based ECE stays small because the label carries the same exposure factor.
   - **IPS-weighted fix.** With a stochastic serving policy (Plackett–Luce sampling over q, so positivity holds)
     and clipped weights, the IPS-weighted Brier or log-loss has minimiser P(like | u,i). The counterfactual
     calibrated point becomes the fixed point.
   - **LLM-specific prediction ("knowledge erosion").** The zero-shot LLM starts with *better* tail calibration
     than MF/SASRec (world knowledge). The naive loop erodes it to the CF-like self-confirming state within a few
     rounds; the IPS loop preserves it.
   - Guessed effect sizes:
     - naive loop: logged ECE ≤0.03 throughout, tail MAR ECE up 2–3×, tail exposure down 30–50% by round 5
       (dominated by *under*-confidence);
     - IPS loop: flat MAR ECE, higher tail exposure, cumulative true utility within 1–2% (possibly higher after
       round 3 through exploration).
3. **Minimum viable experiment (≤2 GPU-h).**
   - The loop uses the KuaiRec small matrix as an **oracle user**: feedback is real fully observed preference,
     so no LLM user simulator is needed and the effect is isolated to the model side.
   - Step 1 (minutes, CPU): run the loop with MF to check the proposition (naive vs IPS arms).
   - Step 2 (one LLM round):
     - warm-start LoRA P(Yes) scorer;
     - 500 users × 200-item pools;
     - show top-10 (or PL-sampled);
     - one retraining step per arm (naive N2: unexposed-as-negative; naive N1: shown only; IPS);
     - measure Δ(MAR tail ECE), Δ(logged ECE), Δ(tail exposure).
   - Full study: 5 rounds × 3 seeds × 3 arms. Estimated ~25–70 GPU-h depending on pool size, plus MF/SASRec in
     the same loop.
4. **Contribution type.** Theoretical result (self-confirming fixed point; log-based calibration is
   uninformative about counterfactual calibration), a new method (exposure-weighted proper scoring for LLM-rec
   fine-tuning), and an empirical result (erosion of the LLM prior).
5. **Risk: MEDIUM–HIGH.**
   - Critiques of simulator fidelity. Mitigation: real fully observed feedback; static preferences are a
     deliberate control.
   - IPS variance. Mitigation: clipping and self-normalisation.
   - Novelty pressure from Krauth et al. ("Breaking feedback loops ... with causal inference", TORS; CAFL) and
     from economics work on self-confirming curator beliefs. Our deltas: calibration as the measured object,
     LLM-prior erosion, and the "logged calibration is uninformative" result.
6. **Effort.** 4–5 weeks. This is the strongest candidate to carry a full paper, with Idea 1 as its
   observation study.
7. **Seeds answered.** S3 (inverted: the loop causes the confidence, and the engine is tail
   *under*-confidence), S5 (sign predicted to flip after the loop), S4.

## Idea 3: Ask the fallback, not the LLM. Pre-hoc comparative deferral for LLM↔CF serving, and the winner's curse in oracle routing  [gen: opus-B]

1. **Summary.**
   - Chow's rule (defer when your own confidence is low) is Bayes-optimal only when the fallback's accuracy is
     constant.
   - In recommendation the LLM and the CF model are right on the *same* easy, head-dominated users. The deferral
     estimand is therefore comparative.
   - Operationally it must be computed **before** paying for the LLM call, so gate on the CF model's
     uncertainty, trained with a consistent learning-to-defer surrogate (Mozannar–Sontag 2020).
   - First, correct the oracle for the winner's curse.
2. **Core hypothesis.**
   - **(i) The +21% oracle is inflated.** It takes, per user, the max of two noisy single-item NDCGs. A
     cross-fitted oracle (route on a validation item, score on a test item) will show far smaller achievable
     headroom (guess: +3–8%). This may by itself explain why the LLM-confidence gate had a correlation of only
     0.07, and RouteRec's negative.
   - **(ii) The LLM's advantage over CF concentrates where CF is uncertain**: high CF softmax entropy, short
     histories, low novelty in CF's top-K. Guess: AUROC 0.62–0.70 for "LLM beats CF", versus ~0.52 for LLM
     confidence.
   - **(iii)** At a 30% LLM-call budget, a pre-hoc gate recovers ≥50% of the *cross-fitted* headroom.
3. **Minimum viable experiment (≤1 GPU-h).**
   - Use a leave-two-out split (validation and test item per user) on two Amazon domains.
   - Train SASRec (minutes). Use the C-CRP/P(Yes) scores, regenerated.
   - Compute the naive vs cross-fitted oracle.
   - Compare gate AUROC for the target 1[NDCG_LLM > NDCG_CF] across features: LLM confidence, CF entropy and
     margin, history length, mean log-popularity of the history.
   - Fit a logistic L2D gate and plot an NDCG-vs-LLM-call-budget curve.
4. **Contribution type.** Diagnostic (winner's curse in recommender-routing headroom) plus a simple method
   (pre-hoc comparative deferral with budget).
5. **Risk: MEDIUM.**
   - Cross-fitted headroom may be ≈0. That is still publishable as a negative that reframes RouteRec and the
     internal N4.
   - The routing space is crowded: S-LLMR (gate on history length, popularity and uncertainty), LSC4Rec,
     confidence-token routers for LLMs, ReDAct. Our deltas are the winner's-curse correction, the comparative
     L2D target and the pre-hoc constraint.
6. **Effort.** 1–2 weeks.
7. **Seeds answered.** S1 and S2, reframed (which system is sure *relative to the other*); also key gap 3.

## Idea 4: Prune the familiar, not the uncertain. Prior-congruence pruning for yes/no LLM-rec fine-tuning  [gen: opus-B]

1. **Summary.**
   - UNIT cuts training claims the model is *unfamiliar* with. For recommendation, do the opposite.
   - Drop the positives the base LLM already predicts **from the item alone**: high history-free P(Yes | ∅, i).
     Their preference information is near zero (the IPS view: a high-propensity click says little about
     preference). Their gradient mainly amplifies the genericness prior (UCalRec: head-ECE 0.075 → 0.143 after
     training).
   - Analogy: word2vec's subsampling of frequent words, with an LLM-computed prior.
2. **Core hypothesis.**
   - At matched 25% pruning, prior-congruence pruning is within 1σ of random on overall NDCG.
   - It beats random on head ECE (guess: −30%) and on tail HR@10 (guess: +2–4 pt).
   - Pruning by *dataset* popularity does worse than pruning by the *LLM* prior, because the LLM prior and
     dataset popularity diverge (KnowSA).
   - UNIT-style "prune the most uncertain" does worst, because it removes hard tail cases (hard–noisy
     confusion).
   - Falsification: if every arm ties random over 5 seeds × 3 domains, that is a clean principled-inverse negative
     that strengthens N3.
3. **Minimum viable experiment (~3 GPU-h, slightly over the target).**
   - One history-free forward pass over 20k training pairs (~10 min).
   - TALLRec-style LoRA, 3 seeds × 2 arms (prior-prune vs random-prune) on one Amazon domain with rating labels.
   - Report UAUC, NDCG over 101 candidates, head ECE (rating-matched labels), tail HR.
4. **Contribution type.** New method plus empirical finding (either sign).
5. **Risk: HIGH.**
   - Every targeted-pruning signal tested so far has tied random (N3; GORACS; MiniRec).
   - UCalRec's "cross-model typicality" and its head label-smoothing were both null in generative DPO. This idea
     differs (pointwise SFT; positives pruned by an item-only prior; prior-amplification metrics as the primary
     endpoint), but reviewers will connect them.
6. **Effort.** 2–3 weeks (≥5 seeds required).
7. **Seeds answered.** S6 (inverted), S5.

## Idea 5: Is popularity a cause of confidence, or a correlate? Mediation analysis by concept erasure  [gen: opus-B]

1. **Summary.**
   - Ask the counterfactual question: what would P(Yes) be if the model could not represent the item's
     popularity?
   - Use LEACE (closed-form linear concept erasure) on the residual stream at the answer position, for two
     candidate mediators taken separately:
     - M1: platform popularity (log interaction count);
     - M2: pretraining familiarity (base-LM title log-likelihood per token, or a KnowSA-style probe).
   - Read the change in confidence as a natural indirect effect. Measure calibration per popularity bucket,
     NDCG, and tail exposure.
2. **Core hypothesis.**
   - Confidence inflation on head items is mediated mostly by **M2, not M1**.
   - Erasing M2 cuts head over-confidence by half with less than a 2% relative NDCG loss.
   - Erasing M1 *hurts* NDCG, because platform popularity is legitimately predictive.
   - Either way, the result decomposes S5's "popular = confident" into a legitimate part and a prior-inherited
     part. Item-specific erasure changes the order within a list, so it escapes N1.
3. **Minimum viable experiment (≤2 GPU-h).**
   - HF forward hooks on Qwen3-8B.
   - 1,000 users × 101 candidates on one domain: cache last-token states at 3 layers.
   - Fit the probes, apply LEACE, recompute P(Yes).
   - Report ΔECE per tercile, ΔNDCG@10, Δ tail share in top-10.
4. **Contribution type.** Diagnostic (causal mediation of confidence) plus a light inference-time method.
5. **Risk: MEDIUM–HIGH.**
   - Popularity steering already exists: SPREE (FAccT'26) for sequential recommenders, PopSteer for neurons,
     and activation steering for LLM-recommender popularity. Our deltas are *confidence/calibration* as the
     outcome and the M1-vs-M2 separation.
   - Linear erasure is a lower bound on mediation, not a full identification.
6. **Effort.** ~2 weeks.
7. **Seeds answered.** S5 (causal version), S2 (do confident errors run through the familiarity mediator?).

## Idea 6: Selective recommendation silences niche users  [gen: opus-B]

1. **Summary.**
   - Jones et al. (ICLR'21) showed that selective classification can *magnify* group disparities. Test whether
     LLM selective serving does the same, using CALIBRA's +0.036–0.083 NDCG@10 at 50% coverage.
   - Check whether that gain is mostly a **composition shift**: serving mainstream users and abstaining on niche
     users.
   - Check whether niche users are exactly where the LLM beats CF, in which case abstain-to-CF hurts them twice.
2. **Core hypothesis.**
   - Abstention concentrates on the lowest-mainstreamness quintile (guess: 2× its share).
   - In an Oaxaca-style split, the between-group composition effect accounts for over half of the selective
     gain.
   - Within the niche group, LLM−CF NDCG is *larger* than within the mainstream group.
   - Fix: group-conditional (Mondrian) conformal risk control thresholds, which give served-risk ≤ α *per
     mainstreamness group* with coverage parity at a small total cost.
3. **Minimum viable experiment (CPU only, after rescoring).**
   - Mainstreamness = mean log-popularity of a user's history, or similarity to the average user.
   - Compute risk–coverage per group and the decomposition of selective gain.
   - Compute LLM−CF per group with SASRec.
   - Two domains.
4. **Contribution type.** Empirical finding plus a simple fix.
5. **Risk: LOW–MEDIUM.**
   - The finding is likely to hold. Novelty is moderate: mainstream-bias literature (Zhu & Caverlee; FairLRM
     user-side popularity) plus Jones et al.
   - The LLM-specific twist (is the LLM better or worse than CF for niche users in the abstained region?) is
     what makes it non-obvious.
6. **Effort.** 1–2 weeks.
7. **Seeds answered.** S5 (inverted to users), S1.

## Idea 7: Is the yes/no LLM blind to user leniency? Per-user label shift in P(Yes)  [gen: opus-B]

1. **Summary.**
   - UCalRec found the recommender's item-vs-item preference is history-content-invariant. That predicts P(Yes)
     ignores each user's base like-rate (rating leniency).
   - The decision-theoretic fix is a per-user prior shift (Saerens EM or a logit offset from the user's training
     like-rate). It is monotone within each user, so UAUC is unchanged (consistent with N1). It does change global
     AUC, LogLoss and every cross-user threshold decision (serve/abstain, notification budgets).
2. **Core hypothesis.**
   - The user like-rate explains less than 5% of the variance in logit P(Yes), but 15–25% of label variance.
   - The offset gives a global AUC gain of +2–4 pt, a large LogLoss/ECE gain, and identical UAUC.
   - TALLRec/CoLLM's AUC−UAUC gap is largely this missing user prior.
3. **Minimum viable experiment (≤2 GPU-h).**
   - TALLRec-style LoRA on ML-1M (rating ≥4 = Yes), 1 seed.
   - Regress logit P(Yes) on the user like-rate.
   - Apply the offset; report AUC, UAUC, LogLoss and ECE before and after.
4. **Contribution type.** Empirical finding plus a trivial fix (section-sized, not paper-sized).
5. **Risk: LOW.** User-wise calibration exists (Kweon's dissertation and PerK; field-aware calibration, Pan et
   al. WWW'20). The contribution is the LLM-specific diagnosis.
6. **Effort.** ~1 week.
7. **Seeds answered.** S4, S1.

## Idea 8: Be honest where you act. Decision-weighted proper scoring for P(Yes) fine-tuning  [gen: opus-B]

1. **Summary.**
   - By the Schervish representation, every proper scoring rule is a mixture over thresholds c of cost-weighted
     misclassification losses. Log-loss and Brier spread that weight over all c.
   - Recommendation *decisions* happen in a narrow band: the abstention threshold, and the score at which an item
     enters the top-K.
   - Fine-tune P(Yes) with a Beta-family proper score (Buja–Stuetzle–Shen 2005) whose weight w(c) is
     concentrated on that band. It is still proper, so calibration is preserved, but capacity is reallocated to
     where decisions are made.
2. **Core hypothesis.**
   - At equal AUC/UAUC, the decision-weighted score improves selective-serving AURC (guess: 5–10% relative) and
     top-decile ECE, compared with BCE (TALLRec) and Brier (ConfTuner with N=1).
   - Focal loss, which is improper, matches it on ECE but loses on AURC.
   - Proposition: for a fixed decision threshold τ, the expected decision regret is bounded by the w-weighted
     score's excess risk. The weighting is therefore decision-optimal, not a heuristic.
3. **Minimum viable experiment (~2 GPU-h).**
   - Closed-form loss via incomplete beta functions (differentiable).
   - LoRA on one domain, BCE vs Beta-weighted (band set at the validation 50%-coverage threshold), 2 seeds as a
     pilot.
   - Report AURC, top-decile ECE, UAUC.
4. **Contribution type.** New method plus a small theoretical result.
5. **Risk: MEDIUM.**
   - Gains may sit inside seed noise (σ≈1.5 pt), so ≥5 seeds are needed.
   - UGR owns confidence tokens and rejection, but not decision-weighted proper scoring.
6. **Effort.** 2–3 weeks.
7. **Seeds answered.** S4, S1.

## Idea 9: Counterfactual personalisation confidence. Score the effect of the user, not the item  [gen: opus-B]

1. **Summary.** The confidence that matters for *personalised* recommendation is the causal effect of this user's
   history on the judgment:

       CPC(u,i) = logit P(Yes | h_u, i) − E_{u'} logit P(Yes | h_{u'}, i)

   It is estimated by swapping in K random other users' histories. Use CPC, not raw P(Yes), as the
   failure-prediction and selective-serving signal.
2. **Core hypothesis.**
   - Raw P(Yes) failure-AUROC is largely the item prior: its correlation with log-popularity is high (guess:
     ρ≈0.4).
   - CPC is nearly popularity-free (ρ≈0.1).
   - CPC gives higher failure-AUROC on tail items and better tail-slice selective NDCG at matched coverage, even
     if overall AUROC is similar.
   - For LoRA-tuned models CPC may be tiny, because SFT collapses history sensitivity (UCalRec). That would
     itself be a striking calibration-relevant finding for pointwise models.
3. **Minimum viable experiment (≤2 GPU-h).** 500 users × 101 candidates × (1 own + 4 swapped histories), about
   250k prefill prompts. Report the AUROC of raw P(Yes) vs CPC per popularity tercile.
4. **Contribution type.** Diagnostic plus a confidence estimator.
5. **Risk: MEDIUM.**
   - Overlaps with UCalRec's personalisation index (generative margins) and with contextual calibration (Zhao et
     al. 2021, PMI scoring).
   - The sibling found a serve-time prior correction to be domain-conditional. Here CPC is a *confidence*
     signal for cross-user decisions, not a ranking score, but the overlap must be stated.
6. **Effort.** 1–2 weeks.
7. **Seeds answered.** S1, S2, S5.

## Idea 10: The abstention feedback loop. Acting on *low* confidence builds the echo chamber  [gen: opus-B]

1. **Summary.**
   - Combine selective serving with retraining.
   - Users the LLM is unsure about are routed to CF. Their feedback then mostly trains CF.
   - The LLM's future training data over-represents mainstream users, which lowers its confidence on niche users
     and raises their abstention rate in the next round: self-reinforcing exclusion.
   - Fix: a randomised abstention band (ε-serving of the uncertain region) plus selection-propensity (IPS)
     weighting of the LLM's training data.
2. **Core hypothesis.**
   - Over 5 rounds, the niche-user abstention rate rises (guess: 50% → 70%) even though the LLM's *true* (MAR)
     advantage on those users is flat.
   - The randomised band plus IPS holds the abstention rate flat at under 1 pt of cumulative utility cost.
   - Reversed causal story for S3: the echo chamber comes from *abstaining* on low confidence, not from serving
     on high confidence.
3. **Minimum viable experiment.** Reuse the Idea 2 KuaiRec oracle loop with an MF fallback. Run 2 rounds × 1 seed
   × {gated, gated+random band}, about 2–3 GPU-h. First validate with an MF-only "LLM" stand-in on CPU.
4. **Contribution type.** Empirical finding (dynamics) plus a simple method.
5. **Risk: MEDIUM–HIGH.** It depends on the loop infrastructure from Idea 2 and inherits its critiques about
   simulator fidelity. The niche/mainstream split in KuaiRec may be weak.
6. **Effort.** +1–2 weeks on top of Idea 2.
7. **Seeds answered.** S3 (inverted), S5 (user side), S1.

## Idea 11: Does deliberation manufacture confidence about taste? Thinking-mode calibration of LLM recommenders  [gen: opus-B]

1. **Summary.**
   - Qwen3-8B can switch thinking on and off for free. Compare P(Yes) and verbalised confidence with and without
     thinking.
   - Use **rating-matched labels** (like/dislike), so ECE is well posed.
   - Measure overall and per popularity bucket.
2. **Core hypothesis.** The QA literature gives opposite predictions, so the answer matters either way:
   - "Reasoning models better express their confidence" (Yoon et al. 2025, Qwen3 thinking vs non-thinking);
   - "over-reasoning impairs calibration" and test-time scaling raises hallucination on knowledge tasks (Hooi
     group, COLM'26).
   - Our prediction: preference information that is absent from the history cannot be reasoned into existence.
     Thinking therefore raises confidence and the share of confident errors on **tail items and niche users**,
     with no NDCG gain, while improving calibration on head items, where world knowledge helps.
   - S2 answer: reasoning converts "unsure-wrong" into "confidently wrong" on the tail.
3. **Minimum viable experiment (~2 GPU-h).**
   - 300 users × 20 rated candidates on ML-1M plus one Amazon domain (12k prompts × 2 modes).
   - Read P(Yes) after `</think>Answer:`.
   - Report ECE and AUROC by popularity tercile, and confidence drift against reasoning length.
4. **Contribution type.** Empirical finding (one RQ or a short paper).
5. **Risk: MEDIUM.**
   - Thinking-mode decoding is expensive at full scale.
   - ThinkRec, ReaRec and R2ec cover reasoning-based recommendation, but not its calibration.
6. **Effort.** 1–2 weeks.
7. **Seeds answered.** S2, S1, S5.

---

## Filtering: ranking and recommended composition

| # | Idea | Risk | Effort | Pilot cost | Inverts | Paper-carrying? |
|---|---|---|---|---|---|---|
| 1 | Confidently wrong vs confidently unlabelled (MNAR vs MAR calibration) | MED | 2–3 wk | ~0.5–2 GPU-h | INV-1 | Observation half |
| 2 | Self-confirming calibration + exposure-weighted proper scoring | MED–HIGH | 4–5 wk | ≤2 GPU-h | INV-2 | **Yes (method half)** |
| 3 | Ask the fallback + winner's-curse-corrected routing | MED | 1–2 wk | ≤1 GPU-h | INV-4 | Strong section or short paper |
| 4 | Prune the familiar, not the uncertain | HIGH | 2–3 wk | ~3 GPU-h | INV-3 | If positive |
| 5 | Mediation of confidence by concept erasure | MED–HIGH | 2 wk | ≤2 GPU-h | — | Section |
| 6 | Selective serving silences niche users | LOW–MED | 1–2 wk | CPU | INV-5 | Section |
| 7 | Per-user label shift / leniency blindness | LOW | 1 wk | ≤2 GPU-h | — | Section |
| 8 | Decision-weighted proper scoring | MED | 2–3 wk | ~2 GPU-h | — | Possible method |
| 9 | Counterfactual personalisation confidence | MED | 1–2 wk | ≤2 GPU-h | — | Section |
| 10 | Abstention feedback loop | MED–HIGH | +1–2 wk | ~2–3 GPU-h | INV-2, INV-5 | RQ inside the paper from Idea 2 |
| 11 | Thinking-mode calibration | MED | 1–2 wk | ~2 GPU-h | — | RQ / short paper |

**Recommended SIGIR full paper: Idea 1 → Idea 2, with Idea 10 as an RQ.**

- One-sentence pitch: *"LLM-recommender confidence is calibrated against the platform's exposure, not users'
  preferences; retraining on self-exposure makes this a self-confirming equilibrium; an exposure-weighted proper
  score is the principled fix."*
- How it maps onto the seeds:
  - S4 and S2: the observation study shows confidently-*unlabelled* errors.
  - S5: popularity–confidence is exposure-driven, and its sign flips under MAR labels.
  - S3: the loop *produces* tail under-confidence.
  - S6, reframed: the "noise" sits in the unexposed negatives, which IPS down-weights instead of pruning.
- One mechanism, restatable in one breath. It escapes N1, because training and exposure are not within-list
  monotone, and it escapes N2, because the conditioning variable is exposure, not interaction popularity.
- Datasets:
  - KuaiRec (fully observed);
  - Coat (MAR);
  - MIND (exposure-known impressions);
  - one or two Amazon-2023 domains for scale and for an internal replication of the CALIBRA sign flip.

**Fast independent hedge: Idea 3.** It costs about one GPU-hour, and either outcome is a result: positive means
pre-hoc deferral works; negative means oracle routing headroom is the winner's curse. Run it first.

**Suggested pilot kill-order (cheapest decisive first):**

1. Idea 3: cross-fitted oracle.
2. Idea 1: KuaiRec MNAR-vs-MAR reliability.
3. Idea 2: MF-only loop check of the proposition, then one LLM round.
4. Idea 6: decomposition (CPU).
5. Idea 4, only if a positive is still needed for S6.

## Novelty sanity checks run for this brainstorm (web, 2026-10-01)

- **Feedback loops + confidence calibration.** No paper found that studies *calibration* of LLM recommenders
  under self-exposure retraining.
  - Krauth et al. (CAFL) break feedback loops causally for classic recommenders, but not as a calibration object.
  - "Calibrating the Evaluator" (arXiv 2606.31371) studies LLM-judge feedback loops, not recommendation.
  - Economics/recommender work on self-confirming curator beliefs exists. Cite it as the conceptual ancestor.
- **Popularity steering and erasure.**
  - SPREE (arXiv 2604.01036, FAccT'26): steering for *sequential* recommenders.
  - PopSteer (arXiv 2601.15122): neuron-level steering.
  - An activation-steering method for LLM-recommender popularity bias also appears in search.
  - None, as far as the search shows, measures the mediation of *confidence/calibration*. This is why Idea 5 is
    rated MEDIUM–HIGH risk.
- **LLM false-negative relabelling** is crowded: Dual-Tree LLM-enhanced negative sampling, LLM hard-negative
  sampling, and generative pseudo-labelling. I therefore folded the "noise is in the negatives" inversion into
  Ideas 1 and 2 (fully observed audit plus IPS), rather than making it a standalone method.
- **Routing.** RouteRec (negative), S-LLMR, confidence-token LLM routers and ReDAct exist. No winner's-curse
  correction of recommender-routing headroom was found.
- **Reasoning calibration.** Yoon et al. 2025 (Qwen3 thinking improves calibration) and "Don't Think Twice"
  (2508.15050) exist for QA. Nothing was found for recommendation.
- **User-wise calibration** exists (Kweon dissertation 2402.16325; PerK). Idea 7 is diagnostic only.

Sources:
- [Breaking Feedback Loops in Recommender Systems with Causal Inference](https://arxiv.org/pdf/2207.01616)
- [Calibrating the Evaluator (LLM agent feedback loops)](https://arxiv.org/abs/2606.31371)
- [Aligning Recommendations with User Popularity Preferences (SPREE)](https://arxiv.org/abs/2604.01036)
- [PopSteer: Interpretable Neuron Steering for Popularity Bias](https://arxiv.org/abs/2601.15122)
- [KuaiRec: A Fully-observed Dataset](https://arxiv.org/abs/2202.10842)
- [RouteRec](https://arxiv.org/pdf/2607.09908)
- [Learning to Route LLMs with Confidence Tokens](https://arxiv.org/pdf/2410.13284)
- [ReDAct: Uncertainty-Aware Deferral for LLM Agents](https://arxiv.org/pdf/2604.07036)
- [Reasoning Models Better Express Their Confidence](https://arxiv.org/pdf/2505.14489)
- [Don't Think Twice! Over-Reasoning Impairs Confidence Calibration](https://arxiv.org/pdf/2508.15050)
- [Dual-Tree LLM-Enhanced Negative Sampling](https://pith.science/paper/2602.18249)
- [Hard Negative Sampling via LLMs for Recommendation](https://arxiv.org/html/2504.04726)
- [Confidence Calibration for Recommender Systems and Its Applications](https://arxiv.org/pdf/2402.16325)
- [Degenerate Feedback Loops in Recommender Systems](https://arxiv.org/html/1902.10730v3)
