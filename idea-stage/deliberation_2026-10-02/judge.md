# ARIS judge synthesis (2026-10-02) — review_independence: same-family (Claude); provisional

## Verdict on gate
GATE_FAIL is final for the registered prompt V0. From decision.json:
- ML-1M raw-like UAUC is 0.5874, below the 0.60 bar.
- Sports raw-next NDCG@10 is 0.2093, above the 0.18632 bar. C-CRP scores 0.2310 on the same 1,000 events.

Diagnosis: the channel is not broken. It is a weak zero-shot channel on a task that is hard for a content-only zero-shot model and easy for collaborative filtering (CF). The PILOT_LOG wording "the task is not hard" overstates this.

Evidence checked against the files:
1. Readout integrity is clean.
   - 0 censored rows out of 1.13M; mean Yes+No mass is at least 0.99999; 0 overlength prompts; panel_join is empty.
   - diag_rated_baselines.py recomputes llm_like UAUC as 0.58745 from the panel's own labels, so labels and candidate order are aligned.
2. The same channel works on relevance: sports next UAUC is 0.719 and NDCG@10 is 0.209, about 91% of C-CRP.
3. It reads question polarity. The dislike logit is consistently anti-informative: UAUC 0.408 on ML-1M and 0.480 on Toys.
4. What it lacks is efficacy on rated panels.
   - ML-1M 0.587 is below plain item popularity at 0.637 (paired difference +0.049 [0.036, 0.062]).
   - Leave-user-out item mean reaches 0.760 on ML-1M and 0.712 on Toys.
   - Spearman correlation of the like-logit with item mean is only 0.15 / 0.07.
   - Caveat: item_mean_loo is transductive. It uses other users' later ratings, on labels with 3 stars dropped. So 0.760 is an optimistic CF ceiling, not "the task is easy".
5. The literature anchor matches. I read CoLLM (TKDE'25, arXiv 2310.19488) Table 2 in the PDF: on ML-1M, labels >3, temporal split, ICL gets UAUC 0.5268, MF 0.6361 and fine-tuned TALLRec 0.6818. So 0.60 was an efficacy bar with no reference point, labelled as a sanity bar. It stays unchanged regardless.

Calibration verified:
- Overconfidence is mostly one global scale and offset: Platt slope 0.062 on ML-1M and 0.022 on Toys.
- After Platt scaling, Brier skill over the base rate is about 2.6% on ML-1M (0.2186 vs 0.2244) and about 0% on Toys (0.2203 vs 0.2202).
- The raw Yes decision makes more errors than always answering Yes: 38.2% vs 34.0% on ML-1M, 40.8% vs 32.7% on Toys.

Not yet established: is the shortfall knowledge-limited or readout/prompt-limited?
- Knowledge-limited means the 8B model lacks item-quality and taste knowledge, so no prompt fix can go far.
- Readout-limited means the model knows but the prompt does not elicit it.
- Hint (exploratory, from burned users): global AUC 0.597 is barely above UAUC 0.587, so the model hardly uses the user's own star history. For comparison, ReLLa's zero-shot Vicuna-7B reaches global AUC 0.674 on ML-1M (labels >=4 vs rest; I verified this in the arXiv HTML). That is a global AUC, not a UAUC anchor.

How to establish it: run the pre-registered diagnosis battery on burned users (T0 item-knowledge probe, T1 positive control, T2 CF-evidence transmission ratio, T3 digit readout, star-permutation contrast, T4 Llama). Its outcomes are interpretive only.

The registration's consequence still binds:
- Every Pilot-1 diagnostic is exploratory.
- The registered remedy is executed exactly once, under Amendment 2, with the bar unchanged.
- The registered decision table has a hole that was actually hit: decision.json has positive = negative = null = false. Close it conservatively: INDETERMINATE means not POSITIVE.

## Gate-fix protocol
PREREG_AMENDMENT_2, section G. Freeze it before any new GPU job, including diagnostics, and record its sha1, the rendered-prompt sha1s and the user-id lists in PILOT_LOG.

G0 Invariants.
- Scoring setup: Qwen3-8B, fp16, chat template, thinking off, prefill-only, Yes/No read by token id, top-50 logprobs, max_model_len 4096 for every variant.
- Never allowed in any variant: CF or popularity statistics, the user's mean rating, few-shot examples, thinking, verbalized readout, digit readout.
- The rated constant is unchanged: ML-1M UAUC >= 0.60 on the point estimate, exactly as pilot1_gate.py computes it.
- The next-item component already PASSED under V0. Its prompt is frozen at V0 and is not re-tested. No sports test event is touched.

G1 Candidate bank (rated panels only). Taken from scratchpad/deliberation/gatefix_prompt_variants_v1.json, with V6 (SUBR) dropped because it needs new embedding code and defeats prefix caching.
- V0: registered control. It must reproduce prompts_sha1 12e83c4fcca40db398ac58acbba2a7eb9c770524 on the ML-1M pilot panel.
- V1: label-aligned question, "Will this user rate the candidate item 4 stars or higher (on a 1-5 scale)?"
- V2: TALLRec layout, the same 10 events split into liked (4-5 stars) and disliked (1-3 stars) lists.
- V3: last 20 history events.
- V4: history lines annotated with genres (ML-1M) or categories[:80] (Amazon).
- V5: system message stating the 1-5 scale, plus the user template without the persona line.
- V7: V1 + V3 + V4 + V5.
- Each variant's dislike and paraphrase strings (the threshold family for V1/V7) are fixed now but are never scored before GATE_PASS.

G2 Users.
- DEV = the burned Pilot-1 panels: ML-1M 1,500 users and Toys 1,500 users. They are already contaminated, so using them costs no fresh data.
- CONFIRM = every fresh eligible ML-1M user:
  - rerun build_rated_panels with seed 0 and n_users >= all eligible users;
  - assert that sha1 of the first 1,500 lines equals 985494c7b44d010bec4b11ec62f35ad274dfe91f;
  - take positions 1,500 onward (expected 1,683 users) and assert user_id disjointness from the pilot panel;
  - if the sha1 differs, select by user_id exclusion instead and log the mismatch.
- Video_Games rated and Sports rated stay untouched. Toys fresh users are used only if Toys has at least 2,000 eligible users.

G3 Stage 1: dev scoring that never looks at MIRROR.
- Score only the like question of V0-V5 and V7 on both dev panels.
- That is about 205k ML-1M and 133k Toys prompts, roughly 2 GPU-h at the measured 87 and 35 prompts/s.

G4 Eligibility.
- E1, per panel: censored=2 rows <= 0.5%, overlength = 0, mean Yes+No mass >= 0.95.
- E2: Toys dev UAUC >= Toys dev UAUC(V0) - 0.010.

G5 Selection.
- V* = the eligible variant with the highest ML-1M dev UAUC(like).
- Variants within 0.005 of the maximum count as tied. Break ties by higher Toys dev UAUC, then by the simplicity order V0 < V1 < V5 < V3 < V4 < V2 < V7.
- A FIX is FOUND only if V* differs from V0 and the one-sided paired user-bootstrap lower bound of UAUC(V*) - UAUC(V0) on ML-1M dev is above 0. The bootstrap uses 2,000 resamples, seed 0, at level 1 - 0.05/6.
- All 7 x 2 dev UAUCs go into an appendix table on prompt sensitivity.

G6 Stage 2: one confirmatory gate.
- Score V* like on the fresh ML-1M confirm users. V0 like is also scored there, as reported context only.
- GATE_PASS if and only if UAUC(V*) >= 0.60 and E1 holds. Report the 95% user-bootstrap CI.

G7 Stage 3, only on GATE_PASS.
- Under V*, score like, dislike, like_para, the swap prior (K=8) and the no-history prior on fresh ML-1M and on the second rated domain: fresh Toys if at least 500 fresh users are eligible, otherwise Video_Games.
- Run MIRROR's next-item no-loss check under V0 on the first 1,000 events of the sports VALID panel, because test events 1-1000 are quarantined.
- Apply the registered section 3.1 table plus P1, with the hole closed: a configuration that matches no branch is INDETERMINATE, which means not POSITIVE and no automatic B9/A9 handoff.
- Any MIRROR claim must also beat a correlation-matched ensemble null, in addition to the registered placebo.

G8 Contingency, written now.
- F0: no fix found. Record GATE_FAIL_AFTER_REMEDY(dev) and skip Stage 2.
- F1: confirm UAUC < 0.60, or E1 fails. Record GATE_FAIL_AFTER_REMEDY(confirm).
- On F0 or F1:
  - no second prompt round, no threshold change, no backbone rescue (Llama-3.1-8B is replication only);
  - zero-shot rated results become the descriptive Finding 1, labelled "below registered bar", re-measured on fresh users with pre-declared descriptive endpoints;
  - Pilot 3 moves to the fine-tuned regime;
  - next-item zero-shot endpoints remain interpretable.
- In ALL branches, Gate-FT is registered now:
  - TALLRec-style LoRA yes/no SFT: r=16, alpha=32, lr 1e-4, 1 epoch, 3 seeds; prompt V*, or V0 under F0;
  - training only on events from ML-1M users outside the confirm set, before the global p80 timestamp (amendment D);
  - evaluation on the fresh confirm users' test-period candidates;
  - PASS if and only if the mean UAUC over seeds is >= 0.65. This is anchored between published MF at 0.636 and fine-tuned TALLRec at 0.682, both on harder labels;
  - report all seeds, and the CPU references (prior-only item mean, MF) as context;
  - Gate-FT FAIL closes the rated-panel method line.

G9 Diagnosis battery.
- T0-T4 and the star-permutation contrast run on burned users after the freeze.
- Their results are interpretive only and cannot change G1-G8.

## Recommended paper framing (robust to gate outcome: True)
Retire MIRROR as the flagship but do not kill it. It stays a registered confidence channel and is evaluated only after a pass.

The recommended paper is a pre-registered audit that compares the zero-shot and fine-tuned regimes. Working title: "What does a yes/no LLM recommender's confidence know? Item priors vs personal evidence, and when uncertainty is actionable".

Object: pointwise token P(Yes) recommenders (the TALLRec/CoLLM family), measured against labels that match the question.

Contributions:
- C1 Protocol.
  - Panels: rated ML-1M, Toys, Video_Games and Sports, using all fresh eligible users (a dataset limit, not a toy choice), plus the full-scale next-item panels (4 domains x 10k users x 101 candidates).
  - Two backbones (Qwen3-8B, Llama-3.1-8B); zero-shot and LoRA with 3 seeds.
  - Non-LLM reference rows in every table: prior-only item mean, popularity, MF.
  - User-cluster CIs, Holm correction within each RQ, and confirmation only on untouched users.
- C2 Decomposition. logit = user offset + item prior pi + personal evidence.
  - Personal share, disattenuated by split-half reliability.
  - Personal information gain over the item-mean prior.
  - A star-permutation contrast: does the model use the user's item-to-rating mapping at all?
  - A two-axis popularity analysis, partialled on quality.
  - The pseudonym knockout, in the fine-tuned regime.
- C3 Actionability.
  - A short lemma: per-user monotone maps cannot change UAUC or within-user NDCG.
  - Each route that can change outcomes is tested against its proper control:
    - item-dependent corrections, against the placebo and a diversity-matched ensemble null;
    - cross-user selective serving and exposure at 10k scale, against random and against the baselines' own confidence;
    - pruning, against matched-random pruning over 5 seeds.
- C4 A one-table guide answering S1-S6:
  - S3 is answered statically on the next-item panels; the dynamic loop is off because Pilot 2 was NULL;
  - S6 is expected to come out about equal to random.

Add one nested method slot with a hard kill date of Nov 30. The default is prior-offset LoRA, killed if its gain over post-hoc stacking is < +0.01 or its CI covers 0 on at least 2 of 4 datasets.

Context and testbed, not headline: C-CRP and the 8 baselines. Cite C-CRP in the third person. Scope every claim to same-candidate evaluation.

Banned claims:
- the two-prompt estimator as new;
- "uncertainty improves ranking";
- 6/8 SOTA;
- any number not confirmed on fresh users.

Why the framing survives the zero-shot gate:
- On PASS, the zero-shot regime is valid, and V0 vs V* is reported as prompt fragility.
- On FAIL, Finding 1 reads: "zero-shot yes/no confidence fails a registered adequacy bar, is item-prior dominated and beaten by CF". The fine-tuned regime carries the main results.

What it does not survive: if Gate-FT also fails, the paper shrinks to a zero-shot audit plus the next-item exposure study, with much weaker acceptance odds.

Acceptance odds (my judgment, not data): roughly 15-25% if the fine-tuned regime and the full-scale next-item audit land; 5% or less for MIRROR as flagship.

## Ranked next actions
### 1. Write and freeze idea-stage/PREREG_AMENDMENT_2.md (section G gate-fix protocol, exhaustive decision table, Gate-FT, forking-path rules) before any GPU job  (≈0 GPU-h)
**Why.** The registered remedy ('fix the prompt first') does not say how to do it. The registered decision table also has a hole, and Pilot 1 landed in it: decision.json shows positive = negative = null = false. Without a frozen protocol, every later step is a forking path.
**Protocol.** 1. Encode G0-G9 from gate_fix_protocol.
2. Close the table hole: a configuration that matches no branch is INDETERMINATE, which means not POSITIVE and no automatic handoff.
3. Add a unit test that enumerates every combination of conditions and asserts that exactly one outcome results.
4. Keep MIRROR as a registered arm. Add the correlation-matched ensemble null as a second required comparator.
5. Declare that CPU references (prior-only item mean, popularity, MF) are reported context and never gates.
6. Move Pilot 3 to the fine-tuned regime.
7. Amend the next-item MIRROR no-loss check to sports VALID events 1-1000.
8. Record the hashes in PILOT_LOG.
9. Get a cross-family review if available; otherwise label it same-family/provisional.
**Decision rule.** No GPU job of any kind, diagnostics included, may start until the amendment's sha1 and the rendered-prompt and user-id-list hashes are in PILOT_LOG.

### 2. Run the CPU artifact checks and forensic pack on the existing Pilot-1 score files (server CPU)  (≈0 GPU-h)
**Why.** Several quotable diagnostics may be artifacts. Fresh-user availability, which the whole confirmation plan rests on, is still unverified.
**Protocol.** Run the artifact_checks_first list. Write new scripts in a scratch area and do not edit src/.
Outputs, all labelled exploratory:
1. Prefix sha1 check and eligible counts for ML-1M, Toys, Video_Games and Sports rated.
2. UAUC(pi), UAUC(L_nohist), split-half reliability of pi, and 4-donor evidence UAUC.
3. UAUC of prior-only leave-user-out item mean, temporal biased MF, and the MF personal residual.
4. MIRROR minus the empirical correlation-matched ensemble null.
5. ECE after intercept-only and after temperature-only recalibration, and user-centred ECE.
6. Redefined error targets.
7. Quality-partialled popularity correlations.
8. Base rate of head share in the sports candidate pool.
9. Reconciled throughput figures.
**Decision rule.** All exploratory; no paper claims.
- pi split-half reliability < 0.7: the evidence collapse is labelled uninterpretable.
- MF personal-residual UAUC <= 0.55: 'no personal signal' is written as a property of consumed-item panels.
- Ensemble-null residual CI covers 0 on ML-1M: MIRROR's ML-1M edge is attributed to two-view diversity in all writing.
- Prefix sha1 mismatch: select confirm users by user_id exclusion and log it.
- Toys eligible < 2,000: no fresh Toys.
- Video_Games or Sports rated eligible < 800: report all eligible users as a dataset limit.

### 3. Run the channel-diagnosis battery on burned users (decides knowledge-limited vs readout-limited)  (≈1 GPU-h)
**Why.** Establishes whether the failure is a broken readout, a readout or prompt limit, or a knowledge limit. That decides what Finding 1 says and how much to expect from the prompt fix.
**Protocol.** Qwen3-8B fp16 on burned ML-1M users, 29k prompts per arm, about 6 minutes each.
- T0: no-user item-quality probe on the 2,909 ML-1M items. Read P(Yes) for 'Is this movie widely considered good?' and E[r] over digit tokens, then compute Spearman against the prior-only item mean.
- T1: positive control. Append '(This user later rated this item r/5.)'.
- T2: CF-evidence injection. Append 'Average rating by other users before this date: m/5 (n ratings)', with a prior-only leave-user-out mean. Also run it on 500 burned Toys users.
- T3: digit-rating readout, E[r] over the tokens 1-5.
- Star permutation: permute the '(rated r/5)' suffixes among the same history items, K=2. Report tau_P = L - mean L_perm.
- T4: Llama-3.1-8B-Instruct, V0 like.
**Decision rule.** All interpretive only; none of these can alter G1-G8.
- T1 UAUC < 0.90: readout broken, stop and debug.
- T2 transmission (UAUC_T2 - 0.5)/(UAUC_itemmean_prior - 0.5): >= 0.8 means knowledge-limited; < 0.5 means readout-limited.
- T0 Spearman >= 0.5 together with like-logit rho of about 0.15: the model knows item quality but the prompt does not use it. T0 < 0.3: knowledge-limited.
- Star permutation |dUAUC| < 0.005 and within-user SD(tau_P) < 0.10: the model ignores the user's ratings. This is a hypothesis to be re-tested on fresh users.
- T3 >= yes/no + 0.03: format bottleneck. Reported as a secondary channel, never as a gate rescue.
- T4 >= 0.62: exploratory only.

### 4. Stage 1: dev selection on burned users that never looks at MIRROR (7 variants, like only)  (≈2 GPU-h)
**Why.** This is the registered remedy, carried out as one finite selection that is blind to MIRROR, placebo and evidence. Dev n=1,500 gives a minimum detectable difference of about 0.01, matching the 0.013 gap to the bar.
**Protocol.** Follow G3-G5.
- Score V0-V5 and V7 'like' on burned ML-1M (1,500 users) and burned Toys (1,500 users).
- Check eligibility E1 and E2.
- V* = argmax of ML-1M dev UAUC, with ties within 0.005 broken by Toys dev UAUC and then by the simplicity order.
- Run the paired bootstrap of V* vs V0 (2,000 resamples, one-sided, alpha 0.05/6).
- Publish all dev numbers.
**Decision rule.** FIX FOUND requires V* != V0 and a bootstrap lower bound > 0. Then go to Stage 2. Otherwise record F0 = GATE_FAIL_AFTER_REMEDY(dev) and skip Stage 2.

### 5. Stage 2: one-shot confirmatory gate on all fresh ML-1M users (then Stage 3 registered arms only on PASS)  (≈1.7 GPU-h)
**Why.** Fresh users are draws from the same seed-0 shuffle, so they are exchangeable with the pilot users, and they are free of the designer's exposure to more than 20 diagnostics. This removes the dev winner's curse with a single test.
**Protocol.** G6: score V* like, plus V0 like as context, on the fresh ML-1M users (about 1,683). Apply the 0.60 point-estimate rule unchanged and report the CI.
G7 on PASS:
- run the registered arm set under V* on fresh ML-1M and on the second rated domain;
- run the MIRROR next-item no-loss check on sports VALID events 1-1000;
- apply the exhaustive table, with the ensemble-null comparator added.
**Decision rule.** GATE_PASS if UAUC(V*) >= 0.60 and E1 holds. Otherwise record F1 = GATE_FAIL_AFTER_REMEDY(confirm), which is final for the zero-shot rated gate. No reruns, no threshold or backbone changes.

### 6. Benchmark LoRA throughput, then run Gate-FT on ML-1M (Qwen3-8B, 3 seeds); this runs in every branch  (≈10 GPU-h)
**Why.** The fine-tuned regime is the main object of the recommended framing and does not depend on the zero-shot prompt. It pre-empts the objection 'you audited a near-chance zero-shot model'.
**Protocol.** - Use src/confrec/train_lora_yesno.py with r=16, alpha=32, lr 1e-4, 1 epoch, bf16 with gradient checkpointing (4-bit if memory requires). Spend the first hour benchmarking throughput.
- Prompt: V*, or V0 under F0.
- Train on events from ML-1M users outside the confirm set, before the global p80 timestamp (amendment D).
- Evaluate on the fresh confirm users' test-period candidates, rescoring zero-shot on the same subset for a paired regime contrast.
- Report UAUC, ECE, Brier, Platt slope and AUROC/AURC for error detection, as mean ± sd over seeds, next to the CPU references.
**Decision rule.** PASS if mean UAUC over seeds >= 0.65, with all seeds reported. Then fund the full fine-tuned grid. FAIL closes the rated-panel method line; the paper becomes the observation study plus the full-scale next-item audit.

### 7. Full-scale next-item audit: CPU exposure audit of the 8 baselines and C-CRP now; token P(Yes) next and like on 4 x 10k x 101 under the frozen V0  (≈13 GPU-h)
**Why.** Gives a full-scale answer to S3 (static exposure), shows the audited channel is not a strawman, and provides the population for cross-user decision tests. It is the only place where 10k x 101 claims can come from.
**Protocol.** CPU:
- top-10 head share, Gini and APLT per method from ranking_eval_records.csv;
- compare against the head share expected from the candidate pool and against target demand.
GPU:
- 4.04M prompts per question at about 175/s;
- list-normalized calibration (softmax over the 101 candidates, temperature fit on 2k VALID events);
- selective serving by max q, margin and entropy vs random, with niche-user coverage at 50% coverage.
**Decision rule.** Scope every claim to same-candidate evaluation. Report sports test events 1-1000 separately from 1001-10000. The C-CRP and baseline comparison is context only, never a SOTA claim. Claim an S3 effect only if the head-share-minus-pool-base-rate CI excludes 0 in at least 3 of 4 domains. No dynamic echo-chamber claim (Pilot 2 was NULL).

### 8. Replication on fresh domains and a second backbone under the final prompt  (≈6 GPU-h)
**Why.** Pilot-1 diagnostics were seen on burned users, so every observation has to replicate before it reaches the paper.
**Protocol.** Run the pre-declared descriptive and confirmatory endpoint set (overconfidence, evidence share, quality-partialled pi-popularity link, star-permutation tau, MIRROR vs placebo and ensemble null) on:
- fresh ML-1M;
- Video_Games rated and Sports rated;
- fresh Toys if eligible.
Run Llama-3.1-8B on fresh ML-1M and one Amazon domain, writing to a separate output root.
**Decision rule.** A finding reaches the abstract only if it has the same sign with a 95% CI excluding 0 on fresh ML-1M and on at least one fresh Amazon domain, and the same sign on Llama. A Llama-only pass is reported as exploratory and never overrides a Qwen failure.

### 9. Full fine-tuned grid plus decomposition and causal knockout (conditional on Gate-FT PASS)  (≈75 GPU-h)
**Why.** Carries C2 and the zero-shot vs fine-tuned contrast: does supervision fix calibration, and does it change what confidence encodes?
**Protocol.** - 4 rated datasets x Qwen x 3 seeds, plus Llama on 2-4 datasets x 3 seeds.
- Swap prior under LoRA, with split-half reliability and a disattenuated personal share.
- Stacker information gain over item mean, fit on dev users and evaluated on confirm users.
- Pseudonym knockout on Toys and Video_Games with the real-brand placebo.
- Two-axis popularity Bias Index.
**Decision rule.** Use the registered knockout thresholds: head-minus-tail delta >= 0.20 and placebo within ±0.05; otherwise drop the word 'familiarity'. Any gain must exceed 2 sigma_seed with 3 of 3 seeds agreeing in sign. If behind schedule by week 5, cut Llama to 2 datasets.

### 10. One nested method slot (default prior-offset LoRA), plus S6 pruning against matched random  (≈75 GPU-h)
**Why.** A small fix derived from the mechanism raises acceptance odds if it survives. S6 is a seed question the user asked explicitly; the expected honest answer is about equal to random.
**Protocol.** Prior-offset LoRA: logit = f_theta(u,i) + b*q_hat(i) with q_hat under stop-gradient, compared with SFT and with post-hoc stacking on 4 datasets x 3 seeds.
S6 arms on one rated dataset, 5 seeds each:
- full data;
- random 25% pruned;
- uncertainty top-25% pruned;
- prior-congruent pruning;
- high-loss pruning.
**Decision rule.** Kill the slot by Nov 30 if its gain over post-hoc stacking is < +0.01 or its CI covers 0 on at least 2 of 4 datasets. For pruning, claim only gains > 2 sigma_seed with a paired Wilcoxon test; otherwise report it as about equal to random.

## Artifact checks first
- Prefix and fresh-user check (CPU): rebuild the ML-1M rated panel with seed 0 and n_users >= all eligible users. Assert that sha1 of the first 1,500 lines equals 985494c7b44d010bec4b11ec62f35ad274dfe91f. Do the same for Toys against 69252b4806bb4ffe05101967ea2a1d94ada57dee. Record eligible_users for Toys, Video_Games and Sports rated (ML-1M is 3,183, so 1,683 fresh). Code reading supports the prefix property: rows are shuffled before slicing, and the per-user candidate shuffles consume the RNG in order. The hash must still confirm it.
- V0 byte-identity (CPU): the new VARIANTS registry must render the ML-1M pilot panel to prompts_sha1 12e83c4fcca40db398ac58acbba2a7eb9c770524 for like, dislike and like_para. max_model_len 4096 does not affect the prompts, and none of the original prompts were overlength.
- Quantities Pilot 1 never reported, from the server score files: UAUC of pi(i) alone; UAUC of L_nohist alone; split-half Spearman reliability of pi (donors 1-4 vs 5-8); evidence UAUC recomputed with a 4-donor pi, plus a Spearman-Brown projection.
- Non-transductive CF references on the same pairs: leave-user-out item mean using only ratings strictly before each candidate's timestamp; temporal biased MF; MF personal residual (MF - b_u - b_i); and the global AUC of the history-star mean, to quantify user leniency. The current item_mean_loo of 0.760/0.712 uses other users' later ratings and is an optimistic ceiling.
- MIRROR ensemble null: UAUC of the empirical within-user z-sum of like + (-dislike), against like + like_para. My binormal recomputation from diag_baselines reproduces the strategist's estimates: ML-1M predicted MIRROR 0.608 vs observed 0.616, placebo predicted 0.597 vs 0.598; Toys predicted 0.536 vs 0.537 and 0.544 vs 0.544. So about two-thirds of the ML-1M +0.019, and all of Toys, is two-view diversity. Run it properly per user before any MIRROR sentence is written.
- Calibration decomposition: ECE after intercept-only recalibration, after temperature-only recalibration, and user-centred ECE. Platt slopes are 0.062 (ML-1M) and 0.022 (Toys). Platt Brier skill is about 2.6% and about 0%. The raw Yes decision errs more often than always-Yes: 38.2% vs 34.0%, and 40.8% vs 32.7%.
- Redefine the error target as errors after Platt scaling and as per-user top-k errors (k = number of likes) before answering S2. Quarantine the tempting diagnostic that |MIRROR| detects errors better (AUROC 0.652/0.651 vs |raw| 0.584/0.571): the current target is close to 'disliked item', so this restates discrimination.
- Quality-partialled popularity links: partial Spearman of pi and of v against log-popularity, controlling for the prior-only item mean. For ML-1M also control for release year, which is already in the titles. For Toys, also control for description length and has_store.
- Sports next-item: the top-10 head share expected from the candidate pool per event (random-ranking expectation), and C-CRP's top-10 head share, before reading the LLM's 0.745.
- Throughput reconciliation: PILOT_LOG says about 191 prompts/s, but report.json gives 87.1 (ML-1M rated), 35.0 (Toys rated) and 174.8 (sports next-item). Budget every rated-panel stage at the measured per-panel rates. The methodologist's 1.5 GPU-h Stage 1 estimate is about 2x optimistic.
- Optional, low value: fp16 vs bf16 on 300 users. The earlier dtype probe already gave rho 0.999 and UAUC is rank-based, so skip it unless GPU time is idle.

## Forking-path rules
- Freeze first. Amendment 2 (prompt bank, user-id lists, metrics, tie rules, contingency, Gate-FT, exhaustive decision table) is hashed into PILOT_LOG before any new GPU job, diagnostics included.
- Immutable constants: zero-shot rated UAUC >= 0.60 (point estimate), next-item NDCG@10 >= 0.18632, and Gate-FT >= 0.65 (registered now, before any fine-tuned data). No post-hoc re-anchoring, CI-based reinterpretation or 'near miss' rule.
- One round only: 7 variants, one selection, one confirmatory gate. A confirm FAIL is final for the zero-shot gate. No second prompt round, no backbone substitution, no reruns.
- Unit roles are fixed. DEV is the burned Pilot-1 users. CONFIRM is the fresh ML-1M users, asserted disjoint by user_id. Video_Games and Sports rated stay untouched until replication. Sports next-item TEST events 1-1000 are quarantined from tuning and are always reported separately from 1001-10000. Next-item prompts stay frozen at V0.
- Selection never looks at MIRROR. Dev scores only the like question. Dislike, paraphrase, swap and no-history arms for a new variant are computed only after GATE_PASS, and only on confirm users.
- Pilot-1 diagnostics generate hypotheses only. MIRROR +0.019/-0.007/-0.018, the evidence collapse, the popularity correlations, |MIRROR| error AUROC, like_para 0.604 and ECE 0.30 can never be cited as evidence or used to choose an arm or prompt. Each one must be re-measured as a pre-declared endpoint on fresh users.
- No arm promotion. like_para or MIRROR may not become the base arm because they cleared 0.60 on burned users.
- The table hole closes conservatively. A Pilot-1-style configuration that matches no branch is INDETERMINATE, which means not POSITIVE and no automatic B9/A9 handoff. A unit test enumerates all condition combinations to guarantee every future rule is exhaustive.
- Any two-prompt claim (MIRROR, VCC, mirror tuning) must beat both the registered paraphrase placebo and a correlation-matched ensemble null.
- Every rated-panel table carries the non-LLM references (prior-only item mean, popularity, MF) as context. They are never gates, and they are reported whichever way they come out.
- Diagnosis-battery results (T0-T4, star permutation) are interpretive only. They cannot change the frozen bank, thresholds, splits or contingency.
- Claim admission. A finding enters the abstract only if confirmed on fresh users, replicated in sign on at least one fresh Amazon domain, and replicated on a second backbone. Until then, use direction-free wording.
- Fine-tuned seeds: 3 for the main grid and 5 for pruning. Report every seed. A gain must exceed 2 sigma_seed with all seeds agreeing in sign. Hyperparameters and kill dates are fixed before training.
- Same-family review is flagged on every verdict. Seek a cross-family review of Amendment 2 before Stage 1 if available.

## Strongest dissent overruled
Strongest dissent I overruled: the gate-fix methodologist's choice of populations. They proposed selecting the prompt on 800 fresh ML-1M users (positions 1500-2299) and re-running the confirmatory gate on the original 1,500 pilot users. Their argument: those users are the registered gate population, and a paired comparison against V0 on the same users has more power.

I overruled it for four reasons:
1. The registration specifies "ML-1M rated pairs", not particular users. Fresh users come from the same seed-0 shuffle and are exchangeable with the pilot users.
2. The pilot users are the ones on which the designers have already seen more than 20 diagnostics. They are the contaminated set, so they belong in dev, not in confirmation.
3. Spending 800 fresh users on dev uses up 48% of the only untouched ML-1M pool. That pool is also needed for the confirmatory observation endpoints.
4. The gate is an absolute threshold, so pairing with V0 adds nothing to the gate decision. Dev on 1,500 burned users also gives more selection power than 800.

I adopted the rest of the methodologist's protocol almost unchanged: literature-grounded variants, selection blind to MIRROR, a deterministic tie rule, a one-shot confirmation, a contingency written in advance, and no change to the threshold. I trimmed SUBR and kept the next-item prompt frozen at V0.

Other dissents I overruled:
- The method inventor's push to make VCC (valence-counterfactual contrast) the lead method candidate. I kept only its star-permutation contrast, as a diagnostic. The estimator form is prior art (ICCD/CAD), and the global-AUC-approximately-equals-UAUC hint suggests tau may be near 0.
- The skeptic's added channel-integrity gate as a formal pass/fail. I kept it as a reported diagnostic, so that a second gate cannot later be read as moving the goalposts.
- Any argument to lower or re-anchor 0.60. It was unanchored, but the statistic has already been seen, so any change would depend on the outcome.
- The area chair's 22-30% acceptance estimate. I tempered it to roughly 15-25%, as judgment rather than data.

Strategist claims that did not survive my verification:
- The skeptic said diag_baselines was not logged in PILOT_LOG. It is logged, under the "Gate-failure diagnosis" entry.
- Stage-1 cost estimates that assumed about 190 prompts/s are too low for rated panels, which measured 87 and 35 prompts/s.

Claims I verified:
- CoLLM Table 2 (ICL UAUC 0.5268, MF 0.6361, TALLRec 0.6818) and ReLLa's zero-shot ML-1M global AUC (0.6739/0.6993/0.7013, with K=15 the peak history length), both read from the papers.
- The binormal ensemble arithmetic, which I recomputed.

Not re-verified by me: the TALLRec in-context AUC of about 0.5 (only its abstract-level claim) and the Kang et al. table values.