# ADVERSARIAL SKEPTIC

**Headline.** The token channel is not broken, but it is weak at this task. On the same ML-1M pairs, zero-shot P(Yes) scores UAUC 0.587. A leave-user-out item mean scores 0.760, plain item popularity scores 0.637, and published zero-shot LLMs score about 0.50-0.53. So the 0.60 gate measured how well the model fits rated-panel prediction, not whether the channel works. Two of the most quotable diagnostics are largely design artifacts. MIRROR's edge on ML-1M is about 60% ensemble diversity measured against a placebo that was not correlation-matched. The "evidence collapse" may simply come from panels built only from items the user already consumed. Every Pilot-1 number is exploratory. Five cheap checks (about 1.5 GPU-h plus CPU work) settle channel-vs-task. Confirmation then has to use the 1,683 untouched ML-1M users, Video_Games, rated Sports and Llama-3.1-8B.

## Analysis
SCOPE AND WHAT I VERIFIED
I read every file listed in the task. I also found D:/Research/Lumen/outputs/confrec_pilot/pilot1/{ml1m,toys}_rated.diag_baselines.json, which is not mentioned in PILOT_LOG.md. It already contains the CF and item-mean comparison that point (c) asks for, and its existence needs to be logged. The per-pair score files live only on the server, so every check that needs them below is a protocol, not a result. Numbers marked "back-of-envelope" are my own approximations from the reported summary statistics.

Pipeline integrity, checked from the files:
- panel_join is empty: there are no item-id mismatches between scores and panel.
- diag_rated_baselines recomputes llm_like UAUC = 0.5874 from the panel's own labels. That matches pilot_mirror exactly, so labels and candidate order are aligned.
- Censoring is 0 on 1.13M rows, and Yes+No mass is about 1.0.
- The sports prompt uses history[-5:]. C-CRP v3 does the same (D:/Research/Lumen/experiments/rsc/run_ccrp_v3_domain.py line 10), so hist_len 5 is not a confound for the next-item gate.
- A join or ordering bug is therefore unlikely.

=== (a) WHICH DIAGNOSTICS COULD BE ARTIFACTS, AND HOW TO CHECK EACH CHEAPLY ===

A1. History rendered as "Title (rated r/5)".
- No UAUC leak by construction:
  - build_rated_panels keeps the first event per item and takes history = events strictly before the first candidate, so a candidate never appears in history.
  - User leniency (the mean of the history stars) is constant within a user, so it is rank-inert for UAUC.
  - Session effects in ML-1M, where sequels are rated back to back, are legitimate collaborative evidence, not leakage.
- Where it does matter is global AUC, ECE and Brier. History stars carry user leniency, so the pooled ECE of 0.30 and Platt slope of 0.062 mix between-user offset with within-user miscalibration.
- Cheap checks, on dev users only:
  - permute the "(rated r/5)" suffixes within each user's history;
  - render titles only.
  - Prediction if the model ignores the stars: |dUAUC| < 0.005.
- Also report a user-centred ECE next to the pooled one (the A10 per-user offset lemma).

A2. Rated panels whose candidates are the user's own last 20 reviewed items. This is the biggest interpretive artifact.
- Every candidate was self-selected, so the label measures satisfaction among items already consumed. It does not measure relevance.
- On these panels the label is item-quality dominated:
  - item_mean_loo UAUC is 0.760 on ML-1M and 0.712 on Toys;
  - item popularity UAUC is 0.637 on ML-1M and 0.535 on Toys.
- The LLM is strong at relevance: sports next-item UAUC is 0.719 and NDCG@10 is 0.209, about 90% of C-CRP. That is exactly the variance consumed-item panels remove.
- So "evidence = like − π collapses to 0.507/0.525" may be a property of the estimand. Personal satisfaction signal is small on consumed items, and the LLM's personal relevance signal has nothing to act on there. It need not mean "LLM confidence carries no personal information".
- Checks:
  - CF decomposition on the same pairs (CPU): UAUC of the item-mean-prior, of a temporal biased MF, and of the MF personal residual (MF − b_u − b_i). If the residual is ≤ 0.55, the collapse is a panel property.
  - A mixed panel of rated plus unconsumed popularity-matched items (recommendation R4).
- The diag item_mean_loo is also transductive: it uses other users' later ratings, and the panel drops 3-star ratings. Its 0.760 is therefore an optimistic ceiling. CoLLM's ML-1M MF baseline gets UAUC 0.636 on a temporal split with 3-star counted as negative. Recompute with ratings strictly before the candidate timestamp; I expect roughly 0.72-0.75.

A3. Popularity uses all-time counts.
- The diagnostics are robust to this:
  - corr(π, log-pop) is 0.428 all-time vs 0.399 prior-only;
  - corr(a, log-pop) is 0.007 vs 0.005;
  - pair-level corr(v, log-pop) is the only one that moves, from 0.30 to 0.21.
- The real confound is quality and era, not the time window. On ML-1M popular items are better liked (popularity UAUC 0.637). The LLM like-logit correlates 0.198 with popularity but only 0.154 with item mean.
- Check (CPU): partial Spearman of π and v with log-pop given item_mean_prior; for ML-1M also given release year parsed from the title; for Toys also given description length and has_store.
- Until that is done, "π tracks popularity" cannot be separated from "π tracks quality, which correlates with popularity".

A4. fp16.
- Not credible as an artifact:
  - bf16 vs fp16 gave Pearson 0.9991 and Spearman 0.9989 on 389 prompts;
  - there were no non-finite values;
  - UAUC is rank-based.
- Expected |dUAUC| ≤ 0.003. Settle it with 300 users in bf16 (about 6k prompts, roughly 1 minute).

A5. Sports panel uses hist_len 5.
- Not an artifact for the gate, because C-CRP v3 also uses the last 5 items.
- It is a confound when comparing panels: rated panels use 10, next-item uses 5.
- MIRROR's next-item loss (dNDCG −0.031 [−0.042, −0.020] vs raw) is better explained by a design flaw. "Would this user dislike X?" is ill-posed for the 100 never-consumed, popularity-sampled negatives: global_mean_a = −10.5, so both questions answer a strong No.
- That makes −dislike a non-relevance signal, and the loss is predictable from the design. It is not evidence about acquiescence.

A6. The 1000-event subset.
- It is representative: C-CRP NDCG@10 is 0.2310 on the subset vs 0.2329 on all 10k.
- The SE of NDCG@10 at n = 1000 is about 0.010, so the 0.023 gate margin is about 2 SE.
- The problem is not sampling. These are the first 1,000 rows of the frozen TEST file (head -n 1000 in run_pilot1_mirror.sh), and they are now burned (see b).

A7. The placebo is not correlation-matched. This artifact matters most for the primary endpoint.
- like_para correlates 0.953 with like (ML-1M) and 0.915 (Toys). −dislike correlates only 0.355 and 0.389.
- MIRROR is algebraically a two-view ensemble, like + (−dislike). Any less-correlated second view gains more from ensembling, whatever acquiescence does.
- Back-of-envelope binormal ensemble null: d = √2·Φ⁻¹(AUC), equal-weight standardized sum, pooled correlations from diag_baselines.
  - Toys: predicted MIRROR 0.536 vs observed 0.537; predicted placebo 0.544 vs observed 0.544. Ensembling explains both exactly.
  - ML-1M: predicted MIRROR 0.6085 vs observed 0.616; predicted placebo 0.597 vs observed 0.598.
  - So about +0.012 of the +0.0185 MIRROR−placebo gap on ML-1M is predicted by diversity alone, leaving a residual of about +0.007.
- Caveat: pooled Spearman rather than within-user, within-class correlation, and unequal logit scales. Redo it properly on CPU.
- Conclusion: the primary endpoint as registered cannot separate acquiescence removal from ensembling.

A8. "Massively overconfident" (ECE 0.30, Platt slope 0.062).
- Directionally robust, but the numbers mix three things:
  - a global yes-bias, which is rank-inert;
  - pooled between-user leniency;
  - weak discrimination.
- With slope 0.062 and intercept 0.507, Platt-calibrated probabilities span only about 0.55-0.70.
- Platt-calibrated Brier skill over the base rate is about 2.6% on ML-1M (0.2186 vs 0.2245) and about 0% on Toys (0.2203 vs 0.2202).
- The thresholded raw decision errs 38.2% (ML-1M) and 40.8% (Toys). The trivial always-Yes rule errs 34.0% and 32.7%.
- Check: ECE after intercept-only recalibration and after temperature-only recalibration, and user-centred ECE. This says how much of the 0.30 is a single global offset.

A9. "|logit| detects errors weakly (AUROC 0.58)".
- The error target is the raw σ ≥ 0.5 decision. The model says Yes almost always, so "error" is close to "disliked item".
- The AUROC therefore restates discrimination; it is not a separate error-detection property.
- Re-define errors after Platt recalibration, or with a per-user top-k decision (k = number of likes), before answering Q2.

A10. Evidence (like − π) collapse and the K = 8 donor noise in π.
- Probably not mainly noise. If the between-item SD of the like logit is about 3 and donor SD about 2.5, the 8-donor reliability is about 0.9.
- Check: split-half reliability of π (donors 1-4 vs 5-8), plus UAUC(π) and UAUC(L_nohist) alone. These two numbers are missing from the JSON.
- pmi_nohist (0.547) sits above evidence (0.507). That suggests about 0.04 UAUC comes from a generic "any rated history" shift rather than from the specific user. Also exploratory.

A11. A hole in the pre-registered decision rule.
- decision.json reports positive = negative = null = false. Even if the gate had passed, the outcome on these diagnostics would have had no branch:
  - corr(a, pop) < 0.2, so not POSITIVE or NULL;
  - the ML-1M CI is > 0, so not NEGATIVE.
- The handoff to B9/A9 is reachable only through NEGATIVE.
- Amendment 2 must make the rule exhaustive before any re-run.

=== (b) FORKING PATHS: WHAT MAY BE USED, WHAT MAY NOT, WHAT MUST BE RE-VALIDATED ===

Burned units:
- ML-1M: 1,500 users of the 3,183 eligible.
- Toys: 1,500 users. The eligible count is unknown locally; read toys_rated.meta.json on the server.
- Sports: next-item test events 1-1,000.
- Seen: about 6 arms × 3 panels × more than 20 estimands, plus the diag baselines.

May be used:
1. The mechanical outcome GATE_FAIL and its registered remedy (fix the prompt).
2. Infrastructure facts: throughput, mass, censoring, the fp16 grid.
3. Non-LLM panel properties as task-difficulty references, because they do not depend on LLM outcomes: like-rate, item-mean and popularity UAUC, eligibility counts.
4. Diagnostics as hypothesis generators for planning. For example: "A1 looks dead: corr(a, pop) = 0.007 [−0.008, 0.022] with a tight CI, and next-item MIRROR loses 0.031." Planning on this is legitimate as long as nothing is reported as confirmatory.
5. Burned users as the prompt-development set. They are already contaminated, so dev use costs nothing.

May NOT be used:
1. Promoting like_para (0.604, which passes 0.60) or MIRROR (0.616) to the primary arm because it passed.
2. Choosing a prompt by its UAUC on burned users and reporting that UAUC.
3. Lowering or re-anchoring the 0.60 threshold after the fact to declare a pass. An added, separate channel-integrity gate is acceptable; a moved threshold is not.
4. Claiming "MIRROR helps on ML-1M". The sign is heterogeneous: +0.019 ML-1M, −0.007 Toys, −0.018 sports.
5. Claiming "LLM confidence is an item prior / no personalization" (evidence 0.507/0.525). This fits the pre-written spine ("knows the item, not the user"), which makes the confirmation-bias risk acute.
6. Any tuning on sports test events 1-1,000. They belong to the frozen 10k test panel used against the 8 official baselines.

Must be re-validated on untouched units, with hypotheses fixed in writing first:
- the gate on fresh ML-1M (1,683 users never scored);
- overconfidence;
- item-prior dominance;
- the π-popularity link, partialled on quality;
- acquiescence not popularity-linked;
- MIRROR vs a correlation-matched ensemble null.

Fresh units available:
- 1,683 ML-1M users;
- Video_Games rated and Sports rated (both fully fresh domains);
- Toys only if the eligible count is above 1,500 + 500;
- Llama-3.1-8B on all of them (an independent replication axis);
- the valid panels for all next-item domains;
- sports test events 1,001-10,000, and the toys/home/tools test panels.

Selection inflation:
- With about 4 prompt candidates and a paired UAUC SE of about 0.002-0.003, choosing the best on dev inflates by roughly 0.003-0.006. That is harmless only if the confirm number comes from fresh users.
- The absolute UAUC SE with about 1,500-1,700 users is about 0.0045 (95% CI ±0.009). A prompt whose true UAUC is exactly 0.60 passes a point-estimate rule only about 50% of the time. State this in Amendment 2 rather than discovering it later.

=== (c) BROKEN CHANNEL OR HARD TASK? ===

The evidence already on disk points to "hard for a zero-shot content model", not "broken readout":
- Same pairs, ML-1M: LLM 0.587, popularity 0.637, item mean 0.760.
  - The LLM recovers (0.587 − 0.5)/(0.760 − 0.5) = 33% of the item-mean signal on ML-1M and 19% on Toys.
  - It is significantly below popularity on ML-1M (paired +0.049 [0.036, 0.062]).
- Literature anchors:
  - TALLRec (RecSys'23): in-context LLMs, including GPT-3.5-class models, sit at AUC ≈ 0.50 on ML-100K like/dislike.
  - CoLLM (TKDE'25), Table II on ML-1M: ICL UAUC 0.527 (Vicuna-7B); fine-tuned TALLRec 0.682; CoLLM up to 0.699.
  - These use 3-star as negative, so they are not directly comparable, but zero-shot 7-8B models are clearly expected around 0.50-0.60. The 0.60 threshold was not literature-anchored. It sat at the top of the plausible zero-shot range, so it tested task fit, not channel integrity.
- Channel integrity is supported independently:
  - next-item UAUC 0.72 and NDCG ≥ 0.8 × C-CRP;
  - dislike is coherently anti-informative (UAUC 0.408, i.e. 0.592 inverted, on ML-1M);
  - like_para correlates 0.95 with like;
  - mass ≈ 1, and the join is validated.

Decisive cheap test battery. Run it on burned users only, with rules written down first. None of these arms can become the reported method.
- T1 positive control: append "(This user later rated this item r/5.)".
  - PASS if UAUC ≥ 0.95.
  - Below 0.90, the readout does not condition on the candidate block: the channel is broken.
- T2 evidence transmission: append "Average rating by other users before this date: m/5 (n ratings)", using the prior-only, leave-user-out item mean.
  - Transmission T = (UAUC_T2 − 0.5)/(UAUC_cf − 0.5), with UAUC_cf the CF baseline on the same pairs.
  - T ≥ 0.8: the readout transmits evidence and the gate failure is a knowledge deficit. No wording fix will reach far beyond 0.60.
  - T < 0.5: readout-limited (yes-bias or format dominates). Fix the readout.
- T3 readout swap:
  - digit rating "1-5", scored as E[r] over the digit tokens;
  - C-CRP-style verbalized probability.
  - If either exceeds the yes/no arm by ≥ 0.03 UAUC, the yes/no format is the bottleneck.
- T4 backbone: Llama-3.1-8B-Instruct, same prompt. ≥ 0.62 means the failure is Qwen-specific.
- T5 relevance vs satisfaction: the mixed panel of R4.
  - If AUC(consumed vs unconsumed) ≥ 0.70 while like-vs-dislike stays around 0.59, the channel carries relevance, not satisfaction.
  - In that case rated-panel UAUC is the wrong integrity metric, and "evidence collapse" is a panel property.
- T6: fp16 vs bf16 on 300 users.

Total about 1.3 GPU-h. The ML-1M rated run measured about 87 prompts/s, not 190; 190 is the next-item rate. Toys measured about 35/s because of its long descriptions.

=== (d) THE SIX SEED QUESTIONS: WHAT CAN BE ANSWERED HONESTLY NOW ===

None can be answered confirmatorily, because the gate failed. Status of each:

Q4 (yes/no confidence vs ground truth): closest to answerable, as a pilot observation for this prompt only.
- Global AUC 0.597 / 0.559.
- Raw ECE 0.30 / 0.36.
- Platt-calibrated Brier skill about 2.6% / about 0%.
- The raw decision is worse than always-Yes (38.2% vs 34.0% error).
- The direction (overconfident, weakly discriminative) is very unlikely to flip. The magnitudes are prompt-dependent.
- Allowed sentence: "In an exploratory pilot that failed its pre-registered channel gate, zero-shot Qwen3-8B P(Yes) was weakly discriminative (UAUC 0.587, below item popularity at 0.637) and strongly overconfident."

Q1 (correct but is it sure?): not answerable.
- Raw logits are near-saturated, so almost every prediction, right or wrong, looks "sure".
- That is a property of the scale, not an answer. It needs the recalibrated error target (A9) and fresh users.

Q2 (wrong because low confidence?): weak and unconfirmed.
- |logit| AUROC is 0.58 / 0.57: errors are only slightly less confident, and most errors are confident.
- But the target is confounded with the label (A9).
- The prior Lumen verbalized result (AUROC 0.60-0.76) used mismatched 1/101 labels, so it does not answer Q2 either.

Q5 (popular high, niche low): partially supported in level terms only, and exploratory.
- corr(π, log-pop) is 0.43 (ML-1M) and 0.28 (Toys).
- Acquiescence is not popularity-linked (0.007 / 0.074).
- Whether high confidence on popular items is miscalibrated or justified is unknown. Popularity itself predicts liking on ML-1M (0.637).
- It needs a quality-partialled correlation and a confidence-matched calibration gap by popularity decile.

Q3 (echo chamber): no evidence.
- Pilot 1 measured no exposure or loop outcome.
- Top-10 head share of 0.745 cannot be read without the candidate-pool base rate. The negatives are popularity-sampled, so the base rate is probably high.
- The only prior result is static, verbalized and next-item: head over-exposure of +1-5pp.

Q6 (prune by uncertainty): not tested.
- The prior Lumen negative stands: never better than random 25% pruning, with σ_seed ≈ 1.5pt.
- Pilot 1's weak error detection says nothing about label noise, because error ≠ noise.

=== STRATEGIC WARNING ===

Both the registered flagship and the registered fallback look weak on exploratory diagnostics:
- A1: corr(a, pop) is about 0 with a tight CI, and MIRROR has a built-in next-item loss.
- B9/A9: evidence UAUC is about 0.51, and |evidence| error AUROC is 0.506.

The central reviewer objection to any zero-shot rated-panel uncertainty study will be "you are calibrating a predictor that is worse than item popularity". Pre-commit now to the fine-tuned (TALLRec-style) regime as the object of the rated-panel confidence study, with zero-shot as secondary, if the frozen prompt still fails on fresh users. Make that commitment before seeing any new data.

## Recommendations
### Write PREREG_AMENDMENT_2 before any new GPU run: exploration/confirmation split, exhaustive decision table, separate integrity gate  (≈0 GPU-h)
- Rationale: Pilot-1 numbers are burned. The registered rule has a hole: decision.json shows positive = negative = null = false, so even a passing gate would have had no branch. The placebo was not correlation-matched. Without a written split, every next step is a forking path, and the spine's own hypothesis ('knows the item, not the user') is what the burned diagnostics already appear to show.
- Protocol: Create D:/Research/Lumen/idea-stage/PREREG_AMENDMENT_2.md dated before any scoring.
(1) Declare all Pilot-1 outputs exploratory. Burned units: the users in outputs/confrec/panels/{ml1m,toys}_rated.jsonl and sports test rows 1-1000 of ranking_test.jsonl.
(2) DEV = burned ML-1M 1,500 + burned Toys 1,500 + the first 1,000 events of the Lumen sports VALID panel.
(3) CONFIRM sets:
- ML-1M eligible users NOT in the burned panel: build with n_users=10000 and filter by user_id against the burned file; expected 1,683.
- Video_Games rated and Sports rated: fresh, ≤1,500 users each.
- Toys fresh users only if toys_rated.meta.json eligible_users ≥ 2,000.
- Llama-3.1-8B-Instruct on all confirm sets.
(4) Fix the prompt candidate set now: P0 current; P1 label-matched 'Will this user rate the candidate item 4 or 5 stars out of 5?'; P2 = P1 + history partitioned into high (4-5★) / low (1-2★) / neutral (3★) lists. Selection rule: highest mean DEV UAUC over ML-1M + Toys; if within 0.005, take the simpler prompt.
(5) Keep the registered usefulness gate unchanged: fresh ML-1M UAUC ≥ 0.60 (point estimate). Add a separate channel-integrity gate: positive-control UAUC ≥ 0.95 and next-item UAUC ≥ 0.65.
(6) Confirmatory hypotheses with Holm correction:
- H2 raw ECE ≥ 0.15 in ≥ 3/3 fresh domains;
- H3 evidence keeps ≤ 25% of like's above-chance UAUC in ≥ 2/3 fresh domains;
- H4 item-level corr(π, log-pop) ≥ 0.20 with a quality-partialled corr ≥ 0.10;
- H5 |corr(a, log-pop)| < 0.10;
- H6 MIRROR − correlation-matched ensemble null: CI includes 0.
(7) Make the decision table exhaustive (add INDETERMINATE → treat as NEGATIVE for the A1 flagship).
(8) Record that sports test events 1-1000 were seen; final next-item results also report events 1,001-10,000 separately.
- Risk: Low. The main risk is reviewers reading the added integrity gate as goalpost-moving. Mitigation: the 0.60 usefulness gate stays untouched and is re-tested only on fresh users.
- Paper value: Makes every later number defensible as confirmatory. Gives a citable, honest pilot-transparency paragraph instead of an uninterpretable one. Blocks the most likely reviewer attack (post-hoc arm and prompt selection).

### CPU forensic pack on the existing Pilot-1 score files (server)  (≈0 GPU-h)
- Rationale: Several headline diagnostics may be artifacts that can be resolved without a GPU:
- MIRROR's edge may be ensemble diversity. Back-of-envelope binormal null predicts MIRROR 0.6085 vs observed 0.616 and placebo 0.597 vs 0.598 on ML-1M, and fits Toys exactly.
- The evidence collapse may be π noise or a panel property.
- ECE 0.30 may be mostly a global offset.
- The π-popularity link may be quality.
- The |logit| error AUROC restates discrimination.
- The top-10 head share of 0.745 has no base rate.
- Protocol: On lumen-gpu, read outputs/confrec/pilot1_mirror/{ml1m_rated,toys_rated,sports_next_1k}/scores.csv.gz, swap_prior.csv.gz and *_nohist. Write a new script under the scratch area; do not edit src/.
(a) UAUC of π(i) alone and of L_nohist alone; split-half Spearman reliability of π (donors 1-4 vs 5-8); evidence UAUC recomputed with 4-donor π, and a Spearman-Brown extrapolation.
(b) CF decomposition on the same pairs:
- item_mean_prior (other users' ratings strictly before the candidate timestamp);
- biased MF (Koren-style) trained on ratings before the 80th-percentile timestamp, excluding panel candidates;
- MF personal residual (MF − b_u − b_i);
- per-user UAUC for each.
(c) Ensemble null: within-user z-score like, −dislike and like_para; UAUC of the equal-weight z-sums; per-user binormal prediction from within-user within-class correlations; report MIRROR − null with a user bootstrap.
(d) ECE after intercept-only recalibration, after temperature-only recalibration, and user-centred ECE (50% user split).
(e) Partial Spearman of π and v with log-pop given item_mean_prior; plus release year (ML-1M, parsed from the title) or description length and has_store (Toys).
(f) Error target re-defined as Platt-at-0.5 errors and per-user top-k errors (k = number of likes); AUROC of |calibrated logit|; compare with the always-Yes error rate (34.0% / 32.7%).
(g) Sports: candidate-pool head share per event (the random-ranking expectation of top-10 head share), and C-CRP top-10 head share on the same events.
Decision notes, all exploratory:
- π split-half reliability < 0.7 → the evidence collapse is uninterpretable.
- MF residual UAUC ≤ 0.55 → 'no personal signal' is a panel property.
- Ensemble-null residual CI includes 0 on ML-1M → drop the acquiescence interpretation of MIRROR.
- Risk: Low. All results are on burned users and stay exploratory. The binormal null is approximate, so use the empirical z-sum as primary.
- Paper value: Removes or confirms the three most quotable but fragile diagnostics before any GPU is spent. The ensemble-null result is itself a methodological contribution: two-prompt debiasing must be tested against diversity-matched ensembles.

### Channel-integrity battery on burned users: positive control, evidence injection, readout swap, backbone swap, dtype  (≈1 GPU-h)
- Rationale: This decides 'broken channel' vs 'hard task'. Today's evidence favours a hard task:
- LLM 0.587 vs popularity 0.637 vs item mean 0.760 on the same pairs;
- literature zero-shot ≈ 0.50-0.53 (TALLRec, CoLLM ICL);
- next-item AUC 0.72.
But nobody has tested whether the yes/no readout transmits evidence it is given, and that is what decides whether 'fix the prompt' can work at all.
- Protocol: Qwen3-8B fp16, burned ML-1M 1,500 users (29,365 pairs; about 5.6 min per arm at about 87 prompts/s). Add Toys 500 users for T2.
- T1: P0 plus the candidate line '(This user later rated this item r/5.)'. PASS if UAUC ≥ 0.95; below 0.90, the readout is broken.
- T2: P0 plus 'Average rating by other users before this date: m/5 (n ratings)', with m = prior-only leave-user-out mean. T = (UAUC_T2 − 0.5)/(UAUC_item_mean_prior − 0.5). T ≥ 0.8 → knowledge-limited: wording fixes cannot rescue zero-shot, so pre-commit to the fine-tuned regime. T < 0.5 → readout-limited: change the readout.
- T3: digit readout 'Predict this user's rating of the candidate (1-5). Answer with one digit.', score = E[r] over digit-token probabilities; and a verbalized JSON probability (C-CRP style, T = 0.1, ≤ 40 tokens, about 15 min). Format bottleneck if either beats the yes/no arm by ≥ 0.03.
- T4: Llama-3.1-8B-Instruct, P0 like. Qwen-specific if ≥ 0.62.
- T6: bf16 P0 like on 300 users. Expect |dUAUC| ≤ 0.005.
Write all rules into Amendment 2 first. None of these arms can be the reported method.
- Risk: T2 injects CF information, so it diagnoses the channel but is not a method; a reviewer might call it trivially expected. That is fine, because its job is the transmission ratio. T3's verbalized arm costs the most (generation).
- Paper value: Turns the gate failure into a clean, reportable finding: 'the yes/no channel transmits evidence (T ≈ x) but zero-shot 8B models lack item-quality knowledge'. Or it identifies a fixable readout bottleneck. Either way it justifies the regime the paper uses.

### Relevance-vs-satisfaction mixed panel (tests whether consumed-item panels manufacture the 'evidence collapse')  (≈0.4 GPU-h)
- Rationale: Rated panels contain only items the user chose. That removes the relevance variance the LLM is good at (next-item AUC 0.72) and leaves item-quality variance it lacks (item mean 0.76 vs LLM 0.587). The 'personal term ≈ 0' diagnostic (evidence 0.507/0.525) may therefore describe the estimand, not LLM confidence in general. If unchecked, a paper built on it would overclaim.
- Protocol: Burned ML-1M, 300 users. For each, take the 20 rated candidates plus 20 unconsumed movies the user never rated, popularity-matched (same log-pop decile distribution as the user's candidates).
Score P0 like and P0 next, plus a K = 8 swap prior for the new items (about 12k pair prompts + about 48k swap prompts ≈ 12 min).
Report per-user AUC for: consumed vs unconsumed, liked vs disliked, liked vs unconsumed, disliked vs unconsumed. Report the same for evidence = like − π.
Pre-registered reading: if AUC(consumed vs unconsumed) ≥ 0.70 and evidence-AUC(consumed vs unconsumed) ≥ 0.60 while liked-vs-disliked stays ≤ 0.60, then the personal term carries relevance but not satisfaction, and every rated-panel 'collapse' claim is scoped to satisfaction.
Then replicate on 300 fresh Video_Games users.
- Risk: Medium. Popularity matching on ML-1M is coarse, and 'unconsumed' is not 'disliked' (MNAR). Report it strictly as relevance.
- Paper value: Gives Q1/Q2/Q5 a correct scope: what LLM-rec confidence encodes depends on how candidates are generated. This is a clean, cheap, novel observation that also pre-empts the strongest reviewer attack on any 'item-prior dominance' claim.

### A-priori label-matched prompt fix on DEV, one frozen prompt confirmed once on fresh users, domains and backbone  (≈3.5 GPU-h)
- Rationale: This is the registered remedy ('fix the prompt first'), executed without forking. The current question 'Would this user like...' does not match the label (rating ≥ 4 vs ≤ 2). A label-matched question is justified before seeing any data. like_para already crossed 0.60 on burned users (0.604), which is exactly why selection must not happen there.
- Protocol: DEV:
- P0, P1 ('Will this user rate the candidate item 4 or 5 stars out of 5?') and P2 (P1 + high/low/neutral partitioned history) on burned ML-1M 1,500 (about 6 min each) and burned Toys 1,500 (about 9 min each).
- Sports VALID first 1,000 events with the next question (about 10 min).
- Pick by the Amendment-2 rule.
CONFIRM (once, frozen prompt):
- Fresh ML-1M 1,683 users: like-form, dislike-form, swap K = 8 (about 25 min).
- Video_Games rated and Sports rated, ≤ 1,500 users each: like/dislike plus swap K = 4. Amazon swap is n_items × K prompts: Toys K = 8 took 131k prompts, about 62 min.
- Llama-3.1-8B on fresh ML-1M.
Evaluate the gate (fresh ML-1M UAUC ≥ 0.60) and H2-H6 with Holm.
If the gate fails again and T2 showed knowledge-limited, close the zero-shot rated line and trigger the fine-tuned fallback.
- Risk: Medium-high. The true UAUC may sit near 0.60, giving about a 50% pass chance on fresh users with SE about 0.0045. Amazon swap priors dominate the cost. Any post-confirm tweak re-burns the fresh users, and Toys may have no fresh users left.
- Paper value: The only route to confirmatory statements on Q4/Q5 and on the item-prior hypothesis. Covers 3 fresh domains × 2 backbones, which matches the paper's 'consistent across datasets and backbones' needs.

### Quarantine burned next-item TEST events; do all next-item prompt work on VALID panels  (≈0.5 GPU-h)
- Rationale: Sports test events 1-1,000 are part of the frozen 10k panel used against C-CRP and the 8 official baselines. Any prompt or arm choice informed by them (MIRROR −0.031, placebo +0.010) is test-set tuning on the paper's main table.
- Protocol: Use the Lumen sports VALID ranking file (rebuild_panels.sh already emits --splits valid,test) for all next-item development:
- first 1,000 valid events × 101 candidates × {next, like} at about 175 prompts/s ≈ 20 min per prompt;
- if C-CRP valid ranks do not exist, gate relative to raw P0 on valid.
Freeze the next-item question before touching test. Score the full 10k test once. Report all 10k and, separately, events 1,001-10,000 as a contamination-free line.
Do not apply MIRROR to next-item unless the dislike question is redefined in Amendment 2 before scoring: it is ill-posed for never-consumed negatives (global_mean_a −10.5).
- Risk: Low. The valid and test distributions may differ slightly.
- Paper value: Keeps the main 8-baseline comparison clean and states the contamination honestly, so a reviewer cannot dismiss the next-item table.

### Pre-commit the fallback regime now: fine-tuned yes/no LLM (TALLRec-style LoRA) as the object of the rated-panel confidence study if zero-shot fails on fresh users  (≈4.5 GPU-h)
- Rationale: Zero-shot P(Yes) is below item popularity on ML-1M (0.587 vs 0.637). Calibrated Brier skill is about 2.6% (ML-1M) and about 0% (Toys). An uncertainty paper about a near-chance predictor invites 'you are calibrating noise'. Fine-tuned yes/no models reach UAUC of about 0.68-0.70 on ML-1M (CoLLM Table II). Committing before seeing fresh data removes the forking path of choosing the regime by results.
- Protocol: In Amendment 2, declare the trigger: fresh-ML-1M frozen-prompt UAUC < 0.60, or T2 transmission ≥ 0.8 with zero-shot UAUC < 0.62.
Action:
- src/confrec/train_lora_yesno.py with Qwen3-8B 4-bit, r = 16, α = 32, 1 epoch, about 20k ML-1M training pairs;
- training candidates earlier than the global 80th-percentile timestamp; training users disjoint from every confirm user;
- 3 seeds (about 1.2 GPU-h each including scoring).
Gate: fine-tuned UAUC on fresh users ≥ 0.65, and ≥ item_mean_prior UAUC − 0.05.
Then run the H2-H5 observation battery on the fine-tuned model, with zero-shot as the secondary regime. Repeat for one Amazon domain (Video_Games) only after ML-1M passes.
- Risk: Medium. Seed σ is about 1.5pt in the sibling project, and LoRA may re-learn popularity priors (the UCalRec finding). That is itself informative for Q5 but complicates attribution.
- Paper value: Makes the observation study about a recommender worth calibrating. It also enables the zero-shot vs fine-tuned contrast (overconfidence amplified or reduced by fine-tuning), which the literature notes list as an open gap for yes/no LLM recommenders.

## Citations
- Bao, Zhang, Zhang, Wang, Feng, He. TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. RecSys 2023. arXiv 2305.00447. [verified this session: in-context LLMs score AUC around 0.5 on the ML-100K like/dislike task]
- Zhang, Feng, Zhang, Bao, Wang, He. CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation. IEEE TKDE 37(5), 2025, doi:10.1109/TKDE.2025.3540912. arXiv 2310.19488. [verified this session: ML-1M Table II UAUC: ICL 0.5268, MF 0.6361, TALLRec 0.6818, CoLLM-SASRec 0.6990; rating >3 counted as positive, temporal split]
- Kang, Ni, Mehta, Sathiamoorthy, Hong, Chi, Cheng. Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction. arXiv 2305.06474, 2023. [verified this session, abstract only: zero-shot LLMs lag behind CF; fine-tuning closes the gap]
- Hou et al. Large Language Models are Zero-Shot Rankers for Recommender Systems. ECIR 2024. arXiv 2305.08845. [verified in repo NOVELTY_DOSSIER; not re-checked this session]
- Burns, Ye, Klein, Steinhardt. Discovering Latent Knowledge in Language Models Without Supervision (CCS). ICLR 2023. arXiv 2212.03827. [verified in repo dossier]
- The yes-no bias of LLMs reflects answer order and wording (crossed symmetrization). arXiv 2607.05552, 2026. [verified in repo dossier; authors not re-checked this session]
- Zhou et al. Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering. ICLR 2024. arXiv 2309.17249. [verified in repo dossier]
- Zhao, Wallace, Feng, Klein, Singh. Calibrate Before Use: Improving Few-Shot Performance of Language Models. ICML 2021. arXiv 2102.09690. [verified in repo dossier]
- Kweon, Kang, Yu. Obtaining Calibrated Probabilities with Personalized Ranking Models. AAAI 2022. doi:10.1609/aaai.v36i4.20326. [verified in repo dossier]
- Kweon, Jang, Kang, Yu. Uncertainty Quantification and Decomposition for LLM-based Recommendation (UQRec). WWW 2025. arXiv 2501.17630. [verified in repo LIT_RECENT_UNC_LLM4REC.md]
- Gelman, Loken. The garden of forking paths: Why multiple comparisons can be a problem, even when there is no 'fishing expedition' or 'p-hacking' and the research hypothesis was posited ahead of time. Unpublished manuscript, Columbia University, 2013. [verified this session: https://sites.stat.columbia.edu/gelman/research/unpublished/p_hacking.pdf]
- Guo, Pleiss, Sun, Weinberger. On Calibration of Modern Neural Networks. ICML 2017. arXiv 1706.04599. [from memory, not verified this session]
- Koren, Bell, Volinsky. Matrix Factorization Techniques for Recommender Systems. IEEE Computer 2009 (biased MF baseline for the CF decomposition). [from memory, not verified this session]
- Simmons, Nelson, Simonsohn. False-Positive Psychology: Undisclosed Flexibility in Data Collection and Analysis Allows Presenting Anything as Significant. Psychological Science 2011. [from memory, not verified this session]
- Local evidence: D:/Research/Lumen/outputs/confrec_pilot/pilot1/{decision.json, ml1m_rated.pilot_mirror.json, toys_rated.pilot_mirror.json, sports_next_1k.pilot_mirror.json, ml1m_rated.diag_baselines.json, toys_rated.diag_baselines.json, *.report.json}; D:/Research/Lumen/src/confrec/{prompting.py, build_rated_panels.py, pilot_mirror.py, diag_rated_baselines.py}; D:/Research/Lumen/scripts/sigir/run_pilot1_mirror.sh; D:/Research/Lumen/experiments/rsc/run_ccrp_v3_domain.py (C-CRP uses history[-5:])