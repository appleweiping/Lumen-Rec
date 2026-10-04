# Pre-registration amendment 3 (2026-10-04): the fine-tuned program

**Status: binding once its sha1 is recorded in `docs/sigir/PILOT_LOG.md` (section 0).**
- Written before any fine-tuned adapter exists and before any outcome statistic of one exists.
- Sources: the ARIS deliberation of 2026-10-02 (`idea-stage/deliberation_2026-10-02/judge.md`, ranked actions 8–10),
  `PREREG_AMENDMENT_2.md` sections F and P, and two independent same-family reviews of the first draft of this file
  (a pre-registration audit and a SIGIR-reviewer/feasibility review, 2026-10-04), whose findings are folded in.
- review_independence: same-family (Claude), provisional. No cross-family review was available.
- Everything not stated here keeps its Amendment 1 and 2 meaning. No immutable constant of Amendment 2 F is touched.

## 0. Freeze and conditionality

**Freeze (one rule for every GPU job of this amendment).** No GPU job of this amendment starts until
`docs/sigir/PILOT_LOG.md` records the sha1 of this file. That includes the training of the Gate-FT adapters, whose only
outcome is a training loss. **Scoring of any adapter or of any zero-shot panel of this amendment, the Gate-FT scoring stages
and section 4 included, additionally waits for the full record**, written by `python -m src.confrec.ftgrid_freeze --print`:
- the sha1 of every file of the FREEZE `core` list (section 11), all of which must exist and be unit-tested;
- the `ftgrid_split.json` of each of the four domains (sha1, T, counts, power facts, sha1 of every user-id list).
Two later lists are recorded before their runs: FREEZE `prune` before any S6 run (section 6) and FREEZE `method` before
any slot run (section 7). `run_gateft.sh` and `run_ftgrid.sh` check this with `ftgrid_freeze --check --stage ...`.

**Conditionality.**
- Sections 1–3 and 5–9 run **only after a recorded GATE_FT_PASS** (Amendment 2 G9). On GATE_FT_FAIL or INCOMPLETE they do
  not run: the paper is the zero-shot audit plus the next-item audit. An INCOMPLETE Gate-FT is rerun as G9 says.
- Section 4 (zero-shot panels) runs **in every Gate-FT branch**, and only after the gate-fix stages of Amendment 2 have
  recorded their decisions, so no selection can be affected by it (G8 pre-declares exactly such descriptive endpoints).

**Not registered** (reported, if at all, as labelled exploratory): any dynamic or multi-round loop; any fine-tuning on the
next-item panels; any second prompt round, backbone swap or threshold change. **No adapter is ever scored on a next-item
panel.** Hence Amendment 1 D's exclusion of the Lumen valid/test users from LoRA training, and its user-disjoint robustness
split, are superseded for this program: the robustness split is the all-rows UAUC of section 3 (EVAL users are disjoint
from TRAIN users by construction). Any future use of an adapter on a next-item panel needs a dated addendum applying A1 D.

## 1. Data roles (`src/confrec/ftgrid_data.py`; every rule below is asserted by the builder)

**Panels.** Rated panels rebuilt with `build_rated_panels.py` (seed 0, all eligible users, 20 candidates, h20 rendering as in
G2; the 1,500-line prefix hashes of ML-1M and Toys are the Amendment-2 check). "Position" always means **row order of the
rebuilt panel** (the seed-0 shuffle), never the sorted user-id lists of `build_confirm_panels.py`. Eligible users (counted
on CPU before this amendment): ML-1M 3,183; Toys 14,202; Video_Games 4,109; Sports 16,712.

| domain | TRAIN users (rows) | EVAL users (rows) |
|---|---|---|
| ML-1M | 0–1,499 = the Pilot-1 users (DEV) | 1,500–3,182 = CONFIRM (1,683) |
| Toys | 0–1,499 = the Pilot-1 users | 1,500–4,499 (3,000; the first 3,000 CONFIRM rows) |
| Video_Games | 0–1,499 | 1,500–4,108 (2,609; untouched until now) |
| Sports | 0–1,499 | 1,500–4,499 (3,000; untouched until now) |

Rule: TRAIN = the first 1,500 rows; EVAL = the next min(3,000, eligible − 1,500) rows. TRAIN and EVAL are disjoint by user_id
(asserted); no Pilot-1 user is ever an EVAL user.

**Time split.** T_d = the 0.8-quantile (numpy linear) of all candidate timestamps of TRAIN_d ∪ EVAL_d; timestamps only, no
labels. It is computed once per domain and written to `ftgrid_split.json`; the ML-1M value must equal Gate-FT's T.
- TRAIN examples = TRAIN users' candidates with ts < T_d, capped at 24,000 (a seed-0 uniform subset; ML-1M is the G9 set).
- CAL rows = EVAL users' candidates with ts < T_d: never trained on; they fit calibrators only.
- TEST rows = EVAL users' candidates with ts ≥ T_d: the primary evaluation set of every endpoint.
- A training example's history precedes all of that user's candidates, hence precedes T_d.

**Facts measured on CPU before this amendment** (rehearsal and probe panels; the registered builder's `ftgrid_split.json`
is authoritative, and any difference is explained in PILOT_LOG):

| domain | TRAIN examples before T | EVAL users | TEST candidates | TEST users with both classes | share of TEST pairs whose item has a TRAIN example |
|---|---|---|---|---|---|
| ML-1M | 23,348 | 1,683 | 6,466 | 366 | 97% |
| Toys | 15,010 | 3,000 | 7,463 | 943 | 11% |
| Video_Games | 15,353 | 2,609 | 6,645 | 820 | 21% |
| Sports | 15,695 | 3,000 | 7,654 | 997 | 14% |

**Minimum n.** `ftgrid_split.json` also records, from labels only, the number of EVAL users with both classes among their
TEST candidates, overall and in the popularity tail (the tail is `stats.rank_bins(pop, 5) == 0` with the bins cut over **all
EVAL candidates, CAL ∪ TEST**, as `pilot_pseudonym` cuts them; every tail endpoint uses the same bins). **An endpoint computed on
fewer than 150 users is descriptive**: no CI-based claim and no Holm family (the G9 warning level). The split file also records
the build arguments (seed, sizes, cap, whether a tokenizer measured the length rule, whether the DEV and Gate-FT checks ran);
a split that was built without a tokenizer, or an ML-1M split not checked against Gate-FT's T, cannot be frozen.

**Scoring scope.**
- Every model scores the **whole EVAL panel** (CAL ∪ TEST) once with the `like` question.
- The decomposition arms of section 3 (swap prior, no-history, star permutation) are scored on **S_d** = the first 1,000 (all,
  if fewer) EVAL users in row order with both classes among their TEST candidates, **restricted to their TEST rows** (a
  TEST-only panel; the pointwise prompts do not depend on the other candidates). Donors of the swap prior come from S_d.
  S_d is recorded in `ftgrid_split.json`.

**TRAIN length rule.** The TRAIN prompts (the examples of `train.jsonl`, after the cap) under the selected variant are tokenised
on CPU; the share above 1,024 tokens is written to `ftgrid_split.json`. If more than 2% of a domain's examples exceed 1,024,
that domain trains with max_len = the 99.5th-percentile length rounded up to 256 (micro-batch × accumulation stays 32; the loss
scale is tested); otherwise (2% or less) G9's 1,024 with skip-and-count applies. The "seen" share of TEST pairs counts items
occurring in `train.jsonl`. Measured under V3 and V7 on 300 Toys DEV users: 0.55% and 7.4% above 1,024 (V0, V1, V2, V4, V5:
0%; ML-1M: 0% for all).

## 2. Regimes, models and training

- **Prompt.** The variant of `selection.json` (`gate_ft_prompt`) for every domain, regime and backbone. No other variant.
- **Zero-shot (ZS).** Qwen3-8B fp16 and Llama-3.1-8B-Instruct fp16 under the G0 invariants and the scorer of Amendment 1 C0.
- **Fine-tuned (FT).** The Gate-FT recipe unchanged (G9 operationalization): LoRA r = 16, alpha = 32, dropout 0.05 on q/k/v/o,
  lr 1e-4, cosine with 3% warmup, 1 epoch, bf16, effective batch 32 (micro-batch 8 × accumulation 4, or 4 × 8 when memory
  requires), last-token loss with the stock `sum / num_items_in_batch` accumulation rule, `standard` mode. Seeds exactly 0, 1, 2.
  The ML-1M adapters of Gate-FT are this program's ML-1M adapters (never retrained).
- **Backbones.** Qwen3-8B on all four domains; Llama-3.1-8B-Instruct on ML-1M and Toys (section 8). Llama writes to a
  separate output root.
- **Integrity.** Every scoring run meets E1 (censored = 2 share ≤ 0.5%, no overlength prompt, mean Yes+No mass ≥ 0.95). A run
  that fails is rerun once; a second failure is recorded as FAILED_INTEGRITY and reported as missing. It never enters a table
  and is never replaced by another seed.

## 3. Endpoint dictionary

TEST rows unless stated. Per (domain, backbone, regime) the summary is the mean over the 3 seeds, every seed listed. CIs resample
**users** (Amendment 1 C1), 2,000 resamples, seed 0. A p-value is p = 2 · min(P*(Δ ≤ 0), P*(Δ ≥ 0)) over those resamples with
+1/(2,001) smoothing. `ftgrid_report.py` implements exactly this text, with a unit test per endpoint; where a function of
`forensics.py` or `metrics.py` is named it supplies the estimator and **the fitting rows are those stated here**.

**Conventions.**
- UAUC = mean over users with both classes among their TEST rows of the per-user AUC (ties 1/2), the G9 statistic.
- **Seed-averaged UAUC** = the mean over users of each user's AUC averaged over the seeds. It is never the UAUC of averaged
  logits (that is an ensemble).
- σ_seed = the SD (ddof 1) of the per-seed value of the contrast in question. With 3 seeds, "mean > 2 σ_seed" is about t > 3.46
  on 2 degrees of freedom.
- q̂(u,i,t) = the prior-only item mean rating from all users other than u strictly before the candidate's timestamp t, shrunk
  with k = 5 (`forensics.prior_means`). It uses EVAL users' ratings before T_d as well; no rating at or after T_d enters any
  TRAIN-side feature.

**E-A Ranking.**
- UAUC on TEST per model and seed. References on identical users and rows, always reported and never gates: q̂, all-time
  popularity, and the temporal biased MF with item parameters fit on ratings with ts < T_d and user fold-in on the user's
  pre-candidate events (`forensics.cf_references` with an explicit cutoff T_d, restricted to TEST rows); its **personal
  residual** (MF − b_u − b_i) is the CF personal-signal reference.
- Secondary: (i) all-rows UAUC on CAL ∪ TEST rows of the EVAL users (user-disjoint from TRAIN); (ii) UAUC split by whether the
  candidate's item has a TRAIN example (seen) or not (unseen). Each carries its own n and the minimum-n rule.

**E-B Regime contrast.** ΔUAUC_s = UAUC(FT seed s) − UAUC(ZS) on identical users and rows; summary = the mean over seeds with
the user-bootstrap CI of that mean; every ΔUAUC_s listed; σ_seed = SD of the ΔUAUC_s.

**E-C Calibration and error anatomy.**
- Calibration (S4): a global Platt map (two parameters) of the logit is fit on **all CAL rows of the model's EVAL users** and
  applied to TEST. ECE = `metrics.ece(p, y, 10, adaptive=True)`, Brier and Brier skill over the CAL base rate, the Platt slope,
  and the 10-bin reliability table.
- Error anatomy (S1, S2): the primary target is the per-user top-k decision: for a user with both classes among TEST rows,
  k = the user's number of TEST likes, c_u = the midpoint of the k-th and (k+1)-th largest TEST logits, decision = 1[L ≥ c_u],
  correct = (decision == label), margin = |L − c_u|. With margin tertiles taken over all TEST pairs of the model, report
  P(wrong | tertile) for the three tertiles, the share of all errors in the top tertile (confident errors, S2), the share of
  all correct decisions in the bottom tertile (correct but unsure, S1), the AUROC of the margin for correctness and the AURC
  (`metrics.auroc`, `metrics.risk_coverage`). The p ≥ 0.5 decision target and fixed 0.6/0.8 confidence masses of the first
  draft are dropped: they are functions of the base rate and of the AUC.

**E-D Decomposition** (S_d, TEST rows). L = the `like` logit; L_nohist = the no-history logit; π(i) = the 8-donor swap prior of
Amendment 1 P1.6 (donors in S_d); ê = L − π. UAUC of π alone, L_nohist alone, ê and L.
- **Reliability and shares.** r_c = the pooled within-user Pearson correlation of the user-centred donor-half means of π
  (donors 1–4 vs 5–8) on the same pairs; r₈c = 2 r_c / (1 + r_c). ρ = the pooled within-user Pearson correlation of centred L and
  centred π. **Item-prior share** = ρ² / r₈c; **non-prior share** = 1 − item-prior share, an upper bound on the personal share
  (it contains donor noise and the user offset's interaction). r_c and r₈c are re-estimated in every user resample; values are
  reported unclipped. If r₈c < 0.7 the shares are reported as "uninterpretable" (the Amendment 2 A2 wording flag).
- **Information gain.** Stackers are logistic regressions of the label on features, with cross-fitting over S_d: the two folds
  are the halves from `stats.user_halves(S_d, 0)`, each stacker is fit on the **other fold's TEST rows**, and UAUC is computed
  on the pooled cross-fitted linear predictors. M0 = [q̂]; M1 = [q̂, π]; M2 = [q̂, π, ê]; M3 = [q̂, MF personal residual].
  G = ΔUAUC(M2 − M1) is the personal information gain of the LLM; G_CF = ΔUAUC(M3 − M0) is the CF reference gain. G/G_CF is
  reported only when G_CF's CI excludes 0.
- **Star permutation** (history lines carry ratings in every domain; K = 2 permutations of the rating suffixes among the same
  history items): τ_P = L − mean L_perm; ΔUAUC = UAUC(L) − UAUC(L_perm) and the within-user SD of τ_P (the A2 thresholds 0.005
  and 0.10 are wording flags, never gates).

**E-E Popularity** (the two-axis analysis). Partial Spearman of L and of π with log popularity controlling for the shrunk q̂
(`forensics.popularity_partial`, item level and pair level; ML-1M also controls release year, Toys and Sports also description
length and has_store). Bias Index head vs tail with confidence = the CAL-fit Platt probability, group = `rank_bins(pop, 5)`
bins 4 vs 0 (bins cut over all EVAL candidates, section 1), 10 confidence bins, residual-adjusted (`metrics.bias_index_ci`).
Popularity is the all-time category count (Amendment 1 C2). Tail-only UAUC is exploratory.

**P1 — the one primary hypothesis of the program (directional; ML-1M, Qwen3-8B).** Supervised fine-tuning increases the personal
information gain: for the 3 seeds, G_FT,s − G_ZS > 0. **P1 holds iff** the mean over seeds is > 0 with p < 0.05, all 3 seeds are
positive, and the mean exceeds 2 σ_seed. Otherwise the paper says "no evidence that supervision adds personal evidence". It is
a single confirmatory test at α = 0.05. Replication for the abstract (Amendment 2 F): the same sign on Llama ML-1M and on at
least one Amazon domain. Everything else in this amendment is secondary.

## 4. Zero-shot panels (Z)

- **Z1 rated.** The ZS regime is scored on the whole EVAL panel of every domain (Qwen: ML-1M CONFIRM, Toys, Video_Games, Sports;
  Llama: ML-1M and Toys), with the decomposition arms on S_d. The ML-1M ZS pass under the selected variant is stage 2's own
  scoring when its run key is identical (same prompt, panel and model); otherwise it is scored once. E-A (without FT), E-C, E-D and
  E-E are reported. These are the "pre-declared descriptive endpoints" of Amendment 2 G8; none of them is a gate.
- **Z2 next-item second backbone.** Llama-3.1-8B-Instruct, frozen V0, questions `next` and `like`, `hist_len` 5, fp16, on TEST
  events 1,001–3,000 of each of sports, toys, home and tools (2,000 events each; sports events 1–1000 stay quarantined). The Qwen
  audit restricted to the same events is the paired comparison for the static-exposure endpoints of
  `docs/sigir/NEXTITEM_AUDIT_SPEC.md` (sections B, D, E). It exists so that an S3 finding can meet Amendment 2 F.

## 5. Pseudonym knockout (FT-K)

- **Models.** Qwen3-8B ZS and FT seeds 0–2 on Toys and Video_Games; Llama-3.1-8B-Instruct ZS and FT seeds 0–2 on Toys. Sports
  (Qwen) is run unless cut at a checkpoint (section 10). The adapters are not retrained on pseudonyms: a test-time intervention.
- **Construction.** `pseudonymize.py` unchanged (Amendment 1 P3). Applied to the whole EVAL panel of the domain; arms real
  (the section 2 scoring), pseudo and placebo, the `like` question.
- **Rows.** The EVAL candidates that are finite in every arm of every model of that domain and backbone; the treated set is
  the panel's store-in-title set.
- **Analysis and reading.** `pilot_pseudonym.py` unchanged, and the label of each model is `pilot_pseudonym.decide()` **verbatim**
  without `--gate`: NULL iff ΔUAUC(real − pseudo) > 0.02 overall and in the tail; POSITIVE iff the head-minus-tail Δlogit(pseudo)
  ≥ 0.20 with its CI lower bound > 0, |head-minus-tail Δlogit(placebo)| ≤ 0.05 and |ΔUAUC_tail(real − pseudo)| ≤ 0.01; NEGATIVE
  iff the pseudo slope CI contains 0 and |hmt(pseudo) − hmt(placebo)| ≤ 0.05; else AMBIGUOUS, and UNDETERMINED when an input
  is missing. AMBIGUOUS is recorded as INDETERMINATE (not POSITIVE) and UNDETERMINED as INCOMPLETE (Amendment 2 closed-hole
  rule). Labels carry no p-values and no Holm family.
- **Wording.** "Familiarity" may appear in the paper only if `decide()` returns POSITIVE for Qwen on Toys and on Video_Games in
  the ZS regime and in all 3 FT seeds, and Llama on Toys has the same sign of the head-minus-tail Δlogit; otherwise the wording
  is "popularity-linked confidence". A seen/unseen-item split of the knockout (CPU) is secondary.

## 6. S6: pruning by uncertainty (FT-P)

- **Domain.** ML-1M only (DEV = TRAIN, CONFIRM = EVAL).
- **Signals** (computed on the TRAIN examples before training; q̂ as in section 3):
  - U: zero-shot uncertainty. With base rate β = the TRAIN label mean, τ = the (1 − β)-quantile of the ZS `like` logit over the
    TRAIN examples (the boundary at which the predicted-Yes share equals the label rate); u = −|L_ZS − τ|. The raw Platt
    uncertainty of the first draft is dropped: with a Platt slope near 0.06 it is monotone in the logit and would prune the
    model's dislikes, not its doubts.
  - C: prior congruence = (2y − 1) · (q̂ − mean q̂).
- **Arms.** In every pruned arm exactly 25% of the TRAIN examples are removed **within each label class** (25% of the positives
  and 25% of the negatives; ties by a seed-0 random key), so class balance is preserved:

| id | removes | seeds |
|---|---|---|
| P0 | nothing (seeds 0–2 are the Gate-FT adapters) | 0, 1, 2, 3, 4 |
| P1 | a uniformly random 25% per class (the draw depends on the seed) | 0, 1, 2, 3, 4 |
| P2 | the 25% per class with the largest u (the ZS model's most uncertain) | 0, 1, 2, 3, 4 |
| P3 | the 25% per class with the largest C (the most prior-congruent) | 0, 1, 2, 3, 4 |

- **Confirmatory test** (the only one; α = 0.05, no Holm): P2 − P1, seed-averaged UAUC contrast with its user-bootstrap p-value.
  P2 **beats** random iff the CI lower bound > 0, the mean exceeds 2 × SD of the 5 paired seed differences, and all 5 differences are
  positive; **is worse than** random under the mirrored condition; **is about equal to** random iff the 95% CI of the contrast lies
  within ±0.01; otherwise the result is "inconclusive". P3 − P1, P1 − P0 and P2 − P0 are descriptive.
- **Wording.** A "beats random" result is reported as "pruning the zero-shot model's uncertain examples helped", scoped to this
  signal, this domain and this backbone. It is never written as "uncertainty improves ranking" (Amendment 2 P, banned claim).
  S6 is single-backbone: it reaches the abstract only with a Llama replication; otherwise it stays in the body, labelled.
- **Order.** P1 and P2 first (P0 seeds 3 and 4 alongside); P3 is run unless cut at the 2026-10-22 checkpoint (section 10).
  FREEZE `prune` (the signal files, the pruned-subset index files and their sha1) is recorded before the first S6 run.

## 7. Nested method slot: prior-offset LoRA (FT-M)

- **Model.** During training, the Yes-token logit of the last position is shifted before the full-vocabulary softmax:
  ℓ'_Yes = ℓ_Yes + b · z(q̂), with z(q̂) the TRAIN-standardised prior-only item mean of the example, b a learnable scalar (init 0,
  saved in `train_config.json`), and no gradient into q̂. The loss is the Gate-FT cross-entropy of the answer token under ℓ'. The
  full-vocabulary loss keeps the Yes+No mass constrained, so E1 is meaningful. At test time the ranking score is the scorer's
  logit(Yes) − logit(No) + b · z(q̂).
- **Comparators** (same seeds 0–2, same TRAIN set, same recipe): (i) SFT = b fixed at 0, which is exactly the §2 FT adapters (on
  ML-1M the Gate-FT adapters); (ii) **post-hoc stacking** = the SFT logit stacked with z(q̂) by a logistic regression fit on CAL.
  Only the three prior-offset runs per dataset are new compute.
- **Endpoint.** ΔUAUC(prior-offset − post-hoc stacking) on TEST, per seed and seed-averaged, user-bootstrap CI and p.
- **Success and kill.** A dataset **passes** if the seed-averaged ΔUAUC ≥ +0.01, its CI excludes 0 and the σ_seed rule holds. Order:
  ML-1M, Toys, Video_Games, Sports. The slot is **killed** as soon as it has failed on 2 datasets; it survives only with passes on
  at least 3 of the 4. Holm family: the datasets run. Hard kill date **2026-11-30**. The slot is single-backbone (it reaches the
  abstract only with a Llama replication) and a killed slot is reported once, in the appendix, as a negative result.
- **Order.** After the main program of section 10; it may start earlier only on ML-1M and only if the GPU would otherwise idle.
  FREEZE `method` (the trainer, the standardisation constants, the stacking script) is recorded before its first run.

## 8. Llama-3.1-8B-Instruct fine-tuned replication (FT-L)

LoRA (seeds 0–2) on ML-1M and Toys with the data, prompt, recipe and endpoints of sections 1–3, and the Toys knockout of section 5.
A Llama result never overrides a Qwen failure; a Llama-only result is exploratory (Amendment 2 F).

## 9. Control: within-item label permutation (FT-C)

- **Purpose.** What does fine-tuning learn beyond item priors? An adapter trained on labels permuted **within each item** keeps
  every item's label marginal (the item prior) and destroys the user–item personal evidence.
- **Construction.** On the ML-1M TRAIN examples, for every item with at least 2 TRAIN examples the labels of its examples are
  randomly permuted among those examples (seed 0; each candidate's star rating moves with its label, so label = [rating ≥ 4]
  still holds); items with one example keep their label. `ftgrid_data.py` writes the file and asserts that every item's label
  sum (and rating multiset) is preserved and that nothing else changes.
- **Runs.** Seeds 0 and 1, the §2 recipe. **Endpoint:** UAUC(real FT, same seed) − UAUC(permuted) on TEST and on all rows, with
  the user-bootstrap CI, and the E-A references. Descriptive; not in any Holm family.

## 10. Order, budget and cut rules (GPU, one job at a time through `scripts/sigir/gpu_queue.sh`)

**Order.**
1. Section 4 (Z1, Z2) as preemptible filler once the full record of section 0 exists.
2. After a recorded GATE_FT_PASS: Toys FT and its endpoints → Llama ML-1M FT → Toys knockout (Qwen) → Video_Games FT and its
   knockout → Sports FT → Llama Toys FT and its knockout → FT-C → S6 P1/P2 (P0 seeds 3, 4) → P3 → Sports knockout → the slot.

**Planning budget (GPU-h; replaced by measurements).** Training at the smoke-test rate of 3.84 examples/s (synthetic 550-token
prompts): about 1.7 h per ML-1M seed, about 1.1 h per Amazon seed. Whole-EVAL `like` passes: ML-1M 6 min, Amazon 16–18 min per
model. Decomposition arms are per item, not per pair: about 25 min per ML-1M model and about 22 min per Amazon model (TEST rows
of S_d). Totals: Amazon Qwen FT grid ≈ 13; Llama FT ≈ 10; decomposition ≈ 9; knockouts ≈ 7 (+ 2 for Sports); FT-C ≈ 4; S6 ≈ 18
(+ 7 for P3); Z2 ≈ 6; the slot ≈ 10 for ML-1M and Toys, ≈ 18 for all four. The program is ≈ 67 without P3, the Sports knockout
and the slot, ≈ 76 with P3 and the Sports knockout, ≈ 86–94 with the slot. The backlog before it (audit, gate-fix, Gate-FT) is
about 40; the planning pessimum (3.5 h per seed, 25% vLLM-LoRA overhead, 10% reruns) is about 2× these figures and still fits.

**Checkpoints and cut rules** (decided on GPU-hours remaining alone, recorded in PILOT_LOG with the date, never after a result they
would affect; a cut item is reported as "not run"):
- **2026-10-22:** decide whether P3, the Sports knockout and Llama Toys FT are cut.
- **2026-10-29:** decide whether Sports FT and the slot's later datasets are cut.
- Anything unfinished on 2026-11-30 is reported as not run. Items not cut are run to completion and reported whatever they show.

## 11. Statistics, admission, verification and code

- **Multiplicity.** P1 and the S6 confirmatory test are single tests at α = 0.05. Holm families (each over its listed members):
  E-B {ΔUAUC per Qwen domain}; E-D {G and the star-permutation ΔUAUC, per domain and regime}; the slot {datasets run}. Descriptive
  and exploratory items belong to no family. "Confirmed" = Holm p < 0.05 and, for fine-tuned results, the σ_seed rule.
- **Seeds.** FT seeds 0–2 (pruning 0–4); every seed reported; a fine-tuned gain counts only if it exceeds 2 σ_seed with all seeds
  agreeing in sign (Amendment 2 F).
- **Claim admission** is Amendment 2 F unchanged, with these consequences stated now: S6 and the method slot are single-backbone
  and reach the abstract only with a Llama replication; rated-panel findings need Z1's Llama arms (which run in every Gate-FT
  branch); S3 findings need Z2.
- **Deviations from the judge's ranked actions** (judge.md 9–10), stated so they cannot be read as silent: the high-loss pruning
  arm is dropped; stackers are cross-fitted within S_d's TEST rows instead of fit on DEV and evaluated on CONFIRM; the swap and
  no-history arms of the selected variant are scored under section 4 on EVAL users after the gate-fix decisions are recorded,
  without a GATE_PASS (G8 allows descriptive endpoints on CONFIRM users in the F0/F1 branches). Amendment 2 P also lists a
  comparison of selective serving with the baselines' own confidence: it is **not run**, because the baselines' score files are
  not available (only their ranks and top-10 lists), and is reported as not run.
- **Verification.** The records in PILOT_LOG are committed and pushed to GitHub (`sigir2027`) at the time of recording, so the push
  time is a public timestamp. An anonymous third-party timestamp of this file's sha1 (OpenTimestamps) may be added by the
  authors. Every verdict is same-family and provisional.
- **Code bound by this amendment.** The sha1 of every listed file is recorded at the freeze; a change to one after the freeze needs
  a dated addendum that says what changed and that no outcome had been seen.

<!-- FREEZE_FILES core -->
idea-stage/PREREG_AMENDMENT_3.md
src/confrec/ftgrid_data.py
src/confrec/ftgrid_report.py
src/confrec/ftgrid_freeze.py
src/confrec/lora_trainer.py
src/confrec/train_lora_yesno.py
src/confrec/pyes_scorer.py
src/confrec/prompting.py
src/confrec/build_rated_panels.py
src/confrec/gateft_data.py
src/confrec/gateft_eval.py
src/confrec/forensics.py
src/confrec/pilot_pseudonym.py
src/confrec/pseudonymize.py
src/confrec/metrics.py
src/confrec/stats.py
scripts/sigir/run_gateft.sh
scripts/sigir/run_ftgrid.sh
scripts/sigir/run_llama_nextitem.sh
<!-- /FREEZE_FILES -->

<!-- FREEZE_FILES prune -->
src/confrec/ftprune.py
scripts/sigir/run_ftprune.sh
<!-- /FREEZE_FILES -->

<!-- FREEZE_FILES method -->
src/confrec/train_lora_offset.py
scripts/sigir/run_ftmethod.sh
<!-- /FREEZE_FILES -->
