# Pre-registration amendment 3, addendum 6 (2026-10-04): additions made after two same-family reviews of the draft

**Status: an extension of Amendment 3** (recorded with it; `ftgrid_freeze` treats every `PREREG_AMENDMENT_3_ADDENDUM_*.md` as part of the
amendment). Unlike addenda 1-5 it adds analyses and corrects two descriptions, so what had been read when it was written is stated here.

**Read before writing:** Gate-FT; the ML-1M fine-tuned grid report (E-A to E-E; P1 holds); the gate-fix selection and confirmation; the
Sports next-item audit analysis; two same-family reviewer reports on the draft (novelty and positioning; methodology), which motivated
the items below; and one read-only recomputation on ML-1M (scratch code, not a registered script) of E-D's G with user-centred stacker
features, which gave G(zero-shot) = +0.0001, G(LoRA, seed mean) = +0.0092 and G_CF = +0.0378 against the registered -0.0018,
+0.0096 and +0.0362. **Not read:** any outcome of the Toys, Video_Games or Sports fine-tuned runs (their scoring may be running or
finished on the server; the files are not opened), of FT-C, Llama, S6, Z2 or the method slot.

**Consequence.** Every statistic below is **exploratory on ML-1M** (its report was read) and **registered, outcome-free** on every other
panel (FT-C: on both datasets, since no permuted adapter has been scored). The paper labels ML-1M values of these items as exploratory.
Hypotheses H-F, H-S and H-J are confirmatory on the Qwen Amazon panels only, each in its own Holm family at alpha = 0.05 (section 11 of
Amendment 3 unchanged: "confirmed" = Holm p < 0.05 and, for fine-tuned results, the sigma_seed rule).

## 1. Two corrections (no analysis changes)

1. **The non-prior share is not an upper bound on the personal share.** Section 3 (E-D) and the draft described 1 - rho^2 / r8c as "an
   upper bound on the personal share". That is withdrawn. The statistic is the share of the within-user variance of L that the item
   prior pi does not explain linearly, Var(e_perp)/Var(L) with e = L - pi orthogonalised to pi; it is at most Var(e)/Var(L) and strictly
   smaller whenever Cov(pi, e) is not zero, so it can be below the share of e, and e itself contains donor-population mismatch (donors have not consumed the item),
   user-by-prior interaction and label-irrelevant history noise as well as personal evidence. The paper reports it as "the non-prior
   share of the within-user variance of the confidence", reports the e-share of item 4 beside it, and uses the information gain G
   (a predictive quantity) and not a variance share as the measure of personal information. e is called the pair-specific residual.
2. **Direction words follow registered tests only.** The comment rule of `sections/experiments.tex` (FILL RULE 2: "write a direction
   only if [the unadjusted interval] excludes 0") is replaced by: a direction word is written only where a registered test or a Holm
   family of Amendment 3 section 11 or of this addendum decides it; every other endpoint is described by its estimate and interval
   with no direction word. (`scripts/sigir/fill_paper.py` already implements this stricter rule.)

## 2. E-W: a within-user stacker estimator (sensitivity for E-D, the estimator of the new endpoints)

The registered stackers of E-D are pooled logistic regressions on uncentred features; user offsets that track the labels can tilt their
weights in a way that changes the within-user ranking (a reviewer's simulation: bias up to -0.018 UAUC with no personal signal). E-W
repeats E-D with **user-centred features** (each feature minus the user's mean over the user's S_d TEST rows, which leaves the within-user
ranking of every linear predictor unchanged and fits the weights on within-user variation only) and with **repeated cross-fitting**:
K = 20 user-half splits (`stats.user_halves(S_d, k)`, k = 0..19; k = 0 is E-D's split); per user the UAUC difference is averaged over the
20 splits, then the user bootstrap (2,000 resamples, seed 0) resamples users with the stacker coefficients of each split held fixed.
Reported per (domain, backbone, regime): UAUC of M0-M3, **G_wu = dUAUC(M2 - M1)**, **G_CF,wu = dUAUC(M3 - M0)**, per seed and seed mean, and
**P1_wu** = the mean over seeds of G_wu,FT,s - G_wu,ZS with its p, sigma_seed and the P1 conditions evaluated on the within-user values.
E-D's G, P1 and its Holm family stay exactly as registered. **Reading rule:** a statement about personal information gain in the paper
is called *robust* only if the within-user estimate has the same sign and an interval excluding 0 (for a fine-tuned mean: also all seeds
the same sign); if the two estimators disagree the paper says the result is estimator-dependent.

## 3. E-F: does the LLM add to collaborative filtering?

Rows and features as E-D, stackers as E-W (user-centred, K = 20 splits). M3 = [q-hat, MF personal residual] and M2 = [q-hat, pi, e-hat] are
E-D's. Add **M4 = [q-hat, MF personal residual, pi, e-hat]** and **G_{LLM|CF} = dUAUC(M4 - M3)** and **G_{CF|LLM} = dUAUC(M4 - M2)**, per
(domain, backbone, regime), FT value = seed mean with the CI of the mean, every seed listed, sigma_seed = SD of the per-seed values.
**H-F (directional).** In the FT regime G_{LLM|CF} > 0. Holm family E-F = {G_{LLM|CF}, FT, per Qwen Amazon panel run}.
ML-1M, Llama and ZS values are descriptive.

## 4. E-G: item share of the references, and the e-share

For a score s on the S_d TEST pairs the *item share* is the squared pooled within-user Pearson correlation between the user-centred s and the
user-centred item component of s, divided by the reliability of that component. Reported for (i) the LLM logit L with its 8-donor swap
prior pi (E-D's item-prior share, unchanged); (ii) the temporal MF reference of E-A with its item bias b_i (`forensics.cf_references`,
`mf_item_bias`, MF warm pairs; deterministic, reliability 1); (iii) the label y with q-hat (a lower bound: q-hat is a noisy item mean;
reliability 1 by convention). Beside E-D's non-prior share the **e-share** of L is reported:
e-share = [Var(L_c - pi_c) - (1 - r8c) Var(pi_c)] / Var(L_c) with pooled within-user variances of the user-centred values and r8c re-estimated
in every user resample (the donor-mean error is independent of the consumer's e). Descriptive; user-bootstrap intervals only.

## 5. E-H: where does the LLM item prior add to the item mean?

Row strata of the S_d TEST pairs, fixed from the data before any scoring outcome: **sparse** = the row's q-hat was computed from fewer than
k = 5 other-user ratings strictly before the candidate (`forensics.prior_means`, `n_prior`), **dense** otherwise; **unseen** = the candidate's
item has no TRAIN example, **seen** otherwise (E-A secondary ii). Stratum UAUC = the mean over users with both classes among their stratum
rows of the per-user AUC; a stratum endpoint on fewer than 150 users is descriptive. Per stratum, regime and panel: UAUC of L, q-hat and
pi; **G_prior = dUAUC(M1 - M0)** and G_wu = dUAUC(M2 - M1), with E-W's predictors (fit on all S_d TEST rows of the other fold; the stratum
restricts the evaluation, not the fit). **H-S (directional).** On sparse rows G_prior > 0. Holm family E-H = {G_prior on sparse rows, per
Qwen Amazon panel run and regime (ZS, FT)}; a panel-regime whose sparse stratum has fewer than 150 users is descriptive and outside the
family. Everything else here (seen/unseen, dense, ML-1M) is descriptive.

## 6. E-J: the LLM against the non-personalised item mean

On the TEST users and rows of E-A, paired on identical users and rows with the user-bootstrap CI and two-sided p of E-B, per regime
(FT: the seed mean): **dUAUC(L - q-hat)**; **dUAUC(L - q-hat_T)** where q-hat_T is the information-matched item mean (other users' ratings
strictly before T_d only, shrunk with k = 5; q-hat of the registered text may use ratings in [T_d, t)); and dUAUC(L - MF) on the MF warm
pairs. **H-J (two-sided).** In the FT regime dUAUC(L - q-hat) is not 0. Holm family E-J = {dUAUC(L - q-hat), FT, per Qwen Amazon panel run};
the sign of a confirmed member is reported as found. Everything else (q-hat_T, MF, ZS, ML-1M, Llama) is descriptive.

## 7. E-C': a deployable top-k decision

E-C's top-k anatomy sets k from the user's TEST likes (an oracle). Add, descriptive and beside it, the same statistics with
k_u = clip(round(r_u n_u), 1, n_u - 1), n_u = the user's TEST rows and r_u the user's like rate among CAL rows (users with fewer than 3 CAL
rows: the pooled CAL like rate of the EVAL users). The paper calls E-C's decision "the oracle top-k decision" and uses E-C' for any
statement about what a system could do.

## 8. FT-C: reading rule and Toys

FT-C (section 9) stays descriptive and outside every Holm family; this item registers how its adapters are read, on both datasets, before any
permuted adapter has been scored. For dataset d with real adapters s0, s1, permuted adapters p0, p1 and the zero-shot model, on identical TEST
users and rows: **retention R = mean over s in {0, 1} of [UAUC(p_s) - UAUC(ZS)] / [UAUC(s) - UAUC(ZS)]**, defined only when the E-B contrast of
the dataset is positive with a CI excluding 0; its CI resamples users and recomputes all UAUCs in each resample. Label: **ITEM_DRIVEN** iff
the CI lower bound of R > 0.5; **USER_DRIVEN** iff the CI upper bound < 0.5; **MIXED** otherwise; **NOT_DEFINED** when R is not defined.
The paper may say that fine-tuning mostly teaches the item only if the label is ITEM_DRIVEN on every dataset on which FT-C was run.
**Toys is added to FT-C**: permuted adapters p0, p1 on a Toys `train_perm.jsonl` built exactly as section 9 builds the ML-1M one (the same
permutation function, seed 0), trained with the recipe of the Toys adapters s0-s2 and scored like them (`like` on the whole EVAL panel; about
3 GPU-hours). It is run by a new script, `scripts/sigir/run_ftc.sh`, so that no bound file changes: the script reads the training and scoring
arguments from the Toys s0 adapter's `train_config.json` and like-scoring run key, writes the permuted panel and its manifest under
`outputs/confrec/ftgrid/ftc/toys/`, and calls `ftgrid_report` (unchanged) with the six models into `outputs/confrec/ftgrid/report/toys_ftc.json`,
a second file beside the registered Toys report.

## 9. Admission and wording

1. **Replication means an interval, not a sign.** A statement that generalises beyond one panel (the abstract, "across domains", "both
   backbones") requires, besides confirmation in its own family or test, that each replication panel used has the same sign **and its own
   unadjusted 95% CI excluding 0**: Llama on ML-1M, and at least one fresh Amazon domain. Panel-scoped statements ("on ML-1M") need only
   their own test. This tightens Amendment 2 F.
2. **Mixed outcomes are reported as counts.** Where an endpoint is registered per domain and regime, the paper states how many panels confirm
   it per regime (for example "G is confirmed in k_ZS of 4 zero-shot and k_FT of 4 fine-tuned panels"), never one word for a mixed outcome.
3. **The wording rules are the decision functions of `scripts/sigir/fill_paper.py` and the rules above.** The sha1 of that script and of
   `docs/sigir/PAPER_DATA_MAP.md` is recorded in PILOT_LOG before any further result file is pulled. An extension that only adds specs for
   result files not yet pulled is recorded the same way (new sha1 in PILOT_LOG) before those files are pulled; a change to an existing
   decision function after results have been read needs a dated addendum that says what changed and why.

## 10. Not registered here (exploratory, labelled as such if run)

A longer-training arm (3 epochs) on ML-1M, a memorisation check of pi on ML-1M, a consumer-matched donor arm, non-LLM predictors of
cross-user serving, a seen/unseen split of the knockout (addendum 1 item 9 stays in force: not run). The S6 composition diagnostics and any
extra pruning arm belong to a later addendum, before the first S6 run.

## 11. Code and record

Implemented in new files `src/confrec/ftgrid_extra.py` (items 2-7's statistics from the stored score files and panels; CPU),
`scripts/sigir/run_ftextra.sh`, `scripts/sigir/run_ftc.sh` and `src/confrec/ftc_panel.py`, with unit tests; their sha1 are recorded in
PILOT_LOG before the statistics are computed for, or the Toys permuted adapters are trained on, any non-ML-1M panel. No bound file changes.
The anonymous third-party timestamp of this file's sha1 (OpenTimestamps) may be added by the authors.
