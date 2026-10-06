# Pre-registration amendment 3, addendum 14 (2026-10-06): instrument validity of the information gain G (exploratory, outcome-free)

**Status: part of Amendment 3** (recorded with it; `ftgrid_freeze` treats every `PREREG_AMENDMENT_3_ADDENDUM_*.md` as part of the amendment). Written after every rated-panel result of the fine-tuned programme
(Qwen3-8B on ML-1M, Toys, Video_Games and Sports; Llama-3.1-8B on ML-1M and Toys) had been read and after three independent same-family reviews of the draft had shown, by recomputation from the stored files, that the registered
E-D estimator of G disagrees with the within-user estimator on some panels (Sports zero-shot: -0.0095 against +0.0016), that its bootstrap holds the stacker coefficients fixed, that M2 = [m, pi-hat, e-hat] spans the space of
[m, pi-hat, l] so that G can be positive from denoising the 8-donor prior alone, and that the minimum detectable gain was never stated; and **before any quantity of this addendum was computed on any panel**. It adds descriptive
CPU-only analyses. It changes no design element, endpoint, threshold or family of Amendment 2, Amendment 3 or addenda 1-13, and never changes a number of the registered reports.

## 1. Questions

(a) Is a null or negative G a property of the estimator and the panel (stacker overfitting, donor-prior noise) rather than of the model? (b) Which planted personal signal would the G estimators recover at the observed numbers of users (power, minimum
detectable gain)? (c) How much of a positive G is personal evidence and how much is denoising of the 8-donor prior? (d) How wide are the intervals when the stacker is refitted inside every resample and when items, not only users, are resampled?
(e) How wide are honest intervals for the retention ratios R and R_Q, whose denominators are small?

## 2. Analyses (rated panels; Qwen3-8B zero-shot and the three LoRA seeds, and Llama-3.1-8B where the registered Llama reports exist; the rows, users S_d, splits, donors and stackers of the registered reports and of `src/confrec/ftgrid_extra.py`, imported unchanged)

1. **Permutation null of G and G_wu.** For each panel and regime, the pair-specific residual e-hat of every S_d user is permuted among that user's TEST pairs (fixed seeds; 500 permutations) and G (E-D estimator) and G_wu (E-W estimator) are recomputed with the
   registered cross-fitted stackers; reported: the mean and the 2.5th and 97.5th percentiles of the null G, the observed G, and the share of permutations with G at least the observed value. (The null is the information gain of a residual that keeps its
   marginal distribution and its user offset but carries no pair-label evidence.)
2. **Planted-signal recovery and power.** For each panel and regime a synthetic personal signal c = rho * y~ + sqrt(1 - rho^2) * eps (y~ the within-user centred TEST label residual after removing the registered item mean, eps independent N(0,1), fixed seeds) is scaled
   by lambda * sd(e-hat) and added to the confidence: l' = l + lambda * sd(e-hat) * c, for rho = 0.3 and lambda in {0.25, 0.5, 1, 2} (100 replications each; the residual is recomputed from l' and the registered prior). Reported per panel, regime and lambda: the mean
   recovered G (E-D and E-W), the power (the share of replications whose 95% user-bootstrap lower bound of G, 500 resamples, is above 0) and the minimum detectable gain (the smallest mean recovered G with power at least 0.8).
3. **Donor placebo.** The swap arm stores, for every S_d TEST row, the confidences of K = 8 donor histories for the row's item. With the eighth donor held out, pi-hat_(-8) is the mean of the other seven and the placebo residual is
   e-hat_pl = l_swap(row, donor 8) - pi-hat_(-8), centred within user like e-hat. Reported per panel and regime: G_7 = Delta UAUC([m, pi-hat_(-8), e-hat_7] - [m, pi-hat_(-8)]) with e-hat_7 = l(u, i) - pi-hat_(-8) (the user's own confidence), G_pl the same with e-hat_pl,
   and the paired contrast G_7 - G_pl (same users, 2,000 user resamples, seed 0): the part of G that the user's own history carries and a held-out donor's history for the same item does not. Both E-D and E-W stackers.
4. **Honest intervals.** For G, G_wu and P1 (ML-1M, Qwen3-8B) and for G of every panel and regime: (a) a full-refit bootstrap (the cross-fitted stackers are refitted inside each of 500 user resamples) beside the registered fixed-coefficient interval; (b) on ML-1M, where
   97.7% of TRAIN examples sit in repeated items, a two-way resampling of users and items (pigeonhole weights; if it is not computable for the UAUC functional the implementer reports why and gives an item-block bootstrap instead); (c) a cross-panel Holm sensitivity computed
   from the stored p-values: over the 8 E-D G tests (four panels, two regimes) and over all E-D tests of the registered reports, reported as adjusted p-values beside the registered family-wise ones.
5. **Ratios.** For the retention ratios R (FT-C) and R_Q (FT-Q) of the stored extra-analysis files: the numerator and denominator intervals, a Fieller interval for the ratio of the seed-mean differences from the paired user resamples, whether the denominator's interval
   excludes 0, and the real-minus-control UAUC contrast with its interval.
6. **Bounds.** From the stored reports (no new computation): the upper 95% bound of G, G_wu, G_LLM|CF for every panel and regime and whether the whole 95% interval lies within [-0.01, +0.01].

## 3. Reading rules (fixed now)

No hypothesis, no family, no Holm correction (item 4c is a sensitivity table), no direction word. A G is called `WITHIN_NULL` if it lies inside the 2.5th-97.5th percentile range of its permutation null (item 1), `BELOW_NULL` if below and `ABOVE_NULL` if above;
the personal part of G is `PERSONAL` if the interval of G_7 - G_pl lies above 0, `DENOISING_ONLY` if the interval of G_7 contains 0 or G_7 - G_pl lies within [-0.005, +0.005], otherwise `INCONCLUSIVE`. Item 2 is a validity check of the estimator and never changes a registered result. A
statement beyond one panel follows the claim-admission rule of addendum 6 section 9. Everything is marked exploratory.

## 4. Code and record

New module and runner only (`src/confrec/ftgrid_instrument.py`, `scripts/sigir/run_ftinstr.sh`) with tests; the registered report and extra-analysis code is imported read-only and stays byte-identical. CPU only; the container has a 60 GiB RAM cap: the run
must stay below 20 GiB; writes only below `outputs/confrec/ftgrid_instr/`. Their sha1 are recorded in the pilot log before the first run on a real panel.
