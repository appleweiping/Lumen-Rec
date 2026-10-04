# Pre-registration amendment 3, addendum 4 (2026-10-04): S6 implementation readings

**Status: part of Amendment 3** (recorded with it). Written after the S6 implementer's report (`ftprune.py`, `run_ftprune.sh`), before the
zero-shot scoring pass of the TRAIN examples that its signals need, before any pruned adapter was trained and before any S6 outcome
existed. It changes no design element, threshold or claim rule; it records how section 6 is read where the text left a choice.

1. **TRAIN set and key.** The S6 TRAIN set is `outputs/confrec/gateft/train.jsonl` (the 23,348 pre-T DEV examples, checked against
   `gateft_split.json`); an example is keyed `user::item`. The 25% rule removes `floor(n_c/4 + 1/2)` examples of class c (exactly 25%
   when the class size is a multiple of 4, as on ML-1M: 15,412 positives and 7,936 negatives, so 3,853 and 1,984 are removed).
2. **Signals.** τ is the numpy-linear (1 − β)-quantile of the zero-shot `like` logit over the TRAIN examples whose logit is finite; an
   example without a finite signal is ranked last (never removed by P2 or P3 before any finite one). P1 removes the examples with the
   smallest sha1 of `"P1:<seed>:<key>"` within each class; ties of u or C are broken by the sha1 of `"tie:0:<key>"`.
3. **The zero-shot TRAIN pass is not an S6 run.** The signals need one zero-shot `like` scoring of the TRAIN examples under V1 (stage A of
   `run_ftprune.sh`). It is a section-0 zero-shot scoring (the `core` record is required) and no pruned adapter exists before the `prune`
   record, which contains the signal, subset and pruned-file hashes (addendum 2, item 1).
4. **Claim labels.** The labels follow section 6 in this order: BEATS (CI lower bound > 0, mean > 2 × SD of the paired differences, all 5
   differences positive), WORSE (mirrored), ABOUT_EQUAL (95% CI within ±0.01), else INCONCLUSIVE; when BEATS and ABOUT_EQUAL both hold (a
   CI inside (0, 0.01]) BEATS is the label and both flags are reported. The bootstrap p-value and whether it is below 0.05 are reported
   beside the label; for 5 seeds the "all positive" clause is implied by "mean > 2 × SD" and is kept as written. Below 150 users the
   label is DESCRIPTIVE_MIN_N and p is not reported (ML-1M has 366).
5. **Per-run AUC rows.** Each run's UAUC uses that run's own finite TEST rows (the G9 statistic), not the rows finite in every model; the
   two coincide unless a row is censored.
