# Pre-registration amendment 3, addendum 10 (2026-10-05): hardening of the method-slot code and a train/test-shift diagnostic

**Status: part of Amendment 3** (recorded with it; `ftgrid_freeze` treats every `PREREG_AMENDMENT_3_ADDENDUM_*.md` as part of the amendment). Written after an
independent same-family code review of the method-slot code and **before any prior-offset adapter was trained, any slot outcome existed or any Toys,
Video_Games or Sports fine-tuned result had been used for the slot**. It changes no design element, threshold, comparator or claim rule of section 7 (the model, b's own
optimizer group at lr 1e-2, the endpoint, the pass rule +0.01 / CI / sigma_seed, the kill rule, the order); it supersedes the code record of addendum 5 for
`ftmethod_report.py` and `run_ftmethod.sh` (the new sha1 are recorded in PILOT_LOG before the `method` record) and leaves `train_lora_offset.py` untouched.

## 1. Implementation corrections (no change of the registered rule)

1. **Rehearsal isolation.** `run_ftmethod.sh` canonicalises its roots (`realpath -m`), accepts a DRY_RUN root only under `tmp_outputs/` or outside the repository's parent
   directory, refuses links, and accepts `DRY_RUN` only as 0 or 1. A rehearsal report can no longer enter the registered root.
2. **Invalid reports.** `ftmethod_report` treats a dataset report whose meta says `dry_run_inputs`, `n_boot != 2000` or `seed != 0` as INVALID: it is never counted
   as a decided dataset, never as a failure and never unlocks the next dataset.
3. **The +0.01 threshold** is compared as `mean >= 0.01 - 1e-12` (a mean of exactly +0.0100 in exact arithmetic is computed as 0.009999999999999998).
4. **Hard kill date.** The date is checked before every seed of stage 3 (not only at the start of a stage), `DRY_TODAY` must be eight digits, and a dataset without a
   finished report on 2026-11-30 reads NOT_RUN.
5. **Cuts.** A dataset cut at the 2026-10-29 checkpoint (Amendment 3 section 10) is recorded as a PILOT_LOG line with the exact token `FTMETHOD_NOT_RUN <dataset>`;
   `ftmethod_report` then reads it as NOT_RUN: it is neither decided nor failed, and the slot's next dataset skips it.
6. **E1 reruns** use the same-key rule of the other control scripts (only a `.e1fail.*` directory whose run key equals the current one uses up the one rerun).
7. **Documentation:** "with b = 0 the run is SFT's" holds for a frozen b (the registered run trains b, and b's gradient then enters the LoRA clip norm).

## 2. A train/test-shift diagnostic (outcome-free, descriptive)

Training shifts only the answer token 'Yes' (token id 9454); the registered test-time score is the scorer's logit(Yes) - logit(No) over all Yes and No ids plus b z(q-hat)
(addendum 5 item 1 stated the mismatch as a limitation). A training-consistent score is C = R + log(s + (1 - s) exp(-b z)) with R the registered score and s the share of the
yes probability mass on the answer token. For every dataset, **before its slot report is built**, a new script records, for each prior-offset adapter o0-o2 on the first 1,000
rows (in panel order) of the dataset's EVAL CAL candidates, the per-id top-50 log-probabilities of the next token, and reports the distribution of s (mean, median, 1st percentile)
and the UAUC of R and of C on those rows. **Reading rule:** if the 1st percentile of s is at least 0.99 the registered score is read as equivalent to the training-consistent one for ranking; if it is
lower, a FAIL of that dataset is reported as *inconclusive for the method* (the test-time rule then works against it) and counts as a failure for the kill rule exactly as before,
while a PASS stands as it is (the mismatch only works against the method). The diagnostic has no hypothesis and no family. It needs about 15 minutes of GPU time per dataset.

## 3. Code and record

`scripts/sigir/run_ftmethod.sh`, `src/confrec/ftmethod_report.py` (corrected), new `src/confrec/ftmethod_shift_diag.py` and `scripts/sigir/run_ftmethod_shift_diag.sh`, with tests. Their sha1 are
recorded before the method record; no bound core file changes; `train_lora_offset.py` keeps its recorded sha1.
