# Pre-registration amendment 3, addendum 1 (2026-10-04): implementation clarifications

**Status: part of Amendment 3** (its sha1 is recorded with the amendment's, `ftgrid_freeze` treats every
`PREREG_AMENDMENT_3_ADDENDUM_*.md` as part of it). Written after the three implementers' reports (data builder, run script,
endpoint report), before any adapter was scored and before any outcome statistic of an adapter, a zero-shot panel of section 4 or
a pruning/slot run existed. It changes no design element, no threshold and no claim rule: it records how the registered text is
implemented where the text left a choice, so that the code that the freeze binds and this text agree.

1. **CF references never train on the panel's own candidates.** `forensics.cf_references` (explicit cutoff T_d) leaves every EVAL
   candidate event out of the item-parameter fit, including CAL rows before T_d (they are labels of the evaluated users; section 1:
   "never trained on"), and folds the user in on the user's own events before the first candidate. Item parameters are fit on all
   other ratings with ts < T_d.
2. **q̂ is cut at the candidate, not at T_d.** A TEST pair's q̂(u,i,t) uses the ratings of other users strictly before t, which can
   include ratings in [T_d, t); that is the non-transductive reading of section 3 ("strictly before the candidate's timestamp").
   No rating at or after T_d enters any TRAIN-side feature (unchanged).
3. **Popularity controls.** E-E controls the shrunk q̂ for every domain (partial Spearman on the shrunk mean, item and pair level);
   ML-1M also release year; Toys and Sports also description length and has_store; Video_Games controls q̂ only, with description
   length and has_store reported as a point-estimate sensitivity (section 3 names no further control for Video_Games).
4. **E-D uses the like arm for L; star permutation averages per-user AUC over the two permutations.** ΔUAUC = UAUC(L) − the mean
   over the K = 2 permutations of the per-user AUC (the `diag_battery` convention); the UAUC of the averaged permuted logit is
   reported beside it. The star permutation permutes the ratings among the history items shown, i.e. within the selected variant's
   registered window.
5. **Reliability tables.** The 10-bin reliability table is reported with equal-mass bins (the ECE's own bins) and equal-width bins.
   Every E-C bootstrap resample refits the Platt map. The invariance-check's user offset is the user's mean logit over TEST rows;
   `max_abs_dAUC_user` is the larger of the Platt and user-centred values (both reported).
6. **Integrity of the swap arm.** For the swap runs the E1 check covers censored = 2 and overlength prompts of the donor prompts
   (their Yes+No mass is not recorded by the scorer).
7. **Missing seeds.** A regime mean computed from fewer than 3 seeds is marked `complete = false`; P1 then reads INCOMPLETE.
8. **Cross-domain families.** The Holm families of section 11 (E-B over the four Qwen domains; E-D per domain and regime; the slot
   over the datasets run) are computed by a cross-domain summary step over the per-domain reports, never inside a per-domain report
   (each domain report carries its own raw p-values).
9. **Knockout secondary.** The seen/unseen-item split of the knockout (section 5, secondary) is not implemented; it is reported as
   not run unless it is added by a later dated addendum before the first knockout result exists.
10. **Run script.** `run_ftgrid.sh`: the Sports knockout runs unless `KNOCKOUT_SPORTS=0` cuts it (section 5); a `VARIANT` other than
    `gate_ft_prompt` is refused (section 2); the Llama output root writes its own `ftgrid_split.json` (the length audit depends on the
    tokenizer) and the record of a Llama job includes it (section 0); the permuted adapters p0 and p1 get the `like` pass only;
    without a recorded GATE_FT_PASS only the zero-shot `like` and decomposition arms run (section 4) and no knockout arm runs,
    zero-shot arms included (section 5); scoring stage 6 (the report) also needs the full record because the report code is bound.
11. **Derived panels.** The star-permuted, pseudonymised and placebo panels are deterministic functions (seed 0) of the files listed
    in `ftgrid_split.json` and of bound code (`starperm_panel.py`, `diag_battery.py`, `pseudonymize.py`); they are not listed in the
    split file's `files`. The placebo's brand popularity comes from the category-wide sidecar (Amendment 1 P3).
