# Pre-registration amendment 3, addendum 2 (2026-10-04): what the `prune` and `method` freezes bind

**Status: part of Amendment 3** (recorded with it; `ftgrid_freeze` includes every `PREREG_AMENDMENT_3_ADDENDUM_*.md`). Written while
the Gate-FT adapters were still training, before any adapter, zero-shot panel, pruning or slot run had produced an outcome statistic.
It changes no design element, threshold or claim rule; it makes the two later freezes of section 0 concrete.

1. **`prune` (section 6).** Besides the code files of its block, the record contains `outputs/confrec/ftprune/prune_manifest.json`: the
   sha1 of the signal file (u and C per TRAIN example), of every subset index file and of every pruned TRAIN file, the per-class
   counts removed, the TRAIN label mean, the threshold τ and the code sha1 of `ftprune.py`. `ftgrid_freeze --print --stage prune`
   writes these lines; no S6 adapter is trained before they are in `docs/sigir/PILOT_LOG.md`.
2. **`method` (section 7).** Besides the code files of its block (this addendum adds the report code below), the record contains, per
   dataset, the manifest of the TRAIN-standardisation constants and q̂ files written by stage 1 of `run_ftmethod.sh`
   (`ftgrid_freeze --print --stage method --split <manifest>`); no prior-offset adapter of a dataset is trained before its manifest is
   recorded. The kill rule of section 7 is evaluated in the registered dataset order by the report code.
3. **The freeze tool itself** (`src/confrec/ftgrid_freeze.py`, a bound file) was extended for 1 and 2 and to read FREEZE blocks from the
   addenda; its new sha1 is recorded with this addendum. Nothing else of the core list changed.

<!-- FREEZE_FILES method -->
src/confrec/ftmethod_report.py
<!-- /FREEZE_FILES -->
