# Pre-registration amendment 3, addendum 5 (2026-10-04): two corrections to addendum 3

**Status: part of Amendment 3** (recorded with it). Written after the slot implementer re-read addendum 3 against the code, before any
prior-offset adapter was trained or any slot outcome existed. Both items correct a statement of addendum 3 that the code cannot honour;
neither changes a design element, threshold or claim rule.

1. **The answer-token share is not recorded.** Addendum 3 item 3 says that the share of Yes mass on the answer token "is recorded in
   the scorer's report". It is not: the scorer (`pyes_scorer.py`, a bound core file) records the summed yes/no log-probabilities, the
   total yes/no mass and the id sets, never a per-token share, and it stays unchanged. The mismatch between the training shift (the single
   answer token "Yes") and the test shift (b · z(q̂) added to the scorer's logit(Yes) − logit(No) over all yes/no ids) is therefore not
   measured; the slot's report states it as a limitation.
2. **Resolving an INCOMPLETE dataset.** Addendum 3 item 2 lets an INCOMPLETE dataset be "recorded as not run at a section-10
   checkpoint". The report code has no mechanical input for that record: `ftmethod_report --check_next` keeps refusing the later datasets
   while a dataset is INCOMPLETE. Lifting the block therefore needs a dated addendum (with its PILOT_LOG line) and a recorded change of
   the report code, written before any later dataset runs; until then the later datasets do not run.
