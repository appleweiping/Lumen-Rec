# Pre-registration amendment 3, addendum 3 (2026-10-04): the optimizer of the offset b, and two readings of section 7

**Status: part of Amendment 3** (recorded with it). Written before any prior-offset adapter was trained, before any slot outcome
existed, and while the Gate-FT adapters were still training. The implementer of the slot (code review of section 7) found that the
registered text leaves b without an optimizer setting and that the literal reading defeats the slot; this addendum fixes that from
the optimizer's arithmetic, not from any result.

1. **b has its own optimizer group.** Section 7 says b is "a learnable scalar (init 0)". Left in the LoRA AdamW group (lr 1e-4), b
   can move by at most about the sum of the learning rates over the run (≈ 0.04 on ML-1M with about 730 optimizer steps, ≈ 0.02 on an
   Amazon domain), so b · z(q̂) would stay below roughly 0.1 logit and the arm would be SFT by construction: the slot would fail for an
   optimizer reason, not a modelling one. b is therefore trained in its own AdamW parameter group with **learning rate 1e-2**, the
   same cosine schedule with 3% warmup, weight decay 0 and initial value 0. The value is fixed here, before any prior-offset run, and
   is never tuned; every other setting of section 2 is unchanged. (A slot that then fails, fails as a modelling result.)
2. **INCOMPLETE is not a failure.** A dataset whose slot result is INCOMPLETE (for example a FAILED_INTEGRITY seed that section 2
   forbids replacing) does not count toward "failed on 2 datasets"; it blocks the next dataset until it is resolved or recorded as
   not run at a section-10 checkpoint.
3. **Train and test shifts.** Training shifts the single answer token "Yes" by b · z(q̂) before the full-vocabulary softmax; the test
   score adds b · z(q̂) to the scorer's logit(Yes) − logit(No) (sum over the yes/no ids), as section 7 says. The two agree when the
   answer token carries nearly all of the Yes mass, which the report states (the share of Yes mass on the answer token is recorded
   in the scorer's report).
