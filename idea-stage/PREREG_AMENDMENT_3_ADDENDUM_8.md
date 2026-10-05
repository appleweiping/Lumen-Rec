# Pre-registration amendment 3, addendum 8 (2026-10-05): the Toys permutation control is withdrawn; an item-only teacher control (FT-Q) replaces it

**Status: an extension of Amendment 3** (recorded with it; `ftgrid_freeze` treats every `PREREG_AMENDMENT_3_ADDENDUM_*.md` as part of the
amendment). Written before any permuted adapter or teacher adapter existed or was scored (no FT-C result on any dataset, no FT-Q result),
and before any Toys, Video_Games or Sports fine-tuned report, extra-analysis file or knockout analysis was read. What was read: the TRAIN panels'
item multiplicities (an input property, not an outcome; table below), the report of the implementer of addendum 6 item 8, and the same
material as addenda 6 and 7.

## 1. The Toys permutation run is withdrawn

A within-item label permutation (Amendment 3 section 9) cannot change an item that occurs once in TRAIN. Measured on the registered TRAIN panels
(the examples before T_d, after the cap):

| dataset | examples | items | share of examples in items with at least 2 examples | in items with both classes |
|---|---|---|---|---|
| ML-1M | 23,348 | 2,729 | 97.7% | 87.5% |
| Toys | 15,010 | 12,894 | 23.8% | 10.8% |
| Video_Games | 15,353 | 7,945 | 64.6% | 45.1% |
| Sports | 15,695 | 13,705 | 21.3% | 10.3% |

On Toys the permutation moves 768 of 15,010 labels (5.1%): the permuted adapters would train on about 95% of the real labels, and a high
retention R (hence ITEM_DRIVEN) would be nearly mechanical, which would license a claim the design cannot support. Therefore "Toys is added to
FT-C" (addendum 6 item 8) is **withdrawn: the Toys permutation run is not run**, and the script written for it is not used. FT-C remains the
registered ML-1M control of Amendment 3 section 9 (97.7% of its examples sit in repeated items), read with the rule of addendum 6 item 8.

## 2. FT-Q: an item-only teacher control

**Teacher.** For a dataset with TRAIN examples e = (user u, candidate i, time t < T_d), let q-hat_e be the prior-only item mean of the example
(`forensics.prior_means`, other users' first ratings of i strictly before t, shrunk with k = 5): exactly the values of the nested method slot's
stage 1, `outputs/confrec/ftmethod/<d>/train_qhat.csv.gz` (recorded manifest `qhat_manifest.json`). With beta = the mean of the real TRAIN labels,
the teacher gives label 1 to the round(beta n) examples with the largest q-hat_e and 0 to the others; ties in q-hat_e are broken by a fixed
seed-0 random key per example (the construction of Amendment 3 section 6's signals), so the teacher label rate equals beta exactly. The
teacher label is a function of the item and the time only: it uses no label of the example itself, no user information and nothing at or after
T_d. `train_q.jsonl` is `train.jsonl` with `candidate_labels` replaced by the teacher labels and nothing else changed (prompts, histories,
ratings shown in histories, order).

**Adapters and scoring.** Two adapters per dataset (seeds 0 and 1), trained on `train_q.jsonl` with exactly the arguments of that dataset's
adapters s0-s2 (the section-2 recipe; only `--train`, `--out`, `--seed` differ), scored with the `like` question on the whole EVAL panel with
exactly the arguments of the s0 like pass. They live in their own root, `outputs/confrec/ftgrid_q/`, under the names p0 and p1 (the control-adapter
names of `ftgrid_report` and `ftgrid_extra`; the real adapters' scores are linked into that root, never copied or rescored), so that no registered
file is rewritten.

**Reading.** Retention R_Q = mean over s in {0, 1} of [UAUC(q_s) - UAUC(ZS)] / [UAUC(s) - UAUC(ZS)] on identical TEST users and rows, defined only
when the E-B contrast of the dataset is positive with a CI excluding 0, the CI by user resampling that recomputes all UAUCs, and the labels
ITEM_DRIVEN (lower bound of R_Q above 0.5), USER_DRIVEN (upper bound below 0.5), MIXED, NOT_DEFINED, all exactly as addendum 6 item 8 defines them for
R; it is computed by `ftgrid_extra`'s FT_C_reading on the teacher root. Descriptive companions: UAUC(q_s) beside the UAUC of q-hat on the same rows
and R_Q above 1 reported as found (an item-only teacher can match or exceed the real adapter because q-hat is itself a strong item-level
predictor). No Holm family.

**Wording.** The paper may say that fine-tuning mostly teaches the item only if FT-C reads ITEM_DRIVEN on ML-1M **and** FT-Q reads ITEM_DRIVEN on
every dataset on which it was run; if either reads USER_DRIVEN or MIXED the paper reports the labels per dataset and control and says that the
evidence is mixed. A statement "the adapter learns item quality from item text" additionally needs FT-Q ITEM_DRIVEN on at least one Amazon dataset.

## 3. Datasets, order and cost

ML-1M and Toys are required; Video_Games and Sports are run unless cut at the 2026-10-29 checkpoint (Amendment 3 section 10; GPU-hours alone, recorded
before any FT-Q result of the affected dataset is read). Cost: about 2 x 46 min of training and 2 x 6 min of scoring on ML-1M (1.7 GPU-hours) and
2 x 48 min plus 2 x 16 min per Amazon dataset (2.1 GPU-hours); required 3.8, optional 4.2. They run after the Gate-FT-dependent grid jobs of the
queue and are single jobs per dataset. Qwen3-8B only.

## 4. Code and record

New files `scripts/sigir/run_ftq.sh` and `src/confrec/ftq_panel.py` (the teacher panel, its manifest and the control-root plumbing), with unit tests;
their sha1 are recorded in PILOT_LOG before the first teacher adapter is trained. No bound file changes (the trainer, scorer, report and
`ftgrid_extra` are used as they are). The registered gates are those of `run_ftgrid.sh` (a recorded GATE_FT_PASS, the amendment and core freeze
checks).
