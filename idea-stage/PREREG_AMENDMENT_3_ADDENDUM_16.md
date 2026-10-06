# Pre-registration amendment 3, addendum 16 (2026-10-06): a readout control for the retention readings, FT-G (exploratory, outcome-free)

**Status: part of Amendment 3** (recorded with it; `ftgrid_freeze` treats every `PREREG_AMENDMENT_3_ADDENDUM_*.md` as part of the amendment). Written after the FT-C reading on ML-1M, the FT-Q readings on ML-1M and Toys and every rated-panel result had been read and after a third independent
same-family review of the draft had pointed out that no control removes a readout or answer-format shift (an adapter can raise UAUC by sharpening the Yes/No contrast without learning anything about items or users), so that the retention ratios R and R_Q of addenda 6 and 8 may be inflated by a gain that
the labels did not cause; and **before any FT-G adapter, panel or score existed**. It adds one exploratory arm and one readout-adjusted retention. It changes no design element, endpoint, threshold or family of Amendment 2, Amendment 3 or addenda 1-15, and never changes a number of the registered reports.

## 1. Question

How much of the gain of the LoRA over zero-shot (E-B) is produced by the training format alone, and how much of the retention of the item-only controls survives once that part is removed?

## 2. Arm

*FT-G (global permutation).* For a dataset, all labels of the registered TRAIN panel (`outputs/confrec/ftgrid/panels/<d>/train.jsonl`, bytes equal to the registered one) are permuted among all TRAIN examples by a fixed seed-0 key (the label multiset, hence the label rate, is unchanged and no label stays attached to its
example except by chance); prompts, histories and the label alphabet are unchanged. Adapters g0 and g1 (seeds 0, 1) are trained with the registered recipe (LoRA r 16, alpha 32, dropout 0.05 on q/k/v/o, lr 1e-4, cosine, 3% warm-up, one epoch, bf16, effective batch 32) and scored with the `like` pass on the registered
TEST rows and with the swap-prior arm of the grid, with the grid's integrity check E1 and its rerun-once rule. Datasets: ML-1M and Toys (required); Video_Games and Sports if the GPU is idle at the 2026-10-22 checkpoint of Amendment 3 section 10 (a cut is recorded as a pilot-log line before any FT-G result is read). The runner,
guards and layout are those of the FT-B runner of addendum 13 (own roots outside the registered ones, DRY_RUN allow-list, link sweep, freeze checks, sticky FAILED_INTEGRITY with a logged override), with a dataset argument.

## 3. Endpoints (exploratory; no family; user-cluster bootstrap, 2,000 resamples, seed 0; the registered estimators of the grid report and the extra-analysis module are used unchanged)

1. The readout shift: UAUC(FT-G) - UAUC(zero-shot) (the seed mean of the two adapters) with its interval, and the share of the E-B contrast (LoRA minus zero-shot) that it represents, with a Fieller interval for the share.
2. The readout-adjusted retention of every control that exists on the dataset: R_adj = [UAUC(control) - UAUC(FT-G)] / [UAUC(real LoRA) - UAUC(FT-G)] for the permuted-label adapters (FT-C, ML-1M), the item-teacher adapters (FT-Q) and the item-balanced adapters (FT-B, FT-S, FT-N of addendum 13), with the seed means, the paired user resamples and a Fieller interval; R_adj is defined only when the interval of the denominator excludes 0.
3. The information gain G (E-D and E-W) and the item-prior share of FT-G, as a check that a label-free adapter carries no personal evidence.

## 4. Reading rule (fixed now)

`READOUT_SHIFT` if the interval of endpoint 1 lies above 0; `NO_READOUT_SHIFT` if it lies within [-0.01, +0.01]; otherwise `INCONCLUSIVE`. The labels ITEM_DRIVEN, USER_DRIVEN and MIXED of addenda 6 and 8 (interval above, below or around 0.5) are applied to R_adj in addition to the registered R, and both are reported; no registered label changes.
A result of this addendum never enters the abstract or a claim beyond the datasets run unless it follows the claim-admission rule of addendum 6 section 9; it is marked exploratory.

## 5. Code and record

Part of the code of addendum 13 (a mode of its panel builder, runner and reading module); their sha1 and that of every built panel are recorded in the pilot log before the first adapter of either addendum is trained. About 1 GPU-hour per adapter on ML-1M and 2.5 on Toys (the queue times of the registered adapters).
