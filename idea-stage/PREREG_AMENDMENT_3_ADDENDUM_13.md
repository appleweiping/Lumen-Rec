# Pre-registration amendment 3, addendum 13 (2026-10-06): item-balanced fine-tuning FT-B and its size-matched control FT-S (exploratory, outcome-free)

**Status: part of Amendment 3** (recorded with it; `ftgrid_freeze` treats every `PREREG_AMENDMENT_3_ADDENDUM_*.md` as part of the amendment). Written after the ML-1M, Toys, Video_Games and Sports fine-tuned
grid reports, the FT-C and FT-Q readings and the Llama results had been read, and **before any FT-B or FT-S adapter was trained, scored or reported**. It adds one exploratory arm to the fine-tuned
programme. It changes no design element, endpoint, threshold or family of Amendment 2, Amendment 3 or addenda 1-12.

## 1. Question

Standard supervised fine-tuning lets the adapter reach the labels through the item: FT-C (labels permuted within items) keeps 0.92 of the gain and FT-Q (labels from the item mean alone) keeps 0.85 on ML-1M,
and the pair-specific residual of the fine-tuned confidence adds at most 0.010 UAUC to the prior and an item mean. A reviewer's objection is that this reflects the objective, not the model: while the item
label rate is learnable, the adapter has no reason to learn user--item evidence. FT-B removes that shortcut from the training labels and asks whether personal evidence is then learnable.

## 2. Arms

- **Data (ML-1M only, Qwen3-8B, the registered Gate-FT TRAIN panel).** The TRAIN examples of the registered grid panel (`outputs/confrec/ftgrid/panels/ml1m/train.jsonl`, bytes equal to Gate-FT's `--train`).
  *FT-B (item-balanced):* for every item with at least one positive and one negative TRAIN example, keep m_i = min(n_i^+, n_i^-) positive and m_i negative examples, chosen by the fixed seed-0 key of
  `src/confrec/ftprune.py` (`tie_key`); every kept item then has TRAIN label rate exactly 0.5, and items with a single class are dropped. The kept panel has K examples (recorded). Prompts, histories
  and the label alphabet are those of the registered panel and are not changed.
  *FT-S (size-matched random):* K examples drawn uniformly without replacement from the full TRAIN panel by the seed of the adapter (item label rates preserved in expectation); it isolates the effect
  of removing the item-level label signal from the effect of fewer examples and optimizer steps.
- **Training and scoring.** Exactly the registered recipe of the SFT adapters (LoRA r 16, alpha 32, dropout 0.05 on q/k/v/o, lr 1e-4, cosine, 3% warm-up, one epoch over the kept panel, bf16, effective batch 32), seeds 0, 1, 2
  for each arm (adapters b0, b1, b2 for FT-B and r0, r1, r2 for FT-S), the `like` pass on the registered TEST rows with the scorer of the grid, the swap-prior arm and the integrity check E1 with the grid's rerun-once rule.
  The number of optimizer steps is recorded and is smaller than the full-panel SFT's.
- **Conditional replication.** If the ML-1M result reads `PERSONAL_EVIDENCE_LEARNABLE` (section 4), the same two arms run on Llama-3.1-8B-Instruct, ML-1M, seeds 0-2; otherwise they are not run. Amazon panels are not run in this addendum (their TRAIN
  items rarely repeat, see the multiplicity table of addendum 8; K per domain is computed from the panels and reported, no adapter is trained).

## 3. Endpoints (all exploratory; no family; user-cluster bootstrap, 2,000 resamples, seed 0; the registered estimators of the grid report and the extra-analysis module are used unchanged)

1. UAUC of the confidence beside zero-shot, the SFT adapters, the item mean q-hat and temporal MF on the same rows; the contrasts UAUC(FT-B) - UAUC(FT-S), UAUC(FT-B) - q-hat and UAUC(FT-B) - MF (warm).
2. The information gain G (E-D) and G_wu (E-W) of each arm and the paired contrast G(FT-B) - G(FT-S) (same users, the seed means), with G_CF as the reference.
3. The stacked UAUC of the cross-fitted stacker [q-hat, confidence of the arm] minus UAUC(q-hat): the personal evidence that the arm adds to an item mean ('stack gain'), and its contrast with the stack gain of MF.
4. The item-prior share of the confidence (swap prior, as in the grid report) and the e-share, to show whether the objective removed the item prior from the confidence.
5. Descriptives: K, the number of kept items, the share of TRAIN examples dropped, the number of optimizer steps, the seed s.d. of every statistic.

## 4. Reading rule (fixed now)

`PERSONAL_EVIDENCE_LEARNABLE` iff, for FT-B (mean over the three seeds), G > 0 with a 95% interval above 0, G(FT-B) - G(FT-S) > 0 with a 95% interval above 0, all three seeds positive and the mean above
2 sigma_seed; `NOT_LEARNED` iff the 95% interval of G(FT-B) lies within [-0.01, +0.01] and the interval of G(FT-B) - G(FT-S) contains 0; otherwise `INCONCLUSIVE` (checked in this order). The stack gain of
endpoint 3 is described with its interval and is never a label. A result under this addendum never enters the abstract or a claim beyond ML-1M unless it is replicated on the second backbone (the claim-admission
rule of addendum 6 section 9), and it is marked exploratory wherever it appears.

## 5. Code and record

New scripts and modules only (the recorded FT-C, FT-Q, grid and extra-analysis code is imported or copied read-only and stays byte-identical): a panel builder for FT-B and FT-S, a runner with the guards of
`run_ftq.sh` (own root outside the registered roots, DRY_RUN allow-list, link sweep, freeze checks, E1 rerun-once rule with the same-key refinement), and a reading module that applies section 4 to the stored
reports. Their sha1 and those of the two built panels are recorded in the pilot log before the first adapter is trained. About 3 GPU-hours per arm and seed block.
