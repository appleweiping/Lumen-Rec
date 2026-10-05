# Pre-registration amendment 3, addendum 9 (2026-10-05): a longer-training robustness arm (FT-L), exploratory

**Status: an extension of Amendment 3** (recorded with it; `ftgrid_freeze` treats every `PREREG_AMENDMENT_3_ADDENDUM_*.md` as part of the amendment).
**Exploratory: no hypothesis, no Holm family, no claim-admission role.** Written **after** the Toys fine-tuned report and the Toys extra analyses of
addendum 6 were read (code and wording rules had been recorded before): on Toys the LoRA model (1 epoch, r = 16) sits near chance (UAUC about 0.56), far
below the item mean (about 0.68), with no personal information gain, and on ML-1M the LoRA model is also below the item mean. The obvious objection is that
the registered recipe under-trains the model. FT-L answers it descriptively; if longer training changes a reading, the paper says so and scopes the
corresponding statement to the registered recipe.

## 1. Definition

Qwen3-8B, the prompt of `selection.json`, the registered TRAIN panels (`panels/<d>/train.jsonl`, real labels), the section-2 recipe in every respect
(LoRA r = 16, alpha 32, dropout 0.05 on q/k/v/o, lr 1e-4, cosine with 3% warm-up, bf16, effective batch 32, last-token loss) **except `--epochs 3`**. Seeds 0 and 1
(two adapters per dataset, named s0 and s1 inside the root `outputs/confrec/ftgrid_len3/`). Datasets: ML-1M and Toys. The adapters are scored with the same `like`
question on the whole EVAL panel and with the swap-prior arm on S_d's TEST rows, exactly as the registered scoring stages score their adapters (the arguments are read
from the real adapters' recorded configuration; only the training file stays and `--epochs` and `--seed`, `--out` change). The zero-shot scores are those of the
registered run (linked, never rescored). No other arm (no-history, star permutation, knockout) is scored.

## 2. Readout (descriptive)

From `ftgrid_report` (unchanged) on the 3-epoch adapters and the zero-shot model: UAUC of the LoRA logit with its CI, the contrast with zero-shot (E-B) and with the
1-epoch adapters on identical users and rows, the references on the same rows, the item-prior share and non-prior share, G (pooled) and, from `ftgrid_extra`
(unchanged), G_wu, dUAUC(L - q-hat) and the e-share, all marked "two seeds, exploratory". The paper may print them in the appendix or in one sentence of Section 6.
A statement that the registered result is robust to longer training needs the 3-epoch UAUC to stay below the item mean on both datasets and G (pooled and
within-user) to stay within 0.01 of the 1-epoch values; otherwise the paper reports the change and withdraws the robustness sentence.

## 3. Cost, order, record

About 13 GPU-hours in total (ML-1M about 5.6, Toys about 7.7), queued after every registered job and cut whenever GPU time is needed for a registered item (no
checkpoint is needed). New files `scripts/sigir/run_ftlen.sh`, `src/confrec/ftlen_plumbing.py` (or the FT-Q helper module if it is reused) and tests; their sha1 are
recorded in PILOT_LOG before the first adapter is trained; no bound file changes.
