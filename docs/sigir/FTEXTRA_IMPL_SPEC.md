# Implementation contract for Amendment 3 addendum 6 (ftextra)

Binding text: `idea-stage/PREREG_AMENDMENT_3_ADDENDUM_6.md` (A3-6), with Amendment 3 (`idea-stage/PREREG_AMENDMENT_3.md`, A3) and
addenda 1-5. This file fixes **interfaces** only; where it and A3-6 or A3 disagree, the registered text wins and the disagreement
is reported. Nothing here is a pre-registered choice.

## Ownership (one writer per file) and prohibitions
| owner | files |
|---|---|
| X1 analysis | `src/confrec/ftgrid_extra.py`, `tests/test_confrec_ftgrid_extra.py`, `scripts/sigir/run_ftextra.sh` |
| X2 FT-C Toys | `scripts/sigir/run_ftc.sh`, `src/confrec/ftc_panel.py`, `tests/test_confrec_ftc.py` |

Nobody edits another owner's file, and **nobody edits a bound file** (the core FREEZE_FILES block of A3: `ftgrid_data.py`,
`ftgrid_report.py`, `ftgrid_freeze.py`, `run_ftgrid.sh`, `train_lora_yesno.py`, `lora_trainer.py`, `pyes_scorer.py`,
`forensics.py`, `stats.py`, `metrics.py`, `diag_battery.py`, `pilot_pseudonym.py`, `pseudonymize.py`, `prompting.py`,
`starperm_panel.py`, ... ; run `python -m src.confrec.ftgrid_freeze --print --stage core` to list them). New code imports them
read-only. Nobody runs git in `D:\Research\Lumen`. No GPU job is started from the local machine; the server is touched only through
`scripts/sigir/remote.ps1`, for read-only inspection or CPU unit tests (`CUDA_VISIBLE_DEVICES=""`, `nice`), never for a registered
run. **Do not open or print any Toys, Video_Games or Sports fine-tuned result** (`outputs/confrec/ftgrid/report/{toys,games,sports}.json`,
`scores/{toys,games,sports}/s*/`): A3-6 is registered outcome-free for them. ML-1M files may be read (A3-6 labels ML-1M exploratory).
Develop against the synthetic worlds of `tests/test_confrec_ftgrid_report.py` and the real ML-1M report/scores
(`docs/sigir/results/grid/qwen/ml1m.json` locally; the scores are on the server). No credential appears in any file.

## X1: `src/confrec/ftgrid_extra.py`
CPU only. A pure function of stored files, deterministic (seed 0, 2,000 user resamples, no clock or host in the output).

CLI (mirrors `ftgrid_report`): `python -m src.confrec.ftgrid_extra build --domain D --split PATH --panels DIR --scores_root DIR
--models zeroshot,s0,s1,s2[,p0,p1] --raw data/raw --out FILE.json [--n_boot 2000] [--seed 0]` and
`python -m src.confrec.ftgrid_extra summarize --files extra/toys.json extra/games.json extra/sports.json [--ml1m extra/ml1m.json]
--out extra/summary.json`. Output root `outputs/confrec/ftgrid/extra/` (Llama: `outputs/confrec/ftgrid_llama/extra/`); per-domain
`D.json` and `D_tables.csv`.

**Reuse, do not re-derive.** Load the rows, scores, contexts and references through `ftgrid_report`'s own functions (`make_ctx`,
`load_run`, `to_pairs`, `compute_refs`/`load_refs`, `Clusters`, `uauc_models`, `contrast_models`, `crossfit`, `logit_fit`,
`shares_from_sums`, `rec`, `seed_summary`, `holm`) following the order of `ftgrid_report.build`; the refs cache (`load_refs`) must be
used when its identity matches, so the MF and q-hat arrays are the report's. A test asserts that every quantity the report already
computes (E-D's M0-M3 UAUCs, G, G_CF, item-prior share, non-prior share) is reproduced **exactly** by this module on the
synthetic ML-1M world. An integrity failure (E1) of a run excludes that run exactly as the report does.

**Blocks (A3-6 items 1-5)** per (domain, backbone), each with regimes ZS (zeroshot), FT (s0-s2; seed mean with the CI of the mean,
every seed listed, sigma_seed, `complete` flag) and, for FT-C, PERM (p0-p1):
- `E_F`: UAUC of M0-M4 (M4 = [q-hat, MF personal residual, pi, e-hat], cross-fitted exactly as E-D); `G_LLM_given_CF` = dUAUC(M4-M3),
  `G_CF_given_LLM` = dUAUC(M4-M2) with the paired user-bootstrap CI and the two-sided p of `contrast_models`; per-seed values.
- `E_G`: item shares of (i) L with pi (equal to E-D's), (ii) the MF reference with its item bias `mf_item_bias` (MF warm pairs;
  reliability 1), (iii) the label with q-hat (reliability 1), each with the user-bootstrap CI, by the pooled within-user squared
  correlation of `shares_from_sums`; the n of pairs and users.
- `E_H`: strata `sparse`/`dense` (q-hat support `n_prior` < 5, `forensics.prior_means`) and `unseen`/`seen` (item has no TRAIN
  example: the same rule as E-A secondary (ii), from `ftgrid_split.json`'s train file); per stratum: n_users with both classes in the
  stratum (descriptive below 150), UAUC of L, q-hat, pi, `G_prior` = dUAUC(M1-M0) and `G` = dUAUC(M2-M1) from E-D's cross-fitted
  predictors (fit on all rows of the other fold; only the evaluation is restricted). The stratum UAUC averages per-user AUCs over
  users with both classes among their stratum rows.
- `E_J`: dUAUC(L - q-hat) paired on the E-A TEST users/rows per regime (FT: seed mean), CI and p as E-B; dUAUC(L - MF) on MF warm
  pairs (descriptive).
- `FT_C_reading`: retention R (A3-6 item 5) with its user-bootstrap CI (one resample drives all UAUCs) and the label ITEM_DRIVEN /
  USER_DRIVEN / MIXED / NOT_DEFINED; `available: false` with the reason when p0/p1 or the E-B condition are missing.
Raw p-values only; **no Holm adjustment inside a domain file** (addendum 1 item 8). `summarize` computes the three Holm families of
A3-6 (E-F: `G_LLM_given_CF` FT per Qwen Amazon panel; E-H: `G_prior` on sparse rows per panel and regime, panel-regimes below 150
users excluded; E-J: `dUAUC(L - q-hat)` FT per panel) with the sigma_seed rule for FT members (mean > 2 sigma_seed, all seeds the
sign of the mean), writes `confirmed` per member, and never includes ML-1M, Llama or ZS FT-regime members except where the family
lists the regime. Every block that cannot be computed is `{"available": false, "reason": ...}` (never null, never an exception);
JSON is strict (`stats.strict_json`).

Tests (`tests/test_confrec_ftgrid_extra.py`, CPU, no torch): exact reproduction of the report's shared quantities; M4 contains M3
and M2 as nested models (UAUC(M4) is not asserted larger, but the contrasts are computed on identical rows); strata partition the
rows exactly and the sparse rule uses `n_prior` < 5; the minimum-n rule; item shares equal a direct numpy computation on a toy
world; R on a planted world (permuted adapter that keeps all / none of the gain); seed aggregation incomplete -> `complete: false`;
Holm over a planted set of p-values; determinism (two runs byte-identical); no field is NaN.

## X2: `scripts/sigir/run_ftc.sh` and `src/confrec/ftc_panel.py` (FT-C on Toys, A3-6 item 5)
No bound file changes. `ftc_panel.py` builds `outputs/confrec/ftgrid/ftc/toys/train_perm.jsonl` from
`outputs/confrec/ftgrid/panels/toys/train.jsonl` with `ftgrid_data.permute_within_item(rows, 0)` (import, never copy), asserts what
`ftgrid_data` asserts for ML-1M (every item's label sum and rating multiset preserved, nothing else changes), and writes a manifest
(`train_perm.manifest.json`: sha1 of both files, n changed, seed, code sha1) next to it. `run_ftc.sh D` (D = toys only; any other
dataset is refused with exit 2): the registered gates first (`ftgrid_freeze --check --stage amendment` and `--stage core` with the
Toys split; a recorded GATE_FT_PASS; Qwen3-8B only), then the permuted panel, then **training p0, p1 (seeds 0, 1) with exactly the
arguments of the Toys adapters s0-s2** read from `outputs/confrec/ftgrid/adapters/toys/s0/train_config.json` (replace only `--train`,
`--out`, `--seed`), then the `like` pass of p0, p1 on `panels/toys/eval.jsonl` with exactly the arguments of the s0 like pass (read
from its `run.key`/`report.json`; same E1 integrity and rerun-once rule as `run_ftgrid.sh`), then `ftgrid_report` (unchanged) with
`--models zeroshot,s0,s1,s2,p0,p1 --out outputs/confrec/ftgrid/report/toys_ftc.json` (a second file; the registered
`report/toys.json` is never rewritten). Resumable (skip a finished step), `DRY_RUN=1` for a CPU rehearsal on a tiny synthetic world
with stand-ins for the trainer and the scorer (as `run_ftgrid.sh`'s DRY_RUN), exit codes as `run_ftgrid.sh`. Tests
(`tests/test_confrec_ftc.py`): the argument vectors equal the sibling's recorded arguments except the three replaced flags; the
permuted panel preserves label sums and differs from `train.jsonl` only in the moved fields; refusals (not toys, no Gate-FT PASS,
missing s0 adapter, missing freeze record); the DRY_RUN chain end to end including the report call.

## Hand-off
Each owner reports: files written, tests run (counts), sha1 of every new file, anything in A3-6 that could not be implemented as
written (never silently substituted), and the exact command lines for the server runs. X1 also lists the result files the paper
pipeline should pull (`pull_results.ps1` and `fill_paper.py` are updated afterwards by the main session).
