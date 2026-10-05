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
`starperm_panel.py`, ... ; run `python -m src.confrec.ftgrid_freeze --print --stage core` to list them with their sha1). New code
imports them read-only. Nobody runs git in `D:\Research\Lumen`. No GPU job is started from the local machine; the server is touched
only through `scripts/sigir/remote.ps1`, for read-only inspection or CPU unit tests (`CUDA_VISIBLE_DEVICES=""`, `nice -n 15`),
never for a registered run. **Do not open or print any Toys, Video_Games or Sports fine-tuned result**
(`outputs/confrec/ftgrid/report/{toys,games,sports}.json`, `scores/{toys,games,sports}/s*/`, any knockout or FT-C output): A3-6 is
registered outcome-free for them. ML-1M files may be read (A3-6 labels ML-1M exploratory): locally
`docs/sigir/results/grid/qwen/ml1m.json`, on the server `outputs/confrec/ftgrid/{panels,scores,report}/ml1m*`. Develop against the
synthetic worlds of `tests/test_confrec_ftgrid_report.py` and the real ML-1M files. No credential appears in any file.
A previous read-only experiment (scratch code, not registered) showed the shape of the within-user estimator: it monkeypatched
`ftgrid_report.crossfit` with a wrapper that user-centres every feature over the user's rows before fitting; the registered pooled values
for ML-1M are G(ZS) -0.0018, G(FT mean) +0.0096, G_CF +0.0362 and the user-centred single-split values were +0.0001, +0.0092, +0.0378
(use these numbers only as an order-of-magnitude sanity check, never as test oracles: E-W uses K = 20 splits).

## X1: `src/confrec/ftgrid_extra.py`
CPU only. A pure function of stored files, deterministic (seed 0, 2,000 user resamples, no clock or host in the output).

CLI (mirrors `ftgrid_report`): `python -m src.confrec.ftgrid_extra build --domain D --split PATH --panels DIR --scores_root DIR
--models zeroshot,s0,s1,s2[,p0,p1] --raw data/raw --out FILE.json [--refs CACHE] [--n_boot 2000] [--seed 0]` and
`python -m src.confrec.ftgrid_extra summarize --files extra/toys.json extra/games.json extra/sports.json [--ml1m extra/ml1m.json]
--out extra/summary.json`. Output root `outputs/confrec/ftgrid/extra/` (Llama: `outputs/confrec/ftgrid_llama/extra/`); per-domain
`D.json` and `D_tables.csv` (long format: block, panel, regime, statistic, estimate, lo, hi, n_users, descriptive flag).

**Reuse, do not re-derive.** Load rows, scores, contexts and references through `ftgrid_report`'s own functions (`make_ctx`,
`load_run`, `to_pairs`, `compute_refs`/`load_refs`, `Clusters`, `uauc_models`, `contrast_models`, `logit_fit`, `logit_eta`,
`shares_from_sums`/`within_user_sums`, `rec`, `seed_summary`, `holm`, `mean_draws`, `user_aucs`) following the order of
`ftgrid_report.build`; use the refs cache when its identity matches, so q-hat, the MF arrays and `mf_item_bias` are the report's (if a
needed array, for example `mf_item_bias` or q-hat_T, is not in the cache, compute it with the same `forensics` function and arguments as
the report and say so in the output's `meta`). A test asserts that every quantity the report already computes (E-D's pooled M0-M3
UAUCs, G, G_CF, item-prior share, non-prior share, P1's inputs) is reproduced **exactly** by this module's `registered_pooled` block on the
synthetic ML-1M world, so the paper can print the registered and the within-user values from one file. An integrity failure (E1) of a run
excludes that run exactly as the report does.

**Blocks (A3-6 items 2-8)** per (domain, backbone); regimes ZS (zeroshot) and FT (s0-s2: seed mean with the CI of the mean, every seed
listed, sigma_seed, `complete` flag); every block that cannot be computed is `{"available": false, "reason": ...}` (never null, never an
exception); JSON is strict (`stats.strict_json`):
- `registered_pooled`: the registered E-D numbers re-derived through `ftgrid_report.stacker_block` and `p1_block` (the equality test).
- `E_W` (A3-6 item 2): user-centred features (each feature minus the user's mean over the user's rows in the evaluation set R), K = 20
  splits `stats.user_halves(S_d, k)` (k = 0 is E-D's split), per user the UAUC difference averaged over splits, then the user bootstrap
  with fixed coefficients: UAUC of M0-M3, `G_wu` = dUAUC(M2-M1), `G_CF_wu` = dUAUC(M3-M0), per model and seed mean; `P1_wu` computed
  like `p1_block` on the G_wu values (same conditions: 3 seeds, not descriptive, mean > 0, p < 0.05, all seeds positive, mean > 2
  sigma_seed) wherever `p1_block` is defined; plus the `robust` flag of the reading rule (same sign and CI excluding 0 for both
  estimators; for FT also all seeds the same sign).
- `E_F` (item 3): M4 = [q-hat, MF personal residual, pi, e-hat] with E-W's estimator; `G_LLM_given_CF` = dUAUC(M4-M3),
  `G_CF_given_LLM` = dUAUC(M4-M2): paired user-bootstrap CI, two-sided p as `contrast_models`, per-seed values.
- `E_G` (item 4): item shares of (i) L with pi (equal to E-D's), (ii) the MF reference with its item bias `mf_item_bias` (MF warm pairs;
  reliability 1), (iii) the label with q-hat (reliability 1), by the pooled within-user squared correlation of `shares_from_sums`; and
  the **e-share** = [Var(L_c - pi_c) - (1 - r8c) Var(pi_c)] / Var(L_c) with r8c re-estimated in every resample; user-bootstrap CIs,
  n of pairs and users.
- `E_H` (item 5): strata `sparse`/`dense` (q-hat support `n_prior` < 5, `forensics.prior_means`) and `unseen`/`seen` (item has no TRAIN
  example: the rule of E-A secondary (ii), items of `train.jsonl`); per stratum: n_users with both classes in the stratum (descriptive
  below 150), UAUC of L, q-hat and pi, `G_prior` = dUAUC(M1-M0) and `G_wu` = dUAUC(M2-M1) from E-W's cross-fitted predictors (fit on all
  rows of the other fold; only the evaluation is restricted). The stratum UAUC averages per-user AUCs over users with both classes
  among their stratum rows.
- `E_J` (item 6): dUAUC(L - q-hat), dUAUC(L - q-hat_T) (q-hat_T: `forensics.prior_means` with every candidate time replaced by T_d, same
  shrinkage k = 5) on the E-A TEST users/rows, per regime (FT: seed mean), CI and p as E-B; dUAUC(L - MF) on MF warm pairs (descriptive).
- `E_Cprime` (item 7): E-C's top-k anatomy statistics (`topk_anatomy`, `anatomy_stats`) with k_u from the user's CAL like rate (fallback
  rule of A3-6), beside the oracle values, descriptive.
- `FT_C_reading` (item 8): retention R with its user-bootstrap CI (one resample drives all UAUCs) and the label ITEM_DRIVEN /
  USER_DRIVEN / MIXED / NOT_DEFINED; `available: false` with the reason when p0/p1 or the E-B condition are missing.
Raw p-values only in a domain file; **no Holm adjustment inside a domain file** (addendum 1 item 8). `summarize` computes the three Holm
families of A3-6 (E-F: `G_LLM_given_CF` FT per Qwen Amazon panel; E-H: `G_prior` on sparse rows per panel and regime, panel-regimes below
150 users excluded; E-J: dUAUC(L - q-hat) FT per panel) with the sigma_seed rule for FT members (mean > 2 sigma_seed and all seeds the sign
of the mean), writes `confirmed` per member, includes only the members the family lists (never ML-1M, Llama, or the other regime), and
copies the registered E-B/E-D family results through unchanged so the paper pipeline reads one file per question.

Tests (`tests/test_confrec_ftgrid_extra.py`, CPU, no torch): exact reproduction of the report's registered quantities; E-W removes a
planted user-offset bias (a world in which user offsets track leniency and there is no personal signal: pooled G < 0, E-W G_wu close to 0)
and keeps a planted personal signal; M4 / M3 / M2 contrasts are on identical rows; strata partition the rows exactly and the sparse rule
uses `n_prior` < 5; the minimum-n rule; item shares equal a direct numpy computation on a toy world and the e-share recovers a planted
Var(e)/Var(L) (including a world with Cov(pi, e) != 0 in which the non-prior share is below it); R on a planted world (a permuted adapter
that keeps all / none of the gain); seed aggregation incomplete -> `complete: false`; Holm and the sigma_seed rule on planted p-values;
determinism (two runs byte-identical); no NaN in the JSON.

## X2: `scripts/sigir/run_ftc.sh` and `src/confrec/ftc_panel.py` (FT-C on Toys, A3-6 item 8)
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
pipeline should pull (`pull_results.ps1` and `fill_paper.py` are updated afterwards by the main session) and runs `build` on the
real ML-1M inputs on the server (CPU, output to `outputs/confrec/ftgrid/extra/ml1m.json`, labelled exploratory) as the end-to-end check.
