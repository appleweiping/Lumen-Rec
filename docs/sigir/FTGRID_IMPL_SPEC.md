# Implementation contract for the Amendment-3 program (ftgrid)

Binding text: `idea-stage/PREREG_AMENDMENT_3.md` (A3). This file only fixes **interfaces** so that three implementers can work in
parallel; where it and A3 disagree, A3 wins and the disagreement is reported. Nothing here is a pre-registered choice.

## Ownership (one writer per file)
| owner | files |
|---|---|
| F1 data | `src/confrec/ftgrid_data.py`, `tests/test_confrec_ftgrid_data.py` |
| F2 report | `src/confrec/ftgrid_report.py`, `tests/test_confrec_ftgrid_report.py`; the minimal additions to `src/confrec/forensics.py` and `src/confrec/stats.py` that A3 section 3 needs (an explicit `cutoff` for the MF/item-mean references; nothing else may change behaviour of an existing function; keep every existing forensics test green) |
| F3 run | `scripts/sigir/run_ftgrid.sh`, `scripts/sigir/starperm_panel.py` (or reuse of `diag_battery`'s builder), `tests/test_confrec_ftgrid_run.py` |
| done | `src/confrec/ftgrid_freeze.py`, `src/confrec/lora_trainer.py`, `src/confrec/train_lora_yesno.py`, `scripts/sigir/run_gateft.sh` |

Nobody edits another owner's file. Nobody runs git in `D:\Research\Lumen`. No GPU job is started from the local machine; the
server is touched only through `scripts/sigir/remote.ps1` and only for read-only inspection or CPU unit tests, never for a real
scoring or training run. No credential ever appears in a file.

## Directory layout (root `outputs/confrec/ftgrid`; Llama uses `outputs/confrec/ftgrid_llama`, selected by `OUT_ROOT`)
```
panels/<d>/            d in ml1m, toys, games, sports
  train.jsonl                 TRAIN examples: one row per TRAIN user, candidate lists restricted to ts < T (rated-panel schema)
  train_perm.jsonl            ml1m only: within-item label permutation of train.jsonl (A3 section 9), seed 0
  eval.jsonl                  EVAL rows, full candidate lists (CAL and TEST rows together)
  eval_sd_test.jsonl          S_d users, candidate lists restricted to TEST rows (ts >= T)
  eval_sd_test_starperm0.jsonl, eval_sd_test_starperm1.jsonl   star-permuted histories (K = 2) of eval_sd_test.jsonl
  eval_pseudo.jsonl, eval_placebo.jsonl      toys and games only: pseudonymize.py outputs for eval.jsonl
  ftgrid_split.json           deterministic (no wall-clock fields), schema below
adapters/<d>/s<seed>/         LoRA adapters; ml1m s0..s2 are Gate-FT's (recorded by path, never retrained); perm adapters p0, p1
scores/<d>/<model>/<arm>/     pyes_scorer output dirs; model in {zeroshot, s0, s1, s2, p0, p1}
                              arm in {like (eval.jsonl), swap (eval_sd_test.jsonl, --swap_k 8), nohist (eval_sd_test.jsonl, --hist_len 0),
                                      starperm0, starperm1, pseudo, placebo}
report/<d>.json               ftgrid_report output; report/<d>_tables.csv for the paper
```
Every scoring dir carries `run.key` (sha1 of panel bytes + model + adapter + args) and `report.json` (completion marker), as in
`run_gateft.sh`; a finished dir with an equal key is skipped, a different key moves it to `<dir>.stale.<stamp>`.

## `ftgrid_split.json` (F1 writes, F2 and F3 read)
```
{ "domain", "panel_all_sha1", "n_rows", "variant", "quantile": 0.8,
  "T",                                          # pooled 0.8-quantile of TRAIN u EVAL candidate timestamps (numpy linear)
  "train": {"users", "user_ids_sha1", "candidates_total", "examples_pre_T", "cap": 24000, "cap_applied", "examples_written",
            "overlength": {"max_len_registered": 1024, "share_above_1024", "p995", "max_len_used", "micro_bsz", "grad_accum"}},
  "eval":  {"users", "user_ids_sha1", "candidates", "cal_candidates", "test_candidates", "users_both_classes_test",
            "users_both_classes_test_tail", "users_both_classes_all_rows", "share_test_pairs_item_seen_in_train"},
  "sd":    {"users", "user_ids_sha1", "user_ids_path"},             # S_d of A3 section 1
  "files": {name: sha1},                                           # every file ftgrid_data.py writes (the derived panels of
                                                                   # stages 4-5 are deterministic functions of these files and of
                                                                   # code that is itself hashed in the core list)
  "code_sha1": {"ftgrid_data.py": ..., "gateft_data.py": ..., "build_rated_panels.py": ...},
  "args": {"seed", "n_train", "n_eval_max", "s_max", "train_cap", "tokenizer_used", "tokenizer" (basename),
           "dev_users_sha1_checked", "gateft_split_checked"},
  "gateft_T_match": true | null }                                  # ml1m: equals gateft_split.json T (asserted); a split with a
                                                                   # null match (ml1m) or without a tokenizer length audit cannot
                                                                   # be frozen (`ftgrid_freeze`)
```
User-id lists are written one id per line in panel row order next to the json (`train_users.txt`, `eval_users.txt`,
`sd_users.txt`), and their sha1 is over the lines joined by `\n` without a trailing newline.

## Interfaces
- **F1.** `python -m src.confrec.ftgrid_data --domain D --panel_all PATH --out_dir DIR [--variant V0] [--tokenizer PATH]
  [--n_train 1500] [--n_eval_max 3000] [--quantile 0.8] [--s_max 1000] [--train_cap 24000] [--seed 0]
  [--gateft_split PATH] [--ml1m_dev_users_sha1 SHA]`. `main(argv) -> dict` returns the split dict. Reuses `gateft_data`'s
  `split_threshold` / `build_train` (import, do not copy) so that the ML-1M T equals Gate-FT's.
- **F2.** `python -m src.confrec.ftgrid_report --domain D --split DIR/ftgrid_split.json --panels DIR --scores_root SCORES
  --models zeroshot,s0,s1,s2[,p0,p1] --raw data/raw --out PATH [--n_boot 2000] [--seed 0] [--refs PATH]`. `main(argv) -> dict`.
  Reads only files of the layout above. Every endpoint of A3 section 3, P1, the knockout labels (A3 section 5, through
  `pilot_pseudonym.decide`) and the FT-C contrast, each with its n and the minimum-n flag (A3 section 1).
- **F3.** `bash scripts/sigir/run_ftgrid.sh D` (env: `MODEL`, `OUT_ROOT`, `VARIANT`, `STAGES`, `PYTHON`, `DRY_RUN=1`). Stages:
  0 data (needs the h20 panels; builds games/sports with `build_rated_panels --hist_len 20` like `build_confirm_panels.py`
  does for ml1m/toys), 1 train seeds 0..2 (skip when `train_config.json` and adapter weights exist; honours `micro_bsz`,
  `grad_accum`, `max_len_used` of the split json), 2 freeze check (`ftgrid_freeze --check --stage core`), 3 score `like` on
  `eval.jsonl` for zeroshot and each seed, 4 decomposition arms on `eval_sd_test.jsonl` (swap with `--swap_k 8`, nohist with
  `--hist_len 0`, starperm0/1) for every model, 5 knockout arms (toys, games) for every model, 6 report. Stage 3 onward refuses
  to start without stage 2. `DRY_RUN=1` runs the whole chain on a tiny synthetic domain with a fake scorer and a fake trainer
  (no GPU, no model) so the chain is tested end to end on CPU.

## Quality bar (all three)
- Unit tests next to the code (`tests/test_confrec_ftgrid_*.py`), CPU only, deterministic, no network, no GPU; they must pass
  locally with `python -m pytest tests/test_confrec_ftgrid_*.py -q` (Python 3.14, pandas 2.3, numpy; no torch, no
  transformers locally: guard such imports with `pytest.importorskip` and keep them out of module import time).
- Re-read A3 section 3 line by line and implement **that text**, not a convenient variant; where the text is ambiguous or
  contradicts the code it names, stop and write the question into your final report instead of choosing silently.
- No wall-clock, hostname or path inside any hashed json; JSON written with `stats.strict_json` and `allow_nan=False`.
- Final report: files written, tests (count, pass/fail), the exact A3 sentences you found ambiguous, anything you could not do.
