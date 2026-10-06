#!/usr/bin/env bash
# Train/test-shift diagnostic of the method slot (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_10.md section 2) for ONE dataset: for each
# prior-offset adapter o0-o2, on the first 1,000 rows (in panel order) of the dataset's EVAL CAL candidates, the per-id top-50
# log-probabilities of the next token under the `like` question as stage 4 of run_ftmethod.sh scores it, the share s of the yes
# mass on the answer token 'Yes' (derived from the tokenizer: 9454 for Qwen3), the registered score R, the training-consistent score
# C = R + log(s + (1 - s) exp(-b z)), and UAUC of R and C (src/confrec/ftmethod_shift_diag.py). Outcome-free and descriptive; it must be
# recorded BEFORE the dataset's slot report is built (run_ftmethod.sh stage 5 refuses without it). A NEW script (no bound file
# changes). GPU server; one job at a time through scripts/sigir/gpu_queue.sh, one job per dataset, after the dataset's stage 3
# (the adapters) and before its stage 5:
#     cd /root/autodl-tmp/lumen-rec && bash scripts/sigir/run_ftmethod_shift_diag.sh ml1m        (then toys, games, sports as the slot reaches them)
#   usage: bash scripts/sigir/run_ftmethod_shift_diag.sh D       D in ml1m, toys, games, sports (the only argument)
#   env:   MODEL     Qwen3-8B dir (default /root/autodl-tmp/lumen/models/Qwen3-8B); any other backbone is refused (the slot is single-backbone)
#          OUT_ROOT  the slot root outputs/confrec/ftmethod, the only root outside DRY_RUN; compared canonically as in run_ftmethod.sh
#                    (realpath -m, lower case, links refused): any spelling of that root is accepted, anything else refused with exit 2
#          VARIANT   default gate_ft_prompt of outputs/confrec/gatefix/dev/selection.json; any other value is refused
#          STAGES    comma list of 1-2 or all (default all): 1 = score (GPU), 2 = report and verify (CPU)
#          PYTHON    interpreter (skips the conda activation)
#          DRY_RUN   0 or 1 (any other value is refused with exit 2 and starts nothing); 1 = a CPU rehearsal with a stand-in model
#          DRY_TODAY (DRY_RUN=1 only) YYYYMMDD, eight digits: the stand-in for the clock (hard kill date rehearsal)
#          DRY_NO_RECORD (DRY_RUN=1 only) 0 or 1: 1 does not do the human step of recording the diagnostic's files on the temporary log
#          DRY_SHIFT_SHARE, DRY_SHIFT_LOW_FRAC (DRY_RUN=1 only) the stand-in's s on a normal row (default 0.9995) and the fraction of rows
#                    with s ~ U(0.5, 0.9) (default 0): they steer the reading of a rehearsal
#   reads (never written): the grid root's panels outputs/confrec/ftgrid/panels/D (eval.jsonl, ftgrid_split.json) and the slot's own
#          files outputs/confrec/ftmethod/D/{adapters/o0-o2, qhat_manifest.json, eval_qhat.csv.gz} (run_ftmethod.sh D, stages 1-3)
#   writes ONLY below OUT_ROOT/D/shift_diag/: o0-o2/{top50.jsonl.gz, meta.json} (stage 1), report.json and report_tables.csv (stage 2),
#          and in a rehearsal _dry/. Covered like run_ftmethod.sh: (1) every path this script names below the slot root, with the temporary
#          names the module writes beside its products (<name>.tmp), must resolve, links included, to the same place below the canonical
#          root, else the run is refused (exit 2) before it writes anything; (2) a sweep for links (find OUT_ROOT -type l) at the start and
#          before each stage allows none; (3) stale .tmp files below shift_diag/ are removed before a stage writes
#   gates    (before stages 1-2, in this order; any failure exits 4 and starts nothing): the registered order and kill rule over the
#            datasets' reports, cuts and the date (`ftmethod_report slot --check_next D`: D is next, not killed, not NOT_RUN); the recorded
#            Gate-FT decision is GATE_FT_PASS (Amendment 3 section 0); the freeze record: `ftgrid_freeze --check --stage amendment`, `--stage
#            core` and `--stage method --split D/qhat_manifest.json` (the adapters scored here are the slot's, section 0); the DIAGNOSTIC'S
#            OWN RECORD: the sha1 of this script, of src/confrec/ftmethod_shift_diag.py and of tests/test_confrec_ftmethod_shift_diag.py are
#            in docs/sigir/PILOT_LOG.md (`ftmethod_shift_diag record`); the hard kill date 2026-11-30 before every seed of stage 1 (stage
#            2 is CPU); and stage 1 is refused once D's slot report exists (the diagnostic is recorded before it, never after). Imported or
#            used as they are, and not part of that record: pyes_scorer.py (load_vllm, score_ids, record_requests, check_lora_variant),
#            prompting.py, ftgrid_report.py, forensics.py, ftmethod_report.py (load_eval_qhat, load_offset: the checks and the constants
#            of the report) and train_lora_offset.py
#   stage 1  (GPU) one process per adapter o0, o1, o2: `ftmethod_shift_diag score`, the scorer's loading and extraction by import (float16,
#            max_model_len 4096, top-50 logprobs, seed 0, vLLM LoRA of the adapter), the selected rows' prompts rendered by
#            pyes_scorer.record_requests exactly as stage 4 renders them. A finished scoring directory with the same run key is skipped
#   stage 2  (CPU) `ftmethod_shift_diag report` (D/shift_diag/report.json, report_tables.csv) and `verify` (the report on disk is the
#            recomputation from the files, OK, of the right mode): what run_ftmethod.sh stage 5 requires
# GPU time: about 12-15 min per dataset (the addendum's estimate is about 15): three vLLM engine starts of the 8B model with a LoRA (the
# bulk) plus 1,000 prompts per adapter at the measured rates of the like passes (ML-1M about 90 prompts/s, an Amazon domain about 55
# prompts/s: 11-18 s per adapter); the first 1,000 CAL rows are about 60 users.
# DRY_RUN=1: the same chain, CPU only, on run_ftmethod.sh's own DRY_RUN world and slot root: tmp_outputs/ftmethod_dryrun/{ftgrid, ftmethod}
# (built by `DRY_RUN=1 STAGES=1,2,3,4 bash scripts/sigir/run_ftmethod.sh D`; this script builds nothing of it and refuses where it is
# missing). The DRY_RUN guard is the allow-list of run_ftmethod.sh: an OUT_ROOT and the world must, in any spelling and through any
# link, lie under tmp_outputs of the repo or outside the repo's parent directory (never in the repo, beside it or above it), an OUT_ROOT
# that is, or lies inside, the world is refused, and so is a world with a link leading anywhere else; every comparison is made on
# lower-case canonical forms. The model and the tokenizer are the module's CPU stand-in (several yes ids, s steered by DRY_SHIFT_*); the
# report is marked dry_run and never satisfies a real run's gate. The rehearsal first shows that an empty pilot log fails the record
# check; the human step of recording the diagnostic's files is done on the temporary log unless DRY_NO_RECORD=1 (the record is then
# missing and the run is refused).
# Exit codes: 0 done; 1 error or an inconsistent input; 2 usage or refused input; 4 refused by the order / kill / gate / freeze / date rules.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
D="${1:-}"
case "$D" in
  ml1m|toys|games|sports) ;;
  *) echo "usage: bash scripts/sigir/run_ftmethod_shift_diag.sh {ml1m|toys|games|sports}" >&2; exit 2 ;;
esac
# the switches are 0 or 1 and nothing else: DRY_RUN=yes must not start a real job, nor DRY_RUN= a rehearsal
DRY_RUN="${DRY_RUN-0}"
DRY_NO_RECORD="${DRY_NO_RECORD-0}"
for sw in DRY_RUN DRY_NO_RECORD; do
  case "${!sw}" in 0|1) ;; *) echo "$sw must be 0 or 1 (it is '${!sw}'): nothing was started" >&2; exit 2 ;; esac
done
# the stand-ins of a rehearsal exist only there: set with a real run they would mislead, so they are refused
if [ "$DRY_RUN" != 1 ]; then
  for sw in DRY_TODAY DRY_SHIFT_SHARE DRY_SHIFT_LOW_FRAC; do
    if [ -n "${!sw+x}" ]; then
      echo "$sw is a stand-in of a DRY_RUN=1 rehearsal: a real run does not take it (unset it): nothing was started" >&2; exit 2
    fi
  done
  if [ "$DRY_NO_RECORD" = 1 ]; then
    echo "DRY_NO_RECORD=1 belongs to a DRY_RUN=1 rehearsal: nothing was started" >&2; exit 2
  fi
fi
if [ -n "${DRY_TODAY+x}" ]; then
  case "$DRY_TODAY" in
    [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]) ;;
    *) echo "DRY_TODAY must be eight digits, YYYYMMDD (it is '$DRY_TODAY'): nothing was started" >&2; exit 2 ;;
  esac
fi
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  set +u; source /root/miniconda3/etc/profile.d/conda.sh; conda activate lumen; set -u
  PY=python
fi
REG=outputs/confrec/ftmethod          # the registered slot root
REGG=outputs/confrec/ftgrid           # the registered grid root: read, never written
DRYB=tmp_outputs/ftmethod_dryrun      # run_ftmethod.sh's DRY_RUN world and slot root: a temporary directory outside outputs/confrec
command -v realpath > /dev/null 2>&1 || { echo "realpath (coreutils) is required" >&2; exit 1; }
FOLD=0                                # MSYS / Cygwin (Git Bash): D:/... and D:\... spellings go through cygpath
case "${OSTYPE:-}" in msys*|cygwin*) FOLD=1 ;; esac
# canon_all PATH...: the array CANON gets the canonical absolute path of every argument (run_ftmethod.sh's function, verbatim)
canon_all() {
  local -a a=("$@")
  if [ "$FOLD" = 1 ] && command -v cygpath > /dev/null 2>&1; then mapfile -t a < <(cygpath -u -- "$@"); fi
  mapfile -t CANON < <(realpath -m -- "${a[@]}")
  [ "${#CANON[@]}" = "$#" ] || return 1
  CANON=("${CANON[@],,}")
}
if [ "$DRY_RUN" = 1 ]; then
  MODEL="${MODEL:-dryrun/Qwen3-8B}"
  GRID=$DRYB/ftgrid                   # run_ftgrid.sh's DRY_RUN root: the synthetic world the script reads
  OUT_ROOT="${OUT_ROOT:-$DRYB/ftmethod}"
else
  MODEL="${MODEL:-/root/autodl-tmp/lumen/models/Qwen3-8B}"
  GRID="$REGG"
  OUT_ROOT="${OUT_ROOT:-$REG}"
fi
while [ "${OUT_ROOT%/}" != "$OUT_ROOT" ]; do OUT_ROOT="${OUT_ROOT%/}"; done      # ftmethod// -> ftmethod
if [ -z "$OUT_ROOT" ]; then echo "OUT_ROOT is empty or the filesystem root: the slot has one registered root, $REG" >&2; exit 2; fi
# Everything this script writes below the slot root (all below D/shift_diag), as a path below it, with the temporary names the module
# writes beside its products. No link is allowed anywhere below the root: sweep_links
TREE=("$D" "$D/shift_diag" "$D/shift_diag/o0" "$D/shift_diag/o1" "$D/shift_diag/o2"
      "$D/shift_diag/o0/top50.jsonl.gz" "$D/shift_diag/o0/top50.jsonl.gz.tmp" "$D/shift_diag/o0/meta.json" "$D/shift_diag/o0/meta.json.tmp"
      "$D/shift_diag/o1/top50.jsonl.gz" "$D/shift_diag/o1/top50.jsonl.gz.tmp" "$D/shift_diag/o1/meta.json" "$D/shift_diag/o1/meta.json.tmp"
      "$D/shift_diag/o2/top50.jsonl.gz" "$D/shift_diag/o2/top50.jsonl.gz.tmp" "$D/shift_diag/o2/meta.json" "$D/shift_diag/o2/meta.json.tmp"
      "$D/shift_diag/report.json" "$D/shift_diag/report.json.tmp" "$D/shift_diag/report_tables.csv"
      "$D/shift_diag/report_tables.csv.tmp" "$D/shift_diag/_dry")
if [ "$DRY_RUN" = 1 ]; then SPELL=$OUT_ROOT; else SPELL=$REG; fi          # the spelling the files are written with
ARGS=(. .. "$OUT_ROOT" "$GRID")
for r in "${TREE[@]}"; do ARGS+=("$SPELL/$r"); done
canon_all "${ARGS[@]}" || { echo "cannot resolve OUT_ROOT=$OUT_ROOT or the paths below it" >&2; exit 2; }
ROOTC=${CANON[0]}; PARENTC=${CANON[1]}; OUTC=${CANON[2]}; GRIDC=${CANON[3]}
under() { case "$1/" in "${2%/}/"*) return 0 ;; esac; return 1; }
# dry_ok PATH: a rehearsal root (canonical) is under the repo's tmp_outputs, or outside the repo's parent directory: never in the repo, never
# beside it and never above it
dry_ok() {
  if under "$1" "$ROOTC/tmp_outputs"; then return 0; fi
  if under "$1" "$PARENTC" || under "$PARENTC" "$1"; then return 1; fi
  return 0
}
if [ "$DRY_RUN" = 1 ]; then
  for p in "$OUTC" "$GRIDC"; do
    if ! dry_ok "$p"; then
      echo "DRY_RUN=1 never writes to a registered output root, nor anywhere in the repo or beside it ($p): a rehearsal root is under" \
        "$ROOTC/tmp_outputs (default $DRYB) or outside $PARENTC; a relative OUT_ROOT resolves from the repo root $ROOTC" >&2
      exit 2
    fi
  done
  if under "$OUTC" "$GRIDC"; then
    echo "DRY_RUN=1: OUT_ROOT=$OUT_ROOT resolves to $OUTC, the synthetic world it reads or a directory inside it" >&2; exit 2
  fi
elif [ "$OUTC" != "$ROOTC/$REG" ]; then
  echo "OUT_ROOT=$OUT_ROOT resolves to $OUTC, not to $ROOTC/$REG: the slot has one registered root, $REG (single backbone), reached" \
    "through no link (the grid root $REGG is read, never written); use DRY_RUN=1 for a rehearsal" >&2
  exit 2
else
  OUT_ROOT=$REG
fi
# nothing written below the slot root may leave it through a link: every path of the tree resolves, links included, to the same place below
# the canonical root
for i in "${!TREE[@]}"; do
  expect="$OUTC/${TREE[$i]}"
  expect="${expect,,}"
  if [ "${CANON[$((i + 4))]}" != "$expect" ]; then
    echo "shift diagnostic refused: $SPELL/${TREE[$i]} resolves to ${CANON[$((i + 4))]}, not to $expect: a link between the slot root and" \
      "the files this script writes would redirect them (nothing was written)" >&2
    exit 2
  fi
done
# DRY_RUN lets run_ftgrid.sh's own DRY_RUN write the synthetic world: a link inside it must stay inside the rehearsal's own places
if [ "$DRY_RUN" = 1 ] && [ -e "$GRID" ]; then
  mapfile -t WLINKS < <(find "$GRID" -type l 2> /dev/null)
  if [ "${#WLINKS[@]}" -gt 0 ]; then
    canon_all "${WLINKS[@]}" || { echo "cannot resolve the links inside the synthetic world" >&2; exit 2; }
    for p in "${CANON[@]}"; do
      if ! dry_ok "$p"; then
        echo "DRY_RUN=1 never writes to a registered output root: a link inside the synthetic world leads to $p, in the repo or beside" \
          "it, outside $ROOTC/tmp_outputs" >&2
        exit 2
      fi
    done
  fi
fi
# sweep_links: no link below the slot root (a tool would write through it: a planted <name>.tmp, a scoring directory that is a link, ...)
sweep_links() {
  [ -d "$OUT_ROOT" ] || return 0
  local -a found=()
  mapfile -t found < <(find "$OUT_ROOT" -type l 2> /dev/null)
  if [ "${#found[@]}" -gt 0 ]; then
    echo "shift diagnostic refused: ${found[0]} is a link: nothing below $OUT_ROOT is a link, and a tool would write through it" \
      "(nothing was written)" >&2
    exit 2
  fi
}
sweep_links
if [ "$(basename "$MODEL")" != Qwen3-8B ]; then
  echo "$MODEL: the method slot is single-backbone, Qwen3-8B only (section 7; a Llama replication is not registered)" >&2
  exit 2
fi
STAGES=" $(printf '%s' "${STAGES:-all}" | tr ',' ' ') "
for s in $STAGES; do
  case "$s" in 1|2|all) ;; *) echo "unknown stage '$s' in STAGES (1-2, all)" >&2; exit 2 ;; esac
done
case "$STAGES" in *" all "*) STAGES="$STAGES 1 2 " ;; esac
want() { case "$STAGES" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

P="$GRID/panels/$D"                  # run_ftgrid.sh's panels of D (read)
SPLIT="$P/ftgrid_split.json"
M="$OUT_ROOT/$D"                     # this dataset's slot directory (read, but for shift_diag/)
SD="$M/shift_diag"                   # everything this script writes
QM="$M/qhat_manifest.json"
REP="$M/report.json"                 # the dataset's slot report (never written here; its existence closes the diagnostic)
HARD_KILL=20261130                   # section 7 hard kill date
if [ "$DRY_RUN" = 1 ]; then
  G="$GRID/_dry/gatefix"; GT="$GRID/_dry/gateft"; PILOT_LOG="$GRID/_dry/PILOT_LOG.md"
  CORE_ARGS=(--split "$SPLIT")
  DRY_FLAGS="--dry_run --dry_share ${DRY_SHIFT_SHARE-0.9995} --dry_low_frac ${DRY_SHIFT_LOW_FRAC-0}"
  DRY_FLAG="--dry_run"               # report and verify take the mode only
  ALLOW_LOG="--allow_missing_log"    # the temporary pilot log of a rehearsal may not exist yet
else
  G=outputs/confrec/gatefix; GT=outputs/confrec/gateft; PILOT_LOG=docs/sigir/PILOT_LOG.md
  CORE_ARGS=(--split "$GRID/panels/ml1m/ftgrid_split.json" --split "$GRID/panels/toys/ftgrid_split.json"
    --split "$GRID/panels/games/ftgrid_split.json" --split "$GRID/panels/sports/ftgrid_split.json")
  DRY_FLAGS=""
  DRY_FLAG=""
  ALLOW_LOG=""
fi
METHOD_ARGS=(--split "$QM")          # addendum 2 item 2: the dataset's q-hat manifest is part of FREEZE method
DIAG_FILES=(scripts/sigir/run_ftmethod_shift_diag.sh src/confrec/ftmethod_shift_diag.py tests/test_confrec_ftmethod_shift_diag.py)

# ---- helpers ----
# jget FILE KEY...: a (nested) value of a json file, '' when absent
jget() {
  "$PY" -c 'import json, sys
v = json.load(open(sys.argv[1], encoding="utf-8"))
for k in sys.argv[2:]:
    v = v.get(k) if isinstance(v, dict) else None
print("" if v is None else v)' "$@" | tr -d '\r'
}
# clean_tmp: a stale <name>.tmp below shift_diag/ is removed before a stage writes (a hard link planted under a temporary name would
# otherwise be written through: the module writes <name>.tmp and then renames it)
clean_tmp() {
  if [ -d "$SD" ]; then find "$SD" -name '*.tmp' -type f -delete 2> /dev/null || true; fi
}
# clock: today's date, YYYYMMDD, read afresh at every call (run_ftmethod.sh's function; DRY_TODAY stands in for it in a rehearsal)
clock() {
  local t
  if [ "$DRY_RUN" = 1 ] && [ -n "${DRY_TODAY+x}" ]; then t=$DRY_TODAY; else t=$(date +%Y%m%d); fi
  case "$t" in
    [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]) printf '%s\n' "$t" ;;
    *) echo "the clock reads '$t', not YYYYMMDD: the hard kill date 2026-11-30 cannot be checked, refused" >&2; return 1 ;;
  esac
}
# date_check N: section 7 hard kill date 2026-11-30, before every seed of stage 1
date_check() {
  local t
  t=$(clock) || return 1
  if [ "$t" -gt "$HARD_KILL" ]; then
    echo "stage $1 refused: the slot's hard kill date 2026-11-30 has passed (it is $t; section 7; section 10: anything unfinished" \
      "is reported as not run)" >&2
    return 1
  fi
}
# gate_ft_pass N: section 0, section 7 runs only after a recorded GATE_FT_PASS
gate_ft_pass() {
  local dec=missing
  if [ -f "$GT/gate_ft.json" ]; then dec=$(jget "$GT/gate_ft.json" decision); fi
  if [ "$dec" != GATE_FT_PASS ]; then
    echo "stage $1 refused: Gate-FT decision $dec (section 0: section 7 runs only after a recorded GATE_FT_PASS," \
      "$GT/gate_ft.json)" >&2
    return 1
  fi
}
# freeze_check STAGE [LOG]: 0 iff the pilot log records every sha1 the freeze stage requires (section 0; addendum 2)
freeze_check() {
  local extra=()
  case "$1" in core) extra=("${CORE_ARGS[@]}") ;; method) extra=("${METHOD_ARGS[@]}") ;; esac
  "$PY" -m src.confrec.ftgrid_freeze --check --stage "$1" --pilot_log "${2:-$PILOT_LOG}" "${extra[@]}"
}
# the diagnostic's own record (the human step: the sha1 of its three files in the pilot log)
check_record() { "$PY" -m src.confrec.ftmethod_shift_diag record --pilot_log "${1:-$PILOT_LOG}" --files "${DIAG_FILES[@]}"; }
# ensure_record: the record is in the pilot log (exit 4 otherwise). DRY_RUN does the human step on the temporary log first (--append writes
# the missing lines), unless DRY_NO_RECORD=1
ensure_record() {
  if [ "$DRY_RUN" = 1 ] && [ "$DRY_NO_RECORD" != 1 ]; then
    "$PY" -m src.confrec.ftmethod_shift_diag record --pilot_log "$PILOT_LOG" --files "${DIAG_FILES[@]}" --append || exit 4
  else
    check_record || { echo "shift diagnostic refused: its record (the sha1 of ${DIAG_FILES[*]}) is not in $PILOT_LOG" >&2; exit 4; }
  fi
}
# rehearse: DRY_RUN only, once per world: an empty pilot log fails the diagnostic's record check
rehearse() {
  local empty="$SD/_dry/PILOT_LOG.empty.md"
  mkdir -p "$SD/_dry"
  [ -f "$empty" ] || printf '# empty temporary pilot log\n' > "$empty"
  if check_record "$empty" > /dev/null 2>&1; then
    echo "DRY_RUN: the diagnostic's record check passed on an empty pilot log" >&2; exit 1
  fi
  echo "[dry] record rehearsal: an empty pilot log fails the diagnostic's record check"
}
# slot_check: the registered order, the kill rule, the cuts and the date over the datasets' reports (exit 4 = refused)
slot_check() {
  local rc=0 t
  t=$(clock) || exit 4
  "$PY" -m src.confrec.ftmethod_report slot --root "$OUT_ROOT" --pilot_log "$PILOT_LOG" $ALLOW_LOG --today "$t" \
    --check_next "$D" || rc=$?
  if [ "$rc" = 4 ]; then return 4; fi
  if [ "$rc" != 0 ]; then echo "the slot check failed (exit $rc)" >&2; exit 1; fi
}
# gates: everything the registered text asks before the diagnostic runs (exit 4 on any failure, nothing started)
gates() {
  slot_check || exit 4
  gate_ft_pass "$1" || exit 4
  if [ "$DRY_RUN" = 1 ] && [ ! -f "$SPLIT" ]; then
    echo "missing $SPLIT: the rehearsal's world is run_ftmethod.sh's (DRY_RUN=1 STAGES=1,2,3,4 bash scripts/sigir/run_ftmethod.sh $D)" >&2
    exit 1
  fi
  [ -f "$QM" ] || { echo "missing $QM (run_ftmethod.sh stage 1)" >&2; exit 1; }
  for st in amendment core method; do
    freeze_check "$st" || { echo "shift diagnostic refused: the record of freeze stage $st is not in $PILOT_LOG (run_ftmethod.sh" \
      "stage 2 checks and records it)" >&2; exit 4; }
  done
  if [ "$DRY_RUN" = 1 ]; then rehearse; fi
  ensure_record
}

# ---- the gates come first: a refused dataset touches nothing ----
GATED=0
for s in 1 2; do if want "$s"; then GATED=1; fi; done
if [ "$GATED" = 1 ]; then gates "$(if want 1; then echo 1; else echo 2; fi)"; fi

# ---- the prompt ----
SEL="$G/dev/selection.json"
[ -f "$SEL" ] || { echo "missing $SEL (the gate-fix stage 1 selection; run_gatefix.sh writes it)" >&2; exit 1; }
SEL_PROMPT=$(jget "$SEL" gate_ft_prompt)
VARIANT="${VARIANT:-$SEL_PROMPT}"
if [ "$VARIANT" != "$SEL_PROMPT" ] && [ "$DRY_RUN" != 1 ]; then
  echo "VARIANT=$VARIANT is not selection.json's gate_ft_prompt $SEL_PROMPT: section 2 registers no other variant" >&2
  exit 2
fi
if [ -f "$SPLIT" ] && [ "$(jget "$SPLIT" variant)" != "$VARIANT" ]; then
  echo "$SPLIT was built under variant $(jget "$SPLIT" variant), not $VARIANT" >&2; exit 2
fi
echo "run_ftmethod_shift_diag $D: model $MODEL, root $OUT_ROOT, variant $VARIANT, stages:$STAGES"

# ================= stage 1: the top-50 log-probabilities of o0, o1, o2 (GPU) =================
if want 1; then
  sweep_links
  echo "== stage 1: top-50 log-probabilities of the prior-offset adapters on the first 1,000 CAL rows ($D)"
  if [ -f "$REP" ]; then
    echo "stage 1 refused: the slot report $REP of $D exists: the diagnostic is recorded before it (addendum 10 section 2), never after" >&2
    exit 4
  fi
  for f in "$P/eval.jsonl" "$SPLIT" "$M/eval_qhat.csv.gz"; do
    [ -f "$f" ] || { echo "missing $f (run_ftgrid.sh stage 0; run_ftmethod.sh stage 1)" >&2; exit 1; }
  done
  for seed in 0 1 2; do
    if ! { [ -f "$M/adapters/o$seed/train_config.json" ] && [ -f "$M/adapters/o$seed/offset.json" ] \
        && ls "$M/adapters/o$seed"/adapter_model.* > /dev/null 2>&1; }; then
      echo "adapter o$seed of $D is missing or incomplete: $M/adapters/o$seed (run_ftmethod.sh stage 3)" >&2; exit 1
    fi
  done
  mkdir -p "$SD"
  clean_tmp
  for seed in 0 1 2; do
    date_check 1 || exit 4             # before EVERY seed (addendum 10 item 4)
    "$PY" -m src.confrec.ftmethod_shift_diag score --domain "$D" --seed "$seed" --split "$SPLIT" --panels "$P" --method_dir "$M" \
      --model "$MODEL" --variant "$VARIANT" --lora_arg "$M/adapters/o$seed" $DRY_FLAGS
  done
fi

# ================= stage 2: the report and its verification (CPU) =================
if want 2; then
  sweep_links
  echo "== stage 2: report and verification ($D)"
  mkdir -p "$SD"
  clean_tmp
  "$PY" -m src.confrec.ftmethod_shift_diag report --domain "$D" --split "$SPLIT" --panels "$P" --method_dir "$M" $DRY_FLAG
  "$PY" -m src.confrec.ftmethod_shift_diag verify --domain "$D" --split "$SPLIT" --panels "$P" --method_dir "$M" $DRY_FLAG
fi
echo "run_ftmethod_shift_diag $D: done (stages:$STAGES)"
