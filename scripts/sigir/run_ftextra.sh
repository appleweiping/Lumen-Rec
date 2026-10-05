#!/usr/bin/env bash
# Amendment 3 addendum 6 (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_6.md) items 2-8 on the stored files of ONE domain and
# backbone, and the cross-domain A3-6 Holm families: src/confrec/ftgrid_extra.py (interfaces:
# docs/sigir/FTEXTRA_IMPL_SPEC.md; addendum 8 for FT-Q). CPU only; never starts a GPU job and writes only OUT_ROOT/extra/.
#   usage: bash scripts/sigir/run_ftextra.sh D           D in ml1m, toys, games, sports
#          bash scripts/sigir/run_ftextra.sh summarize   main root only: the A3-6 families over extra/{toys,games,sports}.json,
#                                                        with ml1m.json (E-B / P1 copies, FT-C condition), the Llama root's
#                                                        and the teacher root's extra files
#   env:   OUT_ROOT  outputs/confrec/ftgrid (default; root label main, Qwen3-8B), outputs/confrec/ftgrid_llama (llama;
#                    ml1m and toys) or outputs/confrec/ftgrid_q (teacher: the FT-Q control of addendum 8); any other
#                    value only with DRY_RUN=1. Spellings are canonicalised (realpath -m: symlinks, //, .., absolute).
#          MODELS    default: main ml1m zeroshot,s0,s1,s2,p0,p1 (FT-C); teacher zeroshot,s0,s1,s2,p0,p1 (p0, p1 = the
#                    teacher adapters); otherwise zeroshot,s0,s1,s2. A model that was not scored gives
#                    {"available": false} blocks, never an error (summarize accepts only the default models)
#          PYTHON    interpreter (skips the conda activation);  FORCE=1  rebuild even when the stored build is current
#          DRY_RUN=1 CPU rehearsal: OUT_ROOT, its panels and scores of D and RAW must lie outside outputs/confrec/ftgrid*
#                    in every spelling; RAW, PILOT_LOG, N_BOOT and ROOT_LABEL then come from the environment; the freeze
#                    check uses OUT_ROOT's own split(s)
# Rules. A3 section 0: the bound code must be the recorded code: `ftgrid_freeze --check --stage core` must pass (the four
# registered splits; under another root also that root's split of D when it has one). A3-6 item 11: the sha1 of the
# three X1 files (src/confrec/ftgrid_extra.py, this script, tests/test_confrec_ftgrid_extra.py) must be in the pilot log
# before the statistics of any panel other than ML-1M are computed: a non-ML-1M domain and summarize refuse (exit 4)
# without the record. ML-1M runs without it, but its FT_C_reading block (registered, outcome-free) is then
# {"available": false, "reason": "record missing"} (ftgrid_extra checks the record itself). CPU only:
# CUDA_VISIBLE_DEVICES="" and nice -n 15.
# Resume: an existing OUT_ROOT/extra/D.json is kept only when `ftgrid_extra check_resume` finds that its stored
# fingerprint equals the current one (n_boot, seed 0, the requested models, the root, the code sha1, the split, every panel
# file, every file of every scoring run directory read, the raw files, the A3-6 record) and its CSV is the one written
# with it; anything else is rebuilt; FORCE=1 always rebuilds. A build removes the old JSON, writes the CSV, then the JSON
# (the completion marker), each through a temporary file and a rename.
# summarize: every input file must be a current build with the default arguments of its root and domain (check_resume;
# a stale file is refused with exit 2 and named); a dataset whose fine-tuned runs were cut is passed as --not_run D when
# the pilot log holds a line with the token 'FTEXTRA_NOT_RUN D' (ftgrid_extra checks the token and the run record).
# Exit codes: 0 done; 1 error; 2 usage or refused input; 4 refused by the freeze / record rules.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
D="${1:-}"
case "$D" in
  ml1m|toys|games|sports|summarize) ;;
  *) echo "usage: bash scripts/sigir/run_ftextra.sh {ml1m|toys|games|sports|summarize}" >&2; exit 2 ;;
esac
DRY_RUN="${DRY_RUN:-0}"
REG=outputs/confrec/ftgrid             # the registered (Qwen3-8B) root: its four splits are the section-0 record
LLAMA=outputs/confrec/ftgrid_llama
TEACHER=outputs/confrec/ftgrid_q       # addendum 8: the FT-Q teacher control
command -v realpath > /dev/null 2>&1 || { echo "realpath (coreutils) is required" >&2; exit 1; }
FOLD=0                                 # MSYS / Cygwin (Git Bash): drive-letter paths and a case-insensitive filesystem
case "$(uname -s 2> /dev/null)" in MINGW*|MSYS*|CYGWIN*) FOLD=1 ;; esac
# canon PATH: the canonical absolute path (realpath -m: symlinks, //, . and .. resolved; on Windows also D:/... spellings
# and letter case)
canon() {
  local p=$1 c
  if [ "$FOLD" = 1 ] && command -v cygpath > /dev/null 2>&1; then p=$(cygpath -u -- "$p"); fi
  c=$(realpath -m -- "$p")
  if [ "$FOLD" = 1 ]; then c=$(printf '%s' "$c" | tr '[:upper:]' '[:lower:]'); fi
  printf '%s\n' "$c"
}
CONF=$(canon outputs/confrec)
# under_registered PATH: 0 iff PATH, in any spelling and with symlinks resolved, is at or under outputs/confrec/ftgrid*
under_registered() {
  case "$(canon "$1")" in "$CONF"/ftgrid*) return 0 ;; esac
  return 1
}
# refuse_registered PATH...: DRY_RUN never reads or writes a path at or under outputs/confrec/ftgrid*
refuse_registered() {
  local p
  for p in "$@"; do
    if under_registered "$p"; then
      echo "DRY_RUN=1 never touches a path at or under outputs/confrec/ftgrid* ($p resolves to $(canon "$p"))" >&2
      exit 2
    fi
  done
}
OUT_ROOT="${OUT_ROOT:-$REG}"
if [ "$DRY_RUN" = 1 ]; then
  refuse_registered "$OUT_ROOT" "$OUT_ROOT/extra" "$OUT_ROOT/extra/$D.json" "$OUT_ROOT/extra/${D}_tables.csv" \
    "$OUT_ROOT/extra/summary.json" "$OUT_ROOT/panels" "$OUT_ROOT/scores" "$OUT_ROOT/panels/$D" "$OUT_ROOT/scores/$D"
  RAW="${RAW:-$OUT_ROOT/raw}"
  refuse_registered "$RAW"
  PILOT_LOG="${PILOT_LOG:?DRY_RUN=1 needs PILOT_LOG (a temporary pilot log)}"
  N_BOOT="${N_BOOT:-200}"
  ROOT_LABEL="${ROOT_LABEL:-main}"
  case "$ROOT_LABEL" in main|llama|teacher) ;; *) echo "ROOT_LABEL must be main, llama or teacher" >&2; exit 2 ;; esac
else
  case "$(canon "$OUT_ROOT")" in
    "$(canon "$REG")") OUT_ROOT=$REG; ROOT_LABEL=main ;;
    "$(canon "$LLAMA")") OUT_ROOT=$LLAMA; ROOT_LABEL=llama ;;
    "$(canon "$TEACHER")") OUT_ROOT=$TEACHER; ROOT_LABEL=teacher ;;
    *) echo "OUT_ROOT=$OUT_ROOT is not a registered root ($REG, $LLAMA or $TEACHER); use DRY_RUN=1 for a rehearsal" >&2
       exit 2 ;;
  esac
  RAW=data/raw
  PILOT_LOG=docs/sigir/PILOT_LOG.md
  N_BOOT=2000                          # A3 section 3: 2,000 user resamples, seed 0
fi
if [ "$ROOT_LABEL" = llama ] && { [ "$D" = games ] || [ "$D" = sports ]; }; then
  echo "the Llama program runs on ML-1M and Toys only (A3 sections 4 and 8)" >&2; exit 2
fi
if [ "$D" = summarize ] && [ "$ROOT_LABEL" != main ]; then
  echo "summarize runs on the main root (it reads the Llama and teacher roots itself)" >&2; exit 2
fi
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  set +u; source /root/miniconda3/etc/profile.d/conda.sh; conda activate lumen; set -u
  PY=python
fi
NICE=()
if command -v nice > /dev/null 2>&1; then NICE=(nice -n 15); fi
EXTRA_CODE=(src/confrec/ftgrid_extra.py scripts/sigir/run_ftextra.sh tests/test_confrec_ftgrid_extra.py)
XDIR="$OUT_ROOT/extra"

# default_models LABEL D: the registered models of a build
default_models() {
  if { [ "$1" = main ] && [ "$2" = ml1m ]; } || [ "$1" = teacher ]; then echo zeroshot,s0,s1,s2,p0,p1
  else echo zeroshot,s0,s1,s2; fi
}
# panels_of LABEL ROOT D: the root's panels of D; a teacher root without its own panels uses the main root's (its
# adapters are scored on the same EVAL panel)
panels_of() {
  if [ "$1" = teacher ] && [ "$DRY_RUN" != 1 ] && [ ! -f "$2/panels/$3/ftgrid_split.json" ]; then
    echo "$REG/panels/$3"
  else
    echo "$2/panels/$3"
  fi
}
# set_args LABEL ROOT D MODELS: ARGS = the build (and check_resume) arguments of domain D on that root
set_args() {
  local p
  p=$(panels_of "$1" "$2" "$3")
  ARGS=(--domain "$3" --split "$p/ftgrid_split.json" --panels "$p" --scores_root "$2/scores" --models "$4"
        --raw "$RAW" --out "$2/extra/$3.json" --n_boot "$N_BOOT" --seed 0 --root_label "$1" --pilot_log "$PILOT_LOG")
}
# current LABEL ROOT D: summarize's inputs must be current builds with the default arguments (exit 2 names a stale one)
current() {
  local why
  set_args "$1" "$2" "$3" "$(default_models "$1" "$3")"
  if ! why=$("$PY" -m src.confrec.ftgrid_extra check_resume "${ARGS[@]}" 2>&1); then
    echo "refused: $2/extra/$3.json is not a current build with the default arguments of its root ($why); rebuild it" \
      "with 'OUT_ROOT=$2 bash scripts/sigir/run_ftextra.sh $3'" >&2
    exit 2
  fi
}
# freeze_core D|summarize: the bound code and the splits are the recorded ones (A3 section 0)
freeze_core() {
  local splits=() f
  if [ "$DRY_RUN" = 1 ]; then
    for f in "$OUT_ROOT"/panels/*/ftgrid_split.json; do
      if [ -f "$f" ]; then splits+=(--split "$f"); fi
    done
    [ "${#splits[@]}" -gt 0 ] || { echo "DRY_RUN: no split under $OUT_ROOT/panels" >&2; exit 2; }
  else
    splits=(--split "$REG/panels/ml1m/ftgrid_split.json" --split "$REG/panels/toys/ftgrid_split.json"
            --split "$REG/panels/games/ftgrid_split.json" --split "$REG/panels/sports/ftgrid_split.json")
    if [ "$1" != summarize ] && [ "$OUT_ROOT" != "$REG" ] && [ -f "$OUT_ROOT/panels/$1/ftgrid_split.json" ]; then
      splits+=(--split "$OUT_ROOT/panels/$1/ftgrid_split.json")
    fi
  fi
  if ! "$PY" -m src.confrec.ftgrid_freeze --check --stage core --pilot_log "$PILOT_LOG" "${splits[@]}"; then
    echo "refused: the bound code or a split is not the recorded one (A3 section 0; ftgrid_freeze --check --stage core)" >&2
    exit 4
  fi
}
# record_extra: A3-6 item 11, the sha1 of the three X1 files are in the pilot log
record_extra() {
  local f sha missing=()
  for f in "${EXTRA_CODE[@]}"; do
    [ -f "$f" ] || { echo "missing $f (A3-6 item 11 records it)" >&2; exit 4; }
    sha=$("$PY" -c 'import hashlib, sys; print(hashlib.sha1(open(sys.argv[1], "rb").read()).hexdigest())' "$f" | tr -d '\r')
    if ! grep -qi "$sha" "$PILOT_LOG"; then missing+=("$f = $sha"); fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "refused (A3-6 item 11): record these sha1 in $PILOT_LOG before computing the A3-6 statistics of a panel other" \
      "than ML-1M: ${missing[*]}" >&2
    exit 4
  fi
}

if [ "$D" = summarize ]; then
  freeze_core summarize
  record_extra
  FILES=()
  NR=()
  for d in toys games sports; do
    if [ -f "$XDIR/$d.json" ]; then
      current main "$OUT_ROOT" "$d"
      FILES+=("$XDIR/$d.json")
      if grep -qE "(^|[^A-Za-z0-9_-])FTEXTRA_NOT_RUN $d([^A-Za-z0-9_-]|\$)" "$PILOT_LOG"; then NR+=(--not_run "$d"); fi
    elif [ -f "$OUT_ROOT/report/$d.json" ]; then
      echo "refused: $d has a registered report but no $XDIR/$d.json (run 'bash scripts/sigir/run_ftextra.sh $d' first):" \
        "a family member cannot be left out" >&2
      exit 2
    fi
  done
  [ "${#FILES[@]}" -gt 0 ] || { echo "no domain file in $XDIR" >&2; exit 2; }
  ML=()
  if [ -f "$XDIR/ml1m.json" ]; then
    current main "$OUT_ROOT" ml1m
    ML=(--ml1m "$XDIR/ml1m.json" --ftc "$XDIR/ml1m.json")
  else
    echo "NOTE: no $XDIR/ml1m.json: the E-B family copy stays incomplete (its p_holm and confirmed are null), P1 is not" \
      "available and the FT-C condition of the FT wording is missing (fine_tuning_mostly_teaches_the_item null)" >&2
  fi
  LL=()
  FQ=()
  if [ "$DRY_RUN" != 1 ]; then
    for d in ml1m toys; do
      if [ -f "$LLAMA/extra/$d.json" ]; then
        current llama "$LLAMA" "$d"
        LL+=("$LLAMA/extra/$d.json")
      elif [ -f "$LLAMA/report/$d.json" ]; then
        echo "refused: the Llama root has a report of $d but no $LLAMA/extra/$d.json" >&2; exit 2
      fi
    done
    for d in ml1m toys games sports; do
      if [ -f "$TEACHER/extra/$d.json" ]; then
        current teacher "$TEACHER" "$d"
        FQ+=("$TEACHER/extra/$d.json")
      elif [ -f "$TEACHER/scores/$d/p0/like/report.json" ] || [ -f "$TEACHER/scores/$d/p1/like/report.json" ]; then
        echo "refused: FT-Q $d was scored but has no $TEACHER/extra/$d.json (the FT wording never reads partial input)" >&2
        exit 2
      fi
    done
  fi
  SARGS=(--files "${FILES[@]}" ${ML[@]+"${ML[@]}"} ${NR[@]+"${NR[@]}"} --pilot_log "$PILOT_LOG" --n_boot "$N_BOOT"
         --out "$XDIR/summary.json")
  if [ "${#LL[@]}" -gt 0 ]; then SARGS+=(--llama "${LL[@]}"); fi
  if [ "${#FQ[@]}" -gt 0 ]; then SARGS+=(--ftq "${FQ[@]}"); fi
  ${NICE[@]+"${NICE[@]}"} "$PY" -m src.confrec.ftgrid_extra summarize "${SARGS[@]}"
  echo "run_ftextra summarize: done ($XDIR/summary.json)"
  exit 0
fi

P=$(panels_of "$ROOT_LABEL" "$OUT_ROOT" "$D")
S="$OUT_ROOT/scores/$D"
OUT="$XDIR/$D.json"
[ -f "$P/ftgrid_split.json" ] || { echo "missing $P/ftgrid_split.json (run_ftgrid.sh stage 0)" >&2; exit 2; }
[ -d "$S" ] || { echo "missing $S (no scoring run of $D under $OUT_ROOT)" >&2; exit 2; }
freeze_core "$D"
if [ "$D" != ml1m ]; then record_extra; fi
MODELS="${MODELS:-$(default_models "$ROOT_LABEL" "$D")}"
set_args "$ROOT_LABEL" "$OUT_ROOT" "$D" "$MODELS"
if [ "${FORCE:-0}" != 1 ] && "$PY" -m src.confrec.ftgrid_extra check_resume "${ARGS[@]}"; then
  echo "[skip] $OUT is a complete build of the current inputs, settings and code (FORCE=1 rebuilds)"
  exit 0
fi
mkdir -p "$XDIR"
echo "run_ftextra $D: root $OUT_ROOT ($ROOT_LABEL), models $MODELS, n_boot $N_BOOT (CPU)"
${NICE[@]+"${NICE[@]}"} "$PY" -m src.confrec.ftgrid_extra build "${ARGS[@]}"
echo "run_ftextra $D: done ($OUT)"
