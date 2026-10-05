#!/usr/bin/env bash
# Amendment 3 addendum 6 (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_6.md) items 2-8 on the stored files of ONE domain and
# backbone, and the cross-domain A3-6 Holm families: src/confrec/ftgrid_extra.py (interfaces:
# docs/sigir/FTEXTRA_IMPL_SPEC.md). CPU only; never starts a GPU job and never writes outside OUT_ROOT/extra/.
#   usage: bash scripts/sigir/run_ftextra.sh D           D in ml1m, toys, games, sports
#          bash scripts/sigir/run_ftextra.sh summarize   the A3-6 families over OUT_ROOT/extra/{toys,games,sports}.json
#                                                        (ml1m.json, when present, for the E-B and P1 copies only)
#   env:   OUT_ROOT  default outputs/confrec/ftgrid (Qwen3-8B); Llama: outputs/confrec/ftgrid_llama (ml1m and toys)
#          MODELS    default zeroshot,s0,s1,s2,p0,p1 for ml1m and toys under the Qwen root (the FT-C datasets of A3
#                    section 9 and A3-6 item 8), zeroshot,s0,s1,s2 otherwise; a model that was not scored gives
#                    {"available": false, "reason": ...} blocks, never an error
#          PYTHON    interpreter (skips the conda activation)
#          FORCE=1   rebuild even when the output is newer than every input
#          DRY_RUN=1 CPU rehearsal: any OUT_ROOT except the registered roots, RAW, PILOT_LOG and N_BOOT from the
#                    environment, the freeze check on OUT_ROOT's own split of D only
# Rules. A3 section 0: the bound code must be the recorded code, so `ftgrid_freeze --check --stage core` must pass
# (the four registered splits; under another OUT_ROOT also that root's split of D). A3-6 item 11: the sha1 of
# src/confrec/ftgrid_extra.py and of this script are recorded in docs/sigir/PILOT_LOG.md before the statistics are
# computed for any panel other than ML-1M; a non-ML-1M domain and summarize refuse (exit 4) without that record. ML-1M
# (exploratory under A3-6) runs without it. CPU only: CUDA_VISIBLE_DEVICES="" and nice -n 15.
# Outputs: OUT_ROOT/extra/D.json and D_tables.csv; summarize: OUT_ROOT/extra/summary.json. Resumable: an output newer
# than all of its inputs (split, panels, run reports, code) is kept unless FORCE=1.
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
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  set +u; source /root/miniconda3/etc/profile.d/conda.sh; conda activate lumen; set -u
  PY=python
fi
REG=outputs/confrec/ftgrid             # the registered (Qwen3-8B) root: its four splits are the section-0 record
LLAMA=outputs/confrec/ftgrid_llama
OUT_ROOT="${OUT_ROOT:-$REG}"
OUT_ROOT="${OUT_ROOT%/}"
OUT_ROOT="${OUT_ROOT#./}"
if [ "$DRY_RUN" = 1 ]; then
  if [ "$OUT_ROOT" = "$REG" ] || [ "$OUT_ROOT" = "$LLAMA" ]; then
    echo "DRY_RUN=1 never writes to a registered output root ($OUT_ROOT)" >&2; exit 2
  fi
  RAW="${RAW:-$OUT_ROOT/raw}"
  PILOT_LOG="${PILOT_LOG:?DRY_RUN=1 needs PILOT_LOG (a temporary pilot log)}"
  N_BOOT="${N_BOOT:-200}"
else
  if [ "$OUT_ROOT" != "$REG" ] && [ "$OUT_ROOT" != "$LLAMA" ]; then
    echo "OUT_ROOT must be $REG or $LLAMA (a registered root); use DRY_RUN=1 for a rehearsal" >&2; exit 2
  fi
  RAW=data/raw
  PILOT_LOG=docs/sigir/PILOT_LOG.md
  N_BOOT=2000                          # A3 section 3: 2,000 user resamples, seed 0
fi
if [ "$OUT_ROOT" = "$LLAMA" ] && { [ "$D" = games ] || [ "$D" = sports ]; }; then
  echo "the Llama program runs on ML-1M and Toys only (A3 sections 4 and 8)" >&2; exit 2
fi
NICE=()
if command -v nice > /dev/null 2>&1; then NICE=(nice -n 15); fi
EXTRA_CODE=(src/confrec/ftgrid_extra.py scripts/sigir/run_ftextra.sh)
XDIR="$OUT_ROOT/extra"

# freeze_core: the bound code and the splits are the recorded ones (A3 section 0)
freeze_core() {
  local splits=() f
  if [ "$DRY_RUN" = 1 ]; then
    if [ "$1" = summarize ]; then
      for f in "$OUT_ROOT"/panels/*/ftgrid_split.json; do
        if [ -f "$f" ]; then splits+=(--split "$f"); fi
      done
      [ "${#splits[@]}" -gt 0 ] || { echo "DRY_RUN summarize: no split under $OUT_ROOT/panels" >&2; exit 2; }
    else
      splits=(--split "$OUT_ROOT/panels/$1/ftgrid_split.json")
    fi
  else
    splits=(--split "$REG/panels/ml1m/ftgrid_split.json" --split "$REG/panels/toys/ftgrid_split.json"
            --split "$REG/panels/games/ftgrid_split.json" --split "$REG/panels/sports/ftgrid_split.json")
    if [ "$OUT_ROOT" != "$REG" ] && [ "$1" != summarize ]; then splits+=(--split "$OUT_ROOT/panels/$1/ftgrid_split.json"); fi
  fi
  if ! "$PY" -m src.confrec.ftgrid_freeze --check --stage core --pilot_log "$PILOT_LOG" "${splits[@]}"; then
    echo "refused: the bound code or a split is not the recorded one (A3 section 0; ftgrid_freeze --check --stage core)" >&2
    exit 4
  fi
}
# record_extra: A3-6 item 11, the sha1 of this file and of ftgrid_extra.py are in the pilot log
record_extra() {
  local f sha missing=()
  for f in "${EXTRA_CODE[@]}"; do
    [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
    sha=$("$PY" -c 'import hashlib, sys; print(hashlib.sha1(open(sys.argv[1], "rb").read()).hexdigest())' "$f" | tr -d '\r')
    if ! grep -qi "$sha" "$PILOT_LOG"; then missing+=("$f = $sha"); fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "refused (A3-6 item 11): record these sha1 in $PILOT_LOG before computing the A3-6 statistics of a panel other" \
      "than ML-1M: ${missing[*]}" >&2
    exit 4
  fi
}
# fresh OUT DEP...: OUT exists and is newer than every DEP
fresh() { local o=$1 i; shift; [ -e "$o" ] || return 1; for i in "$@"; do { [ -e "$i" ] && [ "$o" -nt "$i" ]; } || return 1; done; }

if [ "$D" = summarize ]; then
  freeze_core summarize
  record_extra
  FILES=()
  for d in toys games sports; do
    if [ -f "$XDIR/$d.json" ]; then
      FILES+=("$XDIR/$d.json")
    elif [ -f "$OUT_ROOT/report/$d.json" ]; then
      echo "refused: $d has a registered report but no $XDIR/$d.json (run 'bash scripts/sigir/run_ftextra.sh $d' first):" \
        "a family member cannot be left out" >&2
      exit 2
    fi
  done
  [ "${#FILES[@]}" -gt 0 ] || { echo "no domain file in $XDIR" >&2; exit 2; }
  ML=()
  if [ -f "$XDIR/ml1m.json" ]; then
    ML=(--ml1m "$XDIR/ml1m.json")
  elif [ -f "$OUT_ROOT/report/ml1m.json" ]; then
    echo "NOTE: no $XDIR/ml1m.json: the E-B family copy and P1 are left out of the summary" >&2
  fi
  ${NICE[@]+"${NICE[@]}"} "$PY" -m src.confrec.ftgrid_extra summarize --files "${FILES[@]}" ${ML[@]+"${ML[@]}"} \
    --out "$XDIR/summary.json"
  echo "run_ftextra summarize: done ($XDIR/summary.json)"
  exit 0
fi

P="$OUT_ROOT/panels/$D"
S="$OUT_ROOT/scores/$D"
SPLIT="$P/ftgrid_split.json"
OUT="$XDIR/$D.json"
[ -f "$SPLIT" ] || { echo "missing $SPLIT (run_ftgrid.sh stage 0)" >&2; exit 2; }
[ -d "$S" ] || { echo "missing $S (no scoring run of $D under $OUT_ROOT)" >&2; exit 2; }
freeze_core "$D"
if [ "$D" != ml1m ]; then record_extra; fi
if [ -z "${MODELS:-}" ]; then
  MODELS=zeroshot,s0,s1,s2
  if [ "$OUT_ROOT" != "$LLAMA" ] && { [ "$D" = ml1m ] || [ "$D" = toys ]; }; then MODELS=zeroshot,s0,s1,s2,p0,p1; fi
fi
DEPS=("$SPLIT" "$P/eval.jsonl" "${EXTRA_CODE[@]}" src/confrec/ftgrid_report.py)
for f in "$S"/*/like/report.json "$S"/*/swap/report.json "$S"/*/starperm0/report.json "$S"/*/starperm1/report.json; do
  if [ -f "$f" ]; then DEPS+=("$f"); fi
done
if [ "${FORCE:-0}" != 1 ] && fresh "$OUT" "${DEPS[@]}"; then
  echo "[skip] $OUT is newer than its inputs (FORCE=1 rebuilds)"
  exit 0
fi
mkdir -p "$XDIR"
echo "run_ftextra $D: root $OUT_ROOT, models $MODELS, n_boot $N_BOOT (CPU)"
${NICE[@]+"${NICE[@]}"} "$PY" -m src.confrec.ftgrid_extra build --domain "$D" --split "$SPLIT" --panels "$P" \
  --scores_root "$OUT_ROOT/scores" --models "$MODELS" --raw "$RAW" --out "$OUT" --n_boot "$N_BOOT" --seed 0
echo "run_ftextra $D: done ($OUT)"
