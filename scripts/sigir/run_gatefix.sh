#!/usr/bin/env bash
# Amendment-2 registered remedy: ONE prompt-fix round (idea-stage/PREREG_AMENDMENT_2.md section G). GPU server.
#   usage: bash scripts/sigir/run_gatefix.sh        (PYTHON=/path/python skips conda; MODEL=... overrides Qwen3-8B)
#   stage 0  raw data (slim toys; games/sports for the A1 eligible counts, non-fatal; finished files are skipped)
#            + scripts/sigir/build_confirm_panels.py ->
#            outputs/confrec/gatefix/panels/{ml1m,toys}_{dev,confirm}_h20.jsonl + manifest.json; then print and
#            write outputs/confrec/gatefix/FREEZE.txt (diag_battery freeze): sha1 of the amendment, sha1 of the
#            rendered prompt bank (every prompting.VARIANTS key x first 50 rows of each dev panel x like), sha1 of
#            the diagnosis battery's own prompts and rules (battery bank; run_diag_battery.sh is a GPU job too) and
#            the user-id list sha1s of the manifest. An existing FREEZE.txt that no longer matches is an error.
#            STOPS here unless FREEZE_ACK=1; with FREEZE_ACK=1 it continues only if FREEZE.txt is unchanged and
#            docs/sigir/PILOT_LOG.md already records every REQUIRED sha1 (freeze rule: no GPU job before that).
#   stage 1  G3-G5: score the like question ONLY, for V0,V1,V2,V3,V4,V5,V7 on the ml1m and toys DEV panels
#            (dislike, paraphrase, swap and no-history arms are never scored on dev), yes/no readout, each variant
#            at its registered history window (pyes_scorer default: V3/V7 20, else 10), max_model_len 4096 for every
#            variant (G0); then gatefix_select dev (--manifest: the DEV panels, and the CONFIRM split it records for
#            stage 2, are the ones stage 0 built and froze) -> outputs/confrec/gatefix/dev/selection.json.
#   stage 2  G6, only if gatefix_select dev exits 0 (FIX_FOUND): score V* like and V0 like (reported context only)
#            on the ml1m CONFIRM panel, then gatefix_select confirm -> outputs/confrec/gatefix/confirm/gate.json.
#   Exit code: 0 after the freeze stop; otherwise the last gatefix_select's: dev 4 = F0 (GATE_FAIL_AFTER_REMEDY(dev),
#   Stage 2 skipped), confirm 0 = GATE_PASS, 3 = F1; 2 = input check failed / refused (<out stem>.input_check_failed.json).
# Re-runnable: the panel step reruns only when its inputs or code are newer than manifest.json.done; a scorer run is
# skipped when DIR/report.json exists for the same panel bytes + model + args (DIR/run.key, as in
# run_pilot1_mirror.sh) and moved aside to DIR.stale.<time> when they changed.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONHASHSEED=0
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  set +u
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate lumen
  set -u
  PY=python
fi
case "$(pwd -P)" in
  /root/autodl-tmp/*) ;;
  *) echo "WARNING: repo $(pwd -P) is not under /root/autodl-tmp (data disk); data/ and outputs/ may fill the system disk" >&2 ;;
esac
MODEL="${MODEL:-/root/autodl-tmp/lumen/models/Qwen3-8B}"
G=outputs/confrec/gatefix
GP="$G/panels"
DEV="$G/dev"
CONF="$G/confirm"
FREEZE="$G/FREEZE.txt"
AMEND=idea-stage/PREREG_AMENDMENT_2.md
PILOT_LOG=docs/sigir/PILOT_LOG.md
VARIANTS=(V0 V1 V2 V3 V4 V5 V7)   # G1 bank; V6 dropped
mkdir -p "$GP" "$DEV" "$CONF" data/raw

# ---- helpers (as in run_pilot1_mirror.sh) ----
# fresh OUT DEP...: OUT exists and is newer than every DEP (a missing DEP makes it stale)
fresh() { local o=$1 i; shift; [ -e "$o" ] || return 1; for i in "$@"; do { [ -e "$i" ] && [ "$o" -nt "$i" ]; } || return 1; done; }
# step PRODUCT DEP... -- CMD...: run CMD unless PRODUCT exists and PRODUCT.done is newer than every DEP;
# PRODUCT.done is touched only after CMD succeeds, so an interrupted step always reruns
step() {
  local prod=$1 deps=(); shift
  while [ "$1" != "--" ]; do deps+=("$1"); shift; done; shift
  if [ -e "$prod" ] && fresh "$prod.done" "${deps[@]}"; then echo "[skip] $prod"; return 0; fi
  rm -f "$prod.done"
  "$@" || return 1   # explicit: errexit is off when a caller tests the result
  touch "$prod.done"
}
# score DATA DIR ARGS...: pyes_scorer (fp16, top-50 logprobs, max_model_len 4096, 100-user chunks),
# completion marker DIR/report.json
score() {
  local data=$1 dir=$2 key old=""; shift 2
  key="$(sha1sum "$data" | cut -d' ' -f1) $MODEL $*"
  if [ -f "$dir/run.key" ]; then old=$(cat "$dir/run.key"); fi
  if [ "$old" = "$key" ] && [ -f "$dir/report.json" ]; then echo "[skip] $dir: scored"; return 0; fi
  if [ -e "$dir" ] && [ "$old" != "$key" ]; then  # outputs of another panel/args: keep them aside, start clean
    mv "$dir" "$dir.stale.$(date +%Y%m%d%H%M%S)"; echo "[moved aside] $dir (panel or args changed)"
  fi
  mkdir -p "$dir"
  echo "$key" > "$dir/run.key"
  "$PY" -m src.confrec.pyes_scorer --data "$data" --output "$dir" --model "$MODEL" \
    --dtype float16 --topk_logprobs 50 --max_model_len 4096 --chunk_users 100 "$@"
}
# score_like PANEL DIR VARIANT: the like question only, yes/no readout, the variant's registered history window
# (no --hist_len: gatefix_select requires hist_len == hist_len_registered). G3: the only scoring call of this script.
score_like() {
  score "$1" "$2" --variant "$3" --readout yesno --questions like
}

# ================= stage 0: panels + freeze =================
ML=data/raw/ml-1m
[ -f "$ML/ratings.dat" ] || { echo "ML-1M missing: $ML/ratings.dat (run scripts/sigir/run_pilot1_mirror.sh's data step)" >&2; exit 1; }
"$PY" scripts/sigir/slim_amazon2023.py --domains toys --root data/raw   # skips finished files
# games / sports only feed the A1 eligible counts (build_confirm_panels records a missing raw file, not fatal)
"$PY" scripts/sigir/slim_amazon2023.py --domains games,sports --root data/raw || \
  echo "WARNING: games/sports raw download failed; the manifest will record their eligible counts as missing" >&2
RAW=("$ML/ratings.dat" "$ML/movies.dat" data/raw/amazon_toys/Toys_and_Games.jsonl.gz
     data/raw/amazon_toys/meta_Toys_and_Games.jsonl.gz)
for f in data/raw/amazon_games/Video_Games.jsonl.gz data/raw/amazon_games/meta_Video_Games.jsonl.gz \
         data/raw/amazon_sports/Sports_and_Outdoors.jsonl.gz data/raw/amazon_sports/meta_Sports_and_Outdoors.jsonl.gz; do
  if [ -f "$f" ]; then RAW+=("$f"); fi   # optional: a missing count-only file must not force a rebuild every run
done
PILOT_PANELS=(outputs/confrec/panels/ml1m_rated.jsonl outputs/confrec/panels/toys_rated.jsonl)   # burned = DEV users
for f in "${PILOT_PANELS[@]}"; do
  [ -f "$f" ] || { echo "missing Pilot-1 panel $f (DEV = its users; see run_pilot1_mirror.sh)" >&2; exit 1; }
done
PCODE=(scripts/sigir/build_confirm_panels.py src/confrec/build_rated_panels.py src/confrec/categories.py)
# build_confirm_panels refuses to replace panels / user-id lists that an existing manifest records differently
step "$GP/manifest.json" "${RAW[@]}" "${PILOT_PANELS[@]}" "${PCODE[@]}" -- \
  "$PY" scripts/sigir/build_confirm_panels.py --raw data/raw --ml1m_raw "$ML" --pilot_dir outputs/confrec/panels \
    --out_dir "$GP"
for f in "$GP/ml1m_dev_h20.jsonl" "$GP/toys_dev_h20.jsonl" "$GP/ml1m_confirm_h20.jsonl" "$GP/manifest.json"; do
  [ -f "$f" ] || { echo "build_confirm_panels.py did not produce $f" >&2; exit 1; }
done
FREEZE_ARGS=(--amendment "$AMEND" --dev_panels "$GP/ml1m_dev_h20.jsonl,$GP/toys_dev_h20.jsonl"
             --manifest "$GP/manifest.json" --out "$FREEZE")
echo "== amendment-2 freeze ($FREEZE)"
"$PY" -m src.confrec.diag_battery freeze "${FREEZE_ARGS[@]}"
if [ "${FREEZE_ACK:-0}" != 1 ]; then
  echo "STOPPED at the amendment-2 freeze: record every REQUIRED sha1 above in $PILOT_LOG (sync it here), then rerun"
  echo "with FREEZE_ACK=1. No GPU job of any kind may start before that."
  exit 0
fi
"$PY" -m src.confrec.diag_battery freeze "${FREEZE_ARGS[@]}" --check --pilot_log "$PILOT_LOG"

# ================= stage 1: dev scoring (like only), selection =================
"$PY" -c "import sys, pandas, vllm; print('python', sys.executable, '| pandas', pandas.__version__, '| vllm', vllm.__version__)"
for d in ml1m toys; do
  for v in "${VARIANTS[@]}"; do
    score_like "$GP/${d}_dev_h20.jsonl" "$DEV/$d/$v" "$v"
  done
done
VLIST=$(IFS=,; echo "${VARIANTS[*]}")
set +e
"$PY" -m src.confrec.gatefix_select dev --root "$DEV" --variants "$VLIST" --manifest "$GP/manifest.json" \
  --out "$DEV/selection.json"
rc=$?
set -e
if [ "$rc" != 0 ]; then
  case "$rc" in
    4) echo "gatefix_select dev: F0 = GATE_FAIL_AFTER_REMEDY(dev), no fix found; Stage 2 skipped (G8). See $DEV/selection.json" ;;
    2) echo "gatefix_select dev: input check failed (see $DEV/selection.input_check_failed.json); Stage 2 skipped" >&2 ;;
    *) echo "gatefix_select dev failed with exit code $rc; Stage 2 skipped" >&2 ;;
  esac
  exit "$rc"
fi

# ================= stage 2: one confirmatory gate on the fresh ML-1M users =================
VSTAR=$("$PY" -m src.confrec.diag_battery vstar --selection "$DEV/selection.json")
echo "FIX FOUND on dev: V* = $VSTAR"
CP="$GP/ml1m_confirm_h20.jsonl"
score_like "$CP" "$CONF/ml1m/$VSTAR" "$VSTAR"
score_like "$CP" "$CONF/ml1m/V0" V0   # reported context only (G6)
set +e
"$PY" -m src.confrec.gatefix_select confirm --vstar_dir "$CONF/ml1m/$VSTAR" --v0_dir "$CONF/ml1m/V0" \
  --selection "$DEV/selection.json" --manifest "$GP/manifest.json" --out "$CONF/gate.json"
rc=$?
set -e
case "$rc" in
  2) echo "gatefix_select confirm: input check failed or refused to replace a recorded gate (see" \
       "$CONF/gate.input_check_failed.json); nothing was recorded as F1" >&2 ;;
  *) echo "gatefix_select confirm exit code $rc (see $CONF/gate.json)" ;;
esac
exit "$rc"
