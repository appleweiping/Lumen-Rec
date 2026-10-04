#!/usr/bin/env bash
# Amendment-2 diagnosis battery (idea-stage/PREREG_AMENDMENT_2.md section D) on BURNED Pilot-1 users. Interpretive
# only: no result of this script can change G0-G9. GPU server; separate output root outputs/confrec/diag/.
#   usage: bash scripts/sigir/run_diag_battery.sh
#     PYTHON=/path/python skips conda; MODEL=... (Qwen3-8B), LLAMA=... (T4 backbone) override the model dirs;
#     PANEL_ML / PANEL_TOYS override the burned panels (default: the Pilot-1 panels, checked against their sha1;
#     ALLOW_PANEL_MISMATCH=1 accepts other bytes); SKIP_T4=1 skips the Llama arm; T1_OVERRIDE=1 continues past a
#     broken T1 readout (only after the debugging is recorded in docs/sigir/PILOT_LOG.md).
#   prerequisite: the amendment-2 freeze (run_gatefix.sh stage 0, sha1s recorded in docs/sigir/PILOT_LOG.md, incl.
#   the battery bank sha1); this script re-checks it (diag_battery freeze --check) before any GPU job.
#   arms (pyes_scorer, fp16, top-50 logprobs, max_model_len 4096; Qwen3-8B unless noted), in this order:
#     T1        V0 like + "(This user later rated this item r/5.)"; scored FIRST, then diag_battery t1check:
#               T1 UAUC < 0.90 = "readout broken; stop and debug" -> exit 5 before any other arm is scored
#     base      V0 like yes/no on the burned ML-1M panel (paired reference) and on the first 500 burned Toys users
#     T0        --variant T0_probe --questions like on one row per unique ML-1M panel item (no user; P(Yes) is read);
#               the same probe with --readout digits (E[r]) is a secondary quantity
#     T2        V0 like + prior-only leave-user-out item mean; ML-1M and the first 500 burned Toys users
#     T3        V0 like, --readout digits (E[r] over the tokens 1-5)
#     starperm  V0 like on K=2 copies with the history "(rated r/5)" suffixes permuted among the same items (each copy
#               a uniformly random derangement; the copies differ)
#     T4        Llama-3.1-8B-Instruct, V0 like
#   then diag_battery analyze -> outputs/confrec/diag/diag.json (every reading labelled interpretive-only).
#   Exit code: 0 done; 5 T1 readout broken (stop and debug); 2 T1 check could not run
#   (see outputs/confrec/diag/ml1m/t1check.json).
# Re-runnable with the skip rules of run_pilot1_mirror.sh (battery panels: <panel>.done; scorer runs: DIR/run.key).
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
LLAMA="${LLAMA:-/root/autodl-tmp/lumen/models/Llama-3.1-8B-Instruct}"
D=outputs/confrec/diag
DP="$D/panels"
PML="${PANEL_ML:-outputs/confrec/panels/ml1m_rated.jsonl}"
PTOYS="${PANEL_TOYS:-outputs/confrec/panels/toys_rated.jsonl}"
ML_SHA1=985494c7b44d010bec4b11ec62f35ad274dfe91f     # Pilot-1 ML-1M panel (report.json data_sha1)
TOYS_SHA1=69252b4806bb4ffe05101967ea2a1d94ada57dee   # Pilot-1 Toys panel
N_TOYS=500
ML=data/raw/ml-1m
TOYS_RAW=data/raw/amazon_toys/Toys_and_Games.jsonl.gz
G=outputs/confrec/gatefix
mkdir -p "$D" "$DP"

# ---- freeze rule: no GPU job of any kind before the amendment-2 hashes are in PILOT_LOG ----
[ -f "$G/FREEZE.txt" ] || { echo "no $G/FREEZE.txt: run scripts/sigir/run_gatefix.sh (stage 0) and record the freeze first" >&2; exit 1; }
"$PY" -m src.confrec.diag_battery freeze --amendment idea-stage/PREREG_AMENDMENT_2.md \
  --dev_panels "$G/panels/ml1m_dev_h20.jsonl,$G/panels/toys_dev_h20.jsonl" --manifest "$G/panels/manifest.json" \
  --out "$G/FREEZE.txt" --check --pilot_log docs/sigir/PILOT_LOG.md

# ---- inputs ----
# check_sha1 FILE SHA1: the battery runs on the burned Pilot-1 panels, byte for byte
check_sha1() {
  local got
  got=$(sha1sum "$1" | cut -d' ' -f1)
  [ "$got" = "$2" ] && return 0
  if [ "${ALLOW_PANEL_MISMATCH:-0}" = 1 ]; then
    echo "WARNING: $1 sha1 $got is not the burned Pilot-1 panel $2 (ALLOW_PANEL_MISMATCH=1)" >&2; return 0
  fi
  echo "$1 sha1 $got is not the burned Pilot-1 panel $2 (set PANEL_ML/PANEL_TOYS or ALLOW_PANEL_MISMATCH=1)" >&2
  return 1
}
for f in "$PML" "$PTOYS" "$ML/ratings.dat" "$TOYS_RAW"; do
  [ -f "$f" ] || { echo "missing $f (run scripts/sigir/run_pilot1_mirror.sh's data/panel steps)" >&2; exit 1; }
done
check_sha1 "$PML" "$ML_SHA1"
check_sha1 "$PTOYS" "$TOYS_SHA1"
if [ "${SKIP_T4:-0}" != 1 ] && [ ! -f "$LLAMA/config.json" ]; then
  echo "T4 needs Llama-3.1-8B-Instruct at $LLAMA (download it there, or set LLAMA=..., or SKIP_T4=1)" >&2
  exit 1
fi
"$PY" -c "import sys, pandas, vllm; print('python', sys.executable, '| pandas', pandas.__version__, '| vllm', vllm.__version__)"

# ---- helpers (as in run_pilot1_mirror.sh; score takes the model) ----
fresh() { local o=$1 i; shift; [ -e "$o" ] || return 1; for i in "$@"; do { [ -e "$i" ] && [ "$o" -nt "$i" ]; } || return 1; done; }
step() {
  local prod=$1 deps=(); shift
  while [ "$1" != "--" ]; do deps+=("$1"); shift; done; shift
  if [ -e "$prod" ] && fresh "$prod.done" "${deps[@]}"; then echo "[skip] $prod"; return 0; fi
  rm -f "$prod.done"
  "$@" || return 1
  touch "$prod.done"
}
# score MODEL_DIR DATA DIR ARGS...: pyes_scorer, completion marker DIR/report.json, key = panel sha1 + model + args
score() {
  local model=$1 data=$2 dir=$3 key old=""; shift 3
  key="$(sha1sum "$data" | cut -d' ' -f1) $model $*"
  if [ -f "$dir/run.key" ]; then old=$(cat "$dir/run.key"); fi
  if [ "$old" = "$key" ] && [ -f "$dir/report.json" ]; then echo "[skip] $dir: scored"; return 0; fi
  if [ -e "$dir" ] && [ "$old" != "$key" ]; then
    mv "$dir" "$dir.stale.$(date +%Y%m%d%H%M%S)"; echo "[moved aside] $dir (panel or args changed)"
  fi
  mkdir -p "$dir"
  echo "$key" > "$dir/run.key"
  "$PY" -m src.confrec.pyes_scorer --data "$data" --output "$dir" --model "$model" \
    --dtype float16 --topk_logprobs 50 --max_model_len 4096 --chunk_users 100 "$@"
}
DCODE=(src/confrec/diag_battery.py src/confrec/categories.py src/confrec/stats.py)
MK=("$PY" -m src.confrec.diag_battery make)

# ---- battery panels ----
step "$DP/ml1m_t0.jsonl" "$PML" "$ML/ratings.dat" "${DCODE[@]}" -- \
  "${MK[@]}" --kind t0 --panel "$PML" --raw "$ML" --source ml1m --out "$DP/ml1m_t0.jsonl"
step "$DP/ml1m_t1.jsonl" "$PML" "${DCODE[@]}" -- \
  "${MK[@]}" --kind t1 --panel "$PML" --source ml1m --out "$DP/ml1m_t1.jsonl"
step "$DP/ml1m_t2.jsonl" "$PML" "$ML/ratings.dat" "${DCODE[@]}" -- \
  "${MK[@]}" --kind t2 --panel "$PML" --raw "$ML" --source ml1m --out "$DP/ml1m_t2.jsonl"
step "$DP/ml1m_starperm.jsonl" "$PML" "${DCODE[@]}" -- \
  "${MK[@]}" --kind starperm --panel "$PML" --source ml1m --k 2 --seed 0 --hist_len 10 --out "$DP/ml1m_starperm.jsonl"
step "$DP/toys_t2.jsonl" "$PTOYS" "$TOYS_RAW" "${DCODE[@]}" -- \
  "${MK[@]}" --kind t2 --panel "$PTOYS" --raw data/raw --source amazon --domain toys --n_users "$N_TOYS" \
    --out "$DP/toys_t2.jsonl"

# ---- scoring (each variant at its registered window: V0 the last 10 history events, T0_probe none) ----
V0=(--variant V0 --readout yesno --questions like)
# T1 first: the registered "T1 UAUC < 0.90: readout broken; stop and debug" is applied before any other arm
score "$MODEL" "$DP/ml1m_t1.jsonl" "$D/ml1m/t1" "${V0[@]}"
set +e
"$PY" -m src.confrec.diag_battery t1check --root "$D" --out "$D/ml1m/t1check.json"
rc=$?
set -e
case "$rc" in
  0) ;;
  5) if [ "${T1_OVERRIDE:-0}" = 1 ]; then
       echo "WARNING: T1 readout broken (UAUC < 0.90) but T1_OVERRIDE=1: continuing (record the debugging in docs/sigir/PILOT_LOG.md)" >&2
     else
       echo "T1 UAUC < 0.90: readout broken; stop and debug (amendment 2 section D). No other battery arm was scored." >&2
       echo "See $D/ml1m/t1check.json; rerun with T1_OVERRIDE=1 only after the debugging is recorded in docs/sigir/PILOT_LOG.md." >&2
       exit 5
     fi ;;
  *) echo "T1 check could not run (exit $rc): see $D/ml1m/t1check.json" >&2; exit "$rc" ;;
esac
score "$MODEL" "$PML" "$D/ml1m/base" "${V0[@]}"
score "$MODEL" "$DP/ml1m_t0.jsonl" "$D/ml1m/t0" --variant T0_probe --readout yesno --questions like
# T0 E[r] (judge.md T0 protocol): secondary quantity in diag.json, the registered T0 reading uses the yes/no P(Yes)
score "$MODEL" "$DP/ml1m_t0.jsonl" "$D/ml1m/t0_digits" --variant T0_probe --readout digits --questions like
score "$MODEL" "$DP/ml1m_t2.jsonl" "$D/ml1m/t2" "${V0[@]}"
score "$MODEL" "$PML" "$D/ml1m/t3" --variant V0 --readout digits --questions like
score "$MODEL" "$DP/ml1m_starperm.jsonl" "$D/ml1m/starperm" "${V0[@]}"
score "$MODEL" "$PTOYS" "$D/toys/base" "${V0[@]}" --n_users "$N_TOYS"
score "$MODEL" "$DP/toys_t2.jsonl" "$D/toys/t2" "${V0[@]}"
if [ "${SKIP_T4:-0}" != 1 ]; then
  score "$LLAMA" "$PML" "$D/ml1m/t4_llama" "${V0[@]}"
else
  echo "SKIP_T4=1: T4 (Llama-3.1-8B-Instruct) not scored"
fi

# ---- analysis (CPU) ----
"$PY" -m src.confrec.diag_battery analyze --root "$D" --out "$D/diag.json"
echo "diagnosis battery done: $D/diag.json (interpretive only; cannot change G0-G9)"
