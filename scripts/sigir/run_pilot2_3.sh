#!/usr/bin/env bash
# Pilots 2 (KuaiRec MAR vs logged calibration) and 3 (pseudonym knockout) — triage_verdict.md §3.2/§3.3, as
# amended by idea-stage/PREREG_AMENDMENT_1.md (P2, P3: domains Toys + Video_Games).
#   usage: bash scripts/sigir/run_pilot2_3.sh [2|3|all]     (PYTHON=/path/python skips conda; MODEL=... overrides)
# Re-runnable with the same skip/stale rules as run_pilot1_mirror.sh (see its header). Pilot 3 shares pilot 1's
# token-channel gate: it exits 3 before any pilot-3 scoring when outputs/confrec/pilot1_mirror/GATE_FAIL exists.
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
"$PY" -c "import sys, pandas, vllm; print('python', sys.executable, '| pandas', pandas.__version__, '| vllm', vllm.__version__)"
MODEL="${MODEL:-/root/autodl-tmp/lumen/models/Qwen3-8B}"
PANELS=outputs/confrec/panels
WHICH="${1:-all}"
mkdir -p "$PANELS" data/raw

# ---- helpers (identical in run_pilot1_mirror.sh) ----
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
# fetch URL FILE: resumable download; FILE only appears once complete
fetch() {
  if [ -f "$2" ]; then return 0; fi
  wget -c -q -O "$2.part" "$1" || return 1
  mv "$2.part" "$2"
}
# score DATA DIR ARGS...: pyes_scorer (fp16, top-50 logprobs, 100-user chunks), completion marker DIR/report.json
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
    --dtype float16 --topk_logprobs 50 --chunk_users 100 "$@"
}
STATS="src/confrec/metrics.py src/confrec/stats.py"

if [ "$WHICH" = 2 ] || [ "$WHICH" = all ]; then
  OUT=outputs/confrec/pilot2_kuairec
  # Official release (Gao et al., CIKM'22), verified in github.com/chongminggao/KuaiRec README:
  # Zenodo record 18164998; captions are a separate file in the same record.
  find_small() { find data/raw/KuaiRec -name small_matrix.csv -print -quit 2>/dev/null || true; }
  if [ -z "$(find_small)" ]; then
    fetch https://zenodo.org/records/18164998/files/KuaiRec.zip data/raw/KuaiRec.zip
    rm -rf data/raw/.KuaiRec.tmp
    mkdir -p data/raw/.KuaiRec.tmp
    if ! unzip -q data/raw/KuaiRec.zip -d data/raw/.KuaiRec.tmp; then
      rm -f data/raw/KuaiRec.zip; echo "corrupt data/raw/KuaiRec.zip removed; rerun to download again" >&2; exit 1
    fi
    rm -rf data/raw/KuaiRec   # an earlier, incomplete extraction (no small_matrix.csv)
    mv data/raw/.KuaiRec.tmp data/raw/KuaiRec
  fi
  SM="$(find_small)"
  [ -n "$SM" ] && [ -f "$SM" ] || { echo "KuaiRec small_matrix.csv not found under data/raw/KuaiRec" >&2; exit 1; }
  KDIR="$(dirname "$SM")"
  [ -f "$KDIR/big_matrix.csv" ] || { echo "KuaiRec big_matrix.csv missing in $KDIR" >&2; exit 1; }
  CAP="$KDIR/kuairec_caption_category.csv"
  fetch https://zenodo.org/records/18164998/files/kuairec_caption_category.csv "$CAP"
  step "$PANELS/kuairec.jsonl" "$SM" "$KDIR/big_matrix.csv" "$CAP" src/confrec/build_kuairec_panel.py -- \
    "$PY" -m src.confrec.build_kuairec_panel --root "$KDIR" --caption "$CAP" --out "$PANELS/kuairec.jsonl" \
      --n_users 300 --n_cands 200
  score "$PANELS/kuairec.jsonl" "$OUT" --questions like --hist_len 10
  step "$OUT/pilot_kuairec.json" "$OUT/report.json" "$PANELS/kuairec.jsonl" src/confrec/pilot_kuairec.py $STATS -- \
    "$PY" -m src.confrec.pilot_kuairec --scores "$OUT/scores.csv.gz" --panel "$PANELS/kuairec.jsonl" \
      --out "$OUT/pilot_kuairec.json"
fi

if [ "$WHICH" = 3 ] || [ "$WHICH" = all ]; then
  OUT=outputs/confrec/pilot3_pseudonym
  P1=outputs/confrec/pilot1_mirror
  # triage_verdict.md section 3: the token-channel gate (pilot1_gate.py) is common to pilots 1 and 3; after a FAIL
  # nothing in pilot 3 is interpretable, so no GPU time is spent on it (IGNORE_PILOT1_GATE=1 overrides; the verdict
  # is then recorded as GATE_FAIL_UNINTERPRETABLE). Not run yet: pilot 3 proceeds, its verdict marked provisional.
  if [ -f "$P1/GATE_FAIL" ] && [ "${IGNORE_PILOT1_GATE:-0}" != 1 ]; then
    echo "STOP pilot 3: the pilot-1 token-channel gate FAILED ($P1/GATE_FAIL); fix the prompt/channel first" >&2
    exit 3
  fi
  [ -f "$P1/GATE_PASS" ] || echo "WARNING: pilot-1 gate not passed/run ($P1/GATE_PASS missing): pilot-3 verdicts" \
    "are provisional until bash scripts/sigir/run_pilot1_mirror.sh passes" >&2
  "$PY" scripts/sigir/slim_amazon2023.py --domains toys,games --root data/raw   # skips finished files
  for d in toys games; do
    P="$PANELS/${d}_rated.jsonl"
    CAT="$("$PY" -c "from src.confrec.categories import CATEGORY; print(CATEGORY['$d'])")"
    # toys: the same panel (and stamp) as pilot 1
    step "$P" "data/raw/amazon_$d/$CAT.jsonl.gz" "data/raw/amazon_$d/meta_$CAT.jsonl.gz" \
        src/confrec/build_rated_panels.py src/confrec/categories.py -- \
      "$PY" -m src.confrec.build_rated_panels --source amazon --domain "$d" --raw data/raw --out "$P" --n_users 1500
    users="$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['users'])" "${P%.jsonl}.meta.json")"
    if [ "$users" -lt 300 ]; then
      echo "SKIP pilot 3 / $d: rated panel has $users users (< 300 minimum, amendment P3)" >&2
      continue
    fi
    step "${P%.jsonl}_pseudo.jsonl" "$P" src/confrec/pseudonymize.py src/confrec/categories.py -- \
      "$PY" -m src.confrec.pseudonymize --panel "$P" --seed 0
    if [ "$d" = toys ]; then
      REAL="$P1/toys_rated"   # pilot 1's real-name arm, reused: args identical to run_pilot1_mirror.sh
      score "$P" "$REAL" --questions like,dislike,like_para --swap_k 8 --hist_len 10
    else
      REAL="$OUT/${d}_rated"
      score "$P" "$REAL" --questions like,dislike --swap_k 8 --hist_len 10
    fi
    score "${P%.jsonl}_pseudo.jsonl" "$OUT/${d}_rated_pseudo" --questions like,dislike --swap_k 8 --hist_len 10
    score "${P%.jsonl}_placebo.jsonl" "$OUT/${d}_rated_placebo" --questions like --hist_len 10
    J="$OUT/${d}_pilot_pseudonym.json"
    # depends on the gate's decision.json (rewritten only when its content changes); absent = rerun next time
    step "$J" "$REAL/report.json" "$OUT/${d}_rated_pseudo/report.json" "$OUT/${d}_rated_placebo/report.json" \
        "$P" "${P%.jsonl}_pseudo.jsonl" "${P%.jsonl}_placebo.jsonl" "$P1/decision.json" \
        src/confrec/pilot_pseudonym.py src/confrec/pseudonymize.py $STATS -- \
      "$PY" -m src.confrec.pilot_pseudonym --real "$REAL" --pseudo "$OUT/${d}_rated_pseudo" \
        --placebo "$OUT/${d}_rated_placebo" --panel "$P" --out "$J" --gate "$P1/decision.json"
  done
fi
