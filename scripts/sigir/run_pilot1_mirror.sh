#!/usr/bin/env bash
# Pilot 1 (A1 MIRROR) — pre-registered in idea-stage/triage_verdict.md §3.1, as amended by
# idea-stage/PREREG_AMENDMENT_1.md (P1, C0-C4). Runs on the GPU server.
#   usage: bash scripts/sigir/run_pilot1_mirror.sh          (PYTHON=/path/python skips conda; MODEL=... overrides)
#   prerequisites: bootstrap_server.sh done; sports panel rebuilt (scripts/sigir/rebuild_panels.sh sports).
#   ALLOW_UNANCHORED_PANEL=1 accepts a sports panel whose text is not proven equal to the original (see
#   panel_reference.py; recorded in $OUT/sports_panel.VERIFIED.json).
# Re-runnable: downloads resume and land atomically; a panel is rebuilt only when its inputs or builder code are
# newer than <panel>.done; a scorer run is skipped when DIR/report.json exists for the same panel bytes + args
# (DIR/run.key) and is moved aside to DIR.stale.<time> when the panel changed; an analysis JSON is recomputed when
# its scores, panel or analysis code are newer than <json>.done. Exit code = pilot1_gate.py's (0 PASS, 3 FAIL).
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
OUT=outputs/confrec/pilot1_mirror
PANELS=outputs/confrec/panels
mkdir -p "$OUT" "$PANELS" data/raw

# ---- helpers (identical in run_pilot2_3.sh) ----
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
BCODE="src/confrec/build_rated_panels.py src/confrec/categories.py"
ACODE="src/confrec/pilot_mirror.py src/confrec/metrics.py src/confrec/stats.py"

# --- data ---
ML=data/raw/ml-1m
if [ ! -f "$ML/ratings.dat" ]; then
  fetch https://files.grouplens.org/datasets/movielens/ml-1m.zip data/raw/ml-1m.zip
  rm -rf data/raw/.ml-1m.tmp
  mkdir -p data/raw/.ml-1m.tmp
  if ! unzip -q -o data/raw/ml-1m.zip -d data/raw/.ml-1m.tmp; then
    rm -f data/raw/ml-1m.zip; echo "corrupt data/raw/ml-1m.zip removed; rerun to download again" >&2; exit 1
  fi
  rm -rf "$ML"
  mv data/raw/.ml-1m.tmp/ml-1m "$ML"   # complete extraction only
  rm -rf data/raw/.ml-1m.tmp
fi
[ -f "$ML/ratings.dat" ] || { echo "ML-1M missing: $ML/ratings.dat" >&2; exit 1; }
"$PY" scripts/sigir/slim_amazon2023.py --domains toys --root data/raw   # skips finished files
TOYS_RAW="data/raw/amazon_toys/Toys_and_Games.jsonl.gz data/raw/amazon_toys/meta_Toys_and_Games.jsonl.gz"

step "$PANELS/ml1m_rated.jsonl" "$ML/ratings.dat" "$ML/movies.dat" $BCODE -- \
  "$PY" -m src.confrec.build_rated_panels --source ml1m --raw "$ML" --out "$PANELS/ml1m_rated.jsonl" --n_users 1500
step "$PANELS/toys_rated.jsonl" $TOYS_RAW $BCODE -- \
  "$PY" -m src.confrec.build_rated_panels --source amazon --domain toys --raw data/raw \
    --out "$PANELS/toys_rated.jsonl" --n_users 1500

# Amendment C4: the sports panel is read only after panel_reference.py verify passes. A panel that is not anchored
# (byte-identical to the original C-CRP v3 panel, or matching an anchored content ref) needs ALLOW_UNANCHORED_PANEL=1.
UA=""
if [ "${ALLOW_UNANCHORED_PANEL:-0}" = 1 ]; then UA=--allow_unanchored; fi
R=outputs/baselines/external_tasks/sports_large10000_100neg_test_same_candidate/ranking_test.jsonl
[ -f "$R" ] || { echo "missing $R: run bash scripts/sigir/rebuild_panels.sh sports" >&2; exit 1; }
"$PY" scripts/sigir/panel_reference.py check --domain sports --ranking "$R" $UA || \
  "$PY" scripts/sigir/panel_reference.py verify --domain sports --ranking "$R" $UA
cp "$R.VERIFIED" "$OUT/sports_panel.VERIFIED.json"   # anchoring status travels with the pilot outputs
# First 1,000 events of the file just verified: regenerated every run (cheap), replaced only when its bytes change
# (score() keys on its sha1, analyses on its mtime), so a replaced ranking file never leaves a stale subset behind.
head -n 1000 "$R" > "$PANELS/sports_next_1k.jsonl.tmp"
if cmp -s "$PANELS/sports_next_1k.jsonl.tmp" "$PANELS/sports_next_1k.jsonl"; then
  rm -f "$PANELS/sports_next_1k.jsonl.tmp"
else
  mv "$PANELS/sports_next_1k.jsonl.tmp" "$PANELS/sports_next_1k.jsonl"
fi

# --- scoring (Qwen3-8B, thinking off, prefill-only P(Yes)) ---
for p in ml1m_rated toys_rated; do
  # NOTE: run_pilot2_3.sh reuses toys_rated with exactly these args (same run.key)
  score "$PANELS/$p.jsonl" "$OUT/$p" --questions like,dislike,like_para --swap_k 8 --hist_len 10
  # PMI-style no-history item prior (baseline required by idea-stage/NOVELTY_DOSSIER.md)
  score "$PANELS/$p.jsonl" "$OUT/${p}_nohist" --questions like --hist_len 0
done
score "$PANELS/sports_next_1k.jsonl" "$OUT/sports_next_1k" --questions next,like,dislike,like_para --hist_len 5

# --- analysis (CPU only and mutually independent: run concurrently, fail if any fails) ---
pids=()
for p in ml1m_rated toys_rated; do
  J="$OUT/$p/pilot_mirror.json"
  step "$J" "$OUT/$p/report.json" "$OUT/${p}_nohist/report.json" "$PANELS/$p.jsonl" $ACODE -- \
    "$PY" -m src.confrec.pilot_mirror --scores "$OUT/$p/scores.csv.gz" --panel "$PANELS/$p.jsonl" \
      --swap "$OUT/$p/swap_prior.csv.gz" --nohist "$OUT/${p}_nohist/scores.csv.gz" --base_q like --out "$J" \
      > "$OUT/$p/pilot_mirror.log" 2>&1 &
  pids+=($!)
done
CCRP=docs/sigir/ref_ranks/sports/ccrp_v3.csv.gz
J="$OUT/sports_next_1k/pilot_mirror.json"
step "$J" "$OUT/sports_next_1k/report.json" "$PANELS/sports_next_1k.jsonl" "$CCRP" $ACODE -- \
  "$PY" -m src.confrec.pilot_mirror --scores "$OUT/sports_next_1k/scores.csv.gz" \
    --panel "$PANELS/sports_next_1k.jsonl" --base_q next --ref_ranks "$CCRP" --out "$J" \
    > "$OUT/sports_next_1k/pilot_mirror.log" 2>&1 &
pids+=($!)
afail=0
for pid in "${pids[@]}"; do wait "$pid" || afail=1; done
tail -n 3 "$OUT"/ml1m_rated/pilot_mirror.log "$OUT"/toys_rated/pilot_mirror.log "$OUT"/sports_next_1k/pilot_mirror.log
[ "$afail" = 0 ] || { echo "a pilot_mirror analysis failed (see $OUT/*/pilot_mirror.log)" >&2; exit 1; }

# --- pre-registered token-channel sanity gate (amendment P1.5); its exit code is this script's ---
set +e
"$PY" scripts/sigir/pilot1_gate.py --ml1m "$OUT/ml1m_rated/pilot_mirror.json" \
  --toys "$OUT/toys_rated/pilot_mirror.json" --sports "$OUT/sports_next_1k/pilot_mirror.json" --out_dir "$OUT"
rc=$?
set -e
echo "pilot1_gate exit code $rc (0 = PASS, 3 = FAIL)"
exit "$rc"
