#!/usr/bin/env bash
# Amendment-2 STAGE 3 (idea-stage/PREREG_AMENDMENT_2.md G7 with its 2026-10-04 scale note and the "G7 stage-3 gate
# source" clarification). Runs ONLY after the G6 GATE_PASS that scripts/sigir/run_gatefix.sh stage 2 records in
# outputs/confrec/gatefix/confirm/gate.json. GPU server.
#   usage: bash scripts/sigir/run_gatefix_stage3.sh     (PYTHON=/path/python skips conda; MODEL=... must be G6's model)
# Refused (exit 2) before any panel is written and before any scorer call when
#   * gate.json is missing, or the Amendment-2 freeze check of run_gatefix.sh fails (diag_battery freeze --check:
#     FREEZE.txt unchanged and every REQUIRED sha1 recorded in docs/sigir/PILOT_LOG.md), or
#   * src.confrec.gatefix_stage3 (the input check, see its docstring) fails: gate.json is not GATE_PASS or the recorded
#     stage-3 token-channel gate does not hold (pilot1_gate.stage3_gate on gate.json + the Pilot-1 decision.json),
#     V* of selection.json != gate.json's, MODEL != the G6 backbone, a panel is not the frozen / G6-scored one, or the
#     sports VALID panel is missing or holds a quarantined sports TEST event 1-1000.
# Scoring (pyes_scorer: fp16, top-50 logprobs, max_model_len 4096 (G0), 100-user chunks; the scorer resolves V*'s
# registered history window and question family, i.e. the threshold-family dislike / like_para strings of V1 / V7):
#   rated, under V* (selection.json = gate.json v_star), on CONFIRM ML-1M (ml1m_confirm_h20.jsonl: every fresh user =
#   the G6 users) and on the second rated domain <d>: fresh Toys = the first 1,500 rows of toys_confirm_h20.jsonl when
#   >= 500 fresh Toys users are eligible, else Video_Games = the first 1,500 users of its seed-0 h20 panel (built here):
#     S3/<d>          --questions like,dislike,like_para --swap_k 8   (one run: the 3 questions + the K=8 swap prior)
#     S3/<d>_nohist   --questions like --hist_len 0                    (the no-history prior)
#   next-item, under V0 (G0: frozen, not re-tested): MIRROR's no-loss check on the first 1,000 events of the sports
#   VALID panel (outputs/baselines/external_tasks/sports_large10000_100neg_valid_same_candidate/ranking_valid.jsonl,
#   cut as run_nextitem_audit.sh cuts its valid2k; the TEST panel is only read for the quarantine check):
#     S3/sports_valid_1k  --variant V0 --questions next,like,dislike,like_para --hist_len 5
#   The next-item audit's valid2k output (section N) holds `next` and `like` only (carve-out N; frozen 189c164 checkout,
#   max_model_len 3072), so it is not reused: the four questions are scored here, in one run, by one code version.
# Analysis (CPU): pilot_mirror per panel (rated: --swap --nohist --base_q like; sports: --base_q next --ref_ranks
#   docs/sigir/ref_ranks/sports/ccrp_v3.csv.gz, which covers the TEST events only: on VALID events its join is empty and
#   the C-CRP arm is null, context only), then pilot1_gate.py --stage3_gate gate.json --pilot1_decision
#   outputs/confrec/pilot1_mirror/decision.json -> S3/decision.json (POSITIVE / NULL / NEGATIVE / INDETERMINATE /
#   INCOMPLETE / GATE_FAIL_UNINTERPRETABLE) and the marker S3/GATE_PASS or S3/GATE_FAIL. S3 = outputs/confrec/gatefix/stage3.
# Estimated cost, from the Pilot-1 throughputs (rated ML-1M 87, Toys 35, sports 175 prompts/s): CONFIRM ML-1M about
#   0.45 GPU-h (about 123k prompts incl. the swap prior + 33k no-history), fresh Toys about 1.5 GPU-h (about 190k: the
#   per-item swap prior dominates, nearly every Toys candidate is a distinct item), sports VALID about 0.65 GPU-h (404k)
#   -> about 2.7 GPU-h in total incl. model loads, up to about 3 GPU-h if V* renders 20 history events (V3, V7).
# Exit code: pilot1_gate's (0 = the recorded stage-3 gate holds: decision.json carries the label; 3 = GATE_FAIL ->
#   GATE_FAIL_UNINTERPRETABLE, e.g. an input check failed); 2 = refused (nothing scored); 1 = any other failure.
# Re-runnable: sub-panels are replaced only when their bytes change; a scorer run is skipped when DIR/report.json exists
# for the same panel bytes + model + args (DIR/run.key) and moved aside to DIR.stale.<time> when they changed; an
# analysis reruns only when its scores, panel or code are newer than <json>.done; the input record and pilot1_gate's
# outputs are rewritten only when their content changes (a complete rerun touches no file).
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
S3="$G/stage3"
FREEZE="$G/FREEZE.txt"
AMEND=idea-stage/PREREG_AMENDMENT_2.md
PILOT_LOG=docs/sigir/PILOT_LOG.md
GATE="$CONF/gate.json"
SEL="$DEV/selection.json"
P1DEC=outputs/confrec/pilot1_mirror/decision.json
CP="$GP/ml1m_confirm_h20.jsonl"
TOYS_CONFIRM="$GP/toys_confirm_h20.jsonl"
XT=outputs/baselines/external_tasks
VALID="$XT/sports_large10000_100neg_valid_same_candidate/ranking_valid.jsonl"
TEST="$XT/sports_large10000_100neg_test_same_candidate/ranking_test.jsonl"
CCRP=docs/sigir/ref_ranks/sports/ccrp_v3.csv.gz
GAMES_RAW=(data/raw/amazon_games/Video_Games.jsonl.gz data/raw/amazon_games/meta_Video_Games.jsonl.gz)
BCODE=(src/confrec/build_rated_panels.py src/confrec/categories.py)
ACODE=(src/confrec/pilot_mirror.py src/confrec/metrics.py src/confrec/stats.py)

refuse() { echo "STAGE 3 REFUSED: $* Nothing was scored." >&2; exit 2; }

# ================= refusals: nothing below them writes a panel or calls the scorer =================
[ -f "$GATE" ] || refuse "no G6 gate record $GATE: stage 3 runs only after run_gatefix.sh stage 2 recorded GATE_PASS (G7)."
FREEZE_ARGS=(--amendment "$AMEND" --dev_panels "$GP/ml1m_dev_h20.jsonl,$GP/toys_dev_h20.jsonl"
             --manifest "$GP/manifest.json" --out "$FREEZE")
echo "== amendment-2 freeze check ($FREEZE)"
"$PY" -m src.confrec.diag_battery freeze "${FREEZE_ARGS[@]}" --check --pilot_log "$PILOT_LOG" | tail -n 1 || \
  refuse "the amendment-2 freeze check failed (see above): FREEZE.txt must be unchanged and recorded in $PILOT_LOG."
# the input check writes S3/inputs.json only when it passes; stdout: V*, second domain, its users, sports VALID events
if ! PRE=$("$PY" -m src.confrec.gatefix_stage3 --gate "$GATE" --selection "$SEL" --manifest "$GP/manifest.json" \
    --confirm_panel "$CP" --toys_confirm "$TOYS_CONFIRM" --pilot1_decision "$P1DEC" --model "$MODEL" \
    --sports_valid "$VALID" --sports_test "$TEST" --out "$S3/inputs.json"); then
  refuse "the stage-3 input check failed (reasons above)."
fi
read -r VSTAR SECOND N2 NSP <<< "$PRE"
case "$SECOND" in toys|games) ;; *) echo "unexpected input check output: '$PRE'" >&2; exit 1 ;; esac
[ -n "$VSTAR" ] && [ -n "$N2" ] && [ -n "$NSP" ] || { echo "unexpected input check output: '$PRE'" >&2; exit 1; }
echo "STAGE 3: V* = $VSTAR; second rated domain = $SECOND (first $N2 users); sports VALID first $NSP events"

# ---- helpers (as in run_gatefix.sh) ----
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
# completion marker DIR/report.json, skip key DIR/run.key = panel sha1 + model + args
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
# first_lines SRC N OUT: the first N lines of SRC (frozen-panel convention of run_pilot1_mirror.sh /
# run_nextitem_audit.sh): regenerated every run, OUT replaced only when its bytes change
first_lines() {
  local src=$1 n=$2 out=$3
  head -n "$n" "$src" > "$out.tmp"
  if cmp -s "$out.tmp" "$out"; then rm -f "$out.tmp"; else mv "$out.tmp" "$out"; fi
}
# analysis JSON LOG DEP... -- CMD...: step's skip rule; otherwise CMD runs in the background with its output in LOG
# (CPU only, the analyses are independent), JSON.done is touched after it succeeds; its pid joins PIDS
PIDS=()
analysis() {
  local prod=$1 log=$2 deps=(); shift 2
  while [ "$1" != "--" ]; do deps+=("$1"); shift; done; shift
  if [ -e "$prod" ] && fresh "$prod.done" "${deps[@]}"; then echo "[skip] $prod"; return 0; fi
  rm -f "$prod.done"
  ( "$@" > "$log" 2>&1 && touch "$prod.done" ) &
  PIDS+=($!)
}

# ================= panels =================
mkdir -p "$S3/panels"
SV="$S3/panels/sports_valid_1k.jsonl"
first_lines "$VALID" "$NSP" "$SV"
if [ "$SECOND" = toys ]; then
  SP2="$S3/panels/toys_fresh_first${N2}_h20.jsonl"
  first_lines "$TOYS_CONFIRM" "$N2" "$SP2"
else
  # G7 fallback (fewer than 500 fresh Toys users): the first N2 users of the seed-0 Video_Games panel, rendered as the
  # G2 panels are (20 history events, history_meta / domain_kind), built with the Pilot-1 builder arguments
  "$PY" scripts/sigir/slim_amazon2023.py --domains games --root data/raw   # skips finished files
  SP2="$S3/panels/games_first${N2}_h20.jsonl"
  step "$SP2" "${GAMES_RAW[@]}" "${BCODE[@]}" -- \
    "$PY" -m src.confrec.build_rated_panels --source amazon --domain games --raw data/raw --hist_len 20 \
      --gatefix_fields --n_users "$N2" --out "$SP2"
fi

# ================= scoring (GPU) =================
"$PY" -c "import sys, pandas, vllm; print('python', sys.executable, '| pandas', pandas.__version__, '| vllm', vllm.__version__)"
score "$CP" "$S3/ml1m" --variant "$VSTAR" --readout yesno --questions like,dislike,like_para --swap_k 8
score "$CP" "$S3/ml1m_nohist" --variant "$VSTAR" --readout yesno --questions like --hist_len 0
score "$SP2" "$S3/$SECOND" --variant "$VSTAR" --readout yesno --questions like,dislike,like_para --swap_k 8
score "$SP2" "$S3/${SECOND}_nohist" --variant "$VSTAR" --readout yesno --questions like --hist_len 0
score "$SV" "$S3/sports_valid_1k" --variant V0 --readout yesno --questions next,like,dislike,like_para --hist_len 5

# ================= analysis (CPU; concurrent, fail if any fails) =================
analysis "$S3/ml1m/pilot_mirror.json" "$S3/ml1m/pilot_mirror.log" "$S3/ml1m/report.json" \
    "$S3/ml1m_nohist/report.json" "$CP" "${ACODE[@]}" -- \
  "$PY" -m src.confrec.pilot_mirror --scores "$S3/ml1m/scores.csv.gz" --panel "$CP" \
    --swap "$S3/ml1m/swap_prior.csv.gz" --nohist "$S3/ml1m_nohist/scores.csv.gz" --base_q like \
    --out "$S3/ml1m/pilot_mirror.json"
analysis "$S3/$SECOND/pilot_mirror.json" "$S3/$SECOND/pilot_mirror.log" "$S3/$SECOND/report.json" \
    "$S3/${SECOND}_nohist/report.json" "$SP2" "${ACODE[@]}" -- \
  "$PY" -m src.confrec.pilot_mirror --scores "$S3/$SECOND/scores.csv.gz" --panel "$SP2" \
    --swap "$S3/$SECOND/swap_prior.csv.gz" --nohist "$S3/${SECOND}_nohist/scores.csv.gz" --base_q like \
    --out "$S3/$SECOND/pilot_mirror.json"
analysis "$S3/sports_valid_1k/pilot_mirror.json" "$S3/sports_valid_1k/pilot_mirror.log" \
    "$S3/sports_valid_1k/report.json" "$SV" "$CCRP" "${ACODE[@]}" -- \
  "$PY" -m src.confrec.pilot_mirror --scores "$S3/sports_valid_1k/scores.csv.gz" --panel "$SV" --base_q next \
    --ref_ranks "$CCRP" --out "$S3/sports_valid_1k/pilot_mirror.json"
afail=0
for pid in ${PIDS[@]+"${PIDS[@]}"}; do wait "$pid" || afail=1; done
[ "$afail" = 0 ] || { echo "a pilot_mirror analysis failed (see $S3/*/pilot_mirror.log)" >&2; exit 1; }

# ================= the registered stage-3 decision (amendment 2 G7: the recorded token-channel gate) =================
set +e
"$PY" scripts/sigir/pilot1_gate.py --ml1m "$S3/ml1m/pilot_mirror.json" --toys "$S3/$SECOND/pilot_mirror.json" \
  --sports "$S3/sports_valid_1k/pilot_mirror.json" --out_dir "$S3" --stage3_gate "$GATE" --pilot1_decision "$P1DEC"
rc=$?
set -e
echo "pilot1_gate (stage 3) exit code $rc (0 = the recorded gate holds, 3 = GATE_FAIL); decision: $S3/decision.json"
exit "$rc"
