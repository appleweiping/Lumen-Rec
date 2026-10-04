#!/usr/bin/env bash
# Gate-FT (idea-stage/PREREG_AMENDMENT_2.md G9): TALLRec-style LoRA yes/no SFT, r=16 alpha=32 lr 1e-4, 1 epoch, 3 seeds,
# trained on the burned DEV users' candidates BEFORE T, evaluated on the fresh CONFIRM users' candidates AT/AFTER T.
# PASS iff the mean over the 3 seeds of UAUC(post-T) >= 0.65 (src/confrec/gateft_eval.py). Runs in EVERY branch of the
# prompt-fix round; the prompt is selection.json's `gate_ft_prompt` (V* on FIX_FOUND, else V0), so it starts after
# `run_gatefix.sh` stage 1 (selection.json exists). GPU server.
#   usage: bash scripts/sigir/run_gateft.sh              (PYTHON=... skips conda; MODEL=...; VARIANT=... overrides)
#          BENCH=1 bash scripts/sigir/run_gateft.sh      (256-example throughput benchmark, writes nothing registered)
# Re-runnable: an adapter with train_config.json + adapter weights is not retrained; scoring is skipped when
# DIR/report.json exists for the same panel + model + adapter + args (run.key).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  set +u; source /root/miniconda3/etc/profile.d/conda.sh; conda activate lumen; set -u
  PY=python
fi
MODEL="${MODEL:-/root/autodl-tmp/lumen/models/Qwen3-8B}"
G=outputs/confrec/gatefix
GT=outputs/confrec/gateft
SEL="$G/dev/selection.json"
DEVP="$G/panels/ml1m_dev_h20.jsonl"
CONFP="$G/panels/ml1m_confirm_h20.jsonl"
for f in "$SEL" "$DEVP" "$CONFP"; do [ -f "$f" ] || { echo "missing $f (run scripts/sigir/run_gatefix.sh through stage 1 first)" >&2; exit 1; }; done
VARIANT="${VARIANT:-$("$PY" -c "import json; print(json.load(open('$SEL'))['gate_ft_prompt'])")}"
echo "Gate-FT prompt variant: $VARIANT (selection.json gate_ft_prompt; registered G9)"
mkdir -p "$GT/adapters" "$GT/scores"

if [ "${BENCH:-0}" = 1 ]; then
  "$PY" -m src.confrec.gateft_data --dev_panel "$DEVP" --confirm_panel "$CONFP" --out_dir "$GT/bench_data" > /dev/null
  rm -rf "$GT/bench"
  t0=$(date +%s)
  "$PY" -m src.confrec.train_lora_yesno --train "$GT/bench_data/train.jsonl" --model "$MODEL" \
    --out "$GT/bench" --variant "$VARIANT" --seed 0 --max_examples 256 2>&1 \
    | grep -E "examples:|train_runtime|train_samples_per_second|Error|error|Traceback" || true
  t1=$(date +%s)
  echo "BENCH: 256 examples in $((t1 - t0)) s including model load (see train_runtime above for the pure training time)"
  echo "BENCH: the full run trains $(python -c "import json;print(json.load(open('$GT/bench_data/gateft_split.json'))['train']['candidates'])" 2>/dev/null || echo '?') examples per seed"
  exit 0
fi

# Amendment 3 (section 0), two stages (BENCH is exempt): the CURRENT sha1 of the amendment must be in the pilot log before
# any registered adapter is trained (training has no outcome statistic); the full record (every bound file and every
# ftgrid_split.json) must be there before any adapter is scored.
"$PY" -m src.confrec.ftgrid_freeze --check --stage amendment || exit 1

# ---- stage A: data (CPU) ----
"$PY" -m src.confrec.gateft_data --dev_panel "$DEVP" --confirm_panel "$CONFP" --out_dir "$GT"

# score DATA DIR [--lora DIR] ARGS...: completion marker DIR/report.json; keyed on panel bytes + model + adapter + args
score() {
  local data=$1 dir=$2 key old=""; shift 2
  key="$(sha1sum "$data" | cut -d' ' -f1) $MODEL $*"
  if [ -f "$dir/run.key" ]; then old=$(cat "$dir/run.key"); fi
  if [ "$old" = "$key" ] && [ -f "$dir/report.json" ]; then echo "[skip] $dir: scored"; return 0; fi
  if [ -e "$dir" ] && [ "$old" != "$key" ]; then mv "$dir" "$dir.stale.$(date +%Y%m%d%H%M%S)"; fi
  mkdir -p "$dir"; echo "$key" > "$dir/run.key"
  "$PY" -m src.confrec.pyes_scorer --data "$data" --output "$dir" --model "$MODEL" --dtype float16 \
    --topk_logprobs 50 --max_model_len 4096 --chunk_users 100 --variant "$VARIANT" --readout yesno \
    --questions like "$@"
}

# ---- stage B: train the 3 seeds (all of them before any scoring) ----
for seed in 0 1 2; do
  A="$GT/adapters/s$seed"
  if [ -f "$A/train_config.json" ] && ls "$A"/adapter_model.* >/dev/null 2>&1; then
    echo "[skip] adapter s$seed exists"
  else
    rm -rf "$A"
    "$PY" -m src.confrec.train_lora_yesno --train "$GT/train.jsonl" --model "$MODEL" --out "$A" \
      --variant "$VARIANT" --seed "$seed" 2>&1 | grep -vE "it/s\]|s/it\]" || true
    [ -f "$A/train_config.json" ] || { echo "training of seed $seed did not finish" >&2; exit 1; }
  fi
done

# ---- scoring waits for the full Amendment-3 record (bound code and ftgrid_split.json of the four domains) ----
"$PY" -m src.confrec.ftgrid_freeze --check --stage core || {
  echo "adapters are trained and kept; rerun this script once the record exists (it skips what is done)" >&2; exit 1; }

# ---- stage C: score the 3 adapters, then the zero-shot context on the same rows ----
for seed in 0 1 2; do
  score "$CONFP" "$GT/scores/s$seed" --lora "$GT/adapters/s$seed"
done
score "$CONFP" "$GT/scores/zeroshot"

# ---- stage D: the registered decision ----
set +e
"$PY" -m src.confrec.gateft_eval --confirm_panel "$CONFP" --split_report "$GT/gateft_split.json" \
  --seed_dirs "$GT/scores/s0,$GT/scores/s1,$GT/scores/s2" --zeroshot_dir "$GT/scores/zeroshot" --out "$GT/gate_ft.json"
rc=$?   # stage D
set -e
echo "gateft_eval exit code $rc (0 = GATE_FT_PASS, 3 = GATE_FT_FAIL, 2 = incomplete/input problem; $GT/gate_ft.json)"
exit "$rc"
