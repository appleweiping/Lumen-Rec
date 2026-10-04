#!/usr/bin/env bash
# Infrastructure smoke test of the Gate-FT path (idea-stage/PREREG_AMENDMENT_2.md carve-out O): SYNTHETIC panel only,
# no real user, item or rating data. Trains Qwen3-8B with the G9 recipe for a few optimizer steps, then scores the
# synthetic users with vLLM with and without the adapter. Reports peak GPU memory, training throughput and whether the
# adapter is applied. Everything is written to $OUT (deleted afterwards unless KEEP=1). GPU server.
#   usage: bash scripts/sigir/run_lora_smoke.sh        (PYTHON=... skips conda; MODEL=...; OUT=...; N_EX=256; MICRO_BSZ=8)
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
OUT="${OUT:-/root/autodl-tmp/smoke_lora}"
N_EX="${N_EX:-256}"
MICRO_BSZ="${MICRO_BSZ:-8}"
ACCUM=$((32 / MICRO_BSZ))                      # the registered effective batch is 8 x 4 = 32
rm -rf "$OUT"; mkdir -p "$OUT"

"$PY" scripts/sigir/synth_smoke_panel.py --out "$OUT/panel.jsonl" --n_users 96 --n_cands 8 --hist 10

( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$OUT/mem.log"; sleep 2; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

t0=$(date +%s)
"$PY" -m src.confrec.train_lora_yesno --train "$OUT/panel.jsonl" --model "$MODEL" --out "$OUT/adapter" \
  --variant V0 --seed 0 --max_examples "$N_EX" --bsz "$MICRO_BSZ" --grad_accum "$ACCUM" 2>&1 \
  | grep -E "examples:|train_runtime|train_samples_per_second|train_loss|Error|error|Traceback|OutOfMemory" || true
t1=$(date +%s)
echo "SMOKE: training wall time $((t1 - t0)) s including model load (micro-batch $MICRO_BSZ x accumulation $ACCUM)"
echo "SMOKE: peak GPU memory during training: $(sort -n "$OUT/mem.log" | tail -n 1) MiB"
[ -f "$OUT/adapter/train_config.json" ] || { echo "SMOKE FAIL: training did not finish" >&2; exit 1; }
ls "$OUT/adapter"

score() {  # score TAG [extra scorer args]
  local tag=$1; shift
  "$PY" -m src.confrec.pyes_scorer --data "$OUT/panel.jsonl" --output "$OUT/score_$tag" --model "$MODEL" --dtype float16 \
    --topk_logprobs 50 --max_model_len 4096 --chunk_users 100 --variant V0 --readout yesno --questions like "$@" \
    > "$OUT/score_$tag.log" 2>&1 || { echo "SMOKE FAIL: scorer ($tag) rc=$?" >&2; tail -n 25 "$OUT/score_$tag.log" >&2; exit 1; }
  echo "SMOKE: scored $tag"
}
score base
score lora --lora "$OUT/adapter"

set +e
"$PY" - "$OUT" <<'PY'
import json, sys
import numpy as np
import pandas as pd

out = sys.argv[1]
base = pd.read_csv(f"{out}/score_base/scores.csv.gz").sort_values(["source_event_id", "cand_idx"]).reset_index(drop=True)
lora = pd.read_csv(f"{out}/score_lora/scores.csv.gz").sort_values(["source_event_id", "cand_idx"]).reset_index(drop=True)
assert len(base) == len(lora) > 0, (len(base), len(lora))
res = {"n_rows": int(len(base)),
       "finite_share_base": float(np.isfinite(base.logit).mean()), "finite_share_lora": float(np.isfinite(lora.logit).mean()),
       "mean_yes_no_mass_base": float(base.yes_no_mass.mean()), "mean_yes_no_mass_lora": float(lora.yes_no_mass.mean()),
       "censored2_share_lora": float((lora.censored == 2).mean()),
       "max_abs_logit_diff": float(np.nanmax(np.abs(base.logit.values - lora.logit.values))),
       "mean_abs_logit_diff": float(np.nanmean(np.abs(base.logit.values - lora.logit.values)))}
rep = json.load(open(f"{out}/score_lora/report.json"))
res["scorer_report_lora"] = {k: rep.get(k) for k in ("lora", "variant", "n_overlength", "mean_yes_no_mass") if k in rep}
res["adapter_applied"] = res["max_abs_logit_diff"] > 1e-3
res["pass"] = bool(res["adapter_applied"] and res["finite_share_lora"] > 0.999 and res["mean_yes_no_mass_lora"] >= 0.95
                   and res["censored2_share_lora"] <= 0.005)
json.dump(res, open(f"{out}/smoke_result.json", "w"), indent=2)
print("SMOKE RESULT", json.dumps(res))
sys.exit(0 if res["pass"] else 3)
PY
rc=$?
if [ "${KEEP:-0}" != 1 ]; then
  cp "$OUT/smoke_result.json" "$OUT.result.json" 2>/dev/null || true
  rm -rf "$OUT"
fi
echo "SMOKE: exit $rc (0 = adapter trained, loaded by vLLM, applied, finite)"
exit "$rc"
