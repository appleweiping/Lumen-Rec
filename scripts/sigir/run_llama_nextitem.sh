#!/usr/bin/env bash
# Z2 of Amendment 3 (idea-stage/PREREG_AMENDMENT_3.md section 4): the second-backbone replication of the next-item audit.
# Llama-3.1-8B-Instruct fp16, prompt FROZEN at V0, questions `next` and `like`, hist_len 5, on TEST events 1,001-3,000 of each of
# sports, toys, home and tools (2,000 events per domain; sports events 1-1000 stay quarantined) and on the first 500 VALID events
# of each domain (temperature fit only). Same frozen code checkout as the Qwen audit (commit 189c164), so the prompts are
# identical and only the model differs. About 7.5 GPU-h (TEST 6, VALID 1.5).
#   usage (server, as a queued job after the Qwen audit has finished): bash scripts/sigir/run_llama_nextitem.sh
# Resumable (100-user chunks). Output: $OUT/<domain>_test1001_3000 and $OUT/<domain>_valid500. Starts only when the Amendment-3
# record exists.
set -euo pipefail
REPO="${REPO:-/root/autodl-tmp/lumen-rec}"          # current checkout: holds ftgrid_freeze and the verified panels
CODE="${CODE:-/root/autodl-tmp/lumen-audit}"        # frozen worktree at 189c164
DATA="${DATA:-/root/autodl-tmp/lumen-rec}"
OUT="${OUT:-/root/autodl-tmp/lumen-audit-out-llama}"
MODEL="${MODEL:-/root/autodl-tmp/lumen/models/Llama-3.1-8B-Instruct}"
DOMAINS="${DOMAINS:-sports toys home tools}"
export PATH="/root/miniconda3/bin:$PATH"
source /root/miniconda3/etc/profile.d/conda.sh && conda activate lumen
export PYTHONHASHSEED=0

( cd "$REPO" && PYTHONPATH=. python -m src.confrec.ftgrid_freeze --check --stage core ) || exit 1

cd "$CODE"
export PYTHONPATH=.
mkdir -p "$OUT"
[ "$(git rev-parse --short=7 HEAD)" = "189c164" ] || { echo "CODE is not at 189c164" >&2; exit 1; }

for d in $DOMAINS; do
  T="$DATA/outputs/baselines/external_tasks/${d}_large10000_100neg_test_same_candidate/ranking_test.jsonl"
  [ -f "$T.VERIFIED" ] || python "$DATA/scripts/sigir/panel_reference.py" check --domain "$d" --ranking "$T" >/dev/null \
    || { echo "test panel for $d is not verified" >&2; exit 1; }
  sed -n '1001,3000p' "$T" > "$OUT/${d}_test1001_3000.jsonl.tmp"
  [ "$(wc -l < "$OUT/${d}_test1001_3000.jsonl.tmp")" = 2000 ] || { echo "$d: expected 2000 events" >&2; exit 1; }
  if ! cmp -s "$OUT/${d}_test1001_3000.jsonl.tmp" "$OUT/${d}_test1001_3000.jsonl" 2>/dev/null; then
    mv "$OUT/${d}_test1001_3000.jsonl.tmp" "$OUT/${d}_test1001_3000.jsonl"
  else
    rm -f "$OUT/${d}_test1001_3000.jsonl.tmp"
  fi
  python -m src.confrec.pyes_scorer --data "$OUT/${d}_test1001_3000.jsonl" --output "$OUT/${d}_test1001_3000" \
    --model "$MODEL" --questions next,like --hist_len 5 --dtype float16 --topk_logprobs 50 --chunk_users 100 \
    2>&1 | grep -vE "INFO|WARNING|it/s\]" || true
  [ -f "$OUT/${d}_test1001_3000/report.json" ] || { echo "scorer did not finish for $d" >&2; exit 1; }
  # the first 500 VALID events: used only to fit this backbone's list temperature (Amendment 3 section 4)
  V="$DATA/outputs/baselines/external_tasks/${d}_large10000_100neg_valid_same_candidate/ranking_valid.jsonl"
  head -n 500 "$V" > "$OUT/${d}_valid500.jsonl.tmp"
  if ! cmp -s "$OUT/${d}_valid500.jsonl.tmp" "$OUT/${d}_valid500.jsonl" 2>/dev/null; then
    mv "$OUT/${d}_valid500.jsonl.tmp" "$OUT/${d}_valid500.jsonl"
  else
    rm -f "$OUT/${d}_valid500.jsonl.tmp"
  fi
  python -m src.confrec.pyes_scorer --data "$OUT/${d}_valid500.jsonl" --output "$OUT/${d}_valid500" \
    --model "$MODEL" --questions next,like --hist_len 5 --dtype float16 --topk_logprobs 50 --chunk_users 100 \
    2>&1 | grep -vE "INFO|WARNING|it/s\]" || true
  [ -f "$OUT/${d}_valid500/report.json" ] || { echo "scorer did not finish for ${d}_valid500" >&2; exit 1; }
done
echo "LLAMA_NEXTITEM_DONE: $DOMAINS"
