#!/usr/bin/env bash
# Full-scale zero-shot next-item audit under the prompt FROZEN at V0 (idea-stage/PREREG_AMENDMENT_2.md section N).
# Scores `next` and `like` (nothing else) on the Lumen same-candidate panels with Qwen3-8B fp16:
#   TEST  : all 10,000 events of each domain          -> $OUT/<domain>_test
#   VALID : first 2,000 events (temperature fits only) -> $OUT/<domain>_valid2k
# Code is a FROZEN checkout (commit 189c164, the code behind Pilot 1) so later repo changes cannot alter the scores.
#   usage (server): bash scripts/sigir/run_nextitem_audit.sh            (DOMAINS="sports toys" to restrict)
# Resumable: killed jobs continue from the last finished 100-user chunk; a finished scorer run is skipped.
set -euo pipefail
CODE="${CODE:-/root/autodl-tmp/lumen-audit}"        # git worktree at 189c164 (git worktree add --detach $CODE 189c164)
DATA="${DATA:-/root/autodl-tmp/lumen-rec}"          # holds outputs/baselines/external_tasks (verified panels)
OUT="${OUT:-/root/autodl-tmp/lumen-audit-out}"
MODEL="${MODEL:-/root/autodl-tmp/lumen/models/Qwen3-8B}"
DOMAINS="${DOMAINS:-sports toys home tools}"
export PATH="/root/miniconda3/bin:$PATH"
source /root/miniconda3/etc/profile.d/conda.sh && conda activate lumen
cd "$CODE"
export PYTHONPATH=. PYTHONHASHSEED=0
mkdir -p "$OUT"
# the checkout must be exactly the audited commit
[ "$(git rev-parse --short=7 HEAD)" = "189c164" ] || { echo "CODE is not at 189c164" >&2; exit 1; }

for d in $DOMAINS; do
  T="$DATA/outputs/baselines/external_tasks/${d}_large10000_100neg_test_same_candidate/ranking_test.jsonl"
  V="$DATA/outputs/baselines/external_tasks/${d}_large10000_100neg_valid_same_candidate/ranking_valid.jsonl"
  # TEST must be the byte-identical original panel (panel_reference.py verify wrote this marker)
  [ -f "$T.VERIFIED" ] || python "$DATA/scripts/sigir/panel_reference.py" check --domain "$d" --ranking "$T" >/dev/null \
    || { echo "test panel for $d is not verified" >&2; exit 1; }
  head -n 2000 "$V" > "$OUT/${d}_valid2k.jsonl.tmp"
  if ! cmp -s "$OUT/${d}_valid2k.jsonl.tmp" "$OUT/${d}_valid2k.jsonl" 2>/dev/null; then
    mv "$OUT/${d}_valid2k.jsonl.tmp" "$OUT/${d}_valid2k.jsonl"
  else
    rm -f "$OUT/${d}_valid2k.jsonl.tmp"
  fi
  for part in valid2k test; do
    data="$OUT/${d}_valid2k.jsonl"; [ "$part" = test ] && data="$T"
    python -m src.confrec.pyes_scorer --data "$data" --output "$OUT/${d}_${part}" --model "$MODEL" \
      --questions next,like --hist_len 5 --dtype float16 --topk_logprobs 50 --chunk_users 100 \
      2>&1 | grep -vE "INFO|WARNING|it/s\]" || true
    [ -f "$OUT/${d}_${part}/report.json" ] || { echo "scorer did not finish for ${d}_${part}" >&2; exit 1; }
  done
done
echo "NEXTITEM_AUDIT_DONE: $DOMAINS"
