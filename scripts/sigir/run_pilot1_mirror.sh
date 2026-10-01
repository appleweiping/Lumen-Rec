#!/usr/bin/env bash
# Pilot 1 (A1 MIRROR) — pre-registered in idea-stage/triage_verdict.md §3.1. Runs on the GPU server.
#   prerequisites: bootstrap_server.sh done; data/raw/amazon_toys slimmed; sports panel rebuilt+verified.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
MODEL="${MODEL:-/root/autodl-tmp/lumen/models/Qwen3-8B}"
OUT=outputs/confrec/pilot1_mirror
PANELS=outputs/confrec/panels
mkdir -p "$OUT" "$PANELS" data/raw

# --- data ---
[ -d data/raw/ml-1m ] || (cd data/raw && wget -q https://files.grouplens.org/datasets/movielens/ml-1m.zip && unzip -q ml-1m.zip)
python -m src.confrec.build_rated_panels --source ml1m --raw data/raw/ml-1m --out $PANELS/ml1m_rated.jsonl --n_users 1500
python -m src.confrec.build_rated_panels --source amazon --domain toys --raw data/raw --out $PANELS/toys_rated.jsonl --n_users 1500
head -n 1000 outputs/baselines/external_tasks/sports_large10000_100neg_test_same_candidate/ranking_test.jsonl > $PANELS/sports_next_1k.jsonl

# --- scoring (Qwen3-8B, thinking off, prefill-only P(Yes)) ---
for p in ml1m_rated toys_rated; do
  python -m src.confrec.pyes_scorer --data $PANELS/$p.jsonl --output $OUT/$p --model "$MODEL" \
    --questions like,dislike,like_para --swap_k 8 --hist_len 10
  # PMI-style no-history item prior (baseline required by idea-stage/NOVELTY_DOSSIER.md)
  python -m src.confrec.pyes_scorer --data $PANELS/$p.jsonl --output $OUT/${p}_nohist --model "$MODEL" \
    --questions like --hist_len 0
done
python -m src.confrec.pyes_scorer --data $PANELS/sports_next_1k.jsonl --output $OUT/sports_next_1k --model "$MODEL" \
  --questions next,like,dislike,like_para --hist_len 5

# --- analysis ---
for p in ml1m_rated toys_rated; do
  python -m src.confrec.pilot_mirror --scores $OUT/$p/scores.csv.gz --panel $PANELS/$p.jsonl \
    --swap $OUT/$p/swap_prior.csv.gz --nohist $OUT/${p}_nohist/scores.csv.gz --out $OUT/$p/pilot_mirror.json
done
python -m src.confrec.pilot_mirror --scores $OUT/sports_next_1k/scores.csv.gz --panel $PANELS/sports_next_1k.jsonl \
  --base_q next --out $OUT/sports_next_1k/pilot_mirror.json

# --- pre-registered token-channel sanity gate ---
python - <<'PY'
import json
ml = json.load(open("outputs/confrec/pilot1_mirror/ml1m_rated/pilot_mirror.json"))
sp = json.load(open("outputs/confrec/pilot1_mirror/sports_next_1k/pilot_mirror.json"))
u = ml["arms"]["raw"]["UAUC"]; n = sp["arms"]["raw"]["NDCG@10"]
print(f"GATE ml1m UAUC(raw like)={u:.4f} (>=0.60)  sports NDCG@10(raw next)={n:.4f} (>=0.1863)")
print("GATE:", "PASS" if (u >= 0.60 and n >= 0.8 * 0.2329) else "FAIL -> fix prompt/channel before interpreting")
PY
