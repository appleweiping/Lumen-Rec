#!/usr/bin/env bash
# Rebuild Lumen's frozen 10k-user 1+100 same-candidate test panels on a fresh server, then VERIFY them
# against the committed fingerprints (docs/sigir/panel_refs). Fails loudly on any mismatch.
#   usage: bash scripts/sigir/rebuild_panels.sh "sports toys home tools"
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
DOMAINS="${1:-sports toys home tools}"

python scripts/sigir/slim_amazon2023.py --domains "$(echo $DOMAINS | tr ' ' ',')" --root data/raw

for d in $DOMAINS; do
  python scripts/pipeline/main_preprocess.py --config "configs/data/amazon_${d}.yaml"
  # Arguments recovered from scripts/run_week8_large_scale_10k_100neg.sh and the stored run summaries.
  python scripts/build/main_build_large_scale_same_candidate_runtime.py \
    --processed_dir "data/processed/amazon_${d}" --domain "$d" --dataset_name "amazon_${d}" \
    --output_root outputs --exp_prefix "${d}_large10000_100neg" --user_limit 10000 \
    --num_negatives 100 --max_history_len 50 --min_sequence_length 3 --seed 20260506 \
    --shuffle_seed 42 --splits valid,test --selection_strategy random \
    --negative_sampling popularity --test_history_mode train_plus_valid
  python scripts/sigir/panel_reference.py verify --domain "$d" \
    --ranking "outputs/baselines/external_tasks/${d}_large10000_100neg_test_same_candidate/ranking_test.jsonl"
done
echo "ALL PANELS VERIFIED: $DOMAINS"
