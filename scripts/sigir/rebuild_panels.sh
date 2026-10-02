#!/usr/bin/env bash
# Rebuild Lumen's frozen 10k-user 1+100 same-candidate test panels on a fresh server, then VERIFY them against the
# committed fingerprints (docs/sigir/panel_refs) and the original panels' sha256. Fails loudly on any mismatch.
#   usage: bash scripts/sigir/rebuild_panels.sh "sports toys home tools"    (PYTHON=/path/python skips conda)
# Per domain: skip when <ranking>.VERIFIED matches the file and refs (panel_reference.py check); else verify an
# existing ranking file (IDs, row order, builder args in metadata.json, anchoring); else slim THAT domain ->
# preprocess (skipped when its outputs exist) -> build with the original args (panel_reference.py build-args) -> verify.
# A panel is accepted only when ANCHORED: byte-identical to the original panel C-CRP v3 read, or matching an anchored
# content ref. ALLOW_UNANCHORED_PANEL=1 accepts an ID-verified panel whose text is unproven (recorded in the marker).
# The first verified build freezes <d>_content.json (prompt-text fingerprints): copy it back and commit it.
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
echo "python: $(command -v "$PY")"
# pandas 2 parses the Amazon `timestamp` column as datetime64[ns]: every event key would then differ from the refs.
"$PY" -c "import pandas, sys; v = pandas.__version__; print('pandas', v); sys.exit(0 if v.split('.')[0] == '3' else 'need pandas>=3,<4 (bootstrap_server.sh pins it)')"
DOMAINS="${*:-sports toys home tools}"   # one quoted list or separate arguments
REF=scripts/sigir/panel_reference.py
UA=""
if [ "${ALLOW_UNANCHORED_PANEL:-0}" = 1 ]; then
  UA=--allow_unanchored
  echo "WARNING: ALLOW_UNANCHORED_PANEL=1: panels not proven byte-identical to the originals are accepted" >&2
fi

for d in $DOMAINS; do
  R="outputs/baselines/external_tasks/${d}_large10000_100neg_test_same_candidate/ranking_test.jsonl"
  rc=1
  if "$PY" "$REF" check --domain "$d" --ranking "$R" $UA; then
    rc=0; echo "[skip] $d: already verified"
  elif [ -f "$R" ]; then
    "$PY" "$REF" verify --domain "$d" --ranking "$R" $UA && rc=0 || rc=$?
    if [ "$rc" = 0 ]; then echo "[ok] $d: existing panel verified"; fi
  fi
  if [ "$rc" = 2 ]; then   # IDs, order and builder args match; only the bytes differ from the original
    echo "$d: $R matches every ID and builder arg but is not anchored to the original panel; rebuilding from the" \
         "same inputs would reproduce it. Investigate (data or code drift), delete it to force a rebuild, or accept" \
         "it with ALLOW_UNANCHORED_PANEL=1" >&2
    exit 1
  elif [ "$rc" != 0 ]; then
    "$PY" scripts/sigir/slim_amazon2023.py --domains "$d" --root data/raw
    P="data/processed/amazon_${d}"
    [ -f "$P/popularity_stats.csv" ] || "$PY" scripts/pipeline/main_preprocess.py --config "configs/data/amazon_${d}.yaml"
    # Cheap guard before the long build: event keys use ms timestamps (ns => a pandas-2 preprocess).
    "$PY" - "$P/interactions.csv" <<'PY'
import sys
import pandas as pd
ts = pd.read_csv(sys.argv[1], usecols=["timestamp"])["timestamp"]
if not (1e11 <= ts.min() and ts.max() < 1e14):
    sys.exit(f"{sys.argv[1]}: timestamps are not epoch-ms (min {ts.min()}, max {ts.max()}); "
             "delete that processed dir and rerun under pandas>=3")
PY
    # original args (baseline run summaries; amendment C4): tools seed 42 / history 10, others 20260506 / 50
    BA=$("$PY" "$REF" build-args --domain "$d")
    read -r SEED HIST <<<"$BA"
    echo "[build] $d: --seed $SEED --max_history_len $HIST"
    rm -f "$R.VERIFIED"
    # Remaining arguments recovered from scripts/run_week8_large_scale_10k_100neg.sh and the stored run summaries;
    # verify checks them against the builder's metadata.json (panel_reference.BUILD_ARGS).
    "$PY" scripts/build/main_build_large_scale_same_candidate_runtime.py \
      --processed_dir "$P" --domain "$d" --dataset_name "amazon_${d}" \
      --output_root outputs --exp_prefix "${d}_large10000_100neg" --user_limit 10000 \
      --num_negatives 100 --max_history_len "$HIST" --min_sequence_length 3 --seed "$SEED" \
      --shuffle_seed 42 --splits valid,test --selection_strategy random \
      --negative_sampling popularity --test_history_mode train_plus_valid
    "$PY" "$REF" verify --domain "$d" --ranking "$R" $UA
  fi
  if [ ! -f "docs/sigir/panel_refs/${d}_content.json" ]; then
    "$PY" "$REF" make-content --domain "$d" --ranking "$R" $UA
    "$PY" "$REF" verify --domain "$d" --ranking "$R" $UA
    echo "NOTE: froze docs/sigir/panel_refs/${d}_content.json from this verified panel; copy it back and commit it"
  fi
done
echo "ALL PANELS VERIFIED: $DOMAINS"
