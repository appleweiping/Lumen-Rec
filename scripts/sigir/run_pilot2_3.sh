#!/usr/bin/env bash
# Pilots 2 (KuaiRec MAR vs logged calibration) and 3 (pseudonym knockout) — triage_verdict.md §3.2/§3.3.
#   usage: bash scripts/sigir/run_pilot2_3.sh [2|3|all]
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
MODEL="${MODEL:-/root/autodl-tmp/lumen/models/Qwen3-8B}"
PANELS=outputs/confrec/panels
WHICH="${1:-all}"
mkdir -p $PANELS data/raw

if [ "$WHICH" = 2 ] || [ "$WHICH" = all ]; then
  OUT=outputs/confrec/pilot2_kuairec
  if [ ! -d data/raw/KuaiRec ]; then
    # Official release (Gao et al., CIKM'22), verified in github.com/chongminggao/KuaiRec README:
    # Zenodo record 18164998; captions are a separate file in the same record.
    mkdir -p data/raw/KuaiRec
    wget -q -O data/raw/KuaiRec.zip https://zenodo.org/records/18164998/files/KuaiRec.zip
    (cd data/raw/KuaiRec && unzip -q ../KuaiRec.zip)
  fi
  KDIR=$(dirname "$(find data/raw/KuaiRec -name small_matrix.csv | head -1)")
  [ -f "$KDIR/kuairec_caption_category.csv" ] || \
    wget -q -O "$KDIR/kuairec_caption_category.csv" https://zenodo.org/records/18164998/files/kuairec_caption_category.csv
  python -m src.confrec.build_kuairec_panel --root "$KDIR" --out $PANELS/kuairec.jsonl --n_users 300 --n_cands 200
  python -m src.confrec.pyes_scorer --data $PANELS/kuairec.jsonl --output $OUT --model "$MODEL" --questions like --hist_len 10
  python -m src.confrec.pilot_kuairec --scores $OUT/scores.csv.gz --panel $PANELS/kuairec.jsonl --out $OUT/pilot_kuairec.json
fi

if [ "$WHICH" = 3 ] || [ "$WHICH" = all ]; then
  OUT=outputs/confrec/pilot3_pseudonym
  for d in toys beauty; do
    [ -f $PANELS/${d}_rated.jsonl ] || python -m src.confrec.build_rated_panels --source amazon --domain $d \
        --raw data/raw --out $PANELS/${d}_rated.jsonl --n_users 1500
    python -m src.confrec.pseudonymize --panel $PANELS/${d}_rated.jsonl --out_dir $PANELS
    for v in rated rated_pseudo rated_placebo; do
      python -m src.confrec.pyes_scorer --data $PANELS/${d}_${v}.jsonl --output $OUT/${d}_${v} --model "$MODEL" \
        --questions like --hist_len 10
    done
    python -m src.confrec.pilot_pseudonym --real $OUT/${d}_rated/scores.csv.gz --pseudo $OUT/${d}_rated_pseudo/scores.csv.gz \
      --placebo $OUT/${d}_rated_placebo/scores.csv.gz --panel $PANELS/${d}_rated_pseudo.jsonl --out $OUT/${d}_pilot_pseudonym.json
  done
fi
