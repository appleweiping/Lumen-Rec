#!/usr/bin/env bash
# Amendment-3 nested method slot (idea-stage/PREREG_AMENDMENT_3.md section 7, FT-M; addenda 1 and 2) for ONE dataset: the
# prior-offset LoRA (src/confrec/train_lora_offset.py) against post-hoc stacking of the section-2 SFT adapters
# (src/confrec/ftmethod_report.py). GPU server; one job at a time through scripts/sigir/gpu_queue.sh.
#   usage: bash scripts/sigir/run_ftmethod.sh D        D in ml1m, toys, games, sports (registered order, section 7)
#   env:   MODEL     backbone dir (default /root/autodl-tmp/lumen/models/Qwen3-8B); the slot is single-backbone, so any
#                    backbone other than Qwen3-8B is refused (section 7)
#          STAGES    comma list of 1-5 or all (default all)
#          VARIANT   default gate_ft_prompt of outputs/confrec/gatefix/dev/selection.json; any other value is refused
#          PYTHON    interpreter (skips the conda activation);  DRY_RUN=1  CPU rehearsal, see below
#   inputs: run_ftgrid.sh's panels outputs/confrec/ftgrid/panels/D (train.jsonl, eval.jsonl, ftgrid_split.json), the SFT
#           comparators of section 7 (i) (the section-2 adapters s0-s2; ML-1M: Gate-FT's) with their like passes
#           outputs/confrec/ftgrid/scores/D/s0-s2/like, data/raw, the Gate-FT decision and the pilot log
#   outputs (OUT_ROOT = outputs/confrec/ftmethod): D/train_qhat.csv.gz, D/eval_qhat.csv.gz, D/qhat_manifest.json (stage 1);
#           D/adapters/o0-o2 (stage 3); D/scores/o0-o2/like (stage 4); D/report.json, D/report_tables.csv, slot.json,
#           slot_tables.csv (stage 5); freeze/D.method.ok (stage 2); build/D/done (step markers)
#   stage 1  (CPU) q-hat of every TRAIN example and EVAL pair (forensics.prior_means, k = 5) and the TRAIN standardisation
#            constants: train_lora_offset qhat -> D/qhat_manifest.json (addendum 2 item 2: recorded before the dataset's
#            first prior-offset adapter is trained)
#   stage 2  the gates: Gate-FT PASS (section 0), the hard kill date 2026-11-30, then the freeze record: ftgrid_freeze
#            --check --stage amendment, --stage core (the full record: scoring an adapter needs it, section 0) and --stage
#            method --split D/qhat_manifest.json. Success writes freeze/D.method.ok; stages 3-5 refuse (exit 4) unless
#            stage 2 passed in this run or that marker equals the current record (a bound file, a split or the manifest
#            changed since the check: refused)
#   stage 3  (GPU) the prior-offset adapters o0-o2: seeds 0-2, standard mode, each with the recipe of its SFT comparator
#            (micro-batch, accumulation and max_len of its train_config.json; effective batch 32), which the trainer
#            checks again (--sft_adapter: TRAIN bytes, seed, recipe, examples); b in an AdamW group of its own at lr 1e-2
#            (Amendment 3 addendum 3: the trainer's --b_lr default, so no flag is passed here); an adapter with
#            train_config.json, offset.json and weights is never retrained
#   stage 4  (GPU) like on eval.jsonl for o0-o2: pyes_scorer --lora exactly as for the SFT adapters; E1 (censored-2 share
#            <= 0.5%, no overlength prompt, Yes+No mass >= 0.95): a failing run is moved to DIR.e1fail.<time> and rerun
#            once; a second failure leaves DIR/FAILED_INTEGRITY (the seed is missing, never replaced)
#   stage 5  (CPU) ftmethod_report dataset -> D/report.json, then ftmethod_report slot -> slot.json (the kill rule in the
#            registered order, Holm over the datasets run; addendum 1 item 8)
# Order and kill rule (section 7): before any of stages 2-5, `ftmethod_report slot --check_next D` must accept D: every
# earlier dataset of ML-1M, Toys, Video_Games, Sports has a decided report (PASS or FAIL) and fewer than 2 of them failed.
# So the slot stops after the dataset that makes the kill rule fire: the next dataset is refused (exit 4) and says why.
# Stage 1 (CPU, outcome-free) runs for any dataset, so every manifest can be recorded before the slot's first run.
# Not checked here (a scheduling decision for PILOT_LOG): section 7 runs after the main program of section 10, ML-1M
# earlier only if the GPU would otherwise idle; the section-10 checkpoint cuts of later datasets.
# Conditionality: section 7 runs only after GATE_FT_PASS (outputs/confrec/gateft/gate_ft.json); after 2026-11-30 stages 2-4
# are refused (section 7 hard kill date; section 10: unfinished items are reported as not run), stage 5 still reports.
# Re-runnable: a finished step is skipped (OUT_ROOT/build/D/done markers newer than their inputs); a scoring dir is skipped
# when report.json exists and run.key (panel sha1, model, variant, adapter-weights sha1, args) is unchanged, else it is
# moved to DIR.stale.<time>.
# DRY_RUN=1: the same chain, CPU only, on a tiny synthetic domain shaped like D. OUT_ROOT defaults to
# outputs/confrec/ftmethod_dryrun (the registered root is refused). The world is run_ftgrid.sh's own DRY_RUN world
# (outputs/confrec/ftgrid_dryrun: synthetic raw data, panels, Gate-FT context, SFT adapters s0-s2 and their like passes,
# the temporary pilot log), built first by `DRY_RUN=1 STAGES=0,1,2,3 run_ftgrid.sh D` (skipped once its SFT like passes
# exist); the trainer is a stand-in
# (OUT_ROOT/_dry/ftmethod_fakes.py: train_lora_offset's real argparse, panel check, q-hat / manifest checks, SFT-recipe
# check and PriorOffsetSet on a word tokenizer; no model) and the scorer is run_ftgrid.sh's stand-in (the real
# scorer code with a fake model); every other step is the real code, the freeze checks included (on the temporary pilot
# log). Stage 2 first shows that an empty pilot log fails the method check and that stage 3 then refuses. DRY_E1_FAIL and
# DRY_GATE act as in run_ftgrid.sh; DRY_TODAY=YYYYMMDD stands in for today's date (hard-kill-date rehearsal).
# Exit codes: 0 done; 1 error; 2 usage or refused input; 4 refused by the freeze / order / kill / gate / date rules.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
D="${1:-}"
case "$D" in
  ml1m|toys|games|sports) ;;
  *) echo "usage: bash scripts/sigir/run_ftmethod.sh {ml1m|toys|games|sports}" >&2; exit 2 ;;
esac
DRY_RUN="${DRY_RUN:-0}"
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  set +u; source /root/miniconda3/etc/profile.d/conda.sh; conda activate lumen; set -u
  PY=python
fi
REG=outputs/confrec/ftmethod          # the registered slot root
if [ "$DRY_RUN" = 1 ]; then
  MODEL="${MODEL:-dryrun/Qwen3-8B}"
  OUT_ROOT="${OUT_ROOT:-outputs/confrec/ftmethod_dryrun}"
  GRID=outputs/confrec/ftgrid_dryrun  # run_ftgrid.sh's DRY_RUN root (its synthetic world)
else
  MODEL="${MODEL:-/root/autodl-tmp/lumen/models/Qwen3-8B}"
  OUT_ROOT="${OUT_ROOT:-$REG}"
  GRID=outputs/confrec/ftgrid         # the registered (Qwen3-8B) ftgrid root
fi
OUT_ROOT="${OUT_ROOT%/}"
OUT_ROOT="${OUT_ROOT#./}"
if [ "$DRY_RUN" = 1 ] && { [ "$OUT_ROOT" = "$REG" ] || [ "$OUT_ROOT" = "$GRID" ] || [ "$OUT_ROOT" = outputs/confrec/ftgrid ]; }; then
  echo "DRY_RUN=1 never writes to a registered output root ($OUT_ROOT)" >&2; exit 2
fi
if [ "$DRY_RUN" != 1 ] && [ "$OUT_ROOT" != "$REG" ]; then
  echo "OUT_ROOT=$OUT_ROOT: the slot has one registered root, $REG (single backbone)" >&2; exit 2
fi
if [ "$(basename "$MODEL")" != Qwen3-8B ]; then
  echo "$MODEL: the method slot is single-backbone, Qwen3-8B only (section 7; a Llama replication is not registered)" >&2
  exit 2
fi
STAGES=" $(printf '%s' "${STAGES:-all}" | tr ',' ' ') "
for s in $STAGES; do
  case "$s" in 1|2|3|4|5|all) ;; *) echo "unknown stage '$s' in STAGES (1-5, all)" >&2; exit 2 ;; esac
done
case "$STAGES" in *" all "*) STAGES="$STAGES 1 2 3 4 5 " ;; esac
want() { case "$STAGES" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

P="$GRID/panels/$D"                  # run_ftgrid.sh's panels of D
SPLIT="$P/ftgrid_split.json"
SFT_S="$GRID/scores/$D"              # the SFT comparators' like passes: s0-s2/like
M="$OUT_ROOT/$D"                     # this dataset's slot directory
ADIR="$M/adapters"
S="$M/scores"
QM="$M/qhat_manifest.json"
REP="$M/report.json"
SLOT="$OUT_ROOT/slot.json"
B="$OUT_ROOT/build/$D"
DONE="$B/done"
MARK="$OUT_ROOT/freeze/$D.method.ok"
HARD_KILL=20261130                   # section 7 hard kill date
if [ "$DRY_RUN" = 1 ]; then
  DRYD="$GRID/_dry"
  G="$DRYD/gatefix"; GT="$DRYD/gateft"; RAW="$DRYD/raw"; PILOT_LOG="$DRYD/PILOT_LOG.md"
  N_BOOT=200
  TODAY="${DRY_TODAY:-$(date +%Y%m%d)}"
  CORE_ARGS=(--split "$SPLIT")
else
  G=outputs/confrec/gatefix; GT=outputs/confrec/gateft; RAW=data/raw; PILOT_LOG=docs/sigir/PILOT_LOG.md
  N_BOOT=2000                        # A3 section 3: 2,000 user resamples
  TODAY=$(date +%Y%m%d)
  CORE_ARGS=(--split "$GRID/panels/ml1m/ftgrid_split.json" --split "$GRID/panels/toys/ftgrid_split.json"
    --split "$GRID/panels/games/ftgrid_split.json" --split "$GRID/panels/sports/ftgrid_split.json")
fi
METHOD_ARGS=(--split "$QM")          # addendum 2 item 2: the dataset's q-hat manifest is part of FREEZE method
FREEZE_OK=0

# ---- helpers ----
# jget FILE KEY...: a (nested) value of a json file, '' when absent
jget() {
  "$PY" -c 'import json, sys
v = json.load(open(sys.argv[1], encoding="utf-8"))
for k in sys.argv[2:]:
    v = v.get(k) if isinstance(v, dict) else None
print("" if v is None else v)' "$@" | tr -d '\r'
}
# fresh OUT DEP...: OUT exists and is newer than every DEP (a missing DEP makes it stale)
fresh() { local o=$1 i; shift; [ -e "$o" ] || return 1; for i in "$@"; do { [ -e "$i" ] && [ "$o" -nt "$i" ]; } || return 1; done; }
# step PRODUCT DEP... -- CMD...: run CMD unless PRODUCT exists and its marker DONE/<basename>.done is newer than every DEP;
# the marker is touched only after CMD succeeds, so an interrupted step always reruns
step() {
  local prod=$1 deps=() mark; shift
  while [ "$1" != "--" ]; do deps+=("$1"); shift; done; shift
  mark="$DONE/$(basename "$prod").done"
  if [ -e "$prod" ] && fresh "$mark" "${deps[@]}"; then echo "[skip] $prod"; return 0; fi
  rm -f "$mark"
  "$@" || return 1
  mkdir -p "$DONE"; touch "$mark"
}
adapter_done() { [ -f "$1/train_config.json" ] && [ -f "$1/offset.json" ] && ls "$1"/adapter_model.* >/dev/null 2>&1; }
# e1_ok DIR: section 2 / E1 of a finished scorer run: censored-2 share <= 0.5%, no overlength prompt, mean Yes+No mass
# >= 0.95 on the main prompts
e1_ok() {
  "$PY" - "$1/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
n = int(r["n_main_prompts"])
ok = n > 0 and int(r["censored_main"].get("2", 0)) <= 0.005 * n and int(r["n_overlength"]) == 0 \
    and float(r["mean_yes_no_mass"]) >= 0.95
sys.exit(0 if ok else 1)
PY
}
# MPY runs the entry points that load the model (the trainer, the scorer). Under DRY_RUN it is fake_py: the trainer is
# this script's stand-in, the scorer run_ftgrid.sh's; any other call runs as is
fake_py() {
  if [ "$1" = -m ]; then
    case "$2" in
      src.confrec.pyes_scorer) shift 2; "$PY" "$GFAKES" scorer "$@"; return ;;
      src.confrec.train_lora_offset) shift 2; "$PY" "$MFAKES" trainer "$@"; return ;;
    esac
  fi
  "$PY" "$@"
}
MPY="$PY"
if [ "$DRY_RUN" = 1 ]; then MPY=fake_py; fi
# score DATA DIR --lora A: pyes_scorer (fp16, top-50 logprobs, max_model_len 4096, 100-user chunks, the selected variant,
# yes/no readout, like), as run_ftgrid.sh scores the SFT adapters; completion marker DIR/report.json, run.key = panel
# sha1 + model + variant + adapter weights sha1 + args
score() {
  local data=$1 dir=$2 key old="" lora="" prev="" x wsha=-; shift 2
  for x in "$@"; do if [ "$prev" = --lora ]; then lora="$x"; fi; prev="$x"; done
  if [ -n "$lora" ]; then wsha=$(cat "$lora"/adapter_model.* | sha1sum | cut -d' ' -f1); fi
  key="$(sha1sum "$data" | cut -d' ' -f1) $MODEL $VARIANT $wsha $*"
  if [ -f "$dir/run.key" ]; then old=$(cat "$dir/run.key"); fi
  if [ "$old" = "$key" ] && [ -f "$dir/report.json" ]; then echo "[skip] $dir: scored"; return 0; fi
  if [ -e "$dir" ] && [ "$old" != "$key" ]; then
    mv "$dir" "$dir.stale.$(date +%Y%m%d%H%M%S)"; echo "[moved aside] $dir (panel, model, adapter or args changed)"
  fi
  mkdir -p "$dir"
  echo "$key" > "$dir/run.key"
  "$MPY" -m src.confrec.pyes_scorer --data "$data" --output "$dir" --model "$MODEL" --dtype float16 \
    --topk_logprobs 50 --max_model_len 4096 --chunk_users 100 --variant "$VARIANT" --readout yesno \
    --questions like "$@"
  if ! e1_ok "$dir"; then
    if compgen -G "$dir.e1fail.*" > /dev/null; then
      touch "$dir/FAILED_INTEGRITY"
      echo "FAILED_INTEGRITY: $dir failed E1 twice (section 2: reported as missing, never replaced)" >&2
    else
      mv "$dir" "$dir.e1fail.$(date +%Y%m%d%H%M%S)"
      echo "[E1 failed] $dir: moved aside, rerun once (section 2)" >&2
      score "$data" "$dir" "$@"
    fi
  fi
}
# freeze_check STAGE [LOG]: 0 iff the pilot log records every sha1 the freeze stage requires (section 0; addendum 2)
freeze_check() {
  local extra=()
  case "$1" in core) extra=("${CORE_ARGS[@]}") ;; method) extra=("${METHOD_ARGS[@]}") ;; esac
  "$PY" -m src.confrec.ftgrid_freeze --check --stage "$1" --pilot_log "${2:-$PILOT_LOG}" "${extra[@]}"
}
# freeze_record: the lines the method slot runs on (the full record and FREEZE method with the dataset's manifest)
freeze_record() {
  "$PY" -m src.confrec.ftgrid_freeze --print --stage core "${CORE_ARGS[@]}"
  "$PY" -m src.confrec.ftgrid_freeze --print --stage method "${METHOD_ARGS[@]}"
}
# dry_record STAGE: DRY_RUN only, the human step: write the stage's sha1 lines into the temporary pilot log
dry_record() {
  local extra=()
  if freeze_check "$1" > /dev/null 2>&1; then return 0; fi
  case "$1" in core) extra=("${CORE_ARGS[@]}") ;; method) extra=("${METHOD_ARGS[@]}") ;; esac
  "$PY" -m src.confrec.ftgrid_freeze --print --stage "$1" "${extra[@]}" >> "$PILOT_LOG"
}
# require_freeze N: stages 3-5 need stage 2 in this run, or a marker equal to the current record
require_freeze() {
  if [ "$FREEZE_OK" = 1 ]; then return 0; fi
  if [ -f "$MARK" ] && [ "$(freeze_record 2> /dev/null)" = "$(cat "$MARK")" ]; then
    FREEZE_OK=1; return 0
  fi
  echo "stage $1 refused: the method slot waits for its freeze record (section 7; addendum 2 item 2). Run stage 2" \
    "(STAGES=2,...): $MARK is missing or no longer equals the record (a bound file, a split or the q-hat manifest" \
    "changed since the check)" >&2
  return 1
}
# gate_ft_pass N: section 0, section 7 runs only after a recorded GATE_FT_PASS
gate_ft_pass() {
  local dec=missing
  if [ -f "$GT/gate_ft.json" ]; then dec=$(jget "$GT/gate_ft.json" decision); fi
  if [ "$dec" != GATE_FT_PASS ]; then
    echo "stage $1 refused: Gate-FT decision $dec (section 0: section 7 runs only after a recorded GATE_FT_PASS," \
      "$GT/gate_ft.json)" >&2
    return 1
  fi
}
# date_check N: section 7 hard kill date 2026-11-30 (stages 2-4)
date_check() {
  if [ "$TODAY" -gt "$HARD_KILL" ]; then
    echo "stage $1 refused: the slot's hard kill date 2026-11-30 has passed (section 7; section 10: anything unfinished" \
      "is reported as not run)" >&2
    return 1
  fi
}
# method_stage N: stages 3-5 start only after stage 2's record and gates
method_stage() {
  require_freeze "$1" || exit 4
  gate_ft_pass "$1" || exit 4
  if [ "$1" != 5 ]; then date_check "$1" || exit 4; fi
}
# sft_adapter SEED: the section-7 comparator (i) of the seed: the section-2 SFT adapter (ML-1M: Gate-FT's, by its path)
sft_adapter() { if [ "$D" = ml1m ]; then echo "$GT/adapters/s$1"; else echo "$GRID/adapters/$D/s$1"; fi; }
# train_offset SEED: the prior-offset adapter oSEED with its comparator's recipe
train_offset() {
  local seed=$1 out="$ADIR/o$1" sft r micro accum maxlen
  if adapter_done "$out"; then echo "[skip] adapter $out exists"; return 0; fi
  sft=$(sft_adapter "$seed")
  [ -f "$sft/train_config.json" ] || {
    echo "missing the SFT comparator $sft (run_ftgrid.sh stage 1; ML-1M: run_gateft.sh trains it)" >&2; exit 1; }
  r=$("$PY" -c 'import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
print(c["bsz"], c["grad_accum"], c["max_len"])' "$sft/train_config.json" | tr -d '\r')
  read -r micro accum maxlen <<< "$r"
  if ! [[ "$micro" =~ ^[0-9]+$ && "$accum" =~ ^[0-9]+$ && "$maxlen" =~ ^[0-9]+$ ]] || [ $((micro * accum)) -ne 32 ]; then
    echo "$sft/train_config.json: recipe '$r' (bsz grad_accum max_len) is not a batch of 32 (section 2)" >&2; exit 1
  fi
  rm -rf "$out"
  "$MPY" -m src.confrec.train_lora_offset train --train "$P/train.jsonl" --model "$MODEL" --out "$out" \
    --variant "$VARIANT" --seed "$seed" --max_len "$maxlen" --bsz "$micro" --grad_accum "$accum" \
    --qhat "$M/train_qhat.csv.gz" --manifest "$QM" --sft_adapter "$sft" 2>&1 | grep -vE "it/s\]|s/it\]" || true
  adapter_done "$out" || { echo "training of $out did not finish" >&2; exit 1; }
}
# slot_check: the registered order and the kill rule over the earlier datasets' reports (exit 4 = refused)
slot_check() {
  local rc=0
  "$PY" -m src.confrec.ftmethod_report slot --root "$OUT_ROOT" --check_next "$D" || rc=$?
  if [ "$rc" = 4 ]; then return 4; fi
  if [ "$rc" != 0 ]; then echo "the slot check failed (exit $rc)" >&2; exit 1; fi
}

# ---- the order and the kill rule come first: a refused dataset touches nothing ----
GATED=0
for s in 2 3 4 5; do if want "$s"; then GATED=1; fi; done
if [ "$GATED" = 1 ]; then slot_check || exit 4; fi

# ---- DRY_RUN: run_ftgrid.sh's synthetic world and the trainer stand-in ----
if [ "$DRY_RUN" = 1 ]; then
  mkdir -p "$OUT_ROOT/_dry"
  GFAKES="$GRID/_dry/ftgrid_fakes.py"
  if [ -f "$SPLIT" ] && [ -f "$GFAKES" ] && [ -f "$SFT_S/s0/like/report.json" ] && [ -f "$SFT_S/s1/like/report.json" ] \
      && [ -f "$SFT_S/s2/like/report.json" ]; then
    echo "[skip] run_ftgrid.sh's DRY_RUN world for $D exists ($GRID)"
  else
    echo "== DRY_RUN: run_ftgrid.sh's synthetic world for $D (its stages 0-3: panels, SFT adapters, their like passes)"
    if ! DRY_RUN=1 OUT_ROOT="$GRID" MODEL="$MODEL" STAGES=0,1,2,3 PYTHON="$PY" bash scripts/sigir/run_ftgrid.sh "$D" \
        > "$OUT_ROOT/_dry/ftgrid.$D.log" 2>&1; then
      tail -n 40 "$OUT_ROOT/_dry/ftgrid.$D.log" >&2
      echo "DRY_RUN: run_ftgrid.sh $D failed (log: $OUT_ROOT/_dry/ftgrid.$D.log)" >&2; exit 1
    fi
  fi
  MFAKES="$OUT_ROOT/_dry/ftmethod_fakes.py"
  cat > "$MFAKES.tmp" <<'PYFAKES'
"""DRY_RUN stand-in of scripts/sigir/run_ftmethod.sh (written by it; CPU only, no model, no GPU, no network).

    python ftmethod_fakes.py trainer train <the train_lora_offset train arguments>

trainer  train_lora_offset's own argparse, train_lora_yesno.check_training_panel, load_train_z (the q-hat file and
         train.jsonl against the manifest), sft_comparator (the SFT comparator's recipe and TRAIN file) and the real
         PriorOffsetSet on a word tokenizer whose "Yes" and "No" are single tokens; torch is replaced by stand-in
         modules (nothing is trained). Writes a placeholder adapter_model.safetensors and adapter_config.json and, through
         the module's own writers, train_report.json, offset.json and train_config.json with a fixed stand-in b
         (0.05 x (seed + 1)) and dry_run true.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import types
from pathlib import Path


class Tok:
    """Word tokenizer: "Yes" -> 1, "No" -> 2, every other space-separated piece its own id (a prompt is a few hundred
    pieces, so the comparator's max_len keeps the examples). The chat template wraps the messages and, with thinking
    off, ends in the closed empty think block (amendment 2 G0)."""
    pad_token_id = 0
    eos_token = "<|end|>"

    def __init__(self):
        self.ids, self.text = {"Yes": 1, "No": 2}, {0: "", 1: "Yes", 2: "No"}

    def __len__(self):
        return 1 + len(self.ids)

    def decode(self, ids):
        return " ".join(self.text.get(i, "") for i in ids)

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, enable_thinking=True, **kw):
        out = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in msg) + "<|assistant|>"
        return out + ("" if enable_thinking else "<think>\n\n</think>\n\n")

    def __call__(self, text, add_special_tokens=False):
        out = []
        for w in text.split(" "):
            if w not in self.ids:
                self.ids[w] = len(self.ids) + 1
                self.text[self.ids[w]] = w
            out.append(self.ids[w])
        return {"input_ids": out}


def stub_torch() -> None:
    """train_lora_yesno imports torch at module level; its dataset and panel check need none of it here."""
    for name in ("torch", "torch.utils", "torch.utils.data", "torch.nn", "torch.nn.functional"):
        sys.modules.setdefault(name, types.ModuleType(name))
    if not hasattr(sys.modules["torch.utils.data"], "Dataset"):
        sys.modules["torch.utils.data"].Dataset = object


def trainer(argv) -> None:
    stub_torch()
    from src.confrec import train_lora_offset as tlo
    from src.confrec import train_lora_yesno as tl
    a = tlo.parse_args(argv)
    if a.cmd != "train":
        raise SystemExit("the stand-in replaces the train subcommand only")
    rows = tlo.read_rows(a.train)
    random.Random(a.seed).shuffle(rows)
    panel = tl.check_training_panel(rows, a.variant, a.hist_len, a.train)
    zinfo = tlo.load_train_z(a.train, a.qhat, a.manifest)
    sft = tlo.sft_comparator(a, panel["hist_len_used"])
    ds = tlo.prior_offset_set(rows, zinfo["z_of"], Tok(), a.hist_len, a.max_len, a.variant)
    n_built = len(ds)
    if a.max_examples:
        ds.truncate(a.max_examples)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tlo.write_train_report(out, a, panel, ds, n_built, zinfo, dry_run=True)
    digest = hashlib.sha1(Path(a.train).read_bytes() + f"|{a.seed}|{a.variant}|{a.model}|offset".encode()).hexdigest()
    (out / "adapter_model.safetensors").write_text(f"DRY_RUN placeholder prior-offset adapter {digest}\n",
                                                   encoding="utf-8")
    (out / "adapter_config.json").write_text(json.dumps(
        {"dry_run": True, "base_model_name_or_path": a.model, "r": a.lora_r, "lora_alpha": 2 * a.lora_r,
         "lora_dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}, indent=2), encoding="utf-8")
    b = round(0.05 * (a.seed + 1), 6)
    tlo.write_offset_and_config(out, a, panel, ds, zinfo, sft, b, dry_run=True)
    print(f"dry-run prior-offset trainer: {out} ({len(ds)} examples, {ds.n_skipped} skipped, b {b}, bsz {a.bsz} x "
          f"{a.grad_accum}, max_len {a.max_len})")


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    {"trainer": trainer}[sys.argv[1]](sys.argv[2:])
PYFAKES
  if cmp -s "$MFAKES.tmp" "$MFAKES"; then rm -f "$MFAKES.tmp"; else mv "$MFAKES.tmp" "$MFAKES"; fi
fi

# ---- the prompt ----
SEL="$G/dev/selection.json"
[ -f "$SEL" ] || { echo "missing $SEL (the gate-fix stage 1 selection; run_gatefix.sh writes it)" >&2; exit 1; }
SEL_PROMPT=$(jget "$SEL" gate_ft_prompt)
VARIANT="${VARIANT:-$SEL_PROMPT}"
if [ "$VARIANT" != "$SEL_PROMPT" ] && [ "$DRY_RUN" != 1 ]; then
  echo "VARIANT=$VARIANT is not selection.json's gate_ft_prompt $SEL_PROMPT: section 2 registers no other variant" >&2
  exit 2
fi
if [ -f "$SPLIT" ] && [ "$(jget "$SPLIT" variant)" != "$VARIANT" ]; then
  echo "$SPLIT was built under variant $(jget "$SPLIT" variant), not $VARIANT" >&2; exit 2
fi
echo "run_ftmethod $D: model $MODEL, root $OUT_ROOT, variant $VARIANT, stages:$STAGES"

# ================= stage 1: q-hat and the TRAIN standardisation (CPU) =================
if want 1; then
  echo "== stage 1: q-hat and the TRAIN standardisation constants ($D)"
  for f in "$P/train.jsonl" "$P/eval.jsonl" "$SPLIT"; do
    [ -f "$f" ] || { echo "missing $f (run_ftgrid.sh stage 0)" >&2; exit 1; }
  done
  mkdir -p "$M"
  step "$QM" "$P/train.jsonl" "$P/eval.jsonl" "$SPLIT" src/confrec/train_lora_offset.py src/confrec/forensics.py \
      src/confrec/build_rated_panels.py -- \
    "$PY" -m src.confrec.train_lora_offset qhat --domain "$D" --panels "$P" --raw "$RAW" --out_dir "$M"
fi

# ================= stage 2: gates and the freeze record =================
if want 2; then
  echo "== stage 2: gates and the freeze record ($D)"
  [ -f "$QM" ] || { echo "missing $QM (stage 1)" >&2; exit 1; }
  gate_ft_pass 2 || exit 4
  date_check 2 || exit 4
  if [ "$DRY_RUN" = 1 ]; then
    # rehearsal: an empty pilot log fails the method check, and stage 3 then refuses; the human step is dry_record
    [ -f "$OUT_ROOT/_dry/PILOT_LOG.empty.md" ] || printf '# empty temporary pilot log\n' > "$OUT_ROOT/_dry/PILOT_LOG.empty.md"
    if freeze_check method "$OUT_ROOT/_dry/PILOT_LOG.empty.md" 2> /dev/null; then
      echo "DRY_RUN: the method freeze check passed on an empty pilot log" >&2; exit 1
    fi
    if (MARK="$OUT_ROOT/_dry/no.method.ok"; require_freeze 3) 2> /dev/null; then
      echo "DRY_RUN: stage 3 did not refuse without the freeze record" >&2; exit 1
    fi
    echo "[dry] freeze rehearsal: an empty pilot log fails the method check and stage 3 refuses; recording the temporary log"
    dry_record amendment
    dry_record core
    dry_record method
  fi
  for st in amendment core method; do
    if ! freeze_check "$st"; then
      echo "stage 2: the record of freeze stage $st is not in $PILOT_LOG; record the lines of 'python -m" \
        "src.confrec.ftgrid_freeze --print --stage $st' (method: with --split $QM) and rerun (stage 1 is kept)" >&2
      exit 4
    fi
  done
  REC=$(freeze_record)
  mkdir -p "$(dirname "$MARK")"
  if [ ! -f "$MARK" ] || [ "$(cat "$MARK")" != "$REC" ]; then printf '%s\n' "$REC" > "$MARK"; fi
  FREEZE_OK=1
fi

# ================= stage 3: the prior-offset adapters (GPU) =================
if want 3; then
  echo "== stage 3: prior-offset adapters o0-o2 ($D)"
  method_stage 3
  for seed in 0 1 2; do train_offset "$seed"; done
fi

# ================= stage 4: like on eval.jsonl (GPU) =================
if want 4; then
  echo "== stage 4: like on eval.jsonl for o0-o2 ($D)"
  method_stage 4
  for seed in 0 1 2; do
    adapter_done "$ADIR/o$seed" || { echo "adapter o$seed of $D is missing or incomplete (stage 3)" >&2; exit 1; }
    score "$P/eval.jsonl" "$S/o$seed/like" --lora "$ADIR/o$seed"
  done
fi

# ================= stage 5: the report and the slot state (CPU) =================
if want 5; then
  echo "== stage 5: report and slot state ($D)"
  method_stage 5
  RDEPS=("$SPLIT" "$P/eval.jsonl" "$QM" src/confrec/ftmethod_report.py src/confrec/ftgrid_report.py
    src/confrec/train_lora_offset.py)
  for f in "$SFT_S"/s[012]/like/report.json "$S"/o[012]/like/report.json "$ADIR"/o[012]/offset.json; do
    if [ -f "$f" ]; then RDEPS+=("$f"); fi
  done
  step "$REP" "${RDEPS[@]}" -- \
    "$PY" -m src.confrec.ftmethod_report dataset --domain "$D" --split "$SPLIT" --panels "$P" --sft_scores "$SFT_S" \
      --method_dir "$M" --out "$REP" --n_boot "$N_BOOT" --seed 0
  "$PY" -m src.confrec.ftmethod_report slot --root "$OUT_ROOT" --out "$SLOT"
fi
echo "run_ftmethod $D: done (stages:$STAGES)"
