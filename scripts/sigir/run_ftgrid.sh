#!/usr/bin/env bash
# Amendment-3 fine-tuned program for ONE domain (idea-stage/PREREG_AMENDMENT_3.md sections 0-5, 8, 9; interfaces:
# docs/sigir/FTGRID_IMPL_SPEC.md). GPU server; one job at a time through scripts/sigir/gpu_queue.sh.
#   usage: bash scripts/sigir/run_ftgrid.sh D                  D in ml1m, toys, games, sports
#   env:   MODEL     backbone dir (default /root/autodl-tmp/lumen/models/Qwen3-8B). Another backbone (Llama-3.1-8B-
#                    Instruct, section 8: ML-1M and Toys only) needs its own OUT_ROOT
#          OUT_ROOT  default outputs/confrec/ftgrid (Llama: outputs/confrec/ftgrid_llama)
#          VARIANT   default gate_ft_prompt of outputs/confrec/gatefix/dev/selection.json; any other value is refused
#                    (section 2: no other variant)
#          STAGES    comma list of 0-6, all (= 0-6) and perm (default all; perm is never implied)
#          KNOCKOUT_SPORTS=1  also the Sports knockout (section 5: run unless cut at the 2026-10-22 checkpoint)
#          PYTHON    interpreter (skips the conda activation);  DRY_RUN=1  CPU rehearsal, see below
#   stage 0  data (CPU): the full h20 rated panel of D in row order (ml1m / toys: the frozen gate-fix DEV + CONFIRM
#            panels joined, which their manifest records as a prefix split of the rebuilt panel; games / sports:
#            build_rated_panels --hist_len 20 --gatefix_fields with build_confirm_panels.py's arguments)
#            -> ftgrid_data -> panels/D/; then the K = 2 star-permuted S_d panels (starperm_panel.py) and, for toys /
#            games, pseudonymize.py's pseudo / placebo panels of eval.jsonl, made next to the category-wide brand
#            popularity sidecar as in Pilot 3 (toys: the Pilot-1 panel's; games / sports: stage 0's build's)
#   stage 1  LoRA seeds 0-2 on panels/D/train.jsonl with micro_bsz, grad_accum and max_len_used of ftgrid_split.json
#            (micro-batch x accumulation = 32); ML-1M on the Gate-FT backbone: the Gate-FT adapters, linked and scored
#            by their path, never retrained. An adapter with train_config.json and weights is never retrained.
#   perm     ML-1M, Gate-FT backbone: the FT-C adapters p0, p1 (seeds 0, 1) on train_perm.jsonl (section 9); their
#            like pass is stage 3's (FT-C's endpoints are like-pass UAUCs)
#   stage 2  freeze check `ftgrid_freeze --check --stage core` (section 0: the bound code and the four registered
#            ftgrid_split.json; under another OUT_ROOT also that root's split). Success writes OUT_ROOT/freeze/D.core.ok
#            (the printed record). Stages 3-5 refuse (exit 4) unless stage 2 passed in this run or that marker equals
#            the current record (a bound file or split changed since the check: refused).
#   stage 3  like on panels/D/eval.jsonl (CAL and TEST rows) for zeroshot, s0-s2 and p0 / p1 once trained. ML-1M on
#            the Gate-FT backbone: an identical earlier pass (zeroshot: gate-fix stage 2, else Gate-FT stage C; s0-s2:
#            Gate-FT stage C) is reused when pyes_scorer accepts its report.json as this run (section 4)
#   stage 4  decomposition arms on eval_sd_test.jsonl for zeroshot and s0-s2: swap (--swap_k 8), nohist (--hist_len 0),
#            starperm0, starperm1
#   stage 5  knockout arms pseudo, placebo (toys, games; sports with KNOCKOUT_SPORTS=1) for zeroshot and s0-s2
#   stage 6  ftgrid_report -> OUT_ROOT/report/D.json (and its tables)
# Conditionality (section 0): adapters, their scoring and the knockout run only after outputs/confrec/gateft/gate_ft.json
# records GATE_FT_PASS; otherwise only the zero-shot panels of section 4 run (like + decomposition arms). Any scoring also
# waits for the gate-fix decisions (selection.json; gate.json when a fix was found). Training needs the amendment's sha1
# in docs/sigir/PILOT_LOG.md, scoring the full record (stage 2). Section 10 fixes the order across domains: follow it with
# STAGES (e.g. the Toys knockout, STAGES=5,6, after the Llama ML-1M fine-tuning).
# Re-runnable: a finished step is skipped (OUT_ROOT/build/D/done markers newer than their inputs); a scoring dir is
# skipped when report.json exists and run.key (panel sha1, model, variant, adapter-weights sha1, args) is unchanged,
# else moved to DIR.stale.<time>. Section 2 / E1 (censored-2 share <= 0.5%, no overlength prompt, Yes+No mass >= 0.95):
# a failing run is moved to DIR.e1fail.<time> and rerun once; a second failure leaves DIR/FAILED_INTEGRITY.
# DRY_RUN=1: the same chain, CPU only, on a tiny synthetic domain shaped like D. OUT_ROOT defaults to
# outputs/confrec/ftgrid_dryrun (the registered roots are refused); the inputs live in OUT_ROOT/_dry (synthetic raw files,
# gate-fix panels and decisions, the Gate-FT context, a temporary pilot log); pyes_scorer and train_lora_yesno are
# replaced by stand-ins (OUT_ROOT/_dry/ftgrid_fakes.py: their real argparse, the real scorer code with a fake model, a
# placeholder adapter); every other step is the real code. Stage 2 first shows that an empty pilot log fails the check
# and that stage 3 then refuses. DRY_GATE=GATE_FT_FAIL rehearses the zero-shot-only branch, DRY_E1_FAIL=<text> makes the
# runs whose output dir contains <text> fail E1.
# Exit codes: 0 done; 1 error; 2 usage or refused input; 4 refused by the freeze / stage-order / gate rules.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
D="${1:-}"
case "$D" in
  ml1m|toys|games|sports) ;;
  *) echo "usage: bash scripts/sigir/run_ftgrid.sh {ml1m|toys|games|sports}" >&2; exit 2 ;;
esac
DRY_RUN="${DRY_RUN:-0}"
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  set +u; source /root/miniconda3/etc/profile.d/conda.sh; conda activate lumen; set -u
  PY=python
fi
REG=outputs/confrec/ftgrid      # the registered (Qwen3-8B) root: its four ftgrid_split.json are the section-0 record
if [ "$DRY_RUN" = 1 ]; then
  MODEL="${MODEL:-dryrun/Qwen3-8B}"
  OUT_ROOT="${OUT_ROOT:-outputs/confrec/ftgrid_dryrun}"
else
  MODEL="${MODEL:-/root/autodl-tmp/lumen/models/Qwen3-8B}"
  OUT_ROOT="${OUT_ROOT:-$REG}"
fi
OUT_ROOT="${OUT_ROOT%/}"
OUT_ROOT="${OUT_ROOT#./}"
if [ "$DRY_RUN" = 1 ] && { [ "$OUT_ROOT" = "$REG" ] || [ "$OUT_ROOT" = outputs/confrec/ftgrid_llama ]; }; then
  echo "DRY_RUN=1 never writes to a registered output root ($OUT_ROOT)" >&2; exit 2
fi
QWEN=0
if [ "$(basename "$MODEL")" = Qwen3-8B ]; then QWEN=1; fi
if [ "$QWEN" != 1 ] && [ "$OUT_ROOT" = "$REG" ]; then
  echo "$MODEL is not the Gate-FT backbone Qwen3-8B: it writes to its own OUT_ROOT (section 2), e.g. outputs/confrec/ftgrid_llama" >&2
  exit 2
fi
if [ "$QWEN" != 1 ] && { [ "$D" = games ] || [ "$D" = sports ]; }; then
  echo "$MODEL on $D: the second backbone runs on ML-1M and Toys only (sections 4 and 8)" >&2; exit 2
fi
DEF_STAGES=all
if [ "$DRY_RUN" = 1 ] && [ "$D" = ml1m ] && [ "$QWEN" = 1 ]; then DEF_STAGES=all,perm; fi
STAGES=" $(printf '%s' "${STAGES:-$DEF_STAGES}" | tr ',' ' ') "
for s in $STAGES; do
  case "$s" in 0|1|2|3|4|5|6|all|perm) ;; *) echo "unknown stage '$s' in STAGES (0-6, all, perm)" >&2; exit 2 ;; esac
done
case "$STAGES" in *" all "*) STAGES="$STAGES 0 1 2 3 4 5 6 " ;; esac
want() { case "$STAGES" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

P="$OUT_ROOT/panels/$D"             # ftgrid_data's panels (+ the derived star-permutation / knockout panels)
S="$OUT_ROOT/scores/$D"             # scores/D/<model>/<arm>/
ADIR="$OUT_ROOT/adapters/$D"
B="$OUT_ROOT/build/$D"              # this script's working files: the full panel, knockout work dir, step markers
DONE="$B/done"
SPLIT="$P/ftgrid_split.json"
MARK="$OUT_ROOT/freeze/$D.core.ok"
REP="$OUT_ROOT/report/$D.json"
ALL="$B/${D}_all_h20.jsonl"
if [ "$DRY_RUN" = 1 ]; then
  DRY="$OUT_ROOT/_dry"
  G="$DRY/gatefix"; GT="$DRY/gateft"; RAW="$DRY/raw"; PILOT_PANELS="$DRY/pilot_panels"; PILOT_LOG="$DRY/PILOT_LOG.md"
  N_TRAIN=40; N_EVAL=60; S_MAX=25; N_BOOT=200
else
  G=outputs/confrec/gatefix; GT=outputs/confrec/gateft; RAW=data/raw; PILOT_PANELS=outputs/confrec/panels
  PILOT_LOG=docs/sigir/PILOT_LOG.md
  N_TRAIN=1500; N_EVAL=3000; S_MAX=1000; N_BOOT=2000     # A3 section 1 / 3 constants
fi
KO=0                                 # section 5 knockout domain
if [ "$D" = toys ] || [ "$D" = games ] || { [ "$D" = sports ] && [ "${KNOCKOUT_SPORTS:-0}" = 1 ]; }; then KO=1; fi
ADOPT=""
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
adapter_done() { [ -f "$1/train_config.json" ] && ls "$1"/adapter_model.* >/dev/null 2>&1; }
# e1_ok DIR: section 2 / E1 on the main prompts of a finished scorer run
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
# fake_py: DRY_RUN's interpreter for the two GPU entry points (stand-ins); everything else runs as is
fake_py() {
  if [ "$1" = -m ]; then
    case "$2" in
      src.confrec.pyes_scorer) shift 2; "$PY" "$FAKES" scorer "$@"; return ;;
      src.confrec.train_lora_yesno) shift 2; "$PY" "$FAKES" trainer "$@"; return ;;
    esac
  fi
  "$PY" "$@"
}
GPY="$PY"
if [ "$DRY_RUN" = 1 ]; then GPY=fake_py; fi
# score DATA DIR [--lora A] ARGS...: pyes_scorer (fp16, top-50 logprobs, max_model_len 4096, 100-user chunks, the selected
# variant, yes/no readout, like); completion marker DIR/report.json, run.key = panel sha1 + model + variant + adapter
# weights sha1 + args. ADOPT=<finished dir> (set just before the call) is copied in first and kept only if the scorer
# accepts its report.json as this exact run ("already complete"); otherwise it is dropped and DIR is scored.
score() {
  local data=$1 dir=$2 adopt="$ADOPT" key old="" lora="" prev="" x wsha=-; shift 2
  ADOPT=""
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
  if [ -n "$adopt" ] && [ -f "$adopt/report.json" ] && [ -f "$adopt/scores.csv.gz" ] && [ ! -f "$dir/report.json" ]; then
    cp "$adopt/report.json" "$adopt/scores.csv.gz" "$dir/"
    echo "$adopt" > "$dir/adopted_from"
  fi
  until "$GPY" -m src.confrec.pyes_scorer --data "$data" --output "$dir" --model "$MODEL" --dtype float16 \
      --topk_logprobs 50 --max_model_len 4096 --chunk_users 100 --variant "$VARIANT" --readout yesno \
      --questions like "$@"; do
    [ -f "$dir/adopted_from" ] || return 1
    echo "[not adopted] $(cat "$dir/adopted_from") is not this run: scoring $dir" >&2
    rm -f "$dir/report.json" "$dir/scores.csv.gz" "$dir/adopted_from"
  done
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
# freeze_check STAGE: 0 iff the pilot log records every sha1 the freeze stage requires (section 0)
freeze_check() {
  "$PY" -m src.confrec.ftgrid_freeze --check --stage "$1" --pilot_log "$PILOT_LOG" "${CORE_SPLITS[@]}"
}
# dry_record STAGE: DRY_RUN only, the human step: write the stage's sha1 lines into the temporary pilot log
dry_record() {
  if freeze_check "$1" > /dev/null 2>&1; then return 0; fi
  "$PY" -m src.confrec.ftgrid_freeze --print --stage "$1" "${CORE_SPLITS[@]}" >> "$PILOT_LOG"
}
# gpu_freeze: no GPU job of the amendment before its sha1 is in the pilot log (training; section 0)
gpu_freeze() {
  if [ "$DRY_RUN" = 1 ]; then dry_record amendment; fi
  freeze_check amendment || exit 4
}
# require_freeze STAGE: scoring needs stage 2 in this run, or a marker equal to the current record
require_freeze() {
  if [ "$FREEZE_OK" = 1 ]; then return 0; fi
  if [ -f "$MARK" ] && [ "$("$PY" -m src.confrec.ftgrid_freeze --print --stage core "${CORE_SPLITS[@]}" 2> /dev/null)" = "$(cat "$MARK")" ]; then
    FREEZE_OK=1; return 0
  fi
  echo "stage $1 refused: scoring waits for the full Amendment-3 record (section 0). Run stage 2 (STAGES=2,...): $MARK is" \
    "missing or no longer equals the record (a bound file or ftgrid_split.json changed since the check)" >&2
  return 1
}
# scoring_stage N: stages 3-5 start only after the freeze check and the gate-fix decisions (sections 0 and 4)
scoring_stage() {
  require_freeze "$1" || exit 4
  if [ "$SEL_DECISION" = FIX_FOUND ] && [ ! -f "$G/confirm/gate.json" ]; then
    echo "stage $1 refused: selection.json found a fix but the gate-fix confirm stage has recorded no gate.json (section 4)" >&2
    exit 4
  fi
}
# recipe: MICRO ACCUM MAXLEN of ftgrid_split.json (section 1 length rule; micro-batch x accumulation = 32)
recipe() {
  local r
  r=$("$PY" -c 'import json, sys
o = json.load(open(sys.argv[1], encoding="utf-8"))["train"]["overlength"]
print(o["micro_bsz"], o["grad_accum"], o["max_len_used"])' "$SPLIT" | tr -d '\r')
  read -r MICRO ACCUM MAXLEN <<< "$r"
  if ! [[ "$MICRO" =~ ^[0-9]+$ && "$ACCUM" =~ ^[0-9]+$ && "$MAXLEN" =~ ^[0-9]+$ ]] || [ $((MICRO * ACCUM)) -ne 32 ]; then
    echo "$SPLIT: training recipe '$r' (micro_bsz grad_accum max_len_used) is not a batch of 32" >&2; exit 1
  fi
}
# train_adapter TRAIN OUT SEED: the section-2 recipe; skipped when train_config.json and adapter weights exist
train_adapter() {
  local train=$1 out=$2 seed=$3
  if adapter_done "$out"; then echo "[skip] adapter $out exists"; return 0; fi
  [ -f "$train" ] || { echo "missing $train (stage 0)" >&2; exit 1; }
  rm -rf "$out"
  "$GPY" -m src.confrec.train_lora_yesno --train "$train" --model "$MODEL" --out "$out" --variant "$VARIANT" \
    --seed "$seed" --max_len "$MAXLEN" --bsz "$MICRO" --grad_accum "$ACCUM" 2>&1 | grep -vE "it/s\]|s/it\]" || true
  adapter_done "$out" || { echo "training of $out did not finish" >&2; exit 1; }
}
# link_gateft SEED: ML-1M on the Gate-FT backbone: the Gate-FT adapter is this program's (section 2), never retrained
link_gateft() {
  local src="$GT/adapters/s$1" dst="$ADIR/s$1"
  adapter_done "$src" || { echo "Gate-FT adapter $src is missing or incomplete (run_gateft.sh trains it; never here)" >&2; exit 1; }
  if [ "$(basename "$(jget "$src/train_config.json" model)")" != "$(basename "$MODEL")" ] \
      || [ "$(jget "$src/train_config.json" variant)" != "$VARIANT" ]; then
    echo "$src was not trained on $(basename "$MODEL") with variant $VARIANT (see its train_config.json)" >&2; exit 1
  fi
  if [ ! -e "$dst" ]; then mkdir -p "$ADIR"; ln -sfn "$(cd "$src" && pwd -P)" "$dst"; fi
  echo "[link] $dst -> $src (Gate-FT adapter, scored by its path)"
}
# adapter_of MODEL: the adapter dir scored for a fine-tuned model
adapter_of() {
  if [ "$D" = ml1m ] && [ "$QWEN" = 1 ] && [ "${1#s}" != "$1" ]; then echo "$GT/adapters/$1"; else echo "$ADIR/$1"; fi
}
# set_lora MODEL: LORA = the scorer's adapter arguments of MODEL (none for zeroshot); a missing adapter is an error
set_lora() {
  LORA=()
  if [ "$1" = zeroshot ]; then return 0; fi
  local a
  a=$(adapter_of "$1")
  adapter_done "$a" || { echo "adapter $1 of $D is missing or incomplete: $a (stage 1 / perm)" >&2; exit 1; }
  LORA=(--lora "$a")
}
# adopt_source MODEL: ML-1M's identical earlier like pass (section 4), on the Gate-FT backbone only
adopt_source() {
  if [ "$D" != ml1m ] || [ "$QWEN" != 1 ]; then return 0; fi
  case "$1" in
    zeroshot) if [ -f "$G/confirm/ml1m/$VARIANT/report.json" ]; then echo "$G/confirm/ml1m/$VARIANT"; else echo "$GT/scores/zeroshot"; fi ;;
    s0|s1|s2) echo "$GT/scores/$1" ;;
  esac
}
# like_models: zeroshot, s0-s2 after GATE_FT_PASS, and p0 p1 once both FT-C adapters exist
like_models() {
  local m="$MODELS"
  if [ "$FT_OK" = 1 ] && [ "$D" = ml1m ] && [ "$QWEN" = 1 ] && adapter_done "$ADIR/p0" && adapter_done "$ADIR/p1"; then
    m="$m p0 p1"
  fi
  echo "$m"
}
# join_h20 DEV CONFIRM: the full rebuilt h20 panel = the DEV rows, then the CONFIRM rows; the gate-fix manifest must
# record a prefix split (DEV = the first 1,500 rows) and both files must be its recorded bytes
join_h20() {
  "$PY" - "$G/panels/manifest.json" "$D" "$1" "$2" <<'PY'
import hashlib, json, sys
man, src, dev, conf = sys.argv[1:5]
rec = json.load(open(man, encoding="utf-8"))["sources"][src]
if rec.get("prefix_match") is not True or rec.get("split_method") != "prefix":
    sys.exit(f"{man}: {src} was not split by row prefix, so DEV + CONFIRM is not the rebuilt panel's row order; build the "
             "full panel with build_rated_panels --hist_len 20 --gatefix_fields instead")
for path, split in ((dev, "dev"), (conf, "confirm")):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != rec[split]["sha1"]:
        sys.exit(f"{path}: sha1 {h.hexdigest()} is not the manifest's {split} panel {rec[split]['sha1']}")
PY
  cat "$1" "$2" > "$ALL.tmp"
  mv "$ALL.tmp" "$ALL"
}
# knockout_panels BRAND_POP: pseudonymize.py (unchanged) on a byte copy of eval.jsonl next to the category-wide brand
# popularity sidecar (as Pilot 3 ran it); the pseudo / placebo panels then move to panels/D
knockout_panels() {
  local ko="$B/knockout"
  mkdir -p "$ko"
  cp "$P/eval.jsonl" "$ko/eval.jsonl"
  cp "$1" "$ko/eval.brand_pop.json"
  "$PY" -m src.confrec.pseudonymize --panel "$ko/eval.jsonl" --seed 0 > "$ko/pseudonymize.log"
  mv "$ko/eval_pseudo.jsonl" "$P/eval_pseudo.jsonl"
  mv "$ko/eval_placebo.jsonl" "$P/eval_placebo.jsonl"
}

# ---- DRY_RUN: the synthetic world and the stand-ins ----
if [ "$DRY_RUN" = 1 ]; then
  mkdir -p "$DRY"
  FAKES="$DRY/ftgrid_fakes.py"
  cat > "$FAKES.tmp" <<'PYFAKES'
"""DRY_RUN stand-ins of scripts/sigir/run_ftgrid.sh (written by it; CPU only, no model, no GPU, no network).

    python ftgrid_fakes.py setup --domain D --dry DIR --gatefix G --gateft GT --variant V --n_dev N --gate DECISION
    python ftgrid_fakes.py scorer <the pyes_scorer arguments>
    python ftgrid_fakes.py trainer <the train_lora_yesno arguments>

setup    the synthetic world of one domain shaped like D: raw files in the real layouts under DIR/raw; for ml1m and toys
         the gate-fix h20 DEV / CONFIRM panels (build_rated_panels.build on that raw, DEV = the first N rows) and their
         manifest; for toys the Pilot-1 brand popularity sidecar; selection.json (FIX_FOUND, gate_ft_prompt V) and
         gate.json; for ml1m the Gate-FT context (gateft_data on the dry panels, fake adapters s0-s2 and their fake like
         passes, with run_gateft.sh's arguments); gate_ft.json = DECISION; the temporary pilot log. An existing file is
         never rewritten, so a rerun changes nothing.
scorer   pyes_scorer's own parse_args and run() with a fake vLLM model: a character tokenizer whose chat template closes
         an empty think block, logits from a sha1 of (backbone, adapter, prompt text). Every output file is the real
         scorer's. DRY_E1_FAIL=<text>: a run whose --output contains <text> gets Yes+No mass 0.5 (fails E1).
trainer  train_lora_yesno's own argparse and panel check; writes a placeholder adapter_model.safetensors,
         adapter_config.json, train_report.json and train_config.json (the real keys, plus dry_run).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

GATEFT_MODEL = "dryrun/Qwen3-8B"     # the backbone the fake Gate-FT adapters record
STORES = ["Velorix", "Tamberly", "Quinlo", "Brastow", "Morvane", "Zephyrine", "Kandolo", "Pellarin", "Trivoli", "Hustane"]
NOUNS = ["Rocket", "Castle", "Puzzle Box", "Racer", "Lantern", "Glider", "Robot", "Canvas", "Marble Run", "Kite",
         "Compass", "Drum"]
GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Animation"]
N_USERS, N_ITEMS, PER_USER = 100, 150, 56


def put(path, data) -> None:
    """Write str or bytes only when the file is missing (atomic)."""
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    os.replace(tmp, path)


def gz(text: str) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as f:
        f.write(text.encode("utf-8"))
    return buf.getvalue()


def events(rng):
    """(user, item, stars, time): every user rates PER_USER items over one shared time window."""
    q = [rng.gauss(0, 1) for _ in range(N_ITEMS)]
    for u in range(N_USERS):
        bias = rng.gauss(0, 0.4)
        items = rng.sample(range(N_ITEMS), PER_USER)
        times = sorted(rng.randrange(0, 40_000_000) for _ in items)
        for i, t in zip(items, times):
            yield u, i, min(5, max(1, round(3 + 1.3 * q[i] + bias + rng.gauss(0, 1.1)))), t


def write_raw(raw: Path, domain: str) -> None:
    rng = random.Random(f"dryrun:{domain}")
    if domain == "ml1m":
        d = raw / "ml-1m"
        put(d / "movies.dat", "".join(f"{i + 1}::Synthetic Movie {i + 1} ({1970 + i % 40})::{GENRES[i % 8]}|"
                                      f"{GENRES[(3 * i + 1) % 8]}\n" for i in range(N_ITEMS)))
        put(d / "ratings.dat", "".join(f"{u + 1}::{i + 1}::{r}::{970_000_000 + t}\n" for u, i, r, t in events(rng)))
        return
    from src.confrec.categories import CATEGORY
    cat, d = CATEGORY[domain], raw / f"amazon_{domain}"
    meta = []
    for i in range(N_ITEMS):
        store, noun = STORES[i % len(STORES)], NOUNS[(7 * i) % len(NOUNS)]
        title = f"{store} {noun} {i}" if i % 10 < 7 else f"{noun} Deluxe Edition {i}"
        meta.append(json.dumps({"parent_asin": f"B{i:05d}", "title": title, "store": store,
                                "categories": [cat.replace("_", " "), noun],
                                "description": [f"Synthetic {noun.lower()} number {i} for the dry run."]}) + "\n")
    put(d / f"meta_{cat}.jsonl.gz", gz("".join(meta)))
    put(d / f"{cat}.jsonl.gz", gz("".join(
        json.dumps({"user_id": f"AU{u:04d}", "parent_asin": f"B{i:05d}", "rating": float(r),
                    "timestamp": (1_600_000_000 + t) * 1000}) + "\n" for u, i, r, t in events(rng))))


def gatefix_panels(a, raw: Path) -> tuple[str, str]:
    """The gate-fix h20 DEV / CONFIRM panels of ml1m / toys and their manifest record (a prefix split)."""
    from src.confrec import build_rated_panels as brp
    from src.confrec.stats import strict_json
    items, evs = brp.load_ml1m(raw / "ml-1m") if a.domain == "ml1m" else brp.load_amazon(raw, a.domain)
    rows, n_elig = brp.build(items, evs, n_users=10 ** 9, n_cands=20, hist_len=20, min_hist=3, min_like=3,
                             min_dislike=3, seed=0, source=a.domain)
    pdir, rec = Path(a.gatefix) / "panels", {"prefix_match": True, "split_method": "prefix", "eligible_users": n_elig}
    paths = []
    for name, rs in (("dev", rows[:a.n_dev]), ("confirm", rows[a.n_dev:])):
        text = "".join(brp.row_line(r) for r in rs)
        p = pdir / f"{a.domain}_{name}_h20.jsonl"
        put(p, text)
        ids = "\n".join(sorted(str(r["user_id"]) for r in rs)).encode("utf-8")
        rec[name] = {"built": True, "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
                     "user_ids_sha1": hashlib.sha1(ids).hexdigest(), "n_lines": len(rs)}
        paths.append(str(p).replace("\\", "/"))
    man_path = pdir / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {"dry_run": True, "sources": {}}
    if a.domain not in man["sources"]:
        man["sources"][a.domain] = rec
        tmp = man_path.with_name("manifest.json.tmp")
        tmp.write_text(json.dumps(man, indent=2), encoding="utf-8")
        os.replace(tmp, man_path)
    if a.domain == "toys":
        put(Path(a.dry) / "pilot_panels" / "toys_rated.brand_pop.json",
            json.dumps(strict_json(brp.brand_popularity(items, evs)), ensure_ascii=False))
    return paths[0], paths[1]


def setup(argv) -> None:
    ap = argparse.ArgumentParser()
    for k in ("--domain", "--dry", "--gatefix", "--gateft", "--variant", "--gate"):
        ap.add_argument(k, required=True)
    ap.add_argument("--n_dev", type=int, required=True)
    a = ap.parse_args(argv)
    dry, gfx, gt = Path(a.dry), Path(a.gatefix), Path(a.gateft)
    write_raw(dry / "raw", a.domain)
    put(gfx / "dev" / "selection.json", json.dumps({"stage": "dev", "decision": "FIX_FOUND", "outcome": "FIX_FOUND",
                                                    "fix_found": True, "v_star": a.variant, "gate_ft_prompt": a.variant,
                                                    "dry_run": True}, indent=2))
    put(gfx / "confirm" / "gate.json", json.dumps({"stage": "confirm", "decision": "GATE_PASS", "dry_run": True}))
    put(gt / "gate_ft.json", json.dumps({"decision": a.gate, "dry_run": True}))
    put(dry / "PILOT_LOG.md", "# dry-run pilot log (temporary, written by run_ftgrid.sh DRY_RUN=1)\n")
    if a.domain not in ("ml1m", "toys"):
        return
    dev, conf = gatefix_panels(a, dry / "raw")
    if a.domain != "ml1m":
        return
    if not (gt / "gateft_split.json").exists():
        from src.confrec import gateft_data
        gateft_data.main(["--dev_panel", dev, "--confirm_panel", conf, "--out_dir", a.gateft])
    for k in range(3):
        ad = f"{a.gateft}/adapters/s{k}"
        if not (Path(ad) / "train_config.json").exists():
            trainer(["--train", f"{a.gateft}/train.jsonl", "--model", GATEFT_MODEL, "--out", ad, "--variant",
                     a.variant, "--seed", str(k)])
    for name in ("zeroshot", "s0", "s1", "s2"):
        out = f"{a.gateft}/scores/{name}"
        if not (Path(out) / "report.json").exists():
            lora = [] if name == "zeroshot" else ["--lora", f"{a.gateft}/adapters/{name}"]
            scorer(["--data", conf, "--output", out, "--model", GATEFT_MODEL, "--dtype", "float16", "--topk_logprobs",
                    "50", "--max_model_len", "4096", "--chunk_users", "100", "--variant", a.variant, "--readout",
                    "yesno", "--questions", "like"] + lora)


class Tok:
    """Character tokenizer: id 1 = 'Yes', id 2 = 'No', every other character its own id. The chat template wraps the
    messages and, with thinking off, ends in the closed empty think block (amendment 2 G0)."""

    def __init__(self):
        self.ids, self.text = {}, {1: "Yes", 2: "No"}

    def __len__(self):
        return 3 + len(self.ids)

    def decode(self, ids):
        return "".join(self.text.get(i, "") for i in ids)

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, enable_thinking=True, **kw):
        out = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in msg) + "<|assistant|>"
        return out + ("" if enable_thinking else "<think>\n\n</think>\n\n")

    def __call__(self, text, add_special_tokens=False):
        out = []
        for c in text:
            if c not in self.ids:
                self.ids[c] = len(self.ids) + 3
                self.text[self.ids[c]] = c
            out.append(self.ids[c])
        return {"input_ids": out}


def fake_model(args):
    tok = Tok()
    tag = f"{Path(args.model).name}\x1f{args.lora or 'zeroshot'}"
    fail = os.environ.get("DRY_E1_FAIL", "")
    mass = 0.5 if fail and fail in str(args.output).replace("\\", "/") else 0.99999

    class LLM:
        def generate(self, prompts, sp, use_tqdm=False, **kw):
            res = []
            for p in prompts:
                h = hashlib.sha1((tag + "\x1e" + tok.decode(p["prompt_token_ids"])).encode("utf-8")).digest()
                u1 = (int.from_bytes(h[:6], "big") + 0.5) / 2 ** 48
                u2 = (int.from_bytes(h[6:12], "big") + 0.5) / 2 ** 48
                lg = 1.5 * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
                p_yes = mass / (1 + math.exp(-lg))
                top = {1: SimpleNamespace(logprob=math.log(p_yes)), 2: SimpleNamespace(logprob=math.log(mass - p_yes))}
                for k in range(48):
                    top[10 ** 6 + k] = SimpleNamespace(logprob=math.log(1e-7) - k / 100)
                res.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[top])]))
            return res

    return SimpleNamespace(llm=LLM(), tok=tok, gen_kw={}, to_prompt=lambda ids: {"prompt_token_ids": ids}, sp=None,
                           version="dryrun-fake")


def scorer(argv) -> None:
    from src.confrec import pyes_scorer as ps
    ps.run(ps.parse_args(argv), load_model=fake_model)


def train_parser():
    """(the module, its real ArgumentParser): main() is stopped at parse_args. Without torch (a CPU box) stand-in
    modules let the module import; its parser and panel check need none of it."""
    try:
        import torch  # noqa: F401
    except ImportError:
        for name in ("torch", "torch.utils", "torch.utils.data", "torch.nn", "torch.nn.functional"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["torch.utils.data"].Dataset = object
    from src.confrec import train_lora_yesno as tl
    got = {}
    orig = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        got["parser"] = self
        raise SystemExit(0)
    argparse.ArgumentParser.parse_args = grab
    try:
        tl.main([])
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    return tl, got["parser"]


def trainer(argv) -> None:
    from src.confrec.stats import strict_json
    tl, parser = train_parser()
    a = parser.parse_args(argv)
    rows = [json.loads(x) for x in open(a.train, encoding="utf-8") if x.strip()]
    random.Random(a.seed).shuffle(rows)
    panel = tl.check_training_panel(rows, a.variant, a.hist_len, a.train)
    built = sum(len(r["candidate_labels"]) for r in rows) * (2 if a.mode == "mirror" else 1)
    n = min(built, a.max_examples) if a.max_examples else built
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(Path(a.train).read_bytes() + f"|{a.seed}|{a.variant}|{a.model}".encode("utf-8")).hexdigest()
    (out / "adapter_model.safetensors").write_text(f"DRY_RUN placeholder adapter {digest}\n", encoding="utf-8")
    (out / "adapter_config.json").write_text(json.dumps(
        {"dry_run": True, "base_model_name_or_path": a.model, "r": a.lora_r, "lora_alpha": 2 * a.lora_r,
         "lora_dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}, indent=2), encoding="utf-8")
    (out / "train_report.json").write_text(json.dumps(strict_json(dict(
        n_examples=n, n_examples_built=built, n_skipped_overlength=0, max_len=a.max_len, max_prompt_len=None,
        mode=a.mode, variant=a.variant, hist_len=panel["hist_len_used"], panel_kind=panel["panel_kind"],
        max_history_len_in_panel=panel["max_history_len_in_panel"], answer_ids={"Yes": 1, "No": 2}, train=a.train,
        model=a.model, dry_run=True)), indent=2), encoding="utf-8")
    (out / "train_config.json").write_text(json.dumps(strict_json(
        {**vars(a), "loss": "last_token", "hist_len_used": panel["hist_len_used"], "panel_kind": panel["panel_kind"],
         "max_history_len_in_panel": panel["max_history_len_in_panel"], "n_examples": n, "n_skipped_overlength": 0,
         "dry_run": True}), indent=2), encoding="utf-8")
    print(f"dry-run trainer: {out} ({n} examples, variant {a.variant}, bsz {a.bsz} x {a.grad_accum}, max_len {a.max_len})")


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    {"setup": setup, "scorer": scorer, "trainer": trainer}[sys.argv[1]](sys.argv[2:])
PYFAKES
  if cmp -s "$FAKES.tmp" "$FAKES"; then rm -f "$FAKES.tmp"; else mv "$FAKES.tmp" "$FAKES"; fi
  "$PY" "$FAKES" setup --domain "$D" --dry "$DRY" --gatefix "$G" --gateft "$GT" --variant "${VARIANT:-V3}" \
    --n_dev "$N_TRAIN" --gate "${DRY_GATE:-GATE_FT_PASS}"
fi

# ---- the prompt, the gate decisions, the freeze record ----
SEL="$G/dev/selection.json"
[ -f "$SEL" ] || { echo "missing $SEL (the gate-fix stage 1 selection; run_gatefix.sh writes it)" >&2; exit 1; }
SEL_PROMPT=$(jget "$SEL" gate_ft_prompt)
SEL_DECISION=$(jget "$SEL" decision)
VARIANT="${VARIANT:-$SEL_PROMPT}"
if [ "$VARIANT" != "$SEL_PROMPT" ] && [ "$DRY_RUN" != 1 ]; then
  echo "VARIANT=$VARIANT is not selection.json's gate_ft_prompt $SEL_PROMPT: section 2 registers no other variant" >&2
  exit 2
fi
GATE_DECISION=missing
if [ -f "$GT/gate_ft.json" ]; then GATE_DECISION=$(jget "$GT/gate_ft.json" decision); fi
FT_OK=0
if [ "$GATE_DECISION" = GATE_FT_PASS ]; then FT_OK=1; fi
MODELS=zeroshot
if [ "$FT_OK" = 1 ]; then MODELS="zeroshot s0 s1 s2"; fi
if [ "$DRY_RUN" = 1 ]; then
  CORE_SPLITS=(--split "$SPLIT")
else
  CORE_SPLITS=(--split "$REG/panels/ml1m/ftgrid_split.json" --split "$REG/panels/toys/ftgrid_split.json"
    --split "$REG/panels/games/ftgrid_split.json" --split "$REG/panels/sports/ftgrid_split.json")
  if [ "$OUT_ROOT" != "$REG" ]; then CORE_SPLITS+=(--split "$SPLIT"); fi
fi
FTD=(--variant "$VARIANT" --n_train "$N_TRAIN" --n_eval_max "$N_EVAL" --quantile 0.8 --s_max "$S_MAX" --train_cap 24000 --seed 0)
echo "run_ftgrid $D: model $MODEL, root $OUT_ROOT, variant $VARIANT, Gate-FT $GATE_DECISION, stages:$STAGES"
if [ "$FT_OK" != 1 ]; then
  echo "Gate-FT has not recorded GATE_FT_PASS: only the zero-shot panels run (section 0; section 4 runs in every branch)"
fi

# ================= stage 0: data (CPU) =================
if want 0; then
  echo "== stage 0: data ($D)"
  mkdir -p "$B" "$P"
  if [ "$D" = ml1m ] || [ "$D" = toys ]; then
    DEVP="$G/panels/${D}_dev_h20.jsonl"
    CONFP="$G/panels/${D}_confirm_h20.jsonl"
    for f in "$DEVP" "$CONFP" "$G/panels/manifest.json"; do
      [ -f "$f" ] || { echo "missing $f (the gate-fix stage 0 panels; run_gatefix.sh builds them)" >&2; exit 1; }
    done
    step "$ALL" "$DEVP" "$CONFP" -- join_h20 "$DEVP" "$CONFP"
  else
    CAT=$("$PY" -c "from src.confrec.categories import CATEGORY; print(CATEGORY['$D'])" | tr -d '\r')
    for f in "$RAW/amazon_$D/$CAT.jsonl.gz" "$RAW/amazon_$D/meta_$CAT.jsonl.gz"; do
      [ -f "$f" ] || { echo "missing $f (slim_amazon2023.py --domains $D downloads it)" >&2; exit 1; }
    done
    step "$ALL" "$RAW/amazon_$D/$CAT.jsonl.gz" "$RAW/amazon_$D/meta_$CAT.jsonl.gz" src/confrec/build_rated_panels.py \
        src/confrec/categories.py -- \
      "$PY" -m src.confrec.build_rated_panels --source amazon --domain "$D" --raw "$RAW" --out "$ALL" \
        --n_users 1000000000 --n_cands 20 --hist_len 20 --min_hist 3 --min_like 3 --min_dislike 3 --seed 0 \
        --gatefix_fields
  fi
  FDEPS=("$ALL" src/confrec/ftgrid_data.py src/confrec/gateft_data.py src/confrec/build_rated_panels.py)
  if [ "$DRY_RUN" != 1 ]; then FTD+=(--tokenizer "$MODEL"); fi
  if [ "$D" = ml1m ]; then
    [ -f "$GT/gateft_split.json" ] || { echo "missing $GT/gateft_split.json (Gate-FT stage A)" >&2; exit 1; }
    FTD+=(--gateft_split "$GT/gateft_split.json"
      --ml1m_dev_users_sha1 "$(jget "$G/panels/manifest.json" sources ml1m dev user_ids_sha1)")
    FDEPS+=("$GT/gateft_split.json")
  fi
  step "$SPLIT" "${FDEPS[@]}" -- \
    "$PY" -m src.confrec.ftgrid_data --domain "$D" --panel_all "$ALL" --out_dir "$P" "${FTD[@]}"
  step "$P/eval_sd_test_starperm1.jsonl" "$P/eval_sd_test.jsonl" scripts/sigir/starperm_panel.py \
      src/confrec/diag_battery.py -- \
    "$PY" scripts/sigir/starperm_panel.py --panel "$P/eval_sd_test.jsonl" --variant "$VARIANT" \
      --out_prefix "$P/eval_sd_test_starperm" --meta "$B/starperm.json"
  if [ "$KO" = 1 ]; then
    if [ "$D" = toys ]; then BP="$PILOT_PANELS/toys_rated.brand_pop.json"; else BP="${ALL%.jsonl}.brand_pop.json"; fi
    [ -f "$BP" ] || { echo "missing $BP: the category-wide brand popularity of the placebo (amendment 1 P3)" >&2; exit 1; }
    step "$P/eval_placebo.jsonl" "$P/eval.jsonl" "$BP" src/confrec/pseudonymize.py -- knockout_panels "$BP"
  fi
fi

# ================= stage 1 / perm: adapters (GPU) =================
if want 1; then
  echo "== stage 1: adapters s0-s2 ($D)"
  if [ "$FT_OK" != 1 ]; then
    echo "[stage 1] not run: Gate-FT decision $GATE_DECISION (section 0: fine-tuning only after GATE_FT_PASS)"
  elif [ "$D" = ml1m ] && [ "$QWEN" = 1 ]; then
    for seed in 0 1 2; do link_gateft "$seed"; done
  else
    gpu_freeze
    recipe
    for seed in 0 1 2; do train_adapter "$P/train.jsonl" "$ADIR/s$seed" "$seed"; done
  fi
fi
if want perm; then
  echo "== perm: FT-C adapters p0, p1 ($D)"
  if [ "$D" != ml1m ] || [ "$QWEN" != 1 ]; then
    echo "perm: FT-C runs on ML-1M with the Gate-FT backbone only (section 9)" >&2; exit 2
  fi
  if [ "$FT_OK" != 1 ]; then echo "perm refused: Gate-FT decision $GATE_DECISION (section 0)" >&2; exit 4; fi
  gpu_freeze
  recipe
  for seed in 0 1; do train_adapter "$P/train_perm.jsonl" "$ADIR/p$seed" "$seed"; done
fi

# ================= stage 2: the freeze record =================
if want 2; then
  echo "== stage 2: freeze check ($D)"
  [ -f "$SPLIT" ] || { echo "missing $SPLIT (stage 0)" >&2; exit 1; }
  if [ "$DRY_RUN" = 1 ]; then
    # rehearsal: an empty pilot log fails the check, and stage 3 then refuses; the human step is dry_record
    [ -f "$DRY/PILOT_LOG.empty.md" ] || printf '# empty temporary pilot log\n' > "$DRY/PILOT_LOG.empty.md"
    if "$PY" -m src.confrec.ftgrid_freeze --check --stage core --pilot_log "$DRY/PILOT_LOG.empty.md" "${CORE_SPLITS[@]}" \
        2> /dev/null; then
      echo "DRY_RUN: the freeze check passed on an empty pilot log" >&2; exit 1
    fi
    if (MARK="$DRY/no.core.ok"; require_freeze 3) 2> /dev/null; then
      echo "DRY_RUN: stage 3 did not refuse without the freeze record" >&2; exit 1
    fi
    echo "[dry] freeze rehearsal: an empty pilot log fails the check and stage 3 refuses; recording the temporary log"
    dry_record core
  fi
  if ! freeze_check core; then
    echo "stage 2: the full Amendment-3 record is not in $PILOT_LOG; record the lines of" \
      "'python -m src.confrec.ftgrid_freeze --print --stage core' and rerun (data and adapters are kept)" >&2
    exit 4
  fi
  REC=$("$PY" -m src.confrec.ftgrid_freeze --print --stage core "${CORE_SPLITS[@]}")
  mkdir -p "$(dirname "$MARK")"
  if [ ! -f "$MARK" ] || [ "$(cat "$MARK")" != "$REC" ]; then printf '%s\n' "$REC" > "$MARK"; fi
  FREEZE_OK=1
fi

# ================= stage 3: like on eval.jsonl =================
if want 3; then
  echo "== stage 3: like on eval.jsonl ($D)"
  scoring_stage 3
  for m in $(like_models); do
    set_lora "$m"
    ADOPT=$(adopt_source "$m")
    score "$P/eval.jsonl" "$S/$m/like" "${LORA[@]}"
  done
fi

# ================= stage 4: decomposition arms on S_d's TEST rows =================
if want 4; then
  echo "== stage 4: decomposition arms on eval_sd_test.jsonl ($D)"
  scoring_stage 4
  for m in $MODELS; do
    set_lora "$m"
    score "$P/eval_sd_test.jsonl" "$S/$m/swap" "${LORA[@]}" --swap_k 8
    score "$P/eval_sd_test.jsonl" "$S/$m/nohist" "${LORA[@]}" --hist_len 0
    score "$P/eval_sd_test_starperm0.jsonl" "$S/$m/starperm0" "${LORA[@]}"
    score "$P/eval_sd_test_starperm1.jsonl" "$S/$m/starperm1" "${LORA[@]}"
  done
fi

# ================= stage 5: pseudonym knockout =================
if want 5; then
  echo "== stage 5: knockout arms ($D)"
  if [ "$KO" != 1 ]; then
    echo "[stage 5] not run: no knockout on $D (section 5: toys, games; sports with KNOCKOUT_SPORTS=1)"
  elif [ "$FT_OK" != 1 ]; then
    echo "[stage 5] not run: Gate-FT decision $GATE_DECISION (section 0: section 5 runs only after GATE_FT_PASS)"
  else
    scoring_stage 5
    for m in $MODELS; do
      set_lora "$m"
      score "$P/eval_pseudo.jsonl" "$S/$m/pseudo" "${LORA[@]}"
      score "$P/eval_placebo.jsonl" "$S/$m/placebo" "${LORA[@]}"
    done
  fi
fi

# ================= stage 6: report (CPU) =================
if want 6; then
  echo "== stage 6: report ($D)"
  MLIST=$(like_models | tr ' ' ',')
  RDEPS=("$SPLIT" src/confrec/ftgrid_report.py)
  for f in "$S"/*/*/report.json; do if [ -f "$f" ]; then RDEPS+=("$f"); fi; done
  mkdir -p "$(dirname "$REP")"
  step "$REP" "${RDEPS[@]}" -- \
    "$PY" -m src.confrec.ftgrid_report --domain "$D" --split "$SPLIT" --panels "$P" --scores_root "$S" \
      --models "$MLIST" --raw "$RAW" --out "$REP" --n_boot "$N_BOOT" --seed 0
fi
echo "run_ftgrid $D: done (stages:$STAGES)"
