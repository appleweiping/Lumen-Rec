#!/usr/bin/env bash
# S6 pruning by uncertainty, FT-P (idea-stage/PREREG_AMENDMENT_3.md section 6, with sections 0, 2, 10, 11 and
# idea-stage/PREREG_AMENDMENT_3_ADDENDUM_1.md): ML-1M, the Gate-FT backbone (Qwen3-8B) and prompt. GPU server; one job at a
# time through scripts/sigir/gpu_queue.sh (section 10 order: after FT-C; P1 / P2 with P0 s3, s4 first, then P3).
#   usage: bash scripts/sigir/run_ftprune.sh
#   env:   STAGES    comma list of A-E or all (default all)
#          RUN_P3=0  cut P3 (sections 6 and 10: P3 runs unless cut at the 2026-10-22 checkpoint; default 1)
#          MODEL     backbone dir (default /root/autodl-tmp/lumen/models/Qwen3-8B): the Gate-FT adapters' backbone
#          VARIANT   default gate_ft_prompt of outputs/confrec/gatefix/dev/selection.json; any other value is refused
#                    (section 2: no other variant)
#          PYTHON    interpreter (skips the conda activation);  DRY_RUN=1  CPU rehearsal, see below
#   stage A  the zero-shot like pass of the TRAIN set outputs/confrec/gateft/train.jsonl (Gate-FT's G9 set) under the
#            selected variant, no adapter -> OUT/zs_train. A scoring run: it waits for the full core record (section 0)
#   stage B  (CPU) ftprune signals: L_ZS, tau, u, q-hat, C, the subsets of P0-P3 for seeds 0-4, the pruned TRAIN files
#            and OUT/prune_manifest.json (deterministic; a file whose bytes would not change is left untouched)
#   stage C  freeze checks amendment, core and prune (FREEZE prune = ftprune.py, run_ftprune.sh and prune_manifest.json,
#            which holds the sha1 of every signal, subset and pruned-train file), then ftprune verify (every one of those
#            files still has its recorded sha1). Success writes OUT/freeze/prune.ok (the printed core and prune records).
#            Stages D and E refuse (exit 4) unless stage C passed in this run or that marker equals the current record
#   stage D  ftprune verify, then for seed s = 0..4: train P1 s and P2 s (and P0 s for s = 3, 4) on
#            OUT/train/<arm>_s<seed>.jsonl with the Gate-FT recipe (variant, micro-batch, accumulation and max_len of the
#            Gate-FT adapters' train_config.json, identical across s0-s2), each scored on the CONFIRM panel with
#            run_gateft.sh's pyes_scorer call (like, --lora); then P3 s0-s4 unless RUN_P3=0. P0 s0-s2 are the Gate-FT
#            adapters with their Gate-FT stage-C like passes: never retrained, never rescored here
#   stage E  ftprune analyze -> OUT/pruning_ml1m.json and .csv (the label of the P2 - P1 test; exit 2 when INCOMPLETE)
# Conditionality (section 0): nothing runs unless outputs/confrec/gateft/gate_ft.json records GATE_FT_PASS (exit 4).
# Re-runnable: an adapter with train_config.json and weights is never retrained; a scoring dir is skipped when report.json
# exists and run.key (panel sha1, model, variant, adapter-weights sha1, args) is unchanged, else moved to DIR.stale.<time>;
# stage B is skipped while the manifest's step marker (OUT/build/done) is newer than its inputs; every output of ftprune is
# written only when its bytes change. Section 2 / E1 (censored-2 share <= 0.5%, no overlength prompt, Yes+No mass >= 0.95):
# a failing run is moved to DIR.e1fail.<time> and rerun once; a second failure leaves DIR/FAILED_INTEGRITY (missing, never
# replaced: the analysis then reads INCOMPLETE; on the TRAIN pass S6 has no U signal and stops).
# DRY_RUN=1: the same chain on CPU on a tiny synthetic ML-1M-like world. OUT_ROOT defaults to outputs/confrec/ftprune_dryrun
# (the registered root is refused); the inputs live in OUT_ROOT/_dry (raw files, the gate-fix panels and selection, the
# Gate-FT context with stand-in adapters and their like passes, a stand-in for the core record's splits, a temporary pilot
# log). pyes_scorer and train_lora_yesno are replaced by stand-ins (OUT_ROOT/_dry/ftprune_fakes.py: the real scorer code
# with a fake model, the trainer's real argparse and panel check with a placeholder adapter, as run_ftgrid.sh's DRY_RUN);
# every other step is the real code, the freeze checks included (on the temporary pilot log; stage C first shows that an
# empty log fails the prune check and that stage D then refuses). DRY_GATE=GATE_FT_FAIL rehearses the gate refusal,
# DRY_E1_FAIL=<text> makes the runs whose output dir contains <text> fail E1.
# Exit codes: 0 done; 1 error; 2 usage, refused input or an INCOMPLETE analysis; 4 refused by the gate / freeze rules.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
DRY_RUN="${DRY_RUN:-0}"
RUN_P3="${RUN_P3:-1}"
case "$RUN_P3" in
  0|1) ;;
  *) echo "RUN_P3 must be 0 (P3 cut) or 1, got '$RUN_P3'" >&2; exit 2 ;;
esac
STAGES=" $(printf '%s' "${STAGES:-all}" | tr ',' ' ' | tr '[:lower:]' '[:upper:]') "
for s in $STAGES; do
  case "$s" in A|B|C|D|E|ALL) ;; *) echo "unknown stage '$s' in STAGES (A-E, all)" >&2; exit 2 ;; esac
done
case "$STAGES" in *" ALL "*) STAGES="$STAGES A B C D E " ;; esac
want() { case "$STAGES" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
REG=outputs/confrec/ftprune      # the registered root: FREEZE prune records REG/prune_manifest.json
if [ "$DRY_RUN" = 1 ]; then
  MODEL="${MODEL:-dryrun/Qwen3-8B}"
  OUT="${OUT_ROOT:-outputs/confrec/ftprune_dryrun}"
else
  MODEL="${MODEL:-/root/autodl-tmp/lumen/models/Qwen3-8B}"
  OUT="${OUT_ROOT:-$REG}"
fi
OUT="${OUT%/}"
OUT="${OUT#./}"
if [ "$DRY_RUN" = 1 ] && [ "$OUT" = "$REG" ]; then
  echo "DRY_RUN=1 never writes to the registered output root ($OUT)" >&2; exit 2
fi
if [ "$DRY_RUN" != 1 ] && [ "$OUT" != "$REG" ]; then
  echo "OUT_ROOT=$OUT refused: S6 writes to $REG only (FREEZE prune records its manifest there); OUT_ROOT is for DRY_RUN" >&2
  exit 2
fi
if [ "$(basename "$MODEL")" != Qwen3-8B ]; then
  echo "$MODEL refused: S6 runs on the Gate-FT backbone Qwen3-8B only (section 6: single-backbone)" >&2; exit 2
fi
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  set +u; source /root/miniconda3/etc/profile.d/conda.sh; conda activate lumen; set -u
  PY=python
fi
if [ "$DRY_RUN" = 1 ]; then
  DRY="$OUT/_dry"
  G="$DRY/gatefix"; GT="$DRY/gateft"; RAW="$DRY/raw"; PILOT_LOG="$DRY/PILOT_LOG.md"
  CORE_SPLIT="$DRY/ftgrid/panels/ml1m/ftgrid_split.json"      # stand-in for the four registered splits of the record
else
  G=outputs/confrec/gatefix; GT=outputs/confrec/gateft; RAW=data/raw; PILOT_LOG=docs/sigir/PILOT_LOG.md
fi
SEL="$G/dev/selection.json"
CONFP="$G/panels/ml1m_confirm_h20.jsonl"   # the CONFIRM panel = the EVAL users of ML-1M (section 1), as Gate-FT scores it
TRAIN="$GT/train.jsonl"                    # the Gate-FT TRAIN set (G9; section 1: ML-1M's TRAIN examples)
GSPLIT="$GT/gateft_split.json"
RATINGS="$RAW/ml-1m/ratings.dat"
ZS="$OUT/zs_train"
MAN="$OUT/prune_manifest.json"
ADIR="$OUT/adapters"
SC="$OUT/scores"
RES="$OUT/pruning_ml1m.json"
MARK="$OUT/freeze/prune.ok"
DONE="$OUT/build/done"
FREEZE_OK=0
CORE_ARGS=()
PRUNE_ARGS=()
if [ "$DRY_RUN" = 1 ]; then
  CORE_ARGS=(--split "$CORE_SPLIT")
  PRUNE_ARGS=(--split "$MAN")
fi

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
# e1_ok DIR: section 2 / E1 of a finished scorer run: censored-2 share <= 0.5%, no overlength prompt, mean Yes+No mass
# >= 0.95
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
# MPY runs the entry points that load the model (scorer, trainer). Under DRY_RUN it is fake_py: the scorer and the trainer
# are stand-ins; any other call runs as is
fake_py() {
  if [ "$1" = -m ]; then
    case "$2" in
      src.confrec.pyes_scorer) shift 2; "$PY" "$FAKES" scorer "$@"; return ;;
      src.confrec.train_lora_yesno) shift 2; "$PY" "$FAKES" trainer "$@"; return ;;
    esac
  fi
  "$PY" "$@"
}
MPY="$PY"
if [ "$DRY_RUN" = 1 ]; then MPY=fake_py; fi
# score DATA DIR [--lora A]: run_gateft.sh's pyes_scorer call (fp16, top-50 logprobs, max_model_len 4096, 100-user chunks,
# the selected variant, yes/no readout, like); completion marker DIR/report.json, run.key = panel sha1 + model + variant +
# adapter weights sha1 + args; E1 rerun-once (section 2)
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
# freeze_check STAGE: 0 iff the pilot log records every sha1 the freeze stage requires (section 0; prune: section 6)
freeze_check() {
  case "$1" in
    core) "$PY" -m src.confrec.ftgrid_freeze --check --stage core --pilot_log "$PILOT_LOG" "${CORE_ARGS[@]}" ;;
    prune) "$PY" -m src.confrec.ftgrid_freeze --check --stage prune --pilot_log "$PILOT_LOG" "${PRUNE_ARGS[@]}" ;;
    *) "$PY" -m src.confrec.ftgrid_freeze --check --stage "$1" --pilot_log "$PILOT_LOG" ;;
  esac
}
# record: the printed core and prune records (the content of the stage-C marker)
record() {
  "$PY" -m src.confrec.ftgrid_freeze --print --stage core "${CORE_ARGS[@]}"
  "$PY" -m src.confrec.ftgrid_freeze --print --stage prune "${PRUNE_ARGS[@]}"
}
# dry_record STAGE: DRY_RUN only, the human step: write the stage's sha1 lines into the temporary pilot log
dry_record() {
  if freeze_check "$1" > /dev/null 2>&1; then return 0; fi
  case "$1" in
    core) "$PY" -m src.confrec.ftgrid_freeze --print --stage core "${CORE_ARGS[@]}" >> "$PILOT_LOG" ;;
    prune) "$PY" -m src.confrec.ftgrid_freeze --print --stage prune "${PRUNE_ARGS[@]}" >> "$PILOT_LOG" ;;
    *) "$PY" -m src.confrec.ftgrid_freeze --print --stage "$1" >> "$PILOT_LOG" ;;
  esac
}
# require_freeze STAGE: S6 training, scoring and analysis need stage C in this run, or a marker equal to the current record
require_freeze() {
  if [ "$FREEZE_OK" = 1 ]; then return 0; fi
  if [ -f "$MARK" ] && [ "$(record 2> /dev/null)" = "$(cat "$MARK")" ]; then
    FREEZE_OK=1; return 0
  fi
  echo "stage $1 refused: S6 waits for FREEZE prune and the full record (sections 0 and 6). Run stage C (STAGES=C,...):" \
    "$MARK is missing or no longer equals the record (a bound file, the manifest or a split changed since the check)" >&2
  return 1
}
# recipe: MICRO ACCUM MAXLEN of the Gate-FT adapters s0-s2 (their train_config.json, identical across seeds; the variant
# and backbone must be this run's): P0 s3, s4 and P1-P3 train exactly as P0 s0-s2 did (section 2)
recipe() {
  local r
  r=$("$PY" - "$GT/adapters" "$VARIANT" "$(basename "$MODEL")" <<'PY'
import json, sys
from pathlib import Path
root, variant, backbone = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
keys = ("bsz", "grad_accum", "max_len", "lr", "epochs", "lora_r", "mode", "hist_len", "variant")
cfgs = []
for s in (0, 1, 2):
    p = root / f"s{s}" / "train_config.json"
    if not p.is_file():
        sys.exit(f"{p} does not exist: the Gate-FT adapter s{s} (run_gateft.sh trains it, never this script)")
    c = json.loads(p.read_text(encoding="utf-8"))
    if c.get("variant", "V0") != variant or Path(str(c.get("model", ""))).name != backbone:
        sys.exit(f"{p}: trained with {c.get('variant')} on {Path(str(c.get('model', ''))).name}, not {variant} on {backbone}")
    cfgs.append({k: c.get(k) for k in keys})
if any(c != cfgs[0] for c in cfgs):
    sys.exit(f"the Gate-FT adapters differ in their recipe: {cfgs}")
c = cfgs[0]
if c["bsz"] * c["grad_accum"] != 32 or float(c["lr"]) != 1e-4 or float(c["epochs"]) != 1.0 or int(c["lora_r"]) != 16 \
        or c["mode"] != "standard" or c["hist_len"] is not None:
    sys.exit(f"the Gate-FT recipe {c} is not the registered one (section 2: batch 32, lr 1e-4, 1 epoch, r 16, standard, "
             "the variant's window); this script passes only micro-batch, accumulation and max_len")
print(c["bsz"], c["grad_accum"], c["max_len"])
PY
) || exit 1
  read -r MICRO ACCUM MAXLEN <<< "$(printf '%s' "$r" | tr -d '\r')"
}
# train_adapter TRAIN OUT SEED: the section-2 recipe; skipped when train_config.json and adapter weights exist
train_adapter() {
  local train=$1 out=$2 seed=$3
  if adapter_done "$out"; then echo "[skip] adapter $out exists"; return 0; fi
  [ -f "$train" ] || { echo "missing $train (stage B)" >&2; exit 1; }
  rm -rf "$out"
  "$MPY" -m src.confrec.train_lora_yesno --train "$train" --model "$MODEL" --out "$out" --variant "$VARIANT" \
    --seed "$seed" --max_len "$MAXLEN" --bsz "$MICRO" --grad_accum "$ACCUM" 2>&1 | grep -vE "it/s\]|s/it\]" || true
  adapter_done "$out" || { echo "training of $out did not finish" >&2; exit 1; }
}
# arm_run ARM SEED: train the adapter of one arm and seed, then its like pass on the CONFIRM panel
arm_run() {
  local arm=$1 s=$2
  train_adapter "$OUT/train/${arm}_s$s.jsonl" "$ADIR/$arm/s$s" "$s"
  score "$CONFP" "$SC/$arm/s$s" --lora "$ADIR/$arm/s$s"
}

# ---- DRY_RUN: the synthetic world and the stand-ins ----
if [ "$DRY_RUN" = 1 ]; then
  mkdir -p "$DRY"
  FAKES="$DRY/ftprune_fakes.py"
  cat > "$FAKES.tmp" <<'PYFAKES'
"""DRY_RUN stand-ins of run_ftprune.sh (written by it into OUT_ROOT/_dry; CPU only, no model, no GPU, no network), in the
pattern of the ftgrid_fakes.py that run_ftgrid.sh writes.

    python ftprune_fakes.py setup --dry DIR --variant V --gate DECISION --model M
    python ftprune_fakes.py scorer <the pyes_scorer arguments>
    python ftprune_fakes.py trainer <the train_lora_yesno arguments>

setup    the synthetic ML-1M-like world under DIR; an existing file is never rewritten, so a rerun changes nothing:
         raw/ml-1m/{movies.dat, ratings.dat}; gatefix/panels/ml1m_{dev,confirm}_h20.jsonl (build_rated_panels.build on that
         raw: every eligible user, 20 candidates, hist_len 20; DEV = the first 40 rows); gatefix/dev/selection.json
         (gate_ft_prompt V); gateft/{train.jsonl, gateft_split.json} (gateft_data on the dry panels); gateft/adapters/s0-s2
         (the trainer stand-in with run_gateft.sh's arguments); gateft/scores/s0-s2 (the scorer stand-in with
         run_gateft.sh's arguments on the CONFIRM panel); gateft/gate_ft.json (decision DECISION);
         ftgrid/panels/ml1m/ftgrid_split.json (a stand-in for the core record's four splits: the fields the freeze
         reads); PILOT_LOG.md (the temporary pilot log).
scorer   pyes_scorer's own parse_args and run() with a fake vLLM model: a character tokenizer whose chat template closes an
         empty think block, logits from a sha1 of (backbone, adapter, prompt text). Every output file is the real
         scorer's. DRY_E1_FAIL=<text>: a run whose --output contains <text> gets Yes+No mass 0.5 (fails E1).
trainer  train_lora_yesno's own argparse and panel check (torch is never imported: stand-in modules); writes a
         placeholder adapter_model.safetensors, adapter_config.json, train_report.json and train_config.json (the real
         keys, plus dry_run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Animation"]
N_USERS, N_ITEMS, PER_USER, N_DEV = 100, 150, 56, 40


def put(path, text: str) -> None:
    """Write text only when the file is missing (atomic)."""
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, path)


def write_raw(raw: Path) -> None:
    """ML-1M files (movies.dat, ratings.dat): every user rates PER_USER items over one shared time window."""
    rng = random.Random("dryrun:ftprune")
    d = raw / "ml-1m"
    put(d / "movies.dat", "".join(f"{i + 1}::Synthetic Movie {i + 1} ({1970 + i % 40})::{GENRES[i % 8]}|"
                                  f"{GENRES[(3 * i + 1) % 8]}\n" for i in range(N_ITEMS)))
    q = [rng.gauss(0, 1) for _ in range(N_ITEMS)]
    lines = []
    for u in range(N_USERS):
        bias = rng.gauss(0, 0.4)
        items = rng.sample(range(N_ITEMS), PER_USER)
        times = sorted(rng.randrange(0, 40_000_000) for _ in items)
        for i, t in zip(items, times):
            r = min(5, max(1, round(3 + 1.3 * q[i] + bias + rng.gauss(0, 1.1))))
            lines.append(f"{u + 1}::{i + 1}::{r}::{970_000_000 + t}\n")
    put(d / "ratings.dat", "".join(lines))


def setup(argv) -> None:
    ap = argparse.ArgumentParser()
    for k in ("--dry", "--variant", "--gate", "--model"):
        ap.add_argument(k, required=True)
    a = ap.parse_args(argv)
    from src.confrec import build_rated_panels as brp
    dry = Path(a.dry)
    write_raw(dry / "raw")
    pdir = dry / "gatefix" / "panels"
    dev, conf = (pdir / "ml1m_dev_h20.jsonl").as_posix(), (pdir / "ml1m_confirm_h20.jsonl").as_posix()
    if not (Path(dev).exists() and Path(conf).exists()):
        items, evs = brp.load_ml1m(dry / "raw" / "ml-1m")
        rows, _ = brp.build(items, evs, n_users=10 ** 9, n_cands=20, hist_len=20, min_hist=3, min_like=3,
                            min_dislike=3, seed=0, source="ml1m")
        put(dev, "".join(brp.row_line(r) for r in rows[:N_DEV]))
        put(conf, "".join(brp.row_line(r) for r in rows[N_DEV:]))
    put(dry / "gatefix" / "dev" / "selection.json", json.dumps(
        {"stage": "dev", "decision": "FIX_FOUND", "outcome": "FIX_FOUND", "fix_found": True, "v_star": a.variant,
         "gate_ft_prompt": a.variant, "dry_run": True}, indent=2))
    gt = f"{a.dry}/gateft"
    if not Path(gt, "gateft_split.json").exists():
        from src.confrec import gateft_data
        gateft_data.main(["--dev_panel", dev, "--confirm_panel", conf, "--out_dir", gt])
    for k in range(3):
        ad = f"{gt}/adapters/s{k}"
        if not Path(ad, "train_config.json").exists():
            trainer(["--train", f"{gt}/train.jsonl", "--model", a.model, "--out", ad, "--variant", a.variant,
                     "--seed", str(k)])
        out = f"{gt}/scores/s{k}"
        if not Path(out, "report.json").exists():
            scorer(["--data", conf, "--output", out, "--model", a.model, "--dtype", "float16", "--topk_logprobs", "50",
                    "--max_model_len", "4096", "--chunk_users", "100", "--variant", a.variant, "--readout", "yesno",
                    "--questions", "like", "--lora", ad])
    put(Path(gt) / "gate_ft.json", json.dumps({"decision": a.gate, "dry_run": True}))
    T = json.loads(Path(gt, "gateft_split.json").read_text(encoding="utf-8"))["T"]
    put(dry / "ftgrid" / "panels" / "ml1m" / "ftgrid_split.json", json.dumps(
        {"domain": "ml1m", "dry_run": True, "note": "DRY_RUN stand-in for the four registered splits of the core record",
         "T": T, "gateft_T_match": True, "train": {"overlength": {"share_above_1024": 0.0}}}, indent=2))
    put(dry / "PILOT_LOG.md", "# dry-run pilot log (temporary, written by run_ftprune.sh DRY_RUN=1)\n")


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
    """(the module, its real ArgumentParser): main() is stopped at parse_args. torch is never imported here: stand-in
    modules let the module import (its parser and panel check need none of torch; this process trains nothing)."""
    for name in ("torch", "torch.utils", "torch.utils.data", "torch.nn", "torch.nn.functional"):
        sys.modules.setdefault(name, types.ModuleType(name))
    if not hasattr(sys.modules["torch.utils.data"], "Dataset"):
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
    print(f"dry-run trainer: {out.as_posix()} ({n} examples, seed {a.seed}, variant {a.variant}, bsz {a.bsz} x "
          f"{a.grad_accum}, max_len {a.max_len})")


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    {"setup": setup, "scorer": scorer, "trainer": trainer}[sys.argv[1]](sys.argv[2:])
PYFAKES
  if cmp -s "$FAKES.tmp" "$FAKES"; then rm -f "$FAKES.tmp"; else mv "$FAKES.tmp" "$FAKES"; fi
  "$PY" "$FAKES" setup --dry "$DRY" --variant "${VARIANT:-V1}" --gate "${DRY_GATE:-GATE_FT_PASS}" --model "$MODEL"
fi

# ---- the gate, the prompt, the Gate-FT context ----
GATE_DECISION=missing
if [ -f "$GT/gate_ft.json" ]; then GATE_DECISION=$(jget "$GT/gate_ft.json" decision); fi
if [ "$GATE_DECISION" != GATE_FT_PASS ]; then
  echo "S6 refused: Gate-FT decision $GATE_DECISION ($GT/gate_ft.json). Section 0: sections 1-3 and 5-9 run only after" \
    "a recorded GATE_FT_PASS (on GATE_FT_FAIL or INCOMPLETE S6 is not run)" >&2
  exit 4
fi
[ -f "$SEL" ] || { echo "missing $SEL (the gate-fix stage 1 selection; run_gatefix.sh writes it)" >&2; exit 1; }
SEL_PROMPT=$(jget "$SEL" gate_ft_prompt)
VARIANT="${VARIANT:-$SEL_PROMPT}"
if [ "$VARIANT" != "$SEL_PROMPT" ]; then
  echo "VARIANT=$VARIANT is not selection.json's gate_ft_prompt $SEL_PROMPT: section 2 registers no other variant" >&2
  exit 2
fi
for s in 0 1 2; do
  adapter_done "$GT/adapters/s$s" || { echo "Gate-FT adapter $GT/adapters/s$s is missing or incomplete (P0 s$s)" >&2; exit 1; }
done
echo "run_ftprune: model $MODEL, root $OUT, variant $VARIANT, Gate-FT $GATE_DECISION, P3 $([ "$RUN_P3" = 1 ] && echo run || echo cut), stages:$STAGES"

# ================= stage A: the zero-shot like pass of the TRAIN set =================
if want A; then
  echo "== stage A: zero-shot like pass of $TRAIN"
  for f in "$TRAIN" "$GSPLIT"; do [ -f "$f" ] || { echo "missing $f (Gate-FT stage A, gateft_data)" >&2; exit 1; }; done
  if [ "$DRY_RUN" = 1 ]; then dry_record core; fi
  if ! freeze_check core; then
    echo "stage A refused: scoring a zero-shot panel waits for the full Amendment-3 record (section 0)" >&2; exit 4
  fi
  score "$TRAIN" "$ZS"
fi

# ================= stage B: signals, subsets, pruned TRAIN files, manifest (CPU) =================
if want B; then
  echo "== stage B: signals, subsets, pruned TRAIN files, manifest"
  [ -f "$ZS/report.json" ] || { echo "missing $ZS/report.json (stage A)" >&2; exit 1; }
  if [ -f "$ZS/FAILED_INTEGRITY" ]; then
    echo "$ZS failed E1 twice (FAILED_INTEGRITY): S6 has no U signal and is reported as not run (section 2)" >&2; exit 1
  fi
  step "$MAN" "$TRAIN" "$GSPLIT" "$ZS/scores.csv.gz" "$ZS/report.json" "$RATINGS" src/confrec/ftprune.py \
      src/confrec/forensics.py src/confrec/split_panel.py -- \
    "$PY" -m src.confrec.ftprune signals --train "$TRAIN" --zs_dir "$ZS" --raw "$RAW" --variant "$VARIANT" \
      --split_report "$GSPLIT" --out_dir "$OUT"
fi

# ================= stage C: FREEZE prune =================
if want C; then
  echo "== stage C: freeze checks (amendment, core, prune)"
  [ -f "$MAN" ] || { echo "missing $MAN (stage B)" >&2; exit 1; }
  if [ "$DRY_RUN" = 1 ]; then
    # rehearsal: an empty pilot log fails the prune check, and stage D then refuses; the human step is dry_record
    [ -f "$DRY/PILOT_LOG.empty.md" ] || printf '# empty temporary pilot log\n' > "$DRY/PILOT_LOG.empty.md"
    if "$PY" -m src.confrec.ftgrid_freeze --check --stage prune --pilot_log "$DRY/PILOT_LOG.empty.md" \
        "${PRUNE_ARGS[@]}" 2> /dev/null; then
      echo "DRY_RUN: the prune freeze check passed on an empty pilot log" >&2; exit 1
    fi
    if (MARK="$DRY/no.prune.ok"; require_freeze D) 2> /dev/null; then
      echo "DRY_RUN: stage D did not refuse without the freeze record" >&2; exit 1
    fi
    echo "[dry] freeze rehearsal: an empty pilot log fails the prune check and stage D refuses; recording the temporary log"
    dry_record amendment
    dry_record core
    dry_record prune
  fi
  for st in amendment core prune; do
    if ! freeze_check "$st"; then
      echo "stage C: the $st record is not in $PILOT_LOG; record the lines ftgrid_freeze prints for that stage (see" \
        "above) and rerun: the outputs of stages A and B are kept" >&2
      exit 4
    fi
  done
  "$PY" -m src.confrec.ftprune verify --out_dir "$OUT" || { echo "stage C: a file of $MAN changed after stage B" >&2; exit 1; }
  REC=$(record)
  mkdir -p "$(dirname "$MARK")"
  if [ ! -f "$MARK" ] || [ "$(cat "$MARK")" != "$REC" ]; then printf '%s\n' "$REC" > "$MARK"; fi
  FREEZE_OK=1
fi

# ================= stage D: train and score the arms (GPU) =================
if want D; then
  echo "== stage D: train and score the pruning arms"
  require_freeze D || exit 4
  if ! "$PY" -m src.confrec.ftprune verify --out_dir "$OUT"; then
    echo "stage D refused: a file of $MAN no longer has its frozen sha1 (section 6)" >&2; exit 4
  fi
  [ -f "$CONFP" ] || { echo "missing $CONFP (the CONFIRM panel Gate-FT scores)" >&2; exit 1; }
  recipe
  echo "Gate-FT recipe: micro-batch $MICRO x accumulation $ACCUM, max_len $MAXLEN, variant $VARIANT"
  for s in 0 1 2 3 4; do
    arm_run P1 "$s"
    arm_run P2 "$s"
    if [ "$s" -ge 3 ]; then arm_run P0 "$s"; fi
  done
  if [ "$RUN_P3" = 1 ]; then
    for s in 0 1 2 3 4; do arm_run P3 "$s"; done
  else
    echo "[stage D] P3 not run: cut by RUN_P3=0 (sections 6 and 10, the 2026-10-22 checkpoint)"
  fi
fi

# ================= stage E: the registered analysis (CPU) =================
if want E; then
  echo "== stage E: analysis"
  require_freeze E || exit 4
  ARMS=P0,P1,P2
  if [ "$RUN_P3" = 1 ]; then ARMS=P0,P1,P2,P3; fi
  set +e
  "$PY" -m src.confrec.ftprune analyze --confirm_panel "$CONFP" --split_report "$GSPLIT" --scores_root "$SC" \
    --gateft_scores "$GT/scores" --adapters_root "$ADIR" --gateft_adapters "$GT/adapters" --variant "$VARIANT" \
    --manifest "$MAN" --arms "$ARMS" --out "$RES" --n_boot 2000 --seed 0
  rc=$?
  set -e
  echo "ftprune analyze exit code $rc (0 = labelled, 2 = INCOMPLETE; $RES)"
  if [ "$rc" != 0 ]; then exit "$rc"; fi
fi
echo "run_ftprune: done (stages:$STAGES)"
