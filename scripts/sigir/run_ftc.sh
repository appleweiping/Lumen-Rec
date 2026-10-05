#!/usr/bin/env bash
# FT-C on Toys (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_6.md item 8; the control of idea-stage/PREREG_AMENDMENT_3.md section 9):
# the within-item label-permuted adapters p0, p1 on the Toys TRAIN panel, trained with the recipe of the Toys adapters s0-s2,
# scored like them, then ftgrid_report with the six models. A NEW script, so that no bound file changes. GPU server; one job
# at a time through scripts/sigir/gpu_queue.sh, queued as the single job
#     cd /root/autodl-tmp/lumen-rec && bash scripts/sigir/run_ftc.sh toys
#   usage: bash scripts/sigir/run_ftc.sh toys     Toys only: any other dataset is refused (exit 2). ML-1M's FT-C adapters are
#                                                 run_ftgrid.sh's `perm` stage; games and sports have no FT-C
#   env:   MODEL     Qwen3-8B dir (default /root/autodl-tmp/lumen/models/Qwen3-8B); any other backbone is refused
#          OUT_ROOT  the registered root outputs/confrec/ftgrid, the only root outside DRY_RUN: p0 / p1 sit beside s0-s2
#          VARIANT   default gate_ft_prompt of outputs/confrec/gatefix/dev/selection.json; any other value is refused
#          STAGES    comma list of 1-4 or all (default all)
#          PYTHON    interpreter (skips the conda activation);  DRY_RUN=1  CPU rehearsal, see below
#   gates    (before stages 2-4, in this order; any failure exits 4 and starts nothing): the recorded Gate-FT decision is
#            GATE_FT_PASS (Amendment 3 section 0); `ftgrid_freeze --check --stage amendment` and `--stage core` (the full
#            record: the four registered ftgrid_split.json, Toys included; a CPU check, run again at every start); the gate-fix
#            decisions (selection.json, and gate.json when a fix was found); and the FT-C record (Addendum 6 section 11): the
#            sha1 of this script and of src/confrec/ftc_panel.py are in docs/sigir/PILOT_LOG.md. Stage 1 (CPU, outcome-free)
#            needs only its inputs.
#   stage 1  (CPU) ftc_panel build: outputs/confrec/ftgrid/ftc/toys/train_perm.jsonl = ftgrid_data.permute_within_item(
#            panels/toys/train.jsonl rows, seed 0) (imported, never copied), asserted as ftgrid_data asserts ML-1M's file (every
#            item's label sum and rating multiset preserved, nothing else changed), and train_perm.manifest.json (sha1 of
#            both files, labels changed, seed, code sha1). panels/toys is not touched
#   stage 2  (GPU) train p0, p1 (seeds 0, 1): train_lora_yesno with exactly the arguments the Toys adapters s0-s2 recorded in
#            adapters/toys/s0/train_config.json (s0-s2 must agree and be the registered recipe; the three flags --train, --out
#            and --seed are the only ones replaced, ftc_panel recipe); the trained config is checked against s0's again and
#            the adapter's provenance (the sha1 of the permuted panel) is written. An adapter with train_config.json and
#            weights is never retrained, but must carry the provenance of this permuted panel. Before the first training
#            the record of the s0 like pass (stage 3's arguments) is validated too, so a wrong record stops the job early
#   stage 3  (GPU) like on panels/toys/eval.jsonl (CAL and TEST rows) for p0, p1: pyes_scorer with exactly the arguments of
#            the like pass of s0 (run.key and the config of scores/toys/s0/like/report.json, nothing else of that report is
#            read; ftc_panel scoring), E1 as run_ftgrid.sh (a failing run is moved to DIR.e1fail.<time> and rerun once, a second
#            failure leaves DIR/FAILED_INTEGRITY: the adapter is reported as missing, never replaced); the recorded scoring
#            config of each pass equals s0's except `lora` (ftc_panel verify_scores)
#   stage 4  (CPU) ftgrid_report (unchanged) --models zeroshot,s0,s1,s2,p0,p1 --out outputs/confrec/ftgrid/report/toys_ftc.json
#            (and toys_ftc_tables.csv): a second file; the registered report/toys.json is never rewritten (checked)
# Layout: adapters/toys/p0, p1 and scores/toys/p0/like, p1/like are those of FTGRID_IMPL_SPEC.md (the models of ML-1M's FT-C);
# ftc/toys/ holds the permuted panel and its manifest; build/toys_ftc/done holds the step markers.
# Re-runnable: a finished step is skipped (markers newer than their inputs); a scoring dir is skipped when report.json exists and
# run.key (panel sha1, model, variant, adapter-weights sha1, args) is unchanged, else it is moved to DIR.stale.<time>.
# GPU time (from the file times of the Toys s0-s2 adapters and like passes): 48 min per training and 16 min per like pass,
# so about 2.1 h in all, a few minutes of process start-up included (Amendment 3 section 10 budgets about 3).
# DRY_RUN=1: the same chain, CPU only, on run_ftgrid.sh's own DRY_RUN world of Toys (outputs/confrec/ftgrid_dryrun: synthetic raw
# data, panels, Gate-FT context, s0-s2 and their like passes, the temporary pilot log), built first by `DRY_RUN=1 STAGES=0,1,2,3
# run_ftgrid.sh toys` (skipped once its like passes exist). The trainer and the scorer are run_ftgrid.sh's stand-ins (the real
# argparse, the real scorer code with a fake model); every other step is the real code, the freeze checks included (on the
# temporary pilot log). The rehearsal first shows that an empty pilot log fails the amendment, core and FT-C record checks; the
# human step of recording the FT-C files is done on the temporary log unless DRY_NO_RECORD=1 (the record is then missing and the
# run is refused). DRY_GATE and DRY_E1_FAIL act as in run_ftgrid.sh (DRY_E1_FAIL=p0/like: the p0 like pass fails E1).
# Exit codes (as run_ftgrid.sh): 0 done; 1 error or an inconsistent recorded input; 2 usage or refused input; 4 refused by the
# freeze / gate rules.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
D="${1:-}"
case "$D" in
  toys) ;;
  ml1m|games|sports)
    echo "run_ftc.sh runs FT-C on Toys only (Amendment 3 addendum 6 item 8): ML-1M's FT-C adapters are run_ftgrid.sh's perm" \
      "stage, $D has no FT-C" >&2
    exit 2 ;;
  *) echo "usage: bash scripts/sigir/run_ftc.sh toys" >&2; exit 2 ;;
esac
DRY_RUN="${DRY_RUN:-0}"
DRY_NO_RECORD="${DRY_NO_RECORD:-0}"
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
if [ "$DRY_RUN" != 1 ] && [ "$OUT_ROOT" != "$REG" ]; then
  echo "OUT_ROOT=$OUT_ROOT: FT-C on Toys has one registered root, $REG (p0 and p1 sit beside the adapters s0-s2)" >&2; exit 2
fi
if [ "$(basename "$MODEL")" != Qwen3-8B ]; then
  echo "$MODEL is not Qwen3-8B: FT-C on Toys is registered for the Gate-FT backbone only (Addendum 6 item 8)" >&2; exit 2
fi
STAGES=" $(printf '%s' "${STAGES:-all}" | tr ',' ' ') "
for s in $STAGES; do
  case "$s" in 1|2|3|4|all) ;; *) echo "unknown stage '$s' in STAGES (1-4, all)" >&2; exit 2 ;; esac
done
case "$STAGES" in *" all "*) STAGES="$STAGES 1 2 3 4 " ;; esac
want() { case "$STAGES" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

P="$OUT_ROOT/panels/toys"           # run_ftgrid.sh's panels of Toys (never written here)
S="$OUT_ROOT/scores/toys"           # scores/toys/<model>/like: s0-s2 and zeroshot are run_ftgrid.sh's, p0 and p1 are ours
ADIR="$OUT_ROOT/adapters/toys"
FTC="$OUT_ROOT/ftc/toys"            # the permuted panel and its manifest
PERM="$FTC/train_perm.jsonl"
PMAN="$FTC/train_perm.manifest.json"
B="$OUT_ROOT/build/toys_ftc"        # step markers
DONE="$B/done"
SPLIT="$P/ftgrid_split.json"
REP="$OUT_ROOT/report/toys_ftc.json"
REGREP="$OUT_ROOT/report/toys.json" # the registered Toys report: never rewritten
FTC_FILES=(scripts/sigir/run_ftc.sh src/confrec/ftc_panel.py)    # Addendum 6 section 11: recorded before the training
if [ "$DRY_RUN" = 1 ]; then
  DRY="$OUT_ROOT/_dry"
  G="$DRY/gatefix"; GT="$DRY/gateft"; RAW="$DRY/raw"; PILOT_LOG="$DRY/PILOT_LOG.md"
  N_BOOT=200
  CORE_SPLITS=(--split "$SPLIT")
else
  G=outputs/confrec/gatefix; GT=outputs/confrec/gateft; RAW=data/raw; PILOT_LOG=docs/sigir/PILOT_LOG.md
  N_BOOT=2000                       # A3 section 3: 2,000 user resamples
  CORE_SPLITS=(--split "$REG/panels/ml1m/ftgrid_split.json" --split "$REG/panels/toys/ftgrid_split.json"
    --split "$REG/panels/games/ftgrid_split.json" --split "$REG/panels/sports/ftgrid_split.json")
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
# >= 0.95 on the main prompts; a swap run's donor prompts also need the censored-2 share and no overlength prompt
# (run_ftgrid.sh's function, verbatim: tests/test_confrec_ftc.py compares the two)
e1_ok() {
  "$PY" - "$1/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
n = int(r["n_main_prompts"])
ok = n > 0 and int(r["censored_main"].get("2", 0)) <= 0.005 * n and int(r["n_overlength"]) == 0 \
    and float(r["mean_yes_no_mass"]) >= 0.95
if "censored_swap" in r:
    ns = int(r["swap_prompts"])
    ok = ok and int(r["censored_swap"].get("2", 0)) <= 0.005 * ns and int(r["censored_swap"].get("3", 0)) == 0 \
        and int(r.get("swap_n_overlength") or 0) == 0
sys.exit(0 if ok else 1)
PY
}
# MPY runs the entry points that load the model (the trainer, the scorer). Under DRY_RUN it is fake_py: run_ftgrid.sh's
# stand-ins; any other call runs as is
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
# score DATA DIR --lora A: pyes_scorer with SARGS, the scorer flags of the like pass of s0 (fp16, top-50 logprobs, max_model_len
# 4096, 100-user chunks, the selected variant, yes/no readout, like: run_ftgrid.sh's score(), as its record says); completion
# marker DIR/report.json, run.key = panel sha1 + model + variant + adapter weights sha1 + args (the format of run_ftgrid.sh)
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
  "$MPY" -m src.confrec.pyes_scorer --data "$data" --output "$dir" --model "$MODEL" "${SARGS[@]}" "$@"
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
# the checks of the gates; the optional argument is another pilot log (the DRY_RUN rehearsal of an empty one)
check_amendment() { "$PY" -m src.confrec.ftgrid_freeze --check --stage amendment --pilot_log "${1:-$PILOT_LOG}"; }
check_core() { "$PY" -m src.confrec.ftgrid_freeze --check --stage core --pilot_log "${1:-$PILOT_LOG}" "${CORE_SPLITS[@]}"; }
check_record() { "$PY" -m src.confrec.ftc_panel record --pilot_log "${1:-$PILOT_LOG}" --files "${FTC_FILES[@]}"; }
# dry_record: DRY_RUN only, the human step: write the sha1 lines of this script and of ftc_panel.py into the temporary log
dry_record() {
  if [ "$DRY_NO_RECORD" = 1 ]; then echo "[dry] DRY_NO_RECORD: the FT-C record is not written"; return 0; fi
  if check_record > /dev/null 2>&1; then return 0; fi
  "$PY" -m src.confrec.ftc_panel record --pilot_log "$PILOT_LOG" --files "${FTC_FILES[@]}" --print >> "$PILOT_LOG"
}
# rehearse: DRY_RUN only: an empty pilot log fails the amendment, core and FT-C record checks
rehearse() {
  [ -f "$DRY/PILOT_LOG.empty.md" ] || printf '# empty temporary pilot log\n' > "$DRY/PILOT_LOG.empty.md"
  if check_amendment "$DRY/PILOT_LOG.empty.md" > /dev/null 2>&1 || check_core "$DRY/PILOT_LOG.empty.md" > /dev/null 2>&1 \
      || check_record "$DRY/PILOT_LOG.empty.md" > /dev/null 2>&1; then
    echo "DRY_RUN: a freeze or record check passed on an empty pilot log" >&2; exit 1
  fi
  echo "[dry] gate rehearsal: an empty pilot log fails the amendment, core and FT-C record checks"
}
# gates: everything the registered text asks before a GPU job (Amendment 3 section 0, Addendum 6 sections 8 and 11)
gates() {
  local dec=missing
  if [ -f "$GT/gate_ft.json" ]; then dec=$(jget "$GT/gate_ft.json" decision); fi
  if [ "$dec" != GATE_FT_PASS ]; then
    echo "FT-C refused: Gate-FT decision $dec (section 0: FT-C runs only after a recorded GATE_FT_PASS, $GT/gate_ft.json)" >&2
    exit 4
  fi
  if [ "$DRY_RUN" = 1 ]; then rehearse; dry_record; fi
  check_amendment || { echo "FT-C refused: the freeze record of Amendment 3 (stage amendment) is not in $PILOT_LOG" >&2; exit 4; }
  check_core || { echo "FT-C refused: the full record of Amendment 3 (stage core) is not in $PILOT_LOG" >&2; exit 4; }
  if [ "$SEL_DECISION" = FIX_FOUND ] && [ ! -f "$G/confirm/gate.json" ]; then
    echo "FT-C refused: selection.json found a fix but the gate-fix confirm stage has recorded no gate.json (section 4)" >&2
    exit 4
  fi
  check_record || exit 4
}
# recipe: RECIPE = the trainer flags the Toys adapters s0-s2 recorded in their train_config.json, except --train, --out and
# --seed (ftc_panel validates them: one registered recipe, this model, variant and TRAIN file)
recipe() {
  local out rc=0
  out=$("$PY" -m src.confrec.ftc_panel recipe --adapters "$ADIR" --model "$MODEL" --variant "$VARIANT" \
    --train_ref "$P/train.jsonl" --split "$SPLIT") || rc=$?
  [ "$rc" = 0 ] || exit "$rc"
  mapfile -t RECIPE < <(printf '%s\n' "$out" | tr -d '\r')
  echo "[recipe] trainer flags recorded by s0-s2 (--train, --out, --seed replaced): ${RECIPE[*]}"
}
# scoring_recipe: SARGS = the scorer flags of the like pass of s0 (run.key and the config of its report.json)
scoring_recipe() {
  local out rc=0
  out=$("$PY" -m src.confrec.ftc_panel scoring --scores "$S" --adapters "$ADIR" --data "$P/eval.jsonl" --model "$MODEL" \
    --variant "$VARIANT") || rc=$?
  [ "$rc" = 0 ] || exit "$rc"
  mapfile -t SARGS < <(printf '%s\n' "$out" | tr -d '\r')
  echo "[recipe] scorer flags recorded by the like pass of s0 (--data, --output, --model, --lora replaced): ${SARGS[*]}"
}
# verify_adapter SEED [--write]: p<SEED>'s recorded arguments equal s0's except --train, --out, --seed; its provenance
verify_adapter() {
  local seed=$1; shift
  "$PY" -m src.confrec.ftc_panel verify_adapter --adapters "$ADIR" --p_seed "$seed" --train "$PERM" \
    --perm_manifest "$PMAN" "$@" || exit $?
}
# train_adapter TRAIN OUT SEED: the Toys recipe of s0-s2 on the permuted panel; skipped when train_config.json and weights exist
train_adapter() {
  local train=$1 out=$2 seed=$3
  if adapter_done "$out"; then verify_adapter "$seed"; echo "[skip] adapter $out exists"; return 0; fi
  [ -f "$train" ] || { echo "missing $train (stage 1)" >&2; exit 1; }
  rm -rf "$out"
  "$MPY" -m src.confrec.train_lora_yesno --train "$train" --out "$out" --seed "$seed" "${RECIPE[@]}" 2>&1 \
    | grep -vE "it/s\]|s/it\]" || true
  adapter_done "$out" || { echo "training of $out did not finish" >&2; exit 1; }
  verify_adapter "$seed" --write
}
# registered_sha: the sha1 of the registered Toys report and its tables ('-' when absent)
registered_sha() {
  local f
  for f in "$REGREP" "${REGREP%.json}_tables.csv"; do
    if [ -f "$f" ]; then sha1sum "$f" | cut -d' ' -f1; else echo -; fi
  done
}

# ---- DRY_RUN: run_ftgrid.sh's synthetic world of Toys (its stages 0-3) and its stand-ins ----
if [ "$DRY_RUN" = 1 ]; then
  mkdir -p "$OUT_ROOT/_dry"
  FAKES="$OUT_ROOT/_dry/ftgrid_fakes.py"
  if [ -f "$SPLIT" ] && [ -f "$FAKES" ] && [ -f "$S/zeroshot/like/report.json" ] && [ -f "$S/s0/like/report.json" ] \
      && [ -f "$S/s1/like/report.json" ] && [ -f "$S/s2/like/report.json" ]; then
    echo "[skip] run_ftgrid.sh's DRY_RUN world for toys exists ($OUT_ROOT)"
  else
    echo "== DRY_RUN: run_ftgrid.sh's synthetic world for toys (its stages 0-3: panels, s0-s2, their like passes)"
    if ! DRY_RUN=1 OUT_ROOT="$OUT_ROOT" MODEL="$MODEL" STAGES=0,1,2,3 PYTHON="$PY" bash scripts/sigir/run_ftgrid.sh toys \
        > "$OUT_ROOT/_dry/ftgrid.toys.log" 2>&1; then
      tail -n 40 "$OUT_ROOT/_dry/ftgrid.toys.log" >&2
      echo "DRY_RUN: run_ftgrid.sh toys failed (log: $OUT_ROOT/_dry/ftgrid.toys.log)" >&2; exit 1
    fi
  fi
fi

# ---- the prompt ----
SEL="$G/dev/selection.json"
[ -f "$SEL" ] || { echo "missing $SEL (the gate-fix stage 1 selection; run_gatefix.sh writes it)" >&2; exit 1; }
SEL_PROMPT=$(jget "$SEL" gate_ft_prompt)
SEL_DECISION=$(jget "$SEL" decision)
VARIANT="${VARIANT:-$SEL_PROMPT}"
if [ "$VARIANT" != "$SEL_PROMPT" ] && [ "$DRY_RUN" != 1 ]; then
  echo "VARIANT=$VARIANT is not selection.json's gate_ft_prompt $SEL_PROMPT: section 2 registers no other variant" >&2
  exit 2
fi
if [ -f "$SPLIT" ] && [ "$(jget "$SPLIT" variant)" != "$VARIANT" ]; then
  echo "$SPLIT was built under variant $(jget "$SPLIT" variant), not $VARIANT" >&2; exit 2
fi
echo "run_ftc toys: model $MODEL, root $OUT_ROOT, variant $VARIANT, stages:$STAGES"

# ---- the registered gates come first: a refused run touches nothing ----
GATED=0
for s in 2 3 4; do if want "$s"; then GATED=1; fi; done
if [ "$GATED" = 1 ]; then gates; fi

# ================= stage 1: the permuted TRAIN panel (CPU) =================
if want 1; then
  echo "== stage 1: the permuted TRAIN panel (toys)"
  for f in "$P/train.jsonl" "$SPLIT"; do
    [ -f "$f" ] || { echo "missing $f (run_ftgrid.sh stage 0)" >&2; exit 1; }
  done
  mkdir -p "$FTC"
  step "$PMAN" "$P/train.jsonl" "$SPLIT" src/confrec/ftc_panel.py src/confrec/ftgrid_data.py -- \
    "$PY" -m src.confrec.ftc_panel build --domain toys --train "$P/train.jsonl" --out_dir "$FTC" --split "$SPLIT" --seed 0
fi

# ================= stage 2: the permuted adapters p0, p1 (GPU) =================
if want 2; then
  echo "== stage 2: adapters p0, p1 (toys)"
  { [ -f "$PERM" ] && [ -f "$PMAN" ]; } || { echo "missing $PERM or $PMAN (stage 1)" >&2; exit 1; }
  recipe
  scoring_recipe       # pre-flight: the recorded like pass of s0 must be the registered one before any GPU hour is spent
  for seed in 0 1; do train_adapter "$PERM" "$ADIR/p$seed" "$seed"; done
fi

# ================= stage 3: like on eval.jsonl for p0, p1 (GPU) =================
if want 3; then
  echo "== stage 3: like on eval.jsonl for p0, p1 (toys)"
  for f in "$P/eval.jsonl" "$PMAN"; do
    [ -f "$f" ] || { echo "missing $f (run_ftgrid.sh stage 0 / this script's stage 1)" >&2; exit 1; }
  done
  scoring_recipe
  for seed in 0 1; do
    adapter_done "$ADIR/p$seed" || { echo "adapter p$seed of toys is missing or incomplete: $ADIR/p$seed (stage 2)" >&2; exit 1; }
    verify_adapter "$seed"
    score "$P/eval.jsonl" "$S/p$seed/like" --lora "$ADIR/p$seed"
    "$PY" -m src.confrec.ftc_panel verify_scores --scores "$S" --adapters "$ADIR" --p_seed "$seed" || exit $?
  done
fi

# ================= stage 4: ftgrid_report with the six models (CPU) =================
if want 4; then
  echo "== stage 4: ftgrid_report with zeroshot, s0-s2, p0, p1 -> $REP (toys)"
  for m in zeroshot s0 s1 s2; do
    [ -f "$S/$m/like/report.json" ] || { echo "missing the registered like pass $S/$m/like (run_ftgrid.sh toys)" >&2; exit 1; }
  done
  for m in p0 p1; do
    { [ -f "$S/$m/like/report.json" ] || [ -f "$S/$m/like/FAILED_INTEGRITY" ]; } \
      || { echo "missing the like pass $S/$m/like (stage 3)" >&2; exit 1; }
  done
  RDEPS=("$SPLIT" src/confrec/ftgrid_report.py)
  for f in "$S"/*/*/report.json; do if [ -f "$f" ]; then RDEPS+=("$f"); fi; done
  REG_BEFORE=$(registered_sha)
  mkdir -p "$(dirname "$REP")"
  step "$REP" "${RDEPS[@]}" -- \
    "$PY" -m src.confrec.ftgrid_report --domain toys --split "$SPLIT" --panels "$P" --scores_root "$OUT_ROOT/scores" \
      --models zeroshot,s0,s1,s2,p0,p1 --raw "$RAW" --out "$REP" --n_boot "$N_BOOT" --seed 0
  if [ "$(registered_sha)" != "$REG_BEFORE" ]; then
    echo "the registered Toys report $REGREP changed: it is never rewritten (Addendum 6 item 8)" >&2; exit 1
  fi
fi
for m in p0 p1; do
  if [ -f "$S/$m/like/FAILED_INTEGRITY" ]; then
    echo "NOTE: $S/$m/like is FAILED_INTEGRITY: $m is reported as missing, never replaced (section 2)" >&2
  fi
done
echo "run_ftc toys: done (stages:$STAGES)"
