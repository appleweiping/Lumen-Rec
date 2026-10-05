#!/usr/bin/env bash
# FT-Q, the item-only teacher control (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_8.md section 2; it replaces the withdrawn Toys
# permutation run of addendum 6 item 8) for ONE dataset: two adapters p0, p1 (seeds 0 and 1) trained on train_q.jsonl, the registered
# TRAIN panel with its labels replaced by an item-only teacher (label 1 for the round(beta n) examples with the largest prior-only
# item mean q-hat, ties by a seed-0 random key), with exactly the arguments of the dataset's real adapters s0-s2, scored with exactly
# the arguments of the real s0 like pass, and read by ftgrid_report / ftgrid_extra as the control regime of an own root. A NEW script
# (no bound file changes). GPU server; one job at a time through scripts/sigir/gpu_queue.sh, one job per dataset:
#     cd /root/autodl-tmp/lumen-rec && bash scripts/sigir/run_ftq.sh ml1m        (then toys; games and sports unless cut)
#   usage: bash scripts/sigir/run_ftq.sh D       D in ml1m, toys, games, sports (the only argument)
#   env:   MODEL     Qwen3-8B dir (default /root/autodl-tmp/lumen/models/Qwen3-8B); any other backbone is refused
#          OUT_ROOT  the FT-Q root outputs/confrec/ftgrid_q, the only root outside DRY_RUN
#          VARIANT   default gate_ft_prompt of outputs/confrec/gatefix/dev/selection.json; any other value is refused
#          STAGES    comma list of 1-4 or all (default all)
#          PYTHON    interpreter (skips the conda activation);  DRY_RUN=1  CPU rehearsal, see below
#   reads (never written): the grid root outputs/confrec/ftgrid (panels/D: train.jsonl, eval.jsonl, ftgrid_split.json; adapters/D and
#          scores/D of the real s0-s2; ML-1M: Gate-FT's adapters outputs/confrec/gateft/adapters and train.jsonl) and the nested slot's
#          stage-1 files outputs/confrec/ftmethod/D/{train_qhat.csv.gz, qhat_manifest.json} (run_ftmethod.sh D, STAGES=1)
#   writes (the FT-Q root only, outputs/confrec/ftgrid_q): ftq/D/{train_q.jsonl, train_q.manifest.json}; adapters/D/p0, p1; scores/D/
#          {zeroshot, s0, s1, s2} = symlinks to the real score directories (linked, never copied or rescored), scores/D/p0/like and
#          p1/like; report/D.json (+ D_tables.csv); build/D/ (step markers). Nothing below outputs/confrec/ftgrid is written.
#   gates    (before stages 2-4, in this order; any failure exits 4 and starts nothing): the recorded Gate-FT decision is GATE_FT_PASS
#            (Amendment 3 section 0); `ftgrid_freeze --check --stage amendment` and `--stage core` (the full record, a CPU check run
#            again at every start); the gate-fix decisions (selection.json, and gate.json when a fix was found); and the FT-Q record
#            (addendum 8 section 4): the sha1 of this script and of src/confrec/ftq_panel.py are in docs/sigir/PILOT_LOG.md. Stage 1
#            (CPU, outcome-free) needs only its inputs.
#   stage 1  (CPU) ftq_panel build: train_q.jsonl = train.jsonl with candidate_labels replaced by the teacher labels of addendum 8
#            section 2 and nothing else changed, and train_q.manifest.json (sha1 of train.jsonl, train_qhat.csv.gz, train_q.jsonl,
#            beta, n, k, tie seed, code sha1). Asserted: the stage-1 q-hat rows align one to one with the TRAIN examples (in
#            train.jsonl order) and match their manifest, whose recorded TRAIN sha1 equals the panel's; the teacher rate equals beta
#            exactly; the teacher is a function of q-hat and the tie key only; nothing else of the rows changed
#   stage 2  (GPU) train p0, p1 (seeds 0, 1): train_lora_yesno with exactly the arguments the real adapters s0-s2 recorded in their
#            train_config.json (ML-1M: Gate-FT's; s0-s2 must agree and be the registered recipe; --train, --out and --seed are the
#            only flags replaced; ftq_panel recipe); the trained config is checked against s0's again and the adapter's provenance
#            (the sha1 of the teacher panel) is written. An adapter with train_config.json and weights is never retrained, but must
#            carry the provenance of this teacher panel. Before the first training the record of the real s0 like pass (stage 3's
#            arguments) is validated too, so a wrong record stops the job early
#   stage 3  (GPU) like on panels/D/eval.jsonl (CAL and TEST rows) for p0, p1: pyes_scorer with exactly the arguments of the real s0 like
#            pass (its run.key and the config of its report.json, nothing else of that report is read; ftq_panel scoring), E1 as
#            run_ftgrid.sh (a failing run is moved to DIR.e1fail.<time> and rerun once, a second failure leaves DIR/FAILED_INTEGRITY:
#            the adapter is reported as missing, never replaced); the recorded scoring config of each pass equals s0's except `lora`
#            (ftq_panel verify_scores). Then the real scores are linked into the root (ftq_panel link)
#   stage 4  (CPU) ftgrid_report (unchanged) --models zeroshot,s0,s1,s2,p0,p1 --scores_root outputs/confrec/ftgrid_q/scores --out
#            outputs/confrec/ftgrid_q/report/D.json: the teacher adapters are the control (PERM) models of the report; the registered
#            report/D.json under the grid root is never touched (checked)
# Re-runnable: a finished step is skipped (markers newer than their inputs); a scoring dir is skipped when report.json exists and run.key
# (panel sha1, model, variant, adapter-weights sha1, args) is unchanged, else it is moved to DIR.stale.<time>.
# GPU time (addendum 8 section 3; checked against the file times of the real adapters and like passes): ML-1M 2 x 46 min training +
# 2 x 9 min scoring = 1.8 h (the registered 6 min per like pass is the budget; the real ML-1M like pass took about 9 min); each Amazon
# dataset 2 x 48 min + 2 x 16 min = 2.1 h.
# DRY_RUN=1: the same chain, CPU only, on run_ftgrid.sh's own DRY_RUN world of D (outputs/confrec/ftgrid_dryrun: synthetic raw data, panels,
# Gate-FT context, s0-s2 and their like passes, the temporary pilot log), built first by `DRY_RUN=1 STAGES=0,1,2,3 run_ftgrid.sh D` and
# the nested slot's stage 1 (train_lora_offset qhat, outputs/confrec/ftmethod_dryrun/D); skipped once they exist. The FT-Q root is
# outputs/confrec/ftgrid_q_dryrun. The trainer and the scorer are run_ftgrid.sh's stand-ins (the real argparse, the real scorer code with
# a fake model); every other step is the real code, the freeze checks included (on the temporary pilot log). The rehearsal first shows
# that an empty pilot log fails the amendment, core and FT-Q record checks; the human step of recording the FT-Q files is done on the
# temporary log unless DRY_NO_RECORD=1 (the record is then missing and the run is refused). Without symlinks (a Windows machine) DRY_RUN
# copies the real scores instead of linking them. DRY_GATE and DRY_E1_FAIL act as in run_ftgrid.sh (DRY_E1_FAIL=p0/like: the p0 like fails E1).
# DRY_NO_WORLD=1 neither builds nor repairs that world (a test hook): a damaged world is refused as in a real run, not healed first.
# Exit codes (as run_ftgrid.sh): 0 done; 1 error or an inconsistent recorded input; 2 usage or refused input; 4 refused by the freeze /
# gate rules.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
D="${1:-}"
case "$D" in
  ml1m|toys|games|sports) ;;
  *) echo "usage: bash scripts/sigir/run_ftq.sh {ml1m|toys|games|sports}" >&2; exit 2 ;;
esac
DRY_RUN="${DRY_RUN:-0}"
DRY_NO_RECORD="${DRY_NO_RECORD:-0}"
DRY_NO_WORLD="${DRY_NO_WORLD:-0}"
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  set +u; source /root/miniconda3/etc/profile.d/conda.sh; conda activate lumen; set -u
  PY=python
fi
REG=outputs/confrec/ftgrid        # the registered grid root: read, never written
REGQ=outputs/confrec/ftgrid_q     # the FT-Q root: everything this script writes
if [ "$DRY_RUN" = 1 ]; then
  MODEL="${MODEL:-dryrun/Qwen3-8B}"
  OUT_ROOT="${OUT_ROOT:-outputs/confrec/ftgrid_q_dryrun}"
  GRID=outputs/confrec/ftgrid_dryrun
  QH=outputs/confrec/ftmethod_dryrun
else
  MODEL="${MODEL:-/root/autodl-tmp/lumen/models/Qwen3-8B}"
  OUT_ROOT="${OUT_ROOT:-$REGQ}"
  GRID="$REG"
  QH=outputs/confrec/ftmethod
fi
OUT_ROOT="${OUT_ROOT%/}"
OUT_ROOT="${OUT_ROOT#./}"
if [ "$DRY_RUN" = 1 ]; then
  for r in "$REG" "$REGQ" outputs/confrec/ftgrid_llama outputs/confrec/ftmethod "$GRID" "$QH"; do
    if [ "$OUT_ROOT" = "$r" ]; then echo "DRY_RUN=1 never writes to a registered output root or to the world it reads ($OUT_ROOT)" >&2; exit 2; fi
  done
elif [ "$OUT_ROOT" != "$REGQ" ]; then
  echo "OUT_ROOT=$OUT_ROOT: FT-Q has one registered root, $REGQ (the grid root $REG is read, never written)" >&2; exit 2
fi
if [ "$(basename "$MODEL")" != Qwen3-8B ]; then
  echo "$MODEL is not Qwen3-8B: FT-Q is registered for the Gate-FT backbone only (addendum 8 section 3)" >&2; exit 2
fi
STAGES=" $(printf '%s' "${STAGES:-all}" | tr ',' ' ') "
for s in $STAGES; do
  case "$s" in 1|2|3|4|all) ;; *) echo "unknown stage '$s' in STAGES (1-4, all)" >&2; exit 2 ;; esac
done
case "$STAGES" in *" all "*) STAGES="$STAGES 1 2 3 4 " ;; esac
want() { case "$STAGES" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

P="$GRID/panels/$D"                 # the grid's panels of D (read)
SPLIT="$P/ftgrid_split.json"
RS="$GRID/scores/$D"                # the real scores: zeroshot, s0, s1, s2 (read; linked into the FT-Q root)
QD="$QH/$D"                         # the nested slot's stage-1 files of D (read)
FTQ="$OUT_ROOT/ftq/$D"              # the teacher panel and its manifest
TQ="$FTQ/train_q.jsonl"
TMAN="$FTQ/train_q.manifest.json"
QA="$OUT_ROOT/adapters/$D"          # the teacher adapters p0, p1
QS="$OUT_ROOT/scores/$D"            # scores/D/<model>: links to the real zeroshot, s0-s2; the teacher like passes p0, p1
REP="$OUT_ROOT/report/$D.json"
REGREP="$GRID/report/$D.json"       # the registered report of D: never touched
B="$OUT_ROOT/build/$D"              # step markers
DONE="$B/done"
FTQ_FILES=(scripts/sigir/run_ftq.sh src/confrec/ftq_panel.py)    # addendum 8 section 4: recorded before the first training
if [ "$DRY_RUN" = 1 ]; then
  DRYD="$GRID/_dry"
  G="$DRYD/gatefix"; GT="$DRYD/gateft"; RAW="$DRYD/raw"; PILOT_LOG="$DRYD/PILOT_LOG.md"
  N_BOOT=20                         # the report's cost is its bootstrap: its structure does not depend on the count (the real one is 2,000)
  CORE_SPLITS=(--split "$SPLIT")
  LINK_FLAG="--allow_copy"
else
  G=outputs/confrec/gatefix; GT=outputs/confrec/gateft; RAW=data/raw; PILOT_LOG=docs/sigir/PILOT_LOG.md
  N_BOOT=2000                       # A3 section 3: 2,000 user resamples
  CORE_SPLITS=(--split "$REG/panels/ml1m/ftgrid_split.json" --split "$REG/panels/toys/ftgrid_split.json"
    --split "$REG/panels/games/ftgrid_split.json" --split "$REG/panels/sports/ftgrid_split.json")
  LINK_FLAG=""
fi
if [ "$D" = ml1m ]; then            # the real adapters of ML-1M are Gate-FT's (never retrained, scored by their path)
  RA="$GT/adapters"; TRAIN_REF="$GT/train.jsonl"; SPLIT_RECIPE_FLAG=""
else
  RA="$GRID/adapters/$D"; TRAIN_REF="$P/train.jsonl"; SPLIT_RECIPE_FLAG="--split_recipe"
fi

# ---- helpers ----
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
# (run_ftgrid.sh's function, verbatim: tests/test_confrec_ftq.py compares the two)
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
# score DATA DIR --lora A: pyes_scorer with SARGS, the scorer flags of the real s0 like pass (fp16, top-50 logprobs, max_model_len
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
check_record() { "$PY" -m src.confrec.ftq_panel record --pilot_log "${1:-$PILOT_LOG}" --files "${FTQ_FILES[@]}"; }
# ensure_record: the FT-Q record is in the pilot log (exit 4 otherwise). DRY_RUN does the human step on the temporary log first
# (--append writes the missing lines), unless DRY_NO_RECORD=1
ensure_record() {
  if [ "$DRY_RUN" = 1 ] && [ "$DRY_NO_RECORD" != 1 ]; then
    "$PY" -m src.confrec.ftq_panel record --pilot_log "$PILOT_LOG" --files "${FTQ_FILES[@]}" --append || exit 4
  else
    check_record || exit 4
  fi
}
# rehearse: DRY_RUN only, once per world: an empty pilot log fails the amendment, core and FT-Q record checks
rehearse() {
  if [ -f "$B/rehearsed" ]; then return 0; fi
  [ -f "$DRYD/PILOT_LOG.empty.md" ] || printf '# empty temporary pilot log\n' > "$DRYD/PILOT_LOG.empty.md"
  if check_amendment "$DRYD/PILOT_LOG.empty.md" > /dev/null 2>&1 || check_core "$DRYD/PILOT_LOG.empty.md" > /dev/null 2>&1 \
      || check_record "$DRYD/PILOT_LOG.empty.md" > /dev/null 2>&1; then
    echo "DRY_RUN: a freeze or record check passed on an empty pilot log" >&2; exit 1
  fi
  echo "[dry] gate rehearsal: an empty pilot log fails the amendment, core and FT-Q record checks"
  mkdir -p "$B"; touch "$B/rehearsed"
}
# gates: everything the registered text asks before a GPU job (Amendment 3 section 0, addendum 8 sections 3 and 4)
gates() {
  if [ "$GATE_DECISION" != GATE_FT_PASS ]; then
    echo "FT-Q refused: Gate-FT decision $GATE_DECISION (section 0: FT-Q runs only after a recorded GATE_FT_PASS, $GT/gate_ft.json)" >&2
    exit 4
  fi
  if [ "$DRY_RUN" = 1 ]; then rehearse; fi
  check_amendment || { echo "FT-Q refused: the freeze record of Amendment 3 (stage amendment) is not in $PILOT_LOG" >&2; exit 4; }
  check_core || { echo "FT-Q refused: the full record of Amendment 3 (stage core) is not in $PILOT_LOG" >&2; exit 4; }
  if [ "$SEL_DECISION" = FIX_FOUND ] && [ ! -f "$G/confirm/gate.json" ]; then
    echo "FT-Q refused: selection.json found a fix but the gate-fix confirm stage has recorded no gate.json (section 4)" >&2
    exit 4
  fi
  ensure_record
}
# read_info: SEL_DECISION, SEL_PROMPT (selection.json), SPLIT_VARIANT (the split) and GATE_DECISION (gate_ft.json; missing if absent)
read_info() {
  local out rc=0
  out=$("$PY" -m src.confrec.ftq_panel info --selection "$SEL" --split "$SPLIT" --gate "$GT/gate_ft.json") || rc=$?
  [ "$rc" = 0 ] || exit "$rc"
  { read -r SEL_DECISION; read -r SEL_PROMPT; read -r SPLIT_VARIANT; read -r GATE_DECISION; } < <(printf '%s\n' "$out" | tr -d '\r')
}
# recipe: RECIPE = the trainer flags the real adapters s0-s2 recorded in their train_config.json, except --train, --out and --seed
# (ftq_panel validates them: one registered recipe, this model, variant and TRAIN file; ML-1M: Gate-FT's adapters)
recipe() {
  local out rc=0
  out=$("$PY" -m src.confrec.ftq_panel recipe --adapters "$RA" --model "$MODEL" --variant "$VARIANT" --train_ref "$TRAIN_REF" \
    --train_file "$P/train.jsonl" --split "$SPLIT" $SPLIT_RECIPE_FLAG) || rc=$?
  [ "$rc" = 0 ] || exit "$rc"
  mapfile -t RECIPE < <(printf '%s\n' "$out" | tr -d '\r')
  echo "[recipe] trainer flags recorded by the real s0-s2 (--train, --out, --seed replaced): ${RECIPE[*]}"
}
# scoring_recipe: SARGS = the scorer flags of the real s0 like pass (its run.key and the config of its report.json)
scoring_recipe() {
  local out rc=0
  if [ -n "${SARGS_READ:-}" ]; then return 0; fi
  out=$("$PY" -m src.confrec.ftq_panel scoring --scores "$RS" --adapters "$RA" --data "$P/eval.jsonl" --model "$MODEL" \
    --variant "$VARIANT") || rc=$?
  [ "$rc" = 0 ] || exit "$rc"
  mapfile -t SARGS < <(printf '%s\n' "$out" | tr -d '\r')
  SARGS_READ=1
  echo "[recipe] scorer flags recorded by the real s0 like pass (--data, --output, --model, --lora replaced): ${SARGS[*]}"
}
# verify_adapters SEEDS [--write]: each p<seed>'s recorded arguments equal the real s0's except --train, --out, --seed; its provenance
verify_adapters() {
  local seeds=$1; shift
  "$PY" -m src.confrec.ftq_panel verify_adapter --adapters "$QA" --ref_adapters "$RA" --p_seeds "$seeds" --train "$TQ" \
    --teacher_manifest "$TMAN" "$@" || exit $?
}
# train_adapter TRAIN OUT SEED: the recorded recipe on the teacher panel; skipped when train_config.json and weights exist (their
# provenance was verified before the first training)
train_adapter() {
  local train=$1 out=$2 seed=$3
  if adapter_done "$out"; then echo "[skip] adapter $out exists"; return 0; fi
  [ -f "$train" ] || { echo "missing $train (stage 1)" >&2; exit 1; }
  rm -rf "$out"
  "$MPY" -m src.confrec.train_lora_yesno --train "$train" --out "$out" --seed "$seed" "${RECIPE[@]}" 2>&1 \
    | grep -vE "it/s\]|s/it\]" || true
  adapter_done "$out" || { echo "training of $out did not finish" >&2; exit 1; }
  verify_adapters "$seed" --write
}
# link_scores: the real zeroshot, s0, s1, s2 scores become symlinks of the FT-Q root (never copies; DRY_RUN without symlinks copies)
link_scores() {
  "$PY" -m src.confrec.ftq_panel link --real_scores "$RS" --q_scores "$QS" --models zeroshot,s0,s1,s2 $LINK_FLAG || exit $?
}
# registered_sha: the sha1 of the registered report of D and its tables ('-' when absent)
registered_sha() {
  local f
  for f in "$REGREP" "${REGREP%.json}_tables.csv"; do
    if [ -f "$f" ]; then sha1sum "$f" | cut -d' ' -f1; else echo -; fi
  done
}

# ---- DRY_RUN: run_ftgrid.sh's synthetic world of D (its stages 0-3), the nested slot's stage 1 and the stand-ins ----
if [ "$DRY_RUN" = 1 ]; then
  mkdir -p "$DRYD"
  FAKES="$DRYD/ftgrid_fakes.py"
  if [ "$DRY_NO_WORLD" = 1 ]; then
    echo "[skip] DRY_NO_WORLD=1: the synthetic world of $D is used as it is ($GRID, $QH)"
  elif [ -f "$SPLIT" ] && [ -f "$FAKES" ] && [ -f "$RS/zeroshot/like/report.json" ] && [ -f "$RS/s0/like/report.json" ] \
      && [ -f "$RS/s1/like/report.json" ] && [ -f "$RS/s2/like/report.json" ] && [ -f "$QD/qhat_manifest.json" ] \
      && [ -f "$QD/train_qhat.csv.gz" ]; then
    echo "[skip] run_ftgrid.sh's DRY_RUN world for $D and the nested slot's stage-1 files exist ($GRID, $QH)"
  else
    echo "== DRY_RUN: run_ftgrid.sh's synthetic world for $D (its stages 0-3: panels, the real adapters, their like passes)"
    if ! DRY_RUN=1 OUT_ROOT="$GRID" MODEL="$MODEL" STAGES=0,1,2,3 PYTHON="$PY" bash scripts/sigir/run_ftgrid.sh "$D" \
        > "$DRYD/ftgrid.$D.log" 2>&1; then
      tail -n 40 "$DRYD/ftgrid.$D.log" >&2
      echo "DRY_RUN: run_ftgrid.sh $D failed (log: $DRYD/ftgrid.$D.log)" >&2; exit 1
    fi
    echo "== DRY_RUN: the nested slot's stage 1 (q-hat of every TRAIN example) for $D"
    mkdir -p "$QD"
    if ! "$PY" -m src.confrec.train_lora_offset qhat --domain "$D" --panels "$P" --raw "$RAW" --out_dir "$QD" \
        > "$DRYD/qhat.$D.log" 2>&1; then
      tail -n 40 "$DRYD/qhat.$D.log" >&2
      echo "DRY_RUN: train_lora_offset qhat $D failed (log: $DRYD/qhat.$D.log)" >&2; exit 1
    fi
  fi
fi

# ---- the prompt and the recorded decisions ----
SEL="$G/dev/selection.json"
[ -f "$SEL" ] || { echo "missing $SEL (the gate-fix stage 1 selection; run_gatefix.sh writes it)" >&2; exit 1; }
read_info
VARIANT="${VARIANT:-$SEL_PROMPT}"
if [ "$VARIANT" != "$SEL_PROMPT" ] && [ "$DRY_RUN" != 1 ]; then
  echo "VARIANT=$VARIANT is not selection.json's gate_ft_prompt $SEL_PROMPT: section 2 registers no other variant" >&2
  exit 2
fi
if [ -n "$SPLIT_VARIANT" ] && [ "$SPLIT_VARIANT" != "$VARIANT" ]; then
  echo "$SPLIT was built under variant $SPLIT_VARIANT, not $VARIANT" >&2; exit 2
fi
echo "run_ftq $D: model $MODEL, root $OUT_ROOT, variant $VARIANT, stages:$STAGES"

# ---- the registered gates come first: a refused run touches nothing ----
GATED=0
for s in 2 3 4; do if want "$s"; then GATED=1; fi; done
if [ "$GATED" = 1 ]; then gates; fi

# ================= stage 1: the teacher panel (CPU) =================
if want 1; then
  echo "== stage 1: the teacher panel ($D)"
  for f in "$P/train.jsonl" "$SPLIT"; do
    [ -f "$f" ] || { echo "missing $f (run_ftgrid.sh stage 0)" >&2; exit 1; }
  done
  for f in "$QD/train_qhat.csv.gz" "$QD/qhat_manifest.json"; do
    [ -f "$f" ] || { echo "missing $f (the nested slot's stage 1: run_ftmethod.sh $D with STAGES=1, CPU)" >&2; exit 1; }
  done
  mkdir -p "$FTQ"
  step "$TMAN" "$P/train.jsonl" "$QD/train_qhat.csv.gz" "$QD/qhat_manifest.json" "$SPLIT" src/confrec/ftq_panel.py \
      src/confrec/ftprune.py src/confrec/train_lora_offset.py src/confrec/ftgrid_data.py -- \
    "$PY" -m src.confrec.ftq_panel build --domain "$D" --train "$P/train.jsonl" --qhat "$QD/train_qhat.csv.gz" \
      --qmanifest "$QD/qhat_manifest.json" --split "$SPLIT" --out_dir "$FTQ"
fi

# ================= stage 2: the teacher adapters p0, p1 (GPU) =================
if want 2; then
  echo "== stage 2: adapters p0, p1 ($D)"
  { [ -f "$TQ" ] && [ -f "$TMAN" ]; } || { echo "missing $TQ or $TMAN (stage 1)" >&2; exit 1; }
  recipe
  scoring_recipe       # pre-flight: the real s0 like pass must be the registered one before any GPU hour is spent
  have=""
  for seed in 0 1; do if adapter_done "$QA/p$seed"; then have="${have:+$have,}$seed"; fi; done
  if [ -n "$have" ]; then verify_adapters "$have"; fi
  for seed in 0 1; do train_adapter "$TQ" "$QA/p$seed" "$seed"; done
fi

# ================= stage 3: like on eval.jsonl for p0, p1 (GPU) =================
if want 3; then
  echo "== stage 3: like on eval.jsonl for p0, p1 ($D)"
  for f in "$P/eval.jsonl" "$TMAN"; do
    [ -f "$f" ] || { echo "missing $f (run_ftgrid.sh stage 0 / this script's stage 1)" >&2; exit 1; }
  done
  scoring_recipe
  for seed in 0 1; do
    adapter_done "$QA/p$seed" || { echo "adapter p$seed of $D is missing or incomplete: $QA/p$seed (stage 2)" >&2; exit 1; }
  done
  verify_adapters 0,1
  for seed in 0 1; do
    score "$P/eval.jsonl" "$QS/p$seed/like" --lora "$QA/p$seed"
    "$PY" -m src.confrec.ftq_panel verify_scores --scores "$QS" --ref_scores "$RS" --adapters "$QA" --p_seeds "$seed" || exit $?
  done
  link_scores
fi

# ================= stage 4: ftgrid_report with the six models (CPU) =================
if want 4; then
  echo "== stage 4: ftgrid_report with zeroshot, s0-s2, p0, p1 -> $REP ($D)"
  for m in zeroshot s0 s1 s2; do
    [ -f "$RS/$m/like/report.json" ] || { echo "missing the real like pass $RS/$m/like (run_ftgrid.sh $D)" >&2; exit 1; }
  done
  for m in p0 p1; do
    { [ -f "$QS/$m/like/report.json" ] || [ -f "$QS/$m/like/FAILED_INTEGRITY" ]; } \
      || { echo "missing the like pass $QS/$m/like (stage 3)" >&2; exit 1; }
  done
  link_scores
  RDEPS=("$SPLIT" src/confrec/ftgrid_report.py)
  for f in "$QS"/*/*/report.json; do if [ -f "$f" ]; then RDEPS+=("$f"); fi; done
  REG_BEFORE=$(registered_sha)
  mkdir -p "$(dirname "$REP")"
  step "$REP" "${RDEPS[@]}" -- \
    "$PY" -m src.confrec.ftgrid_report --domain "$D" --split "$SPLIT" --panels "$P" --scores_root "$OUT_ROOT/scores" \
      --models zeroshot,s0,s1,s2,p0,p1 --raw "$RAW" --out "$REP" --n_boot "$N_BOOT" --seed 0
  if [ "$(registered_sha)" != "$REG_BEFORE" ]; then
    echo "the registered report $REGREP changed: it is never touched (addendum 8 section 2)" >&2; exit 1
  fi
fi
for m in p0 p1; do
  if [ -f "$QS/$m/like/FAILED_INTEGRITY" ]; then
    echo "NOTE: $QS/$m/like is FAILED_INTEGRITY: $m is reported as missing, never replaced (section 2)" >&2
  fi
done
echo "run_ftq $D: done (stages:$STAGES)"
