#!/usr/bin/env bash
# Cheap-signal controls for selective serving (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_11.md): an EXPLORATORY, DESCRIPTIVE, CPU-only
# analysis (src/confrec/nextitem_serving_control.py). For every listed domain it runs `nextitem_serving_control run` on the panels and
# scores of the registered audit and, as the second-backbone replication, on the Z2 audits whenever their score files exist:
#   qwen_registered  Qwen3-8B, the registered TEST segment (sports events 1,001-10,000; toys, home, tools all 10,000 events),
#                    scores $AUDIT_QWEN/<d>_test and <d>_valid2k, the paths of the audit job (audit_analysis_job.sh)
#   qwen_z2          the Qwen audit restricted to TEST events 1,001-3,000 (`nextitem_audit restrict`), $AUDIT_QWEN_Z2/<d>_test and
#                    <d>_valid2k, one segment, --segments single --first_event 1001
#   llama_z2         Llama-3.1-8B-Instruct on TEST events 1,001-3,000, $AUDIT_LLAMA/<d>_test1001_3000 and <d>_valid500, one segment,
#                    --segments single --test_role test1001_3000 --valid_role valid500 --first_event 1001 (run_llama_nextitem.sh)
# then `summarize` per panel kind (counts of the per-panel labels; backbones separately).
#   usage (server, repo root of the checkout; CPU only):  bash scripts/sigir/run_servingctrl.sh
#   env:   DOMAINS   words of sports toys home tools (default all four)
#          PANELS    words of registered z2_qwen z2_llama (default all three; a Z2 panel whose TEST score file does not exist is skipped,
#                    a registered panel without one is an error)
#          AUDIT_QWEN, AUDIT_QWEN_Z2, AUDIT_LLAMA   the score roots (defaults /root/autodl-tmp/lumen-audit-out, ...-qwen-restricted,
#                    ...-llama); the audit job's panel paths outputs/baselines/external_tasks/<d>_large10000_100neg_{test,valid}_same_candidate
#          AUDIT_JSON_QWEN_Z2, AUDIT_JSON_LLAMA_Z2   optional directories of the audit's Z2 result files <d>.json: when set, `run` requires
#                    that the p_max and random numbers of the audit are reproduced (the registered panels use outputs/confrec/
#                    nextitem_audit/<d>.json when that file exists)
#          PYTHON    interpreter (skips the conda activation; default: conda env lumen)
#          DRY_RUN   0 or 1 (anything else is refused): 1 = a rehearsal on a synthetic fixture (tests/test_confrec_serving_control.py
#                    write_dry_world): no pilot log, no freeze check, no real path; DRY_N_BOOT (default 40) resamples
# The registered resamples (2,000) and seed (0) are not settings: N_BOOT and SEED in the environment are refused.
# Refuses (exit 4, nothing started) unless `python -m src.confrec.ftgrid_freeze --check --stage amendment` passes and the sha1 of
# src/confrec/nextitem_serving_control.py, tests/test_confrec_serving_control.py and this script are in docs/sigir/PILOT_LOG.md
# (`python -m src.confrec.nextitem_serving_control record` prints the lines `path = sha1`).
# Writes ONLY below outputs/confrec/nextitem_audit_ctrl (<kind>/<d>.json, <kind>/summary.json; a rehearsal below _dry/): never the
# registered audit directory outputs/confrec/nextitem_audit, which is read for --audit_json only. No network, no GPU, no byte code.
# Exit codes: 0 done; 1 a run failed or a registered score file is missing; 2 usage; 4 refused by the freeze or record check.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

DRY_RUN="${DRY_RUN-0}"
case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1 (it is '$DRY_RUN'): nothing was started" >&2; exit 2 ;; esac
for sw in N_BOOT SEED; do
  if [ -n "${!sw+x}" ]; then
    echo "$sw is registered (2,000 resamples, seed 0) and is not a setting of this script: unset it (nothing was started)" >&2; exit 2
  fi
done
if [ "$DRY_RUN" != 1 ] && [ -n "${DRY_N_BOOT+x}" ]; then
  echo "DRY_N_BOOT belongs to a DRY_RUN=1 rehearsal: nothing was started" >&2; exit 2
fi
DOMAINS="${DOMAINS:-sports toys home tools}"
PANELS="${PANELS:-registered z2_qwen z2_llama}"
for d in $DOMAINS; do
  case "$d" in sports|toys|home|tools) ;; *) echo "unknown domain '$d' in DOMAINS (sports toys home tools)" >&2; exit 2 ;; esac
done
for p in $PANELS; do
  case "$p" in registered|z2_qwen|z2_llama) ;; *) echo "unknown panel '$p' in PANELS (registered z2_qwen z2_llama)" >&2; exit 2 ;; esac
done
want() { case " $PANELS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  set +u; source /root/miniconda3/etc/profile.d/conda.sh; conda activate lumen; set -u
  PY=python
fi

OUT=outputs/confrec/nextitem_audit_ctrl            # the only directory this script writes
EXT=outputs/baselines/external_tasks
AUDIT_RESULTS=outputs/confrec/nextitem_audit       # the registered audit's result files: read (--audit_json), never written
if [ "$DRY_RUN" = 1 ]; then
  W="$OUT/_dry/world"
  OUTK="$OUT/_dry/out"
  NB="${DRY_N_BOOT:-40}"
  DOMAINS="toys"
  AUDIT_QWEN="$W/audit"; AUDIT_QWEN_Z2="$W/audit_qwen_z2"; AUDIT_LLAMA="$W/audit_llama"
  AUDIT_RESULTS="$W/audit_results"
  AUDIT_JSON_QWEN_Z2="$W/audit_results_z2_qwen"; AUDIT_JSON_LLAMA_Z2="$W/audit_results_z2_llama"
  test_panel() { echo "$W/panels/${1}_test.jsonl"; }
  valid_panel() { echo "$W/panels/${1}_valid.jsonl"; }
  z2_panel() { echo "$W/panels/${1}_test1001_3000.jsonl"; }
else
  OUTK="$OUT"
  NB=2000
  AUDIT_QWEN="${AUDIT_QWEN:-/root/autodl-tmp/lumen-audit-out}"
  AUDIT_QWEN_Z2="${AUDIT_QWEN_Z2:-/root/autodl-tmp/lumen-audit-out-qwen-restricted}"
  AUDIT_LLAMA="${AUDIT_LLAMA:-/root/autodl-tmp/lumen-audit-out-llama}"
  AUDIT_JSON_QWEN_Z2="${AUDIT_JSON_QWEN_Z2:-}"; AUDIT_JSON_LLAMA_Z2="${AUDIT_JSON_LLAMA_Z2:-}"
  test_panel() { echo "$EXT/${1}_large10000_100neg_test_same_candidate/ranking_test.jsonl"; }
  valid_panel() { echo "$EXT/${1}_large10000_100neg_valid_same_candidate/ranking_valid.jsonl"; }
  z2_panel() { echo "$AUDIT_LLAMA/${1}_test1001_3000.jsonl"; }       # the 2,000-event panel run_llama_nextitem.sh wrote
fi
SEED=0

# ---- the gates come first: a refused run touches nothing ----
if [ "$DRY_RUN" != 1 ]; then
  "$PY" -m src.confrec.ftgrid_freeze --check --stage amendment || {
    echo "refused: the Amendment 3 freeze record is not complete (python -m src.confrec.ftgrid_freeze --check --stage amendment)" >&2
    exit 4
  }
  "$PY" -m src.confrec.nextitem_serving_control record --pilot_log docs/sigir/PILOT_LOG.md || {
    echo "refused: the sha1 of src/confrec/nextitem_serving_control.py, tests/test_confrec_serving_control.py and" \
      "scripts/sigir/run_servingctrl.sh must be in docs/sigir/PILOT_LOG.md first (print the lines with:" \
      "python -m src.confrec.nextitem_serving_control record)" >&2
    exit 4
  }
else
  mkdir -p "$OUT/_dry"
  "$PY" -c "from tests.test_confrec_serving_control import write_dry_world; write_dry_world('$W', $NB)"
fi

# ---- one panel ----
# run_panel KIND DOMAIN AUDIT_DIR PANEL_TEST PANEL_VALID AUDIT_JSON_OR_EMPTY ARGS...
run_panel() {
  local kind=$1 d=$2 audit=$3 pt=$4 pv=$5 aj=$6
  shift 6
  local -a extra=("$@")
  if [ -n "$aj" ] && [ -f "$aj" ]; then extra+=(--audit_json "$aj"); fi
  echo "== $kind / $d -> $OUTK/$kind/$d.json"
  nice -n 19 "$PY" -m src.confrec.nextitem_serving_control run --domain "$d" --audit_dir "$audit" --panel_test "$pt" \
    --panel_valid "$pv" --panel_kind "$kind" --out "$OUTK/$kind/$d.json" --n_boot "$NB" --seed "$SEED" "${extra[@]}"
}
summarize_kind() {
  local kind=$1 d list=""
  for d in $DOMAINS; do
    if [ -f "$OUTK/$kind/$d.json" ]; then list="${list:+$list,}$OUTK/$kind/$d.json"; fi
  done
  if [ -n "$list" ]; then
    nice -n 19 "$PY" -m src.confrec.nextitem_serving_control summarize --inputs "$list" --out "$OUTK/$kind/summary.json"
  fi
}

if want registered; then
  for d in $DOMAINS; do
    if [ ! -f "$AUDIT_QWEN/${d}_test/scores.csv.gz" ]; then
      echo "registered panel $d: no TEST scores at $AUDIT_QWEN/${d}_test/scores.csv.gz (the audit has not scored it)" >&2; exit 1
    fi
    run_panel qwen_registered "$d" "$AUDIT_QWEN" "$(test_panel "$d")" "$(valid_panel "$d")" "$AUDIT_RESULTS/$d.json" --questions next
  done
  summarize_kind qwen_registered
fi
if want z2_qwen; then
  for d in $DOMAINS; do
    if [ ! -f "$AUDIT_QWEN_Z2/${d}_test/scores.csv.gz" ]; then
      echo "skip z2_qwen / $d: no restricted Qwen TEST scores at $AUDIT_QWEN_Z2/${d}_test/scores.csv.gz"; continue
    fi
    run_panel qwen_z2 "$d" "$AUDIT_QWEN_Z2" "$(z2_panel "$d")" "$(valid_panel "$d")" "${AUDIT_JSON_QWEN_Z2:+$AUDIT_JSON_QWEN_Z2/$d.json}" \
      --questions next --segments single --first_event 1001
  done
  summarize_kind qwen_z2
fi
if want z2_llama; then
  for d in $DOMAINS; do
    if [ ! -f "$AUDIT_LLAMA/${d}_test1001_3000/scores.csv.gz" ]; then
      echo "skip z2_llama / $d: no Llama TEST scores at $AUDIT_LLAMA/${d}_test1001_3000/scores.csv.gz"; continue
    fi
    run_panel llama_z2 "$d" "$AUDIT_LLAMA" "$(z2_panel "$d")" "$(valid_panel "$d")" "${AUDIT_JSON_LLAMA_Z2:+$AUDIT_JSON_LLAMA_Z2/$d.json}" \
      --questions next --segments single --test_role test1001_3000 --valid_role valid500 --first_event 1001
  done
  summarize_kind llama_z2
fi
echo "SERVINGCTRL_DONE: panels=$PANELS domains=$DOMAINS dry_run=$DRY_RUN"
