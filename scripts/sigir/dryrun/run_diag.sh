#!/usr/bin/env bash
# The REAL scripts/sigir/run_diag_battery.sh in a dry-run sandbox (setup_sandbox.py) with the fake scorer; the sandbox
# Pilot-1 panels are not the registered ones, hence ALLOW_PANEL_MISMATCH=1 (override with ALLOW_PANEL_MISMATCH=0).
#   usage: bash scripts/sigir/dryrun/run_diag.sh SBX SCENARIO.json TAG [ENV=VAL ...]     e.g.  T1_OVERRIDE=1  SKIP_T4=1
D="$(cd "$(dirname "$0")" && pwd)"
R="${SBX_ROOT:-${TMPDIR:-/tmp}/lumen_dryrun}"
sbx="$1"; scen="$2"; tag="$3"; shift 3
cd "$R/$sbx" || exit 1
export PYTHON="$D/fakepy" NBOOT="${NBOOT:-200}" FAKE_SCENARIO="$D/$scen" MODEL="$R/$sbx/models/Qwen3-8B" \
       LLAMA="$R/$sbx/models/Llama-3.1-8B-Instruct" ALLOW_PANEL_MISMATCH=1
export SIMLOG="$R/calls_$tag.log"
for kv in "$@"; do export "$kv"; done
t0=$(date +%s)
bash scripts/sigir/run_diag_battery.sh
rc=$?
echo "run_diag_battery.sh exit $rc in $(( $(date +%s) - t0 )) s"
exit $rc
