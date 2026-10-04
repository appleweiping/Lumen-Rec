#!/usr/bin/env bash
# The REAL scripts/sigir/run_pilot1_mirror.sh in the dry-run sandbox `sbx` with the fake scorer (writes the Pilot-1
# outputs forensics.py and run_gatefix.sh's reproduction check read). usage: bash scripts/sigir/dryrun/run_p1.sh [SBX]
D="$(cd "$(dirname "$0")" && pwd)"
R="${SBX_ROOT:-${TMPDIR:-/tmp}/lumen_dryrun}"
sbx="${1:-sbx}"
cd "$R/$sbx" || exit 1
export PYTHON="$D/fakepy" NBOOT="${NBOOT:-200}" FAKE_SCENARIO="$D/scen_base.json" MODEL="$R/$sbx/models/Qwen3-8B"
export SIMLOG="$R/calls_p1.log"
: > "$SIMLOG"
t0=$(date +%s)
bash scripts/sigir/run_pilot1_mirror.sh
rc=$?
echo "run_pilot1_mirror.sh exit $rc in $(( $(date +%s) - t0 )) s"
exit $rc
