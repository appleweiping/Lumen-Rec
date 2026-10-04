#!/usr/bin/env bash
# The REAL scripts/sigir/run_gatefix.sh in a dry-run sandbox (setup_sandbox.py) with the fake scorer.
#   usage: bash scripts/sigir/dryrun/run_gf.sh SBX SCENARIO.json TAG [ENV=VAL ...]     e.g.  FREEZE_ACK=1  MODEL=...
# SBX_ROOT (default <tmp>/lumen_dryrun) holds the sandboxes; the call log is $SBX_ROOT/calls_TAG.log.
D="$(cd "$(dirname "$0")" && pwd)"
R="${SBX_ROOT:-${TMPDIR:-/tmp}/lumen_dryrun}"
sbx="$1"; scen="$2"; tag="$3"; shift 3
cd "$R/$sbx" || exit 1
export PYTHON="$D/fakepy" NBOOT="${NBOOT:-200}" FAKE_SCENARIO="$D/$scen" MODEL="$R/$sbx/models/Qwen3-8B"
export SIMLOG="$R/calls_$tag.log"
for kv in "$@"; do export "$kv"; done
t0=$(date +%s)
bash scripts/sigir/run_gatefix.sh
rc=$?
echo "run_gatefix.sh exit $rc in $(( $(date +%s) - t0 )) s"
exit $rc
