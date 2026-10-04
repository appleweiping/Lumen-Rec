#!/usr/bin/env bash
# The REAL scripts/sigir/run_gatefix_stage3.sh in a dry-run sandbox with the fake scorer (PYTHON=fakepy_s3: fakepy plus
# the sandbox-scale G7 constants of fake_stage3.py). Sandbox recipe: scripts/sigir/dryrun/s3_sandbox.py's docstring.
#   usage: bash scripts/sigir/dryrun/run_s3.sh SBX SCENARIO.json TAG [ENV=VAL ...]     e.g.  S3_MIN_FRESH=1000000000
# SBX_ROOT (default <tmp>/lumen_dryrun) holds the sandboxes; the call log is $SBX_ROOT/calls_TAG.log. MODEL is the one
# run_gf.sh scored stages 1-2 with (the input check requires G6's backbone).
D="$(cd "$(dirname "$0")" && pwd)"
R="${SBX_ROOT:-${TMPDIR:-/tmp}/lumen_dryrun}"
sbx="$1"; scen="$2"; tag="$3"; shift 3
chmod +x "$D/fakepy" "$D/fakepy_s3" 2>/dev/null || true
cd "$R/$sbx" || exit 1
export PYTHON="$D/fakepy_s3" NBOOT="${NBOOT:-200}" FAKE_SCENARIO="$D/$scen" MODEL="$R/$sbx/models/Qwen3-8B"
export SIMLOG="$R/calls_$tag.log"
for kv in "$@"; do export "$kv"; done
t0=$(date +%s)
bash scripts/sigir/run_gatefix_stage3.sh
rc=$?
echo "run_gatefix_stage3.sh exit $rc in $(( $(date +%s) - t0 )) s"
exit $rc
