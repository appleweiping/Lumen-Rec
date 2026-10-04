"""CPU dry-run sandbox of the amendment-2 run scripts (scripts/sigir/run_gatefix.sh, run_diag_battery.sh,
run_pilot1_mirror.sh) with synthetic data and a FAKE GPU scorer. Nothing here touches the repo's data/ or outputs/:
the sandbox is a copy of the code under $SBX_ROOT (default <tmp>/lumen_dryrun/<name>).

    python scripts/sigir/dryrun/setup_sandbox.py --fresh [--sbx sbx]       # copy code, synthesize data, Pilot-1 panels
    python scripts/sigir/dryrun/setup_sandbox.py --code_only [--sbx sbx]   # re-copy code after a repo change
    bash scripts/sigir/dryrun/run_p1.sh                                    # run_pilot1_mirror.sh (fake scorer)
    bash scripts/sigir/dryrun/run_gf.sh sbx scen_fix.json gfA              # run_gatefix.sh stage 0 (stops at the freeze)
    # record the REQUIRED sha1s of <sandbox>/outputs/confrec/gatefix/FREEZE.txt in <sandbox>/docs/sigir/PILOT_LOG.md,
    bash scripts/sigir/dryrun/run_gf.sh sbx scen_fix.json gfA FREEZE_ACK=1  # then stages 1-2: FIX_FOUND -> GATE_PASS
    bash scripts/sigir/dryrun/run_diag.sh sbx scen_fix.json diagA           # run_diag_battery.sh end to end
Scenario files (fake_scorer.py documents the schema): scen_fix.json (V3 has a genuine advantage on dev and confirm),
scen_base.json (none: F0), scen_f1.json (advantage on dev only: F1), scen_e1.json (V3 fails E1 on Toys),
scen_t1broken.json (the T1 readout ignores the stated rating: exit 5). The registered identity constants (Pilot-1 panel
sha1s, 1500-line prefix, 1,683 fresh users, gatefix_select.PILOT1_DEV) are re-derived for the sandbox panels by
fake_bcp.py / fake_gatefix.py; every other module is the real one. A sandbox keeps the amendment copy its freeze
recorded (a later edit of the repo's amendment is not copied over it).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path(os.environ.get("SBX_ROOT") or Path(tempfile.gettempdir()) / "lumen_dryrun")


def copy_code(sb: Path) -> None:
    (sb / "src" / "confrec").mkdir(parents=True, exist_ok=True)
    (sb / "scripts" / "sigir").mkdir(parents=True, exist_ok=True)
    (sb / "idea-stage").mkdir(exist_ok=True)
    (sb / "docs" / "sigir").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "src" / "__init__.py", sb / "src" / "__init__.py")
    for f in (REPO / "src" / "confrec").glob("*.py"):
        shutil.copy2(f, sb / "src" / "confrec" / f.name)
    for f in (REPO / "scripts" / "sigir").iterdir():
        if f.is_file():
            shutil.copy2(f, sb / "scripts" / "sigir" / f.name)
    for n in ("PREREG_AMENDMENT_1.md", "PREREG_AMENDMENT_2.md"):   # the sandbox keeps the copy its freeze recorded
        if not (sb / "idea-stage" / n).exists():
            shutil.copy2(REPO / "idea-stage" / n, sb / "idea-stage" / n)
    pl = sb / "docs" / "sigir" / "PILOT_LOG.md"
    if not pl.exists():
        pl.write_text("# PILOT_LOG (sandbox copy)\n\nPilot 1: GATE_FAIL (sandbox).\n", encoding="utf-8")


def sh(cmd, sb: Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(sb), PYTHONHASHSEED="0")
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=sb, env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--code_only", action="store_true")
    ap.add_argument("--sbx", default="sbx")
    ap.add_argument("--ml_users", type=int, default=160)
    ap.add_argument("--amazon_users", type=int, default=420)
    ap.add_argument("--pilot_users", type=int, default=100)
    a = ap.parse_args()
    sb = ROOT / a.sbx
    if a.fresh:
        shutil.rmtree(sb, ignore_errors=True)
    copy_code(sb)
    if a.code_only:
        return
    py = sys.executable
    sh([py, str(HERE / "synth.py"), "--root", str(sb), "--amazon_users", str(a.amazon_users), "--ml_users",
        str(a.ml_users)], sb)
    r = sb / "outputs/baselines/external_tasks/sports_large10000_100neg_test_same_candidate"
    r.mkdir(parents=True, exist_ok=True)
    shutil.move(str(sb / "panels" / "sports_next_1k.jsonl"), str(r / "ranking_test.jsonl"))
    d = sb / "docs/sigir/ref_ranks/sports"
    d.mkdir(parents=True, exist_ok=True)
    shutil.move(str(sb / "ref_ranks" / "sports" / "ccrp_v3.csv.gz"), str(d / "ccrp_v3.csv.gz"))
    P = sb / "outputs/confrec/panels"
    P.mkdir(parents=True, exist_ok=True)
    n = str(a.pilot_users)
    sh([py, "-m", "src.confrec.build_rated_panels", "--source", "ml1m", "--raw", "data/raw/ml-1m", "--out",
        "outputs/confrec/panels/ml1m_rated.jsonl", "--n_users", n], sb)
    sh([py, "-m", "src.confrec.build_rated_panels", "--source", "amazon", "--domain", "toys", "--raw", "data/raw",
        "--out", "outputs/confrec/panels/toys_rated.jsonl", "--n_users", n], sb)
    time.sleep(1.1)   # the .done stamps must be newer than every input and the builder code
    for p in ("ml1m_rated", "toys_rated"):
        (P / f"{p}.jsonl.done").touch()
    m = sb / "models" / "Llama-3.1-8B-Instruct"
    m.mkdir(parents=True, exist_ok=True)
    (m / "config.json").write_text("{}", encoding="utf-8")
    (sb / "models" / "Qwen3-8B").mkdir(parents=True, exist_ok=True)
    print("sandbox ready:", sb)


if __name__ == "__main__":
    main()
