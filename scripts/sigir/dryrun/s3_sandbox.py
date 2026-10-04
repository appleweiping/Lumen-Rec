"""Stage-3 additions to a dry-run sandbox of setup_sandbox.py (scripts/sigir/run_gatefix_stage3.sh, amendment 2 G7).

    python scripts/sigir/dryrun/setup_sandbox.py --fresh --sbx s3 --amazon_users 2200   # Toys >= 2,000 eligible (G2)
    python scripts/sigir/dryrun/s3_sandbox.py prep --sbx s3
    bash scripts/sigir/dryrun/run_gf.sh s3 scen_fix.json gf                  # run_gatefix.sh stage 0 (freeze stop)
    python scripts/sigir/dryrun/s3_sandbox.py record_freeze --sbx s3
    bash scripts/sigir/dryrun/run_gf.sh s3 scen_fix.json gf FREEZE_ACK=1     # stages 1-2: FIX_FOUND (V3) -> GATE_PASS
    bash scripts/sigir/dryrun/run_s3.sh s3 scen_fix.json s3                  # stage 3 on fresh Toys
    bash scripts/sigir/dryrun/run_s3.sh s3 scen_fix.json s3g S3_MIN_FRESH=1000000000   # the Video_Games fallback

prep   writes a synthetic sports VALID ranking file where run_gatefix_stage3.sh reads it (next-item rows as synth.py
       writes the TEST panel: the same users, an earlier event each, split_name "valid", 101 candidates) and puts the
       Pilot-1 decision record at outputs/confrec/pilot1_mirror/decision.json: the repo's local copy of the registered
       record (outputs/confrec_pilot/pilot1/decision.json, sports TEST 1-1000 NDCG@10 0.2093 >= 0.18632) when present,
       else a record with the same gate fields. run_pilot1_mirror.sh is not needed for stage 3.
record_freeze  appends the REQUIRED sha1s of the sandbox's outputs/confrec/gatefix/FREEZE.txt to its
       docs/sigir/PILOT_LOG.md (the manual step between run_gatefix.sh stage 0 and FREEZE_ACK=1).
The sandbox root is $SBX_ROOT (default <tmp>/lumen_dryrun), as for setup_sandbox.py and run_*.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path(os.environ.get("SBX_ROOT") or Path(tempfile.gettempdir()) / "lumen_dryrun")
VALID = "outputs/baselines/external_tasks/sports_large10000_100neg_valid_same_candidate/ranking_valid.jsonl"
P1 = "outputs/confrec/pilot1_mirror/decision.json"
REGISTERED_P1 = REPO / "outputs" / "confrec_pilot" / "pilot1" / "decision.json"


def valid_rows(n_events: int = 40, n_cands: int = 101, seed: int = 11) -> list[dict]:
    """synth.next_item's row format; user e is the TEST panel's user e, at an earlier timestamp (a different event)."""
    rng = random.Random(seed)
    rows = []
    for e in range(n_events):
        uid = f"AE{e:04d}SPORTSUSER{e * 7 % 13:02d}"
        ts = 1566559236892 + e * 88883
        ids = [f"B0V{e:03d}{c:04d}" for c in range(n_cands)]
        pos = rng.randrange(n_cands)
        groups = [rng.choice(["head", "mid", "mid", "tail", "tail"]) for _ in ids]
        hist_n = rng.randint(3, 20)
        rows.append({
            "source_event_id": f"{uid}::{ts}", "user_id": uid,
            "history": [f"Sports gear {e}-{h} (Outdoor)" for h in range(hist_n)],
            "history_item_ids": [f"B0H{e:03d}{h:03d}" for h in range(hist_n)],
            "candidate_item_ids": ids,
            "candidate_titles": [f"Sports item {i}" for i in ids],
            "candidate_texts": [f"Category: Sports. Popularity group {g}." for g in groups],
            "candidate_popularity_groups": groups,
            "candidate_labels": [int(c == pos) for c in range(n_cands)],
            "positive_item_id": ids[pos], "positive_item_title": f"Sports item {ids[pos]}",
            "positive_item_text": "", "positive_item_index": pos, "timestamp": str(ts), "split_name": "valid",
            "num_candidates": n_cands, "source_pointwise_size": n_cands})
    return rows


def pilot1_record() -> dict:
    if REGISTERED_P1.is_file():
        return json.loads(REGISTERED_P1.read_text(encoding="utf-8"))
    return {"decision": "GATE_FAIL_UNINTERPRETABLE", "note": "sandbox stand-in with the registered gate fields",
            "gate": {"pass": False, "input_checks_ok": True, "ml1m_raw_UAUC": 0.5874475521977115,
                     "ml1m_raw_UAUC_min": 0.6, "sports_raw_NDCG@10": 0.20934006726080595,
                     "sports_raw_NDCG@10_min": 0.18632,
                     "input_checks": {"sports": {"panel_type": "next_item", "base_question": "next",
                                                 "expected": ["next_item", "next"], "ok": True}}}}


def prep(sb: Path, n_events: int) -> None:
    v = sb / VALID
    v.parent.mkdir(parents=True, exist_ok=True)
    v.write_text("".join(json.dumps(r) + "\n" for r in valid_rows(n_events)), encoding="utf-8", newline="\n")
    p = sb / P1
    p.parent.mkdir(parents=True, exist_ok=True)
    if REGISTERED_P1.is_file():
        shutil.copy2(REGISTERED_P1, p)
    else:
        p.write_text(json.dumps(pilot1_record(), indent=2), encoding="utf-8")
    print(f"s3_sandbox prep: {v} ({n_events} VALID events), {p}")


def record_freeze(sb: Path) -> list[str]:
    text = (sb / "outputs/confrec/gatefix/FREEZE.txt").read_text(encoding="utf-8")
    req = re.findall(r"= ([0-9a-f]{40})  REQUIRED", text)
    log = sb / "docs/sigir/PILOT_LOG.md"
    log.write_text(log.read_text(encoding="utf-8") + "\n## sandbox: amendment-2 freeze record\n"
                   + "".join(f"- {s}\n" for s in req), encoding="utf-8", newline="\n")
    print(f"s3_sandbox record_freeze: {len(req)} REQUIRED sha1s appended to {log}")
    return req


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "record_freeze"])
    ap.add_argument("--sbx", default="sbx", help="sandbox name under $SBX_ROOT")
    ap.add_argument("--n_valid", type=int, default=40, help="synthetic sports VALID events")
    a = ap.parse_args(argv)
    sb = ROOT / a.sbx
    if a.cmd == "prep":
        prep(sb, a.n_valid)
    else:
        record_freeze(sb)


if __name__ == "__main__":
    main()
