"""Freeze / verify the Lumen 1+100 same-candidate panels by fingerprint.

make   (local):  python scripts/sigir/panel_reference.py make
       reads outputs/<d>_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/tables/ranking_eval_records.csv
       writes docs/sigir/panel_refs/<d>_test.json  {source_event_id: [positive_item_id, sha1(ordered candidates)]}
verify (server): python scripts/sigir/panel_reference.py verify --domain sports \
                     --ranking outputs/baselines/external_tasks/sports_large10000_100neg_test_same_candidate/ranking_test.jsonl
       exits non-zero unless every event, positive and ORDERED candidate list matches.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[2]
REF_DIR = ROOT / "docs" / "sigir" / "panel_refs"


def fp(cands: list[str]) -> str:
    return hashlib.sha1("|".join(cands).encode()).hexdigest()


def make(domains: list[str]) -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    for d in domains:
        src = (ROOT / "outputs" / f"{d}_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate"
               / "tables" / "ranking_eval_records.csv")
        ref = {}
        with open(src, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ref[row["source_event_id"]] = [row["positive_item_id"],
                                               fp(ast.literal_eval(row["candidate_item_ids"]))]
        (REF_DIR / f"{d}_test.json").write_text(json.dumps(ref))
        print(d, len(ref), "events")


def verify(domain: str, ranking: str) -> None:
    ref = json.loads((REF_DIR / f"{domain}_test.json").read_text())
    seen, bad = set(), []
    with open(ranking, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k = r["source_event_id"]
            if k not in ref:
                continue
            seen.add(k)
            if [r["positive_item_id"], fp(r["candidate_item_ids"])] != ref[k]:
                bad.append(k)
    missing = set(ref) - seen
    print(f"{domain}: ref={len(ref)} matched={len(seen) - len(bad)} mismatched={len(bad)} missing={len(missing)}")
    if bad or missing:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "verify"])
    ap.add_argument("--domains", default="sports,toys,home,tools")
    ap.add_argument("--domain")
    ap.add_argument("--ranking")
    a = ap.parse_args()
    if a.mode == "make":
        make(a.domains.split(","))
    else:
        verify(a.domain, a.ranking)


if __name__ == "__main__":
    main()
