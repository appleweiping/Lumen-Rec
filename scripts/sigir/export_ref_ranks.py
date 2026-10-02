"""Export compact per-event positive ranks of C-CRP v3 and the 8 official baselines (local evidence) so the
GPU server can compute paired ΔNDCG against them on the SAME events.

    python scripts/sigir/export_ref_ranks.py          # -> docs/sigir/ref_ranks/<domain>/<method>.csv.gz

Columns: source_event_id, user_id, positive_rank, num_candidates. Rows are checked against the frozen panel
fingerprints (docs/sigir/panel_refs/<domain>_test.json): every event must be present exactly once.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ("sports", "toys", "home", "tools")
CCRP = "outputs/{d}_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/tables/ranking_eval_records.csv"
BASE = "outputs/baselines/official_adapters"
csv.field_size_limit(1 << 30)


def sources(d: str) -> dict[str, Path]:
    out = {"ccrp_v3": ROOT / CCRP.format(d=d)}
    for p in sorted((ROOT / BASE).glob(f"{d}_large10000_100neg_*_official_qwen3base_same_candidate")):
        method = p.name[len(f"{d}_large10000_100neg_"):].split("_official")[0]
        f = p / "tables" / "ranking_eval_records.csv"
        if f.exists():
            out[method] = f
    return out


def main() -> None:
    bad = 0
    for d in DOMAINS:
        ref = json.loads((ROOT / f"docs/sigir/panel_refs/{d}_test.json").read_text())
        ref_ids = set(ref)  # {source_event_id: [positive_item_id, sha1(ordered candidates)]}
        for method, f in sources(d).items():
            rows = {}
            with open(f, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r.get("split_name", "test") != "test":
                        continue
                    sid = r["source_event_id"]
                    if sid in rows:
                        raise SystemExit(f"duplicate {sid} in {f}")
                    rows[sid] = (r["user_id"], float(r["positive_rank"]), int(r.get("num_candidates") or 101))
            missing = len(ref_ids - set(rows)) if ref_ids else -1
            status = "OK" if missing == 0 and len(rows) == len(ref_ids) else "CHECK"
            bad += status != "OK"
            out = ROOT / f"docs/sigir/ref_ranks/{d}/{method}.csv.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(out, "wt", newline="", encoding="utf-8") as fz:
                w = csv.writer(fz)
                w.writerow(["source_event_id", "user_id", "positive_rank", "num_candidates"])
                for sid in sorted(rows):
                    u, rk, nc = rows[sid]
                    w.writerow([sid, u, f"{rk:g}", nc])
            print(f"{d:6s} {method:18s} rows={len(rows)} missing_vs_ref={missing} {status}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
