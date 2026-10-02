"""Split a rated panel jsonl for LoRA training / held-out evaluation.

User-disjoint (default; robustness split): deterministic hash of user_id, no user in two splits.
    python -m src.confrec.split_panel --panel toys_rated.jsonl --ratios 0.8,0.1,0.1 --seed 0
writes toys_rated_{train,val,test}.jsonl next to the input.

Global temporal (amendment D primary split): T = Q-quantile (numpy linear) of ALL candidate_timestamps in the
panel; every row keeps its candidates with ts < T in train and those with ts >= T in test (all per-candidate
`candidate_*` lists are filtered together; history is unchanged). Rows left with < 2 candidates or a single
label class are dropped from that split.
    python -m src.confrec.split_panel --panel toys_rated.jsonl --time_split 0.8
writes toys_rated_time_{train,test}.jsonl.

--exclude_users_csv a.csv,b.csv (columns incl. user_id, e.g. Lumen's valid/test selected_users.csv) removes those
users from the TRAIN split only, in either mode. Counts go to stdout and to <out prefix>split.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.confrec.stats import strict_json


def bucket(user_id: str, seed: int, ratios: list[float]) -> int:
    h = int(hashlib.sha1(f"{seed}:{user_id}".encode()).hexdigest(), 16) / 16 ** 40
    acc = 0.0
    for k, r in enumerate(ratios):
        acc += r
        if h < acc:
            return k
    return len(ratios) - 1


def read_user_ids(paths: str) -> set[str]:
    users: set[str] = set()
    for p in [x for x in paths.split(",") if x]:
        with open(p, encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            if "user_id" not in (rd.fieldnames or []):
                raise ValueError(f"{p}: no user_id column")
            users.update(r["user_id"] for r in rd)
    return users


def filter_candidates(row: dict, keep: list[bool]) -> dict:
    """Copy of `row` keeping only candidates where keep[i]; every candidate_* list must align."""
    n = len(keep)
    out = dict(row)
    for k, v in row.items():
        if k.startswith("candidate_") and isinstance(v, list):
            if len(v) != n:
                raise ValueError(f"{row.get('source_event_id')}: {k} has {len(v)} entries, expected {n}")
            out[k] = [x for x, m in zip(v, keep) if m]
    return out


def usable(row: dict) -> bool:
    labs = row["candidate_labels"]
    return len(labs) >= 2 and 0 < sum(int(x) for x in labs) < len(labs)


def time_split(rows: list[dict], q: float) -> tuple[float, list[dict], list[dict], dict]:
    if any("candidate_timestamps" not in r for r in rows):
        raise ValueError("--time_split needs candidate_timestamps (rebuild the panel with build_rated_panels)")
    T = float(np.quantile(np.concatenate([np.asarray(r["candidate_timestamps"], float) for r in rows]), q))
    parts = {"train": [], "test": []}
    dropped = {"train": 0, "test": 0}
    for r in rows:
        ts = r["candidate_timestamps"]
        for name, keep in (("train", [t < T for t in ts]), ("test", [t >= T for t in ts])):
            if not any(keep):
                continue
            sub = filter_candidates(r, keep)
            if usable(sub):
                parts[name].append(sub)
            else:
                dropped[name] += 1
    return T, parts["train"], parts["test"], dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--ratios", default="0.8,0.1,0.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude_users_csv", default="", help="comma-separated CSVs with a user_id column")
    ap.add_argument("--time_split", type=float, default=None, help="quantile Q of candidate timestamps")
    a = ap.parse_args()
    p = Path(a.panel)
    lines = [line if line.endswith("\n") else line + "\n" for line in open(p, encoding="utf-8") if line.strip()]
    rows = [json.loads(line) for line in lines]
    excl = read_user_ids(a.exclude_users_csv) if a.exclude_users_csv else set()
    report = {"panel": str(p), "rows": len(rows), "exclude_users_csv": a.exclude_users_csv,
              "n_exclude_users": len(excl)}
    if a.time_split is not None:
        if not 0.0 < a.time_split < 1.0:
            raise ValueError("--time_split must be in (0, 1)")
        T, train, test, dropped = time_split(rows, a.time_split)
        prefix = f"{p.stem}_time_"
        splits = {n: [(r, json.dumps(r, ensure_ascii=False) + "\n") for r in part]
                  for n, part in (("train", train), ("test", test))}
        report.update(mode="time", quantile=a.time_split, T=T, dropped_unusable=dropped)
    else:
        ratios = [float(x) for x in a.ratios.split(",")]
        splits = {"train": [], "val": [], "test": []}
        for r, line in zip(rows, lines):  # rows are written byte-identical to the input
            splits[("train", "val", "test")[bucket(r["user_id"], a.seed, ratios)]].append((r, line))
        prefix = f"{p.stem}_"
        report.update(mode="user", ratios=ratios, seed=a.seed)
    n_before = len(splits["train"])
    splits["train"] = [(r, line) for r, line in splits["train"] if r["user_id"] not in excl]
    report["excluded_from_train"] = n_before - len(splits["train"])
    for name, part in splits.items():
        with open(p.with_name(f"{prefix}{name}.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(line for _, line in part)
        report[name] = {"rows": len(part), "candidates": sum(len(r["candidate_labels"]) for r, _ in part)}
    report = strict_json(report)
    p.with_name(f"{prefix}split.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
