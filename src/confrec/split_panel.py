"""Deterministic USER-level split of a panel jsonl into train / val / test (no user appears in two splits).

    python -m src.confrec.split_panel --panel toys_rated.jsonl --ratios 0.8,0.1,0.1 --seed 0
writes toys_rated_{train,val,test}.jsonl next to the input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def bucket(user_id: str, seed: int, ratios: list[float]) -> int:
    h = int(hashlib.sha1(f"{seed}:{user_id}".encode()).hexdigest(), 16) / 16 ** 40
    acc = 0.0
    for k, r in enumerate(ratios):
        acc += r
        if h < acc:
            return k
    return len(ratios) - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--ratios", default="0.8,0.1,0.1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ratios = [float(x) for x in a.ratios.split(",")]
    p = Path(a.panel)
    outs = [open(p.with_name(f"{p.stem}_{n}.jsonl"), "w", encoding="utf-8") for n in ("train", "val", "test")]
    counts = [0, 0, 0]
    for line in open(p, encoding="utf-8"):
        k = bucket(json.loads(line)["user_id"], a.seed, ratios)
        outs[k].write(line if line.endswith("\n") else line + "\n")
        counts[k] += 1
    for f in outs:
        f.close()
    print(json.dumps(dict(zip(("train", "val", "test"), counts))))


if __name__ == "__main__":
    main()
