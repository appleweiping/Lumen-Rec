#!/usr/bin/env python3
"""List the token paths of a result file (the helper of scripts/sigir/build_paper.py).

    python scripts/sigir/jpaths.py FILE [SUBSTRING ...] [--results docs/sigir/results] [--max 400] [--depth 12]

Prints one line per scalar leaf or per estimate record ({est, lo, hi, ...}, shown once as `RECORD`), with `/`-separated paths as
`\\jv{FILE|PATH|FMT}` expects them. Only lines containing every SUBSTRING (case-insensitive) are shown.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def walk(node, path, depth, maxdepth, out):
    if depth > maxdepth:
        return
    if isinstance(node, dict):
        if "est" in node and isinstance(node.get("est"), (int, float)) and not isinstance(node.get("est"), bool):
            extra = " ".join(f"{k}={node[k]:.4g}" if isinstance(node.get(k), float) else f"{k}={node[k]}"
                             for k in ("est", "lo", "hi", "p", "n_users", "n") if k in node)
            out.append((path, f"RECORD {extra}"))
            return
        for k, v in node.items():
            walk(v, f"{path}/{k}" if path else str(k), depth + 1, maxdepth, out)
    elif isinstance(node, list):
        if len(node) <= 12 and all(isinstance(x, (int, float, str, bool)) or x is None for x in node):
            out.append((path, "LIST " + json.dumps(node)[:120]))
        else:
            for i, v in enumerate(node[:6]):
                walk(v, f"{path}/{i}", depth + 1, maxdepth, out)
            if len(node) > 6:
                out.append((path + "/...", f"({len(node)} elements)"))
    else:
        out.append((path, repr(node)[:100]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file", help="path relative to --results, e.g. grid/qwen/ml1m.json")
    ap.add_argument("needles", nargs="*")
    ap.add_argument("--results", default=str(ROOT / "docs" / "sigir" / "results"))
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--depth", type=int, default=12)
    a = ap.parse_args(argv)
    p = Path(a.results) / a.file
    if not p.is_file():
        print(f"{p} does not exist", file=sys.stderr)
        return 2
    data = json.loads(p.read_text(encoding="utf-8"))
    out: list = []
    walk(data, "", 0, a.depth, out)
    n = 0
    for path, desc in out:
        line = f"{path}  ::  {desc}"
        if all(s.lower() in line.lower() for s in a.needles):
            print(line)
            n += 1
            if n >= a.max:
                print(f"... (stopped at --max {a.max})")
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
