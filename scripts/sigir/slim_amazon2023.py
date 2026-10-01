"""Stream Amazon-Reviews-2023 raw files from the HF mirror and keep only the fields Lumen reads.

Writes `data/raw/amazon_<domain>/{<Cat>.jsonl.gz, meta_<Cat>.jsonl.gz}` exactly where
`configs/data/amazon_<domain>.yaml` expects them, without ever storing the multi-GB raw jsonl.
`src/data/raw_loaders.py` reads only these columns, so the processed data (and therefore the frozen
1+100 same-candidate panels) are identical to those built from the full raw files.

Usage (server):  python scripts/sigir/slim_amazon2023.py --domains sports,toys,home,tools
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import requests

CATEGORY = {
    "sports": "Sports_and_Outdoors",
    "toys": "Toys_and_Games",
    "home": "Home_and_Kitchen",
    "tools": "Tools_and_Home_Improvement",
    "beauty": "All_Beauty",
    "books": "Books",
    "electronics": "Electronics",
    "movies": "Movies_and_TV",
}
REVIEW_KEYS = ("user_id", "parent_asin", "rating", "timestamp")
# `store` (brand) is extra: Lumen's loader selects only its own columns, so processed data is unchanged;
# the rated-panel builder uses it for the pseudonym-knockout diagnostic.
META_KEYS = ("parent_asin", "title", "categories", "description", "store")
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
BASE = f"{ENDPOINT}/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"


def stream_slim(url: str, keys: tuple[str, ...], out_path: Path, retries: int = 5) -> int:
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    for attempt in range(1, retries + 1):
        n = 0
        try:
            with requests.get(url, stream=True, timeout=120) as r, gzip.open(tmp, "wt", encoding="utf-8") as f:
                r.raise_for_status()
                for line in r.iter_lines(chunk_size=1 << 20):
                    if not line:
                        continue
                    rec = json.loads(line)
                    f.write(json.dumps({k: rec.get(k) for k in keys}, ensure_ascii=False) + "\n")
                    n += 1
            tmp.replace(out_path)
            return n
        except Exception as exc:  # network hiccups on long streams: restart the file
            print(f"[retry {attempt}/{retries}] {url}: {exc}", file=sys.stderr, flush=True)
            time.sleep(10 * attempt)
    raise RuntimeError(f"failed to stream {url}")


def slim_domain(domain: str, root: str) -> str:
    cat = CATEGORY[domain]
    out_dir = Path(root) / f"amazon_{domain}"
    out_dir.mkdir(parents=True, exist_ok=True)
    msgs = []
    for sub, prefix, keys in (("review_categories", "", REVIEW_KEYS), ("meta_categories", "meta_", META_KEYS)):
        out = out_dir / f"{prefix}{cat}.jsonl.gz"
        if out.exists():
            msgs.append(f"{out.name}: exists")
            continue
        t0 = time.time()
        n = stream_slim(f"{BASE}/{sub}/{prefix}{cat}.jsonl", keys, out)
        msgs.append(f"{out.name}: {n} rows in {time.time() - t0:.0f}s")
    return f"{domain}: " + "; ".join(msgs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="sports,toys,home,tools")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    domains = [d for d in args.domains.split(",") if d]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(slim_domain, domains, [args.root] * len(domains)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
