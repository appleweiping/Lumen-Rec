"""Stream Amazon-Reviews-2023 raw files from the HF mirror and keep only the fields Lumen reads.

Writes `data/raw/amazon_<domain>/{<Cat>.jsonl.gz, meta_<Cat>.jsonl.gz}` exactly where
`configs/data/amazon_<domain>.yaml` expects them, without ever storing the multi-GB raw jsonl.
`src/data/raw_loaders.py` reads only these columns, so the processed data (and therefore the frozen
1+100 same-candidate panels) are identical to those built from the full raw files.

Usage (server):  python scripts/sigir/slim_amazon2023.py --domains sports,toys,home,tools
Domains are the keys of src/confrec/categories.py (e.g. games -> Video_Games); finished files are skipped.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.confrec.categories import CATEGORY  # noqa: E402  (shared with build_rated_panels)

REVIEW_KEYS = ("user_id", "parent_asin", "rating", "timestamp")
# `store` (brand) is extra: Lumen's loader selects only its own columns, so processed data is unchanged;
# the rated-panel builder uses it for the pseudonym-knockout diagnostic.
META_KEYS = ("parent_asin", "title", "categories", "description", "store")
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
BASE = f"{ENDPOINT}/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"


def _slim_line(line: bytes, keys: tuple[str, ...]) -> str:
    rec = json.loads(line)
    out = json.dumps({k: rec.get(k) for k in keys}, ensure_ascii=False)
    try:
        out.encode("utf-8")
    except UnicodeEncodeError:  # lone surrogate from a \ud8xx escape: keep it as an ASCII escape (same string)
        out = json.dumps({k: rec.get(k) for k in keys})
    return out + "\n"


def stream_slim(url: str, keys: tuple[str, ...], out_path: Path, retries: int = 20) -> int:
    """Stream `url` line by line, keep `keys`, gzip to `out_path`. Resumable: on a broken connection it
    re-requests from the byte offset of the last COMPLETE line (HTTP Range) and appends a new gzip member
    (multi-member gzip is read transparently by gzip/pandas), so multi-GB files never restart from zero."""
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()  # a .part from an older non-resumable run has unknown offset
    offset, n = 0, 0
    for attempt in range(1, retries + 1):
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as r, \
                    gzip.open(tmp, "at", encoding="utf-8") as f:
                r.raise_for_status()
                if offset and r.status_code != 206:
                    raise RuntimeError(f"server ignored Range (status {r.status_code}); cannot resume")
                buf = b""
                for chunk in r.iter_content(chunk_size=1 << 20):
                    buf += chunk
                    *lines, buf = buf.split(b"\n")
                    for line in lines:
                        if line.strip():
                            f.write(_slim_line(line, keys))
                            n += 1
                        offset += len(line) + 1  # only after the row is written: a failure retries this line
                if buf.strip():  # final line without trailing newline
                    f.write(_slim_line(buf, keys))
                    n += 1
                    offset += len(buf)
            tmp.replace(out_path)
            return n
        except Exception as exc:
            print(f"[retry {attempt}/{retries}] {url} at byte {offset} ({n} rows): {exc}", file=sys.stderr, flush=True)
            time.sleep(min(60, 10 * attempt))
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
    unknown = [d for d in domains if d not in CATEGORY]
    if unknown:
        ap.error(f"unknown domain(s) {unknown}; known: {sorted(CATEGORY)}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(slim_domain, domains, [args.root] * len(domains)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
