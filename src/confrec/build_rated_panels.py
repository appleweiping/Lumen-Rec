"""Build explicit-feedback (rated) candidate panels whose labels match the yes/no question.

Label: rating >= 4 -> 1 (like), rating <= 2 -> 0 (dislike), rating == 3 dropped (ambiguous).
Per user (chronological, first event per item only -- a re-reviewed item would otherwise appear in the history
with its stars and again as a labelled candidate): candidates = the last `n_cands` labelled events (keeping
>= `min_hist` earlier events as history); history = the `hist_len` events before the first candidate, rendered
with their stars. Users need >= `min_like` likes and >= `min_dislike` dislikes among candidates so that per-user
AUC exists.

Popularity: `candidate_popularity` = all-time category event count (Lumen convention, rank-based uses only);
`candidate_popularity_prior` = events on the item strictly before the candidate timestamp (robustness column).
Sidecars: <out stem>.meta.json (incl. n_dedup_removed) and <out stem>.brand_pop.json {store: category events}.

    python -m src.confrec.build_rated_panels --source ml1m --raw data/raw/ml-1m \
        --out outputs/confrec/panels/ml1m_rated.jsonl --n_users 1500
    python -m src.confrec.build_rated_panels --source amazon --domain toys --raw data/raw \
        --out outputs/confrec/panels/toys_rated.jsonl --n_users 1500
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from src.confrec.categories import CATEGORY
from src.confrec.stats import strict_json

# slim_amazon2023 keeps lone-surrogate escapes (\ud8xx); json.loads turns them into code points that no UTF-8
# writer or tokenizer accepts, so they become U+FFFD here (titles, texts and stores alike).
_SURROGATE = re.compile("[\ud800-\udfff]")


def _text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(_text(v) for v in x if v)
    return _SURROGATE.sub("\ufffd", str(x).strip())


def load_ml1m(raw: Path):
    items = {}
    for line in open(raw / "movies.dat", encoding="latin-1"):
        mid, title, genres = line.rstrip("\n").split("::")
        items[mid] = {"title": title, "text": "Genres: " + genres.replace("|", ", "), "brand": ""}
    events = defaultdict(list)
    for line in open(raw / "ratings.dat", encoding="latin-1"):
        uid, mid, r, ts = line.rstrip("\n").split("::")
        events[uid].append((int(ts), mid, float(r)))
    return items, events


def load_amazon(raw: Path, domain: str):
    cat = CATEGORY[domain]
    d = raw / f"amazon_{domain}"
    items = {}
    with gzip.open(d / f"meta_{cat}.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            title = _text(m.get("title"))
            if not title:
                continue
            cats = _text(m.get("categories"))
            desc = _text(m.get("description"))
            text = (f"Categories: {cats}. " if cats else "") + desc
            items[m["parent_asin"]] = {"title": title, "text": text, "brand": _text(m.get("store"))}
    events = defaultdict(list)
    with gzip.open(d / f"{cat}.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("parent_asin") in items and r.get("rating") is not None:
                events[r["user_id"]].append((int(r["timestamp"]), r["parent_asin"], float(r["rating"])))
    return items, events


def first_per_item(evs: list) -> list:
    """Keep the first event of each item in an already chronological (ts, item, rating) list."""
    seen, out = set(), []
    for e in evs:
        if e[1] not in seen:
            seen.add(e[1])
            out.append(e)
    return out


def brand_popularity(items, events) -> dict:
    """{store: category-wide event count summed over that store's items}; empty store excluded."""
    pop = Counter(i for evs in events.values() for _, i, _ in evs)
    out = Counter()
    for i, n in pop.items():
        b = items[i]["brand"]
        if b:
            out[b] += n
    return dict(sorted(out.items()))


def build(items, events, *, n_users, n_cands, hist_len, min_hist, min_like, min_dislike, seed, source,
          stats: dict | None = None):
    pop = Counter(i for evs in events.values() for _, i, _ in evs)  # all-time, raw events (unchanged)
    rows, n_removed = [], 0
    for uid, evs in events.items():
        n_raw = len(evs)
        evs = first_per_item(sorted(set(evs)))  # chronological (ts, item); one event per (user, item)
        n_removed += n_raw - len(evs)  # raw minus kept: exact repeats and later (user, item) events
        labelled = [k for k, (_, _, r) in enumerate(evs) if r != 3.0]
        labelled = [k for k in labelled if k >= min_hist]
        cand_idx = labelled[-n_cands:]
        if not cand_idx:
            continue
        labs = [int(evs[k][2] >= 4.0) for k in cand_idx]
        if sum(labs) < min_like or len(labs) - sum(labs) < min_dislike:
            continue
        hist = evs[: cand_idx[0]][-hist_len:]
        if len(hist) < min_hist:
            continue
        rows.append((uid, evs, cand_idx, labs, hist))
    rng = random.Random(seed)
    rng.shuffle(rows)
    sel = rows[:n_users]
    need = {evs[k][1] for _, evs, cand_idx, _, _ in sel for k in cand_idx}
    ts_by_item = defaultdict(list)  # raw event timestamps per candidate item, sorted for bisect
    for evs in events.values():
        for t, i, _ in evs:
            if i in need:
                ts_by_item[i].append(t)
    for v in ts_by_item.values():
        v.sort()
    out = []
    for uid, evs, cand_idx, labs, hist in sel:
        order = list(range(len(cand_idx)))
        rng.shuffle(order)  # candidate order carries no label information
        cand = [evs[cand_idx[o]] for o in order]
        out.append({
            "user_id": uid,
            "source_event_id": f"{uid}::{evs[cand_idx[0]][0]}",
            "history": [f"{items[i]['title']} (rated {int(r)}/5)" for _, i, r in hist],
            "history_item_ids": [i for _, i, _ in hist],
            "history_titles": [items[i]["title"] for _, i, _ in hist],
            "history_ratings": [r for _, _, r in hist],
            "history_brands": [items[i]["brand"] for _, i, _ in hist],
            "candidate_item_ids": [i for _, i, _ in cand],
            "candidate_titles": [items[i]["title"] for _, i, _ in cand],
            "candidate_texts": [items[i]["text"] for _, i, _ in cand],
            "candidate_brands": [items[i]["brand"] for _, i, _ in cand],
            "candidate_ratings": [r for _, _, r in cand],
            "candidate_labels": [labs[o] for o in order],
            "candidate_popularity": [pop[i] for _, i, _ in cand],
            "candidate_popularity_prior": [bisect_left(ts_by_item[i], t) for t, i, _ in cand],
            "candidate_timestamps": [int(t) for t, _, _ in cand],
            "source": source,
        })
    if stats is not None:
        stats["n_dedup_removed"] = n_removed
    return out, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["ml1m", "amazon"], required=True)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_users", type=int, default=1500)
    ap.add_argument("--n_cands", type=int, default=20)
    ap.add_argument("--hist_len", type=int, default=10)
    ap.add_argument("--min_hist", type=int, default=3)
    ap.add_argument("--min_like", type=int, default=3)
    ap.add_argument("--min_dislike", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    raw = Path(a.raw)
    items, events = load_ml1m(raw) if a.source == "ml1m" else load_amazon(raw, a.domain)
    info: dict = {}
    rows, n_eligible = build(items, events, n_users=a.n_users, n_cands=a.n_cands, hist_len=a.hist_len,
                             min_hist=a.min_hist, min_like=a.min_like, min_dislike=a.min_dislike,
                             seed=a.seed, source=a.source if a.source == "ml1m" else a.domain, stats=info)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_c = sum(len(r["candidate_labels"]) for r in rows)
    pos = sum(sum(r["candidate_labels"]) for r in rows)
    n_store = sum(1 for v in items.values() if v["brand"])
    meta = {"source": a.source, "domain": a.domain, "eligible_users": n_eligible, "users": len(rows),
            "candidates": n_c, "like_rate": pos / max(1, n_c), "n_dedup_removed": info["n_dedup_removed"],
            "n_items": len(items), "n_items_with_store": n_store,
            **{k: v for k, v in vars(a).items() if k not in ("raw", "out")}}
    if a.source == "amazon" and not n_store:  # pilot 3 (pseudonymize) needs the store field
        print(f"WARNING: no item of {a.domain} has a `store` value: meta_{CATEGORY[a.domain]}.jsonl.gz was probably "
              "slimmed without it; delete that file and rerun scripts/sigir/slim_amazon2023.py before pilot 3")
    # sidecars first: the panel only appears (atomically, last) once everything next to it is complete
    out.with_suffix(".brand_pop.json").write_text(
        json.dumps(strict_json(brand_popularity(items, events)), ensure_ascii=False), encoding="utf-8")
    out.with_suffix(".meta.json").write_text(json.dumps(strict_json(meta), indent=2), encoding="utf-8")
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)
    print(json.dumps(strict_json(meta)))


if __name__ == "__main__":
    main()
