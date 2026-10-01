"""Build explicit-feedback (rated) candidate panels whose labels match the yes/no question.

Label: rating >= 4 -> 1 (like), rating <= 2 -> 0 (dislike), rating == 3 dropped (ambiguous).
Per user (chronological): candidates = the last `n_cands` labelled events (keeping >= `min_hist` earlier
events as history); history = the `hist_len` events before the first candidate, rendered with their stars.
Users need >= `min_like` likes and >= `min_dislike` dislikes among candidates so that per-user AUC exists.

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
from collections import Counter, defaultdict
from pathlib import Path

CATEGORY = {"toys": "Toys_and_Games", "beauty": "All_Beauty", "sports": "Sports_and_Outdoors",
            "home": "Home_and_Kitchen", "tools": "Tools_and_Home_Improvement",
            "games": "Video_Games", "cds": "CDs_and_Vinyl", "books": "Books"}


def _text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(_text(v) for v in x if v)
    return str(x).strip()


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


def build(items, events, *, n_users, n_cands, hist_len, min_hist, min_like, min_dislike, seed, source):
    pop = Counter(i for evs in events.values() for _, i, _ in evs)
    rows = []
    for uid, evs in events.items():
        evs = sorted(set(evs))  # dedupe exact repeats, chronological (ts, item)
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
    out = []
    for uid, evs, cand_idx, labs, hist in rows[:n_users]:
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
            "source": source,
        })
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
    rows, n_eligible = build(items, events, n_users=a.n_users, n_cands=a.n_cands, hist_len=a.hist_len,
                             min_hist=a.min_hist, min_like=a.min_like, min_dislike=a.min_dislike,
                             seed=a.seed, source=a.source if a.source == "ml1m" else a.domain)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_c = sum(len(r["candidate_labels"]) for r in rows)
    pos = sum(sum(r["candidate_labels"]) for r in rows)
    meta = {"source": a.source, "domain": a.domain, "eligible_users": n_eligible, "users": len(rows),
            "candidates": n_c, "like_rate": pos / max(1, n_c), **{k: v for k, v in vars(a).items()
                                                                  if k not in ("raw", "out")}}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
