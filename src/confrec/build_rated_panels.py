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

Amendment 2 (idea-stage/PREREG_AMENDMENT_2.md G1, prompt variants V4/V5/V7) adds two fields, which build() always
returns and the CLI writes only with --gatefix_fields:
  history_meta  one string per history event, aligned with history_item_ids: "Genres: Action, Drama" (ML-1M,
                movies.dat) or "Categories: <meta categories string>[:80]" (Amazon; never the description);
                "" when the item has none;
  domain_kind   "movie" (ml1m) or "product" (Amazon domains).
The CLI default (also spelled --pilot_format) is the Pilot-1 panel format, without them: panel, .meta.json and
.brand_pop.json are byte-identical to the pre-amendment builder. run_pilot1_mirror.sh / run_pilot2_3.sh rebuild their
panels with the default whenever this file changes and key the Pilot-1 scores on the panel sha1, so the default must
never change. An output whose bytes would not change is left untouched (its mtime too), so such a rebuild changes
nothing downstream. Users, candidates, candidate order and labels do not depend on --hist_len (any value >=
--min_hist): --hist_len 20 only lengthens the history, and truncate_history(row, 10) of a --hist_len 20 row is the
--hist_len 10 row.

    python -m src.confrec.build_rated_panels --source ml1m --raw data/raw/ml-1m \
        --out outputs/confrec/panels/ml1m_rated.jsonl --n_users 1500          # the Pilot-1 panel, byte for byte
    python -m src.confrec.build_rated_panels --source amazon --domain toys --raw data/raw \
        --out outputs/confrec/panels/toys_rated.jsonl --n_users 1500
    python -m src.confrec.build_rated_panels --source amazon --domain games --raw data/raw --hist_len 20 \
        --gatefix_fields --out outputs/confrec/panels/games_rated_h20.jsonl --n_users 1500   # V3/V4/V5/V7-ready

The amendment-2 DEV / CONFIRM split (all eligible users, hist_len 20, with the amendment-2 fields) is
scripts/sigir/build_confirm_panels.py.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from src.confrec.categories import CATEGORY
from src.confrec.stats import strict_json

# slim_amazon2023 keeps lone-surrogate escapes (\ud8xx); json.loads turns them into code points that no UTF-8
# writer or tokenizer accepts, so they become U+FFFD here (titles, texts and stores alike).
_SURROGATE = re.compile("[\ud800-\udfff]")
GATEFIX_FIELDS = ("history_meta", "domain_kind")  # amendment 2; absent from the Pilot-1 panels
HISTORY_FIELDS = ("history", "history_item_ids", "history_titles", "history_ratings", "history_brands",
                  "history_meta")
META_CHARS = 80
DOMAIN_KIND = {"ml1m": "movie"}  # every Amazon domain is "product"


def _text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(_text(v) for v in x if v)
    return _SURROGATE.sub("\ufffd", str(x).strip())


def _intern(x):
    return sys.intern(x) if isinstance(x, str) else x


def ml1m_meta(genres: str) -> str:
    """history_meta of an ML-1M movie: its movies.dat genres, "Genres: Action, Drama"; "" without genres."""
    g = genres.replace("|", ", ")
    return f"Genres: {g}" if g else ""


def amazon_meta(cats: str) -> str:
    """history_meta of an Amazon item: "Categories: " + the first 80 characters of its categories string (the same
    string the candidate description starts with); "" without categories."""
    return f"Categories: {cats[:META_CHARS]}" if cats else ""


def load_ml1m(raw: Path):
    items = {}
    for line in open(raw / "movies.dat", encoding="latin-1"):
        mid, title, genres = line.rstrip("\n").split("::")
        items[mid] = {"title": title, "text": "Genres: " + genres.replace("|", ", "), "brand": "",
                      "meta": ml1m_meta(genres)}
    events = defaultdict(list)
    for line in open(raw / "ratings.dat", encoding="latin-1"):
        uid, mid, r, ts = line.rstrip("\n").split("::")
        events[uid].append((int(ts), mid, float(r)))
    return items, events


def amazon_paths(raw: Path, domain: str) -> tuple[Path, Path]:
    """(meta file, review file) of an Amazon-Reviews-2023 domain as slim_amazon2023.py writes them."""
    cat = CATEGORY[domain]
    d = Path(raw) / f"amazon_{domain}"
    return d / f"meta_{cat}.jsonl.gz", d / f"{cat}.jsonl.gz"


def load_amazon(raw: Path, domain: str, light: bool = False):
    """(items, events) of an Amazon domain; only items with a non-empty title, and their rated reviews, are kept.

    light=True keeps no item text (items map to None): enough for eligible_users(), not for build(). Item ids are
    interned and equal rating floats shared, so the ~16M Toys reviews do not hold one string object per review
    (memory only; the values are unchanged)."""
    meta_path, review_path = amazon_paths(raw, domain)
    items = {}
    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            title = _text(m.get("title"))
            if not title:
                continue
            if light:
                items[_intern(m["parent_asin"])] = None
                continue
            cats = _text(m.get("categories"))
            desc = _text(m.get("description"))
            text = (f"Categories: {cats}. " if cats else "") + desc
            items[_intern(m["parent_asin"])] = {"title": title, "text": text, "brand": _text(m.get("store")),
                                                "meta": amazon_meta(cats)}
    events, ratings = defaultdict(list), {}
    with gzip.open(review_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("parent_asin") in items and r.get("rating") is not None:
                x = float(r["rating"])
                events[r["user_id"]].append((int(r["timestamp"]), _intern(r["parent_asin"]),
                                             ratings.setdefault(x, x)))
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


def select_candidates(evs: list, *, n_cands, hist_len, min_hist, min_like, min_dislike):
    """(cand_idx, labels, hist) for one user's deduplicated chronological events, or None when the user is not
    eligible. cand_idx[0] >= min_hist, so for hist_len >= min_hist eligibility, candidates and labels do not depend
    on hist_len: a longer hist_len only lengthens `hist`."""
    labelled = [k for k, (_, _, r) in enumerate(evs) if r != 3.0]
    labelled = [k for k in labelled if k >= min_hist]
    cand_idx = labelled[-n_cands:]
    if not cand_idx:
        return None
    labs = [int(evs[k][2] >= 4.0) for k in cand_idx]
    if sum(labs) < min_like or len(labs) - sum(labs) < min_dislike:
        return None
    hist = evs[: cand_idx[0]][-hist_len:]
    if len(hist) < min_hist:
        return None
    return cand_idx, labs, hist


def eligible_users(events, *, n_cands, hist_len, min_hist, min_like, min_dislike) -> int:
    """The number of eligible users build() reports (its second return value), without building any row."""
    kw = dict(n_cands=n_cands, hist_len=hist_len, min_hist=min_hist, min_like=min_like, min_dislike=min_dislike)
    return sum(select_candidates(first_per_item(sorted(set(evs))), **kw) is not None for evs in events.values())


def pilot_row(row: dict) -> dict:
    """The row without the amendment-2 fields, i.e. in the Pilot-1 panel format (same key order)."""
    return {k: v for k, v in row.items() if k not in GATEFIX_FIELDS}


def truncate_history(row: dict, k: int) -> dict:
    """The row with every history field cut to its last k entries: what a --hist_len k build writes, for
    min_hist <= k <= the hist_len the row was built with."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return {f: (v[-k:] if f in HISTORY_FIELDS else v) for f, v in row.items()}


def row_line(row: dict) -> str:
    """One panel line, exactly as main() writes it."""
    return json.dumps(row, ensure_ascii=False) + "\n"


def same_bytes(a: Path, b: Path) -> bool:
    """True if the two files hold identical bytes (chunked; no filecmp signature cache)."""
    if a.stat().st_size != b.stat().st_size:
        return False
    with open(a, "rb") as f, open(b, "rb") as g:
        while True:
            x, y = f.read(1 << 20), g.read(1 << 20)
            if x != y:
                return False
            if not x:
                return True


def commit(tmp: Path, out: Path) -> bool:
    """Move the finished temp file onto `out`, unless `out` already holds exactly these bytes: then the temp file is
    dropped and `out` (with its mtime, which the run scripts' skip rules read) stays. True if `out` was replaced."""
    if out.is_file() and same_bytes(tmp, out):
        tmp.unlink()
        return False
    tmp.replace(out)
    return True


def build(items, events, *, n_users, n_cands, hist_len, min_hist, min_like, min_dislike, seed, source,
          stats: dict | None = None, domain_kind: str | None = None):
    kind = domain_kind or DOMAIN_KIND.get(source, "product")
    pop = Counter(i for evs in events.values() for _, i, _ in evs)  # all-time, raw events (unchanged)
    rows, n_removed = [], 0
    for uid, evs in events.items():
        n_raw = len(evs)
        evs = first_per_item(sorted(set(evs)))  # chronological (ts, item); one event per (user, item)
        n_removed += n_raw - len(evs)  # raw minus kept: exact repeats and later (user, item) events
        picked = select_candidates(evs, n_cands=n_cands, hist_len=hist_len, min_hist=min_hist, min_like=min_like,
                                   min_dislike=min_dislike)
        if picked is not None:
            rows.append((uid, evs, *picked))  # (uid, evs, cand_idx, labs, hist)
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
            "history_meta": [items[i].get("meta", "") for _, i, _ in hist],
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
            "domain_kind": kind,
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
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("--pilot_format", action="store_true",
                     help="the default, spelled out: the Pilot-1 panel format, without history_meta / domain_kind")
    fmt.add_argument("--gatefix_fields", action="store_true",
                     help="also write the amendment-2 fields history_meta and domain_kind (prompt variants V4/V5/V7)")
    a = ap.parse_args()
    if a.hist_len < a.min_hist:  # below it every user would be dropped (and 0 would mean the whole history)
        ap.error(f"--hist_len {a.hist_len} < --min_hist {a.min_hist}")
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
    # the default meta.json has exactly the pre-amendment keys; --gatefix_fields adds two at the end
    meta = {"source": a.source, "domain": a.domain, "eligible_users": n_eligible, "users": len(rows),
            "candidates": n_c, "like_rate": pos / max(1, n_c), "n_dedup_removed": info["n_dedup_removed"],
            "n_items": len(items), "n_items_with_store": n_store,
            **{k: v for k, v in vars(a).items() if k not in ("raw", "out", "pilot_format", "gatefix_fields")}}
    if a.gatefix_fields:
        meta.update(gatefix_fields=True, n_items_with_meta=sum(1 for v in items.values() if v.get("meta")))
    if a.source == "amazon" and not n_store:  # pilot 3 (pseudonymize) needs the store field
        print(f"WARNING: no item of {a.domain} has a `store` value: meta_{CATEGORY[a.domain]}.jsonl.gz was probably "
              "slimmed without it; delete that file and rerun scripts/sigir/slim_amazon2023.py before pilot 3")
    # sidecars first: the panel only appears (atomically, last) once everything next to it is complete; a file
    # whose bytes are unchanged is not touched (commit)
    for path, text in ((out.with_suffix(".brand_pop.json"),
                        json.dumps(strict_json(brand_popularity(items, events)), ensure_ascii=False)),
                       (out.with_suffix(".meta.json"), json.dumps(strict_json(meta), indent=2))):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")  # as before (pre-amendment bytes on the same OS)
        commit(tmp, path)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:  # "\n" on every OS: the panel bytes are hashed
        for r in rows:
            f.write(row_line(r if a.gatefix_fields else pilot_row(r)))
    if not commit(tmp, out):
        print(f"{out}: bytes unchanged, file left untouched", file=sys.stderr)
    print(json.dumps(strict_json(meta)))


if __name__ == "__main__":
    main()
