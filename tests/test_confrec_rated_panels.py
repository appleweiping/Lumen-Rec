import argparse
import gzip
import hashlib
import json
import os
import random
import re
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from src.confrec import build_rated_panels as brp
from src.confrec.build_rated_panels import build
from src.confrec.categories import CATEGORY
from src.confrec.stats import strict_json

# fields added after the verbatim pre-amendment-1 builder below: amendment 1 (prior popularity, timestamps) and
# amendment 2 (history_meta, domain_kind)
NEW_FIELDS = {"candidate_popularity_prior", "candidate_timestamps", "history_meta", "domain_kind"}
# key order of the real Pilot-1 rated panels (ml1m_rated.jsonl / toys_rated.jsonl, data_sha1 985494c7 / 69252b48)
PILOT1_KEYS = ["user_id", "source_event_id", "history", "history_item_ids", "history_titles", "history_ratings",
               "history_brands", "candidate_item_ids", "candidate_titles", "candidate_texts", "candidate_brands",
               "candidate_ratings", "candidate_labels", "candidate_popularity", "candidate_popularity_prior",
               "candidate_timestamps", "source"]
PILOT_DIR = Path("D:/_Organized/Temp-Review/_RootDirs/temp/claude/D--/22c9b1b2-12a5-4956-a490-79bcb9beb0d8/"
                 "scratchpad/pilot_panels")   # local read-only copies of the real Pilot-1 panels (skipped elsewhere)
PILOT_SHA1 = {"ml1m_rated": "985494c7b44d010bec4b11ec62f35ad274dfe91f",
              "toys_rated": "69252b4806bb4ffe05101967ea2a1d94ada57dee"}
# keys of the Pilot-1 <panel>.meta.json, in order (what the default CLI must keep writing)
PILOT1_META_KEYS = ["source", "domain", "eligible_users", "users", "candidates", "like_rate", "n_dedup_removed",
                    "n_items", "n_items_with_store", "n_users", "n_cands", "hist_len", "min_hist", "min_like",
                    "min_dislike", "seed"]
# the canonical GroupLens ML-1M ratings.dat; a real copy enables the end-to-end default-CLI check (skipped elsewhere)
ML1M_RATINGS_SHA1 = "28770e0e10e29ec977d365f3c65eb63722d86de4"
REAL_ML1M = [Path(p) for p in (os.environ.get("LUMEN_ML1M_RAW", ""),) if p] + [
    Path(__file__).resolve().parents[1] / "data" / "raw" / "ml-1m",      # the server checkout
    Path("D:/Research/LLM/USCRec-LLM4Rec/data/raw/ml-1m")]               # a local copy


def _old_build(items, events, *, n_users, n_cands, hist_len, min_hist, min_like, min_dislike, seed, source):
    """Verbatim copy of the pre-amendment builder (no per-item dedup), the reference for unchanged behaviour."""
    pop = Counter(i for evs in events.values() for _, i, _ in evs)
    rows = []
    for uid, evs in events.items():
        evs = sorted(set(evs))
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
        rng.shuffle(order)
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


KW = dict(n_cands=6, hist_len=10, min_hist=3, min_like=3, min_dislike=3, seed=0, source="toy")


def _toy():
    items = {f"i{k}": {"title": f"T{k}", "text": "", "brand": ""} for k in range(30)}
    events = defaultdict(list)
    # u1: 4 history events, then 6 labelled candidates (3 likes, 3 dislikes) + a 3-star that must be dropped
    for t, (i, r) in enumerate([(0, 5), (1, 3), (2, 4), (3, 1), (4, 5), (5, 4), (6, 3), (7, 5),
                                (8, 1), (9, 2), (10, 2)]):
        events["u1"].append((t, f"i{i}", float(r)))
    # u2: only likes -> excluded (no dislikes)
    for t in range(10):
        events["u2"].append((t, f"i{t}", 5.0))
    return items, events


def test_rated_panel_labels_and_chronology():
    items, events = _toy()
    rows, n_eligible = build(items, events, n_users=10, **KW)
    assert n_eligible == 1 and len(rows) == 1
    r = rows[0]
    assert sorted(r["candidate_ratings"]) == [1.0, 2.0, 2.0, 4.0, 5.0, 5.0]   # 3-star dropped
    assert all(l == int(x >= 4) for l, x in zip(r["candidate_labels"], r["candidate_ratings"]))
    assert sum(r["candidate_labels"]) == 3
    cand = set(r["candidate_item_ids"])
    assert not cand & set(r["history_item_ids"])                              # no leakage
    assert r["history_item_ids"] == ["i0", "i1", "i2", "i3"]                  # strictly before candidates
    assert r["history"][1].endswith("(rated 3/5)")


def _rich_fixture(n_users=40, seed=1):
    """Many users, items shared ACROSS users but never repeated WITHIN a user; same-second ties included."""
    rng = random.Random(seed)
    items = {f"p{k}": {"title": f"Title {k}", "text": f"desc {k}", "brand": f"B{k % 7}" if k % 5 else ""}
             for k in range(200)}
    events = defaultdict(list)
    for u in range(n_users):
        its = rng.sample(sorted(items), rng.randint(5, 40))
        t = rng.randint(0, 1000)
        for i in its:
            t += rng.choice([0, 1, 5, 60])  # 0 -> same-second tie
            events[f"u{u}"].append((t, i, float(rng.choice([1, 2, 3, 4, 5]))))
    return items, events


def test_no_duplicate_fixture_identical_to_old_builder_apart_from_new_fields():
    items, events = _rich_fixture()
    for n_users in (5, 1000):
        stats = {}
        new, n_new = build(items, events, n_users=n_users, stats=stats, **{**KW, "n_cands": 20})
        old, n_old = _old_build(items, events, n_users=n_users, **{**KW, "n_cands": 20})
        assert n_new == n_old > 5 and stats["n_dedup_removed"] == 0
        assert [{k: v for k, v in r.items() if k not in NEW_FIELDS} for r in new] == old
        assert all(set(r) - set(o) == NEW_FIELDS for r, o in zip(new, old))


def test_rereviewed_item_removed_from_candidates():
    items = {f"i{k}": {"title": f"T{k}", "text": "", "brand": ""} for k in range(20)}
    events = defaultdict(list)
    # P = i0 is reviewed at t=1 (history window) and re-reviewed at t=20 (candidate window), both 5 stars;
    # i9 is reviewed twice inside the candidate window with conflicting stars.
    seq = [(0, "i1", 4), (1, "i0", 5), (2, "i2", 2), (3, "i3", 5),
           (10, "i4", 5), (11, "i5", 1), (12, "i6", 4), (13, "i7", 2), (14, "i8", 5), (15, "i9", 1),
           (16, "i10", 2), (18, "i9", 5), (20, "i0", 5)]
    events["u1"] = [(t, i, float(r)) for t, i, r in seq]
    old, _ = _old_build(items, events, n_users=10, **{**KW, "n_cands": 20})
    assert "i0" in old[0]["history_item_ids"] and "i0" in old[0]["candidate_item_ids"]   # the leak
    stats = {}
    rows, _ = build(items, events, n_users=10, stats=stats, **{**KW, "n_cands": 20})
    r = rows[0]
    assert stats["n_dedup_removed"] == 2
    assert r["history_item_ids"] == ["i1", "i0", "i2"]              # first event kept, with its stars
    assert "i0" not in r["candidate_item_ids"]
    assert len(set(r["candidate_item_ids"])) == len(r["candidate_item_ids"])
    i9 = r["candidate_item_ids"].index("i9")
    assert r["candidate_ratings"][i9] == 1.0 and r["candidate_timestamps"][i9] == 15   # first i9 review


def test_popularity_prior_counts_events_strictly_before_candidate():
    items = {f"i{k}": {"title": f"T{k}", "text": "", "brand": ""} for k in range(12)}
    events = defaultdict(list)
    events["u1"] = [(t, f"i{t}", 5.0) for t in range(3)] + \
                   [(10, "i10", 5.0), (11, "i3", 1.0), (12, "i4", 5.0), (13, "i5", 1.0), (14, "i6", 5.0),
                    (15, "i7", 1.0)]
    # other users' events on i10: before (5, 9), same second (10, not counted), after (50)
    events["u2"] = [(5, "i10", 4.0), (9, "i10", 3.0), (10, "i10", 1.0), (50, "i10", 5.0), (60, "i3", 5.0)]
    rows, _ = build(items, events, n_users=10, **{**KW, "n_cands": 20})
    r = next(x for x in rows if x["user_id"] == "u1")
    k = r["candidate_item_ids"].index("i10")
    assert r["candidate_timestamps"][k] == 10
    assert r["candidate_popularity_prior"][k] == 2 and r["candidate_popularity"][k] == 5
    j = r["candidate_item_ids"].index("i3")
    assert r["candidate_popularity_prior"][j] == 0 and r["candidate_popularity"][j] == 2
    assert all(p <= q for p, q in zip(r["candidate_popularity_prior"], r["candidate_popularity"]))
    assert all(isinstance(t, int) for t in r["candidate_timestamps"])


def _write_gz(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")


def test_main_writes_meta_and_brand_pop_sidecar(tmp_path, monkeypatch):
    d = tmp_path / "raw" / "amazon_toys"
    stores = {"a1": "LEGO", "a2": "LEGO", "a3": "Mattel", "a4": "", "a5": "Hasbro", "a6": "Hasbro",
              "a7": "LEGO", "a8": "Mattel", "a9": "", "a10": "Hasbro"}
    _write_gz(d / "meta_Toys_and_Games.jsonl.gz",
              [{"parent_asin": k, "title": f"Toy {k}", "categories": ["Toys"], "description": ["d"], "store": s}
               for k, s in stores.items()] + [{"parent_asin": "x", "title": "", "store": "Ghost"}])
    revs = [("U", k, t, r) for t, (k, r) in enumerate(
        [("a1", 5), ("a2", 4), ("a3", 2), ("a4", 5), ("a5", 1), ("a6", 5), ("a7", 2), ("a8", 4), ("a9", 1),
         ("a10", 5), ("a1", 5)])]                                   # a1 re-reviewed -> removed per user
    revs += [("V", "a1", 100, 5), ("V", "a5", 101, 3), ("V", "x", 102, 5)]   # x has no title -> dropped
    _write_gz(d / "Toys_and_Games.jsonl.gz",
              [{"user_id": u, "parent_asin": k, "rating": float(r), "timestamp": t} for u, k, t, r in revs])
    out = tmp_path / "panels" / "toys_rated.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", "--source", "amazon", "--domain", "toys", "--raw", str(tmp_path / "raw"),
                                      "--out", str(out), "--n_cands", "7"])
    brp.main()
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["n_dedup_removed"] == 1 and meta["users"] == 1
    bp = json.loads(out.with_suffix(".brand_pop.json").read_text())
    # all events (incl. the re-review and other users' / 3-star events) over each store's items; '' excluded
    assert bp == {"LEGO": 5, "Mattel": 2, "Hasbro": 4}
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(rows) == 1 and "a1" in rows[0]["history_item_ids"] and "a1" not in rows[0]["candidate_item_ids"]
    assert not (out.parent / "toys_rated.jsonl.tmp").exists()
    assert meta["n_items"] == 10 and meta["n_items_with_store"] == 8


def test_meta_slimmed_without_store_is_flagged(tmp_path, monkeypatch, capsys):
    # a meta file slimmed before `store` was kept gives a brandless panel; pilot 3 needs the stores
    d = tmp_path / "raw" / "amazon_toys"
    _write_gz(d / "meta_Toys_and_Games.jsonl.gz", [{"parent_asin": f"a{k}", "title": f"Toy {k}",
                                                    "categories": ["Toys"], "description": ["d"]} for k in range(10)])
    _write_gz(d / "Toys_and_Games.jsonl.gz", [{"user_id": "U", "parent_asin": f"a{k}", "rating": float(r),
                                               "timestamp": k} for k, r in enumerate([5, 4, 2, 5, 1, 5, 2, 4, 1, 5])])
    out = tmp_path / "panels" / "toys_rated.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", "--source", "amazon", "--domain", "toys", "--raw", str(tmp_path / "raw"),
                                      "--out", str(out), "--n_cands", "7"])
    brp.main()
    assert json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))["n_items_with_store"] == 0
    assert "WARNING: no item of toys has a `store` value" in capsys.readouterr().out


def test_lone_surrogate_and_non_ascii_store_survive_main(tmp_path, monkeypatch):
    # slim_amazon2023 keeps lone-surrogate escapes as ASCII escapes; json.loads turns them into lone surrogates,
    # which used to crash the UTF-8 panel write. Non-ASCII stores used to go through the locale encoding (cp936
    # here) while pseudonymize reads the sidecar as UTF-8.
    d = tmp_path / "raw" / "amazon_toys"
    d.mkdir(parents=True)
    stores = ["Ravensburger®", "Pokémon", "LEGO"]
    meta = [{"parent_asin": f"a{k}", "title": f"Toy {k}", "categories": ["Toys"], "description": ["d"],
             "store": stores[k % 3]} for k in range(10)]
    lines = [json.dumps(m) for m in meta]
    lines[4] = lines[4].replace('"Toy 4"', '"Toy \\ud800 4"')     # a lone-surrogate escape, as slim writes it
    with gzip.open(d / "meta_Toys_and_Games.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    ratings = [5, 4, 2, 5, 1, 5, 2, 4, 1, 5]
    _write_gz(d / "Toys_and_Games.jsonl.gz", [{"user_id": "U", "parent_asin": f"a{k}", "rating": float(r),
                                                "timestamp": k} for k, r in enumerate(ratings)])
    out = tmp_path / "panels" / "toys_rated.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", "--source", "amazon", "--domain", "toys", "--raw", str(tmp_path / "raw"),
                                      "--out", str(out), "--n_cands", "7"])
    brp.main()
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    titles = rows[0]["history_titles"] + rows[0]["candidate_titles"]
    assert "Toy \ufffd 4" in titles
    assert all(not any(0xD800 <= ord(c) <= 0xDFFF for c in t) for t in titles)
    bp = json.loads(out.with_suffix(".brand_pop.json").read_text(encoding="utf-8"))
    assert bp == {"LEGO": 3, "Pokémon": 3, "Ravensburger®": 4}
    assert json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))["users"] == 1


# ------------------------------------------------------------------ amendment 2: history_meta, domain_kind, hist_len 20
def _long_fixture(n_users=60, seed=3):
    """Users with 5-70 events (histories longer than 10 and 20 occur), items shared across users, same-second ties,
    3-star events and re-reviews (removed by the per-user dedup)."""
    rng = random.Random(seed)
    items = {f"m{k}": {"title": f"Movie {k}", "text": f"Genres: G{k % 4}", "brand": "",
                       "meta": f"Genres: G{k % 4}" if k % 9 else ""} for k in range(300)}
    events = defaultdict(list)
    for u in range(n_users):
        its = rng.sample(sorted(items), rng.randint(5, 70))
        t = rng.randint(0, 1000)
        for i in its:
            t += rng.choice([0, 1, 7])
            events[f"u{u}"].append((t, i, float(rng.choice([1, 2, 3, 4, 5]))))
        if rng.random() < 0.3:
            events[f"u{u}"].append((t + 5, its[0], 5.0))   # re-review of the first item
    return items, events


ELIG_KW = {k: KW[k] for k in ("min_hist", "min_like", "min_dislike")}


def test_hist_len_20_selects_same_users_candidates_order_and_labels_as_hist_len_10():
    items, events = _long_fixture()
    kw = {**KW, "n_cands": 20}
    n_all = brp.eligible_users(events, n_cands=20, hist_len=10, **ELIG_KW)
    assert n_all == brp.eligible_users(events, n_cands=20, hist_len=20, **ELIG_KW) > 9
    for n_users in (9, 10 ** 9):
        r10, n10 = build(items, events, n_users=n_users, **{**kw, "hist_len": 10})
        r20, n20 = build(items, events, n_users=n_users, **{**kw, "hist_len": 20})
        assert n10 == n20 == n_all and len(r10) == len(r20) == min(n_users, n_all)
        assert [r["user_id"] for r in r20] == [r["user_id"] for r in r10]
        for a, b in zip(r10, r20):
            assert {k: v for k, v in b.items() if k not in brp.HISTORY_FIELDS} == \
                   {k: v for k, v in a.items() if k not in brp.HISTORY_FIELDS}       # candidates, order, labels...
            assert brp.truncate_history(b, 10) == a
            assert brp.row_line(brp.pilot_row(brp.truncate_history(b, 10))) == brp.row_line(brp.pilot_row(a))
            # the history is the last 20 deduplicated events strictly before the first candidate
            evs = brp.first_per_item(sorted(set(events[b["user_id"]])))
            pos = {e[1]: k for k, e in enumerate(evs)}
            first = min(pos[i] for i in b["candidate_item_ids"])
            assert b["history_item_ids"] == [i for _, i, _ in evs[:first][-20:]]
            assert b["history_meta"] == [items[i]["meta"] for i in b["history_item_ids"]]
        assert any(len(b["history"]) > 10 for b in r20)
    assert max(len(b["history"]) for b in r20) == 20


def test_eligible_users_matches_build_with_duplicates_and_ties():
    for items, events in (_rich_fixture(), _long_fixture(seed=4)):
        for n_cands in (6, 20):
            _, n = build(items, events, n_users=3, **{**KW, "n_cands": n_cands})
            assert brp.eligible_users(events, n_cands=n_cands, hist_len=10, **ELIG_KW) == n


def test_meta_helpers():
    assert brp.ml1m_meta("Action|Drama") == "Genres: Action, Drama" and brp.ml1m_meta("") == ""
    assert brp.amazon_meta("") == ""
    assert brp.amazon_meta("Toys & Games Puzzles") == "Categories: Toys & Games Puzzles"
    assert brp.amazon_meta("x" * 100) == "Categories: " + "x" * 80
    with pytest.raises(ValueError):
        brp.truncate_history({"history": []}, 0)


def _ml1m_raw(d, genres, n_movies=40):
    d.mkdir(parents=True)
    (d / "movies.dat").write_text("".join(f"{m}::Movie {m} (1990)::{genres[m % len(genres)]}\n"
                                          for m in range(1, n_movies + 1)), encoding="latin-1")
    rng = random.Random(0)
    lines = []
    for u in (1, 2):
        t = 1000 * u
        for m in rng.sample(range(1, n_movies + 1), 30):
            t += 3
            lines.append(f"{u}::{m}::{rng.choice([1, 2, 4, 5])}::{t}\n")
    (d / "ratings.dat").write_text("".join(lines), encoding="latin-1")


def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["x", *map(str, argv)])
    brp.main()


def test_history_meta_and_domain_kind_ml1m(tmp_path, monkeypatch):
    d = tmp_path / "ml-1m"
    _ml1m_raw(d, ["Action|Drama", "Comedy", "Sci-Fi|Thriller|War"])
    out = tmp_path / "p" / "ml1m.jsonl"
    _run_main(monkeypatch, "--source", "ml1m", "--raw", d, "--out", out, "--gatefix_fields")
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    genres = {str(m): ["Action, Drama", "Comedy", "Sci-Fi, Thriller, War"][m % 3] for m in range(1, 41)}
    assert len(rows) == 2
    for r in rows:
        assert r["domain_kind"] == "movie" and r["source"] == "ml1m"
        assert len(r["history_meta"]) == len(r["history_item_ids"]) == 10
        assert r["history_meta"] == [f"Genres: {genres[i]}" for i in r["history_item_ids"]]
        assert r["candidate_texts"] == [f"Genres: {genres[i]}" for i in r["candidate_item_ids"]]   # unchanged
        assert list(r) == PILOT1_KEYS[:7] + ["history_meta"] + PILOT1_KEYS[7:] + ["domain_kind"]
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert list(meta) == PILOT1_META_KEYS + ["gatefix_fields", "n_items_with_meta"]
    assert meta["n_items_with_meta"] == 40 and meta["gatefix_fields"] is True


def test_history_meta_and_domain_kind_amazon(tmp_path, monkeypatch):
    d = tmp_path / "raw" / "amazon_toys"
    long_cats = ["Toys & Games", "Building Toys", "Building Sets", "Construction Kits For Ages 8 And Up", "Extra"]
    cats_of = {k: [long_cats, ["Toys & Games", "Puzzles"], []][k % 3] for k in range(12)}
    meta = [{"parent_asin": f"a{k}", "title": f"Toy {k}", "categories": cats_of[k], "description": ["d"],
             "store": "S"} for k in range(12)]
    meta[4].pop("categories")                                       # no categories key at all -> ""
    _write_gz(d / "meta_Toys_and_Games.jsonl.gz", meta)
    ratings = [5, 4, 2, 5, 1, 5, 2, 4, 1, 5, 2, 4]
    _write_gz(d / "Toys_and_Games.jsonl.gz", [{"user_id": "U", "parent_asin": f"a{k}", "rating": float(r),
                                               "timestamp": k} for k, r in enumerate(ratings)])
    out = tmp_path / "panels" / "toys_rated.jsonl"
    _run_main(monkeypatch, "--source", "amazon", "--domain", "toys", "--raw", tmp_path / "raw", "--out", out,
              "--n_cands", "7", "--hist_len", "20", "--gatefix_fields")
    (r,) = [json.loads(l) for l in open(out, encoding="utf-8")]
    joined = " ".join(long_cats)
    assert len(joined) > 80
    assert r["history_item_ids"] == ["a0", "a1", "a2", "a3", "a4"]
    assert r["history_meta"] == ["Categories: " + joined[:80], "Categories: Toys & Games Puzzles", "",
                                 "Categories: " + joined[:80], ""]
    assert r["domain_kind"] == "product" and r["source"] == "toys"
    # the candidate description keeps the full categories string; only history_meta is cut at 80 characters
    assert r["candidate_texts"][r["candidate_item_ids"].index("a6")] == f"Categories: {joined}. d"
    meta_out = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta_out["n_items_with_meta"] == 7 and meta_out["hist_len"] == 20


def test_default_cli_is_the_pilot1_format_and_gatefix_fields_opt_in(tmp_path, monkeypatch):
    # run_pilot1_mirror.sh / run_pilot2_3.sh rebuild the Pilot-1 panels with the default CLI whenever the builder
    # changes and key the Pilot-1 scores on the panel sha1: the default must stay the Pilot-1 format
    d = tmp_path / "ml-1m"
    _ml1m_raw(d, ["Drama", "Comedy|Romance"])
    dflt, pf, gf = tmp_path / "a" / "p.jsonl", tmp_path / "b" / "p.jsonl", tmp_path / "c" / "p.jsonl"
    _run_main(monkeypatch, "--source", "ml1m", "--raw", d, "--out", dflt)
    _run_main(monkeypatch, "--source", "ml1m", "--raw", d, "--out", pf, "--pilot_format")
    _run_main(monkeypatch, "--source", "ml1m", "--raw", d, "--out", gf, "--gatefix_fields")
    for suffix in (".jsonl", ".meta.json", ".brand_pop.json"):     # --pilot_format only spells out the default
        assert dflt.with_suffix(suffix).read_bytes() == pf.with_suffix(suffix).read_bytes()
    d_bytes, g_bytes = dflt.read_bytes(), gf.read_bytes()
    assert b"\r\n" not in d_bytes + g_bytes                         # "\n" line ends on every OS (bytes are hashed)
    assert all(list(json.loads(l)) == PILOT1_KEYS for l in d_bytes.decode("utf-8").split("\n")[:-1])
    rows_g = [json.loads(l) for l in g_bytes.decode("utf-8").split("\n")[:-1]]
    assert all(set(r) - set(PILOT1_KEYS) == set(brp.GATEFIX_FIELDS) for r in rows_g)
    assert d_bytes == "".join(brp.row_line(brp.pilot_row(r)) for r in rows_g).encode("utf-8")
    assert list(json.loads(dflt.with_suffix(".meta.json").read_text(encoding="utf-8"))) == PILOT1_META_KEYS
    assert json.loads(gf.with_suffix(".meta.json").read_text(encoding="utf-8"))["gatefix_fields"] is True
    with pytest.raises(SystemExit):                                 # the two formats exclude each other
        _run_main(monkeypatch, "--source", "ml1m", "--raw", d, "--out", tmp_path / "x.jsonl", "--pilot_format",
                  "--gatefix_fields")


def test_unchanged_outputs_keep_their_mtime_and_changed_ones_are_replaced(tmp_path, monkeypatch):
    # the run scripts' skip rules read mtimes: rebuilding a panel with identical bytes must not look like a change
    d = tmp_path / "ml-1m"
    _ml1m_raw(d, ["Drama", "Comedy|Romance"])
    out = tmp_path / "p" / "ml1m_rated.jsonl"
    files = [out, out.with_suffix(".meta.json"), out.with_suffix(".brand_pop.json")]
    _run_main(monkeypatch, "--source", "ml1m", "--raw", d, "--out", out)
    before = {p: p.read_bytes() for p in files}
    for p in files:
        os.utime(p, (1_000_000_000, 1_000_000_000))
    _run_main(monkeypatch, "--source", "ml1m", "--raw", d, "--out", out)
    assert all(p.stat().st_mtime == 1_000_000_000 and p.read_bytes() == before[p] for p in files)
    _run_main(monkeypatch, "--source", "ml1m", "--raw", d, "--out", out, "--gatefix_fields")   # other bytes
    assert out.read_bytes() != before[out] and out.stat().st_mtime > 1_000_000_000
    assert out.with_suffix(".meta.json").stat().st_mtime > 1_000_000_000
    assert out.with_suffix(".brand_pop.json").stat().st_mtime == 1_000_000_000              # same bytes: kept
    assert sorted(p.name for p in out.parent.iterdir()) == sorted(p.name for p in files)    # no .tmp left
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"x" * 10)
    b.write_bytes(b"x" * 9 + b"y")
    assert not brp.same_bytes(a, b) and brp.same_bytes(a, a)


@pytest.mark.parametrize("hist_len", ["2", "0"])
def test_hist_len_below_min_hist_is_rejected(tmp_path, monkeypatch, hist_len):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, "--source", "ml1m", "--raw", tmp_path / "none", "--out", tmp_path / "o.jsonl",
                  "--hist_len", hist_len)


def test_light_amazon_loader_gives_the_same_items_events_and_eligible_count(tmp_path):
    d = tmp_path / "raw" / "amazon_toys"
    rng = random.Random(5)
    _write_gz(d / "meta_Toys_and_Games.jsonl.gz",
              [{"parent_asin": f"a{k}", "title": f"Toy {k}" if k % 11 else "", "categories": ["T"],
                "description": ["d"], "store": "S"} for k in range(80)])   # a0, a11, ... have no title -> dropped
    revs = []
    for u in range(40):
        t = 0
        for k in rng.sample(range(80), rng.randint(4, 40)):
            t += rng.choice([0, 1])
            revs.append({"user_id": f"U{u}", "parent_asin": f"a{k}", "rating": float(rng.choice([1, 2, 3, 4, 5])),
                         "timestamp": t})
        revs.append({**revs[-1], "timestamp": t + 1, "rating": 1.0})                     # re-review
    revs += [{"user_id": "U0", "parent_asin": "zz", "rating": 5.0, "timestamp": 1},        # unknown item
             {"user_id": "U0", "parent_asin": "a1", "rating": None, "timestamp": 2}]       # no rating
    _write_gz(d / "Toys_and_Games.jsonl.gz", revs)
    items, events = brp.load_amazon(tmp_path / "raw", "toys")
    l_items, l_events = brp.load_amazon(tmp_path / "raw", "toys", light=True)
    assert set(l_items) == set(items) and len(items) == 72 and all(v is None for v in l_items.values())
    assert dict(l_events) == dict(events)
    _, n = build(items, events, n_users=10 ** 9, **{**KW, "n_cands": 20})
    assert brp.eligible_users(l_events, n_cands=20, hist_len=10, **ELIG_KW) == n > 3


@pytest.mark.parametrize("name", sorted(PILOT_SHA1))
def test_writer_reproduces_the_real_pilot1_panel_bytes(name):
    p = PILOT_DIR / f"{name}.jsonl"
    if not p.exists():
        pytest.skip("local copies of the Pilot-1 panels are not available")
    raw = p.read_bytes()
    assert hashlib.sha1(raw).hexdigest() == PILOT_SHA1[name]
    rows = [json.loads(l) for l in raw.decode("utf-8").split("\n")[:-1]]   # not splitlines(): titles hold U+2028
    assert len(rows) == 1500 and all(list(r) == PILOT1_KEYS for r in rows)
    # row_line / pilot_row / truncate_history(10) leave every Pilot-1 row byte-identical
    assert "".join(brp.row_line(brp.pilot_row(brp.truncate_history(r, 10))) for r in rows).encode("utf-8") == raw
    # a row carrying the amendment-2 fields and 5 more (older) history events reduces back to the same line
    for r in [r for r in rows if len(r["history"]) == 10][:50]:
        ext = {}
        for k, v in r.items():
            ext[k] = ([f"older {j}" for j in range(5)] + v) if k in brp.HISTORY_FIELDS else v
            if k == "history_brands":
                ext["history_meta"] = [f"Genres: g{j}" for j in range(15)]
        ext["domain_kind"] = "movie"
        assert len(ext["history"]) == 15 and list(ext)[7] == "history_meta"
        assert brp.row_line(brp.pilot_row(brp.truncate_history(ext, 10))) == brp.row_line(r)


# ------------------------------------------------------------------ frozen reference: the pre-amendment-2 builder
# Verbatim copy of src/confrec/build_rated_panels.py as it wrote the Pilot-1 panels (branch sigir2027 before
# amendment 2); names prefixed _pa2_, argv passed in, and the panel written with "\n" line ends as on the Linux server
# that wrote them. The default CLI of the current builder must write the same bytes (panel, meta, brand_pop).
_PA2_SURROGATE = re.compile("[\ud800-\udfff]")


def _pa2_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(_pa2_text(v) for v in x if v)
    return _PA2_SURROGATE.sub("�", str(x).strip())


def _pa2_load_ml1m(raw: Path):
    items = {}
    for line in open(raw / "movies.dat", encoding="latin-1"):
        mid, title, genres = line.rstrip("\n").split("::")
        items[mid] = {"title": title, "text": "Genres: " + genres.replace("|", ", "), "brand": ""}
    events = defaultdict(list)
    for line in open(raw / "ratings.dat", encoding="latin-1"):
        uid, mid, r, ts = line.rstrip("\n").split("::")
        events[uid].append((int(ts), mid, float(r)))
    return items, events


def _pa2_load_amazon(raw: Path, domain: str):
    cat = CATEGORY[domain]
    d = raw / f"amazon_{domain}"
    items = {}
    with gzip.open(d / f"meta_{cat}.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            title = _pa2_text(m.get("title"))
            if not title:
                continue
            cats = _pa2_text(m.get("categories"))
            desc = _pa2_text(m.get("description"))
            text = (f"Categories: {cats}. " if cats else "") + desc
            items[m["parent_asin"]] = {"title": title, "text": text, "brand": _pa2_text(m.get("store"))}
    events = defaultdict(list)
    with gzip.open(d / f"{cat}.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("parent_asin") in items and r.get("rating") is not None:
                events[r["user_id"]].append((int(r["timestamp"]), r["parent_asin"], float(r["rating"])))
    return items, events


def _pa2_first_per_item(evs: list) -> list:
    seen, out = set(), []
    for e in evs:
        if e[1] not in seen:
            seen.add(e[1])
            out.append(e)
    return out


def _pa2_brand_popularity(items, events) -> dict:
    pop = Counter(i for evs in events.values() for _, i, _ in evs)
    out = Counter()
    for i, n in pop.items():
        b = items[i]["brand"]
        if b:
            out[b] += n
    return dict(sorted(out.items()))


def _pa2_build(items, events, *, n_users, n_cands, hist_len, min_hist, min_like, min_dislike, seed, source,
               stats: dict | None = None):
    pop = Counter(i for evs in events.values() for _, i, _ in evs)
    rows, n_removed = [], 0
    for uid, evs in events.items():
        n_raw = len(evs)
        evs = _pa2_first_per_item(sorted(set(evs)))
        n_removed += n_raw - len(evs)
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
    ts_by_item = defaultdict(list)
    for evs in events.values():
        for t, i, _ in evs:
            if i in need:
                ts_by_item[i].append(t)
    for v in ts_by_item.values():
        v.sort()
    out = []
    for uid, evs, cand_idx, labs, hist in sel:
        order = list(range(len(cand_idx)))
        rng.shuffle(order)
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


def _pa2_main(argv) -> None:
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
    a = ap.parse_args([str(x) for x in argv])
    raw = Path(a.raw)
    items, events = _pa2_load_ml1m(raw) if a.source == "ml1m" else _pa2_load_amazon(raw, a.domain)
    info: dict = {}
    rows, n_eligible = _pa2_build(items, events, n_users=a.n_users, n_cands=a.n_cands, hist_len=a.hist_len,
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
    out.with_suffix(".brand_pop.json").write_text(
        json.dumps(strict_json(_pa2_brand_popularity(items, events)), ensure_ascii=False), encoding="utf-8")
    out.with_suffix(".meta.json").write_text(json.dumps(strict_json(meta), indent=2), encoding="utf-8")
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:   # newline: as on the Linux server
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)


def _fuzz_amazon_raw(raw: Path, domain: str, seed: int):
    """Amazon-2023-like raw files with what the real ones contain: non-ASCII and U+2028 text, lone-surrogate escapes,
    list titles, missing / null fields, untitled items, same-second ties, re-reviews, null ratings, unknown items."""
    r = random.Random(seed)
    words = ["Toy", "Café", "Pokémon Card", "LEGO®", "Bus \ud83d", "Puzzle", "  padded  ", "Set"]
    meta = []
    for k in range(r.randint(40, 90)):
        m = {"parent_asin": f"B{seed}{k:03d}"}
        t = r.random()
        m["title"] = ("" if t < 0.06 else None if t < 0.1 else [r.choice(words), r.choice(words)] if t < 0.2
                      else " ".join(r.sample(words, r.randint(1, 3))))
        if r.random() < 0.8:
            m["categories"] = [r.choice(["Toys & Games", "", "Building Toys", "Café Sets"])
                               for _ in range(r.randint(0, 6))]
        if r.random() < 0.8:
            m["description"] = [r.choice(words), None, "Long " * r.randint(0, 30)]
        if r.random() < 0.85:
            m["store"] = r.choice(["LEGO", "Pokémon", "", None, "Ravensburger®", "  Hasbro "])
        meta.append(m)
    revs = []
    for u in range(r.randint(10, 30)):
        t = 1_600_000_000_000 + r.randint(0, 10 ** 6)
        for _ in range(r.randint(3, 45)):
            t += r.choice([0, 0, 1000, 86_400_000])
            k = r.randrange(len(meta) + 3)
            asin = meta[k]["parent_asin"] if k < len(meta) else f"UNKNOWN{k}"
            rating = None if r.random() < 0.03 else float(r.randint(1, 5))
            revs.append({"user_id": f"U{seed}_{u}", "parent_asin": asin, "rating": rating, "timestamp": t})
    d = raw / f"amazon_{domain}"
    d.mkdir(parents=True, exist_ok=True)
    for name, recs in ((f"meta_{CATEGORY[domain]}.jsonl.gz", meta), (f"{CATEGORY[domain]}.jsonl.gz", revs)):
        with gzip.open(d / name, "wt", encoding="utf-8") as f:
            f.write("".join(json.dumps(x) + "\n" for x in recs))


def _fuzz_ml1m_raw(d: Path, seed: int):
    r = random.Random(seed)
    d.mkdir(parents=True)
    genres = ["Action|Drama", "Comedy", "Children's|Animation", "Sci-Fi|Thriller|War", "Film-Noir"]
    n_movies = r.randint(40, 120)
    (d / "movies.dat").write_text("".join(f"{m}::{r.choice(['Movie', 'Café', 'Él'])} {m} (19{50 + m % 50})::"
                                          f"{r.choice(genres)}\n" for m in range(1, n_movies + 1)), encoding="latin-1")
    lines = []
    for u in range(1, r.randint(8, 30)):
        t = 978_300_000 + u * 1000
        for m in r.choices(range(1, n_movies + 1), k=r.randint(5, 60)):   # repeats exercise the per-item dedup
            t += r.choice([0, 1, 30])
            lines.append(f"{u}::{m}::{r.randint(1, 5)}::{t}\n")
    (d / "ratings.dat").write_text("".join(lines), encoding="latin-1")


@pytest.mark.parametrize("seed", range(6))
def test_default_cli_is_byte_identical_to_the_pre_amendment_builder(tmp_path, monkeypatch, seed):
    raw = tmp_path / "raw"
    _fuzz_ml1m_raw(raw / "ml-1m", seed)
    _fuzz_amazon_raw(raw, "toys", seed)
    cases = [["--source", "ml1m", "--raw", raw / "ml-1m"],
             ["--source", "amazon", "--domain", "toys", "--raw", raw]]
    for k, src in enumerate(cases):
        for extra in ([], ["--n_users", "3"], ["--n_cands", "7", "--hist_len", "4", "--seed", "5"]):
            old, new = tmp_path / f"old{k}" / "p.jsonl", tmp_path / f"new{k}" / "p.jsonl"
            _pa2_main([*src, "--out", old, *extra])
            _run_main(monkeypatch, *src, "--out", new, *extra)
            for suffix in (".jsonl", ".meta.json", ".brand_pop.json"):
                assert old.with_suffix(suffix).read_bytes() == new.with_suffix(suffix).read_bytes(), (k, extra)
            assert old.read_bytes().count(b"\n") == json.loads(new.with_suffix(".meta.json").read_text(
                encoding="utf-8"))["users"]
    assert any(json.loads(p.read_text(encoding="utf-8"))["users"] > 0 for p in tmp_path.glob("new*/p.meta.json"))


def _real_ml1m():
    for d in REAL_ML1M:
        r = d / "ratings.dat"
        if r.is_file() and (d / "movies.dat").is_file() and \
                hashlib.sha1(r.read_bytes()).hexdigest() == ML1M_RATINGS_SHA1:
            return d
    return None


def test_default_cli_rebuilds_the_real_pilot1_ml1m_panel(tmp_path, monkeypatch):
    # exactly the run_pilot1_mirror.sh command: a rebuild must give the registered data_sha1 again
    d = _real_ml1m()
    if d is None:
        pytest.skip("no canonical ML-1M raw copy (ratings.dat sha1 28770e0e...) available")
    out = tmp_path / "ml1m_rated.jsonl"
    _run_main(monkeypatch, "--source", "ml1m", "--raw", d, "--out", out, "--n_users", "1500")
    assert hashlib.sha1(out.read_bytes()).hexdigest() == PILOT_SHA1["ml1m_rated"]
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert list(meta) == PILOT1_META_KEYS and meta["eligible_users"] == 3183 and meta["users"] == 1500
    pm = PILOT_DIR / "ml1m_rated.meta.json"
    if pm.exists():
        assert meta == json.loads(pm.read_text(encoding="utf-8"))