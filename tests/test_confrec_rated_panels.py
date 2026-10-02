import gzip
import json
import random
import sys
from collections import Counter, defaultdict

from src.confrec import build_rated_panels as brp
from src.confrec.build_rated_panels import build

NEW_FIELDS = {"candidate_popularity_prior", "candidate_timestamps"}


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
