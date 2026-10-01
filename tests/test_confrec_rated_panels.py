from collections import defaultdict

from src.confrec.build_rated_panels import build


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
    rows, n_eligible = build(items, events, n_users=10, n_cands=6, hist_len=10, min_hist=3,
                             min_like=3, min_dislike=3, seed=0, source="toy")
    assert n_eligible == 1 and len(rows) == 1
    r = rows[0]
    assert sorted(r["candidate_ratings"]) == [1.0, 2.0, 2.0, 4.0, 5.0, 5.0]   # 3-star dropped
    assert all(l == int(x >= 4) for l, x in zip(r["candidate_labels"], r["candidate_ratings"]))
    assert sum(r["candidate_labels"]) == 3
    cand = set(r["candidate_item_ids"])
    assert not cand & set(r["history_item_ids"])                              # no leakage
    assert r["history_item_ids"] == ["i0", "i1", "i2", "i3"]                  # strictly before candidates
    assert r["history"][1].endswith("(rated 3/5)")
