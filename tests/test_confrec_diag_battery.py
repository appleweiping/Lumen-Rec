import csv
import gzip
import hashlib
import itertools
import json
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.confrec import diag_battery as db

REPO = Path(__file__).resolve().parents[1]
PILOT = Path("D:/_Organized/Temp-Review/_RootDirs/temp/claude/D--/22c9b1b2-12a5-4956-a490-79bcb9beb0d8/scratchpad/"
             "pilot_panels")
SCORE_COLS = ["source_event_id", "user_id", "item_id", "cand_idx", "label", "question", "lp_yes", "lp_no", "logit",
              "yes_no_mass", "censored"]


# ---------------------------------------------------------------- fixtures
def _row(uid, items, ratings, ts, hist=(("H1", 5), ("H2", 3), ("H3", 1)), **kw):
    return {"user_id": uid, "source_event_id": f"{uid}::{min(ts)}",
            "history": [f"{t} (rated {r}/5)" for t, r in hist], "history_item_ids": [t for t, _ in hist],
            "history_titles": [t for t, _ in hist], "history_ratings": [float(r) for _, r in hist],
            "candidate_item_ids": list(items), "candidate_titles": [f"Title {i}" for i in items],
            "candidate_texts": [f"Genres: G{i}" for i in items], "candidate_ratings": [float(r) for r in ratings],
            "candidate_labels": [int(r >= 4) for r in ratings], "candidate_timestamps": list(ts), "source": "ml1m",
            **kw}


def _write_ml1m(d: Path, events):
    """events: (user, item, rating, ts) -> d/ratings.dat in ML-1M format."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "ratings.dat").write_text("".join(f"{u}::{i}::{r}::{t}\n" for u, i, r, t in events), encoding="latin-1")


def _write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _write_scores(p: Path, rows, exp=False):
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = SCORE_COLS + (["exp_rating"] if exp else [])
    with gzip.open(p, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])


def _sc(sid, user, item, k, label, logit, censored=0, question="like", exp=None, mass=1.0, lp_yes="nan"):
    r = {"source_event_id": sid, "user_id": user, "item_id": item, "cand_idx": k, "label": label,
         "question": question, "lp_yes": lp_yes, "lp_no": "nan", "logit": logit, "yes_no_mass": mass,
         "censored": censored}
    if exp is not None:
        r["exp_rating"] = exp
    return r


def _auc(s, y):
    """Independent reference AUC: pair counting with ties as 1/2."""
    pos = [a for a, b in zip(s, y) if b == 1]
    neg = [a for a, b in zip(s, y) if b == 0]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


# ---------------------------------------------------------------- make: T1
def test_t1_appends_true_rating_and_keeps_everything_else():
    rec = _row("u1", ["a", "b", "c"], [5, 2, 4], [10, 11, 12])
    out = db.make_t1([rec])
    assert out[0]["candidate_extras"] == ["(This user later rated this item 5/5.)",
                                          "(This user later rated this item 2/5.)",
                                          "(This user later rated this item 4/5.)"]
    assert {k: v for k, v in out[0].items() if k not in ("candidate_extras", "battery")} == rec
    assert "candidate_extras" not in rec   # the source row is not mutated
    with pytest.raises(ValueError):
        db.make_t1([_row("u1", ["a"], [3.5], [1])])
    with pytest.raises(ValueError):
        db.make_t1([{**rec, "candidate_extras": ["x", "", ""]}])


# ---------------------------------------------------------------- make: T2 (prior-only leave-user-out mean)
def test_prior_index_strictly_before_leave_user_out_first_rating():
    by_item = {"a": sorted([(5, 4.0, "v1"), (7, 2.0, "v2"), (10, 5.0, "v3"), (3, 1.0, "u1")])}
    pi = db.PriorIndex(by_item)
    assert pi.query("u", "a", 7) == (2.5, 2)            # ts 3 (u1) and 5 (v1); ts 7 is not strictly before
    assert pi.query("u1", "a", 7) == (4.0, 1)           # u1's own earlier rating is left out
    assert pi.query("v9", "a", 8) == (7 / 3, 3)
    m, n = pi.query("u", "a", 3)
    assert n == 0 and np.isnan(m)
    m, n = pi.query("u", "zzz", 100)
    assert n == 0 and np.isnan(m)


def test_first_ratings_dedups_like_the_panel_builder(tmp_path):
    d = tmp_path / "amazon_toys"
    d.mkdir()
    revs = [{"user_id": "v1", "parent_asin": "a", "rating": 5.0, "timestamp": 20},
            {"user_id": "v1", "parent_asin": "a", "rating": 1.0, "timestamp": 30},   # later re-review: ignored
            {"user_id": "v1", "parent_asin": "a", "rating": 5.0, "timestamp": 20},   # exact repeat
            {"user_id": "v2", "parent_asin": "a", "rating": 2.0, "timestamp": 25},
            {"user_id": "v3", "parent_asin": "b", "rating": 3.0, "timestamp": 1},
            {"user_id": "v4", "parent_asin": "a", "rating": None, "timestamp": 2}]
    with gzip.open(d / "Toys_and_Games.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + "\n" for r in revs))
    by = db.first_ratings(tmp_path, "amazon", "toys", items={"a"})
    assert by == {"a": [(20, 5.0, "v1"), (25, 2.0, "v2")]}
    assert db.PriorIndex(by).query("u", "a", 31) == (3.5, 2)


def test_t2_extras_text_and_exact_values(tmp_path):
    _write_ml1m(tmp_path / "ml", [("v1", "a", 4, 5), ("v2", "a", 3, 6), ("u1", "a", 5, 9), ("v3", "a", 1, 9),
                                  ("v1", "b", 5, 1), ("u1", "b", 2, 8), ("u1", "c", 4, 7)])
    rows = [_row("u1", ["a", "b", "c"], [5, 2, 4], [9, 8, 7])]
    _write_jsonl(tmp_path / "p.jsonl", rows)
    meta = db.make("t2", tmp_path / "p.jsonl", tmp_path / "out" / "t2.jsonl", raw=tmp_path / "ml", source="ml1m")
    r = db.read_jsonl(tmp_path / "out" / "t2.jsonl")[0]
    pre = "Average rating by other users before this date: "
    # a at ts 9: v1 (5) and v2 (6) only -- v3 at the same second is not strictly before; u1 is the user
    assert r["candidate_extras"] == [pre + "3.50/5 (2 ratings)", pre + "5.00/5 (1 ratings)", pre + "no ratings yet"]
    assert r["candidate_prior_mean"] == [3.5, 5.0, None] and r["candidate_prior_n"] == [2, 1, 0]
    assert meta["n_pairs_no_prior_rating"] == 1 and meta["panel_sha1"] == db.file_sha1(tmp_path / "p.jsonl")
    assert json.loads((tmp_path / "out" / "t2.meta.json").read_text())["kind"] == "t2"


# ---------------------------------------------------------------- make: T0
def test_t0_one_row_per_unique_item_with_prior_reference(tmp_path):
    _write_ml1m(tmp_path / "ml", [("v1", "a", 4, 5), ("v2", "a", 2, 6), ("v1", "b", 5, 1), ("u2", "a", 1, 20)])
    rows = [_row("u1", ["a", "b"], [5, 2], [7, 8]), _row("u2", ["c", "a"], [4, 1], [9, 20], domain_kind="movie")]
    by = db.first_ratings(tmp_path / "ml", "ml1m")
    out = db.make_t0(rows, db.PriorIndex(by), by, "amazon")
    assert [r["candidate_item_ids"] for r in out] == [["a"], ["b"], ["c"]]
    a = out[0]
    assert a["history"] == [] and a["history_ratings"] == [] and a["candidate_titles"] == ["Title a"]
    assert a["candidate_texts"] == ["Genres: Ga"] and a["candidate_labels"] == [0]
    assert a["user_id"] == a["source_event_id"] == "t0::a"
    assert a["domain_kind"] == "product"      # first row has no domain_kind: default from --source amazon
    # pairs of a: (u1, ts 7) -> mean(4, 2) = 3; (u2, ts 20) -> v1, v2 -> 3 ; u2's own rating is not prior
    assert a["probe_ref_prior_mean"] == 3.0 and a["probe_ref_n_pairs"] == 2
    assert a["probe_ref_alltime_mean"] == pytest.approx(7 / 3) and a["probe_ref_alltime_n"] == 3
    assert out[2]["probe_ref_prior_mean"] is None and out[2]["domain_kind"] == "movie"


# ---------------------------------------------------------------- make: star permutation
def _derangements(n):
    return [p for p in itertools.permutations(range(n)) if all(p[j] != j for j in range(n))]


def test_random_derangement_has_no_fixed_point_and_is_uniform():
    for n in range(0, 13):
        for s in range(30):
            sigma = db.random_derangement(n, random.Random(f"d:{n}:{s}"))
            assert sorted(sigma) == list(range(n))
            assert n < 2 and sigma == list(range(n)) or all(x != j for j, x in enumerate(sigma))
    # uniform over the 9 derangements of n = 4 (not a fixed shift scheme): 9000 draws, 1000 expected each (sd ~ 32)
    cnt = Counter(tuple(db.random_derangement(4, random.Random(f"u:{s}"))) for s in range(9000))
    assert set(cnt) == set(_derangements(4)) and all(800 < c < 1200 for c in cnt.values()), cnt
    assert db.random_derangement(5, random.Random("x")) == db.random_derangement(5, random.Random("x"))


def test_derangements_reach_two_displayed_sequences():
    """Basis of perm_copies' guarantee: >= 3 events with >= 2 distinct ratings always admit two derangements that
    display different rating sequences (so K = 2 copies can differ); < 3 events or one rating value cannot."""
    for n in range(2, 7):
        der = _derangements(n)
        for vals in itertools.product(range(1, 4), repeat=n):
            shown = {tuple(vals[j] for j in p) for p in der}
            assert (len(shown) >= 2) == (n >= 3 and len(set(vals)) >= 2), vals


def test_perm_copies_are_random_derangements_that_differ():
    rng = random.Random(7)
    for t in range(400):
        n = rng.randint(0, 10)
        vals = [rng.choice([1, 2, 3, 4, 5][: rng.randint(1, 5)]) for _ in range(n)]
        sigmas, dup = db.perm_copies(vals, 2, 0, f"s{t}")
        assert len(sigmas) == 2 and sigmas == db.perm_copies(vals, 2, 0, f"s{t}")[0]   # seeded
        assert db.perm_copies(vals, 1, 0, f"s{t}")[0][0] == sigmas[0]                   # copy 0 does not depend on K
        for s in sigmas:
            assert sorted(s) == list(range(n)) and (n < 2 or all(x != j for j, x in enumerate(s)))
        shown = [tuple(vals[j] for j in s) for s in sigmas]
        if n >= 3 and len(set(vals)) >= 2:
            assert shown[0] != shown[1] and dup == 0                                     # two different prompts
        else:
            assert shown[0] == shown[1] and dup == 1
    # the reviewer's window: a fixed-shift scheme reaches 24 displayed sequences; uniform derangements reach thousands
    w = [5, 5, 5, 5, 4, 4, 4, 3, 3, 1]
    seqs = {tuple(w[j] for j in db.perm_copies(w, 1, s, "row")[0][0]) for s in range(2000)}
    assert len(seqs) > 1000
    # not anti-correlated by construction: over random derangements the mean correlation of the displayed with the
    # original ratings is -1/(n-1)-like, far from the maximum-change scheme's strongly negative value
    cors = [np.corrcoef(w, [w[j] for j in db.perm_copies(w, 1, s, "row")[0][0]])[0, 1] for s in range(2000)]
    assert -0.2 < float(np.mean(cors)) < 0.0


def test_starperm_rows_permute_only_the_rendered_window():
    hist = [(f"Old {k} (1999)", 4) for k in range(3)] + [("Brazil (1985)", 5), ("Dune (1984)", 1), ("X", 3)]
    rec = _row("u1", ["a", "b"], [5, 2], [10, 11], hist=hist)
    out, info = db.make_starperm([rec], k=2, seed=0, hist_len=3)
    assert [r["source_event_id"] for r in out] == ["u1::10::perm0", "u1::10::perm1"]
    for r in out:
        assert r["user_id"] == "u1" and r["perm_of"] == "u1::10"
        assert r["history"][:3] == rec["history"][:3]              # outside the rendered window: untouched
        titles = [h.rsplit(" (rated ", 1)[0] for h in r["history"]]
        assert titles == [t for t, _ in hist]                       # titles stay with their items
        stars = [int(h.rsplit(" (rated ", 1)[1][0]) for h in r["history"]]
        assert stars[3:] != [5, 1, 3] and sorted(stars[3:]) == [1, 3, 5]
        assert [int(x) for x in r["history_ratings"]] == stars      # ratings field follows the suffixes
        assert r["perm_n_changed"] == 3 and r["candidate_labels"] == rec["candidate_labels"]
        assert all(r["perm_sigma"][j] != j for j in range(3))        # no suffix stays on its item
    assert out[0]["history"] != out[1]["history"]                    # the K = 2 copies are two different prompts
    assert info["copies_without_change"] == 0 and info["copies_equal_to_an_earlier_copy"] == 0
    same, info = db.make_starperm([_row("u2", ["a"], [5], [1], hist=[("A", 5), ("B", 5)])], k=2)
    assert all(r["perm_n_changed"] == 0 for r in same) and info["copies_without_change"] == 2
    assert info["copies_equal_to_an_earlier_copy"] == 1
    with pytest.raises(ValueError):   # suffix and history_ratings must agree
        db.make_starperm([{**rec, "history_ratings": [1.0] * 6}])


@pytest.mark.skipif(not (PILOT / "ml1m_rated.jsonl").exists(), reason="local pilot panel copy not available")
def test_battery_panels_on_real_pilot_rows(tmp_path):
    rows = []
    with open(PILOT / "ml1m_rated.jsonl", encoding="utf-8") as f:
        for _ in range(40):
            rows.append(json.loads(f.readline()))
    out, info = db.make_starperm(rows, k=2, seed=0, hist_len=10)
    assert len(out) == 80
    for r, src in zip(out[::2], rows):
        assert sorted(r["history"]) != [] and len(r["history"]) == len(src["history"])
        assert Counter(int(x) for x in r["history_ratings"]) == Counter(int(x) for x in src["history_ratings"])
    for a, b, src in zip(out[::2], out[1::2], rows):
        w = [int(x) for x in src["history_ratings"][-10:]]
        assert (a["history"] != b["history"]) == (len(w) >= 3 and len(set(w)) >= 2)
    assert info["copies_equal_to_an_earlier_copy"] == sum(
        not (len(r["history_ratings"][-10:]) >= 3 and len(set(r["history_ratings"][-10:])) >= 2) for r in rows)
    t1 = db.make_t1(rows)
    assert all(len(r["candidate_extras"]) == len(r["candidate_item_ids"]) for r in t1)


def test_make_cli_writes_panel_and_sidecar(tmp_path):
    _write_ml1m(tmp_path / "ml", [("v1", "a", 4, 1)])
    _write_jsonl(tmp_path / "p.jsonl", [_row("u1", ["a", "b"], [5, 2], [7, 8]), _row("u2", ["a"], [1], [9])])
    for kind in db.KINDS:
        out = tmp_path / "o" / f"{kind}.jsonl"
        db.main(["make", "--kind", kind, "--panel", str(tmp_path / "p.jsonl"), "--raw", str(tmp_path / "ml"),
                 "--source", "ml1m", "--out", str(out), "--n_users", "1"])
        rows = db.read_jsonl(out)
        assert len(rows) == {"t0": 2, "t1": 1, "t2": 1, "starperm": 2}[kind]
        assert json.loads(out.with_suffix(".meta.json").read_text())["n_rows_in"] == 1
    with pytest.raises(SystemExit):
        db.main(["make", "--kind", "t2", "--panel", str(tmp_path / "p.jsonl"), "--source", "ml1m",
                 "--out", str(tmp_path / "x.jsonl")])


# ---------------------------------------------------------------- analyze
def _battery_root(tmp_path, *, perm_shift=0.0, t1_signal=5.0):
    rng = np.random.default_rng(0)
    root = tmp_path / "diag"
    base, t1, t2, t3, perm, llama, t0rows, t0sc = [], [], [], [], [], [], [], []
    t2panel, permpanel, ref = [], [], {}
    truth = {"base": {}, "t1": {}, "t2": {}, "item": {}, "t3": {}, "llama": {}}
    items = [f"i{j}" for j in range(30)]
    for u in range(12):
        uid, sid = f"u{u}", f"u{u}::100"
        labels = [1, 0, 1, 0, 1, 0, 1]
        its = list(rng.choice(items, len(labels), replace=False))
        pm = [float(np.round(3 + y + rng.normal(0, 0.8), 3)) for y in labels]
        pn = [0 if k == 6 else 3 for k in range(len(labels))]
        L = [float(0.4 * y + rng.normal()) for y in labels]
        T1 = [float(t1_signal * y + rng.normal()) for y in labels]
        T2 = [float(0.5 * m + rng.normal(0, 0.7)) for m in pm]
        ER = [float(3 + y + rng.normal(0, 0.5)) for y in labels]
        LL = [float(0.2 * y + rng.normal()) for y in labels]
        for k, (it, y) in enumerate(zip(its, labels)):
            base.append(_sc(sid, uid, it, k, y, L[k]))
            t1.append(_sc(sid, uid, it, k, y, T1[k]))
            t2.append(_sc(sid, uid, it, k, y, T2[k]))
            t3.append(_sc(sid, uid, it, k, y, 0.0, exp=ER[k]))
            llama.append(_sc(sid, uid, it, k, y, LL[k]))
            for kk in range(2):
                perm.append(_sc(f"{sid}::perm{kk}", uid, it, k, y, L[k] - perm_shift * y))
        base.append(_sc(sid, uid, "dropped", 99, 1, "nan", censored=2))
        has = [k for k in range(len(labels)) if pn[k] > 0]
        truth["base"][uid] = _auc(L, labels)
        truth["t1"][uid] = _auc(T1, labels)
        truth["t2"][uid] = _auc([T2[k] for k in has], [labels[k] for k in has])
        truth["item"][uid] = _auc([pm[k] for k in has], [labels[k] for k in has])
        truth["t3"][uid] = _auc(ER, labels)
        truth["llama"][uid] = _auc(LL, labels)
        t2panel.append({"user_id": uid, "source_event_id": sid, "candidate_item_ids": its,
                        "candidate_prior_mean": [m if n else None for m, n in zip(pm, pn)], "candidate_prior_n": pn})
        for kk in range(2):
            permpanel.append({"user_id": uid, "source_event_id": f"{sid}::perm{kk}", "perm_n_changed": 3})
        for it, m in zip(its, pm):
            ref.setdefault(it, []).append(m)
    t0dig = []
    for j, it in enumerate(sorted(ref)):
        m = float(np.mean(ref[it]))
        t0rows.append({"user_id": f"t0::{it}", "source_event_id": f"t0::{it}", "candidate_item_ids": [it],
                       "probe_ref_prior_mean": m, "probe_ref_alltime_mean": m + 0.1})
        # P(Yes) rises with the item mean while the yes/no logit falls (the No mass rises faster): the registered
        # T0 quantity is P(Yes), the logit only a sensitivity
        t0sc.append(_sc(f"t0::{it}", f"t0::{it}", it, 0, 0, 7 - 2 * m, lp_yes=m - 6))
        t0dig.append(_sc(f"t0::{it}", f"t0::{it}", it, 0, 0, 0.0, exp=(-m if j % 2 else -m - 0.01)))
    _write_scores(root / "ml1m" / "t0_digits" / "scores.csv.gz", t0dig, exp=True)
    P = root / "panels"
    _write_jsonl(P / "ml1m_t2.jsonl", t2panel)
    _write_jsonl(P / "ml1m_t0.jsonl", t0rows)
    _write_jsonl(P / "ml1m_starperm.jsonl", permpanel)
    for name, rows in [("base", base), ("t1", t1), ("t2", t2), ("starperm", perm), ("t4_llama", llama),
                       ("t0", t0sc)]:
        _write_scores(root / "ml1m" / name / "scores.csv.gz", rows)
    _write_scores(root / "ml1m" / "t3" / "scores.csv.gz", t3, exp=True)
    return root, truth


def test_analyze_readings_match_independent_reference(tmp_path):
    root, truth = _battery_root(tmp_path)
    res = db.analyze(root, n_boot=200, seed=0)
    m = res["ml1m"]
    assert res["interpretive_only"] and m["T1"]["reading"]["interpretive_only"]
    assert m["base"]["UAUC"] == pytest.approx(np.mean(list(truth["base"].values())))
    assert m["base"]["n_pairs"] == 12 * 7                      # the censored=2 row is dropped
    assert res["arms"]["ml1m/base"]["counts"]["censored_2"] == 12
    assert m["T1"]["UAUC"] == pytest.approx(np.mean(list(truth["t1"].values())))
    assert m["T1"]["reading"]["code"] == "readout_responds"
    pr = m["T2"]["primary"]
    t2, it = np.mean(list(truth["t2"].values())), np.mean(list(truth["item"].values()))
    assert pr["UAUC_T2"]["UAUC"] == pytest.approx(t2) and pr["UAUC_itemmean_prior"]["UAUC"] == pytest.approx(it)
    assert pr["transmission"]["est"] == pytest.approx((t2 - 0.5) / (it - 0.5))
    assert m["T2"]["n_pairs_no_prior_rating"] == 12
    tr = pr["transmission"]["est"]
    assert m["T2"]["reading"]["code"] == ("knowledge_limited" if tr >= 0.8 else "readout_limited" if tr < 0.5
                                          else "between_thresholds")
    d3 = np.mean(list(truth["t3"].values())) - np.mean(list(truth["base"].values()))
    assert m["T3"]["dUAUC_digits_minus_yesno"]["est"] == pytest.approx(d3)
    assert m["T3"]["reading"]["code"] == ("format_bottleneck" if d3 >= 0.03 else "no_format_bottleneck")
    sp = m["star_permutation"]
    assert sp["K"] == 2 and sp["dUAUC_base_minus_perm"]["est"] == 0 and sp["tau_sd_within_user"] == 0
    assert sp["reading"]["code"] == "model_ignores_user_ratings"
    assert m["T4"]["UAUC"] == pytest.approx(np.mean(list(truth["llama"].values())))
    assert m["T4"]["reading"]["code"] == ("at_or_above_0.62_exploratory_only" if m["T4"]["UAUC"] >= 0.62
                                          else "below_0.62")
    assert m["T0"]["spearman_T0_pyes_vs_prior_item_mean"]["est"] == pytest.approx(1.0)
    assert m["T0"]["spearman_T0_logit_vs_prior_item_mean"]["est"] == pytest.approx(-1.0)   # sensitivity only
    assert m["T0"]["reading"]["code"] == "knowledge_exists"                                # read on P(Yes)
    assert res["operationalizations"] == list(db.OPERATIONALIZATIONS)
    # the digit probe is a secondary quantity (anti-monotone here) and never enters the reading
    assert m["T0"]["spearman_T0_digits_exp_rating_vs_prior_item_mean"]["est"] < -0.9
    assert np.isfinite(m["T0"]["like_logit_rho_pairs"]["est"])
    assert res["toys"]["base"]["status"] == "missing" and res["toys"]["T2"]["status"] == "missing"
    assert res["readings"]["ml1m/T1"] == "readout_responds"
    out = tmp_path / "diag.json"
    db.main(["analyze", "--root", str(root), "--out", str(out), "--n_boot", "50"])
    assert json.loads(out.read_text())["interpretive_only"] is True


def test_analyze_flags_broken_readout_and_used_ratings(tmp_path):
    root, _ = _battery_root(tmp_path, perm_shift=3.0, t1_signal=0.0)
    m = db.analyze(root, n_boot=100)["ml1m"]
    assert m["T1"]["reading"]["code"] == "readout_broken_stop_and_debug"
    sp = m["star_permutation"]
    assert sp["dUAUC_base_minus_perm"]["est"] > 0.005       # permuted stars lower the positives' logits here
    assert sp["tau_sd_within_user"] > 0.10 and sp["reading"]["code"] == "condition_not_met"


def test_pooled_within_sd_uses_n_minus_users():
    v, u = [1.0, 3.0, 10.0, 14.0], ["a", "a", "b", "b"]
    assert db.pooled_within_sd(v, u) == pytest.approx(np.sqrt((2 + 8) / 2))
    assert np.isnan(db.pooled_within_sd([1.0, 2.0], ["a", "b"]))   # no within-user df


def test_transmission_undefined_when_item_mean_not_above_half():
    r = db.transmission({"a": 0.7, "b": 0.8}, {"a": 0.5, "b": 0.5}, n_boot=20, seed=0)
    assert np.isnan(r["est"]) and r["n_boot_undefined"] == 20 and r["share_boot_undefined"] == 1.0
    r = db.transmission({"a": 0.7, "b": 0.8}, {"a": 0.8, "b": 0.9}, n_boot=20, seed=0)
    assert r["est"] == pytest.approx((0.75 - 0.5) / (0.85 - 0.5)) and r["n_boot_undefined"] == 0


def test_transmission_counts_undefined_replicates_and_flags_a_denominator_near_half():
    """Item-mean UAUC barely above 0.5: the ratio explodes and many replicates are undefined; both are reported and
    the registered point-estimate reading carries a caveat."""
    rng = np.random.default_rng(3)
    users = [f"u{k}" for k in range(40)]
    item = {u: float(x) for u, x in zip(users, 0.505 + rng.normal(0, 0.1, len(users)))}
    t2 = {u: float(x) for u, x in zip(users, 0.6 + rng.normal(0, 0.1, len(users)))}
    r = db.transmission(t2, item, n_boot=500, seed=0)
    defined = 500 - r["n_boot_undefined"]
    assert 0 < r["n_boot_undefined"] < 500 and r["share_boot_undefined"] == pytest.approx(r["n_boot_undefined"] / 500)
    assert defined > 0 and r["n_boot"] == 500
    # through block_t2: a panel whose prior means carry almost no label signal
    rows, scores = [], []
    for k, u in enumerate(users):
        sid, labels = f"{u}::1", [1, 0, 1, 0]
        pm = [float(3.0 + 0.1 * y + rng.normal(0, 1)) for y in labels]
        rows.append({"user_id": u, "source_event_id": sid, "candidate_prior_mean": pm, "candidate_prior_n": [2] * 4})
        scores += [_sc(sid, u, f"i{j}", j, y, float(y + rng.normal(0, 1))) for j, y in enumerate(labels)]
    t2arm ={"rows": {(s["source_event_id"], s["cand_idx"]): (s["user_id"], s["item_id"], s["label"], s["logit"],
                                                                 db.NAN, db.NAN) for s in scores},
             "counts": {"n": len(scores)}, "mean_yes_no_mass": 1.0, "path": "t2"}
    res = db.block_t2(t2arm, rows, None, n_boot=300, seed=0)
    pr = res["primary"]
    assert pr["denominator_ci_covers_0.5"] is True and "covers 0.5" in res["reading"]["caveat"]
    assert "n_boot_undefined" in pr["transmission"] and res["reading"]["interpretive_only"]


def _arm_report(root, dom, name, data_sha1, n_users=1, n_main=None, **over):
    """report.json as pyes_scorer writes it for this battery arm under run_diag_battery.sh's flags."""
    variant, readout, backbone = db.ARM_SPEC[name]
    rep = {"data_sha1": data_sha1, "variant": variant, "readout": readout, "backbone": backbone,
           "model": f"/root/autodl-tmp/lumen/models/{backbone}", "questions": ["like"], "lora": None,
           "dtype": "float16", "topk_logprobs": 50, "max_model_len": 4096, "panel_kind": "rated",
           "hist_len": db.ARM_HIST[variant], "hist_len_registered": db.ARM_HIST[variant], "n_users": n_users,
           "n_main_prompts": n_main, "config": {"variant": variant, "readout": readout}}
    rep.update(over)
    p = root / dom / name / "report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep))


def test_provenance_flags_an_arm_scored_on_other_panel_bytes(tmp_path):
    root = tmp_path / "diag"
    src = tmp_path / "burned.jsonl"
    _write_jsonl(src, [_row("u1", ["a", "b"], [5, 1], [1, 2])])
    db.make("t1", src, root / "panels" / "ml1m_t1.jsonl", source="ml1m")
    t1sha = db.file_sha1(root / "panels" / "ml1m_t1.jsonl")

    _arm_report(root, "ml1m", "base", db.file_sha1(src))
    _arm_report(root, "ml1m", "t1", t1sha)
    _arm_report(root, "ml1m", "t3", db.file_sha1(src))
    pv = db.provenance(root)
    assert pv["ml1m/t1"]["ok"] is True and pv["ml1m/t3"]["ok"] is True and pv["ml1m/base"]["ok"] is True
    assert pv["ml1m/t2"]["ok"] is None and pv["ml1m/t4_llama"]["ok"] is None   # nothing to check yet
    assert db.analyze(root, n_boot=10)["provenance_ok"] is True
    _arm_report(root, "ml1m", "t1", db.file_sha1(src))   # the T1 arm scored the plain panel, not the T1 panel
    assert db.provenance(root)["ml1m/t1"]["ok"] is False
    _arm_report(root, "ml1m", "t1", t1sha)
    _arm_report(root, "ml1m", "base", db.file_sha1(src), n_users=2)   # base scored other rows than the T1 panel's
    assert db.provenance(root)["ml1m/t1"]["ok"] is False
    _arm_report(root, "ml1m", "base", db.file_sha1(src))
    _arm_report(root, "ml1m", "t3", "0" * 40)            # T3 scored other bytes than base
    assert db.analyze(root, n_boot=10)["provenance_ok"] is False


@pytest.mark.parametrize("name,over,needle", [
    ("t3", {"readout": "yesno"}, "readout"),                         # T3 scored with the yes/no readout
    ("t4_llama", {"backbone": "Qwen3-8B", "model": "/m/Qwen3-8B"}, "backbone"),   # T4 scored with Qwen
    ("t1", {"backbone": "Llama-3.1-8B-Instruct"}, "backbone"),       # a Qwen arm scored with Llama
    ("t0", {"variant": "V0", "hist_len": 10}, "variant"),             # T0 not scored with the probe
    ("t0_digits", {"readout": "yesno"}, "readout"),
    ("base", {"hist_len": 20}, "hist_len"),
    ("starperm", {"lora": "/adapters/x"}, "lora"),
    ("t2", {"max_model_len": 8192}, "max_model_len"),
    ("t1", {"dtype": "bfloat16"}, "dtype"),
    ("base", {"questions": ["like", "dislike"]}, "questions"),
    ("t1", {"topk_logprobs": 20}, "topk_logprobs"),
])
def test_provenance_checks_each_arms_scorer_settings(tmp_path, name, over, needle):
    root = tmp_path / "diag"
    _arm_report(root, "ml1m", name, "a" * 40)
    assert db.provenance(root)[f"ml1m/{name}"]["settings_errors"] == []
    _arm_report(root, "ml1m", name, "a" * 40, **over)
    pv = db.provenance(root)[f"ml1m/{name}"]
    assert pv["ok"] is False and any(e.startswith(needle) for e in pv["settings_errors"]), pv


def test_analyze_flags_report_prompts_not_matching_score_rows(tmp_path):
    root, _ = _battery_root(tmp_path)
    n_t1 = 12 * 7
    _arm_report(root, "ml1m", "t1", "a" * 40, n_users=12, n_main=n_t1)
    res = db.analyze(root, n_boot=20)
    assert res["provenance"]["ml1m/t1"]["settings_errors"] == []
    _arm_report(root, "ml1m", "t1", "a" * 40, n_users=12, n_main=n_t1 - 1)   # stale report next to newer scores
    res = db.analyze(root, n_boot=20)
    assert res["provenance_ok"] is False
    assert "n_main_prompts" in res["provenance"]["ml1m/t1"]["settings_errors"][0]


def test_duplicated_score_rows_are_refused(tmp_path):
    p = tmp_path / "s.csv.gz"
    _write_scores(p, [_sc("s", "u", "i", 0, 1, 0.5), _sc("s", "u", "j", 1, 0, 0.1), _sc("s", "u", "i", 0, 1, 99)])
    with pytest.raises(ValueError, match="duplicated"):
        db.load_arm(p)
    _write_scores(p, [_sc("s", "u", "i", 0, 1, 0.5), _sc("s", "u", "i", 0, 1, 0.7, question="dislike")])
    assert len(db.load_arm(p)["rows"]) == 1          # another question's row is not a duplicate


def test_t3_scored_with_the_yesno_readout_is_not_read_as_digits(tmp_path):
    a, b = tmp_path / "t3.csv.gz", tmp_path / "base.csv.gz"
    rows = [_sc(f"u{u}::1", f"u{u}", f"i{k}", k, k % 2, float(k % 2 + 0.1 * u)) for u in range(4) for k in range(4)]
    _write_scores(a, rows)                           # no exp_rating column: a yes/no run
    _write_scores(b, rows)
    res = db.block_t3(db.load_arm(a), db.load_arm(b), n_boot=10, seed=0)
    assert res["reading"]["code"] == "undetermined" and "exp_rating" in res["reading"]["text"]
    assert "UAUC_digits_logit_45_vs_12" not in res


def _t1_root(tmp_path, signal):
    root = tmp_path / "diag"
    src = tmp_path / "burned.jsonl"
    rows = [_row(f"u{k}", ["a", "b", "c", "d"], [5, 2, 4, 1], [1, 2, 3, 4]) for k in range(6)]
    _write_jsonl(src, rows)
    db.make("t1", src, root / "panels" / "ml1m_t1.jsonl", source="ml1m")
    rng = np.random.default_rng(0)
    sc = [_sc(r["source_event_id"], r["user_id"], it, k, y, float(signal * y + rng.normal(0, 1)))
          for r in rows for k, (it, y) in enumerate(zip(r["candidate_item_ids"], r["candidate_labels"]))]
    _write_scores(root / "ml1m" / "t1" / "scores.csv.gz", sc)
    _arm_report(root, "ml1m", "t1", db.file_sha1(root / "panels" / "ml1m_t1.jsonl"), n_users=6, n_main=len(sc))
    return root


def test_t1check_stops_on_a_broken_readout_before_other_arms(tmp_path):
    root = _t1_root(tmp_path / "ok", signal=20.0)
    res, code = db.t1check(root, n_boot=50)
    assert code == 0 and res["status"] == "readout_responds" and res["T1"]["UAUC"] == 1.0
    root = _t1_root(tmp_path / "broken", signal=0.0)
    res, code = db.t1check(root, n_boot=50)
    assert code == db.T1_BROKEN_EXIT == 5 and res["T1"]["UAUC"] < 0.90
    with pytest.raises(SystemExit) as e:
        db.main(["t1check", "--root", str(root), "--n_boot", "20"])
    assert e.value.code == 5 and json.loads((root / "ml1m" / "t1check.json").read_text())["status"] == \
        "readout_broken_stop_and_debug"
    # the T1 arm must have scored the T1 panel with the registered settings, else the check cannot run
    _arm_report(root, "ml1m", "t1", db.file_sha1(root / "panels" / "ml1m_t1.jsonl"), n_users=6, n_main=24,
                readout="digits")
    assert db.t1check(root, n_boot=20)[1] == 2
    _arm_report(root, "ml1m", "t1", "0" * 40, n_users=6, n_main=24)
    assert db.t1check(root, n_boot=20)[1] == 2
    _arm_report(root, "ml1m", "t1", db.file_sha1(root / "panels" / "ml1m_t1.jsonl"), n_users=6, n_main=23)
    assert db.t1check(root, n_boot=20)[1] == 2
    assert db.t1check(tmp_path / "absent", n_boot=20)[1] == 2


def test_label_mismatch_between_arms_is_an_error(tmp_path):
    a, b = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    _write_scores(a, [_sc("s", "u", "i", 0, 1, 0.5), _sc("s", "u", "j", 1, 0, 0.1)])
    _write_scores(b, [_sc("s", "u", "i", 0, 0, 0.5), _sc("s", "u", "j", 1, 1, 0.1)])
    with pytest.raises(ValueError):
        db.block_t1(db.load_arm(a), db.load_arm(b), n_boot=10, seed=0)


# ---------------------------------------------------------------- freeze
def _fake_render(rec, i, question, variant="V0", panel_kind=None, readout="yesno"):
    system = "SYS" if variant == "V5" else None
    extra = (rec.get("candidate_extras") or [""] * (i + 1))[i]
    last = rec["history"][-1] if rec["history"] else "-"
    return system, f"{variant}|{readout}|{question}|{last}|{rec['candidate_titles'][i]}|{extra}"


def _freeze_inputs(tmp_path):
    am = tmp_path / "AMEND.md"
    am.write_text("binding\n", encoding="utf-8")
    pans = []
    for d in ("ml1m", "toys"):
        p = tmp_path / f"{d}_dev_h20.jsonl"
        _write_jsonl(p, [_row(f"{d}{k}", ["a", "b"], [5, 1], [1, 2]) for k in range(60)])
        pans.append(str(p))
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"ml1m": {"dev": {"n_users": 1500, "user_ids_sha1": "a" * 40},
                                        "confirm": {"user_ids_sha1": "B" * 40, "prefix_check": True}},
                               "toys": {"data_sha1": "c" * 40, "user_id_list_sha1": {"dev": "d" * 40}}}))
    return am, pans, man


def test_user_list_sha1s_walks_the_manifest(tmp_path):
    _, _, man = _freeze_inputs(tmp_path)
    got = db.user_list_sha1s(json.loads(man.read_text()))
    assert got == {"ml1m.dev.user_ids_sha1": "a" * 40, "ml1m.confirm.user_ids_sha1": "b" * 40,
                   "toys.user_id_list_sha1.dev": "d" * 40}


def test_user_list_sha1s_prefers_the_manifest_freeze_block():
    """build_confirm_panels.py writes the four DEV/CONFIRM user-id list sha1s into manifest['freeze']."""
    man = {"sources": {"ml1m": {"dev": {"user_ids_sha1": "a" * 40, "sha1": "e" * 40},
                                "confirm": {"user_ids_sha1": "b" * 40}}},
           "freeze": {"ml1m_dev_users_sha1": "a" * 40, "ml1m_confirm_users_sha1": "b" * 40,
                      "toys_dev_users_sha1": "C" * 40, "toys_confirm_users_sha1": None}}
    assert db.user_list_sha1s(man) == {"freeze.ml1m_dev_users_sha1": "a" * 40,
                                       "freeze.ml1m_confirm_users_sha1": "b" * 40,
                                       "freeze.toys_dev_users_sha1": "c" * 40}


def test_prompt_bank_covers_variants_rows_and_system_message(tmp_path):
    rows = [_row(f"u{k}", ["a", "b"], [5, 1], [1, 2]) for k in range(3)]
    b1 = db.prompt_bank([("p", rows)], ["V0", "V5"], _fake_render, n_rows=2)
    assert b1["n_prompts"] == 2 * 2 * 2
    assert db.prompt_bank([("p", rows)], ["V0", "V5"], _fake_render, n_rows=2) == b1
    assert db.prompt_bank([("p", rows)], ["V0", "V5"], _fake_render, n_rows=3)["sha1"] != b1["sha1"]

    def no_system(rec, i, q, variant="V0", panel_kind=None):
        return None, _fake_render(rec, i, q, variant)[1]
    b2 = db.prompt_bank([("p", rows)], ["V0", "V5"], no_system, n_rows=2)
    assert b2["per_variant"]["V0"] == b1["per_variant"]["V0"] and b2["per_variant"]["V5"] != b1["per_variant"]["V5"]


def test_freeze_writes_then_detects_any_change_and_checks_pilot_log(tmp_path):
    am, pans, man = _freeze_inputs(tmp_path)
    out = tmp_path / "gatefix" / "FREEZE.txt"
    kw = dict(variants=["V0", "V5"], render=_fake_render)
    text = db.freeze(am, pans, man, out, **kw)
    assert out.read_text(encoding="utf-8") == text
    assert f"amendment_sha1 = {db.file_sha1(am)}  REQUIRED" in text and "prompt_bank_n_prompts = 400" in text
    assert db.freeze(am, pans, man, out, **kw) == text                   # unchanged inputs: same freeze
    log = tmp_path / "PILOT_LOG.md"
    log.write_text("nothing yet\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        db.freeze(am, pans, man, out, check=True, pilot_log=log, **kw)    # hashes not recorded yet
    req = [ln.split(" = ")[1].split()[0] for ln in text.splitlines() if ln.endswith("REQUIRED")]
    assert len(req) == 6   # amendment, G1 prompt bank, battery bank, 3 user-id lists
    assert any(ln.startswith("battery_bank_sha1 = ") and ln.endswith("REQUIRED") for ln in text.splitlines())
    log.write_text("recorded: " + " ".join(s.upper() for s in req) + "\n", encoding="utf-8")
    db.freeze(am, pans, man, out, check=True, pilot_log=log, **kw)
    with pytest.raises(SystemExit):
        db.freeze(am, pans, man, out, check=True, pilot_log=log, variants=["V0"], render=_fake_render)
    am.write_text("binding, edited\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        db.freeze(am, pans, man, out, **kw)                               # the amendment changed after the freeze
    with pytest.raises(SystemExit):
        db.freeze(am, pans, man, tmp_path / "absent.txt", check=True, pilot_log=log, **kw)


def test_battery_bank_freezes_the_battery_prompts_and_rules(tmp_path, monkeypatch):
    rows = [_row(f"u{k}", ["a", "b", "c"], [5, 2, 4], [10, 11, 12]) for k in range(4)]
    b0 = db.battery_bank([("p", rows)], _fake_render, n_rows=3)
    assert b0["n_prompts"] == 2 * 3 + 3 * 9 + 2 * 9   # T0 x 2 readouts on 3 unique items; T1/T2/T3 9 each; starperm
    assert set(b0["per_arm"]) == {"T0", "T0_digits", "T1", "T2", "T3", "starperm"}
    assert db.battery_bank([("p", rows)], _fake_render, n_rows=3) == b0
    for attr, val in [("T1_EXTRA", "(The user rated this item {r}/5.)"), ("T2_PREFIX", "Mean rating: "),
                      ("OPERATIONALIZATIONS", db.OPERATIONALIZATIONS[:-1]), ("STARPERM_SEED", 1),
                      ("THRESHOLDS", {**db.THRESHOLDS, "T1": {"readout_broken_uauc_lt": 0.8}})]:
        with monkeypatch.context() as mp:
            mp.setattr(db, attr, val)
            assert db.battery_bank([("p", rows)], _fake_render, n_rows=3)["sha1"] != b0["sha1"], attr
    # the permutation scheme itself is frozen: a different suffix permutation changes the starperm prompts
    with monkeypatch.context() as mp:
        mp.setattr(db, "random_derangement", lambda n, rng: list(range(1, n)) + [0] if n > 1 else list(range(n)))
        new = db.battery_bank([("p", rows)], _fake_render, n_rows=3)["per_arm"]
        assert new["starperm"] != b0["per_arm"]["starperm"] and new["T1"] == b0["per_arm"]["T1"]


def test_freeze_needs_user_id_sha1s(tmp_path):
    am, pans, man = _freeze_inputs(tmp_path)
    man.write_text(json.dumps({"ml1m": {"data_sha1": "a" * 40}}))
    with pytest.raises(SystemExit):
        db.freeze(am, pans, man, tmp_path / "F.txt", variants=["V0"], render=_fake_render)


# ---------------------------------------------------------------- vstar
def test_selected_variant():
    assert db.selected_variant({"v_star": "V3", "fix_found": True}) == "V3"
    # the shape gatefix_select dev writes: v_star at top level and inside `selection`, a per-variant table
    real = {"stage": "dev", "decision": "FIX_FOUND", "outcome": "FIX_FOUND", "fix_found": True, "v_star": "V5",
            "gate_ft_prompt": "V5", "selection": {"v_star": "V5", "eligible": ["V0", "V5"], "tied": ["V5"]},
            "table": {"V0": {"ml1m": {"UAUC": 0.58}}, "V5": {"ml1m": {"UAUC": 0.61}}}, "bootstrap": {"est": 0.02}}
    assert db.selected_variant(real) == "V5"
    with pytest.raises(SystemExit):   # F0 under the same shape: V* recorded, but no fix
        db.selected_variant({**real, "decision": "GATE_FAIL_AFTER_REMEDY(dev)", "fix_found": False})
    assert db.selected_variant({"selection": {"V_star": "V7"}}) == "V7"
    for bad in ({"v_star": "V0"}, {"v_star": "V3", "fix_found": False}, {"v_star": "V6"}, {"x": 1},
                {"v_star": "V3", "selection": {"selected": "V4"}}):
        with pytest.raises(SystemExit):
            db.selected_variant(bad)


# ---------------------------------------------------------------- integration: real prompt bank and scorer
def _h20_row(uid, n_hist=14):
    hist = [(f"Movie {k} (19{70 + k})", 1 + (k * 3) % 5) for k in range(n_hist)]
    r = _row(uid, ["a", "b", "c"], [5, 2, 4], [100, 101, 102], hist=hist)
    r.update(history_meta=[f"Genres: G{k}" for k in range(n_hist)], domain_kind="movie")
    return r


def test_freeze_default_bank_renders_every_real_variant(tmp_path):
    prompting = pytest.importorskip("src.confrec.prompting")
    if not hasattr(prompting, "VARIANTS") or not hasattr(prompting, "render"):
        pytest.skip("prompting has no variant bank yet")
    am, _, man = _freeze_inputs(tmp_path)
    pans = []
    for d in ("ml1m", "toys"):
        p = tmp_path / f"{d}_dev_h20.jsonl"
        _write_jsonl(p, [_h20_row(f"{d}{k}") for k in range(55)])
        pans.append(str(p))
    text, _ = db.freeze_text(am, pans, man)
    keys = list(prompting.VARIANTS)
    assert {"V0", "V1", "V2", "V3", "V4", "V5", "V7", "T0_probe"} <= set(keys)
    assert f"variants={','.join(keys)};" in text and f"prompt_bank_n_prompts = {2 * len(keys) * 50 * 3}" in text
    h = hashlib.sha1()   # the V0 entries are exactly the registered Pilot-1 prompt strings
    for p in pans:
        for rec in db.read_jsonl(p)[:50]:
            for i in range(3):
                s = prompting.build_prompt(rec["history"], rec["candidate_titles"][i], rec["candidate_texts"][i],
                                           "like", 10)
                h.update(f"{Path(p).name}\x1fV0\x1f{rec['source_event_id']}\x1f{i}\x1f{s}".encode("utf-8") + b"\x1e")
    assert f"prompt_bank_sha1[V0] = {h.hexdigest()}" in text
    # the battery bank renders with the real bank too (digit readout on V0 / T0_probe, extras, permuted histories):
    # per panel 3 unique items x 2 readouts + 3 arms x 50 rows x 3 candidates + 2 copies x 50 x 3
    assert f"battery_bank_n_prompts = {2 * (3 * 2 + 3 * 150 + 2 * 150)}" in text
    if hasattr(prompting, "PROMPT_STRINGS_SHA1"):
        assert f"prompt_strings_sha1 = {prompting.PROMPT_STRINGS_SHA1}" in text


def test_real_render_of_battery_rows():
    prompting = pytest.importorskip("src.confrec.prompting")
    if not hasattr(prompting, "render"):
        pytest.skip("prompting has no render yet")
    rec = _row("u1", ["a", "b"], [5, 2], [10, 11])
    plain = prompting.build_prompt(rec["history"], "Title a", "Genres: Ga", "like", 10)
    _, t1 = prompting.render(db.make_t1([rec])[0], 0, "like", variant="V0")
    assert t1 == plain.replace("Description: Genres: Ga\n",
                               "Description: Genres: Ga\n(This user later rated this item 5/5.)\n")
    sp = db.make_starperm([rec], k=1)[0][0]
    _, u = prompting.render(sp, 0, "like", variant="V0")
    assert u == prompting.build_prompt(sp["history"], "Title a", "Genres: Ga", "like", 10) and u != plain
    t0 = db.make_t0([rec], db.PriorIndex({}), {}, "ml1m")[0]
    system, u = prompting.render(t0, 0, "like", variant="T0_probe")
    assert system is None and u.startswith("Movie: Title a\nDescription: Genres: Ga\n\n")
    assert "Is this movie widely considered good?" in u


# (panel, scorer flags) exactly as scripts/sigir/run_diag_battery.sh passes them (plus the fixed G0 flags)
V0_FLAGS = ["--variant", "V0", "--readout", "yesno", "--questions", "like"]
BATTERY_ARMS = [("base", V0_FLAGS), ("t0", ["--variant", "T0_probe", "--readout", "yesno", "--questions", "like"]),
                ("t0", ["--variant", "T0_probe", "--readout", "digits", "--questions", "like"]),
                ("t1", V0_FLAGS), ("t2", V0_FLAGS), ("base", ["--variant", "V0", "--readout", "digits",
                                                              "--questions", "like"]), ("starperm", V0_FLAGS)]


def test_real_scorer_accepts_every_battery_panel_with_the_script_flags(tmp_path):
    ps = pytest.importorskip("src.confrec.pyes_scorer")
    script = SCRIPTS[1].read_text(encoding="utf-8")
    for _, flags in BATTERY_ARMS:
        assert " ".join(flags[:4]) in script or flags == V0_FLAGS
    _write_ml1m(tmp_path / "ml", [("v1", "a", 4, 5), ("v2", "b", 2, 6)])
    _write_jsonl(tmp_path / "base.jsonl", [_row(f"u{k}", ["a", "b", "c"], [5, 2, 4], [10, 11, 12]) for k in range(4)])
    for kind in db.KINDS:
        db.make(kind, tmp_path / "base.jsonl", tmp_path / f"{kind}.jsonl", raw=tmp_path / "ml", source="ml1m")
    try:
        ps.parse_args(["--data", "x", "--output", "y", "--model", "m", "--dry_run"])
    except SystemExit:
        pytest.skip("pyes_scorer has no --dry_run yet")
    for name, flags in BATTERY_ARMS:
        out = ps.run(ps.parse_args(["--data", str(tmp_path / f"{name}.jsonl"), "--output", str(tmp_path / "o"),
                                    "--model", "Qwen3-8B", "--dtype", "float16", "--topk_logprobs", "50",
                                    "--max_model_len", "4096", "--chunk_users", "100", "--dry_run", *flags]))
        assert out["panel_kind"] == "rated", name
        assert out["config"]["hist_len"] == (0 if name == "t0" else 10), name
        assert out["config"]["max_model_len"] == 4096


# ---------------------------------------------------------------- run scripts
SCRIPTS = [REPO / "scripts" / "sigir" / "run_gatefix.sh", REPO / "scripts" / "sigir" / "run_diag_battery.sh"]


def _bash():
    for c in ("C:/Program Files/Git/bin/bash.exe", shutil.which("bash")):
        if c and Path(c).exists() and "System32" not in str(c):
            return c
    return None


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_run_script_is_lf_and_bash_n_clean(script):
    raw = script.read_bytes()
    assert b"\r\n" not in raw and raw.startswith(b"#!/usr/bin/env bash\n")
    assert b"set -euo pipefail" in raw
    bash = _bash()
    if bash is None:
        pytest.skip("no bash")
    r = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_gatefix_never_scores_anything_but_like_on_dev():
    text = SCRIPTS[0].read_text(encoding="utf-8")
    code = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())   # comments may name the forbidden arms
    for banned in ("dislike", "like_para", "swap_k", "hist_len 0", "--questions next"):
        assert banned not in code, banned
    assert code.count("--questions") == 1 and "--questions like" in code
    assert "FREEZE_ACK" in code and "--check" in code and "gatefix_select dev" in code
    assert "gatefix_select confirm" in code


def test_diag_battery_script_checks_freeze_and_uses_separate_root():
    code = "\n".join(ln.split("#", 1)[0] for ln in SCRIPTS[1].read_text(encoding="utf-8").splitlines())
    assert "freeze" in code and "--check" in code and "outputs/confrec/diag" in code
    assert "gatefix/dev" not in code and "confirm" not in code.replace("ml1m_confirm", "")
    for kind in db.KINDS:
        assert f"--kind {kind}" in code
    assert "T0_probe" in code and "--readout digits" in code and "Llama-3.1-8B-Instruct" in code


def test_diag_battery_script_scores_t1_and_checks_it_before_any_other_arm():
    code = "\n".join(ln.split("#", 1)[0] for ln in SCRIPTS[1].read_text(encoding="utf-8").splitlines())
    calls = [ln for ln in code.splitlines() if ln.startswith("score ")]
    assert calls[0].startswith('score "$MODEL" "$DP/ml1m_t1.jsonl" "$D/ml1m/t1"'), calls[0]
    check = code.index("diag_battery t1check")
    assert code.index(calls[0]) < check < min(code.index(c) for c in calls[1:])
    assert "exit 5" in code[check:code.index(calls[1])] and "T1_OVERRIDE" in code[check:code.index(calls[1])]
