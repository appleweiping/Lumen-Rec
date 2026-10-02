import json

import numpy as np
import pandas as pd
import pytest

from src.confrec import build_kuairec_panel as bk
from src.confrec import pilot_kuairec as pk
from src.confrec.metrics import bias_index, ece
from src.confrec.stats import rank_bins, sigmoid

SMALL_USERS, BIG_ONLY_USERS = range(8), range(8, 20)
SMALL_VIDEOS, EXTRA_VIDEOS = range(100, 130), range(200, 240)
NO_TITLE = {101, 103, 104, 201}             # NaN/NaN, 'nan'/NaN, absent from the caption file, NaN/'UNKNOWN'


def _kuairec_csvs(tmp_path, overlap=False):
    """Tiny KuaiRec: dense small matrix; big matrix holds NO small pair (as in the release) unless overlap=True."""
    rng = np.random.default_rng(0)
    small = pd.DataFrame([(u, v, float(rng.choice([0.5, 1.2, 2.5, 3.5])), 0.0)
                          for u in SMALL_USERS for v in SMALL_VIDEOS],
                         columns=["user_id", "video_id", "watch_ratio", "timestamp"])
    big = []
    for u in SMALL_USERS:                   # small users' logged rows touch only extra videos
        for k, v in enumerate(EXTRA_VIDEOS):
            if (u + k) % 3:
                big.append((u, v, 2.5 if k % 2 else 0.3, 1000.0 + 10 * k - u))
        big.append((u, 201, 4.0, 9999.0))   # most recent positive, but caption-less
    for u in BIG_ONLY_USERS:                # the other users are the ones exposed to small-matrix videos
        for v in list(SMALL_VIDEOS)[: 12 + u]:
            big.append((u, v, 1.0, 50.0))
        big.append((u, 205, 1.0, 60.0))
    if overlap:
        big.append((0, 105, 3.0, 70.0))
    big = pd.DataFrame(big, columns=["user_id", "video_id", "watch_ratio", "timestamp"])
    caps = []
    for v in [*SMALL_VIDEOS, *EXTRA_VIDEOS]:
        if v == 104:
            continue
        cap, cover = f"cap {v}", np.nan
        tag, c1, c2 = ("[t1,t2]" if v % 2 else np.nan), "Food", ("UNKNOWN" if v % 3 else "Snacks")
        if v in (101, 201):
            cap, cover = np.nan, (np.nan if v == 101 else "UNKNOWN")
        elif v == 102:
            cap, cover, tag = "UNKNOWN", "Cover102", "[]"
        elif v == 103:
            cap = "nan"
        caps.append((v, cover, cap, tag, c1, c2, np.nan))
    cap = pd.DataFrame(caps, columns=["video_id", "manual_cover_text", "caption", "topic_tag",
                                      "first_level_category_name", "second_level_category_name",
                                      "third_level_category_name"])
    small.to_csv(tmp_path / "small_matrix.csv", index=False)
    big.to_csv(tmp_path / "big_matrix.csv", index=False)
    cap.to_csv(tmp_path / "caps.csv", index=False)
    return small, big


def _build(tmp_path, out_name="kuairec.jsonl", overlap=False):
    small, big = _kuairec_csvs(tmp_path, overlap)
    out = tmp_path / out_name
    bk.main(["--root", str(tmp_path), "--caption", str(tmp_path / "caps.csv"), "--out", str(out),
             "--n_users", "5", "--n_cands", "10", "--hist_len", "4", "--min_hist", "1", "--seed", "0"])
    rows = [json.loads(x) for x in open(out, encoding="utf-8")]
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return small, big, rows, meta, out


def test_text_cells_are_nan_safe():
    assert bk.s(np.nan) == "" and bk.s(None) == "" and bk.s(" UNKNOWN ") == "" and bk.s("NaN") == ""
    assert bk.s(" abc ") == "abc" and bk._tags("[a, 'b' ,UNKNOWN]") == "a, b" and bk._tags("[]") == ""
    t = bk.caption_table(pd.DataFrame({"video_id": ["1", "2", "3"], "caption": [np.nan, "nan", "C3"],
                                       "manual_cover_text": ["M1", np.nan, "M3"], "topic_tag": [np.nan] * 3,
                                       "first_level_category_name": ["Food", "UNKNOWN", np.nan]}))
    assert t[1] == ("M1", "Category: Food.") and t[2] == ("", "") and t[3] == ("C3", "")


def test_builder_mar_panel_popularity_history_and_captions(tmp_path):
    small, big, rows, meta, out = _build(tmp_path)
    assert len(rows) == 5 and all(len(r["candidate_item_ids"]) == 10 for r in rows)
    n_v = big.drop_duplicates(["user_id", "video_id"]).groupby("video_id").size()
    n_u = big.groupby("user_id").size()
    wr = small.set_index(["user_id", "video_id"]).watch_ratio
    bigpos = big[big.watch_ratio >= 2.0]
    labels = []
    for r in rows:
        u = int(r["user_id"])
        cands = [int(v) for v in r["candidate_item_ids"]]
        assert "candidate_labels_mnar" not in r
        assert not set(cands) & NO_TITLE and not set(int(v) for v in r["history_item_ids"]) & NO_TITLE
        for t in r["candidate_titles"] + r["history"] + r["candidate_texts"]:
            assert "nan" not in t.casefold() and "UNKNOWN" not in t
        assert all(t for t in r["candidate_titles"] + r["history"])
        for v, t in zip(cands, r["candidate_titles"]):
            assert t == ("Cover102" if v == 102 else f"cap {v}")
        # MAR labels from the small matrix; popularity = distinct users over the UNFILTERED big matrix
        assert r["candidate_watch_ratio"] == [wr[(u, v)] for v in cands]
        assert r["candidate_labels"] == [int(wr[(u, v)] >= 2.0) for v in cands]
        assert r["candidate_popularity"] == [int(n_v.get(v, 0)) for v in cands]
        assert r["user_exposure"] == int(n_u[u])
        # history: last distinct titled positives, chronological, candidates excluded
        mine = bigpos[(bigpos.user_id == u) & ~bigpos.video_id.isin(cands) & ~bigpos.video_id.isin(NO_TITLE)]
        expect = mine.sort_values("timestamp").video_id.tolist()[-4:]
        assert [int(v) for v in r["history_item_ids"]] == expect
        assert r["history"] == [f"cap {v}" for v in expect]
        labels += r["candidate_labels"]
    # small users never touch small videos in big, so a filtered count would be all-zero (statistics#9)
    assert min(x for r in rows for x in r["candidate_popularity"]) > 0
    assert 0 < np.mean(labels) < 1 and meta["mar_pos_rate"] == pytest.approx(np.mean(labels))
    assert meta["shared_pairs"] == 0 and meta["n_caption_missing"] == 3
    on_small = big[big.video_id.isin(SMALL_VIDEOS)].drop_duplicates(["user_id", "video_id"])
    assert meta["target_density"] == pytest.approx(on_small.shape[0] / (big.user_id.nunique() * len(SMALL_VIDEOS)))
    # sensitivity D: only big users outside the small matrix can hold small-video cells
    assert meta["n_small_users_in_big"] == len(SMALL_USERS)
    assert meta["target_density_excl_small_users"] == pytest.approx(
        on_small[on_small.user_id.isin(BIG_ONLY_USERS)].shape[0] / (len(BIG_ONLY_USERS) * len(SMALL_VIDEOS)))
    assert meta["target_density_excl_small_users"] > meta["target_density"]
    first = out.read_text(encoding="utf-8")
    _build(tmp_path)
    assert out.read_text(encoding="utf-8") == first                     # deterministic given the seed


def test_builder_reports_and_drops_shared_pairs(tmp_path):
    _, _, rows, meta, _ = _build(tmp_path, overlap=True)
    assert meta["shared_pairs"] == 1
    assert all("105" not in r["candidate_item_ids"] for r in rows if r["user_id"] == "0")


def test_builder_panel_feeds_pilot_end_to_end(tmp_path):
    _kuairec_csvs(tmp_path)
    out = tmp_path / "kr.jsonl"
    bk.main(["--root", str(tmp_path), "--caption", str(tmp_path / "caps.csv"), "--out", str(out),
             "--n_users", "8", "--n_cands", "20", "--min_hist", "1"])
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(1)
    sc = [(r["source_event_id"], r["user_id"], v, i, y, "like", 0.0, 0.0, 2.0 * y - 1 + rng.normal(0, 0.7), 0.9, 0)
          for line in open(out, encoding="utf-8") for r in [json.loads(line)]
          for i, (v, y) in enumerate(zip(r["candidate_item_ids"], r["candidate_labels"]))]
    scores = tmp_path / "scores.csv.gz"
    pd.DataFrame(sc, columns=["source_event_id", "user_id", "item_id", "cand_idx", "label", "question", "lp_yes",
                              "lp_no", "logit", "yes_no_mass", "censored"]).to_csv(scores, index=False,
                                                                                    compression="gzip")
    res = _run(tmp_path, out, scores)
    assert res["n_pairs"] == 160 and res["target_density"] == meta["target_density"]
    assert abs(res["primary"]["mean_exposure"] - meta["target_density"]) < 1e-9
    assert res["decision"] in {"POSITIVE", "NEGATIVE", "AMBIGUOUS", "NULL"}
    sh = res["group_shares"]
    assert res["endpoint3"]["valid"] == all(0.15 <= sh[g] <= 0.25 for g in ("head", "tail"))


def test_solve_c_hits_target_density_with_clipping():
    rng = np.random.default_rng(0)
    w = np.r_[rng.lognormal(0, 2.5, 5000), np.zeros(200)]
    for target in (0.02, 0.15, 0.6):
        c = pk.solve_c(w, target)
        e = np.minimum(1, c * w)
        assert abs(e.mean() - target) < 1e-9
    assert (np.minimum(1, pk.solve_c(w, 0.6) * w) == 1).any()             # the min(1, .) clip is active
    with pytest.raises(ValueError):
        pk.solve_c(w, 0.99)                                                 # above the share of w > 0


def test_exposure_draws_are_deterministic_per_seed_draw_alpha_thr():
    e = np.full(20000, 0.1)
    a = pk.draw_exposure(e, 0, 3, 1.0, 2.0)
    assert (a == pk.draw_exposure(e, 0, 3, 1.0, 2.0)).all()
    assert (a != pk.draw_exposure(e, 0, 4, 1.0, 2.0)).any() and (a != pk.draw_exposure(e, 0, 3, 0.5, 2.0)).any()
    assert abs(a.mean() - 0.1) < 0.01


def test_weighted_replicas_equal_frozen_metrics_on_the_resample():
    rng = np.random.default_rng(3)
    n, U = 3000, 60
    users = rng.integers(0, U, n)
    logit = np.round(rng.normal(-1, 1.5, n) * 4) / 4                      # heavy ties
    p = sigmoid(logit)
    y = rng.random(n) < p
    g = rng.choice(3, n, p=[0.06, 0.74, 0.20])                             # thin tail -> some cells < min_count
    picks = rng.integers(0, U, (2, U))
    C = np.stack([np.bincount(k, minlength=U) for k in picks]).astype(float)
    lt, nl = pk._tie_index(logit)
    pt, npt = pk._tie_index(p)
    mar = (rng.random(n) < 0.3).astype(float)
    lg = (rng.random(n) < 0.3) & (mar > 0)                                 # logged positives are MAR positives
    eces, bases = pk._ece_reps(C, users, pk._ece_bins(p), p, y), pk._base_reps(C, users, mar, lg)
    ceils = pk._ceiling_reps(C, users, mar, lg)
    for r in range(2):
        idx = np.concatenate([np.flatnonzero(users == k) for k in picks[r]])
        w = C[r][users]
        top = pk._wrank_bins(lt, nl, w, 10) == 9
        assert (top[idx] == (rank_bins(logit[idx], 10) == 9)).all()
        cb = pk._wrank_bins(pt, npt, w, 10)
        assert (cb[idx] == rank_bins(p[idx], 10)).all()
        assert abs(eces[r] - ece(p[idx], y[idx])) < 1e-12
        ref = {pk.GROUPS[k]: v for k, v in bias_index(p[idx], y[idx], g[idx]).items()}
        got = pk._wbias(cb, g, w, p, y)
        assert set(got) == set(ref) and any(np.isfinite(v) for v in ref.values())
        for k in ref:
            assert (np.isnan(ref[k]) and np.isnan(got[k])) or abs(ref[k] - got[k]) < 1e-12
        q = pk.endpoints(p[idx], mar[idx], lg[idx], rank_bins(logit[idx], 10) == 9)
        ti = np.flatnonzero(top)
        assert abs(pk._wshare(w[ti], mar[ti], lg[ti]) - q["share"]) < 1e-12
        assert abs(bases[r] - q["base"]) < 1e-12
        assert abs(ceils[r] - q["gap_ceiling"]) < 1e-12


def test_gap_is_bounded_by_the_unexposed_mar_positive_rate():
    rng = np.random.default_rng(5)
    n = 20000
    mar = (rng.random(n) < 0.3).astype(float)
    lg = (rng.random(n) < 0.4) & (mar > 0)
    top = np.zeros(n, bool)
    ceil = float((mar * ~lg).mean())
    assert ceil == pytest.approx(mar.mean() - lg.mean())
    over = pk.endpoints(np.full(n, 0.9), mar, lg, top)                    # p >= the MAR rate in its bin
    assert over["gap_ceiling"] == pytest.approx(ceil, abs=1e-12) and over["gap"] == pytest.approx(ceil, abs=1e-12)
    assert over["gap_normalized"] == pytest.approx(1.0)
    under = pk.endpoints(np.full(n, lg.mean()), mar, lg, top)             # calibrated to the logged labels
    assert under["gap"] == pytest.approx(-ceil, abs=1e-12) and under["gap_normalized"] == pytest.approx(-1.0)
    for p in (rng.random(n), sigmoid(rng.normal(-1, 2, n) + 2 * mar)):
        q = pk.endpoints(p, mar, lg, top)
        assert abs(q["gap"]) <= q["gap_ceiling"] + 1e-12


def test_decision_rule_branches():
    assert pk.decide(0.57, 0.9, 0.1, 0.2)[0] == "NULL"
    assert pk.decide(0.70, 0.25, 0.01, 0.05, lift_lo=1.3)[0] == "POSITIVE"
    assert pk.decide(0.70, 0.25, -0.05, -0.01, lift_lo=1.3)[0] == "POSITIVE"
    # 2026-10-02 addendum: the near-mechanical gap cannot carry POSITIVE without a lift CI above 1
    assert pk.decide(0.70, 0.25, 0.01, 0.05, lift_lo=0.95)[0] == "AMBIGUOUS"
    assert pk.decide(0.70, 0.25, 0.01, 0.05)[0] == "AMBIGUOUS"  # lift unknown
    assert pk.decide(0.70, 0.05, -0.01, 0.02)[0] == "NEGATIVE"
    assert pk.decide(0.70, 0.05, 0.01, 0.02)[0] == "AMBIGUOUS"
    assert pk.decide(0.70, 0.15, -0.01, 0.02)[0] == "AMBIGUOUS"


def _pilot_files(tmp_path, informative, density=0.15, n_users=120, n_c=50, all_neg=False, user_offsets=False,
                 flat_pop=False):
    rng = np.random.default_rng(7)
    rows, sc = [], []
    nu = rng.lognormal(6, 0.8, n_users).astype(int) + 1
    pop = rng.lognormal(4, 1.5, 4000).astype(int)                          # popularity-skewed exposure
    if flat_pop:
        pop[:] = 7                                                          # every n_v tied
    for u in range(n_users):
        vids = rng.choice(4000, n_c, replace=False)
        z = rng.normal(-1.0, 1.5, n_c)
        y = rng.random(n_c) < sigmoid(z)
        if user_offsets:            # no within-user skill; the user's score offset tracks the user's MAR base rate
            hi = u % 2 == 0
            y = rng.random(n_c) < (0.5 if hi else 0.05)
            z = (1.0 if hi else -1.0) + rng.normal(0, 0.3, n_c)
        wr = np.where(y & (not all_neg), rng.uniform(2, 5, n_c), rng.uniform(0, 1.99, n_c))
        logit = z if informative else rng.normal(-1.0, 1.5, n_c)
        rows.append({"user_id": f"u{u}", "source_event_id": f"u{u}::kuairec", "history": ["h"],
                     "candidate_item_ids": [str(v) for v in vids], "candidate_titles": ["t"] * n_c,
                     "candidate_texts": [""] * n_c, "candidate_labels": [int(x >= 2) for x in wr],
                     "candidate_watch_ratio": wr.tolist(), "candidate_popularity": pop[vids].tolist(),
                     "user_exposure": int(nu[u]), "source": "kuairec"})
        for i in range(n_c):
            cen = int(rng.choice(4, p=[0.9, 0.06, 0.02, 0.02]))
            sc.append((f"u{u}::kuairec", f"u{u}", str(vids[i]), i, int(wr[i] >= 2), "like", 0.0, 0.0,
                       np.nan if cen >= 2 else float(logit[i]), 0.9, cen))
    panel = tmp_path / "kuairec.jsonl"
    panel.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    panel.with_suffix(".meta.json").write_text(json.dumps({"target_density": density, "thr": 2.0}))
    cols = ["source_event_id", "user_id", "item_id", "cand_idx", "label", "question", "lp_yes", "lp_no", "logit",
            "yes_no_mass", "censored"]
    scores = tmp_path / "scores.csv.gz"
    pd.DataFrame(sc, columns=cols).to_csv(scores, index=False, compression="gzip")
    return panel, scores, pd.DataFrame(sc, columns=cols)


def _run(tmp_path, panel, scores, *extra):
    out = tmp_path / "pilot.json"
    pk.main(["--scores", str(scores), "--panel", str(panel), "--out", str(out), "--n_draws", "4",
             "--n_boot", "200", *extra])
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text                    # strict JSON
    return json.loads(text)


def test_pilot_informative_confidence_and_identical_pairs(tmp_path):
    panel, scores, sc = _pilot_files(tmp_path, informative=True)
    res = _run(tmp_path, panel, scores)
    rep = res["scores"]
    cen = sc.censored.value_counts()
    assert rep["censored_counts"] == {str(k): int(cen[k]) for k in sorted(cen.index)}
    assert rep["n_nonfinite_dropped"] == int(cen[2] + cen[3]) and res["n_pairs"] == len(sc) - cen[2] - cen[3]
    assert rep["censored_counts_kept"]["1"] == int(cen[1])                  # imputed bound rows are kept
    prim = res["primary"]
    assert (prim["thr"], prim["alpha"]) == (2.0, 1.0)
    assert abs(prim["mean_exposure"] - 0.15) < 1e-9
    assert 0 < prim["logged_pos_rate"] < prim["mar_pos_rate"]
    # confident logged negatives are mostly unexposed MAR positives
    assert prim["share"]["est"] > 2 * prim["base"]["est"] and prim["lift"]["lo"] > 1
    # gap is ECE(logged) - ECE(MAR) on the same finite pairs and the same p
    fin = sc[np.isfinite(sc.logit)]
    p, mar = sigmoid(fin.logit.to_numpy()), (fin.label.to_numpy() == 1).astype(float)
    assert prim["ECE_MAR"] == pytest.approx(ece(p, mar), abs=1e-12)
    assert prim["gap"]["est"] == pytest.approx(prim["ECE_logged"] - prim["ECE_MAR"], abs=1e-12)
    assert prim["gap"]["lo"] > 0
    pd_ = prim["per_draw"]
    assert all(abs(g) <= c + 1e-12 for g, c in zip(pd_["gap"], pd_["gap_ceiling"]))
    assert prim["gap_ceiling"]["est"] == pytest.approx(prim["mar_pos_rate"] - prim["logged_pos_rate"], abs=1e-12)
    assert -1 <= prim["gap_normalized"]["lo"] <= prim["gap_normalized"]["est"] <= prim["gap_normalized"]["hi"] <= 1
    assert res["caveats"] and res["endpoint3"] == {"valid": True, "issues": [], "share_range": [0.15, 0.25]}
    assert res["gate"]["UAUC_MAR"]["est"] > 0.7
    assert res["decision"] == "POSITIVE"
    cells = {(c["thr"], c["alpha"]) for c in res["sensitivity"]}
    assert cells == {(t, a) for t in (1.0, 2.0, 3.0) for a in (0.5, 1.0, 2.0)}
    assert set(prim["bias_index_MAR"]) == {"head", "mid", "tail"} and "lo" in prim["bias_index_logged"]["head"]


def test_pilot_scores_independent_of_labels_is_not_positive(tmp_path):
    panel, scores, _ = _pilot_files(tmp_path, informative=False)
    res = _run(tmp_path, panel, scores)
    prim = res["primary"]
    assert abs(prim["lift"]["est"] - 1) < 0.2 and prim["lift"]["lo"] < 1 < prim["lift"]["hi"]
    assert res["gate"]["UAUC_MAR"]["est"] < 0.58 and res["decision"] == "NULL"


def test_gate_is_uauc_not_pooled_auc(tmp_path):
    # statistics#8: user-level offsets that track base rates inflate pooled AUC with zero within-user skill
    panel, scores, _ = _pilot_files(tmp_path, informative=True, user_offsets=True)
    res = _run(tmp_path, panel, scores)
    gate = res["gate"]
    assert gate["AUC_MAR_pooled"]["est"] > 0.7 and gate["AUC_MAR_pooled"]["lo"] >= 0.58
    assert gate["UAUC_MAR"]["est"] < 0.58 and gate["UAUC_MAR"]["hi"] < 0.58
    assert res["primary"]["AUC_MAR"] == pytest.approx(gate["AUC_MAR_pooled"]["est"])
    assert res["decision"] == "NULL" and "UAUC_MAR" in res["decision_reason"]


def test_endpoint3_flagged_when_popularity_groups_collapse(tmp_path, capsys):
    panel, scores, _ = _pilot_files(tmp_path, informative=True, flat_pop=True)
    res = _run(tmp_path, panel, scores, "--thrs", "2", "--alphas", "1")
    assert res["group_shares"]["head"] == 0 and res["group_shares"]["tail"] == 0
    e3 = res["endpoint3"]
    assert e3["valid"] is False and len(e3["issues"]) == 2 and "endpoint 3" in capsys.readouterr().out
    prim = res["primary"]
    assert prim["bias_index_MAR"]["head"]["est"] is None and prim["bias_index_MAR"]["mid"]["est"] is not None
    assert res["decision"] == "POSITIVE"                                   # endpoint 3 is not in the decision rule


def test_pilot_aborts_on_degenerate_primary_and_wrong_panel(tmp_path):
    panel, scores, sc = _pilot_files(tmp_path, informative=True, all_neg=True)
    with pytest.raises(ValueError, match="degenerate primary"):
        _run(tmp_path, panel, scores, "--thrs", "2")
    panel, scores, sc = _pilot_files(tmp_path, informative=True)
    sc.loc[sc.index[np.isfinite(sc.logit)][0], "item_id"] = "999999"
    sc.to_csv(scores, index=False, compression="gzip")
    with pytest.raises(ValueError, match="item_id"):
        _run(tmp_path, panel, scores)
