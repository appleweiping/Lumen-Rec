import csv
import gzip
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from src.confrec import pilot_mirror as pm
from src.confrec.metrics import risk_coverage

COLS = ["source_event_id", "user_id", "item_id", "cand_idx", "label", "question", "lp_yes", "lp_no", "logit",
        "yes_no_mass", "censored"]


def _row(ev, u, i, c, lab, q, lg, cens=0):
    s = "nan" if lg is None or not np.isfinite(lg) else f"{lg:.6f}"
    return [ev, u, i, c, int(lab), q, 0, 0, s, 1, cens]


def _write_scores(path, rows):
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    return path


def _write_panel(path, panel):
    path.write_text("\n".join(json.dumps(x) for x in panel), encoding="utf-8")
    return path


def _planted(tmp_path, n_users=300, seed=0):
    """Rated panel: popularity-linked yes-saying a(i) shared by like and dislike; valence carries the label."""
    rng = np.random.default_rng(seed)
    n_items = 200
    pop = rng.integers(1, 5000, n_items)
    a_item = 0.8 * (np.log1p(pop) - np.log1p(pop).mean())
    rows, panel, swap, nohist = [], [], [], []
    for u in range(n_users):
        items = rng.choice(n_items, 20, replace=False)
        y = (rng.uniform(size=20) < 0.5).astype(int)
        pref = 1.5 * (2 * y - 1) + rng.normal(0, 1, 20)
        ev = f"u{u}::0"
        panel.append({"source_event_id": ev, "user_id": f"u{u}", "candidate_item_ids": [f"i{i}" for i in items],
                      "candidate_labels": y.tolist(), "candidate_popularity": [int(pop[i]) for i in items],
                      "candidate_popularity_prior": [int(pop[i]) // 2 for i in items]})
        for c, (i, lab) in enumerate(zip(items, y)):
            like = pref[c] + a_item[i] + rng.normal(0, .3)
            dis = -pref[c] + a_item[i] + rng.normal(0, .3)
            para = pref[c] + a_item[i] + rng.normal(0, .3)
            for q, lg in (("like", like), ("dislike", dis), ("like_para", para)):
                rows.append(_row(ev, f"u{u}", f"i{i}", c, lab, q, lg))
            nohist.append(_row(ev, f"u{u}", f"i{i}", c, lab, "like", a_item[i] + rng.normal(0, .1)))
    for i in range(n_items):
        for d in range(8):
            swap.append([f"i{i}", f"d{d}", "like", f"{a_item[i] + rng.normal(0, .3):.6f}", 0])
    # censoring: code 2 / 3 rows are dropped (NaN logit), code 1 rows are kept
    rows[2][8], rows[2][10] = "nan", 2           # u0 cand 0 like_para
    rows[4][8], rows[4][10] = "", 3              # u0 cand 1 dislike
    rows[6][10] = 1                              # u0 cand 2 like (finite, kept)
    sp = _write_scores(tmp_path / "scores.csv.gz", rows)
    nh = _write_scores(tmp_path / "nohist.csv.gz", nohist)
    sw = tmp_path / "swap_prior.csv.gz"
    with gzip.open(sw, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "donor_user_id", "question", "logit", "censored"])
        w.writerows(swap + [["i0", "d9", "like", "nan", 2]])
    return sp, _write_panel(tmp_path / "panel.jsonl", panel), sw, nh, panel


def test_item_sd_corrected_is_zero_under_pure_pair_noise():
    rng = np.random.default_rng(0)
    n_i = rng.choice([1, 2], 20000)
    items = np.repeat(np.arange(20000), n_i)
    noise = rng.normal(0, 0.3, len(items))
    iv = pm.item_variance(noise, items)
    assert iv["SD_item_raw"] > 0.20                  # the old SD of item means passes the 0.20 gate on pure noise
    assert iv["SD_item_corrected"] < 0.08            # method-of-moments removes the pair noise
    assert iv["n_items_ge2"] == int((n_i == 2).sum()) and iv["item_support"]["n1"] == int((n_i == 1).sum())
    eff = rng.normal(0, 0.4, 20000)
    assert abs(pm.item_variance(eff[items] + noise, items)["SD_item_corrected"] - 0.4) < 0.03


def test_rated_planted_acquiescence_censoring_and_arms(tmp_path):
    sp, pp, sw, nh, panel = _planted(tmp_path)
    sc = pm.load_scores(sp)
    res = pm.analyze(sc, panel, swap=pm.load_swap(sw), nohist=pm.load_scores(nh), n_boot=100)
    assert res["panel_type"] == "rated" and res["base_question"] == "like" and res["panel_join"] == {}
    cs = res["censoring"]["scores"]
    assert cs["like_para"]["2"] == 1 and cs["like_para"]["dropped_nonfinite"] == 1
    assert cs["dislike"]["3"] == 1 and cs["dislike"]["dropped_nonfinite"] == 1
    assert cs["like"]["1"] == 1 and "dropped_nonfinite" not in cs["like"]
    assert res["arms"]["raw_like"]["n_rows"] == 6000                     # censored=1 row kept
    assert res["arms"]["mirror"]["n_rows"] == 5999 and res["arms"]["placebo"]["n_rows"] == 5999
    assert res["censoring"]["swap"]["like"]["2"] == 1
    acq = res["acquiescence"]
    assert acq["corr_a_logpop"]["est"] > 0.6 and acq["corr_a_logpop"]["lo"] > 0.5
    assert acq["corr_a_logpop"]["n_clusters"] == 300                     # user-cluster CI
    assert acq["corr_a_logpop_item"]["est"] > 0.8 and acq["corr_a_logpop_item"]["n_clusters"] == 200
    assert acq["SD_pair"] > 0.4 and acq["SD_item_corrected"] > 0.4 and acq["n_items_ge2"] == 200
    assert abs(res["valence"]["corr_v_logpop"]["est"]) < 0.1
    assert res["prior"]["corr_pi_logpop"]["est"] > 0.8
    assert res["popularity_prior_robustness"]["corr_a_logpop"]["est"] > 0.6
    d = res["dUAUC_mirror_minus_placebo"]
    assert d["est"] > 0 and d["lo"] > 0 and d["n"] == 300 and d["n_rows"] == 5998
    for k in ("dUAUC_mirror_minus_raw_like", "dUAUC_evidence_minus_raw_like", "dUAUC_pmi_nohist_minus_raw_like"):
        assert res[k]["lo"] > 0                                          # a(i) removal helps under planted a(i)
    cal = res["calibration"]
    assert set(cal["raw_scale"]) == {"raw", "raw_like", "mirror_half", "placebo"}
    assert set(cal["platt"]) == {"raw", "raw_like", "mirror", "placebo", "evidence", "pmi_nohist"}
    p = cal["platt"]["raw"]
    assert p["n_fit"] + p["n"] == 6000 and 0 < p["n_fit"] < 6000


def test_mirror_half_is_calibrated_when_mirror_is_not(tmp_path):
    # verifier example (statistics#2): calibrated like p=0.8, 80% positives, negation-consistent dislike
    rows, panel = [], []
    l4 = math.log(4)
    for u in range(60):
        ev, y = f"u{u}::0", [1] * 8 + [0] * 2
        panel.append({"source_event_id": ev, "user_id": f"u{u}", "candidate_item_ids": [f"u{u}i{c}" for c in range(10)],
                      "candidate_labels": y})
        for c in range(10):
            for q, lg in (("like", l4), ("dislike", -l4), ("like_para", l4)):
                rows.append(_row(ev, f"u{u}", f"u{u}i{c}", c, y[c], q, lg))
    res = pm.analyze(pm.load_scores(_write_scores(tmp_path / "s.csv.gz", rows)), panel, n_boot=20)
    rs = res["calibration"]["raw_scale"]                                 # logits are stored with 6 decimals
    assert rs["raw"]["ECE"] < 1e-6 and rs["mirror_half"]["ECE"] < 1e-6   # sigmoid(2 ln 4) would give ECE 0.141
    assert abs(rs["mirror_half"]["Brier"] - 0.16) < 1e-6                 # ... and Brier 0.180
    assert "mirror" not in rs


def test_confident_error_target_is_the_like_decision_for_every_detector(tmp_path):
    rng = np.random.default_rng(1)
    rows, panel, swap = [], [], []
    like_err = mirror_self_err = 0
    for u in range(150):
        ev = f"u{u}::0"
        y = np.array([1, 0] * 5)
        ll = 1.2 * (2 * y - 1) + rng.normal(0, 1.2, 10)
        ids = [f"u{u}i{c}" for c in range(10)]
        panel.append({"source_event_id": ev, "user_id": f"u{u}", "candidate_item_ids": ids,
                      "candidate_labels": y.tolist()})
        for c in range(10):
            dis = -ll[c] + 3.0                    # mirror = 2 ll - 3: its own decision differs from like's
            err = int((ll[c] >= 0) != y[c])
            like_err += err
            mirror_self_err += int((2 * ll[c] - 3 >= 0) != y[c])
            for q, lg in (("like", ll[c]), ("dislike", dis), ("like_para", ll[c] + rng.normal(0, .2))):
                rows.append(_row(ev, f"u{u}", ids[c], c, y[c], q, lg))
            # evidence = ll - pi is tiny exactly on the like-decision errors -> a perfect detector
            pi = ll[c] - (0.01 if err else 2.0 + rng.uniform())
            swap.append([ids[c], "d0", "like", f"{pi:.6f}", 0])
    sw = tmp_path / "swap.csv.gz"
    with gzip.open(sw, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "donor_user_id", "question", "logit", "censored"])
        w.writerows(swap)
    res = pm.analyze(pm.load_scores(_write_scores(tmp_path / "s.csv.gz", rows)), panel,
                     swap=pm.load_swap(sw), n_boot=200)
    ce = res["confident_error"]
    assert like_err != mirror_self_err
    assert ce["n_errors"] == like_err
    assert set(ce["detectors"]) == {"mag_raw", "mag_mirror", "mag_evidence", "mag_placebo", "disagree"}
    for d in ce["detectors"].values():
        assert d["n_errors"] == like_err and d["n_rows"] == 1500       # same target, same rows
    ev = ce["detectors"]["mag_evidence"]
    assert ev["AUROC"] == 1.0 and ev["dAUROC_vs_mag_raw"]["lo"] > 0 and ev["dAURC_vs_mag_raw"]["hi"] < 0
    assert ce["detectors"]["mag_raw"]["AUROC"] > 0.5


def test_unit_bootstrap_equals_cluster_bootstrap_on_singleton_clusters():
    from src.confrec.stats import cluster_bootstrap, spearman
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=300), rng.integers(0, 20, 300).astype(float)
    stat = lambda idx: spearman(x[idx], y[idx])  # noqa: E731
    assert pm.unit_bootstrap(stat, 300, n_boot=50, seed=7) == cluster_bootstrap(stat, np.arange(300), 50, 7)


def test_fast_aurc_matches_metrics_risk_coverage():
    rng = np.random.default_rng(2)
    for _ in range(20):
        conf = rng.integers(0, 5, 200).astype(float)        # heavy ties
        util = rng.integers(0, 2, 200).astype(float)
        assert abs(pm._aurc(conf, util) - risk_coverage(conf, util)[2]) < 1e-12


def _next_item(tmp_path, n_users=40):
    rows, panel = [], []
    for u in range(n_users):
        ids = [f"i{k}" for k in range(11)]
        ev = f"u{u}::0"
        panel.append({"source_event_id": ev, "user_id": f"u{u}", "candidate_item_ids": ids, "positive_item_index": 3,
                      "candidate_popularity_groups": ["head"] * 5 + ["tail"] * 6})
        for c in range(11):
            lab = int(c == 3)
            nxt = 5.0 if c in (0, 3) else 0.0    # positive tied with candidate 0 at the top
            for q, lg in (("next", nxt), ("like", 1.0), ("dislike", 1.0), ("like_para", 2.0 if c == 3 else 1.0)):
                rows.append(_row(ev, f"u{u}", ids[c], c, lab, q, lg))
    return pm.load_scores(_write_scores(tmp_path / "s.csv.gz", rows)), panel


def test_next_item_tie_aware_ndcg(tmp_path):
    sc, panel = _next_item(tmp_path)
    res = pm.analyze(sc, panel, base_q="next", n_boot=50)
    assert res["panel_type"] == "next_item" and res["base_question"] == "next"
    raw, mirror, placebo = res["arms"]["raw"], res["arms"]["mirror"], res["arms"]["placebo"]
    assert abs(raw["NDCG@10"] - 1 / math.log2(2.5)) < 1e-12           # expected rank 1.5 (stable sort gave 2)
    assert abs(raw["MRR"] - 1 / 1.5) < 1e-12 and raw["HR@10"] == 1.0
    assert abs(mirror["NDCG@10"] - 1 / math.log2(7)) < 1e-12          # all 11 tied -> expected rank 6 (not 4)
    assert abs(mirror["MRR"] - 1 / 6) < 1e-12
    exact = sum(1 / math.log2(r + 1) for r in range(1, 11)) / 11         # mean over ranks 1..11, rank 11 scores 0
    assert abs(mirror["NDCG@10_tie_exact"] - exact) < 1e-12 and abs(mirror["HR@10_tie_exact"] - 10 / 11) < 1e-12
    assert abs(raw["NDCG@10_tie_exact"] - (1 + 1 / math.log2(3)) / 2) < 1e-12
    assert placebo["NDCG@10"] == 1.0
    assert abs(raw["top10_head_share"] - (2 + 8 * 3 / 9) / 10) < 1e-12  # boundary tie group shared in expectation
    assert abs(mirror["top10_head_share"] - 5 / 11) < 1e-12
    d = res["dNDCG@10_mirror_minus_raw"]
    assert abs(d["est"] - (1 / math.log2(7) - 1 / math.log2(2.5))) < 1e-12 and d["n"] == 40
    assert d["n_events"] == 40 and d["n_events_with_candidates_dropped"] == 0
    assert abs(d["tie_exact"]["est"] - (exact - (1 + 1 / math.log2(3)) / 2)) < 1e-12
    assert res["calibration"]["applicable"] is False and res["confident_error"]["applicable"] is False


def test_next_item_default_base_q_is_next(tmp_path):
    sc, panel = _next_item(tmp_path, n_users=10)
    res = pm.analyze(sc, panel, n_boot=10)                             # like is scored too
    assert res["base_question"] == "next"
    assert abs(res["arms"]["raw"]["NDCG@10"] - 1 / math.log2(2.5)) < 1e-12


def test_tie_exact_metrics_when_the_tie_group_straddles_rank_10(tmp_path):
    rows, panel = [], []
    for u in range(5):
        ids, ev = [f"i{k}" for k in range(12)], f"u{u}::0"
        panel.append({"source_event_id": ev, "user_id": f"u{u}", "candidate_item_ids": ids, "positive_item_index": 9})
        for c in range(12):
            s = 3.0 if c < 9 else 1.0 if c in (9, 10) else 0.0           # positive tied with c=10 at ranks 10-11
            for q, lg in (("next", s), ("like", s), ("dislike", -s)):
                rows.append(_row(ev, f"u{u}", ids[c], c, int(c == 9), q, lg))
    res = pm.analyze(pm.load_scores(_write_scores(tmp_path / "s.csv.gz", rows)), panel, base_q="next", n_boot=10)
    raw = res["arms"]["raw"]
    assert raw["NDCG@10"] == 0.0 and raw["HR@10"] == 0.0 and abs(raw["MRR"] - 1 / 10.5) < 1e-12   # plug-in 10.5
    assert abs(raw["NDCG@10_tie_exact"] - 0.5 / math.log2(11)) < 1e-12 and raw["HR@10_tie_exact"] == 0.5
    assert abs(raw["MRR_tie_exact"] - (1 / 10 + 1 / 11) / 2) < 1e-12
    for arr, pos in ((np.array([2.0, 1, 1, np.nan, 1, 3]), 1), (np.array([np.nan, 1.0, 2.0]), 0)):
        m = pm._event_metrics(arr, pos)                                  # exact HR / plug-in agree off the cutoff
        assert m["HR@10_tie_exact"] == m["HR@10"] == 1.0
    assert pm._event_metrics(np.array([np.nan, 1.0, 2.0]), 0)["MRR_tie_exact"] == 1 / 3   # unscored positive last


def test_paired_ndcg_ranks_both_arms_on_candidates_finite_in_both(tmp_path):
    # 5 negatives score above the positive in every arm, but their dislike is censored (2): mirror alone drops them
    rows, panel = [], []
    for u in range(30):
        ids, ev = [f"i{k}" for k in range(20)], f"u{u}::0"
        panel.append({"source_event_id": ev, "user_id": f"u{u}", "candidate_item_ids": ids, "positive_item_index": 0})
        for c in range(20):
            s = 1.0 if c == 0 else 2.0 if c <= 5 else 0.0
            cens = 1 <= c <= 5 or (u == 0 and c == 0)                    # u0: the positive itself is unscored
            for q, lg, cd in (("next", s, 0), ("like", s, 0), ("like_para", s, 0),
                              ("dislike", None if cens else -s, 2 if cens else 0)):
                rows.append(_row(ev, f"u{u}", ids[c], c, int(c == 0), q, lg, cd))
    res = pm.analyze(pm.load_scores(_write_scores(tmp_path / "s.csv.gz", rows)), panel, base_q="next", n_boot=50)
    mirror, raw = res["arms"]["mirror"], res["arms"]["raw"]
    assert mirror["n_events_negative_unscored"] == 29 and mirror["n_events_positive_unscored"] == 1
    assert abs(raw["NDCG@10"] - 1 / math.log2(7)) < 1e-12
    assert abs(mirror["NDCG@10"] - 29 / 30) < 1e-12                     # own support: censored negatives fall below
    for other in ("raw", "raw_like", "placebo"):
        d = res[f"dNDCG@10_mirror_minus_{other}"]                       # same candidates for both arms -> no gain
        assert abs(d["est"]) < 1e-12 and d["n"] == 29 and d["n_events"] == 29
        assert d["n_events_positive_unscored_dropped"] == 1 and d["n_events_with_candidates_dropped"] == 29
        assert abs(d["tie_exact"]["est"]) < 1e-12


def test_ref_ranks_join_on_source_event_id(tmp_path):
    sc, panel = _next_item(tmp_path)
    ref = {f"u{u}::0": (f"u{u}", 2.0, 11) for u in range(30)}
    ref["ghost::0"] = ("ghost", 1.0, 11)                                # not in the panel
    ref["u30::0"] = ("u30", 1.0, 101)                                   # different candidate set
    ref["u31::0"] = ("WRONG", 1.0, 11)                                  # user mismatch: excluded from the join
    rp = _write_ref(tmp_path / "ccrp_v3.csv.gz", [(ev, u, r, nc) for ev, (u, r, nc) in ref.items()])
    res = pm.analyze(sc, panel, base_q="next", ref_ranks=pm.load_ref_ranks(rp), n_boot=50)
    rr = res["ref_ranks"]
    assert rr["n_joined"] == 30 and rr["ref_events_not_in_scored_panel"] == 1 and rr["num_candidates_mismatch"] == 1
    assert rr["user_id_mismatch"] == 1 and rr["n_ref_rows"] == 33
    assert abs(res["arms"]["ccrp"]["NDCG@10"] - 1 / math.log2(3)) < 1e-12 and res["arms"]["ccrp"]["n_events"] == 30
    assert abs(rr["arms_on_joined_events"]["raw"]["NDCG@10"] - 1 / math.log2(2.5)) < 1e-12
    d = res["dNDCG@10_raw_minus_ccrp"]
    assert d["n"] == 30 and abs(d["est"] - (1 / math.log2(2.5) - 1 / math.log2(3))) < 1e-12
    for arm in ("raw", "raw_like", "mirror", "placebo"):                # P1.4: each arm - C-CRP on the same events
        assert res[f"dNDCG@10_{arm}_minus_ccrp"]["n"] == 30 and res[f"dNDCG@10_{arm}_minus_ccrp"]["n_events"] == 30
    assert abs(res["dNDCG@10_placebo_minus_ccrp"]["est"] - (1 - 1 / math.log2(3))) < 1e-12
    assert abs(res["dNDCG@10_raw_minus_ccrp"]["tie_exact"]["est"] - ((1 + 1 / math.log2(3)) / 2 - 1 / math.log2(3))) \
        < 1e-12


def _write_ref(path, rows):
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_event_id", "user_id", "positive_rank", "num_candidates"])
        for ev, u, r, nc in rows:
            w.writerow([ev, u, f"{r:g}", nc])
    return path


def test_ref_ranks_duplicate_source_event_id_raises(tmp_path):
    rp = _write_ref(tmp_path / "ref.csv.gz", [("u0::0", "u0", 2, 11), ("u1::0", "u1", 3, 11), ("u0::0", "u0", 99, 11)])
    with pytest.raises(ValueError, match="1 duplicated source_event_id"):
        pm.load_ref_ranks(rp)


def test_panel_events_and_candidates_without_score_rows_are_counted(tmp_path):
    sc, panel = _next_item(tmp_path, n_users=10)
    panel = panel + [{"source_event_id": "ghost::0", "user_id": "ghost", "candidate_item_ids": ["a", "b"],
                      "positive_item_index": 0}]
    panel[0] = dict(panel[0], candidate_item_ids=panel[0]["candidate_item_ids"] + ["extra"])
    res = pm.analyze(sc, panel + [panel[1]], base_q="next", n_boot=10)
    assert res["panel_join"] == {"panel_duplicate_event_ids": 1, "panel_events_without_scores": 1,
                                 "panel_candidates_without_score_rows": 1}
    assert res["next_item"]["n_events"] == 10 and res["arms"]["raw"]["n_events_negative_unscored"] == 1


def test_sd_pair_df_removes_the_small_user_bias(tmp_path):
    rng = np.random.default_rng(5)
    rows, panel = [], []
    for u in range(800):
        ev, y = f"u{u}::0", [1, 1, 1, 0, 0, 0]
        panel.append({"source_event_id": ev, "user_id": f"u{u}", "candidate_item_ids": [f"u{u}i{c}" for c in range(6)],
                      "candidate_labels": y})
        a = rng.normal(0, 0.205, 6)                                     # true within-user SD of a: 0.205
        for c in range(6):
            for q, lg in (("like", a[c] + 2 * y[c] - 1), ("dislike", a[c] - 2 * y[c] + 1)):
                rows.append(_row(ev, f"u{u}", f"u{u}i{c}", c, y[c], q, lg))
    acq = pm.analyze(pm.load_scores(_write_scores(tmp_path / "s.csv.gz", rows)), panel, n_boot=30)["acquiescence"]
    assert acq["n_per_user"] == {"min": 6, "median": 6.0, "mean": 6.0, "max": 6} and acq["n_users_ge2"] == 800
    assert abs(acq["SD_pair_df"] - acq["SD_pair"] * math.sqrt(4800 / 4000)) < 1e-12
    assert abs(acq["SD_pair_df"] - 0.205) < 0.008 and abs(acq["SD_pair"] - 0.205 * math.sqrt(5 / 6)) < 0.008
    assert acq["SD_pair_df_ci"]["lo"] < acq["SD_pair_df"] < acq["SD_pair_df_ci"]["hi"]


def test_cli_writes_strict_json(tmp_path):
    sp, pp, sw, nh, _ = _planted(tmp_path, n_users=60, seed=3)
    out = tmp_path / "res.json"
    subprocess.run([sys.executable, "-m", "src.confrec.pilot_mirror", "--scores", str(sp), "--panel", str(pp),
                    "--swap", str(sw), "--nohist", str(nh), "--out", str(out), "--n_boot", "20"],
                   check=True, capture_output=True)

    def bad(c):
        raise ValueError(c)
    res = json.loads(out.read_text(encoding="utf-8"), parse_constant=bad)   # no NaN / Infinity tokens
    for arm in ("raw", "raw_like", "mirror", "placebo", "evidence", "pmi_nohist"):
        assert "UAUC" in res["arms"][arm]
    assert {"est", "lo", "hi", "n"} <= set(res["dUAUC_mirror_minus_placebo"])
    assert res["inputs"]["swap"] == str(sw)
