import csv
import gzip
import json
import math

import numpy as np
import pytest

from src.confrec import forensics as fx
from src.confrec.metrics import auroc
from src.confrec.stats import sigmoid, spearman, strict_json

COLS = ["source_event_id", "user_id", "item_id", "cand_idx", "label", "question", "lp_yes", "lp_no", "logit",
        "yes_no_mass", "censored"]
SWAP_COLS = ["item_id", "donor_user_id", "question", "logit", "censored"]


def _gz(path, header, rows):
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def _row(ev, u, i, c, lab, q, lg, cens=0):
    s = "nan" if lg is None or not np.isfinite(lg) else f"{lg:.6f}"
    return [ev, u, i, c, int(lab), q, 0, 0, s, 1, cens]


def _rated_rows(n_users, n_cands, rng, items=None):
    """Minimal rated panel rows (both classes per user); items default to distinct ids per pair."""
    rows = []
    for u in range(n_users):
        y = np.zeros(n_cands, int)
        y[: n_cands // 2] = 1
        rng.shuffle(y)
        ids = items[u] if items is not None else [f"i{u}_{c}" for c in range(n_cands)]
        rows.append({"user_id": f"u{u}", "source_event_id": f"u{u}::0", "history_ratings": [4.0, 2.0],
                     "candidate_item_ids": list(ids), "candidate_labels": y.tolist()})
    return rows


# ---------------------------------------------------------------- loaders
def test_load_scores_drop_rules_match_pilot_mirror(tmp_path):
    p = _gz(tmp_path / "s.csv.gz", COLS, [
        _row("e1", "u1", "a", 0, 1, "like", 1.0),
        _row("e1", "u1", "a", 0, 1, "dislike", None, 2),
        _row("e1", "u1", "b", 1, 0, "like", 2.0, 3),        # finite but censored 3 -> dropped
        _row("e1", "u1", "b", 1, 0, "dislike", 0.5, 1),     # censored 1 -> kept
        _row("e1", "u1", "a", 0, 1, "like", 3.0)])          # duplicate: last wins
    sc = fx.load_scores(p)
    assert sc["L"]["like"] == {("e1", 0): 3.0}
    assert sc["L"]["dislike"] == {("e1", 1): 0.5}
    assert sc["censoring"]["like"]["duplicate"] == 1
    assert sc["censoring"]["like"]["dropped_nonfinite"] == 1
    assert sc["meta"][("e1", 1)] == ("u1", "b", 0)


def test_swap_positions_keep_file_order_and_censored_slots(tmp_path):
    rows = [["i0", f"d{k}", "like", f"{k:.1f}", 0] for k in range(8)]
    rows[2] = ["i0", "d2", "like", "nan", 2]
    rows.insert(3, ["i0", "dx", "dislike", "9.0", 0])       # other question: ignored
    sw = fx.load_swap_positions(_gz(tmp_path / "sw.csv.gz", SWAP_COLS, rows))
    v = sw["pos"]["i0"]
    assert len(v) == 8 and math.isnan(v[2]) and v[:2] == [0.0, 1.0] and v[3:] == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert sw["censoring"]["dropped_nonfinite"] == 1


def test_attach_excludes_item_mismatch_and_counts_label_mismatch(tmp_path):
    P = fx.panel_pairs([{"user_id": "u", "source_event_id": "e", "history_ratings": [5],
                         "candidate_item_ids": ["a", "b", "c"], "candidate_labels": [1, 0, 1]}])
    sc = fx.load_scores(_gz(tmp_path / "s.csv.gz", COLS, [
        _row("e", "u", "a", 0, 1, "like", 1.0), _row("e", "u", "WRONG", 1, 0, "like", 2.0),
        _row("e", "u", "c", 2, 0, "like", 3.0), _row("zz", "u", "a", 0, 1, "like", 4.0)]))
    L, join = fx.attach(P, sc)
    assert L["like"][0] == 1.0 and math.isnan(L["like"][1]) and L["like"][2] == 3.0
    assert join == {"score_rows_item_mismatch": 1, "score_rows_label_mismatch": 1,
                    "score_rows_without_panel_pair": 1}
    assert P["kind"] == "rated"


# ---------------------------------------------------------------- A2
def _swap_panel(n_users=200, n_cands=20, seed=0):
    rng = np.random.default_rng(seed)
    P = fx.panel_pairs(_rated_rows(n_users, n_cands, rng))
    return P, fx._groups(P["user"]), rng


def test_pi_split_half_reliability_recovers_planted_value():
    P, g, rng = _swap_panel()
    items = sorted(set(P["item"].tolist()))
    pi = rng.normal(0, 1, len(items))
    # each donor = pi + N(0, 2): a 4-donor mean has reliability 1 / (1 + 4/4) = 0.5 (Pearson);
    # bivariate-normal Spearman = (6/pi) asin(0.5/2) = 0.4826
    swap = {i: list(p + rng.normal(0, 2, 8)) for i, p in zip(items, pi)}
    out, pi_p, pi_item = fx.prior_checks(P, g, None, swap, None, n_boot=50)
    r = out["split_half"]["spearman_r"]["est"]
    assert abs(r - 6 / math.pi * math.asin(0.25)) < 0.04
    assert out["split_half"]["spearman_brown_8"] == pytest.approx(2 * r / (1 + r))
    assert out["split_half"]["n_items"] == len(items)
    assert pi_item[items[0]] == pytest.approx(np.mean(swap[items[0]]))


def test_pi_split_half_uses_donor_positions_1_4_vs_5_8():
    P, g, rng = _swap_panel(n_users=100)
    items = sorted(set(P["item"].tolist()))
    pi = rng.normal(0, 1, len(items))
    # donors 1-4 carry +pi, donors 5-8 carry -pi: only the file-order halves give r = -1
    swap = {i: list(p + rng.normal(0, .05, 4)) + list(-p + rng.normal(0, .05, 4)) for i, p in zip(items, pi)}
    out, _, _ = fx.prior_checks(P, g, None, swap, None, n_boot=20)
    assert out["split_half"]["spearman_r"]["est"] < -0.95


def test_evidence_uaucs_and_projection_formula():
    P, g, rng = _swap_panel(n_users=150)
    items = sorted(set(P["item"].tolist()))
    pi = dict(zip(items, rng.normal(0, 1, len(items))))
    swap = {i: list(pi[i] + rng.normal(0, 1, 8)) for i in items}
    like = np.array([pi[i] for i in P["item"]]) + 0.8 * P["label"] + rng.normal(0, 1, P["n"])
    nohist = np.array([pi[i] for i in P["item"]])
    out, pi_p, _ = fx.prior_checks(P, g, like, swap, nohist, n_boot=20)
    ev = out["evidence"]
    manual = np.mean(list(fx.uauc(like - pi_p, P["label"], g).values()))
    assert ev["UAUC_8_donor"]["est"] == pytest.approx(manual)
    r = out["split_half"]["spearman_r"]["est"]
    u8, u4 = ev["UAUC_8_donor"]["est"], ev["UAUC_4_donor_mean"]
    assert ev["projection_reliable_pi"] == pytest.approx(u8 + (u8 - u4) / r)
    assert ev["projection_clipped"] is False and r >= fx.PROJECTION_MIN_R
    lo, hi = out["split_half"]["spearman_r"]["lo"], out["split_half"]["spearman_r"]["hi"]
    assert ev["projection_at_r_ci"]["r_hi"] == pytest.approx(min(max(u8 + (u8 - u4) / hi, 0), 1))
    assert ev["projection_at_r_ci"]["r_lo"] == pytest.approx(min(max(u8 + (u8 - u4) / lo, 0), 1)) \
        if lo >= fx.PROJECTION_MIN_R else math.isnan(ev["projection_at_r_ci"]["r_lo"])
    assert out["pi"]["UAUC_pi"]["n_users"] == 150
    assert out["L_nohist"]["UAUC_L_nohist"]["est"] == pytest.approx(out["pi"]["UAUC_pi"]["est"], abs=0.05)


def test_projection_is_withheld_for_a_small_r_and_clipped_to_the_unit_interval():
    p = fx.projection_reliable(0.55, 0.53, 0.5)
    assert p["value"] == pytest.approx(0.59) and p["clipped"] is False and p["note"] is None
    small = fx.projection_reliable(0.55, 0.53, 0.05)               # would multiply the gap by 20
    assert math.isnan(small["value"]) and "not reported" in small["note"]
    big = fx.projection_reliable(0.9, 0.6, 0.25)                   # 0.9 + 0.3 / 0.25 = 2.1
    assert big["value"] == 1.0 and big["unclipped"] == pytest.approx(2.1) and big["clipped"] is True
    assert math.isnan(fx.projection_reliable(0.55, 0.53, float("nan"))["value"])


# ---------------------------------------------------------------- A3
def _toy_events():
    return {
        "u1": [(10, "A", 5.0), (20, "B", 4.0)],          # panel user: candidates (u1, A @10) and (u1, B @20)
        "u2": [(5, "B", 2.0), (21, "B", 1.0)],           # re-rating of B dropped (first event per item)
        "u3": [(20, "B", 5.0)],                          # same second as the candidate: not strictly before
        "u4": [(15, "B", 3.0), (30, "C", 1.0)],
        "u5": [(25, "B", 1.0)],                          # after the candidate: all-time reference only
        "u6": [(1, "A", 1.0)],
    }


def test_prior_item_mean_is_strictly_before_and_leave_user_out():
    scan = fx.scan_events(_toy_events(), {"A", "B"}, {("u1", "A"), ("u1", "B")})
    pm = fx.prior_means(scan, ["u1", "u1"], ["B", "A"], [20.0, 10.0], shrink_k=2.0)
    # B @20: other users' first ratings strictly before 20 = u2 (2), u4 (3)
    assert pm["mean_prior"][0] == 2.5 and pm["n_prior"][0] == 2
    # global before 20 without u1's own ratings: u2 B2, u4 B3, u6 A1 -> 2.0; shrunk (5 + 2*2)/(2 + 2)
    assert pm["global_prior"][0] == pytest.approx(2.0)
    assert pm["mean_prior_shrunk"][0] == pytest.approx(2.25)
    assert pm["mean_loo_alltime"][0] == pytest.approx((2 + 5 + 3 + 1) / 4) and pm["n_loo_alltime"][0] == 4
    # A @10: u6 (1) only; global before 10 without u1: u2 B2, u6 A1 -> 1.5; shrunk (1 + 3)/(1 + 2)
    assert pm["mean_prior"][1] == 1.0 and pm["global_prior"][1] == pytest.approx(1.5)
    assert pm["mean_prior_shrunk"][1] == pytest.approx(4 / 3)
    # MF training: every first rating except the panel user's candidate events (9 raw, 8 first, 2 held out)
    assert scan["n_raw_events"] == 9 and scan["n_first_events"] == 8 and len(scan["mf_r"]) == 6
    assert "u1" not in scan["uid"] and scan["own"][("u1", "B")] == (20.0, 4.0)


def _mf_data(personal: bool, seed=0, n_u=1500, n_i=200, n_per=50, n_hold=10):
    rng = np.random.default_rng(seed)
    b_i, b_u = rng.normal(0, 0.7, n_i), rng.normal(0, 0.5, n_u)
    A, C = rng.normal(0, 1, (n_u, 3)), rng.normal(0, 1, (n_i, 3))
    tu, ti, tr, hu, hi, hr = [], [], [], [], [], []
    for u in range(n_u):
        its = rng.choice(n_i, n_per, replace=False)
        r = 3.5 + b_u[u] + b_i[its] + (A[u] @ C[its].T if personal else 0) + rng.normal(0, 0.6, n_per)
        tu += [u] * (n_per - n_hold)
        ti += its[:-n_hold].tolist()
        tr += r[:-n_hold].tolist()
        hu += [u] * n_hold
        hi += its[-n_hold:].tolist()
        hr += r[-n_hold:].tolist()
    hu, hi, hr = np.array(hu), np.array(hi), np.array(hr)
    lab = np.zeros(len(hr), int)
    for u in range(n_u):
        m = hu == u
        lab[m] = hr[m] > np.median(hr[m])
    return (np.array(tu), np.array(ti), np.array(tr)), (hu, hi, lab), n_u, n_i


@pytest.mark.parametrize("personal", [False, True])
def test_mf_personal_residual_null_without_and_detected_with_personal_signal(personal):
    (tu, ti, tr), (hu, hi, lab), n_u, n_i = _mf_data(personal)
    mf = fx.fit_biased_mf(tu, ti, tr, n_u, n_i, dim=8, iters=10, seed=0)
    s = fx.mf_pair_scores(mf, hu, hi)
    g = fx._groups(hu)
    res = np.mean(list(fx.uauc(s["residual"], lab, g).values()))
    item = np.mean(list(fx.uauc(s["item_bias"], lab, g).values()))
    full = np.mean(list(fx.uauc(s["score"], lab, g).values()))
    if personal:
        assert res > 0.65 and full > item
    else:
        assert abs(res - 0.5) < 0.02     # no personal signal -> the residual ranks at chance
        assert item > 0.65
    assert mf["train_rmse"] < 0.75


def _canon(x):
    return json.dumps(strict_json(x), sort_keys=True)


def test_temporal_mf_cutoff_training_set_and_blindness_to_later_ratings(tmp_path):
    from src.confrec.build_rated_panels import first_per_item
    from src.confrec.split_panel import time_split
    _, _, raw, panel = _ml1m_fixture(tmp_path)
    P = fx.panel_pairs(panel)
    g = fx._groups(P["user"])
    events = fx.load_raw_events(raw, "ml1m")
    kw = dict(mf_dim=4, mf_iters=3, n_boot=20)
    out, _ = fx.cf_references(P, g, events, **kw)
    tm = out["mf_temporal"]
    T = time_split(panel, 0.8)[0]                                  # amendment 1 D convention (split_panel)
    assert tm["cutoff"]["T"] == T == fx.mf_cutoff(P["ts"])
    cand = set(zip(P["user"].tolist(), P["item"].tolist()))
    pre = [(u, i) for u, evs in events.items() for t, i, _ in first_per_item(sorted(set(evs)))
           if t < T and (u, i) not in cand]
    assert tm["n_train"] == len(pre) and tm["n_users_train"] == len({u for u, _ in pre})
    test = P["ts"] >= T
    tp = tm["test_pairs_t_ge_T"]
    assert tp["n_pairs"] == int(test.sum()) > 0 and tp["UAUC_personal_residual_incl_cold"]["n_pairs"] == tp["n_pairs"]
    assert tp["UAUC_personal_residual"]["n_pairs"] == tp["n_pairs_warm"] <= tp["n_pairs"]
    assert tm["all_pairs_sensitivity"]["n_pairs"] == P["n"]
    # reversing every non-candidate rating at or after T leaves the temporal MF untouched (it never sees them) and
    # moves the transductive reference (it trains on them)
    flipped = {u: [(t, i, 6.0 - r if t >= T and (u, i) not in cand else r) for t, i, r in evs]
               for u, evs in events.items()}
    out2, _ = fx.cf_references(P, g, flipped, **kw)
    assert _canon(out2["mf_temporal"]) == _canon(tm)
    assert _canon(out2["mf_transductive_reference"]["UAUC_mf"]) != _canon(out["mf_transductive_reference"]["UAUC_mf"])


def test_fold_in_user_recovers_planted_user_parameters():
    rng = np.random.default_rng(0)
    Q = rng.normal(0, 1, (3, 40))
    mf = {"mu": 3.5, "bi": rng.normal(0, 0.5, 40), "bu": np.zeros(1), "P": np.zeros((3, 1)), "Q": Q}
    items = np.arange(40)
    b_u, p_u = 0.7, np.array([0.5, -1.0, 0.25])
    r = mf["mu"] + mf["bi"] + b_u + p_u @ Q + rng.normal(0, 0.01, 40)
    b, p = fx.fold_in_user(mf, items, r, lam=1e-6, lam_bias=1e-6)
    assert b == pytest.approx(b_u, abs=0.01) and np.allclose(p, p_u, atol=0.01)
    assert fx.fold_in_user(mf, [], [], 0.05, 5.0)[0] == 0.0
    Q0 = {**mf, "Q": np.zeros((3, 40))}                            # items without factors inform b_u only
    b0, p0 = fx.fold_in_user(Q0, items, r, lam=0.05, lam_bias=0.0)
    assert np.all(p0 == 0) and b0 == pytest.approx(np.mean(r - mf["mu"] - mf["bi"]))


def _cohort_world(seed=0, n_items=200, dim=2):
    """Items learnt from 'old' users before T; panel cohort C (history and candidates early, mostly before T) and
    cohort B (joined after T: no rating before T at all). Ratings carry a strong personal term A_u.C_i."""
    rng = np.random.default_rng(seed)
    b_i, C = rng.normal(0, 0.3, n_items), rng.normal(0, 1, (n_items, dim))
    events, rows = {}, []

    def user(u, hist_win, cand_win, n_hist, n_cand):
        a = rng.normal(0, 1, dim)
        n_ev = n_hist + n_cand
        its = rng.choice(n_items, n_ev, replace=False)
        ts = np.concatenate([np.sort(rng.uniform(*hist_win, n_hist)), np.sort(rng.uniform(*cand_win, n_cand))])
        rc = 3 + b_i[its] + 1.2 * (C[its] @ a) + rng.normal(0, 0.3, n_ev)
        r = np.clip(np.round(rc), 1, 5)
        events[u] = [(float(t), str(i), float(x)) for t, i, x in zip(ts, its, r)]
        if n_cand:
            k = slice(n_hist, n_ev)
            lab = (rc[k] > np.median(rc[k])).astype(int)
            rows.append({"user_id": u, "source_event_id": f"{u}::0", "history_ratings": r[:n_hist].tolist(),
                         "candidate_item_ids": [str(i) for i in its[k]], "candidate_labels": lab.tolist(),
                         "candidate_ratings": r[k].tolist(), "candidate_timestamps": ts[k].tolist()})

    for k in range(1500):                                         # non-panel users, all before T
        user(f"d{k}", (0, 40), (0, 0), 40, 0)
    for k in range(450):                                          # cohort C: history [0, 40), candidates [40, 60)
        user(f"c{k}", (0, 40), (40, 60), 30, 10)
    for k in range(100):                                          # cohort B: joined after T
        user(f"b{k}", (100, 150), (150, 160), 30, 10)
    return events, rows


def test_temporal_mf_fold_in_gives_post_cutoff_users_a_profile():
    events, rows = _cohort_world()
    P = fx.panel_pairs(rows)
    g = fx._groups(P["user"])
    out, _ = fx.cf_references(P, g, events, mf_dim=2, mf_iters=8, mf_lambda=0.02, mf_lambda_bias=1.0, n_boot=20)
    tm = out["mf_temporal"]
    assert 40 < tm["cutoff"]["T"] < 60
    b_pairs = int(np.char.startswith(P["user"].astype(str), "b").sum())
    fold, nofold = tm["test_pairs_t_ge_T"], tm["test_pairs_t_ge_T_no_fold_in"]
    # cohort B has no rating before T: cold in the ts < T fit, warm after the fold-in
    assert nofold["n_pairs_cold_user"] >= b_pairs and fold["n_pairs_cold_user"] == 0
    assert tm["fold_in"]["n_users_warm"] == 550 and tm["fold_in"]["n_users_without_events"] == 0
    # the planted personal term is found on the post-cutoff pairs once B has a profile; counting cold users (residual
    # identically 0 -> AUC 0.5) drags the residual UAUC toward 0.5
    assert fold["UAUC_personal_residual"]["est"] > 0.7
    assert fold["UAUC_personal_residual"]["n_users"] > nofold["UAUC_personal_residual"]["n_users"] + 90
    assert nofold["UAUC_personal_residual_incl_cold"]["est"] < fold["UAUC_personal_residual"]["est"] - 0.1


def test_mf_cold_items_and_users_get_zero_factors():
    mf = fx.fit_biased_mf([0, 0, 1], [0, 1, 0], [4.0, 2.0, 5.0], n_users=3, n_items=3, dim=4, iters=3)
    assert np.all(mf["Q"][:, 2] == 0) and mf["bi"][2] == 0 and np.all(mf["P"][:, 2] == 0)
    s = fx.mf_pair_scores(mf, [0, -1], [-1, 0])
    assert s["residual"].tolist() == [0.0, 0.0] and s["item_bias"][0] == 0.0


# ---------------------------------------------------------------- A4
def test_binormal_prediction_matches_simulated_sum():
    rng = np.random.default_rng(0)
    n, dx, dz, rho = 200_000, 0.5, 0.3, 0.4
    y = (rng.random(n) < 0.4).astype(int)
    e = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    X, Z = dx * y + e[:, 0], dz * y + e[:, 1]
    ax, az = auroc(X, y), auroc(Z, y)
    r, sx, sz = fx.within_class_corr(X, Z, y), fx.within_class_sd(X, y), fx.within_class_sd(Z, y)
    assert abs(r - rho) < 0.01 and abs(sx - 1) < 0.01
    assert abs(fx.binormal_ensemble_auc(ax, az, r) - auroc(X + Z, y)) < 0.004
    X3 = 3 * X  # raw (unstandardised) weights
    assert abs(fx.binormal_weighted_auc(ax, az, r, 3 * sx, sz) - auroc(X3 + Z, y)) < 0.004
    assert fx.binormal_ensemble_auc(0.6, 0.6, 1.0) == pytest.approx(0.6)   # a duplicated view adds nothing
    assert fx.binormal_ensemble_auc(0.6, 0.6, -1.0) == 1.0


def test_binormal_pooled_reproduces_deliberation_arithmetic():
    # judge.md / lens_skeptic.md A7 from the Pilot-1 ML-1M diag_baselines: MIRROR 0.6085, placebo 0.597
    like, negdis, para = 0.5874475521977115, 1 - 0.40787767045273177, 0.603966160030964
    assert fx.binormal_sum_auc(like, negdis, 0.3554729678005053) == pytest.approx(0.6085, abs=0.001)
    assert fx.binormal_sum_auc(like, para, 0.9529020444443472) == pytest.approx(0.597, abs=0.001)


def _gauss_two_view(seed=1, n_users=500, n_c=60):
    rng = np.random.default_rng(seed)
    P = fx.panel_pairs(_rated_rows(n_users, n_c, rng))
    y = P["label"]
    s = 0.6 * y + rng.normal(0, 1, P["n"])
    like = s + rng.normal(0, 1, P["n"])
    dislike = -(0.8 * s + rng.normal(0, 1.2, P["n"]))      # a less-correlated second view, nothing else
    para = like + rng.normal(0, 0.3, P["n"])
    return P, fx._groups(P["user"]), like, dislike, para


def test_ensemble_null_is_zero_under_a_gaussian_two_view_null():
    P, g, like, dislike, para = _gauss_two_view()
    y = P["label"]
    out = fx.ensemble_null(P, g, like, dislike, para, n_boot=200)
    # the per-user Phi^-1 transform leaves a small (~0.001-0.002) Jensen bias, so only the point estimate is pinned
    assert abs(out["dUAUC_mirror_minus_ensemble_null"]["est"]) < 0.004
    assert abs(out["dUAUC_placebo_minus_ensemble_null"]["est"]) < 0.004
    assert abs(out["gap_residual"]["est"]) < 0.005
    assert abs(out["mirror_pair"]["sensitivity_raw_weight"]["dUAUC_observed_minus_predicted"]["est"]) < 0.005
    m = np.mean([auroc((like - dislike)[i], y[i]) for i in g.values()])
    assert out["mirror_pair"]["observed_UAUC"] == pytest.approx(m)
    # MIRROR beats the placebo here through two-view diversity alone, and the null predicts that gap
    obs_gap = out["mirror_pair"]["observed_UAUC"] - out["placebo_pair"]["observed_UAUC"]
    pred_gap = out["mirror_pair"]["predicted_UAUC"] - out["placebo_pair"]["predicted_UAUC"]
    assert obs_gap > 0.008 and abs(obs_gap - pred_gap) < 0.003
    assert abs(out["pooled_back_of_envelope"]["residual"]) < 0.01


def test_ensemble_null_departs_from_zero_for_non_gaussian_views():
    # sparse heavy-tailed acquiescence shared by like and dislike: MIRROR cancels it, but the binormal null built from
    # the per-user AUCs and Pearson within-class correlation mispredicts the sum -> MIRROR - null is far from 0
    rng = np.random.default_rng(1)
    P = fx.panel_pairs(_rated_rows(400, 40, rng))
    g = fx._groups(P["user"])
    s = 0.6 * P["label"] + rng.normal(0, 1, P["n"])
    a = np.where(rng.random(P["n"]) < 0.15, rng.normal(0, 8, P["n"]), 0.0)
    like = s + a + rng.normal(0, 0.5, P["n"])
    dislike = -s + a + rng.normal(0, 0.5, P["n"])
    out = fx.ensemble_null(P, g, like, dislike, like + rng.normal(0, 0.3, P["n"]), n_boot=200)
    d = out["dUAUC_mirror_minus_ensemble_null"]
    assert abs(d["est"]) > 0.05 and not (d["lo"] <= 0 <= d["hi"])
    assert abs(out["dUAUC_placebo_minus_ensemble_null"]["est"]) < 0.01   # the placebo pair stays on its null
    f = fx.wording_flags({"panels": {"x": {"A4_ensemble_null": out}}})["x"]
    assert f["sensitivity"]["ensemble_null_residual_ci_covers_0_two_sided"] is False
    assert f["ensemble_null_residual_lo_le_0"] is (d["lo"] <= 0)


def test_ensemble_null_matches_the_decision_group_definition():
    pm = pytest.importorskip("src.confrec.pilot_mirror")
    if not hasattr(pm, "ensemble_null_pair"):
        pytest.skip("pilot_mirror has no ensemble_null_pair yet")
    P, g, like, dislike, para = _gauss_two_view(seed=3, n_users=120, n_c=30)
    like[5] = np.nan                                        # a missing row only drops from the pairs that use it
    out = fx.ensemble_null(P, g, like, dislike, para, n_boot=100, seed=7)
    boot = dict(n_boot=100, seed=7)
    for name, (x2, arm) in {"mirror_pair": (-dislike, like - dislike),
                            "placebo_pair": (para, (like + para) / 2)}.items():
        ref = pm.ensemble_null_pair(like, x2, arm, P["label"], g, np.isfinite(arm), boot)
        mine = out[name]
        for k in ("predicted_UAUC", "observed_UAUC", "z_sum_UAUC", "UAUC_view1", "UAUC_view2", "n_users"):
            assert mine[k] == pytest.approx(ref[k], rel=1e-12), (name, k)
        for k in ("est", "lo", "hi"):
            assert mine["dUAUC_observed_minus_predicted"][k] == pytest.approx(
                ref["dUAUC_observed_minus_predicted"][k], rel=1e-12), (name, k)


def test_ensemble_null_predicts_the_gain_of_an_independent_second_view():
    # like and -dislike are independent views; like_para nearly duplicates like: the null predicts MIRROR > placebo
    rng = np.random.default_rng(2)
    P = fx.panel_pairs(_rated_rows(400, 40, rng))
    g = fx._groups(P["user"])
    y = P["label"]
    like = 0.5 * y + rng.normal(0, 1, P["n"])
    dislike = -(0.5 * y + rng.normal(0, 1, P["n"]))
    out = fx.ensemble_null(P, g, like, dislike, like + rng.normal(0, 0.3, P["n"]), n_boot=100)
    assert out["mirror_pair"]["predicted_UAUC"] > out["placebo_pair"]["predicted_UAUC"] + 0.03
    assert abs(out["mirror_pair"]["rho_within_class"]["mean"]) < 0.05
    assert abs(out["dUAUC_mirror_minus_ensemble_null"]["est"]) < 0.01


def test_null_bias_sim_vectorised_helpers_match_the_scalar_definitions():
    rng = np.random.default_rng(0)
    y = np.array([1] * 13 + [0] * 7)
    rng.shuffle(y)
    v1, v2 = rng.normal(size=(6, 20)), rng.normal(size=(6, 20)) + 0.5 * y
    a, c = fx._rowwise_auc(v1, y == 1), fx._rowwise_within_class_corr(v1, v2, y)
    for k in range(6):
        assert a[k] == pytest.approx(auroc(v1[k], y), abs=1e-12)
        assert c[k] == pytest.approx(fx.within_class_corr(v1[k], v2[k], y), abs=1e-12)


def _exact_ensemble(n_users, n_c, n_like, d, rho, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        y = np.zeros(n_c, int)
        y[:n_like] = 1
        rng.shuffle(y)
        rows.append({"user_id": f"u{u}", "source_event_id": f"u{u}::0", "history_ratings": [4.0],
                     "candidate_item_ids": [f"i{u}_{c}" for c in range(n_c)], "candidate_labels": y.tolist()})
    P = fx.panel_pairs(rows)
    e = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], P["n"])
    return P, fx._groups(P["user"]), d * P["label"] + e[:, 0], d * P["label"] + e[:, 1]


def test_null_bias_sim_zero_for_a_duplicated_view_negative_at_pilot_size_and_vanishing_with_many_candidates():
    P, g, x1, x2 = _exact_ensemble(300, 20, 13, 0.32, 0.35, seed=0)
    y, mask = P["label"], np.ones(P["n"], bool)
    dup = fx.ensemble_pair(x1, x1.copy(), 2 * x1, y, g, mask)      # rho = 1: the null is exact
    b = np.array(list(fx.null_bias_sim(dup, y, g, mask, reps=20, seed=0).values()))
    # exact up to rounding; only a draw whose AUC hits 0 or 1 (clipped half a pair) differs, by e / reps
    assert len(b) == 300 and np.median(np.abs(b)) < 1e-12 and abs(b.mean()) < 1e-4
    # an exact Gaussian two-view ensemble at the Pilot-1 ML-1M design (20 candidates, 13 likes, AUC ~0.59, rho 0.35):
    # the per-user plug-in null over-predicts, i.e. a negative bias of about 0.001-0.002
    P, g, x1, x2 = _exact_ensemble(600, 20, 13, 0.32, 0.35, seed=1)
    y, mask = P["label"], np.ones(P["n"], bool)
    r = fx.ensemble_pair(x1, x2, x1 + x2, y, g, mask)
    small = np.mean(list(fx.null_bias_sim(r, y, g, mask, reps=20, seed=0).values()))
    assert -0.004 < small < -0.0005
    P, g, x1, x2 = _exact_ensemble(100, 400, 260, 0.32, 0.35, seed=2)
    y, mask = P["label"], np.ones(P["n"], bool)
    r = fx.ensemble_pair(x1, x2, x1 + x2, y, g, mask)
    big = np.mean(list(fx.null_bias_sim(r, y, g, mask, reps=20, seed=0).values()))
    assert abs(big) < 0.0005 and abs(big) < abs(small) / 3


def test_ensemble_null_reports_the_bias_corrected_residual():
    P, g, like, negdis = _exact_ensemble(300, 20, 13, 0.32, 0.35, seed=3)
    para = like + np.random.default_rng(0).normal(0, 0.3, P["n"])
    out = fx.ensemble_null(P, g, like, -negdis, para, n_boot=50, null_sim_reps=10)
    nb = out["mirror_pair"]["null_bias_parametric"]
    assert nb["n_users"] == out["mirror_pair"]["n_users"] and nb["mean_bias"] < 0
    bc, raw = out["dUAUC_mirror_minus_ensemble_null_bias_corrected"], out["dUAUC_mirror_minus_ensemble_null"]
    assert bc["est"] == pytest.approx(raw["est"] - nb["mean_bias"])
    assert len(out["caveats"]) == len(fx.ENSEMBLE_NULL_CAVEATS)
    assert "null_bias_parametric" not in fx.ensemble_null(P, g, like, -negdis, para, n_boot=20,
                                                          null_sim_reps=0)["mirror_pair"]


# ---------------------------------------------------------------- A5
def test_intercept_and_temperature_fits_recover_planted_values():
    rng = np.random.default_rng(0)
    L = rng.normal(0, 2, 50_000)
    y1 = (rng.random(len(L)) < sigmoid(L - 1.5)).astype(int)
    y2 = (rng.random(len(L)) < sigmoid(L / 4)).astype(int)
    assert fx.fit_intercept(L, y1) == pytest.approx(-1.5, abs=0.05)
    assert fx.fit_slope(L, y2) == pytest.approx(0.25, abs=0.02)


def test_calibration_decomposition_user_centring_removes_user_offsets():
    rng = np.random.default_rng(0)
    P = fx.panel_pairs(_rated_rows(400, 20, rng))
    g = fx._groups(P["user"])
    s = rng.normal(0, 1.5, P["n"])
    P["label"] = (rng.random(P["n"]) < sigmoid(s)).astype(int)
    off = np.repeat(rng.normal(0, 3, 400), 20)
    out = fx.calibration_decomposition(P, g, {"with_offset": s + off, "clean": s, "shifted": s + 3.0}, seed=0)
    a, b = out["arms"]["with_offset"], out["arms"]["clean"]
    assert a["user_centred_platt"]["ECE"] == pytest.approx(b["user_centred_platt"]["ECE"])
    assert a["user_centred_platt"]["a"] == pytest.approx(b["user_centred_platt"]["a"])
    assert a["platt"]["ECE"] > b["platt"]["ECE"]
    sh = out["arms"]["shifted"]
    assert sh["intercept_only"]["b"] == pytest.approx(-3.0, abs=0.15)
    assert sh["intercept_only"]["ECE"] < 0.5 * sh["raw"]["ECE"]
    assert out["split"]["n_users_fit"] + out["split"]["n_users_eval"] == 400


# ---------------------------------------------------------------- A6
def test_topk_errors_and_margins_known_answer():
    g = {"u": np.arange(5), "v": np.arange(5, 8)}
    s = np.array([5, 4, 3, 2, 1, 1, 1, 0], float)
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    err, ties = fx.topk_errors(s, y, g, seed=0)
    assert err[:5].tolist() == [0, 1, 1, 0, 0]               # k = 2: one FP (idx 1), one FN (idx 2)
    assert err[5:7].sum() in (0, 2) and err[7] == 0 and ties == 1
    m = fx.topk_margin(s, err >= 0, y, g)
    assert m[:5].tolist() == [1.5, 0.5, 0.5, 1.5, 2.5]       # threshold (4 + 3) / 2


def test_error_targets_platt_and_topk_blocks():
    rng = np.random.default_rng(0)
    P = fx.panel_pairs(_rated_rows(300, 20, rng))
    g = fx._groups(P["user"])
    like = 2.0 + 1.0 * P["label"] + rng.normal(0, 1, P["n"])   # yes-biased: raw decision ~ always like
    out = fx.error_targets(P, g, {"like": like, "other": like + rng.normal(0, 1, P["n"])}, n_boot=20)
    ap, tk = out["after_platt"], out["per_user_topk"]
    assert ap["error_rate"] < out["raw_decision_reference"]["error_rate"]
    assert tk["n_errors"] % 2 == 0 and 0 < tk["error_rate"] < 0.5
    assert {"mag_like", "platt_margin_like", "mag_other"} <= set(ap["detectors"])
    assert {"mag_like", "topk_margin_like", "topk_margin_other"} <= set(tk["detectors"])
    assert tk["detectors"]["topk_margin_like"]["est"] > 0.55


# ---------------------------------------------------------------- A7
def test_partial_spearman_removes_a_shared_confounder():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 5000)
    x, y = z + rng.normal(0, .5, 5000), z + rng.normal(0, .5, 5000)
    assert spearman(x, y) > 0.7
    assert abs(fx.partial_spearman(x, y, z)) < 0.05
    y2 = z + x + rng.normal(0, .5, 5000)
    assert fx.partial_spearman(x, y2, z) > 0.3
    assert fx.partial_spearman(x, y, None) == pytest.approx(spearman(x, y))


def test_release_year_and_description_length():
    assert fx.release_year("Toy Story (1995)") == 1995
    assert fx.release_year("City of Lost Children, The (Cité des enfants perdus, La) (1995)") == 1995
    assert math.isnan(fx.release_year("Untitled"))
    assert fx.desc_length_from_text("Categories: Toys & Games Pools. Hello world") == 11
    assert fx.desc_length_from_text("Categories: Toys. ") == 0
    assert fx.desc_length_from_text("plain") == 5


# ---------------------------------------------------------------- A8
def test_ccrp_head_share_known_answer(tmp_path):
    hdr = ["user_id", "source_event_id", "candidate_item_ids", "candidate_popularity_groups", "pred_ranked_item_ids",
           "topk_popularity_groups", "positive_popularity_group"]
    ids = [f"c{k}" for k in range(12)]

    def ev(u, heads, top_heads, pos):
        grp = ["head" if k in heads else "tail" for k in range(12)]
        ranked = [f"c{k}" for k in top_heads] + [f"c{k}" for k in range(12) if k not in top_heads]
        return [u, f"{u}::1", str(ids), str(grp), str(ranked), str([grp[int(i[1:])] for i in ranked[:20]]), pos]

    rows = [ev("u1", range(6), [0, 1, 2, 3, 4, 6, 7, 8, 9, 10], "head"),   # pool 6/12, top-10 5 heads
            ev("u2", range(3), [0, 1, 6, 7, 8, 9, 10, 11, 4, 5], "tail"),  # pool 3/12, top-10 2 heads
            ev("u3", range(12), list(range(10)), "head")]                  # pool 1, top-10 1
    p = tmp_path / "rec.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    out = fx.ccrp_head_share(p, first_n=2, n_boot=20)
    a, b, al = out["events_1_2"], out["events_3_3"], out["events_all"]
    assert a["expected_random_top10_head_share"]["est"] == pytest.approx((0.5 + 0.25) / 2)
    assert a["ccrp_top10_head_share"]["est"] == pytest.approx((0.5 + 0.2) / 2)
    assert a["ccrp_minus_expected"]["est"] == pytest.approx(-0.025)
    assert a["positive_head_rate"]["est"] == 0.5 and a["n_topk_groups_mismatch"] == 0
    assert b["n_events"] == 1 and b["ccrp_top10_head_share"]["est"] == 1.0
    assert al["n_events"] == 3 and out["label"] == "exploratory"


def test_ccrp_head_share_splits_on_file_rows_and_checks_the_sports_panel(tmp_path):
    hdr = ["user_id", "source_event_id", "candidate_item_ids", "candidate_popularity_groups", "pred_ranked_item_ids",
           "topk_popularity_groups", "positive_popularity_group"]
    ids = [f"c{k}" for k in range(12)]

    def ev(u, n_heads):
        grp = ["head" if k < n_heads else "tail" for k in range(12)]
        return [u, f"{u}::1", str(ids), str(grp), str(ids), str(grp[:10]), "head"], grp

    evs = [ev(f"u{k}", h) for k, h in enumerate([2, 4, 6, 8])]     # top-10 head shares 0.2, 0.4, 0.6, 0.8
    rows = [r for r, _ in evs] + [["u9", "u9::1", "not a list", "['head']", "[]", "[]", "head"]]   # unparseable
    rows[0][3] = "['head']"                                         # file row 1 malformed (group length mismatch)
    p = tmp_path / "rec.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    panel = tmp_path / "sports_next_1k.jsonl"
    panel.write_text("\n".join(json.dumps({"source_event_id": f"u{k}::1", "candidate_item_ids": ids,
                                           "candidate_popularity_groups": evs[k][1]}) for k in range(2)),
                     encoding="utf-8")
    out = fx.ccrp_head_share(p, first_n=2, n_boot=20, sports_panel=panel)
    a, b = out["events_1_2"], out["events_3_5"]
    # the skipped row stays in its own block: file row 3 (u2) is never pulled into the first block
    assert a["n_events"] == 1 and a["n_rows_skipped"] == {"events_group_length_mismatch": 1}
    assert a["ccrp_top10_head_share"]["est"] == pytest.approx(0.4)
    assert b["n_events"] == 2 and b["n_rows_skipped"] == {"events_unparseable": 1}
    assert b["ccrp_top10_head_share"]["est"] == pytest.approx(0.7)
    assert out["n_file_rows"] == 5 and out["events_all"]["n_events"] == 3
    assert out["n_rows_skipped"] == {"events_group_length_mismatch": 1, "events_unparseable": 1}
    chk = out["sports_panel_check"]
    assert chk["n_record_rows_compared"] == 2 and chk["mismatches"] == {"popularity_groups_mismatch": 1}
    assert chk["identical"] is False
    rows[0][3] = str(evs[0][1])                                     # repaired file: rows 1-2 are the panel's events
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    ok = fx.ccrp_head_share(p, first_n=2, n_boot=20, sports_panel=panel)
    assert ok["sports_panel_check"]["identical"] is True and ok["events_1_2"]["n_events"] == 2
    swapped = fx.sports_panel_check(panel, [("u1::1", ids, evs[1][1]), ("u0::1", ids, evs[0][1])])
    assert swapped["mismatches"] == {"source_event_id_mismatch": 2, "popularity_groups_mismatch": 2}


def test_cluster_mean_ci_equals_stats_cluster_bootstrap():
    from src.confrec.stats import cluster_bootstrap
    rng = np.random.default_rng(0)
    users = rng.choice([f"u{k}" for k in range(60)], 400)
    x = rng.normal(0, 1, 400) + (users == "u3") * 2
    a = fx.cluster_mean_ci(x, users, n_boot=300, seed=4)
    b = cluster_bootstrap(lambda idx: float(x[idx].mean()), users, n_boot=300, seed=4)
    for k in ("est", "lo", "hi", "n_clusters"):
        assert a[k] == pytest.approx(b[k], rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------- wording flags
def test_wording_flags_follow_the_amendment_rules():
    res = {"panels": {"ml1m_rated": {
        "A2_prior": {"split_half": {"spearman_r": {"est": 0.65}, "spearman_brown_8": 0.79}},
        "A3_cf_references": {
            "mf_temporal": {"test_pairs_t_ge_T": {"UAUC_personal_residual": {"est": 0.55}},
                            "all_pairs_sensitivity": {"UAUC_personal_residual": {"est": 0.58}},
                            "test_pairs_t_ge_T_no_fold_in": {"UAUC_personal_residual": {"est": 0.52}}},
            "mf_transductive_reference": {"UAUC_personal_residual": {"est": 0.60}}},
        "A4_ensemble_null": {"dUAUC_mirror_minus_ensemble_null": {"est": 0.004, "lo": -0.001, "hi": 0.009},
                             "dUAUC_mirror_minus_ensemble_null_bias_corrected": {"est": 0.005, "lo": 0.0001},
                             "gap_residual": {"est": 0.006, "lo": -0.002}}},
        "sports_next_1k": {"A4_ensemble_null": {"dUAUC_mirror_minus_ensemble_null": {"est": 0.01, "lo": 0.002,
                                                                                      "hi": 0.02}}}}}
    f = fx.wording_flags(res)
    # primary: Spearman-Brown 8-donor reliability (0.79 -> not < 0.7), temporal MF on t >= T, one-sided lo <= 0
    assert f["ml1m_rated"] == {
        "pi_split_half_reliability_lt_0p7": False, "mf_personal_residual_uauc_le_0p55": True,
        "ensemble_null_residual_lo_le_0": True,
        "sensitivity": {"pi_split_half_raw_r_lt_0p7": True, "mf_temporal_all_pairs_residual_uauc_le_0p55": False,
                        "mf_temporal_no_fold_in_residual_uauc_le_0p55": True,
                        "mf_transductive_reference_residual_uauc_le_0p55": False,
                        "ensemble_null_residual_ci_covers_0_two_sided": True,
                        "ensemble_null_residual_bias_corrected_lo_le_0": False,
                        "ensemble_null_gap_residual_lo_le_0": True}}
    assert set(f["ml1m_rated"]) - {"sensitivity"} == set(fx.WORDING_RULES)
    assert set(f["ml1m_rated"]["sensitivity"]) == set(fx.SENSITIVITY_RULES)
    assert f["sports_next_1k"]["ensemble_null_residual_lo_le_0"] is False
    assert f["sports_next_1k"]["sensitivity"]["ensemble_null_residual_ci_covers_0_two_sided"] is False
    # a CI wholly below 0 (MIRROR at or below the diversity-only prediction): attributed to diversity (one-sided),
    # although the literal two-sided reading says the CI does not cover 0
    below = fx.wording_flags({"panels": {"x": {"A4_ensemble_null": {"dUAUC_mirror_minus_ensemble_null": {
        "est": -0.003, "lo": -0.005, "hi": -0.001}}}}})["x"]
    assert below["ensemble_null_residual_lo_le_0"] is True
    assert below["sensitivity"]["ensemble_null_residual_ci_covers_0_two_sided"] is False
    assert f["sports_next_1k"]["pi_split_half_reliability_lt_0p7"] is None
    assert f["sports_next_1k"]["mf_personal_residual_uauc_le_0p55"] is None
    # raw r 0.6 -> SB 0.75: the evidence collapse stays interpretable on the 8-donor pi
    sb = fx.wording_flags({"panels": {"x": {"A2_prior": {"split_half": {"spearman_r": {"est": 0.6},
                                                                         "spearman_brown_8": 0.75}}}}})["x"]
    assert sb["pi_split_half_reliability_lt_0p7"] is False and sb["sensitivity"]["pi_split_half_raw_r_lt_0p7"] is True


# ---------------------------------------------------------------- end to end (synthetic ML-1M)
def _ml1m_fixture(tmp_path, n_users=60, n_items=40, seed=0):
    rng = np.random.default_rng(seed)
    raw = tmp_path / "raw" / "ml-1m"
    raw.mkdir(parents=True)
    quality = rng.normal(0, 1, n_items + 1)
    with open(raw / "movies.dat", "w", encoding="latin-1") as f:
        for m in range(1, n_items + 1):
            f.write(f"{m}::Movie {m} ({1970 + m})::Drama|Comedy\n")
    events = {}
    with open(raw / "ratings.dat", "w", encoding="latin-1") as f:
        for u in range(1, n_users + 1):
            its = rng.choice(np.arange(1, n_items + 1), 25, replace=False)
            ts = np.sort(rng.choice(10 ** 6, 25, replace=False))
            r = np.clip(np.round(3 + 1.2 * quality[its] + rng.normal(0, 1, 25)), 1, 5).astype(int)
            events[u] = list(zip(ts.tolist(), its.tolist(), r.tolist()))
            for t, i, rr in events[u]:
                f.write(f"{u}::{i}::{rr}::{t}\n")
    panel, pop = [], {}
    for evs in events.values():
        for _, i, _ in evs:
            pop[i] = pop.get(i, 0) + 1
    for u, evs in events.items():
        lab_idx = [k for k, (_, _, r) in enumerate(evs) if r != 3 and k >= 3][-8:]
        labs = [int(evs[k][2] >= 4) for k in lab_idx]
        if sum(labs) < 2 or len(labs) - sum(labs) < 2:
            continue
        hist = evs[: lab_idx[0]][-5:]
        cand = [evs[k] for k in lab_idx]
        panel.append({"user_id": str(u), "source_event_id": f"{u}::{cand[0][0]}",
                      "history": [f"Movie {i} (rated {r}/5)" for _, i, r in hist],
                      "history_item_ids": [str(i) for _, i, _ in hist], "history_ratings": [float(r) for *_, r in hist],
                      "candidate_item_ids": [str(i) for _, i, _ in cand],
                      "candidate_titles": [f"Movie {i} ({1970 + i})" for _, i, _ in cand],
                      "candidate_texts": ["Genres: Drama, Comedy"] * len(cand), "candidate_brands": [""] * len(cand),
                      "candidate_ratings": [float(r) for *_, r in cand], "candidate_labels": labs,
                      "candidate_popularity": [pop[i] for _, i, _ in cand],
                      "candidate_popularity_prior": [1] * len(cand),
                      "candidate_timestamps": [int(t) for t, _, _ in cand], "source": "ml1m"})
    assert len(panel) >= 20
    panels = tmp_path / "panels"
    panels.mkdir()
    (panels / "ml1m_rated.jsonl").write_text("\n".join(json.dumps(r) for r in panel), encoding="utf-8")
    pilot = tmp_path / "pilot"
    (pilot / "ml1m_rated").mkdir(parents=True)
    (pilot / "ml1m_rated_nohist").mkdir()
    rows, nh, swap_items = [], [], []
    for r in panel:
        for c, (i, lab, rr) in enumerate(zip(r["candidate_item_ids"], r["candidate_labels"],
                                             r["candidate_ratings"])):
            q = quality[int(i)]
            like = 0.5 * (rr - 3) + q + rng.normal(0, 1)
            for name, lg in (("like", like), ("dislike", -0.4 * (rr - 3) + 0.3 * q + rng.normal(0, 1)),
                             ("like_para", like + rng.normal(0, 0.3))):
                rows.append(_row(r["source_event_id"], r["user_id"], i, c, lab, name, lg))
            nh.append(_row(r["source_event_id"], r["user_id"], i, c, lab, "like", q))
            swap_items.append(i)
    _gz(pilot / "ml1m_rated" / "scores.csv.gz", COLS, rows)
    nh.append(_row("zz::0", "zz", "1", 0, 1, "like", 0.0))   # a nohist row with no panel pair: join diagnostic
    _gz(pilot / "ml1m_rated_nohist" / "scores.csv.gz", COLS, nh)
    _gz(pilot / "ml1m_rated" / "swap_prior.csv.gz", SWAP_COLS,
        [[i, f"d{k}", "like", f"{quality[int(i)] + rng.normal(0, 1):.6f}", 0]
         for i in dict.fromkeys(swap_items) for k in range(8)])
    sha = fx.file_sha1(panels / "ml1m_rated.jsonl")
    (pilot / "ml1m_rated" / "report.json").write_text(json.dumps({"prompts_per_s": 87.10206897756021,
                                                                   "n_prompts": 111367, "data_sha1": sha}),
                                                       encoding="utf-8")
    (pilot / "ml1m_rated_nohist" / "report.json").write_text(json.dumps({"data_sha1": sha}), encoding="utf-8")
    return pilot, panels, tmp_path / "raw", panel


def _strict_load(path):
    def bad(x):
        raise ValueError(f"non-strict JSON constant {x}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=bad)


def test_cli_end_to_end_on_a_synthetic_ml1m_pilot(tmp_path):
    pilot, panels, raw, panel = _ml1m_fixture(tmp_path)
    out = tmp_path / "f" / "forensics.json"
    args = ["--pilot_dir", str(pilot), "--panels", str(panels), "--raw", str(raw), "--out", str(out),
            "--n_boot", "20", "--mf_dim", "4", "--mf_iters", "3"]
    fx.main(args)
    res = _strict_load(out)
    assert res["label"] == "exploratory" and "EXPLORATORY" in res["status"]
    m = res["panels"]["ml1m_rated"]
    for k in ("A2_prior", "A3_cf_references", "A4_ensemble_null", "A5_calibration", "A6_error_targets",
              "A7_popularity_partial", "A9_throughput"):
        assert m[k]["label"] == "exploratory", k
        assert "skipped" not in m[k], (k, m[k].get("skipped"))
    assert res["panels"]["toys_rated"]["skipped"].startswith("missing")
    assert m["A9_throughput"]["main"]["prompts_per_s"] == pytest.approx(87.10206897756021)
    assert m["A9_throughput"]["registered_reconciled_prompts_per_s"] == 87.1
    a3 = m["A3_cf_references"]
    assert a3["candidate_timestamps"] == "panel candidate_timestamps"
    assert a3["n_candidate_pairs_not_in_raw"] == 0
    ref = a3["mf_transductive_reference"]
    assert a3["item_mean_prior"]["UAUC"]["n_users"] > 0 and ref["n_pairs_cold_user"] == 0
    assert ref["n_train"] == 60 * 25 - sum(len(r["candidate_item_ids"]) for r in panel)
    tm = a3["mf_temporal"]
    assert 0 < tm["n_train"] < ref["n_train"] and tm["test_pairs_t_ge_T"]["n_pairs"] == tm["cutoff"][
        "n_panel_pairs_t_ge_T"] < tm["all_pairs_sensitivity"]["n_pairs"] == m["n_pairs"]
    assert "TRANSDUCTIVE" in ref["definition"] and "PRIMARY" in tm["definition"]
    assert m["A7_popularity_partial"]["controls"] == ["item_mean_prior", "release_year"]
    assert "partial_ci" in m["A7_popularity_partial"]["popularity_all_time"]["control_item_mean_prior"]["v_pair"]
    assert set(m["A6_error_targets"]["after_platt"]["detectors"]) >= {"mag_evidence", "platt_margin_mirror"}
    assert set(res["wording_flags"]["ml1m_rated"]) - {"sensitivity"} == set(fx.WORDING_RULES)
    # the scored panel is the one at --panels: identity verified, history-star block usable
    ident = m["panel_identity"]
    assert ident["consistent"] is True and ident["main"]["match"] is True and ident["nohist"]["match"] is True
    assert a3["history_star_mean"]["usable"] is True and "AUC_global" in a3["history_star_mean"]
    # the nohist file's join diagnostics and censoring reach the output
    assert m["A2_prior"]["nohist_join"] == {"score_rows_without_panel_pair": 1}
    assert m["A2_prior"]["nohist_censoring"]["like"]["n"] == m["n_pairs"] + 1
    assert m["A2_prior"]["nohist_like_finite_pairs"] == m["n_pairs"]
    # candidate timestamps rebuilt from raw give the same prior-only item means and the same temporal MF; the
    # rewritten panel no longer matches the scored run's data_sha1, so the history-star block is marked unusable
    for r in panel:
        del r["candidate_timestamps"]
    (panels / "ml1m_rated.jsonl").write_text("\n".join(json.dumps(r) for r in panel), encoding="utf-8")
    out2 = tmp_path / "f" / "forensics2.json"
    fx.main(args[:-8] + ["--out", str(out2), "--n_boot", "20", "--mf_dim", "4", "--mf_iters", "3",
                         "--names", "ml1m_rated"])
    m2 = _strict_load(out2)["panels"]["ml1m_rated"]
    a3b = m2["A3_cf_references"]
    assert a3b["candidate_timestamps"].startswith("rebuilt from raw")
    assert a3b["item_mean_prior"] == a3["item_mean_prior"]
    assert a3b["item_mean_prior_shrunk"] == a3["item_mean_prior_shrunk"]
    assert a3b["mf_temporal"] == a3["mf_temporal"]
    assert m2["panel_identity"]["consistent"] is False and m2["panel_identity"]["main"]["match"] is False
    assert "not byte-identical" in m2["warning"] and "warning" not in m
    hs = a3b["history_star_mean"]
    assert hs["usable"] is False and "AUC_global" not in hs and hs["AUC_global_unusable"] == a3[
        "history_star_mean"]["AUC_global"]


def _amazon_fixture(tmp_path, n_users=50, n_items=30, seed=0):
    """Synthetic raw/amazon_toys (build_rated_panels.load_amazon format) + toys_rated panel + pilot score files."""
    rng = np.random.default_rng(seed)
    d = tmp_path / "raw" / "amazon_toys"
    d.mkdir(parents=True)
    quality = rng.normal(0, 1, n_items)
    desc = {f"A{k}": ["word " * (k + 1), "tail"] if k % 3 else [] for k in range(n_items)}
    store = {f"A{k}": f"Brand{k % 4}" if k % 5 else "" for k in range(n_items)}
    with gzip.open(d / "meta_Toys_and_Games.jsonl.gz", "wt", encoding="utf-8") as f:
        for k in range(n_items):
            i = f"A{k}"
            f.write(json.dumps({"parent_asin": i, "title": f"Toy {k}", "categories": ["Toys & Games", "Puzzles"],
                                "description": desc[i], "store": store[i]}) + "\n")
    events = {}
    with gzip.open(d / "Toys_and_Games.jsonl.gz", "wt", encoding="utf-8") as f:
        for u in range(n_users):
            its = rng.choice(n_items, 15, replace=False)
            ts = np.sort(rng.choice(10 ** 9, 15, replace=False)) + 10 ** 12
            r = np.clip(np.round(3 + 1.2 * quality[its] + rng.normal(0, 1, 15)), 1, 5)
            events[f"U{u}"] = [(int(t), f"A{i}", float(x)) for t, i, x in zip(ts, its, r)]
            for t, i, x in events[f"U{u}"]:
                f.write(json.dumps({"user_id": f"U{u}", "parent_asin": i, "rating": x, "timestamp": t}) + "\n")
    panel = []
    for u, evs in events.items():
        lab_idx = [k for k, (_, _, r) in enumerate(evs) if r != 3 and k >= 3][-8:]
        labs = [int(evs[k][2] >= 4) for k in lab_idx]
        if sum(labs) < 2 or len(labs) - sum(labs) < 2:
            continue
        cand = [evs[k] for k in lab_idx]
        cats = "Categories: Toys & Games Puzzles. "
        panel.append({"user_id": u, "source_event_id": f"{u}::{cand[0][0]}", "history_ratings": [4.0, 2.0, 5.0],
                      "candidate_item_ids": [i for _, i, _ in cand],
                      "candidate_titles": [f"Toy {i[1:]}" for _, i, _ in cand],
                      "candidate_texts": [cats + " ".join(desc[i]) for _, i, _ in cand],
                      "candidate_brands": [store[i] for _, i, _ in cand], "candidate_labels": labs,
                      "candidate_ratings": [r for *_, r in cand], "candidate_popularity": [5] * len(cand),
                      "candidate_popularity_prior": [1] * len(cand),
                      "candidate_timestamps": [t for t, _, _ in cand], "source": "toys"})
    assert len(panel) >= 15
    panels = tmp_path / "panels"
    panels.mkdir()
    (panels / "toys_rated.jsonl").write_text("\n".join(json.dumps(r) for r in panel), encoding="utf-8")
    run = tmp_path / "pilot" / "toys_rated"
    run.mkdir(parents=True)
    rows = []
    for r in panel:
        for c, (i, lab) in enumerate(zip(r["candidate_item_ids"], r["candidate_labels"])):
            like = quality[int(i[1:])] + 0.7 * lab + rng.normal(0, 1)
            rows += [_row(r["source_event_id"], r["user_id"], i, c, lab, "like", like),
                     _row(r["source_event_id"], r["user_id"], i, c, lab, "dislike", -0.5 * lab + rng.normal(0, 1)),
                     _row(r["source_event_id"], r["user_id"], i, c, lab, "like_para", like + rng.normal(0, .3))]
    _gz(run / "scores.csv.gz", COLS, rows)
    _gz(run / "swap_prior.csv.gz", SWAP_COLS, [[f"A{k}", f"d{j}", "like", f"{quality[k] + rng.normal():.6f}", 0]
                                               for k in range(n_items) for j in range(8)])
    return tmp_path / "pilot", panels, tmp_path / "raw", desc, store


def test_cli_amazon_branch_uses_raw_description_length_and_store(tmp_path):
    pilot, panels, raw, desc, store = _amazon_fixture(tmp_path)
    assert fx.amazon_desc_lengths(raw, "toys", {"A1", "A3"}) == {"A1": len("word word tail"), "A3": 0}
    out = tmp_path / "toys.json"
    fx.main(["--pilot_dir", str(pilot), "--panels", str(panels), "--raw", str(raw), "--out", str(out),
             "--names", "toys_rated", "--n_boot", "20", "--mf_dim", "4", "--mf_iters", "3"])
    m = _strict_load(out)["panels"]["toys_rated"]
    a7 = m["A7_popularity_partial"]
    assert a7["controls"] == ["item_mean_prior", "desc_len", "has_store"]
    assert a7["desc_len_source"] == "raw meta description" and a7["n_items_desc_from_raw"] > 0
    assert "partial" in a7["popularity_all_time"]["control_item_mean_prior"]["pi_item"]
    a3 = m["A3_cf_references"]
    assert "skipped" not in a3 and a3["n_candidate_pairs_not_in_raw"] == 0
    assert a3["mf_transductive_reference"]["n_train"] > a3["mf_temporal"]["n_train"] > 0
    assert m["panel_identity"]["consistent"] is None and a3["history_star_mean"]["usable"] is None  # no data_sha1
    assert m["A2_prior"]["split_half"]["n_items"] > 0 and m["source"] == "toys"


def test_cli_requires_an_input(tmp_path):
    with pytest.raises(SystemExit):
        fx.parse_args(["--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        fx.parse_args(["--ccrp_head_share", "r.csv", "--out", "x.json", "--mf_cutoff_quantile", "1.5"])
    a = fx.parse_args(["--ccrp_head_share", "r.csv", "--out", "x.json", "--panels", str(tmp_path)])
    assert a.sports_panel is None and a.mf_cutoff_quantile == 0.8 and a.null_sim_reps == 50
    (tmp_path / "sports_next_1k.jsonl").write_text("{}\n", encoding="utf-8")
    a = fx.parse_args(["--ccrp_head_share", "r.csv", "--out", "x.json", "--panels", str(tmp_path)])
    assert a.sports_panel == str(tmp_path / "sports_next_1k.jsonl")
