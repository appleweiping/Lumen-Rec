import json
import math

import numpy as np

from src.confrec import stats


def test_spearman_ties_match_scipy_definition():
    # ordinal tie-breaking gave 1.0 vs 0.6 for these two orderings; tie-averaged rho is 0.894 for both
    y = [0, 0, 1, 1]
    assert abs(stats.spearman([1, 2, 3, 4], y) - 0.8944) < 1e-3
    assert abs(stats.spearman([2, 1, 4, 3], y) - 0.8944) < 1e-3
    assert math.isnan(stats.spearman([1, 1, 1, 1], [1, 2, 3, 4]))
    assert abs(stats.spearman([1, 2, np.nan, 4, 5], [2, 4, 9, 8, 10]) - 1.0) < 1e-9


def test_tie_aware_rank():
    assert stats.tie_aware_rank([0.5, 0.9, 0.5, 0.1], 0) == 2.5  # one greater, one tie
    assert stats.tie_aware_rank([3, 2, 1], 0) == 1.0
    assert stats.tie_aware_rank([1, 1, 1, 1], 2) == 2.5          # all tied -> middle


def test_rank_bins_never_empty_under_heavy_ties():
    x = np.array([1] * 300 + list(range(2, 702)))
    b = stats.rank_bins(x, 10)
    assert len(set(b[:300])) == 1                                # ties share a bin
    counts = np.bincount(b, minlength=10)
    assert counts.sum() == len(x) and (counts[counts > 0] >= 1).all()
    assert counts.max() <= 400


def test_cluster_bootstrap_wider_than_pairwise_when_clustered():
    rng = np.random.default_rng(0)
    users = np.repeat(np.arange(50), 20)
    shock = rng.normal(0, 1, 50)[users]                          # strong user-level common shock
    y = shock + rng.normal(0, 0.1, len(users))
    cb = stats.cluster_bootstrap(lambda idx: y[idx].mean(), users, n_boot=500)
    naive = stats.cluster_bootstrap(lambda idx: y[idx].mean(), np.arange(len(y)), n_boot=500)
    assert (cb["hi"] - cb["lo"]) > 2.5 * (naive["hi"] - naive["lo"])
    assert cb["n_clusters"] == 50


def test_paired_bootstrap_and_strict_json():
    a = {"u1": 0.7, "u2": 0.8, "u3": float("nan")}
    b = {"u1": 0.6, "u2": 0.6, "u4": 0.1}
    r = stats.paired_bootstrap(a, b, n_boot=200)
    assert r["n"] == 2 and abs(r["est"] - 0.15) < 1e-12
    s = json.dumps(stats.strict_json({"x": float("nan"), "y": np.float32(1.5), "z": [np.int64(2), np.inf]}))
    assert s == '{"x": null, "y": 1.5, "z": [2, null]}'


def test_platt_recovers_known_map():
    rng = np.random.default_rng(1)
    s = rng.normal(0, 2, 20000)
    y = (rng.random(len(s)) < stats.sigmoid(0.5 * s - 0.3)).astype(int)
    a, b = stats.platt_fit(s, y)
    assert abs(a - 0.5) < 0.05 and abs(b + 0.3) < 0.05


def test_ols_slope_and_user_halves():
    assert abs(stats.ols_slope([0, 1, 2, 3], [1, 3, 5, 7]) - 2.0) < 1e-12
    h = stats.user_halves([f"u{i}" for i in range(1000)], seed=0)
    assert 400 < len(h) < 600 and h == stats.user_halves([f"u{i}" for i in range(1000)], seed=0)
