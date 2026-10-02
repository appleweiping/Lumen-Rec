import numpy as np

from src.confrec import metrics as M


def test_perfect_calibration_has_zero_ece():
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=200_000)
    correct = rng.uniform(size=conf.size) < conf
    assert M.ece(conf, correct) < 0.01
    assert M.ece(conf, correct, adaptive=True) < 0.01


def test_overconfidence_is_detected():
    conf = np.full(1000, 0.9)
    correct = np.r_[np.ones(500), np.zeros(500)]
    assert abs(M.ece(conf, correct) - 0.4) < 1e-9
    assert abs(M.brier(conf, correct) - (0.5 * 0.01 + 0.5 * 0.81)) < 1e-9


def test_auroc_matches_definition_with_ties():
    score = np.array([0.1, 0.4, 0.4, 0.8])
    label = np.array([0, 0, 1, 1])
    # pairs (pos,neg): (0.4,0.1)=1, (0.4,0.4)=0.5, (0.8,0.1)=1, (0.8,0.4)=1 -> 3.5/4
    assert abs(M.auroc(score, label) - 0.875) < 1e-12


def test_risk_coverage_prefers_informative_confidence():
    util = np.r_[np.ones(50), np.zeros(50)]
    _, _, aurc_good = M.risk_coverage(np.r_[np.ones(50), np.zeros(50)], util)
    _, _, aurc_bad = M.risk_coverage(np.r_[np.zeros(50), np.ones(50)], util)
    assert aurc_good < aurc_bad


def test_risk_coverage_is_tie_invariant():
    util = np.r_[np.ones(50), np.zeros(50)]
    const = np.zeros(100)  # uninformative confidence must not get the oracle AURC from row order
    _, _, a1 = M.risk_coverage(const, util)
    _, _, a2 = M.risk_coverage(const, util[::-1])
    assert abs(a1 - a2) < 1e-12 and abs(a1 - 0.5) < 1e-12


def test_bias_index_residual_has_no_sign_under_calibration():
    # both groups perfectly calibrated; head sits higher inside the same equal-mass bins
    rng = np.random.default_rng(0)
    n = 200_000
    grp = rng.choice(["head", "mid", "tail"], size=n, p=[0.2, 0.6, 0.2])
    conf = np.where(grp == "head", rng.uniform(0.55, 1, n), np.where(grp == "tail", rng.uniform(0, 0.75, n),
                                                                    rng.uniform(0, 1, n)))
    correct = (rng.uniform(size=n) < conf).astype(float)
    adj = M.bias_index(conf, correct, grp, adjust=True)
    raw = M.bias_index(conf, correct, grp, adjust=False)
    assert abs(adj["head"]) < 0.004 and abs(adj["tail"]) < 0.004
    assert raw["head"] > adj["head"] and raw["tail"] < adj["tail"]  # the unadjusted form leaks the confound


def test_bias_index_ci_brackets_estimate():
    rng = np.random.default_rng(1)
    users = np.repeat(np.arange(200), 50)
    grp = rng.choice(["head", "tail"], size=len(users))
    conf = rng.uniform(size=len(users))
    correct = (rng.uniform(size=len(users)) < np.where(grp == "head", conf * 0.8, conf)).astype(float)
    ci = M.bias_index_ci(conf, correct, grp, users, n_boot=200)
    assert ci["head"]["lo"] <= ci["head"]["est"] <= ci["head"]["hi"] and ci["head"]["hi"] < 0


def test_ranking_metrics():
    out = M.ranking_metrics(np.array([1, 2, 11]))
    assert abs(out["HR@10"] - 2 / 3) < 1e-12
    assert abs(out["NDCG@10"] - (1 + 1 / np.log2(3)) / 3) < 1e-12
