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


def test_ranking_metrics():
    out = M.ranking_metrics(np.array([1, 2, 11]))
    assert abs(out["HR@10"] - 2 / 3) < 1e-12
    assert abs(out["NDCG@10"] - (1 + 1 / np.log2(3)) / 3) < 1e-12
