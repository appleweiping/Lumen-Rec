"""Calibration / reliability / selective-prediction metrics for LLM-recommender confidence.

All functions are numpy-only so they run on CPU over per-candidate or per-user arrays.
Conventions: `conf` in [0, 1] is a predicted probability of the event `correct` in {0, 1}.
"""
from __future__ import annotations

import numpy as np


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15, adaptive: bool = False) -> float:
    """Expected calibration error (equal-width bins, or equal-mass when adaptive=True)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if adaptive:
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        edges[-1] = 1 + 1e-9
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def reliability_bins(conf, correct, n_bins: int = 15):
    """Rows (lo, hi, n, mean_conf, accuracy) for a reliability diagram."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        if m.any():
            rows.append((lo, hi, int(m.sum()), float(conf[m].mean()), float(correct[m].mean())))
    return rows


def brier(conf, correct) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return float(np.mean((conf - correct) ** 2))


def nll(conf, correct, eps: float = 1e-7) -> float:
    conf = np.clip(np.asarray(conf, float), eps, 1 - eps)
    correct = np.asarray(correct, float)
    return float(-np.mean(correct * np.log(conf) + (1 - correct) * np.log(1 - conf)))


def auroc(score, label) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties averaged; NaN if a class is absent."""
    from .stats import rankdata_avg
    score, label = np.asarray(score, float), np.asarray(label, int)
    pos, neg = label.sum(), (1 - label).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata_avg(score)  # average ranks over ties
    return float((ranks[label == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def risk_coverage(conf, utility):
    """Serve the most-confident units first. Returns (coverage, mean utility of served set) arrays
    and AURC = area under the (1 - utility) risk curve (lower is better).

    Tie-invariant: units with equal confidence are served as a block whose utilities are replaced by the
    block mean, i.e. the expectation over random tie-breaking (input order never matters)."""
    conf, utility = np.asarray(conf, float), np.asarray(utility, float)
    order = np.argsort(-conf, kind="stable")
    c, u = conf[order], utility[order].copy()
    i = 0
    while i < len(c):
        j = i
        while j + 1 < len(c) and c[j + 1] == c[i]:
            j += 1
        if j > i:
            u[i: j + 1] = u[i: j + 1].mean()
        i = j + 1
    cum = np.cumsum(u) / np.arange(1, len(order) + 1)
    cov = np.arange(1, len(order) + 1) / len(order)
    aurc = float(np.mean(1 - cum))
    return cov, cum, aurc


def bias_index(conf, correct, group, n_bins: int = 10, adjust: bool = True, min_count: int = 20) -> dict:
    """ProCal-style bias index: calibration gap between groups at MATCHED confidence.

    Confidence is split into equal-mass bins (on average ranks, so tied confidences share a bin). Within each
    bin, adjust=True compares each group's calibration residual (acc - conf) with the pooled residual, so a
    group that merely sits higher inside the same bin gets no spurious sign; adjust=False compares raw
    accuracy (the original ProCal form). Returns {group: weighted mean gap}. Positive = the group is MORE
    accurate than its confidence suggests relative to others (under-confident), negative = over-confident.
    """
    from .stats import rank_bins
    conf, correct, group = np.asarray(conf, float), np.asarray(correct, float), np.asarray(group)
    idx = rank_bins(conf, n_bins)
    resid = correct - conf if adjust else correct
    out = {}
    for g in np.unique(group):
        num = den = 0.0
        for b in range(n_bins):
            mb = idx == b
            mg = mb & (group == g)
            if mg.sum() >= min_count:
                num += mg.sum() * (resid[mg].mean() - resid[mb].mean())
                den += mg.sum()
        out[str(g)] = num / den if den else float("nan")
    return out


def bias_index_ci(conf, correct, group, clusters, n_bins: int = 10, adjust: bool = True,
                  n_boot: int = 1000, seed: int = 0) -> dict:
    """bias_index per group with a cluster (e.g. user) bootstrap CI: {group: {est, lo, hi, n_clusters}}."""
    from .stats import percentile_ci
    conf, correct, group = np.asarray(conf, float), np.asarray(correct, float), np.asarray(group)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    members = [np.flatnonzero(inv == c) for c in range(len(uniq))]
    est = bias_index(conf, correct, group, n_bins, adjust)
    rng = np.random.default_rng(seed)
    boots = {g: [] for g in est}
    for _ in range(n_boot):
        idx = np.concatenate([members[c] for c in rng.integers(0, len(uniq), len(uniq))])
        b = bias_index(conf[idx], correct[idx], group[idx], n_bins, adjust)
        for g in boots:
            boots[g].append(b.get(g, float("nan")))
    out = {}
    for g, v in est.items():
        lo, hi = percentile_ci(boots[g])
        out[g] = {"est": v, "lo": lo, "hi": hi, "n_clusters": int(len(uniq))}
    return out


def ranking_metrics(pos_rank: np.ndarray, ks=(5, 10, 20)) -> dict:
    r = np.asarray(pos_rank, float)
    out = {f"HR@{k}": float(np.mean(r <= k)) for k in ks}
    out.update({f"NDCG@{k}": float(np.mean(np.where(r <= k, 1 / np.log2(r + 1), 0.0))) for k in ks})
    out["MRR"] = float(np.mean(1 / r))
    return out


def ndcg_from_rank(r: np.ndarray, k: int = 10) -> np.ndarray:
    r = np.asarray(r, float)
    return np.where(r <= k, 1 / np.log2(r + 1), 0.0)
