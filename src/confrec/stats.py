"""Shared statistics for the confrec pilots (numpy-only, deterministic given a seed).

Conventions used across pilot_mirror / pilot_pseudonym / pilot_kuairec:
  * every confidence interval resamples the INDEPENDENT unit (users; items for item-level correlations),
    never individual (user, candidate) pairs, because one user's candidates share a prompt history;
  * rank statistics average ties (LLM logits are discretised by the model dtype, so ties are common);
  * JSON outputs are strict (NaN/inf -> null).
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np


def rankdata_avg(x) -> np.ndarray:
    """1-based ranks with ties replaced by their average rank (scipy.stats.rankdata method='average')."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.array([], float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts).astype(float)          # 1-based rank of each tie group's last member
    return ((ends - counts + 1 + ends) / 2)[inv.reshape(-1)]


def spearman(x, y) -> float:
    """Tie-corrected Spearman rho (Pearson on average ranks); NaN pairs dropped; NaN if < 3 pairs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = rankdata_avg(x[ok]), rankdata_avg(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def percentile_ci(samples: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    s = np.asarray([v for v in samples if np.isfinite(v)], float)
    if len(s) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def cluster_bootstrap(stat: Callable[[np.ndarray], float], clusters, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """Percentile CI for `stat(row_indices)` resampling whole clusters with replacement.

    `clusters` is a per-row cluster id (e.g. user_id). `stat` receives an int index array into the rows
    (with repeats) and returns a float. Returns {"est", "lo", "hi", "n_clusters", "n_boot"}."""
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == c) for c in range(len(uniq))]
    est = float(stat(np.arange(len(clusters))))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[c] for c in pick]) if len(pick) else np.array([], int)
        boots.append(float(stat(idx)))
    lo, hi = percentile_ci(boots, alpha)
    return {"est": est, "lo": lo, "hi": hi, "n_clusters": int(len(uniq)), "n_boot": n_boot}


def paired_bootstrap(a: dict, b: dict, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Mean of per-unit differences a[k] - b[k] over common finite keys, CI resampling units."""
    keys = sorted(k for k in set(a) & set(b) if np.isfinite(a[k]) and np.isfinite(b[k]))
    if not keys:
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    d = np.array([a[k] - b[k] for k in keys], float)
    rng = np.random.default_rng(seed)
    boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n_boot)]
    lo, hi = percentile_ci(boots, alpha)
    return {"est": float(d.mean()), "lo": lo, "hi": hi, "n": int(len(d))}


def tie_aware_rank(scores, pos_idx: int) -> float:
    """Expected 1-based rank of candidate `pos_idx` under uniformly random tie-breaking."""
    s = np.asarray(scores, float)
    sp = s[pos_idx]
    if not np.isfinite(sp):
        return float(len(s))  # unscorable positive ranks last (pessimistic, explicit)
    greater = np.sum(s > sp)
    ties = np.sum(s == sp) - 1
    return float(1 + greater + ties / 2)


def rank_bins(x, n_bins: int = 10) -> np.ndarray:
    """Quantile bins on average ranks: tied values share a bin and no bin is created empty by
    duplicated quantile edges (np.quantile + searchsorted does that on tied integer counts)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.array([], int)
    r = rankdata_avg(x)
    return np.minimum((n_bins * (r - 0.5) / len(x)).astype(int), n_bins - 1)


def ols_slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0:
        return float("nan")
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def sigmoid(z):
    z = np.asarray(z, float)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.clip(z, -500, 500))),
                    np.exp(np.clip(z, -500, 500)) / (1 + np.exp(np.clip(z, -500, 500))))


def platt_fit(score, label, iters: int = 100, l2: float = 1e-4) -> tuple[float, float]:
    """Fit p = sigmoid(a * score + b) by (lightly L2-regularised) Newton steps. Returns (a, b)."""
    s, y = np.asarray(score, float), np.asarray(label, float)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    w = np.zeros(2)
    X = np.stack([s, np.ones_like(s)], 1)
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def platt_apply(score, ab: tuple[float, float]) -> np.ndarray:
    return sigmoid(ab[0] * np.asarray(score, float) + ab[1])


def user_halves(users: Iterable, seed: int = 0) -> set:
    """Deterministic 50% user split (the returned set is the calibration half)."""
    import hashlib
    return {u for u in set(users)
            if int(hashlib.sha1(f"{seed}:{u}".encode()).hexdigest(), 16) % 2 == 0}


def strict_json(obj):
    """Recursively replace NaN/inf with None and numpy scalars with Python scalars."""
    if isinstance(obj, dict):
        return {str(k): strict_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
