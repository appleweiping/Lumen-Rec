"""Pilot 1 (A1 MIRROR) analysis. Registered: idea-stage/triage_verdict.md section 3.1; binding operationalization:
idea-stage/PREREG_AMENDMENT_1.md P1.1-P1.4 and C1 (every CI resamples users, or items for item-level correlations).

    python -m src.confrec.pilot_mirror --scores <dir>/scores.csv.gz --panel <panel.jsonl> \
        [--swap <dir>/swap_prior.csv.gz] [--nohist <nohist_dir>/scores.csv.gz] [--base_q like|next] \
        [--ref_ranks docs/sigir/ref_ranks/sports/ccrp_v3.csv.gz] --out <dir>/pilot_mirror.json [--n_boot 2000] [--seed 0]

Arms per (user, candidate), L(q) = log P(Yes) - log P(No) for question q:
  raw        = L(base_q)                     (default: next on next-item panels when scored, else like)
  raw_like   = L(like)
  mirror     = L(like) - L(dislike)          (ranking; mirror_half = mirror / 2 for raw-scale probability metrics)
  placebo    = (L(like) + L(like_para)) / 2  (equal-compute two-prompt ensemble)
  evidence   = L(like) - pi(i)               (pi = mean like-logit under other users' histories, --swap)
  pmi_nohist = L(like) - L_nohist(like)      (same question with no history, item mean, --nohist)
Rows whose logit is non-finite or flagged censored 2/3 are dropped and counted per question and code; censored=1
rows (one side imputed with the smallest top-k logprob, the logit is a bound) are kept.
Rated panels: per-user UAUC, paired user-bootstrap dUAUC (primary: mirror - placebo, on rows where both arms are
finite), acquiescence/valence/prior estimands (P1.1), calibration (P1.2), confident-error detection (P1.3).
Correlation-matched ensemble null (idea-stage/PREREG_AMENDMENT_2.md G7, A4), block `ensemble_null`: mirror is the
two-view ensemble like + (-dislike), so it is compared with the UAUC a pure ensemble of those two views is predicted
to reach. Per user (rows finite in both views, both classes present): AUC_1, AUC_2 and the pooled within-class
Pearson correlation rho of the two logits; binormal (equal-variance) d'_k = sqrt(2) Phi^-1(AUC_k), clipped half a
pair inside (0, 1); predicted AUC of the equal-weight standardized sum = Phi(((d'_1 + d'_2) / sqrt(2 + 2 rho)) /
sqrt(2)). Reported: mean predicted UAUC, the observed arm's UAUC on the same users, and the paired user-bootstrap
dUAUC(mirror - ensemble_null) (the key pilot1_gate.py gates on); the same construction for the placebo pair (like,
like_para) is the sanity check (observed placebo should sit on its prediction). The empirical within-user z-sums
(their UAUC, paired dUAUC(arm - z-sum) and dUAUC(z-sum mirror - z-sum placebo)) are reported as context only.
ensemble_null.caveats (ENSEMBLE_NULL_CAVEATS) records the known properties: the null absorbs acquiescence
cancellation (it enters as a negative within-class rho), and at ~20 candidates per user the per-user prediction is
biased about 0.001-0.003 high for a pure ensemble (conservative against MIRROR).
Next-item panels (rows carry positive_item_index): NDCG@10 / HR@10 / MRR per user at the tie-aware expected rank
(C1) and, as *_tie_exact, their exact expectation over the positive's tie group; paired dNDCG@10 between arms with
both arms ranked on the candidates finite in both (as for dUAUC); C-CRP reference ranks joined on source_event_id
and every arm - ccrp on the joined events (P1.4); expected head share of the top 10.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

from src.confrec.metrics import auroc, brier, ece, ndcg_from_rank, risk_coverage
from src.confrec.stats import (cluster_bootstrap, paired_bootstrap, percentile_ci, platt_apply, platt_fit, sigmoid,
                               spearman, strict_json, tie_aware_rank, user_halves)

NAN = float("nan")
K = 10
UAUC_PAIRS = [("mirror", "placebo"), ("mirror", "raw_like"), ("evidence", "raw_like"), ("pmi_nohist", "raw_like"),
              ("mirror", "raw"), ("placebo", "raw_like"), ("evidence", "placebo"), ("mirror", "evidence"),
              ("mirror", "pmi_nohist")]
NDCG_PAIRS = [("mirror", "raw"), ("mirror", "raw_like"), ("mirror", "placebo")]  # plus every arm - ccrp
METRICS = ("NDCG@10", "HR@10", "MRR")
EXACT = tuple(f"{m}_tie_exact" for m in METRICS)
PLATT_ARMS = ("raw", "raw_like", "mirror", "placebo", "evidence", "pmi_nohist")
RAW_SCALE_ARMS = ("raw", "raw_like", "mirror_half", "placebo")
_NORMAL = NormalDist()
SQRT2 = math.sqrt(2.0)
# Known properties of the registered binormal ensemble null, recorded next to the gated key (ensemble_null.caveats).
ENSEMBLE_NULL_CAVEATS = [
    "Construction: per-user binormal prediction (contract item 4b). Amendment 2 G7 also words the null as 'the "
    "within-user z-sum of like and -dislike, compared with like + like_para'; that reading is reported as context "
    "(z_sum_UAUC, dUAUC_mirror_minus_zsum_null, dUAUC_zsum_mirror_minus_zsum_placebo) and never gates.",
    "Acquiescence absorption: a shared item-level yes-saying term a enters both views with opposite signs after "
    "negating dislike, so it shows up as a negative within-class correlation rho and the null already credits the "
    "two-view sum with its cancellation. With Gaussian views mirror - null is about 0 by construction; mirror beats "
    "the null only through unequal effective weighting or non-Gaussian structure. A POSITIVE that requires lo > 0 is "
    "therefore hard to reach by design (conservative).",
    "Small-sample bias: with about 20 candidates per user the per-user prediction overstates a pure two-view "
    "ensemble by about 0.001-0.003 UAUC (simulated on the Pilot-1 label designs; the placebo pair at rho about 0.95 "
    "is unbiased). It never produces a spurious lo > 0, but a pure ensemble can sit significantly below its null "
    "(hi < 0): read hi < 0 as 'not above a two-view ensemble', not as a loss.",
]


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _open(path):
    path = str(path)
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.endswith(".gz") else \
        open(path, encoding="utf-8", newline="")


def load_scores(path) -> dict:
    """scores.csv.gz -> {keys, user, item, label, L: {question: logit array aligned to keys}, censoring}.
    Keys are sorted (source_event_id, cand_idx); a duplicated (key, question) keeps the last row and is counted."""
    vals, meta, seen = defaultdict(dict), {}, set()
    cens = defaultdict(Counter)
    with _open(path) as f:
        for r in csv.DictReader(f):
            q, k = r["question"], (r["source_event_id"], int(r["cand_idx"]))
            code = (r.get("censored") or "na").strip()
            cens[q]["n"] += 1
            cens[q][code] += 1
            if (k, q) in seen:
                cens[q]["duplicate"] += 1
            seen.add((k, q))
            meta[k] = (r["user_id"], r["item_id"], int(float(r["label"])))
            lg = _num(r["logit"])
            if np.isfinite(lg) and code not in ("2", "3"):
                vals[k][q] = lg
            else:
                vals[k].pop(q, None)
                cens[q]["dropped_nonfinite"] += 1
    keys = sorted(meta)
    return {"keys": keys,
            "user": np.array([meta[k][0] for k in keys], dtype=object),
            "item": np.array([meta[k][1] for k in keys], dtype=object),
            "label": np.array([meta[k][2] for k in keys], int),
            "L": {q: np.array([vals[k].get(q, NAN) for k in keys], float) for q in sorted(cens)},
            "censoring": {q: dict(c) for q, c in sorted(cens.items())}}


def load_swap(path) -> dict:
    """swap_prior.csv.gz -> {pi: {question: {item: mean finite donor logit}}, n_donors, censoring}."""
    acc, cens = defaultdict(lambda: defaultdict(list)), defaultdict(Counter)
    with _open(path) as f:
        for r in csv.DictReader(f):
            q, code, lg = r.get("question") or "like", (r.get("censored") or "na").strip(), _num(r["logit"])
            cens[q]["n"] += 1
            cens[q][code] += 1
            if np.isfinite(lg) and code not in ("2", "3"):
                acc[q][r["item_id"]].append(lg)
            else:
                cens[q]["dropped_nonfinite"] += 1
    return {"pi": {q: {i: float(np.mean(v)) for i, v in d.items()} for q, d in acc.items()},
            "n_donors": {q: {i: len(v) for i, v in d.items()} for q, d in acc.items()},
            "censoring": {q: dict(c) for q, c in sorted(cens.items())}}


def load_ref_ranks(path) -> dict:
    """docs/sigir/ref_ranks/<domain>/<method>.csv.gz -> {source_event_id: (user_id, positive_rank, num_candidates)}.
    A duplicated source_event_id is an error (export_ref_ranks writes unique ids)."""
    out, dup = {}, set()
    with _open(path) as f:
        for r in csv.DictReader(f):
            ev = r["source_event_id"]
            if ev in out:
                dup.add(ev)
            out[ev] = (r.get("user_id", ""), float(r["positive_rank"]), int(float(r.get("num_candidates") or 0)))
    if dup:
        raise ValueError(f"{path}: {len(dup)} duplicated source_event_id(s), e.g. {sorted(dup)[:3]}")
    return out


def _groups(x) -> dict:
    """{value: row indices} in sorted value order."""
    x = np.asarray(x)
    if len(x) == 0:
        return {}
    uniq, inv = np.unique(x, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    return dict(zip(uniq.tolist(), np.split(order, np.cumsum(np.bincount(inv, minlength=len(uniq)))[:-1])))


def _item_means(x, inv, n_items) -> np.ndarray:
    """Per-item mean of the finite entries of x (NaN for items without any)."""
    ok = np.isfinite(x)
    s = np.bincount(inv[ok], x[ok], minlength=n_items)
    c = np.bincount(inv[ok], minlength=n_items)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), NAN)


def _uauc(score, label, users, mask) -> dict:
    """Per-user AUC over the rows in `mask`, for users with both classes."""
    out = {}
    for u, idx in users.items():
        idx = idx[mask[idx]]
        y = label[idx]
        if 0 < y.sum() < len(y):
            out[u] = auroc(score[idx], y)
    return out


def unit_bootstrap(stat, n_units: int, n_boot: int = 2000, seed: int = 0) -> dict:
    """stats.cluster_bootstrap for one row per cluster (item-level statistics): identical draws for the same seed,
    without its O(rows x clusters) cluster indexing, which dominates at ~25k singleton items."""
    est = float(stat(np.arange(n_units)))
    rng = np.random.default_rng(seed)
    boots = [float(stat(rng.integers(0, n_units, n_units))) for _ in range(n_boot)]
    lo, hi = percentile_ci(boots)
    return {"est": est, "lo": lo, "hi": hi, "n_clusters": int(n_units), "n_boot": n_boot}


def _mean(d: dict) -> float:
    return float(np.mean(list(d.values()))) if d else NAN


def _aurc(conf, utility) -> float:
    """Tie-invariant AURC, identical to metrics.risk_coverage(conf, utility)[2]; vectorised for the bootstrap."""
    conf, utility = np.asarray(conf, float), np.asarray(utility, float)
    if len(conf) == 0:
        return NAN
    order = np.argsort(-conf, kind="stable")
    c, u = conf[order], utility[order]
    g = np.concatenate([[0], np.cumsum(c[1:] != c[:-1])])
    u = (np.bincount(g, u) / np.bincount(g))[g]
    return float(np.mean(1 - np.cumsum(u) / np.arange(1, len(u) + 1)))


def _topk_share(scores, flag, k: int = K) -> float:
    """Expected share of flagged candidates in the top k under uniformly random tie-breaking (unscored last)."""
    s = np.where(np.isfinite(scores), scores, -np.inf)
    f = np.asarray(flag, float)
    o = np.argsort(-s, kind="stable")
    s, f = s[o], f[o]
    k = min(k, len(s))
    if k == 0:
        return NAN
    above, tie = s > s[k - 1], s == s[k - 1]
    return float((f[above].sum() + (k - above.sum()) * f[tie].mean()) / k)


def item_variance(values, items) -> dict:
    """Between-item SD by method of moments (P1.1): sqrt(max(0, Var(abar_i) - mean(s_i^2 / n_i))) over items with
    n_i >= 2 (s_i^2 = within-item sample variance). SD_item_raw (SD of item means over all items) still contains
    the pair-level noise / n_i and is reported only for reference."""
    v, it = np.asarray(values, float), np.asarray(items)
    ok = np.isfinite(v)
    v, it = v[ok], it[ok]
    out = {"n_items": 0, "n_items_ge2": 0, "SD_item_corrected": NAN, "var_item_means_ge2": NAN,
           "mean_within_var_over_n_ge2": NAN, "SD_item_raw": NAN,
           "item_support": {"n1": 0, "n2": 0, "n3_4": 0, "n5plus": 0}}
    if len(v) == 0:
        return out
    _, inv = np.unique(it, return_inverse=True)
    inv = inv.reshape(-1)
    n_i = np.bincount(inv)
    abar = np.bincount(inv, v) / n_i
    ss = np.bincount(inv, (v - abar[inv]) ** 2)
    ge2 = n_i >= 2
    out.update(n_items=int(len(n_i)), n_items_ge2=int(ge2.sum()),
               SD_item_raw=float(abar.std(ddof=1)) if len(abar) >= 2 else NAN,
               item_support={"n1": int((n_i == 1).sum()), "n2": int((n_i == 2).sum()),
                             "n3_4": int(((n_i >= 3) & (n_i <= 4)).sum()), "n5plus": int((n_i >= 5).sum())})
    if ge2.sum() >= 2:
        var_bar = float(abar[ge2].var(ddof=1))
        noise = float(np.mean(ss[ge2] / (n_i[ge2] - 1) / n_i[ge2]))
        out.update(SD_item_corrected=float(np.sqrt(max(0.0, var_bar - noise))), var_item_means_ge2=var_bar,
                   mean_within_var_over_n_ge2=noise)
    return out


def binormal_ensemble_auc(auc1: float, auc2: float, rho: float) -> float:
    """Binormal (equal-variance) AUC of the equal-weight sum of two standardized views with AUCs auc1, auc2 in (0, 1)
    and within-class correlation rho: d'_k = sqrt(2) Phi^-1(AUC_k), AUC = Phi(((d'_1 + d'_2) / sqrt(2 + 2 rho)) /
    sqrt(2)). rho = 1 with equal AUCs returns that AUC (a duplicated view adds nothing)."""
    d = SQRT2 * (_NORMAL.inv_cdf(auc1) + _NORMAL.inv_cdf(auc2))
    v = 2.0 + 2.0 * rho
    if v <= 0:  # rho = -1: the sum is constant within each class
        return 1.0 if d > 0 else 0.0 if d < 0 else 0.5
    return float(_NORMAL.cdf(d / math.sqrt(v) / SQRT2))


def within_class_corr(x1, x2, y) -> float:
    """Pooled within-class Pearson correlation of x1 and x2 (residuals from each class's means); NaN when either view
    is constant within every class (no within-class variation to correlate)."""
    x1, x2, y = np.asarray(x1, float), np.asarray(x2, float), np.asarray(y, int)
    r1, r2, var1, var2 = np.empty(len(y)), np.empty(len(y)), False, False
    for c in (0, 1):
        m = y == c
        if not m.any():
            continue
        p1, p2 = bool(np.ptp(x1[m]) > 0), bool(np.ptp(x2[m]) > 0)
        var1, var2 = var1 or p1, var2 or p2
        # a constant class contributes exact zeros (its mean need not reproduce the value bit for bit)
        r1[m] = x1[m] - x1[m].mean() if p1 else 0.0
        r2[m] = x2[m] - x2[m].mean() if p2 else 0.0
    if not (var1 and var2):
        return NAN
    return float(np.clip((r1 * r2).sum() / math.sqrt((r1 * r1).sum() * (r2 * r2).sum()), -1.0, 1.0))


def _zsum(x1, x2) -> np.ndarray | None:
    """Within-user z-scores (ddof 0) of both views, summed; None if a view is constant for the user."""
    s1, s2 = x1.std(), x2.std()
    if not (s1 > 0 and s2 > 0):
        return None
    return (x1 - x1.mean()) / s1 + (x2 - x2.mean()) / s2


def ensemble_null_pair(x1, x2, observed, label, users, mask, boot) -> dict:
    """Binormal ensemble-null prediction per user for the two views x1, x2 against the observed arm (rank-equivalent
    to x1 + x2) on the rows in `mask`; users need both classes and a defined within-class correlation. Context: the
    empirical within-user z-sum of the two views and the paired dUAUC(observed - z-sum) on the users with both."""
    pred, obs, obs_all, a1s, a2s, rhos, zs = {}, {}, {}, {}, {}, {}, {}
    n_both_classes = n_rho_undefined = n_clipped = 0
    for u, idx in users.items():
        idx = idx[mask[idx]]
        y = label[idx]
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        n_both_classes += 1
        v1, v2 = x1[idx], x2[idx]
        obs_all[u] = auroc(observed[idx], y)
        z = _zsum(v1, v2)
        if z is not None:
            zs[u] = auroc(z, y)
        rho = within_class_corr(v1, v2, y)
        if not np.isfinite(rho):
            n_rho_undefined += 1
            continue
        a1, a2 = auroc(v1, y), auroc(v2, y)
        eps = 0.5 / (n_pos * n_neg)  # half a pair inside (0, 1): Phi^-1 is infinite at 0 and 1
        c1, c2 = min(max(a1, eps), 1 - eps), min(max(a2, eps), 1 - eps)
        n_clipped += int(c1 != a1 or c2 != a2)
        pred[u], obs[u] = binormal_ensemble_auc(c1, c2, rho), obs_all[u]
        a1s[u], a2s[u], rhos[u] = a1, a2, rho
    r = np.array(list(rhos.values()), float)
    d = paired_bootstrap(obs, pred, **boot)
    d["note"] = "observed arm minus its binormal ensemble-null prediction, paired over users"
    dz = paired_bootstrap(obs_all, zs, **boot)
    dz["note"] = ("context, not gating: observed arm minus the empirical within-user z-sum of the two views, paired "
                  "over users")
    return {"n_rows": int(mask.sum()), "n_users_both_classes": n_both_classes, "n_users": len(pred),
            "n_users_rho_undefined": n_rho_undefined, "n_users_auc_clipped": n_clipped,
            "UAUC_view1": _mean(a1s), "UAUC_view2": _mean(a2s),
            "rho_within_class": {"mean": float(r.mean()) if len(r) else NAN,
                                 "median": float(np.median(r)) if len(r) else NAN},
            "predicted_UAUC": _mean(pred), "observed_UAUC": _mean(obs),
            "z_sum_UAUC": _mean(zs), "n_users_z_sum": len(zs), "dUAUC_observed_minus_predicted": d,
            "dUAUC_observed_minus_z_sum": dz, "_z_sum_per_user": zs}


def analyze(sc: dict, panel_rows: list, swap: dict | None = None, nohist: dict | None = None,
            ref_ranks: dict | None = None, base_q: str | None = None, n_boot: int = 2000, seed: int = 0) -> dict:
    keys, user, item, label, Lq = sc["keys"], sc["user"], sc["item"], sc["label"], sc["L"]
    qs = set(Lq)
    panel = {str(r.get("source_event_id", r["user_id"])): r for r in panel_rows}
    next_item = any("positive_item_index" in r for r in panel_rows)
    base_q = base_q or ("next" if next_item and "next" in qs else "like" if "like" in qs else "next")
    if base_q not in qs:
        raise SystemExit(f"--base_q {base_q!r} not among scored questions {sorted(qs)}")
    n = len(keys)
    boot = dict(n_boot=n_boot, seed=seed)

    def L(q):
        return Lq[q]

    def corr(x, y, clusters=None) -> dict:
        """Spearman with a cluster bootstrap over `clusters` (users), or over units (items) when None."""
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            return {"est": NAN, "lo": NAN, "hi": NAN, "n": int(ok.sum())}
        xs, ys = x[ok], y[ok]
        stat = lambda idx: spearman(xs[idx], ys[idx])  # noqa: E731
        out = (unit_bootstrap(stat, int(ok.sum()), **boot) if clusters is None
               else cluster_bootstrap(stat, np.asarray(clusters)[ok], **boot))
        out["n"] = int(ok.sum())
        return out

    # --- panel join (both directions): per-pair popularity columns, next-item positives ---
    pop, pop_prior, head = np.full(n, NAN), np.full(n, NAN), np.full(n, NAN)
    join = Counter()
    if len(panel) < len(panel_rows):
        join["panel_duplicate_event_ids"] += len(panel_rows) - len(panel)
    scored = defaultdict(set)
    for ev, c in keys:
        scored[ev].add(c)
    for ev, r in panel.items():
        if ev not in scored:
            join["panel_events_without_scores"] += 1
            continue
        miss = len(set(range(len(r["candidate_item_ids"]))) - scored[ev])
        if miss:
            join["panel_candidates_without_score_rows"] += miss
    for j, (ev, c) in enumerate(keys):
        r = panel.get(ev)
        if r is None:
            join["score_rows_without_panel_event"] += 1
            continue
        ids = r["candidate_item_ids"]
        if c >= len(ids) or str(ids[c]) != str(item[j]):
            join["score_rows_item_mismatch"] += 1
            continue
        if "candidate_popularity" in r:
            pop[j] = _num(r["candidate_popularity"][c])
        if "candidate_popularity_prior" in r:
            pop_prior[j] = _num(r["candidate_popularity_prior"][c])
        if "candidate_popularity_groups" in r:
            head[j] = float(str(r["candidate_popularity_groups"][c]).strip().lower() == "head")
    if join:
        print(f"WARNING panel join problems: {dict(join)}")
    users = _groups(user)
    item_uniq, item_inv = np.unique(item, return_inverse=True)
    item_inv = item_inv.reshape(-1)
    res = {"panel_type": "next_item" if next_item else "rated", "base_question": base_q, "questions": sorted(qs),
           "n_rows": n, "n_users": len(users), "n_items": int(len(item_uniq)), "n_boot": n_boot, "seed": seed,
           "censoring": {"scores": sc["censoring"]}, "panel_join": dict(join), "arms": {}}

    # --- arms ---
    arms = {"raw": L(base_q)}
    if "like" in qs:
        arms["raw_like"] = L("like")
    if {"like", "dislike"} <= qs:
        arms["mirror"] = L("like") - L("dislike")
    if {"like", "like_para"} <= qs:
        arms["placebo"] = (L("like") + L("like_para")) / 2
    pi_item = None
    if swap is not None:
        res["censoring"]["swap"] = swap["censoring"]
        pi = swap["pi"].get("like")
        if pi and "like" in qs:
            pi_item = np.array([pi.get(i, NAN) for i in item_uniq.tolist()])
            arms["evidence"] = L("like") - pi_item[item_inv]
            nd = np.array(list(swap["n_donors"]["like"].values()))
            res["swap"] = {"n_items_with_prior": len(pi), "n_panel_items_without_prior": int(np.isnan(pi_item).sum()),
                           "donors_per_item_min": int(nd.min()), "donors_per_item_mean": float(nd.mean())}
        else:
            res["swap"] = {"note": "no finite like rows in the swap file (or like not scored): evidence arm skipped"}
    if nohist is not None:
        res["censoring"]["nohist"] = nohist["censoring"]
        if "like" in nohist["L"] and "like" in qs:
            acc0 = defaultdict(list)
            for it, v in zip(nohist["item"], nohist["L"]["like"]):
                if np.isfinite(v):
                    acc0[it].append(v)
            p0 = np.array([float(np.mean(acc0[i])) if acc0.get(i) else NAN for i in item_uniq.tolist()])
            arms["pmi_nohist"] = L("like") - p0[item_inv]
            res["nohist"] = {"n_panel_items_without_prior": int(np.isnan(p0).sum())}
        else:
            res["nohist"] = {"note": "like not scored in the nohist file: pmi_nohist arm skipped"}

    finite = {a: np.isfinite(s) for a, s in arms.items()}
    for name, s in arms.items():
        ok = finite[name]
        pu = _uauc(s, label, users, ok)
        res["arms"][name] = {"UAUC": _mean(pu), "n_users_uauc": len(pu), "AUC_global": auroc(s[ok], label[ok]),
                             "n_rows": int(ok.sum())}
    res["primary_endpoint"] = "dUAUC_mirror_minus_placebo"
    for x, y in UAUC_PAIRS:
        if x in arms and y in arms:
            m = finite[x] & finite[y]  # same pairs for both arms
            d = paired_bootstrap(_uauc(arms[x], label, users, m), _uauc(arms[y], label, users, m), **boot)
            d["n_rows"] = int(m.sum())
            res[f"dUAUC_{x}_minus_{y}"] = d

    # --- amendment 2 G7: correlation-matched ensemble null for mirror (sanity check: the placebo pair) ---
    if "mirror" in arms:
        en = {"definition": "per user (rows finite in both views, both classes): AUC_k of each view, rho = pooled "
                            "within-class Pearson correlation of the two logits; d'_k = sqrt(2) Phi^-1(AUC_k) with "
                            "AUC_k clipped to [0.5/(n_pos n_neg), 1 - 0.5/(n_pos n_neg)]; predicted UAUC of the "
                            "equal-weight standardized sum = Phi(((d'_1 + d'_2) / sqrt(2 + 2 rho)) / sqrt(2)); "
                            "users with an undefined rho (a view constant within both classes) are dropped and "
                            "counted. dUAUC = observed arm - prediction, paired user bootstrap. z_sum_UAUC: UAUC of "
                            "the empirical within-user z-sum of the two views (context, not gating)",
              "gating_key": "dUAUC_mirror_minus_ensemble_null (scripts/sigir/pilot1_gate.py reads its lo)",
              "caveats": ENSEMBLE_NULL_CAVEATS,
              "mirror_pair": ensemble_null_pair(L("like"), -L("dislike"), arms["mirror"], label, users,
                                                finite["mirror"], boot)}
        en["mirror_pair"]["views"] = ["like", "-dislike"]
        en["dUAUC_mirror_minus_ensemble_null"] = en["mirror_pair"]["dUAUC_observed_minus_predicted"]
        en["dUAUC_mirror_minus_zsum_null"] = en["mirror_pair"]["dUAUC_observed_minus_z_sum"]   # context
        z_m = en["mirror_pair"].pop("_z_sum_per_user")
        if "placebo" in arms:
            en["placebo_pair"] = ensemble_null_pair(L("like"), L("like_para"), arms["placebo"], label, users,
                                                    finite["placebo"], boot)
            en["placebo_pair"]["views"] = ["like", "like_para"]
            en["dUAUC_placebo_minus_ensemble_null"] = en["placebo_pair"]["dUAUC_observed_minus_predicted"]
            en["dUAUC_placebo_minus_zsum_null"] = en["placebo_pair"]["dUAUC_observed_minus_z_sum"]   # context
            z_p = en["placebo_pair"].pop("_z_sum_per_user")
            en["dUAUC_zsum_mirror_minus_zsum_placebo"] = paired_bootstrap(z_m, z_p, **boot)
        res["ensemble_null"] = en

    if next_item:
        _next_item(res, keys, user, label, arms, head, panel, ref_ranks, boot)

    # --- P1.1 acquiescence a, valence v, prior pi vs log-popularity (pair level: user-cluster CI; item level:
    # item bootstrap). Popularity = all-time candidate_popularity; candidate_popularity_prior is a robustness block.
    has_ad = {"like", "dislike"} <= qs
    if has_ad:
        acq = (L("like") + L("dislike")) / 2
        val = (L("like") - L("dislike")) / 2
        ca = np.full(n, NAN)  # centred a; NaN for users with < 2 finite a (centring leaves a constant 0)
        n_u = []
        for idx in users.values():
            idx = idx[np.isfinite(acq[idx])]
            if len(idx) >= 2:
                ca[idx] = acq[idx] - acq[idx].mean()
                n_u.append(len(idx))
        n_ge2, pool = len(n_u), np.isfinite(ca)
        sq, sq_users = ca[pool] ** 2, user[pool]
        nan_ci = {"lo": NAN, "hi": NAN}
        sd_ci = cluster_bootstrap(lambda idx: float(np.sqrt(sq[idx].mean())), sq_users, **boot) if n_ge2 else nan_ci
        # each resample draws n_ge2 users, so the within-user df is len(idx) - n_ge2
        sd_df_ci = (cluster_bootstrap(lambda idx: float(np.sqrt(sq[idx].sum() / (len(idx) - n_ge2))), sq_users,
                                      **boot) if n_ge2 else nan_ci)
        a_item = _item_means(ca, item_inv, len(item_uniq))
        v_item = _item_means(val, item_inv, len(item_uniq))
        res["acquiescence"] = {
            "definition": "a(u,i) = (L_like + L_dislike)/2 centred within user; SD_pair = sqrt(mean a_c^2) over "
                          "pairs of users with >= 2 finite a (pooled within-user SD, gating; biased low by "
                          "sqrt(1 - 1/n_u)); SD_pair_df = sqrt(sum a_c^2 / (N - U)), the df-corrected pooled SD "
                          "(reported; gate sensitivity); SD_item_corrected = method-of-moments between-item SD on "
                          "items with n_i >= 2 (reported, not gating)",
            "n_pairs": int(pool.sum()), "n_users_ge2": n_ge2,
            "n_per_user": ({"min": int(min(n_u)), "median": float(np.median(n_u)), "mean": float(np.mean(n_u)),
                            "max": int(max(n_u))} if n_u else {}),
            "global_mean_a": float(np.nanmean(acq)) if np.isfinite(acq).any() else NAN,
            "SD_pair": float(np.sqrt(sq.mean())) if n_ge2 else NAN,
            "SD_pair_ci": {"lo": sd_ci["lo"], "hi": sd_ci["hi"]},
            "SD_pair_df": float(np.sqrt(sq.sum() / (len(sq) - n_ge2))) if n_ge2 else NAN,
            "SD_pair_df_ci": {"lo": sd_df_ci["lo"], "hi": sd_df_ci["hi"]},
            **item_variance(ca, item)}
        res["valence"] = {"definition": "v(u,i) = (L_like - L_dislike)/2 (not centred)",
                          "global_mean_v": float(np.nanmean(val)) if np.isfinite(val).any() else NAN}
    if pi_item is not None:
        res["prior"] = {"definition": "pi(i) = mean finite like-logit of item i under the swap donors' histories"}
    for tag, p in (("", pop), ("prior", pop_prior)):
        if not np.isfinite(p).any() or not (has_ad or pi_item is not None):
            continue
        lp = np.log1p(p)
        lp_item = _item_means(lp, item_inv, len(item_uniq))
        block = {}
        if has_ad:
            block.update(corr_a_logpop=corr(ca, lp, user), corr_a_logpop_item=corr(a_item, lp_item),
                         corr_v_logpop=corr(val, lp, user), corr_v_logpop_item=corr(v_item, lp_item))
        if pi_item is not None:
            block["corr_pi_logpop"] = corr(pi_item, lp_item)
        if tag:
            res["popularity_prior_robustness"] = {
                "popularity": "log1p(candidate_popularity_prior): events on the item strictly before the candidate",
                **block}
            continue
        if has_ad:
            res["acquiescence"].update(popularity="log1p(candidate_popularity): all-time category event count",
                                       corr_a_logpop=block["corr_a_logpop"],
                                       corr_a_logpop_item=block["corr_a_logpop_item"])
            res["valence"].update(corr_v_logpop=block["corr_v_logpop"],
                                  corr_v_logpop_item=block["corr_v_logpop_item"])
        if pi_item is not None:
            res["prior"]["corr_pi_logpop"] = block["corr_pi_logpop"]

    if next_item:
        na = {"applicable": False, "reason": "next-item panel: 1-of-N labels are not a calibration target"}
        res["calibration"] = res["confident_error"] = na
        return res

    # --- P1.2 calibration ---
    cal = {"applicable": True, "raw_scale": {}, "platt": {}}
    raw_scale = {a: arms[a] for a in ("raw", "raw_like", "placebo") if a in arms}
    if "mirror" in arms:
        raw_scale["mirror_half"] = arms["mirror"] / 2
    for a in RAW_SCALE_ARMS:
        if a in raw_scale:
            ok = np.isfinite(raw_scale[a])
            cal["raw_scale"][a] = _calib(sigmoid(raw_scale[a][ok]), label[ok])
    fit_users = user_halves(list(users), seed)
    in_fit = np.array([u in fit_users for u in user], bool)
    cal["platt_split"] = {"n_users_fit": len(fit_users), "n_users_eval": len(users) - len(fit_users), "seed": seed}
    for a in PLATT_ARMS:
        if a not in arms:
            continue
        fit, ev = finite[a] & in_fit, finite[a] & ~in_fit
        if not ev.any() or not (0 < label[fit].sum() < fit.sum()):
            cal["platt"][a] = {"note": "insufficient rows or a single class in a split half"}
            continue
        ab = platt_fit(arms[a][fit], label[fit])
        cal["platt"][a] = {"a": ab[0], "b": ab[1], "n_fit": int(fit.sum()),
                           **_calib(platt_apply(arms[a][ev], ab), label[ev])}
    res["calibration"] = cal

    # --- P1.3 confident-error detection: one target (errors of the raw like decision) for every detector ---
    if "like" not in qs:
        res["confident_error"] = {"applicable": False, "reason": "like not scored"}
        return res
    ll = L("like")
    base = np.isfinite(ll)
    err = ((sigmoid(np.where(base, ll, 0.0)) >= 0.5) != (label == 1)).astype(int)
    mags = {"mag_raw": np.abs(ll)}
    for a in ("mirror", "evidence", "placebo"):
        if a in arms:
            mags[f"mag_{a}"] = np.abs(arms[a])
    dets = {d: (-m, m) for d, m in mags.items()}  # (error score, confidence)
    if "mirror" in arms:
        # decision-space sign: logit >= 0 is the Yes side (sigmoid >= 0.5), as in the error target
        dis = np.where(finite["mirror"] & base, ((arms["mirror"] >= 0) != (ll >= 0)).astype(float), NAN)
        dets["disagree"] = (dis, 1 - dis)
    ce = {"applicable": True, "target": "(sigmoid(L_like) >= 0.5) != label, identical for every detector",
          "scoring": "AUROC(error score, is_error): error score = -|x| for magnitude detectors, disagree as is "
                     "(higher = better); AURC = metrics.risk_coverage(conf = |x| or 1 - disagree, 1 - is_error) "
                     "(lower = better); deltas = detector - mag_raw on the same rows, user-cluster bootstrap",
          "n_rows": int(base.sum()), "n_errors": int(err[base].sum()),
          "error_rate": float(err[base].mean()) if base.any() else NAN, "detectors": {}}
    for d, (es, cf) in dets.items():
        rows = base & np.isfinite(es)
        e_r, es_r, cf_r, mr_r = err[rows], es[rows], cf[rows], np.abs(ll)[rows]
        out = {"n_rows": int(rows.sum()), "n_errors": int(e_r.sum()), "AUROC": auroc(es_r, e_r),
               "AURC": risk_coverage(cf_r, 1 - e_r)[2] if rows.any() else NAN}
        if d != "mag_raw" and rows.sum() >= 2:
            out["dAUROC_vs_mag_raw"] = cluster_bootstrap(
                lambda idx: auroc(es_r[idx], e_r[idx]) - auroc(-mr_r[idx], e_r[idx]), user[rows], **boot)
            out["dAURC_vs_mag_raw"] = cluster_bootstrap(
                lambda idx: _aurc(cf_r[idx], 1 - e_r[idx]) - _aurc(mr_r[idx], 1 - e_r[idx]), user[rows], **boot)
        ce["detectors"][d] = out
    res["confident_error"] = ce
    return res


def _calib(p, y) -> dict:
    if len(y) == 0:
        return {"ECE": NAN, "ECE_adaptive": NAN, "Brier": NAN, "n": 0}
    return {"ECE": ece(p, y, 15), "ECE_adaptive": ece(p, y, 15, adaptive=True), "Brier": brier(p, y), "n": int(len(y))}


def _event_metrics(arr, pos: int) -> dict:
    """NDCG@10 / HR@10 / MRR at the tie-aware expected rank (stats.tie_aware_rank, C1) and, as *_tie_exact, their
    exact expectation over uniformly random orders of the positive's tie group (the plug-in counts a tie group
    straddling rank 10 as a miss). An unscored positive ranks last; unscored negatives rank below the positive."""
    r = tie_aware_rank(arr, pos)
    sp = arr[pos]
    g, t = (int(np.sum(arr > sp)), int(np.sum(arr == sp))) if np.isfinite(sp) else (len(arr) - 1, 1)
    rr = np.arange(g + 1, g + t + 1, dtype=float)  # mean(rr) == r
    return {"NDCG@10": float(ndcg_from_rank(np.array([r]), K)[0]), "HR@10": float(r <= K), "MRR": 1.0 / r,
            "NDCG@10_tie_exact": float(ndcg_from_rank(rr, K).mean()), "HR@10_tie_exact": float(np.mean(rr <= K)),
            "MRR_tie_exact": float(np.mean(1.0 / rr))}


def _rank_metrics(rank: float) -> dict:
    return {"NDCG@10": float(ndcg_from_rank(np.array([rank]), K)[0]), "HR@10": float(rank <= K), "MRR": 1.0 / rank}


def _next_item(res, keys, user, label, arms, head, panel, ref_ranks, boot) -> None:
    """P1.4: tie-aware per-event ranks -> per-user NDCG@10 / HR@10 / MRR, paired dNDCG@10, C-CRP reference join."""
    events = _groups(np.array([k[0] for k in keys], dtype=object))
    cand = np.array([k[1] for k in keys], int)
    ev_user, info, bad = {}, {}, Counter()
    for ev, idx in events.items():
        r = panel.get(ev)
        if r is None or "positive_item_index" not in r:
            bad["events_without_positive_item_index"] += 1
            continue
        nc, pos = len(r["candidate_item_ids"]), int(r["positive_item_index"])
        ci = cand[idx]
        if ci.max() >= nc:
            bad["events_cand_idx_out_of_range"] += 1
            continue
        if label[idx].sum() != 1 or not (label[idx][ci == pos] == 1).any():
            bad["events_label_vs_positive_item_index_mismatch"] += 1
        ev_user[ev] = user[idx[0]]
        info[ev] = (idx, ci, nc, pos)
    metrics, full = {}, {}  # arm -> {metric: {event: value}}; arm -> {event: scores over all candidates}

    def per_user(d: dict, keep=None) -> dict:
        acc = defaultdict(list)
        for ev, v in d.items():
            if keep is None or ev in keep:
                acc[ev_user[ev]].append(v)
        return {u: float(np.mean(v)) for u, v in acc.items()}

    for name, s in arms.items():
        m, unscored, partial, share, full[name] = defaultdict(dict), 0, 0, {}, {}
        for ev, (idx, ci, nc, pos) in info.items():
            arr = np.full(nc, NAN)
            arr[ci] = s[idx]
            full[name][ev] = arr
            unscored += not np.isfinite(arr[pos])
            partial += np.isfinite(arr[pos]) and not np.isfinite(arr).all()  # unscored negatives rank below
            for k, v in _event_metrics(arr, pos).items():
                m[k][ev] = v
            h = np.full(nc, NAN)
            h[ci] = head[idx]
            if np.isfinite(h).all():
                share[ev] = _topk_share(arr, h)
        metrics[name] = m
        res["arms"][name].update({k: _mean(per_user(m[k])) for k in METRICS + EXACT})
        res["arms"][name].update(n_events=len(info), n_events_positive_unscored=int(unscored),
                                 n_events_negative_unscored=int(partial))
        if share:
            res["arms"][name]["top10_head_share"] = _mean(per_user(share))
    res["next_item"] = {"rank": "expected 1-based rank under uniformly random tie-breaking (stats.tie_aware_rank); "
                                "per arm, an unscored positive ranks last and unscored negatives below the positive; "
                                "*_tie_exact = exact expectation over the positive's tie group",
                        "paired": "dNDCG@10 between arms ranks both arms on the candidates finite in both (as dUAUC); "
                                  "events whose positive is unscored in either arm are dropped and counted",
                        "n_events": len(info), **bad}
    for x, y in NDCG_PAIRS:
        if x not in full or y not in full:
            continue
        mx, my = defaultdict(dict), defaultdict(dict)
        cnt = {"n_events_positive_unscored_dropped": 0, "n_events_with_candidates_dropped": 0}
        for ev, (_, _, _, pos) in info.items():
            ax, ay = full[x][ev], full[y][ev]
            ok = np.isfinite(ax) & np.isfinite(ay)
            if not ok[pos]:
                cnt["n_events_positive_unscored_dropped"] += 1
                continue
            cnt["n_events_with_candidates_dropped"] += int(not ok.all())
            p = int(ok[:pos].sum())
            for k, v in _event_metrics(ax[ok], p).items():
                mx[k][ev] = v
            for k, v in _event_metrics(ay[ok], p).items():
                my[k][ev] = v
        d = paired_bootstrap(per_user(mx["NDCG@10"]), per_user(my["NDCG@10"]), **boot)
        d.update(n_events=len(mx["NDCG@10"]), **cnt)
        d["tie_exact"] = paired_bootstrap(per_user(mx["NDCG@10_tie_exact"]), per_user(my["NDCG@10_tie_exact"]),
                                          **boot)
        res[f"dNDCG@10_{x}_minus_{y}"] = d
    if ref_ranks is None:
        return
    joined = {}
    rb = {"ref_events_not_in_scored_panel": 0, "num_candidates_mismatch": 0, "user_id_mismatch": 0}
    for ev, (u, rk, nc) in ref_ranks.items():
        if ev not in info:
            rb["ref_events_not_in_scored_panel"] += 1
        elif nc and nc != info[ev][2]:
            rb["num_candidates_mismatch"] += 1
        elif u != "" and str(u) != str(ev_user[ev]):
            rb["user_id_mismatch"] += 1  # excluded from the join
        else:
            joined[ev] = rk
    m = defaultdict(dict)
    for ev, rk in joined.items():
        for k, v in _rank_metrics(rk).items():
            m[k][ev] = v
    res["arms"]["ccrp"] = {k: _mean(per_user(m[k])) for k in METRICS}
    res["arms"]["ccrp"]["n_events"] = len(joined)
    res["ref_ranks"] = {"n_ref_rows": len(ref_ranks), "n_scored_panel_events": len(info), "n_joined": len(joined),
                        **rb, "arms_on_joined_events": {
                            a: {k: _mean(per_user(metrics[a][k], joined)) for k in METRICS + EXACT} for a in arms}}
    # each arm - C-CRP v3 on the same events (P1.4): C-CRP ranks as exported, each arm on its own candidate support
    cc = per_user(m["NDCG@10"])
    for a in arms:
        d = paired_bootstrap(per_user(metrics[a]["NDCG@10"], joined), cc, **boot)
        d["n_events"] = len(joined)
        d["tie_exact"] = paired_bootstrap(per_user(metrics[a]["NDCG@10_tie_exact"], joined), cc, **boot)
        res[f"dNDCG@10_{a}_minus_ccrp"] = d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--swap", default=None, help="swap_prior.csv.gz (user-marginalised like prior pi(i))")
    ap.add_argument("--nohist", default=None, help="scores.csv.gz of the same panel scored with --hist_len 0")
    ap.add_argument("--base_q", default=None, help="question of the `raw` arm (default: next on next-item panels when scored, else like)")
    ap.add_argument("--ref_ranks", default=None, help="reference positive ranks (e.g. C-CRP v3), joined as arm ccrp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    panel_rows = [json.loads(line) for line in open(a.panel, encoding="utf-8") if line.strip()]
    res = analyze(load_scores(a.scores), panel_rows,
                  swap=load_swap(a.swap) if a.swap else None,
                  nohist=load_scores(a.nohist) if a.nohist else None,
                  ref_ranks=load_ref_ranks(a.ref_ranks) if a.ref_ranks else None,
                  base_q=a.base_q, n_boot=a.n_boot, seed=a.seed)
    res["inputs"] = {"scores": a.scores, "panel": a.panel, "swap": a.swap, "nohist": a.nohist,
                     "ref_ranks": a.ref_ranks}
    text = json.dumps(strict_json(res), indent=2, allow_nan=False)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
