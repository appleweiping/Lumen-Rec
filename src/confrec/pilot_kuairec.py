"""Pilot 2 (B1/A7) analysis: MAR vs simulated-logged calibration on the SAME KuaiRec pairs and scores.

Binding spec: idea-stage/PREREG_AMENDMENT_1.md section P2. It replaces the registered "unobserved in big = negative"
view of triage_verdict.md section 3.2, which is identically 0 because KuaiRec removes every small-matrix pair from
the big matrix.
  MAR label     1[watch_ratio >= thr] for thr in --thrs (CPU sensitivity on the stored watch ratio)
  exposure      e(u,v) = min(1, c * n_u * n_v^alpha), alpha in --alphas; c by bisection so that the mean of e over
                panel pairs equals D, the big-matrix density over small-matrix videos (panel meta sidecar)
  logged label  O * MAR, O ~ Bernoulli(e); --n_draws draws, Generator seeded by (seed, draw, alpha, thr)
Endpoints per (thr, alpha): point = mean over draws of the frozen metrics; CI = user-bootstrap replicates pooled
across draws (max(50, n_boot // n_draws) per draw, user resamples shared by all cells). Replicates are evaluated as
user-count-weighted sums that reproduce rank_bins / ece / bias_index on the resampled rows (tested), which keeps
the 60k-pair default within minutes on CPU.
  (1) share: MAR-positive rate among top-decile-confidence pairs that are logged-negative (decile on average ranks
      of the logit over all analysed pairs); base: MAR-positive rate among all logged negatives; lift = share/base
  (2) gap = ECE(p, logged) - ECE(p, MAR), 15 equal-width bins, identical pairs and p = sigmoid(logit).
      Diagnostic (not in the decision): |gap| <= gap_ceiling = mean(MAR * (1 - O)) = MAR rate - logged rate, with
      gap = +ceiling when p >= the MAR rate in every bin; gap_normalized = gap / ceiling lies in [-1, 1]
  (3) residual-adjusted Bias Index (metrics.bias_index) by n_v quintile on average ranks: head = top, tail = bottom;
      endpoint3.valid is false (and a warning printed) unless the head and tail groups each hold 15-25% of pairs
Gate: UAUC_MAR (mean per-user AUC over users with both classes) at thr=2.0, user-bootstrap CI; pooled AUC secondary.
Decision at thr=2.0, alpha=1: NULL iff UAUC_MAR < 0.58 (zero-shot; the LoRA retry moved to M3); else POSITIVE iff
share >= 0.20, the gap CI excludes 0 and the endpoint-1 lift CI lower bound > 1; NEGATIVE iff share < 0.10 and the
gap CI includes 0; else AMBIGUOUS. P2 plus its 2026-10-02 pre-data addendum (the lift condition): under simulated
thinning the gap CI rarely includes 0, so the gap alone cannot carry POSITIVE.
Rows with non-finite logits (censored 2/3) are dropped and counted per censored code; censored=1 rows are kept.

    python -m src.confrec.pilot_kuairec --scores <dir>/scores.csv.gz --panel kuairec.jsonl --out pilot_kuairec.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.confrec.metrics import auroc, bias_index, ece
from src.confrec.stats import cluster_bootstrap, percentile_ci, rank_bins, sigmoid, strict_json

GROUPS = {"0": "tail", "1": "mid", "2": "head"}
PRIMARY = (2.0, 1.0)            # (thr, alpha)
UAUC_NULL = 0.58
GROUP_SHARE_OK = (0.15, 0.25)   # head / tail n_v quintile share for a usable endpoint 3
CAVEATS = [
    "logged = O * MAR thins the MAR positives, so |gap| <= gap_ceiling = MAR rate - logged rate, with gap = +ceiling "
    "whenever p >= the MAR rate in every ECE bin; 'gap CI includes 0' needs p midway between logged and MAR "
    "calibration, so NEGATIVE is rarely reachable and POSITIVE reduces to share >= 0.20. Read the decision with "
    "lift (endpoint 1) and gap_normalized. The decision rule itself is PREREG_AMENDMENT_1 P2 verbatim."]


def load_scores(path, question: str = "like") -> tuple[pd.DataFrame, dict]:
    """Finite-logit rows of one question + a report with counts per censored code."""
    df = pd.read_csv(path, dtype={"source_event_id": str, "user_id": str, "item_id": str})
    df = df[df["question"] == question]
    dup = df.duplicated(["source_event_id", "cand_idx"], keep="last")
    df = df[~dup]
    logit = pd.to_numeric(df["logit"], errors="coerce").to_numpy(float)
    fin = np.isfinite(logit)

    def counts(m):
        if "censored" not in df:
            return None
        c = pd.to_numeric(df["censored"], errors="coerce").fillna(-1).astype(int)[m]
        return {str(k): int(v) for k, v in sorted(c.value_counts().items())}

    rep = {"question": question, "n_rows": int(len(df)), "n_duplicates_dropped": int(dup.sum()),
           "n_nonfinite_dropped": int((~fin).sum()), "censored_counts": counts(np.ones(len(df), bool)),
           "censored_counts_kept": counts(fin)}
    out = df.loc[fin, ["source_event_id", "cand_idx", "item_id"]].copy()
    out["cand_idx"] = out["cand_idx"].astype(int)
    out["logit"] = logit[fin]
    if "label" in df:
        out["label"] = df.loc[fin, "label"].to_numpy()
    return out.reset_index(drop=True), rep


def load_panel(path) -> tuple[pd.DataFrame, dict]:
    """One row per panel pair (panel order) + the builder's meta sidecar ({} if absent)."""
    cols = {k: [] for k in ("source_event_id", "user_id", "cand_idx", "item_id", "watch_ratio", "n_v", "n_u")}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "user_exposure" not in r or "candidate_watch_ratio" not in r:
            raise ValueError("panel lacks user_exposure / candidate_watch_ratio: rebuild with build_kuairec_panel")
        n = len(r["candidate_item_ids"])
        cols["source_event_id"] += [r["source_event_id"]] * n
        cols["user_id"] += [str(r["user_id"])] * n
        cols["cand_idx"] += list(range(n))
        cols["item_id"] += [str(x) for x in r["candidate_item_ids"]]
        cols["watch_ratio"] += [float(x) for x in r["candidate_watch_ratio"]]
        cols["n_v"] += [float(x) for x in r["candidate_popularity"]]
        cols["n_u"] += [float(r["user_exposure"])] * n
    mp = Path(path).with_suffix(".meta.json")
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    return pd.DataFrame(cols), meta


def solve_c(w, target: float, iters: int = 200) -> float:
    """c such that mean(min(1, c * w)) == target (bisection; the mean is continuous and non-decreasing in c)."""
    w = np.asarray(w, float)
    pos = w[w > 0]
    if not (len(pos) and 0 < target < len(pos) / len(w)):
        raise ValueError(f"target density {target} not attainable (share of w > 0 is {len(pos) / max(1, len(w))})")
    lo, hi = 0.0, 1.0 / pos.min()                         # mean at hi = share of w > 0 >= target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if np.minimum(1.0, mid * w).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def draw_exposure(e, seed: int, d: int, alpha: float, thr: float) -> np.ndarray:
    ss = np.random.SeedSequence([int(seed), int(d), int(round(alpha * 1000)), int(round(thr * 1000))])
    return np.random.default_rng(ss).random(len(e)) < np.asarray(e, float)


def pop_groups(n_v) -> np.ndarray:
    """0 tail / 1 mid / 2 head from n_v quintiles on average ranks (ties share a bin)."""
    b = rank_bins(n_v, 5)
    return np.where(b == 4, 2, np.where(b == 0, 0, 1))


def endpoints(p, mar, logged, top, ece_mar: float | None = None) -> dict:
    """Endpoints (1)-(2) on one set of pairs; both ECEs use the identical p and pairs."""
    neg = ~np.asarray(logged, bool)
    tn = top & neg
    share = float(mar[tn].mean()) if tn.any() else float("nan")
    base = float(mar[neg].mean()) if neg.any() else float("nan")
    em = ece(p, mar) if ece_mar is None else ece_mar
    el = ece(p, logged)
    ceil = float((mar * neg).mean()) if len(neg) else float("nan")       # unexposed MAR positives
    return {"share": share, "base": base, "lift": share / base if base > 0 else float("nan"),
            "gap": el - em, "gap_ceiling": ceil, "gap_normalized": (el - em) / ceil if ceil > 0 else float("nan"),
            "ece_logged": el, "n_top_logged_neg": int(tn.sum())}


def _bias(p, y, g) -> dict:
    return {GROUPS[k]: v for k, v in bias_index(p, y, g, n_bins=10, adjust=True).items()}


# Weighted replicas for the user bootstrap. A user resample is the multiset that repeats each pair w_i times
# (w_i = how often its user was drawn), so every statistic is a weighted sum over the ORIGINAL pairs. Each helper
# reproduces the frozen function evaluated on the resampled rows (tests/test_confrec_kuairec.py checks this).
def _tie_index(x) -> tuple[np.ndarray, int]:
    _, inv = np.unique(np.asarray(x, float), return_inverse=True)
    inv = inv.reshape(-1)
    return inv, int(inv.max()) + 1 if len(inv) else 0


def _wrank_bins(tie, n_ties: int, w, n_bins: int) -> np.ndarray:
    """stats.rank_bins of the weighted multiset, per original row (average rank = L + (W + 1) / 2 per tie group)."""
    W = np.bincount(tie, weights=w, minlength=n_ties)
    r = np.cumsum(W) - W + (W + 1) / 2
    return np.minimum((n_bins * (r - 0.5) / W.sum()).astype(int), n_bins - 1)[tie]


def _ece_bins(p, n_bins: int = 15) -> np.ndarray:
    """metrics.ece equal-width bin of each confidence."""
    edges = np.linspace(0, 1, n_bins + 1)
    edges[-1] = 1 + 1e-9
    return np.clip(np.searchsorted(edges, np.asarray(p, float), side="right") - 1, 0, n_bins - 1)


def _ece_reps(C, inv, eb, p, y, n_bins: int = 15) -> np.ndarray:
    """metrics.ece on every resample at once (rows of C = user multiplicities): sum_b |sum_{i in b} (p - y)| / N."""
    U = C.shape[1]
    A = np.bincount(inv * n_bins + eb, weights=p - np.asarray(y, float), minlength=U * n_bins).reshape(U, n_bins)
    return np.abs(C @ A).sum(1) / (C @ np.bincount(inv, minlength=U))


def _sum_reps(C, inv, x) -> np.ndarray:
    """sum of x over the resampled rows, for every resample."""
    return C @ np.bincount(inv, weights=np.asarray(x, float), minlength=C.shape[1])


def _base_reps(C, inv, mar, logged) -> np.ndarray:
    """MAR-positive rate among logged negatives on every resample."""
    neg = ~np.asarray(logged, bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _sum_reps(C, inv, neg * mar) / _sum_reps(C, inv, neg)


def _ceiling_reps(C, inv, mar, logged) -> np.ndarray:
    """gap_ceiling = mean(MAR * (1 - logged)) on every resample."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return _sum_reps(C, inv, ~np.asarray(logged, bool) * mar) / _sum_reps(C, inv, np.ones(len(inv)))


def _wshare(wt, mar_t, logged_t) -> float:
    """Share on the top decile (wt = weights of its rows): sum w*mar*(1-lg) / sum w*(1-lg); lg implies mar."""
    wl = wt[logged_t].sum()
    den = wt.sum() - wl
    return float((wt @ mar_t - wl) / den) if den > 0 else float("nan")


def _bias_from_sums(n, s, min_count: int = 20) -> dict:
    """metrics.bias_index(adjust=True) from weighted counts n and residual sums s per (group code, conf bin)."""
    nb, sb = n.sum(0), s.sum(0)
    ok = n >= min_count
    with np.errstate(invalid="ignore", divide="ignore"):
        num = np.where(ok, n * (s / n - sb / nb), 0).sum(1)
    den = np.where(ok, n, 0).sum(1)
    return {GROUPS[str(k)]: (float(num[k] / den[k]) if den[k] else float("nan"))
            for k in range(n.shape[0]) if n[k].sum() > 0}


def _wbias(cbins, g, w, p, y, n_bins: int = 10) -> dict:
    """metrics.bias_index on the resample, given conf bins from _wrank_bins and group codes g in {0, 1, 2}."""
    key = g * n_bins + cbins
    n = np.bincount(key, weights=w, minlength=3 * n_bins).reshape(3, n_bins)
    s = np.bincount(key, weights=w * (np.asarray(y, float) - p), minlength=3 * n_bins).reshape(3, n_bins)
    return _bias_from_sums(n, s)


def _push(store: dict, b: dict) -> None:
    for g in ("head", "mid", "tail"):
        store[g].append(b.get(g, np.nan))
    store["head_minus_tail"].append(b.get("head", np.nan) - b.get("tail", np.nan))

def _mean(x) -> float:
    x = np.asarray(x, float)
    return float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else float("nan")


def _ci(est, reps) -> dict:
    lo, hi = percentile_ci(reps)
    return {"est": est, "lo": lo, "hi": hi}


def uauc(users, score, label) -> dict:
    """Per-user AUC for users with both classes."""
    out = {}
    df = pd.DataFrame({"u": users, "s": score, "y": label})
    for u, g in df.groupby("u", sort=True):
        if 0 < g.y.sum() < len(g):
            out[u] = auroc(g.s.to_numpy(), g.y.to_numpy())
    return out


def decide(uauc_mar: float, share: float, gap_lo: float, gap_hi: float,
           lift_lo: float = float("nan")) -> tuple[str, str]:
    """P2 rule plus the 2026-10-02 pre-data addendum: the gap CI excluding 0 is near-mechanical under
    logged = O * MAR, so POSITIVE also needs the endpoint-1 lift CI above 1 (can only turn POSITIVE into AMBIGUOUS)."""
    if uauc_mar < UAUC_NULL:
        return "NULL", f"zero-shot UAUC_MAR {uauc_mar:.4f} < {UAUC_NULL}"
    excl = bool(gap_lo > 0 or gap_hi < 0)
    if share >= 0.20 and excl and lift_lo > 1:
        return "POSITIVE", "share >= 0.20, gap CI excludes 0 and lift CI lower bound > 1"
    if share < 0.10 and not excl:
        return "NEGATIVE", "share < 0.10 and gap CI includes 0"
    return "AMBIGUOUS", f"share={share:.4f}, gap CI excludes 0: {excl}, lift CI lower bound: {lift_lo:.4f}"


def analyze(pairs: pd.DataFrame, density: float, thrs=(1.0, 2.0, 3.0), alphas=(0.5, 1.0, 2.0), n_draws: int = 20,
            n_boot: int = 1000, seed: int = 0) -> dict:
    """pairs: every panel pair in panel order with user_id, watch_ratio, n_v, n_u, logit (NaN = not analysable)."""
    thrs = sorted({float(t) for t in thrs} | {PRIMARY[0]})
    alphas = sorted({float(a) for a in alphas} | {PRIMARY[1]})
    nv_all, nu_all = pairs.n_v.to_numpy(float), pairs.n_u.to_numpy(float)
    ok = np.isfinite(pairs.logit.to_numpy(float))
    logit = pairs.logit.to_numpy(float)[ok]
    p, wr = sigmoid(logit), pairs.watch_ratio.to_numpy(float)[ok]
    users = pairs.user_id.to_numpy().astype(str)[ok]
    grp = pop_groups(nv_all[ok])
    shares = {GROUPS[str(k)]: float((grp == k).mean()) for k in (0, 1, 2)}
    e3_issues = [f"{g} n_v group holds {shares[g]:.4f} of pairs, outside {GROUP_SHARE_OK} (ties at a quintile edge"
                 + ("; empty, so its Bias Index is null)" if not shares[g] else ")")
                 for g in ("head", "tail") if not GROUP_SHARE_OK[0] <= shares[g] <= GROUP_SHARE_OK[1]]
    if e3_issues:
        print("[warn] endpoint 3 (Bias Index head vs tail) not valid: " + "; ".join(e3_issues), flush=True)
    top = rank_bins(logit, 10) == 9
    uniq, inv = np.unique(users, return_inverse=True)
    inv = inv.reshape(-1)
    per_draw = max(50, n_boot // n_draws)
    picks = np.random.default_rng(np.random.SeedSequence([int(seed), 1])).integers(
        0, len(uniq), (per_draw * n_draws, len(uniq)))

    mar = {t: (wr >= t).astype(float) for t in thrs}
    e_all, c_of = {}, {}
    for a in alphas:
        w = nu_all * nv_all ** a
        c_of[a] = solve_c(w, density)
        e_all[a] = np.minimum(1.0, c_of[a] * w)
    cells, logged, ece_mar = {}, {}, {t: ece(p, mar[t]) for t in thrs}
    for t in thrs:
        for a in alphas:
            O = np.stack([draw_exposure(e_all[a], seed, d, a, t)[ok] for d in range(n_draws)])
            lg = O & (mar[t] > 0)                         # logged label = O * MAR, one row per draw
            cell = {"thr": t, "alpha": a, "c": c_of[a], "mean_exposure": float(e_all[a].mean()),
                    "observed_rate": float(O.mean()), "mar_pos_rate": float(mar[t].mean()),
                    "logged_pos_rate": float(lg.mean())}
            bad = [] if 0 < mar[t].mean() < 1 else ["MAR labels have a single class"]
            bad += [f"draw {d}: logged positive rate {x.mean():.4g} not in (0, MAR rate {mar[t].mean():.4g})"
                    for d, x in enumerate(lg) if not 0 < x.mean() < mar[t].mean()]
            if bad:
                if (t, a) == PRIMARY:
                    raise ValueError("degenerate primary cell: " + "; ".join(bad))
                cells[(t, a)] = {**cell, "degenerate": bad}
                continue
            pts = [endpoints(p, mar[t], x, top, ece_mar[t]) for x in lg]
            cell["_pts"] = {k: [q[k] for q in pts] for k in pts[0]}
            cell["_bias"] = [_bias(p, x, grp) for x in lg]
            cell["_auc"] = [auroc(logit, x) for x in lg]
            cell["_reps"] = {k: [] for k in ("share", "base", "lift", "gap", "gap_ceiling", "gap_normalized")}
            cell["_reps_bias"] = {g: [] for g in ("head", "mid", "tail", "head_minus_tail")}
            cells[(t, a)], logged[(t, a)] = cell, lg

    # Pooled user bootstrap: replicate r resamples users (picks[r]) and uses exposure draw r // per_draw; the top
    # decile and the Bias-Index confidence bins are recomputed on every resample, as the point statistic does.
    U, keys = len(uniq), ("head", "mid", "tail", "head_minus_tail")
    C = np.stack([np.bincount(k, minlength=U) for k in picks]).astype(float)     # user multiplicity per replicate
    eb = _ece_bins(p)
    ece_mar_reps = {t: _ece_reps(C, inv, eb, p, mar[t]) for t in thrs}
    for (t, a), lg in logged.items():                     # resample-invariant pair sets: all replicates at once
        reps = cells[(t, a)]["_reps"]
        for d in range(n_draws):
            rows = slice(d * per_draw, (d + 1) * per_draw)
            reps["base"] += _base_reps(C[rows], inv, mar[t], lg[d]).tolist()
            gap = _ece_reps(C[rows], inv, eb, p, lg[d]) - ece_mar_reps[t][rows]
            ceil = _ceiling_reps(C[rows], inv, mar[t], lg[d])
            reps["gap"] += gap.tolist()
            reps["gap_ceiling"] += ceil.tolist()
            with np.errstate(invalid="ignore", divide="ignore"):
                reps["gap_normalized"] += np.where(ceil > 0, gap / ceil, np.nan).tolist()
    lg_tie, n_lg = _tie_index(logit)
    p_tie, n_p = _tie_index(p)
    reps_mar = {t: {g: [] for g in keys} for t in thrs if 0 < mar[t].mean() < 1}
    pos_mar = {t: np.flatnonzero(mar[t]) for t in reps_mar}
    pos_lg = {k: [np.flatnonzero(x) for x in lg] for k, lg in logged.items()}
    for r in range(len(picks)):                           # rank-based sets: recomputed on every resample
        d = r // per_draw
        w = C[r][inv]
        ti = np.flatnonzero(_wrank_bins(lg_tie, n_lg, w, 10) == 9)
        wt = w[ti]
        key = grp * 10 + _wrank_bins(p_tie, n_p, w, 10)
        n = np.bincount(key, weights=w, minlength=30).reshape(3, 10)
        sp = np.bincount(key, weights=w * p, minlength=30).reshape(3, 10)
        for t in reps_mar:
            i = pos_mar[t]
            _push(reps_mar[t], _bias_from_sums(n, np.bincount(key[i], w[i], 30).reshape(3, 10) - sp))
        for (t, a), lg in logged.items():
            cell, i = cells[(t, a)], pos_lg[(t, a)][d]
            cell["_reps"]["share"].append(_wshare(wt, mar[t][ti], lg[d][ti]))
            _push(cell["_reps_bias"], _bias_from_sums(n, np.bincount(key[i], w[i], 30).reshape(3, 10) - sp))
    for cell in cells.values():
        if "_reps" in cell:
            with np.errstate(invalid="ignore", divide="ignore"):
                cell["_reps"]["lift"] = (np.array(cell["_reps"]["share"]) / np.array(cell["_reps"]["base"])).tolist()

    bias_mar = {}
    for t in reps_mar:
        est = _bias(p, mar[t], grp)
        est["head_minus_tail"] = est.get("head", np.nan) - est.get("tail", np.nan)
        bias_mar[t] = {g: _ci(est.get(g, np.nan), reps_mar[t][g]) for g in keys}
    out_cells, ua_of = [], {t: uauc(users, logit, mar[t]) for t in thrs}
    for (t, a), cell in cells.items():
        res = {k: v for k, v in cell.items() if not k.startswith("_")}
        ua = ua_of[t]
        res.update({"UAUC_MAR": float(np.mean(list(ua.values()))) if ua else float("nan"),
                    "n_users_both_classes": len(ua), "AUC_MAR": auroc(logit, mar[t]), "ECE_MAR": ece_mar[t]})
        if "_pts" in cell:
            pts = cell["_pts"]
            for k in ("share", "base", "lift", "gap", "gap_ceiling", "gap_normalized"):
                res[k] = _ci(_mean(pts[k]), cell["_reps"][k])
            hmt = [b.get("head", np.nan) - b.get("tail", np.nan) for b in cell["_bias"]]
            res.update({
                "ECE_logged": float(np.mean(pts["ece_logged"])), "AUC_logged": _mean(cell["_auc"]),
                "n_top_logged_neg": float(np.mean(pts["n_top_logged_neg"])),
                "bias_index_MAR": {g: bias_mar[t][g] for g in keys[:3]},
                "bias_index_logged": {
                    g: _ci(_mean([b.get(g, np.nan) for b in cell["_bias"]]), cell["_reps_bias"][g])
                    for g in keys[:3]},
                "bias_head_minus_tail_MAR": bias_mar[t]["head_minus_tail"],
                "bias_head_minus_tail_logged": _ci(_mean(hmt), cell["_reps_bias"]["head_minus_tail"]),
                "per_draw": {"share": pts["share"], "gap": pts["gap"], "gap_ceiling": pts["gap_ceiling"],
                             "ECE_logged": pts["ece_logged"]},
                "n_boot_pooled": len(cell["_reps"]["gap"])})
        out_cells.append(res)

    ua_vals = np.array(list(ua_of[PRIMARY[0]].values()), float)
    if not len(ua_vals):
        raise ValueError("no user has both MAR classes at thr=2.0: UAUC_MAR undefined")
    u_ci = cluster_bootstrap(lambda i: float(ua_vals[i].mean()), np.arange(len(ua_vals)), n_boot, seed)
    m2 = mar[PRIMARY[0]]
    a_ci = cluster_bootstrap(lambda i: auroc(logit[i], m2[i]), users, n_boot, seed)
    prim = next(c for c in out_cells if (c["thr"], c["alpha"]) == PRIMARY)
    decision, why = decide(u_ci["est"], prim["share"]["est"], prim["gap"]["lo"], prim["gap"]["hi"],
                           prim["lift"]["lo"])
    return {
        "n_pairs": int(ok.sum()), "n_panel_pairs": int(len(ok)), "n_users": int(len(uniq)),
        "target_density": density, "n_draws": n_draws, "n_boot": n_boot, "n_boot_per_draw": per_draw, "seed": seed,
        "thrs": thrs, "alphas": alphas,
        "group_shares": shares,
        "endpoint3": {"valid": not e3_issues, "issues": e3_issues, "share_range": list(GROUP_SHARE_OK)},
        "top_decile_share": float(top.mean()),
        "gate": {"UAUC_MAR": {**u_ci, "n_users": int(len(ua_vals))}, "AUC_MAR_pooled": a_ci,
                 "threshold": UAUC_NULL, "thr": PRIMARY[0]},
        "decision": decision, "decision_reason": why,
        "decision_rule": "thr=2.0, alpha=1: NULL iff zero-shot UAUC_MAR < 0.58; POSITIVE iff share >= 0.20, gap "
                         "CI excludes 0 and lift CI lower bound > 1; NEGATIVE iff share < 0.10 and gap CI includes 0; "
                         "else AMBIGUOUS (PREREG_AMENDMENT_1 P2 + 2026-10-02 addendum; LoRA retry moved to M3)",
        "caveats": CAVEATS,
        "primary": prim,
        "sensitivity": out_cells,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--question", default="like")
    ap.add_argument("--n_draws", type=int, default=20)
    ap.add_argument("--alphas", default="0.5,1,2")
    ap.add_argument("--thrs", default="1,2,3")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", type=float, default=None, help="override the panel meta target_density")
    a = ap.parse_args(argv)
    sc, rep = load_scores(a.scores, a.question)
    panel, meta = load_panel(a.panel)
    density = a.density if a.density is not None else meta.get("target_density")
    if density is None:
        raise ValueError("no target_density in the panel meta sidecar; pass --density")
    m = panel.merge(sc, on=["source_event_id", "cand_idx"], how="left", suffixes=("", "_s"), indicator=True,
                    validate="one_to_one")
    n_orphan = len(sc) - int((m["_merge"] == "both").sum())
    if n_orphan:
        raise ValueError(f"{n_orphan} score rows have no panel pair: wrong --panel?")
    both = m["_merge"] == "both"
    if (m.loc[both, "item_id"] != m.loc[both, "item_id_s"]).any():
        raise ValueError("score item_id differs from panel candidate_item_ids: wrong --panel?")
    if "label" in m and "thr" in meta:
        if (m.loc[both, "label"].astype(int) != (m.loc[both, "watch_ratio"] >= meta["thr"]).astype(int)).any():
            raise ValueError("score labels disagree with the panel watch ratios at the builder threshold")
    rep["n_panel_pairs_without_score"] = int((~both).sum())
    res = analyze(m, float(density), [float(x) for x in a.thrs.split(",")],
                  [float(x) for x in a.alphas.split(",")], a.n_draws, a.n_boot, a.seed)
    res = {"spec": "idea-stage/PREREG_AMENDMENT_1.md P2", "scores": rep, "panel_meta": meta, **res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(strict_json(res), f, indent=2, allow_nan=False)
    print(json.dumps({k: strict_json(res[k]) for k in ("decision", "decision_reason", "gate")}, indent=2))


if __name__ == "__main__":
    main()
