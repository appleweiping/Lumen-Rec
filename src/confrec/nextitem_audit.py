"""Full-scale next-item audit analysis (binding spec: docs/sigir/NEXTITEM_AUDIT_SPEC.md sections A-F; registered in
idea-stage/PREREG_AMENDMENT_2.md section N and the judge's action 7).

    python -m src.confrec.nextitem_audit run --domain sports --audit_dir <audit root> --panel_test P --panel_valid P \
        --ref_ranks docs/sigir/ref_ranks/sports --ref_exposure docs/sigir/ref_exposure/sports --out <dir>/sports.json \
        [--n_boot 2000] [--seed 0] [--questions next,like] [--quarantine_n N]
    python -m src.confrec.nextitem_audit summarize --inputs <dir>/sports.json,<dir>/toys.json,... --out <dir>/summary.json

Inputs per domain d: <audit_dir>/<d>_test/scores.csv.gz and <d>_valid2k/scores.csv.gz (pyes_scorer contract), the
test / valid panel JSONL (streamed; only source_event_id, user_id, history_item_ids, candidate_item_ids,
candidate_popularity_groups, positive_item_index are kept), docs/sigir/ref_ranks/<d>/<method>.csv.gz and
docs/sigir/ref_exposure/<d>/<method>.csv.gz (scripts/sigir/export_ref_exposure.py). Everything is vectorised over the
(events x candidates) arrays; the bootstrap shares one resample matrix per segment so every paired comparison resamples
the same events. The "definition" block of the output JSON repeats the choices below, quantity by quantity.

Quantities are exactly those of the spec (A ranking context, B static exposure / S3, C list-normalised calibration and
error anatomy / S1 S2 S4, D selective serving, E popularity-graded confidence / S5, F cross-domain summary). Where the
spec leaves a detail open the MOST CONSERVATIVE option was taken:

 1. Events and quarantine. Event = one panel row (= one user); every CI is a percentile bootstrap over events (n_boot
    resamples, seed 0 by default) of the segment analysed. Sports TEST events 1-1000 (the first 1000 rows of the test
    panel in FILE order; --quarantine_n overrides) are the segment `events_1_1000` (role "quarantine"), events
    1001-10000 the segment `events_1001_10000` (role "main"); there is NO pooled sports block, and the cross-domain
    summary uses the "main" segment as the sports entry of every 3-of-4 rule (the quarantine segment is reported next to
    it, never counted). Other domains form one segment `all`. The temperature is fitted once per (domain, question)
    on VALID2k and applied unchanged to every segment.
 2. Unscored candidates (censored 2/3, a non-finite logit, or no score row at all) carry no score: they are never
    ranked above a scored candidate; an unscored positive ranks last (rank = number of candidates) and unscored
    negatives rank below a scored positive (stats.tie_aware_rank / pilot_mirror._event_metrics conventions). In the
    softmax they get p = 0; the multiclass Brier sums over all candidates (an unscored positive adds 1); the NLL is
    undefined for an unscored positive and those events are excluded from NLL and from the temperature fit (counted).
    An event with NO scored candidate for a question stays in A (ranking context: a miss) and UAUC, and is excluded
    (counted) from B-E for that question. The pooled pointwise AUROC uses scored rows only. censored == 1 rows (one
    side imputed, the logit is a bound) are kept and counted.
 3. Ranks are the expected 1-based rank under uniformly random tie-breaking; NDCG@10 / HR@10 / MRR are reported at
    the expected rank and, as `*_tie_exact`, as the exact expectation over the positive's tie group (closed form,
    asserted equal to pilot_mirror._event_metrics in the tests). Top-10 / top-1 sets (exposure, correctness, p_max) use the
    deterministic order "highest score first, ties broken by LOWER cand_idx"; the number of events whose top-10
    cutoff (10th vs 11th) is tied and whose top-1 is tied are reported, together with tie-expected sensitivities
    (head share under random tie-breaking, top-1 accuracy under random tie-breaking). HR@1 (selective-serving utility
    and top-1 correctness) is the deterministic top-1 correctness; HR@10 hit (AUROC label) is expected rank <= 10.
 4. Temperature: beta = 1/T minimises the mean NLL of the positive over VALID2k events with a scored positive; the
    objective is convex in beta, so the exact minimiser on [1e-3, 10] is found by bisection on its monotone derivative
    (reported: T, beta, NLL at T=1, NLL at the fit, NLL of the uniform distribution = mean ln(#scored), whether the
    optimum sits on a bound). p = softmax(L / T) over the scored candidates of an event.
 5. Signals p_max, margin (= p_1 - p_2), neg_entropy (= sum p ln p), max_logit (uncalibrated, independent of T) and
    random (a fixed-seed uniform draw, the null). The random baseline of every AURC difference is its expectation,
    1 - mean utility (the random signal's own AURC is reported next to it).
 6. Tertiles (C.3) and quintiles of the niche profile (D) are computed ONCE on the analysed sample with
    stats.rank_bins (tied values share a bin) and held fixed inside the bootstrap (a conditional bootstrap; the
    edge uncertainty is second order at n ~ 10^4). The user profile averages the group score (head 2, mid 1, tail 0)
    over ALL history_item_ids of the panel row (not only the last 5 shown to the LLM); the item -> group map is built
    from every candidate list of the test panel (first-seen group, conflicts counted); history items missing from the
    map are skipped, users with no mapped item are excluded from the quintile statistics (counted) but still take
    part in the serving decision. "Served at 50% coverage" = the round(n/2) events with the highest signal; a tie block
    straddling the cut is served fractionally (random tie-breaking in expectation, as metrics.risk_coverage).
 7. Exposure counts (Gini, coverage) count an item once per event top-10 list; the pool is the domain's whole test pool
    (distinct candidate items of the test panel), also for the sports segments (so segment coverage / Gini are
    comparable across methods inside a segment, not across segments). Reference methods use their own exported
    top-10 lists and n_*_pool columns; an event whose candidate-pool counts, positive group, or top-10 groups disagree
    with the panel is excluded from the join (counted). Distinct-count statistics (coverage, Gini) are biased in
    a resample (about 63% of the events are distinct), so their interval is the percentile interval of the replicates
    RE-CENTRED at the point estimate (est + replicate - mean replicate); the raw percentile interval is reported next to it.
 8. Bootstrap p-values (two-sided, vs the stated null, add-one smoothing) feed the Holm adjustment of the summary
    (m = number of family domains, 4 expected). bias_index uses metrics.bias_index (10 equal-mass bins, adjust=True,
    min_count 20) in an event-cluster bootstrap of min(500, n_boot) resamples drawn from the same stream as
    metrics.bias_index_ci (identical estimates/intervals; the replicates are kept for the head-minus-tail difference).

Everything is numpy + stdlib. The weighted bootstraps (AUROC, AURC / risk-coverage curve / served share, pooled AUROC,
ECE, bias_index, Gini / coverage) are exact re-expressions of resampling events with replacement in terms of per-event
multiplicities (one resample matrix per segment, drawn from the stream of stats.cluster_bootstrap); at run time the
point estimates of the engines are asserted against stats / metrics (auroc, risk_coverage, ece, bias_index) to
SELF_CHECK_TOL, and the tests check every engine replicate-by-replicate against explicit resamples of the reference
implementations. pct_ci has the definition of stats.percentile_ci (asserted equal in the tests) without its Python loop.
Memory: the panel is streamed (peak RSS of a 10,000 x 101 domain with a 470 MB panel file was about 0.3 GB at n_boot 200);
resample matrices are int16 (n_boot x n_events per segment).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.confrec.metrics import auroc, bias_index, ece, ndcg_from_rank, reliability_bins, risk_coverage
from src.confrec.pilot_mirror import load_ref_ranks
from src.confrec.stats import rank_bins, strict_json

NAN = float("nan")
SCHEMA = "nextitem_audit_v1"
SUMMARY_SCHEMA = "nextitem_audit_summary_v1"
GROUPS = ("head", "mid", "tail")
GROUP_ID = {g: i for i, g in enumerate(GROUPS)}
GROUP_SCORE = np.array([2.0, 1.0, 0.0])       # niche profile: head 2, mid 1, tail 0
K = 10                                        # top-K list
COVERAGE = tuple(round(0.1 * i, 1) for i in range(1, 11))
HALF = COVERAGE.index(0.5)
SIGNALS = ("p_max", "margin", "neg_entropy", "max_logit", "random")
UTILS = ("ndcg10", "hr1")
QUESTIONS = ("next", "like")
N_BINS_ECE = 15
N_TERT = 3
N_QUINT = 5
BETA_BOUNDS = (1e-3, 10.0)
BIAS_BOOT = 500
SELF_CHECK_TOL = 1e-9
HOLM_ALPHA = 0.05
QUARANTINE = {"sports": 1000}
_EV_RE = re.compile(r'^\s*\{\s*"source_event_id"\s*:\s*"([^"\\]*)"')

DEFINITIONS = {
    "event": "one panel row = one user; all CIs are percentile bootstraps over the events of a segment (shared "
             "resample matrix per segment, so paired comparisons resample the same events)",
    "quarantine": "sports TEST events 1-1000 (first rows of the test panel in file order) are the segment "
                  "events_1_1000 (role quarantine); events 1001-10000 the segment events_1001_10000 (role main); no pooled "
                  "sports block; other domains are one segment 'all'",
    "unscored": "a candidate with censored 2/3, a non-finite logit or no score row is unscored: it ranks below every scored "
                "candidate, an unscored positive ranks last, p = 0 in the softmax; events without any scored candidate "
                "stay in A (a miss) and are excluded from B-E (counted)",
    "rank": "expected 1-based rank under uniformly random tie-breaking; NDCG@10 / HR@10 / MRR at the expected rank, "
            "*_tie_exact = exact expectation over the positive's tie group",
    "top_k": "top-K set = highest score first, ties broken by lower cand_idx (deterministic); cutoff / top-1 tie counts "
             "reported, tie-expected sensitivities reported",
    "temperature": "beta = 1/T minimises the mean NLL of the positive over VALID2k events with a scored positive; bounded "
                   "[1e-3, 10] exact search by bisection on the monotone derivative of the convex objective; p = softmax(L/T) "
                   "over scored candidates; T applied unchanged to TEST",
    "signals": "p_max, margin = p1 - p2, neg_entropy = sum p ln p, max_logit (uncalibrated), random = fixed-seed uniform null; "
               "higher = more confident; the AURC random baseline is its expectation 1 - mean utility",
    "S3": "head_share_top10 = mean over events of (#head in top-10)/10; pool_head_share = mean (#head candidates)/#candidates; "
          "delta_head = head_share_top10 - pool_head_share (the S3 endpoint); target_head_share = P(positive is head); "
          "coverage = distinct top-10 items / distinct pool items; gini = Gini of per-item top-10 event counts over the whole "
          "pool (zeros included)",
    "coverage_gini_ci": "recentred percentile interval of the event bootstrap (the raw replicates are biased low for "
                        "distinct counts); raw percentile bounds reported as lo_raw / hi_raw",
    "tertiles": "tertiles of p_max by stats.rank_bins on the analysed sample, held fixed in the bootstrap; "
                "confident_error_rate = P(top-1 wrong | top tertile), unsure_correct_rate = P(top-1 correct | bottom tertile)",
    "niche": "profile = mean of head 2 / mid 1 / tail 0 over all history_item_ids mapped via the candidate pool; quintiles by "
             "stats.rank_bins (bin 0 = niche); served = round(n/2) highest-signal events, tie block fractional",
    "S5": "top-10 candidates of each event, p list-normalised; mean p / mean event-centred logit / residual mean(y - p) by "
          "candidate group; bias_index = metrics.bias_index(conf=p, correct=y, group, 10 bins, adjust) with an event-cluster "
          "bootstrap (min(500, n_boot) resamples, same stream as metrics.bias_index_ci)",
    "pointwise": "pooled AUROC of the raw logit over scored candidate rows (event-cluster bootstrap); UAUC = mean per-event AUC "
                 "= 1 - (rank - 1)/(n_candidates - 1) at the expected rank",
    "p_boot": "two-sided percentile-bootstrap p-value vs the stated null, (1 + #{rep on one side}) / (B + 1) doubled",
}


# ----------------------------------------------------------------------------------------------------- helpers
def _log(msg: str) -> None:
    print(f"[nextitem_audit {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _open_text(path):
    p = str(path)
    return gzip.open(p, "rt", encoding="utf-8", newline="") if p.endswith(".gz") else \
        open(p, encoding="utf-8", newline="")


def boot_pvalue(reps, null: float) -> float:
    """Two-sided percentile-bootstrap p-value of H0: statistic == null (add-one smoothing)."""
    r = np.asarray(reps, float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return NAN
    le, ge = int((r <= null).sum()), int((r >= null).sum())
    return float(min(1.0, 2.0 * min((1 + le) / (len(r) + 1), (1 + ge) / (len(r) + 1))))


def pct_ci(reps, alpha: float = 0.05) -> tuple[float, float]:
    """stats.percentile_ci (same quantiles, non-finite replicates dropped) without its per-element Python loop."""
    r = np.asarray(reps, float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return NAN, NAN
    lo, hi = np.quantile(r, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def ci_dict(est, reps, n, null=None) -> dict:
    """{est, lo, hi, n, n_boot[, null, p_boot]}: percentile CI of the replicates (= stats.percentile_ci)."""
    reps = np.asarray(reps, float)
    lo, hi = pct_ci(reps)
    out = {"est": float(est), "lo": lo, "hi": hi, "n": int(n), "n_boot": int(len(reps))}
    if null is not None:
        out["null"] = float(null)
        out["p_boot"] = boot_pvalue(reps, null)
    return out


def ci_recentred(est, reps, n) -> dict:
    """Percentile interval of est + (rep - mean(rep)); the raw interval is kept as lo_raw / hi_raw."""
    reps = np.asarray(reps, float)
    fin = reps[np.isfinite(reps)]
    out = ci_dict(est, reps, n)
    out["lo_raw"], out["hi_raw"] = out["lo"], out["hi"]
    out["boot_mean"] = float(fin.mean()) if len(fin) else NAN
    if len(fin):
        out["lo"], out["hi"] = pct_ci(est + fin - fin.mean())
    return out


# ---------------------------------------------------------------------------------------------- bootstrap engine
class EventBoot:
    """One shared resample matrix per segment: W[b, e] = multiplicity of event e in resample b. Row b is
    bincount(rng.integers(0, n, n)) drawn from default_rng(seed) in sequence, i.e. exactly the resamples of
    stats.cluster_bootstrap / paired_bootstrap with one event per cluster and the same seed."""

    def __init__(self, n: int, n_boot: int, seed: int = 0):
        self.n, self.n_boot, self.seed = int(n), int(n_boot), int(seed)
        W = np.zeros((self.n_boot, self.n), np.int16)
        if self.n > 0:
            rng = np.random.default_rng(seed)
            for b in range(self.n_boot):
                W[b] = np.bincount(rng.integers(0, self.n, self.n), minlength=self.n)
        self.W = W

    def chunk_size(self, n_cols: int, budget: int = 2_000_000) -> int:
        return max(1, min(self.n_boot, budget // max(1, n_cols)))

    def resample_idx(self, b: int, cols=None) -> np.ndarray:
        """Row indices of resample b (sorted multiset), optionally restricted to the columns `cols`."""
        w = self.W[b] if cols is None else self.W[b][cols]
        return np.repeat(np.arange(len(w)), w)


class Stat:
    """A lazily evaluated bootstrap statistic (ratio of two weighted per-event sums, or a derived difference)."""
    __slots__ = ("kind", "a", "b", "const", "col", "null", "n", "est", "reps")

    def __init__(self, kind, null=None, n=0, col=-1, a=None, b=None, const=0.0):
        self.kind, self.null, self.n, self.col, self.a, self.b, self.const = kind, null, n, col, a, b, const
        self.est, self.reps = NAN, None

    def value(self) -> dict:
        return ci_dict(self.est, self.reps, self.n, self.null)


class LinearBatch:
    """Collects ratio statistics sum(w*num)/sum(w*den) over events and evaluates ALL of them in one pass over the
    resample chunks (two matrix products per chunk)."""

    def __init__(self, boot: EventBoot):
        self.boot, self.nums, self.stats, self.done = boot, [], [], False
        self.den_cols, self.den_of, self._den_col = [], [], {}

    def add(self, num, den, null=None, n=None) -> Stat:
        num, den = np.asarray(num, float), np.asarray(den, float)
        s = Stat("ratio", null=null, n=int(round(float(den.sum()))) if n is None else n, col=len(self.nums))
        key = hash(den.tobytes())                     # identical denominators (event masks) share one matrix column
        j = self._den_col.get(key)
        if j is None or not np.array_equal(self.den_cols[j], den):
            j = len(self.den_cols)
            self._den_col[key] = j
            self.den_cols.append(den)
        self.nums.append(num)
        self.den_of.append(j)
        self.stats.append(s)
        return s

    def derive(self, a: Stat, b: Stat, null=None, n=None) -> Stat:
        s = Stat("diff", null=null, n=min(a.n, b.n) if n is None else n, a=a, b=b)
        self.stats.append(s)
        return s

    def shift(self, a: Stat, const: float, null=None) -> Stat:
        s = Stat("shift", null=null, n=a.n, a=a, const=const)
        self.stats.append(s)
        return s

    def run(self) -> None:
        boot = self.boot
        k = len(self.nums)
        reps = np.full((boot.n_boot, k), np.nan)
        est = np.full(k, np.nan)
        if k:
            NUM, DEN = np.stack(self.nums, 1), np.stack(self.den_cols, 1)
            den_of = np.asarray(self.den_of)
            est_num, est_den = NUM.sum(0), DEN.sum(0)[den_of]
            step = boot.chunk_size(max(boot.n, 1))
            for c0 in range(0, boot.n_boot, step):
                Wc = boot.W[c0:c0 + step].astype(np.float64)
                rn, rd = Wc @ NUM, (Wc @ DEN)[:, den_of]
                with np.errstate(divide="ignore", invalid="ignore"):
                    reps[c0:c0 + step] = np.where(rd > 0, rn / rd, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                est = np.where(est_den > 0, est_num / est_den, np.nan)
        for s in self.stats:
            if s.kind == "ratio":
                s.est, s.reps = float(est[s.col]), reps[:, s.col]
            elif s.kind == "diff":
                s.est, s.reps = s.a.est - s.b.est, s.a.reps - s.b.reps
            elif s.kind == "shift":
                s.est, s.reps = s.a.est + s.const, s.a.reps + s.const
        self.done = True


def resolve(obj):
    """Replace every Stat inside a nested dict / list by its value dict (call after batch.run())."""
    if isinstance(obj, Stat):
        return obj.value()
    if isinstance(obj, dict):
        return {k: resolve(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [resolve(v) for v in obj]
    return obj


def weighted_auroc(Wv: np.ndarray, score, label) -> np.ndarray:
    """AUROC (ties averaged, as metrics.auroc) of every resample whose event multiplicities are the rows of Wv (B x n)."""
    score, label = np.asarray(score, float), np.asarray(label, float)
    B, n = Wv.shape
    out = np.full(B, np.nan)
    if n == 0:
        return out
    order = np.argsort(score, kind="stable")
    s, y = score[order], label[order]
    new = np.concatenate([[True], s[1:] != s[:-1]])
    gs = np.flatnonzero(new)
    gid = np.cumsum(new) - 1
    ge = np.concatenate([gs[1:], [n]])
    untied = len(gs) == n
    gs_of, ge_of = gs[gid], ge[gid]                     # tie block start / end (exclusive) of every sorted event
    step = max(1, 1_000_000 // n)
    for c0 in range(0, B, step):
        Wc = Wv[c0:c0 + step][:, order].astype(np.float64)
        wp, wn = Wc * y, Wc * (1.0 - y)
        ex = np.zeros((len(Wc), n + 1))
        np.cumsum(wn, axis=1, out=ex[:, 1:])
        if untied:
            num = (wp * ex[:, :n]).sum(1)
        else:
            below = ex[:, gs_of]                        # negative weight strictly below the block
            num = (wp * (below + 0.5 * (ex[:, ge_of] - below))).sum(1)
        den = wp.sum(1) * wn.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[c0:c0 + step] = np.where(den > 0, num / den, np.nan)
    return out


def ece_bins(conf: np.ndarray, n_bins: int = N_BINS_ECE) -> np.ndarray:
    """Equal-width bin index exactly as metrics.ece."""
    edges = np.linspace(0, 1, n_bins + 1)
    edges[-1] = 1 + 1e-9
    return np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)


def ece_reps(Wv: np.ndarray, conf, correct, adaptive: bool = False) -> np.ndarray:
    """ECE of every resample: equal-width via per-bin weighted sums (exact), adaptive via metrics.ece per resample."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    B, n = Wv.shape
    out = np.full(B, np.nan)
    if n == 0:
        return out
    if adaptive:
        o = np.argsort(conf, kind="stable")
        cs, ys = conf[o], correct[o]
        qs = np.linspace(0, 1, N_BINS_ECE + 1)
        for b in range(B):
            w = Wv[b][o]
            cr, yr = np.repeat(cs, w), np.repeat(ys, w)
            if len(cr) == 0:
                continue
            edges = np.quantile(cr, qs)
            edges[0], edges[-1] = -np.inf, np.inf
            idx = np.clip(np.searchsorted(edges, cr, side="right") - 1, 0, N_BINS_ECE - 1)
            sc = np.bincount(idx, weights=cr, minlength=N_BINS_ECE)
            sy = np.bincount(idx, weights=yr, minlength=N_BINS_ECE)
            out[b] = np.abs(sc - sy).sum() / len(cr)
        return out
    D = np.zeros((n, N_BINS_ECE))
    D[np.arange(n), ece_bins(conf)] = conf - correct
    step = max(1, 1_000_000 // n)
    for c0 in range(0, B, step):
        Wc = Wv[c0:c0 + step].astype(np.float64)
        neff = Wc.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[c0:c0 + step] = np.where(neff > 0, np.abs(Wc @ D).sum(1) / neff, np.nan)
    return out


class ServeEngine:
    """Selective-serving statistics of every bootstrap resample in closed form (no re-sorting).

    Events are sorted once by the signal (descending); a resample is the same order with multiplicities w. Equal-signal
    events form a tie block whose utilities are replaced by the block mean (metrics.risk_coverage), so with block weight
    W_g, utility sum T_g, items before the block s_g and utility sum before U_g the running mean at position s_g + m is
    (U_g + m T_g / W_g) / (s_g + m). Hence AURC = 1 - (1/n) sum_g [T_g + (U_g - s_g T_g / W_g) (H[s_g + W_g] - H[s_g])] with
    H the harmonic numbers. Coverage points use k = max(1, round(c n)); the served share of a quintile uses the expected
    served fraction clip((k - s_g) / W_g, 0, 1) of every block."""

    def __init__(self, signal, utils: dict, quint=None, cov=COVERAGE):
        signal = np.asarray(signal, float)
        n = len(signal)
        self.n, self.cov = n, np.asarray(cov, float)
        self.order = np.argsort(-signal, kind="stable")
        ss = signal[self.order]
        new = np.concatenate([[True], ss[1:] != ss[:-1]]) if n else np.zeros(0, bool)
        self.starts, self.gid = np.flatnonzero(new), np.cumsum(new) - 1
        self.u = {k: np.asarray(v, float)[self.order] for k, v in utils.items()}
        self.q1 = None
        if quint is not None:
            qs = np.asarray(quint)[self.order]
            self.q1 = np.zeros((n, N_QUINT))
            ok = qs >= 0
            self.q1[np.flatnonzero(ok), qs[ok]] = 1.0
        self.H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n + 1))])

    def reps(self, Wv: np.ndarray) -> dict:
        B, n = Wv.shape
        J = len(self.cov)
        res = {"aurc": {u: np.full(B, np.nan) for u in self.u}, "mean_u": {u: np.full(B, np.nan) for u in self.u},
               "curve": {u: np.full((B, J), np.nan) for u in self.u}}
        if self.q1 is not None:
            res["served"] = np.full((B, N_QUINT), np.nan)
            res["size"] = np.full((B, N_QUINT), np.nan)
            res["served_util"] = {u: np.full((B, N_QUINT), np.nan) for u in self.u}
        if n == 0:
            return res
        grouped = len(self.starts) < n
        step = max(1, 600_000 // n)
        for c0 in range(0, B, step):
            sl = slice(c0, min(B, c0 + step))
            Wi = Wv[sl][:, self.order].astype(np.int64)
            neff = Wi.sum(1)
            nsafe = np.where(neff > 0, neff, 1)
            Wg = np.add.reduceat(Wi, self.starts, axis=1) if grouped else Wi
            cs = np.cumsum(Wg, axis=1)
            sg = cs - Wg
            safe = np.where(Wg > 0, Wg, 1).astype(float)
            Hs, He = self.H[sg], self.H[cs]
            for u, us in self.u.items():
                T = np.add.reduceat(Wi * us, self.starts, axis=1) if grouped else Wi * us
                Ug = np.cumsum(T, axis=1) - T
                ubar = np.where(Wg > 0, T / safe, 0.0)
                tot = (T + (Ug - sg * ubar) * (He - Hs)).sum(1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    res["aurc"][u][sl] = np.where(neff > 0, 1.0 - tot / nsafe, np.nan)
                    res["mean_u"][u][sl] = np.where(neff > 0, T.sum(1) / nsafe, np.nan)
                for r in range(Wi.shape[0]):
                    if neff[r] == 0:
                        continue
                    ks = np.minimum(np.maximum(1, np.rint(self.cov * neff[r]).astype(np.int64)), neff[r])
                    gk = np.searchsorted(cs[r], ks, side="left")
                    res["curve"][u][c0 + r] = (Ug[r, gk] + (ks - sg[r, gk]) * ubar[r, gk]) / ks
            if self.q1 is not None:
                k50 = np.maximum(1, np.rint(0.5 * neff))
                frac = np.where(Wg > 0, np.clip((k50[:, None] - sg) / safe, 0.0, 1.0), 0.0)
                sv = Wi * (frac[:, self.gid] if grouped else frac)
                served = sv @ self.q1
                res["served"][sl] = served
                res["size"][sl] = Wi @ self.q1
                for u, us in self.u.items():
                    with np.errstate(divide="ignore", invalid="ignore"):
                        res["served_util"][u][sl] = np.where(served > 0, ((sv * us) @ self.q1) / served, np.nan)
        return res


class PooledAUC:
    """Pooled pointwise AUROC over candidate rows with an event-cluster bootstrap. Rows are bucketed by their exact
    value; a resample's positive / negative histograms are weighted sums of the per-event multiplicities, so no
    row-level sorting is needed per resample (ties are exact)."""

    def __init__(self, pos_val, pos_ev, neg_val, neg_ev):
        vals = np.concatenate([pos_val, neg_val])
        uniq, inv = np.unique(vals, return_inverse=True)
        inv = inv.reshape(-1)
        self.K = len(uniq)
        self.pos_bucket, neg_bucket = inv[:len(pos_val)], inv[len(pos_val):]
        self.pos_ev = np.asarray(pos_ev, np.int32)
        o = np.argsort(neg_bucket, kind="stable")
        self.neg_ev = np.asarray(neg_ev, np.int32)[o]
        nb = neg_bucket[o]
        self.neg_starts = np.flatnonzero(np.concatenate([[True], nb[1:] != nb[:-1]])) if len(nb) else np.zeros(0, int)
        self.neg_buckets = nb[self.neg_starts] if len(nb) else np.zeros(0, int)

    def stat(self, w) -> float:
        hp = np.bincount(self.pos_bucket, weights=w[self.pos_ev], minlength=self.K)
        hn = np.zeros(self.K)
        if len(self.neg_ev):
            hn[self.neg_buckets] = np.add.reduceat(w[self.neg_ev], self.neg_starts)
        den = hp.sum() * hn.sum()
        if den <= 0:
            return NAN
        return float((hp * (np.cumsum(hn) - hn + 0.5 * hn)).sum() / den)

    def reps(self, W: np.ndarray) -> np.ndarray:
        return np.array([self.stat(W[b].astype(np.int32)) for b in range(W.shape[0])])


class BiasEngine:
    """metrics.bias_index (adjust=True) of every event-cluster resample, without re-ranking the rows: rows are sorted
    by confidence once; a resample only changes the row weights, so the equal-mass bins of rank_bins are obtained from
    the weighted average ranks of the tie blocks (rank = rows before + (block weight + 1) / 2, bin = min(int(n_bins *
    (rank - 0.5) / n_rows), n_bins - 1)) and the per-(bin, group) weighted counts / residual sums by bincount."""

    def __init__(self, conf, correct, group_id, event, n_bins: int = 10, min_count: int = 20):
        conf = np.asarray(conf, float)
        o = np.argsort(conf, kind="stable")
        cs = conf[o]
        new = np.concatenate([[True], cs[1:] != cs[:-1]]) if len(cs) else np.zeros(0, bool)
        self.starts, self.gid = np.flatnonzero(new), np.cumsum(new) - 1
        self.grouped = len(self.starts) < len(cs)
        self.ev = np.asarray(event, np.int32)[o]
        self.grp = np.asarray(group_id, np.int64)[o]
        self.resid = (np.asarray(correct, float) - conf)[o]
        self.n_bins, self.min_count = n_bins, min_count

    def stat(self, w) -> np.ndarray:
        """(3,) bias index of head / mid / tail for event weights w (NaN where a group has no admissible bin)."""
        ws = w[self.ev]
        ntot = ws.sum()
        if ntot <= 0 or len(ws) == 0:
            return np.full(3, NAN)
        wg = np.add.reduceat(ws, self.starts) if self.grouped else ws
        rank = np.cumsum(wg) - wg + (wg + 1) / 2.0
        b = np.minimum((self.n_bins * (rank - 0.5) / ntot).astype(np.int64), self.n_bins - 1)
        key = (b[self.gid] if self.grouped else b) * 3 + self.grp
        nbg = np.bincount(key, weights=ws, minlength=3 * self.n_bins).reshape(self.n_bins, 3)
        sbg = np.bincount(key, weights=ws * self.resid, minlength=3 * self.n_bins).reshape(self.n_bins, 3)
        nb, sb = nbg.sum(1), sbg.sum(1)
        keep = nbg >= self.min_count
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = np.where(keep, sbg - nbg * (sb / np.where(nb > 0, nb, 1.0))[:, None], 0.0)
        num, den = dev.sum(0), np.where(keep, nbg, 0.0).sum(0)
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    def reps(self, W: np.ndarray) -> np.ndarray:
        return np.array([self.stat(W[b].astype(np.int32)) for b in range(W.shape[0])]).reshape(-1, 3)


class ExposureEngine:
    """Coverage and Gini of the per-item top-K exposure counts for any event weights (resample multiplicities).
    The Gini is computed from the histogram of counts: with h_v items of count v, G = S2 / (P sum c),
    S2 = sum_{a<b} h_a h_b (b - a), P = pool size (zero counts included)."""

    def __init__(self, top_items, member, pool_size: int):
        ti = np.asarray(top_items)
        n = len(ti)
        srt = np.sort(np.where(ti >= 0, ti, -1), axis=1)
        dup = np.zeros(srt.shape, bool)
        dup[:, 1:] = srt[:, 1:] == srt[:, :-1]
        srt = np.where(dup, -1, srt)
        mask = (srt >= 0) & np.asarray(member, bool)[:, None]
        self.ev_slot = np.broadcast_to(np.arange(n)[:, None], srt.shape)[mask]
        uniq, inv = np.unique(srt[mask], return_inverse=True)
        self.inv, self.K, self.P = inv.reshape(-1), len(uniq), int(pool_size)

    def stat(self, w) -> tuple[float, float]:
        if self.K == 0:
            return 0.0, NAN
        cnt = np.bincount(self.inv, weights=w[self.ev_slot], minlength=self.K).astype(np.int64)
        nz = int((cnt > 0).sum())
        h = np.bincount(cnt).astype(float)
        h[0] = self.P - nz
        a = np.arange(len(h), dtype=float)
        tot = float((a * h).sum())
        if tot <= 0:
            return nz / self.P, NAN
        hc = np.cumsum(h) - h
        m1 = np.cumsum(a * h) - a * h
        return nz / self.P, float((h * (a * hc - m1)).sum() / (self.P * tot))

    def reps(self, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = np.array([self.stat(W[b].astype(np.int32)) for b in range(W.shape[0])]).reshape(-1, 2)
        return out[:, 0], out[:, 1]


# --------------------------------------------------------------------------------------------------- data loading
@dataclass
class Panel:
    ev_ids: list
    users: list
    pos: np.ndarray            # (E,) positive_item_index
    n_cand: np.ndarray         # (E,)
    items: np.ndarray          # (E, N) int32 pool index of every candidate, -1 = padding
    grp: np.ndarray            # (E, N) int8 popularity group id (0 head, 1 mid, 2 tail), -1 = padding / unknown
    pool: dict                 # item id -> pool index
    pool_group: np.ndarray     # (P,) int8 first-seen group of every pool item
    pool_conflicts: int
    history: list              # per event: list of history item ids
    ev_index: dict
    n_users_multi_event: int

    @property
    def E(self) -> int:
        return len(self.ev_ids)

    @property
    def N(self) -> int:
        return self.items.shape[1]

    @property
    def cmask(self) -> np.ndarray:
        return np.arange(self.N)[None, :] < self.n_cand[:, None]


def load_panel(path, only_events=None, need_groups: bool = True) -> Panel:
    """Stream a ranking_{test,valid}.jsonl and keep only the audit fields. `only_events` skips (without parsing the
    JSON) every line whose source_event_id is not in the set."""
    ev_ids, users, hist, pos, ncs, item_rows, grp_rows = [], [], [], [], [], [], []
    pool: dict = {}
    pool_group: list = []
    conflicts = 0
    seen: set = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if only_events is not None:
                m = _EV_RE.match(line)
                if m is not None and m.group(1) not in only_events:
                    continue
            r = json.loads(line)
            ev = str(r["source_event_id"])
            if only_events is not None and ev not in only_events:
                continue
            if ev in seen:
                raise ValueError(f"{path}: duplicated source_event_id {ev}")
            seen.add(ev)
            cands = r["candidate_item_ids"]
            n = len(cands)
            gl = r.get("candidate_popularity_groups")
            if gl is None:
                if need_groups:
                    raise ValueError(f"{path}: event {ev} has no candidate_popularity_groups")
                g = [-1] * n
            else:
                if len(gl) != n:
                    raise ValueError(f"{path}: event {ev}: {len(gl)} groups for {n} candidates")
                try:
                    g = [GROUP_ID[str(x).strip().lower()] for x in gl]
                except KeyError as e:
                    raise ValueError(f"{path}: event {ev}: unknown popularity group {e}") from None
            p = int(r["positive_item_index"])
            if not 0 <= p < n:
                raise ValueError(f"{path}: event {ev}: positive_item_index {p} outside 0..{n - 1}")
            idx = []
            for it, gi in zip(cands, g):
                it = str(it)
                j = pool.get(it)
                if j is None:
                    j = len(pool)
                    pool[it] = j
                    pool_group.append(gi)
                elif pool_group[j] != gi:
                    if pool_group[j] < 0:
                        pool_group[j] = gi
                    elif gi >= 0:
                        conflicts += 1
                idx.append(j)
            ev_ids.append(ev)
            users.append(str(r.get("user_id", "")))
            hist.append([str(x) for x in (r.get("history_item_ids") or [])])
            pos.append(p)
            ncs.append(n)
            item_rows.append(idx)
            grp_rows.append(g)
    E = len(ev_ids)
    N = max(ncs) if E else 0
    items = np.full((E, N), -1, np.int32)
    grp = np.full((E, N), -1, np.int8)
    for e in range(E):
        items[e, :ncs[e]] = item_rows[e]
        grp[e, :ncs[e]] = grp_rows[e]
    uc = Counter(users)
    return Panel(ev_ids=ev_ids, users=users, pos=np.asarray(pos, np.int64), n_cand=np.asarray(ncs, np.int64),
                 items=items, grp=grp, pool=pool, pool_group=np.asarray(pool_group, np.int8),
                 pool_conflicts=conflicts, history=hist, ev_index={e: i for i, e in enumerate(ev_ids)},
                 n_users_multi_event=int(sum(1 for c in uc.values() if c > 1)))


def scan_event_ids(path) -> set:
    """source_event_id column of a scores file (to subset a large valid panel without parsing it)."""
    with _open_text(path) as f:
        rd = csv.reader(f)
        header = next(rd)
        i = header.index("source_event_id")
        return {row[i] for row in rd if row}


def load_scores(path, panel: Panel, questions) -> tuple[dict, dict]:
    """scores.csv.gz -> ({question: (E, N) float64 logit array, NaN = unscored}, diagnostics). A duplicated
    (event, cand, question) keeps the last row (counted); rows whose item id differs from the panel's candidate are
    ignored (counted); rows with censored 2/3 or a non-finite logit leave the candidate unscored."""
    E, N = panel.E, panel.N
    qset = set(questions)
    L = {q: np.full(E * N, np.nan) for q in questions}
    seen = {q: np.zeros(E * N, bool) for q in questions}
    items_flat = panel.items.reshape(-1)
    label_ok = np.zeros(E * N, np.int8)
    label_ok[np.arange(E) * N + panel.pos] = 1
    cens = {q: Counter() for q in questions}
    diag = Counter()
    pool, nc, ev_index = panel.pool, panel.n_cand, panel.ev_index
    with _open_text(path) as f:
        rd = csv.reader(f)
        header = next(rd)
        col = {h: i for i, h in enumerate(header)}
        for need in ("source_event_id", "item_id", "cand_idx", "question", "logit"):
            if need not in col:
                raise ValueError(f"{path}: missing column {need}")
        i_ev, i_it, i_c, i_q, i_lg = (col[k] for k in ("source_event_id", "item_id", "cand_idx", "question", "logit"))
        i_cs, i_lb = col.get("censored", -1), col.get("label", -1)
        for row in rd:
            diag["score_rows"] += 1
            e = ev_index.get(row[i_ev])
            if e is None:
                diag["score_rows_without_panel_event"] += 1
                continue
            q = row[i_q]
            if q not in qset:
                diag["score_rows_other_question"] += 1
                continue
            try:
                c = int(row[i_c])
            except ValueError:
                c = int(float(row[i_c]))
            if c < 0 or c >= nc[e]:
                diag["score_rows_cand_idx_out_of_range"] += 1
                continue
            key = e * N + c
            if pool.get(row[i_it], -2) != items_flat[key]:
                diag["score_rows_item_mismatch"] += 1
                continue
            if i_lb >= 0:
                try:
                    if int(float(row[i_lb])) != label_ok[key]:
                        diag["score_rows_label_mismatch"] += 1
                except ValueError:
                    pass
            code = row[i_cs].strip() if i_cs >= 0 else "na"
            cs = cens[q]
            cs["n"] += 1
            cs[code] += 1
            if seen[q][key]:
                cs["duplicate"] += 1
            seen[q][key] = True
            try:
                x = float(row[i_lg])
            except ValueError:
                x = NAN
            if math.isfinite(x) and code not in ("2", "3"):
                L[q][key] = x
            else:
                L[q][key] = NAN
                cs["dropped_nonfinite_or_censored"] += 1
    cm = panel.cmask.reshape(-1)
    out = {}
    for q in questions:
        out[q] = L[q].reshape(E, N)
        cens[q]["missing_rows"] = int((~seen[q] & cm).sum())
    diag["by_question"] = {q: dict(c) for q, c in cens.items()}
    return out, dict(diag)


def load_ref_exposure(path, panel: Panel) -> dict:
    """ref_exposure/<method>.csv.gz -> arrays aligned to the panel events (see scripts/sigir/export_ref_exposure.py).
    An event is a member of the join only if its user, candidate-pool group counts, positive group and top-10
    groups agree with the panel; every disagreement is counted and excludes the event. (The positive_rank column
    is not used here: positive ranks come from docs/sigir/ref_ranks.)"""
    E = panel.E
    top_grp = np.full((E, K), -1, np.int8)
    top_item = np.full((E, K), -1, np.int32)
    pool_cnt = np.zeros((E, 3), np.int64)
    pos_grp = np.full(E, -1, np.int8)
    seen = np.zeros(E, bool)
    member = np.zeros(E, bool)
    diag = Counter()
    panel_cnt = np.stack([((panel.grp == g) & panel.cmask).sum(1) for g in range(3)], 1)
    panel_pos_grp = panel.grp[np.arange(E), panel.pos]
    with _open_text(path) as f:
        for r in csv.DictReader(f):
            diag["rows"] += 1
            e = panel.ev_index.get(r["source_event_id"])
            if e is None:
                diag["rows_not_in_panel"] += 1
                continue
            if seen[e]:
                diag["duplicate_rows"] += 1
                continue
            seen[e] = True
            if r.get("user_id", "") != "" and str(r["user_id"]) != panel.users[e]:
                diag["user_id_mismatch"] += 1
                continue
            ids, gs = r["top10_item_ids"].split(" "), r["top10_groups"].split(" ")
            try:
                tg = [GROUP_ID[x] for x in gs]
                cnt = (int(r["n_head_pool"]), int(r["n_mid_pool"]), int(r["n_tail_pool"]))
                pg = GROUP_ID[r["positive_group"]]
            except (KeyError, ValueError):
                diag["unparsable_rows"] += 1
                continue
            if len(ids) != K or len(tg) != K:
                diag["bad_top10_length"] += 1
                continue
            ti = [panel.pool.get(i, -1) for i in ids]
            if min(ti) < 0:
                diag["top10_item_not_in_pool"] += 1
                continue
            if list(cnt) != panel_cnt[e].tolist():
                diag["pool_count_mismatch"] += 1
                continue
            if pg != panel_pos_grp[e]:
                diag["positive_group_mismatch"] += 1
                continue
            if any(int(panel.pool_group[j]) != g for j, g in zip(ti, tg)):
                diag["top10_group_vs_panel_mismatch"] += 1
                continue
            top_grp[e], top_item[e], pool_cnt[e], pos_grp[e] = tg, ti, cnt, pg
            member[e] = True
    diag["n_member"] = int(member.sum())
    diag["panel_events_not_joined"] = int(E - member.sum())
    return {"top_grp": top_grp, "top_item": top_item, "pool_cnt": pool_cnt, "pos_grp": pos_grp, "member": member,
            "diag": dict(diag)}


def align_ref_ranks(path, panel: Panel) -> tuple[np.ndarray, dict]:
    """ref_ranks/<method>.csv.gz -> positive rank per panel event (NaN = not joined) with the pilot_mirror join rules."""
    rr = load_ref_ranks(path)
    rank = np.full(panel.E, np.nan)
    diag = Counter()
    for ev, (u, rk, nc) in rr.items():
        e = panel.ev_index.get(ev)
        if e is None:
            diag["ref_events_not_in_panel"] += 1
        elif nc and nc != panel.n_cand[e]:
            diag["num_candidates_mismatch"] += 1
        elif u != "" and str(u) != panel.users[e]:
            diag["user_id_mismatch"] += 1
        else:
            rank[e] = rk
    diag["n_ref_rows"] = len(rr)
    diag["n_joined"] = int(np.isfinite(rank).sum())
    return rank, dict(diag)


def user_profile(panel: Panel) -> tuple[np.ndarray, dict]:
    """Mean group score (head 2, mid 1, tail 0) of the history items found in the candidate pool; NaN if none."""
    pg = panel.pool_group
    prof = np.full(panel.E, np.nan)
    n_hist = n_mapped = 0
    for e, h in enumerate(panel.history):
        n_hist += len(h)
        vals = []
        for it in h:
            j = panel.pool.get(it)
            if j is not None and pg[j] >= 0:
                vals.append(GROUP_SCORE[pg[j]])
        n_mapped += len(vals)
        if vals:
            prof[e] = float(np.mean(vals))
    return prof, {"n_history_items": n_hist, "n_history_items_mapped": n_mapped,
                  "n_events_without_mapped_history": int(np.isnan(prof).sum())}


# ------------------------------------------------------------------------------------- per-question computations
def _harmonic(N: int) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, N + 1))])


_DCG10 = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(1, K + 1) + 1.0))])


def tie_exact_metrics(g: np.ndarray, t: np.ndarray, N: int) -> dict:
    """Exact expectation of NDCG@10 / HR@10 / MRR over the positive's tie group: ranks g+1 .. g+t equally likely
    (g = #candidates strictly above, t = tie group size incl. the positive)."""
    g, t = np.asarray(g, np.int64), np.asarray(t, np.int64)
    gt = g + t
    H = _harmonic(max(N, int(gt.max()) if len(gt) else 0))
    return {"ndcg10_x": (_DCG10[np.minimum(gt, K)] - _DCG10[np.minimum(g, K)]) / t,
            "hr10_x": np.clip(K - g, 0, t) / t,
            "mrr_x": (H[gt] - H[g]) / t}


def rank_core(L: np.ndarray, cmask: np.ndarray, pos: np.ndarray) -> dict:
    """Vectorised ranking quantities of one question over (events x candidates)."""
    E, N = L.shape
    ar = np.arange(E)
    S = np.where(np.isfinite(L) & cmask, L, -np.inf)
    scored = np.isfinite(S)
    n_scored = scored.sum(1)
    n_cand = cmask.sum(1)
    Lpos = S[ar, pos]
    pos_scored = np.isfinite(Lpos)
    greater = (S > Lpos[:, None]).sum(1)
    ties = (S == Lpos[:, None]).sum(1) - 1
    g = np.where(pos_scored, greater, n_cand - 1)
    t = np.where(pos_scored, ties + 1, 1)
    rank = np.where(pos_scored, 1.0 + greater + ties / 2.0, n_cand.astype(float))
    order = np.argsort(-S, axis=1, kind="stable")
    Ssort = np.take_along_axis(S, order, 1)
    s1 = Ssort[:, 0]
    top1 = order[:, 0]
    valid = n_scored >= 1
    top1_correct = ((top1 == pos) & valid).astype(float)
    n_top = (S == s1[:, None]).sum(1)
    out = {"S": S, "scored": scored, "n_scored": n_scored, "n_cand": n_cand, "pos_scored": pos_scored, "valid": valid,
           "rank": rank, "g": g, "t": t, "order": order, "Ssort": Ssort,
           "ndcg10": ndcg_from_rank(rank, K), "hr10": (rank <= K).astype(float), "mrr": 1.0 / rank,
           "uauc": np.where(n_cand > 1, 1.0 - (rank - 1.0) / np.maximum(n_cand - 1, 1), np.nan),
           "top1": top1, "top1_correct": top1_correct, "hit10": (rank <= K).astype(float),
           "tied_top1": np.isfinite(s1) & (Ssort[:, 1] == s1) if N > 1 else np.zeros(E, bool),
           "cutoff_tied": (np.isfinite(Ssort[:, K - 1]) & (Ssort[:, K - 1] == Ssort[:, K])) if N > K else np.zeros(E, bool),
           "top1_acc_tie_exp": np.where(pos_scored & (Lpos == s1) & valid, 1.0 / np.maximum(n_top, 1), 0.0)}
    out.update(tie_exact_metrics(g, t, N))
    return out


def softmax_probs(S: np.ndarray, beta: float) -> np.ndarray:
    """softmax(beta * S) over the finite entries of each row (zero for -inf entries; all zeros if none is finite)."""
    smax = S.max(axis=1)
    ok = np.isfinite(smax)
    D = np.where(ok[:, None], S - np.where(ok, smax, 0.0)[:, None], -np.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        Wt = np.exp(beta * D)
    Z = Wt.sum(1)
    return np.where(Z[:, None] > 0, Wt / np.where(Z > 0, Z, 1.0)[:, None], 0.0)


def _centred(S: np.ndarray, pos: np.ndarray):
    """(D, Dpos, use): scores shifted by the row max (<= 0, -inf unscored), the positive's shifted score, and the events
    that enter the NLL (a scored positive)."""
    ar = np.arange(len(S))
    smax = S.max(axis=1)
    Lpos = S[ar, pos]
    use = np.isfinite(smax) & np.isfinite(Lpos)
    shift = np.where(np.isfinite(smax), smax, 0.0)
    D = (S - shift[:, None])[use]
    return D, (Lpos - shift)[use], use


def mean_nll(S: np.ndarray, pos: np.ndarray, beta: float) -> tuple[float, int]:
    D, Dp, use = _centred(S, pos)
    if not use.any():
        return NAN, 0
    with np.errstate(over="ignore"):
        return float(np.mean(np.log(np.exp(beta * D).sum(1)) - beta * Dp)), int(use.sum())


def fit_temperature(S: np.ndarray, pos: np.ndarray, bounds=BETA_BOUNDS) -> dict:
    """Minimise the mean NLL of the positive over beta = 1/T in `bounds`. The objective is convex in beta and its
    derivative mean(E_p[D] - D_pos) is increasing, so bisection on the derivative sign is the exact bounded search."""
    D, Dp, use = _centred(S, pos)
    n_scored = np.isfinite(S).sum(1)
    out = {"n_events_used": int(use.sum()), "n_events_positive_unscored":
           int((np.isfinite(S.max(axis=1)) & ~np.isfinite(S[np.arange(len(S)), pos])).sum()),
           "n_events_no_scored_candidate": int((~np.isfinite(S.max(axis=1))).sum())}
    if not use.any():
        out.update(T=NAN, beta=NAN, nll_T1=NAN, nll_fit=NAN, nll_uniform=NAN, bound_hit=None)
        return out
    Dz = np.where(np.isfinite(D), D, 0.0)

    def grad(beta):
        with np.errstate(over="ignore"):
            Wt = np.exp(beta * D)
        return float(np.mean((Wt * Dz).sum(1) / Wt.sum(1) - Dp))

    lo, hi = bounds
    hit = None
    if grad(lo) >= 0:
        beta, hit = lo, "lower"
    elif grad(hi) <= 0:
        beta, hit = hi, "upper"
    else:
        a, b = lo, hi
        for _ in range(200):
            m = 0.5 * (a + b)
            if grad(m) < 0:
                a = m
            else:
                b = m
            if b - a < 1e-13 * max(1.0, b):
                break
        beta = 0.5 * (a + b)
    out.update(T=1.0 / beta, beta=float(beta), nll_T1=mean_nll(S, pos, 1.0)[0], nll_fit=mean_nll(S, pos, beta)[0],
               nll_uniform=float(np.mean(np.log(n_scored[use]))), ln_101=math.log(101.0), bound_hit=hit,
               beta_bounds=list(bounds), grad_at_fit=grad(beta))
    return out


def question_arrays(L: np.ndarray, panel: Panel, beta: float, rand: np.ndarray) -> dict:
    """Everything the sections need for one question, computed once over ALL test events (sliced per segment)."""
    rc = rank_core(L, panel.cmask, panel.pos)
    E, N = L.shape
    ar = np.arange(E)
    S, order = rc["S"], rc["order"]
    P = softmax_probs(S, beta)
    p1, p2 = P[ar, order[:, 0]], P[ar, order[:, 1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(P > 0, P * np.log(np.where(P > 0, P, 1.0)), 0.0)
    smax = rc["Ssort"][:, 0]
    p_pos = P[ar, panel.pos]
    valid = rc["valid"]
    top = order[:, :K]
    top_scored = np.take_along_axis(rc["scored"], top, 1)
    with np.errstate(divide="ignore"):
        nll = np.where(rc["pos_scored"] & valid, -np.log(np.maximum(p_pos, 1e-300)), np.nan)
    mu = np.where(valid, np.where(rc["scored"], S, 0.0).sum(1) / np.maximum(rc["n_scored"], 1), np.nan)
    # tie-expected head / mid / tail share of the top-K (random tie-breaking at the cutoff)
    sk = rc["Ssort"][:, K - 1]
    above = S > sk[:, None]
    tie = (S == sk[:, None]) & panel.cmask
    n_above, n_tie = above.sum(1), np.maximum(tie.sum(1), 1)
    tie_exp = np.stack([(((panel.grp == g) & above).sum(1) + (K - n_above) * ((panel.grp == g) & tie).sum(1) / n_tie) / K
                        for g in range(3)], 1)
    qa = dict(rc)
    qa.update(
        P=P, p_max=P.max(axis=1), margin=p1 - p2, neg_entropy=plogp.sum(1), max_logit=np.where(valid, smax, np.nan),
        random=rand, p_pos=p_pos, nll=nll, brier=(P ** 2).sum(1) - 2.0 * p_pos + 1.0,
        top_cand=top, top_scored=top_scored, top_grp=np.take_along_axis(panel.grp, top, 1),
        top_item=np.take_along_axis(panel.items, top, 1), top_p=np.take_along_axis(P, top, 1),
        top_y=(top == panel.pos[:, None]).astype(float),
        top_lc=np.where(top_scored, np.take_along_axis(S, top, 1) - mu[:, None], np.nan),
        tie_exp_share=tie_exp,
        pool_cnt=np.stack([((panel.grp == g) & panel.cmask).sum(1) for g in range(3)], 1),
        pos_grp=panel.grp[ar, panel.pos])
    qa["hr1"] = qa["top1_correct"]
    qa["n_any_unscored"] = (rc["n_scored"] < rc["n_cand"])
    qa["pos"] = panel.pos
    for k in ("Ssort", "order", "scored", "g", "t"):
        qa.pop(k, None)
    return qa            # S (masked scores) and pos stay: the pooled AUROC needs them


# ------------------------------------------------------------------------------------------------------ sections
def _seg(arr: dict, sl: slice) -> dict:
    return {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) else v) for k, v in arr.items()}


def block_ranking(batch: LinearBatch, m: np.ndarray, vals: dict) -> dict:
    mf = m.astype(float)
    return {k: batch.add(np.where(m, v, 0.0), mf) for k, v in vals.items()}


def block_exposure(batch: LinearBatch, boot: EventBoot, top_grp, pool_cnt, pos_grp, top_item, m, pool_size,
                   n_boot: int, tie_exp=None) -> dict:
    """Static exposure block (B.1-B.4) for any method given its per-event top-K groups / items."""
    mf = m.astype(float)
    sh = [(top_grp == g).sum(1) / float(K) for g in range(3)]
    tot = np.maximum(pool_cnt.sum(1), 1)
    pl = [pool_cnt[:, g] / tot for g in range(3)]
    tgt = (pos_grp == 0).astype(float)
    out = {"n_events": int(m.sum())}
    for g, name in enumerate(GROUPS):
        out[f"{name}_share_top10"] = batch.add(sh[g] * mf, mf)
        out[f"pool_{name}_share"] = batch.add(pl[g] * mf, mf)
        out[f"delta_{name}"] = batch.add((sh[g] - pl[g]) * mf, mf, null=0.0)
    out["target_head_share"] = batch.add(tgt * mf, mf)
    out["delta_head_vs_target"] = batch.add((sh[0] - tgt) * mf, mf, null=0.0)
    if tie_exp is not None:
        out["tie_expected"] = {f"{name}_share_top10": batch.add(tie_exp[:, g] * mf, mf) for g, name in enumerate(GROUPS)}
        out["tie_expected"]["delta_head"] = batch.add((tie_exp[:, 0] - pl[0]) * mf, mf, null=0.0)
    eng = ExposureEngine(top_item, m, pool_size)
    cov_est, gini_est = eng.stat(np.ones(boot.n, np.int32))
    cov_r, gini_r = eng.reps(boot.W)
    out["coverage_top10"] = ci_recentred(cov_est, cov_r, int(m.sum()))
    out["gini_exposure"] = ci_recentred(gini_est, gini_r, int(m.sum()))
    out["n_pool_items"] = int(pool_size)
    out["n_distinct_top10_items"] = int(eng.K)
    return out


def exposure_vs(batch: LinearBatch, top_grp_a, top_grp_b, m) -> dict:
    """Paired head-share difference A - B on the events of `m` (same resamples)."""
    mf = m.astype(float)
    return {f"{name}_share_top10_llm_minus_ref": batch.add(((top_grp_a == g).sum(1) - (top_grp_b == g).sum(1)) / float(K)
                                                          * mf, mf, null=0.0) for g, name in enumerate(GROUPS)}


def sec_C(batch: LinearBatch, boot: EventBoot, qv: dict, m: np.ndarray, Tinfo: dict) -> dict:
    """List-normalised calibration, discrimination, error anatomy, pointwise (spec C.1-C.4)."""
    out: dict = {"temperature": Tinfo}
    n = len(m)
    mf = m.astype(float)
    cols = np.flatnonzero(m)
    Wv = boot.W if len(cols) == n else boot.W[:, cols]
    pmax, top1c, hit10 = qv["p_max"], qv["top1_correct"], qv["hit10"]
    pm, c1, h10 = pmax[cols], top1c[cols], hit10[cols]
    # C.1 --------------------------------------------------------------------------------------
    c1blk = {"top1_accuracy": batch.add(top1c * mf, mf), "p_max_mean": batch.add(pmax * mf, mf),
             "brier": batch.add(qv["brier"] * mf, mf)}
    nll_ok = m & np.isfinite(qv["nll"])
    c1blk["nll"] = batch.add(np.where(nll_ok, qv["nll"], 0.0), nll_ok.astype(float))
    c1blk["n_events_nll_excluded_positive_unscored"] = int((m & ~np.isfinite(qv["nll"])).sum())
    if len(cols):
        est_e = ece(pm, c1, N_BINS_ECE)
        est_a = ece(pm, c1, N_BINS_ECE, adaptive=True)
        r_e = ece_reps(Wv, pm, c1)
        _self_check("ECE", weighted_ece_est(pm, c1), est_e)
        c1blk["ece"] = ci_dict(est_e, r_e, len(cols))
        _self_check("adaptive ECE", ece_reps(np.ones((1, len(cols)), np.int16), pm, c1, adaptive=True)[0], est_a)
        c1blk["ece_adaptive"] = ci_dict(est_a, ece_reps(Wv, pm, c1, adaptive=True), len(cols))
        c1blk["reliability_bins"] = [{"lo": lo, "hi": hi, "n": nb, "mean_conf": mc, "acc": ac}
                                     for lo, hi, nb, mc, ac in reliability_bins(pm, c1, N_BINS_ECE)]
    out["list_normalised"] = c1blk
    # C.2 --------------------------------------------------------------------------------------
    disc: dict = {}
    for lname, lab in (("top1", c1), ("hr10", h10)):
        est, reps = {}, {}
        for sg in SIGNALS:
            sc = qv[sg][cols]
            est[sg] = auroc(sc, lab) if len(cols) else NAN
            reps[sg] = weighted_auroc(Wv, sc, lab)
            if len(cols):
                _self_check(f"AUROC {sg}/{lname}", weighted_auroc(np.ones((1, len(cols)), np.int16), sc, lab)[0], est[sg])
        blk = {"auroc": {sg: ci_dict(est[sg], reps[sg], len(cols)) for sg in SIGNALS},
               "paired_vs_pmax": {sg: ci_dict(est[sg] - est["p_max"], reps[sg] - reps["p_max"], len(cols), null=0.0)
                                  for sg in SIGNALS if sg != "p_max"},
               "pmax_minus_random": ci_dict(est["p_max"] - est["random"], reps["p_max"] - reps["random"], len(cols),
                                            null=0.0),
               "n_positive": int(lab.sum()), "n_negative": int(len(lab) - lab.sum())}
        disc[lname] = blk
    out["auroc_discrimination"] = disc
    # C.3 --------------------------------------------------------------------------------------
    tert = np.full(n, -1)
    if len(cols):
        tert[cols] = rank_bins(pm, N_TERT)
    wrong = (1.0 - top1c) * mf
    right = top1c * mf
    anat: dict = {"tertile_p_max_range": [[float(pm[tert[cols] == j].min()), float(pm[tert[cols] == j].max())]
                                          if (tert[cols] == j).any() else None for j in range(N_TERT)],
                  "tertiles": []}
    tb = {}
    for j in range(N_TERT):
        inj = (tert == j).astype(float)
        tb[j] = {"n": int(inj.sum()), "accuracy": batch.add(top1c * inj, inj), "hr10": batch.add(hit10 * inj, inj),
                 "mean_p_max": batch.add(pmax * inj, inj)}
        anat["tertiles"].append(tb[j])
    top, bot = (tert == N_TERT - 1).astype(float), (tert == 0).astype(float)
    anat["confident_error_rate"] = batch.add((1.0 - top1c) * top, top)
    anat["unsure_correct_rate"] = tb[0]["accuracy"]
    anat["share_errors_in_top_tertile"] = batch.add(wrong * top, wrong)
    anat["share_errors_in_top_minus_third"] = batch.shift(anat["share_errors_in_top_tertile"], -1.0 / N_TERT, null=0.0)
    anat["share_correct_in_bottom_tertile"] = batch.add(right * bot, right)
    anat["acc_top_minus_bottom_tertile"] = batch.derive(tb[N_TERT - 1]["accuracy"], tb[0]["accuracy"], null=0.0)
    anat["n_top1_errors"] = int(wrong.sum())
    out["error_anatomy"] = anat
    # C.4 --------------------------------------------------------------------------------------
    uauc = np.where(np.isfinite(qv["uauc"]), qv["uauc"], 0.0)
    uok = np.isfinite(qv["uauc"]).astype(float)
    pw = {"uauc": batch.add(uauc, uok)}
    pw["uauc_minus_half"] = batch.shift(pw["uauc"], -0.5, null=0.0)
    pw.update(pooled_block(boot, qv))
    out["pointwise"] = pw
    return out


def weighted_ece_est(conf, correct) -> float:
    """ECE through the per-bin weighted-sum form used by the bootstrap (unit weights): self-check against metrics.ece."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    D = np.zeros((len(conf), N_BINS_ECE))
    D[np.arange(len(conf)), ece_bins(conf)] = conf - correct
    return float(np.abs(D.sum(0)).sum() / len(conf)) if len(conf) else NAN


def _self_check(name: str, got: float, want: float) -> None:
    if not (math.isnan(got) and math.isnan(want)) and abs(got - want) > SELF_CHECK_TOL:
        raise AssertionError(f"self-check failed for {name}: engine {got!r} != reference {want!r}")


def pooled_block(boot: EventBoot, qv: dict) -> dict:
    """C.4 pooled pointwise AUROC of the raw logit over scored candidate rows (event-cluster bootstrap)."""
    S = qv["S"]
    n, N = S.shape
    scored = np.isfinite(S)
    y = np.zeros(S.shape, bool)
    y[np.arange(n), qv["pos"]] = True
    ev = np.broadcast_to(np.arange(n)[:, None], S.shape)
    pos_m, neg_m = scored & y, scored & ~y
    eng = PooledAUC(S[pos_m], ev[pos_m], S[neg_m], ev[neg_m])
    est_ref = auroc(np.concatenate([S[pos_m], S[neg_m]]),
                    np.concatenate([np.ones(int(pos_m.sum()), int), np.zeros(int(neg_m.sum()), int)]))
    est = eng.stat(np.ones(n, np.int32))
    _self_check("pooled AUROC", est, est_ref)
    reps = eng.reps(boot.W)
    blk = {"pooled_auroc": ci_dict(est, reps, int(scored.sum()))}
    blk["pooled_auroc"]["n_events"] = int(n)
    blk["pooled_auroc"]["n_positive_rows"] = int(pos_m.sum())
    blk["pooled_auroc"]["n_negative_rows"] = int(neg_m.sum())
    blk["pooled_auroc_minus_half"] = ci_dict(est - 0.5, reps - 0.5, int(scored.sum()), null=0.0)
    return blk


def sec_D(batch: LinearBatch, boot: EventBoot, qv: dict, m: np.ndarray, profile: np.ndarray) -> dict:
    """Selective serving (spec D): risk-coverage curves, AURC vs the random expectation, gain at 50%, niche coverage."""
    n = len(m)
    cols = np.flatnonzero(m)
    Wv = boot.W if len(cols) == n else boot.W[:, cols]
    nv = len(cols)
    utils = {"ndcg10": qv["ndcg10"][cols], "hr1": qv["hr1"][cols]}
    prof = profile[cols]
    has = np.isfinite(prof)
    quint = np.full(nv, -1)
    if has.any():
        quint[has] = rank_bins(prof[has], N_QUINT)
    out: dict = {"n_events": nv, "n_events_without_profile": int((~has).sum()),
                 "quintile_sizes": [int((quint == j).sum()) for j in range(N_QUINT)], "coverage_points": list(COVERAGE),
                 "signals": {}}
    for sg in SIGNALS:
        sig = qv[sg][cols]
        eng = ServeEngine(sig, utils, quint)
        reps = eng.reps(Wv)
        est = eng.reps(np.ones((1, nv), np.int16))
        blk: dict = {"aurc": {}, "d_aurc_vs_random_expectation": {}, "random_expectation_aurc": {}, "mean_utility": {},
                     "curve": {}, "gain_at_50_vs_full": {}, "niche": {"utility_among_served": {}}}
        for u in UTILS:
            e_aurc = est["aurc"][u][0]
            if nv:
                _, cum, aurc_ref = risk_coverage(sig, utils[u])
                _self_check(f"AURC {sg}/{u}", e_aurc, aurc_ref)
                ks = np.minimum(np.maximum(1, np.rint(np.asarray(COVERAGE) * nv).astype(int)), nv)
                for j, k in enumerate(ks):
                    _self_check(f"risk-coverage curve {sg}/{u}/{COVERAGE[j]}", est["curve"][u][0, j], cum[k - 1])
            blk["aurc"][u] = ci_dict(e_aurc, reps["aurc"][u], nv)
            rnd_est, rnd_reps = 1.0 - est["mean_u"][u][0], 1.0 - reps["mean_u"][u]
            blk["random_expectation_aurc"][u] = ci_dict(rnd_est, rnd_reps, nv)
            blk["mean_utility"][u] = ci_dict(est["mean_u"][u][0], reps["mean_u"][u], nv)
            blk["d_aurc_vs_random_expectation"][u] = ci_dict(e_aurc - rnd_est, reps["aurc"][u] - rnd_reps, nv, null=0.0)
            blk["curve"][u] = [{"coverage": c, **ci_dict(est["curve"][u][0, j], reps["curve"][u][:, j], nv)}
                               for j, c in enumerate(COVERAGE)]
            blk["gain_at_50_vs_full"][u] = ci_dict(
                est["curve"][u][0, HALF] - est["mean_u"][u][0], reps["curve"][u][:, HALF] - reps["mean_u"][u], nv,
                null=0.0)
            blk["niche"]["utility_among_served"][u] = [
                ci_dict(est["served_util"][u][0, j], reps["served_util"][u][:, j], int(est["size"][0, j]))
                for j in range(N_QUINT)]
        with np.errstate(divide="ignore", invalid="ignore"):
            sh_est = est["served"][0] / est["size"][0]
            sh_reps = reps["served"] / reps["size"]
        blk["niche"]["served_share"] = [ci_dict(sh_est[j], sh_reps[:, j], int(est["size"][0, j])) for j in range(N_QUINT)]
        blk["niche"]["niche_minus_mainstream_served_share"] = ci_dict(
            sh_est[0] - sh_est[N_QUINT - 1], sh_reps[:, 0] - sh_reps[:, N_QUINT - 1],
            int(est["size"][0, 0] + est["size"][0, N_QUINT - 1]), null=0.0)
        out["signals"][sg] = blk
    return out


def sec_E(batch: LinearBatch, boot: EventBoot, qv: dict, m: np.ndarray, n_boot_bias: int) -> dict:
    """Popularity-graded confidence and calibration on the top-K candidates (spec E)."""
    n = len(m)
    G, p, y, lc = qv["top_grp"], qv["top_p"], qv["top_y"], qv["top_lc"]
    rowm = m[:, None] & qv["top_scored"]
    out: dict = {"n_rows": int(rowm.sum()), "n_events": int(m.sum()),
                 "n_top_slots_unscored_excluded": int((m[:, None] & ~qv["top_scored"]).sum()),
                 "mean_p": {}, "mean_centred_logit": {}, "residual_y_minus_p": {}}
    for g, name in enumerate(GROUPS):
        sel = rowm & (G == g)
        cnt = sel.sum(1).astype(float)
        out["mean_p"][name] = batch.add(np.where(sel, p, 0.0).sum(1), cnt)
        out["mean_centred_logit"][name] = batch.add(np.where(sel, np.nan_to_num(lc), 0.0).sum(1), cnt)
        out["residual_y_minus_p"][name] = batch.add(np.where(sel, y - p, 0.0).sum(1), cnt)
        out.setdefault("n_rows_by_group", {})[name] = int(sel.sum())
    out["head_minus_tail_mean_p"] = batch.derive(out["mean_p"]["head"], out["mean_p"]["tail"], null=0.0)
    out["head_minus_tail_residual"] = batch.derive(out["residual_y_minus_p"]["head"], out["residual_y_minus_p"]["tail"],
                                                   null=0.0)
    mf = m.astype(float)
    out["top1_head_fraction"] = batch.add((G[:, 0] == 0) * mf, mf)
    # bias_index: event-cluster bootstrap of metrics.bias_index (same stream as metrics.bias_index_ci)
    cols = np.flatnonzero(m)
    sel = rowm[cols]
    ev_rows = np.broadcast_to(np.arange(len(cols))[:, None], sel.shape)[sel]
    conf, corr, gid = p[cols][sel], y[cols][sel], G[cols][sel]
    est = bias_index(conf, corr, np.asarray(GROUPS)[gid], adjust=True) if len(conf) else {}
    eng = BiasEngine(conf, corr, gid, ev_rows)
    Wb = boot.W[:n_boot_bias] if len(cols) == n else boot.W[:n_boot_bias][:, cols]
    reps = eng.reps(Wb) if len(conf) else np.full((n_boot_bias, 3), np.nan)
    if len(conf):
        for g, name in enumerate(GROUPS):
            _self_check(f"bias_index {name}", eng.stat(np.ones(len(cols), np.int32))[g], est.get(name, NAN))
    bi_out = {name: ci_dict(est.get(name, NAN), reps[:, g], len(cols)) for g, name in enumerate(GROUPS)}
    bi_out["head_minus_tail"] = ci_dict(est.get("head", NAN) - est.get("tail", NAN), reps[:, 0] - reps[:, 2], len(cols),
                                        null=0.0)
    bi_out["n_boot"] = int(n_boot_bias)
    out["bias_index"] = bi_out
    return out


# ------------------------------------------------------------------------------------------------- orchestration
def make_segments(domain: str, E: int, quarantine_n) -> list[dict]:
    q = QUARANTINE.get(domain, 0) if quarantine_n is None else int(quarantine_n)
    if q and q >= E:
        return [{"name": f"events_1_{E}", "role": "quarantine", "lo": 0, "hi": E}]
    if q and q > 0:
        return [{"name": f"events_1_{q}", "role": "quarantine", "lo": 0, "hi": q},
                {"name": f"events_{q + 1}_{E}", "role": "main", "lo": q, "hi": E}]
    return [{"name": "all", "role": "all", "lo": 0, "hi": E}]


def discover_methods(directory) -> dict:
    d = Path(directory) if directory else None
    if d is None or not d.is_dir():
        return {}
    return {p.name[:-len(".csv.gz")]: p for p in sorted(d.glob("*.csv.gz")) if not p.name.startswith("_")}


def _data_diag(panel: Panel, qa: dict) -> dict:
    return {"n_events": int(len(qa["valid"])), "n_candidates_min": int(panel.n_cand.min()),
            "n_candidates_max": int(panel.n_cand.max()), "n_rows_scored": int(qa["n_scored"].sum()),
            "n_rows_unscored": int((qa["n_cand"] - qa["n_scored"]).sum()),
            "n_events_positive_unscored": int((~qa["pos_scored"]).sum()),
            "n_events_any_unscored": int(qa["n_any_unscored"].sum()),
            "n_events_no_scored_candidate": int((~qa["valid"]).sum())}


def analyze_domain(domain: str, panel: Panel, L_test: dict, valid: dict, refs_rank: dict, refs_expo: dict,
                   questions=QUESTIONS, n_boot: int = 2000, seed: int = 0, quarantine_n=None, load_diag=None) -> dict:
    """The whole per-domain analysis from loaded inputs. `valid` = {question: (S_valid (Ev, Nv) masked scores,
    pos_valid)}, refs_rank = {method: (rank array aligned to panel, diag)}, refs_expo = {method: dict from
    load_ref_exposure}."""
    t0 = time.time()
    E = panel.E
    if E == 0:
        raise ValueError("empty test panel")
    if panel.n_cand.min() < K:
        raise ValueError(f"every event needs at least {K} candidates (min {int(panel.n_cand.min())})")
    pool_size = len(panel.pool)
    rand = np.random.default_rng([int(seed), 1]).random(E)
    profile, prof_diag = user_profile(panel)
    Tinfo, qarr = {}, {}
    for q in questions:
        Sv, posv = valid[q]
        Tinfo[q] = fit_temperature(Sv, posv)
        T_test = Tinfo[q]["beta"]
        qarr[q] = question_arrays(L_test[q], panel, T_test if np.isfinite(T_test) else 1.0, rand)
        D, Dp, use = _centred(qarr[q]["S"], panel.pos)
        Tinfo[q]["test"] = {"nll_T1": mean_nll(qarr[q]["S"], panel.pos, 1.0)[0],
                            "nll_fit": mean_nll(qarr[q]["S"], panel.pos, Tinfo[q]["beta"])[0]
                            if np.isfinite(Tinfo[q]["beta"]) else NAN,
                            "nll_uniform": float(np.mean(np.log(qarr[q]["n_scored"][use]))) if use.any() else NAN,
                            "n_events": int(use.sum())}
        _log(f"{domain}/{q}: T={Tinfo[q]['T']!r} (valid events {Tinfo[q]['n_events_used']})")
    result = {"schema": SCHEMA, "domain": domain, "questions": list(questions), "n_boot": int(n_boot), "seed": int(seed),
              "definition": DEFINITIONS, "n_events": E, "n_pool_items": pool_size,
              "data": {"panel": {"n_events": E, "n_candidates_min": int(panel.n_cand.min()),
                                 "n_candidates_max": int(panel.n_cand.max()), "n_pool_items": pool_size,
                                 "n_pool_group_conflicts": int(panel.pool_conflicts),
                                 "n_users_with_several_events": panel.n_users_multi_event, **prof_diag},
                       "scores_test": load_diag or {}, "by_question": {q: _data_diag(panel, qarr[q]) for q in questions},
                       "ref_ranks": {m: d for m, (_, d) in refs_rank.items()},
                       "ref_exposure": {m: d["diag"] for m, d in refs_expo.items()}},
              "temperature": Tinfo, "segments": {}}
    methods = sorted(set(refs_rank) | set(refs_expo))
    for seg in make_segments(domain, E, quarantine_n):
        sl = slice(seg["lo"], seg["hi"])
        n = seg["hi"] - seg["lo"]
        boot = EventBoot(n, n_boot, seed)
        batch = LinearBatch(boot)
        S: dict = {"role": seg["role"], "event_range": [seg["lo"] + 1, seg["hi"]], "n_events": n, "questions": {},
                   "reference": {}}
        prof_seg = profile[sl]
        for q in questions:
            qv = _seg(qarr[q], sl)
            m = qv["valid"]
            A = {"n_events": n, "ranking": block_ranking(batch, np.ones(n, bool), {
                "ndcg10": qv["ndcg10"], "hr10": qv["hr10"], "mrr": qv["mrr"], "ndcg10_tie_exact": qv["ndcg10_x"],
                "hr10_tie_exact": qv["hr10_x"], "mrr_tie_exact": qv["mrr_x"]}),
                 "censoring": {"n_events_positive_unscored": int((~qv["pos_scored"]).sum()),
                               "n_events_any_unscored": int(qv["n_any_unscored"].sum()),
                               "n_events_no_scored_candidate": int((~m).sum()),
                               "n_rows_unscored": int((qv["n_cand"] - qv["n_scored"]).sum())},
                 "vs_reference": {}}
            B = {"n_events": int(m.sum()), "n_events_cutoff_tied": int((qv["cutoff_tied"] & m).sum()),
                 "n_events_top1_tied": int((qv["tied_top1"] & m).sum()),
                 "llm": block_exposure(batch, boot, qv["top_grp"], qv["pool_cnt"], qv["pos_grp"], qv["top_item"], m,
                                       pool_size, n_boot, tie_exp=qv["tie_exp_share"]),
                 "vs_reference": {}}
            B["llm"]["top1_acc_tie_expected"] = batch.add(qv["top1_acc_tie_exp"] * m, m.astype(float))
            for meth in methods:
                if meth in refs_rank:
                    rk = refs_rank[meth][0][sl]
                    mj = np.isfinite(rk)
                    rkf = np.where(mj, rk, 1.0)
                    mp = mj
                    mpf = mp.astype(float)
                    nd_ref = ndcg_from_rank(rkf, K)
                    A["vs_reference"][meth] = {
                        "n_joined": int(mp.sum()),
                        "dndcg10": batch.add((qv["ndcg10"] - nd_ref) * mpf, mpf, null=0.0),
                        "dhr10": batch.add((qv["hr10"] - (rkf <= K)) * mpf, mpf, null=0.0),
                        "dndcg10_tie_exact": batch.add((qv["ndcg10_x"] - nd_ref) * mpf, mpf, null=0.0)}
                if meth in refs_expo:
                    ex = refs_expo[meth]
                    mj = ex["member"][sl] & m
                    B["vs_reference"][meth] = {"n_joined": int(mj.sum()),
                                               **exposure_vs(batch, qv["top_grp"], ex["top_grp"][sl], mj)}
            C = sec_C(batch, boot, qv, m, Tinfo[q])
            D = sec_D(batch, boot, qv, m, prof_seg)
            Esec = sec_E(batch, boot, qv, m, min(BIAS_BOOT, n_boot))
            S["questions"][q] = {"n_events_valid": int(m.sum()), "A_ranking": A, "B_exposure": B, "C_calibration": C,
                                 "D_selective_serving": D, "E_popularity": Esec}
            _log(f"{domain}/{seg['name']}/{q}: sections done ({time.time() - t0:.0f}s)")
        for meth in methods:
            R: dict = {}
            if meth in refs_rank:
                rk = refs_rank[meth][0][sl]
                mj = np.isfinite(rk)
                rkf = np.where(mj, rk, 1.0)
                R["A_ranking"] = {"n_events": int(mj.sum()), "ranking": block_ranking(batch, mj, {
                    "ndcg10": ndcg_from_rank(rkf, K), "hr10": (rkf <= K).astype(float), "mrr": 1.0 / rkf})}
            if meth in refs_expo:
                ex = refs_expo[meth]
                mj = ex["member"][sl]
                R["B_exposure"] = block_exposure(batch, boot, ex["top_grp"][sl], ex["pool_cnt"][sl], ex["pos_grp"][sl],
                                                 ex["top_item"][sl], mj, pool_size, n_boot)
            S["reference"][meth] = R
        batch.run()
        result["segments"][seg["name"]] = resolve(S)
    result["timing_s"] = round(time.time() - t0, 1)
    return result


def run_domain(domain: str, audit_dir, panel_test, panel_valid, ref_ranks=None, ref_exposure=None,
               questions=QUESTIONS, n_boot: int = 2000, seed: int = 0, quarantine_n=None, out=None) -> dict:
    t0 = time.time()
    audit = Path(audit_dir)
    s_test = audit / f"{domain}_test" / "scores.csv.gz"
    s_valid = audit / f"{domain}_valid2k" / "scores.csv.gz"
    for p in (s_test, s_valid, Path(panel_test), Path(panel_valid)):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    panel = load_panel(panel_test)
    _log(f"{domain}: test panel {panel.E} events x {panel.N} candidates, pool {len(panel.pool)} items "
         f"({time.time() - t0:.0f}s)")
    L_test, diag = load_scores(s_test, panel, questions)
    _log(f"{domain}: test scores {diag.get('score_rows')} rows ({time.time() - t0:.0f}s)")
    vpanel = load_panel(panel_valid, only_events=scan_event_ids(s_valid), need_groups=False)
    L_valid, vdiag = load_scores(s_valid, vpanel, questions)
    valid = {}
    for q in questions:
        valid[q] = (np.where(np.isfinite(L_valid[q]) & vpanel.cmask, L_valid[q], -np.inf), vpanel.pos)
    rr = {m: align_ref_ranks(p, panel) for m, p in discover_methods(ref_ranks).items()}
    rx = {m: load_ref_exposure(p, panel) for m, p in discover_methods(ref_exposure).items()}
    pool_json = Path(ref_exposure) / "_pool.json" if ref_exposure else None
    res = analyze_domain(domain, panel, L_test, valid, rr, rx, questions, n_boot, seed, quarantine_n, diag)
    res["data"]["scores_valid"] = vdiag
    res["data"]["valid_events"] = vpanel.E
    if pool_json is not None and pool_json.exists():
        pj = json.loads(pool_json.read_text(encoding="utf-8"))
        res["data"]["pool_json"] = {"n_pool_items": pj.get("n_pool_items"),
                                    "matches_panel": pj.get("n_pool_items") == len(panel.pool)}
    res["inputs"] = {"audit_dir": str(audit), "panel_test": str(panel_test), "panel_valid": str(panel_valid),
                     "ref_ranks": str(ref_ranks) if ref_ranks else None,
                     "ref_exposure": str(ref_exposure) if ref_exposure else None}
    res["timing_s"] = round(time.time() - t0, 1)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(strict_json(res), indent=1, allow_nan=False), encoding="utf-8")
    return res


# -------------------------------------------------------------------------------------------------- summarize
def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for {key: p} (NaN p-values stay NaN and are not counted in m)."""
    items = sorted(((p, k) for k, p in pvals.items() if p is not None and p == p), key=lambda t: t[0])
    m = len(items)
    adj, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    for k in pvals:
        adj.setdefault(k, NAN)
    return adj


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _excl(c) -> int:
    """+1 / -1 when the CI excludes 0 on that side, else 0."""
    if not c or c.get("lo") is None or c.get("hi") is None:
        return 0
    return 1 if c["lo"] > 0 else -1 if c["hi"] < 0 else 0


# (name, path inside segments[seg]["questions"][q], signed: has a p_boot sign statement)
Q_ENDPOINTS = {
    "S1": [("unsure_correct_rate", ("C_calibration", "error_anatomy", "unsure_correct_rate"), False),
           ("share_correct_in_bottom_tertile", ("C_calibration", "error_anatomy", "share_correct_in_bottom_tertile"), False),
           ("acc_top_minus_bottom_tertile", ("C_calibration", "error_anatomy", "acc_top_minus_bottom_tertile"), True),
           ("auroc_pmax_minus_random_top1", ("C_calibration", "auroc_discrimination", "top1", "pmax_minus_random"), True)],
    "S2": [("confident_error_rate", ("C_calibration", "error_anatomy", "confident_error_rate"), False),
           ("share_errors_in_top_tertile", ("C_calibration", "error_anatomy", "share_errors_in_top_tertile"), False),
           ("share_errors_in_top_minus_third", ("C_calibration", "error_anatomy", "share_errors_in_top_minus_third"), True),
           ("auroc_pmax_minus_random_hr10", ("C_calibration", "auroc_discrimination", "hr10", "pmax_minus_random"), True)],
    "S4": [("pooled_auroc", ("C_calibration", "pointwise", "pooled_auroc"), False),
           ("pooled_auroc_minus_half", ("C_calibration", "pointwise", "pooled_auroc_minus_half"), True),
           ("uauc", ("C_calibration", "pointwise", "uauc"), False),
           ("uauc_minus_half", ("C_calibration", "pointwise", "uauc_minus_half"), True),
           ("auroc_pmax_top1", ("C_calibration", "auroc_discrimination", "top1", "auroc", "p_max"), False),
           ("ece", ("C_calibration", "list_normalised", "ece"), False),
           ("ece_adaptive", ("C_calibration", "list_normalised", "ece_adaptive"), False),
           ("brier", ("C_calibration", "list_normalised", "brier"), False),
           ("nll", ("C_calibration", "list_normalised", "nll"), False),
           ("top1_accuracy", ("C_calibration", "list_normalised", "top1_accuracy"), False)],
    "S5": [("head_minus_tail_mean_p", ("E_popularity", "head_minus_tail_mean_p"), True),
           ("bias_index_head", ("E_popularity", "bias_index", "head"), False),
           ("bias_index_tail", ("E_popularity", "bias_index", "tail"), False),
           ("bias_index_head_minus_tail", ("E_popularity", "bias_index", "head_minus_tail"), True),
           ("residual_head_minus_tail", ("E_popularity", "head_minus_tail_residual"), True),
           ("top1_head_fraction", ("E_popularity", "top1_head_fraction"), False)],
    "A": [("ndcg10", ("A_ranking", "ranking", "ndcg10"), False), ("hr10", ("A_ranking", "ranking", "hr10"), False),
          ("mrr", ("A_ranking", "ranking", "mrr"), False)],
}


def _units(docs: list) -> dict:
    """{unit key: (domain, role, segment dict)}; a quarantine segment of a domain with a main segment gets its own key."""
    units: dict = {}
    for doc in docs:
        d = doc["domain"]
        for name, seg in doc["segments"].items():
            role = seg.get("role", "all")
            units[d if role in ("all", "main") else f"{d}_{name}"] = (d, role, seg)
    return units


def summarize(docs: list, alpha: float = HOLM_ALPHA) -> dict:
    """Cross-domain summary (spec F): endpoint values per unit, Holm-adjusted bootstrap sign statements over the family
    domains (units with role all / main), and the S3 admission flags. Numbers and identifiers only."""
    units = _units(docs)
    family = [u for u, (_, role, _) in units.items() if role in ("all", "main")]
    qs = sorted({q for _, _, seg in units.values() for q in seg["questions"]})
    out: dict = {"schema": SUMMARY_SCHEMA, "alpha": alpha, "n_domains_expected": 4,
                 "domains": sorted({d for d, _, _ in units.values()}), "family_units": family,
                 "units": {u: {"domain": d, "role": role, "event_range": seg.get("event_range"),
                               "n_events": seg.get("n_events")} for u, (d, role, seg) in units.items()},
                 "complete_family": len(family) == 4}

    def sign_block(vals: dict) -> dict:
        fam = {u: vals[u] for u in family if u in vals and vals[u] is not None}
        padj = holm({u: c.get("p_boot", NAN) for u, c in fam.items()})
        return {"m": len(fam), "p_boot": {u: c.get("p_boot") for u, c in fam.items()}, "p_holm": padj,
                "sign": {u: (1 if c["est"] > 0 else -1 if c["est"] < 0 else 0) for u, c in fam.items()
                         if c.get("est") is not None},
                "reject_holm": {u: bool(padj[u] == padj[u] and padj[u] < alpha) for u in fam},
                "ci_excludes_0": {u: _excl(c) for u, c in fam.items()}}

    for s, specs in Q_ENDPOINTS.items():
        out[s] = {}
        for name, path, signed in specs:
            out[s][name] = {}
            for q in qs:
                vals = {u: _get(seg, ("questions", q) + path) for u, (_, _, seg) in units.items()}
                vals = {u: c for u, c in vals.items() if c is not None}
                if vals:
                    ent = {"values": vals}
                    if signed:
                        ent["holm"] = sign_block(vals)
                    out[s][name][q] = ent
    # S3 ---------------------------------------------------------------------------------------------
    s3: dict = {"endpoints": {}, "admission": {}}
    method_names = sorted({m for _, _, seg in units.values() for m in seg.get("reference", {})
                           if "B_exposure" in seg["reference"][m]})
    exposure_keys = ("delta_head", "delta_mid", "delta_tail", "delta_head_vs_target", "head_share_top10",
                     "pool_head_share", "target_head_share", "coverage_top10", "gini_exposure")

    def collect(getter) -> dict:
        return {u: getter(seg) for u, (_, _, seg) in units.items() if getter(seg) is not None}

    entities = {f"llm_{q}": (lambda seg, q=q: _get(seg, ("questions", q, "B_exposure", "llm"))) for q in qs}
    entities.update({m: (lambda seg, m=m: _get(seg, ("reference", m, "B_exposure"))) for m in method_names})
    for ent, fn in entities.items():
        blk = {}
        for key in exposure_keys:
            vals = collect(lambda seg: _get(fn(seg), (key,)) if fn(seg) else None)
            if vals:
                blk[key] = {"values": vals, **({"holm": sign_block(vals)} if key.startswith("delta_") else {})}
        if not blk:
            continue
        s3["endpoints"][ent] = blk
        d_head = {u: c for u, c in blk.get("delta_head", {}).get("values", {}).items() if u in family}
        n_pos = sum(1 for c in d_head.values() if _excl(c) == 1)
        n_neg = sum(1 for c in d_head.values() if _excl(c) == -1)
        complete = len(d_head) >= 4                     # the registered rule is 3 of the 4 domains: all 4 must be present
        claimed = complete and max(n_pos, n_neg) >= 3
        s3["admission"][ent] = {"n_domains_with_data": len(d_head), "n_domains_expected": 4,
                                "complete_family": bool(complete), "n_domains_ci_excludes_0_positive": n_pos,
                                "n_domains_ci_excludes_0_negative": n_neg, "required": 3,
                                "effect_claimed": bool(claimed),
                                "sign": (1 if n_pos >= 3 else -1) if claimed else 0,
                                "per_domain_sign": {u: _excl(c) for u, c in d_head.items()}}
    for q in qs:
        # LLM minus reference head-share differences (context for the S3 comparison, with Holm over domains)
        for m in method_names:
            for key in ("head_share_top10_llm_minus_ref",):
                vals = collect(lambda seg: _get(seg, ("questions", q, "B_exposure", "vs_reference", m, key)))
                if vals:
                    s3["endpoints"].setdefault(f"llm_{q}_vs_{m}", {})[key] = {"values": vals, "holm": sign_block(vals)}
    out["S3"] = s3
    # D selective serving + A paired context --------------------------------------------------------
    d_out: dict = {}
    for q in qs:
        d_out[q] = {}
        for sg in SIGNALS:
            ent = {}
            for key, path in (("d_aurc_ndcg10", ("d_aurc_vs_random_expectation", "ndcg10")),
                              ("d_aurc_hr1", ("d_aurc_vs_random_expectation", "hr1")),
                              ("gain50_ndcg10", ("gain_at_50_vs_full", "ndcg10")),
                              ("gain50_hr1", ("gain_at_50_vs_full", "hr1")),
                              ("niche_minus_mainstream_served_share", ("niche", "niche_minus_mainstream_served_share"))):
                vals = collect(lambda seg: _get(seg, ("questions", q, "D_selective_serving", "signals", sg) + path))
                if vals:
                    ent[key] = {"values": vals, "holm": sign_block(vals)}
            if ent:
                d_out[q][sg] = ent
    out["D"] = d_out
    ctx: dict = {}
    for q in qs:
        ctx[q] = {}
        for m in sorted({m for _, _, seg in units.values() for m in
                         _get(seg, ("questions", q, "A_ranking", "vs_reference")) or {}}):
            vals = collect(lambda seg: _get(seg, ("questions", q, "A_ranking", "vs_reference", m, "dndcg10")))
            if vals:
                ctx[q][m] = {"dndcg10_llm_minus_ref": {"values": vals, "holm": sign_block(vals)}}
    out["A_context"] = ctx
    out["temperature"] = {d["domain"]: {q: {k: t.get(k) for k in ("T", "beta", "nll_T1", "nll_fit", "nll_uniform",
                                                                  "bound_hit")} for q, t in d["temperature"].items()}
                          for d in docs}
    return out


# --------------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Full-scale next-item audit analysis (docs/sigir/NEXTITEM_AUDIT_SPEC.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="per-domain analysis")
    r.add_argument("--domain", required=True)
    r.add_argument("--audit_dir", required=True, help="holds <domain>_test/scores.csv.gz and <domain>_valid2k/scores.csv.gz")
    r.add_argument("--panel_test", required=True)
    r.add_argument("--panel_valid", required=True)
    r.add_argument("--ref_ranks", default=None)
    r.add_argument("--ref_exposure", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--n_boot", type=int, default=2000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--questions", default="next,like")
    r.add_argument("--quarantine_n", type=int, default=None, help="events held out as the quarantine segment "
                   "(default: 1000 for sports, none otherwise)")
    s = sub.add_parser("summarize", help="cross-domain summary of per-domain run outputs")
    s.add_argument("--inputs", required=True, help="comma-separated per-domain JSON files")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        res = run_domain(a.domain, a.audit_dir, a.panel_test, a.panel_valid, a.ref_ranks, a.ref_exposure,
                         tuple(x for x in a.questions.split(",") if x), a.n_boot, a.seed, a.quarantine_n, a.out)
        print(f"wrote {a.out} ({res['timing_s']}s, segments {list(res['segments'])})")
    else:
        docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in a.inputs.split(",") if p]
        res = summarize(docs)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(strict_json(res), indent=1, allow_nan=False), encoding="utf-8")
        print(f"wrote {a.out} (domains {res['domains']}, family {res['family_units']})")


if __name__ == "__main__":
    main()
