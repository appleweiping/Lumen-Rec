"""Amendment 2 section A: CPU artifact checks A2-A8 on the Pilot-1 outputs (A9 = report.json prompts/s, copied).

Binding spec: idea-stage/PREREG_AMENDMENT_2.md section A, with idea-stage/PREREG_AMENDMENT_1.md C1 (every CI resamples
users, or items for item-level statistics; Spearman averages ties). A1 (prefix sha1 / eligible counts) belongs to the
panel builders (scripts/sigir/build_confirm_panels.py). EVERY number written here is EXPLORATORY: it generates
hypotheses only, is never a gate and never an input to G1-G9. Section A's three decision-use rules change wording only;
they are reported under `wording_flags`.

    # server: Pilot-1 score files + panels + raw data
    python -m src.confrec.forensics --pilot_dir outputs/confrec/pilot1_mirror --panels outputs/confrec/panels \
        --raw data/raw --out outputs/confrec/forensics/forensics.json
    # local-only A8 (the C-CRP v3 ranking records exist only on the workstation)
    python -m src.confrec.forensics --ccrp_head_share \
        outputs/sports_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/tables/ranking_eval_records.csv \
        --sports_panel <copy of outputs/confrec/panels/sports_next_1k.jsonl> \
        --out outputs/confrec/forensics/forensics_a8_local.json

Inputs per panel NAME (--names, default ml1m_rated,toys_rated,sports_next_1k): PANELS/NAME.jsonl and
PILOT/NAME/{scores.csv.gz, swap_prior.csv.gz, report.json}, PILOT/NAME_nohist/{scores.csv.gz, report.json}.
L(q) = log P(Yes) - log P(No); a row whose logit is non-finite or flagged censored 2/3 is missing (as pilot_mirror).
Raw data (--raw): RAW/ml-1m/{ratings.dat,movies.dat} (or RAW itself when it holds ratings.dat) for ML-1M panels,
RAW/amazon_<domain>/ for Amazon panels (build_rated_panels loaders, first event per (user, item)).

Panel identity: sha1 of the panel file bytes vs report.json data_sha1 of the main and _nohist runs (pyes_scorer hashes
   the --data file); a mismatch marks the history-star block unusable and is reported under panel_identity.
A2 prior (rated): UAUC of pi(i) alone (pi = mean finite donor like-logit, as pilot_mirror) and of L_nohist alone (item
   mean of the hist_len-0 like-logit); split-half Spearman of pi over items, donors = the rows of swap_prior.csv.gz per
   item in file order, positions 1-4 vs 5-8 (a censored donor keeps its position); Spearman-Brown 8-donor reliability
   2r/(1+r) (the split-half reliability of the 8-donor pi the evidence arm uses: the wording rule reads it); evidence
   (L_like - pi) UAUC with the 8-donor pi and each 4-donor half, and its projection to a perfectly reliable pi by linear
   extrapolation in reliability (rho_4 = r, rho_8 = 2r/(1+r)): U8 + (U8 - U4) / r, reported only for r >= 0.2, clipped
   to [0, 1] (flagged) and also evaluated at the CI bounds of r.
A3 CF references on the panel pairs (rated, needs --raw): prior-only leave-user-out item mean (other users' first
   ratings of the item strictly before the candidate timestamp; panel candidate_timestamps, else rebuilt from raw), a
   shrunk variant (k pseudo-ratings at the leave-user-out global mean before t, full coverage) and the transductive
   all-time leave-user-out mean (reference ceiling only). Temporal biased MF (PRIMARY, amendment 2 A3 "non-transductive
   ... temporal biased MF"): T = numpy linear 0.8-quantile of all panel candidate timestamps (amendment 1 D, the
   split_panel --time_split convention); item parameters trained on every first (user, item) rating with ts < T
   except all panel users' candidate events; each panel user's (b_u, p_u) folded in from the user's own non-candidate
   events strictly before the user's first candidate (a user who joined after T has no ts < T rating at all);
   UAUC on the panel pairs with t >= T (non-transductive: no other user's rating at or after T enters) and, as
   sensitivities, on all pairs and without the fold-in. Leave-candidates-out biased MF (all first ratings except the
   candidate events, other users' later ratings included) as a TRANSDUCTIVE reference only. Both: CCD++ on CPU, dim
   32, fixed seed; UAUC of the MF score, of the personal residual MF - mu - b_u - b_i = p_u.q_i and of b_i over warm
   pairs (a cold pair's residual is identically 0 and would pin a cold user's AUC at 0.5; the residual UAUC with
   cold pairs is reported beside it). Global AUC of the history-star mean.
A4 MIRROR vs the correlation-matched binormal ensemble null, with the decision group's definition (pilot_mirror
   `ensemble_null`): per user (rows finite in both views, both classes) the views' AUCs clipped half a pair inside
   (0, 1), d = sqrt(2) Phi^-1(AUC), pooled within-class Pearson rho, predicted AUC of the equal-weight standardised sum
   Phi(((d_1 + d_2) / sqrt(2 + 2 rho)) / sqrt(2)). Primary: dUAUC(MIRROR - null), paired user bootstrap; the placebo
   pair (like, like_para) is the sanity check. Secondary: the gap residual (MIRROR - null) - (placebo - null), the
   raw-weight prediction of the unstandardised sum, the within-user z-sums, the pooled back-of-envelope of the
   deliberation (judge.md: ML-1M 0.608 / 0.597) and a parametric bootstrap of the null's small-sample bias under the
   fitted per-user binormal model (exact Gaussian two-view ensembles on each user's own labels), with the bias-
   corrected dUAUC.
A5 calibration decomposition (rated; 50% user split stats.user_halves(seed), fit on one half, ECE on the other):
   raw, intercept-only sigmoid(L + b), temperature-only sigmoid(L / T), full Platt, user-centred Platt (L minus the
   user's mean logit, then Platt), for like, mirror_half and placebo.
A6 error targets (rated): errors of the Platt-recalibrated like decision (eval half) and per-user top-k errors (k = the
   user's number of likes, ties broken by a seeded random order); detector AUROCs (|x|, the arm's own Platt margin or
   its own per-user top-k margin) with user-cluster CIs.
A7 quality-partialled popularity (rated, needs A3): partial Spearman (ranks residualised on the control ranks) of pi
   and v = (L_like - L_dislike)/2 against log1p(popularity), controlling for the prior-only item mean, plus the release
   year parsed from ML-1M titles, or description length and has_store on Amazon. Pair level (v, user-cluster CI) and
   item level (pi and item-mean v, item bootstrap). All-time popularity is primary (amendment 1 C2); the prior-count
   popularity and the shrunk prior mean are point-estimate sensitivities.
A8 (local, --ccrp_head_share ranking_eval_records.csv): per event, the top-10 head share expected under random ranking
   (= head share of the candidate pool) and C-CRP's top-10 head share (pred_ranked_item_ids[:10]); events 1-1000
   (= the Pilot-1 sports panel), 1001-10000 (reported separately, amendment 2 F) and all. Blocks are cut on the FILE
   row position (a malformed row is skipped inside its own block, never shifting an event across the boundary);
   --sports_panel PANELS/sports_next_1k.jsonl checks that file rows 1-1000 are the panel's events in order.
A9 prompts/s copied from each run's report.json, beside the amendment's reconciled values.
Wording flags (amendment 2 A decision use, wording only): pi_split_half_reliability_lt_0p7 (Spearman-Brown 8-donor),
   mf_personal_residual_uauc_le_0p55 (temporal MF, warm t >= T pairs), ensemble_null_residual_lo_le_0 (one-sided: MIRROR
   not shown above its null, the complement of pilot1_gate's ensemble_null_lo_gt_0); the literal two-sided CI reading,
   the raw half correlation, the all-pairs and transductive MF and the gap residual are recorded under `sensitivity`.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import re
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

from src.confrec.build_rated_panels import first_per_item, load_amazon, load_ml1m
from src.confrec.categories import CATEGORY
from src.confrec.metrics import auroc, brier, ece
from src.confrec.stats import (cluster_bootstrap, paired_bootstrap, percentile_ci, platt_apply, platt_fit,
                               rankdata_avg, sigmoid, spearman, strict_json, user_halves)

NAN = float("nan")
LABEL = "exploratory"
NAMES = ("ml1m_rated", "toys_rated", "sports_next_1k")
HALF_A, HALF_B = slice(0, 4), slice(4, 8)  # swap donors 1-4 vs 5-8, file order within an item
REGISTERED_PROMPTS_PER_S = {"ml1m_rated": 87.1, "toys_rated": 35.0, "sports_next_1k": 174.8}  # amendment 2 A9
MF_CUTOFF_QUANTILE = 0.8  # amendment 1 D: T = 80th percentile of candidate timestamps (split_panel --time_split)
PROJECTION_MIN_R = 0.2    # the U8 + (U8 - U4)/r projection multiplies the 4->8 donor gap by 1/r: not reported below
WORDING_RULES = {
    "pi_split_half_reliability_lt_0p7": "A2 'pi split-half reliability < 0.7': the evidence collapse is labelled "
                                        "uninterpretable. Read on the Spearman-Brown corrected split-half reliability "
                                        "2r/(1+r), i.e. the reliability of the 8-donor pi the evidence arm subtracts "
                                        "(the conventional split-half coefficient; the skeptic's ~0.9 expectation is "
                                        "an 8-donor reliability)",
    "mf_personal_residual_uauc_le_0p55": "A3 MF personal-residual UAUC <= 0.55: 'no personal signal' is stated as a "
                                         "property of consumed-item panels. Read on the TEMPORAL biased MF (item "
                                         "parameters from ts < T ratings, T = 0.8-quantile of the panel candidate "
                                         "timestamps; user parameters folded in from the user's own pre-candidate "
                                         "events) on the warm pairs with t >= T (non-transductive, amendment 2 A3)",
    "ensemble_null_residual_lo_le_0": "A4 'ensemble-null residual CI covers 0' (rule stated for ML-1M): MIRROR's edge "
                                      "is attributed to two-view diversity. Read one-sided on dUAUC(MIRROR - ensemble "
                                      "null): attributed unless the CI lower bound is > 0 (the complement of "
                                      "pilot1_gate's ensemble_null_lo_gt_0). A CI wholly below 0 leaves no edge "
                                      "beyond diversity either, and the per-user binormal null over-predicts a pure "
                                      "ensemble at ~20 candidates per user (see A4 caveats), so the literal two-sided "
                                      "reading would withhold the attribution exactly when MIRROR sits below the "
                                      "diversity-only prediction",
}
SENSITIVITY_RULES = {
    "pi_split_half_raw_r_lt_0p7": "the raw half correlation r of the 1-4 vs 5-8 donor means (the reliability of a "
                                  "4-donor pi) < 0.7",
    "mf_temporal_all_pairs_residual_uauc_le_0p55": "temporal MF residual UAUC on ALL panel pairs (pairs before T see "
                                                   "other users' ratings in [t, T)) <= 0.55",
    "mf_temporal_no_fold_in_residual_uauc_le_0p55": "temporal MF residual UAUC on the t >= T pairs with the ts < T "
                                                    "fit's own user parameters (users without a ts < T rating are "
                                                    "cold and drop out) <= 0.55",
    "mf_transductive_reference_residual_uauc_le_0p55": "leave-candidates-out (transductive) MF residual UAUC <= 0.55",
    "ensemble_null_residual_ci_covers_0_two_sided": "literal two-sided reading: lo <= 0 <= hi on dUAUC(MIRROR - null)",
    "ensemble_null_residual_bias_corrected_lo_le_0": "dUAUC(MIRROR - (null + per-user parametric null bias)) has CI "
                                                     "lo <= 0",
    "ensemble_null_gap_residual_lo_le_0": "the skeptic's residual (MIRROR - null) - (placebo - null) has CI lo <= 0",
}
ENSEMBLE_NULL_CAVEATS = [
    "The per-user binormal null is a plug-in estimate: with about 20 candidates per user the transform of noisy "
    "per-user AUCs over-predicts an exact equal-weight two-view ensemble, so dUAUC(MIRROR - null) is biased below 0 "
    "under an exact Gaussian ensemble (a placebo pair at rho ~0.95 is nearly unbiased).",
    "Exact Gaussian two-view nulls at the Pilot-1 ML-1M size (1,500 users x 20 candidates, 13 likes, AUC ~0.59, "
    "within-class rho 0.35, 300-500 bootstrap resamples), three independent probes: mean dUAUC(MIRROR - null) "
    "-0.0007 to -0.0015; the 95% CI lay wholly BELOW 0 in 1/6, 3/6 and 3/12 runs (nominal 2.5%) and never above 0. "
    "Hence the wording flag is one-sided (lo <= 0) and the literal two-sided reading is a sensitivity.",
    "null_bias_parametric re-estimates that bias on the data's own design (per-user labels, fitted d_k and rho); "
    "dUAUC_bias_corrected bootstraps observed - (null + per-user bias) (secondary). Validation: over 12 exact Gaussian "
    "nulls of 1,000 users x 20 candidates (rho 0.35, AUC ~0.59) the realised mean dUAUC was -0.00169 (se 0.00023) "
    "and the parametric estimate -0.00147 (se 0.00008).",
]
_NORM = NormalDist()
_YEAR = re.compile(r"\((\d{4})\)\s*$")


# ---------------------------------------------------------------- I/O
def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _open(path):
    path = str(path)
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.endswith(".gz") else \
        open(path, encoding="utf-8", newline="")


def read_jsonl(path) -> list:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def read_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def file_sha1(path) -> str:
    """sha1 of the file bytes (pyes_scorer.file_sha1, the report.json data_sha1)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_scores(path) -> dict:
    """scores.csv.gz -> {meta: {(event, cand_idx): (user, item, label)}, L: {question: {key: logit}}, censoring}.
    Drop rule identical to pilot_mirror.load_scores: a non-finite logit or censored 2/3 makes the (key, question)
    missing; censored 1 is kept; a duplicated (key, question) keeps the last row and is counted."""
    meta, L, cens, seen = {}, defaultdict(dict), defaultdict(Counter), set()
    with _open(path) as f:
        for r in csv.DictReader(f):
            q, k = r["question"], (str(r["source_event_id"]), int(r["cand_idx"]))
            code = (r.get("censored") or "na").strip()
            cens[q]["n"] += 1
            cens[q][code] += 1
            if (k, q) in seen:
                cens[q]["duplicate"] += 1
            seen.add((k, q))
            meta[k] = (str(r["user_id"]), str(r["item_id"]), int(float(r["label"])))
            lg = _num(r["logit"])
            if math.isfinite(lg) and code not in ("2", "3"):
                L[q][k] = lg
            else:
                L[q].pop(k, None)
                cens[q]["dropped_nonfinite"] += 1
    return {"meta": meta, "L": dict(L), "censoring": {q: dict(c) for q, c in sorted(cens.items())}}


def load_swap_positions(path, question: str = "like") -> dict:
    """swap_prior.csv.gz -> {pos: {item: [logit of donor 1, donor 2, ... in file order]}, censoring}. A donor row with
    a non-finite logit or censored 2/3 keeps its position as NaN, so the 1-4 / 5-8 halves stay donor-defined."""
    pos, cens = defaultdict(list), Counter()
    with _open(path) as f:
        for r in csv.DictReader(f):
            if (r.get("question") or question) != question:
                continue
            code, lg = (r.get("censored") or "na").strip(), _num(r["logit"])
            ok = math.isfinite(lg) and code not in ("2", "3")
            cens["n"] += 1
            cens[code] += 1
            cens["dropped_nonfinite"] += int(not ok)
            pos[str(r["item_id"])].append(lg if ok else NAN)
    return {"pos": dict(pos), "censoring": dict(cens)}


# ---------------------------------------------------------------- panel pairs
_STR_COLS = ("ev", "user", "item", "title", "text", "brand")
_FLOAT_COLS = ("pop", "pop_prior", "ts", "rating", "hist_mean")


def panel_pairs(rows: list) -> dict:
    """One entry per (event, candidate) of the panel, in file order. kind = rated when every row carries
    candidate_labels and history_ratings, else next_item (labels from positive_item_index when absent)."""
    rated = bool(rows) and all("candidate_labels" in r and "history_ratings" in r for r in rows)
    c = defaultdict(list)
    for r in rows:
        ev, u = str(r.get("source_event_id", r["user_id"])), str(r["user_id"])
        ids = [str(i) for i in r["candidate_item_ids"]]
        n = len(ids)
        lab = r.get("candidate_labels")
        if lab is None:
            lab = [int(k == int(r["positive_item_index"])) for k in range(n)]

        def get(name, default, _r=r, _n=n):
            v = _r.get(name)
            return list(v) if isinstance(v, list) and len(v) == _n else [default] * _n

        hist = [_num(x) for x in r.get("history_ratings") or []]
        hist = [x for x in hist if math.isfinite(x)]
        c["ev"] += [ev] * n
        c["user"] += [u] * n
        c["item"] += ids
        c["cand"] += list(range(n))
        c["label"] += [int(x) for x in lab]
        c["pop"] += [_num(x) for x in get("candidate_popularity", NAN)]
        c["pop_prior"] += [_num(x) for x in get("candidate_popularity_prior", NAN)]
        c["ts"] += [_num(x) for x in get("candidate_timestamps", NAN)]
        c["rating"] += [_num(x) for x in get("candidate_ratings", NAN)]
        c["title"] += [str(x) for x in get("candidate_titles", "")]
        c["text"] += [str(x) for x in get("candidate_texts", "")]
        c["brand"] += [str(x) for x in get("candidate_brands", "")]
        c["hist_mean"] += [float(np.mean(hist)) if hist else NAN] * n
    P = {k: np.array(c[k], dtype=object) for k in _STR_COLS}
    P.update({k: np.array(c[k], float) for k in _FLOAT_COLS})
    P["cand"], P["label"] = np.array(c["cand"], int), np.array(c["label"], int)
    P["n"], P["kind"] = len(c["ev"]), "rated" if rated else "next_item"
    P["source"] = str(rows[0].get("source", "")) if rows else ""
    index, dup = {}, 0
    for j, k in enumerate(zip(c["ev"], c["cand"])):
        dup += k in index
        index[k] = j
    P["index"], P["n_duplicate_pairs"] = index, dup
    return P


def attach(P: dict, sc: dict) -> tuple[dict, dict]:
    """{question: logit array aligned to the panel pairs} (NaN = missing) and join diagnostics; a score row whose
    item differs from the panel's candidate is excluded."""
    join, bad = Counter(), set()
    for k, (_, i, lab) in sc["meta"].items():
        j = P["index"].get(k)
        if j is None:
            join["score_rows_without_panel_pair"] += 1
        elif P["item"][j] != i:
            join["score_rows_item_mismatch"] += 1
            bad.add(k)
        elif int(P["label"][j]) != lab:
            join["score_rows_label_mismatch"] += 1
    keys = list(zip(P["ev"].tolist(), P["cand"].tolist()))
    join["panel_pairs_without_score_row"] = sum(k not in sc["meta"] for k in keys)
    L = {q: np.array([NAN if k in bad else d.get(k, NAN) for k in keys], float) for q, d in sc["L"].items()}
    return L, {k: v for k, v in join.items() if v}


# ---------------------------------------------------------------- shared statistics
def _groups(x) -> dict:
    """{value: row indices} in sorted value order."""
    x = np.asarray(x)
    if len(x) == 0:
        return {}
    uniq, inv = np.unique(x, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    return dict(zip(uniq.tolist(), np.split(order, np.cumsum(np.bincount(inv, minlength=len(uniq)))[:-1])))


def uauc(score, label, groups: dict, mask=None) -> dict:
    """{user: AUC} over the user's rows with a finite score (and in `mask`), for users with both classes."""
    s, y = np.asarray(score, float), np.asarray(label, int)
    ok = np.isfinite(s) if mask is None else (np.asarray(mask, bool) & np.isfinite(s))
    out = {}
    for u, idx in groups.items():
        idx = idx[ok[idx]]
        if 0 < y[idx].sum() < len(idx):
            out[u] = auroc(s[idx], y[idx])
    return out


def mean_ci(d: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of per-user values with a user-bootstrap percentile CI (stats.paired_bootstrap against 0)."""
    b = paired_bootstrap(d, {k: 0.0 for k in d}, n_boot=n_boot, seed=seed)
    return {"est": b["est"], "lo": b["lo"], "hi": b["hi"], "n_users": b["n"]}


def uauc_ci(score, label, groups, n_boot=2000, seed=0, mask=None) -> dict:
    s = np.asarray(score, float)
    ok = np.isfinite(s) if mask is None else (np.asarray(mask, bool) & np.isfinite(s))
    return {**mean_ci(uauc(s, label, groups, mask), n_boot, seed), "n_pairs": int(ok.sum())}


def unit_bootstrap(stat, n_units: int, n_boot: int = 2000, seed: int = 0) -> dict:
    """Percentile CI resampling independent units (items) with replacement."""
    est = float(stat(np.arange(n_units))) if n_units else NAN
    rng = np.random.default_rng(seed)
    boots = [float(stat(rng.integers(0, n_units, n_units))) for _ in range(n_boot)] if n_units else []
    lo, hi = percentile_ci(boots)
    return {"est": est, "lo": lo, "hi": hi, "n": int(n_units), "n_boot": n_boot}


def cluster_mean_ci(x, clusters, n_boot: int = 2000, seed: int = 0) -> dict:
    """stats.cluster_bootstrap(lambda idx: x[idx].mean(), clusters) vectorised: the same cluster draws (np.unique
    order, one rng.integers call per resample) and the same estimator (resampled sum / resampled row count), without
    concatenating member index arrays (10k singleton clusters made that the A8 bottleneck)."""
    x = np.asarray(x, float)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    inv = inv.reshape(-1)
    s = np.bincount(inv, x, minlength=len(uniq))
    c = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        boots.append(float(s[pick].sum() / c[pick].sum()))
    lo, hi = percentile_ci(boots)
    return {"est": float(x.mean()) if len(x) else NAN, "lo": lo, "hi": hi, "n_clusters": int(len(uniq)),
            "n_boot": n_boot}


def _fmean(v) -> float:
    v = [x for x in v if math.isfinite(x)]
    return float(np.mean(v)) if v else NAN


def _item_mean_map(values, items) -> dict:
    """{item: mean of the finite values on its rows}."""
    acc = defaultdict(list)
    for i, v in zip(items, values):
        if math.isfinite(v):
            acc[i].append(v)
    return {i: float(np.mean(v)) for i, v in acc.items()}


def _calib(p, y) -> dict:
    if len(y) == 0:
        return {"ECE": NAN, "ECE_adaptive": NAN, "Brier": NAN, "n": 0}
    return {"ECE": ece(p, y, 15), "ECE_adaptive": ece(p, y, 15, adaptive=True), "Brier": brier(p, y),
            "n": int(len(y))}


def _auroc_ci(score, target, clusters, n_boot, seed) -> dict:
    s, e = np.asarray(score, float), np.asarray(target, int)
    out = {"n": int(len(e)), "n_errors": int(e.sum())}
    if len(e) == 0 or e.min() == e.max():
        return {"est": NAN, "lo": NAN, "hi": NAN, **out}
    b = cluster_bootstrap(lambda idx: auroc(s[idx], e[idx]), clusters, n_boot=n_boot, seed=seed)
    return {"est": b["est"], "lo": b["lo"], "hi": b["hi"], "n_users": b["n_clusters"], **out}


# ---------------------------------------------------------------- A2 prior quantities
def projection_reliable(u8: float, u4: float, r: float) -> dict:
    """U8 + (U8 - U4)/r: linear extrapolation of the evidence UAUC in pi's reliability (rho_4 = r, rho_8 = 2r/(1+r))
    to rho = 1. The 1/r factor makes it unstable for a small r, so it is NaN below PROJECTION_MIN_R and clipped to
    [0, 1] (flagged) otherwise."""
    if not (math.isfinite(u8) and math.isfinite(u4) and math.isfinite(r)):
        return {"value": NAN, "unclipped": NAN, "clipped": None, "note": "non-finite input"}
    if r < PROJECTION_MIN_R:
        return {"value": NAN, "unclipped": NAN, "clipped": None,
                "note": f"not reported: split-half r = {r:.4f} < {PROJECTION_MIN_R}"}
    raw = u8 + (u8 - u4) / r
    val = min(max(raw, 0.0), 1.0)
    return {"value": val, "unclipped": raw, "clipped": val != raw, "note": None}


def prior_checks(P, groups, like, swap_pos: dict | None, nohist_like, n_boot=2000, seed=0):
    """A2 block, plus the per-pair 8-donor pi (NaN without a swap file) and {item: pi} for A6/A7."""
    y, items = P["label"], P["item"]
    out = {"label": LABEL}
    pi_p, pi_item = np.full(P["n"], NAN), None
    if swap_pos is not None:
        pi_item = {i: _fmean(v) for i, v in swap_pos.items()}
        pa = {i: _fmean(v[HALF_A]) for i, v in swap_pos.items()}
        pb = {i: _fmean(v[HALF_B]) for i, v in swap_pos.items()}
        pi_p = np.array([pi_item.get(i, NAN) for i in items], float)
        pa_p = np.array([pa.get(i, NAN) for i in items], float)
        pb_p = np.array([pb.get(i, NAN) for i in items], float)
        panel_items = sorted(set(items.tolist()))
        a = np.array([pa.get(i, NAN) for i in panel_items], float)
        b = np.array([pb.get(i, NAN) for i in panel_items], float)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        r = unit_bootstrap(lambda idx: spearman(a[idx], b[idx]), int(ok.sum()), n_boot, seed)

        def sb(x):
            return 2 * x / (1 + x) if math.isfinite(x) and x > -1 else NAN

        n_pos = Counter(len(swap_pos.get(i, ())) for i in panel_items)
        out["pi"] = {
            "definition": "pi(i) = mean finite like-logit of item i over its swap donors (pilot_mirror); halves = "
                          "donor rows 1-4 vs 5-8 of the item in swap_prior.csv.gz file order",
            "n_panel_items": len(panel_items), "n_panel_items_without_prior": int(sum(i not in pi_item
                                                                                       for i in panel_items)),
            "donor_rows_per_item": {str(k): v for k, v in sorted(n_pos.items())},
            "UAUC_pi": uauc_ci(pi_p, y, groups, n_boot, seed)}
        out["split_half"] = {
            "definition": "Spearman r over panel items of mean(donors 1-4) vs mean(donors 5-8), item bootstrap (r = "
                          "reliability of a 4-donor pi); spearman_brown_8 = 2r/(1+r) = split-half reliability of the "
                          "8-donor pi the evidence arm uses (read by the wording flag; raw r is its sensitivity)",
            "spearman_r": r, "spearman_brown_8": sb(r["est"]),
            "spearman_brown_8_ci": {"lo": sb(r["lo"]), "hi": sb(r["hi"])}, "n_items": int(ok.sum())}
        if like is not None:
            u8 = uauc_ci(like - pi_p, y, groups, n_boot, seed)
            ua = uauc_ci(like - pa_p, y, groups, n_boot, seed)
            ub = uauc_ci(like - pb_p, y, groups, n_boot, seed)
            u4 = float(np.mean([ua["est"], ub["est"]]))
            proj = projection_reliable(u8["est"], u4, r["est"])
            out["evidence"] = {
                "definition": "evidence = L_like - pi; UAUC with the 8-donor pi and each 4-donor half; projection to a "
                              "perfectly reliable pi = U8 + (U8 - U4)/r (linear in reliability: rho_4 = r, "
                              f"rho_8 = 2r/(1+r), rho = 1), reported only for r >= {PROJECTION_MIN_R} and clipped to "
                              "[0, 1]; also at the item-bootstrap CI bounds of r (instability range)",
                "UAUC_8_donor": u8, "UAUC_4_donor_half_1_4": ua, "UAUC_4_donor_half_5_8": ub, "UAUC_4_donor_mean": u4,
                "projection_reliable_pi": proj["value"], "projection_unclipped": proj["unclipped"],
                "projection_clipped": proj["clipped"], "projection_note": proj["note"],
                "projection_min_r": PROJECTION_MIN_R,
                "projection_at_r_ci": {"r_lo": projection_reliable(u8["est"], u4, r["lo"])["value"],
                                       "r_hi": projection_reliable(u8["est"], u4, r["hi"])["value"]}}
    if nohist_like is not None:
        m = _item_mean_map(nohist_like, items)
        nh = np.array([m.get(i, NAN) for i in items], float)
        out["L_nohist"] = {"definition": "item mean of the hist_len-0 like-logit (the prompt depends on the item only; "
                                         "as pilot_mirror's pmi_nohist prior)",
                           "UAUC_L_nohist": uauc_ci(nh, y, groups, n_boot, seed)}
    return out, pi_p, pi_item


# ---------------------------------------------------------------- A3 CF references
def scan_events(events: dict, need_items: set, cand_pairs: set) -> dict:
    """One pass over raw events (user -> [(ts, item, rating)]), first event per (user, item) as build_rated_panels:
    per-item event lists for the panel items, the global (ts, rating) stream, each panel user's own first events,
    the candidate events (own) and the leave-candidates-out MF training triples with their timestamps (mf_t; the
    temporal MF keeps mf_t < T); user_hist = each panel user's non-candidate first events (the temporal MF fold-in)."""
    uid, iid = {}, {}
    mf_u, mf_i, mf_r, mf_t = array("q"), array("q"), array("d"), array("d")
    g_ts, g_r = array("d"), array("d")
    item_events, own, user_own, user_hist = defaultdict(list), {}, {}, defaultdict(list)
    panel_users = {u for u, _ in cand_pairs}
    n_raw = n_first = 0
    for u, evs in events.items():
        n_raw += len(evs)
        fe = first_per_item(sorted(set(evs)))
        n_first += len(fe)
        if u in panel_users:
            user_own[u] = (np.array([e[0] for e in fe], float),
                           np.concatenate([[0.0], np.cumsum([float(e[2]) for e in fe])]))
        for t, i, r in fe:
            g_ts.append(float(t))
            g_r.append(float(r))
            if i in need_items:
                item_events[i].append((float(t), u, float(r)))
            if (u, i) in cand_pairs:
                own[(u, i)] = (float(t), float(r))
                continue
            if u in panel_users:
                user_hist[u].append((float(t), i, float(r)))
            mf_u.append(uid.setdefault(u, len(uid)))
            mf_i.append(iid.setdefault(i, len(iid)))
            mf_r.append(float(r))
            mf_t.append(float(t))
    ts = np.frombuffer(g_ts, dtype=np.float64) if len(g_ts) else np.zeros(0)
    rs = np.frombuffer(g_r, dtype=np.float64) if len(g_r) else np.zeros(0)
    order = np.argsort(ts, kind="stable")
    items_sorted = {}
    for i, lst in item_events.items():
        lst.sort(key=lambda e: e[0])
        items_sorted[i] = (np.array([e[0] for e in lst], float),
                           np.concatenate([[0.0], np.cumsum([e[2] for e in lst])]))
    return {"uid": uid, "iid": iid, "own": own, "user_own": user_own, "user_hist": dict(user_hist),
            "items": items_sorted,
            "global_ts": ts[order], "global_cs": np.concatenate([[0.0], np.cumsum(rs[order])]),
            "mf_u": np.frombuffer(mf_u, dtype=np.int64) if len(mf_u) else np.zeros(0, np.int64),
            "mf_i": np.frombuffer(mf_i, dtype=np.int64) if len(mf_i) else np.zeros(0, np.int64),
            "mf_r": np.frombuffer(mf_r, dtype=np.float64) if len(mf_r) else np.zeros(0),
            "mf_t": np.frombuffer(mf_t, dtype=np.float64) if len(mf_t) else np.zeros(0),
            "n_raw_events": n_raw, "n_first_events": n_first,
            "n_candidate_pairs_not_in_raw": len(cand_pairs) - len(own)}


def prior_means(scan: dict, users, items, times, shrink_k: float = 5.0) -> dict:
    """Per pair (u, i, t): mean / count of OTHER users' first ratings of i strictly before t (prior-only, leave-user-
    out); the leave-user-out global mean before t; the shrunk mean (s + k g)/(n + k); and, as a transductive reference
    ceiling only, the all-time leave-user-out item mean."""
    n = len(users)
    out = {k: np.full(n, NAN) for k in ("mean_prior", "n_prior", "global_prior", "mean_prior_shrunk",
                                         "mean_loo_alltime", "n_loo_alltime")}
    empty = (np.zeros(0), np.zeros(1))
    g_ts, g_cs = scan["global_ts"], scan["global_cs"]
    for j, (u, i, t) in enumerate(zip(users, items, times)):
        if not math.isfinite(t):
            continue
        ts, cs = scan["items"].get(i, empty)
        k = int(np.searchsorted(ts, t, "left"))
        s, m, tot, mt = cs[k], k, cs[-1], len(ts)
        o = scan["own"].get((u, i))
        if o is not None:  # the user's own first rating of i (the candidate itself) never enters
            tot, mt = tot - o[1], mt - 1
            if o[0] < t:
                s, m = s - o[1], m - 1
        out["n_prior"][j], out["n_loo_alltime"][j] = m, mt
        out["mean_prior"][j] = s / m if m > 0 else NAN
        out["mean_loo_alltime"][j] = tot / mt if mt > 0 else NAN
        kg = int(np.searchsorted(g_ts, t, "left"))
        ut, ucs = scan["user_own"].get(u, empty)
        ku = int(np.searchsorted(ut, t, "left"))
        gs, gn = g_cs[kg] - ucs[ku], kg - ku
        g = gs / gn if gn > 0 else NAN
        out["global_prior"][j] = g
        out["mean_prior_shrunk"][j] = (s + shrink_k * g) / (m + shrink_k) if math.isfinite(g) else NAN
    return out


def fit_biased_mf(u, i, r, n_users: int, n_items: int, dim: int = 32, iters: int = 15, lam: float = 0.05,
                  lam_bias: float = 5.0, seed: int = 0) -> dict:
    """Biased MF r ~ mu + b_u + b_i + p_u.q_i by cyclic coordinate descent (CCD++, Yu et al. ICDM'12: closed-form
    updates of the biases, then of one latent dimension at a time), numpy only and deterministic for a seed.
    Loss: sum (r - r_hat)^2 + lam_bias (b_u^2 + b_i^2) + lam (n_u |p_u|^2 + n_i |q_i|^2) (ALS-WR weighting).
    Users / items without training ratings keep zero bias and zero factors. P is (dim, n_users), Q (dim, n_items)."""
    u, i, r = np.asarray(u, np.int64), np.asarray(i, np.int64), np.asarray(r, float)
    rng = np.random.default_rng(seed)
    nu = np.bincount(u, minlength=n_users).astype(float)
    ni = np.bincount(i, minlength=n_items).astype(float)
    mu = float(r.mean()) if len(r) else 0.0
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    Pm = np.zeros((dim, n_users))
    Qm = rng.normal(0.0, 0.1, (dim, n_items)) * (ni > 0)
    e = r - mu

    def solve(idx, w, target, reg, n):
        num = np.bincount(idx, target * w, minlength=n)
        den = np.bincount(idx, w * w, minlength=n) + reg
        return np.divide(num, den, out=np.zeros(n), where=den > 0)

    for _ in range(iters):
        e += bu[u]
        bu = solve(u, np.ones_like(e), e, lam_bias, n_users)
        e -= bu[u]
        e += bi[i]
        bi = solve(i, np.ones_like(e), e, lam_bias, n_items)
        e -= bi[i]
        for t in range(dim):
            qt = Qm[t][i]
            e += Pm[t][u] * qt
            Pm[t] = solve(u, qt, e, lam * nu, n_users)
            pt = Pm[t][u]
            Qm[t] = solve(i, pt, e, lam * ni, n_items)
            e -= pt * Qm[t][i]
    return {"mu": mu, "bu": bu, "bi": bi, "P": Pm, "Q": Qm,
            "train_rmse": float(np.sqrt(np.mean(e ** 2))) if len(e) else NAN}


def mf_pair_scores(mf: dict, ui, ii) -> dict:
    """MF score, personal residual p_u.q_i and item bias b_i for (user index, item index) pairs; index -1 (no
    training rating) contributes zero bias and zero factors."""
    ui, ii = np.asarray(ui, np.int64), np.asarray(ii, np.int64)
    ku, ki = ui >= 0, ii >= 0
    su, si = np.maximum(ui, 0), np.maximum(ii, 0)
    bu = np.where(ku, mf["bu"][su], 0.0) if len(mf["bu"]) else np.zeros(len(ui))
    bi = np.where(ki, mf["bi"][si], 0.0) if len(mf["bi"]) else np.zeros(len(ii))
    pq = ((mf["P"][:, su] * mf["Q"][:, si]).sum(0) * (ku & ki)) if mf["P"].size and mf["Q"].size \
        else np.zeros(len(ui))
    return {"score": mf["mu"] + bu + bi + pq, "residual": pq, "item_bias": bi}


def mf_cutoff(times, q: float = MF_CUTOFF_QUANTILE) -> float:
    """Global temporal cutoff T = numpy linear q-quantile of all finite panel candidate timestamps (amendment 1 D; the
    same T as split_panel.time_split on the panel); NaN without timestamps."""
    t = np.asarray(times, float)
    t = t[np.isfinite(t)]
    return float(np.quantile(t, q)) if len(t) else NAN


def _warm_index(idx, counts) -> np.ndarray:
    """Training index of each pair's user/item, or -1 when it has no training rating in this model."""
    idx = np.asarray(idx, np.int64)
    out = np.full(len(idx), -1, np.int64)
    ok = idx >= 0
    if len(counts) and ok.any():
        out[ok] = np.where(np.asarray(counts)[idx[ok]] > 0, idx[ok], -1)
    return out


def fold_in_user(mf: dict, items_idx, ratings, lam: float, lam_bias: float) -> tuple[float, np.ndarray]:
    """(b_u, p_u) minimising sum (r - mu - b_i - b_u - p_u.q_i)^2 + lam_bias b_u^2 + lam n_u |p_u|^2 over the user's
    ratings with the item parameters of `mf` fixed (fit_biased_mf's user step, solved jointly). An item without
    training ratings has q_i = 0 and b_i = 0 in the model, so it informs b_u only."""
    ii, r = np.asarray(items_idx, np.int64), np.asarray(ratings, float)
    dim, n = mf["P"].shape[0], len(r)
    if n == 0:
        return 0.0, np.zeros(dim)
    X = np.column_stack([np.ones(n), mf["Q"][:, ii].T])
    D = np.diag([lam_bias] + [lam * n] * dim)
    w = np.linalg.lstsq(X.T @ X + D, X.T @ (r - mf["mu"] - mf["bi"][ii]), rcond=None)[0]
    return float(w[0]), w[1:]


def _mf_pair_block(mf, ui, ii, y, groups, held, mask, n_boot, seed) -> dict:
    """On the panel pairs in `mask`: UAUC of the MF score, the personal residual p_u.q_i and the item bias b_i, and
    the held-out RMSE of the candidates' own ratings, all over the WARM pairs (user and item both have training
    data; index -1 = cold). A cold pair's residual is identically 0, which would put a cold user's per-user AUC at
    exactly 0.5 and pull the residual UAUC toward 0.5, so the residual UAUC including cold pairs is reported
    separately."""
    s = mf_pair_scores(mf, ui, ii)
    warm = mask & (ui >= 0) & (ii >= 0)
    okh = warm & np.isfinite(held)
    return {"n_pairs": int(mask.sum()), "n_pairs_warm": int(warm.sum()),
            "n_pairs_cold_item": int((mask & (ii < 0)).sum()), "n_pairs_cold_user": int((mask & (ui < 0)).sum()),
            "heldout_candidate_rmse": float(np.sqrt(np.mean((s["score"][okh] - held[okh]) ** 2))) if okh.any()
            else NAN,
            "UAUC_mf": uauc_ci(s["score"], y, groups, n_boot, seed, warm),
            "UAUC_personal_residual": uauc_ci(s["residual"], y, groups, n_boot, seed, warm),
            "UAUC_item_bias": uauc_ci(s["item_bias"], y, groups, n_boot, seed, warm),
            "UAUC_personal_residual_incl_cold": uauc_ci(s["residual"], y, groups, n_boot, seed, mask)}


def cf_references(P, groups, events: dict, *, shrink_k=5.0, mf_dim=32, mf_iters=15, mf_lambda=0.05,
                  mf_lambda_bias=5.0, mf_cutoff_q=MF_CUTOFF_QUANTILE, n_boot=2000, seed=0) -> tuple[dict, dict]:
    """A3 block and the prior_means arrays (for A7)."""
    users, items, y = P["user"].tolist(), P["item"].tolist(), P["label"]
    print(f"[forensics]   A3: scanning {len(events)} users' raw events", flush=True)
    scan = scan_events(events, set(items), set(zip(users, items)))
    times = P["ts"]
    if len(times) and np.isfinite(times).all():
        ts_src = "panel candidate_timestamps"
    else:
        times = np.array([scan["own"].get((u, i), (NAN,))[0] for u, i in zip(users, items)], float)
        ts_src = "rebuilt from raw: the user's first event of the item (build_rated_panels convention)"
    pm = prior_means(scan, users, items, times, shrink_k)
    fin = np.isfinite(pm["mean_prior"])
    out = {"label": LABEL, "candidate_timestamps": ts_src, "n_pairs": int(P["n"]), "n_users": len(groups),
           "n_pairs_without_timestamp": int((~np.isfinite(times)).sum()),
           "n_raw_events": scan["n_raw_events"], "n_first_events": scan["n_first_events"],
           "n_candidate_pairs_not_in_raw": scan["n_candidate_pairs_not_in_raw"],
           "item_mean_prior": {
               "definition": "mean of other users' first ratings of the item strictly before the candidate timestamp "
                             "(leave-user-out, prior-only); pairs with no such rating are NaN and drop out of UAUC",
               "UAUC": uauc_ci(pm["mean_prior"], y, groups, n_boot, seed),
               "share_pairs_without_prior_rating": float(1 - fin.mean()) if len(fin) else NAN,
               "n_prior_median": float(np.nanmedian(pm["n_prior"])) if np.isfinite(pm["n_prior"]).any() else NAN},
           "item_mean_prior_shrunk": {
               "definition": f"(sum + k g) / (n + k), k = {shrink_k}, g = mean of all other users' first ratings "
                             "strictly before the candidate timestamp (full coverage)",
               "UAUC": uauc_ci(pm["mean_prior_shrunk"], y, groups, n_boot, seed)},
           "item_mean_loo_alltime_reference": {
               "definition": "TRANSDUCTIVE ceiling, reference only: other users' first ratings of the item at any "
                             "time (diag_rated_baselines item_mean_loo)",
               "UAUC": uauc_ci(pm["mean_loo_alltime"], y, groups, n_boot, seed)}}
    if mf_iters > 0:
        hyper = {"solver": "CCD++ (fit_biased_mf)", "dim": mf_dim, "iters": mf_iters, "lambda": mf_lambda,
                 "lambda_bias": mf_lambda_bias, "seed": seed}
        n_u, n_i = len(scan["uid"]), len(scan["iid"])
        ui = np.array([scan["uid"].get(u, -1) for u in users], np.int64)
        ii = np.array([scan["iid"].get(i, -1) for i in items], np.int64)
        held = np.where(np.isfinite(P["rating"]), P["rating"],
                        [scan["own"].get((u, i), (NAN, NAN))[1] for u, i in zip(users, items)])
        every = np.ones(P["n"], bool)
        # temporal biased MF (primary): first ratings with ts < T, all panel candidate events excluded
        T = mf_cutoff(times, mf_cutoff_q)
        if math.isfinite(T):
            tr = scan["mf_t"] < T
            print(f"[forensics]   A3: temporal biased MF on {int(tr.sum())} ratings with ts < T = {T} (dim {mf_dim}, "
                  f"{mf_iters} iters)", flush=True)
            mf_tm = fit_biased_mf(scan["mf_u"][tr], scan["mf_i"][tr], scan["mf_r"][tr], n_u, n_i, mf_dim, mf_iters,
                                  mf_lambda, mf_lambda_bias, seed)
            nu_t = np.bincount(scan["mf_u"][tr], minlength=n_u)
            ni_t = np.bincount(scan["mf_i"][tr], minlength=n_i)
            ui_t, ii_t = _warm_index(ui, nu_t), _warm_index(ii, ni_t)
            # fold-in: each panel user's (b_u, p_u) re-solved from the user's own non-candidate first events strictly
            # before the user's first candidate, item parameters fixed at the ts < T fit
            first_t = {}
            for u, t in zip(users, times):
                if math.isfinite(t):
                    first_t[u] = min(t, first_t.get(u, math.inf))
            mf_fold = {**mf_tm, "bu": mf_tm["bu"].copy(), "P": mf_tm["P"].copy()}
            fold_n, fold_warm = {}, set()
            for u in groups:
                k = scan["uid"].get(u)
                ev = [(i, r) for t, i, r in scan["user_hist"].get(u, ()) if t < first_t.get(u, -math.inf)]
                fold_n[u] = len(ev)
                if k is None:
                    continue
                idx = [scan["iid"][i] for i, _ in ev]
                mf_fold["bu"][k], mf_fold["P"][:, k] = fold_in_user(mf_tm, idx, [r for _, r in ev], mf_lambda,
                                                                    mf_lambda_bias)
                if any(ni_t[j] > 0 for j in idx):
                    fold_warm.add(u)
            ui_f = np.array([k if u in fold_warm else -1 for u, k in zip(users, ui)], np.int64)
            test = np.isfinite(times) & (times >= T)
            fn = np.array(list(fold_n.values()), float)
            out["mf_temporal"] = {
                "definition": "PRIMARY (amendment 2 A3 'temporal biased MF'): item parameters (mu, b_i, q_i) trained "
                              "on every first (user, item) rating with ts < T, except all panel users' candidate "
                              f"events; T = numpy linear {mf_cutoff_q}-quantile of all panel candidate timestamps "
                              "(amendment 1 D, split_panel --time_split); each panel user's (b_u, p_u) folded in "
                              "(fold_in_user) from the user's own non-candidate first events strictly before the "
                              "user's first candidate (users who joined after T have no ts < T rating). residual = "
                              "MF - mu - b_u - b_i = p_u.q_i; UAUCs over warm pairs (item with a ts < T rating, user "
                              "with a fold-in event on such an item). test_pairs_t_ge_T (the wording flag) = pairs "
                              "with t >= T: non-transductive (no other user's rating at or after T is used; the "
                              "user's own events predate the candidates). all_pairs_sensitivity adds the pairs "
                              "before T (their items' parameters see other users' ratings in [t, T)); "
                              "test_pairs_t_ge_T_no_fold_in keeps the ts < T fit's own user parameters",
                "cutoff": {"quantile": mf_cutoff_q, "T": T, "n_panel_pairs_t_ge_T": int(test.sum()),
                           "n_users_with_pairs_t_ge_T": int(len(set(P["user"][test].tolist())))},
                **hyper, "n_train": int(tr.sum()), "n_users_train": int((nu_t > 0).sum()),
                "n_items_train": int((ni_t > 0).sum()), "train_rmse": mf_tm["train_rmse"],
                "fold_in": {"n_panel_users": len(groups), "n_users_warm": len(fold_warm),
                            "events_per_user_median": float(np.median(fn)) if len(fn) else NAN,
                            "n_users_without_events": int((fn == 0).sum())},
                "test_pairs_t_ge_T": _mf_pair_block(mf_fold, ui_f, ii_t, y, groups, held, test, n_boot, seed),
                "all_pairs_sensitivity": _mf_pair_block(mf_fold, ui_f, ii_t, y, groups, held, every, n_boot, seed),
                "test_pairs_t_ge_T_no_fold_in": _mf_pair_block(mf_tm, ui_t, ii_t, y, groups, held, test, n_boot,
                                                               seed)}
        else:
            out["mf_temporal"] = {"skipped": "no finite candidate timestamp to place the temporal cutoff"}
        # leave-candidates-out biased MF: transductive reference
        print(f"[forensics]   A3: leave-candidates-out biased MF on {len(scan['mf_r'])} ratings (dim {mf_dim}, "
              f"{mf_iters} iters)", flush=True)
        mf = fit_biased_mf(scan["mf_u"], scan["mf_i"], scan["mf_r"], n_u, n_i, mf_dim, mf_iters, mf_lambda,
                           mf_lambda_bias, seed)
        out["mf_transductive_reference"] = {
            "definition": "TRANSDUCTIVE reference only: leave-candidates-out biased MF trained on every first (user, "
                          "item) rating of the raw data except all panel users' candidate events (other users' later "
                          "ratings are included, so it is not temporal); residual = MF - mu - b_u - b_i = p_u.q_i",
            **hyper, "n_train": int(len(scan["mf_r"])), "n_users_train": n_u, "n_items_train": n_i,
            "train_rmse": mf["train_rmse"],
            **_mf_pair_block(mf, ui, ii, y, groups, held, every, n_boot, seed)}
    else:
        out["mf_temporal"] = out["mf_transductive_reference"] = {"skipped": "--mf_iters 0"}
    h = P["hist_mean"]
    ok = np.isfinite(h)
    hh, yy = h[ok], y[ok]
    if ok.any() and 0 < yy.sum() < len(yy):
        b = cluster_bootstrap(lambda idx: auroc(hh[idx], yy[idx]), P["user"][ok], n_boot=n_boot, seed=seed)
        g = {"est": b["est"], "lo": b["lo"], "hi": b["hi"], "n_users": b["n_clusters"], "n_pairs": int(ok.sum())}
    else:
        g = {"est": NAN, "lo": NAN, "hi": NAN, "n_pairs": int(ok.sum())}
    out["history_star_mean"] = {"definition": "pooled (global) AUC of the mean of the panel row's history_ratings; "
                                              "constant within a user, so it measures user leniency only",
                                "AUC_global": g}
    return out, pm


# ---------------------------------------------------------------- A4 binormal ensemble null
def auc_to_d(auc: float, n_pos: int | None = None, n_neg: int | None = None) -> float:
    """Equal-variance binormal separation d = sqrt(2) Phi^-1(AUC); AUC clipped half a pair inside (0, 1), i.e. to
    [e, 1 - e] with e = 0.5 / (n_pos n_neg) (1e-6 when the counts are not given)."""
    if not math.isfinite(auc):
        return NAN
    e = 0.5 / (n_pos * n_neg) if n_pos and n_neg else 1e-6
    return math.sqrt(2) * _NORM.inv_cdf(min(max(auc, e), 1 - e))


def d_to_auc(d: float) -> float:
    return _NORM.cdf(d / math.sqrt(2)) if math.isfinite(d) else NAN


def binormal_ensemble_auc(auc1: float, auc2: float, rho: float) -> float:
    """The decision group's null (pilot_mirror.binormal_ensemble_auc, amendment 2 G7/A4): AUC of the equal-weight sum
    of two views standardised to unit within-class SD, equal-variance binormal, AUCs already inside (0, 1):
    Phi(((d_1 + d_2) / sqrt(2 + 2 rho)) / sqrt(2)); rho = -1 makes the sum constant within each class."""
    d = math.sqrt(2) * (_NORM.inv_cdf(auc1) + _NORM.inv_cdf(auc2))
    v = 2.0 + 2.0 * rho
    if v <= 0:
        return 1.0 if d > 0 else 0.0 if d < 0 else 0.5
    return float(_NORM.cdf(d / math.sqrt(v) / math.sqrt(2)))


def binormal_sum_auc(auc_x, auc_z, rho, n_pos=None, n_neg=None) -> float:
    """binormal_ensemble_auc after the half-pair clip of auc_to_d (NaN for a non-finite input)."""
    if not (math.isfinite(auc_x) and math.isfinite(auc_z) and math.isfinite(rho)):
        return NAN
    return binormal_ensemble_auc(d_to_auc(auc_to_d(auc_x, n_pos, n_neg)), d_to_auc(auc_to_d(auc_z, n_pos, n_neg)),
                                 rho)


def binormal_weighted_auc(auc_x, auc_z, rho, sd_x, sd_z, w_x=1.0, w_z=1.0, n_pos=None, n_neg=None) -> float:
    """Predicted AUC of w_x X + w_z Z on the views' own scales (sensitivity: MIRROR is the raw, not the standardised,
    sum): class-mean gaps d * sd, within-class SDs sd_x, sd_z and within-class correlation rho."""
    gap = w_x * auc_to_d(auc_x, n_pos, n_neg) * sd_x + w_z * auc_to_d(auc_z, n_pos, n_neg) * sd_z
    var = (w_x * sd_x) ** 2 + (w_z * sd_z) ** 2 + 2 * w_x * w_z * rho * sd_x * sd_z
    return d_to_auc(gap / math.sqrt(var)) if math.isfinite(var) and var > 0 else NAN


def within_class_corr(x1, x2, y) -> float:
    """Pooled within-class Pearson correlation (residuals from each class's means; a class in which a view is
    constant contributes exact zeros); NaN when either view is constant within every class. Same as
    pilot_mirror.within_class_corr."""
    x1, x2, y = np.asarray(x1, float), np.asarray(x2, float), np.asarray(y, int)
    r1, r2, var1, var2 = np.empty(len(y)), np.empty(len(y)), False, False
    for c in (0, 1):
        m = y == c
        if not m.any():
            continue
        p1, p2 = bool(np.ptp(x1[m]) > 0), bool(np.ptp(x2[m]) > 0)
        var1, var2 = var1 or p1, var2 or p2
        r1[m] = x1[m] - x1[m].mean() if p1 else 0.0
        r2[m] = x2[m] - x2[m].mean() if p2 else 0.0
    if not (var1 and var2):
        return NAN
    return float(np.clip((r1 * r2).sum() / math.sqrt((r1 * r1).sum() * (r2 * r2).sum()), -1.0, 1.0))


def within_class_sd(x, y) -> float:
    """Pooled within-class SD (df = n - 2)."""
    x, y = np.asarray(x, float), np.asarray(y, int)
    ss = sum(float(((x[y == c] - x[y == c].mean()) ** 2).sum()) for c in (0, 1) if (y == c).any())
    return math.sqrt(ss / (len(x) - 2)) if len(x) > 2 else NAN


def _zsum(x1, x2):
    """Within-user z-scores (ddof 0) of both views, summed; None if a view is constant for the user."""
    s1, s2 = x1.std(), x2.std()
    if not (s1 > 0 and s2 > 0):
        return None
    return (x1 - x1.mean()) / s1 + (x2 - x2.mean()) / s2


def ensemble_pair(x1, x2, observed, label, groups, mask) -> dict:
    """Per-user ensemble-null quantities for the views x1, x2 and the observed arm (rank-equivalent to x1 + x2) on the
    rows in `mask` (finite in both views), users with both classes: {obs, pred, pred_raw, a1, a2, rho, zsum} dicts
    and counts. pred = binormal_ensemble_auc of the clipped AUCs (the decision group's definition); pred_raw =
    binormal_weighted_auc of the raw sum x1 + x2 (sensitivity)."""
    out = {k: {} for k in ("obs", "pred", "pred_raw", "a1", "a2", "rho", "zsum")}
    n = Counter()
    for u, idx in groups.items():
        idx = idx[mask[idx]]
        y = label[idx]
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        n["users_both_classes"] += 1
        v1, v2 = x1[idx], x2[idx]
        z = _zsum(v1, v2)
        if z is not None:
            out["zsum"][u] = auroc(z, y)
        rho = within_class_corr(v1, v2, y)
        if not math.isfinite(rho):
            n["users_rho_undefined"] += 1
            continue
        a1, a2 = auroc(v1, y), auroc(v2, y)
        e = 0.5 / (n_pos * n_neg)
        c1, c2 = min(max(a1, e), 1 - e), min(max(a2, e), 1 - e)
        n["users_auc_clipped"] += int(c1 != a1 or c2 != a2)
        out["pred"][u], out["obs"][u] = binormal_ensemble_auc(c1, c2, rho), auroc(observed[idx], y)
        out["a1"][u], out["a2"][u], out["rho"][u] = a1, a2, rho
        pr = binormal_weighted_auc(a1, a2, rho, within_class_sd(v1, y), within_class_sd(v2, y), n_pos=n_pos,
                                   n_neg=n_neg)
        if math.isfinite(pr):
            out["pred_raw"][u] = pr
    out["counts"] = dict(n)
    return out


def _rowwise_auc(v, pos) -> np.ndarray:
    """AUC of each row of v (draws x candidates) for the boolean class vector pos; continuous draws (no ties)."""
    r = np.argsort(np.argsort(v, axis=1), axis=1) + 1.0
    n1 = int(pos.sum())
    return (r[:, pos].sum(1) - n1 * (n1 + 1) / 2) / (n1 * (len(pos) - n1))


def _rowwise_within_class_corr(v1, v2, y) -> np.ndarray:
    """within_class_corr of each row pair (a class of size 1 contributes zeros, as a constant class there)."""
    r1, r2 = np.zeros_like(v1), np.zeros_like(v2)
    for c in (0, 1):
        m = y == c
        if m.sum() > 1:
            r1[:, m] = v1[:, m] - v1[:, m].mean(1, keepdims=True)
            r2[:, m] = v2[:, m] - v2[:, m].mean(1, keepdims=True)
    den = np.sqrt((r1 * r1).sum(1) * (r2 * r2).sum(1))
    return np.clip(np.divide((r1 * r2).sum(1), den, out=np.full(len(den), NAN), where=den > 0), -1.0, 1.0)


def null_bias_sim(pair: dict, label, groups, mask, reps: int = 50, seed: int = 0) -> dict:
    """Parametric bootstrap of the ensemble null's small-sample bias under the fitted per-user binormal model. For
    each user with a null in `pair` (ensemble_pair output; the same rows: `mask` and both classes), `reps` draws of
    an EXACT equal-weight two-view ensemble on the user's own labels: unit-variance Gaussian views with class-mean
    gaps d_k = sqrt(2) Phi^-1(clipped AUC_k) and within-class correlation rho; per draw, AUC(view1 + view2) minus the
    null predicted from the draw's own clipped AUCs and within-class Pearson correlation. Returns {user: mean over
    draws}, the expected per-user (observed - null) of a pure two-view ensemble on this design."""
    rng = np.random.default_rng(seed)
    out = {}
    for u in pair["pred"]:
        idx = groups[u]
        idx = idx[mask[idx]]
        y = np.asarray(label[idx], int)
        pos = y == 1
        n1 = int(pos.sum())
        e = 0.5 / (n1 * (len(y) - n1))
        d1, d2 = (math.sqrt(2) * _NORM.inv_cdf(min(max(pair[k][u], e), 1 - e)) for k in ("a1", "a2"))
        rho = pair["rho"][u]
        z = rng.standard_normal((2, reps, len(y)))
        v1 = d1 * y + z[0]
        v2 = d2 * y + rho * z[0] + math.sqrt(max(0.0, 1.0 - rho * rho)) * z[1]
        obs, a1, a2 = _rowwise_auc(v1 + v2, pos), _rowwise_auc(v1, pos), _rowwise_auc(v2, pos)
        rh = _rowwise_within_class_corr(v1, v2, y)
        diffs = [o - binormal_ensemble_auc(min(max(x1, e), 1 - e), min(max(x2, e), 1 - e), q)
                 for o, x1, x2, q in zip(obs.tolist(), a1.tolist(), a2.tolist(), rh.tolist()) if math.isfinite(q)]
        if diffs:
            out[u] = float(np.mean(diffs))
    return out


def ensemble_null(P, groups, like, dislike, para, n_boot=2000, seed=0, null_sim_reps=50) -> dict:
    """A4: MIRROR (and, as the sanity check, the placebo) against the correlation-matched binormal ensemble null,
    with the decision group's definition (pilot_mirror ensemble_null block); see the module docstring."""
    y = P["label"]
    boot = dict(n_boot=n_boot, seed=seed)
    pairs = {"mirror_pair": (like, -dislike, like - dislike, ["like", "-dislike"]),
             "placebo_pair": (like, para, (like + para) / 2, ["like", "like_para"])}
    res, out = {}, {"label": LABEL,
                    "definition": "per user (rows finite in both views, both classes): AUC_k of each view clipped to "
                                  "[0.5/(n_pos n_neg), 1 - 0.5/(n_pos n_neg)], rho = pooled within-class Pearson "
                                  "correlation, predicted AUC of the equal-weight standardised sum Phi(((d_1 + d_2) / "
                                  "sqrt(2 + 2 rho)) / sqrt(2)), d_k = sqrt(2) Phi^-1(AUC_k) (= pilot_mirror "
                                  "ensemble_null, the decision group's definition). PRIMARY (wording flag): "
                                  "dUAUC_mirror_minus_ensemble_null = observed MIRROR - prediction, paired user "
                                  "bootstrap. Secondary: the placebo pair (sanity check), the difference in gaps "
                                  "(MIRROR - null) - (placebo - null), the raw-weight prediction of the unstandardised "
                                  "sum, the empirical within-user z-sums, the pooled back-of-envelope (judge.md) and "
                                  "the parametric null-bias correction (null_bias_sim); see caveats",
                    "caveats": ENSEMBLE_NULL_CAVEATS}
    for name, (x1, x2, obs_arm, views) in pairs.items():
        mask = np.isfinite(x1) & np.isfinite(x2)
        r = res[name] = ensemble_pair(x1, x2, obs_arm, y, groups, mask)
        rho = np.array(list(r["rho"].values()), float)
        d = paired_bootstrap(r["obs"], r["pred"], **boot)
        out[name] = {"views": views, "n_rows": int(mask.sum()), **r["counts"], "n_users": len(r["pred"]),
                     "UAUC_view1": _fmean(r["a1"].values()), "UAUC_view2": _fmean(r["a2"].values()),
                     "rho_within_class": {"mean": float(rho.mean()) if len(rho) else NAN,
                                          "median": float(np.median(rho)) if len(rho) else NAN},
                     "predicted_UAUC": _fmean(r["pred"].values()), "observed_UAUC": _fmean(r["obs"].values()),
                     "z_sum_UAUC": _fmean(r["zsum"].values()), "n_users_z_sum": len(r["zsum"]),
                     "dUAUC_observed_minus_predicted": d,
                     "sensitivity_raw_weight": {
                         "predicted_UAUC": _fmean(r["pred_raw"].values()),
                         "dUAUC_observed_minus_predicted": paired_bootstrap(r["obs"], r["pred_raw"], **boot)}}
        if null_sim_reps > 0:
            bias = null_bias_sim(r, y, groups, mask, null_sim_reps, seed)
            out[name]["null_bias_parametric"] = {
                "definition": "per user, mean over draws of (AUC of an exact Gaussian two-view ensemble - its binormal "
                              "null) on the user's own labels with the fitted d_k and rho (null_bias_sim); "
                              "dUAUC_bias_corrected = paired user bootstrap of observed - (null + per-user bias)",
                "reps": null_sim_reps, "seed": seed, "n_users": len(bias), "mean_bias": _fmean(bias.values()),
                "dUAUC_bias_corrected": paired_bootstrap(r["obs"], {u: r["pred"][u] + b for u, b in bias.items()},
                                                         **boot)}
    m, p = res["mirror_pair"], res["placebo_pair"]
    out["dUAUC_mirror_minus_ensemble_null"] = out["mirror_pair"]["dUAUC_observed_minus_predicted"]
    out["dUAUC_placebo_minus_ensemble_null"] = out["placebo_pair"]["dUAUC_observed_minus_predicted"]
    if null_sim_reps > 0:
        out["dUAUC_mirror_minus_ensemble_null_bias_corrected"] = \
            out["mirror_pair"]["null_bias_parametric"]["dUAUC_bias_corrected"]
    common = sorted(set(m["pred"]) & set(p["pred"]))
    out["gap_residual"] = {
        "definition": "per user (MIRROR - its null) - (placebo - its null), i.e. the observed MIRROR - placebo gap "
                      "minus the gap the null predicts (the skeptic's residual), user bootstrap",
        **mean_ci({u: (m["obs"][u] - m["pred"][u]) - (p["obs"][u] - p["pred"][u]) for u in common}, **boot)}
    out["dUAUC_zsum_mirror_minus_zsum_placebo"] = paired_bootstrap(m["zsum"], p["zsum"], **boot)
    ok = np.isfinite(like) & np.isfinite(dislike) & np.isfinite(para)
    if ok.sum() >= 3 and m["a1"] and p["a2"]:
        u_like, u_neg, u_para = (_fmean(d.values()) for d in (m["a1"], m["a2"], p["a2"]))
        rho_m, rho_p = spearman(like[ok], -dislike[ok]), spearman(like[ok], para[ok])
        pm, pp = binormal_sum_auc(u_like, u_neg, rho_m), binormal_sum_auc(u_like, u_para, rho_p)
        obs_gap = out["mirror_pair"]["observed_UAUC"] - out["placebo_pair"]["observed_UAUC"]
        out["pooled_back_of_envelope"] = {
            "definition": "deliberation arithmetic (lens_skeptic A7 / judge.md): binormal_sum_auc of the mean per-user "
                          "AUCs with the pooled pair-level Spearman correlation, vs the observed UAUCs",
            "rho_like_negdislike_pooled": rho_m, "rho_like_para_pooled": rho_p, "pred_mirror": pm,
            "pred_placebo": pp, "observed_gap": obs_gap, "predicted_gap": pm - pp, "residual": obs_gap - (pm - pp)}
    return out


# ---------------------------------------------------------------- A5 calibration decomposition
def fit_intercept(x, y) -> float:
    """b minimising the log loss of sigmoid(x + b): the gradient sum(sigmoid(x + b) - y) increases in b (bisection)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    lo, hi, target = -60.0, 60.0, float(y.sum())
    for _ in range(100):
        mid = (lo + hi) / 2
        if float(sigmoid(x + mid).sum()) > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def fit_slope(x, y) -> float:
    """a = 1/T minimising the log loss of sigmoid(a x): the gradient sum((sigmoid(a x) - y) x) is non-decreasing in a
    (bisection on [-50, 50])."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    lo, hi = -50.0, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if float(((sigmoid(mid * x) - y) * x).sum()) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def user_centre(x, groups) -> np.ndarray:
    """x minus the user's mean over the user's finite rows (labels are not used)."""
    x = np.asarray(x, float)
    out = np.full(len(x), NAN)
    for idx in groups.values():
        ok = idx[np.isfinite(x[idx])]
        if len(ok):
            out[ok] = x[ok] - x[ok].mean()
    return out


def _split(P, groups, seed):
    fit_users = user_halves(list(groups), seed)
    return np.array([u in fit_users for u in P["user"]], bool), len(fit_users)


def calibration_decomposition(P, groups, arms: dict, seed=0) -> dict:
    y = P["label"]
    in_fit, n_fit = _split(P, groups, seed)
    out = {"label": LABEL,
           "definition": "fit on the stats.user_halves(seed) half, ECE (15 equal-width bins) / adaptive ECE / Brier on "
                         "the other half: raw sigmoid(x); intercept-only sigmoid(x + b); temperature-only "
                         "sigmoid(x/T); Platt sigmoid(a x + b); user-centred Platt on x - mean_user(x)",
           "split": {"n_users_fit": n_fit, "n_users_eval": len(groups) - n_fit, "seed": seed}, "arms": {}}
    for name, x in arms.items():
        ok = np.isfinite(x)
        fit, ev = ok & in_fit, ok & ~in_fit
        if not ev.any() or not (0 < y[fit].sum() < fit.sum()):
            out["arms"][name] = {"note": "insufficient rows or a single class in a split half"}
            continue
        xf, yf, xe, ye = x[fit], y[fit], x[ev], y[ev]
        b, a, ab = fit_intercept(xf, yf), fit_slope(xf, yf), platt_fit(xf, yf)
        xc = user_centre(x, groups)
        abc = platt_fit(xc[fit], yf)
        out["arms"][name] = {
            "raw": _calib(sigmoid(xe), ye),
            "intercept_only": {"b": b, **_calib(sigmoid(xe + b), ye)},
            "temperature_only": {"a": a, "T": 1 / a if a != 0 else NAN, **_calib(sigmoid(a * xe), ye)},
            "platt": {"a": ab[0], "b": ab[1], **_calib(platt_apply(xe, ab), ye)},
            "user_centred_platt": {"a": abc[0], "b": abc[1], **_calib(platt_apply(xc[ev], abc), ye)},
            "label_rate_eval": float(ye.mean())}
    return out


# ---------------------------------------------------------------- A6 error targets
def topk_errors(score, label, groups, seed=0) -> tuple[np.ndarray, int]:
    """Per user, predict like for the top-k candidates by score (k = the user's number of likes among rows with a
    finite score; ties broken by a seeded random order). Returns the error indicator (-1 = no target: non-finite
    score or a single-class user) and the number of users whose k-boundary fell inside a tie group."""
    s, y = np.asarray(score, float), np.asarray(label, int)
    err = np.full(len(s), -1)
    rng = np.random.default_rng(seed)
    ties = 0
    for idx in groups.values():
        idx = idx[np.isfinite(s[idx])]
        yy = y[idx]
        k = int(yy.sum())
        if k == 0 or k == len(idx):
            continue
        order = np.lexsort((rng.random(len(idx)), -s[idx]))
        top = np.zeros(len(idx), bool)
        top[order[:k]] = True
        ties += bool(s[idx][order[k - 1]] == s[idx][order[k]])
        err[idx] = (top != (yy == 1)).astype(int)
    return err, ties


def topk_margin(x, target_rows, label, groups) -> np.ndarray:
    """|x - thr_u|, thr_u = midpoint of the user's k-th and (k+1)-th largest x over the user's target rows with
    finite x (k = the user's number of likes on those rows)."""
    x, y = np.asarray(x, float), np.asarray(label, int)
    out = np.full(len(x), NAN)
    for idx in groups.values():
        idx = idx[target_rows[idx] & np.isfinite(x[idx])]
        k = int(y[idx].sum())
        if k == 0 or k == len(idx):
            continue
        xs = np.sort(x[idx])[::-1]
        out[idx] = np.abs(x[idx] - (xs[k - 1] + xs[k]) / 2)
    return out


def error_targets(P, groups, arms: dict, base: str = "like", n_boot=2000, seed=0) -> dict:
    y, users = P["label"], P["user"]
    L = arms[base]
    in_fit, n_fit = _split(P, groups, seed)
    out = {"label": LABEL, "base_arm": base,
           "definition": "AUROC(-confidence, is_error) pooled over pairs with a user-cluster bootstrap CI; detectors "
                         "mag_<arm> = |x|, platt_margin_<arm> = |a x + b| (the arm's own Platt fit on the fit half), "
                         "topk_margin_<arm> = |x - thr_u| (the arm's own per-user top-k threshold)"}
    # P1.3's target, for reference: errors of the raw like decision sigmoid(L) >= 0.5
    ok = np.isfinite(L)
    e_raw = ((L[ok] >= 0) != (y[ok] == 1)).astype(int)
    out["raw_decision_reference"] = {"error_rate": float(e_raw.mean()) if ok.any() else NAN,
                                     "always_like_error_rate": float(1 - y[ok].mean()) if ok.any() else NAN,
                                     "AUROC_mag_like": auroc(-np.abs(L[ok]), e_raw) if ok.any() else NAN}
    # (a) errors after Platt scaling, on the eval half
    fit, ev = ok & in_fit, ok & ~in_fit
    if ev.any() and 0 < y[fit].sum() < fit.sum():
        ab = platt_fit(L[fit], y[fit])
        err = np.full(len(L), -1)
        err[ev] = ((platt_apply(L[ev], ab) >= 0.5) != (y[ev] == 1)).astype(int)
        majority = int(y[fit].mean() >= 0.5)
        blk = {"definition": "eval-half errors of the like decision sigmoid(a L + b) >= 0.5 (Platt fit on the other "
                             "half)", "platt": {"a": ab[0], "b": ab[1]},
               "error_rate": float(err[ev].mean()), "n_rows": int(ev.sum()), "n_errors": int(err[ev].sum()),
               "majority_class_error_rate": float(np.mean(y[ev] != majority)), "detectors": {}}
        dets = {}
        for name, x in arms.items():
            dets[f"mag_{name}"] = np.abs(x)
            fx = np.isfinite(x) & in_fit
            if 0 < y[fx].sum() < fx.sum():
                a_x = platt_fit(x[fx], y[fx])
                dets[f"platt_margin_{name}"] = np.abs(a_x[0] * x + a_x[1])
        for d, conf in dets.items():
            rows = (err >= 0) & np.isfinite(conf)
            blk["detectors"][d] = _auroc_ci(-conf[rows], err[rows], users[rows], n_boot, seed)
        out["after_platt"] = blk
    else:
        out["after_platt"] = {"note": "insufficient rows or a single class in a split half"}
    # (b) per-user top-k errors
    err, ties = topk_errors(L, y, groups, seed)
    tgt = err >= 0
    blk = {"definition": "per user, the top-k like-logit candidates are predicted likes, k = the user's number of "
                         "likes (ties at the boundary broken by a seeded random order); errors = FP + FN (FP = FN per "
                         "user)", "n_rows": int(tgt.sum()), "n_errors": int(err[tgt].sum()),
           "error_rate": float(err[tgt].mean()) if tgt.any() else NAN, "n_users_tie_at_boundary": ties,
           "detectors": {}}
    dets = {}
    for name, x in arms.items():
        dets[f"mag_{name}"] = np.abs(x)
        dets[f"topk_margin_{name}"] = topk_margin(x, tgt, y, groups)
    for d, conf in dets.items():
        rows = tgt & np.isfinite(conf)
        blk["detectors"][d] = _auroc_ci(-conf[rows], err[rows], users[rows], n_boot, seed)
    out["per_user_topk"] = blk
    return out


# ---------------------------------------------------------------- A7 quality-partialled popularity
def release_year(title: str) -> float:
    """ML-1M release year from a title ending in '(YYYY)'; NaN when absent."""
    m = _YEAR.search(str(title))
    return float(m.group(1)) if m else NAN


def partial_spearman(x, y, Z=None) -> float:
    """Spearman correlation of x and y partialled on the controls Z: average ranks of every variable over the complete
    rows, ranks of x and y residualised on [1, ranks of Z] by least squares, Pearson of the residuals."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.zeros((len(x), 0)) if Z is None else np.asarray(Z, float).reshape(len(x), -1)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    if ok.sum() < Z.shape[1] + 3:
        return NAN
    rx, ry = rankdata_avg(x[ok]), rankdata_avg(y[ok])
    R = np.column_stack([np.ones(int(ok.sum()))] + [rankdata_avg(Z[ok, j]) for j in range(Z.shape[1])])
    ex = rx - R @ np.linalg.lstsq(R, rx, rcond=None)[0]
    ey = ry - R @ np.linalg.lstsq(R, ry, rcond=None)[0]
    den = math.sqrt(float((ex * ex).sum() * (ey * ey).sum()))
    return float((ex * ey).sum() / den) if den > 1e-12 else NAN


def _desc_text(x) -> str:
    """build_rated_panels._text without the length-preserving surrogate replacement."""
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(_desc_text(v) for v in x if v)
    return str(x).strip()


def amazon_desc_lengths(raw: Path, domain: str, need: set) -> dict:
    """{parent_asin: len(description text)} for the needed items, read from the raw meta file."""
    out = {}
    with gzip.open(Path(raw) / f"amazon_{domain}" / f"meta_{CATEGORY[domain]}.jsonl.gz", "rt",
                   encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            i = m.get("parent_asin")
            if i in need:
                out[i] = len(_desc_text(m.get("description")))
    return out


def desc_length_from_text(text: str) -> float:
    """Fallback without raw meta: candidate_texts = 'Categories: <cats>. ' + description; strip up to the first '. '."""
    t = str(text)
    if t.startswith("Categories: "):
        k = t.find(". ")
        t = t[k + 2:] if k >= 0 else ""
    return float(len(t))


def popularity_partial(P, groups, pi_item: dict | None, v, prior: dict, controls: dict, n_boot=2000,
                       seed=0) -> dict:
    """A7 block. controls = {name: pair-level array} beyond the prior-only item mean (year / desc_len / has_store)."""
    items, users = P["item"], P["user"]
    uniq, inv = np.unique(items, return_inverse=True)
    inv = inv.reshape(-1)

    def item_mean(x):
        x = np.asarray(x, float)
        ok = np.isfinite(x)
        s = np.bincount(inv[ok], x[ok], minlength=len(uniq))
        c = np.bincount(inv[ok], minlength=len(uniq))
        return np.divide(s, c, out=np.full(len(uniq), NAN), where=c > 0)

    pi_i = np.array([pi_item.get(i, NAN) for i in uniq.tolist()], float) if pi_item else None
    v_i = item_mean(v) if v is not None else None
    out = {"label": LABEL,
           "definition": "partial Spearman (rank residuals on [1, control ranks]) against log1p(popularity); primary "
                         "= all-time popularity, controls = prior-only item mean + " + (", ".join(controls) or "none")
                         + "; pair level (v, user-cluster CI) and item level (pi, item-mean v; controls item-averaged, "
                           "item bootstrap)",
           "controls": ["item_mean_prior"] + list(controls)}
    for pop_name, pop in (("all_time", P["pop"]), ("prior_count", P["pop_prior"])):
        if not np.isfinite(pop).any():
            continue
        lp = np.log1p(pop)
        primary = pop_name == "all_time"
        blk = {}
        for ctl_name, ctl in (("item_mean_prior", prior["mean_prior"]),
                              ("item_mean_prior_shrunk", prior["mean_prior_shrunk"])):
            ci_ok = primary and ctl_name == "item_mean_prior"  # CIs on the primary specification only
            Zp = np.column_stack([ctl] + [np.asarray(c, float) for c in controls.values()])
            Zi = np.column_stack([item_mean(ctl)] + [item_mean(c) for c in controls.values()])
            sub = {}
            targets = []
            if v is not None:
                targets.append(("v_pair", np.asarray(v, float), lp, Zp, users))
                targets.append(("v_item", v_i, item_mean(lp), Zi, None))
            if pi_i is not None:
                targets.append(("pi_item", pi_i, item_mean(lp), Zi, None))
            for tname, x, yv, Z, clusters in targets:
                ok = np.isfinite(x) & np.isfinite(yv) & np.isfinite(Z).all(1)
                xs, ys, Zs = x[ok], yv[ok], Z[ok]
                rec = {"n": int(ok.sum()), "spearman_zero_order": spearman(xs, ys),
                       "partial": partial_spearman(xs, ys, Zs)}
                if ci_ok and ok.sum() >= Z.shape[1] + 3:
                    stat = lambda idx, xs=xs, ys=ys, Zs=Zs: partial_spearman(xs[idx], ys[idx], Zs[idx])  # noqa: E731
                    b = (cluster_bootstrap(stat, clusters[ok], n_boot=n_boot, seed=seed) if clusters is not None
                         else unit_bootstrap(stat, int(ok.sum()), n_boot, seed))
                    rec["partial_ci"] = {"lo": b["lo"], "hi": b["hi"]}
                sub[tname] = rec
            blk[f"control_{ctl_name}"] = sub
        out[f"popularity_{pop_name}"] = blk
    return out


# ---------------------------------------------------------------- A8 sports head share (local)
def _parse_list(s: str) -> list:
    try:
        return json.loads(s.replace("'", '"'))
    except (json.JSONDecodeError, AttributeError):
        return ast.literal_eval(s)


def _head_block(rows: list, skipped: Counter, n_boot: int, seed: int) -> dict:
    out = {"n_events": len(rows), "n_rows_skipped": dict(skipped)}
    if not rows:
        return out
    users = np.array([r[0] for r in rows], dtype=object)
    pool, ccrp, pos = (np.array([r[k] for r in rows], float) for k in (1, 2, 4))
    out.update(n_users=int(len(set(users.tolist()))), n_topk_groups_mismatch=int(sum(r[3] for r in rows)))
    for name, x in (("expected_random_top10_head_share", pool), ("ccrp_top10_head_share", ccrp),
                    ("ccrp_minus_expected", ccrp - pool), ("positive_head_rate", pos)):
        b = cluster_mean_ci(x, users, n_boot, seed)
        out[name] = {"est": b["est"], "lo": b["lo"], "hi": b["hi"]}
    return out


def _try_list(s) -> list | None:
    try:
        v = _parse_list(s)
    except (ValueError, SyntaxError, TypeError):
        return None
    return v if isinstance(v, list) else None


def sports_panel_check(panel_path, first: list) -> dict:
    """File rows 1..first_n of the records vs the Pilot-1 sports panel rows in order: source_event_id, candidate ids
    and candidate popularity groups (first = [(source_event_id, candidate ids or None, groups or None)])."""
    panel = read_jsonl(panel_path)
    bad = Counter()
    for k, (ev, cands, grp) in enumerate(first):
        if k >= len(panel):
            bad["record_rows_beyond_panel"] += 1
            continue
        p = panel[k]
        bad["source_event_id_mismatch"] += str(p.get("source_event_id")) != ev
        bad["candidate_ids_mismatch"] += cands is None or [str(x) for x in p.get("candidate_item_ids", [])] != cands
        pg = p.get("candidate_popularity_groups")
        if pg is not None:
            bad["popularity_groups_mismatch"] += grp is None or [str(x).strip().lower() for x in pg] != grp
    return {"panel": str(panel_path), "panel_sha1": file_sha1(panel_path), "n_panel_rows": len(panel),
            "n_record_rows_compared": len(first), "mismatches": {k: v for k, v in bad.items() if v},
            "identical": len(first) == len(panel) and not any(bad.values())}


def ccrp_head_share(path, first_n: int = 1000, n_boot: int = 2000, seed: int = 0, sports_panel=None) -> dict:
    """A8 from ranking_eval_records.csv (file order = ranking_test.jsonl order; rows 1-1000 = the Pilot-1 sports
    panel). Per event: pool head share (= expected top-10 head share under uniformly random ranking), C-CRP top-10
    head share over pred_ranked_item_ids[:10] (cross-checked against topk_popularity_groups[:10]) and whether the
    positive is head (target demand). CIs resample users. The quarantine split is on the FILE row position: a skipped
    (malformed) row is counted in its own block and never moves a later event across the 1000 boundary."""
    csv.field_size_limit(2 ** 31 - 1)
    blocks, bad, first, n_file = ([], []), (Counter(), Counter()), [], 0
    with _open(path) as f:
        for k, r in enumerate(csv.DictReader(f)):
            n_file += 1
            b = int(k >= first_n)
            cands, grp = _try_list(r.get("candidate_item_ids")), _try_list(r.get("candidate_popularity_groups"))
            cands = [str(x) for x in cands] if cands is not None else None
            grp = [str(g).strip().lower() for g in grp] if grp is not None else None
            if sports_panel is not None and b == 0:
                first.append((str(r.get("source_event_id")), cands, grp))
            if cands is None or grp is None:
                bad[b]["events_unparseable"] += 1
                continue
            if len(cands) != len(grp) or not cands:
                bad[b]["events_group_length_mismatch"] += 1
                continue
            g = dict(zip(cands, grp))
            top = [str(x) for x in (_try_list(r.get("pred_ranked_item_ids")) or [])][:10]
            if any(i not in g for i in top) or len(top) < 10:
                bad[b]["events_top10_not_in_candidates"] += 1
                continue
            share = float(np.mean([g[i] == "head" for i in top]))
            tg = [str(x).strip().lower() for x in (_try_list(r.get("topk_popularity_groups") or "[]") or [])][:10]
            mismatch = len(tg) == 10 and abs(float(np.mean([x == "head" for x in tg])) - share) > 1e-12
            pos = str(r.get("positive_popularity_group", "")).strip().lower() == "head"
            blocks[b].append((str(r["user_id"]), float(np.mean([x == "head" for x in grp])), share, mismatch,
                              float(pos)))
    out = {"label": LABEL, "source": str(path),
           "definition": "per event: expected top-10 head share under uniformly random ranking = head share of the "
                         "candidate pool; C-CRP top-10 head share = head share of pred_ranked_item_ids[:10]; "
                         "positive_head_rate = share of events whose positive is head (target demand); means over "
                         "events with user-cluster bootstrap CIs. Events 1-1000 (file rows) are the burned Pilot-1 "
                         "sports events and are reported separately from 1001-10000 (amendment 2 F)",
           "n_file_rows": n_file, "n_rows_skipped": dict(bad[0] + bad[1]),
           f"events_1_{first_n}": _head_block(blocks[0], bad[0], n_boot, seed),
           f"events_{first_n + 1}_{n_file}": _head_block(blocks[1], bad[1], n_boot, seed),
           "events_all": _head_block(blocks[0] + blocks[1], bad[0] + bad[1], n_boot, seed)}
    if sports_panel is not None:
        out["sports_panel_check"] = sports_panel_check(sports_panel, first)
    return out


# ---------------------------------------------------------------- A9 throughput
def throughput(pilot_dir, name: str) -> dict:
    out = {"label": LABEL, "registered_reconciled_prompts_per_s": REGISTERED_PROMPTS_PER_S.get(name)}
    for tag, d in (("main", name), ("nohist", f"{name}_nohist")):
        rep = read_json(Path(pilot_dir) / d / "report.json")
        if rep:
            out[tag] = {k: rep[k] for k in ("prompts_per_s", "n_prompts", "n_main_prompts", "swap_prompts",
                                            "prompts_sent_this_run", "inference_time_s", "chunks_resumed") if k in rep}
    return out


# ---------------------------------------------------------------- driver
def load_raw_events(raw, source: str) -> dict:
    """{user: [(ts, item, rating)]} via build_rated_panels loaders (ML-1M: RAW/ml-1m or RAW itself)."""
    raw = Path(raw)
    if source == "ml1m":
        res = load_ml1m(raw if (raw / "ratings.dat").exists() else raw / "ml-1m")
    else:
        res = load_amazon(raw, source)
    return res[1]


def panel_identity(panel_path, pilot_dir, name: str) -> dict:
    """sha1 of the panel bytes vs report.json data_sha1 of the main and _nohist runs. match: True / False, or None
    when the run or its data_sha1 is absent; consistent: False on any mismatch, None when nothing was checkable."""
    sha = file_sha1(panel_path)
    out = {"panel": str(panel_path), "panel_sha1": sha}
    for tag, d in (("main", name), ("nohist", f"{name}_nohist")):
        rep = read_json(Path(pilot_dir) / d / "report.json") or {}
        ref = rep.get("data_sha1")
        out[tag] = {"report_data_sha1": ref, "report_data_path": rep.get("data_path"),
                    "match": None if ref is None else ref == sha}
    checks = [out[t]["match"] for t in ("main", "nohist") if out[t]["match"] is not None]
    out["consistent"] = None if not checks else all(checks)
    return out


def analyze_panel(name: str, pilot_dir, panels_dir, raw, a, cache: dict) -> dict:
    run = Path(pilot_dir) / name
    out = {"label": LABEL, "A9_throughput": throughput(pilot_dir, name)}
    panel_path, scores_path = Path(panels_dir) / f"{name}.jsonl", run / "scores.csv.gz"
    if not panel_path.exists() or not scores_path.exists():
        out["skipped"] = f"missing {panel_path if not panel_path.exists() else scores_path}"
        return out
    ident = panel_identity(panel_path, pilot_dir, name)
    if ident["consistent"] is False:
        out["warning"] = (f"panel sha1 {ident['panel_sha1']} differs from a scored run's report.json data_sha1: the "
                          "panel at --panels is not byte-identical to the one scored (see panel_identity, panel_join)")
        print(f"[forensics]   WARNING {name}: {out['warning']}", flush=True)
    P = panel_pairs(read_jsonl(panel_path))
    sc = load_scores(scores_path)
    L, join = attach(P, sc)
    groups = _groups(P["user"])
    boot = dict(n_boot=a.n_boot, seed=a.seed)
    out.update(panel_kind=P["kind"], source=P["source"], panel_identity=ident, n_pairs=P["n"], n_users=len(groups),
               n_duplicate_pairs=P["n_duplicate_pairs"], questions=sorted(L), censoring=sc["censoring"],
               panel_join=join)
    like, dis, para = L.get("like"), L.get("dislike"), L.get("like_para")
    swap = load_swap_positions(run / "swap_prior.csv.gz") if (run / "swap_prior.csv.gz").exists() else None
    nh_path = Path(pilot_dir) / f"{name}_nohist" / "scores.csv.gz"
    nh, nh_diag = None, None
    if nh_path.exists():
        nh_sc = load_scores(nh_path)
        nh_L, nh_join = attach(P, nh_sc)
        nh = nh_L.get("like")
        nh_diag = {"nohist_join": nh_join, "nohist_censoring": nh_sc["censoring"],
                   "nohist_like_finite_pairs": int(np.isfinite(nh).sum()) if nh is not None else 0}
    pi_p, pi_item = np.full(P["n"], NAN), None
    if P["kind"] == "rated" and (swap is not None or nh is not None):
        out["A2_prior"], pi_p, pi_item = prior_checks(P, groups, like, swap["pos"] if swap else None, nh, **boot)
        if swap:
            out["A2_prior"]["swap_censoring"] = swap["censoring"]
    else:
        out["A2_prior"] = {"label": LABEL, "skipped": "next-item panel, or no swap_prior.csv.gz / _nohist scores"}
    if nh_diag is not None:
        out["A2_prior"].update(nh_diag)
    if like is not None and dis is not None and para is not None:
        out["A4_ensemble_null"] = ensemble_null(P, groups, like, dis, para, null_sim_reps=a.null_sim_reps, **boot)
    else:
        out["A4_ensemble_null"] = {"label": LABEL, "skipped": "like, dislike and like_para are not all scored"}
    if P["kind"] != "rated":
        for k in ("A3_cf_references", "A5_calibration", "A6_error_targets", "A7_popularity_partial"):
            out[k] = {"label": LABEL, "skipped": "rated panels only"}
        return out
    if like is None:
        for k in ("A5_calibration", "A6_error_targets"):
            out[k] = {"label": LABEL, "skipped": "like not scored"}
    else:
        cal = {"like": like}
        det = {"like": like}
        if dis is not None:
            cal["mirror_half"] = (like - dis) / 2
            det["mirror"] = like - dis
        if para is not None:
            cal["placebo"] = det["placebo"] = (like + para) / 2
        if np.isfinite(pi_p).any():
            det["evidence"] = like - pi_p
        out["A5_calibration"] = calibration_decomposition(P, groups, cal, a.seed)
        out["A6_error_targets"] = error_targets(P, groups, det, "like", **boot)
    prior = None
    if not raw:
        out["A3_cf_references"] = {"label": LABEL, "skipped": "no --raw"}
    else:
        try:
            if P["source"] not in cache:
                cache.clear()  # one raw source in memory at a time
                print(f"[forensics]   loading raw events for {P['source']!r}", flush=True)
                cache[P["source"]] = load_raw_events(raw, P["source"])
            events = cache[P["source"]]
        except (OSError, KeyError) as e:
            events = None
            out["A3_cf_references"] = {"label": LABEL, "skipped": f"raw data for {P['source']!r} unavailable: {e}"}
        if events is not None:
            out["A3_cf_references"], prior = cf_references(
                P, groups, events, shrink_k=a.shrink_k, mf_dim=a.mf_dim, mf_iters=a.mf_iters, mf_lambda=a.mf_lambda,
                mf_lambda_bias=a.mf_lambda_bias, mf_cutoff_q=a.mf_cutoff_quantile, **boot)
            hs = out["A3_cf_references"]["history_star_mean"]
            if ident["consistent"] is False:
                # the history ratings come from the panel at --panels, not provably the scored one
                hs["usable"] = False
                hs["unusable_reason"] = ("panel sha1 differs from the scored run's report.json data_sha1 (see "
                                         "panel_identity): history_ratings may not be the scored panel's")
                hs["AUC_global_unusable"] = hs.pop("AUC_global")
            else:
                hs["usable"] = True if ident["consistent"] else None
    if prior is None:
        out["A7_popularity_partial"] = {"label": LABEL, "skipped": "needs the A3 prior-only item mean (raw data)"}
        return out
    controls = {}
    if P["source"] == "ml1m":
        controls["release_year"] = np.array([release_year(t) for t in P["title"]], float)
        extra = {"release_year_source": "parsed from candidate_titles '(YYYY)'"}
    else:
        need = set(P["item"].tolist())
        try:
            dl = amazon_desc_lengths(Path(raw), P["source"], need)
            src = "raw meta description"
        except (OSError, KeyError):
            dl, src = {}, "candidate_texts fallback (raw meta unavailable)"
        controls["desc_len"] = np.array([dl[i] if i in dl else desc_length_from_text(t)
                                         for i, t in zip(P["item"], P["text"])], float)
        controls["has_store"] = np.array([float(bool(b.strip())) for b in P["brand"]], float)
        extra = {"desc_len_source": src, "n_items_desc_from_raw": len(dl)}
    v = (like - dis) / 2 if like is not None and dis is not None else None
    out["A7_popularity_partial"] = {**popularity_partial(P, groups, pi_item, v, prior, controls, **boot), **extra}
    return out


def wording_flags(res: dict) -> dict:
    def est(d, *path):
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
        return d if isinstance(d, (int, float)) and math.isfinite(d) else None

    def cmp(x, op, c):
        return None if x is None else (x < c if op == "<" else x <= c)

    out = {"label": "exploratory: amendment 2 A decision use changes wording only, never a gate",
           "rules": WORDING_RULES, "sensitivity_rules": SENSITIVITY_RULES}
    a3, a4 = "A3_cf_references", "A4_ensemble_null"
    for name, blk in res.get("panels", {}).items():
        r = est(blk, "A2_prior", "split_half", "spearman_r", "est")
        sb = est(blk, "A2_prior", "split_half", "spearman_brown_8")
        mf = est(blk, a3, "mf_temporal", "test_pairs_t_ge_T", "UAUC_personal_residual", "est")
        mf_all = est(blk, a3, "mf_temporal", "all_pairs_sensitivity", "UAUC_personal_residual", "est")
        mf_nf = est(blk, a3, "mf_temporal", "test_pairs_t_ge_T_no_fold_in", "UAUC_personal_residual", "est")
        mf_tr = est(blk, a3, "mf_transductive_reference", "UAUC_personal_residual", "est")
        lo, hi = (est(blk, a4, "dUAUC_mirror_minus_ensemble_null", k) for k in ("lo", "hi"))
        bc_lo = est(blk, a4, "dUAUC_mirror_minus_ensemble_null_bias_corrected", "lo")
        gap_lo = est(blk, a4, "gap_residual", "lo")
        out[name] = {"pi_split_half_reliability_lt_0p7": cmp(sb, "<", 0.7),
                     "mf_personal_residual_uauc_le_0p55": cmp(mf, "<=", 0.55),
                     "ensemble_null_residual_lo_le_0": cmp(lo, "<=", 0.0),
                     "sensitivity": {
                         "pi_split_half_raw_r_lt_0p7": cmp(r, "<", 0.7),
                         "mf_temporal_all_pairs_residual_uauc_le_0p55": cmp(mf_all, "<=", 0.55),
                         "mf_temporal_no_fold_in_residual_uauc_le_0p55": cmp(mf_nf, "<=", 0.55),
                         "mf_transductive_reference_residual_uauc_le_0p55": cmp(mf_tr, "<=", 0.55),
                         "ensemble_null_residual_ci_covers_0_two_sided": None if lo is None or hi is None
                         else lo <= 0 <= hi,
                         "ensemble_null_residual_bias_corrected_lo_le_0": cmp(bc_lo, "<=", 0.0),
                         "ensemble_null_gap_residual_lo_le_0": cmp(gap_lo, "<=", 0.0)}}
    return out


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot_dir", default=None, help="Pilot-1 output root (outputs/confrec/pilot1_mirror)")
    ap.add_argument("--panels", default="outputs/confrec/panels")
    ap.add_argument("--raw", default=None, help="raw data root (data/raw); A3/A7 need it")
    ap.add_argument("--names", default=",".join(NAMES))
    ap.add_argument("--ccrp_head_share", default=None, help="local ranking_eval_records.csv of C-CRP v3 (A8)")
    ap.add_argument("--sports_first_n", type=int, default=1000)
    ap.add_argument("--sports_panel", default=None,
                    help="Pilot-1 sports_next_1k.jsonl: A8 checks that file rows 1..first_n are its events in order "
                         "(default PANELS/sports_next_1k.jsonl when it exists)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shrink_k", type=float, default=5.0)
    ap.add_argument("--mf_dim", type=int, default=32)
    ap.add_argument("--mf_iters", type=int, default=15, help="0 skips both MF fits")
    ap.add_argument("--mf_lambda", type=float, default=0.05)
    ap.add_argument("--mf_lambda_bias", type=float, default=5.0)
    ap.add_argument("--mf_cutoff_quantile", type=float, default=MF_CUTOFF_QUANTILE,
                    help="temporal MF cutoff T = this quantile of the panel candidate timestamps (amendment 1 D)")
    ap.add_argument("--null_sim_reps", type=int, default=50,
                    help="draws per user of the A4 parametric null-bias bootstrap (0 skips it)")
    a = ap.parse_args(argv)
    if not a.pilot_dir and not a.ccrp_head_share:
        ap.error("give --pilot_dir (A2-A7, A9) and/or --ccrp_head_share (A8)")
    if not 0.0 < a.mf_cutoff_quantile < 1.0:
        ap.error("--mf_cutoff_quantile must be in (0, 1)")
    if a.sports_panel is None and (Path(a.panels) / "sports_next_1k.jsonl").exists():
        a.sports_panel = str(Path(a.panels) / "sports_next_1k.jsonl")
    return a


def main(argv=None) -> dict:
    a = parse_args(argv)
    res = {"label": LABEL,
           "status": "EXPLORATORY ONLY (amendment 2 section A): hypothesis-generating, never cited as evidence, never "
                     "a gate or an input to G1-G9",
           "spec": "idea-stage/PREREG_AMENDMENT_2.md section A; A1 is produced by the panel builders",
           "inputs": {k: v for k, v in vars(a).items()}}
    if a.pilot_dir:
        cache: dict = {}
        res["panels"] = {}
        for name in [x for x in a.names.split(",") if x]:
            print(f"[forensics] {name}", flush=True)
            res["panels"][name] = analyze_panel(name, a.pilot_dir, a.panels, a.raw, a, cache)
        res["wording_flags"] = wording_flags(res)
    if a.ccrp_head_share:
        print("[forensics] A8 C-CRP head share", flush=True)
        res["A8_sports_head_share"] = ccrp_head_share(a.ccrp_head_share, a.sports_first_n, a.n_boot, a.seed,
                                                      a.sports_panel)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(strict_json(res), indent=2, allow_nan=False)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    print(f"wrote {out}")
    return res


if __name__ == "__main__":
    main()
