"""Amendment-3 report: every endpoint of idea-stage/PREREG_AMENDMENT_3.md section 3, P1, the pseudonym-knockout labels
(section 5) and the FT-C contrast (section 9) for one domain and one backbone (interfaces: docs/sigir/FTGRID_IMPL_SPEC.md).

    python -m src.confrec.ftgrid_report --domain ml1m --split outputs/confrec/ftgrid/panels/ml1m/ftgrid_split.json \
        --panels outputs/confrec/ftgrid/panels/ml1m --scores_root outputs/confrec/ftgrid/scores \
        --models zeroshot,s0,s1,s2,p0,p1 --raw data/raw --out outputs/confrec/ftgrid/report/ml1m.json \
        [--n_boot 2000] [--seed 0] [--refs outputs/confrec/ftgrid/report/ml1m_refs.json] [--bi_boot 1000]

Inputs (the layout of the spec, nothing else): --split = panels/<d>/ftgrid_split.json (T, S_d, sha1s, build args);
--panels = panels/<d>/ (eval.jsonl, eval_sd_test.jsonl, train.jsonl, sd_users.txt, eval_sd_test_starperm{0,1}.jsonl,
eval_pseudo.jsonl, eval_placebo.jsonl); --scores_root = scores/ (or scores/<d>/) holding <model>/<arm>/{scores.csv.gz,
report.json[, swap_prior.csv.gz, run.key]} for model in --models and arm in like (eval.jsonl), swap (eval_sd_test.jsonl,
--swap_k 8), nohist (eval_sd_test.jsonl, --hist_len 0), starperm0/1, pseudo, placebo; --raw = data/raw (the CF references:
forensics.load_raw_events / cf_references); --refs = an optional cache of the per-pair CF references (q-hat, temporal MF),
read when its identity (domain, eval.jsonl sha1, T, estimator settings) matches and (re)written from --raw otherwise, so a
second backbone's report reuses them without the raw data. A run or arm that is absent gives a null block with its reason;
a run failing integrity is excluded and listed (never replaced by another seed).
Outputs: --out (stats.strict_json, allow_nan=False; no wall-clock, host or path) and <out stem>_tables.csv (the headline
numbers per regime, model and endpoint for the paper tables). The choices the amendment leaves open are fixed in
OPERATIONALIZATIONS below and copied into the JSON.

Endpoints (A3 section 3; TEST rows unless stated; regimes ZS = zeroshot, FT = s0-s2, PERM = p0-p1 (FT-C)):
  E-A  UAUC per model and seed and the seed mean, with the references q-hat (shrunk, k = 5), all-time popularity, the
       temporal biased MF (forensics.cf_references with the explicit cutoff T of the split, evaluated on TEST rows) and its
       personal residual p_u.q_i (= MF - mu - b_u - b_i; mu is a constant, so every per-user AUC is that of
       MF - b_u - b_i); secondary: all-rows UAUC (CAL u TEST) and the seen / unseen split (item in train.jsonl).
  E-B  dUAUC_s = UAUC(FT seed s) - UAUC(ZS), each seed and the mean over seeds with its CI and p, sigma_seed.
  E-C  global Platt map fit on ALL CAL rows of the EVAL users, applied to TEST: ECE (metrics.ece, 10 adaptive bins),
       Brier, Brier skill over the CAL base rate, Platt slope, the 10-bin reliability table; the per-user top-k error
       anatomy (c_u = midpoint of the k-th and (k+1)-th largest TEST logits, margin tertiles over all TEST pairs).
  E-D  on S_d's TEST rows: UAUC of pi, L_nohist, e-hat = L - pi and L; r_c, r8c, rho, item-prior and non-prior shares
       (re-estimated in every user resample, unclipped, 'uninterpretable' when r8c < 0.7); cross-fitted stackers M0-M3,
       G = dUAUC(M2 - M1), G_CF = dUAUC(M3 - M0), G/G_CF only when G_CF's CI excludes 0; star permutation.
  E-E  partial Spearman (forensics.partial_spearman, the popularity_partial estimator) of L and pi with log1p(all-time
       popularity) controlling for q-hat (+ the domain controls), pair and item level; Bias Index head vs tail
       (metrics.bias_index, conf = the CAL-fit Platt probability); tail-only UAUC (exploratory).
  P1   ML-1M Qwen3-8B: G_FT,s - G_ZS on identical users and rows; holds iff mean > 0 with p < 0.05, all 3 seeds > 0 and
       mean > 2 sigma_seed (elsewhere the same quantity is the replication of its sign).
  FT-K pilot_pseudonym.analyze on the rows finite in every arm of every model; the label is decide()'s verdict with
       AMBIGUOUS -> INDETERMINATE and UNDETERMINED -> INCOMPLETE (no --gate, no p-value, no Holm family).
  FT-C UAUC(s_s) - UAUC(p_s) on TEST and on all rows, s = 0, 1, with the E-A references; descriptive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.confrec import forensics as fx
from src.confrec import pilot_pseudonym as pps
from src.confrec.metrics import auroc, bias_index, brier, ece, reliability_bins
from src.confrec.stats import percentile_ci, platt_fit, rank_bins, sigmoid, strict_json, user_halves

NAN = float("nan")
SPEC = "idea-stage/PREREG_AMENDMENT_3.md sections 1, 3, 5, 9, 11; docs/sigir/FTGRID_IMPL_SPEC.md"
MIN_N = 150                          # A3 section 1: an endpoint on fewer users is descriptive
N_BOOT, SEED = 2000, 0               # A3 section 3: 2,000 user resamples, seed 0
FOLD_SEED = 0                        # A3 section 3: the stacker folds are stats.user_halves(S_d, 0)
SHRINK_K = 5.0                       # q-hat shrinkage (forensics.prior_means)
MF_KW = {"mf_dim": 32, "mf_iters": 15, "mf_lambda": 0.05, "mf_lambda_bias": 5.0}   # forensics.cf_references defaults
CENS2_MAX, MASS_MIN = 0.005, 0.95    # A3 section 2 E1
ECE_BINS = 10
POP_BINS, HEAD_BIN, TAIL_BIN = 5, 4, 0
BI_BINS = 10
R8C_MIN = 0.7                        # Amendment 2 A2 wording flag
STARPERM_FLAGS = {"abs_dUAUC_lt": 0.005, "within_user_sd_tau_lt": 0.10}   # Amendment 2 D thresholds (wording only)
KNOCKOUT_BI_BOOT = 1000              # pilot_pseudonym's CLI default for its Bias Index CIs
LABEL_MAP = {"AMBIGUOUS": "INDETERMINATE", "UNDETERMINED": "INCOMPLETE"}   # A3 section 5 closed-hole rule
STACKERS = {"M0": ("q_hat",), "M1": ("q_hat", "pi"), "M2": ("q_hat", "pi", "e_hat"), "M3": ("q_hat", "mf_residual")}
REGIMES = {"ZS": ("zeroshot",), "FT": ("s0", "s1", "s2"), "PERM": ("p0", "p1")}
MODELS = ("zeroshot", "s0", "s1", "s2", "p0", "p1")
ARM_PANEL = {"like": "eval.jsonl", "swap": "eval_sd_test.jsonl", "nohist": "eval_sd_test.jsonl",
             "starperm0": "eval_sd_test_starperm0.jsonl", "starperm1": "eval_sd_test_starperm1.jsonl",
             "pseudo": "eval_pseudo.jsonl", "placebo": "eval_placebo.jsonl"}
STARPERM_ARMS = ("starperm0", "starperm1")
DOMAIN_CONTROLS = {"ml1m": ("release_year",), "toys": ("desc_len", "has_store"), "sports": ("desc_len", "has_store")}
REFS_FORMAT = "ftgrid_refs_v1"
CSV_COLS = ("domain", "backbone", "block", "regime", "model", "endpoint", "est", "lo", "hi", "p", "n_users", "n_pairs",
            "descriptive_min_n", "note")

OPERATIONALIZATIONS = (
    "pairs: one per (user_id, item_id) of eval.jsonl (one row per user, distinct items in a row); every scoring arm is "
    "joined on that key, so the TEST-only panels (eval_sd_test*.jsonl, whose cand_idx are renumbered) align with "
    "eval.jsonl; a score row whose label differs from the panel's is excluded and counted",
    "TEST = candidate_timestamps >= T, CAL = < T, with T read from ftgrid_split.json (never recomputed)",
    "integrity (A3 section 2, E1) from each report.json, the scores' censoring counts standing in for an absent field: "
    "censored = 2 share of the main rows <= 0.005, overlength = max(n_overlength, censored = 3 rows) = 0, mean Yes+No mass "
    ">= 0.95; a swap run's donor prompts also need a censored = 2 share <= 0.005 and no overlength prompt (their mass is "
    "not recorded); unknown values fail. Excluded and listed, never replaced: FAILED_INTEGRITY (E1), INCOMPLETE (no "
    "report.json or scores), UNREADABLE, PANEL_MISMATCH (report.json data_sha1 differs from the sha1 of the arm's panel "
    "file), VARIANT_MISMATCH (variant differs from the split's), MODEL_MISMATCH (zeroshot scored with an adapter, or an "
    "adapter model without one); an ABSENT arm gives null blocks with the reason",
    "identical users and rows: each regime block (and each contrast) is computed on the rows finite for every model it "
    "involves, so each listed seed and the mean over seeds use the same users and rows; the mean over seeds of the "
    "per-seed UAUCs is then exactly the seed-averaged UAUC (mean over users of the per-user AUC averaged over seeds)",
    "UAUC: mean over users with both classes among the rows of the per-user AUC with average ranks (ties 1/2)",
    "CIs: 95% percentile, 2,000 user resamples, seed 0; per-user statistics use stats.paired_bootstrap's draws (sorted "
    "user ids, one rng.integers(0, n, n) per resample), statistics re-estimated on resampled rows use "
    "stats.cluster_bootstrap's draws (users), item-level correlations resample items (forensics.unit_bootstrap); a "
    "regime's seeds share the draws, so the CI of the mean over seeds is that of the per-draw mean",
    "p = min(1, 2 min((#{d* <= 0} + 1)/(B + 1), (#{d* >= 0} + 1)/(B + 1))) over the B finite resamples (B = 2,000)",
    "minimum n (A3 section 1): an endpoint on fewer than 150 users has descriptive_min_n = true and no p-value or "
    "CI-based claim (p and ci_excludes_0 null; the interval is still printed as a description)",
    "sigma_seed = SD (ddof 1) of the per-seed values; sigma_seed_rule = every registered seed present, every seed of the "
    "mean's sign and |mean| > 2 sigma_seed (A3 section 11, Amendment 2 F)",
    "E-A references on the regime's rows (rows finite in every reference too): q-hat = forensics.prior_means "
    "mean_prior_shrunk (k = 5, other users' first ratings strictly before the candidate's timestamp); popularity = "
    "candidate_popularity; the temporal biased MF = forensics.cf_references(EVAL panel, cutoff = T): item parameters on "
    "the first ratings with ts < T except every EVAL candidate event, each EVAL user folded in from the user's own "
    "non-candidate events strictly before the user's first candidate; cold pairs keep residual 0 on the identical rows, "
    "the warm-pair residual UAUC is the forensics convention (secondary)",
    "seen = the candidate's item occurs among the candidates of train.jsonl (after the cap)",
    "E-C: Platt = stats.platt_fit on every CAL row of the regime's rows; Brier skill = 1 - Brier / Brier(CAL base rate) on "
    "TEST; reliability: the 10 equal-mass bins of the adaptive ECE (sum of share x |gap| = ECE) and metrics."
    "reliability_bins with 10 equal-width bins; anatomy on TEST rows of users with both classes: k = TEST likes, c_u = "
    "midpoint of the k-th and (k+1)-th largest logits, decision = 1[L >= c_u] (a tie at c_u is decided 1, no random "
    "break), tertiles = stats.rank_bins(margin, 3) over the anatomy pairs (0 = smallest margins = bottom); AUROC = "
    "metrics.auroc(margin, correct); AURC = metrics.risk_coverage(margin, correct)[2]; the user resamples refit the "
    "Platt map and recut the tertiles",
    "E-D: L = the like logit of the model's like arm (eval.jsonl) on S_d's TEST rows (the swap arm's own like scores are "
    "a consistency check); pi = mean of the finite donor logits of swap_prior.csv.gz, halves = donor rows 1-4 vs 5-8 in "
    "file order (forensics.load_swap_positions); L_nohist = the item mean of the nohist arm's like logit; shares on the "
    "rows where L, pi and both halves are finite, every variable centred within user on those rows",
    "E-D stackers: logistic regressions with intercept (Newton, features standardised on the fitting fold, L2 1e-4 on the "
    "slopes as stats.platt_fit), fold A = stats.user_halves(S_d users, 0), each fold's rows scored by the fit on the other "
    "fold's rows; UAUC of the pooled cross-fitted linear predictors on the rows where every feature is finite",
    "E-D star permutation (diag_battery convention): dUAUC = UAUC(L) - mean over users of the per-user mean over k of "
    "AUC(L_perm_k) (the UAUC of the mean permuted logit is reported beside it); tau_P = L - mean_k L_perm_k on the rows "
    "finite in L and every copy; within-user SD pooled with divisor N - n_users",
    "popularity bins: stats.rank_bins(candidate_popularity, 5) cut over ALL EVAL candidates (CAL u TEST, A3 section 1); "
    "head = bin 4, tail = bin 0",
    "E-E: pair level on the TEST rows (user-cluster CI), item level on item means over those rows (item bootstrap); pi at "
    "item level over S_d's TEST items; controls = q-hat (shrunk) + release_year (ml1m) or desc_len and has_store (toys, "
    "sports; desc_len from the raw meta description, else from candidate_texts as forensics); games: q-hat only (the "
    "domain is not named in A3), with desc_len and has_store as a point-estimate sensitivity; Bias Index = metrics."
    "bias_index(conf, label, head/mid/tail, 10 bins, residual-adjusted) on TEST rows, conf = the E-C Platt map",
    "P1: on S_d's TEST rows finite for ZS and the 3 FT seeds, stackers M1 and M2 cross-fitted per model; per user "
    "G_m = AUC(M2_m) - AUC(M1_m); the contrast of seed s is mean over users of G_s - G_ZS",
    "knockout: models zeroshot, s0, s1, s2 with OK like (real), pseudo and placebo arms; rows = the (source_event_id, "
    "cand_idx) keys finite in every arm of every such model; pilot_pseudonym.analyze unchanged (n_boot, seed, Bias Index "
    "bootstrap 1000 = its CLI default); treated flags from eval_pseudo.jsonl (pilot_pseudonym.treated_from_panel)",
)


# ---------------------------------------------------------------- small helpers
def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _fin(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and math.isfinite(float(x))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha1_text_lines(ids) -> str:
    """sha1 of a user-id list: ids joined by '\\n', no trailing newline (ftgrid_data.ids_bytes)."""
    return hashlib.sha1("\n".join(str(u) for u in ids).encode("utf-8")).hexdigest()


def _fmean(v) -> float:
    v = [x for x in v if math.isfinite(x)]
    return float(np.mean(v)) if v else NAN


# ---------------------------------------------------------------- statistics
def user_aucs(score, label, users, mask=None) -> dict:
    """{user: AUC} over each user's rows in `mask` with a finite score, for users with both classes there; ranks are
    averaged within a user's ties (ties count 1/2), i.e. metrics.auroc per user (= forensics.uauc), vectorised."""
    s, y, u = np.asarray(score, float), np.asarray(label, int), np.asarray(users)
    ok = np.isfinite(s) if mask is None else (np.asarray(mask, bool) & np.isfinite(s))
    idx = np.flatnonzero(ok)
    if not len(idx):
        return {}
    uu, inv = np.unique(u[idx], return_inverse=True)
    inv = inv.reshape(-1)
    ss, yy = s[idx], y[idx]
    order = np.lexsort((ss, inv))
    inv, ss, yy = inv[order], ss[order], yy[order]
    n = len(inv)
    new_u = np.r_[True, inv[1:] != inv[:-1]]
    new_t = new_u | np.r_[True, ss[1:] != ss[:-1]]
    first = np.maximum.accumulate(np.where(new_u, np.arange(n), 0))
    pos = np.arange(n) - first + 1.0
    tid = np.cumsum(new_t) - 1
    rank = (np.bincount(tid, pos) / np.bincount(tid))[tid]
    k = len(uu)
    npos = np.bincount(inv, yy, minlength=k)
    nneg = np.bincount(inv, minlength=k) - npos
    rsum = np.bincount(inv, rank * yy, minlength=k)
    good = np.flatnonzero((npos > 0) & (nneg > 0))
    auc = (rsum[good] - npos[good] * (npos[good] + 1) / 2) / (npos[good] * nneg[good])
    keys = uu[good].tolist()
    return dict(zip(keys, auc.tolist()))


def boot_p(draws) -> float:
    """A3 section 3: p = 2 min(P*(d <= 0), P*(d >= 0)) over the resamples with +1/(B + 1) smoothing, capped at 1."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return NAN
    b = len(d)
    return float(min(1.0, 2.0 * min((np.sum(d <= 0) + 1) / (b + 1), (np.sum(d >= 0) + 1) / (b + 1))))


def mean_draws(V, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, k) column means of V (n_units x k, units in sorted key order) over unit resamples: one
    rng.integers(0, n, n) per resample, i.e. stats.paired_bootstrap's draws for every column at once."""
    V = np.asarray(V, float).reshape(len(V), -1)
    out = np.full((n_boot, V.shape[1]), NAN)
    n = len(V)
    if n:
        rng = np.random.default_rng(seed)
        for b in range(n_boot):
            out[b] = V[rng.integers(0, n, n)].mean(0)
    return out


class Clusters:
    """Row resampling by whole users with stats.cluster_bootstrap's draws (clusters in np.unique order, one
    rng.integers(0, n_clusters, n_clusters) per resample, member rows in ascending order), vectorised."""

    def __init__(self, clusters):
        c = np.asarray(clusters)
        self.uniq, inv = np.unique(c, return_inverse=True)
        inv = inv.reshape(-1)
        self.order = np.argsort(inv, kind="stable")
        self.lens = np.bincount(inv, minlength=len(self.uniq))
        self.starts = np.cumsum(self.lens) - self.lens
        self.n = len(self.uniq)

    def rows(self, pick) -> np.ndarray:
        ln = self.lens[pick]
        off = np.repeat(np.cumsum(ln) - ln, ln)
        return self.order[np.repeat(self.starts[pick], ln) + np.arange(int(ln.sum())) - off]

    def draws(self, n_boot: int, seed: int):
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            yield self.rows(rng.integers(0, self.n, self.n))


def unit_draws(n_units: int, n_boot: int, seed: int):
    """forensics.unit_bootstrap's draws (items)."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.integers(0, n_units, n_units)


def rec(est, draws=None, n_users: int = 0, n_pairs=None, *, contrast: bool = False, **extra) -> dict:
    """One endpoint: estimate, 95% percentile CI, n and the minimum-n flag; a contrast adds its p-value and
    ci_excludes_0, both null when the endpoint is descriptive (fewer than MIN_N users)."""
    d = np.asarray([] if draws is None else draws, float)
    lo, hi = percentile_ci(d) if len(d) else (NAN, NAN)
    desc = int(n_users) < MIN_N
    out = {"est": _num(est), "lo": lo, "hi": hi, "n_users": int(n_users)}
    if n_pairs is not None:
        out["n_pairs"] = int(n_pairs)
    out["n_boot"] = int(len(d))
    out["descriptive_min_n"] = desc
    if contrast:
        out["p"] = None if desc else boot_p(d)
        out["ci_excludes_0"] = (None if desc or not (math.isfinite(lo) and math.isfinite(hi))
                                else bool(lo > 0 or hi < 0))
    out.update(extra)
    return out


def seed_summary(values, n_registered: int, contrast: bool = False, mean=None) -> dict:
    """sigma_seed (SD ddof 1 of the per-seed values) and, for a contrast, the sigma_seed rule."""
    v = np.asarray([_num(x) for x in values], float)
    sd = float(np.std(v, ddof=1)) if len(v) > 1 and np.isfinite(v).all() else NAN
    out = {"n_seeds": int(len(v)), "n_seeds_registered": int(n_registered), "complete": len(v) == n_registered,
           "per_seed": v.tolist(), "sigma_seed": sd}
    if contrast:
        m = _num(mean)
        if math.isfinite(m) and len(v) and np.isfinite(v).all():
            same = bool(m != 0 and all(np.sign(x) == np.sign(m) for x in v))
            big = bool(math.isfinite(sd) and abs(m) > 2 * sd)
            out.update(all_seeds_same_sign_as_mean=same, abs_mean_gt_2_sigma_seed=big,
                       sigma_seed_rule=bool(out["complete"] and same and big))
        else:
            out.update(all_seeds_same_sign_as_mean=None, abs_mean_gt_2_sigma_seed=None, sigma_seed_rule=None)
    return out


def _rows_of_users(users, rows, keys) -> int:
    rows = np.asarray(rows, bool)
    return int(np.isin(np.asarray(users)[rows], np.asarray(keys)).sum()) if len(keys) else 0


def uauc_models(scores: dict, label, users, rows, n_boot: int, seed: int, extra: dict | None = None,
                n_registered: int | None = None) -> dict:
    """UAUC of each score (models of one regime) on identical rows and users: `rows` restricted to rows finite in every
    score (and every `extra` reference); the mean over models = the seed-averaged UAUC; `extra` references get their
    UAUC on the same rows and users."""
    extra = extra or {}
    names, xnames = list(scores), list(extra)
    allv = {**scores, **extra}
    R = np.asarray(rows, bool).copy()
    for v in allv.values():
        R &= np.isfinite(np.asarray(v, float))
    pu = {k: user_aucs(v, label, users, R) for k, v in allv.items()}
    keys = sorted(set.intersection(*(set(d) for d in pu.values()))) if pu else []
    cols = names + xnames
    V = np.array([[pu[k][u] for k in cols] for u in keys], float).reshape(len(keys), len(cols))
    D = mean_draws(V, n_boot, seed)
    nu, npairs = len(keys), _rows_of_users(users, R, keys)
    out = {"rows": {"n_users": nu, "n_pairs": npairs}}
    if names:
        out["per_model"] = {k: rec(V[:, j].mean() if nu else NAN, D[:, j], nu, npairs) for j, k in enumerate(names)}
        k = len(names)
        out["mean_over_seeds"] = rec(V[:, :k].mean(1).mean() if nu else NAN, D[:, :k].mean(1), nu, npairs)
        out["seeds"] = seed_summary([out["per_model"][m]["est"] for m in names], n_registered or len(names))
    if xnames:
        out["references"] = {x: rec(V[:, len(names) + j].mean() if nu else NAN, D[:, len(names) + j], nu, npairs)
                             for j, x in enumerate(xnames)}
    return out


def contrast_models(pairs: dict, label, users, rows, n_boot: int, seed: int, n_registered: int | None = None) -> dict:
    """{name: (score_a, score_b)}: per name dUAUC = UAUC(a) - UAUC(b) on identical users and rows (finite in every score
    of every pair), the mean over names with its CI and p, and sigma_seed."""
    names = list(pairs)
    R = np.asarray(rows, bool).copy()
    for a, b in pairs.values():
        R &= np.isfinite(np.asarray(a, float)) & np.isfinite(np.asarray(b, float))
    pa = {k: user_aucs(pairs[k][0], label, users, R) for k in names}
    pb = {k: user_aucs(pairs[k][1], label, users, R) for k in names}
    keys = sorted(set.intersection(*(set(pa[k]) & set(pb[k]) for k in names))) if names else []
    A = np.array([[pa[k][u] for k in names] for u in keys], float).reshape(len(keys), len(names))
    B = np.array([[pb[k][u] for k in names] for u in keys], float).reshape(len(keys), len(names))
    V = A - B
    D = mean_draws(V, n_boot, seed)
    nu, npairs = len(keys), _rows_of_users(users, R, keys)
    per = {k: rec(V[:, j].mean() if nu else NAN, D[:, j], nu, npairs, contrast=True,
                  UAUC_a=A[:, j].mean() if nu else NAN, UAUC_b=B[:, j].mean() if nu else NAN)
           for j, k in enumerate(names)}
    mean = rec(V.mean(1).mean() if nu else NAN, D.mean(1), nu, npairs, contrast=True)
    return {"rows": {"n_users": nu, "n_pairs": npairs}, "per_seed": per, "mean_over_seeds": mean,
            "seeds": seed_summary([per[k]["est"] for k in names], n_registered or len(names), True, mean["est"])}


def pooled_within_sd(values, users) -> float:
    """Pooled within-user SD with divisor N - n_users (diag_battery.pooled_within_sd, Amendment 1 P1.5b)."""
    v, u = np.asarray(values, float), np.asarray(users)
    if not len(v):
        return NAN
    _, inv = np.unique(u, return_inverse=True)
    inv = inv.reshape(-1)
    m = np.bincount(inv, v) / np.bincount(inv)
    df = len(v) - (inv.max() + 1)
    return math.sqrt(float(((v - m[inv]) ** 2).sum()) / df) if df > 0 else NAN


def aurc(conf, utility) -> float:
    """metrics.risk_coverage(conf, utility)[2] (tie-invariant AURC), vectorised for the bootstrap."""
    conf, utility = np.asarray(conf, float), np.asarray(utility, float)
    if not len(conf):
        return NAN
    order = np.argsort(-conf, kind="stable")
    c, u = conf[order], utility[order]
    g = np.concatenate([[0], np.cumsum(c[1:] != c[:-1])])
    u = (np.bincount(g, u) / np.bincount(g))[g]
    return float(np.mean(1 - np.cumsum(u) / np.arange(1, len(u) + 1)))


# ---------------------------------------------------------------- inputs: runs
def integrity(rep: dict, cens: dict, swap_cens: dict | None = None) -> dict:
    """A3 section 2 E1 of one scoring run from report.json (the scores' censoring counts when a field is absent)."""
    cm = rep.get("censored_main") if isinstance(rep.get("censored_main"), dict) else {}
    tot = Counter()
    for c in cens.values():
        tot.update(c)
    n = _num(rep.get("n_main_prompts"))
    if not math.isfinite(n):
        n = float(tot.get("n", 0))
    n2 = _num(cm["2"]) if "2" in cm else float(tot.get("2", 0))
    n3 = max(_num(cm["3"]) if "3" in cm else 0.0, float(tot.get("3", 0)))
    over_rep = _num(rep.get("n_overlength"))
    over = max(n3, over_rep) if math.isfinite(over_rep) else n3
    mass = _num(rep.get("mean_yes_no_mass"))
    share = n2 / n if n > 0 else NAN
    out = {"n_rows": n, "n_censored2": n2, "censored2_share": share, "overlength": over, "mean_yes_no_mass": mass,
           "censored2_share_le_0.005": bool(share <= CENS2_MAX), "overlength_eq_0": bool(over == 0),
           "mean_yes_no_mass_ge_0.95": bool(mass >= MASS_MIN)}
    ok = out["censored2_share_le_0.005"] and out["overlength_eq_0"] and out["mean_yes_no_mass_ge_0.95"]
    if swap_cens is not None:
        cs = rep.get("censored_swap") if isinstance(rep.get("censored_swap"), dict) else {}
        ns = _num(rep.get("swap_prompts"))
        if not math.isfinite(ns):
            ns = float(swap_cens.get("n", 0))
        ns2 = _num(cs["2"]) if "2" in cs else float(swap_cens.get("2", 0))
        ns3 = max(_num(cs["3"]) if "3" in cs else 0.0, float(swap_cens.get("3", 0)),
                  _num(rep.get("swap_n_overlength")) if math.isfinite(_num(rep.get("swap_n_overlength"))) else 0.0)
        sshare = ns2 / ns if ns > 0 else NAN
        out["swap"] = {"n_rows": ns, "n_censored2": ns2, "censored2_share": sshare, "overlength": ns3,
                       "censored2_share_le_0.005": bool(sshare <= CENS2_MAX), "overlength_eq_0": bool(ns3 == 0)}
        ok = ok and out["swap"]["censored2_share_le_0.005"] and out["swap"]["overlength_eq_0"]
    out["E1"] = bool(ok)
    return out


def load_run(run_dir: Path, tag: str, panel_sha1: str | None, model: str, variant: str | None,
             need_swap: bool = False) -> dict:
    """One scoring directory -> {status, reason, ...}; status OK or the exclusion code (see OPERATIONALIZATIONS)."""
    out = {"run": tag}
    if not run_dir.is_dir():
        return {**out, "status": "ABSENT", "reason": f"no scoring directory {tag}"}
    rep_p, sc_p, sw_p = run_dir / "report.json", run_dir / "scores.csv.gz", run_dir / "swap_prior.csv.gz"
    miss = [p.name for p in (sc_p, rep_p) + ((sw_p,) if need_swap else ()) if not p.is_file()]
    if miss:
        return {**out, "status": "INCOMPLETE", "reason": f"{tag}: missing {miss} (no completion marker or output)"}
    try:
        rep = json.loads(rep_p.read_text(encoding="utf-8"))
        sc = fx.load_scores(sc_p)
        swap = fx.load_swap_positions(sw_p) if sw_p.is_file() else None
    except (OSError, ValueError, KeyError, EOFError, csv.Error) as e:
        return {**out, "status": "UNREADABLE", "reason": f"{tag}: {type(e).__name__}: {e}"}
    integ = integrity(rep, sc["censoring"], swap["censoring"] if (swap is not None and need_swap) else None)
    sha = rep.get("data_sha1")
    ident = {"report_data_sha1": sha, "panel_sha1": panel_sha1,
             "match": None if sha is None or panel_sha1 is None else sha == panel_sha1}
    backbone = rep.get("backbone") or (Path(str(rep["model"])).name if rep.get("model") else None)
    key = run_dir / "run.key"
    out.update(integrity=integ, panel_identity=ident, backbone=backbone, variant=rep.get("variant"),
               adapter=bool(rep.get("lora")), hist_len=rep.get("hist_len"), swap_k=rep.get("swap_k"),
               run_key_sha1=fx.file_sha1(key) if key.is_file() else None,
               scores_censoring=sc["censoring"])
    if not integ["E1"]:
        status, reason = "FAILED_INTEGRITY", f"{tag}: E1 fails ({_e1_text(integ)})"
    elif ident["match"] is False:
        status, reason = "PANEL_MISMATCH", f"{tag}: report.json data_sha1 {sha} is not the panel's {panel_sha1}"
    elif variant is not None and rep.get("variant") is not None and rep.get("variant") != variant:
        status, reason = "VARIANT_MISMATCH", f"{tag}: variant {rep.get('variant')} != the split's {variant}"
    elif (model == "zeroshot") == bool(rep.get("lora")):
        status, reason = "MODEL_MISMATCH", (f"{tag}: zeroshot scored with an adapter" if model == "zeroshot"
                                            else f"{tag}: no adapter (report.json lora) for an adapter model")
    else:
        status, reason = "OK", None
    out.update(status=status, reason=reason)
    if status == "OK":
        out["sc"], out["swap"] = sc, swap
    return out


def _e1_text(integ: dict) -> str:
    t = (f"censored=2 share {integ['censored2_share']:.4f}, overlength {integ['overlength']:g}, mean Yes+No mass "
         f"{integ['mean_yes_no_mass']:.4f}")
    if "swap" in integ:
        t += f"; swap censored=2 share {integ['swap']['censored2_share']:.4f}, overlength {integ['swap']['overlength']:g}"
    return t


def to_pairs(sc: dict, cx, expected, question: str = "like") -> tuple[np.ndarray, dict]:
    """The question's logits aligned to the EVAL pairs by (user_id, item_id); join diagnostics against the pairs the
    arm's panel holds (`expected` mask)."""
    vec = np.full(cx.n, NAN)
    join, present = Counter(), np.zeros(cx.n, bool)
    for (u, i, lab) in sc["meta"].values():
        j = cx.ui.get((u, i))
        if j is not None:
            present[j] = True
    for k, lg in sc["L"].get(question, {}).items():
        u, i, lab = sc["meta"][k]
        j = cx.ui.get((u, i))
        if j is None:
            join["score_rows_without_panel_pair"] += 1
        elif int(cx.y[j]) != int(lab):
            join["score_rows_label_mismatch_excluded"] += 1
        else:
            join["duplicate_pairs"] += int(math.isfinite(vec[j]))
            vec[j] = lg
    join["panel_pairs_without_score_row"] = int((expected & ~present).sum())
    join["panel_pairs_not_finite"] = int((expected & present & ~np.isfinite(vec)).sum())
    join["score_rows_outside_panel_rows"] = int((present & ~expected).sum())
    vec[~expected] = NAN
    return vec, {k: v for k, v in join.items() if v}


# ---------------------------------------------------------------- inputs: panels and references
def make_ctx(eval_rows: list, T: float, sd_ids: list, train_items: set | None) -> SimpleNamespace:
    E = fx.panel_pairs(eval_rows)
    users = E["user"].astype(str)
    ustr, uc = np.unique(users, return_inverse=True)
    ui, dup = {}, 0
    for j, k in enumerate(zip(users.tolist(), E["item"].astype(str).tolist())):
        dup += k in ui
        ui[k] = j
    ts = E["ts"]
    test, cal = np.isfinite(ts) & (ts >= T), np.isfinite(ts) & (ts < T)
    sd_set = set(sd_ids)
    in_sd = np.array([u in sd_set for u in users.tolist()], bool)
    pop = E["pop"]
    b5 = np.full(E["n"], -1)
    fin = np.isfinite(pop)
    if fin.any():
        b5[fin] = rank_bins(pop[fin], POP_BINS)
    fold_a_ids = user_halves(sd_ids, FOLD_SEED)
    seen = (np.array([i in train_items for i in E["item"].astype(str).tolist()], bool)
            if train_items is not None else None)
    return SimpleNamespace(E=E, n=E["n"], users=users, uc=uc.reshape(-1), ustr=ustr, item=E["item"].astype(str),
                           y=E["label"].astype(int), ts=ts, test=test, cal=cal, sd=in_sd, sd_test=in_sd & test,
                           pop=pop, lpop=np.log1p(pop), b5=b5, head=b5 == HEAD_BIN, tail=b5 == TAIL_BIN, ui=ui,
                           n_duplicate_pairs=dup, fold_a=np.array([u in fold_a_ids for u in users.tolist()], bool),
                           seen=seen, sd_ids=list(sd_ids))


def refs_identity(domain: str, eval_sha1: str, T: float, source: str, n_boot: int, seed: int) -> dict:
    return {"format": REFS_FORMAT, "domain": domain, "eval_sha1": eval_sha1, "T": T, "source": source,
            "shrink_k": SHRINK_K, **MF_KW, "n_boot": n_boot, "seed": seed}


def compute_refs(cx, raw, source: str, T: float, n_boot: int, seed: int) -> dict:
    """q-hat and the temporal MF with the explicit cutoff T (forensics.cf_references on the whole EVAL panel)."""
    events = fx.load_raw_events(raw, source)
    groups = fx._groups(cx.E["user"])
    pairs: dict = {}
    block, pm = fx.cf_references(cx.E, groups, events, shrink_k=SHRINK_K, n_boot=n_boot, seed=seed, cutoff=T,
                                 pairs_out=pairs, **MF_KW)
    if not pairs:
        raise ValueError(f"temporal MF not fitted: {block.get('mf_temporal')}")
    desc = None
    if source != "ml1m":
        try:
            desc = fx.amazon_desc_lengths(Path(raw), source, set(cx.item.tolist()))
        except (OSError, KeyError):
            desc = None
    arrays = {"q_hat": pm["mean_prior_shrunk"], "q_hat_unshrunk": pm["mean_prior"], "n_prior": pm["n_prior"],
              "mf_score": pairs["mf_score"], "mf_residual": pairs["mf_residual"], "mf_item_bias": pairs["mf_item_bias"],
              "mf_user_bias": pairs["mf_user_bias"], "mf_warm": pairs["mf_warm"].astype(float)}
    keep = ("candidate_timestamps", "n_pairs", "n_users", "n_raw_events", "n_first_events",
            "n_candidate_pairs_not_in_raw", "item_mean_prior", "item_mean_prior_shrunk", "mf_temporal")
    return {"arrays": {k: np.asarray(v, float) for k, v in arrays.items()},
            "cf_block": {k: block[k] for k in keep if k in block}, "desc_len": desc}


def save_refs(path: Path, ident: dict, cx, refs: dict) -> None:
    obj = {"identity": ident, "user": cx.users.tolist(), "item": cx.item.tolist(),
           "arrays": {k: v.tolist() for k, v in refs["arrays"].items()}, "cf_block": refs["cf_block"],
           "desc_len": refs["desc_len"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(strict_json(obj), allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def load_refs(path: Path, ident: dict, cx) -> tuple[dict | None, str | None]:
    try:
        obj = read_json(path)
    except (OSError, ValueError) as e:
        return None, f"unreadable ({type(e).__name__})"
    if obj.get("identity") != json.loads(json.dumps(strict_json(ident))):
        return None, "identity differs (domain, eval.jsonl sha1, T or estimator settings)"
    if obj.get("user") != cx.users.tolist() or obj.get("item") != cx.item.tolist():
        return None, "pair keys differ from eval.jsonl"
    arrays = {k: np.array([NAN if x is None else x for x in v], float) for k, v in obj["arrays"].items()}
    return {"arrays": arrays, "cf_block": obj.get("cf_block"), "desc_len": obj.get("desc_len")}, None


# ---------------------------------------------------------------- E-C
def platt_point(L, y, cal_rows, test_rows) -> dict:
    """The CAL-fit Platt map applied to TEST: slope, intercept, ECE, Brier, Brier skill; NaN when CAL has one class."""
    xc, yc = L[cal_rows], y[cal_rows]
    out = {"platt_slope": NAN, "platt_intercept": NAN, "ECE": NAN, "Brier": NAN, "Brier_skill": NAN,
           "base_rate_cal": float(yc.mean()) if len(yc) else NAN}
    xt, yt = L[test_rows], y[test_rows]
    if len(yc) < 2 or not 0 < yc.sum() < len(yc) or not len(yt):
        return out
    a, b = platt_fit(xc, yc)
    p = sigmoid(a * xt + b)
    bs = brier(p, yt)
    ref = float(np.mean((out["base_rate_cal"] - yt) ** 2))
    out.update(platt_slope=a, platt_intercept=b, ECE=ece(p, yt, ECE_BINS, adaptive=True), Brier=bs,
               Brier_skill=1 - bs / ref if ref > 0 else NAN)
    return out


def reliability_equal_mass(p, y, n_bins: int = ECE_BINS) -> list:
    """The bins of metrics.ece(adaptive=True): sum over bins of share x abs_gap equals the ECE."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if not len(p):
        return []
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append({"bin": b, "n": int(m.sum()), "share": float(m.mean()), "mean_p": float(p[m].mean()),
                        "frac_like": float(y[m].mean()), "abs_gap": float(abs(p[m].mean() - y[m].mean()))})
    return out


def topk_anatomy(L, y, users, rows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(margin, correct, c_u) per row (NaN outside): for a user with both classes among `rows` (finite L), k = the
    user's likes there, c_u = midpoint of the k-th and (k+1)-th largest L, decision = 1[L >= c_u]."""
    margin, correct, cu = np.full(len(L), NAN), np.full(len(L), NAN), np.full(len(L), NAN)
    ok = np.asarray(rows, bool) & np.isfinite(L)
    for idx in fx._groups(np.asarray(users)[ok]).values():
        j = np.flatnonzero(ok)[idx]
        k = int(y[j].sum())
        if k == 0 or k == len(j):
            continue
        s = np.sort(L[j])[::-1]
        c = (s[k - 1] + s[k]) / 2
        cu[j] = c
        margin[j] = np.abs(L[j] - c)
        correct[j] = ((L[j] >= c).astype(int) == y[j]).astype(float)
    return margin, correct, cu


ANATOMY_KEYS = ("P_wrong_bottom", "P_wrong_middle", "P_wrong_top", "share_errors_top", "share_correct_bottom",
                "AUROC_margin_correct", "AURC", "topk_error_rate")


def anatomy_stats(margin, correct) -> np.ndarray:
    m, c = np.asarray(margin, float), np.asarray(correct, float)
    if len(m) < 3:
        return np.full(len(ANATOMY_KEYS), NAN)
    t = rank_bins(m, 3)
    w = 1 - c
    pw = [float(w[t == b].mean()) if (t == b).any() else NAN for b in range(3)]
    se = float(w[t == 2].sum() / w.sum()) if w.sum() > 0 else NAN
    sc = float(c[t == 0].sum() / c.sum()) if c.sum() > 0 else NAN
    return np.array(pw + [se, sc, auroc(m, c.astype(int)), aurc(m, c), float(w.mean())], float)


EC_KEYS = ("platt_slope", "platt_intercept", "ECE", "Brier", "Brier_skill") + ANATOMY_KEYS


def ec_block(cx, L: dict, n_boot: int, seed: int, n_reg: int) -> dict:
    """E-C for the models of one regime on identical rows (CAL u TEST rows finite for every model)."""
    models = list(L)
    R = (cx.cal | cx.test).copy()
    for m in models:
        R &= np.isfinite(L[m])
    anat = {m: topk_anatomy(L[m], cx.y, cx.uc, R & cx.test) for m in models}
    aok = R & cx.test & np.isfinite(anat[models[0]][0])
    ridx = np.flatnonzero(R)
    cl = Clusters(cx.uc[ridx])

    def stat(m, rows):
        cal_r, test_r = rows[cx.cal[rows]], rows[cx.test[rows]]
        pt = platt_point(L[m], cx.y, cal_r, test_r)
        ar = rows[aok[rows]]
        return np.r_[[pt[k] for k in EC_KEYS[:5]], anatomy_stats(anat[m][0][ar], anat[m][1][ar])]

    est = np.array([stat(m, ridx) for m in models])
    draws = np.full((n_boot, len(models), len(EC_KEYS)), NAN)
    for b, pos in enumerate(cl.draws(n_boot, seed)):
        rows = ridx[pos]
        draws[b] = [stat(m, rows) for m in models]
    n_test = len(np.unique(cx.uc[R & cx.test]))
    n_anat = len(np.unique(cx.uc[aok]))
    n_pairs_test, n_pairs_anat = int((R & cx.test).sum()), int(aok.sum())

    def nn(k):
        return (n_anat, n_pairs_anat) if k in ANATOMY_KEYS else (n_test, n_pairs_test)
    per = {}
    for j, m in enumerate(models):
        pt = platt_point(L[m], cx.y, ridx[cx.cal[ridx]], ridx[cx.test[ridx]])
        tr = np.flatnonzero(R & cx.test)
        p_test = sigmoid(pt["platt_slope"] * L[m][tr] + pt["platt_intercept"]) if math.isfinite(pt["platt_slope"]) \
            else np.full(len(tr), NAN)
        per[m] = {k: rec(est[j, i], draws[:, j, i], *nn(k)) for i, k in enumerate(EC_KEYS)}
        per[m]["base_rate_cal"] = pt["base_rate_cal"]
        per[m]["n_cal_rows"] = int((R & cx.cal).sum())
        per[m]["reliability_equal_mass"] = reliability_equal_mass(p_test, cx.y[tr]) if np.isfinite(p_test).all() else []
        per[m]["reliability_equal_width"] = [
            {"lo": lo, "hi": hi, "n": n, "mean_p": mp, "frac_like": fy}
            for lo, hi, n, mp, fy in (reliability_bins(p_test, cx.y[tr], ECE_BINS) if np.isfinite(p_test).all() else [])]
    mean = {k: rec(np.nanmean(est[:, i]) if np.isfinite(est[:, i]).any() else NAN, np.nanmean(draws[:, :, i], 1)
                   if n_boot else None, *nn(k)) for i, k in enumerate(EC_KEYS)} if len(models) else {}
    return {"rows": {"n_users_test": n_test, "n_pairs_test": n_pairs_test, "n_users_anatomy": n_anat,
                     "n_pairs_anatomy": n_pairs_anat, "n_pairs_cal": int((R & cx.cal).sum())},
            "per_model": per, "mean_over_seeds": mean,
            "seeds": {k: seed_summary([per[m][k]["est"] for m in models], n_reg) for k in ("ECE", "Brier_skill",
                                                                                          "AUROC_margin_correct")},
            "platt": {m: (per[m]["platt_slope"]["est"], per[m]["platt_intercept"]["est"]) for m in models}}


# ---------------------------------------------------------------- E-D
def logit_fit(X, y, l2: float = 1e-4, iters: int = 100):
    """Logistic regression with intercept by Newton steps; features standardised on the fitting rows, L2 on the slopes
    only (stats.platt_fit's regulariser). Returns (mean, sd, w) for logit_eta."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    m, s = X.mean(0), X.std(0)
    s = np.where(s > 0, s, 1.0)
    Z = np.column_stack([np.ones(len(X)), (X - m) / s])
    w = np.zeros(Z.shape[1])
    R = l2 * np.eye(Z.shape[1])
    R[0, 0] = 0.0
    for _ in range(iters):
        p = sigmoid(Z @ w)
        g = Z.T @ (p - y) + R @ w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + R
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return m, s, w


def logit_eta(model, X) -> np.ndarray:
    m, s, w = model
    return w[0] + ((np.asarray(X, float) - m) / s) @ w[1:]


def crossfit(X, y, fold_a, rows) -> np.ndarray:
    """Pooled cross-fitted linear predictor on `rows`: fold A's rows scored by the fit on fold B's rows and vice versa
    (NaN for a fold whose fitting fold lacks a class)."""
    X = np.asarray(X, float).reshape(len(y), -1)
    eta = np.full(len(y), NAN)
    A, B = rows & fold_a, rows & ~fold_a
    for fit, pred in ((B, A), (A, B)):
        if pred.any() and fit.sum() >= 2 and 0 < y[fit].sum() < fit.sum():
            eta[pred] = logit_eta(logit_fit(X[fit], y[fit]), X[pred])
    return eta


def within_user_sums(users, rows, cols: list, prods: list) -> tuple[list, np.ndarray]:
    """Per user (sorted), sums of products of the within-user centred columns over `rows`."""
    idx = np.flatnonzero(rows)
    if not len(idx):
        return [], np.zeros((0, len(prods)))
    uu, inv = np.unique(np.asarray(users)[idx], return_inverse=True)
    inv = inv.reshape(-1)
    cnt = np.bincount(inv)
    C = [np.asarray(x, float)[idx] - (np.bincount(inv, np.asarray(x, float)[idx]) / cnt)[inv] for x in cols]
    S = np.column_stack([np.bincount(inv, C[a] * C[b], minlength=len(uu)) for a, b in prods])
    return uu.tolist(), S


SHARE_PRODS = [(0, 1), (0, 0), (1, 1), (2, 3), (2, 2), (3, 3)]   # cols: pi_half_a, pi_half_b, L, pi


def shares_from_sums(S) -> dict:
    """r_c, r8c, rho and the shares from pooled within-user sums (columns of SHARE_PRODS)."""
    s = np.asarray(S, float).reshape(-1, 6)
    with np.errstate(divide="ignore", invalid="ignore"):
        rc = s[:, 0] / np.sqrt(s[:, 1] * s[:, 2])
        rho = s[:, 3] / np.sqrt(s[:, 4] * s[:, 5])
        r8 = 2 * rc / (1 + rc)
        share = rho ** 2 / r8
    return {"r_c": rc, "r8c": r8, "rho": rho, "item_prior_share": share, "non_prior_share": 1 - share}


SHARE_KEYS = ("r_c", "r8c", "rho", "item_prior_share", "non_prior_share")


def shares_block(users, rows, per_model: dict, n_boot: int, seed: int, n_reg: int) -> dict:
    """per_model = {m: (pi_a, pi_b, L, pi)}; identical rows (`rows` finite in every column of every model)."""
    models = list(per_model)
    R = np.asarray(rows, bool).copy()
    for cols in per_model.values():
        for x in cols:
            R &= np.isfinite(x)
    stats_ = {m: within_user_sums(users, R, list(per_model[m]), SHARE_PRODS) for m in models}
    keys = stats_[models[0]][0] if models else []
    Sall = np.column_stack([stats_[m][1] for m in models]) if models else np.zeros((0, 0))
    nu = len(keys)
    cnt = Counter(np.asarray(users)[R].tolist())
    nu2 = sum(c >= 2 for c in cnt.values())
    D = mean_draws(Sall, n_boot, seed) * nu if nu else np.full((n_boot, 6 * len(models)), NAN)
    per = {}
    est_cols = {}
    for j, m in enumerate(models):
        e = shares_from_sums(Sall[:, 6 * j:6 * j + 6].sum(0))
        d = shares_from_sums(D[:, 6 * j:6 * j + 6])
        est_cols[m] = (e, d)
        per[m] = {k: rec(float(e[k][0]), d[k], nu2, int(R.sum())) for k in SHARE_KEYS}
        r8 = float(e["r8c"][0])
        per[m]["shares_reading"] = ("uninterpretable" if not math.isfinite(r8) or r8 < R8C_MIN else "interpretable")
    mean = {}
    if models:
        for k in SHARE_KEYS:
            e = np.mean([est_cols[m][0][k][0] for m in models])
            d = np.mean([est_cols[m][1][k] for m in models], 0)
            mean[k] = rec(e, d, nu2, int(R.sum()))
        mean["shares_reading"] = ("uninterpretable" if not math.isfinite(mean["r8c"]["est"])
                                  or mean["r8c"]["est"] < R8C_MIN else "interpretable")
    return {"rows": {"n_pairs": int(R.sum()), "n_users": nu, "n_users_with_2_or_more_pairs": int(nu2)},
            "per_model": per, "mean_over_seeds": mean,
            "seeds": {k: seed_summary([per[m][k]["est"] for m in models], n_reg) for k in ("item_prior_share", "r8c")}}


def stacker_block(cx, models: list, L: dict, PI: dict, q, mfres, rows, n_boot: int, seed: int, n_reg: int) -> dict:
    """Cross-fitted stackers M0-M3 per model on identical rows; G = dUAUC(M2 - M1), G_CF = dUAUC(M3 - M0)."""
    R = np.asarray(rows, bool) & np.isfinite(q) & np.isfinite(mfres)
    for m in models:
        R &= np.isfinite(L[m]) & np.isfinite(PI[m])
    feats_common = {"q_hat": q, "mf_residual": mfres}
    etas = {}
    eta_cf = {k: crossfit(np.column_stack([feats_common[f] for f in STACKERS[k]]), cx.y, cx.fold_a, R)
              for k in ("M0", "M3")}
    for m in models:
        f = {**feats_common, "pi": PI[m], "e_hat": L[m] - PI[m]}
        etas[m] = {k: crossfit(np.column_stack([f[c] for c in STACKERS[k]]), cx.y, cx.fold_a, R)
                   for k in ("M1", "M2")}
        etas[m].update(eta_cf)
    pu = {(m, k): user_aucs(etas[m][k], cx.y, cx.uc, R) for m in models for k in STACKERS}
    keys = sorted(set.intersection(*(set(d) for d in pu.values()))) if pu else []
    nu, npairs = len(keys), _rows_of_users(cx.uc, R, keys)
    cols = []
    for m in models:
        cols.append([pu[(m, "M2")][u] - pu[(m, "M1")][u] for u in keys])
    gcf = [pu[(models[0], "M3")][u] - pu[(models[0], "M0")][u] for u in keys] if models else []
    V = np.column_stack(cols + [gcf]) if keys else np.zeros((0, len(models) + 1))
    D = mean_draws(V, n_boot, seed)
    per = {}
    for j, m in enumerate(models):
        per[m] = {"G": rec(V[:, j].mean() if nu else NAN, D[:, j], nu, npairs, contrast=True),
                  "UAUC": {k: float(np.mean([pu[(m, k)][u] for u in keys])) if nu else NAN for k in STACKERS}}
    k = len(models)
    g_mean = rec(V[:, :k].mean(1).mean() if nu and k else NAN, D[:, :k].mean(1) if k else None, nu, npairs,
                 contrast=True)
    g_cf = rec(V[:, k].mean() if nu else NAN, D[:, k] if len(D) else None, nu, npairs, contrast=True)
    out = {"rows": {"n_pairs": npairs, "n_users": nu, "fold_a_users": int(len(set(cx.uc[R & cx.fold_a].tolist()))),
                    "fold_b_users": int(len(set(cx.uc[R & ~cx.fold_a].tolist())))},
           "per_model": per, "G_mean_over_seeds": g_mean, "G_CF": g_cf,
           "seeds": seed_summary([per[m]["G"]["est"] for m in models], n_reg, True, g_mean["est"])}
    ok_ratio = bool(g_cf["ci_excludes_0"])
    ratio = {}
    for j, m in enumerate(models + ["mean_over_seeds"]):
        num_e = V[:, j].mean() if j < k else V[:, :k].mean(1).mean()
        num_d = D[:, j] if j < k else D[:, :k].mean(1)
        if ok_ratio and nu:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio[m] = rec(num_e / V[:, k].mean(), num_d / D[:, k], nu, npairs)
        else:
            ratio[m] = None
    out["G_over_G_CF"] = ratio
    out["G_over_G_CF_reported"] = ok_ratio
    out["G_over_G_CF_rule"] = "reported only when G_CF's CI excludes 0 (A3 section 3), so never when G_CF is descriptive"
    return out


def starperm_block(cx, models: list, L: dict, LP: dict, rows, n_boot: int, seed: int, n_reg: int) -> dict:
    """dUAUC = UAUC(L) - per-user mean over k of AUC(L_perm_k); tau_P = L - mean_k L_perm_k (diag_battery)."""
    R = np.asarray(rows, bool).copy()
    for m in models:
        R &= np.isfinite(L[m])
        for v in LP[m]:
            R &= np.isfinite(v)
    pu_l = {m: user_aucs(L[m], cx.y, cx.uc, R) for m in models}
    pu_p = {m: [user_aucs(v, cx.y, cx.uc, R) for v in LP[m]] for m in models}
    pu_ml = {m: user_aucs(np.mean(LP[m], 0), cx.y, cx.uc, R) for m in models}
    keys = sorted(set.intersection(*(set(pu_l[m]) for m in models))) if models else []
    nu, npairs = len(keys), _rows_of_users(cx.uc, R, keys)
    V = np.array([[pu_l[m][u] - np.mean([d[u] for d in pu_p[m]]) for m in models] for u in keys],
                 float).reshape(nu, len(models))
    V2 = np.array([[pu_l[m][u] - pu_ml[m][u] for m in models] for u in keys], float).reshape(nu, len(models))
    D, D2 = mean_draws(V, n_boot, seed), mean_draws(V2, n_boot, seed)
    per = {}
    for j, m in enumerate(models):
        tau = (L[m] - np.mean(LP[m], 0))[R]
        sd = pooled_within_sd(tau, cx.uc[R])
        d = rec(V[:, j].mean() if nu else NAN, D[:, j], nu, npairs, contrast=True)
        per[m] = {"dUAUC_L_minus_perm": d,
                  "dUAUC_L_minus_mean_perm_logit": rec(V2[:, j].mean() if nu else NAN, D2[:, j], nu, npairs,
                                                       contrast=True),
                  "UAUC_L": float(np.mean([pu_l[m][u] for u in keys])) if nu else NAN,
                  "UAUC_perm_k": [float(np.mean([dd[u] for u in keys])) if nu else NAN for dd in pu_p[m]],
                  "tau_mean": float(tau.mean()) if len(tau) else NAN, "tau_sd_within_user": sd,
                  "wording_flag_model_ignores_user_ratings": (
                      None if not (math.isfinite(d["est"]) and math.isfinite(sd))
                      else bool(abs(d["est"]) < STARPERM_FLAGS["abs_dUAUC_lt"]
                                and sd < STARPERM_FLAGS["within_user_sd_tau_lt"]))}
    k = len(models)
    mean = rec(V.mean(1).mean() if nu and k else NAN, D.mean(1) if k else None, nu, npairs, contrast=True)
    return {"rows": {"n_pairs": npairs, "n_users": nu, "n_pairs_tau": int(R.sum())}, "K": len(STARPERM_ARMS),
            "per_model": per, "dUAUC_mean_over_seeds": mean,
            "tau_sd_within_user_mean_over_seeds": float(np.mean([per[m]["tau_sd_within_user"] for m in models]))
            if models else NAN,
            "seeds": seed_summary([per[m]["dUAUC_L_minus_perm"]["est"] for m in models], n_reg, True, mean["est"]),
            "flags": STARPERM_FLAGS, "flags_note": "Amendment 2 D thresholds: wording flags, never gates"}


# ---------------------------------------------------------------- E-E
def _item_means(values, inv, n_items) -> np.ndarray:
    x = np.asarray(values, float)
    ok = np.isfinite(x)
    s = np.bincount(inv[ok], x[ok], minlength=n_items)
    c = np.bincount(inv[ok], minlength=n_items)
    return np.divide(s, c, out=np.full(n_items, NAN), where=c > 0)


def partial_pair_block(cx, models, L: dict, Z, rows, n_boot: int, seed: int) -> dict:
    """Pair-level partial Spearman of each model's L with log1p(pop) given Z, user-cluster CI (shared draws)."""
    ok = np.asarray(rows, bool) & np.isfinite(cx.lpop) & np.isfinite(Z).all(1)
    for m in models:
        ok &= np.isfinite(L[m])
    idx = np.flatnonzero(ok)
    est = [fx.partial_spearman(L[m][idx], cx.lpop[idx], Z[idx]) for m in models]
    draws = np.full((n_boot, len(models)), NAN)
    if len(idx) >= Z.shape[1] + 3:
        for b, pos in enumerate(Clusters(cx.uc[idx]).draws(n_boot, seed)):
            r = idx[pos]
            draws[b] = [fx.partial_spearman(L[m][r], cx.lpop[r], Z[r]) for m in models]
    nu = len(np.unique(cx.uc[idx]))
    per = {m: rec(est[j], draws[:, j], nu, len(idx)) for j, m in enumerate(models)}
    return {"per_model": per, "mean_over_seeds": rec(np.mean(est) if est else NAN, draws.mean(1) if models else None,
                                                     nu, len(idx))}


def partial_item_block(items, models, X: dict, lpop_i, Zi, n_boot: int, seed: int, n_users: int) -> dict:
    """Item-level partial Spearman (item bootstrap, shared draws); X = {m: per-item values}."""
    ok = np.isfinite(lpop_i) & np.isfinite(Zi).all(1)
    for m in models:
        ok &= np.isfinite(X[m])
    idx = np.flatnonzero(ok)
    est = [fx.partial_spearman(X[m][idx], lpop_i[idx], Zi[idx]) for m in models]
    draws = np.full((n_boot, len(models)), NAN)
    if len(idx) >= Zi.shape[1] + 3:
        for b, pick in enumerate(unit_draws(len(idx), n_boot, seed)):
            r = idx[pick]
            draws[b] = [fx.partial_spearman(X[m][r], lpop_i[r], Zi[r]) for m in models]
    per = {m: rec(est[j], draws[:, j], n_users, None, n_items=int(len(idx))) for j, m in enumerate(models)}
    return {"per_model": per, "mean_over_seeds": rec(np.mean(est) if est else NAN, draws.mean(1) if models else None,
                                                     n_users, None, n_items=int(len(idx)))}


def bias_index_block(cx, models, L: dict, platt: dict, rows, n_boot: int, seed: int) -> dict:
    """metrics.bias_index (10 confidence bins, residual-adjusted) head vs tail on TEST rows, conf = CAL-fit Platt."""
    ok = np.asarray(rows, bool) & (cx.b5 >= 0)
    for m in models:
        ok &= np.isfinite(L[m])
    idx = np.flatnonzero(ok)
    grp = np.where(cx.head, "head", np.where(cx.tail, "tail", "mid"))
    conf = {m: sigmoid(platt[m][0] * L[m] + platt[m][1]) for m in models}

    def one(m, r):
        bi = bias_index(conf[m][r], cx.y[r], grp[r], BI_BINS, True)
        h, t = bi.get("head", NAN), bi.get("tail", NAN)
        return np.array([h, t, h - t], float)
    est = np.array([one(m, idx) for m in models]).reshape(len(models), 3)
    draws = np.full((n_boot, len(models), 3), NAN)
    if len(idx):
        for b, pos in enumerate(Clusters(cx.uc[idx]).draws(n_boot, seed)):
            draws[b] = [one(m, idx[pos]) for m in models]
    nu = len(np.unique(cx.uc[idx]))
    names = ("head", "tail", "head_minus_tail")
    per = {m: {k: rec(est[j, i], draws[:, j, i], nu, len(idx), contrast=(k == "head_minus_tail"))
               for i, k in enumerate(names)} for j, m in enumerate(models)}
    mean = {k: rec(est[:, i].mean() if len(models) else NAN, draws[:, :, i].mean(1) if models else None, nu, len(idx),
                   contrast=(k == "head_minus_tail")) for i, k in enumerate(names)}
    return {"rows": {"n_pairs": int(len(idx)), "n_users": nu, "n_head": int((ok & cx.head).sum()),
                     "n_tail": int((ok & cx.tail).sum())}, "per_model": per, "mean_over_seeds": mean}


def controls_for(cx, domain: str, refs: dict | None) -> tuple[dict, dict]:
    """(registered controls beyond q-hat, the games sensitivity controls) as pair-level arrays."""
    E = cx.E

    def amazon():
        dl = (refs or {}).get("desc_len") or {}
        d = np.array([dl[i] if i in dl else fx.desc_length_from_text(t) for i, t in zip(cx.item.tolist(),
                                                                                       E["text"].tolist())], float)
        return {"desc_len": d, "has_store": np.array([float(bool(str(b).strip())) for b in E["brand"]], float)}
    if domain == "ml1m":
        return {"release_year": np.array([fx.release_year(t) for t in E["title"]], float)}, {}
    if domain in DOMAIN_CONTROLS:
        return amazon(), {}
    return {}, amazon()


def ee_block(cx, domain, models, L: dict, PI: dict | None, refs, platt: dict, n_boot: int, seed: int,
             n_reg: int) -> dict:
    out = {"definition": "partial Spearman (forensics.partial_spearman) against log1p(candidate_popularity) "
                         "controlling for q-hat (shrunk) and the domain controls; Bias Index head vs tail with the "
                         "CAL-fit Platt probability; tail-only UAUC (exploratory)",
           "popularity_bins": "stats.rank_bins(candidate_popularity, 5) over all EVAL candidates; head = 4, tail = 0"}
    R = cx.test.copy()
    for m in models:
        R &= np.isfinite(L[m])
    if refs is None:
        out["partial_spearman"] = {"available": False, "reason": "no q-hat (needs --raw or --refs)"}
    else:
        q = refs["arrays"]["q_hat"]
        ctl, sens = controls_for(cx, domain, refs)
        out["controls"] = ["q_hat"] + list(ctl)
        uniq, inv = np.unique(cx.item[R], return_inverse=True)
        inv = inv.reshape(-1)
        ridx = np.flatnonzero(R)

        def spec(controls):
            Z = np.column_stack([q] + [controls[c] for c in controls])
            Zi = np.column_stack([_item_means(Z[ridx, j], inv, len(uniq)) for j in range(Z.shape[1])])
            res = {"L_pair": partial_pair_block(cx, models, L, Z, R, n_boot, seed)}
            nu = res["L_pair"]["per_model"][models[0]]["n_users"] if models else 0
            Li = {m: _item_means(L[m][ridx], inv, len(uniq)) for m in models}
            lp_i = _item_means(cx.lpop[ridx], inv, len(uniq))
            res["L_item"] = partial_item_block(uniq, models, Li, lp_i, Zi, n_boot, seed, nu)
            if PI is not None:
                pin = {m: _item_means(np.where(cx.sd_test[ridx], PI[m][ridx], NAN), inv, len(uniq)) for m in models}
                res["pi_item"] = partial_item_block(uniq, models, pin, lp_i, Zi, n_boot, seed,
                                                    len(np.unique(cx.uc[R & cx.sd_test])))
            else:
                res["pi_item"] = {"available": False, "reason": "no swap arm for every model of the regime"}
            return res
        out["partial_spearman"] = spec(ctl)
        if sens:
            out["controls_sensitivity"] = ["q_hat"] + list(sens)
            s = spec({**ctl, **sens}) if n_boot == 0 else None
            if s is None:
                s = spec_point(cx, models, L, PI, q, {**ctl, **sens}, R)
            out["partial_spearman_sensitivity_point_estimates"] = s
    out["bias_index"] = bias_index_block(cx, models, L, platt, R, n_boot, seed)
    out["tail_UAUC_exploratory"] = uauc_models(L, cx.y, cx.uc, R & cx.tail, n_boot, seed, n_registered=n_reg)
    return out


def spec_point(cx, models, L, PI, q, controls, R) -> dict:
    """Point estimates of the E-E partial Spearman under a sensitivity control set (no CI)."""
    ridx = np.flatnonzero(R)
    Z = np.column_stack([q] + [controls[c] for c in controls])
    uniq, inv = np.unique(cx.item[R], return_inverse=True)
    inv = inv.reshape(-1)
    Zi = np.column_stack([_item_means(Z[ridx, j], inv, len(uniq)) for j in range(Z.shape[1])])
    lp_i = _item_means(cx.lpop[ridx], inv, len(uniq))
    ok = np.isfinite(cx.lpop[ridx]) & np.isfinite(Z[ridx]).all(1)
    out = {"L_pair": {m: fx.partial_spearman(L[m][ridx][ok], cx.lpop[ridx][ok], Z[ridx][ok]) for m in models},
           "L_item": {m: fx.partial_spearman(_item_means(L[m][ridx], inv, len(uniq)), lp_i, Zi) for m in models}}
    if PI is not None:
        out["pi_item"] = {m: fx.partial_spearman(_item_means(np.where(cx.sd_test[ridx], PI[m][ridx], NAN), inv,
                                                             len(uniq)), lp_i, Zi) for m in models}
    return out


# ---------------------------------------------------------------- P1
def p1_decision(per_seed: list, mean_rec: dict, n_registered: int = 3) -> dict:
    """A3 section 3: P1 holds iff the mean over the 3 seeds of G_FT,s - G_ZS is > 0 with p < 0.05, all 3 seeds are
    positive and the mean exceeds 2 sigma_seed (and the endpoint is not descriptive by the minimum-n rule)."""
    v = [_num(x) for x in per_seed]
    m, p = _num(mean_rec.get("est")), mean_rec.get("p")
    sd = float(np.std(v, ddof=1)) if len(v) > 1 and all(math.isfinite(x) for x in v) else NAN
    cond = {"three_seeds": len(v) == n_registered and all(math.isfinite(x) for x in v),
            "not_descriptive_min_n": not mean_rec.get("descriptive_min_n", True),
            "mean_gt_0": bool(math.isfinite(m) and m > 0),
            "p_lt_0.05": bool(p is not None and _fin(p) and p < 0.05),
            "all_seeds_positive": bool(v and all(math.isfinite(x) and x > 0 for x in v)),
            "mean_gt_2_sigma_seed": bool(math.isfinite(m) and math.isfinite(sd) and m > 2 * sd)}
    holds = all(cond.values())
    verdict = "P1_HOLDS" if holds else ("INCOMPLETE" if not cond["three_seeds"] else "NO_EVIDENCE")
    return {"verdict": verdict, "holds": holds, "conditions": cond, "mean": m, "p": p, "per_seed": v,
            "sigma_seed": sd,
            "wording": ("supervised fine-tuning increases the personal information gain (P1)" if holds else
                        "no evidence that supervision adds personal evidence"),
            "rule": "P1 holds iff the mean over seeds of G_FT,s - G_ZS is > 0 with p < 0.05, all 3 seeds are positive and "
                    "the mean exceeds 2 sigma_seed (A3 section 3); fewer than 150 users makes it descriptive"}


def p1_block(cx, L: dict, PI: dict, refs, present: set, role: str, n_boot: int, seed: int) -> dict:
    need = ["zeroshot", "s0", "s1", "s2"]
    miss = [m for m in need if m not in present]
    out = {"role": role, "definition": "per user G_m = AUC(M2_m) - AUC(M1_m) of the cross-fitted stackers on S_d's TEST "
                                       "rows finite for ZS and the 3 FT seeds; contrast_s = mean over users of "
                                       "G_s - G_ZS; P1's mean = mean over seeds (identical users and rows)"}
    if miss or refs is None:
        reason = (f"missing E-D inputs (like and swap arms) for {miss}" if miss else "no q-hat (needs --raw or --refs)")
        out.update(available=False, reason=reason,
                   decision={"verdict": "INCOMPLETE", "holds": False, "reason": reason,
                             "wording": "no evidence that supervision adds personal evidence"})
        return out
    q = refs["arrays"]["q_hat"]
    R = cx.sd_test & np.isfinite(q)
    for m in need:
        R &= np.isfinite(L[m]) & np.isfinite(PI[m])
    pu = {}
    for m in need:
        f = {"q_hat": q, "pi": PI[m], "e_hat": L[m] - PI[m]}
        for k in ("M1", "M2"):
            eta = crossfit(np.column_stack([f[c] for c in STACKERS[k]]), cx.y, cx.fold_a, R)
            pu[(m, k)] = user_aucs(eta, cx.y, cx.uc, R)
    keys = sorted(set.intersection(*(set(d) for d in pu.values())))
    nu, npairs = len(keys), _rows_of_users(cx.uc, R, keys)
    G = {m: np.array([pu[(m, "M2")][u] - pu[(m, "M1")][u] for u in keys], float) for m in need}
    V = np.column_stack([G[s] - G["zeroshot"] for s in need[1:]]) if nu else np.zeros((0, 3))
    D = mean_draws(V, n_boot, seed)
    per = {s: rec(V[:, j].mean() if nu else NAN, D[:, j], nu, npairs, contrast=True,
                  G_FT=float(G[s].mean()) if nu else NAN) for j, s in enumerate(need[1:])}
    mean = rec(V.mean(1).mean() if nu else NAN, D.mean(1), nu, npairs, contrast=True)
    out.update(available=True, rows={"n_pairs": npairs, "n_users": nu}, G_ZS=float(G["zeroshot"].mean()) if nu else NAN,
               per_seed=per, mean_over_seeds=mean, decision=p1_decision([per[s]["est"] for s in need[1:]], mean))
    return out


# ---------------------------------------------------------------- knockout (A3 section 5)
def knockout_label(verdict: str) -> str:
    """decide()'s verdict as recorded under A3 section 5 (closed-hole rule)."""
    return LABEL_MAP.get(verdict, verdict)


def knockout_block(cx, panels: Path, eval_rows: list, runs: dict, sdir: Path, models: list, n_boot: int, seed: int,
                   bi_boot: int) -> dict:
    ps, pl = panels / ARM_PANEL["pseudo"], panels / ARM_PANEL["placebo"]
    out = {"definition": "pilot_pseudonym.analyze (unchanged) per model on the rows finite in every arm of every model; "
                         "label = decide()'s verdict with AMBIGUOUS -> INDETERMINATE, UNDETERMINED -> INCOMPLETE; no "
                         "--gate, no p-value, no Holm family", "label_map": LABEL_MAP}
    if not ps.is_file() or not pl.is_file():
        out.update(available=False, reason="no eval_pseudo.jsonl / eval_placebo.jsonl for this domain (the knockout "
                                           "runs on toys and games, and on sports unless cut)")
        return out
    cand = [m for m in models if m in ("zeroshot", "s0", "s1", "s2")]
    labels, arms_by_model = {}, {}
    for m in cand:
        bad = {a: runs[m][a]["status"] for a in ("like", "pseudo", "placebo") if runs[m][a]["status"] != "OK"}
        if bad:
            labels[m] = {"label": "INCOMPLETE", "reason": f"arms not usable: {bad}"}
            continue
        arms_by_model[m] = {"real": pps.load_arm(sdir / m / "like"), "pseudo": pps.load_arm(sdir / m / "pseudo"),
                            "placebo": pps.load_arm(sdir / m / "placebo")}
    if not arms_by_model:
        out.update(available=False, reason="no model with usable real, pseudo and placebo arms", labels=labels)
        return out
    common = None
    for arms in arms_by_model.values():
        for a in arms.values():
            k = set(a["L"].get("like", {}))
            common = k if common is None else common & k
    flags, src, chk = pps.treated_from_panel(panels / ARM_PANEL["like"], eval_rows)
    analyses = {}
    for m, arms in arms_by_model.items():
        f = {a: {**arm, "L": {"like": {k: v for k, v in arm["L"].get("like", {}).items() if k in common}}}
             for a, arm in arms.items()}
        res = pps.analyze(f, eval_rows, flags, n_boot, seed, bi_boot=bi_boot)
        d = res["decision"]
        labels[m] = {"label": knockout_label(d["verdict"]), "decide_verdict": d["verdict"],
                     "missing_inputs": d["missing_inputs"], "values": d["values"], "conditions": d["conditions"],
                     "n_analysed": res["rows"]["n_analysed"], "n_users": res["rows"]["n_users"],
                     "treated_share": res["treated"]["share"]}
        res["treated"].update(source=src, **chk)
        analyses[m] = res
    zs_ft = [m for m in ("zeroshot", "s0", "s1", "s2")]
    out.update(available=True, n_common_rows=len(common), treated_source=src, labels=labels, analyses=analyses,
               all_zs_and_ft_positive=all(labels.get(m, {}).get("label") == "POSITIVE" for m in zs_ft),
               hmt_pseudo_sign={m: (None if not _fin(v.get("values", {}).get("hmt_pseudo", {}).get("est"))
                                    else int(np.sign(v["values"]["hmt_pseudo"]["est"])))
                                for m, v in labels.items() if "values" in v},
               wording_note="'familiarity' needs POSITIVE for Qwen on toys and games in ZS and all 3 FT seeds and the "
                            "same sign of the Llama toys head-minus-tail (A3 section 5); read across the domain reports")
    return out


# ---------------------------------------------------------------- driver
def _regime(models_req: list, ok: set, regime: str) -> tuple[list, list]:
    reg = [m for m in REGIMES[regime] if m in models_req]
    return [m for m in reg if m in ok], [m for m in reg if m not in ok]


def _na(reason: str) -> dict:
    return {"available": False, "reason": reason}


def build(a) -> dict:
    split = read_json(a.split)
    T = float(split["T"])
    panels = Path(a.panels)
    models = [m for m in a.models.split(",") if m]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit(f"--models: unknown {unknown} (choose from {list(MODELS)})")
    eval_path = panels / ARM_PANEL["like"]
    if not eval_path.is_file():
        raise SystemExit(f"{eval_path}: no eval.jsonl")
    eval_rows = fx.read_jsonl(eval_path)
    files = split.get("files") or {}
    problems, checks = [], {}
    panel_sha = {}
    for name in sorted(set(ARM_PANEL.values()) | {"train.jsonl", "sd_users.txt", "eval_users.txt"}):
        p = panels / name
        panel_sha[name] = fx.file_sha1(p) if p.is_file() else None
        if name in files and panel_sha[name] is not None and files[name] != panel_sha[name]:
            problems.append(f"{name}: sha1 {panel_sha[name]} differs from ftgrid_split.json files ({files[name]})")
    eval_ids = [str(r["user_id"]) for r in eval_rows]
    checks["eval_user_ids_sha1_match"] = (None if not (split.get("eval") or {}).get("user_ids_sha1")
                                          else sha1_text_lines(eval_ids) == split["eval"]["user_ids_sha1"])
    sd_path = panels / ((split.get("sd") or {}).get("user_ids_path") or "sd_users.txt")
    sd_test_path = panels / ARM_PANEL["swap"]
    if sd_path.is_file():
        txt = sd_path.read_text(encoding="utf-8")
        sd_ids = [x for x in txt.split("\n") if x]
        checks["sd_user_ids_sha1_match"] = (None if not (split.get("sd") or {}).get("user_ids_sha1")
                                            else hashlib.sha1(txt.encode("utf-8")).hexdigest()
                                            == split["sd"]["user_ids_sha1"])
    elif sd_test_path.is_file():
        sd_ids = [str(r["user_id"]) for r in fx.read_jsonl(sd_test_path)]
        checks["sd_user_ids_source"] = "eval_sd_test.jsonl (sd_users.txt absent)"
    else:
        sd_ids = []
        problems.append("no sd_users.txt and no eval_sd_test.jsonl: E-D and P1 cannot run")
    train_path = panels / "train.jsonl"
    train_items = ({str(i) for r in fx.read_jsonl(train_path) for i in r["candidate_item_ids"]}
                   if train_path.is_file() else None)
    cx = make_ctx(eval_rows, T, sd_ids, train_items)
    if cx.n_duplicate_pairs:
        problems.append(f"eval.jsonl repeats {cx.n_duplicate_pairs} (user_id, item_id) pairs")
    if (~(cx.test | cx.cal)).any():
        problems.append(f"{int((~(cx.test | cx.cal)).sum())} EVAL pairs without a finite candidate timestamp")
    if sd_test_path.is_file():
        sdp = {(str(r["user_id"]), str(i)) for r in fx.read_jsonl(sd_test_path) for i in r["candidate_item_ids"]}
        mine = {(u, i) for u, i, s in zip(cx.users.tolist(), cx.item.tolist(), cx.sd_test.tolist()) if s}
        checks["eval_sd_test_pairs_equal_sd_test_rows"] = sdp == mine
        if sdp != mine:
            problems.append("eval_sd_test.jsonl pairs differ from S_d's TEST rows of eval.jsonl")
    checks["sd_users_have_both_classes_in_test"] = bool(all(
        0 < cx.y[cx.sd_test & (cx.users == u)].sum() < (cx.sd_test & (cx.users == u)).sum() for u in sd_ids[:50]))

    # ---- runs
    sdir = Path(a.scores_root) / a.domain
    sdir = sdir if sdir.is_dir() else Path(a.scores_root)
    runs, excluded = {}, []
    expected = {"like": np.ones(cx.n, bool), "pseudo": np.ones(cx.n, bool), "placebo": np.ones(cx.n, bool),
                "swap": cx.sd_test, "nohist": cx.sd_test, "starperm0": cx.sd_test, "starperm1": cx.sd_test}
    for m in models:
        runs[m] = {}
        for arm, pname in ARM_PANEL.items():
            r = load_run(sdir / m / arm, f"{m}/{arm}", panel_sha.get(pname), m, split.get("variant"),
                         need_swap=(arm == "swap"))
            runs[m][arm] = r
            if r["status"] not in ("OK", "ABSENT"):
                excluded.append({"model": m, "arm": arm, "status": r["status"], "reason": r["reason"]})
    backbones = sorted({r.get("backbone") for m in models for r in runs[m].values()
                        if r["status"] == "OK" and r.get("backbone")})
    if len(backbones) > 1:
        problems.append(f"runs of several backbones in one report: {backbones}")
    loras_ok = [m for m in models if runs[m]["like"]["status"] == "OK"]

    # ---- per-model vectors
    L, PI, PA, PB, NH, LP, joins = {}, {}, {}, {}, {}, {}, defaultdict(dict)
    consistency = {}
    for m in models:
        rl = runs[m]["like"]
        if rl["status"] == "OK":
            L[m], joins[m]["like"] = to_pairs(rl["sc"], cx, expected["like"])
        rs = runs[m]["swap"]
        if rs["status"] == "OK" and m in L:
            pos = rs["swap"]["pos"]
            pi, pa, pb = (np.full(cx.n, NAN) for _ in range(3))
            for j in np.flatnonzero(cx.sd_test):
                v = pos.get(cx.item[j])
                if v:
                    pi[j], pa[j], pb[j] = _fmean(v), _fmean(v[fx.HALF_A]), _fmean(v[fx.HALF_B])
            PI[m], PA[m], PB[m] = pi, pa, pb
            own, joins[m]["swap"] = to_pairs(rs["sc"], cx, expected["swap"])
            both = np.isfinite(own) & np.isfinite(L[m])
            consistency[m] = {"n_pairs": int(both.sum()),
                              "max_abs_diff": float(np.abs(own[both] - L[m][both]).max()) if both.any() else NAN,
                              "pearson": float(np.corrcoef(own[both], L[m][both])[0, 1]) if both.sum() > 2 else NAN}
        rn = runs[m]["nohist"]
        if rn["status"] == "OK":
            v, joins[m]["nohist"] = to_pairs(rn["sc"], cx, expected["nohist"])
            mp = fx._item_mean_map(v[cx.sd_test], cx.item[cx.sd_test])
            NH[m] = np.where(cx.sd_test, np.array([mp.get(i, NAN) for i in cx.item.tolist()], float), NAN)
        if all(runs[m][k]["status"] == "OK" for k in STARPERM_ARMS):
            LP[m] = []
            for k in STARPERM_ARMS:
                v, joins[m][k] = to_pairs(runs[m][k]["sc"], cx, expected[k])
                LP[m].append(v)
    present = set(L)

    # ---- CF references
    source = str(eval_rows[0].get("source") or a.domain) if eval_rows else a.domain
    ident = refs_identity(a.domain, panel_sha.get("eval.jsonl"), T, source, a.n_boot, a.seed)
    refs, refs_info = None, {}
    rp = Path(a.refs) if a.refs else None
    if rp is not None and rp.is_file():
        refs, why = load_refs(rp, ident, cx)
        refs_info = {"source": "cache (--refs)"} if refs is not None else {"cache_rejected": why}
    if refs is None and a.raw:
        try:
            refs = compute_refs(cx, a.raw, source, T, a.n_boot, a.seed)
            refs_info["source"] = "computed from --raw"
            if rp is not None:
                save_refs(rp, ident, cx, refs)
                refs_info["cache_written"] = True
        except (OSError, KeyError, ValueError) as e:
            refs_info["error"] = f"{type(e).__name__}: {e}"
    if refs is None:
        refs_info.setdefault("reason", "no CF references: give --raw (or a matching --refs cache)")

    # ---- endpoints
    boot = (a.n_boot, a.seed)
    res = {"E_A": {}, "E_C": {}, "E_D": {}, "E_E": {}}
    regimes = {}
    for reg in REGIMES:
        ok, miss = _regime(models, present, reg)
        regimes[reg] = {"models": ok, "missing_or_excluded": miss, "n_registered": len(REGIMES[reg])}
    for reg, info in regimes.items():
        ms, nreg = info["models"], info["n_registered"]
        if not ms:
            for blk in res:
                res[blk][reg] = _na(f"no usable like arm for the {reg} models {list(REGIMES[reg])}"
                                    if info["missing_or_excluded"] else f"{reg} models not requested")
            continue
        Lr = {m: L[m] for m in ms}
        hdr = {"models": ms, "missing_or_excluded": info["missing_or_excluded"], "complete": len(ms) == nreg}
        # E-A
        xr = ({"q_hat": refs["arrays"]["q_hat"], "popularity": cx.pop, "mf": refs["arrays"]["mf_score"],
               "mf_personal_residual": refs["arrays"]["mf_residual"]} if refs is not None else {"popularity": cx.pop})
        ea = {**hdr, "UAUC_TEST": uauc_models(Lr, cx.y, cx.uc, cx.test, *boot, extra=xr, n_registered=nreg)}
        if refs is not None:
            warm = refs["arrays"]["mf_warm"] > 0
            ea["mf_personal_residual_warm_pairs_only"] = uauc_models(
                {"mf_personal_residual": refs["arrays"]["mf_residual"]}, cx.y, cx.uc,
                cx.test & warm & np.all([np.isfinite(L[m]) for m in ms], 0), *boot)["per_model"]["mf_personal_residual"]
        else:
            ea["references_note"] = refs_info.get("reason") or refs_info.get("error")
        ea["secondary"] = {"UAUC_all_rows": uauc_models(Lr, cx.y, cx.uc, cx.test | cx.cal, *boot, n_registered=nreg)}
        if cx.seen is not None:
            ea["secondary"]["UAUC_TEST_seen"] = uauc_models(Lr, cx.y, cx.uc, cx.test & cx.seen, *boot, n_registered=nreg)
            ea["secondary"]["UAUC_TEST_unseen"] = uauc_models(Lr, cx.y, cx.uc, cx.test & ~cx.seen, *boot,
                                                              n_registered=nreg)
        else:
            ea["secondary"]["seen_unseen"] = _na("no train.jsonl in --panels")
        res["E_A"][reg] = ea
        # E-C
        ec = {**hdr, **ec_block(cx, Lr, *boot, nreg)}
        platt = ec.pop("platt")
        res["E_C"][reg] = ec
        # E-D
        ed = {**hdr}
        msw = [m for m in ms if m in PI]
        if not sd_ids:
            ed.update(_na("no S_d"))
        else:
            ed["swap_models"] = msw
            if msw:
                Q = {"pi": PI, "e_hat": {m: L[m] - PI[m] for m in msw}, "L": L}
                ed["UAUC"] = {k: uauc_models({m: Q[k][m] for m in msw}, cx.y, cx.uc, cx.sd_test, *boot,
                                             n_registered=nreg) for k in ("pi", "e_hat", "L")}
                ed["shares"] = shares_block(cx.uc, cx.sd_test, {m: (PA[m], PB[m], L[m], PI[m]) for m in msw}, *boot, nreg)
                ed["information_gain"] = (stacker_block(cx, msw, L, PI, refs["arrays"]["q_hat"],
                                                        refs["arrays"]["mf_residual"], cx.sd_test, *boot, nreg)
                                          if refs is not None else _na("no q-hat / MF residual (needs --raw or --refs)"))
                ed["swap_arm_like_consistency"] = {m: consistency.get(m) for m in msw}
            else:
                for k in ("shares", "information_gain"):
                    ed[k] = _na("no usable swap arm for the regime's models")
                ed["UAUC"] = {"L": uauc_models(Lr, cx.y, cx.uc, cx.sd_test, *boot, n_registered=nreg),
                              "pi": _na("no usable swap arm"), "e_hat": _na("no usable swap arm")}
            mnh = [m for m in ms if m in NH]
            ed["UAUC"]["L_nohist"] = (uauc_models({m: NH[m] for m in mnh}, cx.y, cx.uc, cx.sd_test, *boot,
                                                  n_registered=nreg) if mnh else _na("no usable nohist arm"))
            mlp = [m for m in ms if m in LP]
            ed["star_permutation"] = (starperm_block(cx, mlp, L, LP, cx.sd_test, *boot, nreg) if mlp
                                      else _na("no usable starperm0 and starperm1 arms"))
            ed["models_per_sub_block"] = {"swap": msw, "nohist": mnh, "star_permutation": mlp}
        res["E_D"][reg] = ed
        # E-E
        res["E_E"][reg] = {**hdr, **ee_block(cx, a.domain, ms, Lr, PI if all(m in PI for m in ms) else None, refs,
                                             platt, *boot, nreg)}

    # E-B
    ft = regimes["FT"]["models"]
    if "zeroshot" in present and ft:
        res["E_B"] = {"models": ft, "missing_or_excluded": regimes["FT"]["missing_or_excluded"],
                      "complete": len(ft) == 3, "holm_family": "E-B {dUAUC per Qwen domain} (A3 section 11)",
                      **contrast_models({s: (L[s], L["zeroshot"]) for s in ft}, cx.y, cx.uc, cx.test, *boot,
                                        n_registered=3)}
    else:
        res["E_B"] = _na("needs the zeroshot like arm and at least one FT seed")
    # P1
    bb = backbones[0] if len(backbones) == 1 else None
    is_qwen = bb is not None and "qwen" in bb.lower()
    role = ("P1 confirmatory (ML-1M, Qwen3-8B)" if a.domain == "ml1m" and is_qwen else
            "replication of P1's sign (Amendment 2 F)" if bb is not None else "backbone unknown")
    res["P1"] = p1_block(cx, L, PI, refs, {m for m in present if m in PI}, role, *boot)
    # FT-C
    ftc = {}
    for s in (0, 1):
        sm, pm_ = f"s{s}", f"p{s}"
        if sm in present and pm_ in present:
            xr = ({"q_hat": refs["arrays"]["q_hat"], "popularity": cx.pop, "mf": refs["arrays"]["mf_score"],
                   "mf_personal_residual": refs["arrays"]["mf_residual"]} if refs is not None else {"popularity": cx.pop})
            ftc[f"seed{s}"] = {
                "TEST": contrast_models({sm: (L[sm], L[pm_])}, cx.y, cx.uc, cx.test, *boot, n_registered=1),
                "all_rows": contrast_models({sm: (L[sm], L[pm_])}, cx.y, cx.uc, cx.test | cx.cal, *boot, n_registered=1),
                "references_TEST": uauc_models({sm: L[sm], pm_: L[pm_]}, cx.y, cx.uc, cx.test, *boot, extra=xr)}
        else:
            ftc[f"seed{s}"] = _na(f"needs usable like arms of {sm} and {pm_}")
    pairs = {f"s{s}": (L[f"s{s}"], L[f"p{s}"]) for s in (0, 1) if f"s{s}" in present and f"p{s}" in present}
    if pairs:
        ftc["mean_over_seeds_TEST"] = contrast_models(pairs, cx.y, cx.uc, cx.test, *boot, n_registered=2)
        ftc["mean_over_seeds_all_rows"] = contrast_models(pairs, cx.y, cx.uc, cx.test | cx.cal, *boot, n_registered=2)
    res["FT_C"] = {"definition": "UAUC(real FT seed s) - UAUC(within-item permuted adapter p_s) on identical users and "
                                 "rows, TEST and all rows (CAL u TEST), with the E-A references on the TEST rows; "
                                 "descriptive, in no Holm family (A3 section 9)", "descriptive": True, **ftc}
    # knockout
    res["knockout"] = knockout_block(cx, panels, eval_rows, runs, sdir, models, a.n_boot, a.seed, a.bi_boot)

    meta = {"domain": a.domain, "backbone": bb, "backbones_seen": backbones, "T": T, "variant": split.get("variant"),
            "quantile": split.get("quantile"), "split_sha1": fx.file_sha1(a.split),
            "split_counts": {k: split.get(k) for k in ("train", "eval", "sd")}, "split_args": split.get("args"),
            "gateft_T_match": split.get("gateft_T_match"), "split_code_sha1": split.get("code_sha1"),
            "panel_sha1": panel_sha, "n_boot": a.n_boot, "seed": a.seed, "min_n": MIN_N,
            "models_requested": models, "regimes": regimes,
            "n_eval_pairs": cx.n, "n_test_pairs": int(cx.test.sum()), "n_cal_pairs": int(cx.cal.sum()),
            "n_sd_users": len(sd_ids), "n_sd_test_pairs": int(cx.sd_test.sum()),
            "n_users_both_classes_test": len(user_aucs(np.zeros(cx.n), cx.y, cx.uc, cx.test)),
            "n_users_both_classes_test_tail": len(user_aucs(np.zeros(cx.n), cx.y, cx.uc, cx.test & cx.tail))}
    run_meta = {m: {arm: {k: v for k, v in r.items() if k not in ("sc", "swap")} for arm, r in runs[m].items()}
                for m in models}
    out = {"spec": SPEC, "meta": meta, "input_checks": {"problems": problems, **checks},
           "excluded_runs": excluded, "runs": run_meta, "joins": {m: dict(v) for m, v in joins.items()},
           "references": {**refs_info, **({"cf_block": refs["cf_block"]} if refs is not None else {})},
           "operationalizations": list(OPERATIONALIZATIONS), **res}
    out["adapters_distinct"] = len(loras_ok) == len(set(loras_ok))
    return out


# ---------------------------------------------------------------- tables
def table_rows(res: dict) -> list:
    d, bb = res["meta"]["domain"], res["meta"]["backbone"] or ""
    rows = []

    def add(block, regime, model, endpoint, r, note=""):
        if not isinstance(r, dict) or "est" not in r:
            return
        rows.append({"domain": d, "backbone": bb, "block": block, "regime": regime, "model": model, "endpoint": endpoint,
                     "est": r.get("est"), "lo": r.get("lo"), "hi": r.get("hi"), "p": r.get("p"),
                     "n_users": r.get("n_users"), "n_pairs": r.get("n_pairs"),
                     "descriptive_min_n": r.get("descriptive_min_n"), "note": note})

    def models_and_mean(block, regime, name, blk, key_mean="mean_over_seeds", key_per="per_model"):
        if not isinstance(blk, dict):
            return
        for m, r in (blk.get(key_per) or {}).items():
            add(block, regime, m, name, r)
        add(block, regime, "mean_over_seeds", name, blk.get(key_mean))

    for reg, ea in res["E_A"].items():
        if not ea.get("models"):
            continue
        models_and_mean("E-A", reg, "UAUC_TEST", ea["UAUC_TEST"])
        for x, r in (ea["UAUC_TEST"].get("references") or {}).items():
            add("E-A", reg, "reference", f"UAUC_TEST_{x}", r, "same users and rows as the regime's models")
        for k, blk in ea["secondary"].items():
            models_and_mean("E-A", reg, k, blk)
    eb = res["E_B"]
    if eb.get("per_seed"):
        for s, r in eb["per_seed"].items():
            add("E-B", "FT", s, "dUAUC_FT_minus_ZS", r)
        add("E-B", "FT", "mean_over_seeds", "dUAUC_FT_minus_ZS", eb["mean_over_seeds"],
            f"sigma_seed={eb['seeds']['sigma_seed']}")
    for reg, ec in res["E_C"].items():
        if not ec.get("models"):
            continue
        for m, blk in ec["per_model"].items():
            for k in EC_KEYS:
                add("E-C", reg, m, k, blk[k])
        for k in EC_KEYS:
            add("E-C", reg, "mean_over_seeds", k, ec["mean_over_seeds"].get(k))
    for reg, ed in res["E_D"].items():
        if not ed.get("models"):
            continue
        for k, blk in (ed.get("UAUC") or {}).items():
            models_and_mean("E-D", reg, f"UAUC_{k}", blk)
        sh = ed.get("shares") or {}
        for m, blk in (sh.get("per_model") or {}).items():
            for k in SHARE_KEYS:
                add("E-D", reg, m, k, blk[k], blk["shares_reading"])
        for k in SHARE_KEYS:
            add("E-D", reg, "mean_over_seeds", k, (sh.get("mean_over_seeds") or {}).get(k))
        ig = ed.get("information_gain") or {}
        for m, blk in (ig.get("per_model") or {}).items():
            add("E-D", reg, m, "G", blk["G"])
        add("E-D", reg, "mean_over_seeds", "G", ig.get("G_mean_over_seeds"))
        add("E-D", reg, "reference", "G_CF", ig.get("G_CF"))
        for m, r in (ig.get("G_over_G_CF") or {}).items():
            add("E-D", reg, m, "G_over_G_CF", r)
        sp = ed.get("star_permutation") or {}
        for m, blk in (sp.get("per_model") or {}).items():
            add("E-D", reg, m, "starperm_dUAUC", blk["dUAUC_L_minus_perm"],
                f"tau_sd_within_user={blk['tau_sd_within_user']}")
        add("E-D", reg, "mean_over_seeds", "starperm_dUAUC", sp.get("dUAUC_mean_over_seeds"))
    for reg, ee in res["E_E"].items():
        if not ee.get("models"):
            continue
        ps = ee.get("partial_spearman") or {}
        for k in ("L_pair", "L_item", "pi_item"):
            models_and_mean("E-E", reg, f"partial_spearman_{k}", ps.get(k))
        bi = ee.get("bias_index") or {}
        for m, blk in (bi.get("per_model") or {}).items():
            for k, r in blk.items():
                add("E-E", reg, m, f"bias_index_{k}", r)
        for k, r in (bi.get("mean_over_seeds") or {}).items():
            add("E-E", reg, "mean_over_seeds", f"bias_index_{k}", r)
        models_and_mean("E-E", reg, "UAUC_TEST_tail_exploratory", ee.get("tail_UAUC_exploratory"))
    p1 = res["P1"]
    if p1.get("available"):
        for s, r in p1["per_seed"].items():
            add("P1", "FT-ZS", s, "G_FT_minus_G_ZS", r)
        add("P1", "FT-ZS", "mean_over_seeds", "G_FT_minus_G_ZS", p1["mean_over_seeds"],
            f"{p1['decision']['verdict']} ({p1['role']})")
    for s in (0, 1):
        blk = res["FT_C"].get(f"seed{s}") or {}
        for k in ("TEST", "all_rows"):
            if isinstance(blk.get(k), dict):
                for m, r in blk[k]["per_seed"].items():
                    add("FT-C", "FT-PERM", m, f"dUAUC_real_minus_permuted_{k}", r, "descriptive")
    for k in ("TEST", "all_rows"):
        blk = res["FT_C"].get(f"mean_over_seeds_{k}")
        if isinstance(blk, dict):
            add("FT-C", "FT-PERM", "mean_over_seeds", f"dUAUC_real_minus_permuted_{k}", blk["mean_over_seeds"],
                "descriptive")
    ko = res["knockout"]
    for m, v in (ko.get("labels") or {}).items():
        rows.append({"domain": d, "backbone": bb, "block": "FT-K", "regime": "", "model": m, "endpoint": "label",
                     "est": None, "lo": None, "hi": None, "p": None, "n_users": v.get("n_users"),
                     "n_pairs": v.get("n_analysed"), "descriptive_min_n": None, "note": v.get("label")})
    return rows


def _cell(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return format(float(x), ".10g") if math.isfinite(float(x)) else ""
    return str(x)


def write_tables(path: Path, rows: list) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(CSV_COLS)
        for r in rows:
            w.writerow([_cell(r.get(c)) for c in CSV_COLS])
    tmp.replace(path)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True, help="panels/<d>/ftgrid_split.json")
    ap.add_argument("--panels", required=True, help="panels/<d>/")
    ap.add_argument("--scores_root", required=True, help="scores/ (or scores/<d>/)")
    ap.add_argument("--models", default="zeroshot,s0,s1,s2", help="comma list of " + ",".join(MODELS))
    ap.add_argument("--raw", default=None, help="raw data root (data/raw): the CF references")
    ap.add_argument("--out", required=True, help="report/<d>.json; the tables go to <stem>_tables.csv next to it")
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--refs", default=None, help="cache of the CF reference arrays (read if it matches, else written)")
    ap.add_argument("--bi_boot", type=int, default=KNOCKOUT_BI_BOOT,
                    help="pilot_pseudonym Bias Index bootstrap of the knockout (its CLI default)")
    a = ap.parse_args(argv)
    if a.n_boot < 0 or a.bi_boot < 0:
        ap.error("--n_boot and --bi_boot must be >= 0")
    return a


def main(argv=None) -> dict:
    a = parse_args(argv)
    res = strict_json(build(a))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(res, indent=2, allow_nan=False) + "\n"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    write_tables(out.with_name(out.stem + "_tables.csv"), table_rows(res))
    print(f"wrote {out} and {out.with_name(out.stem + '_tables.csv')}")
    p1 = res.get("P1", {}).get("decision", {})
    print(f"P1 ({res.get('P1', {}).get('role')}): {p1.get('verdict')}")
    return res


if __name__ == "__main__":
    main()
