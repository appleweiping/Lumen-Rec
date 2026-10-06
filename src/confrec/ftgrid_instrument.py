"""Amendment 3 addendum 14 (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_14.md, "A3-14"): the instrument validity of the information
gain G, for one panel and one root. EXPLORATORY and outcome-free: every number written here is descriptive, never a gate, a
hypothesis test or a family member, and no output uses the word "significant".

    python -m src.confrec.ftgrid_instrument build --domain toys --split outputs/confrec/ftgrid/panels/toys/ftgrid_split.json \
        --panels outputs/confrec/ftgrid/panels/toys --scores_root outputs/confrec/ftgrid/scores \
        --models zeroshot,s0,s1,s2 --raw data/raw --root_label main --out outputs/confrec/ftgrid_instr/main/toys.json \
        [--report outputs/confrec/ftgrid/report/toys.json] [--extra outputs/confrec/ftgrid/extra/toys.json] \
        [--pilot_log docs/sigir/PILOT_LOG.md] [--n_boot 2000] [--n_perm 500] [--n_rep 100] [--n_boot_power 500] \
        [--n_boot_refit 500] [--seed 0]
    python -m src.confrec.ftgrid_instrument check_resume <the build arguments>      # exit 0: --out is current, 1: rebuild
    python -m src.confrec.ftgrid_instrument holm   --reports <registered report json> ... --out <file>
    python -m src.confrec.ftgrid_instrument bounds --reports <report json> ... [--extras <extra json> ...] --out <file>
    python -m src.confrec.ftgrid_instrument record --print | --pilot_log docs/sigir/PILOT_LOG.md

`build` (one panel, one root; writes ONE json and ONE csv, resumable by a fingerprint of its inputs and code sha1 exactly as
ftgrid_extra does) computes, for every regime present (ZS = zeroshot, FT = s0-s2, the seed mean for FT) and for both the registered
pooled estimator of G (E-D: stackers [m, pi], [m, pi, e-hat] cross-fitted over stats.user_halves(S_d, 0)) and the within-user
estimator (E-W: user-centred features, K = 20 splits stats.user_halves(S_d, k)), exactly the registered code imported read-only
from ftgrid_report and ftgrid_extra (m = the shrunk prior item mean q-hat):

  permutation_null  (item 1) the pair-specific residual e-hat of every S_d user permuted among that user's TEST pairs, 500
                    permutations with fixed seeds: the mean and the 2.5th / 97.5th percentiles of the null G, the observed G,
                    the share of permutations with G at least the observed value, and the reading WITHIN_NULL / BELOW_NULL /
                    ABOVE_NULL.
  planted_signal    (item 2) l' = l + lambda sd(e-hat) c, c = rho y~ + sqrt(1 - rho^2) eps, rho = 0.3, lambda in {0.25, 0.5,
                    1, 2}, 100 replications; e-hat recomputed from l' and the registered prior; per lambda the mean recovered
                    G, the power (share of replications whose 95% user-bootstrap lower bound of G, 500 resamples, is above 0)
                    and the minimum detectable gain.
  donor_placebo     (item 3) the eighth donor held out: pi-hat_(-8), e-hat_7 = l - pi-hat_(-8), e-hat_pl = l_swap(row, donor 8) -
                    pi-hat_(-8); G_7, G_pl and the paired contrast G_7 - G_pl (2,000 user resamples, seed 0); the reading
                    PERSONAL / DENOISING_ONLY / INCONCLUSIVE.
  honest_intervals  (item 4 a, b) the full-refit bootstrap (the cross-fitted stackers refitted inside each of 500 user
                    resamples) for G of every regime and, on ML-1M with Qwen3-8B, G_wu and P1, beside the registered
                    fixed-coefficient interval; on ML-1M the two-way user-and-item resampling (pigeonhole weights).
  ratios            (item 5) for the FT-C retention R (main-root ML-1M) and the FT-Q retention R_Q (teacher root): the
                    numerator and denominator intervals, a Fieller interval for the ratio of the seed-mean differences from
                    the paired user resamples, whether the denominator's interval excludes 0, and UAUC(real) - UAUC(control).
`holm` (item 4 c) is the cross-panel Holm sensitivity computed from the stored p-values of the registered reports (the 8 E-D G
tests of four panels and two regimes, and all E-D tests); `bounds` (item 6) the upper 95% bound of G, G_wu and G_LLM|CF from the
stored reports and extra files and whether the whole 95% interval lies within [-0.01, +0.01]; `record` prints or checks the
pilot-log lines of this module, its runner and its test file. Every interval is a record {est, lo, hi, n_users, n_boot,
descriptive_min_n} (plus n_pairs, and p and ci_excludes_0 for a contrast), as the registered files; the JSON has no timestamp,
host or path and is byte-stable; `status` is "exploratory". The output must lie below outputs/confrec/ftgrid_instr/: a path under
a registered root (outputs/confrec/ftgrid*, ftmethod, ftprune, gatefix, gateft, nextitem_audit*) is refused (exit 2).

Readings. Where the addendum leaves a choice, the most literal reading is taken; the items below are copied into every output as
`readings` (parsed from this docstring):
R1  Status. Everything is exploratory and descriptive: no hypothesis, no family, no Holm correction inside a build (the Holm
    table of `holm` is a sensitivity), no direction word; a statement beyond one panel follows the claim-admission rule of
    addendum 6 section 9. Item 2 is a validity check of the estimator and never changes a registered number.
R2  Rows and references. A regime's rows are E-D's stacker rows: the S_d TEST rows whose like logit and swap prior pi are finite
    for every model of the regime (ftgrid_extra.ed_rows) and whose q-hat is finite. The MF residual that ftgrid_report.stacker_block
    also requires is finite for every pair (forensics.mf_pair_scores: a cold pair keeps residual 0), so these are the registered
    rows (a test reproduces the registered G on the fixture); the matrix factorisation is therefore not fitted here. q-hat is the
    array of ftgrid_report.compute_refs (forensics.prior_means of the raw scan at the candidate times, k = 5), read from --refs
    when its identity matches, else computed from --raw; a test asserts that both are equal.
R3  Permutation null. Within-user permutation of the rows of R (a uniform permutation of each user's rows, one random key per
    row sorted within the user; seed [seed, 1, k] for permutation k = 0..499); the same permutation is applied to the e-hat of
    every model of the regime (the seeds' residuals are strongly correlated, so the null of the seed mean permutes them jointly);
    pi and q-hat are not permuted; M1 does not depend on e-hat and is computed once; the null of the seed-mean G is the mean over the
    models' null G of one permutation. The percentiles are numpy's linear-interpolation quantiles (stats.percentile_ci); WITHIN_NULL
    includes the percentiles themselves. share_ge_observed = #{null G >= observed G} / 500 (no smoothing).
R4  Planted signal, y~. 'The within-user centred TEST label residual after removing the registered item mean': the label is centred
    within user, the user-centred q-hat is projected out with the pooled within-user least-squares slope b = sum(y_c q_c) /
    sum(q_c^2) (a label is 0/1 and q-hat is on the rating scale, so a literal difference y - q-hat would have no meaning) and the
    result is standardised to unit variance over the rows of R (divisor N); --ytilde_mode literal_diff centres y - q-hat instead.
R5  Planted signal, c, scale and noise. c = rho y~ + sqrt(1 - rho^2) eps with y~ of unit variance, so c has unit variance and
    correlation rho with y~; eps ~ N(0, 1) is drawn independently for every replication and pair (seed [seed, 2, r]) and is not
    centred; the replication r uses the same eps for every lambda, regime, estimator and model. The scale sd(e-hat) is the pooled
    within-user SD of the model's own e-hat = l - pi over the rows of R (ftgrid_report.pooled_within_sd, divisor N - n_users), because
    e-hat is the pair-specific residual and its user offset is not signal; the total SD is recorded beside it. l' = l + lambda
    sd(e-hat) c is formed from the like logit and e-hat' = l' - pi is recomputed from it.
R6  Power and minimum detectable gain. The 95% user-bootstrap lower bound is the 2.5th percentile (stats.percentile_ci) of 500
    resamples of the per-user seed-mean G (the draws of ftgrid_report.mean_draws, seed 0, the same for every replication);
    power = the share of the replications whose lower bound is above 0; the recovered G includes the panel's own observed G;
    the minimum detectable gain is the smallest mean recovered G among the lambdas whose power is at least 0.8; at_grid_floor
    says that this lambda is the smallest of the grid (the gain is then an upper bound of the minimum detectable gain) and the
    value is null when no lambda reaches 0.8.
R7  Donor placebo, rows and donors. A swap_prior.csv.gz item holds its donor logits in file order; donor 8 is the eighth row of the
    item (a censored donor keeps its position as NaN), pi-hat_(-8) = ftgrid_report._fmean of the first seven (the finite ones), the
    held-out confidence is the item's eighth donor logit (every row of the item shares it, as the registered donors are per item);
    an item without exactly eight donor rows is excluded and counted. The rows are the regime's rows with finite pi-hat_(-8) and
    finite donor 8 for every model. e-hat_7 uses the like logit of the like arm. 'Centred within user like e-hat': the E-D
    stackers use e-hat_pl as the registered E-D uses e-hat (uncentred); E-W centres every feature within user, so both estimators
    treat e-hat_7 and e-hat_pl identically.
R8  Reading of the personal part. Checked in the order written: PERSONAL if the interval of G_7 - G_pl lies above 0 (lo > 0); else
    DENOISING_ONLY if the interval of G_7 contains 0 (lo <= 0 <= hi) or the ESTIMATE of G_7 - G_pl lies within [-0.005, +0.005];
    else INCONCLUSIVE. The interval-within-band alternative is recorded beside it (contrast_ci_within_band).
R9  Full-refit bootstrap. The units are the users of R; resample b draws rng.integers(0, n_users, n_users) with
    default_rng(seed) (the draws of the registered fixed-coefficient bootstrap when every user of R has both classes); a drawn
    user enters the fits with its multiplicity (rows duplicated, ftgrid_report.logit_fit unchanged), keeps the fold of its
    stats.user_halves hash (E-W: the fold of each of its 20 splits), and the statistic of the resample is the multiplicity-
    weighted mean over the drawn users with both classes of the per-user AUC difference of the stackers refitted on the
    resample's rows (cross-fitted: each fold scored by the fit on the other fold's resampled rows). The interval is the percentile
    interval of the 500 resamples around the registered point estimate; P1's refit contrast is the mean over seeds of G_FT - G_ZS
    on P1's rows.
R10 Two-way resampling. Pigeonhole weights (Owen 2007): users and items are resampled independently (rng.integers on each, the user
    draws as in R9), a pair (u, i) carries the weight W_u W_i; the per-user AUC is the weighted Mann-Whitney AUC (weights W_i, ties
    1/2; users without a weighted positive and negative are excluded); the UAUC functional is therefore computable and no item-block
    bootstrap is needed. Two variants: fixed coefficients (the registered predictions) and full refit (the stackers refitted on
    the rows with weights W_u W_i; E-W's features re-centred with the weighted user mean).
R11 Ratios. The numerator and denominator are the seed-mean differences mean_s [UAUC(p_s) - UAUC(ZS)] and mean_s [UAUC(s) - UAUC(ZS)],
    s = 0, 1, on identical TEST users and rows (every model finite; the registered retention rows), from the per-user AUCs of the
    stored scores through ftgrid_report.user_aucs; the Fieller interval is for N/D (the ratio of the seed means, NOT the registered
    mean of the two per-seed ratios, which is recomputed beside it with ftgrid_extra.retention) with z = 1.959964 and the covariance
    of (N, D) estimated from the 2,000 paired user resamples of ftgrid_report.mean_draws (the analytic per-user covariance is
    recorded beside it); when the denominator is not distinguishable from 0 at that level the set is unbounded and is reported with
    lo = hi = null and its form (complement of an interval, half line, whole line), never as a finite interval.
R12 Holm sensitivity. The p-values are those stored in each report's E_D/<regime>/holm_family_E_D/p (G and star_permutation, the
    seed-mean tests); ftgrid_report.holm is applied (a member without a p-value is outside the family) over the G tests of the
    reports of one backbone (the 8 G tests of the four Qwen panels) and over all E-D tests (G and star permutation, 16), beside
    the registered per-panel-and-regime family-wise p-values; adj_below_0.05 only compares an adjusted p-value with 0.05.
R13 Bounds. The upper 95% bound is the stored hi of the seed-mean record; the whole interval lies within the band when
    -0.01 <= lo and hi <= 0.01 (inclusive).
R14 Seeds and counts. Fixed seeds: the registered 2,000 resamples (seed 0) for the fixed-coefficient intervals, the placebo
    contrast and the ratios, 500 permutations, 100 replications with 500 resamples each and 500 refit resamples; a build whose
    counts or grid differ from these (a rehearsal) says so in meta.registered_counts.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np

from src.confrec import forensics as fx
from src.confrec import ftgrid_extra as fe
from src.confrec import ftgrid_report as fr
from src.confrec.stats import percentile_ci, strict_json

NAN = float("nan")
STATUS = "exploratory"
SPEC = ("idea-stage/PREREG_AMENDMENT_3_ADDENDUM_14.md sections 1-4 (exploratory, outcome-free), with the estimators of "
        "idea-stage/PREREG_AMENDMENT_3.md section 3 (E-D), addendum 6 item 2 (E-W) and addendum 8 (FT-Q)")
SEED = fr.SEED                       # fixed seeds: the registered seed 0
N_BOOT = fr.N_BOOT                   # the registered 2,000 user resamples: fixed-coefficient intervals, placebo, ratios
N_PERM = 500                         # A3-14 item 1
N_REP = 100                          # A3-14 item 2
N_BOOT_POWER = 500                   # A3-14 item 2: the bootstrap of each replication
N_BOOT_REFIT = 500                   # A3-14 item 4
RHO = 0.3                            # A3-14 item 2
LAMBDAS = (0.25, 0.5, 1.0, 2.0)      # A3-14 item 2
POWER_TARGET = 0.8                   # A3-14 item 2: the minimum detectable gain has power at least 0.8
BAND_BOUNDS = 0.01                   # A3-14 item 6 and section 3
BAND_PLACEBO = 0.005                 # A3-14 section 3
K_DONORS = 8                         # the swap arm's donors per item (A3 section 3)
K_SPLITS = fe.K_SPLITS               # E-W: 20 user-half splits
ESTIMATORS = ("E_D", "E_W")
REGS = ("ZS", "FT")
STREAM_PERM, STREAM_EPS, STREAM_ITEMS = 1, 2, 3          # SeedSequence stream ids of the three random streams
YTILDE_MODES = ("fe_residual", "literal_diff")
ROOT_LABELS = ("main", "llama", "teacher")
INSTR_FILES = ("src/confrec/ftgrid_instrument.py", "scripts/sigir/run_ftinstr.sh", "tests/test_confrec_ftinstr.py")
CODE_FILES = ("src/confrec/ftgrid_instrument.py", "src/confrec/ftgrid_extra.py", "src/confrec/ftgrid_report.py",
              "src/confrec/forensics.py", "src/confrec/stats.py", "src/confrec/metrics.py")
OUT_DIRNAME = "ftgrid_instr"                              # the only directory the module writes below
REGISTERED_PREFIXES = ("ftgrid", "ftmethod", "ftprune", "gatefix", "gateft", "nextitem_audit")   # outputs/confrec/<prefix>*
CSV_COLS = ("block", "panel", "root", "regime", "estimator", "statistic", "model", "setting", "est", "lo", "hi", "n_users",
            "n_boot", "descriptive_min_n", "p", "reading", "note")
FTC_MODELS = fe.FTC_MODELS                                # zeroshot, s0, s1, p0, p1
Z95 = NormalDist().inv_cdf(0.975)
NULL_RULE = ("WITHIN_NULL if G lies inside the 2.5th-97.5th percentile range of its permutation null (inclusive), BELOW_NULL if "
             "below and ABOVE_NULL if above (A3-14 section 3)")
PERSONAL_RULE = ("PERSONAL if the interval of G_7 - G_pl lies above 0; DENOISING_ONLY if the interval of G_7 contains 0 or the "
                 "estimate of G_7 - G_pl lies within [-0.005, +0.005]; otherwise INCONCLUSIVE (checked in this order; A3-14 "
                 "section 3, reading R8)")
READINGS = fe._readings(__doc__)


# ---------------------------------------------------------------- small helpers
def _na(reason: str) -> dict:
    return {"available": False, "reason": reason}


def _fin(x) -> bool:
    return fr._fin(x)


def _num(x) -> float:
    return fr._num(x)


def _rec(est, lo, hi, n_users: int, n_boot: int, n_pairs=None, **extra) -> dict:
    """An interval record {est, lo, hi, n_users, [n_pairs,] n_boot, descriptive_min_n} (the registered shape)."""
    out = {"est": _num(est), "lo": _num(lo), "hi": _num(hi), "n_users": int(n_users)}
    if n_pairs is not None:
        out["n_pairs"] = int(n_pairs)
    out["n_boot"] = int(n_boot)
    out["descriptive_min_n"] = int(n_users) < fr.MIN_N
    out.update(extra)
    return out


def _lam_key(lam: float) -> str:
    return format(float(lam), "g")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canon(p) -> Path:
    return fe._canon(p)


def _sha1_file(path) -> str | None:
    return fe._sha1_file(Path(path))


def code_sha1() -> dict:
    """sha1 of this module and of the bound modules it computes with."""
    root = repo_root()
    return {Path(rel).name: _sha1_file(root / rel) for rel in CODE_FILES}


# ---------------------------------------------------------------- the output guard
def instr_root() -> Path:
    return repo_root() / "outputs" / "confrec" / OUT_DIRNAME


def under_registered(p) -> bool:
    """True iff the canonical path p (symlinks resolved) lies under a registered output root: outputs/confrec/<prefix>* for the
    prefixes of REGISTERED_PREFIXES, except outputs/confrec/ftgrid_instr itself."""
    conf = _canon(repo_root() / "outputs" / "confrec")
    c = _canon(p)
    if c != conf and conf not in c.parents:
        return False
    rel = c.relative_to(conf).parts
    if not rel or rel[0] == OUT_DIRNAME:
        return False
    return any(rel[0].startswith(pref) for pref in REGISTERED_PREFIXES)


def check_out_path(out, out_root=None) -> Path:
    """The only way an output path enters the module: --out must lie below out_root (default outputs/confrec/ftgrid_instr),
    and neither may lie under a registered root, in any spelling and through any link (canonical paths); refused with exit 2."""
    root = Path(out_root) if out_root else instr_root()
    why = None
    if under_registered(root):
        why = f"the output root {root} is a registered root"
    elif under_registered(out):
        why = f"--out {out} lies under a registered root (outputs/confrec/ftgrid*, ftmethod, ftprune, gatefix, gateft, nextitem_audit*)"
    else:
        rc, oc = _canon(root), _canon(out)
        if not (oc == rc or rc in oc.parents):
            why = f"--out {out} does not lie below {root} (this module writes only below outputs/confrec/{OUT_DIRNAME}/)"
    if why:
        print(f"refused: {why}", file=sys.stderr)
        raise SystemExit(2)
    return Path(out)


# ---------------------------------------------------------------- the pilot-log record of this module's own files
def record_lines(files=None, root=None) -> list:
    """`path = sha1` lines of the module, its runner and its test file (ftmethod_shift_diag's format)."""
    root = Path(root) if root else repo_root()
    out = []
    for rel in (files or INSTR_FILES):
        try:
            out.append(f"{rel} = {fx.file_sha1(root / rel)}")
        except OSError as e:
            raise SystemExit(f"cannot read {rel} for the record: {e}") from None
    return out


def record_status(pilot_log, files=None, root=None) -> dict:
    """Which of the files have their sha1 in the pilot log (case-insensitive substring test, as ftgrid_freeze)."""
    root = Path(root) if root else repo_root()
    files = list(files or INSTR_FILES)
    sha = {rel: _sha1_file(root / rel) for rel in files}
    if not pilot_log:
        return {"pilot_log_given": False, "sha1": sha, "recorded": {rel: False for rel in files}, "complete": False}
    try:
        log = Path(pilot_log).read_text(encoding="utf-8").lower()
    except OSError:
        log = None
    rec = {rel: bool(log is not None and sha[rel] and sha[rel].lower() in log) for rel in files}
    return {"pilot_log_given": True, "pilot_log_readable": log is not None, "sha1": sha, "recorded": rec,
            "complete": all(rec.values())}


# ---------------------------------------------------------------- resume: the fingerprint of every input
def _source_of(panels: Path, domain: str) -> str:
    return fe._source_of(panels, domain)


def _run_arms(a) -> tuple:
    return ("like",) if a.root_label == "teacher" else ("like", "swap")


def input_fingerprint(a) -> dict:
    """Everything a build reads, by content: the settings, the code sha1, the split, the panels, every file of every scoring run
    directory read, the raw files, the reference cache, the stored files put beside and the record status. A stored build is
    resumed only when its fingerprint equals the current one (check_resume)."""
    split, panels = Path(a.split), Path(a.panels)
    models = [m for m in a.models.split(",") if m]
    arms = _run_arms(a)
    try:
        sd_name = (fr.read_json(split).get("sd") or {}).get("user_ids_path") or "sd_users.txt"
    except (OSError, ValueError, AttributeError):
        sd_name = "sd_users.txt"
    names = sorted({fr.ARM_PANEL[x] for x in arms} | {"sd_users.txt", "eval_users.txt", sd_name})
    sdir = Path(a.scores_root) / a.domain
    sdir = sdir if sdir.is_dir() else Path(a.scores_root)
    runs = {}
    for m in models:
        for arm in arms:
            d = sdir / m / arm
            runs[f"{m}/{arm}"] = ({p.name: fx.file_sha1(p) for p in sorted(d.iterdir()) if p.is_file()}
                                  if d.is_dir() else None)
    source = _source_of(panels, a.domain)
    raw = {}
    if a.raw:
        for p in fe.raw_files(a.raw, source):
            try:
                key = p.relative_to(Path(a.raw)).as_posix()
            except ValueError:
                key = p.name
            raw[key] = _sha1_file(p)
    rec = record_status(getattr(a, "pilot_log", None))
    return {"domain": a.domain, "root_label": a.root_label, "models": models, "settings": settings_of(a),
            "code_sha1": code_sha1(), "split_sha1": _sha1_file(split),
            "panels": {n: _sha1_file(panels / n) for n in names}, "runs": runs, "raw_source": source, "raw": raw,
            "refs_cache_sha1": _sha1_file(Path(a.refs)) if getattr(a, "refs", None) else None,
            "stored": {"report": _sha1_file(Path(a.report)) if getattr(a, "report", None) else None,
                       "extra": _sha1_file(Path(a.extra)) if getattr(a, "extra", None) else None},
            "record": {"recorded": rec["recorded"], "complete": rec["complete"], "sha1": rec["sha1"]}}


def settings_of(a) -> dict:
    return {"n_boot": int(a.n_boot), "n_perm": int(a.n_perm), "n_rep": int(a.n_rep),
            "n_boot_power": int(a.n_boot_power), "n_boot_refit": int(a.n_boot_refit), "seed": int(a.seed),
            "lambdas": [float(x) for x in a.lambdas], "rho": float(a.rho), "ytilde_mode": a.ytilde_mode,
            "k_splits": int(a.k_splits)}


def registered_counts(a) -> dict:
    """Whether the counts and the grid of this build are the registered ones of A3-14 (a rehearsal says so)."""
    s = settings_of(a)
    want = {"n_boot": N_BOOT, "n_perm": N_PERM, "n_rep": N_REP, "n_boot_power": N_BOOT_POWER, "n_boot_refit": N_BOOT_REFIT,
            "seed": SEED, "lambdas": list(LAMBDAS), "rho": RHO, "ytilde_mode": "fe_residual", "k_splits": K_SPLITS}
    diff = {k: s[k] for k in want if s[k] != want[k]}
    return {"registered": not diff, "differs": diff, "registered_values": {k: want[k] for k in sorted(want)}}


# ---------------------------------------------------------------- inputs
def qhat_from_raw(cx, raw, source: str) -> np.ndarray:
    """q-hat of ftgrid_report.compute_refs (forensics.cf_references: scan_events + prior_means at the candidate times, k = 5)
    without the matrix factorisation: the registered arrays' q_hat, bit for bit (a test), at a fraction of the memory."""
    events = fx.load_raw_events(raw, source)
    users, items = cx.E["user"].tolist(), cx.E["item"].tolist()
    scan = fx.scan_events(events, set(items), set(zip(users, items)))
    del events
    times = cx.ts
    if not (len(times) and np.isfinite(times).all()):
        times = np.array([scan["own"].get((u, i), (NAN,))[0] for u, i in zip(users, items)], float)
    pm = fx.prior_means(scan, users, items, times, fr.SHRINK_K)
    q = np.asarray(pm["mean_prior_shrunk"], float)
    del scan
    gc.collect()
    return q


def donor_split(pos: dict) -> tuple[dict, dict, int]:
    """{item: pi-hat_(-8)} and {item: donor-8 logit} from the donor logits of swap_prior.csv.gz in file order (a censored donor is
    NaN at its position); pi-hat_(-8) = the mean of the finite ones among donors 1-7 (ftgrid_report._fmean, the registered pi's
    mean), the held-out donor is the eighth row; an item without exactly K_DONORS rows is excluded (counted)."""
    pi7, d8, bad = {}, {}, 0
    for i, v in pos.items():
        if len(v) != K_DONORS:
            bad += 1
            continue
        pi7[i] = fr._fmean(v[:K_DONORS - 1])
        d8[i] = float(v[K_DONORS - 1])
    return pi7, d8, bad


def load_inputs(a) -> SimpleNamespace:
    """The inputs in the order of ftgrid_extra.load_inputs (the registered rows, runs and integrity rules), restricted to the arms
    this module reads (like; swap for the regime blocks), with the donor logits kept for the placebo and q-hat computed without
    the matrix factorisation."""
    fe.check_root_label(a)
    split = fr.read_json(a.split)
    T = float(split["T"])
    panels = Path(a.panels)
    models = [m for m in a.models.split(",") if m]
    unknown = [m for m in models if m not in fr.MODELS]
    if unknown:
        raise SystemExit(f"--models: unknown {unknown} (choose from {list(fr.MODELS)})")
    if len(set(models)) != len(models):
        raise SystemExit("--models repeats a model")
    teacher = a.root_label == "teacher"
    arms = _run_arms(a)
    eval_path = panels / fr.ARM_PANEL["like"]
    if not eval_path.is_file():
        raise SystemExit("--panels: no eval.jsonl")
    eval_rows = fx.read_jsonl(eval_path)
    files = split.get("files") or {}
    problems, checks, panel_sha = [], {}, {}
    for name in sorted({fr.ARM_PANEL[x] for x in arms} | {"sd_users.txt", "eval_users.txt"}):
        p = panels / name
        panel_sha[name] = fx.file_sha1(p) if p.is_file() else None
        if name in files and panel_sha[name] is not None and files[name] != panel_sha[name]:
            problems.append(f"{name}: sha1 {panel_sha[name]} differs from ftgrid_split.json files ({files[name]})")
    eval_ids = [str(r["user_id"]) for r in eval_rows]
    checks["eval_user_ids_sha1_match"] = (None if not (split.get("eval") or {}).get("user_ids_sha1")
                                          else fr.sha1_text_lines(eval_ids) == split["eval"]["user_ids_sha1"])
    sd_path = panels / ((split.get("sd") or {}).get("user_ids_path") or "sd_users.txt")
    sd_test_path = panels / fr.ARM_PANEL["swap"]
    if sd_path.is_file():
        sd_ids = [x for x in sd_path.read_bytes().decode("utf-8").split("\n") if x]
        checks["sd_user_ids_sha1_match"] = (None if not (split.get("sd") or {}).get("user_ids_sha1")
                                            else fx.file_sha1(sd_path) == split["sd"]["user_ids_sha1"])
    elif sd_test_path.is_file() and not teacher:
        sd_ids = [str(r["user_id"]) for r in fx.read_jsonl(sd_test_path)]
        checks["sd_user_ids_source"] = "eval_sd_test.jsonl (sd_users.txt absent)"
    else:
        sd_ids = []
        if not teacher:
            problems.append("no sd_users.txt and no eval_sd_test.jsonl: the S_d blocks cannot run")
    cx = fr.make_ctx(eval_rows, T, sd_ids, None)
    if cx.n_duplicate_pairs:
        problems.append(f"eval.jsonl repeats {cx.n_duplicate_pairs} (user_id, item_id) pairs")
    if (~(cx.test | cx.cal)).any():
        problems.append(f"{int((~(cx.test | cx.cal)).sum())} EVAL pairs without a finite candidate timestamp")
    if sd_test_path.is_file() and not teacher:
        sdp = {(str(r["user_id"]), str(i)) for r in fx.read_jsonl(sd_test_path) for i in r["candidate_item_ids"]}
        mine = {(u, i) for u, i, s in zip(cx.users.tolist(), cx.item.tolist(), cx.sd_test.tolist()) if s}
        checks["eval_sd_test_pairs_equal_sd_test_rows"] = sdp == mine
        if sdp != mine:
            problems.append("eval_sd_test.jsonl pairs differ from S_d's TEST rows of eval.jsonl")
    if not teacher:
        both = set(fr.user_aucs(np.zeros(cx.n), cx.y, cx.users, cx.test))
        checks["sd_users_are_eval_users_with_both_classes_in_test"] = all(u in both for u in sd_ids)
        if not checks["sd_users_are_eval_users_with_both_classes_in_test"]:
            problems.append("an S_d user is not an EVAL user with both classes among the TEST rows")

    # ---- runs (E1 integrity and the identity checks of ftgrid_report.load_run)
    sdir = Path(a.scores_root) / a.domain
    sdir = sdir if sdir.is_dir() else Path(a.scores_root)
    expected = {"like": np.ones(cx.n, bool), "swap": cx.sd_test}
    runs, excluded = {}, []
    for m in models:
        runs[m] = {}
        for arm in arms:
            r = fr.load_run(sdir / m / arm, f"{m}/{arm}", panel_sha.get(fr.ARM_PANEL[arm]), m, split.get("variant"),
                            need_swap=(arm == "swap"))
            runs[m][arm] = r
            if r["status"] not in ("OK", "ABSENT"):
                excluded.append({"model": m, "arm": arm, "status": r["status"], "reason": r["reason"]})
    backbones = sorted({r.get("backbone") for m in models for r in runs[m].values()
                        if r["status"] == "OK" and r.get("backbone")})
    if len(backbones) > 1:
        problems.append(f"runs of several backbones in one file: {backbones}")
    adapter_of = {}
    for m in models:
        ids = {r.get("adapter_id") for r in runs[m].values() if r["status"] == "OK"}
        if len(ids) > 1:
            problems.append(f"{m}: its arms were scored with different adapters (report.json lora)")
        adapter_of[m] = sorted(i for i in ids if i)
    ids_all = [i for v in adapter_of.values() for i in v]
    checks["adapters_distinct_across_models"] = len(ids_all) == len(set(ids_all))
    if not checks["adapters_distinct_across_models"]:
        problems.append("two models were scored with the same adapter (report.json lora)")

    # ---- per-model vectors (ftgrid_report.build's code), plus the donor logits of the swap arm
    L, PI, DON, consistency, joins = {}, {}, {}, {}, {}
    n_items_bad_donors = {}
    for m in models:
        joins[m] = {}
        rl = runs[m]["like"]
        if rl["status"] == "OK":
            L[m], joins[m]["like"] = fr.to_pairs(rl["sc"], cx, expected["like"])
        rs = runs[m].get("swap")
        if rs is not None and rs["status"] == "OK" and m in L:
            pos = rs["swap"]["pos"]
            pi_item = {i: fr._fmean(v) for i, v in pos.items() if v}
            PI[m] = np.where(cx.sd_test, np.array([pi_item.get(i, NAN) for i in cx.item.tolist()], float), NAN)
            DON[m] = {i: list(v) for i, v in pos.items()}
            own, joins[m]["swap"] = fr.to_pairs(rs["sc"], cx, expected["swap"])
            ok = np.isfinite(own) & np.isfinite(L[m])
            consistency[m] = {"n_pairs": int(ok.sum()),
                              "max_abs_diff": float(np.abs(own[ok] - L[m][ok]).max()) if ok.any() else NAN}
            n_items_bad_donors[m] = sum(len(v) != K_DONORS for v in pos.values())
    present = set(L)
    for m in models:
        for r in runs[m].values():
            r.pop("sc", None)
            r.pop("swap", None)

    # ---- q-hat: the reference cache when its identity matches, else the raw scan
    source = str(eval_rows[0].get("source") or a.domain) if eval_rows else a.domain
    q, refs_info = None, {}
    if getattr(a, "refs", None) and Path(a.refs).is_file():
        ident = fr.refs_identity(a.domain, panel_sha.get("eval.jsonl"), T, source, a.n_boot, a.seed)
        refs, why = fr.load_refs(Path(a.refs), ident, cx)
        if refs is not None and "q_hat" in refs["arrays"]:
            q, refs_info = np.asarray(refs["arrays"]["q_hat"], float), {"source": "cache (--refs)"}
        else:
            refs_info = {"cache_rejected": why or "the cache lacks q_hat"}
    if q is None and a.raw and not teacher:
        try:
            q = qhat_from_raw(cx, a.raw, source)
            refs_info["source"] = "computed from --raw (forensics.prior_means at the candidate times, k = 5; no MF)"
        except (OSError, KeyError, ValueError) as e:
            refs_info["error"] = f"{type(e).__name__}: {e}"
    if q is None:
        refs_info.setdefault("reason", "no q-hat: give --raw (or a matching --refs cache)")
        if a.raw and not teacher:
            problems.append("q-hat could not be computed although --raw is given: "
                            f"{refs_info.get('error') or refs_info.get('cache_rejected') or refs_info.get('reason')}")
    bb = backbones[0] if len(backbones) == 1 else None
    fam = fe._family_of(bb)
    if bb is None or fam is None:
        problems.append(f"backbone missing or unknown ({backbones}): neither Qwen nor Llama")
    elif (a.root_label == "llama") != (fam == "llama"):
        problems.append(f"backbone {bb} does not belong to the {a.root_label} root")
    folds_full = fe.split_folds(cx, int(a.k_splits)) if sd_ids else []
    return SimpleNamespace(a=a, split=split, T=T, models=models, cx=cx, sd_ids=list(sd_ids), runs=runs, excluded=excluded,
                           problems=problems, checks=checks, panel_sha=panel_sha, L=L, PI=PI, DON=DON,
                           consistency=consistency, joins=joins, present=present, q=q, refs_info=refs_info, backbone=bb,
                           backbones=backbones, root_label=a.root_label, domain=a.domain, folds_full=folds_full,
                           n_items_bad_donors=n_items_bad_donors, n_boot=a.n_boot, seed=a.seed)


# ---------------------------------------------------------------- the design of a regime
def make_design(X, rows, models: list, n_registered: int, k_splits: int | None = None) -> SimpleNamespace:
    """The rows `rows` (a mask over the EVAL pairs) of the models as arrays compressed to those rows: users and items as codes
    (np.unique order, so a user's code order is the registered key order), labels, E-D's fold (stats.user_halves(S_d, 0)), E-W's
    splits, q-hat, the like logits, the swap prior and e-hat = l - pi of each model."""
    cx = X.cx
    idx = np.flatnonzero(rows)
    ustr, uc = np.unique(cx.users[idx], return_inverse=True)
    istr, ic = np.unique(cx.item[idx], return_inverse=True)
    n = len(idx)
    k = len(X.folds_full) if k_splits is None else k_splits
    D = SimpleNamespace(idx=idx, n=n, users=ustr, uc=uc.reshape(-1), nu=len(ustr), items=istr, ic=ic.reshape(-1),
                        ni=len(istr), y=cx.y[idx].astype(int), fa=cx.fold_a[idx], q=np.asarray(X.q, float)[idx],
                        folds=[f[idx] for f in X.folds_full[:k]], models=list(models), n_registered=n_registered,
                        rows_all=np.ones(n, bool), L={m: X.L[m][idx] for m in models},
                        PI={m: X.PI[m][idx] for m in models})
    D.E = {m: D.L[m] - D.PI[m] for m in models}
    D.cxs = SimpleNamespace(uc=D.uc, y=D.y)
    D.pairs = None
    return D


def auc_vec(score, y, uc, nu: int) -> np.ndarray:
    """(nu,) per-user AUC (ties 1/2, ftgrid_report.user_aucs) of the users with both classes among their rows with a finite
    score; NaN elsewhere."""
    d = fr.user_aucs(score, y, uc)
    out = np.full(nu, NAN)
    if d:
        out[np.fromiter(d.keys(), int, len(d))] = np.fromiter(d.values(), float, len(d))
    return out


def stack_eta(D, est: str, feats: dict, cols: tuple) -> list:
    """The cross-fitted linear predictors of the stacker [cols]: one array for E-D (ftgrid_report.crossfit over E-D's fold), one
    per split for E-W (ftgrid_extra.wu_fit: user-centred features, K splits)."""
    if est == "E_D":
        return [fr.crossfit(np.column_stack([feats[c] for c in cols]), D.y, D.fa, D.rows_all)]
    return fe.wu_fit(D.cxs, feats, {"S": tuple(cols)}, D.rows_all, D.folds)["S"]


def auc_matrix(D, etas: list) -> np.ndarray:
    """(nu, len(etas)) per-user AUC of each predictor."""
    return np.column_stack([auc_vec(e, D.y, D.uc, D.nu) for e in etas])


def stack_auc(D, est: str, feats: dict, cols: tuple) -> np.ndarray:
    return auc_matrix(D, stack_eta(D, est, feats, cols))


COLS1, COLS2 = ("q_hat", "pi"), ("q_hat", "pi", "e_hat")


def base_auc(D, est: str) -> dict:
    """{model: (nu, K) AUC matrix of M1 = [q-hat, pi]}: M1 does not depend on e-hat, so it is computed once."""
    return {m: stack_auc(D, est, {"q_hat": D.q, "pi": D.PI[m]}, COLS1) for m in D.models}


def g_matrix(D, est: str, E: dict | None = None, base: dict | None = None, PI: dict | None = None) -> np.ndarray:
    """(n_models, nu) per-user G = mean over the splits of AUC(M2) - AUC(M1) of every model (NaN where an AUC is undefined);
    E replaces the models' e-hat (the permuted or planted residual), PI the models' prior (the placebo's pi-hat_(-8))."""
    E = D.E if E is None else E
    PIm = D.PI if PI is None else PI
    out = np.full((len(D.models), D.nu), NAN)
    for j, m in enumerate(D.models):
        a1 = base[m] if base is not None else stack_auc(D, est, {"q_hat": D.q, "pi": PIm[m]}, COLS1)
        a2 = stack_auc(D, est, {"q_hat": D.q, "pi": PIm[m], "e_hat": E[m]}, COLS2)
        out[j] = (a2 - a1).mean(1)
    return out


def user_keys(G: np.ndarray) -> np.ndarray:
    """The users with a finite G for every model (the registered `keys`: both classes among their rows)."""
    return np.flatnonzero(np.isfinite(G).all(0)) if G.size else np.zeros(0, int)


def n_pairs_of(D, keys) -> int:
    return int(np.isin(D.uc, keys).sum())


def seed_recs(D, G: np.ndarray, n_boot: int, seed: int, contrast: bool = True) -> dict:
    """The registered records of G: per model and the mean over seeds, on the users with a defined G for every model, with the
    paired user-bootstrap draws of ftgrid_report.mean_draws (stacker_block's computation)."""
    keys = user_keys(G)
    nu, npairs = len(keys), n_pairs_of(D, keys)
    V = G[:, keys].T if nu else np.zeros((0, len(D.models)))
    Dr = fr.mean_draws(V, n_boot, seed)
    k = len(D.models)
    per = {m: fr.rec(V[:, j].mean() if nu else NAN, Dr[:, j], nu, npairs, contrast=contrast) for j, m in enumerate(D.models)}
    mean = fr.rec(V[:, :k].mean(1).mean() if nu and k else NAN, Dr[:, :k].mean(1) if k else None, nu, npairs,
                  contrast=contrast)
    return {"per_model": per, "mean_over_seeds": mean,
            "seeds": fr.seed_summary([per[m]["est"] for m in D.models], D.n_registered, contrast, mean["est"]),
            "keys": keys}


def percentile_range(x, lo: float = 2.5, hi: float = 97.5) -> tuple:
    s = np.asarray(x, float)
    s = s[np.isfinite(s)]
    if not len(s):
        return NAN, NAN
    return float(np.quantile(s, lo / 100)), float(np.quantile(s, hi / 100))


# ---------------------------------------------------------------- item 1: the permutation null
def within_user_permutation(uc, rng) -> np.ndarray:
    """An index array p such that x[p] permutes x uniformly among the rows of each user (one random key per row, sorted within
    the user): the permutation unit is the user's TEST pairs."""
    uc = np.asarray(uc)
    order_orig = np.argsort(uc, kind="stable")
    order_rand = np.lexsort((rng.random(len(uc)), uc))
    perm = np.empty(len(uc), int)
    perm[order_orig] = order_rand
    return perm


def reading_null(obs: float, lo: float, hi: float) -> str:
    if not (_fin(obs) and _fin(lo) and _fin(hi)):
        return "NOT_AVAILABLE"
    if obs < lo:
        return "BELOW_NULL"
    if obs > hi:
        return "ABOVE_NULL"
    return "WITHIN_NULL"


def null_draws(D, est: str, n_perm: int, seed: int, base: dict, keys) -> np.ndarray:
    """(n_perm, n_models) null G per model: for permutation k the e-hat of every model is permuted among each user's rows with
    the same within-user permutation (seed [seed, STREAM_PERM, k]), the stackers [q-hat, pi, e-hat_perm] are re-run with the
    registered cross-fitting and the per-user AUC is differenced against M1."""
    out = np.full((n_perm, len(D.models)), NAN)
    for k in range(n_perm):
        perm = within_user_permutation(D.uc, np.random.default_rng([seed, STREAM_PERM, k]))
        G = g_matrix(D, est, {m: D.E[m][perm] for m in D.models}, base)
        if len(keys):
            out[k] = G[:, keys].mean(1)
    return out


def null_block(D, est: str, n_perm: int, seed: int, base: dict, obs: dict) -> dict:
    """One estimator of one regime: the null of every model and of the seed mean, the observed G and the reading."""
    keys = obs["keys"]
    nu = len(keys)
    draws = null_draws(D, est, n_perm, seed, base, keys)

    def one(d, est_obs):
        lo, hi = percentile_range(d)
        fin = d[np.isfinite(d)]
        n_ge = int((fin >= est_obs).sum()) if _fin(est_obs) else None
        return {"observed": est_obs,
                "null": _rec(float(fin.mean()) if len(fin) else NAN, lo, hi, nu, len(fin),
                             sd=float(fin.std(ddof=1)) if len(fin) > 1 else NAN),
                "n_perm": int(len(fin)), "n_null_ge_observed": n_ge,
                "share_ge_observed": (n_ge / len(fin)) if (n_ge is not None and len(fin)) else None,
                "reading": reading_null(est_obs, lo, hi)}
    per = {m: one(draws[:, j], obs["per_model"][m]["est"]) for j, m in enumerate(D.models)}
    mean = one(draws.mean(1), obs["mean_over_seeds"]["est"]) if len(D.models) else _na("no model")
    mean["draws"] = [float(x) for x in draws.mean(1)] if len(D.models) else []
    return {"per_model": per, "mean_over_seeds": mean, "n_users": nu, "n_perm": int(n_perm),
            "rule": NULL_RULE, "permutation_unit": "the TEST pairs of each S_d user (stacker rows)"}


# ---------------------------------------------------------------- item 2: the planted signal
def label_residual(D, mode: str = "fe_residual") -> tuple:
    """y~: the within-user centred TEST label after removing the registered item mean, standardised to unit variance over the rows.
    fe_residual: the pooled within-user least-squares slope of the centred label on the centred q-hat is projected out (R4);
    literal_diff: the within-user centred difference y - q-hat. Returns (y~, info)."""
    rows = D.rows_all
    yc = fr.user_centred(D.y.astype(float), D.uc, rows)
    qc = fr.user_centred(D.q, D.uc, rows)
    slope = NAN
    if mode == "fe_residual":
        den = float((qc * qc).sum())
        slope = float((yc * qc).sum() / den) if den > 0 else 0.0
        r = yc - slope * qc
    elif mode == "literal_diff":
        r = fr.user_centred(D.y.astype(float) - D.q, D.uc, rows)
    else:
        raise SystemExit(f"--ytilde_mode must be one of {YTILDE_MODES}")
    sd = math.sqrt(float((r * r).mean())) if len(r) else NAN
    return (r / sd if sd and sd > 0 else r), {"mode": mode, "slope_on_q_hat": slope, "sd_before_standardisation": sd}


def planted_signal(yt, eps, rho: float) -> np.ndarray:
    """c = rho y~ + sqrt(1 - rho^2) eps."""
    return rho * np.asarray(yt, float) + math.sqrt(1.0 - rho * rho) * np.asarray(eps, float)


def planted_scale(D, m: str) -> float:
    """sd(e-hat) of model m: the pooled within-user SD (divisor N - n_users, ftgrid_report.pooled_within_sd)."""
    return fr.pooled_within_sd(D.E[m], D.uc)


def eps_draw(n_all: int, seed: int, r: int) -> np.ndarray:
    """eps of replication r over every EVAL pair: N(0, 1), seed [seed, STREAM_EPS, r]."""
    return np.random.default_rng([seed, STREAM_EPS, r]).standard_normal(n_all)


def planted_residual(D, m: str, lam: float, scale: float, c) -> np.ndarray:
    """e-hat' = l' - pi with l' = l + lambda sd(e-hat) c: the residual recomputed from the planted confidence and the registered
    prior."""
    L_new = D.L[m] + lam * scale * np.asarray(c, float)
    return L_new - D.PI[m]


def bootstrap_index(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) user resamples: one rng.integers(0, n, n) per resample (the draws of ftgrid_report.mean_draws)."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, n, n) for _ in range(n_boot)]) if n and n_boot else np.zeros((n_boot, n), int)


def power_and_mdg(curve: list, lambdas, target: float = POWER_TARGET) -> dict:
    """curve = [(lambda, mean recovered G, power)]: the minimum detectable gain = the smallest mean recovered G among the
    lambdas with power >= target (null when none); at_grid_floor = that lambda is the smallest of the grid."""
    cands = [(g, lam, pw) for lam, g, pw in curve if _fin(g) and _fin(pw) and pw >= target]
    if not cands:
        return {"est": None, "lambda": None, "power": None, "at_grid_floor": None,
                "reason": f"power is below {target} at every lambda of the grid {[float(x) for x in lambdas]}"}
    g, lam, pw = min(cands, key=lambda t: (t[0], t[1]))
    floor = bool(lam == min(lambdas))
    return {"est": float(g), "lambda": float(lam), "power": float(pw), "at_grid_floor": floor,
            "reason": ("the smallest lambda of the grid already reaches the target: the value is an upper bound of the "
                       "minimum detectable gain" if floor else None)}


def planted_block(X, D, est: str, lambdas, n_rep: int, n_boot: int, seed: int, rho: float, mode: str, base: dict,
                  obs: dict) -> dict:
    """One estimator of one regime: for every lambda and replication the recovered G (mean over seeds) and the lower bound of its
    user-bootstrap interval; the mean recovered G, the power and the minimum detectable gain."""
    yt, yinfo = label_residual(D, mode)
    scales = {m: planted_scale(D, m) for m in D.models}
    keys = obs["keys"]
    idxm = bootstrap_index(len(keys), n_boot, seed)
    G_rec = {lam: np.full(n_rep, NAN) for lam in lambdas}
    lb = {lam: np.full(n_rep, NAN) for lam in lambdas}
    for r in range(n_rep):
        c = planted_signal(yt, eps_draw(X.cx.n, seed, r)[D.idx], rho)
        for lam in lambdas:
            E = {m: planted_residual(D, m, lam, scales[m], c) for m in D.models}
            gu = g_matrix(D, est, E, base)[:, keys].mean(0)
            G_rec[lam][r] = gu.mean()
            lb[lam][r] = percentile_ci(gu[idxm].mean(1))[0] if len(keys) else NAN
    per, curve = {}, []
    for lam in lambdas:
        g, l = G_rec[lam], lb[lam]
        fin = np.isfinite(g)
        lo, hi = percentile_range(g)
        power = float((l[np.isfinite(l)] > 0).mean()) if np.isfinite(l).any() else NAN
        per[_lam_key(lam)] = {
            "lambda": float(lam),
            "G_recovered": _rec(float(g[fin].mean()) if fin.any() else NAN, lo, hi, len(keys), int(fin.sum())),
            "G_recovered_minus_observed": float(g[fin].mean() - obs["mean_over_seeds"]["est"]) if fin.any() else None,
            "power": power, "mean_lower_bound": float(np.nanmean(l)) if np.isfinite(l).any() else None,
            "n_lower_bound_above_0": int((l[np.isfinite(l)] > 0).sum()), "n_rep": int(n_rep),
            "n_boot_per_replication": int(n_boot), "G_per_replication": [float(x) for x in g]}
        curve.append((lam, per[_lam_key(lam)]["G_recovered"]["est"], power))
    return {"per_lambda": per, "minimum_detectable_gain": power_and_mdg(curve, lambdas),
            "scale_sd_e_hat_within_user": {m: float(scales[m]) for m in D.models},
            "scale_sd_e_hat_total": {m: float(np.std(D.E[m])) for m in D.models},
            "y_tilde": yinfo, "rho": float(rho), "n_users": len(keys),
            "definition": "l' = l + lambda sd(e-hat) c, c = rho y~ + sqrt(1 - rho^2) eps; e-hat' = l' - pi; power = share of "
                          "replications whose 95% user-bootstrap lower bound of G is above 0; minimum detectable gain = the "
                          "smallest mean recovered G with power >= 0.8 (readings R4-R6)"}


# ---------------------------------------------------------------- item 3: the donor placebo
def reading_personal(g7: dict, contrast: dict) -> dict:
    """PERSONAL / DENOISING_ONLY / INCONCLUSIVE (A3-14 section 3, reading R8), with the conditions it reads."""
    lo, hi = _num(contrast.get("lo")), _num(contrast.get("hi"))
    est = _num(contrast.get("est"))
    g_lo, g_hi = _num(g7.get("lo")), _num(g7.get("hi"))
    if not all(math.isfinite(x) for x in (lo, hi, est, g_lo, g_hi)):
        return {"reading": "NOT_AVAILABLE", "rule": PERSONAL_RULE}
    c = {"contrast_ci_above_0": bool(lo > 0), "G7_ci_contains_0": bool(g_lo <= 0 <= g_hi),
         "contrast_est_within_band": bool(-BAND_PLACEBO <= est <= BAND_PLACEBO),
         "contrast_ci_within_band": bool(lo >= -BAND_PLACEBO and hi <= BAND_PLACEBO)}
    if c["contrast_ci_above_0"]:
        r = "PERSONAL"
    elif c["G7_ci_contains_0"] or c["contrast_est_within_band"]:
        r = "DENOISING_ONLY"
    else:
        r = "INCONCLUSIVE"
    return {"reading": r, "conditions": c, "rule": PERSONAL_RULE}


def placebo_rows(X, base_rows, models: list) -> tuple:
    """(rows, PI7, D8, n_excluded): the regime's rows with a finite pi-hat_(-8) and a finite donor-8 logit for every model."""
    rows = np.asarray(base_rows, bool).copy()
    PI7, D8, bad = {}, {}, 0
    items = X.cx.item.tolist()
    for m in models:
        pi7, d8, nb = donor_split(X.DON[m])
        bad += nb
        PI7[m] = np.where(X.cx.sd_test, np.array([pi7.get(i, NAN) for i in items], float), NAN)
        D8[m] = np.where(X.cx.sd_test, np.array([d8.get(i, NAN) for i in items], float), NAN)
        rows &= np.isfinite(PI7[m]) & np.isfinite(D8[m])
    return rows, PI7, D8, bad


def placebo_block(X, reg: str, info: dict, base_rows, est: str, n_boot: int, seed: int, k_splits: int) -> dict:
    """G_7, G_pl and the paired contrast of one estimator on one regime (R7)."""
    ms = [m for m in info["models"] if m in X.PI]
    rows, PI7, D8, bad = placebo_rows(X, base_rows, ms)
    D = make_design(X, rows, ms, info["n_registered"], k_splits)
    PI7c = {m: PI7[m][D.idx] for m in ms}
    E7 = {m: D.L[m] - PI7c[m] for m in ms}
    Epl = {m: D8[m][D.idx] - PI7c[m] for m in ms}
    base = {m: stack_auc(D, est, {"q_hat": D.q, "pi": PI7c[m]}, COLS1) for m in ms}
    G7 = g_matrix(D, est, E7, base, PI7c)
    Gpl = g_matrix(D, est, Epl, base, PI7c)
    both = np.isfinite(G7).all(0) & np.isfinite(Gpl).all(0)
    keys = np.flatnonzero(both)
    nu, npairs = len(keys), n_pairs_of(D, keys)
    v7 = G7[:, keys].mean(0) if nu else np.zeros(0)
    vpl = Gpl[:, keys].mean(0) if nu else np.zeros(0)
    V = np.column_stack([v7, vpl, v7 - vpl]) if nu else np.zeros((0, 3))
    Dr = fr.mean_draws(V, n_boot, seed)
    r7 = fr.rec(v7.mean() if nu else NAN, Dr[:, 0], nu, npairs, contrast=True)
    rpl = fr.rec(vpl.mean() if nu else NAN, Dr[:, 1], nu, npairs, contrast=True)
    rc = fr.rec((v7 - vpl).mean() if nu else NAN, Dr[:, 2], nu, npairs, contrast=True)
    per = {}
    for j, m in enumerate(ms):
        a, b = G7[j, keys], Gpl[j, keys]
        per[m] = {"G_7": float(a.mean()) if nu else None, "G_pl": float(b.mean()) if nu else None,
                  "G_7_minus_G_pl": float((a - b).mean()) if nu else None}
    return {"models": ms, "estimator": est, "n_rows": int(D.n), "n_users": nu,
            "n_items_without_8_donors": int(bad), "G_7": r7, "G_pl": rpl, "G_7_minus_G_pl": rc, "per_model": per,
            "reading": reading_personal(r7, rc),
            "definition": "pi-hat_(-8) = the mean of donors 1-7; e-hat_7 = l - pi-hat_(-8); e-hat_pl = the eighth donor's logit - "
                          "pi-hat_(-8); G_7 / G_pl = dUAUC([q-hat, pi-hat_(-8), e-hat] - [q-hat, pi-hat_(-8)]); paired user "
                          "resamples (reading R7)"}


# ---------------------------------------------------------------- item 4: honest intervals
def build_pairs(D) -> SimpleNamespace:
    """All (positive, negative) row pairs of every user (rows of D), for the weighted AUC of the two-way resampling."""
    pos, neg, usr = [], [], []
    order = np.argsort(D.uc, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(D.uc, minlength=D.nu))]
    for u in range(D.nu):
        rows = order[bounds[u]:bounds[u + 1]]
        yp, yn = rows[D.y[rows] == 1], rows[D.y[rows] == 0]
        if len(yp) and len(yn):
            pos.append(np.repeat(yp, len(yn)))
            neg.append(np.tile(yn, len(yp)))
            usr.append(np.full(len(yp) * len(yn), u))
    cat = (lambda parts: np.concatenate(parts) if parts else np.zeros(0, int))
    return SimpleNamespace(pos=cat(pos), neg=cat(neg), user=cat(usr))


def auc_weighted(score, w_row, pairs, nu: int) -> np.ndarray:
    """(nu,) per-user AUC with the item weights of the rows: sum over (positive, negative) pairs of w_p w_n [1(s_p > s_n) + 1/2
    1(s_p = s_n)] / sum w_p w_n, NaN for a user without a weighted positive and negative (the Mann-Whitney AUC of the rows
    replicated w times, ties 1/2)."""
    s = np.asarray(score, float)
    sp, sn = s[pairs.pos], s[pairs.neg]
    ww = np.asarray(w_row, float)[pairs.pos] * np.asarray(w_row, float)[pairs.neg]
    c = (sp > sn).astype(float) + 0.5 * (sp == sn)
    ok = np.isfinite(sp) & np.isfinite(sn)
    num = np.bincount(pairs.user[ok], ww[ok] * c[ok], minlength=nu)
    den = np.bincount(pairs.user[ok], ww[ok], minlength=nu)
    out = np.full(nu, NAN)
    pos = den > 0
    out[pos] = num[pos] / den[pos]
    return out


def crossfit_dup(Xm, y, fa, w) -> np.ndarray:
    """ftgrid_report.crossfit on a resample: the fit rows are duplicated by their integer weights (the same logit_fit as the
    registered stacker on the resampled rows), each fold is scored by the fit on the other fold's resampled rows."""
    Xm = np.asarray(Xm, float).reshape(len(y), -1)
    eta = np.full(len(y), NAN)
    for fit, pred in ((~fa, fa), (fa, ~fa)):
        fi = np.flatnonzero(fit & (w > 0))
        rep = np.repeat(fi, w[fi])
        if pred.any() and len(rep) >= 2 and 0 < y[rep].sum() < len(rep):
            eta[pred] = fr.logit_eta(fr.logit_fit(Xm[rep], y[rep]), Xm[pred])
    return eta


def centred_weighted(x, uc, nu: int, w) -> np.ndarray:
    """x minus the user's mean with the row weights w (rows without weight keep a value but are never used)."""
    cnt = np.bincount(uc, w, minlength=nu)
    s = np.bincount(uc, w * x, minlength=nu)
    mean = np.divide(s, cnt, out=np.full(nu, NAN), where=cnt > 0)
    return x - mean[uc]


def refit_auc(D, est: str, feats: dict, cols: tuple, w, Wi=None) -> np.ndarray:
    """(nu, K) per-user AUC of the stacker [cols] refitted on the resample whose row weights are w: E-D one fit pair; E-W
    re-centres the features (with the item weights when given) and refits every split. The AUC is the plain one, or the
    item-weighted one when Wi (the item weights of the rows) is given."""
    if est == "E_D":
        etas = [crossfit_dup(np.column_stack([feats[c] for c in cols]), D.y, D.fa, w)]
    else:
        wc = np.ones(D.n) if Wi is None else Wi
        C = {c: centred_weighted(feats[c], D.uc, D.nu, wc) for c in cols}
        Xc = np.column_stack([C[c] for c in cols])
        etas = [crossfit_dup(Xc, D.y, fa, w) for fa in D.folds]
    if Wi is None:
        return auc_matrix(D, etas)
    if D.pairs is None:
        D.pairs = build_pairs(D)
    return np.column_stack([auc_weighted(e, Wi, D.pairs, D.nu) for e in etas])


def resample_weights(D, scheme: str, n_boot: int, seed: int):
    """Yields (W_u, W_i) per resample: user counts of rng.integers(0, n_users, n_users) from default_rng(seed); for the two-way
    scheme item counts from an independent stream (pigeonhole weights)."""
    rng_u = np.random.default_rng(seed)
    rng_i = np.random.default_rng([seed, STREAM_ITEMS])
    for _ in range(n_boot):
        Wu = np.bincount(rng_u.integers(0, D.nu, D.nu), minlength=D.nu)
        Wi = np.bincount(rng_i.integers(0, D.ni, D.ni), minlength=D.ni) if scheme == "two_way" else None
        yield Wu, Wi


def refit_G_draws(D, est: str, scheme: str, n_boot: int, seed: int, refit: bool = True, fixed_eta: dict | None = None) -> np.ndarray:
    """(n_boot, n_models) resampled G per model: the multiplicity-weighted mean over the drawn users with a defined AUC of the
    per-user difference AUC(M2) - AUC(M1) (mean over splits), with the stackers refitted on each resample (refit) or the
    registered predictions kept (fixed_eta = {(model, 'M1'|'M2'): [eta, ...]}, two-way only)."""
    out = np.full((n_boot, len(D.models)), NAN)
    for b, (Wu, Wi) in enumerate(resample_weights(D, scheme, n_boot, seed)):
        wi_row = None if Wi is None else Wi[D.ic].astype(float)
        w = Wu[D.uc] * (Wi[D.ic] if Wi is not None else 1)
        for j, m in enumerate(D.models):
            if refit:
                f1 = {"q_hat": D.q, "pi": D.PI[m]}
                f2 = {**f1, "e_hat": D.E[m]}
                a1 = refit_auc(D, est, f1, COLS1, w, wi_row)
                a2 = refit_auc(D, est, f2, COLS2, w, wi_row)
            else:
                if D.pairs is None:
                    D.pairs = build_pairs(D)
                a1 = np.column_stack([auc_weighted(e, wi_row, D.pairs, D.nu) for e in fixed_eta[(m, "M1")]])
                a2 = np.column_stack([auc_weighted(e, wi_row, D.pairs, D.nu) for e in fixed_eta[(m, "M2")]])
            g = (a2 - a1).mean(1)
            ok = np.isfinite(g) & (Wu > 0)
            out[b, j] = (Wu[ok] * g[ok]).sum() / Wu[ok].sum() if ok.any() else NAN
    return out


def fixed_predictions(D, est: str) -> dict:
    """{(model, 'M1'|'M2'): [registered cross-fitted predictors]} of every model."""
    out = {}
    for m in D.models:
        out[(m, "M1")] = stack_eta(D, est, {"q_hat": D.q, "pi": D.PI[m]}, COLS1)
        out[(m, "M2")] = stack_eta(D, est, {"q_hat": D.q, "pi": D.PI[m], "e_hat": D.E[m]}, COLS2)
    return out


def interval_record(est_point, draws, n_users: int, n_pairs, registered: dict | None, contrast: bool = True) -> dict:
    """The record of a resampling interval around the registered point estimate (percentile interval of the finite draws), with
    the width ratio to the registered fixed-coefficient interval."""
    d = np.asarray(draws, float)
    rec = fr.rec(est_point, d, n_users, n_pairs, contrast=contrast)
    if registered is not None and all(_fin(registered.get(k)) for k in ("lo", "hi")) and _fin(rec["lo"]) and _fin(rec["hi"]):
        w0 = registered["hi"] - registered["lo"]
        rec["width_ratio_to_registered"] = float((rec["hi"] - rec["lo"]) / w0) if w0 > 0 else None
    rec["n_boot_nonfinite"] = int((~np.isfinite(d)).sum())
    return rec


def refit_block(D, est: str, scheme: str, n_boot: int, seed: int, obs: dict, refit: bool = True) -> dict:
    """The resampling interval of G (per model and the mean over seeds) beside the registered fixed-coefficient interval."""
    fixed = None if refit else fixed_predictions(D, est)
    draws = refit_G_draws(D, est, scheme, n_boot, seed, refit, fixed)
    keys = obs["keys"]
    nu, npairs = len(keys), n_pairs_of(D, keys)
    per = {m: interval_record(obs["per_model"][m]["est"], draws[:, j], nu, npairs, obs["per_model"][m])
           for j, m in enumerate(D.models)}
    mean = interval_record(obs["mean_over_seeds"]["est"], draws.mean(1), nu, npairs, obs["mean_over_seeds"])
    return {"per_model": per, "mean_over_seeds": mean, "registered_fixed_coefficient": obs["mean_over_seeds"],
            "n_users": nu, "n_boot": int(n_boot), "scheme": scheme, "refit": bool(refit)}


def p1_design(X, k_splits: int) -> SimpleNamespace | None:
    """P1's rows (ftgrid_report.p1_block): S_d TEST rows with a finite q-hat and a finite like logit and swap prior for the
    zero-shot model and the three seeds."""
    need = ["zeroshot", "s0", "s1", "s2"]
    if any(m not in X.PI or m not in X.L for m in need) or X.q is None:
        return None
    R = X.cx.sd_test & np.isfinite(X.q)
    for m in need:
        R &= np.isfinite(X.L[m]) & np.isfinite(X.PI[m])
    return make_design(X, R, need, 3, k_splits)


def p1_records(D, draws_G: np.ndarray, G_obs: np.ndarray, n_boot: int, seed: int, refit_draws: np.ndarray | None = None) -> dict:
    """P1 = mean over the seeds of G_FT,s - G_ZS (ftgrid_report.p1_block's computation on per-user G) and, when resample draws
    of the four models' G are given, its resampling interval and decision."""
    keys = user_keys(G_obs)
    nu, npairs = len(keys), n_pairs_of(D, keys)
    Gk = G_obs[:, keys]
    V = (Gk[1:] - Gk[0]).T if nu else np.zeros((0, 3))
    Dr = fr.mean_draws(V, n_boot, seed)
    per = {s: fr.rec(V[:, j].mean() if nu else NAN, Dr[:, j], nu, npairs, contrast=True, G_FT=float(Gk[j + 1].mean()) if nu else NAN)
           for j, s in enumerate(D.models[1:])}
    mean = fr.rec(V.mean(1).mean() if nu else NAN, Dr.mean(1), nu, npairs, contrast=True)
    out = {"registered_recomputed": {"per_seed": per, "mean_over_seeds": mean,
                                     "decision": fr.p1_decision([per[s]["est"] for s in D.models[1:]], mean)}}
    if refit_draws is not None:
        c = refit_draws[:, 1:] - refit_draws[:, [0]]
        cm = c.mean(1)
        rp = {s: interval_record(per[s]["est"], c[:, j], nu, npairs, per[s]) for j, s in enumerate(D.models[1:])}
        rm = interval_record(mean["est"], cm, nu, npairs, mean)
        out["resampled"] = {"per_seed": rp, "mean_over_seeds": rm,
                            "decision_under_these_draws": fr.p1_decision([rp[s]["est"] for s in D.models[1:]], rm)}
    return out


# ---------------------------------------------------------------- item 5: the ratios and Fieller's interval
def fieller(N: float, D: float, vNN: float, vND: float, vDD: float, z: float = Z95) -> dict:
    """Fieller's confidence set for rho = N / D from the estimates N, D and their covariance (vNN, vND, vDD): the rho with
    (N - rho D)^2 <= z^2 (vNN - 2 rho vND + rho^2 vDD), i.e. a rho^2 - 2 B rho + c <= 0 with a = D^2 - z^2 vDD, B = N D - z^2 vND,
    c = N^2 - z^2 vNN. A denominator that is not distinguishable from 0 (a <= 0) gives an UNBOUNDED set (lo = hi = null, the form
    named), never a finite interval."""
    z2 = z * z
    a, B, c = D * D - z2 * vDD, N * D - z2 * vND, N * N - z2 * vNN
    disc = B * B - a * c
    base = {"est": N / D if (D != 0 and math.isfinite(N) and math.isfinite(D)) else None, "z": float(z),
            "a": float(a), "B": float(B), "c": float(c), "discriminant": float(disc)}
    if not all(math.isfinite(x) for x in (N, D, vNN, vND, vDD)):
        return {**base, "bounded": False, "form": "undefined", "lo": None, "hi": None, "excluded_open_interval": None}
    if a > 0:
        if disc < 0:
            return {**base, "bounded": True, "form": "empty", "lo": None, "hi": None, "excluded_open_interval": None}
        r = math.sqrt(disc)
        return {**base, "bounded": True, "form": "interval", "lo": float((B - r) / a), "hi": float((B + r) / a),
                "excluded_open_interval": None}
    if a < 0:
        if disc > 0:
            r = math.sqrt(disc)
            r1, r2 = sorted(((B - r) / a, (B + r) / a))
            return {**base, "bounded": False, "form": "complement_of_interval", "lo": None, "hi": None,
                    "excluded_open_interval": [float(r1), float(r2)]}
        return {**base, "bounded": False, "form": "whole_line", "lo": None, "hi": None, "excluded_open_interval": None}
    if B == 0:                                  # a = 0: -2 B rho + c <= 0
        form = "whole_line" if c <= 0 else "empty"
        return {**base, "bounded": False, "form": form, "lo": None, "hi": None, "excluded_open_interval": None}
    edge = c / (2 * B)
    return {**base, "bounded": False, "form": "half_line_above" if B > 0 else "half_line_below", "lo": None, "hi": None,
            "excluded_open_interval": None, "boundary": float(edge)}


def ratio_block(X, models: tuple, control: str, n_boot: int, seed: int, stored: dict | None) -> dict:
    """The retention of one control (FT-C or FT-Q) from the per-user AUCs of the stored scores on the registered rows."""
    L, cx = X.L, X.cx
    miss = [m for m in models if m not in X.present]
    if miss:
        return _na(f"needs usable like arms of {list(models)}; missing or excluded: {miss}")
    R = cx.test.copy()
    for m in models:
        R &= np.isfinite(L[m])
    pu = {m: fr.user_aucs(L[m], cx.y, cx.uc, R) for m in models}
    keys = sorted(set.intersection(*(set(d) for d in pu.values())))
    nu, npairs = len(keys), fr._rows_of_users(cx.uc, R, keys)
    V = np.array([[pu[m][u] for m in models] for u in keys], float).reshape(nu, len(models))
    Dr = fr.mean_draws(V, n_boot, seed)
    zs, s0, s1, p0, p1 = (V[:, j] for j in range(5))
    Nu, Du = 0.5 * (p0 + p1) - zs, 0.5 * (s0 + s1) - zs
    dz, ds0, ds1, dp0, dp1 = (Dr[:, j] for j in range(5))
    Nd, Dd = 0.5 * (dp0 + dp1) - dz, 0.5 * (ds0 + ds1) - dz
    Nbar, Dbar = float(Nu.mean()), float(Du.mean())
    num = fr.rec(Nbar, Nd, nu, npairs)
    den = fr.rec(Dbar, Dd, nu, npairs)
    den["ci_excludes_0"] = bool(_fin(den["lo"]) and _fin(den["hi"]) and (den["lo"] > 0 or den["hi"] < 0))
    num["ci_excludes_0"] = bool(_fin(num["lo"]) and _fin(num["hi"]) and (num["lo"] > 0 or num["hi"] < 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = Nd / Dd
    fin = np.isfinite(rd)
    lo, hi = percentile_ci(rd[fin]) if fin.any() else (NAN, NAN)
    ratio_pct = _rec(Nbar / Dbar if Dbar != 0 else NAN, lo, hi, nu, int(fin.sum()), npairs,
                     n_draws_nonfinite=int((~fin).sum()))
    cov = np.cov(np.vstack([Nd, Dd])) if len(Nd) > 1 else np.full((2, 2), NAN)
    f_boot = fieller(Nbar, Dbar, float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 1]))
    cova = np.cov(np.vstack([Nu, Du])) / nu if nu > 1 else np.full((2, 2), NAN)
    f_an = fieller(Nbar, Dbar, float(cova[0, 0]), float(cova[0, 1]), float(cova[1, 1]))
    for f in (f_boot, f_an):
        f.update(n_users=int(nu), n_boot=int(len(Nd)), descriptive_min_n=bool(nu < fr.MIN_N))
    r0, r1, Rr = fe.retention(V.mean(0)) if nu else (NAN, NAN, NAN)
    dr0, dr1, dR = fe.retention(Dr)
    reg_R = fr.rec(float(Rr), dR, nu, npairs, n_draws_nonfinite=int((~np.isfinite(dR)).sum()))
    contrast = fr.rec(float((Du - Nu).mean()), Dd - Nd, nu, npairs, contrast=True)
    out = {"control": control, "models": list(models), "rows": {"n_users": nu, "n_pairs": npairs},
           "numerator_UAUC_control_minus_zeroshot": num, "denominator_UAUC_real_minus_zeroshot": den,
           "denominator_ci_excludes_0": den["ci_excludes_0"],
           "ratio_of_seed_means": {"est": Nbar / Dbar if Dbar != 0 else None, "percentile": ratio_pct,
                                   "fieller": f_boot, "fieller_analytic_covariance": f_an,
                                   "covariance_of_resampled_N_D": [[float(cov[0, 0]), float(cov[0, 1])],
                                                                    [float(cov[1, 0]), float(cov[1, 1])]]},
           "registered_retention_recomputed": {"R": reg_R, "R_per_seed": {"s0": float(r0), "s1": float(r1)},
                                               "label": fe.ftc_label(reg_R),
                                               "definition": "the registered mean of the two per-seed ratios "
                                                             "(ftgrid_extra.retention), beside the ratio of the seed means"},
           "UAUC_real_minus_control": contrast,
           "UAUC": {m: fr.rec(float(V[:, j].mean()), Dr[:, j], nu, npairs) for j, m in enumerate(models)},
           "rule": "Fieller's set for N / D with the covariance of the paired user resamples; unbounded (lo = hi = null) when "
                   "the denominator is not distinguishable from 0 at the 95% level (reading R11)"}
    if stored is not None:
        out["stored"] = stored
    return out


# ---------------------------------------------------------------- stored files: Holm sensitivity and bounds
def _load_json(path) -> dict:
    def bad(x):
        raise ValueError(f"non-strict JSON constant {x}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad)


def _backbone_family(doc: dict):
    return fe._family_of((doc.get("meta") or {}).get("backbone"))


def holm_tables(docs: list) -> dict:
    """Cross-panel Holm sensitivity from the stored p-values of registered reports: per backbone the Holm family over the G tests
    of every panel and regime and over all E-D tests (G and star permutation), beside the registered per-(panel, regime)
    family-wise p-values (reading R12)."""
    groups = {}
    for doc in docs:
        fam = _backbone_family(doc)
        if fam is None:
            raise SystemExit(f"report with an unknown backbone {(doc.get('meta') or {}).get('backbone')!r}")
        groups.setdefault(fam, []).append(doc)
    out = {}
    for fam, ds in sorted(groups.items()):
        tests, seen = [], set()
        for doc in ds:
            dom = doc["meta"]["domain"]
            if (fam, dom) in seen:
                raise SystemExit(f"two reports for {fam} {dom}")
            seen.add((fam, dom))
            for reg in REGS:
                fam_e = ((doc.get("E_D") or {}).get(reg) or {}).get("holm_family_E_D") or {}
                for t in ("G", "star_permutation"):
                    p = (fam_e.get("p") or {}).get(t)
                    tests.append({"key": f"{dom}:{reg}:{t}", "panel": dom, "regime": reg, "test": t, "p": p,
                                  "p_holm_registered": (fam_e.get("p_holm") or {}).get(t),
                                  "confirmed_registered": (fam_e.get("confirmed") or {}).get(t)})
        g_only = {t["key"]: t["p"] for t in tests if t["test"] == "G"}
        all_p = {t["key"]: t["p"] for t in tests}
        adj_g, adj_all = fr.holm(g_only), fr.holm(all_p)
        rows = []
        for t in tests:
            rows.append({**t, "p_holm_G_family": adj_g.get(t["key"]), "p_holm_all_E_D_family": adj_all[t["key"]],
                         "adj_below_0.05_G_family": (None if adj_g.get(t["key"]) is None else bool(adj_g[t["key"]] < 0.05)),
                         "adj_below_0.05_all_E_D_family": (None if adj_all[t["key"]] is None
                                                           else bool(adj_all[t["key"]] < 0.05))})
        out[fam] = {"panels": sorted(d for f, d in seen if f == fam), "m_G_family": sum(_fin(p) for p in g_only.values()),
                    "m_all_E_D_family": sum(_fin(p) for p in all_p.values()), "tests": rows,
                    "rule": "ftgrid_report.holm (step-down) over the members that carry a stored p-value; a sensitivity table, "
                            "not a hypothesis test, no family of the registered design (reading R12)"}
    return out


def interval_bound(rec: dict | None, band: float = BAND_BOUNDS) -> dict:
    """The stored record's upper 95% bound and whether its whole interval lies within [-band, +band] (inclusive)."""
    if not isinstance(rec, dict) or not (_fin(rec.get("lo")) and _fin(rec.get("hi")) and _fin(rec.get("est"))):
        return _na("no interval in the stored file")
    lo, hi = float(rec["lo"]), float(rec["hi"])
    return {"est": float(rec["est"]), "lo": lo, "hi": hi, "upper_95_bound": hi, "n_users": rec.get("n_users"),
            "n_boot": rec.get("n_boot"), "descriptive_min_n": rec.get("descriptive_min_n"), "band": float(band),
            "whole_interval_within_band": bool(lo >= -band and hi <= band)}


def bounds_tables(reports: list, extras: list) -> dict:
    """The upper 95% bound of G (E-D, the registered report), G_wu (E-W) and G_LLM|CF (E-F) (the extra files) of every panel and
    regime and whether the whole interval lies within [-0.01, +0.01] (reading R13); no new computation."""
    out = {}

    def slot(doc):
        m = doc.get("meta") or {}
        return f"{m.get('domain')}:{m.get('backbone')}"
    for doc in reports:
        blk = out.setdefault(slot(doc), {})
        for reg in REGS:
            ig = (((doc.get("E_D") or {}).get(reg) or {}).get("information_gain") or {})
            blk.setdefault(reg, {})["G"] = interval_bound(ig.get("G_mean_over_seeds"))
    for doc in extras:
        blk = out.setdefault(slot(doc), {})
        for reg in REGS:
            ew = ((doc.get("E_W") or {}).get(reg) or {})
            ef = ((doc.get("E_F") or {}).get(reg) or {})
            blk.setdefault(reg, {})["G_wu"] = interval_bound(((ew.get("G_wu") or {}).get("mean_over_seeds")))
            blk[reg]["G_LLM_given_CF"] = interval_bound(((ef.get("G_LLM_given_CF") or {}).get("mean_over_seeds")))
    return {"band": [-BAND_BOUNDS, BAND_BOUNDS], "panels": out,
            "rule": "upper 95% bound = the stored hi of the seed-mean record; the whole interval lies within the band when "
                    "-0.01 <= lo and hi <= 0.01 (reading R13); no new computation"}


# ---------------------------------------------------------------- the stored registered values beside the recomputation
def registered_check(rr: dict, p1: dict, report: dict | None, extra: dict | None) -> dict:
    """The recomputed registered-style estimates (rr = {G: {regime: observed}, G_wu: {...}}, the P1 block) against the stored
    report / extra file they reproduce (equality of est to 1e-9; of lo and hi when the stored record was built with the same
    number of resamples)."""
    out = {"report_given": report is not None, "extra_given": extra is not None, "compared": [], "mismatches": []}

    def cmp(label, mine, stored):
        if not isinstance(mine, dict) or not isinstance(stored, dict) or "est" not in mine or "est" not in stored:
            return
        row = {"label": label, "est": mine["est"], "stored_est": stored["est"],
               "equal": bool(_fin(mine["est"]) and _fin(stored["est"]) and abs(mine["est"] - stored["est"]) <= 1e-9)}
        if stored.get("n_boot") == mine.get("n_boot") and all(_fin(stored.get(k)) for k in ("lo", "hi")):
            row["interval_equal"] = bool(abs(mine["lo"] - stored["lo"]) <= 1e-9 and abs(mine["hi"] - stored["hi"]) <= 1e-9)
        out["compared"].append(row)
        if not row["equal"] or row.get("interval_equal") is False:
            out["mismatches"].append(label)
    for reg_name in REGS:
        mine = ((rr.get("G") or {}).get(reg_name) or {}).get("mean_over_seeds")
        if report is not None:
            stored = (((report.get("E_D") or {}).get(reg_name) or {}).get("information_gain") or {}).get("G_mean_over_seeds")
            cmp(f"G:{reg_name}", mine, stored)
        if extra is not None:
            mine = ((rr.get("G_wu") or {}).get(reg_name) or {}).get("mean_over_seeds")
            stored = (((extra.get("E_W") or {}).get(reg_name) or {}).get("G_wu") or {}).get("mean_over_seeds")
            cmp(f"G_wu:{reg_name}", mine, stored)
    if report is not None and isinstance(p1, dict) and p1.get("available") is not False:
        cmp("P1", (p1.get("registered_recomputed") or {}).get("mean_over_seeds"),
            (report.get("P1") or {}).get("mean_over_seeds"))
    return out


# ---------------------------------------------------------------- the build
def regime_infos(X) -> dict:
    out = {}
    for reg in REGS:
        ok, miss = fr._regime(X.models, X.present, reg)
        out[reg] = {"models": ok, "missing_or_excluded": miss, "n_registered": len(fr.REGIMES[reg])}
    return out


def regime_unavailable(X, reg: str, info: dict) -> str | None:
    ms = info["models"]
    if X.root_label == "teacher":
        return "teacher root: only the FT-Q ratio block is computed here (the regime blocks are the main root's)"
    if not ms:
        return (f"no usable like arm for the {reg} models {list(fr.REGIMES[reg])}" if info["missing_or_excluded"]
                else f"{reg} models not requested")
    if not X.sd_ids:
        return "no S_d"
    if not [m for m in ms if m in X.PI]:
        return "no usable swap arm (pi) for the regime's models"
    if X.q is None:
        return "no q-hat (needs --raw or --refs)"
    return None


def is_full_scope(X) -> bool:
    """ML-1M with Qwen3-8B: the scope of the G_wu and P1 refit and of the two-way resampling of G_wu (A3-14 item 4)."""
    return bool(X.domain == "ml1m" and fe._is_qwen(X.backbone))


def honest_blocks(D, est: str, X, a, obs: dict) -> dict:
    """Item 4 a, b for one estimator of one regime: the refit of G (E-D, every panel) or of G_wu (E-W, ML-1M with Qwen3-8B) and,
    on ML-1M, the two-way resampling (G always; G_wu with Qwen3-8B), beside the registered fixed-coefficient interval."""
    out = {}
    full = is_full_scope(X)
    name = "G" if est == "E_D" else "G_wu"
    n_boot, seed = int(a.n_boot_refit), int(a.seed)
    if est == "E_D" or full:
        out[f"{name}_refit_user"] = refit_block(D, est, "user", n_boot, seed, obs)
        if X.domain == "ml1m":
            out[f"{name}_two_way_fixed"] = refit_block(D, est, "two_way", n_boot, seed, obs, refit=False)
            out[f"{name}_two_way_refit"] = refit_block(D, est, "two_way", n_boot, seed, obs)
    else:
        out["note"] = ("G_wu is refitted and resampled two-way on ML-1M with Qwen3-8B only (A3-14 item 4); the registered "
                       "fixed-coefficient interval of this panel is in `observed`")
    return out


def build_regime(X, reg: str, info: dict, a) -> dict:
    """Items 1-4 for one regime: the registered recomputation, the null, the planted signal, the placebo and the intervals,
    for the pooled (E_D) and the within-user (E_W) estimator."""
    why = regime_unavailable(X, reg, info)
    if why:
        return _na(why)
    ms = [m for m in info["models"] if m in X.PI]
    base_rows = fe.ed_rows(X.cx, info["models"], ms, X.L, X.PI) & np.isfinite(X.q)
    D = make_design(X, base_rows, ms, info["n_registered"], int(a.k_splits))
    blk = {"models": ms, "missing_or_excluded": list(info["missing_or_excluded"]) + [m for m in info["models"] if m not in ms],
           "complete": len(ms) == info["n_registered"], "n_rows": int(D.n), "n_users_in_rows": int(D.nu)}
    for est in ESTIMATORS:
        if est == "E_W" and not D.folds:
            blk[est] = _na("no E-W splits")
            continue
        base = base_auc(D, est)
        obs = seed_recs(D, g_matrix(D, est, None, base), int(a.n_boot), int(a.seed))
        blk[est] = {"observed": {k: v for k, v in obs.items() if k != "keys"},
                    "permutation_null": null_block(D, est, int(a.n_perm), int(a.seed), base, obs),
                    "planted_signal": planted_block(X, D, est, [float(x) for x in a.lambdas], int(a.n_rep),
                                                    int(a.n_boot_power), int(a.seed), float(a.rho), a.ytilde_mode, base, obs),
                    "donor_placebo": placebo_block(X, reg, info, base_rows, est, int(a.n_boot), int(a.seed),
                                                   int(a.k_splits)),
                    "honest_intervals": honest_blocks(D, est, X, a, obs)}
    return blk


def build_p1(X, a) -> dict:
    """P1 (ftgrid_report.p1_block's quantity on per-user G) recomputed on P1's rows beside its refit and two-way intervals
    (ML-1M with Qwen3-8B)."""
    if X.root_label == "teacher":
        return _na("teacher root: only the FT-Q ratio block is computed here")
    D = p1_design(X, int(a.k_splits))
    if D is None:
        return _na("P1 needs the like and swap arms of zeroshot and s0-s2 and q-hat")
    G = g_matrix(D, "E_D", None, base_auc(D, "E_D"))
    out = {"role": X.role, "rows": {"n_rows": int(D.n), "n_users_in_rows": int(D.nu)},
           **p1_registered(D, G, int(a.n_boot), int(a.seed))}
    if is_full_scope(X):
        n_boot, seed = int(a.n_boot_refit), int(a.seed)
        res = {"user_refit": refit_G_draws(D, "E_D", "user", n_boot, seed, True),
               "two_way_fixed": refit_G_draws(D, "E_D", "two_way", n_boot, seed, False, fixed_predictions(D, "E_D")),
               "two_way_refit": refit_G_draws(D, "E_D", "two_way", n_boot, seed, True)}
        out["resampling"] = {k: p1_resampled(D, G, v, out["registered_recomputed"]) for k, v in res.items()}
    else:
        out["resampling"] = _na("P1's refit and two-way intervals are computed on ML-1M with Qwen3-8B (A3-14 item 4)")
    return out


def _load_stored(path, flag: str) -> dict | None:
    if not path:
        return None
    try:
        return _load_json(path)
    except (OSError, ValueError) as e:
        print(f"refused: {flag} {path} is unreadable ({type(e).__name__}: {e})", file=sys.stderr)
        raise SystemExit(2) from None


def stored_ratio(extra: dict | None, mine: dict) -> dict | None:
    """The stored FT_C_reading record of the same control beside the recomputed registered retention."""
    if extra is None:
        return None
    st = extra.get("FT_C_reading")
    if not isinstance(st, dict) or st.get("available") is False:
        return {"available": False, "reason": (st or {}).get("reason", "no FT_C_reading in the stored extra file")}
    rm = (mine.get("registered_retention_recomputed") or {}).get("R") or {}
    sr = st.get("R") or {}
    out = {"available": True, "label": st.get("label"), "R": {k: sr.get(k) for k in ("est", "lo", "hi", "n_users", "n_boot")},
           "control": st.get("control")}
    out["R_equal"] = bool(_fin(rm.get("est")) and _fin(sr.get("est")) and abs(rm["est"] - sr["est"]) <= 1e-9)
    if sr.get("n_boot") == rm.get("n_boot") and _fin(sr.get("lo")) and _fin(sr.get("hi")):
        out["R_interval_equal"] = bool(abs(rm["lo"] - sr["lo"]) <= 1e-9 and abs(rm["hi"] - sr["hi"]) <= 1e-9)
    return out


def build(a) -> dict:
    check_out_path(a.out, getattr(a, "out_root", None))        # refused before anything is read or written
    fingerprint = input_fingerprint(a)                        # the inputs as this build reads them
    report, extra = _load_stored(a.report, "--report"), _load_stored(a.extra, "--extra")
    X = load_inputs(a)
    infos = regime_infos(X)
    regimes = {reg: build_regime(X, reg, infos[reg], a) for reg in REGS}
    p1 = build_p1(X, a)
    rr = {"G": {}, "G_wu": {}}
    for reg, blk in regimes.items():
        if blk.get("available") is False:
            continue
        rr["G"][reg] = blk["E_D"]["observed"]
        if isinstance(blk.get("E_W"), dict) and blk["E_W"].get("available") is not False:
            rr["G_wu"][reg] = blk["E_W"]["observed"]
    check = registered_check(rr, p1, report, extra)
    problems = list(X.problems)
    if check["mismatches"]:
        problems.append("the recomputed registered estimates differ from the stored files: " + ", ".join(check["mismatches"]))
    ratios = {}
    if X.root_label == "main" and X.domain == "ml1m":
        ratios["FT_C"] = ratio_block(X, FTC_MODELS, "FT-C", int(a.n_boot), int(a.seed), None)
        if ratios["FT_C"].get("available") is not False:
            st = stored_ratio(extra, ratios["FT_C"])
            if st is not None:
                ratios["FT_C"]["stored"] = st
                if st.get("available") and (st.get("R_equal") is False or st.get("R_interval_equal") is False):
                    problems.append("the recomputed FT-C retention R differs from the stored extra file")
    else:
        ratios["FT_C"] = _na("FT-C is the registered control of ML-1M on the main root only")
    if X.root_label == "teacher":
        ratios["FT_Q"] = ratio_block(X, FTC_MODELS, "FT-Q", int(a.n_boot), int(a.seed), None)
        if ratios["FT_Q"].get("available") is not False:
            st = stored_ratio(extra, ratios["FT_Q"])
            if st is not None:
                ratios["FT_Q"]["stored"] = st
                if st.get("available") and (st.get("R_equal") is False or st.get("R_interval_equal") is False):
                    problems.append("the recomputed FT-Q retention R_Q differs from the stored extra file")
    else:
        ratios["FT_Q"] = _na("FT-Q is the teacher root's control: build with --root_label teacher")
    cx = X.cx
    meta = {"domain": a.domain, "root_label": a.root_label, "backbone": X.backbone, "backbones_seen": X.backbones, "T": X.T,
            "variant": X.split.get("variant"), "split_sha1": fx.file_sha1(a.split), "panel_sha1": X.panel_sha,
            "models_requested": X.models, "regimes": infos, "settings": settings_of(a), "registered_counts": registered_counts(a),
            "n_boot": a.n_boot, "seed": a.seed, "min_n": fr.MIN_N, "k_splits": len(X.folds_full),
            "n_eval_pairs": cx.n, "n_test_pairs": int(cx.test.sum()), "n_sd_users": len(X.sd_ids),
            "n_sd_test_pairs": int(cx.sd_test.sum()), "role_of_P1": getattr(X, "role", None),
            "references_source": X.refs_info.get("source") or X.refs_info.get("reason"), "code_sha1": code_sha1(),
            "record": record_status(getattr(a, "pilot_log", None)), "fingerprint": fingerprint}
    return {"spec": SPEC, "status": STATUS, "meta": meta,
            "input_checks": {"problems": problems, **X.checks,
                             "n_items_without_8_donors": X.n_items_bad_donors},
            "excluded_runs": X.excluded, "runs": X.runs, "joins": X.joins, "swap_arm_like_consistency": X.consistency,
            "references": X.refs_info, "registered_check": check, "rules": {"permutation_null": NULL_RULE,
                                                                             "personal_part": PERSONAL_RULE},
            "readings": list(READINGS), "registered_recomputed": rr, "regimes": regimes, "P1": p1, "ratios": ratios}
