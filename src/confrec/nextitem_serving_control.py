"""Cheap-signal controls for selective serving: an EXPLORATORY, DESCRIPTIVE, CPU-only analysis
(idea-stage/PREREG_AMENDMENT_3_ADDENDUM_11.md; written after the registered serving block of two domains had been read; no hypothesis,
no family, no multiplicity correction, no p-value: the only statements are the registered labels below).

The registered audit (docs/sigir/NEXTITEM_AUDIT_SPEC.md section D) serves the events with the largest list confidence p_max first.
This module compares that selection with signals that need no LLM score: hist_len (the number of history items of the event),
profile_pop (the user's popularity profile), pop_conf (p_max of a popularity-only ranker) and, for reference, the audit's
fixed-seed random signal. Every registered quantity is computed by the audit's own code, imported READ-ONLY from
src/confrec/nextitem_audit.py (load_panel, load_scores, rank_core / question_arrays, fit_temperature, softmax_probs,
user_profile, make_segments, EventBoot, ServeEngine, ci_dict ...): the p_max and random blocks of a result are the numbers of the
audit's own `run` on the same inputs (the optional --audit_json makes `run` check that, to 1e-12, and refuse otherwise).

    python -m src.confrec.nextitem_serving_control run --domain D --audit_dir A --panel_test P --panel_valid V --panel_kind K \
        [--out O] [--n_boot 2000] [--seed 0] [--segments auto|single] [--test_role R] [--valid_role R2] [--first_event N] \
        [--quarantine_n N] [--audit_json J]
    python -m src.confrec.nextitem_serving_control summarize --inputs a.json,b.json,... --out summary.json
    python -m src.confrec.nextitem_serving_control record [--pilot_log docs/sigir/PILOT_LOG.md] [--root R] [--files F ...]

Inputs of `run` are those of the audit's `run` (the same score directories <A>/<D>_<test_role> and <A>/<D>_<valid_role>, the same
panels); --ref_ranks / --ref_exposure are accepted for command-line compatibility and are never read. Only the question `next` is
registered (--questions next). The registered panels are the TEST segment of the audit (sports events 1,001-10,000, the quarantine
segment of events 1-1,000 is NOT analysed; toys / home / tools all events) and, as the second-backbone replication, the Z2 panels
(`--segments single`, one segment for the whole supplied panel). One `run` writes ONE json for ONE panel / segment.

Quantities (one block per signal; the definitions are repeated in the output):
 * serving set. p_max, random, hist_len, pop_conf serve the TEST events with at least one scored candidate (the events of the audit's
   section D). profile_pop serves those of them whose user has a mapped history item; the others are excluded from this signal's
   serving set and counted. Utilities: the LLM's NDCG@10 (audit tie handling) and HR@1.
 * signals. hist_len = len(history_item_ids) of the panel row. profile_pop = the audit's user_profile (mean group score, head 2 /
   mid 1 / tail 0, of the history items found in the panel's item -> group map). pop_conf = p_max of a list softmax whose score of a
   candidate is its group score (equal scores, equal probability), at the temperature fitted on the VALID events by the audit's
   routine. p_max and random are the audit's.
 * direction. p_max and random are used as they are. A control signal is served by the sign of the Spearman correlation (average
   ranks) between the signal and the LLM's NDCG@10 over the VALID events with at least one scored candidate; a zero or an undefined
   correlation reads as +1. TEST labels, TEST scores and TEST positive indices never enter it.
 * gain50(s) = mean utility of the served k = max(1, round(0.5 n)) events (equal signals form a tie block, served in expectation)
   minus the mean utility of the serving set; the curve over the audit's coverage points and the AURC are descriptive.
 * Delta(s) = gain50(p_max) - gain50(s) on the SAME bootstrap resamples (the segment's EventBoot with the given seed, columns
   restricted to each serving set), 95% percentile interval, no p-value; the same for HR@1. For profile_pop the gain of p_max is the
   one on all serving events of the panel (the registered number), the gain of profile_pop that on its own serving set.
 * reading (addendum section 5), per control signal from the NDCG@10 interval of Delta, checked in this order: LLM_BETTER if the lower
   bound is above 0; CHEAP_BETTER if the upper bound is below 0; MATCHED if the whole interval lies within +-0.01; else INCONCLUSIVE.
   Per panel: ADDS if every control signal reads LLM_BETTER; CHEAP_SUFFICES if any reads MATCHED or CHEAP_BETTER; else MIXED. A panel
   without a VALID score file (or with none that has a scored candidate) is NOT_RUN with the reason. `summarize` only counts.

Output. strict JSON, no wall-clock, host or timing; the same input gives the same bytes. `status` is EXPLORATORY_DESCRIPTIVE.
meta records the sha1 of this module and of the audit modules it imports, and the sha1 of the four input files. `run` refuses to
write below outputs/confrec/nextitem_audit (the registered audit directory) and over any existing file that is not one of its own
outputs; the default output directory is outputs/confrec/nextitem_audit_ctrl. The module opens only the paths it is given (and its own
source files for the code sha1); it never reads a reference-method file, the pilot log (except `record --pilot_log`) or a result of
another analysis.

Exit codes: 0 done; 1 an inconsistent input or a failed self-check (nothing is written); 2 refused or unusable input; 4 `record`: a sha1
is not in the pilot log. Memory: one TEST panel (streamed), the (events x candidates) score arrays and one int16 resample matrix
(n_boot x events); numpy + stdlib only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

from src.confrec import metrics, nextitem_audit as na, stats

SCHEMA = "nextitem_serving_control_v1"
SUMMARY_SCHEMA = "nextitem_serving_control_summary_v1"
STATUS = "EXPLORATORY_DESCRIPTIVE"
STATUS_NOTE = ("exploratory and descriptive: written after the registered serving block of two domains had been read "
               "(PREREG_AMENDMENT_3_ADDENDUM_11.md); no hypothesis, no family, no multiplicity correction, no p-value; "
               "the only statements are the registered labels")
SPEC = "idea-stage/PREREG_AMENDMENT_3_ADDENDUM_11.md"
QUESTION = "next"                                   # the only registered question
REG_N_BOOT, REG_SEED = 2000, 0                      # registered resamples and seed
BAND = 0.01                                         # registered MATCHED band
REGISTERED_ROLES = ("main", "all")                  # segments of the audit that the addendum analyses (never "quarantine")
SIGNALS = ("p_max", "random", "hist_len", "profile_pop", "pop_conf")
CONTROLS = ("hist_len", "profile_pop", "pop_conf", "random")        # the control signals of the reading rule (section 2)
VALID_DIRECTION = ("hist_len", "profile_pop", "pop_conf")           # directions fitted on the VALID events
SIGNAL_LABELS = ("LLM_BETTER", "CHEAP_BETTER", "MATCHED", "INCONCLUSIVE")      # in the order of checking
PANEL_LABELS = ("ADDS", "CHEAP_SUFFICES", "MIXED", "NOT_RUN")
AUDIT_DIR_PARTS = ("outputs", "confrec", "nextitem_audit")          # the registered audit directory
DEFAULT_OUT_DIR = "outputs/confrec/nextitem_audit_ctrl"
CODE_FILES = ("nextitem_serving_control.py", "nextitem_audit.py", "metrics.py", "stats.py")
RECORD_FILES = ("src/confrec/nextitem_serving_control.py", "tests/test_confrec_serving_control.py",
                "scripts/sigir/run_servingctrl.sh")
KIND_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
OWN_MARKER = '"schema": "nextitem_serving_control'
AUDIT_TOL = 1e-12                                   # --audit_json: the registered p_max / random numbers must be reproduced
assert na.COVERAGE[na.HALF] == 0.5 and tuple(na.UTILS) == ("ndcg10", "hr1")

DEFINITIONS = {
    "status": STATUS_NOTE,
    "panel": "the registered TEST segment of the audit (sports events 1,001-10,000, the quarantine segment is not analysed; other "
             "domains all events) or, with --segments single, the whole supplied panel as one segment (Z2)",
    "serving_set": "p_max, random, hist_len and pop_conf serve the events with at least one scored candidate (the events of the "
                   "audit's section D); profile_pop serves those of them whose user has a mapped history item, the others are "
                   "excluded from its serving set and counted",
    "utility": "per-event NDCG@10 of the LLM ranking (expected rank under random tie-breaking, the audit's rank_core) and HR@1 "
               "(deterministic top-1 correctness)",
    "p_max": "the audit's list-normalised top-1 probability at the temperature fitted on the VALID events (question next)",
    "random": "the audit's fixed-seed uniform draw, default_rng([seed, 1]) over the panel's events",
    "hist_len": "number of history_item_ids of the panel row (all of them, not only the window shown to the LLM)",
    "profile_pop": "the audit's user_profile: mean group score (head 2, mid 1, tail 0) of the history items found in the item -> "
                   "group map built from the candidate lists of the panel; no mapped history item: not served by this signal",
    "pop_conf": "p_max of a list softmax over the candidates whose score is the group score (head 2, mid 1, tail 0), equal scores "
                "receive equal probability; the temperature minimises the mean NLL of the positive over all events of the VALID "
                "score file (the audit's fit_temperature)",
    "direction": "p_max and random as they are; a control signal is served by the sign of the Spearman correlation (average ranks) "
                 "between the signal and the LLM's NDCG@10 over the VALID events with at least one scored candidate; zero or "
                 "undefined reads as +1; the item -> group map of the VALID profile is that of the VALID panel; no TEST label, "
                 "score or positive index enters",
    "gain50": "mean utility of the served k = max(1, round(0.5 n)) events (tie blocks served in expectation) minus the mean "
              "utility of the serving set (the audit's ServeEngine)",
    "delta": "Delta(s) = gain50(p_max) - gain50(s) on the same resamples (EventBoot of the segment, columns restricted to each "
             "serving set), 95% percentile interval, no p-value; HR@1 reported, only NDCG@10 is read",
    "reading": "per control signal from the NDCG@10 interval of Delta, in this order: LLM_BETTER if the lower bound is above 0; "
               "CHEAP_BETTER if the upper bound is below 0; MATCHED if the interval lies within +-0.01 (bounds included); else "
               "INCONCLUSIVE. Per panel: ADDS if every control signal reads LLM_BETTER; CHEAP_SUFFICES if any reads MATCHED or "
               "CHEAP_BETTER; else MIXED; NOT_RUN without a usable VALID score file. The control signals are hist_len, "
               "profile_pop, pop_conf and random",
    "niche": "the audit's niche block: quintiles of the user profile on the serving events with a profile, niche = lowest and "
             "mainstream = highest non-empty quintile, served share = expected share of the quintile's events served at 50%",
}


class ControlError(Exception):
    """A refused or inconsistent input: the message goes to stderr and `code` is the exit status."""

    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.code = code


def _log(msg: str) -> None:
    print(f"[nextitem_serving_control] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ the reading rule
def read_signal(lo, hi) -> str:
    """Reading of one control signal from the interval (lo, hi) of Delta (NDCG@10), checked in the registered order."""
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        return "INCONCLUSIVE"
    if lo > 0:
        return "LLM_BETTER"
    if hi < 0:
        return "CHEAP_BETTER"
    if lo >= -BAND and hi <= BAND:
        return "MATCHED"
    return "INCONCLUSIVE"


def read_panel(readings) -> str:
    """ADDS if every control signal reads LLM_BETTER; CHEAP_SUFFICES if any reads MATCHED or CHEAP_BETTER; else MIXED."""
    readings = list(readings)
    if readings and all(r == "LLM_BETTER" for r in readings):
        return "ADDS"
    if any(r in ("MATCHED", "CHEAP_BETTER") for r in readings):
        return "CHEAP_SUFFICES"
    return "MIXED"


# ------------------------------------------------------------------------------------- signals, direction (panel file + VALID)
def hist_len_signal(panel) -> np.ndarray:
    """(E,) number of history items of every event: len(history_item_ids) of the panel row."""
    return np.array([len(h) for h in panel.history], dtype=float)


def popularity_scores(panel) -> np.ndarray:
    """(E, N) group score (head 2, mid 1, tail 0, the audit's GROUP_SCORE) of every candidate; NaN where there is no candidate."""
    out = np.full(panel.grp.shape, np.nan)
    ok = (panel.grp >= 0) & panel.cmask
    out[ok] = na.GROUP_SCORE[panel.grp[ok]]
    return out


def _masked(L: np.ndarray, panel) -> np.ndarray:
    """The audit's convention for scores: an unscored candidate is -inf."""
    return np.where(np.isfinite(L) & panel.cmask, L, -np.inf)


def fit_pop_temperature(vpanel) -> dict:
    """The audit's fit_temperature on the popularity scores of the VALID events (positive = positive_item_index)."""
    return na.fit_temperature(_masked(popularity_scores(vpanel), vpanel), vpanel.pos)


def pop_conf_signal(panel, beta: float) -> np.ndarray:
    """(E,) p_max of the popularity-only list softmax at beta = 1 / T (equal scores receive equal probability)."""
    return na.softmax_probs(_masked(popularity_scores(panel), panel), beta).max(axis=1)


def direction_of(signal, utility) -> dict:
    """Direction of one control signal from VALID arrays: the sign of the Spearman correlation with the utility
    (average ranks, non-finite pairs dropped); a zero or an undefined correlation reads +1."""
    x, y = np.asarray(signal, float), np.asarray(utility, float)
    rho = stats.spearman(x, y)
    undefined = not math.isfinite(rho)
    return {"value": -1 if (not undefined and rho < 0.0) else 1, "source": "valid_spearman", "spearman": rho,
            "n_valid_pairs": int((np.isfinite(x) & np.isfinite(y)).sum()), "undefined": bool(undefined)}


# --------------------------------------------------------------------------------------------------------- serving
def serve_signal(name: str, signal: np.ndarray, utils: dict, quint: np.ndarray, Wv: np.ndarray):
    """(point estimates, replicates) of the audit's ServeEngine for one signal, with the audit's self-checks against
    metrics.risk_coverage."""
    nv = len(signal)
    eng = na.ServeEngine(signal, utils, quint)
    top = int(Wv.sum(axis=1).max()) if Wv.size else 0
    if top > nv:
        # A resample matrix restricted to the serving set (a signal that excludes events) can carry more weight than the set has
        # events, and the engine's table of harmonic numbers has nv + 1 entries only (the audit's own `run` raises an IndexError
        # for the same reason as soon as one event has no scored candidate). H[k] is a pure function of k, so a longer table
        # changes no number; the audit module itself is not touched.
        eng.H = na._harmonic(top)
    reps = eng.reps(Wv)
    est = eng.reps(np.ones((1, nv), np.int16))
    if nv:
        ks = np.minimum(np.maximum(1, np.rint(np.asarray(na.COVERAGE) * nv).astype(int)), nv)
        for u in na.UTILS:
            _, cum, aurc_ref = metrics.risk_coverage(signal, utils[u])
            na._self_check(f"AURC {name}/{u}", est["aurc"][u][0], aurc_ref)
            for j, k in enumerate(ks):
                na._self_check(f"risk-coverage curve {name}/{u}/{na.COVERAGE[j]}", est["curve"][u][0, j], cum[k - 1])
    return est, reps


def gain50(est: dict, reps: dict, u: str):
    """(point estimate, replicates) of the gain of serving the confident half: utility at 50% coverage minus the mean utility."""
    return (est["curve"][u][0, na.HALF] - est["mean_u"][u][0], reps["curve"][u][:, na.HALF] - reps["mean_u"][u])


def n_served(nv: int) -> int:
    """k = max(1, round(0.5 n)) (round half to even, as the audit's coverage points)."""
    return max(1, int(np.rint(na.COVERAGE[na.HALF] * nv))) if nv else 0


def _signal_block(est: dict, reps: dict, nv: int) -> dict:
    blk: dict = {"gain50": {}, "mean_utility": {}, "aurc": {}, "curve": {}}
    for u in na.UTILS:
        g_est, g_reps = gain50(est, reps, u)
        blk["gain50"][u] = na.ci_dict(g_est, g_reps, nv)
        blk["mean_utility"][u] = na.ci_dict(est["mean_u"][u][0], reps["mean_u"][u], nv)
        blk["aurc"][u] = na.ci_dict(est["aurc"][u][0], reps["aurc"][u], nv)
        blk["curve"][u] = [{"coverage": c, **na.ci_dict(est["curve"][u][0, j], reps["curve"][u][:, j], nv)}
                           for j, c in enumerate(na.COVERAGE)]
    with np.errstate(divide="ignore", invalid="ignore"):
        sh_est = est["served"][0] / est["size"][0]
        sh_reps = reps["served"] / reps["size"]
    filled = [j for j in range(na.N_QUINT) if est["size"][0, j] > 0]
    lo_bin, hi_bin = (filled[0], filled[-1]) if filled else (0, na.N_QUINT - 1)
    blk["niche"] = {
        "quintile_sizes": [int(est["size"][0, j]) for j in range(na.N_QUINT)],
        "served_share": [na.ci_dict(sh_est[j], sh_reps[:, j], int(est["size"][0, j])) for j in range(na.N_QUINT)],
        "niche_bin": lo_bin, "mainstream_bin": hi_bin,
        "niche_minus_mainstream_served_share": na.ci_dict(
            sh_est[lo_bin] - sh_est[hi_bin], sh_reps[:, lo_bin] - sh_reps[:, hi_bin],
            int(est["size"][0, lo_bin] + est["size"][0, hi_bin]))}
    return blk


def _header(domain, panel_kind, n_boot, seed, layout) -> dict:
    return {"schema": SCHEMA, "status": STATUS, "status_note": STATUS_NOTE, "spec": SPEC, "domain": domain,
            "question": QUESTION, "panel_kind": panel_kind, "n_boot": int(n_boot), "seed": int(seed), "layout": layout}


def not_run_doc(domain, panel_kind, reason, n_boot=REG_N_BOOT, seed=REG_SEED, layout=None) -> dict:
    """A panel that is not analysed: the label NOT_RUN and why."""
    doc = _header(domain, panel_kind, n_boot, seed, layout)
    doc.update(label="NOT_RUN", reason=reason)
    return doc


def analyze_panel(domain: str, panel, L_test: np.ndarray, vpanel, L_valid: np.ndarray, *, n_boot: int = REG_N_BOOT,
                  seed: int = REG_SEED, quarantine_n=None, segments: str = "auto", segment_name=None, first_event=None,
                  panel_kind: str = "", layout=None, load_diag=None, valid_diag=None) -> dict:
    """The analysis of one panel from loaded inputs: `panel` / `L_test` the TEST panel and its (E, N) `next` logits (NaN =
    unscored), `vpanel` / `L_valid` the VALID events (with candidate_popularity_groups) and their logits."""
    na._check_segment_options(segments, quarantine_n, first_event)
    E = panel.E
    if E == 0:
        raise ControlError("empty test panel")
    if panel.n_cand.min() < na.K:
        raise ControlError(f"every event needs at least {na.K} candidates (min {int(panel.n_cand.min())})")
    plan = (na.make_segments(domain, E, quarantine_n) if segments == "auto"
            else na.make_single_segment(domain, E, segment_name, first_event))
    reg = [s for s in plan if s["role"] in REGISTERED_ROLES]
    if len(reg) != 1:
        raise ControlError(f"{domain}: the panel has no registered segment (the quarantine segment is not part of addendum 11): "
                           f"{[s['name'] for s in plan]}")
    seg = reg[0]
    # ---- VALID: the LLM's temperature (as the audit), the popularity temperature and the directions of the controls
    Sv = _masked(L_valid, vpanel)
    if vpanel.E == 0 or not np.isfinite(Sv).any():
        return not_run_doc(domain, panel_kind, "the VALID score file has no event with a scored candidate (no temperature, no "
                           "direction)", n_boot, seed, layout)
    T_llm = na.fit_temperature(Sv, vpanel.pos)
    beta = T_llm["beta"] if np.isfinite(T_llm["beta"]) else 1.0                  # as analyze_domain
    T_pop = fit_pop_temperature(vpanel)
    if not np.isfinite(T_pop["beta"]):
        raise ControlError("the popularity temperature cannot be fitted on the VALID events")
    rcv = na.rank_core(Sv, vpanel.cmask, vpanel.pos)
    mv, ndcg_v = rcv["valid"], rcv["ndcg10"]
    prof_v, _ = na.user_profile(vpanel)
    valid_sig = {"hist_len": hist_len_signal(vpanel), "profile_pop": prof_v,
                 "pop_conf": pop_conf_signal(vpanel, T_pop["beta"])}
    direction = {s: direction_of(valid_sig[s][mv], ndcg_v[mv]) for s in VALID_DIRECTION}
    direction["p_max"] = {"value": 1, "source": "registered"}
    direction["random"] = {"value": 1, "source": "as_is"}
    # ---- TEST: the audit's arrays, the control signals, the segment
    rand = np.random.default_rng([int(seed), 1]).random(E)                       # the audit's fixed-seed random signal
    qa = na.question_arrays(L_test, panel, beta, rand)
    sl = slice(seg["lo"], seg["hi"])
    n = seg["hi"] - seg["lo"]
    qv = na._seg(qa, sl)
    m = qv["valid"]
    profile_all, prof_diag = na.user_profile(panel)
    profile = profile_all[sl]
    test_sig = {"p_max": qv["p_max"], "random": qv["random"], "hist_len": hist_len_signal(panel)[sl], "profile_pop": profile,
                "pop_conf": pop_conf_signal(panel, T_pop["beta"])[sl]}
    boot = na.EventBoot(n, n_boot, seed)
    cols_valid = np.flatnonzero(m)
    has = np.isfinite(profile[cols_valid])
    quint_valid = np.full(len(cols_valid), -1)
    if has.any():
        quint_valid[has] = stats.rank_bins(profile[cols_valid][has], na.N_QUINT)
    utils_all = {"ndcg10": qv["ndcg10"], "hr1": qv["hr1"]}
    res: dict = {}
    for s in SIGNALS:
        keep = has if s == "profile_pop" else np.ones(len(cols_valid), bool)      # profile_pop: only users with a mapped item
        cols = cols_valid[keep]
        Wv = boot.W if len(cols) == n else boot.W[:, cols]
        sig = direction[s]["value"] * test_sig[s][cols]
        est, reps = serve_signal(s, sig, {u: utils_all[u][cols] for u in na.UTILS}, quint_valid[keep], Wv)
        res[s] = {"nv": len(cols), "est": est, "reps": reps, "n_distinct": int(len(np.unique(sig)))}
    signals: dict = {}
    for s in SIGNALS:
        r = res[s]
        blk = {"kind": "llm" if s == "p_max" else "null_reference" if s == "random" else "control",
               "direction": direction[s],
               "serving_set": {"n_events": r["nv"], "n_excluded": int(len(cols_valid) - r["nv"]), "n_served_at_50": n_served(r["nv"]),
                               "n_distinct_values": r["n_distinct"]}}
        blk.update(_signal_block(r["est"], r["reps"], r["nv"]))
        if s != "p_max":
            blk["delta_vs_p_max"] = {}
            for u in na.UTILS:
                gp_est, gp_reps = gain50(res["p_max"]["est"], res["p_max"]["reps"], u)
                gs_est, gs_reps = gain50(r["est"], r["reps"], u)
                blk["delta_vs_p_max"][u] = na.ci_dict(gp_est - gs_est, gp_reps - gs_reps, min(res["p_max"]["nv"], r["nv"]))
        signals[s] = blk
    readings = {s: read_signal(signals[s]["delta_vs_p_max"]["ndcg10"]["lo"], signals[s]["delta_vs_p_max"]["ndcg10"]["hi"])
                for s in CONTROLS}
    for s in CONTROLS:
        signals[s]["reading"] = readings[s]
    fe = 1 if first_event is None else int(first_event)
    doc = _header(domain, panel_kind, n_boot, seed, layout)
    doc.update(
        label=read_panel(readings[s] for s in CONTROLS), reason=None, coverage_point=na.COVERAGE[na.HALF],
        coverage_points=list(na.COVERAGE), band=BAND, definition=DEFINITIONS,
        panel={"segment": seg["name"], "role": seg["role"], "event_range": [fe + seg["lo"], fe - 1 + seg["hi"]], "n_events": n},
        readings=readings,
        counts={"n_events_panel_file": int(E), "n_events_segment": int(n),
                "n_events_no_scored_candidate": int((~m).sum()),
                "n_events_serving_set": int(len(cols_valid)),
                "n_events_without_mapped_history": int((~has).sum()),
                "n_events_serving_set_profile_pop": int(has.sum()),
                "n_valid_events": int(vpanel.E), "n_valid_events_with_llm_scores": int(mv.sum()),
                "n_valid_events_pop_temperature": int(T_pop["n_events_used"]),
                "history": prof_diag, "scores_test": load_diag or {}, "scores_valid": valid_diag or {}},
        temperature={"p_max": T_llm, "pop_conf": T_pop},
        direction=direction, signals=signals)
    return doc


# ----------------------------------------------------------------------------------------- the audit's registered numbers
def check_audit_consistency(audit_doc: dict, doc: dict, tol: float = AUDIT_TOL) -> dict:
    """Require that the p_max and random blocks of `doc` reproduce the audit's own result for the same segment (gain at 50%,
    the risk-coverage curve and the AURC, both utilities; est / lo / hi and n) to `tol`. Returns what was compared."""
    seg = doc["panel"]["segment"]
    a_seg = (audit_doc.get("segments") or {}).get(seg)
    if a_seg is None:
        raise ControlError(f"the audit result has no segment {seg!r} (it has {list((audit_doc.get('segments') or {}))})", 1)
    if (audit_doc.get("n_boot"), audit_doc.get("seed")) != (doc["n_boot"], doc["seed"]):
        raise ControlError(f"the audit result was made with n_boot {audit_doc.get('n_boot')} and seed {audit_doc.get('seed')}, "
                           f"this run with {doc['n_boot']} and {doc['seed']}: its numbers cannot be reproduced", 1)
    D = ((a_seg.get("questions") or {}).get(QUESTION) or {}).get("D_selective_serving")
    if not D:
        raise ControlError(f"the audit result has no D_selective_serving block for question {QUESTION}", 1)
    bad, worst, n_cmp = [], 0.0, 0

    def cmp(label, a, b):
        nonlocal worst, n_cmp
        n_cmp += 1
        if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
            if not (a is None and b is None):
                bad.append(label)
            return
        worst = max(worst, abs(float(a) - float(b)))
        if abs(float(a) - float(b)) > tol:
            bad.append(label)

    if D.get("n_events") != doc["counts"]["n_events_serving_set"]:
        bad.append("n_events")
    for s in ("p_max", "random"):
        A, M = D["signals"][s], doc["signals"][s]
        for u in na.UTILS:
            pairs = [("gain50", A["gain_at_50_vs_full"][u], M["gain50"][u]), ("aurc", A["aurc"][u], M["aurc"][u])]
            pairs += [(f"curve@{ca['coverage']}", ca, cm) for ca, cm in zip(A["curve"][u], M["curve"][u])]
            for name, ca, cm in pairs:
                for key in ("est", "lo", "hi"):
                    cmp(f"{s}/{u}/{name}/{key}", ca[key], cm[key])
                if ca["n"] != cm["n"]:
                    bad.append(f"{s}/{u}/{name}/n")
    if bad:
        raise ControlError("the registered audit numbers are not reproduced (tolerance "
                           f"{tol}): {bad[:8]}{' ...' if len(bad) > 8 else ''}", 1)
    return {"checked": True, "segment": seg, "signals": ["p_max", "random"], "n_values_compared": n_cmp,
            "max_abs_diff": worst, "tolerance": tol}


# ------------------------------------------------------------------------------------------------------ output files
def _parts(p: Path) -> list:
    return [x.lower() for x in p.parts]


def under_audit_dir(path) -> bool:
    """True iff `path` (as written, absolute, or with links resolved) has the components outputs/confrec/nextitem_audit in a row:
    the registered audit directory, whatever the root. `outputs/confrec/nextitem_audit_ctrl` is a different directory."""
    k = len(AUDIT_DIR_PARTS)
    for cand in (Path(os.path.abspath(path)), Path(path).resolve()):
        parts = _parts(cand)
        if any(tuple(parts[i:i + k]) == AUDIT_DIR_PARTS for i in range(len(parts) - k + 1)):
            return True
    return False


def resolve_output(out, domain=None, panel_kind=None) -> Path:
    """The file `run` / `summarize` writes. No `out` (or a directory) means <dir>/<domain>__<panel_kind>.json in the given directory
    (default outputs/confrec/nextitem_audit_ctrl). Refused: anything below the registered audit directory outputs/confrec/
    nextitem_audit, and any existing file that is not an output of this module (an audit result, a document, anything else)."""
    spec = str(out) if out else DEFAULT_OUT_DIR
    p = Path(spec)
    if out is None or p.is_dir() or spec.endswith(("/", "\\")):
        if not (domain and panel_kind):
            raise ControlError("--out must name a file")
        p = p / f"{domain}__{panel_kind}.json"
    if under_audit_dir(p):
        raise ControlError(f"refused: {p} is below outputs/confrec/nextitem_audit, the registered audit directory; this analysis "
                           f"writes only to {DEFAULT_OUT_DIR} (or another directory of its own)")
    if p.is_dir():
        raise ControlError(f"refused: {p} is a directory")
    if p.exists():
        with open(p, "rb") as f:
            head = f.read(512)
        if OWN_MARKER.encode("ascii") not in head:
            raise ControlError(f"refused: {p} exists and is not an output of nextitem_serving_control (an audit file, or another "
                               "document, is never overwritten)")
    return p


def write_json(path: Path, doc: dict) -> None:
    """Strict JSON (NaN / inf -> null), LF line ends, UTF-8, written to <name>.tmp and renamed."""
    text = json.dumps(stats.strict_json(doc), indent=1, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(text.encode("utf-8"))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def code_sha1() -> dict:
    here = Path(__file__).resolve().parent
    return {f"src/confrec/{name}": na._sha1_file(here / name) for name in CODE_FILES}


# ------------------------------------------------------------------------------------------------------------ run
def run_control(domain: str, audit_dir, panel_test, panel_valid, out=None, *, panel_kind: str, n_boot: int = REG_N_BOOT,
                seed: int = REG_SEED, questions=(QUESTION,), quarantine_n=None, segments: str = "auto",
                test_role: str = "test", valid_role: str = "valid2k", first_event=None, audit_json=None) -> dict:
    """Analyse one panel and write its json. Inputs as the audit's run_domain: <audit_dir>/<domain>_<test_role>/scores.csv.gz
    (TEST), <audit_dir>/<domain>_<valid_role>/scores.csv.gz (VALID), the TEST and VALID panels. Without the VALID score file the
    panel reads NOT_RUN (written with the reason); a missing TEST input is an error."""
    na._check_segment_options(segments, quarantine_n, first_event)
    if tuple(questions) != (QUESTION,):
        raise ControlError(f"addendum 11 registers the question {QUESTION!r} only (got {list(questions)})")
    if not KIND_RE.match(str(panel_kind or "")):
        raise ControlError(f"--panel_kind must be a label of [a-z0-9_] (got {panel_kind!r})")
    if int(n_boot) < 1:
        raise ControlError("--n_boot must be >= 1")
    target = resolve_output(out, domain, panel_kind)                      # refusals come before any work
    audit = Path(audit_dir)
    s_test = audit / f"{domain}_{test_role}" / "scores.csv.gz"
    s_valid = audit / f"{domain}_{valid_role}" / "scores.csv.gz"
    need = [s_test, Path(panel_test), Path(panel_valid)] + ([Path(audit_json)] if audit_json else [])
    for p in need:
        if not p.exists():
            raise FileNotFoundError(p)
    layout = {"segments": segments, "test_role": test_role, "valid_role": valid_role,
              "first_event": None if first_event is None else int(first_event),
              "quarantine_n": None if quarantine_n is None else int(quarantine_n)}
    audit_doc = json.loads(Path(audit_json).read_text(encoding="utf-8")) if audit_json else None
    if not s_valid.exists():
        _log(f"{domain}: no VALID score file ({s_valid}): NOT_RUN")
        doc = not_run_doc(domain, panel_kind, f"VALID score file missing: {s_valid}", n_boot, seed, layout)
    else:
        panel = na.load_panel(panel_test)
        _log(f"{domain}: test panel {panel.E} events x {panel.N} candidates, pool {len(panel.pool)} items")
        L_test, diag = na.load_scores(s_test, panel, (QUESTION,))
        vpanel = na.load_panel(panel_valid, only_events=na.scan_event_ids(s_valid), need_groups=True)
        L_valid, vdiag = na.load_scores(s_valid, vpanel, (QUESTION,))
        doc = analyze_panel(domain, panel, L_test[QUESTION], vpanel, L_valid[QUESTION], n_boot=n_boot, seed=seed,
                            quarantine_n=quarantine_n, segments=segments, segment_name=test_role, first_event=first_event,
                            panel_kind=panel_kind, layout=layout, load_diag=diag, valid_diag=vdiag)
    meta = {"code_sha1": code_sha1(),
            "input_sha1": {"panel_test": na._sha1_file(panel_test), "panel_valid": na._sha1_file(panel_valid),
                           "scores_test": na._sha1_file(s_test),
                           "scores_valid": na._sha1_file(s_valid) if s_valid.exists() else None},
            "inputs": {"audit_dir": str(audit_dir), "panel_test": str(panel_test), "panel_valid": str(panel_valid),
                       "test_role": test_role, "valid_role": valid_role},
            "registered_settings": bool(int(n_boot) == REG_N_BOOT and int(seed) == REG_SEED),
            "audit_consistency": {"checked": False}}
    if audit_doc is not None and doc["label"] != "NOT_RUN":
        meta["audit_consistency"] = check_audit_consistency(audit_doc, doc)
    doc["meta"] = meta
    write_json(target, doc)
    return doc


# --------------------------------------------------------------------------------------------------------- summarize
def summarize(docs: list) -> dict:
    """Counts of the per-panel labels (and of the per-signal readings) over run outputs of ONE panel kind. Refused: no input, an
    input that is not a run output, mixed panel kinds, layouts, resample settings or code, a panel twice."""
    if not docs:
        raise ControlError("summarize: no input")
    for d in docs:
        if d.get("schema") != SCHEMA:
            raise ControlError(f"summarize: an input is not a {SCHEMA} document (schema {d.get('schema')!r})")
        if d.get("label") not in PANEL_LABELS:
            raise ControlError(f"summarize: an input has an unknown label {d.get('label')!r}")

    def same(name, getter):
        vals = []
        for d in docs:
            v = getter(d)
            if v not in vals:
                vals.append(v)
        if len(vals) != 1:
            raise ControlError(f"summarize: mixed {name} ({vals}): counts are made per panel kind, run `summarize` once per kind")
        return vals[0]

    kind = same("panel kinds", lambda d: d.get("panel_kind"))
    n_boot = same("n_boot", lambda d: d.get("n_boot"))
    seed = same("seeds", lambda d: d.get("seed"))
    same("layouts", lambda d: d.get("layout"))
    same("code versions", lambda d: (d.get("meta") or {}).get("code_sha1"))
    keys = [(d["domain"], (d.get("panel") or {}).get("segment")) for d in docs]
    dup = sorted({str(k) for k in keys if keys.count(k) > 1})
    if dup:
        raise ControlError(f"summarize: a panel is listed twice: {dup}")
    counts = {lab: 0 for lab in PANEL_LABELS}
    by_label: dict = {lab: [] for lab in PANEL_LABELS}
    sig_counts = {s: {lab: 0 for lab in SIGNAL_LABELS} for s in CONTROLS}
    for d in docs:
        counts[d["label"]] += 1
        by_label[d["label"]].append(d["domain"])
        for s, lab in (d.get("readings") or {}).items():
            if s in sig_counts and lab in sig_counts[s]:
                sig_counts[s][lab] += 1
    return {"schema": SUMMARY_SCHEMA, "status": STATUS, "status_note": STATUS_NOTE, "spec": SPEC, "panel_kind": kind,
            "n_panels": len(docs), "n_boot": n_boot, "seed": seed, "panel_labels": counts,
            "domains_by_label": {lab: sorted(v) for lab, v in by_label.items()}, "signal_readings": sig_counts,
            "panels": sorted(({"domain": d["domain"], "segment": (d.get("panel") or {}).get("segment"), "label": d["label"]}
                              for d in docs), key=lambda r: (r["domain"], str(r["segment"])))}


# ------------------------------------------------------------------------------------------------------------ record
def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def record_lines(files=RECORD_FILES, root=None) -> list:
    """`path = sha1` for each file (the lines recorded in the pilot log)."""
    root = Path(root) if root else default_root()
    try:
        return [f"{rel} = {na._sha1_file(root / rel)}" for rel in files]
    except OSError as e:
        raise ControlError(f"cannot read a file for the record: {e}", 1) from None


def record_missing(pilot_log, files=RECORD_FILES, root=None) -> list:
    """The files whose sha1 is not in the pilot log (a case-insensitive substring test, as ftgrid_freeze)."""
    if not Path(pilot_log).is_file():
        raise ControlError(f"pilot log {pilot_log} does not exist", 4)
    log = Path(pilot_log).read_text(encoding="utf-8").lower()
    return [line.split(" = ")[0] for line in record_lines(files, root) if line.split(" = ")[1].lower() not in log]


def cmd_record(a) -> int:
    files = tuple(a.files) if a.files else RECORD_FILES
    if not a.pilot_log:
        print("\n".join(record_lines(files, a.root)))
        return 0
    missing = record_missing(a.pilot_log, files, a.root)
    if missing:
        print(f"nextitem_serving_control record (addendum 11 section 6): the sha1 of these files is not in {a.pilot_log}: {missing}; "
              "run `python -m src.confrec.nextitem_serving_control record`, record the lines in the pilot log and push it",
              file=sys.stderr)
        return 4
    print(f"nextitem_serving_control record OK: every listed sha1 is in {a.pilot_log}")
    return 0


# ---------------------------------------------------------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0], allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one panel / segment: gain at 50%, contrasts with p_max, readings (CPU)", allow_abbrev=False)
    r.add_argument("--domain", required=True)
    r.add_argument("--audit_dir", required=True, help="holds <domain>_<test_role>/scores.csv.gz and <domain>_<valid_role>/scores.csv.gz")
    r.add_argument("--panel_test", required=True)
    r.add_argument("--panel_valid", required=True)
    r.add_argument("--panel_kind", required=True, help="label of the panel kind / backbone, [a-z0-9_]: `summarize` counts per kind")
    r.add_argument("--out", default=None, help=f"the json file (or a directory); default a file in {DEFAULT_OUT_DIR}")
    r.add_argument("--n_boot", type=int, default=REG_N_BOOT)
    r.add_argument("--seed", type=int, default=REG_SEED)
    r.add_argument("--questions", default=QUESTION, help="only the registered question: next")
    r.add_argument("--quarantine_n", type=int, default=None, help="as the audit's run")
    r.add_argument("--segments", choices=("auto", "single"), default="auto", help="as the audit's run")
    r.add_argument("--test_role", default="test")
    r.add_argument("--valid_role", default="valid2k")
    r.add_argument("--first_event", type=int, default=None)
    r.add_argument("--audit_json", default=None, help="the audit's result for this domain: require that the p_max and random "
                   "numbers are reproduced (to 1e-12) and refuse otherwise")
    r.add_argument("--ref_ranks", default=None, help="accepted for command-line compatibility with the audit's run; never read")
    r.add_argument("--ref_exposure", default=None, help="accepted for command-line compatibility with the audit's run; never read")
    s = sub.add_parser("summarize", help="counts of the per-panel labels over run outputs of one panel kind", allow_abbrev=False)
    s.add_argument("--inputs", required=True, help="comma-separated run json files")
    s.add_argument("--out", required=True)
    t = sub.add_parser("record", help="print `path = sha1` of this analysis's files; with --pilot_log: exit 0 iff the log holds "
                       "every sha1", allow_abbrev=False)
    t.add_argument("--pilot_log", default=None)
    t.add_argument("--root", default=None, help="the repo root (default: this checkout)")
    t.add_argument("--files", nargs="+", default=None, help=f"repo-relative files (default {', '.join(RECORD_FILES)})")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        if a.cmd == "record":
            return cmd_record(a)
        if a.cmd == "run":
            if a.ref_ranks or a.ref_exposure:
                _log("--ref_ranks / --ref_exposure are accepted for compatibility with the audit's run and are not read")
            doc = run_control(a.domain, a.audit_dir, a.panel_test, a.panel_valid, a.out, panel_kind=a.panel_kind,
                              n_boot=a.n_boot, seed=a.seed, questions=tuple(x for x in a.questions.split(",") if x),
                              quarantine_n=a.quarantine_n, segments=a.segments, test_role=a.test_role,
                              valid_role=a.valid_role, first_event=a.first_event, audit_json=a.audit_json)
            print(f"wrote {resolve_output(a.out, a.domain, a.panel_kind)} (panel {doc['domain']}/{a.panel_kind}: {doc['label']})")
            return 0
        docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in a.inputs.split(",") if p]
        res = summarize(docs)
        target = resolve_output(a.out, "summary", res["panel_kind"]) if a.out else None
        write_json(target, res)
        print(f"wrote {target} (panel kind {res['panel_kind']}: {res['panel_labels']})")
        return 0
    except ControlError as e:
        print(f"{a.cmd}: {e}", file=sys.stderr)
        return e.code
    except (FileNotFoundError, ValueError) as e:
        print(f"{a.cmd}: {e}", file=sys.stderr)
        return 2
    except AssertionError as e:
        print(f"{a.cmd}: inconsistent: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
