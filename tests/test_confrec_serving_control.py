"""Tests of the cheap-signal controls for selective serving (src/confrec/nextitem_serving_control.py; registered text:
idea-stage/PREREG_AMENDMENT_3_ADDENDUM_11.md; runner: scripts/sigir/run_servingctrl.sh). CPU only, synthetic fixtures built in tmp_path.

Every number the module reports is checked against something that does not share its code: hand-built tiny panels with explicit
ranks, a brute-force expectation over all tie-breakings, an independent temperature solver, explicit resamples of the audit's own
resample generator with metrics.risk_coverage and stats.percentile_ci, and the audit's own `run` output (p_max and random must be
reproduced to 1e-12). The audit-test fixtures (tests/test_confrec_nextitem_audit.py) are reused by import.

Sections: 1 reading rule; 2 control signals against hand computations; 3 directions (VALID only); 4 the registered p_max / random
numbers = the audit's; 5 serving numbers against brute force and the pairing of the contrasts; 6 labels end to end, NOT_RUN;
7 outputs (refusals, determinism, vocabulary, path guard); 8 summarize and record; 9 the runner script.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.confrec import metrics, nextitem_audit as na, nextitem_serving_control as sc, stats
from tests.test_confrec_nextitem_audit import make_panel_rows, synth_arrays, synth_confident, write_jsonl, write_scores

ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
MODULE = ROOT / "src" / "confrec" / "nextitem_serving_control.py"
SCRIPT = ROOT / "scripts" / "sigir" / "run_servingctrl.sh"
B = 60                                                  # resamples of most tests (the registered value is 2,000)
CONTROLS = ("hist_len", "profile_pop", "pop_conf", "random")


# ================================================================================================ independent references
def avg_ranks(x):
    """1-based average ranks (ties share the mean rank), plain Python."""
    x = list(map(float, x))
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_ref(x, y):
    """Spearman correlation = Pearson of the average ranks (NaN when a rank vector is constant)."""
    rx, ry = avg_ranks(x), avg_ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx, syy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def ref_gain50(sig, util):
    """Gain of serving the confident half by metrics.risk_coverage (tie blocks in expectation), k = max(1, round(0.5 n))."""
    sig, util = np.asarray(sig, float), np.asarray(util, float)
    _, cum, _ = metrics.risk_coverage(sig, util)
    k = max(1, int(round(0.5 * len(sig))))
    return float(cum[k - 1] - util.mean())


def brute_gain50(sig, util):
    """The same quantity as the expectation over ALL orderings that are consistent with the signal (random tie-breaking)."""
    n = len(sig)
    k = max(1, int(round(0.5 * n)))
    vals = [np.mean([util[i] for i in perm[:k]]) for perm in itertools.permutations(range(n))
            if all(sig[perm[i]] >= sig[perm[i + 1]] for i in range(n - 1))]
    return float(np.mean(vals) - np.mean(util))


def expected_beta(S, pos, lo=1e-3, hi=10.0):
    """argmin of the mean NLL of the positive under softmax(beta * S) on [lo, hi]: bisection on the derivative
    mean(E_p[S] - S_pos), written here so that it shares nothing with the audit's routine."""
    def grad(beta):
        g = 0.0
        for s, p in zip(S, pos):
            s = np.asarray(s, float)
            w = np.exp(beta * (s - s.max()))
            g += (w * s).sum() / w.sum() - s[p]
        return g / len(S)
    if grad(lo) >= 0:
        return lo
    if grad(hi) <= 0:
        return hi
    a, b = lo, hi
    for _ in range(200):
        mid = 0.5 * (a + b)
        if grad(mid) < 0:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def scores_with_ranks(pos, ranks, N, seed=0):
    """(E, N) distinct logits whose positive has exactly the given 1-based rank (the others in a shuffled order)."""
    rng = np.random.default_rng(seed)
    L = np.zeros((len(pos), N))
    for e, (p, r) in enumerate(zip(pos, ranks)):
        others = [c for c in range(N) if c != p]
        rng.shuffle(others)
        order = others[:r - 1] + [int(p)] + others[r - 1:]
        for i, c in enumerate(order):
            L[e, c] = float(N - i)
    return L


def ndcg_of_ranks(ranks):
    return metrics.ndcg_from_rank(np.asarray(ranks, float), 10)


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ci_of(blk):
    return blk["est"], blk["lo"], blk["hi"]


# =========================================================================================================== worlds
def build_world(tmp, domain="toys", E=90, N=15, n_valid=48, seed=0, *, L=None, grp=None, pos=None, Lv=None, gv=None, pv=None,
                hist_test=None, hist_valid=None, censored_test=None, censored_valid=None, test_role="test",
                valid_role="valid2k", audit="audit", rows=None, vrows=None, signal=2.0, head_bias=1.5, hist_max=6,
                with_valid=True):
    """A synthetic domain on disk in the layout of the audit's run: panels, <audit>/<d>_<test_role> (TEST scores, questions next
    and like; `like` is a decoy that must never matter) and <audit>/<d>_<valid_role> (VALID scores)."""
    tmp = Path(tmp)
    if L is None:
        L, grp, pos, _ = synth_arrays(E, N, seed, signal=signal, head_bias=head_bias)
    E, N = L.shape
    if rows is None:
        rows = make_panel_rows(E, N, seed, pos, grp, domain=domain, hist_max=hist_max)
    if Lv is None:
        Lv, gv, pv, _ = synth_arrays(n_valid, N, seed + 100, signal=signal, head_bias=head_bias)
    n_valid = Lv.shape[0]
    if vrows is None:
        vrows = make_panel_rows(n_valid, N, seed + 100, pv, gv, domain=domain + "v", hist_max=hist_max)
    for rws, lens, tag in ((rows, hist_test, "t"), (vrows, hist_valid, "v")):
        if lens is not None:
            for e, k in enumerate(lens):
                h = rws[e]["history_item_ids"]
                rws[e]["history_item_ids"] = h[:k] + [f"{domain}{tag}X{e}_{j}" for j in range(max(0, k - len(h)))]
    write_jsonl(tmp / "panels" / f"{domain}_test.jsonl", rows)
    write_jsonl(tmp / "panels" / f"{domain}_valid.jsonl", vrows)
    write_scores(tmp / audit / f"{domain}_{test_role}" / "scores.csv.gz", rows, {"next": L, "like": -L}, censored_test)
    if with_valid:
        write_scores(tmp / audit / f"{domain}_{valid_role}" / "scores.csv.gz", vrows, {"next": Lv, "like": -Lv}, censored_valid)
    L_seen = np.array(L, dtype=float)                       # what the loader makes of the file: censored 2 / 3 = unscored
    for (q, e, c), code in (censored_test or {}).items():
        if q == "next" and code in (2, 3):
            L_seen[e, c] = np.nan
    return SimpleNamespace(tmp=tmp, domain=domain, rows=rows, vrows=vrows, L=L, L_seen=L_seen, grp=grp, pos=pos, Lv=Lv, gv=gv, pv=pv,
                           audit_dir=tmp / audit, panel_test=tmp / "panels" / f"{domain}_test.jsonl",
                           panel_valid=tmp / "panels" / f"{domain}_valid.jsonl", test_role=test_role, valid_role=valid_role)


def run(w, out="out/run.json", *, kind="qwen_registered", n_boot=B, seed=0, **kw):
    """sc.run_control on a world; returns the parsed json that was written."""
    kw.setdefault("segments", "auto")
    return sc.run_control(w.domain, w.audit_dir, w.panel_test, w.panel_valid, w.tmp / out, panel_kind=kind, n_boot=n_boot,
                          seed=seed, test_role=w.test_role, valid_role=w.valid_role, **kw)


def audit_run(w, n_boot=B, seed=0, **kw):
    """The audit's own run_domain of the same inputs (question next only; no reference methods)."""
    return na.run_domain(w.domain, w.audit_dir, w.panel_test, w.panel_valid, None, None, questions=("next",), n_boot=n_boot,
                         seed=seed, test_role=w.test_role, valid_role=w.valid_role, **kw)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One synthetic domain (ties on a 1/16 grid, unscored candidates in a few events, no event without a scored candidate)
    shared by the tests that only read it, with its audit result and its control result."""
    tmp = tmp_path_factory.mktemp("ctrl_world")
    cens = {("next", 3, 2): 2, ("next", 3, 7): 3, ("next", 40, 0): 2, ("next", 71, 5): 2}
    w = build_world(tmp, "toys", E=96, N=15, n_valid=50, seed=7, censored_test=cens, hist_max=7)
    w.audit = audit_run(w)
    w.doc = run(w)
    return w


# ============================================================================================== 1. the reading rule
@pytest.mark.parametrize("lo,hi,want", [
    (0.001, 0.2, "LLM_BETTER"), (1e-12, 1e-12, "LLM_BETTER"),
    (-0.2, -0.001, "CHEAP_BETTER"), (-1e-12, -1e-12, "CHEAP_BETTER"),
    (-0.01, 0.01, "MATCHED"),                    # the interval exactly at the registered band
    (0.0, 0.0, "MATCHED"), (-0.005, 0.0, "MATCHED"), (0.0, 0.005, "MATCHED"), (-0.01, 0.0, "MATCHED"), (0.0, 0.01, "MATCHED"),
    (0.0, 0.02, "INCONCLUSIVE"),                 # lower bound exactly 0: not above 0
    (-0.02, 0.0, "INCONCLUSIVE"),                # upper bound exactly 0: not below 0
    (-0.0100001, 0.005, "INCONCLUSIVE"), (-0.005, 0.0100001, "INCONCLUSIVE"), (-0.02, 0.02, "INCONCLUSIVE"),
    (0.002, 0.008, "LLM_BETTER"),                # inside the band, but the lower bound is above 0: checked first
    (-0.008, -0.002, "CHEAP_BETTER"),            # inside the band, but the upper bound is below 0: checked second
    (float("nan"), 0.1, "INCONCLUSIVE"), (0.1, float("nan"), "INCONCLUSIVE"), (None, None, "INCONCLUSIVE"),
    (float("-inf"), float("inf"), "INCONCLUSIVE")])
def test_reading_of_one_control_signal_at_the_boundaries_and_in_the_registered_order(lo, hi, want):
    assert sc.read_signal(lo, hi) == want


def test_panel_label_rule_every_any_otherwise_and_registered_constants():
    L, C, M, I = "LLM_BETTER", "CHEAP_BETTER", "MATCHED", "INCONCLUSIVE"
    assert sc.read_panel([L, L, L, L]) == "ADDS"
    assert sc.read_panel([L, L, L, I]) == "MIXED" and sc.read_panel([I, I, I, I]) == "MIXED"
    assert sc.read_panel([L, L, L, M]) == "CHEAP_SUFFICES" and sc.read_panel([L, C, L, L]) == "CHEAP_SUFFICES"
    assert sc.read_panel([M, M, M, M]) == "CHEAP_SUFFICES" and sc.read_panel([I, M, I, I]) == "CHEAP_SUFFICES"
    assert sc.read_panel([I, L, L, C]) == "CHEAP_SUFFICES"          # any MATCHED / CHEAP_BETTER, whatever else
    assert sc.read_panel([]) == "MIXED"                             # nothing to read is not ADDS
    # the registered constants of the addendum, and nothing else
    assert (sc.REG_N_BOOT, sc.REG_SEED, sc.BAND, sc.QUESTION) == (2000, 0, 0.01, "next")
    assert na.COVERAGE[na.HALF] == 0.5 and tuple(na.UTILS) == ("ndcg10", "hr1")
    assert sc.SIGNAL_LABELS == ("LLM_BETTER", "CHEAP_BETTER", "MATCHED", "INCONCLUSIVE")       # the order of checking
    assert sc.PANEL_LABELS == ("ADDS", "CHEAP_SUFFICES", "MIXED", "NOT_RUN")
    assert sc.CONTROLS == CONTROLS and set(sc.SIGNALS) == set(CONTROLS) | {"p_max"}


# ========================================================================= 2. control signals against hand computations
GROUP_NAMES = ("head", "mid", "tail")


def tiny_rows(spec, domain, N=12):
    """Rows from explicit specs: (n_head, n_mid, n_tail, history, positive_group, positive_slot). Candidates are the first
    items h0.. / m0.. / t0.. of the three groups (heads first), so an item always has the same group."""
    rows = []
    for e, (nh, nm, nt, hist, pg, ps) in enumerate(spec):
        assert nh + nm + nt == N
        cands = [f"h{i}" for i in range(nh)] + [f"m{i}" for i in range(nm)] + [f"t{i}" for i in range(nt)]
        groups = ["head"] * nh + ["mid"] * nm + ["tail"] * nt
        first = {"head": 0, "mid": nh, "tail": nh + nm}[pg]
        rows.append({"source_event_id": f"{domain}U{e}::{1000 + e}", "user_id": f"{domain}U{e}", "history_item_ids": list(hist),
                     "candidate_item_ids": cands, "candidate_popularity_groups": groups,
                     "positive_item_index": first + ps, "num_candidates": N, "split_name": "test"})
    return rows


# TEST: (heads, mids, tails, history, positive group, slot in the group). The positives are tail wherever there is a tail: a
# popularity temperature fitted on these TEST events would sit at the lower bound, the one fitted on VALID is interior.
TINY_TEST = [
    (3, 4, 5, ["h0", "m0", "t0"], "tail", 1),                       # e0  pattern A
    (0, 0, 12, [], "tail", 0),                                      # e1  no history, no LLM scores (censored below)
    (12, 0, 0, ["h0", "h1", "h2", "h3", "h4"], "head", 0),          # e2  all head
    (2, 2, 8, ["x_unknown"], "tail", 0),                            # e3  unmapped history only
    (6, 3, 3, ["t0", "t1", "x_unknown", "t2"], "tail", 0),          # e4  one unmapped item among three tails
    (3, 4, 5, ["m0", "m1", "h0", "h0"], "tail", 2),                 # e5  pattern A again (a tie with e0), a repeated item
    (4, 4, 4, ["h5", "m2", "m3", "t4", "t5"], "tail", 1),           # e6
    (2, 0, 10, ["h0", "h1", "h2", "m0", "m1", "t0"], "tail", 3),    # e7
]
TINY_TEST_RANKS = [1, 7, 3, 12, 5, 2, 10, 4]                        # e1 is censored: its rank is never used
# VALID: ten events; positives in all three groups so that the popularity temperature is interior; long histories of tail items
# and a high rank of the positive go together (hist_len positive, profile_pop negative, pop_conf negative with respect to NDCG@10)
TINY_VALID = [
    (3, 4, 5, ["t0", "t1", "t2", "t3", "t4", "t5"], "head", 0),     # v0
    (3, 4, 5, ["m0", "m1", "t0", "t1", "t2"], "mid", 1),            # v1
    (2, 2, 8, ["t0", "t1", "t2"], "tail", 0),                       # v2  (no LLM scores)
    (6, 3, 3, ["m0", "m1", "m2", "t0"], "head", 2),                 # v3
    (4, 4, 4, ["h0", "m0", "t0"], "mid", 0),                        # v4
    (1, 5, 6, [], "tail", 1),                                       # v5  (no history)
    (2, 0, 10, ["h0", "h1"], "head", 1),                            # v6
    (5, 5, 2, ["h0"], "tail", 0),                                   # v7  (no LLM scores)
    (3, 3, 6, ["h0", "h1", "m0"], "mid", 2),                        # v8
    (0, 6, 6, ["h0"], "mid", 4),                                    # v9
]
TINY_VALID_RANKS = [1, 2, 9, 3, 6, 12, 8, 8, 9, 10]
TINY_VALID_CENSORED = (2, 7)


def tiny_world(tmp, valid_ranks=None, test_ranks=None):
    """The hand-built panel pair above on disk, with explicit LLM ranks of the positives (distinct logits, no ties)."""
    rows, vrows = tiny_rows(TINY_TEST, "tt"), tiny_rows(TINY_VALID, "tv")
    pos = np.array([r["positive_item_index"] for r in rows])
    pv = np.array([r["positive_item_index"] for r in vrows])
    tr, vr = test_ranks or TINY_TEST_RANKS, valid_ranks or TINY_VALID_RANKS
    L, Lv = scores_with_ranks(pos, tr, 12, seed=1), scores_with_ranks(pv, vr, 12, seed=2)
    cens = {("next", 1, c): 2 for c in range(12)}                    # e1: no scored candidate
    censv = {("next", e, c): 2 for e in TINY_VALID_CENSORED for c in range(12)}
    w = build_world(tmp, "tiny", L=L, pos=pos, grp=None, Lv=Lv, pv=pv, gv=None, rows=rows, vrows=vrows, censored_test=cens,
                    censored_valid=censv)
    w.test_ranks, w.valid_ranks = tr, vr
    return w


def test_hist_len_is_the_number_of_history_items_of_the_event(tmp_path):
    w = tiny_world(tmp_path)
    panel = na.load_panel(w.panel_test)
    assert sc.hist_len_signal(panel).tolist() == [3, 0, 5, 1, 4, 4, 5, 6]        # every item, the unmapped and repeated ones too
    assert sc.hist_len_signal(panel).dtype == float
    w2 = build_world(tmp_path / "w2", E=20, hist_max=40, hist_test=list(range(20)), seed=3)
    assert sc.hist_len_signal(na.load_panel(w2.panel_test)).tolist() == list(range(20))     # not capped at the LLM's window of 5


def test_profile_pop_is_the_mean_group_score_of_the_mapped_history_items(tmp_path):
    w = tiny_world(tmp_path)
    panel = na.load_panel(w.panel_test)
    prof, diag = na.user_profile(panel)
    want = [(2 + 1 + 0) / 3, np.nan, 2.0, np.nan, 0.0, (1 + 1 + 2 + 2) / 4, (2 + 1 + 1 + 0 + 0) / 5, (2 + 2 + 2 + 1 + 1 + 0) / 6]
    np.testing.assert_allclose(prof, want, rtol=0, atol=1e-15)
    assert diag["n_events_without_mapped_history"] == 2                         # e1 (empty) and e3 (unmapped only)
    # the module counts the exclusion among the events it serves: e1 has no scored candidate, so only e3 is excluded for profile_pop
    doc = run(w, n_boot=20)
    c = doc["counts"]
    assert c["n_events_panel_file"] == 8 and c["n_events_no_scored_candidate"] == 1 and c["n_events_serving_set"] == 7
    assert c["n_events_without_mapped_history"] == 1 and c["n_events_serving_set_profile_pop"] == 6
    assert doc["signals"]["profile_pop"]["serving_set"]["n_events"] == 6 and doc["signals"]["profile_pop"]["serving_set"]["n_excluded"] == 1
    for s in ("p_max", "random", "hist_len", "pop_conf"):
        assert doc["signals"][s]["serving_set"]["n_events"] == 7 and doc["signals"][s]["serving_set"]["n_excluded"] == 0
    assert doc["counts"]["history"]["n_events_without_mapped_history"] == 2


def test_popularity_scores_are_the_group_scores_head2_mid1_tail0(tmp_path):
    w = tiny_world(tmp_path)
    panel = na.load_panel(w.panel_test)
    ps = sc.popularity_scores(panel)
    assert ps.shape == panel.grp.shape and np.isfinite(ps).all()
    assert ps[0].tolist() == [2.0] * 3 + [1.0] * 4 + [0.0] * 5 and ps[1].tolist() == [0.0] * 12 and ps[2].tolist() == [2.0] * 12
    # a padded candidate (an event with fewer candidates than the longest one) has no score
    rows = tiny_rows([(3, 4, 5, [], "head", 0)], "pp") + tiny_rows([(2, 2, 6, [], "head", 0)], "pq", N=10)
    write_jsonl(tmp_path / "pad.jsonl", rows)
    p2 = na.load_panel(tmp_path / "pad.jsonl")
    s2 = sc.popularity_scores(p2)
    assert np.isnan(s2[1, 10:]).all() and np.isfinite(s2[1, :10]).all()


def test_pop_conf_is_the_top_probability_of_the_popularity_softmax_with_the_temperature_fitted_on_valid(tmp_path):
    w = tiny_world(tmp_path)
    panel, vpanel = na.load_panel(w.panel_test), na.load_panel(w.panel_valid)
    Sv = [np.repeat([2.0, 1.0, 0.0], [nh, nm, nt]) for nh, nm, nt, *_ in TINY_VALID]
    pv = [r["positive_item_index"] for r in w.vrows]
    beta = expected_beta(Sv, pv)
    assert 1e-3 < beta < 10.0                                                      # an interior optimum: the fit is informative
    fit = sc.fit_pop_temperature(vpanel)
    assert fit["beta"] == pytest.approx(beta, rel=1e-9) and fit["bound_hit"] is None
    assert fit["n_events_used"] == 10                                              # ALL VALID events, with or without LLM scores
    got = sc.pop_conf_signal(panel, fit["beta"])
    for e, (nh, nm, nt, *_ ) in enumerate(TINY_TEST):
        s = np.repeat([2.0, 1.0, 0.0], [nh, nm, nt])
        z = np.exp(beta * s).sum()
        assert got[e] == pytest.approx(np.exp(beta * s.max()) / z, rel=1e-9, abs=0)    # ONE candidate's probability, not the group mass
    assert got[1] == pytest.approx(1 / 12) and got[2] == pytest.approx(1 / 12)      # all tail / all head: uniform
    assert got[0] == got[5]                                                         # equal scores, equal probability: a tie between events
    assert got[0] == pytest.approx(np.exp(2 * beta) / (3 * np.exp(2 * beta) + 4 * np.exp(beta) + 5))
    assert len(set(np.round(got, 12))) == 6                                         # the ties: e0 = e5 and e1 = e2 (uniform lists)
    # the module records the same temperature, and fits it on the VALID events (not on the TEST events)
    doc = run(w, n_boot=20)
    assert doc["temperature"]["pop_conf"]["beta"] == pytest.approx(beta, rel=1e-9)
    # the TEST events have their own optimum (their positives are tail): a fit on them would be another temperature
    t_beta = expected_beta([np.repeat([2.0, 1.0, 0.0], [nh, nm, nt]) for nh, nm, nt, *_ in TINY_TEST],
                           [r["positive_item_index"] for r in w.rows])
    assert abs(t_beta - beta) > 0.1 and doc["temperature"]["pop_conf"]["beta"] != pytest.approx(t_beta, abs=0.05)


def test_pop_conf_fit_set_is_every_valid_event_not_only_those_with_llm_scores(tmp_path):
    w = tiny_world(tmp_path)
    keep = [i for i in range(10) if i not in TINY_VALID_CENSORED]
    Sv = [np.repeat([2.0, 1.0, 0.0], [nh, nm, nt]) for nh, nm, nt, *_ in TINY_VALID]
    pv = [r["positive_item_index"] for r in w.vrows]
    b_all, b_scored = expected_beta(Sv, pv), expected_beta([Sv[i] for i in keep], [pv[i] for i in keep])
    assert abs(b_all - b_scored) > 0.05                                             # the two fit sets give different temperatures
    doc = run(w, n_boot=20)
    assert doc["temperature"]["pop_conf"]["n_events_used"] == 10 == doc["counts"]["n_valid_events_pop_temperature"]
    assert doc["counts"]["n_valid_events"] == 10 and doc["counts"]["n_valid_events_with_llm_scores"] == 8
    assert doc["temperature"]["pop_conf"]["beta"] == pytest.approx(b_all, rel=1e-9)
    assert doc["temperature"]["p_max"]["n_events_used"] == 8                        # the LLM temperature: events with a scored positive


# ======================================================================================== 3. directions (VALID only)
@pytest.mark.parametrize("x,y,want", [
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1), ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], -1),
    ([1, 2, 3, 4, 5], [1, 3, 5, 3, 1], 1),                          # an exact zero correlation reads positive
    ([1, 1, 1, 1, 1], [1, 2, 3, 4, 5], 1), ([1, 2, 3, 4, 5], [2, 2, 2, 2, 2], 1),       # constant: undefined, positive
    ([1, 2], [2, 1], 1),                                            # fewer than three pairs: undefined, positive
    ([1, 2, 2, 3, 3, 3], [3, 2, 2, 1, 1, 1], -1),                   # ties
    ([1, np.nan, 3, 4, 5, 6], [6, 5, 4, 3, 2, np.nan], -1)])       # non-finite pairs are dropped
def test_direction_is_the_sign_of_the_spearman_correlation_zero_and_undefined_read_positive(x, y, want):
    d = sc.direction_of(x, y)
    assert d["value"] == want and d["source"] == "valid_spearman"
    ok = [(a, b) for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)]
    ref = spearman_ref([a for a, _ in ok], [b for _, b in ok]) if len(ok) >= 3 else float("nan")
    if math.isnan(ref):
        assert d["undefined"] and math.isnan(d["spearman"])
    else:
        assert not d["undefined"] and d["spearman"] == pytest.approx(ref, abs=1e-12)
    assert d["n_valid_pairs"] == len(ok)
    if x == [1, 2, 3, 4, 5] and y == [1, 3, 5, 3, 1]:
        assert d["spearman"] == 0.0 and not d["undefined"]          # exactly zero, not merely small


def test_directions_of_the_tiny_world_equal_the_independent_spearman_of_valid_arrays(tmp_path):
    w = tiny_world(tmp_path)
    doc = run(w, n_boot=20)
    keep = [i for i in range(10) if i not in TINY_VALID_CENSORED]
    nd = ndcg_of_ranks(TINY_VALID_RANKS)
    vpanel = na.load_panel(w.panel_valid)
    prof_v, _ = na.user_profile(vpanel)
    beta = expected_beta([np.repeat([2.0, 1.0, 0.0], [nh, nm, nt]) for nh, nm, nt, *_ in TINY_VALID], [r["positive_item_index"] for r in w.vrows])
    popc = []
    for nh, nm, nt, *_ in TINY_VALID:
        s = np.repeat([2.0, 1.0, 0.0], [nh, nm, nt])
        popc.append(np.exp(beta * s.max()) / np.exp(beta * s).sum())
    hist = [len(r["history_item_ids"]) for r in w.vrows]
    signals = {"hist_len": hist, "profile_pop": list(prof_v), "pop_conf": popc}
    expect = {}
    for s, v in signals.items():
        pairs = [(v[i], nd[i]) for i in keep if np.isfinite(v[i])]
        rho = spearman_ref([a for a, _ in pairs], [b for _, b in pairs])
        expect[s] = (-1 if rho < 0 else 1, rho, len(pairs))
    for s, (d, rho, n) in expect.items():
        got = doc["direction"][s]
        assert got["value"] == d and got["spearman"] == pytest.approx(rho, abs=1e-9) and got["n_valid_pairs"] == n and got["source"] == "valid_spearman"
    assert {d for d, _, _ in expect.values()} == {1, -1}            # the fixture exercises both signs: a flipped or ignored sign fails
    assert doc["direction"]["p_max"] == {"value": 1, "source": "registered"}
    assert doc["direction"]["random"] == {"value": 1, "source": "as_is"}
    # profile_pop: the VALID events without a mapped history item are dropped from the correlation
    assert expect["profile_pop"][2] < len(keep)


def _directions(doc):
    return {s: (doc["direction"][s]["value"], doc["direction"][s].get("spearman")) for s in doc["direction"]}


def test_directions_never_depend_on_test_labels_scores_or_positive_indices(tmp_path):
    base = build_world(tmp_path / "a", E=70, n_valid=60, seed=21)
    d0 = run(base, "a.json")
    rng = np.random.default_rng(5)
    variants = {}
    # (1) the TEST positive indices permuted (the panel keeps its candidates; the scores file's label column follows)
    pos2 = rng.permutation(base.pos)
    variants["positives permuted"] = build_world(tmp_path / "b", E=70, n_valid=60, seed=21, L=base.L, grp=base.grp, pos=pos2,
                                                 Lv=base.Lv, gv=base.gv, pv=base.pv, rows=None, vrows=None)
    # (2) the TEST scores replaced (other logits, other ties)
    variants["scores replaced"] = build_world(tmp_path / "c", E=70, n_valid=60, seed=21, L=rng.normal(size=base.L.shape),
                                              grp=base.grp, pos=base.pos, Lv=base.Lv, gv=base.gv, pv=base.pv)
    # (3) scores reversed: the best candidate is now the worst
    variants["scores reversed"] = build_world(tmp_path / "d", E=70, n_valid=60, seed=21, L=-base.L, grp=base.grp, pos=base.pos,
                                              Lv=base.Lv, gv=base.gv, pv=base.pv)
    # the control signals are functions of the panel file and the VALID events: same candidate lists / histories in every variant
    for name, w in variants.items():
        assert [r["history_item_ids"] for r in w.rows] == [r["history_item_ids"] for r in base.rows], name
        doc = run(w, "v.json")
        assert _directions(doc) == _directions(d0), name
        assert doc["signals"]["hist_len"]["serving_set"]["n_distinct_values"] == d0["signals"]["hist_len"]["serving_set"]["n_distinct_values"]
    # sanity: the TEST-dependent numbers DO change, so the comparison above is not vacuous
    doc_r = run(variants["scores reversed"], "v2.json")
    assert doc_r["signals"]["p_max"]["gain50"]["ndcg10"]["est"] != d0["signals"]["p_max"]["gain50"]["ndcg10"]["est"]
    rho_test = {s: stats.spearman(*_test_pairs(variants["scores reversed"], s)) for s in sc.VALID_DIRECTION}
    assert any((rho < 0) != (d0["direction"][s]["value"] < 0) for s, rho in rho_test.items() if np.isfinite(rho)), \
        "fixture: the TEST relationship must disagree with the VALID one for at least one signal"


def _test_pairs(w, s):
    """(signal, NDCG@10) over the events of a world's TEST panel, from the arrays (the relationship a TEST-based rule would use)."""
    panel = na.load_panel(w.panel_test)
    rank = np.array([stats.tie_aware_rank(w.L[e], int(w.pos[e])) for e in range(panel.E)])
    nd = metrics.ndcg_from_rank(rank, 10)
    if s == "hist_len":
        sig = sc.hist_len_signal(panel)
    elif s == "profile_pop":
        sig, _ = na.user_profile(panel)
    else:
        sig = sc.pop_conf_signal(panel, sc.fit_pop_temperature(na.load_panel(w.panel_valid))["beta"])
    return sig, nd


def test_flipping_the_valid_relationship_flips_the_direction(tmp_path):
    E, N, nv = 80, 15, 60
    rng = np.random.default_rng(2)
    hv = rng.integers(0, 8, nv)
    L, grp, pos, _ = synth_arrays(E, N, 4)
    _, gv, pv, _ = synth_arrays(nv, N, 104)
    # NDCG@10 of a VALID event decreases with its history length (long history: the positive ranks first) ...
    ranks_up = np.clip(9 - hv + rng.integers(0, 2, nv), 1, 12)
    w_up = build_world(tmp_path / "up", L=L, grp=grp, pos=pos, Lv=scores_with_ranks(pv, ranks_up, N), gv=gv, pv=pv, hist_valid=hv.tolist(), seed=4)
    # ... and the same VALID events with the relationship reversed
    ranks_dn = np.clip(1 + hv + rng.integers(0, 2, nv), 1, 12)
    w_dn = build_world(tmp_path / "dn", L=L, grp=grp, pos=pos, Lv=scores_with_ranks(pv, ranks_dn, N), gv=gv, pv=pv, hist_valid=hv.tolist(), seed=4)
    d_up, d_dn = run(w_up, "up.json"), run(w_dn, "dn.json")
    assert d_up["direction"]["hist_len"]["value"] == 1 and d_up["direction"]["hist_len"]["spearman"] > 0.5
    assert d_dn["direction"]["hist_len"]["value"] == -1 and d_dn["direction"]["hist_len"]["spearman"] < -0.5
    # the serving set follows: with a negative direction the SHORTEST histories are served first (the gain is that of -hist_len)
    panel = na.load_panel(w_dn.panel_test)
    rank = np.array([stats.tie_aware_rank(L[e], int(pos[e])) for e in range(E)])
    nd = metrics.ndcg_from_rank(rank, 10)
    hl = sc.hist_len_signal(panel)
    assert d_dn["signals"]["hist_len"]["gain50"]["ndcg10"]["est"] == pytest.approx(ref_gain50(-hl, nd), abs=1e-12)
    assert d_up["signals"]["hist_len"]["gain50"]["ndcg10"]["est"] == pytest.approx(ref_gain50(hl, nd), abs=1e-12)
    # a constant VALID utility (all LLM scores tied) has no correlation: positive, flagged undefined
    w_c = build_world(tmp_path / "const", L=L, grp=grp, pos=pos, Lv=np.zeros((nv, N)), gv=gv, pv=pv, hist_valid=hv.tolist(), seed=4)
    d_c = run(w_c, "c.json")
    for s in sc.VALID_DIRECTION:
        assert d_c["direction"][s]["value"] == 1 and d_c["direction"][s]["undefined"] and d_c["direction"][s]["spearman"] is None


# ===================================== 4. the registered p_max and random numbers are the audit's own (to 1e-12)
def assert_ci_equal(a, b, tol=1e-12, where=""):
    for k in ("est", "lo", "hi"):
        x, y = a[k], b[k]
        if x is None or y is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(y, float) and math.isnan(y)):
            assert (x is None or math.isnan(x)) and (y is None or math.isnan(y)), (where, k, x, y)
        else:
            assert x == pytest.approx(y, abs=tol, rel=0), (where, k)
    assert a["n"] == b["n"] and a["n_boot"] == b["n_boot"], where


def assert_registered_blocks_equal(audit_seg, doc, tol=1e-12):
    """p_max and random of `doc` against the audit's section D for the same segment: gain at 50%, the curve at every coverage
    point, AURC, mean utility, the niche served shares and their difference, both utilities."""
    D = audit_seg["questions"]["next"]["D_selective_serving"]
    assert D["n_events"] == doc["counts"]["n_events_serving_set"]
    for s in ("p_max", "random"):
        A, M = D["signals"][s], doc["signals"][s]
        assert M["serving_set"]["n_events"] == D["n_events"]
        for u in ("ndcg10", "hr1"):
            assert_ci_equal(A["gain_at_50_vs_full"][u], M["gain50"][u], tol, f"{s}/{u}/gain50")
            assert_ci_equal(A["aurc"][u], M["aurc"][u], tol, f"{s}/{u}/aurc")
            assert_ci_equal(A["mean_utility"][u], M["mean_utility"][u], tol, f"{s}/{u}/mean_utility")
            assert [c["coverage"] for c in A["curve"][u]] == [c["coverage"] for c in M["curve"][u]] == list(na.COVERAGE)
            for ca, cm in zip(A["curve"][u], M["curve"][u]):
                assert_ci_equal(ca, cm, tol, f"{s}/{u}/curve@{ca['coverage']}")
        assert M["niche"]["quintile_sizes"] == D["quintile_sizes"]
        for j in range(na.N_QUINT):
            assert_ci_equal(A["niche"]["served_share"][j], M["niche"]["served_share"][j], tol, f"{s}/served_share/{j}")
        assert (A["niche"]["niche_bin"], A["niche"]["mainstream_bin"]) == (M["niche"]["niche_bin"], M["niche"]["mainstream_bin"])
        assert_ci_equal(A["niche"]["niche_minus_mainstream_served_share"], M["niche"]["niche_minus_mainstream_served_share"], tol,
                        f"{s}/niche_minus_mainstream")


def test_p_max_and_random_reproduce_the_audit_run_on_the_same_inputs(world):
    audit_seg = world.audit["segments"]["all"]
    assert world.doc["panel"] == {"segment": "all", "role": "all", "event_range": [1, 96], "n_events": 96}
    assert_registered_blocks_equal(audit_seg, world.doc)
    # the unscored candidates of the fixture are real: the audit counts them too
    assert world.audit["data"]["by_question"]["next"]["n_rows_unscored"] == 4 == world.doc["counts"]["scores_test"]["by_question"]["next"]["dropped_nonfinite_or_censored"]
    # the temperature of p_max is the audit's
    for k in ("T", "beta", "n_events_used", "nll_T1", "nll_fit"):
        assert world.doc["temperature"]["p_max"][k] == pytest.approx(world.audit["temperature"]["next"][k], abs=1e-12)


def test_the_random_signal_follows_the_seed_as_the_audit_does(world):
    for seed in (3, 11):
        doc = run(world, f"seed{seed}.json", seed=seed)
        assert_registered_blocks_equal(audit_run(world, seed=seed)["segments"]["all"], doc)
    other = run(world, "seed3.json", seed=3)
    assert other["signals"]["random"]["gain50"]["ndcg10"]["est"] != world.doc["signals"]["random"]["gain50"]["ndcg10"]["est"]
    assert other["signals"]["p_max"]["gain50"]["ndcg10"]["est"] == world.doc["signals"]["p_max"]["gain50"]["ndcg10"]["est"]


def test_sports_quarantine_segment_is_not_analysed_and_the_main_segment_is_the_audits(tmp_path):
    w = build_world(tmp_path, "sports", E=52, N=15, n_valid=36, seed=13)
    doc = run(w, quarantine_n=14)
    assert doc["panel"] == {"segment": "events_15_52", "role": "main", "event_range": [15, 52], "n_events": 38}
    audit = audit_run(w, quarantine_n=14)
    assert list(audit["segments"]) == ["events_1_14", "events_15_52"]
    assert_registered_blocks_equal(audit["segments"]["events_15_52"], doc)
    assert "events_1_14" not in json.dumps(doc) and doc["layout"]["quarantine_n"] == 14
    # the quarantine events change nothing: rewriting their scores leaves every number of the main segment as it was
    L2 = w.L.copy()
    L2[:14] = -L2[:14]
    w2 = build_world(tmp_path / "w2", "sports", L=L2, grp=w.grp, pos=w.pos, Lv=w.Lv, gv=w.gv, pv=w.pv, seed=13)
    doc2 = run(w2, quarantine_n=14)
    assert doc2["panel"] == doc["panel"]
    for s in ("hist_len", "pop_conf"):                           # their direction and the signals inside the segment are unchanged
        assert doc2["direction"][s] == doc["direction"][s]
        assert doc2["signals"][s]["gain50"] == doc["signals"][s]["gain50"]
    # with the default quarantine (1,000 events) a 52-event sports panel is all quarantine: nothing registered to analyse
    with pytest.raises(sc.ControlError, match="no registered segment"):
        run(w, "q.json")


def test_second_backbone_layout_one_segment_named_after_the_scores_role(tmp_path):
    w = build_world(tmp_path, "home", E=44, N=15, n_valid=30, seed=17, test_role="test1001_3000", valid_role="valid500")
    doc = run(w, segments="single", first_event=1001, kind="llama_z2")
    assert doc["panel"] == {"segment": "test1001_3000", "role": "all", "event_range": [1001, 1044], "n_events": 44}
    assert doc["layout"] == {"segments": "single", "test_role": "test1001_3000", "valid_role": "valid500", "first_event": 1001,
                             "quarantine_n": None}
    audit = audit_run(w, segments="single", first_event=1001)
    assert_registered_blocks_equal(audit["segments"]["test1001_3000"], doc)
    assert doc["counts"]["n_valid_events"] == 30 and doc["temperature"]["p_max"]["n_events_used"] == 30
    with pytest.raises(ValueError, match="1-based"):
        run(w, "bad.json", segments="single", first_event=0)
    with pytest.raises(ValueError, match="first_event"):
        run(w, "bad.json", first_event=1001)                      # first_event belongs to --segments single
    with pytest.raises(ValueError, match="quarantine_n"):
        run(w, "bad.json", segments="single", quarantine_n=5)


def test_audit_json_check_reproduces_or_refuses(world, tmp_path):
    aj = tmp_path / "audit_toys.json"
    aj.write_text(json.dumps(stats.strict_json(world.audit), allow_nan=False), encoding="utf-8")
    doc = run(world, "chk.json", audit_json=aj)
    chk = doc["meta"]["audit_consistency"]
    assert chk["checked"] is True and chk["max_abs_diff"] == 0.0 and chk["segment"] == "all" and chk["n_values_compared"] > 100
    assert chk["signals"] == ["p_max", "random"] and chk["tolerance"] == 1e-12
    assert world.doc["meta"]["audit_consistency"] == {"checked": False}
    # a registered number that differs by 1e-9, a different seed, a missing segment: refused with exit status 1, nothing written
    tamper = json.loads(aj.read_text(encoding="utf-8"))
    tamper["segments"]["all"]["questions"]["next"]["D_selective_serving"]["signals"]["p_max"]["gain_at_50_vs_full"]["ndcg10"]["est"] += 1e-9
    seed_doc = json.loads(aj.read_text(encoding="utf-8"))
    seed_doc["seed"] = 9
    seg_doc = json.loads(aj.read_text(encoding="utf-8"))
    seg_doc["segments"] = {"events_1_1": seg_doc["segments"]["all"]}
    for name, d in (("tamper", tamper), ("seed", seed_doc), ("segment", seg_doc)):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises(sc.ControlError) as e:
            run(world, f"{name}_out.json", audit_json=p)
        assert e.value.code == 1, name
        assert not (world.tmp / f"{name}_out.json").exists(), name
    with pytest.raises(FileNotFoundError):
        run(world, "nf.json", audit_json=tmp_path / "missing.json")


# ======================================== 5. serving numbers against brute force; the pairing of the contrasts
def tiny_expected_signals(w):
    """The three controls on the TEST events that have a scored candidate, from the hand-derived formulas (not from the module)."""
    beta = expected_beta([np.repeat([2.0, 1.0, 0.0], [nh, nm, nt]) for nh, nm, nt, *_ in TINY_VALID], [r["positive_item_index"] for r in w.vrows])
    ev = [0, 2, 3, 4, 5, 6, 7]                                          # e1 has no scored candidate
    hist = [3, 0, 5, 1, 4, 4, 5, 6]
    prof = [1.0, np.nan, 2.0, np.nan, 0.0, 1.5, 0.8, 8 / 6]
    pc = []
    for nh, nm, nt, *_ in TINY_TEST:
        s = np.repeat([2.0, 1.0, 0.0], [nh, nm, nt])
        pc.append(np.exp(beta * s.max()) / np.exp(beta * s).sum())
    return ev, np.array(hist, float)[ev], np.array(prof, float)[ev], np.array(pc)[ev]


def test_gain_at_50_equals_the_expectation_over_all_tie_breakings_on_the_tiny_world(tmp_path):
    w = tiny_world(tmp_path)
    doc = run(w, n_boot=30)
    ev, hist, prof, pc = tiny_expected_signals(w)
    ranks = np.array(TINY_TEST_RANKS)[ev]
    util = {"ndcg10": ndcg_of_ranks(ranks), "hr1": (ranks == 1).astype(float)}
    mine = {"hist_len": hist, "profile_pop": prof, "pop_conf": pc}
    assert len(ev) == 7                                                 # odd: k = round(3.5) = 4 (half to even), not 3
    for s, v in mine.items():
        d = doc["direction"][s]["value"]
        keep = np.isfinite(v)
        for u in ("ndcg10", "hr1"):
            want = brute_gain50(list(d * v[keep]), util[u][keep])
            assert doc["signals"][s]["gain50"][u]["est"] == pytest.approx(want, abs=1e-12), (s, u)
            assert ref_gain50(d * v[keep], util[u][keep]) == pytest.approx(want, abs=1e-12)       # the two references agree
        assert doc["signals"][s]["serving_set"]["n_events"] == int(keep.sum())
        assert doc["signals"][s]["serving_set"]["n_served_at_50"] == {7: 4, 6: 3}[int(keep.sum())]
    assert doc["signals"]["p_max"]["serving_set"]["n_served_at_50"] == 4 == doc["signals"]["random"]["serving_set"]["n_served_at_50"]
    # heavy ties are real here: hist_len has 5 distinct values on 7 events, pop_conf has a tie (e0 = e5), profile_pop none on 6 events
    assert [doc["signals"][s]["serving_set"]["n_distinct_values"] for s in ("hist_len", "pop_conf", "profile_pop")] == [5, 6, 6]
    assert doc["signals"]["hist_len"]["serving_set"]["n_excluded"] == 0
    # the contrast is the difference of the two gains, and the contrast of p_max with itself does not exist
    for s in CONTROLS:
        for u in ("ndcg10", "hr1"):
            gp, gs = doc["signals"]["p_max"]["gain50"][u]["est"], doc["signals"][s]["gain50"][u]["est"]
            assert doc["signals"][s]["delta_vs_p_max"][u]["est"] == pytest.approx(gp - gs, abs=1e-12)
    assert "delta_vs_p_max" not in doc["signals"]["p_max"] and "reading" not in doc["signals"]["p_max"]


def explicit_resample_reference(w, doc, B, seed=0):
    """The arrays of every signal (p_max and random from the audit's question_arrays, the controls from the panel file), the
    utilities, the serving columns of each signal and its direction, for explicit resamples with metrics.risk_coverage."""
    panel = na.load_panel(w.panel_test)
    E = panel.E
    qa = na.question_arrays(w.L_seen, panel, doc["temperature"]["p_max"]["beta"], np.random.default_rng([seed, 1]).random(E))
    m = qa["valid"]
    prof, _ = na.user_profile(panel)
    beta_pop = doc["temperature"]["pop_conf"]["beta"]
    sig = {"p_max": qa["p_max"], "random": qa["random"], "hist_len": sc.hist_len_signal(panel), "profile_pop": prof,
           "pop_conf": sc.pop_conf_signal(panel, beta_pop)}
    util = {"ndcg10": qa["ndcg10"], "hr1": qa["hr1"]}
    cols = {s: np.flatnonzero(m & (np.isfinite(prof) if s == "profile_pop" else True)) for s in sig}
    dire = {s: doc["direction"][s]["value"] for s in sig}
    return sig, util, cols, dire


def reference_gain_replicates(sig, util, cols, dire, boot, s, u):
    out = []
    for b in range(boot.n_boot):
        idx = boot.resample_idx(b, cols[s])
        x, y = dire[s] * sig[s][cols[s]][idx], util[u][cols[s]][idx]
        out.append(ref_gain50(x, y))
    return np.array(out)


def test_contrasts_are_computed_on_the_same_resamples_and_unpaired_resamples_would_fail(world):
    doc = world.doc
    sig, util, cols, dire = explicit_resample_reference(world, doc, B)
    n = doc["panel"]["n_events"]
    boot = na.EventBoot(n, B, 0)
    boot_other = na.EventBoot(n, B, 1)                                   # a different, independent stream
    assert len(cols["profile_pop"]) < len(cols["p_max"]) == n            # the restricted serving set is part of the check
    assert int(boot.W[:, cols["profile_pop"]].sum(axis=1).max()) > len(cols["profile_pop"])        # some resamples outweigh it
    for u in ("ndcg10", "hr1"):
        g_p = reference_gain_replicates(sig, util, cols, dire, boot, "p_max", u)
        for s in CONTROLS:
            g_s = reference_gain_replicates(sig, util, cols, dire, boot, s, u)
            mine = doc["signals"][s]
            ref_ci = stats.percentile_ci(g_p - g_s)
            assert (mine["delta_vs_p_max"][u]["lo"], mine["delta_vs_p_max"][u]["hi"]) == pytest.approx(ref_ci, abs=1e-10, rel=0), (s, u)
            ref_gain_ci = stats.percentile_ci(g_s)
            assert (mine["gain50"][u]["lo"], mine["gain50"][u]["hi"]) == pytest.approx(ref_gain_ci, abs=1e-10, rel=0), (s, u)
            assert mine["delta_vs_p_max"][u]["n"] == min(len(cols["p_max"]), len(cols[s])) and mine["delta_vs_p_max"][u]["n_boot"] == B
            # the planted bug: the control signal evaluated on ANOTHER set of resamples gives another interval, so the
            # comparison above is able to tell paired from unpaired contrasts
            g_s_unpaired = reference_gain_replicates(sig, util, cols, dire, boot_other, s, u)
            unpaired = stats.percentile_ci(g_p - g_s_unpaired)
            assert max(abs(unpaired[0] - ref_ci[0]), abs(unpaired[1] - ref_ci[1])) > 1e-4, (s, u)
            assert max(abs(unpaired[0] - mine["delta_vs_p_max"][u]["lo"]), abs(unpaired[1] - mine["delta_vs_p_max"][u]["hi"])) > 1e-4
    # AURC of the restricted serving set, replicate by replicate (the audit's engine alone cannot do this one)
    s, u = "profile_pop", "ndcg10"
    aurc = []
    for b in range(B):
        idx = boot.resample_idx(b, cols[s])
        aurc.append(metrics.risk_coverage(dire[s] * sig[s][cols[s]][idx], util[u][cols[s]][idx])[2])
    assert (doc["signals"][s]["aurc"][u]["lo"], doc["signals"][s]["aurc"][u]["hi"]) == pytest.approx(stats.percentile_ci(aurc), abs=1e-10, rel=0)


def test_a_serving_set_smaller_than_the_segment_is_exact_where_the_audit_engine_alone_would_fail():
    rng = np.random.default_rng(4)
    n, nv, Bn = 100, 80, 30
    cols = np.sort(rng.choice(n, nv, replace=False))
    sig = np.round(rng.normal(size=nv) * 3) / 3                           # ties
    util = {"ndcg10": rng.random(nv), "hr1": (rng.random(nv) < 0.4).astype(float)}
    quint = rng.integers(0, 5, nv)
    boot = na.EventBoot(n, Bn, 0)
    Wv = boot.W[:, cols]
    assert int(Wv.sum(axis=1).max()) > nv                                 # a resample with more weight than the set has events
    est, reps = sc.serve_signal("x", sig, util, quint, Wv)
    for b in range(Bn):
        idx = boot.resample_idx(b, cols)
        for u in ("ndcg10", "hr1"):
            _, cum, aurc = metrics.risk_coverage(sig[idx], util[u][idx])
            ks = np.minimum(np.maximum(1, np.rint(np.asarray(na.COVERAGE) * len(idx)).astype(int)), len(idx))
            assert reps["aurc"][u][b] == pytest.approx(aurc, abs=1e-10)
            assert reps["curve"][u][b] == pytest.approx(cum[ks - 1], abs=1e-10)
            assert reps["mean_u"][u][b] == pytest.approx(util[u][idx].mean(), abs=1e-12)
    g_est, g_reps = sc.gain50(est, reps, "ndcg10")
    assert g_est == pytest.approx(ref_gain50(sig, util["ndcg10"]), abs=1e-12)
    assert sc.n_served(nv) == 40 and sc.n_served(7) == 4 and sc.n_served(5) == 2 and sc.n_served(9) == 4 and sc.n_served(1) == 1
    assert sc.n_served(0) == 0


def test_every_signal_equals_a_plain_per_event_oracle_of_the_pipeline(world):
    """A re-computation of every point estimate from the raw arrays: utilities from stats.tie_aware_rank and the audit's top-1
    rule, the control signals from the panel file, serving by metrics.risk_coverage. p_max itself is the audit's (test 4 pins it
    to the audit's own run bit for bit; an independent softmax would break its exact ties, which float noise decides, differently)."""
    doc, L, pos = world.doc, world.L_seen, world.pos             # the logits as the loader reads them (censored = unscored)
    E, N = L.shape
    rank = np.array([stats.tie_aware_rank(L[e], int(pos[e])) for e in range(E)])
    ndcg = metrics.ndcg_from_rank(rank, 10)
    beta = doc["temperature"]["p_max"]["beta"]
    pmax_plain, hr1 = np.zeros(E), np.zeros(E)
    for e in range(E):
        s = L[e]
        ok = np.isfinite(s)
        p = np.exp(beta * (s[ok] - s[ok].max()))
        pmax_plain[e] = (p / p.sum()).max()
        top = int(np.flatnonzero(ok)[np.argmax(s[ok])])                  # first maximal candidate: the lower cand_idx
        hr1[e] = float(top == pos[e])
    panel = na.load_panel(world.panel_test)
    rnd = np.random.default_rng([0, 1]).random(E)
    qa = na.question_arrays(L, panel, beta, rnd)
    assert qa["p_max"] == pytest.approx(pmax_plain, abs=1e-12, rel=0) and (qa["ndcg10"] == ndcg).all() and (qa["hr1"] == hr1).all()
    prof, _ = na.user_profile(panel)
    hist = np.array([len(r["history_item_ids"]) for r in world.rows], float)
    pc = sc.pop_conf_signal(panel, doc["temperature"]["pop_conf"]["beta"])
    sigs = {"p_max": qa["p_max"], "random": rnd, "hist_len": hist, "profile_pop": prof, "pop_conf": pc}
    for s, v in sigs.items():
        keep = np.isfinite(v)
        d = doc["direction"][s]["value"]
        assert doc["signals"][s]["serving_set"]["n_events"] == int(keep.sum())
        for u, util in (("ndcg10", ndcg), ("hr1", hr1)):
            cov, cum, aurc = metrics.risk_coverage(d * v[keep], util[keep])
            assert doc["signals"][s]["gain50"][u]["est"] == pytest.approx(ref_gain50(d * v[keep], util[keep]), abs=1e-12), (s, u)
            assert doc["signals"][s]["aurc"][u]["est"] == pytest.approx(aurc, abs=1e-12), (s, u)
            assert doc["signals"][s]["mean_utility"][u]["est"] == pytest.approx(util[keep].mean(), abs=1e-12)
            ks = np.minimum(np.maximum(1, np.rint(np.asarray(na.COVERAGE) * keep.sum()).astype(int)), keep.sum())
            assert [c["est"] for c in doc["signals"][s]["curve"][u]] == pytest.approx(list(cum[ks - 1]), abs=1e-12)
    # the popularity profile quintiles and the served shares of one control signal, by an explicit expectation over tie-breaking
    quint = np.full(E, -1)
    quint[np.isfinite(prof)] = stats.rank_bins(prof[np.isfinite(prof)], 5)
    d = doc["direction"]["hist_len"]["value"]
    order = np.argsort(-(d * hist), kind="stable")
    k = max(1, int(round(0.5 * E)))
    served = np.zeros(E)
    i = 0
    while i < E:
        j = i
        while j + 1 < E and d * hist[order[j + 1]] == d * hist[order[i]]:
            j += 1
        served[order[i:j + 1]] = np.clip((k - i) / (j - i + 1), 0, 1)
        i = j + 1
    for q in range(5):
        if (quint == q).any():
            assert doc["signals"]["hist_len"]["niche"]["served_share"][q]["est"] == pytest.approx(served[quint == q].sum() / (quint == q).sum(), abs=1e-12)


# ======================================================= 6. labels end to end, NOT_RUN, the other CLI refusals
def world_adds(tmp, domain="toys", seed=0):
    """The LLM's confidence separates easy from hard events; the cheap signals know nothing about them."""
    L, grp, pos, _ = synth_confident(300, 15, seed, frac_easy=0.4)
    Lv, gv, pv, _ = synth_confident(100, 15, seed + 100, frac_easy=0.4)
    return build_world(tmp, domain, L=L, grp=grp, pos=pos, Lv=Lv, gv=gv, pv=pv, seed=seed)


def world_cheap(tmp, domain="home", seed=0):
    """The history length drives the rank of the positive; the logits of every event are a permutation of the same values, so the
    LLM's confidence is the same for all events up to float noise."""
    E, nv, N = 120, 80, 15
    rng = np.random.default_rng(seed)
    ht, hv = rng.integers(0, 8, E), rng.integers(0, 8, nv)
    _, grp, pos, _ = synth_arrays(E, N, seed)
    _, gv, pv, _ = synth_arrays(nv, N, seed + 100)
    L = scores_with_ranks(pos, np.clip(9 - ht + rng.integers(0, 2, E), 1, 12), N, seed=seed)
    Lv = scores_with_ranks(pv, np.clip(9 - hv + rng.integers(0, 2, nv), 1, 12), N, seed=seed + 1)
    return build_world(tmp, domain, L=L, grp=grp, pos=pos, Lv=Lv, gv=gv, pv=pv, hist_test=ht.tolist(), hist_valid=hv.tolist(), seed=seed)


def world_matched(tmp, domain="tools", seed=0):
    """Every LLM score is tied: the utility is the same for all events, so every gain is exactly 0."""
    E, nv, N = 60, 40, 15
    _, grp, pos, _ = synth_arrays(E, N, seed)
    _, gv, pv, _ = synth_arrays(nv, N, seed + 100)
    return build_world(tmp, domain, L=np.zeros((E, N)), grp=grp, pos=pos, Lv=np.zeros((nv, N)), gv=gv, pv=pv, seed=seed)


def world_mixed(tmp, domain="games", seed=0):
    """A weakly informative confidence on 40 events: every contrast is too wide to read."""
    E, nv, N = 40, 40, 15
    L, grp, pos, _ = synth_arrays(E, N, seed, signal=0.8, sure_spread=0.8)
    Lv, gv, pv, _ = synth_arrays(nv, N, seed + 100, signal=0.8, sure_spread=0.8)
    return build_world(tmp, domain, L=L, grp=grp, pos=pos, Lv=Lv, gv=gv, pv=pv, seed=seed)


@pytest.fixture(scope="module")
def labelled(tmp_path_factory):
    """Five panels of one panel kind with the labels ADDS, CHEAP_SUFFICES (a cheap signal reads CHEAP_BETTER), CHEAP_SUFFICES (all
    MATCHED), MIXED and NOT_RUN, each run once: {name: (world, parsed json, path)}."""
    tmp = tmp_path_factory.mktemp("labelled")
    out = {}
    for name, make in (("adds", world_adds), ("cheap", world_cheap), ("matched", world_matched), ("mixed", world_mixed)):
        w = make(tmp / name)
        doc = run(w, f"{name}.json")
        out[name] = (w, doc, w.tmp / f"{name}.json")
    w = build_world(tmp / "notrun", "books", E=30, N=15, n_valid=20, seed=5, with_valid=False)
    out["notrun"] = (w, run(w, "notrun.json"), w.tmp / "notrun.json")
    return out


def test_labels_end_to_end_and_every_label_is_the_rule_applied_to_the_stored_intervals(labelled):
    want = {"adds": ("ADDS", ["LLM_BETTER"] * 4), "matched": ("CHEAP_SUFFICES", ["MATCHED"] * 4),
            "mixed": ("MIXED", ["INCONCLUSIVE"] * 4)}
    for name, (label, readings) in want.items():
        doc = labelled[name][1]
        assert doc["label"] == label and list(doc["readings"].values()) == readings, name
        assert list(doc["readings"]) == list(CONTROLS)                      # the registered control signals, in the registered order
    cheap = labelled["cheap"][1]
    assert cheap["label"] == "CHEAP_SUFFICES" and cheap["readings"]["hist_len"] == "CHEAP_BETTER"
    assert cheap["signals"]["hist_len"]["delta_vs_p_max"]["ndcg10"]["hi"] < 0
    for name in ("adds", "cheap", "matched", "mixed"):
        doc = labelled[name][1]
        for s in CONTROLS:
            ci = doc["signals"][s]["delta_vs_p_max"]["ndcg10"]
            assert doc["signals"][s]["reading"] == doc["readings"][s] == sc.read_signal(ci["lo"], ci["hi"]), (name, s)
        assert doc["label"] == sc.read_panel(doc["readings"][s] for s in CONTROLS), name
    # ADDS: the confident half pays and every contrast has a positive lower bound
    adds = labelled["adds"][1]
    assert adds["signals"]["p_max"]["gain50"]["ndcg10"]["lo"] > 0.1
    assert all(adds["signals"][s]["delta_vs_p_max"]["ndcg10"]["lo"] > 0 for s in CONTROLS)
    # MATCHED: the contrast is exactly zero on every resample (a constant utility), the interval is [0, 0]
    for s in CONTROLS:
        ci = labelled["matched"][1]["signals"][s]["delta_vs_p_max"]["ndcg10"]
        assert (ci["est"], ci["lo"], ci["hi"]) == (0.0, 0.0, 0.0)


def test_the_reading_uses_ndcg10_not_hr1_and_the_cheap_signal_order_is_the_registered_one(labelled):
    doc = labelled["matched"][1]
    hr1 = doc["signals"]["random"]["delta_vs_p_max"]["hr1"]
    assert (hr1["lo"] > 0.01 or hr1["hi"] < -0.01 or hr1["lo"] < -0.01 or hr1["hi"] > 0.01)       # HR@1 would not read MATCHED
    assert doc["readings"]["random"] == "MATCHED"                                                   # but only NDCG@10 is read
    assert list(doc["signals"]) == list(sc.SIGNALS) == ["p_max", "random", "hist_len", "profile_pop", "pop_conf"]
    assert [doc["signals"][s]["kind"] for s in sc.SIGNALS] == ["llm", "null_reference", "control", "control", "control"]


def test_not_run_when_the_valid_score_file_is_missing(labelled):
    w, doc, path = labelled["notrun"]
    assert doc["label"] == "NOT_RUN" and doc["reason"] == f"VALID score file missing: {w.audit_dir / 'books_valid2k' / 'scores.csv.gz'}"
    assert doc["status"] == "EXPLORATORY_DESCRIPTIVE" and doc["question"] == "next" and doc["panel_kind"] == "qwen_registered"
    for absent in ("signals", "readings", "direction", "counts", "temperature", "panel"):
        assert absent not in doc, absent                                   # nothing was analysed
    assert doc["meta"]["input_sha1"]["scores_valid"] is None and doc["meta"]["input_sha1"]["scores_test"]
    assert doc["meta"]["code_sha1"] and doc["meta"]["registered_settings"] is True
    assert (w.audit_dir / "books_valid2k").exists() is False and path.exists()


def test_not_run_when_no_valid_event_has_a_scored_candidate_and_a_missing_test_input_is_an_error(tmp_path):
    N = 15
    cens = {("next", e, c): 2 for e in range(12) for c in range(N)}
    w = build_world(tmp_path, "tools", E=30, N=N, n_valid=12, seed=8, censored_valid=cens)
    doc = run(w)
    assert doc["label"] == "NOT_RUN" and "no event with a scored candidate" in doc["reason"] and "signals" not in doc
    assert doc["meta"]["input_sha1"]["scores_valid"]                       # the file exists: its sha1 is recorded
    with pytest.raises(FileNotFoundError):                                  # the TEST scores are not optional
        sc.run_control("tools", w.audit_dir, w.panel_test, w.panel_valid, w.tmp / "x.json", panel_kind="k", test_role="nope")
    with pytest.raises(FileNotFoundError):
        sc.run_control("tools", w.audit_dir, w.tmp / "missing.jsonl", w.panel_valid, w.tmp / "x.json", panel_kind="k")
    assert not (w.tmp / "x.json").exists()


def test_cli_exit_codes_and_the_arguments_of_the_audits_run(world, tmp_path, capsys):
    base = ["run", "--domain", "toys", "--audit_dir", str(world.audit_dir), "--panel_test", str(world.panel_test),
            "--panel_valid", str(world.panel_valid), "--panel_kind", "qwen_registered", "--n_boot", "20"]
    out = tmp_path / "cli.json"
    assert sc.main(base + ["--out", str(out)]) == 0
    assert "wrote" in capsys.readouterr().out
    # the audit's flags are accepted (the reference files are never read: see the path-guard test), --questions is `next` only
    assert sc.main(base + ["--out", str(tmp_path / "c2.json"), "--ref_ranks", str(tmp_path / "nowhere"), "--ref_exposure",
                           str(tmp_path / "nowhere"), "--quarantine_n", "0", "--segments", "auto", "--test_role", "test",
                           "--valid_role", "valid2k"]) == 0
    assert jload(tmp_path / "c2.json")["signals"] == jload(out)["signals"]
    assert sc.main(base + ["--out", str(tmp_path / "c3.json"), "--questions", "next,like"]) == 2
    assert sc.main(base + ["--out", str(tmp_path / "c3.json"), "--questions", "like"]) == 2
    assert "registers the question" in capsys.readouterr().err and not (tmp_path / "c3.json").exists()
    assert sc.main([a if a != "qwen_registered" else "Qwen Registered" for a in base] + ["--out", str(tmp_path / "c4.json")]) == 2
    assert sc.main(base + ["--out", str(tmp_path / "c5.json"), "--first_event", "5"]) == 2              # needs --segments single
    assert sc.main(["run", "--domain", "toys", "--audit_dir", str(world.audit_dir), "--panel_test", str(tmp_path / "no.jsonl"),
                    "--panel_valid", str(world.panel_valid), "--panel_kind", "k", "--out", str(tmp_path / "c6.json")]) == 2
    assert sc.main(base + ["--out", str(tmp_path / "c7.json"), "--n_boot", "0"]) == 2
    with pytest.raises(SystemExit):                                          # argparse: a required argument is missing
        sc.main(["run", "--domain", "toys"])
    ns = sc.parse_args(base)
    assert (ns.questions, ns.segments, ns.test_role, ns.valid_role, ns.first_event, ns.quarantine_n) == ("next", "auto", "test", "valid2k", None, None)
    # a failed self-check or a failed audit comparison is exit status 1 and writes nothing
    bad = tmp_path / "bad_audit.json"
    d = json.loads(json.dumps(stats.strict_json(world.audit)))
    d["segments"]["all"]["questions"]["next"]["D_selective_serving"]["signals"]["random"]["gain_at_50_vs_full"]["hr1"]["hi"] += 1e-6
    d["n_boot"] = 20
    bad.write_text(json.dumps(d), encoding="utf-8")
    assert sc.main(base + ["--out", str(tmp_path / "c8.json"), "--audit_json", str(bad)]) == 1
    assert not (tmp_path / "c8.json").exists() and not list(tmp_path.glob("*.tmp"))


# ====================================== 7. outputs: refusals, determinism, vocabulary, the files the module opens
def test_output_refusals_below_the_audit_directory_and_over_other_files_and_the_default_directory(world, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = (world.domain, world.audit_dir, world.panel_test, world.panel_valid)
    kw = dict(panel_kind="qwen_registered", n_boot=B)
    refused = ["outputs/confrec/nextitem_audit/toys.json", "outputs/confrec/nextitem_audit/sub/deeper/x.json",
               "./outputs/confrec/../confrec/nextitem_audit/x.json", str(tmp_path / "outputs" / "confrec" / "nextitem_audit" / "y.json"),
               "OUTPUTS/Confrec/NextItem_Audit/z.json", "outputs/confrec/nextitem_audit_ctrl/../nextitem_audit/w.json",
               "outputs/confrec/nextitem_audit/", "outputs/confrec/nextitem_audit", "elsewhere/outputs/confrec/nextitem_audit/v.json"]
    for out in refused:
        with pytest.raises(sc.ControlError, match="registered audit directory") as e:
            sc.run_control(*args, out, **kw)
        assert e.value.code == 2, out
    # a refusal comes before any work: the inputs may be wrong, the refusal is still what is reported
    with pytest.raises(sc.ControlError, match="registered audit directory"):
        sc.run_control(world.domain, world.audit_dir, tmp_path / "no_panel.jsonl", world.panel_valid,
                       "outputs/confrec/nextitem_audit/toys.json", **kw)
    assert not Path("outputs").exists() and not Path("elsewhere").exists()                # nothing was created
    # `summarize` writes under the same rule
    d1 = tmp_path / "a.json"
    sc.run_control(*args, d1, **kw)
    assert sc.main(["summarize", "--inputs", str(d1), "--out", "outputs/confrec/nextitem_audit/summary.json"]) == 2
    assert not Path("outputs").exists()
    # the controls' own directory is a different directory, although its name starts like the audit's
    assert not sc.under_audit_dir("outputs/confrec/nextitem_audit_ctrl/toys.json") and not sc.under_audit_dir("outputs/confrec/other.json")
    assert sc.under_audit_dir("a/b/outputs/confrec/nextitem_audit/x.json") and not sc.under_audit_dir("outputs/nextitem_audit/x.json")
    ok = sc.run_control(*args, "outputs/confrec/nextitem_audit_ctrl/toys.json", **kw)
    assert ok["label"] in sc.PANEL_LABELS and Path("outputs/confrec/nextitem_audit_ctrl/toys.json").exists()
    # no --out: the default directory outputs/confrec/nextitem_audit_ctrl and the file <domain>__<panel_kind>.json
    cli = ["run", "--domain", "toys", "--audit_dir", str(world.audit_dir), "--panel_test", str(world.panel_test), "--panel_valid",
           str(world.panel_valid), "--panel_kind", "qwen_registered", "--n_boot", str(B)]
    assert sc.main(cli) == 0
    default = Path("outputs/confrec/nextitem_audit_ctrl/toys__qwen_registered.json")
    assert default.exists() and jload(default)["domain"] == "toys"
    # a directory as --out
    (tmp_path / "dir_out").mkdir()
    assert sc.main(cli + ["--out", "dir_out"]) == 0 and (tmp_path / "dir_out" / "toys__qwen_registered.json").exists()
    # an existing audit result (outside the registered directory) and any other existing file are never overwritten
    audit_copy = tmp_path / "elsewhere_audit" / "toys.json"
    audit_copy.parent.mkdir()
    audit_copy.write_text(json.dumps(stats.strict_json(world.audit)), encoding="utf-8")
    other = tmp_path / "notes.json"
    other.write_text('{"schema": "something_else"}', encoding="utf-8")
    plain = tmp_path / "plain.txt"
    plain.write_text("keep me", encoding="utf-8")
    for p in (audit_copy, other, plain):
        before = p.read_bytes()
        with pytest.raises(sc.ControlError, match="never overwritten") as e:
            sc.run_control(*args, p, **kw)
        assert e.value.code == 2 and p.read_bytes() == before and not list(p.parent.glob("*.tmp")), p
        assert sc.main(cli + ["--out", str(p)]) == 2 and p.read_bytes() == before
    # an output of the module itself may be written again (same bytes), and a directory is not a file
    before = d1.read_bytes()
    sc.run_control(*args, d1, **kw)
    assert d1.read_bytes() == before
    # a link into the audit directory is refused as what it points to (skipped where links cannot be made)
    (tmp_path / "outputs" / "confrec" / "nextitem_audit").mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(tmp_path / "outputs" / "confrec" / "nextitem_audit", tmp_path / "link_to_audit", target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        pass
    else:
        with pytest.raises(sc.ControlError, match="registered audit directory"):
            sc.run_control(*args, "link_to_audit/x.json", **kw)


def test_same_input_same_bytes_registered_defaults_and_no_timestamps(world, tmp_path):
    base = ["run", "--domain", "toys", "--audit_dir", str(world.audit_dir), "--panel_test", str(world.panel_test),
            "--panel_valid", str(world.panel_valid), "--panel_kind", "qwen_registered"]
    a, b, c = tmp_path / "a" / "x.json", tmp_path / "b" / "x.json", tmp_path / "c.json"
    assert sc.main(base + ["--out", str(a)]) == 0                                   # no --n_boot, no --seed: the registered ones
    assert sc.main(base + ["--out", str(b), "--n_boot", "2000", "--seed", "0"]) == 0
    assert a.read_bytes() == b.read_bytes()
    doc = jload(a)
    assert (doc["n_boot"], doc["seed"], doc["meta"]["registered_settings"]) == (2000, 0, True)
    assert doc["signals"]["p_max"]["gain50"]["ndcg10"]["n_boot"] == 2000
    assert sc.main(base + ["--out", str(c), "--n_boot", "20", "--seed", "0"]) == 0 and jload(c)["meta"]["registered_settings"] is False
    ns = sc.parse_args(base)
    assert (ns.n_boot, ns.seed) == (2000, 0) and (sc.REG_N_BOOT, sc.REG_SEED) == (2000, 0)
    import inspect
    sig = inspect.signature(sc.run_control).parameters
    assert (sig["n_boot"].default, sig["seed"].default) == (2000, 0)
    raw = a.read_bytes()
    text = raw.decode("utf-8")
    assert raw.endswith(b"\n") and b"\r" not in raw and "NaN" not in text and "Infinity" not in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}", text)
    keys = set()

    def walk(x):
        if isinstance(x, dict):
            keys.update(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(doc)
    assert not [k for k in keys if re.search(r"time|date|clock|elapsed|timing|host|stamp", k, re.I)]
    # another seed changes the random block only (p_max, the controls' directions and the counts do not depend on it)
    d3 = run(world, "s3.json", seed=3, n_boot=2000)
    assert d3["signals"]["random"]["gain50"] != doc["signals"]["random"]["gain50"]
    assert d3["signals"]["p_max"] == doc["signals"]["p_max"] and d3["counts"] == doc["counts"] and d3["direction"] == doc["direction"]


FORBIDDEN = ("significan", "better", "worse", "outperform", "superior", "inferior", "beats", "wins", "loses", "improve", "stronger",
             "weaker", "dominat")


def vocabulary_problems(text: str) -> list:
    """Words that would state a direction (or call something significant) once the registered labels are taken out."""
    low = text.replace("LLM_BETTER", "").replace("CHEAP_BETTER", "").lower()
    return [w for w in FORBIDDEN if w in low]


def test_vocabulary_status_and_no_statement_beyond_the_registered_labels(labelled, world, tmp_path, capsys):
    texts = {"module": MODULE.read_text(encoding="utf-8")}
    for name, (_, doc, path) in labelled.items():
        texts[name] = path.read_text(encoding="utf-8")
        assert doc["status"] == "EXPLORATORY_DESCRIPTIVE" and "exploratory" in doc["status_note"] and doc["spec"].endswith("ADDENDUM_11.md")
    texts["world"] = (world.tmp / "out/run.json").read_text(encoding="utf-8")
    for name, t in texts.items():
        assert vocabulary_problems(t) == [], name
    # every string of a result is a label, a name, a path or a definition: the labels come from the registered sets
    for name in ("adds", "cheap", "matched", "mixed"):
        doc = labelled[name][1]
        assert doc["label"] in sc.PANEL_LABELS and set(doc["readings"].values()) <= set(sc.SIGNAL_LABELS)
        assert {k for k in doc["definition"]} >= {"status", "serving_set", "utility", "direction", "gain50", "delta", "reading"}
    # what the command line prints: the written path and the registered label
    paths = [str(labelled[n][2]) for n in labelled]
    out = tmp_path / "sum.json"
    capsys.readouterr()
    assert sc.main(["summarize", "--inputs", ",".join(paths), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "wrote" in printed and vocabulary_problems(printed) == []
    summary = out.read_text(encoding="utf-8")
    assert vocabulary_problems(summary) == [] and jload(out)["status"] == "EXPLORATORY_DESCRIPTIVE"
    assert sc.main(["run", "--domain", "toys", "--audit_dir", str(world.audit_dir), "--panel_test", str(world.panel_test),
                    "--panel_valid", str(world.panel_valid), "--panel_kind", "qwen_registered", "--n_boot", "20", "--out",
                    str(tmp_path / "r.json")]) == 0
    assert vocabulary_problems(capsys.readouterr().out) == []


# ----- the files the module opens (an audit hook: every `open` and every directory listing while it runs)
_WATCH = {"on": False, "events": []}


def _watch_hook(event, args):
    if _WATCH["on"] and event in ("open", "os.listdir", "os.scandir", "glob.glob") and args:
        _WATCH["events"].append((event, args[0]))


if not getattr(sys, "_servingctrl_watch_installed", False):
    sys.addaudithook(_watch_hook)
    sys._servingctrl_watch_installed = True


class watching:
    def __enter__(self):
        _WATCH["events"].clear()
        _WATCH["on"] = True
        return _WATCH["events"]

    def __exit__(self, *exc):
        _WATCH["on"] = False


def foreign_paths(events, allowed) -> list:
    """The opened / listed paths that are neither allowed nor part of the interpreter (modules, byte code, libraries)."""
    allowed = {os.path.normcase(os.path.realpath(p)) for p in allowed}
    prefixes = {os.path.normcase(os.path.realpath(p)) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
    bad = []
    for ev, p in events:
        if p is None or isinstance(p, int):
            continue
        q = os.path.normcase(os.path.realpath(os.fsdecode(p)))
        if q in allowed or q.endswith((".pyc", ".pyd", ".so", ".dll")) or any(q.startswith(pre + os.sep) for pre in prefixes):
            continue
        bad.append((ev, str(p)))
    return bad


def test_the_module_opens_only_the_paths_it_is_given(world, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    decoys = {}
    for rel in ("docs/sigir/PILOT_LOG.md", "docs/sigir/ref_ranks/toys/ccrp_v3.csv.gz", "docs/sigir/ref_exposure/toys/_pool.json",
                "outputs/confrec/nextitem_audit/toys.json", "idea-stage/PREREG_AMENDMENT_3_ADDENDUM_11.md",
                "outputs/confrec/nextitem_audit_ctrl/old.json"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("decoy", encoding="utf-8")
        decoys[rel] = p
    out = tmp_path / "outputs" / "confrec" / "nextitem_audit_ctrl" / "guard.json"
    audit_json = tmp_path / "audit_for_guard.json"
    audit_json.write_text(json.dumps(stats.strict_json(audit_run(world, n_boot=20))), encoding="utf-8")
    code = [Path(sc.__file__).resolve().parent / n for n in sc.CODE_FILES]
    given = [world.panel_test, world.panel_valid, world.audit_dir / "toys_test" / "scores.csv.gz",
             world.audit_dir / "toys_valid2k" / "scores.csv.gz", out, out.with_name(out.name + ".tmp"), audit_json] + code
    argv = ["run", "--domain", "toys", "--audit_dir", str(world.audit_dir), "--panel_test", str(world.panel_test), "--panel_valid",
            str(world.panel_valid), "--panel_kind", "qwen_registered", "--n_boot", "20", "--out", str(out), "--audit_json", str(audit_json),
            "--ref_ranks", str(decoys["docs/sigir/ref_ranks/toys/ccrp_v3.csv.gz"].parent),
            "--ref_exposure", str(decoys["docs/sigir/ref_exposure/toys/_pool.json"].parent)]
    with watching() as events:
        assert sc.main(argv) == 0
        assert sc.main(["summarize", "--inputs", str(out), "--out", str(tmp_path / "outputs" / "confrec" / "nextitem_audit_ctrl" / "s.json")]) == 0
    opened = {os.path.normcase(os.path.realpath(os.fsdecode(p))) for _, p in events if p is not None and not isinstance(p, int)}
    assert os.path.normcase(os.path.realpath(world.panel_test)) in opened and os.path.normcase(os.path.realpath(out.with_name(out.name + ".tmp"))) in opened
    assert foreign_paths(events, given + [tmp_path / "outputs" / "confrec" / "nextitem_audit_ctrl" / "s.json",
                                          tmp_path / "outputs" / "confrec" / "nextitem_audit_ctrl" / "s.json.tmp"]) == []
    for rel, p in decoys.items():
        assert os.path.normcase(os.path.realpath(p)) not in opened, rel
    # the guard itself works: a planted read of a decoy is found
    with watching() as events:
        decoys["docs/sigir/PILOT_LOG.md"].read_text(encoding="utf-8")
        list(os.scandir(tmp_path / "docs"))
    bad = foreign_paths(events, given)
    assert [os.path.basename(p) for _, p in bad] == ["PILOT_LOG.md", "docs"] or len(bad) == 2
    # `record` opens only its own files and the pilot log it is given
    log = tmp_path / "log.md"
    log.write_text("x", encoding="utf-8")
    with watching() as events:
        sc.main(["record", "--pilot_log", str(log)])
    assert foreign_paths(events, [log] + [ROOT / rel for rel in sc.RECORD_FILES]) == []


# ============================================================================================ 8. summarize and record
def test_summarize_counts_the_labels_and_the_readings_and_nothing_else(labelled, tmp_path):
    docs = [labelled[n][1] for n in ("adds", "cheap", "matched", "mixed", "notrun")]
    s = sc.summarize(docs)
    assert s["schema"] == sc.SUMMARY_SCHEMA and s["status"] == "EXPLORATORY_DESCRIPTIVE" and s["panel_kind"] == "qwen_registered"
    assert s["n_panels"] == 5 and s["panel_labels"] == {"ADDS": 1, "CHEAP_SUFFICES": 2, "MIXED": 1, "NOT_RUN": 1}
    assert s["domains_by_label"] == {"ADDS": ["toys"], "CHEAP_SUFFICES": ["home", "tools"], "MIXED": ["games"], "NOT_RUN": ["books"]}
    tally = {c: {lab: 0 for lab in sc.SIGNAL_LABELS} for c in CONTROLS}
    for d in docs:
        for c, lab in d.get("readings", {}).items():
            tally[c][lab] += 1
    assert s["signal_readings"] == tally and sum(tally["hist_len"].values()) == 4              # NOT_RUN has no readings
    assert s["panels"] == [{"domain": "books", "segment": None, "label": "NOT_RUN"}, {"domain": "games", "segment": "all", "label": "MIXED"},
                           {"domain": "home", "segment": "all", "label": "CHEAP_SUFFICES"},
                           {"domain": "tools", "segment": "all", "label": "CHEAP_SUFFICES"}, {"domain": "toys", "segment": "all", "label": "ADDS"}]

    def leaves(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from leaves(v)
        elif isinstance(x, list):
            for v in x:
                yield from leaves(v)
        else:
            yield x
    assert not [v for v in leaves(s) if isinstance(v, float)]                                    # counts, never a statistic
    # the command line: the same document, the same bytes, strict
    out = tmp_path / "sum" / "summary.json"
    paths = ",".join(str(labelled[n][2]) for n in ("adds", "cheap", "matched", "mixed", "notrun"))
    assert sc.main(["summarize", "--inputs", paths, "--out", str(out)]) == 0
    assert jload(out) == json.loads(json.dumps(s)) and out.read_bytes().endswith(b"\n")
    out2 = tmp_path / "sum2.json"
    assert sc.main(["summarize", "--inputs", paths, "--out", str(out2)]) == 0 and out2.read_bytes() == out.read_bytes()
    # a single NOT_RUN panel is a summary too; an output of `summarize` is its own file and may be written again
    assert sc.summarize([labelled["notrun"][1]])["panel_labels"] == {"ADDS": 0, "CHEAP_SUFFICES": 0, "MIXED": 0, "NOT_RUN": 1}
    assert sc.main(["summarize", "--inputs", paths, "--out", str(out)]) == 0


def test_summarize_refuses_mixed_panel_kinds_layouts_settings_code_and_duplicates(labelled, tmp_path):
    docs = [json.loads(json.dumps(labelled[n][1])) for n in ("adds", "cheap", "mixed")]

    def refused(mutate, match):
        ds = json.loads(json.dumps(docs))
        mutate(ds)
        with pytest.raises(sc.ControlError, match=match) as e:
            sc.summarize(ds)
        assert e.value.code == 2
        files = []
        for i, d in enumerate(ds):
            p = tmp_path / f"in{i}_{abs(hash(match)) % 10 ** 6}.json"
            p.write_text(json.dumps(d), encoding="utf-8")
            files.append(str(p))
        out = tmp_path / f"out_{abs(hash(match)) % 10 ** 6}.json"
        assert sc.main(["summarize", "--inputs", ",".join(files), "--out", str(out)]) == 2 and not out.exists()
    refused(lambda ds: ds[1].update(panel_kind="llama_z2"), "mixed panel kinds")
    refused(lambda ds: ds[2].update(n_boot=2000), "mixed n_boot")
    refused(lambda ds: ds[2].update(seed=1), "mixed seeds")
    refused(lambda ds: ds[0]["layout"].update(segments="single"), "mixed layouts")
    refused(lambda ds: ds[1]["meta"]["code_sha1"].update({"src/confrec/nextitem_serving_control.py": "0" * 40}), "mixed code versions")
    refused(lambda ds: ds.append(json.loads(json.dumps(ds[0]))), "listed twice")
    refused(lambda ds: ds[0].update(schema="nextitem_audit_v1"), "not a nextitem_serving_control_v1")
    refused(lambda ds: ds[0].update(label="SIGNIFICANT"), "unknown label")
    with pytest.raises(sc.ControlError, match="no input"):
        sc.summarize([])
    assert sc.main(["summarize", "--inputs", "", "--out", str(tmp_path / "e.json")]) == 2
    # the same panel kind in several domains, and the backbones one by one: each kind has its own summary
    only_llama = [dict(json.loads(json.dumps(d)), panel_kind="llama_z2") for d in docs]
    assert sc.summarize(only_llama)["panel_kind"] == "llama_z2"


def test_record_prints_path_equals_sha1_and_checks_the_pilot_log(tmp_path, capsys):
    root = tmp_path / "repo"
    for i, rel in enumerate(sc.RECORD_FILES):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(f"file {i}\n".encode())
    lines = [f"{rel} = {hashlib.sha1(f'file {i}{chr(10)}'.encode()).hexdigest()}" for i, rel in enumerate(sc.RECORD_FILES)]
    assert sc.record_lines(root=root) == lines and len(lines) == 3
    assert [r.split(" = ")[0] for r in lines] == ["src/confrec/nextitem_serving_control.py", "tests/test_confrec_serving_control.py",
                                                  "scripts/sigir/run_servingctrl.sh"]
    assert sc.main(["record", "--root", str(root)]) == 0
    assert capsys.readouterr().out.splitlines() == lines
    log = tmp_path / "PILOT_LOG.md"
    log.write_text("# pilot log\n", encoding="utf-8")
    assert sc.main(["record", "--root", str(root), "--pilot_log", str(log)]) == 4
    err = capsys.readouterr().err
    assert all(rel in err for rel in sc.RECORD_FILES)
    log.write_text("\n".join(lines[:2]).upper() + "\n", encoding="utf-8")                        # case-insensitive, two of three
    assert sc.record_missing(log, root=root) == [sc.RECORD_FILES[2]]
    assert sc.main(["record", "--root", str(root), "--pilot_log", str(log)]) == 4
    assert sc.RECORD_FILES[2] in capsys.readouterr().err and sc.RECORD_FILES[0] not in capsys.readouterr().err
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert sc.main(["record", "--root", str(root), "--pilot_log", str(log)]) == 0
    (root / sc.RECORD_FILES[1]).write_bytes(b"edited\n")                                          # a file changed after the record
    assert sc.main(["record", "--root", str(root), "--pilot_log", str(log)]) == 4
    assert sc.main(["record", "--root", str(root), "--pilot_log", str(tmp_path / "missing.md")]) == 4
    assert sc.main(["record", "--root", str(root), "--files", "src/confrec/nextitem_serving_control.py"]) == 0
    assert sc.main(["record", "--root", str(tmp_path / "empty"), "--files", "nope.py"]) == 1
    # the three real files exist, are LF only (their sha1 does not depend on a checkout's line endings) and the real lines are well formed
    real = sc.record_lines()
    assert all(re.fullmatch(r"[a-z_/\.]+ = [0-9a-f]{40}", ln) for ln in real)
    for rel in sc.RECORD_FILES:
        assert b"\r" not in (ROOT / rel).read_bytes(), rel
    assert hashlib.sha1(MODULE.read_bytes()).hexdigest() in real[0] and hashlib.sha1(SCRIPT.read_bytes()).hexdigest() in real[2]
