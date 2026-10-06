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
    return SimpleNamespace(tmp=tmp, domain=domain, rows=rows, vrows=vrows, L=L, grp=grp, pos=pos, Lv=Lv, gv=gv, pv=pv,
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
# VALID: ten events; positives in all three groups so that the popularity temperature is interior
TINY_VALID = [
    (3, 4, 5, ["h0", "h1", "h2", "h3", "h4"], "head", 0),           # v0
    (3, 4, 5, ["m0", "m1"], "mid", 1),                              # v1
    (2, 2, 8, ["t0", "t1", "t2"], "tail", 0),                       # v2  (no LLM scores)
    (6, 3, 3, ["h0", "h1", "h2", "h3"], "head", 2),                 # v3
    (4, 4, 4, ["m0", "t0"], "mid", 0),                              # v4
    (1, 5, 6, [], "tail", 1),                                       # v5
    (2, 0, 10, ["h0", "h1", "h2", "h3", "h4", "h5"], "head", 1),    # v6
    (5, 5, 2, ["t0", "t1", "t2", "t3"], "tail", 0),                 # v7  (no LLM scores)
    (3, 3, 6, ["h0", "m0", "t0"], "mid", 2),                        # v8
    (0, 6, 6, ["m0", "m1", "m2"], "mid", 4),                        # v9
]
TINY_VALID_RANKS = [1, 3, 9, 2, 5, 12, 1, 8, 4, 6]
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
