"""Tests of the full-scale next-item audit (src/confrec/nextitem_audit.py, scripts/sigir/export_ref_exposure.py).

Synthetic fixtures with known answers for every section of docs/sigir/NEXTITEM_AUDIT_SPEC.md; every bootstrap engine is
checked against stats / metrics / pilot_mirror on explicit resamples. The Z2 tests (second backbone on a subsample,
Amendment 3 section 4: `run --segments single`, `--test_role` / `--valid_role`, `restrict`) and the pin of the default
output (a digest of the pre-Z2 module's output on the existing fixtures) are at the end of the end-to-end section.
"""
from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.confrec import metrics, nextitem_audit as na, pilot_mirror, stats

ROOT = Path(__file__).resolve().parents[1]


def _load_export():
    spec = importlib.util.spec_from_file_location("export_ref_exposure_under_test",
                                                  ROOT / "scripts" / "sigir" / "export_ref_exposure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================================ synthetic fixtures
def synth_arrays(E=400, N=21, seed=0, signal=2.0, head_bias=0.0, sure_spread=0.6, grid=1 / 16, p_head=0.6, p_mid=0.25):
    """(L, groups, pos, sure): logit = sure_e * (z + signal * [c == pos]) + head_bias * [head], on a coarse grid (ties)."""
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, N, E)
    grp = rng.choice(3, size=(E, N), p=[p_head, p_mid, 1 - p_head - p_mid]).astype(np.int8)
    sure = np.exp(rng.normal(0, sure_spread, E))
    z = rng.normal(0, 1.0, (E, N))
    z[np.arange(E), pos] += signal
    L = sure[:, None] * z + head_bias * (grp == 0)
    if grid:
        L = np.round(L / grid) * grid
    return L, grp, pos, sure


def synth_confident(E=600, N=21, seed=0, frac_easy=0.4, grid=1 / 16):
    """Events are either easy (the positive stands out: confident and correct) or hard (flat scores: unsure and mostly
    wrong), so p_max is strongly informative about correctness. Returns (L, groups, pos, easy)."""
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, N, E)
    grp = rng.choice(3, size=(E, N), p=[0.6, 0.25, 0.15]).astype(np.int8)
    easy = rng.random(E) < frac_easy
    L = rng.normal(0, 0.4, (E, N))
    L[np.flatnonzero(easy), pos[easy]] += 5.0
    return np.round(L / grid) * grid, grp, pos, easy


def make_panel_rows(E, N, seed, pos, grp, n_items=None, hist_max=6, domain="d", text_len=0):
    """Panel rows (dicts as in ranking_test.jsonl) whose item -> group map is consistent with `grp`."""
    rng = np.random.default_rng(seed + 11)
    n_items = n_items or max(60, E * N // 3)
    item_group = rng.choice(3, n_items, p=[0.6, 0.25, 0.15])
    by_group = [np.flatnonzero(item_group == g) for g in range(3)]      # per call: fixtures never depend on test order
    rows = []
    for e in range(E):
        cand, used = [], set()
        for c in range(N):
            pool = by_group[int(grp[e, c])]
            j = int(pool[rng.integers(0, len(pool))])
            for _ in range(60):
                if j not in used:
                    break
                j = int(pool[rng.integers(0, len(pool))])
            used.add(j)
            cand.append(j)
        assert len(set(cand)) == N, "fixture needs distinct candidates"
        hist = [f"{domain}I{int(j):06d}" for j in rng.integers(0, n_items, int(rng.integers(0, hist_max + 1)))]
        uid = f"{domain}U{e:06d}"
        rows.append({"source_event_id": f"{uid}::{1000 + e}", "user_id": uid, "history_item_ids": hist,
                     "candidate_item_ids": [f"{domain}I{j:06d}" for j in cand],
                     "candidate_popularity_groups": [na.GROUPS[int(g)] for g in grp[e]],
                     "positive_item_index": int(pos[e]), "num_candidates": N, "split_name": "test",
                     "candidate_texts": ["x" * text_len] * N if text_len else []})
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def panel_from(tmp_path, rows, name="panel.jsonl", **kw):
    write_jsonl(tmp_path / name, rows)
    return na.load_panel(tmp_path / name, **kw)


def write_scores(path, rows, Ls: dict, censored=None):
    """scores.csv.gz in the pyes_scorer contract. Ls = {question: (E, N) logits}; censored = {(q, e, c): code}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    censored = censored or {}
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_event_id", "user_id", "item_id", "cand_idx", "label", "question", "lp_yes", "lp_no", "logit",
                    "yes_no_mass", "censored"])
        for e, r in enumerate(rows):
            for c, it in enumerate(r["candidate_item_ids"]):
                for q, L in Ls.items():
                    code = censored.get((q, e, c), 0)
                    x = L[e, c]
                    lg = "nan" if code in (2, 3) or not np.isfinite(x) else f"{x:.6f}"
                    w.writerow([r["source_event_id"], r["user_id"], it, c, int(c == r["positive_item_index"]), q,
                                "-1.0", "-1.0", lg, "1.0", code])


def ref_records(rows, score, path):
    """ranking_eval_records.csv of a synthetic reference method with (E, N) scores (higher = better)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "source_event_id", "split_name", "timestamp", "positive_item_id", "positive_rank",
                    "num_candidates", "candidate_item_ids", "candidate_popularity_groups", "pred_ranked_item_ids",
                    "positive_popularity_group"])
        for e, r in enumerate(rows):
            order = np.argsort(-score[e], kind="stable")
            cands = r["candidate_item_ids"]
            p = r["positive_item_index"]
            rank = int(np.flatnonzero(order == p)[0]) + 1
            w.writerow([r["user_id"], r["source_event_id"], "test", 1000 + e, cands[p], rank, len(cands), repr(cands),
                        repr(r["candidate_popularity_groups"]), repr([cands[i] for i in order]),
                        r["candidate_popularity_groups"][p]])


def make_domain(tmp, domain, E=36, N=15, seed=0, n_valid=30, signal=2.0, head_bias=2.5, methods=("ccrp_v3", "llmesr_sasrec"),
                censored_test=None):
    """A complete synthetic domain on disk: panels, audit scores (test + valid2k), ref_ranks, ref_exposure (written by the
    export script from synthetic ranking_eval_records). Returns the paths and the underlying arrays."""
    tmp = Path(tmp)
    exp = _load_export()
    L1, grp, pos, _ = synth_arrays(E, N, seed, signal=signal, head_bias=head_bias)
    rng = np.random.default_rng(seed + 5)
    L2 = np.round((L1 * 0.5 + rng.normal(0, 1.0, (E, N)) + 0.3 * (grp == 0)) * 16) / 16
    rows = make_panel_rows(E, N, seed, pos, grp, domain=domain)
    write_jsonl(tmp / "panels" / f"{domain}_test.jsonl", rows)
    Lv1, gv, pv, _ = synth_arrays(n_valid, N, seed + 100, signal=signal, head_bias=head_bias)
    Lv2 = np.round((Lv1 * 0.5 + np.random.default_rng(seed + 6).normal(0, 1.0, (n_valid, N))) * 16) / 16
    vrows = make_panel_rows(n_valid, N, seed + 100, pv, gv, domain=domain + "v")
    write_jsonl(tmp / "panels" / f"{domain}_valid.jsonl", vrows)
    audit = tmp / "audit"
    write_scores(audit / f"{domain}_test" / "scores.csv.gz", rows, {"next": L1, "like": L2}, censored_test)
    write_scores(audit / f"{domain}_valid2k" / "scores.csv.gz", vrows, {"next": Lv1, "like": Lv2})
    ref = {r["source_event_id"]: [r["candidate_item_ids"][r["positive_item_index"]],
                                  exp.candidate_fp(r["candidate_item_ids"])] for r in rows}
    (tmp / "docs/sigir/panel_refs").mkdir(parents=True, exist_ok=True)
    (tmp / f"docs/sigir/panel_refs/{domain}_test.json").write_text(json.dumps(ref), encoding="utf-8")
    ref_dir = tmp / "docs/sigir/ref_ranks" / domain
    ref_dir.mkdir(parents=True, exist_ok=True)
    mrng = np.random.default_rng(seed + 9)
    for i, meth in enumerate(methods):
        sc = mrng.normal(0, 1.0, (E, N)) + (1.0 + i) * (np.arange(N)[None, :] == pos[:, None]) + (0.8 - 0.8 * i) * (grp == 0)
        base = (f"outputs/{domain}_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate" if meth == "ccrp_v3" else
                f"outputs/baselines/official_adapters/{domain}_large10000_100neg_{meth}_official_qwen3base_same_candidate")
        ref_records(rows, sc, tmp / base / "tables" / "ranking_eval_records.csv")
        with gzip.open(ref_dir / f"{meth}.csv.gz", "wt", newline="", encoding="utf-8") as fz:
            w = csv.writer(fz)
            w.writerow(["source_event_id", "user_id", "positive_rank", "num_candidates"])
            for e, r in enumerate(rows):
                w.writerow([r["source_event_id"], r["user_id"], int(1 + (sc[e] > sc[e, r["positive_item_index"]]).sum()), N])
    exp.export_domain(domain, tmp, tmp / "docs/sigir/ref_exposure", verbose=False)
    return {"rows": rows, "L": {"next": L1, "like": L2}, "grp": grp, "pos": pos, "audit": audit,
            "panel_test": tmp / "panels" / f"{domain}_test.jsonl", "panel_valid": tmp / "panels" / f"{domain}_valid.jsonl",
            "ref_ranks": ref_dir, "ref_exposure": tmp / "docs/sigir/ref_exposure" / domain}


def qctx(tmp_path, L, grp, pos, beta=1.0, seed=0):
    """(panel, qv) for an (E, N) logit array: a panel on disk whose groups / positives match, and question_arrays."""
    E, N = L.shape
    panel = panel_from(tmp_path, make_panel_rows(E, N, seed, pos, grp))
    rand = np.random.default_rng([0, 1]).random(E)
    return panel, na.question_arrays(L, panel, beta, rand)


def run_sections(qv, profile=None, n_boot=60, n_bias=40, m=None):
    """sec_C / sec_D / sec_E of one question over a single segment, resolved."""
    E = len(qv["valid"])
    m = qv["valid"] if m is None else m
    boot = na.EventBoot(E, n_boot, 0)
    batch = na.LinearBatch(boot)
    prof = np.random.default_rng(1).integers(0, 5, E) / 2.0 if profile is None else profile
    out = {"C": na.sec_C(batch, boot, qv, m, {}), "D": na.sec_D(batch, boot, qv, m, prof),
           "E": na.sec_E(batch, boot, qv, m, n_bias)}
    batch.run()
    return na.resolve(out)


def canonical_digest(doc) -> str:
    """sha256 of a run / summary document made comparable across runs and machines: `timing_s` dropped, the values of
    `inputs` (temporary paths) nulled with their keys kept, every float rounded to 6 decimals (0.0 for -0.0), key ORDER and
    int / float TYPES kept (so a changed layout, a new key or a changed type changes the digest)."""
    def canon(x):
        if isinstance(x, dict):
            return {k: canon(v) for k, v in x.items() if k != "timing_s"}
        if isinstance(x, list):
            return [canon(v) for v in x]
        if isinstance(x, float):
            return None if math.isnan(x) else round(x, 6) + 0.0
        return x
    doc = dict(doc)
    if isinstance(doc.get("inputs"), dict):
        doc["inputs"] = {k: None for k in doc["inputs"]}
    return hashlib.sha256(json.dumps(canon(doc), allow_nan=False).encode("utf-8")).hexdigest()


# =============================================================================================== bootstrap engines
def test_event_bootstrap_reproduces_stats_helpers_and_pct_ci():
    rng = np.random.default_rng(0)
    n, B = 250, 40
    boot = na.EventBoot(n, B, seed=0)
    v, w = rng.normal(size=n), rng.normal(size=n)
    batch = na.LinearBatch(boot)
    s_mean = batch.add(v, np.ones(n))
    s_diff = batch.add(v - w, np.ones(n))
    batch.run()
    cb = stats.cluster_bootstrap(lambda idx: v[idx].mean(), np.arange(n), n_boot=B, seed=0)
    got = s_mean.value()
    assert (got["est"], got["lo"], got["hi"]) == pytest.approx((cb["est"], cb["lo"], cb["hi"]), abs=1e-12)
    pb = stats.paired_bootstrap({i: v[i] for i in range(n)}, {i: w[i] for i in range(n)}, n_boot=B, seed=0)
    got = s_diff.value()
    assert (got["est"], got["lo"], got["hi"]) == pytest.approx((pb["est"], pb["lo"], pb["hi"]), abs=1e-12)
    r = rng.normal(size=1000)
    r[::13] = np.nan
    assert na.pct_ci(r) == stats.percentile_ci(r)
    assert all(math.isnan(x) for x in na.pct_ci([np.nan]))                       # no finite replicate -> (nan, nan)


def test_pvalue_holm_and_shared_denominator_columns():
    assert na.boot_pvalue(np.arange(1, 101), 0.0) == pytest.approx(2 / 101)         # null below every replicate
    assert na.boot_pvalue(np.arange(-50, 50), 0.0) <= 1.0
    assert math.isnan(na.boot_pvalue([np.nan], 0.0))
    adj = na.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert adj["a"] == pytest.approx(0.04) and adj["c"] == pytest.approx(0.09)
    assert adj["b"] == pytest.approx(0.09) and adj["d"] == pytest.approx(0.5)       # monotone step-down
    adj = na.holm({"a": 0.001, "b": 0.01, "c": None, "d": 0.03})                    # a missing p stays in the family (m = 4)
    assert adj["a"] == pytest.approx(0.004) and adj["b"] == pytest.approx(0.03) and adj["d"] == pytest.approx(0.06)
    assert math.isnan(adj["c"])
    rng = np.random.default_rng(1)
    boot = na.EventBoot(80, 20, 0)
    d1 = (rng.random(80) < 0.6).astype(float)
    v = rng.normal(size=80)
    batch = na.LinearBatch(boot)
    s1, s2, s3 = batch.add(v * d1, d1), batch.add(v * d1 ** 2, d1), batch.add(v, np.ones(80))
    batch.run()
    assert len(batch.den_cols) == 2 and s2.est == pytest.approx(s1.est)                # d1 is 0/1: same ratio, shared column
    for b in range(20):
        idx = boot.resample_idx(b)
        assert s1.reps[b] == pytest.approx((v[idx] * d1[idx]).sum() / d1[idx].sum(), abs=1e-12)
    assert s3.value()["n"] == 80


def test_weighted_auroc_ece_and_pooled_auc_match_explicit_resamples():
    rng = np.random.default_rng(0)
    n, B = 200, 25
    boot = na.EventBoot(n, B, 0)
    score = np.round(rng.normal(size=n) * 4) / 4                                   # ties
    label = (rng.random(n) < 0.3 + 0.1 * score).astype(int)
    rep = na.weighted_auroc(boot.W, score, label)
    for b in range(B):
        idx = boot.resample_idx(b)
        assert rep[b] == pytest.approx(metrics.auroc(score[idx], label[idx]), abs=1e-12)
    conf = np.round(rng.random(n) ** 2 * 100) / 100
    corr = (rng.random(n) < conf).astype(float)
    r_e, r_a = na.ece_reps(boot.W, conf, corr), na.ece_reps(boot.W, conf, corr, adaptive=True)
    assert na.weighted_ece_est(conf, corr) == pytest.approx(metrics.ece(conf, corr, 15), abs=1e-12)
    for b in range(B):
        idx = boot.resample_idx(b)
        assert r_e[b] == pytest.approx(metrics.ece(conf[idx], corr[idx], 15), abs=1e-12)
        assert r_a[b] == pytest.approx(metrics.ece(conf[idx], corr[idx], 15, adaptive=True), abs=1e-12)
    # pooled AUROC with an event-cluster bootstrap (scored rows only, ties, unscored candidates)
    E, N = 120, 12
    S = np.round(rng.normal(size=(E, N)) * 3) / 3
    pos = rng.integers(0, N, E)
    S[np.arange(E), pos] += 0.7
    S[rng.random((E, N)) < 0.04] = -np.inf
    b2 = na.EventBoot(E, B, 0)
    sc = np.isfinite(S)
    y = np.zeros(S.shape, bool)
    y[np.arange(E), pos] = True
    ev = np.broadcast_to(np.arange(E)[:, None], S.shape)
    eng = na.PooledAUC(S[sc & y], ev[sc & y], S[sc & ~y], ev[sc & ~y])
    rep = eng.reps(b2.W)
    for b in range(B):
        idx = b2.resample_idx(b)
        assert rep[b] == pytest.approx(metrics.auroc(S[idx][sc[idx]], y[idx][sc[idx]].astype(int)), abs=1e-12)


def test_serve_engine_matches_risk_coverage_and_expected_served_share():
    rng = np.random.default_rng(0)
    n, B = 240, 25
    boot = na.EventBoot(n, B, 0)
    sig = np.round(rng.normal(size=n) * 3) / 3                                     # ties
    u1, u2 = rng.random(n), (rng.random(n) < 0.4).astype(float)
    quint = rng.integers(0, 5, n)
    quint[:7] = -1
    res = na.ServeEngine(sig, {"ndcg10": u1, "hr1": u2}, quint).reps(boot.W)
    for b in range(B):
        idx = boot.resample_idx(b)
        for u, uv in (("ndcg10", u1), ("hr1", u2)):
            cov, cum, aurc = metrics.risk_coverage(sig[idx], uv[idx])
            assert res["aurc"][u][b] == pytest.approx(aurc, abs=1e-10)
            ks = np.minimum(np.maximum(1, np.rint(np.asarray(na.COVERAGE) * len(idx)).astype(int)), len(idx))
            assert res["curve"][u][b] == pytest.approx(cum[ks - 1], abs=1e-10)
        # served fraction: the tie block straddling the 50% cut is served fractionally
        s_i, k50 = sig[idx], max(1, int(np.rint(0.5 * len(idx))))
        order = np.argsort(-s_i, kind="stable")
        served, i = np.zeros(len(idx)), 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and s_i[order[j + 1]] == s_i[order[i]]:
                j += 1
            served[order[i:j + 1]] = np.clip((k50 - i) / (j - i + 1), 0, 1)
            i = j + 1
        for q in range(5):
            m = quint[idx] == q
            assert res["served"][b, q] == pytest.approx(served[m].sum(), abs=1e-9)
            assert res["size"][b, q] == m.sum()


def test_exposure_engine_gini_and_coverage_known_answers_and_bootstrap():
    K = na.K
    eng = na.ExposureEngine(np.array([np.arange(10), np.arange(10, 20)]), np.ones(2, bool), 20)     # uniform
    assert eng.stat(np.ones(2, np.int32)) == pytest.approx((1.0, 0.0), abs=1e-12)
    eng = na.ExposureEngine(np.tile(np.arange(10), (4, 1)), np.ones(4, bool), 20)                  # concentrated
    cov, gini = eng.stat(np.ones(4, np.int32))
    assert (cov, gini) == pytest.approx((0.5, 0.5), abs=1e-12)                                      # 10 items x 4 + 10 zeros
    eng = na.ExposureEngine(np.zeros((4, K), int), np.ones(4, bool), 20)                            # one item in every list
    cov, gini = eng.stat(np.ones(4, np.int32))
    assert cov == pytest.approx(1 / 20) and gini == pytest.approx(1 - 1 / 20)                       # duplicates count once
    rng = np.random.default_rng(0)
    E, P, B = 150, 400, 20
    top = rng.integers(0, 150, size=(E, K))
    member = rng.random(E) < 0.9
    boot = na.EventBoot(E, B, 0)
    cov_r, gini_r = na.ExposureEngine(top, member, P).reps(boot.W)
    for b in range(B):
        cnt = np.zeros(P)
        for e in boot.resample_idx(b):
            if member[e]:
                for it in set(top[e].tolist()):
                    cnt[it] += 1
        x = np.sort(cnt)
        gini = 2 * np.sum(np.arange(1, P + 1) * x) / (P * x.sum()) - (P + 1) / P
        assert cov_r[b] == pytest.approx((cnt > 0).sum() / P, abs=1e-12)
        assert gini_r[b] == pytest.approx(gini, abs=1e-10)
    rec = na.ci_recentred(0.3, np.array([0.1, 0.12, 0.11, 0.13]), 5)
    assert rec["lo"] > 0.25 and rec["hi"] < 0.35 and rec["lo_raw"] < 0.12             # recentred at the point estimate


def test_bias_engine_equals_bias_index_and_bias_index_ci():
    rng = np.random.default_rng(3)
    E, K = 300, 10
    conf = np.round(rng.beta(1, 8, size=(E, K)) * 200) / 200
    conf[:, 1] = conf[:, 0]                                                        # within-event ties
    y = (rng.random((E, K)) < conf + 0.1 * (rng.random((E, K)) < 0.5)).astype(float)
    G = rng.choice(3, size=(E, K), p=[0.6, 0.25, 0.15])
    ev = np.broadcast_to(np.arange(E)[:, None], (E, K)).reshape(-1)
    c, yy, g = conf.reshape(-1), y.reshape(-1), G.reshape(-1)
    names = np.asarray(na.GROUPS)[g]
    eng = na.BiasEngine(c, yy, g, ev)
    est = metrics.bias_index(c, yy, names, adjust=True)
    assert eng.stat(np.ones(E, np.int32)) == pytest.approx([est[n] for n in na.GROUPS], abs=1e-12)
    boot = na.EventBoot(E, 20, 0)
    reps = eng.reps(boot.W)
    ci = metrics.bias_index_ci(c, yy, names, ev, adjust=True, n_boot=20, seed=0)
    for i, name in enumerate(na.GROUPS):
        lo, hi = stats.percentile_ci(reps[:, i])
        assert (ci[name]["est"], ci[name]["lo"], ci[name]["hi"]) == pytest.approx((est[name], lo, hi), abs=1e-9)


# ===================================================================================== ranks, ties, unscored, exposure
def test_tie_aware_ranks_and_exact_tie_group_metrics_match_reference():
    rng = np.random.default_rng(0)
    E, N = 300, 15
    L = np.round(rng.normal(size=(E, N)) * 2) / 2                                  # many ties
    pos = rng.integers(0, N, E)
    L[rng.random((E, N)) < 0.06] = np.nan                                          # unscored candidates
    L[5, pos[5]] = np.nan                                                          # an unscored positive
    L[7, :] = np.nan                                                               # no scored candidate at all
    rc = na.rank_core(L, np.ones((E, N), bool), pos)
    for e in range(E):
        ref = pilot_mirror._event_metrics(L[e].copy(), int(pos[e]))
        assert rc["rank"][e] == stats.tie_aware_rank(L[e], int(pos[e]))
        for mine, theirs in (("ndcg10", "NDCG@10"), ("hr10", "HR@10"), ("mrr", "MRR"), ("ndcg10_x", "NDCG@10_tie_exact"),
                             ("hr10_x", "HR@10_tie_exact"), ("mrr_x", "MRR_tie_exact")):
            assert rc[mine][e] == pytest.approx(ref[theirs], abs=1e-12), (e, mine)
    assert rc["rank"][5] == N and rc["rank"][7] == N                               # unscored positive ranks last
    assert not rc["valid"][7] and rc["valid"][5]
    # padded candidates (event with fewer candidates) rank exactly like the shorter list
    cm = np.ones((E, N), bool)
    cm[3, 9:] = False
    rc2 = na.rank_core(L, cm, pos)
    short = L[3, :9]
    if pos[3] < 9:
        assert rc2["rank"][3] == stats.tie_aware_rank(short, int(pos[3]))
    # a positive tie group straddling rank 10: the exact expectation differs from the plug-in expected-rank metric
    for scores, posi, rank, hr_plug, hr_exact in (
            ([5.0] * 9 + [0.0] * 4, 11, 11.5, 0.0, 1 / 4),             # ranks 10..13 equally likely
            ([9.0] * 9 + [1.0] * 3, 11, 11.0, 0.0, 1 / 3),             # ranks 10..12
            ([9.0] * 8 + [1.0] * 4, 8, 10.5, 0.0, 2 / 4)):             # ranks 9..12
        rt = na.rank_core(np.array([scores]), np.ones((1, len(scores)), bool), np.array([posi]))
        ref = pilot_mirror._event_metrics(np.array(scores), posi)
        assert rt["rank"][0] == rank and rt["hr10"][0] == hr_plug and rt["hr10_x"][0] == pytest.approx(hr_exact)
        assert rt["hr10_x"][0] == pytest.approx(ref["HR@10_tie_exact"]) and rt["ndcg10_x"][0] == pytest.approx(
            ref["NDCG@10_tie_exact"])


def test_top10_head_share_deterministic_ties_and_tied_cutoff(tmp_path):
    N = 15
    grp = np.array([[0, 0, 1, 2, 0, 0, 0, 1,   0, 2, 0, 2, 0,   1, 2],               # event 0: cutoff tie
                    [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]], np.int8)
    L = np.array([[10, 9, 8, 7, 6, 5, 4, 3,   2, 2, 2, 2, 2,   0, 0],                # c8..c12 tied at the cutoff
                  [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]], float)       # no tie
    pos = np.array([3, 14])
    panel, qv = qctx(tmp_path, L, grp, pos)
    assert qv["cutoff_tied"].tolist() == [True, False]
    # deterministic: the tied candidates enter in cand_idx order -> c8, c9 (head, tail)
    assert qv["top_cand"][0].tolist() == list(range(10))
    assert [(qv["top_grp"][0] == g).sum() / 10 for g in range(3)] == [0.6, 0.2, 0.2]
    # random tie-breaking: 8 strictly above (head 5, mid 2, tail 1) + 2 slots filled from 5 tied (3 head, 2 tail)
    assert qv["tie_exp_share"][0] == pytest.approx([(5 + 2 * 3 / 5) / 10, 2 / 10, (1 + 2 * 2 / 5) / 10])
    assert qv["tie_exp_share"][1] == pytest.approx([(grp[1, :10] == g).sum() / 10 for g in range(3)])
    assert qv["pool_cnt"].tolist() == [[(grp[e] == g).sum() for g in range(3)] for e in range(2)]
    boot = na.EventBoot(2, 10, 0)
    batch = na.LinearBatch(boot)
    blk = na.block_exposure(batch, boot, qv["top_grp"], qv["pool_cnt"], qv["pos_grp"], qv["top_item"], qv["valid"], 40, 10,
                            tie_exp=qv["tie_exp_share"])
    batch.run()
    blk = na.resolve(blk)
    head_share = ((qv["top_grp"] == 0).sum(1) / 10).mean()
    pool_head = (grp == 0).sum(1) / N
    assert blk["head_share_top10"]["est"] == pytest.approx(head_share)
    assert blk["pool_head_share"]["est"] == pytest.approx(pool_head.mean())
    assert blk["delta_head"]["est"] == pytest.approx(head_share - pool_head.mean())
    assert blk["target_head_share"]["est"] == pytest.approx(np.mean(grp[np.arange(2), pos] == 0))
    assert blk["delta_head_vs_target"]["est"] == pytest.approx(head_share - np.mean(grp[np.arange(2), pos] == 0))
    assert blk["tie_expected"]["head_share_top10"]["est"] == pytest.approx(np.mean(qv["tie_exp_share"][:, 0]))
    assert qv["tie_exp_share"][0, 0] == pytest.approx(0.62)


def test_unscored_candidates_ranking_probabilities_brier_nll_and_exclusion(tmp_path):
    N, E = 12, 3
    L = np.arange(N, dtype=float)[None, :].repeat(E, 0)[:, ::-1].copy()                 # event e: c0 best ... c11 worst
    grp = np.zeros((E, N), np.int8)
    pos = np.array([2, 4, 6])
    L[0, 0] = np.nan                                                                     # unscored NEGATIVE above the positive
    L[1, 4] = np.nan                                                                     # unscored POSITIVE
    L[2, :] = np.nan                                                                     # event without any scored candidate
    panel, qv = qctx(tmp_path, L, grp, pos)
    assert qv["rank"][0] == 2.0                      # the unscored c0 does not outrank the positive (c2 -> rank 2 of scored)
    assert qv["rank"][1] == N and qv["rank"][2] == N
    assert qv["valid"].tolist() == [True, True, False]
    assert qv["pos_scored"].tolist() == [True, False, False]
    P = qv["P"]
    assert P[0, 0] == 0.0 and P[0].sum() == pytest.approx(1.0) and P[2].sum() == 0.0   # unscored candidates carry p = 0
    assert np.isnan(qv["nll"][1]) and np.isnan(qv["nll"][2]) and np.isfinite(qv["nll"][0])
    assert qv["brier"][1] == pytest.approx((P[1] ** 2).sum() + 1.0)                  # an unscored positive adds 1
    # the unscored positive and the empty event are excluded from the temperature fit (counted)
    Tinfo = na.fit_temperature(qv["S"], panel.pos)
    assert Tinfo["n_events_used"] == 1 and Tinfo["n_events_positive_unscored"] == 1
    assert Tinfo["n_events_no_scored_candidate"] == 1
    assert qv["top1_correct"].tolist() == [0.0, 0.0, 0.0]


def test_load_scores_diagnostics_duplicates_mismatches_and_censoring(tmp_path):
    L, grp, pos, _ = synth_arrays(4, 12, 3)
    rows = make_panel_rows(4, 12, 3, pos, grp)
    panel = panel_from(tmp_path, rows)
    path = tmp_path / "s.csv.gz"
    write_scores(path, rows, {"next": L, "like": L + 1}, censored={("next", 0, 1): 2, ("next", 0, 2): 3, ("next", 1, 0): 1})
    with gzip.open(path, "at", newline="", encoding="utf-8") as f:                     # extra rows exercising every counter
        w = csv.writer(f)
        w.writerow([rows[0]["source_event_id"], "u", rows[0]["candidate_item_ids"][4], 4, 0, "next", "-1", "-1", "99.0", "1", 0])
        w.writerow(["no::such", "u", "x", 0, 0, "next", "-1", "-1", "1.0", "1", 0])             # event not in the panel
        w.writerow([rows[0]["source_event_id"], "u", "WRONG_ITEM", 5, 0, "next", "-1", "-1", "1.0", "1", 0])   # item mismatch
        w.writerow([rows[0]["source_event_id"], "u", rows[0]["candidate_item_ids"][6], 6, 0, "dislike", "-1", "-1", "1", "1", 0])
        w.writerow([rows[0]["source_event_id"], "u", rows[0]["candidate_item_ids"][7], 99, 0, "next", "-1", "-1", "1", "1", 0])
        w.writerow([rows[0]["source_event_id"], "u", rows[0]["candidate_item_ids"][8], 8, 0, "next", "-1", "-1", "inf", "1", 0])
    Ls, diag = na.load_scores(path, panel, ("next", "like"))
    assert np.isnan(Ls["next"][0, 1]) and np.isnan(Ls["next"][0, 2])                  # censored 2 / 3 -> unscored
    assert Ls["next"][1, 0] == pytest.approx(L[1, 0], abs=1e-6)                       # censored 1 is kept
    assert Ls["next"][0, 4] == 99.0                                                   # a duplicated row keeps the last
    assert np.isnan(Ls["next"][0, 8])                                                 # a non-finite logit -> unscored
    assert diag["score_rows_without_panel_event"] == 1 and diag["score_rows_item_mismatch"] == 1
    assert diag["score_rows_other_question"] == 1 and diag["score_rows_cand_idx_out_of_range"] == 1
    cn = diag["by_question"]["next"]
    assert cn["duplicate"] == 2 and cn["1"] == 1 and cn["2"] == 1 and cn["3"] == 1 and cn["dropped_nonfinite_or_censored"] == 3
    assert cn["missing_rows"] == 0 and diag["by_question"]["like"]["missing_rows"] == 0
    # a score file that lacks rows leaves those candidates unscored and counted
    write_scores(path, rows[:3], {"next": L[:3]})
    _, diag = na.load_scores(path, panel, ("next",))
    assert diag["by_question"]["next"]["missing_rows"] == 12


# ======================================================================================= C: calibration and anatomy
def test_temperature_fit_recovers_planted_temperature_and_flags_bounds():
    rng = np.random.default_rng(0)
    E, N, T0 = 5000, 21, 2.5
    z = rng.normal(0, 1.5, (E, N))
    p = np.exp(z / T0)
    p /= p.sum(1, keepdims=True)
    cum = p.cumsum(1)
    pos = np.minimum((rng.random(E)[:, None] > cum).sum(1), N - 1)                  # positive ~ softmax(z / T0)
    fit = na.fit_temperature(z, pos)
    assert fit["T"] == pytest.approx(T0, abs=0.15) and fit["bound_hit"] is None
    assert fit["nll_fit"] < fit["nll_T1"] and fit["nll_fit"] < fit["nll_uniform"]
    assert fit["nll_uniform"] == pytest.approx(math.log(N)) and fit["ln_101"] == pytest.approx(math.log(101))
    assert abs(fit["grad_at_fit"]) < 1e-6
    assert na.mean_nll(z, pos, fit["beta"])[0] == pytest.approx(fit["nll_fit"])
    # bounded search: a perfectly separating score pushes beta to the upper bound (T = 0.1), noise to the lower bound
    z2 = rng.normal(0, 1.0, (1000, N))
    z2[np.arange(1000), pos[:1000]] += 60.0
    f2 = na.fit_temperature(z2, pos[:1000])
    assert f2["bound_hit"] == "upper" and f2["T"] == pytest.approx(0.1)
    pos3 = rng.integers(0, N, 2000)
    z3 = rng.normal(0, 1.0, (2000, N))
    z3[np.arange(2000), pos3] -= 5.0                                                 # anti-informative: flatten as far as allowed
    f3 = na.fit_temperature(z3, pos3)
    assert f3["bound_hit"] == "lower" and f3["T"] == pytest.approx(1000.0)


def test_ece_perfectly_calibrated_lists_and_overconfident_lists(tmp_path):
    N, n_per = 21, 60
    rows_L, rows_pos = [], []
    for conf, acc in ((0.3, 0.3), (0.6, 0.6), (0.9, 0.9)):
        gap = math.log((N - 1) * conf / (1 - conf))                                   # p_max = conf exactly
        for i in range(n_per):
            L = np.zeros(N)
            L[0] = gap
            rows_L.append(L)
            rows_pos.append(0 if i < round(acc * n_per) else 5)
    L, pos = np.array(rows_L), np.array(rows_pos)
    grp = np.zeros(L.shape, np.int8)
    panel, qv = qctx(tmp_path, L, grp, pos)
    assert set(np.round(qv["p_max"], 6)) == {0.3, 0.6, 0.9}
    out = run_sections(qv)
    c = out["C"]["list_normalised"]
    assert c["ece"]["est"] == pytest.approx(0.0, abs=1e-9) and c["ece_adaptive"]["est"] == pytest.approx(0.0, abs=1e-9)
    assert c["top1_accuracy"]["est"] == pytest.approx(0.6) and c["p_max_mean"]["est"] == pytest.approx(0.6)
    assert sum(b["n"] for b in c["reliability_bins"]) == 180
    for b in c["reliability_bins"]:
        assert b["mean_conf"] == pytest.approx(b["acc"], abs=1e-9)
    # the same confidences with half the accuracy of the high-confidence group -> an exact ECE
    pos2 = pos.copy()
    pos2[2 * n_per:] = np.where(np.arange(n_per) < 24, 0, 5)                       # 0.9-confidence lists hit only 40%
    _, qv2 = qctx(tmp_path, L, grp, pos2)
    c2 = run_sections(qv2)["C"]["list_normalised"]
    assert c2["ece"]["est"] == pytest.approx((60 * 0 + 60 * 0 + 60 * 0.5) / 180)
    # multiclass Brier of a one-hot-confident list: p_max = 0.9, others (1 - 0.9)/20 each
    assert qv["brier"][2 * n_per] == pytest.approx((1 - 0.9) ** 2 + 20 * (0.1 / 20) ** 2)


def test_auroc_signals_informative_confidence_beats_random_null(tmp_path):
    L, grp, pos, easy = synth_confident(600, 21, 0)
    panel, qv = qctx(tmp_path, L, grp, pos)
    out = run_sections(qv, n_boot=80)["C"]["auroc_discrimination"]
    for lab, floor in (("top1", 0.9), ("hr10", 0.7)):                              # hr10: hard events hit the top 10 by chance
        a = out[lab]["auroc"]
        assert a["p_max"]["est"] > floor and abs(a["random"]["est"] - 0.5) < 0.08
        assert out[lab]["pmax_minus_random"]["lo"] > 0.15                          # informative beats the fixed-seed null
        for sg in ("margin", "neg_entropy", "max_logit"):
            assert a[sg]["est"] > floor - 0.05
            assert out[lab]["paired_vs_pmax"][sg]["n"] == 600
        assert out[lab]["paired_vs_pmax"]["random"]["hi"] < 0                       # random is worse than p_max
    # the AUROC of p_max equals the reference implementation on the same arrays
    assert out["top1"]["auroc"]["p_max"]["est"] == pytest.approx(metrics.auroc(qv["p_max"], qv["top1_correct"].astype(int)))
    # no signal -> p_max does not beat random
    Lu, gu, pu, _ = synth_arrays(600, 21, 1, signal=0.0)
    _, qvu = qctx(tmp_path, Lu, gu, pu, seed=1)
    ou = run_sections(qvu, n_boot=80)["C"]["auroc_discrimination"]["top1"]
    assert ou["pmax_minus_random"]["lo"] < 0.0 < ou["pmax_minus_random"]["hi"]


def test_error_anatomy_confident_errors_and_unsure_correct(tmp_path):
    L, grp, pos, easy = synth_confident(600, 21, 2, frac_easy=0.4)
    panel, qv = qctx(tmp_path, L, grp, pos)
    an = run_sections(qv, n_boot=60)["C"]["error_anatomy"]
    tert = stats.rank_bins(qv["p_max"], 3)
    wrong = 1 - qv["top1_correct"]
    assert an["confident_error_rate"]["est"] == pytest.approx(wrong[tert == 2].mean())
    assert an["unsure_correct_rate"]["est"] == pytest.approx(qv["top1_correct"][tert == 0].mean())
    assert an["share_errors_in_top_tertile"]["est"] == pytest.approx(wrong[tert == 2].sum() / wrong.sum())
    assert an["share_correct_in_bottom_tertile"]["est"] == pytest.approx(qv["top1_correct"][tert == 0].sum()
                                                                         / qv["top1_correct"].sum())
    assert an["acc_top_minus_bottom_tertile"]["est"] == pytest.approx(
        qv["top1_correct"][tert == 2].mean() - qv["top1_correct"][tert == 0].mean())
    assert an["acc_top_minus_bottom_tertile"]["lo"] > 0.4                          # confidence is informative here
    assert an["confident_error_rate"]["est"] < 0.1 and an["unsure_correct_rate"]["est"] < 0.15
    assert an["share_errors_in_top_minus_third"]["est"] == pytest.approx(an["share_errors_in_top_tertile"]["est"] - 1 / 3)
    assert sum(t["n"] for t in an["tertiles"]) == 600
    assert an["tertiles"][0]["hr10"]["est"] <= an["tertiles"][2]["hr10"]["est"]


def test_pointwise_uauc_equals_mean_rank_auc_and_pooled_auroc(tmp_path):
    L, grp, pos, _ = synth_arrays(240, 21, 4, signal=1.5)
    panel, qv = qctx(tmp_path, L, grp, pos)
    pw = run_sections(qv, n_boot=30)["C"]["pointwise"]
    assert pw["uauc"]["est"] == pytest.approx(np.mean(1 - (qv["rank"] - 1) / 20))
    assert pw["uauc_minus_half"]["est"] == pytest.approx(pw["uauc"]["est"] - 0.5)
    y = np.zeros(L.shape, int)
    y[np.arange(240), pos] = 1
    assert pw["pooled_auroc"]["est"] == pytest.approx(metrics.auroc(L.reshape(-1), y.reshape(-1)))
    assert pw["pooled_auroc"]["n"] == 240 * 21 and pw["pooled_auroc"]["n_positive_rows"] == 240


# ===================================================================================== D: selective serving
def test_selective_serving_informative_confidence_has_lower_aurc_than_random(tmp_path):
    L, grp, pos, _ = synth_confident(600, 21, 3)
    panel, qv = qctx(tmp_path, L, grp, pos)
    d = run_sections(qv, n_boot=60)["D"]
    ps = d["signals"]["p_max"]
    for u in ("ndcg10", "hr1"):
        dd = ps["d_aurc_vs_random_expectation"][u]
        assert dd["est"] < -0.05 and dd["hi"] < 0                                   # lower risk than the random expectation
        assert ps["gain_at_50_vs_full"][u]["lo"] > 0                                # serving the confident half pays
        assert ps["curve"][u][0]["est"] > ps["curve"][u][-1]["est"]                 # utility falls as coverage grows
        assert ps["curve"][u][-1]["est"] == pytest.approx(ps["mean_utility"][u]["est"])
        assert ps["random_expectation_aurc"][u]["est"] == pytest.approx(1 - ps["mean_utility"][u]["est"])
    rnd = d["signals"]["random"]["d_aurc_vs_random_expectation"]["ndcg10"]
    assert rnd["lo"] < 0 < rnd["hi"] or abs(rnd["est"]) < 0.03                      # the null signal is not informative
    # AURC point estimate is metrics.risk_coverage
    assert ps["aurc"]["ndcg10"]["est"] == pytest.approx(metrics.risk_coverage(qv["p_max"], qv["ndcg10"])[2])
    assert ps["aurc"]["hr1"]["est"] == pytest.approx(metrics.risk_coverage(qv["p_max"], qv["hr1"])[2])
    assert [c["coverage"] for c in ps["curve"]["ndcg10"]] == list(na.COVERAGE)


def test_serving_signal_anticorrelated_with_niche_users_serves_fewer_niche_events():
    rng = np.random.default_rng(0)
    n = 4000
    profile = rng.integers(0, 9, n) / 4.0                                            # ties, 9 distinct values
    quint = np.full(n, -1)
    quint[:] = stats.rank_bins(profile, 5)
    util = rng.random(n)
    anti = quint + rng.normal(0, 0.3, n)                                             # confident for mainstream users
    boot = na.EventBoot(n, 40, 0)
    res_anti = na.ServeEngine(anti, {"ndcg10": util}, quint).reps(boot.W)
    res_rand = na.ServeEngine(rng.random(n), {"ndcg10": util}, quint).reps(boot.W)
    est_anti = na.ServeEngine(anti, {"ndcg10": util}, quint).reps(np.ones((1, n), np.int16))
    est_rand = na.ServeEngine(rng.random(n), {"ndcg10": util}, quint).reps(np.ones((1, n), np.int16))
    share_anti = est_anti["served"][0] / est_anti["size"][0]
    share_rand = est_rand["served"][0] / est_rand["size"][0]
    assert share_anti[0] < 0.1 and share_anti[4] > 0.9                               # niche users are silenced
    assert np.all(np.abs(share_rand - 0.5) < 0.07)                                   # the random null serves everyone alike
    assert est_anti["served"][0].sum() == pytest.approx(n / 2, abs=1e-9)             # exactly 50% coverage
    diff = res_anti["served"][:, 0] / res_anti["size"][:, 0] - res_anti["served"][:, 4] / res_anti["size"][:, 4]
    assert diff.max() < -0.7
    drnd = res_rand["served"][:, 0] / res_rand["size"][:, 0] - res_rand["served"][:, 4] / res_rand["size"][:, 4]
    assert abs(drnd.mean()) < 0.15


def test_sec_d_niche_profile_quintiles_and_exclusion(tmp_path):
    L, grp, pos, _ = synth_confident(500, 21, 5)
    panel, qv = qctx(tmp_path, L, grp, pos)
    profile = np.random.default_rng(3).integers(0, 9, 500) / 4.0
    profile[:25] = np.nan                                                            # users without a mapped history item
    d = run_sections(qv, profile=profile, n_boot=40)["D"]
    assert d["n_events_without_profile"] == 25 and sum(d["quintile_sizes"]) == 475
    ps = d["signals"]["p_max"]["niche"]
    assert len(ps["served_share"]) == 5 and len(ps["utility_among_served"]["ndcg10"]) == 5
    assert ps["niche_minus_mainstream_served_share"]["n"] == d["quintile_sizes"][0] + d["quintile_sizes"][4]
    # every served share is a share
    for c in ps["served_share"]:
        assert 0.0 <= c["est"] <= 1.0
    # the half of the events that is served (profile-less events included) is exactly 50%
    served = sum(c["est"] * n for c, n in zip(ps["served_share"], d["quintile_sizes"]))
    assert 0.35 * 475 < served < 0.65 * 475


def test_sec_d_niche_difference_uses_the_extreme_non_empty_quintiles_under_heavy_ties(tmp_path):
    """Real sports profiles are tied at the top (about half of the users have every mapped history item in the head group):
    rank_bins then leaves the top quintile empty and the registered bin-0 vs bin-4 difference would be undefined."""
    L, grp, pos, _ = synth_confident(500, 21, 5)
    panel, qv = qctx(tmp_path, L, grp, pos)
    rng = np.random.default_rng(3)
    profile = np.where(rng.random(500) < 0.55, 2.0, rng.integers(0, 8, 500) / 4.0)       # 55% of the users tied at the top
    d = run_sections(qv, profile=profile, n_boot=40)["D"]
    assert d["quintile_sizes"][4] == 0 and sum(d["quintile_sizes"]) == 500              # the top quintile is empty
    ps = d["signals"]["p_max"]["niche"]
    lo, hi = ps["niche_bin"], ps["mainstream_bin"]
    assert lo == 0 and hi == max(j for j, s in enumerate(d["quintile_sizes"]) if s > 0) < 4
    diff = ps["niche_minus_mainstream_served_share"]
    assert diff["est"] is not None and diff["n"] == d["quintile_sizes"][lo] + d["quintile_sizes"][hi]
    assert diff["est"] == pytest.approx(ps["served_share"][lo]["est"] - ps["served_share"][hi]["est"])


# ====================================================================================== E: popularity and S5
def _fake_e_qv(E=700, K=10, seed=0, shift=(-0.12, 0.0, 0.12), p_by_group=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    G = rng.choice(3, size=(E, K), p=[0.5, 0.3, 0.2]).astype(np.int8)
    p = rng.uniform(0.02, 0.5, (E, K)) + np.asarray(p_by_group)[G]
    y = (rng.random((E, K)) < np.clip(p + np.asarray(shift)[G], 0, 1)).astype(float)
    return {"top_grp": G, "top_p": p, "top_y": y, "top_lc": rng.normal(size=(E, K)) + np.asarray(p_by_group)[G] * 10,
            "top_scored": np.ones((E, K), bool)}, p, y, G


def test_bias_index_planted_group_miscalibration_and_s5_summaries():
    qv, p, y, G = _fake_e_qv()
    E = len(G)
    boot = na.EventBoot(E, 60, 0)
    batch = na.LinearBatch(boot)
    out = na.sec_E(batch, boot, qv, np.ones(E, bool), 60)
    batch.run()
    out = na.resolve(out)
    bi = out["bias_index"]
    assert bi["head"]["est"] < -0.05 and bi["head"]["hi"] < 0                       # head candidates are over-confident
    assert bi["tail"]["est"] > 0.05 and bi["tail"]["lo"] > 0                        # tail candidates are under-confident
    assert bi["head_minus_tail"]["est"] < -0.1 and bi["head_minus_tail"]["hi"] < 0
    # same estimates and intervals as metrics.bias_index_ci (same stream: seed 0, event clusters)
    names = np.asarray(na.GROUPS)[G.reshape(-1)]
    ev = np.broadcast_to(np.arange(E)[:, None], G.shape).reshape(-1)
    ref = metrics.bias_index_ci(p.reshape(-1), y.reshape(-1), names, ev, adjust=True, n_boot=60, seed=0)
    for g in na.GROUPS:
        assert (bi[g]["est"], bi[g]["lo"], bi[g]["hi"]) == pytest.approx((ref[g]["est"], ref[g]["lo"], ref[g]["hi"]), abs=1e-9)
    res = out["residual_y_minus_p"]
    assert res["head"]["est"] < -0.08 and res["tail"]["est"] > 0.08                 # mean(y - p) by group
    assert out["head_minus_tail_residual"]["est"] == pytest.approx(res["head"]["est"] - res["tail"]["est"])
    assert out["head_minus_tail_residual"]["hi"] < 0
    assert out["n_rows"] == E * 10 and sum(out["n_rows_by_group"].values()) == E * 10
    mp = out["mean_p"]
    for g, name in enumerate(na.GROUPS):
        assert mp[name]["est"] == pytest.approx(p[G == g].mean())
    assert out["top1_head_fraction"]["est"] == pytest.approx(np.mean(G[:, 0] == 0))


def test_popularity_graded_confidence_head_minus_tail_mean_p():
    qv, p, y, G = _fake_e_qv(shift=(0, 0, 0), p_by_group=(0.15, 0.05, 0.0))          # popular items get higher p
    E = len(G)
    boot = na.EventBoot(E, 50, 0)
    batch = na.LinearBatch(boot)
    out = na.sec_E(batch, boot, qv, np.ones(E, bool), 30)
    batch.run()
    out = na.resolve(out)
    assert out["head_minus_tail_mean_p"]["est"] == pytest.approx(0.15, abs=0.02)
    assert out["head_minus_tail_mean_p"]["lo"] > 0.1 and out["head_minus_tail_mean_p"]["p_boot"] < 0.1
    assert out["mean_centred_logit"]["head"]["est"] > out["mean_centred_logit"]["tail"]["est"]
    # calibrated by construction -> the confidence-matched gap is ~0 although the confidence level differs
    assert abs(out["bias_index"]["head"]["est"]) < 0.03 and out["bias_index"]["head_minus_tail"]["lo"] < 0.03


# =========================================================================================== F: summary and S3
def _ci(lo, hi, est=None, p=None):
    est = (lo + hi) / 2 if est is None else est
    return {"est": est, "lo": lo, "hi": hi, "n": 100, "n_boot": 100, "null": 0.0,
            "p_boot": (0.001 if (lo > 0 or hi < 0) else 0.4) if p is None else p}


def _doc(domain, delta, role="all", name=None, seg_name="all", ref_delta=None):
    seg = {"role": role, "event_range": [1, 100], "n_events": 100,
           "questions": {"next": {"B_exposure": {"llm": {"delta_head": _ci(*delta)}}}},
           "reference": {"ccrp_v3": {"B_exposure": {"delta_head": _ci(*(ref_delta or (-0.1, 0.1)))}}}}
    return {"domain": domain, "temperature": {"next": {"T": 1.0}}, "segments": {seg_name: seg}}


POS, NULL, NEG = (0.05, 0.15), (-0.05, 0.05), (-0.15, -0.05)


def test_s3_admission_three_of_four_domains_same_sign():
    def admission(*deltas, sports_quarantine=None):
        docs = [_doc("sports", deltas[0], "main", seg_name="events_1001_10000"), _doc("toys", deltas[1]),
                _doc("home", deltas[2]), _doc("tools", deltas[3])]
        if sports_quarantine:
            docs[0]["segments"]["events_1_1000"] = _doc("sports", sports_quarantine, "quarantine")["segments"]["all"]
        s = na.summarize(docs)
        return s, s["S3"]["admission"]["llm_next"]
    s, a = admission(POS, POS, POS, NULL)
    assert a["effect_claimed"] and a["sign"] == 1 and a["n_domains_ci_excludes_0_positive"] == 3
    assert s["family_units"] == ["sports", "toys", "home", "tools"] and s["complete_family"]
    _, a = admission(POS, POS, NULL, NULL)
    assert not a["effect_claimed"] and a["sign"] == 0
    _, a = admission(POS, POS, POS, NEG)                                              # 3 positive, 1 negative: claimed positive
    assert a["effect_claimed"] and a["sign"] == 1 and a["n_domains_ci_excludes_0_negative"] == 1
    _, a = admission(POS, POS, NEG, NEG)
    assert not a["effect_claimed"]
    _, a = admission(NEG, NEG, NEG, NULL)
    assert a["effect_claimed"] and a["sign"] == -1
    _, a = admission(POS, POS, POS, POS)
    assert a["effect_claimed"] and a["n_domains_with_data"] == 4
    # the sports quarantine segment is reported but never counted: main NULL + quarantine POS + two POS domains
    s, a = admission(NULL, POS, POS, NULL, sports_quarantine=POS)
    assert not a["effect_claimed"] and a["n_domains_ci_excludes_0_positive"] == 2
    assert "sports_events_1_1000" in s["units"] and "sports_events_1_1000" not in s["family_units"]
    assert "sports_events_1_1000" in s["S3"]["endpoints"]["llm_next"]["delta_head"]["values"]
    # the registered rule is 3 of the 4 domains: an incomplete family never admits an effect
    s = na.summarize([_doc("toys", POS), _doc("home", POS), _doc("tools", POS)])
    a = s["S3"]["admission"]["llm_next"]
    assert not a["effect_claimed"] and not a["complete_family"] and a["n_domains_ci_excludes_0_positive"] == 3
    assert not s["complete_family"]
    s = na.summarize([_doc("toys", POS), _doc("home", POS)])
    assert not s["S3"]["admission"]["llm_next"]["effect_claimed"]
    # reference methods get the same rule
    docs = [_doc(d, POS, ref_delta=POS) for d in ("sports", "toys", "home", "tools")]
    assert na.summarize(docs)["S3"]["admission"]["ccrp_v3"]["effect_claimed"]


def test_summary_holm_sign_statements_over_family_domains():
    docs = []
    for d, p in zip(("sports", "toys", "home", "tools"), (0.001, 0.01, 0.2, 0.03)):
        doc = _doc(d, POS)
        doc["segments"]["all"]["questions"]["next"]["C_calibration"] = {"error_anatomy": {
            "acc_top_minus_bottom_tertile": _ci(0.1, 0.2, p=p)}}
        docs.append(doc)
    s = na.summarize(docs)
    h = s["S1"]["acc_top_minus_bottom_tertile"]["next"]["holm"]
    assert h["m"] == 4
    assert h["p_holm"]["sports"] == pytest.approx(0.004) and h["p_holm"]["toys"] == pytest.approx(0.03)
    assert h["p_holm"]["tools"] == pytest.approx(0.06) and h["p_holm"]["home"] == pytest.approx(0.2)
    assert h["reject_holm"] == {"sports": True, "toys": True, "home": False, "tools": False}
    assert set(h["sign"].values()) == {1}
    assert s["S1"]["unsure_correct_rate"] == {}                                       # endpoints without data are empty, not invented
    json.dumps(s, allow_nan=False)                                                    # numbers only, strictly serialisable


# ============================================================================================ loading / profiles
def test_load_panel_streaming_subset_validation_and_user_profile(tmp_path):
    L, grp, pos, _ = synth_arrays(6, 12, 7)
    rows = make_panel_rows(6, 12, 7, pos, grp)
    rows[0]["history_item_ids"] = [rows[0]["candidate_item_ids"][0], rows[0]["candidate_item_ids"][1], "NOT_IN_POOL"]
    rows[1]["history_item_ids"] = ["NOT_IN_POOL"]
    rows[2]["history_item_ids"] = []
    panel = panel_from(tmp_path, rows)
    assert panel.E == 6 and panel.N == 12 and panel.pool_conflicts == 0
    sub = panel_from(tmp_path, rows, name="p2.jsonl", only_events={rows[1]["source_event_id"], rows[4]["source_event_id"]})
    assert sub.ev_ids == [rows[1]["source_event_id"], rows[4]["source_event_id"]]
    prof, diag = na.user_profile(panel)
    g = {rows[0]["candidate_item_ids"][0]: rows[0]["candidate_popularity_groups"][0],
         rows[0]["candidate_item_ids"][1]: rows[0]["candidate_popularity_groups"][1]}
    score = {"head": 2.0, "mid": 1.0, "tail": 0.0}
    assert prof[0] == pytest.approx(np.mean([score[v] for v in g.values()]))          # unmapped history items are skipped
    assert np.isnan(prof[1]) and np.isnan(prof[2])                                    # no mapped item -> excluded
    assert diag["n_events_without_mapped_history"] >= 2
    bad = [dict(rows[0], candidate_popularity_groups=["huge"] + rows[0]["candidate_popularity_groups"][1:])]
    write_jsonl(tmp_path / "bad.jsonl", bad)
    with pytest.raises(ValueError, match="unknown popularity group"):
        na.load_panel(tmp_path / "bad.jsonl")
    write_jsonl(tmp_path / "dup.jsonl", [rows[0], rows[0]])
    with pytest.raises(ValueError, match="duplicated source_event_id"):
        na.load_panel(tmp_path / "dup.jsonl")
    write_jsonl(tmp_path / "pos.jsonl", [dict(rows[0], positive_item_index=99)])
    with pytest.raises(ValueError, match="positive_item_index"):
        na.load_panel(tmp_path / "pos.jsonl")


def test_ref_exposure_join_excludes_disagreeing_events(tmp_path):
    d = make_domain(tmp_path, "toys", E=12, N=15, seed=2, n_valid=10)
    panel = na.load_panel(d["panel_test"])
    ok = na.load_ref_exposure(d["ref_exposure"] / "ccrp_v3.csv.gz", panel)
    assert ok["member"].all() and ok["diag"]["n_member"] == 12
    # tamper with a copy: wrong pool count, wrong positive group, unknown item, other user, duplicate row, foreign event
    with gzip.open(d["ref_exposure"] / "ccrp_v3.csv.gz", "rt", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0]["n_head_pool"] = str(int(rows[0]["n_head_pool"]) + 1)
    rows[1]["positive_group"] = "tail" if rows[1]["positive_group"] != "tail" else "head"
    rows[2]["top10_item_ids"] = "NOPE " + " ".join(rows[2]["top10_item_ids"].split(" ")[1:])
    rows[3]["user_id"] = "someone_else"
    rows.append(dict(rows[4]))
    rows.append(dict(rows[5], source_event_id="foreign::1"))
    bad = tmp_path / "bad_exposure.csv.gz"
    with gzip.open(bad, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    t = na.load_ref_exposure(bad, panel)
    assert not t["member"][:4].any() and t["member"][4:].all()
    dg = t["diag"]
    assert dg["pool_count_mismatch"] == 1 and dg["positive_group_mismatch"] == 1 and dg["top10_item_not_in_pool"] == 1
    assert dg["user_id_mismatch"] == 1 and dg["duplicate_rows"] == 1 and dg["rows_not_in_panel"] == 1
    rank, rd = na.align_ref_ranks(d["ref_ranks"] / "ccrp_v3.csv.gz", panel)
    assert np.isfinite(rank).all() and rd["n_joined"] == 12


# ==================================================================================== quarantine / end-to-end run
def test_sports_quarantine_segments_and_separate_reporting(tmp_path):
    segs = na.make_segments("sports", 10000, None)
    assert [(s["name"], s["role"], s["lo"], s["hi"]) for s in segs] == [
        ("events_1_1000", "quarantine", 0, 1000), ("events_1001_10000", "main", 1000, 10000)]
    assert na.make_segments("toys", 10000, None) == [{"name": "all", "role": "all", "lo": 0, "hi": 10000}]
    assert na.make_segments("sports", 1000, None)[0]["role"] == "quarantine" and len(na.make_segments("sports", 1000, None)) == 1
    d = make_domain(tmp_path, "sports", E=40, N=15, seed=3, n_valid=30)
    res = na.run_domain("sports", d["audit"], d["panel_test"], d["panel_valid"], d["ref_ranks"], d["ref_exposure"],
                        n_boot=20, quarantine_n=15)
    assert list(res["segments"]) == ["events_1_15", "events_16_40"]
    q, mn = res["segments"]["events_1_15"], res["segments"]["events_16_40"]
    assert (q["role"], q["event_range"], q["n_events"]) == ("quarantine", [1, 15], 15)
    assert (mn["role"], mn["event_range"], mn["n_events"]) == ("main", [16, 40], 25)
    # the two segments are computed on their own events: direct recomputation from the arrays
    L = d["L"]["next"]
    pos = d["pos"]
    rank = np.array([stats.tie_aware_rank(L[e], int(pos[e])) for e in range(40)])
    nd = metrics.ndcg_from_rank(rank, 10)
    assert q["questions"]["next"]["A_ranking"]["ranking"]["ndcg10"]["est"] == pytest.approx(nd[:15].mean())
    assert mn["questions"]["next"]["A_ranking"]["ranking"]["ndcg10"]["est"] == pytest.approx(nd[15:].mean())
    hs = lambda sl: np.mean([(d["grp"][e][np.argsort(-L[e], kind="stable")[:10]] == 0).mean() for e in range(40)[sl]])
    assert q["questions"]["next"]["B_exposure"]["llm"]["head_share_top10"]["est"] == pytest.approx(hs(slice(0, 15)))
    assert mn["questions"]["next"]["B_exposure"]["llm"]["head_share_top10"]["est"] == pytest.approx(hs(slice(15, 40)))
    # temperature: fitted once per question on VALID2k and shared by the segments
    assert q["questions"]["next"]["C_calibration"]["temperature"]["T"] == res["temperature"]["next"]["T"]
    assert q["questions"]["next"]["C_calibration"]["temperature"] is not None and mn["questions"]["like"]["n_events_valid"] == 25


@pytest.fixture(scope="module")
def four_domains(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("audit4")
    outs = []
    # three domains with a strong head bias in the LLM scores (S3 effect), one without
    for i, (dom, bias) in enumerate((("sports", 2.5), ("toys", 2.5), ("home", 2.5), ("tools", 0.0))):
        d = make_domain(tmp, dom, E=36, N=15, seed=10 + i, n_valid=30, head_bias=bias)
        out = tmp / f"{dom}.json"
        na.main(["run", "--domain", dom, "--audit_dir", str(d["audit"]), "--panel_test", str(d["panel_test"]),
                 "--panel_valid", str(d["panel_valid"]), "--ref_ranks", str(d["ref_ranks"]),
                 "--ref_exposure", str(d["ref_exposure"]), "--out", str(out), "--n_boot", "40", "--seed", "0",
                 "--quarantine_n", "12" if dom == "sports" else "0"])
        outs.append(out)
    summ = tmp / "summary.json"
    na.main(["summarize", "--inputs", ",".join(map(str, outs)), "--out", str(summ)])
    return outs, summ, tmp


def test_end_to_end_run_json_structure_and_strictness(four_domains):
    outs, _, _ = four_domains
    for p in outs:
        text = p.read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text
        doc = json.loads(text)
        assert doc["schema"] == na.SCHEMA and doc["questions"] == ["next", "like"] and doc["n_boot"] == 40
        assert set(doc["definition"]) >= {"quarantine", "unscored", "rank", "temperature", "S3", "tertiles", "niche"}
        for seg in doc["segments"].values():
            for q in ("next", "like"):
                qb = seg["questions"][q]
                assert set(qb) >= {"A_ranking", "B_exposure", "C_calibration", "D_selective_serving", "E_popularity"}
                for sg in na.SIGNALS:
                    assert sg in qb["D_selective_serving"]["signals"]
                assert set(qb["A_ranking"]["vs_reference"]) == {"ccrp_v3", "llmesr_sasrec"}
                assert set(qb["B_exposure"]["vs_reference"]) == {"ccrp_v3", "llmesr_sasrec"}
                ex = qb["B_exposure"]["llm"]
                for key in ("head_share_top10", "mid_share_top10", "tail_share_top10", "pool_head_share", "delta_head",
                            "delta_tail", "target_head_share", "delta_head_vs_target", "coverage_top10", "gini_exposure"):
                    assert key in ex and ex[key]["n"] > 0
                assert ex["delta_head"]["est"] == pytest.approx(ex["head_share_top10"]["est"] - ex["pool_head_share"]["est"])
            assert set(seg["reference"]) == {"ccrp_v3", "llmesr_sasrec"}
            for r in seg["reference"].values():
                assert "delta_head" in r["B_exposure"] and "ndcg10" in r["A_ranking"]["ranking"]
        assert doc["data"]["ref_exposure"]["ccrp_v3"]["n_member"] == 36
        assert doc["data"]["pool_json"]["matches_panel"] is True                       # _pool.json n_pool_items == panel pool
        assert doc["data"]["panel"]["n_pool_items"] == doc["n_pool_items"] == doc["data"]["pool_json"]["n_pool_items"]


def test_end_to_end_summarize_s3_admission_and_units(four_domains):
    _, summ, _ = four_domains
    s = json.loads(summ.read_text(encoding="utf-8"))
    assert s["schema"] == na.SUMMARY_SCHEMA and s["complete_family"]
    assert s["family_units"] == ["sports", "toys", "home", "tools"] or sorted(s["family_units"]) == sorted(
        ["sports", "toys", "home", "tools"])
    assert any(u.startswith("sports_events_1_12") for u in s["units"]) and s["units"]["sports"]["role"] == "main"
    adm = s["S3"]["admission"]["llm_next"]
    assert adm["effect_claimed"] and adm["sign"] == 1 and adm["n_domains_ci_excludes_0_positive"] == 3
    assert adm["per_domain_sign"]["tools"] == 0                                       # the unbiased domain shows no effect
    assert set(s) >= {"S1", "S2", "S3", "S4", "S5", "D", "A_context", "temperature", "alpha"}
    for name in ("pooled_auroc", "uauc", "ece", "nll"):
        assert "next" in s["S4"][name] and "sports" in s["S4"][name]["next"]["values"]
    holm = s["S4"]["pooled_auroc_minus_half"]["next"]["holm"]
    assert holm["m"] == 4 and all(0 <= v <= 1 for v in holm["p_holm"].values())
    assert s["S5"]["bias_index_head_minus_tail"]["next"]["holm"]["m"] == 4
    assert "ccrp_v3" in s["S3"]["admission"] and "llmesr_sasrec" in s["S3"]["admission"]
    # the LLM-minus-reference head-share differences exist per domain
    assert "head_share_top10_llm_minus_ref" in s["S3"]["endpoints"]["llm_next_vs_ccrp_v3"]


def test_end_to_end_censored_candidates_are_unscored_and_counted(tmp_path):
    cens = {("next", 0, 3): 2, ("next", 0, 4): 3, ("next", 5, 0): 1}
    d = make_domain(tmp_path, "tools", E=20, N=15, seed=21, n_valid=20, censored_test=cens)
    res = na.run_domain("tools", d["audit"], d["panel_test"], d["panel_valid"], d["ref_ranks"], d["ref_exposure"], n_boot=20)
    sc = res["data"]["scores_test"]["by_question"]["next"]
    assert sc["2"] == 1 and sc["3"] == 1 and sc["1"] == 1 and sc["dropped_nonfinite_or_censored"] == 2
    assert res["data"]["by_question"]["next"]["n_rows_unscored"] == 2
    assert res["data"]["by_question"]["like"]["n_rows_unscored"] == 0
    cens_a = res["segments"]["all"]["questions"]["next"]["A_ranking"]["censoring"]
    assert cens_a["n_events_any_unscored"] == 1 and cens_a["n_rows_unscored"] == 2


# ===================================================== Z2: second backbone on a subsample (Amendment 3 section 4)
# sha256 digests (`canonical_digest`) of the documents the module frozen in PILOT_LOG (sha1 8e3fa3d0...) wrote for the
# `four_domains` fixture: the four `run` documents and the `summarize` document. They were taken BEFORE the Z2 options
# existed; a change of any default output (a value, a key, the key order, a type) changes a digest.
FROZEN_DEFAULT_DIGESTS = {
    "sports": "2827d5c29370d4e76974a6a9a167248f0cbd5ce0f84a1b85b5d59645459baf12",
    "toys": "6d46d262f98155ead0d8169392287abfd85433bba8d4b4c69d6581f75f0e445a",
    "home": "878d74d0e8702353087351cded7a3a11688bd259bc0d1151e5b506ddec970df5",
    "tools": "6d590e551325807cdd1c16bc14db2d17388fdee545b3c70b08e6f656f0ab2f2a",
    "summary": "3182a6dc55ef46c59b2d9db9e2f0e906942df027f35273545317d20138c674b5",
}


def _jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _strict(x) -> str:
    return json.dumps(na.strict_json(x), allow_nan=False)


def _without(doc, *keys):
    return {k: v for k, v in doc.items() if k not in keys}


def test_default_output_is_identical_to_the_frozen_module(four_domains):
    outs, summ, tmp = four_domains
    got = {}
    for p in outs:
        doc = _jload(p)
        got[doc["domain"]] = canonical_digest(doc)
        assert "segment_mode" not in doc                                              # the Z2 options leave no trace by default
        assert list(doc["inputs"]) == ["audit_dir", "panel_test", "panel_valid", "ref_ranks", "ref_exposure"]
    got["summary"] = canonical_digest(_jload(summ))
    assert got == FROZEN_DEFAULT_DIGESTS
    # spelling the new options out with their default values is the very same run (tools: segment `all`, no quarantine)
    res = na.run_domain("tools", tmp / "audit", tmp / "panels" / "tools_test.jsonl", tmp / "panels" / "tools_valid.jsonl",
                        tmp / "docs/sigir/ref_ranks/tools", tmp / "docs/sigir/ref_exposure/tools", n_boot=40, quarantine_n=0,
                        segments="auto", test_role="test", valid_role="valid2k", first_event=None)
    assert canonical_digest(json.loads(_strict(res))) == FROZEN_DEFAULT_DIGESTS["tools"]


def test_segments_single_is_one_segment_named_after_the_scores_role(tmp_path):
    d = make_domain(tmp_path, "sports", E=30, N=15, seed=51, n_valid=24, head_bias=1.5)
    kw = dict(ref_ranks=d["ref_ranks"], ref_exposure=d["ref_exposure"], n_boot=16)
    args = (d["audit"], d["panel_test"], d["panel_valid"])
    auto = na.run_domain("sports", *args, **kw)                                       # the registered segmentation of sports
    assert list(auto["segments"]) == ["events_1_30"] and auto["segments"]["events_1_30"]["role"] == "quarantine"
    assert "segment_mode" not in auto
    out = tmp_path / "single.json"
    one = na.run_domain("sports", *args, segments="single", first_event=1001, out=out, **kw)
    assert list(one["segments"]) == ["test"]                                          # named after the scores role
    seg = one["segments"]["test"]
    assert (seg["role"], seg["event_range"], seg["n_events"]) == ("all", [1001, 1030], 30)
    sm = one["segment_mode"]
    assert (sm["mode"], sm["segment"], sm["first_event"], sm["n_events"]) == ("single", "test", 1001, 30)
    assert "full test panel" in sm["event_range_basis"] and "ONE segment" in sm["definition"]
    assert {k: one["inputs"][k] for k in ("segments", "test_role", "valid_role", "first_event")} == {
        "segments": "single", "test_role": "test", "valid_role": "valid2k", "first_event": 1001}
    plain = na.run_domain("sports", *args, segments="single", **kw)                   # first_event unknown: panel positions
    assert plain["segments"]["test"]["event_range"] == [1, 30] and plain["segment_mode"]["first_event"] is None
    assert "first_event" not in plain["inputs"] and "positions" in plain["segment_mode"]["event_range_basis"]
    # `single` is the standard one-segment analysis of the panel: another domain name with the same files, default options
    shutil.copytree(d["audit"] / "sports_test", d["audit"] / "toys_test")
    shutil.copytree(d["audit"] / "sports_valid2k", d["audit"] / "toys_valid2k")
    toys = na.run_domain("toys", *args, **kw)
    assert list(toys["segments"]) == ["all"] and _strict(toys["segments"]["all"]) == _strict(plain["segments"]["test"])
    assert _strict(toys["temperature"]) == _strict(plain["temperature"])
    # option guards (the first three fail before any file is read, the last one once the panel size is known)
    for bad, msg in ((dict(segments="single", quarantine_n=5), "quarantine_n"), (dict(first_event=1001), "first_event"),
                     (dict(segments="pooled"), "segments must be"), (dict(segments="single", first_event=0), "1-based")):
        with pytest.raises(ValueError, match=msg):
            na.run_domain("sports", *args, **kw, **bad)
    with pytest.raises(ValueError, match="straddles"):       # events 985-1014 would pool quarantined and other events
        na.run_domain("sports", *args, segments="single", first_event=985, **kw)
    inside = na.run_domain("sports", *args, segments="single", first_event=971, **kw)             # events 971-1000: all quarantined
    assert inside["segments"]["test"]["event_range"] == [971, 1000]
    # summarize treats the single segment as the domain's entry (a one-domain family is incomplete: no admission)
    s = na.summarize([_jload(out)])
    assert s["units"]["sports"] == {"domain": "sports", "role": "all", "event_range": [1001, 1030], "n_events": 30}
    assert s["family_units"] == ["sports"] and not s["complete_family"]
    assert s["S4"]["pooled_auroc"]["next"]["values"]["sports"]["n"] == 30 * 15
    assert not s["S3"]["admission"]["llm_next"]["effect_claimed"]


def test_roles_select_the_scores_directories_of_a_second_backbone(tmp_path):
    d = make_domain(tmp_path, "home", E=24, N=15, seed=61, n_valid=20, head_bias=1.5)
    llama = tmp_path / "llama_audit"                  # <audit>/<d>_test1001_3000 and <d>_valid500, as scored for Z2
    shutil.copytree(d["audit"] / "home_test", llama / "home_test1001_3000")
    na.restrict_scores(d["audit"] / "home_valid2k" / "scores.csv.gz", llama / "home_valid500" / "scores.csv.gz",
                       na.read_event_ids(d["panel_valid"])[:10])                       # a 10-event VALID sample (valid500)
    kw = dict(ref_ranks=d["ref_ranks"], ref_exposure=d["ref_exposure"], n_boot=12)
    res = na.run_domain("home", llama, d["panel_test"], d["panel_valid"], segments="single", test_role="test1001_3000",
                        valid_role="valid500", first_event=1001, **kw)
    assert list(res["segments"]) == ["test1001_3000"]
    seg = res["segments"]["test1001_3000"]
    assert (seg["role"], seg["event_range"], seg["n_events"]) == ("all", [1001, 1024], 24)
    assert res["segment_mode"]["segment"] == "test1001_3000"
    assert (res["inputs"]["test_role"], res["inputs"]["valid_role"]) == ("test1001_3000", "valid500")
    # the temperature is fitted on the VALID role's own sample (10 events), not on the 20 events of valid2k
    assert res["temperature"]["next"]["n_events_used"] == 10 and res["data"]["valid_events"] == 10
    assert res["data"]["scores_valid"]["by_question"]["next"]["n"] == 10 * 15
    full = na.run_domain("home", d["audit"], d["panel_test"], d["panel_valid"], **kw)            # registered layout
    assert full["temperature"]["next"]["n_events_used"] == 20
    assert res["temperature"]["next"]["T"] != full["temperature"]["next"]["T"]
    # same TEST events and seeds: the ranking metrics (T-free) are identical, the calibration block is not
    ranking = lambda r, name: _strict(r["segments"][name]["questions"]["next"]["A_ranking"]["ranking"])     # noqa: E731
    assert ranking(res, "test1001_3000") == ranking(full, "all")
    cal = lambda r, name: r["segments"][name]["questions"]["next"]["C_calibration"]["list_normalised"]["ece"]["est"]   # noqa: E731
    assert cal(res, "test1001_3000") != cal(full, "all")
    with pytest.raises(FileNotFoundError, match="valid999"):
        na.run_domain("home", llama, d["panel_test"], d["panel_valid"], segments="single", test_role="test1001_3000",
                      valid_role="valid999", **kw)
    # the command line does the same
    out = tmp_path / "llama_home.json"
    na.main(["run", "--domain", "home", "--audit_dir", str(llama), "--panel_test", str(d["panel_test"]),
             "--panel_valid", str(d["panel_valid"]), "--ref_ranks", str(d["ref_ranks"]), "--ref_exposure", str(d["ref_exposure"]),
             "--out", str(out), "--n_boot", "12", "--segments", "single", "--test_role", "test1001_3000",
             "--valid_role", "valid500", "--first_event", "1001"])
    assert _strict(_without(_jload(out), "timing_s")) == _strict(json.loads(_strict(_without(res, "timing_s"))))


def test_restrict_writes_the_subset_scores_and_an_unchanged_valid_copy(tmp_path):
    d = make_domain(tmp_path, "toys", E=60, N=15, seed=41, n_valid=30, head_bias=1.5)
    rows, lo, hi = d["rows"], 10, 30
    subset = tmp_path / "panels" / "toys_test_11_30.jsonl"
    write_jsonl(subset, rows[lo:hi])
    out_dir = tmp_path / "restricted"
    man = na.restrict_audit("toys", d["audit"], subset, out_dir)
    src_test, src_valid = d["audit"] / "toys_test" / "scores.csv.gz", d["audit"] / "toys_valid2k" / "scores.csv.gz"
    got_test = out_dir / "toys_test" / "scores.csv.gz"
    src_lines = gzip.open(src_test, "rb").read().splitlines(keepends=True)
    got_lines = gzip.open(got_test, "rb").read().splitlines(keepends=True)
    ids = {r["source_event_id"] for r in rows[lo:hi]}
    want = [src_lines[0]] + [ln for ln in src_lines[1:] if ln.split(b",", 1)[0].decode() in ids]
    assert got_lines == want                                           # header + verbatim lines of those events, source order
    assert len(got_lines) == 1 + (hi - lo) * 15 * 2                    # 20 events x 15 candidates x (next, like)
    assert (out_dir / "toys_valid2k" / "scores.csv.gz").read_bytes() == src_valid.read_bytes()    # VALID copied unchanged
    rt = man["restricted_test"]
    assert (rt["n_rows_in"], rt["n_rows_kept"], rt["n_events_found"], rt["n_events_missing"]) == (60 * 15 * 2, 20 * 15 * 2, 20, 0)
    assert man["n_events_requested"] == 20 and man["ids_sha1"] == hashlib.sha1(
        "\n".join(r["source_event_id"] for r in rows[lo:hi]).encode("utf-8")).hexdigest()
    assert rt["sha1"] == hashlib.sha1(got_test.read_bytes()).hexdigest()
    assert man["valid_copy"]["sha1"] == hashlib.sha1(src_valid.read_bytes()).hexdigest()
    assert _jload(out_dir / "toys_restrict.json")["restricted_test"]["sha1"] == rt["sha1"]
    # level-9 gzip with a zero mtime: the same restriction twice is the same file
    na.restrict_audit("toys", d["audit"], subset, tmp_path / "restricted_again")
    assert (tmp_path / "restricted_again" / "toys_test" / "scores.csv.gz").read_bytes() == got_test.read_bytes()


def test_restricted_qwen_analysis_equals_the_same_events_analysed_directly(tmp_path):
    # a panel of six blocks of 10 events: blocks 2-3 (events 11-30) play the role of events 1,001-3,000 of the real panel
    d = make_domain(tmp_path, "toys", E=60, N=15, seed=41, n_valid=30, head_bias=1.5)
    rows, lo, hi = d["rows"], 10, 30
    subset = tmp_path / "panels" / "toys_test_11_30.jsonl"
    write_jsonl(subset, rows[lo:hi])
    restricted = tmp_path / "qwen_restricted"
    na.restrict_audit("toys", d["audit"], subset, restricted)
    direct = tmp_path / "direct"                                       # the same events written straight from the arrays
    write_scores(direct / "toys_test" / "scores.csv.gz", rows[lo:hi], {q: d["L"][q][lo:hi] for q in ("next", "like")})
    (direct / "toys_valid2k").mkdir(parents=True)
    shutil.copyfile(d["audit"] / "toys_valid2k" / "scores.csv.gz", direct / "toys_valid2k" / "scores.csv.gz")
    kw = dict(ref_ranks=d["ref_ranks"], ref_exposure=d["ref_exposure"], n_boot=16, segments="single", first_event=lo + 1)
    r_restricted = na.run_domain("toys", restricted, subset, d["panel_valid"], **kw)
    r_direct = na.run_domain("toys", direct, subset, d["panel_valid"], **kw)
    r_unrestricted = na.run_domain("toys", d["audit"], subset, d["panel_valid"], **kw)       # all 60 events' scores, 20-event panel
    gone = ("timing_s", "inputs")
    assert _strict(_without(r_restricted, *gone)) == _strict(_without(r_direct, *gone))
    seg = r_restricted["segments"]["test"]
    assert seg["event_range"] == [11, 30] and seg["n_events"] == 20 and r_restricted["data"]["panel"]["n_events"] == 20
    # restricting is a convenience: the rows of events outside the panel are ignored (and counted) by `run` itself
    sd = r_unrestricted["data"]["scores_test"]
    assert sd["score_rows"] == 60 * 15 * 2 and sd["score_rows_without_panel_event"] == 40 * 15 * 2
    assert _strict(sd["by_question"]) == _strict(r_restricted["data"]["scores_test"]["by_question"])
    same_but_row_counts = lambda r: _strict(_without(dict(r, data=_without(r["data"], "scores_test")), *gone))   # noqa: E731
    assert same_but_row_counts(r_unrestricted) == same_but_row_counts(r_restricted)
    # the numbers are those of the 20 events: a per-event recomputation on the full arrays
    L, pos = d["L"]["next"], d["pos"]
    rank = np.array([stats.tie_aware_rank(L[e], int(pos[e])) for e in range(lo, hi)])
    assert seg["questions"]["next"]["A_ranking"]["ranking"]["ndcg10"]["est"] == pytest.approx(
        metrics.ndcg_from_rank(rank, 10).mean())
    top = [np.argsort(-L[e], kind="stable")[:10] for e in range(lo, hi)]
    assert seg["questions"]["next"]["B_exposure"]["llm"]["head_share_top10"]["est"] == pytest.approx(
        np.mean([(d["grp"][e][t] == 0).mean() for e, t in zip(range(lo, hi), top)]))
    # Qwen keeps its own temperature: the one fitted on the (unchanged) VALID2k sample of the full audit
    full = na.run_domain("toys", d["audit"], d["panel_test"], d["panel_valid"], ref_ranks=d["ref_ranks"],
                         ref_exposure=d["ref_exposure"], n_boot=8)
    for q in ("next", "like"):
        assert r_restricted["temperature"][q]["T"] == full["temperature"][q]["T"]
        assert r_restricted["temperature"][q]["n_events_used"] == 30


def test_restrict_error_paths_roles_and_id_list_input(tmp_path):
    d = make_domain(tmp_path, "tools", E=24, N=15, seed=71, n_valid=16, head_bias=1.0)
    sub = d["rows"][4:12]
    subset = tmp_path / "subset.jsonl"
    write_jsonl(subset, sub)
    idlist = tmp_path / "ids.txt"
    idlist.write_text("\n".join(r["source_event_id"] for r in sub) + "\n", encoding="utf-8")
    a = na.restrict_audit("tools", d["audit"], subset, tmp_path / "out_a")
    b = na.restrict_audit("tools", d["audit"], idlist, tmp_path / "out_b")                    # a plain id list works too
    f_a, f_b = (tmp_path / o / "tools_test" / "scores.csv.gz" for o in ("out_a", "out_b"))
    assert f_a.read_bytes() == f_b.read_bytes() and a["ids_sha1"] == b["ids_sha1"]
    odd = tmp_path / "odd.jsonl"                                                              # id not the first JSON key
    odd.write_text(json.dumps({"user_id": "u", "source_event_id": "a::1"}) + "\n" +
                   json.dumps({"source_event_id": "b::2", "x": 1}) + "\n\n", encoding="utf-8")
    assert na.read_event_ids(odd) == ["a::1", "b::2"]
    # an id without any score row is an error and nothing is left behind; allow_missing counts it instead
    ghost = tmp_path / "ghost.txt"
    ghost.write_text("\n".join([r["source_event_id"] for r in sub] + ["nobody::1"]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no score row"):
        na.restrict_audit("tools", d["audit"], ghost, tmp_path / "out_c")
    assert not (tmp_path / "out_c" / "tools_test" / "scores.csv.gz").exists() and not list(tmp_path.rglob("*.tmp"))
    man = na.restrict_audit("tools", d["audit"], ghost, tmp_path / "out_d", allow_missing=True)
    rt = man["restricted_test"]
    assert (rt["n_events_found"], rt["n_events_missing"], rt["missing_examples"]) == (8, 1, ["nobody::1"])
    assert (tmp_path / "out_d" / "tools_test" / "scores.csv.gz").read_bytes() == f_a.read_bytes()
    # refusals: a repeated id, out_dir == audit_dir, a missing source scores file
    dup = tmp_path / "dup.txt"
    dup.write_text(f"{sub[0]['source_event_id']}\n{sub[0]['source_event_id']}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated"):
        na.restrict_audit("tools", d["audit"], dup, tmp_path / "out_e")
    with pytest.raises(ValueError, match="out_dir must differ"):
        na.restrict_audit("tools", d["audit"], subset, d["audit"])
    with pytest.raises(FileNotFoundError):
        na.restrict_audit("tools", d["audit"], subset, tmp_path / "out_e", test_role="nope")
    # roles name the directories on both sides: <d>_<test_role> and <d>_<valid_role>
    llama = tmp_path / "llama"
    shutil.copytree(d["audit"] / "tools_test", llama / "tools_test1001_3000")
    shutil.copytree(d["audit"] / "tools_valid2k", llama / "tools_valid500")
    na.restrict_audit("tools", llama, subset, tmp_path / "out_f", test_role="test1001_3000", valid_role="valid500")
    assert (tmp_path / "out_f" / "tools_test1001_3000" / "scores.csv.gz").read_bytes() == f_a.read_bytes()
    assert (tmp_path / "out_f" / "tools_valid500" / "scores.csv.gz").read_bytes() == (
        llama / "tools_valid500" / "scores.csv.gz").read_bytes()
    # the command line: success, and a one-line failure
    cli = ["restrict", "--domain", "tools", "--audit_dir", str(d["audit"]), "--panel_subset"]
    na.main(cli + [str(subset), "--out_dir", str(tmp_path / "out_g")])
    assert (tmp_path / "out_g" / "tools_test" / "scores.csv.gz").read_bytes() == f_a.read_bytes()
    assert _jload(tmp_path / "out_g" / "tools_restrict.json")["n_events_requested"] == 8
    with pytest.raises(SystemExit, match="restrict:"):
        na.main(cli + [str(ghost), "--out_dir", str(tmp_path / "out_h")])
    na.main(cli + [str(ghost), "--out_dir", str(tmp_path / "out_h"), "--allow_missing"])
    assert _jload(tmp_path / "out_h" / "tools_restrict.json")["restricted_test"]["n_events_missing"] == 1


def _naive_gini_coverage(top_item_lists, pool_size):
    cnt = {}
    for lst in top_item_lists:
        for it in set(lst):
            cnt[it] = cnt.get(it, 0) + 1
    x = np.sort(np.array(list(cnt.values()) + [0] * (pool_size - len(cnt)), float))
    n = len(x)
    return len(cnt) / pool_size, 2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n


def test_pipeline_matches_naive_per_event_oracle(tmp_path):
    """Every point estimate of the run output equals a plain per-event Python re-computation on the same arrays."""
    d = make_domain(tmp_path, "toys", E=40, N=15, seed=31, n_valid=30, head_bias=1.5)
    res = na.run_domain("toys", d["audit"], d["panel_test"], d["panel_valid"], d["ref_ranks"], d["ref_exposure"], n_boot=12)
    seg = res["segments"]["all"]
    rows, grp, pos = d["rows"], d["grp"], d["pos"]
    E, N = grp.shape
    pool = {it for r in rows for it in r["candidate_item_ids"]}
    panel = na.load_panel(d["panel_test"])
    profile, _ = na.user_profile(panel)
    assert res["n_pool_items"] == len(pool)

    def est(blk):
        return blk["est"]

    for q in ("next", "like"):
        L = d["L"][q]
        qb = seg["questions"][q]
        beta = res["temperature"][q]["beta"]
        order = [np.argsort(-L[e], kind="stable") for e in range(E)]
        rank = np.array([stats.tie_aware_rank(L[e], int(pos[e])) for e in range(E)])
        ndcg = metrics.ndcg_from_rank(rank, 10)
        A = qb["A_ranking"]["ranking"]
        assert est(A["ndcg10"]) == pytest.approx(ndcg.mean()) and est(A["hr10"]) == pytest.approx((rank <= 10).mean())
        assert est(A["mrr"]) == pytest.approx((1 / rank).mean())
        # B: static exposure of the deterministic top-10
        top = [o[:10] for o in order]
        share = np.array([[(grp[e][t] == g).mean() for g in range(3)] for e, t in enumerate(top)])
        poolsh = np.array([[(grp[e] == g).mean() for g in range(3)] for e in range(E)])
        B = qb["B_exposure"]["llm"]
        for g, name in enumerate(na.GROUPS):
            assert est(B[f"{name}_share_top10"]) == pytest.approx(share[:, g].mean())
            assert est(B[f"pool_{name}_share"]) == pytest.approx(poolsh[:, g].mean())
            assert est(B[f"delta_{name}"]) == pytest.approx((share[:, g] - poolsh[:, g]).mean())
        tgt = (grp[np.arange(E), pos] == 0).mean()
        assert est(B["target_head_share"]) == pytest.approx(tgt)
        assert est(B["delta_head_vs_target"]) == pytest.approx(share[:, 0].mean() - tgt)
        cov, gini = _naive_gini_coverage([[rows[e]["candidate_item_ids"][i] for i in t] for e, t in enumerate(top)], len(pool))
        assert est(B["coverage_top10"]) == pytest.approx(cov) and est(B["gini_exposure"]) == pytest.approx(gini)
        # C: list-normalised probabilities, calibration, anatomy, pointwise
        P = np.exp(beta * (L - L.max(1, keepdims=True)))
        P /= P.sum(1, keepdims=True)
        pmax = P.max(1)
        top1 = np.array([float(order[e][0] == pos[e]) for e in range(E)])
        C = qb["C_calibration"]
        cn = C["list_normalised"]
        assert est(cn["top1_accuracy"]) == pytest.approx(top1.mean()) and est(cn["p_max_mean"]) == pytest.approx(pmax.mean())
        y1 = np.eye(N)[pos]
        assert est(cn["brier"]) == pytest.approx(((P - y1) ** 2).sum(1).mean())
        assert est(cn["nll"]) == pytest.approx(-np.log(P[np.arange(E), pos]).mean())
        assert est(cn["ece"]) == pytest.approx(metrics.ece(pmax, top1, 15))
        assert est(cn["ece_adaptive"]) == pytest.approx(metrics.ece(pmax, top1, 15, adaptive=True))
        ad = C["auroc_discrimination"]["top1"]["auroc"]
        sig = {"p_max": pmax, "margin": np.array([np.sort(P[e])[-1] - np.sort(P[e])[-2] for e in range(E)]),
               "neg_entropy": (P * np.log(P)).sum(1), "max_logit": L.max(1)}
        for k, v in sig.items():
            assert est(ad[k]) == pytest.approx(metrics.auroc(v, top1.astype(int)))
        tert = stats.rank_bins(pmax, 3)
        an = C["error_anatomy"]
        assert est(an["confident_error_rate"]) == pytest.approx(1 - top1[tert == 2].mean())
        assert est(an["unsure_correct_rate"]) == pytest.approx(top1[tert == 0].mean())
        assert est(C["pointwise"]["uauc"]) == pytest.approx(np.mean(1 - (rank - 1) / (N - 1)))
        yy = np.zeros(L.shape, int)
        yy[np.arange(E), pos] = 1
        assert est(C["pointwise"]["pooled_auroc"]) == pytest.approx(metrics.auroc(L.reshape(-1), yy.reshape(-1)))
        # D: selective serving on the p_max signal (NDCG@10 utility) and the niche served share at 50%
        Dp = qb["D_selective_serving"]["signals"]["p_max"]
        _, cum, aurc = metrics.risk_coverage(pmax, ndcg)
        assert est(Dp["aurc"]["ndcg10"]) == pytest.approx(aurc)
        assert est(Dp["d_aurc_vs_random_expectation"]["ndcg10"]) == pytest.approx(aurc - (1 - ndcg.mean()))
        assert est(Dp["gain_at_50_vs_full"]["ndcg10"]) == pytest.approx(cum[int(round(0.5 * E)) - 1] - ndcg.mean())
        has = np.isfinite(profile)
        quint = np.full(E, -1)
        quint[has] = stats.rank_bins(profile[has], 5)
        o = np.argsort(-pmax, kind="stable")
        served = np.zeros(E)
        k50 = int(round(0.5 * E))
        i = 0
        while i < E:
            j = i
            while j + 1 < E and pmax[o[j + 1]] == pmax[o[i]]:
                j += 1
            served[o[i:j + 1]] = np.clip((k50 - i) / (j - i + 1), 0, 1)
            i = j + 1
        for j in range(5):
            if (quint == j).any():
                assert est(Dp["niche"]["served_share"][j]) == pytest.approx(served[quint == j].sum() / (quint == j).sum())
        # E: popularity-graded confidence on the top-10 rows
        Ep = qb["E_popularity"]
        G10 = np.array([grp[e][t] for e, t in enumerate(top)])
        p10 = np.array([P[e][t] for e, t in enumerate(top)])
        y10 = np.array([(t == pos[e]).astype(float) for e, t in enumerate(top)])
        for g, name in enumerate(na.GROUPS):
            assert est(Ep["mean_p"][name]) == pytest.approx(p10[G10 == g].mean())
            assert est(Ep["residual_y_minus_p"][name]) == pytest.approx((y10 - p10)[G10 == g].mean())
        bi = metrics.bias_index(p10.reshape(-1), y10.reshape(-1), np.asarray(na.GROUPS)[G10.reshape(-1)], adjust=True)
        for name in na.GROUPS:
            got = Ep["bias_index"][name]["est"]
            if name not in bi or math.isnan(bi[name]):
                assert got is None or math.isnan(got)
            else:
                assert got == pytest.approx(bi[name])
        assert est(Ep["top1_head_fraction"]) == pytest.approx(np.mean([grp[e][order[e][0]] == 0 for e in range(E)]))
    # reference methods: rank files and exported exposure lists
    with gzip.open(d["ref_exposure"] / "ccrp_v3.csv.gz", "rt", newline="", encoding="utf-8") as f:
        ex = {r["source_event_id"]: r for r in csv.DictReader(f)}
    with gzip.open(d["ref_ranks"] / "ccrp_v3.csv.gz", "rt", newline="", encoding="utf-8") as f:
        rk = {r["source_event_id"]: float(r["positive_rank"]) for r in csv.DictReader(f)}
    ids = [r["source_event_id"] for r in rows]
    ref_rank = np.array([rk[i] for i in ids])
    ref_head = np.array([[g == "head" for g in ex[i]["top10_groups"].split(" ")] for i in ids]).mean(1)
    R = seg["reference"]["ccrp_v3"]
    assert R["A_ranking"]["ranking"]["ndcg10"]["est"] == pytest.approx(metrics.ndcg_from_rank(ref_rank, 10).mean())
    assert R["B_exposure"]["head_share_top10"]["est"] == pytest.approx(ref_head.mean())
    cov, gini = _naive_gini_coverage([ex[i]["top10_item_ids"].split(" ") for i in ids], len(pool))
    assert R["B_exposure"]["coverage_top10"]["est"] == pytest.approx(cov)
    assert R["B_exposure"]["gini_exposure"]["est"] == pytest.approx(gini)
    qn = seg["questions"]["next"]
    L = d["L"]["next"]
    rank_llm = np.array([stats.tie_aware_rank(L[e], int(pos[e])) for e in range(E)])
    assert qn["A_ranking"]["vs_reference"]["ccrp_v3"]["dndcg10"]["est"] == pytest.approx(
        (metrics.ndcg_from_rank(rank_llm, 10) - metrics.ndcg_from_rank(ref_rank, 10)).mean())
    llm_head = np.array([(grp[e][np.argsort(-L[e], kind="stable")[:10]] == 0).mean() for e in range(E)])
    assert qn["B_exposure"]["vs_reference"]["ccrp_v3"]["head_share_top10_llm_minus_ref"]["est"] == pytest.approx(
        (llm_head - ref_head).mean())


def test_fully_unscored_question_and_missing_references_degrade_gracefully(tmp_path):
    E, N = 14, 12
    cens = {("like", e, c): 2 for e in range(E) for c in range(N)}                   # every `like` candidate is censored
    d = make_domain(tmp_path, "home", E=E, N=N, seed=5, n_valid=12, censored_test=cens)
    out = tmp_path / "home.json"
    res = na.run_domain("home", d["audit"], d["panel_test"], d["panel_valid"], d["ref_ranks"], d["ref_exposure"], n_boot=10,
                        out=out)
    like = res["segments"]["all"]["questions"]["like"]
    assert like["n_events_valid"] == 0 and like["A_ranking"]["censoring"]["n_events_no_scored_candidate"] == E
    assert like["A_ranking"]["ranking"]["ndcg10"]["est"] == 0.0                       # a miss in A, not a dropped event
    assert like["D_selective_serving"]["n_events"] == 0 and like["E_popularity"]["n_rows"] == 0
    assert res["segments"]["all"]["questions"]["next"]["n_events_valid"] == E         # the other question is unaffected
    assert "NaN" not in out.read_text(encoding="utf-8")
    assert na.summarize([json.loads(out.read_text(encoding="utf-8"))])["schema"] == na.SUMMARY_SCHEMA
    nxt = na.run_domain("home", d["audit"], d["panel_test"], d["panel_valid"], None, None, questions=("next",), n_boot=10)
    assert list(nxt["segments"]["all"]["questions"]) == ["next"] and nxt["segments"]["all"]["reference"] == {}
    one = na.run_domain("home", d["audit"], d["panel_test"], d["panel_valid"], None, None, questions=("next",), n_boot=10,
                        quarantine_n=1)
    assert list(one["segments"]) == ["events_1_1", "events_2_14"]


# ========================================================================================== export script
def _export_fixture(tmp_path, E=6, N=12):
    L, grp, pos, _ = synth_arrays(E, N, 5)
    rows = make_panel_rows(E, N, 5, pos, grp, domain="x")
    exp = _load_export()
    ref = {r["source_event_id"]: [r["candidate_item_ids"][r["positive_item_index"]],
                                  exp.candidate_fp(r["candidate_item_ids"])] for r in rows}
    (tmp_path / "docs/sigir/panel_refs").mkdir(parents=True)
    (tmp_path / "docs/sigir/panel_refs/x_test.json").write_text(json.dumps(ref), encoding="utf-8")
    rng = np.random.default_rng(1)
    paths = {}
    for meth in ("ccrp_v3", "llmemb"):
        base = ("outputs/x_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate" if meth == "ccrp_v3" else
                f"outputs/baselines/official_adapters/x_large10000_100neg_{meth}_official_qwen3base_same_candidate")
        sc = rng.normal(size=(E, N))
        ref_records(rows, sc, tmp_path / base / "tables" / "ranking_eval_records.csv")
        paths[meth] = (tmp_path / base / "tables" / "ranking_eval_records.csv", sc)
    return exp, rows, paths


def test_export_ref_exposure_columns_values_pool_and_reproducible_bytes(tmp_path):
    exp, rows, paths = _export_fixture(tmp_path)
    out = tmp_path / "docs/sigir/ref_exposure"
    info = exp.export_domain("x", tmp_path, out, verbose=False)
    assert info["n_pool_items"] == len({i for r in rows for i in r["candidate_item_ids"]})
    assert info["n_events"] == 6 and set(info["methods"]) == {"ccrp_v3", "llmemb"}
    assert json.loads((out / "x" / "_pool.json").read_text())["n_pool_items"] == info["n_pool_items"]
    with gzip.open(out / "x" / "ccrp_v3.csv.gz", "rt", newline="", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert list(got[0]) == ["source_event_id", "user_id", "positive_rank", "positive_group", "top10_item_ids", "top10_groups",
                            "n_head_pool", "n_mid_pool", "n_tail_pool"]
    sc = paths["ccrp_v3"][1]
    for e, (g, r) in enumerate(zip(got, rows)):                                       # panel order, first 10 predictions
        order = np.argsort(-sc[e], kind="stable")
        assert g["source_event_id"] == r["source_event_id"] and g["user_id"] == r["user_id"]
        assert g["top10_item_ids"].split(" ") == [r["candidate_item_ids"][i] for i in order[:10]]
        assert g["top10_groups"].split(" ") == [r["candidate_popularity_groups"][i] for i in order[:10]]
        assert g["positive_group"] == r["candidate_popularity_groups"][r["positive_item_index"]]
        assert int(g["positive_rank"]) == int(np.flatnonzero(order == r["positive_item_index"])[0]) + 1
        cg = r["candidate_popularity_groups"]
        assert [int(g["n_head_pool"]), int(g["n_mid_pool"]), int(g["n_tail_pool"])] == [cg.count(x) for x in na.GROUPS]
    before = (out / "x" / "ccrp_v3.csv.gz").read_bytes()
    exp.export_domain("x", tmp_path, out, verbose=False)
    assert (out / "x" / "ccrp_v3.csv.gz").read_bytes() == before                      # level-9 gzip with zero mtime


def test_export_ref_exposure_assertions_missing_duplicate_candidates_groups(tmp_path):
    exp, rows, paths = _export_fixture(tmp_path)
    out = tmp_path / "docs/sigir/ref_exposure"
    f_llm = paths["llmemb"][0]
    text = f_llm.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    ok = lines[:]
    # 1. a missing event
    f_llm.write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(exp.ExportError, match="missing"):
        exp.export_domain("x", tmp_path, out, verbose=False)
    # 2. a duplicated event
    f_llm.write_text("".join(lines + [lines[2]]), encoding="utf-8")
    with pytest.raises(exp.ExportError, match="duplicated"):
        exp.export_domain("x", tmp_path, out, verbose=False)
    # 3. a reordered candidate list (the frozen fingerprint is of the ORDERED list)
    rd = list(csv.DictReader(text.splitlines()))
    cands = ast.literal_eval(rd[1]["candidate_item_ids"])
    rd[1]["candidate_item_ids"] = repr(cands[1:] + cands[:1])
    with open(f_llm, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rd[0]))
        w.writeheader()
        w.writerows(rd)
    with pytest.raises(exp.ExportError, match="fingerprint"):
        exp.export_domain("x", tmp_path, out, verbose=False)
    # 4. candidate popularity groups that differ between methods (ids identical)
    rd = list(csv.DictReader(text.splitlines()))
    gl = ast.literal_eval(rd[2]["candidate_popularity_groups"])
    gl[0] = "tail" if gl[0] != "tail" else "head"
    rd[2]["candidate_popularity_groups"] = repr(gl)
    with open(f_llm, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rd[0]))
        w.writeheader()
        w.writerows(rd)
    with pytest.raises(exp.ExportError, match="popularity groups differ"):
        exp.export_domain("x", tmp_path, out, verbose=False)
    # 5. restoring the file makes the export pass again
    f_llm.write_text("".join(ok), encoding="utf-8")
    assert exp.export_domain("x", tmp_path, out, verbose=False)["n_events"] == 6
