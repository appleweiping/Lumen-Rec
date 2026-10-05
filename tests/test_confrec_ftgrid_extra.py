"""src/confrec/ftgrid_extra.py: Amendment 3 addendum 6 items 2-8 and its Holm families.
Synthetic panels, raw ratings and scorer-format runs (the worlds of tests/test_confrec_ftgrid_report.py) and planted
context-level worlds; every estimator is checked against the registered report, a brute-force computation or a planted
truth. CPU, deterministic, no torch / transformers."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.confrec import forensics as fx
from src.confrec import ftgrid_extra as fe
from src.confrec import ftgrid_report as fr
from src.confrec.metrics import auroc
from src.confrec.stats import paired_bootstrap, user_halves
from tests.test_confrec_ftgrid_report import ML1M_SPECS, make_world, run_main, write_models

N_BOOT = 60


def _dump(x) -> str:
    return json.dumps(x, sort_keys=True)


def _strict_load(path):
    def bad(x):
        raise ValueError(f"non-strict JSON constant {x}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad)


def run_extra(w, out_name, models, n_boot=N_BOOT, refs=True, raw=True):
    out = w.root / "extra" / f"{out_name}.json"
    args = ["build", "--domain", w.domain, "--split", str(w.split), "--panels", str(w.d), "--scores_root",
            str(w.root / "scores"), "--models", ",".join(models), "--out", str(out), "--n_boot", str(n_boot)]
    if raw and w.raw is not None:
        args += ["--raw", str(w.raw)]
    if refs:
        args += ["--refs", str(w.root / "report" / "ml1m_refs.json")]
    return fe.main(args), out


@pytest.fixture(scope="module")
def ml1m(tmp_path_factory):
    """The report's synthetic ML-1M world (every arm, FT-C adapters included): the report and this module on it."""
    w = make_world(tmp_path_factory.mktemp("ml1m_extra"), "ml1m", seed=0)
    write_models(w, ML1M_SPECS)
    rep, _ = run_main(w, "ml1m", list(ML1M_SPECS), n_boot=N_BOOT,
                      extra=("--refs", str(w.root / "report" / "ml1m_refs.json")))
    res, out = run_extra(w, "ml1m", list(ML1M_SPECS))
    return SimpleNamespace(w=w, rep=rep, res=res, out=out)


# ---------------------------------------------------------------- planted context-level worlds
def ctx_world(n_users=600, n_rows=12, n_items=500, kappa=1.0, gamma=0.0, mf_gamma=0.0, n_models=1, seed=0,
              sd_eps=1.0, sd_dn=0.2, lenient=1.5, n_boot=100):
    """S_d TEST rows only. Label = 1[b_u + mu_i + e_ui + noise > 0] with the user's leniency b_u, the item quality mu_i
    and a personal term e_ui. The LLM: L = kappa b_u (a user offset that tracks leniency) + p_i (its item prior, a noisy
    mu_i) + gamma e_ui + noise; pi = p_i + the mean of 8 donor errors (halves 1-4, 5-8); q-hat = mu_i + noise; the MF
    personal residual = mf_gamma e_ui + noise."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, n_items)
    p = mu + rng.normal(0, 0.5, n_items)
    dn = rng.normal(0, sd_dn, (n_items, 8))
    us, its, bs, es, ys = [], [], [], [], []
    k = 0
    while len(set(us)) < n_users:
        b = rng.normal(0, lenient)
        it = rng.choice(n_items, n_rows, replace=False)
        e = rng.normal(0, 1, n_rows)
        y = (b + mu[it] + e + rng.normal(0, 1, n_rows) > 0).astype(int)
        if 0 < y.sum() < n_rows:
            us += [f"u{k:05d}"] * n_rows
            its += it.tolist()
            bs += [b] * n_rows
            es += e.tolist()
            ys += y.tolist()
        k += 1
    users, items = np.array(us), np.array(its)
    b, e, y = np.array(bs), np.array(es), np.array(ys)
    n = len(users)
    q = mu[items] + rng.normal(0, 0.3, n)
    mfres = mf_gamma * e + rng.normal(0, 1, n)
    names = ["zeroshot"] if n_models == 1 else [f"s{j}" for j in range(n_models)]
    L, PI, PA, PB = {}, {}, {}, {}
    for m in names:
        L[m] = kappa * b + p[items] + gamma * e + rng.normal(0, sd_eps, n)
        PI[m] = p[items] + dn.mean(1)[items]
        PA[m], PB[m] = p[items] + dn[:, :4].mean(1)[items], p[items] + dn[:, 4:].mean(1)[items]
    sd_ids = sorted(set(users.tolist()))
    uc = np.unique(users, return_inverse=True)[1].reshape(-1)
    fa = user_halves(sd_ids, 0)
    cx = SimpleNamespace(users=users, uc=uc, y=y, sd_ids=sd_ids, sd_test=np.ones(n, bool), test=np.ones(n, bool),
                         fold_a=np.array([u in fa for u in users.tolist()], bool), n=n, item=items)
    return SimpleNamespace(cx=cx, L=L, PI=PI, PA=PA, PB=PB, models=names, sd_ids=sd_ids, present=set(names),
                           refs={"arrays": {"q_hat": q, "mf_residual": mfres}}, n_boot=n_boot, seed=0, items=items,
                           b=b, e=e, mu=mu, p=p)


def info_of(X, reg="ZS"):
    return {"models": X.models, "missing_or_excluded": [], "n_registered": len(X.models) if reg == "ZS" else 3}


def bf_logit(Xf, y, l2=1e-4):
    """Independent logistic regression with the registered objective: intercept, features standardised on the fitting
    rows, L2 l2 on the slopes only; Newton to convergence."""
    m, s = Xf.mean(0), Xf.std(0)
    s = np.where(s > 0, s, 1.0)
    Z = np.column_stack([np.ones(len(Xf)), (Xf - m) / s])
    w = np.zeros(Z.shape[1])
    P = np.diag([0.0] + [l2] * (Z.shape[1] - 1))
    for _ in range(200):
        pr = 1.0 / (1.0 + np.exp(-(Z @ w)))
        step = np.linalg.solve(Z.T @ (Z * (pr * (1 - pr))[:, None]) + P, Z.T @ (pr - y) + P @ w)
        w = w - step
        if np.abs(step).max() < 1e-13:
            break
    return lambda Xn: w[0] + ((Xn - m) / s) @ w[1:]


def bf_user_auc(score, y, users, rows):
    """Per-user AUC by metrics.auroc over each user's rows (one call per user)."""
    out = {}
    rows = np.asarray(rows, bool)
    ridx = np.flatnonzero(rows)
    for u, idx in fx._groups(np.asarray(users)[rows]).items():
        j = ridx[idx]
        if 0 < y[j].sum() < len(j):
            out[u] = auroc(score[j], y[j])
    return out


def bf_within_user(users, y, feats: dict, stackers: dict, rows, sd_ids, K):
    """Brute force of E-W: centre each feature over the user's rows, K hash folds, fit on the other fold, per-user AUC
    with metrics.auroc, averaged over the splits -> {stacker: {user: mean AUC}}."""
    C = {}
    ridx = np.flatnonzero(rows)
    groups = [ridx[idx] for idx in fx._groups(np.asarray(users)[rows]).values()]
    for f, v in feats.items():
        c = np.full(len(v), np.nan)
        for idx in groups:
            c[idx] = v[idx] - v[idx].mean()
        C[f] = c
    acc = {s: {} for s in stackers}
    for k in range(K):
        A = {u for u in sd_ids if int(hashlib.sha1(f"{k}:{u}".encode()).hexdigest(), 16) % 2 == 0}
        fa = np.array([u in A for u in users.tolist()], bool)
        for s, cols in stackers.items():
            Xf = np.column_stack([C[c] for c in cols])
            eta = np.full(len(y), np.nan)
            for fit, pred in ((rows & ~fa, rows & fa), (rows & fa, rows & ~fa)):
                eta[pred] = bf_logit(Xf[fit], y[fit])(Xf[pred])
            for u, a in bf_user_auc(eta, y, users, rows).items():
                acc[s].setdefault(u, []).append(a)
    return {s: {u: float(np.mean(v)) for u, v in d.items() if len(v) == K} for s, d in acc.items()}


# ---------------------------------------------------------------- the registered numbers, reproduced exactly
def test_registered_pooled_reproduces_the_report_exactly(ml1m):
    rp, rep = ml1m.res["registered_pooled"], ml1m.rep
    for reg in ("ZS", "FT"):
        for k in ("shares", "information_gain", "star_permutation"):
            assert _dump(rp["E_D"][reg][k]) == _dump(rep["E_D"][reg][k]), (reg, k)
        ig = rp["E_D"][reg]["information_gain"]
        for m, blk in ig["per_model"].items():          # the pooled M0-M3 UAUCs, G, G_CF (spot check of the dump)
            assert blk["UAUC"] == rep["E_D"][reg]["information_gain"]["per_model"][m]["UAUC"]
        assert ig["G_CF"] == rep["E_D"][reg]["information_gain"]["G_CF"]
        sh = rp["E_D"][reg]["shares"]["mean_over_seeds"]
        assert sh["item_prior_share"] == rep["E_D"][reg]["shares"]["mean_over_seeds"]["item_prior_share"]
        assert sh["non_prior_share"] == rep["E_D"][reg]["shares"]["mean_over_seeds"]["non_prior_share"]
    assert _dump(rp["E_B"]) == _dump(rep["E_B"]) and _dump(rp["P1"]) == _dump(rep["P1"])   # P1's inputs and decision
    assert rp["P1"]["decision"]["verdict"] == "P1_HOLDS"
    # the E-D family that summarize copies through equals the report's (ed_holm_family on the copied block)
    for reg in ("ZS", "FT"):
        assert _dump(strict_json_of(fr.ed_holm_family(rp["E_D"][reg], reg))) == _dump(rep["E_D"][reg]["holm_family_E_D"])
    # E-G (i) is E-D's share block, unchanged
    for reg in ("ZS", "FT"):
        assert _dump(ml1m.res["E_G"][reg]["item_share_L_pi"]) == _dump(rep["E_D"][reg]["shares"])
    assert ml1m.res["meta"]["fold_k0_equals_E_D_fold_a"] is True
    assert ml1m.res["references"]["source"] == "cache (--refs)"


def strict_json_of(x):
    return json.loads(json.dumps(fe.strict_json(x)))


def test_e_c_prime_oracle_values_equal_the_reports_e_c_anatomy(ml1m):
    for reg in ("ZS", "FT"):
        ec, rep = ml1m.res["E_Cprime"][reg], ml1m.rep["E_C"][reg]
        for m, blk in ec["oracle"]["per_model"].items():
            for k in fr.ANATOMY_KEYS:
                assert _dump(blk[k]) == _dump(rep["per_model"][m][k]), (reg, m, k)
        for k in fr.ANATOMY_KEYS:
            assert _dump(ec["oracle"]["mean_over_seeds"][k]) == _dump(rep["mean_over_seeds"][k])
        assert ec["rows"]["n_users"] == rep["rows"]["n_users_anatomy"]
        assert ec["rows"]["n_pairs_deployable"] == ec["rows"]["n_pairs"]          # identical rows for both decisions
        d = ec["deployable_minus_oracle"]["mean_over_seeds"]["topk_error_rate"]["est"]
        assert d == pytest.approx(ec["deployable"]["mean_over_seeds"]["topk_error_rate"]["est"]
                                  - ec["oracle"]["mean_over_seeds"]["topk_error_rate"]["est"])
        assert ec["k_rule"]["n_users_pooled_rate_fallback"] <= ec["k_rule"]["n_users"]


def test_domain_file_is_strict_json_without_holm_and_has_every_block(ml1m):
    loaded = _strict_load(ml1m.out)
    assert loaded == ml1m.res
    for k in ("spec", "status", "meta", "input_checks", "excluded_runs", "runs", "references", "readings",
              "registered_pooled", "E_W", "E_F", "E_G", "E_H", "E_J", "E_Cprime", "FT_C_reading"):
        assert k in loaded, k
    text = ml1m.out.read_text(encoding="utf-8")
    assert "p_holm" not in text and "holm_family_E_D" not in text           # raw p-values only (addendum 1 item 8)
    assert ml1m.w.root.name not in text and json.dumps(str(ml1m.w.root))[1:-1] not in text   # no path
    assert loaded["status"]["items_2_to_7"] == "exploratory" and loaded["status"]["family_eligible_panel"] is False
    assert loaded["input_checks"]["problems"] == [] and loaded["excluded_runs"] == []
    assert loaded["references"]["q_hat_T"]["q_hat_recomputed_equals_reference"] is True
    assert loaded["meta"]["references_source"] == "cache (--refs)" and "T_d" in loaded["meta"]["q_hat_T_source"]
    assert loaded["readings"] == list(fe.READINGS) and len(fe.READINGS) == 14          # the docstring's R1-R14
    assert all(r.startswith(f"R{k + 1} ") for k, r in enumerate(fe.READINGS))
    for blk in ("registered_pooled", "E_W", "E_F", "E_G", "E_H", "E_J", "E_Cprime", "FT_C_reading"):   # never null
        assert isinstance(loaded[blk], dict)
        for reg in ("ZS", "FT"):
            if reg in loaded[blk]:
                assert isinstance(loaded[blk][reg], dict), (blk, reg)
    rows = list(csv.DictReader(open(ml1m.out.with_name("ml1m_tables.csv"), encoding="utf-8")))
    assert tuple(rows[0]) == fe.CSV_COLS
    blocks = {r["block"] for r in rows}
    assert {"registered_pooled", "E-W", "E-F", "E-G", "E-H", "E-J", "E-C'", "FT-C"} <= blocks
    assert all(r["descriptive"] == "true" for r in rows if r["block"] != "registered_pooled")   # ML-1M: exploratory


def test_build_is_deterministic_byte_for_byte(ml1m, tmp_path):
    _, out2 = run_extra(ml1m.w, "ml1m_again", list(ML1M_SPECS))
    assert out2.read_bytes() == ml1m.out.read_bytes()
    assert out2.with_name("ml1m_again_tables.csv").read_bytes() == ml1m.out.with_name("ml1m_tables.csv").read_bytes()


# ---------------------------------------------------------------- E-W
def test_e_w_removes_a_planted_user_offset_bias_and_keeps_a_planted_personal_signal():
    # user offsets track leniency, no personal signal: the pooled stacker's G is negative, the within-user G is ~0
    X = ctx_world(kappa=1.0, gamma=0.0)
    cx, R = X.cx, np.ones(X.cx.n, bool)
    q, mf = X.refs["arrays"]["q_hat"], X.refs["arrays"]["mf_residual"]
    pooled = fr.stacker_block(cx, X.models, X.L, X.PI, q, mf, R, X.n_boot, 0, 1)
    folds = fe.split_folds(cx)
    assert len(folds) == fe.K_SPLITS == 20 and np.array_equal(folds[0], cx.fold_a)
    info = info_of(X)
    rp = {"E_D": {"ZS": {"information_gain": pooled}}}
    ew, ef, _ = fe.wu_regime(X, "ZS", info, folds, {}, rp)
    g, gw = pooled["G_mean_over_seeds"], ew["G_wu"]["mean_over_seeds"]
    assert g["est"] < -0.015 and g["hi"] < 0                          # the reviewer's bias of the pooled estimator
    assert abs(gw["est"]) < 0.002 and gw["lo"] < 0 < gw["hi"]           # removed by the within-user estimator
    assert ew["reading_G"]["reading"] == "estimator_dependent" and ew["reading_G"]["robust"] is False
    # a planted personal signal is kept, by both estimators: robust
    X = ctx_world(kappa=1.0, gamma=1.0, seed=1)
    cx = X.cx
    pooled = fr.stacker_block(cx, X.models, X.L, X.PI, X.refs["arrays"]["q_hat"], X.refs["arrays"]["mf_residual"],
                              np.ones(cx.n, bool), X.n_boot, 0, 1)
    ew, _, _ = fe.wu_regime(X, "ZS", info_of(X), fe.split_folds(cx), {}, {"E_D": {"ZS": {"information_gain": pooled}}})
    gw = ew["G_wu"]["mean_over_seeds"]
    assert gw["est"] > 0.05 and gw["lo"] > 0 and gw["p"] < 0.05 and gw["descriptive_min_n"] is False
    assert ew["reading_G"]["reading"] == "robust" and ew["reading_G"]["robust"] is True
    assert ew["G_wu"]["per_model"]["zeroshot"]["est"] == pytest.approx(gw["est"])
    u = ew["UAUC"]
    assert gw["est"] == pytest.approx(u["M2"]["mean_over_seeds"]["est"] - u["M1"]["mean_over_seeds"]["est"])
    assert ew["G_CF_wu"]["est"] == pytest.approx(u["M3"]["est"] - u["M0"]["est"])


def test_e_w_equals_a_brute_force_twenty_split_computation_and_its_bootstrap():
    X = ctx_world(n_users=200, n_rows=10, n_items=300, kappa=1.0, gamma=0.7, mf_gamma=0.5, seed=3, n_boot=150)
    cx = X.cx
    folds = fe.split_folds(cx)
    ew, ef, _ = fe.wu_regime(X, "ZS", info_of(X), folds, {}, {})
    m = "zeroshot"
    feats = {"q_hat": X.refs["arrays"]["q_hat"], "mf_residual": X.refs["arrays"]["mf_residual"], "pi": X.PI[m],
             "e_hat": X.L[m] - X.PI[m]}
    bf = bf_within_user(cx.users, cx.y, feats, fe.STACKERS_X, np.ones(cx.n, bool), X.sd_ids, 20)
    users = sorted(bf["M0"])
    assert ew["rows"]["n_users"] == len(users) == 200
    gwu = {u: bf["M2"][u] - bf["M1"][u] for u in users}
    assert ew["G_wu"]["mean_over_seeds"]["est"] == pytest.approx(np.mean(list(gwu.values())), abs=1e-10)
    for s in ("M0", "M3"):
        assert ew["UAUC"][s]["est"] == pytest.approx(np.mean([bf[s][u] for u in users]), abs=1e-10)
    assert ew["G_CF_wu"]["est"] == pytest.approx(np.mean([bf["M3"][u] - bf["M0"][u] for u in users]), abs=1e-10)
    assert ef["G_LLM_given_CF"]["mean_over_seeds"]["est"] == pytest.approx(
        np.mean([bf["M4"][u] - bf["M3"][u] for u in users]), abs=1e-10)
    assert ef["G_CF_given_LLM"]["mean_over_seeds"]["est"] == pytest.approx(
        np.mean([bf["M4"][u] - bf["M2"][u] for u in users]), abs=1e-10)
    # the user bootstrap of the per-user split averages (coefficients fixed): stats.paired_bootstrap's draws
    pb = paired_bootstrap(gwu, {u: 0.0 for u in gwu}, n_boot=X.n_boot, seed=0)
    assert (ew["G_wu"]["mean_over_seeds"]["lo"], ew["G_wu"]["mean_over_seeds"]["hi"]) == pytest.approx(
        (pb["lo"], pb["hi"]), abs=1e-9)
    draws = fr.mean_draws(np.array([[gwu[u]] for u in users]), X.n_boot, 0)[:, 0]
    assert ew["G_wu"]["mean_over_seeds"]["p"] == pytest.approx(fr.boot_p(draws))
    # k = 0 alone and no centring is E-D's registered stacker exactly: the machinery differs only by the two registered
    # changes (centring, 20 splits)
    one = fe.wu_fit(cx, feats, {"M1": fe.STACKERS_X["M1"]}, np.ones(cx.n, bool), folds[:1])["M1"][0]
    Xc = np.column_stack([fr.user_centred(feats[c], cx.uc, np.ones(cx.n, bool)) for c in fe.STACKERS_X["M1"]])
    assert np.allclose(one, fr.crossfit(Xc, cx.y, cx.fold_a, np.ones(cx.n, bool)), atol=0, rtol=0)


def test_e_w_seed_mean_sigma_seed_minimum_n_and_p1_wu():
    X = ctx_world(n_users=400, kappa=1.0, gamma=0.8, n_models=3, seed=4)        # models s0, s1, s2
    cx = X.cx
    ew, ef, _ = fe.wu_regime(X, "FT", info_of(X, "FT"), fe.split_folds(cx), {}, {})
    g = ew["G_wu"]
    per = [g["per_model"][m]["est"] for m in X.models]
    assert g["mean_over_seeds"]["est"] == pytest.approx(np.mean(per))
    assert g["seeds"]["sigma_seed"] == pytest.approx(np.std(per, ddof=1))
    assert g["seeds"]["complete"] is True and g["seeds"]["n_seeds"] == 3
    assert g["seeds"]["sigma_seed_rule"] is bool(all(x > 0 for x in per) and np.mean(per) > 2 * np.std(per, ddof=1))
    # two seeds of three: never complete, the sigma_seed rule fails
    X2 = copy.copy(X)
    X2.models = X.models[:2]
    ew2, _, _ = fe.wu_regime(X2, "FT", {"models": X.models[:2], "missing_or_excluded": ["s2"], "n_registered": 3},
                             fe.split_folds(cx), {}, {})
    assert ew2["complete"] is False and ew2["G_wu"]["seeds"]["complete"] is False
    assert ew2["G_wu"]["seeds"]["sigma_seed_rule"] is False
    # fewer than 150 users: descriptive, no p-value, no CI-based reading
    S = ctx_world(n_users=120, kappa=1.0, gamma=1.0, seed=5)
    ews, _, _ = fe.wu_regime(S, "ZS", info_of(S), fe.split_folds(S.cx), {}, {"E_D": {"ZS": {"information_gain": {
        "G_mean_over_seeds": {"est": 0.05, "lo": 0.01, "hi": 0.09, "p": None, "ci_excludes_0": None,
                              "descriptive_min_n": True}}}}})
    assert ews["G_wu"]["mean_over_seeds"]["descriptive_min_n"] is True and ews["G_wu"]["mean_over_seeds"]["p"] is None
    assert ews["reading_G"]["reading"] == "descriptive_min_n"
    # P1_wu: p1_block's rows and rule on the within-user G values
    Z = ctx_world(n_users=400, kappa=1.0, gamma=0.2, n_models=1, seed=6)
    F = ctx_world(n_users=400, kappa=1.0, gamma=1.0, n_models=3, seed=6)       # same users, items and labels (seed)
    P = SimpleNamespace(cx=F.cx, L={"zeroshot": Z.L["zeroshot"], **F.L}, PI={"zeroshot": Z.PI["zeroshot"], **F.PI},
                        refs=F.refs, present={"zeroshot", "s0", "s1", "s2"}, n_boot=100, seed=0)
    assert np.array_equal(Z.cx.y, F.cx.y)
    p1w = fe.p1_wu_block(P, fe.split_folds(F.cx), {})
    assert p1w["available"] is True and p1w["decision"]["verdict"] == "P1_HOLDS"
    per = [p1w["per_seed"][s]["est"] for s in ("s0", "s1", "s2")]
    assert p1w["mean_over_seeds"]["est"] == pytest.approx(np.mean(per))
    assert p1w["per_seed"]["s0"]["est"] == pytest.approx(p1w["per_seed"]["s0"]["G_FT"] - p1w["G_ZS"])
    miss = fe.p1_wu_block(SimpleNamespace(**{**vars(P), "present": {"zeroshot", "s0", "s1"}}), fe.split_folds(F.cx), {})
    assert miss["available"] is False and miss["decision"]["verdict"] == "INCOMPLETE"


def test_robust_reading_rule_branches():
    def r(est, lo, hi, desc=False):
        return {"est": est, "lo": lo, "hi": hi, "ci_excludes_0": None if desc else bool(lo > 0 or hi < 0),
                "descriptive_min_n": desc}
    assert fe.robust_reading(r(0.01, 0.002, 0.02), r(0.009, 0.001, 0.018))["reading"] == "robust"
    assert fe.robust_reading(r(-0.018, -0.03, -0.01), r(0.0, -0.005, 0.005))["reading"] == "estimator_dependent"
    assert fe.robust_reading(r(0.01, 0.002, 0.02), r(-0.01, -0.02, -0.002))["reading"] == "estimator_dependent"
    assert fe.robust_reading(r(0.001, -0.005, 0.007), r(0.002, -0.004, 0.008))["reading"] == "both_intervals_include_0"
    assert fe.robust_reading(r(0.01, 0.002, 0.02, True), r(0.01, 0.002, 0.02))["reading"] == "descriptive_min_n"
    ft = fe.robust_reading(r(0.01, 0.002, 0.02), r(0.01, 0.002, 0.02), [0.01, 0.012, -0.001], [0.009, 0.01, 0.011],
                           fine_tuned=True)
    assert ft["reading"] == "seed_sign_disagreement" and ft["robust"] is False
    two = fe.robust_reading(r(0.01, 0.002, 0.02), r(0.01, 0.002, 0.02), [0.01, 0.012], [0.009, 0.01], fine_tuned=True)
    assert two["robust"] is False                                          # a missing seed is never robust
    assert fe.robust_reading(None, r(0.01, 0.002, 0.02))["reading"] == "not_available"


# ---------------------------------------------------------------- E-F
def test_e_f_contrasts_are_on_identical_rows_and_separate_llm_from_cf_evidence():
    # the personal evidence is in MF only: the LLM adds (almost) nothing given CF, CF adds a lot given the LLM
    X = ctx_world(n_users=500, kappa=1.0, gamma=0.0, mf_gamma=1.2, seed=7)
    ew, ef, _ = fe.wu_regime(X, "ZS", info_of(X), fe.split_folds(X.cx), {}, {})
    assert ef["rows"] == ew["rows"]                                          # identical users and rows for M2, M3, M4
    llm, cf = ef["G_LLM_given_CF"]["mean_over_seeds"], ef["G_CF_given_LLM"]["mean_over_seeds"]
    assert cf["est"] > 0.05 and cf["lo"] > 0 and abs(llm["est"]) < 0.01
    assert llm["n_users"] == cf["n_users"] == ew["G_wu"]["mean_over_seeds"]["n_users"]
    # the personal evidence is in the LLM only
    Y = ctx_world(n_users=500, kappa=1.0, gamma=1.2, mf_gamma=0.0, seed=8)
    ew, ef, _ = fe.wu_regime(Y, "ZS", info_of(Y), fe.split_folds(Y.cx), {}, {})
    llm, cf = ef["G_LLM_given_CF"]["mean_over_seeds"], ef["G_CF_given_LLM"]["mean_over_seeds"]
    assert llm["est"] > 0.05 and llm["lo"] > 0 and llm["p"] < 0.05 and abs(cf["est"]) < 0.01
    u4 = ef["UAUC_M4"]["mean_over_seeds"]["est"]
    assert llm["est"] == pytest.approx(u4 - ew["UAUC"]["M3"]["est"])
    assert cf["est"] == pytest.approx(u4 - ew["UAUC"]["M2"]["mean_over_seeds"]["est"])


# ---------------------------------------------------------------- E-H
def test_e_h_strata_partition_the_rows_the_sparse_rule_and_the_minimum_n():
    X = ctx_world(n_users=500, kappa=1.0, gamma=0.8, seed=9)
    cx, items = X.cx, X.items
    n_prior = (items % 11).astype(float)                       # 0..10 per item: sparse iff < 5 (4 sparse, 5 dense)
    n_prior[3] = np.nan                                        # a pair without a support count: in neither stratum
    seen = items >= 40                                         # 40 of 500 items unseen: few users have both classes
    X.refs["arrays"]["n_prior"] = n_prior
    strata = {"sparse": np.isfinite(n_prior) & (n_prior < 5), "dense": np.isfinite(n_prior) & (n_prior >= 5),
              "unseen": ~seen, "seen": seen}
    folds = fe.split_folds(cx)
    ew, _, eh = fe.wu_regime(X, "ZS", info_of(X), folds, strata, {})
    part = eh["partition"]
    R = np.ones(cx.n, bool)
    assert part["sparse_and_dense_disjoint"] is True and part["seen_and_unseen_partition_R"] is True
    assert part["n_pairs_sparse"] + part["n_pairs_dense"] + part["n_pairs_R_without_n_prior"] == part["n_pairs_R"]
    assert part["n_pairs_R_without_n_prior"] == 1 and part["n_pairs_unseen"] + part["n_pairs_seen"] == cx.n
    assert part["n_pairs_sparse"] == int((np.isfinite(n_prior) & (n_prior <= 4)).sum())
    st = eh["strata"]
    assert st["unseen"]["n_users_both_classes"] < fr.MIN_N and st["unseen"]["descriptive_min_n"] is True
    assert st["unseen"]["G_prior"]["mean_over_seeds"]["p"] is None              # descriptive: no p-value
    assert st["sparse"]["n_users_both_classes"] >= fr.MIN_N and st["sparse"]["G_prior"]["mean_over_seeds"]["p"] is not None
    # E-W's predictors fit on ALL rows of the other fold; the stratum restricts the evaluation only (brute force)
    sh, per = fe.regime_etas(cx, X.models, X.L, X.PI, X.refs["arrays"]["q_hat"], X.refs["arrays"]["mf_residual"], R,
                             folds)
    rows = strata["sparse"]
    m = X.models[0]
    acc = {}
    for k in range(len(folds)):
        a1 = bf_user_auc(per[m]["M1"][k], cx.y, cx.users, rows)
        a0 = bf_user_auc(sh["M0"][k], cx.y, cx.users, rows)
        for u in a1:
            acc.setdefault(u, []).append(a1[u] - a0[u])
    want = np.mean([np.mean(v) for v in acc.values()])
    assert st["sparse"]["G_prior"]["mean_over_seeds"]["est"] == pytest.approx(want, abs=1e-12)
    assert st["sparse"]["n_users_both_classes"] == len(acc)
    lv = bf_user_auc(X.L[m], cx.y, cx.users, rows)
    assert st["sparse"]["UAUC_L"]["mean_over_seeds"]["est"] == pytest.approx(np.mean(list(lv.values())), abs=1e-12)
    qv = bf_user_auc(X.refs["arrays"]["q_hat"], cx.y, cx.users, rows)
    assert st["sparse"]["UAUC_q_hat"]["est"] == pytest.approx(np.mean(list(qv.values())), abs=1e-12)
    # the whole-row G_wu is E-W's; the strata partition its users' rows
    assert ew["rows"]["n_pairs"] == cx.n


# ---------------------------------------------------------------- E-G
def test_item_shares_equal_a_direct_numpy_computation():
    rng = np.random.default_rng(10)
    n_users, n_rows, n_items = 300, 9, 200
    users = np.repeat(np.arange(n_users), n_rows)
    items = rng.integers(0, n_items, len(users))
    bi = rng.normal(0, 1, n_items)[items]
    s = np.repeat(rng.normal(0, 2, n_users), n_rows) + bi + rng.normal(0, 0.8, len(users))   # MF score
    rows = rng.random(len(users)) < 0.9
    out = fe.rel1_share(users, rows, s, bi, 200, 0)

    def centred(x, r):
        c = np.full(len(x), np.nan)
        for u in np.unique(users[r]):
            idx = np.flatnonzero(r & (users == u))
            c[idx] = x[idx] - x[idx].mean()
        return c[r]
    sc, bc = centred(s, rows), centred(bi, rows)
    rho = (sc * bc).sum() / math.sqrt((sc * sc).sum() * (bc * bc).sum())
    assert out["item_share"]["est"] == pytest.approx(rho ** 2, abs=1e-12)
    assert out["rho"]["est"] == pytest.approx(rho, abs=1e-12) and out["reliability"] == 1.0
    assert out["rows"]["n_pairs"] == int(rows.sum())
    assert out["item_share"]["lo"] < rho ** 2 < out["item_share"]["hi"]
    # the label with q-hat (binary score)
    y = (s + rng.normal(0, 1, len(s)) > 0).astype(float)
    qh = bi + rng.normal(0, 0.3, len(s))
    lab = fe.rel1_share(users, rows, y, qh, 50, 0)
    yc, qc = centred(y, rows), centred(qh, rows)
    assert lab["item_share"]["est"] == pytest.approx((yc * qc).sum() ** 2 / ((yc * yc).sum() * (qc * qc).sum()),
                                                     abs=1e-12)


def _eshare_world(c, n_users=1500, n_rows=12, n_items=1500, sd_dn=0.8, seed=11):
    """L = off_u + prior_i + e_ui with e = c prior + independent: Cov(prior, e) = c Var(prior); pi = prior + the mean of
    8 donor errors (halves 1-4, 5-8). Returns the planted Var(e_c) / Var(L_c) of the user-centred values."""
    rng = np.random.default_rng(seed)
    prior_i = rng.normal(0, 1, n_items)
    donors = rng.normal(0, sd_dn, (n_items, 8))
    users = np.repeat(np.arange(n_users), n_rows)
    items = rng.integers(0, n_items, len(users))
    prior = prior_i[items]
    e = c * prior + rng.normal(0, 0.8, len(users))
    L = np.repeat(rng.normal(0, 2, n_users), n_rows) + prior + e
    pi = prior + donors.mean(1)[items]
    pa, pb = prior + donors[:, :4].mean(1)[items], prior + donors[:, 4:].mean(1)[items]
    rows = np.ones(len(users), bool)
    ec = fr.user_centred(e, users, rows)
    lc = fr.user_centred(L, users, rows)
    return users, rows, (pa, pb, L, pi), float((ec ** 2).sum() / (lc ** 2).sum())


def test_e_share_recovers_the_planted_share_and_exceeds_the_non_prior_share_when_pi_and_e_covary():
    users, rows, cols, truth = _eshare_world(0.0)
    out = fe.eshare_block(users, rows, {"m": cols}, 200, 0, 1)
    m = out["per_model"]["m"]
    assert m["e_share"]["est"] == pytest.approx(truth, abs=0.02)
    assert m["e_share"]["lo"] < truth < m["e_share"]["hi"]
    assert m["non_prior_share"] == pytest.approx(truth, abs=0.03)          # Cov(pi, e) = 0: the two agree
    assert m["var_e_over_var_L_uncorrected"] > m["e_share"]["est"]        # the donor error inflates Var(L - pi)
    users, rows, cols, truth = _eshare_world(0.5, seed=12)
    out = fe.eshare_block(users, rows, {"m": cols}, 200, 0, 1)
    m = out["per_model"]["m"]
    assert m["e_share"]["est"] == pytest.approx(truth, abs=0.02)
    assert m["non_prior_share"] < truth - 0.05                              # A3-6 item 1: below the e-share
    # the same rows, sums and resamples as shares_block: its non-prior share is the one reported beside
    sb = fr.shares_block(users, rows, {"m": cols}, 200, 0, 1)
    assert sb["per_model"]["m"]["non_prior_share"]["est"] == pytest.approx(m["non_prior_share"])
    assert out["rows"]["n_pairs"] == sb["rows"]["n_pairs"]


# ---------------------------------------------------------------- E-J
def test_e_j_uses_e_a_rows_q_hat_T_and_paired_contrasts(ml1m):
    w, res, rep = ml1m.w, ml1m.res, ml1m.rep
    for reg in ("ZS", "FT"):
        ej = res["E_J"][reg]
        ua = rep["E_A"][reg]["UAUC_TEST"]["rows"]
        assert ej["rows_E_A"] == {"n_pairs": ua["n_pairs"], "n_users": ua["n_users"]}     # E-A's TEST users and rows
        blk = ej["dUAUC_L_minus_q_hat"]
        assert blk["rows"]["n_users"] == ua["n_users"]
        for m in blk["per_seed"]:
            assert blk["per_seed"][m]["UAUC_a"] == pytest.approx(rep["E_A"][reg]["UAUC_TEST"]["per_model"][m]["est"])
            assert blk["per_seed"][m]["UAUC_b"] == pytest.approx(rep["E_A"][reg]["UAUC_TEST"]["references"]["q_hat"]["est"])
    # q-hat_T: prior_means at T_d from an independent scan; differs from q-hat on TEST rows
    rows = fx.read_jsonl(w.d / "eval.jsonl")
    P = fx.panel_pairs(rows)
    events = fx.load_raw_events(w.raw, "ml1m")
    scan = fx.scan_events(events, set(P["item"].tolist()), set(zip(P["user"].tolist(), P["item"].tolist())))
    qT = fx.prior_means(scan, P["user"].tolist(), P["item"].tolist(), np.full(P["n"], w.T), 5.0)["mean_prior_shrunk"]
    qt = fx.prior_means(scan, P["user"].tolist(), P["item"].tolist(), P["ts"], 5.0)["mean_prior_shrunk"]
    test = P["ts"] >= w.T
    assert not np.allclose(qT[test], qt[test])
    L = {}
    sc = fx.load_scores(w.root / "scores" / "ml1m" / "s0" / "like" / "scores.csv.gz")
    for (ev, c), lg in sc["L"]["like"].items():
        L[(sc["meta"][(ev, c)][0], sc["meta"][(ev, c)][1])] = lg
    Ls0 = np.array([L.get((u, i), np.nan) for u, i in zip(P["user"].tolist(), P["item"].tolist())])
    ref = rep["E_A"]["FT"]
    users = P["user"]
    Rr = test & np.isfinite(Ls0)                                  # the FT rows: every seed finite on this world
    ok = Rr & np.isfinite(qT)
    a, b = bf_user_auc(Ls0, P["label"], users, ok), bf_user_auc(qT, P["label"], users, ok)
    want = np.mean([a[u] - b[u] for u in a])
    assert res["E_J"]["FT"]["dUAUC_L_minus_q_hat_T"]["per_seed"]["s0"]["est"] == pytest.approx(want, abs=1e-12)
    assert ref["UAUC_TEST"]["rows"]["n_users"] == len(a)
    assert ref["UAUC_TEST"]["rows"]["n_pairs"] == int(np.isin(users[Rr], list(a)).sum())
    warm = res["E_J"]["FT"]["dUAUC_L_minus_MF_warm"]
    assert warm["descriptive"] is True and warm["rows"]["n_pairs"] <= res["E_J"]["FT"]["rows_E_A"]["n_pairs"]


# ---------------------------------------------------------------- E-C'
def test_cal_rate_k_rounds_half_up_falls_back_and_clips():
    #            user:  a (4 CAL, 2 likes)  b (4 CAL, 1 like)  c (2 CAL: pooled)  d (4 CAL, 0 likes)  e (4 CAL, 4 likes)
    cal_y = {"a": [1, 1, 0, 0], "b": [1, 0, 0, 0], "c": [1, 1], "d": [0, 0, 0, 0], "e": [1, 1, 1, 1]}
    test_y = {"a": [1, 0, 0], "b": [1, 0], "c": [0, 1, 0, 0, 1], "d": [1, 0, 0], "e": [1, 0, 1]}
    users, y, cal = [], [], []
    for u in cal_y:
        users += [u] * (len(cal_y[u]) + len(test_y[u]))
        y += cal_y[u] + test_y[u]
        cal += [True] * len(cal_y[u]) + [False] * len(test_y[u])
    users, y, cal = np.array(users), np.array(y), np.array(cal)
    uc = np.unique(users, return_inverse=True)[1].reshape(-1)
    cx = SimpleNamespace(uc=uc, y=y, cal=cal)
    k_row, info = fe.cal_rate_k(cx, ~cal)
    k = {u: int(k_row[np.flatnonzero((users == u) & ~cal)[0]]) for u in cal_y}
    pooled = 9 / 18                                              # CAL likes 2 + 1 + 2 + 0 + 4 = 9 of 18 rows
    assert info["pooled_cal_like_rate"] == pytest.approx(pooled) and info["n_users_pooled_rate_fallback"] == 1
    assert k["a"] == 2                     # 0.5 x 3 = 1.5 -> half up 2 (banker's rounding would give 2 as well)
    assert k["b"] == 1                     # 0.25 x 2 = 0.5 -> half up 1 (banker's rounding would give 0, then clip 1)
    assert k["c"] == 3                     # pooled 0.5 x 5 = 2.5 -> half up 3 (banker's rounding: 2)
    assert k["d"] == 1 and k["e"] == 2     # 0 -> clip 1; 1.0 x 3 = 3 -> clip n_u - 1 = 2
    assert np.all(k_row[cal] == -1)
    # with the oracle k (the user's TEST likes) the deployable anatomy is E-C's topk_anatomy exactly
    rng = np.random.default_rng(13)
    L = rng.normal(0, 1, len(y))
    rows = ~cal
    oracle = np.zeros(len(y), int)
    for u in cal_y:
        idx = np.flatnonzero((users == u) & rows)
        oracle[idx] = y[idx].sum()
    mine = fe.topk_anatomy_at(L, y, users, rows, oracle)
    ref = fr.topk_anatomy(L, y, users, rows)
    for a_, b_ in zip(mine, ref):
        assert np.array_equal(np.isnan(a_), np.isnan(b_)) and np.allclose(a_[~np.isnan(a_)], b_[~np.isnan(b_)], 0, 0)


# ---------------------------------------------------------------- FT-C reading
def _ftc_world(keep: str, seed=14, n_users=400, eb_positive=True):
    rng = np.random.default_rng(seed)
    n = n_users * 10
    users = np.repeat([f"u{k:04d}" for k in range(n_users)], 10)
    e = rng.normal(0, 1, n)
    y = (e + rng.normal(0, 1, n) > 0).astype(int)
    zs = 0.2 * e + rng.normal(0, 1, n)
    L = {"zeroshot": zs}
    for s in ("s0", "s1", "s2"):
        L[s] = (1.2 * e if eb_positive else -0.5 * e) + rng.normal(0, 1, n)     # E-B positive / negative
    for s in ("p0", "p1"):
        L[s] = {"all": L["s" + s[1]].copy(), "none": zs.copy(), "noise": zs + rng.normal(0, 0.01, n)}[keep]
    uc = np.unique(users, return_inverse=True)[1].reshape(-1)
    cx = SimpleNamespace(test=np.ones(n, bool), y=y, uc=uc, users=users)
    eb = {"complete": True, **fr.contrast_models({s: (L[s], zs) for s in ("s0", "s1", "s2")}, y, uc, cx.test, 200, 0,
                                                 n_registered=3)}
    return SimpleNamespace(cx=cx, L=L, present=set(L), n_boot=200, seed=0), eb


def test_ft_c_retention_on_planted_worlds_and_the_label_rule():
    X, eb = _ftc_world("all")                                   # the permuted adapter keeps all of the gain
    out = fe.ftc_reading(X, eb)
    assert out["defined"] is True and out["R"]["est"] == pytest.approx(1.0)
    assert out["R"]["lo"] == pytest.approx(1.0) and out["label"] == "ITEM_DRIVEN"
    X, eb = _ftc_world("none")                                  # ... and none of it
    out = fe.ftc_reading(X, eb)
    assert out["R"]["est"] == pytest.approx(0.0, abs=1e-12) and out["label"] == "USER_DRIVEN"
    # one resample drives all UAUCs: the CI equals the ratio computed per resample of the per-user AUC matrix
    X, eb = _ftc_world("noise")
    out = fe.ftc_reading(X, eb)
    pu = {m: fr.user_aucs(X.L[m], X.cx.y, X.cx.uc) for m in fe.FTC_MODELS}
    keys = sorted(pu["zeroshot"])
    V = np.array([[pu[m][u] for m in fe.FTC_MODELS] for u in keys])
    D = fr.mean_draws(V, X.n_boot, 0)
    r = ((D[:, 3] - D[:, 0]) / (D[:, 1] - D[:, 0]) + (D[:, 4] - D[:, 0]) / (D[:, 2] - D[:, 0])) / 2
    assert (out["R"]["lo"], out["R"]["hi"]) == pytest.approx(tuple(np.quantile(r, [0.025, 0.975])), abs=1e-12)
    assert out["UAUC"]["p0"]["est"] == pytest.approx(np.mean(V[:, 3]))
    # E-B not positive with a CI excluding 0: R is not defined and not reported
    X, eb = _ftc_world("all", eb_positive=False)
    out = fe.ftc_reading(X, eb)
    assert out["defined"] is False and out["R"]["available"] is False and out["label"] == "NOT_DEFINED"
    assert "E_B_mean_gt_0" in out["R"]["reason"] and "est" not in out["R"]          # not reported when undefined
    # E-B incomplete (a seed missing): not defined (addendum 1 item 7)
    X, eb = _ftc_world("all")
    out = fe.ftc_reading(X, {**eb, "complete": False})
    assert out["label"] == "NOT_DEFINED" and out["E_B_condition"]["E_B_complete"] is False
    # p1 missing: available false with the reason
    X.present.discard("p1")
    out = fe.ftc_reading(X, eb)
    assert out["available"] is False and "p1" in out["reason"] and out["label"] == "NOT_DEFINED"
    # the label rule on planted intervals (strict inequalities at 0.5)
    lab = fe.ftc_label
    assert lab({"lo": 0.6, "hi": 0.9}) == "ITEM_DRIVEN" and lab({"lo": 0.1, "hi": 0.4}) == "USER_DRIVEN"
    assert lab({"lo": 0.4, "hi": 0.6}) == "MIXED" and lab({"lo": 0.5, "hi": 0.9}) == "MIXED"
    assert lab({"lo": 0.1, "hi": 0.5}) == "MIXED" and lab({"lo": None, "hi": 0.4}) == "NOT_DEFINED"


def test_ft_c_reading_end_to_end_and_its_e_b_condition(ml1m):
    ftc, eb = ml1m.res["FT_C_reading"], ml1m.rep["E_B"]
    assert ftc["available"] is True and ftc["E_B_mean_over_seeds"]["est"] == eb["mean_over_seeds"]["est"]
    assert ftc["defined"] is True and ftc["label"] in ("ITEM_DRIVEN", "USER_DRIVEN", "MIXED")
    u = {m: ftc["UAUC"][m]["est"] for m in fe.FTC_MODELS}
    want = ((u["p0"] - u["zeroshot"]) / (u["s0"] - u["zeroshot"]) + (u["p1"] - u["zeroshot"]) / (u["s1"] - u["zeroshot"])) / 2
    assert ftc["R"]["est"] == pytest.approx(want, abs=1e-12)
    assert ftc["label"] == fe.ftc_label(ftc["R"])


# ---------------------------------------------------------------- integrity, missing seeds
def test_failed_integrity_run_is_excluded_and_the_regime_is_incomplete(tmp_path):
    w = make_world(tmp_path, "ml1m", seed=13, n_eval=300, n_bg=60)
    n_pairs = sum(len(r["candidate_item_ids"]) for r in w.eval_rows)
    specs = {"zeroshot": {"beta": 0.3}, "s0": {"beta": 1.0}, "s1": {"beta": 1.0}, "s2": {"beta": 1.0}}
    write_models(w, specs, arms=("like", "swap"), cens2={("s2", "like"): tuple(range(0, n_pairs, 20))})   # 5%
    res, _ = run_extra(w, "bad", list(specs), n_boot=20, refs=False)
    assert [(e["model"], e["arm"], e["status"]) for e in res["excluded_runs"]] == [("s2", "like", "FAILED_INTEGRITY")]
    assert res["runs"]["s2"]["like"]["integrity"]["E1"] is False
    for blk in ("E_W", "E_F", "E_J", "E_Cprime"):
        assert res[blk]["FT"]["models"] == ["s0", "s1"] and res[blk]["FT"]["complete"] is False, blk
    assert res["E_W"]["FT"]["G_wu"]["seeds"]["complete"] is False
    assert res["E_W"]["FT"]["G_wu"]["seeds"]["sigma_seed_rule"] is False
    assert res["E_F"]["FT"]["G_LLM_given_CF"]["seeds"]["sigma_seed_rule"] is False
    assert res["E_J"]["FT"]["dUAUC_L_minus_q_hat"]["seeds"]["complete"] is False
    assert res["E_W"]["P1_wu"]["decision"]["verdict"] == "INCOMPLETE"
    assert res["registered_pooled"]["P1"]["decision"]["verdict"] == "INCOMPLETE"
    assert res["registered_pooled"]["E_D"]["FT"]["star_permutation"]["available"] is False
    assert res["FT_C_reading"]["available"] is False and res["FT_C_reading"]["label"] == "NOT_DEFINED"
    assert res["references"]["source"].startswith("computed from --raw")


def test_without_references_every_dependent_block_is_unavailable_never_an_exception(tmp_path):
    w = make_world(tmp_path, "toys", seed=12, n_eval=200, n_bg=40, raw=False)
    specs = {m: {"beta": 0.5} for m in ("zeroshot", "s0", "s1", "s2")}
    write_models(w, specs, arms=("like", "swap"))
    res, out = run_extra(w, "toys", list(specs), n_boot=10, refs=False, raw=False)
    for blk in ("E_W", "E_F", "E_H"):
        assert res[blk]["ZS"]["available"] is False, blk
    assert res["E_J"]["available"] is False and res["E_G"]["MF_score_item_bias"]["available"] is False
    assert res["E_G"]["FT"]["e_share"]["mean_over_seeds"]["n_users"] >= 0            # E-G (i) needs no reference
    assert res["E_Cprime"]["FT"]["models"] == ["s0", "s1", "s2"]
    assert res["references"]["q_hat_T"]["available"] is False
    assert res["status"]["family_eligible_panel"] is True and res["status"]["items_2_to_7"] == "registered_outcome_free"
    _strict_load(out)


# ---------------------------------------------------------------- summarize: the A3-6 families
def _stat(est, p, n=400, desc=False):
    return {"est": est, "lo": est - 0.01, "hi": est + 0.01, "p": None if desc else p, "n_users": n,
            "descriptive_min_n": desc}


def _seeds(rule=True, complete=True):
    return {"sigma_seed_rule": rule, "complete": complete, "per_seed": [0.01, 0.011, 0.012], "sigma_seed": 0.001}


def _doc(domain, backbone="Qwen3-8B", ef=(0.02, 0.01, True), eh_zs=(0.02, 0.01, 400), eh_ft=(0.02, 0.01, 400, True),
         ej=(-0.03, 0.002, True), eb=(0.1, 0.001, True)):
    ef_blk = {"mean_over_seeds": _stat(ef[0], ef[1]), "seeds": _seeds(ef[2])}
    return {"meta": {"domain": domain, "backbone": backbone, "code_sha1": {}},
            "status": {"family_eligible_panel": domain in fe.AMAZON and "qwen" in backbone.lower()},
            "registered_pooled": {
                "E_D": {reg: {"information_gain": {"G_mean_over_seeds": {"p": 0.01}, "seeds": {"sigma_seed_rule": True}},
                              "star_permutation": {"dUAUC_mean_over_seeds": {"p": 0.03},
                                                   "seeds": {"sigma_seed_rule": False}}} for reg in ("ZS", "FT")},
                "E_B": {"complete": eb[2], "mean_over_seeds": _stat(eb[0], eb[1]), "seeds": _seeds(True, eb[2])},
                "P1": {"available": True, "role": "r", "decision": {"verdict": "P1_HOLDS"}, "mean_over_seeds": _stat(0.01, 0.004)}},
            "E_W": {"P1_wu": {"decision": {"verdict": "P1_HOLDS"}, "mean_over_seeds": _stat(0.01, 0.004),
                              "reading_P1": {"reading": "robust"}}},
            "E_F": {"ZS": {"G_LLM_given_CF": {"mean_over_seeds": _stat(0.05, 0.0005), "seeds": _seeds()}},
                    "FT": {"G_LLM_given_CF": ef_blk}},
            "E_H": {"ZS": {"strata": {"sparse": {"G_prior": {"mean_over_seeds": _stat(eh_zs[0], eh_zs[1], eh_zs[2],
                                                                                      eh_zs[2] < 150),
                                                             "seeds": _seeds()}}}},
                    "FT": {"strata": {"sparse": {"G_prior": {"mean_over_seeds": _stat(eh_ft[0], eh_ft[1], eh_ft[2],
                                                                                      eh_ft[2] < 150),
                                                             "seeds": _seeds(eh_ft[3])}}}}},
            "E_J": {"ZS": {"dUAUC_L_minus_q_hat": {"mean_over_seeds": _stat(0.04, 0.001), "seeds": _seeds()}},
                    "FT": {"dUAUC_L_minus_q_hat": {"mean_over_seeds": _stat(ej[0], ej[1]), "seeds": _seeds(ej[2])}}}}


def _write_docs(tmp_path, docs: dict):
    paths = {}
    for name, d in docs.items():
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        paths[name] = str(p)
    return paths


def test_summarize_holm_families_sigma_seed_rule_direction_and_membership(tmp_path):
    docs = {"toys": _doc("toys", ef=(0.02, 0.01, True), eh_zs=(0.02, 0.01, 400), eh_ft=(0.03, 0.02, 400, True),
                         ej=(-0.03, 0.002, True)),
            "games": _doc("games", ef=(0.02, 0.02, False), eh_zs=(0.02, 0.001, 120), eh_ft=(0.03, 0.001, 400, True),
                          ej=(0.03, 0.06, True)),
            "sports": _doc("sports", ef=(-0.03, 0.001, True), eh_zs=(-0.02, 0.001, 400), eh_ft=(0.03, 0.03, 400, False),
                           ej=(0.02, 0.001, False)),
            "ml1m": _doc("ml1m", ef=(0.05, 0.0001, True)), "llama_toys": _doc("toys", backbone="Llama-3.1-8B-Instruct")}
    p = _write_docs(tmp_path, docs)
    out = tmp_path / "summary.json"
    res = fe.main(["summarize", "--files", p["toys"], p["games"], p["sports"], p["llama_toys"], "--ml1m", p["ml1m"],
                   "--out", str(out)])
    _strict_load(out)
    F = res["families"]["E_F"]
    assert set(F["members"]) == {"toys", "games", "sports"} and F["m"] == 3        # never ML-1M, Llama or ZS
    adj = fr.holm({"toys": 0.01, "games": 0.02, "sports": 0.001})
    for d in ("toys", "games", "sports"):
        assert F["members"][d]["p_holm"] == pytest.approx(adj[d])
    assert F["members"]["toys"]["confirmed"] is True                            # Holm 0.02 < 0.05, rule, > 0
    assert F["members"]["games"]["confirmed"] is False                          # the sigma_seed rule fails
    assert F["members"]["sports"]["confirmed"] is False and F["members"]["sports"]["direction_ok"] is False   # H-F > 0
    H = res["families"]["E_H"]
    assert "games:ZS" in H["outside_family"] and "descriptive" in H["outside_family"]["games:ZS"]["reason"]   # < 150
    assert set(H["members"]) == {"toys:ZS", "toys:FT", "games:FT", "sports:ZS", "sports:FT"} and H["m"] == 5
    adj = fr.holm({"toys:ZS": 0.01, "toys:FT": 0.02, "games:FT": 0.001, "sports:ZS": 0.001, "sports:FT": 0.03})
    assert all(H["members"][k]["p_holm"] == pytest.approx(adj[k]) for k in adj)
    assert H["members"]["toys:ZS"]["confirmed"] is True and H["members"]["toys:ZS"]["fine_tuned"] is False
    assert H["members"]["sports:ZS"]["confirmed"] is False                     # negative: H-S is directional
    assert H["members"]["sports:FT"]["confirmed"] is False                     # Holm p 0.03 x ... and the rule fails
    assert H["members"]["games:FT"]["confirmed"] is True
    J = res["families"]["E_J"]
    assert J["members"]["toys"]["confirmed"] is True and J["members"]["toys"]["sign"] == -1   # two-sided: as found
    assert J["members"]["games"]["confirmed"] is False                          # Holm p 0.06 > 0.05
    assert J["members"]["games"]["p_holm"] == pytest.approx(0.06)
    assert J["members"]["sports"]["confirmed"] is False                         # the sigma_seed rule fails
    assert res["families"]["E_F"]["n_confirmed"] == 1
    # the registered families copied through
    eb = res["registered"]["E_B"]
    assert set(eb["members"]) == {"toys", "games", "sports", "ml1m"} and eb["complete_family"] is True
    assert all(v["confirmed"] for v in eb["members"].values())
    ed = res["registered"]["E_D"]["toys:Qwen3-8B"]["FT"]
    assert ed == strict_json_of(fr.ed_holm_family(docs["toys"]["registered_pooled"]["E_D"]["FT"], "FT"))
    assert res["registered"]["P1"]["P1"]["decision"]["verdict"] == "P1_HOLDS"
    why = {(o["domain"], o["backbone"]): o["why"] for o in res["descriptive_outside_families"]}
    assert "ML-1M" in why[("ml1m", "Qwen3-8B")] and "Llama" in why[("toys", "Llama-3.1-8B-Instruct")]
    # without ML-1M the E-B family copy is marked incomplete
    res2 = fe.main(["summarize", "--files", p["toys"], p["games"], p["sports"], "--out", str(tmp_path / "s2.json")])
    assert res2["registered"]["E_B"]["complete_family"] is False and res2["registered"]["P1"]["available"] is False
    with pytest.raises(SystemExit):
        fe.main(["summarize", "--files", p["toys"], p["toys"], "--out", str(tmp_path / "s3.json")])


def _posix_bash():
    import os
    import shutil
    if os.name == "nt":                     # the PATH bash of Windows is WSL's; use Git Bash when it exists
        for p in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
            if Path(p).is_file():
                return p
        return None
    return shutil.which("bash")


def _freezable(split: Path, ml1m: bool) -> None:
    """A synthetic split as a stage-0 split looks (tokenizer audit present; ML-1M T checked against Gate-FT)."""
    js = json.loads(split.read_text(encoding="utf-8"))
    js["train"]["overlength"] = {"share_above_1024": 0.0}
    if ml1m:
        js["gateft_T_match"] = True
    split.write_text(json.dumps(js, indent=2), encoding="utf-8")


@pytest.mark.skipif(_posix_bash() is None, reason="no POSIX bash")
def test_run_script_dry_run_freeze_record_rule_resume_and_summarize(tmp_path):
    import os
    import subprocess
    import sys
    from src.confrec import ftgrid_freeze as ff
    repo = Path(fe.__file__).resolve().parents[2]
    script = (repo / "scripts" / "sigir" / "run_ftextra.sh").as_posix()
    root = tmp_path / "dry"
    wm = make_world(root, "ml1m", seed=21, n_eval=220, n_bg=40)
    specs = {m: {"beta": b} for m, b in (("zeroshot", 0.3), ("s0", 1.0), ("s1", 1.0), ("s2", 1.0))}
    write_models(wm, specs, arms=("like", "swap"))
    wt = make_world(tmp_path / "toysw", "toys", seed=22, n_eval=200, n_bg=30, raw=False)
    write_models(wt, {m: {"beta": 0.5} for m in specs}, arms=("like", "swap"))
    wt.d.rename(root / "panels" / "toys")
    (wt.root / "scores" / "toys").rename(root / "scores" / "toys")
    for d in ("ml1m", "toys"):
        _freezable(root / "panels" / d / "ftgrid_split.json", d == "ml1m")
    log = tmp_path / "PILOT_LOG.md"
    core = ff.lines_for("core", repo, [str(root / "panels" / d / "ftgrid_split.json") for d in ("ml1m", "toys")])
    log.write_text("\n".join(core) + "\n", encoding="utf-8")
    env = {**os.environ, "DRY_RUN": "1", "OUT_ROOT": root.as_posix(), "PILOT_LOG": log.as_posix(), "N_BOOT": "20",
           "PYTHON": Path(sys.executable).as_posix(), "MODELS": "zeroshot,s0,s1,s2"}

    def run(arg, **kw):
        return subprocess.run([_posix_bash(), script, arg], env={**env, **kw}, capture_output=True, text=True,
                              timeout=900)
    r = run("ml1m")                                            # ML-1M: exploratory, no A3-6 record needed
    assert r.returncode == 0, r.stderr[-2000:]
    out = root / "extra" / "ml1m.json"
    res = _strict_load(out)
    assert res["meta"]["n_boot"] == 20 and (root / "extra" / "ml1m_tables.csv").is_file()
    assert res["registered_pooled"]["E_B"]["complete"] is True
    r = run("ml1m")
    assert r.returncode == 0 and "[skip]" in r.stdout          # resumable
    r = run("toys")                                            # a non-ML-1M panel without the A3-6 record: refused
    assert r.returncode == 4 and "A3-6 item 11" in r.stderr and not (root / "extra" / "toys.json").exists()
    r = run("summarize")
    assert r.returncode == 4
    extra = [f"{rel} = {fe.fx.file_sha1(repo / rel)}" for rel in ("src/confrec/ftgrid_extra.py",
                                                                    "scripts/sigir/run_ftextra.sh")]
    log.write_text(log.read_text(encoding="utf-8") + "\n".join(extra) + "\n", encoding="utf-8")
    r = run("toys")
    assert r.returncode == 0, r.stderr[-2000:]
    assert _strict_load(root / "extra" / "toys.json")["status"]["family_eligible_panel"] is True
    r = run("summarize")
    assert r.returncode == 0, r.stderr[-2000:]
    summ = _strict_load(root / "extra" / "summary.json")
    assert [i["domain"] for i in summ["inputs"]] == ["toys", "ml1m"]
    # a changed bound file (here: a split) is refused by the freeze check
    sp = root / "panels" / "ml1m" / "ftgrid_split.json"
    sp.write_text(sp.read_text(encoding="utf-8") + " ", encoding="utf-8")
    r = run("ml1m", FORCE="1")
    assert r.returncode == 4 and "refused" in r.stderr
    # registered roots are refused in a rehearsal; a non-registered root is refused outside one
    assert run("ml1m", OUT_ROOT="outputs/confrec/ftgrid").returncode == 2
    assert run("ml1m", DRY_RUN="0").returncode == 2
    assert run("nonsense").returncode == 2


def test_summarize_on_real_domain_files_reads_the_registered_blocks(ml1m, tmp_path):
    docs = {}
    for d in fe.AMAZON:                         # the synthetic ML-1M file relabelled as three Qwen Amazon panels
        doc = copy.deepcopy(ml1m.res)
        doc["meta"]["domain"] = d
        doc["status"]["family_eligible_panel"] = True
        docs[d] = doc
    p = _write_docs(tmp_path, {**docs, "ml1m": ml1m.res})
    res = fe.main(["summarize", "--files", p["toys"], p["games"], p["sports"], "--ml1m", p["ml1m"],
                   "--out", str(tmp_path / "summary.json")])
    src = ml1m.res
    pf = src["E_F"]["FT"]["G_LLM_given_CF"]["mean_over_seeds"]["p"]
    assert all(res["families"]["E_F"]["members"][d]["p"] == pf for d in fe.AMAZON)
    assert res["families"]["E_F"]["members"]["toys"]["p_holm"] == pytest.approx(min(1.0, 3 * pf))
    pj = src["E_J"]["FT"]["dUAUC_L_minus_q_hat"]["mean_over_seeds"]["p"]
    assert res["families"]["E_J"]["members"]["games"]["p"] == pj
    sparse = src["E_H"]["ZS"]["strata"]["sparse"]
    assert sparse["n_users_both_classes"] == 0 and "toys:ZS" in res["families"]["E_H"]["outside_family"]
    ed = res["registered"]["E_D"]["ml1m:Qwen3-8B"]
    for reg in ("ZS", "FT"):
        assert ed[reg] == ml1m.rep["E_D"][reg]["holm_family_E_D"]           # copied through unchanged
    assert res["registered"]["E_B"]["members"]["ml1m"]["p"] == ml1m.rep["E_B"]["mean_over_seeds"]["p"]
