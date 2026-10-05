"""src/confrec/ftgrid_extra.py and scripts/sigir/run_ftextra.sh: Amendment 3 addendum 6 items 2-8, its Holm families,
the FT-C / FT-Q reading of addendum 8, the resume rule and the runner's guards.
Synthetic panels, raw ratings and scorer-format runs (the worlds of tests/test_confrec_ftgrid_report.py) and planted
context-level worlds; every estimator is checked against the registered report, a brute-force computation or a planted
truth. The expensive worlds are module fixtures shared by the tests. CPU, deterministic, no torch / transformers."""
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
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

N_BOOT = 40
REPO = Path(fe.__file__).resolve().parents[2]


def _dump(x) -> str:
    return json.dumps(x, sort_keys=True)


def _strict_load(path):
    def bad(x):
        raise ValueError(f"non-strict JSON constant {x}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad)


def strict_json_of(x):
    return json.loads(json.dumps(fe.strict_json(x)))


def record_lines() -> list:
    """The A3-6 item 11 record of the three X1 files, as the pilot log would hold it."""
    return [f"{rel} = {fx.file_sha1(REPO / rel)}" for rel in fe.X1_FILES]


def write_log(path: Path, record: bool = True, extra=()) -> Path:
    path.write_text("\n".join(list(extra) + (record_lines() if record else ["no record"])) + "\n", encoding="utf-8")
    return path


def build_args(w, out, models, n_boot=N_BOOT, refs=None, raw=True, pilot_log=None, root="main"):
    args = ["--domain", w.domain, "--split", str(w.split), "--panels", str(w.d), "--scores_root", str(w.root / "scores"),
            "--models", ",".join(models), "--out", str(out), "--n_boot", str(n_boot), "--root_label", root]
    if raw and w.raw is not None:
        args += ["--raw", str(w.raw)]
    if refs:
        args += ["--refs", str(refs)]
    if pilot_log:
        args += ["--pilot_log", str(pilot_log)]
    return args


def run_extra(w, out_name, models, n_boot=N_BOOT, refs=None, raw=True, pilot_log=None, root="main"):
    out = w.root / "extra" / f"{out_name}.json"
    return fe.main(["build", *build_args(w, out, models, n_boot, refs, raw, pilot_log, root)]), out


@pytest.fixture(scope="module")
def ml1m(tmp_path_factory):
    """The report's synthetic ML-1M world (every arm, FT-C adapters included): the report and this module on it."""
    root = tmp_path_factory.mktemp("ml1m_extra")
    w = make_world(root / "w", "ml1m", seed=0)
    write_models(w, ML1M_SPECS)
    cache = w.root / "report" / "ml1m_refs.json"
    rep, _ = run_main(w, "ml1m", list(ML1M_SPECS), n_boot=N_BOOT, extra=("--refs", str(cache)))
    log = write_log(root / "PILOT_LOG.md")
    res, out = run_extra(w, "ml1m", list(ML1M_SPECS), refs=cache, pilot_log=log)
    return SimpleNamespace(w=w, rep=rep, res=res, out=out, log=log, cache=cache)


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """A small ML-1M world with every arm and model (FT-C included), built once with the record: the resume, crash,
    determinism, record-gate and runner tests copy it."""
    root = tmp_path_factory.mktemp("small_extra")
    w = make_world(root / "w", "ml1m", seed=5, n_eval=120, n_bg=30)
    write_models(w, ML1M_SPECS)
    log = write_log(root / "PILOT_LOG.md")
    res, out = run_extra(w, "ml1m", list(ML1M_SPECS), n_boot=20, pilot_log=log)
    return SimpleNamespace(w=w, root=root, log=log, res=res, out=out, models=list(ML1M_SPECS))


def copy_world(w, dst: Path):
    """A copy of a synthetic world (panels, scores, raw) under dst, with the world's attributes re-pointed."""
    shutil.copytree(w.root, dst)
    return SimpleNamespace(**{**vars(w), "root": dst, "d": dst / "panels" / w.domain,
                              "split": dst / "panels" / w.domain / "ftgrid_split.json",
                              "raw": (dst / "raw") if w.raw is not None else None})


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
                           b=b, e=e, mu=mu, p=p, root_label="main", domain="ml1m")


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
        sh = rp["E_D"][reg]["shares"]["mean_over_seeds"]
        assert sh["item_prior_share"] == rep["E_D"][reg]["shares"]["mean_over_seeds"]["item_prior_share"]
        assert sh["non_prior_share"] == rep["E_D"][reg]["shares"]["mean_over_seeds"]["non_prior_share"]
        # the E-D family that summarize copies through equals the report's (ed_holm_family on the copied block)
        assert _dump(strict_json_of(fr.ed_holm_family(rp["E_D"][reg], reg))) == _dump(rep["E_D"][reg]["holm_family_E_D"])
        assert _dump(ml1m.res["E_G"][reg]["item_share_L_pi"]) == _dump(rep["E_D"][reg]["shares"])   # E-G (i)
    assert _dump(rp["E_B"]) == _dump(rep["E_B"]) and _dump(rp["P1"]) == _dump(rep["P1"])   # P1's inputs and decision
    assert rp["P1"]["decision"]["verdict"] == "P1_HOLDS"
    assert ml1m.res["meta"]["fold_k0_equals_E_D_fold_a"] is True
    assert ml1m.res["references"]["source"] == "cache (--refs)"


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


def test_domain_file_strict_json_blocks_csv_and_readings(ml1m):
    loaded = _strict_load(ml1m.out)
    assert loaded == ml1m.res
    for k in fe.DOC_KEYS + ("excluded_runs", "runs", "references", "readings"):
        assert k in loaded, k
    text = ml1m.out.read_text(encoding="utf-8")

    def keys(x):
        if isinstance(x, dict):
            for k, v in x.items():
                yield k
                yield from keys(v)
        elif isinstance(x, list):
            for v in x:
                yield from keys(v)
    assert not {"p_holm", "holm_family_E_D", "confirmed"} & set(keys(loaded))   # raw p-values only (addendum 1 item 8)
    tmp_name = ml1m.w.root.parent.name                                  # the fixture's unique directory name
    assert tmp_name not in text and json.dumps(str(ml1m.w.root))[1:-1] not in text           # no path in the JSON
    st, meta = loaded["status"], loaded["meta"]
    assert st["items_2_to_7"] == "exploratory" and st["family_eligible_panel"] is False and st["root_label"] == "main"
    assert loaded["input_checks"]["problems"] == [] and loaded["excluded_runs"] == []
    assert loaded["references"]["q_hat_T"]["q_hat_recomputed_equals_reference"] is True
    assert meta["references_source"] == "cache (--refs)" and "T_d" in meta["q_hat_T_source"]
    assert meta["x1_record"]["complete"] is True and meta["fingerprint"]["n_boot"] == N_BOOT
    assert set(meta["code_sha1"]) == {"ftgrid_extra.py", "ftgrid_report.py", "forensics.py", "stats.py", "metrics.py"}
    assert loaded["readings"] == list(fe.READINGS) and len(fe.READINGS) == 19          # the docstring's R1-R19
    assert all(r.startswith(f"R{k + 1} ") for k, r in enumerate(fe.READINGS))
    assert "DEVIATION" in fe.READINGS[12] and "stricter than the registered text" in fe.READINGS[12]   # R13 (MINOR-3)
    for blk in fe.DOC_KEYS[4:]:                                               # never null
        assert isinstance(loaded[blk], dict)
        for reg in ("ZS", "FT"):
            if reg in loaded[blk]:
                assert isinstance(loaded[blk][reg], dict), (blk, reg)
    # the CSV: written first, its sha1 in the JSON; holm_member on the registered E-D / E-B rows of a main-root Qwen
    # panel; ML-1M's A3-6 rows are descriptive and in no family
    csv_path = fe.tables_path(ml1m.out)
    assert fx.file_sha1(csv_path) == meta["tables_csv_sha1"]
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    assert tuple(rows[0]) == fe.CSV_COLS
    assert {"registered_pooled", "E-W", "E-F", "E-G", "E-H", "E-J", "E-C'", "FT-C"} <= {r["block"] for r in rows}
    assert all(r["descriptive"] == "true" and r["holm_member"] == "" for r in rows if r["block"] != "registered_pooled")
    hm = {(r["statistic"], r["regime"], r["model"]): r["holm_member"] for r in rows if r["block"] == "registered_pooled"}
    assert hm[("G", "ZS", "mean_over_seeds")] == "E-D" and hm[("G", "FT", "mean_over_seeds")] == "E-D"
    assert hm[("E_B_dUAUC_FT_minus_ZS", "FT", "mean_over_seeds")] == "E-B" and hm[("G", "FT", "s0")] == ""
    p1wu = [r for r in rows if r["statistic"] == "P1_wu_G_FT_minus_G_ZS" and r["model"] == "mean_over_seeds"][0]
    assert p1wu["note"].startswith("P1_WU_") and "P1_HOLDS" not in p1wu["note"]          # MINOR-1


def test_holm_member_marks_the_a3_6_family_candidates_of_a_qwen_amazon_main_root_panel(ml1m):
    res = copy.deepcopy(ml1m.res)
    res["meta"]["domain"], res["status"]["family_eligible_panel"] = "toys", True
    rows = fe.table_rows(res)
    mark = {(r["block"], r["regime"], r["statistic"], r["model"], r["stratum"]): r["holm_member"] for r in rows}
    assert mark[("E-F", "FT", "G_LLM_given_CF", "mean_over_seeds", "")] == "E-F"
    assert mark[("E-J", "FT", "dUAUC_L_minus_q_hat", "mean_over_seeds", "")] == "E-J"
    assert mark[("E-F", "ZS", "G_LLM_given_CF", "mean_over_seeds", "")] == ""             # the other regime: never
    assert mark[("E-J", "FT", "dUAUC_L_minus_q_hat_T", "mean_over_seeds", "")] == ""      # descriptive
    assert mark[("E-F", "FT", "G_LLM_given_CF", "s0", "")] == ""                          # a seed is not a member
    sparse = [r for r in rows if r["block"] == "E-H" and r["stratum"] == "sparse" and r["statistic"] == "G_prior"]
    assert sparse and all(r["holm_member"] == "" for r in sparse)            # 0 users here: outside by minimum n
    assert "confirmed" not in fe.CSV_COLS                                    # confirmation only in summarize
    res["status"]["root_label"] = "llama"                                    # never under the Llama root
    assert all(r["holm_member"] in ("", "E-F", "E-H", "E-J") for r in fe.table_rows(res))
    assert not any(r["holm_member"] in ("E-D", "E-B") for r in fe.table_rows(res))


def test_ft_c_reading_end_to_end_with_the_q_hat_companion(ml1m):
    ftc, eb, rep = ml1m.res["FT_C_reading"], ml1m.rep["E_B"], ml1m.rep
    assert ftc["available"] is True and ftc["control"] == "FT-C"
    assert ftc["E_B_mean_over_seeds"]["est"] == eb["mean_over_seeds"]["est"]
    assert ftc["defined"] is True and ftc["label"] in ("ITEM_DRIVEN", "USER_DRIVEN", "MIXED")
    u = {m: ftc["UAUC"][m]["est"] for m in fe.FTC_MODELS}
    want = ((u["p0"] - u["zeroshot"]) / (u["s0"] - u["zeroshot"]) + (u["p1"] - u["zeroshot"]) / (u["s1"] - u["zeroshot"])) / 2
    assert ftc["R"]["est"] == pytest.approx(want, abs=1e-12)
    assert ftc["label"] == fe.ftc_label(ftc["R"])
    # addendum 8's companion: the UAUC of q-hat on the same rows (every TEST row is finite here: E-A's rows)
    assert ftc["UAUC_q_hat"]["est"] == pytest.approx(rep["E_A"]["ZS"]["UAUC_TEST"]["references"]["q_hat"]["est"], abs=1e-12)
    assert ftc["UAUC_q_hat"]["n_users"] == ftc["rows"]["n_users"]


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
    users = P["user"]
    Rr = test & np.isfinite(Ls0)
    ok = Rr & np.isfinite(qT)
    a, b = bf_user_auc(Ls0, P["label"], users, ok), bf_user_auc(qT, P["label"], users, ok)
    assert res["E_J"]["FT"]["dUAUC_L_minus_q_hat_T"]["per_seed"]["s0"]["est"] == pytest.approx(
        np.mean([a[u] - b[u] for u in a]), abs=1e-12)
    assert rep["E_A"]["FT"]["UAUC_TEST"]["rows"]["n_users"] == len(a)
    assert rep["E_A"]["FT"]["UAUC_TEST"]["rows"]["n_pairs"] == int(np.isin(users[Rr], list(a)).sum())
    warm = res["E_J"]["FT"]["dUAUC_L_minus_MF_warm"]
    assert warm["descriptive"] is True and warm["rows"]["n_pairs"] <= res["E_J"]["FT"]["rows_E_A"]["n_pairs"]


# ---------------------------------------------------------------- determinism, resume, crash, record gate
def test_build_is_deterministic_byte_for_byte(small, tmp_path):
    out2 = tmp_path / "again" / "ml1m.json"
    fe.main(["build", *build_args(small.w, out2, small.models, 20, pilot_log=small.log)])
    assert out2.read_bytes() == small.out.read_bytes()
    assert fe.tables_path(out2).read_bytes() == fe.tables_path(small.out).read_bytes()


def _resume(args) -> tuple:
    return fe.check_resume(fe.parse_args(["check_resume", *args]))


def test_check_resume_follows_the_stored_fingerprint_and_every_input(small, tmp_path):
    w = copy_world(small.w, tmp_path / "w")
    log = write_log(tmp_path / "PILOT_LOG.md")
    out = tmp_path / "extra" / "ml1m.json"
    args = build_args(w, out, small.models, 20, pilot_log=log)
    assert _resume(args) == (False, "no output yet")
    fe.main(["build", *args])
    ok, why = _resume(args)
    assert ok, why
    # a rehearsal with another n_boot, a MODELS=zeroshot file, another root or seed: never resumed
    for alt, key in ((build_args(w, out, small.models, 30, pilot_log=log), "n_boot"),
                     (build_args(w, out, ["zeroshot"], 20, pilot_log=log), "models"),
                     (build_args(w, out, small.models, 20, pilot_log=log, root="teacher"), "root_label"),
                     (args + ["--seed", "1"], "seed")):
        ok, why = _resume(alt)
        assert not ok and key in why, (key, why)
    # every input file, the record and the code identity: one changed byte -> rebuild (MAJOR-1 b, c)
    S = w.root / "scores" / "ml1m"
    for path in (w.split, w.d / "train.jsonl", w.d / "sd_users.txt", w.d / "eval_sd_test.jsonl", w.d / "eval.jsonl",
                 w.d / "eval_sd_test_starperm0.jsonl", S / "s1" / "swap" / "swap_prior.csv.gz",
                 S / "s2" / "starperm1" / "scores.csv.gz", S / "zeroshot" / "like" / "run.key",
                 S / "p0" / "like" / "report.json", w.raw / "ml-1m" / "ratings.dat", w.raw / "ml-1m" / "movies.dat"):
        orig = path.read_bytes()
        path.write_bytes(orig + b" ")
        ok, why = _resume(args)
        assert not ok, path
        path.write_bytes(orig)
    extra_file = S / "s0" / "like" / "FAILED_INTEGRITY"                  # a new file in a run directory
    extra_file.write_text("x", encoding="utf-8")
    assert _resume(args)[0] is False
    extra_file.unlink()
    log.write_text(log.read_text(encoding="utf-8") + "a later, unrelated pilot-log entry\n", encoding="utf-8")
    assert _resume(args)[0] is True                       # the log grows all the time: only the record status counts
    write_log(log, record=False)                          # the record status changed: rebuild
    ok, why = _resume(args)
    assert not ok and "x1_record" in why
    write_log(log)
    assert _resume(args)[0] is True                                      # everything restored: current again
    # the stored code identity is compared too
    doc = json.loads(out.read_text(encoding="utf-8"))
    doc["meta"]["fingerprint"]["code_sha1"]["ftgrid_extra.py"] = "0" * 40
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    ok, why = _resume(args)
    assert not ok and "code_sha1" in why
    # a CSV that is not the one written with the JSON
    fe.main(["build", *args])
    tp = fe.tables_path(out)
    tp.write_text(tp.read_text(encoding="utf-8") + "x\n", encoding="utf-8")
    ok, why = _resume(args)
    assert not ok and "CSV" in why


def test_a_crash_between_the_csv_and_the_json_never_leaves_a_stale_pair(small, tmp_path, monkeypatch):
    out = tmp_path / "extra" / "ml1m.json"
    args = build_args(small.w, out, small.models, 20, pilot_log=small.log)
    fe.main(["build", *args])
    old_csv = fe.tables_path(out).read_bytes()
    real = fe._atomic_write

    def crash_on_json(path, text):
        if Path(path).suffix == ".json":
            raise OSError("simulated crash before the JSON")
        return real(path, text)
    monkeypatch.setattr(fe, "_atomic_write", crash_on_json)
    with pytest.raises(OSError):
        fe.main(["build", *build_args(small.w, out, ["zeroshot", "s0", "s1", "s2"], 20, pilot_log=small.log)])
    assert not out.exists()                                       # the old JSON was removed before the CSV was written
    assert fe.tables_path(out).read_bytes() != old_csv            # the new CSV is there, with no JSON beside it
    monkeypatch.setattr(fe, "_atomic_write", real)
    assert _resume(args) == (False, "no output yet")              # so the next run rebuilds


def test_ft_c_needs_the_record_and_nothing_else_changes_without_it(small, tmp_path):
    out = tmp_path / "norec" / "ml1m.json"
    res = fe.main(["build", *build_args(small.w, out, small.models, 20,
                                        pilot_log=write_log(tmp_path / "LOG.md", record=False))])
    ftc = res["FT_C_reading"]
    assert ftc["available"] is False and ftc["reason"].startswith("record missing") and ftc["label"] == "NOT_DEFINED"
    for rel in fe.X1_FILES:
        assert rel in ftc["reason"]
    assert small.res["FT_C_reading"]["available"] is True                # the same world with the record
    a, b = copy.deepcopy(res), copy.deepcopy(small.res)
    for d in (a, b):
        d.pop("FT_C_reading")
        for k in ("x1_record", "fingerprint", "tables_csv_sha1"):
            d["meta"].pop(k)
    assert _dump(a) == _dump(b)
    no_log = fe.main(["build", *build_args(small.w, tmp_path / "nolog" / "ml1m.json", small.models, 20)])
    assert no_log["FT_C_reading"]["available"] is False and no_log["meta"]["x1_record"]["pilot_log_given"] is False


def test_a_q_hat_that_differs_from_the_reference_is_a_problem_and_summarize_refuses_it(small, tmp_path):
    cache = tmp_path / "refs.json"
    fe.main(["build", *build_args(small.w, tmp_path / "a" / "ml1m.json", small.models, 20, refs=cache,
                                  pilot_log=small.log)])
    obj = json.loads(cache.read_text(encoding="utf-8"))
    k = next(j for j, v in enumerate(obj["arrays"]["q_hat"]) if v is not None)
    obj["arrays"]["q_hat"][k] += 1e-6                                    # same identity, a different array
    cache.write_text(json.dumps(obj), encoding="utf-8")
    out = tmp_path / "b" / "ml1m.json"
    res = fe.main(["build", *build_args(small.w, out, small.models, 20, refs=cache, pilot_log=small.log)])
    assert res["references"]["source"] == "cache (--refs)"
    assert any("q-hat recomputed" in p for p in res["input_checks"]["problems"])
    cur = fe.code_sha1()["ftgrid_extra.py"]
    assert fe.load_checked(str(small.out), "ml1m", 20, cur)["meta"]["domain"] == "ml1m"     # the clean file passes
    with pytest.raises(SystemExit) as e:
        fe.load_checked(str(out), "ml1m", 20, cur)                        # the file with the problem is refused
    assert e.value.code == 2


# ---------------------------------------------------------------- E-W
def test_e_w_removes_a_planted_user_offset_bias_and_keeps_a_planted_personal_signal():
    # user offsets track leniency, no personal signal: the pooled stacker's G is negative, the within-user G is ~0
    X = ctx_world(kappa=1.0, gamma=0.0)
    cx, R = X.cx, np.ones(X.cx.n, bool)
    q, mf = X.refs["arrays"]["q_hat"], X.refs["arrays"]["mf_residual"]
    pooled = fr.stacker_block(cx, X.models, X.L, X.PI, q, mf, R, X.n_boot, 0, 1)
    folds = fe.split_folds(cx)
    assert len(folds) == fe.K_SPLITS == 20 and np.array_equal(folds[0], cx.fold_a)
    rp = {"E_D": {"ZS": {"information_gain": pooled}}}
    ew, ef, _ = fe.wu_regime(X, "ZS", info_of(X), folds, {}, rp)
    g, gw = pooled["G_mean_over_seeds"], ew["G_wu"]["mean_over_seeds"]
    assert g["est"] < -0.015 and g["hi"] < 0                          # the reviewer's bias of the pooled estimator
    assert abs(gw["est"]) < 0.002 and gw["lo"] < 0 < gw["hi"]           # removed by the within-user estimator
    assert ew["reading_G"]["reading"] == "estimator_dependent" and ew["reading_G"]["robust"] is False
    # robust uses the unadjusted interval; the registered confirmation is beside it and set by summarize (MINOR-2)
    assert ew["reading_G"]["registered_confirmed"] is None and "UNADJUSTED" in ew["reading_G"]["description"]
    assert "summarize" in ew["reading_G"]["registered_confirmed_source"]
    # a planted personal signal is kept, by both estimators: robust
    X = ctx_world(kappa=1.0, gamma=1.0, seed=1)
    cx = X.cx
    pooled = fr.stacker_block(cx, X.models, X.L, X.PI, X.refs["arrays"]["q_hat"], X.refs["arrays"]["mf_residual"],
                              np.ones(cx.n, bool), X.n_boot, 0, 1)
    ew, _, _ = fe.wu_regime(X, "ZS", info_of(X), fe.split_folds(cx), {}, {"E_D": {"ZS": {"information_gain": pooled}}})
    gw = ew["G_wu"]["mean_over_seeds"]
    assert gw["est"] > 0.05 and gw["lo"] > 0 and gw["p"] < 0.05 and gw["descriptive_min_n"] is False
    assert ew["reading_G"]["reading"] == "robust" and ew["reading_G"]["robust"] is True
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
    pb = paired_bootstrap(gwu, {u: 0.0 for u in gwu}, n_boot=X.n_boot, seed=0)
    assert (ew["G_wu"]["mean_over_seeds"]["lo"], ew["G_wu"]["mean_over_seeds"]["hi"]) == pytest.approx(
        (pb["lo"], pb["hi"]), abs=1e-9)
    draws = fr.mean_draws(np.array([[gwu[u]] for u in users]), X.n_boot, 0)[:, 0]
    assert ew["G_wu"]["mean_over_seeds"]["p"] == pytest.approx(fr.boot_p(draws))
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
    X2 = copy.copy(X)
    X2.models = X.models[:2]
    ew2, _, _ = fe.wu_regime(X2, "FT", {"models": X.models[:2], "missing_or_excluded": ["s2"], "n_registered": 3},
                             fe.split_folds(cx), {}, {})
    assert ew2["complete"] is False and ew2["G_wu"]["seeds"]["complete"] is False
    assert ew2["G_wu"]["seeds"]["sigma_seed_rule"] is False and ew2["missing_or_excluded"] == ["s2"]
    S = ctx_world(n_users=120, kappa=1.0, gamma=1.0, seed=5)
    ews, _, _ = fe.wu_regime(S, "ZS", info_of(S), fe.split_folds(S.cx), {}, {"E_D": {"ZS": {"information_gain": {
        "G_mean_over_seeds": {"est": 0.05, "lo": 0.01, "hi": 0.09, "p": None, "ci_excludes_0": None,
                              "descriptive_min_n": True}}}}})
    assert ews["G_wu"]["mean_over_seeds"]["descriptive_min_n"] is True and ews["G_wu"]["mean_over_seeds"]["p"] is None
    assert ews["reading_G"]["reading"] == "descriptive_min_n"
    # P1_wu: p1_block's rows and rule on the within-user G values, under its own verdict names (MINOR-1)
    Z = ctx_world(n_users=400, kappa=1.0, gamma=0.2, n_models=1, seed=6)
    F = ctx_world(n_users=400, kappa=1.0, gamma=1.0, n_models=3, seed=6)       # same users, items and labels (seed)
    P = SimpleNamespace(cx=F.cx, L={"zeroshot": Z.L["zeroshot"], **F.L}, PI={"zeroshot": Z.PI["zeroshot"], **F.PI},
                        refs=F.refs, present={"zeroshot", "s0", "s1", "s2"}, n_boot=100, seed=0, root_label="main")
    assert np.array_equal(Z.cx.y, F.cx.y)
    rp = {"P1": {"available": True, "role": "P1 confirmatory (ML-1M, Qwen3-8B)", "decision": {"verdict": "P1_HOLDS"},
                 "mean_over_seeds": {"est": 0.04, "lo": 0.02, "hi": 0.06, "p": 0.001, "ci_excludes_0": True,
                                     "descriptive_min_n": False},
                 "per_seed": {s: {"est": 0.04} for s in ("s0", "s1", "s2")}}}
    p1w = fe.p1_wu_block(P, fe.split_folds(F.cx), rp)
    assert p1w["available"] is True and p1w["decision"]["verdict"] == "P1_WU_HOLDS"
    assert "wording" not in p1w["decision"] and "P1_HOLDS" not in json.dumps(p1w["decision"])
    per = [p1w["per_seed"][s]["est"] for s in ("s0", "s1", "s2")]
    assert p1w["mean_over_seeds"]["est"] == pytest.approx(np.mean(per))
    assert p1w["per_seed"]["s0"]["est"] == pytest.approx(p1w["per_seed"]["s0"]["G_FT"] - p1w["G_ZS"])
    assert p1w["reading_P1"]["reading"] == "robust" and p1w["reading_P1"]["registered_confirmed"] is True
    lam = fe.p1_wu_block(SimpleNamespace(**{**vars(P), "root_label": "llama"}), fe.split_folds(F.cx), rp)
    assert lam["reading_P1"]["registered_confirmed"] is None                 # a replication confirms nothing
    miss = fe.p1_wu_block(SimpleNamespace(**{**vars(P), "present": {"zeroshot", "s0", "s1"}}), fe.split_folds(F.cx), {})
    assert miss["available"] is False and miss["decision"]["verdict"] == "P1_WU_INCOMPLETE"
    assert set(fe.P1_WU_VERDICT.values()) == {"P1_WU_HOLDS", "P1_WU_DOES_NOT_HOLD", "P1_WU_INCOMPLETE"}


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
    assert two["robust"] is False
    na = fe.robust_reading(None, r(0.01, 0.002, 0.02), registered_confirmed=True, confirmed_source="x")
    assert na["reading"] == "not_available" and na["registered_confirmed"] is True and "description" in na


# ---------------------------------------------------------------- E-F
def test_e_f_contrasts_are_on_identical_rows_and_separate_llm_from_cf_evidence():
    X = ctx_world(n_users=500, kappa=1.0, gamma=0.0, mf_gamma=1.2, seed=7)
    ew, ef, _ = fe.wu_regime(X, "ZS", info_of(X), fe.split_folds(X.cx), {}, {})
    assert ef["rows"] == ew["rows"]                                          # identical users and rows for M2, M3, M4
    llm, cf = ef["G_LLM_given_CF"]["mean_over_seeds"], ef["G_CF_given_LLM"]["mean_over_seeds"]
    assert cf["est"] > 0.05 and cf["lo"] > 0 and abs(llm["est"]) < 0.01
    assert llm["n_users"] == cf["n_users"] == ew["G_wu"]["mean_over_seeds"]["n_users"]
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
    assert st["unseen"]["G_prior"]["mean_over_seeds"]["p"] is None
    assert st["sparse"]["n_users_both_classes"] >= fr.MIN_N and st["sparse"]["G_prior"]["mean_over_seeds"]["p"] is not None
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
    assert st["sparse"]["G_prior"]["mean_over_seeds"]["est"] == pytest.approx(
        np.mean([np.mean(v) for v in acc.values()]), abs=1e-12)
    assert st["sparse"]["n_users_both_classes"] == len(acc)
    lv = bf_user_auc(X.L[m], cx.y, cx.users, rows)
    assert st["sparse"]["UAUC_L"]["mean_over_seeds"]["est"] == pytest.approx(np.mean(list(lv.values())), abs=1e-12)
    qv = bf_user_auc(X.refs["arrays"]["q_hat"], cx.y, cx.users, rows)
    assert st["sparse"]["UAUC_q_hat"]["est"] == pytest.approx(np.mean(list(qv.values())), abs=1e-12)
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
    assert m["non_prior_share"] == pytest.approx(truth, abs=0.03)
    assert m["var_e_over_var_L_uncorrected"] > m["e_share"]["est"]
    users, rows, cols, truth = _eshare_world(0.5, seed=12)
    out = fe.eshare_block(users, rows, {"m": cols}, 200, 0, 1)
    m = out["per_model"]["m"]
    assert m["e_share"]["est"] == pytest.approx(truth, abs=0.02)
    assert m["non_prior_share"] < truth - 0.05                              # A3-6 item 1: below the e-share
    sb = fr.shares_block(users, rows, {"m": cols}, 200, 0, 1)
    assert sb["per_model"]["m"]["non_prior_share"]["est"] == pytest.approx(m["non_prior_share"])
    assert out["rows"]["n_pairs"] == sb["rows"]["n_pairs"]


# ---------------------------------------------------------------- E-C'
def test_cal_rate_k_rounds_half_up_falls_back_and_clips():
    cal_y = {"a": [1, 1, 0, 0], "b": [1, 0, 0, 0], "c": [1, 1], "d": [0, 0, 0, 0], "e": [1, 1, 1, 1]}
    test_y = {"a": [1, 0, 0], "b": [1, 0], "c": [0, 1, 0, 0, 1], "d": [1, 0, 0], "e": [1, 0, 1]}
    users, y, cal = [], [], []
    for u in cal_y:
        users += [u] * (len(cal_y[u]) + len(test_y[u]))
        y += cal_y[u] + test_y[u]
        cal += [True] * len(cal_y[u]) + [False] * len(test_y[u])
    users, y, cal = np.array(users), np.array(y), np.array(cal)
    uc = np.unique(users, return_inverse=True)[1].reshape(-1)
    k_row, info = fe.cal_rate_k(SimpleNamespace(uc=uc, y=y, cal=cal), ~cal)
    k = {u: int(k_row[np.flatnonzero((users == u) & ~cal)[0]]) for u in cal_y}
    assert info["pooled_cal_like_rate"] == pytest.approx(9 / 18) and info["n_users_pooled_rate_fallback"] == 1
    assert k["a"] == 2                     # 0.5 x 3 = 1.5 -> half up 2
    assert k["b"] == 1                     # 0.25 x 2 = 0.5 -> half up 1
    assert k["c"] == 3                     # pooled 0.5 x 5 = 2.5 -> half up 3 (banker's rounding would give 2)
    assert k["d"] == 1 and k["e"] == 2     # 0 -> clip 1; 1.0 x 3 = 3 -> clip n_u - 1 = 2
    assert np.all(k_row[cal] == -1)
    rng = np.random.default_rng(13)
    L = rng.normal(0, 1, len(y))
    rows = ~cal
    oracle = np.zeros(len(y), int)
    for u in cal_y:
        idx = np.flatnonzero((users == u) & rows)
        oracle[idx] = y[idx].sum()
    for a_, b_ in zip(fe.topk_anatomy_at(L, y, users, rows, oracle), fr.topk_anatomy(L, y, users, rows)):
        assert np.array_equal(np.isnan(a_), np.isnan(b_)) and np.allclose(a_[~np.isnan(a_)], b_[~np.isnan(b_)], 0, 0)


# ---------------------------------------------------------------- FT-C / FT-Q reading
def _ftc_world(keep: str, seed=14, n_users=400, eb_positive=True, root="main", domain="ml1m", q=False):
    rng = np.random.default_rng(seed)
    n = n_users * 10
    users = np.repeat([f"u{k:04d}" for k in range(n_users)], 10)
    e = rng.normal(0, 1, n)
    y = (e + rng.normal(0, 1, n) > 0).astype(int)
    zs = 0.2 * e + rng.normal(0, 1, n)
    L = {"zeroshot": zs}
    for s in ("s0", "s1", "s2"):
        L[s] = (1.2 * e if eb_positive else -0.5 * e) + rng.normal(0, 1, n)
    for s in ("p0", "p1"):
        L[s] = {"all": L["s" + s[1]].copy(), "none": zs.copy(), "noise": zs + rng.normal(0, 0.01, n)}[keep]
    uc = np.unique(users, return_inverse=True)[1].reshape(-1)
    cx = SimpleNamespace(test=np.ones(n, bool), y=y, uc=uc, users=users)
    eb = {"complete": True, **fr.contrast_models({s: (L[s], zs) for s in ("s0", "s1", "s2")}, y, uc, cx.test, 200, 0,
                                                 n_registered=3)}
    refs = {"arrays": {"q_hat": 0.5 * e + rng.normal(0, 1, n)}} if q else None
    return SimpleNamespace(cx=cx, L=L, present=set(L), n_boot=200, seed=0, refs=refs, root_label=root, domain=domain,
                           x1={"complete": True, "recorded": {r: True for r in fe.X1_FILES}}), eb


def test_ft_c_retention_on_planted_worlds_the_label_rule_and_the_scope():
    X, eb = _ftc_world("all", q=True)                          # the permuted adapter keeps all of the gain
    out = fe.ftc_reading(X, eb)
    assert out["defined"] is True and out["R"]["est"] == pytest.approx(1.0) and out["control"] == "FT-C"
    assert out["R"]["lo"] == pytest.approx(1.0) and out["label"] == "ITEM_DRIVEN"
    qv = fr.user_aucs(X.refs["arrays"]["q_hat"], X.cx.y, X.cx.uc)
    assert out["UAUC_q_hat"]["est"] == pytest.approx(np.mean(list(qv.values())), abs=1e-12)   # addendum 8 companion
    X, eb = _ftc_world("none")                                  # ... and none of it
    out = fe.ftc_reading(X, eb)
    assert out["R"]["est"] == pytest.approx(0.0, abs=1e-12) and out["label"] == "USER_DRIVEN"
    assert out["UAUC_q_hat"]["available"] is False
    X, eb = _ftc_world("noise")                                 # one resample drives all UAUCs
    out = fe.ftc_reading(X, eb)
    pu = {m: fr.user_aucs(X.L[m], X.cx.y, X.cx.uc) for m in fe.FTC_MODELS}
    V = np.array([[pu[m][u] for m in fe.FTC_MODELS] for u in sorted(pu["zeroshot"])])
    D = fr.mean_draws(V, X.n_boot, 0)
    r = ((D[:, 3] - D[:, 0]) / (D[:, 1] - D[:, 0]) + (D[:, 4] - D[:, 0]) / (D[:, 2] - D[:, 0])) / 2
    assert (out["R"]["lo"], out["R"]["hi"]) == pytest.approx(tuple(np.quantile(r, [0.025, 0.975])), abs=1e-12)
    assert out["UAUC"]["p0"]["est"] == pytest.approx(np.mean(V[:, 3]))
    X, eb = _ftc_world("all", eb_positive=False)                # E-B not positive: R not defined, not reported
    out = fe.ftc_reading(X, eb)
    assert out["defined"] is False and out["R"]["available"] is False and out["label"] == "NOT_DEFINED"
    assert "E_B_mean_gt_0" in out["R"]["reason"] and "est" not in out["R"]
    X, eb = _ftc_world("all")                                   # E-B incomplete: not defined (R13, stricter)
    out = fe.ftc_reading(X, {**eb, "complete": False})
    assert out["label"] == "NOT_DEFINED" and out["E_B_condition"]["E_B_complete"] is False
    X.present.discard("p1")
    out = fe.ftc_reading(X, eb)
    assert out["available"] is False and "p1" in out["reason"] and out["label"] == "NOT_DEFINED"
    # scope (addendum 8): FT-Q on the teacher root, nothing on Llama, FT-C on the main root's ML-1M only
    X, eb = _ftc_world("all", root="teacher", domain="toys")
    t = fe.ftc_reading(X, eb)
    assert t["control"] == "FT-Q" and t["label"] == "ITEM_DRIVEN" and "FT-Q" in t["status"]
    X, eb = _ftc_world("all", root="main", domain="toys")
    t = fe.ftc_reading(X, eb)
    assert t["available"] is False and "withdrew" in t["reason"]
    X, eb = _ftc_world("all", root="llama", domain="ml1m")
    assert fe.ftc_reading(X, eb)["available"] is False
    X, eb = _ftc_world("all")
    X.x1 = {"complete": False, "recorded": {r: r != fe.X1_FILES[2] for r in fe.X1_FILES}}
    t = fe.ftc_reading(X, eb)                                   # the record gate comes first (MINOR-5)
    assert t["available"] is False and t["reason"].startswith("record missing") and fe.X1_FILES[2] in t["reason"]
    lab = fe.ftc_label
    assert lab({"lo": 0.6, "hi": 0.9}) == "ITEM_DRIVEN" and lab({"lo": 0.1, "hi": 0.4}) == "USER_DRIVEN"
    assert lab({"lo": 0.4, "hi": 0.6}) == "MIXED" and lab({"lo": 0.5, "hi": 0.9}) == "MIXED"
    assert lab({"lo": 0.1, "hi": 0.5}) == "MIXED" and lab({"lo": None, "hi": 0.4}) == "NOT_DEFINED"


# ---------------------------------------------------------------- integrity, missing seeds, missing references
def test_failed_integrity_run_is_excluded_and_the_regime_is_incomplete(tmp_path):
    w = make_world(tmp_path, "ml1m", seed=13, n_eval=200, n_bg=40)
    n_pairs = sum(len(r["candidate_item_ids"]) for r in w.eval_rows)
    specs = {"zeroshot": {"beta": 0.3}, "s0": {"beta": 1.0}, "s1": {"beta": 1.0}, "s2": {"beta": 1.0}}
    write_models(w, specs, arms=("like", "swap"), cens2={("s2", "like"): tuple(range(0, n_pairs, 20))})   # 5%
    res, _ = run_extra(w, "bad", list(specs), n_boot=20, pilot_log=write_log(tmp_path / "LOG.md"))
    assert [(e["model"], e["arm"], e["status"]) for e in res["excluded_runs"]] == [("s2", "like", "FAILED_INTEGRITY")]
    assert res["runs"]["s2"]["like"]["integrity"]["E1"] is False
    for blk in ("E_W", "E_F", "E_J", "E_Cprime"):
        assert res[blk]["FT"]["models"] == ["s0", "s1"] and res[blk]["FT"]["complete"] is False, blk
    assert res["E_W"]["FT"]["G_wu"]["seeds"]["complete"] is False
    assert res["E_W"]["FT"]["G_wu"]["seeds"]["sigma_seed_rule"] is False
    assert res["E_F"]["FT"]["G_LLM_given_CF"]["seeds"]["sigma_seed_rule"] is False
    assert res["E_J"]["FT"]["dUAUC_L_minus_q_hat"]["seeds"]["complete"] is False
    assert res["E_W"]["P1_wu"]["decision"]["verdict"] == "P1_WU_INCOMPLETE"
    assert res["registered_pooled"]["P1"]["decision"]["verdict"] == "INCOMPLETE"
    assert res["registered_pooled"]["E_D"]["FT"]["star_permutation"]["available"] is False
    assert res["FT_C_reading"]["available"] is False and res["FT_C_reading"]["label"] == "NOT_DEFINED"
    assert res["references"]["source"].startswith("computed from --raw")


def test_without_references_every_dependent_block_is_unavailable_never_an_exception(tmp_path):
    w = make_world(tmp_path, "toys", seed=12, n_eval=150, n_bg=30, raw=False)
    specs = {m: {"beta": 0.5} for m in ("zeroshot", "s0", "s1", "s2")}
    write_models(w, specs, arms=("like", "swap"))
    res, out = run_extra(w, "toys", list(specs), n_boot=10, raw=False)
    for blk in ("E_W", "E_F", "E_H"):
        assert res[blk]["ZS"]["available"] is False, blk
    assert res["E_J"]["available"] is False and res["E_G"]["MF_score_item_bias"]["available"] is False
    assert res["E_G"]["FT"]["e_share"]["mean_over_seeds"]["n_users"] >= 0
    assert res["E_Cprime"]["FT"]["models"] == ["s0", "s1", "s2"]
    assert res["references"]["q_hat_T"]["available"] is False
    assert res["status"]["family_eligible_panel"] is True and res["status"]["items_2_to_7"] == "registered_outcome_free"
    assert res["FT_C_reading"]["available"] is False                    # no record and not the ML-1M control
    assert res["input_checks"]["problems"] == []                        # no --raw: nothing was asked of the references
    _strict_load(out)
    empty = tmp_path / "empty_raw"                                      # --raw given, references not computable (MINOR-1)
    empty.mkdir()
    res = fe.main(["build", *build_args(w, tmp_path / "r2" / "toys.json", list(specs), 10, raw=False),
                   "--raw", str(empty)])
    assert any(p.startswith("the CF references could not be computed") for p in res["input_checks"]["problems"])
    with pytest.raises(SystemExit) as e:                                # so summarize refuses the file
        fe.load_checked(str(tmp_path / "r2" / "toys.json"), "files", 10, fe.code_sha1()["ftgrid_extra.py"])
    assert e.value.code == 2


def test_root_label_must_match_the_registered_root_of_the_scores(small, tmp_path):
    out = tmp_path / "x" / "ml1m.json"
    for rel, label in (("outputs/confrec/ftgrid_q/scores", "main"), ("outputs/confrec/ftgrid/scores", "teacher"),
                       ("outputs/confrec/ftgrid_llama/scores", "main"), ("outputs/confrec/ftgrid/scores", "llama")):
        args = build_args(small.w, out, small.models, 20)
        args[args.index("--scores_root") + 1] = str(REPO / rel)
        args[args.index("--root_label") + 1] = label
        for cmd in ("check_resume", "build"):                           # refused before anything is read or written
            with pytest.raises(SystemExit) as e:
                fe.main([cmd, *args])
            assert e.value.code == 2, (rel, label, cmd)
    assert not out.exists() and not out.parent.exists()
    assert fe.root_label_of(REPO / "outputs" / "confrec" / "ftgrid_q" / "scores") == "teacher"
    assert fe.root_label_of(REPO / "outputs" / "confrec" / "ftgrid" / "scores" / "ml1m") == "main"
    assert fe.root_label_of(REPO / "outputs" / "confrec" / "ftgrid" / ".." / "ftgrid_llama" / "scores") == "llama"
    assert fe.root_label_of(small.w.root / "scores") is None             # a rehearsal: any label
    fe.check_root_label(SimpleNamespace(scores_root=str(REPO / "outputs/confrec/ftgrid_q/scores"), root_label="teacher"))
    link = tmp_path / "link"                                             # a symlinked spelling resolves into the root
    try:
        os.symlink(REPO / "outputs" / "confrec" / "ftgrid_q", link, target_is_directory=True)
    except (OSError, NotImplementedError):
        link = None
    if link is not None and link.is_symlink():
        assert fe.root_label_of(link / "scores") == "teacher"


# ---------------------------------------------------------------- summarize: the A3-6 families and its refusals
def _stat(est, p, n=400, desc=False):
    return {"est": est, "lo": est - 0.01, "hi": est + 0.01, "p": None if desc else p, "n_users": n,
            "descriptive_min_n": desc}


def _seeds(rule=True, complete=True):
    return {"sigma_seed_rule": rule, "complete": complete, "per_seed": [0.01, 0.011, 0.012], "sigma_seed": 0.001}


def _runs(ft: str = "ok", models=("zeroshot", "s0", "s1", "s2", "p0", "p1")) -> dict:
    """The run record of a domain file: every arm OK, or the FT models ABSENT (not run) / FAILED_INTEGRITY (run)."""
    st = {"ok": "OK", "absent": "ABSENT", "failed": "FAILED_INTEGRITY"}[ft]
    return {m: {arm: {"status": st if m in fe.FT_MODELS else "OK"} for arm in fe.ARMS} for m in models}


def _doc(domain, backbone="Qwen3-8B", root="main", ef=(0.02, 0.01, True), eh_zs=(0.02, 0.01, 400),
         eh_ft=(0.02, 0.01, 400, True), ej=(-0.03, 0.002, True), eb=(0.1, 0.001, True), n_boot=2000, label=None,
         control=None, ft="ok", models=("zeroshot", "s0", "s1", "s2", "p0", "p1")):
    """A fabricated domain file of the schema summarize reads (ft='absent': the FT runs were not run)."""
    if ft != "ok":                                  # no FT run usable: every FT statistic is unavailable
        ef = None
        eh_ft = None
        ej = None
    ef_blk = {"mean_over_seeds": _stat(ef[0], ef[1]), "seeds": _seeds(ef[2])} if ef is not None else None
    ftc = ({"available": True, "control": control, "label": label} if label else
           {"available": False, "reason": "no p0/p1", "label": "NOT_DEFINED"})
    na = {"available": False, "reason": "no usable like arm for the FT models"}
    return {"spec": "s", "meta": {"domain": domain, "backbone": backbone, "root_label": root, "n_boot": n_boot,
                                  "seed": 0, "code_sha1": fe.code_sha1(), "models_requested": list(models)},
            "runs": _runs(ft, models),
            "status": {"family_eligible_panel": root == "main" and domain in fe.AMAZON and "qwen" in backbone.lower()},
            "input_checks": {"problems": []},
            "registered_pooled": {
                "E_D": {reg: {"information_gain": {"G_mean_over_seeds": {"p": 0.01}, "seeds": {"sigma_seed_rule": True}},
                              "star_permutation": {"dUAUC_mean_over_seeds": {"p": 0.03},
                                                   "seeds": {"sigma_seed_rule": False}}} for reg in ("ZS", "FT")},
                "E_B": {"complete": eb[2], "mean_over_seeds": _stat(eb[0], eb[1]), "seeds": _seeds(True, eb[2])},
                "P1": {"available": True, "role": "P1 confirmatory (ML-1M, Qwen3-8B)" if "qwen" in backbone.lower()
                       else "replication of P1's sign (Amendment 2 F)", "decision": {"verdict": "P1_HOLDS"},
                       "mean_over_seeds": _stat(0.01, 0.004)}},
            "E_W": {"P1_wu": {"decision": {"verdict": "P1_WU_HOLDS"}, "mean_over_seeds": _stat(0.01, 0.004),
                              "reading_P1": {"reading": "robust", "registered_confirmed": True}},
                    **{reg: {"reading_G": {"reading": "robust", "robust": True, "registered_confirmed": None},
                             "reading_G_CF": {"reading": "robust"}} for reg in ("ZS", "FT")}},
            "E_F": {"ZS": {"G_LLM_given_CF": {"mean_over_seeds": _stat(0.05, 0.0005), "seeds": _seeds()}},
                    "FT": ({"G_LLM_given_CF": ef_blk} if ef_blk else na)},
            "E_G": {}, "E_Cprime": {}, "FT_C_reading": ftc,
            "E_H": {"ZS": {"strata": {"sparse": {"G_prior": {"mean_over_seeds": _stat(eh_zs[0], eh_zs[1], eh_zs[2],
                                                                                      eh_zs[2] < 150),
                                                             "seeds": _seeds()}}}},
                    "FT": ({"strata": {"sparse": {"G_prior": {"mean_over_seeds": _stat(eh_ft[0], eh_ft[1], eh_ft[2],
                                                                                       eh_ft[2] < 150),
                                                              "seeds": _seeds(eh_ft[3])}}}} if eh_ft else na)},
            "E_J": {"ZS": {"dUAUC_L_minus_q_hat": {"mean_over_seeds": _stat(0.04, 0.001), "seeds": _seeds()}},
                    "FT": ({"dUAUC_L_minus_q_hat": {"mean_over_seeds": _stat(ej[0], ej[1]), "seeds": _seeds(ej[2])}}
                           if ej else na)}}


def _write_docs(tmp_path, docs: dict):
    paths = {}
    for name, d in docs.items():
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        paths[name] = str(p)
    return paths


def _summ(args, out):
    return fe.main(["summarize", *args, "--out", str(out)])


def test_summarize_holm_families_sigma_seed_rule_direction_and_missing_members(tmp_path):
    docs = {"toys": _doc("toys", ef=(0.02, 0.01, True), eh_zs=(0.02, 0.01, 400), eh_ft=(0.03, 0.02, 400, True),
                         ej=(-0.03, 0.002, True)),
            "games": _doc("games", ef=(0.02, 0.02, False), eh_zs=(0.02, 0.001, 120), eh_ft=(0.03, 0.001, 400, True),
                          ej=(0.03, 0.06, True)),
            "sports": _doc("sports", ef=(-0.03, 0.001, True), eh_zs=(-0.02, 0.001, 400), eh_ft=(0.03, 0.03, 400, False),
                           ej=(0.02, 0.001, False)),
            "ml1m": _doc("ml1m", ef=(0.05, 0.0001, True)),
            "llama_toys": _doc("toys", backbone="Llama-3.1-8B-Instruct", root="llama"),
            "llama_ml1m": _doc("ml1m", backbone="Llama-3.1-8B-Instruct", root="llama")}
    p = _write_docs(tmp_path, docs)
    out = tmp_path / "summary.json"
    res = _summ(["--files", p["toys"], p["games"], p["sports"], "--ml1m", p["ml1m"], "--llama", p["llama_toys"],
                 p["llama_ml1m"]], out)
    _strict_load(out)
    F = res["families"]["E_F"]
    assert set(F["members"]) == {"toys", "games", "sports"} and F["m"] == 3        # never ML-1M, Llama or ZS
    adj = fr.holm({"toys": 0.01, "games": 0.02, "sports": 0.001})
    assert all(F["members"][d]["p_holm"] == pytest.approx(adj[d]) for d in adj)
    assert F["members"]["toys"]["confirmed"] is True
    assert F["members"]["games"]["confirmed"] is False                          # the sigma_seed rule fails
    assert F["members"]["sports"]["confirmed"] is False and F["members"]["sports"]["direction_ok"] is False
    H = res["families"]["E_H"]
    assert "games:ZS" in H["outside_family"] and "descriptive" in H["outside_family"]["games:ZS"]["reason"]
    assert set(H["members"]) == {"toys:ZS", "toys:FT", "games:FT", "sports:ZS", "sports:FT"} and H["m"] == 5
    adj = fr.holm({"toys:ZS": 0.01, "toys:FT": 0.02, "games:FT": 0.001, "sports:ZS": 0.001, "sports:FT": 0.03})
    assert all(H["members"][k]["p_holm"] == pytest.approx(adj[k]) for k in adj)
    assert H["members"]["toys:ZS"]["confirmed"] is True and H["members"]["sports:ZS"]["confirmed"] is False
    assert H["members"]["sports:FT"]["confirmed"] is False and H["members"]["games:FT"]["confirmed"] is True
    J = res["families"]["E_J"]
    assert J["members"]["toys"]["confirmed"] is True and J["members"]["toys"]["sign"] == -1   # two-sided: as found
    assert J["members"]["games"]["confirmed"] is False and J["members"]["games"]["p_holm"] == pytest.approx(0.06)
    assert J["members"]["sports"]["confirmed"] is False
    eb = res["registered"]["E_B"]                                                   # complete: decided
    assert eb["complete_family"] is True and set(eb["members"]) == {"ml1m", "toys", "games", "sports"}
    assert all(v["confirmed"] is True for v in eb["members"].values())
    assert res["registered"]["E_D"]["toys:Qwen3-8B"]["FT"] == strict_json_of(
        fr.ed_holm_family(docs["toys"]["registered_pooled"]["E_D"]["FT"], "FT"))
    assert "toys:Llama-3.1-8B-Instruct" not in res["registered"]["E_D"]                # Llama apart (MINOR-2)
    assert res["registered"]["P1"]["decision"]["verdict"] == "P1_HOLDS"
    assert "P1_wu" not in json.dumps(res["registered"]) and res["sensitivity"]["P1_wu"]["decision"]["verdict"] == "P1_WU_HOLDS"
    assert res["llama"]["ml1m"]["P1_replication"]["role"].startswith("replication")
    assert res["llama"]["toys"]["P1_replication"]["available"] is False
    rob = res["robust_readings"]
    assert rob["toys:FT"]["G"]["reading"] == "robust" and rob["toys:FT"]["G"]["registered_confirmed"] is True
    assert rob["ml1m:P1"]["registered_confirmed"] is True
    assert res["ft_wording"]["available"] is False                              # not requested
    # FT runs that were run but excluded (E1): the members stay in their families with p = 1 (m never shrinks)
    docs["sports"] = _doc("sports", ft="failed")
    p = _write_docs(tmp_path, docs)
    res = _summ(["--files", p["toys"], p["games"], p["sports"]], tmp_path / "s2.json")
    F = res["families"]["E_F"]
    assert F["m"] == 3 and F["n_missing_members"] == 1 and F["members"]["sports"]["confirmed"] is False
    assert F["members"]["toys"]["p_holm"] == pytest.approx(fr.holm({"toys": 0.01, "games": 0.02, "sports": 1.0})["toys"])
    assert res["families"]["E_J"]["m"] == 3 and res["families"]["E_H"]["members"]["sports:FT"]["missing"] is True
    eb = res["registered"]["E_B"]                                                   # no ML-1M: not decided (MAJOR-3)
    assert eb["complete_family"] is False and "ml1m: file not given" in eb["reason"]
    assert all(v["p_holm"] is None and v["confirmed"] is None for v in eb["members"].values())
    assert res["registered"]["P1"]["available"] is False


def test_a_panel_whose_ft_runs_were_cut_needs_not_run_and_keeps_only_its_zs_member(tmp_path, capsys):
    docs = {"toys": _doc("toys", ef=(0.02, 0.01, True), ej=(-0.03, 0.002, True)),
            "games": _doc("games", ef=(0.03, 0.03, True), ej=(0.04, 0.02, True)),
            "sports": _doc("sports", ft="absent"), "sports_failed": _doc("sports", ft="failed"),
            "toys_no_ft": _doc("toys", models=("zeroshot", "p0", "p1"))}
    p = _write_docs(tmp_path, docs)
    base = ["--files", p["toys"], p["games"], p["sports"]]
    log = tmp_path / "PILOT_LOG.md"

    def refused(args, *needles):
        with pytest.raises(SystemExit) as e:
            _summ(args, tmp_path / "r.json")
        err = capsys.readouterr().err
        assert e.value.code == 2 and all(n in err for n in needles), err
    refused(base, "--not_run sports", "FTEXTRA_NOT_RUN sports")                # a cut panel is never kept with p = 1
    refused(base + ["--not_run", "sports"], "FTEXTRA_NOT_RUN sports")          # the flag needs the pilot-log record
    log.write_text("2026-10-29 cut: FTEXTRA_NOT_RUN sports_extra; FTEXTRA_NOT_RUN sportsX\n", encoding="utf-8")
    refused(base + ["--not_run", "sports", "--pilot_log", str(log)], "FTEXTRA_NOT_RUN sports")   # near misses
    log.write_text("2026-10-29 checkpoint: Sports FT cut (FTEXTRA_NOT_RUN sports).\n", encoding="utf-8")
    assert fe.not_run_recorded(log, "sports") and not fe.not_run_recorded(log, "games")
    res = _summ(base + ["--not_run", "sports", "--pilot_log", str(log)], tmp_path / "s.json")
    F, H, J = (res["families"][k] for k in ("E_F", "E_H", "E_J"))
    assert F["m"] == 2 and set(F["members"]) == {"toys", "games"} and F["n_missing_members"] == 0
    assert J["m"] == 2 and set(J["members"]) == {"toys", "games"}
    assert H["m"] == 5 and set(H["members"]) == {"toys:ZS", "toys:FT", "games:ZS", "games:FT", "sports:ZS"}
    for fam, ps in ((F, {"toys": 0.01, "games": 0.03}), (J, {"toys": 0.002, "games": 0.02})):
        adj = fr.holm(ps)
        assert all(fam["members"][d]["p_holm"] == pytest.approx(adj[d]) for d in ps)   # the registered m, the right Holm
    assert set(res["not_run"]) == {"sports"} and "E_H:sports:ZS" in res["not_run"]["sports"]["kept"][0]
    assert "sports:FT" not in H["outside_family"] and "sports" not in F["outside_family"]   # not members at all
    refused(["--files", p["toys"], p["games"], "--not_run", "games", "--pilot_log", str(log)],
            "FTEXTRA_NOT_RUN games")                                             # games' cut is not recorded
    log.write_text(log.read_text(encoding="utf-8") + "FTEXTRA_NOT_RUN games\n", encoding="utf-8")
    refused(["--files", p["toys"], p["games"], "--not_run", "games", "--pilot_log", str(log)],
            "--not_run games", "holds fine-tuned runs")                          # games was run: no cut
    refused(["--files", p["toys"], "--not_run", "sports", "--pilot_log", str(log)], "no --files file of sports")
    res = _summ(["--files", p["toys"], p["games"], p["sports_failed"]], tmp_path / "f.json")   # run, then excluded
    assert res["families"]["E_F"]["m"] == 3 and res["families"]["E_F"]["members"]["sports"]["missing"] is True
    refused(["--files", p["sports_failed"], "--not_run", "sports", "--pilot_log", str(log)], "holds fine-tuned runs")
    refused(["--files", p["toys_no_ft"]], "models_requested lacks", "s0")       # the registered models of the root


def test_eb_copy_is_decided_only_on_a_complete_family(tmp_path):
    docs = {d: _doc(d) for d in ("toys", "games", "sports", "ml1m")}
    docs["games"] = _doc("games", eb=(0.1, 0.001, False))                         # a seed missing in Video_Games
    p = _write_docs(tmp_path, docs)
    res = _summ(["--files", p["toys"], p["games"], p["sports"], "--ml1m", p["ml1m"]], tmp_path / "s.json")
    eb = res["registered"]["E_B"]
    assert eb["complete_family"] is False and "games: E-B incomplete" in eb["reason"] and eb["m"] is None
    assert all(v["p_holm"] is None and v["confirmed"] is None for v in eb["members"].values())
    assert eb["members"]["toys"]["p"] == 0.001                                    # the raw values stay visible


def test_summarize_refuses_every_bad_input_instead_of_dropping_it(tmp_path):
    good = {d: _doc(d) for d in ("toys", "games", "sports", "ml1m")}
    bad = {"ml1m_as_file": _doc("ml1m"), "llama_as_file": _doc("toys", backbone="Llama-3.1-8B-Instruct", root="llama"),
           "no_backbone": {**_doc("toys"), "meta": {**_doc("toys")["meta"], "backbone": None}},
           "odd_backbone": {**_doc("toys"), "meta": {**_doc("toys")["meta"], "backbone": "Mistral-7B"}},
           "problems": {**_doc("toys"), "input_checks": {"problems": ["a split mismatch"]}},
           "n_boot": _doc("toys", n_boot=60), "seed": {**_doc("toys"), "meta": {**_doc("toys")["meta"], "seed": 1}},
           "old_code": {**_doc("toys"), "meta": {**_doc("toys")["meta"], "code_sha1": {"ftgrid_extra.py": "0" * 40}}},
           "teacher_as_file": _doc("toys", root="teacher"), "not_eligible": {**_doc("toys"), "status": {
               "family_eligible_panel": False}}, "schema": {"meta": {}}, "qwen_as_llama": _doc("ml1m"),
           "ftc_other": _doc("ml1m", label="ITEM_DRIVEN", control="FT-C")}
    p = _write_docs(tmp_path, {**good, **bad})
    base = ["--files", p["toys"], p["games"], p["sports"]]
    cases = [["--files", p["ml1m_as_file"]], ["--files", p["llama_as_file"]], ["--files", p["no_backbone"]],
             ["--files", p["odd_backbone"]], ["--files", p["problems"]], ["--files", p["n_boot"]],
             ["--files", p["seed"]], ["--files", p["old_code"]], ["--files", p["teacher_as_file"]],
             ["--files", p["not_eligible"]], ["--files", p["schema"]], base + ["--ml1m", p["toys"]],
             base + ["--ml1m", p["llama_as_file"]], base + ["--llama", p["qwen_as_llama"]],
             base + ["--ftq", p["ml1m"]], ["--files", p["toys"], p["toys"]],
             base + ["--ml1m", p["ml1m"], "--ftc", p["ftc_other"]]]
    for k, args in enumerate(cases):
        with pytest.raises(SystemExit) as e:
            _summ(args, tmp_path / f"r{k}.json")
        assert e.value.code == 2, args
        assert not (tmp_path / f"r{k}.json").exists()
    res = _summ(base + ["--ml1m", p["ml1m"], "--ftc", p["ml1m"]], tmp_path / "ok.json")   # the same file twice is fine
    assert [i["role"] for i in res["inputs"]] == ["files", "files", "files", "ml1m", "ftc"]
    res = _summ(["--files", p["n_boot"], "--n_boot", "60"], tmp_path / "ok60.json")       # a rehearsal's own n_boot
    assert res["n_boot"] == 60


def test_ft_wording_follows_addendum_8_and_is_never_decided_on_partial_input(tmp_path):
    files = {d: _doc(d) for d in ("toys", "games", "sports")}
    ftc = {"ITEM": _doc("ml1m", label="ITEM_DRIVEN", control="FT-C"), "NONE": _doc("ml1m"),
           "MIXED": _doc("ml1m", label="MIXED", control="FT-C")}
    q = {(d, lab): _doc(d, root="teacher", label=lab, control="FT-Q") for d in ("ml1m", "toys", "games")
         for lab in ("ITEM_DRIVEN", "USER_DRIVEN", "NOT_DEFINED")}
    paths = _write_docs(tmp_path, {**files, **{f"ftc_{k}": v for k, v in ftc.items()},
                                   **{f"q_{d}_{lab}": v for (d, lab), v in q.items()}})
    base = ["--files", paths["toys"], paths["games"], paths["sports"]]

    def wording(ftc_key, q_keys):
        args = base + (["--ftc", paths[f"ftc_{ftc_key}"]] if ftc_key else [])
        if q_keys:
            args += ["--ftq"] + [paths[f"q_{d}_{lab}"] for d, lab in q_keys]
        return _summ(args, tmp_path / "w.json")["ft_wording"]
    w = wording("ITEM", [("ml1m", "ITEM_DRIVEN"), ("toys", "ITEM_DRIVEN")])
    assert w["fine_tuning_mostly_teaches_the_item"] is True and w["item_quality_from_item_text"] is True
    assert w["labels"] == {"FT-C": {"ml1m": "ITEM_DRIVEN"}, "FT-Q": {"ml1m": "ITEM_DRIVEN", "toys": "ITEM_DRIVEN"}}
    assert w["evidence_mixed"] is False and w["complete"] is True
    w = wording("ITEM", [("ml1m", "ITEM_DRIVEN"), ("toys", "ITEM_DRIVEN"), ("games", "USER_DRIVEN")])
    assert w["fine_tuning_mostly_teaches_the_item"] is False and w["evidence_mixed"] is True   # every dataset run
    w = wording("MIXED", [("ml1m", "ITEM_DRIVEN"), ("toys", "ITEM_DRIVEN")])
    assert w["fine_tuning_mostly_teaches_the_item"] is False and w["evidence_mixed"] is True
    w = wording("ITEM", [("ml1m", "ITEM_DRIVEN"), ("toys", "NOT_DEFINED")])
    assert w["fine_tuning_mostly_teaches_the_item"] is False and w["evidence_mixed"] is False  # defined, not ITEM_DRIVEN
    w = wording("ITEM", [("ml1m", "ITEM_DRIVEN")])                                # Toys FT-Q missing: never decided
    assert w["fine_tuning_mostly_teaches_the_item"] is None and "toys" in w["reason"]
    w = wording(None, [("ml1m", "ITEM_DRIVEN"), ("toys", "ITEM_DRIVEN")])          # FT-C missing
    assert w["fine_tuning_mostly_teaches_the_item"] is None and "FT-C" in w["reason"]
    w = wording("NONE", [("ml1m", "ITEM_DRIVEN"), ("toys", "ITEM_DRIVEN")])        # FT-C block unavailable
    assert w["fine_tuning_mostly_teaches_the_item"] is None and w["item_quality_from_item_text"] is None


def test_summarize_on_real_domain_files_reads_the_registered_blocks(ml1m, tmp_path):
    docs = {}
    for d in fe.AMAZON:                         # the synthetic ML-1M file relabelled as three Qwen Amazon panels
        doc = copy.deepcopy(ml1m.res)
        doc["meta"]["domain"] = d
        doc["status"]["family_eligible_panel"] = True
        docs[d] = doc
    p = _write_docs(tmp_path, {**docs, "ml1m": ml1m.res})
    res = _summ(["--files", p["toys"], p["games"], p["sports"], "--ml1m", p["ml1m"], "--ftc", p["ml1m"],
                 "--n_boot", str(N_BOOT)], tmp_path / "summary.json")
    src = ml1m.res
    pf = src["E_F"]["FT"]["G_LLM_given_CF"]["mean_over_seeds"]["p"]
    assert all(res["families"]["E_F"]["members"][d]["p"] == pf for d in fe.AMAZON)
    assert res["families"]["E_F"]["members"]["toys"]["p_holm"] == pytest.approx(min(1.0, 3 * pf))
    pj = src["E_J"]["FT"]["dUAUC_L_minus_q_hat"]["mean_over_seeds"]["p"]
    assert res["families"]["E_J"]["members"]["games"]["p"] == pj
    assert src["E_H"]["ZS"]["strata"]["sparse"]["n_users_both_classes"] == 0
    assert "toys:ZS" in res["families"]["E_H"]["outside_family"]
    for reg in ("ZS", "FT"):
        assert res["registered"]["E_D"]["ml1m:Qwen3-8B"][reg] == ml1m.rep["E_D"][reg]["holm_family_E_D"]
        assert res["robust_readings"][f"ml1m:{reg}"]["G"]["registered_confirmed"] == \
            ml1m.rep["E_D"][reg]["holm_family_E_D"]["confirmed"]["G"]
    assert res["registered"]["E_B"]["members"]["ml1m"]["p"] == ml1m.rep["E_B"]["mean_over_seeds"]["p"]
    assert res["registered"]["P1"]["decision"] == ml1m.rep["P1"]["decision"]
    w = res["ft_wording"]
    assert w["labels"]["FT-C"]["ml1m"] == src["FT_C_reading"]["label"] and w["fine_tuning_mostly_teaches_the_item"] is None


# ---------------------------------------------------------------- the runner (bash)
def _posix_bash():
    if os.name == "nt":                     # the PATH bash of Windows is WSL's; use Git Bash when it exists
        for p in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
            if Path(p).is_file():
                return p
        return None
    return shutil.which("bash")


SCRIPT = (REPO / "scripts" / "sigir" / "run_ftextra.sh").as_posix()


def _run(arg, env, **kw):
    return subprocess.run([_posix_bash(), SCRIPT, arg], env={**os.environ, **env, **kw}, capture_output=True, text=True,
                          timeout=900)


@pytest.mark.skipif(_posix_bash() is None, reason="no POSIX bash")
def test_runner_refuses_the_registered_roots_in_any_spelling(tmp_path):
    reg = (REPO / "outputs" / "confrec" / "ftgrid").as_posix()
    env = {"DRY_RUN": "1", "PILOT_LOG": "x", "PYTHON": Path(sys.executable).as_posix()}
    spellings = ["outputs/confrec/ftgrid", "outputs//confrec/ftgrid", "./outputs/confrec/ftgrid/", reg,
                 "outputs/confrec/./ftgrid", "outputs/confrec/ftgrid/../ftgrid_llama", "outputs/confrec/ftgrid_q",
                 "outputs/confrec/ftgrid_dryrun/x", "scripts/../outputs/confrec/ftgrid"]
    if os.name == "nt":                     # a case-insensitive filesystem and backslash / drive-letter spellings
        spellings += ["OUTPUTS/Confrec/FTGRID", str(REPO / "outputs" / "confrec" / "ftgrid"), reg.upper()]
    for s in spellings:
        r = _run("ml1m", env, OUT_ROOT=s)
        assert r.returncode == 2 and "never touches" in r.stderr, (s, r.stderr)
    ok_root = (tmp_path / "ok").as_posix()                                    # the inputs are guarded too (MINOR-3)
    for raw in ("outputs/confrec/ftgrid/raw", reg + "/../ftgrid_llama/raw"):
        r = _run("toys", env, OUT_ROOT=ok_root, RAW=raw)
        assert r.returncode == 2 and "never touches" in r.stderr, (raw, r.stderr)
    link = tmp_path / "link"                                                  # a symlink into the registered root
    try:
        os.symlink(REPO / "outputs" / "confrec", link, target_is_directory=True)
    except (OSError, NotImplementedError):
        link = None
    if link is not None and link.is_symlink():
        r = _run("ml1m", env, OUT_ROOT=(link / "ftgrid").as_posix())
        assert r.returncode == 2 and "never touches" in r.stderr
        for sub in ("extra", "panels", "scores"):                             # an output, panel or score link inside
            root = tmp_path / f"root_{sub}"
            root.mkdir()
            os.symlink(REPO / "outputs" / "confrec" / "ftgrid" / sub, root / sub, target_is_directory=True)
            r = _run("ml1m", env, OUT_ROOT=root.as_posix())
            assert r.returncode == 2 and "never touches" in r.stderr, (sub, r.stderr)
        root = tmp_path / "root_domain"
        (root / "panels").mkdir(parents=True)
        os.symlink(REPO / "outputs" / "confrec" / "ftgrid" / "panels" / "ml1m", root / "panels" / "ml1m",
                   target_is_directory=True)
        r = _run("ml1m", env, OUT_ROOT=root.as_posix())                        # panels/D itself is a link inside
        assert r.returncode == 2 and "never touches" in r.stderr
        r = _run("ml1m", env, OUT_ROOT=ok_root, RAW=(link / "ftgrid" / "raw").as_posix())
        assert r.returncode == 2 and "never touches" in r.stderr
    r = _run("ml1m", {"PYTHON": Path(sys.executable).as_posix()}, OUT_ROOT=(tmp_path / "nowhere").as_posix(),
             DRY_RUN="0")
    assert r.returncode == 2 and "not a registered root" in r.stderr
    assert _run("nonsense", env).returncode == 2
    assert not (REPO / "outputs" / "confrec" / "ftgrid" / "extra" / "ml1m.json.tmp").exists()


def _freezable(split: Path, ml1m: bool) -> None:
    """A synthetic split as a stage-0 split looks (tokenizer audit present; ML-1M T checked against Gate-FT)."""
    js = json.loads(split.read_text(encoding="utf-8"))
    js["train"]["overlength"] = {"share_above_1024": 0.0}
    if ml1m:
        js["gateft_T_match"] = True
    split.write_text(json.dumps(js, indent=2), encoding="utf-8")


@pytest.mark.skipif(_posix_bash() is None, reason="no POSIX bash")
def test_runner_dry_run_freeze_record_resume_and_summarize(small, tmp_path):
    from src.confrec import ftgrid_freeze as ff
    root = tmp_path / "dry"
    copy_world(small.w, root)                                           # panels/ml1m, scores/ml1m, raw
    shutil.rmtree(root / "extra", ignore_errors=True)
    for sub in ("panels", "scores"):           # a 'toys' panel: the ML-1M world relabelled (its references are computable)
        shutil.copytree(root / sub / "ml1m", root / sub / "toys")
    for d in ("ml1m", "toys"):
        _freezable(root / "panels" / d / "ftgrid_split.json", True)
    log = tmp_path / "PILOT_LOG.md"
    core = ff.lines_for("core", REPO, [str(root / "panels" / d / "ftgrid_split.json") for d in ("ml1m", "toys")])
    log.write_text("\n".join(core) + "\n", encoding="utf-8")
    env = {"DRY_RUN": "1", "OUT_ROOT": root.as_posix(), "PILOT_LOG": log.as_posix(), "N_BOOT": "20",
           "PYTHON": Path(sys.executable).as_posix()}
    out = root / "extra" / "ml1m.json"
    r = _run("ml1m", env)                         # ML-1M runs without the record; its FT_C_reading does not (MINOR-5)
    assert r.returncode == 0, r.stderr[-2000:]
    res = _strict_load(out)
    assert res["FT_C_reading"]["available"] is False and res["FT_C_reading"]["reason"].startswith("record missing")
    assert res["meta"]["models_requested"] == ["zeroshot", "s0", "s1", "s2", "p0", "p1"]
    r = _run("ml1m", env)
    assert r.returncode == 0 and "[skip]" in r.stdout                            # current: resumed
    r = _run("ml1m", env, MODELS="zeroshot")                    # a partial file (MAJOR-1 b; n_boot, seed, every input
    assert r.returncode == 0 and "[skip]" not in r.stdout       # and the CSV pairing: the check_resume test)
    r = _run("ml1m", env)                                                        # the zeroshot-only file is rebuilt
    assert r.returncode == 0 and "models" in r.stdout and len(_strict_load(out)["meta"]["models_requested"]) == 6
    r = _run("toys", env)                                                        # non-ML-1M without the record: refused
    assert r.returncode == 4 and "A3-6 item 11" in r.stderr and not (root / "extra" / "toys.json").exists()
    assert _run("summarize", env).returncode == 4
    lines = record_lines()
    log.write_text(log.read_text(encoding="utf-8") + "\n".join(lines[:2]) + "\n", encoding="utf-8")
    r = _run("toys", env)                                                        # the test file's sha1 is required too
    assert r.returncode == 4 and fe.X1_FILES[2] in r.stderr
    log.write_text(log.read_text(encoding="utf-8") + lines[2] + "\n", encoding="utf-8")
    r = _run("ml1m", env)                                                        # the record changed: rebuilt, FT-C now
    assert r.returncode == 0 and "x1_record" in r.stdout
    assert _strict_load(out)["FT_C_reading"]["available"] is True
    r = _run("toys", env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert _strict_load(root / "extra" / "toys.json")["status"]["family_eligible_panel"] is True
    assert _strict_load(root / "extra" / "toys.json")["input_checks"]["problems"] == []
    assert "[skip]" in _run("toys", env).stdout and "[skip]" not in _run("toys", env, FORCE="1").stdout   # FORCE=1
    rep = root / "scores" / "toys" / "s0" / "like" / "report.json"            # a stale input (MINOR-2): refused, named
    orig = rep.read_bytes()
    rep.write_bytes(orig + b" ")
    r = _run("summarize", env)
    assert r.returncode == 2 and "toys.json" in r.stderr and "not a current build" in r.stderr
    rep.write_bytes(orig)
    base_log = log.read_text(encoding="utf-8")                                 # a cut recorded for a panel whose FT
    log.write_text(base_log + "FTEXTRA_NOT_RUN toys\n", encoding="utf-8")      # runs exist: passed on and refused
    r = _run("summarize", env)
    assert r.returncode == 2 and "--not_run toys" in r.stderr
    log.write_text(base_log, encoding="utf-8")
    r = _run("summarize", env)
    assert r.returncode == 0, r.stderr[-2000:]
    summ = _strict_load(root / "extra" / "summary.json")
    assert [i["domain"] for i in summ["inputs"]] == ["toys", "ml1m", "ml1m"] and summ["not_run"] == {}
    assert summ["registered"]["E_B"]["complete_family"] is False                 # Video_Games and Sports not given
    sp = root / "panels" / "ml1m" / "ftgrid_split.json"
    sp.write_text(sp.read_text(encoding="utf-8") + " ", encoding="utf-8")
    r = _run("ml1m", env, FORCE="1")
    assert r.returncode == 4 and "refused" in r.stderr
    assert _run("summarize", env, ROOT_LABEL="llama").returncode == 2
