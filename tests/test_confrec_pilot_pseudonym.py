import csv
import gzip
import json
import math
import subprocess
import sys

import numpy as np

from src.confrec.pilot_pseudonym import (ARMS, analyze, apply_gate, decide, head_minus_tail, load_arm, token_gate,
                                         treated_from_panel)
from src.confrec.stats import rank_bins

COLS = ["source_event_id", "user_id", "item_id", "cand_idx", "label", "question", "lp_yes", "lp_no", "logit",
        "yes_no_mass", "censored"]


def _fixture(tmp, scenario, n_users=150, n_c=20, seed=0, censor=False, flags_in_panel=True, prior=False):
    """Three scored arms + REAL panel with known Δ = L_real - L_arm. Items have distinct popularity; 80% treated."""
    rng = np.random.default_rng(seed)
    n_items = 1500
    ipop = rng.permutation(np.arange(1, n_items + 1)) * 3 + rng.integers(0, 3, n_items)
    itr = rng.uniform(size=n_items) < 0.8
    panel, recs = [], []
    for u in range(n_users):
        its = rng.choice(n_items, n_c, replace=False)
        y = rng.permutation([1] * (n_c // 2) + [0] * (n_c - n_c // 2))
        brands = [f"Store{i % 40:02d}" for i in its]
        row = {"user_id": f"u{u}", "source_event_id": f"u{u}::0", "candidate_item_ids": [f"i{i}" for i in its],
               "candidate_titles": [f"{b} Widget {i}" if itr[i] else f"Widget {i}" for b, i in zip(brands, its)],
               "candidate_brands": brands, "candidate_popularity": [int(ipop[i]) for i in its],
               "candidate_labels": [int(v) for v in y]}
        if flags_in_panel:
            row["candidate_brand_in_title"] = [bool(itr[i]) for i in its]
        if prior:  # strictly monotone in the all-time popularity
            row["candidate_popularity_prior"] = [int(ipop[i] // 3) for i in its]
        panel.append(row)
        recs += [(u, c, int(i), int(y[c])) for c, i in enumerate(its)]
    u_, c_, i_, y_ = map(np.array, zip(*recs))
    N = len(recs)
    pop, tr = ipop[i_].astype(float), itr[i_]
    head = np.zeros(N, bool)
    head[np.flatnonzero(tr)[rank_bins(pop[tr], 5) == 4]] = True

    def e(s):
        return rng.normal(0, s, N)
    pref = 1.2 * (2 * y_ - 1) + e(1.0)
    L = {"dislike": None}
    if scenario == "positive":      # head-only familiarity knock-out, placebo flat
        L["real"] = pref
        L["pseudo"], L["placebo"] = pref - 0.5 * head - e(.02), pref - e(.02)
    elif scenario == "null":        # pseudonyms destroy the evidence everywhere
        L["real"] = pref
        L["pseudo"], L["placebo"] = 0.2 * pref + e(1.5), pref - e(.02)
    elif scenario == "negative":    # uniform perturbation shift, identical for pseudo and placebo
        L["real"] = pref
        L["pseudo"], L["placebo"] = pref - 0.3 - e(.05), pref - 0.3 - e(.05)
    elif scenario == "ambiguous":   # popularity-linked but below the 0.20 bar
        L["real"] = pref
        L["pseudo"], L["placebo"] = pref - 0.12 * head - e(.02), pref - e(.02)
    elif scenario == "cross":       # acquiescence a(i) and prior pi(i) track popularity only under real names
        z = np.log1p(pop) - np.log1p(pop).mean()
        L["real"], L["placebo"] = pref + 0.8 * z, pref + 0.8 * z
        L["pseudo"] = pref + e(.02)
        L["dislike"] = {"real": -pref + 0.8 * z + e(.3), "pseudo": -pref + e(.3)}
    dirs = {}
    for a in ARMS:
        d = tmp / a
        d.mkdir()
        cens = np.zeros(N, int)
        if censor:
            cens[:5] = {"real": 1, "pseudo": 0, "placebo": 0}[a]
            cens[5:10] = {"real": 0, "pseudo": 2, "placebo": 0}[a]
            cens[10:15] = {"real": 0, "pseudo": 0, "placebo": 3}[a]
        with gzip.open(d / "scores.csv.gz", "wt", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            qs = [("like", L[a])] + ([("dislike", L["dislike"][a])] if L["dislike"] and a in L["dislike"] else [])
            for q, v in qs:
                for j in range(N):
                    lg = "nan" if cens[j] in (2, 3) else f"{v[j]:.6f}"
                    w.writerow([f"u{u_[j]}::0", f"u{u_[j]}", f"i{i_[j]}", c_[j], y_[j], q, 0, 0, lg, 1, cens[j]])
        if scenario == "cross" and a in ("real", "pseudo"):
            items = np.unique(i_)
            zi = np.log1p(ipop[items]) - np.log1p(ipop[items]).mean()
            with gzip.open(d / "swap_prior.csv.gz", "wt", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["item_id", "donor_user_id", "question", "logit", "censored"])
                for i, zz in zip(items, zi):
                    for k in range(4):
                        v = (0.8 * zz if a == "real" else 0.0) + rng.normal(0, .2)
                        w.writerow([f"i{i}", f"d{k}", "like", f"{v:.6f}", 0])
        (d / "report.json").write_text("{}")
        dirs[a] = d
    pp = tmp / "toys_rated.jsonl"
    pp.write_text("\n".join(json.dumps(r) for r in panel), encoding="utf-8")
    return dirs, pp, panel


def _run(tmp, scenario, n_boot=200, **kw):
    dirs, pp, panel = _fixture(tmp, scenario, **kw)
    arms = {a: load_arm(dirs[a]) for a in ARMS}
    flags, _, _ = treated_from_panel(pp, panel)
    return analyze(arms, panel, flags, n_boot=n_boot, seed=0, bi_boot=10)


def test_positive_fixture(tmp_path):
    res = _run(tmp_path, "positive")
    hp, hl = res["delta"]["pseudo"]["head_minus_tail"], res["delta"]["placebo"]["head_minus_tail"]
    assert abs(hp["est"] - 0.5) < 0.01 and hp["lo"] > 0.45 and abs(hl["est"]) < 0.01
    assert abs(res["dUAUC"]["real_minus_pseudo"]["tail"]["est"]) <= 0.01
    assert res["delta"]["pseudo"]["slope_logpop"]["lo"] > 0
    assert sum(b["n"] for b in res["delta"]["by_decile"]) == res["treated"]["n_treated"]
    assert all(b["n"] > 0 for b in res["delta"]["by_decile"])
    assert res["uauc"]["real"]["all"]["n_rows"] == res["rows"]["n_analysed"] > res["treated"]["n_treated"]
    assert res["decision"]["verdict"] == "POSITIVE"
    assert set(res["bias_index"]["real"]) == {"head", "mid", "tail"}
    assert res["cross_check"]["corr_a_logpop"]["available"] is False


def test_null_fixture(tmp_path):
    res = _run(tmp_path, "null")
    d = res["dUAUC"]["real_minus_pseudo"]
    assert d["all"]["est"] > 0.02 and d["tail"]["est"] > 0.02 and d["all"]["lo"] > 0.02
    assert res["decision"]["verdict"] == "NULL"


def test_negative_fixture(tmp_path):
    res = _run(tmp_path, "negative")
    sl = res["delta"]["pseudo"]["slope_logpop"]
    assert sl["lo"] <= 0 <= sl["hi"]
    assert abs(res["delta"]["pseudo"]["mean"]["est"] - 0.3) < 0.01
    assert res["decision"]["verdict"] == "NEGATIVE"


def test_ambiguous_fixture(tmp_path):
    res = _run(tmp_path, "ambiguous")
    assert abs(res["delta"]["pseudo"]["head_minus_tail"]["est"] - 0.12) < 0.01
    assert res["decision"]["verdict"] == "AMBIGUOUS"


def test_decide_branches_and_order():
    ci = lambda est, lo, hi: {"est": est, "lo": lo, "hi": hi}
    flat = ci(0.0, -0.01, 0.01)
    assert decide(0.03, 0.03, ci(0.5, 0.4, 0.6), flat, ci(0.1, 0.05, 0.15))["verdict"] == "NULL"   # NULL first
    assert decide(0.03, 0.0, ci(0.5, 0.4, 0.6), flat, ci(0.1, 0.05, 0.15))["verdict"] == "POSITIVE"
    assert decide(0.0, 0.0, ci(0.2, 0.01, 0.4), ci(0.05, 0, .1), flat)["verdict"] == "POSITIVE"  # boundaries inclusive
    assert decide(0.0, 0.0, ci(0.5, -0.1, 0.9), flat, flat)["verdict"] == "AMBIGUOUS"   # lo <= 0, gap 0.5
    assert decide(0.0, 0.011, ci(0.5, 0.4, 0.6), flat, flat)["verdict"] == "AMBIGUOUS"  # tail UAUC moved
    assert decide(0.0, 0.0, ci(0.5, 0.4, 0.6), ci(0.06, 0, .1), flat)["verdict"] == "AMBIGUOUS"
    assert decide(0.0, 0.0, ci(0.03, -0.01, 0.07), ci(0.0, -.02, .02), flat)["verdict"] == "NEGATIVE"
    assert decide(0.0, 0.0, ci(0.03, -0.01, 0.07), ci(0.0, -.02, .02), ci(.1, .05, .2))["verdict"] == "AMBIGUOUS"
    nan = float("nan")
    out = decide(nan, nan, ci(nan, nan, nan), ci(nan, nan, nan), ci(nan, nan, nan))   # used to read AMBIGUOUS
    assert out["verdict"] == "UNDETERMINED" and all(v is None for v in out["conditions"].values())
    assert len(out["missing_inputs"]) == 7
    # an empty tail: NULL is already False overall, POSITIVE False on hmt, NEGATIVE decidable -> NEGATIVE
    assert decide(0.0, nan, ci(0.03, -0.01, 0.07), ci(0.0, -.02, .02), flat)["verdict"] == "NEGATIVE"
    # ... but it cannot rule out NULL when overall dUAUC > 0.02, nor POSITIVE when hmt passes
    assert decide(0.03, nan, ci(0.5, 0.4, 0.6), flat, flat)["verdict"] == "UNDETERMINED"
    out = decide(0.0, nan, ci(0.5, 0.4, 0.6), flat, flat)
    assert out["verdict"] == "UNDETERMINED" and out["rules"] == {"NULL": False, "POSITIVE": None, "NEGATIVE": False}
    assert out["missing_inputs"] == ["dUAUC_real_minus_pseudo_tail"]
    assert decide(None, 0.03, ci(0.5, 0.4, 0.6), flat, flat)["verdict"] == "UNDETERMINED"  # None = missing
    assert decide(None, 0.0, ci(0.5, 0.4, 0.6), flat, flat)["verdict"] == "POSITIVE"  # tail alone rules out NULL


def test_empty_tail_bin_is_undetermined(tmp_path):
    # >= 40% of candidates tied at the minimum popularity: rank_bins puts the tied block in bin 1, so the tail
    # (bin 0) is empty, hmt and tail dUAUC are NaN; this used to fall through to the registered AMBIGUOUS
    dirs, pp, panel = _fixture(tmp_path, "positive")
    cut = np.quantile([p for r in panel for p in r["candidate_popularity"]], 0.45)
    for r in panel:
        r["candidate_popularity"] = [1 if p <= cut else p for p in r["candidate_popularity"]]
    arms = {a: load_arm(dirs[a]) for a in ARMS}
    flags, _, _ = treated_from_panel(pp, panel)
    res = analyze(arms, panel, flags, n_boot=50, seed=0, bi_boot=5)
    d = res["decision"]
    assert d["verdict"] == "UNDETERMINED" and d["values"]["n_tail_treated"] == 0
    assert d["values"]["n_rows_tail_all_candidates"] == 0 and "hmt_pseudo.est" in d["missing_inputs"]
    assert d["rules"]["NULL"] is False


def test_cluster_ci_wider_than_pair_ci_under_user_shocks():
    rng = np.random.default_rng(0)
    n_u, k = 200, 20
    users = np.repeat(np.arange(n_u), k)
    shock = rng.normal(0, 1, n_u)[users]                       # one history substitution shifts all rows
    head = np.repeat(np.arange(n_u) % 2 == 0, k)               # mainstream users hold the head ...
    tail = ~head                                               # ... niche users the tail
    d = 0.2 * head + shock + rng.normal(0, .05, n_u * k)
    cl = head_minus_tail(d, head, tail, users, n_boot=200)
    pair = head_minus_tail(d, head, tail, np.arange(n_u * k), n_boot=200)
    assert abs(cl["est"] - pair["est"]) < 1e-12
    assert (cl["hi"] - cl["lo"]) > 3 * (pair["hi"] - pair["lo"])
    assert cl["n_clusters"] == n_u


def test_censoring_dropped_and_counted(tmp_path):
    res = _run(tmp_path, "positive", censor=True)
    c = res["censoring"]
    assert c["real"]["like"]["1"] == 5 and c["real"]["like"].get("dropped_nonfinite", 0) == 0
    assert c["pseudo"]["like"]["2"] == 5 and c["pseudo"]["like"]["dropped_nonfinite"] == 5
    assert c["placebo"]["like"]["3"] == 5 and c["placebo"]["like"]["dropped_nonfinite"] == 5
    assert res["rows"]["n_analysed"] == 150 * 20 - 10                  # censored=1 rows are kept
    assert res["rows"]["n_dropped_not_finite_in_all_arms"] == 10


def test_cross_check_real_vs_pseudo(tmp_path):
    res = _run(tmp_path, "cross", prior=True)
    for blk in (res["cross_check"], res["cross_check"]["popularity_prior_robustness"]):
        ca = blk["corr_a_logpop"]["all"]
        assert ca["real"]["est"] > 0.6 and abs(ca["pseudo"]["est"]) < 0.1 and ca["real_minus_pseudo"]["lo"] > 0.4
        cp = blk["corr_pi_logpop"]
        assert cp["real"]["est"] > 0.6 and abs(cp["pseudo"]["est"]) < 0.15 and cp["real_minus_pseudo"]["lo"] > 0.4
        assert cp["n_items"] == cp["real"]["n_clusters"] > 1000        # item-level bootstrap
    assert "candidate_popularity_prior" in res["cross_check"]["popularity_prior_robustness"]["popularity"]


def test_cross_check_prior_block_absent_without_column(tmp_path):
    res = _run(tmp_path, "cross")
    assert res["cross_check"]["popularity_prior_robustness"]["available"] is False
    assert res["cross_check"]["corr_a_logpop"]["all"]["real"]["est"] > 0.6


def test_cli_recomputes_or_reads_treated_flags(tmp_path):
    dirs, pp, panel = _fixture(tmp_path, "positive", n_users=60, flags_in_panel=False)
    out = tmp_path / "res.json"
    cmd = [sys.executable, "-m", "src.confrec.pilot_pseudonym", *sum(([f"--{a}", str(dirs[a])] for a in ARMS), []),
           "--panel", str(pp), "--out", str(out), "--n_boot", "20", "--bi_boot", "7"]
    subprocess.run(cmd, check=True, capture_output=True)
    res = json.loads(out.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    assert res["treated"]["source"] == "recomputed" and math.isclose(res["treated"]["share"], 0.8, abs_tol=0.06)
    assert res["n_boot"] == 20 and res["bi_boot"] == 7
    flags = [{**r, "candidate_brand_in_title": [t.startswith(b) for b, t in zip(r["candidate_brands"],
                                                                               r["candidate_titles"])]}
             for r in panel]
    (tmp_path / "toys_rated_pseudo.jsonl").write_text("\n".join(json.dumps(r) for r in flags), encoding="utf-8")
    subprocess.run(cmd, check=True, capture_output=True)
    res2 = json.loads(out.read_text())
    assert res2["treated"]["source"] == "pseudo_panel" and res2["treated"]["n_flag_mismatch_vs_recomputed"] == 0
    assert res2["treated"]["n_treated"] == res["treated"]["n_treated"]
    assert res2["decision"]["verdict"] == res["decision"]["verdict"]
    assert res2["token_channel_gate"]["checked"] is False and res2["decision"]["token_channel_gate"] is None


def test_token_channel_gate_shared_with_pilot1(tmp_path):
    # triage_verdict.md section 3: the token-channel gate is common to pilots 1 and 3; a failed gate makes the pilot-3
    # verdict uninterpretable (before: pilot 3 never read the gate, and --gate did not exist)
    def gate_file(name, ok):
        p = tmp_path / name
        p.write_text(json.dumps({"decision": "AMBIGUOUS", "gate": {"pass": ok, "ml1m_raw_UAUC": 0.7}}), "utf-8")
        return p
    g = token_gate(gate_file("fail.json", False))
    d = apply_gate({"verdict": "POSITIVE"}, g)
    assert g["checked"] and d == {"verdict": "GATE_FAIL_UNINTERPRETABLE", "verdict_if_gate_passed": "POSITIVE",
                                  "token_channel_gate": False}
    assert apply_gate({"verdict": "NULL"}, token_gate(gate_file("pass.json", True))) == {
        "verdict": "NULL", "token_channel_gate": True}
    bad = tmp_path / "bad.json"
    bad.write_text("{trunc", encoding="utf-8")
    for p in (tmp_path / "missing.json", bad, None):
        g = token_gate(p)
        assert g["checked"] is False and g["pass"] is None
        assert apply_gate({"verdict": "NEGATIVE"}, g) == {"verdict": "NEGATIVE", "token_channel_gate": None}
    dirs, pp, _ = _fixture(tmp_path, "positive", n_users=40)
    out = tmp_path / "res.json"
    subprocess.run([sys.executable, "-m", "src.confrec.pilot_pseudonym",
                    *sum(([f"--{a}", str(dirs[a])] for a in ARMS), []), "--panel", str(pp), "--out", str(out),
                    "--n_boot", "10", "--bi_boot", "5", "--gate", str(tmp_path / "fail.json")],
                   check=True, capture_output=True)
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["decision"]["verdict"] == "GATE_FAIL_UNINTERPRETABLE" and res["token_channel_gate"]["pass"] is False
    assert res["decision"]["verdict_if_gate_passed"] in ("NULL", "POSITIVE", "NEGATIVE", "AMBIGUOUS", "UNDETERMINED")
