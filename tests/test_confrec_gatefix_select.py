"""Amendment-2 G4-G6 selection and confirmatory gate (src/confrec/gatefix_select.py) on synthetic scorer outputs."""
import csv
import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.confrec import gatefix_select as gs
from src.confrec import pilot_mirror as pm

ROOT = Path(__file__).resolve().parents[1]
COLS = ["source_event_id", "user_id", "item_id", "cand_idx", "label", "question", "lp_yes", "lp_no", "logit",
        "yes_no_mass", "censored", "exp_rating"]
N_USERS, N_CAND = 100, 10                    # 1,000 like rows per run: 0.5% = 5 censored=2 rows
Y = np.array([1, 0] * 5)
SIG = {"V0": 0.3, "V1": 0.3, "V2": 0.2, "V3": 1.0, "V4": 0.3, "V5": 0.25, "V7": 0.6}   # V3 is the planted fix


def report(v, n_rows, **kw):
    hist = 20 if v in ("V3", "V7") else 10
    r = {"variant": v, "readout": "yesno", "panel_kind": "rated", "questions": ["like"], "hist_len": hist,
         "hist_len_registered": hist, "lora": None, "dtype": "float16", "topk_logprobs": 50, "max_model_len": 4096,
         "model": "/models/Qwen3-8B", "data_sha1": "panel-sha1", "n_main_prompts": n_rows, "n_overlength": 0,
         "mean_yes_no_mass": 0.999, "system_sha1": "sys-movie" if v in ("V5", "V7") else None,
         "prompt_spec": {"variant": v, "hist": hist}, "scorer": "token_pyes_multiq_v3", "yes_ids": [9454, 9693],
         "no_ids": [2152, 2753], "config": {"prompts_sha1": f"prompts-{v}"}}
    r.update(kw)
    return r


def write_run(d, variant, logits, users, cens=None, mass=1.0, questions=("like",), rep=None, labels=Y):
    """A pyes_scorer output dir: scores.csv.gz (+ exp_rating) and report.json; cens = {(user idx, cand): code}."""
    d.mkdir(parents=True, exist_ok=True)
    rows, cens = [], cens or {}
    for ui, u in enumerate(users):
        for c in range(N_CAND):
            for q in questions:
                code = cens.get((ui, c), 0)
                lg = "nan" if code in (2, 3) else f"{logits[ui, c]:.6f}"
                m = 0.0 if code in (2, 3) else mass
                rows.append([f"{u}::0", u, f"{u}i{c}", c, int(labels[c]), q, 0, 0, lg, f"{m:.6f}", code, "nan"])
    with gzip.open(d / "scores.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    (d / "report.json").write_text(json.dumps(report(variant, len(rows), **(rep or {}))), encoding="utf-8")
    return d


def logits(s, seed, extra=None, n=N_USERS):
    """s * label + N(0, 1) noise shared within a panel (seed), plus independent variant noise (extra)."""
    x = s * Y + np.random.default_rng(seed).normal(size=(n, N_CAND))
    if extra is not None:
        x = x + 0.5 * np.random.default_rng(extra).normal(size=(n, N_CAND))
    return x


def users(prefix, n=N_USERS):
    return [f"{prefix}{i}" for i in range(n)]


@pytest.fixture(autouse=True)
def synthetic_pilot1_dev(monkeypatch):
    """The synthetic DEV panels (users m*, t*) stand in for the burned Pilot-1 panels: their user-id sha1s replace the
    registered ones (G2 check); everything else in PILOT1_DEV stays registered. Returns the registered constants."""
    reg = gs.PILOT1_DEV
    monkeypatch.setattr(gs, "PILOT1_DEV", {p: dict(reg[p], user_ids_sha1=gs._ids_sha1(users(p[0])))
                                           for p in gs.PANELS})
    return reg


def dev_root(tmp_path, sig=None, toys_sig=None, override=None):
    """ROOT/{ml1m,toys}/{V}; override[(panel, V)] = dict of write_run kwargs replacing the defaults."""
    sig, toys_sig, override = sig or SIG, toys_sig or {}, override or {}
    root = tmp_path / "dev"
    for p, seed in (("ml1m", 1), ("toys", 2)):
        for k, v in enumerate(gs.VARIANTS):
            s = sig[v] if p == "ml1m" else toys_sig.get(v, 0.3)
            kw = dict(logits=logits(s, seed, extra=None if v == "V0" else 100 + k), users=users(p[0]))
            kw.update(override.get((p, v), {}))
            write_run(root / p / v, v, **kw)
    return root


def run_dev(tmp_path, root, *extra, out="selection.json"):
    o = tmp_path / out
    code = gs.main(["dev", "--root", str(root), "--variants", ",".join(gs.VARIANTS), "--out", str(o), *extra])
    return code, (json.loads(o.read_text(encoding="utf-8")) if o.exists() else None), o


# ------------------------------------------------------------------ G5 selection rule (unit)
def tab(ml, toys, e1=None):
    e1 = e1 or {}
    return {v: {"ml1m": {"UAUC": ml[v], "E1": e1.get((v, "ml1m"), True)},
                "toys": {"UAUC": toys[v], "E1": e1.get((v, "toys"), True)}} for v in ml}


def test_select_tie_within_0005_is_broken_by_toys_then_simplicity():
    ml = dict(V0=0.587, V1=0.62, V2=0.600, V3=0.615, V4=0.590, V5=0.6149, V7=0.618)
    toys = dict(V0=0.55, V1=0.55, V2=0.56, V3=0.56, V4=0.55, V5=0.55, V7=0.555)
    s = gs.select(tab(ml, toys))
    assert s["max_ml1m_UAUC"] == 0.62 and s["tied"] == ["V1", "V3", "V7"]          # 0.62 - 0.615 is a tie (fp)
    assert s["v_star"] == "V3" and s["rule"] == "tie within 0.005 broken by Toys dev UAUC"   # V5 is 0.0051 below
    toys["V1"] = 0.56                                                             # V1 and V3 equal on Toys
    s = gs.select(tab(ml, toys))
    assert s["v_star"] == "V1" and s["tied_after_toys"] == ["V1", "V3"] and "simplicity" in s["rule"]
    ml2 = dict(ml, V5=0.62, V3=0.62, V1=0.60)
    toys2 = dict(toys, V5=0.56, V3=0.56)
    assert gs.select(tab(ml2, toys2))["v_star"] == "V5"                           # V5 precedes V3 in G5's order
    ml3 = dict(ml, V1=0.70)
    s = gs.select(tab(ml3, toys))
    assert s["v_star"] == "V1" and s["tied"] == ["V1"] and s["rule"] == "unique maximum"


def test_select_eligibility_e1_on_both_panels_and_e2_boundary():
    ml = dict(V0=0.58, V1=0.65, V2=0.60, V3=0.64, V4=0.59, V5=0.59, V7=0.63)
    toys = dict(V0=0.55, V1=0.5399, V2=0.55, V3=0.54, V4=0.55, V5=0.55, V7=0.55)
    t = tab(ml, toys)
    s = gs.select(t)
    assert t["V3"]["E2"] is True                                       # 0.54 = 0.55 - 0.010 counts (fp-safe)
    assert t["V1"]["E2"] is False and s["ineligible"]["V1"][0].startswith("E2 fails")
    assert s["v_star"] == "V3" and "V1" not in s["eligible"]
    s = gs.select(tab(ml, toys, e1={("V3", "toys"): False}))           # E1 must hold on BOTH panels
    assert s["v_star"] == "V7" and s["ineligible"]["V3"] == ["E1 fails on toys"]
    s = gs.select(tab(ml, dict(toys, V0=float("nan"))))                 # E2 undefined -> nobody is eligible
    assert s["v_star"] is None and s["eligible"] == []
    s = gs.select(tab(dict(ml, V3=float("nan")), toys))
    assert "ML-1M dev UAUC undefined" in s["ineligible"]["V3"]


def test_constants_match_the_gate_and_the_prompt_bank():
    spec = importlib.util.spec_from_file_location("pilot1_gate", ROOT / "scripts" / "sigir" / "pilot1_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    assert gs.UAUC_MIN == gate.UAUC_MIN == 0.60
    from src.confrec import prompting
    assert gs.SIMPLICITY == tuple(prompting.SIMPLICITY_ORDER) == ("V0", "V1", "V5", "V3", "V4", "V2", "V7")
    assert gs.VARIANTS == tuple(prompting.GATE_VARIANTS)
    assert abs(gs.ALPHA_ONE_SIDED - 0.05 / 6) < 1e-15 and gs.N_BOOT == 2000 and gs.SEED == 0


# ------------------------------------------------------------------ stage 1 end to end
def test_dev_fix_found_publishes_the_full_table(tmp_path, capsys):
    root = dev_root(tmp_path, toys_sig={"V3": 0.4})
    code, sel, _ = run_dev(tmp_path, root)
    assert code == 0 and sel["decision"] == "FIX_FOUND" and sel["fix_found"] and sel["v_star"] == "V3"
    assert sel["gate_ft_prompt"] == "V3"
    t = sel["table"]
    assert list(t) == list(gs.VARIANTS) and all(set(t[v]) >= {"ml1m", "toys", "E2", "eligible"} for v in t)
    assert all(t[v][p]["n_users_uauc"] == N_USERS and t[v][p]["E1"] for v in t for p in gs.PANELS)
    assert t["V3"]["ml1m"]["UAUC"] > 0.70 and t["V0"]["ml1m"]["UAUC"] < 0.62
    assert t["V2"]["ml1m"]["prompts_sha1"] == "prompts-V2"
    b = sel["bootstrap"]
    assert b["lower_bound"] > 0.05 and b["n_boot"] == 2000 and b["seed"] == 0 and b["n_users"] == N_USERS
    assert abs(b["level"] - (1 - 0.05 / 6)) < 1e-15
    assert abs(b["est"] - (t["V3"]["ml1m"]["UAUC"] - t["V0"]["ml1m"]["UAUC"])) < 1e-12   # nothing censored
    assert sel["dev_user_ids"]["ml1m"] == sorted(users("m"))
    assert sel["dev_user_ids_sha1"] == {"ml1m": gs._ids_sha1(users("m")), "toys": gs._ids_sha1(users("t"))}
    assert sel["dev_panel_identity"]["users_are_pilot1"] is True and sel["dev_panel_identity"]["manifest"] is None
    assert sel["confirm_panel_expected"] is None and len(sel["input_fingerprint"]["sha1"]) == 40
    assert set(sel["input_fingerprint"]["files"]) == {f"{p}/{v}" for p in gs.PANELS for v in gs.VARIANTS} | {
        "manifest"}
    assert t["V3"]["ml1m"]["prompt_spec_sha1"] and t["V3"]["ml1m"]["scorer"] == "token_pyes_multiq_v3"
    rep = sel["pilot1_reproduction"]["ml1m"]                                      # context: synthetic != Pilot 1
    assert rep["reproduces"] is False and rep["rows_equal_pilot1"] is False
    assert abs(rep["dUAUC_V0_dev_minus_pilot1"] - (t["V0"]["ml1m"]["UAUC"] - 0.5874475521977115)) < 1e-12
    out = capsys.readouterr().out
    assert "DECISION: FIX_FOUND" in out and out.count("\nV") >= 7                # 7 x 2 table printed
    assert "WARNING ml1m: V0 on DEV does not reproduce Pilot 1" in out


def test_dev_uauc_is_the_number_pilot_mirror_and_the_gate_compute(tmp_path):
    lg = logits(0.4, 7)
    cens = {(0, 1): 2, (3, 4): 1, (5, 0): 3}
    d = write_run(tmp_path / "r", "V0", lg, users("m"), cens=cens)
    run = gs.load_run(d)
    res = pm.analyze(pm.load_scores(d / "scores.csv.gz"),
                     [{"source_event_id": f"{u}::0", "user_id": u, "candidate_item_ids": [f"{u}i{c}" for c in
                                                                                         range(N_CAND)],
                       "candidate_labels": Y.tolist()} for u in users("m")], base_q="like", n_boot=5)
    row = gs.panel_row(run)
    assert row["UAUC"] == res["arms"]["raw"]["UAUC"] and row["n_rows_scored"] == 998      # censored 2/3 dropped
    assert row["n_censored2"] == 1 and row["n_censored3_scores"] == 1 and row["E1"] is False  # overlength 1


def test_dev_f0_when_the_control_is_selected(tmp_path):
    code, sel, _ = run_dev(tmp_path, dev_root(tmp_path, sig=dict(SIG, V0=1.0)))
    assert code == 4 and sel["decision"] == "GATE_FAIL_AFTER_REMEDY(dev)" and sel["outcome"] == "F0"
    assert sel["v_star"] == "V0" and sel["bootstrap"] is None and sel["gate_ft_prompt"] == "V0"


def test_dev_f0_when_the_selected_fix_is_not_significant(tmp_path):
    # V1 = V0 except one ML-1M user and one Toys user, each improved: V1 ties V0 on ML-1M (within 0.005), wins the
    # Toys tie-break, but the paired bootstrap's 0.05/6 quantile is 0 (resamples without that user)
    base_m, base_t = logits(0.3, 1), logits(0.3, 2)
    v1_m, v1_t = base_m.copy(), base_t.copy()
    v1_m[0], v1_t[0] = 5.0 * Y, 5.0 * Y
    sig = {v: 0.0 for v in gs.VARIANTS}
    ov = {("ml1m", "V0"): {"logits": base_m}, ("ml1m", "V1"): {"logits": v1_m},
          ("toys", "V0"): {"logits": base_t}, ("toys", "V1"): {"logits": v1_t}}
    code, sel, _ = run_dev(tmp_path, dev_root(tmp_path, sig=sig, toys_sig=sig, override=ov))
    assert sel["v_star"] == "V1" and sel["selection"]["tied"] == ["V0", "V1"]
    assert sel["bootstrap"]["est"] > 0 and sel["bootstrap"]["lower_bound"] == 0.0
    assert code == 4 and sel["decision"] == "GATE_FAIL_AFTER_REMEDY(dev)" and sel["gate_ft_prompt"] == "V0"


def test_fix_bootstrap_lower_bound_is_the_one_sided_0_05_over_6_quantile(tmp_path):
    a = gs.load_run(write_run(tmp_path / "a", "V3", logits(0.45, 1, extra=5), users("m")))
    b = gs.load_run(write_run(tmp_path / "b", "V0", logits(0.3, 1), users("m")))
    got = gs.fix_bootstrap(a, b)
    pa, pb = gs.user_aucs(a["sc"]), gs.user_aucs(b["sc"])
    d = np.array([pa[u] - pb[u] for u in sorted(pa)])
    rng = np.random.default_rng(0)
    boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(2000)]
    assert got["lower_bound"] == float(np.quantile(boots, 0.05 / 6))
    assert got["lower_bound"] != float(np.quantile(boots, 0.025))                 # not the two-sided 95% bound
    assert got["est"] == float(d.mean()) and got["lower_bound_gt_0"] == (got["lower_bound"] > 0)


def test_dev_e1_and_e2_make_a_variant_ineligible(tmp_path):
    many = {(u, 0): 2 for u in range(6)}                                          # 6 / 1000 = 0.6% > 0.5%
    ok5 = {(u, 0): 2 for u in range(5)}                                           # 5 / 1000 = 0.5% passes
    ov = {("toys", "V3"): {"cens": many}, ("ml1m", "V7"): {"mass": 0.94}, ("toys", "V1"): {"cens": ok5},
          ("ml1m", "V2"): {"rep": {"n_overlength": 1}}}
    code, sel, _ = run_dev(tmp_path, dev_root(tmp_path, sig=dict(SIG, V2=0.9, V1=0.8), toys_sig={"V1": 0.4},
                                              override=ov))
    t = sel["table"]
    assert t["V3"]["toys"]["censored2_share"] == 0.006 and t["V3"]["toys"]["E1"] is False and not t["V3"]["eligible"]
    assert t["V7"]["ml1m"]["mean_yes_no_mass_ge_0.95"] is False and not t["V7"]["eligible"]
    assert t["V2"]["ml1m"]["overlength"] == 1 and not t["V2"]["eligible"]
    assert t["V1"]["toys"]["E1"] is True and t["V1"]["eligible"] and sel["v_star"] == "V1"
    assert code == 0 and sel["decision"] == "FIX_FOUND"
    # E2: the ML-1M winner is dropped when it loses more than 0.010 Toys UAUC against V0
    code, sel, _ = run_dev(tmp_path / "e2", dev_root(tmp_path / "e2", toys_sig={"V3": -0.3, "V7": 0.4}))
    assert sel["table"]["V3"]["E2"] is False and sel["v_star"] == "V7" and code == 0


@pytest.mark.parametrize("panel,variant,kw,msg", [
    ("ml1m", "V2", {"questions": ("like", "dislike")}, "must be exactly ['like']"),
    ("toys", "V5", {"rep": {"variant": "V4"}}, "variant 'V4' != 'V5'"),
    ("ml1m", "V3", {"rep": {"hist_len": 10}}, "registered window"),
    ("ml1m", "V0", {"rep": {"max_model_len": 3072}}, "max_model_len 3072 != 4096"),
    ("toys", "V1", {"rep": {"readout": "digits"}}, "readout 'digits'"),
    ("ml1m", "V4", {"rep": {"lora": "/adapters/x"}}, "lora"),
    ("toys", "V7", {"rep": {"data_sha1": "other-panel"}}, "different panels"),
    ("ml1m", "V1", {"rep": {"model": "/models/Llama-3.1-8B"}}, "different models"),
    ("ml1m", "V5", {"rep": {"n_main_prompts": 1999}}, "stale or mixed"),
    ("toys", "V2", {"labels": np.array([0, 1] * 5)}, "same (event, candidate, user, label) rows"),
])
def test_dev_input_checks_refuse_without_recording_f0(tmp_path, panel, variant, kw, msg, capsys):
    code, sel, o = run_dev(tmp_path, dev_root(tmp_path, override={(panel, variant): kw}))
    assert code == 2 and sel is None and not o.exists()
    side = json.loads((tmp_path / "selection.input_check_failed.json").read_text(encoding="utf-8"))
    assert side["decision"] == "INPUT_CHECK_FAILED" and any(msg in e for e in side["errors"]), side["errors"]
    assert "INPUT CHECK FAILED" in capsys.readouterr().err


def test_dev_refuses_runs_that_are_not_on_the_burned_pilot1_users(tmp_path, monkeypatch, synthetic_pilot1_dev):
    root = dev_root(tmp_path, toys_sig={"V3": 0.4})
    monkeypatch.setattr(gs, "PILOT1_DEV", synthetic_pilot1_dev)                   # the registered constants
    code, sel, o = run_dev(tmp_path, root)
    side = json.loads((tmp_path / "selection.input_check_failed.json").read_text(encoding="utf-8"))
    assert code == 2 and sel is None and not o.exists()
    assert sum("not on the burned Pilot-1" in e for e in side["errors"]) == 2     # one error per panel
    assert all(e.count("'V") == 7 for e in side["errors"] if "Pilot-1" in e)     # listing all 7 variants
    # a single run on other users (e.g. the CONFIRM panel) is refused even when the rest are the DEV users
    monkeypatch.setattr(gs, "PILOT1_DEV", {p: dict(synthetic_pilot1_dev[p], user_ids_sha1=gs._ids_sha1(users(p[0])))
                                           for p in gs.PANELS})
    ov = {("ml1m", "V3"): {"users": users("c")}}
    code, _, _ = run_dev(tmp_path / "one", dev_root(tmp_path / "one", override=ov))
    side = json.loads((tmp_path / "one" / "selection.input_check_failed.json").read_text(encoding="utf-8"))
    assert code == 2 and any("ml1m: runs ['V3'] are not on the burned Pilot-1 ml1m users" in e for e in side["errors"])


def test_registered_pilot1_dev_constants_and_their_derivation(tmp_path):
    reg = {p: gs.PILOT1_DEV[p] for p in gs.PANELS}
    assert reg["ml1m"]["panel_sha1"] == "985494c7b44d010bec4b11ec62f35ad274dfe91f"            # amendment 2 G2
    assert reg["toys"]["panel_sha1"] == "69252b4806bb4ffe05101967ea2a1d94ada57dee"            # amendment 2 A1
    assert reg["ml1m"]["n_users"] == reg["toys"]["n_users"] == 1500 and reg["ml1m"]["raw_like_UAUC"] < 0.60
    # panel_dev_signature (how the constants were derived from the Pilot-1 panels) == the scores path
    d = write_run(tmp_path / "r", "V0", logits(0.3, 1), users("m"))
    panel = tmp_path / "panel.jsonl"
    panel.write_text("".join(json.dumps({"user_id": u, "source_event_id": f"{u}::0", "candidate_labels": Y.tolist()})
                             + "\n" for u in users("m")), encoding="utf-8")
    sc = gs.load_run(d)["sc"]
    sig = gs.panel_dev_signature(panel)
    assert sig == {"n_users": N_USERS, "n_rows": N_USERS * N_CAND, "user_ids_sha1": gs._ids_sha1(sc["user"].tolist()),
                   "rows_sha1": gs._rows_sig(sc)}
    # the user-id sha1 is build_confirm_panels' user-id list format (= the manifest freeze {src}_dev_users_sha1)
    import hashlib
    spec = importlib.util.spec_from_file_location("bcp", ROOT / "scripts" / "sigir" / "build_confirm_panels.py")
    bcp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bcp)
    ids = ["10", "2", "1", "AB", "2"][:4]
    assert gs._ids_sha1(ids * 3) == hashlib.sha1(bcp.user_list_bytes(ids)).hexdigest()


def test_dev_broken_control_is_an_input_check_not_a_final_f0(tmp_path):
    every = {(u, c): 2 for u in range(N_USERS) for c in range(N_CAND)}             # V0 Toys fully censored
    for k, ov in enumerate([{("toys", "V0"): {"cens": every}},                     # UAUC undefined + E1 fails
                            {("ml1m", "V0"): {"mass": 0.5}}]):                     # Yes+No mass collapsed
        code, sel, o = run_dev(tmp_path / f"c{k}", dev_root(tmp_path / f"c{k}", override=ov))
        side = json.loads((tmp_path / f"c{k}" / "selection.input_check_failed.json").read_text(encoding="utf-8"))
        assert code == 2 and sel is None and not o.exists(), k
        assert any("does not reproduce Pilot 1's channel" in e for e in side["errors"])
    assert any("toys/V0: UAUC undefined" in e for e in json.loads(
        (tmp_path / "c0" / "selection.input_check_failed.json").read_text(encoding="utf-8"))["errors"])


def test_pilot1_reproduction_is_recorded_and_never_gates(tmp_path, monkeypatch, capsys):
    root = dev_root(tmp_path, toys_sig={"V3": 0.4})
    v0 = {p: gs.load_run(root / p / "V0") for p in gs.PANELS}
    rows = {p: gs.panel_row(v0[p]) for p in gs.PANELS}
    monkeypatch.setattr(gs, "PILOT1_DEV", {p: dict(gs.PILOT1_DEV[p], rows_sha1=rows[p]["rows_sha1"],
                                                   v0_like_prompts_sha1="prompts-V0",
                                                   raw_like_UAUC=rows[p]["UAUC"] - 0.004) for p in gs.PANELS})
    code, sel, _ = run_dev(tmp_path, root)
    assert code == 0 and all(sel["pilot1_reproduction"][p]["reproduces"] for p in gs.PANELS)
    assert "does not reproduce Pilot 1" not in capsys.readouterr().out
    monkeypatch.setattr(gs, "PILOT1_DEV", {p: dict(gs.PILOT1_DEV[p], raw_like_UAUC=rows[p]["UAUC"] - 0.006)
                                           for p in gs.PANELS})
    code, sel2, _ = run_dev(tmp_path, root, out="s2.json")
    assert sel2["pilot1_reproduction"]["toys"]["reproduces"] is False and "WARNING toys" in capsys.readouterr().out
    assert code == 0 and (sel2["decision"], sel2["v_star"], sel2["bootstrap"]) == (sel["decision"], sel["v_star"],
                                                                                    sel["bootstrap"])


def manifest(tmp_path, dev_sha=None, confirm_ids=None, confirm_sha="panel-sha1"):
    m = {"sources": {p: {"dev": {"built": True, "path": f"{p}_dev_h20.jsonl", "sha1": dev_sha or "panel-sha1",
                                 "user_ids_sha1": gs._ids_sha1(users(p[0]))}} for p in gs.PANELS}}
    m["sources"]["ml1m"]["confirm"] = {"built": True, "path": "ml1m_confirm_h20.jsonl", "sha1": confirm_sha,
                                       "user_ids_sha1": gs._ids_sha1(confirm_ids or users("c"))}
    m["sources"]["toys"]["confirm"] = {"built": False, "reason": "1800 eligible users < 2000"}
    p = tmp_path / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)      # callers pass a fresh sub-directory (tmp_path / "m")
    p.write_text(json.dumps(m), encoding="utf-8")
    return p


def test_dev_manifest_pins_the_dev_panels_and_records_the_confirm_split(tmp_path):
    root = dev_root(tmp_path, toys_sig={"V3": 0.4})
    code, sel, _ = run_dev(tmp_path, root, "--manifest", str(manifest(tmp_path)))
    assert code == 0 and sel["dev_panel_identity"]["manifest"]["identity_sha1"] and sel["input_fingerprint"]["files"]["manifest"]
    assert sel["confirm_panel_expected"] == {"sha1": "panel-sha1", "user_ids_sha1": gs._ids_sha1(users("c")),
                                             "path": "ml1m_confirm_h20.jsonl"}
    code, sel, o = run_dev(tmp_path, root, "--manifest", str(manifest(tmp_path / "m", dev_sha="other")),
                           out="s2.json")
    side = json.loads((tmp_path / "s2.input_check_failed.json").read_text(encoding="utf-8"))
    assert code == 2 and not o.exists() and sum("the manifest split" in e for e in side["errors"]) == 14
    code, _, _ = run_dev(tmp_path, root, "--manifest", str(tmp_path / "missing.json"), out="s3.json")
    assert code == 2


def test_a_manifest_rebuilt_with_identical_panels_keeps_the_input_fingerprint(tmp_path):
    """Stage 0 rewrites the manifest's created_utc and code hashes whenever it reruns, although the panels come out
    byte-identical; run_gatefix.sh passes the manifest, so the rerun of the dev stage must not be refused as 'other
    inputs' (one selection, one gate), while another panel identity must be."""
    root = dev_root(tmp_path, toys_sig={"V3": 0.4})
    m1 = manifest(tmp_path)
    code, sel, _ = run_dev(tmp_path, root, "--manifest", str(m1))
    assert code == 0
    sel_bytes = (tmp_path / "selection.json").read_bytes()
    code, g1 = run_confirm(tmp_path / "c1", tmp_path / "selection.json", {"logits": logits(1.0, 3)})
    assert code == 0 and g1["confirm_panel_identity"]["checked"] is True
    doc = json.loads(m1.read_text(encoding="utf-8"))
    before = gs._sha1_file(m1)
    doc.update(created_utc="2026-10-05T00:00:00Z", code_sha1={"build_confirm_panels.py": "another-build"})
    m1.write_text(json.dumps(doc), encoding="utf-8")                           # stage 0 reran: same path, new bytes
    assert gs._sha1_file(m1) != before
    code, sel2, _ = run_dev(tmp_path, root, "--manifest", str(m1))             # same --out: accepted, same fingerprint
    assert code == 0 and sel2["input_fingerprint"] == sel["input_fingerprint"]
    assert sel2["dev_panel_identity"]["manifest"]["identity_sha1"] == sel["dev_panel_identity"]["manifest"][
        "identity_sha1"]
    assert (tmp_path / "selection.json").read_bytes() == sel_bytes   # byte-stable: gate.json fingerprints these bytes
    code, g2 = run_confirm(tmp_path / "c1", tmp_path / "selection.json", {"logits": logits(1.0, 3)})
    assert code == 0 and g2["input_fingerprint"] == g1["input_fingerprint"]          # the gate rerun is accepted too
    doc["sources"]["ml1m"]["confirm"]["user_ids_sha1"] = gs._ids_sha1(users("other-fresh"))   # another CONFIRM split
    m3 = tmp_path / "m3" / "manifest.json"
    m3.parent.mkdir()
    m3.write_text(json.dumps(doc), encoding="utf-8")
    code, _, _ = run_dev(tmp_path, root, "--manifest", str(m3))
    side = json.loads((tmp_path / "selection.input_check_failed.json").read_text(encoding="utf-8"))
    assert code == 2 and any("refusing to replace" in e for e in side["errors"])


def test_confirm_checks_the_panel_identity_from_the_manifest_or_the_selection(tmp_path, capsys):
    root = dev_root(tmp_path, toys_sig={"V3": 0.4})
    code, _, sp = run_dev(tmp_path, root, "--manifest", str(manifest(tmp_path)))
    assert code == 0
    code, g = run_confirm(tmp_path, sp, {"logits": logits(1.0, 3)})                # split recorded by dev
    assert code == 0 and g["confirm_panel_identity"]["checked"] is True
    assert g["confirm_panel_identity"]["source"].endswith("confirm_panel_expected") and g["rendering_equals_dev"]
    assert "panel identity is unchecked" not in capsys.readouterr().out
    other = manifest(tmp_path / "o", confirm_ids=users("x"))                      # another fresh-user list
    c = tmp_path / "confirm"
    code = gs.main(["confirm", "--vstar_dir", str(c / "V3"), "--v0_dir", str(c / "V0"), "--selection", str(sp),
                    "--out", str(tmp_path / "g2.json"), "--manifest", str(other)])
    side = json.loads((tmp_path / "g2.input_check_failed.json").read_text(encoding="utf-8"))
    assert code == 2 and sum("the manifest split" in e for e in side["errors"]) == 2
    # a dev run without a manifest: identity recorded as unchecked, with a warning (not refused)
    sp2 = selection(tmp_path / "nm")
    code, g = run_confirm(tmp_path / "nm", sp2, {"logits": logits(1.0, 3)})
    assert code == 0 and g["confirm_panel_identity"] == {"checked": False, "source": None, "expected": None}
    assert "WARNING: the CONFIRM panel identity is unchecked" in capsys.readouterr().out


def test_confirm_refuses_a_rendering_or_reading_that_differs_from_dev(tmp_path):
    sp = selection(tmp_path)
    for k, (star_rep, v0_rep, key) in enumerate([
            ({"prompt_spec": {"variant": "V3", "hist": 20, "template": "changed"}}, {}, "prompt_spec_sha1"),
            ({"system_sha1": "sys-product"}, {}, "system_sha1"),             # e.g. another domain's system message
            ({"scorer": "token_pyes_multiq_v4"}, {}, "scorer"),
            ({}, {"yes_ids": [9454]}, "yes_ids")]):                           # the V0 context run is checked too
        code, g = run_confirm(tmp_path / f"r{k}", sp, {"logits": logits(1.0, 3), "rep": star_rep},
                              v0_kw={"rep": v0_rep})
        side = json.loads((tmp_path / f"r{k}" / "gate.input_check_failed.json").read_text(encoding="utf-8"))
        assert code == 2 and g is None and any(key in e and "differ from the dev" in e for e in side["errors"]), k


def test_dev_rerun_refuses_other_inputs_even_with_the_same_decision(tmp_path):
    root = dev_root(tmp_path, toys_sig={"V3": 0.4})
    code, sel, o = run_dev(tmp_path, root)
    assert code == 0 and sel["v_star"] == "V3"
    ov = {("ml1m", "V2"): {"logits": logits(0.2, 1, extra=999)}}                  # V2 rescored, still not selected
    other = dev_root(tmp_path / "x", toys_sig={"V3": 0.4}, override=ov)
    code, _, _ = run_dev(tmp_path, other)
    side = json.loads((tmp_path / "selection.input_check_failed.json").read_text(encoding="utf-8"))
    assert code == 2 and "input fingerprint differ" in side["errors"][0] and "decision" not in \
        side["errors"][0].split("(")[-1]
    assert json.loads(o.read_text(encoding="utf-8"))["input_fingerprint"] == sel["input_fingerprint"]   # kept
    code, sel2, _ = run_dev(tmp_path, other, "--overwrite")
    assert code == 0 and sel2["v_star"] == "V3" and sel2["input_fingerprint"] != sel["input_fingerprint"]
    old = json.loads(o.read_text(encoding="utf-8"))                               # a record without a fingerprint
    del old["input_fingerprint"]
    o.write_text(json.dumps(old), encoding="utf-8")
    assert run_dev(tmp_path, other)[0] == 2


def test_dev_requires_the_whole_registered_bank(tmp_path):
    root = dev_root(tmp_path)
    o = tmp_path / "s.json"
    assert gs.main(["dev", "--root", str(root), "--variants", "V0,V1,V3", "--out", str(o)]) == 2 and not o.exists()
    import shutil
    shutil.rmtree(root / "toys" / "V4")
    code, sel, _ = run_dev(tmp_path, root)
    side = json.loads((tmp_path / "selection.input_check_failed.json").read_text(encoding="utf-8"))
    assert code == 2 and any("toys" in e and "V4" in e and e.startswith("missing") for e in side["errors"])


def test_dev_rerun_never_silently_replaces_a_recorded_selection(tmp_path):
    root = dev_root(tmp_path, toys_sig={"V3": 0.4})
    code, sel, o = run_dev(tmp_path, root)
    assert code == 0 and sel["v_star"] == "V3"
    assert run_dev(tmp_path, root)[0] == 0                                        # same inputs: same record
    (tmp_path / "selection.input_check_failed.json").write_text("{}", encoding="utf-8")
    changed = dev_root(tmp_path / "x", sig=dict(SIG, V0=1.0))                     # now F0 / V0
    code, _, _ = run_dev(tmp_path, changed)
    assert code == 2 and json.loads(o.read_text(encoding="utf-8"))["v_star"] == "V3"   # kept
    code, sel, _ = run_dev(tmp_path, changed, "--overwrite")
    assert code == 4 and sel["v_star"] == "V0"
    assert not (tmp_path / "selection.input_check_failed.json").exists()          # cleared on success


# ------------------------------------------------------------------ stage 2
def auc_logits(k_per_user):
    """Per user 5 positives / 5 negatives with exactly k of the 25 pairs ordered (AUC = k / 25), no ties."""
    out = np.zeros((len(k_per_user), N_CAND))
    for ui, k in enumerate(k_per_user):
        neg, pos = np.where(Y == 0)[0], np.where(Y == 1)[0]
        out[ui, neg] = np.arange(5.0)
        out[ui, pos] = [(k // 5 + (i < k % 5)) - 0.5 for i in range(5)]
    return out


def selection(tmp_path, root=None):
    code, sel, o = run_dev(tmp_path, root or dev_root(tmp_path, toys_sig={"V3": 0.4}))
    assert code == 0 and sel["v_star"] == "V3"
    return o


def run_confirm(tmp_path, sel_path, star_kw, v0_kw=None, out="gate.json"):
    c = tmp_path / "confirm"
    write_run(c / "V3", "V3", **{"users": users("c"), **star_kw})
    write_run(c / "V0", "V0", **{"users": users("c"), "logits": logits(0.3, 9), **(v0_kw or {})})
    o = tmp_path / out
    code = gs.main(["confirm", "--vstar_dir", str(c / "V3"), "--v0_dir", str(c / "V0"), "--selection",
                    str(sel_path), "--out", str(o)])
    return code, (json.loads(o.read_text(encoding="utf-8")) if o.exists() else None)


def test_confirm_gate_threshold_on_the_point_estimate(tmp_path):
    sp = selection(tmp_path)
    above = [10, 20] * (N_USERS // 2)                                             # per-user AUC 0.4 / 0.8
    above[0] = 11                                                                 # UAUC = 0.6 + 0.04 / 100
    code, g = run_confirm(tmp_path, sp, {"logits": auc_logits(above)})
    assert code == 0 and g["decision"] == "GATE_PASS" and g["outcome"] == "PASS" and g["v_star"] == "V3"
    assert abs(g["UAUC"] - 0.6004) < 1e-12 and g["E1"]["E1"] and g["n_users_uauc"] == N_USERS
    ci = g["ci95"]
    assert ci["n_boot"] == 2000 and ci["lo"] <= g["UAUC"] <= ci["hi"] and ci["level"] == 0.95
    ctx = g["v0_context"]
    assert ctx["UAUC"] < 0.62 and set(ctx["dUAUC_vstar_minus_v0"]) >= {"est", "lo", "hi", "n"}
    assert g["confirm_users_disjoint_from_dev"] is True
    below = [10, 20] * (N_USERS // 2)
    below[1] = 19                                                                 # UAUC = 0.6 - 0.04 / 100
    code, g = run_confirm(tmp_path, sp, {"logits": auc_logits(below)}, out="g2.json")
    assert code == 3 and g["decision"] == "GATE_FAIL_AFTER_REMEDY(confirm)" and g["outcome"] == "F1"
    assert abs(g["UAUC"] - 0.5996) < 1e-12 and g["E1"]["E1"] and g["UAUC_ge_min"] is False
    assert g["ci95"]["hi"] > 0.60                                                 # the CI never rescues the gate


def test_confirm_e1_failure_is_f1_even_above_the_bar(tmp_path):
    sp = selection(tmp_path)
    cens = {(u, 0): 2 for u in range(6)}                                          # 6 / 1000 > 0.5%
    code, g = run_confirm(tmp_path, sp, {"logits": logits(1.0, 3), "cens": cens})
    assert g["UAUC"] > 0.7 and g["E1"]["E1"] is False and code == 3 and g["decision"].endswith("(confirm)")
    code, g = run_confirm(tmp_path, sp, {"logits": logits(1.0, 3)}, v0_kw={"mass": 0.5}, out="g2.json")
    assert code == 0 and g["v0_context"]["E1"]["E1"] is False                    # V0's E1 is context only


def test_confirm_refuses_without_a_found_fix_overlapping_users_or_another_model(tmp_path):
    sp = selection(tmp_path)
    f0 = tmp_path / "f0.json"
    f0.write_text(json.dumps({**json.loads(sp.read_text(encoding="utf-8")), "decision": gs.F0}), encoding="utf-8")
    code, g = run_confirm(tmp_path, f0, {"logits": logits(1.0, 3)})
    side = json.loads((tmp_path / "gate.input_check_failed.json").read_text(encoding="utf-8"))
    assert code == 2 and g is None and "runs only after FIX_FOUND" in side["errors"][0]
    dev_users = users("m")                                                        # CONFIRM must be fresh users
    code, g = run_confirm(tmp_path, sp, {"logits": logits(1.0, 3), "users": dev_users},
                          v0_kw={"users": dev_users})
    side = json.loads((tmp_path / "gate.input_check_failed.json").read_text(encoding="utf-8"))
    assert code == 2 and any("CONFIRM users are DEV users" in e for e in side["errors"])
    code, g = run_confirm(tmp_path, sp, {"logits": logits(1.0, 3), "rep": {"model": "/m/other"}},
                          v0_kw={"rep": {"model": "/m/other"}})
    side = json.loads((tmp_path / "gate.input_check_failed.json").read_text(encoding="utf-8"))
    assert code == 2 and any("different models" in e for e in side["errors"])
    code, g = run_confirm(tmp_path, sp, {"logits": logits(1.0, 3), "rep": {"variant": "V7"}})
    assert code == 2 and g is None


# The CLI in a fresh interpreter. The registered DEV identity (G2: the burned Pilot-1 users) is hard-coded, so the
# synthetic DEV panels' user-id sha1s are patched in exactly as the autouse fixture does in-process; everything else
# (argparse, sys.exit(main()), the exit codes) is the real command line.
CLI = ("import os, sys; from src.confrec import gatefix_select as gs; "
       "gs.PILOT1_DEV = {p: dict(gs.PILOT1_DEV[p], user_ids_sha1=os.environ['SYNTH_DEV_IDS_' + p]) for p in gs.PANELS}; "
       "sys.exit(gs.main(sys.argv[1:]))")


def cli(*argv):
    import os
    env = dict(os.environ, **{f"SYNTH_DEV_IDS_{p}": gs._ids_sha1(users(p[0])) for p in gs.PANELS})
    return subprocess.run([sys.executable, "-c", CLI, *argv], capture_output=True, text=True, cwd=ROOT, env=env)


def test_cli_exit_codes(tmp_path):
    root = dev_root(tmp_path, toys_sig={"V3": 0.4})
    p = cli("dev", "--root", str(root), "--variants", ",".join(gs.VARIANTS), "--out", str(tmp_path / "sel.json"))
    assert p.returncode == 0 and "DECISION: FIX_FOUND" in p.stdout, p.stderr
    sel = json.loads((tmp_path / "sel.json").read_text(encoding="utf-8"), parse_constant=lambda c: 1 / 0)
    assert sel["v_star"] == "V3"
    root2 = dev_root(tmp_path / "f0", sig=dict(SIG, V0=1.0))
    p = cli("dev", "--root", str(root2), "--out", str(tmp_path / "sel2.json"))
    assert p.returncode == 4 and "GATE_FAIL_AFTER_REMEDY(dev)" in p.stdout, p.stderr


def test_cli_refuses_a_dev_run_that_is_not_on_the_registered_pilot1_users(tmp_path):
    # the real, unpatched command line (what run_gatefix.sh runs): synthetic users are not the burned Pilot-1 users
    root = dev_root(tmp_path, toys_sig={"V3": 0.4})
    p = subprocess.run([sys.executable, "-m", "src.confrec.gatefix_select", "dev", "--root", str(root), "--out",
                        str(tmp_path / "sel.json")], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 2 and "DEV must be the Pilot-1 panel's users" in p.stderr
    assert not (tmp_path / "sel.json").exists() and (tmp_path / "sel.input_check_failed.json").exists()
