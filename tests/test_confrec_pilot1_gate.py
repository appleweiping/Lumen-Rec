import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pilot1_gate", ROOT / "scripts" / "sigir" / "pilot1_gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def rated(sd=0.30, corr=0.30, lo=0.01, hi=0.05, v=0.0, pi=0.0, uauc=0.65, sd_df=None, corr_prior=None):
    cp = corr if corr_prior is None else corr_prior
    return {"panel_type": "rated", "base_question": "like", "n_users": 1500, "panel_join": {},
            "arms": {"raw": {"UAUC": uauc}},
            "acquiescence": {"SD_pair": sd, "SD_pair_df": sd if sd_df is None else sd_df,
                             "corr_a_logpop": {"est": corr, "lo": corr - .1, "hi": corr + .1}},
            "dUAUC_mirror_minus_placebo": {"est": (lo + hi) / 2, "lo": lo, "hi": hi, "n": 1500},
            "valence": {"corr_v_logpop": {"est": v}}, "prior": {"corr_pi_logpop": {"est": pi}},
            "popularity_prior_robustness": {"corr_a_logpop": {"est": cp}, "corr_v_logpop": {"est": v},
                                            "corr_pi_logpop": {"est": pi}}}


WEAK = dict(sd=0.15, corr=0.05, lo=-0.01, hi=0.02)     # no criterion met in this domain


def sports(raw=0.20, raw_like=0.20, mirror=0.20, ccrp=0.2310, paired=None, exact_loss=None, unscored=0):
    """Arm-level NDCG@10 plus the paired common-support deltas; `paired` overrides (mirror - raw, mirror - raw_like)
    and `exact_loss` sets the tie-exact losses vs (raw, raw_like)."""
    dr, dl = paired or (mirror - raw, mirror - raw_like)
    er, el = exact_loss or (-dr, -dl)
    arm = lambda x: {"NDCG@10": x, "NDCG@10_tie_exact": x, "n_events_positive_unscored": 0,  # noqa: E731
                     "n_events_negative_unscored": unscored}
    return {"panel_type": "next_item", "base_question": "next", "n_users": 1000, "next_item": {"n_events": 1000},
            "arms": {"raw": arm(raw), "raw_like": arm(raw_like), "mirror": arm(mirror), "placebo": arm(0.2),
                     "ccrp": {"NDCG@10": ccrp}},
            "ref_ranks": {"n_joined": 1000},
            "dNDCG@10_mirror_minus_raw": {"est": dr, "lo": -0.01, "hi": 0.01, "n": 1000, "tie_exact": {"est": -er}},
            "dNDCG@10_mirror_minus_raw_like": {"est": dl, "lo": -0.01, "hi": 0.01, "n": 1000,
                                               "tie_exact": {"est": -el}},
            "dNDCG@10_placebo_minus_ccrp": {"est": 0.2 - ccrp, "lo": -0.03, "hi": 0.0, "n": 1000}}


def run(tmp_path, ml, toys, sp, out="out"):
    paths = []
    for name, d in (("ml1m", ml), ("toys", toys), ("sports", sp)):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        paths.append(str(p))
    od = tmp_path / out
    code = gate.main(["--ml1m", paths[0], "--toys", paths[1], "--sports", paths[2], "--out_dir", str(od)])
    return code, json.loads((od / "decision.json").read_text(encoding="utf-8")), od


def test_positive(tmp_path):
    code, dec, od = run(tmp_path, rated(), rated(**WEAK), sports(mirror=0.196))   # loss 0.004 <= 0.005
    assert code == 0 and dec["decision"] == "POSITIVE" and dec["handoff"] is None
    assert (od / "GATE_PASS").exists() and not (od / "GATE_FAIL").exists()
    c = dec["criteria"]["ml1m"]
    assert c["SD_pair"] == 0.30 and c["corr_a_logpop"] == 0.30 and c["dUAUC_mirror_minus_placebo"]["lo"] == 0.01
    nl = dec["next_item_no_loss"]
    assert abs(nl["loss_vs_raw"] - 0.004) < 1e-12 and nl["no_loss"] is True
    assert nl["dNDCG@10_placebo_minus_ccrp"]["n"] == 1000 and not nl["unscored_candidates_present"]
    assert dec["gate"]["ccrp_same_events_NDCG@10"] == 0.2310 and dec["gate"]["input_checks_ok"]
    assert dec["missing_inputs"] == [] and dec["sensitivity"]["all_agree_with_primary"]
    assert dec["input_summary"]["sports"]["n_events"] == 1000 and dec["input_summary"]["ml1m"]["n_users"] == 1500
    txt = (od / "GATE_PASS").read_text(encoding="utf-8")
    assert "same 1000 events" in txt and "0.1848" in txt and "0.1863" in txt     # reference printed, not gated


def test_positive_blocked_by_next_item_loss_vs_raw_or_raw_like(tmp_path):
    _, dec, _ = run(tmp_path, rated(), rated(**WEAK), sports(raw=0.21, raw_like=0.20, mirror=0.20))
    assert dec["decision"] == "AMBIGUOUS" and dec["next_item_no_loss"]["no_loss"] is False
    _, dec, _ = run(tmp_path, rated(), rated(**WEAK), sports(raw=0.20, raw_like=0.21, mirror=0.20), out="o2")
    assert dec["decision"] == "AMBIGUOUS" and dec["next_item_no_loss"]["loss_vs_raw"] == 0.0


def test_no_loss_uses_the_paired_common_support_delta(tmp_path):
    # arm-level NDCG says mirror gains 0.05 (its censored negatives fell below the positive); on the candidates
    # scored in both arms mirror loses 0.01 -> no-loss fails
    sp = sports(raw=0.20, raw_like=0.20, mirror=0.25, paired=(-0.01, -0.01), unscored=7)
    _, dec, od = run(tmp_path, rated(), rated(**WEAK), sp)
    nl = dec["next_item_no_loss"]
    assert abs(nl["arm_level"]["loss_vs_raw"] + 0.05) < 1e-12 and nl["loss_vs_raw"] == 0.01
    assert nl["no_loss"] is False and nl["unscored_candidates_present"] and dec["decision"] == "AMBIGUOUS"
    assert nl["unscored"]["mirror"]["n_events_negative_unscored"] == 7
    assert "7 sports events have unscored negatives" in (od / "GATE_PASS").read_text(encoding="utf-8")


def test_negative_small_sd_everywhere_with_and_without_handoff(tmp_path):
    small = dict(sd=0.05, corr=0.30, lo=-0.01, hi=0.02)
    _, dec, _ = run(tmp_path, rated(**small, v=0.25), rated(**small), sports())
    assert dec["decision"] == "NEGATIVE" and dec["handoff"] == "B9/A9"
    _, dec, _ = run(tmp_path, rated(**small), rated(**small, pi=-0.30), sports(), out="o2")
    assert dec["decision"] == "NEGATIVE" and dec["handoff"] == "B9/A9"              # |corr(pi)| counts
    _, dec, _ = run(tmp_path, rated(**small, v=0.1), rated(**small, pi=0.1), sports(), out="o3")
    assert dec["decision"] == "NEGATIVE" and dec["handoff"] is None
    ml, toys = rated(**small), rated(**small)
    for d in (ml, toys):
        del d["valence"], d["prior"]
    _, dec, _ = run(tmp_path, ml, toys, sports(), out="o4")
    assert dec["decision"] == "NEGATIVE" and dec["handoff"].startswith("UNDETERMINED")


def test_negative_and_null_overlap_is_flagged_with_the_null_alternative(tmp_path):
    _, dec, _ = run(tmp_path, rated(lo=-0.03, hi=-0.001), rated(lo=-0.02, hi=0.0), sports())
    assert dec["conditions"]["null"] and dec["conditions"]["negative"] and dec["overlap_negative_and_null"]
    # sign-off 2026-10-02 (amendment P1.5b): NULL wins the overlap; NEGATIVE-first is the reported alternative
    assert dec["decision"] == "NULL" and dec["handoff"] == gate.NULL_HANDOFF
    alt = dec["sensitivity"]["NEGATIVE_before_NULL"]
    assert alt["decision"] == "NEGATIVE"
    assert not dec["sensitivity"]["all_agree_with_primary"]


def test_small_sd_in_only_one_domain_is_not_negative(tmp_path):
    _, dec, _ = run(tmp_path, rated(sd=0.05, corr=0.05, lo=-0.01, hi=0.02), rated(**WEAK), sports())
    assert dec["decision"] == "AMBIGUOUS"


def test_null(tmp_path):
    _, dec, _ = run(tmp_path, rated(lo=-0.01, hi=0.02), rated(**WEAK), sports())
    assert dec["decision"] == "NULL" and dec["handoff"].startswith("B9/A9") and not dec["overlap_negative_and_null"]


def test_missing_primary_endpoint_is_incomplete_not_null(tmp_path):
    toys = rated(**WEAK)
    del toys["dUAUC_mirror_minus_placebo"]                                         # block absent
    _, dec, _ = run(tmp_path, rated(lo=-0.01, hi=0.02), toys, sports())
    assert dec["decision"] == "INCOMPLETE" and dec["conditions"]["null"] is None and dec["handoff"] is None
    assert dec["missing_inputs"] == ["toys.dUAUC_mirror_minus_placebo.lo", "toys.dUAUC_mirror_minus_placebo.hi"]
    ml = rated()
    ml["dUAUC_mirror_minus_placebo"] = {"est": None, "lo": None, "hi": None, "n": 0}   # strict-JSON NaN, n = 0
    _, dec, _ = run(tmp_path, ml, rated(**WEAK), sports(), out="o2")
    assert dec["decision"] == "INCOMPLETE" and dec["criteria"]["ml1m"]["dUAUC_lo_gt_0"] is None
    _, dec, _ = run(tmp_path, rated(lo=-0.01, hi=0.02), rated(**WEAK), sports(), out="o3")
    assert dec["decision"] == "NULL"                                               # finite CI spanning 0 in both


def test_three_valued_logic_decides_when_the_available_values_suffice(tmp_path):
    toys = rated(**WEAK)
    del toys["dUAUC_mirror_minus_placebo"], toys["acquiescence"]
    _, dec, _ = run(tmp_path, rated(), toys, sports(mirror=0.196))                 # ml1m alone meets POSITIVE
    assert dec["decision"] == "POSITIVE" and len(dec["missing_inputs"]) == 4
    sp = sports()
    del sp["dNDCG@10_mirror_minus_raw"]                                            # no-loss input missing
    _, dec, _ = run(tmp_path, rated(), rated(**WEAK), sp, out="o2")
    assert dec["decision"] == "INCOMPLETE" and dec["next_item_no_loss"]["no_loss"] is None
    assert dec["missing_inputs"] == ["sports.loss_vs_raw"]


def test_ambiguous_gain_without_large_popularity_linked_acquiescence(tmp_path):
    _, dec, _ = run(tmp_path, rated(sd=0.15, corr=0.30), rated(**WEAK), sports())
    assert dec["decision"] == "AMBIGUOUS" and dec["any_domain_dUAUC_lo_gt_0"]


def test_sensitivity_to_sd_estimator_popularity_and_exact_ties(tmp_path):
    # primary gates on SD_pair_df (divisor N - n_users); divisor N is the reported alternative
    _, dec, _ = run(tmp_path, rated(sd=0.19, sd_df=0.205), rated(**WEAK), sports(mirror=0.196))
    s = dec["sensitivity"]
    assert dec["decision"] == "POSITIVE" and s["SD_pair_divisor_N"]["decision"] == "AMBIGUOUS"
    assert dec["criteria"]["ml1m"]["SD_pair"] == 0.205 and not s["all_agree_with_primary"]
    _, dec, _ = run(tmp_path, rated(corr_prior=0.15), rated(**WEAK), sports(mirror=0.196), out="o2")
    s = dec["sensitivity"]
    assert dec["decision"] == "POSITIVE" and s["prior_popularity"]["decision"] == "AMBIGUOUS"
    assert dec["criteria"]["ml1m"]["reference_corr_a_logpop_prior_pop"] == 0.15
    _, dec, _ = run(tmp_path, rated(), rated(**WEAK), sports(mirror=0.196, exact_loss=(0.006, 0.004)), out="o3")
    s = dec["sensitivity"]
    # primary no-loss uses the exact tie-group NDCG; the plug-in expected rank is the reported alternative
    assert dec["decision"] == "AMBIGUOUS" and s["ndcg_plugin_expected_rank"]["decision"] == "POSITIVE"
    assert s["SD_pair_divisor_N"]["decision"] == s["prior_popularity"]["decision"] == "AMBIGUOUS"


def test_gate_fail_ml1m_uauc(tmp_path):
    code, dec, od = run(tmp_path, rated(uauc=0.59), rated(), sports())
    assert code == 3 and dec["decision"] == "GATE_FAIL_UNINTERPRETABLE"
    assert (od / "GATE_FAIL").exists() and not (od / "GATE_PASS").exists()
    assert "0.5900" in (od / "GATE_FAIL").read_text(encoding="utf-8")
    assert dec["criteria"]["ml1m"]["SD_pair"] == 0.30                               # values still recorded


def test_gate_fails_on_wrong_panel_type_or_base_question(tmp_path):
    sp = sports()
    sp["base_question"] = "like"                                                   # raw arm is like, not next
    code, dec, od = run(tmp_path, rated(), rated(), sp)
    assert code == 3 and dec["decision"] == "GATE_FAIL_UNINTERPRETABLE" and not dec["gate"]["input_checks_ok"]
    assert dec["gate"]["input_checks"]["sports"] == {"panel_type": "next_item", "base_question": "like",
                                                     "expected": ["next_item", "next"], "ok": False}
    assert "INPUT CHECK FAILED: sports" in (od / "GATE_FAIL").read_text(encoding="utf-8")
    ml = rated()
    ml["panel_type"] = "next_item"
    code, dec, _ = run(tmp_path, ml, rated(), sports(), out="o2")
    assert code == 3 and not dec["gate"]["input_checks"]["ml1m"]["ok"]
    toys = rated()
    del toys["panel_type"]
    code, dec, _ = run(tmp_path, rated(), toys, sports(), out="o3")
    assert code == 3 and dec["gate"]["input_checks"]["toys"]["panel_type"] is None


def test_gate_sports_threshold_and_missing_values(tmp_path):
    code, dec, _ = run(tmp_path, rated(), rated(), sports(raw=0.1863))               # 0.8 * 0.2329 = 0.18632
    assert code == 3 and not dec["gate"]["pass"]
    code, _, _ = run(tmp_path, rated(), rated(), sports(raw=0.1864), out="o2")
    assert code == 0
    ml = rated()
    ml["arms"]["raw"]["UAUC"] = None                                                 # strict-JSON NaN
    code, dec, _ = run(tmp_path, ml, rated(), sports(), out="o3")
    assert code == 3 and dec["gate"]["ml1m_raw_UAUC"] is None


def test_stale_marker_removed_and_cli_exit_code(tmp_path):
    code, _, od = run(tmp_path, rated(uauc=0.5), rated(), sports())
    assert code == 3 and (od / "GATE_FAIL").exists()
    code, _, od = run(tmp_path, rated(), rated(), sports())
    assert code == 0 and (od / "GATE_PASS").exists() and not (od / "GATE_FAIL").exists()
    (tmp_path / "ml1m.json").write_text(json.dumps(rated(uauc=0.5)), encoding="utf-8")
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "sigir" / "pilot1_gate.py"),
                        "--ml1m", str(tmp_path / "ml1m.json"), "--toys", str(tmp_path / "toys.json"),
                        "--sports", str(tmp_path / "sports.json"), "--out_dir", str(od)], capture_output=True)
    assert p.returncode == 3 and (od / "GATE_FAIL").exists() and not (od / "GATE_PASS").exists()


def test_rerun_with_unchanged_inputs_keeps_the_outputs_untouched(tmp_path):
    # run_pilot2_3.sh's pilot-3 analysis depends on decision.json: a mere rerun of pilot 1 must not re-trigger it
    import os
    _, dec, od = run(tmp_path, rated(), rated(), sports())
    old = 1_000_000_000_000_000_000
    for f in ("decision.json", "GATE_PASS"):
        os.utime(od / f, ns=(old, old))
    _, dec2, _ = run(tmp_path, rated(), rated(), sports())
    assert dec2 == dec and all((od / f).stat().st_mtime_ns == old for f in ("decision.json", "GATE_PASS"))
    _, dec3, _ = run(tmp_path, rated(uauc=0.5), rated(), sports())                  # changed content is rewritten
    assert (od / "decision.json").stat().st_mtime_ns != old and dec3["gate"]["pass"] is False
    assert (od / "GATE_FAIL").exists() and not (od / "GATE_PASS").exists()
