import importlib.util
import itertools
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pilot1_gate", ROOT / "scripts" / "sigir" / "pilot1_gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def rated(sd=0.30, corr=0.30, lo=0.01, hi=0.05, v=0.0, pi=0.0, uauc=0.65, sd_df=None, corr_prior=None,
          elo=0.004, ehi=0.03):
    """pilot_mirror output; elo/ehi = CI of dUAUC(mirror - ensemble_null) (amendment 2 G7), None drops the block."""
    cp = corr if corr_prior is None else corr_prior
    d = {"panel_type": "rated", "base_question": "like", "n_users": 1500, "panel_join": {},
         "arms": {"raw": {"UAUC": uauc}},
         "acquiescence": {"SD_pair": sd, "SD_pair_df": sd if sd_df is None else sd_df,
                          "corr_a_logpop": {"est": corr, "lo": corr - .1, "hi": corr + .1}},
         "dUAUC_mirror_minus_placebo": {"est": (lo + hi) / 2, "lo": lo, "hi": hi, "n": 1500},
         "valence": {"corr_v_logpop": {"est": v}}, "prior": {"corr_pi_logpop": {"est": pi}},
         "popularity_prior_robustness": {"corr_a_logpop": {"est": cp}, "corr_v_logpop": {"est": v},
                                         "corr_pi_logpop": {"est": pi}}}
    if elo is not None:
        d["ensemble_null"] = {"mirror_pair": {"predicted_UAUC": 0.60, "observed_UAUC": 0.617},
                              "dUAUC_mirror_minus_ensemble_null": {"est": (elo + ehi) / 2, "lo": elo, "hi": ehi,
                                                                   "n": 1500},
                              "dUAUC_placebo_minus_ensemble_null": {"est": 0.001, "lo": -0.004, "hi": 0.006,
                                                                    "n": 1500}}
    return d


WEAK = dict(sd=0.15, corr=0.05, lo=-0.01, hi=0.02, elo=-0.02, ehi=0.01)     # no criterion met in this domain


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
    e = c["dUAUC_mirror_minus_ensemble_null"]                                     # amendment 2 G7: recorded
    assert e["lo"] == 0.004 and e["n"] == 1500 and e["predicted_UAUC"] == 0.60 and e["observed_UAUC"] == 0.617
    assert c["ensemble_null_lo_gt_0"] is True and c["beats_placebo_and_ensemble_null"] is True
    assert dec["any_domain_beats_placebo_and_ensemble_null"] is True and dec["decision_table"]["POSITIVE"] is True
    assert dec["criteria"]["toys"]["beats_placebo_and_ensemble_null"] is False
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
    assert dec["decision"] == "INDETERMINATE" and dec["next_item_no_loss"]["no_loss"] is False
    _, dec, _ = run(tmp_path, rated(), rated(**WEAK), sports(raw=0.20, raw_like=0.21, mirror=0.20), out="o2")
    assert dec["decision"] == "INDETERMINATE" and dec["next_item_no_loss"]["loss_vs_raw"] == 0.0


def test_no_loss_uses_the_paired_common_support_delta(tmp_path):
    # arm-level NDCG says mirror gains 0.05 (its censored negatives fell below the positive); on the candidates
    # scored in both arms mirror loses 0.01 -> no-loss fails
    sp = sports(raw=0.20, raw_like=0.20, mirror=0.25, paired=(-0.01, -0.01), unscored=7)
    _, dec, od = run(tmp_path, rated(), rated(**WEAK), sp)
    nl = dec["next_item_no_loss"]
    assert abs(nl["arm_level"]["loss_vs_raw"] + 0.05) < 1e-12 and nl["loss_vs_raw"] == 0.01
    assert nl["no_loss"] is False and nl["unscored_candidates_present"] and dec["decision"] == "INDETERMINATE"
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
    assert dec["decision"] == "INDETERMINATE"


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
    assert dec["decision"] == "INDETERMINATE" and dec["any_domain_dUAUC_lo_gt_0"]


def test_sensitivity_to_sd_estimator_popularity_and_exact_ties(tmp_path):
    # primary gates on SD_pair_df (divisor N - n_users); divisor N is the reported alternative
    _, dec, _ = run(tmp_path, rated(sd=0.19, sd_df=0.205), rated(**WEAK), sports(mirror=0.196))
    s = dec["sensitivity"]
    assert dec["decision"] == "POSITIVE" and s["SD_pair_divisor_N"]["decision"] == "INDETERMINATE"
    assert dec["criteria"]["ml1m"]["SD_pair"] == 0.205 and not s["all_agree_with_primary"]
    _, dec, _ = run(tmp_path, rated(corr_prior=0.15), rated(**WEAK), sports(mirror=0.196), out="o2")
    s = dec["sensitivity"]
    assert dec["decision"] == "POSITIVE" and s["prior_popularity"]["decision"] == "INDETERMINATE"
    assert dec["criteria"]["ml1m"]["reference_corr_a_logpop_prior_pop"] == 0.15
    _, dec, _ = run(tmp_path, rated(), rated(**WEAK), sports(mirror=0.196, exact_loss=(0.006, 0.004)), out="o3")
    s = dec["sensitivity"]
    # primary no-loss uses the exact tie-group NDCG; the plug-in expected rank is the reported alternative
    assert dec["decision"] == "INDETERMINATE" and s["ndcg_plugin_expected_rank"]["decision"] == "POSITIVE"
    assert s["SD_pair_divisor_N"]["decision"] == s["prior_popularity"]["decision"] == "INDETERMINATE"


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


# ------------------------------------------------------------------ amendment 2: closed table + ensemble null
def test_pilot1_hole_is_indeterminate_not_positive_and_without_handoff(tmp_path):
    # Pilot-1 decision.json landed here: gate passed in the scenario, positive = negative = null = False
    _, dec, _ = run(tmp_path, rated(sd=0.15, corr=0.30, lo=0.002, hi=0.04, v=0.5), rated(**WEAK), sports())
    assert dec["conditions"] == {"positive": False, "negative": False, "null": False}
    assert dec["decision"] == "INDETERMINATE" and dec["handoff"] is None            # no automatic B9/A9
    assert [k for k, v in dec["decision_table"].items() if v] == ["INDETERMINATE"]


def test_positive_requires_beating_the_ensemble_null_where_the_placebo_is_beaten(tmp_path):
    # ml1m beats the placebo but not the ensemble null; toys beats the ensemble null but not the placebo
    ml, toys = rated(elo=-0.001, ehi=0.02), rated(**dict(WEAK, elo=0.01, ehi=0.03))
    _, dec, _ = run(tmp_path, ml, toys, sports(mirror=0.196))
    c = dec["criteria"]
    assert c["ml1m"]["dUAUC_lo_gt_0"] is True and c["ml1m"]["ensemble_null_lo_gt_0"] is False
    assert c["toys"]["dUAUC_lo_gt_0"] is False and c["toys"]["ensemble_null_lo_gt_0"] is True
    assert dec["any_domain_dUAUC_lo_gt_0"] is True and dec["any_domain_beats_placebo_and_ensemble_null"] is False
    assert dec["decision"] == "INDETERMINATE" and dec["handoff"] is None
    assert c["ml1m"]["dUAUC_mirror_minus_ensemble_null"]["lo"] == -0.001          # the value is recorded
    # the same domain beating both is enough even when acquiescence comes from the other domain (registered "some")
    _, dec, _ = run(tmp_path, rated(elo=-0.001), rated(sd=0.15, corr=0.05, lo=0.001, hi=0.03, elo=0.002),
                    sports(mirror=0.196), out="o2")
    assert dec["decision"] == "POSITIVE"
    # ensemble null missing where the placebo is beaten: undetermined -> INCOMPLETE, listed as missing
    _, dec, _ = run(tmp_path, rated(elo=None), rated(**WEAK), sports(mirror=0.196), out="o3")
    assert dec["decision"] == "INCOMPLETE" and dec["conditions"]["positive"] is None
    assert dec["missing_inputs"] == ["ml1m.dUAUC_mirror_minus_ensemble_null.lo"]
    # ... but irrelevant (decision determined) where the placebo is not beaten
    _, dec, _ = run(tmp_path, rated(lo=-0.01, hi=0.02, elo=None), rated(**WEAK), sports(), out="o4")
    assert dec["decision"] == "NULL" and dec["missing_inputs"] == ["ml1m.dUAUC_mirror_minus_ensemble_null.lo"]


def _reference_label(cond, gate_pass, null_first):
    """The registered precedence, written independently of decision_table, for fully determined conditions."""
    if gate_pass is not True:
        return "GATE_FAIL_UNINTERPRETABLE"
    for k in ("positive", "null", "negative") if null_first else ("positive", "negative", "null"):
        if cond[k]:
            return k.upper()
    return "INDETERMINATE"


def test_decision_table_maps_every_condition_combination_to_exactly_one_label():
    """Amendment 2 section 0: every combination of the three-valued per-domain conditions (2 rated domains x 5),
    the sports no-loss condition and the gate maps to exactly one label; POSITIVE never co-occurs with a failed or
    unknown ensemble-null condition in the domains that beat the placebo."""
    V = (True, False, None)
    doms = [dict(zip(gate.DOMAIN_CONDITIONS, c)) for c in itertools.product(V, repeat=len(gate.DOMAIN_CONDITIONS))]
    assert len(doms) == 243
    labels = set(gate.LABELS)
    seen, n = Counter(), 0
    for d1 in doms:
        for d2 in doms:
            crit = {"ml1m": d1, "toys": d2}
            ens_beaten_with_gain = any(x["dUAUC_lo_gt_0"] is True and x["ensemble_null_lo_gt_0"] is True
                                       for x in (d1, d2))
            acq = any(x["acquiescence_large_and_pop_linked"] is True for x in (d1, d2))
            for nl in V:
                cond = gate.conditions(crit, nl)
                three = {k: cond[k] for k in ("positive", "negative", "null")}
                determined = None not in three.values()
                for null_first in (True, False):
                    for gp in V:
                        table = gate.decision_table(three, gp, null_first)
                        assert table.keys() == labels and set(map(type, table.values())) == {bool}
                        assert sum(table.values()) == 1
                        label = gate.classify(table)
                        seen[label] += 1
                        n += 1
                        if label == "POSITIVE":   # never with a failed / unknown ensemble null where gain holds
                            assert gp is True and nl is True and ens_beaten_with_gain and acq
                        if determined:
                            assert label == _reference_label(three, gp, null_first)
    assert n == 243 * 243 * 3 * 2 * 3 and set(seen) == set(gate.LABELS)


def test_determined_labels_survive_every_completion_of_the_unknown_conditions():
    """Three-valued soundness: a label other than INCOMPLETE is emitted only if every True/False completion of the
    unknown per-domain and no-loss conditions gives the same label (the gate itself is a given boolean)."""
    rng = random.Random(0)
    V = (True, False, None)
    names = [(d, k) for d in ("ml1m", "toys") for k in gate.DOMAIN_CONDITIONS] + [("sports", "no_loss")]

    def label(vs, null_first):
        crit = {d: {k: vs[(d, k)] for k in gate.DOMAIN_CONDITIONS} for d in ("ml1m", "toys")}
        cond = gate.conditions(crit, vs[("sports", "no_loss")])
        return gate.classify(gate.decision_table({k: cond[k] for k in ("positive", "negative", "null")},
                                                 True, null_first))
    checked = determined = 0
    while checked < 1500:
        vals = {nm: rng.choice(V) for nm in names}
        unknown = [nm for nm, v in vals.items() if v is None]
        if not 1 <= len(unknown) <= 5:
            continue
        checked += 1
        for null_first in (True, False):
            got = label(vals, null_first)
            if got == "INCOMPLETE":
                continue
            determined += 1
            for fill in itertools.product((True, False), repeat=len(unknown)):
                assert label({**vals, **dict(zip(unknown, fill))}, null_first) == got
    assert determined > 300                                                       # the check is not vacuous


def test_handoff_only_for_negative_and_null():
    for lab in gate.LABELS:
        for link in (True, False, None):
            h = gate.handoff_for(lab, link)
            if lab == "NEGATIVE":
                assert h == ("B9/A9" if link else None if link is False else
                             "UNDETERMINED: corr(v or pi, log-pop) missing")
            elif lab == "NULL":
                assert h == gate.NULL_HANDOFF
            else:
                assert h is None                                                  # INDETERMINATE: no handoff


def test_classify_rejects_a_non_exclusive_table():
    import pytest
    with pytest.raises(AssertionError, match="not exhaustive and exclusive"):
        gate.classify({"POSITIVE": True, "NULL": True})
    with pytest.raises(AssertionError):
        gate.classify({lab: False for lab in gate.LABELS})


# ------------------------------------------------------------------ amendment 2 stage 3: the recorded gate
def g6(decision="GATE_PASS", uauc=0.612, e1=True, n_users=1500):
    """gatefix_select confirm gate.json (the fields stage3_gate reads)."""
    return {"stage": "confirm", "decision": decision, "gate_pass": decision == "GATE_PASS", "v_star": "V3",
            "UAUC": uauc, "UAUC_min": 0.60, "E1": {"E1": e1}, "n_users": n_users}


def pilot1(sports_ndcg=0.20934, ok=True):
    """Pilot-1 decision.json: the gate failed on ML-1M, the sports next-item component passed."""
    return {"decision": "GATE_FAIL_UNINTERPRETABLE",
            "gate": {"pass": False, "ml1m_raw_UAUC": 0.5874, "sports_raw_NDCG@10": sports_ndcg,
                     "sports_raw_NDCG@10_min": 0.18632,
                     "input_checks": {"sports": {"panel_type": "next_item", "base_question": "next", "ok": ok}}}}


def run3(tmp_path, ml, toys, sp, g, p1, out="s3"):
    for name, d in (("gate.json", g), ("p1_decision.json", p1)):
        (tmp_path / name).write_text(json.dumps(d), encoding="utf-8")
    paths = []
    for name, d in (("ml1m", ml), ("toys", toys), ("sports", sp)):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        paths.append(str(p))
    od = tmp_path / out
    code = gate.main(["--ml1m", paths[0], "--toys", paths[1], "--sports", paths[2], "--out_dir", str(od),
                      "--stage3_gate", str(tmp_path / "gate.json"),
                      "--pilot1_decision", str(tmp_path / "p1_decision.json")])
    return code, json.loads((od / "decision.json").read_text(encoding="utf-8")), od


def test_stage3_gate_is_the_g6_record_plus_the_pilot1_sports_pass(tmp_path, capsys):
    # sports VALID NDCG 0.15 < 0.18632 (a TEST-event bar) and a re-measured ML-1M UAUC of 0.59 do not gate stage 3
    sp = sports(raw=0.15, raw_like=0.15, mirror=0.146)
    code, dec, od = run3(tmp_path, rated(uauc=0.59), rated(**WEAK), sp, g6(), pilot1())
    g = dec["gate"]
    assert code == 0 and g["pass"] is True and g["source"] == "stage3" and (od / "GATE_PASS").exists()
    assert dec["decision"] == "POSITIVE"                                      # the section 3.1 table is computed
    assert g["rated_component"]["ok"] and g["rated_component"]["UAUC"] == 0.612
    assert g["next_item_component"]["ok"] and g["next_item_component"]["sports_raw_NDCG@10"] == 0.20934
    assert g["context_remeasured"]["ml1m_raw_UAUC"] == 0.59
    assert g["context_remeasured"]["sports_valid_raw_NDCG@10"] == 0.15
    assert dec["inputs"]["stage3_gate"].endswith("gate.json") and "stage 3" in dec["rule"]
    txt = (od / "GATE_PASS").read_text(encoding="utf-8")
    assert "context only" in txt and "not re-tested" in txt and "DECISION: POSITIVE" in capsys.readouterr().out
    # without the stage-3 records the same inputs fail the re-measured gate (the Pilot-1 behaviour is unchanged)
    code, dec, _ = run(tmp_path, rated(uauc=0.59), rated(**WEAK), sp, out="pilot1_mode")
    assert code == 3 and dec["gate"]["source"] == "pilot1" and dec["decision"] == "GATE_FAIL_UNINTERPRETABLE"


def test_stage3_gate_fails_without_a_g6_pass_a_pilot1_sports_pass_or_the_g6_users(tmp_path):
    ok_args = (rated(), rated(**WEAK), sports(mirror=0.196))
    for k, (g, p1) in enumerate([(g6(decision="GATE_FAIL_AFTER_REMEDY(confirm)", uauc=0.59), pilot1()),
                                 (g6(e1=False), pilot1()),                 # a hand-edited record is re-checked
                                 (g6(uauc=0.5999), pilot1()),
                                 (g6(), pilot1(sports_ndcg=0.1863)),       # 0.8 x 0.2329 = 0.18632
                                 (g6(), pilot1(ok=False)),
                                 ({}, pilot1()), (g6(), {})]):
        code, dec, od = run3(tmp_path, *ok_args, g, p1, out=f"o{k}")
        assert code == 3 and dec["decision"] == "GATE_FAIL_UNINTERPRETABLE" and (od / "GATE_FAIL").exists(), k
    code, dec, od = run3(tmp_path, *ok_args, g6(n_users=1683), pilot1(), out="users")
    assert code == 3 and dec["gate"]["ml1m_n_users_equals_g6"] is False
    assert "INPUT CHECK FAILED: ml1m n_users 1500 != G6 gate.json n_users 1683" in \
        (od / "GATE_FAIL").read_text(encoding="utf-8")
    ml = rated()
    ml["panel_type"] = "next_item"                                          # the per-input checks still apply
    code, dec, _ = run3(tmp_path, ml, rated(**WEAK), sports(mirror=0.196), g6(), pilot1(), out="types")
    assert code == 3 and not dec["gate"]["input_checks_ok"]


def test_stage3_flags_go_together(tmp_path):
    import pytest
    for name, d in (("ml1m", rated()), ("toys", rated()), ("sports", sports()), ("g", g6())):
        (tmp_path / f"{name}.json").write_text(json.dumps(d), encoding="utf-8")
    base = ["--ml1m", str(tmp_path / "ml1m.json"), "--toys", str(tmp_path / "toys.json"), "--sports",
            str(tmp_path / "sports.json"), "--out_dir", str(tmp_path / "o")]
    with pytest.raises(SystemExit) as e:
        gate.main(base + ["--stage3_gate", str(tmp_path / "g.json")])
    assert e.value.code == 2 and not (tmp_path / "o").exists()


def test_stage3_reads_the_real_pilot1_decision_record(tmp_path):
    real = ROOT / "outputs" / "confrec_pilot" / "pilot1" / "decision.json"
    if not real.exists():
        import pytest
        pytest.skip("local Pilot-1 decision.json copy not present")
    p1 = json.loads(real.read_text(encoding="utf-8"))
    g = gate.stage3_gate(g6(), p1, rated(), sports(), gate.input_checks({"ml1m": rated(), "toys": rated(),
                                                                          "sports": sports()}))
    assert g["next_item_component"]["ok"] is True and abs(g["next_item_component"]["sports_raw_NDCG@10"] - 0.2093) < 1e-4
    assert g["pass"] is True


def test_domain_criteria_expose_every_decision_condition(tmp_path):
    _, dec, _ = run(tmp_path, rated(), rated(**WEAK), sports())
    for c in dec["criteria"].values():
        assert set(gate.DOMAIN_CONDITIONS) <= set(c)
