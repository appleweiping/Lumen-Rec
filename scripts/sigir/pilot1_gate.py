"""Pilot 1 token-channel sanity gate and the amended section-3.1 decision (idea-stage/triage_verdict.md section 3;
idea-stage/PREREG_AMENDMENT_1.md P1.1, P1.4, P1.5).

    python scripts/sigir/pilot1_gate.py --ml1m <out>/ml1m_rated/pilot_mirror.json \
        --toys <out>/toys_rated/pilot_mirror.json --sports <out>/sports_next_1k/pilot_mirror.json --out_dir <out>

Registered gate (constants unchanged): ml1m arms.raw.UAUC >= 0.60 AND sports arms.raw.NDCG@10 >= 0.8 * 0.2329.
C-CRP v3 on the same events (sports arms.ccrp.NDCG@10 from pilot_mirror --ref_ranks) is printed alongside and
does not gate. The gate also fails when an input is not what it claims to be (ml1m and toys: panel_type rated,
base_question like; sports: panel_type next_item, base_question next). Writes GATE_PASS or GATE_FAIL (content = the
numbers; the other marker is removed) and decision.json with every criterion value, each rewritten only when its
content changes (pilot 3 shares this gate and its analysis step depends on decision.json). Exit 0 on PASS, 3 on FAIL.

Decision (rated domains ml1m and toys; point estimates unless a CI bound is named). SD_pair below is SD_pair_df,
the pooled within-user SD with divisor N - n_users; NDCG is the exact expectation over the positive's tie group:
  POSITIVE   some domain has SD_pair >= 0.20 and corr_a_logpop.est >= 0.20, AND some domain has
             dUAUC(mirror - placebo).lo > 0, AND sports no-loss: loss vs raw(next) and vs raw_like <= 0.005, where
             loss = -dNDCG@10(mirror - arm).est (paired over users, both arms ranked on the candidates finite in
             both; equal to NDCG(arm) - NDCG(mirror) when nothing is censored)
  NULL       some domain has SD_pair >= 0.20 and corr >= 0.20, but no domain has dUAUC lo > 0
  NEGATIVE   SD_pair < 0.10 in every rated domain, or dUAUC(mirror - placebo).hi <= 0 in both; handoff B9/A9 if
             |corr(v, log-pop)| (pair level) >= 0.20 or |corr(pi, log-pop)| (item level) >= 0.20 in some domain
  AMBIGUOUS  otherwise. Precedence POSITIVE > NULL > NEGATIVE > AMBIGUOUS; every condition is recorded.
A failed gate gives GATE_FAIL_UNINTERPRETABLE (nothing below the gate is interpretable; fix the prompt/channel).
Missing or null values are unknown (three-valued logic): a label is emitted only when the available values
determine it, otherwise the decision is INCOMPLETE (e.g. NULL needs the dUAUC CI in both rated domains).
`sensitivity` repeats the decision under the rejected alternatives (SD_pair with divisor N, prior-only popularity,
plug-in expected-rank NDCG, NEGATIVE before NULL where both hold); it reports, it does not decide.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.confrec.stats import strict_json  # noqa: E402

UAUC_MIN, CCRP_NDCG10, NDCG_FRAC = 0.60, 0.2329, 0.8
SD_POS, SD_NEG, CORR_MIN, LOSS_MAX = 0.20, 0.10, 0.20, 0.005
RATED = ("ml1m", "toys")
EXPECT = {"ml1m": ("rated", "like"), "toys": ("rated", "like"), "sports": ("next_item", "next")}
NULL_HANDOFF = "B9/A9 unless pilot 3 shows a(i) is familiarity-driven"


def _get(d, *path) -> float:
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return math.nan
        d = d[k]
    try:
        return float(d)
    except (TypeError, ValueError):
        return math.nan


# three-valued logic: True / False / None (unknown, from a missing or non-finite value)
def _cmp(x: float, op: str, t: float):
    if not math.isfinite(x):
        return None
    return {">=": x >= t, ">": x > t, "<=": x <= t, "<": x < t}[op]


def _all(xs):
    xs = list(xs)
    return False if any(x is False for x in xs) else None if any(x is None for x in xs) else True


def _any(xs):
    xs = list(xs)
    return True if any(x is True for x in xs) else None if any(x is None for x in xs) else False


def _not(x):
    return None if x is None else not x


def input_checks(docs: dict) -> dict:
    out = {}
    for name, d in docs.items():
        got = (d.get("panel_type"), d.get("base_question"))
        out[name] = {"panel_type": got[0], "base_question": got[1], "expected": list(EXPECT[name]),
                     "ok": got == EXPECT[name]}
    return out


def gate(ml: dict, sp: dict, checks: dict) -> dict:
    u, nd, c = _get(ml, "arms", "raw", "UAUC"), _get(sp, "arms", "raw", "NDCG@10"), _get(sp, "arms", "ccrp", "NDCG@10")
    ok = all(x["ok"] for x in checks.values())
    return {"pass": bool(ok and u >= UAUC_MIN and nd >= NDCG_FRAC * CCRP_NDCG10), "input_checks_ok": ok,
            "ml1m_raw_UAUC": u, "ml1m_raw_UAUC_min": UAUC_MIN,
            "sports_raw_NDCG@10": nd, "sports_raw_NDCG@10_min": NDCG_FRAC * CCRP_NDCG10,
            "ccrp_same_events_NDCG@10": c, "ccrp_same_events_0.8x": NDCG_FRAC * c,
            "ccrp_same_events_n": _get(sp, "ref_ranks", "n_joined"),
            "sports_raw_n_events_negative_unscored": _get(sp, "arms", "raw", "n_events_negative_unscored"),
            "input_checks": checks}


def domain_criteria(r: dict, sd_key: str = "SD_pair", prior_pop: bool = False) -> dict:
    sd = _get(r, "acquiescence", sd_key)
    rob = ("popularity_prior_robustness",)
    corr = _get(r, *(rob if prior_pop else ("acquiescence",)), "corr_a_logpop", "est")
    v = _get(r, *(rob if prior_pop else ("valence",)), "corr_v_logpop", "est")
    pi = _get(r, *(rob if prior_pop else ("prior",)), "corr_pi_logpop", "est")
    d = {k: _get(r, "dUAUC_mirror_minus_placebo", k) for k in ("est", "lo", "hi", "n")}
    return {"SD_pair_estimator": sd_key, "popularity": "prior (strictly before)" if prior_pop else "all-time",
            "SD_pair": sd, "corr_a_logpop": corr, "dUAUC_mirror_minus_placebo": d,
            "corr_v_logpop": v, "corr_pi_logpop": pi,
            "reference_SD_pair_df": _get(r, "acquiescence", "SD_pair_df"),
            "reference_corr_a_logpop_prior_pop": _get(r, *rob, "corr_a_logpop", "est"),
            "SD_pair_ge_0.20": _cmp(sd, ">=", SD_POS), "corr_a_logpop_ge_0.20": _cmp(corr, ">=", CORR_MIN),
            "acquiescence_large_and_pop_linked": _all([_cmp(sd, ">=", SD_POS), _cmp(corr, ">=", CORR_MIN)]),
            "SD_pair_lt_0.10": _cmp(sd, "<", SD_NEG),
            "dUAUC_lo_gt_0": _cmp(d["lo"], ">", 0), "dUAUC_hi_le_0": _cmp(d["hi"], "<=", 0),
            "valence_or_prior_pop_linked": _any([_cmp(abs(v), ">=", CORR_MIN), _cmp(abs(pi), ">=", CORR_MIN)])}


def no_loss(sp: dict, tie_exact: bool = False) -> dict:
    def loss(arm):
        d = sp.get(f"dNDCG@10_mirror_minus_{arm}")
        if tie_exact and isinstance(d, dict):
            d = d.get("tie_exact")
        return -_get(d, "est")

    lr, ll = loss("raw"), loss("raw_like")
    nr, nl, nm = (_get(sp, "arms", a, f"NDCG@10{'_tie_exact' if tie_exact else ''}")
                  for a in ("raw", "raw_like", "mirror"))
    uns = {a: {k: _get(sp, "arms", a, k) for k in ("n_events_positive_unscored", "n_events_negative_unscored")}
           for a in ("raw", "raw_like", "mirror", "placebo")}
    out = {"metric": "NDCG@10_tie_exact" if tie_exact else "NDCG@10 at the tie-aware expected rank",
           "loss_vs_raw": lr, "loss_vs_raw_like": ll, "max_loss": LOSS_MAX,
           "no_loss": _all([_cmp(lr, "<=", LOSS_MAX), _cmp(ll, "<=", LOSS_MAX)]),
           "arm_level": {"NDCG@10_raw": nr, "NDCG@10_raw_like": nl, "NDCG@10_mirror": nm,
                         "loss_vs_raw": nr - nm, "loss_vs_raw_like": nl - nm},
           "unscored": uns, "unscored_candidates_present": any(v > 0 for x in uns.values() for v in x.values())}
    for k, v in sp.items():
        if k.startswith("dNDCG@10_") and isinstance(v, dict):
            out[k] = v  # CIs reported, not gated
    return out


def decide(rated: dict, sp: dict, gate_pass: bool, sd_key: str = "SD_pair_df", prior_pop: bool = False,
           tie_exact: bool = True, null_first: bool = True) -> dict:
    crit = {name: domain_criteria(r, sd_key, prior_pop) for name, r in rated.items()}
    loss = no_loss(sp, tie_exact)
    c = list(crit.values())
    acq = _any(x["acquiescence_large_and_pop_linked"] for x in c)
    gain = _any(x["dUAUC_lo_gt_0"] for x in c)
    cond = {"positive": _all([acq, gain, loss["no_loss"]]),
            "negative": _any([_all(x["SD_pair_lt_0.10"] for x in c), _all(x["dUAUC_hi_le_0"] for x in c)]),
            "null": _all([acq, _not(gain)])}
    if not gate_pass:
        decision = "GATE_FAIL_UNINTERPRETABLE"
    else:
        decision = "AMBIGUOUS"
        for k in ("positive", "null", "negative") if null_first else ("positive", "negative", "null"):
            if cond[k] is None:
                decision = "INCOMPLETE"
                break
            if cond[k]:
                decision = k.upper()
                break
    link = _any(x["valence_or_prior_pop_linked"] for x in c)
    handoff = None
    if decision == "NEGATIVE":
        handoff = "B9/A9" if link else "UNDETERMINED: corr(v or pi, log-pop) missing" if link is None else None
    elif decision == "NULL":
        handoff = NULL_HANDOFF
    missing = [f"{name}.{k}" for name, x in crit.items() for k, v in (
        ("SD_pair", x["SD_pair"]), ("corr_a_logpop", x["corr_a_logpop"]),
        ("dUAUC_mirror_minus_placebo.lo", x["dUAUC_mirror_minus_placebo"]["lo"]),
        ("dUAUC_mirror_minus_placebo.hi", x["dUAUC_mirror_minus_placebo"]["hi"])) if not math.isfinite(v)]
    missing += [f"sports.{k}" for k in ("loss_vs_raw", "loss_vs_raw_like") if not math.isfinite(loss[k])]
    return {"decision": decision, "handoff": handoff, "conditions": cond,
            "overlap_negative_and_null": bool(cond["negative"] and cond["null"]),
            "any_domain_acquiescence_large_and_pop_linked": acq, "any_domain_dUAUC_lo_gt_0": gain,
            "missing_inputs": missing, "criteria": crit, "next_item_no_loss": loss}


def _write_if_changed(path: Path, text: str) -> None:
    """Rewrite only when the content differs: an unchanged gate keeps its mtime, so pilot 3's analysis step in
    run_pilot2_3.sh (which depends on decision.json) is not re-run by a mere rerun of pilot 1."""
    if not (path.exists() and path.read_text(encoding="utf-8") == text):
        path.write_text(text, encoding="utf-8")


# Primary choices (orchestrator sign-off 2026-10-02, before any pilot data): SD_pair_df (textbook pooled within-user
# SD, divisor N - n_users), all-time popularity (amendment C2), exact tie-group NDCG, NULL before NEGATIVE when both
# hold (the registered NULL row describes that case). The alternatives below are reported, they do not decide.
SENSITIVITY = {"SD_pair_divisor_N": dict(sd_key="SD_pair"), "prior_popularity": dict(prior_pop=True),
               "ndcg_plugin_expected_rank": dict(tie_exact=False), "NEGATIVE_before_NULL": dict(null_first=False)}


def sensitivity(rated: dict, sp: dict, gate_pass: bool, primary: str) -> dict:
    out = {}
    for name, kw in SENSITIVITY.items():
        d = decide(rated, sp, gate_pass, **kw)
        out[name] = {"decision": d["decision"], "handoff": d["handoff"]}
    out["all_agree_with_primary"] = all(v["decision"] == primary for v in out.values())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml1m", required=True)
    ap.add_argument("--toys", required=True)
    ap.add_argument("--sports", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    ml, toys, sp = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a.ml1m, a.toys, a.sports))
    checks = input_checks({"ml1m": ml, "toys": toys, "sports": sp})
    g = gate(ml, sp, checks)
    lines = [f"INPUT CHECK FAILED: {n} panel_type={x['panel_type']!r} base_question={x['base_question']!r} "
             f"(expected {x['expected'][0]!r}/{x['expected'][1]!r}) -> rerun pilot_mirror on the right panel/--base_q"
             for n, x in checks.items() if not x["ok"]]
    lines += [f"ml1m arms.raw.UAUC = {g['ml1m_raw_UAUC']:.4f} (registered >= {UAUC_MIN:.2f})",
              f"sports arms.raw.NDCG@10 = {g['sports_raw_NDCG@10']:.4f} "
              f"(registered >= {NDCG_FRAC} x {CCRP_NDCG10} = {NDCG_FRAC * CCRP_NDCG10:.4f})"]
    if math.isfinite(g["ccrp_same_events_NDCG@10"]):
        lines.append(f"reference only: C-CRP v3 on the same {g['ccrp_same_events_n']:.0f} events NDCG@10 = "
                     f"{g['ccrp_same_events_NDCG@10']:.4f} -> 0.8 x = {g['ccrp_same_events_0.8x']:.4f}")
    if g["sports_raw_n_events_negative_unscored"] > 0:
        lines.append(f"note: {g['sports_raw_n_events_negative_unscored']:.0f} sports events have unscored negatives "
                     "(ranked below the positive in arms.raw.NDCG@10)")
    lines.append("GATE: " + ("PASS" if g["pass"] else "FAIL -> fix prompt/channel (or the inputs) before "
                                                       "interpreting anything below"))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("GATE_FAIL" if g["pass"] else "GATE_PASS")).unlink(missing_ok=True)
    _write_if_changed(out / ("GATE_PASS" if g["pass"] else "GATE_FAIL"), "\n".join(lines) + "\n")
    rated = dict(zip(RATED, (ml, toys)))
    dec = decide(rated, sp, g["pass"])
    sens = sensitivity(rated, sp, g["pass"], dec["decision"])
    dec = {"decision": dec.pop("decision"), "gate": g, **dec, "sensitivity": sens,
           "input_summary": {n: {"n_users": _get(d, "n_users"), "n_rows": _get(d, "n_rows"),
                                 "n_events": _get(d, "next_item", "n_events"), "panel_join": d.get("panel_join")}
                             for n, d in (("ml1m", ml), ("toys", toys), ("sports", sp))},
           "inputs": {"ml1m": a.ml1m, "toys": a.toys, "sports": a.sports},
           "rule": "triage_verdict.md section 3.1 as amended by PREREG_AMENDMENT_1.md P1.1/P1.4/P1.5"}
    _write_if_changed(out / "decision.json", json.dumps(strict_json(dec), indent=2, allow_nan=False))
    print("\n".join(lines))
    print(f"DECISION: {dec['decision']}" + (f"  handoff: {dec['handoff']}" if dec["handoff"] else ""))
    if dec["missing_inputs"]:
        print(f"WARNING missing inputs: {dec['missing_inputs']}")
    if not dec["sensitivity"]["all_agree_with_primary"]:
        print("WARNING decision is sensitive to: " + ", ".join(
            f"{k} -> {v['decision']}" for k, v in dec["sensitivity"].items()
            if isinstance(v, dict) and v["decision"] != dec["decision"]))
    return 0 if g["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
