"""FT-B reading: section 3 (endpoints) and section 4 (the reading rule) of idea-stage/PREREG_AMENDMENT_3_ADDENDUM_13.md applied to the
stored outputs of FT-B, FT-S and FT-N. EXPLORATORY: no hypothesis, no Holm family; every output says so; nothing here is a direction word
beyond the registered labels. CPU only, deterministic (2,000 user resamples, seed 0, no clock, host or path in the output).

    python -m src.confrec.ftb_reading build --split outputs/confrec/ftgrid/panels/ml1m/ftgrid_split.json \
        --panels outputs/confrec/ftgrid/panels/ml1m --grid_scores outputs/confrec/ftgrid/scores --root outputs/confrec/ftgrid_ftb \
        --raw data/raw --refs outputs/confrec/ftb_reading/ml1m_refs.json --out outputs/confrec/ftb_reading/ml1m.json [--n_boot 2000] [--seed 0]

Inputs: the like and swap passes of FT-B (root/scores/b/ml1m/s0-s2 = the adapters b0-b2), FT-S (root/scores/r/ml1m, r0-r2) and FT-N
(root/scores/n/ml1m, n0-n1), the registered zero-shot and SFT passes (grid_scores/ml1m: zeroshot, s0-s2; reference arms, optional), the
registered panels and split, the CF references (the cache --refs, or computed once from --raw by ftgrid_report.compute_refs through
ftgrid_extra.load_inputs), and, when present, the arms' manifests (root/ftb/ml1m), adapter provenance (ftb.json) and stored reports
(root/report/<tag>/ml1m.json) and extra-analysis files (root/extra/<tag>/ml1m.json), which the reading cross-checks. A pass that failed
E1 is excluded exactly as the report excludes it (listed, never replaced), which leaves its arm incomplete.

Estimators (imported read-only from ftgrid_report and ftgrid_extra, none re-implemented; the per-user arrangement of the stackers is
checked at run time against the registered functions on the same rows):
  * G (E-D) = dUAUC(M2 - M1) of the cross-fitted pooled stackers (ftgrid_report.crossfit, user_aucs; fold A = stats.user_halves(S_d, 0)),
    G_CF = dUAUC(M3 - M0); G_wu (E-W) = ftgrid_extra.wu_regime on the same rows (user-centred features, K = 20 splits);
  * the paired contrasts are mean-over-users differences of per-user quantities, resampled with ONE set of draws per block
    (ftgrid_extra.Cols: ftgrid_report.mean_draws, 2,000 user resamples, seed 0), so the contrast of two arms is the difference of columns
    that were resampled together; seed means are the per-user means over the arm's seeds (the registered seed-averaged convention);
  * the stack gain of an arm = per-user AUC of the cross-fitted stacker [q-hat, confidence of the arm] minus the per-user AUC of q-hat, on
    E-D's rows and fold; the stack gain of MF = the same with the temporal MF score in place of the confidence (the personal-residual
    variant, [q-hat, MF residual], is listed beside it); UAUC / contrasts on E-A's TEST rows through ftgrid_report.uauc_models and
    contrast_models; the item-prior share through ftgrid_report.shares_block and the e-share through ftgrid_extra.eshare_block.
Rows: every statistic of a block is computed on the rows finite for every model of every arm that is present (and for q-hat and MF), so
the arms, the seeds and the contrasts of a block use the same users and rows (the registered "identical users and rows"); the rows each
arm's own report would use are compared with them and recorded.

Reading rule (section 4; checked in this order; intervals are the 95% percentile intervals of the user bootstrap, closed):
  INVALID_PROBE  iff G of FT-N (mean over its two seeds) has an interval above 0 (lo > 0) or its seed mean of G_wu has one: the probe
                 is invalid and FT-B is reported without a label;
  PERSONAL_EVIDENCE_LEARNABLE  iff for FT-B (mean over its three seeds) G > 0 with lo > 0, G(FT-B) - G(FT-S) > 0 with lo > 0 and
                 G(FT-B) - G(FT-N) > 0 with lo > 0, all three seeds of G(FT-B) positive and the mean above 2 sigma_seed (the registered
                 P1 conditions all_seeds_positive and mean_gt_2_sigma_seed of ftgrid_report.p1_decision on FT-B's per-seed G);
  NOT_LEARNED    iff -0.01 <= lo and hi <= 0.01 for G(FT-B) and the interval of G(FT-B) - G(FT-S) contains 0 (lo <= 0 <= hi);
  INCONCLUSIVE   otherwise.
INCOMPLETE is not a label of the addendum: it is the state in which the rule cannot be applied (a seed of an arm missing or excluded, never
replaced; fewer than 150 users, the minimum n of Amendment 3 section 1; an input that is not the recorded one); no label is given then.
The stack gain, UAUC and the shares are described with their intervals and are never a label.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.confrec import forensics as fx
from src.confrec import ftgrid_extra as fe
from src.confrec import ftgrid_report as fr
from src.confrec.stats import strict_json

NAN = float("nan")
SPEC = "idea-stage/PREREG_AMENDMENT_3_ADDENDUM_13.md sections 3 and 4 (FT-B, FT-S, FT-N; exploratory)"
FORMAT = "ftb_reading_v1"
STATUS = "exploratory"
DOMAIN = "ml1m"
N_BOOT, SEED = fr.N_BOOT, fr.SEED            # 2,000 user resamples, seed 0 (Amendment 3 section 3)
EQUIV = 0.01                                 # section 4: NOT_LEARNED needs the interval of G(FT-B) within [-0.01, +0.01]
MIN_N = fr.MIN_N                             # Amendment 3 section 1: fewer than 150 users is descriptive
ARMS = ("FT-B", "FT-S", "FT-N")
TAGS = {"FT-B": "b", "FT-S": "r", "FT-N": "n"}
SEEDS = {"FT-B": (0, 1, 2), "FT-S": (0, 1, 2), "FT-N": (0, 1)}
REFERENCE_ARMS = ("ZS", "SFT")
LABELS = ("PERSONAL_EVIDENCE_LEARNABLE", "NOT_LEARNED", "INCONCLUSIVE")
INVALID_PROBE, INCOMPLETE = "INVALID_PROBE", "INCOMPLETE"
WORDING = ("exploratory (addendum 13): no hypothesis, no Holm family, no claim-admission role; a result enters the abstract or a claim "
           "beyond ML-1M only if it is replicated on the second backbone (the claim-admission rule of addendum 6 section 9); the "
           "P1, E-B and Holm blocks of the per-arm ftgrid_report files of this program are not registered results and are never cited")
RULE = ("INVALID_PROBE iff lo(G of FT-N, seed mean) > 0 or lo(G_wu of FT-N, seed mean) > 0; else PERSONAL_EVIDENCE_LEARNABLE iff G(FT-B) > 0 "
        "with lo > 0, G(FT-B) - G(FT-S) > 0 with lo > 0, G(FT-B) - G(FT-N) > 0 with lo > 0, all three seeds of G(FT-B) positive and its "
        "mean above 2 sigma_seed; else NOT_LEARNED iff the interval of G(FT-B) lies within [-0.01, +0.01] and the interval of "
        "G(FT-B) - G(FT-S) contains 0; else INCONCLUSIVE (checked in this order)")
CSV_COLS = ("block", "arm", "model", "statistic", "estimate", "lo", "hi", "p", "n_users", "n_pairs", "descriptive_min_n", "note")
TOL = 1e-12                                  # the run-time reproduction of the registered estimators


# ---------------------------------------------------------------- the reading rule (a pure function of the numbers)
def _fin(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and math.isfinite(float(x))


def usable(rec) -> bool:
    """An estimate with a finite interval on at least MIN_N users (fewer is descriptive, Amendment 3 section 1)."""
    return (isinstance(rec, dict) and all(_fin(rec.get(k)) for k in ("est", "lo", "hi")) and not rec.get("descriptive_min_n")
            and (rec.get("n_users") is None or int(rec["n_users"]) >= MIN_N))


def read_rule(inp: dict) -> dict:
    """Section 4 of addendum 13 on plain numbers.

    inp: n_seeds {arm: seeds with a usable like and swap pass}; G {arm: rec of the seed mean of G (E-D)}; G_wu {arm: rec of the seed mean
    of G_wu (E-W)}; per_seed_G_B [G of b0, b1, b2]; contrast {"B_minus_S": rec, "B_minus_N": rec}. A rec holds est, lo, hi (and
    n_users, descriptive_min_n). Returns the label (one of LABELS, INVALID_PROBE, or INCOMPLETE when the rule cannot be applied) and
    every check in the order it was made."""
    n_seeds, G, Gwu, con = inp.get("n_seeds") or {}, inp.get("G") or {}, inp.get("G_wu") or {}, inp.get("contrast") or {}
    out = {"rule": RULE, "equivalence_margin": EQUIV, "min_n": MIN_N, "checks": []}
    checks = out["checks"]

    def done(label: str, why=None):
        out["label"] = label
        if why:
            out["reasons"] = why
        return out

    # 1. the probe: FT-N must show no personal evidence
    miss = []
    if n_seeds.get("FT-N") != len(SEEDS["FT-N"]):
        miss.append(f"FT-N has {n_seeds.get('FT-N')} usable seeds of {len(SEEDS['FT-N'])}")
    for what, rec in (("G", G.get("FT-N")), ("G_wu", Gwu.get("FT-N"))):
        if not usable(rec):
            miss.append(f"the seed mean of {what} of FT-N has no usable interval (missing, or fewer than {MIN_N} users)")
    if miss:
        checks.append({"name": "probe_valid", "decided": False, "inputs_missing": miss})
        return done(INCOMPLETE, miss)
    inv_g, inv_wu = G["FT-N"]["lo"] > 0, Gwu["FT-N"]["lo"] > 0
    checks.append({"name": "probe_valid", "decided": True, "G_N_interval_above_0": bool(inv_g),
                   "G_wu_N_interval_above_0": bool(inv_wu), "holds": not (inv_g or inv_wu),
                   "G_N": {k: G["FT-N"][k] for k in ("est", "lo", "hi")}, "G_wu_N": {k: Gwu["FT-N"][k] for k in ("est", "lo", "hi")}})
    if inv_g or inv_wu:
        return done(INVALID_PROBE)

    # 2. the labels need the three FT-B and the three FT-S seeds and the intervals they are read on
    per = [x for x in (inp.get("per_seed_G_B") or [])]
    miss = []
    for arm in ("FT-B", "FT-S"):
        if n_seeds.get(arm) != len(SEEDS[arm]):
            miss.append(f"{arm} has {n_seeds.get(arm)} usable seeds of {len(SEEDS[arm])}")
    for what, rec in (("G(FT-B)", G.get("FT-B")), ("G(FT-B) - G(FT-S)", con.get("B_minus_S")), ("G(FT-B) - G(FT-N)", con.get("B_minus_N"))):
        if not usable(rec):
            miss.append(f"{what} has no usable interval (missing, or fewer than {MIN_N} users)")
    if len(per) != len(SEEDS["FT-B"]) or not all(_fin(v) for v in per):
        miss.append("the three per-seed values of G(FT-B) are not all finite")
    if miss:
        checks.append({"name": "labels", "decided": False, "inputs_missing": miss})
        return done(INCOMPLETE, miss)
    g, bs, bn = G["FT-B"], con["B_minus_S"], con["B_minus_N"]
    p1 = fr.p1_decision([float(v) for v in per], g, n_registered=len(SEEDS["FT-B"]))["conditions"]   # the registered seed conditions
    cond = {"G_FT_B_gt_0_interval_above_0": bool(g["est"] > 0 and g["lo"] > 0),
            "G_FT_B_minus_FT_S_gt_0_interval_above_0": bool(bs["est"] > 0 and bs["lo"] > 0),
            "G_FT_B_minus_FT_N_gt_0_interval_above_0": bool(bn["est"] > 0 and bn["lo"] > 0),
            "all_three_seeds_of_G_FT_B_positive": bool(p1["all_seeds_positive"]),
            "G_FT_B_mean_above_2_sigma_seed": bool(p1["mean_gt_2_sigma_seed"])}
    learnable = all(cond.values())
    checks.append({"name": "PERSONAL_EVIDENCE_LEARNABLE", "decided": True, "holds": learnable, "conditions": cond,
                   "G_FT_B": {k: g[k] for k in ("est", "lo", "hi")}, "G_FT_B_per_seed": [float(v) for v in per],
                   "B_minus_S": {k: bs[k] for k in ("est", "lo", "hi")}, "B_minus_N": {k: bn[k] for k in ("est", "lo", "hi")}})
    if learnable:
        return done("PERSONAL_EVIDENCE_LEARNABLE")
    within = bool(-EQUIV <= g["lo"] and g["hi"] <= EQUIV)
    contains_0 = bool(bs["lo"] <= 0 <= bs["hi"])
    checks.append({"name": "NOT_LEARNED", "decided": True, "holds": within and contains_0,
                   "G_FT_B_interval_within_equivalence_margin": within, "G_FT_B_minus_FT_S_interval_contains_0": contains_0})
    if within and contains_0:
        return done("NOT_LEARNED")
    return done("INCONCLUSIVE")


# ---------------------------------------------------------------- the stored inputs
def _ns(a, scores_root, models: str, raw) -> argparse.Namespace:
    """The arguments ftgrid_extra.load_inputs reads (the loader of the registered report's inputs: E1 exclusion, panel and adapter
    identity, like logits, swap prior, CF references). root_label main only keeps the loader from flagging a Llama mismatch; the reading
    writes no extra-analysis file."""
    return argparse.Namespace(domain=DOMAIN, split=a.split, panels=a.panels, scores_root=str(scores_root), models=models, raw=raw,
                              refs=a.refs, n_boot=a.n_boot, seed=a.seed, root_label="main", pilot_log=None)


def adapter_problems(scores_dir, expected: dict) -> list:
    """The like and swap passes of each model must have scored the adapter the model stands for (the last component of report.json's
    lora): a model of the FT-B tree scored with another arm's adapter, or with a registered one, is a mix-up of roots."""
    out = []
    for model, name in expected.items():
        for kind in ("like", "swap"):
            p = Path(scores_dir) / model / kind / "report.json"
            if not p.is_file():
                continue
            try:
                rep = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            lora = rep.get("lora") or (rep.get("config") or {}).get("lora")
            last = str(lora or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            if last != name:
                out.append(f"{model}/{kind}: scored with the adapter {lora!r}, not {name}")
    return out


def _load(a, scores_root, models: list, raw, label: str, problems: list):
    """One load through ftgrid_extra.load_inputs. The raw data is read only when no reference cache matches: a load without it that
    finds no references is repeated with it (q-hat_T, which the loader computes whenever raw is given, is not needed here)."""
    X = fe.load_inputs(_ns(a, scores_root, ",".join(models), None if raw is None or _cache_exists(a) else raw))
    if X.refs is None and raw is not None and _cache_exists(a):
        X = fe.load_inputs(_ns(a, scores_root, ",".join(models), raw))
    problems += [f"{label}: {p}" for p in X.problems]
    if X.backbone is not None and not fe._is_qwen(X.backbone):
        problems.append(f"{label}: backbone {X.backbone} is not Qwen3-8B (addendum 13 runs Qwen3-8B only)")
    return X


def _cache_exists(a) -> bool:
    return bool(a.refs) and Path(a.refs).is_file()


def load_world(a, problems: list) -> SimpleNamespace:
    """Every arm's inputs through the registered loader, the models of each arm that have a usable like and swap pass, and the
    registered references. FT-B is loaded first, with --raw: it writes the CF reference cache that the other loads read (without a
    cache path every load computes the references itself)."""
    root = Path(a.root)
    arms, mods, Xs, excluded = {}, {}, {}, {}
    raw = a.raw
    for arm in ARMS:
        tag = TAGS[arm]
        ids = [f"{tag}{k}" for k in SEEDS[arm]]
        mnames = [f"s{k}" for k in SEEDS[arm]]
        sdir = root / "scores" / tag
        X = _load(a, sdir, mnames, raw, arm, problems)
        raw = None if a.refs else a.raw
        problems += [f"{arm}: {p}" for p in adapter_problems(sdir / DOMAIN, dict(zip(mnames, ids)))]
        Xs[arm] = X
        mods.update({i: (arm, m) for i, m in zip(ids, mnames)})
        excluded[arm] = X.excluded
        arms[arm] = [i for i, m in zip(ids, mnames) if m in X.L and m in X.PI]
    if a.grid_scores and (Path(a.grid_scores) / DOMAIN).is_dir():
        X = _load(a, a.grid_scores, ["zeroshot", "s0", "s1", "s2"], raw, "registered", problems)
        problems += [f"registered: {p}" for p in adapter_problems(Path(a.grid_scores) / DOMAIN, {"s0": "s0", "s1": "s1", "s2": "s2"})]
        Xs["REG"] = X
        excluded["REG"] = X.excluded
        for arm, ids, mnames in (("ZS", ["zs"], ["zeroshot"]), ("SFT", ["sft0", "sft1", "sft2"], ["s0", "s1", "s2"])):
            mods.update({i: ("REG", m) for i, m in zip(ids, mnames)})
            arms[arm] = [i for i, m in zip(ids, mnames) if m in X.L and m in X.PI]
    return SimpleNamespace(arms=arms, mods=mods, X=Xs, excluded=excluded)


def same_world(world, problems: list) -> None:
    """The arms were read against the same panels and the same CF references."""
    ref = next(iter(world.X.values()))
    for label, X in world.X.items():
        for f in ("y", "uc", "test", "sd_test", "cal", "fold_a"):
            if not np.array_equal(getattr(X.cx, f), getattr(ref.cx, f)):
                problems.append(f"{label}: the panel context differs from the others ({f}): the arms were read against different panels")
        if (X.refs is None) != (ref.refs is None) or (X.refs is not None and not np.array_equal(
                X.refs["arrays"]["q_hat"], ref.refs["arrays"]["q_hat"], equal_nan=True)):
            problems.append(f"{label}: the CF references differ from the others")


def present_ids(world) -> list:
    return [i for arm in ARMS + REFERENCE_ARMS for i in world.arms.get(arm, [])]


def _vec(world, i: str, kind: str) -> np.ndarray:
    label, m = world.mods[i]
    X = world.X[label]
    return {"L": X.L, "PI": X.PI, "PA": X.PA, "PB": X.PB}[kind][m]


def _masked(world, ids, R, kinds=("L", "PI", "PA", "PB")) -> dict:
    return {k: {i: np.where(R, _vec(world, i, k), NAN) for i in ids} for k in kinds}


# ---------------------------------------------------------------- E-D's rows: G, the stack gain and the paired contrasts
def per_user_aucs(cx, q, mfres, mf, models: dict, R) -> dict:
    """{key: {user: AUC}} on the rows R: the cross-fitted stackers of E-D (M0, M3 shared; M1, M2 per model; features in the order of
    ftgrid_report.STACKERS) through ftgrid_report.crossfit and user_aucs, the stackers [q-hat, L] and [q-hat, MF score] of the stack
    gain, and q-hat itself."""
    y, uc, fold = cx.y, cx.uc, cx.fold_a
    common = {"q_hat": q, "mf_residual": mfres}
    pu = {}
    for k in ("M0", "M3"):
        pu[k] = fr.user_aucs(fr.crossfit(np.column_stack([common[f] for f in fr.STACKERS[k]]), y, fold, R), y, uc, R)
    pu["QMF"] = fr.user_aucs(fr.crossfit(np.column_stack([q, mf]), y, fold, R), y, uc, R)
    pu["q"] = fr.user_aucs(q, y, uc, R)
    for i, (L, PI) in models.items():
        f = {**common, "pi": PI, "e_hat": L - PI}
        for k in ("M1", "M2"):
            pu[(i, k)] = fr.user_aucs(fr.crossfit(np.column_stack([f[c] for c in fr.STACKERS[k]]), y, fold, R), y, uc, R)
        pu[(i, "QL")] = fr.user_aucs(fr.crossfit(np.column_stack([q, L]), y, fold, R), y, uc, R)
    return pu


def sd_columns(world, cx, q, mfres, mf, R, n_boot: int, seed: int) -> SimpleNamespace:
    """The per-user columns of E-D's rows, resampled together (one set of user draws): G of every model and its seed mean per arm,
    G_CF, the stack gains (arm and MF), and the paired differences."""
    ids = present_ids(world)
    pu = per_user_aucs(cx, q, mfres, mf, {i: (_vec(world, i, "L"), _vec(world, i, "PI")) for i in ids}, R)
    keys = sorted(set.intersection(*(set(d) for d in pu.values())))
    nu, npairs = len(keys), fr._rows_of_users(cx.uc, R, keys)
    c = fe.Cols(nu)

    def arr(d):
        return np.array([d[u] for u in keys], float).reshape(nu)
    q_u = arr(pu["q"])
    c.add("G_CF", arr(pu["M3"]) - arr(pu["M0"]))
    c.add("stack_MF", arr(pu["QMF"]) - q_u)
    c.add("stack_MF_residual", arr(pu["M3"]) - q_u)
    for i in ids:
        c.add(("G", i), arr(pu[(i, "M2")]) - arr(pu[(i, "M1")]))
        c.add(("stack", i), arr(pu[(i, "QL")]) - q_u)

    def col(key):
        return c.cols[c.idx[key]]
    for arm in ARMS + REFERENCE_ARMS:
        ai = world.arms.get(arm)
        if ai:
            for what in ("G", "stack"):
                c.add((what, "mean", arm), np.mean([col((what, i)) for i in ai], 0))
            c.add(("contrast", "stack_minus_MF", arm), col(("stack", "mean", arm)) - col("stack_MF"))
    for x, y_ in (("FT-B", "FT-S"), ("FT-B", "FT-N")):
        if world.arms.get(x) and world.arms.get(y_):
            c.add(("contrast", "G", x, y_), col(("G", "mean", x)) - col(("G", "mean", y_)))
    c.draws(n_boot, seed)
    return SimpleNamespace(cols=c, keys=keys, n_users=nu, n_pairs=npairs, ids=ids)


def reproduction(world, cx, q, mfres, R, sd, n_boot: int, seed: int) -> dict:
    """This module's per-user arrangement against ftgrid_report.stacker_block on the same rows: G of every model, the seed mean of
    each arm and G_CF, with their intervals, equal to 1e-12 (the same estimators, the same keys, the same user resamples)."""
    c, out = sd.cols, {}
    for arm in ARMS + REFERENCE_ARMS:
        ai = world.arms.get(arm)
        if not ai:
            continue
        mk = _masked(world, ai, R, ("L", "PI"))
        ig = fr.stacker_block(cx, ai, mk["L"], mk["PI"], q, mfres, R, n_boot, seed, len(ai))
        pairs = [(ig["per_model"][i]["G"], c.one(("G", i), sd.n_pairs, contrast=True)) for i in ai]
        pairs += [(ig["G_mean_over_seeds"], c.one(("G", "mean", arm), sd.n_pairs, contrast=True)),
                  (ig["G_CF"], c.one("G_CF", sd.n_pairs, contrast=True))]
        out[arm] = bool(all(_fin(r[k]) and _fin(m[k]) and abs(r[k] - m[k]) <= TOL for r, m in pairs for k in ("est", "lo", "hi")))
    return out


def arm_block(world, arm: str, cx, refs, R, folds, sd, n_boot: int, seed: int, own_equal: dict) -> dict:
    """Everything one arm reports on E-D's rows: G (E-D) and G_wu (E-W) per seed and as the seed mean with their intervals and the
    upper 95% bound, the stack gain, the item-prior share and the e-share."""
    ai = world.arms[arm]
    n_reg = len(SEEDS[arm]) if arm in SEEDS else len(ai)
    c = sd.cols
    own = cx.sd_test & np.isfinite(refs["arrays"]["q_hat"]) & np.isfinite(refs["arrays"]["mf_residual"])
    for i in ai:
        own &= np.isfinite(_vec(world, i, "L")) & np.isfinite(_vec(world, i, "PI"))
    own_equal[arm] = bool(np.array_equal(own, R))
    g = c.models({i: ("G", i) for i in ai}, sd.n_pairs, n_reg, True)
    sg = c.models({i: ("stack", i) for i in ai}, sd.n_pairs, n_reg, True)
    mk = _masked(world, ai, R)
    xs = SimpleNamespace(cx=cx, L=mk["L"], PI=mk["PI"], refs=refs, sd_ids=cx.sd_ids, n_boot=n_boot, seed=seed)
    ew, _, _ = fe.wu_regime(xs, "ZS" if arm == "ZS" else "FT", {"models": ai, "missing_or_excluded": [], "n_registered": n_reg}, folds, {}, {})
    pm = {i: (mk["PA"][i], mk["PB"][i], mk["L"][i], mk["PI"][i]) for i in ai}
    shares = fr.shares_block(cx.uc, R, pm, n_boot, seed, n_reg)
    esh = fe.eshare_block(cx.uc, R, pm, n_boot, seed, n_reg)
    if ew.get("available") is False:
        gwu = ew
    else:
        gwu = {"per_model": ew["G_wu"]["per_model"], "mean_over_seeds": ew["G_wu"]["mean_over_seeds"], "seeds": ew["G_wu"]["seeds"],
               "rows": ew["rows"], "upper_95_bound_of_the_seed_mean": ew["G_wu"]["mean_over_seeds"]["hi"]}
    return {"available": True, "models": ai, "n_seeds": len(ai), "n_registered": n_reg, "complete": len(ai) == n_reg,
            "rows_equal_to_the_arm_own_rows": own_equal[arm],
            "G": {**g, "upper_95_bound_of_the_seed_mean": g["mean_over_seeds"]["hi"]}, "G_wu": gwu, "stack_gain": sg,
            "item_prior_share": {"per_model": {i: shares["per_model"][i]["item_prior_share"] for i in ai},
                                 "mean_over_seeds": shares["mean_over_seeds"].get("item_prior_share"),
                                 "reading": shares["mean_over_seeds"].get("shares_reading"),
                                 "seeds": shares["seeds"]["item_prior_share"]},
            "e_share": {"per_model": {i: esh["per_model"][i]["e_share"] for i in ai}, "mean_over_seeds": esh["mean_over_seeds"],
                        "seeds": esh["seeds"]}}


def stored_crosscheck(root, world, sd, own_equal: dict) -> dict:
    """The stored per-arm report and extra-analysis file (when present) against this reading: equal to 1e-9 where the arm's own rows are
    the common rows; where they are not, the stored numbers are listed and not compared."""
    out, c = {}, sd.cols
    for arm in ARMS:
        tag = TAGS[arm]
        res = {"report": "not present", "extra": "not present", "rows_equal_to_the_arm_own_rows": own_equal.get(arm)}
        mine = {i: c.one(("G", i), sd.n_pairs)["est"] for i in world.arms.get(arm, [])}
        for what in ("report", "extra"):
            path = Path(root) / what / tag / f"{DOMAIN}.json"
            if not path.is_file():
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                res[what] = "unreadable"
                continue
            if what == "report":
                per = (((doc.get("E_D") or {}).get("FT") or {}).get("information_gain") or {}).get("per_model") or {}
                stored = {f"{tag}{k}": ((per.get(f"s{k}") or {}).get("G") or {}).get("est") for k in SEEDS[arm]}
            else:
                per = (((doc.get("E_W") or {}).get("FT") or {}).get("G_wu") or {}).get("per_model") or {}
                stored = {f"{tag}{k}": (per.get(f"s{k}") or {}).get("est") for k in SEEDS[arm]}
            res[what] = {"stored_sha1": fx.file_sha1(path), "stored": stored}
            if what == "report":
                res[what]["G_equals_this_reading"] = (None if not own_equal.get(arm) else
                                                      bool(all(_fin(stored.get(i)) and abs(stored[i] - mine[i]) <= 1e-9 for i in mine)))
        out[arm] = res
    return out


# ---------------------------------------------------------------- E-A's rows: UAUC and its paired contrasts
def uauc_block(world, cx, q, mf, warm, R_A, n_boot: int, seed: int) -> dict:
    ids = present_ids(world)
    scores = {i: _vec(world, i, "L") for i in ids}
    out = {"available": True, "per_arm": {}, "paired": {}}
    for arm in ARMS + REFERENCE_ARMS:
        ai = world.arms.get(arm)
        if ai:
            n_reg = len(SEEDS[arm]) if arm in SEEDS else len(ai)
            u = fr.uauc_models({i: scores[i] for i in ai}, cx.y, cx.uc, R_A, n_boot, seed, extra={"q_hat": q, "mf": mf}, n_registered=n_reg)
            out["rows"] = u["rows"]
            out["per_arm"][arm] = {k: u[k] for k in ("per_model", "mean_over_seeds", "seeds")}
            out["references"] = u["references"]

    def paired(name, left, right, rows):
        """dUAUC of an arm's seeds against another arm's (seeds paired by number) or against a reference score."""
        li = world.arms.get(left, [])
        if isinstance(right, str):
            ri = world.arms.get(right, [])
            ok = bool(li) and len(li) == len(ri) == len(SEEDS[left])
            pairs = {i: (scores[i], scores[j]) for i, j in zip(li, ri)}
        else:
            ok = len(li) == len(SEEDS[left])
            pairs = {i: (scores[i], right) for i in li}
        if not ok:
            out["paired"][name] = {"available": False, "reason": "an arm is incomplete"}
            return
        r = fr.contrast_models(pairs, cx.y, cx.uc, rows, n_boot, seed, n_registered=len(SEEDS[left]))
        out["paired"][name] = {k: r[k] for k in ("rows", "per_seed", "mean_over_seeds", "seeds")}
    paired("FT-B_minus_FT-S", "FT-B", "FT-S", R_A)
    paired("FT-B_minus_q_hat", "FT-B", q, R_A)
    paired("FT-B_minus_MF_warm", "FT-B", mf, R_A & warm)
    return out


# ---------------------------------------------------------------- the descriptives: K, kept items, share dropped, optimizer steps
def descriptives(root) -> dict:
    out = {"panels": {}, "adapters": {}}
    ftb = Path(root) / "ftb" / DOMAIN
    for name in ("train_b", "train_r0", "train_r1", "train_r2", "train_n"):
        p = ftb / f"{name}.manifest.json"
        if not p.is_file():
            out["panels"][name] = {"available": False, "reason": "no manifest"}
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        lr = m.get("label_rate") or {}
        out["panels"][name] = {"available": True, "manifest_sha1": fx.file_sha1(p), "arm": m.get("arm"), "K": m.get("K"),
                               "examples_source": (m.get("source") or {}).get("examples"), "kept_items": m.get("kept_items"),
                               "items_in_panel": m.get("items_in_panel"), "share_examples_dropped": m.get("share_examples_dropped"),
                               "label_rate_overall": lr.get("overall"),
                               "per_item_all_exactly_half": (lr.get("per_item") or {}).get("all_exactly_half"),
                               "panel_sha1": (m.get("panel") or {}).get("sha1")}
    for tag in "brn":
        for k in (0, 1, 2):
            p = Path(root) / "adapters" / DOMAIN / f"{tag}{k}" / "ftb.json"
            if p.is_file():
                rec = json.loads(p.read_text(encoding="utf-8"))
                out["adapters"][f"{tag}{k}"] = {"K": rec.get("K"), "n_examples": rec.get("n_examples"),
                                                "optimizer_steps": rec.get("optimizer_steps")}
    return out


# ---------------------------------------------------------------- the build
def analyze(world, *, root, split_sha1: str, n_boot: int, seed: int, problems: list) -> dict:
    """Section 3 and section 4 on a loaded world (in memory: the files are read by load_world). `problems` holds the input problems
    found so far; any problem leaves the rule unapplied (INCOMPLETE)."""
    problems = list(problems)
    first = world.X["FT-B"]
    cx, refs = first.cx, first.refs
    n_seeds = {arm: len(world.arms.get(arm, [])) for arm in ARMS}
    notes = [f"{arm}: {n_seeds[arm]} of {len(SEEDS[arm])} adapters have a usable like and swap pass (a failed or missing pass is "
             "reported as missing, never replaced)" for arm in ARMS if n_seeds[arm] != len(SEEDS[arm])]
    res = {"format": FORMAT, "spec": SPEC, "status": STATUS, "wording": WORDING, "domain": DOMAIN}
    ids = present_ids(world)
    if refs is None or not cx.sd_ids:
        problems.append("no CF references (q-hat, MF) or no S_d: the information gain cannot be computed")
    arms_out = {arm: {"available": False, "reason": "not read"} for arm in ARMS + REFERENCE_ARMS}
    paired, uauc, stored, repro = {"available": False, "reason": "no usable rows"}, {"available": False, "reason": "no usable rows"}, {}, {}
    rule_in = {"n_seeds": n_seeds, "G": {}, "G_wu": {}, "contrast": {}, "per_seed_G_B": None}
    rows = {"arms_in_the_common_rows": {arm: world.arms.get(arm, []) for arm in ARMS + REFERENCE_ARMS if arm in world.arms}}
    if refs is not None and ids:
        q, mfres, mf = refs["arrays"]["q_hat"], refs["arrays"]["mf_residual"], refs["arrays"]["mf_score"]
        warm = refs["arrays"]["mf_warm"] > 0
        R = cx.sd_test & np.isfinite(q) & np.isfinite(mfres) & np.isfinite(mf)
        R_A = cx.test & np.isfinite(q) & np.isfinite(mf)
        for i in ids:
            R &= np.isfinite(_vec(world, i, "L")) & np.isfinite(_vec(world, i, "PI"))
            R_A &= np.isfinite(_vec(world, i, "L"))
        rows["S_d_TEST_common"], rows["E_A_TEST_common"] = {"n_pairs": int(R.sum())}, {"n_pairs": int(R_A.sum())}
        if R.any():
            folds = fe.split_folds(cx)
            sd = sd_columns(world, cx, q, mfres, mf, R, n_boot, seed)
            c, own_equal = sd.cols, {}
            repro = reproduction(world, cx, q, mfres, R, sd, n_boot, seed)
            problems += [f"{arm}: the per-user G of this reading differs from ftgrid_report.stacker_block on the same rows"
                         for arm, ok in repro.items() if not ok]
            for arm in ARMS + REFERENCE_ARMS:
                if world.arms.get(arm):
                    arms_out[arm] = arm_block(world, arm, cx, refs, R, folds, sd, n_boot, seed, own_equal)
            for arm in ARMS:
                blk = arms_out[arm]
                if blk.get("available"):
                    rule_in["G"][arm] = blk["G"]["mean_over_seeds"]
                    if "mean_over_seeds" in blk["G_wu"]:
                        rule_in["G_wu"][arm] = blk["G_wu"]["mean_over_seeds"]
            if world.arms.get("FT-B"):
                rule_in["per_seed_G_B"] = [c.one(("G", i), sd.n_pairs, contrast=True)["est"] for i in world.arms["FT-B"]]
            paired = {"available": True, "rows": {"n_pairs": sd.n_pairs, "n_users": sd.n_users},
                      "definition": "per-user differences of the arms' seed means of G (E-D), resampled together with the arms' own "
                                    "columns: one set of 2,000 user resamples (seed 0), the same users and rows in every column",
                      "G": {}, "stack_gain_minus_MF": {}}
            for name, key, slot in (("FT-B_minus_FT-S", ("contrast", "G", "FT-B", "FT-S"), "B_minus_S"),
                                    ("FT-B_minus_FT-N", ("contrast", "G", "FT-B", "FT-N"), "B_minus_N")):
                if key in c.idx:
                    paired["G"][name] = rule_in["contrast"][slot] = c.one(key, sd.n_pairs, contrast=True)
            for arm in ARMS + REFERENCE_ARMS:
                if ("contrast", "stack_minus_MF", arm) in c.idx:
                    paired["stack_gain_minus_MF"][arm] = c.one(("contrast", "stack_minus_MF", arm), sd.n_pairs, contrast=True)
            paired["reference"] = {"G_CF": c.one("G_CF", sd.n_pairs, contrast=True), "stack_gain_MF": c.one("stack_MF", sd.n_pairs),
                                   "stack_gain_MF_residual": c.one("stack_MF_residual", sd.n_pairs)}
            stored = stored_crosscheck(root, world, sd, own_equal)
        if R_A.any():
            uauc = uauc_block(world, cx, q, mf, warm, R_A, n_boot, seed)
    res.update(rows=rows, arms=arms_out, paired=paired, uauc=uauc)
    rule = read_rule(rule_in)
    if problems and rule["label"] != INCOMPLETE:       # an input that is not the recorded one: the rule is not applied
        rule = {**rule, "label_if_the_inputs_were_accepted": rule["label"], "label": INCOMPLETE,
                "reasons": ["input_checks.problems is not empty: " + "; ".join(problems)]}
    res.update(label=rule["label"], reading=rule, descriptives=descriptives(root), stored_crosscheck=stored,
               registered_reproduction={"G_equals_ftgrid_report_stacker_block_on_the_same_rows": repro})
    res["input_checks"] = {"problems": problems, "incomplete_arms": notes, "excluded_runs": world.excluded}
    res["meta"] = {"n_boot": n_boot, "seed": seed, "min_n": MIN_N, "equivalence_margin": EQUIV,
                   "code_sha1": {Path(p).name: fx.file_sha1(p) for p in (__file__, fr.__file__, fe.__file__, fx.__file__)},
                   "split_sha1": split_sha1, "n_sd_users": len(cx.sd_ids),
                   "references": "ftgrid_report.compute_refs (the reference cache or --raw): q-hat, temporal MF"}
    return res


def build(a) -> dict:
    """Load the stored inputs and apply section 3 and section 4 (analyze)."""
    problems: list = []
    world = load_world(a, problems)
    same_world(world, problems)
    res = analyze(world, root=a.root, split_sha1=fx.file_sha1(a.split), n_boot=int(a.n_boot), seed=int(a.seed), problems=problems)
    del world
    gc.collect()
    return res


# ---------------------------------------------------------------- tables and files
def table_rows(res: dict) -> list:
    rows = [{"block": "reading", "arm": "FT-B", "statistic": "label", "note": f"{res['label']} ({res['status']})"}]

    def add(block, arm, model, stat, r, note=""):
        if isinstance(r, dict) and "est" in r:
            rows.append({"block": block, "arm": arm, "model": model, "statistic": stat, "estimate": r.get("est"), "lo": r.get("lo"),
                         "hi": r.get("hi"), "p": r.get("p"), "n_users": r.get("n_users"), "n_pairs": r.get("n_pairs"),
                         "descriptive_min_n": r.get("descriptive_min_n"), "note": note})
    for arm, blk in res.get("arms", {}).items():
        if blk.get("available"):
            for stat in ("G", "G_wu", "stack_gain", "item_prior_share", "e_share"):
                sub = blk.get(stat) or {}
                for m, r in (sub.get("per_model") or {}).items():
                    add("arm", arm, m, stat, r)
                add("arm", arm, "seed_mean", stat, sub.get("mean_over_seeds"))
    pr = res.get("paired") or {}
    for name, r in (pr.get("G") or {}).items():
        add("paired", name, "seed_mean", "G_contrast", r)
    for arm, r in (pr.get("stack_gain_minus_MF") or {}).items():
        add("paired", arm, "seed_mean", "stack_gain_minus_stack_gain_MF", r)
    for name, r in (pr.get("reference") or {}).items():
        add("reference", "MF", "", name, r)
    u = res.get("uauc") or {}
    for arm, blk in (u.get("per_arm") or {}).items():
        for m, r in blk["per_model"].items():
            add("uauc", arm, m, "UAUC_TEST", r)
        add("uauc", arm, "seed_mean", "UAUC_TEST", blk["mean_over_seeds"])
    for name, r in (u.get("references") or {}).items():
        add("uauc", "reference", name, "UAUC_TEST", r)
    for name, blk in (u.get("paired") or {}).items():
        if isinstance(blk, dict) and "mean_over_seeds" in blk:
            add("uauc_paired", name, "seed_mean", "dUAUC", blk["mean_over_seeds"])
    return rows


def tables_text(rows: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLS)
    for r in rows:
        w.writerow([fr._cell(r.get(c)) for c in CSV_COLS])
    return buf.getvalue()


def write_outputs(out, res: dict) -> dict:
    """The CSV first, then the JSON (which records the CSV's sha1) last: the JSON is the completion marker. Both through a temporary
    file and a rename; the JSON is strict (allow_nan=False) and holds no clock, host or path."""
    out = Path(out)
    res = strict_json(res)
    text = tables_text(table_rows(res))
    res["meta"]["tables_csv_sha1"] = hashlib.sha1(text.encode("utf-8")).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    for path, data in ((out.with_suffix(".csv"), text), (out, json.dumps(res, indent=2, allow_nan=False) + "\n")):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(data)
        tmp.replace(path)
    return res


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="apply the reading rule to the stored outputs of FT-B, FT-S and FT-N")
    b.add_argument("--split", required=True, help="the registered panels/ml1m/ftgrid_split.json")
    b.add_argument("--panels", required=True, help="the registered panels/ml1m/")
    b.add_argument("--grid_scores", default=None, help="the registered scores/ (zeroshot, s0-s2: the reference arms); optional")
    b.add_argument("--root", required=True, help="the FT-B root: scores/<b|r|n>/ml1m, ftb/ml1m, adapters/ml1m, report/<tag>, extra/<tag>")
    b.add_argument("--raw", default=None, help="data/raw: the CF references (needed unless --refs holds a matching cache)")
    b.add_argument("--refs", default=None, help="the CF reference cache (read if it matches, else written from --raw)")
    b.add_argument("--out", required=True, help="outputs/confrec/ftb_reading/ml1m.json; the tables go to the .csv beside it")
    b.add_argument("--n_boot", type=int, default=N_BOOT)
    b.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    res = write_outputs(a.out, build(a))
    print(f"wrote {a.out} and {Path(a.out).with_suffix('.csv')}")
    print(f"FT-B reading ({res['status']}): {res['label']}")
    for p in res["input_checks"]["problems"]:
        print(f"problem: {p}", file=sys.stderr)
    return 0 if res["label"] != INCOMPLETE else 2


if __name__ == "__main__":
    sys.exit(main())
