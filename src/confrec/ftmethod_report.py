"""Method-slot report (idea-stage/PREREG_AMENDMENT_3.md section 7, FT-M; addendum 1 item 8; addendum 2 item 2).

    # one dataset (stage 5 of scripts/sigir/run_ftmethod.sh)
    python -m src.confrec.ftmethod_report dataset --domain ml1m \
        --split outputs/confrec/ftgrid/panels/ml1m/ftgrid_split.json --panels outputs/confrec/ftgrid/panels/ml1m \
        --sft_scores outputs/confrec/ftgrid/scores/ml1m --method_dir outputs/confrec/ftmethod/ml1m \
        --out outputs/confrec/ftmethod/ml1m/report.json [--n_boot 2000] [--seed 0]
    # the slot-level state over the per-dataset reports <root>/<d>/report.json (registered order, kill rule, Holm)
    python -m src.confrec.ftmethod_report slot --root outputs/confrec/ftmethod [--out outputs/confrec/ftmethod/slot.json] \
        [--check_next D]

dataset (TEST rows; seeds 0-2 paired by seed; the conventions of src/confrec/ftgrid_report.py, imported):
  SFT_b0             the section-2 SFT adapters' like logits on eval.jsonl (--sft_scores/s<k>/like; ML-1M: Gate-FT's),
                     i.e. comparator (i), b fixed at 0: UAUC per seed and seed-averaged
  post_hoc_stacking  comparator (ii): per seed, a logistic regression of the label on [SFT logit, z(q-hat)] fit on the
                     CAL rows of the EVAL users (ftgrid_report.logit_fit) and applied to TEST: UAUC of its linear
                     predictor; the coefficients
  prior_offset_LoRA  the scorer's logit(Yes) - logit(No) of the prior-offset adapter o<k> (--method_dir/scores/o<k>/like)
                     + b_k * z(q-hat), b_k from the adapter's offset.json (b trained in its own AdamW group at lr 1e-2,
                     Amendment 3 addendum 3; any other group makes the dataset INVALID), z with the recorded TRAIN
                     constants
  difference         the endpoint dUAUC(prior-offset - post-hoc stacking): per seed and seed-averaged (mean over users
                     of the per-user AUC difference averaged over the seeds) with the user-bootstrap CI and p
                     (ftgrid_report.contrast_models: 2,000 resamples, seed 0, p = 2 min(P*(d <= 0), P*(d >= 0)) with
                     +1/(B + 1)), sigma_seed = SD (ddof 1) of the per-seed differences
  reference          UAUC of q-hat (the prior-only item mean, forensics.prior_means k = 5) on the same rows
  decision           PASS iff the seed-averaged dUAUC >= +0.01, its CI excludes 0 and the sigma_seed rule holds (all 3
                     per-seed differences positive and the mean > 2 sigma_seed); INCOMPLETE when a seed is missing (a
                     like pass absent, incomplete or failing E1, or no offset.json); INVALID on an input problem; FAIL
                     otherwise (an endpoint on fewer than 150 users is descriptive: no CI-based claim, so no pass).
  Every block uses the identical rows (TEST rows finite in every SFT and prior-offset logit of the seeds present, and in
  z) and users (those with both classes there). The report carries the raw p only (addendum 1 item 8).
slot (the cross-dataset step of addendum 1 item 8, the kill rule "in the registered dataset order by the report code" of
  addendum 2 item 2): datasets in the order ML-1M, Toys, Video_Games, Sports; killed as soon as 2 have failed; survives
  only with passes on at least 3 of the 4; Holm over the datasets run (the decided ones with a p-value); single-backbone.
  --check_next D exits 4 (refused) unless every dataset before D is decided (PASS or FAIL) and fewer than 2 of them failed.
Outputs: strict JSON (stats.strict_json, allow_nan=False; no wall-clock, host or path) and <out stem>_tables.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

from src.confrec import forensics as fx
from src.confrec import ftgrid_report as fr
from src.confrec import train_lora_offset as tlo
from src.confrec.stats import strict_json

NAN = float("nan")
SPEC = ("idea-stage/PREREG_AMENDMENT_3.md section 7 (FT-M) with sections 1, 3 and 11; PREREG_AMENDMENT_3_ADDENDUM_1.md "
        "item 8; PREREG_AMENDMENT_3_ADDENDUM_2.md item 2")
DATASETS = ("ml1m", "toys", "games", "sports")     # the registered order (A3 section 7)
SEEDS = (0, 1, 2)                                   # A3 section 2
PASS_MIN = 0.01                                     # seed-averaged dUAUC >= +0.01
MAX_FAILS = 2                                       # killed as soon as it has failed on 2 datasets
MIN_PASSES = 3                                      # survives only with passes on at least 3 of the 4
HARD_KILL_DATE = "2026-11-30"
DECIDED = ("PASS", "FAIL")
STATUSES = ("PASS", "FAIL", "INCOMPLETE", "INVALID")
ALIAS = ("SFT_b0", "post_hoc_stacking", "prior_offset_LoRA", "difference", "reference")
SLOT_COLS = ("dataset", "order", "status", "dUAUC", "lo", "hi", "p", "p_holm", "confirmed", "sigma_seed",
             "sigma_seed_rule", "n_users", "descriptive_min_n", "note")

OPERATIONALIZATIONS = (
    "pairs: one per (user_id, item_id) of eval.jsonl (ftgrid_report.make_ctx); TEST = candidate_timestamps >= T and CAL "
    "= < T with T read from ftgrid_split.json; every like pass joined on that key (ftgrid_report.to_pairs)",
    "runs: ftgrid_report.load_run (E1 of A3 section 2 from report.json, panel sha1, variant); a seed is present when the "
    "SFT like pass s<k>, the prior-offset like pass o<k> and o<k>'s offset.json are all usable; a run failing E1 is "
    "excluded and listed, never replaced by another seed",
    "z(q-hat) on EVAL pairs = eval_qhat.csv.gz of run_ftmethod.sh stage 1: q-hat = forensics.prior_means "
    "mean_prior_shrunk (k = 5, other users' first ratings strictly before the candidate's timestamp), z = (q-hat - "
    "mean) / sd with the TRAIN constants of qhat_manifest.json (recomputed here and compared with the stored z); z = 0 "
    "where q-hat is not finite, as in training",
    "prior-offset ranking score of seed k = like logit of o<k> + b_k z, b_k = offset.json b of the adapter that was "
    "scored (its weights sha1 equals the like pass's run.key, its standardisation constants and manifest sha1 equal "
    "the current manifest's, and b was trained in its own AdamW group at lr 1e-2 with weight decay 0: Amendment 3 "
    "addendum 3); training shifted the single answer token tok('Yes'), the test-time score adds b z to the scorer's "
    "logit(Yes) - logit(No) over the yes / no id sets, as registered",
    "post-hoc stacking of seed k: ftgrid_report.logit_fit (logistic regression with intercept, Newton, features "
    "standardised on the fitting rows, L2 1e-4 on the slopes) of the label on [like logit of s<k>, z] over the CAL rows "
    "of the EVAL users (finite in every logit of the seeds present), applied to the TEST rows (linear predictor); "
    "coefficients reported on the raw scale",
    "identical rows and users: TEST rows finite in every SFT and prior-offset logit of the seeds present and in z; "
    "users with both classes there; every UAUC is the mean over those users of the per-user AUC (ties 1/2), and the "
    "seed-averaged UAUC is the mean over users of the per-user AUC averaged over the seeds (never the UAUC of averaged "
    "logits); the q-hat reference drops rows where q-hat is not finite",
    "endpoint = ftgrid_report.contrast_models({seed k: (prior-offset score, stacking predictor)}): per-seed dUAUC, the "
    "seed-averaged dUAUC with the 95% percentile CI over 2,000 user resamples (seed 0) and p = min(1, 2 min((#{d* <= 0} "
    "+ 1)/(B + 1), (#{d* >= 0} + 1)/(B + 1))); sigma_seed = SD (ddof 1) of the 3 per-seed values",
    "pass = complete (3 seeds) and seed-averaged dUAUC >= 0.01 and ci_excludes_0 (the literal 'CI excludes 0', kept by "
    "the main-session decision of 2026-10-04) and every per-seed dUAUC > 0 and mean > 2 sigma_seed; fewer than 150 "
    "users makes the endpoint descriptive (p and ci_excludes_0 null, A3 section 1), which cannot pass: FAIL with that "
    "reason",
    "status INCOMPLETE (a registered seed missing) and INVALID (an input problem: sha1, pairing, manifest or b-group "
    "mismatch) are neither PASS nor FAIL: the slot step does not count them as failed and refuses every later dataset "
    "until they are resolved (main-session decision of 2026-10-04)",
)


# ---------------------------------------------------------------- inputs
def _fin(x) -> bool:
    return fr._fin(x)


def same_path(a, b) -> bool:
    if not a or not b:
        return False
    return Path(str(a)).resolve() == Path(str(b)).resolve()


def adapter_weights_sha1(adir: Path) -> str | None:
    """sha1 of the adapter weights as run_ftmethod.sh's run.key hashes them (`cat adapter_model.* | sha1sum`)."""
    ws = sorted(Path(adir).glob("adapter_model.*"))
    if not ws:
        return None
    h = hashlib.sha1()
    for w in ws:
        h.update(w.read_bytes())
    return h.hexdigest()


def run_key_weights(run_dir: Path) -> str | None:
    """The adapter-weights sha1 of a scoring dir's run.key (panel sha1, model, variant, weights sha1, args)."""
    p = Path(run_dir) / "run.key"
    if not p.is_file():
        return None
    parts = p.read_text(encoding="utf-8").split()
    return parts[3] if len(parts) > 3 else None


def run_lora(run_dir: Path):
    try:
        return json.loads((Path(run_dir) / "report.json").read_text(encoding="utf-8")).get("lora")
    except (OSError, ValueError):
        return None


def load_eval_qhat(mdir: Path, cx, eval_sha: str, T: float, domain: str, problems: list) -> dict:
    """q-hat and z of every EVAL pair (aligned to cx) from stage 1, checked against the manifest."""
    man_p = Path(mdir) / tlo.MANIFEST
    out = {"manifest": None, "manifest_sha1": None, "q": np.full(cx.n, NAN), "z": np.full(cx.n, NAN),
           "n_pairs_without_qhat_row": int(cx.n)}
    if not man_p.is_file():
        problems.append(f"{tlo.MANIFEST}: missing in --method_dir (run_ftmethod.sh stage 1)")
        return out
    try:
        man = json.loads(man_p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        problems.append(f"{tlo.MANIFEST}: unreadable ({type(e).__name__})")
        return out
    out["manifest"], out["manifest_sha1"] = man, fx.file_sha1(man_p)
    if man.get("format") != tlo.MANIFEST_FORMAT:
        problems.append(f"{tlo.MANIFEST}: format {man.get('format')!r} is not {tlo.MANIFEST_FORMAT}")
    if man.get("domain") != domain:
        problems.append(f"{tlo.MANIFEST}: domain {man.get('domain')!r} is not {domain!r}")
    if (man.get("eval") or {}).get("sha1") != eval_sha:
        problems.append(f"{tlo.MANIFEST}: its eval.jsonl sha1 {(man.get('eval') or {}).get('sha1')} is not eval.jsonl's "
                        f"{eval_sha}")
    if not _fin(man.get("T")) or float(man["T"]) != T:
        problems.append(f"{tlo.MANIFEST}: T {man.get('T')!r} is not the split's {T!r}")
    st = man.get("standardisation") or {}
    if not (_fin(st.get("mean")) and _fin(st.get("sd")) and st["sd"] > 0):
        problems.append(f"{tlo.MANIFEST}: no usable standardisation constants")
        return out
    qp = Path(mdir) / tlo.EVAL_QHAT
    if not qp.is_file():
        problems.append(f"{tlo.EVAL_QHAT}: missing in --method_dir (stage 1)")
        return out
    if fx.file_sha1(qp) != (man.get("files") or {}).get(tlo.EVAL_QHAT):
        problems.append(f"{tlo.EVAL_QHAT}: sha1 differs from the manifest's")
    rows = tlo.read_qhat(qp)
    keys = list(zip(cx.users.tolist(), cx.item.tolist()))
    have = np.array([k in rows for k in keys], bool)
    q = np.array([rows[k]["q_hat"] if k in rows else NAN for k in keys], float)
    stored = np.array([rows[k]["z"] if k in rows else NAN for k in keys], float)
    z = np.where(have, tlo.zscore(q, st), NAN)
    if (~have).any() or len(rows) != cx.n:
        problems.append(f"{tlo.EVAL_QHAT}: {int((~have).sum())} EVAL pairs without a row ({len(rows)} rows for {cx.n} "
                        "pairs)")
    if not np.array_equal(z[have], stored[have]):
        problems.append(f"{tlo.EVAL_QHAT}: stored z differs from (q-hat - mean) / sd with the manifest's constants")
    out.update(q=q, z=z, n_pairs_without_qhat_row=int((~have).sum()), n_qhat_nonfinite=int((have & ~np.isfinite(q)).sum()))
    return out


def load_offset(adir: Path, k: int, qh: dict, domain: str) -> dict:
    """{status OK | ABSENT | INVALID, reason, b, ...} of the prior-offset adapter o<k> (offset.json, train_config.json)."""
    p, c = Path(adir) / tlo.OFFSET_JSON, Path(adir) / "train_config.json"
    if not p.is_file() or not c.is_file():
        return {"status": "ABSENT", "reason": f"o{k}: no {tlo.OFFSET_JSON} / train_config.json (stage 3)"}
    try:
        off = json.loads(p.read_text(encoding="utf-8"))
        cfg = json.loads(c.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"status": "INVALID", "reason": f"o{k}: unreadable ({type(e).__name__})"}
    probs = []
    if off.get("format") != tlo.OFFSET_FORMAT:
        probs.append(f"format {off.get('format')!r}")
    if off.get("seed") != k or cfg.get("seed") != k:
        probs.append(f"seed {off.get('seed')!r} / {cfg.get('seed')!r} is not {k}")
    if off.get("domain") != domain:
        probs.append(f"domain {off.get('domain')!r}")
    b = off.get("b")
    if not _fin(b):
        probs.append("no finite b")
    elif (cfg.get("prior_offset") or {}).get("b") != b:
        probs.append("train_config.json and offset.json disagree on b")
    if qh["manifest_sha1"] is not None and off.get("qhat_manifest_sha1") != qh["manifest_sha1"]:
        probs.append("trained on another q-hat manifest")
    if qh["manifest"] is not None and off.get("standardisation") != qh["manifest"].get("standardisation"):
        probs.append("standardisation constants differ from the manifest's")
    grp = off.get("b_group") or {}
    if not (grp.get("own_group") is True and grp.get("lr") == tlo.B_LR and grp.get("weight_decay") == 0.0):
        probs.append(f"b's optimizer group {({k: grp.get(k) for k in ('own_group', 'lr', 'weight_decay')})} is not the "
                     f"registered one (its own group, lr {tlo.B_LR}, weight decay 0; Amendment 3 addendum 3)")
    return {"status": "INVALID" if probs else "OK", "reason": f"o{k}: " + "; ".join(probs) if probs else None,
            "b": float(b) if _fin(b) else None, "b_lr": off.get("b_lr"),
            "b_group": {k: grp.get(k) for k in ("own_group", "lr", "weight_decay", "source")},
            "yes_token_id": off.get("yes_token_id"), "sft_adapter": off.get("sft_adapter"),
            "weights_sha1": adapter_weights_sha1(adir), "n_examples": off.get("n_examples"),
            "dry_run": bool(off.get("dry_run"))}


# ---------------------------------------------------------------- statistics
def raw_coefficients(model) -> dict:
    """logit_fit's (mean, sd, w) as the raw-scale linear predictor intercept + slope_L L + slope_z z."""
    m, s, w = model
    slopes = np.asarray(w[1:], float) / np.asarray(s, float)
    return {"intercept": float(w[0] - float((slopes * np.asarray(m, float)).sum())), "slope_sft_logit": float(slopes[0]),
            "slope_z": float(slopes[1])}


def stacking(L, z, y, fit_rows) -> tuple[np.ndarray | None, dict]:
    """Post-hoc stacking of one seed: fit on `fit_rows` (CAL), linear predictor on every row with finite inputs."""
    L, z = np.asarray(L, float), np.asarray(z, float)
    X = np.column_stack([L, z])
    yf = np.asarray(y, int)[fit_rows]
    n = int(np.asarray(fit_rows, bool).sum())
    info = {"n_fit_rows": n, "n_fit_positive": int(yf.sum())}
    if n < 2 or not 0 < yf.sum() < n:
        return None, {**info, "available": False, "reason": "the CAL rows lack a class"}
    model = fr.logit_fit(X[fit_rows], yf)
    eta = np.full(len(L), NAN)
    ok = np.isfinite(L) & np.isfinite(z)
    eta[ok] = fr.logit_eta(model, X[ok])
    return eta, {**info, "available": True, **raw_coefficients(model)}


def decide(diff: dict | None, n_seeds_present: int, problems: list) -> dict:
    """A3 section 7: PASS iff the seed-averaged dUAUC >= +0.01, its CI excludes 0 and the sigma_seed rule holds (all 3
    per-seed differences positive, mean > 2 sigma_seed); INVALID on an input problem; INCOMPLETE without all 3 seeds."""
    rule = ("PASS iff complete (seeds 0-2) and seed-averaged dUAUC(prior-offset - post-hoc stacking) >= +0.01 and its CI "
            "excludes 0 and every per-seed dUAUC > 0 and mean > 2 sigma_seed (SD ddof 1 of the 3 per-seed values)")
    if problems:
        return {"status": "INVALID", "reason": "input problems: " + "; ".join(problems), "conditions": None, "rule": rule}
    if n_seeds_present < len(SEEDS) or diff is None or "mean_over_seeds" not in diff:
        return {"status": "INCOMPLETE", "reason": f"{n_seeds_present} of {len(SEEDS)} seeds present (A3 section 2: "
                                                  "a missing seed is never replaced)", "conditions": None, "rule": rule}
    mean, seeds = diff["mean_over_seeds"], diff["seeds"]
    est, sd = mean.get("est"), seeds.get("sigma_seed")
    per = [diff["per_seed"][f"seed{k}"]["est"] for k in SEEDS]
    if not _fin(est):
        return {"status": "INVALID", "reason": "no evaluable user (no user with both classes among the rows)",
                "conditions": None, "rule": rule}
    cond = {"complete": True,
            "n_users_ge_150": not mean.get("descriptive_min_n"),
            "mean_ge_0.01": bool(est >= PASS_MIN),
            "ci_excludes_0": mean.get("ci_excludes_0") is True,
            "all_seeds_positive": all(_fin(x) and x > 0 for x in per),
            "mean_gt_2_sigma_seed": bool(_fin(sd) and est > 2 * sd)}
    cond["sigma_seed_rule"] = cond["all_seeds_positive"] and cond["mean_gt_2_sigma_seed"]
    passed = cond["mean_ge_0.01"] and cond["ci_excludes_0"] and cond["sigma_seed_rule"]
    why = {"n_users_ge_150": f"descriptive: {mean.get('n_users')} users < {fr.MIN_N}, no CI-based claim (A3 section 1)",
           "mean_ge_0.01": f"seed-averaged dUAUC {est:.4f} < +0.01",
           "ci_excludes_0": (f"its CI [{mean.get('lo')}, {mean.get('hi')}] does not exclude 0"
                             if mean.get("ci_excludes_0") is False else "no CI-based claim (descriptive or no CI)"),
           "all_seeds_positive": f"a per-seed dUAUC is not positive ({per})",
           "mean_gt_2_sigma_seed": f"mean {est:.4f} is not above 2 sigma_seed ({sd})"}
    reasons = [why[k] for k in ("n_users_ge_150", "mean_ge_0.01", "ci_excludes_0", "all_seeds_positive",
                                "mean_gt_2_sigma_seed") if not cond[k]]
    return {"status": "PASS" if passed else "FAIL", "reason": "; ".join(reasons) or None, "conditions": cond,
            "dUAUC": est, "lo": mean.get("lo"), "hi": mean.get("hi"), "p": mean.get("p"), "sigma_seed": sd,
            "per_seed": per, "n_users": mean.get("n_users"), "descriptive_min_n": mean.get("descriptive_min_n"),
            "rule": rule}


# ---------------------------------------------------------------- one dataset
def build(a) -> dict:
    split = fr.read_json(a.split)
    T, variant = float(split["T"]), split.get("variant")
    panels, mdir = Path(a.panels), Path(a.method_dir)
    problems = []
    if split.get("domain") != a.domain:
        problems.append(f"--split is the split of {split.get('domain')!r}, not of {a.domain!r}")
    eval_p = panels / "eval.jsonl"
    if not eval_p.is_file():
        raise SystemExit(f"{eval_p}: no eval.jsonl")
    eval_sha = fx.file_sha1(eval_p)
    rec = (split.get("files") or {}).get("eval.jsonl")
    if rec is not None and rec != eval_sha:
        problems.append(f"eval.jsonl: sha1 {eval_sha} differs from ftgrid_split.json files ({rec})")
    cx = fr.make_ctx(fx.read_jsonl(eval_p), T, [], None)
    if cx.n_duplicate_pairs:
        problems.append(f"eval.jsonl repeats {cx.n_duplicate_pairs} (user_id, item_id) pairs")
    qh = load_eval_qhat(mdir, cx, eval_sha, T, a.domain, problems)

    # ---- runs and adapters
    runs, offsets, excluded = {}, {}, []
    for k in SEEDS:
        for tag, d in ((f"s{k}", Path(a.sft_scores) / f"s{k}" / "like"), (f"o{k}", mdir / "scores" / f"o{k}" / "like")):
            runs[tag] = fr.load_run(d, f"{tag}/like", eval_sha, tag, variant)
            runs[tag]["dir"] = d
            if runs[tag]["status"] not in ("OK", "ABSENT"):
                excluded.append({"model": tag, "status": runs[tag]["status"], "reason": runs[tag]["reason"]})
        adir = mdir / "adapters" / f"o{k}"
        offsets[k] = load_offset(adir, k, qh, a.domain)
        if offsets[k]["status"] == "INVALID":
            problems.append(offsets[k]["reason"])
        if offsets[k]["status"] == "OK":
            ro, rs = runs[f"o{k}"], runs[f"s{k}"]
            if ro["status"] == "OK":
                if not same_path(run_lora(ro["dir"]), adir):
                    problems.append(f"o{k}/like scored the adapter {run_lora(ro['dir'])!r}, not {adir}")
                rk = run_key_weights(ro["dir"])
                if rk is not None and rk != offsets[k]["weights_sha1"]:
                    problems.append(f"o{k}/like: run.key weights sha1 {rk} is not the adapter's "
                                    f"{offsets[k]['weights_sha1']} (offset.json b would not belong to the scored "
                                    "adapter)")
            if rs["status"] == "OK" and not same_path(run_lora(rs["dir"]), offsets[k]["sft_adapter"]):
                problems.append(f"s{k}/like scored {run_lora(rs['dir'])!r}, but o{k} was trained against the SFT "
                                f"comparator {offsets[k]['sft_adapter']!r}")
    backbones = sorted({r.get("backbone") for r in runs.values() if r["status"] == "OK" and r.get("backbone")})
    if len(backbones) > 1:
        problems.append(f"runs of several backbones: {backbones}")

    # ---- vectors on the EVAL pairs
    L, joins = {}, {}
    for tag, r in runs.items():
        if r["status"] == "OK":
            L[tag], joins[tag] = fr.to_pairs(r["sc"], cx, np.ones(cx.n, bool))
    seeds = [k for k in SEEDS if f"s{k}" in L and f"o{k}" in L and offsets[k]["status"] == "OK"]
    z, q = qh["z"], qh["q"]
    R, C = cx.test & np.isfinite(z), cx.cal & np.isfinite(z)
    for k in seeds:
        fin = np.isfinite(L[f"s{k}"]) & np.isfinite(L[f"o{k}"])
        R &= fin
        C &= fin
    S, stack_info, Ro = {}, {}, {}
    for k in seeds:
        eta, info = stacking(L[f"s{k}"], z, cx.y, C)
        stack_info[f"s{k}"] = info
        if eta is not None:
            S[k] = eta
            Ro[k] = L[f"o{k}"] + offsets[k]["b"] * z
    seeds_ok = [k for k in seeds if k in S]
    boot = (a.n_boot, a.seed)
    nreg = len(SEEDS)
    if seeds_ok:
        diff = fr.contrast_models({f"seed{k}": (Ro[k], S[k]) for k in seeds_ok}, cx.y, cx.uc, R, *boot,
                                  n_registered=nreg)
        slot = {
            "SFT_b0": {"definition": "comparator (i): the section-2 SFT adapters (b fixed at 0; ML-1M: Gate-FT's), "
                                     "their like logit",
                       **fr.uauc_models({f"s{k}": L[f"s{k}"] for k in seeds_ok}, cx.y, cx.uc, R, *boot,
                                        n_registered=nreg)},
            "post_hoc_stacking": {"definition": "comparator (ii): logistic regression of the label on [SFT logit, z] "
                                                "fit on CAL rows, applied to TEST (linear predictor)",
                                  "fit": stack_info,
                                  **fr.uauc_models({f"s{k}": S[k] for k in seeds_ok}, cx.y, cx.uc, R, *boot,
                                                   n_registered=nreg)},
            "prior_offset_LoRA": {"definition": "the prior-offset adapter's logit(Yes) - logit(No) + b z(q-hat)",
                                  "b": {f"o{k}": offsets[k]["b"] for k in SEEDS if offsets[k]["status"] == "OK"},
                                  **fr.uauc_models({f"o{k}": Ro[k] for k in seeds_ok}, cx.y, cx.uc, R, *boot,
                                                   n_registered=nreg)},
            "difference": {"definition": "dUAUC(prior-offset - post-hoc stacking) on TEST, per seed (paired by seed) and "
                                         "seed-averaged, user-bootstrap CI and p; sigma_seed", **diff},
            "reference": {"definition": "the prior-only item mean q-hat (forensics.prior_means, k = 5) on the same rows "
                                        "(rows with a non-finite q-hat dropped)",
                          **fr.uauc_models({}, cx.y, cx.uc, R & np.isfinite(q), *boot, extra={"q_hat": q})}}
    else:
        diff = None
        why = "no seed with usable SFT and prior-offset like passes, offset.json and a CAL fit"
        slot = {k: fr._na(why) for k in ALIAS}
        slot["post_hoc_stacking"]["fit"] = stack_info
    decision = decide(diff, len(seeds_ok), problems)
    present = {"seeds_present": [f"seed{k}" for k in seeds_ok],
               "seeds_missing": [{"seed": k, "sft": runs[f"s{k}"]["status"], "prior_offset": runs[f"o{k}"]["status"],
                                  "offset_json": offsets[k]["status"],
                                  "reasons": [x for x in (runs[f"s{k}"].get("reason"), runs[f"o{k}"].get("reason"),
                                                          offsets[k].get("reason"),
                                                          None if k not in seeds or k in S else
                                                          stack_info[f"s{k}"].get("reason")) if x]}
                                 for k in SEEDS if k not in seeds_ok]}
    bb = backbones[0] if len(backbones) == 1 else None
    meta = {"domain": a.domain, "backbone": bb, "single_backbone": True, "T": T, "variant": variant,
            "split_sha1": fx.file_sha1(a.split), "eval_sha1": eval_sha, "qhat_manifest_sha1": qh["manifest_sha1"],
            "standardisation": (qh["manifest"] or {}).get("standardisation"), "n_boot": a.n_boot, "seed": a.seed,
            "min_n": fr.MIN_N, "n_eval_pairs": int(cx.n), "n_test_pairs": int(cx.test.sum()),
            "n_cal_pairs": int(cx.cal.sum()), "rows": {"test_rows_used": int(R.sum()), "cal_fit_rows": int(C.sum())},
            "n_pairs_without_qhat_row": qh["n_pairs_without_qhat_row"], "n_qhat_nonfinite": qh.get("n_qhat_nonfinite"),
            "sft_comparator": "the section-2 FT adapters s0-s2 (ML-1M: Gate-FT's), scored by run_ftgrid.sh stage 3",
            "dry_run_inputs": any(o.get("dry_run") for o in offsets.values())}
    run_meta = {t: {k: v for k, v in r.items() if k not in ("sc", "swap", "dir")} for t, r in runs.items()}
    off_meta = {f"o{k}": {**{kk: vv for kk, vv in v.items() if kk != "sft_adapter"},
                          **({"sft_adapter_id": hashlib.sha1(str(v["sft_adapter"]).encode("utf-8")).hexdigest()[:12]}
                             if v.get("sft_adapter") else {})} for k, v in offsets.items()}   # no path in the json
    return {"spec": SPEC, "alias": "slot", "meta": meta,
            "input_checks": {"problems": problems, "excluded_runs": excluded, **present},
            "runs": run_meta, "offsets": off_meta, "joins": joins,
            "slot": slot, "decision": decision, "operationalizations": list(OPERATIONALIZATIONS)}


def table_rows(res: dict) -> list:
    d, bb = res["meta"]["domain"], res["meta"]["backbone"] or ""
    rows = []

    def add(part, model, endpoint, r, note=""):
        if not isinstance(r, dict) or "est" not in r:
            return
        rows.append({"domain": d, "backbone": bb, "block": "slot", "regime": part, "model": model, "endpoint": endpoint,
                     "est": r.get("est"), "lo": r.get("lo"), "hi": r.get("hi"), "p": r.get("p"),
                     "n_users": r.get("n_users"), "n_pairs": r.get("n_pairs"),
                     "descriptive_min_n": r.get("descriptive_min_n"), "note": note})

    slot = res["slot"]
    for part in ("SFT_b0", "post_hoc_stacking", "prior_offset_LoRA"):
        blk = slot.get(part) or {}
        for m, r in (blk.get("per_model") or {}).items():
            note = f"b = {slot['prior_offset_LoRA']['b'].get(m)}" if part == "prior_offset_LoRA" else ""
            add(part, m, "UAUC_TEST", r, note)
        add(part, "mean_over_seeds", "UAUC_TEST", blk.get("mean_over_seeds"))
    diff = slot.get("difference") or {}
    for m, r in (diff.get("per_seed") or {}).items():
        add("difference", m, "dUAUC_TEST", r)
    dec = res["decision"]
    add("difference", "mean_over_seeds", "dUAUC_TEST", diff.get("mean_over_seeds"),
        f"sigma_seed {(diff.get('seeds') or {}).get('sigma_seed')}; slot sigma_seed rule (every seed > 0, mean > 2 "
        f"sigma_seed) {(dec.get('conditions') or {}).get('sigma_seed_rule')}")
    add("reference", "q_hat", "UAUC_TEST", ((slot.get("reference") or {}).get("references") or {}).get("q_hat"))
    rows.append({"domain": d, "backbone": bb, "block": "slot", "regime": "decision", "model": "", "endpoint": "status",
                 "est": dec.get("dUAUC"), "lo": dec.get("lo"), "hi": dec.get("hi"), "p": dec.get("p"),
                 "n_users": dec.get("n_users"), "n_pairs": None, "descriptive_min_n": dec.get("descriptive_min_n"),
                 "note": f"{dec['status']}" + (f": {dec['reason']}" if dec.get("reason") else "")})
    return rows


# ---------------------------------------------------------------- the slot across datasets
def slot_state(statuses: dict) -> dict:
    """The slot-level state of A3 section 7 from per-dataset statuses (PASS, FAIL, INCOMPLETE, INVALID; None or absent =
    no report), walked in the registered order: killed as soon as 2 datasets failed; survives with >= 3 passes; a
    dataset that is not decided stops the walk (it is the next to complete); a report after the kill point or after an
    undecided dataset is a violation of the order."""
    passes, fails, decided, violations = [], [], [], []
    killed_after = pending = None
    for d in DATASETS:
        s = statuses.get(d)
        if s is not None and s not in STATUSES:
            raise ValueError(f"{d}: unknown status {s!r}")
        if killed_after is not None or pending is not None:
            if s is not None:
                violations.append(f"{d}: a report exists although " + (
                    f"the slot was killed after {killed_after}" if killed_after is not None
                    else f"{pending} (earlier in the registered order) is not decided"))
            continue
        if s in DECIDED:
            decided.append(d)
            (passes if s == "PASS" else fails).append(d)
            if len(fails) >= MAX_FAILS:
                killed_after = d
        else:
            pending = d
    if killed_after is not None:
        state = "KILLED"
    elif len(passes) >= MIN_PASSES:
        state = "SURVIVES"
    else:
        state = "OPEN"
    final = killed_after is not None or len(decided) == len(DATASETS)
    nxt = None if killed_after is not None else pending
    return {"state": state, "final": final, "killed_after": killed_after, "passes": passes, "fails": fails,
            "decided": decided, "pending": pending, "pending_status": statuses.get(pending) if pending else None,
            "next_dataset": nxt, "violations": violations}


def check_next(statuses: dict, d: str) -> tuple[bool, str]:
    """May dataset d run (its training, scoring and report)? Every earlier dataset in the registered order must be
    decided, and fewer than 2 of them failed (a dataset already decided may be resumed)."""
    if d not in DATASETS:
        raise ValueError(f"unknown dataset {d!r}")
    before = DATASETS[:DATASETS.index(d)]
    open_ = [x for x in before if statuses.get(x) not in DECIDED]
    if open_:
        x = open_[0]
        return False, (f"{d} refused: the registered order is {', '.join(DATASETS)} (A3 section 7) and {x} has "
                       f"{'no report' if statuses.get(x) is None else 'status ' + str(statuses.get(x))}: it is "
                       "decided (PASS or FAIL) before a later dataset runs")
    failed = [x for x in before if statuses.get(x) == "FAIL"]
    if len(failed) >= MAX_FAILS:
        return False, (f"{d} refused: the slot was killed after {failed[MAX_FAILS - 1]} (it failed on "
                       f"{' and '.join(failed[:MAX_FAILS])}; A3 section 7: killed as soon as it has failed on 2 "
                       "datasets); a killed slot is reported once, in the appendix, as a negative result")
    return True, f"{d} may run ({len(before)} earlier dataset(s) decided, {len(failed)} failed)"


def read_dataset_report(root: Path, d: str) -> dict | None:
    p = Path(root) / d / "report.json"
    if not p.is_file():
        return None
    try:
        rep = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"status": "INVALID", "reason": f"unreadable report ({type(e).__name__})", "report_sha1": fx.file_sha1(p)}
    dec = rep.get("decision") or {}
    status = dec.get("status")
    reason = dec.get("reason")
    if (rep.get("meta") or {}).get("domain") != d:
        status, reason = "INVALID", f"the report at {d}/report.json is of {(rep.get('meta') or {}).get('domain')!r}"
    elif status not in STATUSES:
        status, reason = "INVALID", f"no decision status ({status!r})"
    return {"status": status, "reason": reason, "report_sha1": fx.file_sha1(p),
            "dUAUC": dec.get("dUAUC"), "lo": dec.get("lo"), "hi": dec.get("hi"), "p": dec.get("p"),
            "sigma_seed": dec.get("sigma_seed"), "sigma_seed_rule": (dec.get("conditions") or {}).get("sigma_seed_rule"),
            "n_users": dec.get("n_users"), "descriptive_min_n": dec.get("descriptive_min_n"),
            "backbone": (rep.get("meta") or {}).get("backbone"),
            "dry_run_inputs": bool((rep.get("meta") or {}).get("dry_run_inputs"))}


def slot_summary(root) -> dict:
    infos = {d: read_dataset_report(root, d) for d in DATASETS}
    statuses = {d: (i or {}).get("status") for d, i in infos.items()}
    st = slot_state(statuses)
    run = [d for d in DATASETS if infos[d] is not None]
    pvals = {d: infos[d]["p"] for d in st["decided"]}
    adj = fr.holm(pvals)
    confirmed = {d: bool(adj[d] is not None and adj[d] < 0.05 and infos[d].get("sigma_seed_rule") is True)
                 for d in st["decided"]}
    backbones = sorted({i["backbone"] for i in infos.values() if i and i.get("backbone")})
    return {"spec": SPEC, "alias": "slot", "registered_order": list(DATASETS),
            "rules": {"pass": "seed-averaged dUAUC(prior-offset - post-hoc stacking) >= +0.01, CI excludes 0, sigma_seed "
                              "rule (per-dataset report)",
                      "kill": "killed as soon as the slot has failed on 2 datasets (registered order)",
                      "survive": "only with passes on at least 3 of the 4 datasets",
                      "holm": "Holm family = the datasets run (decided, with a p-value; a descriptive endpoint has none)",
                      "confirmed": "Holm p < 0.05 and the sigma_seed rule (A3 section 11)",
                      "undecided": "INCOMPLETE / INVALID datasets are not counted as failed; every later dataset is "
                                   "refused until they are decided",
                      "hard_kill_date": HARD_KILL_DATE,
                      "reporting": "a killed slot is reported once, in the appendix, as a negative result"},
            "datasets": {d: infos[d] or {"status": None, "reason": "not run"} for d in DATASETS},
            "datasets_run": run, **st,
            "holm": {"family": "the slot {datasets run} (A3 section 11)", "members": [d for d in st["decided"]
                                                                                      if _fin(pvals[d])],
                     "p_raw": pvals, "p_holm": adj, "confirmed": confirmed},
            "single_backbone": True, "backbones": backbones, "abstract_eligible": False,
            "abstract_rule": "single-backbone: it reaches the abstract only with a Llama replication (A3 sections 7 and "
                             "11, Amendment 2 F), which is not registered",
            "dry_run_inputs": any(i and i.get("dry_run_inputs") for i in infos.values())}


def slot_table_rows(s: dict) -> list:
    rows = []
    for k, d in enumerate(DATASETS):
        i = s["datasets"][d]
        rows.append({"dataset": d, "order": k + 1, "status": i.get("status") or "not run", "dUAUC": i.get("dUAUC"),
                     "lo": i.get("lo"), "hi": i.get("hi"), "p": i.get("p"), "p_holm": s["holm"]["p_holm"].get(d),
                     "confirmed": s["holm"]["confirmed"].get(d), "sigma_seed": i.get("sigma_seed"),
                     "sigma_seed_rule": i.get("sigma_seed_rule"), "n_users": i.get("n_users"),
                     "descriptive_min_n": i.get("descriptive_min_n"), "note": i.get("reason") or ""})
    rows.append({"dataset": "slot", "order": None, "status": s["state"], "note":
                 f"killed after {s['killed_after']}" if s["killed_after"] else
                 f"next dataset: {s['next_dataset']}" if s["next_dataset"] else "all datasets decided"})
    return rows


# ---------------------------------------------------------------- output
def csv_bytes(cols, rows) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow([fr._cell(r.get(c)) for c in cols])
    return buf.getvalue().encode("utf-8")


def write_outputs(out: Path, obj: dict, cols, rows) -> None:
    """JSON and <stem>_tables.csv, each rewritten only when its bytes change."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tlo.write_if_changed(out, (json.dumps(strict_json(obj), indent=2, allow_nan=False) + "\n").encode("utf-8"))
    tlo.write_if_changed(out.with_name(out.stem + "_tables.csv"), csv_bytes(cols, rows))


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset", help="the slot endpoint of one dataset")
    d.add_argument("--domain", required=True, choices=list(DATASETS))
    d.add_argument("--split", required=True, help="the ftgrid panels/<d>/ftgrid_split.json")
    d.add_argument("--panels", required=True, help="the ftgrid panels/<d>/ (eval.jsonl)")
    d.add_argument("--sft_scores", required=True, help="the ftgrid scores/<d>/ (s0-s2/like of the SFT comparators)")
    d.add_argument("--method_dir", required=True, help="outputs/confrec/ftmethod/<d>/ (q-hat, adapters, scores)")
    d.add_argument("--out", required=True, help="<method_dir>/report.json; the tables go to <stem>_tables.csv")
    d.add_argument("--n_boot", type=int, default=fr.N_BOOT)
    d.add_argument("--seed", type=int, default=fr.SEED)
    s = sub.add_parser("slot", help="the slot-level state over the per-dataset reports")
    s.add_argument("--root", required=True, help="outputs/confrec/ftmethod (holding <d>/report.json)")
    s.add_argument("--out", default=None, help="slot.json; the tables go to <stem>_tables.csv")
    s.add_argument("--check_next", default=None, choices=list(DATASETS),
                   help="exit 4 unless this dataset may run (registered order, kill rule)")
    a = ap.parse_args(argv)
    if a.cmd == "dataset" and a.n_boot < 0:
        ap.error("--n_boot must be >= 0")
    return a


def main(argv=None) -> dict:
    a = parse_args(argv)
    if a.cmd == "dataset":
        res = strict_json(build(a))
        out = Path(a.out)
        write_outputs(out, res, fr.CSV_COLS, table_rows(res))
        dec = res["decision"]
        print(f"wrote {out}: {a.domain} {dec['status']}" + (f" ({dec['reason']})" if dec.get("reason") else ""))
        return res
    summary = strict_json(slot_summary(a.root))
    if a.out:
        write_outputs(Path(a.out), summary, SLOT_COLS, slot_table_rows(summary))
    st = summary["state"]
    tail = (f"killed after {summary['killed_after']}: no further dataset runs" if st == "KILLED" else
            f"next dataset: {summary['next_dataset']}" if summary["next_dataset"] else "every dataset decided")
    print(f"slot state {st} (passes {summary['passes']}, fails {summary['fails']}); {tail}")
    for v in summary["violations"]:
        print(f"WARNING order violation: {v}", file=sys.stderr)
    if a.check_next:
        ok, why = check_next({d: (i or {}).get("status") for d, i in summary["datasets"].items()}, a.check_next)
        if not ok:
            print(why, file=sys.stderr)
            raise SystemExit(4)
        print(why)
    return summary


if __name__ == "__main__":
    main()
