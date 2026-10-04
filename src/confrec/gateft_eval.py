"""Gate-FT evaluation (idea-stage/PREREG_AMENDMENT_2.md G9): the registered pass rule and its context.

    python -m src.confrec.gateft_eval --confirm_panel outputs/confrec/gatefix/panels/ml1m_confirm_h20.jsonl \
        --split_report outputs/confrec/gateft/gateft_split.json --seed_dirs <s0>,<s1>,<s2> \
        --zeroshot_dir <zero-shot like scores of the same prompt on the same panel> --out outputs/confrec/gateft/gate_ft.json

Each seed directory is a pyes_scorer output (`--lora <adapter> --questions like`) on the whole CONFIRM panel.
Primary endpoint: UAUC(like) over the CONFIRM users' candidates with ts >= T (T from gateft_split.json) = mean over users
with both classes among those candidates of the per-user AUC (ties averaged), averaged over the seeds.
**PASS iff the mean over seeds of that UAUC is >= 0.65 (point estimate; the registered constant) and every seed run
passes the integrity checks** (censored=2 share <= 0.5% of the like rows, no overlength prompt, mean Yes+No mass
>= 0.95; the same as gatefix_select's E1) and exactly 3 seeds were given. FAIL closes the rated-panel method line.
Reported, never gating: per-seed UAUC and SD across seeds, the 95% user-bootstrap CI of the seed-averaged UAUC, the
all-CONFIRM UAUC, the zero-shot UAUC on the same rows and the paired difference (user bootstrap), ECE / Brier / Platt slope
of sigmoid(L) on the post-T rows.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.confrec import pilot_mirror as pm
from src.confrec.metrics import brier, ece
from src.confrec.stats import paired_bootstrap, platt_fit, sigmoid, strict_json

UAUC_MIN, N_SEEDS = 0.65, 3
CENS2_MAX, MASS_MIN = 0.005, 0.95
N_BOOT, SEED = 2000, 0
NAN = float("nan")


def panel_timestamps(path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r.get("source_event_id", r["user_id"]))] = [float(t) for t in r["candidate_timestamps"]]
    return out


def run_stats(run_dir, ts_map: dict, T: float) -> dict:
    """Per-user AUCs (post-T and all), integrity and calibration of one scored directory."""
    d = Path(run_dir)
    sc = pm.load_scores(d / "scores.csv.gz")
    rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
    L, y = sc["L"]["like"], sc["label"]
    try:
        ts = np.array([ts_map[ev][c] for ev, c in sc["keys"]], float)
    except (KeyError, IndexError) as e:
        raise SystemExit(f"{d}: score row {e} is not in the CONFIRM panel (wrong panel for this run?)") from None
    ok, post = np.isfinite(L), ts >= T
    users = pm._groups(sc["user"])
    cen = sc["censoring"].get("like", {})
    n = int(cen.get("n", 0))
    n2 = int(cen.get("2", 0))
    over = max(int(cen.get("3", 0)), int(rep.get("n_overlength") or 0))
    mass = float(rep.get("mean_yes_no_mass", NAN))
    integrity = {"n_rows": n, "censored2_share": n2 / n if n else NAN, "overlength": over, "mean_yes_no_mass": mass,
                 "ok": bool(n and n2 / n <= CENS2_MAX and over == 0 and mass >= MASS_MIN)}
    m = ok & post
    cal = {"n_rows_post_T": int(m.sum())}
    if m.sum() > 2 and 0 < y[m].sum() < m.sum():
        p = sigmoid(L[m])
        a, b = platt_fit(L[m], y[m])
        cal.update(ECE=ece(p, y[m], 15), Brier=brier(p, y[m]), platt_slope=a, platt_intercept=b,
                   mean_logit=float(L[m].mean()), positive_rate=float(y[m].mean()))
    return {"dir": str(d), "variant": rep.get("variant"), "lora": rep.get("lora"), "model": rep.get("model"),
            "hist_len": rep.get("hist_len"), "data_sha1": rep.get("data_sha1"), "integrity": integrity,
            "calibration_post_T": cal, "auc_post": pm._uauc(L, y, users, m), "auc_all": pm._uauc(L, y, users, ok)}


def mean(d: dict) -> float:
    return float(np.mean(list(d.values()))) if d else NAN


def evaluate(confirm_panel, split_report, seed_dirs, zeroshot_dir=None, n_boot: int = N_BOOT) -> dict:
    T = float(json.loads(Path(split_report).read_text(encoding="utf-8"))["T"])
    ts_map = panel_timestamps(confirm_panel)
    runs = [run_stats(d, ts_map, T) for d in seed_dirs]
    per_seed = [mean(r["auc_post"]) for r in runs]
    problems = []
    if len(runs) != N_SEEDS:
        problems.append(f"{len(runs)} seed runs given, {N_SEEDS} registered")
    if len({(r["variant"], r["hist_len"], r["model"], r["data_sha1"]) for r in runs}) != 1:
        problems.append("seed runs differ in variant / hist_len / model / panel")
    if len({r["lora"] for r in runs}) != len(runs) or any(not r["lora"] for r in runs):
        problems.append("each seed run must score its own adapter (report.json lora)")
    bad = [r["dir"] for r in runs if not r["integrity"]["ok"]]
    if bad:
        problems.append(f"integrity checks failed for {bad}")
    common = set.intersection(*[set(r["auc_post"]) for r in runs]) if runs else set()
    avg = {u: float(np.mean([r["auc_post"][u] for r in runs])) for u in sorted(common)}
    v = np.array([avg[u] for u in sorted(avg)], float)
    ci = pm.unit_bootstrap(lambda idx: float(v[idx].mean()), len(v), n_boot=n_boot, seed=SEED) if len(v) else {}
    u_mean = float(np.mean(per_seed)) if per_seed else NAN
    passed = bool(not problems and math.isfinite(u_mean) and u_mean >= UAUC_MIN)
    decision = "GATE_FT_PASS" if passed else ("GATE_FT_INCOMPLETE" if problems else "GATE_FT_FAIL")
    out = {"decision": decision, "gate_pass": passed, "UAUC_post_T_mean_over_seeds": u_mean, "UAUC_min": UAUC_MIN,
           "UAUC_post_T_per_seed": per_seed, "UAUC_post_T_sd_over_seeds": float(np.std(per_seed, ddof=1))
           if len(per_seed) > 1 else NAN,
           "UAUC_post_T_seed_averaged_ci95": {k: ci.get(k, NAN) for k in ("est", "lo", "hi")} | {"n_users": len(v),
                                                                                                    "n_boot": n_boot},
           "UAUC_all_confirm_per_seed": [mean(r["auc_all"]) for r in runs], "problems": problems, "T": T,
           "variant": runs[0]["variant"] if runs else None,
           "runs": [{k: r[k] for k in ("dir", "lora", "integrity", "calibration_post_T")} for r in runs],
           "rule": "PREREG_AMENDMENT_2.md G9: PASS iff the mean over 3 seeds of UAUC on CONFIRM candidates with "
                   f"ts >= T is >= {UAUC_MIN} (point estimate) and every run passes the integrity checks"}
    if zeroshot_dir:
        z = run_stats(zeroshot_dir, ts_map, T)
        zc = set(z["auc_post"]) & set(avg)
        d = paired_bootstrap({u: avg[u] for u in zc}, {u: z["auc_post"][u] for u in zc}, n_boot=n_boot, seed=SEED)
        out["zero_shot_context"] = {"note": "reported only, never gates", "dir": z["dir"], "variant": z["variant"],
                                    "UAUC_post_T": mean({u: z["auc_post"][u] for u in zc}),
                                    "UAUC_all_confirm": mean(z["auc_all"]), "integrity": z["integrity"],
                                    "calibration_post_T": z["calibration_post_T"],
                                    "dUAUC_finetuned_minus_zeroshot_post_T": d}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm_panel", required=True)
    ap.add_argument("--split_report", required=True)
    ap.add_argument("--seed_dirs", required=True, help="comma-separated pyes_scorer output dirs, one per seed")
    ap.add_argument("--zeroshot_dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    res = evaluate(a.confirm_panel, a.split_report, [d for d in a.seed_dirs.split(",") if d], a.zeroshot_dir,
                   a.n_boot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(strict_json(res), indent=2, allow_nan=False), encoding="utf-8")
    ci = res["UAUC_post_T_seed_averaged_ci95"]
    print(f"UAUC(post-T, mean of {len(res['UAUC_post_T_per_seed'])} seeds) = {res['UAUC_post_T_mean_over_seeds']:.4f} "
          f"(registered >= {UAUC_MIN}); seeds {[round(x, 4) for x in res['UAUC_post_T_per_seed']]}; "
          f"95% user CI [{ci['lo']:.4f}, {ci['hi']:.4f}] on {ci['n_users']} users")
    for p in res["problems"]:
        print("PROBLEM:", p)
    print("DECISION:", res["decision"])
    return {"GATE_FT_PASS": 0, "GATE_FT_FAIL": 3}.get(res["decision"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
