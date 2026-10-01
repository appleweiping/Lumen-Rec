"""Pilot 3 (A2) analysis — pre-registered in idea-stage/triage_verdict.md §3.3.

    python -m src.confrec.pilot_pseudonym --real <dir_real>/scores.csv.gz --pseudo <dir_ps>/scores.csv.gz \
        --placebo <dir_pl>/scores.csv.gz --panel <panel_pseudo.jsonl> --out pilot_pseudonym.json

Only TREATED candidates (brand string present in their own title) enter the Δ analyses.
Δ(real−pseudo) by popularity decile; head(top 20% popularity) − tail(bottom 20%) with bootstrap CI;
placebo Δ(real−placebo); ΔUAUC overall/tail; confidence-matched Bias Index (head vs tail) per arm.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.confrec.metrics import auroc, bias_index
from src.confrec.pilot_mirror import load_scores, uauc_per_user


def arm(path, q="like"):
    s, meta = load_scores(path)
    return {k: v[q] for k, v in s.items() if q in v}, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    for x in ("real", "pseudo", "placebo", "panel", "out"):
        ap.add_argument(f"--{x}", required=True)
    a = ap.parse_args()
    real, meta = arm(a.real)
    ps, _ = arm(a.pseudo)
    pl, _ = arm(a.placebo)
    treated, popm = {}, {}
    for line in open(a.panel, encoding="utf-8"):
        r = json.loads(line)
        for c, (f, p) in enumerate(zip(r["candidate_brand_in_title"], r["candidate_popularity"])):
            treated[(r["source_event_id"], c)] = f
            popm[(r["source_event_id"], c)] = p
    keys = sorted(k for k in real if k in ps and k in pl and treated.get(k))
    pop = np.array([popm[k] for k in keys], float)
    d_ps = np.array([real[k] - ps[k] for k in keys])
    d_pl = np.array([real[k] - pl[k] for k in keys])
    lab = np.array([meta[k][2] for k in keys])
    usr = np.array([meta[k][0] for k in keys])
    q = np.quantile(pop, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(q, pop, side="right") - 1, 0, 9)
    hi, lo = pop >= np.quantile(pop, 0.8), pop <= np.quantile(pop, 0.2)
    rng = np.random.default_rng(0)

    def head_minus_tail(d):
        h, t = np.where(hi)[0], np.where(lo)[0]
        bs = [d[rng.choice(h, len(h))].mean() - d[rng.choice(t, len(t))].mean() for _ in range(2000)]
        return {"mean": float(d[hi].mean() - d[lo].mean()),
                "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}

    users = {}
    for j, u in enumerate(usr):
        users.setdefault(u, []).append(j)
    users = {u: np.array(v) for u, v in users.items()}
    tail_users = {u: v[lo[v]] for u, v in users.items()}
    res = {"n_treated_candidates": len(keys), "n_users": len(users),
           "delta_pseudo_by_decile": [float(d_ps[dec == b].mean()) for b in range(10)],
           "delta_placebo_by_decile": [float(d_pl[dec == b].mean()) for b in range(10)],
           "head_minus_tail_pseudo": head_minus_tail(d_ps),
           "head_minus_tail_placebo": head_minus_tail(d_pl),
           "mean_delta_pseudo": float(d_ps.mean()), "mean_delta_placebo": float(d_pl.mean())}
    for name, sc in (("real", np.array([real[k] for k in keys])), ("pseudo", np.array([ps[k] for k in keys])),
                     ("placebo", np.array([pl[k] for k in keys]))):
        uu = uauc_per_user(users, sc, lab)
        ut = uauc_per_user(tail_users, sc, lab)
        p = 1 / (1 + np.exp(-sc))
        grp = np.where(hi, "head", np.where(lo, "tail", "mid"))
        res[f"arm_{name}"] = {"UAUC": float(np.mean(list(uu.values()))) if uu else None,
                              "UAUC_tail_users": float(np.mean(list(ut.values()))) if ut else None,
                              "AUC_global": auroc(sc, lab),
                              "bias_index": bias_index(p, lab, grp)}
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
