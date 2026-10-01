"""Pilot 2 (B1/A7) analysis — pre-registered in idea-stage/triage_verdict.md §3.2.

    python -m src.confrec.pilot_kuairec --scores <dir>/scores.csv.gz --panel kuairec.jsonl --out pilot_kuairec.json

Endpoints: (1) share of top-decile-confidence MNAR false positives that are MAR positives; (2) exposure gap =
ECE(logged/MNAR) - ECE(MAR) with a user-bootstrap CI; (3) sign of the confidence-matched head-vs-tail Bias
Index under each label view; plus AUC under each view (null gate: MAR AUC < 0.58).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.confrec.metrics import auroc, bias_index, ece
from src.confrec.pilot_mirror import load_scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--question", default="like")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s, meta = load_scores(a.scores)
    mnar, popm = {}, {}
    for line in open(a.panel, encoding="utf-8"):
        r = json.loads(line)
        for c, (m, p) in enumerate(zip(r["candidate_labels_mnar"], r["candidate_popularity"])):
            mnar[(r["source_event_id"], c)] = m
            popm[(r["source_event_id"], c)] = p
    keys = sorted(k for k in s if a.question in s[k])
    lg = np.array([s[k][a.question] for k in keys])
    p = 1 / (1 + np.exp(-lg))
    y_mar = np.array([meta[k][2] for k in keys])
    y_mnar = np.array([mnar[k] for k in keys])
    usr = np.array([meta[k][0] for k in keys])
    pop = np.array([popm[k] for k in keys], float)
    grp = np.where(pop >= np.quantile(pop, 0.8), "head", np.where(pop <= np.quantile(pop, 0.2), "tail", "mid"))

    top = p >= np.quantile(p, 0.9)
    fp_mnar = top & (y_mnar == 0)
    rng = np.random.default_rng(0)
    users = np.unique(usr)
    idx = {u: np.where(usr == u)[0] for u in users}
    gaps = []
    for _ in range(1000):
        sel = np.concatenate([idx[u] for u in rng.choice(users, len(users))])
        gaps.append(ece(p[sel], y_mnar[sel]) - ece(p[sel], y_mar[sel]))
    res = {
        "n_pairs": len(keys), "n_users": len(users),
        "AUC_MAR": auroc(lg, y_mar), "AUC_MNAR": auroc(lg, y_mnar),
        "ECE_MAR": ece(p, y_mar), "ECE_MNAR": ece(p, y_mnar),
        "exposure_gap": {"mean": float(ece(p, y_mnar) - ece(p, y_mar)),
                         "ci95": [float(np.quantile(gaps, .025)), float(np.quantile(gaps, .975))]},
        "top_decile_mnar_false_positives": int(fp_mnar.sum()),
        "share_fp_that_are_MAR_positive": float(y_mar[fp_mnar].mean()) if fp_mnar.any() else None,
        "bias_index_MAR": bias_index(p, y_mar, grp), "bias_index_MNAR": bias_index(p, y_mnar, grp),
        "pos_rate_MAR": float(y_mar.mean()), "pos_rate_MNAR": float(y_mnar.mean()),
    }
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
