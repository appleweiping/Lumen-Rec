"""Diagnostic: is within-user like/dislike discrimination on the rated panels intrinsically hard, or is the LLM
channel weak? Compares per-user AUC (UAUC) of the LLM's like-logit with simple non-LLM item signals on the SAME
panel pairs. Item signals use only OTHER users' ratings (leave-user-out), so the user's own label never leaks.

    python -m src.confrec.diag_rated_baselines --panel outputs/confrec/panels/ml1m_rated.jsonl --source ml1m \
        --raw data/raw/ml-1m --scores outputs/confrec/pilot1_mirror/ml1m_rated/scores.csv.gz --out diag.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .build_rated_panels import load_amazon, load_ml1m
from .metrics import auroc
from .stats import paired_bootstrap, strict_json


def uauc(df: pd.DataFrame, col: str) -> dict:
    out = {}
    for u, g in df.groupby("user_id", sort=False):
        s, y = g[col].to_numpy(float), g["label"].to_numpy(int)
        ok = np.isfinite(s)
        if ok.sum() >= 2 and 0 < y[ok].sum() < ok.sum():
            out[u] = auroc(s[ok], y[ok])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--source", choices=["ml1m", "amazon"], required=True)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _, events = load_ml1m(Path(a.raw)) if a.source == "ml1m" else load_amazon(Path(a.raw), a.domain)
    s_sum, s_n, s_sq = defaultdict(float), defaultdict(int), defaultdict(float)
    by_user = {}
    for u, evs in events.items():
        seen = {}
        for ts, i, r in sorted(evs):
            seen.setdefault(i, r)          # first rating per (user, item), as the panel builder does
        by_user[u] = seen
        for i, r in seen.items():
            s_sum[i] += r; s_n[i] += 1; s_sq[i] += r * r
    rows = []
    for line in open(a.panel, encoding="utf-8"):
        rec = json.loads(line)
        u, mine = rec["user_id"], by_user.get(rec["user_id"], {})
        for k, (i, y) in enumerate(zip(rec["candidate_item_ids"], rec["candidate_labels"])):
            n = s_n[i] - (1 if i in mine else 0)
            tot = s_sum[i] - mine.get(i, 0.0)
            sq = s_sq[i] - mine.get(i, 0.0) ** 2
            mean = tot / n if n > 0 else np.nan
            var = sq / n - mean ** 2 if n > 1 else np.nan
            # Bayesian-shrunk mean (prior = global mean 3.6, 5 pseudo-ratings) keeps tail items comparable
            shr = (tot + 5 * 3.6) / (n + 5)
            rows.append((u, rec["source_event_id"], k, i, int(y), mean, shr, n, var))
    df = pd.DataFrame(rows, columns=["user_id", "source_event_id", "cand_idx", "item_id", "label",
                                     "item_mean_loo", "item_mean_shrunk", "item_pop_loo", "item_var_loo"])
    sc = pd.read_csv(a.scores)
    for q in sorted(sc.question.unique()):
        m = sc[sc.question == q][["source_event_id", "cand_idx", "logit"]].rename(columns={"logit": f"llm_{q}"})
        df = df.merge(m, on=["source_event_id", "cand_idx"], how="left")
    if {"llm_like", "llm_dislike"} <= set(df.columns):
        df["llm_mirror"] = df["llm_like"] - df["llm_dislike"]
    df["neg_item_var"] = -df["item_var_loo"]
    cols = ["item_mean_loo", "item_mean_shrunk", "item_pop_loo", "neg_item_var"] + \
           [c for c in df.columns if c.startswith("llm_")]
    res = {"n_pairs": int(len(df)), "n_users": int(df.user_id.nunique()), "like_rate": float(df.label.mean()),
           "uauc": {}, "corr_with_llm_like": {}, "paired_vs_llm_like": {}}
    per = {c: uauc(df, c) for c in cols}
    for c in cols:
        v = list(per[c].values())
        res["uauc"][c] = {"est": float(np.mean(v)) if v else float("nan"), "n_users": len(v)}
    if "llm_like" in per:
        for c in cols:
            if c != "llm_like":
                res["paired_vs_llm_like"][c] = paired_bootstrap(per[c], per["llm_like"], n_boot=1000)
            ok = df[[c, "llm_like"]].dropna()
            res["corr_with_llm_like"][c] = float(ok.rank().corr().iloc[0, 1]) if len(ok) > 2 else float("nan")
    Path(a.out).write_text(json.dumps(strict_json(res), indent=2), encoding="utf-8")
    print(json.dumps(strict_json(res["uauc"]), indent=1))
    print("vs llm_like:", json.dumps(strict_json({k: {x: round(v[x], 4) for x in ("est", "lo", "hi")}
                                                   for k, v in res["paired_vs_llm_like"].items()})))


if __name__ == "__main__":
    main()
