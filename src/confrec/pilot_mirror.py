"""Pilot 1 (A1 MIRROR) analysis — pre-registered estimands in idea-stage/triage_verdict.md §3.1.

    python -m src.confrec.pilot_mirror --scores <dir>/scores.csv.gz --panel <panel.jsonl> \
        [--swap <dir>/swap_prior.csv.gz] --out <dir>/pilot_mirror.json

Arms per (user, candidate):
  raw      = logit_like                         (or logit_next on next-item panels)
  mirror   = logit_like - logit_dislike
  placebo  = (logit_like + logit_like_para) / 2  (equal-compute two-prompt ensemble)
  evidence = logit_like - pi(item)              (pi = mean logit under other users' histories)
Estimands: a(u,i) = (logit_like + logit_dislike)/2 centred within user then averaged per item;
v(i) = user-average of (logit_like - logit_dislike)/2; pi(i). Primary endpoint: Delta-UAUC(mirror - placebo)
with a paired bootstrap over users. Rated panels use candidate_labels; next-item panels use the positive.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict

import numpy as np

from src.confrec.metrics import auroc, brier, ece, ndcg_from_rank


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def load_scores(path):
    s = defaultdict(dict)  # (event, cand_idx) -> {question: logit}
    meta = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["source_event_id"], int(r["cand_idx"]))
            s[k][r["question"]] = float(r["logit"])
            meta[k] = (r["user_id"], r["item_id"], int(r["label"]))
    return s, meta


def uauc_per_user(users, score, label):
    out = {}
    for u, idx in users.items():
        y = label[idx]
        if 0 < y.sum() < len(y):
            out[u] = auroc(score[idx], y)
    return out


def boot_delta(a: dict, b: dict, n=2000, seed=0):
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "n_users": len(keys)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--swap", default=None)
    ap.add_argument("--base_q", default=None, help="question used for the `raw` arm (default like/next)")
    ap.add_argument("--nohist", default=None, help="scores.csv.gz of the same panel scored with --hist_len 0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    s, meta = load_scores(a.scores)
    keys = sorted(s)
    qs = set().union(*[set(v) for v in s.values()])
    base_q = a.base_q or ("like" if "like" in qs else "next")
    pop = {}
    groups = {}
    for line in open(a.panel, encoding="utf-8"):
        r = json.loads(line)
        if "candidate_popularity" in r:
            pop.update(dict(zip(r["candidate_item_ids"], r["candidate_popularity"])))
        if "candidate_popularity_groups" in r:
            groups.update(dict(zip(r["candidate_item_ids"], r["candidate_popularity_groups"])))

    user = np.array([meta[k][0] for k in keys])
    item = np.array([meta[k][1] for k in keys])
    label = np.array([meta[k][2] for k in keys])
    L = lambda q: np.array([s[k].get(q, np.nan) for k in keys])  # noqa: E731
    # `raw` = the base question; MIRROR/placebo are always built from the like/dislike/like_para family.
    arms = {"raw": L(base_q)}
    if base_q != "like" and "like" in qs:
        arms["raw_like"] = L("like")
    if {"like", "dislike"} <= qs:
        arms["mirror"] = L("like") - L("dislike")
    if {"like", "like_para"} <= qs:
        arms["placebo"] = (L("like") + L("like_para")) / 2
    if a.swap:
        acc = defaultdict(list)
        with gzip.open(a.swap, "rt", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                acc[r["item_id"]].append(float(r["logit"]))
        pi = {i: float(np.mean(v)) for i, v in acc.items()}
        arms["evidence"] = arms["raw"] - np.array([pi.get(i, np.nan) for i in item])
    if a.nohist:
        # PMI-style item-prior subtraction baseline (required by the novelty dossier): the same question
        # asked with NO user history, averaged per item.
        s0, m0 = load_scores(a.nohist)
        acc0 = defaultdict(list)
        for k, qd in s0.items():
            if base_q in qd:
                acc0[m0[k][1]].append(qd[base_q])
        prior0 = {i: float(np.mean(v)) for i, v in acc0.items()}
        arms["pmi_nohist"] = arms["raw"] - np.array([prior0.get(i, np.nan) for i in item])

    users = defaultdict(list)
    for j, u in enumerate(user):
        users[u].append(j)
    users = {u: np.array(v) for u, v in users.items()}
    is_next_item = all(label[v].sum() == 1 for v in users.values())
    res = {"n_rows": len(keys), "n_users": len(users), "base_question": base_q,
           "panel_type": "next_item" if is_next_item else "rated", "arms": {}}
    per_user = {}
    for name, sc in arms.items():
        ok = ~np.isnan(sc)
        per_user[name] = uauc_per_user({u: v[ok[v]] for u, v in users.items()}, sc, label)
        p = 1 / (1 + np.exp(-sc[ok]))
        res["arms"][name] = {
            "UAUC": float(np.mean(list(per_user[name].values()))) if per_user[name] else float("nan"),
            "AUC_global": auroc(sc[ok], label[ok]),
            "ECE_sigmoid": ece(p, label[ok]), "Brier_sigmoid": brier(p, label[ok]),
            # confidence = |p - 0.5| predicting whether the thresholded decision is correct
            "conf_error_AUROC": auroc(np.abs(p - 0.5), ((p >= 0.5) == label[ok]).astype(int)),
        }
        if is_next_item:
            ranks = []
            top10_head = []
            for u, v in users.items():
                order = v[np.argsort(-sc[v], kind="stable")]
                ranks.append(int(np.where(label[order] == 1)[0][0]) + 1 if label[v].any() else 999)
                if groups:
                    top10_head.append(np.mean([groups.get(item[j]) == "head" for j in order[:10]]))
            res["arms"][name]["NDCG@10"] = float(ndcg_from_rank(np.array(ranks)).mean())
            if top10_head:
                res["arms"][name]["top10_head_share"] = float(np.mean(top10_head))
    for x, y in [("mirror", "placebo"), ("evidence", "placebo"), ("mirror", "raw"), ("placebo", "raw"),
                 ("mirror", "pmi_nohist"), ("mirror", "evidence"), ("pmi_nohist", "raw")]:
        if x in per_user and y in per_user:
            res[f"dUAUC_{x}_minus_{y}"] = boot_delta(per_user[x], per_user[y])

    if {"like", "dislike"} <= qs:
        acq = (L("like") + L("dislike")) / 2
        val = (L("like") - L("dislike")) / 2
        centred = acq.copy()
        for u, v in users.items():
            centred[v] = acq[v] - np.nanmean(acq[v])
        a_item, v_item = defaultdict(list), defaultdict(list)
        for j, i in enumerate(item):
            a_item[i].append(centred[j])
            v_item[i].append(val[j])
        its = sorted(a_item)
        a_bar = np.array([np.nanmean(a_item[i]) for i in its])
        v_bar = np.array([np.nanmean(v_item[i]) for i in its])
        res["acquiescence"] = {"global_mean_a": float(np.nanmean(acq)), "SD_items_centred_a": float(np.nanstd(a_bar)),
                               "n_items": len(its)}
        if pop:
            lp = np.log1p([pop.get(i, 0) for i in its])
            res["acquiescence"]["spearman_a_logpop"] = spearman(a_bar, lp)
            res["valence_spearman_v_logpop"] = spearman(v_bar, lp)
            if a.swap:
                pis = [i for i in its if i in pi]
                res["prior_spearman_pi_logpop"] = spearman([pi[i] for i in pis],
                                                           np.log1p([pop.get(i, 0) for i in pis]))
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
