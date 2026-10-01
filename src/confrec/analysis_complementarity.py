"""CPU pilot: per-user complementarity between the LLM scorer (C-CRP) and official baselines.

Reads the local `tables/ranking_eval_records.csv` of every method on a domain's frozen 1+100 panel and
reports, per baseline: NDCG@10 of each method, the per-user oracle of {LLM, baseline}, the fraction of
users where the baseline strictly beats the LLM, and the same split by the target's popularity group.
This bounds the headroom of any LLM->baseline deferral policy before any GPU work.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.confrec.metrics import ndcg_from_rank

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[2] / "outputs"
BASELINES = ["elmrec_graph", "irllrec_intent", "llm2rec_sasrec", "llmemb", "llmesr_sasrec",
             "proex_profile", "promax_profile", "rlmrec_graphcl"]


def load_ranks(path: Path) -> dict[str, tuple[int, str]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["source_event_id"]] = (int(row["positive_rank"]), row["positive_popularity_group"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="sports,toys,home,tools")
    ap.add_argument("--out", default="outputs/confrec_pilot/complementarity.json")
    args = ap.parse_args()
    result = {}
    for d in args.domains.split(","):
        llm = load_ranks(ROOT / f"{d}_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate"
                         / "tables" / "ranking_eval_records.csv")
        keys = sorted(llm)
        r_llm = np.array([llm[k][0] for k in keys])
        grp = np.array([llm[k][1] for k in keys])
        n_llm = ndcg_from_rank(r_llm)
        dom = {"n_users": len(keys), "llm_ndcg10": float(n_llm.mean()),
               "llm_ndcg10_by_group": {g: float(n_llm[grp == g].mean()) for g in np.unique(grp)},
               "group_share": {g: float((grp == g).mean()) for g in np.unique(grp)}, "baselines": {}}
        for b in BASELINES:
            p = (ROOT / "baselines" / "official_adapters"
                 / f"{d}_large10000_100neg_{b}_official_qwen3base_same_candidate" / "tables"
                 / "ranking_eval_records.csv")
            base = load_ranks(p)
            assert set(base) == set(keys), f"{d}/{b}: event keys differ from the LLM panel"
            n_b = ndcg_from_rank(np.array([base[k][0] for k in keys]))
            oracle = np.maximum(n_llm, n_b)
            dom["baselines"][b] = {
                "ndcg10": float(n_b.mean()),
                "oracle_ndcg10": float(oracle.mean()),
                "oracle_gain_over_llm": float(oracle.mean() - n_llm.mean()),
                "frac_base_strictly_better": float((n_b > n_llm).mean()),
                "frac_llm_strictly_better": float((n_llm > n_b).mean()),
                "base_better_by_group": {g: float((n_b[grp == g] > n_llm[grp == g]).mean())
                                         for g in np.unique(grp)},
                "ndcg10_by_group": {g: float(n_b[grp == g].mean()) for g in np.unique(grp)},
            }
        result[d] = dom
        best = max(dom["baselines"].items(), key=lambda kv: kv[1]["oracle_gain_over_llm"])
        print(f"{d}: LLM NDCG@10={dom['llm_ndcg10']:.4f} | best-complement={best[0]} "
              f"oracle={best[1]['oracle_ndcg10']:.4f} (+{best[1]['oracle_gain_over_llm']:.4f}), "
              f"base>LLM on {best[1]['frac_base_strictly_better']:.1%} users | "
              f"LLM by group {dom['llm_ndcg10_by_group']}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
