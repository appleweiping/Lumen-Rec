# src/eval/ranking_metrics.py

from __future__ import annotations

import numpy as np
import pandas as pd


def _dcg_at_k(labels: list[int]) -> float:
    dcg = 0.0
    for idx, rel in enumerate(labels, start=1):
        dcg += (2 ** rel - 1) / np.log2(idx + 1)
    return float(dcg)


def _ndcg_for_user(user_df: pd.DataFrame, k: int) -> float:
    topk = user_df.nsmallest(k, "rank")
    actual_labels = topk["label"].astype(int).tolist()
    dcg = _dcg_at_k(actual_labels)

    ideal_labels = sorted(user_df["label"].astype(int).tolist(), reverse=True)[:k]
    idcg = _dcg_at_k(ideal_labels)

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def _hit_for_user(user_df: pd.DataFrame, k: int) -> float:
    topk = user_df.nsmallest(k, "rank")
    return float((topk["label"].astype(int) == 1).any())


def _mrr_for_user(user_df: pd.DataFrame, k: int) -> float:
    topk = user_df.nsmallest(k, "rank")
    positives = topk[topk["label"].astype(int) == 1]
    if len(positives) == 0:
        return 0.0
    best_rank = int(positives["rank"].min())
    return float(1.0 / best_rank)


def compute_ranking_metrics(
    ranked_df: pd.DataFrame,
    k: int = 10,
    user_col: str = "user_id",
    rank_col: str = "rank",
) -> dict[str, float | int]:
    """Compute binary relevance ranking metrics with a validated input contract.

    ``ranked_df`` must contain one row per user-item candidate, a positive
    integer rank, and a binary ``label`` (0 or 1).  Failing fast here prevents
    silent metric corruption from string ranks, fractional labels, NaNs, or an
    invalid cutoff.  Tied ranks remain supported because callers may use them
    for diagnostic outputs.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("`k` must be a positive integer.")
    if user_col not in ranked_df.columns:
        raise ValueError(f"Column `{user_col}` not found in dataframe.")
    if rank_col not in ranked_df.columns:
        raise ValueError(f"Column `{rank_col}` not found in dataframe.")
    if "label" not in ranked_df.columns:
        raise ValueError("Column `label` not found in dataframe.")
    if ranked_df[user_col].isna().any():
        raise ValueError(f"Column `{user_col}` must not contain missing values.")

    numeric_ranks = pd.to_numeric(ranked_df[rank_col], errors="coerce")
    if numeric_ranks.isna().any() or (~np.isfinite(numeric_ranks)).any():
        raise ValueError(f"Column `{rank_col}` must contain finite numeric ranks.")
    if (numeric_ranks < 1).any() or (numeric_ranks % 1 != 0).any():
        raise ValueError(f"Column `{rank_col}` must contain positive integer ranks.")

    numeric_labels = pd.to_numeric(ranked_df["label"], errors="coerce")
    if numeric_labels.isna().any() or (~np.isfinite(numeric_labels)).any():
        raise ValueError("Column `label` must contain finite binary values.")
    if ~numeric_labels.isin([0, 1]).all():
        raise ValueError("Column `label` must contain only 0 or 1.")

    df = ranked_df.copy()
    df["rank"] = numeric_ranks.astype(int)
    df["label"] = numeric_labels.astype(int)

    hits = []
    ndcgs = []
    mrrs = []

    for _, user_df in df.groupby(user_col):
        user_df = user_df.sort_values("rank").copy()
        hits.append(_hit_for_user(user_df, k))
        ndcgs.append(_ndcg_for_user(user_df, k))
        mrrs.append(_mrr_for_user(user_df, k))

    return {
        f"HR@{k}": float(np.mean(hits)) if hits else float("nan"),
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        f"MRR@{k}": float(np.mean(mrrs)) if mrrs else float("nan"),
        "num_users": df[user_col].nunique(),
        "num_samples": len(df),
    }
