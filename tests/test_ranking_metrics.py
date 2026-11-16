from __future__ import annotations

import unittest

import pandas as pd

from src.eval.ranking_metrics import compute_ranking_metrics


class RankingMetricsTest(unittest.TestCase):
    def test_single_positive_ranking_metrics(self) -> None:
        df = pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "a", "label": 0, "rank": 1},
                {"user_id": "u1", "item_id": "b", "label": 1, "rank": 2},
                {"user_id": "u2", "item_id": "c", "label": 1, "rank": 1},
                {"user_id": "u2", "item_id": "d", "label": 0, "rank": 2},
            ]
        )
        metrics = compute_ranking_metrics(df, k=1)
        self.assertAlmostEqual(metrics["HR@1"], 0.5)
        self.assertAlmostEqual(metrics["MRR@1"], 0.5)

    def test_custom_rank_column_is_supported(self) -> None:
        df = pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "a", "label": 0, "position": 2},
                {"user_id": "u1", "item_id": "b", "label": 1, "position": 1},
            ]
        )
        metrics = compute_ranking_metrics(df, k=1, rank_col="position")
        self.assertEqual(metrics["HR@1"], 1.0)
        self.assertEqual(metrics["NDCG@1"], 1.0)

    def test_rejects_invalid_cutoff(self) -> None:
        df = pd.DataFrame([{"user_id": "u1", "label": 1, "rank": 1}])
        with self.assertRaisesRegex(ValueError, "positive integer"):
            compute_ranking_metrics(df, k=0)

    def test_rejects_non_binary_labels(self) -> None:
        df = pd.DataFrame([{"user_id": "u1", "label": 2, "rank": 1}])
        with self.assertRaisesRegex(ValueError, "only 0 or 1"):
            compute_ranking_metrics(df)

    def test_rejects_fractional_or_missing_ranks(self) -> None:
        fractional = pd.DataFrame([{"user_id": "u1", "label": 1, "rank": 1.5}])
        with self.assertRaisesRegex(ValueError, "positive integer ranks"):
            compute_ranking_metrics(fractional)

        missing = pd.DataFrame([{"user_id": "u1", "label": 1, "rank": None}])
        with self.assertRaisesRegex(ValueError, "finite numeric ranks"):
            compute_ranking_metrics(missing)


if __name__ == "__main__":
    unittest.main()
