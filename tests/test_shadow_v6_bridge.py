from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from src.shadow.decision_bridge import (
    build_shadow_v6_bridge_rows,
    build_shadow_v6_decision,
    build_shadow_v6_decision_predictions,
    rank_shadow_v6_bridge_rows,
)
from src.shadow.scoring import compute_shadow_scores

BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build" / "main_build_shadow_v6_bridge.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_shadow_v6_decision_accepts_reliable_signal() -> None:
    decision = build_shadow_v6_decision(
        signal_score=0.9,
        signal_uncertainty=0.1,
        anchor_score=0.5,
        matched_signal=True,
    )

    assert decision["fallback_flag"] is False
    assert decision["correction_gate"] > 0.0
    assert decision["decision_score"] > 0.5
    assert decision["pair_type"] == "shadow_preferred_over_anchor"
    assert decision["pair_weight"] > 0.0


def test_shadow_v6_decision_falls_back_without_signal() -> None:
    decision = build_shadow_v6_decision(
        signal_score=0.7,
        signal_uncertainty=0.2,
        anchor_score=0.4,
        matched_signal=False,
    )

    assert decision["fallback_flag"] is True
    assert decision["fallback_reason"] == "missing_signal"
    assert decision["correction_gate"] == 0.0
    assert decision["decision_score"] == 0.4
    assert decision["pair_weight"] == 0.0


def test_shadow_v6_bridge_prefers_event_specific_signal() -> None:
    ranking_records = [
        {
            "user_id": "u1",
            "source_event_id": "event_1",
            "positive_item_id": "b",
            "split_name": "test",
            "timestamp": 1,
            "candidate_item_ids": ["a", "b"],
            "candidate_titles": ["A", "B"],
            "candidate_popularity_groups": ["head", "tail"],
            "pred_ranked_item_ids": ["a", "b"],
            "topk_item_ids": ["a", "b"],
            "parse_success": True,
        },
        {
            "user_id": "u1",
            "source_event_id": "event_2",
            "positive_item_id": "a",
            "split_name": "test",
            "timestamp": 2,
            "candidate_item_ids": ["a", "b"],
            "candidate_titles": ["A", "B"],
            "candidate_popularity_groups": ["head", "tail"],
            "pred_ranked_item_ids": ["b", "a"],
            "topk_item_ids": ["b", "a"],
            "parse_success": True,
        },
    ]
    signal_records = [
        {
            "user_id": "u1",
            "source_event_id": "event_1",
            "candidate_item_id": "a",
            "shadow_calibrated_score": 0.1,
            "shadow_uncertainty": 0.05,
        },
        {
            "user_id": "u1",
            "source_event_id": "event_1",
            "candidate_item_id": "b",
            "shadow_calibrated_score": 0.95,
            "shadow_uncertainty": 0.05,
        },
        {
            "user_id": "u1",
            "source_event_id": "event_2",
            "candidate_item_id": "a",
            "shadow_calibrated_score": 0.95,
            "shadow_uncertainty": 0.05,
        },
        {
            "user_id": "u1",
            "source_event_id": "event_2",
            "candidate_item_id": "b",
            "shadow_calibrated_score": 0.1,
            "shadow_uncertainty": 0.05,
        },
    ]

    rows = build_shadow_v6_bridge_rows(
        ranking_records,
        signal_records,
        metadata={
            "dataset": "unit",
            "domain": "unit",
            "split": "test",
            "seed": 11,
            "method": "shadow_v6_decision_bridge",
            "model_backend": "mock",
            "backend": "mock",
            "prompt_id": "shadow_v6_signal_to_decision",
            "prompt_template_id": "shadow_v6_signal_to_decision",
            "config_hash": "unit-config",
            "timestamp_utc": "2026-01-01T00:00:00Z",
        },
    )
    ranked_rows = rank_shadow_v6_bridge_rows(rows)
    predictions = build_shadow_v6_decision_predictions(ranked_rows, ranking_records, topk=2)

    by_event = {row["source_event_id"]: row for row in predictions}
    assert by_event["event_1"]["pred_ranked_item_ids"][0] == "b"
    assert by_event["event_2"]["pred_ranked_item_ids"][0] == "a"
    assert by_event["event_1"]["config_hash"] == "unit-config"


def test_shadow_v6_scoring_uses_anchor_disagreement() -> None:
    low_disagreement = compute_shadow_scores(
        {
            "decision_score": 0.7,
            "signal_score": 0.7,
            "signal_uncertainty": 0.2,
            "correction_gate": 0.7,
            "anchor_score": 0.65,
            "anchor_disagreement": 0.05,
        },
        variant="shadow_v6",
    )
    high_disagreement = compute_shadow_scores(
        {
            "decision_score": 0.7,
            "signal_score": 0.7,
            "signal_uncertainty": 0.2,
            "correction_gate": 0.7,
            "anchor_score": 0.1,
            "anchor_disagreement": 0.6,
        },
        variant="shadow_v6",
    )

    assert high_disagreement["shadow_uncertainty"] > low_disagreement["shadow_uncertainty"]


def test_shadow_v6_cli_smoke(tmp_path: Path) -> None:
    rank_path = tmp_path / "rank_predictions.jsonl"
    signal_path = tmp_path / "test_calibrated.jsonl"
    output_root = tmp_path / "outputs"

    _write_jsonl(
        rank_path,
        [
            {
                "user_id": "u1",
                "source_event_id": "smoke_event",
                "positive_item_id": "b",
                "split_name": "test",
                "timestamp": 123,
                "candidate_item_ids": ["a", "b", "c"],
                "candidate_titles": ["A", "B", "C"],
                "candidate_popularity_groups": ["head", "tail", "mid"],
                "pred_ranked_item_ids": ["a", "b", "c"],
                "topk_item_ids": ["a", "b"],
                "parse_success": True,
                "latency": 0.0,
            }
        ],
    )
    _write_jsonl(
        signal_path,
        [
            {
                "user_id": "u1",
                "source_event_id": "smoke_event",
                "candidate_item_id": "a",
                "shadow_calibrated_score": 0.2,
                "shadow_uncertainty": 0.1,
            },
            {
                "user_id": "u1",
                "source_event_id": "smoke_event",
                "candidate_item_id": "b",
                "shadow_calibrated_score": 0.95,
                "shadow_uncertainty": 0.05,
            },
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--exp_name",
            "smoke_shadow_v6",
            "--rank_input_path",
            str(rank_path),
            "--signal_input_path",
            str(signal_path),
            "--output_root",
            str(output_root),
            "--winner_signal_variant",
            "shadow_v1",
            "--dataset",
            "smoke_dataset",
            "--domain",
            "smoke",
            "--split",
            "test",
            "--model_backend",
            "mock_bridge",
            "--artifact_class",
            "smoke",
            "--seed",
            "7",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1])
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    prediction_path = output_root / "smoke_shadow_v6" / "reranked" / "shadow_v6_decision_reranked.jsonl"
    rows_path = output_root / "smoke_shadow_v6" / "reranked" / "shadow_v6_bridge_rows.jsonl"
    result_path = output_root / "smoke_shadow_v6" / "tables" / "rerank_results.csv"
    predictions = _read_jsonl(prediction_path)
    result_rows = list(csv.DictReader(result_path.open(newline="", encoding="utf-8")))

    assert "Built 3 v6 bridge rows" in completed.stdout
    assert rows_path.exists()
    assert result_path.exists()
    assert predictions[0]["pred_ranked_item_ids"][0] == "b"
    assert predictions[0]["dataset"] == "smoke_dataset"
    assert predictions[0]["domain"] == "smoke"
    assert predictions[0]["artifact_class"] == "smoke"
    assert predictions[0]["is_paper_result"] is False
    assert predictions[0]["config_hash"]
    assert {row["method"] for row in result_rows} == {"direct_candidate_ranking", "shadow_v6_decision_bridge"}
    assert all(row["dataset"] == "smoke_dataset" for row in result_rows)
