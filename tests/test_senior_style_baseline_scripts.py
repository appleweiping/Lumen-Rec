from __future__ import annotations

import csv
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

TRAIN_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "train"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_task(tmp_path: Path) -> tuple[Path, Path, Path]:
    task_dir = tmp_path / "task"
    train_rows = [
        {"user_id": "u1", "item_id": "i1", "timestamp": 1, "sequence_index": 1},
        {"user_id": "u1", "item_id": "i2", "timestamp": 2, "sequence_index": 2},
        {"user_id": "u1", "item_id": "i3", "timestamp": 3, "sequence_index": 3},
        {"user_id": "u2", "item_id": "i2", "timestamp": 1, "sequence_index": 1},
        {"user_id": "u2", "item_id": "i4", "timestamp": 2, "sequence_index": 2},
        {"user_id": "u2", "item_id": "i5", "timestamp": 3, "sequence_index": 3},
    ]
    candidate_rows = [
        {"source_event_id": "e1", "user_id": "u1", "item_id": "i3", "candidate_index": 0},
        {"source_event_id": "e1", "user_id": "u1", "item_id": "i4", "candidate_index": 1},
        {"source_event_id": "e1", "user_id": "u1", "item_id": "i5", "candidate_index": 2},
        {"source_event_id": "e2", "user_id": "u2", "item_id": "i1", "candidate_index": 0},
        {"source_event_id": "e2", "user_id": "u2", "item_id": "i2", "candidate_index": 1},
        {"source_event_id": "e2", "user_id": "u2", "item_id": "i5", "candidate_index": 2},
    ]
    item_map_rows = [{"item_id": f"i{idx}", "llmesr_item_idx": idx} for idx in range(1, 6)]
    _write_csv(task_dir / "train_interactions.csv", train_rows, ["user_id", "item_id", "timestamp", "sequence_index"])
    _write_csv(task_dir / "candidate_items.csv", candidate_rows, ["source_event_id", "user_id", "item_id", "candidate_index"])
    item_map_path = tmp_path / "item_id_map.csv"
    _write_csv(item_map_path, item_map_rows, ["item_id", "llmesr_item_idx"])
    embedding_path = tmp_path / "itm_emb_np.pkl"
    matrix = [[float(idx), float(idx + 1), float(idx + 2), float(idx + 3)] for idx in range(5)]
    with embedding_path.open("wb") as fh:
        pickle.dump(matrix, fh)
    return task_dir, embedding_path, item_map_path


def _read_scores(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_llmemb_style_script_scores_all_candidates(tmp_path: Path) -> None:
    task_dir, embedding_path, item_map_path = _make_task(tmp_path)
    scores_path = tmp_path / "llmemb_scores.csv"

    subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPTS / "main_train_llmemb_style_same_candidate.py"),
            "--task_dir",
            str(task_dir),
            "--embedding_path",
            str(embedding_path),
            "--item_map_path",
            str(item_map_path),
            "--output_scores_path",
            str(scores_path),
            "--hidden_size",
            "8",
            "--num_heads",
            "1",
            "--epochs",
            "1",
            "--batch_size",
            "2",
            "--device",
            "cpu",
            "--log_every",
            "0",
        ],
        check=True,
    )

    rows = _read_scores(scores_path)
    assert len(rows) == 6
    assert {row["source_event_id"] for row in rows} == {"e1", "e2"}
    assert all(row["score"] for row in rows)


def test_rlmrec_style_script_scores_all_candidates(tmp_path: Path) -> None:
    task_dir, embedding_path, item_map_path = _make_task(tmp_path)
    scores_path = tmp_path / "rlmrec_scores.csv"

    subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPTS / "main_train_rlmrec_style_same_candidate.py"),
            "--task_dir",
            str(task_dir),
            "--embedding_path",
            str(embedding_path),
            "--item_map_path",
            str(item_map_path),
            "--output_scores_path",
            str(scores_path),
            "--embedding_size",
            "8",
            "--epochs",
            "1",
            "--batch_size",
            "4",
            "--device",
            "cpu",
            "--log_every",
            "0",
        ],
        check=True,
    )

    rows = _read_scores(scores_path)
    assert len(rows) == 6
    assert {row["source_event_id"] for row in rows} == {"e1", "e2"}
    assert all(row["score"] for row in rows)


def test_irllrec_style_script_scores_all_candidates(tmp_path: Path) -> None:
    task_dir, embedding_path, item_map_path = _make_task(tmp_path)
    scores_path = tmp_path / "irllrec_scores.csv"

    subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPTS / "main_train_irllrec_style_same_candidate.py"),
            "--task_dir",
            str(task_dir),
            "--embedding_path",
            str(embedding_path),
            "--item_map_path",
            str(item_map_path),
            "--output_scores_path",
            str(scores_path),
            "--hidden_size",
            "8",
            "--num_intents",
            "2",
            "--epochs",
            "1",
            "--batch_size",
            "2",
            "--device",
            "cpu",
            "--log_every",
            "0",
        ],
        check=True,
    )

    rows = _read_scores(scores_path)
    assert len(rows) == 6
    assert {row["source_event_id"] for row in rows} == {"e1", "e2"}
    assert all(row["score"] for row in rows)


def test_setrec_style_script_scores_all_candidates(tmp_path: Path) -> None:
    task_dir, embedding_path, item_map_path = _make_task(tmp_path)
    scores_path = tmp_path / "setrec_scores.csv"

    subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPTS / "main_train_setrec_style_same_candidate.py"),
            "--task_dir",
            str(task_dir),
            "--embedding_path",
            str(embedding_path),
            "--item_map_path",
            str(item_map_path),
            "--output_scores_path",
            str(scores_path),
            "--hidden_size",
            "8",
            "--epochs",
            "1",
            "--batch_size",
            "2",
            "--device",
            "cpu",
            "--log_every",
            "0",
        ],
        check=True,
    )

    rows = _read_scores(scores_path)
    assert len(rows) == 6
    assert {row["source_event_id"] for row in rows} == {"e1", "e2"}
    assert all(row["score"] for row in rows)
