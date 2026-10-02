import json
import sys

import numpy as np

from src.confrec import split_panel as sp


def _row(uid, ts, labels):
    n = len(ts)
    return {"user_id": uid, "source_event_id": f"{uid}::{ts[0]}", "history": ["h (rated 5/5)"],
            "candidate_item_ids": [f"{uid}_i{k}" for k in range(n)], "candidate_titles": [f"t{k}" for k in range(n)],
            "candidate_labels": labels, "candidate_ratings": [5.0 if l else 1.0 for l in labels],
            "candidate_popularity": list(range(n)), "candidate_timestamps": ts}


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["split_panel", *args])
    sp.main()


def test_user_split_excludes_listed_users_from_train_only(tmp_path, monkeypatch):
    users = [f"u{i}" for i in range(200)]
    rows = [_row(u, [1, 2, 3, 4], [1, 0, 1, 0]) for u in users]
    _write(tmp_path / "p.jsonl", rows)
    split_of = {u: sp.bucket(u, 0, [0.8, 0.1, 0.1]) for u in users}
    train_users = [u for u in users if split_of[u] == 0]
    test_users = [u for u in users if split_of[u] == 2]
    excl = train_users[:7] + test_users[:2]                       # test users stay in test
    (tmp_path / "valid_sel.csv").write_text("user_id,selection_index\n" +
                                            "".join(f"{u},{k}\n" for k, u in enumerate(excl[:4])))
    (tmp_path / "test_sel.csv").write_text("user_id,selection_index\n" +
                                           "".join(f"{u},{k}\n" for k, u in enumerate(excl[4:])))
    _run(monkeypatch, "--panel", str(tmp_path / "p.jsonl"),
         "--exclude_users_csv", f"{tmp_path / 'valid_sel.csv'},{tmp_path / 'test_sel.csv'}")
    tr = {r["user_id"] for r in _read(tmp_path / "p_train.jsonl")}
    te = {r["user_id"] for r in _read(tmp_path / "p_test.jsonl")}
    assert tr == set(train_users) - set(excl)
    assert te == set(test_users)
    rep = json.loads((tmp_path / "p_split.json").read_text())
    assert rep["excluded_from_train"] == 7 and rep["n_exclude_users"] == 9


def test_user_split_without_exclusion_is_byte_identical(tmp_path, monkeypatch):
    rows = [_row(f"u{i}", [1, 2], [1, 0]) for i in range(50)]
    src = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)   # non-default formatting kept
    (tmp_path / "p.jsonl").write_text(src, encoding="utf-8")
    _run(monkeypatch, "--panel", str(tmp_path / "p.jsonl"))
    out = "".join(open(tmp_path / f"p_{n}.jsonl", encoding="utf-8").read() for n in ("train", "val", "test"))
    assert sorted(out.splitlines()) == sorted(src.splitlines())


def test_global_time_split(tmp_path, monkeypatch):
    rows = [
        _row("a", [1, 2, 3, 4, 90, 95], [1, 0, 1, 0, 1, 0]),   # contributes to both splits
        _row("b", [5, 6, 7, 8, 9, 10], [1, 1, 0, 0, 1, 0]),    # train only
        _row("c", [3, 4, 96, 97, 98, 99], [1, 1, 1, 0, 1, 0]),  # train part single-class -> dropped
        _row("d", [11, 12, 13, 14, 15, 100], [0, 1, 0, 1, 0, 1]),  # test part has 1 candidate -> dropped
    ]
    _write(tmp_path / "p.jsonl", rows)
    (tmp_path / "ex.csv").write_text("user_id\nb\n")
    _run(monkeypatch, "--panel", str(tmp_path / "p.jsonl"), "--time_split", "0.7", "--exclude_users_csv",
         str(tmp_path / "ex.csv"))
    T = float(np.quantile(np.concatenate([r["candidate_timestamps"] for r in rows]).astype(float), 0.7))
    assert 15 < T <= 90
    train, test = _read(tmp_path / "p_time_train.jsonl"), _read(tmp_path / "p_time_test.jsonl")
    assert all(t < T for r in train for t in r["candidate_timestamps"])
    assert all(t >= T for r in test for t in r["candidate_timestamps"])
    assert [r["user_id"] for r in train] == ["a", "d"]           # b excluded, c single-class
    assert [r["user_id"] for r in test] == ["a", "c"]            # d left with one candidate
    a_tr = train[0]
    assert a_tr["candidate_item_ids"] == ["a_i0", "a_i1", "a_i2", "a_i3"]
    assert a_tr["candidate_labels"] == [1, 0, 1, 0] and a_tr["candidate_ratings"] == [5.0, 1.0, 5.0, 1.0]
    assert a_tr["candidate_popularity"] == [0, 1, 2, 3] and a_tr["history"] == rows[0]["history"]
    c_te = test[1]
    assert c_te["candidate_titles"] == ["t2", "t3", "t4", "t5"] and c_te["candidate_timestamps"] == [96, 97, 98, 99]
    rep = json.loads((tmp_path / "p_time_split.json").read_text())
    assert rep["mode"] == "time" and abs(rep["T"] - T) < 1e-9
    assert rep["dropped_unusable"] == {"train": 1, "test": 1} and rep["excluded_from_train"] == 1


def test_time_split_rejects_misaligned_candidate_lists():
    r = _row("a", [1, 2, 3], [1, 0, 1])
    r["candidate_titles"] = ["only one"]
    try:
        sp.time_split([r], 0.5)
    except ValueError as e:
        assert "candidate_titles" in str(e)
    else:
        raise AssertionError("misaligned lists must raise")
