import gzip
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("slim", ROOT / "scripts" / "sigir" / "slim_amazon2023.py")
slim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slim)

PAYLOAD = b"".join(json.dumps({"user_id": f"u{i}", "parent_asin": f"i{i}", "rating": 5.0,
                               "timestamp": i, "junk": "x" * 50}).encode() + b"\n" for i in range(300))


class FakeResp:
    def __init__(self, data, status, break_after=None):
        self.data, self.status_code, self.break_after = data, status, break_after

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        sent = 0
        for k in range(0, len(self.data), 997):  # odd chunk size splits lines mid-way
            if self.break_after is not None and sent >= self.break_after:
                raise ConnectionError("simulated IncompleteRead")
            piece = self.data[k:k + 997]
            sent += len(piece)
            yield piece


def test_resume_from_last_complete_line(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, stream, timeout, headers):
        calls.append(headers)
        if not headers:
            return FakeResp(PAYLOAD, 200, break_after=len(PAYLOAD) // 2)
        start = int(headers["Range"].split("=")[1].rstrip("-"))
        return FakeResp(PAYLOAD[start:], 206)

    monkeypatch.setattr(slim.requests, "get", fake_get)
    monkeypatch.setattr(slim.time, "sleep", lambda s: None)
    out = tmp_path / "x.jsonl.gz"
    n = slim.stream_slim("http://x", slim.REVIEW_KEYS, out)
    rows = [json.loads(l) for l in gzip.open(out, "rt", encoding="utf-8")]
    assert n == 300 and len(rows) == 300                       # no row lost or duplicated
    assert [r["user_id"] for r in rows] == [f"u{i}" for i in range(300)]
    assert set(rows[0]) == set(slim.REVIEW_KEYS)              # only the kept fields
    assert len(calls) == 2 and "Range" in calls[1]


def test_category_map_is_shared_with_builder():
    from src.confrec.categories import CATEGORY
    assert slim.CATEGORY is CATEGORY and CATEGORY["games"] == "Video_Games"


def test_unwritable_row_is_kept_not_skipped(tmp_path, monkeypatch):
    # a lone-surrogate escape used to raise on the utf-8 write AFTER the offset had advanced past the line,
    # so the Range retry silently dropped that row
    bad = b'{"user_id": "u1", "parent_asin": "i1", "rating": 5.0, "timestamp": 1, "title": "x\\ud800y"}\n'
    payload = bad + PAYLOAD
    monkeypatch.setattr(slim.requests, "get", lambda url, stream, timeout, headers: FakeResp(payload[
        int(headers["Range"].split("=")[1].rstrip("-")) if headers else 0:], 206 if headers else 200))
    monkeypatch.setattr(slim.time, "sleep", lambda s: None)
    out = tmp_path / "m.jsonl.gz"
    n = slim.stream_slim("http://x", ("user_id", "title"), out)
    rows = [json.loads(l) for l in gzip.open(out, "rt", encoding="utf-8")]
    assert n == 301 and len(rows) == 301
    assert rows[0] == {"user_id": "u1", "title": "x\ud800y"}       # same string as the source escape
    assert [r["user_id"] for r in rows[1:]] == [f"u{i}" for i in range(300)]


def test_failed_row_is_retried_not_dropped(tmp_path, monkeypatch):
    calls = []
    real = slim._slim_line

    def flaky(line, keys):
        if b'"u5"' in line and not calls:
            calls.append(1)
            raise OSError("simulated write failure")
        return real(line, keys)

    monkeypatch.setattr(slim, "_slim_line", flaky)
    monkeypatch.setattr(slim.requests, "get", lambda url, stream, timeout, headers: FakeResp(PAYLOAD[
        int(headers["Range"].split("=")[1].rstrip("-")) if headers else 0:], 206 if headers else 200))
    monkeypatch.setattr(slim.time, "sleep", lambda s: None)
    out = tmp_path / "r.jsonl.gz"
    assert slim.stream_slim("http://x", slim.REVIEW_KEYS, out) == 300
    rows = [json.loads(l) for l in gzip.open(out, "rt", encoding="utf-8")]
    assert [r["user_id"] for r in rows] == [f"u{i}" for i in range(300)]
