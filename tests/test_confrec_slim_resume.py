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
