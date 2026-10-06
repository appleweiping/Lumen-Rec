"""scripts/sigir/build_paper.py: result tokens in free TeX prose are replaced by values read from result files."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_paper_under_test", ROOT / "scripts" / "sigir" / "build_paper.py")
bp = importlib.util.module_from_spec(spec)
sys.modules["build_paper_under_test"] = bp
spec.loader.exec_module(bp)


@pytest.fixture()
def world(tmp_path):
    res = tmp_path / "results"
    (res / "grid").mkdir(parents=True)
    (res / "grid" / "a.json").write_text(json.dumps({
        "E_A": {"ZS": {"UAUC": {"est": 0.5959, "lo": 0.5734, "hi": 0.618, "n_users": 366},
                       "neg": {"est": -0.0004, "lo": -0.02, "hi": 0.02},
                       "p": 0.0004, "p2": 0.04, "share": 0.9021, "n": 17383, "name": "ml_1m", "lst": [1, {"x": 2.5}]},
                "blk": {"available": False, "reason": "no record"}},
        "null": None,
    }), encoding="utf-8")
    src = tmp_path / "src"
    (src / "sections").mkdir(parents=True)
    (src / "figures").mkdir()
    (src / "figures" / "f.pdf").write_bytes(b"%PDF-1.4 fake")
    return SimpleWorld(res, src, tmp_path / "out")


class SimpleWorld:
    def __init__(self, res, src, out):
        self.res, self.src, self.out = res, src, out

    def run(self, text, strict=False):
        (self.src / "sections" / "t.tex").write_text(text, encoding="utf-8")
        summary = bp.build(self.src, self.res, self.out, strict)
        return (self.out / "sections" / "t.tex").read_text(encoding="utf-8"), summary


def test_formats_of_a_record_and_scalars(world):
    j = "grid/a.json"
    text, s = world.run(
        rf"\jv{{{j}|E_A/ZS/UAUC|est3}} \jv{{{j}|E_A/ZS/UAUC|ci3}} \jv{{{j}|E_A/ZS/UAUC|pm3}} \jv{{{j}|E_A/ZS/UAUC|estci2}} "
        rf"\jv{{{j}|E_A/ZS/UAUC|hw3}} \jv{{{j}|E_A/ZS/neg|est3}} \jv{{{j}|E_A/ZS/neg|s3}} \jv{{{j}|E_A/ZS/p|p3}} \jv{{{j}|E_A/ZS/p2|p3}} "
        rf"\jv{{{j}|E_A/ZS/share|pct1}} \jv{{{j}|E_A/ZS/n|int}} \jv{{{j}|E_A/ZS/name|str}} \jv{{{j}|E_A/ZS/lst/1/x|f1}}")
    assert text.split() == ["0.596", "[0.573,", "0.618]", "0.596$\\pm$.022", "0.60", "[0.57,", "0.62]", "0.022", "0.000", "0.000",
                            "$<$0.001", "0.040", "90.2\\%", "17,383", "ml\\_1m", "2.5"]
    assert s["unresolved"] == 0


def test_negative_values_use_a_math_minus_and_a_zero_has_no_sign(world):
    text, _ = world.run(r"\jv{grid/a.json|E_A/ZS/neg|ci3} \jd{a-0.6|s3|a=grid/a.json|E_A/ZS/UAUC} \jd{a-0.5959|s3|a=grid/a.json|E_A/ZS/UAUC}")
    assert text.split() == ["[$-$0.020,", "0.020]", "$-$0.004", "0.000"]


def test_derived_values_use_only_arithmetic(world):
    ok, _ = world.run(r"\jd{(a-b)*100|f1|a=grid/a.json|E_A/ZS/UAUC|b=grid/a.json|E_A/ZS/neg}")
    assert ok.strip() == "59.6"
    for bad in ("__import__('os')", "a.real", "a if a else 0", "open('x')", "[a]"):
        t, s = world.run(rf"\jd{{{bad}|f1|a=grid/a.json|E_A/ZS/UAUC}}")
        assert "TBD" in t and s["unresolved"] == 1, bad


def test_missing_things_become_red_boxes_and_are_listed(world):
    t, s = world.run(r"\jv{grid/none.json|a|f3} \jv{grid/a.json|E_A/zz|f3} \jv{grid/a.json|E_A/blk/x|f3} \jv{grid/a.json|null|f3} "
                     r"\jv{grid/a.json|E_A/ZS/UAUC|bogus} \jp{no_such_phrase}")
    assert t.count("[TBD:") == 6 and s == {"tokens": 6, "resolved": 0, "unresolved": 6}
    listed = json.loads((world.out / "UNRESOLVED.json").read_text(encoding="utf-8"))["unresolved"]
    assert len(listed) == 6 and "not available (no record)" in listed[2]["reason"]
    with pytest.raises(SystemExit):
        world.run(r"\jv{grid/none.json|a|f3}", strict=True)


def test_commented_tokens_are_left_alone_and_other_files_are_copied(world):
    t, s = world.run("% \\jv{grid/none.json|a|f3}\nvalue \\jv{grid/a.json|E_A/ZS/UAUC|f2}\n")
    assert t == "% \\jv{grid/none.json|a|f3}\nvalue 0.60\n" and s["unresolved"] == 0
    assert (world.out / "figures" / "f.pdf").read_bytes() == b"%PDF-1.4 fake"


def test_assets_json_records_the_provenance_of_every_token(world):
    world.run(r"\jv{grid/a.json|E_A/ZS/UAUC|est3} \jd{a*2|f2|a=grid/a.json|E_A/ZS/UAUC}")
    a = json.loads((world.out / "ASSETS.json").read_text(encoding="utf-8"))["tokens"]
    assert [x["token"] for x in a] == ["jv", "jd"]
    assert a[0]["file"] == "grid/a.json" and len(a[0]["sha1"]) == 40 and a[0]["path"] == "E_A/ZS/UAUC" and a[0]["printed"] == "0.596"
    assert a[1]["inputs"]["a"]["path"] == "E_A/ZS/UAUC" and a[1]["printed"] == "1.19"


def test_the_build_is_deterministic(world):
    world.run(r"\jv{grid/a.json|E_A/ZS/UAUC|est3}")
    first = (world.out / "ASSETS.json").read_bytes()
    world.run(r"\jv{grid/a.json|E_A/ZS/UAUC|est3}")
    assert (world.out / "ASSETS.json").read_bytes() == first
