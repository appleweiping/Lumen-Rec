import json
from pathlib import Path

import pytest

from src.confrec import ftgrid_freeze as ff

AM_TEXT = """# amendment
text
<!-- FREEZE_FILES core -->
src/a.py
scripts/run.sh
<!-- /FREEZE_FILES -->

<!-- FREEZE_FILES prune -->
src/prune.py
<!-- /FREEZE_FILES -->
"""


def repo(tmp_path, with_prune=True, with_splits=True):
    root = tmp_path / "repo"
    (root / "idea-stage").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "scripts").mkdir()
    (root / "docs" / "sigir").mkdir(parents=True)
    (root / ff.AMENDMENT).write_text(AM_TEXT, encoding="utf-8")
    (root / "src" / "a.py").write_text("print(1)\n", encoding="utf-8")
    (root / "scripts" / "run.sh").write_text("echo hi\n", encoding="utf-8")
    if with_prune:
        (root / "src" / "prune.py").write_text("x = 1\n", encoding="utf-8")
    if with_splits:
        for d in ff.DOMAINS:
            p = root / "outputs" / "confrec" / "ftgrid" / "panels" / d
            p.mkdir(parents=True)
            (p / "ftgrid_split.json").write_text(json.dumps({"domain": d}), encoding="utf-8")
    (root / "docs" / "sigir" / "PILOT_LOG.md").write_text("# log\n", encoding="utf-8")
    return root


def record(root, stage):
    log = root / "docs" / "sigir" / "PILOT_LOG.md"
    log.write_text(log.read_text(encoding="utf-8") + "\n" + "\n".join(ff.lines_for(stage, root)) + "\n", encoding="utf-8")


def test_blocks_are_parsed_and_unknown_or_duplicate_stages_refused():
    assert ff.parse_blocks(AM_TEXT) == {"core": ["src/a.py", "scripts/run.sh"], "prune": ["src/prune.py"]}
    with pytest.raises(SystemExit, match="unknown"):
        ff.parse_blocks("<!-- FREEZE_FILES weird -->\nx\n<!-- /FREEZE_FILES -->")
    with pytest.raises(SystemExit, match="twice"):
        ff.parse_blocks(AM_TEXT + AM_TEXT)


def test_amendment_stage_needs_only_the_amendment_hash(tmp_path):
    root = repo(tmp_path)
    log = root / "docs" / "sigir" / "PILOT_LOG.md"
    assert ff.check("amendment", root, log) == [ff.AMENDMENT]
    record(root, "amendment")
    assert ff.check("amendment", root, log) == []
    assert ff.check("core", root, log) != []                       # the core stage needs more than the amendment


def test_core_stage_needs_every_listed_file_and_every_split(tmp_path):
    root = repo(tmp_path)
    log = root / "docs" / "sigir" / "PILOT_LOG.md"
    record(root, "core")
    assert ff.check("core", root, log) == []
    assert ff.main(["--check", "--stage", "core", "--root", str(root)]) == 0
    (root / "src" / "a.py").write_text("print(2)\n", encoding="utf-8")        # a bound file changes after the freeze
    assert ff.check("core", root, log) == ["src/a.py"]
    assert ff.main(["--check", "--stage", "core", "--root", str(root)]) == 1


def test_a_missing_listed_file_or_split_is_an_error_not_a_skip(tmp_path):
    root = repo(tmp_path, with_prune=False)
    with pytest.raises(SystemExit, match="do not exist"):
        ff.required("prune", root)
    root2 = repo(tmp_path / "b", with_splits=False)
    with pytest.raises(SystemExit, match="ftgrid_split.json"):
        ff.required("core", root2)
    assert ff.required("amendment", root2)                         # the amendment stage never needs splits


def test_check_is_case_insensitive_and_print_lists_one_hash_per_file(tmp_path):
    root = repo(tmp_path)
    log = root / "docs" / "sigir" / "PILOT_LOG.md"
    log.write_text("\n".join(line.upper() for line in ff.lines_for("prune", root)), encoding="utf-8")
    assert ff.check("prune", root, log) == []
    lines = ff.lines_for("core", root)
    assert len(lines) == 1 + 2 + len(ff.DOMAINS)                   # amendment + two listed files + four splits
    assert all(len(line.split(" = ")[1]) == 40 for line in lines)


def test_the_real_amendment_blocks_parse_and_core_lists_the_gate_ft_trainer():
    text = (Path(__file__).resolve().parents[1] / ff.AMENDMENT).read_text(encoding="utf-8")
    blocks = ff.parse_blocks(text)
    assert set(blocks) == {"core", "prune", "method"}
    assert "src/confrec/train_lora_yesno.py" in blocks["core"] and "src/confrec/lora_trainer.py" in blocks["core"]
    assert ff.AMENDMENT in blocks["core"]
