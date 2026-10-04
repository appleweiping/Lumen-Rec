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


def split_json(domain, tokenized=True, gateft_match=True):
    """The parts of an ftgrid_split.json that the freeze reads."""
    return {"domain": domain, "train": {"overlength": {"share_above_1024": 0.0 if tokenized else None}},
            "gateft_T_match": gateft_match if domain == "ml1m" else None}


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
            (p / "ftgrid_split.json").write_text(json.dumps(split_json(d)), encoding="utf-8")
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
    (root / "outputs" / "confrec" / "ftprune").mkdir(parents=True)                  # the manifest exists, src/prune.py does not
    (root / "outputs" / "confrec" / "ftprune" / "prune_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="do not exist"):
        ff.required("prune", root)
    root2 = repo(tmp_path / "b", with_splits=False)
    with pytest.raises(SystemExit, match="ftgrid_split.json"):
        ff.required("core", root2)
    assert ff.required("amendment", root2)                         # the amendment stage never needs splits


def test_dated_addenda_of_the_amendment_are_part_of_every_stage(tmp_path):
    root = repo(tmp_path)
    log = root / "docs" / "sigir" / "PILOT_LOG.md"
    record(root, "amendment")
    assert ff.check("amendment", root, log) == []
    add = root / "idea-stage" / "PREREG_AMENDMENT_3_ADDENDUM_1.md"
    add.write_text("# addendum 1\nclarification\n", encoding="utf-8")
    assert ff.check("amendment", root, log) == ["idea-stage/PREREG_AMENDMENT_3_ADDENDUM_1.md"]   # must be recorded too
    record(root, "amendment")
    assert ff.check("amendment", root, log) == []
    add.write_text("# addendum 1\nchanged after the record\n", encoding="utf-8")
    assert ff.check("amendment", root, log) == ["idea-stage/PREREG_AMENDMENT_3_ADDENDUM_1.md"]
    assert "idea-stage/PREREG_AMENDMENT_3_ADDENDUM_1.md" in [r for r, _ in ff.required("core", root)]


def test_prune_needs_its_manifest_method_takes_extra_files_and_addenda_extend_the_lists(tmp_path):
    root = repo(tmp_path)
    (root / "src" / "m.py").write_text("m = 1\n", encoding="utf-8")
    (root / "outputs" / "confrec" / "ftprune").mkdir(parents=True)
    # an addendum may add files to a stage list (the method stage has none of its own in the repo() amendment text)
    (root / "idea-stage" / "PREREG_AMENDMENT_3_ADDENDUM_2.md").write_text(
        "# add 2\n<!-- FREEZE_FILES method -->\nsrc/m.py\n<!-- /FREEZE_FILES -->\n"
        "<!-- FREEZE_FILES prune -->\nsrc/prune.py\nsrc/a.py\n<!-- /FREEZE_FILES -->\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="prune_manifest.json"):
        ff.required("prune", root)                                   # the manifest does not exist yet
    (root / "outputs" / "confrec" / "ftprune" / "prune_manifest.json").write_text("{}", encoding="utf-8")
    labels = [r for r, _ in ff.required("prune", root)]
    assert labels.count("src/prune.py") == 1 and "src/a.py" in labels and "outputs/confrec/ftprune/prune_manifest.json" in labels
    assert [r for r, _ in ff.required("method", root)][-1] == "src/m.py"                  # no manifest by default
    extra = root / "outputs" / "confrec" / "ftmethod_ml1m_qhat.json"
    extra.write_text("{}", encoding="utf-8")
    labels = [r for r, _ in ff.required("method", root, splits=["outputs/confrec/ftmethod_ml1m_qhat.json"])]
    assert labels[-1] == "outputs/confrec/ftmethod_ml1m_qhat.json" and "src/m.py" in labels


def test_a_split_without_a_tokenizer_audit_or_without_the_gateft_check_cannot_be_frozen(tmp_path):
    root = repo(tmp_path)
    sp = root / "outputs" / "confrec" / "ftgrid" / "panels"
    (sp / "toys" / "ftgrid_split.json").write_text(json.dumps(split_json("toys", tokenized=False)), encoding="utf-8")
    with pytest.raises(SystemExit, match="no tokenizer measurement"):
        ff.required("core", root)
    (sp / "toys" / "ftgrid_split.json").write_text(json.dumps(split_json("toys")), encoding="utf-8")
    (sp / "ml1m" / "ftgrid_split.json").write_text(json.dumps(split_json("ml1m", gateft_match=None)), encoding="utf-8")
    with pytest.raises(SystemExit, match="Gate-FT"):
        ff.required("core", root)
    (sp / "ml1m" / "ftgrid_split.json").write_text(json.dumps(split_json("ml1m")), encoding="utf-8")
    assert ff.required("core", root)                                 # now it can be frozen
    assert ff.required("amendment", root)                            # other stages never read the splits


def test_check_is_case_insensitive_and_print_lists_one_hash_per_file(tmp_path):
    root = repo(tmp_path)
    (root / "outputs" / "confrec" / "ftprune").mkdir(parents=True)
    (root / "outputs" / "confrec" / "ftprune" / "prune_manifest.json").write_text("{}", encoding="utf-8")
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
