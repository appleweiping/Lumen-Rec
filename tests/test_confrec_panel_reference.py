import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("panel_reference", ROOT / "scripts" / "sigir" / "panel_reference.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)


def _rows(n=5):
    rows = []
    for e in range(n):
        cands = [f"i{e}_{k}" for k in range(4)]
        rows.append({"source_event_id": f"u{e}::{1576559236892 + e}", "user_id": f"u{e}",
                     "history": [f"Hist title {e}", "desc only"], "history_item_ids": [f"h{e}", f"h{e}b"],
                     "candidate_item_ids": cands, "candidate_titles": [f"Title {c}" for c in cands],
                     "candidate_texts": [f"Text {c} é" for c in cands], "positive_item_id": cands[2],
                     "candidate_popularity_groups": ["head", "mid", "tail", "mid"],
                     "candidate_labels": [0, 0, 1, 0], "positive_item_index": 2})
    return rows


def _write(path, rows, ensure_ascii=False):
    path.write_text("".join(json.dumps(r, ensure_ascii=ensure_ascii) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A 5-event ranking file whose bytes are registered as the 'original' panel (anchored)."""
    refs = tmp_path / "refs"
    refs.mkdir()
    monkeypatch.setattr(pr, "REF_DIR", refs)
    rows = _rows()
    (refs / "sports_test.json").write_text(json.dumps(
        {r["source_event_id"]: [r["positive_item_id"], pr.fp(r["candidate_item_ids"])] for r in rows}))
    ranking = tmp_path / "task" / "ranking_test.jsonl"
    ranking.parent.mkdir()
    _write(ranking, rows)
    monkeypatch.setattr(pr, "ORIGINAL", {"sports": {"test": {"ranking": pr.sha256(ranking),
                                                             "candidate_items": "0" * 64}}})
    return refs, ranking, rows


def test_content_round_trip_and_marker(env):
    refs, ranking, rows = env
    pr.verify("sports", str(ranking))                       # anchored by bytes, no content ref yet
    m = json.loads(pr.marker(ranking).read_text())
    assert m["matched"] == 5 and m["content_checked"] is False
    assert m["byte_identical"] is True and m["anchored"] and m["anchored_by"] == "bytes"
    pr.make_content("sports", str(ranking))
    cref = json.loads((refs / "sports_content.json").read_text())
    assert cref["anchored"] is True and cref["fields"] == list(pr.CONTENT_FIELDS)
    assert set(cref["events"]) == {r["source_event_id"] for r in rows}
    assert not pr.check("sports", str(ranking))             # marker predates the content ref
    pr.verify("sports", str(ranking))
    assert json.loads(pr.marker(ranking).read_text())["content_mismatched"] == 0
    assert pr.check("sports", str(ranking))
    with pytest.raises(SystemExit):                          # frozen ref is not overwritten silently
        pr.make_content("sports", str(ranking))


def test_marker_is_bound_to_the_ref_files(env):
    refs, ranking, rows = env
    pr.make_content("sports", str(ranking))
    pr.verify("sports", str(ranking))
    assert pr.check("sports", str(ranking))
    c = json.loads((refs / "sports_content.json").read_text())
    c["events"][rows[0]["source_event_id"]] = "f" * 40     # a different content ref (e.g. another server's freeze)
    (refs / "sports_content.json").write_text(json.dumps(c))
    assert not pr.check("sports", str(ranking))
    pr.make_content("sports", str(ranking), force=True)
    pr.verify("sports", str(ranking))
    assert pr.check("sports", str(ranking))
    t = json.loads((refs / "sports_test.json").read_text())
    (refs / "sports_test.json").write_text(json.dumps(t, indent=1))   # ID ref replaced
    assert not pr.check("sports", str(ranking))


def test_text_drift_fails_verify_and_drops_marker(env):
    refs, ranking, rows = env
    pr.make_content("sports", str(ranking))
    pr.verify("sports", str(ranking))
    assert pr.marker(ranking).exists()
    rows[3]["candidate_texts"][1] = "re-uploaded description"   # same IDs, different prompt text
    _write(ranking, rows)
    assert not pr.check("sports", str(ranking))             # file changed since verification
    res = pr.compare("sports", str(ranking))
    assert res["mismatched"] == 0 and res["content_mismatched"] == 1
    assert res["content_mismatch_examples"] == [rows[3]["source_event_id"]]
    with pytest.raises(SystemExit):                          # a content mismatch is never overridable
        pr.verify("sports", str(ranking), allow_unanchored=True)
    assert not pr.marker(ranking).exists()


def test_history_and_popularity_group_drift_detected(env):
    refs, ranking, rows = env
    pr.make_content("sports", str(ranking))
    rows[0]["history"][0] = "Hist title changed"
    rows[1]["candidate_popularity_groups"][0] = "tail"     # pilot_mirror's head/tail flag on next-item panels
    _write(ranking, rows)
    assert pr.compare("sports", str(ranking))["content_mismatched"] == 2


def test_reserialized_file_is_anchored_by_the_anchored_content_ref(env):
    refs, ranking, rows = env
    pr.make_content("sports", str(ranking))                 # frozen from the byte-identical original
    _write(ranking, rows, ensure_ascii=True)                # same rows, different bytes ("é" escaped)
    assert pr.anchor("sports", str(ranking))["byte_identical"] is False
    pr.verify("sports", str(ranking))
    m = json.loads(pr.marker(ranking).read_text())
    assert m["anchored"] and m["anchored_by"] == "content_ref"


def test_unanchored_panel_needs_explicit_override(env, monkeypatch):
    refs, ranking, rows = env
    monkeypatch.setattr(pr, "ORIGINAL", {"sports": {"test": {"ranking": "a" * 64, "candidate_items": "b" * 64}}})
    with pytest.raises(SystemExit) as e:                     # IDs match, but the text is unproven
        pr.verify("sports", str(ranking))
    assert e.value.code == 2                                 # unanchored only: rebuild_panels.sh does not rebuild
    assert not pr.marker(ranking).exists()
    (ranking.parent / "metadata.json").write_text(json.dumps({**pr.BUILD_ARGS, "seed": 1}))
    with pytest.raises(SystemExit) as e:                     # any other failure as well -> 1 (rebuild)
        pr.verify("sports", str(ranking))
    assert e.value.code == 1
    (ranking.parent / "metadata.json").unlink()
    with pytest.raises(SystemExit):
        pr.make_content("sports", str(ranking))
    assert not (refs / "sports_content.json").exists()
    pr.verify("sports", str(ranking), allow_unanchored=True)
    m = json.loads(pr.marker(ranking).read_text())
    assert m["anchored"] is False and m["allow_unanchored"] is True and m["byte_identical"] is False
    assert not pr.check("sports", str(ranking))             # the override must be repeated
    assert pr.check("sports", str(ranking), allow_unanchored=True)
    pr.make_content("sports", str(ranking), allow_unanchored=True)
    assert json.loads((refs / "sports_content.json").read_text())["anchored"] is False
    with pytest.raises(SystemExit):                          # an unanchored content ref proves nothing
        pr.verify("sports", str(ranking))


def test_candidate_items_csv_is_part_of_the_anchor(env, monkeypatch):
    refs, ranking, rows = env
    ci = ranking.with_name("candidate_items.csv")
    ci.write_text("source_event_id,item_id\n", encoding="utf-8")
    assert pr.anchor("sports", str(ranking))["byte_identical"] is False
    monkeypatch.setitem(pr.ORIGINAL["sports"]["test"], "candidate_items", pr.sha256(ci))
    assert pr.anchor("sports", str(ranking))["byte_identical"] is True


def test_make_content_refuses_on_id_mismatch(env):
    refs, ranking, rows = env
    rows[1]["candidate_item_ids"] = list(reversed(rows[1]["candidate_item_ids"]))
    _write(ranking, rows)
    with pytest.raises(SystemExit):
        pr.make_content("sports", str(ranking), allow_unanchored=True)
    assert not (refs / "sports_content.json").exists()
    with pytest.raises(SystemExit):
        pr.verify("sports", str(ranking), allow_unanchored=True)
    assert not pr.marker(ranking).exists()


def test_missing_event_fails(env):
    refs, ranking, rows = env
    _write(ranking, rows[:4])
    with pytest.raises(SystemExit):
        pr.verify("sports", str(ranking), allow_unanchored=True)


def test_extra_duplicate_and_reordered_rows_fail(env):
    refs, ranking, rows = env
    extra = dict(rows[0], source_event_id="u99::99")
    _write(ranking, [extra] + rows[::-1])                   # head -n 1000 would start with an unverified event
    res = pr.compare("sports", str(ranking))
    assert res["missing"] == 0 and res["mismatched"] == 0 and res["extra"] == 1 and res["out_of_order"] == 4
    with pytest.raises(SystemExit):
        pr.verify("sports", str(ranking), allow_unanchored=True)
    _write(ranking, rows + rows[:1])
    assert pr.compare("sports", str(ranking))["duplicate"] == 1
    with pytest.raises(SystemExit):
        pr.verify("sports", str(ranking), allow_unanchored=True)


def test_builder_args_checked_against_metadata(env):
    refs, ranking, rows = env
    meta = ranking.with_name("metadata.json")
    meta.write_text(json.dumps({**pr.BUILD_ARGS, "seed": 42, "max_history_len": 10}))   # tools args on sports
    assert set(pr.meta_check("sports", str(ranking))["args_mismatch"]) == {"seed", "max_history_len"}
    with pytest.raises(SystemExit):
        pr.verify("sports", str(ranking))
    assert pr.meta_check("tools", str(ranking))["args_mismatch"] == {}
    meta.write_text(json.dumps({**pr.BUILD_ARGS, "exp_name": "x"}))
    pr.verify("sports", str(ranking))
    assert json.loads(pr.marker(ranking).read_text())["args_mismatch"] == {}


def test_build_args_cli(monkeypatch, capsys):
    for d, want in (("tools", "42 10"), ("sports", "20260506 50")):
        monkeypatch.setattr(sys, "argv", ["x", "build-args", "--domain", d])
        pr.main()
        assert capsys.readouterr().out.strip() == want


def test_original_hashes_match_local_provenance():
    base = ROOT / "outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606"
    checked = 0
    for d, splits in pr.ORIGINAL.items():
        p = base / f"ccrp_ablation_{d}" / "ccrp_internal_provenance.json"
        if not p.exists():
            continue
        prov = json.loads(p.read_text(encoding="utf-8"))
        for s, h in splits.items():
            assert prov[f"{s}_ranking_sha256"] == h["ranking"]
            assert prov[f"{s}_candidate_items_sha256"] == h["candidate_items"]
            checked += 1
    if not checked:
        pytest.skip("C-CRP provenance files not present")
