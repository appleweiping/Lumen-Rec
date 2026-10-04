"""scripts/sigir/build_confirm_panels.py (amendment 2 G2 / A1) on tiny raw fixtures.

The Pilot-1 panels (and their .meta.json sidecars) are simulated by the default `build_rated_panels --n_users P` CLI
on the same raw data, as Pilot 1 ran it, and the module's registered constants (pilot sha1s, 1500-line prefix, 1683
fresh users, 2000-user Toys threshold) are monkeypatched to the fixture's values; the real-data identity of the
writer is covered in test_confrec_rated_panels.
"""
import gzip
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.confrec import build_rated_panels as brp
from src.confrec.categories import CATEGORY

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_confirm_panels",
                                               ROOT / "scripts" / "sigir" / "build_confirm_panels.py")
bcp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bcp)

KW = dict(n_cands=20, min_hist=3, min_like=3, min_dislike=3)


# ------------------------------------------------------------------ fixtures
def _gz(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")


def ml1m_raw(d, n_users=40, n_movies=120, seed=1):
    r = random.Random(seed)
    d.mkdir(parents=True, exist_ok=True)
    genres = ["Drama|Comedy", "Action", "Sci-Fi|Thriller", "Horror"]
    (d / "movies.dat").write_text("".join(f"{m}::Movie {m} ({1990 + m % 9})::{genres[m % 4]}\n"
                                          for m in range(1, n_movies + 1)), encoding="latin-1")
    lines = []
    for u in range(1, n_users + 1):
        t = 978300000 + u * 10 ** 5
        for m in r.sample(range(1, n_movies + 1), r.randint(8, 60)):
            t += r.choice([0, 1, 60])
            lines.append(f"{u}::{m}::{r.choice([1, 2, 3, 4, 5])}::{t}\n")
    (d / "ratings.dat").write_text("".join(lines), encoding="latin-1")


def amazon_raw(raw, domain, n_users=40, n_items=150, seed=2):
    r = random.Random(seed)
    cats = [["Toys & Games", "Puzzles"], [],
            ["Toys & Games", "Building Toys", "Building Sets For Ages 8 And Up", "Construction Kits", "Extra"]]
    _gz(raw / f"amazon_{domain}" / f"meta_{CATEGORY[domain]}.jsonl.gz",
        [{"parent_asin": f"B{k:04d}", "title": f"Item {k}" if k % 17 else "", "categories": cats[k % 3],
          "description": ["Fun."], "store": f"S{k % 5}"} for k in range(n_items)])
    revs = []
    for u in range(n_users):
        t = 1_600_000_000_000 + u * 10 ** 7
        for k in r.sample(range(n_items), r.randint(6, 50)):
            t += r.choice([0, 1000])
            revs.append({"user_id": f"U{u:03d}", "parent_asin": f"B{k:04d}", "rating": float(r.randint(1, 5)),
                         "timestamp": t})
        revs.append({**revs[-1], "timestamp": t + 1, "rating": 6.0 - revs[-1]["rating"]})   # re-review
    _gz(raw / f"amazon_{domain}" / f"{CATEGORY[domain]}.jsonl.gz", revs)


def pilot_panel(monkeypatch, raw, pilot_dir, name, n_users, seed=0):
    """What Pilot 1 ran: the default build_rated_panels CLI (hist_len 10, Pilot-1 format), with its sidecars."""
    src = ["--source", "ml1m", "--raw", str(raw / "ml-1m")] if name == "ml1m" else \
        ["--source", "amazon", "--domain", name, "--raw", str(raw)]
    out = pilot_dir / f"{name}_rated.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", *src, "--out", str(out), "--n_users", str(n_users), "--seed", str(seed)])
    brp.main()
    return out


def tmp_files(d: Path) -> list:
    return sorted(p.name for p in d.iterdir() if p.name.endswith(".tmp") or p.name.startswith(".")) \
        if d.exists() else []


def sha1(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def rows_of(path) -> list:
    return [json.loads(l) for l in Path(path).read_bytes().decode("utf-8").split("\n")[:-1]]


def full_build(raw, name, hist_len=10):
    items, events = brp.load_ml1m(raw / "ml-1m") if name == "ml1m" else brp.load_amazon(raw, name)
    return brp.build(items, events, n_users=10 ** 9, hist_len=hist_len, seed=0, source=name, **KW)


def setup(tmp_path, monkeypatch, n_pilot=12, pilot_seed=0, toys_confirm_min=10, games=True):
    raw, pilot = tmp_path / "raw", tmp_path / "pilot"
    ml1m_raw(raw / "ml-1m")
    amazon_raw(raw, "toys")
    if games:
        amazon_raw(raw, "games", seed=5)
    for name in ("ml1m", "toys"):
        pilot_panel(monkeypatch, raw, pilot, name, n_pilot, seed=pilot_seed)
    monkeypatch.setattr(bcp, "PILOT_SHA1", {n: sha1(pilot / f"{n}_rated.jsonl") for n in ("ml1m", "toys")})
    monkeypatch.setattr(bcp, "PREFIX_LINES", n_pilot)
    monkeypatch.setattr(bcp, "TOYS_CONFIRM_MIN", toys_confirm_min)
    monkeypatch.setattr(bcp, "FRESH_EXPECTED", {"ml1m": full_build(raw, "ml1m")[1] - n_pilot})
    out = tmp_path / "out"
    argv = ["--raw", raw, "--ml1m_raw", raw / "ml-1m", "--pilot_dir", pilot, "--out_dir", out,
            "--count_domains", "games,sports"]
    return SimpleNamespace(raw=raw, pilot=pilot, out=out, argv=[str(a) for a in argv])


def run(env, *extra):
    return bcp.run(bcp.parse_args(env.argv + list(extra)))


def h10_bytes(rows) -> bytes:
    return "".join(brp.row_line(brp.pilot_row(brp.truncate_history(r, 10))) for r in rows).encode("utf-8")


# ------------------------------------------------------------------ tests
def test_prefix_path_dev_is_the_pilot_panel_and_confirm_the_fresh_users(tmp_path, monkeypatch):
    env = setup(tmp_path, monkeypatch)
    man = run(env)
    assert json.loads((env.out / "manifest.json").read_text(encoding="utf-8")) == json.loads(json.dumps(man))
    for name, kind in (("ml1m", "movie"), ("toys", "product")):
        s = man["sources"][name]
        assert s["prefix_match"] and s["split_method"] == "prefix" and s["mismatch"] is None
        assert s["pilot_panel_sha1_ok"] and s["h20_rows_truncate_to_h10"]
        assert s["dev_reproduces_pilot_panel"] and s["dev_rows_equal_to_pilot"] == 12
        full10, n_elig = full_build(env.raw, name)
        full20, _ = full_build(env.raw, name, hist_len=20)
        assert s["eligible_users"] == n_elig == man["eligible_counts"][name] > 20
        dev, conf = rows_of(env.out / f"{name}_dev_h20.jsonl"), rows_of(env.out / f"{name}_confirm_h20.jsonl")
        pilot_file = env.pilot / f"{name}_rated.jsonl"
        assert [r["user_id"] for r in dev] == [r["user_id"] for r in rows_of(pilot_file)]
        assert not {r["user_id"] for r in dev} & {r["user_id"] for r in conf}
        assert dev + conf == full20                                # panel order: DEV = first 12, CONFIRM = rest
        assert h10_bytes(dev) == pilot_file.read_bytes()           # DEV cut back to hist_len 10 = the pilot panel
        assert h10_bytes(dev + conf) == "".join(brp.row_line(brp.pilot_row(r)) for r in full10).encode("utf-8")
        assert all(r["domain_kind"] == kind and len(r["history_meta"]) == len(r["history_item_ids"])
                   for r in dev + conf)
        assert all(len(r["history"]) <= 20 for r in dev + conf)
        for split, rs in (("dev", dev), ("confirm", conf)):
            rec = s[split]
            b = (env.out / f"{name}_{split}_users.txt").read_bytes()
            assert b == "\n".join(sorted(r["user_id"] for r in rs)).encode("utf-8")
            assert hashlib.sha1(b).hexdigest() == rec["user_ids_sha1"] == man["freeze"][f"{name}_{split}_users_sha1"]
            assert sha1(env.out / f"{name}_{split}_h20.jsonl") == rec["sha1"]
            assert rec["built"] and rec["n_users"] == rec["n_lines"] == len(rs)
    assert any(len(r["history"]) > 10 for r in rows_of(env.out / "ml1m_confirm_h20.jsonl"))
    for name in ("ml1m", "toys"):   # A1 cross-check with the Pilot-1 meta.json (eligible_users written by Pilot 1)
        s = man["sources"][name]
        assert s["eligible_users_expected"] == s["eligible_users"] and s["eligible_matches_pilot_meta"] is True
        assert s["pilot_meta"]["users"] == 12 and "error" not in s["pilot_meta"]
        assert s["fresh"] == s["confirm"]["n_users"] == s["eligible_users"] - 12 and s["fresh_matches_expected"]
    assert man["sources"]["ml1m"]["fresh_expected_source"].startswith("registered")
    assert man["sources"]["toys"]["fresh_expected"] == man["sources"]["toys"]["eligible_users"] - 12
    assert man["sources"]["toys"]["fresh_expected_source"].startswith("Pilot-1 meta.json")
    assert man["sources"]["ml1m"]["dev"]["history_meta_nonempty_share"] == 1.0          # every movie has genres
    assert 0 < man["sources"]["toys"]["dev"]["history_meta_nonempty_share"] < 1           # empty categories -> ""
    # count-only domains: games equals the full builder's eligible count; sports has no raw files here
    assert man["eligible_counts"]["games"] == full_build(env.raw, "games")[1] > 0
    assert man["count_details"]["sports"]["status"] == "missing_raw" and man["eligible_counts"]["sports"] is None
    assert not [p for p in env.out.iterdir() if p.name.endswith(".tmp") or p.name.startswith(".")]


def test_toys_confirm_is_not_written_below_2000_eligible_users(tmp_path, monkeypatch):
    env = setup(tmp_path, monkeypatch, toys_confirm_min=2000)
    (env.out).mkdir()
    (env.out / "toys_confirm_h20.jsonl").write_text("stale\n", encoding="utf-8")   # from some earlier run
    man = run(env)
    t = man["sources"]["toys"]
    assert t["confirm"]["built"] is False and t["confirm"]["n_fresh_eligible"] == t["eligible_users"] - 12
    assert "< 2000" in t["confirm"]["reason"] and man["freeze"]["toys_confirm_users_sha1"] is None
    assert not (env.out / "toys_confirm_h20.jsonl").exists() and not (env.out / "toys_confirm_users.txt").exists()
    assert (env.out / "toys_dev_h20.jsonl").exists() and t["dev"]["n_users"] == 12   # DEV is always built
    assert man["sources"]["ml1m"]["confirm"]["built"] and (env.out / "ml1m_confirm_h20.jsonl").exists()


def test_prefix_mismatch_falls_back_to_user_id_exclusion(tmp_path, monkeypatch):
    # a "pilot" panel drawn with another seed: its first lines are not the seed-0 prefix
    env = setup(tmp_path, monkeypatch, pilot_seed=7)
    man = run(env)
    for name in ("ml1m", "toys"):
        s = man["sources"][name]
        assert not s["prefix_match"] and s["split_method"] == "user_id_exclusion"
        assert s["mismatch"]["expected"] == bcp.PILOT_SHA1[name] != s["mismatch"]["got"] == s["prefix_sha1"]
        assert s["mismatch"]["first_line_differing_from_pilot_file"] is not None
        assert not s["dev_reproduces_pilot_panel"] and s["dev_rows_equal_to_pilot"] < 12
        pilot_ids = {r["user_id"] for r in rows_of(env.pilot / f"{name}_rated.jsonl")}
        dev, conf = rows_of(env.out / f"{name}_dev_h20.jsonl"), rows_of(env.out / f"{name}_confirm_h20.jsonl")
        full20, n_elig = full_build(env.raw, name, hist_len=20)
        assert {r["user_id"] for r in dev} == pilot_ids and not pilot_ids & {r["user_id"] for r in conf}
        assert dev == [r for r in full20 if r["user_id"] in pilot_ids]        # panel order, not pilot order
        assert conf == [r for r in full20 if r["user_id"] not in pilot_ids]
        assert len(dev) + len(conf) == n_elig


def test_registered_sha1_mismatch_with_the_true_pilot_file_still_splits_by_user_id(tmp_path, monkeypatch):
    env = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(bcp, "PILOT_SHA1", {**bcp.PILOT_SHA1, "ml1m": "0" * 40})
    s = run(env)["sources"]["ml1m"]
    assert not s["pilot_panel_sha1_ok"] and not s["prefix_match"] and s["split_method"] == "user_id_exclusion"
    assert s["dev_rows_equal_to_pilot"] == 12 and s["mismatch"]["first_line_differing_from_pilot_file"] is None
    assert h10_bytes(rows_of(env.out / "ml1m_dev_h20.jsonl")) == (env.pilot / "ml1m_rated.jsonl").read_bytes()


def test_dev_users_must_equal_the_pilot_users(tmp_path, monkeypatch):
    env = setup(tmp_path, monkeypatch)
    p = env.pilot / "ml1m_rated.jsonl"
    ghost = {**rows_of(p)[0], "user_id": "ghost"}                # a pilot user the rebuild cannot contain
    with open(p, "ab") as f:
        f.write(brp.row_line(ghost).encode("utf-8"))
    with pytest.raises(SystemExit, match="DEV users != Pilot-1 users"):
        run(env)
    assert not (env.out / "manifest.json").exists()


def test_hist_len_20_rows_must_match_the_hist_len_10_build(tmp_path, monkeypatch):
    env = setup(tmp_path, monkeypatch)
    real = brp.build

    def drifting_build(*a, **kw):   # a builder whose hist_len-20 rows differ beyond the history (must be caught)
        rows, n = real(*a, **kw)
        if kw["hist_len"] == 20:
            rows[3]["candidate_labels"] = rows[3]["candidate_labels"][::-1]
        return rows, n
    monkeypatch.setattr(bcp.brp, "build", drifting_build)
    with pytest.raises(SystemExit, match="differ from the hist_len 10 build"):
        run(env)
    # the full hist_len 10 temp panel is kept as evidence; nothing else is left behind, no manifest
    assert tmp_files(env.out) == [".ml1m_full_h10.pilot_format.jsonl.tmp"]
    assert not (env.out / "manifest.json").exists()


def test_rerun_is_deterministic_and_frozen_outputs_are_not_replaced_without_force(tmp_path, monkeypatch):
    env = setup(tmp_path, monkeypatch)
    first = run(env)
    before = {p.name: p.read_bytes() for p in env.out.iterdir() if p.name != "manifest.json"}
    again = run(env)                                         # same inputs: identical outputs, no conflict
    assert again["replaced"] == [] and again["freeze"] == first["freeze"]
    assert {p.name: p.read_bytes() for p in env.out.iterdir() if p.name != "manifest.json"} == before
    # a different pilot panel (other users) would change the frozen DEV/CONFIRM lists
    pilot_panel(monkeypatch, env.raw, env.pilot, "ml1m", 12, seed=3)
    monkeypatch.setattr(bcp, "PILOT_SHA1", {**bcp.PILOT_SHA1, "ml1m": sha1(env.pilot / "ml1m_rated.jsonl")})
    with pytest.raises(SystemExit, match="--force"):
        run(env)
    assert {p.name: p.read_bytes() for p in env.out.iterdir() if p.name != "manifest.json"} == before
    assert json.loads((env.out / "manifest.json").read_text(encoding="utf-8"))["freeze"] == first["freeze"]
    forced = run(env, "--force")
    assert forced["freeze"]["ml1m_dev_users_sha1"] != first["freeze"]["ml1m_dev_users_sha1"]
    assert any(c.startswith("ml1m.dev.user_ids_sha1") for c in forced["replaced"])


def test_partial_rerun_keeps_the_records_it_did_not_rebuild(tmp_path, monkeypatch):
    env = setup(tmp_path, monkeypatch)
    first = run(env)
    part = run(env, "--sources", "ml1m", "--count_domains", "")
    assert part["carried_over_from_previous_manifest"] == ["toys", "games", "sports"]
    assert part["freeze"] == first["freeze"] and part["eligible_counts"] == first["eligible_counts"]
    assert part["sources"]["toys"] == first["sources"]["toys"]
    assert json.loads((env.out / "manifest.json").read_text(encoding="utf-8"))["freeze"] == first["freeze"]
    assert (env.out / "toys_confirm_h20.jsonl").exists()


def test_count_domain_light_loader_equals_the_builder(tmp_path):
    raw = tmp_path / "raw"
    amazon_raw(raw, "sports", seed=9)
    rec = bcp.count_domain(raw, "sports")
    assert rec["status"] == "ok" and rec["eligible_users"] == full_build(raw, "sports")[1] > 0
    assert rec["n_reviews"] == sum(len(v) for v in brp.load_amazon(raw, "sports")[1].values())
    assert bcp.count_domain(raw, "games")["status"] == "missing_raw"


def test_bad_inputs_fail_before_any_raw_file_is_loaded(tmp_path, monkeypatch):
    env = setup(tmp_path, monkeypatch)

    def no_load(*a, **kw):
        raise AssertionError("raw data loaded before the input checks")
    monkeypatch.setattr(bcp.brp, "load_ml1m", no_load)
    monkeypatch.setattr(bcp.brp, "load_amazon", no_load)
    with pytest.raises(SystemExit, match=r"unknown count domain\(s\) \['bogus'\]"):
        run(env, "--count_domains", "games,bogus")
    with pytest.raises(SystemExit, match=r"unknown source\(s\)"):
        run(env, "--sources", "ml1m,games")
    (env.pilot / "toys_rated.jsonl").rename(env.pilot / "toys_rated.jsonl.bak")       # Toys pilot panel missing
    with pytest.raises(SystemExit, match="missing input file.*toys_rated.jsonl"):
        run(env)
    (env.pilot / "toys_rated.jsonl.bak").rename(env.pilot / "toys_rated.jsonl")
    (env.raw / "ml-1m" / "movies.dat").rename(env.raw / "ml-1m" / "movies.bak")       # ML-1M raw file missing
    with pytest.raises(SystemExit, match="missing input file.*movies.dat"):
        run(env)
    assert not env.out.exists()                                                     # nothing created at all


def test_a_failure_after_staging_removes_the_temp_files_and_keeps_the_outputs(tmp_path, monkeypatch):
    env = setup(tmp_path, monkeypatch)
    first = run(env)
    before = {p.name: p.read_bytes() for p in env.out.iterdir()}

    def broken_count(raw, domain):   # fails after both sources were built and staged
        raise RuntimeError("count failed")
    monkeypatch.setattr(bcp, "count_domain", broken_count)
    with pytest.raises(RuntimeError, match="count failed"):
        run(env)
    assert tmp_files(env.out) == []
    assert {p.name: p.read_bytes() for p in env.out.iterdir()} == before
    # a refused overwrite (frozen outputs differ) also leaves no staged temp file behind
    monkeypatch.setattr(bcp, "count_domain", lambda raw, d: {"status": "ok", "eligible_users": 0})
    pilot_panel(monkeypatch, env.raw, env.pilot, "ml1m", 12, seed=3)
    monkeypatch.setattr(bcp, "PILOT_SHA1", {**bcp.PILOT_SHA1, "ml1m": sha1(env.pilot / "ml1m_rated.jsonl")})
    with pytest.raises(SystemExit, match="--force"):
        run(env)
    assert tmp_files(env.out) == [] and {p.name: p.read_bytes() for p in env.out.iterdir()} == before
    assert first["freeze"] == json.loads((env.out / "manifest.json").read_text(encoding="utf-8"))["freeze"]


def test_eligible_count_drift_against_the_pilot_meta_is_recorded_not_fatal(tmp_path, monkeypatch):
    env = setup(tmp_path, monkeypatch)
    tm = env.pilot / "toys_rated.meta.json"
    m = json.loads(tm.read_text(encoding="utf-8"))
    tm.write_text(json.dumps({**m, "eligible_users": m["eligible_users"] + 5}), encoding="utf-8")
    (env.pilot / "ml1m_rated.meta.json").unlink()
    man = run(env)
    t, ml = man["sources"]["toys"], man["sources"]["ml1m"]
    assert t["eligible_users_expected"] == t["eligible_users"] + 5 and t["eligible_matches_pilot_meta"] is False
    assert t["fresh_expected"] == t["eligible_users"] + 5 - 12 and t["fresh_matches_expected"] is False
    assert t["prefix_match"] and t["confirm"]["built"]          # recorded, not fatal: the split is unchanged
    assert ml["pilot_meta"]["error"] == "missing" and ml["eligible_users_expected"] is None
    assert ml["eligible_matches_pilot_meta"] is None
    assert ml["fresh_expected"] == bcp.FRESH_EXPECTED["ml1m"] and ml["fresh_matches_expected"]   # registered value


def test_user_list_and_prefix_helpers(tmp_path):
    assert bcp.user_list_bytes(["10", "2", "1"]) == b"1\n10\n2"          # Python string order, no trailing "\n"
    p = tmp_path / "x.jsonl"
    p.write_bytes(b"a\nb\nc\n")
    assert bcp.prefix_sha1(p, 2) == (hashlib.sha1(b"a\nb\n").hexdigest(), 2)
    assert bcp.prefix_sha1(p, 9) == (hashlib.sha1(b"a\nb\nc\n").hexdigest(), 3)
