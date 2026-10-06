"""scripts/sigir/make_figures.py: the figures of the SIGIR 2027 paper and their manifest. CPU only, no GPU, no network, Agg backend.

Fixtures: a synthetic results tree built in tmp_path with the REAL schemas of the files the figures read (grid reports, addendum-6
extra files, next-item audits, the Gate-FT decision). Every number in it comes from a seeded RNG: no real result is used, and no
result number is typed into an assertion. One test (skipped when the tree is absent) runs the generator on a snapshot of the real
docs/sigir/results tree and checks the same round trip, whatever panels that tree holds at the time.

What is checked:
  * every value of figures_manifest.json equals the value in its source file, and every sha1 (sources, PDFs) equals the file's;
  * what is drawn (marker and interval coordinates, hollow or filled) equals the manifest;
  * a missing panel is an explicit 'not run yet' placeholder with a manifest entry (status not_run) naming the missing file;
    a malformed file raises an error naming the file and the key;
  * exploratory / descriptive flags and withheld (uninterpretable) values are carried to the manifest and the drawing;
  * no read outside the results root (a decoy outside the root is never opened);
  * the generator is deterministic (same inputs, byte-identical PDFs and manifest, also across processes);
  * style: 7.0 in wide, fonts embedded as TrueType (no Type 3), text inside the canvas; no result-like literal in the script;
    captions of the proposed LaTeX snippets carry no result number; decomp.tex compiles (when LaTeX is installed)."""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np                                    # noqa: E402
import pytest                                         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sigir" / "make_figures.py"
FILL = ROOT / "scripts" / "sigir" / "fill_paper.py"
REAL = ROOT / "docs" / "sigir" / "results"
FIGDIR = ROOT / "Paper" / "sigir2027" / "figures"
ALL3 = ("tracks", "shares", "serving")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                           # dataclasses with postponed annotations need the module registered
    spec.loader.exec_module(mod)
    return mod


mf = _load("_make_figures_under_test", SCRIPT)


# ------------------------------------------------------------------------------------------------ synthetic world
def rec(centre, half=0.02, *, n_users=300, **extra) -> dict:
    """A result record {est, lo, hi, n_users, n_pairs, n_boot, descriptive_min_n} as the report writers produce it."""
    est = float(centre)
    d = {"est": est, "lo": est - half, "hi": est + half, "n_users": int(n_users), "n_pairs": int(n_users) * 10, "n_boot": 2000,
         "descriptive_min_n": bool(n_users < 150)}
    d.update(extra)
    return d


def grid_report(rng, domain, backbone="Qwen3-8B", *, n_users=300, zs_reading="interpretable", ft_reading="interpretable") -> dict:
    r = lambda lo, hi, half=0.02: rec(rng.uniform(lo, hi), half, n_users=n_users)              # noqa: E731
    refs = {"q_hat": r(0.55, 0.7), "mf": r(0.5, 0.7), "popularity": r(0.5, 0.6), "mf_personal_residual": r(0.5, 0.6)}
    rows = {"n_users": n_users, "n_pairs": n_users * 10}
    seeds = ("s0", "s1", "s2")
    share_keys = lambda reading: {"item_prior_share": r(0.2, 0.9), "non_prior_share": r(0.1, 0.8),          # noqa: E731
                                  "r8c": r(0.6, 0.97), "shares_reading": reading}

    def gain(models):
        per = {m: {"G": rec(rng.uniform(-0.01, 0.03), 0.01, n_users=n_users, p=0.3, ci_excludes_0=False)} for m in models}
        return {"rows": {"n_pairs": n_users * 10, "n_users": n_users}, "per_model": per,
                "G_mean_over_seeds": rec(rng.uniform(-0.01, 0.03), 0.01, n_users=n_users, p=0.2, ci_excludes_0=False),
                "G_CF": rec(rng.uniform(0.0, 0.05), 0.01, n_users=n_users, p=0.1, ci_excludes_0=True)}

    shares_rows = {"n_pairs": n_users * 10, "n_users": n_users, "n_users_with_2_or_more_pairs": n_users}
    return {
        "spec": "synthetic", "meta": {"domain": domain, "backbone": backbone, "n_boot": 2000},
        "E_A": {
            "ZS": {"models": ["zeroshot"], "missing_or_excluded": [], "complete": True,
                   "UAUC_TEST": {"rows": rows, "per_model": {"zeroshot": r(0.45, 0.62)}, "mean_over_seeds": r(0.45, 0.62),
                                 "references": refs}},
            "FT": {"models": list(seeds), "missing_or_excluded": [], "complete": True,
                   "UAUC_TEST": {"rows": rows, "per_model": {m: r(0.55, 0.78) for m in seeds}, "mean_over_seeds": r(0.55, 0.78),
                                 "references": json.loads(json.dumps(refs))}}},
        "E_D": {
            "ZS": {"models": ["zeroshot"], "complete": True,
                   "shares": {"rows": shares_rows, "per_model": {"zeroshot": share_keys(zs_reading)},
                              "mean_over_seeds": share_keys(zs_reading)},
                   "information_gain": gain(["zeroshot"])},
            "FT": {"models": list(seeds), "complete": True,
                   "shares": {"rows": shares_rows, "per_model": {m: share_keys(ft_reading) for m in seeds},
                              "mean_over_seeds": share_keys(ft_reading)},
                   "information_gain": gain(list(seeds))}},
    }


def extra_report(rng, domain, backbone="Qwen3-8B", *, status="registered_outcome_free", n_users=300, g_descriptive=True) -> dict:
    r = lambda lo, hi, half=0.02: rec(rng.uniform(lo, hi), half, n_users=n_users)              # noqa: E731
    rows = {"n_users": n_users, "n_pairs": n_users * 10}

    def j(models):
        return {"rows_E_A": dict(rows), "complete": True,
                "dUAUC_L_minus_q_hat_T": {"rows": dict(rows), "per_seed": {
                    m: {**r(-0.1, 0.1), "UAUC_a": 0.6, "UAUC_b": float(rng.uniform(0.5, 0.7))} for m in models}}}

    def w(models):
        return {"models": list(models), "complete": True, "rows": dict(rows),
                "G_wu": {"per_model": {m: rec(rng.uniform(-0.01, 0.03), 0.01, n_users=n_users) for m in models},
                         "mean_over_seeds": rec(rng.uniform(-0.01, 0.03), 0.01, n_users=n_users)},
                "G_CF_wu": rec(rng.uniform(0.0, 0.05), 0.01, n_users=n_users)}

    def g(models):
        return {"models": list(models), "complete": True,
                "e_share": {"per_model": {m: {"e_share": r(0.05, 0.8), "var_e_over_var_L_uncorrected": 0.3} for m in models},
                            "mean_over_seeds": r(0.05, 0.8)}}

    ft = ("s0", "s1", "s2")
    return {"status": {"items_2_to_7": status}, "meta": {"domain": domain, "backbone": backbone, "n_boot": 2000},
            "E_J": {"q_hat_T": {"available": True}, "ZS": j(["zeroshot"]), "FT": j(ft)},
            "E_G": {"descriptive": g_descriptive, "ZS": g(["zeroshot"]), "FT": g(ft)},
            "E_W": {"ZS": w(["zeroshot"]), "FT": w(ft)}}


def aud_report(rng, domain, segments=(("all", "all"),), n_events=500) -> dict:
    cov = [round(0.1 * k, 1) for k in range(1, 11)]

    def curve(base, slope):
        out = []
        for c in cov:
            est = base + slope * (1.0 - c)
            out.append({"coverage": c, "est": est, "lo": est - 0.02, "hi": est + 0.02, "n": n_events, "n_boot": 2000})
        return out

    base = float(rng.uniform(0.15, 0.3))
    segs = {}
    for seg, role in segments:
        segs[seg] = {"role": role, "event_range": [1, n_events], "n_events": n_events,
                     "questions": {"next": {"D_selective_serving": {
                         "n_events": n_events, "coverage_points": cov,
                         "signals": {"p_max": {"curve": {"ndcg10": curve(base, 0.4)}},
                                     "margin": {"curve": {"ndcg10": curve(base, 0.3)}},       # in the file, never drawn
                                     "random": {"curve": {"ndcg10": curve(base, 0.0)}}}}}}}
    return {"schema": "nextitem_audit_v1", "domain": domain, "n_boot": 2000, "segments": segs}


def write_json(root: Path, rel: str, obj) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    return p


def build_tree(root: Path, *, seed: int = 0, skip=(), gate="GATE_FT_PASS", over=None) -> Path:
    """A results tree with the real layout: grid/{qwen,llama}, extra/..., aud/..., gft/gate_ft.json. `skip`: relative paths to leave
    out; `over[rel]`: keyword overrides of that file's builder."""
    rng = np.random.default_rng(seed)
    over = over or {}
    root.mkdir(parents=True, exist_ok=True)

    def put(rel, builder, *a, **k):
        if rel not in skip:
            write_json(root, rel, builder(rng, *a, **{**k, **over.get(rel, {})}))

    for d in mf.RATED:
        put(f"grid/qwen/{d}.json", grid_report, d)
        put(f"extra/{d}.json", extra_report, d, status="exploratory" if d == "ml1m" else "registered_outcome_free")
    for d in ("ml1m", "toys"):
        put(f"grid/llama/{d}.json", grid_report, d, "Llama-3.1-8B-Instruct")
        put(f"extra/llama/{d}.json", extra_report, d, "Llama-3.1-8B-Instruct",
            status="exploratory" if d == "ml1m" else "registered_outcome_free")
    for d in ("sports", "toys", "home", "tools"):
        segs = (("events_1_1000", "quarantine"), ("events_1001_10000", "main")) if d == "sports" else (("all", "all"),)
        put(f"aud/{d}.json", aud_report, d, segs)
    if "gft/gate_ft.json" not in skip:
        write_json(root, "gft/gate_ft.json", {"decision": gate})
    return root


def load_rel(root: Path, rel: str):
    return json.loads((root / rel).read_text(encoding="utf-8"))


def resolve(root: Path, source: str):
    """The node a manifest source '<file>:<dotted key path>' points to, read straight from the file on disk."""
    rel, key = source.split(":", 1)
    node = load_rel(root, rel)
    for k in key.split("."):
        node = node[int(k)] if isinstance(node, list) else node[k]
    return node


def sha1_of(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def all_values(entry: dict):
    for p in entry.get("panels", []):
        for v in p["values"]:
            yield p, v


def collect_all(root: Path):
    """The data layer of the three figures (no drawing): where a malformed file must raise."""
    res = mf.Results(root)
    return mf.collect_tracks(res), mf.collect_shares(res), mf.collect_serving(res)


def entry_of(spec, root: Path) -> dict:
    """The manifest entry of a spec without rendering it (the file hash is not under test there)."""
    return spec.entry(mf.Results(root), file=f"{spec.fid}.pdf", sha1="0" * 40, nbytes=0)


def _with(root: Path, new_root: Path, rel: str, mutate) -> Path:
    shutil.copytree(root, new_root)
    d = load_rel(new_root, rel)
    mutate(d)
    write_json(new_root, rel, d)
    return new_root


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_tree(tmp_path_factory.mktemp("results_full"))


@pytest.fixture(scope="module")
def full_run(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("figs_full")
    return world, out, mf.run(world, out, only=ALL3)


HOLES = ("grid/qwen/sports.json", "aud/home.json", "aud/tools.json", "extra/toys.json")


@pytest.fixture(scope="module")
def holey(tmp_path_factory):
    """A tree with a rated panel, two next-item panels and one addendum-6 file missing, rendered once."""
    root = build_tree(tmp_path_factory.mktemp("results_holes"), skip=HOLES)
    out = tmp_path_factory.mktemp("figs_holes")
    return root, out, mf.run(root, out, only=ALL3)


# ------------------------------------------------------------------------------------------------ manifest round trip
def test_manifest_values_equal_source_files(full_run):
    root, out, manifest = full_run
    assert manifest["schema"] == mf.SCHEMA and set(manifest["figures"]) == set(ALL3)
    n = 0
    for fid, e in manifest["figures"].items():
        assert e["status"] == "ok" and e["missing"] == []
        for panel, v in all_values(e):
            node = resolve(root, v["source"])
            assert set(v["fields"]) <= {"est", "lo", "hi", "coverage", "n_users", "n_events"}, v["fields"]   # no p-value, no Holm flag
            if isinstance(node, dict):
                for k, x in v["fields"].items():
                    assert node[k] == x, (v["uid"], k)
            else:
                assert v["fields"] == {"est": node}, v["uid"]
            assert v["source"].split(":")[0] in {s["path"] for s in panel["sources"]}, "value read from an unlisted file"
            n += 1
    assert n > 200                                    # the synthetic tree fills every panel of all three figures


def test_manifest_sha1s_equal_file_hashes(full_run):
    root, out, manifest = full_run
    for fid, e in manifest["figures"].items():
        pdf = out / e["file"]
        assert e["sha1"] == sha1_of(pdf) and e["bytes"] == pdf.stat().st_size
        assert pdf.read_bytes()[:5] == b"%PDF-"
        listed = {s["path"]: s["sha1"] for s in e["sources"]}
        assert listed, fid
        for rel, sha in listed.items():
            assert sha == sha1_of(root / rel), rel
        for panel in e["panels"]:
            for s in panel["sources"]:
                assert listed[s["path"]] == s["sha1"]
    assert json.loads((out / mf.MANIFEST_NAME).read_text(encoding="utf-8")) == manifest


def test_manifest_lists_every_figure_source_and_n_boot(full_run):
    root, out, manifest = full_run
    tracks = manifest["figures"]["tracks"]
    assert {"grid/qwen/ml1m.json", "extra/ml1m.json", "extra/llama/toys.json", mf.GATE_FT} <= {s["path"] for s in tracks["sources"]}
    assert tracks["n_boot_values"] == [2000]
    assert [s["path"] for s in manifest["figures"]["serving"]["sources"]] == [f"aud/{d}.json" for d in ("home", "sports", "tools", "toys")]


# ------------------------------------------------------------------------------------------------ drawn = manifest
def _artists_by_gid(fig):
    return {a.get_gid(): a for a in fig.findobj(lambda a: a.get_gid() is not None) if a.get_gid()}


@pytest.mark.parametrize("fid,minimum", [("tracks", 40), ("shares", 100)])
def test_drawn_marks_equal_manifest(world, fid, minimum):
    spec = getattr(mf, f"collect_{fid}")(mf.Results(world))
    fig = getattr(mf, f"draw_{fid}")(spec)
    gids = _artists_by_gid(fig)
    horizontal = fid == "tracks"                     # F1 plots the estimate on x, F2 on y
    n_pt = 0
    for p in spec.panels:
        for v in p.vals:
            if v.role == "count":
                continue
            pt = gids[f"{v.uid}:pt"]
            assert (pt.get_xdata() if horizontal else pt.get_ydata())[0] == v.est
            assert (pt.get_markerfacecolor() == "white") == bool(v.flags), v.uid
            n_pt += 1
            if v.lo is not None:
                ci = gids[f"{v.uid}:ci"]
                assert list(ci.get_xdata() if horizontal else ci.get_ydata()) == [v.lo, v.hi]
            else:
                assert f"{v.uid}:ci" not in gids
    assert n_pt >= minimum
    mf.plt.close(fig)


def test_drawn_curves_equal_manifest(world):
    spec = mf.collect_serving(mf.Results(world))
    fig = mf.draw_serving(spec)
    gids = _artists_by_gid(fig)
    for p in spec.panels:
        for sig in mf.SERVING_SIGNALS:
            vals = p.by_series(sig)
            line = gids[f"serving/{p.pid}/{sig}:line"]
            assert list(line.get_xdata()) == [v.fields["coverage"] for v in vals]
            assert list(line.get_ydata()) == [v.est for v in vals]
            ys = {round(y, 12) for y in gids[f"serving/{p.pid}/{sig}:band"].get_paths()[0].vertices[:, 1]}
            assert {round(v.lo, 12) for v in vals} | {round(v.hi, 12) for v in vals} <= ys
        assert "margin" not in " ".join(g for g in gids if g)                      # a signal in the file that is not drawn
    mf.plt.close(fig)


def test_text_stays_inside_the_canvas(holey):
    """Layout is placed by hand in inches: nothing may spill over the edge of the PDF page, placeholders included."""
    root = holey[0]
    res = mf.Results(root)
    for fid in ALL3:
        fig = getattr(mf, f"draw_{fid}")(getattr(mf, f"collect_{fid}")(res))
        with mf.style():
            fig.canvas.draw()
            box = fig.bbox
            for t in fig.findobj(mf.plt.Text):
                if not t.get_visible() or not t.get_text().strip():
                    continue
                bb = t.get_window_extent()
                assert bb.x0 >= box.x0 - 1 and bb.x1 <= box.x1 + 1 and bb.y0 >= box.y0 - 1 and bb.y1 <= box.y1 + 1, \
                    (fid, t.get_text(), bb.bounds)
        mf.plt.close(fig)


def test_pdf_is_vector_truetype_and_exact_size(full_run):
    root, out, manifest = full_run
    for fid, e in manifest["figures"].items():
        data = (out / e["file"]).read_bytes()
        assert b"/Type3" not in data and (b"/CIDFontType2" in data or b"/TrueType" in data), fid
        assert b"/CreationDate" not in data and b"/ModDate" not in data
        m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", data)
        w, h = float(m.group(1)) / 72, float(m.group(2)) / 72
        assert abs(w - 7.0) < 0.01 and abs(h - e["size_in"][1]) < 0.01, (fid, w, h)


# ------------------------------------------------------------------------------------------------ placeholders and errors
def test_missing_panel_is_an_explicit_placeholder(holey):
    root, out, m = holey
    for fid, pid, miss in (("tracks", "qwen:sports", "grid/qwen/sports.json"), ("shares", "qwen:sports", "grid/qwen/sports.json"),
                           ("serving", "home:all", "aud/home.json")):
        e = m["figures"][fid]
        assert e["status"] == "partial"
        p = {p["id"]: p for p in e["panels"]}[pid]
        assert p["status"] == "not_run" and p["values"] == []
        assert p["missing"][0]["path"] == miss and p["missing"][0]["kind"] == "file"
        assert any(x["path"] == miss for x in e["missing"])
        assert (out / e["file"]).stat().st_size > 1000
    assert {x["path"] for x in m["figures"]["serving"]["missing"]} == {"aud/home.json", "aud/tools.json"}
    assert "extra/toys.json" in {x["path"] for x in m["figures"]["tracks"]["missing"]}
    fig = mf.draw_serving(mf.collect_serving(mf.Results(root)))           # the placeholder is drawn: its text names the missing file
    texts = " ".join(t.get_text() for t in fig.findobj(mf.plt.Text))
    assert "not run yet" in texts and "aud/home.json" in texts and "aud/tools.json" in texts
    mf.plt.close(fig)


def test_every_input_missing_gives_not_run_figures_never_a_crash(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    m = mf.run(empty, tmp_path / "o", only=ALL3)
    for fid, e in m["figures"].items():
        assert e["status"] == "not_run" and e["sources"] == [] and e["missing"], fid
        assert all(p["status"] == "not_run" and p["values"] == [] for p in e["panels"])
        assert (tmp_path / "o" / e["file"]).is_file()
    assert m["figures"]["tracks"]["meta"]["gate_ft"]["state"] == "missing"


def test_a_missing_element_inside_a_panel_is_marked_not_run(holey):
    root = holey[0]
    spec = mf.collect_tracks(mf.Results(root))
    toys = {p.pid: p for p in spec.panels}["qwen:toys"]
    assert toys.status == "partial" and toys.one("zero_shot") is not None and toys.one("matched_mean") is None
    assert any(m["series"] == "matched_mean" and m["path"] == "extra/toys.json" for m in toys.missing)
    fig = mf.draw_tracks(spec)
    texts = [t.get_text() for t in fig.findobj(mf.plt.Text)]
    assert sum("not run yet" in t for t in texts) >= 2 and any("extra/toys.json" in t for t in texts)
    mf.plt.close(fig)
    sh = {p.pid: p for p in mf.collect_shares(mf.Results(root)).panels}["qwen:toys"]
    assert sh.one("e_share.zs") is None and sh.one("item_prior_share.zs") is not None
    assert {m["path"] for m in sh.missing if m["kind"] == "file"} == {"extra/toys.json"}
    assert sh.one("G.zs") is not None                  # the grid-only gain is still drawn


def test_fig_not_run_placeholder_draws_reason_and_file():
    with mf.style():
        fig = mf.plt.figure(figsize=(2.0, 1.2))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        rec_ = mf.fig_not_run_placeholder(ax, "Home", [{"path": "aud/home.json", "reason": "result file not found"}])
        texts = [t.get_text() for t in ax.texts]
    assert rec_["status"] == "not_run" and rec_["missing"][0]["path"] == "aud/home.json"
    assert texts[0] == "not run yet" and "aud/home.json" in texts[1] and "result file not found" in texts[1]
    mf.plt.close(fig)


MALFORMED = [
    ("grid/qwen/ml1m.json", lambda d: d["E_A"]["ZS"]["UAUC_TEST"]["per_model"]["zeroshot"].pop("est"),
     "E_A.ZS.UAUC_TEST.per_model.zeroshot.est"),
    ("grid/qwen/ml1m.json", lambda d: d["E_A"].pop("ZS"), "E_A.ZS"),
    ("grid/qwen/ml1m.json", lambda d: d["E_A"]["ZS"]["UAUC_TEST"]["per_model"]["zeroshot"].update(est="0.6"),
     "E_A.ZS.UAUC_TEST.per_model.zeroshot.est"),
    ("grid/qwen/ml1m.json", lambda d: d["E_A"]["ZS"]["UAUC_TEST"]["per_model"]["zeroshot"].update(lo=9.0),
     "E_A.ZS.UAUC_TEST.per_model.zeroshot"),
    ("grid/qwen/ml1m.json", lambda d: d["E_A"]["ZS"]["UAUC_TEST"]["per_model"]["zeroshot"].update(lo=None),
     "E_A.ZS.UAUC_TEST.per_model.zeroshot"),
    ("grid/qwen/ml1m.json", lambda d: d["meta"].update(domain="toys"), "meta.domain"),
    ("grid/llama/toys.json", lambda d: d["meta"].update(backbone="Qwen3-8B"), "meta.backbone"),
    ("grid/qwen/toys.json", lambda d: d["E_D"]["ZS"]["shares"]["per_model"]["zeroshot"].update(shares_reading="maybe"),
     "E_D.ZS.shares.per_model.zeroshot.shares_reading"),
    ("extra/ml1m.json", lambda d: d["status"].update(items_2_to_7="provisional"), "status.items_2_to_7"),
    ("extra/toys.json", lambda d: d["E_J"]["ZS"].pop("rows_E_A"), "E_J.ZS.rows_E_A"),
    ("aud/toys.json", lambda d: d.update(domain="home"), "domain"),
    ("aud/toys.json", lambda d: d["segments"]["all"]["questions"]["next"]["D_selective_serving"]["signals"]["p_max"]["curve"]["ndcg10"][2]
     .update(coverage=0.05), "coverage"),
]


@pytest.mark.parametrize("rel,mutate,key", MALFORMED, ids=[f"{r.split('/')[-1]}:{k}" for r, _, k in MALFORMED])
def test_malformed_file_raises_a_clear_error_naming_file_and_key(tmp_path, rel, mutate, key):
    root = build_tree(tmp_path / "r")
    d = load_rel(root, rel)
    mutate(d)
    write_json(root, rel, d)
    with pytest.raises(mf.FigureDataError) as e:
        collect_all(root)
    assert rel in str(e.value) and key in str(e.value), str(e.value)


def test_invalid_json_and_wrong_top_level_name_the_file(tmp_path):
    root = build_tree(tmp_path / "r")
    (root / "aud" / "toys.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(mf.FigureDataError, match=r"aud/toys\.json: not valid JSON"):
        mf.collect_serving(mf.Results(root))
    (root / "aud" / "toys.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(mf.FigureDataError, match=r"aud/toys\.json: top level is list"):
        mf.collect_serving(mf.Results(root))


def test_null_or_unavailable_values_are_not_run_not_errors(tmp_path):
    root = build_tree(tmp_path / "r")
    d = load_rel(root, "grid/qwen/toys.json")
    d["E_A"]["ZS"]["UAUC_TEST"]["per_model"]["zeroshot"]["est"] = None
    d["E_A"]["FT"] = {"available": False, "reason": "arm not requested yet"}
    write_json(root, "grid/qwen/toys.json", d)
    p = {p.pid: p for p in mf.collect_tracks(mf.Results(root)).panels}["qwen:toys"]
    why = {m["series"]: m for m in p.missing}
    assert p.one("zero_shot") is None and "null" in why["zero_shot"]["reason"]
    assert p.one("lora") is None and why["lora"]["reason"] == "arm not requested yet" and why["lora"]["path"] == "grid/qwen/toys.json"
    assert p.one("item_mean") is not None              # the references of the zero-shot rows are still there


def test_incomplete_lora_regime_is_never_replaced(tmp_path):
    root = build_tree(tmp_path / "r")
    d = load_rel(root, "grid/qwen/games.json")
    d["E_A"]["FT"].update(complete=False, models=["s0", "s2"], missing_or_excluded=["s1"])
    write_json(root, "grid/qwen/games.json", d)
    p = {p.pid: p for p in mf.collect_tracks(mf.Results(root)).panels}["qwen:games"]
    assert p.one("lora") is None and not p.by_series("lora_seed0")
    assert "incomplete" in {m["series"]: m for m in p.missing}["lora"]["reason"]


# ------------------------------------------------------------------------------------------------ flags, withheld values, gate
def test_exploratory_and_descriptive_flags_reach_the_manifest_and_the_marks(tmp_path):
    root = build_tree(tmp_path / "r", over={"grid/qwen/games.json": {"n_users": 100}, "extra/games.json": {"n_users": 100}})
    spec = mf.collect_tracks(mf.Results(root))
    pan = {p.pid: p for p in spec.panels}
    mt = pan["qwen:ml1m"].one("matched_mean")
    assert mt.flags == ("exploratory",) and mt.flag_sources == ("extra/ml1m.json:status.items_2_to_7",) and mt.hollow
    assert pan["qwen:toys"].one("matched_mean").flags == () and pan["qwen:toys"].one("matched_mean").flag_sources     # checked, not flagged
    assert pan["llama:ml1m"].one("matched_mean").flags == ("exploratory",)              # either backbone
    z = pan["qwen:games"].one("zero_shot")                                                # 100 users: below the minimum
    assert z.flags == ("descriptive",) and "descriptive_min_n" in z.flag_sources[0]
    assert pan["qwen:games"].one("matched_mean").flags == ()                              # same rows (100 users in both files)
    assert pan["qwen:ml1m"].one("zero_shot").flags == ()
    sh = {p.pid: p for p in mf.collect_shares(mf.Results(root)).panels}
    e_ml1m, e_toys = sh["qwen:ml1m"].one("e_share.zs"), sh["qwen:toys"].one("e_share.zs")
    assert e_ml1m.flags == ("exploratory", "descriptive") and "extra/ml1m.json:E_G.descriptive" in e_ml1m.flag_sources
    assert e_toys.flags == ("descriptive",)             # E_G is flagged descriptive in every file
    assert sh["qwen:ml1m"].one("G_wu.zs").flags == ("exploratory",) and sh["qwen:toys"].one("G_wu.zs").flags == ()
    assert sh["qwen:ml1m"].one("G.zs").flags == ()      # the registered gain is not an exploratory analysis
    entry = entry_of(spec, root)
    assert entry["flags_used"] == ["descriptive", "exploratory"] and entry["meta"]["gate_ft"]["state"] == "pass"
    flagged = [v for _, v in all_values(entry) if "exploratory" in v["flags"]]
    assert flagged and all(v["flag_sources"] for v in flagged)
    for v in flagged:
        assert resolve(root, v["flag_sources"][0]) == "exploratory"
    fig = mf.draw_tracks(spec)                          # the marker is hollow exactly where a flag stands
    g = _artists_by_gid(fig)
    assert g[f"{mt.uid}:pt"].get_markerfacecolor() == "white" and g[f"{z.uid}:pt"].get_markerfacecolor() == "white"
    assert g[f"{pan['qwen:toys'].one('matched_mean').uid}:pt"].get_markerfacecolor() != "white"
    mf.plt.close(fig)


def test_uninterpretable_shares_are_withheld_not_drawn(tmp_path):
    root = build_tree(tmp_path / "r", over={"grid/qwen/toys.json": {"zs_reading": "uninterpretable"}})
    spec = mf.collect_shares(mf.Results(root))
    p = {p.pid: p for p in spec.panels}["qwen:toys"]
    assert p.one("item_prior_share.zs") is None and p.one("non_prior_share.zs") is None
    assert {w["series"] for w in p.withheld} == {"item_prior_share.zs", "non_prior_share.zs"}
    assert all("uninterpretable" in w["reason"] and w["source"].startswith("grid/qwen/toys.json:E_D.ZS.shares") for w in p.withheld)
    assert p.one("item_prior_share.lora") is not None and p.one("e_share.zs") is not None      # the other regime and e-share stay
    assert p.status == "partial" and any("e_share.zs" in n and "withheld" in n for n in p.notes)
    withheld = load_rel(root, "grid/qwen/toys.json")["E_D"]["ZS"]["shares"]["per_model"]["zeroshot"]["item_prior_share"]["est"]
    assert withheld not in [v["fields"].get("est") for _, v in all_values(entry_of(spec, root))]   # the withheld number is not listed
    fig = mf.draw_shares(spec)
    assert sum(1 for t in fig.findobj(mf.plt.Text) if t.get_text() == "n/i") == 2
    mf.plt.close(fig)
    bad = _with(root, tmp_path / "bad", "grid/qwen/toys.json", lambda d: d["E_D"]["FT"]["shares"]["mean_over_seeds"].update(shares_reading="x"))
    with pytest.raises(mf.FigureDataError):
        mf.collect_shares(mf.Results(bad))


def test_lora_is_drawn_only_after_a_recorded_gate_ft_pass(tmp_path):
    failed = build_tree(tmp_path / "f", gate="GATE_FT_FAIL")
    spec = mf.collect_tracks(mf.Results(failed))
    for p in spec.panels:
        assert not p.by_series("lora") and p.one("zero_shot") is not None
        assert any(m["series"] == "lora" and m["kind"] == "gate" for m in p.missing), p.pid
    assert spec.meta["gate_ft"] == {"state": "not_passed", "decision": "GATE_FT_FAIL"}
    sh = mf.collect_shares(mf.Results(failed))
    assert all(not [v for v in p.vals if "lora" in v.series] for p in sh.panels)
    assert all(any(m["series"] == "lora" for m in p.missing) for p in sh.panels)
    nogate = build_tree(tmp_path / "n", skip=("gft/gate_ft.json",))
    spec = mf.collect_tracks(mf.Results(nogate))
    assert spec.meta["gate_ft"]["state"] == "missing" and all(not p.by_series("lora") for p in spec.panels)
    assert any(m["path"] == mf.GATE_FT and m["kind"] == "file" for p in spec.panels for m in p.missing)
    ok = mf.collect_tracks(mf.Results(build_tree(tmp_path / "g")))
    assert all(len(p.by_series("lora")) == 1 and all(p.by_series(f"lora_seed{k}") for k in range(3)) for p in ok.panels)


def test_references_and_matched_mean_on_other_rows_are_flagged(tmp_path):
    root = build_tree(tmp_path / "r")
    d = load_rel(root, "grid/qwen/toys.json")
    d["E_A"]["FT"]["UAUC_TEST"]["references"]["q_hat"]["est"] += 0.05
    write_json(root, "grid/qwen/toys.json", d)
    e = load_rel(root, "extra/games.json")
    e["E_J"]["ZS"]["rows_E_A"]["n_users"] += 1
    write_json(root, "extra/games.json", e)
    pan = {p.pid: p for p in mf.collect_tracks(mf.Results(root)).panels}
    assert pan["qwen:toys"].one("item_mean").flags == ("rows_differ",) and pan["qwen:toys"].notes
    assert pan["qwen:toys"].one("mf").flags == ()
    assert pan["qwen:games"].one("matched_mean").flags == ("rows_differ",)


def test_f3_descriptive_when_fewer_than_the_minimum_events(tmp_path):
    root = build_tree(tmp_path / "r", over={"aud/tools.json": {"n_events": 80}})
    pan = {p.pid: p for p in mf.collect_serving(mf.Results(root)).panels}
    assert all(v.flags == ("descriptive",) for v in pan["tools:all"].by_series("p_max") + pan["tools:all"].by_series("random"))
    assert not any(v.flags for v in pan["toys:all"].data())


def test_serving_uses_the_family_segment_not_the_quarantine(world):
    spec = mf.collect_serving(mf.Results(world))
    sports = spec.panels[0]
    assert sports.pid == "sports:events_1001_10000" and sports.title.startswith("Sports")
    assert all(":segments.events_1001_10000" in v.source for v in sports.vals)
    assert not any("events_1_1000" in v.source for p in spec.panels for v in p.vals)


# ------------------------------------------------------------------------------------------------ path guard
def test_results_refuses_paths_outside_the_root(tmp_path):
    root = build_tree(tmp_path / "results")
    decoy = tmp_path / "outside" / "decoy.json"
    decoy.parent.mkdir()
    decoy.write_text(json.dumps({"E_A": {}}), encoding="utf-8")
    res = mf.Results(root)
    for bad in ("../outside/decoy.json", "grid/../../outside/decoy.json", str(decoy), decoy.as_posix(), "/etc/passwd",
                "C:/x.json", "grid\\qwen\\ml1m.json", "", "..", "grid/./../.."):
        with pytest.raises(mf.PathGuardError):
            res.load(bad)
        with pytest.raises(mf.PathGuardError):
            res.exists(bad)
    assert res.opened == []                            # nothing was opened by the refused calls
    assert res.load("gft/gate_ft.json") == {"decision": "GATE_FT_PASS"} and res.opened == ["gft/gate_ft.json"]


def test_a_link_pointing_outside_the_root_is_refused(tmp_path):
    root = build_tree(tmp_path / "results")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.json").write_text("{}", encoding="utf-8")
    try:
        os.symlink(outside, root / "linked", target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("symbolic links are not available here")
    with pytest.raises(mf.PathGuardError):
        mf.Results(root).load("linked/x.json")


def test_no_figure_function_opens_a_file_outside_the_results_root(tmp_path, monkeypatch):
    root = build_tree(tmp_path / "results", skip=("grid/qwen/sports.json",))
    decoy = tmp_path / "outside" / "decoy.json"
    decoy.parent.mkdir()
    decoy.write_text(json.dumps({"meta": {"domain": "sports"}}), encoding="utf-8")
    (root / "notes").mkdir()
    (root / "notes" / "ref.json").write_text(json.dumps({"path": "../../outside/decoy.json"}), encoding="utf-8")   # a lure
    opened = []
    real_io_open, real_os_open = io.open, os.open

    def spy_open(file, *a, **k):
        opened.append(os.fspath(file) if not isinstance(file, int) else file)
        return real_io_open(file, *a, **k)

    def spy_os_open(path, *a, **k):
        opened.append(os.fspath(path))
        return real_os_open(path, *a, **k)

    monkeypatch.setattr(io, "open", spy_open)
    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(os, "open", spy_os_open)
    out = tmp_path / "o"
    collect_all(root)                                  # the data layer of every figure
    entry = mf.fig_serving(root, out)                  # and one figure function end to end (reads, draws, writes)
    monkeypatch.undo()
    names = {str(Path(x).resolve()) for x in opened if isinstance(x, str)}
    assert str(decoy.resolve()) not in names
    res_root = root.resolve()
    data_files = {x for x in names if x.lower().endswith(".json") and Path(x).is_relative_to(tmp_path.resolve())}
    assert data_files, "the spy saw no data file"
    assert all(Path(x).is_relative_to(res_root) for x in data_files), data_files
    assert entry["status"] == "ok" and (out / "serving.pdf").is_file()


# ------------------------------------------------------------------------------------------------ determinism, CLI
def test_same_inputs_give_byte_identical_pdfs_and_entries(full_run, tmp_path):
    root, out, manifest = full_run
    again = mf.fig_serving(root, tmp_path)
    assert again == manifest["figures"]["serving"]
    assert (tmp_path / "serving.pdf").read_bytes() == (out / "serving.pdf").read_bytes()
    text = json.dumps(manifest)
    assert not re.search(r"[A-Za-z]:[\\/]", text) and "CreationDate" not in text and Path.home().name not in text


def test_determinism_across_processes_with_different_hash_seeds(world, tmp_path):
    """Two fresh interpreters (different PYTHONHASHSEED), same inputs: byte-identical PDF and manifest, through the CLI script."""
    procs = []
    for k, seed in enumerate(("1", "2")):
        out = tmp_path / f"p{k}"
        env = {**os.environ, "PYTHONHASHSEED": seed, "MPLBACKEND": "Agg"}
        procs.append((out, subprocess.Popen([sys.executable, str(SCRIPT), "--results", str(world), "--out", str(out), "--only", "tracks"],
                                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT)))
    for out, p in procs:
        so, se = p.communicate(timeout=150)
        assert p.returncode == 0, se
        assert "tracks" in so and "sha1" in so
    a, b = procs[0][0], procs[1][0]
    for name in ("tracks.pdf", mf.MANIFEST_NAME):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_cli_main_merges_entries_and_drops_stale_ones(world, tmp_path, capsys):
    a = tmp_path / "a"
    assert mf.main(["--results", str(world), "--out", str(a), "--only", "serving"]) == 0
    assert "serving" in capsys.readouterr().out
    (a / "decomp.tex").write_text("\\begin{tikzpicture}\\end{tikzpicture}\n", encoding="utf-8")
    assert mf.main(["--results", str(world), "--out", str(a), "--only", "decomp"]) == 0
    man = json.loads((a / mf.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert list(man["figures"]) == ["serving", "decomp"]                             # the figure not asked for is kept
    assert man["figures"]["decomp"]["kind"] == "tikz" and man["figures"]["decomp"]["sha1"] == sha1_of(a / "decomp.tex")
    mtime = (a / "serving.pdf").stat().st_mtime_ns
    assert mf.main(["--results", str(world), "--out", str(a), "--only", "serving"]) == 0     # same bytes: the file is not rewritten
    assert (a / "serving.pdf").stat().st_mtime_ns == mtime
    (a / "decomp.tex").write_text("changed\n", encoding="utf-8")                      # a kept entry must still match its file
    assert mf.main(["--results", str(world), "--out", str(a), "--only", "serving"]) == 0
    assert list(json.loads((a / mf.MANIFEST_NAME).read_text(encoding="utf-8"))["figures"]) == ["serving"]
    capsys.readouterr()


def test_cli_errors(world, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        mf.main(["--results", str(world), "--out", str(tmp_path / "x"), "--only", "nonsense"])
    assert e.value.code == 2 and "unknown figure id" in capsys.readouterr().err
    assert mf.main(["--results", str(tmp_path / "no_such_dir"), "--out", str(tmp_path / "x"), "--only", "serving"]) == 2
    assert "not a directory" in capsys.readouterr().err
    root = build_tree(tmp_path / "r")
    d = load_rel(root, "aud/toys.json")
    del d["segments"]["all"]["questions"]
    write_json(root, "aud/toys.json", d)
    assert mf.main(["--results", str(root), "--out", str(tmp_path / "x"), "--only", "F3"]) == 2
    err = capsys.readouterr().err
    assert "aud/toys.json" in err and "segments.all.questions" in err
    assert mf.normalise_ids(["F1,F2", "fig_serving"]) == list(ALL3) and mf.normalise_ids(None) == list(mf.FIG_IDS)


def test_decomp_is_listed_by_hash_and_must_exist(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    with pytest.raises(mf.FigureDataError, match="decomp.tex"):
        mf.run(tmp_path, out, only=("decomp",))
    (out / "decomp.tex").write_text("\\begin{tikzpicture}\\end{tikzpicture}\n", encoding="utf-8")
    e = mf.run(tmp_path, out, only=("decomp",))["figures"]["decomp"]
    assert e["kind"] == "tikz" and e["sha1"] == sha1_of(out / "decomp.tex") and e["sources"] == [] and e["status"] == "ok"


# ------------------------------------------------------------------------------------------------ the real tree
@pytest.mark.skipif(not (REAL / "grid" / "qwen" / "ml1m.json").is_file(), reason="the real results are not pulled")
def test_real_tree_round_trip(tmp_path):
    """Whatever panels the committed tree holds today render with data, the rest as placeholders; the round trip holds for both."""
    snap = tmp_path / "snapshot"                                   # one consistent copy: a pull may rewrite the tree meanwhile
    shutil.copytree(REAL, snap, ignore=shutil.ignore_patterns("*.csv"))
    specs = collect_all(snap)
    n = 0
    for spec in specs:
        e = entry_of(spec, snap)
        for s in e["sources"]:
            assert s["sha1"] == sha1_of(snap / s["path"])
        for panel, v in all_values(e):
            node = resolve(snap, v["source"])
            if isinstance(node, dict):
                assert all(node[k] == x for k, x in v["fields"].items()), v["uid"]
            else:
                assert v["fields"] == {"est": node}
            n += 1
        for p in e["panels"]:
            assert p["status"] in ("ok", "partial") or (p["status"] == "not_run" and p["missing"])
            for miss in p["missing"]:
                if miss["kind"] == "file":
                    assert not (snap / miss["path"]).exists(), miss
    tracks = {p.pid: p for p in specs[0].panels}
    assert tracks["qwen:ml1m"].status in ("ok", "partial") and tracks["llama:ml1m"].status in ("ok", "partial")
    assert tracks["qwen:toys"].status in ("ok", "partial") and n > 100
    m = mf.run(snap, tmp_path / "figs", only=("tracks",))             # and the heaviest figure renders from the real files
    assert m["figures"]["tracks"]["sha1"] == sha1_of(tmp_path / "figs" / "tracks.pdf")


# ------------------------------------------------------------------------------------------------ style and discipline
def test_style_constants_follow_the_brief():
    assert mf.STYLE["pdf.fonttype"] == 42 and mf.STYLE["font.size"] <= 8.0 and mf.STYLE["xtick.labelsize"] >= 6.0
    markers = [s["marker"] for s in mf.TRACK_STYLE.values()]
    assert len(set(markers)) == len(markers)           # every F1 series has its own marker: it reads without colour
    assert len({mf.SHARE_STYLE[k]["marker"] for k in ("zs", "lora", "ref")}) == 3
    lum = lambda h: sum(c * w for c, w in zip((int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)), (0.2126, 0.7152, 0.0722)))     # noqa: E731
    assert abs(lum(mf.C_ZS) - lum(mf.C_LORA)) > 0.05 and abs(lum(mf.C_REF) - lum(mf.C_ZS)) > 0.05
    for k in (mf.INK, mf.INK2):
        assert lum(k) < 0.35                            # text wears dark ink, never a series colour
    assert mf.UAUC_AXIS_START in (0.45, 0.5)


def test_constants_agree_with_fill_paper():
    fill = _load("_fill_paper_for_figures", FILL)
    assert mf.RATED == fill.RATED and mf.RATED_NAME == fill.RATED_NAME and mf.MIN_N == fill.MIN_N and mf.FT_MODELS == fill.FT_MODELS
    assert tuple((d, seg) for _, d, seg in mf.SERVING_UNITS) == tuple((d, seg) for _, d, seg in fill.FAMILY_UNITS)
    assert mf.ext_rel("qwen", "toys") == fill.ext_rel("qwen", "toys") and mf.ext_rel("llama", "toys") == fill.ext_rel("llama", "toys")
    assert mf.GATE_FT == fill.GFT


def test_script_holds_no_result_like_literal():
    """A pasted result looks like 0.742; layout constants have at most two decimals."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            txt = repr(node.value)
            if "e" not in txt and len(txt.split(".")[1]) >= 3:
                bad.append((node.lineno, txt))
    assert not bad, bad


def test_proposed_captions_hold_no_result_number():
    notes = FIGDIR / "FIGURE_NOTES.md"
    if not notes.is_file():
        pytest.skip("FIGURE_NOTES.md not written yet")
    text = notes.read_text(encoding="utf-8")
    caps = re.findall(r"\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", text)
    assert len(caps) >= 4, "one caption per figure expected"
    for c in caps:
        body = re.sub(r"\\(?:ref|label|cite[a-z]*)\{[^}]*\}", "", c)
        body = re.sub(r"\$[^$]*\$", "", body)                                    # symbols such as $m_T$ or $S_d$
        nums = re.findall(r"\d[\d.,]*", body)
        assert set(nums) <= {"95"}, (nums, c[:80])                               # the interval level is a definition, not a result


def _have_latex() -> bool:
    return shutil.which("pdflatex") is not None


@pytest.mark.skipif(not (FIGDIR / "decomp.tex").is_file() or not _have_latex(), reason="decomp.tex or LaTeX missing")
def test_decomp_tex_compiles_in_the_paper_class(tmp_path):
    """The schematic loads with \\input into the sigconf class with the packages main.tex needs plus tikz; no overfull box."""
    tex = tmp_path / "scratch.tex"
    tex.write_text("\n".join([
        r"\documentclass[sigconf,natbib=true,anonymous=true]{acmart}",
        r"\usepackage{booktabs,array,amsmath,graphicx,xcolor,multirow,subcaption}",
        r"\usepackage{tikz}\usetikzlibrary{arrows.meta,positioning}",
        r"\setcopyright{none}\settopmatter{printacmref=false}\pagestyle{empty}",
        r"\begin{document}",
        r"\newsavebox{\decompbox}\sbox{\decompbox}{\input{" + (FIGDIR / "decomp").as_posix() + r"}}",
        r"\typeout{DECOMPSIZE wd=\the\wd\decompbox ht=\the\ht\decompbox col=\the\columnwidth}",
        r"\begin{figure}[t]\centering\input{" + (FIGDIR / "decomp").as_posix() + r"}\caption{x}\end{figure}",
        r"\end{document}"]), encoding="utf-8")
    try:
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=tmp_path, capture_output=True,
                           text=True, timeout=100, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pytest.skip("pdflatex did not finish (package installer prompt?)")
    log = (tmp_path / "scratch.log").read_text(encoding="latin-1", errors="replace").replace("\n", "")
    assert r.returncode == 0 and (tmp_path / "scratch.pdf").is_file(), log[-1500:]
    m = re.search(r"DECOMPSIZE wd=([\d.]+)pt ht=([\d.]+)pt col=([\d.]+)pt", log)
    assert m and float(m.group(1)) <= float(m.group(3)) + 0.01, m.groups()          # fits one column
    assert "Overfull \\hbox" not in log
