"""scripts/sigir/fill_paper.py (and the file list of scripts/sigir/pull_results.ps1): the deterministic path from the
committed result files to the numbers of the paper. CPU only, deterministic, no network, no GPU, no torch.

Fixtures, all with the REAL schemas:
  * the real skeleton Paper/sigir2027 (copied; the originals are checked byte-identical after every run);
  * the real result files pulled into docs/sigir/results (selection.json, gate.json, gate_ft.json, the sports next-item
    audit, the ftgrid split files), copied into an isolated results directory (skipped when not pulled);
  * synthetic results produced by the producing code itself: src.confrec.ftgrid_report on the synthetic worlds of
    tests/test_confrec_ftgrid_report.py (ML-1M with every arm, a small Toys world with the knockout arms, a world whose
    seed s2 fails E1), src.confrec.ftprune.analyze on the world of tests/test_confrec_ftprune.py (P3 cut),
    src.confrec.ftmethod_report.decide / slot_summary (slot killed after two failures), scripts/sigir/pilot1_gate.decide
    (stage 3), src.confrec.nextitem_audit.summarize on domain copies of the real sports audit (Holm families).
Expected cells are recomputed here from the result files with an independent formatter (FILL RULE 2)."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "Paper" / "sigir2027"
RESULTS = ROOT / "docs" / "sigir" / "results"
SCRIPT = ROOT / "scripts" / "sigir" / "fill_paper.py"
PULL = ROOT / "scripts" / "sigir" / "pull_results.ps1"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                      # dataclasses with postponed annotations need the module registered
    spec.loader.exec_module(mod)
    return mod


fill = _load("_fill_paper_under_test", SCRIPT)


# ------------------------------------------------------------------------------------------------ independent helpers
def n3(x) -> str:
    s = f"{abs(x):.3f}"
    return ("$-$" if x < 0 and float(s) != 0 else "") + s


def s3(x) -> str:
    s = f"{abs(x):.3f}"
    s = s[1:] if s.startswith("0.") else s
    return ("$-$" if x < 0 and float(s) != 0 else "") + s


def cell(est, lo=None, hi=None, sd=None, desc=False) -> str:
    """FILL RULE 2 as specified: estimate, then the half-width of the 95% interval (three decimals); LoRA: + seed s.d."""
    extra = ""
    if lo is not None and hi is not None:
        extra += r"$\pm$" + s3((hi - lo) / 2)
    if sd is not None:
        extra += r"\,(" + s3(sd) + ")"
    out = n3(est) + (r"{\scriptsize" + extra + "}" if extra else "")
    return out + (r"{\scriptsize\,(descriptive)}" if desc else "")


def rec_cell(rec, sd=None, n_key="n_users") -> str:
    n = rec.get(n_key)
    desc = rec.get("descriptive_min_n") is True or (n is not None and n < 150)
    return cell(rec["est"], rec.get("lo"), rec.get("hi"), sd, desc)


def sd1(xs) -> float:
    return float(np.std(np.asarray(xs, float), ddof=1))


def sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def slot_spans(text: str) -> list:
    """(start, end, inner) of every non-comment \\DATANEEDED{...} (brace-balanced), independently of the script."""
    out, i, key = [], 0, "\\DATANEEDED{"
    while (j := text.find(key, i)) >= 0:
        ls = text.rfind("\n", 0, j) + 1
        commented = re.search(r"(?<!\\)%", text[ls:j]) is not None
        depth, p = 1, j + len(key)
        while depth:
            if text[p] == "\\":
                p += 2
                continue
            depth += {"{": 1, "}": -1}.get(text[p], 0)
            p += 1
        if not commented:
            out.append((j, p, text[j + len(key):p - 1]))
        i = p
    return out


def row_cells(text: str, label: str) -> list:
    """The cells of the table row whose label is `label` (one row per line), \\multicolumn unwrapped."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(label + " &"):
            body = s[len(label):].strip()
            body = body[:-2] if body.endswith("\\\\") else body
            cells = [c.strip() for c in re.split(r"(?<!\\)&", body)][1:]     # empty spacer columns kept
            out = []
            for c in cells:
                m = re.match(r"^\\multicolumn\{\d+\}\{[^}]*\}\{(.*)\}$", c)
                out.append(m.group(1) if m else c)
            return out
    raise AssertionError(f"no row {label!r}")


def table_text(text: str, label: str) -> str:
    i = text.index(f"\\label{{{label}}}")
    return text[i:text.index("\\end{tabular}", i)]


def copy_skeleton(dst: Path) -> Path:
    (dst / "sections").mkdir(parents=True)
    for name in ("main.tex", "references.bib"):
        shutil.copyfile(PAPER / name, dst / name)
    for f in fill.SECTION_FILES:
        shutil.copyfile(PAPER / "sections" / f"{f}.tex", dst / "sections" / f"{f}.tex")
    return dst


def skeleton_sha1() -> dict:
    files = [PAPER / "main.tex", PAPER / "references.bib", *sorted((PAPER / "sections").glob("*.tex"))]
    return {str(p.relative_to(PAPER)): sha1(p) for p in files}


def read(out: Path, rel: str) -> str:
    return (out / rel).read_text(encoding="utf-8")


def unfilled_of(doc: dict, **match) -> list:
    return [u for u in doc["unfilled"] if all(u.get(k) == v for k, v in match.items())]


# ------------------------------------------------------------------------------------------------ fixtures: real files
REAL_FILES = ("sel/selection.json", "gate/gate.json", "gft/gate_ft.json", "aud/sports.json",
              *(f"grid/qwen/{d}_split.json" for d in ("ml1m", "toys", "games", "sports")),
              *(f"grid/llama/{d}_split.json" for d in ("ml1m", "toys")))


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    missing = [r for r in REAL_FILES if not (RESULTS / r).is_file()]
    if missing:
        pytest.skip(f"real result files not pulled (scripts/sigir/pull_results.ps1): {missing}")
    base = tmp_path_factory.mktemp("real")
    res = base / "results"
    for r in REAL_FILES:
        (res / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RESULTS / r, res / r)
    before = skeleton_sha1()
    out = base / "filled"
    run = fill.run(PAPER, res, out)
    return SimpleNamespace(res=res, out=out, run=run, doc=json.loads(read(out, "UNFILLED.json")),
                           filled=json.loads(read(out, "FILLED.json")), before=before,
                           j={r: json.loads((res / r).read_text(encoding="utf-8")) for r in REAL_FILES})


# ------------------------------------------------------------------------------------------------ fixtures: synthetic
def _put(res: Path, rel: str, obj) -> None:
    from src.confrec.stats import strict_json
    p = res / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, (dict, list)):
        p.write_text(json.dumps(strict_json(obj), indent=1, allow_nan=False), encoding="utf-8")
    else:
        shutil.copyfile(obj, p)


def _slot_report(fr, fm, d: str, delta: float, seed: int) -> dict:
    """A per-dataset method-slot report: its blocks from ftgrid_report.uauc_models / contrast_models (as
    ftmethod_report.build calls them) and its decision from ftmethod_report.decide."""
    rng = np.random.default_rng(seed)
    n_u = 200
    users = np.repeat(np.arange(n_u), 6)
    y = np.tile([1, 1, 1, 0, 0, 0], n_u)
    rows = np.ones(len(y), bool)
    stack = {k: y * 0.8 + rng.normal(0, 1, len(y)) for k in range(3)}
    offs = {k: stack[k] + delta * y + rng.normal(0, 0.3, len(y)) for k in range(3)}
    diff = fr.contrast_models({f"seed{k}": (offs[k], stack[k]) for k in range(3)}, y, users, rows, 100, 0,
                              n_registered=3)
    slot = {"SFT_b0": fr.uauc_models({f"s{k}": stack[k] for k in range(3)}, y, users, rows, 100, 0, n_registered=3),
            "post_hoc_stacking": fr.uauc_models({f"s{k}": stack[k] for k in range(3)}, y, users, rows, 100, 0,
                                                n_registered=3),
            "prior_offset_LoRA": fr.uauc_models({f"o{k}": offs[k] for k in range(3)}, y, users, rows, 100, 0,
                                                n_registered=3),
            "difference": diff}
    return {"spec": fm.SPEC, "alias": "slot", "meta": {"domain": d, "backbone": "Qwen3-8B"}, "slot": slot,
            "decision": fm.decide(diff, 3, [])}


def _mirror_inputs(d_lo: float, e_lo: float) -> dict:
    """The fields of a pilot_mirror.json that pilot1_gate.domain_criteria reads."""
    return {"acquiescence": {"SD_pair_df": 0.3, "SD_pair": 0.3, "corr_a_logpop": {"est": 0.3}},
            "valence": {"corr_v_logpop": {"est": 0.1}}, "prior": {"corr_pi_logpop": {"est": 0.1}},
            "dUAUC_mirror_minus_placebo": {"est": d_lo + 0.01, "lo": d_lo, "hi": d_lo + 0.02, "n": 1683},
            "ensemble_null": {"dUAUC_mirror_minus_ensemble_null": {"est": e_lo + 0.01, "lo": e_lo, "hi": e_lo + 0.02,
                                                                   "n": 1683},
                              "mirror_pair": {"predicted_UAUC": 0.6, "observed_UAUC": 0.61},
                              "dUAUC_placebo_minus_ensemble_null": {"est": 0.0}}}


def _audit_docs(sports: dict) -> dict:
    """Toys / Home / Tools audits as one-segment copies of the real sports main segment (the registered layout of a
    non-sports domain), and the Z2 audits (one segment each, named after the test role)."""
    main = sports["segments"]["events_1001_10000"]
    docs = {"sports": sports}
    for d in ("toys", "home", "tools"):
        doc = copy.deepcopy(sports)
        doc["domain"] = d
        doc["segments"] = {"all": {**copy.deepcopy(main), "role": "all"}}
        docs[d] = doc
    z2 = {}
    for which, seg in (("aud2q", "test"), ("aud2l", "test1001_3000")):
        for d in ("sports", "toys", "home", "tools"):
            doc = copy.deepcopy(sports)
            doc["domain"] = d
            doc["segments"] = {seg: {**copy.deepcopy(main), "role": "all"}}
            z2[(which, d)] = doc
    return docs, z2


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    for r in ("sel/selection.json", "gate/gate.json", "gft/gate_ft.json", "aud/sports.json"):
        if not (RESULTS / r).is_file():
            pytest.skip(f"real result file not pulled: {r}")
    from src.confrec import ftgrid_report as fr
    from src.confrec import ftmethod_report as fm
    from src.confrec import nextitem_audit as na
    tfr = _load("_tfr_worlds", ROOT / "tests" / "test_confrec_ftgrid_report.py")
    tfp = _load("_tfp_worlds", ROOT / "tests" / "test_confrec_ftprune.py")
    p1g = _load("_pilot1_gate", ROOT / "scripts" / "sigir" / "pilot1_gate.py")
    base = tmp_path_factory.mktemp("synth")
    res = base / "results"
    for r in ("sel/selection.json", "gate/gate.json", "gft/gate_ft.json"):
        _put(res, r, RESULTS / r)
    for bb, ds in (("qwen", ("ml1m", "toys", "games", "sports")), ("llama", ("ml1m", "toys"))):
        for d in ds:
            if (RESULTS / f"grid/{bb}/{d}_split.json").is_file():
                _put(res, f"grid/{bb}/{d}_split.json", RESULTS / f"grid/{bb}/{d}_split.json")
    # grid reports: the real ftgrid_report on synthetic worlds
    w1 = tfr.make_world(base / "w_ml1m", "ml1m", seed=0, n_eval=320, n_bg=80)
    tfr.write_models(w1, tfr.ML1M_SPECS)
    ml1m, ml1m_out = tfr.run_main(w1, "ml1m", list(tfr.ML1M_SPECS), n_boot=20)
    w2 = tfr.make_world(base / "w_toys", "toys", seed=12, n_eval=140, n_bg=40, raw=False)
    ko_specs = {m: {"beta": 0.5} for m in ("zeroshot", "s0", "s1", "s2")}
    tfr.write_models(w2, ko_specs, arms=("like", "pseudo", "placebo"))
    toys, toys_out = tfr.run_main(w2, "toys", list(ko_specs), n_boot=20, extra=("--bi_boot", "5"))
    w3 = tfr.make_world(base / "w_games", "ml1m", seed=13, n_eval=300, n_bg=60)
    n_pairs = sum(len(r["candidate_item_ids"]) for r in w3.eval_rows)
    g_specs = {"zeroshot": {"beta": 0.3}, "s0": {"beta": 1.0}, "s1": {"beta": 1.0}, "s2": {"beta": 1.0}}
    tfr.write_models(w3, g_specs, arms=("like", "swap"), cens2={("s2", "like"): tuple(range(0, n_pairs, 20))})
    games, games_out = tfr.run_main(w3, "games", list(g_specs), n_boot=20)
    _put(res, "grid/qwen/ml1m.json", ml1m_out)
    _put(res, "grid/qwen/toys.json", toys_out)
    _put(res, "grid/qwen/games.json", games_out)
    _put(res, "grid/qwen/sports.json", ml1m_out)            # a fourth domain with the ML-1M report's numbers
    _put(res, "grid/llama/ml1m.json", ml1m_out)
    _put(res, "grid/llama/toys.json", toys_out)
    # pruning: the real ftprune.analyze, P3 cut at the checkpoint
    pw = tfp.analysis_world(base / "w_prune", {**tfp.BASE, "P2": 1.5}, n_users=200)
    prn = tfp.run_analyze(pw, arms=("P0", "P1", "P2"), n_boot=100)
    _put(res, "prn/pruning_ml1m.json", prn)
    # method slot: ML-1M passes, Toys and Video Games fail -> killed after games, Sports not run
    slot_root = base / "ftmethod"
    for d, delta, seed in (("ml1m", 0.6, 1), ("toys", 0.0, 2), ("games", 0.0, 3)):
        from src.confrec.stats import strict_json
        rep = strict_json(_slot_report(fr, fm, d, delta, seed))
        (slot_root / d).mkdir(parents=True)
        (slot_root / d / "report.json").write_text(json.dumps(rep), encoding="utf-8")
        _put(res, f"slot/{d}.json", rep)
    slot = fm.slot_summary(slot_root)
    _put(res, "slot/slot.json", slot)
    # stage 3 (GATEPASS): the real pilot1_gate.decide, assembled as its main() writes decision.json
    sp = {"dNDCG@10_mirror_minus_raw": {"est": -0.001, "lo": -0.004, "hi": 0.002, "n": 1000,
                                        "tie_exact": {"est": -0.0012, "lo": -0.0041, "hi": 0.0019, "n": 1000}},
          "dNDCG@10_mirror_minus_raw_like": {"est": 0.0, "lo": -0.003, "hi": 0.003, "n": 1000,
                                             "tie_exact": {"est": 0.0, "lo": -0.003, "hi": 0.003, "n": 1000}},
          "arms": {a: {"NDCG@10_tie_exact": 0.21, "NDCG@10": 0.21} for a in ("raw", "raw_like", "mirror", "placebo")}}
    dec = p1g.decide({"ml1m": _mirror_inputs(0.005, 0.002), "toys": _mirror_inputs(-0.01, -0.02)}, sp, True)
    mir = {"decision": dec.pop("decision"), "gate": {"pass": True}, **dec,
           "inputs": {"ml1m": "outputs/confrec/gatefix/stage3/ml1m/pilot_mirror.json",
                      "toys": "outputs/confrec/gatefix/stage3/toys/pilot_mirror.json",
                      "sports": "outputs/confrec/gatefix/stage3/sports_valid_1k/pilot_mirror.json"}}
    _put(res, "mir/decision.json", mir)
    # next-item audits and the cross-domain summary (the real summarize)
    sports = json.loads((RESULTS / "aud/sports.json").read_text(encoding="utf-8"))
    docs, z2 = _audit_docs(sports)
    for d, doc in docs.items():
        _put(res, f"aud/{d}.json", doc)
    summary = na.summarize(list(docs.values()))
    _put(res, "aud/summary.json", summary)
    for (which, d), doc in z2.items():
        _put(res, f"{which}/{d}.json", doc)
    out = base / "filled"
    run = fill.run(PAPER, res, out)
    return SimpleNamespace(res=res, out=out, run=run, doc=json.loads(read(out, "UNFILLED.json")), ml1m=ml1m,
                           toys=toys, games=games, prn=prn, slot=slot, mir=mir, docs=docs, summary=summary, base=base,
                           na=na, tfp=tfp, pw=pw)


# ================================================================================================ the skeleton
def test_every_slot_is_classified_and_the_specs_cover_the_skeleton(real):
    n_slots = sum(len(slot_spans((PAPER / "sections" / f"{f}.tex").read_text(encoding="utf-8")))
                  for f in fill.SECTION_FILES)
    sm = real.doc["summary"]
    assert sm["slots_total"] == n_slots == sm["filled"] + sm["unfilled"]
    assert sum(sm["unfilled_by_reason"].values()) == sm["unfilled"]
    assert set(sm["unfilled_by_reason"]) <= set(fill.REASONS)
    # every table row and prose slot of the current skeleton is known to TABLE_SPECS / PROSE_SPECS
    assert unfilled_of(real.doc, reason="skeleton_changed") == []
    labels = {u["table"] for u in real.doc["unfilled"] if u["kind"] == "table"} | {
        f["table"] for f in real.filled["filled"] if f["kind"] == "table"}
    assert labels <= set(fill.TABLE_SPECS)


def test_the_skeleton_is_never_modified_and_runs_are_idempotent_and_deterministic(real, tmp_path):
    assert skeleton_sha1() == real.before                     # the module run did not touch the skeleton
    again = fill.run(PAPER, real.res, real.out)                # same inputs, same output dir: nothing rewritten
    assert again["changed"] == []
    other = tmp_path / "filled2"
    fill.run(PAPER, real.res, other)                           # a fresh directory: byte-identical files
    files = sorted(p.relative_to(real.out) for p in real.out.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(other) for p in other.rglob("*") if p.is_file())
    for rel in files:
        assert (real.out / rel).read_bytes() == (other / rel).read_bytes(), rel
    assert skeleton_sha1() == real.before
    assert (real.out / "main.tex").read_bytes() == (PAPER / "main.tex").read_bytes()
    # no absolute path in the reports (they depend on the inputs only)
    for name in ("UNFILLED.json", "FILLED.json"):
        txt = read(real.out, name)
        for p in (str(ROOT), str(real.res), str(real.out)):
            assert p not in txt and json.dumps(p)[1:-1] not in txt
    with pytest.raises(SystemExit):
        fill.run(PAPER, real.res, PAPER / "sections")          # refuses to write into the skeleton


def test_branch_slots_are_never_touched(real):
    branch = []
    for f in fill.SECTION_FILES:
        text = (PAPER / "sections" / f"{f}.tex").read_text(encoding="utf-8")
        branch += [(f, text[a:b]) for a, b, inner in slot_spans(text) if inner.strip().startswith("branch:")]
    assert len(branch) == len(unfilled_of(real.doc, reason="branch_slot")) > 0
    for f, raw in branch:
        filled = read(real.out, f"sections/{f}.tex")
        assert filled.count(raw) == (PAPER / "sections" / f"{f}.tex").read_text(encoding="utf-8").count(raw), raw


def test_the_unfilled_report_lists_exactly_the_red_slots(real):
    n_red = 0
    for f in fill.SECTION_FILES:
        skel = (PAPER / "sections" / f"{f}.tex").read_text(encoding="utf-8")
        filled = read(real.out, f"sections/{f}.tex")
        red = slot_spans(filled)
        n_red += len(red)
        listed = [u for u in real.doc["unfilled"] if u["file"] == f"sections/{f}.tex"]
        assert len(listed) == len(red)
        assert [u["slot"] for u in listed] == [inner for _, _, inner in red]   # same slots, same order
        lines = skel.split("\n")
        for u in listed:
            assert "\\DATANEEDED{" + u["slot"].split("\n")[0] in lines[u["line"] - 1]
            assert u["reason"] in fill.REASONS and u["detail"]
    assert n_red == real.doc["summary"]["unfilled"]
    # every filled slot names the result-file fields it was read from
    assert all(f["from"] for f in real.filled["filled"])
    assert all(k in real.doc["sources"] for f in real.filled["filled"] for k in
               {x.split(":")[0] for x in f["from"]})


# ================================================================================================ real files
def test_real_gate_files_fill_the_gate_table(real):
    """tab:gate-outcomes (in the protocol since the editor pass of 2026-10-05): the bars sit in the row labels, the G5/G6 rows
    read the selection and confirmation files, the Gate-FT rows gate_ft.json; the zero-shot UAUC, the paired gain and the
    references on the Gate-FT rows are printed once, in tab:tracks, so this table has no grid cell."""
    sel, gate, gft = real.j["sel/selection.json"], real.j["gate/gate.json"], real.j["gft/gate_ft.json"]
    text = read(real.out, "sections/observation.tex")
    t = table_text(text, "tab:gate-outcomes")
    vs = sel["v_star"]
    assert row_cells(t, r"G5 DEV: ML-1M UAUC (bound $>0$)") == [n3(sel["table"]["V0"]["ml1m"]["UAUC"]),
                                                                n3(sel["table"][vs]["ml1m"]["UAUC"]),
                                                                sel["decision"].replace("_", "\\_")]
    assert row_cells(t, "G5 DEV: Toys UAUC (E2)") == [n3(sel["table"]["V0"]["toys"]["UAUC"]),
                                                      n3(sel["table"][vs]["toys"]["UAUC"]),
                                                      "yes" if sel["table"][vs]["E2"] else "no"]
    ci = gate["ci95"]
    v0 = gate["v0_context"]
    assert row_cells(t, r"G6 CONFIRM: ML-1M UAUC ($\ge0.60$)") == [cell(v0["UAUC"], v0["ci95"]["lo"], v0["ci95"]["hi"]),
                                                                   cell(gate["UAUC"], ci["lo"], ci["hi"]),
                                                                   gate["decision"].replace("_", "\\_")]
    assert row_cells(t, "G9 Gate-FT: seeds 0 / 1 / 2") == ["--", " / ".join(n3(x) for x in gft["UAUC_post_T_per_seed"]), ""]
    m = gft["UAUC_post_T_seed_averaged_ci95"]
    assert row_cells(t, r"G9 Gate-FT: mean ($\ge0.65$)") == [
        "--", cell(gft["UAUC_post_T_mean_over_seeds"], m["lo"], m["hi"], sd1(gft["UAUC_post_T_per_seed"])),
        gft["decision"].replace("_", "\\_")]
    # gate_ft.json's own zero_shot_context is never printed (a second scoring of the same prompts; FILLED.json notes)
    assert n3(gft["zero_shot_context"]["UAUC_post_T"]) not in t
    assert f"CONFIRM: {gate['n_users']:,} fresh ML-1M" in text.replace("\n", " ")
    assert unfilled_of(real.doc, table="tab:gate-outcomes") == []
    note = json.loads(read(real.out, "FILLED.json"))["notes"][0]
    assert note["not_printed"] == {"gft/gate_ft.json:zero_shot_context.UAUC_post_T": gft["zero_shot_context"]["UAUC_post_T"],
                                   "gft/gate_ft.json:zero_shot_context.dUAUC_finetuned_minus_zeroshot_post_T.est":
                                       gft["zero_shot_context"]["dUAUC_finetuned_minus_zeroshot_post_T"]["est"]}
    assert note["slots"] == [] and "second scoring" in note["note"] and "tab:tracks" in note["note"]
    # the protocol states the gate outcomes in words and prints no gate number of its own
    flat = " ".join(text.split())
    assert "found a fix (FIX\\_FOUND)" in flat and "passed (GATE\\_FT\\_PASS)" in flat
    assert n3(gate["UAUC"]) not in flat.split(r"\begin{table}")[0]


def test_resolved_gate_branches_are_filled_from_the_gate_files_and_refused_for_any_other_decision(real, tmp_path):
    gate, gft = real.j["gate/gate.json"], real.j["gft/gate_ft.json"]
    ci, est, v0 = gate["ci95"], n3(gate["UAUC"]), n3(gate["v0_context"]["UAUC"])
    mean = n3(gft["UAUC_post_T_mean_over_seeds"])
    seeds = ", ".join(n3(x) for x in gft["UAUC_post_T_per_seed"])

    def flat(out, f):
        return " ".join(read(out, f"sections/{f}.tex").split())
    intro = flat(real.out, "introduction")
    # the introduction as rewritten on 2026-10-04 (anchors of PROSE_SPECS updated by the editor pass of 2026-10-05)
    assert (f"4 stars or higher; on {gate['n_users']:,} untouched ML-1M users it reached UAUC {est} "
            f"(95\\% CI {n3(ci['lo'])}--{n3(ci['hi'])}) against {v0} for the registered prompt, so the bar (a point "
            "estimate) was met") in intro
    assert f"It did (mean {mean}; seeds {seeds})" in intro
    assert (f"the remedied zero-shot prompt reached UAUC {est} on untouched users (bar 0.60) and LoRA tuning reached "
            f"{mean} (bar 0.65)") in flat(real.out, "conclusion")
    # any other registered decision leaves them red as branch_not_taken (the branch that occurred is not the one written)
    res = tmp_path / "results"
    shutil.copytree(real.res, res)
    for rel, dec in (("gate/gate.json", "GATE_FAIL_AFTER_REMEDY"), ("gft/gate_ft.json", "GATE_FT_FAIL")):
        d = json.loads((res / rel).read_text(encoding="utf-8"))
        d["decision"] = dec
        (res / rel).write_text(json.dumps(d), encoding="utf-8")
    out = tmp_path / "filled"
    fill.run(PAPER, res, out)
    doc = json.loads(read(out, "UNFILLED.json"))
    wanted = {"gate:UAUC", "gate:UAUC, ci95", "gate:n_users", "gate:v0_context.UAUC", "gft:UAUC_post_T_mean_over_seeds",
              "gft:UAUC_post_T_per_seed"}
    reds = [u for u in doc["unfilled"] if fill.norm(u["slot"]) in wanted and u["kind"] == "prose"
            and u["file"] in ("sections/introduction.tex", "sections/conclusion.tex")]
    assert len(reds) == 5 + 2 and all(u["reason"] == "branch_not_taken" for u in reds)


SENS_TEX = r"""
\begin{table}[!htbp]
\caption{Prompt sensitivity (G5).}
\label{tab:app-sens}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Variant & ML-1M & Toys & E1 & E2 & Eligible & Tied\\
\midrule
""" + "".join(rf"{v} & \DATANEEDED{{sel:UAUC}} & \DATANEEDED{{sel:UAUC}} & \DATANEEDED{{sel:E1}} & \DATANEEDED{{sel:E2}} & "
              rf"\DATANEEDED{{sel:eligible}} & \DATANEEDED{{sel:tied\_with\_max}}\\" + "\n"
              for v in ("V0", "V1", "V2", "V3", "V4", "V5", "V7")) + r"""\bottomrule
\end{tabular}
\end{table}
"""


def test_real_selection_fills_the_sensitivity_table_when_it_is_restored(real, tmp_path):
    """The variant-bank and sensitivity tables were cut to two sentences (editor pass 2026-10-05); the tab:app-sens spec is
    kept dormant so that the table can be restored, or rendered for the artefact, unchanged."""
    assert "tab:app-sens" not in (PAPER / "sections" / "appendix.tex").read_text(encoding="utf-8")
    assert "tab:app-sens" in fill.TABLE_SPECS
    paper = copy_skeleton(tmp_path / "paper")
    p = paper / "sections" / "appendix.tex"
    p.write_text(p.read_text(encoding="utf-8") + SENS_TEX, encoding="utf-8")
    fill.run(paper, real.res, tmp_path / "filled")
    sel = real.j["sel/selection.json"]
    t = table_text(read(tmp_path / "filled", "sections/appendix.tex"), "tab:app-sens")
    for v in ("V0", "V1", "V2", "V3", "V4", "V5", "V7"):
        row = sel["table"][v]
        yn = {True: "yes", False: "no"}
        assert row_cells(t, v) == [n3(row["ml1m"]["UAUC"]), n3(row["toys"]["UAUC"]),
                                   yn[row["ml1m"]["E1"] and row["toys"]["E1"]], yn[row["E2"]], yn[row["eligible"]],
                                   yn[row["tied_with_max"]]]
    doc = json.loads(read(tmp_path / "filled", "UNFILLED.json"))
    assert unfilled_of(doc, table="tab:app-sens") == []


def test_real_sports_audit_fills_its_columns_and_leaves_the_other_domains_red(real):
    aud = real.j["aud/sports.json"]
    text = read(real.out, "sections/experiments.tex")
    t = table_text(text, "tab:anatomy")
    segs = [aud["segments"]["events_1_1000"]["questions"]["next"], aud["segments"]["events_1001_10000"]["questions"]["next"]]
    T = aud["temperature"]["next"]["T"]
    assert row_cells(t, "Temperature $T$")[:2] == [n3(T), n3(T)]
    for label, path in ((r"ECE (top label)", ("C_calibration", "list_normalised", "ece")),
                        (r"$\mathrm{acc}_{\rm high}$", ("C_calibration", "error_anatomy", "tertiles", 2, "accuracy")),
                        (r"AUROC($p_{\max}$)", ("C_calibration", "auroc_discrimination", "top1", "auroc", "p_max")),
                        (r"NDCG@10", ("A_ranking", "ranking", "ndcg10")),
                        (r"Bias Index, head $-$ tail", ("E_popularity", "bias_index", "head_minus_tail"))):
        cells = row_cells(t, label)
        for k, q in enumerate(segs):
            r = q
            for p in path:
                r = r[p]
            assert cells[k] == cell(r["est"], r["lo"], r["hi"]), label
        assert all(c.startswith("\\DATANEEDED{") for c in cells[2:]), label
    gain = row_cells(t, r"NDCG@10 gain, $p_{\max}$")                   # block C: estimates only
    assert gain[:2] == [n3(q["D_selective_serving"]["signals"]["p_max"]["gain_at_50_vs_full"]["ndcg10"]["est"])
                        for q in segs]
    ex = row_cells(table_text(text, "tab:exposure"), r"LLM, \textsf{next}")
    b = [q["B_exposure"]["llm"] for q in segs]
    assert ex[0:2] == [cell(x["delta_head"]["est"], x["delta_head"]["lo"], x["delta_head"]["hi"]) for x in b]
    assert ex[6:8] == [n3(x["gini_exposure"]["est"]) for x in b]          # Gini block: estimates only
    assert ex[12:14] == [n3(x["tail_share_top10"]["est"]) for x in b]     # APLT block: estimates only


def test_exposure_table_summarises_the_eight_published_recommenders(real):
    """tab:exposure (editor pass 2026-10-05): the verbalised reranker keeps its own row; the eight published recommenders are
    summarised per cell by the minimum, median and maximum of their estimates (estimates only, a source recorded for each)."""
    aud = real.j["aud/sports.json"]
    assert len(fill.PUBLISHED) == 8 and "ccrp_v3" not in fill.PUBLISHED
    assert set(fill.PUBLISHED) | {"ccrp_v3"} == set(fill.REF_METHODS.values())
    assert fill.REF_METHODS[r"Verbalised reranker"] == "ccrp_v3"
    text = read(real.out, "sections/experiments.tex")
    t = table_text(text, "tab:exposure")
    assert "C-CRP" not in text and "C-CRP" not in t
    segs = [aud["segments"][s] for s in ("events_1_1000", "events_1001_10000")]
    for blk, key, off in (("dh", "delta_head", 0), ("gini", "gini_exposure", 6), ("aplt", "tail_share_top10", 12)):
        rr = row_cells(t, "Verbalised reranker")
        ref = [seg["reference"]["ccrp_v3"]["B_exposure"][key] for seg in segs]
        assert rr[off:off + 2] == ([cell(r["est"], r["lo"], r["hi"]) for r in ref] if blk == "dh"
                                   else [n3(r["est"]) for r in ref]), blk
        for label, fn in (("Published, min", np.min), ("Published, median", np.median), ("Published, max", np.max)):
            want = [n3(float(fn([seg["reference"][m]["B_exposure"][key]["est"] for m in fill.PUBLISHED]))) for seg in segs]
            assert row_cells(t, label)[off:off + 2] == want, (label, blk)
    ceq = json.loads(read(real.out, "CHECK_EQUAL.json"))
    assert ceq["summary"]["numbers_without_a_source"] == 0
    # the other domains wait for their audit files
    red = [u for u in unfilled_of(real.doc, table="tab:exposure") if u["row"].startswith("Published")]
    assert red and {u["reason"] for u in red} == {"result_file_missing"}
    red = unfilled_of(real.doc, table="tab:anatomy")
    assert len(red) == 17 * 3 and {(u["reason"], u["detail"]) for u in red} == {
        ("result_file_missing", "aud/toys.json"), ("result_file_missing", "aud/home.json"),
        ("result_file_missing", "aud/tools.json")}


@pytest.fixture(scope="module")
def real_grid(real, tmp_path_factory):
    """The real files plus the real Qwen ML-1M grid report (skipped until it is pulled)."""
    if not (RESULTS / "grid/qwen/ml1m.json").is_file():
        pytest.skip("the Qwen ML-1M grid report is not pulled yet")
    base = tmp_path_factory.mktemp("real_grid")
    res = base / "results"
    shutil.copytree(real.res, res)
    shutil.copyfile(RESULTS / "grid/qwen/ml1m.json", res / "grid/qwen/ml1m.json")
    out = base / "filled"
    fill.run(PAPER, res, out)
    return SimpleNamespace(out=out, doc=json.loads(read(out, "UNFILLED.json")),
                           rep=json.loads((res / "grid/qwen/ml1m.json").read_text(encoding="utf-8")), gft=real.j["gft/gate_ft.json"],
                           sel=real.j["sel/selection.json"], gate=real.j["gate/gate.json"])


def test_real_ml1m_grid_report_fills_its_columns_and_passes_the_checks(real_grid):
    rep, gft = real_grid.rep, real_grid.gft
    text = read(real_grid.out, "sections/experiments.tex")
    t = table_text(text, "tab:tracks")
    seeds = ("s0", "s1", "s2")
    if rep["E_A"]["ZS"].get("available", True):
        zs = rep["E_A"]["ZS"]["UAUC_TEST"]
        assert row_cells(t, r"UAUC of $\ell$")[0] == rec_cell(zs["per_model"]["zeroshot"])
        # the references of the Gate-FT rows are the ML-1M cells of tab:tracks A (printed once, zero-shot rows)
        assert row_cells(t, r"\quad item mean $m$ (ref.)")[0] == rec_cell(zs["references"]["q_hat"])
        assert row_cells(t, r"\quad temporal MF (ref.)")[0] == rec_cell(zs["references"]["mf"])
        # deviation row 7 prints the DEV, CONFIRM and TEST zero-shot UAUCs as estimates
        sel, gate = real_grid.sel, real_grid.gate
        dev = table_text(read(real_grid.out, "sections/appendix.tex"), "tab:deviations")
        row7 = row_cells(dev, "7")[0]
        assert (f"DEV {n3(sel['table'][sel['v_star']]['ml1m']['UAUC'])}, CONFIRM {n3(gate['UAUC'])}, zero-shot on TEST rows "
                f"{n3(zs['per_model']['zeroshot']['est'])}") in " ".join(row7.split())
    if rep["E_A"]["FT"].get("available", True) and rep["E_A"]["FT"].get("complete"):
        ft = rep["E_A"]["FT"]["UAUC_TEST"]
        assert row_cells(t, r"UAUC of $\ell$")[1] == rec_cell(ft["mean_over_seeds"],
                                                             sd=sd1([ft["per_model"][s]["est"] for s in seeds]))
        app = table_text(read(real_grid.out, "sections/appendix.tex"), "tab:app-seeds")
        assert row_cells(app, "ML-1M")[0] == " / ".join(n3(ft["per_model"][s]["est"]) for s in seeds)
    for c in real_grid.doc["checks"]:
        if c["check"].startswith("grid ml1m"):
            assert c["ok"] is (n3(c["a"]) == n3(c["b"])), c          # identical when printed, or flagged
    # decision (1): the zero-shot UAUC and E-B of the Gate-FT rows are the grid values; since the editor pass they are printed
    # in tab:tracks only (the gate table prints gate_ft.json's LoRA values), and the second scoring is recorded, not printed
    g = table_text(read(real_grid.out, "sections/observation.tex"), "tab:gate-outcomes")
    assert row_cells(g, r"G9 Gate-FT: mean ($\ge0.65$)")[0] == "--"
    if rep["E_A"]["ZS"].get("available", True) and rep["E_B"].get("complete"):
        note = json.loads(read(real_grid.out, "FILLED.json"))["notes"][0]
        zs_est = rep["E_A"]["ZS"]["UAUC_TEST"]["per_model"]["zeroshot"]["est"]
        assert note["printed_instead"]["grid/qwen/ml1m.json:E_A.ZS.UAUC_TEST.per_model.zeroshot.est"] == zs_est
        assert note["abs_difference"]["UAUC_zero_shot"] == abs(gft["zero_shot_context"]["UAUC_post_T"] - zs_est)
    # --check_equal: every quantity printed in two places agrees on its estimate; the Gate-FT seeds duplicate the grid seeds
    # of tab:app-seeds, and the zero-shot UAUC appears in tab:tracks, tab:corrections and tab:deviations
    ceq = json.loads(read(real_grid.out, "CHECK_EQUAL.json"))
    assert ceq["summary"]["estimates_differ"] == 0 and ceq["summary"]["numbers_without_a_source"] == 0
    groups = {grp["quantity"]: grp for grp in ceq["groups"]}
    for q, tabs in (("grid/qwen/ml1m.json:E_A.ZS.UAUC_TEST.per_model.zeroshot", {"tab:tracks", "tab:deviations"}),
                    ("grid/qwen/ml1m.json:E_A.FT.UAUC_TEST.per_model.s0", {"tab:gate-outcomes", "tab:app-seeds"})):
        if q in groups:
            assert tabs <= set(groups[q]["tables"]) and groups[q]["estimates_equal"], q
    p1 = rep["P1"]
    if p1.get("available") and p1["decision"]["verdict"] in ("P1_HOLDS", "NO_EVIDENCE"):
        word = {"P1_HOLDS": "holds", "NO_EVIDENCE": "does not hold"}[p1["decision"]["verdict"]]
        npos = sum(1 for s in seeds if p1["per_seed"][s]["est"] > 0)
        flat = text.replace("\n", " ")
        assert f"P1 {word} (mean" in flat and f"{npos} of 3 seeds positive" in flat
    # cross-domain sentences and counts wait for the other three panels
    for prefix in ("raises / leaves", "grid:E\\_B confirmed", "grid:G confirmed"):
        u = [x for x in real_grid.doc["unfilled"] if x["slot"].startswith(prefix)]
        assert u and {x["reason"] for x in u} == {"result_file_missing"}, prefix


def test_split_files_fill_the_user_counts(real):
    t = table_text(read(real.out, "sections/experiments.tex"), "tab:tracks")
    sp = {d: real.j[f"grid/qwen/{d}_split.json"] for d in fill.RATED}
    assert row_cells(t, r"Users: TEST / $S_d$") == [f"{sp[d]['eval']['users_both_classes_test']:,} / {sp[d]['sd']['users']:,}"
                                                    for d in fill.RATED]


# ================================================================================================ synthetic, real schemas
def test_tracks_cells_regimes_lora_sd_descriptive_and_uninterpretable(synth):
    """tab:tracks (the former tab:reliability, editor pass 2026-10-05)."""
    t = table_text(read(synth.out, "sections/experiments.tex"), "tab:tracks")
    m = synth.ml1m
    zs = m["E_A"]["ZS"]["UAUC_TEST"]["per_model"]["zeroshot"]
    ft = m["E_A"]["FT"]["UAUC_TEST"]
    uauc = row_cells(t, r"UAUC of $\ell$")
    assert uauc[0] == rec_cell(zs)
    assert uauc[1] == rec_cell(ft["mean_over_seeds"], sd=sd1([ft["per_model"][s]["est"] for s in ("s0", "s1", "s2")]))
    toys_zs = synth.toys["E_A"]["ZS"]["UAUC_TEST"]["per_model"]["zeroshot"]
    assert toys_zs["n_users"] < 150 and uauc[2] == rec_cell(toys_zs) and uauc[2].endswith(r"{\scriptsize\,(descriptive)}")
    assert uauc[5].startswith("\\DATANEEDED{")                          # games: seed s2 failed E1
    eb = m["E_B"]
    assert row_cells(t, r"\quad LoRA $-$ ZS (E-B)")[0] == rec_cell(
        eb["mean_over_seeds"], sd=sd1([eb["per_seed"][s]["est"] for s in ("s0", "s1", "s2")]))
    ref = m["E_A"]["ZS"]["UAUC_TEST"]["references"]
    assert row_cells(t, r"\quad item mean $m$ (ref.)")[0] == rec_cell(ref["q_hat"])
    assert row_cells(t, r"\quad temporal MF (ref.)")[0] == rec_cell(ref["mf"])
    ec = m["E_C"]["FT"]
    assert row_cells(t, "ECE (after Platt)")[1] == rec_cell(ec["mean_over_seeds"]["ECE"], sd=sd1(
        [ec["per_model"][s]["ECE"]["est"] for s in ("s0", "s1", "s2")]))
    assert row_cells(t, "Margin AUROC, oracle")[0] == rec_cell(m["E_C"]["ZS"]["per_model"]["zeroshot"]["AUROC_margin_correct"])
    for k, reg in ((0, "ZS"), (1, "FT")):
        sh = m["E_D"][reg]["shares"]
        rec = sh["per_model"]["zeroshot"] if reg == "ZS" else sh["mean_over_seeds"]
        want = ("uninterpretable" if rec["shares_reading"] == "uninterpretable" else
                rec_cell(rec["item_prior_share"], sd=None if reg == "ZS" else sd1(
                    [sh["per_model"][s]["item_prior_share"]["est"] for s in ("s0", "s1", "s2")])))
        assert row_cells(t, r"Item-prior share $\rho^2/r_8$")[k] == want, reg
    ig = m["E_D"]["FT"]["information_gain"]
    assert row_cells(t, r"$\mathcal G=\Delta$UAUC(M2$-$M1), E-D")[1] == rec_cell(
        ig["G_mean_over_seeds"], sd=sd1([ig["per_model"][s]["G"]["est"] for s in ("s0", "s1", "s2")]))
    # G_CF: one value per panel (no LLM feature; zero-shot rows), beside the pending within-user value of addendum 6
    assert row_cells(t, r"$\mathcal G_{\rm CF}$: E-D / E-W$^\dagger$")[0] == (
        rec_cell(m["E_D"]["ZS"]["information_gain"]["G_CF"]) + r" / \DATANEEDED{ext:E\_W.G\_CF\_wu}")
    p1 = m["P1"]
    assert row_cells(t, r"P1: $\mathcal G_{\rm FT}-\mathcal G_{\rm ZS}$")[0] == (
        " / ".join(n3(p1["per_seed"][s]["est"]) for s in ("s0", "s1", "s2")) + "; "
        + rec_cell(p1["mean_over_seeds"]) + ", $p$ " + f"{p1['mean_over_seeds']['p']:.3f}")
    games = [u for u in unfilled_of(synth.doc, table="tab:tracks") if u["column"] == 6 and not u["slot"].startswith("ext:")]
    assert games and {u["reason"] for u in games} == {"incomplete_regime"}
    # the Llama table follows tab:tracks with the same row specs (the synthetic Llama ML-1M report is the ML-1M world)
    la = table_text(read(synth.out, "sections/appendix.tex"), "tab:app-llama")
    assert row_cells(la, r"UAUC of $\ell$")[0] == rec_cell(zs)
    assert row_cells(la, r"Item-prior share $\rho^2/r_8$")[:2] == row_cells(t, r"Item-prior share $\rho^2/r_8$")[:2]


def test_seed_table_failed_integrity_and_the_permutation_control(synth):
    t = table_text(read(synth.out, "sections/appendix.tex"), "tab:app-seeds")
    pm = synth.ml1m["E_A"]["FT"]["UAUC_TEST"]["per_model"]
    assert row_cells(t, "ML-1M")[0] == " / ".join(n3(pm[s]["est"]) for s in ("s0", "s1", "s2"))
    gpm = synth.games["E_A"]["FT"]["UAUC_TEST"]["per_model"]
    assert synth.games["runs"]["s2"]["like"]["status"] == "FAILED_INTEGRITY"
    assert row_cells(t, "Video Games")[0] == f"{n3(gpm['s0']['est'])} / {n3(gpm['s1']['est'])} / FAILED\\_INTEGRITY"
    ftc = synth.ml1m["FT_C"]
    assert row_cells(t, "ML-1M, permuted labels (seeds 0 / 1)")[0] == " / ".join(
        n3(ftc[f"seed{k}"]["TEST"]["per_seed"][f"s{k}"]["UAUC_b"]) for k in (0, 1))
    assert row_cells(t, r"ML-1M, real $-$ permuted (seeds 0 / 1)")[1] == " / ".join(
        rec_cell(ftc[f"seed{k}"]["all_rows"]["per_seed"][f"s{k}"]) for k in (0, 1))


def test_pruning_table_mean_sd_negated_contrast_and_a_registered_cut(synth):
    t = table_text(read(synth.out, "sections/experiments.tex"), "tab:pruning")
    prn = synth.prn
    a0 = prn["arms"]["P0"]
    c10, c21, c20 = prn["contrasts"]["P1-P0"], prn["contrasts"]["P2-P1"], prn["contrasts"]["P2-P0"]
    assert prn["arms"]["P3"]["status"] == "NOT_RUN"
    assert row_cells(t, "P0 full data") == [cell(a0["UAUC_seed_averaged"], sd=a0["sd_seed"]),
                                            cell(-c10["est"], -c10["hi"], -c10["lo"]),
                                            str(sum(1 for x in c10["per_seed"].values() if x < 0)), "--"]
    assert row_cells(t, r"P1 random (class-matched)")[3] == cell(c10["est"], c10["lo"], c10["hi"])
    assert row_cells(t, r"P2 uncertainty-selected")[1:] == [cell(c21["est"], c21["lo"], c21["hi"]),
                                                          str(sum(1 for x in c21["per_seed"].values() if x > 0)),
                                                          cell(c20["est"], c20["lo"], c20["hi"])]
    assert row_cells(t, r"P3 prior-congruent") == ["not run"] * 4


def test_slot_table_kill_rule_reads_not_run(synth):
    t = table_text(read(synth.out, "sections/appendix.tex"), "tab:slot")
    assert synth.slot["state"] == "KILLED" and synth.slot["killed_after"] == "games"
    for label in ("Post-hoc stacking", "Prior-offset LoRA", r"Offset $-$ stacking"):
        assert row_cells(t, label)[3] == "not run"
    rep = json.loads((synth.res / "slot/ml1m.json").read_text(encoding="utf-8"))
    d = rep["slot"]["difference"]
    per = [d["per_seed"][f"seed{k}"]["est"] for k in range(3)]
    want = cell(d["mean_over_seeds"]["est"], d["mean_over_seeds"]["lo"], d["mean_over_seeds"]["hi"])
    assert row_cells(t, r"Offset $-$ stacking")[0] == want[:-1] + rf"\,({s3(min(per))} to {s3(max(per))})" + "}"


def test_knockout_rows_labels_seed_means_and_the_minimum_n_rule(synth):
    """The knockout rows of tab:teaches (block D; the former tab:popularity) and of tab:app-llama (Llama Toys)."""
    t = table_text(read(synth.out, "sections/experiments.tex"), "tab:teaches")
    ko = synth.toys["knockout"]
    an = ko["analyses"]
    hmt = an["zeroshot"]["delta"]["pseudo"]["head_minus_tail"]
    assert hmt["n_clusters"] < 150
    cells = row_cells(t, r"Head $-$ tail drop of $\ell$")
    assert cells[:2] == ["--", "--"]                                    # ML-1M has no store field
    assert cells[2] == cell(hmt["est"], hmt["lo"], hmt["hi"], desc=True)
    ests = [an[s]["delta"]["pseudo"]["head_minus_tail"]["est"] for s in ("s0", "s1", "s2")]
    assert cells[3] == cell(sum(ests) / 3, sd=sd1(ests), desc=True)
    plc = an["zeroshot"]["delta"]["placebo"]["head_minus_tail"]
    assert row_cells(t, "Placebo drop")[2] == cell(plc["est"], plc["lo"], plc["hi"], desc=True)
    assert row_cells(t, "Registered label")[2:4] == [
        ko["labels"]["zeroshot"]["label"], " / ".join(ko["labels"][s]["label"] for s in ("s0", "s1", "s2"))]
    # the games and sports reports have no knockout arms: their cells stay red with the report's own reason
    red = [u for u in unfilled_of(synth.doc, table="tab:teaches") if u["column"] in (5, 6, 7, 8) and u["slot"].startswith("ko:")]
    assert red and all(u["reason"] == "result_not_in_report" and "knockout is not available" in u["detail"] for u in red)
    # Llama Toys knockout in tab:app-llama (the synthetic Llama Toys report is the Toys world)
    la = table_text(read(synth.out, "sections/appendix.tex"), "tab:app-llama")
    assert row_cells(la, r"Knockout: head $-$ tail drop of $\ell$")[2:4] == cells[2:4]
    assert row_cells(la, "Knockout: registered label")[2:4] == row_cells(t, "Registered label")[2:4]


def test_stage3_corrections_and_second_backbone_tables(synth):
    a = read(synth.out, "sections/appendix.tex")
    assert "2nd rated: Toys; Sports: VALID events 1--1000." in a
    t = table_text(a, "tab:corrections")
    cr = synth.mir["criteria"]
    # cells after the row number: correction name, ML-1M, 2nd rated, Sports
    assert row_cells(t, "5")[1:3] == [rec_cell(cr[n]["dUAUC_mirror_minus_placebo"], n_key="n") for n in ("ml1m", "toys")]
    nd = synth.mir["next_item_no_loss"]["dNDCG@10_mirror_minus_raw"]["tie_exact"]
    assert row_cells(t, "7")[3] == cell(nd["est"], nd["lo"], nd["hi"])
    inv = synth.ml1m["E_A"]["ZS"]["invariance"]["per_model"]["zeroshot"]
    assert row_cells(t, "1")[1] == n3(inv["max_abs_dAUC_user"])
    # the Sports max_abs_dNDCG_event cell is dropped from the skeleton ("--"); while present it reads to_be_removed
    assert {u["reason"] for u in unfilled_of(synth.doc, table="tab:corrections", row="1", column=4)} <= {"to_be_removed"}
    z = table_text(a, "tab:app-z2")
    q = synth.docs["toys"]["segments"]["all"]["questions"]["next"]["B_exposure"]["llm"]["delta_head"]
    assert row_cells(z, r"$\Delta_{\rm head}$, \textsf{next}")[1] == " / ".join([cell(q["est"], q["lo"], q["hi"])] * 2)


def test_direction_words_follow_the_registered_tests(synth):
    text = read(synth.out, "sections/experiments.tex").replace("\n", " ")
    s1 = synth.summary["S1"]["acc_top_minus_bottom_tertile"]["next"]["holm"]
    assert all(s1["reject_holm"].values()) and set(s1["sign"].values()) == {1}
    assert r"On 4 of 4 next-item domains $\mathrm{acc}_{\rm high}$ is above $\mathrm{acc}_{\rm low}$" in text
    assert "so correct top-1 decisions are concentrated in the high tertile, and top-1 errors" in text
    s2 = synth.summary["S2"]["share_errors_in_top_minus_third"]["next"]["holm"]
    word = {1: "more often than", -1: "less often than"}[next(iter(set(s2["sign"].values())))]
    assert all(s2["reject_holm"].values()) and f"fall in the high tertile {word} one third" in text
    adm = synth.summary["S3"]["admission"]["llm_next"]
    assert adm["effect_claimed"] and r"$\Delta_{\rm head}$ is above 0 on 4 of 4 domains" in text
    assert "with the same sign on the second backbone" in text
    # addendum 7 wording: P2 is uncertainty-selected pruning, P1 class-matched random pruning
    assert "so uncertainty-selected pruning is better than class-matched random pruning" in text
    assert synth.prn["claim"]["label"] == "BEATS_RANDOM" and "noisy" not in text
    p1 = synth.ml1m["P1"]
    word = {"P1_HOLDS": "holds", "NO_EVIDENCE": "does not hold"}[p1["decision"]["verdict"]]
    npos = sum(1 for s in ("s0", "s1", "s2") if p1["per_seed"][s]["est"] > 0)
    assert f"P1 {word} (mean" in text and f"{npos} of 3 seeds positive" in text
    # the registered stage-3 label of the two-view contrast, verbatim (after GATE_PASS)
    assert f"two-view contrast is labelled {fill.tex(synth.mir['decision'])} (rows 5--7)" in text
    labels = {synth.toys["knockout"]["labels"][m]["label"] for m in ("zeroshot", "s0", "s1", "s2")}
    # Video Games has no knockout: the Qwen label is not written (it would need both domains)
    assert unfilled_of(synth.doc, slot="POSITIVE / NEGATIVE / NULL / INDETERMINATE")[0]["reason"] == "result_not_in_report"
    assert len(labels) == 1
    # E-B over four domains with one incomplete (games): no direction word
    u = [x for x in synth.doc["unfilled"] if x["slot"].startswith("raises / leaves")]
    assert u and u[0]["reason"] == "incomplete_regime"
    # the baselines: some contrasts are Holm-significant, some not -> no single word
    assert unfilled_of(synth.doc, slot="larger / similar / smaller")[0]["reason"] == "not_decided"


def _variant(synth, tmp_path, edit) -> dict:
    res = tmp_path / "results"
    shutil.copytree(synth.res, res)
    edit(res)
    out = tmp_path / "filled"
    fill.run(PAPER, res, out)
    return SimpleNamespace(text=read(out, "sections/experiments.tex").replace("\n", " "),
                           doc=json.loads(read(out, "UNFILLED.json")))


def test_mixed_holm_signs_leave_the_direction_red(synth, tmp_path):
    def edit(res):
        home = json.loads((res / "aud/home.json").read_text(encoding="utf-8"))
        c = home["segments"]["all"]["questions"]["next"]["C_calibration"]["error_anatomy"]["acc_top_minus_bottom_tertile"]
        c["est"], c["lo"], c["hi"] = -c["est"], -c["hi"], -c["lo"]
        (res / "aud/home.json").write_text(json.dumps(home), encoding="utf-8")
        docs = [json.loads((res / f"aud/{d}.json").read_text(encoding="utf-8")) for d in ("sports", "toys", "home", "tools")]
        (res / "aud/summary.json").write_text(json.dumps(synth.na.summarize(docs)), encoding="utf-8")
    v = _variant(synth, tmp_path, edit)
    for slot, anchor in (("k of 4", r"unsure when wrong (S1, S2).} On"),
                         ("above / indistinguishable from / below", r"domains $\mathrm{acc}_{\rm high}$ is"),
                         ("concentrated in the high tertile / spread evenly / concentrated in the low tertile",
                          "so correct top-1 decisions are")):
        u = [x for x in v.doc["unfilled"] if fill.norm(x["slot"]) == slot and x.get("anchor") == anchor]
        assert u and u[0]["reason"] == "not_decided", slot


def test_a_zero_shot_run_failing_integrity_reads_failed_integrity(synth, tmp_path):
    def edit(res):
        rep = json.loads((res / "grid/qwen/ml1m.json").read_text(encoding="utf-8"))
        for blk in ("E_A", "E_C", "E_D", "E_E"):     # what ftgrid_report writes when the zeroshot like arm is excluded
            rep[blk]["ZS"] = {"available": False, "reason": "no usable like arm for the ZS models ['zeroshot']"}
        rep["runs"]["zeroshot"]["like"]["status"] = "FAILED_INTEGRITY"
        (res / "grid/qwen/ml1m.json").write_text(json.dumps(rep), encoding="utf-8")
    _variant(synth, tmp_path, edit)
    t = table_text(read(tmp_path / "filled", "sections/experiments.tex"), "tab:tracks")
    for label in (r"UAUC of $\ell$", "ECE (after Platt)", r"Item-prior share $\rho^2/r_8$",
                  r"$\mathcal G=\Delta$UAUC(M2$-$M1), E-D"):
        assert row_cells(t, label)[0] == r"FAILED\_INTEGRITY", label
    ft = synth.ml1m["E_A"]["FT"]["UAUC_TEST"]
    assert row_cells(t, r"UAUC of $\ell$")[1] == rec_cell(ft["mean_over_seeds"], sd=sd1(
        [ft["per_model"][s]["est"] for s in ("s0", "s1", "s2")]))          # the LoRA column is unaffected
    assert row_cells(t, r"\quad item mean $m$ (ref.)")[0] == rec_cell(ft["references"]["q_hat"])   # FT rows
    # G_CF (one value per panel) falls back to the fine-tuned regime's rows when the zero-shot block is unavailable
    gcf = synth.ml1m["E_D"]["FT"]["information_gain"]["G_CF"]
    assert row_cells(t, r"$\mathcal G_{\rm CF}$: E-D / E-W$^\dagger$")[0].startswith(rec_cell(gcf) + " / ")


@pytest.mark.parametrize("p, rule, word", [(0.001, True, "raises"), (0.4, True, "leaves"), (0.001, False, "leaves")])
def test_regime_contrast_word_needs_holm_and_the_sigma_seed_rule(synth, tmp_path, p, rule, word):
    def edit(res):
        rep = json.loads((res / "grid/qwen/ml1m.json").read_text(encoding="utf-8"))
        rep["E_B"]["mean_over_seeds"]["p"] = p
        rep["E_B"]["seeds"]["sigma_seed_rule"] = rule
        for d in fill.RATED:
            (res / f"grid/qwen/{d}.json").write_text(json.dumps(rep), encoding="utf-8")
    v = _variant(synth, tmp_path, edit)
    # the unanimous word (c_rq4_ft) and the count of confirmed members (count rule, addendum 6 item 9.2)
    k = 4 if word == "raises" else 0
    assert f"Fine-tuning {word} UAUC (E-B, confirmed on {k} of 4 Qwen panels)" in v.text
    assert synth.ml1m["E_B"]["mean_over_seeds"]["est"] > 0


def _count_world(synth, conf_zs, conf_ft, g_first_sign=1.0, eb_p=0.001, eb_rule=True):
    """Four Qwen grid reports built from the synthetic ML-1M report, with planted E-D confirmations of G per regime, the sign
    of the first domain's G, and planted E-B raw p-values and sigma_seed rule."""
    def edit(res):
        for j, d in enumerate(fill.RATED):
            rep = copy.deepcopy(synth.ml1m)
            for reg, conf in (("ZS", conf_zs), ("FT", conf_ft)):
                rep["E_D"][reg]["holm_family_E_D"]["confirmed"]["G"] = bool(conf[j])
                g = (rep["E_D"][reg]["information_gain"]["per_model"]["zeroshot"]["G"] if reg == "ZS"
                     else rep["E_D"][reg]["information_gain"]["G_mean_over_seeds"])
                g["est"] = (abs(g["est"]) or 0.01) * (g_first_sign if j == 0 else 1.0)
                g["descriptive_min_n"] = False
            rep["E_B"]["mean_over_seeds"]["p"] = eb_p[j] if isinstance(eb_p, list) else eb_p
            rep["E_B"]["seeds"]["sigma_seed_rule"] = eb_rule
            (res / f"grid/qwen/{d}.json").write_text(json.dumps(rep), encoding="utf-8")
    return edit


def test_count_rule_reports_confirmed_members_per_regime(synth, tmp_path):
    """Addendum 6 item 9.2 (count slots added by the editor pass of 2026-10-05): G confirmed (E-D Holm family) in k of the four
    Qwen panels per regime, and the regime contrast E-B confirmed (Holm over the domains and the sigma_seed rule) in k of 4;
    confirmed members of both signs are not decided; the unanimous words of the existing decision functions are unchanged."""
    v = _variant(synth, tmp_path / "a", _count_world(synth, [1, 0, 0, 0], [1, 1, 1, 1]))
    assert ("$\\mathcal G$ is confirmed (E-D) on 1 of 4 zero-shot and 4 of 4 fine-tuned panels" in v.text)
    assert "Fine-tuning raises UAUC (E-B, confirmed on 4 of 4 Qwen panels)" in v.text
    v = _variant(synth, tmp_path / "b", _count_world(synth, [0, 0, 0, 0], [1, 1, 1, 1], g_first_sign=-1.0,
                                                     eb_p=[0.001, 0.4, 0.4, 0.4]))
    assert "is confirmed (E-D) on 0 of 4 zero-shot and" in v.text
    u = [x for x in v.doc["unfilled"] if fill.norm(x["slot"]) == "grid:G confirmed, k of 4 LoRA"]
    assert u and u[0]["reason"] == "not_decided"                        # a confirmed G of each sign: a written sentence
    assert "(E-B, confirmed on 1 of 4 Qwen panels)" in v.text            # Holm: only the domain with p = 0.001
    u = [x for x in v.doc["unfilled"] if x["slot"] == "raises / leaves / lowers"]
    assert u and u[0]["reason"] == "not_decided"                        # mixed: no single word


EXT_TABLE_SLOTS = {   # alias ext (A3-6; editor pass 2026-10-05): table -> the slot texts that pass 2 specifies against the schema
    "tab:tracks": {"ext:E_J.UAUC_q_hat_T", "ext:E_J.dUAUC_L_minus_q_hat", "ext:E_J.dUAUC_L_minus_mf", "ext:E_G.e_share",
                   "ext:E_G.item_share_mf", "ext:E_G.item_share_label", "ext:E_W.G_wu", "ext:E_W.G_CF_wu", "ext:E_W.P1_wu",
                   "ext:E_Cprime.share_correct_bottom", "ext:E_Cprime.share_errors_top", "ext:E_Cprime.AUROC_margin"},
    "tab:teaches": {"ext:E_F.G_LLM_given_CF", "ext:E_F.G_CF_given_LLM", "ext:E_H.sparse.G_prior", "ext:E_H.dense.G_prior",
                    "ext:E_H.unseen.G_prior", "ext:E_H.seen.G_prior", "ext:FT_C_reading.R", "ext:FT_C_reading.label"},
    "tab:app-llama": {"ext:E_J.dUAUC_L_minus_q_hat", "ext:E_G.e_share", "ext:E_W.G_wu", "ext:E_W.G_CF_wu", "ext:E_W.P1_wu"},
}
EXT_PROSE_SLOTS = {"ext:E_J.H_J_count", "ext:E_W.robust_count", "ext:E_W.P1_wu_verdict", "ext:FT_C_reading.wording",
                   "ext:E_F.H_F_count", "ext:E_H.H_S_count_ZS", "ext:E_H.H_S_count_FT"}


def test_ext_slots_stay_red_until_the_extra_analysis_exists(synth):
    """Every slot of the addendum-6 analysis (alias ext) has a spec that checks its slot text and keeps it red with
    result_file_missing until the file exists; nothing else is affected (the other cells of their rows are filled)."""
    ext = [u for u in synth.doc["unfilled"] if fill.norm(u["slot"]).startswith("ext:")]
    assert ext and {u["reason"] for u in ext} == {"result_file_missing"} and {u["detail"] for u in ext} == {fill.EXT_DETAIL}
    by_table: dict = {}
    for u in ext:
        if u["kind"] == "table":
            by_table.setdefault(u["table"], set()).add(fill.norm(u["slot"]))
    assert by_table == EXT_TABLE_SLOTS
    assert {fill.norm(u["slot"]) for u in ext if u["kind"] == "prose"} == EXT_PROSE_SLOTS
    assert len(ext) == 147 and fill.ALIASES[-1] == "ext"
    assert synth.doc["summary"]["unfilled_by_alias"]["ext"] == len(ext)
    t = table_text(read(synth.out, "sections/experiments.tex"), "tab:tracks")
    assert row_cells(t, r"$\mathcal G_{\rm CF}$: E-D / E-W$^\dagger$")[0].endswith(r" / \DATANEEDED{ext:E\_W.G\_CF\_wu}")
    assert not unfilled_of(synth.doc, reason="skeleton_changed")


def test_the_deviations_table_keeps_every_row_of_the_record():
    """tab:deviations condenses the 18 rows of the deviations record (none dropped); the introduction points to it."""
    text = (PAPER / "sections" / "appendix.tex").read_text(encoding="utf-8")
    body = table_text(text, "tab:deviations")
    nums = [int(m) for m in re.findall(r"(?m)^(\d+) & ", body)]
    assert nums == list(range(1, 19))
    record = (ROOT / "docs" / "sigir" / "DEVIATIONS.md").read_text(encoding="utf-8")
    assert len(re.findall(r"(?m)^\| (\d+) \|", record)) == 18
    assert r"\label{app:deviations}" in text
    assert r"Appendix~\ref{app:deviations}" in (PAPER / "sections" / "introduction.tex").read_text(encoding="utf-8")


def test_pruning_p3_minus_p0_is_filled_when_p3_runs(synth, tmp_path):
    prn = synth.tfp.run_analyze(synth.pw, n_boot=100)            # every arm: the five contrasts of ftprune.CONTRASTS
    assert "P3-P0" in prn["contrasts"] and prn["arms"]["P3"]["status"] == "RUN"
    v = _variant(synth, tmp_path, lambda res: _put(res, "prn/pruning_ml1m.json", prn))
    t = table_text(read(tmp_path / "filled", "sections/experiments.tex"), "tab:pruning")
    a3, c31, c30 = prn["arms"]["P3"], prn["contrasts"]["P3-P1"], prn["contrasts"]["P3-P0"]
    assert row_cells(t, r"P3 prior-congruent") == [
        cell(a3["UAUC_seed_averaged"], sd=a3["sd_seed"]), cell(c31["est"], c31["lo"], c31["hi"]),
        str(sum(1 for x in c31["per_seed"].values() if x > 0)), cell(c30["est"], c30["lo"], c30["hi"])]
    assert unfilled_of(v.doc, table="tab:pruning") == []


def test_check_equal_flags_a_quantity_printed_with_different_estimates(synth, tmp_path, capsys):
    # synthetic grid ML-1M report beside the real gate_ft.json: the LoRA seed mean (and the seeds) differ between
    # tab:gate-outcomes (gate_ft.json) and tab:tracks / tab:app-seeds (grid): exactly what --check_equal is for
    res = fill.main(["--paper", str(PAPER), "--results", str(synth.res), "--out", str(tmp_path / "f"), "--check_equal"])
    ceq = res["check_equal"]
    groups = {g["quantity"]: g for g in ceq["groups"]}
    g = groups["grid/qwen/ml1m.json:E_A.FT.UAUC_TEST.mean_over_seeds"]
    assert not g["estimates_equal"] and {"tab:gate-outcomes", "tab:tracks"} <= set(g["tables"])
    assert {o["source"] for o in g["occurrences"]} >= {"gft/gate_ft.json:UAUC_post_T_mean_over_seeds",
                                                       "grid/qwen/ml1m.json:E_A.FT.UAUC_TEST.mean_over_seeds"}
    assert ceq["summary"]["estimates_differ"] >= 1
    out = capsys.readouterr().out
    assert "[DIFFERENT ESTIMATES] grid/qwen/ml1m.json:E_A.FT.UAUC_TEST.mean_over_seeds" in out and "CHECK_EQUAL:" in out
    assert json.loads(read(tmp_path / "f", "CHECK_EQUAL.json")) == json.loads(json.dumps(ceq))
    # the zero-shot UAUC is one quantity wherever it is printed (decision 1; since the editor pass: tab:tracks, the
    # corrections table and deviation row 7, which prints the estimate only): never a difference of estimates
    q = "grid/qwen/ml1m.json:E_A.ZS.UAUC_TEST.per_model.zeroshot"
    assert groups[q]["estimates_equal"] and {"tab:tracks", "tab:corrections", "tab:deviations"} <= set(groups[q]["tables"])
    assert "tab:gate-outcomes" not in groups[q]["tables"]


def test_dropped_slots_read_to_be_removed_until_the_skeleton_drops_them(real, tmp_path):
    paper = copy_skeleton(tmp_path / "paper")
    p = paper / "sections" / "appendix.tex"
    txt = p.read_text(encoding="utf-8")
    old = r"\DATANEEDED{corr:max\_abs\_dAUC\_user} & --\\"
    if old not in txt:
        pytest.skip("tab:corrections row 1 is not in the layout this test reinstates")
    p.write_text(txt.replace(old, r"\DATANEEDED{corr:max\_abs\_dAUC\_user} & \DATANEEDED{corr:max\_abs\_dNDCG\_event}\\", 1),
                 encoding="utf-8")
    fill.run(paper, real.res, tmp_path / "filled")
    doc = json.loads(read(tmp_path / "filled", "UNFILLED.json"))
    u = unfilled_of(doc, table="tab:corrections", row="1", column=4)
    assert [x["reason"] for x in u] == ["to_be_removed"] and "decision 2026-10-04" in u[0]["detail"]
    assert doc["summary"]["unfilled_by_reason"]["to_be_removed"] == 1


def test_prose_matching_survives_a_removed_slot_with_the_same_text(real, tmp_path):
    paper = copy_skeleton(tmp_path / "paper")
    p = paper / "sections" / "experiments.tex"
    txt = p.read_text(encoding="utf-8")
    n = txt.count(r"\DATANEEDED{k of 4}")
    assert n >= 2
    p.write_text(txt.replace(r"\DATANEEDED{k of 4}", "all", 1), encoding="utf-8")      # drop the first one (RQ1)
    fill.run(paper, real.res, tmp_path / "filled")
    doc = json.loads(read(tmp_path / "filled", "UNFILLED.json"))
    assert unfilled_of(doc, reason="skeleton_changed") == []
    left = [u for u in doc["unfilled"] if u["slot"] == "k of 4"]
    assert len(left) == n - 1 and r"unsure when wrong (S1, S2).} On" not in {u["anchor"] for u in left}
    assert all(u["anchor"] for u in left)


# ================================================================================================ every kind of cell
KINDS_TEX = r"""
\begin{table}[!htbp]
\caption{Every kind of cell.}
\label{tab:kinds}
\begin{tabular}{@{}lcc@{}}
\toprule
 & A & B\\
\midrule
Estimate & \DATANEEDED{grid:UAUC} & --\\
LoRA mean & -- & \DATANEEDED{grid:UAUC}\\
Mean (s.d.) & \DATANEEDED{prn:UAUC\_mean\_sd} & --\\
Cut & \DATANEEDED{prn:UAUC\_mean\_sd} & --\\
Descriptive & \DATANEEDED{grid:UAUC} & --\\
Direction word & \DATANEEDED{prn:label} & --\\
Registered label & \DATANEEDED{ko:label} & --\\
Missing file & \DATANEEDED{aud2l:B.delta\_head} & --\\
% a comment & with an ampersand \\ and a row break
\bottomrule
\end{tabular}
\end{table}
Prose \DATANEEDED{branch: kept / dropped} and a free slot \DATANEEDED{one sentence}.
"""


def test_one_table_with_every_kind_of_cell(synth, tmp_path, monkeypatch):
    paper = tmp_path / "paper"
    (paper / "sections").mkdir(parents=True)
    shutil.copyfile(PAPER / "main.tex", paper / "main.tex")
    for f in fill.SECTION_FILES:
        (paper / "sections" / f"{f}.tex").write_text("" if f != "method" else KINDS_TEX, encoding="utf-8")
    rows = {
        "Estimate": lambda R, c, k, s: fill.fmt(fill.regime_val(R, "qwen", "ml1m", "ZS", "E_A", ("UAUC_TEST",))),
        "LoRA mean": lambda R, c, k, s: fill.fmt(fill.regime_val(R, "qwen", "ml1m", "FT", "E_A", ("UAUC_TEST",))),
        "Mean (s.d.)": lambda R, c, k, s: fill.prune_handler("P0 full data")(R, "UAUC", k, s),
        "Cut": lambda R, c, k, s: fill.prune_handler(r"P3 prior-congruent")(R, "UAUC", k, s),
        "Descriptive": lambda R, c, k, s: fill.fmt(fill.regime_val(R, "qwen", "toys", "ZS", "E_A", ("UAUC_TEST",))),
        "Direction word": lambda R, c, k, s: fill.c_prn(R, s),
        "Registered label": lambda R, c, k, s: fill.ko_handler("Registered label (per seed)")(R, ("toys", "ZS"), k,
                                                                                            "ko:label"),
        "Missing file": lambda R, c, k, s: fill.z2_handler(r"$\Delta_{\rm head}$, \textsf{next}")(
            R, "kitchen", k, "aud2q:B.delta_head / aud2l:B.delta_head"),
    }
    monkeypatch.setitem(fill.TABLE_SPECS, "tab:kinds",
                        fill.TableSpec("tab:kinds", 3, fill._cols({1: "A", 2: "B"}), rows))
    out = tmp_path / "filled"
    fill.run(paper, synth.res, out)
    t = table_text(read(out, "sections/method.tex"), "tab:kinds")
    ft = synth.ml1m["E_A"]["FT"]["UAUC_TEST"]
    a0 = synth.prn["arms"]["P0"]
    toys = synth.toys["E_A"]["ZS"]["UAUC_TEST"]["per_model"]["zeroshot"]
    assert row_cells(t, "Estimate")[0] == rec_cell(synth.ml1m["E_A"]["ZS"]["UAUC_TEST"]["per_model"]["zeroshot"])
    assert row_cells(t, "LoRA mean")[1] == rec_cell(ft["mean_over_seeds"], sd=sd1(
        [ft["per_model"][s]["est"] for s in ("s0", "s1", "s2")]))
    assert re.fullmatch(r"0\.\d{3}\{\\scriptsize\$\\pm\$\.\d{3}\\,\(\.\d{3}\)\}", row_cells(t, "LoRA mean")[1])
    assert row_cells(t, "Mean (s.d.)")[0] == cell(a0["UAUC_seed_averaged"], sd=a0["sd_seed"])
    assert row_cells(t, "Cut")[0] == "not run"
    assert row_cells(t, "Descriptive")[0] == rec_cell(toys) and "(descriptive)" in row_cells(t, "Descriptive")[0]
    assert row_cells(t, "Direction word")[0] == "better than"
    assert row_cells(t, "Registered label")[0] == synth.toys["knockout"]["labels"]["zeroshot"]["label"]
    assert row_cells(t, "Missing file")[0] == r"\DATANEEDED{aud2l:B.delta\_head}"
    doc = json.loads(read(out, "UNFILLED.json"))
    assert [(u["slot"], u["reason"]) for u in doc["unfilled"]] == [
        (r"aud2l:B.delta\_head", "result_file_missing"), ("branch: kept / dropped", "branch_slot"),
        ("one sentence", "free_text")]
    assert "Prose \\DATANEEDED{branch: kept / dropped} and a free slot \\DATANEEDED{one sentence}." in read(
        out, "sections/method.tex")


# ================================================================================================ guards and units
def test_a_changed_skeleton_is_refused_not_guessed(real, tmp_path):
    paper = copy_skeleton(tmp_path / "paper")
    p = paper / "sections" / "experiments.tex"
    txt = p.read_text(encoding="utf-8")
    # swap the slot texts of two anatomy rows: the positions no longer match their statistic
    txt = txt.replace(r"ECE (top label) & \DATANEEDED{aud:C.ece}", r"ECE (top label) & \DATANEEDED{aud:C.brier}", 1)
    txt = txt.replace("Brier & \\DATANEEDED{aud:C.brier}", "Brier score & \\DATANEEDED{aud:C.brier}", 1)
    p.write_text(txt, encoding="utf-8")
    fill.run(paper, real.res, tmp_path / "filled")
    doc = json.loads(read(tmp_path / "filled", "UNFILLED.json"))
    bad = unfilled_of(doc, reason="skeleton_changed")
    assert {(u["row"], u["column"]) for u in bad} >= {("ECE (top label)", 1), ("Brier score", 1), ("Brier score", 2)}
    assert all(u["table"] == "tab:anatomy" for u in bad)


def test_formatting_rules():
    V = fill.Val
    assert fill.fmt(V(est=0.61249, lo=0.598, hi=0.626)) == r"0.612{\scriptsize$\pm$.014}"
    assert fill.fmt(V(est=0.742, lo=0.723, hi=0.761, sd=0.00234)) == r"0.742{\scriptsize$\pm$.019\,(.002)}"
    assert fill.fmt(V(est=-0.0123, lo=-0.02, hi=-0.004)) == r"$-$0.012{\scriptsize$\pm$.008}"
    assert fill.fmt(V(est=-0.0004)) == "0.000"                            # rounds to zero: no sign
    assert fill.fmt(V(est=0.5, lo=0.4, hi=0.6, est_only=True)) == "0.500"
    assert fill.fmt(V(est=0.5, lo=0.4, hi=0.6, descriptive=True)) == r"0.500{\scriptsize$\pm$.100}{\scriptsize\,(descriptive)}"
    assert fill.fmt(V(est=0.742, lo=0.723, hi=0.761, sd=0.002), prose=True) == r"0.742$\pm$.019"
    assert fill.fmt(V(text="not run")) == "not run"
    assert fill.integer(1683) == "1,683" and fill.yesno(True) == "yes" and fill.tex("GATE_FT_PASS") == r"GATE\_FT\_PASS"
    assert fill.pval(0.0009995) == "0.001" and fill.pval(0.0001) == "$<$0.001"
    v = fill.val({"est": 0.6, "lo": 0.5, "hi": 0.7, "n_users": 149})
    assert v.descriptive and not fill.val({"est": 0.6, "n_users": 150}).descriptive
    with pytest.raises(fill.Missing) as e:
        fill.val({"est": None})
    assert e.value.reason == "field_null"
    assert fill.stdev([0.740006226084526, 0.7446659443846868, 0.7427260253041084]) == pytest.approx(0.0023407110198572154)


def test_holm_matches_the_report_code():
    from src.confrec import ftgrid_report as fr
    rng = np.random.default_rng(3)
    for _ in range(50):
        p = {f"d{i}": (float(rng.uniform(0, 0.2)) if rng.uniform() > 0.2 else None) for i in range(4)}
        assert fill.holm_adjust(p) == fr.holm(p)


def test_pull_script_and_fill_agree_on_the_result_layout():
    txt = PULL.read_text(encoding="utf-8")
    assert txt.isascii() and "BatchMode=yes" in txt and "MANIFEST.json" in txt
    code = "\n".join(ln for ln in txt.splitlines() if not ln.lstrip().startswith("#"))   # comments may say "no password"
    assert not re.search(r"(?i)password|passwd|-pw\b|sshpass|token|secret|\.pem\b|id_ed25519|id_rsa", code)
    pairs = re.findall(r'Add-F "(\w+)"\s+"([^"]+)"\s+"([^"]+)"', txt)
    loops = re.findall(r'foreach \(\$d in ([^)]+)\) \{(.*?)\n\}', txt, re.S)
    entries = []
    for alias, server, local in pairs:
        if "$d" in server or "${d}" in server:
            continue
        entries.append((alias, server, local))
    for doms, body in loops:
        for d in re.findall(r'"(\w+)"', doms):
            for alias, server, local in re.findall(r'Add-F "(\w+)"\s+"([^"]+)"\s+"([^"]+)"', body):
                entries.append((alias, server.replace("${d}", d).replace("$d", d),
                                local.replace("${d}", d).replace("$d", d)))
    local = {l for _, _, l in entries}
    # every file fill_paper reads is pulled to the path it reads it from
    need = {"sel/selection.json", "gate/gate.json", "gft/gate_ft.json", "aud/summary.json", "prn/pruning_ml1m.json",
            "slot/slot.json", "mir/decision.json",
            *(f"aud/{d}.json" for d in fill.NEXT), *(f"aud2q/{d}.json" for d in fill.NEXT),
            *(f"aud2l/{d}.json" for d in fill.NEXT), *(f"grid/qwen/{d}.json" for d in fill.RATED),
            *(f"grid/qwen/{d}_split.json" for d in fill.RATED), *(f"slot/{d}.json" for d in fill.RATED),
            *(f"grid/llama/{d}.json" for d in ("ml1m", "toys")), *(f"grid/llama/{d}_split.json" for d in ("ml1m", "toys"))}
    assert need <= local, need - local
    assert {a for a, _, _ in entries} <= set(fill.ALIASES)
    # the server is only listed (stat) and copied from (scp): no write, move or delete command is sent
    remote = re.findall(r'\$lines\.Add\("([^"]*)"\)', txt)
    assert remote and all(re.match(r"^(cd |if \[ -f )", r) for r in remote)
    assert not re.search(r"\b(rm|mv|cp|touch|tee|chmod|kill|nohup|bash scripts)\b", " ".join(remote))
