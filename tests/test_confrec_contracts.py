"""Cross-module and script contract tests of the amendment-2 integration audit.

1. Every CLI flag the shell scripts pass to a repo entry point exists in that entry point's REAL argparse
   (introspected, subcommand aware, required options supplied, literal values satisfy choices / type).
2. run_gatefix.sh hands the stage-0 manifest to both gatefix_select stages (panel identity of DEV and CONFIRM).
3. The registered Pilot-1 identity constants agree across gatefix_select, build_confirm_panels, run_diag_battery.sh and
   the amendment text.
4. The real pyes_scorer output (not a hand-written fixture) is accepted by gatefix_select (dev + confirm), by the
   diagnosis battery's provenance checks and by pilot1_gate's stage-3 reader.
"""
import argparse
import importlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.confrec import build_rated_panels as brp
from src.confrec import diag_battery as db
from src.confrec import gatefix_select as gs
from src.confrec import pyes_scorer as ps
from src.confrec.prompting import prompt_key

ROOT = Path(__file__).resolve().parents[1]
SIGIR = ROOT / "scripts" / "sigir"
_spec = importlib.util.spec_from_file_location("pilot1_gate", SIGIR / "pilot1_gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


# ---------------------------------------------------------------- 1. every script flag exists in the real argparse
class _Captured(Exception):
    pass


def _capture(fn):
    orig = (argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args)

    def boom(self, *a, **k):
        raise _Captured(self)
    argparse.ArgumentParser.parse_args = argparse.ArgumentParser.parse_known_args = boom
    old = sys.argv
    sys.argv = ["prog"]
    try:
        fn()
    except _Captured as c:
        return c.args[0]
    except (SystemExit, Exception):   # this entry point does not take these arguments: try the next one
        return None
    finally:
        argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args = orig
        sys.argv = old
    return None


def _parser(target):
    """The argparse parser of 'module:src.confrec.x' or 'script:scripts/sigir/y.py' (None: could not be loaded)."""
    kind, name = target.split(":", 1)
    try:
        if kind == "module":
            mod = importlib.import_module(name)
        else:
            spec = importlib.util.spec_from_file_location("audit_" + Path(name).stem, ROOT / name)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
    except ImportError:   # e.g. a download script whose third-party import is absent on this machine
        return None
    for call in (lambda: mod.main(), lambda: mod.parse_args(), lambda: mod.main([])):
        p = _capture(call) if hasattr(mod, "main") or hasattr(mod, "parse_args") else None
        if p is not None:
            return p
    return None


def _declared(parser, sub):
    """(option actions, subcommand names) of the parser, or of its subcommand `sub`; (None, None) when the parser has
    subcommands but not this one. A token after the entry point that is a positional with choices (panel_reference.py
    check|verify) is not a subcommand: its flags belong to the top-level parser."""
    p = parser
    subparsers = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    if sub and subparsers:
        p = subparsers[0].choices.get(sub)
        if p is None:
            return None, None
    acts = {s: a for a in p._actions for s in a.option_strings}
    subs = {n for a in p._actions if isinstance(a, argparse._SubParsersAction) for n in a.choices}
    return acts, subs


_FLAG = re.compile(r"(?<![\w-])(--[A-Za-z][A-Za-z0-9_]*)(?:[ =]+(\S+))?")


def _logical(text):
    for s in re.sub(r"\\\n\s*", " ", text).splitlines():
        s = s.strip()
        if s and not s.startswith("#"):
            yield s


def _arrays(text):
    return dict(re.findall(r"^([A-Za-z_][A-Za-z_0-9]*)=\(([^)]*)\)", re.sub(r"\\\n\s*", " ", text), re.M))


def _invocations(sh: Path):
    """(line, target, subcommand, [(flag, literal value)]) of every call of a repo entry point in a shell script."""
    text = sh.read_text(encoding="utf-8")
    arrs = _arrays(text)
    sd = re.search(r"score\(\) \{(.*?)\n\}", text, re.S)
    fixed = [x for ln in (sd.group(1).splitlines() if sd else []) if "pyes_scorer" in ln or ln.strip().startswith("--")
             for x in _FLAG.findall(ln)]
    for s in _logical(text):
        if re.match(r"^(score\(\)|local |fresh\(\)|step\(\)|fetch\(\)|score_like\(\)|[A-Za-z_]+=|echo)", s) \
                and not s.startswith("MK="):
            continue
        tgt = sub = None
        extra = []
        m = re.search(r"-m (src\.confrec\.[a-z_0-9]+)(?: ([a-z0-9_]+))?", s)
        if m:
            tgt = f"module:{m.group(1)}"
        else:
            m = re.search(r"(scripts/sigir/[a-z_0-9]+\.py)(?: ([a-z0-9_]+))?", s)
            if m and "$DATA/" not in s:
                tgt = f"script:{m.group(1)}"
        if m and m.group(2) and not m.group(2).startswith(("-", "$")):
            sub = m.group(2)
        if re.match(r"^(score|score_like)\b", s):
            tgt, extra = "module:src.confrec.pyes_scorer", list(fixed)
        for name in re.findall(r"\$\{([A-Za-z_0-9]+)\[@\]\}", s):
            if name == "MK":
                tgt, sub = "module:src.confrec.diag_battery", "make"
            elif name == "V0":
                tgt = tgt or "module:src.confrec.pyes_scorer"
            elif name == "FREEZE_ARGS":
                tgt, sub = "module:src.confrec.diag_battery", "freeze"
            extra += _FLAG.findall(arrs.get(name, ""))
        if tgt:
            yield s, tgt, sub, _FLAG.findall(s) + extra


@pytest.mark.parametrize("sh", sorted(SIGIR.glob("*.sh")), ids=lambda p: p.name)
def test_every_script_flag_exists_in_the_real_argparse(sh):
    problems, cache, n = [], {}, 0
    for line, tgt, sub, flags in _invocations(sh):
        if "MK=(" in line:     # the array definition itself: its flags are checked where it is expanded
            continue
        if tgt not in cache:
            cache[tgt] = _parser(tgt)
        if cache[tgt] is None:
            continue
        acts, subs = _declared(cache[tgt], sub)
        n += 1
        if acts is None:
            problems.append(f"{tgt} has no subcommand {sub!r}: {line[:100]}")
            continue
        if subs and sub is None:
            problems.append(f"{tgt} needs a subcommand {sorted(subs)}: {line[:100]}")
            continue
        used = {f for f, _ in flags}
        miss = used - set(acts) - {"--help"}
        if miss:
            problems.append(f"{tgt} {sub or ''} has no flag {sorted(miss)}: {line[:110]}")
        if not re.search(r"\$\{(?!V0|MK|FREEZE_ARGS)", line):
            req = [a.option_strings[0] for a in set(acts.values()) if a.option_strings and a.required]
            lacking = [r for r in req if r not in used]
            if lacking:
                problems.append(f"{tgt} {sub or ''} lacks required {lacking}: {line[:110]}")
        for f, v in flags:
            a = acts.get(f)
            if a is None or not v or v.startswith(("$", '"', "'", "-", "(")):
                continue
            if a.choices and v not in a.choices:
                problems.append(f"{tgt} {f} {v!r} not in {list(a.choices)}")
            if a.type in (int, float) and not re.fullmatch(r"-?\d+(\.\d+)?", v):
                problems.append(f"{tgt} {f} {v!r} is not a {a.type.__name__}")
    assert not problems, "\n".join(problems)


def test_the_flag_audit_itself_catches_a_wrong_flag(tmp_path):
    bad = tmp_path / "bad.sh"
    bad.write_text('"$PY" -m src.confrec.diag_battery freeze --bogus 1 --amendment a --dev_panels b --manifest c '
                   '--out d\n"$PY" -m src.confrec.gatefix_select confirm --vstar_dir a --v0_dir b --selection c '
                   '--outx d\n', encoding="utf-8")
    seen = []
    for line, tgt, sub, flags in _invocations(bad):
        acts, _ = _declared(_parser(tgt), sub)
        seen.append(sorted({f for f, _ in flags} - set(acts)))
    assert seen == [["--bogus"], ["--outx"]]


# ---------------------------------------------------------------- 2. the script hands the manifest to the selection
def _code(path):
    return "\n".join(ln.split("#", 1)[0] for ln in path.read_text(encoding="utf-8").splitlines())


def test_run_gatefix_passes_the_stage0_manifest_to_both_selection_stages():
    """Without --manifest gatefix_select cannot check that DEV and CONFIRM are the frozen panels (their user-id list
    sha1s are what the freeze records in PILOT_LOG): confirm would only warn 'panel identity is unchecked'."""
    code = re.sub(r"\\\n\s*", " ", _code(SIGIR / "run_gatefix.sh"))
    for stage in ("dev", "confirm"):
        calls = re.findall(r'-m src\.confrec\.gatefix_select %s [^\n]*' % stage, code)
        assert len(calls) == 1, stage
        assert '--manifest "$GP/manifest.json"' in calls[0], calls[0]
    assert re.search(r'^GP="\$G/panels"', code, re.M)


# ---------------------------------------------------------------- 3. registered identity constants agree
def test_registered_pilot1_constants_agree_across_modules():
    spec = importlib.util.spec_from_file_location("bcp_audit", SIGIR / "build_confirm_panels.py")
    bcp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bcp)
    diag = dict(re.findall(r"^(ML_SHA1|TOYS_SHA1)=([0-9a-f]{40})", (SIGIR / "run_diag_battery.sh").read_text(
        encoding="utf-8"), re.M))
    amend = (ROOT / "idea-stage" / "PREREG_AMENDMENT_2.md").read_text(encoding="utf-8")
    for src, key in (("ml1m", "ML_SHA1"), ("toys", "TOYS_SHA1")):
        sha = gs.PILOT1_DEV[src]["panel_sha1"]
        assert sha == bcp.PILOT_SHA1[src] == diag[key]
        assert sha in amend, f"the amendment text does not carry the {src} Pilot-1 panel sha1"
    assert bcp.PREFIX_LINES == gs.PILOT1_DEV["ml1m"]["n_users"] == gs.PILOT1_DEV["toys"]["n_users"] == 1500
    assert gs.PILOT1_DEV["ml1m"]["n_rows"] == 29365 and gs.PILOT1_DEV["toys"]["n_rows"] == 19022
    assert gs.UAUC_MIN == gate.UAUC_MIN == 0.60


# ---------------------------------------------------------------- 4. real scorer output into the consumers
class _Tok:
    """Character tokenizer; the chat template keeps a system message (V5 / V7) and honours enable_thinking."""
    def __init__(self):
        self.v, self.r = {}, {1: "Yes", 2: "No"}

    def __len__(self):
        return 3 + len(self.v)

    def decode(self, ids):
        return "".join(self.r.get(i, "") for i in ids)

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, enable_thinking=True, **kw):
        out = "".join(f"<{m['role'][0]}>{m['content']}</{m['role'][0]}>" for m in msg) + "<a>"
        return out + ("<think>\n\n</think>\n\n" if not enable_thinking else "")

    def __call__(self, text, add_special_tokens=False):
        for c in text:
            if c not in self.v:
                self.v[c] = len(self.v) + 3
                self.r[self.v[c]] = c
        return {"input_ids": [self.v[c] for c in text]}


def _noise(*parts):
    import hashlib
    h = int(hashlib.sha1("\x1f".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16 ** 12
    h2 = int(hashlib.sha1("\x1f".join(map(str, parts + ("b",))).encode()).hexdigest()[:12], 16) / 16 ** 12
    return math.sqrt(-2 * math.log(max(h, 1e-12))) * math.cos(2 * math.pi * h2)


def _score(panel, out, variant, signal):
    """The REAL pyes_scorer.run (variant bank, chat template, parts, report.json) against a planted fake model."""
    args = ps.parse_args(["--data", str(panel), "--output", str(out), "--model", "/models/Qwen3-8B", "--dtype",
                          "float16", "--topk_logprobs", "50", "--max_model_len", "4096", "--chunk_users", "7",
                          "--variant", variant, "--readout", "yesno", "--questions", "like"])
    rows = [json.loads(x) for x in open(panel, encoding="utf-8")]
    kind = "rated"
    hist = ps.resolve_hist_len(variant, kind, None)
    planted = {}
    for r in rows:
        y = ps.labels_of(r)
        for i, q, p in ps.record_requests(r, ["like"], hist, variant, kind, "yesno"):
            planted[prompt_key(p)] = signal * (2 * y[i] - 1) + _noise(r["user_id"], i, variant)

    def load(_a):
        tok = _Tok()

        class LLM:
            def generate(self, prompts, sp, use_tqdm=False, **kw):
                out_ = []
                for pr in prompts:
                    t = tok.decode(pr["prompt_token_ids"])
                    assert t.endswith("<think>\n\n</think>\n\n"), "thinking must be off (G0)"
                    t = t[: -len("<think>\n\n</think>\n\n")]
                    m = re.match(r"^(?:<s>(.*?)</s>)?<u>(.*)</u><a>$", t, re.S)
                    key = m.group(2) if m.group(1) is None else m.group(1) + "\x1d" + m.group(2)
                    lg = planted[key]
                    py = 0.99999 / (1 + math.exp(-lg))
                    top = {-k: SimpleNamespace(logprob=math.log(1e-7) - k / 100) for k in range(1, 49)}
                    top[1] = SimpleNamespace(logprob=math.log(py))
                    top[2] = SimpleNamespace(logprob=math.log(0.99999 - py))
                    out_.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[top])]))
                return out_
        return SimpleNamespace(llm=LLM(), tok=tok, gen_kw={}, to_prompt=lambda ids: {"prompt_token_ids": ids},
                               sp=None, version="fake")
    ps.run(args, load_model=load)
    return Path(out)


def _ml1m_raw(d, n_users=60, n_movies=70, seed=1):
    import random
    r = random.Random(seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "movies.dat").write_text("".join(f"{m}::Movie {m} ({1990 + m % 9})::Drama|Comedy\n"
                                          for m in range(1, n_movies + 1)), encoding="latin-1")
    q = {m: r.gauss(0, 1) for m in range(1, n_movies + 1)}
    lines = []
    for u in range(1, n_users + 1):
        t = 978300000 + u * 10 ** 5
        for m in r.sample(range(1, n_movies + 1), 40):
            t += r.randint(1, 900)
            lines.append(f"{u}::{m}::{min(5, max(1, round(3 + 1.2 * q[m] + r.gauss(0, 1.3))))}::{t}\n")
    (d / "ratings.dat").write_text("".join(lines), encoding="latin-1")


def _h20_panels(tmp_path):
    """DEV 'ml1m' (users m*), DEV 'toys' (the same rows, users t*) and CONFIRM (fresh users c*), hist_len 20 with the
    amendment-2 fields (history_meta, domain_kind), as build_confirm_panels writes them."""
    _ml1m_raw(tmp_path / "raw")
    items, events = brp.load_ml1m(tmp_path / "raw")
    rows, _ = brp.build(items, events, n_users=10 ** 6, n_cands=10, hist_len=20, min_hist=3, min_like=2,
                        min_dislike=2, seed=0, source="ml1m")
    assert len(rows) >= 50 and max(len(r["history"]) for r in rows) > 10
    dev, conf = rows[:30], rows[30:50]

    def write(path, rs, prefix):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                r = dict(r, user_id=prefix + str(r["user_id"]))
                r["source_event_id"] = prefix + r["source_event_id"]
                f.write(brp.row_line(r))
        return path
    return (write(tmp_path / "ml1m_dev_h20.jsonl", dev, "m"), write(tmp_path / "toys_dev_h20.jsonl", dev, "t"),
            write(tmp_path / "ml1m_confirm_h20.jsonl", conf, "c"))


def test_gatefix_select_and_the_battery_accept_the_real_scorer_output(tmp_path, monkeypatch, capsys):
    ml, toys, conf = _h20_panels(tmp_path)
    sig = {v: 0.2 for v in gs.VARIANTS}
    sig["V3"] = 1.6                                           # the planted fix
    root = tmp_path / "dev"
    for name, panel in (("ml1m", ml), ("toys", toys)):
        for v in gs.VARIANTS:
            d = _score(panel, root / name / v, v, sig[v])
            run = gs.load_run(d)
            assert gs.check_run(run, v, f"{name}/{v}") == []   # report.json / scores as gatefix_select requires them
            row = gs.panel_row(run)
            assert row["E1"] is True and math.isfinite(row["UAUC"]) and row["n_users"] == 30
            assert row["hist_len"] == (20 if v in ("V3", "V7") else 10)
            assert (row["system_sha1"] is not None) == (v in ("V5", "V7"))
    rep = json.loads((root / "ml1m" / "V0" / "report.json").read_text(encoding="utf-8"))
    assert db.arm_settings_errors("base", rep) == []           # the battery's provenance reads the same keys
    monkeypatch.setattr(gs, "PILOT1_DEV", {p: dict(gs.PILOT1_DEV[p], user_ids_sha1=gs.panel_dev_signature(pp)[
        "user_ids_sha1"]) for p, pp in (("ml1m", ml), ("toys", toys))})   # the synthetic DEV users stand in for Pilot 1
    sel_path = tmp_path / "selection.json"
    man = {"sources": {p: {"dev": {"built": True, "sha1": gs._sha1_file(pp),
                                   "user_ids_sha1": gs.panel_dev_signature(pp)["user_ids_sha1"], "path": pp.name}}
                       for p, pp in (("ml1m", ml), ("toys", toys))}}
    man["sources"]["ml1m"]["confirm"] = {"built": True, "sha1": gs._sha1_file(conf), "path": conf.name,
                                         "user_ids_sha1": gs.panel_dev_signature(conf)["user_ids_sha1"]}
    (tmp_path / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    code = gs.main(["dev", "--root", str(root), "--variants", ",".join(gs.VARIANTS), "--out", str(sel_path),
                    "--manifest", str(tmp_path / "manifest.json")])
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    assert code == 0 and sel["decision"] == gs.FIX_FOUND and sel["v_star"] == "V3"
    assert db.selected_variant(sel) == "V3"                    # run_gatefix.sh's `diag_battery vstar`
    assert sel["confirm_panel_expected"]["user_ids_sha1"] == man["sources"]["ml1m"]["confirm"]["user_ids_sha1"]
    # stage 2 on the fresh users, then the stage-3 reader of pilot1_gate
    cdir = tmp_path / "confirm" / "ml1m"
    _score(conf, cdir / "V3", "V3", 1.6)
    _score(conf, cdir / "V0", "V0", 0.2)
    g_path = tmp_path / "gate.json"
    code = gs.main(["confirm", "--vstar_dir", str(cdir / "V3"), "--v0_dir", str(cdir / "V0"), "--selection",
                    str(sel_path), "--out", str(g_path), "--manifest", str(tmp_path / "manifest.json")])
    g6 = json.loads(g_path.read_text(encoding="utf-8"))
    assert code == 0 and g6["decision"] == gs.GATE_PASS and g6["confirm_panel_identity"]["checked"] is True
    assert "panel identity is unchecked" not in capsys.readouterr().out
    checks = {k: {"ok": True} for k in ("ml1m", "toys", "sports")}
    p1 = {"gate": {"sports_raw_NDCG@10": 0.2093, "input_checks": {"sports": {"ok": True}}}}
    s3 = gate.stage3_gate(g6, p1, {"n_users": g6["n_users"]}, {}, checks)
    assert s3["rated_component"]["ok"] is True and s3["rated_component"]["v_star"] == "V3"
    assert s3["next_item_component"]["ok"] is True and s3["ml1m_n_users_equals_g6"] is True and s3["pass"] is True
