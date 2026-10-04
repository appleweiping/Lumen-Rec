"""Tests of scripts/sigir/run_gatefix_stage3.sh and src/confrec/gatefix_stage3.py: idea-stage/PREREG_AMENDMENT_2.md G7,
stage 3, which runs only on the G6 GATE_PASS. CPU only, deterministic, no network, no GPU, no model.

1. The script file: LF, bash shebang, `set -euo pipefail`, `bash -n` clean; the header logs the GPU-h estimate; every
   registered arm is requested exactly once with the flags G7 / G0 fix; the refusals precede every write and scorer call;
   the freeze check is run_gatefix.sh's own.
2. Every flag the script passes exists in the real argparse of its entry points (pyes_scorer, pilot_mirror, pilot1_gate,
   gatefix_stage3, diag_battery, build_rated_panels, slim_amazon2023): the integration audit of
   tests/test_confrec_contracts.py, and a stricter audit here that also reads `if ! X=$(...)` substitutions, expands
   arrays and score()'s fixed flags, checks literal values, and catches a wrong flag and a wrong value.
3. The input check (src.confrec.gatefix_stage3) in-process: every refusal reason, the G7 second-domain rule and its
   1,500-row cut, the 1,000-event sports VALID cut and the quarantine of TEST events 1-1000, a record that a rerun leaves
   untouched.
4. The refusal paths of the real script in a minimal repo copy (a real FREEZE.txt from diag_battery.freeze and its
   PILOT_LOG record): no gate.json, gate.json not GATE_PASS, the freeze check failing (record missing, amendment changed
   after the freeze): exit 2, zero scorer calls, nothing written.
5. End-to-end DRY RUN in the integration sandbox (scripts/sigir/dryrun, the integration agent's fake scorer): the real
   run_gatefix.sh stages 0-2 (run_gf.sh, 2,200 synthetic Amazon users so that Toys has a CONFIRM split: V3 fix found,
   GATE_PASS), then the real run_gatefix_stage3.sh (run_s3.sh): fresh Toys -> stage3/decision.json; a second run skips
   everything and touches no file; the forced Video_Games fallback (S3_MIN_FRESH) builds its panel and reuses the other
   scored panels. About 15 minutes on Windows (Git Bash and Python start-up dominate), a few on Linux.
Sections 4-5 need bash (Git Bash on Windows; WSL's bash.exe is not used) and are skipped with that reason otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path

import pytest

from src.confrec import build_rated_panels as brp
from src.confrec import diag_battery as db
from src.confrec import gatefix_stage3 as gs3

ROOT = Path(__file__).resolve().parents[1]
SIGIR = ROOT / "scripts" / "sigir"
SCRIPT = SIGIR / "run_gatefix_stage3.sh"
DRY = SIGIR / "dryrun"
GATEFIX = "outputs/confrec/gatefix"
S3_REL = f"{GATEFIX}/stage3"
XT = "outputs/baselines/external_tasks"
VALID_REL = f"{XT}/sports_large10000_100neg_valid_same_candidate/ranking_valid.jsonl"
TEST_REL = f"{XT}/sports_large10000_100neg_test_same_candidate/ranking_test.jsonl"
P1_REL = "outputs/confrec/pilot1_mirror/decision.json"
REGISTERED_P1 = ROOT / "outputs" / "confrec_pilot" / "pilot1" / "decision.json"

_spec = importlib.util.spec_from_file_location("pilot1_gate_for_stage3", SIGIR / "pilot1_gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)
_spec = importlib.util.spec_from_file_location("panel_reference_for_stage3", SIGIR / "panel_reference.py")
panel_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel_reference)


def _bash():
    for c in ("C:/Program Files/Git/bin/bash.exe", shutil.which("bash")):
        if c and Path(c).exists() and "system32" not in str(c).lower():   # not WSL's bash.exe
            return str(c)
    return None


BASH = _bash()
NO_BASH = "no bash (Git Bash or a POSIX bash) on PATH: the shell runs of run_gatefix_stage3.sh are skipped"
needs_bash = pytest.mark.skipif(BASH is None, reason=NO_BASH)


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_file(p) -> str:
    return sha1_bytes(Path(p).read_bytes())


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return path


def joined(text: str) -> str:
    return re.sub(r"\\\n\s*", " ", text)


def code_lines(text: str) -> list:
    """The shell code: continuation lines joined, comments and blank lines dropped."""
    return [s.strip() for s in joined(text).splitlines() if s.strip() and not s.strip().startswith("#")]


def script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def tail(r: subprocess.CompletedProcess, n: int = 4000) -> str:
    return f"rc={r.returncode}\n--- stdout ---\n{r.stdout[-n:]}\n--- stderr ---\n{r.stderr[-n:]}"


def snapshot(root: Path) -> dict:
    """{relative path: (size, mtime_ns, sha1)} of every file under root."""
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns, sha1_file(p))
            for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- 1. the script file
def test_script_is_lf_bash_and_bash_n_clean():
    for f in (SCRIPT, DRY / "run_s3.sh", DRY / "fakepy_s3"):
        raw = f.read_bytes()
        assert b"\r\n" not in raw and raw.startswith(b"#!/usr/bin/env bash\n"), f
    assert b"set -euo pipefail" in SCRIPT.read_bytes()
    if BASH is None:
        pytest.skip(NO_BASH)
    for f in (SCRIPT, DRY / "run_s3.sh", DRY / "fakepy_s3"):
        r = subprocess.run([BASH, "-n", f.as_posix()], capture_output=True, text=True)
        assert r.returncode == 0, (f, r.stderr)


def header() -> str:
    head = script_text().split("\nset -euo pipefail\n", 1)[0]
    return " ".join(ln.lstrip("#").strip() for ln in head.splitlines())


def test_the_header_logs_the_gpu_hour_estimate_and_the_g7_scope():
    h = header()
    assert "Estimated cost" in h and "about 2.7 GPU-h in total" in h and "up to about 3 GPU-h" in h
    for s in ("ONLY after the G6 GATE_PASS", "the first 1,500 rows of toys_confirm_h20.jsonl",
              ">= 500 fresh Toys users are eligible", "the first 1,500 users of its seed-0 h20 panel",
              "the first 1,000 events of the sports VALID panel", "--stage3_gate gate.json --pilot1_decision",
              "max_model_len 4096", "not reused", "Exit code"):
        assert s in h, s


# the five scorer runs of G7 exactly as the script requests them: the rated arms under V* at its registered window
# (no --hist_len) and question family, the no-history prior at hist_len 0, the next-item check under V0 at hist_len 5
SCORE_ARMS = {
    '"$CP" "$S3/ml1m"': '--variant "$VSTAR" --readout yesno --questions like,dislike,like_para --swap_k 8',
    '"$CP" "$S3/ml1m_nohist"': '--variant "$VSTAR" --readout yesno --questions like --hist_len 0',
    '"$SP2" "$S3/$SECOND"': '--variant "$VSTAR" --readout yesno --questions like,dislike,like_para --swap_k 8',
    '"$SP2" "$S3/${SECOND}_nohist"': '--variant "$VSTAR" --readout yesno --questions like --hist_len 0',
    '"$SV" "$S3/sports_valid_1k"': '--variant V0 --readout yesno --questions next,like,dislike,like_para --hist_len 5',
}


def test_each_registered_arm_is_requested_once_with_its_flags():
    got = {}
    for s in code_lines(script_text()):
        if re.match(r"score\s", s):
            w = s.split(" ", 3)
            assert f"{w[1]} {w[2]}" not in got, s
            got[f"{w[1]} {w[2]}"] = w[3]
    assert got == SCORE_ARMS
    body = joined(re.search(r"\nscore\(\) \{(.*?)\n\}", script_text(), re.S).group(1))
    assert '"$PY" -m src.confrec.pyes_scorer --data "$data" --output "$dir" --model "$MODEL"' in body
    assert "--dtype float16 --topk_logprobs 50 --max_model_len 4096 --chunk_users 100" in body      # G0, A1 C0
    assert "key=\"$(sha1sum \"$data\" | cut -d' ' -f1) $MODEL $*\"" in body   # skip key: panel bytes + model + args
    assert 'mv "$dir" "$dir.stale.' in body and 'echo "[skip] $dir: scored"' in body


def test_the_next_item_arm_reads_the_sports_valid_panel_never_test():
    t, code = script_text(), "\n".join(code_lines(script_text()))
    assert re.search(r"^XT=outputs/baselines/external_tasks$", t, re.M)
    assert re.search(r'^VALID="\$XT/sports_large10000_100neg_valid_same_candidate/ranking_valid\.jsonl"$', t, re.M)
    assert code.count('"$TEST"') == 1 and '--sports_test "$TEST"' in code    # read only by the quarantine check
    assert 'SV="$S3/panels/sports_valid_1k.jsonl"' in code and 'first_lines "$VALID" "$NSP" "$SV"' in code
    assert re.search(r'^\s*head -n "\$n" "\$src" > "\$out\.tmp"$', t, re.M)          # run_nextitem_audit.sh's cut
    assert 'first_lines "$TOYS_CONFIRM" "$N2" "$SP2"' in code


def test_pilot_mirror_runs_on_each_scored_panel_and_pilot1_gate_on_the_records():
    t, code = script_text(), "\n".join(code_lines(script_text()))
    pm = re.findall(r'"\$PY" -m src\.confrec\.pilot_mirror ([^\n]*)', code)
    rated, sports = [c for c in pm if "--base_q like" in c], [c for c in pm if "--base_q next" in c]
    assert len(pm) == 3 and len(rated) == 2 and len(sports) == 1
    for c in rated:
        assert "--swap " in c and "--nohist " in c and "--ref_ranks" not in c
    assert '--ref_ranks "$CCRP"' in sports[0] and "--swap" not in sports[0] and "--nohist" not in sports[0]
    assert re.search(r"^CCRP=docs/sigir/ref_ranks/sports/ccrp_v3\.csv\.gz$", t, re.M)
    calls = re.findall(r'"\$PY" scripts/sigir/pilot1_gate\.py ([^\n]*)', code)
    assert len(calls) == 1
    for f in ('--ml1m "$S3/ml1m/pilot_mirror.json"', '--toys "$S3/$SECOND/pilot_mirror.json"',
              '--sports "$S3/sports_valid_1k/pilot_mirror.json"', '--out_dir "$S3"', '--stage3_gate "$GATE"',
              '--pilot1_decision "$P1DEC"'):
        assert f in calls[0], f
    for a in ('G=outputs/confrec/gatefix', 'CONF="$G/confirm"', 'S3="$G/stage3"', 'GATE="$CONF/gate.json"',
              'P1DEC=outputs/confrec/pilot1_mirror/decision.json', 'SEL="$DEV/selection.json"'):
        assert re.search(r"^" + re.escape(a) + r"$", t, re.M), a
    assert code.rstrip().endswith('exit "$rc"')                                    # pilot1_gate's exit code


def test_the_refusals_precede_every_write_and_scorer_call():
    code = code_lines(script_text())

    def first(pred):
        return next(k for k, s in enumerate(code) if pred(s))
    g = first(lambda s: s.startswith('[ -f "$GATE" ] || refuse'))
    fr = first(lambda s: "-m src.confrec.diag_battery freeze" in s)
    pre = first(lambda s: "-m src.confrec.gatefix_stage3" in s)
    write = first(lambda s: s.startswith(("mkdir", "first_lines ", "score ", "step ", "analysis ", '"$PY" -c')))
    assert g < fr < pre < write
    assert re.search(r"\|\|\s+refuse ", code[fr]) and code[pre].startswith("if ! PRE=$(")
    assert code[pre + 1].startswith("refuse ")
    assert 'refuse() { echo "STAGE 3 REFUSED: $* Nothing was scored." >&2; exit 2; }' in code


def test_the_freeze_check_is_run_gatefix_sh_s_own():
    def arr(text):
        return re.search(r"^FREEZE_ARGS=\(([^)]*)\)", joined(text), re.M).group(1).split()
    gf = (SIGIR / "run_gatefix.sh").read_text(encoding="utf-8")
    assert arr(script_text()) == arr(gf)
    for a in ("G=outputs/confrec/gatefix", 'GP="$G/panels"', 'FREEZE="$G/FREEZE.txt"',
              "AMEND=idea-stage/PREREG_AMENDMENT_2.md", "PILOT_LOG=docs/sigir/PILOT_LOG.md"):
        for text in (script_text(), gf):
            assert re.search(r"^" + re.escape(a) + r"$", text, re.M), a
    assert '"$PY" -m src.confrec.diag_battery freeze "${FREEZE_ARGS[@]}" --check --pilot_log "$PILOT_LOG"' in \
        joined(script_text())


# ---------------------------------------------------------------- 2. every flag exists in the real argparse
def test_the_integration_audit_of_test_confrec_contracts_accepts_the_script():
    spec = importlib.util.spec_from_file_location("contracts_for_stage3", ROOT / "tests" / "test_confrec_contracts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert SCRIPT in sorted(SIGIR.glob("*.sh"))                     # found by its glob: no registration needed
    mod.test_every_script_flag_exists_in_the_real_argparse(SCRIPT)


class _Captured(Exception):
    pass


def _parser(target: str, monkeypatch):
    """The real argparse parser of 'src.confrec.x' or 'scripts/sigir/y.py' (slim_amazon2023 without `requests`: a stub)."""
    if importlib.util.find_spec("requests") is None:
        monkeypatch.setitem(sys.modules, "requests", types.ModuleType("requests"))
    if target.endswith(".py"):
        spec = importlib.util.spec_from_file_location("audit_s3_" + Path(target).stem, ROOT / target)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    orig = (argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args)

    def boom(self, *a, **k):
        raise _Captured(self)
    argparse.ArgumentParser.parse_args = argparse.ArgumentParser.parse_known_args = boom
    old = sys.argv
    sys.argv = ["prog"]
    try:
        for call in (lambda: mod.main(), lambda: mod.parse_args(), lambda: mod.main([])):
            try:
                call()
            except _Captured as c:
                return c.args[0]
            except (SystemExit, Exception):
                continue
    finally:
        argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args = orig
        sys.argv = old
    raise AssertionError(f"no argparse parser captured for {target}")


_CUT = re.compile(r"\s(?:\|\||\||&&|;|>|2>&1|2>)\s|;\s*(?:then|do)\b|;\s*$")


def _words(rest: str) -> list:
    return shlex.split(re.sub(r"\)\s*$", "", _CUT.split(rest)[0]), comments=True)


def _flags(words: list, arrays: dict) -> list:
    """[(flag, value or None)] of shell words, every ${NAME[@]} expanded from its definition."""
    out, plain = [], []
    for w in words:
        m = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\[@\]\}", w)
        if m:
            for body in arrays.get(m.group(1), []):
                out += _flags(shlex.split(body), arrays)
        else:
            plain.append(w)
    for i, w in enumerate(plain):
        if w.startswith("--"):
            nxt = plain[i + 1] if i + 1 < len(plain) else None
            out.append((w, None if nxt is None or nxt.startswith("--") else nxt))
    return out


def invocations(text: str) -> list:
    """(target, subcommand, [(flag, value)], line) of every "$PY" call of a repo entry point (assignments and
    `if ! X=$(...)` included); a `score` line carries the flags of score()'s own pyes_scorer call."""
    code = joined(text)
    arrays = defaultdict(list)
    for name, body in re.findall(r"(?<![\w$])([A-Za-z_][A-Za-z_0-9]*)\+?=\(([^)]*)\)", code, re.S):
        arrays[name].append(body)
    lines = code_lines(text)
    calls, fixed = [], []
    for s in lines:
        m = re.search(r'"\$PY" (?:-m (src\.confrec\.[a-z_0-9]+)|(scripts/sigir/[a-z_0-9]+\.py))(.*)$', s)
        if m:
            words = _words(m.group(3))
            sub = words[0] if words and re.fullmatch(r"[a-z_0-9]+", words[0]) else None
            flags = _flags(words[1:] if sub else words, arrays)
            calls.append((m.group(1) or m.group(2), sub, flags, s))
            if m.group(1) == "src.confrec.pyes_scorer":
                fixed = flags
    for s in lines:
        if re.match(r"score\s", s):
            calls.append(("src.confrec.pyes_scorer", None, fixed + _flags(_words(s)[3:], arrays), s))
    return calls


def audit(text: str, monkeypatch) -> tuple[list, dict]:
    problems, parsers, count = [], {}, defaultdict(int)
    for target, sub, flags, line in invocations(text):
        count[target] += 1
        if target not in parsers:
            parsers[target] = _parser(target, monkeypatch)
        p = parsers[target]
        subs = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
        if subs:
            if sub not in subs[0].choices:
                problems.append(f"{target} has no subcommand {sub!r}: {line[:120]}")
                continue
            p = subs[0].choices[sub]
        acts = {s: a for a in p._actions for s in a.option_strings}
        used = {f for f, _ in flags}
        if used - set(acts):
            problems.append(f"{target} has no flag {sorted(used - set(acts))}: {line[:120]}")
        req = {a.option_strings[0] for a in set(acts.values()) if a.option_strings and a.required}
        if req - used:
            problems.append(f"{target} lacks required {sorted(req - used)}: {line[:120]}")
        for f, v in flags:
            a = acts.get(f)
            if a is None or v is None or "$" in v:
                continue
            if a.choices and v not in a.choices:
                problems.append(f"{target} {f} {v!r} not in {list(a.choices)}")
            if a.type in (int, float):
                try:
                    a.type(v)
                except ValueError:
                    problems.append(f"{target} {f} {v!r} is not a {a.type.__name__}")
    return problems, dict(count)


def test_every_flag_the_script_passes_exists_in_the_real_argparse(monkeypatch):
    problems, count = audit(script_text(), monkeypatch)
    assert not problems, "\n".join(problems)
    # every entry point of the chain was seen (an audit that finds nothing cannot pass): score()'s own call + 5 arms
    assert count == {"src.confrec.pyes_scorer": 6, "src.confrec.pilot_mirror": 3, "scripts/sigir/pilot1_gate.py": 1,
                     "src.confrec.gatefix_stage3": 1, "src.confrec.diag_battery": 1,
                     "src.confrec.build_rated_panels": 1, "scripts/sigir/slim_amazon2023.py": 1}, count


def test_the_flag_audit_catches_a_wrong_flag_and_a_wrong_value(monkeypatch):
    t = script_text()
    bad = (t.replace("--swap_k 8", "--swapk 8").replace("--readout yesno", "--readout yes")
           .replace("--stage3_gate", "--stage_3_gate").replace("--sports_valid", "--sport_valid")
           .replace("--hist_len 5", "--hist_len five").replace("--source amazon", "--source amazn"))
    problems, _ = audit(bad, monkeypatch)
    for s in ("--swapk", "'yes' not in", "--stage_3_gate", "--sport_valid", "'five' is not a int", "'amazn' not in"):
        assert any(s in p for p in problems), (s, problems)
    problems, _ = audit(t.replace('--toys_confirm "$TOYS_CONFIRM" ', ""), monkeypatch)   # a required flag dropped
    assert any("lacks required ['--toys_confirm']" in p for p in problems), problems


# ---------------------------------------------------------------- 3. the input check, in-process
def g6_record(**kw) -> dict:
    """gatefix_select confirm gate.json: the fields the input check and pilot1_gate.stage3_gate read."""
    rec = {"stage": "confirm", "decision": "GATE_PASS", "outcome": "PASS", "gate_pass": True, "v_star": "V3",
           "UAUC": 0.612, "UAUC_min": 0.60, "E1": {"E1": True}, "n_users": 20, "data_sha1": None,
           "model": "/models/Qwen3-8B"}
    rec.update(kw)
    return rec


def selection_record(v: str = "V3", decision: str = "FIX_FOUND") -> dict:
    found = decision == "FIX_FOUND"
    return {"stage": "dev", "decision": decision, "outcome": "FIX_FOUND" if found else "F0", "fix_found": found,
            "v_star": v, "gate_ft_prompt": v if found else "V0"}


def pilot1_record(ndcg: float = 0.20934, ok: bool = True) -> dict:
    return {"decision": "GATE_FAIL_UNINTERPRETABLE",
            "gate": {"pass": False, "ml1m_raw_UAUC": 0.5874, "sports_raw_NDCG@10": ndcg, "sports_raw_NDCG@10_min": 0.18632,
                     "input_checks": {"sports": {"panel_type": "next_item", "base_question": "next", "ok": ok}}}}


def next_row(eid: str, split: str = "valid") -> dict:
    return {"source_event_id": eid, "user_id": eid.split("::")[0], "candidate_item_ids": ["a", "b"],
            "positive_item_index": 0, "split_name": split}


class Inputs:
    """The files the input check reads, in the registered state: G6 GATE_PASS of V3 on 20 CONFIRM ML-1M users, a built
    Toys CONFIRM split of n_toys fresh users, a sports VALID file of n_valid events, the TEST file of n_test events."""

    def __init__(self, d: Path, n_toys: int = 600, n_valid: int = 1200, n_test: int = 1000):
        self.d = d
        self.confirm = write_rows(d / "panels" / "ml1m_confirm_h20.jsonl", [{"user_id": f"c{k}"} for k in range(20)])
        self.toys = write_rows(d / "panels" / "toys_confirm_h20.jsonl", [{"user_id": f"t{k}"} for k in range(n_toys)])
        self.valid = write_rows(d / "valid" / "ranking_valid.jsonl", [next_row(f"u{k}::1") for k in range(n_valid)])
        self.test = write_rows(d / "test" / "ranking_test.jsonl", [next_row(f"u{k}::2", "test") for k in range(n_test)])
        self.model = "/models/Qwen3-8B"
        self.gate = g6_record(data_sha1=sha1_file(self.confirm))
        self.selection = selection_record()
        self.p1 = pilot1_record()
        self.manifest = {"sources": {
            "ml1m": {"confirm": {"built": True, "sha1": sha1_file(self.confirm), "n_users": 20}},
            "toys": {"fresh": n_toys, "confirm": {"built": True, "sha1": sha1_file(self.toys), "n_users": n_toys}}}}
        self.out = d / "stage3" / "inputs.json"

    def argv(self) -> list:
        for name in ("gate", "selection", "p1", "manifest"):
            p, obj = self.d / f"{name}.json", getattr(self, name)
            if obj is None:
                p.unlink(missing_ok=True)
            else:
                p.write_text(json.dumps(obj), encoding="utf-8")
        return ["--gate", str(self.d / "gate.json"), "--selection", str(self.d / "selection.json"),
                "--manifest", str(self.d / "manifest.json"), "--confirm_panel", str(self.confirm),
                "--toys_confirm", str(self.toys), "--pilot1_decision", str(self.d / "p1.json"), "--model", self.model,
                "--sports_valid", str(self.valid), "--sports_test", str(self.test), "--out", str(self.out)]


def check(inp: Inputs, capsys):
    rc = gs3.main(inp.argv())
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def test_the_input_check_passes_on_the_registered_state_and_records_it(tmp_path, capsys):
    inp = Inputs(tmp_path)
    rc, out, err = check(inp, capsys)
    assert rc == 0, err
    assert out == "V3 toys 600 1000\n"                       # the one stdout line the run script reads
    rec = read_json(inp.out)
    assert rec["v_star"] == "V3" and rec["gate"]["decision"] == "GATE_PASS" and rec["model"] == inp.model
    assert rec["recorded_stage3_gate"] == {"rated_component_ok": True, "next_item_component_ok": True,
                                           "G6_UAUC": 0.612, "pilot1_sports_raw_NDCG@10": 0.20934}
    assert rec["ml1m"] == {"panel": inp.confirm.as_posix(), "sha1": sha1_file(inp.confirm), "n_users": 20}
    s = rec["second"]
    assert (s["domain"], s["n_users"], s["toys_fresh_eligible"], s["toys_confirm_built"]) == ("toys", 600, 600, True)
    assert s["rows_sha1"] == sha1_file(inp.toys) and ">= 500" in s["reason"]
    sp = rec["sports_valid"]
    assert sp["n_events"] == 1000 and sp["quarantine_checked"] is True and sp["test_panel"] == inp.test.as_posix()
    assert sp["rows_sha1"] == sha1_bytes(b"".join(inp.valid.read_bytes().splitlines(keepends=True)[:1000]))
    assert sp["anchor"]["byte_identical_to_original"] is False
    assert sp["anchor"]["original_sha256"] == panel_reference.ORIGINAL["sports"]["valid"]["ranking"]
    assert rec["constants"] == {"toys_min_fresh": 500, "n_second": 1500, "n_sports": 1000}
    for k in ("gate", "selection", "pilot1_decision", "manifest"):
        assert re.fullmatch(r"[0-9a-f]{40}", rec[k]["sha1"]), k
    m1, text = inp.out.stat().st_mtime_ns, inp.out.read_text(encoding="utf-8")
    rc, out2, _ = check(inp, capsys)                          # a rerun leaves the record (and its mtime) alone
    assert rc == 0 and out2 == out and inp.out.stat().st_mtime_ns == m1 and inp.out.read_text(encoding="utf-8") == text


def test_the_second_domain_is_the_first_1500_fresh_toys_rows(tmp_path, capsys):
    inp = Inputs(tmp_path, n_toys=1600)
    rc, out, err = check(inp, capsys)
    assert rc == 0, err
    assert out.split()[1:3] == ["toys", "1500"]
    rows = inp.toys.read_bytes().splitlines(keepends=True)
    assert read_json(inp.out)["second"]["rows_sha1"] == sha1_bytes(b"".join(rows[:1500]))


@pytest.mark.parametrize("built, n_fresh, domain, n_users", [
    (True, 500, "toys", 500),          # G7: >= 500 fresh users -> fresh Toys
    (True, 499, "games", 1500),        # < 500 -> Video_Games, its first 1,500 users
    (False, 300, "games", 1500),       # the CONFIRM split not built (G2: < 2,000 eligible) -> Video_Games
])
def test_g7_second_domain_rule(tmp_path, built, n_fresh, domain, n_users):
    toys = write_rows(tmp_path / "toys_confirm_h20.jsonl", [{"user_id": f"t{k}"} for k in range(n_fresh)])
    conf = ({"built": True, "sha1": sha1_file(toys), "n_users": n_fresh} if built else
            {"built": False, "n_fresh_eligible": n_fresh, "reason": "1800 eligible users < 2000 (amendment 2 G2)"})
    rec, errs = gs3.second_domain({"sources": {"toys": {"confirm": conf, "fresh": n_fresh}}}, toys)
    assert errs == [] and (rec["domain"], rec["n_users"]) == (domain, n_users)
    assert rec["toys_fresh_eligible"] == n_fresh and (domain == "toys") == ("source_sha1" in rec)
    if domain == "games":
        assert "--domain games --hist_len 20 --gatefix_fields --n_users 1500" in rec["rows"]
    rec, errs = gs3.second_domain({"sources": {"ml1m": {}}}, toys)
    assert len(errs) == 1 and "no toys source" in errs[0]


def _append(path: Path) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"user_id": "extra"}) + "\n")


def _valid(inp: Inputs, first: dict) -> None:
    rows = [json.loads(x) for x in inp.valid.read_text(encoding="utf-8").splitlines()]
    write_rows(inp.valid, [first] + rows[1:])


REFUSALS = {
    "no_gate": (lambda i: setattr(i, "gate", None), "no G6 gate record"),
    "gate_f1": (lambda i: i.gate.update(decision="GATE_FAIL_AFTER_REMEDY(confirm)", gate_pass=False, UAUC=0.59),
                "not GATE_PASS"),
    "gate_uauc_edited": (lambda i: i.gate.update(UAUC=0.5999), "rated component does not hold"),
    "gate_e1_edited": (lambda i: i.gate.update(E1={"E1": False}), "rated component does not hold"),
    "pilot1_sports_below": (lambda i: setattr(i, "p1", pilot1_record(ndcg=0.1863)), "sports next-item component"),
    "pilot1_input_check": (lambda i: setattr(i, "p1", pilot1_record(ok=False)), "sports next-item component"),
    "pilot1_missing": (lambda i: setattr(i, "p1", None), "Pilot-1 decision record"),
    "selection_f0": (lambda i: setattr(i, "selection", selection_record(decision="GATE_FAIL_AFTER_REMEDY(dev)")),
                     "not FIX_FOUND"),
    "selection_other_vstar": (lambda i: setattr(i, "selection", selection_record(v="V1")), "recorded for 'V3'"),
    "model": (lambda i: setattr(i, "model", "/models/Llama-3.1-8B-Instruct"), "is not the backbone G6 scored"),
    "confirm_bytes": (lambda i: _append(i.confirm), "is not the panel G6 scored"),
    "manifest_ml1m_split": (lambda i: i.manifest["sources"]["ml1m"]["confirm"].update(sha1="0" * 40),
                            "manifest's ml1m confirm split"),
    "toys_bytes": (lambda i: _append(i.toys), "manifest's toys confirm split"),
    "no_toys_source": (lambda i: i.manifest["sources"].pop("toys"), "no toys source"),
    "valid_missing": (lambda i: i.valid.unlink(), "no sports VALID panel"),
    "valid_quarantine": (lambda i: _valid(i, next_row("u7::2")), "quarantined"),
    "valid_split": (lambda i: _valid(i, next_row("w0::1", "test")), "split_name"),
    "valid_not_next_item": (lambda i: _valid(i, {"source_event_id": "w0::1", "candidate_labels": [1, 0]}),
                            "not a next-item panel row"),
    "valid_duplicate": (lambda i: _valid(i, next_row("u1::1")), "duplicated source_event_id"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_the_input_check_refuses_and_writes_nothing(tmp_path, capsys, case):
    inp = Inputs(tmp_path)
    mutate, msg = REFUSALS[case]
    mutate(inp)
    rc, out, err = check(inp, capsys)
    assert rc == gs3.EXIT_REFUSED == 2 and msg in err and out == "", (case, err)
    assert "STAGE 3 INPUT CHECK FAILED" in err and not inp.out.exists() and not inp.out.parent.exists()


def test_every_refusal_reason_is_reported_and_the_quarantine_covers_test_events_1_1000(tmp_path, capsys):
    inp = Inputs(tmp_path)
    inp.model, inp.p1 = "/models/other", pilot1_record(ndcg=0.1)
    _valid(inp, next_row("u3::2"))
    rc, _, err = check(inp, capsys)
    assert rc == 2 and all(m in err for m in ("is not the backbone", "sports next-item component", "quarantined"))
    inp = Inputs(tmp_path / "b", n_test=1001)
    _valid(inp, next_row("u1000::2"))                         # the TEST event at position 1,001 is not quarantined
    rc, _, err = check(inp, capsys)
    assert rc == 0, err


# ---------------------------------------------------------------- 4. refusals of the real script (minimal repo copy)
def _ml1m_raw(d: Path, n_users: int = 60, n_movies: int = 70, seed: int = 1) -> None:
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


def mini_repo(base: Path) -> dict:
    """The script, src/confrec and the scripts it loads; h20 DEV / CONFIRM panels (build_rated_panels rows), their
    manifest, FREEZE.txt written by the real diag_battery.freeze with the script's relative paths and its REQUIRED sha1s
    in docs/sigir/PILOT_LOG.md; a G6 GATE_PASS record, selection.json, the Pilot-1 record and sports panels."""
    repo = base / "repo"
    (repo / "src" / "confrec").mkdir(parents=True)
    shutil.copy2(ROOT / "src" / "__init__.py", repo / "src" / "__init__.py")
    for f in (ROOT / "src" / "confrec").glob("*.py"):
        shutil.copy2(f, repo / "src" / "confrec" / f.name)
    for rel in ("scripts/sigir/run_gatefix_stage3.sh", "scripts/sigir/pilot1_gate.py", "scripts/sigir/panel_reference.py",
                "idea-stage/PREREG_AMENDMENT_2.md"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, repo / rel)
    _ml1m_raw(base / "raw")
    items, events = brp.load_ml1m(base / "raw")
    rows, _ = brp.build(items, events, n_users=10 ** 6, n_cands=10, hist_len=20, min_hist=3, min_like=2,
                        min_dislike=2, seed=0, source="ml1m")
    gp = repo / GATEFIX / "panels"
    gp.mkdir(parents=True)

    def panel(name, rs, prefix):
        out = [brp.row_line(dict(r, user_id=prefix + str(r["user_id"]), source_event_id=prefix + r["source_event_id"]))
               for r in rs]
        (gp / name).write_text("".join(out), encoding="utf-8", newline="\n")
        ids = "\n".join(sorted(prefix + str(r["user_id"]) for r in rs)).encode("utf-8")
        return gp / name, sha1_bytes(ids)
    (ml_dev, ml_ids), (toys_dev, toys_ids) = panel("ml1m_dev_h20.jsonl", rows[:20], "m"), panel("toys_dev_h20.jsonl",
                                                                                                rows[:20], "t")
    conf, conf_ids = panel("ml1m_confirm_h20.jsonl", rows[20:40], "c")
    man = {"sources": {"ml1m": {"dev": {"built": True, "sha1": sha1_file(ml_dev), "user_ids_sha1": ml_ids},
                                "confirm": {"built": True, "sha1": sha1_file(conf), "n_users": 20,
                                            "user_ids_sha1": conf_ids}},
                       "toys": {"dev": {"built": True, "sha1": sha1_file(toys_dev), "user_ids_sha1": toys_ids},
                                "confirm": {"built": False, "n_fresh_eligible": 0, "reason": "test repo"}}},
           "freeze": {"ml1m_dev_users_sha1": ml_ids, "ml1m_confirm_users_sha1": conf_ids, "toys_dev_users_sha1": toys_ids}}
    (gp / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        text = db.freeze("idea-stage/PREREG_AMENDMENT_2.md", [f"{GATEFIX}/panels/ml1m_dev_h20.jsonl",
                         f"{GATEFIX}/panels/toys_dev_h20.jsonl"], f"{GATEFIX}/panels/manifest.json",
                         f"{GATEFIX}/FREEZE.txt")
    finally:
        os.chdir(cwd)
    required = re.findall(r"= ([0-9a-f]{40})  REQUIRED", text)
    log = repo / "docs" / "sigir" / "PILOT_LOG.md"
    log.parent.mkdir(parents=True)
    log.write_text("# PILOT_LOG (test copy)\n\n" + "".join(f"- {s}\n" for s in required), encoding="utf-8")
    model = f"{repo.as_posix()}/models/Qwen3-8B"
    files = {f"{GATEFIX}/confirm/gate.json": g6_record(data_sha1=sha1_file(conf), model=model),
             f"{GATEFIX}/dev/selection.json": selection_record(),
             P1_REL: read_json(REGISTERED_P1) if REGISTERED_P1.is_file() else pilot1_record()}
    for rel, obj in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    write_rows(repo / VALID_REL, [next_row(f"u{k}::1") for k in range(5)])
    write_rows(repo / TEST_REL, [next_row(f"u{k}::2", "test") for k in range(5)])
    return {"repo": repo, "model": model, "required": required}


def run_stage3(repo: Path, simlog: Path, **env) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MODEL", "PYTHON", "SIMLOG", "S3_MIN_FRESH")}
    e.update(PYTHON=(DRY / "fakepy_s3").as_posix(), REALPY=Path(sys.executable).as_posix(), SIMLOG=simlog.as_posix(),
             FAKE_SCENARIO=(DRY / "scen_fix.json").as_posix(), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    e.update(env)
    return subprocess.run([BASH, (repo / "scripts" / "sigir" / "run_gatefix_stage3.sh").as_posix()], env=e,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)


def _calls(simlog: Path) -> str:
    return simlog.read_text(encoding="utf-8") if simlog.exists() else ""


@pytest.fixture
def mini(tmp_path):
    if BASH is None:
        pytest.skip(NO_BASH)
    return mini_repo(tmp_path)


@needs_bash
def test_refused_without_a_g6_gate_record_before_any_python_call(mini, tmp_path):
    (mini["repo"] / GATEFIX / "confirm" / "gate.json").unlink()
    log = tmp_path / "calls.log"
    r = run_stage3(mini["repo"], log, MODEL=mini["model"])
    assert r.returncode == 2 and "STAGE 3 REFUSED: no G6 gate record" in r.stderr, tail(r)
    assert _calls(log) == "" and not (mini["repo"] / S3_REL).exists()


@needs_bash
def test_refused_when_the_g6_record_is_not_gate_pass(mini, tmp_path):
    g = mini["repo"] / GATEFIX / "confirm" / "gate.json"
    g.write_text(json.dumps(dict(read_json(g), decision="GATE_FAIL_AFTER_REMEDY(confirm)", outcome="F1",
                                 gate_pass=False, UAUC=0.581)), encoding="utf-8")
    log = tmp_path / "calls.log"
    r = run_stage3(mini["repo"], log, MODEL=mini["model"])
    assert r.returncode == 2 and "not GATE_PASS" in r.stderr and "STAGE 3 REFUSED" in r.stderr, tail(r)
    assert "freeze check OK" in r.stdout                          # the freeze passed: the G6 record refused
    calls = _calls(log)
    assert "-m src.confrec.diag_battery freeze" in calls and "-m src.confrec.gatefix_stage3" in calls
    assert "pyes_scorer" not in calls and not (mini["repo"] / S3_REL).exists()


@needs_bash
@pytest.mark.parametrize("breakage, msg", [
    ("unrecorded", "lacks the required sha1"),                    # PILOT_LOG without the freeze record
    ("amendment_changed", "differs from the recomputed freeze"),  # a hashed file changed after the freeze
])
def test_refused_when_the_amendment2_freeze_check_fails(mini, tmp_path, breakage, msg):
    repo = mini["repo"]
    if breakage == "unrecorded":
        (repo / "docs" / "sigir" / "PILOT_LOG.md").write_text("# PILOT_LOG (test copy)\n", encoding="utf-8")
    else:
        with open(repo / "idea-stage" / "PREREG_AMENDMENT_2.md", "a", encoding="utf-8", newline="\n") as f:
            f.write("\nan edit after the freeze\n")
    log = tmp_path / "calls.log"
    r = run_stage3(repo, log, MODEL=mini["model"])
    assert r.returncode == 2 and msg in r.stderr and "the amendment-2 freeze check failed" in r.stderr, tail(r)
    calls = _calls(log)
    assert "-m src.confrec.diag_battery freeze" in calls and "gatefix_stage3" not in calls and "pyes_scorer" not in calls
    assert not (repo / S3_REL).exists()


# ---------------------------------------------------------------- 5. end-to-end dry run in the integration sandbox
@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """setup_sandbox.py (2,200 Amazon users: Toys has >= 2,000 eligible, so its CONFIRM split is built), s3_sandbox.py
    prep (sports VALID panel, the Pilot-1 record), run_gf.sh = the real run_gatefix.sh stage 0, the freeze record,
    stages 1-2 (scen_fix: V3 fix -> GATE_PASS); then run_s3.sh = the real run_gatefix_stage3.sh three times: fresh
    Toys, the same again, and the Video_Games fallback (S3_MIN_FRESH above every fresh count)."""
    if BASH is None:
        pytest.skip(NO_BASH)
    root = tmp_path_factory.mktemp("stage3_sbx")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MODEL", "PYTHON", "SIMLOG", "S3_MIN_FRESH")}
    env.update(SBX_ROOT=root.as_posix(), REALPY=Path(sys.executable).as_posix(), PYTHONUTF8="1",
               PYTHONIOENCODING="utf-8")

    def run(cmd):
        return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=3600)

    def py(*args):
        r = run([sys.executable, *map(str, args)])
        assert r.returncode == 0, tail(r)

    def sh(script, *args):
        return run([BASH, f"scripts/sigir/dryrun/{script}", *args])
    py(DRY / "setup_sandbox.py", "--fresh", "--sbx", "s3", "--amazon_users", "2200")
    py(DRY / "s3_sandbox.py", "prep", "--sbx", "s3")
    r = sh("run_gf.sh", "s3", "scen_fix.json", "gf")
    assert r.returncode == 0 and "STOPPED at the amendment-2 freeze" in r.stdout, tail(r)
    py(DRY / "s3_sandbox.py", "record_freeze", "--sbx", "s3")
    r = sh("run_gf.sh", "s3", "scen_fix.json", "gf", "FREEZE_ACK=1")
    assert r.returncode == 0 and "DECISION: GATE_PASS" in r.stdout, tail(r)
    sb = root / "s3"
    s3 = sb / S3_REL
    out = {"root": root, "sb": sb, "s3": s3, "model": f"{root.as_posix()}/s3/models/Qwen3-8B"}
    for k, tag, extra in ((1, "s3a", ()), (2, "s3b", ()), (3, "s3c", ("S3_MIN_FRESH=1000000000",))):
        out[f"r{k}"] = sh("run_s3.sh", "s3", "scen_fix.json", tag, *extra)
        out[f"calls{k}"] = _calls(root / f"calls_{tag}.log")
        out[f"snap{k}"] = snapshot(s3) if s3.exists() else {}
        for name in ("decision.json", "inputs.json"):
            out[f"{name.split('.')[0]}{k}"] = read_json(s3 / name) if (s3 / name).exists() else None
    return out


def test_dry_run_stages_0_2_record_a_gate_pass_and_a_toys_confirm_split(e2e):
    g6 = read_json(e2e["sb"] / GATEFIX / "confirm" / "gate.json")
    assert (g6["decision"], g6["v_star"], g6["model"]) == ("GATE_PASS", "V3", e2e["model"])
    toys = read_json(e2e["sb"] / GATEFIX / "panels" / "manifest.json")["sources"]["toys"]["confirm"]
    assert toys["built"] is True and toys["n_users"] >= gs3.TOYS_MIN_FRESH


def test_dry_run_stage3_writes_the_decision_file(e2e):
    r = e2e["r1"]
    assert r.returncode == 0 and "run_gatefix_stage3.sh exit 0" in r.stdout, tail(r)
    dec, s3 = e2e["decision1"], e2e["s3"]
    assert dec["decision"] in gate.LABELS and dec["decision"] not in ("GATE_FAIL_UNINTERPRETABLE", "INCOMPLETE")
    assert dec["missing_inputs"] == [] and f"DECISION: {dec['decision']}" in r.stdout
    g = dec["gate"]
    assert g["source"] == "stage3" and g["pass"] is True and g["input_checks_ok"] is True
    assert g["rated_component"]["ok"] and g["rated_component"]["v_star"] == "V3" and g["next_item_component"]["ok"]
    assert g["ml1m_n_users_equals_g6"] is True and "stage 3" in dec["rule"]
    assert (s3 / "GATE_PASS").is_file() and not (s3 / "GATE_FAIL").exists()
    assert dec["inputs"] == {"ml1m": f"{S3_REL}/ml1m/pilot_mirror.json", "toys": f"{S3_REL}/toys/pilot_mirror.json",
                             "sports": f"{S3_REL}/sports_valid_1k/pilot_mirror.json",
                             "stage3_gate": f"{GATEFIX}/confirm/gate.json", "pilot1_decision": P1_REL}
    g6 = read_json(e2e["sb"] / GATEFIX / "confirm" / "gate.json")
    s = dec["input_summary"]
    assert s["ml1m"]["n_users"] == g6["n_users"] and s["toys"]["n_users"] == 100 and s["sports"]["n_events"] == 34
    assert all(s[k]["panel_join"] == {} for k in ("ml1m", "toys", "sports"))


def test_dry_run_scores_each_registered_arm_once_with_its_flags(e2e):
    sb, s3, model = e2e["sb"], e2e["s3"], e2e["model"]
    cp = sb / GATEFIX / "panels" / "ml1m_confirm_h20.jsonl"
    toys, sv = s3 / "panels" / "toys_fresh_first100_h20.jsonl", s3 / "panels" / "sports_valid_1k.jsonl"
    rated, nohist = "--questions like,dislike,like_para --swap_k 8", "--questions like --hist_len 0"
    expect = {"ml1m": (cp, f"--variant V3 --readout yesno {rated}", ["like", "dislike", "like_para"], 20, 8),
              "ml1m_nohist": (cp, f"--variant V3 --readout yesno {nohist}", ["like"], 0, 0),
              "toys": (toys, f"--variant V3 --readout yesno {rated}", ["like", "dislike", "like_para"], 20, 8),
              "toys_nohist": (toys, f"--variant V3 --readout yesno {nohist}", ["like"], 0, 0),
              "sports_valid_1k": (sv, "--variant V0 --readout yesno --questions next,like,dislike,like_para --hist_len 5",
                                  ["next", "like", "dislike", "like_para"], 5, 0)}
    for d, (panel, args, qs, hist, swap) in expect.items():
        assert (s3 / d / "run.key").read_text(encoding="utf-8").strip() == f"{sha1_file(panel)} {model} {args}", d
        cfg = read_json(s3 / d / "report.json")["config"]
        assert cfg["data_sha1"] == sha1_file(panel) and cfg["questions"] == qs and cfg["hist_len"] == hist, d
        assert cfg["swap_k"] == swap and cfg["variant"] == ("V0" if d.startswith("sports") else "V3"), d
        assert (cfg["readout"], cfg["dtype"], cfg["topk_logprobs"], cfg["max_model_len"], cfg["chunk_users"],
                cfg["model"], cfg["lora"]) == ("yesno", "float16", 50, 4096, 100, model, None), d
        assert (s3 / d / "swap_prior.csv.gz").is_file() == (swap == 8), d
        assert (s3 / d / "pilot_mirror.json").is_file() == (not d.endswith("_nohist")), d
    scorer = [c for c in e2e["calls1"].splitlines() if c.startswith("-m src.confrec.pyes_scorer")]
    assert len(scorer) == 5 and sum("--max_model_len 4096" in c for c in scorer) == 5
    assert sum(c.startswith("-m src.confrec.pilot_mirror") for c in e2e["calls1"].splitlines()) == 3


def test_dry_run_sub_panels_are_the_registered_first_rows_and_never_sports_test(e2e):
    sb, s3 = e2e["sb"], e2e["s3"]
    toys_src = (sb / GATEFIX / "panels" / "toys_confirm_h20.jsonl").read_bytes().splitlines(keepends=True)
    assert (s3 / "panels" / "toys_fresh_first100_h20.jsonl").read_bytes() == b"".join(toys_src[:100])
    valid = (sb / VALID_REL).read_bytes().splitlines(keepends=True)
    sv = s3 / "panels" / "sports_valid_1k.jsonl"
    assert len(valid) == 40 and sv.read_bytes() == b"".join(valid[:34])    # the sandbox's "first 1,000" = 34 events
    ids = {json.loads(x)["source_event_id"] for x in sv.read_text(encoding="utf-8").splitlines()}
    test_ids = {json.loads(x)["source_event_id"] for x in (sb / TEST_REL).read_text(encoding="utf-8").splitlines()}
    assert len(ids) == 34 and not ids & test_ids
    pm = read_json(s3 / "sports_valid_1k" / "pilot_mirror.json")
    assert (pm["panel_type"], pm["base_question"], pm["n_users"]) == ("next_item", "next", 34)
    assert pm["ref_ranks"]["n_joined"] == 0          # the C-CRP v3 reference ranks cover the TEST events only
    for d in ("ml1m", "toys"):
        pm = read_json(s3 / d / "pilot_mirror.json")
        assert (pm["panel_type"], pm["base_question"]) == ("rated", "like")
        assert {"mirror", "placebo", "evidence", "pmi_nohist"} <= set(pm["arms"]) and "ensemble_null" in pm


def test_dry_run_input_record(e2e):
    rec = e2e["inputs1"]
    assert rec["v_star"] == "V3" and rec["model"] == e2e["model"]
    assert rec["constants"] == {"toys_min_fresh": 500, "n_second": 100, "n_sports": 34}   # fake_stage3's scale
    s = rec["second"]
    assert (s["domain"], s["n_users"], s["toys_confirm_built"]) == ("toys", 100, True)
    assert s["toys_fresh_eligible"] >= 500
    sp = rec["sports_valid"]
    assert sp["n_events"] == 34 and sp["quarantine_checked"] is True and sp["source"] == VALID_REL
    assert sp["rows_sha1"] == sha1_file(e2e["s3"] / "panels" / "sports_valid_1k.jsonl")
    assert rec["recorded_stage3_gate"]["rated_component_ok"] and rec["recorded_stage3_gate"]["next_item_component_ok"]


def test_dry_run_second_run_skips_everything_and_touches_nothing(e2e):
    r = e2e["r2"]
    assert r.returncode == 0, tail(r)
    assert e2e["snap1"] and e2e["snap2"] == e2e["snap1"]
    for d in ("ml1m", "ml1m_nohist", "toys", "toys_nohist", "sports_valid_1k"):
        assert f"[skip] {S3_REL}/{d}: scored" in r.stdout, d
    for d in ("ml1m", "toys", "sports_valid_1k"):
        assert f"[skip] {S3_REL}/{d}/pilot_mirror.json" in r.stdout, d
    assert r.stdout.count(": scored") == 5 and "scores chunk" not in r.stdout and "moved aside" not in r.stdout
    calls = e2e["calls2"]
    assert "-m src.confrec.pyes_scorer" not in calls and "-m src.confrec.pilot_mirror" not in calls
    assert e2e["decision2"] == e2e["decision1"] and e2e["inputs2"] == e2e["inputs1"]


def test_dry_run_video_games_fallback_builds_its_panel_and_reuses_the_others(e2e):
    r, s3 = e2e["r3"], e2e["s3"]
    assert r.returncode == 0, tail(r)
    s = e2e["inputs3"]["second"]
    assert (s["domain"], s["n_users"]) == ("games", 100) and "< 1000000000" in s["reason"]
    gp = s3 / "panels" / "games_first100_h20.jsonl"
    meta = read_json(gp.with_suffix(".meta.json"))
    assert (meta["domain"], meta["hist_len"], meta["n_users"], meta["seed"], meta["n_cands"], meta["min_hist"],
            meta["min_like"], meta["min_dislike"], meta.get("gatefix_fields")) == ("games", 20, 100, 0, 20, 3, 3, 3, True)
    rows = [json.loads(x) for x in gp.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 100 and all(x["domain_kind"] == "product" and "history_meta" in x for x in rows)
    for d, args in (("games", "--questions like,dislike,like_para --swap_k 8"), ("games_nohist",
                                                                                  "--questions like --hist_len 0")):
        key = (s3 / d / "run.key").read_text(encoding="utf-8").strip()
        assert key == f"{sha1_file(gp)} {e2e['model']} --variant V3 --readout yesno {args}", d
    for d in ("ml1m", "ml1m_nohist", "sports_valid_1k"):
        assert f"[skip] {S3_REL}/{d}: scored" in r.stdout, d
    scorer = [c for c in e2e["calls3"].splitlines() if c.startswith("-m src.confrec.pyes_scorer")]
    assert len(scorer) == 2 and all("games_first100_h20.jsonl" in c for c in scorer)
    dec = e2e["decision3"]
    assert dec["inputs"]["toys"] == f"{S3_REL}/games/pilot_mirror.json" and dec["gate"]["pass"] is True
    assert dec["decision"] in gate.LABELS and dec["decision"] != "INCOMPLETE"
