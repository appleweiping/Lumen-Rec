"""Tests of scripts/sigir/run_ftgrid.sh (F3 of docs/sigir/FTGRID_IMPL_SPEC.md: the Amendment-3 fine-tuned program runner)
and scripts/sigir/starperm_panel.py. CPU only, deterministic, no network, no GPU, no model.

  * the script is LF, `bash -n` clean, and passes the integration audit of tests/test_confrec_contracts.py;
  * every flag it passes to a repo entry point exists in that entry point's real argparse (a stricter audit than the
    integration one: heredocs ignored, arrays and appended arrays and score()'s fixed flags expanded, literal values
    checked against choices and types), and the audit catches a wrong flag;
  * the DRY_RUN stand-ins: the trainer uses train_lora_yesno's real argparse and panel check (also without torch), the
    scorer is pyes_scorer's real code with a fake model;
  * DRY_RUN=1 end to end in a copy of the repo, real code everywhere except those stand-ins: ml1m (Gate-FT adapters by
    path, the identical Gate-FT like passes adopted, FT-C adapters, every decomposition arm, the freeze record, the
    report), its resume run (nothing rerun, no file touched), a changed run key, toys (trained adapters, the knockout
    arms), games (the zero-shot-only branch of a failed Gate-FT, the E1 rerun-once rule), the stage-order and freeze
    refusals, the second backbone, the input guards;
  * the produced files equal the layout block of the spec;
  * starperm_panel.py writes diag_battery.make_starperm's copies, one file per copy, over the variant's window.
ftgrid_report.py (F2) is a stub here while it does not exist (the stub lists the score directories it finds);
test_real_cli_dry_run runs the chain with the real F2 once it exists. ftgrid_data.py (F1) is the real one.
"""
from __future__ import annotations

import argparse
import gzip
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
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.confrec import diag_battery as db
from src.confrec import ftgrid_freeze as ff
from src.confrec import pyes_scorer as ps

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sigir" / "run_ftgrid.sh"
STARPERM = ROOT / "scripts" / "sigir" / "starperm_panel.py"
SPEC = ROOT / "docs" / "sigir" / "FTGRID_IMPL_SPEC.md"
F1 = ROOT / "src" / "confrec" / "ftgrid_data.py"
F2 = ROOT / "src" / "confrec" / "ftgrid_report.py"
DRY_ROOT = "outputs/confrec/ftgrid_dryrun"          # the DRY_RUN default OUT_ROOT
GT_DRY = f"{DRY_ROOT}/_dry/gateft"                    # the DRY_RUN Gate-FT context
QWEN, LLAMA = "dryrun/Qwen3-8B", "dryrun/Llama-3.1-8B-Instruct"
DECOMP = ["like", "swap", "nohist", "starperm0", "starperm1"]
KNOCK = ["pseudo", "placebo"]

_spec = importlib.util.spec_from_file_location("starperm_panel_under_test", STARPERM)
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

F2_STUB = '''"""TEST STUB of src.confrec.ftgrid_report (F2), written by tests/test_confrec_ftgrid_run.py: the CLI of
docs/sigir/FTGRID_IMPL_SPEC.md; it computes no endpoint and lists the score directories it finds."""
import argparse
import json
from pathlib import Path


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    for k in ("--domain", "--split", "--panels", "--scores_root", "--models", "--raw", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refs", default=None)
    a = ap.parse_args(argv)
    split = json.loads(Path(a.split).read_text(encoding="utf-8"))
    root, models = Path(a.scores_root), a.models.split(",")
    root = root / a.domain if (root / a.domain).is_dir() else root      # scores/ (or scores/<d>/), as F2 reads it
    found = {m: sorted(p.name for p in (root / m).iterdir() if (p / "report.json").is_file() and "." not in p.name)
             if (root / m).is_dir() else [] for m in models}
    out = {"stub": True, "domain": a.domain, "split_domain": split["domain"], "models": models, "arms_found": found,
           "panels": sorted(p.name for p in Path(a.panels).iterdir()), "raw_is_dir": Path(a.raw).is_dir(),
           "n_boot": a.n_boot, "seed": a.seed}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    Path(a.out).with_name(Path(a.out).stem + "_tables.csv").write_text(
        "model,arm\\n" + "".join(f"{m},{x}\\n" for m in models for x in found[m]), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------- helpers
def _bash():
    for c in ("C:/Program Files/Git/bin/bash.exe", shutil.which("bash")):
        if c and Path(c).exists() and "system32" not in str(c).lower():   # not WSL's bash.exe
            return str(c)
    return None


BASH = _bash()
NO_BASH = "no bash (Git Bash or a POSIX bash) on PATH"
NO_F1 = "src/confrec/ftgrid_data.py (F1) does not exist"
needs_chain = pytest.mark.skipif(BASH is None or not F1.exists(), reason=NO_BASH if BASH is None else NO_F1)


def core_files() -> list:
    return ff.parse_blocks((ROOT / ff.AMENDMENT).read_text(encoding="utf-8"))["core"]


def make_repo(dest: Path, real_f2: bool = False) -> Path:
    """A copy of the repo parts the chain runs: src/confrec, the two F3 scripts and every file of the A3 core freeze
    list (a listed file that does not exist yet becomes a placeholder, so the freeze check can hash it). F2 is the
    stub unless real_f2."""
    (dest / "src" / "confrec").mkdir(parents=True)
    shutil.copy2(ROOT / "src" / "__init__.py", dest / "src" / "__init__.py")
    for f in (ROOT / "src" / "confrec").glob("*.py"):
        shutil.copy2(f, dest / "src" / "confrec" / f.name)
    for rel in ["scripts/sigir/run_ftgrid.sh", "scripts/sigir/starperm_panel.py", ff.AMENDMENT] + core_files():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        if (ROOT / rel).exists():
            shutil.copy2(ROOT / rel, dest / rel)
        elif not (dest / rel).exists():
            (dest / rel).write_text("# placeholder: a file the amendment lists that does not exist yet\n",
                                    encoding="utf-8", newline="\n")
    if not real_f2:
        (dest / "src" / "confrec" / "ftgrid_report.py").write_text(F2_STUB, encoding="utf-8", newline="\n")
    return dest


def run_script(repo: Path, domain: str, dry: bool = True, **env) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in (
        "MODEL", "OUT_ROOT", "VARIANT", "STAGES", "DRY_RUN", "DRY_GATE", "DRY_E1_FAIL", "KNOCKOUT_SPORTS", "PYTHONPATH")}
    e.update(PYTHON=sys.executable.replace("\\", "/"), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if dry:
        e["DRY_RUN"] = "1"
    e.update(env)
    script = (repo / "scripts" / "sigir" / "run_ftgrid.sh").as_posix()
    return subprocess.run([BASH, script, domain], env=e, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=3600)


def tail(r: subprocess.CompletedProcess, n: int = 4000) -> str:
    return f"rc={r.returncode}\n--- stdout ---\n{r.stdout[-n:]}\n--- stderr ---\n{r.stderr[-n:]}"


def trained(r: subprocess.CompletedProcess) -> list:
    """The adapters the run's stand-in trainer wrote under OUT_ROOT/adapters (not the DRY_RUN Gate-FT context)."""
    return re.findall(r"^dry-run trainer: \S*[\\/]adapters[\\/][a-z0-9]+[\\/](\w+) ", r.stdout, re.M)


def snapshot(root: Path) -> dict:
    """{relative path: (size, mtime_ns, sha1)} of every file under root except the DRY_RUN inputs (_dry)."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "_dry" not in p.relative_to(root).parts:
            st = p.stat()
            out[p.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns, hashlib.sha1(p.read_bytes()).hexdigest())
    return out


def sha1_file(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def csv_header(path) -> list:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.readline().strip().split(",")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the layout block of the spec
def spec_layout() -> dict:
    """Panel files (with their domain condition), user-id lists, models and arms named by the spec's layout block."""
    text = SPEC.read_text(encoding="utf-8")
    block = re.search(r"## Directory layout.*?```\n(.*?)```", text, re.S).group(1)
    panels = {}
    for line in block.split("adapters/")[0].splitlines()[1:]:
        names = re.findall(r"\b([a-z_0-9]+\.jsonl?)\b", line.split("  ")[1] if line.startswith("  ") else "")
        cond = "ml1m" if "ml1m only" in line else ("toys_games" if "toys and games only" in line else "all")
        panels.update({n: cond for n in names})
    arms = re.sub(r"\([^)]*\)", "", re.search(r"arm in \{([^}]*)\}", block, re.S).group(1))
    return {"panels": panels, "user_lists": re.findall(r"`([a-z_]+_users\.txt)`", text),
            "models": [m.strip() for m in re.search(r"model in \{([^}]*)\}", block).group(1).split(",")],
            "arms": [a.strip() for a in arms.split(",")]}


def expected_panels(domain: str) -> set:
    lay = spec_layout()
    keep = {"all": True, "ml1m": domain == "ml1m", "toys_games": domain in ("toys", "games")}
    return {n for n, c in lay["panels"].items() if keep[c]} | set(lay["user_lists"])


def test_the_spec_layout_block_parses_as_expected():
    lay = spec_layout()
    assert lay["arms"] == ["like", "swap", "nohist", "starperm0", "starperm1", "pseudo", "placebo"]
    assert lay["models"] == ["zeroshot", "s0", "s1", "s2", "p0", "p1"]
    assert lay["panels"]["train_perm.jsonl"] == "ml1m" and lay["panels"]["eval_placebo.jsonl"] == "toys_games"
    assert lay["user_lists"] == ["train_users.txt", "eval_users.txt", "sd_users.txt"]
    assert expected_panels("sports") == {"train.jsonl", "eval.jsonl", "eval_sd_test.jsonl", "ftgrid_split.json",
                                         "eval_sd_test_starperm0.jsonl", "eval_sd_test_starperm1.jsonl",
                                         "train_users.txt", "eval_users.txt", "sd_users.txt"}


def check_layout(root: Path, domain: str, models_arms: dict, adapters: set) -> None:
    """panels/D, adapters/D, scores/D/<model>/<arm> and report/ of one domain equal the spec (aside dirs *.stale.* and
    *.e1fail.* may sit next to an arm)."""
    assert {p.name for p in (root / "panels" / domain).iterdir()} == expected_panels(domain)
    if adapters:
        assert {p.name for p in (root / "adapters" / domain).iterdir()} == adapters
    else:
        assert not (root / "adapters" / domain).exists()
    lay = spec_layout()
    got = {m.name: {a.name for a in m.iterdir() if "." not in a.name} for m in (root / "scores" / domain).iterdir()}
    assert got == {m: set(a) for m, a in models_arms.items()}
    for m, arms in got.items():
        assert m in lay["models"] and arms <= set(lay["arms"])
        for a in arms:
            d = root / "scores" / domain / m / a
            names = {p.name for p in d.iterdir()}
            assert {"run.key", "report.json", "scores.csv.gz"} <= names, (d, names)
            assert ("swap_prior.csv.gz" in names) == (a == "swap"), (d, names)
            assert csv_header(d / "scores.csv.gz") == ps.SCORE_COLS
            if a == "swap":
                assert csv_header(d / "swap_prior.csv.gz") == ps.SWAP_COLS
    assert {p.name for p in (root / "report").iterdir()} >= {f"{domain}.json", f"{domain}_tables.csv"}


def arm_panel(root: Path, domain: str, arm: str) -> Path:
    p = root / "panels" / domain
    return {"like": p / "eval.jsonl", "swap": p / "eval_sd_test.jsonl", "nohist": p / "eval_sd_test.jsonl",
            "starperm0": p / "eval_sd_test_starperm0.jsonl", "starperm1": p / "eval_sd_test_starperm1.jsonl",
            "pseudo": p / "eval_pseudo.jsonl", "placebo": p / "eval_placebo.jsonl"}[arm]


def check_scoring_dir(repo: Path, domain: str, model: str, arm: str, lora=None, root_rel: str = DRY_ROOT,
                      variant: str = "V3", backbone: str = QWEN) -> dict:
    """run.key = panel sha1, model, variant, adapter-weights sha1 ('-' zero-shot), args; report.json scored that panel
    with that adapter (the repo-relative path the script passes) and the registered settings and the arm's window."""
    root = repo / root_rel
    d = root / "scores" / domain / model / arm
    panel = arm_panel(root, domain, arm)
    key = (d / "run.key").read_text(encoding="utf-8").split()
    assert key[:3] == [sha1_file(panel), backbone, variant], key
    extra = {"swap": ["--swap_k", "8"], "nohist": ["--hist_len", "0"]}.get(arm, [])
    if lora is None:
        assert key[3:] == ["-"] + extra, key
    else:
        weights = b"".join(w.read_bytes() for w in sorted((repo / lora).glob("adapter_model.*")))
        assert key[3:] == [hashlib.sha1(weights).hexdigest(), "--lora", lora] + extra, key
    rep = read_json(d / "report.json")
    cfg = rep["config"]
    assert cfg["data_sha1"] == sha1_file(panel) and cfg["variant"] == variant and cfg["readout"] == "yesno"
    assert cfg["questions"] == ["like"] and cfg["dtype"] == "float16" and cfg["topk_logprobs"] == 50
    assert cfg["max_model_len"] == 4096 and cfg["model"] == backbone and cfg["lora"] == lora
    assert cfg["hist_len"] == (0 if arm == "nohist" else 20) and cfg["swap_k"] == (8 if arm == "swap" else 0)
    return rep


# ---------------------------------------------------------------- 1. the script file
def test_script_is_lf_bash_and_bash_n_clean():
    raw = SCRIPT.read_bytes()
    assert b"\r\n" not in raw and raw.startswith(b"#!/usr/bin/env bash\n")
    assert b"set -euo pipefail" in raw
    if BASH is None:
        pytest.skip(NO_BASH)
    r = subprocess.run([BASH, "-n", SCRIPT.as_posix()], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_script_is_in_the_amendment_core_freeze_list():
    assert "scripts/sigir/run_ftgrid.sh" in core_files()


# ---------------------------------------------------------------- 2. every flag exists in the real argparse
class _Captured(Exception):
    pass


def _capture(calls):
    orig = (argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args)

    def boom(self, *a, **k):
        raise _Captured(self)
    argparse.ArgumentParser.parse_args = argparse.ArgumentParser.parse_known_args = boom
    old = sys.argv
    sys.argv = ["prog"]
    try:
        for call in calls:
            try:
                call()
            except _Captured as c:
                return c.args[0]
            except (SystemExit, Exception):
                continue
    finally:
        argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args = orig
        sys.argv = old
    return None


def _spec_parser(module: str):
    """The documented CLI of the spec's interface section (for an entry point that does not exist yet)."""
    line = re.search(r"`python -m " + re.escape(module) + r" ([^`]*)`", SPEC.read_text(encoding="utf-8")).group(1)
    ap = argparse.ArgumentParser()
    optional = set(re.findall(r"\[(--[a-z_0-9]+)", line))
    for flag in dict.fromkeys(re.findall(r"(--[a-z_0-9]+)", line)):
        ap.add_argument(flag, required=flag not in optional)
    return ap


@contextmanager
def stub_torch():
    """train_lora_yesno imports torch at module level, its argparse parser needs none of it (and these tests must not
    need torch). Inside the block torch resolves to stand-in modules and train_lora_yesno / lora_trainer are imported
    afresh against them; sys.modules and the package attributes are restored afterwards."""
    import src.confrec as pkg
    mods = ("src.confrec.train_lora_yesno", "src.confrec.lora_trainer")
    torch_names = ("torch", "torch.utils", "torch.utils.data", "torch.nn", "torch.nn.functional")
    saved = {k: sys.modules.get(k) for k in mods + torch_names}
    attrs = {m.rsplit(".", 1)[1]: getattr(pkg, m.rsplit(".", 1)[1], None) for m in mods}
    try:
        for k in torch_names:
            sys.modules[k] = types.ModuleType(k)
        sys.modules["torch.utils.data"].Dataset = object
        for k in mods:
            sys.modules.pop(k, None)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for a, v in attrs.items():
            if v is not None:
                setattr(pkg, a, v)
            elif hasattr(pkg, a):
                delattr(pkg, a)


def _parser(target: str):
    """The real argparse parser of 'src.confrec.x' or 'scripts/sigir/y.py' (the spec's documented CLI for an entry
    point that does not exist yet)."""
    if target.endswith(".py"):
        spec = importlib.util.spec_from_file_location("audit_" + Path(target).stem, ROOT / target)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    elif not (ROOT / (target.replace(".", "/") + ".py")).exists():
        return _spec_parser(target)
    elif target == "src.confrec.train_lora_yesno":
        with stub_torch():
            mod = importlib.import_module(target)
            return _capture([lambda: mod.main()])
    else:
        mod = importlib.import_module(target)
    p = _capture([lambda: mod.main(), lambda: mod.parse_args(), lambda: mod.main([])])
    assert p is not None, f"no argparse parser captured for {target}"
    return p


def _code(text: str) -> str:
    """The shell code: heredoc bodies blanked (they are python), continuation lines joined."""
    out, end = [], None
    for ln in text.splitlines():
        if end is not None:
            out.append("")
            if ln == end:
                end = None
            continue
        out.append(ln)
        m = re.search(r"(?<!<)<<-?\s*'([A-Za-z_]+)'", ln)
        if m:
            end = m.group(1)
    return re.sub(r"\\\n\s*", " ", "\n".join(out))


def _flags(words: list, arrays: dict) -> list:
    """[(flag, value or None)] of shell words, every ${NAME[@]} expanded from every definition of NAME."""
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


_CUT = re.compile(r"\s(?:\|\||\||&&|;|>|2>&1|2>)\s|;\s*(?:then|do)\b|;\s*$")


def _words(rest: str) -> list:
    return shlex.split(re.sub(r"\)\s*$", "", _CUT.split(rest)[0]))


def invocations(text: str) -> list:
    """(target, [(flag, value)], line) of every call of a repo entry point through "$PY" / "$MPY"; a `score` line
    carries the flags of score()'s own pyes_scorer call."""
    code = _code(text)
    arrays = defaultdict(list)
    for name, body in re.findall(r"(?<![\w$])([A-Za-z_][A-Za-z_0-9]*)\+?=\(([^)]*)\)", code, re.S):
        arrays[name].append(body)
    calls, fixed = [], []
    lines = [ln.strip() for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    for s in lines:
        m = re.search(r'"\$(?:PY|MPY)" (?:-m (src\.confrec\.[a-z_0-9]+)|(scripts/sigir/[a-z_0-9]+\.py))(.*)$', s)
        if m:
            target, flags = m.group(1) or m.group(2), _flags(_words(m.group(3)), arrays)
            calls.append((target, flags, s))
            if target == "src.confrec.pyes_scorer":
                fixed = flags
    for s in lines:
        if re.match(r"score\s", s):
            calls.append(("src.confrec.pyes_scorer", fixed + _flags(_words(s)[3:], arrays), s))
    return calls


def audit(text: str) -> tuple[list, dict]:
    problems, parsers, count = [], {}, defaultdict(int)
    for target, flags, line in invocations(text):
        count[target] += 1
        if target not in parsers:
            parsers[target] = _parser(target)
            assert parsers[target] is not None, f"no argparse parser captured for {target}"
        acts = {s: a for a in parsers[target]._actions for s in a.option_strings}
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


def test_every_flag_the_script_passes_exists_in_the_real_argparse():
    problems, count = audit(SCRIPT.read_text(encoding="utf-8"))
    assert not problems, "\n".join(problems)
    # every entry point of the chain was seen (an audit that finds nothing cannot pass)
    for target, n in {"src.confrec.pyes_scorer": 8, "src.confrec.train_lora_yesno": 1, "src.confrec.ftgrid_data": 1,
                      "src.confrec.ftgrid_freeze": 5, "src.confrec.build_rated_panels": 1,
                      "src.confrec.pseudonymize": 1, "scripts/sigir/starperm_panel.py": 1,
                      "src.confrec.ftgrid_report": 1}.items():
        assert count.get(target, 0) >= n, (target, count)


def test_the_flag_audit_catches_a_wrong_flag_and_a_wrong_value():
    bad = SCRIPT.read_text(encoding="utf-8").replace("--swap_k 8", "--swapk 8").replace(
        "--source amazon", "--source amazn").replace('--dev_users_sha1 "$DEV_SHA"', '--dev_user_sha1 "$DEV_SHA"')
    bad = bad.replace("--max_len \"$MAXLEN\"", "--maxlen \"$MAXLEN\"")
    problems, _ = audit(bad)
    assert any("--swapk" in p for p in problems) and any("'amazn'" in p for p in problems)
    assert any("--dev_user_sha1" in p for p in problems) and any("--maxlen" in p for p in problems)


def test_the_integration_audit_of_test_confrec_contracts_accepts_the_script():
    spec = importlib.util.spec_from_file_location("contracts_for_ftgrid_run", ROOT / "tests" / "test_confrec_contracts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert SCRIPT in sorted((ROOT / "scripts" / "sigir").glob("*.sh"))   # found by its glob: no registration needed
    with stub_torch():                                                   # its train_lora_yesno parser, without torch
        mod.test_every_script_flag_exists_in_the_real_argparse(SCRIPT)


# ---------------------------------------------------------------- 3. the DRY_RUN stand-ins
def fakes_source() -> str:
    return re.search(r"<<'PYFAKES'\n(.*?)\nPYFAKES\n", SCRIPT.read_text(encoding="utf-8"), re.S).group(1) + "\n"


def rated_rows(n: int = 6, hist: int = 20, n_c: int = 6, seed: int = 0) -> list:
    rng = random.Random(seed)
    rows = []
    for u in range(n):
        hr = [float(rng.randint(1, 5)) for _ in range(hist)]
        ht = [f"Movie {u}-{k}" for k in range(hist)]
        lab = [k % 2 for k in range(n_c)]
        rows.append({"user_id": f"u{u}", "source_event_id": f"u{u}::{1000 + u}",
                     "history": [f"{t} (rated {int(r)}/5)" for t, r in zip(ht, hr)],
                     "history_item_ids": [f"h{u}_{k}" for k in range(hist)], "history_titles": ht,
                     "history_ratings": hr, "history_brands": [""] * hist, "history_meta": [""] * hist,
                     "candidate_item_ids": [f"c{u}_{k}" for k in range(n_c)],
                     "candidate_titles": [f"Candidate {u}-{k}" for k in range(n_c)], "candidate_texts": [""] * n_c,
                     "candidate_brands": [""] * n_c, "candidate_ratings": [4.0 if y else 2.0 for y in lab],
                     "candidate_labels": lab, "candidate_popularity": [3] * n_c,
                     "candidate_popularity_prior": [1] * n_c, "candidate_timestamps": [2000 + k for k in range(n_c)],
                     "source": "ml1m", "domain_kind": "movie"})
    return rows


def write_rows(path: Path, rows: list) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return path


def test_fake_trainer_uses_the_real_argparse_and_panel_check_without_torch(tmp_path):
    """The stand-in runs on a box without torch, refuses an unknown flag like the real trainer (exit 2, nothing
    written) and writes a train_config.json that pyes_scorer --lora accepts for the trained window and for 0."""
    fakes = tmp_path / "ftgrid_fakes.py"
    fakes.write_text(fakes_source(), encoding="utf-8")
    train = write_rows(tmp_path / "train.jsonl", rated_rows())
    block = ("import importlib.abc, runpy, sys\n"
             "class NoTorch(importlib.abc.MetaPathFinder):\n"
             "    def find_spec(self, name, path=None, target=None):\n"
             "        if name == 'torch' or name.startswith('torch.'):\n"
             "            raise ImportError('torch is blocked by the test')\n"
             "sys.meta_path.insert(0, NoTorch())\n"
             "sys.argv = sys.argv[1:]\n"
             "runpy.run_path(sys.argv[0], run_name='__main__')\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    base = [sys.executable, "-c", block, str(fakes), "trainer", "--train", str(train), "--model", QWEN]
    ok = subprocess.run(base + ["--out", str(tmp_path / "a"), "--variant", "V3", "--seed", "1", "--bsz", "4",
                                "--grad_accum", "8", "--max_len", "1280"], capture_output=True, text=True, env=env,
                        cwd=ROOT)
    assert ok.returncode == 0, ok.stderr
    cfg = read_json(tmp_path / "a" / "train_config.json")
    assert cfg["variant"] == "V3" and cfg["hist_len_used"] == 20 and cfg["seed"] == 1 and cfg["dry_run"] is True
    assert (cfg["bsz"], cfg["grad_accum"], cfg["max_len"], cfg["lora_r"], cfg["lr"]) == (4, 8, 1280, 16, 1e-4)
    assert (tmp_path / "a" / "adapter_model.safetensors").is_file()
    ps.check_lora_variant(tmp_path / "a", "V3", 20)
    ps.check_lora_variant(tmp_path / "a", "V3", 0)
    with pytest.raises(SystemExit):
        ps.check_lora_variant(tmp_path / "a", "V0", 10)
    bad = subprocess.run(base + ["--out", str(tmp_path / "b"), "--bogus", "1"], capture_output=True, text=True,
                         env=env, cwd=ROOT)
    assert bad.returncode == 2 and "--bogus" in bad.stderr and not (tmp_path / "b").exists()


def test_fake_scorer_is_the_real_scorer_with_a_fake_model(tmp_path, monkeypatch):
    (tmp_path / "f.py").write_text(fakes_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ftgrid_fakes_under_test", tmp_path / "f.py")
    fk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fk)
    panel = write_rows(tmp_path / "p.jsonl", rated_rows(n=5))
    argv = ["--data", str(panel), "--model", QWEN, "--dtype", "float16", "--topk_logprobs", "50", "--max_model_len",
            "4096", "--chunk_users", "2", "--variant", "V0", "--readout", "yesno", "--questions", "like"]
    fk.scorer(argv + ["--output", str(tmp_path / "a"), "--swap_k", "3"])
    fk.scorer(argv + ["--output", str(tmp_path / "b")])
    rep = read_json(tmp_path / "a" / "report.json")
    assert rep["scorer"] == ps.SCORER and rep["n_overlength"] == 0 and rep["mean_yes_no_mass"] > 0.95
    assert rep["censored_main"]["0"] == rep["n_main_prompts"] == 30
    assert rep["swap_k"] == 3 and rep["swap_prompts"] == 30 * 3          # every item, 3 donors without it
    assert csv_header(tmp_path / "a" / "scores.csv.gz") == ps.SCORE_COLS
    assert csv_header(tmp_path / "a" / "swap_prior.csv.gz") == ps.SWAP_COLS
    with gzip.open(tmp_path / "a" / "scores.csv.gz", "rt") as fa, gzip.open(tmp_path / "b" / "scores.csv.gz", "rt") as fb:
        assert fa.read() == fb.read()                                    # deterministic logits
    monkeypatch.setenv("DRY_E1_FAIL", "c_fail")
    fk.scorer(argv + ["--output", str(tmp_path / "c_fail")])
    assert abs(read_json(tmp_path / "c_fail" / "report.json")["mean_yes_no_mass"] - 0.5) < 1e-6


# ---------------------------------------------------------------- 4. starperm_panel.py
def test_starperm_panel_writes_diag_battery_copies_one_file_per_copy(tmp_path):
    rows = rated_rows(n=7)
    panel = write_rows(tmp_path / "eval_sd_test.jsonl", rows)
    meta = sp.main(["--panel", str(panel), "--variant", "V0", "--out_prefix", str(tmp_path / "eval_sd_test_starperm"),
                    "--meta", str(tmp_path / "m.json")])
    ref, _ = db.make_starperm(rows, k=2, seed=0, hist_len=10)           # V0 renders the last 10 events
    for k in (0, 1):
        got = db.read_jsonl(tmp_path / f"eval_sd_test_starperm{k}.jsonl")
        assert got == [json.loads(json.dumps(r)) for r in ref if r["perm_k"] == k]
        assert [r["perm_of"] for r in got] == [r["source_event_id"] for r in rows]
        assert [r["source_event_id"] for r in got] == [f"{r['source_event_id']}::perm{k}" for r in rows]
        for r, o in zip(got, rows):
            assert r["history"][:10] == o["history"][:10]                # outside the rendered window: unchanged
            assert r["history_titles"] == o["history_titles"] and r["candidate_labels"] == o["candidate_labels"]
            stars = [int(re.search(r"\(rated (\d)/5\)$", h).group(1)) for h in r["history"]]
            assert stars == [int(x) for x in r["history_ratings"]]      # the ratings follow the suffixes
            assert sorted(stars[10:]) == sorted(int(x) for x in o["history_ratings"][10:])
            assert r["perm_window"] == 10 and sorted(r["perm_sigma"]) == list(range(10))
            assert all(s != j for j, s in enumerate(r["perm_sigma"]))    # a derangement
    first = (tmp_path / "eval_sd_test_starperm1.jsonl").read_bytes()
    sp.main(["--panel", str(panel), "--variant", "V0", "--out_prefix", str(tmp_path / "eval_sd_test_starperm")])
    assert (tmp_path / "eval_sd_test_starperm1.jsonl").read_bytes() == first     # seeded: byte-identical rerun
    m = read_json(tmp_path / "m.json")
    assert m == json.loads(json.dumps(meta)) and (m["k"], m["seed"], m["hist_len"], m["variant"]) == (2, 0, 10, "V0")
    assert m["outputs"]["eval_sd_test_starperm1.jsonl"] == hashlib.sha1(first).hexdigest()
    assert m["panel_sha1"] == sha1_file(panel) and m["panel"] == "eval_sd_test.jsonl" and "\\" not in json.dumps(m)


def test_starperm_window_is_the_variants_registered_window(tmp_path):
    rows = rated_rows(n=4)
    panel = write_rows(tmp_path / "p.jsonl", rows)
    for variant, window in (("V0", 10), ("V3", 20), ("V7", 20), ("V1", 10)):
        sp.main(["--panel", str(panel), "--variant", variant, "--out_prefix", str(tmp_path / variant)])
        got = db.read_jsonl(tmp_path / f"{variant}0.jsonl")
        assert {r["perm_window"] for r in got} == {window}
        assert all(r["history"][:20 - window] == o["history"][:20 - window] for r, o in zip(got, rows))
    nxt = [{k: v for k, v in r.items() if k != "history_ratings"} for r in rows]   # a next-item panel
    with pytest.raises(SystemExit, match="rated"):
        sp.build(nxt, "V0")


# ---------------------------------------------------------------- 5. DRY_RUN chains
def _chain(tmp_path_factory, name: str, domain: str, **env) -> dict:
    if BASH is None:
        pytest.skip(NO_BASH)
    if not F1.exists():
        pytest.skip(NO_F1)
    repo = make_repo(tmp_path_factory.mktemp(name) / "repo")
    r = run_script(repo, domain, **env)
    assert r.returncode == 0, tail(r)
    return {"repo": repo, "root": repo / DRY_ROOT, "r": r}


@pytest.fixture(scope="module")
def ml1m(tmp_path_factory):
    """The whole chain on ml1m (STAGES all,perm: the DRY_RUN default there), then the same command a second time."""
    c = _chain(tmp_path_factory, "ml1m", "ml1m")
    c["s1"] = snapshot(c["root"])
    c["r2"] = run_script(c["repo"], "ml1m")
    assert c["r2"].returncode == 0, tail(c["r2"])
    c["s2"] = snapshot(c["root"])
    return c


def test_ml1m_chain_layout_equals_the_spec(ml1m):
    check_layout(ml1m["root"], "ml1m", {"zeroshot": DECOMP, "s0": DECOMP, "s1": DECOMP, "s2": DECOMP,
                                        "p0": ["like"], "p1": ["like"]}, {"s0", "s1", "s2", "p0", "p1"})
    rep = read_json(ml1m["root"] / "report" / "ml1m.json")
    assert rep["models"] == ["zeroshot", "s0", "s1", "s2", "p0", "p1"] and rep["split_domain"] == "ml1m"
    assert rep["raw_is_dir"] and (rep["n_boot"], rep["seed"]) == (200, 0)


def test_ml1m_scores_the_gate_ft_adapters_by_path_and_adopts_their_identical_like_passes(ml1m):
    repo, root, gt = ml1m["repo"], ml1m["root"], ml1m["repo"] / GT_DRY
    for k in range(3):
        a = root / "adapters" / "ml1m" / f"s{k}"
        assert read_json(a / "train_config.json") == read_json(gt / "adapters" / f"s{k}" / "train_config.json")
        for arm in DECOMP:
            check_scoring_dir(repo, "ml1m", f"s{k}", arm, f"{GT_DRY}/adapters/s{k}")
        like = root / "scores" / "ml1m" / f"s{k}" / "like"
        assert (like / "adopted_from").read_text(encoding="utf-8").strip() == f"{GT_DRY}/scores/s{k}"
        assert (like / "scores.csv.gz").read_bytes() == (gt / "scores" / f"s{k}" / "scores.csv.gz").read_bytes()
    assert (root / "scores" / "ml1m" / "zeroshot" / "like" / "adopted_from").read_text(
        encoding="utf-8").strip() == f"{GT_DRY}/scores/zeroshot"
    for arm in DECOMP:
        check_scoring_dir(repo, "ml1m", "zeroshot", arm)
    out = ml1m["r"].stdout + ml1m["r"].stderr
    assert out.count("already complete:") == 4 and "[not adopted]" not in out
    assert trained(ml1m["r"]) == ["p0", "p1"]                           # s0-s2 are never trained here


def test_ml1m_ft_c_adapters_follow_the_split_recipe_and_only_their_like_pass_is_scored(ml1m):
    repo, root = ml1m["repo"], ml1m["root"]
    split = read_json(root / "panels" / "ml1m" / "ftgrid_split.json")
    ov = split["train"]["overlength"]
    assert ov["share_above_1024"] is not None and split["gateft_T_match"] is True   # what the freeze requires
    for k in (0, 1):
        cfg = read_json(root / "adapters" / "ml1m" / f"p{k}" / "train_config.json")
        assert Path(cfg["train"]).name == "train_perm.jsonl" and cfg["seed"] == k and cfg["variant"] == "V3"
        assert (cfg["bsz"], cfg["grad_accum"], cfg["max_len"]) == (ov["micro_bsz"], ov["grad_accum"], ov["max_len_used"])
        assert cfg["bsz"] * cfg["grad_accum"] == 32
        rep = check_scoring_dir(repo, "ml1m", f"p{k}", "like", f"{DRY_ROOT}/adapters/ml1m/p{k}")
        assert not (root / "scores" / "ml1m" / f"p{k}" / "like" / "adopted_from").exists()
        assert rep["n_users"] == split["eval"]["users"]


def test_ml1m_freeze_rehearsal_record_and_marker(ml1m):
    root, out = ml1m["root"], ml1m["r"].stdout
    assert "[dry] freeze rehearsal" in out and "freeze check OK (stage core)" in out
    assert "freeze check OK (stage amendment)" in out                   # before the FT-C training
    p0 = re.search(r"dry-run trainer: \S*adapters[\\/]ml1m[\\/]p0", out).start()
    assert out.index("freeze check OK (stage amendment)") < p0
    assert out.index("freeze check OK (stage core)") < out.index("== stage 3")
    log = (root / "_dry" / "PILOT_LOG.md").read_text(encoding="utf-8").lower()
    for rel in core_files() + [f"{DRY_ROOT}/panels/ml1m/ftgrid_split.json"]:
        assert sha1_file(ml1m["repo"] / rel) in log, rel
    mark = (root / "freeze" / "ml1m.core.ok").read_text(encoding="utf-8")
    assert sha1_file(root / "panels" / "ml1m" / "ftgrid_split.json") in mark
    assert sha1_file(ml1m["repo"] / "scripts" / "sigir" / "run_ftgrid.sh") in mark


def test_ml1m_second_run_skips_everything_and_touches_nothing(ml1m):
    assert ml1m["s1"] == ml1m["s2"]
    out = ml1m["r2"].stdout
    assert "already complete" not in out and "scores chunk" not in out and "dry-run trainer" not in out
    assert out.count(": scored") == 22                                  # 4 models x 5 arms + 2 FT-C like passes
    for product in ("build/ml1m/ml1m_all_h20.jsonl", "panels/ml1m/ftgrid_split.json",
                    "panels/ml1m/eval_sd_test_starperm1.jsonl", "report/ml1m.json"):
        assert f"[skip] {DRY_ROOT}/{product}" in out, product
    assert out.count("[skip] adapter") == 2                             # p0, p1 (s0-s2 are Gate-FT's)


def test_a_changed_run_key_moves_the_dir_aside_and_rescores_only_it(ml1m):
    d = ml1m["root"] / "scores" / "ml1m" / "zeroshot" / "starperm0"
    d.joinpath("run.key").write_text("an older key\n", encoding="utf-8")
    r = run_script(ml1m["repo"], "ml1m", STAGES="4")
    assert r.returncode == 0, tail(r)
    assert f"[moved aside] {DRY_ROOT}/scores/ml1m/zeroshot/starperm0" in r.stdout
    assert len(list(d.parent.glob("starperm0.stale.*"))) == 1 and r.stdout.count(": scored") == 15
    check_scoring_dir(ml1m["repo"], "ml1m", "zeroshot", "starperm0")


@pytest.fixture(scope="module")
def toys(tmp_path_factory):
    """toys with STAGES 0,1,2,5,6: three trained adapters (the split's recipe) and the knockout arms only."""
    return _chain(tmp_path_factory, "toys", "toys", STAGES="0,1,2,5,6")


def test_toys_trains_three_seeds_and_scores_the_knockout_arms(toys):
    repo, root = toys["repo"], toys["root"]
    check_layout(root, "toys", {"zeroshot": KNOCK, "s0": KNOCK, "s1": KNOCK, "s2": KNOCK}, {"s0", "s1", "s2"})
    split = read_json(root / "panels" / "toys" / "ftgrid_split.json")
    ov = split["train"]["overlength"]
    for k in range(3):
        cfg = read_json(root / "adapters" / "toys" / f"s{k}" / "train_config.json")
        assert cfg["model"] == QWEN and cfg["seed"] == k and Path(cfg["train"]).name == "train.jsonl"
        assert (cfg["bsz"], cfg["grad_accum"], cfg["max_len"]) == (ov["micro_bsz"], ov["grad_accum"], ov["max_len_used"])
        for arm in KNOCK:
            check_scoring_dir(repo, "toys", f"s{k}", arm, f"{DRY_ROOT}/adapters/toys/s{k}")
    for arm in KNOCK:
        check_scoring_dir(repo, "toys", "zeroshot", arm)
    ko = root / "build" / "toys" / "knockout"
    assert read_json(ko / "eval_pseudonym_report.json")["popularity_source"] == "sidecar"   # category-wide (P3)
    assert (ko / "eval.jsonl").read_bytes() == (root / "panels" / "toys" / "eval.jsonl").read_bytes()
    assert read_json(root / "report" / "toys.json")["models"] == ["zeroshot", "s0", "s1", "s2"]
    assert split["args"]["dev_users_sha1_checked"] is True              # TRAIN = the DEV (Pilot-1) users


@pytest.fixture(scope="module")
def games(tmp_path_factory):
    """games after a failed Gate-FT (the zero-shot panels only), with E1 failing on one arm."""
    return _chain(tmp_path_factory, "games", "games", DRY_GATE="GATE_FT_FAIL", DRY_E1_FAIL="zeroshot/nohist")


def test_games_failed_gate_ft_runs_only_the_zero_shot_panels(games):
    repo, root, out = games["repo"], games["root"], games["r"].stdout
    check_layout(root, "games", {"zeroshot": DECOMP}, set())
    for arm in ("like", "swap", "starperm0", "starperm1"):
        check_scoring_dir(repo, "games", "zeroshot", arm)
    assert "[stage 1] not run: Gate-FT decision GATE_FT_FAIL" in out and "[stage 5] not run" in out
    assert read_json(root / "report" / "games.json")["models"] == ["zeroshot"]
    built = {p.name for p in (root / "build" / "games").iterdir()}
    assert {"games_all_h20.jsonl", "games_all_h20.brand_pop.json", "games_all_h20.meta.json"} <= built
    meta = read_json(root / "build" / "games" / "games_all_h20.meta.json")
    assert (meta["hist_len"], meta["n_cands"], meta["min_hist"], meta["min_like"], meta["min_dislike"], meta["seed"],
            meta["n_users"], meta["gatefix_fields"]) == (20, 20, 3, 3, 3, 0, 10 ** 9, True)
    assert read_json(root / "panels" / "games" / "ftgrid_split.json")["args"]["dev_users_sha1_checked"] is False


def test_e1_failure_is_rerun_once_then_recorded_as_failed_integrity(games):
    root = games["root"] / "scores" / "games" / "zeroshot"
    assert (root / "nohist" / "FAILED_INTEGRITY").is_file() and len(list(root.glob("nohist.e1fail.*"))) == 1
    assert not any((root / a / "FAILED_INTEGRITY").exists() for a in ("like", "swap", "starperm0", "starperm1"))
    assert games["r"].stderr.count("[E1 failed]") == 1 and games["r"].stderr.count("FAILED_INTEGRITY:") == 1


@pytest.fixture(scope="module")
def guard_repo(tmp_path_factory):
    if BASH is None:
        pytest.skip(NO_BASH)
    return make_repo(tmp_path_factory.mktemp("guards") / "repo")


@pytest.mark.parametrize("domain, env, msg", [
    ("beauty", {}, "usage"),
    ("ml1m", {"STAGES": "0,7"}, "unknown stage"),
    ("ml1m", {"OUT_ROOT": "outputs/confrec/ftgrid"}, "never writes to a registered output root"),
    ("ml1m", {"OUT_ROOT": "./outputs/confrec/ftgrid_llama/"}, "never writes to a registered output root"),
    ("games", {"MODEL": LLAMA}, "ML-1M and Toys only"),
])
def test_input_guards_refuse_before_anything_is_written(guard_repo, domain, env, msg):
    r = run_script(guard_repo, domain, **env)
    assert r.returncode == 2 and msg in r.stderr, tail(r)
    assert not (guard_repo / "outputs").exists()


def test_a_second_backbone_needs_its_own_output_root(guard_repo):
    r = run_script(guard_repo, "ml1m", dry=False, MODEL="/models/Llama-3.1-8B-Instruct")
    assert r.returncode == 2 and "own OUT_ROOT" in r.stderr, tail(r)
    assert not (guard_repo / "outputs").exists()


@needs_chain
def test_stage_order_refusals(tmp_path):
    """Stage 3 refuses without stage 2 (no marker); after stage 2 a bound file changes and stages 3-6 refuse again (the
    marker no longer equals the record). Nothing is scored or reported."""
    repo = make_repo(tmp_path / "repo")
    root = repo / DRY_ROOT
    r = run_script(repo, "games", STAGES="3")
    assert r.returncode == 4 and "stage 3 refused" in r.stderr, tail(r)
    r = run_script(repo, "games", STAGES="0,2")
    assert r.returncode == 0 and "[dry] freeze rehearsal" in r.stdout, tail(r)
    assert (root / "freeze" / "games.core.ok").is_file()
    bound = repo / "src" / "confrec" / "metrics.py"
    bound.write_text(bound.read_text(encoding="utf-8") + "\n# changed after the freeze check\n", encoding="utf-8")
    for stage in ("3", "4", "5", "6"):
        r = run_script(repo, "games", STAGES=stage)
        assert r.returncode == 4 and f"stage {stage} refused" in r.stderr and "no longer equals" in r.stderr, tail(r)
    assert not (root / "scores").exists() and not (root / "report").exists()


@needs_chain
def test_second_backbone_trains_its_ml1m_adapters_and_has_no_ft_c(tmp_path):
    repo = make_repo(tmp_path / "repo")
    rel = "outputs/confrec/ftgrid_llama_dryrun"
    env = {"MODEL": LLAMA, "OUT_ROOT": rel}
    r = run_script(repo, "ml1m", STAGES="0,1", **env)
    assert r.returncode == 0, tail(r)
    assert "[link]" not in r.stdout and trained(r) == ["s0", "s1", "s2"]
    for k in range(3):
        cfg = read_json(repo / rel / "adapters" / "ml1m" / f"s{k}" / "train_config.json")
        assert cfg["model"] == LLAMA and cfg["seed"] == k and Path(cfg["train"]).name == "train.jsonl"
    split = read_json(repo / rel / "panels" / "ml1m" / "ftgrid_split.json")
    assert split["gateft_T_match"] is True and split["args"]["tokenizer"] == "Llama-3.1-8B-Instruct"
    r = run_script(repo, "ml1m", STAGES="perm", **env)
    assert r.returncode == 2 and "Gate-FT backbone only" in r.stderr, tail(r)


@needs_chain
@pytest.mark.skipif(not F2.exists(), reason="src/confrec/ftgrid_report.py (F2) does not exist yet: the chain ran with "
                                            "the F2 stub only")
def test_real_cli_dry_run(tmp_path):
    """The whole chain on toys with the real F1 and the real F2 (every arm, so the report has all its inputs)."""
    repo = make_repo(tmp_path / "repo", real_f2=True)
    r = run_script(repo, "toys")
    assert r.returncode == 0, tail(r)
    root = repo / DRY_ROOT
    assert read_json(root / "report" / "toys.json")
    check_layout(root, "toys", {m: DECOMP + KNOCK for m in ("zeroshot", "s0", "s1", "s2")}, {"s0", "s1", "s2"})
