#!/usr/bin/env python3
"""Build the paper TeX from hand-written sources with result tokens (the successor of the slot-filling skeleton of fill_paper.py).

    python scripts/sigir/build_paper.py [--src Paper/sigir2027/src] [--results docs/sigir/results] [--out Paper/sigir2027/build]
                                        [--strict] [--list]

Every `*.tex`, `*.bib` and `*.sty` file below --src is copied to --out (other files, such as figure PDFs, are copied too), and every
result token of a TeX file is replaced by the value read from a committed result file. A number printed in the paper therefore comes from
a result file by construction; the source is free prose. Tokens (brace-balanced; the argument may span lines):

    \\jv{FILE|PATH|FMT}          one value of the JSON file docs/sigir/results/FILE; PATH is `/`-separated (a list index is an integer)
    \\jd{EXPR|FMT|x=FILE|PATH|y=...}   a derived value: EXPR is arithmetic over the names x, y, ... (each bound to FILE|PATH)
    \\jp{NAME}                   a phrase of the registered wording rules (the decision functions of fill_paper.py: counts per regime,
                                 labels), see PHRASES

Formats FMT (all print a minus sign as $-$; a value that rounds to zero is printed without a sign):
    f1 f2 f3 f4   number with that many decimals          s3  signed (+0.010 / $-$0.010)       pct0 pct1  a share as a percentage
    int           integer with thousands separators        str the value as text (LaTeX-escaped) p3  a p-value (< 0.001 shown as $<$0.001)
    For a record {est, lo, hi}:  est3 (the estimate), lo3, hi3, ci3 ([lo, hi]), estci3 (est [lo, hi]), pm3 (est$\\pm$half-width),
    hw3 (the half-width); the digit may be 2 or 4. A record whose field is missing prints the red box [TBD: key].

A token that cannot be resolved (missing file, field, null value) is replaced by a red box and listed in UNRESOLVED.json; --strict then
exits 1. ASSETS.json lists, for every token, the file (sha1), the path, the raw value and the printed text. Nothing is read from anywhere
except --results (no server, no network).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RED = r"\textcolor{red}{\textbf{[TBD: %s]}}"


def load_fill():
    spec = importlib.util.spec_from_file_location("_fill_paper_build", ROOT / "scripts" / "sigir" / "fill_paper.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fill_paper_build"] = mod
    spec.loader.exec_module(mod)
    return mod


class Unresolved(Exception):
    pass


def esc(s) -> str:
    return re.sub(r"([_&%#$])", r"\\\1", str(s))


def fin(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def numtxt(x: float, nd: int, signed: bool = False) -> str:
    s = f"{abs(float(x)):.{nd}f}"
    zero = float(s) == 0.0
    if float(x) < 0 and not zero:
        return "$-$" + s
    return ("+" + s) if (signed and not zero) else s


class Tokens:
    def __init__(self, results: Path):
        self.results = Path(results)
        self.cache: dict = {}
        self.sha1: dict = {}
        self.assets: list = []
        self.unresolved: list = []
        self.fill = None
        self.R = None

    # ---- result access
    def load(self, rel: str):
        if rel not in self.cache:
            p = self.results / rel
            if not p.is_file():
                raise Unresolved(f"result file missing: {rel}")
            data = p.read_bytes()
            self.sha1[rel] = hashlib.sha1(data).hexdigest()
            self.cache[rel] = json.loads(data.decode("utf-8"))
        return self.cache[rel]

    def get(self, rel: str, path: str):
        node = self.load(rel)
        done = []
        for k in [x for x in path.split("/") if x != ""]:
            if isinstance(node, dict) and node.get("available") is False:
                raise Unresolved(f"{rel}: {'/'.join(done) or '(top)'} is not available ({node.get('reason')})")
            if isinstance(node, dict) and k in node:
                node = node[k]
            elif isinstance(node, list):
                try:
                    node = node[int(k)]
                except (ValueError, IndexError):
                    raise Unresolved(f"{rel}: no list element {'/'.join(done + [k])}") from None
            else:
                raise Unresolved(f"{rel}: no field {'/'.join(done + [k])}")
            done.append(k)
        if isinstance(node, dict) and node.get("available") is False:
            raise Unresolved(f"{rel}: {'/'.join(done)} is not available ({node.get('reason')})")
        return node

    # ---- formatting
    def fmt(self, node, fmt: str, key: str) -> str:
        m = re.fullmatch(r"(est|lo|hi|ci|estci|pm|hw|f|s|pct|p)(\d)?", fmt)
        if fmt == "int":
            if not fin(node):
                raise Unresolved(f"{key}: not a number")
            return f"{int(round(float(node))):,}"
        if fmt == "str":
            if isinstance(node, (dict, list)) or node is None:
                raise Unresolved(f"{key}: not a text value")
            return esc(node)
        if m is None:
            raise Unresolved(f"{key}: unknown format {fmt!r}")
        kind, nd = m.group(1), int(m.group(2)) if m.group(2) else 3
        if kind in ("est", "lo", "hi", "ci", "estci", "pm", "hw"):
            if not isinstance(node, dict):
                if kind in ("est",) and fin(node):
                    return numtxt(node, nd)
                raise Unresolved(f"{key}: not a record {{est, lo, hi}}")
            est, lo, hi = node.get("est"), node.get("lo"), node.get("hi")
            if kind == "est":
                return numtxt(self._need(est, key, "est"), nd)
            if kind == "lo":
                return numtxt(self._need(lo, key, "lo"), nd)
            if kind == "hi":
                return numtxt(self._need(hi, key, "hi"), nd)
            if kind == "ci":
                return f"[{numtxt(self._need(lo, key, 'lo'), nd)}, {numtxt(self._need(hi, key, 'hi'), nd)}]"
            if kind == "estci":
                return (f"{numtxt(self._need(est, key, 'est'), nd)} "
                        f"[{numtxt(self._need(lo, key, 'lo'), nd)}, {numtxt(self._need(hi, key, 'hi'), nd)}]")
            hw = (self._need(hi, key, "hi") - self._need(lo, key, "lo")) / 2
            if kind == "hw":
                return numtxt(hw, nd)
            hwtxt = numtxt(hw, nd)
            return numtxt(self._need(est, key, "est"), nd) + r"$\pm$" + (hwtxt[1:] if hwtxt.startswith("0.") else hwtxt)
        if kind == "p":
            x = self._need(node, key, "p")
            return "$<$0.001" if x < 0.0005 else f"{x:.{nd}f}"
        x = self._need(node["est"] if isinstance(node, dict) and "est" in node else node, key, "value")
        if kind == "f":
            return numtxt(x, nd)
        if kind == "s":
            return numtxt(x, nd, signed=True)
        if kind == "pct":
            return f"{100 * float(x):.{nd}f}\\%"
        raise Unresolved(f"{key}: unhandled format {fmt!r}")

    @staticmethod
    def _need(x, key, what):
        if not fin(x):
            raise Unresolved(f"{key}: {what} is missing or null")
        return float(x)

    # ---- token kinds
    def jv(self, arg: str):
        parts = [p.strip() for p in arg.split("|")]
        if len(parts) != 3:
            raise Unresolved(f"\\jv needs FILE|PATH|FMT, got {arg!r}")
        rel, path, fmt = parts
        node = self.get(rel, path)
        text = self.fmt(node, fmt, f"{rel}:{path}")
        self.assets.append({"token": "jv", "file": rel, "sha1": self.sha1.get(rel), "path": path, "format": fmt,
                            "raw": node if not isinstance(node, (dict, list)) else node, "printed": text})
        return text

    def jd(self, arg: str):
        parts = [p.strip() for p in arg.split("|")]
        if len(parts) < 3:
            raise Unresolved(f"\\jd needs EXPR|FMT|x=FILE|PATH..., got {arg!r}")
        expr, fmt, binds = parts[0], parts[1], parts[2:]
        env, srcs = {}, {}
        i = 0
        while i < len(binds):
            if "=" not in binds[i]:
                raise Unresolved(f"\\jd: bad binding {binds[i]!r} (use name=FILE|PATH)")
            name, rel = binds[i].split("=", 1)
            if i + 1 >= len(binds):
                raise Unresolved(f"\\jd: binding {name} has no path")
            path = binds[i + 1]
            node = self.get(rel.strip(), path)
            node = node["est"] if isinstance(node, dict) and "est" in node else node
            env[name.strip()] = self._need(node, f"{rel}:{path}", "value")
            srcs[name.strip()] = {"file": rel.strip(), "sha1": self.sha1.get(rel.strip()), "path": path}
            i += 2
        val = safe_eval(expr, env)
        text = self.fmt(val, fmt, f"expr {expr}")
        self.assets.append({"token": "jd", "expr": expr, "inputs": srcs, "format": fmt, "raw": val, "printed": text})
        return text

    def jp(self, arg: str):
        name = arg.strip()
        fp = self.phrase_module()
        h = PHRASE_BUILDERS(fp).get(name)
        if h is None:
            raise Unresolved(f"\\jp: unknown phrase {name!r} (known: {', '.join(sorted(PHRASE_BUILDERS(fp)))})")
        try:
            text = h(self.R, None)
        except fp.Missing as e:                                  # a registered wording rule that cannot decide
            raise Unresolved(f"phrase {name}: {e}") from None
        self.assets.append({"token": "jp", "phrase": name, "printed": text})
        return text

    def phrase_module(self):
        if self.fill is None:
            self.fill = load_fill()
            self.R = self.fill.Results(self.results)
            self.fill._ACTIVE = self.R
        return self.fill


def PHRASE_BUILDERS(fp) -> dict:
    return {
        "eb_count": fp.c_eb_count, "g_count_zs": fp.c_g_count("ZS"), "g_count_ft": fp.c_g_count("FT"),
        "ej_count": fp.c_ext_ej, "ef_count": fp.c_ext_ef, "eh_count_zs": fp.c_ext_eh("ZS"), "eh_count_ft": fp.c_ext_eh("FT"),
        "robust_count": fp.c_ext_robust, "p1_reading": fp.p_ext_p1_reading, "ft_wording": fp.c_ext_ft_wording,
        "mir_decision": fp.p_mir_decision,
    }


_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}
_FUNCS = {"abs": abs, "min": min, "max": max, "round": round}


def safe_eval(expr: str, env: dict) -> float:
    """Arithmetic over the bound names (+ - * / **, unary signs, abs/min/max/round): a manual walk of the parsed expression, so
    nothing is executed (no eval, no attribute access, no other call)."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id in env:
            return float(env[n.id])
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return float(_FUNCS[n.func.id](*[ev(a) for a in n.args]))
        raise Unresolved(f"expression {expr!r}: construct {type(n).__name__} is not allowed or a name is unbound")
    try:
        return ev(ast.parse(expr, mode="eval"))
    except (SyntaxError, ZeroDivisionError, OverflowError) as e:
        raise Unresolved(f"expression {expr!r}: {e}") from None


TOKEN = re.compile(r"\\(jv|jd|jp)\{")


def find_tokens(text: str):
    """(start, end, kind, argument) of every token (brace-balanced), skipping commented-out lines."""
    out, i = [], 0
    while True:
        m = TOKEN.search(text, i)
        if m is None:
            return out
        ls = text.rfind("\n", 0, m.start()) + 1
        if re.search(r"(?<!\\)%", text[ls:m.start()]):                 # inside a % comment: left as it is
            i = m.end()
            continue
        depth, p = 1, m.end()
        while depth and p < len(text):
            c = text[p]
            if c == "\\":
                p += 2
                continue
            depth += {"{": 1, "}": -1}.get(c, 0)
            p += 1
        if depth:
            raise SystemExit(f"unbalanced braces in a \\{m.group(1)} token at offset {m.start()}")
        out.append((m.start(), p, m.group(1), text[m.end():p - 1]))
        i = p


def build(src: Path, results: Path, out: Path, strict: bool = False) -> dict:
    src, out = Path(src), Path(out)
    if not src.is_dir():
        raise SystemExit(f"{src} does not exist")
    tk = Tokens(results)
    out.mkdir(parents=True, exist_ok=True)
    keep = set()
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        keep.add(dst.resolve())
        if f.suffix == ".tex":
            text = f.read_text(encoding="utf-8")
            pieces, last = [], 0
            for a, b, kind, arg in find_tokens(text):
                pieces.append(text[last:a])
                key = f"{kind}{{{' '.join(arg.split())}}}"
                try:
                    pieces.append(getattr(tk, kind)(arg))
                except Unresolved as e:
                    pieces.append(RED % esc(" ".join(arg.split())[:70]))
                    tk.unresolved.append({"file": rel.as_posix(), "token": key[:200], "reason": str(e)})
                last = b
            pieces.append(text[last:])
            new = "".join(pieces)
            if not dst.exists() or dst.read_text(encoding="utf-8") != new:
                dst.write_text(new, encoding="utf-8", newline="\n")
        else:
            if not dst.exists() or dst.read_bytes() != f.read_bytes():
                shutil.copyfile(f, dst)
    (out / "ASSETS.json").write_text(json.dumps({"results_root": str(Path(results)), "tokens": tk.assets}, indent=1, default=str),
                                     encoding="utf-8", newline="\n")
    (out / "UNRESOLVED.json").write_text(json.dumps({"unresolved": tk.unresolved}, indent=1), encoding="utf-8", newline="\n")
    summary = {"tokens": len(tk.assets) + len(tk.unresolved), "resolved": len(tk.assets), "unresolved": len(tk.unresolved)}
    print(f"build_paper: {summary['tokens']} tokens, {summary['resolved']} resolved, {summary['unresolved']} unresolved -> {out}")
    for u in tk.unresolved[:25]:
        print(f"  unresolved {u['file']}: {u['token'][:90]} -- {u['reason'][:110]}")
    if strict and tk.unresolved:
        raise SystemExit(1)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(ROOT / "Paper" / "sigir2027" / "src"))
    ap.add_argument("--results", default=str(ROOT / "docs" / "sigir" / "results"))
    ap.add_argument("--out", default=str(ROOT / "Paper" / "sigir2027" / "build"))
    ap.add_argument("--strict", action="store_true", help="exit 1 if any token is unresolved")
    a = ap.parse_args(argv)
    try:
        build(Path(a.src), Path(a.results), Path(a.out), a.strict)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
