#!/usr/bin/env python3
"""List the numeric literals typed by hand in the paper sources (anything that is not a result token).

    python scripts/sigir/audit_numbers.py [--src Paper/sigir2027/src] [--allow Paper/sigir2027/src/NUMBERS_OK.txt] [--strict]

A result number must come from `\\jv`, `\\jd` or `\\jp` (scripts/sigir/build_paper.py). This script strips comments, the tokens, the
arguments of \\cite/\\ref/\\label/\\input/\\includegraphics-like commands and the LaTeX markup, and prints every remaining number-like
literal (a decimal with at least two digits after the point, a percentage, an integer of three or more digits, a ratio such as 4/5) with
its line and context, except those listed in the allow file (one literal per line, `#` comments: registered constants such as 0.05,
0.60, 2,000, dates and counts that are facts of the design, not results). --strict exits 1 when anything remains.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NUM = re.compile(r"(?<![\w.])(?:\d+\.\d{2,}|\d{1,3}(?:,\d{3})+|\d{3,}|\d+(?:\.\d+)?\\%|\d+/\d+)(?![\w])")
DROP_ARGS = re.compile(r"\\(?:citep?|citet|cite|ref|eqref|label|input|include|includegraphics|bibliography|bibliographystyle|"
                       r"cref|Cref|url|href|acmConference|acmDOI|acmISBN|acmYear|setcopyright|copyrightyear)\*?(?:\[[^\]]*\])?\{[^}]*\}")


def strip_tokens(text: str) -> str:
    out, i = [], 0
    pat = re.compile(r"\\(?:jv|jd|jp)\{")
    while True:
        m = pat.search(text, i)
        if m is None:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:m.start()])
        depth, p = 1, m.end()
        while depth and p < len(text):
            c = text[p]
            if c == "\\":
                p += 2
                continue
            depth += {"{": 1, "}": -1}.get(c, 0)
            p += 1
        out.append(" TOKEN ")
        i = p


def scan(src: Path, allow: set) -> list:
    hits = []
    for f in sorted(src.rglob("*.tex")):
        text = f.read_text(encoding="utf-8")
        cleaned_lines = []
        for line in text.splitlines():
            m = re.search(r"(?<!\\)%", line)
            cleaned_lines.append(line[:m.start()] if m else line)
        body = strip_tokens("\n".join(cleaned_lines))
        body = DROP_ARGS.sub(" ", body)
        for ln, line in enumerate(body.splitlines(), 1):
            for m in NUM.finditer(line):
                lit = m.group(0)
                if lit in allow:
                    continue
                ctx = line[max(0, m.start() - 40):m.end() + 30].strip()
                hits.append((f.relative_to(src).as_posix(), ln, lit, ctx))
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(ROOT / "Paper" / "sigir2027" / "src"))
    ap.add_argument("--allow", default=None, help="default <src>/NUMBERS_OK.txt when it exists")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    src = Path(a.src)
    allow_file = Path(a.allow) if a.allow else src / "NUMBERS_OK.txt"
    allow = set()
    if allow_file.is_file():
        for line in allow_file.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                allow.add(line)
    hits = scan(src, allow)
    for f, ln, lit, ctx in hits:
        print(f"{f}:{ln}: {lit}   ...{ctx}...")
    print(f"audit_numbers: {len(hits)} hand-typed numeric literal(s) outside result tokens (allow list: {len(allow)} entries)")
    return 1 if (a.strict and hits) else 0


if __name__ == "__main__":
    sys.exit(main())
