#!/usr/bin/env python3
"""Page-budget measurement for the SIGIR paper (9 pages including appendices, references excluded).

    python scripts/sigir/page_budget.py --results RESULTS_DIR --out OUT_DIR [--prose_words 30] [--table_cell "0.123"]

Fills the skeleton (scripts/sigir/fill_paper.py) from RESULTS_DIR (the real results, or a synthetic set with the real schemas),
gives every slot that is still red a filler of a typical size (a table cell: a three-decimal estimate with a half-width; a prose
slot: --prose_words words, or the longer alternative of a `branch:` slot), writes the result into OUT_DIR, compiles it with
latexmk and prints the number of the last page before the references and how full its last column is. Nothing in OUT_DIR is a
paper: the numbers of the filler are not results.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_fill():
    spec = importlib.util.spec_from_file_location("_fill_paper_budget", ROOT / "scripts" / "sigir" / "fill_paper.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fill_paper_budget"] = mod
    spec.loader.exec_module(mod)
    return mod


def slot_spans(text: str):
    """(start, end, inner) of every non-comment \\DATANEEDED{...} (brace-balanced)."""
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


def in_tabular(text: str, pos: int) -> bool:
    a = text.rfind("\\begin{tabular", 0, pos)
    b = text.rfind("\\end{tabular", 0, pos)
    return a > b


def filler(inner: str, text: str, pos: int, words: int, cell: str) -> str:
    n = " ".join(inner.replace("\\_", "_").split())
    if in_tabular(text, pos):
        return cell if re.fullmatch(r"[\w.:, ]+", n) else cell
    if n.startswith("branch:"):
        alts = [a for a in re.split(r" / | -- |; else ", n[len("branch:"):]) if a.strip()]
        w = max((len(a.split()) for a in alts), default=words)
        return " ".join(["filler"] * min(max(w, 8), 45))
    if re.fullmatch(r"[A-Za-z_]+:[\w.\\,]+( ?[\w.,\\ ]*)?", n) and len(n.split()) <= 3:
        return "0.123$\\pm$.012"
    return " ".join(["filler"] * words)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paper", default=str(ROOT / "Paper" / "sigir2027"))
    ap.add_argument("--prose_words", type=int, default=30)
    ap.add_argument("--table_cell", default=r"0.123{\scriptsize$\pm$.012}")
    ap.add_argument("--no_compile", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    fill = load_fill()
    filled = out / "filled"
    fill.run(Path(a.paper), Path(a.results), filled)
    n_left = 0
    for f in sorted((filled / "sections").glob("*.tex")):
        text = f.read_text(encoding="utf-8")
        spans = slot_spans(text)
        for s, e, inner in reversed(spans):
            text = text[:s] + filler(inner, text, s, a.prose_words, a.table_cell) + text[e:]
        n_left += len(spans)
        f.write_text(text, encoding="utf-8", newline="\n")
    main_tex = (filled / "main.tex").read_text(encoding="utf-8")
    marker = "\\bibliographystyle{ACM-Reference-Format}"
    if marker not in main_tex or "\\begin{document}" not in main_tex:
        raise SystemExit("main.tex has no \\bibliographystyle line or no \\begin{document}")
    # Bug fix (editor pass 2026-10-05): the end of the text alone is not the end of the content. A float that is still deferred
    # when the text ends is placed after it by the \clearpage below (in the paper itself it would land after the references), so
    # every float records where it ends (page, position, environment) at shipout, and the measure is the later of the text end and
    # the last float end. The text end itself is measured as the used share of the column (top floats included).
    hook = ("\\makeatletter\n"
            "\\providecommand\\budgetfloatpos[4]{}\n"
            "\\newcommand\\budget@floatpos[1]{\\par\\pdfsavepos\\write\\@auxout{\\string\\budgetfloatpos{\\thepage}"
            "{\\the\\pdflastxpos}{\\the\\pdflastypos}{#1}}}\n"
            "\\AddToHook{env/table/end}{\\budget@floatpos{table}}\\AddToHook{env/table*/end}{\\budget@floatpos{table*}}\n"
            "\\AddToHook{env/figure/end}{\\budget@floatpos{figure}}\\AddToHook{env/figure*/end}{\\budget@floatpos{figure*}}\n"
            "\\makeatother\n")
    main_tex = main_tex.replace("\\begin{document}", hook + "\\begin{document}", 1)
    main_tex = main_tex.replace(marker, "\\makeatletter\\par\\typeout{BUDGET: page=\\thepage col=\\if@firstcolumn L\\else R\\fi "
                                "used=\\the\\pagetotal goal=\\the\\pagegoal th=\\the\\textheight ph=\\the\\pdfpageheight "
                                "pw=\\the\\pdfpagewidth vo=\\the\\voffset tm=\\the\\topmargin hh=\\the\\headheight "
                                "hs=\\the\\headsep ho=\\the\\hoffset om=\\the\\oddsidemargin em=\\the\\evensidemargin "
                                "cw=\\the\\columnwidth cs=\\the\\columnsep}\\makeatother\n"
                                "\\clearpage\n" + marker)
    (filled / "main.tex").write_text(main_tex, encoding="utf-8", newline="\n")
    print(f"slots still red before filler: {n_left}")
    if a.no_compile:
        return 0
    for _ in range(2):
        r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-f", "main.tex"], cwd=filled,
                           capture_output=True, text=True)
    log = (filled / "main.log").read_text(encoding="latin-1", errors="replace")
    log = log.replace("\n", "")      # the TeX log wraps long lines at 79 characters
    num = r"(-?[\d.]+)pt"
    m = re.findall(r"BUDGET: page=(\d+)\s*col=([LR])\s*used=" + num + r"\s*goal=" + num + r"\s*th=" + num + r"\s*ph=" + num
                   + r"\s*pw=" + num + r"\s*vo=" + num + r"\s*tm=" + num + r"\s*hh=" + num + r"\s*hs=" + num + r"\s*ho="
                   + num + r"\s*om=" + num + r"\s*em=" + num + r"\s*cw=" + num + r"\s*cs=" + num, log)
    pages = re.findall(r"Output written on main.pdf \((\d+) pages", log)
    # bug fix (2026-10-05): one match per warning (a greedy '.*' over the joined log counted every warning as one)
    undefined = len(re.findall(r"(?:Citation|Reference) `[^']+' on page \S+ undefined", log))
    errs = len(re.findall(r"^! ", log, re.M))
    if m:
        (page, col, used, goal, th, ph, pw, vo, tm, hh, hs, ho, om, em, cw, cs) = (
            int(m[-1][0]), m[-1][1], *map(float, m[-1][2:]))
        text_frac = min(1.0, max(0.0, 1.0 - (goal - used) / th))   # used share of the column, top floats included

        def value(pg, c, f):
            return pg - 1 + (0.5 * f if c == "L" else 0.5 + 0.5 * f)
        items = [("text", page, col, text_frac)]
        aux = (filled / "main.aux").read_text(encoding="latin-1", errors="replace")
        top = ph - (72.27 + vo + tm + hh + hs)                     # y of the top of the text area, from the page bottom (pt)
        for pg, xs, ys, env in re.findall(r"\\budgetfloatpos\{(\d+)\}\{(-?\d+)\}\{(-?\d+)\}\{([a-z*]+)\}", aux):
            pg, x, y = int(pg), int(xs) / 65536, int(ys) / 65536
            left = 72.27 + ho + (om if pg % 2 else em)
            c = "R" if env.endswith("*") or x > left + cw + cs / 2 else "L"
            items.append((env, pg, c, min(1.0, max(0.0, (top - y) / th))))
        last = max(items, key=lambda it: value(*it[1:]))
        pages_used = value(*last[1:])
        deferred = [it for it in items[1:] if (it[1], it[2]) > (page, col) and value(*it[1:]) > value(*items[0][1:])]
        print(f"text ends on page {page}, {col} column, {100 * text_frac:.0f}% full -> {value(*items[0][1:]):.2f}")
        if deferred:
            print(f"WARNING: {len(deferred)} float(s) end after the last line of text (deferred floats; without the "
                  f"measuring \\clearpage they would follow the references)")
        print(f"content ends on page {last[1]}, {last[2]} column, {100 * last[3]:.0f}% full ({last[0]}) -> "
              f"{pages_used:.2f} pages used")
    print(f"total pages with references: {pages[-1] if pages else '?'}; LaTeX errors: {errs}; undefined references/citations: {undefined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
