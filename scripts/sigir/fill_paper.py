#!/usr/bin/env python3
"""Fill the SIGIR 2027 paper skeleton from the committed result files (and from nothing else).

    python scripts/sigir/fill_paper.py [--paper Paper/sigir2027] [--results docs/sigir/results] [--out Paper/sigir2027/filled]
                                       [--check_equal]

Reads the skeleton (main.tex, sections/*.tex: every result is a \\DATANEEDED{...} slot) and the result files that
scripts/sigir/pull_results.ps1 copied into docs/sigir/results/<alias>/..., and writes a FILLED COPY of main.tex, sections/*
and references.bib into --out, plus
  UNFILLED.json     every slot that was not filled: section file, line, slot text, table / row / column, reason, detail;
  FILLED.json       every filled slot with its text and the result-file fields it was read from (the audit trail), and
                    "notes" on values deliberately not printed;
  CHECK_EQUAL.json  every quantity printed in more than one place (table cell or prose) with its sources and printed
                    texts, flagging different estimates (--check_equal prints it).
The skeleton is never modified. Same inputs -> byte-identical outputs (no clock, host or absolute path is written; a file
is rewritten only when its bytes change). Prints a summary: slots filled, unfilled by reason.

Fill rules (sections/experiments.tex FILL RULES 1-6; idea-stage/PREREG_AMENDMENT_3.md sections 3 and 11 and addenda 1-4):
  * a slot is filled only from the result file its alias names (alias legend of experiments.tex; the alias follows the
    script that writes the file: cpu, ko and corr (rated) are fields of the grid reports, see docs/sigir/PAPER_DATA_MAP.md);
  * table cells: estimate, then the half-width of the 95% interval, three decimals (0.612{\\scriptsize$\\pm$.014}); the
    Gini and APLT blocks of tab:exposure and block C of tab:anatomy give estimates only; LoRA cells give the mean over
    seeds 0-2 with its interval and the seed s.d. in parentheses; a value without an interval in its file is printed as
    an estimate; an endpoint on fewer than 150 users carries "(descriptive)" (no interval-based claim);
  * a table's meaning comes from its position: TABLE_SPECS (one entry per table label) maps every row label and column to
    the statistic, panel, backbone, regime and segment, and checks the slot's own text (a changed skeleton is refused,
    never guessed: reason skeleton_changed);
  * prose slots alias:field are filled through PROSE_SPECS (keyed by file, slot text and occurrence, guarded by the text
    before the slot); a cross-domain sentence gets the per-domain values in the registered order; a slot whose panel or
    regime the sentence does not fix stays a red slot (ambiguous_slot);
  * direction words are written only where the registered test decides them: Holm p < 0.05 within the registered family
    (summary.json for the next-item endpoints; E-B over the four Qwen domains, computed here from the per-domain raw p
    values with the sigma_seed rule; E-D per domain and regime from the report), the S3 admission rule, the single tests
    P1 and P2 - P1 (their registered labels), the method-slot Holm family; a mixed or partial outcome stays a red slot
    (not_decided);
  * "not run" only where a result file records a registered cut (ftprune: arm NOT_RUN; ftmethod: slot KILLED before the
    dataset); "FAILED_INTEGRITY" where the report records it; branch:... slots are never touched; free-text slots
    (sentences, clauses, admission verdicts) are listed, never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECTION_FILES = ("abstract", "introduction", "related_work", "preliminaries", "observation", "method", "experiments",
                 "conclusion", "appendix")
ALIASES = ("sel", "gate", "gft", "aud", "aud2q", "aud2l", "grid", "cpu", "ko", "prn", "slot", "corr", "mir", "ext")
# ext: the addendum-6 analysis (src/confrec/ftgrid_extra.py, A3-6 items 2-8, addendum 8 for FT-Q): extra/<d>.json (Qwen main root),
# extra/llama/<d>.json, extra/ftq/<d>.json (the FT-Q teacher root) and extra/summary.json (section "alias ext" below).
MIN_N = 150                                   # A3 section 1: fewer users = descriptive
RATED = ("ml1m", "toys", "games", "sports")   # rated panels, registered order
RATED_NAME = {"ml1m": "ML-1M", "toys": "Toys", "games": "Video Games", "sports": "Sports"}
NEXT = ("sports", "toys", "home", "tools")    # next-item panels
FAMILY_UNITS = (("S10k", "sports", "events_1001_10000"), ("Toys", "toys", "all"), ("Home", "home", "all"),
                ("Tools", "tools", "all"))  # the four family units of the next-item audit (sports = its main segment)
ANATOMY_UNITS = {"S1k": ("sports", "events_1_1000"), "S10k": ("sports", "events_1001_10000"), "Toys": ("toys", "all"),
                 "Home": ("home", "all"), "Tools": ("tools", "all")}
FT_MODELS = ("s0", "s1", "s2")
REF_METHODS = {  # tab:exposure row label -> reference method of docs/sigir/ref_exposure (all nine: c_rq3_baselines reads them)
    # The verbalised reranker of the earlier benchmark protocol is anonymised in the paper (double-blind); its method key stays.
    r"Verbalised reranker": "ccrp_v3", r"ELMRec~\citep{wang2024elmrec}": "elmrec_graph",
    r"IRLLRec~\citep{wang2025irllrec}": "irllrec_intent", r"LLM2Rec~\citep{he2025llm2rec}": "llm2rec_sasrec",
    r"LLMEmb~\citep{liu2025llmemb}": "llmemb", r"LLM-ESR~\citep{liu2024llmesr}": "llmesr_sasrec",
    r"ProEx~\citep{zhang2026proex}": "proex_profile", r"ProMax~\citep{zhang2026promax}": "promax_profile",
    r"RLMRec~\citep{ren2024rlmrec}": "rlmrec_graphcl"}
REASONS = {
    "result_file_missing": "the result file that feeds the slot does not exist yet (pull it, or run the job)",
    "result_not_in_report": "the result file exists but marks the block unavailable (an arm not run or not requested "
                            "yet, or excluded): the detail quotes the report's own reason",
    "branch_slot": "branch:... alternatives (FILL RULE 4): keep the one that occurred; never written by this script",
    "free_text": "a sentence, clause or verdict to be written from the filled tables; not a result field",
    "field_not_produced": "no script writes this field (or the result file does not carry it)",
    "field_null": "the result file carries the field as null (not computable on its rows)",
    "below_min_n": "an interval-based claim on fewer than 150 users (A3 section 1): no direction word",
    "ambiguous_slot": "the sentence does not fix the panel, regime or field: no mechanical reading",
    "not_decided": "the registered test gives a mixed or partial outcome: the wording needs a written sentence",
    "interpretive": "interpretive wording that no registered test decides",
    "incomplete_regime": "a registered seed or run is missing (never replaced): the seed mean is not written",
    "branch_not_taken": "the block belongs to a branch that did not occur (gate decision): delete it, do not fill it",
    "skeleton_changed": "the slot text or table layout differs from TABLE_SPECS / PROSE_SPECS: update the spec first",
    "to_be_removed": "the main session drops this slot from the skeleton (decision 2026-10-04; no script writes it): it "
                     "leaves this list when the skeleton edit lands",
}
REMOVAL_DECISION = "main-session decision 2026-10-04: dropped from the skeleton"


class Missing(Exception):
    """A slot that cannot be filled: reason (a key of REASONS) and a detail."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


# ------------------------------------------------------------------------------------------------ results access
class Results:
    """Lazy, read-only access to docs/sigir/results/<alias>/... with a per-slot trace of the fields read."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.cache: dict = {}
        self.sha1: dict = {}
        self.trace: list = []
        self.seen: dict = {}        # id(object read) -> (object, "file:path"): the source of every number printed
        self.printed: list = []     # (source, estimate text, printed text) of the slot being filled

    def _remember(self, node, key: str) -> None:
        def trackable(x):
            return isinstance(x, (dict, float)) or (isinstance(x, int) and not isinstance(x, bool) and abs(x) > 256)
        if trackable(node) or isinstance(node, list):
            self.seen[id(node)] = (node, key)
        if isinstance(node, list):
            for i, x in enumerate(node):
                if trackable(x):
                    self.seen[id(x)] = (x, f"{key}.{i}")

    def load(self, rel: str):
        if rel not in self.cache:
            p = self.root / rel
            if not p.is_file():
                raise Missing("result_file_missing", rel)
            data = p.read_bytes()
            self.sha1[rel] = hashlib.sha1(data).hexdigest()
            self.cache[rel] = json.loads(data.decode("utf-8"))
        return self.cache[rel]

    def get(self, rel: str, *path):
        """The object at `path` inside the JSON file `rel`; a block with available = false raises result_not_in_report
        with the block's own reason, a missing key field_not_produced."""
        node = self.load(rel)
        done = []
        for k in path:
            if isinstance(node, dict) and node.get("available") is False:
                raise Missing("result_not_in_report", f"{rel}: {'.'.join(map(str, done)) or '(top)'} is not "
                                                      f"available ({node.get('reason')})")
            if isinstance(node, dict) and k in node:
                node = node[k]
            elif isinstance(node, list) and isinstance(k, int) and -len(node) <= k < len(node):
                node = node[k]
            else:
                raise Missing("field_not_produced", f"{rel}: no field {'.'.join(map(str, done + [k]))}")
            done.append(k)
        if isinstance(node, dict) and node.get("available") is False:
            raise Missing("result_not_in_report", f"{rel}: {'.'.join(map(str, done))} is not available "
                                                  f"({node.get('reason')})")
        key = f"{rel}:{'.'.join(map(str, done))}"
        self.trace.append(key)
        self._remember(node, key)
        return node


_ACTIVE: "Results | None" = None     # the Results of the slot being filled (registry of the printed numbers)


def _src_of(obj):
    """The result field ("file:path") an object was read from, if it was read for the slot being filled."""
    if _ACTIVE is None:
        return None
    hit = _ACTIVE.seen.get(id(obj))
    return hit[1] if hit is not None and hit[0] is obj else None


def _record(src, est: str, text: str) -> None:
    if _ACTIVE is not None:
        _ACTIVE.printed.append((src, est, text))


# ------------------------------------------------------------------------------------------------ values and format
def _fin(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


@dataclass
class Val:
    """One reported quantity: estimate with its 95% interval, a seed s.d., the minimum-n flag, or a text value."""
    est: float | None = None
    lo: float | None = None
    hi: float | None = None
    sd: float | None = None
    descriptive: bool = False
    text: str | None = None
    est_only: bool = False
    src: str | None = None          # the result field the estimate was read from (--check_equal)


def val(rec, *, sd=None, est_only=False, n_key=None, src=None) -> Val:
    """Val from a result record {est, lo, hi, n_users | n, descriptive_min_n}."""
    src = src or _src_of(rec) or (_ACTIVE.trace[-1] if _ACTIVE is not None and _ACTIVE.trace else None)
    if not isinstance(rec, dict):
        if _fin(rec):
            return Val(est=float(rec), sd=sd, est_only=True, src=src)
        raise Missing("field_null", f"expected a number or a record, got {rec!r}")
    est = rec.get("est")
    if not _fin(est):
        raise Missing("field_null", "the estimate is null in the result file")
    desc = rec.get("descriptive_min_n") is True
    n = rec.get(n_key) if n_key else rec.get("n_users")
    if _fin(n) and int(n) < MIN_N:
        desc = True
    lo, hi = rec.get("lo"), rec.get("hi")
    return Val(est=float(est), lo=float(lo) if _fin(lo) else None, hi=float(hi) if _fin(hi) else None, sd=sd,
               descriptive=desc, est_only=est_only, src=src)


def _numstr(x, nd: int = 3) -> str:
    if not _fin(x):
        raise Missing("field_null", "a needed value is null in the result file")
    s = f"{abs(float(x)):.{nd}f}"
    return ("$-$" if float(x) < 0 and float(s) != 0.0 else "") + s


def num(x, nd: int = 3, src=None) -> str:
    """0.612 / $-$0.012 (a value that rounds to zero is printed without a sign); recorded for --check_equal."""
    s = _numstr(x, nd)
    _record(src or _src_of(x), s, s)
    return s


def short(x, nd: int = 3) -> str:
    """Half-widths and s.d.: .014 (no leading zero)."""
    if not _fin(x):
        raise Missing("field_null", "a needed interval or s.d. is null in the result file")
    s = f"{abs(float(x)):.{nd}f}"
    s = s[1:] if s.startswith("0.") else s
    return ("$-$" if float(x) < 0 and float(s) != 0.0 else "") + s


def integer(x) -> str:
    if not _fin(x):
        raise Missing("field_null", "a needed count is null in the result file")
    s = f"{int(x):,}"
    _record(_src_of(x), s, s)
    return s


def tex(s) -> str:
    return re.sub(r"([_&%#$])", r"\\\1", str(s))


def yesno(b) -> str:
    if not isinstance(b, bool):
        raise Missing("field_null", f"expected a boolean, got {b!r}")
    return "yes" if b else "no"


def pval(p) -> str:
    if not _fin(p):
        raise Missing("below_min_n", "no p-value in the result file (descriptive endpoint or not computed)")
    return "$<$0.001" if p < 0.0005 else f"{p:.3f}"


def stdev(xs) -> float:
    """SD with ddof 1 (sigma_seed, A3 section 3)."""
    xs = [float(x) for x in xs]
    if len(xs) < 2 or not all(math.isfinite(x) for x in xs):
        raise Missing("field_null", "a per-seed value is null: no seed s.d.")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt(v: Val, prose: bool = False) -> str:
    """Table: 0.742{\\scriptsize$\\pm$.019\\,(.002)}; prose: 0.742$\\pm$.019."""
    if v.text is not None:
        return v.text
    s = _numstr(v.est)
    extra = ""
    if not v.est_only and v.lo is not None and v.hi is not None:
        extra = r"$\pm$" + short((v.hi - v.lo) / 2)
    if v.sd is not None and not prose:
        extra += r"\,(" + short(v.sd) + ")"
    if prose:
        out = s + extra + (" (descriptive)" if v.descriptive else "")
    else:
        out = s + (r"{\scriptsize" + extra + "}" if extra else "") + (r"{\scriptsize\,(descriptive)}"
                                                                       if v.descriptive else "")
    _record(v.src, s, out)
    return out


def norm(s: str) -> str:
    """Slot text as compared with the specs: \\_ -> _, whitespace collapsed."""
    return re.sub(r"\s+", " ", s.replace("\\_", "_").replace("\\%", "%")).strip()


class SpecError(Exception):
    pass


def expect(slot: str, *options: str) -> str:
    n = norm(slot)
    if n not in options:
        raise SpecError(f"slot text {n!r} is not the expected {' or '.join(repr(o) for o in options)}")
    return n


# ------------------------------------------------------------------------------------------------ gates
def require_ft(R: Results) -> None:
    """A3 section 0 / FILL RULE 4b: fine-tuned items exist only after a recorded GATE_FT_PASS."""
    dec = R.get("gft/gate_ft.json", "decision")
    if dec != "GATE_FT_PASS":
        raise Missing("branch_not_taken", f"gft:decision = {dec}: fine-tuned items are deleted (FILL RULE 4b)")


def require_gatepass(R: Results) -> None:
    dec = R.get("gate/gate.json", "decision")
    if dec != "GATE_PASS":
        raise Missing("branch_not_taken", f"gate:decision = {dec}: the GATEPASS block is deleted (FILL RULE 4)")


# ------------------------------------------------------------------------------------------------ grid helpers
def grid_rel(bb: str, d: str) -> str:
    return f"grid/{bb}/{d}.json"


def regime_ok(R: Results, rel: str, blk: str, reg: str) -> None:
    node = R.get(rel, blk, reg)
    if reg == "FT" and node.get("complete") is False:
        raise Missing("incomplete_regime", f"{rel}: {blk}.FT has models {node.get('models')}, missing or excluded "
                                           f"{node.get('missing_or_excluded')} (A3 addendum 1 item 7; never replaced)")


def arms_of(blk: str, sub: tuple) -> tuple:
    """The scoring arms (ftgrid_report.ARM_PANEL) a block of the grid report is computed from."""
    if blk != "E_D":
        return ("like",)
    s = sub[0] if sub else ""
    if s == "UAUC":
        return {"L": ("like",), "pi": ("like", "swap"), "e_hat": ("like", "swap"), "L_nohist": ("nohist",)}.get(
            sub[1] if len(sub) > 1 else "L", ("like",))
    return {"shares": ("like", "swap"), "information_gain": ("like", "swap"),
            "star_permutation": ("like", "starperm0", "starperm1")}.get(s, ("like",))


def zs_failed(R: Results, rel: str, arms: tuple) -> bool:
    """True when the report excluded one of the zero-shot model's arms for E1 (FILL RULE 3: it reads FAILED_INTEGRITY)."""
    try:
        runs = R.get(rel, "runs", "zeroshot")
    except Missing:
        return False
    return any((runs.get(a) or {}).get("status") == "FAILED_INTEGRITY" for a in arms)


def regime_val(R: Results, bb: str, d: str, reg: str, blk: str, sub: tuple, key=None, mean_key="mean_over_seeds",
               per_key="per_model") -> Val:
    """The ZS value (model zeroshot) or the FT seed mean with its interval and the seed s.d. of a block of the grid
    report: rec = <blk>.<reg>.<sub...>.<per_key>.<model>[.<key>] / <blk>.<reg>.<sub...>.<mean_key>[.<key>]."""
    rel = grid_rel(bb, d)
    if reg == "FT":
        require_ft(R)
    k = (key,) if key is not None else ()
    if reg == "ZS":
        try:
            regime_ok(R, rel, blk, reg)
            return val(R.get(rel, blk, reg, *sub, per_key, "zeroshot", *k))
        except Missing as e:
            if e.reason in ("result_not_in_report", "field_not_produced") and zs_failed(R, rel, arms_of(blk, sub)):
                return Val(text=r"FAILED\_INTEGRITY")
            raise
    regime_ok(R, rel, blk, reg)
    have = R.get(rel, blk, reg, *sub, per_key)
    if any(m not in have for m in FT_MODELS):
        raise Missing("incomplete_regime", f"{rel}: {'.'.join((blk, reg) + tuple(sub))} has seeds {sorted(have)} "
                                           "(s0-s2 registered; a missing seed is never replaced)")
    per = [R.get(rel, blk, reg, *sub, per_key, m, *k) for m in FT_MODELS]
    rec = R.get(rel, blk, reg, *sub, mean_key, *k)
    sd = stdev([p["est"] if isinstance(p, dict) else p for p in per])
    return val(rec, sd=sd)


def ref_val(R: Results, bb: str, d: str, ref: str, prefer=("ZS", "FT")) -> Val:
    """A non-LLM reference (alias cpu) on the identical users and rows of a regime of the grid report (E-A): q_hat,
    popularity, mf (the temporal biased MF) or mf_personal_residual_warm (the warm-pair personal residual)."""
    rel = grid_rel(bb, d)
    first = None
    for reg in prefer:
        try:
            if ref == "mf_personal_residual_warm":
                return val(R.get(rel, "E_A", reg, "mf_personal_residual_warm_pairs_only"))
            return val(R.get(rel, "E_A", reg, "UAUC_TEST", "references", ref))
        except Missing as e:
            if e.reason == "result_file_missing":
                raise
            first = first or e
    raise first


def split_rel(bb: str, d: str) -> str:
    return f"grid/{bb}/{d}_split.json"


# ------------------------------------------------------------------------------------------------ table parsing
@dataclass
class Cell:
    start: int            # offset of the cell text in the file
    end: int
    col: int              # logical column index (0 = row label)
    span: int


@dataclass
class TableInfo:
    label: str | None
    start: int
    end: int
    tab_start: int
    tab_end: int
    rows: list = field(default_factory=list)      # [(row_label, [Cell])]


def _skip_comment(text: str, i: int) -> int:
    j = text.find("\n", i)
    return len(text) if j < 0 else j


def _balanced(text: str, i: int) -> int:
    """End offset (exclusive) of the brace group starting at text[i] == '{'."""
    depth = 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "%":
            i = _skip_comment(text, i)
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces")


RULE_RE = re.compile(r"^(?:\s*(?:\\(?:toprule|midrule|bottomrule|hline)\b|\\cmidrule(?:\([^)]*\))?\{[^}]*\}|%[^\n]*\n?))*")


def clean_label(s: str) -> str:
    s = RULE_RE.sub("", s)
    s = re.sub(r"(?<!\\)%[^\n]*", "", s)            # comments, not an escaped \%
    s = re.sub(r"^\s*(\\quad\s*)+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_tables(text: str) -> list:
    out = []
    for m in re.finditer(r"\\begin\{(table\*?)\}", text):
        env = m.group(1)
        end_tok = "\\end{" + env + "}"
        e = text.find(end_tok, m.end())
        if e < 0:
            continue
        body = text[m.start():e + len(end_tok)]
        lab = re.search(r"\\label\{([^}]*)\}", body)
        t = text.find("\\begin{tabular}", m.start(), e)
        if t < 0:
            continue
        spec_end = _balanced(text, t + len("\\begin{tabular}"))
        te = text.find("\\end{tabular}", spec_end, e)
        info = TableInfo(lab.group(1) if lab else None, m.start(), e + len(end_tok), spec_end, te)
        # rows: split at \\ (depth 0, outside comments); cells at & (depth 0)
        i, depth, cells, cell_start = spec_end, 0, [], spec_end
        rows_raw = []
        while i < te:
            c = text[i]
            if c == "%":
                i = _skip_comment(text, i)
                continue
            if c == "\\":
                if text.startswith("\\\\", i) and depth == 0:
                    cells.append((cell_start, i))
                    rows_raw.append(cells)
                    cells, i = [], i + 2
                    cell_start = i
                    continue
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == "&" and depth == 0:
                cells.append((cell_start, i))
                cell_start = i + 1
            i += 1
        for cells in rows_raw:
            col, parsed = 0, []
            for (a, b) in cells:
                s = text[a:b]
                stripped = clean_label(s)
                span = 1
                mm = re.match(r"\\multicolumn\{(\d+)\}", stripped)
                if mm:
                    span = int(mm.group(1))
                parsed.append(Cell(a, b, col, span))
                col += span
            info.rows.append((clean_label(text[cells[0][0]:cells[0][1]]), parsed))
        out.append(info)
    return out


# ------------------------------------------------------------------------------------------------ slots
@dataclass
class Slot:
    file: str
    start: int
    end: int
    inner: str
    line: int
    kind: str = ""          # branch | table | caption | prose
    table: str | None = None
    row: str | None = None
    col: int | None = None
    span: int | None = None
    ncols: int | None = None
    k: int = 0              # index of the slot within its cell
    occ: int = 0            # occurrence of its normalized text among the file's prose slots
    before: str = ""        # normalized text before the slot (other slots as [S])
    markers: tuple = ()
    fill: str | None = None
    reason: str | None = None
    detail: str | None = None
    sources: tuple = ()
    anchor: str | None = None   # the PROSE_SPECS anchor that matched (prose slots)
    printed: tuple = ()         # (source field, estimate text, printed text) of every number the fill prints


def find_slots(text: str, fname: str) -> list:
    key = "\\DATANEEDED{"
    out, i = [], 0
    while True:
        j = text.find(key, i)
        if j < 0:
            break
        ls = text.rfind("\n", 0, j) + 1
        line_txt, k, commented = text[ls:j], 0, False
        while k < len(line_txt):
            if line_txt[k] == "\\":
                k += 2
                continue
            if line_txt[k] == "%":
                commented = True
                break
            k += 1
        e = _balanced(text, j + len(key) - 1)
        if not commented:
            out.append(Slot(fname, j, e, text[j + len(key):e - 1], text.count("\n", 0, j) + 1))
        i = e
    return out


MARKERS = ("BRANCH-GATEFT", "BRANCH-GATEPASS", "METHOD-SLOT")   # FILL RULE 4: conditional blocks (reported per slot)


def marker_spans(text: str) -> list:
    spans, open_ = [], {}
    for m in re.finditer(r"^%\s*(" + "|".join(MARKERS) + r")-(BEGIN|END)\b", text, re.M):
        name, kind = m.group(1), m.group(2)
        if kind == "BEGIN":
            open_[name] = m.start()
        elif name in open_:
            spans.append((name, open_.pop(name), m.end()))
    return spans


# ================================================================================================ TABLE_SPECS
# Every handler gets (R, colkey, k, slot_text) and returns the cell text or raises Missing / SpecError.

def _aud_rec(R, rel, seg, q, *path):
    return R.get(rel, "segments", seg, "questions", q, *path)


def _z2_seg(R, rel) -> str:
    segs = R.get(rel, "segments")
    if not isinstance(segs, dict) or len(segs) != 1:
        raise Missing("field_not_produced", f"{rel}: expected one segment (Z2, A3 section 4), found {sorted(segs)}")
    return next(iter(segs))


# ---- tab:gate-outcomes
SEL, GATE, GFT = "sel/selection.json", "gate/gate.json", "gft/gate_ft.json"


def _sel_uauc(R, variant_key: str, panel: str) -> str:
    v = R.get(SEL, "v_star") if variant_key == "v_star" else "V0"
    return num(R.get(SEL, "table", v, panel, "UAUC"))


def go_g5_ml1m(R, c, k, s):
    if c == "c1":
        expect(s, "sel:UAUC")
        return _sel_uauc(R, "V0", "ml1m")
    if c == "c2":
        expect(s, "sel:UAUC of v_star")
        return _sel_uauc(R, "v_star", "ml1m")
    if c == "c4":
        expect(s, "sel:decision")
        return tex(R.get(SEL, "decision"))
    raise SpecError(f"no slot expected in column {c}")


def go_g5_toys(R, c, k, s):
    if c == "c1":
        expect(s, "sel:UAUC")
        return _sel_uauc(R, "V0", "toys")
    if c == "c2":
        expect(s, "sel:UAUC of v_star")
        return _sel_uauc(R, "v_star", "toys")
    if c == "c4":
        expect(s, "sel:E2")
        return yesno(R.get(SEL, "table", R.get(SEL, "v_star"), "E2"))
    raise SpecError(f"no slot expected in column {c}")


def _ci_val(est, ci) -> Val:
    if not isinstance(ci, dict):
        return val(est)
    return val({"est": est, "lo": ci.get("lo"), "hi": ci.get("hi"), "n_users": ci.get("n_users", ci.get("n"))},
               src=_src_of(est))


def go_g6(R, c, k, s):
    if c == "c1":
        expect(s, "gate:v0_context.UAUC")
        return fmt(_ci_val(R.get(GATE, "v0_context", "UAUC"), R.get(GATE, "v0_context", "ci95")))
    if c == "c2":
        expect(s, "gate:UAUC, ci95")
        return fmt(_ci_val(R.get(GATE, "UAUC"), R.get(GATE, "ci95")))
    if c == "c4":
        expect(s, "gate:decision")
        return tex(R.get(GATE, "decision"))
    raise SpecError(f"no slot expected in column {c}")


def go_seeds(R, c, k, s):
    expect(s, "gft:UAUC_post_T_per_seed")
    per = R.get(GFT, "UAUC_post_T_per_seed")
    if not isinstance(per, list) or len(per) != 3:
        raise Missing("incomplete_regime", f"{GFT}: UAUC_post_T_per_seed has {per!r} (seeds 0, 1, 2 registered)")
    return " / ".join(num(x) for x in per)


# Main-session decision (2026-10-04): the ML-1M zero-shot UAUC and the paired delta (LoRA - zero-shot) on the Gate-FT rows are
# the grid values (E_A zero-shot, E_B); gate_ft.json's own zero_shot_context is a second scoring of the same prompts and is not
# printed (it is recorded in FILLED.json notes). Editor pass 2026-10-05: tab:gate-outcomes (now in the protocol) no longer
# repeats them; they, and the item-mean and MF references on these rows, are printed once, in the ML-1M cells of tab:tracks A.
# The per-seed LoRA UAUCs, their mean with its interval and the G9 decision stay gate_ft.json's (the registered output).
GATE_ZS_NOTE = ("gate_ft.json zero_shot_context (UAUC_post_T, dUAUC_finetuned_minus_zeroshot_post_T) is a second scoring "
                "of the same V1 prompts on the same users (vLLM batching; |dUAUC| about 1e-4): not printed; the zero-shot "
                "UAUC and the paired gain on these rows are printed once, from the grid report, in tab:tracks "
                "(main-session decision 2026-10-04; editor pass 2026-10-05)")


def go_mean(R, c, k, s):
    if c == "c1":
        expect(s, "gft:zero_shot_context.UAUC_post_T", "grid:UAUC")
        return fmt(regime_val(R, "qwen", "ml1m", "ZS", "E_A", ("UAUC_TEST",)))
    if c == "c2":
        expect(s, "gft:UAUC_post_T_mean_over_seeds, ci95")
        ci = R.get(GFT, "UAUC_post_T_seed_averaged_ci95")
        v = _ci_val(R.get(GFT, "UAUC_post_T_mean_over_seeds"), ci)
        v.sd = R.get(GFT, "UAUC_post_T_sd_over_seeds")
        return fmt(v)
    if c == "c4":
        expect(s, "gft:decision")
        return tex(R.get(GFT, "decision"))
    raise SpecError(f"no slot expected in column {c}")


def go_caption(R, s):
    expect(s, "gate:n_users")
    return integer(R.get(GATE, "n_users"))


# ---- tab:anatomy
ANATOMY_ROWS = {  # row label -> (slot text, path inside questions[next], estimate only)
    r"Temperature $T$": ("aud:temperature.T", None, True),
    r"ECE (top label)": ("aud:C.ece", ("C_calibration", "list_normalised", "ece"), False),
    r"Brier": ("aud:C.brier", ("C_calibration", "list_normalised", "brier"), False),
    r"$\mathrm{acc}_{\rm low}$": ("aud:C.unsure_correct_rate", ("C_calibration", "error_anatomy", "unsure_correct_rate"),
                                  False),
    r"$\mathrm{acc}_{\rm high}$": ("aud:C.tertile2_accuracy", ("C_calibration", "error_anatomy", "tertiles", 2,
                                                               "accuracy"), False),
    r"Errors in high tertile": ("aud:C.share_errors_in_top_tertile", ("C_calibration", "error_anatomy",
                                                                       "share_errors_in_top_tertile"), False),
    r"AUROC($p_{\max}$)": ("aud:C.auroc_p_max_top1", ("C_calibration", "auroc_discrimination", "top1", "auroc", "p_max"),
                           False),
    r"AUROC($\ell$), pooled": ("aud:C.pointwise.pooled_auroc", ("C_calibration", "pointwise", "pooled_auroc"), False),
    r"UAUC (per event)": ("aud:C.pointwise.uauc", ("C_calibration", "pointwise", "uauc"), False),
    r"NDCG@10": ("aud:A.ranking.ndcg10", ("A_ranking", "ranking", "ndcg10"), False),
    r"Mean $p$, head $-$ tail": ("aud:E.head_minus_tail_mean_p", ("E_popularity", "head_minus_tail_mean_p"), False),
    r"Bias Index, head $-$ tail": ("aud:E.bias_index.head_minus_tail", ("E_popularity", "bias_index",
                                                                        "head_minus_tail"), False),
    r"Top-1 is head (share)": ("aud:E.top1_head_fraction", ("E_popularity", "top1_head_fraction"), False),
    # block C (former tab:serving): estimates only (FILL RULE 2)
    r"NDCG@10 gain, $p_{\max}$": ("aud:D.gain_at_50_vs_full", ("D_selective_serving", "signals", "p_max",
                                                               "gain_at_50_vs_full", "ndcg10"), True),
    r"NDCG@10 gain, random": ("aud:D.gain_at_50_vs_full", ("D_selective_serving", "signals", "random",
                                                           "gain_at_50_vs_full", "ndcg10"), True),
    r"Niche $-$ mainstream, $p_{\max}$": ("aud:D.niche_minus_mainstream_served_share", (
        "D_selective_serving", "signals", "p_max", "niche", "niche_minus_mainstream_served_share"), True),
    r"Niche $-$ mainstream, random": ("aud:D.niche_minus_mainstream_served_share", (
        "D_selective_serving", "signals", "random", "niche", "niche_minus_mainstream_served_share"), True),
}


def anatomy_handler(label):
    slot_text, path, est_only = ANATOMY_ROWS[label]

    def h(R, c, k, s):
        expect(s, slot_text)
        d, seg = ANATOMY_UNITS[c]
        rel = f"aud/{d}.json"
        if path is None:
            return num(R.get(rel, "temperature", "next", "T"))
        v = val(_aud_rec(R, rel, seg, "next", *path), n_key="n_users")
        v.est_only = est_only
        return fmt(v)
    return h


# ---- tab:reliability (Qwen) and tab:app-llama (Llama): shared rows
def _p1_cell(R, bb, d, prose=False) -> str:
    rel = grid_rel(bb, d)
    require_ft(R)
    if R.get(rel, "P1", "decision", "verdict") == "INCOMPLETE" or R.get(rel, "P1", "available") is not True:
        raise Missing("incomplete_regime", f"{rel}: P1 is INCOMPLETE ({R.get(rel, 'P1', 'decision').get('reason')})")
    per = [R.get(rel, "P1", "per_seed", m, "est") for m in FT_MODELS]
    mean = val(R.get(rel, "P1", "mean_over_seeds"))
    p = R.get(rel, "P1", "mean_over_seeds").get("p")
    ptxt = ("$p$ " + pval(p)) if _fin(p) else "no $p$ (descriptive)"
    if prose:
        return f"{fmt(mean, prose=True)}, {ptxt}"
    return " / ".join(num(x) for x in per) + "; " + fmt(mean) + ", " + ptxt


def eb_cell(R, bb: str, d: str, need_ft: bool = True) -> str:
    """E-B (A3 section 3): the seed-averaged dUAUC(FT - ZS) with its interval and the s.d. of the per-seed contrasts."""
    rel = grid_rel(bb, d)
    if need_ft:
        require_ft(R)
    eb = R.get(rel, "E_B")
    if eb.get("complete") is False:
        raise Missing("incomplete_regime", f"{rel}: E_B models {eb.get('models')}, missing or excluded "
                                           f"{eb.get('missing_or_excluded')}")
    per = [R.get(rel, "E_B", "per_seed", m, "est") for m in FT_MODELS]
    return fmt(val(R.get(rel, "E_B", "mean_over_seeds"), sd=stdev(per)))


def rel_rows(bb: str, p1_label: str) -> dict:
    def stat(slot_text, blk, sub, key=None, mean_key="mean_over_seeds", per_key="per_model", special=None):
        def h(R, c, k, s):
            expect(s, slot_text)
            d, reg = c
            if reg == "span":
                raise SpecError("a per-regime row has a spanning cell")
            v = regime_val(R, bb, d, reg, blk, sub, key, mean_key, per_key)
            if special == "shares" and v.text is None:
                reading = (R.get(grid_rel(bb, d), "E_D", reg, "shares", "per_model", "zeroshot", "shares_reading")
                           if reg == "ZS" else R.get(grid_rel(bb, d), "E_D", reg, "shares", "mean_over_seeds",
                                                     "shares_reading"))
                if reading == "uninterpretable":
                    return "uninterpretable"
            return fmt(v)
        return h

    def span(fn):
        def h(R, c, k, s):
            d, reg = c
            if reg != "span":
                raise SpecError("a spanning row has a per-regime cell")
            return fn(R, d, s)
        return h

    def e_b(R, d, s):
        expect(s, "grid:dUAUC_ft_minus_zs")
        return eb_cell(R, bb, d)

    def ref(name, slot_text):
        def f(R, d, s):
            expect(s, slot_text)
            return fmt(ref_val(R, bb, d, name))
        return f

    def n_users(R, d, s):
        expect(s, "grid:n_users")
        return integer(R.get(split_rel(bb, d), "eval", "users_both_classes_test"))

    def n_sd(R, d, s):
        expect(s, "grid:n_sd")
        return integer(R.get(split_rel(bb, d), "sd", "users"))

    def p1(R, d, s):
        expect(s, "grid:P1")
        if d != "ml1m":
            raise SpecError("P1 is an ML-1M cell")
        return _p1_cell(R, bb, d)

    def margin(R, c, k, s):
        expect(s, "grid:auroc_margin_correct, aurc_margin")
        d, reg = c
        return " / ".join(fmt(regime_val(R, bb, d, reg, "E_C", (), key)) for key in ("AUROC_margin_correct", "AURC"))

    def users(R, c, k, s):
        """'Users: TEST / S_d': the users with both classes among TEST candidates (k = 0) and the users of S_d (k = 1)."""
        d, reg = c
        if reg != "span":
            raise SpecError("a spanning row has a per-regime cell")
        if k == 0:
            return n_users(R, d, s)
        if k == 1:
            return n_sd(R, d, s)
        raise SpecError(f"no third slot expected in the users cell ({k})")

    def g_cf_span(R, c, k, s):
        """'G_CF: E-D / E-W': the registered G_CF of E-D (k = 0; no LLM feature, one value per panel, read from the zero-shot
        regime's rows and from the fine-tuned regime's when the zero-shot block is unavailable) and the within-user G_CF,wu of
        the addendum-6 analysis (k = 1, alias ext, the same rows rule)."""
        d, reg = c
        if reg != "span":
            raise SpecError("a spanning row has a per-regime cell")
        if k == 1:
            expect(s, "ext:E_W.G_CF_wu")
            return ext_g_cf_wu(R, ext_rel(bb, d))
        if k != 0:
            raise SpecError(f"no third slot expected in the G_CF cell ({k})")
        expect(s, "grid:G_CF")
        h, first = g_cf_handler(bb), None
        for reg_ in ("ZS", "FT"):
            try:
                out = h(R, (d, reg_), 0, s)
            except Missing as e:
                if e.reason == "result_file_missing":
                    raise
                first = first or e
                continue
            if out != r"FAILED\_INTEGRITY":
                return out
            first = first or out
        if isinstance(first, str):
            return first
        raise first

    rows = {
        # ---- tab:tracks block A (and tab:app-llama, which uses the same labels)
        r"UAUC of $\ell$": stat("grid:UAUC", "E_A", ("UAUC_TEST",)),
        r"LoRA $-$ ZS (E-B)": span(e_b),
        r"item mean $m$ (ref.)": span(ref("q_hat", "cpu:UAUC_item_mean_prior")),
        r"matched mean $m_T$ (ref.)$^\dagger$": ext_span_q_hat_T(bb),
        r"temporal MF (ref.)": span(ref("mf", "cpu:UAUC_mf_temporal")),
        r"$\Delta$UAUC($\ell-m$), E-J$^\dagger$": ext_stat(bb, "ext:E_J.dUAUC_L_minus_q_hat", "E_J", ("dUAUC_L_minus_q_hat",)),
        r"$\Delta$UAUC($\ell-$MF), warm$^\dagger$": ext_stat(bb, "ext:E_J.dUAUC_L_minus_mf", "E_J", ("dUAUC_L_minus_MF_warm",)),
        r"Users: TEST / $S_d$": users,
        # ---- block B
        r"Reliability $r_8$ of $\hat\pi$": stat("grid:r8c", "E_D", ("shares",), "r8c"),
        r"Item-prior share $\rho^2/r_8$": stat("grid:item_prior_share", "E_D", ("shares",), "item_prior_share",
                                               special="shares"),
        r"e-share$^\dagger$": ext_stat(bb, "ext:E_G.e_share", "E_G", ("e_share",), leaf="e_share"),
        r"Item share of MF / label$^\dagger$": ext_item_shares(bb),
        # G and the star permutation keep their seed mean under their own keys (G_mean_over_seeds,
        # dUAUC_mean_over_seeds; ftgrid_report.stacker_block / starperm_block)
        r"$\mathcal G=\Delta$UAUC(M2$-$M1), E-D": mean_keyed(bb, "grid:G", ("information_gain",), "G",
                                                             "G_mean_over_seeds"),
        r"$\mathcal G_{\rm wu}$ (E-W)$^\dagger$": ext_stat(bb, "ext:E_W.G_wu", "E_W", ("G_wu",)),
        r"$\mathcal G_{\rm CF}$: E-D / E-W$^\dagger$": g_cf_span,
        r"MF personal residual (ref.)": span(ref("mf_personal_residual_warm", "cpu:UAUC_mf_personal_residual")),
        r"Star permutation $\Delta$UAUC": mean_keyed(bb, "grid:starperm_dUAUC", ("star_permutation",),
                                                     "dUAUC_L_minus_perm", "dUAUC_mean_over_seeds"),
        r"Popularity link of $\hat\pi$": stat("grid:partial_rho_pi_logpop", "E_E", ("partial_spearman", "pi_item")),
        r"Popularity link of $\ell$": stat("grid:partial_rho_L_logpop", "E_E", ("partial_spearman", "L_item")),
        p1_label: span(p1),
        r"P1$_{\rm wu}$$^\dagger$": ext_p1_wu(bb),
        # ---- block C (oracle rows: E-C of the grid report; deployable rows: E-C' of the addendum-6 analysis). The UAUC of pi-hat
        # alone, the Platt slope and the AURC stay in the released files (editor pass 2026-10-05, page budget).
        r"ECE (after Platt)": stat("grid:ece10_platt", "E_C", (), "ECE"),
        r"Correct in bottom tertile, oracle": stat("grid:share_correct_bottom_tertile", "E_C", (),
                                                   "share_correct_bottom"),
        r"Errors in top tertile, oracle": stat("grid:share_errors_top_tertile", "E_C", (), "share_errors_top"),
        r"Margin AUROC, oracle": stat("grid:auroc_margin_correct", "E_C", (), "AUROC_margin_correct"),
        # the three deployable rows share their label; the slot text names the statistic
        r"deployable$^\dagger$": ext_deployable(bb),
        # ---- tab:app-llama: the Llama Toys knockout (alias ko of the Llama grid report)
        r"Knockout: head $-$ tail drop of $\ell$": ko_handler(r"Head $-$ tail drop of $\ell$", bb),
        r"Knockout: registered label": ko_handler(r"Registered label (per seed)", bb),
    }
    return rows


# ================================================================================================ alias ext
# The addendum-6 analysis (src/confrec/ftgrid_extra.py; A3-6 items 2-8, addendum 8 for FT-Q) at the local paths of
# scripts/sigir/pull_results.ps1: extra/<d>.json (Qwen main root), extra/llama/<d>.json (Llama root), extra/ftq/<d>.json (the FT-Q
# teacher root) and extra/summary.json (the A3-6 Holm families E-F, E-H, E-J, the robust readings, the addendum-8 wording). Editor
# pass 2 (2026-10-05): written against the module's real schema (its readings R1-R19, `summarize`, the tests of
# tests/test_confrec_ftgrid_extra.py) and the real ML-1M file. A block the file marks unavailable stays red with its own reason.
EXT_SUM = "extra/summary.json"


def ext_rel(bb: str, d: str) -> str:
    """The domain file of a backbone's root: the Qwen main root extra/<d>.json, the Llama root extra/llama/<d>.json."""
    return f"extra/{d}.json" if bb == "qwen" else f"extra/{bb}/{d}.json"


def ftq_rel(d: str) -> str:
    """The FT-Q teacher root's domain file (addendum 8; its FT_C_reading block is the FT-Q reading R_Q)."""
    return f"extra/ftq/{d}.json"


def ext_mark(R, rel: str, text: str) -> str:
    """A3-6 Consequence (ftgrid_extra R1): items 2-7 are exploratory on ML-1M (either backbone) and registered, outcome-free on every
    other panel. An exploratory value is set in italics; the status is read from the file (status.items_2_to_7), never from the
    column."""
    st = R.get(rel, "status", "items_2_to_7")
    if st == "exploratory":
        return r"\textit{" + text + "}"
    if st != "registered_outcome_free":
        raise Missing("field_not_produced", f"{rel}: status.items_2_to_7 is {st!r} (exploratory or registered_outcome_free)")
    return text


def ext_zs_failed(R, rel: str) -> bool:
    """True when the extra file excluded the zero-shot model's like or swap arm for E1 (the cell reads FAILED_INTEGRITY)."""
    try:
        runs = R.get(rel, "runs", "zeroshot")
    except Missing:
        return False
    return any((runs.get(a) or {}).get("status") == "FAILED_INTEGRITY" for a in ("like", "swap"))


def ext_regime_val(R, rel: str, block: str, reg: str, sub: tuple = (), leaf: str | None = None,
                   mean_leaf: str | None = None) -> Val:
    """A per-regime statistic of an extra file: <block>.<reg>.<sub...> holds per-model records (per_model; per_seed in E-J) and
    their mean_over_seeds (ftgrid_extra's seed summaries). ZS: the zeroshot record (at its leaf); FT: the seed mean with its
    interval and the s.d. (ddof 1) of the three seed estimates; fewer than the registered seeds -> incomplete_regime (never
    replaced, as the grid cells)."""
    if reg == "FT":
        require_ft(R)
    try:
        node = R.get(rel, block, reg)
        if reg == "FT" and node.get("complete") is False:
            raise Missing("incomplete_regime", f"{rel}: {block}.FT has models {node.get('models')}, missing or excluded "
                                               f"{node.get('missing_or_excluded')} (never replaced)")
        blk = R.get(rel, block, reg, *sub)
        per_key = "per_model" if isinstance(blk, dict) and "per_model" in blk else "per_seed"
        lf = (leaf,) if leaf else ()
        if reg == "ZS":
            return val(R.get(rel, block, reg, *sub, per_key, "zeroshot", *lf))
        have = R.get(rel, block, reg, *sub, per_key)
        if any(m not in have for m in FT_MODELS):
            raise Missing("incomplete_regime", f"{rel}: {'.'.join((block, reg) + tuple(sub))} has seeds {sorted(have)} "
                                               "(s0-s2 registered; a missing seed is never replaced)")
        per = [R.get(rel, block, reg, *sub, per_key, m, *lf, "est") for m in FT_MODELS]
        ml = (mean_leaf,) if mean_leaf else ()
        return val(R.get(rel, block, reg, *sub, "mean_over_seeds", *ml), sd=stdev(per))
    except Missing as e:
        if reg == "ZS" and e.reason in ("result_not_in_report", "field_not_produced") and ext_zs_failed(R, rel):
            return Val(text=r"FAILED\_INTEGRITY")
        raise


def _ext_cell(R, c, bb: str, block: str, sub: tuple, leaf: str | None = None, mean_leaf: str | None = None) -> str:
    d, reg = c
    if reg == "span":
        raise SpecError("a per-regime row has a spanning cell")
    rel = ext_rel(bb, d)
    v = ext_regime_val(R, rel, block, reg, sub, leaf, mean_leaf)
    return v.text if v.text is not None else ext_mark(R, rel, fmt(v))


def ext_stat(bb: str, slot_text: str, block: str, sub: tuple = (), leaf: str | None = None, mean_leaf: str | None = None):
    """A per-regime row of tab:tracks, tab:teaches or tab:app-llama read from the extra file of the backbone's root."""
    def h(R, c, k, s):
        expect(s, slot_text)
        return _ext_cell(R, c, bb, block, sub, leaf, mean_leaf)
    return h


EXT_DEPLOYABLE = {"ext:E_Cprime.share_correct_bottom": "share_correct_bottom",
                  "ext:E_Cprime.share_errors_top": "share_errors_top",
                  "ext:E_Cprime.AUROC_margin": "AUROC_margin_correct"}


def ext_deployable(bb: str):
    """The three deployable rows of E-C' (A3-6 item 7) share one row label: the slot text names the statistic of
    E_Cprime.<reg>.deployable."""
    def h(R, c, k, s):
        key = EXT_DEPLOYABLE[expect(s, *EXT_DEPLOYABLE)]
        return _ext_cell(R, c, bb, "E_Cprime", ("deployable",), key, key)
    return h


def ext_span_q_hat_T(bb: str):
    """'matched mean m_T (ref.)': the UAUC of q-hat_T on E-A's TEST rows, i.e. UAUC_b of E-J's dUAUC(L - q-hat_T) record (an
    estimate: the file holds no interval for it); one value per panel, from the zero-shot rows, else the fine-tuned rows."""
    def h(R, c, k, s):
        expect(s, "ext:E_J.UAUC_q_hat_T")
        d, reg = c
        if reg != "span":
            raise SpecError("a spanning row has a per-regime cell")
        rel, first = ext_rel(bb, d), None
        for reg_, m in (("ZS", "zeroshot"), ("FT", "s0")):
            try:
                x = R.get(rel, "E_J", reg_, "dUAUC_L_minus_q_hat_T", "per_seed", m, "UAUC_b")
            except Missing as e:
                if e.reason == "result_file_missing":
                    raise
                first = first or e
                continue
            return ext_mark(R, rel, fmt(val(x)))
        raise first
    return h


def ext_item_shares(bb: str):
    """'Item share of MF / label' (E-G ii and iii; once per panel): the MF score with its item bias (k = 0) and the label with
    q-hat (k = 1), E_G.<MF_score_item_bias | label_q_hat>.item_share."""
    def h(R, c, k, s):
        d, reg = c
        if reg != "span":
            raise SpecError("a spanning row has a per-regime cell")
        if k > 1:
            raise SpecError(f"no third slot expected in the item-share cell ({k})")
        name, blk = (("ext:E_G.item_share_mf", "MF_score_item_bias"), ("ext:E_G.item_share_label", "label_q_hat"))[k]
        expect(s, name)
        rel = ext_rel(bb, d)
        return ext_mark(R, rel, fmt(val(R.get(rel, "E_G", blk, "item_share"))))
    return h


def ext_g_cf_wu(R, rel: str) -> str:
    """G_CF,wu = dUAUC(M3 - M0) of E-W (no LLM feature: one value per panel), from the zero-shot rows, else the fine-tuned rows."""
    first = None
    for reg in ("ZS", "FT"):
        try:
            return ext_mark(R, rel, fmt(val(R.get(rel, "E_W", reg, "G_CF_wu"))))
        except Missing as e:
            if e.reason == "result_file_missing":
                raise
            first = first or e
    raise first


def ext_p1_wu(bb: str):
    """P1_wu (A3-6 item 2): P1's quantity with the within-user estimator, an ML-1M cell as P1's: the three seed contrasts, their
    mean with its interval and p (a sensitivity estimate; P1 itself is the grid row above)."""
    def h(R, c, k, s):
        expect(s, "ext:E_W.P1_wu")
        d, reg = c
        if reg != "span" or d != "ml1m":
            raise SpecError("P1_wu is an ML-1M spanning cell")
        require_ft(R)
        rel = ext_rel(bb, d)
        R.get(rel, "E_W", "P1_wu")                    # available false -> result_not_in_report with its reason
        per = [R.get(rel, "E_W", "P1_wu", "per_seed", m, "est") for m in FT_MODELS]
        rec = R.get(rel, "E_W", "P1_wu", "mean_over_seeds")
        p = rec.get("p")
        ptxt = ("$p$ " + pval(p)) if _fin(p) else "no $p$ (descriptive)"
        return ext_mark(R, rel, " / ".join(num(x) for x in per) + "; " + fmt(val(rec)) + ", " + ptxt)
    return h


def ext_reading_cell(R, rel: str, control: str, kind: str) -> str:
    """A cell of an FT-C / FT-Q reading (A3-6 item 8; addendum 8): the FT_C_reading block of the file, which must be the control
    the row names; an unavailable block stays red with the file's own reason. kind R: the retention with its interval, or 'not
    defined' when the file records that R is not defined (R13; its label then reads NOT_DEFINED); label: the label verbatim; UAUC:
    addendum 8's descriptive companion, the UAUC of the two control adapters and of q-hat on the same rows (estimates)."""
    require_ft(R)
    b = R.get(rel, "FT_C_reading")
    if b.get("control") != control:
        raise Missing("field_not_produced", f"{rel}: FT_C_reading is the {b.get('control')!r} control, not {control}")
    if kind == "label":
        return tex(R.get(rel, "FT_C_reading", "label"))
    if kind == "UAUC":
        return " / ".join([num(R.get(rel, "FT_C_reading", "UAUC", m, "est")) for m in ("p0", "p1")]
                          + [num(R.get(rel, "FT_C_reading", "UAUC_q_hat", "est"))])
    defined = R.get(rel, "FT_C_reading", "defined")
    if defined is False:
        return "not defined"
    if defined is not True:
        raise Missing("field_null", f"{rel}: FT_C_reading.defined is {defined!r}")
    return fmt(val(R.get(rel, "FT_C_reading", "R")))


def ext_ftc(R, c, k, s):
    """'FT-C (permuted): R / reading': the ML-1M LoRA cell only (addendum 8 withdrew the Toys run): R (k = 0), label (k = 1)."""
    d, reg = c
    if (d, reg) != ("ml1m", "FT") or k > 1:
        raise SpecError("FT-C is the ML-1M LoRA cell (R / label)")
    kind = ("R", "label")[k]
    expect(s, f"ext:FT_C_reading.{kind}")
    return ext_reading_cell(R, ext_rel("qwen", "ml1m"), "FT-C", kind)


def ext_ftq(kind: str):
    """The FT-Q rows (addendum 8): the LoRA cell of each dataset, read from the teacher root's file."""
    def h(R, c, k, s):
        d, reg = c
        if reg != "FT":
            raise SpecError("FT-Q is a LoRA cell")
        expect(s, f"ext:FT_Q_reading.{kind}")
        return ext_reading_cell(R, ftq_rel(d), "FT-Q", kind)
    return h


def mean_keyed(bb, slot_text, sub, per_key_leaf, mean_leaf):
    def h(R, c, k, s):
        expect(s, slot_text)
        d, reg = c
        rel = grid_rel(bb, d)
        if reg == "FT":
            require_ft(R)
        if reg == "ZS":
            try:
                regime_ok(R, rel, "E_D", reg)
                return fmt(val(R.get(rel, "E_D", reg, *sub, "per_model", "zeroshot", per_key_leaf)))
            except Missing as e:
                if e.reason in ("result_not_in_report", "field_not_produced") and zs_failed(R, rel, arms_of("E_D", sub)):
                    return r"FAILED\_INTEGRITY"
                raise
        regime_ok(R, rel, "E_D", reg)
        have = R.get(rel, "E_D", reg, *sub, "per_model")
        if any(m not in have for m in FT_MODELS):
            raise Missing("incomplete_regime", f"{rel}: E_D.FT.{'.'.join(sub)} has seeds {sorted(have)} (s0-s2 "
                                               "registered; never replaced)")
        per = [R.get(rel, "E_D", reg, *sub, "per_model", m, per_key_leaf, "est") for m in FT_MODELS]
        return fmt(val(R.get(rel, "E_D", reg, *sub, mean_leaf), sd=stdev(per)))
    return h


def g_cf_handler(bb):
    def h(R, c, k, s):
        expect(s, "grid:G_CF")
        d, reg = c
        rel = grid_rel(bb, d)
        if reg == "FT":
            require_ft(R)
        try:
            regime_ok(R, rel, "E_D", reg)
            return fmt(val(R.get(rel, "E_D", reg, "information_gain", "G_CF")))   # no seed variation: no s.d.
        except Missing as e:
            if reg == "ZS" and e.reason in ("result_not_in_report", "field_not_produced") and zs_failed(R, rel, ("like", "swap")):
                return r"FAILED\_INTEGRITY"
            raise
    return h


# ---- tab:exposure
EXPOSURE_UNITS = ("S1k", "S10k", "Toys", "Home", "Tools")


def exposure_col(start, span):
    if span != 1:
        return None
    if 1 <= start <= 5:
        return ("dh", EXPOSURE_UNITS[start - 1])
    if 7 <= start <= 11:
        return ("gini", EXPOSURE_UNITS[start - 7])
    if 13 <= start <= 17:
        return ("aplt", EXPOSURE_UNITS[start - 13])
    return None


def exposure_handler(label):
    def h(R, c, k, s):
        blk, unit = c
        d, seg = ANATOMY_UNITS[unit]
        rel = f"aud/{d}.json"
        if label in (r"Pool (level)", r"Target head share (level)"):
            want = {("Pool (level)", "dh"): ("aud:B.pool_head_share", "pool_head_share"),
                    ("Pool (level)", "aplt"): ("aud:B.pool_tail_share", "pool_tail_share"),
                    ("Target head share (level)", "dh"): ("aud:B.target_head_share", "target_head_share")}
            if (label, blk) not in want:
                raise SpecError(f"no slot expected in the {blk} block of row {label}")
            st, key = want[(label, blk)]
            expect(s, st)
            v = val(_aud_rec(R, rel, seg, "next", "B_exposure", "llm", key), n_key="n_users")
        else:
            st, key = {"dh": ("aud:B.delta_head", "delta_head"), "gini": ("aud:B.gini_exposure", "gini_exposure"),
                       "aplt": ("aud:B.tail_share_top10", "tail_share_top10")}[blk]
            expect(s, st)
            if label.startswith("LLM, "):
                q = {r"LLM, \textsf{next}": "next", r"LLM, \textsf{like}": "like"}[label]
                v = val(_aud_rec(R, rel, seg, q, "B_exposure", "llm", key), n_key="n_users")
            else:
                v = val(R.get(rel, "segments", seg, "reference", REF_METHODS[label], "B_exposure", key),
                        n_key="n_users")
        v.est_only = blk in ("gini", "aplt")            # FILL RULE 2: Gini and APLT blocks give estimates only
        return fmt(v)
    return h


# Editor pass 2026-10-05 (page budget): the eight published recommenders of tab:exposure are summarised per cell by the minimum,
# median and maximum of their estimates (context, estimates only); their per-method values stay in the released files, and the
# verbalised reranker keeps its own row. c_rq3_baselines still reads all nine reference methods.
PUBLISHED = tuple(m for lab, m in REF_METHODS.items() if lab != r"Verbalised reranker")
assert len(PUBLISHED) == 8, PUBLISHED
EXPOSURE_SUMMARY = {r"Published, min": "min", r"Published, median": "median", r"Published, max": "max"}


def exposure_summary_handler(label):
    stat = EXPOSURE_SUMMARY[label]

    def h(R, c, k, s):
        blk, unit = c
        d, seg = ANATOMY_UNITS[unit]
        rel = f"aud/{d}.json"
        st, key = {"dh": ("aud:B.delta_head", "delta_head"), "gini": ("aud:B.gini_exposure", "gini_exposure"),
                   "aplt": ("aud:B.tail_share_top10", "tail_share_top10")}[blk]
        expect(s, f"{st}, {stat} of 8")
        ests = []
        for m in PUBLISHED:
            rec = R.get(rel, "segments", seg, "reference", m, "B_exposure", key)
            est = rec.get("est") if isinstance(rec, dict) else rec
            if not _fin(est):
                raise Missing("field_null", f"{rel}: {seg}.reference.{m}.B_exposure.{key} has no estimate")
            ests.append(float(est))
        xs = sorted(ests)
        value = {"min": xs[0], "max": xs[-1], "median": (xs[3] + xs[4]) / 2}[stat]
        return num(value, src=f"{rel}:segments.{seg}.reference.(8 published).B_exposure.{key}.est ({stat})")
    return h


# ---- knockout rows (tab:teaches block D, Qwen; tab:app-llama, Llama Toys; the former tab:popularity)
KO_ROWS = {r"Head $-$ tail drop of $\ell$": ("ko:head_minus_tail.delta", ("delta", "pseudo", "head_minus_tail")),
           r"Placebo drop (real-brand swap)": ("ko:placebo.delta", ("delta", "placebo", "head_minus_tail")),
           r"Tail $\Delta$UAUC": ("ko:tail_dUAUC", ("dUAUC", "real_minus_pseudo", "tail")),
           r"Overall $\Delta$UAUC": ("ko:overall_dUAUC", ("dUAUC", "real_minus_pseudo", "all")),
           r"Registered label (per seed)": ("ko:label", None)}


def _ko_n(rec):
    n = rec.get("n_clusters", rec.get("n"))
    return int(n) if _fin(n) else None


def ko_handler(label, bb="qwen"):
    slot_text, path = KO_ROWS[label]

    def h(R, c, k, s):
        expect(s, slot_text)
        require_ft(R)                                 # A3 section 5 is part of the fine-tuned program (rule 4b)
        d, reg = c
        rel = grid_rel(bb, d)
        R.get(rel, "knockout")
        models = ("zeroshot",) if reg == "ZS" else FT_MODELS
        if path is None:
            return " / ".join(tex(R.get(rel, "knockout", "labels", m, "label")) for m in models)
        recs = [R.get(rel, "knockout", "analyses", m, *path) for m in models]
        desc = any((_ko_n(r) or 0) < MIN_N for r in recs)
        if reg == "ZS":
            v = val({**recs[0], "n_users": _ko_n(recs[0])})
            return fmt(v)
        ests = [r.get("est") for r in recs]
        if not all(_fin(x) for x in ests):
            raise Missing("field_null", f"{rel}: a seed's knockout estimate is null")
        return fmt(Val(est=sum(ests) / 3, sd=stdev(ests), descriptive=desc, est_only=True,
                       src=f"{rel}:knockout.analyses.(s0,s1,s2 mean).{'.'.join(path)}"))
    return h


# ---- tab:pruning
PRN = "prn/pruning_ml1m.json"
PRUNE_ROWS = {r"P0 full data": "P0", r"P1 random (class-matched)": "P1", r"P2 uncertainty-selected": "P2",
              r"P3 prior-congruent": "P3"}       # names of addendum 7 (2026-10-05)


def _prn_arm(R, arm):
    a = R.get(PRN, "arms", arm)
    if a.get("status") == "NOT_RUN":
        return None
    return a


def _prn_contrast(R, a, b):
    """(rec, sign) of contrast a - b; P0 - P1 is the exact negation of the registered P1 - P0 contrast."""
    cons = R.get(PRN, "contrasts")
    name, sign = (f"{a}-{b}", 1) if f"{a}-{b}" in cons else (f"{b}-{a}", -1)
    if name not in cons:
        raise Missing("field_not_produced", f"{PRN}: no contrast {a}-{b} in the file (ftprune.CONTRASTS: P2-P1, "
                                            f"P3-P1, P1-P0, P2-P0, P3-P0)")
    c = R.get(PRN, "contrasts", name)
    if c.get("status") == "NOT_RUN":
        return None, sign
    if c.get("status") != "OK":
        raise Missing("incomplete_regime", f"{PRN}: contrast {name} is {c.get('status')} (excluded runs "
                                           f"{c.get('excluded_runs')})")
    return c, sign


def prune_handler(label):
    arm = PRUNE_ROWS[label]

    def h(R, c, k, s):
        require_ft(R)
        if c == "UAUC":
            expect(s, "prn:UAUC_mean_sd")
        elif c == "dP1":
            expect(s, "prn:d_vs_P1")
        elif c == "npos":
            expect(s, "prn:n_pos_of_5")
        elif c == "dP0":
            expect(s, "prn:d_vs_P0")
        else:
            raise SpecError(f"no slot expected in column {c}")
        a = _prn_arm(R, arm)
        if a is None:
            return "not run"                          # registered cut recorded by ftprune (A3 section 10)
        if c == "UAUC":
            if not a.get("complete"):
                raise Missing("incomplete_regime", f"{PRN}: arm {arm} seeds ok {a.get('seeds_ok')} (5 registered)")
            v = Val(est=a.get("UAUC_seed_averaged"), sd=a.get("sd_seed"), est_only=True,
                    descriptive=_fin(a.get("n_users_common")) and a["n_users_common"] < MIN_N,
                    src=f"{PRN}:arms.{arm}.UAUC_seed_averaged")
            return fmt(v)
        other = "P1" if c in ("dP1", "npos") else "P0"
        rec, sign = _prn_contrast(R, arm, other)
        if rec is None:
            return "not run"
        if c == "npos":
            per = list((rec.get("per_seed") or {}).values())
            if len(per) != 5 or not all(_fin(x) for x in per):
                raise Missing("incomplete_regime", f"{PRN}: {len(per)} paired seed differences (5 registered)")
            return str(sum(1 for x in per if sign * x > 0))
        v = val(rec)
        if sign < 0:
            v = Val(est=-v.est, lo=-v.hi if v.hi is not None else None, hi=-v.lo if v.lo is not None else None,
                    descriptive=v.descriptive, src=f"{v.src} (negated)")
        return fmt(v)
    return h


def prune_refs(R, c, k, s):
    expect(s, "cpu:UAUC_item_mean_prior, UAUC_popularity, UAUC_mf_temporal")
    require_ft(R)
    return " / ".join(fmt(ref_val(R, "qwen", "ml1m", r, prefer=("FT", "ZS"))) for r in ("q_hat", "popularity", "mf"))


# ---- tab:app-sens (dormant since the editor pass of 2026-10-05: the variant-bank and sensitivity tables were cut to two sentences
# and a pointer to the artefact; the spec is kept so that the table can be restored, or rendered for the artefact, unchanged)
SENS_COLS = {1: "ml1m", 2: "toys", 3: "E1", 4: "E2", 5: "eligible", 6: "tied"}


def sens_handler(variant):
    def h(R, c, k, s):
        if c in ("ml1m", "toys"):
            expect(s, "sel:UAUC")
            return num(R.get(SEL, "table", variant, c, "UAUC"))
        if c == "E1":
            expect(s, "sel:E1")
            return yesno(bool(R.get(SEL, "table", variant, "ml1m", "E1") is True
                              and R.get(SEL, "table", variant, "toys", "E1") is True))
        key = {"E2": "E2", "eligible": "eligible", "tied": "tied_with_max"}[c]
        expect(s, f"sel:{key}")
        return yesno(R.get(SEL, "table", variant, key))
    return h


def sens_ref(R, c, k, s):
    expect(s, "cpu:UAUC_item_mean_prior")
    raise Missing("to_be_removed", f"{REMOVAL_DECISION} (the DEV-user item-mean row of tab:app-sens: no script computes "
                                   "the prior-only item-mean UAUC on the 1,500 burned DEV users)")


# ---- tab:corrections
MIR = "mir/decision.json"


def second_rated(R) -> str:
    require_gatepass(R)
    p = str(R.get(MIR, "inputs", "toys"))           # pilot1_gate --toys <s3>/<second>/pilot_mirror.json
    d = Path(p.replace("\\", "/")).parent.name
    if d not in RATED:
        raise Missing("field_not_produced", f"{MIR}: inputs.toys {p!r} names no rated domain")
    return d


def corr_domain(R, c) -> str:
    return "ml1m" if c == "ml1m" else second_rated(R)


def corr_row(rowno):
    def h(R, c, k, s):
        if rowno == "0":
            expect(s, "grid:UAUC; cpu:UAUC_item_mean_prior, UAUC_mf_temporal")
            if c == "sports":
                raise SpecError("no slot expected in the Sports column")
            d = corr_domain(R, c)
            return " / ".join([fmt(regime_val(R, "qwen", d, "ZS", "E_A", ("UAUC_TEST",)))]
                              + [fmt(ref_val(R, "qwen", d, r, prefer=("ZS",))) for r in ("q_hat", "mf")])
        if rowno == "1" and c == "sports":
            expect(s, "corr:max_abs_dNDCG_event")
            raise Missing("to_be_removed", f"{REMOVAL_DECISION} (Sports max_abs_dNDCG_event: no script writes it)")
        if rowno in ("1", "2"):
            key = {"1": "max_abs_dAUC_user", "2": "dAUC_pooled_offset_removal"}[rowno]
            expect(s, f"corr:{key}")
            d = corr_domain(R, c)
            rel = grid_rel("qwen", d)
            regime_ok(R, rel, "E_A", "ZS")
            return num(R.get(rel, "E_A", "ZS", "invariance", "per_model", "zeroshot", key))
        if rowno in ("3", "4"):
            key = {"3": "dUAUC_nohist_minus_raw", "4": "dUAUC_evidence_minus_raw"}[rowno]
            expect(s, f"corr:{key}")
            d = corr_domain(R, c)
            rel = grid_rel("qwen", d)
            regime_ok(R, rel, "E_D", "ZS")
            return fmt(val(R.get(rel, "E_D", "ZS", "corrections", key, "per_seed", "zeroshot")))
        if rowno in ("5", "6"):
            key = {"5": "dUAUC_mirror_minus_placebo", "6": "dUAUC_mirror_minus_ensemble_null"}[rowno]
            expect(s, f"mir:{key}")
            require_gatepass(R)
            name = "ml1m" if c == "ml1m" else "toys"   # pilot1_gate.RATED: the second rated domain sits under "toys"
            if c != "ml1m":
                second_rated(R)
            return fmt(val(R.get(MIR, "criteria", name, key), n_key="n"))
        if rowno == "7":
            expect(s, "mir:dNDCG10_mirror_minus_raw")
            require_gatepass(R)
            rec = R.get(MIR, "next_item_no_loss", "dNDCG@10_mirror_minus_raw", "tie_exact")   # the registered metric
            return fmt(val(rec, n_key="n"))
        raise SpecError(f"row {rowno}: no slot expected in column {c}")
    return h


def corr_caption(R, s):
    expect(s, "mir:second_rated_domain")
    return RATED_NAME[second_rated(R)]


# ---- tab:slot
SLOT = "slot/slot.json"


def slot_not_run(R, d) -> bool:
    """True when ftmethod_report's slot state records the slot killed before dataset d (A3 section 7)."""
    st = R.get(SLOT, "state")
    info = R.get(SLOT, "datasets", d)
    return st == "KILLED" and info.get("status") is None


def slot_handler(label):
    def h(R, c, k, s):
        require_ft(R)
        d = c
        if label.startswith("SFT"):
            expect(s, "grid:UAUC")
            return fmt(regime_val(R, "qwen", d, "FT", "E_A", ("UAUC_TEST",)))
        if label.startswith("Item mean"):
            expect(s, "cpu:UAUC_item_mean_prior, UAUC_mf_temporal")
            return " / ".join(fmt(ref_val(R, "qwen", d, r, prefer=("FT", "ZS"))) for r in ("q_hat", "mf"))
        part = {r"Post-hoc stacking": "post_hoc_stacking", r"Prior-offset LoRA": "prior_offset_LoRA",
                r"Offset $-$ stacking": "difference"}[label]
        expect(s, "slot:d_offset_minus_stack" if part == "difference" else "slot:UAUC_mean_sd")
        if slot_not_run(R, d):
            return "not run"
        rel = f"slot/{d}.json"
        status = R.get(rel, "decision", "status")
        if status in ("INCOMPLETE", "INVALID"):
            raise Missing("incomplete_regime", f"{rel}: decision {status} ({R.get(rel, 'decision').get('reason')})")
        if part == "difference":
            per = [R.get(rel, "slot", "difference", "per_seed", f"seed{i}", "est") for i in range(3)]
            v = val(R.get(rel, "slot", "difference", "mean_over_seeds"))
            lo, hi = min(per), max(per)
            base = fmt(v)
            rng = f"({short(lo)} to {short(hi)})"
            return base[:-1] + r"\," + rng + "}" if base.endswith("}") else base + r"{\scriptsize\," + rng + "}"
        models = list(R.get(rel, "slot", part, "per_model"))
        if len(models) != 3:
            raise Missing("incomplete_regime", f"{rel}: {part} has seeds {models} (3 registered)")
        per = [R.get(rel, "slot", part, "per_model", m, "est") for m in models]
        return fmt(val(R.get(rel, "slot", part, "mean_over_seeds"), sd=stdev(per)))
    return h


# ---- tab:app-seeds
SEEDS_DOMAIN_ROWS = {"ML-1M": "ml1m", "Toys": "toys", "Video Games": "games", "Sports": "sports"}


def _seed_value(R, rel, blk_path, m):
    """A per-seed value of the FT regime, or FAILED_INTEGRITY when the report excluded that run for E1."""
    try:
        return num(R.get(rel, *blk_path, "per_model", m, "est"))
    except Missing as e:
        if e.reason == "result_file_missing":
            raise
        st = R.get(rel, "runs", m, "like", "status") if m in (R.get(rel, "runs") or {}) else None
        if st == "FAILED_INTEGRITY":
            return r"FAILED\_INTEGRITY"
        raise


def seeds_handler(label):
    def h(R, c, k, s):
        require_ft(R)
        if label in SEEDS_DOMAIN_ROWS:
            d = SEEDS_DOMAIN_ROWS[label]
            rel = grid_rel("qwen", d)
            if c == "UAUC":
                expect(s, f"grid:UAUC seed {k}")
                R.get(rel, "E_A", "FT")
                return _seed_value(R, rel, ("E_A", "FT", "UAUC_TEST"), f"s{k}")
            if c == "all":
                expect(s, f"grid:UAUC_all_rows seed {k}")
                R.get(rel, "E_A", "FT")
                return _seed_value(R, rel, ("E_A", "FT", "secondary", "UAUC_all_rows"), f"s{k}")
            if c == "ref":
                expect(s, "cpu:UAUC_item_mean_prior")
                return fmt(ref_val(R, "qwen", d, "q_hat", prefer=("FT", "ZS")))
            raise SpecError(f"no slot expected in column {c}")
        rel = grid_rel("qwen", "ml1m")
        part = "TEST" if c == "UAUC" else "all_rows"
        if label.startswith("ML-1M, permuted labels"):
            expect(s, f"grid:ftc_UAUC seed {k}" if c == "UAUC" else f"grid:ftc_UAUC_all_rows seed {k}")
            return num(R.get(rel, "FT_C", f"seed{k}", part, "per_seed", f"s{k}", "UAUC_b"))
        expect(s, f"grid:ftc_dUAUC_real_minus_perm seed {k}" if c == "UAUC" else f"grid:ftc_dUAUC_all_rows seed {k}")
        return fmt(val(R.get(rel, "FT_C", f"seed{k}", part, "per_seed", f"s{k}")))
    return h


# ---- tab:app-z2
Z2_ROWS = {
    r"$\mathrm{acc}_{\rm high}-\mathrm{acc}_{\rm low}$ ($p_{\max}$ tertiles)": (
        "C.acc_top_minus_bottom_tertile", ("C_calibration", "error_anatomy", "acc_top_minus_bottom_tertile")),
    r"Top-1 errors in the top tertile": ("C.share_errors_in_top_tertile", ("C_calibration", "error_anatomy",
                                                                           "share_errors_in_top_tertile")),
    r"$\Delta_{\rm head}$, \textsf{next}": ("B.delta_head", ("B_exposure", "llm", "delta_head")),
    r"Gini of exposure, \textsf{next}": ("B.gini_exposure", ("B_exposure", "llm", "gini_exposure")),
    r"NDCG@10 gain, confident half ($p_{\max}$)": ("D.gain_at_50_vs_full", ("D_selective_serving", "signals", "p_max",
                                                                            "gain_at_50_vs_full", "ndcg10")),
    r"Niche $-$ mainstream served share ($p_{\max}$)": ("D.niche_minus_mainstream_served_share", (
        "D_selective_serving", "signals", "p_max", "niche", "niche_minus_mainstream_served_share")),
    r"Mean $p$, head $-$ tail": ("E.head_minus_tail_mean_p", ("E_popularity", "head_minus_tail_mean_p")),
    r"Bias Index, head $-$ tail": ("E.bias_index.head_minus_tail", ("E_popularity", "bias_index", "head_minus_tail")),
}


def z2_handler(label):
    field_, path = Z2_ROWS[label]

    def h(R, c, k, s):
        expect(s, f"aud2q:{field_} / aud2l:{field_}")
        out = []
        for which in ("aud2q", "aud2l"):
            rel = f"{which}/{c}.json"
            seg = _z2_seg(R, rel)
            out.append(fmt(val(_aud_rec(R, rel, seg, "next", *path), n_key="n_users")))
        return " / ".join(out)
    return h


# ---- tab:teaches (editor pass 2026-10-05; pass 2): E-F, E-H, the FT-C reading (ML-1M only, addendum 8) and the FT-Q readings of the
# addendum-6 analysis (alias ext) and the Qwen knockout rows of the former tab:popularity (alias ko); columns as tab:tracks (ML-1M
# has no store field: no knockout cell).
TEACH_ROWS = {
    r"$\mathcal G_{\rm LLM|CF}=\Delta$UAUC(M4$-$M3)": ext_stat("qwen", "ext:E_F.G_LLM_given_CF", "E_F", ("G_LLM_given_CF",)),
    r"$\mathcal G_{\rm CF|LLM}=\Delta$UAUC(M4$-$M2)": ext_stat("qwen", "ext:E_F.G_CF_given_LLM", "E_F", ("G_CF_given_LLM",)),
    **{label: ext_stat("qwen", f"ext:E_H.{st}.G_prior", "E_H", ("strata", st, "G_prior"))
       for label, st in ((r"Sparse rows ($m$ from $<5$ ratings)", "sparse"), (r"Dense rows", "dense"),
                         (r"Unseen items (no TRAIN example)", "unseen"), (r"Seen items", "seen"))},
    r"FT-C (permuted): $R$ / reading": ext_ftc,
    r"FT-Q (teacher): $R_Q$": ext_ftq("R"),
    r"reading": ext_ftq("label"),
    r"UAUC of $q_0$ / $q_1$ / $m$": ext_ftq("UAUC"),
    r"Head $-$ tail drop of $\ell$": ko_handler(r"Head $-$ tail drop of $\ell$"),
    r"Placebo drop": ko_handler(r"Placebo drop (real-brand swap)"),
    r"Tail $\Delta$UAUC": ko_handler(r"Tail $\Delta$UAUC"),
    r"Registered label": ko_handler(r"Registered label (per seed)"),
}


# ---- tab:deviations (editor pass 2026-10-05): the deviations record condensed; row 7 carries three estimates
def deviation_row7(R, c, k, s):
    if c != "what":
        raise SpecError(f"no slot expected in column {c} of deviation row 7")
    if k == 0:
        expect(s, "sel:UAUC of v_star")                        # the DEV UAUC of V* on ML-1M (G5)
        return _sel_uauc(R, "v_star", "ml1m")
    if k == 1:
        expect(s, "gate:UAUC")                                 # the CONFIRM UAUC (G6), estimate only
        return num(R.get(GATE, "UAUC"))
    if k == 2:
        expect(s, "grid:UAUC")                                 # the ML-1M zero-shot UAUC on TEST rows (E-A), estimate only
        v = regime_val(R, "qwen", "ml1m", "ZS", "E_A", ("UAUC_TEST",))
        v.est_only = True
        return fmt(v)
    raise SpecError(f"no fourth slot expected in deviation row 7 ({k})")


# ---- registry
@dataclass
class TableSpec:
    label: str
    ncols: int
    colkey: object                      # (start, span) -> column key (None: no slot expected there)
    rows: dict
    caption: object = None              # (R, slot_text) -> str
    free: bool = False                  # every slot is free text (tab:guide)


def _cols(mapping: dict, spans: dict | None = None):
    def f(start, span):
        if span == 1:
            return mapping.get(start)
        return (spans or {}).get((start, span))
    return f


def _rel_cols(domains):
    m, sp = {}, {}
    for j, d in enumerate(domains):
        m[1 + 2 * j], m[2 + 2 * j] = (d, "ZS"), (d, "FT")
        sp[(1 + 2 * j, 2)] = (d, "span")
    return _cols(m, sp)


def build_table_specs() -> dict:
    specs = {}
    # editor pass 2026-10-05: the gate table moved to the protocol, its "Rule" column folded into the row labels; the zero-shot
    # UAUC, the paired gain and the references on the Gate-FT rows are printed once, in tab:tracks
    specs["tab:gate-outcomes"] = TableSpec(
        "tab:gate-outcomes", 4, _cols({1: "c1", 2: "c2", 3: "c4"}),
        {r"G5 DEV: ML-1M UAUC (bound $>0$)": go_g5_ml1m, r"G5 DEV: Toys UAUC (E2)": go_g5_toys,
         r"G6 CONFIRM: ML-1M UAUC ($\ge0.60$)": go_g6, r"G9 Gate-FT: seeds 0 / 1 / 2": go_seeds,
         r"G9 Gate-FT: mean ($\ge0.65$)": go_mean},
        caption=go_caption)
    specs["tab:anatomy"] = TableSpec(
        "tab:anatomy", 6, _cols({1: "S1k", 2: "S10k", 3: "Toys", 4: "Home", 5: "Tools"}),
        {lab: anatomy_handler(lab) for lab in ANATOMY_ROWS})
    # tab:tracks is the former tab:reliability (editor pass 2026-10-05: same-row references, E-J, shares, G and G_wu, oracle and
    # deployable top-k); tab:app-llama follows it for the second backbone
    specs["tab:tracks"] = TableSpec("tab:tracks", 9, _rel_cols(RATED),
                                    rel_rows("qwen", r"P1: $\mathcal G_{\rm FT}-\mathcal G_{\rm ZS}$"))
    llama_rows = rel_rows("llama", r"P1: $\mathcal G_{\rm FT}-\mathcal G_{\rm ZS}$")
    specs["tab:app-llama"] = TableSpec("tab:app-llama", 5, _rel_cols(("ml1m", "toys")), llama_rows)
    specs["tab:teaches"] = TableSpec("tab:teaches", 9, _rel_cols(RATED), dict(TEACH_ROWS))
    specs["tab:exposure"] = TableSpec(
        "tab:exposure", 18, exposure_col,
        {**{lab: exposure_handler(lab) for lab in [r"Pool (level)", r"Target head share (level)", r"LLM, \textsf{next}",
                                                   r"LLM, \textsf{like}", *REF_METHODS]},
         **{lab: exposure_summary_handler(lab) for lab in EXPOSURE_SUMMARY}})
    specs["tab:deviations"] = TableSpec("tab:deviations", 3, _cols({1: "what", 2: "effect"}), {"7": deviation_row7})
    specs["tab:pruning"] = TableSpec(
        "tab:pruning", 5, _cols({1: "UAUC", 2: "dP1", 3: "npos", 4: "dP0"}, {(1, 4): "ref"}),
        {**{lab: prune_handler(lab) for lab in PRUNE_ROWS}, r"Item mean, popularity, MF (ref.)": prune_refs})
    specs["tab:app-sens"] = TableSpec(
        "tab:app-sens", 7, _cols(SENS_COLS),
        {**{v: sens_handler(v) for v in ("V0", "V1", "V2", "V3", "V4", "V5", "V7")}, r"Item mean (ref.)": sens_ref})
    specs["tab:corrections"] = TableSpec(
        "tab:corrections", 5, _cols({2: "ml1m", 3: "second", 4: "sports"}),
        {str(i): corr_row(str(i)) for i in range(8)}, caption=corr_caption)
    specs["tab:slot"] = TableSpec(
        "tab:slot", 5, _cols({1: "ml1m", 2: "toys", 3: "games", 4: "sports"}),
        {lab: slot_handler(lab) for lab in [r"SFT, $b=0$ (LoRA of the main grid)~\citep{bao2023tallrec}",
                                            r"Post-hoc stacking", r"Prior-offset LoRA", r"Offset $-$ stacking",
                                            r"Item mean / MF (ref.)"]})
    specs["tab:app-seeds"] = TableSpec(
        "tab:app-seeds", 4, _cols({1: "UAUC", 2: "all", 3: "ref"}),
        {lab: seeds_handler(lab) for lab in [*SEEDS_DOMAIN_ROWS, r"ML-1M, permuted labels (seeds 0 / 1)",
                                             r"ML-1M, real $-$ permuted (seeds 0 / 1)"]})
    specs["tab:app-z2"] = TableSpec(
        "tab:app-z2", 5, _cols({1: "sports", 2: "toys", 3: "home", 4: "tools"}),
        {lab: z2_handler(lab) for lab in Z2_ROWS})
    specs["tab:guide"] = TableSpec("tab:guide", 4, _cols({}), {}, free=True)
    return specs


TABLE_SPECS = build_table_specs()


# ================================================================================================ PROSE_SPECS
LIST_SEP = "; "      # inside a per-domain list, so that two list slots in one parenthesis stay apart


def _aud_list(R, path, q="next") -> str:
    parts = []
    for lab, d, seg in FAMILY_UNITS:
        parts.append(f"{lab} {fmt(val(_aud_rec(R, f'aud/{d}.json', seg, q, *path), n_key='n_users'), prose=True)}")
    return LIST_SEP.join(parts)


def _z2_list(R, which, path) -> str:
    parts = []
    for d in NEXT:
        rel = f"{which}/{d}.json"
        parts.append(f"{RATED_NAME.get(d, d.capitalize())} "
                     f"{fmt(val(_aud_rec(R, rel, _z2_seg(R, rel), 'next', *path), n_key='n_users'), prose=True)}")
    return LIST_SEP.join(parts)


def _grid_list(R, fn) -> str:
    return LIST_SEP.join(f"{RATED_NAME[d]} {fn(d)}" for d in RATED)


def p_gft_mean(R, s):
    v = _ci_val(R.get(GFT, "UAUC_post_T_mean_over_seeds"), R.get(GFT, "UAUC_post_T_seed_averaged_ci95"))
    return fmt(v, prose=True)


def p_gft_mean_est(R, s):
    """Gate-FT mean post-T_d UAUC, estimate only (abstract, introduction, conclusion): after a recorded GATE_FT_PASS."""
    require_ft(R)
    return num(R.get(GFT, "UAUC_post_T_mean_over_seeds"))


def p_gft_seeds(R, s):
    require_ft(R)
    per = R.get(GFT, "UAUC_post_T_per_seed")
    if not isinstance(per, list) or len(per) != 3:
        raise Missing("incomplete_regime", f"{GFT}: UAUC_post_T_per_seed has {per!r} (seeds 0, 1, 2 registered)")
    return ", ".join(num(x) for x in per)


def p_gate_est(R, s):
    """The G6 confirmation-user UAUC of V*, estimate only: after a recorded GATE_PASS."""
    require_gatepass(R)
    return num(R.get(GATE, "UAUC"))


def p_gate_ci(R, s):
    """0.603$\\pm$.009 (the table convention: estimate and half-width of the 95% interval)."""
    require_gatepass(R)
    return fmt(_ci_val(R.get(GATE, "UAUC"), R.get(GATE, "ci95")), prose=True)


def p_gate_range(R, s):
    """0.603 (95\\% CI 0.594--0.612): the same interval written out for the introduction."""
    require_gatepass(R)
    ci = R.get(GATE, "ci95")
    lo, hi = num(ci.get("lo"), src=f"{GATE}:ci95.lo"), num(ci.get("hi"), src=f"{GATE}:ci95.hi")
    return f"{num(R.get(GATE, 'UAUC'))} (95\\% CI {lo}--{hi})"


def p_gate_v0(R, s):
    require_gatepass(R)
    return num(R.get(GATE, "v0_context", "UAUC"))


def p_gate_n(R, s):
    require_gatepass(R)
    return integer(R.get(GATE, "n_users"))


def p_aud(path):
    return lambda R, s: _aud_list(R, path)


def p_grid_ec(reg, key):
    def h(R, s):
        return _grid_list(R, lambda d: fmt(regime_val(R, "qwen", d, reg, "E_C", (), key), prose=True))
    return h


def p_grid_uauc(reg):
    """Per rated panel: the UAUC of l (E-A) of a regime, with its interval (no paired test is registered: no direction)."""
    def h(R, s):
        return _grid_list(R, lambda d: fmt(regime_val(R, "qwen", d, reg, "E_A", ("UAUC_TEST",)), prose=True))
    return h


def p_grid_ref(ref):
    """Per rated panel: a non-LLM reference (alias cpu) on the zero-shot regime's rows."""
    def h(R, s):
        return _grid_list(R, lambda d: fmt(ref_val(R, "qwen", d, ref), prose=True))
    return h


def p_nonprior(reg):
    def h(R, s):
        def one(d):
            rel = grid_rel("qwen", d)
            v = regime_val(R, "qwen", d, reg, "E_D", ("shares",), "non_prior_share")
            if v.text is not None:
                return v.text
            reading = (R.get(rel, "E_D", reg, "shares", "per_model", "zeroshot", "shares_reading") if reg == "ZS"
                       else R.get(rel, "E_D", reg, "shares", "mean_over_seeds", "shares_reading"))
            return "uninterpretable" if reading == "uninterpretable" else fmt(v, prose=True)
        return _grid_list(R, one)
    return h


def p_gcf(R, s):
    def one(d):
        rel = grid_rel("qwen", d)
        regime_ok(R, rel, "E_D", "ZS")
        return fmt(val(R.get(rel, "E_D", "ZS", "information_gain", "G_CF")), prose=True)
    return _grid_list(R, one)


def p_eb(R, s):
    require_ft(R)

    def one(d):
        rel = grid_rel("qwen", d)
        eb = R.get(rel, "E_B")
        if eb.get("complete") is False:
            raise Missing("incomplete_regime", f"{rel}: E_B incomplete")
        return fmt(val(R.get(rel, "E_B", "mean_over_seeds")), prose=True)
    return _grid_list(R, one)


def p_p1(R, s):
    return _p1_cell(R, "qwen", "ml1m", prose=True)


def p_prn(kind):
    def h(R, s):
        require_ft(R)
        c, _ = _prn_contrast(R, "P2", "P1")
        if c is None:
            raise Missing("field_not_produced", f"{PRN}: the confirmatory contrast P2-P1 is NOT_RUN")
        if kind == "est":
            return num(c.get("est"))
        if kind == "ci":
            return f"[{num(c.get('lo'))}, {num(c.get('hi'))}]"
        if kind == "p":
            if c.get("descriptive_min_n"):
                raise Missing("below_min_n", f"{PRN}: P2-P1 is descriptive ({c.get('n_users')} users)")
            return pval(c.get("p"))
        per = list((c.get("per_seed") or {}).values())
        if len(per) != 5:
            raise Missing("incomplete_regime", f"{PRN}: {len(per)} paired seed differences (5 registered)")
        return str(sum(1 for x in per if _fin(x) and x > 0))
    return h


def p_corr_max(R, s):
    vals = []
    for d in RATED:
        rel = grid_rel("qwen", d)
        regime_ok(R, rel, "E_A", "ZS")
        vals.append(R.get(rel, "E_A", "ZS", "invariance", "per_model", "zeroshot", "max_abs_dAUC_user"))
    if not all(_fin(x) for x in vals):
        raise Missing("field_null", "a panel's max_abs_dAUC_user is null")
    return num(max(vals))


def p_mir(key):
    def h(R, s):
        require_gatepass(R)
        sec = second_rated(R)
        return LIST_SEP.join(f"{RATED_NAME[d]} {fmt(val(R.get(MIR, 'criteria', name, key), n_key='n'), prose=True)}"
                             for d, name in (("ml1m", "ml1m"), (sec, "toys")))
    return h


def p_mir_ndcg(R, s):
    require_gatepass(R)
    return fmt(val(R.get(MIR, "next_item_no_loss", "dNDCG@10_mirror_minus_raw", "tie_exact"), n_key="n"), prose=True)


def p_slot_list(R, s):
    require_ft(R)
    decided = R.get(SLOT, "decided")
    if not decided:
        raise Missing("not_decided", f"{SLOT}: no dataset decided yet")
    return LIST_SEP.join(
        f"{RATED_NAME[d]} {fmt(val(R.get(f'slot/{d}.json', 'slot', 'difference', 'mean_over_seeds')), prose=True)}"
        for d in decided)


# ---- direction words (registered tests only)
SUMMARY = "aud/summary.json"


def _family_holm(R, *path) -> dict:
    if R.get(SUMMARY, "complete_family") is not True:
        raise Missing("not_decided", f"{SUMMARY}: the family of four domains is incomplete")
    h = R.get(SUMMARY, *path, "holm")
    if h.get("m") != 4:
        raise Missing("not_decided", f"{SUMMARY}: {'.'.join(path)} has m = {h.get('m')} family members (4 registered)")
    return h


def _holm_direction(h) -> tuple:
    """(k, sign) from a summary Holm block: k domains reject (Holm p < 0.05) all with one sign; (4, 0) when none
    rejects; mixed signs are not decided."""
    pos = sum(1 for u, r in h["reject_holm"].items() if r and (h["sign"].get(u) or 0) > 0)
    neg = sum(1 for u, r in h["reject_holm"].items() if r and (h["sign"].get(u) or 0) < 0)
    if pos and neg:
        raise Missing("not_decided", f"Holm rejections with both signs ({pos} positive, {neg} negative)")
    if pos:
        return pos, 1
    if neg:
        return neg, -1
    return 4, 0


S1_PATH = ("S1", "acc_top_minus_bottom_tertile", "next")
S2_PATH = ("S2", "share_errors_in_top_minus_third", "next")
WORD3 = {1: "above", 0: "indistinguishable from", -1: "below"}


def c_rq1_k(R, s):
    k, _ = _holm_direction(_family_holm(R, *S1_PATH))
    return f"{k} of 4"


def c_rq1_dir(R, s):
    _, sign = _holm_direction(_family_holm(R, *S1_PATH))
    return WORD3[sign]


def c_rq1_cons(R, s):
    k, sign = _holm_direction(_family_holm(R, *S1_PATH))
    if sign and k < 4:
        raise Missing("not_decided", f"the direction holds on {k} of 4 domains only")
    return {1: "concentrated in the high tertile", 0: "spread evenly", -1: "concentrated in the low tertile"}[sign]


def c_rq2(R, s):
    k, sign = _holm_direction(_family_holm(R, *S2_PATH))
    if sign and k < 4:
        raise Missing("not_decided", f"the direction holds on {k} of 4 domains only")
    return {1: "more often than", 0: "as often as", -1: "less often than"}[sign]


def _s3(R) -> tuple:
    adm = R.get(SUMMARY, "S3", "admission", "llm_next")
    if adm.get("complete_family") is not True:
        raise Missing("not_decided", f"{SUMMARY}: S3 admission of llm_next has an incomplete family")
    if adm.get("effect_claimed"):
        sign = adm.get("sign")
        return (adm["n_domains_ci_excludes_0_positive"] if sign > 0 else adm["n_domains_ci_excludes_0_negative"]), sign
    if adm.get("n_domains_ci_excludes_0_positive") == 0 and adm.get("n_domains_ci_excludes_0_negative") == 0:
        return 4, 0
    raise Missing("not_decided", "S3 not admitted (needs one sign in at least 3 of 4 domains) but some domains exclude "
                                 "0: a partial outcome needs a written sentence")


def c_rq3_dir(R, s):
    return WORD3[_s3(R)[1]]


def c_rq3_k(R, s):
    return f"{_s3(R)[0]} of 4"


def c_rq3_backbone(R, s):
    same = opp = 0
    for d in NEXT:
        vals = []
        for which in ("aud2q", "aud2l"):
            rel = f"{which}/{d}.json"
            vals.append(_aud_rec(R, rel, _z2_seg(R, rel), "next", "B_exposure", "llm", "delta_head")["est"])
        sq, sl = (0 if v == 0 else (1 if v > 0 else -1) for v in vals)
        same += sq != 0 and sq == sl
        opp += sq != 0 and sq == -sl
    if same >= 3:
        return "the same"
    if opp >= 3:
        return "the opposite"
    raise Missing("not_decided", f"the Llama and Qwen Z2 signs of delta_head agree in {same} and differ in {opp} of 4 "
                                 "domains (the registered clause needs 3 of 4)")


def c_rq3_baselines(R, s):
    pos = neg = tot = 0
    for m in sorted(REF_METHODS.values()):
        h = _family_holm(R, "S3", "endpoints", f"llm_next_vs_{m}", "head_share_top10_llm_minus_ref")
        for u, r in h["reject_holm"].items():
            tot += 1
            pos += bool(r and (h["sign"].get(u) or 0) > 0)
            neg += bool(r and (h["sign"].get(u) or 0) < 0)
    if tot == pos:
        return "larger"
    if tot == neg:
        return "smaller"
    if pos == neg == 0:
        return "similar"
    raise Missing("not_decided", f"of {tot} (baseline, domain) contrasts {pos} are Holm-positive and {neg} "
                                 "Holm-negative: no single word")


def holm_adjust(pvals: dict) -> dict:
    """Holm step-down over the members with a p-value (ftgrid_report.holm): others are outside the family (None)."""
    items = sorted((p, k) for k, p in pvals.items() if _fin(p))
    out, run, m = {k: None for k in pvals}, 0.0, len(items)
    for j, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - j) * p))
        out[k] = run
    return out


def eb_family(R) -> dict:
    """E-B (A3 section 11): Holm over the four Qwen domains of the seed-averaged dUAUC(FT - ZS); confirmed = Holm p <
    0.05 and the sigma_seed rule (addendum 1 item 8: the cross-domain step, computed here from the per-domain raw p)."""
    require_ft(R)
    p, rule, est = {}, {}, {}
    for d in RATED:                                   # completeness of the family first, then the minimum n
        rel = grid_rel("qwen", d)
        if R.get(rel, "E_B").get("complete") is False:
            raise Missing("incomplete_regime", f"{rel}: E_B incomplete (a seed missing or excluded)")
    for d in RATED:
        rel = grid_rel("qwen", d)
        mean = R.get(rel, "E_B", "mean_over_seeds")
        if mean.get("descriptive_min_n"):
            raise Missing("below_min_n", f"{rel}: E_B on {mean.get('n_users')} users (< {MIN_N}): no interval-based "
                                         "claim, so no family-wide direction word")
        p[d] = mean.get("p")
        rule[d] = R.get(rel, "E_B", "seeds").get("sigma_seed_rule")
        est[d] = R.get(rel, "E_B", "mean_over_seeds", "est")
    adj = holm_adjust(p)
    return {d: {"p": p[d], "p_holm": adj[d], "sigma_seed_rule": rule[d], "est": est[d],
                "confirmed": bool(adj[d] is not None and adj[d] < 0.05 and rule[d] is True)} for d in RATED}


def c_rq4_ft(R, s):
    fam = eb_family(R)
    conf = [v for v in fam.values() if v["confirmed"]]
    if len(conf) == 4 and all(v["est"] > 0 for v in conf):
        return "raises"
    if len(conf) == 4 and all(v["est"] < 0 for v in conf):
        return "lowers"
    if not conf:
        return "leaves"
    raise Missing("not_decided", f"E-B confirmed in {len(conf)} of 4 Qwen domains")


def c_rq4_g(R, s):
    require_ft(R)
    units = []
    for d in RATED:
        rel = grid_rel("qwen", d)
        for reg in ("ZS", "FT"):
            regime_ok(R, rel, "E_D", reg)
            fam = R.get(rel, "E_D", reg, "holm_family_E_D")
            g = (R.get(rel, "E_D", reg, "information_gain", "per_model", "zeroshot", "G") if reg == "ZS"
                 else R.get(rel, "E_D", reg, "information_gain", "G_mean_over_seeds"))
            if g.get("descriptive_min_n"):
                raise Missing("below_min_n", f"{rel}: G ({reg}) on {g.get('n_users')} users (< {MIN_N})")
            units.append((fam["confirmed"].get("G") is True, g["est"]))
    if all(c and g > 0 for c, g in units):
        return "above"
    if not any(c for c, _ in units):
        return "not distinguishable from"
    raise Missing("not_decided", "G is confirmed (E-D Holm family) in some but not all domains and regimes")


def c_p1_holds(R, s):
    require_ft(R)
    v = R.get(grid_rel("qwen", "ml1m"), "P1", "decision", "verdict")
    if v == "P1_HOLDS":
        return "holds"
    if v == "NO_EVIDENCE":
        return "does not hold"
    raise Missing("incomplete_regime", f"P1 verdict {v}")


def c_p1_n(R, s):
    require_ft(R)
    rel = grid_rel("qwen", "ml1m")
    if R.get(rel, "P1", "decision", "verdict") == "INCOMPLETE":
        raise Missing("incomplete_regime", "P1 is INCOMPLETE")
    per = [R.get(rel, "P1", "per_seed", m, "est") for m in FT_MODELS]
    return f"{sum(1 for x in per if _fin(x) and x > 0)} of 3"


def c_ko_label(R, s):
    require_ft(R)
    labels = []
    for d in ("toys", "games"):
        for m in ("zeroshot", *FT_MODELS):
            labels.append(R.get(grid_rel("qwen", d), "knockout", "labels", m, "label"))
    if len(set(labels)) == 1 and labels[0] in ("POSITIVE", "NEGATIVE", "NULL", "INDETERMINATE"):
        return labels[0]
    raise Missing("not_decided", f"the Qwen knockout labels differ across Toys / Video Games and models: {labels}")


PRN_WORD = {"BEATS_RANDOM": "better than", "WORSE_THAN_RANDOM": "worse than", "ABOUT_EQUAL": "about equal to",
            "INCONCLUSIVE": "inconclusive against"}


def c_prn(R, s):
    require_ft(R)
    lab = R.get(PRN, "claim", "label")
    if lab in PRN_WORD:
        return PRN_WORD[lab]
    if lab == "DESCRIPTIVE_MIN_N":
        raise Missing("below_min_n", f"{PRN}: label DESCRIPTIVE_MIN_N")
    raise Missing("incomplete_regime", f"{PRN}: label {lab}")


def _slot_dir(R):
    require_ft(R)
    decided = R.get(SLOT, "decided")
    if not decided:
        raise Missing("not_decided", f"{SLOT}: no dataset decided yet")
    conf = R.get(SLOT, "holm", "confirmed")
    pos = sum(1 for d in decided if conf.get(d) and R.get(SLOT, "datasets", d, "dUAUC") > 0)
    neg = sum(1 for d in decided if conf.get(d) and R.get(SLOT, "datasets", d, "dUAUC") < 0)
    if pos and neg:
        raise Missing("not_decided", "the slot family has confirmed gains of both signs")
    if pos:
        return pos, 1
    if neg:
        return neg, -1
    return len(decided), 0


def c_slot_dir(R, s):
    return WORD3[_slot_dir(R)[1]]


def c_slot_k(R, s):
    return f"{_slot_dir(R)[0]} of 4"


def unfillable(reason, detail):
    def h(R, s):
        raise Missing(reason, detail)
    return h


# ---- count rule (addendum 6 item 9.2; editor pass 2026-10-05): a registered family with mixed outcomes is reported as counts
# per regime, never with one direction word. These functions only count confirmed members (the existing decision functions
# above are unchanged and still decide the unanimous words).
def c_eb_count(R, s):
    """E-B: on how many of the four Qwen domains the regime contrast is confirmed (eb_family: Holm over the four domains and the
    sigma_seed rule); confirmed members of both signs are not decided (a written sentence)."""
    fam = eb_family(R)
    conf = [v for v in fam.values() if v["confirmed"]]
    if any(v["est"] > 0 for v in conf) and any(v["est"] < 0 for v in conf):
        raise Missing("not_decided", "E-B is confirmed with both signs: a mixed outcome needs a written sentence")
    return f"{len(conf)} of 4"


def c_g_count(reg):
    """E-D: on how many of the four Qwen domains G is confirmed in regime reg (the report's E-D Holm family; for FT also the
    sigma_seed rule, as the report's 'confirmed' flag encodes); confirmed members of both signs are not decided, and a member on
    fewer than 150 users (descriptive) stops the count, as in c_rq4_g."""
    def h(R, s):
        if reg == "FT":
            require_ft(R)
        pos = neg = 0
        for d in RATED:
            rel = grid_rel("qwen", d)
            regime_ok(R, rel, "E_D", reg)
            fam = R.get(rel, "E_D", reg, "holm_family_E_D")
            g = (R.get(rel, "E_D", reg, "information_gain", "per_model", "zeroshot", "G") if reg == "ZS"
                 else R.get(rel, "E_D", reg, "information_gain", "G_mean_over_seeds"))
            if g.get("descriptive_min_n"):
                raise Missing("below_min_n", f"{rel}: G ({reg}) on {g.get('n_users')} users (< {MIN_N})")
            if fam["confirmed"].get("G") is True:
                pos += g["est"] > 0
                neg += g["est"] < 0
        if pos and neg:
            raise Missing("not_decided", f"G is confirmed with both signs in the {reg} regime")
        return f"{pos + neg} of 4"
    return h


def p_mir_decision(R, s):
    """The registered stage-3 label of the two-view contrast (pilot1_gate --stage3_gate), verbatim; after GATE_PASS only."""
    require_gatepass(R)
    return tex(R.get(MIR, "decision"))


# ---- alias ext, prose (editor pass 2, 2026-10-05). The A3-6 families are summarize's (extra/summary.json): their members,
# confirmations (R14: Holm p < 0.05, the sigma_seed rule for FT members, the hypothesised sign for H-F and H-S) and the outside
# members are printed as counts (the count rule of A3-6 item 9.2), never as one word for a mixed outcome; E-J is two-sided and the
# sign of a confirmed member is reported as found. Nothing here confirms anything that summarize did not confirm.
AMAZON = ("toys", "games", "sports")


def _names(ds) -> str:
    names = [RATED_NAME[d] for d in ds]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _ext_family(R, fam: str) -> tuple:
    """(members, outside) of an A3-6 family of the summary (ftgrid_extra.holm_family)."""
    f = R.get(EXT_SUM, "families", fam)
    return f.get("members") or {}, f.get("outside_family") or {}


def _not_run_note(R) -> str:
    """The panels whose fine-tuned runs are a registered cut recorded in the summary (not members of the FT families)."""
    cut = [d for d in AMAZON if d in (R.get(EXT_SUM, "not_run") or {})]
    return f" ({_names(cut)}: fine-tuning not run)" if cut else ""


def c_ext_ef(R, s):
    """H-F (A3-6 item 3): the E-F members confirmed by summarize out of the family's members (Qwen Amazon panels, FT)."""
    require_ft(R)
    mem, _ = _ext_family(R, "E_F")
    k = sum(1 for v in mem.values() if v.get("confirmed") is True)
    return f"{k} of {len(mem)} Qwen Amazon panels" + _not_run_note(R)


def c_ext_eh(reg: str):
    """H-S (A3-6 item 5): the E-H members of a regime (panel runs with at least 150 users on sparse rows) confirmed by summarize,
    out of the regime's members; the panel runs outside the family by the minimum-n rule are named."""
    word = {"ZS": "zero-shot", "FT": "fine-tuned"}[reg]

    def h(R, s):
        if reg == "FT":
            require_ft(R)
        mem, out = _ext_family(R, "E_H")
        mem = {k: v for k, v in mem.items() if k.endswith(":" + reg)}
        out = [k.split(":")[0] for k in out if k.endswith(":" + reg)]
        below = f" ({_names([d for d in AMAZON if d in out])} below 150 users)" if out else ""
        if not mem:
            if not out:
                raise Missing("not_decided", f"{EXT_SUM}: the E-H family has no {word} panel run")
            return f"none of the {word} panel runs{below}"
        k = sum(1 for v in mem.values() if v.get("confirmed") is True)
        return f"{k} of {len(mem)} {word} panel runs{below}"
    return h


def c_ext_ej(R, s):
    """H-J (A3-6 item 6, two-sided): the E-J members confirmed by summarize out of the family's members, with the sign of each
    confirmed member as found (below or above the item mean)."""
    require_ft(R)
    mem, _ = _ext_family(R, "E_J")
    conf = [d for d in AMAZON if (mem.get(d) or {}).get("confirmed") is True]
    if any(d not in AMAZON for d in mem):
        raise Missing("field_not_produced", f"{EXT_SUM}: E-J members {sorted(mem)} (the Qwen Amazon panels)")
    below = [d for d in conf if (mem[d].get("sign") or 0) < 0]
    above = [d for d in conf if (mem[d].get("sign") or 0) > 0]
    if len(below) + len(above) != len(conf):
        raise Missing("field_null", f"{EXT_SUM}: a confirmed E-J member has no sign")
    parts = ([f"below it on {_names(below)}"] if below else []) + ([f"above it on {_names(above)}"] if above else [])
    return (f"{len(conf)} of {len(mem)} Qwen Amazon panels" + (f" ({'; '.join(parts)})" if parts else "")
            + _not_run_note(R))


ROBUST_WORDS = {"robust": "robust", "estimator_dependent": "estimator-dependent",
                "seed_sign_disagreement": "seed-sign dependent", "both_intervals_include_0": "both intervals include 0",
                "descriptive_min_n": "descriptive (fewer than 150 users)"}


def c_ext_robust(R, s):
    """A3-6 item 2 reading rule (R6, R17) on the Qwen Amazon panels (registered, outcome-free), per regime: how many panels read
    'robust' among those with a reading; the estimator-dependent ones are named (the paper calls them so)."""
    require_ft(R)
    rr = R.get(EXT_SUM, "robust_readings")
    n, k, dep = {"ZS": 0, "FT": 0}, {"ZS": 0, "FT": 0}, []
    for reg in ("ZS", "FT"):
        for d in AMAZON:
            reading = ((rr.get(f"{d}:{reg}") or {}).get("G") or {}).get("reading")
            if reading is None or reading == "not_available":
                continue
            if reading not in ROBUST_WORDS:
                raise Missing("field_not_produced", f"{EXT_SUM}: robust_readings.{d}:{reg}.G.reading = {reading!r}")
            n[reg] += 1
            k[reg] += reading == "robust"
            if reading == "estimator_dependent":
                dep.append(f"{RATED_NAME[d]} ({'zero-shot' if reg == 'ZS' else 'fine-tuned'})")
    if not (n["ZS"] or n["FT"]):
        raise Missing("not_decided", f"{EXT_SUM}: no Qwen Amazon panel has a reading of G")
    out = (f"robust on {k['ZS']} of {n['ZS']} zero-shot and {k['FT']} of {n['FT']} fine-tuned Qwen Amazon panels")
    return out + (f"; estimator-dependent on {', '.join(dep)}" if dep else "")


def p_ext_p1_reading(R, s):
    """The A3-6 item 2 reading of P1 (ML-1M, Qwen3-8B): the registered P1 against its within-user estimate (E_W.P1_wu.reading_P1),
    labelled exploratory when the file says so (A3-6 Consequence)."""
    require_ft(R)
    rel = ext_rel("qwen", "ml1m")
    reading = R.get(rel, "E_W", "P1_wu", "reading_P1", "reading")
    if reading not in ROBUST_WORDS:
        raise Missing("field_not_produced", f"{rel}: E_W.P1_wu.reading_P1.reading = {reading!r}")
    expl = R.get(rel, "status", "items_2_to_7") == "exploratory"
    return ROBUST_WORDS[reading] + (" (exploratory)" if expl else "")


def c_ext_ft_wording(R, s):
    """Addendum 8 section 2 (ftgrid_extra R18, summarize's ft_wording): 'fine-tuning mostly teaches the item' only if FT-C reads
    ITEM_DRIVEN on ML-1M and FT-Q on every dataset run; a USER_DRIVEN or MIXED label: the evidence is mixed, with the labels per
    control and dataset; never decided on partial input."""
    require_ft(R)
    w = R.get(EXT_SUM, "ft_wording")              # not requested: available false -> result_not_in_report
    if w.get("complete") is not True:
        raise Missing("not_decided", f"{EXT_SUM}: ft_wording is incomplete ({w.get('reason')}); never decided on partial "
                                     "input (R18)")
    labs = w.get("labels") or {}
    ftc, ftq = labs.get("FT-C") or {}, labs.get("FT-Q") or {}
    if "ml1m" not in ftc or not ftq:
        raise Missing("field_not_produced", f"{EXT_SUM}: ft_wording.labels lacks FT-C ml1m or FT-Q")
    order = [d for d in ("ml1m",) + AMAZON if d in ftq]
    lab = (f"FT-C on ML-1M {tex(ftc['ml1m'])}; FT-Q on " + ", ".join(f"{RATED_NAME[d]} {tex(ftq[d])}" for d in order))
    yes, mixed = w.get("fine_tuning_mostly_teaches_the_item"), w.get("evidence_mixed")
    if yes is True:
        txt = "fine-tuning mostly teaches the item"
        if w.get("item_quality_from_item_text") is True:
            txt += ", and the adapter learns item quality from item text"
        return f"{txt} ({lab})"
    if yes is not False:
        raise Missing("field_null", f"{EXT_SUM}: ft_wording.fine_tuning_mostly_teaches_the_item is {yes!r}")
    if mixed is True:
        return f"the evidence is mixed ({lab})"
    return f"the labels do not support saying that fine-tuning mostly teaches the item ({lab})"


@dataclass
class ProseSpec:
    before: str           # the normalized text right before the slot ends with this
    handler: object


PROSE_SPECS = {
    # The gate branches that occurred (GATE_PASS, GATE_FT_PASS) were resolved in the skeleton by the main session on 2026-10-04;
    # their numbers still come from gate.json / gate_ft.json (and the handlers refuse any other decision). The introduction's
    # anchors follow its 2026-10-04 rewrite (editor pass 2026-10-05); the protocol's gate paragraph prints no number (the gate
    # table does).
    ("introduction", "gate:n_users", 0): ProseSpec("4 stars or higher; on", p_gate_n),
    ("introduction", "gate:UAUC, ci95", 0): ProseSpec("untouched ML-1M users it reached UAUC", p_gate_range),
    ("introduction", "gate:v0_context.UAUC", 0): ProseSpec("against", p_gate_v0),
    ("introduction", "gft:UAUC_post_T_mean_over_seeds", 0): ProseSpec("It did (mean", p_gft_mean_est),
    ("introduction", "gft:UAUC_post_T_per_seed", 0): ProseSpec("[S]; seeds", p_gft_seeds),
    ("conclusion", "gate:UAUC", 0): ProseSpec("the remedied zero-shot prompt reached UAUC", p_gate_est),
    ("conclusion", "gft:UAUC_post_T_mean_over_seeds", 0): ProseSpec("and LoRA tuning reached", p_gft_mean_est),
    # ---- Findings 6.1, what the confidence tracks (numbers are in tab:tracks; the prose carries the registered words and counts)
    ("experiments", "raises / leaves / lowers", 0): ProseSpec("Fine-tuning", c_rq4_ft),
    ("experiments", "grid:E_B confirmed, k of 4", 0): ProseSpec("(E-B, confirmed on", c_eb_count),
    ("experiments", "ext:summary.families.E_J count", 0): ProseSpec("the item mean (H-J) on", c_ext_ej),
    ("experiments", "grid:G confirmed, k of 4 zero-shot", 0): ProseSpec("is confirmed (E-D) on", c_g_count("ZS")),
    ("experiments", "grid:G confirmed, k of 4 LoRA", 0): ProseSpec("zero-shot and", c_g_count("FT")),
    ("experiments", "ext:summary.robust_readings count", 0): ProseSpec("the within-user estimator finds it", c_ext_robust),
    ("experiments", "holds / does not hold", 0): ProseSpec("[S]. P1", c_p1_holds),
    ("experiments", "grid:P1", 0): ProseSpec(r"(mean $\mathcal G_{\rm FT}-\mathcal G_{\rm ZS}$", p_p1),
    ("experiments", "n of 3", 0): ProseSpec(r"\mathcal G_{\rm ZS}$ [S];", c_p1_n),
    ("experiments", "ext:E_W.P1_wu.reading_P1", 0): ProseSpec("seeds positive; within-user reading", p_ext_p1_reading),
    # ---- Findings 6.2, what fine-tuning teaches
    ("experiments", "ext:summary.ft_wording (the wording it allows, with the labels per control and dataset)", 0): ProseSpec(
        "By the wording rule of addendum 8,", c_ext_ft_wording),
    ("experiments", "ext:summary.families.E_F count", 0): ProseSpec("H-F) on", c_ext_ef),
    ("experiments", "ext:summary.families.E_H count ZS", 0): ProseSpec("H-S) in", c_ext_eh("ZS")),
    ("experiments", "ext:summary.families.E_H count FT", 0): ProseSpec("[S] and", c_ext_eh("FT")),
    ("experiments", "POSITIVE / NEGATIVE / NULL / INDETERMINATE", 0): ProseSpec("the Qwen3-8B knockout is labelled",
                                                                                 c_ko_label),
    # ---- Findings 6.3, which uses survive a matched control
    ("experiments", "k of 4", 0): ProseSpec(r"unsure when wrong (S1, S2).} On", c_rq1_k),
    ("experiments", "above / indistinguishable from / below", 0): ProseSpec(r"domains $\mathrm{acc}_{\rm high}$ is",
                                                                             c_rq1_dir),
    ("experiments", "aud:C.acc_top_minus_bottom_tertile", 0): ProseSpec(
        r"$\mathrm{acc}_{\rm low}$ (difference", p_aud(("C_calibration", "error_anatomy",
                                                        "acc_top_minus_bottom_tertile"))),
    ("experiments", "concentrated in the high tertile / spread evenly / concentrated in the low tertile", 0): ProseSpec(
        "so correct top-1 decisions are", c_rq1_cons),
    ("experiments", "more often than / as often as / less often than", 0): ProseSpec(
        "top-1 errors fall in the high tertile", c_rq2),
    ("experiments", "aud:C.share_errors_in_top_tertile", 0): ProseSpec(
        "one third of the time (share", p_aud(("C_calibration", "error_anatomy", "share_errors_in_top_tertile"))),
    ("experiments", "above / indistinguishable from / below", 1): ProseSpec(r"scores $\Delta_{\rm head}$ is",
                                                                             c_rq3_dir),
    ("experiments", "k of 4", 1): ProseSpec("[S] 0 on", c_rq3_k),
    ("experiments", "aud:B.delta_head", 0): ProseSpec("0 on [S] domains (", p_aud(("B_exposure", "llm", "delta_head"))),
    ("experiments", "the same / the opposite / no", 0): ProseSpec("domains ([S]), with", c_rq3_backbone),
    ("experiments", "aud2l:B.delta_head", 0): ProseSpec(
        "sign on the second backbone (", lambda R, s: _z2_list(R, "aud2l", ("B_exposure", "llm", "delta_head"))),
    ("experiments", "larger / similar / smaller", 0): ProseSpec("on the second backbone ([S]), and", c_rq3_baselines),
    ("experiments", "corr:max_abs_dAUC_user", 0): ProseSpec("change per-user AUC by at most", p_corr_max),
    ("experiments", "mir:decision", 0): ProseSpec("two-view contrast is labelled", p_mir_decision),
    ("experiments", "prn:d_vs_P1", 0): ProseSpec("The contrast P2 $-$ P1 is", p_prn("est")),
    ("experiments", "prn:ci_P2_minus_P1", 0): ProseSpec("(interval", p_prn("ci")),
    ("experiments", "prn:p_P2_minus_P1", 0): ProseSpec("$p$", p_prn("p")),
    ("experiments", "prn:n_pos_of_5", 0): ProseSpec(") with", p_prn("npos")),
    ("experiments", "better than / worse than / about equal to / inconclusive against", 0): ProseSpec(
        "so uncertainty-selected pruning is", c_prn),
    ("experiments", "above / indistinguishable from / below", 6): ProseSpec("Prior-offset LoRA is", c_slot_dir),
    ("experiments", "k of 4", 3): ProseSpec("post-hoc stacking on", c_slot_k),
    ("experiments", "slot:d_offset_minus_stack", 0): ProseSpec("datasets (gain", p_slot_list),
}
# The occurrence number in the keys is documentation only: a prose slot is matched on (file, text) and its anchor.
PROSE_BY_TEXT: dict = {}
for (_file, _text, _occ), _spec in PROSE_SPECS.items():
    PROSE_BY_TEXT.setdefault((_file, _text), []).append(_spec)
for (_file, _text), _specs in PROSE_BY_TEXT.items():
    _anchors = [p.before for p in _specs]
    assert not any(a != b and a.endswith(b) for a in _anchors for b in _anchors) and len(set(_anchors)) == len(_anchors), \
        f"PROSE_SPECS anchors of {_text!r} are not distinct"


# ================================================================================================ driver
def masked_contexts(text: str, slots: list) -> None:
    """slot.before = the normalized text before the slot (the last 400 characters), every other slot as [S]."""
    tail, pos = "", 0
    for s in slots:
        tail = (tail + text[pos:s.start])[-400:]
        s.before = re.sub(r"\s+", " ", tail).strip()
        tail += "[S]"
        pos = s.end


def classify(fname: str, text: str, slots: list) -> None:
    tables = parse_tables(text)
    marks = marker_spans(text)
    masked_contexts(text, slots)
    occ: dict = {}
    for s in slots:
        s.markers = tuple(sorted({n for n, a, b in marks if a <= s.start < b}))
        n = norm(s.inner)
        if n.startswith("branch:"):
            s.kind = "branch"
            continue
        t = next((t for t in tables if t.start <= s.start < t.end), None)
        if t is not None:
            s.table = t.label
            if t.tab_start <= s.start < t.tab_end:
                s.kind = "table"
                for label, cells in t.rows:
                    hit = [(j, c) for j, c in enumerate(cells) if c.start <= s.start < c.end]
                    if hit:
                        j, c = hit[0]
                        s.row, s.col, s.span = label, c.col, c.span
                        s.ncols = sum(x.span for x in cells)
                        break
            else:
                s.kind = "caption"
            continue
        s.kind = "prose"
        s.occ = occ.get((fname, n), 0)          # reported only; prose specs match on the text before the slot
        occ[(fname, n)] = s.occ + 1
    # the slot index inside its cell
    by_cell: dict = {}
    for s in slots:
        if s.kind == "table":
            key = (s.table, s.row, s.col)
            s.k = by_cell.get(key, 0)
            by_cell[key] = s.k + 1


def prose_spec(s: Slot):
    """The PROSE_SPECS entry of a prose slot: same file and slot text, and the text before the slot ends with the entry's
    anchor. Matching on the anchor (not on the occurrence index) keeps every other slot valid when the skeleton drops or
    adds a slot with the same text elsewhere in the file."""
    n = norm(s.inner)
    cands = PROSE_BY_TEXT.get((s.file, n), [])
    hits = [p for p in cands if s.before.endswith(p.before)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise SpecError(f"several PROSE_SPECS anchors match {n!r}: {[p.before for p in hits]}")
    if cands:
        raise SpecError(f"the text before the slot ...{s.before[-60:]!r} ends with none of the anchors "
                        f"{[p.before for p in cands]} of {n!r}")
    if s.file == "experiments":
        raise SpecError(f"no PROSE_SPECS entry for {n!r}")
    raise Missing("free_text", "a sentence to be written from the filled tables")


def fill_slot(R: Results, s: Slot) -> None:
    global _ACTIVE
    R.trace, R.seen, R.printed = [], {}, []
    _ACTIVE = R
    try:
        if s.kind == "branch":
            raise Missing("branch_slot", "kept verbatim")
        if s.kind in ("table", "caption"):
            spec = TABLE_SPECS.get(s.table)
            if spec is None:
                raise SpecError(f"no TABLE_SPECS entry for table {s.table}")
            if spec.free:
                raise Missing("free_text", f"{s.table}: interpretive clause or claim-admission verdict")
            if s.kind == "caption":
                if spec.caption is None:
                    raise SpecError(f"no caption handler for {s.table}")
                s.fill = spec.caption(R, s.inner)
            else:
                if s.ncols != spec.ncols:
                    raise SpecError(f"row {s.row!r} has {s.ncols} columns, TABLE_SPECS expects {spec.ncols}")
                handler = spec.rows.get(s.row)
                if handler is None:
                    raise SpecError(f"no TABLE_SPECS row {s.row!r} in {s.table}")
                colkey = spec.colkey(s.col, s.span)
                if colkey is None:
                    raise SpecError(f"no column key for column {s.col} (span {s.span}) in {s.table}")
                s.fill = handler(R, colkey, s.k, s.inner)
        else:
            spec = prose_spec(s)
            s.anchor = spec.before
            s.fill = spec.handler(R, s.inner)
        s.sources = tuple(dict.fromkeys(R.trace))
        s.printed = tuple(R.printed)
    except Missing as e:
        s.fill, s.reason, s.detail = None, e.reason, e.detail
    except SpecError as e:
        s.fill, s.reason, s.detail = None, "skeleton_changed", str(e)
    finally:
        _ACTIVE = None


def decisions(R: Results) -> dict:
    out = {}
    for name, rel, path in (("sel:decision", SEL, ("decision",)), ("sel:v_star", SEL, ("v_star",)),
                            ("gate:decision", GATE, ("decision",)), ("gft:decision", GFT, ("decision",)),
                            ("P1 (grid ml1m)", grid_rel("qwen", "ml1m"), ("P1", "decision", "verdict")),
                            ("prn:claim.label", PRN, ("claim", "label")), ("slot:state", SLOT, ("state",)),
                            ("mir:decision", MIR, ("decision",))):
        try:
            out[name] = R.get(rel, *path)
        except Missing:
            out[name] = None
    return out


def checks(R: Results) -> list:
    """Cross-file consistency checks the skeleton states (reported, never used to fill)."""
    out = []

    def cmp(name, a_get, b_get, tol=None):
        """ok = the two values print identically (three decimals), or differ by at most tol (counts: tol 0)."""
        try:
            a, b = a_get(), b_get()
        except Missing:
            return
        if not (_fin(a) and _fin(b)):
            ok = False
        elif tol is not None:
            ok = abs(a - b) <= tol
        else:
            ok = _numstr(a) == _numstr(b)
        out.append({"check": name, "a": a, "b": b, "ok": bool(ok)})
    g = grid_rel("qwen", "ml1m")
    # (grid E_B vs gate_ft.json's own zero-shot context: not a check any more; both tables print the grid value and the
    # second scoring is recorded in FILLED.json notes, main-session decision 2026-10-04)
    for i, m in enumerate(FT_MODELS):
        cmp(f"grid ml1m E_A.FT UAUC {m} = gft UAUC_post_T_per_seed[{i}] (tab:app-seeds comment)",
            lambda m=m: R.get(g, "E_A", "FT", "UAUC_TEST", "per_model", m, "est"),
            lambda i=i: R.get(GFT, "UAUC_post_T_per_seed", i))
    for bb, ds in (("qwen", RATED), ("llama", ("ml1m", "toys"))):
        for d in ds:
            cmp(f"{bb} {d}: report meta n_users_both_classes_test = ftgrid_split eval.users_both_classes_test",
                lambda bb=bb, d=d: R.get(grid_rel(bb, d), "meta", "n_users_both_classes_test"),
                lambda bb=bb, d=d: R.get(split_rel(bb, d), "eval", "users_both_classes_test"), tol=0)
    return out


def notes(R: Results, slots: list) -> list:
    """Recorded facts about values that are deliberately not printed (FILLED.json "notes")."""
    out = []
    try:
        gz = R.get(GFT, "zero_shot_context", "UAUC_post_T")
        gd = R.get(GFT, "zero_shot_context", "dUAUC_finetuned_minus_zeroshot_post_T", "est")
    except Missing:
        return out
    g = grid_rel("qwen", "ml1m")
    entry = {"topic": "ML-1M zero-shot source of tab:gate-outcomes", "note": GATE_ZS_NOTE,
             "not_printed": {f"{GFT}:zero_shot_context.UAUC_post_T": gz,
                             f"{GFT}:zero_shot_context.dUAUC_finetuned_minus_zeroshot_post_T.est": gd},
             "printed_instead": None,
             "slots": [{"file": f"sections/{s.file}.tex", "line": s.line, "row": s.row, "column": s.col}
                       for s in slots if s.table == "tab:gate-outcomes" and norm(s.inner) in (
                           "gft:zero_shot_context.UAUC_post_T", "grid:UAUC", "gft:dUAUC_finetuned_minus_zeroshot_post_T",
                           "grid:dUAUC_ft_minus_zs")]}
    try:
        zz = R.get(g, "E_A", "ZS", "UAUC_TEST", "per_model", "zeroshot", "est")
        eb = R.get(g, "E_B", "mean_over_seeds", "est")
        entry["printed_instead"] = {f"{g}:E_A.ZS.UAUC_TEST.per_model.zeroshot.est": zz, f"{g}:E_B.mean_over_seeds.est": eb}
        entry["abs_difference"] = {"UAUC_zero_shot": abs(gz - zz), "dUAUC_LoRA_minus_zero_shot": abs(gd - eb)}
    except Missing as e:
        entry["printed_instead"] = f"not yet: {e.detail}"
    out.append(entry)
    return out


# ---- --check_equal: every quantity printed in more than one place, with its sources
GFT_EQUIV = {   # gate_ft.json fields that are the same registered quantity as a field of the Qwen ML-1M grid report
    "zero_shot_context.UAUC_post_T": "E_A.ZS.UAUC_TEST.per_model.zeroshot",
    "zero_shot_context.dUAUC_finetuned_minus_zeroshot_post_T": "E_B.mean_over_seeds",
    "UAUC_post_T_mean_over_seeds": "E_A.FT.UAUC_TEST.mean_over_seeds",
    "UAUC_post_T_seed_averaged_ci95": "E_A.FT.UAUC_TEST.mean_over_seeds",
    **{f"UAUC_post_T_per_seed.{i}": f"E_A.FT.UAUC_TEST.per_model.{m}" for i, m in enumerate(FT_MODELS)},
}


def canonical(src: str) -> str:
    """One key per reported quantity: the source field without '.est', the gate_ft.json fields mapped to the grid fields
    they duplicate, and the regime dropped where it only selects the rows (references, the CF reference gain)."""
    rel, _, path = src.partition(":")
    path = re.sub(r"\.est$", "", path)
    if rel == GFT and path in GFT_EQUIV:
        return f"{grid_rel('qwen', 'ml1m')}:{GFT_EQUIV[path]}"
    path = re.sub(r"^E_A\.(?:ZS|FT)\.(UAUC_TEST\.references\.|mf_personal_residual_warm_pairs_only)", r"E_A.*.\1", path)
    path = re.sub(r"^E_D\.(?:ZS|FT)\.information_gain\.G_CF", "E_D.*.information_gain.G_CF", path)
    return f"{rel}:{path}"


def check_equal(slots: list) -> dict:
    """Every quantity printed at more than one place (table cell or prose slot) with its sources and printed texts;
    estimates_equal = false is an inconsistency of the paper, printed_equal = false only a format difference."""
    groups, unattributed = {}, 0
    for s in slots:
        if s.fill is None:
            continue
        for src, est, text in s.printed:
            if not src:
                unattributed += 1
                continue
            groups.setdefault(canonical(src), []).append(
                {"file": f"sections/{s.file}.tex", "line": s.line, "table": s.table, "row": s.row, "column": s.col,
                 "source": src, "estimate": est, "printed": text})
    out = []
    for key in sorted(groups):
        occ = groups[key]
        places = {(o["file"], o["line"], o["table"], o["row"], o["column"]) for o in occ}
        if len(places) < 2:
            continue
        tables = sorted({o["table"] or f"prose:{o['file']}" for o in occ})
        out.append({"quantity": key, "n_printed": len(occ), "n_tables": len(tables), "tables": tables,
                    "estimates_equal": len({o["estimate"] for o in occ}) == 1,
                    "printed_equal": len({o["printed"] for o in occ}) == 1, "occurrences": occ})
    return {"schema": "fill_paper_check_equal_v1",
            "summary": {"quantities_printed_more_than_once": len(out),
                        "printed_in_more_than_one_table": sum(g["n_tables"] > 1 for g in out),
                        "estimates_differ": sum(not g["estimates_equal"] for g in out),
                        "format_or_interval_differs": sum(g["estimates_equal"] and not g["printed_equal"] for g in out),
                        "numbers_without_a_source": unattributed},
            "rule": "a quantity is a result field (gate_ft.json fields that duplicate grid ML-1M fields, and references "
                    "taken from either regime's identical rows, count as one quantity); estimates_equal = false is an "
                    "inconsistency to resolve, printed_equal = false a format or interval difference",
            "groups": out}


def alias_of(s: Slot) -> str:
    m = re.match(r"^\s*([a-z0-9]+)\s*:", norm(s.inner))
    return m.group(1) if m and m.group(1) in ALIASES else ("branch" if s.kind == "branch" else "free")


def run(paper: Path, results: Path, out: Path) -> dict:
    paper, results, out = Path(paper), Path(results), Path(out)
    if out.resolve() in (paper.resolve(), (paper / "sections").resolve()):
        raise SystemExit(f"--out {out} would overwrite the skeleton")
    R = Results(results)
    all_slots, texts = [], {}
    for f in SECTION_FILES:
        p = paper / "sections" / f"{f}.tex"
        text = p.read_bytes().decode("utf-8")
        texts[f] = text
        slots = find_slots(text, f)
        classify(f, text, slots)
        for s in slots:
            fill_slot(R, s)
        all_slots.extend(slots)
    # filled copies
    outputs = {}
    for f, text in texts.items():
        slots = [s for s in all_slots if s.file == f and s.fill is not None]
        for s in sorted(slots, key=lambda s: -s.start):
            text = text[:s.start] + s.fill + text[s.end:]
        outputs[f"sections/{f}.tex"] = text.encode("utf-8")
    for name in ("main.tex", "references.bib"):
        if (paper / name).is_file():
            outputs[name] = (paper / name).read_bytes()
    filled = [s for s in all_slots if s.fill is not None]
    unfilled = [s for s in all_slots if s.fill is None]
    by_reason, by_alias_f, by_alias_u = {}, {}, {}
    for s in unfilled:
        by_reason[s.reason] = by_reason.get(s.reason, 0) + 1
        by_alias_u[alias_of(s)] = by_alias_u.get(alias_of(s), 0) + 1
    for s in filled:
        by_alias_f[alias_of(s)] = by_alias_f.get(alias_of(s), 0) + 1

    def where(s):
        w = {"file": f"sections/{s.file}.tex", "line": s.line, "slot": s.inner, "kind": s.kind,
             "table": s.table, "row": s.row, "column": s.col, "blocks": list(s.markers)}
        if s.kind == "prose" and s.anchor:
            w["anchor"] = s.anchor
        return w
    summary = {"slots_total": len(all_slots), "filled": len(filled), "unfilled": len(unfilled),
               "unfilled_by_reason": dict(sorted(by_reason.items())),
               "filled_by_alias": dict(sorted(by_alias_f.items())),
               "unfilled_by_alias": dict(sorted(by_alias_u.items()))}
    ceq = check_equal(all_slots)
    unfilled_doc = {"schema": "fill_paper_unfilled_v1", "summary": summary, "reasons": REASONS,
                    "decisions": decisions(R), "checks": checks(R), "check_equal_summary": ceq["summary"],
                    "sources": dict(sorted(R.sha1.items())),
                    "unfilled": [{**where(s), "reason": s.reason, "detail": s.detail} for s in unfilled]}
    filled_doc = {"schema": "fill_paper_filled_v1", "summary": summary, "notes": notes(R, all_slots),
                  "sources": dict(sorted(R.sha1.items())),
                  "filled": [{**where(s), "text": s.fill, "from": list(s.sources)} for s in filled]}
    outputs["UNFILLED.json"] = (json.dumps(unfilled_doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode()
    outputs["FILLED.json"] = (json.dumps(filled_doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode()
    outputs["CHECK_EQUAL.json"] = (json.dumps(ceq, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode()
    changed = []
    for rel, data in outputs.items():
        p = out / rel
        if p.is_file() and p.read_bytes() == data:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
        changed.append(rel)
    return {"summary": summary, "changed": changed, "slots": all_slots, "unfilled": unfilled_doc, "check_equal": ceq}


def print_check_equal(ceq: dict) -> None:
    sm = ceq["summary"]
    print(f"check_equal: {sm['quantities_printed_more_than_once']} quantities printed more than once "
          f"({sm['printed_in_more_than_one_table']} in more than one table); estimates differ: {sm['estimates_differ']}; "
          f"format or interval differs: {sm['format_or_interval_differs']}; numbers without a source: "
          f"{sm['numbers_without_a_source']}")
    for g in ceq["groups"]:
        flag = "DIFFERENT ESTIMATES" if not g["estimates_equal"] else ("format differs" if not g["printed_equal"] else "equal")
        print(f"  [{flag}] {g['quantity']}  ({g['n_printed']}x in {', '.join(g['tables'])})")
        for o in g["occurrences"]:
            where = (f"{o['table']} row {o['row']!r} col {o['column']}" if o["table"] else f"{o['file']}:{o['line']}")
            print(f"      {o['printed']:<44} {where}  <- {o['source']}")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper", default=str(ROOT / "Paper" / "sigir2027"))
    ap.add_argument("--results", default=str(ROOT / "docs" / "sigir" / "results"))
    ap.add_argument("--out", default=None, help="default: <paper>/filled")
    ap.add_argument("--check_equal", action="store_true",
                    help="print every number printed in more than one place with its sources (also written, always, "
                         "to <out>/CHECK_EQUAL.json)")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path(a.paper) / "filled"
    res = run(Path(a.paper), Path(a.results), out)
    sm = res["summary"]
    print(f"slots: {sm['slots_total']}  filled: {sm['filled']}  unfilled: {sm['unfilled']}")
    for k, v in sm["unfilled_by_reason"].items():
        print(f"  unfilled {k:<20} {v}")
    print("filled by alias:   " + ", ".join(f"{k} {v}" for k, v in sm["filled_by_alias"].items()))
    print("unfilled by alias: " + ", ".join(f"{k} {v}" for k, v in sm["unfilled_by_alias"].items()))
    bad = [c for c in res["unfilled"]["checks"] if not c["ok"]]
    for c in bad:
        print(f"  CHECK FAILED: {c['check']}: {c['a']} vs {c['b']}")
    ceq = res["check_equal"]["summary"]
    if ceq["estimates_differ"]:
        print(f"  CHECK_EQUAL: {ceq['estimates_differ']} quantity(ies) printed with different estimates (--check_equal)")
    if a.check_equal:
        print_check_equal(res["check_equal"])
    print(f"wrote {len(res['changed'])} changed file(s) to {out}")
    return res


if __name__ == "__main__":
    main()
