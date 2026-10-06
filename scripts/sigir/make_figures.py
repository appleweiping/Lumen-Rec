#!/usr/bin/env python3
"""Figures of the SIGIR 2027 paper, drawn from the committed result files and from nothing else.

    python scripts/sigir/make_figures.py [--results docs/sigir/results] [--out Paper/sigir2027/figures]
                                         [--only tracks,shares,serving,decomp]

Writes into --out: tracks.pdf, shares.pdf, serving.pdf (vector, TrueType-embedded, no creation date) and figures_manifest.json;
decomp.tex (the hand-written TikZ schematic, no data) is only hashed into the manifest. Nothing is typed by hand: every plotted
value is read from a file under --results at generation time, and figures_manifest.json lists, per figure and per panel, the
source files (relative path + sha1), every plotted record as "<file>:<dotted key path>" with the plotted fields, the flags that
mark a value exploratory or descriptive, the values withheld, and the panels or elements that are not run yet. The same inputs
give byte-identical outputs (no clock, host or absolute path is written; a file is rewritten only when its bytes change).

Figures (field names as read by scripts/sigir/fill_paper.py, see docs/sigir/PAPER_DATA_MAP.md):
  tracks   F1  UAUC of the confidence beside its same-row references, rated panels (Qwen3-8B: ML-1M, Toys, Video Games, Sports;
               Llama-3.1-8B: ML-1M, Toys). grid/<bb>/<d>.json: E_A.ZS.UAUC_TEST.per_model.zeroshot,
               E_A.FT.UAUC_TEST.{mean_over_seeds, per_model.s0-s2}, E_A.<ZS|FT>.UAUC_TEST.references.{q_hat, mf};
               extra[/<bb>]/<d>.json: E_J.<ZS|FT>.dUAUC_L_minus_q_hat_T.per_seed.<zeroshot|s0>.UAUC_b (the matched mean) and
               status.items_2_to_7.
  shares   F2  the within-user variance of the confidence logit and the information it adds. grid E_D.<reg>.shares.
               {per_model, mean_over_seeds}.{item_prior_share, non_prior_share, shares_reading}, E_D.<reg>.information_gain.
               {per_model.<m>.G, G_mean_over_seeds, G_CF}; extra E_G.<reg>.e_share, E_W.<reg>.{G_wu, G_CF_wu}, E_G.descriptive,
               status.items_2_to_7.
  serving  F3  NDCG@10 of serving only the most confident fraction of events against the random-subset reference.
               aud/<d>.json: segments.<seg>.questions.next.D_selective_serving.signals.{p_max, random}.curve.ndcg10.
  decomp   F4  the conceptual schematic figures/decomp.tex (no data).

Rules kept here (the paper's data discipline):
  * a missing result file draws an explicit "not run yet" placeholder naming the file; never an omission, never synthetic data;
  * a file that exists but lacks a key or holds a wrong type raises FigureDataError naming file and key; a block the file marks
    unavailable (available = false) or a null value is drawn as "not run yet" with the file's own reason;
  * LoRA values are drawn only after a recorded GATE_FT_PASS (gft/gate_ft.json, FILL RULE 4b);
  * shares the grid report reads "uninterpretable" are withheld (marked n/i);
  * values of an exploratory or descriptive analysis (status.items_2_to_7 = exploratory, a record or block flagged descriptive,
    fewer than 150 users or events) get a hollow marker; no p-value, Holm family or claim-admission label is read or drawn;
  * every read of a data file goes through Results, which refuses any path outside the results root.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")                                      # before pyplot: no display, no GUI backend
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402
from matplotlib.patches import Rectangle                   # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "docs" / "sigir" / "results"
DEFAULT_OUT = ROOT / "Paper" / "sigir2027" / "figures"
SCHEMA = "lumen_figures_manifest_v1"
MANIFEST_NAME = "figures_manifest.json"

# ------------------------------------------------------------------------------------------------ registered constants
# Design constants (not results), kept in step with scripts/sigir/fill_paper.py; a test compares them.
MIN_N = 150                                    # fewer users or events: descriptive (A3 section 1)
RATED = ("ml1m", "toys", "games", "sports")    # rated panels, registered order
RATED_NAME = {"ml1m": "ML-1M", "toys": "Toys", "games": "Video Games", "sports": "Sports"}
FT_MODELS = ("s0", "s1", "s2")                 # LoRA seeds
BACKBONE_TOKEN = {"qwen": "qwen", "llama": "llama"}      # substring of meta.backbone expected under each root, any case
BACKBONE_DEFAULT = {"qwen": "Qwen3-8B", "llama": "Llama-3.1-8B"}
TRACKS_PANELS = tuple(("qwen", d) for d in RATED) + (("llama", "ml1m"), ("llama", "toys"))
SHARES_PANELS = tuple(("qwen", d) for d in RATED)
SERVING_UNITS = (("Sports (S10k)", "sports", "events_1001_10000"), ("Toys", "toys", "all"), ("Home", "home", "all"),
                 ("Tools", "tools", "all"))      # the four family units of the next-item audit (fill_paper.FAMILY_UNITS)
SERVING_SIGNALS = ("p_max", "random")
GATE_FT = "gft/gate_ft.json"
FIG_IDS = ("tracks", "shares", "serving", "decomp")
FIG_ALIASES = {"F1": "tracks", "F2": "shares", "F3": "serving", "F4": "decomp", "fig_tracks": "tracks",
               "fig_shares": "shares", "fig_serving": "serving", "fig_decomp": "decomp"}
UAUC_AXIS_START = 0.45                         # UAUC axes start here (chance is 0.5) and say so; lower only if the data need it

# ------------------------------------------------------------------------------------------------ style
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"      # primary and secondary ink; muted (rules and ticks only, never text)
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f2f1ec"       # hairline grid, axis rule, band behind the references
C_ZS, C_LORA = "#2a78d6", "#eb6834"                     # categorical slots 1 and 2 of the validated default palette
C_REF = "#4d4d4d"                                        # non-LLM references: neutral gray, told apart by marker and row
STYLE = {
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.0, "axes.titlesize": 7.5,
    "axes.labelsize": 7.0, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.5, "axes.edgecolor": AXIS, "axes.facecolor": "none", "figure.facecolor": "none",
    "savefig.facecolor": "none", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "lines.solid_capstyle": "butt", "lines.solid_joinstyle": "round",
    "hatch.linewidth": 0.4, "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.compression": 6,
    "figure.dpi": 100, "savefig.dpi": 300,
}
LW_CI, LW_LINE, EDGE_W = 0.9, 1.1, 0.7
LABEL_PT, SMALL_PT = 7.0, 6.0
# F1: one row per series; ZS circle, LoRA diamond, references square / triangle / cross (distinct in greyscale)
TRACK_ROWS = {"zero_shot": 4.9, "lora": 3.9, "item_mean": 2.5, "matched_mean": 1.5, "mf": 0.5}
TRACK_BAND_TOP = 3.1
TRACK_STYLE = {
    "zero_shot": dict(label=r"zero-shot $\ell$", marker="o", color=C_ZS, ms=3.5),
    "lora": dict(label=r"LoRA $\ell$", marker="D", color=C_LORA, ms=3.0),
    "item_mean": dict(label=r"item mean $m$", marker="s", color=C_REF, ms=3.0),
    "matched_mean": dict(label=r"matched mean $m_T^{\,\dagger}$", marker="^", color=C_REF, ms=3.6),
    "mf": dict(label="temporal MF", marker="P", color=C_REF, ms=3.8),
}
SEED_MS = 2.2
CAP_MS = 3.0                                   # end caps of the intervals
# F2: marker per series (colour = regime as in F1; the MF reference is gray)
SHARE_STYLE = {"zs": dict(marker="o", color=C_ZS, ms=4.2), "lora": dict(marker="D", color=C_LORA, ms=3.6),
               "ref": dict(marker="s", color=C_REF, ms=3.6)}
FLAG_MEANING = {
    "exploratory": "the extra file's status.items_2_to_7 is 'exploratory' (addendum 6, ML-1M): no confirmatory reading",
    "descriptive": "the record or block is flagged descriptive in the file (descriptive_min_n, E_G.descriptive) or rests on fewer "
                   "than the minimum number of users or events: no interval-based claim",
    "rows_differ": "the reference was computed on other rows than the LLM score it is drawn beside",
}


# ------------------------------------------------------------------------------------------------ errors
class FigureDataError(RuntimeError):
    """A result file that exists but cannot be used (invalid JSON, a missing key, a wrong type): names the file and the key."""


class PathGuardError(FigureDataError):
    """A read outside the results root was asked for; it is refused and nothing is opened."""


class MissingFile(Exception):
    """The result file does not exist in the results tree: the panel or element is drawn as 'not run yet'."""

    def __init__(self, rel: str):
        super().__init__(rel)
        self.rel = rel


class NotAvailable(Exception):
    """A block the file marks unavailable (available = false), or a null / non-finite value: 'not run yet' with the reason."""

    def __init__(self, rel: str, key: str, reason, kind: str = "block"):
        super().__init__(f"{rel}: {key}: {reason}")
        self.rel, self.key, self.reason, self.kind = rel, key, (str(reason) if reason else ""), kind


# ------------------------------------------------------------------------------------------------ results access
def _dotted(path) -> str:
    return ".".join(str(k) for k in path)


def _safe_rel(rel) -> tuple:
    """The parts of a relative POSIX path inside the results root; anything else is refused."""
    if not isinstance(rel, str) or not rel.strip():
        raise PathGuardError(f"empty or non-string result path {rel!r}")
    if "\\" in rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        raise PathGuardError(f"result path {rel!r} must be a relative POSIX path inside the results root")
    parts = PurePosixPath(rel).parts
    if not parts or any(p in ("..", "") for p in parts):
        raise PathGuardError(f"result path {rel!r} leaves the results root")
    return parts


class Results:
    """Read-only, cached, guarded access to the result files under one root. Every read of a data file goes through load()."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FigureDataError(f"results root {str(root)!r} is not a directory")
        self._cache: dict = {}
        self._sha1: dict = {}
        self.opened: list = []                      # relative paths opened, in order (audit trail)

    def path(self, rel: str) -> Path:
        p = self.root.joinpath(*_safe_rel(rel)).resolve()
        if not p.is_relative_to(self.root):           # also catches a link that points outside the root
            raise PathGuardError(f"result path {rel!r} resolves outside the results root")
        return p

    def exists(self, rel: str) -> bool:
        return self.path(rel).is_file()

    def load(self, rel: str):
        if rel in self._cache:
            return self._cache[rel]
        p = self.path(rel)
        if not p.is_file():
            raise MissingFile(rel)
        data = p.read_bytes()
        self.opened.append(rel)
        self._sha1[rel] = hashlib.sha1(data).hexdigest()
        try:
            obj = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as e:
            raise FigureDataError(f"{rel}: not valid JSON ({e})") from None
        if not isinstance(obj, dict):
            raise FigureDataError(f"{rel}: top level is {type(obj).__name__}, expected an object")
        self._cache[rel] = obj
        return obj

    def sha1(self, rel: str) -> str:
        self.load(rel)
        return self._sha1[rel]

    def get(self, rel: str, *path):
        """The node at `path` inside `rel`. A block with available = false raises NotAvailable (with the file's own reason),
        a missing key FigureDataError naming file and key."""
        node, done = self.load(rel), []
        for k in path:
            if isinstance(node, dict) and node.get("available") is False:
                raise NotAvailable(rel, _dotted(done) or "(top)", node.get("reason"))
            if isinstance(node, dict) and k in node:
                node = node[k]
            elif isinstance(node, list) and isinstance(k, int) and -len(node) <= k < len(node):
                node = node[k]
            else:
                have = sorted(node) if isinstance(node, dict) else type(node).__name__
                raise FigureDataError(f"{rel}: missing key {_dotted(done + [k])!r} (found: {have})")
            done.append(k)
        if isinstance(node, dict) and node.get("available") is False:
            raise NotAvailable(rel, _dotted(done), node.get("reason"))
        return node


def _as_results(results) -> Results:
    return results if isinstance(results, Results) else Results(results)


def _number(rel: str, key: str, x):
    """A finite float. null or non-finite -> NotAvailable (not computable on its rows); any other type -> FigureDataError."""
    if x is None:
        raise NotAvailable(rel, key, "null in the result file (not computable on its rows)")
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise FigureDataError(f"{rel}: key {key!r} is not a number: {x!r}")
    if not math.isfinite(float(x)):
        raise NotAvailable(rel, key, "not finite in the result file")
    return float(x)


# ------------------------------------------------------------------------------------------------ plotted values
@dataclass
class Val:
    """One plotted quantity: the fields exactly as the result file holds them, with the key path they were read from."""
    uid: str
    series: str
    source: str                     # "<relative file>:<dotted key path>" of the record (or scalar) read
    fields: dict                    # plotted fields, same names and values as in the file
    role: str = "point"             # point | seed | count
    flags: tuple = ()               # subset of exploratory, descriptive, rows_differ: drawn hollow
    flag_sources: tuple = ()        # the file keys the flags (or the absence of a flag) were read from
    n: int | None = None            # users / events behind the record

    @property
    def rel(self) -> str:
        return self.source.split(":", 1)[0]

    @property
    def est(self):
        return self.fields.get("est")

    @property
    def lo(self):
        return self.fields.get("lo")

    @property
    def hi(self):
        return self.fields.get("hi")

    @property
    def hollow(self) -> bool:
        return bool(self.flags)

    def add_flag(self, flag: str, source: str) -> None:
        self.flags = tuple(dict.fromkeys(self.flags + (flag,)))
        self.flag_sources = tuple(dict.fromkeys(self.flag_sources + (source,)))

    def manifest(self) -> dict:
        return {"uid": self.uid, "series": self.series, "role": self.role, "source": self.source, "fields": dict(self.fields),
                "flags": list(self.flags), "flag_sources": list(self.flag_sources)}


def _is_count(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def record_val(res: "Results", rel: str, path: tuple, *, uid: str, series: str, fields=("est", "lo", "hi"), role: str = "point",
               flags=(), flag_sources=(), n_key: str = "n_users") -> Val:
    """A Val from a result record {est, lo, hi, n_users | n, descriptive_min_n}. The estimate is required; the interval is
    optional but never half given."""
    node = res.get(rel, *path)
    key = _dotted(path)
    if not isinstance(node, dict):
        raise FigureDataError(f"{rel}: key {key!r} is not a result record (a mapping with 'est'): {type(node).__name__}")
    out: dict = {}
    for f in fields:
        if f in ("lo", "hi"):
            continue
        if f not in node:
            raise FigureDataError(f"{rel}: missing key {key + '.' + f!r}")
        out[f] = _number(rel, f"{key}.{f}", node[f])
    if "lo" in fields or "hi" in fields:
        lo, hi = node.get("lo"), node.get("hi")
        if (lo is None) != (hi is None):
            raise FigureDataError(f"{rel}: key {key!r} has only one end of its interval (lo / hi)")
        if lo is not None:
            lo, hi = _number(rel, f"{key}.lo", lo), _number(rel, f"{key}.hi", hi)
            if lo > hi:
                raise FigureDataError(f"{rel}: key {key!r} has lo > hi")
            out["lo"], out["hi"] = lo, hi
    fl, fs = list(flags), list(flag_sources)
    n = node.get(n_key)
    if node.get("descriptive_min_n") is True:
        fl.append("descriptive")
        fs.append(f"{rel}:{key}.descriptive_min_n")
    elif _is_count(n) and n < MIN_N:
        fl.append("descriptive")
        fs.append(f"{rel}:{key}.{n_key}")
    return Val(uid=uid, series=series, source=f"{rel}:{key}", fields=out, role=role, flags=tuple(dict.fromkeys(fl)),
               flag_sources=tuple(dict.fromkeys(fs)), n=int(n) if _is_count(n) else None)


def scalar_val(res: "Results", rel: str, path: tuple, *, uid: str, series: str, flags=(), flag_sources=()) -> Val:
    """A Val from a bare number (an estimate that has no interval in its file)."""
    x = _number(rel, _dotted(path), res.get(rel, *path))
    return Val(uid=uid, series=series, source=f"{rel}:{_dotted(path)}", fields={"est": x}, flags=tuple(flags),
               flag_sources=tuple(flag_sources))


def count_val(res: "Results", rel: str, path: tuple, key: str, *, uid: str, series: str) -> Val:
    """A count read from a mapping (users, events): printed in a panel header, not plotted as a mark."""
    node = res.get(rel, *path)
    if not isinstance(node, dict) or key not in node:
        raise FigureDataError(f"{rel}: missing key {_dotted(tuple(path) + (key,))!r}")
    x = node[key]
    if not _is_count(x):
        raise FigureDataError(f"{rel}: key {_dotted(tuple(path) + (key,))!r} is not a count: {x!r}")
    return Val(uid=uid, series=series, source=f"{rel}:{_dotted(path)}", fields={key: x}, role="count", n=int(x))


# ------------------------------------------------------------------------------------------------ panels and figures
@dataclass
class Panel:
    pid: str
    title: str
    group: str = ""
    group_label: str = ""
    vals: list = field(default_factory=list)
    missing: list = field(default_factory=list)       # files or elements not run yet, with the path and the reason
    withheld: list = field(default_factory=list)      # values the file marks uninterpretable: not drawn
    notes: list = field(default_factory=list)
    extra_sources: list = field(default_factory=list)  # files read that carry no plotted value (the gate)

    def add(self, series: str, build):
        """Run build(); a missing file or a block marked unavailable becomes an explicit 'not run yet' entry."""
        try:
            out = build()
        except MissingFile as e:
            self.missing.append({"series": series, "path": e.rel, "kind": "file", "reason": "result file not found"})
            return None
        except NotAvailable as e:
            self.missing.append({"series": series, "path": e.rel, "key": e.key, "kind": e.kind,
                                 "reason": e.reason or "marked not available in the result file"})
            return None
        self.vals.extend(out if isinstance(out, list) else [out])
        return out

    def not_found(self, rel: str, series: str | None = None) -> None:
        self.missing.append({"series": series, "path": rel, "kind": "file", "reason": "result file not found"})

    def data(self) -> list:
        return [v for v in self.vals if v.role != "count"]

    def by_series(self, series: str) -> list:
        return [v for v in self.vals if v.series == series]

    def one(self, series: str):
        found = self.by_series(series)
        return found[0] if found else None

    @property
    def status(self) -> str:
        if not self.data():
            return "not_run"
        return "partial" if (self.missing or self.withheld) else "ok"

    def sources(self) -> list:
        rels = {v.rel for v in self.vals} | set(self.extra_sources)
        for v in self.vals:
            rels |= {s.split(":", 1)[0] for s in v.flag_sources}
        return sorted(rels)

    def n_text(self):
        c = self.one("n_users") or self.one("n_events")
        return None if c is None else f"{c.n:,}"


@dataclass
class FigureSpec:
    fid: str
    title: str
    size_in: tuple
    panels: list
    headers: dict = field(default_factory=dict)      # group -> label shown above its panels
    notes: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        sts = [p.status for p in self.panels]
        if sts and all(s == "not_run" for s in sts):
            return "not_run"
        return "ok" if all(s == "ok" for s in sts) else "partial"

    def used_files(self) -> list:
        return sorted({r for p in self.panels for r in p.sources()})

    def missing_files(self) -> list:
        out: dict = {}
        for p in self.panels:
            for m in p.missing:
                if m.get("kind") == "file" and m.get("path"):
                    out.setdefault(m["path"], []).append(p.pid if not m.get("series") else f"{p.pid}/{m['series']}")
        return [{"path": k, "needed_by": v} for k, v in sorted(out.items())]

    def n_boot_values(self, res: "Results") -> list:
        seen = set()
        for p in self.panels:
            for v in p.vals:
                node = res.get(v.rel, *_split_key(v.source))
                if isinstance(node, dict) and isinstance(node.get("n_boot"), int):
                    seen.add(node["n_boot"])
        return sorted(seen)

    def flags_used(self) -> list:
        return sorted({f for p in self.panels for v in p.vals for f in v.flags})

    def entry(self, res: "Results", *, file: str, sha1: str, nbytes: int) -> dict:
        def srcs(rels):
            return [{"path": r, "sha1": res.sha1(r)} for r in rels]
        return {
            "id": self.fid, "kind": "plot", "title": self.title, "file": file, "sha1": sha1, "bytes": nbytes,
            "size_in": list(self.size_in), "status": self.status, "sources": srcs(self.used_files()),
            "missing": self.missing_files(), "n_boot_values": self.n_boot_values(res), "flags_used": self.flags_used(),
            "flag_meaning": FLAG_MEANING, "notes": list(self.notes), "meta": dict(self.meta),
            "panels": [{"id": p.pid, "title": p.title, "group": p.group, "group_label": p.group_label, "status": p.status,
                        "sources": srcs(p.sources()), "values": [v.manifest() for v in p.vals], "withheld": list(p.withheld),
                        "missing": list(p.missing), "notes": list(p.notes)} for p in self.panels],
        }


def _split_key(source: str) -> tuple:
    key = source.split(":", 1)[1]
    return tuple(int(k) if k.isdigit() else k for k in key.split("."))


# ------------------------------------------------------------------------------------------------ gates, flags, meta checks
@dataclass
class FtGate:
    ok: bool
    state: str                 # pass | not_passed | missing
    decision: str | None
    reason: str


def ft_gate(res: Results) -> FtGate:
    """FILL RULE 4b: LoRA items exist only after a recorded GATE_FT_PASS (gft/gate_ft.json decision)."""
    try:
        dec = res.get(GATE_FT, "decision")
    except MissingFile:
        return FtGate(False, "missing", None, f"{GATE_FT} not found: the LoRA series wait for the recorded Gate-FT decision")
    if dec == "GATE_FT_PASS":
        return FtGate(True, "pass", dec, "")
    return FtGate(False, "not_passed", str(dec), f"{GATE_FT} decision is {dec}: fine-tuned items are not reported (FILL RULE 4b)")


def _gate_block(g: FtGate):
    """The exception that leaves a LoRA element undrawn because the gate is not passed (file missing or decision not a pass)."""
    if g.state == "missing":
        return MissingFile(GATE_FT)
    return NotAvailable(GATE_FT, "decision", g.reason, kind="gate")


def check_meta(res: Results, rel: str, *, domain: str, backbone: str | None = None, top: bool = False) -> str:
    """The file must belong where it is read: its domain (and backbone) are checked. Returns the backbone label of the file."""
    key = "domain" if top else "meta.domain"
    dom = res.get(rel, "domain") if top else res.get(rel, "meta", "domain")
    if dom != domain:
        raise FigureDataError(f"{rel}: key {key!r} is {dom!r}, expected {domain!r} (wrong file in this place)")
    if backbone is None:
        return ""
    bb = res.get(rel, "meta", "backbone")
    if BACKBONE_TOKEN[backbone] not in str(bb).lower():
        raise FigureDataError(f"{rel}: key 'meta.backbone' is {bb!r}, expected a {backbone} backbone (wrong root)")
    return str(bb)


def ext_rel(bb: str, d: str) -> str:
    """The addendum-6 file of a backbone's root: extra/<d>.json (Qwen main root), extra/llama/<d>.json (Llama root)."""
    return f"extra/{d}.json" if bb == "qwen" else f"extra/{bb}/{d}.json"


def ext_status_flags(res: Results, rel: str) -> tuple:
    """(flags, flag_sources) of an addendum-6 value: items 2-7 are exploratory on ML-1M and registered, outcome-free elsewhere."""
    st = res.get(rel, "status", "items_2_to_7")
    src = (f"{rel}:status.items_2_to_7",)
    if st == "exploratory":
        return ("exploratory",), src
    if st == "registered_outcome_free":
        return (), src
    raise FigureDataError(f"{rel}: key 'status.items_2_to_7' is {st!r}, expected 'exploratory' or 'registered_outcome_free'")


def _optional_flag(res: Results, rel: str, path: tuple, name: str) -> tuple:
    """(flags, flag_sources) from an optional boolean flag of a block (E_G.descriptive); absent means not flagged."""
    try:
        v = res.get(rel, *path)
    except (FigureDataError, NotAvailable):
        return (), ()
    return ((name,), (f"{rel}:{_dotted(path)}",)) if v is True else ((), ())


def _regime_ok(res: Results, rel: str, block: str, reg: str) -> None:
    """A regime block must be complete (never replaced: a missing or excluded seed leaves the cell undrawn)."""
    node = res.get(rel, block, reg)
    if isinstance(node, dict) and node.get("complete") is False:
        raise NotAvailable(rel, f"{block}.{reg}", f"incomplete: models {node.get('models')}, missing or excluded "
                                                  f"{node.get('missing_or_excluded')} (a missing seed is never replaced)")


def _prefer(builders):
    """The first builder that does not raise NotAvailable (the zero-shot rows first, then the fine-tuned rows, as fill_paper)."""
    first = None
    for b in builders:
        try:
            return b()
        except NotAvailable as e:
            first = first or e
    raise first if first is not None else RuntimeError("no builder given")


def _seed_check(res: Results, rel: str, container: tuple) -> None:
    have = res.get(rel, *container)
    absent = [m for m in FT_MODELS if m not in have]
    if absent:
        raise NotAvailable(rel, _dotted(container), f"seeds {absent} missing (s0-s2 registered; a missing seed is never replaced)")


# ------------------------------------------------------------------------------------------------ F1 tracks: collect
def collect_tracks(res: Results, panels=TRACKS_PANELS) -> FigureSpec:
    gate = ft_gate(res)
    out = [_tracks_panel(res, bb, d, gate) for bb, d in panels]
    headers: dict = {}
    for p in out:
        headers.setdefault(p.group, BACKBONE_DEFAULT.get(p.group, p.group))
        if p.group_label:
            headers[p.group] = p.group_label
    notes = [] if gate.ok else [gate.reason]
    return FigureSpec("tracks", "UAUC of the confidence beside its same-row references", (7.0, 2.3), out, headers, notes,
                      {"gate_ft": {"state": gate.state, "decision": gate.decision}})


def _tracks_panel(res: Results, bb: str, d: str, gate: FtGate) -> Panel:
    p = Panel(pid=f"{bb}:{d}", title=RATED_NAME[d], group=bb)
    grid, ext = f"grid/{bb}/{d}.json", ext_rel(bb, d)
    if not res.exists(grid):
        p.not_found(grid)
        return p
    p.group_label = check_meta(res, grid, domain=d, backbone=bb)
    uid = lambda s: f"tracks/{p.pid}/{s}"                                            # noqa: E731
    regs = ("ZS",) + (("FT",) if gate.ok else ())

    def n_users():
        return _prefer([lambda r=r: count_val(res, grid, ("E_A", r, "UAUC_TEST", "rows"), "n_users", uid=uid("n_users"),
                                              series="n_users") for r in regs])

    def zs():
        _regime_ok(res, grid, "E_A", "ZS")
        return record_val(res, grid, ("E_A", "ZS", "UAUC_TEST", "per_model", "zeroshot"), uid=uid("zero_shot"), series="zero_shot")

    def lora():
        if not gate.ok:
            raise _gate_block(gate)
        _regime_ok(res, grid, "E_A", "FT")
        base = ("E_A", "FT", "UAUC_TEST")
        _seed_check(res, grid, base + ("per_model",))
        mean = record_val(res, grid, base + ("mean_over_seeds",), uid=uid("lora"), series="lora")
        seeds = [record_val(res, grid, base + ("per_model", m), uid=uid(f"lora_seed{k}"), series=f"lora_seed{k}",
                            fields=("est",), role="seed") for k, m in enumerate(FT_MODELS)]
        return [mean] + seeds

    def reference(name: str, series: str):
        def f():
            v = _prefer([lambda r=r: record_val(res, grid, ("E_A", r, "UAUC_TEST", "references", name), uid=uid(series),
                                                series=series) for r in regs])
            if "FT" in regs and v.source.endswith(f"E_A.ZS.UAUC_TEST.references.{name}"):
                other = None
                try:
                    other = res.get(grid, "E_A", "FT", "UAUC_TEST", "references", name)
                except (FigureDataError, NotAvailable):
                    pass
                if isinstance(other, dict) and any(other.get(k) != v.fields[k] for k in v.fields):
                    v.add_flag("rows_differ", f"{grid}:E_A.FT.UAUC_TEST.references.{name}")
                    p.notes.append(f"{name}: the zero-shot and fine-tuned blocks differ (different rows); the zero-shot block is drawn")
            return v
        return f

    def matched():
        if not res.exists(ext):
            raise MissingFile(ext)
        check_meta(res, ext, domain=d, backbone=bb)
        flags, fsrc = ext_status_flags(res, ext)
        first = None
        for reg, m in (("ZS", "zeroshot"), ("FT", "s0")):
            if reg not in regs:
                continue
            try:
                v = scalar_val(res, ext, ("E_J", reg, "dUAUC_L_minus_q_hat_T", "per_seed", m, "UAUC_b"), uid=uid("matched_mean"),
                               series="matched_mean", flags=flags, flag_sources=fsrc)
                rows_j = res.get(ext, "E_J", reg, "rows_E_A")
                rows_a = res.get(grid, "E_A", reg, "UAUC_TEST", "rows")
            except NotAvailable as e:
                first = first or e
                continue
            if any(rows_j.get(k) != rows_a.get(k) for k in ("n_users", "n_pairs")):
                v.add_flag("rows_differ", f"{ext}:E_J.{reg}.rows_E_A")
                p.notes.append("matched mean: computed on other rows than E-A's TEST rows")
            return v
        raise first or NotAvailable(ext, "E_J", "no usable block")

    p.add("n_users", n_users)
    p.add("zero_shot", zs)
    p.add("lora", lora)
    p.add("item_mean", reference("q_hat", "item_mean"))
    p.add("matched_mean", matched)
    p.add("mf", reference("mf", "mf"))
    if gate.state != "missing":
        p.extra_sources.append(GATE_FT)
    return p


# ------------------------------------------------------------------------------------------------ F2 shares: collect
def _q_paths(qid: str, grid: str, ext: str) -> dict:
    """Where a quantity lives. zs: the zero-shot record; mean: the LoRA seed mean; container + (m,) + leaf: the record of seed m."""
    if qid in ("item_prior_share", "non_prior_share"):
        return dict(rel=grid, ext=False, top="E_D", zs=("E_D", "ZS", "shares", "per_model", "zeroshot", qid),
                    mean=("E_D", "FT", "shares", "mean_over_seeds", qid), container=("E_D", "FT", "shares", "per_model"),
                    leaf=(qid,), read_zs=("E_D", "ZS", "shares", "per_model", "zeroshot", "shares_reading"),
                    read_mean=("E_D", "FT", "shares", "mean_over_seeds", "shares_reading"), read_leaf=("shares_reading",))
    if qid == "e_share":
        return dict(rel=ext, ext=True, top="E_G", zs=("E_G", "ZS", "e_share", "per_model", "zeroshot", "e_share"),
                    mean=("E_G", "FT", "e_share", "mean_over_seeds"), container=("E_G", "FT", "e_share", "per_model"),
                    leaf=("e_share",))
    if qid == "G":
        return dict(rel=grid, ext=False, top="E_D", zs=("E_D", "ZS", "information_gain", "per_model", "zeroshot", "G"),
                    mean=("E_D", "FT", "information_gain", "G_mean_over_seeds"),
                    container=("E_D", "FT", "information_gain", "per_model"), leaf=("G",))
    if qid == "G_wu":
        return dict(rel=ext, ext=True, top="E_W", zs=("E_W", "ZS", "G_wu", "per_model", "zeroshot"),
                    mean=("E_W", "FT", "G_wu", "mean_over_seeds"), container=("E_W", "FT", "G_wu", "per_model"), leaf=())
    raise KeyError(qid)


SHARE_QUANTITIES = ("item_prior_share", "non_prior_share", "e_share")
GAIN_QUANTITIES = ("G", "G_wu")


def collect_shares(res: Results, panels=SHARES_PANELS) -> FigureSpec:
    gate = ft_gate(res)
    out = [_shares_panel(res, bb, d, gate) for bb, d in panels]
    headers = {p.group: p.group_label or BACKBONE_DEFAULT.get(p.group, p.group) for p in out}
    notes = [] if gate.ok else [gate.reason]
    return FigureSpec("shares", "What the confidence is made of, and the information it adds", (7.0, 2.3), out, headers, notes,
                      {"gate_ft": {"state": gate.state, "decision": gate.decision}})


def _shares_panel(res: Results, bb: str, d: str, gate: FtGate) -> Panel:
    p = Panel(pid=f"{bb}:{d}", title=RATED_NAME[d], group=bb)
    grid, ext = f"grid/{bb}/{d}.json", ext_rel(bb, d)
    if not res.exists(grid):
        p.not_found(grid)
        return p
    p.group_label = check_meta(res, grid, domain=d, backbone=bb)
    uid = lambda s: f"shares/{p.pid}/{s}"                                            # noqa: E731
    regs = ("ZS",) + (("FT",) if gate.ok else ())

    def n_users():
        return _prefer([lambda r=r: count_val(res, grid, ("E_D", r, "shares", "rows"), "n_users", uid=uid("n_users"),
                                              series="n_users") for r in regs])

    def build_q(qid: str, reg: str):
        """The zero-shot record (reg ZS) or the LoRA seed mean followed by the three seed records (reg FT) of one quantity."""
        q = _q_paths(qid, grid, ext)
        rel = q["rel"]
        fl, fs = (), ()
        if q["ext"]:
            if not res.exists(rel):
                raise MissingFile(rel)
            check_meta(res, rel, domain=d, backbone=bb)
            fl, fs = ext_status_flags(res, rel)
            if qid == "e_share":
                f2, s2 = _optional_flag(res, rel, ("E_G", "descriptive"), "descriptive")
                fl, fs = tuple(dict.fromkeys(fl + f2)), fs + s2
        if reg == "FT" and not gate.ok:
            raise _gate_block(gate)
        _regime_ok(res, rel, q["top"], reg)
        is_share = "read_zs" in q
        tag = "zs" if reg == "ZS" else "lora"

        def one(path, series, role, reading_path):
            if is_share:
                reading = res.get(rel, *reading_path)
                if reading == "uninterpretable":
                    p.withheld.append({"series": series, "source": f"{rel}:{_dotted(path)}",
                                       "reason": f"{rel}:{_dotted(reading_path)} is 'uninterpretable': the value is not drawn"})
                    return None
                if reading != "interpretable":
                    raise FigureDataError(f"{rel}: key {_dotted(reading_path)!r} is {reading!r}, expected 'interpretable' or "
                                          "'uninterpretable'")
            return record_val(res, rel, path, uid=uid(series), series=series, role=role,
                              fields=("est", "lo", "hi") if role == "point" else ("est",), flags=fl, flag_sources=fs)

        if reg == "ZS":
            v = one(q["zs"], f"{qid}.{tag}", "point", q.get("read_zs"))
            return [] if v is None else [v]
        _seed_check(res, rel, q["container"])
        out = []
        mean = one(q["mean"], f"{qid}.{tag}", "point", q.get("read_mean"))
        if mean is not None:
            out.append(mean)
        for k, m in enumerate(FT_MODELS):
            sv = one(q["container"] + (m,) + q["leaf"], f"{qid}.{tag}_seed{k}", "seed",
                     q["container"] + (m,) + q["read_leaf"] if is_share else None)
            if sv is not None:
                out.append(sv)
        return out

    def single(rel: str, path_with_reg: tuple, series: str, flags=(), flag_sources=()):
        """A quantity with no LLM feature (G_CF, G_CF_wu): one value per panel, from the zero-shot rows, else the LoRA rows."""
        def at(reg):
            path = tuple(reg if k is None else k for k in path_with_reg)
            return lambda: record_val(res, rel, path, uid=uid(series), series=series, flags=flags, flag_sources=flag_sources)
        return _prefer([at(r) for r in regs])

    def g_cf_wu():
        if not res.exists(ext):
            raise MissingFile(ext)
        check_meta(res, ext, domain=d, backbone=bb)
        fl, fs = ext_status_flags(res, ext)
        return single(ext, ("E_W", None, "G_CF_wu"), "G_CF_wu", fl, fs)

    p.add("n_users", n_users)
    for qid in SHARE_QUANTITIES + GAIN_QUANTITIES:
        for reg in regs:
            p.add(f"{qid}.{'zs' if reg == 'ZS' else 'lora'}", lambda qid=qid, reg=reg: build_q(qid, reg))
    p.add("G_CF", lambda: single(grid, ("E_D", None, "information_gain", "G_CF"), "G_CF"))
    p.add("G_CF_wu", g_cf_wu)
    if gate.state != "missing":
        p.extra_sources.append(GATE_FT)
    if not gate.ok:                                                           # one explicit entry: the LoRA series wait
        p.missing.append({"series": "lora", "path": GATE_FT, "key": "decision", "reason": gate.reason,
                          "kind": "file" if gate.state == "missing" else "gate"})
    return p


# ------------------------------------------------------------------------------------------------ F3 serving: collect
def collect_serving(res: Results, units=SERVING_UNITS) -> FigureSpec:
    out = [_serving_panel(res, label, d, seg) for label, d, seg in units]
    return FigureSpec("serving", "Serving only the most confident events against a random subset", (7.0, 2.0), out, {}, [], {})


def _serving_panel(res: Results, label: str, d: str, seg: str) -> Panel:
    p = Panel(pid=f"{d}:{seg}", title=label, group="audit")
    rel = f"aud/{d}.json"
    if not res.exists(rel):
        p.not_found(rel)
        return p
    check_meta(res, rel, domain=d, top=True)
    base = ("segments", seg, "questions", "next", "D_selective_serving")
    uid = lambda s: f"serving/{p.pid}/{s}"                                           # noqa: E731

    def n_events():
        return count_val(res, rel, ("segments", seg), "n_events", uid=uid("n_events"), series="n_events")

    def curve(sig: str):
        def f():
            pts = res.get(rel, *base, "signals", sig, "curve", "ndcg10")
            if not isinstance(pts, list) or not pts:
                raise FigureDataError(f"{rel}: key {_dotted(base + ('signals', sig, 'curve', 'ndcg10'))!r} is not a non-empty list")
            vals = [record_val(res, rel, base + ("signals", sig, "curve", "ndcg10", i), uid=uid(f"{sig}.{i}"), series=sig,
                               fields=("coverage", "est", "lo", "hi"), n_key="n") for i in range(len(pts))]
            cov = [v.fields["coverage"] for v in vals]
            if any(b <= a for a, b in zip(cov, cov[1:])):
                raise FigureDataError(f"{rel}: coverage points under {_dotted(base + ('signals', sig, 'curve', 'ndcg10'))!r} "
                                      "are not increasing")
            return vals
        return f

    p.add("n_events", n_events)
    for sig in SERVING_SIGNALS:
        p.add(sig, curve(sig))
    return p


# ------------------------------------------------------------------------------------------------ drawing helpers
class Canvas:
    """Places artists in inches from the lower left corner of a figure of a given size."""

    def __init__(self, fig, w: float, h: float):
        self.fig, self.w, self.h = fig, w, h

    def axes(self, left, bottom, width, height):
        return self.fig.add_axes([left / self.w, bottom / self.h, width / self.w, height / self.h])

    def text(self, x, y, s, **kw):
        return self.fig.text(x / self.w, y / self.h, s, **kw)

    def line(self, x0, y0, x1, y1, **kw):
        ln = Line2D([x0 / self.w, x1 / self.w], [y0 / self.h, y1 / self.h], transform=self.fig.transFigure, **kw)
        self.fig.add_artist(ln)
        return ln


def style():
    """The rc context every figure is drawn and saved in (global rcParams are never touched)."""
    return plt.rc_context(STYLE)


def _axes_style(ax, *, left: bool = False) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["left"].set_color(AXIS)
    ax.set_axisbelow(True)


def _wrap_path(path: str, width: int) -> list:
    """Wrap a relative path after '/' so that it fits `width` characters per line."""
    lines, cur = [], ""
    for part in re.findall(r"[^/]+/?", path):
        if cur and len(cur) + len(part) > width:
            lines.append(cur)
            cur = ""
        cur += part
    if cur:
        lines.append(cur)
    return lines


def fig_not_run_placeholder(ax, title: str, missing: list, *, note: str = "not run yet") -> dict:
    """Draw an explicit 'not run yet' placeholder into `ax`: a hatched empty frame, the panel title, `note`, the reason and the
    missing result file(s). Used for any panel whose inputs are missing (never an omission, never synthetic data). Returns the
    record that goes into the manifest."""
    ax.cla()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f7f6f2", edgecolor="#dddcd3", hatch="////", lw=0))
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="none", edgecolor=AXIS, lw=0.5))
    fig = ax.figure
    width_in = ax.get_position().width * fig.get_figwidth()
    chars = max(10, int(width_in * 26))                                     # about 26 characters per inch at the small size
    reasons = list(dict.fromkeys(m.get("reason") or "result file not found" for m in missing)) or ["result file not found"]
    lines = []
    for r in reasons:
        lines += textwrap.wrap(r, chars) or [""]
    for path in dict.fromkeys(m["path"] for m in missing if m.get("path")):
        lines += _wrap_path(path, chars)
    box = dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9)
    ax.text(0.5, 0.60, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=LABEL_PT, style="italic", color=INK2, bbox=box,
            gid="placeholder:note")
    ax.text(0.5, 0.55, "\n".join(lines), transform=ax.transAxes, ha="center", va="top", fontsize=SMALL_PT, color=INK2, bbox=box,
            linespacing=1.15, gid="placeholder:reason")
    return {"title": title, "status": "not_run", "note": note, "missing": [dict(m) for m in missing]}


def _marker(ax, x, y, st: dict, *, hollow: bool, gid: str, zorder: int = 5, **kw):
    ax.plot([x], [y], ls="none", marker=st["marker"], ms=st["ms"], mfc="white" if hollow else st["color"], mec=st["color"],
            mew=EDGE_W, zorder=zorder, gid=gid, clip_on=False, **kw)


def _seed_dot(ax, x, y, color, gid: str, hollow: bool = False):
    ax.plot([x], [y], ls="none", marker="o", ms=SEED_MS, mfc="white" if hollow else color, mec=color if hollow else "white",
            mew=0.6 if hollow else 0.3, alpha=0.85, zorder=4, gid=gid, clip_on=False)


def _header_text(cv: Canvas, x, y, s, **kw):
    return cv.text(x, y, s, ha="center", va="center", **kw)


def _legend(fig, handles, labels, *, loc, anchor, ncol):
    return fig.legend(handles=handles, labels=labels, loc=loc, bbox_to_anchor=anchor, ncol=ncol, frameon=False, handletextpad=0.4,
                      columnspacing=1.3, handlelength=1.6, borderaxespad=0.0, labelcolor=INK2)


def _proxy(marker="o", color=C_REF, hollow=False, ms=4.0, ls="none", lw=LW_CI):
    return Line2D([], [], ls=ls, lw=lw, marker=marker, ms=ms, mfc="white" if hollow else color, mec=color, mew=EDGE_W, color=color)


def _any_hollow(spec: "FigureSpec") -> bool:
    return any(v.hollow for p in spec.panels for v in p.data())


def _nice_ceil(x: float, step: float) -> float:
    return math.ceil(round(x / step, 9)) * step


def _fmt(t: float, nd: int = 1) -> str:
    return f"{t:.{nd}f}".replace("-", "−")


# ------------------------------------------------------------------------------------------------ F1 tracks: draw
def _uauc_limits(spec: FigureSpec) -> tuple:
    lows, highs = [], []
    for p in spec.panels:
        for v in p.data():
            lows.append(v.lo if v.lo is not None else v.est)
            highs.append(v.hi if v.hi is not None else v.est)
    xmin = UAUC_AXIS_START
    if lows:
        while min(lows) < xmin:
            xmin = round(xmin - 0.05, 2)
    xmax = _nice_ceil(max(highs) + 0.02, 0.05) if highs else xmin + 0.4
    return xmin, xmax


def draw_tracks(spec: FigureSpec):
    """F1: one panel per (backbone, domain); rows = zero-shot, LoRA (seed mean, seeds as dots), then the same-row references."""
    with style():
        W, H = spec.size_in
        fig = plt.figure(figsize=(W, H))
        cv = Canvas(fig, W, H)
        n = len(spec.panels)
        groups = list(dict.fromkeys(p.group for p in spec.panels))
        foot = _tracks_footnote(spec)
        L, R, GAP, GGAP, BOT, TOP = 1.02, 0.06, 0.07, 0.20, (0.68 if foot else 0.58), 0.46
        pw = (W - L - R - (n - len(groups)) * GAP - (len(groups) - 1) * GGAP) / n
        ph = H - BOT - TOP
        xmin, xmax = _uauc_limits(spec)
        t0 = math.ceil(round(xmin * 10, 6))
        ticks = [round(k / 10, 1) for k in range(t0, int(math.floor(round(xmax * 10, 6))) + 1)]
        x, spans, prev = L, {}, None
        axes = []
        for p in spec.panels:
            if prev is not None:
                x += GGAP if p.group != prev else GAP
            ax = cv.axes(x, BOT, pw, ph)
            axes.append(ax)
            lo_x, hi_x = spans.get(p.group, (x, x))
            spans[p.group] = (min(lo_x, x), max(hi_x, x + pw))
            prev = p.group
            _header_text(cv, x + pw / 2, H - 0.27, p.title, fontsize=7.5, color=INK)
            if p.status == "not_run":
                fig_not_run_placeholder(ax, p.title, p.missing)
            else:
                _draw_track_panel(ax, p, xmin, xmax, ticks)
                if p.n_text():
                    _header_text(cv, x + pw / 2, H - 0.38, f"{p.n_text()} users", fontsize=SMALL_PT, color=INK2)
            x += pw
        for g, (x0, x1) in spans.items():
            _header_text(cv, (x0 + x1) / 2, H - 0.09, spec.headers.get(g, g), fontsize=7.5, color=INK, weight="bold")
            cv.line(x0, H - 0.17, x1, H - 0.17, color=AXIS, lw=0.5)
        first = axes[0]
        if spec.panels[0].status != "not_run":
            first.set_yticks(list(TRACK_ROWS.values()))
            first.set_yticklabels([TRACK_STYLE[k]["label"] for k in TRACK_ROWS], fontsize=7.0, color=INK2)
            first.tick_params(axis="y", length=0, pad=3)
        else:                                                      # row labels still stand left of a placeholder panel
            for k, y in TRACK_ROWS.items():
                cv.text(L - 0.04, BOT + ph * y / 5.5, TRACK_STYLE[k]["label"], ha="right", va="center", fontsize=7.0, color=INK2)
        mid = L + (W - L - R) / 2
        cv.text(mid, BOT - 0.31, f"UAUC (axis starts at {xmin:.2f}; vertical rule: chance, 0.5)", ha="center", va="center",
                fontsize=LABEL_PT, color=INK2)
        keys = [_proxy("o", C_LORA, ms=SEED_MS + 0.6), Line2D([], [], color=C_REF, lw=LW_CI, marker="|", ms=CAP_MS, mew=0.8)]
        names = ["LoRA seeds (dots above the mean)", "95% bootstrap interval (users)"]
        if _any_hollow(spec):
            keys.append(_proxy("o", C_REF, hollow=True, ms=3.6))
            names.append("exploratory or descriptive (hollow)")
        _legend(fig, keys, names, loc="center", anchor=(mid / W, (0.27 if foot else 0.12) / H), ncol=len(keys))
        if foot:
            cv.text(mid, 0.09, foot, ha="center", va="center", fontsize=SMALL_PT, color=INK2, style="italic")
        return fig


TRACK_NAME = {"zero_shot": "zero-shot", "lora": "LoRA", "item_mean": "item mean", "matched_mean": "matched mean",
              "mf": "temporal MF"}
REASON_SHORT = {"file": "(file missing)", "block": "(not in report)", "gate": "(gate not passed)"}


def _tracks_footnote(spec: FigureSpec) -> str:
    """One line naming the files of the elements that are not run yet inside panels that have data."""
    items: dict = {}
    for p in spec.panels:
        if p.status == "not_run":
            continue
        for m in p.missing:
            if m.get("series") in TRACK_ROWS and m.get("kind") == "file" and m.get("path"):
                items.setdefault(m["series"], []).append(m["path"])
    if not items:
        return ""
    parts = [f"{TRACK_NAME[s]}: " + ", ".join(dict.fromkeys(items[s])) for s in TRACK_ROWS if s in items]
    return "not run yet – no result file for " + "; ".join(parts)


def _draw_track_panel(ax, p: Panel, xmin: float, xmax: float, ticks: list) -> None:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 5.5)
    _axes_style(ax)
    ax.axhspan(0.0, TRACK_BAND_TOP, facecolor=BAND, edgecolor="none", zorder=0)
    for y in TRACK_ROWS.values():
        ax.axhline(y, color=GRID, lw=0.3, zorder=1)
    ax.axvline(0.5, color=AXIS, lw=0.6, zorder=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt(t) for t in ticks])
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax.set_yticks([])
    for series, y in TRACK_ROWS.items():
        st = TRACK_STYLE[series]
        for v in p.by_series(series):
            if v.lo is not None:
                ax.plot([v.lo, v.hi], [y, y], color=st["color"], lw=LW_CI, marker="|", ms=CAP_MS, mew=0.8, zorder=3,
                        gid=f"{v.uid}:ci", solid_capstyle="butt")
            _marker(ax, v.est, y, st, hollow=v.hollow, gid=f"{v.uid}:pt")
    for k in range(len(FT_MODELS)):
        for v in p.by_series(f"lora_seed{k}"):
            _seed_dot(ax, v.est, TRACK_ROWS["lora"] + 0.34, C_LORA, f"{v.uid}:pt", v.hollow)
    mid = (xmin + xmax) / 2
    for m in p.missing:
        if m.get("series") in TRACK_ROWS:
            ax.text(mid, TRACK_ROWS[m["series"]], "not run yet\n" + REASON_SHORT.get(m.get("kind"), ""), ha="center", va="center",
                    fontsize=SMALL_PT, style="italic", color=INK2, linespacing=1.0, gid=f"{p.pid}:{m['series']}:not_run")


# ------------------------------------------------------------------------------------------------ F2 shares: draw
SHARE_X = {"item_prior_share": 0.0, "non_prior_share": 1.0, "e_share": 2.0}
SHARE_LABEL = {"item_prior_share": "item prior", "non_prior_share": "non-prior", "e_share": r"$e$-share$^{\,\dagger}$"}
GAIN_X = {"G": 0.0, "G_wu": 1.6}
GAIN_LABEL = {"G": "E-D (pooled)", "G_wu": r"E-W (within-user)$^{\,\dagger}$"}
DX_SHARE = {"zs": -0.19, "lora": 0.17}
DX_GAIN = {"zs": -0.30, "lora": 0.0, "ref": 0.30}
SEED_DX = 0.12


def draw_shares(spec: FigureSpec):
    """F2: top row = shares of the within-user variance of the confidence logit (item prior, non-prior, e), bottom row = the
    information gains (E-D pooled, E-W within-user) with the MF-residual reference; one column per rated panel."""
    with style():
        W, H = spec.size_in
        fig = plt.figure(figsize=(W, H))
        cv = Canvas(fig, W, H)
        n = len(spec.panels)
        L, R, GAP = 0.56, 0.05, 0.11
        pw = (W - L - R - (n - 1) * GAP) / n
        TOP1, H1, GAP12, H2, BOT2 = 0.40, 0.66, 0.26, 0.66, 0.34
        y1, y2 = H - TOP1 - H1, BOT2
        ok = [p for p in spec.panels if p.status != "not_run"]
        s_lo = [v.lo if v.lo is not None else v.est for p in ok for v in p.data() if v.series.split(".")[0] in SHARE_QUANTITIES]
        s_hi = [v.hi if v.hi is not None else v.est for p in ok for v in p.data() if v.series.split(".")[0] in SHARE_QUANTITIES]
        g_lo = [v.lo if v.lo is not None else v.est for p in ok for v in p.data() if v.series.split(".")[0] in GAIN_QUANTITIES + ("G_CF", "G_CF_wu")]
        g_hi = [v.hi if v.hi is not None else v.est for p in ok for v in p.data() if v.series.split(".")[0] in GAIN_QUANTITIES + ("G_CF", "G_CF_wu")]
        ylim1 = (min([0.0] + s_lo) - 0.04, max([1.0] + s_hi) + 0.04)
        glo, ghi = min([0.0] + g_lo), max([0.0] + g_hi)
        pad = 0.10 * (ghi - glo or 0.1)
        ylim2 = (glo - pad if glo < 0 else 0.0 - pad * 0.5, ghi + pad)
        axes_top, axes_bot = [], []
        for i, p in enumerate(spec.panels):
            x = L + i * (pw + GAP)
            _header_text(cv, x + pw / 2, H - 0.25, p.title + (f"  ({p.n_text()} users)" if p.n_text() else ""), fontsize=7.5, color=INK)
            if p.status == "not_run":
                ax = cv.axes(x, y2, pw, y1 + H1 - y2)
                fig_not_run_placeholder(ax, p.title, p.missing)
                axes_top.append(None)
                axes_bot.append(None)
                continue
            at, ab = cv.axes(x, y1, pw, H1), cv.axes(x, y2, pw, H2)
            axes_top.append(at)
            axes_bot.append(ab)
            _draw_share_top(at, p, ylim1, show_y=(i == 0))
            _draw_share_bottom(ab, p, ylim2, show_y=(i == 0))
        cv.text(0.13, y1 + H1 / 2, "share of within-user\nvariance of " + r"$\ell$", rotation=90, ha="center", va="center", fontsize=LABEL_PT,
                color=INK2, linespacing=1.1)
        cv.text(0.13, y2 + H2 / 2, r"gain $\Delta$UAUC", rotation=90, ha="center", va="center", fontsize=LABEL_PT, color=INK2)
        handles = [_proxy("o", C_ZS, ms=4.2), _proxy("D", C_LORA, ms=3.6), _proxy("o", C_LORA, ms=SEED_MS + 0.6),
                   _proxy("s", C_REF, ms=3.6)]
        labels = ["zero-shot", "LoRA (seed mean)", "LoRA seeds", r"MF residual, $\mathcal{G}_{\rm CF}$ (no LLM)"]
        if _any_hollow(spec):
            handles.append(_proxy("o", C_REF, hollow=True, ms=4.0))
            labels.append("exploratory or descriptive (hollow)")
        _legend(fig, handles, labels, loc="upper center", anchor=(0.5, 1.0), ncol=len(handles))
        if any(p.withheld for p in spec.panels):
            cv.text(L + (W - L - R) / 2, 0.07, "n/i: not interpretable (reliability of the item-prior estimate below its registered floor); "
                    "value withheld.", ha="center", va="center", fontsize=SMALL_PT, color=INK2, style="italic")
        return fig


def _xpos_label(ax, xs: dict, labels: dict, lim: tuple) -> None:
    ax.set_xlim(*lim)
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels([labels[k] for k in xs], fontsize=6.5, color=INK2)
    ax.tick_params(axis="x", length=0, pad=2.5)


def _vmark(ax, v: Val, x: float, st: dict) -> None:
    if v.lo is not None:
        ax.plot([x, x], [v.lo, v.hi], color=st["color"], lw=LW_CI, marker="_", ms=CAP_MS, mew=0.8, zorder=3, gid=f"{v.uid}:ci",
                solid_capstyle="butt")
    _marker(ax, x, v.est, st, hollow=v.hollow, gid=f"{v.uid}:pt")


def _draw_share_top(ax, p: Panel, ylim: tuple, *, show_y: bool) -> None:
    _axes_style(ax, left=show_y)
    ax.set_ylim(*ylim)
    _xpos_label(ax, SHARE_X, SHARE_LABEL, (-0.6, 2.6))
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels([_fmt(t) for t in (0.0, 0.5, 1.0)] if show_y else [])
    ax.tick_params(axis="y", length=2.0 if show_y else 0)
    ax.grid(axis="y", color=GRID, lw=0.4)
    for qid, x0 in SHARE_X.items():
        for tag in ("zs", "lora"):
            st = SHARE_STYLE[tag]
            xm = x0 + DX_SHARE[tag]
            for v in p.by_series(f"{qid}.{tag}"):
                _vmark(ax, v, xm, st)
            for k in range(len(FT_MODELS)):
                for v in p.by_series(f"{qid}.{tag}_seed{k}"):
                    _seed_dot(ax, xm + SEED_DX, v.est, C_LORA, f"{v.uid}:pt", v.hollow)
            for w in p.withheld:
                if w["series"] == f"{qid}.{tag}":
                    ax.text(xm, ylim[0] + 0.09, "n/i", ha="center", va="center", fontsize=SMALL_PT, style="italic", color=st["color"],
                            gid=f"{p.pid}:{qid}.{tag}:withheld")
        for m in p.missing:
            if m.get("series") == f"{qid}.zs" and qid not in ("item_prior_share", "non_prior_share"):
                ax.text(x0, 0.5, "not run\nyet", ha="center", va="center", fontsize=SMALL_PT, style="italic", color=INK2, linespacing=1.0,
                        gid=f"{p.pid}:{qid}:not_run")


def _draw_share_bottom(ax, p: Panel, ylim: tuple, *, show_y: bool) -> None:
    _axes_style(ax, left=show_y)
    ax.set_ylim(*ylim)
    _xpos_label(ax, GAIN_X, GAIN_LABEL, (-0.62, 2.22))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", length=2.0 if show_y else 0, labelleft=show_y)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.axhline(0.0, color=AXIS, lw=0.6, zorder=2)
    cf = {"G": "G_CF", "G_wu": "G_CF_wu"}
    for qid, x0 in GAIN_X.items():
        for tag in ("zs", "lora"):
            st = SHARE_STYLE[tag]
            xm = x0 + DX_GAIN[tag]
            for v in p.by_series(f"{qid}.{tag}"):
                _vmark(ax, v, xm, st)
            for k in range(len(FT_MODELS)):
                for v in p.by_series(f"{qid}.{tag}_seed{k}"):
                    _seed_dot(ax, xm + SEED_DX, v.est, C_LORA, f"{v.uid}:pt", v.hollow)
        for v in p.by_series(cf[qid]):
            _vmark(ax, v, x0 + DX_GAIN["ref"], SHARE_STYLE["ref"])
        absent = [m for m in p.missing if m.get("series") in (f"{qid}.zs", f"{qid}.lora", cf[qid])]
        if absent and not p.by_series(f"{qid}.zs") and qid == "G_wu":
            ax.text(x0, (ylim[0] + ylim[1]) / 2, "not run yet", ha="center", va="center", fontsize=SMALL_PT, style="italic", color=INK2,
                    gid=f"{p.pid}:{qid}:not_run")


# ------------------------------------------------------------------------------------------------ F3 serving: draw
def draw_serving(spec: FigureSpec):
    """F3: NDCG@10 among the served events against the fraction served (most confident first), one panel per next-item domain."""
    with style():
        W, H = spec.size_in
        fig = plt.figure(figsize=(W, H))
        cv = Canvas(fig, W, H)
        n = len(spec.panels)
        L, R, GAP, BOT, TOP = 0.56, 0.05, 0.12, 0.46, 0.42
        pw = (W - L - R - (n - 1) * GAP) / n
        ph = H - BOT - TOP
        his = [v.hi for p in spec.panels for v in p.data()]
        ymax = _nice_ceil(max(his) * 1.04, 0.1) if his else 1.0
        for i, p in enumerate(spec.panels):
            x = L + i * (pw + GAP)
            ax = cv.axes(x, BOT, pw, ph)
            _header_text(cv, x + pw / 2, H - 0.25, p.title, fontsize=7.5, color=INK)
            if p.status == "not_run":
                fig_not_run_placeholder(ax, p.title, p.missing)
                continue
            if p.n_text():
                _header_text(cv, x + pw / 2, H - 0.36, f"{p.n_text()} events", fontsize=SMALL_PT, color=INK2)
            _draw_serving_panel(ax, p, ymax, show_y=(i == 0))
        cv.text(0.13, BOT + ph / 2, "NDCG@10 of served events", rotation=90, ha="center", va="center", fontsize=LABEL_PT, color=INK2)
        cv.text(L + (W - L - R) / 2, 0.10, "fraction of events served (most confident first)", ha="center", va="center", fontsize=LABEL_PT,
                color=INK2)
        handles = [Line2D([], [], color=C_ZS, lw=LW_LINE, marker="o", ms=2.8), Line2D([], [], color=C_REF, lw=LW_LINE, marker="s", ms=2.6)]
        names = [r"most confident first ($p_{\max}$, zero-shot)", "random subset (fixed-seed null)"]
        if _any_hollow(spec):
            handles.append(_proxy("o", C_REF, hollow=True, ms=3.2))
            names.append("fewer than the minimum events: descriptive (hollow)")
        _legend(fig, handles, names, loc="upper center", anchor=(0.5, 1.0), ncol=len(handles))
        return fig


def _draw_serving_panel(ax, p: Panel, ymax: float, *, show_y: bool) -> None:
    _axes_style(ax, left=show_y)
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels([_fmt(t) for t in (0.2, 0.4, 0.6, 0.8, 1.0)])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="y", length=2.0 if show_y else 0, labelleft=show_y)
    ax.grid(axis="y", color=GRID, lw=0.4)
    styles = {"p_max": dict(color=C_ZS, marker="o", ms=2.8, z=4), "random": dict(color=C_REF, marker="s", ms=2.6, z=3)}
    for sig in ("random", "p_max"):
        vals = p.by_series(sig)
        if not vals:
            continue
        st = styles[sig]
        x = [v.fields["coverage"] for v in vals]
        ax.fill_between(x, [v.lo for v in vals], [v.hi for v in vals], color=st["color"], alpha=0.16, lw=0, zorder=st["z"] - 1,
                        gid=f"serving/{p.pid}/{sig}:band")
        ax.plot(x, [v.est for v in vals], color=st["color"], lw=LW_LINE, zorder=st["z"], gid=f"serving/{p.pid}/{sig}:line")
        for v in vals:
            ax.plot([v.fields["coverage"]], [v.est], ls="none", marker=st["marker"], ms=st["ms"],
                    mfc="white" if v.hollow else st["color"], mec=st["color"], mew=EDGE_W, zorder=st["z"] + 1, gid=f"{v.uid}:pt")
    for m in p.missing:
        if m.get("series") in SERVING_SIGNALS:
            ax.text(0.55, ymax * (0.7 if m["series"] == "p_max" else 0.55), f"{m['series']}: not run yet", ha="center", va="center",
                    fontsize=SMALL_PT, style="italic", color=INK2)


# ------------------------------------------------------------------------------------------------ saving and manifest
def _render_pdf(fig) -> bytes:
    buf = io.BytesIO()
    with style():
        fig.savefig(buf, format="pdf", metadata={"CreationDate": None}, bbox_inches=None)
    plt.close(fig)
    return buf.getvalue()


def _write_if_changed(path: Path, data: bytes) -> bool:
    if path.is_file() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return True


def _finish(fig, spec: FigureSpec, res: Results, out_dir, file: str) -> dict:
    data = _render_pdf(fig)
    _write_if_changed(Path(out_dir) / file, data)
    return spec.entry(res, file=file, sha1=hashlib.sha1(data).hexdigest(), nbytes=len(data))


def fig_tracks(results, out_dir) -> dict:
    """F1 'tracks': UAUC of the confidence beside its same-row references. Writes tracks.pdf, returns its manifest entry."""
    res = _as_results(results)
    return _finish(draw_tracks(spec := collect_tracks(res)), spec, res, out_dir, "tracks.pdf")


def fig_shares(results, out_dir) -> dict:
    """F2 'shares': shares of the within-user variance of the confidence logit, and the information gains. Writes shares.pdf."""
    res = _as_results(results)
    return _finish(draw_shares(spec := collect_shares(res)), spec, res, out_dir, "shares.pdf")


def fig_serving(results, out_dir) -> dict:
    """F3 'serving': NDCG@10 of serving only the most confident events against the random subset. Writes serving.pdf."""
    res = _as_results(results)
    return _finish(draw_serving(spec := collect_serving(res)), spec, res, out_dir, "serving.pdf")


def fig_decomp(results, out_dir) -> dict:
    """F4 'decomp': the hand-written TikZ schematic figures/decomp.tex. No data is read; the file is hashed into the manifest."""
    p = Path(out_dir) / "decomp.tex"
    if not p.is_file():
        raise FigureDataError(f"{p.name}: not found in {Path(out_dir).name}/ (the schematic is written by hand, not generated)")
    data = p.read_bytes()
    return {"id": "decomp", "kind": "tikz", "title": "Decomposition of the confidence logit (conceptual schematic)",
            "file": "decomp.tex", "sha1": hashlib.sha1(data).hexdigest(), "bytes": len(data), "status": "ok",
            "data": "none: conceptual schematic, no result file is read", "sources": [], "missing": [], "panels": []}


FIG_FUNCS = {"tracks": fig_tracks, "shares": fig_shares, "serving": fig_serving, "decomp": fig_decomp}


def normalise_ids(only) -> list:
    """The figure ids asked for (comma or space separated, F1..F4 accepted); order of FIG_IDS; unknown ids are an error."""
    if not only:
        return list(FIG_IDS)
    toks = [t for item in ([only] if isinstance(only, str) else only) for t in re.split(r"[,\s]+", item) if t]
    out = []
    for t in toks:
        t = FIG_ALIASES.get(t, t)
        if t not in FIG_IDS:
            raise ValueError(f"unknown figure id {t!r} (known: {', '.join(FIG_IDS)})")
        out.append(t)
    return [f for f in FIG_IDS if f in out]


def _root_label(root: Path) -> str:
    try:
        return root.relative_to(ROOT).as_posix()
    except ValueError:
        return root.name


def run(results, out_dir, only=None) -> dict:
    """Generate the figures asked for into out_dir and write figures_manifest.json (entries of figures not asked for are kept
    when their file still has the recorded sha1). Returns the manifest."""
    res = _as_results(results)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = normalise_ids(only)
    entries = {fid: FIG_FUNCS[fid](res, out) for fid in ids}
    kept: dict = {}
    mpath = out / MANIFEST_NAME
    if mpath.is_file() and len(ids) < len(FIG_IDS):
        try:
            old = json.loads(mpath.read_text(encoding="utf-8"))
            for fid, e in (old.get("figures") or {}).items():
                f = out / e.get("file", "")
                if fid in FIG_IDS and fid not in entries and f.is_file() and hashlib.sha1(f.read_bytes()).hexdigest() == e.get("sha1"):
                    kept[fid] = e
        except (ValueError, OSError, AttributeError):
            kept = {}
    figures = {fid: (entries.get(fid) or kept[fid]) for fid in FIG_IDS if fid in entries or fid in kept}
    manifest = {
        "schema": SCHEMA,
        "generator": {"script": "scripts/sigir/make_figures.py",
                      "script_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest(),
                      "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__}},
        "results_root": _root_label(res.root),
        "figures": figures,
    }
    _write_if_changed(mpath, (json.dumps(manifest, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(DEFAULT_RESULTS), help="root of the committed result files")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="directory of the figures and figures_manifest.json")
    ap.add_argument("--only", nargs="+", default=None, help="figure ids (tracks shares serving decomp; comma or space separated)")
    a = ap.parse_args(argv)
    try:
        ids = normalise_ids(a.only)
        manifest = run(a.results, a.out, ids)
    except ValueError as e:
        ap.error(str(e))
    except FigureDataError as e:
        print(f"make_figures: {e}", file=sys.stderr)
        return 2
    for fid in ids:
        e = manifest["figures"][fid]
        panels = e.get("panels") or []
        nr = [p["id"] for p in panels if p["status"] == "not_run"]
        part = [p["id"] for p in panels if p["status"] == "partial"]
        print(f"{fid:<8} {e['status']:<8} {e['file']}  sha1 {e['sha1'][:12]}  panels {len(panels)}  not run: "
              f"{', '.join(nr) or '-'}; partial: {', '.join(part) or '-'}")
    print(f"manifest: {Path(a.out) / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
