"""Amendment 3 freeze helper (idea-stage/PREREG_AMENDMENT_3.md section 0).

The amendment lists its bound code files in blocks
    <!-- FREEZE_FILES core --> ... one repo-relative path per line ... <!-- /FREEZE_FILES -->
(stages `core`, `prune`, `method`). A GPU job of the amendment may start only when `docs/sigir/PILOT_LOG.md` records the sha1
of the amendment itself (stage `amendment`); scoring of any adapter or zero-shot panel additionally needs every file of the
`core` list and every `ftgrid_split.json` recorded (stage `core`); S6 needs `prune`, the method slot needs `method`.

    python -m src.confrec.ftgrid_freeze --print --stage core       # the lines to paste into PILOT_LOG
    python -m src.confrec.ftgrid_freeze --check --stage amendment  # exit 0 iff every required sha1 is in the pilot log

A check is a case-insensitive substring test of the sha1 in the log, as the Amendment-2 freeze check does. Files listed in a
stage must exist (a listed file that is missing is an error, not a skip). Split files default to
`<ftgrid_dir>/panels/<domain>/ftgrid_split.json` for the four domains and are required from stage `core` on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

AMENDMENT = "idea-stage/PREREG_AMENDMENT_3.md"
DOMAINS = ("ml1m", "toys", "games", "sports")
STAGES = ("amendment", "core", "prune", "method")
_BLOCK = re.compile(r"<!--\s*FREEZE_FILES\s+(\w+)\s*-->(.*?)<!--\s*/FREEZE_FILES\s*-->", re.S)


def file_sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_blocks(text: str) -> dict:
    """{stage: [repo-relative path, ...]} from the FREEZE_FILES blocks of the amendment text."""
    out: dict = {}
    for stage, body in _BLOCK.findall(text):
        if stage not in STAGES or stage == "amendment":
            raise SystemExit(f"unknown FREEZE_FILES stage {stage!r} (use core, prune, method)")
        if stage in out:
            raise SystemExit(f"FREEZE_FILES stage {stage!r} appears twice")
        out[stage] = [ln.strip() for ln in body.splitlines() if ln.strip()]
    return out


def required(stage: str, root: Path, splits=None, ftgrid_dir: str = "outputs/confrec/ftgrid") -> list:
    """[(label, path)] whose sha1 the pilot log must hold for `stage` (raises SystemExit when a file is missing)."""
    if stage not in STAGES:
        raise SystemExit(f"stage must be one of {STAGES}")
    am = root / AMENDMENT
    if not am.exists():
        raise SystemExit(f"{am} does not exist")
    items = [(AMENDMENT, am)]
    # dated addenda of the amendment (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_*.md) are part of it, for every stage
    addenda = sorted((root / "idea-stage").glob("PREREG_AMENDMENT_3_ADDENDUM_*.md"))
    items += [(p.relative_to(root).as_posix(), p) for p in addenda]
    if stage != "amendment":
        blocks: dict = {}                      # the amendment's FREEZE_FILES blocks, extended by those of its addenda
        for src in [am] + addenda:
            for st, files in parse_blocks(src.read_text(encoding="utf-8")).items():
                cur = blocks.setdefault(st, [])
                cur += [f for f in files if f not in cur]
        if stage not in blocks:
            raise SystemExit(f"{AMENDMENT} has no FREEZE_FILES block for stage {stage!r}")
        items += [(rel, root / rel) for rel in blocks[stage] if rel != AMENDMENT]    # the amendment is listed once
        # files the stage needs besides code: the four splits (core), the prune manifest (prune: signals, subsets and pruned
        # train files by sha1), per-dataset q-hat manifests given with --split (method); `splits` replaces the default list
        extra = {"core": [str(Path(ftgrid_dir) / "panels" / d / "ftgrid_split.json") for d in DOMAINS],
                 "prune": ["outputs/confrec/ftprune/prune_manifest.json"], "method": []}[stage]
        items += [(rel, root / rel) for rel in (splits or extra)]
    missing = [rel for rel, p in items if not p.exists()]
    if missing:
        raise SystemExit(f"freeze stage {stage}: these required files do not exist yet: {missing}")
    if stage == "core":
        bad = split_problems([(rel, p) for rel, p in items if rel.endswith("ftgrid_split.json")])
        if bad:
            raise SystemExit("freeze stage core: a split file cannot be frozen as it is: " + "; ".join(bad))
    return items


def split_problems(splits) -> list:
    """Content a frozen ftgrid_split.json must have: the TRAIN length audit measured with a tokenizer (stage 0 of
    run_ftgrid.sh passes --tokenizer), and for ml1m a T that was checked against Gate-FT's."""
    out = []
    for rel, p in splits:
        try:
            js = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            out.append(f"{rel}: unreadable ({e})")
            continue
        if ((js.get("train") or {}).get("overlength") or {}).get("share_above_1024") is None:
            out.append(f"{rel}: the TRAIN length audit has no tokenizer measurement (rebuild with --tokenizer)")
        if js.get("domain") == "ml1m" and js.get("gateft_T_match") is not True:
            out.append(f"{rel}: ml1m T was not checked against Gate-FT's (rebuild with --gateft_split)")
    return out


def lines_for(stage: str, root: Path, splits=None, ftgrid_dir: str = "outputs/confrec/ftgrid") -> list:
    return [f"{rel} = {file_sha1(p)}" for rel, p in required(stage, root, splits, ftgrid_dir)]


def check(stage: str, root: Path, pilot_log: Path, splits=None, ftgrid_dir: str = "outputs/confrec/ftgrid") -> list:
    """Returns the list of labels whose sha1 is NOT in the pilot log (empty = frozen)."""
    if not pilot_log.exists():
        raise SystemExit(f"pilot log {pilot_log} does not exist")
    log = pilot_log.read_text(encoding="utf-8").lower()
    return [rel for rel, p in required(stage, root, splits, ftgrid_dir) if file_sha1(p).lower() not in log]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print", dest="do_print", action="store_true")
    mode.add_argument("--check", dest="do_check", action="store_true")
    ap.add_argument("--stage", choices=STAGES, default="core")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--pilot_log", default=None, help="default <root>/docs/sigir/PILOT_LOG.md")
    ap.add_argument("--ftgrid_dir", default="outputs/confrec/ftgrid")
    ap.add_argument("--split", action="append", default=None, help="ftgrid_split.json paths (default: the four domains)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.do_print:
        print("\n".join(lines_for(a.stage, root, a.split, a.ftgrid_dir)))
        return 0
    log = Path(a.pilot_log) if a.pilot_log else root / "docs" / "sigir" / "PILOT_LOG.md"
    missing = check(a.stage, root, log, a.split, a.ftgrid_dir)
    if missing:
        print(f"freeze rule (Amendment 3 section 0, stage {a.stage}): the sha1 of these files is not in {log}: {missing}; "
              "run `python -m src.confrec.ftgrid_freeze --print --stage " + a.stage + "`, record the lines in the pilot log "
              "and push it", file=sys.stderr)
        return 1
    print(f"freeze check OK (stage {a.stage}): every required sha1 is in {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
