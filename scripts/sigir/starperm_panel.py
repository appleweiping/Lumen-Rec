"""Star-permuted history panels of the Amendment-3 decomposition arms (idea-stage/PREREG_AMENDMENT_3.md section 3, E-D
"Star permutation": K = 2 permutations of the rating suffixes among the same history items), one panel per copy.

    python scripts/sigir/starperm_panel.py --panel outputs/confrec/ftgrid/panels/toys/eval_sd_test.jsonl --variant V3 \
        --out_prefix outputs/confrec/ftgrid/panels/toys/eval_sd_test_starperm [--meta outputs/confrec/ftgrid/build/toys/starperm.json]

writes <out_prefix>0.jsonl and <out_prefix>1.jsonl. The permutation is the diagnosis battery's own builder,
`diag_battery.make_starperm` (imported, unchanged): copy k of a row is a uniformly random derangement of the positions of
the rendered history window drawn from random.Random("<seed>:<k>:<source_event_id>"), redrawn while its displayed ratings
equal an earlier copy's whenever another sequence exists; titles stay with their items, history_ratings follow the
suffixes, source_event_id becomes "<source_event_id>::perm<k>" and perm_of / perm_k / perm_sigma / perm_window /
perm_n_changed record the draw (a reader pairs a copy with its original row by perm_of, or by user_id and cand_idx).
The window is the prompt variant's registered history window on a rated panel (V0 10, V3 / V7 20), i.e. exactly the
history pyes_scorer renders for --variant without --hist_len, so every rendered rating suffix is permuted. Copy k of every
row goes to file k, in the panel's row order. --meta writes the builder's summary with the sha1 of the input and of both
outputs (strict JSON, no wall-clock, no path).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.confrec import diag_battery as db  # noqa: E402
from src.confrec.prompting import GATE_VARIANTS, panel_kind_of, resolve_hist_len  # noqa: E402
from src.confrec.stats import strict_json  # noqa: E402

K, SEED = db.STARPERM_K, db.STARPERM_SEED   # the registered K = 2 and the battery's seed 0


def window(rows: list, variant: str) -> int:
    """The history window `variant` renders on these (rated) rows: its registered window."""
    kinds = sorted({panel_kind_of(r) for r in rows})
    if kinds != ["rated"]:
        raise SystemExit(f"star permutation needs rated-panel rows (history lines '<title> (rated r/5)'), got {kinds}")
    return resolve_hist_len(variant, "rated")


def build(rows: list, variant: str, k: int = K, seed: int = SEED) -> tuple[list[list[dict]], dict]:
    """([copy-0 rows, copy-1 rows, ...] in row order, the builder's summary)."""
    hist_len = window(rows, variant)
    out, info = db.make_starperm(rows, k=k, seed=seed, hist_len=hist_len)
    copies = [[r for r in out if r["perm_k"] == kk] for kk in range(k)]
    for kk, cp in enumerate(copies):
        if [r["perm_of"] for r in cp] != [str(r.get("source_event_id", r["user_id"])) for r in rows]:
            raise AssertionError(f"copy {kk} is not one row per panel row in panel order")
    return copies, {**info, "variant": variant}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True, help="the S_d TEST-row panel (eval_sd_test.jsonl)")
    ap.add_argument("--variant", required=True, choices=list(GATE_VARIANTS), help="selection.json gate_ft_prompt")
    ap.add_argument("--out_prefix", required=True, help="writes <out_prefix><k>.jsonl for k = 0 .. K-1")
    ap.add_argument("--meta", default=None, help="optional summary json")
    ap.add_argument("--k", type=int, default=K, help="copies (registered K = 2)")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    if a.k < 1:
        ap.error("--k must be >= 1")
    rows = db.read_jsonl(a.panel)
    if not rows:
        raise SystemExit(f"no rows in {a.panel}")
    copies, info = build(rows, a.variant, a.k, a.seed)
    paths = [Path(f"{a.out_prefix}{kk}.jsonl") for kk in range(a.k)]
    for path, cp in zip(paths, copies):     # the last copy is written last: run_ftgrid.sh's completion product
        db.write_jsonl(path, cp)
    meta = strict_json({"builder": "src.confrec.diag_battery.make_starperm", "panel": Path(a.panel).name,
                        "panel_sha1": db.file_sha1(a.panel), "n_rows": len(rows), "k": a.k, "seed": a.seed,
                        "outputs": {p.name: db.file_sha1(p) for p in paths}, **info})
    if a.meta:
        Path(a.meta).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(a.meta).with_name(Path(a.meta).name + ".tmp")
        tmp.write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(a.meta)
    print(json.dumps(meta, allow_nan=False))
    return meta


if __name__ == "__main__":
    main()
