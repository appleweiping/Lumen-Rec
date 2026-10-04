"""Export the per-event top-10 EXPOSURE of C-CRP v3 and the 8 official baselines (local evidence) so the full-scale
next-item audit (src/confrec/nextitem_audit.py, docs/sigir/NEXTITEM_AUDIT_SPEC.md section B) can compare the head share
of their top-10 lists with the one of the zero-shot LLM, on the SAME events and candidate pools.

    python scripts/sigir/export_ref_exposure.py                   # -> docs/sigir/ref_exposure/<domain>/<method>.csv.gz
    python scripts/sigir/export_ref_exposure.py --domains sports  # one domain

Reads, per domain, the ranking_eval_records.csv of C-CRP v3 and of the official baselines (methods are discovered
exactly as in export_ref_ranks.py, whose path constants are imported). Writes, per method,
`<out>/<domain>/<method>.csv.gz` with the columns

    source_event_id, user_id, positive_rank, positive_group, top10_item_ids, top10_groups,
    n_head_pool, n_mid_pool, n_tail_pool

  * top10_item_ids = the FIRST 10 entries of `pred_ranked_item_ids`, space-joined (item ids must not contain a space);
  * top10_groups   = the popularity group (head/mid/tail) of each of them, looked up from the event's
                     `candidate_item_ids` / `candidate_popularity_groups`; space-joined, same order;
  * positive_group = group of the positive item (looked up the same way; also cross-checked against the
                     `positive_popularity_group` column);
  * n_head_pool / n_mid_pool / n_tail_pool = number of head / mid / tail items among the event's candidates.
Rows are written in the order of docs/sigir/panel_refs/<domain>_test.json, which is the order of the test panel file.
The gzip stream is written at level 9 with a zero mtime (byte-reproducible).

and `<out>/<domain>/_pool.json` with `n_pool_items` (distinct candidate items over all events of the domain's test
panel) plus the per-domain counts used to cross-check the analysis (n_events, candidates per event, pool group counts,
sha1 of the sorted pool, per-method row counts and consistency counters).

Asserts (the script stops with a non-zero exit code on any violation):
  * every event of docs/sigir/panel_refs/<domain>_test.json is present EXACTLY ONCE in every method's records
    (rows of another split are ignored as in export_ref_ranks.py; no extra events);
  * the ordered candidate list of every event equals the frozen fingerprint sha1("|".join(candidates)) of the panel ref
    (hence the candidate lists are identical across methods), the positive item equals the ref's positive item, and the
    candidate popularity-group lists are identical across methods;
  * the first 10 predicted items are distinct candidates of the event and every group is head / mid / tail.
Informational counters (not asserted, recorded in _pool.json): events whose `positive_rank` differs from the position
of the positive in `pred_ranked_item_ids`; events whose `positive_popularity_group` column differs from the lookup.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import export_ref_ranks as _rr  # noqa: E402  (method discovery constants shared with the rank export)
from panel_reference import fp as candidate_fp  # noqa: E402  (sha1 of the ORDERED candidate ids, as the panel refs)

ROOT = _rr.ROOT
DOMAINS = _rr.DOMAINS
GROUPS = ("head", "mid", "tail")
TOPK = 10
COLUMNS = ["source_event_id", "user_id", "positive_rank", "positive_group", "top10_item_ids", "top10_groups",
           "n_head_pool", "n_mid_pool", "n_tail_pool"]
MAX_TOTAL_MB = 25.0
csv.field_size_limit(1 << 30)


class ExportError(AssertionError):
    """A registered assertion of the export failed."""


def parse_pylist(s: str) -> list:
    """Python-literal list of simple quoted strings -> list. Fast path for the item-id / group columns (about 7x
    faster than ast.literal_eval); anything unusual (escapes, double quotes, embedded ', ') falls back to it."""
    s = s.strip()
    if s == "[]":
        return []
    if s[:1] == "[" and s[-1:] == "]":
        body = s[1:-1]
        if "\\" not in body and '"' not in body:
            parts = body.split(", ")
            if all(len(p) >= 2 and p[0] == "'" and p[-1] == "'" and "'" not in p[1:-1] for p in parts):
                return [p[1:-1] for p in parts]
    return ast.literal_eval(s)


def discover(domain: str, root: Path) -> dict[str, Path]:
    """{method: ranking_eval_records.csv} of C-CRP v3 and the official baselines (as export_ref_ranks.sources)."""
    root = Path(root)
    out = {"ccrp_v3": root / _rr.CCRP.format(d=domain)}
    for p in sorted((root / _rr.BASE).glob(f"{domain}_large10000_100neg_*_official_qwen3base_same_candidate")):
        method = p.name[len(f"{domain}_large10000_100neg_"):].split("_official")[0]
        f = p / "tables" / "ranking_eval_records.csv"
        if f.exists():
            out[method] = f
    return out


def _group_fp(groups: list) -> str:
    return hashlib.sha1("|".join(groups).encode()).hexdigest()


def read_method(path: Path, ref: dict, ref_groups: dict, domain: str, method: str) -> tuple[dict, dict, dict]:
    """Parse one ranking_eval_records.csv against the panel refs. Returns (rows by event, pool info, counters).
    `ref_groups` ({event: sha1 of the group list}) is filled by the first method and checked against by the others."""
    rows: dict[str, list] = {}
    pool: dict[str, str] = {}          # item -> popularity group (first seen)
    conflicts = 0
    cnt = {"n_rows_read": 0, "n_rows_other_split": 0, "rank_vs_position_mismatch": 0,
           "positive_group_column_mismatch": 0, "positive_not_in_prediction": 0, "n_candidates_hist": {}}
    first_method = not ref_groups
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cnt["n_rows_read"] += 1
            if r.get("split_name", "test") != "test":
                cnt["n_rows_other_split"] += 1
                continue
            sid = r["source_event_id"]
            where = f"{domain}/{method} event {sid}"
            if sid in rows:
                raise ExportError(f"{where}: duplicated source_event_id in {path}")
            if sid not in ref:
                raise ExportError(f"{where}: event is not in the panel refs ({path})")
            cands, grps = parse_pylist(r["candidate_item_ids"]), parse_pylist(r["candidate_popularity_groups"])
            pred = parse_pylist(r["pred_ranked_item_ids"])
            if len(cands) != len(grps):
                raise ExportError(f"{where}: {len(cands)} candidate ids but {len(grps)} popularity groups")
            if candidate_fp(cands) != ref[sid][1]:
                raise ExportError(f"{where}: ordered candidate list differs from the panel ref fingerprint")
            if str(r["positive_item_id"]) != str(ref[sid][0]):
                raise ExportError(f"{where}: positive item {r['positive_item_id']} != panel ref {ref[sid][0]}")
            bad = sorted(set(grps) - set(GROUPS))
            if bad:
                raise ExportError(f"{where}: unknown popularity group(s) {bad}")
            gfp = _group_fp(grps)
            if first_method:
                ref_groups[sid] = gfp
            elif ref_groups.get(sid) != gfp:
                raise ExportError(f"{where}: candidate popularity groups differ from the first method's")
            lookup = dict(zip(cands, grps))
            top = pred[:TOPK]
            if len(top) < TOPK or len(set(top)) != TOPK:
                raise ExportError(f"{where}: fewer than {TOPK} distinct predicted items ({len(set(top))})")
            outside = [i for i in top if i not in lookup]
            if outside:
                raise ExportError(f"{where}: top-{TOPK} items outside the candidate list: {outside[:3]}")
            if any(" " in i for i in top):
                raise ExportError(f"{where}: an item id contains a space")
            pos = str(r["positive_item_id"])
            pgrp = lookup.get(pos)
            if pgrp is None:
                raise ExportError(f"{where}: positive item is not among the candidates")
            if r.get("positive_popularity_group", pgrp) != pgrp:
                cnt["positive_group_column_mismatch"] += 1
            rank = float(r["positive_rank"])
            if pos in pred:
                cnt["rank_vs_position_mismatch"] += int(float(pred.index(pos) + 1) != rank)
            else:
                cnt["positive_not_in_prediction"] += 1
            nh, nm, nt = (sum(1 for g in grps if g == k) for k in GROUPS)
            n = len(cands)
            cnt["n_candidates_hist"][n] = cnt["n_candidates_hist"].get(n, 0) + 1
            if first_method:
                for i, g in lookup.items():
                    if pool.setdefault(i, g) != g:
                        conflicts += 1
            rows[sid] = [sid, r["user_id"], f"{rank:g}", pgrp, " ".join(top), " ".join(lookup[i] for i in top),
                         nh, nm, nt]
    missing = [e for e in ref if e not in rows]
    if missing:
        raise ExportError(f"{domain}/{method}: {len(missing)} panel-ref event(s) missing from {path}, "
                          f"e.g. {missing[:3]}")
    if len(rows) != len(ref):
        raise ExportError(f"{domain}/{method}: {len(rows)} events but {len(ref)} in the panel refs")
    cnt["n_pool_group_conflicts"] = conflicts
    return rows, pool, cnt


def write_rows(path: Path, rows: dict, order: list) -> int:
    """Write `rows` in `order` as a level-9, zero-mtime (byte-reproducible) gzip CSV. Returns the file size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as txt:
                w = csv.writer(txt, lineterminator="\n")
                w.writerow(COLUMNS)
                for sid in order:
                    w.writerow(rows[sid])
    return path.stat().st_size


def export_domain(domain: str, root: Path, out_root: Path, verbose: bool = True) -> dict:
    """Export every method of one domain; returns the _pool.json content (also written)."""
    root, out_root = Path(root), Path(out_root)
    ref = json.loads((root / f"docs/sigir/panel_refs/{domain}_test.json").read_text(encoding="utf-8"))
    if not ref:
        raise ExportError(f"{domain}: empty panel ref")
    order = list(ref)
    srcs = discover(domain, root)
    missing_src = [m for m, f in srcs.items() if not f.exists()]
    if missing_src:
        raise ExportError(f"{domain}: missing ranking_eval_records.csv for {missing_src}")
    ref_groups: dict[str, str] = {}
    pool: dict[str, str] = {}
    methods: dict[str, dict] = {}
    for method, f in srcs.items():
        rows, p, cnt = read_method(f, ref, ref_groups, domain, method)
        if not pool:
            pool = p
            pool_conflicts = cnt["n_pool_group_conflicts"]
        out = out_root / domain / f"{method}.csv.gz"
        size = write_rows(out, rows, order)
        methods[method] = {"n_events": len(rows), "bytes": size, "source": str(f), **{
            k: v for k, v in cnt.items() if k != "n_pool_group_conflicts"}}
        if verbose:
            print(f"{domain:6s} {method:18s} rows={len(rows)} bytes={size} "
                  f"rank_vs_position_mismatch={cnt['rank_vs_position_mismatch']} "
                  f"positive_group_column_mismatch={cnt['positive_group_column_mismatch']}", flush=True)
    n_hist = next(iter(methods.values()))["n_candidates_hist"]
    pool_info = {
        "domain": domain, "n_pool_items": len(pool), "n_events": len(ref),
        "n_candidates_per_event_hist": {str(k): v for k, v in sorted(n_hist.items())},
        "n_pool_items_by_group": {g: sum(1 for v in pool.values() if v == g) for g in GROUPS},
        "n_pool_items_with_conflicting_group": pool_conflicts,
        "pool_items_sha1": hashlib.sha1("\n".join(sorted(pool)).encode()).hexdigest(),
        "panel_refs": f"docs/sigir/panel_refs/{domain}_test.json", "topk": TOPK, "columns": COLUMNS,
        "methods": methods}
    (out_root / domain).mkdir(parents=True, exist_ok=True)
    (out_root / domain / "_pool.json").write_text(json.dumps(pool_info, indent=2), encoding="utf-8")
    return pool_info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--root", default=str(ROOT), help="repository root (inputs under outputs/, docs/sigir/panel_refs)")
    ap.add_argument("--out", default=None, help="output root (default <root>/docs/sigir/ref_exposure)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    out_root = Path(a.out) if a.out else root / "docs/sigir/ref_exposure"
    try:
        for d in [x for x in a.domains.split(",") if x]:
            info = export_domain(d, root, out_root)
            print(f"{d}: n_pool_items={info['n_pool_items']} pool_group_conflicts="
                  f"{info['n_pool_items_with_conflicting_group']} candidates/event={info['n_candidates_per_event_hist']}",
                  flush=True)
    except ExportError as e:
        print(f"EXPORT ASSERTION FAILED: {e}", file=sys.stderr)
        return 1
    total = sum(p.stat().st_size for p in out_root.glob("*/*.csv.gz"))     # every domain already exported, not only this run
    print(f"total size of {out_root} method files: {total} bytes = {total / 1e6:.2f} MB")
    if total / 1e6 > MAX_TOTAL_MB:
        # gzip level 9 is already used and every spec column is required, so this is a warning, not a failure
        print(f"WARNING: total exceeds ~{MAX_TOTAL_MB:.0f} MB (level-9 gzip, no spec column can be dropped)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
