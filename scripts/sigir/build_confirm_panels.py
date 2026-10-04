"""Amendment 2 G2: the DEV / CONFIRM rated panels (hist_len 20) of the gate-fix round, plus CPU artifact check A1.

    python scripts/sigir/build_confirm_panels.py --raw data/raw --ml1m_raw data/raw/ml-1m \
        --pilot_dir outputs/confrec/panels --out_dir outputs/confrec/gatefix/panels

idea-stage/PREREG_AMENDMENT_2.md G2 and A1. For each rated source (ml1m, toys), with the Pilot-1 builder arguments
(seed 0, n_cands 20, min_hist 3, min_like 3, min_dislike 3) and n_users above every eligible user:
  1. build the full panel at hist_len 10 in the Pilot-1 format into a temp file; the sha1 of its first 1500 lines
     must equal the Pilot-1 data_sha1 (ml1m 985494c7..., toys 69252b48...). On a mismatch the split falls back to
     user_id exclusion against the Pilot-1 panel file (--pilot_dir/{src}_rated.jsonl) and the mismatch is logged;
  2. build the same at hist_len 20 and assert that every row, cut back to its last 10 history events and written in
     the Pilot-1 format, is byte-identical to its hist_len 10 line (same users, candidates, candidate order, labels);
  3. DEV = the rows whose user_id is in the Pilot-1 panel, CONFIRM = the rest, both in panel order. Asserted: the
     two are disjoint and DEV users == Pilot-1 users; when the prefix matched, also DEV = the first 1500 rows and DEV
     cut back to hist_len 10 reproduces the Pilot-1 panel bytes (positions 1500 onward are the fresh users);
  4. write {src}_{dev,confirm}_h20.jsonl (with history_meta and domain_kind) and the user-id lists
     {src}_{dev,confirm}_users.txt: user ids sorted as Python strings, joined by "\\n", no trailing newline, UTF-8;
     the manifest's user_ids_sha1 is the sha1 of exactly those bytes (= `sha1sum` of the file).
Toys CONFIRM is written only if Toys has >= 2000 eligible users; Toys DEV always (stage-1 dev scoring needs it).
The eligible-user counts of the Video_Games and Sports rated panels are recorded as well (A1); no panel is built for
them (G2 keeps those users untouched), and a missing raw file is recorded as missing, not fatal.

manifest.json (strict JSON, written last) holds per source the prefix check, split method, mismatch log, per-split
counts, panel sha1s and user-id list sha1s, plus the eligible counts and a `freeze` block with the four user-id list
sha1s for docs/sigir/PILOT_LOG.md. The eligible and fresh counts are checked against the Pilot-1 sidecar
{pilot_dir}/{src}_rated.meta.json (eligible_users 3183 / 14202) and the registered 1,683 fresh ML-1M users; a
difference is recorded and logged, not fatal (G2: a mismatch is logged). Every output is staged as a temp file first;
if an existing manifest records a different sha1 for a panel or user-id list (e.g. after the PILOT_LOG freeze),
nothing is replaced and the run fails unless --force. Bad arguments and missing inputs (unknown source or count
domain, a missing Pilot-1 panel or source raw file) fail before any raw file is loaded; on any failure the staged
temp files are removed (only the full hist_len 10 temp panel of a failed h20/h10 consistency check is kept, for
inspection). Memory: one source is loaded at a time; the count-only domains keep no item text.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.confrec import build_rated_panels as brp  # noqa: E402
from src.confrec.stats import strict_json  # noqa: E402

PILOT_SHA1 = {"ml1m": "985494c7b44d010bec4b11ec62f35ad274dfe91f",   # Pilot-1 report.json data_sha1
              "toys": "69252b4806bb4ffe05101967ea2a1d94ada57dee"}
PREFIX_LINES = 1500                  # the Pilot-1 panels hold 1500 users each
FRESH_EXPECTED = {"ml1m": 1683}      # registered: 3183 eligible - 1500 (amendment 2 G2; judge.md artifact checks)
TOYS_CONFIRM_MIN = 2000              # G2: fresh Toys users only if Toys has >= 2000 eligible users
BASE_HIST, LONG_HIST = 10, 20
ALL_USERS = 10 ** 9                  # n_users above every eligible user
BUILD_KW = dict(n_cands=20, min_hist=3, min_like=3, min_dislike=3, seed=0)   # the Pilot-1 builder arguments
USER_IDS_FORMAT = ("user ids sorted as Python strings, joined by '\\n', no trailing newline, UTF-8; "
                   "user_ids_sha1 = sha1 of exactly these bytes = sha1 of the file")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def file_sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prefix_sha1(path, n: int) -> tuple[str, int]:
    """(sha1 of the bytes of the first n lines, newline included; number of lines read)."""
    h, k = hashlib.sha1(), 0
    with open(path, "rb") as f:
        for line in f:
            if k == n:
                break
            h.update(line)
            k += 1
    return h.hexdigest(), k


def write_lines(path: Path, lines) -> tuple[str, int]:
    """Write str lines as UTF-8 bytes (no newline translation on any OS); (sha1 of the bytes, number of lines)."""
    h, n = hashlib.sha1(), 0
    with open(path, "wb") as f:
        for line in lines:
            b = line.encode("utf-8")
            f.write(b)
            h.update(b)
            n += 1
    return h.hexdigest(), n


def user_list_bytes(ids) -> bytes:
    return "\n".join(sorted(ids)).encode("utf-8")


def read_pilot(path: Path) -> tuple[list, dict]:
    """(user ids in file order, {user_id: line bytes}) of a Pilot-1 panel; duplicate user ids are fatal."""
    ids, lines = [], {}
    with open(path, "rb") as f:
        for line in f:
            uid = json.loads(line)["user_id"]
            if uid in lines:
                raise SystemExit(f"{path}: user_id {uid!r} occurs twice")
            ids.append(uid)
            lines[uid] = line
    return ids, lines


def read_pilot_meta(pilot_path: Path) -> dict:
    """The Pilot-1 builder sidecar <panel stem>.meta.json: its eligible / selected user counts (None if absent)."""
    p = pilot_path.with_suffix(".meta.json")
    out = {"path": p.as_posix(), "eligible_users": None, "users": None}
    if not p.is_file():
        out["error"] = "missing"
        return out
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
        out.update(eligible_users=m.get("eligible_users"), users=m.get("users"))
    except (OSError, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def h10_line(row: dict) -> bytes:
    """The row cut back to its last BASE_HIST history events, as a Pilot-1 format panel line."""
    return brp.row_line(brp.pilot_row(brp.truncate_history(row, BASE_HIST))).encode("utf-8")


def panel_stats(rows: list) -> dict:
    n_c = sum(len(r["candidate_labels"]) for r in rows)
    pos = sum(sum(r["candidate_labels"]) for r in rows)
    hl = [len(r["history"]) for r in rows]
    n_h = sum(hl)
    return {"n_users": len(rows), "n_candidates": n_c, "like_rate": pos / n_c if n_c else None,
            "mean_history_len": n_h / len(rows) if rows else None, "max_history_len": max(hl, default=0),
            f"n_users_history_gt_{BASE_HIST}": sum(h > BASE_HIST for h in hl),
            "history_meta_nonempty_share": (sum(1 for r in rows for m in r["history_meta"] if m) / n_h
                                            if n_h else None)}


def stage_split(rows: list, panel_path: Path, ids_path: Path, staged: dict) -> dict:
    """Write the panel and its user-id list to temp files (committed later); their manifest record."""
    tmp = panel_path.with_name(panel_path.name + ".tmp")
    sha, n = write_lines(tmp, (brp.row_line(r) for r in rows))
    staged[panel_path] = tmp
    ids_b = user_list_bytes(r["user_id"] for r in rows)
    tmp_ids = ids_path.with_name(ids_path.name + ".tmp")
    tmp_ids.write_bytes(ids_b)
    staged[ids_path] = tmp_ids
    return {"built": True, "path": panel_path.as_posix(), "sha1": sha, "n_lines": n,
            "user_ids_path": ids_path.as_posix(), "user_ids_sha1": sha1_bytes(ids_b), **panel_stats(rows)}


def build_source(name: str, items, events, pilot_path: Path, out_dir: Path, staged: dict, *,
                 confirm_min: int = 0, keep_tmp: bool = False, pilot: tuple | None = None) -> dict:
    """Steps 1-4 of the module docstring for one rated source; returns its manifest record. `pilot` is
    read_pilot(pilot_path) when the caller has already read it."""
    expected = PILOT_SHA1[name]
    pilot_ids, pilot_lines = pilot if pilot is not None else read_pilot(pilot_path)
    pilot_set = set(pilot_ids)
    rec = {"domain_kind": brp.DOMAIN_KIND.get(name, "product"), "pilot_panel": pilot_path.as_posix(),
           "pilot_panel_sha1": file_sha1(pilot_path), "pilot_users": len(pilot_ids), "registered_sha1": expected}
    rec["pilot_panel_sha1_ok"] = rec["pilot_panel_sha1"] == expected
    if not rec["pilot_panel_sha1_ok"]:
        log(f"WARNING {name}: {pilot_path} sha1 {rec['pilot_panel_sha1']} != registered {expected}; "
            "its user ids are still the DEV reference")

    # 1. hist_len 10, Pilot-1 format, every eligible user -> temp file -> prefix sha1
    stats: dict = {}
    rows, n_elig = brp.build(items, events, n_users=ALL_USERS, hist_len=BASE_HIST, source=name, stats=stats,
                             **BUILD_KW)
    tmp10 = out_dir / f".{name}_full_h{BASE_HIST}.pilot_format.jsonl.tmp"
    keep10 = keep_tmp  # also set when the h20/h10 check fails: that temp panel is the evidence
    try:
        rec["full_h10_sha1"], n10 = write_lines(tmp10, (brp.row_line(brp.pilot_row(r)) for r in rows))
        del rows
        gc.collect()
        rec.update(eligible_users=n_elig, n_dedup_removed=stats["n_dedup_removed"], prefix_lines=PREFIX_LINES)
        # A1: the eligible count the Pilot-1 builder recorded next to its panel (3183 / 14202); not a gate
        rec["pilot_meta"] = read_pilot_meta(pilot_path)
        exp_elig = rec["pilot_meta"]["eligible_users"]
        rec["eligible_users_expected"] = exp_elig
        rec["eligible_matches_pilot_meta"] = None if exp_elig is None else n_elig == exp_elig
        if exp_elig is None:
            log(f"WARNING {name}: no eligible_users in {rec['pilot_meta']['path']} "
                f"({rec['pilot_meta'].get('error')}); eligible count not cross-checked")
        elif n_elig != exp_elig:
            log(f"WARNING {name}: {n_elig} eligible users, the Pilot-1 meta.json records {exp_elig}")
        rec["prefix_sha1"], n_pre = prefix_sha1(tmp10, PREFIX_LINES)
        match = n_pre == PREFIX_LINES and rec["prefix_sha1"] == expected
        rec["prefix_match"] = match
        rec["split_method"] = "prefix" if match else "user_id_exclusion"
        rec["mismatch"] = None
        if not match:
            first_diff = None
            with open(tmp10, "rb") as f, open(pilot_path, "rb") as g:
                for k, (a, b) in enumerate(zip(f, g)):
                    if a != b:
                        first_diff = k
                        break
            rec["mismatch"] = {"expected": expected, "got": rec["prefix_sha1"], "prefix_lines_read": n_pre,
                               "first_line_differing_from_pilot_file": first_diff,
                               "action": "DEV/CONFIRM selected by user_id exclusion against the Pilot-1 panel file"}
            log(f"WARNING {name}: prefix sha1 {rec['prefix_sha1']} != {expected} ({n_pre} lines): "
                "falling back to user_id exclusion")
        log(f"{name}: {n_elig} eligible users; first {PREFIX_LINES} lines sha1 {rec['prefix_sha1']} "
            f"({'MATCH' if match else 'MISMATCH'})")

        # 2. hist_len 20: identical rows apart from the longer history
        rows, n_elig20 = brp.build(items, events, n_users=ALL_USERS, hist_len=LONG_HIST, source=name, **BUILD_KW)
        if n_elig20 != n_elig or len(rows) != n10:
            raise SystemExit(f"{name}: hist_len {LONG_HIST} build has {n_elig20} eligible / {len(rows)} rows, "
                             f"hist_len {BASE_HIST} {n_elig} / {n10}")
        with open(tmp10, "rb") as f:
            bad = [k for k, (r, line) in enumerate(zip(rows, f)) if h10_line(r) != line]
        if bad:
            keep10 = True
            raise SystemExit(f"{name}: {len(bad)} hist_len {LONG_HIST} rows differ from the hist_len {BASE_HIST} "
                             f"build beyond the longer history (first: row {bad[0]}); temp panel kept: {tmp10}")
        rec["h20_rows_truncate_to_h10"] = True
    finally:
        if not keep10:
            tmp10.unlink(missing_ok=True)

    # 3. DEV / CONFIRM split
    ids = [r["user_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{name}: the rebuilt panel has duplicate user ids")
    dev = [r for r in rows if r["user_id"] in pilot_set]
    confirm = [r for r in rows if r["user_id"] not in pilot_set]
    dev_ids, conf_ids = {r["user_id"] for r in dev}, {r["user_id"] for r in confirm}
    if dev_ids & conf_ids:
        raise SystemExit(f"{name}: DEV and CONFIRM share {len(dev_ids & conf_ids)} users")
    if dev_ids != pilot_set:
        missing = sorted(pilot_set - dev_ids)
        raise SystemExit(f"{name}: DEV users != Pilot-1 users: {len(missing)} Pilot-1 users are not eligible in the "
                         f"rebuilt panel (e.g. {missing[:5]})")
    h = hashlib.sha1()
    for r in dev:
        h.update(h10_line(r))
    rec["dev_h10_sha1"] = h.hexdigest()
    rec["dev_reproduces_pilot_panel"] = rec["dev_h10_sha1"] == expected
    rec["dev_rows_equal_to_pilot"] = sum(
        brp.pilot_row(brp.truncate_history(r, BASE_HIST)) == brp.pilot_row(json.loads(pilot_lines[r["user_id"]]))
        for r in dev)
    if match:
        if ids[:PREFIX_LINES] != pilot_ids:
            raise SystemExit(f"{name}: prefix sha1 matched but the first {PREFIX_LINES} user ids are not the "
                             "Pilot-1 panel's, in order")
        if not rec["dev_reproduces_pilot_panel"] or rec["dev_rows_equal_to_pilot"] != len(dev):
            raise SystemExit(f"{name}: DEV cut back to hist_len {BASE_HIST} does not reproduce the Pilot-1 panel")
    else:
        log(f"WARNING {name}: {rec['dev_rows_equal_to_pilot']}/{len(dev)} DEV rows equal their Pilot-1 rows "
            f"(cut back to hist_len {BASE_HIST})")
    # fresh users expected: the registered 1683 (ml1m), else Pilot-1 meta eligible_users - Pilot-1 users (toys)
    if name in FRESH_EXPECTED:
        fresh_exp, fresh_src = FRESH_EXPECTED[name], "registered (amendment 2 G2)"
    elif rec["eligible_users_expected"] is not None:
        fresh_exp = rec["eligible_users_expected"] - len(pilot_ids)
        fresh_src = "Pilot-1 meta.json eligible_users - Pilot-1 panel users"
    else:
        fresh_exp, fresh_src = None, None
    rec.update(fresh=len(confirm), fresh_expected=fresh_exp, fresh_expected_source=fresh_src,
               fresh_matches_expected=None if fresh_exp is None else len(confirm) == fresh_exp)
    if fresh_exp is not None and len(confirm) != fresh_exp:
        log(f"WARNING {name}: {len(confirm)} fresh users, expected {fresh_exp} ({fresh_src})")

    # 4. stage the outputs
    rec["dev"] = stage_split(dev, out_dir / f"{name}_dev_h{LONG_HIST}.jsonl", out_dir / f"{name}_dev_users.txt",
                             staged)
    if n_elig >= confirm_min:
        rec["confirm"] = stage_split(confirm, out_dir / f"{name}_confirm_h{LONG_HIST}.jsonl",
                                     out_dir / f"{name}_confirm_users.txt", staged)
    else:
        rec["confirm"] = {"built": False, "n_fresh_eligible": len(confirm),
                          "reason": f"{n_elig} eligible users < {confirm_min} (amendment 2 G2)"}
        log(f"{name}: CONFIRM not built ({rec['confirm']['reason']})")
    log(f"{name}: DEV {len(dev)} users, CONFIRM {len(confirm)} users "
        f"({'written' if rec['confirm']['built'] else 'not written'})")
    return rec


def count_domain(raw: Path, domain: str) -> dict:
    """Eligible users of an Amazon rated panel with the Pilot-1 builder arguments (A1); no panel is built."""
    paths = brp.amazon_paths(raw, domain)
    missing = [p.as_posix() for p in paths if not p.exists()]
    if missing:
        log(f"WARNING {domain}: missing {missing}: eligible count not recorded "
            f"(python scripts/sigir/slim_amazon2023.py --domains {domain})")
        return {"status": "missing_raw", "eligible_users": None, "missing": missing}
    items, events = brp.load_amazon(raw, domain, light=True)
    kw = {k: BUILD_KW[k] for k in ("n_cands", "min_hist", "min_like", "min_dislike")}
    out = {"status": "ok", "eligible_users": brp.eligible_users(events, hist_len=BASE_HIST, **kw),
           "n_items_with_title": len(items), "n_reviewers": len(events),
           "n_reviews": sum(len(v) for v in events.values())}
    del items, events
    gc.collect()
    log(f"{domain}: {out['eligible_users']} eligible users")
    return out


def frozen_conflicts(old: dict, new: dict) -> list:
    """Panel / user-id list sha1s that an existing manifest records differently."""
    out = []
    for name, rec in new.get("sources", {}).items():
        prev = (old.get("sources") or {}).get(name) or {}
        for split in ("dev", "confirm"):
            for key in ("sha1", "user_ids_sha1"):
                a, b = (prev.get(split) or {}).get(key), (rec.get(split) or {}).get(key)
                if a is not None and a != b:
                    out.append(f"{name}.{split}.{key}: {a} -> {b}")
    return out


def source_raw_paths(name: str, raw: Path, ml1m_raw: Path) -> list:
    """The raw files a rated source is built from."""
    if name == "ml1m":
        return [ml1m_raw / "ratings.dat", ml1m_raw / "movies.dat"]
    return list(brp.amazon_paths(raw, name))


def preflight(sources: list, count_domains: list, pilot_dir: Path, raw: Path, ml1m_raw: Path) -> dict:
    """Every check that needs no raw loading, before the minutes-long builds: known sources and count domains, the
    Pilot-1 panels (read and parsed here: {source: read_pilot(...)}) and the sources' raw files. A count domain's
    missing raw file is not checked here: it is recorded as missing, not fatal (A1)."""
    unknown = [s for s in sources if s not in PILOT_SHA1]
    if unknown:
        raise SystemExit(f"unknown source(s) {unknown}; known: {sorted(PILOT_SHA1)}")
    unknown = [d for d in count_domains if d not in brp.CATEGORY]
    if unknown:
        raise SystemExit(f"unknown count domain(s) {unknown}; known: {sorted(brp.CATEGORY)}")
    missing = [p.as_posix() for s in sources
               for p in [pilot_dir / f"{s}_rated.jsonl", *source_raw_paths(s, raw, ml1m_raw)] if not p.is_file()]
    if missing:
        raise SystemExit(f"missing input file(s), nothing built: {missing}")
    return {s: read_pilot(pilot_dir / f"{s}_rated.jsonl") for s in sources}


def run(args) -> dict:
    out_dir, pilot_dir, raw, ml1m_raw = Path(args.out_dir), Path(args.pilot_dir), Path(args.raw), Path(args.ml1m_raw)
    sources = [s for s in args.sources.split(",") if s]
    count_domains = [d for d in args.count_domains.split(",") if d]
    pilots = preflight(sources, count_domains, pilot_dir, raw, ml1m_raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    staged: dict = {}
    try:
        return _run(args, sources, count_domains, pilots, out_dir, pilot_dir, raw, ml1m_raw, staged)
    finally:
        for tmp in staged.values():  # staged but not committed (an error, or a refused overwrite)
            tmp.unlink(missing_ok=True)


def _run(args, sources, count_domains, pilots, out_dir, pilot_dir, raw, ml1m_raw, staged) -> dict:
    manifest = {
        "amendment": "idea-stage/PREREG_AMENDMENT_2.md G2 (DEV/CONFIRM users) and A1 (prefix / eligible counts)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha1": {"build_confirm_panels.py": file_sha1(__file__), "build_rated_panels.py": file_sha1(brp.__file__)},
        "builder_args": {**BUILD_KW, "n_users": ALL_USERS, "hist_len_prefix_check": BASE_HIST,
                         "hist_len_panels": LONG_HIST, "prefix_lines": PREFIX_LINES},
        "user_ids_format": USER_IDS_FORMAT, "toys_confirm_min_eligible": TOYS_CONFIRM_MIN,
        "sources": {}, "eligible_counts": {}, "count_details": {}, "replaced": [],
        "carried_over_from_previous_manifest": []}
    for name in sources:
        log(f"{name}: loading")
        if name == "ml1m":
            items, events = brp.load_ml1m(ml1m_raw)
            raw_paths = [ml1m_raw.as_posix()]
        else:
            items, events = brp.load_amazon(raw, name)
            raw_paths = [p.as_posix() for p in brp.amazon_paths(raw, name)]
        rec = build_source(name, items, events, pilot_dir / f"{name}_rated.jsonl", out_dir, staged,
                           confirm_min=TOYS_CONFIRM_MIN if name == "toys" else 0, keep_tmp=args.keep_tmp,
                           pilot=pilots.pop(name))
        manifest["sources"][name] = {"raw": raw_paths, **rec}
        manifest["eligible_counts"][name] = rec["eligible_users"]
        del items, events
        gc.collect()
    for d in count_domains:
        manifest["count_details"][d] = count_domain(raw, d)
        manifest["eligible_counts"][d] = manifest["count_details"][d]["eligible_users"]

    man_path = out_dir / "manifest.json"
    if man_path.exists():
        old = json.loads(man_path.read_text(encoding="utf-8"))
        conflicts = frozen_conflicts(old, manifest)
        if conflicts and not args.force:  # run() removes the staged temp files
            raise SystemExit(f"{man_path} records different outputs (frozen in PILOT_LOG?); nothing replaced; "
                             "rerun with --force to overwrite:\n  " + "\n  ".join(conflicts))
        manifest["replaced"] = conflicts
        # a partial run (--sources / --count_domains) keeps the records of what it did not rebuild
        for key in ("sources", "count_details"):
            for n, r in (old.get(key) or {}).items():
                if n not in manifest[key]:
                    manifest[key][n] = r
                    manifest["eligible_counts"][n] = (old.get("eligible_counts") or {}).get(n)
                    manifest["carried_over_from_previous_manifest"].append(n)
    manifest["freeze"] = {f"{n}_{s}_users_sha1": (r.get(s) or {}).get("user_ids_sha1")
                          for n, r in manifest["sources"].items() for s in ("dev", "confirm")}
    for final, tmp in staged.items():
        tmp.replace(final)
    for name in sources:  # a CONFIRM panel this run did not build must not linger
        if not manifest["sources"][name]["confirm"]["built"]:
            for p in (out_dir / f"{name}_confirm_h{LONG_HIST}.jsonl", out_dir / f"{name}_confirm_users.txt"):
                p.unlink(missing_ok=True)
    tmp = man_path.with_name(man_path.name + ".tmp")
    tmp.write_text(json.dumps(strict_json(manifest), indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(man_path)
    log(f"wrote {man_path}")
    print(json.dumps(strict_json({"eligible_counts": manifest["eligible_counts"], "freeze": manifest["freeze"],
                                  "prefix_match": {n: r["prefix_match"] for n, r in manifest["sources"].items()}}),
                     indent=2))
    return manifest


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--raw", default="data/raw", help="Amazon root: <raw>/amazon_<domain>/{,meta_}<Cat>.jsonl.gz")
    ap.add_argument("--ml1m_raw", default="data/raw/ml-1m")
    ap.add_argument("--pilot_dir", default="outputs/confrec/panels", help="holds the Pilot-1 {src}_rated.jsonl")
    ap.add_argument("--out_dir", default="outputs/confrec/gatefix/panels")
    ap.add_argument("--sources", default="ml1m,toys", help="rated sources to split into DEV / CONFIRM")
    ap.add_argument("--count_domains", default="games,sports", help="Amazon domains to count only ('' = none)")
    ap.add_argument("--keep_tmp", action="store_true", help="keep the full hist_len 10 Pilot-1-format temp panels")
    ap.add_argument("--force", action="store_true", help="replace outputs an existing manifest records differently")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
