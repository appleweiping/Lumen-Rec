"""Freeze / verify the Lumen 1+100 same-candidate panels by fingerprint.

The original panels left with the old server, but their sha256 survive in the C-CRP v3 provenance (ORIGINAL).
A rebuilt panel is ANCHORED -- proven to carry the exact text C-CRP v3 and the 8 baselines read -- only when it is
byte-identical to the original, or when it matches every event of an anchored content ref (frozen from such a
byte-identical file). An ID match alone does not prove the prompt text; accepting an unanchored panel needs an
explicit --allow_unanchored, which the VERIFIED marker records (rebuild_panels.sh / pilot 1:
ALLOW_UNANCHORED_PANEL=1).

make   (local):  python scripts/sigir/panel_reference.py make
       reads outputs/<d>_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/tables/ranking_eval_records.csv
       writes docs/sigir/panel_refs/<d>_test.json  {source_event_id: [positive_item_id, sha1(ordered candidates)]}
make-content (server, once, after the first verified rebuild; commit the output):
       python scripts/sigir/panel_reference.py make-content --domain sports --ranking <ranking_test.jsonl>
       writes docs/sigir/panel_refs/<d>_content.json {fields, anchored, ranking_sha256, events: {source_event_id:
       sha1(CONTENT_FIELDS)}} -- what the prompts and pilot analyses read beyond the ID fingerprint. Refuses unless
       the IDs verify and the file is byte-identical to the original (--allow_unanchored: drift detection only).
verify (server): python scripts/sigir/panel_reference.py verify --domain sports \
                     --ranking outputs/baselines/external_tasks/sports_large10000_100neg_test_same_candidate/ranking_test.jsonl
       exits non-zero unless every ref event occurs once, in ref order, with no extra rows, and its positive and
       ORDERED candidates match; the content ref (if any) matches every event; metadata.json beside the file (if
       any) records BUILD_ARGS; and the panel is anchored (or --allow_unanchored). On success writes <ranking>.VERIFIED.
       Exit 2 when the only failure is that the panel is unanchored (a rebuild from the same inputs reproduces it).
check:  python scripts/sigir/panel_reference.py check --domain sports --ranking <ranking_test.jsonl>
       exit 0 iff <ranking>.VERIFIED was written for this exact file (size + mtime) against the current ref files
       and, if it records an unanchored panel, --allow_unanchored is given again.
build-args --domain d: prints "<seed> <max_history_len>" for rebuild_panels.sh.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[2]
REF_DIR = ROOT / "docs" / "sigir" / "panel_refs"
# Everything the scorer / pilot analyses read from a next-item row that the ID fingerprint does not pin down.
CONTENT_FIELDS = ("history_item_ids", "history", "candidate_titles", "candidate_texts",
                  "candidate_popularity_groups", "candidate_labels", "positive_item_index", "user_id")
MARKER_VERSION = 2
# sha256 of the original panels (ranking_<split>.jsonl, candidate_items.csv) that C-CRP v3 read, copied from
# outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_ablation_<d>/
# ccrp_internal_provenance.json ({split}_ranking_sha256, {split}_candidate_items_sha256); the test ranking hash
# is also ccrp_signal_rows_<d>/test/test_ccrp_signal_rows_provenance.json data_sha256 (sports, home, tools).
ORIGINAL = {
    "sports": {
        "test": {"ranking": "7654f9a9121cc31531924a88258b0d7dce91cc40afb731ebba2b14c0262b2381",
                 "candidate_items": "6fd502c4ff2f622168eca009fe306e01a55b80134f532da80e073fdf7aad60c0"},
        "valid": {"ranking": "2538001458e48675073bac5e1610f8ad136cde662f2453ce4c33fbddf419ed11",
                  "candidate_items": "6931da4b218ea97f2bfcb0b1bcb979d0d8f1f16c6ff70f1771469817d6e8ca6b"}},
    "toys": {
        "test": {"ranking": "8f94a9c83c987067d45400f37bf8962405c5f8cb3c602a84bb995c55fed842d3",
                 "candidate_items": "06432f71b04f419b1e39b8916720dfe8c938edacc2001ed956c042fb6c8fffa6"},
        "valid": {"ranking": "e7da6bbb48c32b11f9b03ce81591c03c9363fb7fe6a70a15c7b6eca22d865e5f",
                  "candidate_items": "c55d7b6b04f86d5137415ca06e75c74796ccb1faeee78952d14a999aad2e5737"}},
    "home": {
        "test": {"ranking": "b4c84d37cf588ca833cd76ef452941c7535b2ef437f328fc42afc06349e42f3d",
                 "candidate_items": "e13de5e2517133ef917b27143656446e7f010e397e457f4bff2736f7958c88b2"},
        "valid": {"ranking": "9118761df680b59d6bcaad4ec030cccd94e1f4278c69585d8f8bc92fd31eee40",
                  "candidate_items": "1b6528c4b5928948c832dccfc1b83b519c81de0d7087f652221f41d1fa1bed42"}},
    "tools": {
        "test": {"ranking": "576303e1730a2062dceb13edcee45c010a71c067e08484e9498906b3ec88d61b",
                 "candidate_items": "e866e9ee8b96910061b5a4c7d67aa3a4867aaa7e6575d0c02108a945724bea7b"},
        "valid": {"ranking": "6802366bb4d85c16b3daeb1a1d99b761b388cc1d0750238616d876b1b92da59d",
                  "candidate_items": "2561887f855026b32a553e703142213ac7e86bf27131bbc55bcb85f4aedfec1d"}},
}
# main_build_large_scale_same_candidate_runtime.py args of the original panels (baseline run summaries; amendment
# C4). tools differs in seed and history length only.
BUILD_ARGS = {"seed": 20260506, "max_history_len": 50, "shuffle_seed": 42, "user_limit": 10000,
              "num_negatives": 100, "min_sequence_length": 3, "selection_strategy": "random",
              "test_history_mode": "train_plus_valid"}
BUILD_ARGS_BY_DOMAIN = {"tools": {**BUILD_ARGS, "seed": 42, "max_history_len": 10}}


def build_args(domain: str) -> dict:
    return BUILD_ARGS_BY_DOMAIN.get(domain, BUILD_ARGS)


def fp(cands: list[str]) -> str:
    return hashlib.sha1("|".join(cands).encode()).hexdigest()


def content_fp(row: dict) -> str:
    blob = json.dumps([row.get(k) for k in CONTENT_FIELDS], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def marker(ranking) -> Path:
    r = Path(ranking)
    return r.with_name(r.name + ".VERIFIED")


def _split(ranking) -> str:
    m = re.fullmatch(r"ranking_(\w+)\.jsonl", Path(ranking).name)
    return m.group(1) if m else ""


def _rows(ranking):
    with open(ranking, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _content_ref(domain: str) -> dict | None:
    p = REF_DIR / f"{domain}_content.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text(encoding="utf-8"))
    if c.get("fields") != list(CONTENT_FIELDS):
        sys.exit(f"{p} was frozen over fields {c.get('fields')}, not {list(CONTENT_FIELDS)}")
    return c


def ref_hashes(domain: str, ranking) -> dict:
    """What a VERIFIED marker was checked against: the ref files' sha256 and the original's hashes."""
    c = REF_DIR / f"{domain}_content.json"
    return {"test_ref": sha256(REF_DIR / f"{domain}_test.json"), "content_ref": sha256(c) if c.exists() else None,
            "original": ORIGINAL.get(domain, {}).get(_split(ranking))}


def make(domains: list[str]) -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    for d in domains:
        src = (ROOT / "outputs" / f"{d}_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate"
               / "tables" / "ranking_eval_records.csv")
        ref = {}
        with open(src, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ref[row["source_event_id"]] = [row["positive_item_id"],
                                               fp(ast.literal_eval(row["candidate_item_ids"]))]
        (REF_DIR / f"{d}_test.json").write_text(json.dumps(ref), encoding="utf-8")
        print(d, len(ref), "events")


def compare(domain: str, ranking: str, content: bool = True) -> dict:
    """Counts of ID and (if a content ref exists) content agreement between `ranking` and the refs."""
    ref = json.loads((REF_DIR / f"{domain}_test.json").read_text(encoding="utf-8"))
    pos = {k: n for n, k in enumerate(ref)}  # ref order = the builder's write order (sorted user ids)
    cref = _content_ref(domain) if content else None
    events = cref["events"] if cref else None
    seen, bad, cbad = set(), [], []
    extra = dup = out_of_order = 0
    last = -1
    for r in _rows(ranking):
        k = r["source_event_id"]
        if k not in ref:
            extra += 1
            continue
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        out_of_order += pos[k] < last
        last = max(last, pos[k])
        if [r["positive_item_id"], fp(r["candidate_item_ids"])] != ref[k]:
            bad.append(k)
        if events is not None and k in events and content_fp(r) != events[k]:
            cbad.append(k)
    out = {"ref": len(ref), "matched": len(seen) - len(bad), "mismatched": len(bad), "missing": len(set(ref) - seen),
           "extra": extra, "duplicate": dup, "out_of_order": out_of_order, "content_checked": cref is not None}
    if cref is not None:
        out.update(content_ref=len(events), content_anchored=bool(cref.get("anchored")), content_mismatched=len(cbad),
                   content_missing=len(set(events) - seen), content_mismatch_examples=cbad[:5])
    return out


def anchor(domain: str, ranking: str) -> dict:
    """sha256 of the ranking file and of candidate_items.csv beside it vs the original panel's."""
    orig = ORIGINAL.get(domain, {}).get(_split(ranking))
    ci = Path(ranking).with_name("candidate_items.csv")
    out = {"sha256": sha256(ranking), "candidate_items_sha256": sha256(ci) if ci.exists() else None,
           "orig_sha256": orig["ranking"] if orig else None,
           "orig_candidate_items_sha256": orig["candidate_items"] if orig else None}
    out["byte_identical"] = None if orig is None else (
        out["sha256"] == orig["ranking"] and out["candidate_items_sha256"] in (None, orig["candidate_items"]))
    return out


def meta_check(domain: str, ranking: str) -> dict:
    """Builder args recorded in metadata.json beside the ranking file vs BUILD_ARGS (absent file: not checked)."""
    p = Path(ranking).with_name("metadata.json")
    if not p.exists():
        return {"metadata": None, "args_mismatch": {}}
    meta = json.loads(p.read_text(encoding="utf-8"))
    return {"metadata": str(p),
            "args_mismatch": {k: [meta.get(k), v] for k, v in build_args(domain).items() if meta.get(k) != v}}


def _ids_ok(res: dict) -> bool:
    return not (res["mismatched"] or res["missing"] or res["extra"] or res["duplicate"] or res["out_of_order"])


def _content_ok(res: dict) -> bool:
    return not res["content_checked"] or (not res["content_mismatched"] and not res["content_missing"])


def verify(domain: str, ranking: str, allow_unanchored: bool = False) -> None:
    m = marker(ranking)
    m.unlink(missing_ok=True)  # never leave a stale marker next to a file that is being re-verified
    res, a, meta = compare(domain, ranking), anchor(domain, ranking), meta_check(domain, ranking)
    by_content = res["content_checked"] and res["content_anchored"] and _content_ok(res)
    anchored = a["byte_identical"] is True or bool(by_content)
    msg = (f"{domain}: ref={res['ref']} matched={res['matched']} mismatched={res['mismatched']} "
           f"missing={res['missing']} extra={res['extra']} duplicate={res['duplicate']} "
           f"out_of_order={res['out_of_order']}")
    if res["content_checked"]:
        msg += (f" content_mismatched={res['content_mismatched']} content_missing={res['content_missing']}"
                f" (content ref {'anchored' if res['content_anchored'] else 'UNANCHORED'})")
    else:
        msg += " (no content ref)"
    print(msg)
    print(f"{domain}: sha256={a['sha256']} original={a['orig_sha256']} byte_identical={a['byte_identical']} "
          f"candidate_items={a['candidate_items_sha256']} original={a['orig_candidate_items_sha256']}")
    fails = []
    if not _ids_ok(res):
        fails.append("ID fingerprints")
    if not _content_ok(res):
        fails.append(f"content (e.g. {res['content_mismatch_examples']})")
    if meta["args_mismatch"]:
        fails.append(f"builder args in {meta['metadata']} [found, expected]: {meta['args_mismatch']}")
    if not anchored and not allow_unanchored:
        fails.append("not anchored: the file is not byte-identical to the original panel C-CRP v3 read and no "
                     "anchored content ref covers it, so its prompt text is unproven (IDs alone do not pin it). "
                     "Accept explicitly with --allow_unanchored (ALLOW_UNANCHORED_PANEL=1)")
    if fails:
        print(f"{domain}: VERIFY FAILED: " + "; ".join(fails), file=sys.stderr)
        sys.exit(2 if len(fails) == 1 and fails[0].startswith("not anchored") else 1)
    if not anchored:
        print(f"WARNING {domain}: accepted UNANCHORED (--allow_unanchored): comparisons against C-CRP v3 / baseline "
              "numbers on this panel are not proven like-for-like", file=sys.stderr)
    st = Path(ranking).stat()
    m.write_text(json.dumps({"marker_version": MARKER_VERSION, "domain": domain, "ranking": str(ranking),
                             "size": st.st_size, "mtime_ns": st.st_mtime_ns, "anchored": anchored,
                             "anchored_by": ("bytes" if a["byte_identical"] else "content_ref") if anchored else None,
                             "allow_unanchored": not anchored, "refs": ref_hashes(domain, ranking),
                             **a, **meta, **res}, indent=2), encoding="utf-8")


def check(domain: str, ranking: str, allow_unanchored: bool = False) -> bool:
    m, r = marker(ranking), Path(ranking)
    if not (m.exists() and r.exists()):
        return False
    rec = json.loads(m.read_text(encoding="utf-8"))
    st = r.stat()
    if (rec.get("marker_version") != MARKER_VERSION or rec.get("domain") != domain
            or rec.get("size") != st.st_size or rec.get("mtime_ns") != st.st_mtime_ns):
        return False
    if rec.get("refs") != ref_hashes(domain, ranking):  # a ref file was replaced or added since verification
        return False
    return bool(rec.get("anchored")) or allow_unanchored


def make_content(domain: str, ranking: str, force: bool = False, allow_unanchored: bool = False) -> None:
    out = REF_DIR / f"{domain}_content.json"
    if out.exists() and not force:
        sys.exit(f"{out} exists (frozen); pass --force to overwrite")
    res = compare(domain, ranking, content=False)
    if not _ids_ok(res):
        sys.exit(f"{domain}: ID fingerprints do not match ({res}); refusing to freeze content")
    a = anchor(domain, ranking)
    if a["byte_identical"] is not True:
        if not allow_unanchored:
            sys.exit(f"{domain}: {ranking} is not byte-identical to the original panel (sha256 {a['sha256']}, "
                     f"original {a['orig_sha256']}); its text is not proven equal to what C-CRP v3 read. Refusing "
                     "to freeze; --allow_unanchored freezes a drift-detection-only (unanchored) ref")
        print(f"WARNING {domain}: freezing an UNANCHORED content ref: it detects later drift but does not prove the "
              "text equals the original panel's", file=sys.stderr)
    ref = json.loads((REF_DIR / f"{domain}_test.json").read_text(encoding="utf-8"))
    events = {r["source_event_id"]: content_fp(r) for r in _rows(ranking) if r["source_event_id"] in ref}
    out.write_text(json.dumps({"fields": list(CONTENT_FIELDS), "anchored": a["byte_identical"] is True,
                               "ranking_sha256": a["sha256"], "orig_sha256": a["orig_sha256"], "events": events}),
                   encoding="utf-8")
    print(f"{domain}: wrote {len(events)} content fingerprints ({'anchored' if a['byte_identical'] else 'UNANCHORED'})"
          f" -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "make-content", "verify", "check", "build-args"])
    ap.add_argument("--domains", default="sports,toys,home,tools")
    ap.add_argument("--domain")
    ap.add_argument("--ranking")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow_unanchored", action="store_true",
                    help="accept / freeze a panel whose text is not proven equal to the original (recorded)")
    a = ap.parse_args()
    if a.mode == "make":
        make(a.domains.split(","))
    elif a.mode == "make-content":
        make_content(a.domain, a.ranking, a.force, a.allow_unanchored)
    elif a.mode == "verify":
        verify(a.domain, a.ranking, a.allow_unanchored)
    elif a.mode == "build-args":
        b = build_args(a.domain)
        print(b["seed"], b["max_history_len"])
    else:
        ok = check(a.domain, a.ranking, a.allow_unanchored)
        how = ""
        if ok:
            rec = json.loads(marker(a.ranking).read_text(encoding="utf-8"))
            how = f" by {rec['anchored_by']}" if rec["anchored"] else " UNANCHORED (--allow_unanchored)"
        print(f"{a.domain}: {'VERIFIED' + how if ok else 'not verified'} ({a.ranking})")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
