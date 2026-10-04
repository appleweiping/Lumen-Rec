"""Amendment-2 STAGE 3 input check (idea-stage/PREREG_AMENDMENT_2.md G7, its 2026-10-04 scale note and the "G7 stage-3
gate source" clarification), called by scripts/sigir/run_gatefix_stage3.sh before any panel is written or scored.

    python -m src.confrec.gatefix_stage3 --gate outputs/confrec/gatefix/confirm/gate.json \
        --selection outputs/confrec/gatefix/dev/selection.json --manifest outputs/confrec/gatefix/panels/manifest.json \
        --confirm_panel outputs/confrec/gatefix/panels/ml1m_confirm_h20.jsonl \
        --toys_confirm outputs/confrec/gatefix/panels/toys_confirm_h20.jsonl \
        --pilot1_decision outputs/confrec/pilot1_mirror/decision.json --model <the G6 backbone> \
        --sports_valid outputs/baselines/external_tasks/sports_large10000_100neg_valid_same_candidate/ranking_valid.jsonl \
        [--sports_test <the sports TEST ranking file>] --out outputs/confrec/gatefix/stage3/inputs.json

G7 runs only on GATE_PASS. REFUSED (exit 2, every reason printed, nothing written) unless all of:
  gate     the G6 record (gatefix_select confirm gate.json) says GATE_PASS and the stage-3 token-channel gate of
           pilot1_gate.py holds on the records (stage3_gate, i.e. what `--stage3_gate/--pilot1_decision` will gate on):
           rated component = gate.json (stage confirm, gate_pass, UAUC >= 0.60, E1); next-item component = the Pilot-1
           decision.json records sports NDCG@10 >= 0.8 x 0.2329 with its input check ok (G0: not re-tested);
  V*       selection.json is the FIX_FOUND selection (diag_battery.selected_variant, as run_gatefix.sh's `vstar`) and
           its V* is gate.json's v_star;
  model    --model is the backbone G6 scored (gate.json model; G0: one backbone);
  ML-1M    the CONFIRM ML-1M panel holds the bytes G6 scored (sha1 == gate.json data_sha1 == the manifest's ml1m
           confirm split) and one row per G6 user (lines == gate.json n_users);
  second   the second rated domain is decidable (G7): fresh Toys iff the manifest's Toys CONFIRM split is built (G2:
           >= 2,000 eligible) with >= TOYS_MIN_FRESH = 500 fresh users, then the first N_SECOND = 1,500 rows of
           toys_confirm_h20.jsonl, whose sha1 must be the manifest's (its user ids are in the freeze record); otherwise
           Video_Games, the first 1,500 users of its seed-0 panel (built by the run script with build_rated_panels
           --source amazon --domain games --hist_len 20 --gatefix_fields --n_users 1500, the h20 rendering of G2);
  sports   the sports VALID panel exists and its first N_SPORTS = 1,000 rows are next-item rows with unique event ids,
           none flagged as another split (split_name), none among the sports TEST events 1-1000 (quarantined; checked
           when --sports_test exists).
On success --out records every input (sha1s, counts, the second-domain decision and its reason; deterministic JSON,
rewritten only when its content changes, so a rerun touches nothing) and stdout carries one line
"<V*> <toys|games> <second-domain users> <sports VALID events>" for the run script; messages go to stderr.
Recorded, not required: whether the VALID file is byte-identical to the panel C-CRP v3 read (scripts/sigir/
panel_reference.py anchor(); the next-item audit of amendment 2 section N reads VALID unverified as well).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOYS_MIN_FRESH = 500   # G7: "fresh Toys if >= 500 fresh users are eligible, otherwise Video_Games"
N_SECOND = 1500        # G7 scale note: the first 1,500 fresh Toys users (Video_Games: the first 1,500 of its panel)
N_SPORTS = 1000        # G7: the first 1,000 events of the sports VALID panel (TEST events 1-1000 are quarantined)
EXIT_REFUSED = 2
RULE = ("PREREG_AMENDMENT_2.md G7 (+ 2026-10-04 scale note, G7 stage-3 gate source clarification): stage 3 runs only "
        "on the G6 GATE_PASS; second rated domain = fresh Toys if >= 500 fresh users are eligible (first 1,500 rows of "
        "toys_confirm_h20.jsonl), otherwise Video_Games (first 1,500 users of its seed-0 panel); MIRROR's next-item "
        "no-loss check under V0 on the first 1,000 events of the sports VALID panel")


def file_sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def first_lines(path, n: int) -> list[bytes]:
    """The first n lines of a file as bytes, newline included (what `head -n n` writes)."""
    out = []
    with open(path, "rb") as f:
        for line in f:
            if len(out) == n:
                break
            out.append(line)
    return out


def count_lines(path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def load_script(name: str):
    """scripts/sigir/<name>.py of this checkout as a module (pilot1_gate, panel_reference)."""
    spec = importlib.util.spec_from_file_location(f"_stage3_{name}", ROOT / "scripts" / "sigir" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_json(path, what: str, errs: list):
    p = Path(path)
    if not p.is_file():
        errs.append(f"{what}: {path} does not exist")
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        errs.append(f"{what}: {path} is not valid JSON ({e})")
        return None
    if not isinstance(d, dict):
        errs.append(f"{what}: {path} is not a JSON object")
        return None
    return d


def check_gate(gate_path) -> tuple[dict | None, list[str]]:
    """The G6 record must exist and say GATE_PASS (the refusals worded for the two registered non-pass states)."""
    if not Path(gate_path).is_file():
        return None, [f"no G6 gate record {gate_path}: stage 3 runs only after scripts/sigir/run_gatefix.sh stage 2 "
                      "recorded GATE_PASS (amendment 2 G7)"]
    errs: list = []
    g6 = read_json(gate_path, "G6 gate record", errs)
    if g6 is None:
        return None, errs
    if g6.get("decision") != "GATE_PASS":
        return None, [f"{gate_path} records decision {g6.get('decision')!r}, not GATE_PASS: stage 3 runs only on the "
                      "G6 GATE_PASS (G7); F0 / F1 are G8 branches without a stage 3"]
    return g6, []


def recorded_gate(g6: dict, p1: dict) -> tuple[dict, list[str]]:
    """pilot1_gate.stage3_gate on the two records alone: the components the stage-3 gate will read."""
    pg = load_script("pilot1_gate")
    s3 = pg.stage3_gate(g6, p1, {"n_users": g6.get("n_users")}, {}, {})
    r, s = s3["rated_component"], s3["next_item_component"]
    errs = []
    if not r["ok"]:
        errs.append(f"gate.json says GATE_PASS but its rated component does not hold for pilot1_gate.stage3_gate "
                    f"(needs stage 'confirm', gate_pass true, UAUC >= {pg.UAUC_MIN}, E1 true; got stage "
                    f"{g6.get('stage')!r}, gate_pass {g6.get('gate_pass')!r}, UAUC {r['UAUC']}, E1 {r['E1']})")
    if not s["ok"]:
        errs.append(f"the Pilot-1 decision record does not show the sports next-item component as passed "
                    f"(sports_raw_NDCG@10 {s['sports_raw_NDCG@10']} vs >= {s['sports_raw_NDCG@10_min']:.5f}, input "
                    f"check ok {s['input_check_ok']!r}): the stage-3 gate (G7 clarification) cannot pass")
    return {"rated_component_ok": r["ok"], "next_item_component_ok": s["ok"], "G6_UAUC": r["UAUC"],
            "pilot1_sports_raw_NDCG@10": s["sports_raw_NDCG@10"]}, errs


def selected_vstar(sel: dict, sel_path, g6: dict) -> tuple[str | None, list[str]]:
    from src.confrec.diag_battery import selected_variant   # the reader run_gatefix.sh stage 2 used (`vstar`)
    try:
        v = selected_variant(sel)
    except SystemExit as e:
        return None, [f"{sel_path}: {e}"]
    if v != g6.get("v_star"):
        return v, [f"V* of {sel_path} is {v!r} but the G6 gate.json was recorded for {g6.get('v_star')!r}"]
    return v, []


def confirm_panel(path, g6: dict, man: dict) -> tuple[dict, list[str]]:
    p = Path(path)
    if not p.is_file():
        return {}, [f"no CONFIRM ML-1M panel {path} (run_gatefix.sh stage 0 builds it)"]
    sha, n = file_sha1(p), count_lines(p)
    split = ((man.get("sources") or {}).get("ml1m") or {}).get("confirm") or {}
    errs = []
    if sha != g6.get("data_sha1"):
        errs.append(f"{path} sha1 {sha} is not the panel G6 scored (gate.json data_sha1 {g6.get('data_sha1')})")
    if not (split.get("built") is True and split.get("sha1") == sha):
        errs.append(f"{path} sha1 {sha} is not the manifest's ml1m confirm split ({split.get('sha1')})")
    if n != g6.get("n_users"):
        errs.append(f"{path} holds {n} rows but G6 scored {g6.get('n_users')} CONFIRM users")
    return {"panel": p.as_posix(), "sha1": sha, "n_users": n}, errs


def second_domain(man: dict, toys_confirm) -> tuple[dict, list[str]]:
    """G7: fresh Toys if >= TOYS_MIN_FRESH fresh users are eligible, otherwise Video_Games; N_SECOND users."""
    rec = {"min_fresh": TOYS_MIN_FRESH, "n_max": N_SECOND}
    src = (man.get("sources") or {}).get("toys")
    if not isinstance(src, dict):
        return rec, ["the manifest has no toys source record: the G7 second-domain rule (fresh Toys if >= 500 fresh "
                     "users are eligible) cannot be applied"]
    conf = src.get("confirm") if isinstance(src.get("confirm"), dict) else {}
    built = conf.get("built") is True
    n_fresh = conf.get("n_users") if built else conf.get("n_fresh_eligible", src.get("fresh"))
    rec.update(toys_confirm_built=built, toys_fresh_eligible=n_fresh)
    if built and not isinstance(n_fresh, int):
        return rec, [f"the manifest's toys confirm split is built but records no user count ({n_fresh!r})"]
    if built and n_fresh >= TOYS_MIN_FRESH:
        p = Path(toys_confirm)
        if not p.is_file():
            return rec, [f"the manifest records a built toys confirm split but {toys_confirm} does not exist"]
        sha = file_sha1(p)
        errs = []
        if sha != conf.get("sha1"):
            errs.append(f"{toys_confirm} sha1 {sha} is not the manifest's toys confirm split ({conf.get('sha1')}; "
                        "frozen: its user ids are in the PILOT_LOG freeze record)")
        n_lines = count_lines(p)
        if n_lines != n_fresh:
            errs.append(f"{toys_confirm} holds {n_lines} rows, the manifest records {n_fresh}")
        n = min(N_SECOND, n_lines)
        rec.update(domain="toys", n_users=n, reason=f"{n_fresh} fresh Toys users eligible >= {TOYS_MIN_FRESH}",
                   source_panel=p.as_posix(), source_sha1=sha,
                   rows=f"the first {n} rows of {p.name} (row order = the seed-0 panel order)",
                   rows_sha1=hashlib.sha1(b"".join(first_lines(p, n))).hexdigest())
        return rec, errs
    reason = (f"{n_fresh} fresh Toys users eligible < {TOYS_MIN_FRESH}" if built else
              f"no fresh Toys users usable: the Toys CONFIRM split is not built ({conf.get('reason', 'no record')}; "
              f"{n_fresh} fresh eligible)")
    rec.update(domain="games", n_users=N_SECOND, reason=reason,
               rows=f"the first {N_SECOND} users of the seed-0 Video_Games panel (build_rated_panels --source amazon "
                    f"--domain games --hist_len 20 --gatefix_fields --n_users {N_SECOND}; fewer if fewer are eligible)")
    return rec, []


def sports_valid(valid, test) -> tuple[dict, list[str]]:
    """The first N_SPORTS rows of the VALID panel: next-item rows, unique ids, no other split, no quarantined event."""
    p = Path(valid)
    if not p.is_file():
        return {}, [f"no sports VALID panel {valid} (G7: MIRROR's next-item no-loss check runs on its first "
                    f"{N_SPORTS} events)"]
    lines = first_lines(p, N_SPORTS)
    errs, ids, splits = [], [], set()
    for k, line in enumerate(lines):
        try:
            r = json.loads(line)
        except ValueError:
            errs.append(f"{valid} line {k + 1} is not JSON")
            break
        if not isinstance(r, dict) or "positive_item_index" not in r or "candidate_item_ids" not in r:
            errs.append(f"{valid} line {k + 1} is not a next-item panel row (positive_item_index, candidate_item_ids)")
            break
        ids.append(str(r.get("source_event_id", r.get("user_id"))))
        if "split_name" in r:
            splits.add(str(r["split_name"]))
    if not lines:
        errs.append(f"{valid} is empty")
    if len(set(ids)) != len(ids):
        errs.append(f"{valid}: duplicated source_event_id among its first {len(lines)} rows")
    if splits - {"valid"}:
        errs.append(f"{valid}: rows flagged split_name {sorted(splits)}, not 'valid' (sports TEST events 1-1000 are "
                    "quarantined)")
    rec = {"source": p.as_posix(), "n_events": len(lines), "rows": f"the first {len(lines)} rows of {p.name}",
           "rows_sha1": hashlib.sha1(b"".join(lines)).hexdigest(), "quarantine_checked": False, "test_panel": None}
    t = Path(test) if test else None
    if t is not None and t.is_file():
        tids = set()
        for line in first_lines(t, N_SPORTS):
            r = json.loads(line)
            tids.add(str(r.get("source_event_id", r.get("user_id"))))
        hit = sorted(tids & set(ids))
        if hit:
            errs.append(f"{valid}: {len(hit)} of its first {len(lines)} events are sports TEST events 1-{N_SPORTS} "
                        f"(quarantined), e.g. {hit[:3]}")
        rec.update(quarantine_checked=True, test_panel=t.as_posix())
    try:
        a = load_script("panel_reference").anchor("sports", str(p))
        rec["anchor"] = {"file_sha256": a["sha256"], "original_sha256": a["orig_sha256"],
                         "byte_identical_to_original": a["byte_identical"]}
    except Exception as e:   # recorded, never required (see the module docstring)
        rec["anchor"] = {"error": f"{type(e).__name__}: {e}"}
    return rec, errs


def precheck(gate, selection, manifest, confirm, toys_confirm, pilot1_decision, model, valid, test=None):
    """(record, errors): the record is complete only when errors is empty."""
    g6, errs = check_gate(gate)
    if g6 is None:
        return None, errs
    sel = read_json(selection, "dev selection", errs)
    p1 = read_json(pilot1_decision, "Pilot-1 decision record", errs)
    man = read_json(manifest, "build_confirm_panels manifest", errs)
    rec = {"stage": "stage3", "rule": RULE,
           "constants": {"toys_min_fresh": TOYS_MIN_FRESH, "n_second": N_SECOND, "n_sports": N_SPORTS},
           "gate": {"path": Path(gate).as_posix(), "sha1": file_sha1(gate), "decision": g6.get("decision"),
                    "v_star": g6.get("v_star"), "UAUC": g6.get("UAUC"), "n_users": g6.get("n_users"),
                    "model": g6.get("model")},
           "model": model}
    if p1 is not None:
        comp, e = recorded_gate(g6, p1)
        errs += e
        rec["recorded_stage3_gate"] = comp
        rec["pilot1_decision"] = {"path": Path(pilot1_decision).as_posix(), "sha1": file_sha1(pilot1_decision)}
    if sel is not None:
        v, e = selected_vstar(sel, selection, g6)
        errs += e
        rec["v_star"] = v
        rec["selection"] = {"path": Path(selection).as_posix(), "sha1": file_sha1(selection)}
    if model != g6.get("model"):
        errs.append(f"MODEL {model} is not the backbone G6 scored ({g6.get('model')}); G0: one backbone")
    if man is not None:
        rec["ml1m"], e = confirm_panel(confirm, g6, man)
        errs += e
        rec["second"], e = second_domain(man, toys_confirm)
        errs += e
        rec["manifest"] = {"path": Path(manifest).as_posix(), "sha1": file_sha1(manifest)}
    rec["sports_valid"], e = sports_valid(valid, test)
    errs += e
    return rec, errs


def write_if_changed(path: Path, text: str) -> bool:
    """Write (atomically) only when the content differs, so a rerun leaves the file and its mtime alone."""
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--gate", required=True, help="G6 record: outputs/confrec/gatefix/confirm/gate.json")
    ap.add_argument("--selection", required=True, help="stage-1 selection.json (V*)")
    ap.add_argument("--manifest", required=True, help="build_confirm_panels manifest.json (the frozen splits)")
    ap.add_argument("--confirm_panel", required=True, help="ml1m_confirm_h20.jsonl")
    ap.add_argument("--toys_confirm", required=True, help="toys_confirm_h20.jsonl (used when fresh Toys qualifies)")
    ap.add_argument("--pilot1_decision", required=True, help="Pilot-1 decision.json (its sports next-item record)")
    ap.add_argument("--model", required=True, help="the backbone the run script scores with (must be G6's)")
    ap.add_argument("--sports_valid", required=True, help="the sports VALID ranking file")
    ap.add_argument("--sports_test", default=None, help="the sports TEST ranking file (quarantine check)")
    ap.add_argument("--out", required=True, help="input record JSON (written only when every check passes)")
    a = ap.parse_args(argv)
    rec, errs = precheck(a.gate, a.selection, a.manifest, a.confirm_panel, a.toys_confirm, a.pilot1_decision,
                         a.model, a.sports_valid, a.sports_test)
    if errs:
        print("STAGE 3 INPUT CHECK FAILED (nothing written):\n  " + "\n  ".join(errs), file=sys.stderr)
        return EXIT_REFUSED
    write_if_changed(Path(a.out), json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    s, sp = rec["second"], rec["sports_valid"]
    print(f"stage 3 inputs OK: V* = {rec['v_star']}; CONFIRM ML-1M {rec['ml1m']['n_users']} users; second rated "
          f"domain {s['domain']} ({s['reason']}): {s['rows']}; sports VALID: {sp['rows']} (quarantine checked: "
          f"{sp['quarantine_checked']}); record {a.out}", file=sys.stderr)
    print(f"{rec['v_star']} {s['domain']} {s['n_users']} {sp['n_events']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
