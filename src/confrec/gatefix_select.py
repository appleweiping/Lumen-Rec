"""Registered prompt-fix round of idea-stage/PREREG_AMENDMENT_2.md: stage-1 dev selection (G3-G5) and the stage-2
confirmatory gate (G6), with the G8 contingency labels F0 / F1. Executed exactly once; nothing here is tunable.

    python -m src.confrec.gatefix_select dev --root outputs/confrec/gatefix/dev --variants V0,V1,V2,V3,V4,V5,V7 \
        --out outputs/confrec/gatefix/dev/selection.json [--manifest outputs/confrec/gatefix/panels/manifest.json]
    python -m src.confrec.gatefix_select confirm --vstar_dir <confirm>/<V*> --v0_dir <confirm>/V0 \
        --selection outputs/confrec/gatefix/dev/selection.json --out outputs/confrec/gatefix/confirm/gate.json \
        [--manifest outputs/confrec/gatefix/panels/manifest.json]

Inputs are pyes_scorer output directories (scores.csv.gz + report.json) scored with `--questions like` only; dev
reads ROOT/{ml1m,toys}/{V}/ for each of the 7 registered variants (--variants must name exactly that bank).

UAUC(like) = mean over users with both classes of the per-user AUC (ties averaged) of L(like) on the rows with a finite
logit and censored not in {2, 3}: pilot_mirror's arms.raw_like UAUC, i.e. the number pilot1_gate.py gates on.

dev (G4, G5):
  E1, per dev panel: censored=2 rows <= 0.5% of the like rows; overlength = 0 (the larger of report.json n_overlength
      and the scores' censored=3 count); mean yes_no_mass over the like rows >= 0.95. A variant needs E1 on BOTH panels.
  E2: Toys dev UAUC(V) >= Toys dev UAUC(V0) - 0.010. A missing or non-finite value fails eligibility.
  V* = the eligible variant with the highest ML-1M dev UAUC; eligible variants within 0.005 of that maximum are tied
      and broken by the higher Toys dev UAUC, then by the simplicity order V0 < V1 < V5 < V3 < V4 < V2 < V7.
  FIX_FOUND iff V* != V0 and the one-sided lower bound of the paired user bootstrap of UAUC(V*) - UAUC(V0) on ML-1M
      dev is > 0 (per-user AUCs on the rows finite in both variants, 2000 resamples, seed 0, lower bound = the
      0.05/6 quantile, i.e. level 1 - 0.05/6); otherwise F0 = GATE_FAIL_AFTER_REMEDY(dev) and stage 2 is skipped.
  The full 7 x 2 table (every variant, eligible or not) is written to selection.json and printed. selection.json
  records v_star (the G5 selection, also under F0), decision, the bootstrap, gate_ft_prompt (G9: V* on FIX_FOUND,
  else V0), the sorted ML-1M DEV user ids that stage 2 checks disjointness against, and per run the identity of its
  rendering (system_sha1, prompt_spec_sha1, scorer, yes/no ids, hist_len) that stage 2 must reproduce.
confirm (G6): GATE_PASS iff UAUC(V*) on the CONFIRM ML-1M users >= 0.60 (point estimate, the pilot1_gate.py
  constant) and E1 holds for V* on that panel; otherwise F1 = GATE_FAIL_AFTER_REMEDY(confirm). Reported: the 95%
  user-bootstrap CI of UAUC(V*) (2000, seed 0) and, as context only, V0 on the same users (UAUC, CI, E1, paired dUAUC).

Input checks. A failure exits 2, prints the reasons and writes them to <out>.input_check_failed.json; it is never
recorded as F0/F1 and never touches <out>. Every run: report.json variant equals the run's variant, readout yesno,
panel_kind rated, questions == [like] and only like rows in the scores (selection is MIRROR-blind, G3), hist_len equals
the variant's registered window (hist_len_registered), zero-shot (lora null), the G0 invariants (float16, top-50
logprobs, max_model_len 4096) and one model for every run, report rows == score rows, no duplicated rows; all runs of
one panel share data_sha1 and the same (event, candidate, user, label) rows.
  dev: (G2) every run's users are the burned Pilot-1 users of its panel (user-id sha1 == PILOT1_DEV, i.e. not the
      CONFIRM panel); the registered control V0 has a finite UAUC and E1 on both dev panels (V0 is the Pilot-1
      prompt on the Pilot-1 users, which had no censoring and Yes+No mass >= 0.99999: a failure there is a broken
      scorer, not an F0). Recorded, not gating (a WARNING is printed): whether V0's rows and like prompts reproduce
      the Pilot-1 panel byte for byte (rows_sha1, v0_like_prompts_sha1) and dUAUC(V0 dev - Pilot-1 raw_like UAUC).
  confirm: selection.json decision FIX_FOUND with V* != V0, the dev model, CONFIRM users disjoint by user_id from
      the ML-1M DEV users, and each of V* / V0 rendered and read as on dev (system_sha1, prompt_spec_sha1, scorer,
      yes/no ids and hist_len equal to its dev ML-1M row: the prompt bank, scorer and tokenizer did not change
      between the stages; for V5/V7 the system message also pins the movie domain).
  --manifest (build_confirm_panels manifest.json): dev panels must be its {src}.dev split and the confirm panel its
      ml1m.confirm split (data_sha1 == split sha1, user-id sha1 == split user_ids_sha1). dev records the ml1m.confirm
      split in selection.json (confirm_panel_expected), so confirm checks it even without --manifest; when neither is
      available confirm records confirm_panel_identity.checked = false and prints a WARNING.
An existing <out> is not replaced (exit 2) unless --overwrite when its decision, V* or input fingerprint (sha1 of
every report.json and decompressed scores.csv.gz, the selection.json bytes, and the manifest's panel identity
(`manifest_identity`: per split built / panel sha1 / user-id sha1, not the file's created_utc)) differs: one
selection, one gate.
Exit codes: dev 0 = FIX_FOUND, 4 = F0; confirm 0 = GATE_PASS, 3 = F1; 2 = input check failed / refused.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from src.confrec import pilot_mirror as pm
from src.confrec.stats import paired_bootstrap, strict_json

PANELS = ("ml1m", "toys")
VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5", "V7")            # G1 bank (V6 dropped)
SIMPLICITY = ("V0", "V1", "V5", "V3", "V4", "V2", "V7")          # G5 tie-break order
CONTROL = "V0"
CENS2_MAX, MASS_MIN = 0.005, 0.95                                # G4 E1
E2_TOL, TIE_TOL = 0.010, 0.005                                   # G4 E2, G5 ties
UAUC_MIN = 0.60                                                  # scripts/sigir/pilot1_gate.py UAUC_MIN (unchanged)
N_BOOT, SEED = 2000, 0
N_COMPARISONS = len(VARIANTS) - 1                                # 6 candidate fixes vs V0
ALPHA_ONE_SIDED = 0.05 / N_COMPARISONS
EPS = 1e-12      # inclusive comparisons of derived differences ("within 0.005", ">= ... - 0.010") ignore fp noise
G0 = {"dtype": "float16", "topk_logprobs": 50, "max_model_len": 4096}
FIX_FOUND, F0 = "FIX_FOUND", "GATE_FAIL_AFTER_REMEDY(dev)"
GATE_PASS, F1 = "GATE_PASS", "GATE_FAIL_AFTER_REMEDY(confirm)"
INPUT_CHECK_FAILED = "INPUT_CHECK_FAILED"
EXIT = {FIX_FOUND: 0, F0: 4, GATE_PASS: 0, F1: 3, INPUT_CHECK_FAILED: 2}
NAN = float("nan")
# G2: DEV = the burned Pilot-1 panels outputs/confrec/panels/{ml1m,toys}_rated.jsonl (file sha1 panel_sha1).
# Derived from those files: user_ids_sha1 = _ids_sha1 of their user ids (= build_confirm_panels' user-id list format,
# i.e. the manifest freeze {src}_dev_users_sha1); rows_sha1 = _rows_sig of a like-only score file of the panel
# (panel_dev_signature reproduces both from the panel file); v0_like_prompts_sha1 = pyes_scorer prompts_sha1 of V0 like
# at hist_len 10 (the same computation over like,dislike,like_para gives the Pilot-1 report prompts_sha1 12e83c4f... /
# cd47bdb8...); raw_like_UAUC = Pilot-1 pilot_mirror arms.raw_like.UAUC (context).
PILOT1_DEV = {
    "ml1m": {"panel_sha1": "985494c7b44d010bec4b11ec62f35ad274dfe91f", "n_users": 1500, "n_rows": 29365,
             "user_ids_sha1": "b7124d7b2d000ede32ea76078a775685a3734120",
             "rows_sha1": "294075fb328da3ddf133d7be5e35415fe020bc56",
             "v0_like_prompts_sha1": "a4726659025b5e5e3504376c80ded0d3f81261a8",
             "raw_like_UAUC": 0.5874475521977115},
    "toys": {"panel_sha1": "69252b4806bb4ffe05101967ea2a1d94ada57dee", "n_users": 1500, "n_rows": 19022,
             "user_ids_sha1": "77b46ce3839d2191a3cf211ddad8e7ebfa905045",
             "rows_sha1": "ded825b26255720fc608c75eb07c02fc91febebe",
             "v0_like_prompts_sha1": "7aac249cd9ce4a8b7e8e7bac956d207baf70479b",
             "raw_like_UAUC": 0.539624516361647}}
V0_REPRO_WARN = 0.005   # |UAUC(V0 dev) - Pilot-1 raw_like UAUC| above this prints a WARNING (context, never gates)
STAGE_IDENTITY = ("system_sha1", "prompt_spec_sha1", "scorer", "yes_ids", "no_ids", "hist_len")


class InputCheckError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _sha1_file(path, gz: bool = False) -> str:
    h = hashlib.sha1()
    with (gzip.open(path, "rb") if gz else open(path, "rb")) as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _json_sha1(x):
    return None if x is None else hashlib.sha1(json.dumps(x, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- one scored run
def _like_mass(path) -> tuple[float, int]:
    """(mean yes_no_mass, n) over the like rows; NaN mean when the column is absent or a value is non-finite."""
    s, n, ok = 0.0, 0, True
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["question"] != "like":
                continue
            n += 1
            v = _num(r.get("yes_no_mass"))
            ok &= math.isfinite(v)
            s += v if math.isfinite(v) else 0.0
    return (s / n if n and ok else NAN), n


def load_run(d) -> dict:
    """A pyes_scorer output dir -> {dir, report, sc (pilot_mirror.load_scores), mass, n_mass, fingerprint}."""
    d = Path(d)
    missing = [str(p) for p in (d / "report.json", d / "scores.csv.gz") if not p.exists()]
    if missing:
        raise InputCheckError([f"missing {m}" for m in missing])
    raw = (d / "report.json").read_bytes()
    rep = json.loads(raw.decode("utf-8"))
    sc = pm.load_scores(d / "scores.csv.gz")
    mass, n_mass = _like_mass(d / "scores.csv.gz")
    return {"dir": str(d), "report": rep, "sc": sc, "mass": mass, "n_mass": n_mass,
            "fingerprint": {"report_sha1": hashlib.sha1(raw).hexdigest(),
                            "scores_sha1": _sha1_file(d / "scores.csv.gz", gz=True)}}


def report_variant(rep: dict):
    return rep.get("variant", (rep.get("config") or {}).get("variant"))


def check_run(run: dict, variant: str, label: str) -> list[str]:
    """Protocol checks of one run (module docstring); returns the failures."""
    rep, sc, err = run["report"], run["sc"], []
    if report_variant(rep) != variant:
        err.append(f"{label}: report.json variant {report_variant(rep)!r} != {variant!r}")
    if rep.get("readout", "yesno") != "yesno":
        err.append(f"{label}: readout {rep.get('readout')!r} (the gate reads the yes/no logit; G0 bans digits)")
    if rep.get("panel_kind") != "rated":
        err.append(f"{label}: panel_kind {rep.get('panel_kind')!r} != 'rated'")
    if rep.get("questions") != ["like"] or set(sc["L"]) != {"like"}:
        err.append(f"{label}: questions {rep.get('questions')!r} / score rows {sorted(sc['L'])} must be exactly "
                   "['like'] (G3: dev and confirm score only like)")
    if "hist_len_registered" not in rep or rep.get("hist_len") != rep["hist_len_registered"]:
        err.append(f"{label}: hist_len {rep.get('hist_len')!r} != registered window "
                   f"{rep.get('hist_len_registered')!r}")
    if rep.get("lora") is not None:
        err.append(f"{label}: lora {rep.get('lora')!r} (the zero-shot gate scores the base model)")
    for k, v in G0.items():
        if rep.get(k) != v:
            err.append(f"{label}: {k} {rep.get(k)!r} != {v!r} (G0)")
    cen = sc["censoring"].get("like", {})
    if cen.get("duplicate", 0):
        err.append(f"{label}: {cen['duplicate']} duplicated (row, question) entries in scores.csv.gz")
    if _num(rep.get("n_main_prompts")) != cen.get("n", 0):
        err.append(f"{label}: report.json n_main_prompts {rep.get('n_main_prompts')!r} != {cen.get('n', 0)} score "
                   "rows (stale or mixed output directory)")
    return err


def _rows_sig(sc: dict) -> str:
    """sha1 of the (event, candidate, user, label) rows: runs of one panel must score identical rows."""
    h = hashlib.sha1()
    for (ev, c), u, y in zip(sc["keys"], sc["user"], sc["label"]):
        h.update(f"{ev}\x1f{c}\x1f{u}\x1f{y}\x1e".encode("utf-8"))
    return h.hexdigest()


def _ids_sha1(ids) -> str:
    """sha1 of the distinct user ids sorted as strings and joined by '\\n' (build_confirm_panels' user-id lists)."""
    return hashlib.sha1("\n".join(sorted(set(map(str, ids)))).encode("utf-8")).hexdigest()


def panel_dev_signature(panel_path) -> dict:
    """{n_users, n_rows, user_ids_sha1, rows_sha1} of a rated panel file, equal to what _ids_sha1 / _rows_sig give on
    a like-only score file of it (pyes_scorer writes source_event_id (else user_id), user_id, cand_idx, label): the
    derivation of PILOT1_DEV from the Pilot-1 panels."""
    rows, users = [], []
    with open(panel_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            users.append(str(r["user_id"]))
            ev = str(r.get("source_event_id", r["user_id"]))
            rows += [((ev, c), str(r["user_id"]), int(y)) for c, y in enumerate(r["candidate_labels"])]
    rows.sort(key=lambda t: t[0])
    h = hashlib.sha1()
    for (ev, c), u, y in rows:
        h.update(f"{ev}\x1f{c}\x1f{u}\x1f{y}\x1e".encode("utf-8"))
    return {"n_users": len(set(users)), "n_rows": len(rows), "user_ids_sha1": _ids_sha1(users),
            "rows_sha1": h.hexdigest()}


def check_same_panel(runs: dict, label: str) -> list[str]:
    err = []
    sha = {k: r["report"].get("data_sha1") for k, r in runs.items()}
    if len(set(sha.values())) != 1 or None in sha.values():
        err.append(f"{label}: runs were scored on different panels (data_sha1 {sha})")
    sig = {k: _rows_sig(r["sc"]) for k, r in runs.items()}
    if len(set(sig.values())) != 1:
        err.append(f"{label}: runs do not score the same (event, candidate, user, label) rows ({sig})")
    return err


def check_dev_users(runs: dict) -> list[str]:
    """G2: DEV = the burned Pilot-1 users. Every dev run's user set must hash to PILOT1_DEV[panel] (a run on the
    CONFIRM panel would burn fresh users and must not feed the one registered selection)."""
    err = []
    for p in PANELS:
        exp = PILOT1_DEV[p]["user_ids_sha1"]
        bad = {v: _ids_sha1(r["sc"]["user"].tolist()) for v, r in runs[p].items()}
        bad = {v: (s, len(set(runs[p][v]["sc"]["user"].tolist()))) for v, s in bad.items() if s != exp}
        if bad:
            err.append(f"{p}: runs {sorted(bad)} are not on the burned Pilot-1 {p} users (user-id sha1 / n users "
                       f"{sorted(set(bad.values()))} != {exp} / {PILOT1_DEV[p]['n_users']}): DEV must be the "
                       "Pilot-1 panel's users (G2), e.g. not the CONFIRM panel")
    return err


def manifest_split(man: dict, src: str, split: str):
    """The {sha1, user_ids_sha1} of a built split of a build_confirm_panels manifest, or None."""
    rec = (((man or {}).get("sources") or {}).get(src) or {}).get(split) or {}
    if rec.get("built") and rec.get("sha1") and rec.get("user_ids_sha1"):
        return {"sha1": rec["sha1"], "user_ids_sha1": rec["user_ids_sha1"], "path": rec.get("path")}
    return None


def check_split(runs: dict, expected: dict, label: str) -> list[str]:
    """Runs scored on exactly the manifest split: data_sha1 == its sha1 and the user set == its user-id list."""
    err = []
    for v, r in runs.items():
        sha, ids = r["report"].get("data_sha1"), _ids_sha1(r["sc"]["user"].tolist())
        if sha != expected["sha1"] or ids != expected["user_ids_sha1"]:
            err.append(f"{label}/{v}: panel data_sha1 {sha} / user-id sha1 {ids} != the manifest split "
                       f"{expected.get('path')} ({expected['sha1']} / {expected['user_ids_sha1']})")
    return err


def _load_manifest(path):
    if path is None:
        return None, None
    p = Path(path)
    if not p.exists():
        raise InputCheckError([f"missing --manifest {p}"])
    return json.loads(p.read_text(encoding="utf-8")), _sha1_file(p)


def manifest_identity(man) -> dict | None:
    """What a build_confirm_panels manifest says about WHICH panels exist: per source and split (dev, confirm) whether it
    was built and its panel sha1 and user-id list sha1. The file's own bytes also hold created_utc and the builder's
    code hashes, which a rerun of stage 0 rewrites although the panels come out byte-identical; the input fingerprint
    (one registered selection, one gate) must follow the panels, not the timestamp, or every benign rerun of
    run_gatefix.sh after a manifest rebuild would be refused as 'other inputs'."""
    if man is None:
        return None
    return {src: {split: {"built": bool(((rec or {}).get(split) or {}).get("built")),
                          "sha1": ((rec or {}).get(split) or {}).get("sha1"),
                          "user_ids_sha1": ((rec or {}).get(split) or {}).get("user_ids_sha1")}
                  for split in ("dev", "confirm")}
            for src, rec in sorted(((man.get("sources") or {}).items()))}


def user_aucs(sc: dict, mask=None) -> dict:
    """{user: AUC of L(like)} over the rows in mask (default: finite like logit), users with both classes."""
    s = sc["L"]["like"]
    return pm._uauc(s, sc["label"], pm._groups(sc["user"]), np.isfinite(s) if mask is None else mask)


def _mean(d: dict) -> float:
    return float(np.mean(list(d.values()))) if d else NAN


def e1(run: dict) -> dict:
    """G4 E1 on one panel; unknown values fail."""
    rep, cen = run["report"], run["sc"]["censoring"].get("like", {})
    n, n2, n3 = int(cen.get("n", 0)), int(cen.get("2", 0)), int(cen.get("3", 0))
    over_rep = _num(rep.get("n_overlength"))
    over = max(n3, over_rep) if math.isfinite(over_rep) else float(n3)
    share = n2 / n if n else NAN
    out = {"n_rows": n, "n_censored2": n2, "censored2_share": share, "n_censored1": int(cen.get("1", 0)),
           "n_overlength_report": over_rep, "n_censored3_scores": n3, "overlength": over,
           "mean_yes_no_mass": run["mass"], "mean_yes_no_mass_report": _num(rep.get("mean_yes_no_mass")),
           "censored2_share_le_0.005": bool(share <= CENS2_MAX), "overlength_eq_0": bool(over == 0),
           "mean_yes_no_mass_ge_0.95": bool(run["mass"] >= MASS_MIN)}
    out["E1"] = out["censored2_share_le_0.005"] and out["overlength_eq_0"] and out["mean_yes_no_mass_ge_0.95"]
    return out


def run_identity(run: dict) -> dict:
    """How a run was rendered and read (STAGE_IDENTITY): stage 2 must reproduce stage 1's for V* and V0."""
    rep = run["report"]
    return {"system_sha1": rep.get("system_sha1"), "prompt_spec_sha1": _json_sha1(rep.get("prompt_spec")),
            "scorer": rep.get("scorer"), "yes_ids": rep.get("yes_ids"), "no_ids": rep.get("no_ids"),
            "hist_len": rep.get("hist_len")}


def panel_row(run: dict) -> dict:
    pu = user_aucs(run["sc"])
    rep = run["report"]
    return {"UAUC": _mean(pu), "n_users_uauc": len(pu), "n_users": int(len(set(run["sc"]["user"].tolist()))),
            "n_rows_scored": int(np.isfinite(run["sc"]["L"]["like"]).sum()), **e1(run),
            "prompts_sha1": (rep.get("config") or {}).get("prompts_sha1"), **run_identity(run),
            "data_sha1": rep.get("data_sha1"), "user_ids_sha1": _ids_sha1(run["sc"]["user"].tolist()),
            "rows_sha1": _rows_sig(run["sc"]), "dir": run["dir"]}


def _fingerprint(files: dict) -> dict:
    """sha1 over every input file's sha1 (report.json bytes, decompressed scores, selection / manifest)."""
    return {"sha1": hashlib.sha1(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest(), "files": files}


# ---------------------------------------------------------------- G4 / G5
def select(table: dict) -> dict:
    """Eligibility (E1 on both panels, E2) and V* with the G5 tie rule, from table[V][panel]["UAUC"/"E1"]."""
    v0_toys = table[CONTROL]["toys"]["UAUC"]
    elig, why = {}, {}
    for v, row in table.items():
        ml, ty = row["ml1m"]["UAUC"], row["toys"]["UAUC"]
        e2 = bool(math.isfinite(ty) and math.isfinite(v0_toys) and ty >= v0_toys - E2_TOL - EPS)
        reasons = [f"E1 fails on {p}" for p in PANELS if not row[p]["E1"]]
        reasons += [] if e2 else ["E2 fails (Toys dev UAUC < Toys V0 - 0.010, or undefined)"]
        reasons += [] if math.isfinite(ml) else ["ML-1M dev UAUC undefined"]
        row["E2"], row["eligible"] = e2, not reasons
        elig[v], why[v] = not reasons, reasons
    ok = [v for v in table if elig[v]]
    if not ok:
        return {"v_star": None, "eligible": [], "ineligible": why, "max_ml1m_UAUC": NAN, "tied": [],
                "rule": "no eligible variant"}
    top = max(table[v]["ml1m"]["UAUC"] for v in ok)
    tied = [v for v in ok if top - table[v]["ml1m"]["UAUC"] <= TIE_TOL + EPS]
    best_toys = max(table[v]["toys"]["UAUC"] for v in tied)
    tied_toys = [v for v in tied if table[v]["toys"]["UAUC"] == best_toys]
    v_star = min(tied_toys, key=SIMPLICITY.index)
    for v in table:
        table[v]["tied_with_max"] = v in tied
    rule = ("unique maximum" if len(tied) == 1 else "tie within 0.005 broken by Toys dev UAUC"
            if len(tied_toys) == 1 else "tie within 0.005 and equal Toys dev UAUC broken by the simplicity order")
    return {"v_star": v_star, "eligible": ok, "ineligible": {v: r for v, r in why.items() if r},
            "max_ml1m_UAUC": top, "tied": sorted(tied, key=SIMPLICITY.index),
            "tied_after_toys": sorted(tied_toys, key=SIMPLICITY.index), "rule": rule}


def fix_bootstrap(star: dict, ctrl: dict) -> dict:
    """One-sided paired user bootstrap of UAUC(V*) - UAUC(V0) on rows finite in both (same row order: checked)."""
    m = np.isfinite(star["sc"]["L"]["like"]) & np.isfinite(ctrl["sc"]["L"]["like"])
    b = paired_bootstrap(user_aucs(star["sc"], m), user_aucs(ctrl["sc"], m), n_boot=N_BOOT, seed=SEED,
                         alpha=2 * ALPHA_ONE_SIDED)   # its lo is the alpha/2 = 0.05/6 quantile
    return {"comparison": "UAUC(V*) - UAUC(V0) on ML-1M dev, per-user AUCs on rows finite in both, paired over users",
            "est": b["est"], "lower_bound": b["lo"], "level": 1 - ALPHA_ONE_SIDED, "alpha_one_sided": ALPHA_ONE_SIDED,
            "n_users": b["n"], "n_rows": int(m.sum()), "n_boot": N_BOOT, "seed": SEED,
            "lower_bound_gt_0": bool(b["lo"] > 0)}


def check_control(table: dict) -> list[str]:
    """The registered control must work on both dev panels: V0 is the Pilot-1 prompt on the Pilot-1 users (no
    censoring, Yes+No mass >= 0.99999 there), so an undefined UAUC or an E1 failure is a broken scorer or channel,
    not a finding; it must not record the final F0 (G8)."""
    err = []
    for p in PANELS:
        r = table[CONTROL][p]
        if not math.isfinite(r["UAUC"]):
            err.append(f"{p}/V0: UAUC undefined (the E2 reference)")
        if not r["E1"]:
            err.append(f"{p}/V0: E1 fails (censored=2 share {r['censored2_share']:.4f}, overlength "
                       f"{r['overlength']:g}, mean Yes+No mass {r['mean_yes_no_mass']:.4f})")
    if err:
        err.append("the registered control V0 does not reproduce Pilot 1's channel on the DEV users: fix the scorer "
                   "/ tokenizer / chat template before the one registered selection (an F0 would be final, G8)")
    return err


def pilot1_reproduction(table: dict) -> dict:
    """Context, never gating: does V0 on DEV reproduce Pilot 1 (rows, like prompts, raw-like UAUC)?"""
    out = {}
    for p in PANELS:
        r, ref = table[CONTROL][p], PILOT1_DEV[p]
        d = r["UAUC"] - ref["raw_like_UAUC"]
        x = {"rows_equal_pilot1": r["rows_sha1"] == ref["rows_sha1"],
             "v0_like_prompts_equal_pilot1": r["prompts_sha1"] == ref["v0_like_prompts_sha1"],
             "UAUC_V0_dev": r["UAUC"], "UAUC_pilot1_raw_like": ref["raw_like_UAUC"],
             "dUAUC_V0_dev_minus_pilot1": d, "warn_above": V0_REPRO_WARN}
        x["reproduces"] = bool(x["rows_equal_pilot1"] and x["v0_like_prompts_equal_pilot1"] and abs(d) <= V0_REPRO_WARN)
        out[p] = x
    return out


def table_lines(table: dict, sel: dict) -> list[str]:
    yn = {True: "yes", False: "no"}
    out = ["G5 dev table: UAUC(like); E1 per panel; E2 = Toys >= Toys(V0) - 0.010",
           f"{'variant':8}{'ML-1M':>9}{'Toys':>9}  {'E1 ml1m':8}{'E1 toys':8}{'E2':4}{'eligible':9}tied"]
    for v, r in table.items():
        out.append(f"{v:8}{r['ml1m']['UAUC']:9.4f}{r['toys']['UAUC']:9.4f}  {yn[r['ml1m']['E1']]:8}"
                   f"{yn[r['toys']['E1']]:8}{yn[r['E2']]:4}{yn[r['eligible']]:9}{yn[r.get('tied_with_max', False)]}")
    return out


def run_dev(root, variants, manifest=None) -> dict:
    root = Path(root)
    if sorted(variants) != sorted(VARIANTS) or len(variants) != len(VARIANTS):
        raise InputCheckError([f"--variants {','.join(variants)} must be exactly the registered bank "
                               f"{','.join(VARIANTS)} (G1/G5: one selection over all 7)"])
    man, _ = _load_manifest(manifest)
    runs, err = {p: {} for p in PANELS}, []
    for p in PANELS:
        for v in VARIANTS:
            try:
                runs[p][v] = load_run(root / p / v)
            except InputCheckError as e:
                err += e.errors
                continue
            err += check_run(runs[p][v], v, f"{p}/{v}")
        if len(runs[p]) == len(VARIANTS):
            err += check_same_panel(runs[p], p)
        if man is not None:
            exp = manifest_split(man, p, "dev")
            err += (check_split(runs[p], exp, f"{p}") if exp else
                    [f"{manifest}: no built {p} dev split with sha1 and user_ids_sha1"])
    err += check_dev_users(runs)
    models = {r["report"].get("model") for p in PANELS for r in runs[p].values()}
    if len(models) > 1:
        err.append(f"runs use different models {sorted(map(str, models))} (G0: one backbone)")
    if err:
        raise InputCheckError(err)
    table = {v: {p: panel_row(runs[p][v]) for p in PANELS} for v in VARIANTS}
    err = check_control(table)
    if err:
        raise InputCheckError(err)
    repro = pilot1_reproduction(table)
    sel = select(table)
    v_star = sel["v_star"]
    boot = None
    if v_star is not None and v_star != CONTROL:
        boot = fix_bootstrap(runs["ml1m"][v_star], runs["ml1m"][CONTROL])
    found = bool(boot is not None and boot["lower_bound_gt_0"])
    dev_users = sorted(set(map(str, runs["ml1m"][CONTROL]["sc"]["user"].tolist())))
    lines = table_lines(table, sel)
    for p, x in repro.items():
        if not x["reproduces"]:
            lines.append(f"WARNING {p}: V0 on DEV does not reproduce Pilot 1 (rows {x['rows_equal_pilot1']}, like "
                         f"prompts {x['v0_like_prompts_equal_pilot1']}, dUAUC(V0 dev - Pilot-1 raw_like) = "
                         f"{x['dUAUC_V0_dev_minus_pilot1']:+.4f}); recorded as context, not gating")
    if v_star is None:
        lines.append("no eligible variant -> F0")
    elif boot is None:
        lines.append(f"V* = {v_star} (the control) -> no fix -> F0")
    else:
        lines.append(f"V* = {v_star} ({sel['rule']}); one-sided lower bound (level {boot['level']:.5f}) of "
                     f"UAUC(V*) - UAUC(V0) on ML-1M dev = {boot['lower_bound']:.4f} (est {boot['est']:.4f})")
    decision = FIX_FOUND if found else F0
    lines.append(f"DECISION: {decision}" + ("" if found else "  (G8 F0: stage 2 is skipped)"))
    files = {f"{p}/{v}": runs[p][v]["fingerprint"] for p in PANELS for v in VARIANTS}
    files["manifest"] = _json_sha1(manifest_identity(man))
    return {"stage": "dev", "decision": decision, "outcome": "FIX_FOUND" if found else "F0", "fix_found": found,
            "v_star": v_star, "gate_ft_prompt": v_star if found else CONTROL,   # G9: prompt V*, or V0 under F0
            "selection": sel, "bootstrap": boot,
            "table": table, "table_text": lines, "model": models.pop(),
            "dev_user_ids": {"ml1m": dev_users},
            "dev_user_ids_sha1": {p: table[CONTROL][p]["user_ids_sha1"] for p in PANELS},
            "dev_panel_identity": {"users_are_pilot1": True, "pilot1_dev": PILOT1_DEV,
                                   "manifest": None if man is None else {"path": str(manifest),
                                                         "identity_sha1": _json_sha1(manifest_identity(man))}},
            "pilot1_reproduction": repro,
            "confirm_panel_expected": None if man is None else manifest_split(man, "ml1m", "confirm"),
            "input_fingerprint": _fingerprint(files),
            "constants": _constants(), "inputs": {"root": str(root), "variants": list(VARIANTS),
                                                  "manifest": None if manifest is None else str(manifest)},
            "rule": "PREREG_AMENDMENT_2.md G2-G5 and G8 (F0)"}


# ---------------------------------------------------------------- G6
def _uauc_ci(pu: dict) -> dict:
    v = np.array([pu[u] for u in sorted(pu)], float)
    if not len(v):
        return {"est": NAN, "lo": NAN, "hi": NAN, "n_users": 0}
    b = pm.unit_bootstrap(lambda idx: float(v[idx].mean()), len(v), n_boot=N_BOOT, seed=SEED)
    return {"est": b["est"], "lo": b["lo"], "hi": b["hi"], "n_users": len(v), "n_boot": N_BOOT, "seed": SEED,
            "level": 0.95}


def check_stage_identity(runs: dict, sel: dict, label: str) -> list[str]:
    """V* and V0 on CONFIRM are rendered and read exactly as on ML-1M dev (STAGE_IDENTITY)."""
    err = []
    for v, r in runs.items():
        dev = ((sel.get("table") or {}).get(v) or {}).get("ml1m")
        if not isinstance(dev, dict):
            err.append(f"selection.json has no dev ML-1M row for {v}")
            continue
        cur = run_identity(r)
        diff = {k: (dev.get(k), cur[k]) for k in STAGE_IDENTITY if dev.get(k) != cur[k]}
        if diff:
            err.append(f"{label}/{v}: {sorted(diff)} differ from the dev {v} run (dev, confirm) = {diff}: the prompt "
                       "bank, scorer, tokenizer or domain changed between the stages (G0/G1)")
    return err


def run_confirm(vstar_dir, v0_dir, selection_path, manifest=None) -> dict:
    sel_bytes = Path(selection_path).read_bytes()
    sel = json.loads(sel_bytes.decode("utf-8"))
    v_star = sel.get("v_star")
    if sel.get("decision") != FIX_FOUND or v_star not in VARIANTS or v_star == CONTROL:
        raise InputCheckError([f"{selection_path}: decision {sel.get('decision')!r}, v_star {v_star!r}: stage 2 "
                               "runs only after FIX_FOUND with V* != V0 (G6, G8)"])
    man, _ = _load_manifest(manifest)
    err, runs = [], {}
    for v, d in ((v_star, vstar_dir), (CONTROL, v0_dir)):
        try:
            runs[v] = load_run(d)
        except InputCheckError as e:
            err += e.errors
            continue
        err += check_run(runs[v], v, f"confirm/{v}")
    if len(runs) == 2:
        err += check_same_panel(runs, "confirm")
    err += check_stage_identity(runs, sel, "confirm")
    models = {r["report"].get("model") for r in runs.values()} | {sel.get("model")}
    if len(models) > 1:
        err.append(f"confirm runs and the dev selection use different models {sorted(map(str, models))} (G0)")
    dev_users = set(map(str, (sel.get("dev_user_ids") or {}).get("ml1m") or []))
    if not dev_users:
        err.append(f"{selection_path}: no ML-1M dev user ids recorded; CONFIRM disjointness cannot be checked")
    for v, r in runs.items():
        overlap = dev_users & set(map(str, r["sc"]["user"].tolist()))
        if overlap:
            err.append(f"confirm/{v}: {len(overlap)} CONFIRM users are DEV users (e.g. {sorted(overlap)[:3]}); "
                       "CONFIRM must be disjoint by user_id (G2)")
    if man is not None:
        expected, source = manifest_split(man, "ml1m", "confirm"), f"--manifest {manifest}"
        if expected is None:
            err.append(f"{manifest}: no built ml1m confirm split with sha1 and user_ids_sha1")
    else:
        expected, source = sel.get("confirm_panel_expected"), f"{selection_path} confirm_panel_expected"
    if expected:
        err += check_split(runs, expected, "confirm")
    if err:
        raise InputCheckError(err)
    identity = {"checked": bool(expected), "source": source if expected else None, "expected": expected}
    star, ctrl = runs[v_star], runs[CONTROL]
    pu_s, pu_0 = user_aucs(star["sc"]), user_aucs(ctrl["sc"])
    u_star = _mean(pu_s)
    e1_star = e1(star)
    passed = bool(u_star >= UAUC_MIN and e1_star["E1"])
    m = np.isfinite(star["sc"]["L"]["like"]) & np.isfinite(ctrl["sc"]["L"]["like"])
    d = paired_bootstrap(user_aucs(star["sc"], m), user_aucs(ctrl["sc"], m), n_boot=N_BOOT, seed=SEED)
    decision = GATE_PASS if passed else F1
    lines = [] if expected else ["WARNING: the CONFIRM panel identity is unchecked (no --manifest here or at dev); "
                                 "only the DEV disjointness and the dev rendering identity were checked"]
    lines += [f"CONFIRM ML-1M UAUC({v_star}) = {u_star:.4f} (registered >= {UAUC_MIN:.2f}, point estimate); E1 "
              f"{'holds' if e1_star['E1'] else 'FAILS'}",
              f"context only: UAUC(V0) = {_mean(pu_0):.4f}; dUAUC(V* - V0) = {d['est']:.4f} [{d['lo']:.4f}, "
              f"{d['hi']:.4f}]",
              f"DECISION: {decision}" + ("" if passed else "  (G8 F1: final for the zero-shot rated gate)")]
    files = {f"confirm/{v}": r["fingerprint"] for v, r in runs.items()}
    files.update(selection=hashlib.sha1(sel_bytes).hexdigest(), manifest=_json_sha1(manifest_identity(man)))
    return {"stage": "confirm", "decision": decision, "outcome": "PASS" if passed else "F1", "gate_pass": passed,
            "v_star": v_star, "UAUC": u_star, "UAUC_min": UAUC_MIN, "UAUC_ge_min": bool(u_star >= UAUC_MIN),
            "ci95": _uauc_ci(pu_s), "E1": e1_star, "n_users": len(set(star["sc"]["user"].tolist())),
            "n_users_uauc": len(pu_s), "prompts_sha1": (star["report"].get("config") or {}).get("prompts_sha1"),
            "data_sha1": star["report"].get("data_sha1"), "model": star["report"].get("model"),
            "user_ids_sha1": _ids_sha1(star["sc"]["user"].tolist()), "rendering": run_identity(star),
            "v0_context": {"note": "reported context only, never gates (G6)", "UAUC": _mean(pu_0),
                           "ci95": _uauc_ci(pu_0), "E1": e1(ctrl),
                           "dUAUC_vstar_minus_v0": {**d, "n_rows": int(m.sum()), "level": 0.95}},
            "confirm_users_disjoint_from_dev": True, "dev_user_ids_sha1": sel.get("dev_user_ids_sha1"),
            "rendering_equals_dev": True, "confirm_panel_identity": identity,
            "input_fingerprint": _fingerprint(files),
            "text": lines, "constants": _constants(),
            "inputs": {"vstar_dir": str(vstar_dir), "v0_dir": str(v0_dir), "selection": str(selection_path),
                       "manifest": None if manifest is None else str(manifest)},
            "rule": "PREREG_AMENDMENT_2.md G6 and G8 (F1)"}


def _constants() -> dict:
    return {"variants": list(VARIANTS), "simplicity_order": list(SIMPLICITY), "E1_censored2_max_share": CENS2_MAX,
            "E1_overlength": 0, "E1_mass_min": MASS_MIN, "E2_tolerance": E2_TOL, "tie_tolerance": TIE_TOL,
            "UAUC_min": UAUC_MIN, "n_boot": N_BOOT, "seed": SEED, "alpha_one_sided": ALPHA_ONE_SIDED, **G0}


# ---------------------------------------------------------------- CLI
def _write(out: Path, res: dict, overwrite: bool) -> None:
    """Write <out>; an existing <out> with another decision, V* or input fingerprint is kept unless overwrite (one
    registered round: a recorded selection / gate is never silently recomputed from other inputs)."""
    if out.exists() and not overwrite:
        try:
            old = json.loads(out.read_text(encoding="utf-8"))
        except ValueError:
            old = {}
        fp_old = (old.get("input_fingerprint") or {}).get("sha1")
        diff = [k for k, a, b in (("decision", old.get("decision"), res["decision"]),
                                  ("V*", old.get("v_star"), res["v_star"]),
                                  ("input fingerprint", fp_old, res["input_fingerprint"]["sha1"])) if a != b]
        if diff:
            raise InputCheckError([f"{out} already records decision {old.get('decision')!r} / V* "
                                   f"{old.get('v_star')!r} / input fingerprint {fp_old!r}; refusing to replace it "
                                   f"with {res['decision']!r} / V* {res['v_star']!r} / "
                                   f"{res['input_fingerprint']['sha1']!r} ({', '.join(diff)} differ; one registered "
                                   "round; pass --overwrite deliberately)"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(strict_json(res), indent=2, allow_nan=False), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    d = sub.add_parser("dev", help="stage 1: G4 eligibility + G5 selection on the DEV panels")
    d.add_argument("--root", required=True, help="ROOT/{ml1m,toys}/{V}/{scores.csv.gz,report.json}")
    d.add_argument("--variants", default=",".join(VARIANTS))
    d.add_argument("--out", required=True)
    d.add_argument("--manifest", default=None, help="build_confirm_panels manifest.json (panel identity checks)")
    d.add_argument("--overwrite", action="store_true")
    c = sub.add_parser("confirm", help="stage 2: G6 confirmatory gate on the CONFIRM ML-1M users")
    c.add_argument("--vstar_dir", required=True)
    c.add_argument("--v0_dir", required=True)
    c.add_argument("--selection", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--manifest", default=None, help="build_confirm_panels manifest.json (default: the split "
                                                    "recorded in selection.json, if dev had one)")
    c.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    side = out.with_name(out.stem + ".input_check_failed.json")
    try:
        if a.stage == "dev":
            res = run_dev(a.root, [v for v in a.variants.split(",") if v], a.manifest)
        else:
            res = run_confirm(a.vstar_dir, a.v0_dir, a.selection, a.manifest)
        _write(out, res, a.overwrite)
    except InputCheckError as e:
        side.parent.mkdir(parents=True, exist_ok=True)
        side.write_text(json.dumps({"stage": a.stage, "decision": INPUT_CHECK_FAILED, "errors": e.errors,
                                    "argv": sys.argv[1:] if argv is None else list(argv)}, indent=2),
                        encoding="utf-8")
        print("INPUT CHECK FAILED:\n  " + "\n  ".join(e.errors), file=sys.stderr)
        return EXIT[INPUT_CHECK_FAILED]
    side.unlink(missing_ok=True)
    print("\n".join(res["table_text"] if a.stage == "dev" else res["text"]))
    return EXIT[res["decision"]]


if __name__ == "__main__":
    sys.exit(main())
