"""V0 byte identity of the CURRENT prompting / pyes_scorer code on the real Pilot-1 panels (amendment 2 G1: V0 must
reproduce the Pilot-1 prompts_sha1 12e83c4fcca40db398ac58acbba2a7eb9c770524 on ML-1M).

For each panel the stored <name>.report.json holds the Pilot-1 `config` (questions, hist_len, swap_k, prompts_sha1,
swap_prompts_sha1). The CURRENT pyes_scorer.run is called in --dry_run mode with exactly those arguments (the real code
path: record_requests -> RowRenderer -> ChatPrompt -> prompts_sha1) and its config is compared key by key. Also: the
amendment-2 h20 panel format (history extended to 20 events, history_meta, domain_kind) scored with V0 gives the same
prompts as the 10-entry panel, and the hard-coded identity constants of gatefix_select.PILOT1_DEV /
build_confirm_panels.PILOT_SHA1 / run_diag_battery.sh agree with the real panel files.

    PILOT_PANELS=<dir with {ml1m_rated,toys_rated,sports_next_1k}.jsonl and their .report.json> \
        python scripts/sigir/dryrun/v0_identity.py          (about 2 minutes, CPU only, writes only <tmp>/v0_identity)
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
HERE = Path(tempfile.gettempdir()) / "v0_identity"
HERE.mkdir(exist_ok=True)
PANELS = Path(os.environ["PILOT_PANELS"]) if os.environ.get("PILOT_PANELS") else None

from src.confrec import gatefix_select as gs  # noqa: E402
from src.confrec import prompting  # noqa: E402
from src.confrec import pyes_scorer as ps  # noqa: E402

KEYS = ("data_sha1", "n_records", "n_users", "model", "lora", "questions", "hist_len", "swap_k", "seed", "dtype",
        "topk_logprobs", "max_model_len", "chunk_users", "chunk_items", "scorer", "prompts_sha1",
        "swap_prompts_sha1")


def t() -> str:
    return time.strftime("%H:%M:%S")


def file_sha1(p) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dry(panel: Path, stored: dict, *, questions=None, hist_len="stored", swap_k="stored") -> dict:
    cfg = stored["config"]
    argv = ["--data", str(panel), "--output", str(HERE / "unused_dry_run_dir"), "--model", stored["model"],
            "--questions", ",".join(questions or cfg["questions"]), "--dtype", cfg["dtype"],
            "--topk_logprobs", str(cfg["topk_logprobs"]), "--max_model_len", str(cfg["max_model_len"]),
            "--chunk_users", str(cfg["chunk_users"]), "--chunk_items", str(cfg["chunk_items"]), "--dry_run"]
    hl = cfg["hist_len"] if hist_len == "stored" else hist_len
    if hl is not None:
        argv += ["--hist_len", str(hl)]
    sk = cfg["swap_k"] if swap_k == "stored" else swap_k
    argv += ["--swap_k", str(sk)]
    with contextlib.redirect_stdout(io.StringIO()):
        return ps.run(ps.parse_args(argv))


def main() -> int:
    if PANELS is None or not (PANELS / "ml1m_rated.jsonl").exists():
        sys.exit("set PILOT_PANELS to the directory holding the Pilot-1 panels and their report.json files")
    print(f"[{t()}] V0 byte identity, current code vs stored Pilot-1 report.json (prompts_sha1)")
    print("python", sys.version.split()[0], "| prompting.PROMPT_STRINGS_SHA1", prompting.PROMPT_STRINGS_SHA1)
    ok_all = True
    for name in ("ml1m_rated", "toys_rated", "sports_next_1k"):
        panel = PANELS / f"{name}.jsonl"
        stored = json.loads((PANELS / f"{name}.report.json").read_text(encoding="utf-8"))
        scfg = stored["config"]
        t0 = time.time()
        out = dry(panel, stored)
        ccfg = out["config"]
        print(f"\n== {name}: file {panel.name} sha1 {file_sha1(panel)} (stored data_sha1 {stored['data_sha1']})")
        print(f"   stored args: questions {scfg['questions']} hist_len {scfg['hist_len']} swap_k {scfg['swap_k']} "
              f"max_model_len {scfg['max_model_len']} scorer {scfg['scorer']}")
        print(f"   current: variant {out['variant']} readout {out['readout']} panel_kind {out['panel_kind']} "
              f"hist_len {ccfg['hist_len']} (registered {out['hist_len_registered']}) system_sha1 {out['system_sha1']} "
              f"unregistered {out['unregistered']}   [{time.time() - t0:.0f}s]")
        for k in KEYS:
            same = scfg.get(k) == ccfg.get(k)
            if k in ("prompts_sha1", "swap_prompts_sha1", "data_sha1", "hist_len", "questions"):
                print(f"   {k:20} stored {scfg.get(k)}  current {ccfg.get(k)}  {'IDENTICAL' if same else 'MISMATCH'}")
            elif not same:
                print(f"   {k:20} stored {scfg.get(k)!r}  current {ccfg.get(k)!r}  MISMATCH")
            ok_all &= same
        sys.stdout.flush()

    print(f"\n[{t()}] h20 format (a real h20 row differs from its h10 row only when the h10 history is full)")
    for name in ("ml1m_rated", "toys_rated"):
        panel = PANELS / f"{name}.jsonl"
        stored = json.loads((PANELS / f"{name}.report.json").read_text(encoding="utf-8"))
        h20 = HERE / f"{name}_h20_synth.jsonl"
        with open(panel, encoding="utf-8") as f, open(h20, "w", encoding="utf-8", newline="\n") as g:
            for line in f:
                r = json.loads(line)
                ext = 10 if len(r["history"]) == 10 else 0
                r["history"] = [f"Older item {k} (rated {1 + k % 5}/5)" for k in range(ext)] + r["history"]
                r["history_item_ids"] = [f"OLD{k}" for k in range(ext)] + r["history_item_ids"]
                r["history_titles"] = [f"Older item {k}" for k in range(ext)] + r["history_titles"]
                r["history_ratings"] = [float(1 + k % 5) for k in range(ext)] + r["history_ratings"]
                r["history_brands"] = [""] * ext + r["history_brands"]
                r["history_meta"] = ["Genres: Drama"] * len(r["history"])
                r["domain_kind"] = "movie" if name.startswith("ml1m") else "product"
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
        for label, hl in (("default window", None), ("--hist_len 10", 10)):
            out = dry(h20, stored, swap_k=0, hist_len=hl)
            ref = dry(panel, stored, swap_k=0, hist_len=hl)
            same = out["config"]["prompts_sha1"] == ref["config"]["prompts_sha1"]
            ok_all &= same
            print(f"   {name} h20 {label}: V0 prompts_sha1 {out['config']['prompts_sha1']} vs 10-entry panel "
                  f"{ref['config']['prompts_sha1']}  {'IDENTICAL' if same else 'MISMATCH'}")
        sys.stdout.flush()

    print(f"\n[{t()}] hard-coded constants vs the real panel files")
    shas = {"ml1m": file_sha1(PANELS / "ml1m_rated.jsonl"), "toys": file_sha1(PANELS / "toys_rated.jsonl")}
    diag = dict(re.findall(r"^(ML_SHA1|TOYS_SHA1)=([0-9a-f]{40})",
                           (REPO / "scripts/sigir/run_diag_battery.sh").read_text(encoding="utf-8"), re.M))
    from importlib import util
    spec = util.spec_from_file_location("bcp", REPO / "scripts/sigir/build_confirm_panels.py")
    bcp = util.module_from_spec(spec)
    spec.loader.exec_module(bcp)
    for p in ("ml1m", "toys"):
        exp = gs.PILOT1_DEV[p]
        panel = PANELS / f"{p}_rated.jsonl"
        sig = gs.panel_dev_signature(panel)
        records = [json.loads(line) for line in open(panel, encoding="utf-8") if line.strip()]
        v0_like = ps.prompts_sha1(pr for rec in records for *_, pr in ps.record_requests(rec, ["like"], 10))
        rows = {
            "gatefix_select.PILOT1_DEV.panel_sha1": (exp["panel_sha1"], shas[p]),
            "build_confirm_panels.PILOT_SHA1": (bcp.PILOT_SHA1[p], shas[p]),
            "run_diag_battery.sh " + ("ML_SHA1" if p == "ml1m" else "TOYS_SHA1"):
                (diag["ML_SHA1" if p == "ml1m" else "TOYS_SHA1"], shas[p]),
            "PILOT1_DEV.n_users": (exp["n_users"], sig["n_users"]),
            "PILOT1_DEV.n_rows": (exp["n_rows"], sig["n_rows"]),
            "PILOT1_DEV.user_ids_sha1": (exp["user_ids_sha1"], sig["user_ids_sha1"]),
            "PILOT1_DEV.rows_sha1": (exp["rows_sha1"], sig["rows_sha1"]),
            "PILOT1_DEV.v0_like_prompts_sha1": (exp["v0_like_prompts_sha1"], v0_like),
        }
        for k, (a, b) in rows.items():
            ok_all &= a == b
            print(f"   {p:4} {k:42} {a} {'==' if a == b else '!='} {b}")
    print(f"\n[{t()}] OVERALL: {'ALL IDENTICAL' if ok_all else 'MISMATCH FOUND'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
