"""gatefix_select with the registered DEV identity (G2: the burned Pilot-1 users) re-derived for the sandbox panels.

The real constants gatefix_select.PILOT1_DEV were derived from the real Pilot-1 panels exactly like this (see its
comment): panel sha1, panel_dev_signature (n_users, n_rows, user_ids_sha1, rows_sha1), pyes_scorer prompts_sha1 of V0
like at hist_len 10 and the Pilot-1 pilot_mirror arms.raw_like.UAUC. Everything else is the real gatefix_select.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from src.confrec import gatefix_select as gs  # noqa: E402
from src.confrec import pyes_scorer as ps  # noqa: E402


def patch() -> None:
    P = Path("outputs/confrec/panels")
    new = {}
    for p in gs.PANELS:
        panel = P / f"{p}_rated.jsonl"
        sig = gs.panel_dev_signature(panel)
        records = [json.loads(line) for line in open(panel, encoding="utf-8") if line.strip()]
        v0 = ps.prompts_sha1(pr for rec in records for *_, pr in ps.record_requests(rec, ["like"], 10))
        pmj = Path(f"outputs/confrec/pilot1_mirror/{p}_rated/pilot_mirror.json")
        u = (json.loads(pmj.read_text(encoding="utf-8"))["arms"]["raw_like"]["UAUC"] if pmj.exists()
             else gs.PILOT1_DEV[p]["raw_like_UAUC"])
        new[p] = dict(gs.PILOT1_DEV[p], panel_sha1=gs._sha1_file(panel), n_users=sig["n_users"], n_rows=sig["n_rows"],
                      user_ids_sha1=sig["user_ids_sha1"], rows_sha1=sig["rows_sha1"], v0_like_prompts_sha1=v0,
                      raw_like_UAUC=u)
    gs.PILOT1_DEV = new


if __name__ == "__main__":
    patch()
    sys.exit(gs.main(sys.argv[1:]))
