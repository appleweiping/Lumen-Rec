"""build_confirm_panels with its registered Pilot-1 constants re-derived for the sandbox panels: the Pilot-1 panel sha1s,
PREFIX_LINES (the Pilot-1 panels' user count; both sandbox panels hold the same number), the registered fresh ML-1M
count (eligible - Pilot-1 users) and the Toys CONFIRM minimum (kept at 2000: the sandbox Toys has fewer eligible users,
so its CONFIRM panel is not built, as the amendment prescribes). Everything else is the real script.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("bcp", "scripts/sigir/build_confirm_panels.py")
bcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bcp)

pd_ = Path("outputs/confrec/panels")
for a, b in zip(sys.argv, sys.argv[1:]):
    if a == "--pilot_dir":
        pd_ = Path(b)
sha, n_lines = {}, set()
for s in ("ml1m", "toys"):
    p = pd_ / f"{s}_rated.jsonl"
    sha[s] = hashlib.sha1(p.read_bytes()).hexdigest()
    n_lines.add(sum(1 for line in open(p, "rb") if line.strip()))
assert len(n_lines) == 1, f"the sandbox Pilot-1 panels must hold equally many users: {n_lines}"
bcp.PILOT_SHA1 = sha
bcp.PREFIX_LINES = n_lines.pop()
meta = json.loads((pd_ / "ml1m_rated.meta.json").read_text(encoding="utf-8"))
bcp.FRESH_EXPECTED = {"ml1m": meta["eligible_users"] - bcp.PREFIX_LINES}
print("fake_bcp: PILOT_SHA1", bcp.PILOT_SHA1, "PREFIX_LINES", bcp.PREFIX_LINES, "FRESH_EXPECTED", bcp.FRESH_EXPECTED,
      flush=True)
bcp.main()
