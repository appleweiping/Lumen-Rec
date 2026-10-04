"""src.confrec.gatefix_stage3 with its G7 scale constants re-derived for the dry-run sandbox, as fake_bcp.py and
fake_gatefix.py re-derive theirs. The sandbox stands for the server at the scale of its Pilot-1 panels:
  N_SECOND (the first 1,500 fresh Toys / Video_Games users)  -> the sandbox Pilot-1 Toys panel's user count (100);
  N_SPORTS (the first 1,000 sports VALID events)             -> the sandbox sports TEST panel's event count (34: that
                                                                file stands for the TEST events 1-1000);
  TOYS_MIN_FRESH (500) stays the registered value; S3_MIN_FRESH=<n> overrides it (a scenario forcing the G7
  Video_Games fallback).
A constant whose sandbox file is absent keeps its registered value. Everything else is the real module (run from the
sandbox: cwd = the sandbox repo copy).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from src.confrec import gatefix_stage3 as gs3  # noqa: E402

XT = Path("outputs/baselines/external_tasks")


def n_lines(path, default: int) -> int:
    if not Path(path).is_file():
        return default
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def patch() -> None:
    gs3.N_SECOND = n_lines("outputs/confrec/panels/toys_rated.jsonl", gs3.N_SECOND)
    gs3.N_SPORTS = n_lines(XT / "sports_large10000_100neg_test_same_candidate" / "ranking_test.jsonl", gs3.N_SPORTS)
    if os.environ.get("S3_MIN_FRESH"):
        gs3.TOYS_MIN_FRESH = int(os.environ["S3_MIN_FRESH"])
    print(f"fake_stage3: N_SECOND {gs3.N_SECOND} N_SPORTS {gs3.N_SPORTS} TOYS_MIN_FRESH {gs3.TOYS_MIN_FRESH}",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    patch()
    sys.exit(gs3.main(sys.argv[1:]))
