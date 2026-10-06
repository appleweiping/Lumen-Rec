#!/usr/bin/env python3
"""Write a SHA-256 manifest of the registration documents (for an optional third-party timestamp, e.g. OpenTimestamps).

    python scripts/sigir/make_timestamp_manifest.py [--out docs/sigir/TIMESTAMP_MANIFEST.sha256]

The manifest lists the SHA-256 of every `idea-stage/PREREG_*.md`, of `docs/sigir/PILOT_LOG.md` and `docs/sigir/DEVIATIONS.md`, one
`<sha256>  <path>` line per file in sorted order (the format of `sha256sum`), so the file itself can be stamped with one command:

    ots stamp docs/sigir/TIMESTAMP_MANIFEST.sha256        (opentimestamps-client; contacts public calendar servers: only the digest
                                                          of the manifest leaves the machine)
    ots verify docs/sigir/TIMESTAMP_MANIFEST.sha256.ots

Nothing here talks to a network. A stamp proves that the manifest (hence the listed files at that state) existed at the time of the
Bitcoin block that confirms it; it does not prove authorship.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "sigir" / "TIMESTAMP_MANIFEST.sha256"))
    a = ap.parse_args(argv)
    files = sorted((ROOT / "idea-stage").glob("PREREG_*.md")) + [ROOT / "docs" / "sigir" / "PILOT_LOG.md",
                                                                  ROOT / "docs" / "sigir" / "DEVIATIONS.md"]
    lines = []
    for f in files:
        if not f.is_file():
            print(f"missing: {f}", file=sys.stderr)
            return 1
        data = f.read_bytes().replace(b"\r\n", b"\n")      # LF-normalised, as the committed bytes
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {f.relative_to(ROOT).as_posix()}")
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {a.out}: {len(lines)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
