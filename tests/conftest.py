from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The repository keeps executable entry points in several script subtrees.
# Adding those directories once during test collection makes imports behave
# the same way as direct script execution, without requiring a machine-local
# ``PYTHONPATH`` or a working-directory side effect.
TEST_IMPORT_ROOTS = (
    PROJECT_ROOT,
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "scripts" / "audit",
    PROJECT_ROOT / "scripts" / "adapters",
    PROJECT_ROOT / "scripts" / "analysis",
    PROJECT_ROOT / "scripts" / "build",
    PROJECT_ROOT / "scripts" / "misc",
    PROJECT_ROOT / "scripts" / "pipeline",
    PROJECT_ROOT / "scripts" / "train",
)

for import_root in reversed(TEST_IMPORT_ROOTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
