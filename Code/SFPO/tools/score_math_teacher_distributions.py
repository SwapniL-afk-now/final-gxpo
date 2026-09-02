#!/usr/bin/env python3
"""Score stored math traces with teacher prompt log-probabilities only.

The implementation is shared with the existing sparse sidecar writer; this
entrypoint makes the math pipeline explicit and pins the requested models.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from score_code_teacher_distributions import main as _main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(_main())
