"""Run the feature-gated permission-aware repair/clarification entry point."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

main = import_module("itinerary_system.interaction.cli").main


if __name__ == "__main__":
    raise SystemExit(main())
