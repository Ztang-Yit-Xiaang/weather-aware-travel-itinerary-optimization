"""Register browser verification screenshots in a product dashboard manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from itinerary_system.product_dashboard_renderer import (  # noqa: E402
    register_product_dashboard_screenshots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hash dashboard_product/screenshots/*.png into its manifest."
    )
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    hashes = register_product_dashboard_screenshots(parse_args().run_dir)
    for path, digest in sorted(hashes.items()):
        print(f"{path}: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
