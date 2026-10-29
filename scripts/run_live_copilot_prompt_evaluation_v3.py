"""Run future exact-I/O gates with normalized output paths.

This thin version preserves the v2 live/evidence behavior, adds its own source
identity to the frozen snapshot, and resolves the output directory before any
provider call so file-inventory finalization cannot repeat the v5 relative-path
failure.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import run_live_copilot_prompt_evaluation_v2 as v2

SOURCE_INPUTS = tuple(
    dict.fromkeys((*v2.SOURCE_INPUTS, "scripts/run_live_copilot_prompt_evaluation_v3.py"))
)


def build_source_snapshot() -> dict[str, Any]:
    files = [
        {
            "path": relative,
            "sha256": v2.file_sha256(v2.REPOSITORY_ROOT / relative),
            "size_bytes": (v2.REPOSITORY_ROOT / relative).stat().st_size,
        }
        for relative in SOURCE_INPUTS
    ]
    snapshot = {"schema_version": "copilot-live-source-snapshot-v2", "files": files}
    snapshot["content_hash"] = v2.canonical_sha256(snapshot)
    return snapshot


async def main_async() -> int:
    args = v2.parse_args()
    args.output_dir = Path(args.output_dir).resolve()
    original_builder = v2.build_source_snapshot
    v2.build_source_snapshot = build_source_snapshot
    try:
        return await v2._main(args)
    finally:
        v2.build_source_snapshot = original_builder


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())

