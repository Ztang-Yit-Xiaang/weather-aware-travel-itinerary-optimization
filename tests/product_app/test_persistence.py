from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from itinerary_system.product_app.persistence import (
    LAYOUT_DIRECTORIES,
    LocalStateLayout,
)

ROOT = Path(__file__).resolve().parents[2]


def legacy_snapshot(paths: list[Path]) -> dict[str, tuple[str, int]]:
    return {
        path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in paths
    }


def test_state_layout_probe_and_legacy_detection_preserve_legacy_files(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    decisions = state_root / "decisions"
    decisions.mkdir(parents=True)
    pointer = state_root / "workspace_pointer.json"
    pointer.write_bytes(b'{"legacy":true}\n')
    decision_paths = []
    for index in range(3):
        path = decisions / f"decision-{index}.json"
        path.write_bytes(b'{"legacy":true}\n')
        decision_paths.append(path)
    legacy_paths = [pointer, *decision_paths]
    before = legacy_snapshot(legacy_paths)

    layout = LocalStateLayout(state_root)
    readiness = layout.initialize()

    assert readiness.ready
    assert readiness.code == "state_store_ready"
    assert readiness.legacy is not None
    assert readiness.legacy.workspace_pointer_count == 1
    assert readiness.legacy.decision_file_count == 3
    assert readiness.legacy.import_status == "deferred_w5"
    assert all((state_root / directory).is_dir() for directory in LAYOUT_DIRECTORIES)
    metadata = json.loads((state_root / "state.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "product-app-state-v1"
    assert metadata["layout_version"] == 1
    report = json.loads((state_root / "migrations" / "legacy-v0.json").read_text(encoding="utf-8"))
    assert report == {
        "schema_version": "legacy-state-report-v1",
        "workspace_pointer_count": 1,
        "decision_file_count": 3,
        "import_status": "deferred_w5",
    }
    assert layout.probe().ready
    assert list((state_root / "temporary").iterdir()) == []
    assert legacy_snapshot(legacy_paths) == before


def test_state_layout_rejects_unsupported_metadata_without_rewriting_it(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    metadata = state_root / "state.json"
    metadata.write_bytes(b'{"schema_version":"future-state-v9","layout_version":9}\n')
    before = metadata.read_bytes()

    readiness = LocalStateLayout(state_root).initialize()

    assert readiness.status == "failed"
    assert readiness.code == "state_metadata_invalid"
    assert metadata.read_bytes() == before


def test_state_probe_failure_preserves_valid_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = LocalStateLayout(tmp_path / "state")
    assert layout.initialize().ready
    metadata_before = layout.state_path.read_bytes()
    original_write = LocalStateLayout._write_json_file

    def fail_probe(path: Path, payload: dict, *, exclusive: bool) -> None:
        if path.name.startswith(".state-probe-"):
            raise PermissionError("simulated probe denial")
        original_write(path, payload, exclusive=exclusive)

    monkeypatch.setattr(LocalStateLayout, "_write_json_file", staticmethod(fail_probe))

    readiness = layout.probe()

    assert readiness.status == "failed"
    assert readiness.code == "state_probe_failed"
    assert layout.state_path.read_bytes() == metadata_before
    assert list((layout.root / "temporary").iterdir()) == []


def test_state_lock_times_out_across_processes(tmp_path: Path) -> None:
    layout = LocalStateLayout(tmp_path / "state")
    assert layout.initialize().ready
    child_code = (
        "import sys,time; from pathlib import Path; "
        "from itinerary_system.product_app.persistence import StateFileLock; "
        "lock=StateFileLock(Path(sys.argv[1]),2.0); lock.__enter__(); "
        "print('locked',flush=True); time.sleep(0.5); lock.__exit__(None,None,None)"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(layout.lock_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "locked"
        blocked = LocalStateLayout(layout.root, lock_timeout_seconds=0.1).probe()
        assert blocked.status == "failed"
        assert blocked.code == "state_lock_timeout"
    finally:
        _, error = process.communicate(timeout=3)
    assert process.returncode == 0, error
