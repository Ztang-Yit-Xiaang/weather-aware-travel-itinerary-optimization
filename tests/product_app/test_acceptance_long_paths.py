"""Actual acceptance snapshot I/O at the Windows legacy path-length boundary."""

import json
from pathlib import Path
from uuid import UUID

import pytest

from itinerary_system.product_app import acceptance


def test_snapshot_in_long_directory_preserves_atomic_append_contract(tmp_path, monkeypatch):
    store = acceptance.AcceptanceStore(tmp_path / "state")
    # The final path is supported (<260), but duplicating its decision filename
    # in the temporary name crosses the Windows legacy path limit.
    padding = 185 - len(str(tmp_path)) - 1
    assert 1 <= padding <= 240, "run this boundary test from a short --basetemp"
    directory = tmp_path / ("d" * padding)
    directory.mkdir()
    target = directory / ("decision_" + "a" * 32 + ".json")
    payload = {"plan_id": "plan_child_path_boundary", "markers": ["pfeiffer_big_sur_state_park"]}
    assert len(str(target)) < 260
    assert len(str(directory / ("." + target.name + "." + "b" * 32 + ".tmp"))) >= 260
    events = []
    real_open = Path.open
    real_fsync = acceptance.os.fsync
    real_link = acceptance.os.link

    def traced_open(path, mode="r", *args, **kwargs):
        if path.parent == directory:
            events.append({"operation": "open", "path": str(path), "mode": mode})
        return real_open(path, mode, *args, **kwargs)

    def traced_fsync(fd):
        events.append({"operation": "file_fsync"})
        return real_fsync(fd)

    def traced_link(source, destination):
        events.append({"operation": "hard_link", "source": str(source), "destination": str(destination)})
        assert json.loads(Path(source).read_text(encoding="utf-8")) == payload
        assert not Path(destination).exists()
        return real_link(source, destination)

    monkeypatch.setattr(Path, "open", traced_open)
    monkeypatch.setattr(acceptance.os, "fsync", traced_fsync)
    monkeypatch.setattr(acceptance.os, "link", traced_link)
    store._append_only_json(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert list(directory.iterdir()) == [target]
    created = next(event for event in events if event["operation"] == "open")
    assert created["mode"] == "x"
    assert Path(created["path"]).parent == target.parent
    assert len(created["path"]) < 260
    assert [event["operation"] for event in events].index("file_fsync") < [
        event["operation"] for event in events
    ].index("hard_link")
    (tmp_path / "exact_long_path_io.json").write_text(
        json.dumps(
            {
                "input": {"target_path": str(target), "target_path_length": len(str(target)), "payload": payload},
                "io_events": events,
                "output": json.loads(target.read_text(encoding="utf-8")),
                "temporary_files_remaining": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_existing_snapshot_is_idempotent_but_different_payload_cannot_replace_it(tmp_path):
    store = acceptance.AcceptanceStore(tmp_path)
    target = store.snapshots_dir / "decision_existing.json"
    store._append_only_json(target, {"value": 1})
    original = target.read_bytes()
    store._append_only_json(target, {"value": 1})
    with pytest.raises(acceptance.AcceptanceError, match="acceptance_snapshot_conflict"):
        store._append_only_json(target, {"value": 2})
    assert target.read_bytes() == original
    assert list(target.parent.iterdir()) == [target]


def test_link_failure_cleans_only_this_writer_temporary_file(tmp_path, monkeypatch):
    store = acceptance.AcceptanceStore(tmp_path)
    target = store.snapshots_dir / "decision_link_failure.json"

    def failed_link(source, destination):
        assert Path(source).is_file()
        raise OSError("injected local hard-link failure")

    monkeypatch.setattr(acceptance.os, "link", failed_link)
    with pytest.raises(OSError, match="injected local hard-link failure"):
        store._append_only_json(target, {"value": 1})
    assert list(target.parent.iterdir()) == []


def test_exclusive_temporary_collision_does_not_delete_other_writer_file(tmp_path, monkeypatch):
    store = acceptance.AcceptanceStore(tmp_path)
    target = store.snapshots_dir / "decision_collision.json"
    fixed_uuid = UUID("12345678-1234-5678-1234-567812345678")
    monkeypatch.setattr(acceptance, "uuid4", lambda: fixed_uuid)
    # Intercept only the pathname to test the ownership invariant with either
    # temporary naming scheme, without weakening actual O_EXCL behavior.
    real_open = Path.open
    collision = []

    def collide(path, mode="r", *args, **kwargs):
        if mode == "x" and path.parent == target.parent:
            with real_open(path, "w", encoding="utf-8") as handle:
                handle.write("other active writer")
            collision.append(path)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", collide)
    with pytest.raises(FileExistsError):
        store._append_only_json(target, {"value": 1})
    assert len(collision) == 1
    assert collision[0].read_text(encoding="utf-8") == "other active writer"
    assert not target.exists()
