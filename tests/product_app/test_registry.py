from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from itinerary_system.product_app.registry import ProductRunRegistry, RegistryError


def valid_registry_payload(root: Path) -> tuple[Path, dict[str, Any]]:
    run_dir = root / "runs" / "demo"
    run_dir.mkdir(parents=True)
    manifest = run_dir / "manifest.json"
    manifest.write_text(json.dumps({"run_id": "demo"}), encoding="utf-8")
    payload = {
        "schema_version": "product-run-registry-v1",
        "runs": [
            {
                "run_id": "demo",
                "trip_id": "demo_trip",
                "label": "Demo fixture",
                "relative_path": "runs/demo",
                "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
                "capabilities": [
                    "read_only_artifacts",
                    "fixture_copilot",
                    "typed_drafts",
                    "registered_preview",
                    "experimental_pointer_decisions",
                ],
                "default": True,
            }
        ],
    }
    registry = root / "registry.json"
    return registry, payload


def write_registry(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_strict_registry_accepts_pinned_run_without_dashboard(tmp_path: Path) -> None:
    path, payload = valid_registry_payload(tmp_path)
    write_registry(path, payload)

    registry = ProductRunRegistry(tmp_path, path)

    assert registry.default.trip_id == "demo_trip"
    assert registry.default.manifest_hash == payload["runs"][0]["manifest_sha256"]
    assert not (registry.run_dir("demo") / "dashboard_product").exists()


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (lambda value: value.update(schema_version="wrong"), "registry_schema_invalid"),
        (lambda value: value.update(extra=True), "registry_schema_invalid"),
        (lambda value: value["runs"][0].update(extra=True), "run_record_schema_invalid"),
        (lambda value: value["runs"][0].pop("trip_id"), "run_record_schema_invalid"),
        (
            lambda value: value["runs"][0].update(relative_path="../outside"),
            "unsafe_run_path",
        ),
        (
            lambda value: value["runs"][0].update(manifest_sha256="0" * 64),
            "manifest_hash_mismatch",
        ),
        (
            lambda value: value["runs"][0].update(
                capabilities=["read_only_artifacts", "read_only_artifacts"]
            ),
            "duplicate_capability",
        ),
        (
            lambda value: value["runs"][0].update(capabilities=["live_acceptance"]),
            "unknown_capability",
        ),
        (lambda value: value["runs"][0].update(default="yes"), "invalid_default_flag"),
    ],
)
def test_registry_rejects_schema_path_hash_and_capability_drift(
    tmp_path: Path,
    mutate: Any,
    expected_code: str,
) -> None:
    path, payload = valid_registry_payload(tmp_path)
    mutate(payload)
    write_registry(path, payload)

    with pytest.raises(RegistryError) as error:
        ProductRunRegistry(tmp_path, path)
    assert error.value.code == expected_code


def test_registry_rejects_duplicate_run_and_non_single_default(tmp_path: Path) -> None:
    path, payload = valid_registry_payload(tmp_path)
    payload["runs"].append(dict(payload["runs"][0]))
    write_registry(path, payload)
    with pytest.raises(RegistryError) as duplicate:
        ProductRunRegistry(tmp_path, path)
    assert duplicate.value.code == "duplicate_run_id"

    payload["runs"] = [payload["runs"][0]]
    payload["runs"][0]["default"] = False
    write_registry(path, payload)
    with pytest.raises(RegistryError) as missing_default:
        ProductRunRegistry(tmp_path, path)
    assert missing_default.value.code == "expected_one_default_run"


def test_registry_sanitizes_missing_and_unreadable_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = valid_registry_payload(tmp_path)
    write_registry(path, payload)
    manifest = tmp_path / "runs" / "demo" / "manifest.json"
    manifest.unlink()
    with pytest.raises(RegistryError) as missing:
        ProductRunRegistry(tmp_path, path)
    assert missing.value.code == "missing_run_artifact"

    manifest.write_text(json.dumps({"run_id": "demo"}), encoding="utf-8")
    payload["runs"][0]["manifest_sha256"] = sha256(manifest.read_bytes()).hexdigest()
    write_registry(path, payload)
    original = Path.read_bytes

    def denied(self: Path) -> bytes:
        if self.name == "manifest.json":
            raise PermissionError("host path must not escape")
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", denied)
    with pytest.raises(RegistryError) as unavailable:
        ProductRunRegistry(tmp_path, path)
    assert unavailable.value.code == "run_artifact_unavailable"
    assert str(tmp_path) not in str(unavailable.value)

