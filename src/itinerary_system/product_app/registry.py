"""Validated, explicitly pinned product run registry."""

from __future__ import annotations

import json
import secrets
from hashlib import sha256
from pathlib import Path
from typing import Any

from .models import ProductRunRecord

REGISTRY_SCHEMA = "product-run-registry-v1"
TOP_LEVEL_FIELDS = frozenset({"schema_version", "runs"})
RUN_FIELDS = frozenset(
    {"run_id", "trip_id", "label", "relative_path", "manifest_sha256", "capabilities", "default"}
)
ALLOWED_CAPABILITIES = frozenset(
    {
        "read_only_artifacts",
        "fixture_copilot",
        "typed_drafts",
        "registered_preview",
        "experimental_pointer_decisions",
        "multi_plan_product_demo",
    }
)


class RegistryError(ValueError):
    """Raised with a stable code when the product registry is unsafe or unusable."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ProductRunRegistry:
    def __init__(self, repository_root: Path, config_path: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.config_path = config_path.resolve()
        self._records = self._load()

    def _load(self) -> dict[str, ProductRunRecord]:
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RegistryError("registry_unavailable") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RegistryError("registry_invalid_json") from exc
        if not isinstance(payload, dict) or set(payload) != TOP_LEVEL_FIELDS:
            raise RegistryError("registry_schema_invalid")
        if payload.get("schema_version") != REGISTRY_SCHEMA:
            raise RegistryError("registry_schema_invalid")
        rows = payload.get("runs")
        if not isinstance(rows, list) or not rows:
            raise RegistryError("registry_has_no_runs")
        records: dict[str, ProductRunRecord] = {}
        defaults = 0
        for row in rows:
            record = self._validate_row(row)
            if record.run_id in records:
                raise RegistryError("duplicate_run_id")
            records[record.run_id] = record
            defaults += int(record.default)
        if defaults != 1:
            raise RegistryError("expected_one_default_run")
        return records

    def _validate_row(self, row: Any) -> ProductRunRecord:
        if not isinstance(row, dict) or set(row) != RUN_FIELDS:
            raise RegistryError("run_record_schema_invalid")
        run_id = self._identifier(row.get("run_id"), "invalid_run_id")
        trip_id = self._identifier(row.get("trip_id"), "invalid_trip_id")
        label = row.get("label")
        if not isinstance(label, str) or not label.strip() or len(label) > 160:
            raise RegistryError("invalid_run_label")
        relative_path = row.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise RegistryError("unsafe_run_path")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise RegistryError("unsafe_run_path")
        try:
            run_dir = (self.repository_root / relative).resolve(strict=False)
            if self.repository_root not in run_dir.parents:
                raise RegistryError("run_outside_repository")
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                raise RegistryError("missing_run_artifact")
            manifest_bytes = manifest_path.read_bytes()
        except RegistryError:
            raise
        except OSError as exc:
            raise RegistryError("run_artifact_unavailable") from exc
        pinned_hash = row.get("manifest_sha256")
        if not isinstance(pinned_hash, str) or len(pinned_hash) != 64:
            raise RegistryError("invalid_manifest_hash")
        actual_hash = sha256(manifest_bytes).hexdigest()
        if not secrets.compare_digest(pinned_hash.lower(), actual_hash):
            raise RegistryError("manifest_hash_mismatch")
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RegistryError("malformed_run_artifact") from exc
        if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
            raise RegistryError("run_manifest_id_mismatch")
        capabilities_raw = row.get("capabilities")
        if not isinstance(capabilities_raw, list) or not all(isinstance(item, str) for item in capabilities_raw):
            raise RegistryError("invalid_capabilities")
        capabilities = tuple(capabilities_raw)
        if len(capabilities) != len(set(capabilities)):
            raise RegistryError("duplicate_capability")
        if not set(capabilities).issubset(ALLOWED_CAPABILITIES):
            raise RegistryError("unknown_capability")
        if not isinstance(row.get("default"), bool):
            raise RegistryError("invalid_default_flag")
        return ProductRunRecord(
            run_id=run_id,
            trip_id=trip_id,
            label=label.strip(),
            relative_path=relative.as_posix(),
            manifest_hash=actual_hash,
            capabilities=capabilities,
            default=row["default"],
        )

    @staticmethod
    def _identifier(value: Any, code: str) -> str:
        if not isinstance(value, str):
            raise RegistryError(code)
        text = value.strip()
        if not text or len(text) > 128 or not all(character.isalnum() or character in "_-" for character in text):
            raise RegistryError(code)
        return text

    @property
    def default(self) -> ProductRunRecord:
        return next(record for record in self._records.values() if record.default)

    def all(self) -> tuple[ProductRunRecord, ...]:
        return tuple(self._records.values())

    def get(self, run_id: str) -> ProductRunRecord:
        try:
            return self._records[run_id]
        except KeyError as exc:
            raise RegistryError("unknown_run") from exc

    def run_dir(self, run_id: str) -> Path:
        return (self.repository_root / self.get(run_id).relative_path).resolve()
