"""Append-only JSON repository for immutable plan artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..research_artifacts import PlanArtifactV2


class PlanRepositoryConflict(RuntimeError):
    """Raised when a plan ID already exists with different content."""


class PlanNotFound(FileNotFoundError):
    """Raised when a requested plan artifact is not present."""


def load_plan(path: Path | str) -> PlanArtifactV2:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    return _plan_from_record(record)


def save_plan_append_only(plan: PlanArtifactV2, root: Path | str) -> Path:
    repository = PlanRepository(Path(root))
    return repository.save(plan)


class PlanRepository:
    """Small append-only store for reviewed parents and generated children."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.json"

    def exists(self, plan_id: str) -> bool:
        return self._plan_path(plan_id).exists()

    def load(self, plan_id: str) -> PlanArtifactV2:
        path = self._plan_path(plan_id)
        if not path.exists():
            raise PlanNotFound(f"plan {plan_id!r} was not found under {self.root}")
        return load_plan(path)

    def save(self, plan: PlanArtifactV2) -> Path:
        if not plan.plan_id:
            raise ValueError("plan_id must be nonempty")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._plan_path(plan.plan_id)
        record = plan.to_record(include_content_hash=True)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != record:
                raise PlanRepositoryConflict(f"plan_id {plan.plan_id!r} already exists with different content")
            return path
        path.write_text(_stable_json(record), encoding="utf-8")
        self._write_index()
        return path

    def verify_hash(self, plan_id: str) -> bool:
        plan = self.load(plan_id)
        record = json.loads(self._plan_path(plan_id).read_text(encoding="utf-8"))
        return record.get("content_hash") == plan.content_hash

    def _plan_path(self, plan_id: str) -> Path:
        safe_plan_id = str(plan_id).replace("/", "_").replace("\\", "_")
        return self.root / f"{safe_plan_id}.json"

    def _write_index(self) -> None:
        entries: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")):
            if path.name == self.index_path.name:
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            entries.append(
                {
                    "plan_id": record.get("plan_id", path.stem),
                    "path": path.name,
                    "content_hash": record.get("content_hash", ""),
                    "parent_plan_id": record.get("parent_plan_id"),
                    "schema_version": record.get("schema_version", ""),
                }
            )
        self.index_path.write_text(_stable_json({"plans": entries}), encoding="utf-8")


def _plan_from_record(record: dict[str, Any]) -> PlanArtifactV2:
    route_ids_by_day = {
        int(day): str(route_id)
        for day, route_id in dict(record.get("route_ids_by_day", {})).items()
        if str(day).lstrip("-").isdigit()
    }
    return PlanArtifactV2(
        plan_id=str(record["plan_id"]),
        parent_plan_id=record.get("parent_plan_id"),
        source_run_id=str(record.get("source_run_id", "")),
        planning_request_id=str(record.get("planning_request_id", "")),
        catalog_snapshot_id=str(record.get("catalog_snapshot_id", "")),
        context_snapshot_id=str(record.get("context_snapshot_id", "")),
        selected_stops=tuple(dict(stop) for stop in record.get("selected_stops", ())),
        day_assignments={str(key): int(value) for key, value in dict(record.get("day_assignments", {})).items()},
        sequence=tuple(str(stop_id) for stop_id in record.get("sequence", ())),
        lodging_assignments={str(key): str(value) for key, value in dict(record.get("lodging_assignments", {})).items()},
        ordered_days=tuple(dict(day) for day in record.get("ordered_days", ())),
        route_ids_by_day=route_ids_by_day,
        owned_constraints=tuple(dict(constraint) for constraint in record.get("owned_constraints", ())),
        modeled_metrics={str(key): float(value) for key, value in dict(record.get("modeled_metrics", {})).items()},
        context_exposure_components={
            str(key): float(value) for key, value in dict(record.get("context_exposure_components", {})).items()
        },
        change_components={str(key): float(value) for key, value in dict(record.get("change_components", {})).items()},
        certificate_id=record.get("certificate_id"),
        created_at=str(record.get("created_at", "")),
        schema_version=str(record.get("schema_version", "plan-artifact-v2")),
    )


def _stable_json(record: dict[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True, default=str) + "\n"
