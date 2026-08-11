"""Build the deterministic California Coast product-demo artifact package.

The builder reads the frozen v6 parent, disruption, and road matrix without
modifying them.  Each child is produced by the existing deterministic repair
executor and independently evaluated by the existing pipeline evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from itinerary_system.pipeline_runner import (  # noqa: E402
    build_deterministic_context_aware_heuristic_executor,
    run_research_pipeline,
)
from itinerary_system.plans import load_plan  # noqa: E402
from itinerary_system.repair.day_route_solver import DayRouteSolverConfig  # noqa: E402
from itinerary_system.repair_planner import RepairRequest  # noqa: E402
from itinerary_system.research_artifacts import stable_content_hash  # noqa: E402
from itinerary_system.routing import RouteMatrix, RouteMatrixCell  # noqa: E402

PACKAGE_SCHEMA = "product-demo-package-v1"
PRODUCT_DEMO_RUN_ID = "california_coast_product_demo_v2"
SOURCE_RUN_RELATIVE = Path("runs/e3ux-weather-repair-demo-v6")
PARENT_RELATIVE = Path("plans/plan_e1c4f803691e3188.json")
REQUEST_RELATIVE = Path("requests/bench_weather_deterioration_4062619eeaf4.json")
MATRIX_RELATIVE = Path("routing/route_matrix_68ab535465b06808.json")

PINNED_SOURCE_SHA256 = {
    "manifest.json": "bcc65f820f6555211915bd129552dd6d6b674902ef1da6510b120b661ff6f2dc",
    PARENT_RELATIVE.as_posix(): "4036ca84db5b1194ddacdc82aab344141104edf81eba6b42a1b05af950f8939d",
    REQUEST_RELATIVE.as_posix(): "4044fec09dc1096696937a214a4ed600168508e29b5362f40180cd7fc21757f5",
    MATRIX_RELATIVE.as_posix(): "6f3b5b8eda1a9f73e28b6404ef15126086dbd7861c683790af6f6e87b41b4677",
}
PINNED_PARENT_CONTENT_HASH = "20b540fdc5ed5cc9"
PINNED_MATRIX_ID = "route_matrix_68ab535465b06808"
PINNED_ROUTE_BUNDLE_ID = "route_bundle_a60c80047098a3b6"
PINNED_ROUTE_SOURCE_SHA256 = "3d2ded2d2a862aa6bcf026429b0a0376386928cfa3af109758301d1647584f71"
DEMO_EVALUATED_AT = "2026-08-04T00:00:00+00:00"

DAY_ANCHORS = {
    1: ("los_angeles_international_airport", "the_line_la"),
    2: ("the_line_la", "the_line_la"),
    3: ("the_line_la", "the_line_la"),
    4: ("the_line_la", "hotel_milo_santa_barbara"),
    5: ("hotel_milo_santa_barbara", "hotel_milo_santa_barbara"),
    6: ("hotel_milo_santa_barbara", "hotel_milo_santa_barbara"),
    7: ("hotel_milo_santa_barbara", "san_francisco_international_airport"),
}

LOW_DRIVING_CANDIDATE = {
    "stop_id": "santa_barbara_museum_of_natural_history_sea_center",
    "name": "Santa Barbara Museum of Natural History Sea Center",
    "city": "Santa Barbara",
    "day": 7,
    "stop_order": 1,
    "latitude": 34.4093803,
    "longitude": -119.6853711,
    "final_poi_value": 0.6613132306435597,
    "visit_duration_minutes": 45,
    "weather_risk": 0.1,
    "route_snap_validated": True,
    "candidate_source": "w2_deterministic_fixture_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "runs" / "california-coast-product-demo-v2",
        help="New package directory; existing paths are never overwritten.",
    )
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    return parser.parse_args()


def build_product_demo(output_dir: Path, *, repository_root: Path = REPOSITORY_ROOT) -> Path:
    """Build and validate the parent-plus-two-child product demo package."""

    repository_root = repository_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"product demo output already exists: {output_dir}")
    source_run = repository_root / SOURCE_RUN_RELATIVE
    _verify_frozen_sources(source_run)

    parent = load_plan(source_run / PARENT_RELATIVE)
    if parent.content_hash != PINNED_PARENT_CONTENT_HASH:
        raise ValueError("frozen parent content hash does not match the W2 fixture contract")
    source_request = _read_json(source_run / REQUEST_RELATIVE)
    route_matrix = _load_route_matrix(source_run / MATRIX_RELATIVE)
    day_config = _day_route_config()
    _verify_anchor_coverage(route_matrix, day_config)

    requests = (
        (
            "recommended",
            "w2_weather_recommended_v2",
            tuple(dict(record) for record in source_request["candidate_pois"]),
            "Replace the weather-infeasible stop with the preservation-oriented route-validated candidate.",
        ),
        (
            "low_driving",
            "w2_weather_low_driving_v2",
            (dict(LOW_DRIVING_CANDIDATE),),
            "Replace the weather-infeasible stop with the lower-driving route-validated candidate.",
        ),
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    output_created = False
    try:
        alternatives: list[dict[str, Any]] = []
        for role, run_id, candidates, intent in requests:
            request = _repair_request(
                source_request,
                request_id=run_id,
                candidates=candidates,
                user_intent=intent,
            )
            pipeline_run = run_research_pipeline(
                config_path=repository_root / "configs/default_trip_config.yaml",
                catalog_snapshot_id=parent.catalog_snapshot_id,
                context_snapshot_id=parent.context_snapshot_id,
                parent_plan_id=parent.plan_id,
                repair_request_id=request.request_id,
                output_root=temporary / "alternatives",
                run_id=run_id,
                executor=build_deterministic_context_aware_heuristic_executor(
                    parent_plan=parent,
                    repair_request=request,
                    route_matrix=route_matrix,
                    day_route_config=day_config,
                    publication_mode=True,
                ),
                strict=True,
            )
            _canonicalize_evaluation_certificate(pipeline_run.output_dir)
            _redact_resolved_config_path(pipeline_run.output_dir)
            alternatives.append(_alternative_record(role, pipeline_run.output_dir, route_matrix, day_config))

        _validate_alternatives(alternatives)
        package = {
            "schema_version": PACKAGE_SCHEMA,
            "run_id": PRODUCT_DEMO_RUN_ID,
            "package_id": PRODUCT_DEMO_RUN_ID,
            "source_run": {
                "run_id": "e3ux_weather_repair_demo_v6",
                "relative_path": SOURCE_RUN_RELATIVE.as_posix(),
                "manifest_sha256": PINNED_SOURCE_SHA256["manifest.json"],
            },
            "parent": {
                "plan_id": parent.plan_id,
                "content_hash": parent.content_hash,
                "relative_path": (SOURCE_RUN_RELATIVE / PARENT_RELATIVE).as_posix(),
                "route_legs": _route_legs(
                    parent,
                    day_config,
                    evidence_scope="road_validated_daily_route_leg",
                ),
            },
            "route_evidence": {
                "matrix_id": route_matrix.matrix_id,
                "source_bundle_id": route_matrix.source_bundle_id,
                "source_content_sha256": route_matrix.source_content_sha256,
                "matrix_file_sha256": PINNED_SOURCE_SHA256[MATRIX_RELATIVE.as_posix()],
                "relative_path": (SOURCE_RUN_RELATIVE / MATRIX_RELATIVE).as_posix(),
            },
            "alternatives": alternatives,
        }
        package["artifacts_sha256"] = {
            path.relative_to(temporary).as_posix(): _file_sha256(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest_text = _stable_json(package)
        output_dir.mkdir()
        output_created = True
        for source in sorted(temporary.iterdir()):
            destination = output_dir / source.name
            if source.is_dir():
                shutil.copytree(source, destination, copy_function=shutil.copy2)
            else:
                shutil.copy2(source, destination)
        (output_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    except Exception:
        if output_created:
            shutil.rmtree(output_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return output_dir / "manifest.json"


def _verify_frozen_sources(source_run: Path) -> None:
    for relative, expected in PINNED_SOURCE_SHA256.items():
        path = source_run / relative
        if not path.is_file():
            raise FileNotFoundError(f"required frozen W2 source is missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen W2 source hash mismatch: {relative}")


def _canonicalize_evaluation_certificate(run_dir: Path) -> None:
    """Remove wall-clock variance from the deterministic product fixture."""

    manifest = _read_json(run_dir / "manifest.json")
    certificate_path = run_dir / manifest["artifacts"]["evaluations"][0]
    certificate = _read_json(certificate_path)
    certificate["evaluated_at"] = DEMO_EVALUATED_AT
    payload = dict(certificate)
    payload.pop("content_hash", None)
    certificate["content_hash"] = stable_content_hash(payload)
    certificate_path.write_text(_stable_json(certificate), encoding="utf-8")


def _redact_resolved_config_path(run_dir: Path) -> None:
    config_path = run_dir / "resolved_config.redacted.json"
    config = _read_json(config_path)
    config["_source_path"] = "configs/default_trip_config.yaml"
    config_path.write_text(_stable_json(config), encoding="utf-8")


def _load_route_matrix(path: Path) -> RouteMatrix:
    record = _read_json(path)
    cells: dict[tuple[str, str], RouteMatrixCell] = {}
    for raw_cell in record.get("cells", ()):
        cell_record = dict(raw_cell)
        cell_record["geometry"] = tuple(tuple(point) for point in cell_record.get("geometry", ()))
        cell = RouteMatrixCell(**cell_record)
        cells[(cell.origin_id, cell.destination_id)] = cell
    matrix = RouteMatrix(
        matrix_id=str(record.get("matrix_id", "")),
        context_snapshot_id=str(record.get("context_snapshot_id", "")),
        entity_ids=tuple(str(value) for value in record.get("entity_ids", ())),
        cells=cells,
        source_bundle_id=str(record.get("source_bundle_id", "")),
        source_content_sha256=str(record.get("source_content_sha256", "")),
    )
    if (
        matrix.matrix_id != PINNED_MATRIX_ID
        or matrix.source_bundle_id != PINNED_ROUTE_BUNDLE_ID
        or matrix.source_content_sha256 != PINNED_ROUTE_SOURCE_SHA256
    ):
        raise ValueError("frozen route-matrix provenance does not match the W2 fixture contract")
    return matrix


def _day_route_config() -> DayRouteSolverConfig:
    return DayRouteSolverConfig(
        max_day_minutes=720.0,
        default_visit_minutes=45.0,
        strict_route_matrix=True,
        start_anchor_by_day={day: anchors[0] for day, anchors in DAY_ANCHORS.items()},
        end_anchor_by_day={day: anchors[1] for day, anchors in DAY_ANCHORS.items()},
    )


def _verify_anchor_coverage(matrix: RouteMatrix, config: DayRouteSolverConfig) -> None:
    for day in sorted(DAY_ANCHORS):
        start = config.start_anchor_by_day[day]
        end = config.end_anchor_by_day[day]
        if start != end:
            matrix.cell(start, end).require_publication_eligible()
    for candidate_id in (
        "bixby_creek_bridge_viewpoint",
        str(LOW_DRIVING_CANDIDATE["stop_id"]),
    ):
        matrix.cell(config.start_anchor_by_day[7], candidate_id).require_publication_eligible()
        matrix.cell(candidate_id, config.end_anchor_by_day[7]).require_publication_eligible()


def _repair_request(
    source: dict[str, Any],
    *,
    request_id: str,
    candidates: tuple[dict[str, Any], ...],
    user_intent: str,
) -> RepairRequest:
    constraints = dict(source.get("confirmed_constraints", {}))
    constraints["disruption_id"] = request_id
    return RepairRequest(
        request_id=request_id,
        baseline_route=tuple(dict(record) for record in source.get("baseline_route", ())),
        user_intent=user_intent,
        evidence_records=tuple(dict(record) for record in source.get("evidence_records", ())),
        tolerance_profile=dict(source.get("tolerance_profile", {})),
        confirmed_constraints=constraints,
        candidate_pois=candidates,
    )


def _alternative_record(
    role: str,
    run_dir: Path,
    matrix: RouteMatrix,
    config: DayRouteSolverConfig,
) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    plan_paths = manifest["artifacts"]["plans"]
    output_plan_id = manifest["pipeline_run"]["output_plan_ids"][0]
    plan_path = next(path for path in plan_paths if Path(path).stem == output_plan_id)
    plan = load_plan(run_dir / plan_path)
    certificate_path = run_dir / manifest["artifacts"]["evaluations"][0]
    certificate = _read_json(certificate_path)
    return {
        "role": role,
        "run_relative_path": f"alternatives/{run_dir.name}",
        "plan_relative_path": f"alternatives/{run_dir.name}/{plan_path}",
        "certificate_relative_path": (
            f"alternatives/{run_dir.name}/{certificate_path.relative_to(run_dir).as_posix()}"
        ),
        "plan_id": plan.plan_id,
        "plan_content_hash": plan.content_hash,
        "parent_plan_id": plan.parent_plan_id,
        "certificate_id": certificate.get("certificate_id"),
        "evaluation_status": certificate.get("evaluation_status"),
        "comparison_eligibility": certificate.get("comparison_eligibility"),
        "route_publication_ready": bool((certificate.get("route_validation") or {}).get("publication_ready")),
        "route_required_leg_count": (certificate.get("route_validation") or {}).get("required_leg_count"),
        "route_fallback_leg_count": (certificate.get("route_validation") or {}).get("fallback_leg_count"),
        "route_total_minutes": round(_route_total_minutes(plan, matrix, config), 6),
        "route_legs": _route_legs(plan, config),
    }


def _route_total_minutes(plan: Any, matrix: RouteMatrix, config: DayRouteSolverConfig) -> float:
    stops_by_day = {
        int(day["day"]): tuple(str(stop_id) for stop_id in day.get("stop_ids", ()))
        for day in plan.ordered_days
    }
    total = 0.0
    for day in sorted(stops_by_day):
        sequence = (
            config.start_anchor_by_day[day],
            *stops_by_day[day],
            config.end_anchor_by_day[day],
        )
        total += sum(
            matrix.duration_minutes(origin, destination, strict=True)
            for origin, destination in zip(sequence[:-1], sequence[1:], strict=False)
        )
    return total


def _route_legs(
    plan: Any,
    config: DayRouteSolverConfig,
    *,
    evidence_scope: str = "certified_daily_route_leg",
) -> list[dict[str, Any]]:
    stops_by_day = {
        int(day["day"]): tuple(str(stop_id) for stop_id in day.get("stop_ids", ()))
        for day in plan.ordered_days
    }
    result: list[dict[str, Any]] = []
    for day in sorted(stops_by_day):
        sequence = (
            config.start_anchor_by_day[day],
            *stops_by_day[day],
            config.end_anchor_by_day[day],
        )
        result.extend(
            {
                "day": day,
                "origin_id": origin,
                "destination_id": destination,
                "evidence_scope": evidence_scope,
            }
            for origin, destination in zip(sequence[:-1], sequence[1:], strict=False)
        )
    for previous, current in zip(result[:-1], result[1:], strict=False):
        if previous["destination_id"] != current["origin_id"]:
            raise ValueError(
                "product demo route is discontinuous between "
                f"day {previous['day']} and day {current['day']}: "
                f"{previous['destination_id']} != {current['origin_id']}"
            )
    return result


def _validate_alternatives(alternatives: list[dict[str, Any]]) -> None:
    if len(alternatives) != 2:
        raise ValueError("the W2 product demo requires exactly two evaluated child alternatives")
    if len({record["plan_id"] for record in alternatives}) != 2:
        raise ValueError("W2 child plan IDs are not distinct")
    if len({record["plan_content_hash"] for record in alternatives}) != 2:
        raise ValueError("W2 child plan content hashes are not distinct")
    for record in alternatives:
        if record["comparison_eligibility"] != "eligible" or record["evaluation_status"] not in {
            "PASSED",
            "PASSED_WITH_WARNINGS",
        }:
            raise ValueError(f"W2 alternative was independently rejected: {record['role']}")
        if not record["route_publication_ready"] or record["route_fallback_leg_count"] != 0:
            raise ValueError(f"W2 alternative lacks strict road evidence: {record['role']}")
    by_role = {record["role"]: record for record in alternatives}
    if by_role["low_driving"]["route_total_minutes"] >= by_role["recommended"]["route_total_minutes"]:
        raise ValueError("the labeled low-driving W2 alternative does not reduce route minutes")


def _read_json(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return record


def _stable_json(record: dict[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    manifest = build_product_demo(args.output, repository_root=args.repository_root)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
