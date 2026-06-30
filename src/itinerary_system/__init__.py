"""Configurable enrichment-first itinerary optimization package."""

from .config import TripConfig, load_trip_config, save_resolved_config
from .data import DatasetBundle, DatasetValidationReport, load_dataset_bundle, validate_dataset_bundle
from .phase0_exporter import PHASE0_ARTIFACT_FILES, write_phase0_research_artifacts
from .repair_planner import (
    CounterfactualExplanation,
    EvaluationReport,
    EvidenceConflict,
    ParsedRepairIntent,
    RepairOperation,
    RepairPlan,
    RepairRequest,
    build_repair_plan,
    detect_evidence_conflicts,
    evaluate_repair_plan,
    generate_repair_alternatives,
    parse_repair_intent,
)
from .research_artifacts import PlanArtifact, PlannerRun, ResearchEvaluationReport, evaluate_phase0_plan
from .request_schema import TripPlanningRequest, normalize_interest_weights, request_to_config_overrides
from .routing import (
    ROAD_ROUTE_CACHE_AUDIT_FILENAME,
    ROAD_ROUTE_CACHE_FILENAME,
    RoadRouteCache,
    RouteLegResult,
    RouteResult,
    build_road_route_cache_from_artifacts,
    load_road_route_cache,
    osrm_cache_key,
)

__all__ = [
    "CounterfactualExplanation",
    "DatasetBundle",
    "DatasetValidationReport",
    "EvaluationReport",
    "EvidenceConflict",
    "PHASE0_ARTIFACT_FILES",
    "PlanArtifact",
    "ParsedRepairIntent",
    "PlannerRun",
    "ROAD_ROUTE_CACHE_AUDIT_FILENAME",
    "ROAD_ROUTE_CACHE_FILENAME",
    "ResearchEvaluationReport",
    "RepairOperation",
    "RepairPlan",
    "RepairRequest",
    "RoadRouteCache",
    "RouteLegResult",
    "RouteResult",
    "TripConfig",
    "TripPlanningRequest",
    "build_repair_plan",
    "build_road_route_cache_from_artifacts",
    "detect_evidence_conflicts",
    "evaluate_phase0_plan",
    "evaluate_repair_plan",
    "generate_repair_alternatives",
    "load_dataset_bundle",
    "load_road_route_cache",
    "load_trip_config",
    "normalize_interest_weights",
    "osrm_cache_key",
    "parse_repair_intent",
    "request_to_config_overrides",
    "save_resolved_config",
    "validate_dataset_bundle",
    "write_phase0_research_artifacts",
]
