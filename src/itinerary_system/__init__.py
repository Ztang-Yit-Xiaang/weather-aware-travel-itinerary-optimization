"""Configurable enrichment-first itinerary optimization package."""

from .config import TripConfig, load_trip_config, save_resolved_config
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
from .request_schema import TripPlanningRequest, normalize_interest_weights, request_to_config_overrides

__all__ = [
    "CounterfactualExplanation",
    "EvaluationReport",
    "EvidenceConflict",
    "ParsedRepairIntent",
    "RepairOperation",
    "RepairPlan",
    "RepairRequest",
    "TripConfig",
    "TripPlanningRequest",
    "build_repair_plan",
    "detect_evidence_conflicts",
    "evaluate_repair_plan",
    "generate_repair_alternatives",
    "load_trip_config",
    "normalize_interest_weights",
    "parse_repair_intent",
    "request_to_config_overrides",
    "save_resolved_config",
]
