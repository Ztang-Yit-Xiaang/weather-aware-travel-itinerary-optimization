"""Allow-listed compilation from semantic interpretations to repair inputs."""

from __future__ import annotations

import math
from dataclasses import replace
from enum import StrEnum
from typing import Any

from ..plans import OwnedConstraint
from ..repair.neighborhood import RepairRadius
from ..repair_planner import ParsedRepairIntent, RepairRequest
from ..research_artifacts import PlanArtifactV2, stable_content_hash
from .models import ModelPatch, SemanticInterpretationCandidate


class PatchType(StrEnum):
    REDUCE_WALKING_BURDEN = "reduce_walking_burden"
    REDUCE_DRIVING_BURDEN = "reduce_driving_burden"
    ADD_MAXIMUM_ARRIVAL_TIME = "add_maximum_arrival_time"
    REQUIRE_EARLIER_LODGING_ARRIVAL = "require_earlier_lodging_arrival"
    PRESERVE_NAMED_STOP = "preserve_named_stop"
    ADJUST_ORDINAL_PACE = "adjust_ordinal_pace"
    SAME_DAY_REPLACEMENT = "same_day_replacement"
    DAY_MOVE = "day_move"
    LODGING_CHANGE = "lodging_change"
    PROBE_PERMISSION_GATED_RELAXATION = "probe_permission_gated_relaxation"


_ALLOWED_PARAMETERS: dict[PatchType, frozenset[str]] = {
    PatchType.REDUCE_WALKING_BURDEN: frozenset({"ordinal_reduction", "max_walking_minutes"}),
    PatchType.REDUCE_DRIVING_BURDEN: frozenset({"target_minutes_reduction", "max_daily_travel_minutes"}),
    PatchType.ADD_MAXIMUM_ARRIVAL_TIME: frozenset({"maximum_arrival_time", "day"}),
    PatchType.REQUIRE_EARLIER_LODGING_ARRIVAL: frozenset({"arrival_shift_minutes", "maximum_arrival_time", "day"}),
    PatchType.PRESERVE_NAMED_STOP: frozenset(),
    PatchType.ADJUST_ORDINAL_PACE: frozenset({"ordinal_reduction", "target_stop_count"}),
    PatchType.SAME_DAY_REPLACEMENT: frozenset({"replacement_stop_id", "day"}),
    PatchType.DAY_MOVE: frozenset({"to_day"}),
    PatchType.LODGING_CHANGE: frozenset({"replacement_lodging_id", "day", "monetary_cost_delta"}),
    PatchType.PROBE_PERMISSION_GATED_RELAXATION: frozenset({"constraint_ids"}),
}

_RADIUS_BY_PATCH: dict[PatchType, RepairRadius] = {
    PatchType.REDUCE_WALKING_BURDEN: RepairRadius.SAME_DAY_REPLACEMENT,
    PatchType.REDUCE_DRIVING_BURDEN: RepairRadius.HOTEL_PRESERVING_REROUTE,
    PatchType.ADD_MAXIMUM_ARRIVAL_TIME: RepairRadius.SAME_STOP_TIME_SHIFT,
    PatchType.REQUIRE_EARLIER_LODGING_ARRIVAL: RepairRadius.HOTEL_PRESERVING_REROUTE,
    PatchType.PRESERVE_NAMED_STOP: RepairRadius.SAME_DAY_REPLACEMENT,
    PatchType.ADJUST_ORDINAL_PACE: RepairRadius.SAME_DAY_REPLACEMENT,
    PatchType.SAME_DAY_REPLACEMENT: RepairRadius.SAME_DAY_REPLACEMENT,
    PatchType.DAY_MOVE: RepairRadius.ADJACENT_DAY_MOVE,
    PatchType.LODGING_CHANGE: RepairRadius.HOTEL_CHANGING_REPAIR,
    PatchType.PROBE_PERMISSION_GATED_RELAXATION: RepairRadius.HOTEL_CHANGING_REPAIR,
}

_REQUIRED_PARAMETERS: dict[PatchType, frozenset[str]] = {
    PatchType.ADD_MAXIMUM_ARRIVAL_TIME: frozenset({"maximum_arrival_time"}),
    PatchType.REQUIRE_EARLIER_LODGING_ARRIVAL: frozenset({"arrival_shift_minutes"}),
    PatchType.SAME_DAY_REPLACEMENT: frozenset({"replacement_stop_id"}),
    PatchType.DAY_MOVE: frozenset({"to_day"}),
    PatchType.LODGING_CHANGE: frozenset({"replacement_lodging_id", "day"}),
    PatchType.PROBE_PERMISSION_GATED_RELAXATION: frozenset({"constraint_ids"}),
}

_TARGET_REQUIRED = frozenset(
    {
        PatchType.PRESERVE_NAMED_STOP,
        PatchType.SAME_DAY_REPLACEMENT,
        PatchType.DAY_MOVE,
        PatchType.LODGING_CHANGE,
        PatchType.PROBE_PERMISSION_GATED_RELAXATION,
    }
)


class AllowListedPatchCompiler:
    def compile(self, parent: PlanArtifactV2, candidate: SemanticInterpretationCandidate) -> ModelPatch:
        reasons: list[str] = []
        try:
            patch_type = PatchType(candidate.interpretation_type)
        except ValueError:
            patch_type = None
            reasons.append("patch_type_not_allow_listed")

        known_targets = _known_target_ids(parent)
        unknown_targets = tuple(target for target in candidate.target_ids if target not in known_targets)
        if unknown_targets:
            reasons.append("target_not_in_parent_or_constraints")
        if patch_type is not None:
            parameter_names = set(candidate.normalized_parameters)
            extra_parameters = parameter_names - set(_ALLOWED_PARAMETERS[patch_type])
            if extra_parameters:
                reasons.append("parameter_not_allow_listed")
            missing_parameters = set(_REQUIRED_PARAMETERS.get(patch_type, ())) - parameter_names
            if missing_parameters:
                reasons.append("required_parameter_missing")
            if patch_type in _TARGET_REQUIRED and not candidate.target_ids:
                reasons.append("target_required")
            if not _json_compatible(candidate.normalized_parameters):
                reasons.append("parameter_value_not_json_compatible")
            if not _parameters_well_formed(candidate.normalized_parameters):
                reasons.append("parameter_value_invalid")
            if any(name.startswith("replacement_") for name in parameter_names) and not candidate.evidence_refs:
                reasons.append("replacement_entity_evidence_missing")
        affected = _affected_constraint_ids(parent, candidate.target_ids)
        patch_seed = {
            "interpretation": candidate.interpretation_id,
            "type": candidate.interpretation_type,
            "targets": candidate.target_ids,
            "parameters": candidate.normalized_parameters,
            "affected_constraints": affected,
        }
        return ModelPatch(
            patch_id=f"model_patch_{stable_content_hash(patch_seed)}",
            interpretation_id=candidate.interpretation_id,
            patch_type=candidate.interpretation_type,
            target_ids=candidate.target_ids,
            parameters=dict(candidate.normalized_parameters),
            affected_constraint_ids=affected,
            validation_status="valid" if not reasons else "invalid",
            evidence_refs=candidate.evidence_refs,
            reason_codes=tuple(reasons),
        )

    def repair_radius(self, patch: ModelPatch) -> RepairRadius:
        if not patch.is_valid:
            raise ValueError("cannot select a repair radius for an invalid model patch")
        return _RADIUS_BY_PATCH[PatchType(patch.patch_type)]

    def build_repair_request(
        self,
        *,
        parent: PlanArtifactV2,
        candidate: SemanticInterpretationCandidate,
        patch: ModelPatch,
        request_id: str,
        test_only: bool,
        granted_constraint_ids: tuple[str, ...] = (),
        time_limit_seconds: float | None = None,
    ) -> RepairRequest:
        if not patch.is_valid:
            raise ValueError("invalid model patches cannot enter a repair request")
        constraints = _confirmed_constraints(parent, patch)
        constraints["parent_plan_id"] = parent.plan_id
        constraints["target_stop_ids"] = patch.target_ids
        constraints["allowed_radii"] = (self.repair_radius(patch).value,)
        constraints["semantic_interpretation_id"] = candidate.interpretation_id
        constraints["model_patch_id"] = patch.patch_id
        constraints["test_only"] = bool(test_only)
        if time_limit_seconds is not None:
            constraints["solver_time_limit_seconds"] = float(time_limit_seconds)
        granted = set(granted_constraint_ids)
        if granted.intersection(patch.affected_constraint_ids):
            constraints["allow_booked_relaxation"] = True
            constraints["permission_constraint_ids"] = tuple(sorted(granted))

        parsed = ParsedRepairIntent(
            raw_text=f"confirmed interpretation {candidate.interpretation_id}",
            confirmed=True,
            confidence=float(candidate.support_score or 0.0),
            must_keep=patch.target_ids if patch.patch_type == PatchType.PRESERVE_NAMED_STOP.value else (),
            must_delete=patch.target_ids if patch.patch_type == PatchType.ADJUST_ORDINAL_PACE.value else (),
            parser_name="permission_aware_rule_based_compiler",
        )
        return RepairRequest(
            request_id=request_id,
            baseline_route=tuple(dict(stop) for stop in parent.selected_stops),
            user_intent=f"confirmed semantic interpretation {candidate.interpretation_id}",
            evidence_records=tuple({"evidence_ref": ref} for ref in patch.evidence_refs),
            tolerance_profile=_tolerance_profile(patch),
            confirmed_constraints=constraints,
            parsed_intent=parsed,
        )


def confirmed_candidate(candidate: SemanticInterpretationCandidate) -> SemanticInterpretationCandidate:
    return replace(candidate, confirmed=True)


def _known_target_ids(parent: PlanArtifactV2) -> set[str]:
    targets = set(parent.sequence)
    targets.update(str(stop.get("stop_id") or stop.get("poi_id") or "") for stop in parent.selected_stops)
    targets.update(str(value) for value in parent.lodging_assignments.values())
    for raw in parent.owned_constraints:
        constraint = OwnedConstraint.from_record(dict(raw))
        targets.add(constraint.target_id)
        targets.add(constraint.constraint_id)
    targets.discard("")
    return targets


def _affected_constraint_ids(parent: PlanArtifactV2, target_ids: tuple[str, ...]) -> tuple[str, ...]:
    target_set = set(target_ids)
    affected: list[str] = []
    for raw in parent.owned_constraints:
        constraint = OwnedConstraint.from_record(dict(raw))
        if constraint.target_id in target_set or constraint.constraint_id in target_set:
            affected.append(constraint.constraint_id)
    return tuple(affected)


def _confirmed_constraints(parent: PlanArtifactV2, patch: ModelPatch) -> dict[str, Any]:
    patch_type = PatchType(patch.patch_type)
    parameters = dict(patch.parameters)
    constraints: dict[str, Any] = {"affected_constraint_ids": patch.affected_constraint_ids}
    if patch_type == PatchType.PRESERVE_NAMED_STOP:
        constraints["must_keep"] = patch.target_ids
    elif patch_type == PatchType.SAME_DAY_REPLACEMENT:
        constraints["must_delete"] = patch.target_ids
        if parameters.get("replacement_stop_id"):
            constraints["must_include"] = (str(parameters["replacement_stop_id"]),)
    elif patch_type == PatchType.DAY_MOVE and patch.target_ids:
        constraints["move_stops"] = {target: int(parameters["to_day"]) for target in patch.target_ids}
    elif patch_type == PatchType.LODGING_CHANGE:
        constraints["requested_lodging_change"] = parameters
    elif patch_type == PatchType.PROBE_PERMISSION_GATED_RELAXATION:
        constraints["probe_relax_constraint_ids"] = tuple(
            parameters.get("constraint_ids", patch.affected_constraint_ids)
        )
    elif patch_type == PatchType.ADJUST_ORDINAL_PACE:
        constraints["pace_ordinal_reduction"] = int(parameters.get("ordinal_reduction", 1))
    elif patch_type == PatchType.REDUCE_WALKING_BURDEN:
        constraints["walking_burden_reduction"] = parameters
    elif patch_type == PatchType.REDUCE_DRIVING_BURDEN:
        constraints["driving_burden_reduction"] = parameters
    elif patch_type in {PatchType.ADD_MAXIMUM_ARRIVAL_TIME, PatchType.REQUIRE_EARLIER_LODGING_ARRIVAL}:
        constraints["lodging_arrival_requirement"] = parameters
    return constraints


def _tolerance_profile(patch: ModelPatch) -> dict[str, float]:
    profile: dict[str, float] = {}
    for key in ("max_daily_travel_minutes", "max_walking_minutes"):
        if key in patch.parameters:
            profile[key] = float(patch.parameters[key])
    return profile


def _json_compatible(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_compatible(item) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return all(_json_compatible(item) for item in value)
    return False


def _parameters_well_formed(parameters: dict[str, Any]) -> bool:
    numeric_keys = (
        "ordinal_reduction",
        "target_minutes_reduction",
        "max_walking_minutes",
        "max_daily_travel_minutes",
        "arrival_shift_minutes",
        "monetary_cost_delta",
    )
    if any(not _finite_number(parameters[key]) for key in numeric_keys if key in parameters):
        return False
    return all(
        isinstance(parameters[key], int) and not isinstance(parameters[key], bool) and parameters[key] >= 1
        for key in ("day", "to_day")
        if key in parameters
    )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
