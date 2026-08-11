"""Route-aware POI insertion burden without recommendation overclaiming.

This module is a product-session calculation boundary.  Geographic proximity
may establish discovery context, but only three road-validated directed legs
can establish route-nearness.  Ranking and recommendation remain owned by an
independent evaluator record rather than this calculator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..routing.matrix import RouteMatrix, RouteMatrixError

SCHEMA_VERSION = "product-candidate-burden-v1"
MAX_CANDIDATE_RESULTS = 50

_PRECHECK_STATUSES = frozenset({"passed", "failed", "unavailable"})
_CANDIDATE_STATES = frozenset(
    {
        "unavailable",
        "nearby",
        "route_near",
        "likely_feasible",
        "evaluated_feasible",
        "recommended",
    }
)


class CandidateBurdenError(ValueError):
    """Stable candidate-burden contract error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class FastFeasibilityPrecheckV1:
    """Non-certifying fast schedule/context precheck."""

    status: str
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _PRECHECK_STATUSES:
            raise CandidateBurdenError("precheck_status_invalid")
        _validate_evidence_refs(self.evidence_refs)
        if self.status in {"passed", "failed"} and not self.evidence_refs:
            raise CandidateBurdenError("precheck_evidence_required")


@dataclass(frozen=True, slots=True)
class EvaluatorCandidateEvidenceV1:
    """Independent evaluator authority for feasibility and ranking labels."""

    owner: str
    decision_eligible: bool
    ranking_eligible: bool
    evaluator_rank: int | None
    recommended: bool
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.owner != "independent_evaluator":
            raise CandidateBurdenError("evaluator_owner_invalid")
        if not isinstance(self.decision_eligible, bool):
            raise CandidateBurdenError("decision_eligibility_invalid")
        if not isinstance(self.ranking_eligible, bool):
            raise CandidateBurdenError("ranking_eligibility_invalid")
        if not isinstance(self.recommended, bool):
            raise CandidateBurdenError("recommendation_flag_invalid")
        _validate_evidence_refs(self.evidence_refs)
        if not self.evidence_refs:
            raise CandidateBurdenError("evaluator_evidence_required")
        if self.ranking_eligible and not self.decision_eligible:
            raise CandidateBurdenError("ranking_requires_decision_eligibility")
        if self.ranking_eligible:
            if (
                isinstance(self.evaluator_rank, bool)
                or not isinstance(self.evaluator_rank, int)
                or self.evaluator_rank < 1
            ):
                raise CandidateBurdenError("evaluator_rank_required")
        elif self.evaluator_rank is not None:
            raise CandidateBurdenError("rank_without_ranking_eligibility")
        if self.recommended and not self.ranking_eligible:
            raise CandidateBurdenError("recommendation_requires_ranking_eligibility")


@dataclass(frozen=True, slots=True)
class CandidateInsertionBurdenV1:
    """Evidence-preserving assessment for one insertion slot.

    All time fields are minutes and all distance fields are meters.  A missing
    visit, parking, walking, or waiting component keeps ``total_insertion``
    missing; it is never silently treated as zero.
    """

    schema_version: str
    context_kind: str
    candidate_id: str
    place_id: str
    predecessor_id: str
    successor_id: str
    replacement_target_id: str | None
    baseline_route_leg_ids: tuple[str, ...]
    baseline_travel_minutes: float | None
    baseline_travel_distance_m: float | None
    geographic_distance_m: float | None
    predecessor_to_candidate_minutes: float | None
    candidate_to_successor_minutes: float | None
    predecessor_to_successor_minutes: float | None
    marginal_travel_minutes: float | None
    marginal_travel_distance_m: float | None
    visit_minutes: float | None
    parking_minutes: float | None
    walking_minutes: float | None
    waiting_minutes: float | None
    total_insertion_minutes: float | None
    nearby: bool
    route_near: bool
    likely_feasible: bool
    evaluated_feasible: bool
    ranking_eligible: bool
    recommended: bool
    candidate_state: str
    evaluator_rank: int | None
    blocking_codes: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    evaluator_evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise CandidateBurdenError("candidate_burden_schema_invalid")
        if self.context_kind not in {"insertion", "replacement"}:
            raise CandidateBurdenError("candidate_context_kind_invalid")
        if self.context_kind == "replacement" and not self.replacement_target_id:
            raise CandidateBurdenError("replacement_target_required")
        if self.context_kind == "insertion" and self.replacement_target_id is not None:
            raise CandidateBurdenError("replacement_target_unexpected")
        if self.candidate_state not in _CANDIDATE_STATES:
            raise CandidateBurdenError("candidate_state_invalid")
        if any(
            type(value) is not bool
            for value in (
                self.nearby,
                self.route_near,
                self.likely_feasible,
                self.evaluated_feasible,
                self.ranking_eligible,
                self.recommended,
            )
        ):
            raise CandidateBurdenError("candidate_state_flag_invalid")
        _validate_evidence_refs(self.evidence_refs)
        _validate_evidence_refs(self.evaluator_evidence_refs)
        if not set(self.evaluator_evidence_refs).issubset(self.evidence_refs):
            raise CandidateBurdenError("evaluator_evidence_not_grounded")
        if self.ranking_eligible and not self.evaluated_feasible:
            raise CandidateBurdenError("ranking_requires_evaluated_feasibility")
        if self.recommended and not self.ranking_eligible:
            raise CandidateBurdenError("recommendation_requires_ranking_eligibility")
        if (self.evaluated_feasible or self.ranking_eligible or self.recommended) and not (
            self.evaluator_evidence_refs
        ):
            raise CandidateBurdenError("evaluator_evidence_required")
        if self.ranking_eligible:
            if (
                isinstance(self.evaluator_rank, bool)
                or not isinstance(self.evaluator_rank, int)
                or self.evaluator_rank < 1
            ):
                raise CandidateBurdenError("evaluator_rank_required")
        elif self.evaluator_rank is not None:
            raise CandidateBurdenError("rank_without_ranking_eligibility")
        expected_state = _candidate_state(
            nearby=self.nearby,
            route_near=self.route_near,
            likely_feasible=self.likely_feasible,
            evaluated_feasible=self.evaluated_feasible,
            recommended=self.recommended,
        )
        if self.candidate_state != expected_state:
            raise CandidateBurdenError("candidate_state_inconsistent")


def assess_candidate_insertion(
    *,
    candidate_id: str,
    place_id: str,
    predecessor_id: str,
    successor_id: str,
    route_matrix: RouteMatrix,
    geographic_distance_m: float | None,
    visit_minutes: float | None,
    parking_minutes: float | None,
    walking_minutes: float | None,
    waiting_minutes: float | None,
    nearby_radius_m: float,
    maximum_detour_minutes: float,
    precheck: FastFeasibilityPrecheckV1 | None = None,
    evaluator: EvaluatorCandidateEvidenceV1 | None = None,
) -> CandidateInsertionBurdenV1:
    """Assess one candidate using directed road evidence for ``i -> c -> j``."""

    return _assess_candidate_context(
        context_kind="insertion",
        replacement_target_id=None,
        baseline_legs=(("predecessor_successor", predecessor_id, successor_id),),
        candidate_id=candidate_id,
        place_id=place_id,
        predecessor_id=predecessor_id,
        successor_id=successor_id,
        route_matrix=route_matrix,
        geographic_distance_m=geographic_distance_m,
        visit_minutes=visit_minutes,
        parking_minutes=parking_minutes,
        walking_minutes=walking_minutes,
        waiting_minutes=waiting_minutes,
        nearby_radius_m=nearby_radius_m,
        maximum_detour_minutes=maximum_detour_minutes,
        precheck=precheck,
        evaluator=evaluator,
    )


def assess_candidate_replacement(
    *,
    candidate_id: str,
    place_id: str,
    replacement_target_id: str,
    predecessor_id: str,
    successor_id: str,
    route_matrix: RouteMatrix,
    geographic_distance_m: float | None,
    visit_minutes: float | None,
    parking_minutes: float | None,
    walking_minutes: float | None,
    waiting_minutes: float | None,
    nearby_radius_m: float,
    maximum_detour_minutes: float,
    precheck: FastFeasibilityPrecheckV1 | None = None,
    evaluator: EvaluatorCandidateEvidenceV1 | None = None,
) -> CandidateInsertionBurdenV1:
    """Assess replacement travel against the exact two-leg target baseline.

    The travel delta is ``T(i,c) + T(c,j) - T(i,target) - T(target,j)``.
    It intentionally does not compare a replacement to a direct ``i -> j``
    shortcut, because that would erase the travel already owned by the target.
    """

    if not isinstance(replacement_target_id, str) or not replacement_target_id.strip():
        raise CandidateBurdenError("replacement_target_invalid")
    return _assess_candidate_context(
        context_kind="replacement",
        replacement_target_id=replacement_target_id,
        baseline_legs=(
            ("predecessor_target", predecessor_id, replacement_target_id),
            ("target_successor", replacement_target_id, successor_id),
        ),
        candidate_id=candidate_id,
        place_id=place_id,
        predecessor_id=predecessor_id,
        successor_id=successor_id,
        route_matrix=route_matrix,
        geographic_distance_m=geographic_distance_m,
        visit_minutes=visit_minutes,
        parking_minutes=parking_minutes,
        walking_minutes=walking_minutes,
        waiting_minutes=waiting_minutes,
        nearby_radius_m=nearby_radius_m,
        maximum_detour_minutes=maximum_detour_minutes,
        precheck=precheck,
        evaluator=evaluator,
    )


def _assess_candidate_context(
    *,
    context_kind: str,
    replacement_target_id: str | None,
    baseline_legs: tuple[tuple[str, str, str], ...],
    candidate_id: str,
    place_id: str,
    predecessor_id: str,
    successor_id: str,
    route_matrix: RouteMatrix,
    geographic_distance_m: float | None,
    visit_minutes: float | None,
    parking_minutes: float | None,
    walking_minutes: float | None,
    waiting_minutes: float | None,
    nearby_radius_m: float,
    maximum_detour_minutes: float,
    precheck: FastFeasibilityPrecheckV1 | None,
    evaluator: EvaluatorCandidateEvidenceV1 | None,
) -> CandidateInsertionBurdenV1:
    """Shared evidence-preserving assessment for insertion and replacement."""

    identifiers = (candidate_id, place_id, predecessor_id, successor_id)
    if any(not isinstance(value, str) or not value.strip() for value in identifiers):
        raise CandidateBurdenError("candidate_identifier_invalid")
    geographic_distance = _optional_nonnegative(
        geographic_distance_m, "geographic_distance_invalid"
    )
    nearby_radius = _positive(nearby_radius_m, "nearby_radius_invalid")
    maximum_detour = _nonnegative(maximum_detour_minutes, "maximum_detour_invalid")
    components = (
        _optional_nonnegative(visit_minutes, "visit_minutes_invalid"),
        _optional_nonnegative(parking_minutes, "parking_minutes_invalid"),
        _optional_nonnegative(walking_minutes, "walking_minutes_invalid"),
        _optional_nonnegative(waiting_minutes, "waiting_minutes_invalid"),
    )

    candidate_legs = (
        ("predecessor_candidate", predecessor_id, place_id),
        ("candidate_successor", place_id, successor_id),
    )
    requested_legs = (*candidate_legs, *baseline_legs)
    resolved: dict[str, object] = {}
    blocking_codes: list[str] = []
    evidence_refs: list[str] = [route_matrix.matrix_id, route_matrix.context_snapshot_id]
    for label, origin_id, destination_id in requested_legs:
        try:
            cell = route_matrix.cell(origin_id, destination_id)
            cell.require_publication_eligible()
        except (RouteMatrixError, ValueError):
            blocking_codes.append(f"route_leg_{label}_unavailable")
            continue
        resolved[label] = cell
        evidence_refs.extend((cell.route_leg_id, cell.query_hash))

    marginal_minutes: float | None = None
    marginal_distance: float | None = None
    predecessor_candidate_minutes: float | None = None
    candidate_successor_minutes: float | None = None
    predecessor_successor_minutes: float | None = None
    baseline_minutes: float | None = None
    baseline_distance: float | None = None
    baseline_route_leg_ids: tuple[str, ...] = ()
    if len(resolved) == len(requested_legs):
        predecessor_candidate = resolved["predecessor_candidate"]
        candidate_successor = resolved["candidate_successor"]
        predecessor_candidate_minutes = predecessor_candidate.require_duration_s() / 60.0
        candidate_successor_minutes = candidate_successor.require_duration_s() / 60.0
        baseline_cells = tuple(resolved[label] for label, _, _ in baseline_legs)
        baseline_minutes = sum(cell.require_duration_s() / 60.0 for cell in baseline_cells)
        baseline_distance = sum(cell.require_distance_m() for cell in baseline_cells)
        baseline_route_leg_ids = tuple(cell.route_leg_id for cell in baseline_cells)
        if context_kind == "insertion":
            predecessor_successor_minutes = baseline_minutes
        marginal_minutes = (
            predecessor_candidate_minutes
            + candidate_successor_minutes
            - baseline_minutes
        )
        marginal_distance = (
            predecessor_candidate.require_distance_m()
            + candidate_successor.require_distance_m()
            - baseline_distance
        )

    missing_component_codes = tuple(
        code
        for value, code in zip(
            components,
            (
                "visit_duration_unavailable",
                "parking_time_unavailable",
                "walking_time_unavailable",
                "waiting_time_unavailable",
            ),
            strict=True,
        )
        if value is None
    )
    blocking_codes.extend(missing_component_codes)
    total_insertion = (
        marginal_minutes + sum(value for value in components if value is not None)
        if marginal_minutes is not None and not missing_component_codes
        else None
    )

    nearby = geographic_distance is not None and geographic_distance <= nearby_radius
    route_near = marginal_minutes is not None and marginal_minutes <= maximum_detour
    if marginal_minutes is not None and not route_near:
        blocking_codes.append("route_detour_exceeds_limit")

    likely_feasible = bool(route_near and precheck is not None and precheck.status == "passed")
    if precheck is None or precheck.status == "unavailable":
        blocking_codes.append("schedule_precheck_unavailable")
    elif precheck.status == "failed":
        blocking_codes.append("schedule_precheck_failed")
    if precheck is not None:
        evidence_refs.extend(precheck.evidence_refs)

    evaluated_feasible = bool(
        marginal_minutes is not None and evaluator is not None and evaluator.decision_eligible
    )
    ranking_eligible = bool(
        evaluated_feasible and evaluator is not None and evaluator.ranking_eligible
    )
    recommended = bool(
        ranking_eligible and evaluator is not None and evaluator.recommended
    )
    if evaluator is None:
        blocking_codes.append("evaluator_evidence_unavailable")
    else:
        evidence_refs.extend(evaluator.evidence_refs)
        if not evaluator.decision_eligible:
            blocking_codes.append("evaluator_decision_ineligible")
        elif not evaluator.ranking_eligible:
            blocking_codes.append("evaluator_ranking_ineligible")

    candidate_state = _candidate_state(
        nearby=nearby,
        route_near=route_near,
        likely_feasible=likely_feasible,
        evaluated_feasible=evaluated_feasible,
        recommended=recommended,
    )

    return CandidateInsertionBurdenV1(
        schema_version=SCHEMA_VERSION,
        context_kind=context_kind,
        candidate_id=candidate_id,
        place_id=place_id,
        predecessor_id=predecessor_id,
        successor_id=successor_id,
        replacement_target_id=replacement_target_id,
        baseline_route_leg_ids=baseline_route_leg_ids,
        baseline_travel_minutes=baseline_minutes,
        baseline_travel_distance_m=baseline_distance,
        geographic_distance_m=geographic_distance,
        predecessor_to_candidate_minutes=predecessor_candidate_minutes,
        candidate_to_successor_minutes=candidate_successor_minutes,
        predecessor_to_successor_minutes=predecessor_successor_minutes,
        marginal_travel_minutes=marginal_minutes,
        marginal_travel_distance_m=marginal_distance,
        visit_minutes=components[0],
        parking_minutes=components[1],
        walking_minutes=components[2],
        waiting_minutes=components[3],
        total_insertion_minutes=total_insertion,
        nearby=nearby,
        route_near=route_near,
        likely_feasible=likely_feasible,
        evaluated_feasible=evaluated_feasible,
        ranking_eligible=ranking_eligible,
        recommended=recommended,
        candidate_state=candidate_state,
        evaluator_rank=evaluator.evaluator_rank if ranking_eligible and evaluator else None,
        blocking_codes=_unique(tuple(blocking_codes)),
        evidence_refs=_unique(tuple(value for value in evidence_refs if value)),
        evaluator_evidence_refs=(evaluator.evidence_refs if evaluator is not None else ()),
    )


def select_bounded_candidate_top_k(
    candidates: tuple[CandidateInsertionBurdenV1, ...], *, limit: int
) -> tuple[CandidateInsertionBurdenV1, ...]:
    """Return a deterministic bounded set without inventing a recommendation.

    Evaluator-ranked candidates use the evaluator's ordinal.  Candidates that
    lack ranking eligibility use only their stable ID for ordering; their route
    burden is exposed for inspection but is not promoted into an owned rank.
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_CANDIDATE_RESULTS:
        raise CandidateBurdenError("candidate_limit_invalid")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise CandidateBurdenError("candidate_id_duplicate")
    return tuple(sorted(candidates, key=_candidate_sort_key)[:limit])


def _candidate_sort_key(candidate: CandidateInsertionBurdenV1) -> tuple[int, int, str]:
    if candidate.ranking_eligible and candidate.evaluator_rank is not None:
        return (0, candidate.evaluator_rank, candidate.candidate_id.casefold())
    return (1, 0, candidate.candidate_id.casefold())


def _candidate_state(
    *,
    nearby: bool,
    route_near: bool,
    likely_feasible: bool,
    evaluated_feasible: bool,
    recommended: bool,
) -> str:
    if recommended:
        return "recommended"
    if evaluated_feasible:
        return "evaluated_feasible"
    if likely_feasible:
        return "likely_feasible"
    if route_near:
        return "route_near"
    if nearby:
        return "nearby"
    return "unavailable"


def _validate_evidence_refs(values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(value, str) or not value.strip() or len(value) > 256 for value in values
    ):
        raise CandidateBurdenError("evidence_refs_invalid")
    if len(values) > 32 or len(set(values)) != len(values):
        raise CandidateBurdenError("evidence_refs_invalid")


def _optional_nonnegative(value: float | None, code: str) -> float | None:
    return None if value is None else _nonnegative(value, code)


def _nonnegative(value: float, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateBurdenError(code)
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise CandidateBurdenError(code)
    return converted


def _positive(value: float, code: str) -> float:
    converted = _nonnegative(value, code)
    if converted == 0:
        raise CandidateBurdenError(code)
    return converted


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
