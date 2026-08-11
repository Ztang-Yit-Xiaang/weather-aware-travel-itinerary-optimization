"""RouteMatrix-backed day-route subproblem for repair candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..research_artifacts import stable_content_hash
from ..routing import RouteMatrix, RouteMatrixError, RouteMatrixMissing
from .master_model import RepairModel, RepairSolution


@dataclass(frozen=True)
class DayRouteSolverConfig:
    max_day_minutes: float = 480.0
    day_start_time: str | int | float = "09:00"
    default_visit_minutes: float = 60.0
    enforce_opening_windows: bool = True
    strict_route_matrix: bool = False
    start_anchor_by_day: dict[int, str] = field(default_factory=dict)
    end_anchor_by_day: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteSequenceEvaluation:
    """Static route/schedule feasibility for a supplied day sequence."""

    day: int
    stop_sequence: tuple[str, ...]
    route_sequence: tuple[str, ...]
    route_pairs: tuple[tuple[str, str], ...]
    travel_minutes: float
    visit_minutes: float
    waiting_minutes: float
    total_minutes: float
    feasible: bool
    violations: tuple[str, ...]
    route_evidence_ids: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DayRouteCandidate:
    day: int
    stop_sequence: tuple[str, ...]
    route_sequence: tuple[str, ...]
    route_pairs: tuple[tuple[str, str], ...]
    travel_minutes: float
    visit_minutes: float
    waiting_minutes: float
    total_minutes: float
    feasible: bool
    violations: tuple[str, ...]
    repair_solution: RepairSolution
    route_evidence_ids: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["repair_solution"] = {
            "selected_stop_ids": list(self.repair_solution.selected_stop_ids),
            "day_assignments": self.repair_solution.day_assignments,
            "lodging_assignments": self.repair_solution.lodging_assignments,
            "relaxed_constraint_ids": list(self.repair_solution.relaxed_constraint_ids),
            "route_ids_by_day": {str(day): route_id for day, route_id in self.repair_solution.route_ids_by_day.items()},
            "metadata": self.repair_solution.metadata,
        }
        return record


@dataclass(frozen=True)
class DayRouteSubproblemResult:
    model_id: str
    day: int
    matrix_id: str
    candidates: tuple[DayRouteCandidate, ...]
    required_route_pairs: tuple[tuple[str, str], ...]
    publication_mode: bool = False

    @property
    def feasible_candidates(self) -> tuple[DayRouteCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.feasible)

    def to_record(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "day": self.day,
            "matrix_id": self.matrix_id,
            "publication_mode": self.publication_mode,
            "required_route_pairs": [list(pair) for pair in self.required_route_pairs],
            "candidates": [candidate.to_record() for candidate in self.candidates],
        }


class DayRouteSolver:
    """Generate and score one-day repair candidates with route matrix evidence."""

    def __init__(
        self,
        route_matrix: RouteMatrix,
        *,
        config: DayRouteSolverConfig | None = None,
    ) -> None:
        if route_matrix.empty:
            raise RouteMatrixMissing("day route solver requires a non-empty RouteMatrix")
        self.route_matrix = route_matrix
        self.config = config or DayRouteSolverConfig()

    def solve(
        self,
        model: RepairModel,
        *,
        day: int,
        candidate_sequences: tuple[tuple[str, ...], ...] | list[tuple[str, ...]] | None = None,
    ) -> DayRouteSubproblemResult:
        return solve_day_route_subproblem(
            model,
            self.route_matrix,
            day=day,
            config=self.config,
            candidate_sequences=tuple(candidate_sequences or ()),
        )


def solve_day_route_subproblem(
    model: RepairModel,
    route_matrix: RouteMatrix,
    *,
    day: int,
    config: DayRouteSolverConfig | None = None,
    candidate_sequences: tuple[tuple[str, ...], ...] | list[tuple[str, ...]] | None = None,
) -> DayRouteSubproblemResult:
    if route_matrix.empty:
        raise RouteMatrixMissing("day route solver requires a non-empty RouteMatrix")
    config = config or DayRouteSolverConfig()
    sequences = _candidate_sequences(model, day, tuple(candidate_sequences or ()))
    candidates: list[DayRouteCandidate] = []
    for sequence in sequences:
        candidates.append(_evaluate_sequence(model, route_matrix, day, tuple(sequence), config))
    required_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        for pair in candidate.route_pairs:
            if pair not in seen:
                seen.add(pair)
                required_pairs.append(pair)
    return DayRouteSubproblemResult(
        model_id=model.model_id,
        day=day,
        matrix_id=route_matrix.matrix_id,
        candidates=tuple(candidates),
        required_route_pairs=tuple(required_pairs),
        publication_mode=config.strict_route_matrix,
    )


def evaluate_route_sequence(
    route_matrix: RouteMatrix,
    *,
    day: int,
    stop_sequence: tuple[str, ...],
    stop_records: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    config: DayRouteSolverConfig | None = None,
) -> RouteSequenceEvaluation:
    """Evaluate a supplied sequence using only static route/schedule inputs."""

    config = config or DayRouteSolverConfig()
    lookup = {
        str(record.get("stop_id") or record.get("poi_id") or record.get("name") or "").strip(): dict(record)
        for record in stop_records
    }
    sequence = tuple(str(stop_id) for stop_id in stop_sequence)
    route_parts: list[str] = []
    if config.start_anchor_by_day.get(day):
        route_parts.append(str(config.start_anchor_by_day[day]))
    route_parts.extend(sequence)
    if config.end_anchor_by_day.get(day):
        route_parts.append(str(config.end_anchor_by_day[day]))
    route_sequence = tuple(route_parts)
    route_pairs = tuple(zip(route_sequence[:-1], route_sequence[1:], strict=False))
    violations: list[str] = []
    travel_minutes = 0.0
    visit_minutes = 0.0
    waiting_minutes = 0.0
    current_time = float(_parse_minutes(config.day_start_time) or 0)
    evidence_ids: list[str] = []
    current_id = route_sequence[0] if route_sequence and route_sequence[0] not in sequence else None
    for stop_id in sequence:
        if current_id:
            leg_minutes, evidence_id = _route_leg_minutes(
                route_matrix,
                current_id,
                stop_id,
                strict=config.strict_route_matrix,
            )
            if evidence_id:
                evidence_ids.append(evidence_id)
            if leg_minutes is None:
                violations.append(f"missing_or_invalid_route:{current_id}->{stop_id}")
                leg_minutes = 0.0
            travel_minutes += float(leg_minutes)
            current_time += float(leg_minutes)
        stop = lookup.get(stop_id, {"stop_id": stop_id})
        if stop_id not in lookup:
            violations.append(f"unknown_stop:{stop_id}")
        window_start, window_end = _opening_window(stop)
        if config.enforce_opening_windows and window_start is not None and current_time < window_start:
            waiting_minutes += float(window_start - current_time)
            current_time = float(window_start)
        if config.enforce_opening_windows and window_end is not None and current_time > window_end:
            violations.append(f"opening_window_missed:{stop_id}")
        duration = _visit_duration(stop, config.default_visit_minutes)
        visit_minutes += duration
        current_time += duration
        current_id = stop_id
    end_anchor = route_sequence[-1] if route_sequence and route_sequence[-1] not in sequence else None
    if end_anchor and current_id:
        leg_minutes, evidence_id = _route_leg_minutes(
            route_matrix,
            current_id,
            end_anchor,
            strict=config.strict_route_matrix,
        )
        if evidence_id:
            evidence_ids.append(evidence_id)
        if leg_minutes is None:
            violations.append(f"missing_or_invalid_route:{current_id}->{end_anchor}")
            leg_minutes = 0.0
        travel_minutes += float(leg_minutes)
    total_minutes = travel_minutes + visit_minutes + waiting_minutes
    if total_minutes > config.max_day_minutes:
        violations.append(f"day_time_exceeded:{day}")
    return RouteSequenceEvaluation(
        day=day,
        stop_sequence=sequence,
        route_sequence=route_sequence,
        route_pairs=route_pairs,
        travel_minutes=float(travel_minutes),
        visit_minutes=float(visit_minutes),
        waiting_minutes=float(waiting_minutes),
        total_minutes=float(total_minutes),
        feasible=not violations,
        violations=tuple(dict.fromkeys(violations)),
        route_evidence_ids=tuple(evidence_ids),
    )


def _evaluate_sequence(
    model: RepairModel,
    route_matrix: RouteMatrix,
    day: int,
    sequence: tuple[str, ...],
    config: DayRouteSolverConfig,
) -> DayRouteCandidate:
    stop_lookup = _stop_lookup(model)
    route_sequence = _route_sequence(model, day, sequence, config)
    route_pairs = tuple(zip(route_sequence[:-1], route_sequence[1:], strict=False))
    violations: list[str] = []
    travel_minutes = 0.0
    visit_minutes = 0.0
    waiting_minutes = 0.0
    current_time = float(_parse_minutes(config.day_start_time) or 0)
    evidence_ids: list[str] = []

    current_id = route_sequence[0] if route_sequence and route_sequence[0] not in sequence else None
    for stop_id in sequence:
        if current_id:
            leg_minutes, leg_evidence_id = _route_leg_minutes(
                route_matrix,
                current_id,
                stop_id,
                strict=config.strict_route_matrix,
            )
            if leg_evidence_id:
                evidence_ids.append(leg_evidence_id)
            if leg_minutes is None:
                violations.append(f"missing_or_invalid_route:{current_id}->{stop_id}")
                leg_minutes = 0.0
            travel_minutes += float(leg_minutes)
            current_time += float(leg_minutes)
        stop = stop_lookup.get(stop_id, {"stop_id": stop_id})
        window_start, window_end = _opening_window(stop)
        if config.enforce_opening_windows and window_start is not None and current_time < window_start:
            waiting_minutes += float(window_start - current_time)
            current_time = float(window_start)
        if config.enforce_opening_windows and window_end is not None and current_time > window_end:
            violations.append(f"opening_window_missed:{stop_id}")
        duration = _visit_duration(stop, config.default_visit_minutes)
        visit_minutes += duration
        current_time += duration
        current_id = stop_id
    end_anchor = route_sequence[-1] if route_sequence and route_sequence[-1] not in sequence else None
    if end_anchor and current_id:
        leg_minutes, leg_evidence_id = _route_leg_minutes(
            route_matrix,
            current_id,
            end_anchor,
            strict=config.strict_route_matrix,
        )
        if leg_evidence_id:
            evidence_ids.append(leg_evidence_id)
        if leg_minutes is None:
            violations.append(f"missing_or_invalid_route:{current_id}->{end_anchor}")
            leg_minutes = 0.0
        travel_minutes += float(leg_minutes)
        current_time += float(leg_minutes)

    total_minutes = travel_minutes + visit_minutes + waiting_minutes
    if total_minutes > config.max_day_minutes:
        violations.append(f"day_time_exceeded:{day}")

    solution = _solution_for_sequence(model, day, sequence, route_matrix)
    violations.extend(model.validate_solution(solution))
    for stop_id in sequence:
        if stop_id not in stop_lookup:
            violations.append(f"unknown_stop:{stop_id}")
        assigned_day = solution.day_assignments.get(stop_id)
        if assigned_day != day:
            violations.append(f"fixed_day_assignment_mismatch:{stop_id}")

    feasible = not violations
    return DayRouteCandidate(
        day=day,
        stop_sequence=sequence,
        route_sequence=route_sequence,
        route_pairs=route_pairs,
        travel_minutes=float(travel_minutes),
        visit_minutes=float(visit_minutes),
        waiting_minutes=float(waiting_minutes),
        total_minutes=float(total_minutes),
        feasible=feasible,
        violations=tuple(dict.fromkeys(violations)),
        repair_solution=solution,
        route_evidence_ids=tuple(evidence_ids),
    )


def _candidate_sequences(
    model: RepairModel,
    day: int,
    supplied_sequences: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    if supplied_sequences:
        return _dedupe_sequences(supplied_sequences)

    parent_day_sequence = _parent_day_sequence(model, day)
    sequences: list[tuple[str, ...]] = [parent_day_sequence]
    candidate_ids = tuple(
        str(stop.get("stop_id"))
        for stop in model.metadata.get("candidate_stops", ())
        if _coerce_int(stop.get("day")) in (None, day)
    )
    editable_parent_stops = tuple(stop_id for stop_id in parent_day_sequence if stop_id in set(model.neighborhood.editable_stop_ids))
    for candidate_id in candidate_ids:
        for replaced_stop in editable_parent_stops:
            if replaced_stop in set(model.metadata.get("locked_stop_ids", ())):
                continue
            sequence = tuple(candidate_id if stop_id == replaced_stop else stop_id for stop_id in parent_day_sequence)
            sequences.append(sequence)
        if candidate_id not in parent_day_sequence:
            sequences.append((*parent_day_sequence, candidate_id))
    return _dedupe_sequences(tuple(sequences))


def _solution_for_sequence(
    model: RepairModel,
    day: int,
    sequence: tuple[str, ...],
    route_matrix: RouteMatrix,
) -> RepairSolution:
    parent = model.metadata["parent_plan"]
    parent_day_by_stop = model.metadata.get("parent_day_by_stop", {})
    parent_day_sequence = set(_parent_day_sequence(model, day))
    selected: list[str] = []
    for stop_id in parent.sequence:
        if parent_day_by_stop.get(stop_id) == day:
            continue
        selected.append(stop_id)
    selected.extend(sequence)
    day_assignments = {
        str(stop_id): int(parent_day)
        for stop_id, parent_day in parent_day_by_stop.items()
        if stop_id in selected
    }
    for stop_id in sequence:
        day_assignments[str(stop_id)] = int(day)
    lodging_assignments = {str(raw_day): str(lodging) for raw_day, lodging in parent.lodging_assignments.items()}
    route_ids_by_day = {int(raw_day): str(route_id) for raw_day, route_id in parent.route_ids_by_day.items()}
    if tuple(sequence) != tuple(_parent_day_sequence(model, day)):
        route_ids_by_day[int(day)] = f"route_matrix:{route_matrix.matrix_id}:day:{day}:{stable_content_hash(sequence)}"
    metadata = {
        "candidate_id": f"day_route:{day}:{stable_content_hash(sequence)}",
        "day_sequences": {str(day): tuple(sequence)},
        "replaced_stop_ids": tuple(stop_id for stop_id in parent_day_sequence if stop_id not in set(sequence)),
    }
    return RepairSolution(
        selected_stop_ids=tuple(selected),
        day_assignments=day_assignments,
        lodging_assignments=lodging_assignments,
        route_ids_by_day=route_ids_by_day,
        metadata=metadata,
    )


def _route_leg_minutes(
    route_matrix: RouteMatrix,
    origin_id: str,
    destination_id: str,
    *,
    strict: bool,
) -> tuple[float | None, str]:
    try:
        cell = route_matrix.cell(origin_id, destination_id)
        if strict:
            cell.require_publication_eligible()
        return cell.require_duration_s() / 60.0, cell.route_leg_id
    except RouteMatrixError:
        if strict:
            raise
        return None, ""


def _route_sequence(
    model: RepairModel,
    day: int,
    sequence: tuple[str, ...],
    config: DayRouteSolverConfig,
) -> tuple[str, ...]:
    start_anchor = config.start_anchor_by_day.get(day) or _anchor_from_day_stops(model, day, "route_start_id", "route_start_name")
    end_anchor = config.end_anchor_by_day.get(day) or _anchor_from_day_stops(model, day, "route_end_id", "route_end_name")
    parts: list[str] = []
    if start_anchor:
        parts.append(str(start_anchor))
    parts.extend(sequence)
    if end_anchor:
        parts.append(str(end_anchor))
    return tuple(parts)


def _anchor_from_day_stops(model: RepairModel, day: int, *keys: str) -> str:
    for stop in model.metadata.get("parent_selected_stops", ()):
        if _coerce_int(stop.get("day")) != day:
            continue
        for key in keys:
            value = stop.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _parent_day_sequence(model: RepairModel, day: int) -> tuple[str, ...]:
    parent = model.metadata["parent_plan"]
    parent_day_by_stop = model.metadata.get("parent_day_by_stop", {})
    return tuple(str(stop_id) for stop_id in parent.sequence if parent_day_by_stop.get(str(stop_id)) == day)


def _stop_lookup(model: RepairModel) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for stop in (*model.metadata.get("parent_selected_stops", ()), *model.metadata.get("candidate_stops", ())):
        stop_id = str(stop.get("stop_id") or stop.get("poi_id") or stop.get("name") or "").strip()
        if stop_id:
            lookup[stop_id] = dict(stop)
    return lookup


def _opening_window(stop: dict[str, Any]) -> tuple[int | None, int | None]:
    start = _parse_minutes(
        _first_nonempty(stop, "opening_start", "window_start", "start_time", "earliest_start", "open_time")
    )
    end = _parse_minutes(
        _first_nonempty(stop, "opening_end", "window_end", "latest_start", "close_time")
    )
    return start, end


def _visit_duration(stop: dict[str, Any], default: float) -> float:
    for key in ("visit_duration_minutes", "duration_minutes", "service_minutes", "dwell_minutes"):
        value = stop.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return max(0.0, float(value))
        except Exception:
            continue
    return float(default)


def _parse_minutes(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        if ":" in text:
            hours, minutes = text.split(":", 1)
            return int(hours) * 60 + int(minutes[:2])
        return int(float(text))
    except Exception:
        return None


def _first_nonempty(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _dedupe_sequences(sequences: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
    seen: set[tuple[str, ...]] = set()
    result: list[tuple[str, ...]] = []
    for sequence in sequences:
        normalized = tuple(str(stop_id) for stop_id in sequence if str(stop_id).strip())
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)
