"""Canonical data and variable indices for the D1 direct exact MILP.

This module deliberately does not optimize.  It converts the existing
solver-neutral :class:`RepairModel` and publication-eligible
:class:`RouteMatrix` into a deterministic finite index that a later Gurobi
builder can consume without enumerating stop permutations.  Keeping this
layer independent from ``gurobipy`` makes the representation auditable and
lets tiny-fixture equivalence tests run without claiming that D1 is solved.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any

from ..routing import RouteMatrix, RouteMatrixError
from .context import context_burden, stop_utility
from .day_route_solver import DayRouteSolverConfig, _visit_duration
from .master_model import RepairModel


class ExactModelDataError(ValueError):
    """Raised when the canonical exact model cannot be built truthfully."""


@dataclass(frozen=True)
class ExactStopDomainV2:
    """One selectable stop and its complete finite day domain."""

    stop_id: str
    allowed_days: tuple[int, ...]
    selection_fixed: int | None
    day_fixed: int | None
    utility: float
    contextual_burden: float
    estimated_cost: float | None
    visit_minutes: int

    @property
    def assignment_values(self) -> tuple[int | None, ...]:
        if self.selection_fixed == 0:
            return (None,)
        if self.selection_fixed == 1:
            return self.allowed_days
        return (None, *self.allowed_days)


@dataclass(frozen=True)
class ExactLodgingDomainV2:
    """One day's finite lodging-choice domain."""

    day: int
    lodging_ids: tuple[str, ...]
    fixed_lodging_id: str | None


@dataclass(frozen=True)
class ExactRouteArcV2:
    """One directed, publication-eligible route arc available to the MILP."""

    day: int
    origin_id: str
    destination_id: str
    route_leg_id: str
    duration_seconds: float
    distance_meters: float
    query_hash: str


@dataclass(frozen=True)
class ExactStructuralAssignmentV2:
    """Tiny-fixture structural assignment used only for equivalence audits."""

    selected_day_by_stop: tuple[tuple[str, int], ...]
    lodging_by_day: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class ExactVariableIndexV2:
    """Deterministic keys for the direct MILP variable families."""

    select: tuple[str, ...]
    assign_day: tuple[tuple[str, int], ...]
    assign_lodging: tuple[tuple[int, str], ...]
    route_arc: tuple[tuple[int, str, str], ...]
    first_stop: tuple[tuple[int, str], ...]
    last_stop: tuple[tuple[int, str], ...]
    order: tuple[tuple[int, str], ...]
    visit_start: tuple[tuple[int, str], ...]
    wait: tuple[tuple[int, str], ...]

    @property
    def variable_count(self) -> int:
        return sum(
            len(family)
            for family in (
                self.select,
                self.assign_day,
                self.assign_lodging,
                self.route_arc,
                self.first_stop,
                self.last_stop,
                self.order,
                self.visit_start,
                self.wait,
            )
        )


@dataclass(frozen=True)
class GurobiExactModelDataV2:
    """Canonical D1 input and index; not an optimization result."""

    model_id: str
    parent_plan_id: str
    request_id: str
    route_matrix_id: str
    context_snapshot_id: str
    days: tuple[int, ...]
    stops: tuple[ExactStopDomainV2, ...]
    lodging: tuple[ExactLodgingDomainV2, ...]
    route_arcs: tuple[ExactRouteArcV2, ...]
    start_anchor_by_day: tuple[tuple[int, str], ...]
    end_anchor_by_day: tuple[tuple[int, str], ...]
    variable_index: ExactVariableIndexV2
    planner_context_mode: str
    schema_version: str = "gurobi-exact-model-data-v2"

    @property
    def structural_candidate_lower_bound(self) -> int:
        """Match the legacy preflight count without enumerating permutations."""

        assignment_count = prod(len(stop.assignment_values) for stop in self.stops)
        empty_assignment_count = int(
            bool(self.stops) and all(None in stop.assignment_values for stop in self.stops)
        )
        lodging_count = prod(len(domain.lodging_ids) for domain in self.lodging)
        return max(0, assignment_count - empty_assignment_count) * lodging_count


def build_gurobi_exact_model_data_v2(
    model: RepairModel,
    route_matrix: RouteMatrix,
    *,
    day_route_config: DayRouteSolverConfig,
) -> GurobiExactModelDataV2:
    """Build strict canonical data for a future direct Gurobi formulation.

    Every arc admitted here has complete road-validated distance and duration
    evidence.  Missing or fallback cells raise instead of being omitted or
    replaced by a geometric proxy.
    """

    if model.route_matrix_id and model.route_matrix_id != route_matrix.matrix_id:
        raise ExactModelDataError("route_matrix_id_mismatch")
    expected_context = str(model.metadata.get("context_snapshot_id") or "")
    if expected_context and expected_context != route_matrix.context_snapshot_id:
        raise ExactModelDataError("route_matrix_context_mismatch")

    days = tuple(sorted(int(day) for day in model.neighborhood.editable_days))
    if not days:
        raise ExactModelDataError("editable_day_domain_empty")

    stop_records = _stop_records(model)
    day_variables = {
        variable.target_id: variable
        for variable in model.variables.day_assignment_variables
    }
    stops: list[ExactStopDomainV2] = []
    for selection in sorted(model.variables.selection_variables, key=lambda row: row.target_id):
        day_variable = day_variables.get(selection.target_id)
        if day_variable is None:
            raise ExactModelDataError("missing_day_variable")
        allowed_days = _allowed_days(day_variable, days)
        if selection.fixed_value != 0 and not allowed_days:
            raise ExactModelDataError("selected_stop_has_no_allowed_day")
        record = stop_records.get(selection.target_id, {})
        stops.append(
            ExactStopDomainV2(
                stop_id=selection.target_id,
                allowed_days=allowed_days,
                selection_fixed=_optional_binary(selection.fixed_value),
                day_fixed=_optional_day(day_variable.fixed_value),
                utility=float(stop_utility(record)),
                contextual_burden=float(context_burden(record)),
                estimated_cost=_stop_cost(record),
                visit_minutes=int(
                    _visit_duration(record, day_route_config.default_visit_minutes)
                ),
            )
        )

    lodging = tuple(
        ExactLodgingDomainV2(
            day=int(variable.target_id),
            lodging_ids=_lodging_ids(variable),
            fixed_lodging_id=(
                str(variable.fixed_value)
                if variable.fixed_value not in {None, ""}
                else None
            ),
        )
        for variable in sorted(
            model.variables.lodging_variables,
            key=lambda row: int(row.target_id),
        )
        if _lodging_ids(variable)
    )

    starts = _required_anchors(day_route_config.start_anchor_by_day, days, "start")
    ends = _required_anchors(day_route_config.end_anchor_by_day, days, "end")
    arcs = _build_route_arcs(
        days=days,
        stops=tuple(stops),
        starts=dict(starts),
        ends=dict(ends),
        route_matrix=route_matrix,
    )
    index = _build_variable_index(days, tuple(stops), lodging, arcs)
    return GurobiExactModelDataV2(
        model_id=model.model_id,
        parent_plan_id=model.parent_plan_id,
        request_id=model.request_id,
        route_matrix_id=route_matrix.matrix_id,
        context_snapshot_id=route_matrix.context_snapshot_id,
        days=days,
        stops=tuple(stops),
        lodging=lodging,
        route_arcs=arcs,
        start_anchor_by_day=starts,
        end_anchor_by_day=ends,
        variable_index=index,
        planner_context_mode=str(model.metadata.get("planner_context_mode") or ""),
    )


def iter_structural_assignments_v2(
    data: GurobiExactModelDataV2,
) -> Iterator[ExactStructuralAssignmentV2]:
    """Enumerate only tiny structural domains for equivalence tests.

    Production solving must use the direct MILP; this iterator intentionally
    contains no stop-sequence permutation loop.
    """

    stop_domains = tuple(stop.assignment_values for stop in data.stops)
    lodging_domains = tuple(domain.lodging_ids for domain in data.lodging)
    for selected_days in product(*stop_domains):
        selected = tuple(
            (stop.stop_id, int(day))
            for stop, day in zip(data.stops, selected_days, strict=True)
            if day is not None
        )
        if not selected:
            continue
        for lodging_ids in product(*lodging_domains):
            lodging = tuple(
                (domain.day, lodging_id)
                for domain, lodging_id in zip(data.lodging, lodging_ids, strict=True)
            )
            yield ExactStructuralAssignmentV2(selected, lodging)


def _build_variable_index(
    days: tuple[int, ...],
    stops: tuple[ExactStopDomainV2, ...],
    lodging: tuple[ExactLodgingDomainV2, ...],
    arcs: tuple[ExactRouteArcV2, ...],
) -> ExactVariableIndexV2:
    assign_day = tuple(
        (stop.stop_id, day)
        for stop in stops
        for day in stop.allowed_days
    )
    day_stops = tuple(
        (day, stop.stop_id)
        for day in days
        for stop in stops
        if day in stop.allowed_days
    )
    return ExactVariableIndexV2(
        select=tuple(stop.stop_id for stop in stops),
        assign_day=assign_day,
        assign_lodging=tuple(
            (domain.day, lodging_id)
            for domain in lodging
            for lodging_id in domain.lodging_ids
        ),
        route_arc=tuple(
            (arc.day, arc.origin_id, arc.destination_id) for arc in arcs
        ),
        first_stop=day_stops,
        last_stop=day_stops,
        order=day_stops,
        visit_start=day_stops,
        wait=day_stops,
    )


def _build_route_arcs(
    *,
    days: tuple[int, ...],
    stops: tuple[ExactStopDomainV2, ...],
    starts: dict[int, str],
    ends: dict[int, str],
    route_matrix: RouteMatrix,
) -> tuple[ExactRouteArcV2, ...]:
    rows: list[ExactRouteArcV2] = []
    for day in days:
        stop_ids = tuple(stop.stop_id for stop in stops if day in stop.allowed_days)
        pairs = (
            *((starts[day], stop_id) for stop_id in stop_ids),
            *((left, right) for left in stop_ids for right in stop_ids if left != right),
            *((stop_id, ends[day]) for stop_id in stop_ids),
        )
        for origin_id, destination_id in pairs:
            try:
                cell = route_matrix.cell(origin_id, destination_id)
                cell.require_publication_eligible()
            except (RouteMatrixError, ValueError) as exc:
                raise ExactModelDataError(
                    f"publication_route_arc_unavailable:{origin_id}->{destination_id}"
                ) from exc
            rows.append(
                ExactRouteArcV2(
                    day=day,
                    origin_id=origin_id,
                    destination_id=destination_id,
                    route_leg_id=cell.route_leg_id,
                    duration_seconds=cell.require_duration_s(),
                    distance_meters=cell.require_distance_m(),
                    query_hash=cell.query_hash,
                )
            )
    return tuple(rows)


def _allowed_days(variable: Any, days: tuple[int, ...]) -> tuple[int, ...]:
    if variable.fixed_value not in {None, 0, ""}:
        fixed = int(variable.fixed_value)
        return (fixed,) if fixed in days else ()
    return tuple(
        sorted(
            {
                int(value)
                for value in variable.allowed_values
                if str(value).strip() and int(value) in days
            }
        )
    )


def _lodging_ids(variable: Any) -> tuple[str, ...]:
    if variable.fixed_value not in {None, ""}:
        return (str(variable.fixed_value),)
    return tuple(sorted({str(value) for value in variable.allowed_values if str(value).strip()}))


def _required_anchors(
    anchors: dict[int, str],
    days: tuple[int, ...],
    kind: str,
) -> tuple[tuple[int, str], ...]:
    rows: list[tuple[int, str]] = []
    for day in days:
        anchor = str(anchors.get(day) or "").strip()
        if not anchor:
            raise ExactModelDataError(f"missing_{kind}_anchor:day:{day}")
        rows.append((day, anchor))
    return tuple(rows)


def _stop_records(model: RepairModel) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for raw in (
        *model.metadata.get("parent_selected_stops", ()),
        *model.metadata.get("candidate_stops", ()),
    ):
        record = dict(raw)
        stop_id = str(
            record.get("stop_id")
            or record.get("poi_id")
            or record.get("name")
            or ""
        ).strip()
        if stop_id:
            records[stop_id] = record
    return records


def _optional_binary(value: Any) -> int | None:
    if value is None:
        return None
    integer = int(value)
    if integer not in {0, 1}:
        raise ExactModelDataError("selection_fixed_value_not_binary")
    return integer


def _optional_day(value: Any) -> int | None:
    return None if value in {None, 0, ""} else int(value)


def _stop_cost(record: dict[str, Any]) -> float | None:
    for key in ("estimated_cost", "cost", "price", "visit_cost"):
        value = record.get(key)
        if value in {None, ""}:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None
