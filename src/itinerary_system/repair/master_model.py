"""Ownership-aware repair master model scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..plans import (
    ConstraintScope,
    ConstraintStrength,
    OwnedConstraint,
    OwnershipPolicy,
    active_owned_constraints,
)
from ..plans.diff import compute_plan_diff
from ..research_artifacts import PlanArtifactV2, stable_content_hash
from ..routing import RouteMatrix, RouteMatrixMissing
from .change_variables import (
    ChangeVariableType,
    ObjectiveComponent,
    ObjectiveTerm,
    RepairDecisionVariable,
    RepairVariableKind,
    RepairVariableSet,
    TypedChangeVariable,
    VariableDomain,
    change_weight,
)
from .context import (
    ContextSelectionRequirements,
    PlannerContextMode,
    context_selection_requirements,
    contextualize_stop_records,
)
from .neighborhood import RepairNeighborhood, build_parent_plan_index


@dataclass(frozen=True)
class RepairConstraint:
    constraint_id: str
    constraint_type: str
    target_id: str
    variable_name: str
    required_value: Any
    reason_codes: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepairSolution:
    """A candidate assignment evaluated against a repair master model."""

    selected_stop_ids: tuple[str, ...]
    day_assignments: dict[str, int] = field(default_factory=dict)
    lodging_assignments: dict[str, str] = field(default_factory=dict)
    relaxed_constraint_ids: tuple[str, ...] = ()
    route_ids_by_day: dict[int, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairModel:
    model_id: str
    parent_plan_id: str
    request_id: str
    neighborhood: RepairNeighborhood
    variables: RepairVariableSet
    constraints: tuple[RepairConstraint, ...]
    objective_components: tuple[ObjectiveComponent, ...]
    required_route_pairs: tuple[tuple[str, str], ...] = ()
    route_matrix_id: str | None = None
    allow_booked_relaxation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_solution(self, solution: RepairSolution) -> tuple[str, ...]:
        violations: list[str] = []
        selected = set(solution.selected_stop_ids)
        relaxed = set(solution.relaxed_constraint_ids)
        for constraint in self.constraints:
            if constraint.constraint_type == "fixed_selection":
                actual = 1 if constraint.target_id in selected else 0
            elif constraint.constraint_type == "fixed_day":
                actual = solution.day_assignments.get(
                    constraint.target_id,
                    0 if constraint.required_value == 0 else None,
                )
            elif constraint.constraint_type == "fixed_lodging":
                actual = solution.lodging_assignments.get(
                    str(constraint.target_id),
                    "" if constraint.required_value == "" else None,
                )
            elif constraint.constraint_type == "fixed_relaxation":
                actual = 1 if constraint.target_id in relaxed else 0
            else:
                continue
            if actual != constraint.required_value:
                violations.append(f"{constraint.constraint_id}:{constraint.target_id}")
        for stop_id in self.metadata.get("context_conflict_stop_ids", ()):
            violations.append(f"context_selection_conflict:{stop_id}")
        for stop_id in self.metadata.get("context_required_stop_ids", ()):
            if stop_id not in selected:
                violations.append(f"context_required_stop_missing:{stop_id}")
        for stop_id in self.metadata.get("context_excluded_stop_ids", ()):
            if stop_id in selected:
                violations.append(f"context_excluded_stop_selected:{stop_id}")
        for route_id in solution.route_ids_by_day.values():
            if str(route_id) in set(self.metadata.get("context_closed_route_ids", ())):
                violations.append(f"context_closed_route_selected:{route_id}")
        for lodging_id in solution.lodging_assignments.values():
            if str(lodging_id) in set(self.metadata.get("context_unavailable_lodging_ids", ())):
                violations.append(f"context_unavailable_lodging_selected:{lodging_id}")
        for constraint_id in self.metadata.get("locked_constraint_ids", ()):
            if constraint_id in relaxed:
                violations.append(f"locked_constraint_relaxed:{constraint_id}")
        for stop_id in self.metadata.get("locked_stop_ids", ()):
            if stop_id not in selected:
                violations.append(f"locked_stop_deleted:{stop_id}")
        for constraint_id in self.metadata.get("booked_constraint_ids", ()):
            if constraint_id in relaxed and not self.allow_booked_relaxation:
                violations.append(f"booked_constraint_relaxed_without_permission:{constraint_id}")
        return tuple(violations)

    def variable_values(self, solution: RepairSolution) -> dict[str, float]:
        values: dict[str, float] = {}
        selected = set(solution.selected_stop_ids)
        relaxed = set(solution.relaxed_constraint_ids)
        for variable in self.variables.selection_variables:
            values[variable.name] = 1.0 if variable.target_id in selected else 0.0
        for variable in self.variables.relaxation_variables:
            values[variable.name] = 1.0 if variable.target_id in relaxed else 0.0
        for change in self.variables.change_variables:
            values[change.name] = _change_indicator_value(change, solution)
        return values

    def export_components(self, solution: RepairSolution) -> dict[str, float]:
        values = self.variable_values(solution)
        objective_values = {
            component.name: component.evaluate(values)
            for component in self.objective_components
        }
        change_counts = _change_counts(self.variables.change_variables, values)
        violations = self.validate_solution(solution)
        weighted_edit_cost = float(
            sum(change.coefficient * values.get(change.name, 0.0) for change in self.variables.change_variables)
        )
        hard_relaxation = sum(
            1.0
            for change in self.variables.change_variables
            if change.change_type == ChangeVariableType.CONSTRAINT_RELAXATION
            and change.owner_strength in {"locked", "booked", "hard", "strong"}
            and values.get(change.name, 0.0) > 0.0
        )
        return {
            **objective_values,
            **change_counts,
            "weighted_edit_cost": weighted_edit_cost,
            "hard_or_booked_relaxation_count": float(hard_relaxation),
            "fixed_assignment_violation_count": float(
                sum(1 for violation in violations if violation.startswith("fix_") or violation.startswith("fixed_"))
            ),
            "locked_violation_count": float(sum(1 for violation in violations if "locked" in violation)),
            "validation_violation_count": float(len(violations)),
        }

    def extract_plan(self, solution: RepairSolution, *, child_plan_id: str | None = None) -> PlanArtifactV2:
        selected = set(solution.selected_stop_ids)
        parent_stops = {str(stop.get("stop_id") or stop.get("poi_id") or stop.get("name")): dict(stop) for stop in self.metadata["parent_selected_stops"]}
        candidate_stops = {str(stop.get("stop_id") or stop.get("poi_id") or stop.get("name")): dict(stop) for stop in self.metadata.get("candidate_stops", ())}
        selected_records: list[dict[str, Any]] = []
        day_assignments: dict[str, int] = {}
        ordered_by_day: dict[int, list[str]] = {}
        for stop_id in _ordered_solution_stop_ids(self, solution):
            if stop_id not in selected:
                continue
            record = dict(parent_stops.get(stop_id) or candidate_stops.get(stop_id) or {"stop_id": stop_id})
            record["stop_id"] = stop_id
            day = int(solution.day_assignments.get(stop_id, record.get("day", 0)) or 0)
            if day:
                record["day"] = day
                day_assignments[stop_id] = day
                ordered_by_day.setdefault(day, []).append(stop_id)
            selected_records.append(record)
        ordered_days = tuple({"day": day, "stop_ids": tuple(stop_ids)} for day, stop_ids in sorted(ordered_by_day.items()))
        child_id = child_plan_id or f"plan_{stable_content_hash({'parent': self.parent_plan_id, 'solution': solution.metadata})}"
        child = PlanArtifactV2(
            plan_id=child_id,
            parent_plan_id=self.parent_plan_id,
            source_run_id=f"{self.request_id}:repair_master",
            planning_request_id=self.request_id,
            catalog_snapshot_id=str(self.metadata["catalog_snapshot_id"]),
            context_snapshot_id=str(self.metadata["context_snapshot_id"]),
            selected_stops=tuple(selected_records),
            day_assignments=day_assignments,
            sequence=tuple(stop["stop_id"] for stop in selected_records),
            lodging_assignments={str(day): lodging for day, lodging in solution.lodging_assignments.items()},
            ordered_days=ordered_days,
            route_ids_by_day={int(day): route_id for day, route_id in solution.route_ids_by_day.items()},
            owned_constraints=tuple(dict(record) for record in self.metadata.get("owned_constraints", ())),
        )
        diff = compute_plan_diff(self.metadata["parent_plan"], child, OwnershipPolicy())
        change_components = dict(child.change_components)
        change_components["weighted_edit_cost"] = diff.weighted_edit_cost
        return PlanArtifactV2(
            **{
                **child.__dict__,
                "change_components": change_components,
            }
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "parent_plan_id": self.parent_plan_id,
            "request_id": self.request_id,
            "neighborhood": self.neighborhood.to_record(),
            "variables": self.variables.to_record(),
            "constraints": [constraint.to_record() for constraint in self.constraints],
            "objective_components": [component.to_record() for component in self.objective_components],
            "required_route_pairs": [list(pair) for pair in self.required_route_pairs],
            "route_matrix_id": self.route_matrix_id,
            "allow_booked_relaxation": self.allow_booked_relaxation,
            "metadata": {key: value for key, value in self.metadata.items() if key != "parent_plan"},
        }


class RepairMasterModel:
    """Builder object for solver-neutral ownership-aware repair models."""

    def __init__(
        self,
        parent: PlanArtifactV2,
        request: Any,
        neighborhood: RepairNeighborhood,
        route_matrix: RouteMatrix | None,
        *,
        ownership_policy: OwnershipPolicy | None = None,
        publication_mode: bool = False,
        planner_context_mode: PlannerContextMode | str = PlannerContextMode.AWARE,
    ) -> None:
        self.parent = parent
        self.request = request
        self.neighborhood = neighborhood
        self.route_matrix = route_matrix
        self.ownership_policy = ownership_policy or OwnershipPolicy()
        self.publication_mode = publication_mode
        self.planner_context_mode = PlannerContextMode(str(planner_context_mode))

    def build(self) -> RepairModel:
        return build_repair_master_model(
            self.parent,
            self.request,
            self.neighborhood,
            self.route_matrix,
            ownership_policy=self.ownership_policy,
            publication_mode=self.publication_mode,
            planner_context_mode=self.planner_context_mode,
        )


def build_repair_master_model(
    parent: PlanArtifactV2,
    request: Any,
    neighborhood: RepairNeighborhood,
    matrix: RouteMatrix | None,
    *,
    ownership_policy: OwnershipPolicy | None = None,
    publication_mode: bool = False,
    planner_context_mode: PlannerContextMode | str = PlannerContextMode.AWARE,
) -> RepairModel:
    """Build a solver-neutral master model with repair variables and constraints."""

    if publication_mode and matrix is None:
        raise RouteMatrixMissing("repair master publication mode requires a RouteMatrix")
    policy = ownership_policy or OwnershipPolicy()
    context_mode = PlannerContextMode(str(planner_context_mode))
    index = build_parent_plan_index(parent)
    parent_stops = contextualize_stop_records(parent.selected_stops, request)
    candidate_stops = contextualize_stop_records(_candidate_stops(request), request)
    selection_requirements = context_selection_requirements(
        request,
        (*parent_stops, *candidate_stops),
        mode=context_mode,
    )
    active_constraints = active_owned_constraints(tuple(dict(record) for record in parent.owned_constraints))
    allow_booked = _allow_booked_relaxation(request)
    owner_strengths = _owner_strengths_by_target(active_constraints, policy)
    ordered_stop_ids = (*index.stop_ids, *tuple(stop["stop_id"] for stop in candidate_stops))
    selection_variables = _selection_variables(
        index,
        candidate_stops,
        neighborhood,
        active_constraints,
        selection_requirements,
    )
    day_variables = _day_assignment_variables(index, candidate_stops, neighborhood, active_constraints)
    lodging_variables = _lodging_variables(index, candidate_stops, neighborhood, active_constraints, allow_booked)
    relaxation_variables = _relaxation_variables(active_constraints, neighborhood, allow_booked)
    change_variables = _change_variables(
        index=index,
        candidate_stops=candidate_stops,
        neighborhood=neighborhood,
        selection_variables=selection_variables,
        day_variables=day_variables,
        lodging_variables=lodging_variables,
        relaxation_variables=relaxation_variables,
        owner_strengths=owner_strengths,
        policy=policy,
    )
    variable_set = RepairVariableSet(
        selection_variables=tuple(selection_variables),
        day_assignment_variables=tuple(day_variables),
        lodging_variables=tuple(lodging_variables),
        relaxation_variables=tuple(relaxation_variables),
        change_variables=tuple(change_variables),
    )
    constraints = _constraints(variable_set, active_constraints, index, allow_booked)
    objectives = _objective_components(change_variables)
    required_route_pairs = _required_route_pairs(index, neighborhood)
    if publication_mode and matrix is not None:
        for origin_id, destination_id in required_route_pairs:
            matrix.cell(origin_id, destination_id).require_publication_eligible()
    request_id = _request_id(request)
    metadata = {
        "catalog_snapshot_id": parent.catalog_snapshot_id,
        "context_snapshot_id": parent.context_snapshot_id,
        "owned_constraints": tuple(dict(record) for record in parent.owned_constraints),
        "parent_selected_stops": tuple(parent_stops),
        "parent_day_by_stop": dict(index.stop_day),
        "candidate_stops": tuple(candidate_stops),
        "planner_context_mode": context_mode.value,
        "context_required_stop_ids": selection_requirements.required_stop_ids,
        "context_excluded_stop_ids": selection_requirements.excluded_stop_ids,
        "context_conflict_stop_ids": selection_requirements.conflict_stop_ids,
        "context_closed_route_ids": _context_values(request, "closed_route_ids") if context_mode == PlannerContextMode.AWARE else (),
        "context_unavailable_lodging_ids": _context_values(request, "unavailable_lodging_ids")
        if context_mode == PlannerContextMode.AWARE
        else (),
        "ordered_stop_ids": ordered_stop_ids,
        "locked_constraint_ids": tuple(
            constraint.constraint_id for constraint in active_constraints if constraint.strength == ConstraintStrength.LOCKED
        ),
        "booked_constraint_ids": tuple(
            constraint.constraint_id for constraint in active_constraints if constraint.strength == ConstraintStrength.BOOKED
        ),
        "locked_stop_ids": tuple(
            constraint.target_id
            for constraint in active_constraints
            if constraint.strength == ConstraintStrength.LOCKED and constraint.scope in {ConstraintScope.STOP, ConstraintScope.POI}
        ),
        "parent_plan": parent,
    }
    model_seed = {
        "parent_plan_id": parent.plan_id,
        "request_id": request_id,
        "neighborhood": neighborhood.to_record(),
        "route_matrix_id": matrix.matrix_id if matrix is not None else None,
        "planner_context_mode": context_mode.value,
    }
    return RepairModel(
        model_id=f"repair_model_{stable_content_hash(model_seed)}",
        parent_plan_id=parent.plan_id,
        request_id=request_id,
        neighborhood=neighborhood,
        variables=variable_set,
        constraints=tuple(constraints),
        objective_components=tuple(objectives),
        required_route_pairs=required_route_pairs,
        route_matrix_id=matrix.matrix_id if matrix is not None else None,
        allow_booked_relaxation=allow_booked,
        metadata=metadata,
    )


def _context_values(request: Any, key: str) -> tuple[str, ...]:
    raw_constraints = getattr(request, "confirmed_constraints", {})
    if not isinstance(raw_constraints, dict):
        return ()
    raw = raw_constraints.get(key, ())
    if isinstance(raw, str):
        values = (raw,)
    else:
        try:
            values = tuple(raw)
        except TypeError:
            values = (raw,)
    return tuple(sorted(str(value).strip() for value in values if str(value).strip()))


def _selection_variables(
    index: Any,
    candidate_stops: tuple[dict[str, Any], ...],
    neighborhood: RepairNeighborhood,
    constraints: tuple[OwnedConstraint, ...],
    context_requirements: ContextSelectionRequirements,
) -> list[RepairDecisionVariable]:
    locked_stops = _locked_stop_ids(constraints)
    variables: list[RepairDecisionVariable] = []
    editable = set(neighborhood.editable_stop_ids)
    required = set(context_requirements.required_stop_ids)
    excluded = set(context_requirements.excluded_stop_ids)
    for stop_id in index.stop_ids:
        if stop_id in required:
            fixed = 1
        elif stop_id in excluded:
            fixed = 0
        else:
            fixed = 1 if stop_id in locked_stops or stop_id not in editable else None
        variables.append(
            RepairDecisionVariable(
                name=f"select_stop[{stop_id}]",
                kind=RepairVariableKind.SELECT_STOP,
                domain=VariableDomain.BINARY,
                target_id=stop_id,
                parent_value=1,
                allowed_values=(0, 1),
                lower_bound=1.0 if fixed == 1 else 0.0,
                upper_bound=0.0 if fixed == 0 else 1.0,
                fixed_value=fixed,
                day=index.stop_day.get(stop_id),
                reason_codes=_fixed_reasons(fixed, "locked_or_outside_neighborhood"),
            )
        )
    for stop in candidate_stops:
        stop_id = stop["stop_id"]
        day = _coerce_int(stop.get("day"))
        editable_candidate = day is None or day in set(neighborhood.editable_days)
        if stop_id in required:
            fixed = 1
        elif stop_id in excluded:
            fixed = 0
        else:
            fixed = None if editable_candidate else 0
        variables.append(
            RepairDecisionVariable(
                name=f"select_stop[{stop_id}]",
                kind=RepairVariableKind.SELECT_STOP,
                domain=VariableDomain.BINARY,
                target_id=stop_id,
                parent_value=0,
                allowed_values=(0, 1),
                lower_bound=0.0,
                upper_bound=0.0 if fixed == 0 else 1.0,
                fixed_value=fixed,
                day=day,
                reason_codes=_fixed_reasons(fixed, "candidate_outside_neighborhood"),
            )
        )
    return variables


def _day_assignment_variables(
    index: Any,
    candidate_stops: tuple[dict[str, Any], ...],
    neighborhood: RepairNeighborhood,
    constraints: tuple[OwnedConstraint, ...],
) -> list[RepairDecisionVariable]:
    locked_day_targets = _locked_day_targets(constraints)
    variables: list[RepairDecisionVariable] = []
    editable = set(neighborhood.editable_stop_ids)
    allowed_days = neighborhood.editable_days
    for stop_id in index.stop_ids:
        parent_day = index.stop_day.get(stop_id)
        fixed = parent_day if stop_id not in editable or stop_id in locked_day_targets else None
        variables.append(
            RepairDecisionVariable(
                name=f"assign_day[{stop_id}]",
                kind=RepairVariableKind.ASSIGN_DAY,
                domain=VariableDomain.INTEGER,
                target_id=stop_id,
                parent_value=parent_day,
                allowed_values=(parent_day,) if fixed is not None else allowed_days,
                lower_bound=float(min(allowed_days)) if allowed_days and fixed is None else float(parent_day or 0),
                upper_bound=float(max(allowed_days)) if allowed_days and fixed is None else float(parent_day or 0),
                fixed_value=fixed,
                day=parent_day,
                reason_codes=_fixed_reasons(fixed, "day_outside_neighborhood_or_locked"),
            )
        )
    for stop in candidate_stops:
        stop_id = stop["stop_id"]
        day = _coerce_int(stop.get("day"))
        allowed = allowed_days or ((day,) if day is not None else ())
        fixed = 0 if not allowed else None
        variables.append(
            RepairDecisionVariable(
                name=f"assign_day[{stop_id}]",
                kind=RepairVariableKind.ASSIGN_DAY,
                domain=VariableDomain.INTEGER,
                target_id=stop_id,
                parent_value=None,
                allowed_values=allowed,
                lower_bound=float(min(allowed)) if allowed else None,
                upper_bound=float(max(allowed)) if allowed else None,
                fixed_value=fixed,
                day=day,
                reason_codes=_fixed_reasons(fixed, "candidate_day_outside_neighborhood"),
            )
        )
    return variables


def _lodging_variables(
    index: Any,
    candidate_stops: tuple[dict[str, Any], ...],
    neighborhood: RepairNeighborhood,
    constraints: tuple[OwnedConstraint, ...],
    allow_booked: bool,
) -> list[RepairDecisionVariable]:
    booked_lodging_ids = _booked_lodging_ids(constraints)
    candidate_lodging_ids = {
        str(stop.get("lodging_id") or stop.get("hotel_id") or stop.get("overnight_city") or "").strip()
        for stop in candidate_stops
    }
    allowed_lodgings = tuple(sorted({*index.lodging_by_day.values(), *tuple(value for value in candidate_lodging_ids if value)}))
    variables: list[RepairDecisionVariable] = []
    editable_nights = set(neighborhood.editable_lodging_nights)
    for day in index.days:
        parent_lodging = index.lodging_by_day.get(day, "")
        booked_fixed = bool(parent_lodging in booked_lodging_ids and not allow_booked)
        fixed = parent_lodging if day not in editable_nights or booked_fixed else None
        variables.append(
            RepairDecisionVariable(
                name=f"assign_lodging[{day}]",
                kind=RepairVariableKind.ASSIGN_LODGING,
                domain=VariableDomain.CATEGORICAL,
                target_id=str(day),
                parent_value=parent_lodging,
                allowed_values=(parent_lodging,) if fixed is not None else allowed_lodgings,
                fixed_value=fixed,
                day=day,
                reason_codes=_fixed_reasons(fixed, "lodging_outside_neighborhood_or_booked"),
            )
        )
    return variables


def _relaxation_variables(
    constraints: tuple[OwnedConstraint, ...],
    neighborhood: RepairNeighborhood,
    allow_booked: bool,
) -> list[RepairDecisionVariable]:
    variables: list[RepairDecisionVariable] = []
    frozen = set(neighborhood.frozen_constraint_ids)
    for constraint in constraints:
        fixed = 0 if constraint.constraint_id in frozen or not constraint.can_relax(allow_booked_relaxation=allow_booked) else None
        variables.append(
            RepairDecisionVariable(
                name=f"relax_constraint[{constraint.constraint_id}]",
                kind=RepairVariableKind.RELAX_CONSTRAINT,
                domain=VariableDomain.BINARY,
                target_id=constraint.constraint_id,
                parent_value=0,
                allowed_values=(0, 1),
                lower_bound=0.0,
                upper_bound=0.0 if fixed == 0 else 1.0,
                fixed_value=fixed,
                reason_codes=_fixed_reasons(fixed, "constraint_frozen_or_not_relaxable"),
            )
        )
    return variables


def _change_variables(
    *,
    index: Any,
    candidate_stops: tuple[dict[str, Any], ...],
    neighborhood: RepairNeighborhood,
    selection_variables: list[RepairDecisionVariable],
    day_variables: list[RepairDecisionVariable],
    lodging_variables: list[RepairDecisionVariable],
    relaxation_variables: list[RepairDecisionVariable],
    owner_strengths: dict[str, str],
    policy: OwnershipPolicy,
) -> list[TypedChangeVariable]:
    changes: list[TypedChangeVariable] = []
    selection_by_target = {variable.target_id: variable for variable in selection_variables}
    for stop_id in index.stop_ids:
        variable = selection_by_target[stop_id]
        strength = owner_strengths.get(stop_id, "")
        changes.append(
            TypedChangeVariable(
                name=f"delete_stop[{stop_id}]",
                change_type=ChangeVariableType.DELETE_STOP,
                target_id=stop_id,
                source_variable=variable.name,
                parent_value=1,
                owner_strength=strength,
                coefficient=change_weight(ChangeVariableType.DELETE_STOP, owner_strength=strength, policy=policy),
                upper_bound=0.0 if variable.fixed_value == 1 else 1.0,
            )
        )
    for stop in candidate_stops:
        stop_id = stop["stop_id"]
        variable = selection_by_target[stop_id]
        changes.append(
            TypedChangeVariable(
                name=f"add_stop[{stop_id}]",
                change_type=ChangeVariableType.ADD_STOP,
                target_id=stop_id,
                source_variable=variable.name,
                parent_value=0,
                coefficient=change_weight(ChangeVariableType.ADD_STOP, policy=policy),
                upper_bound=0.0 if variable.fixed_value == 0 else 1.0,
            )
        )
    for variable in day_variables:
        if variable.parent_value is None:
            continue
        strength = owner_strengths.get(variable.target_id, "")
        changes.append(
            TypedChangeVariable(
                name=f"day_move[{variable.target_id}]",
                change_type=ChangeVariableType.DAY_MOVE,
                target_id=variable.target_id,
                source_variable=variable.name,
                parent_value=variable.parent_value,
                owner_strength=strength,
                coefficient=change_weight(ChangeVariableType.DAY_MOVE, owner_strength=strength, policy=policy),
                upper_bound=0.0 if variable.fixed_value == variable.parent_value else 1.0,
            )
        )
    for variable in lodging_variables:
        strength = owner_strengths.get(str(variable.parent_value), "")
        changes.append(
            TypedChangeVariable(
                name=f"lodging_change[{variable.target_id}]",
                change_type=ChangeVariableType.LODGING_CHANGE,
                target_id=variable.target_id,
                source_variable=variable.name,
                parent_value=variable.parent_value,
                owner_strength=strength,
                coefficient=change_weight(ChangeVariableType.LODGING_CHANGE, owner_strength=strength, policy=policy),
                upper_bound=0.0 if variable.fixed_value == variable.parent_value else 1.0,
            )
        )
    for day in neighborhood.boundary_route_days:
        route_id = index.route_ids_by_day.get(day, "")
        strength = owner_strengths.get(route_id, "")
        changes.append(
            TypedChangeVariable(
                name=f"road_change[{day}]",
                change_type=ChangeVariableType.ROAD_CHANGE,
                target_id=str(day),
                source_variable=f"route_day[{day}]",
                parent_value=route_id,
                owner_strength=strength,
                coefficient=change_weight(ChangeVariableType.ROAD_CHANGE, owner_strength=strength, policy=policy),
            )
        )
    for variable in relaxation_variables:
        constraint_id = variable.target_id
        strength = owner_strengths.get(constraint_id, "")
        changes.append(
            TypedChangeVariable(
                name=f"constraint_relaxation[{constraint_id}]",
                change_type=ChangeVariableType.CONSTRAINT_RELAXATION,
                target_id=constraint_id,
                source_variable=variable.name,
                parent_value=0,
                owner_strength=strength,
                coefficient=policy.weight_for_strength(strength),
                upper_bound=variable.upper_bound if variable.upper_bound is not None else 1.0,
            )
        )
    return changes


def _constraints(
    variable_set: RepairVariableSet,
    constraints: tuple[OwnedConstraint, ...],
    index: Any,
    allow_booked: bool,
) -> list[RepairConstraint]:
    rows: list[RepairConstraint] = []
    for variable in variable_set.all_decision_variables:
        if not variable.is_fixed:
            continue
        constraint_type = {
            RepairVariableKind.SELECT_STOP: "fixed_selection",
            RepairVariableKind.ASSIGN_DAY: "fixed_day",
            RepairVariableKind.ASSIGN_LODGING: "fixed_lodging",
            RepairVariableKind.RELAX_CONSTRAINT: "fixed_relaxation",
        }.get(variable.kind, "fixed_variable")
        rows.append(
            RepairConstraint(
                constraint_id=f"fix_{variable.name}",
                constraint_type=constraint_type,
                target_id=variable.target_id,
                variable_name=variable.name,
                required_value=variable.fixed_value,
                reason_codes=variable.reason_codes,
            )
        )
    for constraint in constraints:
        if constraint.strength == ConstraintStrength.LOCKED:
            rows.append(
                RepairConstraint(
                    constraint_id=f"locked_{constraint.constraint_id}",
                    constraint_type="fixed_relaxation",
                    target_id=constraint.constraint_id,
                    variable_name=f"relax_constraint[{constraint.constraint_id}]",
                    required_value=0,
                    reason_codes=("locked_constraint",),
                )
            )
            if constraint.scope in {ConstraintScope.STOP, ConstraintScope.POI}:
                rows.append(
                    RepairConstraint(
                        constraint_id=f"locked_select_{constraint.target_id}",
                        constraint_type="fixed_selection",
                        target_id=constraint.target_id,
                        variable_name=f"select_stop[{constraint.target_id}]",
                        required_value=1,
                        reason_codes=("locked_stop",),
                    )
                )
        if constraint.strength == ConstraintStrength.BOOKED and not allow_booked:
            rows.append(
                RepairConstraint(
                    constraint_id=f"booked_{constraint.constraint_id}",
                    constraint_type="fixed_relaxation",
                    target_id=constraint.constraint_id,
                    variable_name=f"relax_constraint[{constraint.constraint_id}]",
                    required_value=0,
                    reason_codes=("booked_without_permission",),
                )
            )
            if constraint.scope == ConstraintScope.LODGING:
                for day, lodging_id in index.lodging_by_day.items():
                    if lodging_id == constraint.target_id:
                        rows.append(
                            RepairConstraint(
                                constraint_id=f"booked_lodging_{day}",
                                constraint_type="fixed_lodging",
                                target_id=str(day),
                                variable_name=f"assign_lodging[{day}]",
                                required_value=lodging_id,
                                reason_codes=("booked_lodging_without_permission",),
                            )
                        )
    return rows


def _objective_components(change_variables: list[TypedChangeVariable]) -> list[ObjectiveComponent]:
    hard_terms: list[ObjectiveTerm] = []
    edit_terms: list[ObjectiveTerm] = []
    context_terms: list[ObjectiveTerm] = []
    utility_terms: list[ObjectiveTerm] = []
    for change in change_variables:
        term = ObjectiveTerm(
            variable_name=change.name,
            coefficient=change.coefficient,
            component=change.change_type.value,
            target_id=change.target_id,
        )
        if change.change_type == ChangeVariableType.CONSTRAINT_RELAXATION and change.owner_strength in {
            "locked",
            "booked",
            "hard",
            "strong",
        }:
            hard_terms.append(term)
        elif change.change_type in {ChangeVariableType.ROAD_CHANGE, ChangeVariableType.LODGING_CHANGE}:
            context_terms.append(term)
            edit_terms.append(term)
        else:
            edit_terms.append(term)
        if change.change_type in {ChangeVariableType.ADD_STOP, ChangeVariableType.DELETE_STOP}:
            utility_terms.append(term)
    return [
        ObjectiveComponent(name="hard_and_booked_relaxation", stage="preserve_commitments", priority=0, terms=tuple(hard_terms)),
        ObjectiveComponent(name="ownership_weighted_change", stage="preserve_parent", priority=1, terms=tuple(edit_terms)),
        ObjectiveComponent(name="contextual_burden_proxy", stage="reduce_contextual_burden", priority=2, terms=tuple(context_terms)),
        ObjectiveComponent(name="utility_proxy_change", stage="retain_utility", priority=3, terms=tuple(utility_terms)),
    ]


def _change_indicator_value(change: TypedChangeVariable, solution: RepairSolution) -> float:
    selected = set(solution.selected_stop_ids)
    if change.change_type == ChangeVariableType.DELETE_STOP:
        return 0.0 if change.target_id in selected else 1.0
    if change.change_type == ChangeVariableType.ADD_STOP:
        return 1.0 if change.target_id in selected else 0.0
    if change.change_type == ChangeVariableType.DAY_MOVE:
        if change.target_id not in selected:
            return 0.0
        return 1.0 if solution.day_assignments.get(change.target_id) != change.parent_value else 0.0
    if change.change_type == ChangeVariableType.LODGING_CHANGE:
        return 1.0 if solution.lodging_assignments.get(str(change.target_id)) != change.parent_value else 0.0
    if change.change_type == ChangeVariableType.ROAD_CHANGE:
        day = int(change.target_id)
        return 1.0 if solution.route_ids_by_day.get(day) != change.parent_value else 0.0
    if change.change_type == ChangeVariableType.CONSTRAINT_RELAXATION:
        return 1.0 if change.target_id in set(solution.relaxed_constraint_ids) else 0.0
    return 0.0


def _ordered_solution_stop_ids(model: RepairModel, solution: RepairSolution) -> tuple[str, ...]:
    day_sequences = solution.metadata.get("day_sequences", {})
    if not isinstance(day_sequences, dict):
        return tuple(model.metadata["ordered_stop_ids"])
    emitted: set[str] = set()
    ordered: list[str] = []
    parent = model.metadata["parent_plan"]
    parent_day_by_stop = model.metadata.get("parent_day_by_stop", {})
    for stop_id in parent.sequence:
        day = parent_day_by_stop.get(str(stop_id))
        override_key = str(day) if day is not None else ""
        override = day_sequences.get(override_key)
        if override_key in day_sequences:
            for override_stop_id in override:
                stop_text = str(override_stop_id)
                if stop_text not in emitted:
                    ordered.append(stop_text)
                    emitted.add(stop_text)
            continue
        if stop_id not in emitted:
            ordered.append(str(stop_id))
            emitted.add(str(stop_id))
    for stop_id in model.metadata["ordered_stop_ids"]:
        stop_text = str(stop_id)
        if stop_text in emitted:
            continue
        ordered.append(stop_text)
        emitted.add(stop_text)
    return tuple(ordered)


def _change_counts(change_variables: tuple[TypedChangeVariable, ...], values: dict[str, float]) -> dict[str, float]:
    counts = {
        "added_stop_count": 0.0,
        "deleted_stop_count": 0.0,
        "day_move_count": 0.0,
        "lodging_change_count": 0.0,
        "road_change_count": 0.0,
        "constraint_relaxation_count": 0.0,
    }
    key_by_type = {
        ChangeVariableType.ADD_STOP: "added_stop_count",
        ChangeVariableType.DELETE_STOP: "deleted_stop_count",
        ChangeVariableType.DAY_MOVE: "day_move_count",
        ChangeVariableType.LODGING_CHANGE: "lodging_change_count",
        ChangeVariableType.ROAD_CHANGE: "road_change_count",
        ChangeVariableType.CONSTRAINT_RELAXATION: "constraint_relaxation_count",
    }
    for change in change_variables:
        key = key_by_type.get(change.change_type)
        if key:
            counts[key] += float(values.get(change.name, 0.0))
    return counts


def _candidate_stops(request: Any) -> tuple[dict[str, Any], ...]:
    candidates: list[dict[str, Any]] = []
    for index, raw_candidate in enumerate(getattr(request, "candidate_pois", ()) or (), start=1):
        candidate = dict(raw_candidate)
        stop_id = _stop_id(candidate, index, prefix="candidate")
        candidate["stop_id"] = stop_id
        candidates.append(candidate)
    return tuple(candidates)


def _required_route_pairs(index: Any, neighborhood: RepairNeighborhood) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for day in neighborhood.boundary_route_days:
        stop_ids = index.stops_by_day.get(day, ())
        for left, right in zip(stop_ids, stop_ids[1:], strict=False):
            pairs.append((left, right))
    return tuple(pairs)


def _owner_strengths_by_target(constraints: tuple[OwnedConstraint, ...], policy: OwnershipPolicy) -> dict[str, str]:
    strengths: dict[str, str] = {}
    for constraint in constraints:
        targets = {constraint.constraint_id, constraint.target_id}
        current_weight = policy.weight_for_strength(strengths.get(constraint.target_id, ""))
        new_weight = policy.weight_for_strength(constraint.strength)
        for target in targets:
            if new_weight >= current_weight:
                strengths[target] = constraint.strength.value
    return strengths


def _locked_stop_ids(constraints: tuple[OwnedConstraint, ...]) -> set[str]:
    return {
        constraint.target_id
        for constraint in constraints
        if constraint.strength == ConstraintStrength.LOCKED and constraint.scope in {ConstraintScope.STOP, ConstraintScope.POI}
    }


def _locked_day_targets(constraints: tuple[OwnedConstraint, ...]) -> set[str]:
    return {
        constraint.target_id
        for constraint in constraints
        if constraint.strength == ConstraintStrength.LOCKED
        and constraint.scope in {ConstraintScope.DAY, ConstraintScope.TIME, ConstraintScope.TIME_WINDOW}
    }


def _booked_lodging_ids(constraints: tuple[OwnedConstraint, ...]) -> set[str]:
    return {
        constraint.target_id
        for constraint in constraints
        if constraint.strength == ConstraintStrength.BOOKED and constraint.scope == ConstraintScope.LODGING
    }


def _allow_booked_relaxation(request: Any) -> bool:
    confirmed_constraints = getattr(request, "confirmed_constraints", {})
    if not isinstance(confirmed_constraints, dict):
        confirmed_constraints = {}
    return bool(
        getattr(request, "allow_booked_relaxation", False)
        or confirmed_constraints.get("allow_booked_relaxation", False)
    )


def _request_id(request: Any) -> str:
    return str(getattr(request, "request_id", "") or getattr(request, "repair_request_id", "") or "repair_request")


def _stop_id(stop: dict[str, Any], index: int, *, prefix: str = "stop") -> str:
    for key in ("stop_id", "poi_id", "attraction_id", "attraction_name", "name", "poi", "stop_name"):
        value = stop.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"{prefix}_{index}"


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _fixed_reasons(fixed: Any | None, reason: str) -> tuple[str, ...]:
    return (reason,) if fixed is not None else ()
