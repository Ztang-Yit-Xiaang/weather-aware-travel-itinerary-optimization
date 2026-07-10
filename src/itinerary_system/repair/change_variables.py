"""Typed repair variables relative to an immutable parent plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from ..plans import ConstraintStrength, OwnershipPolicy
from ..plans.models import enum_value


class VariableDomain(StrEnum):
    BINARY = "binary"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"


class RepairVariableKind(StrEnum):
    SELECT_STOP = "select_stop"
    ASSIGN_DAY = "assign_day"
    ASSIGN_LODGING = "assign_lodging"
    RELAX_CONSTRAINT = "relax_constraint"
    CHANGE_INDICATOR = "change_indicator"


class ChangeVariableType(StrEnum):
    ADD_STOP = "added_stop"
    DELETE_STOP = "deleted_stop"
    DAY_MOVE = "day_move"
    TIME_SHIFT = "time_shift"
    ORDER_CHANGE = "order_change"
    LODGING_CHANGE = "lodging_change"
    ROAD_CHANGE = "road_change"
    CONSTRAINT_RELAXATION = "constraint_relaxation"


@dataclass(frozen=True)
class RepairDecisionVariable:
    """One solver-facing variable declaration for a repair model."""

    name: str
    kind: RepairVariableKind
    domain: VariableDomain
    target_id: str
    parent_value: Any
    allowed_values: tuple[Any, ...] = ()
    lower_bound: float | None = None
    upper_bound: float | None = None
    fixed_value: Any | None = None
    day: int | None = None
    reason_codes: tuple[str, ...] = ()

    @property
    def is_fixed(self) -> bool:
        return self.fixed_value is not None

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["kind"] = self.kind.value
        record["domain"] = self.domain.value
        return record


@dataclass(frozen=True)
class TypedChangeVariable:
    """A typed edit indicator used by preservation objectives."""

    name: str
    change_type: ChangeVariableType
    target_id: str
    source_variable: str
    parent_value: Any
    changed_value: Any | None = None
    owner_strength: str = ""
    coefficient: float = 1.0
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    reason_codes: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.upper_bound > 0.0

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["change_type"] = self.change_type.value
        return record


@dataclass(frozen=True)
class ObjectiveTerm:
    variable_name: str
    coefficient: float
    component: str
    target_id: str = ""

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveComponent:
    name: str
    stage: str
    priority: int
    terms: tuple[ObjectiveTerm, ...]
    sense: str = "minimize"

    def evaluate(self, values: dict[str, float]) -> float:
        return float(sum(term.coefficient * float(values.get(term.variable_name, 0.0)) for term in self.terms))

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "priority": self.priority,
            "sense": self.sense,
            "terms": [term.to_record() for term in self.terms],
        }


@dataclass(frozen=True)
class RepairVariableSet:
    selection_variables: tuple[RepairDecisionVariable, ...]
    day_assignment_variables: tuple[RepairDecisionVariable, ...]
    lodging_variables: tuple[RepairDecisionVariable, ...]
    relaxation_variables: tuple[RepairDecisionVariable, ...]
    change_variables: tuple[TypedChangeVariable, ...]

    @property
    def all_decision_variables(self) -> tuple[RepairDecisionVariable, ...]:
        return (
            *self.selection_variables,
            *self.day_assignment_variables,
            *self.lodging_variables,
            *self.relaxation_variables,
        )

    def decision_variable(self, name: str) -> RepairDecisionVariable:
        for variable in self.all_decision_variables:
            if variable.name == name:
                return variable
        raise KeyError(name)

    def change_variable(self, name: str) -> TypedChangeVariable:
        for variable in self.change_variables:
            if variable.name == name:
                return variable
        raise KeyError(name)

    def to_record(self) -> dict[str, Any]:
        return {
            "selection_variables": [variable.to_record() for variable in self.selection_variables],
            "day_assignment_variables": [variable.to_record() for variable in self.day_assignment_variables],
            "lodging_variables": [variable.to_record() for variable in self.lodging_variables],
            "relaxation_variables": [variable.to_record() for variable in self.relaxation_variables],
            "change_variables": [variable.to_record() for variable in self.change_variables],
        }


def change_weight(
    change_type: ChangeVariableType | str,
    *,
    owner_strength: ConstraintStrength | str | None = None,
    policy: OwnershipPolicy | None = None,
) -> float:
    policy = policy or OwnershipPolicy()
    return float(policy.weight_for_change(enum_value(change_type)) * policy.weight_for_strength(owner_strength))
