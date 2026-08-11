"""Select one highest-priority unresolved tradeoff from structured consequences."""

from __future__ import annotations

from itertools import combinations

from ..research_artifacts import stable_content_hash
from .models import ConsequenceVector, CriticalTradeoff

_DIMENSION_PRIORITY = {
    "hard_feasibility": 0,
    "permission": 1,
    "core_commitment": 2,
    "repair_scope": 3,
    "general_burden": 4,
}


class CriticalTradeoffSelector:
    def select(
        self,
        *,
        repair_session_id: str,
        consequences: tuple[ConsequenceVector, ...],
    ) -> CriticalTradeoff | None:
        pairs = list(combinations(sorted(consequences, key=lambda item: item.interpretation_id), 2))
        if not pairs:
            return None
        ranked = sorted(
            ((_primary_dimension(left, right), left, right) for left, right in pairs),
            key=lambda item: (_DIMENSION_PRIORITY[item[0]], item[1].interpretation_id, item[2].interpretation_id),
        )
        dimension, left, right = ranked[0]
        numerical = _numerical_deltas(left, right)
        evidence_refs = tuple(dict.fromkeys((*left.evidence_refs, *right.evidence_refs)))
        seed = {
            "session": repair_session_id,
            "left": left.consequence_id,
            "right": right.consequence_id,
            "dimension": dimension,
        }
        return CriticalTradeoff(
            tradeoff_id=f"tradeoff_{stable_content_hash(seed)}",
            repair_session_id=repair_session_id,
            left_interpretation_id=left.interpretation_id,
            right_interpretation_id=right.interpretation_id,
            primary_dimension=dimension,
            left_summary=_summary(left),
            right_summary=_summary(right),
            numerical_deltas=numerical,
            permission_required=bool(left.permission_change_count or right.permission_change_count),
            evidence_refs=evidence_refs,
        )

    def question(self, tradeoff: CriticalTradeoff, *, permission: bool) -> str:
        if permission:
            return (
                f"Option one: {tradeoff.left_summary}. Option two: {tradeoff.right_summary}. "
                "May this repair relax the permission-gated commitment described by these options?"
            )
        return (
            f"Option one: {tradeoff.left_summary}. Option two: {tradeoff.right_summary}. "
            "Which interpretation should this repair use?"
        )


def _primary_dimension(left: ConsequenceVector, right: ConsequenceVector) -> str:
    if left.hard_feasible != right.hard_feasible:
        return "hard_feasibility"
    if (
        left.permission_change_count,
        left.locked_change_count,
        left.booked_change_count,
    ) != (
        right.permission_change_count,
        right.locked_change_count,
        right.booked_change_count,
    ):
        return "permission"
    if left.core_commitment_changes != right.core_commitment_changes:
        return "core_commitment"
    if (
        left.accepted_repair_radius != right.accepted_repair_radius
        or left.affected_day_count != right.affected_day_count
    ):
        return "repair_scope"
    return "general_burden"


def _summary(vector: ConsequenceVector) -> str:
    parts = ["feasible bounded repair" if vector.hard_feasible else "no feasible bounded repair"]
    if vector.permission_change_count:
        parts.append(f"changes {vector.permission_change_count} permission-gated commitment(s)")
    if vector.lodging_change_count:
        parts.append(f"changes {vector.lodging_change_count} lodging assignment(s)")
    if vector.travel_minutes_delta is not None:
        parts.append(f"travel changes by {vector.travel_minutes_delta:+.0f} minutes")
    if vector.monetary_cost_delta is not None:
        parts.append(f"cost changes by {vector.monetary_cost_delta:+.2f}")
    if vector.weighted_edit_cost is not None:
        parts.append(f"weighted edit cost {vector.weighted_edit_cost:.2f}")
    parts.append(f"affects {vector.affected_day_count} day(s)")
    return ", ".join(parts)


def _numerical_deltas(left: ConsequenceVector, right: ConsequenceVector) -> dict[str, float]:
    values: dict[str, float] = {}
    for field in ("weighted_edit_cost", "travel_minutes_delta", "monetary_cost_delta", "walking_burden_delta"):
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value is not None and right_value is not None:
            values[field] = float(right_value) - float(left_value)
    values["affected_day_count"] = float(right.affected_day_count - left.affected_day_count)
    return values
