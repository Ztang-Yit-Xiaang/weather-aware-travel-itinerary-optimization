"""Scope-relative disruption accounting over the canonical typed PlanDiff."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

from ..plans import OwnershipPolicy, compute_plan_diff
from ..research_artifacts import PlanArtifactV2

CHANGE_FAMILIES = (
    "added_stops",
    "deleted_stops",
    "day_moves",
    "time_shifts",
    "duration_changes",
    "time_window_changes",
    "role_changes",
    "reorder_changes",
    "lodging_changes",
    "road_changes",
)


@dataclass(frozen=True)
class DisruptionScope:
    """Declare scope before solving; direct edits are exact (diff family, target) keys.

    Stop edits target a stop ID; lodging/road edits target a day as a string.
    Scope is attribution, never permission to relax a commitment. Direct keys
    declare intended edits, not a proof that every such edit was unavoidable.
    """

    stop_ids: tuple[str, ...]
    days: tuple[int, ...]
    direct_edits: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if any(family not in CHANGE_FAMILIES for family, _ in self.direct_edits):
            raise ValueError("unknown PlanDiff change family")


@dataclass(frozen=True)
class InducedDisruption:
    weighted_edit_cost: float
    direct_cost: float
    secondary_cost: float
    outside_neighborhood_cost: float
    secondary_event_count: int
    secondary_stop_ids: tuple[str, ...]
    outside_stop_ids: tuple[str, ...]
    changed_lodging_days: tuple[int, ...]
    changed_assignment_stop_ids: tuple[str, ...]
    parent_order_propagation_radius: int | None
    unrelated_order_inversions: int
    events: tuple[dict, ...]

    def to_record(self) -> dict:
        return asdict(self)


def measure_induced_disruption(
    parent: PlanArtifactV2,
    child: PlanArtifactV2,
    scope: DisruptionScope,
    *,
    policy: OwnershipPolicy | None = None,
) -> InducedDisruption:
    """Partition each canonical edit exactly once; do not add secondary cost to D.

    Radius is distance in the parent's displayed sequence from a scoped parent
    stop, not road distance or causal influence. It is unavailable without an
    affected parent stop. Positional shifts remain in PlanDiff; inversions among
    surviving unrelated stops separately distinguish reordering from reindexing.
    This pure function performs no feasibility checks or artifact writes.
    Incompatible snapshots fail through compute_plan_diff.
    """

    diff = compute_plan_diff(parent, child, policy)
    direct_keys = set(scope.direct_edits)
    scope_stops, scope_days = set(scope.stop_ids), set(scope.days)
    events: list[dict] = []
    secondary_stops: set[str] = set()
    outside_stops: set[str] = set()
    for family in CHANGE_FAMILIES:
        for change in getattr(diff, family):
            stop_id = getattr(change, "stop_id", None)
            target = str(stop_id if stop_id is not None else change.day)
            days = {
                day
                for day in (getattr(change, name, None) for name in ("day", "from_day", "to_day"))
                if day is not None
            }
            direct = (family, target) in direct_keys
            outside = (stop_id not in scope_stops if stop_id is not None else False) or bool(days - scope_days)
            events.append(
                {
                    "family": family,
                    "target": target,
                    "cost": change.cost,
                    "direct": direct,
                    "secondary": not direct,
                    "outside": outside,
                    "owner_strength": change.owner_strength,
                }
            )
            if not direct and stop_id is not None:
                secondary_stops.add(stop_id)
                if outside:
                    outside_stops.add(stop_id)
    positions = {stop_id: index for index, stop_id in enumerate(parent.sequence)}
    centers = [positions[stop_id] for stop_id in scope.stop_ids if stop_id in positions]
    distances = (
        [
            min(abs(positions[stop_id] - center) for center in centers)
            for stop_id in secondary_stops
            if stop_id in positions
        ]
        if centers
        else []
    )
    common = [stop_id for stop_id in parent.sequence if stop_id in child.sequence and stop_id not in scope_stops]
    child_positions = {stop_id: index for index, stop_id in enumerate(child.sequence)}
    inversions = sum(
        child_positions[a] > child_positions[b]
        for a, b in combinations(common, 2)
        if parent.day_assignments.get(a)
        == parent.day_assignments.get(b)
        == child.day_assignments.get(a)
        == child.day_assignments.get(b)
    )
    return InducedDisruption(
        weighted_edit_cost=diff.weighted_edit_cost,
        direct_cost=sum(event["cost"] for event in events if event["direct"]),
        secondary_cost=sum(event["cost"] for event in events if event["secondary"]),
        outside_neighborhood_cost=sum(event["cost"] for event in events if event["secondary"] and event["outside"]),
        secondary_event_count=sum(event["secondary"] for event in events),
        secondary_stop_ids=tuple(sorted(secondary_stops)),
        outside_stop_ids=tuple(sorted(outside_stops)),
        changed_lodging_days=tuple(change.day for change in diff.lodging_changes),
        changed_assignment_stop_ids=tuple(change.stop_id for change in diff.day_moves),
        parent_order_propagation_radius=max(distances, default=0) if centers else None,
        unrelated_order_inversions=inversions,
        events=tuple(events),
    )
