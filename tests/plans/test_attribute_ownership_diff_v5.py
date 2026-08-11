from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from itinerary_system.plans import (
    ConstraintOrigin,
    ConstraintScope,
    ConstraintStrength,
    OwnedConstraint,
    OwnershipPolicy,
    PlanDiff,
    RelaxationPolicy,
    RoleChange,
    compute_plan_diff,
)
from itinerary_system.product_app.draft_compiler import _verify_diff
from itinerary_system.product_app.workspace import WorkspaceError
from itinerary_system.research_artifacts import PlanArtifactV2, stable_content_hash


def _constraint(
    *,
    constraint_id: str,
    target: str,
    relation: str,
    value: object,
    strength: ConstraintStrength = ConstraintStrength.SOFT,
) -> dict[str, object]:
    return OwnedConstraint(
        constraint_id=constraint_id,
        origin=ConstraintOrigin.USER,
        strength=strength,
        scope=ConstraintScope.STOP,
        target_id=target,
        relation=relation,
        value=value,
        confirmed=True,
        relaxation_policy=RelaxationPolicy.ALWAYS,
    ).to_record()


def _plan(*, constraints: tuple[dict[str, object], ...] = ()) -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="plan_parent_v5",
        source_run_id="run_v5",
        planning_request_id="request_v5",
        catalog_snapshot_id="catalog_v5",
        context_snapshot_id="context_v5",
        selected_stops=(
            {
                "stop_id": "a",
                "day": 1,
                "stop_order": 1,
                "start_time": "09:00",
            },
            {
                "stop_id": "b",
                "day": 1,
                "stop_order": 2,
                "start_time": "10:00",
            },
        ),
        day_assignments={"a": 1, "b": 1},
        sequence=("a", "b"),
        lodging_assignments={"1": "hotel"},
        route_ids_by_day={1: "route"},
        owned_constraints=constraints,
        created_at="2026-08-08T00:00:00+00:00",
    )


def _child(
    parent: PlanArtifactV2,
    *,
    stops: tuple[dict[str, object], ...],
    constraints: tuple[dict[str, object], ...] | None = None,
) -> PlanArtifactV2:
    return replace(
        parent,
        plan_id="plan_child_v5",
        parent_plan_id=parent.plan_id,
        selected_stops=stops,
        day_assignments={
            str(stop["stop_id"]): int(stop["day"])
            for stop in stops
        },
        sequence=tuple(str(stop["stop_id"]) for stop in stops),
        owned_constraints=(
            parent.owned_constraints if constraints is None else constraints
        ),
    )


def _evidence(record: dict[str, object]) -> dict[str, object]:
    assert record["schema_version"] == "plan-diff-v5"
    evidence = record["attribute_ownership_evidence"]
    assert isinstance(evidence, dict)
    payload = dict(evidence)
    content_hash = payload.pop("content_hash")
    assert stable_content_hash(payload) == content_hash
    return evidence


def test_existence_constraint_emits_hash_bound_v5_cost_and_satisfaction() -> None:
    must_keep = _constraint(
        constraint_id="keep_b",
        target="b",
        relation="must_keep",
        value=True,
        strength=ConstraintStrength.LOCKED,
    )
    parent = _plan(constraints=(must_keep,))
    child = _child(parent, stops=(dict(parent.selected_stops[0]),))

    record = compute_plan_diff(parent, child).to_record()
    evidence = _evidence(record)

    assert record["deleted_stops"] == [
        {
            "stop_id": "b",
            "day": 1,
            "order": 2,
            "owner_strength": "locked",
            "cost": 2000.0,
        }
    ]
    assert set(evidence) == {
        "schema_version",
        "parent_plan_id",
        "parent_plan_content_hash",
        "semantic_scope",
        "ownership_policy",
        "constraints",
        "changes",
        "content_hash",
    }
    assert evidence["parent_plan_content_hash"] == parent.content_hash
    assert evidence["semantic_scope"] == {
        "cost_semantics": "attribute_sensitivity",
        "permission_claim": "not_evaluated",
        "feasibility_claim": "not_evaluated",
        "cost_coverage": "all_listed_diff_changes",
        "relation_specific_attributes": [
            "day",
            "duration",
            "end_time",
            "existence",
            "order",
            "start_time",
            "time_window",
        ],
        "lodging_and_road_relations": "not_migrated",
    }
    assert evidence["constraints"][0]["canonical_value"] is True
    change = evidence["changes"][0]
    assert change["change_kind"] == "deleted_stop"
    assert change["before_value"] == {"existence": True}
    assert change["after_value"] == {"existence": False}
    assert change["constraint_satisfaction"] == [
        {
            "constraint_id": "keep_b",
            "attribute": "existence",
            "canonical_value": True,
            "satisfaction_before": "satisfied",
            "satisfaction_after": "unsatisfied",
        }
    ]
    assert change["owner_strength"] == "locked"
    assert change["cost"] == 2000.0


def test_day_cost_is_relation_specific_and_existence_is_independent() -> None:
    keep_a = _constraint(
        constraint_id="keep_a",
        target="a",
        relation="must_keep",
        value=True,
        strength=ConstraintStrength.LOCKED,
    )
    parent = _plan(constraints=(keep_a,))
    changed = [dict(stop) for stop in parent.selected_stops]
    changed[0]["day"] = 2
    unrelated = compute_plan_diff(parent, _child(parent, stops=tuple(changed))).to_record()
    assert unrelated["schema_version"] == "plan-diff-v1"
    assert unrelated["day_moves"][0]["owner_strength"] == ""
    assert unrelated["day_moves"][0]["cost"] == 2.0
    assert "attribute_ownership_evidence" not in unrelated

    day_two = _constraint(
        constraint_id="day_a",
        target="a",
        relation="day",
        value=2,
    )
    owned_parent = _plan(constraints=(keep_a, day_two))
    owned = compute_plan_diff(
        owned_parent,
        _child(owned_parent, stops=tuple(changed)),
    ).to_record()
    evidence = _evidence(owned)
    assert owned["day_moves"][0]["owner_strength"] == "soft"
    assert owned["day_moves"][0]["cost"] == 20.0
    row = next(
        item for item in evidence["changes"] if item["change_kind"] == "day_move"
    )
    assert row["applicable_constraint_ids"] == ["day_a"]
    assert row["before_value"] == {"day": 1}
    assert row["after_value"] == {"day": 2}
    assert row["constraint_satisfaction"][0]["satisfaction_before"] == "unsatisfied"
    assert row["constraint_satisfaction"][0]["satisfaction_after"] == "satisfied"


@pytest.mark.parametrize("mutation", ["inject", "remove", "upgrade"])
def test_child_constraint_mutation_cannot_change_v5_cost(mutation: str) -> None:
    day_two = _constraint(
        constraint_id="day_a",
        target="a",
        relation="day",
        value=2,
    )
    parent = _plan(constraints=(day_two,))
    changed = [dict(stop) for stop in parent.selected_stops]
    changed[0]["day"] = 2
    if mutation == "inject":
        child_constraints = parent.owned_constraints + (
            _constraint(
                constraint_id="injected",
                target="a",
                relation="day",
                value=2,
            ),
        )
    elif mutation == "remove":
        child_constraints = ()
    else:
        upgraded = dict(day_two)
        upgraded["strength"] = "locked"
        child_constraints = (upgraded,)

    with pytest.raises(ValueError, match="exactly match accepted parent"):
        compute_plan_diff(
            parent,
            _child(parent, stops=tuple(changed), constraints=child_constraints),
        )


def test_contradictory_or_malformed_relevant_constraints_fail_deterministically() -> None:
    day_two = _constraint(
        constraint_id="day_two",
        target="a",
        relation="day",
        value=2,
    )
    day_three = _constraint(
        constraint_id="day_three",
        target="a",
        relation="day",
        value=3,
    )
    changed = [dict(stop) for stop in _plan().selected_stops]
    changed[0]["day"] = 2
    for constraints in ((day_two, day_three), (day_three, day_two)):
        parent = _plan(constraints=constraints)
        with pytest.raises(ValueError, match="contradictory"):
            compute_plan_diff(parent, _child(parent, stops=tuple(changed)))

    malformed = dict(day_two)
    malformed["value"] = True
    parent = _plan(constraints=(malformed,))
    with pytest.raises(ValueError, match="day value is invalid"):
        compute_plan_diff(parent, _child(parent, stops=tuple(changed)))

    duplicate = dict(day_two)
    duplicate["value"] = 2
    parent = _plan(constraints=(day_two, duplicate))
    with pytest.raises(ValueError, match="identity or scope"):
        compute_plan_diff(parent, _child(parent, stops=tuple(changed)))

    extra = dict(day_two)
    extra["browser_owned"] = True
    parent = _plan(constraints=(extra,))
    with pytest.raises(ValueError, match="owned constraint record"):
        compute_plan_diff(parent, _child(parent, stops=tuple(changed)))


def test_duration_window_and_order_evidence_use_exact_noncoercing_schemas() -> None:
    float_duration = _constraint(
        constraint_id="duration_a",
        target="a",
        relation="duration",
        value={
            "mode": "exact",
            "preferred_minutes": 60.0,
            "minimum_minutes": 60.0,
            "maximum_minutes": 60.0,
        },
    )
    parent = _plan(constraints=(float_duration,))
    changed = [dict(stop) for stop in parent.selected_stops]
    changed[0]["duration_constraint"] = {
        "mode": "exact",
        "preferred_minutes": 60,
        "minimum_minutes": 60,
        "maximum_minutes": 60,
    }
    changed[0]["visit_duration_minutes"] = 60
    with pytest.raises(ValueError, match="canonical exact"):
        compute_plan_diff(parent, _child(parent, stops=tuple(changed)))

    one_sided = _constraint(
        constraint_id="window_a",
        target="a",
        relation="required_time_window",
        value={
            "schema_version": "stop-time-window-constraint-v1",
            "earliest_arrival": "10:00",
            "latest_departure": None,
            "early_arrival_policy": "wait_until_earliest_arrival",
            "latest_departure_semantics": "departure_after_visit",
        },
    )
    parent = _plan(constraints=(one_sided,))
    changed = [dict(stop) for stop in parent.selected_stops]
    changed[0]["time_window_constraint"] = one_sided["value"]
    window_record = compute_plan_diff(
        parent,
        _child(parent, stops=tuple(changed)),
    ).to_record()
    assert window_record["schema_version"] == "plan-diff-v5"
    assert window_record["time_window_changes"][0]["owner_strength"] == "soft"

    ambiguous_order = _constraint(
        constraint_id="order_a",
        target="a",
        relation="sequence_order",
        value=1,
    )
    parent = _plan(constraints=(ambiguous_order,))
    changed = [dict(stop) for stop in parent.selected_stops]
    changed[0]["stop_order"] = 2
    changed[1]["stop_order"] = 1
    order_record = compute_plan_diff(
        parent,
        _child(parent, stops=tuple(changed)),
    ).to_record()
    assert order_record["schema_version"] == "plan-diff-v1"
    assert all(row["owner_strength"] == "" for row in order_record["reorder_changes"])

    one_based_order = _constraint(
        constraint_id="stop_order_a",
        target="a",
        relation="attribute:stop_order",
        value=2,
    )
    parent = _plan(constraints=(one_based_order,))
    changed = [dict(stop) for stop in parent.selected_stops]
    changed[0]["stop_order"] = 2
    changed[1]["stop_order"] = 1
    one_based_record = compute_plan_diff(
        parent,
        _child(parent, stops=tuple(changed)),
    ).to_record()
    evidence = _evidence(one_based_record)
    constrained = next(
        row
        for row in evidence["changes"]
        if row["change_kind"] == "order_change"
        and row["target_stop_id"] == "a"
    )
    assert constrained["before_value"] == {"order": 1}
    assert constrained["after_value"] == {"order": 2}
    assert constrained["owner_strength"] == "soft"
    assert constrained["cost"] == 5.0
    assert constrained["constraint_satisfaction"][0]["satisfaction_before"] == (
        "unsatisfied"
    )
    assert constrained["constraint_satisfaction"][0]["satisfaction_after"] == (
        "satisfied"
    )

    for invalid_order in (True, 0, -1, 4097):
        invalid = _constraint(
            constraint_id="invalid_order",
            target="a",
            relation="stop_order",
            value=invalid_order,
        )
        parent = _plan(constraints=(invalid,))
        with pytest.raises(ValueError, match="one-based order value is invalid"):
            compute_plan_diff(parent, _child(parent, stops=tuple(changed)))


def test_time_aliases_are_isolated_and_invalid_clocks_fail_closed() -> None:
    start_only = _constraint(
        constraint_id="start_a",
        target="a",
        relation="start_time",
        value="09:30",
    )
    parent = _plan(constraints=(start_only,))
    end_changed = [dict(stop) for stop in parent.selected_stops]
    end_changed[0]["end_time"] = "10:30"
    unrelated = compute_plan_diff(
        parent,
        _child(parent, stops=tuple(end_changed)),
    ).to_record()
    assert unrelated["schema_version"] == "plan-diff-v1"
    assert unrelated["time_shifts"][0]["owner_strength"] == ""

    end_only = _constraint(
        constraint_id="end_a",
        target="a",
        relation="attribute:end_time",
        value="10:30",
    )
    parent = _plan(constraints=(end_only,))
    owned = compute_plan_diff(
        parent,
        _child(parent, stops=tuple(end_changed)),
    ).to_record()
    evidence = _evidence(owned)
    row = next(
        item for item in evidence["changes"] if item["change_kind"] == "time_shift"
    )
    assert row["attributes"] == ["end_time"]
    assert row["before_value"] == {"end_time": None}
    assert row["after_value"] == {"end_time": "10:30"}
    assert row["constraint_satisfaction"][0]["satisfaction_before"] == "unavailable"
    assert row["constraint_satisfaction"][0]["satisfaction_after"] == "satisfied"

    invalid_clock = _constraint(
        constraint_id="bad_clock",
        target="a",
        relation="start_time",
        value="9:30",
    )
    parent = _plan(constraints=(invalid_clock,))
    start_changed = [dict(stop) for stop in parent.selected_stops]
    start_changed[0]["start_time"] = "09:30"
    with pytest.raises(ValueError, match="exact HH:MM"):
        compute_plan_diff(parent, _child(parent, stops=tuple(start_changed)))


def test_two_bound_time_shift_uses_canonical_attribute_order() -> None:
    start = _constraint(
        constraint_id="start_a",
        target="a",
        relation="attribute:start_time",
        value="09:30",
    )
    end = _constraint(
        constraint_id="end_a",
        target="a",
        relation="attribute:end_time",
        value="10:30",
    )
    parent = _plan(constraints=(start, end))
    changed = [dict(stop) for stop in parent.selected_stops]
    changed[0]["start_time"] = "09:30"
    changed[0]["end_time"] = "10:30"

    record = compute_plan_diff(
        parent,
        _child(parent, stops=tuple(changed)),
    ).to_record()
    evidence = _evidence(record)
    row = next(
        item for item in evidence["changes"] if item["change_kind"] == "time_shift"
    )
    assert row["attributes"] == ["end_time", "start_time"]
    assert row["applicable_constraint_ids"] == ["end_a", "start_a"]
    assert row["before_value"] == {
        "end_time": None,
        "start_time": "09:00",
    }
    assert row["after_value"] == {
        "end_time": "10:30",
        "start_time": "09:30",
    }


@pytest.mark.parametrize(
    "earliest,latest",
    [(None, None), ("12:00", "11:00")],
)
def test_invalid_time_window_constraint_values_fail_closed(
    earliest: str | None,
    latest: str | None,
) -> None:
    invalid = _constraint(
        constraint_id="window_a",
        target="a",
        relation="time_window",
        value={
            "schema_version": "stop-time-window-constraint-v1",
            "earliest_arrival": earliest,
            "latest_departure": latest,
            "early_arrival_policy": "wait_until_earliest_arrival",
            "latest_departure_semantics": "departure_after_visit",
        },
    )
    parent = _plan(constraints=(invalid,))
    changed = [dict(stop) for stop in parent.selected_stops]
    changed[0]["time_window_constraint"] = {
        "schema_version": "stop-time-window-constraint-v1",
        "earliest_arrival": "10:00",
        "latest_departure": None,
        "early_arrival_policy": "wait_until_earliest_arrival",
        "latest_departure_semantics": "departure_after_visit",
    }
    with pytest.raises(ValueError, match="time-window value must be canonical"):
        compute_plan_diff(parent, _child(parent, stops=tuple(changed)))


def test_equivalent_constraint_order_has_deterministic_sanitized_rows() -> None:
    weak = _constraint(
        constraint_id="a_weak",
        target="a",
        relation="day",
        value=2,
        strength=ConstraintStrength.WEAK,
    )
    soft = _constraint(
        constraint_id="z_soft",
        target="a",
        relation="assigned_day",
        value=2,
        strength=ConstraintStrength.SOFT,
    )
    changed = [dict(stop) for stop in _plan().selected_stops]
    changed[0]["day"] = 2
    records = []
    for constraints in ((soft, weak), (weak, soft)):
        parent = _plan(constraints=constraints)
        child = _child(parent, stops=tuple(changed), constraints=tuple(reversed(constraints)))
        records.append(compute_plan_diff(parent, child).to_record())

    for record in records:
        evidence = _evidence(record)
        assert [row["constraint_id"] for row in evidence["constraints"]] == [
            "a_weak",
            "z_soft",
        ]
        day_row = next(
            row for row in evidence["changes"] if row["change_kind"] == "day_move"
        )
        assert day_row["applicable_constraint_ids"] == ["a_weak", "z_soft"]
        assert day_row["selected_constraint_ids"] == ["z_soft"]
        assert day_row["owner_strength"] == "soft"


def test_v5_is_domain_only_and_product_diff_verifier_fails_closed() -> None:
    day_two = _constraint(
        constraint_id="day_a",
        target="a",
        relation="attribute:day",
        value=2,
    )
    parent = _plan(constraints=(day_two,))
    changed = [dict(stop) for stop in parent.selected_stops]
    changed[0]["day"] = 2
    record = compute_plan_diff(parent, _child(parent, stops=tuple(changed))).to_record()

    with pytest.raises(WorkspaceError, match="draft_pipeline_diff_invalid"):
        _verify_diff(record)


def test_nonfinite_policy_and_forged_v5_dto_fail_closed() -> None:
    day_two = _constraint(
        constraint_id="day_a",
        target="a",
        relation="day",
        value=2,
    )
    parent = _plan(constraints=(day_two,))
    changed = [dict(stop) for stop in parent.selected_stops]
    changed[0]["day"] = 2
    child = _child(parent, stops=tuple(changed))
    for invalid in (float("nan"), float("inf")):
        policy = OwnershipPolicy(
            change_type_weights={**OwnershipPolicy().change_type_weights, "day_move": invalid}
        )
        with pytest.raises(ValueError, match="finite"):
            compute_plan_diff(parent, child, policy)
        policy = OwnershipPolicy(
            constraint_strength_weights={
                **OwnershipPolicy().constraint_strength_weights,
                "soft": invalid,
            }
        )
        with pytest.raises(ValueError, match="finite"):
            compute_plan_diff(parent, child, policy)

    valid = compute_plan_diff(parent, child)
    assert isinstance(valid, PlanDiff)
    missing = replace(valid, attribute_ownership_evidence=None)
    with pytest.raises(ValueError, match="requires ownership evidence"):
        missing.to_record()

    forged_evidence = dict(valid.attribute_ownership_evidence or {})
    forged_evidence["content_hash"] = "0" * 16
    forged = replace(valid, attribute_ownership_evidence=forged_evidence)
    with pytest.raises(ValueError, match="ownership evidence is invalid"):
        forged.to_record()

    role_mixed = replace(
        valid,
        role_changes=(
            RoleChange(
                stop_id="a",
                day=2,
                from_role=None,
                to_role="meal",
                from_source="unavailable",
                to_source="user_declared_itinerary_role",
            ),
        ),
    )
    with pytest.raises(ValueError, match="excludes role changes"):
        role_mixed.to_record()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("target_stop_id", "b"),
        ("change_kind", "order_change"),
        ("attributes", ["order"]),
        ("before_value", {"day": 3}),
    ],
)
def test_coherently_rehashed_change_evidence_must_match_typed_diff(
    field: str,
    replacement: object,
) -> None:
    day_two = _constraint(
        constraint_id="day_a",
        target="a",
        relation="day",
        value=2,
    )
    parent = _plan(constraints=(day_two,))
    changed = [dict(stop) for stop in parent.selected_stops]
    changed[0]["day"] = 2
    valid = compute_plan_diff(parent, _child(parent, stops=tuple(changed)))
    forged = deepcopy(valid.attribute_ownership_evidence)
    assert isinstance(forged, dict)
    forged["changes"][0][field] = replacement
    payload = dict(forged)
    payload.pop("content_hash")
    forged["content_hash"] = stable_content_hash(payload)

    with pytest.raises(ValueError, match="ownership change evidence is invalid"):
        replace(valid, attribute_ownership_evidence=forged).to_record()


def test_unconsumed_coherently_rehashed_constraint_evidence_is_rejected() -> None:
    day_two = _constraint(
        constraint_id="day_a",
        target="a",
        relation="day",
        value=2,
    )
    parent = _plan(constraints=(day_two,))
    changed = [dict(stop) for stop in parent.selected_stops]
    changed[0]["day"] = 2
    valid = compute_plan_diff(parent, _child(parent, stops=tuple(changed)))
    forged = deepcopy(valid.attribute_ownership_evidence)
    assert isinstance(forged, dict)
    extra = dict(forged["constraints"][0])
    extra.update(
        {
            "constraint_id": "unused_day_b",
            "target_stop_id": "b",
            "canonical_value": 1,
        }
    )
    forged["constraints"].append(extra)
    forged["constraints"].sort(key=lambda row: row["constraint_id"])
    payload = dict(forged)
    payload.pop("content_hash")
    forged["content_hash"] = stable_content_hash(payload)

    with pytest.raises(ValueError, match="ownership constraint evidence is invalid"):
        replace(valid, attribute_ownership_evidence=forged).to_record()
