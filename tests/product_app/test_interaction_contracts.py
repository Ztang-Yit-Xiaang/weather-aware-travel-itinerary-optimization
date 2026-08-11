from __future__ import annotations

from itinerary_system.product_app.interaction_contracts import (
    V1_OPERATION_MIGRATION,
    CommitmentStrengthV1,
    DraftOperationTypeV2,
    DurationModeV1,
    FeedbackTierV1,
    InteractionStateV1,
    StopRoleV1,
)


def test_direct_interaction_operation_vocabulary_is_closed_and_complete() -> None:
    assert {value.value for value in DraftOperationTypeV2} == {
        "relocate_custom_waypoint",
        "replace_stop_near_location",
        "add_custom_waypoint",
        "add_place_stop",
        "add_route_waypoint",
        "set_stop_role",
        "set_stop_day",
        "set_stop_order",
        "set_stop_duration",
        "set_stop_time_window",
        "set_stop_commitment",
        "set_attribute_constraint",
        "remove_stop",
        "exclude_place",
        "add_route_via",
        "avoid_route_segment",
        "avoid_road",
        "avoid_area",
        "protect_corridor",
        "change_route_mode",
        "change_route_preference",
        "change_departure_time",
        "report_route_issue",
    }


def test_stop_duration_commitment_and_feedback_vocabularies_are_closed() -> None:
    assert {value.value for value in StopRoleV1} == {
        "attraction",
        "activity",
        "meal",
        "lodging",
        "transport_hub",
        "rest_stop",
        "scenic_stop",
        "route_waypoint",
        "origin",
        "destination",
    }
    assert {value.value for value in DurationModeV1} == {
        "exact",
        "preferred",
        "minimum",
        "maximum",
        "range",
    }
    assert {value.value for value in CommitmentStrengthV1} == {
        "optional",
        "prefer_keep",
        "strong_preference",
        "must_keep",
        "booked",
        "excluded",
    }
    assert {value.value for value in FeedbackTierV1} == {
        "visual_only",
        "route_checked",
        "evaluated",
    }


def test_interaction_state_keeps_preview_and_evaluated_states_distinct() -> None:
    assert InteractionStateV1.GHOST_PREVIEW != InteractionStateV1.SNAP_PREVIEW
    assert InteractionStateV1.ROUTE_CHECKED != InteractionStateV1.EVALUATED
    assert InteractionStateV1.EVALUATED != InteractionStateV1.ACCEPTED


def test_v1_operation_names_have_explicit_migration_targets() -> None:
    assert set(V1_OPERATION_MIGRATION) == {
        "keep_stop",
        "lock_stop",
        "mark_flexible",
        "move_day",
        "route_feedback",
        "replace_nearby",
        "add_candidate",
    }
    assert all(isinstance(value, DraftOperationTypeV2) for value in V1_OPERATION_MIGRATION.values())
    assert V1_OPERATION_MIGRATION["add_candidate"] is DraftOperationTypeV2.ADD_PLACE_STOP
