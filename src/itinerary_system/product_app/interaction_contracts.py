"""Versioned W4R direct-interaction vocabularies.

These contracts are frozen before the corresponding operations are enabled in
WorkspaceStore. Their presence is not an execution capability claim.
"""

from __future__ import annotations

from enum import StrEnum


class DraftOperationTypeV2(StrEnum):
    RELOCATE_CUSTOM_WAYPOINT = "relocate_custom_waypoint"
    REPLACE_STOP_NEAR_LOCATION = "replace_stop_near_location"
    ADD_CUSTOM_WAYPOINT = "add_custom_waypoint"
    ADD_PLACE_STOP = "add_place_stop"
    ADD_ROUTE_WAYPOINT = "add_route_waypoint"
    SET_STOP_ROLE = "set_stop_role"
    SET_STOP_DAY = "set_stop_day"
    SET_STOP_ORDER = "set_stop_order"
    SET_STOP_DURATION = "set_stop_duration"
    SET_STOP_TIME_WINDOW = "set_stop_time_window"
    SET_STOP_COMMITMENT = "set_stop_commitment"
    SET_ATTRIBUTE_CONSTRAINT = "set_attribute_constraint"
    REMOVE_STOP = "remove_stop"
    EXCLUDE_PLACE = "exclude_place"
    ADD_ROUTE_VIA = "add_route_via"
    AVOID_ROUTE_SEGMENT = "avoid_route_segment"
    AVOID_ROAD = "avoid_road"
    AVOID_AREA = "avoid_area"
    PROTECT_CORRIDOR = "protect_corridor"
    CHANGE_ROUTE_MODE = "change_route_mode"
    CHANGE_ROUTE_PREFERENCE = "change_route_preference"
    CHANGE_DEPARTURE_TIME = "change_departure_time"
    REPORT_ROUTE_ISSUE = "report_route_issue"


class StopRoleV1(StrEnum):
    ATTRACTION = "attraction"
    ACTIVITY = "activity"
    MEAL = "meal"
    LODGING = "lodging"
    TRANSPORT_HUB = "transport_hub"
    REST_STOP = "rest_stop"
    SCENIC_STOP = "scenic_stop"
    ROUTE_WAYPOINT = "route_waypoint"
    ORIGIN = "origin"
    DESTINATION = "destination"


class DurationModeV1(StrEnum):
    EXACT = "exact"
    PREFERRED = "preferred"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    RANGE = "range"


class CommitmentStrengthV1(StrEnum):
    OPTIONAL = "optional"
    PREFER_KEEP = "prefer_keep"
    STRONG_PREFERENCE = "strong_preference"
    MUST_KEEP = "must_keep"
    BOOKED = "booked"
    EXCLUDED = "excluded"


class ConstraintScopeLifetimeV1(StrEnum):
    CURRENT_DRAFT_ONLY = "current_draft_only"
    CURRENT_REPAIR_SESSION = "current_repair_session"
    REMAINDER_OF_TRIP = "remainder_of_trip"


class AttributeConstraintNameV1(StrEnum):
    EXISTENCE = "existence"
    DAY = "day"
    DURATION = "duration"
    SEQUENCE_ORDER = "sequence_order"
    ROLE = "role"


class RouteIssueTypeV1(StrEnum):
    SUSPECTED_CLOSURE = "suspected_closure"
    ACCESS_PROBLEM = "access_problem"
    ROUTE_QUALITY = "route_quality"


class InteractionStateV1(StrEnum):
    IDLE = "idle"
    SELECTED = "selected"
    EDITING = "editing"
    GHOST_PREVIEW = "ghost_preview"
    EXPLORATORY_PREVIEW = "exploratory_preview"
    SNAP_PREVIEW = "snap_preview"
    PENDING_CONFIRMATION = "pending_confirmation"
    DRAFT_APPENDED = "draft_appended"
    ROUTE_CHECKED = "route_checked"
    REPAIR_RUNNING = "repair_running"
    EVALUATED = "evaluated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


class FeedbackTierV1(StrEnum):
    VISUAL_ONLY = "visual_only"
    ROUTE_CHECKED = "route_checked"
    EVALUATED = "evaluated"


V1_OPERATION_MIGRATION: dict[str, DraftOperationTypeV2] = {
    "keep_stop": DraftOperationTypeV2.SET_STOP_COMMITMENT,
    "lock_stop": DraftOperationTypeV2.SET_ATTRIBUTE_CONSTRAINT,
    "mark_flexible": DraftOperationTypeV2.SET_ATTRIBUTE_CONSTRAINT,
    "move_day": DraftOperationTypeV2.SET_STOP_DAY,
    "route_feedback": DraftOperationTypeV2.REPORT_ROUTE_ISSUE,
    "replace_nearby": DraftOperationTypeV2.REPLACE_STOP_NEAR_LOCATION,
    "add_candidate": DraftOperationTypeV2.ADD_PLACE_STOP,
}


TYPED_EDIT_OPERATION_NAMES = (
    DraftOperationTypeV2.SET_STOP_ROLE.value,
    DraftOperationTypeV2.SET_STOP_DAY.value,
    DraftOperationTypeV2.SET_STOP_ORDER.value,
    DraftOperationTypeV2.SET_STOP_DURATION.value,
    DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value,
    DraftOperationTypeV2.SET_STOP_COMMITMENT.value,
    DraftOperationTypeV2.SET_ATTRIBUTE_CONSTRAINT.value,
    DraftOperationTypeV2.CHANGE_ROUTE_PREFERENCE.value,
    DraftOperationTypeV2.REPORT_ROUTE_ISSUE.value,
)

EVALUATED_TYPED_EDIT_OPERATIONS = frozenset(
    {
        DraftOperationTypeV2.SET_STOP_DAY.value,
        DraftOperationTypeV2.SET_STOP_ORDER.value,
        DraftOperationTypeV2.SET_STOP_ROLE.value,
        DraftOperationTypeV2.SET_STOP_DURATION.value,
        DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value,
    }
)
DRAFT_ONLY_TYPED_EDIT_OPERATIONS = frozenset(TYPED_EDIT_OPERATION_NAMES) - EVALUATED_TYPED_EDIT_OPERATIONS


def typed_edit_capabilities() -> dict[str, object]:
    """Return the closed W4R edit vocabulary without overstating execution support."""

    operations: dict[str, dict[str, object]] = {}
    for operation in TYPED_EDIT_OPERATION_NAMES:
        enabled = operation != DraftOperationTypeV2.CHANGE_ROUTE_PREFERENCE.value
        evaluated = operation in EVALUATED_TYPED_EDIT_OPERATIONS
        blocking_code = None if evaluated else "full_evaluation_not_supported"
        if not enabled:
            blocking_code = "route_preference_not_supported"
        operations[operation] = {
            "enabled": enabled,
            "feedback_tier": "evaluated" if evaluated else "draft_only",
            "preview_executable": evaluated,
            "evaluated_repair": evaluated,
            "blocking_code": blocking_code,
        }
    operations[DraftOperationTypeV2.SET_STOP_COMMITMENT.value].update(
        {
            "protected_strengths": [
                CommitmentStrengthV1.MUST_KEEP.value,
                CommitmentStrengthV1.BOOKED.value,
            ],
            "protected_strengths_blocking_code": "commitment_permission_required",
        }
    )
    operations[DraftOperationTypeV2.SET_ATTRIBUTE_CONSTRAINT.value].update(
        {
            "protected_strengths": [
                CommitmentStrengthV1.MUST_KEEP.value,
                CommitmentStrengthV1.BOOKED.value,
            ],
            "protected_strengths_blocking_code": "commitment_permission_required",
        }
    )
    operations[DraftOperationTypeV2.SET_STOP_ORDER.value].update(
        {
            "supported_scope": "same_day",
            "sequence_index_base": 0,
        }
    )
    operations[DraftOperationTypeV2.SET_STOP_DURATION.value].update(
        {
            "feedback_tier": "conditional",
            "supported_evaluated_modes": [DurationModeV1.EXACT.value],
            "draft_only_modes": [
                DurationModeV1.PREFERRED.value,
                DurationModeV1.MINIMUM.value,
                DurationModeV1.MAXIMUM.value,
                DurationModeV1.RANGE.value,
            ],
            "unsupported_mode_blocking_code": "duration_mode_evaluation_not_supported",
            "scalar_plan_field": "visit_duration_minutes",
            "typed_plan_field": "duration_constraint",
        }
    )
    operations[DraftOperationTypeV2.SET_STOP_ROLE.value].update(
        {
            "feedback_tier": "conditional",
            "supported_evaluated_roles": [
                StopRoleV1.ATTRACTION.value,
                StopRoleV1.ACTIVITY.value,
                StopRoleV1.MEAL.value,
                StopRoleV1.REST_STOP.value,
                StopRoleV1.SCENIC_STOP.value,
            ],
            "draft_only_roles": [
                StopRoleV1.LODGING.value,
                StopRoleV1.TRANSPORT_HUB.value,
                StopRoleV1.ROUTE_WAYPOINT.value,
                StopRoleV1.ORIGIN.value,
                StopRoleV1.DESTINATION.value,
            ],
            "unsupported_role_blocking_code": (
                "stop_role_evaluation_not_supported"
            ),
            "typed_plan_field": "itinerary_role",
            "typed_source_field": "itinerary_role_source",
            "typed_source_value": "user_declared_itinerary_role",
            "combinable_operation_types": [
                DraftOperationTypeV2.SET_STOP_ROLE.value
            ],
        }
    )
    operations[DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value].update(
        {
            "typed_plan_field": "time_window_constraint",
            "constraint_schema_version": "stop-time-window-constraint-v1",
            "early_arrival_policy": "wait_until_earliest_arrival",
            "latest_departure_semantics": "departure_after_visit",
            "combinable_operation_types": [
                DraftOperationTypeV2.SET_STOP_TIME_WINDOW.value
            ],
        }
    )
    return {
        "schema_version": "product-typed-edit-capabilities-v1",
        "vocabularies": {
            "stop_roles": [item.value for item in StopRoleV1],
            "duration_modes": [item.value for item in DurationModeV1],
            "commitment_strengths": [item.value for item in CommitmentStrengthV1],
            "scope_lifetimes": [item.value for item in ConstraintScopeLifetimeV1],
            "attributes": [item.value for item in AttributeConstraintNameV1],
            "route_issue_types": [item.value for item in RouteIssueTypeV1],
        },
        "operations": operations,
    }
