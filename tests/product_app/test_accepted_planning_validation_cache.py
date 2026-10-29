"""Reuse only material bound to freshly verified snapshot and ancestor hashes."""

from copy import deepcopy
from threading import RLock
from types import SimpleNamespace

import pytest

from itinerary_system.product_app import trip_replanner
from itinerary_system.product_app.service import ProductService
from itinerary_system.product_app.workspace import WorkspaceError
from itinerary_system.research_artifacts import stable_content_hash


@pytest.fixture
def baseline(monkeypatch):
    parent = {"plan_id": "ancestor", "content_hash": "ancestor-hash"}
    child = {
        "plan_id": "accepted",
        "content_hash": "child-hash",
        "parent_plan_id": "ancestor",
        "context_snapshot_id": "ctx",
        "catalog_snapshot_id": "catalog",
    }
    route = {
        "publication_ready": True,
        "matrix_id": "matrix",
        "context_snapshot_id": "ctx",
        "source_bundle_id": "roads",
        "source_content_sha256": "road-hash",
    }
    cert = {
        "plan_id": "accepted",
        "plan_content_hash": "child-hash",
        "comparison_eligibility": "eligible",
        "evaluation_status": "PASSED",
        "route_validation": route,
    }
    cert["content_hash"] = stable_content_hash(cert)
    snapshot = {
        "content_hash": "verified-snapshot-a",
        "trip_id": "trip",
        "run_id": "run",
        "plan": child,
        "certificate": cert,
        "diff": {"parent_plan_id": "ancestor", "child_plan_id": "accepted"},
        "planning_bundle": {
            "parent_plan_id": "ancestor",
            "parent_content_hash": "ancestor-hash",
            "content_hash": "declared-planning-hash",
            "route_matrix": route,
        },
    }
    calls = []

    def validate(ancestor, planning, **kwargs):
        calls.append(deepcopy(planning))
        if planning.get("corrupt") or ancestor["content_hash"] != planning["parent_content_hash"]:
            raise WorkspaceError("trip_planning_bundle_invalid", 409)
        return (
            SimpleNamespace(plan_id="ancestor", context_snapshot_id="ctx", catalog_snapshot_id="catalog"),
            {},
            {},
            {"comparison_eligibility": "eligible"},
        )

    monkeypatch.setattr(trip_replanner, "validate_planning_bundle", validate)
    service = object.__new__(ProductService)
    service._accepted_compiler_lock = RLock()
    service._accepted_planning_validation = None
    service.load = lambda run: (SimpleNamespace(parent_plan={"plan_id": "original"}), {})
    service.registry = SimpleNamespace(get=lambda run: SimpleNamespace(trip_id="trip"))
    service.acceptance = SimpleNamespace(
        load_pointer=lambda trip: {
            "accepted_plan_id": "accepted",
            "accepted_plan_content_hash": "child-hash",
            "original_plan_id": "original",
            "decision_id": "decision",
        },
        load_snapshot=lambda decision: deepcopy(snapshot),
        history=lambda trip: [
            {
                "decision_id": "decision",
                "snapshot_content_hash": snapshot["content_hash"],
                "selected_plan_id": "accepted",
                "selected_plan_content_hash": "child-hash",
            }
        ],
        plan_repository=SimpleNamespace(
            load=lambda identity: SimpleNamespace(
                to_record=lambda **kwargs: deepcopy(child if identity == "accepted" else parent)
            )
        ),
    )
    return service, snapshot, parent, child, calls


def test_same_verified_snapshot_reuses_only_the_expensive_planning_validation(baseline):
    service, _, _, _, calls = baseline
    for _ in range(3):
        service._current_baseline("run")
    assert len(calls) == 1


def test_changed_snapshot_cannot_reuse_declared_inner_hash(baseline):
    service, snapshot, _, _, calls = baseline
    service._current_baseline("run")
    snapshot["planning_bundle"]["corrupt"] = True
    snapshot["content_hash"] = "verified-snapshot-b"
    with pytest.raises(WorkspaceError, match="trip_planning_bundle_invalid"):
        service._current_baseline("run")
    assert len(calls) == 2


def test_changed_ancestor_content_is_revalidated(baseline):
    service, _, parent, _, calls = baseline
    service._current_baseline("run")
    parent["content_hash"] = "changed-ancestor"
    with pytest.raises(WorkspaceError, match="trip_planning_bundle_invalid"):
        service._current_baseline("run")
    assert len(calls) == 2


def test_cache_hit_still_checks_current_parent_context_and_certificate(baseline):
    service, snapshot, _, child, calls = baseline
    service._current_baseline("run")
    child["context_snapshot_id"] = "wrong-context"
    with pytest.raises(WorkspaceError, match="accepted_baseline_route_evidence_mismatch"):
        service._current_baseline("run")
    child["context_snapshot_id"] = "ctx"
    snapshot["certificate"]["evaluation_status"] = "FAILED"
    with pytest.raises(WorkspaceError, match="accepted_baseline_evidence_mismatch"):
        service._current_baseline("run")
    assert len(calls) == 1


def test_cache_retains_one_content_identity(baseline):
    service, snapshot, _, _, calls = baseline
    for identity in ("a", "b", "a"):
        snapshot["content_hash"] = identity
        service._current_baseline("run")
    assert len(calls) == 3
