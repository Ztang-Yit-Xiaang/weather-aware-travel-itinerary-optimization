"""Removing a stop in the coherent demo: chat intent -> repair -> independent evaluation.

A removal drops one accepted stop with no replacement. The repair heuristic
rejoins the stops around it with the same bypass leg a move away would use,
and the evaluator checks the trip without it. The fresh demo session still has
the weather-unsafe Bixby stop, so each removal is staged next to that fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "product_app_registry.json"
REPLACE_BIXBY = {
    "type": "replace_nearby",
    "target": "bixby_creek_bridge_viewpoint",
    "parameters": {"candidate_id": "point_lobos_state_natural_reserve"},
}


def _remove(stop_id: str) -> dict:
    return {"type": "remove_stop", "target": stop_id, "parameters": {}}


def _stage(client: TestClient, operations: list[dict]):
    created = client.post("/api/sessions", json={}).json()
    session_id = created["session"]["session_id"]
    headers = {"X-Session-Token": created["mutation_token"]}
    staged = client.post(
        f"/api/sessions/{session_id}/draft/operations",
        headers=headers,
        json={
            "expected_revision": created["session"]["revision"],
            "operations": [{**row, "source": "typed_stop_editor", "evidence_refs": []} for row in operations],
        },
    )
    return created, session_id, headers, staged


@pytest.fixture
def client(tmp_path):
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    "stop_id",
    [
        "sequoia_national_park",  # one of two stops on its day
        "stearns_wharf",  # the only stop on its day
    ],
)
def test_removal_is_evaluated_and_the_stop_leaves_the_trip(client, stop_id):
    created, session_id, headers, staged = _stage(client, [REPLACE_BIXBY, _remove(stop_id)])
    capability = created["workspace"]["draft_capabilities"]["operations"]["remove_stop"]
    assert capability["preview_executable"] is True
    assert staged.status_code == 200, staged.text
    preview = client.post(
        f"/api/sessions/{session_id}/preview",
        headers=headers,
        json={"expected_revision": staged.json()["session"]["revision"]},
    )
    assert preview.status_code == 200, preview.text
    proposal = preview.json()["proposal"]
    assert proposal["state"] == "eligible", proposal["message"]
    assert proposal["acceptance_eligible"] is True
    constraints = proposal["runtime_execution"]["repair_request"]["confirmed_constraints"]
    assert stop_id in constraints["must_delete"] and stop_id not in constraints["must_include"]
    assert constraints["remove_stops"] == [stop_id]


def test_a_removed_stop_cannot_also_be_kept(client):
    _, _, _, staged = _stage(
        client,
        [_remove("sequoia_national_park"), {"type": "keep_stop", "target": "sequoia_national_park", "parameters": {}}],
    )
    assert staged.status_code == 409
    assert "draft_conflicts_with_removal" in staged.text


def test_removal_needs_an_accepted_stop_and_no_parameters(client):
    _, _, _, unknown = _stage(client, [_remove("point_lobos_state_natural_reserve")])
    assert unknown.status_code == 422
    _, _, _, with_parameters = _stage(
        client, [{"type": "remove_stop", "target": "sequoia_national_park", "parameters": {"day": 2}}]
    )
    assert with_parameters.status_code == 422
