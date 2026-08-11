from __future__ import annotations

import asyncio
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

from fastapi.testclient import TestClient

from itinerary_system.product_app.api import PRODUCT_ID, create_product_app
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.models import CopilotContextV1
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.workspace import WorkspaceError, WorkspaceStore

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "product_app_registry.json"


def client(tmp_path: Path, *, enable_legacy: bool = False) -> TestClient:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        enable_legacy=enable_legacy,
        additional_allowed_authorities=("testserver",),
    )
    return TestClient(app)


def create_session(test_client: TestClient) -> tuple[dict, dict[str, str]]:
    response = test_client.post("/api/sessions", json={})
    assert response.status_code == 200
    payload = response.json()
    return payload["session"], {"X-Session-Token": payload["mutation_token"]}


def test_registry_has_one_pinned_valid_default() -> None:
    registry = ProductRunRegistry(ROOT, REGISTRY)

    assert registry.default.run_id == "california_coast_product_demo_v2"
    assert registry.default.relative_path == "runs/california-coast-product-demo-v2"
    assert len(registry.default.manifest_hash) == 64
    assert registry.default.trip_id == "california_coast_demo"
    assert "multi_plan_product_demo" in registry.default.capabilities
    assert "local_acceptance" not in registry.default.capabilities


def test_registry_does_not_require_the_frozen_legacy_dashboard(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"run_id": "demo"}), encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "product-run-registry-v1",
                "runs": [
                    {
                        "run_id": "demo",
                        "trip_id": "demo_trip",
                        "label": "Demo fixture",
                        "relative_path": "runs/demo",
                        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
                        "capabilities": ["read_only_artifacts"],
                        "default": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    registry = ProductRunRegistry(tmp_path, registry_path)

    assert registry.default.run_id == "demo"
    assert not (run_dir / "dashboard_product").exists()


def test_health_redirect_shell_security_and_legacy_boundary(tmp_path: Path) -> None:
    with client(tmp_path) as test_client:
        health = test_client.get("/api/health").json()
        assert health["product_id"] == PRODUCT_ID
        assert health["ready"] is True
        assert health["legacy_enabled"] is False

        redirect = test_client.get("/", follow_redirects=False)
        assert redirect.status_code == 307
        assert redirect.headers["location"] == "/app"

        shell = test_client.get("/app/compare")
        assert shell.status_code == 200
        assert "Itinerary Repair Copilot" in shell.text
        assert "frame-ancestors 'none'" in shell.headers["content-security-policy"]
        assert test_client.get("/legacy/folium").status_code == 404


def test_deterministic_copilot_is_typed_and_fail_closed() -> None:
    adapter = DeterministicCopilotAdapter()
    context = CopilotContextV1(
        run_id="run_demo",
        trip_id="trip_demo",
        session_id="session_demo",
        session_revision=0,
        accepted_plan_id="plan_demo",
        selected_day=7,
        selected_stop_id=None,
        selected_segment_id=None,
        selected_candidate_id=None,
        selected_alternative_id=None,
        draft_operations=(),
        evaluated_proposal=None,
        allowed_stop_ids=(),
        allowed_candidate_ids=(),
        allowed_days=tuple(range(1, 8)),
    )

    async def interpret(message: str):
        return await adapter.interpret(context=context, history=(), message=message)

    proposal = asyncio.run(interpret("Review a safer weather repair"))
    assert proposal.state == "proposal_ready"
    assert proposal.intents[0].type == "review_registered_repair"
    assert "deterministic adapter" in proposal.assistant_message

    unsupported = asyncio.run(interpret("Do something magical"))
    assert unsupported.state == "clarification_required"

    booking = asyncio.run(interpret("Cancel reservation"))
    assert booking.state == "permission_required"


def test_authenticated_session_restore_survives_application_restart(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    first_app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=state_root,
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(first_app) as first:
        created = first.post("/api/sessions", json={}).json()
        session = created["session"]
        token = created["mutation_token"]
        selected = first.post(
            f"/api/sessions/{session['session_id']}/selection",
            headers={"X-Session-Token": token},
            json={
                "expected_revision": 0,
                "selected_day": 7,
                "selected_stop_id": "golden_gate_bridge",
            },
        )
        assert selected.status_code == 200

    second_app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=state_root,
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(second_app) as second:
        missing = second.get(f"/api/sessions/{session['session_id']}")
        restored = second.get(
            f"/api/sessions/{session['session_id']}",
            headers={"X-Session-Token": token},
        )

    assert missing.status_code == 403
    assert restored.status_code == 200
    payload = restored.json()
    assert payload["session"]["revision"] == 1
    assert payload["session"]["selected_stop_id"] == "golden_gate_bridge"
    assert payload["workspace"]["draft_capabilities"]["schema_version"] == "draft-capabilities-v1"
    assert token not in "".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in state_root.rglob("*")
        if path.is_file()
    )


def test_session_token_revision_draft_preview_and_acceptance_is_deferred(tmp_path: Path) -> None:
    parent_path = ROOT / "runs" / "e3ux-weather-repair-demo-v6" / "plans" / "plan_e1c4f803691e3188.json"
    parent_hash_before = sha256(parent_path.read_bytes()).hexdigest()

    with client(tmp_path) as test_client:
        session, headers = create_session(test_client)
        session_id = session["session_id"]
        parent_plan_id = session["accepted_plan_id"]

        missing_token = test_client.post(
            f"/api/sessions/{session_id}/selection",
            json={"expected_revision": 0, "selected_day": 7},
        )
        assert missing_token.status_code == 403

        selected = test_client.post(
            f"/api/sessions/{session_id}/selection",
            headers=headers,
            json={"expected_revision": 0, "selected_day": 7, "selected_stop_id": "bixby_creek_bridge_viewpoint"},
        )
        assert selected.status_code == 200
        session = selected.json()["session"]
        assert session["revision"] == 1

        stale = test_client.post(
            f"/api/sessions/{session_id}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 0,
                "type": "route_feedback",
                "target": "selected_route",
                "parameters": {"preference": "reduce_contextual_risk"},
            },
        )
        assert stale.status_code == 409
        assert stale.json()["detail"] == "stale_session_revision"

        drafted = test_client.post(
            f"/api/sessions/{session_id}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 1,
                "type": "route_feedback",
                "target": "selected_route",
                "parameters": {"preference": "reduce_contextual_risk"},
            },
        )
        session = drafted.json()["session"]
        assert len(session["draft"]) == 1

        previewed = test_client.post(
            f"/api/sessions/{session_id}/preview",
            headers=headers,
            json={"expected_revision": session["revision"]},
        )
        assert previewed.status_code == 200
        payload = previewed.json()
        assert payload["proposal"]["state"] == "eligible"
        session = payload["session"]

        accepted = test_client.post(
            f"/api/sessions/{session_id}/accept",
            headers=headers,
            json={"expected_revision": session["revision"]},
        )
        assert accepted.status_code == 409
        assert accepted.json() == {"detail": "acceptance_not_enabled_until_w5"}
        assert not (tmp_path / "state" / "workspace_pointer.json").exists()
        assert not (tmp_path / "state" / "decisions").exists()

        restored, _ = create_session(test_client)
        assert restored["accepted_plan_id"] == parent_plan_id

    assert sha256(parent_path.read_bytes()).hexdigest() == parent_hash_before


def test_infeasible_move_day_preview_cannot_be_accepted(tmp_path: Path) -> None:
    with client(tmp_path) as test_client:
        session, headers = create_session(test_client)
        session_id = session["session_id"]
        drafted = test_client.post(
            f"/api/sessions/{session_id}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 0,
                "type": "move_day",
                "target": "golden_gate_bridge",
                "parameters": {"day": 6},
            },
        ).json()["session"]

        previewed = test_client.post(
            f"/api/sessions/{session_id}/preview",
            headers=headers,
            json={"expected_revision": drafted["revision"]},
        ).json()
        assert previewed["proposal"]["state"] == "ineligible"
        assert previewed["proposal"]["reason"] == "no_feasible_evaluated_child"

        rejected = test_client.post(
            f"/api/sessions/{session_id}/accept",
            headers=headers,
            json={"expected_revision": previewed["session"]["revision"]},
        )
        assert rejected.status_code == 409


def test_origin_guard_evidence_allowlist_and_keep_original(tmp_path: Path) -> None:
    with client(tmp_path) as test_client:
        blocked = test_client.post(
            "/api/sessions",
            headers={"Origin": "https://example.com"},
            json={},
        )
        assert blocked.status_code == 403

        session, headers = create_session(test_client)
        kept = test_client.post(
            f"/api/sessions/{session['session_id']}/keep-original",
            headers=headers,
            json={"expected_revision": 0},
        )
        assert kept.status_code == 409
        assert kept.json() == {"detail": "acceptance_not_enabled_until_w5"}
        assert not (tmp_path / "state" / "decisions").exists()

        bundle = test_client.get("/api/runs/e3ux_weather_repair_demo_v6/evidence-bundle")
        assert bundle.status_code == 200
        archive_path = tmp_path / "evidence.zip"
        archive_path.write_bytes(bundle.content)
        with ZipFile(archive_path) as archive:
            names = archive.namelist()
        assert names
        assert all(".." not in name and not Path(name).is_absolute() for name in names)


def test_workspace_acceptance_methods_are_deferred_until_w5(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "state")
    session, _ = store.create_session("run", "parent", 1)
    session.proposal = {"state": "eligible", "child_plan_id": "child"}

    try:
        store.accept(
            session,
            expected_revision=0,
            parent_plan_id="parent",
            child_plan_id="child",
            certificate_id="certificate",
            diff_id="diff",
        )
    except WorkspaceError as exc:
        assert exc.code == "acceptance_not_enabled_until_w5"
        assert exc.status_code == 409
    else:
        raise AssertionError("acceptance must remain disabled before W5")

    try:
        store.keep_original(session, expected_revision=0)
    except WorkspaceError as exc:
        assert exc.code == "acceptance_not_enabled_until_w5"
        assert exc.status_code == 409
    else:
        raise AssertionError("Keep original must remain disabled before W5")

    assert not (tmp_path / "state" / "workspace_pointer.json").exists()
    assert not (tmp_path / "state" / "decisions").exists()


def test_workspace_does_not_trust_or_rewrite_a_legacy_pointer_before_w5(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    pointer = state_root / "workspace_pointer.json"
    original = b'{"run_id":"run","accepted_plan_id":"legacy-child"}\n'
    pointer.write_bytes(original)

    session, _ = WorkspaceStore(state_root).create_session("run", "validated-parent", 1)

    assert session.accepted_plan_id == "validated-parent"
    assert pointer.read_bytes() == original
