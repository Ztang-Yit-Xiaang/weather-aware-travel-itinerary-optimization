"""R11: accepted children are immutable, verified next repair baselines."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.acceptance import AcceptanceError, _content_hash
from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler
from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService, _replay_direct_certificate_with_persisted_timestamp
from itinerary_system.product_app.workspace import WorkspaceError
from itinerary_system.research_artifacts import stable_content_hash

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "california_coast_coherent_demo_v1"
PFEIFFER = "pfeiffer_big_sur_state_park"
BIXBY = "bixby_creek_bridge_viewpoint"


def duration(minutes=60):
    return {"type": "set_stop_duration", "target": PFEIFFER, "parameters": {"duration": {
        "mode": "exact", "preferred_minutes": minutes, "minimum_minutes": minutes, "maximum_minutes": minutes,
    }}, "source": "typed_stop_editor"}


@pytest.fixture(scope="module")
def compiler_and_child():
    package = load_product_demo_package(ROOT, ROOT / "runs/california-coast-coherent-demo-v1")
    compiler = FrozenDraftCompiler(package.primary_bundle.parent_plan, package.evidence_bundles, repository_root=ROOT)
    child = compiler.compile([{"type": "replace_nearby", "target": BIXBY, "parameters": {"candidate_id": PFEIFFER}}], accepted_plan_id=compiler.parent_plan_id)
    assert child.state == "eligible"
    return compiler, child.child_plan


def test_factory_preserves_source_evidence_and_recomputes_current_applicability(compiler_and_child):
    compiler, child = compiler_and_child
    original = deepcopy(compiler.parent_plan)
    rebased = compiler.for_parent(child)
    assert compiler.parent_plan == original
    assert rebased.parent_plan == child
    assert rebased._base_bundle.parent_plan == child
    assert rebased._base_bundle.route_matrix == compiler._base_bundle.route_matrix
    caps = rebased.capabilities()
    assert PFEIFFER in caps["parent_targets"] and BIXBY not in caps["parent_targets"]
    assert {row["candidate_id"] for row in caps["candidate_choices"]} == {"point_lobos_state_natural_reserve"}
    assert caps["operations"]["replace_nearby"]["preview_executable"] is False
    assert caps["operations"]["add_candidate"]["preview_executable"] is False
    result = rebased.compile([duration()], accepted_plan_id=child["plan_id"])
    assert result.state == "eligible"
    assert result.child_plan["parent_plan_id"] == child["plan_id"]
    assert result.certificate["plan_id"] == result.child_plan["plan_id"]


def test_factory_cannot_weaken_candidate_source_lineage(compiler_and_child):
    compiler, child = compiler_and_child
    rebased = compiler.for_parent(child)
    with pytest.raises(WorkspaceError, match="draft_candidate_target_mismatch"):
        rebased.compile([{"type": "replace_nearby", "target": PFEIFFER,
                          "parameters": {"candidate_id": "point_lobos_state_natural_reserve"}}], accepted_plan_id=child["plan_id"])
    with pytest.raises(WorkspaceError, match="draft_candidate_already_in_plan"):
        rebased.compile([{"type": "add_candidate", "target": PFEIFFER, "parameters": {"day": 5}}], accepted_plan_id=child["plan_id"])


def test_factory_rejects_tampered_parent_hash(compiler_and_child):
    compiler, child = compiler_and_child
    with pytest.raises(WorkspaceError, match="draft_parent_hash_mismatch"):
        compiler.for_parent({**child, "content_hash": "wrong_hash"})


def test_real_replace_accept_restart_duration_accept_restart(tmp_path):
    state_root = tmp_path / "s"
    evidence = {"provider_mode": "deterministic_no_provider", "exchanges": []}

    def client():
        return TestClient(create_product_app(repository_root=ROOT, registry_path=ROOT / "configs/product_app_registry.json",
            state_root=state_root, additional_allowed_authorities=("testserver",), copilot=DeterministicCopilotAdapter()))

    def post(api, path, body, headers=None):
        response = api.post(path, json=body, headers=headers)
        output = response.json()
        evidence["exchanges"].append({"path": path, "input": body, "status": response.status_code,
                                      "output": {key: value for key, value in output.items() if key != "mutation_token"}})
        (tmp_path / "exact_chain_io.json").write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
        assert response.status_code == 200, response.text
        return output

    with client() as api:
        created = post(api, "/api/sessions", {"run_id": RUN_ID})
        session = created["session"]
        a = session["accepted_plan_id"]
        url = f"/api/sessions/{session['session_id']}"
        headers = {"X-Session-Token": created["mutation_token"]}
        staged = post(api, url + "/draft/operations", {"expected_revision": session["revision"],
            "type": "replace_nearby", "target": BIXBY, "parameters": {"candidate_id": PFEIFFER}, "source": "typed_stop_editor"}, headers)
        preview_b = post(api, url + "/preview", {"expected_revision": staged["session"]["revision"]}, headers)
        b = preview_b["proposal"]["child_plan_id"]
        assert preview_b["proposal"]["state"] == "eligible"
        accepted_b = post(api, url + "/accept", {"expected_revision": preview_b["session"]["revision"],
            "confirmation": "accept_evaluated_repair", "idempotency_key": "r11_accept_baseline_b"}, headers)
        assert accepted_b["session"]["accepted_plan_id"] == b
        snapshot_b = deepcopy(api.app.state.product_service.acceptance.load_snapshot(accepted_b["result"]["decision"]["decision_id"]))

    with client() as api:
        restored = api.get(url, headers=headers)
        assert restored.status_code == 200, restored.text
        restored = restored.json()
        session = restored["session"]
        assert session["accepted_plan_id"] == b
        capability = restored["workspace"]["typed_edit_capabilities"]["operations"]["set_stop_duration"]
        assert capability["enabled"] is capability["preview_executable"] is True
        assert PFEIFFER in restored["workspace"]["draft_capabilities"]["parent_targets"]
        assert BIXBY not in restored["workspace"]["draft_capabilities"]["parent_targets"]
        service = api.app.state.product_service
        compiler_b = service.compiler_for_session(service.workspace.get(session["session_id"]))
        staged = post(api, url + "/draft/operations", {"expected_revision": session["revision"], **duration()}, headers)
        impact = post(api, url + "/draft/impact-preview", {"expected_revision": staged["session"]["revision"]}, headers)
        assert impact["parent_plan_id"] == b
        assert impact["summary"]["can_run_evaluated_preview"] is True
        preview_c = post(api, url + "/preview", {"expected_revision": staged["session"]["revision"]}, headers)
        proposal = preview_c["proposal"]
        assert proposal["state"] == "eligible"
        assert proposal["parent_plan_id"] == b
        c = proposal["child_plan_id"]
        assert c not in {a, b}
        assert proposal["geography_plan"]["plan_id"] == c
        assert proposal["geography_plan"]["content_hash"] == proposal["child_plan_content_hash"]
        assert proposal["geography_plan"]["coverage"]["complete"] is True
        accepted_c = post(api, url + "/accept", {"expected_revision": preview_c["session"]["revision"],
            "confirmation": "accept_evaluated_repair", "idempotency_key": "r11_accept_baseline_c"}, headers)
        assert accepted_c["session"]["accepted_plan_id"] == c
        compiler_c = service.compiler_for_session(service.workspace.get(session["session_id"]))
        assert compiler_c is not compiler_b
        assert compiler_b.parent_plan_id == b
        assert set(service._accepted_compilers) == {(RUN_ID, c, proposal["child_plan_content_hash"])}
        assert service.acceptance.load_snapshot(snapshot_b["decision_id"]) == snapshot_b
        snapshot_c = service.acceptance.load_snapshot(accepted_c["result"]["decision"]["decision_id"])
        assert snapshot_c["certificate"] == proposal["certificate_schedule_evidence"]["certificate_record"]
        assert snapshot_c["plan"]["parent_plan_id"] == b
        assert snapshot_c["plan"]["owned_constraints"] == snapshot_b["plan"]["owned_constraints"]

    with client() as api:
        restored = api.get(url, headers=headers)
        assert restored.status_code == 200, restored.text
        restored = restored.json()
        assert restored["session"]["accepted_plan_id"] == c
        accepted_map = next(row for row in restored["workspace"]["geography"]["plans"] if row["plan_id"] == c)
        assert accepted_map["content_hash"] == proposal["child_plan_content_hash"]
        assert {row["properties"]["stop_id"] for row in accepted_map["stops"]["features"]} == {
            row["properties"]["stop_id"] for row in preview_b["proposal"]["geography_plan"]["stops"]["features"]}
        created = post(api, "/api/sessions", {"run_id": RUN_ID})
        assert created["session"]["accepted_plan_id"] == c
        history = api.app.state.product_service.acceptance.history(session["trip_id"])
        assert [row["selected_plan_id"] for row in history] == [b, c]
        evidence["final_identity"] = {
            "original": a, "accepted_b": b, "accepted_c": c,
            "b_hash": snapshot_b["plan"]["content_hash"], "c_hash": proposal["child_plan_content_hash"],
            "c_diff_id": proposal["diff_id"], "c_diff_hash": proposal["diff_content_hash"],
            "c_evaluation_id": proposal["certificate_id"], "c_evaluation_hash": proposal["certificate_content_hash"],
            "c_route_validation": snapshot_c["certificate"]["route_validation"],
            "c_geometry_hash": stable_content_hash(snapshot_c["geography_plan"]),
        }
        (tmp_path / "exact_chain_io.json").write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")


@pytest.fixture(scope="module")
def accepted_state(tmp_path_factory):
    state_root = tmp_path_factory.mktemp("b") / "s"
    app = create_product_app(repository_root=ROOT, registry_path=ROOT / "configs/product_app_registry.json",
        state_root=state_root, additional_allowed_authorities=("testserver",), copilot=DeterministicCopilotAdapter())
    with TestClient(app) as api:
        created = api.post("/api/sessions", json={"run_id": RUN_ID}).json()
        session = created["session"]
        original_id = session["accepted_plan_id"]
        url = f"/api/sessions/{session['session_id']}"
        headers = {"X-Session-Token": created["mutation_token"]}
        staged = api.post(url + "/draft/operations", headers=headers, json={"expected_revision": 0,
            "type": "replace_nearby", "target": BIXBY, "parameters": {"candidate_id": PFEIFFER}})
        assert staged.status_code == 200, staged.text
        preview = api.post(url + "/preview", headers=headers, json={"expected_revision": 1})
        assert preview.status_code == 200, preview.text
        accepted = api.post(url + "/accept", headers=headers, json={"expected_revision": 2,
            "confirmation": "accept_evaluated_repair", "idempotency_key": "r11_adversarial_baseline"})
        assert accepted.status_code == 200, accepted.text
    return state_root, session["session_id"], original_id


def restored_service(tmp_path, accepted_state):
    state_root, session_id, original_id = accepted_state
    copied = tmp_path / "s"
    shutil.copytree(state_root, copied)
    service = ProductService(ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"), copied,
        copilot=DeterministicCopilotAdapter())
    return service, service.workspace.get(session_id), original_id


def test_poi_discovery_uses_the_accepted_snapshot_geography(tmp_path, accepted_state):
    service, session, _ = restored_service(tmp_path, accepted_state)
    view = service.workspace_view_for_session(session)
    accepted_geography = next(
        plan for plan in view["geography"]["plans"]
        if plan["plan_id"] == session.accepted_plan_id
    )
    incoming_leg = next(
        feature["properties"]["route_leg_id"]
        for feature in accepted_geography["validated_legs"]["features"]
        if feature["properties"]["route_leg_id"].endswith(PFEIFFER)
    )

    result = service.poi_candidates(
        session,
        expected_revision=session.revision,
        day=5,
        route_leg_id=incoming_leg,
        replacement_target_id=PFEIFFER,
        role=None,
        maximum_detour_minutes=60,
        limit=10,
    )

    assert result["context"]["route_leg_id"] == incoming_leg
    assert result["context"]["replacement_target_id"] == PFEIFFER
    assert all(row["candidate_id"] != PFEIFFER for row in result["candidates"])


def test_waypoint_route_context_uses_the_accepted_snapshot_geography(tmp_path, accepted_state):
    service, session, _ = restored_service(tmp_path, accepted_state)
    view = service.workspace_view_for_session(session)
    accepted_geography = next(
        plan for plan in view["geography"]["plans"]
        if plan["plan_id"] == session.accepted_plan_id
    )
    incoming = next(
        feature for feature in accepted_geography["validated_legs"]["features"]
        if feature["properties"]["route_leg_id"].endswith(PFEIFFER)
    )

    predecessor, successor = service.resolve_validated_route_context(
        session=session,
        route_leg_id=incoming["properties"]["route_leg_id"],
        selected_day=5,
    )

    assert predecessor[0] == incoming["properties"]["origin_id"]
    assert successor[0] == PFEIFFER
    start, end = incoming["geometry"]["coordinates"][0], incoming["geometry"]["coordinates"][-1]
    assert predecessor[1] == (start[1], start[0])
    assert successor[1] == (end[1], end[0])


def test_stale_session_cannot_reuse_accepted_compiler(tmp_path, accepted_state):
    service, session, original_id = restored_service(tmp_path, accepted_state)
    compiler = service.compiler_for_session(session)
    session.accepted_plan_id = original_id
    with pytest.raises(WorkspaceError, match="acceptance_pointer_conflict"):
        service.compiler_for_session(session)
    assert compiler.parent_plan_id != original_id


@pytest.mark.parametrize("rehash", [False, True])
def test_warm_cache_rechecks_snapshot_and_ledger_binding(tmp_path, accepted_state, rehash):
    service, session, _ = restored_service(tmp_path, accepted_state)
    assert service.compiler_for_session(session) is service.compiler_for_session(session)
    pointer = service.acceptance.load_pointer(session.trip_id)
    path = service.acceptance.snapshots_dir / f"{pointer['decision_id']}.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["proposal_identity"]["diff_content_hash"] = "tampered"
    if rehash:
        snapshot["content_hash"] = _content_hash({key: value for key, value in snapshot.items() if key != "content_hash"})
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    expected = "accepted_baseline_evidence_mismatch" if rehash else "acceptance_snapshot_invalid"
    with pytest.raises((WorkspaceError, AcceptanceError), match=expected):
        service.compiler_for_session(session)


@pytest.mark.parametrize("enveloped", [False, True])
def test_direct_certificate_replay_preserves_exact_record_and_rejects_semantic_drift(compiler_and_child, enveloped):
    compiler, child = compiler_and_child
    result = compiler.for_parent(child).compile([duration()], accepted_plan_id=child["plan_id"])
    stored = json.loads(json.dumps(result.certificate))
    proposal = {"certificate_content_hash": stored["content_hash"],
        "certificate_identity": {"content_hash": stored["content_hash"]}}
    if enveloped:
        envelope = {"certificate_record": stored, "certificate_content_hash": stored["content_hash"]}
        proposal["certificate_schedule_evidence"] = {**envelope, "content_hash": stable_content_hash(envelope)}
        proposal["schedule_impact_identity"] = {"certificate_content_hash": stored["content_hash"]}
    else:
        proposal["evaluation_certificate"] = stored
    fresh = deepcopy(stored)
    fresh["evaluated_at"] = "2026-09-08T10:00:00+00:00"
    fresh["content_hash"] = stable_content_hash({key: value for key, value in fresh.items() if key != "content_hash"})
    assert _replay_direct_certificate_with_persisted_timestamp(fresh, proposal) == stored
    for field, changed in (("comparison_eligibility", "ineligible"), ("plan_content_hash", "wrong"),
                           ("evaluator_version", "other"), ("metrics", {}), ("route_validation", {})):
        with pytest.raises(WorkspaceError, match="acceptance_proposal_invalid"):
            _replay_direct_certificate_with_persisted_timestamp({**fresh, field: changed}, proposal)
    tampered = deepcopy(proposal)
    tampered["certificate_content_hash"] = "wrong"
    with pytest.raises(WorkspaceError, match="acceptance_proposal_invalid"):
        _replay_direct_certificate_with_persisted_timestamp(fresh, tampered)


@pytest.mark.parametrize("operation", [
    {"type": "set_stop_order", "target": "surf_n_wear_s_beach_house", "parameters": {"day": 4, "sequence_index": 0}},
    {"type": "set_stop_time_window", "target": "stearns_wharf", "parameters": {"earliest_arrival": "10:00", "latest_departure": None}},
])
def test_other_direct_edit_families_accept_exact_preview_certificate_after_restart(tmp_path, operation):
    state_root = tmp_path / "s"

    def client():
        return TestClient(create_product_app(repository_root=ROOT, registry_path=ROOT / "configs/product_app_registry.json",
            state_root=state_root, additional_allowed_authorities=("testserver",), copilot=DeterministicCopilotAdapter()))

    with client() as api:
        created = api.post("/api/sessions", json={"run_id": "california_coast_product_demo_v2"}).json()
        url = f"/api/sessions/{created['session']['session_id']}"
        headers = {"X-Session-Token": created["mutation_token"]}
        staged = api.post(url + "/draft/operations", headers=headers,
            json={"expected_revision": 0, "source": "typed_stop_editor", **operation})
        assert staged.status_code == 200, staged.text
        response = api.post(url + "/preview", headers=headers, json={"expected_revision": 1})
        assert response.status_code == 200, response.text
        proposal = response.json()["proposal"]
        assert proposal["state"] == "eligible"
        stored = proposal.get("evaluation_certificate") or proposal["certificate_schedule_evidence"]["certificate_record"]
        assert proposal["plan_diff"]["child_plan_id"] == proposal["child_plan_id"]

    with client() as api:
        response = api.post(url + "/accept", headers=headers, json={"expected_revision": 2,
            "confirmation": "accept_evaluated_repair", "idempotency_key": "r11_direct_edit_acceptance"})
        assert response.status_code == 200, response.text
        accepted = response.json()
        snapshot = api.app.state.product_service.acceptance.load_snapshot(accepted["result"]["decision"]["decision_id"])
        assert snapshot["certificate"] == stored

    with client() as api:
        response = api.get(url, headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["session"]["accepted_plan_id"] == proposal["child_plan_id"]
        (tmp_path / "direct_edit_identity.json").write_text(json.dumps({"input": operation,
            "proposal_identity": {key: proposal[key] for key in ("parent_plan_id", "child_plan_id", "child_plan_content_hash", "diff_id")},
            "certificate": stored, "persisted_certificate": snapshot["certificate"]}, indent=2), encoding="utf-8")
