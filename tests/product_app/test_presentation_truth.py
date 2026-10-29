"""Accepted presentation facts must come from the persisted evaluated child."""

import json
import subprocess
from pathlib import Path

from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "california_coast_coherent_demo_v1"


def test_frontend_selection_writes_are_serialized():
    subprocess.run(["node", str(Path(__file__).with_name("selection_queue_contract.test.mjs"))], check=True, cwd=ROOT)


def test_build_recovery_closes_dialogs_and_preserves_state():
    subprocess.run(["node", str(Path(__file__).with_name("build_recovery_frontend.test.mjs"))], check=True, cwd=ROOT)


def test_accepted_reload_exposes_exact_persisted_evaluation_and_diff(tmp_path):
    registry = ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json")
    service = ProductService(registry, tmp_path / "state")
    session, token, _ = service.create_session(RUN_ID)
    service.workspace.add_operation(session, {
        "expected_revision": 0, "type": "replace_nearby", "target": "bixby_creek_bridge_viewpoint",
        "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"}, "source": "map",
    }, valid_stop_ids=service.valid_stops(RUN_ID), day_count=7)
    proposal = service.preview(session.session_id, 1)
    service.accept_evaluated(session, expected_revision=2, confirmation="accept_evaluated_repair",
                             idempotency_key="presentation-truth-test", mutation_token=token)
    restored = ProductService(registry, tmp_path / "state")
    accepted_session = restored.workspace.get(session.session_id)
    view = restored.workspace_view_for_session(accepted_session)
    record = view["alternative_evidence"].get(proposal["child_plan_id"])
    assert record is not None, "accepted child evidence must not disappear on reload"
    assert record["plan_id"] == proposal["child_plan_id"]
    assert record["plan_content_hash"] == proposal["child_plan_content_hash"]
    assert record["certificate"]["certificate_id"] == proposal["certificate_id"]
    assert record["certificate"]["plan_id"] == record["plan_id"]
    assert record["certificate"]["plan_content_hash"] == record["plan_content_hash"]
    assert record["diff"]["diff_id"] == proposal["diff_id"]
    actual = tmp_path / "accepted_evidence_context.json"
    actual.write_text(json.dumps({"workspace":view,"session":accepted_session.public_dict()}), encoding="utf-8")
    subprocess.run(["node", str(Path(__file__).with_name("accepted_evidence_context.test.mjs")), str(actual)], check=True, cwd=ROOT)
