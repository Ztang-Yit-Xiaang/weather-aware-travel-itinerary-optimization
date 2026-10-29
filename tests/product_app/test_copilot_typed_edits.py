"""Typed chat -> actual Node mapper -> authenticated API -> evaluated artifact."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from openai.lib._pydantic import to_strict_json_schema

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.copilot_prompt import build
from itinerary_system.product_app.models import ComponentHealthV1
from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter, OpenAIInterpretationSchemaV2
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.runtime import ProductRuntime
from itinerary_system.product_app.service import ProductService

ROOT = Path(__file__).resolve().parents[2]
CASES = (
    ("Set Stearns Wharf duration to 90 minutes.", "set_stop_duration", "stearns_wharf", {"duration": {"mode": "exact", "preferred_minutes": 90, "minimum_minutes": 90, "maximum_minutes": 90}}),
    ("Move Kings Canyon National Park to position 1 on day 4.", "set_stop_order", "kings_canyon_national_park", {"day": 4, "sequence_index": 0}),
    ("Set Stearns Wharf time window to 10:00-16:00.", "set_stop_time_window", "stearns_wharf", {"earliest_arrival": "10:00", "latest_departure": "16:00"}),
    ("Set Stearns Wharf role to scenic stop.", "set_stop_role", "stearns_wharf", {"role": "scenic_stop"}),
)


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "deterministic")
    monkeypatch.setattr(ProductRuntime, "_probe_map", lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready"))
    app = create_product_app(repository_root=ROOT, registry_path=ROOT / "configs/product_app_registry.json",
        state_root=tmp_path / "state", additional_allowed_authorities=("testserver",), copilot=DeterministicCopilotAdapter())
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={}).json()
        yield client, app.state.product_service, created["session"], {"X-Session-Token": created["mutation_token"]}


def node_operation(intent, workspace):
    url = (ROOT / "src/itinerary_system/product_app/static/js/copilot.js").as_uri()
    source = f"""import {{readFileSync}} from 'node:fs';
import {{copilotOperationForIntent,copilotDraftability}} from {json.dumps(url)};
const value=JSON.parse(readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify({{operation:copilotOperationForIntent(value.intent),draftability:copilotDraftability(value.intent,value.workspace)}}));"""
    result = subprocess.run(["node", "--input-type=module", "-e", source], input=json.dumps({"intent": intent, "workspace": workspace}),
                            text=True, capture_output=True, check=True, timeout=15)
    return json.loads(result.stdout)


@pytest.mark.parametrize(("message", "kind", "target", "parameters"), CASES)
def test_chat_node_mapper_real_api_and_evaluation_are_connected(api, message, kind, target, parameters):
    client, service, session, headers = api
    session_id = session["session_id"]
    request = {"expected_revision": 0, "client_message_id": "client_message_" + uuid4().hex, "message": message}
    response = client.post(f"/api/sessions/{session_id}/copilot/messages", headers=headers, json=request)
    assert response.status_code == 200, response.text
    turn = response.json()["turn"]
    assert turn["interpretation"]["schema_version"] == "copilot-interpretation-v2"
    assert turn["interpretation"]["state"] == "proposal_ready", turn
    intent = turn["interpretation"]["intents"][0]
    assert intent["type"] == kind and intent["target_stop_id"] == target and intent["parameters"] == parameters
    current = service.workspace.authenticate(session_id, headers["X-Session-Token"])
    assert current.draft == [] and current.accepted_plan_id == session["accepted_plan_id"]
    mapped = node_operation(intent, service.workspace_view_for_session(current))
    assert mapped["draftability"]["allowed"] is True, mapped
    assert mapped["operation"] == {"type": kind, "target": target, "parameters": parameters}
    draft_request = {"expected_revision": 0, **mapped["operation"], "source": "copilot", "evidence_refs": []}
    staged = client.post(f"/api/sessions/{session_id}/draft/operations", headers=headers, json=draft_request)
    assert staged.status_code == 200, staged.text
    preview = client.post(f"/api/sessions/{session_id}/preview", headers=headers, json={"expected_revision": 1})
    assert preview.status_code == 200, preview.text
    proposal = preview.json()["proposal"]
    assert proposal.get("child_plan_id") and proposal.get("certificate_id")
    assert proposal["parent_plan_id"] == session["accepted_plan_id"]
    assert proposal["plan_diff"]["child_plan_id"] == proposal["child_plan_id"]
    assert current.accepted_plan_id == session["accepted_plan_id"]
    recovered = service.read_conversation(current).turns[0]
    assert recovered.interpretation.as_dict() == turn["interpretation"] or json.loads(json.dumps(recovered.interpretation.as_dict())) == turn["interpretation"]
    directory = os.environ.get("COPILOT_TYPED_EVIDENCE_DIR")
    if directory:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{kind}.json").write_text(json.dumps({"mode": "deterministic_chat_actual_node_actual_internal_api", "provider_calls": 0,
            "chat_request": request, "chat_response": response.json(), "node_mapping": mapped, "draft_request": draft_request,
            "draft_response": staged.json(), "preview_request": {"expected_revision": 1}, "preview_response": preview.json()}, indent=2), encoding="utf-8")


@pytest.mark.parametrize("message", [
    "Set Stearns Wharf duration to at least 45 minutes.", "Set Stearns Wharf duration to 900 minutes.",
    "Set Stearns Wharf role to lodging.", "Move Kings Canyon National Park to position 1 on day 3.",
    "Set Stearns Wharf time window to 16:00-10:00.", "Set Stearns Wharf duration.",
    "Set Stearns Wharf duration to 90 minutes and remove Golden Gate Bridge.",
    "Set Stearns Wharf duration to 90 minutes and change its role to meal.",
    "Do not spend 90 minutes at Stearns Wharf.",
])
def test_invalid_partial_or_non_evaluated_chat_edit_is_not_proposed(api, message):
    client, _, session, headers = api
    response = client.post(f"/api/sessions/{session['session_id']}/copilot/messages", headers=headers,
        json={"expected_revision": 0, "client_message_id": "client_message_" + uuid4().hex, "message": message})
    assert response.status_code == 200, response.text
    assert response.json()["turn"]["interpretation"]["state"] in {"unsupported", "clarification_required"}


def test_current_provider_schema_contains_closed_typed_parameters_and_uses_one_dispatch(api):
    _, service, session, headers = api
    current = service.workspace.authenticate(session["session_id"], headers["X-Session-Token"])
    context = service.copilot_context(current)
    message, kind, target, parameters = CASES[0]
    calls = []
    async def parse(**kwargs):
        calls.append(kwargs)
        assert kwargs["text_format"] is OpenAIInterpretationSchemaV2
        return SimpleNamespace(output=[], output_parsed=OpenAIInterpretationSchemaV2(schema_version="copilot-interpretation-v2",
            state="proposal_ready", assistant_message="Untrusted provider text.", intents=[{"type": kind, "target_stop_id": target, "parameters": parameters}]))
    adapter = OpenAICopilotAdapter(model="fake-model", api_key="not-a-real-key", client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    output = asyncio.run(adapter.interpret(context=context, history=(), message=message))
    assert len(calls) == 1 and output.schema_version == "copilot-interpretation-v2" and output.intents[0].parameters == parameters
    schema = to_strict_json_schema(OpenAIInterpretationSchemaV2)
    for value in schema["$defs"].values():
        if value.get("type") == "object":
            assert value["additionalProperties"] is False and set(value["required"]) == set(value["properties"])
    directory = os.environ.get("COPILOT_TYPED_EVIDENCE_DIR")
    if directory:
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        service.workspace.select(current, {"expected_revision": 0, "selected_day": 5, "selected_stop_id": "bixby_creek_bridge_viewpoint"})
        live_context = service.copilot_context(current)
        prompt = build(live_context, [], "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park on Day 5, preserving the rest of the seven-day trip.")
        (output_dir / "current_live_context.json").write_text(json.dumps({
            "mode": "real_service_context_fake_provider_only", "provider_calls": 0, "fake_transport_calls": 1,
            "context": live_context.as_dict(), "prompt_input": prompt.input, "prompt_instructions": prompt.instructions,
            "strict_response_schema": schema, "fake_request": {**calls[0], "text_format": OpenAIInterpretationSchemaV2.__name__},
            "fake_output": output.as_dict(),
        }, indent=2), encoding="utf-8")


def test_typed_turn_survives_restart_and_same_message_never_dispatches_again(api, tmp_path):
    client, _, session, headers = api
    request = {"expected_revision": 0, "client_message_id": "client_message_" + uuid4().hex, "message": CASES[0][0]}
    first = client.post(f"/api/sessions/{session['session_id']}/copilot/messages", headers=headers, json=request)
    assert first.status_code == 200
    class NeverDispatch(DeterministicCopilotAdapter):
        async def interpret(self, **kwargs):
            pytest.fail("restored completed turn must not dispatch")
    registry = ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json")
    restored = ProductService(registry, tmp_path / "state", copilot=NeverDispatch())
    result = asyncio.run(restored.interpret_copilot_message(session_id=session["session_id"],
        mutation_token=headers["X-Session-Token"], **request))
    assert json.loads(json.dumps(result["turn"])) == first.json()["turn"]


COMBINED_MESSAGE = "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes."


@pytest.mark.parametrize("message,types", [
    ("The coast looks stormy. Please swap the Bixby stop for Pfeiffer Big Sur State Park and leave the other days alone.", ["replace_nearby"]),
    ("Give Griffith Observatory 60 minutes, and replace Bixby Creek Bridge with Pfeiffer Big Sur State Park.", ["set_stop_duration", "replace_nearby"]),
    ("I'd like 75 minutes at Griffith Observatory instead of 45.", ["set_stop_duration"]),
    ("Treat Stearns Wharf as a scenic stop instead of an activity.", ["set_stop_role"]),
])
def test_recorded_english_variations_preserve_the_requested_edits(api, message, types):
    _, service, session, headers = api
    current = service.workspace.authenticate(session["session_id"], headers["X-Session-Token"])
    result = asyncio.run(service.copilot.interpret(context=service.copilot_context(current), history=(), message=message))
    assert result.state == "proposal_ready", result.assistant_message
    assert [intent.type for intent in result.intents] == types
    if types == ["set_stop_duration"]:
        assert result.intents[0].parameters["duration"]["preferred_minutes"] == 75
    assert current.draft == [] and current.proposal is None


@pytest.mark.parametrize("message", [
    "Set the arrival window for Griffith Observatory to 09:00–18:00.",
    "Arrive at Griffith Observatory between 09:00 and 18:00.",
])
def test_arrival_only_upper_bound_is_clarified_before_provider_dispatch(api, message):
    _, service, session, headers = api
    current = service.workspace.authenticate(session["session_id"], headers["X-Session-Token"])
    class NeverDispatch:
        async def parse(self, **kwargs):
            pytest.fail("An unsupported arrival upper bound must not reach the provider")
    adapter = OpenAICopilotAdapter(model="fake-model", api_key="unused", client=SimpleNamespace(responses=NeverDispatch()))
    result = asyncio.run(adapter.interpret(context=service.copilot_context(current), history=(), message=message))
    assert result.state == "clarification_required"
    assert "departure" in result.assistant_message.lower() and "arrival" in result.assistant_message.lower()
    assert current.draft == []


def test_recorded_two_row_day_four_order_reaches_independent_evaluation(api):
    client, _, session, headers = api
    operations = [{"type":"set_stop_order", "target":target, "parameters":{"day":4, "sequence_index":index},
                   "source":"copilot", "evidence_refs":[]} for index, target in enumerate(
                       ["kings_canyon_national_park", "sequoia_national_park"])]
    url = f"/api/sessions/{session['session_id']}"
    stage = client.post(url + "/draft/operations", headers=headers, json={"expected_revision":0, "operations":operations})
    assert stage.status_code == 200, stage.text
    response = client.post(url + "/preview", headers=headers, json={"expected_revision":1})
    assert response.status_code == 200, response.text
    proposal = response.json()["proposal"]
    assert proposal["state"] == "ineligible"  # Existing Bixby weather disruption is still unresolved.
    certificate = proposal["evaluation_certificate"]
    assert [row["code"] for row in certificate["failures"]] == ["weather_infeasible"]
    assert proposal["child_plan_id"] and proposal["plan_diff"]["reorder_changes"]


def test_combined_chat_maps_and_stages_every_clause_atomically_then_evaluates(api):
    client, service, session, headers = api
    session_id = session["session_id"]
    request = {"expected_revision": 0, "client_message_id": "client_message_" + uuid4().hex, "message": COMBINED_MESSAGE}
    response = client.post(f"/api/sessions/{session_id}/copilot/messages", headers=headers, json=request)
    assert response.status_code == 200, response.text
    interpretation = response.json()["turn"]["interpretation"]
    assert interpretation["state"] == "proposal_ready", interpretation
    assert [row["type"] for row in interpretation["intents"]] == ["replace_nearby", "set_stop_duration"]
    current = service.workspace.authenticate(session_id, headers["X-Session-Token"])
    workspace = service.workspace_view_for_session(current)
    mapped = [node_operation(intent, workspace) for intent in interpretation["intents"]]
    assert all(row["draftability"]["allowed"] for row in mapped)
    operations = [{**row["operation"], "source": "copilot", "evidence_refs": []} for row in mapped]
    stage_input = {"expected_revision": 0, "operations": operations}
    staged = client.post(f"/api/sessions/{session_id}/draft/operations", headers=headers, json=stage_input)
    assert staged.status_code == 200, staged.text
    assert staged.json()["session"]["revision"] == 1
    assert [row["type"] for row in staged.json()["session"]["draft"]] == ["replace_nearby", "set_stop_duration"]
    preview = client.post(f"/api/sessions/{session_id}/preview", headers=headers, json={"expected_revision": 1})
    assert preview.status_code == 200, preview.text
    proposal = preview.json()["proposal"]
    assert proposal["state"] == "eligible", proposal.get("blocking_findings")
    assert proposal["provenance"] == "independent_evaluated_combined_edit"
    assert proposal["geography_plan"]["plan_id"] == proposal["child_plan_id"]
    assert current.accepted_plan_id == session["accepted_plan_id"]
    directory = os.environ.get("COPILOT_TYPED_EVIDENCE_DIR")
    if directory:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / "combined_chat.json").write_text(json.dumps({"provider_calls": 0, "chat_input": request,
            "chat_output": response.json(), "node_mapping": mapped, "stage_input": stage_input,
            "stage_output": staged.json(), "preview_input": {"expected_revision": 1}, "preview_output": preview.json()}, indent=2), encoding="utf-8")


@pytest.mark.parametrize("mutation", ["complete", "drop", "retarget", "reorder"])
def test_combined_provider_dispatch_is_real_and_a_dropped_clause_is_never_proposed(api, mutation):
    _, service, session, headers = api
    current = service.workspace.authenticate(session["session_id"], headers["X-Session-Token"])
    context = service.copilot_context(current)
    calls = []
    intents = [
        {"type":"replace_nearby","target_stop_id":"bixby_creek_bridge_viewpoint","candidate_id":"pfeiffer_big_sur_state_park"},
        {"type":"set_stop_duration","target_stop_id":"griffith_observatory","parameters":{"duration":{
            "mode":"exact","preferred_minutes":90,"minimum_minutes":90,"maximum_minutes":90}}},
    ]
    if mutation == "drop":
        intents.pop()
    elif mutation == "retarget":
        intents[1]["target_stop_id"] = "stearns_wharf"
    elif mutation == "reorder":
        intents.reverse()
    raw_output = OpenAIInterpretationSchemaV2(schema_version="copilot-interpretation-v2",
        state="proposal_ready", assistant_message="Untrusted provider response.", intents=intents)
    async def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output=[], output_parsed=raw_output)
    adapter = OpenAICopilotAdapter(model="fake-model", api_key="not-a-real-key", client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    result = asyncio.run(adapter.interpret(context=context, history=(), message=COMBINED_MESSAGE))
    assert len(calls) == 1, "The combined request must reach the model, not stop at a deterministic shortcut"
    if mutation == "complete":
        assert result.state == "proposal_ready" and len(result.intents) == 2
        assert result.highlights.stop_ids == ("bixby_creek_bridge_viewpoint", "griffith_observatory")
        assert result.highlights.candidate_ids == ("pfeiffer_big_sur_state_park",)
        assert result.highlights.day_ids == (5, 7)
    else:
        assert result.state == "clarification_required"
        assert all(intent.type == "request_clarification" for intent in result.intents)
    directory = os.environ.get("COPILOT_TYPED_EVIDENCE_DIR")
    if directory:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"combined_fake_{mutation}.json").write_text(json.dumps({
            "mode":"injected_sdk_not_a_billed_call", "provider_calls":0, "fake_transport_calls":len(calls),
            "user_input":COMBINED_MESSAGE, "request":{**calls[0],"text_format":OpenAIInterpretationSchemaV2.__name__},
            "strict_response_schema":to_strict_json_schema(OpenAIInterpretationSchemaV2),
            "raw_fake_output":raw_output.model_dump(), "validated_local_output":result.as_dict(),
        }, indent=2), encoding="utf-8")


@pytest.mark.parametrize("message", [
    "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park on Day 3; set Griffith Observatory duration to 90 minutes.",
    "Keep Golden Gate Bridge on Day 3; replace Bixby Creek Bridge with Pfeiffer Big Sur State Park.",
])
def test_combined_chat_does_not_drop_a_contradictory_day_clause(api, message):
    client, service, session, headers = api
    request = {"expected_revision":0,"client_message_id":"client_message_"+uuid4().hex,"message":message}
    response = client.post(f"/api/sessions/{session['session_id']}/copilot/messages", headers=headers,
        json=request)
    assert response.status_code == 200, response.text
    assert response.json()["turn"]["interpretation"]["state"] in {"clarification_required","unsupported"}
    assert response.json()["session"]["draft"] == []
    current = service.workspace.authenticate(session["session_id"], headers["X-Session-Token"])
    async def never_dispatch(**kwargs):
        pytest.fail("A contradictory day must be blocked before provider dispatch")
    adapter = OpenAICopilotAdapter(model="fake-model", api_key="not-a-real-key",
        client=SimpleNamespace(responses=SimpleNamespace(parse=never_dispatch)))
    guarded = asyncio.run(adapter.interpret(context=service.copilot_context(current),history=(),message=message))
    assert guarded.state in {"clarification_required","unsupported"}
    directory = os.environ.get("COPILOT_TYPED_EVIDENCE_DIR")
    if directory:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        label = "replace" if message.startswith("Replace") else "keep"
        (output / f"contradictory_day_{label}.json").write_text(json.dumps({
            "provider_calls":0, "chat_input":request, "chat_output":response.json(),
            "pre_dispatch_guard_output":guarded.as_dict(),
        }, indent=2), encoding="utf-8")


def test_atomic_batch_rejects_invalid_last_operation_without_staging_the_first(api):
    client, service, session, headers = api
    payload = {"expected_revision": 0, "operations": [
        {"type": "replace_nearby", "target": "bixby_creek_bridge_viewpoint", "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"}},
        {"type": "set_stop_duration", "target": "griffith_observatory", "parameters": {"duration": {"mode": "exact", "preferred_minutes": 900, "minimum_minutes": 900, "maximum_minutes": 900}}},
    ]}
    response = client.post(f"/api/sessions/{session['session_id']}/draft/operations", headers=headers, json=payload)
    assert response.status_code == 422, response.text
    current = service.workspace.authenticate(session["session_id"], headers["X-Session-Token"])
    assert current.revision == 0 and current.draft == [] and current.proposal is None


@pytest.mark.parametrize("message", [
    "Set Stearns Wharf duration to 90 minutes; set Stearns Wharf duration to 120 minutes.",
    "Set Stearns Wharf duration to 90 minutes; set Stearns Wharf duration to 90 minutes.",
    "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Bixby Creek Bridge duration to 90 minutes.",
])
def test_conflicting_combined_chat_is_clarified_before_provider_dispatch(api, message):
    client, service, session, headers = api
    request = {"expected_revision": 0, "client_message_id": "client_message_" + uuid4().hex, "message": message}
    response = client.post(f"/api/sessions/{session['session_id']}/copilot/messages", headers=headers, json=request)
    assert response.status_code == 200, response.text
    output = response.json()
    assert output["turn"]["interpretation"]["state"] == "clarification_required", output
    assert [row["type"] for row in output["turn"]["interpretation"]["intents"]] == ["request_clarification"]
    assert output["session"]["draft"] == [] and output["session"]["revision"] == 0
    current = service.workspace.authenticate(session["session_id"], headers["X-Session-Token"])
    calls = []

    async def never_dispatch(**kwargs):
        calls.append(kwargs)
        pytest.fail("Conflicting edits must be clarified before a billed attempt")

    adapter = OpenAICopilotAdapter(model="fake-model", api_key="not-a-real-key",
        client=SimpleNamespace(responses=SimpleNamespace(parse=never_dispatch)))
    guarded = asyncio.run(adapter.interpret(context=service.copilot_context(current), history=(), message=message))
    assert guarded.state == "clarification_required" and calls == []
    directory = os.environ.get("COPILOT_TYPED_EVIDENCE_DIR")
    if directory:
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        label = "duplicate" if message.count("90 minutes") == 2 else "replacement" if message.startswith("Replace") else "values"
        (output_dir / f"combined_conflict_{label}.json").write_text(json.dumps({
            "provider_calls": 0, "chat_input": request, "chat_output": output,
            "pre_dispatch_guard_output": guarded.as_dict(), "fake_transport_calls": len(calls),
        }, indent=2), encoding="utf-8")


def test_atomic_batch_cannot_replace_and_edit_the_same_stop(api):
    client, service, session, headers = api
    request = {"expected_revision": 0, "operations": [
        {"type": "replace_nearby", "target": "bixby_creek_bridge_viewpoint", "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"}},
        {"type": "set_stop_duration", "target": "bixby_creek_bridge_viewpoint", "parameters": {"duration": {
            "mode": "exact", "preferred_minutes": 90, "minimum_minutes": 90, "maximum_minutes": 90}}},
    ]}
    response = client.post(f"/api/sessions/{session['session_id']}/draft/operations", headers=headers, json=request)
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "draft_conflicts_with_replacement"
    current = service.workspace.authenticate(session["session_id"], headers["X-Session-Token"])
    assert current.revision == 0 and current.draft == [] and current.proposal is None
    directory = os.environ.get("COPILOT_TYPED_EVIDENCE_DIR")
    if directory:
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "atomic_replaced_target_batch.json").write_text(json.dumps({
            "provider_calls":0, "input":request, "status_code":response.status_code,
            "output":response.json(), "revision_after":current.revision,
            "draft_after":current.public_dict()["draft"], "proposal_after":current.proposal,
        }, indent=2), encoding="utf-8")


@pytest.mark.parametrize("quantity", [
    "up to 90 minutes", "no more than 90 minutes", "no less than 90 minutes",
    "more than 90 minutes", "less than 90 minutes", "over 90 minutes", "under 90 minutes",
    "about 90 minutes", "approximately 90 minutes", "around 90 minutes", "roughly 90 minutes",
    "nearly 90 minutes", "not 90 minutes", "anything except 90 minutes", "90 minutes tops",
    "~90 minutes", "≈90 minutes", "<90 minutes", ">90 minutes", "<=90 minutes", ">=90 minutes",
    "≤90 minutes", "≥90 minutes", "90 minutes+",
])
def test_non_exact_duration_never_becomes_an_exact_chat_edit(api, quantity):
    client, service, session, headers = api
    message = f"Give Griffith Observatory {quantity}."
    response = client.post(f"/api/sessions/{session['session_id']}/copilot/messages", headers=headers,
        json={"expected_revision": 0, "client_message_id": "client_message_" + uuid4().hex, "message": message})
    assert response.status_code == 200, response.text
    output = response.json()["turn"]["interpretation"]
    assert output["state"] in {"unsupported", "clarification_required"}, output
    assert all(row["type"] in {"unsupported_request", "request_clarification"} for row in output["intents"])
    current = service.workspace.authenticate(session["session_id"], headers["X-Session-Token"])
    assert current.revision == 0 and current.draft == [] and current.proposal is None
    assert current.accepted_plan_id == session["accepted_plan_id"]

    async def never_dispatch(**kwargs):
        pytest.fail("A non-exact duration must not be converted to an exact provider request")
    adapter = OpenAICopilotAdapter(model="fake-model", api_key="unused",
        client=SimpleNamespace(responses=SimpleNamespace(parse=never_dispatch)))
    guarded = asyncio.run(adapter.interpret(context=service.copilot_context(current), history=(), message=message))
    assert guarded.state in {"unsupported", "clarification_required"}
    assert all(intent.type in {"unsupported_request", "request_clarification"} for intent in guarded.intents)


@pytest.mark.parametrize("quantity", ["up to 90 minutes", "about 90 minutes", "≤90 minutes"])
def test_combined_request_does_not_drop_a_non_exact_duration_clause(api, quantity):
    client, service, session, headers = api
    message = "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; " + f"give Griffith Observatory {quantity}."
    response = client.post(f"/api/sessions/{session['session_id']}/copilot/messages", headers=headers,
        json={"expected_revision": 0, "client_message_id": "client_message_" + uuid4().hex, "message": message})
    assert response.status_code == 200, response.text
    output = response.json()
    assert output["turn"]["interpretation"]["state"] in {"unsupported", "clarification_required"}
    assert output["session"]["draft"] == [] and output["session"]["revision"] == 0
    current = service.workspace.authenticate(session["session_id"], headers["X-Session-Token"])
    async def never_dispatch(**kwargs):
        pytest.fail("No subset of the combined request may be dispatched")
    adapter = OpenAICopilotAdapter(model="fake-model", api_key="unused",
        client=SimpleNamespace(responses=SimpleNamespace(parse=never_dispatch)))
    guarded = asyncio.run(adapter.interpret(context=service.copilot_context(current), history=(), message=message))
    assert guarded.state in {"unsupported", "clarification_required"}


@pytest.mark.parametrize("message", [
    "Give Griffith Observatory exactly 90 minutes instead of 45 minutes.",
    "Give Griffith Observatory only 90 minutes.",
    "Give Griffith Observatory 90 minutes on day 7.",
])
def test_exact_duration_wording_retains_exact_parameters(api, message):
    client, service, session, headers = api
    current = service.workspace.authenticate(session["session_id"], headers["X-Session-Token"])
    expected = {"type": "set_stop_duration", "target_stop_id": "griffith_observatory", "parameters": {
        "duration": {"mode": "exact", "preferred_minutes": 90, "minimum_minutes": 90, "maximum_minutes": 90}}}
    calls = []
    async def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output=[], output_parsed=OpenAIInterpretationSchemaV2(
            schema_version="copilot-interpretation-v2", state="proposal_ready", assistant_message="Injected exact edit",
            intents=[expected]))
    adapters = [DeterministicCopilotAdapter(), OpenAICopilotAdapter(model="fake-model", api_key="unused",
        client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))]
    for adapter in adapters:
        result = asyncio.run(adapter.interpret(context=service.copilot_context(current), history=(), message=message))
        assert result.state == "proposal_ready"
        assert [{key:value for key,value in intent.as_dict().items() if value is not None} for intent in result.intents] == [expected]
    assert len(calls) == 1
