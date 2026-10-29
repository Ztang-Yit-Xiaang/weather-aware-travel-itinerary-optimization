"""Bounded-request and explicit evidence-refresh regressions from the human audit."""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService
from itinerary_system.product_app.trip_planning import (
    IndividualizedTripParameters,
    TripPlanParameters,
    resolve_trip_profile,
    validate_trip_parameters,
)
from itinerary_system.product_app.workspace import WorkspaceError
from itinerary_system.research_artifacts import stable_content_hash

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    service = ProductService(ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"),
        tmp_path_factory.mktemp("planner_audit"), copilot=DeterministicCopilotAdapter())
    session, _, _ = service.create_session(service.registry.default.run_id)
    return service.copilot_context(session)


def interpretation(context, message, provider="deterministic"):
    forbidden = Mock(side_effect=AssertionError("Bounded language cases must not invoke a provider"))
    adapter = (DeterministicCopilotAdapter() if provider == "deterministic" else
        OpenAICopilotAdapter(model="never-network", api_key="synthetic-unusable-test-key",
            client=SimpleNamespace(responses=SimpleNamespace(parse=forbidden))))
    result = asyncio.run(adapter.interpret(context=context, history=(), message=message))
    forbidden.assert_not_called()
    return result


@pytest.mark.parametrize("message,minutes", [
    ("Plan a 10 day trip, drive at most 1.5 hours per day", 90),
    ("Plan a 10 day trip, drive at most 2.5 hours per day", 150),
    ("10天旅行，每天開車不超過1.5小時", 90),
    ("Plan a 10 day relaxed trip, drive no more than 1.5 hours per day", 90),
])
def test_decimal_drive_cap(context, message, minutes):
    result = interpretation(context, message)
    assert result.state == "proposal_ready"
    assert result.intents[0].parameters["max_daily_drive_minutes"] == minutes


@pytest.mark.parametrize("provider", ["deterministic", "openai"])
@pytest.mark.parametrize("message,total_days", [
    ("Keep daily driving under 1.5 hours.", None),
    ("Plan 14 days with a daily driving limit of 1.5 hours.", 14),
])
def test_daily_driving_paraphrases_preserve_accepted_profile(context, provider, message, total_days):
    before = context.as_dict()
    result = interpretation(context, message, provider)
    assert result.state == "proposal_ready"
    assert len(result.intents) == 1 and result.intents[0].type == "replan_trip"
    requested = validate_trip_parameters(result.intents[0].parameters)
    assert isinstance(requested, IndividualizedTripParameters)
    assert requested.total_days == (total_days or max(context.allowed_days))
    assert requested.max_daily_drive_minutes == 90
    assert requested.profile_updates == ["max_daily_drive_minutes"]
    baseline = {"effective_parameters": IndividualizedTripParameters(
        total_days=max(context.allowed_days), pace="relaxed", interests=["culture"],
        coastal=True, max_daily_drive_minutes=180,
        excluded_place_ids=["sequoia_national_park"]).model_dump()}
    effective = resolve_trip_profile(requested, baseline)
    assert (effective.total_days, effective.max_daily_drive_minutes) == (requested.total_days, 90)
    assert (effective.pace, effective.interests, effective.coastal) == ("relaxed", ["culture"], True)
    assert effective.excluded_place_ids == ["sequoia_national_park"]
    assert context.as_dict() == before


@pytest.mark.parametrize("provider", ["deterministic", "openai"])
def test_yosemite_total_plus_three_clarification_survives_outer_composer(context, provider):
    before = context.as_dict()
    result = interpretation(context, "Plan a 14 day trip, add three days at Yosemite", provider)
    assert result.state == "clarification_required"
    assert len(result.intents) == 1 and result.intents[0].type == "request_clarification"
    assert result.intents[0].clarification_question == result.assistant_message
    text = result.assistant_message.lower()
    assert all(term in text for term in ("14", "total", "three", "17", "7", "10", "yosemite"))
    assert "separate the trip requirements" not in text
    assert "nothing was staged or changed" in text
    assert context.as_dict() == before


@pytest.mark.parametrize("provider", ["deterministic", "openai"])
def test_hearst_factual_question_has_targeted_readonly_reference_reply(context, provider):
    from itinerary_system.product_app.official_travel import OFFICIAL_URLS

    before = context.as_dict()
    result = interpretation(context, "What is Hearst Castle?", provider)
    assert result.state != "proposal_ready"
    assert all(intent.type in {"request_clarification", "unsupported_request"} for intent in result.intents)
    assert "Hearst Castle" in result.assistant_message
    assert OFFICIAL_URLS["hearst_castle"] in result.assistant_message
    text = result.assistant_message.lower()
    assert "reference description" in text
    assert "have not been checked" in text
    assert "nothing was staged or changed" in text
    assert "separate the trip requirements" not in text
    assert context.as_dict() == before


@pytest.mark.parametrize("message", [
    "Plan a 14 day trip, I do not like museums",
    "Plan a 14 day trip, I hate museums",
    "Plan a 14 day trip, no museums",
    "Plan a 14 day trip, I am not a fan of museums",
    "Plan a 10 day trip, do not visit Yosemite",
    "Plan a 10 day coastal trip and include Yosemite",
    "Plan a 14 day trip and start from San Diego",
    "Plan a 10 day trip starting from Los Angeles and ending in San Francisco",
    "Plan a 10 day trip, starting in San Francisco or Los Angeles",
    "Plan a 14 day trip, add three days at Yosemite",
    "Plan a 14 day trip in November",
    "Plan a 10 day trip, keep my booked hotel in Los Angeles",
    "Tell me about Redwood National Park",
    "Where do I catch the ferry to Channel Islands?",
    "How long is the drive to Hearst Castle?",
    "Is Redwood National Park open?",
    "What can I see in Channel Islands?",
])
def test_unrepresented_or_informational_clause_is_not_a_partial_edit(context, message):
    result = interpretation(context, message)
    assert result.state in {"clarification_required", "unsupported"}
    assert not any(intent.type == "replan_trip" for intent in result.intents)


def test_required_places_and_instead_of_exclusion(context):
    result = interpretation(context, "Plan 2 weeks, starting in Los Angeles. I prefer coastal stops "
        "instead of Kings Canyon. Include Hearst Castle.")
    assert result.state == "proposal_ready"
    parameters = result.intents[0].parameters
    assert parameters["total_days"] == 14
    assert parameters["start_city"] == "Los Angeles"
    assert parameters["coastal"] is True
    assert parameters["excluded_place_ids"] == ["kings_canyon_national_park"]
    assert parameters["required_place_ids"] == ["hearst_castle"]
    result = interpretation(context, "Plan a 10 day trip, include Yosemite and Sequoia")
    assert set(result.intents[0].parameters["required_place_ids"]) == {"yosemite_national_park", "sequoia_national_park"}


@pytest.mark.parametrize("followup", [TripPlanParameters(required_place_ids=["hearst_castle"], verify_official=True),
    TripPlanParameters(start_city="Los Angeles")])
def test_standard_followup_retains_accepted_profile(followup):
    baseline = {"effective_parameters": IndividualizedTripParameters(total_days=10, pace="relaxed",
        interests=["culture"], coastal=True, max_daily_drive_minutes=180,
        excluded_place_ids=["sequoia_national_park"]).model_dump()}
    effective = resolve_trip_profile(followup, baseline)
    assert isinstance(effective, IndividualizedTripParameters)
    assert (effective.total_days, effective.pace, effective.interests) == (10, "relaxed", ["culture"])
    assert (effective.max_daily_drive_minutes, effective.coastal) == (180, True)
    assert effective.excluded_place_ids == ["sequoia_national_park"]


@pytest.fixture
def expired_cache(tmp_path, monkeypatch):
    from itinerary_system.product_app import trip_replanner
    compiler = SimpleNamespace(parent_plan={"plan_id":"parent", "content_hash":"parent_hash"},
        planning_state_root=tmp_path, _planning_baseline_bundle=None, _runtime_cache_lock=threading.RLock())
    operations = ({"type":"replan_trip", "target":"trip", "parameters":TripPlanParameters().model_dump()},)
    identity = stable_content_hash({"parent":"parent_hash", "operation":operations[0]})
    cache = tmp_path / (identity + ".json")
    cache.write_text(json.dumps({"parameters":operations[0]["parameters"], "request_id":"trip_request_" + identity}))
    original = cache.read_bytes()
    def validate(parent, bundle, *, check_age=True):
        if check_age:
            raise WorkspaceError("trip_planning_evidence_expired", 409)
    monkeypatch.setattr(trip_replanner, "validate_planning_bundle", validate)
    catalog = Mock(return_value=({},set(),{},set(),set(),"start","end",{}))
    official = Mock(side_effect=WorkspaceError("official_source_unavailable",409))
    monkeypatch.setattr(trip_replanner, "_catalog", catalog)
    monkeypatch.setattr(trip_replanner, "read_official_sources", official)
    return trip_replanner, compiler, operations, cache, original, catalog, official


def test_expired_cache_default_still_rejects_acceptance_recompile(expired_cache):
    module, compiler, operations, cache, original, catalog, official = expired_cache
    with pytest.raises(WorkspaceError, match="trip_planning_evidence_expired"):
        module.compile_trip_replan(compiler, operations)
    catalog.assert_not_called()
    official.assert_not_called()
    assert cache.read_bytes() == original


def test_explicit_refresh_fetches_and_preserves_old_cache_if_fetch_fails(expired_cache):
    module, compiler, operations, cache, original, catalog, official = expired_cache
    with pytest.raises(WorkspaceError, match="official_source_unavailable"):
        module.compile_trip_replan(compiler, operations, refresh_expired=True)
    catalog.assert_called_once()
    official.assert_called_once()
    assert cache.read_bytes() == original


def test_explicit_refresh_does_not_mask_corrupt_evidence(expired_cache, monkeypatch):
    module, compiler, operations, cache, original, catalog, official = expired_cache
    def corrupt(*args, **kwargs):
        raise WorkspaceError("trip_planning_evidence_mismatch",409)
    monkeypatch.setattr(module, "validate_planning_bundle", corrupt)
    with pytest.raises(WorkspaceError, match="trip_planning_evidence_mismatch"):
        module.compile_trip_replan(compiler, operations, refresh_expired=True)
    catalog.assert_not_called()
    official.assert_not_called()
    assert cache.read_bytes() == original


def test_readonly_cannot_request_refresh(expired_cache):
    module, compiler, operations, _, _, catalog, official = expired_cache
    with pytest.raises(WorkspaceError, match="trip_planning_refresh_read_only"):
        module.compile_trip_replan(compiler, operations, read_only=True, refresh_expired=True)
    catalog.assert_not_called()
    official.assert_not_called()
