"""Planning behavior, solver oracles, official-data boundaries and serialization regressions."""

from __future__ import annotations

import asyncio
import itertools
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from itinerary_system.plans.diff import PlanDiffBuilder
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.official_travel import OFFICIAL_URLS, read_official_sources
from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService
from itinerary_system.product_app.trip_planning import TripPlanParameters, validate_trip_parameters
from itinerary_system.product_app.trip_replanner import _crosses_big_sur, _matrix_record
from itinerary_system.product_app.trip_solver import solve_trip
from itinerary_system.product_app.workspace import WorkspaceError
from itinerary_system.research_artifacts import PlanArtifactV2
from itinerary_system.routing import RouteMatrix, RouteMatrixCell

ROOT = Path(__file__).resolve().parents[2]
# Original human review inputs; now development regressions, not an unseen benchmark.
HUMAN_CASES = (
    ("我想去 red wood national park，可以加進去嗎？", "required_place_ids", ["redwood_national_park"]),
    ("我想更多的集中在海岸線，你可以修改一下路線嗎，我不是很想去國王峽谷。", "coastal", True),
    ("我朋友要求我從洛杉磯出發玩加州。", "start_city", "Los Angeles"),
    ("big sur那邊有山火，可以幫我重新確認一下官網並修改路線嗎。", "avoid_regions", ["big_sur"]),
    ("我想去Hearst castle，你可以安排一下嗎。", "required_place_ids", ["hearst_castle"]),
    ("我想去 Channel Islands，可不可以帮我安排一下？", "required_place_ids", ["channel_islands"]),
)


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    service = ProductService(
        ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"),
        tmp_path_factory.mktemp("trip_planning_contract"),
        copilot=DeterministicCopilotAdapter(),
    )
    session, _, _ = service.create_session(service.registry.default.run_id)
    return service.copilot_context(session)


@pytest.mark.parametrize("message,key,value", HUMAN_CASES)
@pytest.mark.parametrize("provider", ("deterministic", "openai"))
def test_original_human_requests_become_complete_typed_planning_requests(context, message, key, value, provider):
    calls = []

    async def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("Known bounded requests must not invoke a provider")

    adapter = (
        DeterministicCopilotAdapter()
        if provider == "deterministic"
        else OpenAICopilotAdapter(
            model="never-network",
            api_key="synthetic-unusable-test-key",
            client=SimpleNamespace(responses=SimpleNamespace(parse=forbidden)),
        )
    )
    result = asyncio.run(adapter.interpret(context=context, history=(), message=message))
    assert result.state == "proposal_ready"
    assert len(result.intents) == 1 and result.intents[0].type == "replan_trip"
    assert result.intents[0].parameters[key] == value
    assert not calls
    assert result.intents[0].parameters["max_daily_drive_minutes"] == 720
    if "國王峽谷" in message:
        assert result.intents[0].parameters["excluded_place_ids"] == ["kings_canyon_national_park"]


@pytest.mark.parametrize(
    "message",
    (
        "Don't change the coastal route",
        "不要修改海岸線",
        "Do not include Hearst Castle",
        "Make it coastal and spend two hours at Yosemite",
        "Book Hearst Castle for this coastal trip",
    ),
)
def test_unsupported_or_negated_compound_request_never_offers_a_plannable_subset(context, message):
    result = asyncio.run(DeterministicCopilotAdapter().interpret(context=context, history=(), message=message))
    assert result.state in {"clarification_required", "unsupported"}
    assert not any(i.type == "replan_trip" for i in result.intents)


@pytest.mark.parametrize(
    "changes",
    (
        {"required_place_ids": ["invented_place"]},
        {"required_place_ids": ["https://evil.example/"]},
        {"start_city": "New York"},
        {"max_daily_drive_minutes": True},
        {"max_daily_drive_minutes": 721},
        {"required_place_ids": ["hearst_castle", "hearst_castle"]},
        {"required_place_ids": ["hearst_castle"], "excluded_place_ids": ["hearst_castle"]},
        {"avoid_regions": ["big_sur"], "verify_official": False},
        {"arbitrary_tool": "shell"},
    ),
)
def test_invalid_or_unregistered_planning_parameters_are_rejected_before_execution(changes):
    with pytest.raises(ValueError):
        validate_trip_parameters({**TripPlanParameters().model_dump(), **changes})


def _matrix():
    nodes = ("start", "end", "hotel", "a", "b")
    costs = {(u, v): 10.0 for u in nodes for v in nodes if u != v}
    costs["start", "a"] = 1.0
    costs["start", "b"] = 5.0
    costs["a", "hotel"] = 2.0
    costs["b", "hotel"] = 7.0
    costs["hotel", "a"] = 8.0
    costs["hotel", "b"] = 3.0
    costs["a", "end"] = 9.0
    costs["b", "end"] = 4.0
    cells = {
        (u, v): RouteMatrixCell(
            u,
            v,
            distance_m=c * 100,
            duration_s=c * 60,
            road_validated=True,
            provider="synthetic_unit_fixture",
            context_snapshot_id="ctx",
            geometry=((36.0, -120.0), (36.01, -120.01)),
        )
        for (u, v), c in costs.items()
    }
    return RouteMatrix("fixture_matrix", "ctx", nodes, cells), costs


def test_solver_matches_independent_exhaustive_two_day_oracle():
    matrix, costs = _matrix()
    stops = {"a": {"visit_duration_minutes": 45}, "b": {"visit_duration_minutes": 45}}
    result = solve_trip(
        stops=stops,
        hotels={"hotel"},
        matrix=matrix,
        required={"a", "b"},
        start="start",
        end="end",
        days=2,
        max_drive=60,
        max_day=120,
        time_limit=2,
    )
    oracle = min(
        costs["start", a] + costs[a, "hotel"] + costs["hotel", b] + costs[b, "end"]
        for a, b in itertools.permutations(("a", "b"))
    )
    assert result["status"] == "OPTIMAL"
    assert result["objective_value"] == pytest.approx(oracle)
    assert result["optimality_gap"] == 0
    assert result["days"][0]["stop_ids"] == ["a"]
    assert result["days"][1]["stop_ids"] == ["b"]


def test_solver_does_not_relax_an_infeasible_daily_budget():
    matrix, _ = _matrix()
    with pytest.raises(WorkspaceError, match="trip_planning_constraints_infeasible"):
        solve_trip(
            stops={"a": {"visit_duration_minutes": 45}, "b": {"visit_duration_minutes": 45}},
            hotels={"hotel"},
            matrix=matrix,
            required={"a", "b"},
            start="start",
            end="end",
            days=2,
            max_drive=1,
            time_limit=2,
        )


def test_timeout_without_incumbent_is_not_a_plan(monkeypatch):
    from itinerary_system.product_app import trip_solver

    monkeypatch.setattr(trip_solver, "milp", lambda **kwargs: SimpleNamespace(x=None, status=1))
    matrix, _ = _matrix()
    with pytest.raises(WorkspaceError, match="trip_planning_solver_timeout"):
        solve_trip(
            stops={"a": {"visit_duration_minutes": 45}, "b": {"visit_duration_minutes": 45}},
            hotels={"hotel"},
            matrix=matrix,
            required={"a", "b"},
            start="start",
            end="end",
            days=2,
        )


def test_fresh_matrix_geometry_has_the_same_wire_types_and_hash_as_reloaded_geometry():
    matrix, _ = _matrix()
    fresh = _matrix_record(matrix)
    restored = json.loads(json.dumps(fresh))
    assert fresh == restored
    assert type(fresh["cells"][0]["geometry"]) is list
    assert type(fresh["cells"][0]["geometry"][0]) is list


@pytest.mark.parametrize(
    "path,expected",
    (
        (((36.371, -121.90), (36.25, -121.78)), True),
        (((36.6, -122.0), (35.5, -121.0)), True),
        (((36.6777, -121.6555), (35.6369, -120.6545)), False),
        (((35.65082726, -121.186034), (35.64, -121.18)), False),
    ),
)
def test_avoidance_checks_segments_not_only_visit_points(path, expected):
    assert _crosses_big_sur(path) is expected


def test_snapshot_change_requires_explicit_lineage_and_opt_in():
    parent = PlanArtifactV2(
        plan_id="parent",
        source_run_id="run",
        planning_request_id="request",
        catalog_snapshot_id="old_catalog",
        context_snapshot_id="old_context",
        selected_stops=(),
        created_at="",
    )
    child = replace(
        parent,
        plan_id="child",
        parent_plan_id="parent",
        catalog_snapshot_id="new_catalog",
        context_snapshot_id="new_context",
    )
    with pytest.raises(ValueError, match="share catalog_snapshot_id"):
        PlanDiffBuilder().diff(parent, child)
    assert PlanDiffBuilder(allow_snapshot_transition=True).diff(parent, child).parent_plan_id == "parent"
    with pytest.raises(ValueError, match="explicit parent-child lineage"):
        PlanDiffBuilder(allow_snapshot_transition=True).diff(parent, replace(child, parent_plan_id="someone_else"))


def test_official_reader_uses_only_fixed_sources_and_reports_conflicting_access(monkeypatch):
    calls = []
    documents = {
        OFFICIAL_URLS[
            "big_sur_park"
        ]: b"<h1>Pfeiffer Big Sur</h1><p>Current Restrictions</p><p>Pfeiffer Big Sur SP is currently Closed.</p><p>The park will reopen on October 16, 2026.</p>",
        OFFICIAL_URLS[
            "highway_1"
        ]: b"<p>Latest reported: SR 1</p><p>[IN THE CENTRAL CALIFORNIA AREA]</p><p>SR 1 is closed in Big Sur, Monterey County.</p><p>[IN THE NORTHERN CALIFORNIA AREA]</p>",
        OFFICIAL_URLS[
            "hearst_castle"
        ]: b"<h1>Hearst Castle</h1><p>Are tour reservations required?</p><p>All tours require a bus shuttle.</p><p>Arrive before the tour.</p>",
    }

    class Response:
        status = 200

        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def geturl(self):
            return self.url

        def read(self, n):
            return documents[self.url][:n]

    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            return Response(request.full_url)

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    result = read_official_sources(big_sur=True, hearst=True)
    assert calls == [OFFICIAL_URLS[k] for k in ("big_sur_park", "highway_1", "hearst_castle")]
    assert result[0]["park_access"] == "conflicting"
    assert result[1]["big_sur_highway_closure"] is True
    assert result[2]["private_vehicle_to_castle"] is False


@pytest.mark.parametrize("port", ["0", "65536", "-1", "1.5", "bad"])
def test_map_port_rejects_invalid_config(tmp_path, monkeypatch, port):
    from itinerary_system.product_app.config import ProductConfigError, ProductRuntimeConfig

    monkeypatch.setenv("PRODUCT_MAP_PORT", port)
    with pytest.raises(ProductConfigError, match="invalid_map_port"):
        ProductRuntimeConfig.from_environment(
            repository_root=tmp_path, registry_path=tmp_path / "registry.json", state_root=tmp_path / "state"
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:8186",
        "https://127.0.0.1:8186",
        "http://user@127.0.0.1:8186",
        "http://127.0.0.1:8186/path",
        "http://127.0.0.1:8080",
    ],
)
def test_explicit_map_port_preserves_loopback_boundary(tmp_path, monkeypatch, url):
    from itinerary_system.product_app.config import ProductConfigError, ProductRuntimeConfig

    monkeypatch.setenv("PRODUCT_MAP_PORT", "8186")
    monkeypatch.setenv("PRODUCT_MAP_BASE_URL", url)
    with pytest.raises(ProductConfigError, match="invalid_loopback_url"):
        ProductRuntimeConfig.from_environment(
            repository_root=tmp_path, registry_path=tmp_path / "registry.json", state_root=tmp_path / "state"
        )


def test_explicit_map_port_allows_isolated_map(tmp_path, monkeypatch):
    from itinerary_system.product_app.config import ProductRuntimeConfig

    monkeypatch.setenv("PRODUCT_MAP_PORT", "8186")
    monkeypatch.setenv("PRODUCT_MAP_BASE_URL", "http://127.0.0.1:8186")
    config = ProductRuntimeConfig.from_environment(
        repository_root=tmp_path, registry_path=tmp_path / "registry.json", state_root=tmp_path / "state"
    )
    assert config.map_base_url == "http://127.0.0.1:8186"


def test_planning_hash_matches_actual_frontend_for_mixed_numbers_and_unicode():
    import subprocess

    from itinerary_system.product_app.trip_replanner import _hash

    source = (ROOT / "src/itinerary_system/product_app/static/js/app.js").read_text(encoding="utf-8")
    function = source[
        source.index("function planningCanonicalText(") : source.index("async function normalizeTripPlanningExecution(")
    ]
    values = [
        {"confidence": 1, "cost": 1.0, "text": "海岸線與Park’s"},
        {"geometry": [[-121.1, 35.0], [-0.0, 1e-20]], "gap": 1e-7},
        [True, False, None, "#3ff0000000000000", 1],
        {"same": 1.0000000000000002},
        {"same": 1.0},
    ]
    script = (
        function
        + "\nconst {createHash}=require('node:crypto'); const values=JSON.parse(require('node:fs').readFileSync(0,'utf8')); console.log(JSON.stringify(values.map(value=>createHash('sha256').update(planningCanonicalText(value)).digest('hex'))));"
    )
    result = subprocess.run(
        ["node", "-e", script], input=json.dumps(values), capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == [_hash(value) for value in values]
    assert _hash(values[-2]) != _hash(values[-1])
    with pytest.raises(ValueError, match="non_finite"):
        _hash(float("nan"))


@pytest.mark.parametrize(
    "mutation",
    ["valid", "unknown_operation", "mixed_combination", "disabled_capability", "cardinality", "boolean_cardinality"],
)
def test_live_profile_binds_the_new_capability_without_authorizing_extra_actions(context, mutation):
    import importlib.util

    script = ROOT / "scripts/verify_e2e1_live_copilot_chat.py"
    spec = importlib.util.spec_from_file_location("trip_planning_strict_live_verifier", script)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    value = json.loads(json.dumps(context.as_dict()))
    caps = value["capabilities"]
    if mutation == "unknown_operation":
        caps["operations"]["book_trip"] = dict(caps["operations"]["replan_trip"])
    elif mutation == "mixed_combination":
        caps["executable_combinations"][-1]["operations"].append("replace_nearby")
    elif mutation == "disabled_capability":
        caps["operations"]["replan_trip"]["enabled"] = False
    elif mutation == "cardinality":
        caps["executable_combinations"][-1]["cardinality"] = 2
    if mutation == "boolean_cardinality":
        caps["executable_combinations"][-1]["cardinality"] = True
    if mutation == "valid":
        verifier._verify_current_context_extensions(value)
    else:
        with pytest.raises(verifier.LiveE2E1VerificationError, match="context_v2_invalid"):
            verifier._verify_current_context_extensions(value)


@pytest.mark.parametrize(
    "message",
    [
        "Please add Channel Islands to the itinerary",
        "我想去海峡群岛，可以安排吗？",
        "我想去海峽群島，可以安排嗎？",
    ],
)
def test_channel_islands_supported_aliases(context, message):
    result = asyncio.run(DeterministicCopilotAdapter().interpret(context=context, history=(), message=message))
    assert result.state == "proposal_ready"
    assert result.intents[0].parameters["required_place_ids"] == ["channel_islands"]
    assert result.intents[0].parameters["verify_official"] is True


@pytest.mark.parametrize(
    "message",
    [
        "我不想去 Channel Islands",
        "Don't visit Channel Islands",
        "Do not want to go to Channel Islands",
        "Book Channel Islands tickets",
        "Add Channel Islands at Anacapa",
        "Channel Islands camping trip",
        "Drive to Channel Islands Harbor",
        "Channel Islands via Prisoners Harbor",
    ],
)
def test_channel_islands_does_not_silently_drop_a_conflicting_clause(context, message):
    result = asyncio.run(DeterministicCopilotAdapter().interpret(context=context, history=(), message=message))
    assert not any(i.type == "replan_trip" for i in result.intents)


def test_port_excursion_reserves_complete_round_trip_time_without_routing_the_island(tmp_path):
    from itinerary_system.product_app.island_excursion import EXCURSION, PORT, SCORPION
    from itinerary_system.product_app.trip_replanner import _catalog

    service = ProductService(
        ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"),
        tmp_path / "catalog",
        copilot=DeterministicCopilotAdapter(),
    )
    service.load(service.registry.default.run_id)
    compiler = service._draft_compilers[service.registry.default.run_id]
    stops, _, points, required, _, _, _, _ = _catalog(
        compiler, TripPlanParameters(required_place_ids=["channel_islands"])
    )
    assert "channel_islands" in required
    assert points["channel_islands"] == (PORT[1], PORT[0])
    assert (SCORPION[1], SCORPION[0]) not in points.values()
    assert stops["channel_islands"]["visit_duration_minutes"] == 480
    t = EXCURSION["time_allowance"]
    assert (
        t["check_in_minutes"] + 2 * t["boat_each_way_minutes"] + t["island_minutes"] + t["contingency_minutes"] == 480
    )


def test_marine_geography_is_separate_from_driving_and_bound_to_the_plan():
    from copy import deepcopy

    from itinerary_system.product_app.island_excursion import EXCURSION, PORT, SCORPION, island_geography

    plan = {
        "plan_id": "child",
        "content_hash": "abc",
        "selected_stops": [{"poi_id": "channel_islands", "day": 6, "excursion": deepcopy(EXCURSION)}],
    }
    result = island_geography(plan)
    route = result["marine_routes"]["features"][0]
    assert route["geometry"]["coordinates"] == [list(PORT), list(SCORPION)]
    assert route["properties"]["plan_id"] == "child" and route["properties"]["day"] == 6
    assert route["properties"]["road_validated"] is False
    assert route["properties"]["schedule_verified"] is False
    assert route["properties"]["reservation_verified"] is False
    assert route["properties"]["round_trip"] is True
    assert "duration_s" not in route["properties"]
    plan["selected_stops"][0]["excursion"]["schedule_status"] = "verified"
    with pytest.raises(ValueError, match="island_excursion_invalid"):
        island_geography(plan)


def test_island_excursion_survives_the_real_plan_diff_duration_boundary():
    from copy import deepcopy

    from itinerary_system.product_app.island_excursion import EXCURSION, PORT

    parent = PlanArtifactV2(
        plan_id="parent",
        source_run_id="run",
        planning_request_id="request",
        catalog_snapshot_id="catalog",
        context_snapshot_id="context",
        selected_stops=(),
        created_at="",
    )
    stop = {
        "poi_id": "channel_islands",
        "day": 6,
        "stop_order": 1,
        "latitude": PORT[1],
        "longitude": PORT[0],
        "attraction_name": "Channel Islands day trip · Ventura Harbor ferry",
        "visit_duration_minutes": 480,
        "excursion": deepcopy(EXCURSION),
    }
    child = replace(
        parent,
        plan_id="child",
        parent_plan_id="parent",
        selected_stops=(stop,),
        sequence=("channel_islands",),
        day_assignments={"channel_islands": 6},
    )
    diff = PlanDiffBuilder().diff(parent, child)
    assert diff.child_plan_id == "child"
    too_long = {**stop, "visit_duration_minutes": 540}
    with pytest.raises(ValueError, match="15 to 480"):
        PlanDiffBuilder().diff(parent, replace(child, selected_stops=(too_long,)))


@pytest.fixture(scope="module")
def selection_client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from itinerary_system.product_app.api import create_product_app

    app = create_product_app(
        repository_root=ROOT,
        state_root=tmp_path_factory.mktemp("island_selection"),
        additional_allowed_authorities=("testserver",),
        copilot=DeterministicCopilotAdapter(),
    )
    with TestClient(app) as client:
        yield client, app.state.product_service


@pytest.mark.parametrize("mode", ["matching", "wrong_day", "wrong_hash", "unknown_stop", "ineligible", "absent"])
def test_real_selection_api_binds_new_stop_to_current_compiled_preview(selection_client, monkeypatch, mode):
    client, service = selection_client
    response = client.post("/api/sessions", json={})
    assert response.status_code == 200
    data = response.json()
    session = service.workspace.get(data["session"]["session_id"])
    proposal = {
        "execution_mode": "trip_replan_independent_evaluation",
        "state": "eligible",
        "child_plan_id": "child",
        "child_plan_content_hash": "actual_hash",
    }
    if mode == "wrong_hash":
        proposal["child_plan_content_hash"] = "wrong_hash"
    if mode != "absent":
        service.workspace.set_proposal(session, proposal, session.revision)
    compiled = SimpleNamespace(
        state="ineligible" if mode == "ineligible" else "eligible",
        child_plan={
            "plan_id": "child",
            "content_hash": "actual_hash",
            "selected_stops": [{"poi_id": "channel_islands", "day": 7}],
        },
    )
    compiler = SimpleNamespace(compile=lambda *args, **kwargs: compiled)
    monkeypatch.setattr(service, "compiler_for_session", lambda *args: compiler)
    response = client.post(
        f"/api/sessions/{session.session_id}/selection",
        headers={"X-Session-Token": data["mutation_token"]},
        json={
            "expected_revision": session.revision,
            "selected_day": 6 if mode == "wrong_day" else 7,
            "selected_stop_id": "invented_island" if mode == "unknown_stop" else "channel_islands",
        },
    )
    assert response.status_code == (200 if mode == "matching" else 400), response.text
    if mode == "matching":
        assert response.json()["session"]["selected_stop_id"] == "channel_islands"
        assert response.json()["session"]["selected_day"] == 7
    else:
        assert response.json()["detail"] == "invalid_selected_stop"


@pytest.mark.parametrize(
    "message",
    [
        "Please add Redwood National Park",
        "Add Red Wood National Park",
        "我想去红木国家公园，可以加进去吗？",
        "Add Redwood National and State Parks",
    ],
)
def test_redwood_aliases_are_a_required_forest_visit(context, message):
    result = asyncio.run(DeterministicCopilotAdapter().interpret(context=context, history=(), message=message))
    assert result.state == "proposal_ready"
    assert result.intents[0].parameters["required_place_ids"] == ["redwood_national_park"]
    assert result.intents[0].parameters["verify_official"] is True
    assert "Lady Bird Johnson Grove" in result.assistant_message


@pytest.mark.parametrize(
    "message",
    [
        "我不想去 red wood national park",
        "Don't visit Redwood National Park",
        "Do not want to go to Redwood National Park",
        "Book Redwood National Park",
        "Add Redwood National Park and spend two hours there",
    ],
)
def test_redwood_negation_or_unsupported_clause_is_not_silently_added(context, message):
    result = asyncio.run(DeterministicCopilotAdapter().interpret(context=context, history=(), message=message))
    assert not any(i.type == "replan_trip" for i in result.intents)


def test_redwood_and_channel_islands_both_survive_one_request(context):
    result = asyncio.run(
        DeterministicCopilotAdapter().interpret(
            context=context, history=(), message="我想去 red wood national park，也想去 Channel Islands，可以加進去嗎？"
        )
    )
    assert set(result.intents[0].parameters["required_place_ids"]) == {"redwood_national_park", "channel_islands"}


def test_redwood_catalog_uses_official_trailhead_not_discovery_centroid(tmp_path):
    from itinerary_system.product_app.trip_replanner import _catalog

    service = ProductService(
        ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"),
        tmp_path / "redwood",
        copilot=DeterministicCopilotAdapter(),
    )
    service.load(service.registry.default.run_id)
    compiler = service._draft_compilers[service.registry.default.run_id]
    stops, hotels, points, required, _, _, _, _ = _catalog(
        compiler, TripPlanParameters(required_place_ids=["redwood_national_park"])
    )
    assert "redwood_national_park" in required
    assert points["redwood_national_park"] == (41.304183, -124.018552)
    assert stops["redwood_national_park"]["visit_duration_minutes"] == 90
    assert len(points) <= 25
    assert "Bald Hills Road" in stops["redwood_national_park"]["access_note"]
    assert hotels == set(compiler.parent_plan["lodging_assignments"].values()) | {"arcata_overnight_area"}


@pytest.mark.parametrize(
    "status,expected",
    [("Open/ Hazard Conditions", "open_with_hazards"), ("Open", "open"), ("Closed", "closed"), ("Unclear", "unknown")],
)
def test_redwood_official_reader_binds_road_status_and_only_reads_its_fixed_sources(monkeypatch, status, expected):
    from itinerary_system.product_app.official_travel import REDWOOD_URLS

    calls = []
    documents = {
        REDWOOD_URLS[
            "redwood_lodging"
        ]: b"<h1>Staying Near Redwood National and State Parks</h1><p>Lodging outside the park.</p><p>Arcata Chamber of Commerce</p>",
        REDWOOD_URLS[
            "redwood_walk"
        ]: b"<h1>Walk the Lady Bird Johnson Grove</h1><p>Bald Hills Road.</p><p>Duration 45-60 Minutes</p>",
        REDWOOD_URLS["redwood_conditions"]: (
            "<h1>Redwood National and State Parks</h1><h2>Roads &amp; Day Use Areas</h2><p>Bald Hills Road "
            + status
            + " Road Damage</p><h2>Alerts</h2><h2>Trails</h2><p>park trails not listed below are open</p>"
        ).encode(),
    }

    class Response:
        status = 200

        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def geturl(self):
            return self.url

        def read(self, n):
            return documents[self.url][:n]

    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            return Response(request.full_url)

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    rows = read_official_sources(big_sur=False, hearst=False, redwood=True)
    assert calls == list(REDWOOD_URLS.values())
    assert next(r for r in rows if r["source_id"] == "redwood_conditions")["park_access"] == expected


def test_retained_overview_consumes_actual_day_bounds_and_rejects_wrong_shapes():
    import subprocess

    source = (ROOT / "src/itinerary_system/product_app/static/js/map-controller-v2.js").read_text(encoding="utf-8")
    source = source.replace(
        'from "./interest-regions.js"',
        "from " + json.dumps((ROOT / "src/itinerary_system/product_app/static/js/interest-regions.js").as_uri()),
    )
    source += "\nimport assert from 'node:assert/strict'; const h={minLon:-123,minLat:33.5,maxLon:-117.5,maxLat:38.5,maxZoom:15}; const points=[[-124.1,41.31],[-122.4,37.8]]; const plan={route_path:{features:points.map(p=>({properties:{day:2},geometry:{type:'Point',coordinates:p}}))},stops:{features:[]},validated_legs:{features:[]}}; const b=boundsForDay([plan],2); assert.deepEqual(b,[-124.1,37.8,-122.4,41.31]); assert.equal(retainedOverview(b,h).maxzoom,2); assert.ok(retainedOverview([-122.5,34,-118,38],h).maxzoom < h.maxZoom); assert.equal(retainedOverview([-121.6,35.9,-121.4,36.1],h),null); assert.throws(()=>retainedOverview([[0,0],[1,1]],h),/bounds/); assert.throws(()=>retainedOverview(b,{}),/header/); console.log('6 overview integration checks passed');"
    result = subprocess.run(["node", "--input-type=module"], input=source, capture_output=True, text=True, check=True)
    assert "6 overview integration checks passed" in result.stdout


@pytest.mark.parametrize(
    "message",
    [
        "我現在時間更充裕了，想多三天去優勝美地玩，幫我安排一下",
        "Add three days to Yosemite",
        "優勝美地增加3天",
        "Yosemite extra 3 days",
    ],
)
@pytest.mark.parametrize("provider", ["deterministic", "openai"])
def test_yosemite_extension_language_without_paid_calls(context, message, provider):
    async def forbidden(**kwargs):
        raise AssertionError("No paid call authorized for bounded extension")

    adapter = (
        DeterministicCopilotAdapter()
        if provider == "deterministic"
        else OpenAICopilotAdapter(
            model="never-network",
            api_key="synthetic-unusable-test-key",
            client=SimpleNamespace(responses=SimpleNamespace(parse=forbidden)),
        )
    )
    result = asyncio.run(adapter.interpret(context=context, history=(), message=message))
    assert result.state == "proposal_ready"
    params = result.intents[0].parameters
    assert params["additional_yosemite_days"] == 3
    assert params["required_place_ids"] == ["yosemite_national_park"]
    assert params["verify_official"] is True


@pytest.mark.parametrize(
    "message",
    [
        "不要增加三天去優勝美地",
        "Do not add three days to Yosemite",
        "優勝美地增加兩天",
        "Add three hours to Yosemite",
        "Cancel Yosemite extra three days",
    ],
)
def test_yosemite_extension_does_not_assume_unsupported_duration(context, message):
    result = asyncio.run(DeterministicCopilotAdapter().interpret(context=context, history=(), message=message))
    assert not any(i.type == "replan_trip" for i in result.intents)


@pytest.mark.parametrize(
    "changes",
    [
        {"additional_yosemite_days": True},
        {"additional_yosemite_days": 4},
        {"additional_yosemite_days": "3"},
        {"verify_official": False},
        {"required_place_ids": []},
    ],
)
def test_yosemite_extension_rejects_incoherent_or_untyped_parameters(changes):
    from itinerary_system.product_app.trip_planning import YosemiteExtensionParameters

    params = YosemiteExtensionParameters(
        additional_yosemite_days=3, required_place_ids=["yosemite_national_park"], verify_official=True
    ).model_dump()
    with pytest.raises(ValueError):
        validate_trip_parameters({**params, **changes})


def _yosemite_fixture(cost=10):
    from itinerary_system.product_app.yosemite_extension import YOSEMITE_POIS

    nodes = ("base", *YOSEMITE_POIS)
    cells = {
        (u, v): RouteMatrixCell(
            u,
            v,
            distance_m=1000,
            duration_s=cost * 60,
            road_validated=True,
            provider="synthetic_unit_fixture",
            context_snapshot_id="ctx",
        )
        for u in nodes
        for v in nodes
        if u != v
    }
    base = {
        "days": [
            {
                "day": d,
                "start_anchor_id": "base",
                "end_anchor_id": "base",
                "stop_ids": ["yosemite_national_park"] if d == 3 else [f"old_{d}"],
            }
            for d in range(1, 8)
        ],
        "status": "OPTIMAL",
        "objective": 0,
    }
    return base, RouteMatrix("fixture", "ctx", nodes, cells)


def test_yosemite_three_day_orders_match_independent_directed_oracle():
    from itinerary_system.product_app.yosemite_extension import GROUPS, extend_yosemite_solution

    base, matrix = _yosemite_fixture()
    matrix.cells[("base", GROUPS[0][0])] = replace(matrix.cell("base", GROUPS[0][0]), duration_s=6000)
    out = extend_yosemite_solution(base, matrix, 720)
    assert len(out["days"]) == 10 and len(base["days"]) == 7
    assert [d["day"] for d in out["days"]] == list(range(1, 11))
    for group, day in zip(GROUPS, out["days"][3:6], strict=True):
        oracle = min(
            itertools.permutations(group),
            key=lambda order: sum(
                matrix.cell(u, v).require_duration_s() for u, v in zip(("base", *order), (*order, "base"), strict=True)
            ),
        )
        assert day["stop_ids"] == list(oracle)
        assert day["start_anchor_id"] == day["end_anchor_id"] == "base"
    assert out["days"][-1]["stop_ids"] == ["old_7"]
    assert out["status"] == "FEASIBLE" and out["best_bound"] is None
    assert "no joint ten-day global optimum" in out["scope"]


def test_yosemite_extension_refuses_infeasible_day():
    from itinerary_system.product_app.yosemite_extension import extend_yosemite_solution

    base, matrix = _yosemite_fixture(250)
    with pytest.raises(WorkspaceError, match="yosemite_extension_day_infeasible"):
        extend_yosemite_solution(base, matrix, 720)


@pytest.mark.parametrize("tamper", ["guide", "latitude", "visit_duration_minutes", "day", "missing"])
def test_yosemite_validator_binds_every_guide_and_day(tamper):
    from copy import deepcopy

    from itinerary_system.product_app.yosemite_extension import (
        YOSEMITE_POIS,
        extend_yosemite_solution,
        validate_yosemite_extension,
    )

    base, matrix = _yosemite_fixture()
    days = extend_yosemite_solution(base, matrix, 720)["days"]
    stops = [{"poi_id": "yosemite_national_park", "day": 3}]
    stops += [{**deepcopy(YOSEMITE_POIS[key]), "day": d["day"]} for d in days[3:6] for key in d["stop_ids"]]
    validate_yosemite_extension(SimpleNamespace(selected_stops=stops), days, True)
    if tamper == "missing":
        stops.pop()
    elif tamper == "guide":
        stops[1]["guide"]["photo"]["url"] = "https://evil.example/photo"
    elif tamper == "day":
        stops[1]["day"] = 7
    else:
        stops[1][tamper] = 0
    with pytest.raises(WorkspaceError):
        validate_yosemite_extension(SimpleNamespace(selected_stops=stops), days, True)


@pytest.mark.parametrize("mode", ["matching", "wrong_hash", "absent", "unknown_day", "ineligible"])
def test_extended_day_selection_requires_current_evaluated_preview(selection_client, monkeypatch, mode):
    client, service = selection_client
    data = client.post("/api/sessions", json={}).json()
    session = service.workspace.get(data["session"]["session_id"])
    if mode != "absent":
        service.workspace.set_proposal(
            session,
            {
                "execution_mode": "trip_replan_independent_evaluation",
                "state": "eligible",
                "child_plan_id": "child",
                "child_plan_content_hash": "wrong" if mode == "wrong_hash" else "hash",
            },
            session.revision,
        )
    compiled = SimpleNamespace(
        state="ineligible" if mode == "ineligible" else "eligible",
        child_plan={"plan_id": "child", "content_hash": "hash", "selected_stops": [{"poi_id": "old_7", "day": 10}]},
    )
    monkeypatch.setattr(
        service, "compiler_for_session", lambda *args: SimpleNamespace(compile=lambda *args, **kwargs: compiled)
    )
    response = client.post(
        f"/api/sessions/{session.session_id}/selection",
        headers={"X-Session-Token": data["mutation_token"]},
        json={"expected_revision": session.revision, "selected_day": 11 if mode == "unknown_day" else 10},
    )
    assert response.status_code == (200 if mode == "matching" else 400), response.text
    if mode == "matching":
        assert response.json()["session"]["selected_day"] == 10


@pytest.mark.parametrize(
    "mode",
    [
        "matching",
        "extended_day",
        "no_day",
        "wrong_hash",
        "wrong_plan",
        "absent",
        "ineligible",
        "wrong_day",
        "unknown_segment",
        "unused_edge",
        "unvalidated",
        "fallback",
    ],
)
def test_preview_route_selection_requires_current_compiled_plan_and_matching_day(selection_client, monkeypatch, mode):
    client, service = selection_client
    data = client.post("/api/sessions", json={}).json()
    session = service.workspace.get(data["session"]["session_id"])
    if mode != "absent":
        service.workspace.set_proposal(
            session,
            {
                "execution_mode": "trip_replan_independent_evaluation",
                "state": "eligible",
                "child_plan_id": "wrong" if mode == "wrong_plan" else "child",
                "child_plan_content_hash": "wrong" if mode == "wrong_hash" else "hash",
            },
            session.revision,
        )
    compiled = SimpleNamespace(
        state="ineligible" if mode == "ineligible" else "eligible",
        child_plan={
            "plan_id": "child",
            "content_hash": "hash",
            "selected_stops": [{"poi_id": "old_7", "day": 14}],
            "route_ids_by_day": {"5": "trip_route_5", "14": "trip_route_14"},
        },
        route_legs=(
            {"day": 5, "origin_id": "a", "destination_id": "b"},
            {"day": 14, "origin_id": "b", "destination_id": "c"},
        ),
        source_bundle=SimpleNamespace(
            route_matrix={
                "cells": [
                    {
                        "route_leg_id": "leg_preview_only",
                        "origin_id": "a",
                        "destination_id": "unused" if mode == "unused_edge" else "b",
                        "road_validated": mode != "unvalidated",
                        "fallback_used": mode == "fallback",
                    },
                    {
                        "route_leg_id": "leg_preview_fourteen",
                        "origin_id": "b",
                        "destination_id": "c",
                        "road_validated": True,
                        "fallback_used": False,
                    },
                ]
            }
        ),
    )
    monkeypatch.setattr(
        service, "compiler_for_session", lambda *args: SimpleNamespace(compile=lambda *args, **kwargs: compiled)
    )
    day = None if mode == "no_day" else 14 if mode == "extended_day" else 4 if mode == "wrong_day" else 5
    route = (
        "leg_preview_fourteen"
        if mode == "extended_day"
        else "leg_unknown"
        if mode == "unknown_segment"
        else "leg_preview_only"
    )
    before = session.public_dict()
    response = client.post(
        f"/api/sessions/{session.session_id}/selection",
        headers={"X-Session-Token": data["mutation_token"]},
        json={
            "expected_revision": session.revision,
            "selected_day": day,
            "selected_segment_id": route,
            "mutation_id": "mutation_" + "a" * 32,
        },
    )
    allowed = mode in {"matching", "extended_day", "no_day"}
    assert response.status_code == (200 if allowed else 400), response.text
    if allowed:
        assert response.json()["session"]["selected_segment_id"] == route
        assert response.json()["session"]["accepted_plan_id"] == before["accepted_plan_id"]
        assert response.json()["session"]["draft"] == before["draft"]
    else:
        assert service.workspace.get(session.session_id).public_dict() == before


@pytest.mark.parametrize("extension", [False, True])
def test_real_openai_wire_schema_preserves_extension_and_ordinary_parameters(extension):
    from itinerary_system.product_app.openai_copilot import OpenAIInterpretationSchemaV2, _to_dataclass
    from itinerary_system.product_app.trip_planning import YosemiteExtensionParameters

    params = (
        YosemiteExtensionParameters(
            additional_yosemite_days=3, required_place_ids=["yosemite_national_park"], verify_official=True
        ).model_dump()
        if extension
        else TripPlanParameters(required_place_ids=["hearst_castle"], verify_official=True).model_dump()
    )
    parsed = OpenAIInterpretationSchemaV2.model_validate(
        {
            "schema_version": "copilot-interpretation-v2",
            "state": "proposal_ready",
            "assistant_message": "Suggestion only",
            "intents": [{"type": "replan_trip", "parameters": params}],
        }
    )
    actual = _to_dataclass(parsed).intents[0].parameters
    assert actual == params
    assert ("additional_yosemite_days" in actual) == extension
    assert validate_trip_parameters(actual).model_dump() == params


def test_yosemite_unmatched_paraphrase_uses_provider_and_preserves_ten_day_intent(context):
    from itinerary_system.product_app.openai_copilot import OpenAIInterpretationSchemaV2
    from itinerary_system.product_app.trip_planning import YosemiteExtensionParameters

    params = YosemiteExtensionParameters(
        additional_yosemite_days=3, required_place_ids=["yosemite_national_park"], verify_official=True
    ).model_dump()
    calls = []

    async def parse(**kwargs):
        calls.append(kwargs)
        assert "additional_yosemite_days=3" in kwargs["instructions"]
        return SimpleNamespace(
            output=[],
            output_parsed=OpenAIInterpretationSchemaV2.model_validate(
                {
                    "schema_version": "copilot-interpretation-v2",
                    "state": "proposal_ready",
                    "assistant_message": "Suggestion only",
                    "intents": [{"type": "replan_trip", "parameters": params}],
                }
            ),
        )

    adapter = OpenAICopilotAdapter(
        model="never-network", api_key="injected-client", client=SimpleNamespace(responses=SimpleNamespace(parse=parse))
    )
    result = asyncio.run(
        adapter.interpret(
            context=context,
            history=(),
            message="行程從七天改成十天，把新增的時間都留給優勝美地，請安排不同景點，附照片、簡介和有來源的遊客評論。",
        )
    )
    assert len(calls) == 1 and result.state == "proposal_ready"
    assert result.intents[0].parameters == params


@pytest.mark.parametrize("provider", ["deterministic", "openai"])
def test_one_day_trip_extension_is_a_personalized_replan(context, provider):
    async def forbidden(**kwargs):
        raise AssertionError("An unsupported trip duration must not invoke a provider")

    adapter = (
        DeterministicCopilotAdapter()
        if provider == "deterministic"
        else OpenAICopilotAdapter(
            model="never-network",
            api_key="injected-client",
            client=SimpleNamespace(responses=SimpleNamespace(parse=forbidden)),
        )
    )
    result = asyncio.run(
        adapter.interpret(context=context, history=(), message="I want to add one more day .can u plan that for me?")
    )
    assert result.state == "proposal_ready"
    assert result.intents[0].type == "replan_trip"
    assert result.intents[0].parameters["total_days"] == 8
    assert "insertion road-route evidence" not in result.assistant_message


@pytest.mark.parametrize(
    "message,days",
    [
        ("我的旅行計劃有變，我想製作一個10天的旅行計劃", 10),
        ("我想花兩週去旅行", 14),
        ("我想製作十天的行程", 10),
        ("Plan a two week trip", 14),
        ("Plan a fourteen day itinerary", 14),
        ("把行程從七天改成十天", 10),
        ("I want to add one more day", 8),
        ("我想多三天去旅行", 10),
        ("add two more days", 9),
    ],
)
@pytest.mark.parametrize("provider", ["deterministic", "openai"])
def test_individualized_duration_is_explicit_and_does_not_assume_yosemite(context, message, days, provider):
    async def forbidden(**kwargs):
        raise AssertionError("Recognized typed durations must not make a paid provider call")

    adapter = (
        DeterministicCopilotAdapter()
        if provider == "deterministic"
        else OpenAICopilotAdapter(
            model="never-network",
            api_key="injected-client",
            client=SimpleNamespace(responses=SimpleNamespace(parse=forbidden)),
        )
    )
    result = asyncio.run(adapter.interpret(context=context, history=(), message=message))
    assert result.state == "proposal_ready"
    p = result.intents[0].parameters
    assert p["total_days"] == days
    assert "additional_yosemite_days" not in p
    assert "yosemite_national_park" not in p["required_place_ids"]
    assert validate_trip_parameters(p).total_days == days


@pytest.mark.parametrize(
    "message",
    [
        "我想玩30天",
        "10天或14天的旅行",
        "如果下雨就改成10天",
        "不要10天",
        "I do not want ten days",
        "14天旅行，幫我付款預訂酒店",
        "14天旅行，不喜歡博物館",
        "14天旅行，預算一千美元",
        "10天旅行，需要無障礙路線",
        "10天旅行，11月出發",
        "10天旅行，每天三個景點",
    ],
)
def test_individualized_duration_never_executes_an_incomplete_or_negated_subset(context, message):
    result = asyncio.run(DeterministicCopilotAdapter().interpret(context=context, history=(), message=message))
    assert result.state in {"clarification_required", "unsupported"}
    assert not any(i.type == "replan_trip" for i in result.intents)


@pytest.mark.parametrize("message", ["第14天", "把第14天的景點移到第10天", "Tell me about this museum stop"])
def test_local_day_labels_and_place_mentions_do_not_create_a_whole_trip_request(context, message):
    from itinerary_system.product_app.trip_planning import trip_request_interpretation

    assert trip_request_interpretation(context, message) is None


def test_personalization_carries_pace_interests_and_daily_drive_budget_together(context):
    result = asyncio.run(
        DeterministicCopilotAdapter().interpret(
            context=context, history=(), message="我想花兩週去旅行，節奏輕鬆，喜歡文化和博物館，每天開車不超過三小時"
        )
    )
    assert result.state == "proposal_ready"
    p = result.intents[0].parameters
    assert (p["total_days"], p["pace"], p["interests"], p["max_daily_drive_minutes"]) == (
        14,
        "relaxed",
        ["culture"],
        180,
    )
    assert set(p["profile_updates"]) == {"pace", "interests", "max_daily_drive_minutes"}


@pytest.mark.parametrize(
    "bad",
    [
        {"total_days": True},
        {"total_days": 0},
        {"total_days": 22},
        {"pace": "invented"},
        {"interests": ["culture", "culture"]},
        {"interests": ["booking"]},
        {"profile_updates": ["total_days"]},
    ],
)
def test_individualized_contract_rejects_invalid_personalization(bad):
    from itinerary_system.product_app.trip_planning import IndividualizedTripParameters

    with pytest.raises(ValueError):
        validate_trip_parameters({**IndividualizedTripParameters(total_days=14).model_dump(), **bad})


def test_fourteen_day_solver_visits_distinct_places_without_padding_or_relaxing_limits():
    stops = {f"p{i}": {"visit_duration_minutes": 45} for i in range(14)}
    nodes = ("start", "end", "hotel", *stops)
    cells = {
        (u, v): RouteMatrixCell(
            u,
            v,
            distance_m=100,
            duration_s=60,
            road_validated=True,
            provider="synthetic_unit_fixture",
            context_snapshot_id="ctx",
            geometry=((36.0, -120.0), (36.01, -120.01)),
        )
        for u in nodes
        for v in nodes
        if u != v
    }
    result = solve_trip(
        stops=stops,
        hotels={"hotel"},
        matrix=RouteMatrix("matrix14", "ctx", nodes, cells),
        required=set(stops),
        start="start",
        end="end",
        days=14,
        max_drive=2,
        max_day=47,
        max_stops_per_day=2,
        time_limit=5,
    )
    assert len(result["days"]) == 14
    assert {s for d in result["days"] for s in d["stop_ids"]} == set(stops)
    assert all(len(d["stop_ids"]) == 1 for d in result["days"])
    assert result["objective_value"] == pytest.approx(28.0)


def test_new_duration_inherits_only_accepted_profile_and_allows_explicit_updates():
    from itinerary_system.product_app.trip_planning import IndividualizedTripParameters, resolve_trip_profile

    previous = IndividualizedTripParameters(
        total_days=10,
        pace="relaxed",
        interests=["nature"],
        max_daily_drive_minutes=180,
        coastal=True,
        excluded_place_ids=["kings_canyon_national_park"],
    )
    baseline = {"effective_parameters": previous.model_dump()}
    requested = IndividualizedTripParameters(total_days=14)
    effective = resolve_trip_profile(requested, baseline)
    assert effective.total_days == 14
    assert (effective.pace, effective.interests, effective.max_daily_drive_minutes, effective.coastal) == (
        "relaxed",
        ["nature"],
        180,
        True,
    )
    assert effective.excluded_place_ids == ["kings_canyon_national_park"]
    corrected = resolve_trip_profile(
        IndividualizedTripParameters(
            total_days=14, pace="active", interests=["culture"], profile_updates=["pace", "interests"]
        ),
        baseline,
    )
    assert (corrected.pace, corrected.interests, corrected.max_daily_drive_minutes) == ("active", ["culture"], 180)
    assert resolve_trip_profile(requested, {}).pace == "balanced"
    assert requested.pace == "balanced"  # The reviewable request remains immutable.
