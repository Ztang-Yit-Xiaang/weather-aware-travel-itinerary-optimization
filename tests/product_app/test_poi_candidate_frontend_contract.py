from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "product_app_registry.json"
STATIC = ROOT / "src" / "itinerary_system" / "product_app" / "static"
APP = STATIC / "js" / "app.js"
MAP = STATIC / "js" / "map.js"
CONTROLLER = STATIC / "js" / "map-controller-v2.js"
CSS = STATIC / "css" / "app.css"


def test_candidate_discovery_uses_exact_route_context_and_bounded_filters() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "/map/poi-candidates?${query.toString()}" in source
    assert "expected_revision: String(expected.revision)" in source
    assert "route_leg_id: routeLegId" in source
    assert 'query.set("replacement_target_id", state.poiDiscovery.replacementTargetId)' in source
    assert 'query.set("role", state.poiDiscovery.role)' in source
    assert "Math.min(10, state.poiDiscovery.limit)" in source
    assert "normalizePoiCandidateResponse(payload, expected)" in source


def test_candidate_selection_has_no_fallback_and_replacement_is_explicitly_mapped() -> None:
    source = APP.read_text(encoding="utf-8")
    selection = source[source.index("async function selectPoiCandidate"):]
    selection = selection[: selection.index("async function useSelectedCandidateReplacement")]
    replacement = source[source.index("async function useSelectedCandidateReplacement"):]
    replacement = replacement[: replacement.index("async function closePoiDiscovery")]

    assert "row.candidate_id === candidateId" in selection
    assert "selected_candidate_id: candidate.candidate_id" in selection
    assert "candidates[0]" not in selection
    assert 'mapping?.draft_type !== "replace_nearby"' in replacement
    assert "mapping.candidate_id !== candidate.place?.place_id" in replacement
    assert "mapping.target_stop_id !== context.replacement_target_id" in replacement
    assert '"replace_nearby"' in replacement
    assert "{ candidate_id: mapping.candidate_id }" in replacement
    assert "accept" not in replacement.lower()
    assert "keep-original" not in replacement.lower()


def test_null_metrics_and_distinct_candidate_states_are_rendered_truthfully() -> None:
    source = APP.read_text(encoding="utf-8")

    assert 'return value === null || value === undefined || value === "" ? "Unavailable"' in source
    for label in (
        "Nearby",
        "Route-near",
        "Likely feasible",
        "Evaluated feasible",
        "Recommended",
        "Unavailable / insufficient evidence",
    ):
        assert label in source
    assert 'burden.recommended === true && burden.ranking_eligible === true' in source
    for field in (
        "marginal_travel_minutes",
        "marginal_travel_distance_m",
        "predicted_arrival",
        "open_at_arrival",
        "parking_minutes",
        "walking_minutes",
        "waiting_minutes",
        "weather_suitability",
        "blocking_codes",
    ):
        assert field in source


def test_external_links_are_scheme_checked_noopener_and_host_visible() -> None:
    source = APP.read_text(encoding="utf-8")
    helper = source[source.index("function safeExternalLink"):]
    helper = helper[: helper.index("function candidateStateLabel")]

    assert '["http:", "https:"]' in helper
    assert "parsed.username || parsed.password || !parsed.hostname" in helper
    assert 'rel="noopener noreferrer"' in helper
    assert "parsed.hostname" in helper


def test_candidate_layer_is_separate_bounded_and_has_keyboard_equivalent() -> None:
    app = APP.read_text(encoding="utf-8")
    controller = CONTROLLER.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")

    assert 'candidates.length > 10' in controller
    assert "FROZEN_ROUTE_QUERY_HASH_PATTERN = /^[0-9a-f]{16}$/" in MAP.read_text(encoding="utf-8")
    assert 'map.addSource("v2-poi-candidates"' in controller
    assert 'id: "v2-poi-access-points"' in controller
    assert "candidate?.place?.display_coordinate?.longitude" in controller
    assert "candidate?.selected_access_point?.coordinate?.longitude" in controller
    assert "onCandidateSelect" in controller
    assert 'data-poi-candidate="${escapeHtml(candidate.candidate_id)}"' in app
    assert 'aria-pressed="${selectedRow}"' in app
    assert 'id="poi-layer-toggle"' in app
    assert ".poi-candidate-list button { width: 100%; min-height: 52px;" in css
    assert '.poi-candidate-card button { width: 100%; min-height: 44px;' in css
    assert "@media (max-width: 430px)" in css


def test_exact_normalizer_rejects_context_recommendation_url_and_density_forgery() -> None:
    module_url = MAP.as_uri()
    script = f"""
      import {{ normalizePoiCandidateResponse }} from {module_url!r};
      const hash = "a".repeat(64);
      const routeHash = "001da97c5482f649";
      const expected = {{sessionId:"session_" + "b".repeat(32), revision:4, day:7,
        routeLegId:"leg_pt", replacementTargetId:"parent_stop"}};
      const burden = {{schema_version:"product-candidate-burden-v1", context_kind:"replacement",
        candidate_id:"registered_place", place_id:"registered_place",
        predecessor_id:"before", successor_id:"after", replacement_target_id:"parent_stop",
        baseline_route_leg_ids:["leg_pt","leg_ts"], baseline_travel_minutes:20,
        baseline_travel_distance_m:1000, geographic_distance_m:50,
        predecessor_to_candidate_minutes:10, candidate_to_successor_minutes:15,
        predecessor_to_successor_minutes:null, marginal_travel_minutes:-59.07,
        marginal_travel_distance_m:-200, visit_minutes:null, parking_minutes:null,
        walking_minutes:null, waiting_minutes:null, total_insertion_minutes:null,
        nearby:true, route_near:true, likely_feasible:false, evaluated_feasible:true,
        ranking_eligible:false, recommended:false, candidate_state:"evaluated_feasible",
        evaluator_rank:null, evaluator_evidence_refs:["evidence"],
        blocking_codes:["ranking_unavailable"], evidence_refs:[
          "matrix","context","leg_pc",routeHash,"leg_cs","101da97c5482f649",
          "leg_pt","201da97c5482f649","leg_ts","301da97c5482f649","evidence"
        ]}};
      const candidate = {{candidate_id:"registered_place", place:{{
        place_id:"registered_place", name:"Exact place", place_categories:["attraction"],
        display_coordinate:{{longitude:-121.9, latitude:36.3}}, description:null,
        official_url:"https://example.com/place", informational_urls:[], source_refs:["matrix"],
        source_freshness:null, opening_hours_evidence_ref:null,
        recommended_visit_minutes:null, weather_suitability:null}},
        selected_access_point:{{access_point_id:"access", access_type:"provider_endpoint",
          coordinate:{{longitude:-121.91, latitude:36.31}}, source_ref:"matrix",
          road_validated:true, access_confidence:"provider_access_point",
          evidence_refs:[routeHash,"101da97c5482f649"]}},
        sources:[{{source_id:"matrix", source_type:"curated_dataset",
          source_url:"https://example.com/source", retrieved_at:null}}], burden,
        precheck:{{predicted_arrival:null, open_at_arrival:null, status:"unavailable", evidence_refs:[]}},
        route_evidence_refs:[
          {{role:"predecessor_candidate", route_leg_id:"leg_pc", query_hash:routeHash}},
          {{role:"candidate_successor", route_leg_id:"leg_cs", query_hash:"101da97c5482f649"}},
          {{role:"predecessor_target", route_leg_id:"leg_pt", query_hash:"201da97c5482f649"}},
          {{role:"target_successor", route_leg_id:"leg_ts", query_hash:"301da97c5482f649"}}
        ],
        registered_replacement:{{draft_type:"replace_nearby", target_stop_id:"parent_stop",
          candidate_id:"registered_place"}}}};
      const payload = {{schema_version:"product-poi-candidates-v1", session_id:expected.sessionId,
        session_revision:4, context:{{kind:"replacement", day:7, route_leg_id:"leg_pt",
          replacement_target_id:"parent_stop", predecessor_id:"before", successor_id:"after",
          baseline_route_leg_ids:["leg_pt","leg_ts"], baseline_travel_minutes:20,
          baseline_travel_distance_m:1000}}, catalog:{{catalog_id:"catalog", catalog_sha256:hash,
          generated_at:"2026-08-08T00:00:00Z"}}, routing:{{matrix_id:"matrix",
          context_snapshot_id:"context", source_bundle_id:"bundle", source_content_sha256:hash,
          road_validated_only:true, fallback_allowed:false}}, candidates:[candidate]}};
      if (normalizePoiCandidateResponse(payload, expected).candidates.length !== 1) process.exit(2);
      const rejects = [];
      const reject = (mutate) => {{ const forged = structuredClone(payload); mutate(forged);
        try {{ normalizePoiCandidateResponse(forged, expected); rejects.push(false); }}
        catch {{ rejects.push(true); }} }};
      reject((value) => {{ value.context.route_leg_id = "leg_other"; }});
      reject((value) => {{ value.candidates[0].burden.candidate_state = "recommended"; }});
      reject((value) => {{ value.candidates[0].sources[0].source_url = "javascript:alert(1)"; }});
      reject((value) => {{ value.candidates = Array(11).fill(candidate); }});
      if (!rejects.every(Boolean)) process.exit(3);
      console.log("poi normalizer adversarial cases passed");
    """
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "poi normalizer adversarial cases passed" in completed.stdout


def test_actual_api_payload_normalizer_rejects_cross_field_forgery(tmp_path: Path) -> None:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "poi-normalizer-state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={})
        assert created.status_code == 200
        session = created.json()["session"]
        headers = {"X-Session-Token": created.json()["mutation_token"]}
        response = client.get(
            f"/api/sessions/{session['session_id']}/map/poi-candidates",
            headers=headers,
            params={
                "expected_revision": session["revision"],
                "day": 7,
                "route_leg_id": "leg_hotel_milo_santa_barbara_golden_gate_bridge",
                "replacement_target_id": "golden_gate_bridge",
                "limit": 10,
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["candidates"]
    assert payload["candidates"][0]["registered_replacement"] is not None

    module_url = MAP.as_uri()
    script = f"""
      import {{ normalizePoiCandidateResponse }} from {module_url!r};
      const payload = {json.dumps(payload, separators=(",", ":"))};
      const expected = {{sessionId:payload.session_id, revision:payload.session_revision,
        day:payload.context.day, routeLegId:payload.context.route_leg_id,
        replacementTargetId:payload.context.replacement_target_id}};
      if (normalizePoiCandidateResponse(payload, expected).candidates.length < 1) process.exit(2);
      const cases = [
        ["burden_context", (value) => {{
          value.candidates[0].burden.baseline_route_leg_ids = ["leg_forged"];
          value.candidates[0].burden.baseline_travel_minutes += 1;
          value.candidates[0].burden.baseline_travel_distance_m += 1;
        }}],
        ["evaluated_state", (value) => {{
          value.candidates[0].burden.evaluated_feasible = false;
        }}],
        ["precheck_evidence", (value) => {{
          value.candidates[0].precheck.status = "passed";
          value.candidates[0].precheck.predicted_arrival = "2026-08-08T13:00:00Z";
          value.candidates[0].precheck.open_at_arrival = true;
          value.candidates[0].precheck.evidence_refs = [];
          value.candidates[0].burden.likely_feasible = true;
        }}],
        ["unavailable_precheck_claims", (value) => {{
          value.candidates[0].precheck.predicted_arrival = "2026-08-08T13:00:00Z";
          value.candidates[0].precheck.open_at_arrival = true;
        }}],
        ["replacement_target", (value) => {{
          value.candidates[0].registered_replacement.target_stop_id = "other_stop";
        }}],
        ["road_validation", (value) => {{
          value.candidates[0].selected_access_point.road_validated = false;
        }}],
        ["access_provenance", (value) => {{
          value.candidates[0].selected_access_point.source_ref = "foreign_matrix";
        }}],
        ["route_evidence", (value) => {{
          value.candidates[0].route_evidence_refs.forEach((reference) => {{
            reference.route_leg_id = `foreign_${{reference.role}}`;
          }});
        }}],
      ];
      const accepted = [];
      for (const [label, mutate] of cases) {{
        const forged = structuredClone(payload);
        mutate(forged);
        try {{ normalizePoiCandidateResponse(forged, expected); accepted.push(label); }}
        catch {{ /* Expected fail-closed behavior. */ }}
      }}
      if (accepted.length) {{ console.error(accepted.join(",")); process.exit(3); }}
      console.log("actual API cross-field forgeries rejected");
    """
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "actual API cross-field forgeries rejected" in completed.stdout
