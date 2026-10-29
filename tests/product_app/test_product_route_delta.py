"""Non-network guard tests for the bounded immutable product-road delta."""

import importlib.util
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler
from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_app.route_delta import load_frozen_route_delta
from itinerary_system.routing.road_cache_builder import iter_route_leg_requests

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("product_route_delta", ROOT / "scripts/build_product_route_delta.py")
delta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delta)
FROZEN = ROOT / "results/route_deltas/product_pfeiffer_point_lobos_20260908_v1"
DESCRIPTOR_HASH = "525e1bb70f864759957a9e7fb34dc3901ca31ed72424c1fdf6bbaf30bef32c9e"


def witness():
    image = "ghcr.io/project-osrm/osrm-backend@sha256:" + "a" * 64
    provenance = {
        "osrm_image": image,
        "endpoint": delta.ENDPOINT,
        "provider_kind": "local_osrm",
        "osm_pbf_sha256": "b" * 64,
    }
    container = {
        "running": True,
        "configured_image": image,
        "image_id": "sha256:" + "a" * 64,
        "command": ["osrm-routed", "--algorithm", "mld", "/data/california.osrm"],
        "mounts": [{"Destination": "/data", "Source": str(delta.DATASET), "RW": False}],
        "ports": {"5000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5000"}]},
    }
    return container, provenance


def test_observed_runtime_fields_must_bind_to_pinned_provider():
    container, provenance = witness()
    delta.validate_runtime_witness(container, provenance, "b" * 64)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("running", False, "provider_container_not_running"),
        ("configured_image", "mutable:latest", "running_provider_image_mismatch"),
        ("image_id", "sha256:" + "c" * 64, "running_provider_image_mismatch"),
        ("command", ["osrm-routed", "/data/other.osrm"], "running_provider_dataset_mismatch"),
        (
            "mounts",
            [{"Destination": "/data", "Source": str(delta.DATASET), "RW": True}],
            "running_provider_mount_mismatch",
        ),
        ("ports", {"5000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "5000"}]}, "running_provider_port_mismatch"),
    ],
)
def test_runtime_mismatch_fails_closed(field, value, code):
    container, provenance = witness()
    container[field] = value
    with pytest.raises(ValueError, match=code):
        delta.validate_runtime_witness(container, provenance, "b" * 64)


def test_pbf_mismatch_and_external_provider_fail_closed():
    container, provenance = witness()
    with pytest.raises(ValueError, match="running_provider_pbf_mismatch"):
        delta.validate_runtime_witness(container, provenance, "c" * 64)
    provenance["endpoint"] = "https://external.invalid"
    with pytest.raises(ValueError, match="provider_endpoint_not_approved"):
        delta.validate_runtime_witness(container, provenance, "b" * 64)


def test_single_directed_pair_uses_catalog_road_access_not_display_coordinates():
    base = json.loads(delta.BASE_PATH.read_text(encoding="utf-8"))
    before = deepcopy(base)
    rows, overrides, records = delta.route_inputs(base)
    requests = iter_route_leg_requests(rows, route_point_overrides=overrides)
    assert len(requests) == 1
    request = requests[0]
    assert (request["origin_label"], request["destination_label"]) == delta.PAIR
    assert request["points"] == [(36.252418, -121.787339), (36.517094, -121.941786)]
    assert request["destination_input_latitude"] == 36.5159
    assert request["destination_coordinate_source"] == "configured_routing_anchor"
    assert records["catalog_sha256"] == "20b3f41d4dfa8907c058f5fef1f1d917a46baf34c7a67cd4fc20104aee40bb97"
    assert base == before


def test_access_claim_requires_original_evidence_reference():
    base = json.loads(delta.BASE_PATH.read_text(encoding="utf-8"))
    for cell in base["cells"]:
        cell["query_hash"] = "unrelated"
    with pytest.raises(ValueError, match="delta_access_evidence_mismatch"):
        delta.route_inputs(base)


def test_existing_output_is_rejected_before_runtime_or_provider_calls(tmp_path, monkeypatch):
    def unexpected(*args):
        pytest.fail("immutable output must be rejected before runtime/provider work")

    monkeypatch.setattr(delta, "runtime_witness", unexpected)
    marker = tmp_path / "retained.txt"
    marker.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable_delta_output_already_exists"):
        delta.build_delta(tmp_path, "unused", tmp_path / "not-read.json")
    assert marker.read_text(encoding="utf-8") == "original"


def test_actual_frozen_delta_preserves_every_base_cell_and_enables_exact_addition(tmp_path):
    package = load_product_demo_package(ROOT, ROOT / "runs/california-coast-coherent-demo-v1")
    base = package.primary_bundle.route_matrix
    combined = load_frozen_route_delta(ROOT, FROZEN / "descriptor.json", DESCRIPTOR_HASH, base)
    assert combined["cells"][: len(base["cells"])] == base["cells"]
    assert len(combined["cells"]) == len(base["cells"]) + 1
    compiler = FrozenDraftCompiler(
        package.primary_bundle.parent_plan, package.evidence_bundles, repository_root=ROOT, route_matrix=combined
    )
    original_caps = compiler.capabilities()
    assert original_caps["operations"]["add_candidate"]["preview_executable"] is False
    replacement = compiler.compile(
        [
            {
                "type": "replace_nearby",
                "target": "bixby_creek_bridge_viewpoint",
                "parameters": {"candidate_id": delta.PAIR[0]},
            }
        ],
        accepted_plan_id=compiler.parent_plan_id,
    )
    assert replacement.state == "eligible"
    original = FrozenDraftCompiler(package.primary_bundle.parent_plan, package.evidence_bundles, repository_root=ROOT)
    original_replacement = original.compile(
        [
            {
                "type": "replace_nearby",
                "target": "bixby_creek_bridge_viewpoint",
                "parameters": {"candidate_id": delta.PAIR[0]},
            }
        ],
        accepted_plan_id=original.parent_plan_id,
    )
    assert original_replacement.child_plan["plan_id"] != replacement.child_plan["plan_id"]
    assert original_replacement.source_request_id != replacement.source_request_id
    rebased = compiler.for_parent(replacement.child_plan)
    rows = [rebased.candidate_addition_readiness(delta.PAIR[1], day) for day in range(1, 8)]
    assert [row["day"] for row in rows if row["status"] == "route_evidence_ready"] == [5]
    operation = {"type": "add_candidate", "target": delta.PAIR[1], "parameters": {"day": 5}}
    result = rebased.compile([operation], accepted_plan_id=replacement.child_plan["plan_id"])
    assert result.state == "eligible"
    assert result.child_plan["parent_plan_id"] == replacement.child_plan["plan_id"]
    assert set(result.child_plan["sequence"]) == set(replacement.child_plan["sequence"]) | {delta.PAIR[1]}
    assert not result.diff["deleted_stops"]
    assert result.certificate["route_validation"]["missing_leg_count"] == 0
    assert result.certificate["route_validation"]["source_content_sha256"] == DESCRIPTOR_HASH
    (tmp_path / "actual_delta_compiler_io.json").write_text(
        json.dumps(
            {
                "input": operation,
                "parent": replacement.child_plan,
                "child": result.child_plan,
                "diff": result.diff,
                "certificate": result.certificate,
                "readiness": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "filename",
    [
        "raw_response.json",
        "runtime_witness.json",
        "production_road_route_cache.csv",
        "route_evidence_bundle_manifest.json",
    ],
)
def test_frozen_delta_tampering_blocks_loading(tmp_path, filename):
    copied = tmp_path / "delta"
    shutil.copytree(FROZEN, copied)
    path = copied / filename
    path.write_bytes(path.read_bytes() + b" ")
    base = json.loads(delta.BASE_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        load_frozen_route_delta(ROOT, copied / "descriptor.json", DESCRIPTOR_HASH, base)


def test_base_or_descriptor_hash_mismatch_blocks_loading():
    base = json.loads(delta.BASE_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="descriptor_hash"):
        load_frozen_route_delta(ROOT, FROZEN / "descriptor.json", "wrong", base)
    base["cells"][0]["duration_s"] += 1
    with pytest.raises(ValueError, match="base_hash"):
        load_frozen_route_delta(ROOT, FROZEN / "descriptor.json", DESCRIPTOR_HASH, base)


def test_real_api_accept_pfeiffer_add_point_lobos_accept_restart(tmp_path):
    state_root = tmp_path / "s"
    exchanges = []

    def client():
        return TestClient(
            create_product_app(
                repository_root=ROOT,
                registry_path=ROOT / "configs/product_app_registry.json",
                state_root=state_root,
                additional_allowed_authorities=("testserver",),
                copilot=DeterministicCopilotAdapter(),
            )
        )

    def post(api, path, body, headers=None):
        response = api.post(path, json=body, headers=headers)
        value = response.json()
        exchanges.append(
            {
                "path": path,
                "input": body,
                "status": response.status_code,
                "output": {key: item for key, item in value.items() if key != "mutation_token"},
            }
        )
        (tmp_path / "exact_addition_api_io.json").write_text(json.dumps(exchanges, indent=2), encoding="utf-8")
        assert response.status_code == 200, response.text
        return value

    with client() as api:
        created = post(api, "/api/sessions", {"run_id": "california_coast_coherent_demo_v1"})
        session = created["session"]
        url = f"/api/sessions/{session['session_id']}"
        headers = {"X-Session-Token": created["mutation_token"]}
        post(
            api,
            url + "/draft/operations",
            {
                "expected_revision": 0,
                "type": "replace_nearby",
                "target": "bixby_creek_bridge_viewpoint",
                "parameters": {"candidate_id": delta.PAIR[0]},
            },
            headers,
        )
        replacement = post(api, url + "/preview", {"expected_revision": 1}, headers)
        assert replacement["proposal"]["state"] == "eligible"
        accepted = post(
            api,
            url + "/accept",
            {
                "expected_revision": 2,
                "confirmation": "accept_evaluated_repair",
                "idempotency_key": "r03_accept_pfeiffer",
            },
            headers,
        )
        b = accepted["session"]["accepted_plan_id"]

    with client() as api:
        restored = api.get(url, headers=headers)
        assert restored.status_code == 200, restored.text
        caps = restored.json()["workspace"]["draft_capabilities"]
        rows = caps["operations"]["add_candidate"]["candidate_day_readiness"]
        assert [(row["candidate_id"], row["day"]) for row in rows if row["preview_executable"]] == [(delta.PAIR[1], 5)]
        assert caps["operations"]["add_candidate"]["preview_executable"] is True
        staged = post(
            api,
            url + "/draft/operations",
            {"expected_revision": 3, "type": "add_candidate", "target": delta.PAIR[1], "parameters": {"day": 5}},
            headers,
        )
        assert staged["session"]["accepted_plan_id"] == b
        preview = post(api, url + "/preview", {"expected_revision": 4}, headers)
        proposal = preview["proposal"]
        assert proposal["state"] == "eligible"
        assert proposal["parent_plan_id"] == b
        c = proposal["child_plan_id"]
        marker_ids = {feature["properties"]["stop_id"] for feature in proposal["geography_plan"]["stops"]["features"]}
        assert set(delta.PAIR).issubset(marker_ids)
        assert proposal["geography_plan"]["plan_id"] == c
        assert proposal["geography_plan"]["coverage"]["complete"] is True
        assert proposal["route_validation"]["source_content_sha256"] == DESCRIPTOR_HASH
        accepted = post(
            api,
            url + "/accept",
            {
                "expected_revision": 5,
                "confirmation": "accept_evaluated_repair",
                "idempotency_key": "r03_accept_point_lobos",
            },
            headers,
        )
        assert accepted["session"]["accepted_plan_id"] == c

    with client() as api:
        restored = api.get(url, headers=headers)
        assert restored.status_code == 200, restored.text
        value = restored.json()
        assert value["session"]["accepted_plan_id"] == c
        geography = next(plan for plan in value["workspace"]["geography"]["plans"] if plan["plan_id"] == c)
        assert geography["content_hash"] == proposal["child_plan_content_hash"]
        assert {feature["properties"]["stop_id"] for feature in geography["stops"]["features"]} == marker_ids
        assert not value["workspace"]["draft_capabilities"]["candidate_choices"]
