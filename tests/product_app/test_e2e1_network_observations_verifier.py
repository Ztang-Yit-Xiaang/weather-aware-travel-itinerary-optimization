"""Independent synthetic counterexamples; not browser cancellation evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_network_observations as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402

BASE = "http://127.0.0.1:8127"
MAP = "http://127.0.0.1:8080"
STYLE = MAP + "/styles/pinned.json"
ARCHIVE = MAP + "/data/pinned.pmtiles"
SPRITE = MAP + "/sprites/pinned"


def inputs():
    configuration = {
        "schema_version": "product-map-configuration-v2",
        "provider": "maplibre_pmtiles",
        "status": "ready",
        "base_url": MAP,
        "style_url": STYLE,
    }
    sources = {"base": {"type": "vector", "url": "pmtiles://" + ARCHIVE}}
    rendered_style = {"sources": sources, "sprite": SPRITE}
    record = {
        "event": "request_failed",
        "url": ARCHIVE,
        "method": "GET",
        "resource_type": "fetch",
        "phase": "real_drag",
        "observed_at": "2026-09-08T16:00:00+00:00",
        "range": "bytes=0-16383",
        "failure": "net::ERR_ABORTED",
        "http_status": None,
    }
    policy = {
        "schema_version": "local-map-cancellation-policy-v1",
        "source": "actual_api_map_configuration_and_real_map_public_style_getter",
        "map_configuration": deepcopy(configuration),
        "rendered_vector_sources": deepcopy(sources),
        "rendered_sprite_base": SPRITE,
        "allowlisted_urls": sorted(
            [STYLE, ARCHIVE, SPRITE + ".json", SPRITE + ".png", SPRITE + "@2x.json", SPRITE + "@2x.png"]
        ),
        "scope": "Browser-reported cancellation only; no map-disposal cause is inferred. Readiness and identity gates remain mandatory.",
    }
    document = {
        "map_configuration_http_file": "http_001.json",
        "schema_version": "browser-network-classification-v1",
        "policy": policy,
        "raw_record_count": 1,
        "classifications": [
            {"raw_record_index": 0, "classification": "browser_cancelled_local_map_asset", "blocking": False}
        ],
        "browser_cancelled_local_map_asset_indexes": [0],
        "blocking_failure_indexes": [],
    }
    return document, [record], configuration, rendered_style


def receipt(configuration):
    row = {
        "schema_version": "e2e1-product-http-io-v1",
        "url": BASE + "/api/map/config",
        "method": "GET",
        "response_status_code": 200,
        "request_headers_persisted": False,
        "response_headers_persisted": False,
        "authorization_persisted": False,
        "session_token_persisted": False,
        "credential_redactions": {"request": [], "response": []},
        "body_capture": "exact_utf8",
    }
    for side, body in (("request", ""), ("response", json.dumps(configuration))):
        row[side + "_body_utf8"] = body
        row[side + "_body_sha256"] = row[side + "_original_body_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return row


def run(tmp_path, document, records, configuration, rendered_style, captured=None):
    (tmp_path / "network_classification.json").write_text(json.dumps(document), encoding="utf-8")
    (tmp_path / "http_001.json").write_text(json.dumps(captured or receipt(configuration)), encoding="utf-8")
    return verifier.verify(tmp_path, BASE, records, rendered_style)


@pytest.mark.parametrize(
    "url,resource", [(STYLE, "fetch"), (ARCHIVE, "xhr"), (SPRITE + ".json", "fetch"), (SPRITE + ".png", "fetch")]
)
def test_exact_observed_map_cancellation_is_retained_and_bound(tmp_path, url, resource):
    document, records, configuration, sources = inputs()
    records[0].update(url=url, resource_type=resource)
    before = deepcopy(records)
    result = run(tmp_path, document, records, configuration, sources)
    assert result == {
        "verified": True,
        "raw_observations": 1,
        "browser_cancelled_local_map_assets": 1,
        "blocking_network_failures": 0,
        "map_disposal_cause_claimed": False,
    }
    assert records == before


def test_zero_events_still_require_real_source_binding(tmp_path):
    document, _, configuration, sources = inputs()
    document.update(raw_record_count=0, classifications=[], browser_cancelled_local_map_asset_indexes=[])
    assert run(tmp_path, document, [], configuration, sources)["raw_observations"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"url": BASE + "/api/sessions"},
        {"url": ARCHIVE + "?x=1"},
        {"url": ARCHIVE.replace(":8080", ":8081")},
        {"url": MAP + "/data/other.pmtiles"},
        {"url": "https://example.com/data/pinned.pmtiles"},
        {"method": "POST"},
        {"resource_type": "document"},
        {"failure": "net::ERR_CONNECTION_REFUSED"},
        {"event": "http_error", "http_status": 503, "failure": None},
        {"http_status": 404},
        {"range": "bytes=invalid"},
        {"phase": ""},
        {"observed_at": "2026-09-08T16:00:00"},
        {"observed_at": "invalid"},
        {"failure": None},
        {"method": None},
        {"extra_unverified": True},
    ],
)
def test_other_failures_or_incomplete_records_cannot_be_excused(tmp_path, change):
    document, records, configuration, sources = inputs()
    records[0].update(change)
    with pytest.raises(core.E2E1VerificationError):
        run(tmp_path, document, records, configuration, sources)


@pytest.mark.parametrize(
    "fault",
    [
        "allowlist",
        "policy_config",
        "policy_sources",
        "scope",
        "count",
        "indexes",
        "duplicate",
        "omitted",
        "classification",
        "blocking",
        "boolean_index",
        "boolean_cancelled_index",
        "numeric_blocking",
        "foreign_file",
        "wrong_endpoint",
        "http_status",
        "http_hash",
        "source_mismatch",
    ],
)
def test_forged_classification_or_receipt_binding_is_rejected(tmp_path, fault):
    document, records, configuration, sources = inputs()
    captured = receipt(configuration)
    if fault == "allowlist":
        document["policy"]["allowlisted_urls"].append(BASE + "/api/health")
    elif fault == "policy_config":
        document["policy"]["map_configuration"]["style_url"] = MAP + "/styles/other.json"
    elif fault == "policy_sources":
        document["policy"]["rendered_vector_sources"]["base"]["url"] = "pmtiles://" + MAP + "/data/other.pmtiles"
    elif fault == "scope":
        document["policy"]["scope"] = "All failures ignored"
    elif fault == "count":
        document["raw_record_count"] = 0
    elif fault == "indexes":
        document["browser_cancelled_local_map_asset_indexes"] = []
    elif fault == "duplicate":
        document["classifications"] *= 2
    elif fault == "omitted":
        document["classifications"] = []
    elif fault == "classification":
        document["classifications"][0]["classification"] = "success"
    elif fault == "blocking":
        document["blocking_failure_indexes"] = [0]
    elif fault == "boolean_index":
        document["classifications"][0]["raw_record_index"] = False
    elif fault == "boolean_cancelled_index":
        document["browser_cancelled_local_map_asset_indexes"] = [False]
    elif fault == "numeric_blocking":
        document["classifications"][0]["blocking"] = 0
    elif fault == "foreign_file":
        document["map_configuration_http_file"] = "../http_001.json"
    elif fault == "wrong_endpoint":
        captured["url"] = BASE + "/api/health"
    elif fault == "http_status":
        captured["response_status_code"] = 500
    elif fault == "http_hash":
        captured["response_body_sha256"] = "0" * 64
    elif fault == "source_mismatch":
        sources["sources"]["base"]["url"] = "pmtiles://" + MAP + "/data/other.pmtiles"
    with pytest.raises(core.E2E1VerificationError):
        run(tmp_path, document, records, configuration, sources, captured)


def asset_inputs():
    parent = {
        "active_plan_id": "parent",
        "active_plan_content_hash": "parent-hash",
        "rendered_map_plan_id": "parent",
        "rendered_geometry_hash": "g" * 64,
        "rendered_marker_ids": ["stop"],
    }
    observed = {
        "schema_version": "real-map-asset-sources-v1",
        "context": "original",
        "phase": "capability_inventory",
        "container_id": "geographic-map",
        "map_dataset": {
            "mapReady": "true",
            "mapTileCoverage": "1",
            "mapCoverageMethod": "loaded_tile_projected_union_area_v1",
            "mapReadyAt": "2026-09-08T16:00:00Z",
            "renderedMapPlanId": "parent",
            "renderedPlanContentHash": "parent-hash",
            "renderedGeometryHash": "g" * 64,
            "renderedMarkerIds": '["stop"]',
        },
    }
    return observed, parent


def test_renderer_source_observation_is_bound_to_ready_canonical_parent():
    verifier.verify_asset_observation(*asset_inputs(), "original")


@pytest.mark.parametrize(
    "field,value",
    [
        ("mapReady", "false"),
        ("mapTileCoverage", "0.94"),
        ("mapTileCoverage", "nan"),
        ("mapCoverageMethod", "source_count"),
        ("renderedMapPlanId", "other"),
        ("renderedPlanContentHash", "other"),
        ("renderedGeometryHash", "other"),
        ("renderedMarkerIds", "[]"),
    ],
)
def test_asset_source_cannot_excuse_an_unready_or_different_map(field, value):
    observed, parent = asset_inputs()
    observed["map_dataset"][field] = value
    with pytest.raises(core.E2E1VerificationError):
        verifier.verify_asset_observation(observed, parent, "original")
