"""Independently bind retained map cancellations; never import browser classifiers."""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from urllib.parse import urlsplit

import verify_e2e1_vertical_slice as core


def require(condition, reason):
    core._require(condition, "e2e1_network_" + reason)


def verify_classification(document, records, configuration, rendered_style):
    """Only exact local-map GET cancellations can be nonblocking observations."""
    require(isinstance(configuration, dict) and isinstance(rendered_style, dict), "source_invalid")
    config = {key: configuration[key] for key in ("schema_version", "provider", "status", "base_url", "style_url")}
    require(config["schema_version"] == "product-map-configuration-v2"
        and config["provider"] == "maplibre_pmtiles" and config["status"] == "ready", "configuration_invalid")
    origin = urlsplit(config["base_url"])
    require(origin.scheme == "http" and origin.hostname in {"127.0.0.1", "localhost", "::1"}
        and origin.port is not None and origin.username is None and origin.password is None
        and not (origin.path or origin.query or origin.fragment), "origin_invalid")

    def asset(value, directory, suffix):
        require(isinstance(value, str), "asset_invalid")
        parsed = urlsplit(value)
        require(parsed.scheme == origin.scheme and parsed.netloc == origin.netloc
            and parsed.path.startswith(directory) and parsed.path.endswith(suffix)
            and not parsed.query and not parsed.fragment
            and not any(token in parsed.path for token in ("%", "..", "\\")), "asset_invalid")
        return value

    urls = {asset(config["style_url"], "/styles/", ".json")}
    sources = rendered_style.get("sources")
    require(isinstance(sources, dict), "renderer_sources_missing")
    vectors = {}
    for name, source in sources.items():
        require(isinstance(source, dict), "renderer_source_invalid")
        if source.get("type") != "vector":
            continue
        require(isinstance(name, str) and bool(name) and isinstance(source.get("url"), str)
            and source["url"].startswith("pmtiles://"), "vector_source_invalid")
        urls.add(asset(source["url"][len("pmtiles://"):], "/data/", ".pmtiles"))
        vectors[name] = {"type": "vector", "url": source["url"]}
    require(bool(vectors), "archive_missing")
    sprite_base=rendered_style.get("sprite")
    if sprite_base is not None:
        require(isinstance(sprite_base,str),"sprite_invalid")
        parsed=urlsplit(sprite_base)
        require(parsed.scheme==origin.scheme and parsed.netloc==origin.netloc
            and parsed.path.startswith("/sprites/") and not parsed.path.endswith((".json",".png"))
            and not parsed.query and not parsed.fragment
            and not any(token in parsed.path for token in ("%","..","\\")),"sprite_invalid")
        urls.update(sprite_base+suffix for suffix in (".json",".png","@2x.json","@2x.png"))
    policy = document["policy"]
    require(set(policy) == {"schema_version", "source", "map_configuration", "rendered_vector_sources",
            "rendered_sprite_base", "allowlisted_urls", "scope"}
        and policy["schema_version"] == "local-map-cancellation-policy-v1"
        and policy["source"] == "actual_api_map_configuration_and_real_map_public_style_getter"
        and policy["scope"] == "Browser-reported cancellation only; no map-disposal cause is inferred. Readiness and identity gates remain mandatory."
        and policy["map_configuration"] == config and policy["rendered_vector_sources"] == vectors
        and policy["rendered_sprite_base"] == sprite_base
        and policy["allowlisted_urls"] == sorted(urls), "policy_source_unbound")
    require(isinstance(records, list), "records_invalid")
    expected = []
    for index, row in enumerate(records):
        require(isinstance(row, dict) and set(row) == {"event", "url", "method", "resource_type", "phase",
            "observed_at", "range", "failure", "http_status"}, "record_shape_invalid")
        require(all(isinstance(row[key], str) and row[key] for key in ("url", "method", "resource_type", "phase", "observed_at")),
            "record_metadata_missing")
        require(datetime.fromisoformat(row["observed_at"]).utcoffset() is not None, "timestamp_invalid")
        require(row["range"] is None or isinstance(row["range"], str)
            and re.fullmatch(r"bytes=(?:[0-9]+-[0-9]*|-[0-9]+)", row["range"]) is not None, "range_invalid")
        require(row["event"] == "request_failed" and row["failure"] == "net::ERR_ABORTED"
            and row["http_status"] is None and row["method"] == "GET"
            and row["resource_type"] in {"fetch", "xhr"} and row["url"] in urls, "blocking_observation")
        expected.append({"raw_record_index": index, "classification": "browser_cancelled_local_map_asset", "blocking": False})
    require(document["schema_version"] == "browser-network-classification-v1"
        and type(document["raw_record_count"]) is int and document["raw_record_count"] == len(records)
        and isinstance(document["classifications"], list)
        and all(isinstance(row, dict) and type(row.get("raw_record_index")) is int and row.get("blocking") is False
            for row in document["classifications"])
        and isinstance(document["browser_cancelled_local_map_asset_indexes"], list)
        and all(type(index) is int for index in document["browser_cancelled_local_map_asset_indexes"])
        and document["classifications"] == expected and document["blocking_failure_indexes"] == []
        and document["browser_cancelled_local_map_asset_indexes"] == list(range(len(records))), "classification_unbound")
    return {"verified": True, "raw_observations": len(records), "browser_cancelled_local_map_assets": len(records),
        "blocking_network_failures": 0, "map_disposal_cause_claimed": False}


def verify_asset_observation(observation, parent, context, *, phase="capability_inventory"):
    require(observation["schema_version"] == "real-map-asset-sources-v1"
        and observation["context"] == context and observation["phase"] == phase
        and observation["container_id"] == "geographic-map", "asset_observation_invalid")
    dataset = observation["map_dataset"]
    coverage = float(dataset["mapTileCoverage"])
    require(dataset["mapReady"] == "true" and math.isfinite(coverage) and .95 <= coverage <= 1
        and dataset["mapCoverageMethod"] == "loaded_tile_projected_union_area_v1"
        and datetime.fromisoformat(dataset["mapReadyAt"].replace("Z", "+00:00")).utcoffset() is not None
        and dataset["renderedMapPlanId"] == parent["active_plan_id"] == parent["rendered_map_plan_id"]
        and dataset["renderedPlanContentHash"] == parent["active_plan_content_hash"]
        and dataset["renderedGeometryHash"] == parent["rendered_geometry_hash"]
        and json.loads(dataset["renderedMarkerIds"]) == parent["rendered_marker_ids"], "asset_map_identity_unbound")


def verify(base, base_url, records, rendered_style, *, asset_observation=None):
    """The caller already verifies full HTTP inventory and all screenshot/map gates."""
    try:
        document = core._load(base / "network_classification.json")
        name = document["map_configuration_http_file"]
        require(isinstance(name, str) and re.fullmatch(r"http_[0-9]+\.json", name) is not None, "configuration_file_invalid")
        receipt = core._load(core._safe_path(base, name))
        core._verify_product_http_capture(receipt)
        require(receipt["method"] == "GET" and receipt["url"] == base_url + "/api/map/config"
            and receipt["response_status_code"] == 200, "configuration_http_unbound")
        if asset_observation is not None:
            require(document.get("rendered_map_asset_sources_file") == "map_asset_sources.json", "asset_file_unbound")
            source_name = asset_observation["map_configuration_http_file"]
            source_receipt = core._load(core._safe_path(base, source_name))
            core._verify_product_http_capture(source_receipt)
            require(source_receipt["method"] == "GET" and source_receipt["url"] == receipt["url"]
                and source_receipt["response_status_code"] == 200
                and json.loads(source_receipt["response_body_utf8"]) == json.loads(receipt["response_body_utf8"]), "asset_configuration_changed")
        return verify_classification(document, records, json.loads(receipt["response_body_utf8"]), rendered_style)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        if isinstance(error, core.E2E1VerificationError):
            raise
        raise core.E2E1VerificationError("e2e1_network_evidence_malformed") from error
