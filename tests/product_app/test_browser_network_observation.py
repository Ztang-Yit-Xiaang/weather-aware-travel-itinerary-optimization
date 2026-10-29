"""Exact configured map cancellations remain visible without excusing failures."""

from __future__ import annotations

import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent / "browser"))
from waypoint_ui_fixture import (  # noqa: E402
    browser_network_record,
    classify_browser_network,
    map_cancellation_policy,
)

BASE = "http://127.0.0.1:8080"
STYLE = BASE + "/styles/protomaps-light.json"
ARCHIVE = BASE + "/data/california-coast-v1.pmtiles"
SPRITE = BASE + "/sprites/light"


def inputs():
    return {
        "schema_version": "product-map-configuration-v2",
        "status": "ready",
        "provider": "maplibre_pmtiles",
        "base_url": BASE,
        "style_url": STYLE,
    }, {
        "sprite": SPRITE,
        "sources": {
            "protomaps": {"type": "vector", "url": "pmtiles://" + ARCHIVE},
            "v2-draft-routes": {"type": "geojson", "data": {"type": "FeatureCollection", "features": []}},
        },
    }


def record(**changes):
    row = {
        "event": "request_failed",
        "url": ARCHIVE,
        "method": "GET",
        "resource_type": "fetch",
        "phase": "select_waypoint_and_drag",
        "observed_at": "2026-09-08T20:00:01.234+00:00",
        "range": "bytes=0-16383",
        "failure": "net::ERR_ABORTED",
        "http_status": None,
    }
    return {**row, **changes}


def test_policy_uses_observed_urls_not_a_hardcoded_archive_name():
    config, style = inputs()
    config["style_url"] = BASE + "/styles/other-pinned.json"
    style["sources"]["protomaps"]["url"] = "pmtiles://" + BASE + "/data/other-pinned.pmtiles"
    policy = map_cancellation_policy(config, style)
    assert policy["allowlisted_urls"] == sorted(
        [
            BASE + "/data/other-pinned.pmtiles",
            BASE + "/styles/other-pinned.json",
            SPRITE + ".json",
            SPRITE + ".png",
            SPRITE + "@2x.json",
            SPRITE + "@2x.png",
        ]
    )
    assert classify_browser_network([record()], policy)["blocking_failure_indexes"] == [0]


@pytest.mark.parametrize(
    "url,resource_type,byte_range",
    [
        (ARCHIVE, "fetch", "bytes=0-16383"),
        (STYLE, "xhr", None),
        (SPRITE + ".json", "fetch", None),
        (SPRITE + ".png", "fetch", None),
    ],
)
def test_exact_local_asset_cancellation_is_retained_and_nonblocking(url, resource_type, byte_range):
    raw = [record(url=url, resource_type=resource_type, range=byte_range)]
    original = deepcopy(raw)
    result = classify_browser_network(raw, map_cancellation_policy(*inputs()))
    assert raw == original
    assert result["raw_record_count"] == 1 and result["browser_cancelled_local_map_asset_indexes"] == [0]
    assert result["blocking_failure_indexes"] == []
    assert result["classifications"] == [
        {"raw_record_index": 0, "classification": "browser_cancelled_local_map_asset", "blocking": False}
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"url": "http://127.0.0.1:8127/api/sessions/demo/preview"},
        {"url": BASE + "/api/map/config"},
        {"url": ARCHIVE.replace("127.0.0.1", "localhost")},
        {"url": ARCHIVE.replace(":8080", ":8081")},
        {"url": ARCHIVE + "?ignored=1"},
        {"url": ARCHIVE + "#ignored"},
        {"url": ARCHIVE.replace("california-coast-v1", "another")},
        {"url": "https://example.com/data/california-coast-v1.pmtiles"},
        {"method": "POST"},
        {"method": "get"},
        {"resource_type": "document"},
        {"resource_type": "script"},
        {"failure": "net::ERR_CONNECTION_REFUSED"},
        {"failure": "net::ERR_FAILED"},
        {"event": "http_error", "failure": None, "http_status": 404},
        {"event": "http_error", "failure": None, "http_status": 503},
    ],
)
def test_cancelled_api_unknown_asset_transport_and_http_errors_still_block(changes):
    result = classify_browser_network([record(**changes)], map_cancellation_policy(*inputs()))
    assert result["blocking_failure_indexes"] == [0]
    assert result["browser_cancelled_local_map_asset_indexes"] == []


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        "net::ERR_ABORTED",
        {},
        record(event=[]),
        record(method=None),
        record(url=None),
        record(phase=""),
        record(observed_at="not-a-time"),
        record(observed_at="2026-09-08T20:00:01"),
        record(range={}),
        record(range="bytes=oops"),
        record(http_status=500),
        record(failure={}),
        record(extra=True),
        {key: value for key, value in record().items() if key != "method"},
        {key: value for key, value in record().items() if key != "range"},
    ],
)
def test_malformed_raw_observation_is_retained_as_blocking(bad):
    raw = [bad]
    original = deepcopy(raw)
    result = classify_browser_network(raw, map_cancellation_policy(*inputs()))
    assert raw == original
    assert result["blocking_failure_indexes"] == [0]
    assert result["classifications"][0]["classification"] == "malformed_network_record"


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_host",
        "credentialed_origin",
        "unknown_provider",
        "not_ready",
        "no_sources",
        "no_archive",
        "nonlocal_archive",
        "api_style",
        "escaped_archive",
        "archive_query",
        "sprite_api",
        "sprite_query",
    ],
)
def test_unbound_or_malformed_configuration_cannot_expand_allowlist(mutation):
    config, style = inputs()
    if mutation == "wrong_host":
        config["base_url"] = "http://example.com:8080"
    elif mutation == "credentialed_origin":
        config["base_url"] = "http://user@127.0.0.1:8080"
    elif mutation == "unknown_provider":
        config["provider"] = "unknown"
    elif mutation == "not_ready":
        config["status"] = "degraded"
    elif mutation == "no_sources":
        style = {}
    elif mutation == "no_archive":
        style["sources"] = {}
    elif mutation == "nonlocal_archive":
        style["sources"]["protomaps"]["url"] = "pmtiles://http://example.com/data/a.pmtiles"
    elif mutation == "api_style":
        config["style_url"] = BASE + "/api/map/config.json"
    elif mutation == "escaped_archive":
        style["sources"]["protomaps"]["url"] = "pmtiles://" + BASE + "/data/%2e%2e/a.pmtiles"
    elif mutation == "archive_query":
        style["sources"]["protomaps"]["url"] = "pmtiles://" + ARCHIVE + "?x=1"
    elif mutation == "sprite_api":
        style["sprite"] = BASE + "/api/sprites/light"
    elif mutation == "sprite_query":
        style["sprite"] = SPRITE + "?x=1"
    with pytest.raises(ValueError):
        map_cancellation_policy(config, style)


def test_classifier_rederives_policy_and_rejects_allowlist_tampering():
    policy = map_cancellation_policy(*inputs())
    policy["allowlisted_urls"].append(BASE + "/api/map/config")
    with pytest.raises(ValueError, match="policy_mismatch"):
        classify_browser_network([], policy)


def test_mixed_records_keep_indexes_and_http_error_does_not_become_cancellation():
    rows = [record(), record(url=BASE + "/api/map/config"), record(event="http_error", failure=None, http_status=500)]
    result = classify_browser_network(rows, map_cancellation_policy(*inputs()))
    assert result["browser_cancelled_local_map_asset_indexes"] == [0]
    assert result["blocking_failure_indexes"] == [1, 2]
    assert [row["classification"] for row in result["classifications"]] == [
        "browser_cancelled_local_map_asset",
        "browser_cancelled_other_request",
        "http_error",
    ]


def test_raw_recorder_keeps_failure_metadata_and_http_status_distinct():
    request = SimpleNamespace(
        url=ARCHIVE, method="GET", resource_type="fetch", headers={"range": "bytes=3-9"}, failure="net::ERR_ABORTED"
    )
    observed = browser_network_record(request, "real_drag")
    assert observed["range"] == "bytes=3-9" and observed["phase"] == "real_drag"
    assert observed["failure"] == "net::ERR_ABORTED" and observed["http_status"] is None
    assert datetime.fromisoformat(observed["observed_at"]).utcoffset() is not None
    request.failure = None
    failed_http = browser_network_record(request, "real_drag", http_status=503)
    assert failed_http["event"] == "http_error" and failed_http["failure"] is None and failed_http["http_status"] == 503
