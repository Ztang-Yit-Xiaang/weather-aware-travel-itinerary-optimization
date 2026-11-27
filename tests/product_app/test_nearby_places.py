import json
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import pytest

from itinerary_system.product_app import nearby_places as nearby


@pytest.fixture
def lookup(tmp_path, monkeypatch):
    clock = [1800000000.0]
    monkeypatch.setattr(nearby, "time", SimpleNamespace(time=lambda: clock[0]))
    search = {"coordinate": {"longitude": -122.1697, "latitude": 37.4275}, "radius_km": 2}
    payload = {
        "elements": [
            {
                "type": "node",
                "id": 12,
                "lat": 37.428,
                "lon": -122.169,
                "tags": {
                    "name": "Campus Museum",
                    "tourism": "museum",
                    "website": "https://example.org/museum",
                    "opening_hours": "Tu-Su 10:00-17:00",
                },
            }
        ]
    }
    calls = []

    def fetch(query):
        calls.append(query)
        return deepcopy(payload)

    monkeypatch.setattr(nearby, "_fetch", fetch)
    return nearby.NearbyPlaceCache(tmp_path), search, payload, calls, clock


def test_online_data_is_reused_after_restart_and_preferences_do_not_change_cache_key(lookup, monkeypatch):
    cache, search, _, calls, _ = lookup
    points, data = cache.search(search)
    assert data["status"] == "live" and data["cache_persisted"]
    assert points[0]["source_refs"] == ["https://www.openstreetmap.org/node/12", "https://example.org/museum"]
    assert "not independently verified" in points[0]["description"]
    assert "around:2000,37.427500,-122.169700" in calls[0]
    monkeypatch.setattr(nearby, "_fetch", lambda _: pytest.fail("Fresh saved data must not contact the API"))
    reopened, source = nearby.NearbyPlaceCache(cache.root).search({**search, "nature_preference": 1})
    assert reopened == points and source["status"] == "fresh_cache"
    assert source["fetched_at"] == data["fetched_at"]


def test_hosted_catalog_mode_never_fetches_even_on_refresh(lookup, monkeypatch):
    cache, search, _, calls, _ = lookup
    hosted = nearby.NearbyPlaceCache(cache.root, online_enabled=False)
    points, data = hosted.search(search, refresh=True)
    assert not points and data["status"] == "bundled_fallback"
    assert data["error_code"] == "nearby_online_disabled_for_hosting"
    assert not calls
    original, _ = cache.search(search)
    monkeypatch.setattr(nearby, "_fetch", lambda _: pytest.fail("Hosted refresh must stay offline"))
    points, data = hosted.search(search, refresh=True)
    assert points == original and data["status"] == "fresh_cache"
    assert data["error_code"] == "nearby_online_disabled_for_hosting"


def test_stale_cache_survives_timeout_and_failure_backoff_survives_restart(lookup, monkeypatch):
    cache, search, _, _, clock = lookup
    original, _ = cache.search(search)
    path = next(p for p in cache.root.glob("*.json") if p.name != "provider.json")
    before = path.read_bytes()
    clock[0] += nearby.FRESH_SECONDS + 1
    calls = []

    def offline(query):
        calls.append(query)
        raise TimeoutError("offline")

    monkeypatch.setattr(nearby, "_fetch", offline)
    points, data = nearby.NearbyPlaceCache(cache.root).search(search)
    assert points == original and data["status"] == "stale_cache"
    assert data["error_code"] == "nearby_api_unavailable" and data["retry_after"]
    assert path.read_bytes() == before
    nearby.NearbyPlaceCache(cache.root).search(search, refresh=True)
    assert len(calls) == 1


def test_cold_offline_search_returns_bundled_fallback_instead_of_a_false_empty_online_result(lookup, monkeypatch):
    cache, search, _, _, _ = lookup
    monkeypatch.setattr(nearby, "_fetch", lambda _: (_ for _ in ()).throw(OSError("offline")))
    points, data = cache.search(search)
    assert points == [] and data["status"] == "bundled_fallback"
    assert data["fetched_at"] is None and data["cache_persisted"] is False


def test_successful_empty_response_is_cached_and_is_distinct_from_outage(lookup):
    cache, search, payload, calls, _ = lookup
    payload["elements"] = []
    assert cache.search(search)[1]["status"] == "live"
    points, data = cache.search(search)
    assert not points and data["status"] == "fresh_cache" and len(calls) == 1


def test_explicit_refresh_replaces_old_results_after_the_request_interval(lookup):
    cache, search, payload, calls, clock = lookup
    cache.search(search)
    clock[0] += nearby.MIN_INTERVAL_SECONDS + 1
    payload["elements"][0]["tags"]["name"] = "Updated Museum"
    points, data = cache.search(search, refresh=True)
    assert points[0]["name"] == "Updated Museum" and data["status"] == "live"
    assert len(calls) == 2


def test_location_and_radius_cache_entries_do_not_leak_into_other_searches(lookup):
    cache, search, _, calls, clock = lookup
    cache.search(search)
    other = {"coordinate": {"longitude": -119.58445, "latitude": 37.746508}, "radius_km": 2}
    points, data = cache.search(other)
    assert not points and data["status"] == "bundled_fallback"
    clock[0] += 11
    cache.search({**search, "radius_km": 1})
    assert len(calls) == 2 and "around:1000," in calls[-1]


@pytest.mark.parametrize(
    "corrupt",
    [
        {"remark": "runtime error: Query timed out", "elements": []},
        {"elements": "invalid"},
        {"elements": [{"type": "way", "id": 1, "center": None, "tags": {"name": "bad"}}]},
        {"elements": [{"type": "node", "id": 1, "lat": True, "lon": -122, "tags": {"name": "bad"}}]},
        {"elements": [{}] * (nearby.MAX_PLACES + 1)},
    ],
)
def test_incomplete_or_invalid_api_data_never_overwrites_the_last_success(lookup, monkeypatch, corrupt):
    cache, search, _, _, clock = lookup
    before, _ = cache.search(search)
    clock[0] += 11
    monkeypatch.setattr(nearby, "_fetch", lambda _: corrupt)
    after, data = cache.search(search, refresh=True)
    assert after == before and data["error_code"]
    assert data["status"] == "fresh_cache"


def test_429_retry_after_is_observed_without_automatic_retry(lookup, monkeypatch):
    cache, search, _, _, clock = lookup
    calls = []

    def throttled(query):
        calls.append(query)
        raise urllib.error.HTTPError(nearby.ENDPOINT, 429, "rate limited", {"Retry-After": "900"}, None)

    monkeypatch.setattr(nearby, "_fetch", throttled)
    _, data = cache.search(search)
    assert data["error_code"] == "nearby_api_rate_limited"
    clock[0] += 899
    nearby.NearbyPlaceCache(cache.root).search(search, refresh=True)
    assert len(calls) == 1
    clock[0] += 2
    cache.search(search)
    assert len(calls) == 2


def test_concurrent_queries_use_one_public_api_request_and_reuse_its_result(lookup, monkeypatch):
    cache, search, payload, _, _ = lookup
    started, release = threading.Event(), threading.Event()

    def slow(_):
        started.set()
        assert release.wait(5)
        return payload

    monkeypatch.setattr(nearby, "_fetch", slow)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(cache.search, search)
        assert started.wait(3)
        _, second = nearby.NearbyPlaceCache(cache.root).search(search)
        assert second["error_code"] == "nearby_refresh_in_progress"
        release.set()
        assert first.result(timeout=3)[1]["status"] == "live"
    assert cache.search(search)[1]["status"] == "fresh_cache"


def test_corrupt_cache_and_gate_are_repaired_without_becoming_place_data(lookup):
    cache, search, _, calls, clock = lookup
    cache.search(search)
    path = next(p for p in cache.root.glob("*.json") if p.name != "provider.json")
    path.write_text("{bad", encoding="utf-8")
    (cache.root / "provider.json").write_text("[]", encoding="utf-8")
    clock[0] += 11
    assert cache.search(search)[1]["status"] == "live"
    assert len(calls) == 2


def test_a_failed_cache_write_does_not_claim_persistence_or_destroy_old_data(lookup, monkeypatch):
    cache, search, payload, _, clock = lookup
    original, _ = cache.search(search)
    clock[0] += 11
    payload["elements"][0]["tags"]["name"] = "Unsaved new name"
    write = nearby.LocalStateLayout._atomic_json_write

    def save(path, record):
        if path.name != "provider.json":
            raise OSError("disk unavailable")
        write(path, record)

    monkeypatch.setattr(nearby.LocalStateLayout, "_atomic_json_write", staticmethod(save))
    points, data = cache.search(search, refresh=True)
    assert points[0]["name"] == "Unsaved new name"
    assert data["cache_persisted"] is False and data["error_code"] == "nearby_cache_write_failed"
    assert cache.search(search)[0] == original


def test_transport_sends_one_bounded_query_and_rejects_an_oversized_body(monkeypatch, offline_nearby_transport):
    seen = []

    def opened(request, timeout):
        seen.append((request.full_url, request.data, timeout))
        return BytesIO(json.dumps({"elements": []}).encode())

    monkeypatch.setattr(nearby.urllib.request, "urlopen", opened)
    assert offline_nearby_transport("[out:json];node(1);out;") == {"elements": []}
    assert len(seen) == 1 and seen[0][0].startswith(nearby.ENDPOINT + "?data=") and seen[0][2] == 8
    assert seen[0][1] is None
    monkeypatch.setattr(nearby.urllib.request, "urlopen", lambda *_, **__: BytesIO(b"x" * (nearby.MAX_BYTES + 1)))
    with pytest.raises(ValueError, match="nearby_response_too_large"):
        offline_nearby_transport("query")
