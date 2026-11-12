"""Bounded online place discovery with durable, source-labelled offline fallback."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

from .persistence import LocalStateLayout, StateFileLock, StateLockTimeoutError

ENDPOINT = "https://overpass-api.de/api/interpreter"
FRESH_SECONDS = 7 * 86400
MAX_BYTES = 4_000_000
MAX_PLACES = 400  # Leave room for official catalog entries and exact route candidates.
MIN_INTERVAL_SECONDS = 10
FAILURE_BACKOFF_SECONDS = 300


def _read(path: Path):
    try:
        if path.is_symlink() or path.stat().st_size > MAX_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _iso(timestamp):
    return datetime.fromtimestamp(timestamp, UTC).isoformat() if timestamp is not None else None


def _query(search):
    longitude, latitude = (float(search["coordinate"][key]) for key in ("longitude", "latitude"))
    radius = float(search["radius_km"])
    if not all(math.isfinite(value) for value in (longitude, latitude, radius)) or not (
        -180 <= longitude <= 180 and -90 <= latitude <= 90 and 1 <= radius <= 25
    ):
        raise ValueError("nearby_search_invalid")
    meters = math.ceil(radius * 1000)
    around = f"(around:{meters},{latitude:.6f},{longitude:.6f})"
    return (
        f"[out:json][timeout:6][maxsize:16777216];("
        f'nwr{around}["tourism"~"^(attraction|museum|viewpoint|gallery|artwork|zoo|aquarium|theme_park|picnic_site)$"]["name"];'
        f'nwr{around}["leisure"~"^(park|garden|nature_reserve)$"]["name"];'
        f'nwr{around}["natural"~"^(beach|peak|waterfall)$"]["name"];'
        f'nwr{around}["historic"~"^(monument|memorial|ruins)$"]["name"];'
        f");out center tags {MAX_PLACES + 1};"
    )


def _fetch(query):
    request = urllib.request.Request(
        ENDPOINT + "?" + urllib.parse.urlencode({"data": query}),
        headers={
            "User-Agent": "ItineraryRepairCopilot/0.2 (cached local research prototype)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        payload = response.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("nearby_response_too_large")
    return json.loads(payload)


def _places(payload):
    elements = payload.get("elements") if isinstance(payload, dict) else None
    if not isinstance(elements, list) or payload.get("remark"):
        raise ValueError("nearby_response_invalid")
    if len(elements) > MAX_PLACES:
        raise ValueError("nearby_response_limit_exceeded")
    places = {}
    for item in elements:
        if not isinstance(item, dict) or item.get("type") not in {"node", "way", "relation"}:
            raise ValueError("nearby_response_invalid")
        identity, tags = item.get("id"), item.get("tags")
        if type(identity) is not int or identity <= 0 or not isinstance(tags, dict):
            raise ValueError("nearby_response_invalid")
        name = tags.get("name:en") or tags.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 500:
            raise ValueError("nearby_response_invalid")
        point = item if item["type"] == "node" else item.get("center", {})
        if not isinstance(point, dict):
            raise ValueError("nearby_response_invalid")
        latitude, longitude = point.get("lat"), point.get("lon")
        if any(type(value) not in {int, float} or not math.isfinite(value) for value in (latitude, longitude)):
            raise ValueError("nearby_response_invalid")
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("nearby_response_invalid")
        if tags.get("access") in {"private", "no"} or tags.get("tourism") == "artwork" and tags.get("indoor") == "yes":
            continue
        kind = tags.get("tourism")
        categories = (
            ["culture"] if kind in {"museum", "gallery", "artwork"} or tags.get("historic") else ["nature", "outdoor"]
        )
        source = f"https://www.openstreetmap.org/{item['type']}/{identity}"
        sources = [source]
        website = tags.get("website")
        if isinstance(website, str) and len(website) < 2048:
            url = urllib.parse.urlparse(website)
            if url.scheme == "https" and url.hostname and not url.username and not url.password:
                sources.append(website)
        description = tags.get("description")
        hours = tags.get("opening_hours")
        details = [description[:1000]] if isinstance(description, str) else []
        if isinstance(hours, str):
            details.append(f"Mapped opening hours: {hours[:300]} (not independently verified).")
        place_id = f"osm_{item['type']}_{identity}"
        places[place_id] = {
            "place_id": place_id,
            "name": name.strip(),
            "display_coordinate": {"longitude": longitude, "latitude": latitude},
            "place_categories": categories,
            "source_refs": sources,
            "source_attribution": "OpenStreetMap contributors",
            "description": " ".join(details)
            or "Community-mapped place. Current opening conditions and route access have not been verified.",
        }
    return list(places.values())


class NearbyPlaceCache:
    def __init__(self, root: Path, *, online_enabled: bool = True):
        self.root = Path(root)
        self.online_enabled = online_enabled

    def search(self, search: dict, *, refresh: bool = False) -> tuple[list[dict], dict]:
        query = _query(search)
        path = self.root / (hashlib.sha256(query.encode()).hexdigest() + ".json")
        cached = self._cached(path, query)
        if not self.online_enabled:
            return self._fallback(cached, "nearby_online_disabled_for_hosting")
        now = time.time()
        if cached and not refresh and now - cached["fetched_at"] <= FRESH_SECONDS:
            return self._result(cached, "fresh_cache")
        try:
            # A competing request can immediately use saved data instead of waiting on the network.
            with StateFileLock(self.root / "provider.lock", timeout_seconds=0.05):
                cached = self._cached(path, query)
                now = time.time()
                if cached and not refresh and now - cached["fetched_at"] <= FRESH_SECONDS:
                    return self._result(cached, "fresh_cache")
                gate_path = self.root / "provider.json"
                gate = _read(gate_path)
                gate = gate if isinstance(gate, dict) else {}
                next_attempt = gate.get("next_attempt_at", 0)
                if type(next_attempt) not in {int, float} or not math.isfinite(next_attempt):
                    next_attempt = 0
                if now < next_attempt <= now + 86400:
                    return self._fallback(cached, gate.get("error_code") or "nearby_refresh_cooldown", next_attempt)
                LocalStateLayout._atomic_json_write(gate_path, {"next_attempt_at": now + MIN_INTERVAL_SECONDS})
                try:
                    payload = _fetch(query)
                    _places(payload)  # Reject incomplete/invalid results before replacing the last good cache.
                except (OSError, ValueError, TypeError, http.client.HTTPException) as error:
                    code = (
                        str(error)
                        if isinstance(error, ValueError) and str(error).startswith("nearby_")
                        else "nearby_api_unavailable"
                    )
                    backoff = FAILURE_BACKOFF_SECONDS
                    if code == "nearby_response_limit_exceeded":
                        backoff = MIN_INTERVAL_SECONDS
                    if isinstance(error, urllib.error.HTTPError) and error.code in {429, 503}:
                        code = "nearby_api_rate_limited"
                        retry = (error.headers or {}).get("Retry-After", "")
                        try:
                            backoff = (
                                float(retry) if retry.isdecimal() else parsedate_to_datetime(retry).timestamp() - now
                            )
                        except (ValueError, TypeError, AttributeError, OverflowError):
                            pass
                        backoff = min(3600, max(FAILURE_BACKOFF_SECONDS, backoff))
                    retry_at = time.time() + backoff
                    LocalStateLayout._atomic_json_write(gate_path, {"next_attempt_at": retry_at, "error_code": code})
                    return self._fallback(cached, code, retry_at)
                record = {
                    "schema_version": "nearby-online-cache-v1",
                    "query": query,
                    "fetched_at": time.time(),
                    "payload": payload,
                }
                try:
                    LocalStateLayout._atomic_json_write(path, record)
                except OSError:
                    return self._result(record, "live", error="nearby_cache_write_failed", persisted=False)
                return self._result(record, "live")
        except StateLockTimeoutError:
            return self._fallback(cached, "nearby_refresh_in_progress")
        except OSError:
            return self._fallback(cached, "nearby_cache_unavailable")

    def _cached(self, path, query):
        record = _read(path)
        try:
            if (
                not isinstance(record, dict)
                or record.get("schema_version") != "nearby-online-cache-v1"
                or record.get("query") != query
            ):
                return None
            timestamp = record["fetched_at"]
            if (
                type(timestamp) not in {int, float}
                or not math.isfinite(timestamp)
                or not 0 < timestamp <= time.time() + 60
            ):
                return None
            _places(record["payload"])
            return record
        except (ValueError, KeyError, TypeError, AttributeError):
            return None

    def _fallback(self, cached, error, retry_at=None):
        status = "fresh_cache" if cached and time.time() - cached["fetched_at"] <= FRESH_SECONDS else "stale_cache"
        return self._result(cached, status if cached else "bundled_fallback", error=error, retry_at=retry_at)

    @staticmethod
    def _result(record, status, *, error=None, retry_at=None, persisted=True):
        return (_places(record["payload"]) if record else []), {
            "status": status,
            "provider": "OpenStreetMap / Overpass",
            "fetched_at": _iso(record["fetched_at"]) if record else None,
            "expires_at": _iso(record["fetched_at"] + FRESH_SECONDS) if record else None,
            "cache_persisted": bool(record and persisted),
            "error_code": error,
            "retry_after": _iso(retry_at),
            "attribution": "OpenStreetMap contributors",
            "license_url": "https://www.openstreetmap.org/copyright",
        }
