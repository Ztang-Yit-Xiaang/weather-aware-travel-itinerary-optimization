from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType

import pytest

from itinerary_system.product_app.config import ProductConfigError, ProductRuntimeConfig
from itinerary_system.product_app.map_asset_contract import (
    EXPECTED_SOURCE_BUILD,
    EXPECTED_SOURCE_URL,
    EXPECTED_UPSTREAM_BLAKE3,
    EXPECTED_UPSTREAM_PACKAGES,
)
from itinerary_system.product_app.models import ComponentHealthV1
from itinerary_system.product_app.runtime import MAP_CACHE_SECONDS, ProductRuntime

ENVIRONMENT_KEYS = (
    "PRODUCT_APP_ORIGIN",
    "PRODUCT_MAP_PROVIDER",
    "PRODUCT_MAP_BASE_URL",
    "PRODUCT_ROUTING_BASE_URL",
    "MAPBOX_ATLAS_LICENSE",
    "PRODUCT_COPILOT_ADAPTER",
    "OPENAI_COPILOT_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_COPILOT_TIMEOUT_SECONDS",
    "OPENAI_COPILOT_HISTORY_MESSAGES",
    "OPENAI_COPILOT_HISTORY_CHARACTERS",
)


def runtime_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **environment: str
) -> ProductRuntimeConfig:
    for key in ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    return ProductRuntimeConfig.from_environment(
        repository_root=tmp_path,
        registry_path=tmp_path / "registry.json",
        state_root=tmp_path / "state",
    )


def test_runtime_config_defaults_and_secret_repr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runtime_config(
        tmp_path,
        monkeypatch,
        MAPBOX_ATLAS_LICENSE="atlas-browser-license",
        OPENAI_API_KEY="openai-secret",
    )

    assert config.application_host == "127.0.0.1"
    assert config.application_port == 8127
    assert config.application_origin == "http://127.0.0.1:8127"
    assert config.map_provider == "maplibre_pmtiles"
    assert config.map_base_url == "http://127.0.0.1:8080"
    assert config.routing_base_url == "http://127.0.0.1:5000"
    assert config.copilot_adapter == "deterministic"
    assert config.openai_model == "gpt-5.6-terra"
    assert config.openai_timeout_seconds == 30
    assert config.openai_history_messages == 12
    assert config.openai_history_characters == 12_000
    assert config.allowed_authorities == frozenset({"127.0.0.1:8127", "localhost:8127"})
    assert "atlas-browser-license" not in repr(config)
    assert "openai-secret" not in repr(config)


@pytest.mark.parametrize(
    ("environment", "code"),
    [
        ({"PRODUCT_APP_ORIGIN": "https://127.0.0.1:8127"}, "invalid_loopback_url"),
        ({"PRODUCT_APP_ORIGIN": "http://127.0.0.1:9999"}, "application_origin_port_mismatch"),
        ({"PRODUCT_MAP_PROVIDER": "automatic"}, "invalid_map_provider"),
        ({"PRODUCT_MAP_BASE_URL": "http://example.com:8080"}, "invalid_loopback_url"),
        ({"PRODUCT_MAP_BASE_URL": "http://127.0.0.1:8081"}, "invalid_loopback_url"),
        ({"PRODUCT_MAP_BASE_URL": "http://user@127.0.0.1:8080"}, "invalid_loopback_url"),
        ({"PRODUCT_ROUTING_BASE_URL": "http://example.com:5000"}, "invalid_loopback_url"),
        ({"PRODUCT_ROUTING_BASE_URL": "https://127.0.0.1:5000"}, "invalid_loopback_url"),
        ({"PRODUCT_ROUTING_BASE_URL": "http://user@127.0.0.1:5000"}, "invalid_loopback_url"),
        ({"PRODUCT_ROUTING_BASE_URL": "http://127.0.0.1:5000/route"}, "invalid_loopback_url"),
        ({"PRODUCT_COPILOT_ADAPTER": "fixture"}, "invalid_copilot_adapter"),
        ({"OPENAI_COPILOT_MODEL": "  "}, "invalid_openai_model"),
        (
            {"OPENAI_COPILOT_TIMEOUT_SECONDS": "0"},
            "invalid_openai_copilot_timeout_seconds",
        ),
        (
            {"OPENAI_COPILOT_TIMEOUT_SECONDS": "61"},
            "invalid_openai_copilot_timeout_seconds",
        ),
        (
            {"OPENAI_COPILOT_TIMEOUT_SECONDS": "1.5"},
            "invalid_openai_copilot_timeout_seconds",
        ),
        (
            {"OPENAI_COPILOT_HISTORY_MESSAGES": "21"},
            "invalid_openai_copilot_history_messages",
        ),
        (
            {"OPENAI_COPILOT_HISTORY_CHARACTERS": "-1"},
            "invalid_openai_copilot_history_characters",
        ),
    ],
)
def test_runtime_config_rejects_unsafe_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    code: str,
) -> None:
    with pytest.raises(ProductConfigError, match=code) as error:
        runtime_config(tmp_path, monkeypatch, **environment)
    assert error.value.code == code


@pytest.mark.parametrize(
    ("host", "port", "code"),
    [
        ("0.0.0.0", 8127, "non_loopback_host_not_allowed"),
        ("127.0.0.1", 0, "invalid_application_port"),
        ("127.0.0.1", 65_536, "invalid_application_port"),
        ("127.0.0.1", True, "invalid_application_port"),
    ],
)
def test_runtime_config_rejects_non_loopback_or_invalid_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
    port: int,
    code: str,
) -> None:
    for key in ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ProductConfigError, match=code):
        ProductRuntimeConfig.from_environment(
            repository_root=tmp_path,
            registry_path=tmp_path / "registry.json",
            state_root=tmp_path / "state",
            application_host=host,
            application_port=port,
        )


def _healthy_maplibre_fetch(
    path_or_url: str, *, range_request: bool = False
) -> tuple[bytes, dict[str, str], int]:
    cors = {"Access-Control-Allow-Origin": "http://127.0.0.1:8127"}
    if path_or_url == "http://127.0.0.1:8080/styles/protomaps-light.json":
        style = {
            "version": 8,
            "sprite": "http://127.0.0.1:8080/sprites/light",
            "glyphs": "http://127.0.0.1:8080/fonts/{fontstack}/{range}.pbf",
            "sources": {
                "protomaps": {
                    "type": "vector",
                    "url": (
                        "pmtiles://http://127.0.0.1:8080/data/"
                        "california-coast-v1.pmtiles"
                    ),
                    "attribution": (
                        'Protomaps | <a href="https://www.openstreetmap.org/copyright" '
                        'target="_blank" rel="noopener">© OpenStreetMap contributors</a>'
                    ),
                }
            },
            "layers": [
                {
                    "id": "labels",
                    "type": "symbol",
                    "source": "protomaps",
                    "layout": {"text-field": ["get", "name"], "text-font": ["Noto Sans Regular"]},
                }
            ],
        }
        return json.dumps(style).encode(), cors, 200
    if path_or_url == "http://127.0.0.1:8080/provenance.json":
        return json.dumps(
            {
                "schema_version": "map-data-provenance-v1",
                "source": "OpenStreetMap",
                "source_url": EXPECTED_SOURCE_URL,
                "source_build": EXPECTED_SOURCE_BUILD,
                "upstream_blake3": EXPECTED_UPSTREAM_BLAKE3,
                "upstream_packages": EXPECTED_UPSTREAM_PACKAGES,
                "license": "ODbL-1.0",
                "attribution": "Protomaps | © OpenStreetMap contributors",
                "attribution_url": "https://www.openstreetmap.org/copyright",
                "artifact_url": "/data/california-coast-v1.pmtiles",
                "artifact_sha256": "a" * 64,
                "artifact_size": 2048,
                "bounds": [-123.0, 33.5, -117.5, 38.5],
                "maxzoom": 15,
                "generated_at": "2026-08-04T00:00:00Z",
                "extraction_command": (
                    "pmtiles extract SOURCE data/california-coast-v1.pmtiles "
                    "--bbox=-123.0,33.5,-117.5,38.5 --maxzoom=15"
                ),
                "header_json_sha256": "c" * 64,
                "metadata_json_sha256": "d" * 64,
                "verify_output_sha256": "e" * 64,
                "tool_versions": {
                    "maplibre-gl": "5.24.0",
                    "pmtiles-js": "4.4.1",
                    "protomaps-basemap-style": "5.7.2",
                    "pmtiles-cli": "1.30.0",
                },
                "license_notices": {
                    "BSD-3-Clause": "licenses/BSD-3-Clause.txt",
                    "CC0-1.0": "licenses/CC0-1.0.txt",
                    "MIT": "licenses/MIT.txt",
                    "ODbL-1.0": "licenses/ODbL-1.0.txt",
                    "OFL-1.1": "licenses/OFL-1.1.txt",
                    "Protomaps-Data-Notices": "licenses/PROTOMAPS_DATA.txt",
                },
                "glyph_ranges": ["fonts/NotoSans/0-255.pbf"],
                "assets": [
                    {"path": path, "sha256": "f" * 64, "license": license_id}
                    for path, license_id in (
                        ("maplibre/maplibre-gl.js", "BSD-3-Clause"),
                        ("maplibre/maplibre-gl.css", "BSD-3-Clause"),
                        ("pmtiles/pmtiles.js", "BSD-3-Clause"),
                        ("styles/protomaps-light.json", "CC0-1.0"),
                        ("fonts/NotoSans/0-255.pbf", "OFL-1.1"),
                        ("licenses/BSD-3-Clause.txt", "BSD-3-Clause"),
                        ("licenses/CC0-1.0.txt", "CC0-1.0"),
                        ("licenses/MIT.txt", "MIT"),
                        ("licenses/ODbL-1.0.txt", "ODbL-1.0"),
                        ("licenses/OFL-1.1.txt", "OFL-1.1"),
                        ("licenses/PROTOMAPS_DATA.txt", "Protomaps-Data-Notices"),
                    )
                ],
            }
        ).encode(), cors, 200
    if range_request:
        assert path_or_url == "http://127.0.0.1:8080/data/california-coast-v1.pmtiles"
        return b"PMTiles" + bytes([3]) + bytes(1016), {
            **cors,
            "Accept-Ranges": "bytes",
            "Content-Range": "bytes 0-1023/2048",
            "Content-Length": "1024",
        }, 206
    return b"ok", cors, 200


def test_maplibre_probe_accepts_loopback_style_provenance_and_range_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ProductRuntime(runtime_config(tmp_path, monkeypatch))
    monkeypatch.setattr(runtime, "_fetch", _healthy_maplibre_fetch)

    health = runtime._probe_map()

    assert health.name == "map"
    assert health.status == "ready"
    assert health.code == "maplibre_ready"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("range_200", "pmtiles_range_status_invalid"),
        ("missing_accept_ranges", "pmtiles_accept_ranges_missing"),
        ("bad_content_range", "pmtiles_content_range_invalid"),
        ("low_content_range_total", "pmtiles_content_range_invalid"),
        ("wrong_cors_origin", "map_cors_origin_invalid"),
        ("foreign_style_url", "map_style_url_not_loopback"),
        ("foreign_source_data", "map_style_url_not_loopback"),
        ("style_import", "map_style_imports_not_supported"),
        ("invalid_style_json", "map_style_invalid"),
        ("style_not_object", "map_style_invalid"),
        ("missing_sources", "map_style_invalid"),
        ("missing_attribution", "map_attribution_invalid"),
        ("plain_attribution", "map_attribution_invalid"),
        ("invalid_provenance", "map_provenance_invalid"),
        ("wrong_tool_version", "map_provenance_invalid"),
        ("wrong_upstream_source", "map_provenance_invalid"),
        ("missing_license_notices", "map_provenance_invalid"),
        ("mismatched_artifact", "map_provenance_invalid"),
        ("invalid_header", "pmtiles_header_invalid"),
        ("short_range_body", "pmtiles_range_length_invalid"),
        ("wrong_content_length", "pmtiles_range_length_invalid"),
        ("missing_sprite", "map_sprite_unavailable"),
        ("missing_glyph", "map_glyph_unavailable"),
    ],
)
def test_maplibre_probe_fails_closed_with_stable_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    runtime = ProductRuntime(runtime_config(tmp_path, monkeypatch))

    def fetch(path_or_url: str, *, range_request: bool = False):
        body, headers, status = _healthy_maplibre_fetch(path_or_url, range_request=range_request)
        if mutation == "invalid_style_json" and str(path_or_url).endswith("protomaps-light.json"):
            return b"{", headers, 200
        if mutation == "style_not_object" and str(path_or_url).endswith("protomaps-light.json"):
            return b"[]", headers, 200
        if mutation == "missing_sources" and str(path_or_url).endswith("protomaps-light.json"):
            return b'{"version":8}', headers, 200
        if mutation == "foreign_style_url" and str(path_or_url).endswith("protomaps-light.json"):
            return json.dumps(
                {"version": 8, "sprite": "https://example.com/sprite", "sources": {}}
            ).encode(), headers, 200
        if mutation == "foreign_source_data" and str(path_or_url).endswith(
            "protomaps-light.json"
        ):
            value = json.loads(body)
            value["sources"]["protomaps"]["data"] = "https://example.com/data.geojson"
            return json.dumps(value).encode(), headers, 200
        if mutation == "style_import" and str(path_or_url).endswith("protomaps-light.json"):
            value = json.loads(body)
            value["imports"] = [{"id": "foreign", "url": "https://example.com/style.json"}]
            return json.dumps(value).encode(), headers, 200
        if mutation == "missing_sprite" and str(path_or_url).endswith("/sprites/light.png"):
            return body, headers, 404
        if mutation == "missing_glyph" and "/fonts/" in str(path_or_url):
            return body, headers, 404
        if mutation == "invalid_provenance" and str(path_or_url).endswith("provenance.json"):
            return b"{}", headers, 200
        if mutation == "wrong_tool_version" and str(path_or_url).endswith("provenance.json"):
            value = json.loads(body)
            value["tool_versions"]["maplibre-gl"] = "latest"
            return json.dumps(value).encode(), headers, 200
        if mutation == "wrong_upstream_source" and str(path_or_url).endswith(
            "provenance.json"
        ):
            value = json.loads(body)
            value["upstream_packages"]["maplibre-gl"]["git_commit"] = "0" * 40
            return json.dumps(value).encode(), headers, 200
        if mutation == "missing_license_notices" and str(path_or_url).endswith("provenance.json"):
            value = json.loads(body)
            del value["license_notices"]
            return json.dumps(value).encode(), headers, 200
        if mutation == "mismatched_artifact" and str(path_or_url).endswith("provenance.json"):
            value = json.loads(body)
            value["artifact_url"] = "/data/different.pmtiles"
            return json.dumps(value).encode(), headers, 200
        if mutation == "missing_attribution" and str(path_or_url).endswith("protomaps-light.json"):
            value = json.loads(body)
            del value["sources"]["protomaps"]["attribution"]
            return json.dumps(value).encode(), headers, 200
        if mutation == "plain_attribution" and str(path_or_url).endswith(
            "protomaps-light.json"
        ):
            value = json.loads(body)
            value["sources"]["protomaps"]["attribution"] = (
                "© OpenStreetMap contributors https://www.openstreetmap.org/copyright"
            )
            return json.dumps(value).encode(), headers, 200
        if mutation == "wrong_cors_origin" and path_or_url == "/healthz":
            return body, {**headers, "Access-Control-Allow-Origin": "http://localhost:9999"}, status
        if range_request and mutation == "range_200":
            return body, headers, 200
        if range_request and mutation == "missing_accept_ranges":
            return body, {
                "Access-Control-Allow-Origin": headers["Access-Control-Allow-Origin"],
                "Content-Range": headers["Content-Range"],
            }, status
        if range_request and mutation == "bad_content_range":
            return body, {**headers, "Content-Range": "bytes 1-1024/2048"}, status
        if range_request and mutation == "low_content_range_total":
            return body, {**headers, "Content-Range": "bytes 0-1023/100"}, status
        if range_request and mutation == "invalid_header":
            return b"NotTile" + bytes([3]) + bytes(1016), headers, status
        if range_request and mutation == "short_range_body":
            return body[:127], headers, status
        if range_request and mutation == "wrong_content_length":
            return body, {**headers, "Content-Length": "127"}, status
        return body, headers, status

    monkeypatch.setattr(runtime, "_fetch", fetch)
    assert runtime._probe_map().code == expected_code


def test_map_fetch_rejects_redirect_outside_configured_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ProductRuntime(runtime_config(tmp_path, monkeypatch))

    class RedirectedResponse:
        status = 200
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def geturl(self) -> str:
            return "http://127.0.0.1:8081/redirected"

        def read(self, _: int) -> bytes:
            return b"redirected"

    def redirected(request: object, timeout: float) -> RedirectedResponse:
        assert timeout == 1.0
        assert request.get_header("Origin") == "http://127.0.0.1:8127"
        return RedirectedResponse()

    monkeypatch.setattr("itinerary_system.product_app.runtime.urllib.request.urlopen", redirected)
    with pytest.raises(ValueError, match="map_redirect_not_loopback"):
        runtime._fetch("/healthz")


@pytest.mark.parametrize(
    ("failed_url", "expected_code"),
    [
        ("/healthz", "map_health_invalid"),
        ("/maplibre/maplibre-gl.js", "map_script_unavailable"),
        ("/maplibre/maplibre-gl.css", "map_stylesheet_unavailable"),
        ("/pmtiles/pmtiles.js", "pmtiles_protocol_unavailable"),
        ("/styles/protomaps-light.json", "map_style_unavailable"),
    ],
)
def test_map_probe_reports_each_missing_local_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_url: str,
    expected_code: str,
) -> None:
    runtime = ProductRuntime(runtime_config(tmp_path, monkeypatch))

    def fetch(path_or_url: str, *, range_request: bool = False):
        body, headers, status = _healthy_maplibre_fetch(path_or_url, range_request=range_request)
        if str(path_or_url).endswith(failed_url):
            return body, headers, 404
        return body, headers, status

    monkeypatch.setattr(runtime, "_fetch", fetch)
    assert runtime._probe_map().code == expected_code


def test_map_probe_is_cached_for_two_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ProductRuntime(runtime_config(tmp_path, monkeypatch))
    runtime._initialized = True
    runtime._components = {
        "registry": ComponentHealthV1("registry", "ready", True, "registry_ready"),
        "default_workspace": ComponentHealthV1("default_workspace", "ready", True, "workspace_ready"),
        "state_store": ComponentHealthV1("state_store", "ready", True, "state_store_ready"),
        "openai": ComponentHealthV1("openai", "disabled", False, "deterministic_adapter_selected"),
        "map": ComponentHealthV1("map", "degraded", False, "maplibre_unavailable"),
    }
    runtime._map_checked_monotonic = time.monotonic()
    calls = 0

    def probe() -> ComponentHealthV1:
        nonlocal calls
        calls += 1
        return ComponentHealthV1("map", "ready", False, "maplibre_ready")

    monkeypatch.setattr(runtime, "_probe_map", probe)
    runtime.health()
    runtime.health()
    assert calls == 0

    runtime._map_checked_monotonic -= MAP_CACHE_SECONDS
    runtime.health()
    runtime.health()
    assert calls == 1


def test_atlas_is_explicit_opt_in_and_requires_runtime_license(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ProductRuntime(
        runtime_config(tmp_path, monkeypatch, PRODUCT_MAP_PROVIDER="mapbox_atlas_v3")
    )
    assert runtime._probe_map().code == "atlas_runtime_license_missing"
    assert runtime.map_configuration().provider == "mapbox_atlas_v3"


def test_atlas_backup_can_pass_only_when_explicitly_selected_and_licensed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ProductRuntime(
        runtime_config(
            tmp_path,
            monkeypatch,
            PRODUCT_MAP_PROVIDER="mapbox_atlas_v3",
            MAPBOX_ATLAS_LICENSE="read-only-runtime-license",
        )
    )
    cors = {"Access-Control-Allow-Origin": "http://127.0.0.1:8127"}

    def fetch(path_or_url: str, *, range_request: bool = False):
        if str(path_or_url).endswith("streets-v12.json"):
            return json.dumps(
                {
                    "version": 8,
                        "sprite": "/data/sprites/streets",
                        "glyphs": "/data/fonts/{fontstack}/{range}.pbf",
                        "sources": {"streets": {"url": "pmtiles:///data/tiles/streets.pmtiles"}},
                        "layers": [
                            {
                                "id": "labels",
                                "type": "symbol",
                                "source": "streets",
                                "layout": {
                                    "text-field": ["get", "name"],
                                    "text-font": ["Atlas Sans"],
                                },
                            }
                        ],
                }
            ).encode(), cors, 200
        if range_request:
            return b"PMTiles" + bytes([3]) + bytes(1016), {
                **cors,
                    "Accept-Ranges": "bytes",
                    "Content-Range": "bytes 0-1023/2048",
                    "Content-Length": "1024",
            }, 206
        return b"ok", cors, 200

    monkeypatch.setattr(runtime, "_fetch", fetch)
    assert runtime._probe_map().code == "atlas_ready"
    assert runtime.map_configuration().runtime_license == "read-only-runtime-license"


def test_openai_health_is_explicit_and_does_not_expose_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deterministic = ProductRuntime(runtime_config(tmp_path, monkeypatch))
    assert deterministic._openai_health().code == "deterministic_adapter_selected"

    unconfigured = ProductRuntime(
        runtime_config(tmp_path, monkeypatch, PRODUCT_COPILOT_ADAPTER="openai")
    )
    assert unconfigured._openai_health().code == "openai_not_configured"

    configured = ProductRuntime(
        runtime_config(
            tmp_path,
            monkeypatch,
            PRODUCT_COPILOT_ADAPTER="openai",
            OPENAI_API_KEY="never-render-this-key",
        )
    )
    monkeypatch.setattr(configured, "_construct_openai_adapter", object)
    serialized = json.dumps(configured._openai_health().as_dict())
    configured_health = configured._openai_health()
    assert configured_health.status == "ready"
    assert configured_health.code == "openai_configured"
    assert "never-render-this-key" not in serialized


def test_runtime_config_accepts_explicit_copilot_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runtime_config(
        tmp_path,
        monkeypatch,
        OPENAI_COPILOT_TIMEOUT_SECONDS="60",
        OPENAI_COPILOT_HISTORY_MESSAGES="0",
        OPENAI_COPILOT_HISTORY_CHARACTERS="20000",
    )

    assert config.openai_timeout_seconds == 60
    assert config.openai_history_messages == 0
    assert config.openai_history_characters == 20_000


def test_openai_health_reports_local_transport_construction_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ProductRuntime(
        runtime_config(
            tmp_path,
            monkeypatch,
            PRODUCT_COPILOT_ADAPTER="openai",
            OPENAI_API_KEY="not-a-real-key",
        )
    )

    def fail_construction() -> object:
        raise RuntimeError("must not escape into health")

    monkeypatch.setattr(runtime, "_construct_openai_adapter", fail_construction)
    health = runtime._openai_health()

    assert health.status == "degraded"
    assert health.code == "openai_transport_unavailable"
    assert "must not escape" not in json.dumps(health.as_dict())


def _load_launcher_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_product_app.py"
    spec = importlib.util.spec_from_file_location("w4_runtime_launcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_environment_loader_is_literal_atomic_and_does_not_override(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher_module()
    (tmp_path / ".env.local").write_text(
        "# local only\nOPENAI_API_KEY=local-secret\n"
        "OPENAI_COPILOT_MODEL=$(not-a-command)\n",
        encoding="utf-8",
    )
    environment = {"OPENAI_COPILOT_MODEL": "explicit-process-value"}

    loaded = launcher.load_local_environment(tmp_path, environ=environment)

    assert loaded == 1
    assert environment == {
        "OPENAI_API_KEY": "local-secret",
        "OPENAI_COPILOT_MODEL": "explicit-process-value",
    }


def test_local_environment_loader_rejects_malformed_file_without_partial_load(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher_module()
    (tmp_path / ".env.local").write_text(
        "OPENAI_API_KEY=must-not-load\nexport INVALID=value\n",
        encoding="utf-8",
    )
    environment: dict[str, str] = {}

    with pytest.raises(launcher.LocalEnvironmentError) as error:
        launcher.load_local_environment(tmp_path, environ=environment)

    assert error.value.code == "local_environment_malformed"
    assert environment == {}
    assert "must-not-load" not in str(error.value)

    (tmp_path / ".env.local").write_text("UNRELATED_SETTING=value\n", encoding="utf-8")
    with pytest.raises(launcher.LocalEnvironmentError) as unknown_key:
        launcher.load_local_environment(tmp_path, environ=environment)
    assert unknown_key.value.code == "local_environment_key_not_allowed"
    assert environment == {}


def test_local_environment_loader_rejects_oversized_and_symlink_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher_module()
    local_environment = tmp_path / ".env.local"
    local_environment.write_bytes(b"A=" + b"x" * launcher.LOCAL_ENV_MAX_BYTES)
    with pytest.raises(launcher.LocalEnvironmentError) as oversized:
        launcher.load_local_environment(tmp_path, environ={})
    assert oversized.value.code == "local_environment_too_large"

    local_environment.unlink()
    local_environment.write_text("OPENAI_API_KEY=must-not-load\n", encoding="utf-8")
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda value: value == local_environment or original_is_symlink(value),
    )
    with pytest.raises(launcher.LocalEnvironmentError) as symlinked:
        launcher.load_local_environment(tmp_path, environ={})
    assert symlinked.value.code == "local_environment_symlink_not_allowed"


def test_local_environment_loader_rejects_resolved_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher_module()
    local_environment = tmp_path / ".env.local"
    local_environment.write_text("OPENAI_API_KEY=must-not-load\n", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.env"
    outside.write_text("OPENAI_API_KEY=outside\n", encoding="utf-8")
    original_resolve = Path.resolve

    def escaped_resolve(value: Path, *, strict: bool = False) -> Path:
        if value == local_environment:
            return outside
        return original_resolve(value, strict=strict)

    monkeypatch.setattr(Path, "resolve", escaped_resolve)
    with pytest.raises(launcher.LocalEnvironmentError) as escaped:
        launcher.load_local_environment(tmp_path, environ={})
    assert escaped.value.code == "local_environment_outside_repository"
