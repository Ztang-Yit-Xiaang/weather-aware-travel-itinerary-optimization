"""Adversarial runtime-witness tests with no Docker, network, or provider calls."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("runtime_environment", ROOT / "scripts/e2e1_runtime_environment.py")
environment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(environment)


@pytest.fixture
def captured(tmp_path, monkeypatch):
    import build_product_route_delta as delta

    image = "ghcr.io/project-osrm/osrm-backend@sha256:" + "a" * 64
    container = {
        "container_id": "c" * 64,
        "running": True,
        "configured_image": image,
        "image_id": "sha256:" + "a" * 64,
        "command": ["osrm-routed", "--algorithm", "mld", "/data/california.osrm"],
        "mounts": [{"Destination": "/data", "Source": str(delta.DATASET), "RW": False, "Ignored": "not-persisted"}],
        "ports": {"5000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5000"}]},
    }
    provenance = tmp_path / "osrm-source.json"
    provenance.write_text(
        json.dumps(
            {"osrm_image": image, "endpoint": delta.ENDPOINT, "provider_kind": "local_osrm", "osm_pbf_sha256": "b" * 64}
        ),
        encoding="utf-8",
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command == ["node", "--version"]:
            return SimpleNamespace(stdout="v22.15.0\n")
        assert command[:3] == ["docker", "inspect", "c" * 64]
        assert "Config.Env" not in command[-1]
        return SimpleNamespace(stdout=json.dumps(container))

    monkeypatch.setattr(environment.subprocess, "run", run)
    monkeypatch.setattr(environment.importlib.metadata, "version", lambda name: "1.2.3")
    monkeypatch.setattr(delta, "sha256", lambda path: "b" * 64)
    assets = ROOT / "docker/maplibre/assets"
    map_provenance = json.loads((assets / "provenance.json").read_text(encoding="utf-8"))
    requests = []

    def fetch(url, *, origin, maximum, byte_range=None):
        requests.append({"url": url, "maximum": maximum, "range": byte_range})
        assert url.startswith("http://127.0.0.1:8080/")
        if byte_range:
            return (
                206,
                {
                    "content-range": f"bytes 0-1023/{map_provenance['artifact_size']}",
                    "content-length": "1024",
                    "accept-ranges": "bytes",
                    "access-control-allow-origin": origin,
                },
                b"PMTiles\x03" + b"x" * 1016,
            )
        return 200, {}, (assets / url.removeprefix("http://127.0.0.1:8080/")).read_bytes()

    monkeypatch.setattr(environment, "_fetch", fetch)
    browser = environment.browser_identity(
        SimpleNamespace(version="140.0.1.2", browser_type=SimpleNamespace(name="chromium")),
        launch_choice="installed_chrome",
    )

    def capture():
        return environment.capture_runtime_environment(
            root=ROOT,
            run_id="synthetic_r24",
            actual_browser=browser,
            osrm_container_id="c" * 64,
            osrm_provenance_path=provenance,
            app_origin="http://127.0.0.1:8127",
        )

    return SimpleNamespace(
        capture=capture, container=container, browser=browser, requests=requests, calls=calls, fetch=fetch
    )


def test_runtime_witness_allowlist_and_declared_archive_scope(captured, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret-not-for-evidence")
    monkeypatch.setenv("UNRELATED_PASSWORD", "synthetic-password-not-for-evidence")
    record = captured.capture()
    environment.validate_runtime_environment(record, run_id="synthetic_r24", root=ROOT)
    serialized = json.dumps(record)
    assert "synthetic-secret" not in serialized and "synthetic-password" not in serialized
    assert "Ignored" not in serialized and "Config.Env" not in serialized
    assert set(record["dependencies"]["packages"]) == set(environment.PACKAGES)
    assert record["map"]["pmtiles"]["full_archive_rehashed"] is False
    assert record["osrm"]["pbf_full_rehashed"] is True and record["osrm"]["graph_build_attested"] is False
    assert len(captured.requests) == 6 and captured.requests[-1]["maximum"] == 1024
    assert captured.calls[0] == ["node", "--version"]


@pytest.mark.parametrize(
    "mutation",
    [
        "run_id",
        "extra_secret",
        "browser",
        "map_hash",
        "range",
        "full_archive",
        "image",
        "mount",
        "bool_counter",
        "dependency",
        "timestamp",
    ],
)
def test_tampered_environment_record_is_rejected(captured, mutation):
    record = deepcopy(captured.capture())
    if mutation == "run_id":
        record["run_id"] = "different"
    elif mutation == "extra_secret":
        record["dependencies"]["packages"]["OPENAI_API_KEY"] = "synthetic-private"
    elif mutation == "browser":
        record["browser"]["source"] = "separate_probe"
    elif mutation == "map_hash":
        record["map"]["served_assets"][1]["sha256"] = "not-a-hash"
    elif mutation == "range":
        record["map"]["pmtiles"]["content_range"] = "bytes 0-1023/1024"
    elif mutation == "full_archive":
        record["map"]["pmtiles"]["full_archive_rehashed"] = True
    elif mutation == "image":
        record["osrm"]["image_id"] = "sha256:" + "d" * 64
    elif mutation == "mount":
        record["osrm"]["data_mount"]["RW"] = True
    elif mutation == "bool_counter":
        record["map"]["pmtiles"]["declared_size_bytes"] = True
    elif mutation == "dependency":
        record["dependencies"]["dependency_files"][0]["sha256"] = "0" * 64
    else:
        record["completed_at"] = "2020-01-01T00:00:00+00:00"
    with pytest.raises(environment.RuntimeEnvironmentError):
        environment.validate_runtime_environment(record, run_id="synthetic_r24", root=ROOT)


@pytest.mark.parametrize(
    "url",
    [
        "https://external.invalid",
        "http://user:secret@127.0.0.1:8080",
        "http://127.0.0.1:8080/?secret=x",
        "http://127.0.0.1:8080/#x",
    ],
)
def test_external_or_credential_origins_rejected_before_fetch(url, monkeypatch):
    monkeypatch.setattr(environment, "_fetch", lambda *args, **kwargs: pytest.fail("must not fetch"))
    with pytest.raises(environment.RuntimeEnvironmentError, match="origin_invalid"):
        environment.collect_map(app_origin="http://127.0.0.1:8127", map_origin=url)


def test_served_map_asset_substitution_is_rejected(captured, monkeypatch):
    def fetch(url, **kwargs):
        if url.endswith("maplibre-gl.js"):
            return 200, {}, b"substituted renderer"
        return captured.fetch(url, **kwargs)

    monkeypatch.setattr(environment, "_fetch", fetch)
    with pytest.raises(environment.RuntimeEnvironmentError, match="served_asset_mismatch"):
        captured.capture()


def test_range_failure_is_not_success(captured, monkeypatch):
    def fetch(url, **kwargs):
        status, headers, body = captured.fetch(url, **kwargs)
        return (200 if kwargs.get("byte_range") else status), headers, body

    monkeypatch.setattr(environment, "_fetch", fetch)
    with pytest.raises(environment.RuntimeEnvironmentError, match="pmtiles_range_invalid"):
        captured.capture()


@pytest.mark.parametrize(
    ("field", "value"),
    [("running", False), ("image_id", "sha256:" + "d" * 64), ("command", ["osrm-routed", "/data/other.osrm"])],
)
def test_observed_docker_mismatch_rejected(captured, field, value):
    captured.container[field] = value
    with pytest.raises(environment.RuntimeEnvironmentError, match="osrm_identity_failed"):
        captured.capture()


def test_missing_actual_browser_fails_before_any_observation(captured):
    captured.browser.clear()
    with pytest.raises(environment.RuntimeEnvironmentError, match="browser_identity_invalid"):
        captured.capture()
    assert captured.calls == captured.requests == []


def test_unavailable_node_and_unsafe_version_output_are_sanitized(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("private environment should not be included")

    monkeypatch.setattr(environment.subprocess, "run", fail)
    with pytest.raises(environment.RuntimeEnvironmentError, match="^runtime_environment_node_unavailable$"):
        environment.collect_dependencies(ROOT)
    monkeypatch.setattr(
        environment.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="v22.1.2\nSECRET=value")
    )
    with pytest.raises(environment.RuntimeEnvironmentError, match="^runtime_environment_node_version_invalid$"):
        environment.collect_dependencies(ROOT)


def test_redirect_is_never_followed():
    with pytest.raises(environment.RuntimeEnvironmentError, match="redirect_forbidden"):
        environment._RejectRedirects().redirect_request(None)
