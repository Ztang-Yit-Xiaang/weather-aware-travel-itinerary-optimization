"""Secret-free observed E2E runtime inventory; not a whole-machine snapshot."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "src", ROOT / "scripts"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from itinerary_system.product_app.map_asset_contract import EXPECTED_TOOL_VERSIONS, valid_provenance  # noqa: E402

PACKAGES = ("anyio", "fastapi", "folium", "geopy", "gurobipy", "httpcore", "httpx", "jupyter",
            "matplotlib", "numpy", "openai", "openrouteservice", "ortools", "pandas", "playwright",
            "pydantic", "pytest", "requests", "scikit-learn", "starlette", "urllib3", "uvicorn", "xgboost")
DEPENDENCY_FILES = ("pyproject.toml", "requirements.txt", "uv.lock", "poetry.lock", "package-lock.json")
MAP_ASSETS = ("maplibre/maplibre-gl.js", "maplibre/maplibre-gl.css", "pmtiles/pmtiles.js", "styles/protomaps-light.json")
VERSION = re.compile(r"[0-9][0-9A-Za-z.+_-]{0,79}")
SHA256 = re.compile(r"[0-9a-f]{64}")
SCHEMA = "e2e1-runtime-environment-v1"


class RuntimeEnvironmentError(ValueError):
    """Stable error; never contains raw process output or credentials."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeEnvironmentError(code)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _origin(value: str) -> str:
    try:
        parsed = urlparse(value)
        valid = (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
                 and parsed.port is not None and 1 <= parsed.port <= 65535
                 and parsed.username is None and parsed.password is None
                 and not parsed.query and not parsed.fragment and parsed.path in {"", "/"})
    except (ValueError, TypeError):
        valid = False
    _require(valid, "runtime_environment_origin_invalid")
    return value.rstrip("/")


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeEnvironmentError("runtime_environment_redirect_forbidden")


def _fetch(url: str, *, origin: str, maximum: int, byte_range: str | None = None) -> tuple[int, dict, bytes]:
    headers = {"Origin": origin}
    if byte_range:
        headers["Range"] = byte_range
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _RejectRedirects())
    try:
        with opener.open(urllib.request.Request(url, headers=headers), timeout=5) as response:
            body = response.read(maximum + 1)
            _require(len(body) <= maximum, "runtime_environment_response_oversize")
            return response.status, {key.lower(): value for key, value in response.headers.items()}, body
    except OSError:
        raise RuntimeEnvironmentError("runtime_environment_http_unavailable") from None


def browser_identity(browser: Any, *, launch_choice: str) -> dict[str, str]:
    """Called by the journey with its actual browser, never a separate probe."""
    result = {"source": "actual_test_browser", "engine": browser.browser_type.name,
              "version": browser.version, "launch_choice": launch_choice}
    _validate_browser(result)
    return result


def _validate_browser(value: dict) -> None:
    _require(type(value) is dict and set(value) == {"source", "engine", "version", "launch_choice"}
             and value.get("source") == "actual_test_browser" and value.get("engine") == "chromium"
             and value.get("launch_choice") in {"installed_chrome", "playwright_chromium"}
             and type(value.get("version")) is str and VERSION.fullmatch(value["version"]) is not None,
             "runtime_environment_browser_identity_invalid")


def collect_dependencies(root: Path) -> dict:
    packages = {}
    for package in PACKAGES:
        try:
            value = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            value = None
        _require(value is None or VERSION.fullmatch(value) is not None, "runtime_environment_package_version_invalid")
        packages[package] = value
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        raise RuntimeEnvironmentError("runtime_environment_node_unavailable") from None
    node = result.stdout.strip()
    _require(re.fullmatch(r"v\d+\.\d+\.\d+", node) is not None, "runtime_environment_node_version_invalid")
    files = []
    for relative in DEPENDENCY_FILES:
        path = root / relative
        data = path.read_bytes() if path.is_file() else None
        files.append({"path": relative, "present": data is not None,
                      "size_bytes": len(data) if data is not None else None,
                      "sha256": _sha256(data) if data is not None else None})
    _require(all(row["present"] for row in files[:2]), "runtime_environment_dependency_inputs_missing")
    return {"python": {"implementation": platform.python_implementation(), "version": platform.python_version()},
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "node": {"version": node}, "packages": packages, "dependency_files": files}


def collect_map(*, app_origin: str, map_origin: str) -> dict:
    app_origin, map_origin = _origin(app_origin), _origin(map_origin)
    status, _, raw = _fetch(map_origin + "/provenance.json", origin=app_origin, maximum=100_000)
    try:
        provenance = json.loads(raw)
    except (ValueError, UnicodeError):
        raise RuntimeEnvironmentError("runtime_environment_map_provenance_invalid") from None
    _require(status == 200 and valid_provenance(provenance, expected_artifact_url="/data/california-coast-v1.pmtiles"),
             "runtime_environment_map_provenance_invalid")
    declared = {row["path"]: row["sha256"] for row in provenance["assets"]}
    assets = [{"path": "provenance.json", "sha256": _sha256(raw), "size_bytes": len(raw)}]
    for relative in MAP_ASSETS:
        status, _, data = _fetch(map_origin + "/" + relative, origin=app_origin, maximum=5_000_000)
        _require(status == 200 and _sha256(data) == declared.get(relative), "runtime_environment_served_asset_mismatch")
        assets.append({"path": relative, "sha256": _sha256(data), "size_bytes": len(data)})
    status, headers, prefix = _fetch(map_origin + provenance["artifact_url"], origin=app_origin, maximum=1024, byte_range="bytes=0-1023")
    content_range = f"bytes 0-1023/{provenance['artifact_size']}"
    _require(status == 206 and len(prefix) == 1024 and prefix[:8] == b"PMTiles\x03"
             and headers.get("content-range") == content_range and headers.get("content-length") == "1024"
             and headers.get("accept-ranges") == "bytes" and headers.get("access-control-allow-origin") == app_origin,
             "runtime_environment_pmtiles_range_invalid")
    return {"origin": map_origin, "served_assets": assets, "tool_versions": provenance["tool_versions"],
            "pmtiles": {"artifact_url": provenance["artifact_url"], "declared_archive_sha256": provenance["artifact_sha256"],
                        "declared_size_bytes": provenance["artifact_size"], "full_archive_rehashed": False,
                        "range_status": status, "content_range": content_range, "range_size_bytes": len(prefix),
                        "range_sha256": _sha256(prefix), "scope": "declared_archive_identity_and_observed_prefix_only"}}


def collect_osrm(*, container_id: str, provenance_path: Path) -> dict:
    from build_product_route_delta import runtime_witness

    _require(type(container_id) is str and re.fullmatch(r"[0-9a-f]{12,64}", container_id) is not None,
             "runtime_environment_container_id_invalid")
    try:
        raw = provenance_path.read_bytes()
        witness = runtime_witness(container_id, json.loads(raw))
    except (OSError, ValueError, subprocess.SubprocessError):
        raise RuntimeEnvironmentError("runtime_environment_osrm_identity_failed") from None
    container = witness["container"]
    return {"observed_at": witness["observed_at"], "provenance_sha256": _sha256(raw),
            "container_id": container["container_id"], "running": container["running"],
            "configured_image": container["configured_image"], "image_id": container["image_id"],
            "command": container["command"], "data_mount": [
                {key: row[key] for key in ("Source", "Destination", "RW")}
                for row in container["mounts"] if row["Destination"] == "/data"
            ][0], "ports": container["ports"], "pbf_sha256": witness["pbf_sha256"],
            "pbf_full_rehashed": True, "graph_build_attested": False}


def capture_runtime_environment(*, root: Path, run_id: str, actual_browser: dict,
                                osrm_container_id: str, osrm_provenance_path: Path,
                                app_origin: str, map_origin: str = "http://127.0.0.1:8080") -> dict:
    _require(type(run_id) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id) is not None,
             "runtime_environment_run_id_invalid")
    _validate_browser(actual_browser)
    result = {"schema_version": SCHEMA, "run_id": run_id, "observed_at": datetime.now(UTC).isoformat(),
              "scope": "allowlisted_runtime_identity_not_whole_machine_or_graph_build", "application_origin": _origin(app_origin),
              "dependencies": collect_dependencies(root), "browser": dict(actual_browser),
              "map": collect_map(app_origin=app_origin, map_origin=map_origin),
              "osrm": collect_osrm(container_id=osrm_container_id, provenance_path=osrm_provenance_path)}
    result["completed_at"] = datetime.now(UTC).isoformat()
    validate_runtime_environment(result, run_id=run_id, root=root)
    return result


def validate_runtime_environment(record: dict, *, run_id: str, root: Path | None = None) -> None:
    """Check the allowlisted witness; the enclosing manifest binds artifact bytes."""
    code = "runtime_environment_record_invalid"
    def keys(value: Any, names: set[str]) -> bool:
        return type(value) is dict and set(value) == names
    def digest(value: Any) -> bool:
        return type(value) is str and SHA256.fullmatch(value) is not None
    def version(value: Any) -> bool:
        return type(value) is str and VERSION.fullmatch(value) is not None
    _require(keys(record, {"schema_version", "run_id", "observed_at", "completed_at", "scope", "application_origin", "dependencies", "browser", "map", "osrm"})
             and record["schema_version"] == SCHEMA and record["run_id"] == run_id
             and type(run_id) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id) is not None
             and record["scope"] == "allowlisted_runtime_identity_not_whole_machine_or_graph_build", code)
    _origin(record["application_origin"])
    _validate_browser(record["browser"])
    try:
        observed, completed, osrm_observed = [datetime.fromisoformat(value) for value in (
            record["observed_at"], record["completed_at"], record["osrm"]["observed_at"])]
        _require(all(value.tzinfo is not None for value in (observed, completed, osrm_observed))
                 and observed <= osrm_observed <= completed, code)
    except (ValueError, TypeError, KeyError):
        raise RuntimeEnvironmentError(code) from None
    deps = record["dependencies"]
    _require(keys(deps, {"python", "platform", "node", "packages", "dependency_files"})
             and keys(deps["python"], {"implementation", "version"}) and deps["python"]["implementation"] in {"CPython", "PyPy"}
             and version(deps["python"]["version"]) and keys(deps["node"], {"version"})
             and type(deps["node"]["version"]) is str and re.fullmatch(r"v\d+\.\d+\.\d+", deps["node"]["version"]) is not None
             and keys(deps["platform"], {"system", "machine"})
             and all(type(value) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,32}", value) for value in deps["platform"].values())
             and keys(deps["packages"], set(PACKAGES)) and all(value is None or version(value) for value in deps["packages"].values()), code)
    rows = deps["dependency_files"]
    _require(type(rows) is list and len(rows) == len(DEPENDENCY_FILES), code)
    for row, expected in zip(rows, DEPENDENCY_FILES, strict=True):
        _require(keys(row, {"path", "present", "size_bytes", "sha256"}) and row["path"] == expected and type(row["present"]) is bool, code)
        if row["present"]:
            _require(type(row["size_bytes"]) is int and row["size_bytes"] >= 0 and digest(row["sha256"]), code)
        else:
            _require(expected not in DEPENDENCY_FILES[:2] and row["size_bytes"] is None and row["sha256"] is None, code)
        if root is not None:
            path = root / expected
            _require(path.is_file() == row["present"] and (not row["present"] or
                     path.stat().st_size == row["size_bytes"] and _sha256(path.read_bytes()) == row["sha256"]), "runtime_environment_dependency_hash_mismatch")
    maps = record["map"]
    _require(keys(maps, {"origin", "served_assets", "tool_versions", "pmtiles"}) and maps["tool_versions"] == EXPECTED_TOOL_VERSIONS, code)
    _origin(maps["origin"])
    assets = maps["served_assets"]
    _require(type(assets) is list and len(assets) == 5, code)
    for row, expected in zip(assets, ("provenance.json", *MAP_ASSETS), strict=True):
        _require(keys(row, {"path", "sha256", "size_bytes"}) and row["path"] == expected and digest(row["sha256"])
                 and type(row["size_bytes"]) is int and 0 < row["size_bytes"] <= 5_000_000, code)
    tile = maps["pmtiles"]
    _require(keys(tile, {"artifact_url", "declared_archive_sha256", "declared_size_bytes", "full_archive_rehashed", "range_status", "content_range", "range_size_bytes", "range_sha256", "scope"})
             and tile["artifact_url"] == "/data/california-coast-v1.pmtiles" and digest(tile["declared_archive_sha256"])
             and type(tile["declared_size_bytes"]) is int and tile["declared_size_bytes"] > 1024
             and tile["full_archive_rehashed"] is False and type(tile["range_status"]) is int and tile["range_status"] == 206
             and type(tile["range_size_bytes"]) is int and tile["range_size_bytes"] == 1024 and digest(tile["range_sha256"])
             and tile["content_range"] == f"bytes 0-1023/{tile['declared_size_bytes']}"
             and tile["scope"] == "declared_archive_identity_and_observed_prefix_only", code)
    osrm = record["osrm"]
    _require(keys(osrm, {"observed_at", "provenance_sha256", "container_id", "running", "configured_image", "image_id", "command", "data_mount", "ports", "pbf_sha256", "pbf_full_rehashed", "graph_build_attested"})
             and digest(osrm["provenance_sha256"]) and digest(osrm["pbf_sha256"]) and digest(osrm["container_id"])
             and osrm["running"] is True and osrm["pbf_full_rehashed"] is True and osrm["graph_build_attested"] is False
             and type(osrm["image_id"]) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", osrm["image_id"]) is not None
             and osrm["configured_image"] == "ghcr.io/project-osrm/osrm-backend@" + osrm["image_id"]
             and osrm["command"] == ["osrm-routed", "--algorithm", "mld", "/data/california.osrm"]
             and keys(osrm["data_mount"], {"Source", "Destination", "RW"}) and osrm["data_mount"]["Destination"] == "/data"
             and osrm["data_mount"]["RW"] is False and type(osrm["data_mount"]["Source"]) is str
             and Path(osrm["data_mount"]["Source"]).resolve() == (ROOT / "docker/osrm/data").resolve()
             and osrm["ports"] == {"5000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5000"}]}, code)
