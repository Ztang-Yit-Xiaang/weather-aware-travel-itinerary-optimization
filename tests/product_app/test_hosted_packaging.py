"""Offline checks of the real Docker inputs, frozen evidence, and IaC graph."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


def _in_context(relative: str) -> bool:
    # The packaging allowlist uses literal paths, **, and suffix globs only.
    candidates = [relative, *(str(path) for path in PurePosixPath(relative).parents if str(path) != ".")]
    included = True
    for rule in (ROOT / ".dockerignore").read_text().splitlines():
        if not rule or rule.startswith("#"):
            continue
        negate = rule.startswith("!")
        pattern = rule.removeprefix("!").rstrip("/")
        if any(fnmatch.fnmatchcase(candidate, pattern) for candidate in candidates):
            included = negate
    return included


@pytest.fixture(scope="module")
def packaged_root(tmp_path_factory):
    target = tmp_path_factory.mktemp("hosted-image-inputs")
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        if not line.startswith("COPY "):
            continue
        *sources, destination = json.loads(line.removeprefix("COPY "))
        if destination.startswith("/"):
            continue  # Requirements go outside /app; they are checked separately.
        for source in sources:
            path = ROOT / source
            assert path.exists(), source
            for item in path.rglob("*") if path.is_dir() else [path]:
                if not item.is_file() or not _in_context(item.relative_to(ROOT).as_posix()):
                    continue
                assert not item.is_symlink()
                relative = item.relative_to(path) if path.is_dir() else Path(item.name)
                output = target / destination / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item, output)
    return target


@pytest.mark.parametrize(
    "relative",
    [
        ".env",
        ".git/config",
        ".product_app_state/sessions/private.json",
        "docs/report.md",
        "tests/product_app/test_hosted.py",
        "notebook/research.ipynb",
        "tmp/private/session.json",
        "docker/maplibre/assets/california.pmtiles",
        "data/raw/california.osm.pbf",
        "src/itinerary_system/.env.production",
        "src/itinerary_system/__pycache__/api.pyc",
        "configs/interest_regions/sessions/private.json",
        "configs/interest_regions/data.sqlite3",
        "runs/e3ux-weather-repair-demo-v6/cache/provider.json",
        "results/route_deltas/product_pfeiffer_point_lobos_20260908_v1/raw_cache/response.json",
    ],
)
def test_build_context_excludes_state_secrets_and_research_inputs(relative):
    assert not _in_context(relative)


def test_image_retains_every_fingerprinted_python_source_and_local_launcher(packaged_root):
    from itinerary_system.product_app.runtime import product_build_id

    for original in (ROOT / "src").rglob("*.py"):
        if "__pycache__" not in original.parts:
            assert (packaged_root / original.relative_to(ROOT)).read_bytes() == original.read_bytes()
    for name in ["run_hosted_app.py", "run_product_app.py"]:
        assert (packaged_root / "scripts" / name).read_bytes() == (ROOT / "scripts" / name).read_bytes()
    assert product_build_id(packaged_root) == product_build_id(ROOT)


def _sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def test_registered_frozen_packages_are_complete_and_keep_their_hashes(packaged_root):
    from itinerary_system.product_app.registry import ProductRunRegistry

    registry = ProductRunRegistry(packaged_root, packaged_root / "configs/product_app_registry.json")
    assert len(registry.all()) == 3
    for record in registry.all():
        directory = registry.run_dir(record.run_id)
        manifest = json.loads((directory / "manifest.json").read_text())
        for relative, expected in manifest.get("artifacts_sha256", {}).items():
            assert _sha256(directory / relative) == expected, relative
        for paths in manifest.get("artifacts", {}).values():
            for relative in paths:
                assert (directory / relative).is_file(), relative


def test_registered_delta_and_catalogs_load_without_local_caches(packaged_root):
    from itinerary_system.product_app.product_demo import load_product_demo_package
    from itinerary_system.product_app.route_delta import configured_route_matrix

    package = load_product_demo_package(packaged_root, packaged_root / "runs/california-coast-coherent-demo-v1")
    base = package.primary_bundle.route_matrix
    combined = configured_route_matrix(packaged_root, "california_coast_coherent_demo_v1", base)
    assert combined["matrix_id"].startswith("route_matrix_delta_")
    assert len(combined["cells"]) > len(base["cells"])
    for directory in ("interest_regions", "product_poi_catalogs"):
        for original in (ROOT / "configs" / directory).rglob("*"):
            if original.is_file():
                assert (packaged_root / original.relative_to(ROOT)).read_bytes() == original.read_bytes()
    for relative in ("maplibre-gl.js", "maplibre-gl.css", "fonts/Noto Sans Regular/0-255.pbf"):
        assert (packaged_root / "src/itinerary_system/product_app/static/vendor" / relative).is_file()
    assert not list(packaged_root.rglob("raw_cache"))
    for path in packaged_root.rglob("*.csv"):
        with path.open("rb") as handle:
            assert not handle.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"), path


def test_hosted_requirements_are_pinned_and_compatible_with_project_constraints():
    requirements = [
        Requirement(line)
        for line in (ROOT / "deploy/requirements-hosted.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    declared = {r.name.lower(): r for r in map(Requirement, project["project"]["dependencies"])}
    assert {r.name.lower() for r in requirements} == {
        "fastapi",
        "uvicorn",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "geopy",
        "requests",
        "shapely",
        "pydantic",
        "openai",
        "psycopg",
        "pyyaml",
    }
    for requirement in requirements:
        specs = list(requirement.specifier)
        assert len(specs) == 1 and specs[0].operator == "==" and "*" not in specs[0].version
        if requirement.name.lower() in declared:
            assert specs[0].version in declared[requirement.name.lower()].specifier
        if requirement.name == "psycopg":
            assert requirement.extras == {"binary"}


def test_packaged_launcher_configuration_check_needs_no_research_dependencies_or_network(packaged_root):
    # Import in a fresh process so pytest's loaded modules cannot hide missing imports.
    program = """
import importlib.abc, runpy, socket, sys
class NoResearch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'notebook','jupyter','xgboost','gurobipy','ortools','sklearn','folium','openrouteservice','orjson'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, NoResearch())
def offline(*args, **kwargs):
    raise AssertionError('network unavailable in packaging validation')
socket.socket.connect = offline
socket.create_connection = offline
sys.path.insert(0, sys.argv[1] + '/src')
script = sys.argv[1] + '/scripts/run_hosted_app.py'
sys.argv = [script, '--check']
runpy.run_path(script, run_name='__main__')
"""
    environment = {
        **os.environ,
        "HOSTED_API_ORIGIN": "https://api.example.com",
        "HOSTED_FRONTEND_ORIGIN": "https://example.github.io",
        "HOSTED_OWNER_ID": "11111111-1111-4111-8111-111111111111",
        "SUPABASE_URL": "https://packaging.supabase.co",
        "SUPABASE_PUBLISHABLE_KEY": "sb_publishable_packaging-test",
        "DATABASE_URL": "postgresql://test:unused@db.invalid/packaging?sslmode=require",
        "ORS_API_KEY": "offline-test-key",
        "PRODUCT_COPILOT_ADAPTER": "deterministic",
        "HOSTED_ENABLE_PAID_OPENAI": "false",
        "WEB_CONCURRENCY": "1",
        "PORT": "9087",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
    }
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, str(packaged_root)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no resources or provider calls" in result.stdout


def test_docker_entrypoint_uses_nonroot_single_worker_and_dedicated_runtime_requirements():
    source = (ROOT / "Dockerfile").read_text()
    assert 'CMD ["python", "scripts/run_hosted_app.py"]' in source
    assert "USER 10001:10001" in source and "WEB_CONCURRENCY=1" in source
    assert "--only-binary=:all:" in source and "-r /tmp/requirements-hosted.txt" in source
    assert "pip install -e" not in source and "ARG " not in source


def test_railway_configuration_evaluates_to_one_service_with_preserved_external_settings():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js required to evaluate the TypeScript file's JavaScript subset")
    # Exercise the declared graph without the Railway CLI, accounts, or plan/apply.
    script = r"""
const fs = require('node:fs'), vm = require('node:vm');
let text = fs.readFileSync(process.argv[1], 'utf8');
text = text.replace(/^import .* from "railway\/iac";$/m, '').replace('export default', 'result =');
const scope = {defineRailway: f => f(), github: (repository, options) => ({repository, ...options}),
  preserve: () => ({preserve: true}), project: (name, config) => ({name, ...config}),
  service: (name, config) => ({name, ...config})};
vm.runInNewContext(text, scope, {timeout: 1000});
process.stdout.write(JSON.stringify(scope.result));
"""
    result = subprocess.run(
        [node, "-e", script, str(ROOT / ".railway/railway.ts")], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    graph = json.loads(result.stdout)
    assert len(graph["resources"]) == 1
    api = graph["resources"][0]
    assert api["source"] == {
        "repository": "Ztang-Yit-Xiaang/weather-aware-travel-itinerary-optimization",
        "branch": "codex/deploy-latest-travel-site",
    }
    assert api["replicas"] == 1 and api["healthcheck"] == "/healthz"
    assert api["start"] == "python scripts/run_hosted_app.py" and "build" not in api
    assert api["env"]["WEB_CONCURRENCY"] == "1"
    assert api["env"]["PRODUCT_COPILOT_ADAPTER"] == "deterministic"
    assert api["env"]["HOSTED_ENABLE_PAID_OPENAI"] == "false"
    for name in (
        "HOSTED_API_ORIGIN",
        "HOSTED_FRONTEND_ORIGIN",
        "HOSTED_OWNER_ID",
        "DATABASE_URL",
        "SUPABASE_URL",
        "SUPABASE_PUBLISHABLE_KEY",
        "ORS_API_KEY",
    ):
        assert api["env"][name] == {"preserve": True}
