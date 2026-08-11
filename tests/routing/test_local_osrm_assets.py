from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OSRM_DIR = ROOT / "docker" / "osrm"


def test_local_osrm_assets_enforce_pinned_private_service():
    compose = (OSRM_DIR / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (OSRM_DIR / ".env.example").read_text(encoding="utf-8")
    preprocess = (OSRM_DIR / "scripts" / "preprocess.sh").read_text(encoding="utf-8")

    assert "127.0.0.1:" in compose
    assert "--algorithm" in compose and "mld" in compose
    assert "@sha256:" in env_example
    assert "OSM_PBF_SHA256" in env_example
    assert "*:latest" in preprocess
    assert "registry@sha256:digest" in preprocess
    assert "sha256sum" in preprocess


def test_local_osrm_large_graph_data_is_ignored_and_healthchecked():
    data_ignore = (OSRM_DIR / "data" / ".gitignore").read_text(encoding="utf-8").splitlines()
    healthcheck = (OSRM_DIR / "scripts" / "healthcheck.sh").read_text(encoding="utf-8")
    readme = (OSRM_DIR / "README.md").read_text(encoding="utf-8")

    assert data_ignore == ["*", "!.gitignore"]
    assert '"code":"Ok"' in healthcheck
    assert "--require-complete" in readme
    assert "source-provenance.json" in readme
