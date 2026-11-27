from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "itinerary_system" / "product_app" / "static"


def test_g4r_product_content_remediation_is_fail_closed_and_user_facing() -> None:
    markup = (STATIC / "index.html").read_text(encoding="utf-8")
    app = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    compare = (STATIC / "js" / "compare.js").read_text(encoding="utf-8")
    copilot = (STATIC / "js" / "copilot.js").read_text(encoding="utf-8")
    controller = (STATIC / "js" / "map-controller-v2.js").read_text(encoding="utf-8")

    assert 'id="settings-button"' not in markup
    assert "$(`#settings-button`)" not in app
    assert "Choose option (not enabled)" not in app + compare
    assert "Choose option for Trip preview" in app
    assert "Choose option for Trip preview" in compare
    assert "await onSelectAlternative(resolution.option.plan_id)" in compare
    assert "geographyFeature?.properties?.name || humanLabel(stopId)" in app
    assert "state.session.selected_alternative_id}`" not in copilot
    assert "selectedAlternative?.method_label" in copilot
    assert 'typeof map?.areTilesLoaded !== "function"' in controller
    assert "tilesLoaded: false, tileCoverageRatio: 0" in controller
    assert "container.dataset.mapReadyAt = mapReadyAt" in controller
    assert "container.dataset.mapTileCoverage = String(readiness.tileCoverageRatio)" in controller


def test_map_tile_readiness_executable_contract() -> None:
    completed = subprocess.run(
        ["node", str(Path(__file__).with_name("map_tile_readiness_contract.test.mjs"))],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "map tile readiness contract passed" in completed.stdout
