from __future__ import annotations

import json

import pytest

from itinerary_system.product_dashboard_renderer import (
    register_product_dashboard_screenshots,
    render_product_dashboard,
)
from scripts.validate_product_dashboard import validate_product_dashboard


def test_render_registers_separately_versioned_artifact(product_run_factory) -> None:
    run_dir = product_run_factory()
    product = render_product_dashboard(run_dir)
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert product.product_version == "1.0.0"
    assert run_manifest["artifacts"]["dashboard_product"] == [
        "dashboard_product/manifest.json"
    ]
    assert (run_dir / "dashboard_product/index.html").is_file()
    assert validate_product_dashboard(run_dir) == []


def test_render_is_non_overwritable(product_run_factory) -> None:
    run_dir = product_run_factory()
    render_product_dashboard(run_dir)

    with pytest.raises(FileExistsError):
        render_product_dashboard(run_dir)


def test_embedded_data_escapes_script_breakout(product_run_factory) -> None:
    run_dir = product_run_factory()
    request_path = run_dir / "requests/request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["user_intent"] = "</script><script>alert(1)</script>"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    render_product_dashboard(run_dir)
    data = (run_dir / "dashboard_product/assets/product_data.js").read_text(
        encoding="utf-8"
    )

    assert "</script>" not in data
    assert "\\u003c/script\\u003e" in data


def test_validator_detects_asset_tampering(product_run_factory) -> None:
    run_dir = product_run_factory()
    render_product_dashboard(run_dir)
    css = run_dir / "dashboard_product/assets/product.css"
    css.write_text(css.read_text(encoding="utf-8") + "\n.tampered{}", encoding="utf-8")

    assert any("hash mismatch" in error for error in validate_product_dashboard(run_dir))


def test_read_only_artifact_excludes_ux5_controls(product_run_factory) -> None:
    run_dir = product_run_factory()
    render_product_dashboard(run_dir)
    html = (run_dir / "dashboard_product/index.html").read_text(encoding="utf-8")
    ui = (run_dir / "dashboard_product/assets/product_ui.js").read_text(
        encoding="utf-8"
    )

    for label in ("Accept repair", "Keep original", "Ask permission", "Clarify meaning"):
        assert label not in html
        assert label not in ui


def test_map_controller_synchronizes_route_and_ownership_states(
    product_run_factory,
) -> None:
    run_dir = product_run_factory()
    render_product_dashboard(run_dir)
    map_script = (run_dir / "dashboard_product/assets/product_map.js").read_text(
        encoding="utf-8"
    )

    assert "routeLines.forEach" in map_script
    assert "line.setStyle(routeStyle" in map_script
    assert "marker.productStrength" in map_script
    assert "Affected-day evidence" in map_script


def test_generated_output_does_not_expose_host_paths(product_run_factory) -> None:
    run_dir = product_run_factory()
    render_product_dashboard(run_dir)
    data = (run_dir / "dashboard_product/assets/product_data.js").read_text(
        encoding="utf-8"
    )

    assert str(run_dir) not in data
    assert "C:\\\\" not in data
    assert "F:\\\\" not in data


def test_browser_screenshots_receive_manifest_hashes(product_run_factory) -> None:
    run_dir = product_run_factory()
    render_product_dashboard(run_dir)
    screenshot = run_dir / "dashboard_product/screenshots/mobile.png"
    screenshot.parent.mkdir()
    screenshot.write_bytes(b"\x89PNG\r\n\x1a\ncontract-test")

    hashes = register_product_dashboard_screenshots(run_dir)

    assert "screenshots/mobile.png" in hashes
    assert validate_product_dashboard(run_dir) == []
