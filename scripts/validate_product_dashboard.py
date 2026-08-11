"""Validate product dashboard integrity, security, and compatibility contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from itinerary_system.product_dashboard_models import (  # noqa: E402
    PRODUCT_DASHBOARD_DATA_SCHEMA,
    PRODUCT_DASHBOARD_SCHEMA,
    TRUTH_STATE_LABELS,
    assert_finite_json,
    file_sha256,
    safe_run_relative_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one dashboard_product artifact.")
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def validate_product_dashboard(run_dir: Path) -> list[str]:
    """Return contract violations; an empty list means validation passed."""

    run_dir = Path(run_dir).resolve()
    product_dir = run_dir / "dashboard_product"
    errors: list[str] = []
    manifest = _read_json(product_dir / "manifest.json", errors)
    run_manifest = _read_json(run_dir / "manifest.json", errors)
    if manifest is None or run_manifest is None:
        return errors

    if manifest.get("schema_version") != PRODUCT_DASHBOARD_SCHEMA:
        errors.append("unsupported product manifest schema")
    if not str(manifest.get("product_version") or ""):
        errors.append("product version is missing")
    registered = (run_manifest.get("artifacts") or {}).get("dashboard_product") or []
    if registered != ["dashboard_product/manifest.json"]:
        errors.append("top-level run manifest does not register the product manifest")
    if manifest.get("run_id") != run_manifest.get("run_id"):
        errors.append("product/run manifest run_id mismatch")

    for path_text, expected in (manifest.get("source_content_hashes") or {}).items():
        try:
            path = safe_run_relative_path(run_dir, str(path_text))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.is_file():
            errors.append(f"missing source artifact: {path_text}")
        elif file_sha256(path) != expected:
            errors.append(f"source artifact hash mismatch: {path_text}")

    for path_text, expected in (manifest.get("asset_hashes") or {}).items():
        try:
            path = safe_run_relative_path(product_dir, str(path_text))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.is_file():
            errors.append(f"missing product asset: {path_text}")
        elif file_sha256(path) != expected:
            errors.append(f"product asset hash mismatch: {path_text}")

    html = _read_text(product_dir / "index.html", errors)
    css = _read_text(product_dir / "assets" / "product.css", errors)
    ui_js = _read_text(product_dir / "assets" / "product_ui.js", errors)
    map_js = _read_text(product_dir / "assets" / "product_map.js", errors)
    data_js = _read_text(product_dir / "assets" / "product_data.js", errors)
    for token in (
        'name="viewport"',
        '<main id="product-main"',
        'aria-live="polite"',
        'data-mode="customer"',
        'data-mode="research"',
        'role="img"',
        "Text alternative for the map",
        "Read-only",
    ):
        if token not in html:
            errors.append(f"product HTML missing token: {token}")
    for token in (
        ":focus-visible",
        "@media (min-width: 720px)",
        "@media (min-width: 1080px)",
        "@media (prefers-reduced-motion: reduce)",
        "min-height: 44px",
        "grid-template-areas",
    ):
        if token not in css:
            errors.append(f"product CSS missing contract: {token}")
    for token in (
        "textContent",
        "createElement",
        "product-day-selected",
        "aria-pressed",
        "setMode",
    ):
        if token not in ui_js:
            errors.append(f"product UI runtime missing contract: {token}")
    for forbidden in ("innerHTML", "eval(", "localStorage", "sessionStorage", "document.write"):
        if forbidden in ui_js or forbidden in map_js:
            errors.append(f"product runtime uses forbidden API: {forbidden}")
    for forbidden_label in ("Accept repair", "Keep original", "Ask permission", "Clarify meaning"):
        if forbidden_label in html or forbidden_label in ui_js:
            errors.append(f"E3.UX5 control leaked into read-only product UI: {forbidden_label}")

    data = _parse_data_script(data_js, errors)
    if data is not None:
        if data.get("schema_version") != PRODUCT_DASHBOARD_DATA_SCHEMA:
            errors.append("unsupported product data schema")
        try:
            assert_finite_json(data)
        except ValueError as exc:
            errors.append(str(exc))
        catalog = {state.get("id") for state in data.get("truth_state_catalog") or []}
        missing_states = set(TRUTH_STATE_LABELS) - catalog
        if missing_states:
            errors.append(f"truth-state catalog incomplete: {sorted(missing_states)}")
        for row in data.get("alternatives") or []:
            if not row.get("ranking_eligible") and row.get("rank") is not None:
                errors.append("ineligible alternative received a rank")
        if data.get("interaction", {}).get("enabled") is not False:
            errors.append("read-only product artifact enabled interaction")
        if data.get("run", {}).get("interaction_enabled") is not False:
            errors.append("run view model enabled interaction")
        if manifest.get("parent_plan_id") != data.get("research", {}).get("lineage", {}).get(
            "parent_plan_id"
        ):
            errors.append("product manifest/data parent lineage mismatch")
        if manifest.get("child_plan_id") != data.get("research", {}).get("lineage", {}).get(
            "child_plan_id"
        ):
            errors.append("product manifest/data child lineage mismatch")
    return errors


def _parse_data_script(text: str, errors: list[str]) -> dict[str, Any] | None:
    prefix = "window.PRODUCT_DASHBOARD_DATA = "
    if not text.startswith(prefix) or not text.rstrip().endswith(";"):
        errors.append("product data asset has an invalid wrapper")
        return None
    raw = text[len(prefix) :].strip()
    raw = raw[:-1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors.append(f"product data asset is malformed: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append("product data root is not an object")
        return None
    return data


def _read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    text = _read_text(path, errors)
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON {path}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"JSON root is not an object: {path}")
        return None
    return value


def _read_text(path: Path, errors: list[str]) -> str:
    if not path.is_file():
        errors.append(f"missing file: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def main() -> int:
    args = parse_args()
    errors = validate_product_dashboard(args.run_dir)
    if errors:
        print("Product dashboard validation FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Product dashboard validation PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
