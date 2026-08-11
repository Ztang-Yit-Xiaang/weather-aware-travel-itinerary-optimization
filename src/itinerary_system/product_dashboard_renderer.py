"""Render the additive, read-only product dashboard artifact."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .product_dashboard_adapter import load_product_dashboard_source
from .product_dashboard_assets import (
    product_dashboard_html,
    product_dashboard_map_script,
    product_dashboard_stylesheet,
    product_dashboard_ui_script,
)
from .product_dashboard_models import (
    ProductDashboardManifest,
    ProductDashboardValidationError,
    file_sha256,
    json_text,
)
from .product_dashboard_view_models import build_product_dashboard_view_model


def render_product_dashboard(run_dir: Path) -> ProductDashboardManifest:
    """Build `dashboard_product/` once and register it in a new run manifest."""

    run_dir = Path(run_dir).resolve()
    output_dir = run_dir / "dashboard_product"
    if output_dir.exists():
        raise FileExistsError(f"product dashboard already exists: {output_dir}")

    bundle = load_product_dashboard_source(run_dir)
    view_model = build_product_dashboard_view_model(bundle)
    with tempfile.TemporaryDirectory(prefix=".dashboard_product_build_", dir=run_dir) as temp_text:
        staging = Path(temp_text)
        assets = staging / "assets"
        assets.mkdir()
        _write_text(staging / "index.html", product_dashboard_html())
        _write_text(assets / "product.css", product_dashboard_stylesheet())
        _write_text(assets / "product_ui.js", product_dashboard_ui_script())
        _write_text(assets / "product_map.js", product_dashboard_map_script())
        _write_text(assets / "product_data.js", _data_script(view_model))

        asset_paths = (
            staging / "index.html",
            assets / "product.css",
            assets / "product_data.js",
            assets / "product_ui.js",
            assets / "product_map.js",
        )
        asset_hashes = {
            path.relative_to(staging).as_posix(): file_sha256(path) for path in asset_paths
        }
        requested = tuple(
            dict.fromkeys(
                str(record.get("method_requested") or "")
                for record in bundle.planner_runs
                if record.get("method_requested")
            )
        )
        executed = tuple(
            dict.fromkeys(
                str(record.get("method_executed") or "")
                for record in bundle.planner_runs
                if record.get("method_executed")
            )
        )
        manifest = ProductDashboardManifest(
            run_id=str(bundle.run_manifest.get("run_id") or ""),
            source_artifact_refs=bundle.source_paths,
            source_content_hashes=bundle.source_hashes,
            parent_plan_id=str(bundle.parent_plan.get("plan_id") or ""),
            parent_content_hash=str(bundle.parent_plan.get("content_hash") or ""),
            child_plan_id=(bundle.child_plan or {}).get("plan_id"),
            child_content_hash=(bundle.child_plan or {}).get("content_hash"),
            diff_id=(bundle.diff or {}).get("diff_id"),
            certificate_id=(bundle.certificate or {}).get("certificate_id"),
            explanation_ids=tuple(
                str(record.get("evidence_id") or "")
                for record in bundle.explanations
                if record.get("evidence_id")
            ),
            route_matrix_id=(bundle.route_matrix or {}).get("matrix_id"),
            requested_method_ids=requested,
            executed_method_ids=executed,
            run_status=str(bundle.run_manifest.get("status") or "unknown"),
            truth_states=bundle.truth_states,
            asset_hashes=asset_hashes,
            generated_at=datetime.now(UTC).isoformat(),
        )
        _write_text(staging / "manifest.json", json_text(manifest.to_record()) + "\n")
        os.replace(staging, output_dir)

    updated_run_manifest = _register_product_artifact(
        run_dir=run_dir,
        run_manifest=bundle.run_manifest,
        product_manifest=manifest,
    )
    if "dashboard_product/manifest.json" not in updated_run_manifest["artifacts"]["dashboard_product"]:
        raise ProductDashboardValidationError(["product_manifest_registration_failed"])
    return manifest


def register_product_dashboard_screenshots(run_dir: Path) -> dict[str, str]:
    """Hash browser screenshots and add them to the product manifest."""

    run_dir = Path(run_dir).resolve()
    product_dir = run_dir / "dashboard_product"
    manifest_path = product_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    screenshot_dir = product_dir / "screenshots"
    screenshots = sorted(screenshot_dir.glob("*.png"))
    if not screenshots:
        raise ProductDashboardValidationError(["missing_product_screenshots"])
    hashes = {
        path.relative_to(product_dir).as_posix(): file_sha256(path)
        for path in screenshots
    }
    assets = manifest.setdefault("asset_hashes", {})
    overlap = set(hashes) & set(assets)
    if overlap:
        raise ProductDashboardValidationError(
            [f"screenshot_hash_already_registered:{path}" for path in sorted(overlap)]
        )
    assets.update(hashes)
    _write_json_atomic(manifest_path, manifest)
    return hashes


def _register_product_artifact(
    *,
    run_dir: Path,
    run_manifest: dict[str, Any],
    product_manifest: ProductDashboardManifest,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(run_manifest))
    artifacts = updated.setdefault("artifacts", {})
    if artifacts.get("dashboard_product"):
        raise ProductDashboardValidationError(["dashboard_product_manifest_entry_already_exists"])
    artifacts["dashboard_product"] = ["dashboard_product/manifest.json"]
    counts = updated.setdefault("artifact_counts", {})
    counts["dashboard_product"] = 1
    updated["product_dashboard"] = {
        "schema_version": product_manifest.schema_version,
        "product_version": product_manifest.product_version,
        "manifest_path": "dashboard_product/manifest.json",
        "compatibility_boundary": product_manifest.compatibility_boundary,
    }
    target = run_dir / "manifest.json"
    _write_json_atomic(target, updated)
    return updated


def _write_json_atomic(target: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=target.parent,
        prefix=".manifest.",
        suffix=".tmp",
    ) as handle:
        handle.write(json_text(payload))
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, target)


def _data_script(view_model: dict[str, Any]) -> str:
    payload = json.dumps(
        view_model,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    payload = (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return f"window.PRODUCT_DASHBOARD_DATA = {payload};\n"


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
