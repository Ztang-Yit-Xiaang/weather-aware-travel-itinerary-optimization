from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from itinerary_system.plans import load_plan
from itinerary_system.product_app.product_demo import ProductDemoError, load_product_demo_package

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "build_product_demo.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_product_demo", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_builder_emits_two_distinct_independently_evaluated_children(tmp_path: Path) -> None:
    builder = _load_builder()
    source_run = ROOT / builder.SOURCE_RUN_RELATIVE
    before = {
        relative: hashlib.sha256((source_run / relative).read_bytes()).hexdigest()
        for relative in builder.PINNED_SOURCE_SHA256
    }

    manifest_path = builder.build_product_demo(tmp_path / "demo", repository_root=ROOT)
    package = _json(manifest_path)

    assert package["schema_version"] == "product-demo-package-v1"
    assert package["run_id"] == "california_coast_product_demo_v2"
    assert package["parent"]["content_hash"] == builder.PINNED_PARENT_CONTENT_HASH
    assert package["route_evidence"]["matrix_id"] == builder.PINNED_MATRIX_ID
    assert package["route_evidence"]["matrix_file_sha256"] == builder.PINNED_SOURCE_SHA256[
        builder.MATRIX_RELATIVE.as_posix()
    ]
    declared_hashes = package["artifacts_sha256"]
    actual_paths = {
        path.relative_to(manifest_path.parent).as_posix()
        for path in manifest_path.parent.rglob("*")
        if path.is_file() and path != manifest_path
    }
    assert set(declared_hashes) == actual_paths
    assert all(
        hashlib.sha256((manifest_path.parent / relative).read_bytes()).hexdigest() == expected
        for relative, expected in declared_hashes.items()
    )
    for config_path in manifest_path.parent.rglob("resolved_config.redacted.json"):
        assert _json(config_path)["_source_path"] == "configs/default_trip_config.yaml"
    alternatives = package["alternatives"]
    assert [record["role"] for record in alternatives] == ["recommended", "low_driving"]
    assert len({record["plan_id"] for record in alternatives}) == 2
    assert len({record["plan_content_hash"] for record in alternatives}) == 2

    for record in alternatives:
        plan = load_plan(manifest_path.parent / record["plan_relative_path"])
        certificate = _json(manifest_path.parent / record["certificate_relative_path"])
        assert plan.plan_id == record["plan_id"]
        assert plan.content_hash == record["plan_content_hash"]
        assert plan.parent_plan_id == package["parent"]["plan_id"]
        assert certificate["plan_id"] == plan.plan_id
        assert certificate["plan_content_hash"] == plan.content_hash
        assert certificate["comparison_eligibility"] == "eligible"
        assert certificate["evaluation_status"] in {"PASSED", "PASSED_WITH_WARNINGS"}
        assert certificate["route_validation"]["publication_ready"] is True
        assert certificate["route_validation"]["required_leg_count"] == 16
        assert certificate["route_validation"]["road_validated_leg_count"] == 16
        assert certificate["route_validation"]["fallback_leg_count"] == 0
        assert len(record["route_legs"]) == certificate["route_validation"]["required_leg_count"]
        assert all(
            previous["destination_id"] == current["origin_id"]
            for previous, current in zip(record["route_legs"][:-1], record["route_legs"][1:], strict=False)
        )
        day_four_first = next(leg for leg in record["route_legs"] if leg["day"] == 4)
        assert day_four_first["origin_id"] == "the_line_la"

    by_role = {record["role"]: record for record in alternatives}
    assert by_role["low_driving"]["route_total_minutes"] < by_role["recommended"]["route_total_minutes"]
    after = {
        relative: hashlib.sha256((source_run / relative).read_bytes()).hexdigest()
        for relative in builder.PINNED_SOURCE_SHA256
    }
    assert after == before == builder.PINNED_SOURCE_SHA256

    certificate_path = manifest_path.parent / alternatives[0]["certificate_relative_path"]
    certificate_path.write_text(certificate_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    try:
        load_product_demo_package(ROOT, manifest_path.parent)
    except ProductDemoError as exc:
        assert str(exc) == "product_demo_artifact_hash_mismatch"
    else:
        raise AssertionError("the W2 loader accepted a certificate whose declared hash changed")


def test_builder_is_identity_deterministic_and_non_overwriting(tmp_path: Path) -> None:
    builder = _load_builder()
    first_manifest = builder.build_product_demo(tmp_path / "first", repository_root=ROOT)
    second_manifest = builder.build_product_demo(tmp_path / "second", repository_root=ROOT)

    first = _json(first_manifest)
    second = _json(second_manifest)
    assert first == second
    first_identity = [
        (item["role"], item["plan_id"], item["plan_content_hash"], item["certificate_id"])
        for item in first["alternatives"]
    ]
    second_identity = [
        (item["role"], item["plan_id"], item["plan_content_hash"], item["certificate_id"])
        for item in second["alternatives"]
    ]
    assert first_identity == second_identity

    try:
        builder.build_product_demo(tmp_path / "first", repository_root=ROOT)
    except FileExistsError:
        pass
    else:
        raise AssertionError("the product-demo builder overwrote an existing package")
