import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from itinerary_system.data import (
    CatalogBundle,
    ContextBundle,
    SnapshotTableMissing,
    load_context_bundle,
    load_dataset_bundle,
    validate_dataset_bundle,
)
from itinerary_system.data.context import CONTEXT_TABLES
from itinerary_system.data.snapshot import CATALOG_TABLES


def copy_data_tree(target_root: Path, *, missing_context_tables: tuple[str, ...] = ()) -> None:
    data_dir = target_root / "data"
    shutil.copytree(REPO_ROOT / "data" / "snapshots", data_dir / "snapshots")
    ignore = shutil.ignore_patterns(*missing_context_tables) if missing_context_tables else None
    shutil.copytree(REPO_ROOT / "data" / "contexts", data_dir / "contexts", ignore=ignore)


class ContextSnapshotTests(unittest.TestCase):
    def test_clean_clone_loads_separated_catalog_and_context(self):
        bundle = load_dataset_bundle(root=REPO_ROOT)
        report = validate_dataset_bundle(bundle)

        self.assertIsInstance(bundle.catalog, CatalogBundle)
        self.assertIsInstance(bundle.context, ContextBundle)
        self.assertEqual(bundle.catalog_snapshot_id, "california_v1")
        self.assertEqual(bundle.context_snapshot_id, "context_static_demo_2026_06")
        self.assertEqual(bundle.manifest["snapshot_schema_version"], "catalog-manifest-v1")
        self.assertEqual(bundle.context_manifest["context_schema_version"], "context-manifest-v1")
        self.assertFalse(bundle.context.legacy_combined_snapshot)
        self.assertEqual(set(bundle.catalog.tables), set(CATALOG_TABLES))
        self.assertEqual(set(bundle.context.tables), set(CONTEXT_TABLES))
        self.assertEqual(report.errors, ())
        self.assertFalse(any("legacy combined" in warning for warning in report.warnings))

    def test_mismatched_context_id_blocks_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            copy_data_tree(root)
            source = root / "data" / "contexts" / "context_static_demo_2026_06"
            target = root / "data" / "contexts" / "context_bad_id"
            shutil.copytree(source, target)
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["context_snapshot_id"] = "context_bad_id"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            bundle = load_dataset_bundle(root=root, context_snapshot_id="context_bad_id")
            report = validate_dataset_bundle(bundle)

        self.assertIn("weather_scenarios context_snapshot_id does not match bundle context", report.errors)
        self.assertIn("route_options context_snapshot_id does not match bundle context", report.errors)
        self.assertFalse(report.can_optimize)

    def test_missing_context_table_raises_typed_snapshot_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            copy_data_tree(root, missing_context_tables=("route_options.csv",))

            with self.assertRaises(SnapshotTableMissing):
                load_context_bundle("context_static_demo_2026_06", root=root)

    def test_invalid_context_hash_blocks_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            copy_data_tree(root)
            manifest_path = root / "data" / "contexts" / "context_static_demo_2026_06" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["weather_scenarios.csv"] = "bad_hash"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            bundle = load_dataset_bundle(root=root)
            report = validate_dataset_bundle(bundle)

        self.assertIn("context manifest hash mismatch: weather_scenarios.csv", report.errors)
        self.assertFalse(report.can_optimize)

    def test_legacy_combined_snapshot_loads_with_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot_target = root / "data" / "snapshots" / "california_v1"
            snapshot_target.parent.mkdir(parents=True)
            shutil.copytree(REPO_ROOT / "data" / "snapshots" / "california_v1", snapshot_target)
            manifest_path = snapshot_target / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["context_snapshot_id"] = "context_static_demo_2026_06"
            manifest["context_tables"] = ["weather_scenarios.csv", "route_options.csv"]
            manifest.setdefault("files", {})["weather_scenarios.csv"] = (
                "760e05f2e2b8f64143767131060d587153cda47a9f76427c26b0b0278fe5fa0f"
            )
            manifest.setdefault("files", {})["route_options.csv"] = (
                "0238d0c2fc71c6c818c20deecb47cba9729c1835944f93004b27b751a8355e31"
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            bundle = load_dataset_bundle(root=root)
            report = validate_dataset_bundle(bundle)

        self.assertTrue(bundle.context.legacy_combined_snapshot)
        self.assertEqual(report.errors, ())
        self.assertTrue(any("legacy combined catalog snapshot" in warning for warning in report.warnings))


if __name__ == "__main__":
    unittest.main()
