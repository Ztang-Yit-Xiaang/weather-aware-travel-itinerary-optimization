"""Every normal runner mode reaches complete-manifest checks with fixture I/O.

These are explicit manifest-contract unit fixtures, not executed browser runs.
Self-managed modes have a separate manifest and lifecycle verifier.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_e2e1_aggregate as aggregate  # noqa: E402
import verify_e2e1_vertical_slice as independent  # noqa: E402
from test_e2e1_runner import _evidence, runner  # noqa: E402

SPECS = {
    "vertical-slice": (runner.SCENARIOS, sorted([f"{name}/llm_io.json" for _, name in runner.SCENARIOS]
        + ["registered_replacement/one_click_llm_io.json"])),
    "map-recovery": (runner.MAP_RECOVERY_SCENARIOS, [f"map_initialization_recovery/{context}/llm_io.json"
        for context in ("accepted_child", "original")]),
    "completed-response-race": (runner.COMPLETED_RACE_SCENARIOS, ["completed_preview_race/llm_io.json"]),
    "workspace-stability": (runner.WORKSPACE_STABILITY_SCENARIOS, sorted(f"workspace_stability/{size}/llm_io.json"
        for size in ("1440x900", "1280x800", "1024x768", "390x844"))),
    "direct-edits": (runner.DIRECT_EDITS_SCENARIOS, sorted(f"direct_edits/{kind}/llm_io.json"
        for kind in ("combined", "duration", "time_window", "role", "order"))),
    "addition-continuation": (runner.ADDITION_SCENARIOS, ["addition_continuation/llm_io.json"]),
    "mobile-chat": (runner.MOBILE_CHAT_SCENARIOS, ["mobile_chat/llm_io.json"]),
    "waypoint-pointer": (runner.WAYPOINT_SCENARIOS, ["waypoint_pointer/llm_io.json"]),
    "capability-controls": (runner.CONTROLS_SCENARIOS, ["capability_controls/accepted_child/llm_io.json", "capability_controls/original/llm_io.json"]),
    "evaluated-variants": (runner.EVALUATED_SCENARIOS, ["evaluated_variants/accepted_child/llm_io.json", "evaluated_variants/original/llm_io.json"]),
    "webgl-recovery": (runner.WEBGL_SCENARIOS, ["webgl_recovery/llm_io.json"]),
    "chat-closure": (runner.CHAT_CLOSURE_SCENARIOS, ["chat_closure/accepted_child/llm_io.json", "chat_closure/original/llm_io.json"]),
    "same-plan-remount": (runner.REMOUNT_SCENARIOS, ["same_plan_remount/llm_io.json"]),
    "waypoint-contexts": (runner.WAYPOINT_CONTEXTS_SCENARIOS, ["waypoint_contexts/accepted_child/llm_io.json", "waypoint_contexts/original/llm_io.json"]),
    "candidate-day-matrix": (runner.CANDIDATE_DAYS_SCENARIOS, ["candidate_day_matrix/llm_io.json"]),
}
NORMAL = {mode: filename for mode, filename, _ in aggregate.SUITES if mode not in {"conversation", "combined-conflicts", "owned-edits", "build-coherence", "bootstrap-recovery", "optional-evidence-fault", "preview-geography-fault", "queued-preview"}}


def fixture(tmp_path, mode, *, corrupt=None):
    seed, output = tmp_path / "unit_seed", tmp_path / "unit_manifest"
    seed.mkdir()
    output.mkdir()
    _evidence(seed)
    (output / "provider_isolation.json").write_bytes((seed / "provider_isolation.json").read_bytes())
    scenarios, paths = SPECS[mode]
    for relative in paths:
        record = {"llm_input": None, "llm_output": None, "provider_call_count": 0,
            "reason": "Manifest-contract unit fixture only; no browser or provider was run."}
        if relative == "unsupported_removal/llm_io.json":
            record["verification_test"] = independent.UNSUPPORTED_TEST
        if relative.startswith("chat_closure/"):
            record["actual_openai_connectivity_claimed"] = False
        if corrupt:
            corrupt(record)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record), encoding="utf-8")
    browser = "tests/product_app/browser/" + NORMAL[mode]
    manifest = runner.build_manifest(output, command=["python", "-m", "pytest", browser, "-q", "-p", "no:cacheprovider"],
        pytest_exit_code=0, health={"build_id": "unit_fixture", "components": {"map": {"code": "maplibre_ready"},
            "routing": {"code": "runtime_osrm_ready"}, "openai": {"code": "deterministic_adapter_selected"}}},
        base_url="http://127.0.0.1:8127", run_id="unit-fixture", server_pid=123,
        source_inputs=runner._source_rows(browser), scenarios=tuple(name for _, name in scenarios))
    return output, manifest


def test_manifest_regression_inventory_covers_every_normal_registered_mode():
    assert set(SPECS) == set(NORMAL)


def test_manifest_regression_inventory_also_covers_all_cli_modes(monkeypatch):
    discovered = []
    original = runner.argparse.ArgumentParser.add_argument

    def observe(parser, *args, **kwargs):
        if "--browser-test" in args:
            discovered.extend(kwargs["choices"])
        return original(parser, *args, **kwargs)

    monkeypatch.setattr(runner.argparse.ArgumentParser, "add_argument", observe)
    monkeypatch.setattr(sys, "argv", ["manifest-contract-unit-test"])
    runner.parse_args()
    assert set(discovered) - {"aggregate", "conversation", "combined-conflicts", "owned-edits", "build-coherence", "bootstrap-recovery", "optional-evidence-fault", "preview-geography-fault", "queued-preview"} == set(SPECS)


@pytest.mark.parametrize("mode", sorted(NORMAL))
def test_every_normal_mode_complete_manifest_and_independent_metadata_paths(tmp_path, mode):
    output, manifest = fixture(tmp_path, mode)
    runner.verify_manifest(output, manifest, require_complete=True)
    generated = independent._verify_root(manifest)
    independent._verify_sources(manifest, generated)
    independent._verify_llm_records(output, manifest)
    assert [row["path"] for row in manifest["llm_records"]] == SPECS[mode][1]


@pytest.mark.parametrize("mode", ["chat-closure", "same-plan-remount"])
def test_new_mode_missing_llm_record_is_not_hidden_by_registration(tmp_path, mode):
    output, manifest = fixture(tmp_path, mode)
    manifest["llm_records"] = []
    manifest["content_hash"] = runner._canonical_sha256({key: value for key, value in manifest.items() if key != "content_hash"})
    with pytest.raises(runner.E2E1RunnerError, match="e2e1_llm_record_universe_mismatch"):
        runner.verify_manifest(output, manifest, require_complete=True)
    with pytest.raises(independent.E2E1VerificationError, match="e2e1_llm_universe_invalid"):
        independent._verify_llm_records(output, manifest)


def test_chat_closure_cannot_claim_live_provider_connectivity_in_manifest(tmp_path):
    output, manifest = fixture(tmp_path, "chat-closure", corrupt=lambda record: record.update(actual_openai_connectivity_claimed=True))
    with pytest.raises(runner.E2E1RunnerError, match="e2e1_unexpected_llm_io"):
        runner.verify_manifest(output, manifest, require_complete=True)
    with pytest.raises(independent.E2E1VerificationError, match="e2e1_llm_live_connectivity_falsely_claimed"):
        independent._verify_llm_records(output, manifest)
