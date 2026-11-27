"""Safe focused R19 selection; no provider, server, or browser dispatch."""

import json

import pytest
from test_e2e1_runner import _evidence, runner

FOCUSED_LAYOUTS = [
    (
        "direct_edits",
        runner.DIRECT_EDITS_BROWSER_TEST,
        [f"direct_edits/{kind}/llm_io.json" for kind in ("combined", "duration", "order", "role", "time_window")],
    ),
    ("addition_continuation", runner.ADDITION_BROWSER_TEST, ["addition_continuation/llm_io.json"]),
    ("completed_preview_race", runner.COMPLETED_RACE_BROWSER_TEST, ["completed_preview_race/llm_io.json"]),
    (
        "map_initialization_recovery",
        runner.MAP_RECOVERY_BROWSER_TEST,
        [f"map_initialization_recovery/{context}/llm_io.json" for context in ("accepted_child", "original")],
    ),
    (
        "workspace_stability",
        runner.WORKSPACE_STABILITY_BROWSER_TEST,
        [f"workspace_stability/{size}/llm_io.json" for size in ("1024x768", "1280x800", "1440x900", "390x844")],
    ),
]


def _focused_manifest(tmp_path, scenario, browser_test, paths):
    seed = tmp_path / "unit_seed"
    seed.mkdir()
    _evidence(seed)
    output = tmp_path / "focused"
    output.mkdir()
    (output / "provider_isolation.json").write_bytes((seed / "provider_isolation.json").read_bytes())
    for relative in paths:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "llm_input": None,
                    "llm_output": None,
                    "provider_call_count": 0,
                    "reason": "explicit_unit_fixture_not_a_browser_run",
                }
            ),
            encoding="utf-8",
        )
    manifest = runner.build_manifest(
        output,
        command=["python", "-m", "pytest", browser_test, "-q", "-p", "no:cacheprovider"],
        pytest_exit_code=0,
        health={
            "build_id": "unit_fixture",
            "components": {
                "map": {"code": "maplibre_ready"},
                "routing": {"code": "runtime_osrm_ready"},
                "openai": {"code": "deterministic_adapter_selected"},
            },
        },
        base_url="http://127.0.0.1:49152",
        source_inputs=runner._source_rows(browser_test),
        run_id="unit-fixture",
        server_pid=123,
        scenarios=(scenario,),
    )
    return output, manifest


@pytest.mark.parametrize("scenario,browser_test,paths", FOCUSED_LAYOUTS)
def test_positive_focused_manifest_has_exact_complete_layout(tmp_path, scenario, browser_test, paths):
    from verify_e2e1_vertical_slice import _verify_llm_records, _verify_root

    output, manifest = _focused_manifest(tmp_path, scenario, browser_test, paths)
    runner.verify_manifest(output, manifest)
    _verify_root(manifest)
    _verify_llm_records(output, manifest)
    assert [row["path"] for row in manifest["llm_records"]] == paths


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_path", "wrong_scenario"])
def test_focused_manifest_does_not_bypass_exact_evidence_universe(tmp_path, mutation):
    output, manifest = _focused_manifest(tmp_path, *FOCUSED_LAYOUTS[2])
    if mutation == "missing":
        manifest["llm_records"] = []
    elif mutation == "duplicate":
        manifest["llm_records"] *= 2
    elif mutation == "wrong_path":
        manifest["llm_records"][0]["path"] = "registered_replacement/llm_io.json"
    else:
        manifest["scenarios"] = ["unknown_scenario"]
    manifest.pop("content_hash")
    manifest["content_hash"] = runner._canonical_sha256(manifest)
    code = "e2e1_scenario_universe_invalid" if mutation == "wrong_scenario" else "e2e1_llm_record_universe_mismatch"
    with pytest.raises(runner.E2E1RunnerError, match=code):
        runner.verify_manifest(output, manifest)


@pytest.mark.parametrize("mode", ["completed-response-race", "workspace-stability"])
def test_completed_response_race_is_allowlisted_without_changing_default(monkeypatch, mode):
    monkeypatch.setattr(runner.sys, "argv", ["runner", "--browser-test", mode])
    assert runner.parse_args().browser_test == mode
    monkeypatch.setattr(runner.sys, "argv", ["runner"])
    assert runner.parse_args().browser_test == "vertical-slice"
    assert len(runner.SCENARIOS) == 6


@pytest.mark.parametrize("flag", ["live_copilot_chat", "confirm_billed_synthetic_context"])
@pytest.mark.parametrize(
    "mode,error",
    [
        ("completed-response-race", "e2e1_completed_race_is_deterministic_only"),
        ("workspace-stability", "e2e1_workspace_stability_is_deterministic_only"),
    ],
)
def test_completed_response_race_cannot_dispatch_a_billed_runner(monkeypatch, flag, mode, error):
    arguments = {
        "browser_test": mode,
        "output_dir": None,
        "port": 8127,
        "live_copilot_chat": False,
        "confirm_billed_synthetic_context": False,
    }
    arguments[flag] = True
    monkeypatch.setattr(runner, "parse_args", lambda: type("Args", (), arguments)())
    with pytest.raises(runner.E2E1RunnerError, match=error):
        runner.main()


@pytest.mark.parametrize("browser_test", [runner.COMPLETED_RACE_BROWSER_TEST, runner.WORKSPACE_STABILITY_BROWSER_TEST])
def test_completed_response_race_hashes_actual_shared_browser_helpers(browser_test):
    paths = [row["path"] for row in runner._source_rows(browser_test)]
    assert browser_test in paths
    assert "tests/product_app/browser/test_e2e1_live_copilot_chat.py" in paths
    if browser_test == runner.COMPLETED_RACE_BROWSER_TEST:
        assert "tests/product_app/browser/waypoint_ui_fixture.py" in paths
        assert "scripts/verify_e2e1_network_observations.py" in paths
    assert "scripts/e2e1_provider_isolation.py" in paths
    assert "scripts/verify_e2e1_vertical_slice.py" in paths
    assert len(paths) == len(set(paths))


def test_active_route_delta_registration_and_exact_bundle_are_hashed():
    paths = {row["path"] for row in runner._source_rows(runner.COMPLETED_RACE_BROWSER_TEST)}
    registration = runner.ROOT / "configs/product_route_delta.json"
    if not registration.exists():
        pytest.skip("No additive route evidence configured")
    descriptor = runner.ROOT / json.loads(registration.read_text(encoding="utf-8"))["descriptor_path"]
    assert registration.relative_to(runner.ROOT).as_posix() in paths
    assert descriptor.relative_to(runner.ROOT).as_posix() in paths
    assert all(
        path.relative_to(runner.ROOT).as_posix() in paths for path in descriptor.parent.rglob("*") if path.is_file()
    )
