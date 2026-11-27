"""Fresh deterministic browser suites plus fail-closed runtime coverage.

The registered journeys are a minimum regression inventory, not proof that all
enabled interactions are integrated. Missing capability rows remain failures.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from e2e1_interaction_coverage import coverage_report  # noqa: E402
from e2e1_provider_isolation import deterministic_environment  # noqa: E402
from e2e1_source_universe import deterministic_source_inputs  # noqa: E402
from run_e2e1_vertical_slice import (  # noqa: E402
    ROOT,
    E2E1RunnerError,
    _canonical_sha256,
    _isolation_passed,
    _port_is_available,
    _sha256,
)
from verify_e2e1_owned_edits import CASES as OWNED_EDIT_CASES  # noqa: E402
from verify_e2e1_vertical_slice import E2E1VerificationError, _verify_product_http_capture  # noqa: E402
from verify_e2e1_vertical_slice import verify as verify_child  # noqa: E402

REGISTRY = "configs/e2e1_interaction_coverage.json"
MANIFEST = "aggregate_manifest.json"
REPORT = "aggregate_exact_io_report.md"
SUITES = (
    (
        "vertical-slice",
        "test_e2e1_vertical_slice_live.py",
        (
            "test_registered_replacement_renders_the_exact_evaluated_child",
            "test_copilot_one_click_preview_from_empty_draft_renders_exact_child",
            "test_route_aware_candidate_discovery_stages_and_renders_exact_child",
            "test_unsupported_removal_never_stages_or_changes_the_map",
            "test_ineligible_duration_keeps_parent_map_and_drafts",
            "test_older_preview_response_cannot_overwrite_newer_preview",
            "test_accepted_replacement_survives_reload_with_exact_map_identity",
        ),
    ),
    (
        "completed-response-race",
        "test_completed_preview_race_live.py",
        ("test_completed_older_child_response_cannot_replace_distinct_newer_child",),
    ),
    (
        "map-recovery",
        "test_map_initialization_recovery_live.py",
        (
            "test_active_map_reload_recovers_without_replacing_session_or_plan[original]",
            "test_active_map_reload_recovers_without_replacing_session_or_plan[accepted_child]",
            "test_loaded_source_cannot_acknowledge_a_viewport_without_renderable_tiles",
        ),
    ),
    (
        "workspace-stability",
        "test_workspace_stability_live.py",
        tuple(
            f"test_workspace_navigation_focus_and_map_identity[{width}-{height}]"
            for width, height in ((1440, 900), (1280, 800), (1024, 768), (390, 844))
        ),
    ),
    (
        "direct-edits",
        "test_direct_edit_journeys_live.py",
        tuple(
            f"test_direct_edit_browser_renders_exact_evaluated_child[{kind}-{target}-{day}]"
            for kind, target, day in (
                ("combined", "bixby_creek_bridge_viewpoint", 5),
                ("duration", "griffith_observatory", 7),
                ("time_window", "stearns_wharf", 6),
                ("role", "stearns_wharf", 6),
                ("order", "kings_canyon_national_park", 4),
            )
        ),
    ),
    (
        "addition-continuation",
        "test_addition_continuation_live.py",
        ("test_browser_accept_replacement_then_add_candidate_and_preserve_lineage",),
    ),
    (
        "mobile-chat",
        "test_mobile_chat_journey_live.py",
        ("test_mobile_chat_preserves_semantics_and_accepts_exact_rendered_child",),
    ),
    (
        "conversation",
        "test_copilot_conversation_journeys_live.py",
        (
            "test_clarification_survives_reload_and_short_answer_renders_exact_child",
            "test_combined_chat_stages_all_edits_and_renders_exact_child",
            "test_combined_chat_continuation_and_one_click[original-one_click]",
            "test_combined_chat_continuation_and_one_click[accepted_child-staged]",
            "test_combined_chat_continuation_and_one_click[accepted_child-one_click]",
            "test_preference_survives_seven_turns_corrects_and_resets_without_plan_change",
            "test_capacity_blocks_real_chat_before_dispatch_and_settings_reset_recovers[turns]",
            "test_capacity_blocks_real_chat_before_dispatch_and_settings_reset_recovers[bytes]",
            "test_fake_sdk_stop_waiting_reload_and_retry_reconciles_one_attempt",
            "test_late_selection_preserves_deleted_conversation_generation",
            "test_accepted_child_conversation_controls[delete_all]",
            "test_accepted_child_conversation_controls[capacity_turns]",
            "test_accepted_child_conversation_controls[capacity_bytes]",
            "test_accepted_child_conversation_controls[recovery]",
        ),
    ),
    (
        "combined-conflicts",
        "test_copilot_combined_conflicts_live.py",
        tuple(
            f"test_conflicting_combined_chat_never_offers_or_stages_partial_edits[{context}]"
            for context in ("original", "accepted_child")
        ),
    ),
    (
        "queued-preview",
        "test_copilot_queued_preview_live.py",
        tuple(
            f"test_deleted_chat_cannot_resume_queued_preview[{context}-{kind}]"
            for context in ("original", "accepted_child")
            for kind in ("single", "combined")
        ),
    ),
    ("owned-edits", "test_owned_combined_journey_live.py", tuple(OWNED_EDIT_CASES.values())),
    (
        "build-coherence",
        "test_build_coherence_live.py",
        tuple(
            f"test_cached_page_requires_reload_and_preserves_active_plan_across_real_restart[{context}]"
            for context in ("original", "accepted_child")
        ),
    ),
    (
        "bootstrap-recovery",
        "test_bootstrap_recovery_live.py",
        (
            "test_actual_bootstrap_fault_retry_renders_ready_parent[core_not_ready]",
            "test_actual_bootstrap_fault_retry_renders_ready_parent[invalid_health]",
        ),
    ),
    (
        "optional-evidence-fault",
        "test_optional_evidence_fault_live.py",
        ("test_optional_evidence_projection_fault_preserves_accepted_child_and_back_navigation",),
    ),
    (
        "preview-geography-fault",
        "test_preview_geography_fault_live.py",
        tuple(
            f"test_real_eligible_child_with_bad_geography_is_blocked_without_parent_or_draft_loss[{kind}]"
            for kind in ("missing", "foreign")
        ),
    ),
    (
        "waypoint-pointer",
        "test_waypoint_pointer_live.py",
        ("test_map_pointer_add_relocate_and_two_undos_preserve_accepted_plan",),
    ),
    (
        "capability-controls",
        "test_capability_controls_live.py",
        tuple(
            f"test_real_control_variants_preserve_draft_only_and_disabled_semantics[{context}]"
            for context in ("original", "accepted_child")
        ),
    ),
    (
        "evaluated-variants",
        "test_evaluated_variants_live.py",
        tuple(
            f"test_real_evaluated_variants_preview_reload_and_undo[{context}]"
            for context in ("original", "accepted_child")
        ),
    ),
    (
        "webgl-recovery",
        "test_webgl_recovery_live.py",
        ("test_real_webgl_loss_withdraws_ack_and_restores_exact_evaluated_child",),
    ),
    (
        "chat-closure",
        "test_chat_runtime_closure_live.py",
        tuple(
            f"test_keep_and_chat_families_preserve_runtime_truth[{context}]"
            for context in ("original", "accepted_child")
        ),
    ),
    (
        "same-plan-remount",
        "test_same_plan_remount_live.py",
        ("test_ui_remount_withdraws_acceptance_until_same_plan_is_ready",),
    ),
    (
        "waypoint-contexts",
        "test_waypoint_contexts_live.py",
        tuple(
            f"test_route_checked_waypoint_contexts_preserve_accepted_plan[{context}]"
            for context in ("original", "accepted_child")
        ),
    ),
    (
        "candidate-day-matrix",
        "test_candidate_day_matrix_live.py",
        ("test_browser_candidate_day_readiness_in_original_and_accepted_child",),
    ),
)
EXPECTED_TEST_COUNT = sum(len(names) for _, _, names in SUITES)


def _now():
    return datetime.now(UTC).isoformat()


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_rows():
    paths = {
        "scripts/run_e2e1_aggregate.py",
        "scripts/e2e1_interaction_coverage.py",
        "scripts/e2e1_runtime_gate_coverage.py",
        REGISTRY,
    }
    for _, filename, _ in SUITES:
        paths.update(deterministic_source_inputs(f"tests/product_app/browser/{filename}"))
    return [
        {"path": relative, "size_bytes": (ROOT / relative).stat().st_size, "sha256": _sha256(ROOT / relative)}
        for relative in sorted(paths)
    ]


def _require(condition, code):
    if not condition:
        raise E2E1RunnerError(code)


def test_inventory(directory, filename, names):
    """JUnit exact node inventory rejects skips, duplicate nodes and count-only passes."""
    root = ET.parse(directory / "pytest_junit.xml").getroot()
    cases = list(root.iter("testcase"))
    expected_class = "tests.product_app.browser." + filename.removesuffix(".py")
    _require(
        len(cases) == len(names)
        and sorted(case.get("name") for case in cases) == sorted(names)
        and all(case.get("classname") == expected_class for case in cases),
        "e2e1_aggregate_test_inventory_mismatch",
    )
    _require(
        not any(node.tag in {"failure", "error", "skipped"} for node in root.iter()), "e2e1_aggregate_test_not_passed"
    )
    return [f"tests/product_app/browser/{filename}::{name}" for name in names]


def child_lifecycles(directory, manifest):
    """Independently check every owned server shutdown, including failed suites."""
    if manifest.get("schema_version") != "e2e1-self-managed-browser-manifest-v1":
        _require(_isolation_passed(manifest.get("provider_isolation", {})), "e2e1_aggregate_cleanup_unverified")
        return [manifest["provider_isolation"]["record"]]
    from verify_e2e1_self_managed import COMBINED_CONFLICT_CASES, CONVERSATION_CASES, QUEUED_PREVIEW_CASES, _lifecycle

    process = json.loads((directory / "test_process.json").read_text(encoding="utf-8"))
    mode, run_id = manifest["mode"], manifest["run_id"]
    _require(
        mode
        in {
            "conversation",
            "combined-conflicts",
            "owned-edits",
            "build-coherence",
            "bootstrap-recovery",
            "optional-evidence-fault",
            "preview-geography-fault",
            "queued-preview",
        },
        "e2e1_aggregate_self_managed_mode_invalid",
    )
    if mode == "owned-edits":
        return [
            _lifecycle(directory / "conversation" / case, run_id + "_" + case, process) for case in OWNED_EDIT_CASES
        ]
    if mode == "preview-geography-fault":
        return [
            _lifecycle(
                directory / "preview_geography_fault" / kind / generation,
                run_id + "_" + kind + "_" + generation,
                process,
                child_process=True,
            )
            for kind in ("missing", "foreign")
            for generation in ("generation_a", "generation_b")
        ]
    if mode == "optional-evidence-fault":
        return [
            _lifecycle(
                directory / "optional_evidence_fault" / generation,
                run_id + "_" + generation,
                process,
                child_process=True,
            )
            for generation in ("generation_a", "generation_b")
        ]
    if mode == "bootstrap-recovery":
        from verify_e2e1_bootstrap_recovery import CASES, CONTEXTS, fault_lifecycle

        records = []
        for case, context in [(case, context) for case in CASES for context in CONTEXTS]:
            base = directory / "bootstrap_recovery" / case / context
            identity = run_id + "_" + case + "_" + context
            records.append(fault_lifecycle(base / "fault", identity + "_fault", process, case))
            records.append(_lifecycle(base / "recovered", identity + "_recovered", process, child_process=True))
        return records
    if mode == "build-coherence":
        return [
            _lifecycle(
                directory / "build_coherence" / context / generation,
                run_id + "_" + context + "_" + generation,
                process,
                child_process=True,
            )
            for context in ("original", "accepted_child")
            for generation in ("generation_a", "generation_b")
        ]
    return [
        _lifecycle(
            directory / "conversation" / case,
            run_id + "_" + case,
            process,
            fake=case.removesuffix("_accepted_child") == "fake_sdk_cancel",
        )
        for case in (
            QUEUED_PREVIEW_CASES
            if mode == "queued-preview"
            else COMBINED_CONFLICT_CASES
            if mode == "combined-conflicts"
            else CONVERSATION_CASES
        )
    ]


def capability_snapshots(directory, suites):
    observed = []
    for suite in suites:
        if suite["status"] != "passed":
            continue
        base = directory / "suites" / suite["mode"]
        for path in sorted(base.rglob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(row, dict) or row.get("schema_version") != "e2e1-product-http-io-v1":
                continue
            _verify_product_http_capture(row)
            if row.get("response_status_code") != 200:
                continue
            response = json.loads(row["response_body_utf8"]) if row["response_body_utf8"].startswith("{") else {}
            if not isinstance(response, dict) or not {"session", "workspace"} <= response.keys():
                continue
            session, workspace = response["session"], response["workspace"]
            _require(
                isinstance(session, dict)
                and isinstance(workspace, dict)
                and all(
                    isinstance(session.get(key), str) and session[key] for key in ("session_id", "accepted_plan_id")
                )
                and type(session.get("revision")) is int
                and session["revision"] >= 0,
                "e2e1_capability_session_invalid",
            )
            pointer = workspace.get("accepted_plan_pointer") or {}
            geography = workspace.get("geography")
            _require(
                isinstance(pointer, dict) and isinstance(geography, dict) and isinstance(geography.get("plans"), list),
                "e2e1_capability_geography_invalid",
            )
            plans = geography["plans"]
            _require(
                all(
                    isinstance(plan, dict)
                    and all(isinstance(plan.get(key), str) and plan[key] for key in ("plan_id", "content_hash"))
                    for plan in plans
                ),
                "e2e1_capability_geography_invalid",
            )
            plan = next((plan for plan in plans if plan["plan_id"] == session["accepted_plan_id"]), None)
            _require(
                plan is not None
                and pointer.get("accepted_plan_id") == plan["plan_id"]
                and pointer.get("accepted_plan_content_hash") == plan["content_hash"],
                "e2e1_capability_context_identity_mismatch",
            )
            groups = {
                name: workspace.get(name)
                for name in ("draft_capabilities", "typed_edit_capabilities", "map_edit_capabilities")
            }
            _require(
                all(isinstance(group, dict) and isinstance(group.get("operations"), dict) for group in groups.values()),
                "e2e1_capability_snapshot_incomplete",
            )
            observed.append(
                {
                    "artifact": path.relative_to(directory).as_posix(),
                    "artifact_sha256": _sha256(path),
                    "mode": suite["mode"],
                    "session_id": session["session_id"],
                    "revision": session["revision"],
                    "plan_id": plan["plan_id"],
                    "plan_content_hash": plan["content_hash"],
                    "context": "accepted_child" if pointer.get("decision_id") else "original",
                    "capabilities": groups,
                    "interaction": workspace.get("interaction"),
                }
            )

    # The coverage universe is one independently verified authoritative parent
    # per required context. Intermediate A -> B -> C HTTP responses remain
    # validated above, but they cannot silently create extra coverage baselines.
    suite = next((row for row in suites if row["mode"] == "capability-controls"), None)
    _require(
        suite is not None and suite["status"] == "passed" and suite.get("verification", {}).get("verified") is True,
        "e2e1_canonical_capability_suite_missing",
    )
    child = directory / "suites" / "capability-controls"
    manifest = json.loads((child / "file_io_manifest.json").read_text(encoding="utf-8"))
    snapshots = []
    for context in ("original", "accepted_child"):
        relative = Path("capability_controls") / context / "capability_snapshot.json"
        path = child / relative
        canonical = json.loads(path.read_text(encoding="utf-8"))
        entries = [row for row in manifest.get("files", []) if row.get("path") == relative.as_posix()]
        _require(
            len(entries) == 1
            and entries[0].get("role") == "exact_output"
            and entries[0].get("sha256") == _sha256(path)
            and entries[0].get("size_bytes") == path.stat().st_size,
            "e2e1_canonical_capability_artifact_unbound",
        )
        groups = canonical.get("capabilities")
        _require(
            canonical.get("schema_version") == "runtime-capability-snapshot-v1"
            and canonical.get("context") == context
            and all(
                isinstance(canonical.get(key), str) and canonical[key]
                for key in ("session_id", "plan_id", "plan_content_hash")
            )
            and isinstance(groups, dict)
            and set(groups) == {"draft_capabilities", "typed_edit_capabilities", "map_edit_capabilities"}
            and all(isinstance(group, dict) and isinstance(group.get("operations"), dict) for group in groups.values()),
            "e2e1_canonical_capability_snapshot_invalid",
        )
        matches = [
            row
            for row in observed
            if row["mode"] == "capability-controls"
            and all(row[key] == canonical[key] for key in ("context", "session_id", "plan_id", "plan_content_hash"))
            and row["capabilities"] == groups
        ]
        _require(matches, "e2e1_canonical_capability_snapshot_not_runtime_bound")
        runtime = sorted(matches, key=lambda row: row["artifact"])[0]
        snapshots.append(
            {
                "artifact": (Path("suites") / "capability-controls" / relative).as_posix(),
                "artifact_sha256": entries[0]["sha256"],
                "mode": "capability-controls",
                "runtime_http_artifact": runtime["artifact"],
                "runtime_http_artifact_sha256": runtime["artifact_sha256"],
                "session_id": canonical["session_id"],
                "revision": runtime["revision"],
                "plan_id": canonical["plan_id"],
                "plan_content_hash": canonical["plan_content_hash"],
                "context": context,
                "capabilities": groups,
                "interaction": runtime["interaction"],
            }
        )
    return snapshots


def _files(root):
    return [
        {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path not in {root / MANIFEST, root / REPORT}
    ]


def render_report(manifest):
    lines = [
        "# Aggregate E2E exact input and output",
        "",
        f"- Browser journeys: {manifest['passed_test_count']}/{manifest['expected_test_count']} passed with independent child verification",
        f"- Deterministic interaction gate passed: {str(manifest['assertions_passed']).lower()}",
        "- Full phase completion is not claimed by this zero-provider aggregate. Live website Copilot requires a separate authorized and independently verified evidence bundle.",
        f"- Source unchanged: {str(manifest['source_unchanged']).lower()}",
        "- No billed provider dispatch is authorized by this deterministic command.",
        "- Missing runtime rows remain failures even if every registered regression journey passes.",
        "",
        "## Exact file inputs and outputs",
        "",
        "| File | Bytes | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    lines += [
        f"| [{row['path']}]({row['path']}) | {row['size_bytes']} | {row['sha256']} |" for row in manifest["files"]
    ]
    lines += ["", "## Exact source inputs", "", "| File | Bytes | SHA-256 |", "| --- | ---: | --- |"]
    lines += [f"| {row['path']} | {row['size_bytes']} | {row['sha256']} |" for row in manifest["source_inputs"]]
    lines += [
        "",
        "## Exact LLM inputs and outputs",
        "",
        "Each child manifest binds its llm_io.json records and application-server isolation receipt. "
        "The full records, including null inputs/outputs for zero-call scenarios, remain in the linked child exact_io_report.md files.",
    ]
    for suite in manifest["suites"]:
        lines += [
            "",
            f"- {suite['mode']}: {suite['status']}; [{suite['mode']} exact I/O](suites/{suite['mode']}/exact_io_report.md)",
        ]
    return "\n".join(lines) + "\n"


def run_aggregate(*, output_dir=None, port=None):
    import os

    port = 8127 if port is None else port
    _require(type(port) is int and port == 8127, "e2e1_aggregate_requires_disposable_port_8127")
    root = (output_dir or ROOT / "results/e2e" / datetime.now(UTC).strftime("aggregate_%Y%m%dT%H%M%S%fZ")).resolve()
    _require(not root.exists() or not any(root.iterdir()), "e2e1_output_directory_not_empty")
    _require(_port_is_available(port), f"e2e1_port_{port}_occupied")
    root.mkdir(parents=True, exist_ok=True)
    start = _now()
    sources = source_rows()
    results = []
    stop_reason = None
    for mode, filename, names in SUITES:
        result = {"mode": mode, "status": "not_run", "test_nodeids": [], "started_at": _now()}
        if source_rows() != sources:
            stop_reason = "e2e1_aggregate_source_changed"
        if stop_reason:
            result.update(code=stop_reason, completed_at=_now())
            results.append(result)
            continue
        child = root / "suites" / mode
        command = [
            sys.executable,
            "scripts/run_e2e1_vertical_slice.py",
            "--browser-test",
            mode,
            "--output-dir",
            str(child),
            "--port",
            str(port),
        ]
        result["command"] = command
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=deterministic_environment(os.environ),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            (root / f"{mode}.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (root / f"{mode}.stderr.log").write_text(completed.stderr, encoding="utf-8")
            result["exit_code"] = completed.returncode
            child_manifest = json.loads((child / "file_io_manifest.json").read_text(encoding="utf-8"))
            lifecycles = child_lifecycles(child, child_manifest)
            result["cleanup_verified"] = bool(lifecycles) and _port_is_available(port)
            result["server_lifecycle_count"] = len(lifecycles)
            _require(result["cleanup_verified"], "e2e1_aggregate_cleanup_unverified")
            _require(completed.returncode == 0, "e2e1_child_runner_failed")
            result["verification"] = verify_child(child)
            result["test_nodeids"] = test_inventory(child, filename, names)
            result["status"] = "passed"
        except (
            E2E1RunnerError,
            E2E1VerificationError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            ET.ParseError,
        ) as error:
            result.update(status="failed", code=str(error), error_type=type(error).__name__)
            if result.get("cleanup_verified") is not True:
                stop_reason = "e2e1_aggregate_cleanup_unverified"
        result["completed_at"] = _now()
        results.append(result)
    snapshot_error = None
    try:
        snapshots = capability_snapshots(root, results)
    except (E2E1RunnerError, E2E1VerificationError, OSError, ValueError, KeyError, TypeError) as error:
        snapshots = []
        snapshot_error = (
            str(error)
            if isinstance(error, (E2E1RunnerError, E2E1VerificationError))
            else "e2e1_capability_evidence_malformed"
        )
    registry = json.loads((ROOT / REGISTRY).read_text(encoding="utf-8"))
    coverage = coverage_report(snapshots, results, registry, root)
    if snapshot_error:
        coverage["passed"] = False
        coverage["findings"].append({"code": snapshot_error})
    _write(root / "capability_snapshots.json", snapshots)
    _write(root / "interaction_coverage.json", coverage)
    unchanged = sources == source_rows()
    manifest = {
        "schema_version": "e2e1-aggregate-v1",
        "started_at": start,
        "completed_at": _now(),
        "source_inputs": sources,
        "source_unchanged": unchanged,
        "suites": results,
        "expected_test_count": EXPECTED_TEST_COUNT,
        "passed_test_count": sum(len(row["test_nodeids"]) for row in results),
        "assertions_passed": unchanged and coverage["passed"] and all(row["status"] == "passed" for row in results),
        "external_gates": coverage["external_gates"],
        "full_phase_completion_claimed": False,
        "files": _files(root),
    }
    manifest["content_hash"] = _canonical_sha256(manifest)
    _write(root / MANIFEST, manifest)
    (root / REPORT).write_text(render_report(manifest), encoding="utf-8")
    if manifest["assertions_passed"]:
        candidate_manifest = (root / MANIFEST).read_bytes()
        candidate_report = (root / REPORT).read_bytes()
        try:
            verify_aggregate(root)
        except (
            E2E1RunnerError,
            E2E1VerificationError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            ET.ParseError,
        ) as error:
            # Preserve exactly what verification inspected; never rewrite child evidence.
            prior_manifest = root / "aggregate_pre_verification_manifest.json"
            prior_report = root / "aggregate_pre_verification_report.md"
            prior_manifest.write_bytes(candidate_manifest)
            prior_report.write_bytes(candidate_report)
            _write(
                root / "aggregate_verification_failure.json",
                {
                    "schema_version": "e2e1-aggregate-verification-failure-v1",
                    "observed_at": _now(),
                    "assertions_passed": False,
                    "error_type": type(error).__name__,
                    "code": str(error),
                    "candidate_manifest_file": prior_manifest.name,
                    "candidate_manifest_sha256": _sha256(prior_manifest),
                    "candidate_report_file": prior_report.name,
                    "candidate_report_sha256": _sha256(prior_report),
                },
            )
            manifest.update(
                assertions_passed=False,
                source_unchanged=sources == source_rows(),
                completed_at=_now(),
                files=_files(root),
            )
            manifest.pop("content_hash")
            manifest["content_hash"] = _canonical_sha256(manifest)
            _write(root / MANIFEST, manifest)
            (root / REPORT).write_text(render_report(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(root),
                "assertions_passed": manifest["assertions_passed"],
                "passed_test_count": manifest["passed_test_count"],
                "coverage_findings": len(coverage["findings"]),
            },
            indent=2,
        )
    )
    return 0 if manifest["assertions_passed"] else 1


def verify_aggregate(root):
    root = root.resolve()
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    seed = dict(manifest)
    digest = seed.pop("content_hash")
    _require(digest == _canonical_sha256(seed), "e2e1_aggregate_manifest_hash_mismatch")
    _require(
        manifest["schema_version"] == "e2e1-aggregate-v1" and manifest["files"] == _files(root),
        "e2e1_aggregate_artifact_binding_mismatch",
    )
    _require(
        manifest["source_unchanged"] is True and manifest["source_inputs"] == source_rows(),
        "e2e1_aggregate_source_changed",
    )
    _require((root / REPORT).read_text(encoding="utf-8") == render_report(manifest), "e2e1_aggregate_report_mismatch")
    _require(
        [row["mode"] for row in manifest["suites"]] == [mode for mode, _, _ in SUITES],
        "e2e1_aggregate_suite_universe_invalid",
    )
    previous = datetime.fromisoformat(manifest["started_at"])
    sources = {row["path"]: row for row in manifest["source_inputs"]}
    for result, (mode, filename, names) in zip(manifest["suites"], SUITES, strict=True):
        _require(
            result["status"] == "passed" and result["exit_code"] == 0 and result["cleanup_verified"] is True,
            "e2e1_aggregate_suite_not_passed",
        )
        started, ended = datetime.fromisoformat(result["started_at"]), datetime.fromisoformat(result["completed_at"])
        _require(previous <= started <= ended, "e2e1_aggregate_run_order_invalid")
        previous = ended
        child = root / "suites" / mode
        _require(result["verification"] == verify_child(child), "e2e1_aggregate_child_verification_mismatch")
        _require(result["test_nodeids"] == test_inventory(child, filename, names), "e2e1_aggregate_nodeids_mismatch")
        child_manifest = json.loads((child / "file_io_manifest.json").read_text(encoding="utf-8"))
        _require(
            all(sources.get(row["path"]) == row for row in child_manifest["source_inputs"]),
            "e2e1_aggregate_mixed_sources",
        )
        lifecycles = child_lifecycles(child, child_manifest)
        _require(result["server_lifecycle_count"] == len(lifecycles), "e2e1_aggregate_lifecycle_count_invalid")
        prior_finalized = started
        for receipt in lifecycles:
            installed, finalized = map(datetime.fromisoformat, (receipt["installed_at"], receipt["finalized_at"]))
            _require(
                prior_finalized
                <= installed
                <= finalized
                <= datetime.fromisoformat(child_manifest["generated_at"])
                <= ended,
                "e2e1_aggregate_reused_child_evidence",
            )
            prior_finalized = finalized
    _require(
        previous <= datetime.fromisoformat(manifest["completed_at"])
        and manifest["expected_test_count"] == manifest["passed_test_count"] == EXPECTED_TEST_COUNT,
        "e2e1_aggregate_test_total_invalid",
    )
    snapshots = capability_snapshots(root, manifest["suites"])
    coverage = coverage_report(
        snapshots, manifest["suites"], json.loads((ROOT / REGISTRY).read_text(encoding="utf-8")), root
    )
    _require(
        snapshots == json.loads((root / "capability_snapshots.json").read_text(encoding="utf-8"))
        and coverage == json.loads((root / "interaction_coverage.json").read_text(encoding="utf-8")),
        "e2e1_aggregate_coverage_projection_mismatch",
    )
    _require(
        coverage["passed"] and manifest["assertions_passed"] is True, "e2e1_aggregate_interaction_coverage_incomplete"
    )
    _require(
        manifest["external_gates"] == coverage["external_gates"] and manifest["full_phase_completion_claimed"] is False,
        "e2e1_aggregate_external_scope_invalid",
    )
    return {
        "verified": True,
        "test_count": EXPECTED_TEST_COUNT,
        "content_hash": digest,
        "scope": "deterministic_runtime_integration",
        "provider_calls": 0,
        "full_phase_completion_claimed": False,
        "external_gates": coverage["external_gates"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a fresh aggregate bundle without rerunning browsers.")
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(verify_aggregate(parser.parse_args().output_dir), indent=2))
