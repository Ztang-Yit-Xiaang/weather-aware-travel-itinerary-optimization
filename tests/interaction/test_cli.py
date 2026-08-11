from __future__ import annotations

from types import SimpleNamespace

import pytest

from itinerary_system.interaction import cli
from itinerary_system.interaction.semantic_candidates import RuleBasedSemanticCandidateProvider
from itinerary_system.research_artifacts import PlanArtifactV2


def parent() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_cli",
        source_run_id="run",
        planning_request_id="request",
        catalog_snapshot_id="catalog",
        context_snapshot_id="context",
        selected_stops=({"stop_id": "stop", "name": "Stop", "day": 1},),
        sequence=("stop",),
    )


def args(**overrides):
    values = {
        "parent_plan": "parent.json",
        "catalog_snapshot_id": None,
        "context_snapshot_id": None,
        "clarification_mode": "rule_based_consequence",
        "semantic_candidates": None,
        "probe_results": "probes.json",
        "repair_session_id": "session",
        "user_edit": "less driving",
        "selected_interpretation_id": None,
        "question_count": 0,
        "continuation_of_session_id": None,
        "permission_decisions": None,
        "route_cache": None,
        "route_source_bundle_id": "",
        "route_source_sha256": "",
        "publication_mode": False,
        "config": "config.yaml",
        "max_questions": 2,
        "probe_time_limit_seconds": 2.0,
        "run_id": "run",
        "output_root": "runs",
        "permissive": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_rule_based_cli_mode_does_not_require_semantic_fixture(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(cli, "load_plan", lambda _path: parent())
    monkeypatch.setattr(cli, "FrozenCounterfactualProbeExecutor", lambda _path: object())

    def fake_run(**kwargs):
        captured.update(kwargs)
        return "run"

    monkeypatch.setattr(cli, "run_permission_aware_research_pipeline", fake_run)
    assert cli.run_from_args(args()) == "run"
    assert isinstance(captured["controller"].candidate_provider, RuleBasedSemanticCandidateProvider)


def test_frozen_cli_mode_requires_semantic_fixture(monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_plan", lambda _path: parent())
    with pytest.raises(ValueError, match="requires --semantic-candidates"):
        cli.run_from_args(args(clarification_mode="frozen_fixture"))
