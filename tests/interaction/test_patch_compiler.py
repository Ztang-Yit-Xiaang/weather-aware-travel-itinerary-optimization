from __future__ import annotations

from itinerary_system.interaction.models import SemanticInterpretationCandidate
from itinerary_system.interaction.patch_compiler import AllowListedPatchCompiler
from itinerary_system.research_artifacts import PlanArtifactV2


def parent() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_patch",
        source_run_id="run",
        planning_request_id="request",
        catalog_snapshot_id="catalog",
        context_snapshot_id="context",
        selected_stops=({"stop_id": "stop", "name": "Stop", "day": 1},),
        sequence=("stop",),
    )


def candidate(
    interpretation_type: str,
    *,
    targets: tuple[str, ...] = ("stop",),
    parameters: dict | None = None,
    evidence_refs: tuple[str, ...] = ("catalog:fixture",),
) -> SemanticInterpretationCandidate:
    return SemanticInterpretationCandidate(
        interpretation_id=f"interpretation_{interpretation_type}",
        repair_session_id="session",
        user_text_hash="hash",
        target_ids=targets,
        interpretation_type=interpretation_type,
        normalized_parameters=parameters or {},
        support_score=None,
        evidence_refs=evidence_refs,
    )


def test_compiler_rejects_unknown_patch_target_parameter_and_non_json_value() -> None:
    compiler = AllowListedPatchCompiler()
    unknown_patch = compiler.compile(parent(), candidate("arbitrary_gurobi_expression"))
    unknown_target = compiler.compile(parent(), candidate("preserve_named_stop", targets=("nonexistent",)))
    unknown_parameter = compiler.compile(
        parent(), candidate("reduce_driving_burden", parameters={"objective_coefficient": 999})
    )
    non_json = compiler.compile(
        parent(), candidate("reduce_driving_burden", parameters={"target_minutes_reduction": object()})
    )
    assert "patch_type_not_allow_listed" in unknown_patch.reason_codes
    assert "target_not_in_parent_or_constraints" in unknown_target.reason_codes
    assert "parameter_not_allow_listed" in unknown_parameter.reason_codes
    assert "parameter_value_not_json_compatible" in non_json.reason_codes
    assert "parameter_value_invalid" in non_json.reason_codes


def test_compiler_requires_structural_fields_and_replacement_evidence() -> None:
    compiler = AllowListedPatchCompiler()
    missing_replacement = compiler.compile(parent(), candidate("same_day_replacement"))
    missing_target = compiler.compile(parent(), candidate("day_move", targets=(), parameters={"to_day": 2}))
    missing_evidence = compiler.compile(
        parent(),
        candidate(
            "same_day_replacement",
            parameters={"replacement_stop_id": "catalog_stop"},
            evidence_refs=(),
        ),
    )
    assert "required_parameter_missing" in missing_replacement.reason_codes
    assert "target_required" in missing_target.reason_codes
    assert "replacement_entity_evidence_missing" in missing_evidence.reason_codes
