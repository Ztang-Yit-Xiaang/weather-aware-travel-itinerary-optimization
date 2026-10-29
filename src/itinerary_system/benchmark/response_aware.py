"""Deterministic synthetic experiment, deliberately separate from publication gates.

No synthetic edge is road validated. Canonical evaluator certificates are retained
unchanged; model feasibility excludes only the expected road-evidence rejection.
This module does not make an uncertified toy itinerary eligible for product use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import permutations, product
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from ..evaluation import PlanEvaluator, PlanEvaluatorConfig
from ..plans import OwnedConstraint, compute_plan_diff
from ..plans.repository import PlanRepository
from ..repair.day_route_solver import DayRouteSolverConfig, solve_day_route_subproblem
from ..repair.exact_baselines import plan_full_reoptimization
from ..repair.induced_disruption import DisruptionScope, measure_induced_disruption
from ..repair.lexicographic import planner_runs_from_lexicographic_result, solve_lexicographically
from ..repair.master_model import build_repair_master_model
from ..repair.neighborhood import REPAIR_RADIUS_ORDER, RepairRadius, build_repair_neighborhood
from ..repair.progressive import ProgressiveRepairController
from ..research_artifacts import PlanArtifactV2, PlannerRun, stable_content_hash
from ..routing import RouteMatrix, RouteMatrixCell

CREATED_AT = "2026-10-19T00:00:00+00:00"


@dataclass(frozen=True)
class ToyScenario:
    scenario_id: str
    improved_edge_minutes: float = 1.0
    weather_excluded: tuple[str, ...] = ()
    replacements: bool = False


SCENARIOS = (
    ToyScenario("travel_before", 40.0),
    ToyScenario("travel_after", 1.0),
    ToyScenario("travel_threshold", 5.0),
    ToyScenario("weather_choice", 1.0, ("a",), True),
    ToyScenario("weather_two_stops", 1.0, ("a", "o"), True),
    ToyScenario("booked_infeasible", 1.0, ("b",)),
)


def toy_inputs(
    scenario: ToyScenario,
) -> tuple[PlanArtifactV2, SimpleNamespace, RouteMatrix, DayRouteSolverConfig, DisruptionScope]:
    """Build an abstract directed graph with two hotel anchors and five parent stops.

    l is locked, b is booked at 10:30--10:45, p is strongly preferred but
    flexible in order; a and o are optional. x/y are alternative indoor stops.
    Inputs contain no live providers, geodesic estimates, or road assertions.
    """
    context = f"synthetic:{scenario.scenario_id}:{scenario.improved_edge_minutes}"
    records = tuple(
        {
            "stop_id": sid,
            "day": 1,
            "lodging_id": "hotel",
            "visit_duration_minutes": 15,
            "opening_start": "09:00",
            "opening_end": "17:00",
            "utility": 1.0,
            "weather_risk": 0.0,
            **(
                {
                    "opening_start": "10:30",
                    "opening_end": "10:30",
                    "time_window_constraint": {
                        "schema_version": "stop-time-window-constraint-v1",
                        "earliest_arrival": "10:30",
                        "latest_departure": "10:45",
                        "early_arrival_policy": "wait_until_earliest_arrival",
                        "latest_departure_semantics": "departure_after_visit",
                    },
                }
                if sid == "b"
                else {}
            ),
        }
        for sid in ("l", "a", "p", "b", "o", "x", "y")
    )
    constraints = tuple(
        OwnedConstraint(
            constraint_id=f"protect_{sid}",
            origin="user_booking" if strength == "booked" else "user",
            strength=strength,
            scope="stop",
            target_id=sid,
            relation="must_keep",
            value=True,
            confirmed=True,
            relaxation_policy="never"
            if strength == "locked"
            else "explicit_only"
            if strength == "booked"
            else "auto_with_penalty",
        ).to_record()
        for sid, strength in (("l", "locked"), ("b", "booked"), ("p", "strong"))
    )
    sequence = ("l", "a", "p", "b", "o")
    parent = PlanArtifactV2(
        plan_id="toy_parent",
        source_run_id="toy_accepted",
        planning_request_id="toy_original",
        catalog_snapshot_id="synthetic_catalog_v1",
        context_snapshot_id=context,
        selected_stops=records[:5],
        day_assignments=dict.fromkeys(sequence, 1),
        sequence=sequence,
        lodging_assignments={"1": "hotel"},
        route_ids_by_day={1: "accepted_route"},
        owned_constraints=constraints,
        created_at=CREATED_AT,
    )
    request = SimpleNamespace(
        request_id=f"repair:{context}",
        parent_plan_id=parent.plan_id,
        allowed_radii=REPAIR_RADIUS_ORDER,
        confirmed_constraints={
            "affected_days": (1,),
            "weather_feasible": dict.fromkeys(scenario.weather_excluded, False),
        },
        candidate_pois=records[5:] if scenario.replacements else (),
    )
    edges = {(u, v): 40.0 for u in ("s", *sequence, "x", "y", "t") for v in ("s", *sequence, "x", "y", "t") if u != v}
    for path in (
        ("s", "l", "a", "p", "b", "o", "t"),
        ("s", "l", "p", "b", "a", "o", "t"),
        ("s", "l", "x", "p", "b", "x", "o", "t"),
    ):
        edges.update(dict.fromkeys(zip(path[:-1], path[1:], strict=True), 5.0))
    edges.update(
        {
            ("b", "a"): scenario.improved_edge_minutes,
            ("b", "x"): 1.0,
            ("l", "y"): 4.0,
            ("y", "p"): 4.0,
            ("x", "t"): 5.0,
            ("y", "t"): 5.0,
        }
    )
    cells = {
        (u, v): RouteMatrixCell(
            u,
            v,
            distance_m=None,
            duration_s=minutes * 60,
            provider="synthetic_fixture",
            road_validated=False,
            context_snapshot_id=context,
            duration_source="declared_toy_graph",
        )
        for (u, v), minutes in edges.items()
    }
    matrix = RouteMatrix(
        matrix_id=f"toy_matrix:{stable_content_hash(edges_to_records(edges))}",
        context_snapshot_id=context,
        entity_ids=("s", *sequence, "x", "y", "t"),
        cells=cells,
    )
    config = DayRouteSolverConfig(
        max_day_minutes=180, default_visit_minutes=15, start_anchor_by_day={1: "s"}, end_anchor_by_day={1: "t"}
    )
    affected = scenario.weather_excluded or ("a", "b")
    direct = tuple(("deleted_stops", sid) for sid in scenario.weather_excluded)
    if scenario.replacements:
        direct += (("added_stops", "x"), ("added_stops", "y"))
    # A road-ID change is an expected consequence of replacing a stop. For a
    # pure time update, keeping the accepted sequence needs no itinerary edit.
    if scenario.weather_excluded:
        direct += (("road_changes", "1"),)
    return parent, request, matrix, config, DisruptionScope(tuple(affected), (1,), direct)


def edges_to_records(edges: dict[tuple[str, str], float]) -> list[dict]:
    return [{"origin": u, "destination": v, "minutes": minutes} for (u, v), minutes in sorted(edges.items())]


class SyntheticModelEvaluator:
    """Benchmark-only model gate; never changes the independent certificate."""

    def __init__(self, parent: PlanArtifactV2, matrix: RouteMatrix, config: DayRouteSolverConfig) -> None:
        if not matrix.cells or any(
            cell.provider != "synthetic_fixture" or cell.road_validated or cell.fallback_used
            for cell in matrix.cells.values()
        ):
            raise ValueError("synthetic evaluator requires explicitly unvalidated toy cells")
        self.parent = parent
        self.evaluator = PlanEvaluator(
            route_matrix=matrix,
            reference_plan=parent,
            config=PlanEvaluatorConfig(
                strict_routes=False,
                max_day_minutes=config.max_day_minutes,
                default_visit_minutes=config.default_visit_minutes,
                start_anchor_by_day=config.start_anchor_by_day,
                end_anchor_by_day=config.end_anchor_by_day,
                include_schedule_metrics=True,
            ),
        )

    def evaluate(self, child: PlanArtifactV2, planner_run: PlannerRun | None = None) -> dict:
        # The progressive extractor uses :repair_master rather than its stage
        # run ID. This explicit adapter receipt is not an optimality certificate;
        # the controller's original stage records are retained separately.
        run = planner_run or PlannerRun(
            run_id=child.source_run_id,
            planning_request_id=child.planning_request_id,
            catalog_snapshot_id=child.catalog_snapshot_id,
            context_snapshot_id=child.context_snapshot_id,
            planner_specification_id="synthetic-model-evaluation-adapter-v1",
            method_requested="progressive_repair",
            method_executed="progressive_repair",
            execution_status="COMPLETED",
            solver_certification="HEURISTIC_ONLY",
            result_plan_id=child.plan_id,
        )
        certificate = self.evaluator.evaluate_final_plan(child, planner_run=run)
        modeled_failures = [
            finding.to_record()
            for finding in certificate.failures
            if finding.code != "route_matrix_not_publication_ready"
        ]
        return {
            "eligible": not modeled_failures,
            "evaluation_status": "SYNTHETIC_MODEL_ONLY",
            "comparison_eligibility": "ineligible",
            "modeled_failures": modeled_failures,
            "certificate": certificate.to_record(),
            "evaluation_run": run.to_record(),
        }


def response_row(
    action_id: str,
    child: PlanArtifactV2,
    radius: RepairRadius,
    evaluator: SyntheticModelEvaluator,
    scope: DisruptionScope,
    *,
    planner_run: PlannerRun | None = None,
) -> dict:
    evaluation = evaluator.evaluate(child, planner_run)
    metrics = evaluation["certificate"]["metrics"]
    return {
        "action_id": action_id,
        "radius": radius.value,
        "plan": child.to_record(),
        "diff": compute_plan_diff(evaluator.parent, child).to_record(),
        "disruption": measure_induced_disruption(evaluator.parent, child, scope).to_record(),
        "evaluation": evaluation,
        "travel_minutes": metrics["day_1_travel_minutes"],
        "utility": metrics["utility_retained"],
        "context_burden": sum(float(stop.get("weather_risk", 0)) for stop in child.selected_stops),
    }


def select_response(rows: list[dict]) -> dict | None:
    """Choose an independently checked toy response, preserving radius then D.

    The hard model gate precedes selection. Secondary disruption is only a
    tie-break after total D; context, utility and travel follow preservation.
    Selection is exact only over supplied rows, not all possible repairs.
    """
    eligible = [row for row in rows if row["evaluation"]["eligible"]]
    return min(
        eligible,
        key=lambda row: (
            REPAIR_RADIUS_ORDER.index(RepairRadius(row["radius"])),
            row["disruption"]["weighted_edit_cost"],
            row["disruption"]["secondary_cost"],
            row["context_burden"],
            -row["utility"],
            row["travel_minutes"],
            row["action_id"],
        ),
        default=None,
    )


def compare_matched_pool() -> dict:
    """Compare preservation objectives on identical independently checked routes.

    Both indoor replacements use new route IDs. Only their downstream order
    differs. Reversing pool order and restricting coverage expose tie sensitivity
    and the limit of objective correction. No extra follower calls or routes are
    available to either selector. Canonical road certificates remain failed.
    """
    parent, request, matrix, config, scope = toy_inputs(SCENARIOS[3])
    parent = replace(
        parent,
        selected_stops=tuple({**stop, "stop_order": index} for index, stop in enumerate(parent.selected_stops, 1)),
    )
    radius = RepairRadius.SAME_STOP_TIME_SHIFT
    model = build_repair_master_model(parent, request, build_repair_neighborhood(parent, request, radius), matrix)
    evaluator = SyntheticModelEvaluator(parent, matrix, config)
    routes = solve_day_route_subproblem(
        model, matrix, day=1, config=config, candidate_sequences=(("l", "p", "b", "x", "o"), ("l", "y", "p", "b", "o"))
    )
    candidates = tuple(
        candidate.repair_solution
        for candidate in routes.feasible_candidates
        if evaluator.evaluate(model.extract_plan(candidate.repair_solution))["eligible"]
    )
    if len(candidates) != 2:
        raise AssertionError("matched-pool fixture requires two independently model-feasible alternatives")
    cases = {}
    for name, pool in (
        ("disruptive_first", candidates),
        ("preserving_first", tuple(reversed(candidates))),
        ("restricted_pool", candidates[:1]),
    ):
        outcomes = {}
        for use_diff in (False, True):
            result = solve_lexicographically(model, candidate_solutions=pool, use_plan_diff=use_diff)
            child = model.extract_plan(result.selected_solution)
            outcomes[result.preservation_cost_basis] = {
                "sequence": child.sequence,
                "diff": compute_plan_diff(parent, child).to_record(),
                "evaluation": evaluator.evaluate(child),
                "solver": result.to_record(),
                "stage_runs": [run.to_record() for run in planner_runs_from_lexicographic_result(result)],
            }
        cases[name] = {
            "candidate_ids": [candidate.metadata["candidate_id"] for candidate in pool],
            "outcomes": outcomes,
        }
    return {
        "parent": parent.to_record(),
        "scope": asdict(scope),
        "routes": routes.to_record(),
        "cases": cases,
        "publication_eligible": False,
        "evidence_mode": "synthetic_graph_only",
    }


def downstream_responses(
    parent: PlanArtifactV2,
    request: SimpleNamespace,
    matrix: RouteMatrix,
    config: DayRouteSolverConfig,
    scope: DisruptionScope,
    evaluator: SyntheticModelEvaluator,
) -> tuple[list[dict], list[dict]]:
    """Enumerate local replacement/drop actions and a minimum-travel follower.

    Each action fixes its selected stops; the follower permutes that set on day
    one using the existing route solver and the independent booking evaluator.
    Earliest feasible schedule is the existing static schedule model. A parent
    sequence tie-break is NOT used: follower ties use tuple ordering. The outer
    selector can also retain the Progressive Repair incumbent (caller-owned).
    """
    # Use the same first neighborhood as B. In current code this radius permits
    # same-day substitutions too; its name is not an enforced edit-type gate.
    radius = RepairRadius.SAME_STOP_TIME_SHIFT
    model = build_repair_master_model(parent, request, build_repair_neighborhood(parent, request, radius), matrix)
    excluded = tuple(request.confirmed_constraints["weather_feasible"])
    candidate_ids = tuple(stop["stop_id"] for stop in request.candidate_pois)
    actions = product((None, *candidate_ids), repeat=len(excluded)) if excluded else ((),)
    responses: list[dict] = []
    traces: list[dict] = []
    for replacements in actions:
        additions = [sid for sid in replacements if sid is not None]
        if len(set(additions)) != len(additions):
            continue
        mapping = dict(zip(excluded, replacements, strict=True))
        selected = tuple(mapping.get(sid, sid) for sid in parent.sequence if mapping.get(sid, sid) is not None)
        action_id = f"local:{stable_content_hash(mapping)}"
        result = solve_day_route_subproblem(
            model, matrix, day=1, config=config, candidate_sequences=tuple(permutations(selected))
        )
        rows = []
        for candidate in result.feasible_candidates:
            child = replace(model.extract_plan(candidate.repair_solution), created_at=CREATED_AT)
            run = PlannerRun(
                run_id=child.source_run_id,
                planning_request_id=child.planning_request_id,
                catalog_snapshot_id=child.catalog_snapshot_id,
                context_snapshot_id=child.context_snapshot_id,
                planner_specification_id="toy-fixed-selection-travel-follower-v1",
                method_requested="minimum_travel_follower",
                method_executed="minimum_travel_follower",
                execution_status="COMPLETED",
                solver_certification="HEURISTIC_ONLY",
                solver_backend="complete_fixed_selection_permutations",
                result_plan_id=child.plan_id,
            )
            row = response_row(action_id, child, radius, evaluator, scope, planner_run=run)
            rows.append(row)
        valid = [row for row in rows if row["evaluation"]["eligible"]]
        response = min(valid, key=lambda row: (row["travel_minutes"], tuple(row["plan"]["sequence"])), default=None)
        traces.append(
            {
                "action_id": action_id,
                "replacement_map": mapping,
                "search_complete": True,
                "sequence_count": len(result.candidates),
                "model_feasible_count": len(valid),
                "response": response,
                "evaluated_routes": [
                    {
                        "sequence": list(candidate.stop_sequence),
                        "travel_minutes": candidate.travel_minutes,
                        "violations": list(candidate.violations),
                    }
                    for candidate in result.candidates
                ],
                "independent_checks": [
                    {"sequence": row["plan"]["sequence"], "modeled_failures": row["evaluation"]["modeled_failures"]}
                    for row in rows
                ],
            }
        )
        if response is not None:
            responses.append(response)
    return responses, traces


def run_toy_scenario(scenario: ToyScenario, output_dir: Path) -> dict[str, Any]:
    """Run A/B/C with identical immutable inputs; retain failures and ablations."""
    parent, request, matrix, config, scope = toy_inputs(scenario)
    parent_hash = parent.content_hash
    evaluator = SyntheticModelEvaluator(parent, matrix, config)
    repo = PlanRepository(output_dir / "plans")
    repo.save(parent)
    started = perf_counter()
    full = plan_full_reoptimization(parent, request, matrix, day_route_config=config)
    a = (
        response_row(
            "full_reoptimization",
            full.child_plan,
            RepairRadius.FULL_REOPTIMIZATION,
            evaluator,
            scope,
            planner_run=full.planner_run,
        )
        if full.child_plan is not None
        else None
    )
    full_seconds = perf_counter() - started
    started = perf_counter()
    progressive = ProgressiveRepairController(
        plan_repository=repo, route_matrix=matrix, evaluator=evaluator, day_route_config=config
    ).repair_progressively(request)
    b = (
        response_row("progressive_incumbent", progressive.child_plan, progressive.accepted_radius, evaluator, scope)
        if progressive.child_plan is not None
        else None
    )
    progressive_seconds = perf_counter() - started
    started = perf_counter()
    responses, traces = downstream_responses(parent, request, matrix, config, scope, evaluator)
    existing_rows = []
    for attempt in progressive.attempts:
        if attempt.radius != progressive.accepted_radius:
            continue
        model = build_repair_master_model(
            parent, request, build_repair_neighborhood(parent, request, attempt.radius), matrix
        )
        for day_result in attempt.day_route_results:
            for candidate in day_result.feasible_candidates:
                child = replace(model.extract_plan(candidate.repair_solution), created_at=CREATED_AT)
                existing_rows.append(
                    response_row(f"existing:{candidate.stop_sequence}", child, attempt.radius, evaluator, scope)
                )
    # C retains the original controller's safe incumbent; the expanded-response
    # pool is reported so any non-worsening result is not mistaken for a theorem
    # about the production controller or a separate traveler response.
    c = select_response(([b] if b is not None else []) + responses)
    response_seconds = perf_counter() - started
    reranked = select_response(existing_rows)
    greedy = min(responses, key=lambda row: (row["travel_minutes"], row["action_id"]), default=None)
    if parent.content_hash != parent_hash:
        raise AssertionError("accepted parent mutated")
    matrix_record = {
        "matrix_id": matrix.matrix_id,
        "context_snapshot_id": matrix.context_snapshot_id,
        "entity_ids": matrix.entity_ids,
        "cells": [asdict(cell) for _, cell in sorted(matrix.cells.items())],
    }
    return {
        "scenario": asdict(scenario),
        "input_hash": stable_content_hash(
            {"parent": parent.to_record(), "request": vars(request), "matrix": matrix_record, "config": asdict(config)}
        ),
        "parent": parent.to_record(),
        "request": vars(request),
        "matrix": matrix_record,
        "route_config": asdict(config),
        "scope": asdict(scope),
        "strategies": {"full_reoptimization": a, "progressive_repair": b, "response_aware": c},
        "runtime_seconds": {
            "full_reoptimization": full_seconds,
            "progressive_repair": progressive_seconds,
            "response_aware_total": progressive_seconds + response_seconds,
        },
        "full_solver": full.to_record(),
        "progressive_solver": progressive.to_record(),
        "downstream_actions": traces,
        "greedy_local_response": greedy,
        "rerank_existing_only": reranked,
        "response_candidate_count": len(responses) + int(b is not None),
        "publication_eligible": False,
        "evidence_mode": "synthetic_graph_only",
        "timing_note": "C includes B plus downstream enumeration and rerank ablation; single run, no speed claim",
    }
