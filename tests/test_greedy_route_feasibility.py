from __future__ import annotations

import pandas as pd

from itinerary_system.experiment_runner import _solve_greedy_day_route, _without_placeholder_candidates


def candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "Affordable viewpoint",
                "category": "viewpoint",
                "latitude": 34.0,
                "longitude": -118.0,
                "route_candidate_score": 2.0,
                "final_poi_value": 2.0,
                "cost": 10.0,
            },
            {
                "name": "Second viewpoint",
                "category": "viewpoint",
                "latitude": 34.001,
                "longitude": -118.001,
                "route_candidate_score": 1.0,
                "final_poi_value": 1.0,
                "cost": 10.0,
            },
        ]
    )


def test_greedy_day_route_respects_remaining_budget():
    selected, result = _solve_greedy_day_route(
        candidate_frame(),
        route_start=(34.0, -118.0),
        route_end=(34.0, -118.0),
        available_visit_minutes=500.0,
        remaining_budget=15.0,
        max_stops=4,
    )

    assert result["solver_status"] == "HEURISTIC"
    assert selected["cost"].sum() <= 15.0
    assert len(selected) == 1


def test_greedy_empty_candidate_day_is_explicit_not_solver_failure():
    selected, result = _solve_greedy_day_route(
        pd.DataFrame(),
        route_start=(34.0, -118.0),
        route_end=(34.0, -118.0),
        available_visit_minutes=500.0,
        remaining_budget=15.0,
        max_stops=4,
    )

    assert selected.empty
    assert result["solver_status"] == "EMPTY_DAY"


def test_placeholder_candidates_never_enter_route_solver():
    candidates = pd.concat(
        [
            candidate_frame(),
            pd.DataFrame(
                [
                    {
                        "name": "San Luis Obispo attraction catalog pending",
                        "category": "data_ingestion_needed",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    filtered = _without_placeholder_candidates(candidates)

    assert len(filtered) == 2
    assert not filtered["name"].str.contains("catalog pending").any()
