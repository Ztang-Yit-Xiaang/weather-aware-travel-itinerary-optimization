import unittest

from itinerary_system.repair_planner import (
    ParsedRepairIntent,
    RepairRequest,
    build_repair_plan,
    detect_evidence_conflicts,
    evaluate_repair_plan,
    generate_repair_alternatives,
    parse_repair_intent,
)


def baseline_route():
    return (
        {
            "name": "San Francisco Ferry Building",
            "day": 1,
            "category": "food market",
            "utility": 7.0,
            "weather_risk": 0.10,
            "travel_minutes": 20,
            "overnight_city": "San Francisco",
        },
        {
            "name": "Yosemite Valley Hike",
            "day": 2,
            "category": "national park",
            "utility": 9.5,
            "weather_risk": 0.85,
            "travel_minutes": 210,
            "overnight_city": "Mariposa",
        },
        {
            "name": "Big Sur Coast",
            "day": 3,
            "category": "scenic drive",
            "utility": 8.5,
            "weather_risk": 0.15,
            "travel_minutes": 160,
            "overnight_city": "Monterey",
        },
    )


def candidate_pois():
    return (
        {
            "name": "Monterey Bay Aquarium",
            "day": 2,
            "category": "indoor museum",
            "utility": 8.0,
            "weather_risk": 0.05,
            "travel_minutes": 95,
            "overnight_city": "Monterey",
        },
        {
            "name": "Muir Woods Visitor Center",
            "day": 2,
            "category": "national park indoor visitor center",
            "utility": 7.2,
            "weather_risk": 0.25,
            "travel_minutes": 80,
            "overnight_city": "San Francisco",
        },
    )


def confirmed_request() -> RepairRequest:
    return RepairRequest(
        request_id="weather-repair",
        baseline_route=baseline_route(),
        user_intent="replace risky outdoor stops, keep Big Sur Coast",
        candidate_pois=candidate_pois(),
        tolerance_profile={"max_stop_weather_risk": 0.40, "max_daily_travel_minutes": 480.0},
        parsed_intent=ParsedRepairIntent(
            raw_text="replace risky outdoor stops, keep Big Sur Coast",
            confirmed=True,
            confidence=0.94,
            must_keep=("Big Sur Coast",),
        ),
    )


class RepairPlannerTests(unittest.TestCase):
    def test_unconfirmed_parser_never_controls_solver(self):
        request = RepairRequest(
            request_id="unconfirmed",
            baseline_route=baseline_route(),
            user_intent="replace Yosemite Valley Hike",
            candidate_pois=candidate_pois(),
            parsed_intent=parse_repair_intent("replace Yosemite Valley Hike", confirmed=False),
        )

        plan = build_repair_plan(request)

        self.assertEqual(plan.repaired_route, baseline_route())
        self.assertEqual(plan.operations, ())
        self.assertEqual(plan.metadata["status"], "blocked_unconfirmed_intent")
        self.assertEqual(plan.counterfactual_explanations[0].constraint, "confirmed_intent")

    def test_evidence_conflict_detection_labels_all_types(self):
        conflicts = detect_evidence_conflicts(
            (
                {
                    "affected_constraint": "yosemite_weather",
                    "source_id": "forecast_a",
                    "value": "open",
                    "confidence": 0.9,
                    "age_hours": 2,
                },
                {
                    "affected_constraint": "yosemite_weather",
                    "source_id": "forecast_b",
                    "value": "closed",
                    "confidence": 0.4,
                    "age_hours": 30,
                },
                {
                    "affected_constraint": "hotel_availability",
                    "source_id": "hotel_api",
                    "confidence": 0.8,
                },
            )
        )

        conflict_types = {conflict.conflict_type for conflict in conflicts}
        self.assertIn("contradictory", conflict_types)
        self.assertIn("low_confidence", conflict_types)
        self.assertIn("stale", conflict_types)
        self.assertIn("missing", conflict_types)

    def test_repair_replaces_high_weather_stop_and_reports_counterfactual(self):
        plan = build_repair_plan(confirmed_request())

        replaced = [operation for operation in plan.operations if operation.operation == "REPLACE"]
        self.assertEqual(len(replaced), 1)
        self.assertEqual(replaced[0].target_stop, "Yosemite Valley Hike")
        self.assertEqual(replaced[0].replacement_stop, "Muir Woods Visitor Center")
        self.assertTrue(plan.evaluation_report.feasible)
        self.assertGreater(plan.evaluation_report.metrics["weather_risk_delta"], 0.0)
        self.assertGreater(plan.evaluation_report.metrics["preservation_rate"], 0.0)
        self.assertTrue(
            any(
                explanation.target == "Yosemite Valley Hike"
                and explanation.constraint == "weather_risk"
                and "max_stop_weather_risk" in explanation.minimal_required_change
                for explanation in plan.counterfactual_explanations
            )
        )

    def test_generate_repair_alternatives_returns_frontier_policies(self):
        plans = generate_repair_alternatives(confirmed_request())

        self.assertEqual(len(plans), 3)
        self.assertEqual(
            {plan.metadata["frontier_policy"] for plan in plans},
            {"risk_averse", "balanced", "preservation_first"},
        )

    def test_move_operation_from_confirmed_constraints(self):
        request = RepairRequest(
            request_id="move-window",
            baseline_route=baseline_route(),
            user_intent="move Big Sur Coast to day 2",
            confirmed_constraints={"move_stops": {"Big Sur Coast": 2}},
            tolerance_profile={"max_stop_weather_risk": 1.0, "max_daily_travel_minutes": 480.0},
            parsed_intent=ParsedRepairIntent(
                raw_text="move Big Sur Coast to day 2",
                confirmed=True,
                confidence=0.95,
            ),
        )

        plan = build_repair_plan(request)

        moved = [operation for operation in plan.operations if operation.operation == "MOVE"]
        self.assertEqual(len(moved), 1)
        self.assertEqual(moved[0].target_stop, "Big Sur Coast")
        self.assertEqual(moved[0].day, 2)
        self.assertEqual(next(stop for stop in plan.repaired_route if stop["name"] == "Big Sur Coast")["day"], 2)

    def test_evaluation_hard_gate_separates_feasibility(self):
        repaired = (
            {
                "name": "Closed Trail",
                "day": 1,
                "utility": 5.0,
                "weather_risk": 0.95,
                "travel_minutes": 30,
                "status": "closed",
            },
        )

        report = evaluate_repair_plan(baseline_route(), repaired, max_stop_weather_risk=0.50)

        self.assertFalse(report.feasible)
        self.assertEqual(report.eligibility, "hard_infeasible")
        self.assertTrue(any("closed" in failure for failure in report.hard_failures))
        self.assertTrue(any("weather-risk gate" in failure for failure in report.hard_failures))


if __name__ == "__main__":
    unittest.main()
