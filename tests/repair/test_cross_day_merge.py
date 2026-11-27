from types import SimpleNamespace

from itinerary_system.repair.baselines import _merge_day_candidates


def _candidate(day, sequence, route_ids):
    return SimpleNamespace(
        stop_sequence=sequence,
        repair_solution=SimpleNamespace(route_ids_by_day=route_ids, metadata={"candidate_id": f"day_route:{day}"}),
    )


def test_merged_move_keeps_each_changed_day_route_id():
    parent = SimpleNamespace(
        sequence=("a", "b"),
        route_ids_by_day={1: "parent1", 2: "parent2"},
        lodging_assignments={"1": "hotel", "2": "hotel"},
    )
    model = SimpleNamespace(metadata={"parent_plan": parent, "parent_day_by_stop": {"a": 1, "b": 2}})
    # Each day candidate carries the parent's id for the other day, as the day solver builds them.
    selected = {
        1: _candidate(1, (), {1: "new1", 2: "parent2"}),
        2: _candidate(2, ("b", "a"), {1: "parent1", 2: "new2"}),
    }
    merged = _merge_day_candidates(model, selected, method_id="heuristic")
    assert merged.route_ids_by_day == {1: "new1", 2: "new2"}
    assert merged.day_assignments == {"b": 2, "a": 2}
