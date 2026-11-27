"""Joint stop, day, order and overnight selection over a finite road matrix."""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .workspace import WorkspaceError


def solve_trip(
    *,
    stops,
    hotels,
    matrix,
    required,
    start,
    end,
    days=7,
    max_drive=360,
    max_day=720,
    time_limit=12,
    max_stops_per_day=None,
):
    """Minimize road minutes plus a declared 15-minute penalty per new stop.

    Every day has a path from one selected overnight base to the next, at
    least one unique visit, and separate drive/day limits. Positive elapsed
    time on each selected arc eliminates disconnected stop cycles.
    """
    names = sorted(stops)
    if isinstance(days, bool) or not 1 <= days <= 21 or not required <= set(names) or len(names) < days or not hotels:
        raise WorkspaceError("trip_planning_constraints_infeasible", 409)
    objective, upper, integral = [], [], []
    entries, lower_rows, upper_rows = [], [], []

    def variable(*, binary=True, cost=0.0):
        index = len(objective)
        objective.append(cost)
        upper.append(1.0 if binary else max_day)
        integral.append(int(binary))
        return index

    def constraint(coefficients, low=-np.inf, high=np.inf):
        entries.append(coefficients)
        lower_rows.append(low)
        upper_rows.append(high)

    def minutes(u, v):
        if u == v:
            return 0.0
        cell = matrix.cell(u, v)
        cell.require_publication_eligible()
        return cell.require_duration_s() / 60

    visits, arrivals, starts, ends, arcs, end_times = {}, {}, {}, {}, {}, {}
    for day in range(1, days + 1):
        start_options = [start] if day == 1 else sorted(hotels)
        end_options = [end] if day == days else sorted(hotels)
        for point in start_options:
            starts[day, point] = variable()
        for point in end_options:
            ends[day, point] = variable()
        constraint({starts[day, h]: 1 for h in start_options}, 1, 1)
        constraint({ends[day, h]: 1 for h in end_options}, 1, 1)
        end_times[day] = variable(binary=False)
        for point in names:
            visits[day, point] = variable(
                cost=15.0 + float(stops[point].get("preference_penalty_minutes", 0)) if point not in required else 0.0
            )
            arrivals[day, point] = variable(binary=False)
            constraint({arrivals[day, point]: 1, visits[day, point]: -max_day}, high=0)
        candidates = (
            [("start", h, p) for h in start_options for p in names]
            + [("visit", u, v) for u in names for v in names if u != v]
            + [("end", p, h) for p in names for h in end_options]
        )
        drive_terms = {}
        for kind, origin, destination in candidates:
            travel = minutes(origin, destination)
            if not math.isfinite(travel) or travel > min(max_day, max_drive):
                continue
            edge = variable(cost=travel)
            arcs[day, kind, origin, destination] = edge
            drive_terms[edge] = travel
            big_m = 3 * max_day
            if kind == "start":
                constraint({arrivals[day, destination]: 1, edge: -big_m}, low=travel - big_m)
            else:
                target_time = end_times[day] if kind == "end" else arrivals[day, destination]
                visit = float(stops[origin].get("visit_duration_minutes", 45))
                constraint({target_time: 1, arrivals[day, origin]: -1, edge: -big_m}, low=visit + travel - big_m)
        constraint(drive_terms, high=max_drive)
        constraint({visits[day, p]: 1 for p in names}, low=1, high=max_stops_per_day or np.inf)
        for point in names:
            incoming = {index: 1 for (d, k, u, v), index in arcs.items() if d == day and v == point and k != "end"}
            outgoing = {index: 1 for (d, k, u, v), index in arcs.items() if d == day and u == point and k != "start"}
            constraint({**incoming, visits[day, point]: -1}, 0, 0)
            constraint({**outgoing, visits[day, point]: -1}, 0, 0)
        for h in start_options:
            constraint(
                {
                    **{index: 1 for (d, k, u, v), index in arcs.items() if d == day and k == "start" and u == h},
                    starts[day, h]: -1,
                },
                0,
                0,
            )
        for h in end_options:
            constraint(
                {
                    **{index: 1 for (d, k, u, v), index in arcs.items() if d == day and k == "end" and v == h},
                    ends[day, h]: -1,
                },
                0,
                0,
            )
    for point in names:
        constraint({visits[d, point]: 1 for d in range(1, days + 1)}, 1 if point in required else 0, 1)
    for day in range(1, days):
        for hotel in hotels:
            constraint({ends[day, hotel]: 1, starts[day + 1, hotel]: -1}, 0, 0)
    row_indices, columns, values = [], [], []
    for row, terms in enumerate(entries):
        for column, value in terms.items():
            row_indices.append(row)
            columns.append(column)
            values.append(value)
    a = coo_matrix((values, (row_indices, columns)), shape=(len(entries), len(objective))).tocsc()
    began = time.monotonic()
    result = milp(
        c=np.asarray(objective),
        integrality=np.asarray(integral),
        bounds=Bounds(np.zeros(len(upper)), np.asarray(upper)),
        constraints=LinearConstraint(a, np.asarray(lower_rows), np.asarray(upper_rows)),
        options={"time_limit": float(time_limit), "mip_rel_gap": 0.0},
    )
    elapsed = time.monotonic() - began
    if result.x is None:
        code = "trip_planning_constraints_infeasible" if result.status == 2 else "trip_planning_solver_timeout"
        raise WorkspaceError(code, 409)
    x = result.x
    if (
        np.max(np.abs(x[np.asarray(integral) == 1] - np.rint(x[np.asarray(integral) == 1]))) > 1e-6
        or np.any(a @ x < np.asarray(lower_rows) - 1e-5)
        or np.any(a @ x > np.asarray(upper_rows) + 1e-5)
    ):
        raise WorkspaceError("trip_planning_solver_result_invalid", 409)
    routes = []
    for day in range(1, days + 1):
        origin = next(h for (d, h), index in starts.items() if d == day and x[index] > 0.5)
        destination = next(h for (d, h), index in ends.items() if d == day and x[index] > 0.5)
        sequence, point = [], origin
        kind = "start"
        for _ in range(len(names) + 1):
            selected = [
                (k, v)
                for (d, k, u, v), index in arcs.items()
                if d == day and u == point and k == kind and x[index] > 0.5
            ]
            if len(selected) != 1:
                raise WorkspaceError("trip_planning_solver_path_invalid", 409)
            edge_kind, point = selected[0]
            if edge_kind == "end":
                break
            sequence.append(point)
            selected_end = [
                v for (d, k, u, v), index in arcs.items() if d == day and k == "end" and u == point and x[index] > 0.5
            ]
            kind = "end" if selected_end else "visit"
        if point != destination or not sequence or len(sequence) != len(set(sequence)):
            raise WorkspaceError("trip_planning_solver_path_invalid", 409)
        routes.append({"day": day, "start_anchor_id": origin, "end_anchor_id": destination, "stop_ids": sequence})
    return {
        "days": routes,
        "solver_backend": "scipy_highs_milp",
        "status": "OPTIMAL" if result.status == 0 else "FEASIBLE",
        "objective": "road travel minutes plus 15 minutes per added stop and declared interest mismatch penalties",
        "objective_value": float(result.fun),
        "best_bound": float(result.mip_dual_bound),
        "optimality_gap": float(result.mip_gap),
        "runtime_seconds": elapsed,
        "scope": "Finite validated California candidate catalog, selected hotel bases and stated constraints",
        "variables": len(objective),
        "constraints": len(entries),
    }
