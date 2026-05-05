import os

import numpy as np
import pandas as pd


def _infer_city(df):
    for col in ["city", "location_city"]:
        if col in df.columns:
            values = df[col].dropna()
            if not values.empty:
                mode = values.mode()
                if not mode.empty:
                    return str(mode.iloc[0])
    return "the destination"


def _safe_get_series(df, col, default=0.0):
    if col in df.columns:
        return df[col].astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _build_greedy_ranking(df):
    ranked_df = df.copy()
    if "stars" not in ranked_df.columns:
        ranked_df["stars"] = 0.0
    if "review_count" not in ranked_df.columns:
        ranked_df["review_count"] = 0.0
    if "utility" not in ranked_df.columns:
        ranked_df["utility"] = 0.0

    ranked_df = ranked_df.sort_values(
        ["stars", "review_count", "utility", "name"],
        ascending=[False, False, False, True],
    )
    return ranked_df.index.tolist()


def _theme_counts(df, indices):
    if not indices:
        return {}
    if "content_theme" in df.columns:
        return df.iloc[indices]["content_theme"].value_counts().to_dict()
    if "type" in df.columns:
        return df.iloc[indices]["type"].astype(str).value_counts().to_dict()
    return {}


def _format_hours(minutes):
    hours = minutes / 60.0
    if abs(hours - round(hours)) < 1e-9:
        return f"{int(round(hours))}-hour"
    return f"{hours:.1f}-hour"


def _compute_optimized_metrics(profile_name, context):
    top100 = context["top100_with_waiting_time"]
    hotels_df = context["hotels_df"]
    route_details = context["route_details"]
    routes = context["routes"]
    travel_time_matrix = np.asarray(context["travel_time_matrix"], dtype=float)
    hotel_to_attr_time = np.asarray(context["hotel_to_attr_time"], dtype=float)
    hotel_to_hotel_time = np.asarray(context["hotel_to_hotel_time"], dtype=float)

    details = route_details[profile_name]
    selected_indices = routes.get(profile_name, [])
    hotels_by_day = details.get("hotels_by_day", {})
    day_used_by_day = details.get("day_used_by_day", {})
    selected_by_day = details.get("selected_by_day", {})
    travel_edges_by_day = details.get("travel_edges_by_day", {})
    start_edges_by_day = details.get("start_edges_by_day", {})
    end_edges_by_day = details.get("end_edges_by_day", {})
    transitions_by_day = details.get("transitions_by_day", {})

    utility = _safe_get_series(top100, "utility")
    waiting = _safe_get_series(top100, "waiting_final")
    visit = _safe_get_series(top100, "visit_duration_sim")
    attraction_cost = _safe_get_series(top100, "cost")
    hotel_price = hotels_df["nightly_price"].astype(float)

    total_utility = float(utility.iloc[selected_indices].sum()) if selected_indices else 0.0
    total_wait = float(waiting.iloc[selected_indices].sum()) if selected_indices else 0.0
    total_visit = float(visit.iloc[selected_indices].sum()) if selected_indices else 0.0
    total_attraction_cost = float(attraction_cost.iloc[selected_indices].sum()) if selected_indices else 0.0
    total_hotel_cost = float(sum(hotel_price.iloc[h] for h in hotels_by_day.values() if h is not None))

    total_travel_between = float(
        sum(travel_time_matrix[i, j] for day_edges in travel_edges_by_day.values() for i, j in day_edges)
    )
    total_hotel_legs = float(
        sum(hotel_to_attr_time[h, i] for day_edges in start_edges_by_day.values() for h, i in day_edges)
        + sum(hotel_to_attr_time[h, i] for day_edges in end_edges_by_day.values() for i, h in day_edges)
    )
    total_switch_time = float(
        sum(hotel_to_hotel_time[h, g] for day_edges in transitions_by_day.values() for h, g in day_edges if h != g)
    )

    daily_times = {}
    for day_idx, day_nodes in selected_by_day.items():
        if not day_used_by_day.get(day_idx, 1):
            daily_times[day_idx] = 0.0
            continue
        visit_wait = float(sum(visit.iloc[i] + waiting.iloc[i] for i in day_nodes))
        travel_between = float(sum(travel_time_matrix[i, j] for i, j in travel_edges_by_day.get(day_idx, [])))
        hotel_legs = float(
            sum(hotel_to_attr_time[h, i] for h, i in start_edges_by_day.get(day_idx, []))
            + sum(hotel_to_attr_time[h, i] for i, h in end_edges_by_day.get(day_idx, []))
        )
        daily_times[day_idx] = visit_wait + travel_between + hotel_legs

    return {
        "profile": profile_name,
        "selected_indices": selected_indices,
        "selected_by_day": selected_by_day,
        "travel_edges_by_day": travel_edges_by_day,
        "start_edges_by_day": start_edges_by_day,
        "end_edges_by_day": end_edges_by_day,
        "hotels_by_day": hotels_by_day,
        "day_used_by_day": day_used_by_day,
        "transitions_by_day": transitions_by_day,
        "total_utility": total_utility,
        "total_wait": total_wait,
        "total_visit": total_visit,
        "total_travel_between": total_travel_between,
        "total_hotel_legs": total_hotel_legs,
        "total_switch_time": total_switch_time,
        "total_time": total_visit + total_wait + total_travel_between + total_hotel_legs,
        "total_hotel_cost": total_hotel_cost,
        "total_attraction_cost": total_attraction_cost,
        "total_cost": total_hotel_cost + total_attraction_cost,
        "content_mix": _theme_counts(top100, selected_indices),
        "status": details.get("status", "UNKNOWN"),
        "daily_times": daily_times,
        "is_feasible": True,
    }


def _compute_greedy_baseline(profile_name, context):
    top100 = context["top100_with_waiting_time"].copy()
    hotels_df = context["hotels_df"]
    route_details = context["route_details"]
    tourist_profiles = context["TOURIST_PROFILES"]
    solver_settings = context.get("SOLVER_SETTINGS", {})
    k_days = int(context["K"])

    travel_time_matrix = np.asarray(context["travel_time_matrix"], dtype=float)
    hotel_to_attr_time = np.asarray(context["hotel_to_attr_time"], dtype=float)
    hotel_to_hotel_time = np.asarray(context["hotel_to_hotel_time"], dtype=float)

    profile_cfg = tourist_profiles[profile_name]
    details = route_details[profile_name]

    daily_time_budget = float(solver_settings.get("daily_time_budget", 720))
    total_budget = float(solver_settings.get("total_budget", 1000))
    utility = _safe_get_series(top100, "utility")
    waiting = _safe_get_series(top100, "waiting_final")
    visit = _safe_get_series(top100, "visit_duration_sim")
    attraction_cost = _safe_get_series(top100, "cost")
    hotel_price = hotels_df["nightly_price"].astype(float)

    ranking = _build_greedy_ranking(top100)
    hotels_by_day = dict(details.get("hotels_by_day", {}))
    day_used_by_day = dict(details.get("day_used_by_day", {}))
    transitions_by_day = dict(details.get("transitions_by_day", {}))

    total_hotel_cost = float(sum(hotel_price.iloc[h] for h in hotels_by_day.values() if h is not None))
    remaining_budget = float(total_budget - total_hotel_cost)

    used_attractions = set()
    selected_by_day = {}
    travel_edges_by_day = {}
    start_edges_by_day = {}
    end_edges_by_day = {}
    daily_times = {}
    wine_count = 0
    whale_count = 0

    for day_idx in range(k_days):
        hotel_idx = hotels_by_day.get(day_idx)
        should_use_day = int(day_used_by_day.get(day_idx, 1))

        if hotel_idx is None or not should_use_day:
            selected_by_day[day_idx] = []
            travel_edges_by_day[day_idx] = []
            start_edges_by_day[day_idx] = []
            end_edges_by_day[day_idx] = []
            daily_times[day_idx] = 0.0
            continue

        selected_today = []
        day_wait = 0.0
        day_core_time = 0.0
        current_attr = None

        while len(selected_today) < int(profile_cfg["max_attractions"]):
            chosen_candidate = None

            for cand_idx in ranking:
                if cand_idx in used_attractions:
                    continue

                cand_name = str(top100.iloc[cand_idx].get("name", "")).lower()
                if "wine" in cand_name and wine_count >= 2:
                    continue
                if "whale" in cand_name and whale_count >= 2:
                    continue

                travel_from_current = (
                    float(hotel_to_attr_time[hotel_idx, cand_idx])
                    if current_attr is None
                    else float(travel_time_matrix[current_attr, cand_idx])
                )
                return_to_hotel = float(hotel_to_attr_time[hotel_idx, cand_idx])
                proposed_wait = day_wait + float(waiting.iloc[cand_idx])
                proposed_core_time = day_core_time + travel_from_current + float(visit.iloc[cand_idx]) + float(waiting.iloc[cand_idx])
                proposed_day_total = proposed_core_time + return_to_hotel

                if proposed_day_total > daily_time_budget:
                    continue
                if remaining_budget < float(attraction_cost.iloc[cand_idx]):
                    continue

                proposed_count = len(selected_today) + 1
                if proposed_wait > float(profile_cfg["max_wait"]) * proposed_count:
                    continue

                chosen_candidate = cand_idx
                break

            if chosen_candidate is None:
                break

            used_attractions.add(chosen_candidate)
            selected_today.append(chosen_candidate)
            remaining_budget -= float(attraction_cost.iloc[chosen_candidate])
            if current_attr is None:
                day_core_time += float(hotel_to_attr_time[hotel_idx, chosen_candidate])
            else:
                day_core_time += float(travel_time_matrix[current_attr, chosen_candidate])
            day_core_time += float(visit.iloc[chosen_candidate]) + float(waiting.iloc[chosen_candidate])
            day_wait += float(waiting.iloc[chosen_candidate])
            current_attr = chosen_candidate

            cand_name = str(top100.iloc[chosen_candidate].get("name", "")).lower()
            if "wine" in cand_name:
                wine_count += 1
            if "whale" in cand_name:
                whale_count += 1

        selected_by_day[day_idx] = selected_today
        if selected_today:
            start_edges_by_day[day_idx] = [(hotel_idx, selected_today[0])]
            travel_edges_by_day[day_idx] = list(zip(selected_today[:-1], selected_today[1:]))
            end_edges_by_day[day_idx] = [(selected_today[-1], hotel_idx)]
            daily_times[day_idx] = day_core_time + float(hotel_to_attr_time[hotel_idx, selected_today[-1]])
        else:
            start_edges_by_day[day_idx] = []
            travel_edges_by_day[day_idx] = []
            end_edges_by_day[day_idx] = []
            daily_times[day_idx] = 0.0

    selected_indices = sorted(used_attractions)
    total_utility = float(utility.iloc[selected_indices].sum()) if selected_indices else 0.0
    total_wait = float(waiting.iloc[selected_indices].sum()) if selected_indices else 0.0
    total_visit = float(visit.iloc[selected_indices].sum()) if selected_indices else 0.0
    total_attraction_cost = float(attraction_cost.iloc[selected_indices].sum()) if selected_indices else 0.0
    total_travel_between = float(
        sum(travel_time_matrix[i, j] for day_edges in travel_edges_by_day.values() for i, j in day_edges)
    )
    total_hotel_legs = float(
        sum(hotel_to_attr_time[h, i] for day_edges in start_edges_by_day.values() for h, i in day_edges)
        + sum(hotel_to_attr_time[h, i] for day_edges in end_edges_by_day.values() for i, h in day_edges)
    )
    total_switch_time = float(
        sum(hotel_to_hotel_time[h, g] for day_edges in transitions_by_day.values() for h, g in day_edges if h != g)
    )

    total_cost = total_hotel_cost + total_attraction_cost
    total_time = total_visit + total_wait + total_travel_between + total_hotel_legs
    is_feasible = (
        total_cost <= total_budget + 1e-9
        and all(day_minutes <= daily_time_budget + 1e-9 for day_minutes in daily_times.values())
        and len(selected_indices) >= int(profile_cfg["min_attractions"])
    )

    return {
        "profile": profile_name,
        "selected_indices": selected_indices,
        "selected_by_day": selected_by_day,
        "travel_edges_by_day": travel_edges_by_day,
        "start_edges_by_day": start_edges_by_day,
        "end_edges_by_day": end_edges_by_day,
        "hotels_by_day": hotels_by_day,
        "day_used_by_day": day_used_by_day,
        "transitions_by_day": transitions_by_day,
        "total_utility": total_utility,
        "total_wait": total_wait,
        "total_visit": total_visit,
        "total_travel_between": total_travel_between,
        "total_hotel_legs": total_hotel_legs,
        "total_switch_time": total_switch_time,
        "total_time": total_time,
        "total_hotel_cost": total_hotel_cost,
        "total_attraction_cost": total_attraction_cost,
        "total_cost": total_cost,
        "remaining_budget": remaining_budget,
        "content_mix": _theme_counts(top100, selected_indices),
        "ranking_method": "greedy_rating_based",
        "daily_times": daily_times,
        "is_feasible": bool(is_feasible),
    }


def build_route_stop_rows_from_details(context, detail_payload, profile_name, variant="baseline"):
    """Convert optimized/baseline detail payloads into map-renderable stop rows."""
    if not detail_payload:
        return pd.DataFrame()
    if profile_name in detail_payload:
        detail_payload = detail_payload[profile_name]
    details = detail_payload.get(variant, detail_payload)
    if not isinstance(details, dict):
        return pd.DataFrame()

    top100 = context["top100_with_waiting_time"]
    hotels_df = context["hotels_df"]
    k_days = int(context["K"])
    selected_by_day = details.get("selected_by_day", {})
    hotels_by_day = details.get("hotels_by_day", {})
    rows = []

    for day_idx in range(k_days):
        hotel_idx = hotels_by_day.get(day_idx)
        if hotel_idx is None or day_idx not in selected_by_day:
            continue
        hotel_row = hotels_df.iloc[hotel_idx]
        selected_today = list(selected_by_day.get(day_idx, []))
        for stop_order, attr_idx in enumerate(selected_today, start=1):
            attr_row = top100.iloc[attr_idx]
            rows.append(
                {
                    "comparison_type": "method",
                    "comparison_label": f"{profile_name} {variant}",
                    "method": f"{variant}_legacy_baseline",
                    "day": int(day_idx + 1),
                    "stop_order": int(stop_order),
                    "city": attr_row.get("city", attr_row.get("location_city", "unknown")),
                    "overnight_city": attr_row.get("city", attr_row.get("location_city", "unknown")),
                    "hotel_name": hotel_row.get("name", f"hotel_{hotel_idx}"),
                    "hotel_latitude": hotel_row.get("latitude"),
                    "hotel_longitude": hotel_row.get("longitude"),
                    "attraction_name": attr_row.get("name", f"attraction_{attr_idx}"),
                    "latitude": attr_row.get("latitude"),
                    "longitude": attr_row.get("longitude"),
                    "category": attr_row.get("category", attr_row.get("content_theme", attr_row.get("type", "attraction"))),
                    "social_must_go": False,
                    "social_score": 0.0,
                    "final_poi_value": float(attr_row.get("utility", 0.0) or 0.0),
                    "route_start_latitude": hotel_row.get("latitude"),
                    "route_start_longitude": hotel_row.get("longitude"),
                    "route_end_latitude": hotel_row.get("latitude"),
                    "route_end_longitude": hotel_row.get("longitude"),
                    "route_type": f"legacy_{variant}_route",
                }
            )
    return pd.DataFrame(rows)


def compare_optimized_vs_greedy_rating_baseline(context, save_path=None):
    required = [
        "top100_with_waiting_time",
        "travel_time_matrix",
        "hotel_to_attr_time",
        "hotel_to_hotel_time",
        "hotels_df",
        "routes",
        "route_details",
        "TOURIST_PROFILES",
        "K",
    ]
    missing = [key for key in required if key not in context]
    if missing:
        raise KeyError(f"Missing required context keys for baseline comparison: {missing}")

    top100 = context["top100_with_waiting_time"]
    city_name = _infer_city(top100)
    solver_settings = context.get("SOLVER_SETTINGS", {})
    daily_time_budget = float(solver_settings.get("daily_time_budget", 720))
    total_budget = float(solver_settings.get("total_budget", 1000))
    k_days = int(context["K"])
    n_candidates = len(top100)

    comparison_rows = []
    detail_rows = {}

    for profile_name in context["TOURIST_PROFILES"]:
        if profile_name not in context["route_details"]:
            continue

        optimized = _compute_optimized_metrics(profile_name, context)
        baseline = _compute_greedy_baseline(profile_name, context)

        baseline_utility = baseline["total_utility"]
        optimized_utility = optimized["total_utility"]
        utility_lift_pct = np.nan
        if abs(baseline_utility) > 1e-9:
            utility_lift_pct = 100.0 * (optimized_utility - baseline_utility) / baseline_utility

        sentence = (
            f"For a {k_days}-day trip in {city_name} with {n_candidates} candidate attractions, "
            f"the optimized {profile_name} itinerary achieves {utility_lift_pct:.1f}% higher total utility "
            f"than a greedy rating-based baseline while respecting a {_format_hours(daily_time_budget)} daily budget "
            f"and a ${int(round(total_budget))} total budget."
        )

        comparison_rows.append(
            {
                "profile": profile_name,
                "candidate_attractions": n_candidates,
                "days": k_days,
                "daily_time_budget_min": daily_time_budget,
                "total_budget": total_budget,
                "optimized_total_utility": optimized_utility,
                "baseline_total_utility": baseline_utility,
                "utility_lift_pct": utility_lift_pct,
                "optimized_attractions": len(optimized["selected_indices"]),
                "baseline_attractions": len(baseline["selected_indices"]),
                "optimized_total_cost": optimized["total_cost"],
                "baseline_total_cost": baseline["total_cost"],
                "optimized_total_time": optimized["total_time"],
                "baseline_total_time": baseline["total_time"],
                "optimized_total_wait": optimized["total_wait"],
                "baseline_total_wait": baseline["total_wait"],
                "optimized_hotel_cost": optimized["total_hotel_cost"],
                "baseline_hotel_cost": baseline["total_hotel_cost"],
                "optimized_switch_time": optimized["total_switch_time"],
                "baseline_switch_time": baseline["total_switch_time"],
                "optimized_feasible": optimized["is_feasible"],
                "baseline_feasible": baseline["is_feasible"],
                "optimized_status": optimized["status"],
                "baseline_method": "Greedy by rating/review under the same hotel schedule",
                "demo_sentence": sentence,
            }
        )

        detail_rows[profile_name] = {
            "optimized": optimized,
            "baseline": baseline,
            "demo_sentence": sentence,
        }

    comparison_df = pd.DataFrame(comparison_rows)

    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        comparison_df.to_csv(save_path, index=False)

    return comparison_df, detail_rows
