"""Escaped HTML summaries and interactive controls for the Folium renderer."""

from __future__ import annotations

import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd

from .blueprint_core import PROFILE_CONFIGS, _load_csv, _numeric_catalog_series, _unique_in_order
from .blueprint_render_primitives import _escape, _finite_float
from .blueprint_route_selector import add_route_debug_controls


def _build_profile_comparison(profile_day_plans):
    rows = []
    for profile_name, plan_df in profile_day_plans.items():
        config = PROFILE_CONFIGS[profile_name]
        if plan_df.empty:
            hotel_summary = "n/a"
            stop_count = 0
            must_go_count = 0
        else:
            city_column = "overnight_city" if "overnight_city" in plan_df.columns else "city"
            bases = plan_df[["day", city_column, "hotel_name"]].drop_duplicates()
            hotel_summary = ", ".join(
                f"D{int(row.day)} {getattr(row, city_column)}: {row.hotel_name}"
                for row in bases.itertuples(index=False)
            )
            stop_count = int(len(plan_df))
            must_go_count = int(plan_df["social_must_go"].astype(bool).sum())
        rows.append(
            f"""
            <tr>
                <td>{_escape(config["label"])}</td>
                <td>{stop_count}</td>
                <td>{must_go_count}</td>
                <td>{_escape(config["route_mode"])}</td>
                <td>{_escape(hotel_summary)}</td>
            </tr>
            """
        )
    return "".join(rows)



def _build_method_comparison(output_dir):
    output_dir = Path(output_dir)
    method_df = _load_csv(output_dir / "production_method_comparison.csv")
    if not method_df.empty and "method" in method_df.columns:
        return method_df
    return pd.DataFrame()



def _build_method_comparison_html(method_comparison_df):
    if method_comparison_df.empty:
        return '<div class="panel-section-title">Method Comparison</div><div class="summary-line">Method comparison not generated yet.</div>'

    if "method" in method_comparison_df.columns and "total_utility" in method_comparison_df.columns:
        rows = []
        for row in method_comparison_df.itertuples(index=False):
            utility = _escape(
                f"{float(getattr(row, 'total_utility', np.nan)):.2f}"
                if np.isfinite(_finite_float(getattr(row, "total_utility", np.nan)))
                else "n/a"
            )
            travel_minutes = _finite_float(getattr(row, "total_travel_time", np.nan))
            travel = _escape(f"{travel_minutes / 60.0:.1f}h" if np.isfinite(travel_minutes) else "n/a")
            budget = _escape(
                f"${float(getattr(row, 'budget_used', getattr(row, 'total_cost', np.nan))):,.0f}"
                if np.isfinite(_finite_float(getattr(row, "budget_used", getattr(row, "total_cost", np.nan))))
                else "n/a"
            )
            stops = _escape(
                f"{int(getattr(row, 'selected_attractions', 0))}"
                if np.isfinite(_finite_float(getattr(row, "selected_attractions", np.nan)))
                else "n/a"
            )
            status = _escape(str(getattr(row, "status", "n/a")))
            score_value = _finite_float(getattr(row, "comparison_score", np.nan))
            score = _escape(f"{score_value:.3f}" if np.isfinite(score_value) else "n/a")
            solvers = _escape(
                f"{getattr(row, 'allocation_solver', 'n/a')} / {getattr(row, 'local_route_solver', 'n/a')}"
            )
            display_name = _escape(str(getattr(row, "method_display_name", getattr(row, "method", "unknown"))))
            rows.append(
                f"""
                <tr>
                    <td>{display_name}</td>
                    <td>{score}</td>
                    <td>{utility}</td>
                    <td>{stops}</td>
                    <td>{travel}</td>
                    <td>{budget}</td>
                    <td>{solvers}</td>
                    <td>{status}</td>
                </tr>
                """
            )
        return f"""
        <div class="panel-section-title">Method Comparison</div>
        <table class="method-table">
            <thead><tr><th>Method</th><th>Score</th><th>Utility</th><th>Stops</th><th>Travel</th><th>Cost</th><th>Solvers</th><th>Status</th></tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
        <div class="summary-line muted-note"><b>Metric math:</b> U=sum selected POI values; T=sum route travel minutes; C=hotel+transport+POI cost; M=count of must-go stops; H=normalized Shannon category diversity. Score=0.40U' + 0.15M' + 0.15H + 0.15T' + 0.10C' + 0.05W', with lower travel/cost/wait inverted.</div>
        <div class="summary-line muted-note">Full comparison saved to production_method_comparison.csv with matching route stops in production_method_route_stops.csv.</div>
        """
    return '<div class="panel-section-title">Method Comparison</div><div class="summary-line">Method comparison not generated yet.</div>'



def _build_trip_length_comparison_html(output_dir):
    comparison_df = _load_csv(Path(output_dir) / "production_trip_length_comparison.csv")
    if comparison_df.empty:
        return '<div class="panel-section-title">Trip Length Comparison</div><div class="summary-line">Trip-length comparison not generated yet.</div>'
    rows = []
    for row in comparison_df.itertuples(index=False):
        drive_hours = _finite_float(getattr(row, "intercity_drive_hours", np.nan))
        budget = _finite_float(getattr(row, "estimated_budget", np.nan))
        status = _escape(str(getattr(row, "status", getattr(row, "solver_status", "n/a"))))
        drive_text = f"{drive_hours:.1f}h" if np.isfinite(drive_hours) else "n/a"
        budget_text = f"${budget:,.0f}" if np.isfinite(budget) else "n/a"
        rows.append(
            f"""
            <tr>
                <td>{int(row.trip_days)}</td>
                <td>{_escape(row.gateway_start)} → {_escape(row.gateway_end)}</td>
                <td>{_escape(row.days_by_city)}</td>
                <td>{_escape(drive_text)}</td>
                <td>{_escape(budget_text)}</td>
                <td>{status}</td>
            </tr>
            """
        )
    return f"""
    <div class="panel-section-title">Trip Length Comparison</div>
    <table class="method-table">
        <thead><tr><th>Days</th><th>Gateway</th><th>Cities/Bases</th><th>Drive</th><th>Budget</th><th>Status</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    <div class="summary-line muted-note">Saved to production_trip_length_comparison.csv with map-ready stops in production_trip_length_route_stops.csv.</div>
    """



def _build_route_debug_summary_html(route_debug_df):
    if route_debug_df is None or route_debug_df.empty:
        return """
        <div class="panel-section-title">Map Route Validation</div>
        <div class="summary-line">No route validation rows were generated.</div>
        """
    status_counts = route_debug_df["draw_status"].astype(str).value_counts().to_dict()
    ok_count = int(status_counts.get("OK", 0))
    warn_count = int(status_counts.get("WARN", 0))
    failed_count = int(status_counts.get("FAILED", 0))
    visible_count = int(route_debug_df["show_by_default"].astype(bool).sum())
    geometry_counts = (
        route_debug_df.get("geometry_source", pd.Series(dtype=str))
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .head(5)
        .to_dict()
    )
    fallback_count = int(
        pd.to_numeric(route_debug_df.get("straight_line_fallback_count", pd.Series(dtype=float)), errors="coerce")
        .fillna(0)
        .sum()
    )
    road_percent = pd.to_numeric(route_debug_df.get("road_geometry_percent", pd.Series(dtype=float)), errors="coerce")
    road_percent_text = f"{float(road_percent.dropna().mean()):.1f}%" if not road_percent.dropna().empty else "n/a"
    groups = ", ".join(
        f"{group}: {count}"
        for group, count in route_debug_df["layer_group"].astype(str).value_counts().sort_index().items()
    )
    problem_df = route_debug_df[route_debug_df["draw_status"].astype(str).ne("OK")].head(4)
    if problem_df.empty:
        problem_html = '<div class="summary-line muted-note">All audited route layers have drawable geometry.</div>'
    else:
        problem_items = "".join(
            f"<li>{_escape(row.layer_name)}: {_escape(row.issue)}</li>" for row in problem_df.itertuples(index=False)
        )
        problem_html = f"<ol>{problem_items}</ol>"
    table_rows = []
    for group, group_df in route_debug_df.groupby("layer_group", sort=True):
        visible = int(group_df["show_by_default"].astype(bool).sum())
        max_distance = _finite_float(pd.to_numeric(group_df["distance_km"], errors="coerce").max(), 0.0)
        max_points = int(pd.to_numeric(group_df["unique_points"], errors="coerce").fillna(0).max())
        layer_vars = []
        if "layer_var" in group_df.columns:
            layer_vars = [
                str(value)
                for value in group_df["layer_var"].dropna().astype(str).unique().tolist()
                if value and value.lower() != "nan"
            ]
        layer_var_text = ", ".join(layer_vars[:2]) if layer_vars else "HTML audit"
        html_state = "added; remove checked by validator" if visible else "toggle-only"
        table_rows.append(
            f"""
            <tr>
                <td>{_escape(group)}</td>
                <td>{len(group_df)}</td>
                <td>{visible}</td>
                <td>{max_distance:.1f} km</td>
                <td>{max_points}</td>
                <td>{_escape(layer_var_text)}</td>
                <td>{_escape(html_state)}</td>
            </tr>
            """
        )
    family_table_html = f"""
    <table class="method-table">
        <thead><tr><th>Route family</th><th>Rows</th><th>Visible</th><th>Distance</th><th>Pts</th><th>Layer var</th><th>HTML state</th></tr></thead>
        <tbody>{"".join(table_rows)}</tbody>
    </table>
    """
    return f"""
    <div class="panel-section-title">Map Route Validation</div>
    <div class="summary-line"><b>Audited routes:</b> {len(route_debug_df)} ({ok_count} OK, {warn_count} warnings, {failed_count} failed)</div>
    <div class="summary-line"><b>Visible by default:</b> {visible_count} route layers</div>
    <div class="summary-line"><b>Road geometry:</b> avg road/cached coverage {road_percent_text}; fallback flags {fallback_count}; sources {_escape(", ".join(f"{k}: {v}" for k, v in geometry_counts.items()))}</div>
    <div class="summary-line muted-note">Debug rows saved to production_map_route_debug.csv. Layer groups: {_escape(groups)}</div>
    {family_table_html}
    {problem_html}
    """



def _build_html_data_source_summary(output_dir):
    output_dir = Path(output_dir)
    enriched = _load_csv(output_dir / "production_enriched_poi_catalog.csv")
    hotels = _load_csv(output_dir / "production_city_hotel_catalog.csv")
    day_plan = _load_csv(output_dir / "production_day_plan.csv")
    routes = _load_csv(output_dir / "production_intercity_legs.csv")
    source_counts = {}
    if not enriched.empty and "source_list" in enriched.columns:
        source_counts = (
            enriched["source_list"]
            .astype(str)
            .str.replace("|", ";", regex=False)
            .str.split(";")
            .explode()
            .str.strip()
            .value_counts()
            .head(4)
            .to_dict()
        )
    source_text = (
        ", ".join(f"{key}: {value}" for key, value in source_counts.items()) if source_counts else "open/local catalog"
    )
    route_source = "not generated"
    if not routes.empty and "geometry_source" in routes.columns:
        route_source = ", ".join(routes["geometry_source"].astype(str).drop_duplicates().head(3).tolist())
    return f"""
    <div class="panel-section-title">Data Used By This HTML</div>
    <div class="summary-line"><b>POIs:</b> {len(enriched)} enriched rows ({_escape(source_text)})</div>
    <div class="summary-line"><b>Hotels:</b> {len(hotels)} open/project hotel candidates; selected bases shown as red hotel pins.</div>
    <div class="summary-line"><b>Day plan:</b> {len(day_plan)} displayed stops from production_day_plan.csv.</div>
    <div class="summary-line"><b>Roads:</b> production_intercity_legs.csv + OSRM/cache geometry ({_escape(route_source)}).</div>
    <div class="summary-line"><b>Bandit/Gurobi:</b> production_hybrid_bandit_optimization_summary.csv and production_bandit_stress_summary.csv.</div>
    """



def _build_must_go_summary_html(output_dir):
    coverage = _load_csv(Path(output_dir) / "production_must_go_coverage.csv")
    candidates = _load_csv(Path(output_dir) / "production_social_must_go_candidates.csv")
    if coverage.empty and candidates.empty:
        return """
        <div class="panel-section-title">Must-Go Coverage</div>
        <div class="summary-line muted-note">Must-go coverage artifact not generated yet.</div>
        """
    if not coverage.empty:
        balanced = coverage[coverage["method"].astype(str).eq("hierarchical_gurobi_pipeline")].copy()
        if balanced.empty:
            balanced = coverage.copy()
        total = int(balanced["must_go_name"].nunique())
        selected = int(balanced[balanced["selected"].astype(bool)]["must_go_name"].nunique())
        skipped_names = sorted(
            balanced.loc[~balanced["selected"].astype(bool), "must_go_name"].dropna().astype(str).unique().tolist()
        )
        policy = str(balanced.get("must_go_policy", pd.Series(["soft_reward_not_mandatory"])).dropna().iloc[0])
    else:
        total = int(candidates["name"].nunique())
        selected = 0
        skipped_names = sorted(candidates["name"].dropna().astype(str).unique().tolist())
        policy = "soft_reward_not_mandatory"
    skipped_text = "; ".join(skipped_names[:6]) if skipped_names else "none"
    if len(skipped_names) > 6:
        skipped_text += f"; +{len(skipped_names) - 6} more"
    return f"""
    <div class="panel-section-title">Must-Go Coverage</div>
    <div class="summary-line"><b>Policy:</b> {_escape(policy)}</div>
    <div class="summary-line"><b>Selected:</b> {selected} of {total}</div>
    <div class="summary-line"><b>Skipped:</b> {_escape(skipped_text)}</div>
    """



def _build_hotel_summary_html(output_dir):
    hotel_debug = _load_csv(Path(output_dir) / "production_hotel_selection_debug.csv")
    if hotel_debug.empty:
        return """
        <div class="panel-section-title">Hotel Choices</div>
        <div class="summary-line muted-note">Hotel selection debug artifact not generated yet.</div>
        """
    selected = hotel_debug[hotel_debug["selected"].astype(bool)].copy()
    rows = []
    for row in selected.head(8).itertuples(index=False):
        score = _finite_float(getattr(row, "hotel_score", np.nan))
        rows.append(
            f"""
            <tr>
                <td>{_escape(getattr(row, "city", ""))}</td>
                <td>{_escape(getattr(row, "hotel_name", ""))}</td>
                <td>{_escape(f"{score:.2f}" if np.isfinite(score) else "n/a")}</td>
            </tr>
            """
        )
    body = "".join(rows) if rows else '<tr><td colspan="3">No selected hotel rows found.</td></tr>'
    return f"""
    <div class="panel-section-title">Hotel Choices</div>
    <table>
        <thead><tr><th>City</th><th>Selected hotel/base</th><th>Score</th></tr></thead>
        <tbody>{body}</tbody>
    </table>
    """



def _add_route_debug_controls(map_object, route_registry, hidden_default_layers=None):
    add_route_debug_controls(map_object, route_registry, hidden_default_layers)



def _add_blueprint_panel_behavior(map_object):
    behavior_js = """
    (function() {
        function clamp(value, minValue, maxValue) {
            return Math.max(minValue, Math.min(maxValue, value));
        }

        function setCollapsed(panel, collapsed) {
            var header = panel.querySelector('.blueprint-panel-header');
            var chevron = panel.querySelector('.blueprint-panel-chevron');
            panel.classList.toggle('blueprint-collapsed', collapsed);
            if (header) {
                header.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
            }
            if (chevron) {
                chevron.textContent = collapsed ? '+' : '-';
            }
        }

        function initPanel(panel) {
            if (!panel || panel.dataset.blueprintPanelReady) {
                return;
            }
            panel.dataset.blueprintPanelReady = '1';
            var header = panel.querySelector('.blueprint-panel-header');
            if (!header) {
                return;
            }
            if (window.L && L.DomEvent) {
                L.DomEvent.disableClickPropagation(panel);
                L.DomEvent.disableScrollPropagation(panel);
            }
            header.addEventListener('click', function(event) {
                if (panel.dataset.dragSuppressClick === '1') {
                    panel.dataset.dragSuppressClick = '0';
                    event.preventDefault();
                    return;
                }
                setCollapsed(panel, !panel.classList.contains('blueprint-collapsed'));
            });
            if (!panel.classList.contains('blueprint-draggable')) {
                return;
            }
            var dragging = false;
            var moved = false;
            var startX = 0;
            var startY = 0;
            var startLeft = 0;
            var startTop = 0;
            header.addEventListener('pointerdown', function(event) {
                if (event.button !== 0) {
                    return;
                }
                dragging = true;
                moved = false;
                startX = event.clientX;
                startY = event.clientY;
                var rect = panel.getBoundingClientRect();
                startLeft = rect.left;
                startTop = rect.top;
                panel.classList.add('blueprint-dragging');
                header.setPointerCapture(event.pointerId);
            });
            header.addEventListener('pointermove', function(event) {
                if (!dragging) {
                    return;
                }
                var dx = event.clientX - startX;
                var dy = event.clientY - startY;
                if (Math.abs(dx) + Math.abs(dy) > 4) {
                    moved = true;
                }
                var maxLeft = Math.max(8, window.innerWidth - panel.offsetWidth - 8);
                var maxTop = Math.max(8, window.innerHeight - panel.offsetHeight - 8);
                panel.style.left = clamp(startLeft + dx, 8, maxLeft) + 'px';
                panel.style.top = clamp(startTop + dy, 8, maxTop) + 'px';
                panel.style.right = 'auto';
                panel.style.bottom = 'auto';
            });
            function stopDrag(event) {
                if (!dragging) {
                    return;
                }
                dragging = false;
                panel.classList.remove('blueprint-dragging');
                panel.dataset.dragSuppressClick = moved ? '1' : '0';
                try {
                    header.releasePointerCapture(event.pointerId);
                } catch (err) {}
            }
            header.addEventListener('pointerup', stopDrag);
            header.addEventListener('pointercancel', stopDrag);
        }

        function initBlueprintPanels() {
            document.querySelectorAll('.blueprint-floating-panel').forEach(initPanel);
        }

        window.initBlueprintPanels = initBlueprintPanels;
        window.setTimeout(initBlueprintPanels, 50);
        window.setTimeout(initBlueprintPanels, 500);
        window.addEventListener('resize', function() {
            document.querySelectorAll('.blueprint-floating-panel').forEach(function(panel) {
                var rect = panel.getBoundingClientRect();
                if (rect.right > window.innerWidth || rect.bottom > window.innerHeight) {
                    panel.style.left = clamp(rect.left, 8, Math.max(8, window.innerWidth - panel.offsetWidth - 8)) + 'px';
                    panel.style.top = clamp(rect.top, 8, Math.max(8, window.innerHeight - panel.offsetHeight - 8)) + 'px';
                    panel.style.right = 'auto';
                    panel.style.bottom = 'auto';
                }
            });
        });
    })();
    """
    map_object.get_root().script.add_child(folium.Element(behavior_js))



def _build_day_panel(day_plan_df):
    cards = []
    for day, group in day_plan_df.groupby("day", sort=True):
        first = group.iloc[0]
        route_type = str(getattr(first, "route_type", "base_city_local"))
        overnight_city = str(getattr(first, "overnight_city", getattr(first, "city", "")))
        if route_type.startswith("relocation"):
            visited_cities = _unique_in_order(group["city"].astype(str).tolist() + [overnight_city])
            title_city = " -> ".join(visited_cities)
            drive_line = (
                f'<div class="hotel-line">Drive: {float(getattr(first, "drive_minutes_to_next_base", 0.0) or 0.0):.0f} min; '
                f"visit window after drive: {float(getattr(first, 'available_visit_minutes', 0.0) or 0.0):.0f} min</div>"
            )
        else:
            title_city = overnight_city
            drive_line = ""
        stops = "".join(
            f'<li><span>{int(row.stop_order)}.</span> {_escape(row.attraction_name)} <span class="muted-city">({_escape(row.city)})</span></li>'
            for row in group.itertuples(index=False)
        )
        cards.append(
            f"""
            <div class="day-card">
                <div class="day-card-title">Day {int(day)} · {_escape(title_city)}</div>
                <div class="hotel-line">Overnight base: {_escape(overnight_city)} · {_escape(first.hotel_name)}</div>
                {drive_line}
                <ol>{stops}</ol>
            </div>
            """
        )
    return "".join(cards)



def _add_interest_bar_preview_panel(map_object, output_dir):
    output_dir = Path(output_dir)
    preview_path = output_dir / "production_interest_bar_preview.json"
    if preview_path.exists():
        try:
            preview_payload = json.loads(preview_path.read_text(encoding="utf-8"))
        except Exception:
            preview_payload = {}
    else:
        preview_payload = {}
    if not preview_payload:
        preview_payload = {
            "enabled": False,
            "weights": {"nature": 0.25, "city": 0.25, "culture": 0.25, "history": 0.25},
            "lambdas": {
                "lambda_fit": 0.65,
                "lambda_park": 0.35,
                "lambda_weather": 0.30,
                "lambda_season": 0.20,
                "lambda_detour": 0.006,
            },
            "top_boosted_pois": [],
            "route_mix": {},
        }
    preview_json = json.dumps(preview_payload).replace("</", "<\\/")
    panel_html = (
        """
    <style>
    #blueprint-interest-panel {
        bottom: 46px;
        right: 18px;
        width: 360px;
        z-index: 9999;
    }
    #blueprint-interest-panel.blueprint-collapsed {
        width: 230px;
    }
    #blueprint-interest-panel .blueprint-panel-body {
        max-height: 420px;
    }
    .interest-axis-row {
        align-items: center;
        display: grid;
        gap: 8px;
        grid-template-columns: 58px minmax(0, 1fr) 42px;
        margin: 7px 0;
    }
    .interest-axis-row label {
        font-weight: 800;
    }
    .interest-axis-row input[type="range"] {
        accent-color: #2A9D8F;
        width: 100%;
    }
    .interest-percent {
        color: #334155;
        font-weight: 800;
        text-align: right;
    }
    .interest-preview-list {
        border-top: 1px solid #E5E7EB;
        margin-top: 9px;
        padding-top: 8px;
    }
    .interest-preview-row {
        border-top: 1px solid #EEF2F7;
        display: grid;
        gap: 8px;
        grid-template-columns: minmax(0, 1fr) auto;
        padding: 5px 0;
    }
    .interest-preview-name {
        font-weight: 800;
        overflow-wrap: anywhere;
    }
    .interest-preview-meta {
        color: #64748B;
        font-size: 10px;
    }
    .interest-preview-score {
        color: #166534;
        font-weight: 900;
        white-space: nowrap;
    }
    .interest-mix-row {
        align-items: center;
        display: grid;
        gap: 8px;
        grid-template-columns: 58px minmax(0, 1fr) 42px;
        margin-top: 5px;
    }
    .interest-mix-bar {
        background: #E5E7EB;
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
    }
    .interest-mix-fill {
        background: #2A9D8F;
        display: block;
        height: 100%;
    }
    .interest-preview-note {
        color: #64748B;
        font-size: 10px;
        line-height: 1.35;
        margin-top: 8px;
    }
    </style>
    <script id="blueprint-interest-preview-data" type="application/json">"""
        + preview_json
        + """</script>
    <div id="blueprint-interest-panel" class="blueprint-floating-panel blueprint-draggable blueprint-collapsed" data-panel-id="interest-preview">
        <button class="blueprint-panel-header" type="button" aria-expanded="false">
            <span>Interest Mix</span>
            <span class="blueprint-panel-chevron">+</span>
        </button>
        <div class="blueprint-panel-body">
            <div class="interest-axis-row"><label for="interest-axis-nature">Nature</label><input id="interest-axis-nature" type="range" min="0" max="100" step="1" data-interest-axis="nature"><span class="interest-percent" data-interest-percent="nature">25%</span></div>
            <div class="interest-axis-row"><label for="interest-axis-city">City</label><input id="interest-axis-city" type="range" min="0" max="100" step="1" data-interest-axis="city"><span class="interest-percent" data-interest-percent="city">25%</span></div>
            <div class="interest-axis-row"><label for="interest-axis-culture">Culture</label><input id="interest-axis-culture" type="range" min="0" max="100" step="1" data-interest-axis="culture"><span class="interest-percent" data-interest-percent="culture">25%</span></div>
            <div class="interest-axis-row"><label for="interest-axis-history">History</label><input id="interest-axis-history" type="range" min="0" max="100" step="1" data-interest-axis="history"><span class="interest-percent" data-interest-percent="history">25%</span></div>
            <div class="interest-preview-list">
                <div class="panel-section-title">Top Boosted Places</div>
                <div id="interest-preview-rows"></div>
            </div>
            <div class="interest-preview-list">
                <div class="panel-section-title">Route Mix Preview</div>
                <div id="interest-route-mix"></div>
                <div class="summary-line"><b>Selected-route match:</b> <span id="interest-match-score">n/a</span></div>
            </div>
            <div class="interest-preview-note">Preview ranking updates instantly. Full optimized routes use these weights when the Python pipeline is rerun.</div>
        </div>
    </div>
    """
    )
    panel_js = """
    (function() {
        var axes = ["nature", "city", "culture", "history"];
        var payloadNode = document.getElementById("blueprint-interest-preview-data");
        var payload = {};
        try {
            payload = JSON.parse(payloadNode ? payloadNode.textContent : "{}");
        } catch (err) {
            payload = {};
        }
        var weights = Object.assign({nature: 0.25, city: 0.25, culture: 0.25, history: 0.25}, payload.weights || {});
        var lambdas = Object.assign({lambda_fit: 0.65, lambda_park: 0.35, lambda_weather: 0.30, lambda_season: 0.20, lambda_detour: 0.006}, payload.lambdas || {});
        var pois = payload.top_boosted_pois || [];

        function normalizedWeights() {
            var raw = {};
            var total = 0;
            axes.forEach(function(axis) {
                var slider = document.querySelector('[data-interest-axis="' + axis + '"]');
                var value = slider ? Number(slider.value || 0) : Number(weights[axis] || 0) * 100;
                raw[axis] = Math.max(0, value);
                total += raw[axis];
            });
            if (!total) {
                total = axes.length;
                axes.forEach(function(axis) { raw[axis] = 1; });
            }
            var normalized = {};
            axes.forEach(function(axis) { normalized[axis] = raw[axis] / total; });
            return normalized;
        }

        function scorePoi(poi, w) {
            var fit = w.nature * Number(poi.nature || 0)
                + w.city * Number(poi.city_axis || 0)
                + w.culture * Number(poi.culture || 0)
                + w.history * Number(poi.history || 0);
            return Number(poi.final_poi_value || 0)
                + lambdas.lambda_fit * fit
                + lambdas.lambda_park * Number(poi.park_bonus || 0)
                - lambdas.lambda_weather * Number(poi.weather_sensitivity || 0) * Number(poi.weather_risk || 0.15)
                - lambdas.lambda_season * Number(poi.seasonality_risk || 0)
                - lambdas.lambda_detour * Number(poi.detour_minutes || 0);
        }

        function updatePreview() {
            var w = normalizedWeights();
            axes.forEach(function(axis) {
                var percent = Math.round(w[axis] * 100);
                var label = document.querySelector('[data-interest-percent="' + axis + '"]');
                if (label) { label.textContent = percent + "%"; }
            });
            var scored = pois.map(function(poi) {
                var score = scorePoi(poi, w);
                return Object.assign({}, poi, {preview_score: score, preview_delta: score - Number(poi.final_poi_value || 0)});
            }).sort(function(a, b) {
                return b.preview_delta - a.preview_delta || b.preview_score - a.preview_score;
            }).slice(0, 8);
            var rowsNode = document.getElementById("interest-preview-rows");
            if (rowsNode) {
                rowsNode.innerHTML = scored.length ? scored.map(function(poi, index) {
                    return '<div class="interest-preview-row"><div><div class="interest-preview-name">' +
                        (index + 1) + '. ' + escapeHtml(poi.name || "Unnamed") +
                        '</div><div class="interest-preview-meta">' + escapeHtml(poi.city || "") +
                        (poi.park_type ? ' · ' + escapeHtml(poi.park_type) : '') +
                        '</div></div><div class="interest-preview-score">+' + poi.preview_delta.toFixed(2) + '</div></div>';
                }).join("") : '<div class="summary-line">Interest preview data not generated yet.</div>';
            }
            renderMix(scored, w);
        }

        function renderMix(scored, w) {
            var mix = {nature: 0, city: 0, culture: 0, history: 0};
            if (scored.length) {
                scored.forEach(function(poi) {
                    mix.nature += Number(poi.nature || 0);
                    mix.city += Number(poi.city_axis || 0);
                    mix.culture += Number(poi.culture || 0);
                    mix.history += Number(poi.history || 0);
                });
                axes.forEach(function(axis) { mix[axis] = mix[axis] / scored.length; });
            }
            var error = axes.reduce(function(acc, axis) { return acc + Math.abs(mix[axis] - w[axis]); }, 0);
            var match = Math.max(0, 1 - error);
            var mixNode = document.getElementById("interest-route-mix");
            if (mixNode) {
                mixNode.innerHTML = axes.map(function(axis) {
                    var pct = Math.round(mix[axis] * 100);
                    return '<div class="interest-mix-row"><span>' + axis.charAt(0).toUpperCase() + axis.slice(1) +
                        '</span><span class="interest-mix-bar"><span class="interest-mix-fill" style="width:' + Math.max(0, Math.min(100, pct)) +
                        '%"></span></span><span class="interest-percent">' + pct + '%</span></div>';
                }).join("");
            }
            var scoreNode = document.getElementById("interest-match-score");
            if (scoreNode) { scoreNode.textContent = match.toFixed(2); }
        }

        function escapeHtml(value) {
            return String(value).replace(/[&<>"']/g, function(ch) {
                return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[ch];
            });
        }

        function initInterestPreview() {
            axes.forEach(function(axis) {
                var slider = document.querySelector('[data-interest-axis="' + axis + '"]');
                if (!slider) { return; }
                slider.value = Math.round(Number(weights[axis] || 0.25) * 100);
                slider.addEventListener("input", updatePreview);
            });
            updatePreview();
        }
        window.setTimeout(initInterestPreview, 80);
        window.setTimeout(initInterestPreview, 600);
    })();
    """
    map_object.get_root().html.add_child(folium.Element(panel_html))
    map_object.get_root().script.add_child(folium.Element(panel_js))



def _load_default_hierarchical_gurobi_day_plan(output_dir, fallback_df):
    route_stops = _load_csv(Path(output_dir) / "production_method_route_stops.csv")
    if route_stops.empty or "method" not in route_stops.columns:
        return fallback_df
    route_stops = route_stops[route_stops["method"].astype(str).eq("hierarchical_gurobi_pipeline")].copy()
    if route_stops.empty:
        return fallback_df

    route_stops["day"] = _numeric_catalog_series(route_stops, "day", 1).astype(int)
    route_stops["stop_order"] = _numeric_catalog_series(route_stops, "stop_order", 1).astype(int)
    route_stops = route_stops.sort_values(["day", "stop_order", "attraction_name"]).reset_index(drop=True)
    total_days = int(route_stops["day"].max()) if not route_stops.empty else 0
    output = route_stops.copy()
    def scalar_series(default):
        return pd.Series(default, index=output.index)
    output["profile"] = "balanced"
    output["profile_label"] = "Balanced"
    output["hotel_source"] = output.get("hotel_source", "production_method_route_stops")
    output["overnight_base"] = output["city"].astype(str).eq(output["overnight_city"].astype(str))
    output["stop_is_overnight_city"] = output["overnight_base"]
    output["hotel_booked"] = output["day"].astype(int) < total_days
    output["route_start_city"] = output.get("route_start_city", output["city"])
    output["route_end_city"] = output.get("route_end_city", output["overnight_city"])
    output["pass_through_cities"] = output.get("pass_through_cities", "")
    output["drive_minutes_to_next_base"] = pd.to_numeric(
        output.get("drive_minutes_to_next_base", scalar_series(0.0)),
        errors="coerce",
    ).fillna(0.0)
    output["available_visit_minutes"] = pd.to_numeric(
        output.get("available_visit_minutes", scalar_series(0.0)),
        errors="coerce",
    ).fillna(0.0)
    output["drive_time_source"] = output.get("drive_time_source", "production_method_route_stops")
    output["attraction_source"] = output.get(
        "attraction_source", output.get("source_list", "production_method_route_stops")
    )
    output["must_go_weight"] = pd.to_numeric(output.get("must_go_weight", scalar_series(0.0)), errors="coerce").fillna(
        0.0
    )
    output["corridor_fit"] = pd.to_numeric(output.get("corridor_fit", scalar_series(0.0)), errors="coerce").fillna(0.0)
    output["detour_minutes"] = pd.to_numeric(output.get("detour_minutes", scalar_series(0.0)), errors="coerce").fillna(
        0.0
    )
    output["data_confidence"] = pd.to_numeric(
        output.get("data_confidence", scalar_series(0.5)), errors="coerce"
    ).fillna(0.5)
    output["social_reason"] = output.get("social_reason", "")
    output["pass_through_cities"] = output.get("pass_through_cities", "")
    return output
