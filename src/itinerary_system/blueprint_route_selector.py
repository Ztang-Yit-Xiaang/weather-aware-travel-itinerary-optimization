"""Route-selector model and exact HTML/JavaScript assets for the Folium map."""

from __future__ import annotations

import json

import folium
import numpy as np

from .blueprint_render_primitives import _escape, _finite_float


def _validate_route_registry(route_registry):
    control_ids = set()
    for index, route in enumerate(route_registry):
        control_id = str(route.get("control_id", route.get("label", "route")))
        if control_id in control_ids:
            raise ValueError(f"duplicate route selector control_id: {control_id}")
        control_ids.add(control_id)

        layer_var = route.get("layer_var")
        if not isinstance(layer_var, str) or not layer_var.strip():
            raise ValueError(f"route selector entry {index} ({control_id}) is missing layer_var")

        default_visible = bool(route.get("default_visible", False))
        default_checked = bool(route.get("default_checked", default_visible))
        if default_checked != default_visible:
            raise ValueError(
                f"route selector entry {index} ({control_id}) has mismatched default_checked/default_visible"
            )


def _route_checkbox_html(route, *, compact=False):
        color = _escape(str(route.get("color", "#2563EB")))
        control_id = _escape(str(route.get("control_id", route.get("label", "route"))))
        control_label = _escape(str(route.get("control_label", route.get("label", "route"))))
        family = _escape(str(route.get("family", "route")))
        checked = " checked" if bool(route.get("default_checked", route.get("default_visible", False))) else ""
        distance = _finite_float(route.get("distance_km"), 0.0)
        unique_points = int(route.get("unique_points", 0) or 0)
        meta_text = (
            f"{distance:.0f} km"
            if np.isfinite(distance) and distance > 0
            else f"{unique_points} pts"
            if unique_points
            else ""
        )
        compact_class = " blueprint-route-check-compact" if compact else ""
        return f"""
            <label class="blueprint-route-check{compact_class}" data-route-family="{family}">
                <input type="checkbox" data-route-checkbox="{control_id}" data-route-family="{family}"{checked}>
                <span class="blueprint-route-swatch" style="background:{color};"></span>
                <span class="blueprint-route-check-text">{control_label}</span>
                <span class="blueprint-route-meta">{_escape(meta_text)}</span>
            </label>
            """


def _build_route_checkbox_html(route_registry):
    matrix_full_routes = [
            route
            for route in route_registry
            if str(route.get("selector_group")) == "route_matrix" and str(route.get("layer_role")) == "full_route"
        ]
    if matrix_full_routes:

            child_routes = {}
            for route in route_registry:
                role = str(route.get("layer_role", ""))
                if str(route.get("selector_group")) != "route_matrix" or role in {"full_route", "selected_hotels"}:
                    continue
                parent_key = str(route.get("parent_route_key", route.get("route_key", "")))
                child_routes.setdefault(parent_key, []).append(route)
            day_values = sorted(
                {
                    int(_finite_float(route.get("trip_days"), 0))
                    for route in matrix_full_routes
                    if int(_finite_float(route.get("trip_days"), 0)) > 0
                }
            )
            method_order = [
                "hierarchical_gurobi_pipeline",
                "hierarchical_greedy_baseline",
                "hierarchical_bandit_gurobi_repair",
            ]
            profile_order = ["relaxed", "balanced", "explorer"]
            matrix_html = []
            for day_value in day_values:
                day_routes = [
                    route for route in matrix_full_routes if int(_finite_float(route.get("trip_days"), 0)) == day_value
                ]
                method_blocks = []
                for method in method_order:
                    method_routes = [route for route in day_routes if str(route.get("method")) == method]
                    if not method_routes:
                        continue
                    method_label = str(method_routes[0].get("method_label", method_routes[0].get("method", "Method")))
                    method_routes = sorted(
                        method_routes,
                        key=lambda route: (
                            profile_order.index(str(route.get("profile")))
                            if str(route.get("profile")) in profile_order
                            else 99
                        ),
                    )
                    route_rows = []
                    for route in method_routes:
                        route_key = str(route.get("route_key"))
                        children = sorted(
                            child_routes.get(route_key, []),
                            key=lambda child: (
                                ["city_loop", "transition", "must_go"].index(str(child.get("layer_role")))
                                if str(child.get("layer_role")) in ["city_loop", "transition", "must_go"]
                                else 99
                            ),
                        )
                        child_rows = "".join(_route_checkbox_html(child, compact=True) for child in children)
                        route_rows.append(
                            f"""
                        <details class="blueprint-route-leaf" data-route-key="{_escape(route_key)}">
                            <summary>{_route_checkbox_html(route)}<span class="blueprint-route-subhint">details</span></summary>
                            <div class="blueprint-route-subsection">
                                <button type="button" class="blueprint-route-hotel-note" data-route-action="hotel_showcase">
                                    Hotel Showcase controls selected hotels for checked routes.
                                </button>
                                {child_rows}
                            </div>
                        </details>
                        """
                        )
                    method_blocks.append(
                        f"""
                    <details class="blueprint-route-method" data-method="{_escape(method)}">
                        <summary>{_escape(method_label)} <span>{len(method_routes)}</span></summary>
                        {"".join(route_rows)}
                    </details>
                    """
                    )
                matrix_html.append(
                    f"""
                <details class="blueprint-route-group blueprint-route-matrix-day" data-selector-group="route_matrix" open>
                    <summary>{int(day_value)}-Day Routes <span>{len(day_routes)}</span></summary>
                    {"".join(method_blocks)}
                </details>
                """
                )
            auxiliary_rows = []
            for route in route_registry:
                if str(route.get("selector_group")) == "route_matrix":
                    continue
                if str(route.get("layer_role")) == "selected_hotels":
                    continue
                auxiliary_rows.append(_route_checkbox_html(route))
            if auxiliary_rows:
                matrix_html.append(
                    f"""
                <details class="blueprint-route-group" data-selector-group="context">
                    <summary>Context / Candidates <span>{len(auxiliary_rows)}</span></summary>
                    {"".join(auxiliary_rows)}
                </details>
                """
                )
            checkbox_html = "".join(matrix_html)
    else:
            group_order = [
                ("core", "Core"),
                ("trip_length", "Trip Length"),
                ("method", "Methods"),
                ("traveler_profile", "Travelers"),
                ("city_detail", "City Details"),
                ("transition", "Transitions"),
                ("must_go", "Must-Go"),
                ("hotel", "Hotels"),
                ("context", "Context Routes"),
            ]
            grouped_routes = {key: [] for key, _ in group_order}
            for route in route_registry:
                selector_group = str(route.get("selector_group", route.get("family", "context")))
                if selector_group not in grouped_routes:
                    grouped_routes[selector_group] = []
                grouped_routes[selector_group].append(_route_checkbox_html(route))
            group_html = []
            for group_key, group_label in group_order:
                rows = grouped_routes.get(group_key, [])
                if not rows:
                    continue
                open_attr = " open" if group_key in {"core", "trip_length", "method"} else ""
                group_html.append(
                    f"""
                <details class="blueprint-route-group" data-selector-group="{_escape(group_key)}"{open_attr}>
                    <summary>{_escape(group_label)} <span>{len(rows)}</span></summary>
                    {"".join(rows)}
                </details>
                """
                )
            checkbox_html = "".join(group_html)
    return checkbox_html


def build_route_selector_model(map_object, route_registry, hidden_default_layers=None):
    route_registry = list(route_registry or [])
    _validate_route_registry(route_registry)
    map_var = map_object.get_name()
    hidden_default_layers = hidden_default_layers or []
    hidden_vars = [
            layer.get_name() for layer in hidden_default_layers if layer is not None and hasattr(layer, "get_name")
        ]
    bounds_points = []
    for route in route_registry:
            bounds = route.get("bounds") or []
            if len(bounds) == 2:
                bounds_points.extend(bounds)
    registry_json = json.dumps(route_registry).replace("</", "<\\/")
    hidden_json = json.dumps(hidden_vars)
    bounds_json = json.dumps(bounds_points)
    families = sorted({str(route.get("family", "")) for route in route_registry})
    checked_count = sum(
            1 for route in route_registry if bool(route.get("default_checked", route.get("default_visible", False)))
        )
    checkbox_html = _build_route_checkbox_html(route_registry)
    return {
        "map_var": map_var,
        "registry_json": registry_json,
        "hidden_json": hidden_json,
        "bounds_json": bounds_json,
        "checkbox_html": checkbox_html,
        "checked_count": checked_count,
        "route_count": len(route_registry),
        "hidden_count": len(hidden_vars),
        "families_label": _escape(", ".join(families)),
    }


_CONTROLS_HTML_PARTS = (
    '''
    <style>
    #blueprint-route-selector {
        position: fixed;
        left: 74px;
        top: 82px;
        z-index: 9996;
        width: 430px;
    }
    #blueprint-route-selector.blueprint-collapsed {
        width: 230px;
    }
    @media (max-width: 519px) {
        #blueprint-route-selector {
            left: 12px;
            width: calc(100vw - 24px);
        }
    }
    #blueprint-route-selector .blueprint-panel-body {
        max-height: 420px;
    }
    .blueprint-route-actions {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 6px;
        margin-bottom: 10px;
    }
    .blueprint-route-action {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        color: #1F2937;
        cursor: pointer;
        font: inherit;
        font-weight: 800;
        padding: 5px 7px;
    }
    .blueprint-route-action:hover {
        background: #EEF2F7;
    }
    .blueprint-route-check {
        align-items: center;
        border-top: 1px solid #EDF0F3;
        cursor: pointer;
        display: grid;
        column-gap: 8px;
        grid-template-columns: 17px 14px minmax(0, 1fr) minmax(48px, auto);
        padding: 6px 0;
        min-width: 0;
    }
    .blueprint-route-check input {
        margin: 0;
    }
    .blueprint-route-swatch {
        border: 1px solid rgba(17,24,39,0.24);
        border-radius: 999px;
        display: inline-block;
        height: 12px;
        width: 12px;
    }
    .blueprint-route-check-text {
        line-height: 1.18;
        min-width: 0;
        overflow-wrap: anywhere;
        padding-right: 4px;
    }
    .blueprint-route-meta {
        color: #64748B;
        font-size: 10px;
        font-weight: 700;
        justify-self: end;
        min-width: 48px;
        text-align: right;
        white-space: nowrap;
    }
    .blueprint-route-check-compact {
        margin-left: 18px;
        padding: 5px 0;
    }
    .blueprint-route-leaf {
        border-top: 1px solid #EDF0F3;
    }
    .blueprint-route-leaf > summary {
        align-items: center;
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 8px;
        list-style: none;
    }
    .blueprint-route-leaf > summary::-webkit-details-marker {
        display: none;
    }
    .blueprint-route-leaf > summary .blueprint-route-check {
        border-top: 0;
    }
    .blueprint-route-method {
        border-top: 1px solid #E5E7EB;
        margin-left: 6px;
    }
    .blueprint-route-method > summary {
        cursor: pointer;
        font-weight: 800;
        list-style: none;
        padding: 5px 0;
    }
    .blueprint-route-method > summary::-webkit-details-marker {
        display: none;
    }
    .blueprint-route-method > summary span {
        color: #64748B;
        float: right;
        font-size: 10px;
    }
    .blueprint-route-subsection {
        background: #F8FAFC;
        border-left: 2px solid #CBD5E1;
        margin: 0 0 5px 20px;
        padding: 5px 8px;
    }
    .blueprint-route-subhint {
        background: #EEF2F7;
        border: 1px solid #CBD5E1;
        border-radius: 999px;
        color: #64748B;
        font-size: 10px;
        font-weight: 700;
        line-height: 1;
        padding: 4px 7px;
        white-space: nowrap;
    }
    .blueprint-route-hotel-note {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        color: #475569;
        cursor: pointer;
        display: block;
        font-size: 10px;
        font-weight: 700;
        margin: 2px 0 6px 18px;
        padding: 5px 7px;
        text-align: left;
        width: calc(100% - 18px);
    }
    .blueprint-route-hotel-note:hover {
        background: #EEF2F7;
    }
    .blueprint-route-check input:disabled + .blueprint-route-swatch,
    .blueprint-route-check input:disabled ~ .blueprint-route-check-text,
    .blueprint-route-check input:disabled ~ .blueprint-route-meta {
        opacity: 0.48;
    }
    #blueprint-hotel-showcase-toggle.blueprint-active {
        background: #0F172A;
        border-color: #0F172A;
        color: #FFFFFF;
    }
    .blueprint-route-group {
        border-top: 1px solid #E5E7EB;
        padding: 4px 0;
    }
    .blueprint-route-group summary {
        color: #111827;
        cursor: pointer;
        font-weight: 900;
        list-style: none;
        padding: 6px 0;
    }
    .blueprint-route-group summary::-webkit-details-marker {
        display: none;
    }
    .blueprint-route-group summary span {
        color: #64748B;
        float: right;
        font-size: 10px;
        padding-top: 2px;
    }
    #blueprint-route-debug-tab {
        align-items: center;
        background: #0F172A;
        border: 1px solid rgba(255,255,255,0.55);
        border-left: 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.22);
        color: #FFFFFF;
        cursor: pointer;
        display: flex;
        font-size: 11px;
        font-weight: 900;
        height: 88px;
        justify-content: center;
        left: 0;
        letter-spacing: 0;
        padding: 6px 4px;
        position: fixed;
        text-orientation: mixed;
        top: 44%;
        transform: translateY(-50%);
        writing-mode: vertical-rl;
        width: 21px;
        z-index: 10001;
    }
    #blueprint-route-debug-tab:hover {
        background: #1E293B;
    }
    #blueprint-route-debug-panel {
        left: -326px;
        top: 250px;
        width: 310px;
        z-index: 10000;
        transition: left 180ms ease, opacity 180ms ease;
        opacity: 0.0;
        pointer-events: none;
    }
    #blueprint-route-debug-panel.blueprint-debug-open {
        left: 24px;
        opacity: 1.0;
        pointer-events: auto;
    }
    #blueprint-route-debug-panel .blueprint-panel-body {
        max-height: 300px;
    }
    </style>
    <script id="blueprint-route-debug-registry" type="application/json">''',
    '''</script>
    <div id="blueprint-route-selector" class="blueprint-floating-panel blueprint-draggable" data-panel-id="route-selector">
        <button class="blueprint-panel-header" type="button" aria-expanded="true">
            <span>Route Selector</span>
            <span class="blueprint-panel-chevron">-</span>
        </button>
        <div class="blueprint-panel-body">
            <div class="blueprint-route-actions">
                <button type="button" class="blueprint-route-action" data-route-action="clear">Clear</button>
                <button type="button" class="blueprint-route-action" data-route-action="balanced">Balanced default</button>
                <button type="button" class="blueprint-route-action" data-route-action="day_7">Show 7-day methods</button>
                <button type="button" class="blueprint-route-action" data-route-action="day_9">Show 9-day methods</button>
                <button type="button" class="blueprint-route-action" data-route-action="day_12">Show 12-day methods</button>
                <button type="button" class="blueprint-route-action" data-route-action="all_balanced">Show all balanced</button>
                <button type="button" class="blueprint-route-action" data-route-action="city_detail">Show city loops</button>
                <button type="button" class="blueprint-route-action" data-route-action="transition">Show transitions</button>
            </div>
            <button type="button" id="blueprint-hotel-showcase-toggle" class="blueprint-route-action" data-route-action="hotel_showcase" aria-pressed="false" style="width:100%; margin-bottom:6px;">Hotel Showcase: Off</button>
            <button type="button" class="blueprint-route-action" data-route-action="zoom" style="width:100%; margin-bottom:8px;">Zoom to checked routes</button>
            <div class="blueprint-route-checkboxes">''',
    '''</div>
        </div>
    </div>
    <button id="blueprint-route-debug-tab" type="button" aria-controls="blueprint-route-debug-panel" aria-expanded="false">Debug</button>
    <div id="blueprint-route-debug-panel" class="blueprint-floating-panel blueprint-collapsed" data-panel-id="route-debug">
        <button class="blueprint-panel-header" type="button" aria-expanded="false">
            <span>Route Debug</span>
            <span class="blueprint-panel-chevron">+</span>
        </button>
        <div class="blueprint-panel-body">
            <div class="summary-line"><b>Default checked routes:</b> ''',
    '''</div>
            <div class="summary-line"><b>Registered checkbox routes:</b> ''',
    '''</div>
            <div class="summary-line"><b>Unchecked on load:</b> ''',
    '''</div>
            <div class="summary-line"><b>Families:</b> ''',
    '''</div>
            <div class="summary-line muted-note">Browser registry: blueprint-route-debug-registry. The Route Selector owns route visibility on first open.</div>
        </div>
    </div>
    ''',
)

_CONTROLS_JS_PARTS = (
    '''
    (function() {
        var registry = ''',
    ''';
        var hiddenLayerVars = ''',
    ''';
        var routeBounds = ''',
    ''';
        var mapVarName = "''',
    '''";
        var mapObject = null;
        var listenersAttached = false;
        var hotelShowcaseEnabled = false;
        var lastDiagnostics = {
            map_ready: false,
            registry_count: registry.length,
            missing_layers: [],
            checked_routes: [],
            visible_checked_routes: [],
            unchecked_routes: [],
            last_sync: null
        };

        function getMapObject() {
            if (mapObject && mapObject.fitBounds && mapObject.hasLayer) {
                return mapObject;
            }
            mapObject = window[mapVarName] || (typeof globalThis !== "undefined" ? globalThis[mapVarName] : null) || null;
            return mapObject;
        }

        function getLayerByName(name) {
            if (!name) {
                return null;
            }
            return window[name] || (typeof globalThis !== "undefined" ? globalThis[name] : null) || null;
        }

        function addLayer(layer) {
            var map = getMapObject();
            if (map && layer && !map.hasLayer(layer)) {
                layer.addTo(map);
            }
        }

        function removeLayer(layer) {
            var map = getMapObject();
            if (map && layer && map.hasLayer(layer)) {
                map.removeLayer(layer);
            }
        }

        function routeById(controlId) {
            return registry.find(function(route) {
                return String(route.control_id) === String(controlId);
            });
        }

        function checkboxForRoute(route) {
            return document.querySelector('[data-route-checkbox="' + route.control_id + '"]');
        }

        function fullRouteForKey(routeKey) {
            return registry.find(function(route) {
                return String(route.route_key) === String(routeKey) && String(route.layer_role) === "full_route";
            });
        }

        function parentRouteChecked(route) {
            var parentKey = route.parent_route_key || route.route_key;
            var parent = fullRouteForKey(parentKey);
            var parentCheckbox = parent ? checkboxForRoute(parent) : null;
            return Boolean(parentCheckbox && parentCheckbox.checked);
        }

        function isRouteChecked(route) {
            if (route.hotel_showcase_controlled || String(route.layer_role) === "selected_hotels") {
                return Boolean(hotelShowcaseEnabled && parentRouteChecked(route));
            }
            if (["city_loop", "transition", "must_go"].indexOf(String(route.layer_role)) !== -1) {
                var childCheckbox = checkboxForRoute(route);
                return Boolean(childCheckbox && childCheckbox.checked && parentRouteChecked(route));
            }
            var checkbox = checkboxForRoute(route);
            return Boolean(checkbox && checkbox.checked);
        }

        function checkedRoutes() {
            return registry.filter(function(route) {
                return isRouteChecked(route);
            });
        }

        function syncCheckboxState() {
            var map = getMapObject();
            var missingLayers = [];
            var checkedIds = [];
            var visibleCheckedIds = [];
            var uncheckedIds = [];
            registry.forEach(function(route) {
                var layer = getLayerByName(route.layer_var);
                var checked = isRouteChecked(route);
                route.runtime_checked = checked;
                route.runtime_layer_found = Boolean(layer);
                route.runtime_hotel_showcase = Boolean(hotelShowcaseEnabled);
                if (!layer) {
                    missingLayers.push(route.layer_var || route.control_id);
                }
                if (checked) {
                    checkedIds.push(route.control_id);
                    addLayer(layer);
                    if (layer && layer.bringToFront) {
                        layer.bringToFront();
                    }
                    if (map && layer && map.hasLayer(layer)) {
                        visibleCheckedIds.push(route.control_id);
                        route.runtime_visible = true;
                    } else {
                        route.runtime_visible = false;
                    }
                } else {
                    uncheckedIds.push(route.control_id);
                    removeLayer(layer);
                    route.runtime_visible = false;
                }
            });
            lastDiagnostics = {
                map_ready: Boolean(map),
                registry_count: registry.length,
                missing_layers: missingLayers,
                checked_routes: checkedIds,
                visible_checked_routes: visibleCheckedIds,
                unchecked_routes: uncheckedIds,
                last_sync: new Date().toISOString()
            };
            window.blueprintCheckedRoutes = checkedIds;
            window.blueprintHotelShowcaseEnabled = hotelShowcaseEnabled;
            window.blueprintRouteRuntimeDiagnostics = lastDiagnostics;
            return Boolean(map) && missingLayers.length === 0;
        }

        function setAllChecked(value) {
            registry.forEach(function(route) {
                var checkbox = checkboxForRoute(route);
                if (checkbox) {
                    checkbox.checked = Boolean(value);
                }
            });
        }

        function setOnly(predicate) {
            registry.forEach(function(route) {
                var checkbox = checkboxForRoute(route);
                if (checkbox) {
                    checkbox.checked = Boolean(predicate(route));
                }
            });
        }

        function checkedFullRouteKeys() {
            var keys = [];
            registry.forEach(function(route) {
                if (!routeIsFullRoute(route)) {
                    return;
                }
                var checkbox = checkboxForRoute(route);
                if (checkbox && checkbox.checked && route.route_key) {
                    keys.push(String(route.route_key));
                }
            });
            return keys;
        }

        function setChildRoleForCheckedRoutes(roleName) {
            var activeKeys = checkedFullRouteKeys();
            var activeSet = new Set(activeKeys);
            var eligibleChildren = registry.filter(function(route) {
                var parentKey = String(route.parent_route_key || route.route_key || "");
                return String(route.layer_role) === String(roleName) && activeSet.has(parentKey);
            });
            var shouldEnable = eligibleChildren.some(function(route) {
                var checkbox = checkboxForRoute(route);
                return checkbox && !checkbox.checked;
            });
            registry.forEach(function(route) {
                if (String(route.layer_role) !== String(roleName)) {
                    return;
                }
                var checkbox = checkboxForRoute(route);
                if (!checkbox) {
                    return;
                }
                var parentKey = String(route.parent_route_key || route.route_key || "");
                checkbox.checked = Boolean(shouldEnable && activeSet.has(parentKey));
            });
        }

        function updateChildAvailability() {
            registry.forEach(function(route) {
                var role = String(route.layer_role || "");
                if (["city_loop", "transition", "must_go"].indexOf(role) === -1) {
                    return;
                }
                var checkbox = checkboxForRoute(route);
                if (!checkbox) {
                    return;
                }
                var disabled = !parentRouteChecked(route);
                checkbox.disabled = disabled;
                var row = checkbox.closest(".blueprint-route-check");
                if (row) {
                    row.classList.toggle("blueprint-route-disabled", disabled);
                }
            });
        }

        function routeHasQuickGroup(route, groupName) {
            return Array.isArray(route.quick_groups) && route.quick_groups.indexOf(groupName) !== -1;
        }

        function routeIsFullRoute(route) {
            return String(route.layer_role || "") === "full_route" || String(route.family || "") === "traveler_profile";
        }

        function updateHotelShowcaseButton() {
            var button = document.getElementById("blueprint-hotel-showcase-toggle");
            if (!button) {
                return;
            }
            button.classList.toggle("blueprint-active", Boolean(hotelShowcaseEnabled));
            button.setAttribute("aria-pressed", hotelShowcaseEnabled ? "true" : "false");
            button.textContent = hotelShowcaseEnabled ? "Hotel Showcase: On" : "Hotel Showcase: Off";
        }

        function zoomToCheckedRoutes() {
            var checked = checkedRoutes();
            var points = [];
            checked.forEach(function(route) {
                if (Array.isArray(route.bounds) && route.bounds.length === 2) {
                    points.push(route.bounds[0]);
                    points.push(route.bounds[1]);
                }
            });
            if (!points.length) {
                points = routeBounds || [];
            }
            var map = getMapObject();
            if (map && points && points.length >= 2) {
                map.fitBounds(points, { padding: [70, 70] });
            }
        }

        function attachRouteControlListeners() {
            if (listenersAttached) {
                return;
            }
            listenersAttached = true;
            var selector = document.getElementById("blueprint-route-selector");
            var debugPanel = document.getElementById("blueprint-route-debug-panel");
            var debugTab = document.getElementById("blueprint-route-debug-tab");
            [selector, debugPanel, debugTab].forEach(function(node) {
                if (node && window.L && L.DomEvent) {
                    L.DomEvent.disableClickPropagation(node);
                    L.DomEvent.disableScrollPropagation(node);
                }
            });
            function setDebugDrawerOpen(open) {
                if (!debugPanel || !debugTab) {
                    return;
                }
                debugPanel.classList.toggle("blueprint-debug-open", Boolean(open));
                debugTab.setAttribute("aria-expanded", open ? "true" : "false");
                if (open) {
                    debugPanel.classList.remove("blueprint-collapsed");
                    var debugHeader = debugPanel.querySelector(".blueprint-panel-header");
                    var chevron = debugPanel.querySelector(".blueprint-panel-chevron");
                    if (debugHeader) {
                        debugHeader.setAttribute("aria-expanded", "true");
                    }
                    if (chevron) {
                        chevron.textContent = "-";
                    }
                }
            }
            window.blueprintSetDebugDrawerOpen = setDebugDrawerOpen;
            if (debugTab) {
                debugTab.addEventListener("click", function(event) {
                    event.preventDefault();
                    setDebugDrawerOpen(!debugPanel.classList.contains("blueprint-debug-open"));
                });
            }
            if (debugPanel) {
                var debugHeader = debugPanel.querySelector(".blueprint-panel-header");
                if (debugHeader) {
                    debugHeader.addEventListener("click", function() {
                        window.setTimeout(function() {
                            if (debugPanel.classList.contains("blueprint-collapsed")) {
                                setDebugDrawerOpen(false);
                            }
                        }, 0);
                    });
                }
            }
            document.querySelectorAll("[data-route-checkbox]").forEach(function(checkbox) {
                checkbox.addEventListener("click", function(event) {
                    event.stopPropagation();
                });
                checkbox.addEventListener("change", function() {
                    updateChildAvailability();
                    syncCheckboxState();
                });
            });
            document.querySelectorAll("[data-route-action]").forEach(function(button) {
                button.addEventListener("click", function(event) {
                    var action = event.currentTarget.getAttribute("data-route-action");
                    if (action === "clear") {
                        setAllChecked(false);
                        hotelShowcaseEnabled = false;
                        updateHotelShowcaseButton();
                    } else if (action === "balanced") {
                        setOnly(function(route) {
                            return route.control_id === "route_matrix_full__d7__bandit_repair__balanced" || route.control_id === "traveler_balanced";
                        });
                    } else if (action === "day_7") {
                        setOnly(function(route) { return routeIsFullRoute(route) && Number(route.trip_days) === 7; });
                    } else if (action === "day_9") {
                        setOnly(function(route) { return routeIsFullRoute(route) && Number(route.trip_days) === 9; });
                    } else if (action === "day_12") {
                        setOnly(function(route) { return routeIsFullRoute(route) && Number(route.trip_days) === 12; });
                    } else if (action === "all_balanced") {
                        setOnly(function(route) { return routeIsFullRoute(route) && String(route.profile) === "balanced"; });
                    } else if (action === "traveler") {
                        setOnly(function(route) { return routeHasQuickGroup(route, "traveler"); });
                    } else if (action === "city_detail") {
                        setChildRoleForCheckedRoutes("city_loop");
                    } else if (action === "transition") {
                        setChildRoleForCheckedRoutes("transition");
                    } else if (action === "must_go") {
                        setChildRoleForCheckedRoutes("must_go");
                    } else if (action === "hotel") {
                        setOnly(function(route) {
                            return routeHasQuickGroup(route, "hotel");
                        });
                    } else if (action === "hotel_showcase") {
                        hotelShowcaseEnabled = !hotelShowcaseEnabled;
                        updateHotelShowcaseButton();
                    }
                    updateChildAvailability();
                    syncCheckboxState();
                    if (action === "zoom" || action !== "clear") {
                        zoomToCheckedRoutes();
                    }
                });
            });
            updateHotelShowcaseButton();
            updateChildAvailability();
        }

        function bootstrapRouteSelector(attempt) {
            attachRouteControlListeners();
            hiddenLayerVars.forEach(function(layerVar) {
                removeLayer(getLayerByName(layerVar));
            });
            updateChildAvailability();
            var ready = syncCheckboxState();
            if (ready) {
                zoomToCheckedRoutes();
                return;
            }
            if (attempt < 60) {
                window.setTimeout(function() {
                    bootstrapRouteSelector(attempt + 1);
                }, 120);
            }
        }

        window.blueprintMapDebug = {
            registry: registry,
            hiddenLayerVars: hiddenLayerVars,
            syncCheckboxState: syncCheckboxState,
            zoomToCheckedRoutes: zoomToCheckedRoutes,
            routeById: routeById
        };

        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", function() {
                bootstrapRouteSelector(0);
            });
        } else {
            bootstrapRouteSelector(0);
        }
        window.addEventListener("load", function() {
            bootstrapRouteSelector(0);
        });
    })();
    ''',
)

def _join_asset_parts(parts, values):
    output = []
    for index, part in enumerate(parts):
        output.append(part)
        if index < len(values):
            output.append(str(values[index]))
    return "".join(output)


def route_selector_controls_html(model):
    return _join_asset_parts(
        _CONTROLS_HTML_PARTS,
        [
            model["registry_json"],
            model["checkbox_html"],
            model["checked_count"],
            model["route_count"],
            model["hidden_count"],
            model["families_label"],
        ],
    )


def route_selector_controls_script(model):
    return _join_asset_parts(
        _CONTROLS_JS_PARTS,
        [
            model["registry_json"],
            model["hidden_json"],
            model["bounds_json"],
            model["map_var"],
        ],
    )


def add_route_debug_controls(map_object, route_registry, hidden_default_layers=None):
    if not route_registry:
        return
    model = build_route_selector_model(map_object, route_registry, hidden_default_layers)
    controls_html = route_selector_controls_html(model)
    controls_js = route_selector_controls_script(model)
    map_object.get_root().html.add_child(folium.Element(controls_html))
    map_object.get_root().script.add_child(folium.Element(controls_js))
