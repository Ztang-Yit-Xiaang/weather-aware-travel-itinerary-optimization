from __future__ import annotations

import json

import folium
import pytest

from itinerary_system import blueprint_route_selector as selector


def _route(control_id: str, layer_var: str, **overrides) -> dict:
    route = {
        "control_id": control_id,
        "control_label": control_id,
        "family": "context",
        "selector_group": "context",
        "layer_var": layer_var,
        "default_checked": False,
        "default_visible": False,
        "bounds": [],
    }
    route.update(overrides)
    return route


def test_empty_registry_builds_empty_model_and_attaches_no_controls():
    trip_map = folium.Map(location=[0.0, 0.0])
    root = trip_map.get_root()
    html_children = tuple(root.html._children)
    script_children = tuple(root.script._children)

    model = selector.build_route_selector_model(trip_map, [])
    selector.add_route_debug_controls(trip_map, [])

    assert model["route_count"] == 0
    assert model["checked_count"] == 0
    assert model["checkbox_html"] == ""
    assert json.loads(model["registry_json"]) == []
    assert json.loads(model["bounds_json"]) == []
    assert tuple(root.html._children) == html_children
    assert tuple(root.script._children) == script_children


def test_selector_model_preserves_matrix_order_defaults_bounds_and_escaping():
    trip_map = folium.Map(location=[0.0, 0.0])
    layers = [folium.FeatureGroup(name=f"Layer {index}").add_to(trip_map) for index in range(7)]
    registry = [
        _route(
            "d12_greedy_explorer",
            layers[0].get_name(),
            selector_group="route_matrix",
            layer_role="full_route",
            route_key="d12_greedy_explorer",
            trip_days=12,
            method="hierarchical_greedy_baseline",
            method_label="Greedy",
            profile="explorer",
        ),
        _route(
            "d7_gurobi_explorer",
            layers[1].get_name(),
            control_label="Explorer",
            selector_group="route_matrix",
            layer_role="full_route",
            route_key="d7_gurobi_explorer",
            trip_days=7,
            method="hierarchical_gurobi_pipeline",
            method_label="Gurobi",
            profile="explorer",
        ),
        _route(
            "d7_gurobi_relaxed",
            layers[2].get_name(),
            control_label="Relaxed Route </script><b>&",
            family="route <family>",
            selector_group="route_matrix",
            layer_role="full_route",
            route_key="d7_gurobi_relaxed",
            trip_days=7,
            method="hierarchical_gurobi_pipeline",
            method_label="Gurobi",
            profile="relaxed",
            default_checked=True,
            default_visible=True,
            distance_km=float("nan"),
            unique_points=2,
            bounds=[[1.0, 2.0], [3.0, 4.0]],
        ),
        _route(
            "child_must_go",
            layers[3].get_name(),
            selector_group="route_matrix",
            layer_role="must_go",
            parent_route_key="d7_gurobi_relaxed",
            control_label="Must-go",
        ),
        _route(
            "child_transition",
            layers[4].get_name(),
            selector_group="route_matrix",
            layer_role="transition",
            parent_route_key="d7_gurobi_relaxed",
            control_label="Transitions",
        ),
        _route(
            "child_city",
            layers[5].get_name(),
            selector_group="route_matrix",
            layer_role="city_loop",
            parent_route_key="d7_gurobi_relaxed",
            control_label="City loops",
        ),
        _route(
            "context_route",
            layers[6].get_name(),
            control_label="Context route",
            bounds=[[9.0, 10.0]],
        ),
    ]

    model = selector.build_route_selector_model(trip_map, registry, hidden_default_layers=[None, layers[0]])
    checkbox_html = model["checkbox_html"]

    assert checkbox_html.index("7-Day Routes") < checkbox_html.index("12-Day Routes")
    assert checkbox_html.index("Relaxed") < checkbox_html.index("Explorer")
    assert checkbox_html.index("City loops") < checkbox_html.index("Transitions") < checkbox_html.index("Must-go")
    assert "Relaxed Route &lt;/script&gt;&lt;b&gt;&amp;" in checkbox_html
    assert "<script>" not in checkbox_html
    assert "2 pts" in checkbox_html
    assert model["checked_count"] == 1
    assert json.loads(model["bounds_json"]) == [[1.0, 2.0], [3.0, 4.0]]
    assert json.loads(model["hidden_json"]) == [layers[0].get_name()]
    assert "<\\/script>" in model["registry_json"]
    assert model["families_label"] == "context, route &lt;family&gt;"

    controls_html = selector.route_selector_controls_html(model)
    controls_script = selector.route_selector_controls_script(model)
    assert "Route Selector" in controls_html
    assert 'data-route-action="clear">Clear</button>' in controls_html
    assert "Hotel Showcase: Off" in controls_html
    assert "data-route-action" in controls_script
    assert model["map_var"] in controls_script


@pytest.mark.parametrize(
    ("registry", "message"),
    [
        (
            [
                _route("duplicate", "layer_a"),
                _route("duplicate", "layer_b"),
            ],
            "duplicate route selector control_id",
        ),
        ([_route("missing", "")], "is missing layer_var"),
        (
            [
                _route(
                    "mismatch",
                    "layer_a",
                    default_checked=True,
                    default_visible=False,
                )
            ],
            "mismatched default_checked/default_visible",
        ),
    ],
)
def test_selector_rejects_registry_entries_that_would_emit_broken_controls(registry, message):
    with pytest.raises(ValueError, match=message):
        selector.build_route_selector_model(folium.Map(location=[0.0, 0.0]), registry)


def test_valid_registry_attaches_both_selector_assets():
    trip_map = folium.Map(location=[0.0, 0.0])
    layer = folium.FeatureGroup(name="Context").add_to(trip_map)
    root = trip_map.get_root()

    selector.add_route_debug_controls(trip_map, [_route("context", layer.get_name())])

    assert any("Route Selector" in child.render() for child in root.html._children.values())
    assert any("data-route-action" in child.render() for child in root.script._children.values())


def test_selector_css_preserves_desktop_width_and_contains_mobile_viewports():
    trip_map = folium.Map(location=[0.0, 0.0])
    layer = folium.FeatureGroup(name="Context").add_to(trip_map)
    model = selector.build_route_selector_model(trip_map, [_route("context", layer.get_name())])

    controls_html = selector.route_selector_controls_html(model)

    assert "left: 74px;" in controls_html
    assert "width: 430px;" in controls_html
    assert "@media (max-width: 519px)" in controls_html
    assert "left: 12px;" in controls_html
    assert "width: calc(100vw - 24px);" in controls_html
