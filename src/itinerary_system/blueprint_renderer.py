"""Package-owned production Folium map renderer."""

from __future__ import annotations

from .blueprint_renderer_sections import (
    _add_comparison_layer_section,
    _add_context_route_section,
    _add_dashboard_section,
    _add_hotel_layer_section,
    _add_must_go_layer_section,
    _add_nature_layer_section,
    _add_profile_route_section,
    _finalize_renderer,
    _prepare_renderer_state,
)


def build_production_trip_map(context, output_path=None, run_live_routing=None):
    state = _prepare_renderer_state(context, output_path, run_live_routing)
    _add_context_route_section(state)
    _add_hotel_layer_section(state)
    _add_must_go_layer_section(state)
    _add_nature_layer_section(state)
    _add_profile_route_section(state)
    _add_comparison_layer_section(state)
    _add_dashboard_section(state)
    return _finalize_renderer(state)
