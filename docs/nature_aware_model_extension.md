# Nature-Aware Model Extension

## Why Pace And Interest Are Separate

The existing system uses `traveler_profile` for pace: `relaxed`, `balanced`, and `explorer`. The nature-aware extension keeps that axis intact and adds a separate interest bar. A relaxed traveler can still prefer national parks, and an explorer can still prefer city attractions.

## Interest Bar

The UI interest bar is represented as:

```text
p = [p_nature, p_city, p_culture, p_history]
p_k >= 0
sum_k p_k = 1
```

Preset labels such as `nature_heavy` only set the bar values. They are not separate optimizer modes.

Each POI receives an attribute vector:

```text
a_i = [a_i,nature, a_i,city, a_i,culture, a_i,history]
0 <= a_i,k <= 1
```

The preference fit is:

```text
interest_fit_i = p^T a_i
               = p_nature*a_i,nature
               + p_city*a_i,city
               + p_culture*a_i,culture
               + p_history*a_i,history
```

## Interest-Adjusted Utility

The backward-compatible base value remains `final_poi_value_i`. When interest mode is enabled, the model computes:

```text
park_bonus_i =
    1.00 * is_national_park_i
  + 0.60 * is_state_park_i
  + 0.45 * is_protected_area_i
  + 0.25 * is_scenic_viewpoint_i
```

```text
u_i_interest =
    final_poi_value_i
  + lambda_fit * interest_fit_i
  + lambda_park * park_bonus_i
  - lambda_weather * weather_sensitivity_i * weather_risk_i
  - lambda_season * seasonality_risk_i
  - lambda_detour * detour_minutes_i
```

If `interest.enabled` is false:

```text
u_i_interest = final_poi_value_i
interest_delta_i = 0
```

This is the backward compatibility guarantee for the California coast workflow.

Users can now choose presets in two equivalent ways:

```yaml
interest:
  enabled: true
  mode: nature_heavy
```

or:

```yaml
trip:
  interest_profile: nature_heavy
```

`interest.mode` has precedence. `interest.mode: custom` uses the explicit bar weights.

## Phase 2 Region Planning

Phase 2 makes nature regions first-class planning units. The optimizer still allocates days to overnight base cities, but scenarios can now attach nature regions to the route before POI-level selection. This keeps the city/day model stable while allowing a plan to say "Yosemite is selected for two days" or "Death Valley was skipped because the weather exposure and detour are too high."

The object types are intentionally separate:

- `overnight_base_city`: a city or gateway that receives hotel/base days.
- `nature_region`: a park or outdoor region that receives region-level value and may consume suggested nature days.
- `pass_through_scenic_stop`: a scenic stop that can improve route value without becoming a base city.
- `POI`: a visitable stop used by candidate selection, bandit scoring, and route repair.

The new `california_statewide_nature` scenario includes Yosemite, Sequoia, Kings Canyon, Joshua Tree, Death Valley, Lake Tahoe, Point Reyes, Pinnacles, Lassen Volcanic, Redwood National and State Parks, and Big Sur. Yosemite is represented as a region with gateways such as Mariposa, Oakhurst, Fresno, and Yosemite Valley rather than as a single boosted POI.

Region-level scoring uses the same spirit as POI scoring:

```text
region_score_r =
    nature_fit_r
  + national/state/protected/scenic_region_bonus_r
  - detour_penalty_r
  - weather_exposure_r
```

Selected plans are annotated with `selected_nature_regions`, `nature_days_by_region`, `gateway_bases_by_region`, and component terms for nature value, scenic value, detour penalty, and weather exposure.

## Why-Not-Selected Explanations

High-value unselected regions are kept in the plan audit with compact reason codes:

```text
coast_only_scenario_restriction
excessive_inland_detour
insufficient_trip_days
daily_drive_limit_exceeded
weather_risk_too_high
budget_or_time_conflict
lower_interest_fit_than_selected_regions
```

These explanations are designed for plan summaries, scenario comparison tables, and map metadata. For example, Yosemite can be skipped in the original `california_coast` scenario because the route is corridor-focused, while Death Valley can be skipped in a short nature-heavy trip because it exceeds weather or drive-risk thresholds.

## NPS API Versus Curated Fallback

Nature enrichment uses open/free sources only. When:

```yaml
nature:
  use_nps_api: true
```

and the configured environment variable, default `NPS_API_KEY`, exists, the system fetches and caches NPS park data from `https://developer.nps.gov/api/v1/parks`. Parsed fields include park name/code, designation, states, coordinates, activities, topics, entrance fees, operating hours, URL, description, and weather notes.

If live NPS is disabled or unavailable, curated scenario seeds are used with explicit audit statuses:

```text
nps_api_success
nps_disabled_curated_fallback
missing_NPS_API_KEY_curated_fallback
nps_api_error_cache_fallback
nps_api_error_curated_fallback
```

Basic runs and tests do not require a live API key.

## Route Balance

For a selected route `S`, the realized route mix is:

```text
a_bar_S = (1 / |S|) * sum_{i in S} a_i
```

The route-level bar match is:

```text
Balance(S, p) = 1 - ||a_bar_S - p||_1
```

The current implementation reports balance metrics. A future option can add the balance term directly to the small-route objective:

```text
maximize
    sum_i u_i_interest x_i
  + lambda_div * Diversity(x)
  + lambda_balance * Balance(S, p)
  - lambda_travel * sum_ij travel_ij y_ij
```

The existing time, cost, detour, weather, diversity, and MTZ routing constraints remain unchanged.

## Bandit Arm Formula

Nature-aware bandit arms score candidate bundles with:

```text
bundle_score_i(a) =
    base_score_i
  + phi_n(a) * nature_score_i
  + phi_s(a) * scenic_score_i
  + phi_h(a) * hiking_score_i
  + phi_p(a) * park_bonus_i
  + phi_safe(a) * (1 - weather_sensitivity_i * weather_risk_i)
  - phi_d(a) * detour_minutes_i
```

`base_score_i` is `interest_adjusted_value_i` when interest mode is enabled, otherwise `final_poi_value_i`.

## Browser Preview Versus Full Optimization

The map panel recomputes preview scores in JavaScript:

```text
preview_score_i =
    final_poi_value_i
  + lambda_fit * p^T a_i
  + lambda_park * park_bonus_i
  - lambda_weather * weather_sensitivity_i * weather_risk_i
  - lambda_season * seasonality_risk_i
  - lambda_detour * detour_minutes_i
```

This preview is intentionally instant and local to the browser. It updates rankings and route-mix estimates when the bar moves, but it does not rerun Gurobi. Full optimized routes use the selected weights when the Python pipeline is rerun.

## Map Export Architecture

The legacy Folium map remains available for compatibility, but Phase 2 adds two production export modes:

- `results/figures/lightweight_share_map.html`: selected/default route only, essential markers, compact dashboard, and no comparison/debug/candidate layers.
- `results/figures/full_interactive_dashboard/`: modular dashboard with `index.html`, external CSS/JS, route GeoJSON, POI JSON, and lazy-loadable optional layers.

The artifact report `results/outputs/production_map_artifact_size_report.csv` records file sizes, layer counts, route counts, marker counts, and notes. The map focus remains selected-route-first so optional nature layers do not zoom the initial view out too far.

## Opening The Dashboard

The full modular dashboard supports two asset-loading modes. On localhost or GitHub Pages it uses JSON and GeoJSON fetches. When opened directly with `file://`, it uses generated JavaScript fallback assets such as `route_index.js`, `dashboard_metrics.js`, route `.js`, and POI `.js` files.

Serving over localhost is still recommended for development:

```bash
python scripts/serve_dashboard.py
```

Then open:

```text
http://localhost:8000/
```

If port 8000 is already in use, the helper prints the next available port. The lightweight share map is standalone and can be opened directly:

```text
results/figures/lightweight_share_map.html
```

To validate generated artifacts:

```bash
python scripts/validate_dashboard_export.py
```

## Evaluation Metrics

Nature-aware outputs report:

```text
total_utility = sum final_poi_value_i
interest_adjusted_utility = sum interest_adjusted_value_i
interest_delta = interest_adjusted_utility - total_utility
route_mix = mean selected POI attribute vectors
balance_error = ||route_mix - p||_1
balance_score = 1 - balance_error
nature_score = sum nature_score_i
scenic_score = sum scenic_score_i
national_parks_selected = sum is_national_park_i
protected_parks_selected = sum max(is_state_park_i, is_protected_area_i)
outdoor_weather_risk = sum weather_sensitivity_i * weather_risk_i
seasonality_risk = sum seasonality_risk_i
```
