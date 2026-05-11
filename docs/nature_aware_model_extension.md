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

