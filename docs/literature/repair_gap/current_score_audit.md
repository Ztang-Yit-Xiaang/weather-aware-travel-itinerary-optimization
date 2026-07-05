# Current Score Audit

Audit date: 2026-07-04

Purpose: document how the current repository scores POIs, cities, hotels, nature exposure, and routes so the repair literature framing does not overclaim what the prototype measures.

## Summary Judgment

The current project score is a transparent heuristic utility proxy. It is useful for generating and stress-testing itinerary alternatives, but it is not a calibrated measure of traveler satisfaction, real-time hotel availability, real congestion, or certified road-valid travel. In the repair paper framing, use the current score as `Utility(plan)` or `ContextualBurden(plan)` only after clearly disclosing its proxy components and validation limits.

## Audit Table

| Component | Where implemented | Formula or behavior | Evidence source | Interpretation | Risk / limitation |
|---|---|---|---|---|---|
| Default utility weights | `src/itinerary_system/utility_model.py:19` | `base_score=.18`, `yelp_signal=.14`, `social_signal=.18`, `must_go_signal=.14`, `corridor_fit=.11`, `wikipedia_signal=.08`, `data_confidence=.10`, `weather_safety=.04`, `low_detour=.03`. | Code constants. | Weighted MCDA-style POI score. | Weights are design choices, not learned/calibrated. |
| Yelp signal | `src/itinerary_system/utility_model.py:81` | If normalized signal absent, uses `yelp_rating * log1p(yelp_review_count)`, then min-max; missing Yelp effectively becomes 0. | Yelp columns if present. | Popularity/quality proxy. | Missing data can be penalized as zero rather than unknown. |
| Data confidence | `src/itinerary_system/utility_model.py:95`; `src/itinerary_system/data_enrichment.py:967`; `notebook/production_enrichment.py:562` | `0.35*OSM + 0.25*Yelp + 0.20*curated + 0.10*wikidata + 0.10*wikipedia`, clipped roughly to `[0.15, 1]`. | Source flags and enrichment coverage. | Source coverage score. | Should not be called calibrated uncertainty. |
| Weather safety | `src/itinerary_system/utility_model.py:104` | `weather_safety = 1 - weather_risk`. | Weather risk field; defaults used when absent. | Context penalty/benefit. | Weather risk is a proxy; must preserve retrieval time and scenario provenance. |
| Bayesian UCB utility | `src/itinerary_system/utility_model.py:171` | Evidence strength is `1 + review_strength + 2*source_count + 5*data_confidence`; UCB adds corridor fit, must-go, and subtracts detour/weather terms. | Enriched feature table. | Alternative uncertainty-aware ranking score. | Posterior assumptions are heuristic; not externally calibrated. |
| Final selected POI value | `src/itinerary_system/utility_model.py:218` and `:245` | `final_poi_value` becomes selected utility method, defaulting to Bayesian UCB when configured. | Utility model output. | Main POI utility consumed by route solvers. | Depends on upstream proxies. |
| Legacy final POI value | `notebook/production_enrichment.py:570` | `0.42*base_score_norm + 0.20*yelp_signal_norm + 1.10*social_score + 0.85*must_go_weight*social_score + 0.30*corridor_fit - 0.010*detour_minutes`, clipped. | Notebook enrichment. | Earlier heuristic scoring model. | Must-go term multiplies social score; not direct user-owned commitment semantics. |
| Current recomputed final POI value | `src/itinerary_system/data_enrichment.py:995` | Similar weighted combination of base, Yelp, social, must-go/social, corridor/route fit, detour, and Wikipedia pageviews. | Enrichment pipeline. | Current enriched POI value before utility model selection. | Utility and provenance mixed in one scalar. |
| Corridor fit and detour | `notebook/production_enrichment.py:238` | Nearest distance to California corridor; `detour_minutes = nearest_km*2/48*60`; `corridor_fit = max(0, 1 - detour_minutes/threshold)`. | Geodesic waypoint proxy. | Low-detour preference. | Not actual road detour. |
| City value score | `notebook/production_enrichment.py:641` | `0.30*popularity + 0.25*social_signal + 0.20*external_poi + 0.15*yelp_signal + 0.10*route_importance`. | Aggregated city/POI features. | Hierarchical city planning score. | City-level abstraction hides parent-plan repair costs. |
| Hotel price proxy | `src/itinerary_system/data_enrichment.py:283`; `notebook/production_enrichment.py:676` | Combines city prior and lodging-type prior; fallback city priors if no hotel data. | Static priors, OSM/curated when present. | Budget approximation. | Not live hotel availability or booking price. |
| Curated/OSM hotel value | `src/itinerary_system/data_enrichment.py:320` and `:498` | Rating priors and price proxy produce `experience_score` and `value_score = rating_score / nightly_price * 100`; `price_estimated=True`. | Curated fallback or OSM entities. | Hotel comparison proxy. | Do not claim real-time lodging optimization. |
| Nature attributes | `src/itinerary_system/nature_catalog.py:317` | `outdoor_intensity` combines nature, hiking, and scenic signals; `weather_sensitivity = 0.25 + 0.65*outdoor_intensity`. | OSM/category tags and derived flags. | Weather-to-nature exposure model. | Derived tags can be sparse/noisy. |
| Interest-adjusted utility | `src/itinerary_system/nature_catalog.py:339`; `:367` | `u_i_interest = final_poi_value_i + lambda_fit*interest_fit + lambda_park*park_bonus - lambda_weather*weather_sensitivity*weather_risk - lambda_season*seasonality_risk - lambda_detour*detour_minutes`. | POI utility, interest profile, nature flags, weather risk. | Good candidate for `Utility(plan)` under nature/weather preferences. | Still a heuristic feature model. |
| Route interest metrics | `src/itinerary_system/nature_catalog.py:588` | Reports interest-adjusted utility, balance, nature/scenic/hiking sums, weather exposure, seasonality risk. | Selected route frame. | Useful repair evaluation dimensions. | Needs parent-child metrics for repair. |
| Internal nature route score | `src/itinerary_system/nature_site_routes.py:303` | `route_score = 0.45*confidence + 0.30*distance_fit + 0.25*type_fit`. | Nature route extraction/cache status. | Measures confidence/fit of internal routes. | Not an external road certificate by itself. |
| Nature route bonus | `src/itinerary_system/nature_site_routes.py:662` | Adds `weight*nature_weight*best_internal_route_score - long_penalty` to `final_poi_value`. | Internal nature route score. | Rewards good on-site nature route evidence. | Bonus is small and heuristic. |
| Route travel time | `src/itinerary_system/multi_objective_route.py:51` | `geodesic_km * 1.25 / 38 * 60`. | Coordinates. | Fallback travel time proxy. | Not road-valid and not final-comparison eligible unless independently certified. |
| Route objective | `src/itinerary_system/multi_objective_route.py:346` | Maximize selected POI values plus diversity bonus minus travel penalty, subject to time/cost/detour/weather/diversity constraints. | Enriched POIs and config. | Multi-objective route generation. | Not parent-aware repair; no typed edit variables. |
| Greedy epsilon repair name | `src/itinerary_system/multi_objective_route.py:135` | Greedy heuristic labeled repair, but selects POIs under constraints from scratch. | POI pool. | Feasible fallback selection. | "Repair" is not parent-child repair. Rename or qualify in paper. |
| Hierarchical city objective | `src/itinerary_system/hierarchical_gurobi.py:522` | City value, must-go city value, pass-through value, uncertainty bonus, drive penalty, nature/scenic bonuses, weather exposure penalties. | City summary and config. | City-level route skeleton. | Uses geodesic drive proxy and no parent ownership semantics. |
| Route oracle | `src/itinerary_system/route_gurobi_oracle.py:115` | `solve_enriched_route_with_gurobi` immediately returns `solve_multi_objective_route`; legacy code after return is unreachable. | Code inspection. | Current oracle is a wrapper/fallback. | Do not claim an independent Gurobi road oracle from this path. |
| Snapshot provenance | `data/snapshots/california_v1/feature_provenance.csv` | Current seed utilities are curated planning seeds; rows mark stable coordinates and curated candidate roles. | Local snapshot. | Good reproducibility basis. | Curated seed utility is not observed demand. |
| Source audit | `data/snapshots/california_v1/source_audit.csv` | `routing_demo_context` notes straight/geodesic rows are approximate and not final-comparison eligible. | Local snapshot. | Explicit provenance caveat. | Must be carried into evaluation claims. |

## Recommended Reframing

Use:

- `utility_proxy`, `contextual utility`, or `research utility score`.
- `source coverage` instead of calibrated `data confidence`.
- `geodesic fallback` or `demo routing context` unless road API validation is used.
- `hotel price prior` or `estimated lodging burden` instead of hotel availability.
- `weather-adjusted nature exposure` instead of real safety prediction.

Avoid:

- "Accurate real-world congestion."
- "Real-time booking."
- "Certified route-valid" for plans that use `multi_objective_route.travel_minutes` geodesic estimates only.
- "Learned preference model" unless trained on real user preference feedback.
- "Online bandit personalization" unless sequential user feedback is logged and evaluated.

## Score Model for the Repair Paper

For the repair paper, separate the score into three layers:

1. Parent-child preservation layer:
   - `LockedPreservation`
   - `BookedPreservation`
   - `UnaffectedDayPreservation`
   - `WeightedEditCost`

2. Disruption mitigation layer:
   - `WeatherRiskReduction`
   - `NatureExposureReduction`
   - `RouteClosureAvoidance`
   - `CertificationStatus`

3. Utility layer:
   - `UtilityRetained = Utility(child) / Utility(parent)`
   - `UtilityRegret = Utility(full_reoptimization) - Utility(repair)`
   - `InterestAdjustedUtility`
   - `EstimatedBudgetDelta`

This keeps the current utility proxy valuable while preventing it from swallowing the actual novelty: ownership-aware minimal repair.

## Minimal Implementation Implications

To make the literature claim executable, the next implementation phase should add:

- A `ParentPlan` / `ChildPlan` artifact with immutable IDs and lineage.
- Ownership labels on stops, hotels, routes, time windows, and user constraints.
- Typed edit variables and typed change costs.
- Progressive neighborhoods: same stop/time shift, same-day replacement, adjacent-day move, hotel-preserving reroute, hotel-changing repair, full reoptimization.
- Sequential objective stages: preservation, feasibility/certification, disruption mitigation, utility.
- Independent certificate artifact recording road route source, weather snapshot, hotel-price/availability status, and fallback flags.
- Explanation generator constrained to evidence IDs from the diff and certificate.
