# california_v1

`california_v1` is the first clean-clone fallback catalog snapshot for the
California itinerary-repair research path.

It deliberately separates stable catalog tables from time-sensitive context
tables while keeping them in one compact folder for early Phase 0 work:

- `poi_entities.csv`: stable POI identity and coordinates.
- `poi_observations.csv`: source-level facts and descriptions.
- `poi_features.csv`: derived planning features for this snapshot.
- `feature_provenance.csv`: feature-level provenance and transformation labels.
- `hotel_entities.csv`: stable hotel planning entities and priors.
- `weather_scenarios.csv`: timestamped scenario weather context.
- `route_options.csv`: routing context with explicit road validation flags.
- `source_audit.csv`: source roles, licenses, and redistribution notes.

This snapshot excludes private Yelp raw data. It is a research seed and should
not be described as a comprehensive, live, or booking-grade travel dataset.

