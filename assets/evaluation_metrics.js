window.DASHBOARD_EVALUATION_METRICS = {
  "available": true,
  "source_files": [
    "production_method_comparison.csv",
    "production_method_route_stops.csv"
  ],
  "methods": [
    {
      "method": "hierarchical_gurobi_pipeline",
      "label": "Hierarchical Gurobi Pipeline",
      "short_label": "Hierarchical Gurobi",
      "status": "OPTIMAL",
      "comparison_score": 0.3484673643081107,
      "objective": 18.326643677845503,
      "route_distance_km": 1601.8529320395082,
      "route_time_minutes": 2545.922457970165,
      "nature_score": 1.95,
      "scenic_score": 1.9849999999999999,
      "weather_risk": 0.3778526785714286,
      "selected_nature_stops": 4,
      "selected_stop_count": 7,
      "runtime_seconds": 0.0570574000594206,
      "notes": "Canonical route matrix row for 7-Day \u00b7 Gurobi \u00b7 Balanced; gurobi city/base/day allocation and legacy local gurobi route; profile controls stops/day and hotel scoring; strategy=balanced."
    },
    {
      "method": "hierarchical_greedy_baseline",
      "label": "Hierarchical Greedy Baseline",
      "short_label": "Hierarchical Greedy",
      "status": "FAILED",
      "comparison_score": 0.4086082363034558,
      "objective": 17.73708274124567,
      "route_distance_km": 1534.9182366119967,
      "route_time_minutes": 2413.8145064684973,
      "nature_score": 1.95,
      "scenic_score": 1.9849999999999999,
      "weather_risk": 0.3778526785714286,
      "selected_nature_stops": 4,
      "selected_stop_count": 7,
      "runtime_seconds": 0.0,
      "notes": "Canonical route matrix row for 7-Day \u00b7 Greedy \u00b7 Balanced; greedy city/base/day allocation and greedy local route; profile controls stops/day and hotel scoring; strategy=balanced."
    },
    {
      "method": "hierarchical_bandit_gurobi_repair",
      "label": "Hierarchical + Bandit + Small Gurobi Repair",
      "short_label": "Bandit + Repair",
      "status": "FAILED",
      "comparison_score": 0.5468195092755338,
      "objective": 18.326643677845503,
      "route_distance_km": 1594.8005104525428,
      "route_time_minutes": 2532.003204837996,
      "nature_score": 1.95,
      "scenic_score": 1.9849999999999999,
      "weather_risk": 0.3778526785714286,
      "selected_nature_stops": 4,
      "selected_stop_count": 7,
      "runtime_seconds": 1.3262959001003765,
      "notes": "Canonical route matrix row for 7-Day \u00b7 Bandit + Small Gurobi \u00b7 Balanced; bandit strategy selection with small gurobi repair; profile controls stops/day and hotel scoring; strategy=national_park_priority."
    }
  ],
  "chart_fields": [
    {
      "key": "comparison_score",
      "label": "Comparison score",
      "higher_is_better": true
    },
    {
      "key": "route_distance_km",
      "label": "Route distance (km)",
      "higher_is_better": false
    },
    {
      "key": "route_time_minutes",
      "label": "Route time (min)",
      "higher_is_better": false
    },
    {
      "key": "nature_score",
      "label": "Nature score",
      "higher_is_better": true
    },
    {
      "key": "weather_risk",
      "label": "Weather risk",
      "higher_is_better": false
    },
    {
      "key": "selected_nature_stops",
      "label": "Selected nature stops",
      "higher_is_better": true
    },
    {
      "key": "runtime_seconds",
      "label": "Runtime (sec)",
      "higher_is_better": false
    }
  ],
  "tradeoff_explanation": [
    "Hierarchical Gurobi optimizes the saved city/base allocation and local route with the strongest exact-solver signal.",
    "Hierarchical Greedy is fastest and useful as a transparent baseline, but can miss globally valuable nature anchors.",
    "Bandit + Repair explores route strategies, then repairs promising candidates with a small Gurobi pass; this is the default saved dashboard route."
  ]
};
