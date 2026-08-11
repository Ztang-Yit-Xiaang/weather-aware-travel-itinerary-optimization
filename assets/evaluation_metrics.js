window.DASHBOARD_EVALUATION_METRICS = {
  "available": false,
  "data_status": "not_available",
  "source_files": [
    "production_method_comparison.csv",
    "production_method_route_stops.csv"
  ],
  "methods": [],
  "chart_fields": [
    {
      "key": "weighted_edit_cost",
      "label": "Weighted edit cost",
      "higher_is_better": false
    },
    {
      "key": "utility_retained",
      "label": "Utility retained",
      "higher_is_better": true
    },
    {
      "key": "weather_risk_delta",
      "label": "Weather-risk reduction",
      "higher_is_better": true
    },
    {
      "key": "runtime_seconds",
      "label": "Runtime (sec)",
      "higher_is_better": false
    }
  ],
  "tradeoff_explanation": [
    "Context-blind optimization intentionally omits contextual constraints; independent evaluation may reject its output.",
    "The deterministic context-aware heuristic is reproducible but does not claim exact optimality.",
    "Progressive repair expands its neighborhood only as needed and preserves lexicographic priorities.",
    "Full reoptimization is exact only for the frozen candidate universe and reports a refusal status when its safety cap is exceeded."
  ],
  "empty_message": "No canonical E3 method evidence is available in this export."
};
