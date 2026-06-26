# Limitation-Driven Itinerary Repair Method

## Core Publication Thesis

The publication should move from "LLM plus optimizer for travel planning" to:

> Temporal, user-specific, conflict-aware, counterfactual itinerary repair for multi-day trips.

The implemented scaffold in `src/itinerary_system/repair_planner.py` treats the LLM as an interpretation layer only. The repair decision is made by typed records, deterministic constraints, explicit evidence conflicts, whole-trip evaluation, and counterfactual explanations.

## Limitation-Derived Gap

The gap should be stated as a direct response to prior work:

| Paper | Limitation to cite | Gap used by this method |
|---|---|---|
| [TTG](https://arxiv.org/abs/2410.16456) | Limited schema, synthetic setting, guarantees depend on correct translation | We separate confirmed interpretation from solver control and expose lineage |
| [TRIP-PAL](https://arxiv.org/abs/2406.10196) | Day-planning scope, discretized time, limited user-facing adaptation | We target multi-day repair with lodging and preservation metrics |
| [LLMAP](https://arxiv.org/abs/2509.12273) | Route-specific parser/search, not multi-day lodging/weather repair | We repair weather-sensitive routes rather than only generate/search routes |
| [iTIMO](https://arxiv.org/abs/2601.10609) | ADD/DELETE/REPLACE task is user-agnostic and lacks temporal constraints | We extend operations with MOVE, RELAX, KEEP and user tolerance profiles |
| [TripTide](https://arxiv.org/abs/2510.21329) | Disruption revisions are benchmarked as LLM revisions | We make disruption repair solver/evaluator backed |
| [TravelEval](https://arxiv.org/abs/2606.01046) | Whole-trip evaluation, not a repair method | We use whole-trip evaluation as a gate and comparison artifact |
| [Plan-Grounded LLMs](https://arxiv.org/abs/2402.01053) | General plan-grounded dialogue, limited long-context consistency | We constrain dialogue outputs to confirmed repair intents |
| [DRAGged into Conflicts](https://arxiv.org/abs/2506.08500) | Conflict-aware RAG is not tied to route decisions | We label evidence conflicts that affect itinerary constraints |
| [User-Controllable Recommendation](https://arxiv.org/abs/2308.00894) | Counterfactual control is not constrained itinerary repair | We answer what must change to keep, replace, or relax a stop |
| [ChinaTravel](https://arxiv.org/abs/2412.13682) | Benchmark-oriented and geographically scoped | We provide a repair method and experiment design |
| [From Stay to Play](https://www.sciencedirect.com/science/article/pii/S0143622816306051?via%3Dihub) | Unverified lead | Keep out of core claims until exact citation/PDF is available |

## Implemented Research Records

The current code defines these contracts:

- `RepairRequest`: baseline route, user intent, evidence records, tolerance profile, candidates, travel graph, and optional parsed intent.
- `ParsedRepairIntent`: confirmed typed parser output with confidence, warnings, must-keep/delete/include fields, and allowed operations.
- `RepairOperation`: ADD, DELETE, REPLACE, MOVE, RELAX, and KEEP operation record.
- `RepairPlan`: repaired route linked to parent route hash, operations, conflicts, evaluation, and explanations.
- `EvidenceConflict`: stale, contradictory, low-confidence, or missing source conflict.
- `EvaluationReport`: hard feasibility first, then preservation, edit distance, utility, weather, travel, and lodging metrics.
- `CounterfactualExplanation`: binding constraint, minimal relaxation or candidate substitution, and affected metrics.

## Method Flow

1. Freeze the baseline route and evidence in a `RepairRequest`.
2. Parse user text into `ParsedRepairIntent`, but only compile it when `confirmed=True`.
3. Detect evidence conflicts before repair and surface conditional alternatives when needed.
4. Generate minimal-change repair operations over the baseline route.
5. Evaluate the repaired route with hard feasibility gates before soft metrics.
6. Return counterfactual explanations such as the threshold needed to keep a risky stop.
7. Generate a small frontier of risk-averse, balanced, and preservation-first plans for offline comparison.

## Experimental Plan

Offline TRB experiments should compare original route, full replanning, context-blind solver, heuristic repair, and this repair planner across:

- severe rain;
- outdoor closure;
- travel-time delay;
- must-go conflict;
- hotel or base-city disruption;
- conflicting weather or closure evidence.

Core metrics:

- hard feasibility;
- weather or disruption risk reduction;
- preserved stops and route edit distance;
- utility retained;
- added travel time;
- lodging consistency;
- runtime and solver/fallback status.

Ablations:

- remove minimal-change penalty;
- remove temporal weather window;
- remove user tolerance;
- remove conflict handling;
- remove counterfactual frontier.

## Claims to Make

Primary TRB claim:

> The proposed repair planner reduces weather/disruption risk while preserving more original itinerary intent than full replanning, heuristic replacement, or context-blind optimization.

Secondary IUI/CHI fallback claim:

> Counterfactual repair explanations improve users' understanding, control, and error detection compared with static route explanations.

Avoid claiming that LLM plus optimizer is novel, that the system guarantees real-world itineraries, that it is an autonomous travel agent, or that weather-aware recommendation alone is the contribution.
