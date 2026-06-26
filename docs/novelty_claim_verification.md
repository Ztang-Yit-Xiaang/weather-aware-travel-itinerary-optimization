# Novelty Claim Verification: Limitation-Driven Itinerary Repair

**Date:** 2026-06-26  
**Scope:** Local paper summaries plus targeted web search for current travel-planning, itinerary-modification, conflict-aware RAG, counterfactual recommendation, and dynamic replanning literature.

## Bottom Line

The exact idea does not appear in the checked literature under the searched phrases and closest-paper comparison:

> temporal, user-specific, evidence-conflict-aware, counterfactual, minimal-change repair for multi-day travel itineraries, with confirmed LLM interpretation, solver/evaluator-backed repair operations, and whole-trip feasibility metrics.

This should be stated as a scoped novelty claim, not as a proof that no related system exists. The safer wording is:

> To the best of our search, prior work has studied language-to-symbolic travel planning, itinerary modification datasets, disruption-revision benchmarks, whole-trip evaluation, evidence-conflict handling, and counterfactual recommendation separately, but not their combination as a solver-backed minimal-change repair method for weather-sensitive multi-day itineraries.

## Search Result

Targeted phrase searches returned no direct match for:

- "minimal-change itinerary repair"
- "counterfactual itinerary repair"
- "solver-backed itinerary repair"
- "conflict-aware counterfactual itinerary"
- "temporal user-specific conflict-aware counterfactual itinerary"
- "evidence conflict travel planning optimization"

The absence of exact matches is only negative evidence. The claim must still be defended through closest-work differentiation.

## Closest Literature and Novelty Boundary

| Work | What it covers | Why it does not subsume this idea |
|---|---|---|
| [TTG, 2024](https://arxiv.org/abs/2410.16456) | Natural-language requests translated to symbolic form and solved with MILP for guaranteed travel itineraries | It supports the LLM-to-symbolic precedent; the guarantee depends on correct translation and the paper is not framed as disruption repair with evidence conflicts or counterfactual user control |
| TRIP-PAL, 2024 | LLM extracts travel information and an automated planner produces constraint-satisfying plans | Strong hybrid precedent, but not a temporal multi-day repair method with preservation, evidence conflicts, and counterfactual repair explanations |
| [ITINERA, 2024](https://arxiv.org/abs/2402.07204) | LLM plus spatial optimization for open-domain urban itinerary planning | Very relevant to LLM plus optimizer; it reinforces that this combination is not novel. It is initial urban itinerary generation, not multi-day disruption repair |
| [LLMAP, 2025](https://arxiv.org/abs/2509.12273) | LLM-as-parser plus graph search for route planning with user preferences | Close parser-search architecture, but the evaluated constraints are route/task constraints, not multi-day lodging/weather disruption repair with original-intent preservation |
| [iTIMO, 2026](https://arxiv.org/abs/2601.10609) | Defines itinerary modification and creates ADD/DELETE/REPLACE perturbation data | Closest to modification operations, but it is a synthesis dataset and benchmark, not a solver-backed temporal repair planner; it omits MOVE/RELAX/KEEP, lodging consistency, evidence conflicts, and counterfactual frontier |
| [TripTide, 2025](https://arxiv.org/abs/2510.21329) | Benchmark for LLM itinerary revision under disruptions with preservation/responsiveness/adaptability metrics | Closest to disruption repair, but it evaluates LLM revisions rather than proposing a mathematical minimal-change repair optimizer or conflict-aware evidence handling |
| [TravelEval, 2026](https://arxiv.org/abs/2606.01046) | Six-dimensional whole-trip evaluation benchmark across accuracy, compliance, temporality, spatiality, economy, and utility | Directly useful as evaluator vocabulary, but not a repair method |
| [TripCraft, 2025](https://arxiv.org/abs/2502.20508), [TP-RAG, 2025](https://arxiv.org/abs/2504.08694), [TripScore, 2025](https://arxiv.org/abs/2510.09011) | More realistic travel-planning datasets, spatiotemporal metrics, retrieval-augmented planning, and fine-grained rewards | They improve generation/evaluation realism, but do not define the proposed evidence-conflict-aware repair operation model |
| [TravelBench, 2025](https://arxiv.org/abs/2512.22673) | Multi-turn and tool-using travel benchmark with single-turn, multi-turn, and unsolvable tasks | Important for interaction and capability-boundary evaluation, but not a repair optimizer |
| [GroupTravelBench, 2026](https://arxiv.org/abs/2605.25200) | Multi-user travel planning with preference conflicts and fairness | "Conflict" here means user preference conflict, not contradictory weather/closure evidence tied to route-repair decisions |
| [VeriTrip, 2026](https://arxiv.org/abs/2605.28683) | Verifiable benchmark over unstructured multimodal web corpora; explicitly emphasizes information noise and multi-source contradictions | Strong threat to the evidence-conflict angle, but it is a benchmark for evidence-grounded reasoning, not a solver-backed itinerary repair method |
| [DRAGged into Conflicts, 2025](https://arxiv.org/abs/2506.08500), and newer conflict-aware RAG papers | Taxonomies/benchmarks/frameworks for conflicting retrieved sources | Relevant method inspiration, but not travel-route optimization or repair |
| [User-controllable counterfactual recommendation, 2023](https://arxiv.org/abs/2308.00894) | Counterfactual retrospective/prospective explanations for recommendation control | Supports user-control framing, but not constrained itinerary repair |
| [Dynamic public-transport replanning, 2025](https://arxiv.org/abs/2505.14193) | Formal dynamic replanning under delays, including pull and push approaches | Real replanning precedent, but for public transport routing rather than personalized tourism itineraries with POIs, lodging, evidence conflicts, and counterfactual explanations |

## Claim Strength

**Strong and defensible after the audit:**

- The project should not claim novelty for "LLM plus optimizer."
- The strongest novelty is the integration of repair-specific artifacts: confirmed parser boundary, minimal-change repair operations, evidence-conflict labels, user tolerance profile, whole-trip feasibility evaluation, and counterfactual explanations.
- The TRB claim is credible if evaluated against original route, full replanning, context-blind solver, and heuristic repair on disruption scenarios.

**Needs empirical support before publication:**

- "Reduces disruption/weather risk while preserving more original itinerary intent" needs offline experiments.
- "Improves user understanding/control/error detection" needs a user or expert study.
- "Solver-backed" must be phrased as modeled feasibility under frozen data, not real-world guarantee.

**Potential novelty threats to cite explicitly:**

- iTIMO owns the "itinerary modification" framing and ADD/DELETE/REPLACE vocabulary.
- TripTide owns disruption-revision benchmarking and preservation/adaptability metrics.
- TravelEval and TripScore own whole-plan evaluation vocabulary.
- VeriTrip and TP-RAG weaken any broad claim that evidence conflict handling is new in travel-agent evaluation.
- ITINERA, TTG, TRIP-PAL, and LLMAP make LLM-plus-symbolic/optimizer architecture established.

## Revised Safe Contribution Statement

Use this version:

> We introduce a limitation-driven itinerary repair formulation that converts confirmed user intent and frozen disruption evidence into auditable ADD/DELETE/REPLACE/MOVE/RELAX/KEEP operations, explicitly labels conflicting evidence, optimizes for minimal change under temporal and weather-sensitive constraints, and reports whole-trip feasibility plus counterfactual explanations. Unlike prior LLM-plus-planner systems, the contribution is not initial generation; unlike modification or disruption benchmarks, the contribution is an executable repair mechanism; unlike conflict-aware RAG, conflicts are tied to route constraints and repair alternatives.

Avoid this version:

> We are the first LLM-plus-optimizer travel planner.

## Next Verification Steps

1. Add VeriTrip, TP-RAG, TripCraft, TravelBench, GroupTravelBench, TripScore, and ITINERA to the related-work section.
2. Run exact ablation experiments so the claim becomes evidence-backed rather than only literature-backed.
3. Keep "From Stay to Play" out of the core citation set until the PDF/citation is found; targeted searches did not locate it.
