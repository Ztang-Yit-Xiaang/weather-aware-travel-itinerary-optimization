# Recent Papers Quick Index: 2023-2026

Prepared: 2026-06-17  
Updated: 2026-07-02

This file is only a quick index for the newer corpus. It intentionally avoids duplicating the long summaries already integrated into [docs/literature_deep_read_study_report.md](literature_deep_read_study_report.md). Each recent-paper entry in that evidence bank ends with a **Project Action Takeaway** covering the paper's main goal, limitation, publication use, concrete project-polishing action, and current implementation status.

Recommended reading path:

1. Start with [docs/literature_onboarding_guide.md](literature_onboarding_guide.md).
2. Use the [integrated core literature review](core_paper_reading_cards.md) for the detailed eight-paper learning sequence and project synthesis.
3. Use the deep report for full paper-by-paper notes and figure/table interpretation.

## Must-Use Recent Papers

| Cluster | Papers |
| --- | --- |
| LLM travel-planning evaluation | TravelPlanner; TravelEval; TripScore; Revisiting the Travel Planning Capabilities of LLMs; GroupTravelBench |
| LLM + symbolic guarantees | TRIP-PAL; To the Globe (TTG); LLMAP; Logic-LM as general neuro-symbolic support |
| Disruption/weather replanning | TripTide |
| Accountable tourism recommendation | TRACE |
| Explainable travel recommendation | CityHood |
| Recent personalization and agentic baselines | BTRec; +Tour; TravelAgent; Vaiage; Collab-Rec; SynthTRIPs |
| Modern explainability theory | Visualization for Recommendation Explainability; Whom do Explanations Serve; Review of Explainable Graph-Based Recommender Systems |
| Novelty-threat and gap-verification papers | iTIMO; VeriTrip; TP-RAG; ITINERA; TripCraft; TravelBench; COMPASS; CostBench; DRAGged into Conflicts; RAG with Conflicting Evidence; CONFACT; User-Controllable Recommendation |

## Professor-Recommended Companion Papers

| Local file | Paper | Role | Citation priority |
| --- | --- | --- | --- |
| `2509.12273v1.pdf` | LLMAP | Natural-language preference parsing plus multi-objective route search | Essential for future AI/solver framing; supporting for the current route model |
| `2305.12295v2.pdf` | Logic-LM | General language-to-symbolic formulation and solver-error refinement | Supporting neuro-symbolic citation |
| `2510.09011v3.pdf` | TripScore | Fine-grained travel-plan evaluation and expert-calibrated unified reward | Essential modern evaluation citation |
| `2410.16456v1.pdf` | To the Globe (TTG) | Duplicate arXiv copy of the existing EMNLP Demo paper | Cite once using the verified publication entry |
| `1-s2.0-S0143622816306051-main.pdf` | From Stay to Play | UGC/social-media travel decision-support background | Optional background only; not a core repair-method citation |

## Updated Project Angle

The freshest framing is:

> Temporal, user-specific, evidence-conflict-aware, counterfactual minimal-change repair for weather-sensitive multi-day itineraries.

This connects the project to 2024-2026 work on LLM-agent travel-planning failure, solver-backed guarantees, grounded evidence, multidimensional plan evaluation, and disruption-aware replanning, while explicitly avoiding the already-covered claim that "LLM plus optimizer" is novel.

## July 2026 Novelty Audit Additions

| Paper | Why it matters | Citation use |
| --- | --- | --- |
| ITINERA (2024) | LLM plus spatial optimization for open-domain urban itinerary planning | Threatens any broad LLM-plus-optimizer novelty claim |
| TripCraft (2025) | Spatiotemporally fine-grained travel-planning benchmark | Supports need for temporal/spatial evaluation realism |
| TP-RAG (2025) | Retrieval-augmented spatiotemporal travel planning with noisy/conflicting references | Threatens broad retrieval/evidence novelty; supports conflict-aware motivation |
| RAG with Conflicting Evidence / CONFACT (2025) | Generic conflicting-evidence RAG and fact-checking benchmarks | Threatens broad conflict-aware RAG novelty; use only to motivate evidence-conflict handling |
| iTIMO (2026) | Formal itinerary modification task using ADD/DELETE/REPLACE perturbations | Essential threat to operation-level modification novelty |
| VeriTrip (2026) | Verifiable travel-planning benchmark over noisy, contradictory multimodal web evidence | Essential threat to broad evidence-conflict novelty |
| TravelBench (2025/2026 preprint set) | Multi-turn/tool-use travel benchmark with unsolvable cases | Supports interaction and capability-boundary discussion |
| COMPASS (2025) | Multi-turn, tool-mediated constrained preference optimization in travel planning | Threatens broad "constrained interactive travel agent" novelty; supports preference-optimization framing |
| CostBench (2025) | Cost-optimal tool-use planning and adaptation under dynamic blocking events | Threatens broad dynamic adaptation novelty; still not tourism-itinerary repair |
| From Stay to Play (2017) | UGC-based hotel, attraction, and route planning tool | Background for social-media/UGC travel information, not repair |
