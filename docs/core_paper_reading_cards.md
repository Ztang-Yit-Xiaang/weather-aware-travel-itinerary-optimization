# Core Paper Reading Cards

These eight papers are the minimum reading spine for this project. They are not the only relevant papers, but together they explain the main intellectual structure: itinerary recommendation, Tourist Trip Design, rich constraints, LLM planning limitations, solver-backed planning, disruption adaptation, visual explanation, and explainable travel recommendation.

Use `docs/literature_deep_read_study_report.md` for the full evidence bank after reading these cards.

## Halder, Lim, Chan, and Zhang (2024) - A Survey on Personalized Itinerary Recommendation: From Optimisation to Deep Learning

**Research area:**  
Personalized Travel Recommendation.

**Local file:**  
`reference/1-s2.0-S1568494623012188-main.pdf`

**Research problem:**  
Personalized itinerary recommendation has been studied with many formulations, techniques, constraints, and data sources, making it hard for new researchers to understand the field.

**Why prior work was insufficient:**  
Older surveys did not fully organize the movement from optimization-based itinerary methods to deep-learning and personalization approaches. A new reader needs a taxonomy that connects POI recommendation, itinerary construction, user satisfaction, provider satisfaction, constraints, and learned models.

**Core method:**  
The paper is a survey and taxonomy. It reviews personalized itinerary recommendation work and organizes it by problem formulation, techniques, constraints, features, and satisfaction goals.

**Main result:**  
The paper shows that itinerary recommendation is broader than POI ranking: a system must select POIs and schedule them into a route while considering user preferences and many constraints.

**Important limitation:**  
It is primarily a literature survey. It does not provide a user-facing dashboard, weather-aware route adaptation, or evidence that users understand route tradeoffs better.

**Connection to our project:**  
This paper supports the personalization layer: interest profiles, POI utility, itinerary recommendation, and the distinction between user preference and route feasibility.

**Feature, metric, or concept we may borrow:**  
Use its taxonomy to describe the project as personalized itinerary recommendation plus constrained planning, rather than as a simple POI recommender.

**One sentence this paper can support in our paper:**  
Personalized itinerary recommendation requires both preference-aware POI selection and itinerary construction under constraints.

**Sections to read:**  
Read the abstract, introduction, taxonomy/problem formulation sections, constraints/features discussion, and future directions. Skim detailed algorithm tables on the first pass.

**Questions I should be able to answer after reading:**  
What is the difference between POI recommendation and itinerary recommendation? What kinds of user preferences and constraints appear in prior itinerary systems? Why is route feasibility still necessary even when POI utility is personalized?

## Ruiz-Meza and Montoya-Torres (2022) - A Systematic Literature Review for the Tourist Trip Design Problem: Extensions, Solution Techniques and Future Research Lines

**Research area:**  
Operations Research and Tourist Trip Design.

**Local file:**  
`reference/1-s2.0-S2214716022000069-main.pdf`

**Research problem:**  
The Tourist Trip Design Problem (TTDP) literature contains many variants, objectives, and solution methods, and needs a systematic review to organize extensions and future research directions.

**Why prior work was insufficient:**  
Classic orienteering work explains the mathematical root, but tourism applications add specific constraints such as multiple days, time windows, transport, preferences, and realistic travel behavior.

**Core method:**  
The paper conducts a systematic literature review and proposes a taxonomy of TTDP variants, objectives, and solution techniques.

**Main result:**  
The paper positions TTDP as a tourism-specific extension of the Orienteering Problem and shows that most practical trip design systems require richer constraints than the classic model.

**Important limitation:**  
The review is mostly algorithmic and operations-research oriented. It does not focus on user understanding, explainability, or mixed-initiative dashboard design.

**Connection to our project:**  
This is the main foundation for explaining why the project is a constrained route-planning system, not a ranked attraction list.

**Feature, metric, or concept we may borrow:**  
Use TTDP language for multi-day route design, POI selection, sequencing, constraints, and solution-technique positioning.

**One sentence this paper can support in our paper:**  
Tourist trip planning is commonly modeled as a TTDP, where a traveler selects and sequences attractions under practical constraints.

**Sections to read:**  
Read the abstract, introduction, TTDP definition, taxonomy, extensions, and future research lines. Skim exhaustive classification tables until the terminology is familiar.

**Questions I should be able to answer after reading:**  
How is TTDP related to the Orienteering Problem? What makes tourist trip design different from generic routing? Which constraints are common enough that our project should not claim them as new?

## Vu, Kergosien, Mendoza, and Desport (2022) - Branch-and-Check Approaches for the Tourist Trip Design Problem with Rich Constraints

**Research area:**  
Operations Research and Tourist Trip Design.

**Local file:**  
`reference/1-s2.0-S0305054821002963-main.pdf`

**Research problem:**  
Realistic TTDP instances include not only budgets, opening hours, and trip duration, but also mandatory visits, limits on types of locations, and ordering constraints.

**Why prior work was insufficient:**  
Simpler TTDP formulations do not fully capture rich practical tourism constraints. Directly solving all constraints in one model can also become difficult.

**Core method:**  
The paper proposes branch-and-check approaches. A master problem selects candidate locations while satisfying non-time constraints, and a subproblem checks whether the selected locations can be scheduled feasibly.

**Main result:**  
The method shows how separating selection and feasibility checking can handle richer tourism constraints more effectively than a simpler monolithic approach.

**Important limitation:**  
The paper is not about interface design or user explanations. It also does not address weather-aware replanning or trust calibration.

**Connection to our project:**  
It supports the claim that rich route constraints are real and technically hard. It also helps frame why solver/fallback status and feasibility labels matter to users.

**Feature, metric, or concept we may borrow:**  
Borrow the selection-versus-scheduling distinction and the idea that a route can be attractive but infeasible.

**One sentence this paper can support in our paper:**  
Rich TTDP variants require explicit feasibility handling because selected attractions may not be schedulable under practical tourism constraints.

**Sections to read:**  
Read the abstract, problem definition, rich-constraint description, branch-and-check method, and computational conclusion. Skim detailed mathematical proofs unless needed for model implementation.

**Questions I should be able to answer after reading:**  
What is branch-and-check? Why can selecting good POIs still fail when scheduling is checked? How should our project describe solver-backed feasibility without overclaiming global real-world optimality?

## Chen et al. (2026) - TravelEval: A Comprehensive Benchmarking Framework for Evaluating LLM-Powered Travel Planning Agents

**Research area:**  
LLM Travel-Planning Evaluation.

**Local file:**  
`reference/2606.01046v1.pdf`

**Research problem:**  
Existing LLM travel-planning benchmarks overemphasize constraint compliance and do not fully evaluate spatial, temporal, lodging, transportation, economy, and complete-plan quality.

**Why prior work was insufficient:**  
Many evaluations judge whether a plan sounds plausible or satisfies isolated constraints. Travel planning needs multi-dimensional assessment of the whole itinerary.

**Core method:**  
The paper introduces TravelEval, a benchmark with a six-dimensional evaluation framework covering accuracy, compliance, temporality, spatiality, economy, and utility, plus a realistic data sandbox and simulation-based global evaluation.

**Main result:**  
TravelEval provides a richer vocabulary for evaluating complete travel plans and shows that LLM-powered agents should be assessed beyond surface fluency.

**Important limitation:**  
The local PDF appears as a recent arXiv-style paper; no peer-reviewed venue is verified from the local first pages. It evaluates LLM agents, not this project's dashboard directly.

**Connection to our project:**  
It helps modernize the evaluation plan. Our project can borrow dimensions such as compliance, temporality, spatiality, economy, utility, and complete-plan assessment.

**Feature, metric, or concept we may borrow:**  
Use TravelEval-style dimensions to organize objective route metrics and explain why evaluation should include lodging, pacing, geography, budget, and utility.

**One sentence this paper can support in our paper:**  
Recent LLM travel-planning evaluation argues that complete itineraries should be assessed across constraint, spatial, temporal, economic, and utility dimensions.

**Sections to read:**  
Read the abstract, benchmark motivation, six-dimensional framework, data sandbox, evaluation method, and main results. Skim model-by-model tables at first.

**Questions I should be able to answer after reading:**  
Why is travel-plan evaluation multi-dimensional? Which dimensions match our dashboard metrics? Which evaluation claims would require new experiments in our project?

## de la Rosa, Gopalakrishnan, Pozanco, Zeng, and Borrajo (2024) - TRIP-PAL: Travel Planning with Guarantees by Combining Large Language Models and Automated Planners

**Research area:**  
LLM and Solver-Backed Travel Planning.

**Local file:**  
`reference/2406.10196v1.pdf`

**Research problem:**  
LLM-only travel planners can generate fluent itineraries that fail to satisfy constraints or produce coherent high-quality plans.

**Why prior work was insufficient:**  
Traditional planning systems require formal problem definitions, while LLM-only systems can miss formal validity. The paper tries to combine natural-language flexibility with automated planning guarantees.

**Core method:**  
TRIP-PAL uses LLMs to extract and translate travel information and user information into data structures for an automated planner, then uses the planner to generate valid plans under the formal model.

**Main result:**  
The paper reports that TRIP-PAL produces more valid and higher-utility travel plans than an LLM-only baseline in the tested scenarios.

**Important limitation:**  
The guarantee is only as strong as the formal planning model, extracted data, and planner assumptions. The local PDF appears as an arXiv-style paper; no venue is verified from the local first pages.

**Connection to our project:**  
It supports the hybrid argument: natural-language or LLM interfaces should compile into auditable constraints and solver-backed planning, not replace validation.

**Feature, metric, or concept we may borrow:**  
Borrow the LLM-to-symbolic framing for future work, especially if natural language preferences are added.

**One sentence this paper can support in our paper:**  
Hybrid travel-planning systems can use LLMs for information extraction while relying on symbolic planners for formal constraint satisfaction within a defined model.

**Sections to read:**  
Read the abstract, introduction, system architecture, planning formulation, experiments, and invalidity analysis. Skim low-level prompt examples unless implementing an LLM layer.

**Questions I should be able to answer after reading:**  
Why are LLM-only travel plans unreliable under constraints? What does a symbolic planner add? What would our project need before claiming LLM-to-symbolic travel planning?

## Karmakar et al. (2025) - TripTide: A Benchmark for Adaptive Travel Planning under Disruptions

**Research area:**  
Weather, Disruptions, and Adaptive Replanning.

**Local file:**  
`reference/2510.21329v1.pdf`

**Research problem:**  
Most travel-planning benchmarks focus on generating an initial itinerary, but real trips face disruptions such as transit cancellations, weather-related closures, and overbooked attractions.

**Why prior work was insufficient:**  
Static itinerary generation does not measure whether a planner can adapt while preserving traveler intent, responding appropriately, and changing the route only as much as needed.

**Core method:**  
TripTide introduces disruption scenarios, severity levels, traveler tolerance profiles, and metrics such as Preservation of Intent, Responsiveness, and Adaptability.

**Main result:**  
The paper provides an evaluation framework for adaptive travel planning under disruptions and argues that replanning quality involves semantic, spatial, and sequential changes, not only feasibility.

**Important limitation:**  
The local PDF appears as a recent arXiv-style paper; no venue is verified from the local first pages. It benchmarks LLM responses rather than implementing this project's solver/dashboard design.

**Connection to our project:**  
It is the most direct paper for the next research stage: compare original and weather-adjusted itineraries, preserve intent, and measure what changed.

**Feature, metric, or concept we may borrow:**  
Borrow Preservation of Intent, Responsiveness, disruption severity, tolerance profiles, and semantic/spatial/sequential change metrics.

**One sentence this paper can support in our paper:**  
Adaptive travel planning should be evaluated by how well a revised itinerary responds to disruption while preserving the traveler's original intent.

**Sections to read:**  
Read the abstract, disruption taxonomy, evaluation metrics, benchmark construction, and discussion. Skim model leaderboard details first.

**Questions I should be able to answer after reading:**  
What is preservation of intent? How can weather risk be framed as a disruption? Which TripTide metrics can be adapted to solver-backed route comparison?

## Chatti, Guesmi, and Muslim (2024) - Visualization for Recommendation Explainability: A Survey and New Perspectives

**Research area:**  
HCI, Explainability, and Mixed-Initiative Planning.

**Local file:**  
`reference/2305.11755v3.pdf`

**Research problem:**  
Explainable recommender systems need human-understandable rationales, and visual explanations have been studied across many aims, scopes, methods, and formats.

**Why prior work was insufficient:**  
Explanation research can be fragmented. Designers need guidance on what an explanation is for, what it explains, how it is generated, and how it is displayed.

**Core method:**  
The paper systematically reviews visual explanation work in recommender systems using dimensions such as explanation aim, scope, method, and format.

**Main result:**  
The paper provides a structure and design guidelines for visual explanations that can make recommender outputs more transparent and actionable.

**Important limitation:**  
It is a broad survey and not travel-specific. It does not provide a solver-backed route dashboard or test weather-aware itinerary explanations.

**Connection to our project:**  
It supports the dashboard as an explanatory visualization, not just a map. Route metrics, weather-risk layers, provenance, and comparison tables can be framed as visual explanations.

**Feature, metric, or concept we may borrow:**  
Borrow the distinction between explanation aim, scope, method, and format when designing why-selected, why-skipped, and what-would-change explanations.

**One sentence this paper can support in our paper:**  
Visual explanations in recommender systems should be designed around explicit explanation goals such as transparency, actionability, and user understanding.

**Sections to read:**  
Read the abstract, explanation dimensions, visualization guidelines, and future-work discussion. Skim exhaustive literature tables at first.

**Questions I should be able to answer after reading:**  
What is a visual explanation? What explanation goal does each dashboard element serve? Which explanations help users act, rather than only feel reassured?

## Santos, Delgado, Silver, and Silva (2025) - CityHood: An Explainable Travel Recommender System for Cities and Neighborhoods

**Research area:**  
HCI, Explainability, and Mixed-Initiative Planning.

**Local file:**  
`reference/2507.18778v1.pdf`

**Research problem:**  
Location-based recommenders often suggest places without explaining why a city or neighborhood matches a user's interests.

**Why prior work was insufficient:**  
Many systems focus on recommending POIs but provide limited transparency about spatial similarity, cultural alignment, or interest matching at city and neighborhood levels.

**Core method:**  
CityHood models user interests from large-scale Google Places reviews enriched with geographic, socio-demographic, political, and cultural indicators, then uses an explainable technique and natural-language explanations in a visual interface.

**Main result:**  
The paper demonstrates an interactive explainable travel recommender that helps users explore city and neighborhood recommendations with visible reasoning.

**Important limitation:**  
The local PDF is a short system/demo-style paper. It recommends cities and neighborhoods, not multi-day constrained itineraries with hotels, route repair, weather risk, or solver validation.

**Connection to our project:**  
It gives a travel-domain example of user-facing explainability. Our project can extend this idea from location explanation to route-tradeoff explanation.

**Feature, metric, or concept we may borrow:**  
Borrow multi-level spatial explanation and natural-language rationale design, while adding route feasibility and weather-risk tradeoffs.

**One sentence this paper can support in our paper:**  
Recent travel recommender systems increasingly expose explanations to help users understand why places match their interests.

**Sections to read:**  
Read the abstract, introduction, system description, explanation/interface sections, and conclusion. Skim implementation details unless reproducing the system.

**Questions I should be able to answer after reading:**  
How does CityHood explain place recommendations? What is missing for full itinerary planning? How can our project explain routes rather than only locations?
