# Core Paper Reading Cards

These eight papers are the minimum reading spine for this project. They are not the only relevant papers, but together they explain the main intellectual structure: itinerary recommendation, Tourist Trip Design, rich constraints, LLM planning limitations, solver-backed planning, disruption adaptation, visual explanation, and explainable travel recommendation.

Use `docs/literature_deep_read_study_report.md` for the full evidence bank after reading these cards.

## Halder, Lim, Chan, and Zhang (2024) - A Survey on Personalized Itinerary Recommendation: From Optimisation to Deep Learning

**Research area:**  
Personalized Travel Recommendation.

**Local file:**  
`reference/1-s2.0-S1568494623012188-main.pdf`

**Main goal:**  
Organize personalized itinerary recommendation across optimization, conventional recommendation, and deep-learning approaches.

**Research problem:**  
Personalized itinerary recommendation has been studied with many formulations, techniques, constraints, and data sources, making it hard for new researchers to understand the field.

**Why prior work was insufficient:**  
Older surveys did not fully organize the movement from optimization-based itinerary methods to deep-learning and personalization approaches. A new reader needs a taxonomy that connects POI recommendation, itinerary construction, user satisfaction, provider satisfaction, constraints, and learned models.

**Core method:**  
The paper is a survey and taxonomy. It reviews personalized itinerary recommendation work and organizes it by problem formulation, techniques, constraints, features, and satisfaction goals.

**Main result:**  
The paper shows that itinerary recommendation is broader than POI ranking: a system must select POIs and schedule them into a route while considering user preferences and many constraints.

**Important limitation:**  
**Project-team inference:** It is primarily a literature survey. It does not provide a user-facing dashboard, weather-aware route adaptation, or evidence that users understand route tradeoffs better.

**Publication use:**  
Essential citation for positioning the system between personalization and constrained planning. It cannot support claims that the current interface improves decisions.

**Project-polishing action:**  
Connect every interest control to a visible utility change and route consequence, while labeling browser previews separately from saved solver output.

**Current implementation status:**  
**partially implemented** - interest profiles and preview controls exist, but browser changes do not rerun the optimizer and explanations are incomplete.

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

**Main goal:**  
Systematically organize TTDP extensions, objectives, solution methods, and future research directions.

**Research problem:**  
The Tourist Trip Design Problem (TTDP) literature contains many variants, objectives, and solution methods, and needs a systematic review to organize extensions and future research directions.

**Why prior work was insufficient:**  
Classic orienteering work explains the mathematical root, but tourism applications add specific constraints such as multiple days, time windows, transport, preferences, and realistic travel behavior.

**Core method:**  
The paper conducts a systematic literature review and proposes a taxonomy of TTDP variants, objectives, and solution techniques.

**Main result:**  
The paper positions TTDP as a tourism-specific extension of the Orienteering Problem and shows that most practical trip design systems require richer constraints than the classic model.

**Important limitation:**  
**Project-team inference:** The review is mostly algorithmic and operations-research oriented. It does not focus on user understanding, explainability, or mixed-initiative dashboard design.

**Publication use:**  
Essential citation for the technical field definition and for identifying established constraints that should not be presented as algorithmic novelty.

**Project-polishing action:**  
Map every modeled objective and constraint to a dashboard element, validation check, and carefully scoped paper claim.

**Current implementation status:**  
**implemented** - constrained multi-day planning is present, while its complete human-centered inspection layer remains partial.

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

**Main goal:**  
Solve a TTDP with rich interdependent constraints through exact branch-and-check and branch-and-solve methods.

**Research problem:**  
Realistic TTDP instances include not only budgets, opening hours, and trip duration, but also mandatory visits, limits on types of locations, and ordering constraints.

**Why prior work was insufficient:**  
Simpler TTDP formulations do not fully capture rich practical tourism constraints. Directly solving all constraints in one model can also become difficult.

**Core method:**  
The paper proposes branch-and-check approaches. A master problem selects candidate locations while satisfying non-time constraints, and a subproblem checks whether the selected locations can be scheduled feasibly.

**Main result:**  
The method shows how separating selection and feasibility checking can handle richer tourism constraints more effectively than a simpler monolithic approach.

**Important limitation:**  
**Project-team inference:** The computational study is not about interface design, weather-aware replanning, or trust calibration, and its findings are bounded by its formulation and test instances.

**Publication use:**  
Essential algorithmic citation for explicit feasibility checking under rich tourism constraints. It cannot support claims about traveler comprehension or explanation usefulness.

**Project-polishing action:**  
Expose which constraint caused exclusion or repair and distinguish Gurobi-backed results from heuristic fallback routes.

**Current implementation status:**  
**partially implemented** - solver/fallback paths and status labels exist, but generalized constraint-block explanations do not.

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

**Main goal:**  
Evaluate complete LLM-generated travel plans across multiple dimensions rather than relying on fluency or isolated constraint checks.

**Research problem:**  
Existing LLM travel-planning benchmarks overemphasize constraint compliance and do not fully evaluate spatial, temporal, lodging, transportation, economy, and complete-plan quality.

**Why prior work was insufficient:**  
Many evaluations judge whether a plan sounds plausible or satisfies isolated constraints. Travel planning needs multi-dimensional assessment of the whole itinerary.

**Core method:**  
The paper introduces TravelEval, a benchmark with a six-dimensional evaluation framework covering accuracy, compliance, temporality, spatiality, economy, and utility, plus a realistic data sandbox and simulation-based global evaluation.

**Main result:**  
TravelEval provides a richer vocabulary for evaluating complete travel plans and shows that LLM-powered agents should be assessed beyond surface fluency.

**Important limitation:**  
**Project-team inference:** The local PDF appears as a recent arXiv-style paper; no peer-reviewed venue is verified from the local first pages. It evaluates LLM agents, not this project's dashboard directly.

**Publication use:**  
Essential evaluation citation for separating compliance, temporality, spatiality, economy, and utility. It cannot establish the project's performance before those metrics are computed against baselines.

**Project-polishing action:**  
Produce one reproducible metric table per route covering feasibility, travel time, budget, utility, weather risk, lodging completeness, runtime, and solver/fallback status.

**Current implementation status:**  
**partially implemented** - several route and validation metrics exist, but no complete benchmark-style comparison has been run.

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

**Main goal:**  
Combine LLM-based travel knowledge with automated planning to produce valid, utility-aware plans within a formal model.

**Research problem:**  
LLM-only travel planners can generate fluent itineraries that fail to satisfy constraints or produce coherent high-quality plans.

**Why prior work was insufficient:**  
Traditional planning systems require formal problem definitions, while LLM-only systems can miss formal validity. The paper tries to combine natural-language flexibility with automated planning guarantees.

**Core method:**  
TRIP-PAL uses LLMs to extract and translate travel information and user information into data structures for an automated planner, then uses the planner to generate valid plans under the formal model.

**Main result:**  
The paper reports that TRIP-PAL produces more valid and higher-utility travel plans than an LLM-only baseline in the tested scenarios.

**Important limitation:**  
**Author-reported limitation:** Evaluation is limited to day planning with time discretized into 15-minute segments. **Project-team inference:** The guarantee is only as strong as the formal model, extracted data, and planner assumptions. The local PDF appears arXiv-style, with no venue verified from its first pages.

**Publication use:**  
Essential hybrid-planning citation for formal feasibility within an encoded model. It cannot prove that this project's multi-day routes are globally optimal or its explanations are useful.

**Project-polishing action:**  
Display the modeled scope, omitted constraints, input provenance, solver result, and fallback path behind every validity claim.

**Current implementation status:**  
**partially implemented** - solver-backed planning exists, while the LLM knowledge-to-planner layer does not.

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

**Main goal:**  
Benchmark adaptive travel planning under disruptions while measuring repair scope, responsiveness, and preservation of traveler intent.

**Research problem:**  
Most travel-planning benchmarks focus on generating an initial itinerary, but real trips face disruptions such as transit cancellations, weather-related closures, and overbooked attractions.

**Why prior work was insufficient:**  
Static itinerary generation does not measure whether a planner can adapt while preserving traveler intent, responding appropriately, and changing the route only as much as needed.

**Core method:**  
TripTide introduces disruption scenarios, severity levels, traveler tolerance profiles, and metrics such as Preservation of Intent, Responsiveness, and Adaptability.

**Main result:**  
The paper provides an evaluation framework for adaptive travel planning under disruptions and argues that replanning quality involves semantic, spatial, and sequential changes, not only feasibility.

**Important limitation:**  
**Project-team inference:** The local PDF appears as a recent arXiv-style paper; no venue is verified from the local first pages. It benchmarks LLM responses rather than implementing this project's solver/dashboard design.

**Publication use:**  
Essential disruption citation for preservation of intent and scoped repair. It does not prove that the project's current route alternatives constitute live replanning.

**Project-polishing action:**  
Compare original and weather-adjusted routes using preserved stops, edit distance, utility change, risk reduction, and explanations for changed stops.

**Current implementation status:**  
**partially implemented** - weather-aware alternatives exist, but live disruption handling and explicit intent-preservation metrics do not.

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

**Main goal:**  
Classify visual explanations in recommender systems and connect explanation purposes to methods, formats, interaction, and evaluation.

**Research problem:**  
Explainable recommender systems need human-understandable rationales, and visual explanations have been studied across many aims, scopes, methods, and formats.

**Why prior work was insufficient:**  
Explanation research can be fragmented. Designers need guidance on what an explanation is for, what it explains, how it is generated, and how it is displayed.

**Core method:**  
The paper systematically reviews visual explanation work in recommender systems using dimensions such as explanation aim, scope, method, and format.

**Main result:**  
The paper provides a structure and design guidelines for visual explanations that can make recommender outputs more transparent and actionable.

**Important limitation:**  
**Project-team inference:** It is a broad survey and not travel-specific. It does not provide a solver-backed route dashboard or test weather-aware itinerary explanations.

**Publication use:**  
Essential citation for framing route overlays, weather layers, and aligned metrics as visual explanation. It cannot establish their effectiveness without a user study.

**Project-polishing action:**  
Assign each visual element a concrete explanation task, such as spatial comparison, tradeoff inspection, or skipped-stop diagnosis.

**Current implementation status:**  
**partially implemented** - visual route comparison exists, while route-difference and skipped-stop explanations need stronger integration.

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

**Main goal:**  
Recommend cities and neighborhoods while explaining how regional features align with a traveler's interests.

**Research problem:**  
Location-based recommenders often suggest places without explaining why a city or neighborhood matches a user's interests.

**Why prior work was insufficient:**  
Many systems focus on recommending POIs but provide limited transparency about spatial similarity, cultural alignment, or interest matching at city and neighborhood levels.

**Core method:**  
CityHood models user interests from large-scale Google Places reviews enriched with geographic, socio-demographic, political, and cultural indicators, then uses an explainable technique and natural-language explanations in a visual interface.

**Main result:**  
The paper demonstrates an interactive explainable travel recommender that helps users explore city and neighborhood recommendations with visible reasoning.

**Important limitation:**  
**Project-team inference:** The local PDF is a short system/demo-style paper. It recommends cities and neighborhoods, not multi-day constrained itineraries with hotels, route repair, weather risk, or solver validation.

**Publication use:**  
Essential recent travel-interface citation for interest-linked explanations. It does not validate contrastive route explanations or constrained itinerary construction.

**Project-polishing action:**  
Explain each selected stop through both interest fit and route feasibility, then contrast it with a high-value skipped alternative.

**Current implementation status:**  
**partially implemented** - interest scores and basic why-selected text exist, but contrastive explanations remain incomplete.

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
