# Integrated Core Literature Review

## Personalized, Solver-Backed, Adaptive, and Explainable Travel Itinerary Planning

## 1. Purpose And How To Use This Guide

This is the second document a new researcher should read after `docs/literature_onboarding_guide.md`. The onboarding guide defines the project and the field. This guide teaches the eight papers that provide the minimum intellectual spine for understanding the proposed research. The full 43-paper notes remain in `docs/literature_deep_read_study_report.md` and should be used as an evidence bank rather than read from beginning to end on the first pass.

The eight core papers are:

1. Halder, Lim, Chan, and Zhang (2024), *A Survey on Personalized Itinerary Recommendation: From Optimisation to Deep Learning*.
2. Ruiz-Meza and Montoya-Torres (2022), *A Systematic Literature Review for the Tourist Trip Design Problem: Extensions, Solution Techniques and Future Research Lines*.
3. Vu, Kergosien, Mendoza, and Desport (2022), *Branch-and-Check Approaches for the Tourist Trip Design Problem with Rich Constraints*.
4. Chen et al. (2026), *TravelEval: A Comprehensive Benchmarking Framework for Evaluating LLM-Powered Travel Planning Agents*.
5. de la Rosa, Gopalakrishnan, Pozanco, Zeng, and Borrajo (2024), *TRIP-PAL: Travel Planning with Guarantees by Combining Large Language Models and Automated Planners*.
6. Karmakar et al. (2025), *TripTide: A Benchmark for Adaptive Travel Planning under Disruptions*.
7. Chatti, Guesmi, and Muslim (2024), *Visualization for Recommendation Explainability: A Survey and New Perspectives*.
8. Santos, Delgado, Silver, and Silva (2025), *CityHood: An Explainable Travel Recommender System for Cities and Neighborhoods*.

Together, these papers show that travel planning is not one problem. It combines preference modeling, selective routing, constraint satisfaction, complete-plan evaluation, disruption repair, explanation, and user control. The practical lesson is:

> A useful travel-planning system must estimate what the traveler values, construct a feasible itinerary under an explicit model, adapt carefully when conditions change, and expose enough reasoning for the traveler to inspect and revise the result.

The user-provided integrated review was used as the organizational seed for this document. Claims here were checked against the local PDFs and repository. When they disagree, the PDF is the authority for the paper and the code is the authority for the current project.

## 2. Five Learning Layers

The papers form five connected learning layers. These layers describe the flow of a planning system, while the onboarding guide's four research areas describe the academic communities surrounding that flow.

### Layer 1: Preference And Recommendation

**Question:** Which attractions would this traveler value?

Halder et al. explain personalized Point of Interest (POI) and itinerary recommendation. This layer supplies interest profiles, POI utility, popularity, contextual signals, and estimates of user satisfaction. It corresponds to **Personalized Travel Recommendation** in the onboarding guide. Preference is necessary but not sufficient: a traveler may value ten distant outdoor attractions even though only four fit and two become poor choices in severe weather.

### Layer 2: Constrained Route Construction

**Question:** Which preferred attractions can actually be visited, and in what order?

Ruiz-Meza and Montoya-Torres define the Tourist Trip Design Problem (TTDP), and Vu et al. show why rich constraints require explicit feasibility handling. This layer corresponds to **Operations Research and Tourist Trip Design**. A plan must choose a subset of optional POIs and arrange them into a route that respects modeled time, cost, travel, lodging, and other constraints.

### Layer 3: Language And Agent Interaction

**Question:** How might a traveler express goals without manually defining an optimization model?

TRIP-PAL places a Large Language Model (LLM) in front of an automated planner. The LLM gathers or structures information; the planner handles encoded feasibility and utility. This layer intersects personalization and solver-backed planning, but it is **future work for this repository**. The current project has no natural-language-to-symbolic compiler or monitoring agent.

### Layer 4: Evaluation And Adaptation

**Question:** Is the complete itinerary good, and what should happen when conditions change?

TravelEval broadens plan evaluation beyond surface plausibility or isolated constraint compliance. TripTide evaluates how itineraries change under disruptions and whether revisions preserve intent. This layer corresponds to **Weather, Disruptions, and Adaptive Replanning**, with TravelEval supplying an objective evaluation vocabulary.

### Layer 5: Explanation And User Control

**Question:** Can the traveler understand, compare, challenge, and revise the system's choices?

Chatti et al. organize visual explanation design, while CityHood demonstrates a user-facing explainable travel recommender. This layer corresponds to **HCI, Explainability, and Mixed-Initiative Planning**.

```text
Traveler values and trip context
                ->
Personalized POI utility
                ->
Formal constraints and route construction
                ->
Complete-plan validation
                ->
Weather/disruption-aware adaptation
                ->
Visual and textual explanation
                ->
User comparison, critique, and acceptance
```

The current project begins with structured inputs rather than natural language. It already reaches route generation, alternatives, weather-risk representation, dashboard export, and validation. Its strongest next step lies between adaptation and explanation, not in adding an autonomous agent first.

## 3. Halder Et Al. (2024): Personalized Itinerary Recommendation

**Verified citation and publication status:** Halder, Lim, Chan, and Zhang (2024), *A Survey on Personalized Itinerary Recommendation: From Optimisation to Deep Learning*, *Applied Soft Computing*, volume 152, article 111200.

**Local PDF:** `reference/1-s2.0-S1568494623012188-main.pdf`

**Research layer:** Preference and recommendation.

**Main goal:** Organize personalized itinerary recommendation across optimization, conventional recommendation, and deep-learning approaches.

**Main idea:** POI recommendation and itinerary recommendation are related but different. A POI recommender estimates which places are attractive. An itinerary recommender must additionally choose a feasible subset, order it, and account for the time and travel needed to execute the complete plan.

**Important terminology and method:** This is a survey and taxonomy rather than a new planner. It distinguishes next-POI, top-k POI, package, and sequential itinerary recommendation; personalized and non-personalized methods; and user-side and provider-side objectives. Reviewed features include interests, popularity, history, social and spatial influence, travel time, visit duration, opening windows, queues, and other constraints.

```text
Preference model: How valuable is POI i to this traveler?
Planning model: Can the selected POIs form an executable itinerary?
```

**Important finding or experimental takeaway:** Personalization does not remove the need for optimization. A highly personalized POI set may remain infeasible because it is geographically scattered, exceeds available time, or conflicts with opening or travel requirements. The survey identifies explainability and richer user settings as future directions.

**Principal limitation:** **Project-team inference:** This broad survey does not provide an end-to-end weather-aware system, route-comparison interface, disruption repair method, or evidence that users understand optimization tradeoffs. Its future-work discussion mentions explainable recommendation and group or family planning but does not establish how to build or evaluate this dashboard.

**Connection to the implemented project:** The repository implements interest profiles, interest-adjusted utility, POI scoring, weather-risk signals, and multi-day route generation. This paper supports describing that layer as personalized itinerary recommendation rather than generic attraction ranking.

**What is already established and should not be claimed as novel:** Personalized rewards, category interests, popularity signals, and preference-aware itineraries are established ideas. Their presence is necessary integration, not sufficient novelty.

**Defensible publication use:** Essential citation for the claim that personalized itinerary recommendation combines preference-aware selection with itinerary construction under constraints. It cannot support a claim that the interface improves decisions or calibrated trust.

**Concrete project-polishing upgrade:** Make preference effects inspectable. Each interest control should show its effect on utility and route consequences. Saved solver-backed routes must remain distinct from browser-only previews that do not rerun Gurobi.

**Current implementation status:** **partially implemented** - profiles, scores, and previews exist, but browser changes do not trigger solver-backed revision and explanations do not cover every decision.

**Sections to read carefully:** Abstract; introduction; problem-formulation and taxonomy sections; constraint and feature discussions; future directions.

**Sections to skim initially:** Exhaustive algorithm and dataset tables. Return to them when choosing a personalization baseline.

**Questions to answer after reading:** What separates POI recommendation from itinerary recommendation? Which preference features are common? Why can a personalized set remain infeasible? Which part of the current system is recommendation, and which is planning?

## 4. Ruiz-Meza And Montoya-Torres (2022): Tourist Trip Design

**Verified citation and publication status:** Ruiz-Meza and Montoya-Torres (2022), *A Systematic Literature Review for the Tourist Trip Design Problem: Extensions, Solution Techniques and Future Research Lines*, *Operations Research Perspectives*, volume 9, article 100228.

**Local PDF:** `reference/1-s2.0-S2214716022000069-main.pdf`

**Research layer:** Constrained route construction.

**Main goal:** Systematically organize TTDP formulations, extensions, objectives, solution methods, and future directions.

**Main idea:** TTDP is a tourism-specific selective-routing problem rooted in the Orienteering Problem. Many attractions are optional, each provides utility, and the planner must select as well as sequence visits under limited resources.

**Important terminology and method:** The systematic review classifies deterministic and uncertain settings, single- and multi-objective formulations, exact methods, heuristics, metaheuristics, and application extensions. The core abstraction is to maximize collected tourist utility subject to route continuity and resource constraints. TTDP enriches this with multiple days, time windows, visit durations, budgets, transport modes, traveler preferences, groups, and sustainability concerns.

**Important finding or experimental takeaway:** Travel planning is not shortest-path computation. The shortest route through an unattractive set is not a good itinerary, while the most attractive set may not fit. Selection and sequencing must be considered together.

**Principal limitation:** **Project-team inference:** The review is operations-research centered and gives limited attention to explanation, mixed-initiative revision, LLM interfaces, or controlled dashboard evaluation. It also cannot establish that a combination of known constraints is a new algorithmic contribution.

**Connection to the implemented project:** This is the main foundation for describing the repository as a weather-aware, multi-day TTDP-style prototype with optional POI selection, hotels or base cities, route alternatives, weather-risk terms, interest-adjusted utility, travel penalties, and daily feasibility conditions.

**What is already established and should not be claimed as novel:** POI selection, multiple days, time budgets, opening windows, budget limits, personalized rewards, and travel-distance objectives are established. Hotels, weather exposure, and nature regions may enrich the application, but integration alone is not proven novelty.

**Defensible publication use:** Essential citation for defining the technical problem and identifying established formulation elements. It supports positioning the contribution at the intersection of constrained planning and inspectable adaptation.

**Concrete project-polishing upgrade:** Add a formulation-to-interface table. For each variable, hard constraint, soft objective, and uncertainty signal, identify its dashboard representation and validation check.

**Current implementation status:** **implemented** - constrained multi-day selection and routing are present, with Gurobi-backed and heuristic/fallback paths. The human-centered inspection layer remains partial.

**Sections to read carefully:** Abstract; introduction; TTDP definition; taxonomy; solution-technique synthesis; future research lines.

**Sections to skim initially:** Long classification tables and individual method summaries. Use them later to check whether a proposed feature is already common.

**Questions to answer after reading:** How is TTDP related to orienteering? Why is TTDP selective? Which project features are established constraints? Where could the contribution lie if the base formulation is not new?

## 5. Vu Et Al. (2022): Rich Constraints And Branch-And-Check

**Verified citation and publication status:** Vu, Kergosien, Mendoza, and Desport (2022), *Branch-and-Check Approaches for the Tourist Trip Design Problem with Rich Constraints*, *Computers & Operations Research*, volume 138, article 105566.

**Local PDF:** `reference/1-s2.0-S0305054821002963-main.pdf`

**Research layer:** Constrained route construction.

**Main goal:** Develop exact branch-and-check and branch-and-solve methods for TTDP with rich logical, category, budget, and scheduling constraints.

**Main idea:** Selecting attractive locations and scheduling them are separable sources of difficulty. A master problem selects a candidate set while enforcing non-temporal rules. A subproblem checks whether those locations can be arranged into a feasible trip. An attractive set can be rejected when its timing is impossible.

**Important terminology and method:** The formulation includes mandatory visits, category limits, exclusion and implication rules, precedence, budgets, time windows, and maximum duration. In branch-and-check, a master-model incumbent is tested by a scheduling subproblem. If scheduling fails, a feasibility cut excludes the incompatible selection. If it succeeds, the incumbent and optimality information can be updated. Exactness remains scoped to the stated formulation and assumptions.

**Important finding or experimental takeaway:** The customized methods perform strongly against the tested CPLEX configurations. The broader lesson is decomposition: rich constraints require a mechanism that distinguishes an attractive selection from a schedulable itinerary.

**Principal limitation:** **Project-team inference:** The study evaluates exact optimization on designed instances. It does not address multi-day weather adaptation, user explanations, traveler understanding, or trust calibration. Performance can depend on instance structure, solver configuration, and encoded constraints.

**Connection to the implemented project:** The repository has Gurobi-backed route optimization and heuristic/fallback repair, not this exact branch-and-check architecture. The selection-versus-feasibility distinction still applies when a high-scoring POI is omitted because of time, detour, weather, diversity, lodging, or route-flow conditions.

**What is already established and should not be claimed as novel:** Explicit feasibility checks and decomposition for rich TTDP are established. Merely using Gurobi or separating candidate generation from route checking is not sufficient novelty.

**Defensible publication use:** Essential citation for explaining why rich constraints require structured validation. It cannot support claims about explanation usefulness or real-world global optimality.

**Concrete project-polishing upgrade:** Export a reason code whenever an important POI or region is excluded, moved, or removed during repair. Reasons should identify time, detour, weather, cost, diversity, base-city, or mandatory-stop conflicts.

**Current implementation status:** **partially implemented** - solver/fallback states and some nature-region why-not codes exist, but generalized constraint-block explanations do not.

**Sections to read carefully:** Abstract; rich TTDP definition; master/subproblem decomposition; branch-and-check logic; computational conclusion.

**Sections to skim initially:** Detailed proofs and full tables unless reproducing the exact method.

**Questions to answer after reading:** Why can selection succeed while scheduling fails? What is a feasibility cut? Which project exclusions could be explained from solver or repair state? Why must exactness remain scoped to the model?

## 6. Chen Et Al. (2026): TravelEval

**Verified citation and publication status:** Chen et al. (2026), *TravelEval: A Comprehensive Benchmarking Framework for Evaluating LLM-Powered Travel Planning Agents*. The local `v1` PDF is an arXiv-style preprint; no peer-reviewed venue is verified from the paper.

**Local PDF:** `reference/2606.01046v1.pdf`

**Research layer:** Evaluation and adaptation.

**Main goal:** Evaluate complete LLM-generated travel plans across multiple dimensions rather than relying on fluency, isolated fields, or constraint compliance alone.

**Main idea:** Two plans can satisfy explicit constraints yet differ in practical quality. One may zigzag, waste time, choose inconvenient lodging, or allocate budget poorly. Evaluation must represent dependencies among activities, transport, lodging, timing, and cost.

**Important terminology and method:** The six dimensions are exactly **accuracy, compliance, temporality, spatiality, economy, and utility**. TravelEval also introduces a realistic data sandbox and simulation-based global evaluation. Global evaluation matters because one event changes what remains possible: lodging affects the next morning, queueing changes later arrivals, and transport alters both time and cost.

**Important finding or experimental takeaway:** Across evaluated LLM agents and planning strategies, planning remains difficult when preferences and constraints interact. Tool use or elaborate prompting can help some dimensions without reliably improving every part of the complete plan. Fluency is not evidence of spatial, temporal, or economic quality.

**Principal limitation:** **Project-team inference:** TravelEval is an evaluation benchmark, not a solver-backed repair method or explanation study. Its target is LLM agents, so its dimensions must be operationalized for California route artifacts. Its recent preprint status requires cautious citation.

**Connection to the implemented project:** The dashboard exposes route metrics, weather risk, hotels, alternatives, solver/fallback state, and validation artifacts. TravelEval supplies a structure for organizing these into complete-plan evaluation rather than scattered debug values.

**What is already established and should not be claimed as novel:** Multidimensional evaluation and simulation-based checking are established. Adding a metric panel is not a contribution unless measures are operationalized, compared, and tied to a research question.

**Defensible publication use:** Essential citation for separating compliance, time, space, economy, and utility. It cannot show that this project performs well until alternatives are evaluated against baselines.

**Concrete project-polishing upgrade:** Produce a reproducible route table containing feasibility, daily utilization, travel time, spatial coherence, budget, lodging completeness, utility, weather exposure, runtime, and solver/fallback status. Validate routes sequentially rather than checking POIs in isolation.

**Current implementation status:** **partially implemented** - several objective and validation fields exist, but no complete TravelEval-style comparative experiment has been run.

**Sections to read carefully:** Abstract; motivation; six-dimensional framework; sandbox; global simulation; main failure analysis.

**Sections to skim initially:** Model leaderboards. Return to them when selecting baselines.

**Questions to answer after reading:** Why is compliance insufficient? What does each dimension measure? Which dimensions appear in project artifacts? What is needed for complete-plan simulation?

## 7. De La Rosa Et Al. (2024): TRIP-PAL

**Verified citation and publication status:** de la Rosa, Gopalakrishnan, Pozanco, Zeng, and Borrajo (2024), *TRIP-PAL: Travel Planning with Guarantees by Combining Large Language Models and Automated Planners*. The local PDF is an arXiv-style preprint; no peer-reviewed venue is verified from its first pages.

**Local PDF:** `reference/2406.10196v1.pdf`

**Research layer:** Language and agent interaction, connected to constrained route construction.

**Main goal:** Combine LLM travel knowledge with automated planning so the final plan satisfies encoded constraints and optimizes encoded utility.

**Main idea:** LLMs and planners have complementary roles. An LLM can identify candidate POIs, structure visit information, and help translate user information. A formal planner can solve an oversubscription problem: choose the most valuable feasible subset when more activities are available than time permits.

**Important terminology and method:** TRIP-PAL asks the LLM for candidate locations, visit durations in 15-minute units, and popularity represented as utility from 1 to 10. It creates a travel-time graph and passes the problem to an automated planner. In the experiment, travel times are randomly assigned from 15 to 60 minutes in 15-minute units, although the architecture could query an external service. The guarantee applies to supplied data and the encoded model.

**Important finding or experimental takeaway:** GPT-4-only plans frequently violate visit duration, travel time, or total tourism-time requirements in the tested scenarios. TRIP-PAL returns valid plans under its model and improves encoded utility relative to the LLM-only comparison.

**Principal limitation:** **Author-reported limitation:** Experiments are limited to day planning with 15-minute discretization. **Project-team inference:** Popularity is only a utility proxy, random travel times reduce realism, and correctness depends on extraction and translation. The guarantee does not imply full real-world feasibility or faithful language interpretation.

**Connection to the implemented project:** The repository resembles the planner side:

```text
Structured configuration -> utility and constraints -> Gurobi/heuristic route -> dashboard
```

It has no LLM extractor, free-text intent schema, extraction validation, or compiler linking user language to solver inputs.

**What is already established and should not be claimed as novel:** Hybrid LLM-to-symbolic travel planning is prior work. A future language feature needs evaluated extraction and validation, not only a chat box in front of the solver.

**Defensible publication use:** Essential citation for assigning language interpretation to an LLM and encoded feasibility to a planner. It cannot justify claiming that the current project is agent-assisted.

**Concrete project-polishing upgrade:** Keep natural-language input outside the next primary milestone. If added later, require a visible schema, ambiguity handling, user confirmation, deterministic compilation, solver validation, and grounded explanations.

**Current implementation status:** **partially implemented** - solver-backed and fallback planning exist; the LLM information-to-planner layer is **unsupported by current code**.

**Sections to read carefully:** Abstract; architecture; oversubscription formulation; data translation; experiments; invalid-plan analysis; limitations.

**Sections to skim initially:** Prompt examples and planner syntax unless implementing the language layer.

**Questions to answer after reading:** What should the LLM do, and what should the planner do? What does the guarantee cover? Which assumptions reduce realism? What is missing before claiming LLM-to-symbolic planning?

## 8. Karmakar Et Al. (2025): TripTide

**Verified citation and publication status:** Karmakar et al. (2025), *TripTide: A Benchmark for Adaptive Travel Planning under Disruptions*. The local PDF is arXiv:2510.21329v1 and contains placeholder conference metadata, so it should be cited as a preprint unless a venue is independently verified.

**Local PDF:** `reference/2510.21329v1.pdf`

**Research layer:** Evaluation and adaptation.

**Main goal:** Benchmark whether travel planners adapt appropriately to disruptions while respecting traveler tolerance and preserving original intent.

**Main idea:** Initial generation is only the first planning problem. Weather closures, transit cancellations, accommodation problems, illness, fatigue, and safety concerns can invalidate a trip. Good adaptation changes enough to resolve the problem while preserving valuable unaffected content.

**Important terminology and method:** TripTide distinguishes step-level, day-level, and plan-level disruptions. It includes profiles such as the flexible **Flexi-Venturer** and change-averse **Plan-Bound** traveler. Metrics include **Preservation of Intent**, **Responsiveness**, and semantic, spatial, and sequential **Adaptability**. Sequential change uses normalized edit distance; the broader set compares feasibility, goals, meaning, geography, and order.

**Important finding or experimental takeaway:** LLM performance differs by disruption scope, persona, and metric. A revision can appear responsive while making unnecessary semantic, spatial, or sequential changes. Preservation and repair scope are therefore first-class outcomes.

**Principal limitation:** **Project-team inference:** TripTide benchmarks LLM revisions rather than providing a mathematical repair optimizer or explanation study. It does not establish formal constraint satisfaction, and its preprint status and placeholder venue text require caution.

**Connection to the implemented project:** The repository produces weather-aware alternatives but not a true live disruption workflow. Current artifacts can support an offline original-versus-adjusted experiment. Weather severity and duration can induce local, day-level, or plan-level repair.

**What is already established and should not be claimed as novel:** Disruption taxonomies, intent preservation, responsiveness, and semantic/spatial/sequential change metrics have direct travel-planning precedent. A contribution must operationalize them in a solver-backed setting or test them in an inspectable interface.

**Defensible publication use:** Essential citation for defining adaptation as scoped repair that preserves intent. It cannot support a claim that current alternatives are live replanning or outperform a baseline.

**Concrete project-polishing upgrade:** Add original-versus-weather-adjusted comparison with preserved stops, route edit distance, utility retained, risk reduction, added travel, lodging changes, and a reason for every changed stop.

**Current implementation status:** **partially implemented** - weather-aware alternatives exist, but explicit disruption events, live monitoring, repair-scope rules, and intent metrics do not.

**Sections to read carefully:** Abstract; disruption taxonomy; tolerance profiles; automatic metrics; benchmark construction; discussion.

**Sections to skim initially:** Full leaderboard and prompt appendix.

**Questions to answer after reading:** What separates local repair from replanning? How is intent preservation measured? How could weather severity determine repair scope? Which metrics can be computed from current artifacts?

## 9. Chatti Et Al. (2024): Visualization For Recommendation Explainability

**Verified citation and publication status:** Chatti, Guesmi, and Muslim (2024), *Visualization for Recommendation Explainability: A Survey and New Perspectives*. The local file is arXiv:2305.11755v3; no peer-reviewed venue is verified from the PDF.

**Local PDF:** `reference/2305.11755v3.pdf`

**Research layer:** Explanation and user control.

**Main goal:** Classify visual explanations in recommender systems and derive design guidance.

**Main idea:** An explanation should have a purpose. A chart or map is not explanatory merely because it looks polished. Visual form, interaction, and evidence must match what the user needs to understand or do.

**Important terminology and method:** The survey reviews work from 2000 through 2021 across **explanation aim, scope, method, and format**. Aims include transparency, effectiveness, efficiency, satisfaction, persuasiveness, scrutability, trust-related goals, and control. It distinguishes transparency, which should faithfully expose reasoning, from a plausible justification that may be decoupled from the algorithm.

**Important finding or experimental takeaway:** Visual explanation needs theory-informed design and evaluation. Explanations can support two-way interaction: the system exposes assumptions, and the user corrects preferences or constraints. Effectiveness must be measured against the intended aim.

**Principal limitation:** **Project-team inference:** The survey is broad rather than travel-specific and covers literature through 2021. It does not identify which map design improves route comprehension or guarantee that post-hoc visuals and summaries are faithful to an optimizer.

**Connection to the implemented project:** The dashboard contains potential explanation surfaces: route maps, weather layers, metrics, hotels, stop details, source confidence, and alternatives. They become explanations only when each supports a defined user task.

**What is already established and should not be claimed as novel:** Visual explanation, interactive explanation, transparency, and scrutability are established. Adding a map or rationale is not enough for novelty or effectiveness.

**Defensible publication use:** Essential citation for framing route overlays, aligned metrics, provenance, and differences as visual explanations. It cannot establish that the interface improves understanding.

**Concrete project-polishing upgrade:** Assign a purpose to each element: map for spatial change, timeline for temporal feasibility, metrics for tradeoffs, skipped-POI panel for constraint consequences, provenance for evidence, and before/after view for adaptation.

**Current implementation status:** **partially implemented** - comparison and weather layers exist, while generalized why-skipped, what-would-change, and route-difference explanations remain incomplete.

**Sections to read carefully:** Abstract; four dimensions; explanation aims; visualization and interaction guidance; future perspectives.

**Sections to skim initially:** Exhaustive classification tables. Return to them for design precedents and measures.

**Questions to answer after reading:** What makes a visualization an explanation? How do transparency and justification differ? What task does each dashboard element support? Which outcomes require a user study?

## 10. Santos Et Al. (2025): CityHood

**Verified citation and publication status:** Santos, Delgado, Silver, and Silva (2025), *CityHood: An Explainable Travel Recommender System for Cities and Neighborhoods*. The local six-page paper is demo-style; no archival venue is verified from the PDF.

**Local PDF:** `reference/2507.18778v1.pdf`

**Research layer:** Explanation and user control, connected to preference and recommendation.

**Main goal:** Recommend cities and neighborhoods while exposing how regional features align with traveler interests.

**Main idea:** Explanation can be integrated into geographic exploration rather than appended as a generic paragraph. CityHood supports city- and neighborhood-level recommendations and lets users inspect associated features.

**Important terminology and method:** CityHood models interests from Google Places reviews enriched with geographic, socio-demographic, political, and cultural indicators. Review volume proxies interest in an area. Separate LightGBM classifiers predict city and neighborhood interest. LIME provides local feature explanations, and an LLM turns regional features into natural-language descriptions. The interface supports maps and feedback.

**Important finding or experimental takeaway:** The demo shows users exploring recommendations, inspecting features, moving between spatial levels, and providing feedback. It demonstrates explainable recommendation rather than validating complete itineraries.

**Principal limitation:** **Project-team inference:** CityHood recommends regions rather than feasible multi-day routes. Review volume can conflate activity with preference. LIME and LLM summaries are post-hoc components whose fidelity should not be assumed. The short paper does not evaluate weather adaptation, hotels, or solver validation.

**Connection to the implemented project:** CityHood is a nearby interface precedent. This project can extend explanation from place fit to route consequences: why a POI appears on Day 2, why a hotel is the base, why a weather-adjusted route changed, or why a distant scenic stop was skipped.

**What is already established and should not be claimed as novel:** Explainable travel recommendation, multiscale exploration, feature attribution, and natural-language rationale exist. The differentiator must concern constrained route decisions and validated adaptation.

**Defensible publication use:** Essential travel-interface citation for interest-linked place explanations. It cannot validate contrastive route explanations or weather-aware planning.

**Concrete project-polishing upgrade:** Support progressive inspection from trip to day to route to POI to explanation. Show interest fit and feasibility for a selected stop, then the binding constraint and required change for a skipped alternative.

**Current implementation status:** **partially implemented** - map exploration, interest scores, and basic why-selected text exist, but contrastive and counterfactual route explanations do not.

**Sections to read carefully:** Abstract; introduction; interest modeling; LightGBM/LIME pipeline; interface; conclusion.

**Sections to skim initially:** Geographic feature construction unless reproducing the model.

**Questions to answer after reading:** How does CityHood infer and explain interest? Which explanations are model-grounded versus generated? What is missing between place and route explanation? How can the dashboard support progressive inspection?

## 11. Cross-Paper Synthesis

| Layer | Core question | What prior work supplies | What remains for this project |
| --- | --- | --- | --- |
| Preference | What does the traveler value? | Personalized utility and itinerary taxonomies | Make interest effects inspectable and weather-sensitive |
| Route construction | What fits, and in what order? | TTDP formulations and rich feasibility methods | Report solver/fallback scope and explain exclusions |
| Language interaction | How are informal goals structured? | LLM-to-planner hybrid precedent | Future only: validated natural-language compiler |
| Evaluation | Is the complete route good? | Six-dimensional and simulation-based evaluation | Operationalize metrics for route artifacts |
| Adaptation | What changes after disruption? | Repair scope, responsiveness, and intent metrics | Solver-backed weather repair and before/after comparison |
| Explanation | Can users inspect and revise? | Visual taxonomy and travel UI precedent | Contrastive explanations and controlled evaluation |

Within this eight-paper core set, no single paper covers all layers. This is a useful integration observation, not proof that no such system exists in the broader literature. The 43-paper evidence bank and future searches remain necessary before making a novelty claim.

The strongest intersection is:

> Weather-aware travel planning as accountable, inspectable, solver-backed itinerary adaptation under uncertainty.

This joins three focused gaps: formal planners provide limited traveler-facing inspection; TripTide evaluates LLM revisions rather than solver-backed repair; and explainable travel systems explain place fit rather than constrained multi-day route changes.

## 12. Current Project Positioning

The repository can support a careful description as a California-focused prototype with structured profiles, interest-adjusted POI utility, hotels or base cities, nature regions, weather-risk signals, multi-day route alternatives, Gurobi-backed optimization where available, heuristic/fallback repair, dashboard exports, route states, source confidence, and validation artifacts.

Integration is useful engineering but is not automatically a research result. Future writing should distinguish implemented capability, algorithmic contribution requiring baseline experiments, human-centered contribution requiring a study, and future architecture.

The strongest evidence-supported contribution is not a better optimizer or autonomous agent. It is an inspectable decision-support workflow exposing weather-aware alternatives, constraint consequences, and solver/fallback status so users can compare and revise plans.

## 13. Architecture In Three Horizons

### Horizon 1: Current System

```text
Structured preferences + constraints + weather-risk inputs
                         ->
              Candidate POIs and utility
                         ->
       Gurobi-backed optimization or heuristic repair
                         ->
       Routes, hotels/base cities, and alternatives
                         ->
          Dashboard and validation artifacts
```

Browser previews are not solver reruns, and not every displayed route should be called globally optimal.

### Horizon 2: Next Research Stage

```text
Original route + defined weather disruption
                       ->
        Scoped solver-backed route repair
                       ->
 Complete-route validation and intent metrics
                       ->
 Original-versus-adjusted visual comparison
                       ->
 User changes tolerance and chooses a route
```

This should be the primary target. It directly connects TripTide's repair concepts, TTDP feasibility, TravelEval's metrics, and explanation research.

### Horizon 3: Future Agent-Assisted Extension

```text
Natural-language request or monitored event
                       ->
 Agent extracts structured intent/disruption
                       ->
 User or validator confirms the structure
                       ->
 Solver generates and validates a route
                       ->
 Agent communicates grounded change reasons
```

This is not implemented. It requires extraction, a validated schema, ambiguity handling, tool orchestration, live data, monitoring, and end-to-end evaluation. It should remain future work unless developed as a separate contribution.

## 14. Evidence-Supported Research Gaps

### Gap 1: Solver-Backed Intent-Preserving Weather Repair

TripTide evaluates adaptive revisions, while TTDP and TRIP-PAL support formal planning. A plausible contribution is a repair formulation that reduces weather risk while retaining utility and penalizing unnecessary change.

Evidence needed: feasibility, risk reduction, retained utility, preserved stops, route edit distance, added travel, runtime, solver status, and comparison with full replanning and a local heuristic.

### Gap 2: Contrastive Explanations For Route Decisions

CityHood explains place fit, and Chatti et al. organize visual explanation. A plausible interface contribution answers why selected, why skipped, why changed, and what would need to change.

Evidence needed: explanation-fidelity checks and a user study measuring constraint comprehension, decision accuracy, perceived control, workload, revision quality, time, and calibrated reliance.

### Gap 3: Weather-Specific Repair Scope

Weather has spatial extent, time windows, confidence, activity sensitivity, and severity. A plausible modeling contribution connects these properties to local, day-level, or plan-level repair.

Evidence needed: defined scenarios, sensitivity analysis, weather-term ablation, and error analysis when forecasts or thresholds change.

### Gap 4: Agent-Solver Coordination

TRIP-PAL establishes precedent. This remains interesting future work but is less focused and less implementation-ready than the first three gaps.

## 15. Claims The Project Can And Cannot Make

**Supported with caveats:** The repository implements a weather-aware, multi-day itinerary prototype combining personalized POI scoring, constrained route generation, hotels or base cities, alternatives, weather-risk signals, dashboard inspection, and Gurobi-backed or fallback routes.

**Require algorithmic experiments:** Better intent preservation, utility-risk tradeoffs, feasibility, route quality, runtime, or model-scoped optimality.

**Require a user study:** Better understanding, decision accuracy, perceived control, explanation usefulness, revision quality, lower overreliance, or calibrated trust.

**Should not yet be claimed:** Full language-to-symbolic planning, autonomous monitoring, real-time weather response, guaranteed real-world feasibility, generalized faithful explanations, proven intent-preserving repair, or a fully autonomous travel agent.

## 16. Prioritized Near-Term Upgrades

1. **Original versus weather-adjusted comparison:** show changed POIs, geometry, schedule, lodging, travel, utility, risk, preserved stops, and solver status.
2. **Minimal-repair objective and metrics:** penalize unnecessary removal, semantic substitution, spatial movement, and sequence change.
3. **Selected, skipped, and changed reason codes:** generate explanations from actual model and validation state.
4. **Traveler adaptation control:** map preserve, balanced, and optimize-freely settings to documented weights or constraints.
5. **Reproducible evaluation suite:** compare repair, full replanning, and a simple heuristic across feasibility, utility, risk, travel, budget, preservation, and runtime.
6. **Focused user study:** compare a basic route view with an inspectable treatment containing alternatives, differences, reasons, weather evidence, and solver/fallback state.
7. **Agent prototype only after the core study:** if time remains, translate a narrow natural-language request into a reviewable schema without adding monitoring, booking, and autonomous repair at once.

## 17. Recommended Reading Sequence And Final Takeaways

Read Halder et al. for recommendation versus planning; Ruiz-Meza and Montoya-Torres for TTDP; Vu et al. for rich feasibility; TravelEval for complete-plan measurement; TRIP-PAL for the LLM/planner boundary; TripTide for disruption repair; Chatti et al. for explanation design; and CityHood for a concrete travel interface.

The eight papers support six durable conclusions:

1. Personalization and feasibility are different and both necessary.
2. Travel planning is selective routing, not only shortest-path search or POI ranking.
3. Rich constraints require explicit feasibility handling and careful solver-scope claims.
4. LLMs can interpret information but are unreliable standalone constraint solvers.
5. Adaptation should preserve intent and change only as much as required.
6. Explanations must expose actual preferences, constraints, tradeoffs, uncertainty, and route changes, then be evaluated against a user task.

The practical research direction is deliberately narrower than a complete autonomous agent:

> Build and evaluate an inspectable, solver-backed interface for comparing and revising multi-day itinerary adaptations under weather and disruption uncertainty.

This direction is grounded in the literature, close to the implemented system, and capable of producing algorithmic and human-centered evidence without claiming capabilities the repository does not yet possess.

## 18. Professor-Recommended Companion Papers

These four files are a focused companion pass after the eight core papers. Three add unique studies; `2410.16456v1.pdf` is a duplicate arXiv copy of TTG, already represented by the EMNLP Demo version in the evidence bank.

### LLMAP: Language To Multi-Objective Route Search

**Paper:** Yuan, Han, Brinton, and Brunswicker (2025), *LLMAP: LLM-Assisted Multi-Objective Route Planning with User Preferences*.

**Local file:** `reference/2509.12273v1.pdf`

**Why it is worthy:** LLMAP is the closest new reference to a future version of this project. It uses an LLM as a parser for POI types, time limits, preferences, and task dependencies, then delegates route construction to the MSGS graph-search algorithm. It evaluates quality, distance, task completion, opening hours, dependencies, and time constraints across 1,000 prompts and 27 cities.

**Main caution:** Its route guarantees apply to the modeled constraints and search setup. The authors report that MSGS becomes expensive as the number of POI types grows. It does not evaluate multi-day lodging or weather repair.

**Project use:** Cite it as essential for a future language-to-route-solver path and as supporting evidence for separating soft preferences from hard feasibility. Do not claim the current repository implements its parser or algorithm.

### Logic-LM: General Neuro-Symbolic Validation Pattern

**Paper:** Pan, Albalak, Wang, and Wang (2023), *Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning*.

**Local file:** `reference/2305.12295v2.pdf`

**Why it is worthy:** Logic-LM is not a travel paper, but it explains the architecture behind auditable natural-language constraint compilation. An LLM creates a symbolic formulation, a deterministic solver performs the reasoning, and solver errors can trigger self-refinement. This makes the translation boundary visible rather than assuming that LLM output is correct.

**Main caution:** The authors state that applicability is bounded by solver expressiveness and that complex symbolic grammars are difficult to convey through limited demonstrations. Its accuracy gains on logical benchmarks do not imply better itineraries.

**Project use:** Use it as a supporting citation if a future interface converts natural language into Gurobi variables or constraints. The parsed schema, validation errors, revisions, and user confirmation should all remain inspectable.

### TripScore: Fine-Grained Evaluation And Reward

**Paper:** Qu et al. (2025), *TripScore: Benchmarking and Rewarding Real-World Travel Planning with Fine-Grained Evaluation*.

**Local file:** `reference/2510.09011v3.pdf`

**Why it is worthy:** TripScore complements TravelEval by combining format, commonsense, soft-quality, and personal-preference criteria into a unified reward. Its dataset contains 4,870 queries, including 219 real-world free-form requests, and its evaluator reaches 60.75% agreement with travel-expert annotations.

**Main caution:** Agreement is moderate, not definitive. Aggregated rewards and LLM-based criteria can encode weighting or judge bias, so detailed dimensions should remain visible.

**Project use:** Treat it as an essential modern evaluation reference. Build a route scorecard with hard-feasibility gates and separate utility, risk, travel, lodging, and intent-preservation measures before presenting any aggregate score.

### TTG: Duplicate File, Important Existing Paper

**Paper:** Ju et al. (2024), *To the Globe (TTG): Towards Language-Driven Guaranteed Travel Planning*.

**Local file:** `reference/2410.16456v1.pdf`

**Duplicate note:** This is the arXiv version of the paper already stored as `reference/2024.emnlp-demo.25.pdf` and already covered in the deep report. It should be cited once, using the verified EMNLP Demo publication when appropriate.

**Project use:** TTG remains essential evidence that natural-language travel requests can be translated into symbolic form and solved with mixed-integer optimization. Its guarantee still depends on correct translation, supplied data, and the encoded model.
