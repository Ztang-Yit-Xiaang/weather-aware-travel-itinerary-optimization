# Literature Onboarding Guide

This guide is the recommended starting point for a new researcher joining the weather-aware travel itinerary optimization project. It explains the project in plain language, organizes the literature into four research areas, and points to the smaller set of papers that are worth reading first.

After this introduction, read the [integrated core literature review](core_paper_reading_cards.md) for the detailed eight-paper learning sequence. The full paper-by-paper evidence remains in `docs/literature_deep_read_study_report.md`; use that file as a reference bank after you understand the field structure and core papers.

Every paper entry in the evidence bank ends with a **Project Action Takeaway**. Read that compact block first when deciding whether a paper matters: **Main goal** states what the paper tried to accomplish; **Main limitation** separates author-reported constraints from project-team inference; **Publication use** identifies the defensible claim and citation role; **Project-polishing action** turns the paper into a concrete implementation or evaluation task; and **Current project status** prevents planned work from being mistaken for an implemented contribution. These blocks are navigation aids, not substitutes for the method, results, and critical-evaluation notes above them.

## 1. Project Problem In Plain Language

Travel planning looks simple until the plan has to become real. A ranked list of attractions is not enough. A traveler also needs to know whether the locations can actually be visited in the available days, whether the route wastes too much driving time, where to stay overnight, whether outdoor stops are risky under bad weather, and what tradeoffs are being made when one place is selected instead of another.

This project studies a practical version of that problem:

> How can a system build feasible multi-day itineraries while letting a traveler inspect, compare, and revise the tradeoffs behind the recommendation?

The current repository is not just a travel recommender and not just a mathematical optimizer. It is a research prototype that combines both. It scores candidate points of interest (POIs), considers traveler interests, plans across multiple days and base cities, accounts for weather-risk signals, generates route alternatives, and exports static dashboards that show the plan, hotels, routes, source confidence, weather risk, and validation artifacts.

The strongest research opportunity is not to claim that the system has invented a new optimization algorithm. The stronger and more defensible direction is human-centered:

> Weather-aware travel planning as accountable, inspectable, solver-backed itinerary adaptation under uncertainty.

In simpler language: the system should not merely say "this is the best route." It should help the user see why a route was built, what could go wrong, what alternatives exist, what constraints are binding, and what would need to change to include or avoid specific stops.

## 2. Current System Snapshot

Based on the repository files, the current system is a California-focused travel-planning research prototype. It supports multi-day route generation with POI scoring, interest profiles, hotels or base cities, nature regions, weather-risk fields, route alternatives, Gurobi-backed or heuristic route repair, dashboard exports, and validation scripts.

The main implementation evidence is in `README.md`, `docs/nature_aware_model_extension.md`, `configs/nature_trip_config.yaml`, `src/itinerary_system/utility_model.py`, `src/itinerary_system/multi_objective_route.py`, `src/itinerary_system/hierarchical_gurobi.py`, `src/itinerary_system/bandit_candidate_selector.py`, `src/itinerary_system/map_exporter.py`, and `scripts/validate_dashboard_export.py`.

The system accepts trip and planning inputs such as:

- trip length, start city, end city, and scenario
- traveler pace/profile and interest profile
- POI, city, hotel, nature-region, and route candidates
- weather-risk and weather-sensitivity fields
- detour, travel-time, budget, and visit-duration estimates
- social or must-go signals
- curated, cached, or live-enriched data sources
- generated route artifacts used by the dashboard

The optimization and repair logic is careful but should be described precisely. Some route artifacts are Gurobi-backed. Some are greedy or heuristic repair outputs. Browser-side interest controls are preview-only and do not rerun Gurobi. Therefore, project writing should use phrases such as "solver-backed," "Gurobi-backed where available," "heuristic repair," "validated route artifact," and "saved optimized route artifact," rather than broadly saying every displayed route is globally optimal.

The dashboard currently exposes customer, research, and evaluation views. It shows saved route artifacts, preview-only routes, route alternatives, trip-length alternatives, profile/method alternatives, selected and candidate hotels, route playback, active stop details, city details, nature exploration, weather-risk layers, route metrics, debug summaries, source-confidence values, and validation artifacts. It also has basic "why selected" text for displayed stops. More general why-skipped and what-would-change explanations are not yet implemented for every candidate.

Concise implementation understanding:

1. What the current system does: it builds and exports multi-day California travel-route artifacts using candidate POIs, interest-aware scoring, route repair, hotels/base cities, weather-risk fields, and dashboard validation.
2. What inputs it accepts: trip configs, traveler pace/profile, interest profile, start/end gateways, trip length, POI and hotel candidates, nature-region definitions, route context, weather-risk fields, detour/travel-time estimates, budget values, and generated artifacts.
3. What the objective and constraints are: the route layer maximizes interest-aware POI value with diversity and travel penalties while respecting time, cost, maximum-POI, detour, weather-risk, diversity, route-flow, and subtour constraints where the Gurobi model is used. Higher-level planning also allocates days to base cities and handles nature-anchor policies.
4. What the dashboard exposes: route alternatives, customer/research/evaluation modes, weather-risk layers, selected and candidate hotels, playback, city and stop details, nature exploration, metrics, debug summaries, source confidence, saved route labels, and preview-only labels.
5. What is fully or partially implemented: POI scoring, interest-adjusted utility, route alternatives, hotels/base cities, weather-risk scoring/constraints, Gurobi/heuristic repair, dashboard export, validation, source-confidence fields, and basic why-selected text are implemented. Nature-region why-not-selected codes, provenance display, interest previews, and weather-adaptation comparison are partial. General why-skipped, what-would-change, live browser optimization, user logging, calibrated-trust evaluation, and natural-language-to-symbolic planning are planned or unsupported.
6. Strongest publishable contribution: the most defensible contribution is an inspectable, solver-backed decision-support interface for understanding and revising weather-aware itinerary tradeoffs, not a standalone claim of a new superior optimizer.

## 3. Simple End-To-End Project Pipeline

Use this mental model before reading the papers:

```text
Traveler preferences + trip constraints + weather
                      |
                      v
             Candidate POIs and scores
                      |
                      v
        Solver-backed itinerary generation
                      |
                      v
 Feasible routes, lodging choices, and alternatives
                      |
                      v
 Inspectable dashboard with explanations and comparisons
                      |
                      v
        User revises, compares, and accepts a plan
```

The first line is about inputs: who is traveling, how many days they have, what they like, what they can spend, and what context might affect the plan. The second line turns places into scored candidates. The third and fourth lines turn candidates into actual feasible routes with days, lodging, and route alternatives. The fifth line is the human-facing contribution: the user should be able to inspect the route, not simply trust it. The final line is the future mixed-initiative loop: the traveler changes preferences or constraints and the system helps repair or compare the plan.

## 4. Four Literature Areas

### A. Operations Research And Tourist Trip Design

Main question:

> Which locations should be visited, in what order, under time, budget, lodging, opening-hour, and routing constraints?

This area gives the mathematical foundation. The key idea is that travel planning is a selection-and-sequencing problem. The system cannot simply pick the highest-rated attractions, because the traveler has limited time and must move through space. Selecting POIs and ordering them are coupled decisions.

The classic foundation is the Orienteering Problem (OP). In OP, each candidate location has a value or reward. The route has a time or distance budget. The goal is to choose a subset of locations and an order that maximizes collected reward without exceeding the budget. Tourist Trip Design Problem (TTDP) extends this idea to tourism-specific requirements such as opening hours, visit durations, multiple days, hotels, mandatory visits, optional visits, categories, and user preferences.

This area explains why "recommendation" and "planning" are not the same. A recommender might say Yosemite is attractive. A planner must decide whether Yosemite can fit into a seven-day route from San Francisco to Los Angeles with the available lodging, travel time, weather risk, and other desired stops.

Important concepts:

- POI selection: choosing which attractions to include.
- Route sequencing: ordering the chosen attractions.
- Time windows: visiting a location only when it is open or appropriate.
- Mandatory visits: stops that must be included if feasible.
- Optional visits: stops that can be skipped if they do not fit.
- Multi-day planning: assigning visits across days, often with lodging or base-city decisions.
- Exact methods: optimization methods that can prove optimality under a formal model and solver assumptions.
- Heuristics: faster methods that find good routes but do not generally prove global optimality.
- Solver-backed guarantees: guarantees only apply to the variables, constraints, data, and solver status of the specific model, not to every real-world travel uncertainty.

For this project, the OR literature supports the claim that route planning is a constrained optimization problem. It does not by itself solve the human question: can a traveler understand and revise the tradeoffs?

### B. Personalized Travel Recommendation

Main question:

> Which destinations and activities best match a particular traveler?

This area explains how systems estimate what a traveler might like. A POI utility score may combine ratings, popularity, category fit, previous behavior, user profile, social signals, and context. In this repository, POI value is influenced by base utility, interest fit, nature/scenic signals, social or must-go signals, data confidence, weather risk, and detour cost.

Personalized itinerary recommendation sits between recommender systems and route optimization. It asks not only "what does this person like?" but also "how can those liked places become an itinerary?" This matters because a high-scoring list can still be unusable. A traveler might love national parks, museums, and beaches, but a feasible plan must still obey time, budget, geography, lodging, and weather constraints.

Important distinctions:

- Static preferences are stable user interests, such as nature, city, culture, or history.
- Contextual preferences depend on situation, such as weather, season, crowding, queue time, or trip pace.
- Recommendation predicts or ranks attractive items.
- Planning builds a feasible sequence under constraints.
- Personalization improves utility but can conflict with feasibility.

For this project, personalization provides the POI scoring layer. It supports interest profiles and route-mix metrics. But personalization alone does not produce a feasible itinerary, explain skipped locations, or help a user understand why the plan changed under weather risk.

### C. Weather, Disruptions, And Adaptive Replanning

Main question:

> What should change when weather, closures, delays, fatigue, or other disruptions invalidate part of an itinerary?

This area is the most direct bridge to the weather-aware contribution. A route can be feasible when planned but become less desirable or infeasible later. Outdoor stops may become risky under rain, heat, wildfire smoke, snow, or seasonal closure. Transit delays, overbooked attractions, road closures, or user fatigue can also disrupt a plan.

The key idea is that adaptation is not the same as starting over. If the original plan reflected the traveler's intent, a good repair should preserve as much of that intent as possible. A user who asked for a nature-heavy trip should not receive a completely city-heavy plan just because one park became risky. Similarly, a plan should not replace every stop when only one local repair is needed.

Important concepts:

- Disruption severity: how serious the problem is.
- Local repair: changing one stop or one segment.
- Day-level repair: changing one day while preserving the rest of the trip.
- Plan-level repair: restructuring the full itinerary.
- Preservation of intent: keeping the traveler's original goals and important stops when possible.
- Responsiveness: reacting quickly and appropriately to the disruption.
- Semantic change: whether the type of experience changes, such as nature to museum.
- Spatial change: whether replacement stops are far away.
- Sequential change: whether the order of the itinerary changes.
- Traveler tolerance: how flexible the traveler is about risk, distance, schedule, and substitutions.

The current code supports weather-risk scoring and constraints, weather-risk visualization, and route alternatives. It does not yet implement a complete live disruption-response workflow. Therefore, the best next-stage research scope is not "we solve all travel disruptions." It is narrower: compare an original itinerary against a weather-adjusted itinerary, explain what changed, and let the user inspect preservation of intent and tradeoffs.

### D. HCI, Explainability, And Mixed-Initiative Planning

Main question:

> Can travelers understand, compare, challenge, and revise the system's recommendations?

This area is where the project can stand out. A route optimizer can produce a feasible plan, but the user may still not understand it. Why was a scenic stop included? Why was a famous location skipped? What would need to change to include it? Is the system avoiding outdoor places because of weather risk, because of distance, or because the budget is tight? Should the traveler trust the result?

Human-computer interaction (HCI) and explainable recommendation research provide vocabulary and evaluation methods for these questions.

Important concepts:

- Transparency: the user can see what the system considered.
- Scrutability: the user can inspect and correct assumptions.
- Perceived control: the user feels able to influence the plan.
- Trust calibration: the user relies on the system appropriately, not blindly.
- Why-selected explanation: why this stop is in the route.
- Why-skipped explanation: why a candidate was not included.
- What-would-change explanation: what constraint or preference must change to include a skipped item.
- Contrastive explanation: why this option instead of that option.
- Provenance: where data, scores, and route artifacts came from.
- Mixed-initiative planning: the system and user take turns shaping the plan.
- Route comparison: showing alternatives so the user can judge tradeoffs.

This area cautions against measuring trust as simply "higher is better." A user who blindly accepts a bad route is not a success. The goal is calibrated trust: users should accept the system when it is reliable and question it when the evidence is weak, the route is a preview, or the assumptions do not match their goals.

## 5. Important Terminology

POI: Point of interest, such as a park, bridge, museum, scenic viewpoint, restaurant, or landmark.

Utility: A numerical estimate of how valuable a stop is for the trip.

Interest profile: A representation of traveler preferences, such as nature, city, culture, and history weights.

Weather risk: A signal representing how exposed a stop or route is to unfavorable weather. In this project, weather risk is a modeled input and dashboard field, not a production-grade forecast guarantee.

Detour cost: Extra travel time or distance caused by including a stop.

Base city: A city or gateway used as an overnight base for one or more days.

Hotel node: Lodging associated with a base city or route segment.

Route alternative: A saved route artifact generated under a different method, trip length, profile, or context.

Solver-backed: Generated with a formal optimization model and solver where available. This does not always mean globally optimal in the real world.

Heuristic repair: A fast method that repairs or builds a route without proving global optimality.

Preview-only route: A browser-side approximation used for interaction. It is not written back to optimizer artifacts.

Artifact provenance: Metadata showing where a route or score came from, including source confidence, freshness, method, and validation status.

## 6. How The Research Areas Connect

The four areas are not separate silos. They map onto the project pipeline.

Operations Research and TTDP explain how to turn candidate places into feasible routes. Personalized Travel Recommendation explains how to score those candidate places for a particular traveler. Weather and disruption research explains why a route must adapt when context changes. HCI and explainability research explains why the final route must be inspectable and revisable.

The project's research gap appears at the intersection:

- OR gives feasibility but usually little user-facing explanation.
- Recommender systems give personalization but often weak route feasibility.
- Weather-aware systems show context matters but often focus on recommendation rather than multi-day constrained adaptation.
- LLM travel agents offer natural language convenience but often fail constraint validation without symbolic tools.
- HCI and explainable recommender work provide explanation concepts but rarely apply them to solver-backed, weather-aware multi-day itineraries.

That intersection is where this project can make a careful contribution.

## 7. What Previous Work Already Solves

The literature already provides strong foundations:

- OP and TTDP work formalizes travel planning as constrained selection and sequencing.
- Rich TTDP methods show how hard constraints such as time windows, budgets, mandatory visits, categories, and order requirements can be modeled.
- Personalized itinerary recommendation surveys show how POI utility can include user preferences and learned signals.
- Context-aware recommendation shows that time, queues, crowding, and context can change POI value.
- LLM travel-planning benchmarks show that fluent language output is not enough; plans must be checked for constraint compliance, spatial logic, economy, temporality, and utility.
- Hybrid LLM-plus-symbolic planning shows why natural-language systems benefit from formal validation and planning backends.
- Explainable recommender research gives goals such as transparency, scrutability, effectiveness, satisfaction, and trust calibration.
- Visual analytics and mixed-initiative HCI show why users need controls, comparisons, provenance, and feedback loops.

This means the project should not frame common pieces as if they are new. Constrained itinerary planning, POI personalization, and explanation theory already exist. The contribution should come from the combination and evaluation of these ideas in a weather-aware, inspectable route-planning interface.

## 8. What Previous Work Still Does Not Solve

The remaining gap is practical and human-centered.

OR papers often focus on the route model and computational performance. They rarely ask whether a traveler understands why the solver skipped a famous stop or chose a less obvious base city.

Tourism recommender papers often improve POI ranking or personalization. They may not handle hard constraints, hotels, multi-day route structure, or route-level feasibility.

Weather-aware recommenders show that weather matters, but many do not connect weather to solver-backed multi-day route adaptation, route alternatives, and user-facing explanations.

LLM travel agents can produce readable itineraries, but recent benchmarks show that language-only plans can violate constraints or miss important spatial, temporal, lodging, and cost details. Hybrid approaches are more promising, but the user still needs to inspect the plan.

Explainability research provides useful design concepts, but it does not by itself decide how to explain a skipped POI caused by a combination of weather risk, detour time, lodging choice, and day budget.

The unsolved piece is not simply "make the algorithm better." It is:

> Make constrained travel-planning tradeoffs visible, understandable, and revisable when context changes.

## 9. Likely Research Gap For This Project

The most defensible gap is:

> Existing itinerary systems can optimize, recommend, or generate travel plans, but travelers still lack inspectable support for understanding and revising multi-day route tradeoffs under weather and disruption uncertainty.

This gap fits the codebase because the project already has route artifacts, weather-risk fields, route alternatives, hotels/base cities, interest profiles, source confidence, and dashboard validation. It also matches what is missing: generalized contrastive explanations, weather-adjusted route comparison, user revision logging, and a study-ready interface.

The gap should be stated cautiously. The current system does not yet prove that explanations improve user decisions. It does not yet support live browser-side Gurobi replanning. It does not yet implement a full LLM travel agent. Those could be future work, but they should not be claimed as completed contributions.

## 10. Proposed Research Questions

Primary research question:

> How can an inspectable, solver-backed travel-planning interface help users understand, compare, and revise multi-day itinerary tradeoffs under weather and disruption uncertainty?

Secondary questions:

1. How do why-selected, why-skipped, and what-would-change explanations affect users' understanding of route constraints?
2. Can side-by-side original versus weather-adjusted route comparison improve preservation-of-intent judgments?
3. Do route alternatives and solver/fallback labels help users calibrate trust rather than blindly accept a route?
4. Which objective metrics best correspond to users' perceived route quality under weather risk?
5. How much control should users have before the interface becomes confusing or slow?

## 11. Recommended Paper-Reading Order

Do not start by reading all papers. Start with a small spine:

1. Halder et al. (2024), personalized itinerary recommendation survey.
2. Ruiz-Meza and Montoya-Torres (2022), TTDP systematic review.
3. Vu et al. (2022), rich-constraint TTDP branch-and-check.
4. TravelEval (2026), recent LLM travel-planning benchmark.
5. TRIP-PAL (2024), LLM-plus-automated-planning approach.
6. TripTide (2025), adaptive travel planning under disruptions.
7. Chatti et al. (2024), visualization for recommendation explainability.
8. CityHood (2025), explainable travel recommender system.

Then complete the professor-recommended companion pass:

1. LLMAP (2025), language parsing plus multi-objective graph route search.
2. Logic-LM (2023), general LLM-to-symbolic formulation and solver-error refinement.
3. TripScore (2025), fine-grained travel-plan evaluation and unified reward design.
4. TTG (2024), using the existing EMNLP Demo entry; `2410.16456v1.pdf` is a duplicate arXiv copy, not a separate paper.

Then read supporting HCI and explanation papers such as Horvitz on mixed initiative, Amershi et al. on human-AI guidelines, Bucinca et al. on overreliance and cognitive forcing, Tintarev and Masthoff on explanation goals, Knijnenburg et al. on user-experience evaluation, and Chen and Pu on critiquing-based recommenders.

## 12. Concrete Project Roadmap

The next stage should be focused, not expansive.

First, define a study scenario. Use a California multi-day trip with a clear traveler profile, budget, lodging assumptions, and a weather-risk change that affects outdoor stops.

Second, generate two route artifacts: the original itinerary and a weather-adjusted itinerary. Both should have visible route metrics, selected stops, skipped important candidates, hotels/base cities, and solver/fallback status.

Third, add explanations that answer three user questions:

- Why was this stop selected?
- Why was this important candidate skipped?
- What would need to change to include it?

Fourth, expose preservation of intent. The system should show whether the revised route preserved the original trip's nature/city/culture mix, must-go stops, lodging structure, and travel burden.

Fifth, create a study-ready dashboard mode. It should hide research clutter, keep labels truthful, log user actions, and support repeatable tasks.

Sixth, evaluate the system against a simpler baseline.

## Recommended Next Research Stage

Minimum viable implementation:

- A study mode in the dashboard with original itinerary and weather-adjusted itinerary views.
- A route comparison table showing travel time, route distance, budget use, display utility, weather risk, lodging/base city, selected stops, skipped important candidates, and solver/fallback status.
- Why-selected explanations for selected stops.
- Why-skipped explanations for high-value skipped POIs or nature regions.
- What-would-change explanations for including a skipped stop, such as relaxing time budget, increasing detour tolerance, lowering weather-risk avoidance, changing trip length, or changing lodging/base city.
- A preservation-of-intent summary comparing original and adjusted routes.
- Clear labels distinguishing saved optimized artifacts from preview-only routes.

Minimum viable experiment:

- Participants complete a route-selection or route-revision task under a weather-risk scenario.
- They compare an original route and a weather-adjusted route.
- They answer comprehension questions and choose or revise a plan.

Baseline condition:

- A basic itinerary map or ranked-list route display with POI details and limited tradeoff explanation.

Treatment condition:

- The inspectable dashboard with route alternatives, original versus weather-adjusted comparison, why-selected explanations, why-skipped explanations, what-would-change explanations, route metrics, and solver/fallback labels.

Dependent variables:

- Constraint comprehension.
- Decision accuracy on route feasibility questions.
- Perceived control.
- Explanation usefulness.
- Workload.
- Satisfaction.
- Calibrated trust.
- Revision quality.
- Time to make a decision.

Objective route metrics:

- Feasibility status.
- Total travel time.
- Budget use.
- Total utility or display utility.
- Weather-risk exposure.
- Number of preserved original stops.
- Route edit distance.
- Number of high-value skipped candidates.
- Lodging/base-city changes.
- Solver status, fallback status, runtime, and optimality gap where available.

Subjective user-study metrics:

- "I understood why this route was recommended."
- "I understood why important stops were skipped."
- "I knew what to change to get a different route."
- "The system gave me appropriate control."
- "I trusted the route for the right reasons."
- "The explanations were useful for making a decision."
- "The task felt manageable."

Major confounds:

- Participant familiarity with California.
- Individual weather-risk tolerance.
- Preference for driving versus staying in one base city.
- Prior experience with maps and travel planning.
- Differences in outdoor activity interest.
- Confusion caused by too many route alternatives.
- Stale or imperfect source data.

Likely limitations:

- Current artifacts are California-focused.
- Weather risk is modeled, not a production weather forecast.
- Browser preview controls do not rerun the optimizer.
- User-study results may depend on scenario design.
- Solver-backed claims only apply to the implemented model and successful solver status.

Explicitly out of scope for the next stage:

- Nationwide deployment.
- Real-time booking or hotel purchase.
- Full mobile app.
- Fully autonomous LLM travel agent.
- Browser-side Gurobi rerun.
- Production-grade weather forecasting.
- Strong claims that trust, safety, or decision quality improved without a user study.
