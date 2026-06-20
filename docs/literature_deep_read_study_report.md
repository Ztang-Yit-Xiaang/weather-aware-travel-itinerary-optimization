# Weather-Aware Itinerary Optimization Literature Deep Read

> **Reader note:** This is the detailed evidence bank, not the recommended starting point. New readers should begin with [docs/literature_onboarding_guide.md](literature_onboarding_guide.md), then use [docs/core_paper_reading_cards.md](core_paper_reading_cards.md), [docs/project_literature_evidence_matrix.md](project_literature_evidence_matrix.md), and [docs/related_work_outline.md](related_work_outline.md) before returning here for full paper-by-paper notes.
>
> **Caution:** Several recent 2024-2026 PDFs are arXiv-style or demo/preprint papers in the local folder. Unless a venue is explicitly verified in the PDF entry, treat publication status as unverified and cite accordingly.

Prepared: 2026-06-16

Updated with downloaded 2023-2026 papers: 2026-06-17

This report reads the PDFs in `reference/` as one connected research corpus for the project: an inspectable, weather-aware, multi-day itinerary optimization system with Gurobi/heuristic repair, route alternatives, hotels/base cities, interest profiles, weather/detour risk, provenance, and dashboard validation. The original 21-paper corpus provides foundations; the newly downloaded 2023-2026 PDFs add the current frontier around LLM travel agents, solver-backed guarantees, accountable tourism recommendation, explainable travel interfaces, and adaptive replanning under disruptions.

Method note: local PDFs were inspected with `pypdf`, `pdfplumber`, and PyMuPDF-capable extraction. Poppler command-line rendering was not installed, so figure and table review is based on extracted captions, page text, and paper structure rather than rendered page screenshots.

## Executive Synthesis

The strongest publication story is not "a better optimizer" and not "another travel recommender." After adding the 2023-2026 papers, the strongest story is:

> Weather-aware travel planning as accountable, inspectable, solver-backed itinerary adaptation under uncertainty, where users can compare feasible route alternatives, understand weather/detour/budget/lodging tradeoffs, and revise the plan through meaningful controls.

The local papers form eight clusters:

1. **TTDP / orienteering foundations**: OP and TTDP papers show that route planning is a constrained selection-and-sequencing problem, not a ranked-list problem.
2. **Rich constraints and exact methods**: branch-and-check TTDP work shows how hard it is to solve realistic routes with time, budget, mandatory visits, precedence, exclusions, and categories.
3. **Tourism recommenders**: survey papers show how tourism recommendation moved from knowledge-based and context-aware systems toward personalized itinerary recommendation.
4. **Time, queue, and context awareness**: POI and itinerary papers show that value changes with time, waiting, crowding, and operating conditions. Weather is a natural next contextual burden.
5. **Explainable and controllable recommenders**: explanation, critique, mental-model, and UX-evaluation papers provide the human-study vocabulary: transparency, scrutability, perceived control, trust calibration, effort, satisfaction, and behavior.
6. **Mixed-initiative HCI and visual analytics**: HCI papers justify a dashboard where optimization output is not final authority; it is evidence users can inspect, compare, and change.
7. **LLM and neuro-symbolic travel planning**: TravelPlanner, TravelEval, TRIP-PAL, TTG, TravelAgent, and related 2024-2026 papers show that LLMs alone struggle with constraints, while LLM-to-symbolic or solver-backed planning is promising when paired with validation and user-facing inspection.
8. **Accountability, grounding, and disruption adaptation**: TRACE, TripTide, CityHood, and recent explainability surveys show that current travel systems need evidence, rejection recovery, disruption handling, and explanation designs that fit user characteristics.

What the project can borrow immediately:

- From TTDP/OP: formulate the route as a constrained path with selected POIs, day budgets, hotels/base cities, and route feasibility.
- From recommender systems: evaluate not only route utility but user experience, explanation usefulness, control, and satisfaction.
- From visual analytics: make the dashboard a route-evidence workspace with filtering, comparison, provenance, and history.
- From cognitive forcing: require comparison and inspection before accepting an "optimal" route.
- From LLM planning: use natural language as an input layer only if constraints are compiled into symbolic, auditable route constraints.
- From recent benchmarks: evaluate complete plans on compliance, temporality, spatiality, economy, utility, disruption response, and grounding rather than only recommendation accuracy.

Main gaps in prior work:

- OR papers optimize routes but rarely study whether travelers understand or can revise the resulting tradeoffs.
- Recommender papers personalize POIs but often underplay hard feasibility, hotels, multi-day structure, and route explainability.
- HCI/XAI papers study decision support broadly but rarely use route optimization with real spatial, weather, and lodging constraints.
- LLM travel systems target natural-language convenience but still need better constraint validation, uncertainty display, disruption response, grounding, and user control.

## Local Paper Deep Reads

### 1. Vansteenwegen, Souffriau, and Van Oudheusden (2011) - The Orienteering Problem: A Survey

- **PDF**: `1-s2.0-S0377221710002973-main.pdf`
- **Topic**: orienteering problem (OP), team OP, OP with time windows, solution approaches, applications.
- **Main research question**: How has the OP been formulated, extended, solved, and applied up to 2010?
- **Method**: invited literature survey with formal problem definitions, variant taxonomy, benchmark-instance discussion, and comparison of exact and heuristic methods.
- **Conclusion**: OP variants are central to practical routing applications, especially tourism, but new benchmarks and richer variants are needed.
- **One-sentence purpose**: The paper establishes OP as the mathematical foundation for selecting and sequencing valuable stops under a limited travel budget.

**Research problem**

- **Problem solved**: organizing a scattered OP literature into definitions, variants, algorithms, benchmarks, and applications.
- **Importance**: many real systems cannot visit everything; they must choose a subset of valuable stops within time or distance limits.
- **Already known**: OP combines knapsack-like selection with TSP-like sequencing and is NP-hard.
- **New in the paper**: a focused survey of OP, TOP, OPTW, and TOPTW with application and benchmark comparisons.

**Introduction notes**

- Background: OP comes from the sport of orienteering: visit scored checkpoints within a time frame.
- Gap: earlier routing surveys mentioned OP only briefly.
- Objective: define OP and variants, compare methods, and identify research directions.
- Claimed contribution: a dedicated OP survey with exact and heuristic methods and practical applications.

**Necessary background**

- **OP**: choose nodes and order them to maximize collected score under a travel budget.
- **TOP**: multiple routes or agents.
- **OPTW/TOPTW**: time-window-constrained visits.
- Core formulation: maximize `sum(score_i * x_i)` subject to travel budget and path-continuity constraints.

**Method / reproducibility**

- Data/materials: published OP literature and benchmark instances.
- Procedure: taxonomy by variant, method, and benchmark family.
- Assumptions: OP variants can be meaningfully compared by benchmark performance and formulation features.
- Reproducibility: strong as a conceptual map; benchmark comparisons depend on cited papers' settings.

**Figures and tables**

- Tables 1 and 2 summarize available OP and TOP benchmark instances, node counts, path counts, and references. They measure benchmark coverage and show that shared instances exist but may be too small or saturated for modern comparison.

**Discussion / conclusion**

- Interpretation: OP is highly applicable to tourism and logistics.
- Limitations: survey predates many modern personalized, context-aware, and deep-learning itinerary systems.
- Practical meaning: tourism route planning needs very fast, effective algorithms.
- Future work: larger test instances, more realistic constraints, time dependence, compulsory vertices, and better application-specific variants.

**Critical evaluation**

- Method is appropriate for a foundation survey.
- Evidence is sufficient for taxonomy and algorithm orientation.
- It does not address user understanding, explanation, or interaction.
- Conclusions match results: OP is a strong base, but practical tourism needs richer constraints.

**From-memory summary**

- Problem: select a valuable subset of places and visit them within a budget.
- Solution: model as OP and variants.
- Method: survey of formulations, benchmarks, and algorithms.
- Findings: OP is mature but tourism pushes it toward richer, faster variants.
- Limitations: algorithmic, not human-centered.

**Teach-back**

Imagine a tourist has one day and twenty possible stops. They cannot visit all twenty. OP asks: which stops should they choose, and in what order, so the trip gets the most value without running out of time?

**Project relevance**

This is the root citation for why the project is not just ranking attractions. Your route is a constrained selection-and-sequencing artifact, so the dashboard should expose skipped stops, travel budget, and objective tradeoffs.

**Project Action Takeaway**

- **Main goal**: Establish the Orienteering Problem (OP) and its major variants as a formal basis for selecting and sequencing valuable stops under a limited travel budget.
- **Main limitation**: **Project-team inference** - this algorithm-centered survey predates modern personalization, weather context, and user-facing explanation; it cannot establish that travelers understand or prefer an optimized route.
- **Publication use**: **Foundational citation** for the claim that itinerary construction is joint selection and sequencing rather than POI ranking alone. Our paper can extend this foundation with weather-aware comparison and inspectability, but should not cite it as evidence for interface effectiveness.
- **Project-polishing action**: Show the route time budget, collected utility, and important skipped POIs together so the OP tradeoff is visible rather than hidden inside the solver.
- **Current project status**: **partially implemented** - constrained selection and sequencing are implemented, while generalized skipped-POI explanations remain planned.

### 2. Gunawan, Lau, and Vansteenwegen (2016) - Orienteering Problem: A Survey of Recent Variants, Solution Approaches and Applications

- **PDF**: `1-s2.0-S037722171630296X-main.pdf`
- **Topic**: recent OP variants and practical applications after the 2011 survey.
- **Main research question**: What newer OP variants and applications have emerged, and how are they solved?
- **Method**: literature survey of variants including time-dependent, stochastic, generalized, arc, clustered, and multi-agent OP.
- **Conclusion**: OP has expanded into many practical domains, including tourist trip design and mobile crowdsourcing.
- **One-sentence purpose**: The paper updates OP research by mapping newer rich variants that move closer to real-world itinerary planning.

**Research problem**

- **Problem solved**: earlier OP surveys no longer covered newer variants and applications.
- **Importance**: real-world route planning involves uncertainty, clusters, teams, arcs, time dependence, and application-specific rules.
- **Already known**: OP, TOP, OPTW, and TOPTW were established.
- **New in the paper**: systematic coverage of recent variants and applications, especially TTDP.

**Introduction notes**

- Background: OP maximizes collected node score with limited time.
- Gap: newer variants had not been surveyed thoroughly.
- Objective: update the state of OP variants, solution approaches, and applications.
- Claimed contribution: tables and taxonomy of recent variants and practical uses.

**Necessary background**

- **Stochastic OP**: scores or travel conditions are uncertain.
- **Generalized OP**: nodes may belong to clusters or categories.
- **Arc OP**: value can be on edges/routes, not only nodes.
- **Time-dependent OP**: travel time changes by departure time.

**Method / reproducibility**

- Data/materials: research papers up to 2016.
- Procedure: variant-by-variant review.
- Assumptions: variants can be compared through modeling features and algorithmic families.
- Reproducibility: good for conceptual taxonomy; not a computational benchmark paper.

**Figures and tables**

- The main contribution is tabular taxonomy rather than visual results. The tables organize papers by variant, solution method, and application. They measure the breadth of OP extensions and show that real applications repeatedly require richer constraints than classic OP.

**Discussion / conclusion**

- Interpretation: OP is a flexible modeling base for practical planning.
- Limitations: user-facing interfaces and HCI evaluation remain outside scope.
- Practical meaning: the project can cite this paper when adding time dependence, nature regions, hotels, and weather exposure.
- Future work: more realistic applications and variants.

**Critical evaluation**

- Appropriate survey method.
- Evidence supports the claim that OP variants are diverse and application-driven.
- Does not address how a non-expert user understands solver output.
- Good bridge from classic OP to your richer model.

**From-memory summary**

- Problem: OP literature became broader after classic variants.
- Solution: survey new variants and applications.
- Method: taxonomy of recent OP work.
- Findings: TTDP and mobile applications need richer OP models.
- Limitations: no user study or interface design guidance.

**Teach-back**

Classic OP says "pick stops under a time budget." This paper says real life is messier: travel times change, places come in categories, multiple people or routes may be involved, and some value may come from the road itself.

**Project relevance**

Use this to justify weather exposure, scenic route value, nature-region selection, and multi-day variants as legitimate extensions rather than ad hoc features.

**Project Action Takeaway**

- **Main goal**: Update the OP literature by organizing richer variants, solution approaches, and applications that move the model closer to real planning settings.
- **Main limitation**: **Project-team inference** - the survey maps model variants but does not evaluate traveler-facing interaction, explanation, or control.
- **Publication use**: **Foundational supporting citation** for treating time dependence, multiple routes, regions, and contextual constraints as recognized OP extensions. It does not prove that this project's particular combination is novel.
- **Project-polishing action**: Document each added objective term and constraint as a named OP/TTDP extension, then expose its route-level effect in the comparison dashboard.
- **Current project status**: **implemented** - the prototype includes multi-day routes, regions, hotels/base cities, weather risk, alternatives, and solver/fallback paths.

### 3. Ruiz-Meza and Montoya-Torres (2022) - A Systematic Literature Review for the Tourist Trip Design Problem

- **PDF**: `1-s2.0-S2214716022000069-main.pdf`
- **Topic**: TTDP extensions, objectives, techniques, and future research.
- **Main research question**: What has been published on TTDP, how are variants classified, and what future lines remain?
- **Method**: systematic literature review with taxonomy of variants, objectives, solution methods, and formulations.
- **Conclusion**: TTDP is growing and strongly tied to OP, but richer real-world conditions and new solution directions remain open.
- **One-sentence purpose**: The paper maps TTDP as the tourism-specific branch of OP and identifies where realistic itinerary planning is still underdeveloped.

**Research problem**

- **Problem solved**: fragmented TTDP literature with inconsistent formulations and extensions.
- **Importance**: personalized tourism depends on feasible route construction, not only attraction discovery.
- **Already known**: TTDP commonly derives from OP and variants.
- **New in the paper**: a systematic taxonomy spanning problem variants, objectives, methods, and research gaps.

**Introduction notes**

- Background: tourism is economically important and increasingly personalized.
- Gap: existing reviews were outdated or narrower.
- Objective: review TTDP contributions through February 2021.
- Claimed contribution: taxonomy and future research directions.

**Necessary background**

- **TTDP**: plan one or more tourist routes through POIs under budgets, opening hours, travel times, preferences, and other constraints.
- **Single vs. multi-route/day**: one-day tour vs. multi-day itinerary.
- **User-POI interest score**: objective term connecting preferences to selected POIs.
- **Time windows**: POIs can only be visited during open hours.

**Method / reproducibility**

- Data/materials: TTDP papers from early work through 2021.
- Procedure: systematic review and classification.
- Assumptions: paper features can be coded into a shared taxonomy.
- Reproducibility: mostly reproducible if the search protocol and inclusion criteria are followed; extraction noise in the PDF makes some tables dense but the taxonomy is clear.

**Figures and tables**

- Table 2/3 style taxonomy tables measure which extensions and objectives appear across papers. They show heavy emphasis on single-objective OP-style models and less coverage of prior tourist experience, uncertainty, and richer personalization.
- Table 5 defines indexes, parameters, variables, and mathematical formulations for major route problems. It supports the claim that TTDP is mathematically rooted in OP/TSP variants.
- Equations around user-POI interest maximize selected-route value while enforcing time budget and visit constraints. They directly support a utility-based itinerary model.

**Discussion / conclusion**

- Interpretation: TTDP research is mature enough to provide foundations but still limited in realistic and user-centered dimensions.
- Limitations: review is OR-centered and not primarily about interface agency or explainability.
- Practical meaning: your project should use TTDP literature for formulation legitimacy but seek novelty at the HCI/interaction layer.
- Future work: more uncertainty, richer constraints, better data, stronger realism, and new algorithms.

**Critical evaluation**

- Systematic-review method is appropriate.
- Evidence is broad and sufficient for mapping the field.
- Alternative explanation: some "gaps" may exist in adjacent recommender/HCI literature rather than OR venues.
- Conclusions match the taxonomy.

**From-memory summary**

- Problem: TTDP literature is broad and inconsistent.
- Solution: classify extensions, objectives, and methods.
- Method: systematic review.
- Findings: OP dominates; rich constraints and realistic personalization remain incomplete.
- Limitations: not a human-subject or UI paper.

**Teach-back**

This paper is the map of the itinerary-planning math neighborhood. It tells you which route-planning problems people already solved and where realistic tourism still needs work.

**Project relevance**

This should be one of the core citations for your technical background. It helps avoid overclaiming algorithmic novelty and supports the pivot toward "rich, inspectable, human-centered TTDP."

**Project Action Takeaway**

- **Main goal**: Systematically organize Tourist Trip Design Problem (TTDP) extensions, solution techniques, data practices, and unresolved research directions.
- **Main limitation**: **Project-team inference** - its operations-research scope gives limited treatment to explanation, user agency, and interface evaluation; some apparent TTDP gaps may already be studied in adjacent HCI or recommender literature.
- **Publication use**: **Essential citation** for the technical field definition and for showing that many routing constraints are established prior work. Use it to locate the contribution at the intersection of rich TTDP and human-centered inspection, not to claim a new base formulation.
- **Project-polishing action**: Add a formulation-to-interface table mapping each hard constraint and objective term to its dashboard representation and validation output.
- **Current project status**: **implemented** - the repository supports constrained multi-day planning, but the complete human-centered inspection layer is only partially implemented.

### 4. Vu, Kergosien, Mendoza, and Desport (2022) - Branch-and-Check Approaches for the TTDP with Rich Constraints

- **PDF**: `1-s2.0-S0305054821002963-main.pdf`
- **Topic**: exact methods for TTDP with rich practical constraints.
- **Main research question**: Can branch-and-check solve a TTDP variant with budget, time windows, mandatory visits, categories, order rules, and logical constraints?
- **Method**: mathematical formulation plus branch-and-check, preprocessing, valid inequalities, local branching, VNS, and computational experiments on 148 instances.
- **Conclusion**: customized exact methods outperform a baseline solver configuration and solve many rich TTDP instances.
- **One-sentence purpose**: The paper demonstrates that realistic itinerary planning needs decomposition and specialized feasibility checks when constraints become rich.

**Research problem**

- **Problem solved**: direct MILP solving becomes difficult when TTDP includes practical tourism rules.
- **Importance**: real tourists have mandatory stops, categories, budgets, opening hours, precedence, and exclusions.
- **Already known**: OP/TTDP can be modeled mathematically, but rich constraints complicate exact solving.
- **New in the paper**: branch-and-check decomposition tailored to rich TTDP.

**Introduction notes**

- Background: SmartLoire motivates tourist route planning in real city settings.
- Gap: most TTDP studies focus on common constraints, not a broader set of practical rules.
- Objective: solve rich TTDP exactly.
- Claimed contribution: exact branch-and-check methods with strengthening components.

**Necessary background**

- **Branch-and-check**: master problem proposes selected locations; subproblem checks whether a feasible schedule exists.
- **Valid inequalities**: additional constraints that remove impossible or weak regions of the search space.
- **Local branching / VNS**: heuristic support to find good feasible solutions and bounds.
- **Master-subproblem split**: a useful pattern for your project if route candidate selection and schedule feasibility are separated.

**Method / reproducibility**

- Data/materials: 148 generated/benchmark-like instances with different rich constraints.
- Procedure: compare baseline and enhanced branch-and-check variants against solver behavior.
- Assumptions: POI reward sums represent tourist benefit; constraints encode practical preferences.
- Reproducibility: possible if instances and algorithms are available; reproducing exact performance requires solver setup.

**Figures and tables**

- Algorithm listings describe master selection, subproblem feasibility checks, cut generation, and branching decisions. They measure the procedural path by which infeasible candidate selections are excluded.
- Experimental tables compare solution status, bounds, and solved instances across configurations. They show that preprocessing, node/branching strategies, and generated cuts matter.
- Reported conclusion: 123 out of 148 instances solved optimally by proposed methods, supporting effectiveness.

**Discussion / conclusion**

- Interpretation: rich TTDP is solvable but needs customized methods.
- Limitations: no user-facing explanation or traveler study; objective remains reward-sum focused.
- Practical meaning: the project should expose solver/fallback status because rich constraints can fail or require repair.
- Future work: more realistic constraints and scalable decision support.

**Critical evaluation**

- Method is appropriate for an exact OR contribution.
- Evidence is sufficient for algorithmic performance on designed instances.
- Alternative explanation: gains may depend on instance structure and solver tuning.
- Conclusions match results for exact solving, not user usefulness.

**From-memory summary**

- Problem: rich tourism constraints make TTDP hard.
- Solution: split selection from schedule feasibility through branch-and-check.
- Method: exact decomposition plus cuts and heuristics.
- Findings: customized methods solve many instances better than naive solver use.
- Limitations: no user interaction, weather, or explanation layer.

**Teach-back**

The solver first guesses which places to include. Then another checker asks, "Can these places actually be scheduled?" If not, the system adds a rule saying "do not try this impossible combination again."

**Project relevance**

Your project can stand out by turning these hidden feasibility checks into visible explanations: why a stop was skipped, why a route was repaired, and which constraints blocked alternatives.

**Project Action Takeaway**

- **Main goal**: Develop exact branch-and-check and branch-and-solve approaches for a TTDP formulation with rich interdependent constraints.
- **Main limitation**: **Project-team inference** - the experiments establish solver performance on designed instances, not traveler usefulness, explanation quality, or broad generalization beyond the tested formulation.
- **Publication use**: **Essential algorithmic citation** for why rich TTDP constraints require explicit feasibility checking. Our differentiator can be making solver/fallback state and constraint consequences inspectable; this paper cannot support claims about those human outcomes.
- **Project-polishing action**: Add reason codes for infeasibility and repair, including which budget, time, detour, weather, diversity, or lodging constraint blocked a requested stop.
- **Current project status**: **partially implemented** - Gurobi-backed optimization, heuristic fallback, and status labels exist, while generalized constraint-block explanations are planned.

### 5. Halder, Lim, Chan, and Zhang (2024) - A Survey on Personalized Itinerary Recommendation

- **PDF**: `1-s2.0-S1568494623012188-main.pdf`
- **Topic**: personalized itinerary recommendation from optimization to deep learning.
- **Main research question**: How have personalized itinerary recommendation methods evolved across formulations, constraints, data sources, optimization, and deep learning?
- **Method**: survey and taxonomy of POI/itinerary recommendation work.
- **Conclusion**: personalized itinerary recommendation must combine user preferences, route constraints, and modern data-driven methods; future work includes fairness, sentiment, explainability, and group planning.
- **One-sentence purpose**: The paper connects OR-style itinerary planning with recommender-system personalization and newer deep-learning approaches.

**Research problem**

- **Problem solved**: new researchers face inconsistent definitions of itinerary recommendation.
- **Importance**: tourism requires personalized, feasible plans, not generic top POI lists.
- **Already known**: route recommendation involves both selection/ranking and constrained scheduling.
- **New in the paper**: taxonomy from optimization to deep learning and from user satisfaction to provider satisfaction.

**Introduction notes**

- Background: tourism contributes heavily to global economy.
- Gap: literature is fragmented across OR and recommender systems.
- Objective: review formulations, techniques, constraints, features, data, and evaluation.
- Claimed contribution: a modern survey covering optimization and deep learning.

**Necessary background**

- **POI recommendation**: pick relevant places.
- **Itinerary recommendation**: sequence places into a feasible trip.
- **User satisfaction vs. provider satisfaction**: tourist value vs. business/exposure/fairness concerns.
- **Spatio-temporal features**: user behavior changes by location and time.

**Method / reproducibility**

- Data/materials: published itinerary recommendation papers.
- Procedure: categorize methods, data sources, constraints, and evaluation.
- Assumptions: literature can be grouped by objective and method family.
- Reproducibility: good for taxonomy; not computational.

**Figures and tables**

- Table 2 compares deep-learning-based POI/itinerary models by spatio-temporal features, user interest, queue time, multitasking, and technique. It shows that many deep models address personalization but unevenly cover operational constraints.
- Other survey tables summarize problem formulations and datasets; they show that evaluation often focuses on accuracy/recommendation quality rather than user understanding.

**Discussion / conclusion**

- Interpretation: personalized itinerary recommendation is increasingly important and technically diverse.
- Limitations: survey does not deeply solve interface-level agency.
- Practical meaning: your project can be positioned as a hybrid of optimization, personalization, and explanation.
- Future work: explainable recommendation, seasonal fairness, sentiment analysis, group/family planning.

**Critical evaluation**

- Method is appropriate for a broad survey.
- Evidence supports the fragmentation claim.
- The paper is useful for algorithmic positioning but less precise on HCI study design.
- Conclusions match reviewed trends.

**From-memory summary**

- Problem: itinerary recommendation spans OR and RS with inconsistent assumptions.
- Solution: survey and classify the field.
- Method: taxonomy of formulations, methods, constraints, data, and evaluation.
- Findings: personalization is central, but constraints and explainability remain uneven.
- Limitations: little interface evaluation.

**Teach-back**

This paper says travel recommendation has two jobs: recommend places the person will like and arrange those places into a route that actually works.

**Project relevance**

This is the best modern survey for explaining why your system needs both preference controls and route feasibility. It also directly supports explainability as a future direction.

**Project Action Takeaway**

- **Main goal**: Unify personalized itinerary recommendation research across optimization, conventional recommendation, and deep-learning approaches.
- **Main limitation**: **Project-team inference** - the survey provides broad algorithmic coverage but does not specify or validate an interface-level model of agency, comparison, or revision.
- **Publication use**: **Essential citation** for positioning the system between personalization and constrained planning and for motivating explainability as an open direction. It cannot demonstrate that this dashboard's controls improve decisions.
- **Project-polishing action**: Connect every interest control to a visible utility change and route consequence, and clearly distinguish saved solver output from browser-only previews.
- **Current project status**: **partially implemented** - interest profiles and preview controls exist, but browser changes do not rerun the optimizer and explanation coverage is incomplete.

### 6. Borras, Moreno, and Valls (2014) - Intelligent Tourism Recommender Systems: A Survey

- **PDF**: `26959665_2309998766120001701.pdf`
- **Topic**: tourism recommender systems up to 2014.
- **Main research question**: What types of intelligent tourism recommender systems exist, and how do they model tourists, context, groups, and recommended items?
- **Method**: survey of tourism recommender systems, especially AI-related systems from 2008 onward.
- **Conclusion**: tourism recommenders help tourists manage overwhelming travel choices; future systems should better integrate context, group needs, mobile use, and multiple recommendation types.
- **One-sentence purpose**: The paper establishes tourism recommender systems as decision-support tools for personalized travel choices.

**Research problem**

- **Problem solved**: tourists face too many web search results and need personalized support.
- **Importance**: travel decisions involve accommodations, restaurants, museums, events, routes, and real-time context.
- **Already known**: recommender systems personalize items in many domains.
- **New in the paper**: tourism-specific survey and design guidance.

**Introduction notes**

- Background: web tourism information is abundant and overwhelming.
- Gap: tourism has special needs: location, time, weather, group composition, mobility.
- Objective: review tourism recommender approaches.
- Claimed contribution: guidelines and comparison of tourism recommender systems.

**Necessary background**

- **Explicit preferences**: interests entered by the user.
- **Implicit preferences**: inferred from behavior.
- **Context-aware recommendation**: considers location, time, weather, and situational state.
- **Mobile tourism recommender**: supports in-destination decisions.

**Method / reproducibility**

- Data/materials: published tourism recommender systems.
- Procedure: classify by recommendation object, preference acquisition, context, device, group support, and method.
- Assumptions: system features can be compared across papers.
- Reproducibility: moderate; survey categories are interpretive.

**Figures and tables**

- The extracted PDF did not yield clear figure/table captions, but the paper's survey structure functions as comparative tables/categories. The measured dimensions are recommender object, user model, recommendation method, context, and device support.

**Discussion / conclusion**

- Interpretation: tourism recommenders are useful because they reduce choice overload.
- Limitations: older survey, predates LLMs and recent deep itinerary models.
- Practical meaning: weather appears as a context factor, but route-level optimization and inspectability remain weak.
- Future work: richer mobile/context-aware and group-aware recommendation.

**Critical evaluation**

- Appropriate survey method.
- Evidence is broad for tourism recommender framing.
- It is less useful for modern algorithms and CHI-style evaluation.
- Conclusions fit the surveyed systems.

**From-memory summary**

- Problem: tourists have too many options.
- Solution: intelligent recommenders personalize choices using preferences and context.
- Method: survey of tourism recommender systems.
- Findings: context and mobile use matter.
- Limitations: older and less route-optimization focused.

**Teach-back**

This paper is about systems that help travelers decide what to do by filtering huge tourism information into personalized suggestions.

**Project relevance**

Use it for background, then differentiate your project: you do not only suggest places; you build feasible multi-day routes and expose optimization tradeoffs.

**Project Action Takeaway**

- **Main goal**: Survey intelligent tourism recommender systems by recommendation type, knowledge source, context, and system architecture.
- **Main limitation**: **Project-team inference** - the 2014 review predates LLM agents and recent itinerary models and offers limited evidence about solver-backed route feasibility or modern explanation interfaces.
- **Publication use**: **Foundational supporting citation** for tourism recommendation, context awareness, and choice-overload motivation. Use newer sources for current methods and do not use this survey to claim contemporary state of the art.
- **Project-polishing action**: Make the distinction between recommending attractive places and constructing a feasible multi-day plan explicit in both the dashboard terminology and the paper's system diagram.
- **Current project status**: **implemented** - the project combines recommendation signals with route construction, hotels/base cities, alternatives, and weather-aware constraints.

### 7. Yuan, Cong, Ma, Sun, and Magnenat-Thalmann (2013) - Time-Aware Point-of-Interest Recommendation

- **PDF**: `2484028.2484030.pdf`
- **Topic**: POI recommendation with temporal and spatial behavior.
- **Main research question**: How can POI recommendation improve by considering the time of day and geographical behavior?
- **Method**: collaborative recommendation model using temporal influence and spatial influence, evaluated on Foursquare and Gowalla check-ins.
- **Conclusion**: time-aware and spatially aware models outperform baselines, improving POI recommendation accuracy substantially.
- **One-sentence purpose**: The paper shows that POI value is context-dependent, especially by time and location.

**Research problem**

- **Problem solved**: conventional POI recommendation ignores when a user wants to visit.
- **Importance**: people visit different places at different times, such as restaurants at noon and nightlife at night.
- **Already known**: user-based collaborative filtering works for POI recommendation.
- **New in the paper**: formal time-aware POI recommendation using temporal and spatial influence.

**Introduction notes**

- Background: LBSNs provide large-scale check-in data.
- Gap: previous POI recommenders did not model temporal information.
- Objective: recommend unvisited POIs for a user at a specified time.
- Claimed contribution: time-aware problem definition and temporal-spatial model.

**Necessary background**

- **LBSN**: location-based social network with user check-ins.
- **Temporal influence**: user and POI behavior vary by hour/day.
- **Spatial influence**: users tend to visit nearby POIs.
- **Precision/recall at N**: ranking accuracy metrics.

**Method / reproducibility**

- Data/materials: Foursquare and Gowalla check-in datasets.
- Procedure: split data into train/development/test, compare temporal, spatial, and combined recommendation variants.
- Assumptions: check-in frequency reflects preference/attractiveness.
- Reproducibility: likely if datasets and preprocessing are available; old LBSN datasets may be hard to recover exactly.

**Figures and tables**

- Figure 1 shows a check-in example and motivates POI recommendation.
- Figure 2 measures behavior similarity across hours and shows temporal patterns are not uniform.
- Figure 3 measures distance between successive check-ins, supporting spatial influence.
- Figure 4 measures popular POI check-ins over time, showing POI attractiveness changes by hour.
- Tables 1-3 define notation, dataset statistics, and baseline methods.
- Figures 5-10 compare recommendation performance across temporal/spatial methods and times of day. They support the claim that temporal and spatial features improve accuracy.

**Discussion / conclusion**

- Interpretation: POI recommendations should be time-aware and location-aware.
- Limitations: recommends POIs, not full constrained multi-day itineraries; no weather; no user interface study.
- Practical meaning: weather can be argued as another context signal like time.
- Future work: day-of-week/month effects and other temporal contexts.

**Critical evaluation**

- Method fits the problem.
- Evidence is sufficient for ranking accuracy.
- Alternative explanations include dataset-specific check-in biases.
- Conclusions match the results.

**From-memory summary**

- Problem: POI recommenders ignore visit time.
- Solution: add temporal and spatial behavior to collaborative filtering.
- Method: evaluate on check-in datasets.
- Findings: time and geography improve accuracy.
- Limitations: no route feasibility or weather.

**Teach-back**

A coffee shop, museum, and bar are not equally attractive at every hour. This paper teaches the recommender to care about when the visit happens.

**Project relevance**

Use it to justify the broader principle: contextual variables change POI value. Your project extends this from time to weather, detour, seasonality, and outdoor exposure.

**Project Action Takeaway**

- **Main goal**: Improve POI recommendation by learning how user preferences and venue popularity vary across temporal states.
- **Main limitation**: **Project-team inference** - the method predicts individual POI relevance from check-ins rather than producing a feasible multi-day itinerary; dataset bias and the absence of weather or interface evaluation limit transfer to this project.
- **Publication use**: **Supporting citation** for the claim that context changes POI utility. The project can extend the context argument to weather exposure and detour burden, but this paper does not support route-feasibility or HCI claims.
- **Project-polishing action**: Separate context-adjusted POI utility from route feasibility in the data model and dashboard, showing when weather lowers a stop's value versus when a hard constraint excludes it.
- **Current project status**: **partially implemented** - context-adjusted utility and weather risk exist, but learned time-aware recommendation from check-in data does not.

### 8. Lim, Chan, Karunasekera, and Leckie (2017) - Personalized Itinerary Recommendation with Queuing Time Awareness

- **PDF**: `3077136.3080778.pdf`
- **Topic**: queue-aware personalized itinerary recommendation.
- **Main research question**: How can an itinerary recommender account for attraction popularity, user interests, and time-dependent queueing?
- **Method**: PersQ algorithm plus framework extracting popularity, user interests, and queue times from geo-tagged photos; evaluated on five theme parks.
- **Conclusion**: queue-aware itinerary recommendation outperforms baselines across queue, popularity, interest, precision, recall, and F1 metrics.
- **One-sentence purpose**: The paper shows that dynamic burden signals like waiting time should shape itinerary construction.

**Research problem**

- **Problem solved**: itinerary recommenders may choose popular places but ignore time-varying queue cost.
- **Importance**: long queues can consume touring time and reduce trip quality.
- **Already known**: itinerary recommendation must balance popularity, user interests, and time limits.
- **New in the paper**: automatic queue-time derivation and queue-aware routing.

**Introduction notes**

- Background: tourism is popular and itinerary planning is complex.
- Gap: existing itinerary recommenders rarely determine and account for queue times.
- Objective: recommend personalized itineraries that consider popularity, interest, and queueing.
- Claimed contribution: PersQ and photo-based queue/popularity/interest estimation.

**Necessary background**

- **Queue-aware itinerary**: route whose quality depends on when each attraction is visited.
- **Geo-tagged photo sequence**: proxy for attraction visits and timing.
- **MCTS / search**: used to construct itineraries under constraints.
- **Queue metrics**: maximum queue time, queue-to-cost ratio, queue-to-popularity ratio.

**Method / reproducibility**

- Data/materials: Flickr data spanning nine years for five major theme parks.
- Procedure: infer attraction visits, estimate queue distributions, recommend itineraries, compare to baselines.
- Assumptions: photo timestamps and geotags approximate user visits and crowd/queue patterns.
- Reproducibility: possible in principle; photo API availability and preprocessing may limit exact replication.

**Figures and tables**

- Figure 1 compares queue-aware and non-queue itineraries and shows how ignoring queues can waste time.
- Table 1 defines key notation for attractions, visits, times, and itineraries.
- Table 2 describes datasets by theme park, photos, users, attraction visits, and sequences.
- Table 3 compares PersQ with baselines on MQI, QCI, QPI, popularity, interest, recall, precision, and F1. It directly supports the claim that PersQ improves both queue burden and recommendation quality.

**Discussion / conclusion**

- Interpretation: PersQ works because it jointly considers popularity, interests, and queue times.
- Limitations: theme parks are closed-world environments; queue estimates are proxies.
- Practical meaning: your weather-risk model is analogous to queue-time burden.
- Future work named by authors: cold start, social ties, holidays, special events, weather, and uncertainty.

**Critical evaluation**

- Method is appropriate and directly relevant.
- Evidence supports the claim in theme park settings.
- Alternative explanations: photo-taking behavior may be biased by attraction type.
- Conclusions match results but may not generalize to open-road multi-day travel without adaptation.

**From-memory summary**

- Problem: itineraries can look good but waste time in queues.
- Solution: estimate queue time and include it in recommendation.
- Method: Flickr-derived queue/popularity/interest signals and PersQ algorithm.
- Findings: queue-aware routes perform better.
- Limitations: proxy data and theme-park domain.

**Teach-back**

The best route is not just the route with the most popular rides. It is the route that visits good rides when the lines are not too painful.

**Project relevance**

This is one of the closest analogies to weather-aware planning. Replace "queue burden" with "weather exposure and outdoor risk," then expose that burden in the dashboard.

**Project Action Takeaway**

- **Main goal**: Recommend personalized attraction sequences while accounting for time-dependent queue burden as well as popularity and user interests.
- **Main limitation**: **Author-reported limitation** - noisy geotagged photos can distort inferred visit sequences and queue estimates; **project-team inference** - the closed theme-park setting may not generalize to open-road, multi-day travel.
- **Publication use**: **Supporting citation** and close methodological analogy for treating weather exposure as a time-varying burden. It cannot validate the project's weather model or claim equivalence between queues and weather risk.
- **Project-polishing action**: Report weather-risk data quality and sensitivity alongside route utility, and test whether routes remain stable when risk estimates are perturbed.
- **Current project status**: **implemented** - weather-risk scoring and constraints provide the analogous burden, although live queues and uncertainty sensitivity experiments are not implemented.

### 9. Horvitz (1999) - Principles of Mixed-Initiative User Interfaces

- **PDF**: `302979.303030.pdf`
- **Topic**: mixed-initiative interaction between users and intelligent agents.
- **Main research question**: What principles help automated services and direct manipulation interfaces collaborate effectively?
- **Method**: conceptual design principles illustrated through the Lookout scheduling system, decision theory, uncertainty, and expected utility.
- **Conclusion**: effective intelligent interfaces should reason about uncertainty, costs, timing, user control, and when to act or ask.
- **One-sentence purpose**: The paper explains how systems should share initiative between user and automation instead of fully automating decisions.

**Research problem**

- **Problem solved**: interface agents often guess user goals poorly and interrupt or act at bad times.
- **Importance**: automation can help, but inappropriate automation damages user experience and trust.
- **Already known**: direct manipulation and agent automation were seen as competing UI paradigms.
- **New in the paper**: mixed-initiative principles and utility-based timing/action decisions.

**Introduction notes**

- Background: debate between direct manipulation and automated agents.
- Gap: little design work on collaboration between users and intelligent services.
- Objective: define principles for mixed-initiative UI.
- Claimed contribution: principles plus Lookout examples.

**Necessary background**

- **Expected utility**: compare utility of acting vs. not acting under uncertainty.
- **Threshold for action**: automation should act only when probability and benefit justify cost.
- **Dialog option**: system can ask the user when uncertainty is too high.

**Method / reproducibility**

- Data/materials: Lookout scheduling prototype, user timing observations.
- Procedure: derive design principles and model automation decisions with probability/utility.
- Assumptions: user goals and action costs can be estimated.
- Reproducibility: conceptual; Lookout results are illustrative rather than broad empirical evidence.

**Figures and tables**

- Table 1 describes a text-classifier setup for calendar-relevant messages, measuring whether automation should infer scheduling intent.
- Figures 4-6 plot expected utility of action, inaction, and dialog. They show how decision thresholds shift based on cost/benefit.
- Figure 7 models dwell time vs. message length, showing timing matters for interruption or service invocation.

**Discussion / conclusion**

- Interpretation: automation must be valuable, uncertain, and interruptible in the right ways.
- Limitations: old scheduling domain; predates modern AI and map dashboards.
- Practical meaning: your optimizer should propose, explain, and ask, not simply decide.
- Future work: better user models and timing models.

**Critical evaluation**

- Excellent conceptual foundation.
- Evidence is illustrative but not enough for all modern AI contexts.
- Conclusions remain highly relevant.

**From-memory summary**

- Problem: agents can be annoying or wrong.
- Solution: design mixed-initiative systems that know when to act, defer, ask, or expose control.
- Method: principles plus utility models.
- Findings: uncertainty and action cost should shape automation.
- Limitations: conceptual and older.

**Teach-back**

A good assistant should not grab the steering wheel all the time. It should know when to suggest, when to ask, when to wait, and when the user should stay in control.

**Project relevance**

This is a foundation for framing the itinerary dashboard as mixed initiative: the optimizer proposes and repairs, but the traveler inspects, critiques, and accepts.

**Project Action Takeaway**

- **Main goal**: Provide principles for deciding when an intelligent system should act, ask, defer, or return control in mixed-initiative interaction.
- **Main limitation**: **Project-team inference** - the principles are illustrated in older scheduling and assistance domains rather than empirically validated for travel optimization interfaces.
- **Publication use**: **Foundational HCI citation** for keeping the traveler in control of route acceptance and revision. It supports the design rationale, not a claim that the current prototype already achieves effective mixed initiative.
- **Project-polishing action**: Turn route adaptation into a proposal-review-accept loop with explicit user critiques, reversible changes, and a visible explanation of what the system changed.
- **Current project status**: **partially implemented** - users can inspect alternatives and preview preference changes, but durable critiques and solver-backed interactive revision are absent.

### 10. Amershi et al. (2019) - Guidelines for Human-AI Interaction

- **PDF**: `3290605.3300233.pdf`
- **Topic**: design guidelines for AI-infused user interfaces.
- **Main research question**: What general guidelines help designers build better human-AI systems?
- **Method**: distillation of over 150 recommendations, multiple evaluation rounds, user study with 49 design practitioners, and expert validation.
- **Conclusion**: 18 guidelines are broadly relevant across AI product categories and can guide human-centered AI design.
- **One-sentence purpose**: The paper provides a practical checklist for making AI systems understandable, controllable, and recoverable.

**Research problem**

- **Problem solved**: AI-infused systems can be unpredictable, confusing, or disruptive.
- **Importance**: users need to understand what AI can do, when it may be wrong, and how to correct it.
- **Already known**: HCI had older AI design principles.
- **New in the paper**: empirically validated set of 18 modern human-AI guidelines.

**Introduction notes**

- Background: AI capabilities are increasingly embedded in user-facing systems.
- Gap: designers need updated, general guidance.
- Objective: propose and validate guidelines.
- Claimed contribution: guideline set, user study, and expert revisions.

**Necessary background**

- Guidelines include: make clear what the system can do, show how well it can do it, time services based on context, support correction, convey uncertainty, and enable graceful failure.

**Method / reproducibility**

- Data/materials: 20 AI-infused products and 49 design practitioners.
- Procedure: participants applied guidelines to products, rated clarity/relevance, and experts reviewed revisions.
- Assumptions: design practitioners can judge applicability across AI products.
- Reproducibility: possible but product versions change over time.

**Figures and tables**

- Table 1 lists 18 guidelines by interaction phase and examples. It is the core design artifact.
- Table 2 lists product categories and assignment counts.
- Figure 1 counts guideline applications, violations, and non-applicability by product category, showing broad relevance.
- Figure 2 rates guideline clarity.
- Figure 3 and Tables 3-4 show expert preference for revised guideline wording and distinguishability.

**Discussion / conclusion**

- Interpretation: guidelines are useful but not exhaustive.
- Limitations: broad guidance; not itinerary-specific.
- Practical meaning: map project features to guidelines: scope, uncertainty, provenance, correction, fallback, feedback.
- Future work: guideline refinement for emerging AI.

**Critical evaluation**

- Strong method for design guidance.
- Evidence supports general relevance.
- Does not answer which UI design works best for route optimization.
- Conclusions match validation results.

**From-memory summary**

- Problem: AI UI behavior can confuse users.
- Solution: 18 design guidelines.
- Method: literature distillation, practitioner study, expert review.
- Findings: guidelines apply broadly.
- Limitations: broad, not domain-specific.

**Teach-back**

This paper is a checklist for not making AI feel magical in a bad way. Tell users what it can do, what it cannot do, how confident it is, and how they can fix it.

**Project relevance**

Use these guidelines to audit the dashboard. Especially important: distinguish saved optimized routes from preview-only controls, expose uncertainty, and support correction.

**Project Action Takeaway**

- **Main goal**: Synthesize and validate general guidelines for designing interactions with AI-infused systems across the interaction lifecycle.
- **Main limitation**: **Author-reported scope and project-team inference** - the guidelines are intentionally broad and some do not apply uniformly across products; they do not identify the best interface for constrained itinerary planning.
- **Publication use**: **Essential design citation** for communicating system scope, uncertainty, correction, and failure states. It cannot establish that compliance with guidelines improves route decisions without a study.
- **Project-polishing action**: Run a guideline audit covering saved versus preview-only routes, Gurobi versus fallback status, uncertain data, user correction, and recovery from infeasible requests.
- **Current project status**: **partially implemented** - route-state, source-confidence, validation, and fallback signals exist, while correction and feedback loops remain limited.

### 11. Shneiderman (2020) - Human-Centered Artificial Intelligence: Reliable, Safe and Trustworthy

- **PDF**: `Human-Centered Artificial Intelligence  Reliable  Safe   Trustworthy.pdf`
- **Topic**: human-centered AI framework.
- **Main research question**: How can designers achieve high automation while preserving high human control?
- **Method**: conceptual framework critiquing one-dimensional autonomy levels and proposing a two-dimensional human-control/computer-automation model.
- **Conclusion**: reliable, safe, trustworthy AI often requires both high computer automation and high human control.
- **One-sentence purpose**: The paper reframes AI design away from "more autonomy" toward high automation plus strong human agency.

**Research problem**

- **Problem solved**: autonomy is often treated as a one-dimensional ladder where more automation means less human control.
- **Importance**: excessive automation and excessive human control both create failures.
- **Already known**: levels-of-automation frameworks are common.
- **New in the paper**: two-dimensional HCAI framework and RST design emphasis.

**Introduction notes**

- Background: AI design debates often pit human control against automation.
- Gap: that framing limits design imagination.
- Objective: define a framework for reliable, safe, trustworthy HCAI.
- Claimed contribution: high-control/high-automation model.

**Necessary background**

- **HCAI**: human-centered artificial intelligence.
- **RST**: reliable, safe, trustworthy.
- **Two-dimensional model**: human control and computer automation are independent design axes.

**Method / reproducibility**

- Data/materials: conceptual examples including automation, self-driving cars, and control systems.
- Procedure: critique existing frameworks and propose a new model.
- Assumptions: user agency and automation can be designed together.
- Reproducibility: conceptual, not empirical.

**Figures and tables**

- Tables 1-2 summarize one-dimensional automation/autonomy levels and their limitations.
- Figure 1 shows the misleading tradeoff between human control and automation.
- Figure 2 presents the two-dimensional target: high human control and high computer automation.
- Figures 3-6 illustrate rapid-action regions, excessive automation/control risks, and examples such as self-driving cars and pain control.

**Discussion / conclusion**

- Interpretation: better AI systems should preserve user mastery while automating difficult computation.
- Limitations: broad manifesto, not travel-specific.
- Practical meaning: your project fits the high-control/high-automation quadrant.
- Future work: design strategies and governance for RST systems.

**Critical evaluation**

- Strong framing paper.
- Evidence is conceptual but influential.
- Conclusions are normative, not experimentally proven here.

**From-memory summary**

- Problem: people think AI design is a tradeoff between human and machine.
- Solution: design systems with high automation and high user control.
- Method: conceptual HCAI framework.
- Findings: RST systems need both.
- Limitations: broad.

**Teach-back**

The goal is not to make the computer do everything. The goal is to let the computer do hard calculations while the human remains able to understand and direct the decision.

**Project relevance**

This gives the cleanest high-level framing: route optimization is automated, but preference setting, inspection, critique, and acceptance remain human-controlled.

**Project Action Takeaway**

- **Main goal**: Advocate human-centered artificial intelligence that combines high automation with high human control to achieve reliable, safe, and trustworthy systems.
- **Main limitation**: **Project-team inference** - this is a broad normative framework rather than a travel-specific experiment, so it does not prove that any particular dashboard produces reliability or appropriate trust.
- **Publication use**: **Foundational framing citation** for describing the optimizer as decision support under human control. It should not be used as empirical evidence that the prototype is trustworthy.
- **Project-polishing action**: Preserve user authority at the final decision points and expose enough route state, constraints, and provenance for users to challenge automated choices.
- **Current project status**: **partially implemented** - substantial inspection is available, while solver-backed user revisions and outcome evaluation are still missing.

### 12. Bucinca, Malaya, and Gajos (2021) - To Trust or to Think

- **PDF**: `3449287.pdf`
- **Topic**: cognitive forcing functions for reducing overreliance on AI.
- **Main research question**: Can interface interventions reduce users' overreliance on incorrect AI recommendations?
- **Method**: controlled experiment with 199 participants comparing cognitive forcing interventions, simple explainable AI, and no-AI baseline.
- **Conclusion**: cognitive forcing reduces overreliance more than simple explanations, but may lower subjective satisfaction and trust.
- **One-sentence purpose**: The paper shows that explanations alone may not make users think critically about AI output.

**Research problem**

- **Problem solved**: users often accept AI suggestions even when wrong.
- **Importance**: decision support can fail if humans stop evaluating.
- **Already known**: explanations are often proposed as a fix for overreliance.
- **New in the paper**: cognitive forcing functions tested against simple explanations.

**Introduction notes**

- Background: human+AI teams often underperform AI alone because people misuse AI advice.
- Gap: explanations alone may not reduce overreliance.
- Objective: test interventions that elicit analytical thinking.
- Claimed contribution: experimental evidence for cognitive forcing tradeoffs.

**Necessary background**

- **Overreliance**: accepting wrong AI output.
- **Cognitive forcing function**: UI design that compels users to pause and reason.
- **Need for Cognition**: trait measuring motivation for effortful thought.

**Method / reproducibility**

- Data/materials: online decision task, AI recommendations and explanations, participant measures.
- Procedure: compare conditions and measure objective decisions plus subjective ratings.
- Assumptions: task generalizes to AI-assisted decision-making.
- Reproducibility: good in principle if task materials are available.

**Figures and tables**

- Table 1 reports objective measures and shows cognitive forcing improves correct decisions.
- Table 2 focuses on instances where AI is wrong; it measures whether users accept, reject, or correctly override AI.
- Table 3 reports subjective measures, showing lower preference/trust for more demanding designs.
- Tables 4-6 disaggregate performance and subjective effects by Need for Cognition.

**Discussion / conclusion**

- Interpretation: forcing users to think can improve decisions but may feel worse.
- Limitations: domain is not travel; interventions can produce inequality by cognitive style.
- Practical meaning: route dashboards should encourage comparison without becoming annoying.
- Future work: better interventions that balance performance and user experience.

**Critical evaluation**

- Method is strong and relevant.
- Evidence supports tradeoff claims.
- Alternative explanations: task type may amplify overreliance differently than travel.
- Conclusions match results.

**From-memory summary**

- Problem: users blindly accept AI.
- Solution: make them engage before accepting.
- Method: controlled experiment.
- Findings: cognitive forcing reduces overreliance but hurts subjective ratings.
- Limitations: not route planning.

**Teach-back**

Just showing "why the AI thinks this" is not enough. Sometimes the interface must make users compare, question, or commit to their own reasoning first.

**Project relevance**

Add route-comparison tasks and "inspect before accept" moments. For example: before accepting a route, show skipped high-value stops, weather risk, and driving burden.

**Project Action Takeaway**

- **Main goal**: Test whether cognitive forcing interventions reduce overreliance on AI recommendations and improve human-AI decision performance.
- **Main limitation**: **Author-reported limitation** - the studies use one non-critical decision task, and effective forcing functions reduce perceived usability; effects can also differ across cognitive styles.
- **Publication use**: **Essential trust-calibration citation** for arguing that explanation and friction should improve appropriate reliance, not maximize trust. It cannot predict the effect size in travel planning.
- **Project-polishing action**: Add a lightweight compare-before-accept task that asks users to identify a route tradeoff, then evaluate decision accuracy, workload, usability, and calibration together.
- **Current project status**: **planned** - route alternatives exist, but no cognitive-forcing workflow or calibrated-trust study has been implemented.

### 13. Heer and Shneiderman (2012) - Interactive Dynamics for Visual Analysis

- **PDF**: `2133806.2133821.pdf`
- **Topic**: taxonomy of interaction techniques for visual analytics.
- **Main research question**: What interaction dynamics support visual analysis?
- **Method**: conceptual taxonomy organized around data/view specification, view manipulation, and analysis process/provenance.
- **Conclusion**: effective visual analysis systems need flexible interaction for filtering, sorting, selecting, navigating, coordinating views, annotating, and preserving history.
- **One-sentence purpose**: The paper provides a vocabulary for designing a dashboard as an interactive analysis workspace, not a static chart.

**Research problem**

- **Problem solved**: visualization research needed a coherent taxonomy of interaction.
- **Importance**: insight comes from interaction, not only visual encoding.
- **Already known**: many visualization systems had interactive features.
- **New in the paper**: organized taxonomy of interactive dynamics.

**Introduction notes**

- Background: data analysis requires exploration and iterative sensemaking.
- Gap: interaction techniques were scattered.
- Objective: categorize interactive dynamics for visual analysis.
- Claimed contribution: taxonomy and examples.

**Necessary background**

- **Data/view specification**: choose what to show.
- **View manipulation**: select, navigate, coordinate, organize.
- **Process/provenance**: record history, annotate, share.

**Method / reproducibility**

- Data/materials: examples from visualization systems.
- Procedure: classify techniques.
- Assumptions: interaction techniques generalize across domains.
- Reproducibility: conceptual taxonomy.

**Figures and tables**

- Figures 1-2 illustrate data/view specification via drag-and-drop and dynamic query filters.
- Figures 3-7 cover view manipulation: zoomable maps, reorderable matrices, time-series queries, parallel-coordinate selection, and degree-of-interest trees.
- Figures 8-9 show coordinated multiple views and brushing/linking.
- Figures 10-13 cover visual analysis history, collaboration, systematic process, and data storytelling.
- Together they measure no numerical result; they demonstrate interaction design patterns that support analysis.

**Discussion / conclusion**

- Interpretation: visual analytics depends on interactive control and provenance.
- Limitations: no AI/optimization focus.
- Practical meaning: your dashboard should support route filtering, comparison, linked map/details, and history/diff.
- Future work: better interaction systems and research on analysis processes.

**Critical evaluation**

- Appropriate conceptual taxonomy.
- Evidence is example-based.
- Highly useful for design vocabulary.

**From-memory summary**

- Problem: visual analysis needs interaction vocabulary.
- Solution: taxonomy of interactive dynamics.
- Method: synthesis of systems.
- Findings: filtering, selection, coordination, history, and annotation are core.
- Limitations: not itinerary-specific.

**Teach-back**

A useful dashboard is not just a picture. It lets users ask "show me this," "compare that," "what changed," and "how did I get here?"

**Project relevance**

Use this to justify the map dashboard as visual analytics: route layers, filters, route details, provenance, and comparison are research features.

**Project Action Takeaway**

- **Main goal**: Organize the interaction techniques that support iterative visual analysis, including selection, navigation, filtering, coordination, organization, and provenance.
- **Main limitation**: **Project-team inference** - the taxonomy is not centered on AI or route optimization and relies mainly on design examples rather than controlled evidence for this domain.
- **Publication use**: **Essential visual-analytics citation** for route comparison, linked views, and provenance as analytical interactions. It does not show that the current map design improves comprehension.
- **Project-polishing action**: Add linked highlighting among map stops, metric rows, skipped candidates, weather evidence, and a persistent history of route comparisons.
- **Current project status**: **implemented** - route layers, selectors, details, playback, and comparison views exist, while history/diff interaction can be strengthened.

### 14. Endert et al. (2017) - The State of the Art in Integrating Machine Learning into Visual Analytics

- **PDF**: `Computer Graphics Forum - 2017 - Endert - The State of the Art in Integrating Machine Learning into Visual Analytics.pdf`
- **Topic**: integration of machine learning and visual analytics.
- **Main research question**: How can ML and visual analytics be combined so users can make sense of large, complex data?
- **Method**: state-of-the-art survey organized by models, algorithm types, and interaction intents.
- **Conclusion**: ML and visual analytics have complementary strengths; future systems should better support user feedback, interpretability, and trust.
- **One-sentence purpose**: The paper explains how automated models and interactive visualization can work together in sensemaking systems.

**Research problem**

- **Problem solved**: ML and visualization are often integrated awkwardly or opaquely.
- **Importance**: users need powerful automation without losing interpretability.
- **Already known**: VA combines visualization and automated analysis.
- **New in the paper**: state-of-the-art synthesis and categorization of ML/VA integration.

**Introduction notes**

- Background: data scale and complexity require automated methods.
- Gap: combining ML and VA tactfully remains under-explored.
- Objective: summarize progress and opportunities.
- Claimed contribution: synthesis and future research directions.

**Necessary background**

- **Sensemaking loop**: users forage for information and synthesize understanding.
- **Semantic interaction**: user actions in visualization steer underlying models.
- **Interactive ML**: user feedback changes model behavior.

**Method / reproducibility**

- Data/materials: visual analytics and ML literature.
- Procedure: categorize by algorithm type and interaction intent.
- Assumptions: VA systems can be meaningfully compared by how users steer computation.
- Reproducibility: conceptual survey.

**Figures and tables**

- Figure 1 shows Pirolli and Card's sensemaking loop.
- Figures 2-4 show data-frame, knowledge-generation, and visualization-pipeline models.
- Figures 5-6 show semantic interaction and interactive ML feedback loops.
- Table 1 classifies literature by algorithm type and interaction intent.
- Figures 7-14 illustrate clustering, feature relevance, topic modeling, text analytics, user feedback, and automation surprise.

**Discussion / conclusion**

- Interpretation: users should not passively receive model output; they should steer and understand it.
- Limitations: broad VA/ML focus, not optimization route planning.
- Practical meaning: expose utility components, solver status, weather risk, and route alternatives as model state.
- Future work: transparency, user feedback, complex computation, and automation surprise.

**Critical evaluation**

- Strong survey for interface positioning.
- Evidence is broad but qualitative.
- It supports design rationale more than algorithm evaluation.

**From-memory summary**

- Problem: ML can be powerful but opaque.
- Solution: integrate ML with visual analytics and user feedback.
- Method: state-of-the-art survey.
- Findings: users need visible ways to steer models.
- Limitations: not tourism-specific.

**Teach-back**

Instead of showing a model's answer and saying "trust me," visual analytics shows the model's pieces and lets the user interact with them.

**Project relevance**

Your dashboard should make the optimization pipeline inspectable: objective terms, constraints, source quality, alternatives, and fallback behavior.

**Project Action Takeaway**

- **Main goal**: Survey how machine learning and visual analytics can be integrated so users can understand and steer computational models.
- **Main limitation**: **Project-team inference** - its broad and largely qualitative synthesis does not directly address mathematical optimization or evaluate travel-route interfaces.
- **Publication use**: **Supporting HCI citation** for exposing model state and supporting user steering. It offers design rationale, not evidence that the project's inspection controls are effective.
- **Project-polishing action**: Present utility components, hard constraints, data confidence, solver status, and fallback behavior as linked model state rather than isolated debug fields.
- **Current project status**: **partially implemented** - much of the state is exported, but the causal link from user input to solver outcome is not yet fully interactive or explained.

### 15. Sacha et al. (2017) - What You See Is What You Can Change

- **PDF**: `what_you_see_is_what_can_change.pdf`
- **Topic**: human-centered machine learning through interactive visualization.
- **Main research question**: How can interactive visualization let humans inspect and modify ML processes?
- **Method**: conceptual framework and literature synthesis of VA/ML interactions.
- **Conclusion**: visual interfaces are the lens between analysts and ML models; useful systems let users observe, interpret, validate, and refine model behavior.
- **One-sentence purpose**: The paper argues that users can only meaningfully control model behavior when the relevant model state is visible.

**Research problem**

- **Problem solved**: ML systems often hide what users need to inspect or change.
- **Importance**: hidden model assumptions reduce effective human control.
- **Already known**: VA and interactive ML both include feedback loops.
- **New in the paper**: framework connecting ML pipeline stages, visualization, and human-centered interaction.

**Introduction notes**

- Background: VA combines automated data analysis with interactive visualization.
- Gap: need clearer model of how human interactions affect ML components.
- Objective: propose conceptual framework and research challenges.
- Claimed contribution: framework plus examples and five open challenges.

**Necessary background**

- **Execution/evaluation loop**: users act, observe, interpret, and compare results to goals.
- **Model steering**: interactions change data, features, parameters, model, or interpretation.
- **Visual lens**: users see the model through visualization.

**Method / reproducibility**

- Data/materials: literature examples and conceptual models.
- Procedure: synthesize frameworks and examples.
- Assumptions: user interaction can be mapped to ML pipeline components.
- Reproducibility: conceptual.

**Figures and tables**

- Figure 1 shows the VA framework and ML interactions.
- Figure 2 uses Norman's stages of interaction to frame user action and evaluation.
- Figure 3 presents the core VA/ML pipeline and interaction options.
- Figure 4 shows examples of interactive visualization in ML processes.
- Table 1 maps example systems to framework components.
- Figure 5 details the human-centered ML process loop.

**Discussion / conclusion**

- Interpretation: visibility and changeability are linked.
- Limitations: ML-focused, not mathematical optimization.
- Practical meaning: show route utility components if users are expected to change route preferences.
- Future work: better interaction abstractions and support for model refinement.

**Critical evaluation**

- Excellent design principle.
- Evidence is conceptual and example-based.
- Needs adaptation for optimization, where "model state" includes constraints, objective terms, and feasibility.

**From-memory summary**

- Problem: users cannot control hidden models.
- Solution: interactive visualization of model components.
- Method: framework and examples.
- Findings: visible model state enables meaningful change.
- Limitations: not route optimization.

**Teach-back**

If the user cannot see what the system is using to decide, they cannot really change the decision in an informed way.

**Project relevance**

This gives a headline design rule for the dashboard: expose weather risk, detour penalties, interest fit, and skipped regions as handles users can understand and adjust.

**Project Action Takeaway**

- **Main goal**: Argue that users can meaningfully steer a model only when the relevant model state is visible and represented through understandable interaction objects.
- **Main limitation**: **Project-team inference** - the principle is developed for machine-learning visual analytics and is conceptual rather than validated for constrained route optimization.
- **Publication use**: **Supporting design citation** for making route factors both visible and changeable. It cannot support claims about the usability or correctness of the project's controls.
- **Project-polishing action**: Ensure every adjustable preference has a corresponding visible route metric and every displayed constraint exposes whether the user can change it.
- **Current project status**: **partially implemented** - interest and method controls exist, but several visible route factors cannot trigger a saved solver-backed revision.

### 16. Kulesza, Stumpf, Burnett, and Kwan (2012) - Tell Me More?

- **PDF**: `2207676.2207678.pdf`
- **Topic**: mental model soundness and personalization/debugging of intelligent agents.
- **Main research question**: Does helping users build sound mental models improve their ability to personalize an intelligent agent?
- **Method**: empirical study with a music recommender; some users received scaffolding about the system's reasoning.
- **Conclusion**: users can quickly build sound mental models, and improvements in mental-model soundness predict better personalization and satisfaction.
- **One-sentence purpose**: The paper shows that explaining how an intelligent system reasons can help users debug and personalize it.

**Research problem**

- **Problem solved**: users need to correct intelligent agents but may not understand how agent behavior is produced.
- **Importance**: personalized systems learn in use, so end users are often the only people who can fix mismatches.
- **Already known**: interactive machine learning supports user feedback.
- **New in the paper**: empirical link between mental-model soundness and personalization success.

**Introduction notes**

- Background: intelligent agents and recommenders are widespread.
- Gap: users may lack the knowledge to debug them.
- Objective: test feasibility, accuracy, confidence, and user experience effects of mental models.
- Claimed contribution: first empirical exploration of mental models in end-user debugging of an intelligent agent.

**Necessary background**

- **Mental model soundness**: how accurately users understand system reasoning.
- **Debugging an agent**: adjusting system reasoning after it behaves unexpectedly.
- **Scaffolding**: instructional support that helps users build a model.

**Method / reproducibility**

- Data/materials: AuPair music recommender and participant interactions over five days.
- Procedure: compare with-scaffolding vs. without-scaffolding groups.
- Assumptions: music-recommender personalization generalizes to other intelligent agents.
- Reproducibility: possible with system/task recreation.

**Figures and tables**

- Figures 1-2 show debugging actions: explaining why a song was good/bad and adding guidelines.
- Figure 3 shows scaffolding improved mental-model soundness.
- Table 1 defines metrics.
- Table 2 links positive mental-model transformation to benefits, costs, and satisfaction.
- Tables 3 and Figures 5-6 examine self-efficacy and user descriptions.

**Discussion / conclusion**

- Interpretation: mental models matter for effective personalization.
- Limitations: music domain, small/older system, not route optimization.
- Practical meaning: users must understand how interest, weather, and detour weights affect route changes.
- Future work: better explanations for end-user debugging.

**Critical evaluation**

- Method directly measures user understanding and behavior.
- Evidence supports mental-model claims.
- Generalization to travel needs validation.
- Conclusions match results.

**From-memory summary**

- Problem: users cannot personalize systems they do not understand.
- Solution: scaffold mental models.
- Method: user study with music recommender.
- Findings: better mental models predict better debugging and satisfaction.
- Limitations: not travel.

**Teach-back**

If a recommender keeps choosing wrong things, users need to know what knobs matter. Better understanding helps them fix it.

**Project relevance**

Add a study measure for route mental models: can users predict what happens when they increase nature preference or weather sensitivity?

**Project Action Takeaway**

- **Main goal**: Study how explanation completeness and soundness affect users' mental models and their ability to debug a personalized recommender.
- **Main limitation**: **Project-team inference** - the study uses a music recommender and an older, relatively small system; transfer to route constraints and spatial reasoning requires validation.
- **Publication use**: **Supporting explanation-evaluation citation** for measuring whether users can predict and correct system behavior. It does not show that more explanation is always better.
- **Project-polishing action**: Add prediction questions to the study, such as what will happen when weather sensitivity rises or a must-go stop is added, and compare answers with actual route changes.
- **Current project status**: **planned** - explanatory text exists, but route mental models and debugging performance have not been evaluated.

### 17. Tintarev and Masthoff (2007) - A Survey of Explanations in Recommender Systems

- **PDF**: `A_Survey_of_Explanations_in_Recommender_Systems.pdf`
- **Topic**: explanation goals in recommender systems.
- **Main research question**: What purposes can explanations serve in recommender systems, and how can they be evaluated?
- **Method**: survey and taxonomy of explanation aims, presentation types, and interaction modes.
- **Conclusion**: explanation design must be evaluated according to its intended aim, such as transparency, scrutability, trust, effectiveness, persuasiveness, efficiency, or satisfaction.
- **One-sentence purpose**: The paper gives a vocabulary for deciding what route explanations are supposed to achieve.

**Research problem**

- **Problem solved**: explanations are often discussed without clear goals.
- **Importance**: an explanation that persuades may not help users make better decisions.
- **Already known**: expert systems and recommenders used explanations.
- **New in the paper**: seven explanation aims and evaluation distinctions.

**Introduction notes**

- Background: recommender evaluation often focuses on accuracy.
- Gap: user satisfaction and explanation usefulness require broader evaluation.
- Objective: review explanations and evaluation measures.
- Claimed contribution: taxonomy of explanation aims and interaction forms.

**Necessary background**

- **Transparency**: explain how the system works.
- **Scrutability**: let users tell the system it is wrong.
- **Effectiveness**: help users make good decisions.
- **Persuasiveness**: convince users to try/buy.
- **Efficiency**: help users decide faster.

**Method / reproducibility**

- Data/materials: academic and commercial recommender examples.
- Procedure: classify explanation aims, presentation, and interaction.
- Assumptions: explanation goals can be separated analytically.
- Reproducibility: conceptual survey.

**Figures and tables**

- Table 1 defines seven explanation aims.
- Table 2 maps academic systems to aims.
- Tables 3-4 compare commercial and academic systems by item type, presentation, explanation, and interaction.
- Figures 1-3 show examples such as scrutable adaptive hypertext, treemap visualization, and rating influence.

**Discussion / conclusion**

- Interpretation: explanation quality depends on the goal being measured.
- Limitations: old survey; predates modern XAI and LLMs.
- Practical meaning: your dashboard explanations should target transparency, scrutability, effectiveness, and trust calibration, not persuasion.
- Future work: better explanation evaluation.

**Critical evaluation**

- Strong conceptual taxonomy.
- Evidence is example-based.
- Still useful if paired with modern XAI/HCI work.

**From-memory summary**

- Problem: explanations have different purposes.
- Solution: classify aims and evaluation criteria.
- Method: survey.
- Findings: transparency, scrutability, trust, effectiveness, persuasiveness, efficiency, and satisfaction differ.
- Limitations: older.

**Teach-back**

Before adding an explanation, ask: is this meant to help the user understand, correct, trust, decide, decide faster, or feel satisfied?

**Project relevance**

Use this to design explanation panels: "why selected," "why skipped," "what would change," and "what is uncertain" each serve different aims.

**Project Action Takeaway**

- **Main goal**: Classify the purposes of recommender explanations and show that explanation quality must be evaluated against its intended goal.
- **Main limitation**: **Project-team inference** - the survey predates modern explainable AI, LLMs, and interactive map systems, and much of its evidence is example-based.
- **Publication use**: **Foundational explanation citation** for separating transparency, scrutability, effectiveness, and trust-related goals. It cannot demonstrate that the proposed route explanations achieve those goals.
- **Project-polishing action**: Give each explanation panel a declared user task and metric: understand selection, diagnose skipping, identify a feasible change, or judge uncertainty.
- **Current project status**: **partially implemented** - basic why-selected text and nature-region reason codes exist; generalized why-skipped and what-would-change explanations are incomplete.

### 18. Knijnenburg et al. (2012) - Explaining the User Experience of Recommender Systems

- **PDF**: `s11257-011-9118-4.pdf`
- **Topic**: user-centric recommender-system evaluation framework.
- **Main research question**: How should recommender systems be evaluated beyond prediction accuracy?
- **Method**: framework development, literature mapping, and empirical validation using structural equation modeling.
- **Conclusion**: recommender UX depends on objective system aspects, subjective perceptions, experience, behavior, and user/context characteristics.
- **One-sentence purpose**: The paper provides an evaluation model for studying how system design affects recommender user experience.

**Research problem**

- **Problem solved**: accuracy alone does not explain recommender user experience.
- **Importance**: users may prefer, trust, or reject systems for reasons not captured by algorithmic metrics.
- **Already known**: recommender evaluation often uses accuracy metrics.
- **New in the paper**: integrated user-experience framework and empirical validation.

**Introduction notes**

- Background: recommenders help users choose from large catalogs.
- Gap: algorithmic accuracy only partially determines user experience.
- Objective: propose and validate a user-centric evaluation framework.
- Claimed contribution: links objective system aspects, subjective system aspects, experience, interaction, and outcomes.

**Necessary background**

- **Objective system aspects**: manipulated system features.
- **Subjective system aspects**: perceived qualities, such as usefulness or transparency.
- **Experience constructs**: satisfaction, trust, effort, choice difficulty.
- **SEM**: structural equation modeling linking constructs.

**Method / reproducibility**

- Data/materials: recommender studies and empirical validation datasets.
- Procedure: construct framework, map literature, validate via SEM.
- Assumptions: user experience can be modeled through mediating constructs.
- Reproducibility: conceptual plus empirical if datasets and measures are available.

**Figures and tables**

- The paper's key framework diagrams and SEM results connect system aspects to perceptions, experience, and behavior. They measure how subjective perceptions mediate objective design effects.
- Empirical tables/figures validate parts of the framework and show why logged behavior and questionnaires should be combined.

**Discussion / conclusion**

- Interpretation: real-user testing must integrate algorithm, UI, perception, behavior, and context.
- Limitations: not itinerary-specific; framework is broad.
- Practical meaning: your evaluation should measure objective route quality and subjective user agency/trust/control.
- Future work: stronger multi-construct recommender evaluation.

**Critical evaluation**

- Method is very appropriate for user-study planning.
- Evidence supports broader evaluation.
- Framework complexity may be heavy for a class project but useful for publication.

**From-memory summary**

- Problem: accuracy is not enough.
- Solution: user-experience framework.
- Method: theory, literature review, empirical validation.
- Findings: perceptions mediate system effects.
- Limitations: broad.

**Teach-back**

A recommender can be accurate and still feel bad. To evaluate it, measure what users perceive, how they behave, and whether the outcome helps them.

**Project relevance**

Use this as the backbone for a user study: perceived control, explanation usefulness, trust calibration, workload, satisfaction, and final route choice.

**Project Action Takeaway**

- **Main goal**: Provide a framework connecting recommender-system characteristics, user perceptions, interaction behavior, personal characteristics, and situational context.
- **Main limitation**: **Project-team inference** - the framework is broad and measurement-heavy rather than itinerary-specific, so a focused study must select constructs carefully.
- **Publication use**: **Essential evaluation citation** for combining objective system measures with subjective experience and behavioral outcomes. It cannot substitute for collecting project-specific user evidence.
- **Project-polishing action**: Pre-register a compact construct model linking explanation exposure to comprehension, perceived control, calibrated reliance, route revision quality, and final choice.
- **Current project status**: **planned** - the dashboard and route metrics exist, but no controlled user study currently measures these constructs.

### 19. Chen and Pu (2012) - Critiquing-Based Recommenders

- **PDF**: `s11257-011-9108-6.pdf`
- **Topic**: critiquing-based recommender systems.
- **Main research question**: What types of critiquing interfaces and algorithms exist, and how do they help users refine recommendations?
- **Method**: survey with categorization into natural-language, system-suggested, user-initiated, and hybrid critiquing systems.
- **Conclusion**: critiquing supports conversational recommendation and hybrid approaches can combine strengths of different critique styles.
- **One-sentence purpose**: The paper explains how users can iteratively steer recommendations by saying what should be different.

**Research problem**

- **Problem solved**: one-shot recommendations do not let users easily express refinements.
- **Importance**: users often know relative preferences only after seeing an option.
- **Already known**: recommenders can elicit ratings or preferences.
- **New in the paper**: systematic survey of critiquing interfaces and hybrid framework.

**Introduction notes**

- Background: online catalogs are large and recommender systems help search.
- Gap: standard recommendation often lacks conversational refinement.
- Objective: review critiquing designs and algorithms.
- Claimed contribution: categorization and future trends.

**Necessary background**

- **Critique**: feedback such as cheaper, closer, more scenic, less risky.
- **System-suggested critique**: interface proposes useful refinement options.
- **User-initiated critique**: user chooses what attribute to change.
- **Hybrid critiquing**: combines both.

**Method / reproducibility**

- Data/materials: recommender-system literature and example systems.
- Procedure: classify systems and discuss pros/cons.
- Assumptions: critique types generalize across domains.
- Reproducibility: conceptual survey.

**Figures and tables**

- The paper is organized around categories and example systems rather than extracted visual captions. Its major "results" are the taxonomy of natural-language, system-suggested, user-initiated, and hybrid critiques plus empirical findings summarized from prior studies.

**Discussion / conclusion**

- Interpretation: critiquing helps recommendation become interactive.
- Limitations: product/case-based domains more than hard-constrained routing.
- Practical meaning: route controls should be critiques: less driving, more nature, lower weather risk, include/avoid a place.
- Future work: adaptive and hybrid critiquing.

**Critical evaluation**

- Survey method is appropriate.
- Evidence supports critiquing as interaction pattern.
- It does not solve feasibility under hard constraints; your project can add that.

**From-memory summary**

- Problem: users need to refine recommendations.
- Solution: critique-based interaction.
- Method: survey and taxonomy.
- Findings: hybrid critiques can be powerful.
- Limitations: not route optimization.

**Teach-back**

Instead of asking users to specify everything upfront, show a recommendation and let them say "more like this, less like that."

**Project relevance**

This is central to a mixed-initiative route planner. A user should be able to critique: "less driving," "more outdoor stops," "avoid rainy hikes," or "include Yosemite."

**Project Action Takeaway**

- **Main goal**: Review critiquing-based recommenders in which users iteratively state directional preferences to refine recommendations.
- **Main limitation**: **Project-team inference** - most reviewed systems operate in product or case-based domains where revisions do not require solving interdependent route constraints.
- **Publication use**: **Essential mixed-initiative citation** for critique controls such as less driving, more nature, or lower risk. The project's research opportunity is validating critiques through a solver, which this survey does not provide.
- **Project-polishing action**: Implement a small critique vocabulary that maps to auditable objective weights or constraints and reports infeasibility instead of silently ignoring a critique.
- **Current project status**: **partially implemented** - interest previews support limited steering, but general critique-to-solver revision is not implemented.

### 20. TravelPlanner (Xie et al., 2024) - A Benchmark for Real-World Planning with Language Agents

- **PDF**: `2402.01622v4.pdf`
- **Online**: [arXiv:2402.01622](https://arxiv.org/abs/2402.01622)
- **Topic**: benchmark for real-world travel planning by language agents.
- **Main research question**: Can LLM agents handle complex, multi-constraint travel planning with tool use?
- **Method**: benchmark with sandbox tools, nearly four million records, 1,225 planning intents/reference plans, and evaluation of LLM agent strategies.
- **Conclusion**: current language agents struggle severely; even strong models achieve very low final pass rates due to tool-use, constraint tracking, and planning failures.
- **One-sentence purpose**: The paper demonstrates that real travel planning is a hard constraint-satisfaction problem for language agents.

**Research problem**

- **Problem solved**: existing planning benchmarks are too constrained and do not capture real travel complexity.
- **Importance**: travel planning requires collecting information, satisfying hard constraints, and producing coherent plans.
- **Already known**: LLM agents can use tools and reason in simpler tasks.
- **New in the paper**: travel-planning benchmark with tools, data, constraints, and failure analysis.

**Introduction notes**

- Background: planning is a hallmark of intelligence.
- Gap: agent benchmarks often use fixed, simplified action settings.
- Objective: evaluate agents in realistic travel planning.
- Claimed contribution: TravelPlanner benchmark and analysis.

**Necessary background**

- **Language agent**: LLM that reasons and uses tools.
- **Hard constraints**: budget, dates, lodging, transport, etc.
- **Commonsense constraints**: sensible route and meals, plausible daily plan.
- **Pass rate**: whether final plan satisfies constraints.

**Method / reproducibility**

- Data/materials: city, flight, accommodation, restaurant, attraction, and distance data.
- Procedure: test multiple LLMs and strategies; evaluate constraint satisfaction.
- Assumptions: sandbox approximates real planning while staying static.
- Reproducibility: strong if benchmark is released.

**Figures and tables**

- Figure 1 shows workflow: query, tool use, information collection, final plan, and constraint checking.
- Table 1 defines constraint categories.
- Table 2 lists data entries, including millions of flights.
- Table 3 reports main model/strategy results.
- Table 4 reports constraint pass rates.
- Figure 2 shows tool-use error distribution.
- Table 5 compares agent vs. reference tool usage.
- Figure 3 presents failure cases such as wrong dates and hallucinated details.

**Discussion / conclusion**

- Interpretation: LLM agents do not reliably plan under many constraints.
- Limitations: benchmark constraints may differ from open-world tourism planning; not an interface paper.
- Practical meaning: do not rely on LLM-only itinerary planning for your project.
- Future work: better agents, memory, tools, validation, and planning decomposition.

**Critical evaluation**

- Strong benchmark contribution.
- Evidence convincingly shows current agent weakness.
- Could understate hybrid symbolic systems if agents are given stronger planners.
- Conclusions match results.

**From-memory summary**

- Problem: can LLMs plan real travel?
- Solution: build a hard benchmark.
- Method: sandbox plus constraint evaluation.
- Findings: agents mostly fail.
- Limitations: benchmark-specific.

**Teach-back**

LLMs can write nice itineraries, but this benchmark checks whether the plan actually satisfies all constraints. Most do not.

**Project relevance**

This supports your solver-first architecture. LLMs may help collect preferences or explain plans, but feasibility should be validated by structured code and optimization.

**Project Action Takeaway**

- **Main goal**: Benchmark language agents on realistic multi-day travel planning with interdependent constraints, tools, and complete-plan evaluation.
- **Main limitation**: **Project-team inference** - benchmark constraints and datasets simplify parts of open-world tourism and the work does not evaluate user-facing inspection or hybrid solver interfaces.
- **Publication use**: **Essential modern benchmark citation** for the claim that fluent LLM output is not reliable evidence of plan feasibility. It does not compare against this project's optimizer or prove the dashboard is better than an LLM agent.
- **Project-polishing action**: Add automated completeness and constraint checks to every exported route and display validation failures, missing fields, and fallback status next to the plan.
- **Current project status**: **implemented** - structured optimization and dashboard-export validation exist; an LLM planning component is unsupported by current code.

### 21. Ju et al. (2024) - To the Globe (TTG): Towards Language-Driven Guaranteed Travel Planning

- **PDF**: `2024.emnlp-demo.25.pdf`; duplicate local arXiv copy also exists as `2410.16456v1.pdf`
- **Online**: [arXiv:2410.16456](https://arxiv.org/abs/2410.16456)
- **Topic**: LLM-to-symbolic travel planning with MILP guarantees.
- **Main research question**: Can natural-language travel requests be translated into symbolic form and solved with MILP to produce guaranteed itineraries in real time?
- **Method**: fine-tuned LLM translator, synthetic data generation, symbolic travel generator, MILP travel solver, automatic and human evaluation.
- **Conclusion**: hybrid LLM + solver architecture can produce fast, feasible, near-optimal travel itineraries when the NL-to-symbolic translation succeeds.
- **One-sentence purpose**: The paper shows a practical neuro-symbolic path for travel planning: language in, symbolic constraints, solver out.

**Research problem**

- **Problem solved**: pure LLM planning is unreliable and symbolic solvers require structured input.
- **Importance**: users prefer natural language, but travel planning needs guarantees.
- **Already known**: LLMs can parse/generate text; MILP can solve structured optimization.
- **New in the paper**: real-time demo combining fine-tuned NL-to-symbolic translation with MILP travel solving.

**Introduction notes**

- Background: travel planning involves interdependent constraints for flights, hotels, attractions, and budgets.
- Gap: LLMs hallucinate and solvers lack natural-language interfaces.
- Objective: build TTG system with guaranteed travel plans.
- Claimed contribution: synthetic pipeline, translator, solver integration, fast demo, evaluation.

**Necessary background**

- **MILP**: mixed integer linear programming.
- **Backtranslation exact match**: whether symbolic form can recover original request.
- **Constrained decoding**: restricts generated structure to valid JSON/symbolic output.
- **Optimality ratio**: returned itinerary cost/value relative to ground truth optimum.

**Method / reproducibility**

- Data/materials: synthetic user requests, flight/hotel info grounded in real-world statistics.
- Procedure: train translator, solve MILP, return itinerary; evaluate exact match, validity, speed, and users.
- Assumptions: synthetic requests cover target use cases; symbolic schema captures user intent.
- Reproducibility: depends on access to generator, model, and solver.

**Figures and tables**

- Figure 1 shows TTG front-end.
- Table 1 lists factors in request generation.
- Figure 2 shows workflow: NL request to symbolic form to MILP solver to NL itinerary.
- Tables 2-3 report exact match and validity under decoding/constraint settings.
- Figure 3 breaks down translation error sources.
- Table 4 reports time per phase.
- Table 5 reports NPS-like human evaluation.
- Figures 4-5 show itinerary, flight routes, and hotel detail UI.

**Discussion / conclusion**

- Interpretation: symbolic solvers can restore reliability to language-driven travel planning.
- Limitations: generated domain, limited schema, risk of translator errors, less focus on user revision and explanation.
- Practical meaning: your project could add an optional NL preference layer only if it compiles to auditable constraints.
- Future work: broader constraints, better NL translation, richer user-facing interaction.

**Critical evaluation**

- Architecture is highly relevant and technically persuasive.
- Evidence supports feasibility of hybrid planning.
- Alternative explanation: performance may depend heavily on synthetic request distribution.
- Conclusions are mostly supported, but "guarantee" applies after correct symbolic translation.

**From-memory summary**

- Problem: LLM travel plans are unreliable; solvers are not user-friendly.
- Solution: translate language to symbolic constraints and solve with MILP.
- Method: fine-tuned translator, synthetic data, MILP solver, demo evaluation.
- Findings: fast and near-optimal when translation works.
- Limitations: schema and translation dependence.

**Teach-back**

The LLM does not make the final plan by vibes. It converts the user's request into a structured optimization problem, and the solver computes the itinerary.

**Project relevance**

This is the strongest AI-agent bridge for your project. A publishable extension could let users type "more nature, less rain risk, keep driving under 4 hours/day" and compile that into route constraints and weights.

**Project Action Takeaway**

- **Main goal**: Translate natural-language travel requests into a symbolic optimization problem whose resulting itinerary satisfies the encoded constraints.
- **Main limitation**: **Author-reported scope and project-team inference** - the system uses a limited schema and generated evaluation setting; any guarantee holds only after the request has been translated correctly into the symbolic model.
- **Publication use**: **Essential hybrid-planning citation** for separating language understanding from formal feasibility. It cannot support a natural-language contribution in this project because no such translator is currently implemented.
- **Project-polishing action**: Keep natural-language input out of the primary contribution unless a future parser exposes the generated constraints for user review and validates them before solving.
- **Current project status**: **unsupported by current code** - solver-backed planning exists, but natural-language-to-symbolic translation does not.

## Recent Paper Deep Reads: 2023-2026

### 22. Chatti, Guesmi, and Muslim (2024) - Visualization for Recommendation Explainability

- **PDF**: `2305.11755v3.pdf`
- **Topic**: visual explanations in recommender systems.
- **Main research question**: How have visual explanation techniques been used to make recommender systems more transparent, trustworthy, and actionable?
- **Method**: systematic survey organized by explanation aim, scope, method, and format.
- **Conclusion**: visual explanations need clearer theory, better evaluation, and design guidelines that connect visualization form to explanation purpose.
- **One-sentence purpose**: The paper updates older recommender-explanation work by focusing specifically on visual explanation formats and interaction.

**Research problem**: recommender explanations are often textual or ad hoc, while many user decisions benefit from visual comparison and interaction.
**Importance**: your map dashboard is already a visual explanation surface, not just an output viewer.
**Already known**: explanations can support transparency, trust, scrutability, and decision quality.
**New in the paper**: a modern taxonomy of visual explanation aims, methods, and formats.

**Introduction notes**: the gap is not whether explainability matters, but how visual explanation choices should be designed and evaluated.
**Necessary background**: explanation aim, explanation scope, explanation method, explanation format, visual task abstraction.
**Method/data/assumptions/reproducibility**: survey of accessible English papers with inclusion/exclusion criteria; reproducible as a literature review if databases and criteria are reused.
**Important figures/tables**: Tables 2-3 give selection criteria and database search counts; Table 4 classifies aim/scope/method; Tables 5-6 classify visual forms and interactions. These support the claim that visual explainability needs systematic design.
**Discussion/conclusion**: visual explanation research lacks theory-informed design and evaluation. This supports measuring whether route visualizations actually improve understanding.
**Critical evaluation**: strong taxonomy, but not travel-specific; must be translated into map/route explanation design.
**From-memory summary**: the paper says visual explanations need to be designed by purpose, not added as decoration.
**Teach-back**: if a recommender explains with a chart or map, the visual form should match what the user needs to understand or compare.
**Project relevance**: use it to justify route diffs, skipped-stop visuals, weather-risk layers, and comparison panels as explanation mechanisms.

**Project Action Takeaway**

- **Main goal**: Systematically classify how visualization is used to explain recommender systems and identify gaps in theory, design, and evaluation.
- **Main limitation**: **Project-team inference** - the taxonomy is not travel-specific and does not determine which map or route visualizations improve traveler decisions.
- **Publication use**: **Essential visual-explanation citation** for treating maps, route differences, and metric comparisons as explanation mechanisms. It cannot establish explanation effectiveness without a user study.
- **Project-polishing action**: Match each visual form to a task: route overlays for spatial change, aligned metrics for tradeoffs, and highlighted candidates for selection or skipping explanations.
- **Current project status**: **partially implemented** - visual comparison and weather layers exist, while skipped-stop and route-difference explanations need stronger integration.

### 23. Ho, Lee, and Lim (2023) - BTRec: BERT-Based Trajectory Recommendation for Personalized Tours

- **PDF**: `2310.19886v1.pdf`
- **Topic**: BERT-style trajectory recommendation for personalized tours.
- **Main research question**: Can a BERT-based model recommend personalized sequences of POIs using prior visits and demographics?
- **Method**: modifies POIBERT into an iterative trajectory recommender and evaluates recall, F1, and precision against baselines.
- **Conclusion**: BTRec improves trajectory prediction over baseline methods, showing that transformer-style sequence modeling can support personalized tour recommendation.
- **One-sentence purpose**: The paper represents a recent personalization baseline for predicting tourist POI sequences.

**Research problem**: many tour recommenders use popularity and route constraints but miss individual sequence preferences.
**Importance**: modern baselines increasingly use representation learning, not only OP/TSP-style optimization.
**Already known**: POI visits can be modeled as sequences, and BERT-style models can learn contextual sequence representations.
**New in the paper**: demographic information and iterative next-POI generation are incorporated into a personalized tour recommender.

**Introduction notes**: the gap is alignment between itinerary recommendations and individual users, especially in unfamiliar cities.
**Necessary background**: BERT, POIBERT, trajectory sequence, check-in records, recall/F1/precision.
**Method/data/assumptions/reproducibility**: uses user trajectories and demographic/visit histories; assumes historical trajectories are useful proxies for preference.
**Important figures/tables**: Table 1 defines notation and compared methods; Table 2 reports average recall, F1, and precision and shows BTRec outperforming POIBERT and other baselines.
**Discussion/conclusion**: BERT-style models can learn personalized sequence preferences but do not guarantee route feasibility under weather, hotels, or budgets.
**Critical evaluation**: useful as a modern personalization baseline; weaker for hard constraints and explainability.
**From-memory summary**: BTRec predicts what POI a tourist is likely to visit next using a transformer-like trajectory model.
**Teach-back**: it treats a trip like a sentence and POIs like words, then predicts the next good "word" in the itinerary.
**Project relevance**: cite as a recent learned personalization baseline, while positioning your project as feasibility- and explanation-centered.

**Project Action Takeaway**

- **Main goal**: Predict personalized POI trajectories with a BERT-style sequence model that uses visit histories and demographic information.
- **Main limitation**: **Project-team inference** - trajectory prediction does not enforce the project's hard time, lodging, budget, weather, and route-feasibility constraints, and historical trajectories may encode dataset-specific bias.
- **Publication use**: **Supporting recent baseline citation** for learned sequence personalization. Use it to distinguish relevance prediction from feasible planning, not to claim the optimizer outperforms BTRec without a direct experiment.
- **Project-polishing action**: Keep personalized POI utility modular so a learned trajectory score could later be added and evaluated against the current transparent scoring baseline.
- **Current project status**: **unsupported by current code** - the project has interest-based scoring but no BERT trajectory model.

### 24. Ho and Lim (2023) - Utilizing Language Models for Tour Itinerary Recommendation

- **PDF**: `2311.12355v1.pdf`
- **Topic**: language models for tour itinerary recommendation.
- **Main research question**: How can NLP/language-model techniques be adapted to tour itinerary recommendation?
- **Method**: short position/survey paper connecting OR, recommender systems, embeddings, and transformers.
- **Conclusion**: language models can represent POIs and itineraries, but itinerary recommendation still requires both personalized relevance and constraint satisfaction.
- **One-sentence purpose**: The paper bridges older itinerary recommendation and the newer language-model wave.

**Research problem**: tour itinerary recommendation is both an OR problem and an RS problem, but language-model approaches are still emerging.
**Importance**: this helps connect your project to modern AI without claiming an LLM-only route planner.
**Already known**: OR handles constraints; RS handles relevance/ranking.
**New in the paper**: a compact discussion of how POIs and itineraries can be represented as language-model inputs.

**Introduction notes**: the gap is the lack of clear mapping between NLP representations and constrained itinerary planning.
**Necessary background**: embeddings, transformers, POI-as-token, itinerary-as-sequence.
**Method/data/assumptions/reproducibility**: conceptual overview; no major reproducible experiment.
**Important figures/tables**: Figure 1 shows POIs/itineraries represented like words/sentences and passed to language models for downstream recommendation.
**Discussion/conclusion**: language models are promising but need constraint-aware planning.
**Critical evaluation**: useful background, but less substantial than TravelPlanner, TTG, or TRIP-PAL.
**From-memory summary**: it says itinerary recommendation can borrow language-model sequence ideas but cannot ignore OR constraints.
**Teach-back**: language models can help understand trip sequences, but a valid trip still has to obey time and travel limits.
**Project relevance**: use as a minor bridge citation for learned itinerary representations.

**Project Action Takeaway**

- **Main goal**: Explain how language-model representations can be adapted to tour itinerary recommendation while retaining links to operations research and recommender systems.
- **Main limitation**: **Author-reported scope and project-team inference** - this short conceptual paper offers no substantial end-to-end experiment or formal guarantee.
- **Publication use**: **Optional bridge citation** between itinerary recommendation and language-model sequence representations. It is too limited to anchor claims about LLM planning performance or solver integration.
- **Project-polishing action**: Mention learned itinerary representations only as an extension path unless an implemented model and comparative evaluation are added.
- **Current project status**: **unsupported by current code** - no language-model itinerary representation is currently used.

### 25. Zhang et al. (2024) - A Survey of Route Recommendations

- **PDF**: `2403.00284v2.pdf`
- **Topic**: route recommendation methods, applications, datasets, and opportunities.
- **Main research question**: What methods and applications define modern route recommendation in urban computing?
- **Method**: broad survey of search-based, probability-based, machine learning, and deep learning route recommendation.
- **Conclusion**: route recommendation is moving toward multimodal, personalized, and urban-computing settings, but dynamic context and multimodal learning remain underexplored.
- **One-sentence purpose**: The paper gives a recent route-recommendation map beyond tourism-specific TTDP.

**Research problem**: route recommendation spans transportation, safety, traffic, tourism, and urban systems, but surveys often treat only narrow slices.
**Importance**: your project can be positioned as route recommendation under tourism, context, and optimization constraints.
**Already known**: classic methods include Dijkstra, A*, Markov models, matrix factorization, and trajectory mining.
**New in the paper**: updated taxonomy linking classic route algorithms, deep models, datasets, and applications.

**Introduction notes**: urban computing creates large multimodal datasets and new expectations for intelligent route choice.
**Necessary background**: road-network graph, trajectory data, route query, route preference, multimodal learning.
**Method/data/assumptions/reproducibility**: literature survey with dataset and method taxonomies.
**Important figures/tables**: Figure 1 defines route recommendation; Figure 2 abstracts road networks as graphs; Tables 1-3 summarize surveys, datasets, and method classes. These justify route recommendation as a broader research field.
**Discussion/conclusion**: future work includes multimodal learning, cross-domain route recommendation, and better dynamic context.
**Critical evaluation**: broad and useful, but less focused on tourism user experience.
**From-memory summary**: route recommendation is a modern urban-computing field with methods from graph search to deep learning.
**Teach-back**: it is the recent "map of the map-routing literature."
**Project relevance**: cite when framing your work beyond tourism recommender systems and toward context-aware route decision support.

**Project Action Takeaway**

- **Main goal**: Organize modern route-recommendation algorithms, datasets, application domains, and open challenges across urban computing.
- **Main limitation**: **Project-team inference** - its breadth limits depth on tourist decision-making, multi-day lodging, and user-facing route explanation.
- **Publication use**: **Supporting field-map citation** for positioning the project as context-aware route decision support, not only tourism recommendation. It does not establish novelty for the project's particular route model.
- **Project-polishing action**: State clearly which route-recommendation dimensions the prototype covers and which it omits, especially multimodality, live traffic, and learned trajectory models.
- **Current project status**: **implemented** - the project recommends and compares routes, but not all route-recommendation modalities reviewed in the survey.

### 26. de la Rosa et al. (2024) - TRIP-PAL

- **PDF**: `2406.10196v1.pdf`
- **Topic**: combining LLMs with automated planners for guaranteed travel planning.
- **Main research question**: Can LLM travel planning be made feasible and utility-maximizing by pairing LLMs with formal planners?
- **Method**: hybrid architecture where LLMs retrieve/structure travel information and automated planning solves oversubscription planning problems.
- **Conclusion**: TRIP-PAL produces valid and higher-utility plans than GPT-4-only approaches by using formal planning guarantees.
- **One-sentence purpose**: The paper is direct evidence that travel planning should be solver-backed rather than LLM-only.

**Research problem**: LLMs know a lot about destinations but are weak at strict long-horizon constraint satisfaction.
**Importance**: this matches your system philosophy exactly: use AI or heuristics where useful, but validate and optimize structurally.
**Already known**: automated planning can guarantee feasibility if the problem is formalized.
**New in the paper**: a travel-planning workflow that combines LLM destination knowledge with PDDL-style planning.

**Introduction notes**: the gap is between natural-language travel knowledge and formal feasibility.
**Necessary background**: automated planning, PDDL, oversubscription planning, soft goals, utility-maximizing plan.
**Method/data/assumptions/reproducibility**: tests travel scenarios across cities and POI sets; assumes travel information can be encoded into a planner domain.
**Important figures/tables**: Figure 1 shows GPT-4 vs. TRIP-PAL components; Figure 2 gives a PDDL visit action; Figure 3 compares utility; Figure 5 analyzes GPT-4 invalidity. These support the solver-backed claim.
**Discussion/conclusion**: hybrid planning improves validity and utility, but depends on extracting accurate domain facts.
**Critical evaluation**: highly relevant; less focused on user-facing explanations and weather adaptation.
**From-memory summary**: let the LLM gather/describe travel options, then let a planner guarantee a valid route.
**Teach-back**: the LLM helps fill the menu; the planner decides what can actually fit on the plate.
**Project relevance**: one of the best modern citations for your solver-backed route architecture.

**Project Action Takeaway**

- **Main goal**: Combine LLM-based travel knowledge with automated planning so generated itineraries are valid within a formal domain and maximize encoded utility.
- **Main limitation**: **Author-reported limitation** - evaluation is limited to day planning with time discretized into 15-minute segments; **project-team inference** - success also depends on accurate domain extraction and does not address user-facing weather adaptation.
- **Publication use**: **Essential hybrid-planning citation** for solver-backed feasibility within an encoded model. It supports the architecture choice but does not prove this project's multi-day routes are globally optimal or its explanations are useful.
- **Project-polishing action**: Display the exact model scope behind every solver-status claim, including discretization, input provenance, omitted constraints, and fallback use.
- **Current project status**: **partially implemented** - solver-backed and fallback planning exist, while the LLM knowledge-to-planner layer does not.

### 27. Markchom, Liang, and Ferryman (2024) - Review of Explainable Graph-Based Recommender Systems

- **PDF**: `2408.00166v2.pdf`
- **Topic**: explainable graph-based recommender systems.
- **Main research question**: How do graph-based recommenders learn, explain, and evaluate recommendations?
- **Method**: review categorized by learning methods, explaining methods, explanation types, datasets, and metrics.
- **Conclusion**: graph-based explainable recommenders need better user-centered evaluation and clearer explanation quality measures.
- **One-sentence purpose**: The paper is useful if your route graph, POI graph, or constraint graph becomes part of explanation generation.

**Research problem**: graph recommenders can be accurate but opaque.
**Importance**: itinerary systems naturally contain graphs: routes, POIs, hotels, regions, constraints, and alternatives.
**Already known**: graph embeddings, paths, and knowledge graphs can support recommendation.
**New in the paper**: updated taxonomy of graph-based explanation methods and metrics.

**Introduction notes**: the gap is explainability for graph-based RS, not just accuracy.
**Necessary background**: embedding-based methods, path-based methods, hybrid graph recommenders, explanation precision/recall.
**Method/data/assumptions/reproducibility**: literature review of datasets, qualitative and quantitative evaluation methods.
**Important figures/tables**: Figure 2 shows categorization; Table 1 organizes learning/explaining/explanation types; Tables 3-4 summarize qualitative and quantitative evaluation methods.
**Discussion/conclusion**: visualization and user surveys remain important but resource-intensive.
**Critical evaluation**: relevant only if the project leans into route/POI graph explanations.
**From-memory summary**: graph recommenders can explain by showing paths, structures, or learned graph signals.
**Teach-back**: if a system recommends a place because it is connected to your interests through a graph, it should show that path.
**Project relevance**: optional citation for future graph-based "why this stop" explanation.

**Project Action Takeaway**

- **Main goal**: Review graph-based recommender architectures, explanation methods, explanation types, datasets, and evaluation practices.
- **Main limitation**: **Project-team inference** - the review is relevant only if route, POI, hotel, and preference relationships are explicitly represented and evaluated as an explanation graph.
- **Publication use**: **Optional citation** for a future graph-based explanation design. It should not be used to describe the current rule- and score-based explanation strings as graph explanations.
- **Project-polishing action**: Defer graph explanations unless they materially improve provenance paths such as preference to utility component to selected stop to route consequence.
- **Current project status**: **unsupported by current code** - no graph-based recommender or explanation model is implemented.

### 28. Chen et al. (2024) - Can We Rely on LLM Agents to Draft Long-Horizon Plans?

- **PDF**: `2408.06318v1.pdf`
- **Topic**: LLM-agent behavior on TravelPlanner long-horizon tasks.
- **Main research question**: How do context, shots, feedback, and refinement affect LLM agents on long-horizon travel planning?
- **Method**: experiments using multiple agents, feedback generators, refinement, and fine-tuning variants on TravelPlanner.
- **Conclusion**: context, feedback, and refinement can help but do not solve long-horizon planning reliability; feedback itself can be erroneous.
- **One-sentence purpose**: The paper gives failure-analysis evidence for why travel-agent repair loops need validation.

**Research problem**: LLM-agent planning papers often report end results without explaining why agents fail or how refinement behaves.
**Importance**: your project's validation and route-repair artifacts can be framed as necessary safeguards.
**Already known**: TravelPlanner is difficult for LLM agents.
**New in the paper**: focused analysis of context, shots, feedback, and fine-tuning effects.

**Introduction notes**: the gap is understanding LLM-agent failure in demanding real-world planning.
**Necessary background**: planner agent, refiner agent, feedback generator, final pass rate, hard/commonsense constraints.
**Method/data/assumptions/reproducibility**: TravelPlanner-based experiments with different prompting/refinement/fine-tuning settings.
**Important figures/tables**: Figure 1 shows the multi-agent workflow; Tables 1-3 report planner/refiner, feedback-generator, and fine-tuning performance.
**Discussion/conclusion**: automated feedback helps inconsistently; future work should improve annotated feedback and post-training.
**Critical evaluation**: strong for LLM failure diagnosis, but still benchmark-specific.
**From-memory summary**: LLM agents can revise plans, but their feedback and revisions can still be wrong.
**Teach-back**: asking another LLM to fix the first LLM is not enough unless something checks the result.
**Project relevance**: supports explicit validation scripts and solver status in your dashboard.

**Project Action Takeaway**

- **Main goal**: Diagnose how prompting context, examples, feedback, and iterative refinement affect LLM agents on long-horizon travel plans.
- **Main limitation**: **Project-team inference** - findings are benchmark- and agent-specific and do not establish the performance of symbolic optimizers or user-facing validation interfaces.
- **Publication use**: **Supporting benchmark citation** for the claim that self-refinement and LLM feedback can remain unreliable without an external checker. It does not directly validate this project's checks.
- **Project-polishing action**: Treat validation as a first-class artifact by retaining failed checks, repair reasons, solver/fallback state, and the final accepted route version.
- **Current project status**: **partially implemented** - export validation and solver/fallback labels exist, but complete repair provenance is not yet exposed.

### 29. Chen et al. (2024) - TravelAgent

- **PDF**: `2409.08069v1.pdf`
- **Topic**: LLM assistant for personalized travel planning.
- **Main research question**: Can an LLM-based agent framework with tools, memory, recommendation, and route planning generate personalized itineraries under multidimensional constraints?
- **Method**: system architecture with memory, recommendation tools, spatiotemporal-aware algorithms, route planner, and human evaluation.
- **Conclusion**: TravelAgent improves human-rated rationality, comprehensiveness, and personalization compared with a GPT-4+ agent baseline.
- **One-sentence purpose**: The paper shows where current LLM travel-agent systems are heading and what they still leave open.

**Research problem**: real-world personalized travel planning needs dynamic data, preferences, constraints, and route structure.
**Importance**: it is a modern system comparison for your dashboard, but less rigorous on optimization guarantees.
**Already known**: LLM agents can use tools and memory.
**New in the paper**: a tailored travel-agent framework with recommendation and route-planning modules.

**Introduction notes**: the gap is robust personalization in dynamic travel scenarios.
**Necessary background**: memory module, soft/hard constraints, attraction recommendation, route planner, human evaluation.
**Method/data/assumptions/reproducibility**: case studies, simulated users, tool calls, and human scores; exact reproducibility depends on system/tool access.
**Important figures/tables**: Figures 1-4 show constraint modeling, hotel tools, recommendation flow, and route planner; Figure 5 reports prediction error trends; Table 1 reports human evaluation.
**Discussion/conclusion**: TravelAgent provides a practical LLM travel pipeline but relies on generated recommendations and online tools.
**Critical evaluation**: good system reference; weaker evidence than benchmark papers for feasibility.
**From-memory summary**: TravelAgent wraps LLMs with memory, tools, recommendations, and planning modules for personalized trips.
**Teach-back**: it is an AI travel assistant that tries to remember preferences, search options, and assemble a trip.
**Project relevance**: compare against it by emphasizing solver-backed feasibility, weather risk, and inspectable route alternatives.

**Project Action Takeaway**

- **Main goal**: Build an LLM travel assistant that combines preference memory, tools, recommendation, and route-planning modules for personalized itineraries.
- **Main limitation**: **Project-team inference** - the system evidence is weaker than dedicated benchmarks for hard-constraint feasibility, and formal optimality or route-validity guarantees are not its central contribution.
- **Publication use**: **Supporting system citation** for the breadth of agentic travel assistance. Use it as a contrast case for validated optimization and inspectable alternatives, not as a directly tested baseline unless its workflow is reproduced.
- **Project-polishing action**: Make the prototype's narrower scope and stronger validation visible: structured inputs, explicit constraints, route metrics, weather risk, and solver/fallback labels.
- **Current project status**: **unsupported by current code** - the repository does not implement an LLM assistant, memory, or autonomous tool-use workflow.

### 30. Wardatzky et al. (2025) - Whom do Explanations Serve?

- **PDF**: `2412.14193v2.pdf`
- **Topic**: user characteristics in explainable recommender-system evaluation.
- **Main research question**: Which user characteristics are recorded or analyzed in evaluations of recommender explanations?
- **Method**: systematic literature survey of explainable recommender evaluations.
- **Conclusion**: many studies under-report or under-analyze user characteristics; explanation effects may depend on who the user is.
- **One-sentence purpose**: The paper warns that explanation evaluations should account for user differences.

**Research problem**: explanation studies often treat users as interchangeable.
**Importance**: traveler expertise, planning style, risk tolerance, and technical background can affect route-explanation usefulness.
**Already known**: explanations can affect trust, transparency, and satisfaction.
**New in the paper**: systematic focus on participant/user characteristics in XRS evaluation.

**Introduction notes**: the gap is not enough attention to whom explanations help.
**Necessary background**: user characteristics, explanation effects, participant reporting, crowdsourcing bias.
**Method/data/assumptions/reproducibility**: large database search narrowed to a final corpus; records which characteristics were reported/analyzed.
**Important figures/tables**: Tables 1-2 document search and inclusion/exclusion; Table 3 records user characteristics; later tables analyze evidence sparsity and mixed effects.
**Discussion/conclusion**: participant pools skew toward certain locations and education levels; evidence about characteristic effects is sparse.
**Critical evaluation**: highly useful for study design, less directly tied to route algorithms.
**From-memory summary**: explanation studies need to say who was tested, because explanation usefulness varies by user.
**Teach-back**: a route explanation that helps an expert planner may not help a casual traveler.
**Project relevance**: include traveler type, planning confidence, weather-risk tolerance, and map familiarity in study measures.

**Project Action Takeaway**

- **Main goal**: Examine which user characteristics explanation studies report and test, and for whom recommender explanations are actually evaluated.
- **Main limitation**: **Project-team inference** - the review informs study design but does not identify a universally effective travel explanation or route algorithm.
- **Publication use**: **Supporting user-study citation** for measuring heterogeneity in explanation usefulness. It cannot support claims about this project's users until those characteristics are collected and analyzed.
- **Project-polishing action**: Record planning experience, map familiarity, optimization familiarity, weather-risk tolerance, and confidence, then test whether treatment effects differ across these groups.
- **Current project status**: **planned** - no user-characteristic study data are currently collected.

### 31. Espera et al. (2025) - +Tour

- **PDF**: `2502.17345v2.pdf`
- **Topic**: personalized tourist itinerary recommendation for smart tourism and edge/MEC services.
- **Main research question**: How can itinerary recommendation account for user preferences, POI popularity, time constraints, and ICT application/resource constraints?
- **Method**: system model, optimization formulation, solution approach, and experiments on real-world POI visitation data across multiple cities.
- **Conclusion**: next-generation tourism recommendation can jointly consider itinerary quality and network/application service requirements.
- **One-sentence purpose**: The paper is a recent example of richer multi-objective itinerary recommendation beyond simple POI ranking.

**Research problem**: smart tourism may require both good routes and adequate digital service quality at POIs.
**Importance**: it shows itinerary recommendation is expanding toward multi-objective infrastructure-aware planning.
**Already known**: TTDP considers popularity, time, and user interest.
**New in the paper**: combines personalized tour recommendation with MEC/application-mode considerations.

**Introduction notes**: the gap is that future immersive tourism services need route planning plus service/resource planning.
**Necessary background**: PTIR, MEC, application mode, POI visiting time, category preference.
**Method/data/assumptions/reproducibility**: uses POI visitation histories and a mathematical system model; reproducibility depends on dataset availability.
**Important figures/tables**: Figure 2 classifies tour research; Figure 3 shows the system model; Table 1 compares related PTIR work; Table 2 defines notation; Figures 6-8 analyze visit duration and POI popularity.
**Discussion/conclusion**: the paper supports richer route objectives but is not focused on weather or human explanation.
**Critical evaluation**: useful recent optimization-adjacent citation; less central for CHI/HCI.
**From-memory summary**: +Tour plans personalized itineraries while considering digital services tourists may use at POIs.
**Teach-back**: it asks not only "where should you go?" but also "can the smart tourism service support what you will do there?"
**Project relevance**: optional support for multi-objective itinerary planning and future service-aware constraints.

**Project Action Takeaway**

- **Main goal**: Jointly recommend personalized tourist itineraries and account for computing or communication resources needed by smart-tourism services.
- **Main limitation**: **Author-reported future scope and project-team inference** - joint graph-neural itinerary and resource allocation remains future work, and the resource model is peripheral to the current weather-aware decision-support question.
- **Publication use**: **Optional citation** for rich multi-objective itinerary constraints. It should not expand the paper's central framing unless service capacity becomes an implemented and evaluated constraint.
- **Project-polishing action**: Keep service-resource constraints out of scope and use the paper only to illustrate how TTDP objectives can expand beyond attraction utility and travel time.
- **Current project status**: **unsupported by current code** - no mobile-edge or digital-service resource allocation is modeled.

### 32. Banerjee et al. (2025) - SynthTRIPs

- **PDF**: `2504.09277v1.pdf`
- **Topic**: synthetic query generation for personalized tourism recommender benchmarks.
- **Main research question**: How can realistic, grounded, personalized, and sustainability-aware tourism queries be generated for evaluation?
- **Method**: LLM-driven query generation grounded in a curated knowledge base, personas, filters, and human/LLM validation.
- **Conclusion**: SynthTRIPs generates diverse and realistic tourism queries with better personalization, sustainability, and groundedness controls.
- **One-sentence purpose**: The paper helps build study scenarios and benchmark prompts for personalized tourism systems.

**Research problem**: public tourism datasets often lack enough rich, personalized, context-heavy queries.
**Importance**: your study needs realistic route-planning tasks, personas, and constraints.
**Already known**: tourism recommenders need preferences and context, but datasets are thin.
**New in the paper**: grounded synthetic query generation with personas and sustainability filters.

**Introduction notes**: the gap is benchmark breadth/depth for advanced personalization.
**Necessary background**: persona-based query, sustainability filter, groundedness, synthetic benchmark generation.
**Method/data/assumptions/reproducibility**: open/reproducible pipeline with knowledge base and validation; assumes LLM-generated queries can be reliable when grounded.
**Important figures/tables**: Table 1 compares datasets; Figure 1 formalizes the generation framework; Tables 2-3 evaluate filter/persona alignment, groundedness, sustainability, and diversity.
**Discussion/conclusion**: grounded query generation can support better tourism recommender evaluation.
**Critical evaluation**: valuable for experiment design, but it evaluates queries rather than route outcomes.
**From-memory summary**: SynthTRIPs makes realistic tourism prompts so systems can be tested on richer user needs.
**Teach-back**: it creates better fake travelers for testing travel recommenders.
**Project relevance**: use it to generate participant scenarios: nature-heavy, budget-limited, weather-sensitive, low-driving, sustainability-aware.

**Project Action Takeaway**

- **Main goal**: Generate realistic, personalized, grounded, and sustainability-aware tourism queries for evaluating recommender systems.
- **Main limitation**: **Author-reported limitation** - persona interpretation is subjective, sustainable intent may be underrepresented within mixed queries, and generation can favor common destinations.
- **Publication use**: **Supporting experiment-design citation** for constructing diverse traveler scenarios. It evaluates query generation rather than route quality, so it cannot support itinerary-performance claims.
- **Project-polishing action**: Build a small, manually reviewed scenario suite spanning risk tolerance, driving tolerance, nature interest, budget, must-go stops, and disruption severity.
- **Current project status**: **planned** - route configurations exist, but no SynthTRIPs-style scenario generator or reviewed benchmark suite is implemented.

### 33. Ge, Liu, and Wang et al. (2025) - Vaiage

- **PDF**: `2505.10922v1.pdf`
- **Topic**: multi-agent LLM system for personalized travel planning.
- **Main research question**: Can specialized LLM agents collaborate to infer intent, recommend destinations/activities, and synthesize itineraries under real-time constraints?
- **Method**: graph-structured multi-agent framework with prompt-specialized agents, memory, map/UI components, weather, budget, and route interfaces.
- **Conclusion**: Vaiage demonstrates a modular multi-agent travel-planning prototype with dynamic, user-adaptive itineraries.
- **One-sentence purpose**: The paper shows a recent multi-agent prototype, but also highlights the need for stronger validation and optimization.

**Research problem**: static travel tools do not support real-time interaction, context adaptation, and intent refinement.
**Importance**: your project can borrow the multi-agent interface motivation without relying on unvalidated LLM routing.
**Already known**: LLMs can act as recommenders and planners.
**New in the paper**: agent specialization and graph memory for personalized travel planning.

**Introduction notes**: the gap is flexible and interactive planning under budget, timing, group size, and weather.
**Necessary background**: multi-agent workflow, graph memory, goal-conditioned recommender, route planner, map feedback.
**Method/data/assumptions/reproducibility**: prototype and LLM evaluation scores; reproducibility may be limited by prompts/tools.
**Important figures/tables**: Figures 1-9 show workflow, chat, information processing, attraction selection, route planning, map, budget, and itinerary confirmation; Figure 10 summarizes LLM evaluation scores.
**Discussion/conclusion**: agent specialization is promising, but future work includes long-term personalization, collaborative planning, and booking.
**Critical evaluation**: good demo reference, but not enough evidence for route feasibility guarantees.
**From-memory summary**: Vaiage is a multi-agent LLM travel planner with interactive map and budget interfaces.
**Teach-back**: it splits travel planning among several AI specialists, then combines their outputs.
**Project relevance**: compare as an LLM-agent system; your differentiator is validated optimization and inspectable weather-aware tradeoffs.

**Project Action Takeaway**

- **Main goal**: Coordinate specialized LLM agents to infer traveler intent, recommend options, and assemble interactive travel itineraries.
- **Main limitation**: **Author-reported future scope and project-team inference** - long-term personalization, collaborative planning, and booking integration remain future work, while route-feasibility evidence is not equivalent to a formal solver guarantee.
- **Publication use**: **Optional system citation** for agentic and interactive travel planning. Use it to contrast breadth with the project's narrower validated optimization, not to claim direct performance superiority.
- **Project-polishing action**: Emphasize auditable inputs and route tradeoffs instead of adding a multi-agent layer that is not required by the primary research question.
- **Current project status**: **unsupported by current code** - no multi-agent LLM architecture is present.

### 34. Santos et al. (2025) - CityHood

- **PDF**: `2507.18778v1.pdf`
- **Topic**: explainable city and neighborhood travel recommendation.
- **Main research question**: Can a travel recommender explain city/neighborhood suggestions using user interests and regional features?
- **Method**: interactive system using Google Places reviews, geographic/socio-demographic/cultural indicators, LIME, Wikipedia context, and natural-language explanations.
- **Conclusion**: CityHood supports transparent city and neighborhood exploration through explainable, multi-level recommendations.
- **One-sentence purpose**: The paper is a current travel-domain example of explainable recommendation with a visual interface.

**Research problem**: travelers need to understand why a city or neighborhood matches their interests.
**Importance**: it is closer to your HCI goal than pure algorithmic itinerary papers.
**Already known**: LIME and natural-language explanations can support interpretability.
**New in the paper**: multi-level city/neighborhood recommendations grounded in reviews and regional indicators.

**Introduction notes**: the gap is explainable travel recommendation at both city and neighborhood scales.
**Necessary background**: CBSA, ZIP-level recommendation, LIME, review-based interest modeling.
**Method/data/assumptions/reproducibility**: demo system; assumes review-derived features and regional indicators capture traveler interests.
**Important figures/tables**: extracted captions were sparse, but the paper describes a visual interface and natural-language explanation workflow.
**Discussion/conclusion**: future work includes broader geography, richer NLP, and time/intent factors.
**Critical evaluation**: highly relevant for explanation/interface positioning but not a route optimizer.
**From-memory summary**: CityHood recommends places to stay or explore and explains why they fit user interests.
**Teach-back**: it tells a traveler not only "go here," but "go here because these neighborhood features match you."
**Project relevance**: use as a recent explainable travel UI citation; your extension is route feasibility and weather-aware alternatives.

**Project Action Takeaway**

- **Main goal**: Recommend cities or neighborhoods and explain how their features align with a traveler's interests.
- **Main limitation**: **Project-team inference** - the short demo focuses on place-level recommendation and feature-based explanation rather than feasible multi-day route construction or controlled explanation evaluation.
- **Publication use**: **Essential recent travel-interface citation** for interest-linked explanations. The project can extend from place fit to route consequences, but should not claim CityHood validates why-skipped or what-would-change explanations.
- **Project-polishing action**: Tie each selected stop explanation to both interest fit and route feasibility, then add a contrast with the highest-value skipped alternative.
- **Current project status**: **partially implemented** - basic why-selected text and interest scores exist, but contrastive route explanations are incomplete.

### 35. Banerjee et al. (2025) - Collab-Rec

- **PDF**: `2508.15030v6.pdf`
- **Topic**: multi-agent LLM recommendation balancing personalization, popularity, and sustainability in tourism.
- **Main research question**: Can specialist LLM agents and a deterministic moderator reduce popularity bias and improve balanced tourism recommendations?
- **Method**: Personalization, Popularity, and Sustainability agents generate proposals; a non-LLM moderator validates and merges recommendations through constrained refinement.
- **Conclusion**: multi-agent, multi-round refinement improves moderator success and can improve diversity/coverage, but real-time deployment remains future work.
- **One-sentence purpose**: The paper gives a modern multi-stakeholder tourism recommendation framing.

**Research problem**: tourism recommenders can over-concentrate attention on popular places and ignore sustainability or distributional goals.
**Importance**: your route utility could include diversity, environmental burden, crowding, or regional balance.
**Already known**: recommender systems can suffer popularity bias.
**New in the paper**: agentic decomposition by stakeholder objective with deterministic moderation.

**Introduction notes**: the gap is balancing competing objectives in tourism recommendation.
**Necessary background**: popularity bias, sustainability objective, specialist agents, moderator success, Gini/entropy, coverage.
**Method/data/assumptions/reproducibility**: 900 benchmark queries across model backbones and rejection strategies.
**Important figures/tables**: Figure 2 shows specialist agents; Table 1 compares model/rejection strategies; Table 3 reports Gini and coverage; Figure 8 shows ablations.
**Discussion/conclusion**: transparent moderation can coordinate competing objectives but system remains a prototype.
**Critical evaluation**: useful for multi-objective framing, less directly tied to route feasibility.
**From-memory summary**: Collab-Rec balances what the user likes, what is popular, and what is sustainable.
**Teach-back**: instead of one recommender voice, it has several voices and a moderator.
**Project relevance**: supports future route objectives for diversity, congestion avoidance, sustainability, and less over-touristed routes.

**Project Action Takeaway**

- **Main goal**: Balance personalization, popularity, and sustainability through specialist recommendation agents and a deterministic moderator.
- **Main limitation**: **Project-team inference** - recommendation balance is not the same as route feasibility, and multi-agent outputs add dependencies that the current project does not validate.
- **Publication use**: **Supporting or optional citation** for multi-objective and responsible-tourism framing. It cannot support claims about the current route optimizer's sustainability performance without explicit metrics and baselines.
- **Project-polishing action**: If sustainability is retained, define one measurable route objective such as popularity dispersion or congestion avoidance rather than adding a vague sustainability score.
- **Current project status**: **partially implemented** - diversity signals exist, but multi-agent moderation and validated sustainability objectives do not.

### 36. Karmakar et al. (2025) - TripTide

- **PDF**: `2510.21329v1.pdf`
- **Topic**: adaptive travel planning under disruptions.
- **Main research question**: Can LLMs adapt itineraries when disruptions occur at step, day, or plan level under different traveler tolerance profiles?
- **Method**: benchmark with disruption categories, severity levels, traveler tolerance profiles, and metrics for intent preservation, adaptability, and responsiveness.
- **Conclusion**: TripTide exposes how LLMs handle real-world disruptions and shows that adaptive replanning requires more than static itinerary generation.
- **One-sentence purpose**: The paper is the most direct recent support for weather/disruption-aware replanning.

**Research problem**: travel plans often break because of weather, closures, delays, and other disruptions.
**Importance**: weather-aware itinerary optimization is strongest when framed as adaptation under uncertainty, not just static scoring.
**Already known**: travel-planning benchmarks focus mostly on initial plan generation.
**New in the paper**: disruption severity plus traveler tolerance profiles.

**Introduction notes**: the gap is evaluation of adaptive planning after real-world disruptions.
**Necessary background**: disruption category, severity level, tolerance profile, preservation of user intent, adaptability, responsiveness.
**Method/data/assumptions/reproducibility**: benchmarked disruption scenarios and model responses; assumes plausible disruptions can stand in for real-time events.
**Important figures/tables**: Figure 1 motivates persona-guided adaptation; Table 1 gives disruption taxonomy; Figure 2 illustrates multi-level replanning; Table 3 reports preservation/adaptability/responsiveness.
**Discussion/conclusion**: TripTide is positioned as the first benchmark combining disruption severity and tolerance profiles in travel planning.
**Critical evaluation**: extremely relevant, but benchmark uses LLM replanning rather than mathematical optimization.
**From-memory summary**: TripTide asks whether a travel planner can fix a trip when something goes wrong.
**Teach-back**: if rain closes a hike or a flight is delayed, does the planner preserve the spirit of the trip or panic-rewrite everything?
**Project relevance**: central citation for weather-aware replanning and route repair under uncertainty.

**Project Action Takeaway**

- **Main goal**: Benchmark whether travel planners adapt itineraries appropriately to step-, day-, and plan-level disruptions while respecting traveler tolerance and preserving unaffected content.
- **Main limitation**: **Project-team inference** - the benchmark evaluates LLM-based adaptation rather than mathematical optimization and does not itself provide a user-facing explanation or controlled traveler study.
- **Publication use**: **Essential disruption citation** for repair scope, responsiveness, and preservation of intent. It does not prove the current alternatives constitute live replanning or that they preserve intent better than a baseline.
- **Project-polishing action**: Implement original-versus-weather-adjusted comparison with preserved-stop count, route edit distance, utility change, risk reduction, and an explanation of why each changed stop moved.
- **Current project status**: **partially implemented** - weather-aware routes and alternatives exist, but live disruption handling and an explicit preservation-of-intent metric do not.

### 37. Zhang et al. (2026) - Revisiting the Travel Planning Capabilities of Large Language Models

- **PDF**: `2605.03308v1.pdf`
- **Topic**: decomposed evaluation of LLM travel-planning capabilities.
- **Main research question**: Which atomic sub-capabilities cause LLM failures in travel planning?
- **Method**: decoupled evaluation of constraint extraction, tool use, plan generation, error identification, and correction using oracle intermediate contexts.
- **Conclusion**: LLMs show different weaknesses across sub-tasks; end-to-end scores hide root causes.
- **One-sentence purpose**: The paper explains why travel-planning systems should separate preference extraction, tool use, optimization, and validation.

**Research problem**: end-to-end travel benchmarks reveal failure but not why the failure happens.
**Importance**: your pipeline is already decomposed into data, scoring, optimization, export, and validation.
**Already known**: LLMs perform poorly on full travel planning.
**New in the paper**: atomic sub-capability analysis with oracle intermediate contexts.

**Introduction notes**: the gap is interpretability of benchmark failures.
**Necessary background**: constraint extraction, tool-selection accuracy, parameter accuracy, POI match rate, plan coverage.
**Method/data/assumptions/reproducibility**: controlled evaluations with oracle context to isolate errors.
**Important figures/tables**: Figure 1 shows protocol; Table 1 reports constraint extraction; Figure 2 shows constraint-wise precision/recall; Table 2 reports tool use; Table 3 reports plan generation.
**Discussion/conclusion**: future work should test dynamic and multimodal settings and training improvements.
**Critical evaluation**: strong diagnostic framing; still LLM-benchmark centric.
**From-memory summary**: it opens the black box of travel-agent failure into smaller skills.
**Teach-back**: a bad itinerary might fail because the model misunderstood the request, used the wrong tool, chose bad POIs, or could not repair errors.
**Project relevance**: supports your modular validation and explains why dashboard provenance matters.

**Project Action Takeaway**

- **Main goal**: Decompose travel-planning performance into atomic capabilities so researchers can diagnose where LLM systems fail.
- **Main limitation**: **Project-team inference** - the diagnostic framework is centered on LLM agents and does not evaluate optimization-dashboard comprehension or solver quality.
- **Publication use**: **Essential diagnostic citation** for modular validation rather than relying on one end-to-end success score. It cannot establish that the project's current validation modules are complete.
- **Project-polishing action**: Organize validation into input interpretation, data retrieval, POI scoring, route feasibility, repair, explanation, and export checks with separate failure summaries.
- **Current project status**: **partially implemented** - route/export checks and provenance fields exist, but capability-level failure reporting is incomplete.

### 38. Zhao et al. (2026) - TRACE

- **PDF**: `2605.07677v1.pdf`
- **Topic**: accountable tourism conversational recommendation with citation evidence.
- **Main research question**: How can tourism conversational recommenders be evaluated for recommendation accuracy, grounding, and rejection recovery?
- **Method**: benchmark of 10,000 multi-turn dialogues with review-span evidence, spatial/multi-aspect preferences, rejection turns, and 14 baselines.
- **Conclusion**: accuracy-only evaluation misses the core risk; systems must jointly optimize right recommendation, verifiable evidence, and adaptive repair.
- **One-sentence purpose**: TRACE gives the strongest recent grounding/accountability lens for tourism recommendation.

**Research problem**: fluent tourism suggestions can waste real money and trip time if not grounded.
**Importance**: your route explanations should be auditable: why selected, why skipped, and what evidence/constraints support it.
**Already known**: conversational recommender benchmarks often evaluate Recall@k.
**New in the paper**: multi-turn tourism CRS with review-span citations, spatial reasoning, and rejection recovery.

**Introduction notes**: the gap is trustworthy, verifiable, adaptive tourism recommendation.
**Necessary background**: CRS, citation grounding score, POI recommendation, rejection recovery, multi-aspect preference.
**Method/data/assumptions/reproducibility**: Yelp tourism KB, 10,000 dialogues, closed-set/open-set evaluation, multiple baselines.
**Important figures/tables**: Figure 1 defines benchmark gaps; Tables 1-2 compare benchmarks and mechanisms; Table 3 gives corpus statistics; Table 5 reports main results across accuracy/grounding/recovery.
**Discussion/conclusion**: future work should use staged retrieve-justify-repair, NLI-aware grounding, and spatial reasoning.
**Critical evaluation**: very strong for accountability, though it recommends POIs rather than solving routes.
**From-memory summary**: TRACE asks recommenders to be right, cite evidence, and recover when users reject suggestions.
**Teach-back**: a travel assistant should not just sound convincing; it should prove why a place fits and adapt when the traveler says no.
**Project relevance**: use for accountable route explanations and evidence-backed skipped/selected reasoning.

**Project Action Takeaway**

- **Main goal**: Evaluate tourism conversational recommenders on recommendation accuracy, evidence grounding, and recovery after a user rejects a suggestion.
- **Main limitation**: **Author-reported limitation** - parts of the evaluation rely on a single LLM judge, additional cross-judge checks remain future work, and the full Yelp-derived corpus is not independently redistributed; **project-team inference** - TRACE recommends POIs rather than solving routes.
- **Publication use**: **Essential accountability citation** for evidence-backed rationale and rejection recovery. It cannot support claims that current source-confidence fields constitute citation-level grounding.
- **Project-polishing action**: Link selected and skipped explanations to inspectable source fields, show confidence separately from preference fit, and log how a rejected POI changes the validated route.
- **Current project status**: **partially implemented** - source confidence and rationale fields exist, while citation-level grounding and rejection recovery remain unimplemented.

### 39. Cheng et al. (2026) - GroupTravelBench

- **PDF**: `2605.25200v2.pdf`
- **Topic**: multi-user, multi-turn travel planning benchmark for LLM agents.
- **Main research question**: Can LLM agents elicit private preferences, coordinate conflicts, and produce fair group travel plans?
- **Method**: 650 tasks across group archetypes and difficulty levels, grounded in real profiles, POIs, transport prices, sandbox tools, and rule/LLM-judge evaluation.
- **Conclusion**: even strong agents fall short on group preference completeness, fairness, and process quality.
- **One-sentence purpose**: The paper extends travel planning from single-user optimization to group negotiation and fairness.

**Research problem**: real travel is often group travel, but benchmarks usually assume one user.
**Importance**: future versions of your system could model family/group tradeoffs and fairness.
**Already known**: single-user TravelPlanner-style benchmarks are increasingly saturated.
**New in the paper**: multi-user, multi-turn group-chat travel-planning sandbox.

**Introduction notes**: the gap is discovering private preferences and coordinating disagreement.
**Necessary background**: group archetype, preference completeness, group fairness, process quality, LLM judge.
**Method/data/assumptions/reproducibility**: real user profiles, hundreds of thousands of POIs/tool records, offline sandbox comparison.
**Important figures/tables**: Figure 1 shows benchmark components; Table 1 reports model results; Figure 2 profiles LLM-judge dimensions; Tables 2-3 validate sandbox fidelity and stability.
**Discussion/conclusion**: group planning requires elicitation, compromise, subgrouping, and fairness, not only route generation.
**Critical evaluation**: valuable future direction; not necessary for the current single-traveler route dashboard.
**From-memory summary**: GroupTravelBench tests whether agents can plan a trip for several people with conflicting preferences.
**Teach-back**: planning for a group is harder because the planner must ask, negotiate, and be fair.
**Project relevance**: optional future extension: group-aware interest bars and route compromise explanations.

**Project Action Takeaway**

- **Main goal**: Benchmark multi-turn travel planning for groups whose members have private, conflicting, and differently weighted preferences.
- **Main limitation**: **Project-team inference** - group preference elicitation and fairness introduce a separate research problem beyond the current single-traveler dashboard and would substantially broaden the evaluation.
- **Publication use**: **Optional future-work citation** for group negotiation and compromise explanations. It should not appear as evidence for current personalization or fairness capabilities.
- **Project-polishing action**: Keep group planning out of the next study; design current preference and explanation schemas so multiple users could be added later without claiming support now.
- **Current project status**: **unsupported by current code** - group preference elicitation, negotiation, and fairness metrics are absent.

### 40. Chen et al. (2026) - TravelEval

- **PDF**: `2606.01046v1.pdf`
- **Topic**: comprehensive benchmark for LLM-powered travel-planning agents.
- **Main research question**: How should full travel plans be evaluated beyond surface plausibility?
- **Method**: benchmark with realistic sandbox and six-dimensional evaluation: accuracy, compliance, temporality, spatiality, economy, and utility.
- **Conclusion**: even state-of-the-art LLM approaches struggle with multidimensional global planning, especially feasibility and budget satisfaction.
- **One-sentence purpose**: TravelEval gives the most current evaluation vocabulary for complete travel plans.

**Research problem**: current travel-agent evaluation lacks realistic, multidimensional, global plan assessment.
**Importance**: your project needs evaluation criteria that capture feasibility, utility, budget, time, space, and route quality.
**Already known**: TravelPlanner exposed low pass rates, but evaluation dimensions can be richer.
**New in the paper**: six-dimensional framework and simulation-based global evaluation.

**Introduction notes**: the gap is comprehensive evaluation for LLM travel agents.
**Necessary background**: BCS, TTCS, spatial/temporal/economic/utility metrics, sandbox evaluation.
**Method/data/assumptions/reproducibility**: evaluates 12 approaches in a sandbox with normalized multi-dimensional metrics.
**Important figures/tables**: Table 1 compares benchmarks; Figure 2 shows TravelEval framework; Figure 4 compares six-dimensional scores; Figure 5 shows metric correlations; Table 3 reports model-strategy results.
**Discussion/conclusion**: LLM-generated plans have severe feasibility defects, and future systems need better constrained planning.
**Critical evaluation**: highly useful for evaluation design; focuses on LLM agents rather than solver dashboards.
**From-memory summary**: TravelEval says a travel plan should be graded across many dimensions, not just whether it sounds good.
**Teach-back**: a nice itinerary fails if it breaks the budget, makes impossible jumps, or wastes the user's time.
**Project relevance**: use its dimensions to define evaluation metrics for your system: compliance, temporal feasibility, spatial coherence, economy, utility, and context risk.

**Project Action Takeaway**

- **Main goal**: Provide multidimensional evaluation for complete LLM-generated travel plans rather than judging only fluency or surface plausibility.
- **Main limitation**: **Project-team inference** - its evaluation target is LLM agents, not solver-backed dashboards, and benchmark dimensions still require operationalization for California route artifacts and weather adaptation.
- **Publication use**: **Essential evaluation citation** for compliance, temporal feasibility, spatial coherence, economy, and utility as separate outcomes. It cannot show that this project performs well until those metrics are computed against baselines.
- **Project-polishing action**: Create a reproducible evaluation table for every route alternative containing feasibility, travel time, budget, utility, weather risk, lodging completeness, runtime, and solver/fallback status.
- **Current project status**: **partially implemented** - several route and validation metrics exist, but a complete benchmark-style comparison has not been run.

## Remaining Online Leads and External Context

Most of the 2023-2026 online expansion is now represented by downloaded PDFs and full entries above. The table below keeps a compact index of external context and a few still-useful papers that are not all full local deep reads.

| Paper | Source | Why relevant | Gap supported | Priority |
| --- | --- | --- | --- | --- |
| Xie et al. (2024), **TravelPlanner: A Benchmark for Real-World Planning with Language Agents** | [arXiv](https://arxiv.org/abs/2402.01622) | Shows LLM agents fail at multi-constraint travel planning. | Need structured validation and solver-backed planning. | Must cite |
| Ju et al. (2024), **To the Globe (TTG)** | [arXiv](https://arxiv.org/abs/2410.16456) | Demonstrates LLM-to-symbolic + MILP travel planning. | Natural-language interface should compile to constraints. | Must cite |
| Shao et al. (2026), **ChinaTravel** | [arXiv](https://arxiv.org/abs/2412.13682) | Open-ended travel benchmark with compositional constraint validation and human queries. | Need compositional constraints and validation beyond slot filling. | Must cite for AI-agent path |
| Valmeekam et al. (2022), **RADAR-X** | [arXiv](https://arxiv.org/abs/2011.09644) | Mixed-initiative planning with contrastive explanations and revised plan suggestions. | Route explanations should answer "why this route instead of that one?" | Must cite for explanation path |
| Krarup et al. (2021), **Contrastive Explanations of Plans Through Model Restrictions** | [arXiv](https://arxiv.org/abs/2103.15575) | Formalizes plan negotiation through contrastive questions. | Supports why selected/why skipped/why not alternative explanations. | Supporting cite |
| Zhang et al. (2024), **A Survey of Route Recommendations** | [arXiv](https://arxiv.org/abs/2403.00284) | Updates route recommendation methods and applications in urban computing. | Shows route recommendation is broader than tourism and increasingly data-driven. | Full local entry above |
| Jayasuriya and Sumanathilaka (2025), **Trip Route Planning with Travel Time Estimation** | [arXiv](https://arxiv.org/abs/2503.23486) | Recent review of adaptive route planning, preferences, and travel-time estimation. | Supports travel-time and dynamic preference framing. | Optional/supporting |
| Ho and Lim (2023), **Utilizing Language Models for Tour Itinerary Recommendation** | [arXiv](https://arxiv.org/abs/2311.12355) | Discusses language models for tour itinerary recommendation. | Bridges traditional itinerary recommendation and LLM approaches. | Full local entry above |
| Chen et al. (2024), **TravelAgent** | [arXiv](https://arxiv.org/abs/2409.08069) | LLM travel assistant with tool-use, recommendation, planning, and memory. | Shows current AI travel system direction and evaluation dimensions. | Full local entry above |
| Liu, Wood, and Lim (2019), **Strategic and Crowd-Aware Itinerary Recommendation** | [arXiv](https://arxiv.org/abs/1909.07775) | Models crowd-aware itinerary recommendation and social welfare. | Extends queue/crowd burden beyond individual route quality. | Supporting cite |
| Alves et al. (2026), **Context Notifications in a Tourism Recommender System** | [arXiv](https://arxiv.org/abs/2605.07960) | Pilot study with weather, air quality, noise, and proximity notifications. | Shows context delivery itself is an HCI problem. | Interesting, optional |

Note: a previously seeded citation titled "Design of Weather-Aware Mobile Recommender Systems in Tourism" could not be reliably verified from the searches run here; the CEUR URL in the seed review opened a different paper. Treat it as unverified until a correct source is found.

## Difficult Sections Revisited

### OP / TTDP objective, in project language

Classic OP:

```text
maximize selected_stop_value
subject to route_time <= budget
           route starts/ends correctly
           selected stops are visited at most once
```

TTDP adds tourism realism:

```text
maximize tourist_utility(route)
subject to daily_time_budget
           POI opening hours
           hotel/base-city structure
           mandatory or forbidden stops
           category limits
           travel-time feasibility
```

Your project adds contextual and inspectable terms:

```text
maximize base_value
       + interest_fit
       + nature/scenic bonus
       - weather_exposure
       - detour burden
       - travel fatigue
subject to time, budget, route, lodging, and scenario constraints
```

The publication opportunity is not the equation alone. It is that each term can become a user-facing explanation and critique handle.

### Branch-and-check, step by step

1. Master problem selects a set of candidate POIs.
2. Subproblem checks whether those POIs can be scheduled with time windows and route feasibility.
3. If infeasible, the algorithm generates a cut that blocks similar impossible selections.
4. It repeats until it finds feasible/optimal solutions or proves bounds.

Project translation: a dashboard should tell users not just "Yosemite was skipped" but "Yosemite plus Big Sur plus 7 days violates daily driving and time budget under this scenario."

### Cognitive forcing for route planning

Bad route UI:

```text
Here is the optimal route. Accept?
```

Better study UI:

```text
Route A has higher nature value but higher weather exposure.
Route B has lower weather risk but misses Yosemite.
Route C minimizes driving but loses two high-value outdoor stops.
Choose after inspecting why each route was selected and what it skips.
```

The goal is calibrated reliance: users should neither blindly trust the route nor reject useful optimization.

### Contrastive explanation pattern

For travel, the most useful explanations are contrastive:

```text
Why Big Sur instead of Yosemite?
Why this hotel base instead of Monterey?
Why did the optimizer skip Death Valley?
What would need to change to include Sequoia?
```

This maps directly to RADAR-X and contrastive planning literature.

## Ranked Publishable Ideas

| Rank | Idea | Novelty | Feasibility | Evidence needed | Why it stands out |
| --- | --- | --- | --- | --- | --- |
| 1 | **Accountable, inspectable, solver-backed itinerary adaptation under weather/disruption uncertainty** | High | High | Weather/disruption scenarios, route repair artifacts, user study | Connects TripTide/TravelEval-style evaluation with your existing weather-aware optimizer and dashboard |
| 2 | **Contrastive route explanations: why selected, why skipped, what would change** | High | Medium-high | Explanation panel plus comprehension/decision study | Converts solver constraints into accountable user-facing explanations aligned with TRACE and RADAR-X |
| 3 | **Mixed-initiative route critiquing and repair** | High | Medium | Critique controls, route repair loop, revision logs | Lets users say less driving, more nature, lower weather risk, include Yosemite, or avoid closures |
| 4 | **LLM-to-symbolic weather-aware trip planning** | Medium-high | Medium | NL parser, schema, validation tests, solver integration | Builds on TRIP-PAL and TTG while adding weather, route inspection, and user revision |
| 5 | **Trust-calibrated itinerary acceptance through comparison/cognitive forcing** | Medium-high | Medium | Explanation-only vs. compare-before-accept study | Reduces blind reliance on "optimal" or fluent AI-generated routes |
| 6 | **Modern route-quality evaluation framework for human-centered TTDP** | Medium | High | Metrics mapped to TravelEval: compliance, temporality, spatiality, economy, utility, risk | Makes your evaluation feel current without needing a new algorithmic solver claim |

Recommended primary path: **Idea 1 plus Idea 2**. Build a study-ready dashboard that exposes weather/disruption tradeoffs, skipped alternatives, route repair logic, and accountable explanations.

## Concrete Project Recommendations

### What to build next

- Add a **study mode** to the dashboard that hides debug clutter and presents stable participant tasks.
- Add explanation panels:
  - **Why selected**: high utility, interest fit, feasible day placement, weather acceptable.
  - **Why skipped**: detour too high, weather risk too high, insufficient days, budget/time conflict, lower fit.
  - **What would change**: "include Yosemite requires 9 days or higher daily driving limit."
- Add **route diff** when switching saved routes, trip lengths, or interest profiles.
- Add explicit labels for **saved optimized route** vs. **browser preview**.
- Add participant logging: route viewed, layers toggled, controls changed, stops inspected, final route accepted, time on task.

### What to evaluate

- Baseline condition: ranked POI list or static route map with basic details.
- Proposed condition: optimization dashboard with route alternatives, weather/detour explanations, and contrastive skipped-stop reasoning.
- Tasks:
  - choose a nature-heavy California route under rainy outdoor risk;
  - reduce driving while preserving high-value stops;
  - decide whether to include a high-value but high-detour national park;
  - explain why a skipped stop was excluded.
- Measures:
  - comprehension of objective/constraints;
  - perceived control;
  - trust calibration;
  - explanation usefulness;
  - workload;
  - route satisfaction;
  - final route quality;
  - revision behavior.

### How to frame the contribution

Use this updated framing:

> We contribute an accountable, human-centered optimization interface for weather-aware multi-day itinerary adaptation. Unlike LLM travel agents that often fail hidden constraints, tourism recommenders that rank POIs, or OR systems that output routes without explanation, the system makes route feasibility, weather exposure, disruption response, detour burden, lodging/base decisions, and skipped alternatives inspectable and revisable.

Avoid claiming:

- "new best TTDP solver";
- "fully autonomous AI travel agent";
- "weather-aware recommendation alone";
- "map dashboard as visualization polish."

Claim instead:

- **design contribution**: inspectable route optimization interface;
- **technical artifact**: weather/nature-aware multi-day TTDP prototype with route alternatives, adaptation logic, and provenance;
- **empirical contribution**: evidence that explanations and comparison improve user understanding/control/trust calibration;
- **framework contribution**: taxonomy of route explanations: selected, skipped, changed, uncertain.

## What I Learned From the Corpus

1. The math literature gives legitimacy but not publication distinctiveness by itself.
2. Weather belongs naturally with time, queue, congestion, and context as a dynamic burden signal.
3. The project's visible provenance, fallback status, and route alternatives are not side features; they are the HCI contribution.
4. Explanations should be designed by goal: transparency, scrutability, decision quality, and calibrated trust.
5. A serious user study should measure mental models and revision behavior, not only satisfaction.
6. LLM travel planning is exciting but currently risky unless paired with symbolic constraints, validation, and grounded evidence.
7. Recent benchmarks make the project more current: TravelEval supplies evaluation dimensions, TripTide supplies disruption framing, TRACE supplies accountability, and TRIP-PAL/TTG supply solver-backed guarantees.
8. The project can stand out by making optimization constraints conversational and inspectable: "why this route, why not that route, and what must change?"

## Recommended Citation Backbone

For a CHI/HCI-oriented paper:

- Horvitz (1999) for mixed initiative.
- Shneiderman (2020) for high automation plus high human control.
- Amershi et al. (2019) for human-AI guidelines.
- Bucinca et al. (2021) for overreliance/cognitive forcing.
- Kulesza et al. (2012) for mental models.
- Tintarev and Masthoff (2007), Knijnenburg et al. (2012), and Chen and Pu (2012) for explanations, UX, and critiquing.
- Heer and Shneiderman (2012), Endert et al. (2017), and Sacha et al. (2017) for visual analytics and human-centered model steering.
- Vansteenwegen et al. (2011), Gunawan et al. (2016), Ruiz-Meza and Montoya-Torres (2022), and Vu et al. (2022) for TTDP/OP foundations.
- Lim et al. (2017), Yuan et al. (2013), Borras et al. (2014), and Halder et al. (2024) for context-aware tourism and itinerary recommendation.
- TravelPlanner, TravelEval, Revisiting Travel Planning Capabilities, and GroupTravelBench for modern LLM travel-planning evaluation.
- TRIP-PAL and TTG for LLM + symbolic planning guarantees.
- TripTide for disruption/weather-aware adaptive replanning.
- TRACE for accountable, grounded tourism recommendation and rejection recovery.
- CityHood for explainable travel recommendation interfaces.
- BTRec, +Tour, TravelAgent, Vaiage, Collab-Rec, and SynthTRIPs for recent personalization, agentic, and benchmark-generation baselines.
- Visualization for Recommendation Explainability, Whom do Explanations Serve, and Review of Explainable Graph-Based Recommender Systems for modern visual/XRS theory.

## Modern Citation Backbone

- **LLM travel-planning benchmarks**: TravelPlanner; TravelEval; Revisiting the Travel Planning Capabilities of LLMs; GroupTravelBench; ChinaTravel if kept as an external citation.
- **LLM + symbolic guarantees**: TRIP-PAL; TTG.
- **Disruption/weather replanning**: TripTide, plus your own weather-risk route repair as the project contribution.
- **Accountable tourism recommendation**: TRACE for evidence, grounding, rejection recovery, and spatial/multi-aspect recommendation.
- **Explainable travel recommendation**: CityHood for travel-domain explanation UI; RADAR-X and contrastive planning for "why this, not that" explanations.
- **Recent personalization baselines**: BTRec, +Tour, TravelAgent, Vaiage, Collab-Rec, SynthTRIPs.
- **Visual/explainable recommender theory**: Visualization for Recommendation Explainability; Whom do Explanations Serve; Review of Explainable Graph-Based Recommender Systems; Tintarev and Masthoff; Knijnenburg et al.; Chen and Pu.
