# CHI-Oriented Literature Review for Weather-Aware Itinerary Optimization

Prepared: 2026-06-10

## Executive Framing

The strongest publication path is not to claim a new pure optimization algorithm. The better claim is that this project is an **inspectable optimization interface for travel planning under contextual uncertainty**. The system already has a rare combination: explicit objectives and constraints, weather and detour risk, nature/city interest controls, hotel/base-city decisions, route alternatives, solver/fallback status, and dashboard validation. The literature below suggests that the missing 40-50 percent is mainly:

- a sharper HCI contribution statement,
- a study-ready prototype and baseline,
- human-subject evidence about understanding, agency, trust, and revision behavior,
- a cleaner bridge between TTDP optimization quality and human decision quality.

The review is grouped into five buckets. Each paper includes summary, methodology/data, project relevance, limitation/gap, publication-pivot implication, relevance score, and reading priority.

## Bucket A: HCI / Human-AI Decision Support

### 1. Horvitz, E. (1999). Principles of Mixed-Initiative User Interfaces. CHI 1999.

- Link: https://doi.org/10.1145/302979.303030
- Summary: Horvitz argues that intelligent systems should negotiate initiative between human and machine rather than fully automate user decisions. The paper develops principles for when systems should act, defer, interrupt, ask, explain, or transfer control.
- Methodology and data: Conceptual and system-design paper grounded in examples of intelligent interfaces and decision-theoretic reasoning.
- Why it matters: Your itinerary system should not be framed as "the optimizer decides the trip." It should be framed as a mixed-initiative planning tool where the optimizer proposes, the dashboard exposes tradeoffs, and the traveler remains able to inspect and revise.
- Limitation or gap: It predates modern AI recommender systems and does not address map-based travel planning or optimization dashboards directly.
- Publication pivot: Use this paper to justify why saved optimized routes, preview-only controls, and user revision loops are central rather than extra UI polish.
- Relevance: 5/5
- Priority: must read

### 2. Amershi, S., Weld, D., Vorvoreanu, M., Fourney, A., Nushi, B., et al. (2019). Guidelines for Human-AI Interaction. CHI 2019.

- Link: https://doi.org/10.1145/3290605.3300233
- Summary: The paper provides empirically grounded guidelines for human-AI systems, including making clear what the system can do, showing confidence/uncertainty, supporting correction, and learning from user feedback.
- Methodology and data: Literature synthesis, expert review, guideline generation, and validation across AI product scenarios.
- Why it matters: Your dashboard already exposes artifact freshness, route state, solver role, and selected-route details. These are exactly the kinds of interaction properties that CHI reviewers will recognize as human-AI design decisions.
- Limitation or gap: The guidelines are broad and do not solve the specific research question of itinerary planning under weather, budget, and travel constraints.
- Publication pivot: Map the project features to 4-6 guidelines: scope/limits, uncertainty, explanations, user control, feedback, and graceful fallback.
- Relevance: 5/5
- Priority: must read

### 3. Shneiderman, B. (2020). Human-Centered Artificial Intelligence: Reliable, Safe and Trustworthy. International Journal of Human-Computer Interaction.

- Link: https://doi.org/10.1080/10447318.2020.1741118
- Summary: Shneiderman proposes human-centered AI as a model that combines high levels of automation with high levels of human control. The core argument is that AI should amplify human ability while preserving agency and accountability.
- Methodology and data: Conceptual framework with design principles and examples across AI-assisted decision domains.
- Why it matters: This gives the highest-level theoretical umbrella for your project: the route is computed by algorithms, but the user must remain in control of preferences, interpretation, and acceptance.
- Limitation or gap: It is a broad manifesto rather than a specific empirical study of travel planning interfaces.
- Publication pivot: Use HCAI language carefully: "high automation plus high human control" is a clean way to defend an optimization dashboard over a black-box itinerary generator.
- Relevance: 5/5
- Priority: must read

### 4. Bucinca, Z., Malaya, M. B., and Gajos, K. Z. (2021). To Trust or to Think: Cognitive Forcing Functions Can Reduce Overreliance on AI in AI-assisted Decision-making. Proceedings of the ACM on Human-Computer Interaction.

- Link: https://doi.org/10.1145/3449287
- Summary: This paper shows that explanations alone may not reduce overreliance on AI. Cognitive forcing functions can make users think more carefully, but they may reduce subjective satisfaction.
- Methodology and data: Controlled experiment with 199 participants comparing cognitive forcing designs, simple explanations, and no-AI conditions.
- Why it matters: Your dashboard should not only show an "optimal" route. It should require users to compare alternatives, inspect skipped anchors, and notice weather/travel tradeoffs before accepting.
- Limitation or gap: The task domain is not travel, and the paper focuses on AI decision support rather than optimization interfaces.
- Publication pivot: Design the study to test whether route comparison, warnings, and preview labels reduce blind acceptance while preserving perceived control.
- Relevance: 5/5
- Priority: must read

## Bucket B: Optimization Interfaces / Visual Analytics

### 5. Heer, J. and Shneiderman, B. (2012). Interactive Dynamics for Visual Analysis. Communications of the ACM.

- Link: https://doi.org/10.1145/2133806.2133821
- Summary: The paper presents a taxonomy of interactive visual analysis: data/view specification, view manipulation, and process/provenance. It explains how interactive systems help users explore, filter, compare, annotate, and guide analysis.
- Methodology and data: Conceptual taxonomy derived from visual analytics practice and examples.
- Why it matters: Your map dashboard is a visual analytics artifact, not merely a map. It supports route comparison, filtering layers, inspecting hotels, seeing weather risk, and validating provenance.
- Limitation or gap: The paper does not focus on AI/optimization recommendations or travel planning.
- Publication pivot: Use this paper to describe the dashboard as a visual interface for optimization evidence: selected route, alternatives, route metadata, provenance, and validation.
- Relevance: 4/5
- Priority: important

### 6. Endert, A., Ribarsky, W., Turkay, C., Wong, B. L. W., Nabney, I., Blanco, I. D., and Rossi, F. (2017). The State of the Art in Integrating Machine Learning into Visual Analytics. Computer Graphics Forum.

- Link: https://doi.org/10.1111/cgf.13092
- Summary: This survey explains how machine learning and visual analytics can be integrated so users can steer, inspect, and understand model outputs.
- Methodology and data: State-of-the-art report synthesizing visual analytics systems and design patterns.
- Why it matters: Your system integrates predictive signals, optimization, and dashboard inspection. This paper helps position the dashboard as a way to make the model pipeline inspectable.
- Limitation or gap: It is general visual analytics and does not address itinerary planning or weather-aware travel.
- Publication pivot: Use it to justify interface features that expose intermediate model states: utility, uncertainty, source status, solver status, and route alternatives.
- Relevance: 4/5
- Priority: important

### 7. Sacha, D., Sedlmair, M., Zhang, L., Lee, J. A., Peltonen, J., Weiskopf, D., North, S. C., and Keim, D. A. (2017). What You See Is What You Can Change: Human-Centered Machine Learning by Interactive Visualization. Neurocomputing.

- Link: https://doi.org/10.1016/j.neucom.2017.01.105
- Summary: The authors argue for interactive visualization as a way for humans to inspect and modify machine-learning processes rather than passively receive model outputs.
- Methodology and data: Conceptual and survey-oriented discussion of human-centered ML workflows and interactive visualization loops.
- Why it matters: Your interest sliders and preview-only route logic can be framed as "what users can see and change" in a constrained planning pipeline.
- Limitation or gap: It focuses on ML more than mathematical optimization and does not provide a travel-specific evaluation.
- Publication pivot: Use it to argue that users need visible handles for model assumptions, not only final recommendations.
- Relevance: 4/5
- Priority: important

### 8. Kulesza, T., Stumpf, S., Burnett, M., and Kwan, I. (2012). Tell Me More? The Effects of Mental Model Soundness on Personalizing an Intelligent Agent. CHI 2012.

- Link: https://doi.org/10.1145/2207676.2207678
- Summary: The paper studies how explanation detail affects users' mental models of intelligent systems and their ability to personalize them.
- Methodology and data: User study of explanation interfaces for an intelligent agent, measuring mental-model quality and personalization behavior.
- Why it matters: Your future user study should measure whether users understand what the optimizer is optimizing and how changing interests, weather sensitivity, or pace changes route logic.
- Limitation or gap: It does not involve geographic routing or multi-objective optimization.
- Publication pivot: Include mental-model soundness as an evaluation construct, not just satisfaction or trust.
- Relevance: 4/5
- Priority: important

## Bucket C: Tourist Trip Design / Orienteering

### 9. Vansteenwegen, P., Souffriau, W., and Van Oudheusden, D. (2011). The Orienteering Problem: A Survey. European Journal of Operational Research.

- Link: https://doi.org/10.1016/j.ejor.2010.03.045
- Summary: This survey defines the orienteering problem family: selecting and sequencing valuable stops under a travel/time budget. It reviews classic models, variants, and solution methods.
- Methodology and data: Literature survey of OP variants and algorithms.
- Why it matters: It is the mathematical root of your attraction-selection-and-routing layer.
- Limitation or gap: It focuses on algorithmic formulations, not human-facing route explanation or interactive planning.
- Publication pivot: Cite it to show that your route problem is not a ranked list problem; it is a constrained path selection problem.
- Relevance: 5/5
- Priority: must read

### 10. Gunawan, A., Lau, H. C., and Vansteenwegen, P. (2016). Orienteering Problem: A Survey of Recent Variants, Solution Approaches and Applications. European Journal of Operational Research.

- Link: https://doi.org/10.1016/j.ejor.2016.04.059
- Summary: This paper extends OP coverage to richer variants, applications, exact methods, heuristics, and metaheuristics.
- Methodology and data: Survey and taxonomy of OP variants and solution approaches.
- Why it matters: It helps place your method among rich OP/TTDP variants with multiple constraints and real-world application requirements.
- Limitation or gap: It is still algorithm-centered, with little attention to user interaction or explanation.
- Publication pivot: Use it to avoid overclaiming mathematical novelty while showing your system is a rich applied OP variant.
- Relevance: 5/5
- Priority: must read

### 11. Ruiz-Meza, J. and Montoya-Torres, J. R. (2022). A Systematic Literature Review for the Tourist Trip Design Problem: Extensions, Solution Techniques and Future Research Lines. Operations Research Perspectives.

- Link: https://doi.org/10.1016/j.orp.2022.100228
- Summary: This systematic review maps TTDP extensions, including time windows, multiple days, transport, preferences, and solution methods.
- Methodology and data: Systematic literature review with classification of TTDP papers and future research directions.
- Why it matters: It gives the most direct OR literature map for your technical model.
- Limitation or gap: It does not deeply treat HCI evaluation, user agency, or explanation interfaces.
- Publication pivot: Use this paper to identify the boundary: the OR literature is strong on constraints but weak on how users inspect and revise the resulting itinerary.
- Relevance: 5/5
- Priority: must read

### 12. Vu, D. M., Kergosien, Y., Mendoza, J. E., and Desport, P. (2022). Branch-and-Check Approaches for the Tourist Trip Design Problem with Rich Constraints. Computers & Operations Research.

- Link: https://doi.org/10.1016/j.cor.2021.105566
- Summary: The paper studies exact branch-and-check methods for TTDP instances with richer practical constraints.
- Methodology and data: Mathematical formulation, branch-and-check algorithms, and computational experiments on benchmark/problem instances.
- Why it matters: It is close to your optimization side: rich constraints, feasibility, and exact/fallback solver status.
- Limitation or gap: It does not address how non-expert travelers understand solver choices, infeasibility, or tradeoffs.
- Publication pivot: Your paper should not compete as a better exact solver. Instead, it should show how rich constraints become inspectable by users.
- Relevance: 4/5
- Priority: important

## Bucket D: Tourism Recommender Systems and Context-Aware Itinerary Recommendation

### 13. Borras, J., Moreno, A., and Valls, A. (2014). Intelligent Tourism Recommender Systems: A Survey. Expert Systems with Applications.

- Link: https://doi.org/10.1016/j.eswa.2014.05.028
- Summary: This survey reviews tourism recommender systems, including personalization, knowledge-based methods, context-awareness, and group/trip recommendation.
- Methodology and data: Survey and taxonomy of tourism recommender systems.
- Why it matters: It positions your project in tourism recommendation but also shows why your contribution is beyond attraction ranking.
- Limitation or gap: It predates recent deep-learning, LLM, and modern mixed-initiative interfaces.
- Publication pivot: Use it as background, then argue that your system adds route feasibility, weather risk, and inspectable optimization.
- Relevance: 4/5
- Priority: important

### 14. Halder, S., Lim, K. H., Chan, J., and Zhang, X. (2024). A Survey on Personalized Itinerary Recommendation: From Optimisation to Deep Learning. Applied Soft Computing.

- Link: https://doi.org/10.1016/j.asoc.2023.111200
- Summary: This recent survey connects optimization-based itinerary recommendation, deep learning, and personalization.
- Methodology and data: Broad literature review of personalized itinerary methods and trends.
- Why it matters: It is the clearest current survey for explaining how itinerary recommendation has moved from pure optimization to learned personalization.
- Limitation or gap: Like most recommender surveys, it emphasizes algorithmic families more than interface-level agency and user-study evidence.
- Publication pivot: Use it to justify your hybrid position: optimization remains useful, but personalization and interaction are where the publishable HCI contribution lives.
- Relevance: 5/5
- Priority: must read

### 15. Lim, K. H., Chan, J., Karunasekera, S., and Leckie, C. (2017). Personalized Itinerary Recommendation with Queuing Time Awareness. SIGIR 2017.

- Link: https://doi.org/10.1145/3077136.3080778
- Summary: This paper adds queuing/waiting time awareness to personalized itinerary recommendation, making route quality depend on dynamic congestion rather than static POI value alone.
- Methodology and data: Algorithmic recommendation model evaluated on itinerary/POI data with queue-time considerations.
- Why it matters: It strongly supports your weather-aware waiting-time and congestion framing.
- Limitation or gap: It is not centered on weather risk, hotels, nature-region planning, or human-facing explanation.
- Publication pivot: Use it to show that contextual burden matters, then argue that your dashboard makes such burden inspectable.
- Relevance: 5/5
- Priority: must read

### 16. Yuan, Q., Cong, G., Ma, Z., Sun, A., and Magnenat-Thalmann, N. (2013). Time-Aware Point-of-Interest Recommendation. SIGIR 2013.

- Link: https://doi.org/10.1145/2484028.2484030
- Summary: The paper studies how time changes POI recommendation, showing that user mobility and POI attractiveness are temporally sensitive.
- Methodology and data: Data-driven POI recommendation using temporal patterns from location/check-in behavior.
- Why it matters: It supports the general claim that context changes destination value, which your system extends to weather, seasonality, waiting, and travel burden.
- Limitation or gap: It recommends POIs, not full multi-day constrained routes with hotels and user-facing optimization explanations.
- Publication pivot: Use it as a stepping stone from context-aware POI ranking to constrained itinerary decision support.
- Relevance: 4/5
- Priority: important

## Bucket E: Explainable / Controllable Recommenders and Weather-Aware Tourism

### 17. Tintarev, N. and Masthoff, J. (2007). A Survey of Explanations in Recommender Systems. ICDE Workshops 2007.

- Link: https://doi.org/10.1109/ICDEW.2007.4401070
- Summary: The paper identifies purposes of explanations in recommender systems, including transparency, scrutability, trust, effectiveness, persuasiveness, efficiency, and satisfaction.
- Methodology and data: Survey and taxonomy of explanation goals in recommender systems.
- Why it matters: Your dashboard explanations should not be generic. They should be designed around specific goals: helping users see why the route was selected, what was skipped, and how to revise.
- Limitation or gap: It is a taxonomy and not a modern empirical evaluation of travel dashboards.
- Publication pivot: Build your study measures around explanation purposes: transparency, scrutability, control, and satisfaction.
- Relevance: 5/5
- Priority: must read

### 18. Knijnenburg, B. P., Willemsen, M. C., Gantner, Z., Soncu, H., and Newell, C. (2012). Explaining the User Experience of Recommender Systems. User Modeling and User-Adapted Interaction.

- Link: https://doi.org/10.1007/s11257-011-9118-4
- Summary: This paper proposes a framework connecting objective system aspects, subjective user experience, interaction behavior, and user outcomes in recommender systems.
- Methodology and data: User-experience model and empirical recommender-system evaluation.
- Why it matters: It gives you a framework for study variables: perceived control, trust, effort, satisfaction, choice difficulty, and acceptance.
- Limitation or gap: It is not itinerary-specific and does not involve mathematical optimization.
- Publication pivot: Use its evaluation model to design a mixed-method study instead of reporting only route utility and solver metrics.
- Relevance: 5/5
- Priority: must read

### 19. Chen, L. and Pu, P. (2012). Critiquing-Based Recommenders: Survey and Emerging Trends. User Modeling and User-Adapted Interaction.

- Link: https://doi.org/10.1007/s11257-011-9108-6
- Summary: This survey reviews recommenders that let users critique recommendations, such as "cheaper," "closer," or "more scenic," instead of accepting passive suggestions.
- Methodology and data: Survey of critique-based recommendation approaches and interface patterns.
- Why it matters: Your interest bars and route controls are a form of critique, but currently they are partly preview-only. This paper can help turn them into a more rigorous interaction loop.
- Limitation or gap: It focuses more on product/recommendation interaction than optimization under hard feasibility constraints.
- Publication pivot: Add explicit critique tasks in the user study: "make this route less weather-exposed," "include more nature," "reduce driving," or "keep budget lower."
- Relevance: 4/5
- Priority: important

### 20. Braunhofer, M., Elahi, M., Ge, M., Ricci, F., and Schievenin, T. (2013). STS: Design of Weather-Aware Mobile Recommender Systems in Tourism. Workshop on Intelligent User Interfaces: Artificial Intelligence Meets Human-Computer Interaction.

- Link: https://ceur-ws.org/Vol-1125/paper7.pdf
- Summary: This paper presents a weather-aware tourism recommender concept, arguing that weather can materially affect which tourist activities are appropriate.
- Methodology and data: System/design paper for weather-aware mobile tourism recommendation.
- Why it matters: This is one of the most directly related papers for your weather-aware travel angle.
- Limitation or gap: It is a recommender-system design paper, not a multi-day optimization dashboard with hotels, route alternatives, and solver transparency.
- Publication pivot: Use it to show that weather-aware tourism exists, then differentiate your work by emphasizing constrained route optimization plus inspectable tradeoffs.
- Relevance: 4/5
- Priority: important

## Synthesis: What This Means for the Project

### Strongest Publishable Framing

Frame the project as:

> A human-centered optimization dashboard for itinerary planning under contextual uncertainty, where travelers inspect and revise route recommendations by seeing explicit tradeoffs among weather risk, travel burden, budget, lodging/base choices, nature interest, and solver confidence.

This framing is stronger than:

- "a new travel recommender,"
- "a Gurobi itinerary optimizer,"
- "a weather-aware route planner,"
- "a dashboard for maps."

The publishable novelty is the combination of:

- formal route feasibility,
- contextual risk,
- route alternatives,
- user-facing explanations,
- preview versus saved optimization state,
- artifact provenance and validation.

### What the Project Already Has Compared with Prior Work

- Compared with OP/TTDP literature, it already has a richer user-facing artifact: route layers, hotels, weather risk, alternatives, audit state, and interest controls.
- Compared with recommender literature, it goes beyond ranking: it handles sequencing, time, budget, base cities, route methods, and feasibility.
- Compared with HAI guidelines, it already implements several relevant design ideas: limitations, provenance, route state, fallback status, and user controls.
- Compared with explainable recommender work, it has unusually concrete explanations because optimization components can be shown as objective terms, constraints, skipped anchors, and route alternatives.

### What Is Missing for Publication

- A study-ready interface condition and a baseline condition.
- Human-subject evidence, not just computational validation.
- A precise operational definition of user agency in this planning task.
- A study protocol that measures both objective behavior and subjective experience.
- Better handling of preview-only interactions so users understand when the route is approximate versus fully optimized.
- A clean explanation taxonomy for the dashboard: why selected, why skipped, what would change, and what is uncertain.

### Recommended Next Technical Work

- Add a study mode to the dashboard with stable tasks, reset buttons, logging, and no research/debug clutter.
- Add explanation panels for selected and skipped route elements:
  - selected because high utility / nature fit / route feasibility,
  - skipped because detour / budget / time / weather / lower objective fit.
- Add a "what changed" diff when users switch interest profiles or route lengths.
- Make preview-only routes visually and textually distinct from saved optimized routes.
- Export a compact participant log: route viewed, controls changed, stops inspected, final route accepted, and time on task.

### Recommended Next Study / Evaluation Work

- Compare two conditions:
  - baseline: ranked-list or static map with basic POI details,
  - proposed: optimization dashboard with tradeoff explanations and route alternatives.
- Use scenario tasks:
  - nature-heavy California trip,
  - rainy/weather-risk scenario,
  - budget-constrained trip,
  - low-driving relaxed traveler scenario.
- Measure:
  - perceived control,
  - trust calibration,
  - explanation usefulness,
  - workload,
  - route satisfaction,
  - ability to answer comprehension questions,
  - revision behavior and final route choice.
- Collect qualitative interview responses about confusion, agency, and overconfidence.

### Revised Related-Work Outline for the CHI Draft

1. Travel planning as constrained decision support
   - OP/TTDP surveys and rich constraints.
   - Why ranking alone is insufficient.

2. Context-aware and weather-aware itinerary recommendation
   - Tourism recommender surveys.
   - Queue/time/weather-aware POI recommendation.
   - Gap: context signals are rarely made inspectable in a route dashboard.

3. Human-AI decision support and mixed initiative planning
   - Mixed initiative systems.
   - Human-AI guidelines.
   - High automation plus high human control.

4. Explainable and controllable recommenders
   - Explanation purposes.
   - User-experience evaluation.
   - Critiquing-based recommenders.

5. Visual analytics for optimization artifacts
   - Interactive visual analysis.
   - Human-centered ML/visual analytics.
   - Gap: little work connects TTDP optimization with CHI-style user agency evaluation.

## Talking Points for Prof. Choi

- "I agree the project is around 50-60 percent complete. The technical substrate is strong, but the publication contribution needs a deeper literature-grounded pivot."
- "The strongest path seems to be CHI/HCI: not a better optimizer, but an inspectable optimization interface for travel planning under uncertainty."
- "The OR literature gives me the route-feasibility foundation, but it does not study whether travelers understand or trust the route tradeoffs."
- "The HCI literature says I should evaluate user agency, mental models, appropriate trust, explanation usefulness, and revision behavior."
- "My next milestone should be a study-ready dashboard condition, a simple baseline, and scenario-based tasks around weather risk, nature preference, budget, and driving burden."
- "I want your guidance on which gap is most publishable: mixed-initiative route planning, explanation of optimization tradeoffs, or user control under uncertain weather/context signals."

## Suggested Reading Order

1. Horvitz 1999
2. Amershi et al. 2019
3. Shneiderman 2020
4. Ruiz-Meza and Montoya-Torres 2022
5. Halder et al. 2024
6. Lim et al. 2017
7. Tintarev and Masthoff 2007
8. Knijnenburg et al. 2012
9. Chen and Pu 2012
10. Bucinca et al. 2021

After these ten, read the remaining papers to fill technical and interface-design depth.

