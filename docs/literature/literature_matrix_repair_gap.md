# Literature Matrix: Repair Gap

Search date: 2026-07-04

Purpose: preserve the compact repair-gap matrix as a current-project artifact. This file keeps the scanning style of the original matrix while adding a short column guide and a full reference key so the table is easier to defend in a report or meeting.

Evidence levels use the engineering/CS hierarchy from the literature-evidence-synthesis skill: E1 industrial deployment, E2 multi-site experiment, E3 replication, E4 controlled experiment, E5 benchmark/simulation, E6 case study, E7 proof of concept/demo, Review for survey papers.

## How to Read This Matrix

| Column | Meaning |
|---|---|
| Citation | Short author-year key. Full title and venue are listed in the Reference Key below. |
| Stream | Literature family, such as OP/TTDP, LLM travel planning, dynamic routing, or explainable optimization. |
| Research question | Main question the paper answers. |
| Method | Study type or technical approach. |
| Data/source | Evidence base used by the paper. |
| Key contribution | Main result or conceptual contribution. |
| Repair relation | How the work relates to itinerary repair: backbone, adjacent precedent, direct threat, or gap evidence. |
| Ownership labels | Whether the work models locked, booked, strong, weak, or flexible user commitments. |
| Parent-child diff | Whether the work explicitly compares an accepted parent itinerary with a repaired child itinerary. |
| Progressive neighborhoods | Whether repair starts locally and expands only when necessary. |
| Lexicographic objectives | Whether objectives are ordered, for example preservation first, feasibility second, utility later. |
| Independent certification | Whether the final plan is checked by a separate solver, route validator, planner, benchmark oracle, or certificate artifact. |
| Explanation grounding | Whether explanations are tied to plan diffs, constraints, solver outcomes, route/weather/hotel evidence, or counterfactual artifacts. |
| Evidence level | Strength/type of evidence using the E1-E7 hierarchy. |
| Citation priority | A = core citation, B = supporting citation, C = optional/background. |

## Matrix

| # | Citation | Stream | Research question | Method | Data/source | Key contribution | Repair relation | Ownership labels | Parent-child diff | Progressive neighborhoods | Lexicographic objectives | Independent certification | Explanation grounding | Evidence level | Citation priority |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Vansteenwegen et al. 2011 | OP/TTDP | How is the OP modeled and solved? | Survey | OP literature | Establishes OP variants and algorithms. | Backbone only. | No | No | Local search variants, not parent repair. | No | No | No | Review | A |
| 2 | Gunawan et al. 2016 | OP/TTDP | What recent OP variants and applications exist? | Survey | OP literature | Updates variants, solution approaches, applications. | Shows maturity of generation/reoptimization. | No | No | Heuristics/metaheuristics, not ownership-aware repair. | No | No | No | Review | A |
| 3 | Ruiz-Meza and Montoya-Torres 2022 | TTDP | What TTDP extensions and future lines exist? | Systematic review | TTDP literature | Taxonomy of objectives, constraints, methods. | Supports gap that TTDP emphasizes design/generation. | Rare/No | Rare/No | No explicit repair semantics. | No | No | No | Review | A |
| 4 | Halder et al. 2024 | Personalized itinerary recommendation | How did itinerary recommendation evolve from optimization to deep learning? | Survey | Recommender/optimization literature | Bridges OR, ML, and deep learning. | Helps position project against generation/personalization. | No | No | No | No | No | Limited | Review | A |
| 5 | Borras et al. 2014 | Tourism recommender systems | What intelligent tourism recommender systems exist? | Survey | Tourism RS literature | Context, AI, user modeling taxonomy. | Supports user/context features. | No | No | No | No | No | Some recommender explanation context | Review | B |
| 6 | Braunhofer et al. 2013 | Weather-aware tourism RS | Can weather improve tourism recommendations? | Prototype/recommender | Tourism context and weather | Weather-aware mobile recommender concept. | Supports weather as contextual feature. | No | No | No | No | No | Limited | E7 | B |
| 7 | Lim et al. 2017 | Queue-aware itinerary recommendation | Can queueing time be included in itinerary recommendation? | Algorithm/experiment | POI and queueing data | QueueTourRec with time-aware recommendation. | Shows context-aware utility beyond static POI score. | No | No | No | Constraint/utility objective, not preservation-first. | No | Limited | E5 | A |
| 8 | Quercia et al. 2014 | Pleasant/scenic routes | Can routes be optimized for beauty/quiet/happiness? | Data-driven route recommendation | Crowdsourcing, Flickr proxies | Optimizes pleasant routes with small extra travel cost. | Supports scenic/pleasantness precedent; not novelty. | No | No | No | Multi-criteria route tradeoff. | No | User-facing rationale but not repair diff. | E5/E6 | B |
| 9 | Pillac et al. 2013 | Dynamic VRP | How are dynamic vehicle routing problems classified? | Review | DVRP literature | Taxonomy of dynamism, online routing, reoptimization. | Supports disruption framing. | No | No | Dynamic reoptimization, not itinerary ownership. | No | No | No | Review | A |
| 10 | Psaraftis et al. 2016 | Dynamic VRP | What changed after three decades of DVRP? | Review | DVRP literature | Updated taxonomy and research agenda. | Supports adaptive routing background. | No | No | No | No | No | No | Review | A |
| 11 | Nazari et al. 2018 | RL routing | Can RL solve VRP instances? | Policy-gradient model | Synthetic VRP instances | Learns real-time route construction policy. | Algorithmic contrast: learned generation, not repair. | No | No | No | Reward optimization, not lexicographic preservation. | Feasibility rules, not independent travel certificate. | No | E5 | B |
| 12 | Kool et al. 2019 | Learned routing heuristics | Can attention models solve routing problems including OP? | Neural heuristic | Synthetic routing benchmarks | Strong learned heuristic across routing variants. | Useful baseline family, not human repair. | No | No | No | No | Benchmark feasibility only. | No | E5 | B |
| 13 | Gama and Fernandes 2021 | RL OPTW | Can pointer networks solve OPTW with tourist variation? | RL / pointer network | OPTW benchmark instances | Applies RL to tourist-relevant OP with time windows. | Close algorithmic neighbor, still generation. | No | No | No | No | Benchmark feasibility only. | No | E5 | A |
| 14 | TravelPlanner / Xie et al. 2024 | LLM travel benchmark | Can language agents plan complex trips? | Benchmark | 1,225 intents, large travel sandbox | Shows LLM agents struggle with complex constraints. | Motivates solver-certified planning. | User constraints, not ownership levels | No typed parent-child repair | No | Evaluates generated plans | Tool/database checks | Textual/agent traces | E5 | A |
| 15 | TTG / Ju et al. 2024 | LLM + MILP | Can NL requests be translated to guaranteed travel plans? | Fine-tuned LLM + MILP | Synthetic symbolic data and travel stats | NL-to-symbolic solver architecture. | Strong precedent for LLM as compiler. | Constraint strengths, not accepted-plan ownership | No | No | Cost/feasibility optimization | Solver guarantee | Some solver grounding | E5/E7 | A |
| 16 | TRIP-PAL / de la Rosa et al. 2024 | LLM + automated planner | Can LLMs and planners provide guaranteed trip plans? | LLM + planning | Planning domain artifacts | Combines language and automated planning. | Supports bounded LLM role. | Limited | No | No | Planner objective | Planner guarantees | Planner-grounded | E5/E7 | B |
| 17 | ITINERA / Tang et al. 2024 | Spatial optimization + LLM | Can spatial optimization improve open-domain urban itinerary planning? | LLM + spatial optimization | Urban POI data | Connects open-domain language to spatial optimization. | Adjacent to solver-backed generation. | No | No | No | Utility/constraint objective | Solver/optimization checks | Some generated explanation | E5 | B |
| 18 | iTIMO / Huang et al. 2026 | Itinerary modification | Can a dataset support itinerary modification research? | Synthetic dataset | Real-world itineraries perturbed by LLMs | Defines itinerary modification with ADD/DELETE/REPLACE edits. | Very relevant; makes "first modification" unsafe. | No ownership hierarchy | Atomic edits, but not solver-certified parent-child minimality | No | No | Evaluation metric, not route certificate | Limited | E5/preprint | A |
| 19 | TripTide / Karmakar et al. 2025 | Disruption benchmark | Can LLMs adapt plans under disruptions? | Benchmark | Simulated travel disruptions | Evaluates preservation of intent, responsiveness, adaptability. | Very relevant; makes "first disruption benchmark" unsafe. | Traveler tolerance, not ownership commitments | Semantic/spatial/sequential divergence, not typed edit cost | No | No | LLM/manual evaluation, not independent route certificate | Judge/manual explanations | E5/preprint | A |
| 20 | TripScore / Qu et al. 2025 | Travel plan evaluation | How to score real-world travel planning? | Benchmark/reward | 4,870 queries, expert annotations | Fine-grained reward and RL signal for travel planning. | Useful evaluation contrast. | No | No | No | Unified reward, not repair lexicographic objective | Evaluation model | Limited | E5/preprint | B |
| 21 | TravelEval / Chen et al. 2026 | LLM travel evaluation | How to benchmark LLM-powered travel planning agents? | Benchmark | Travel planning tasks | Multi-dimensional evaluation of LLM planning. | Useful external validation lens. | No | No | No | Evaluation, not repair solver. | Benchmark checks | Limited | E5/preprint | B |
| 22 | Horvitz 1999 | Mixed initiative | How should user and system share initiative? | HCI principles | Conceptual/HCI examples | Establishes mixed-initiative design principles. | Supports confirmation and user control. | User control concept, not itinerary ownership schema | No | No | No | No | Interaction principles | Conceptual | B |
| 23 | OptiChat / Chen et al. 2025 | Explainable optimization | Can LLMs help practitioners interpret optimization models? | LLM + function calls/code | Optimization explanation dataset | Infeasibility, sensitivity, modification, counterfactual explanation. | Strong explanation precedent. | No itinerary ownership | Modification analysis, not typed plan diff | No | No | Optimization model evidence | Strong optimization-grounding | E5/E7 | A |
| 24 | CLEMO / Otto et al. 2025 | Explainable optimization | Can local explanations be coherent with optimization structure? | Sampling-based explanation | Shortest path, knapsack, VRP experiments | Structure-aware explanations for objectives and variables. | Supports objective/variable explanation. | No | No itinerary diff | No | No | Model-structure coherence | Strong optimization-grounding | E5/preprint | A |
| 25 | Current repository baseline | Weather-aware itinerary prototype | Can a California itinerary demo combine POI utility, nature, weather, hotels, and routes? | Python optimization/prototype | Curated California seed, optional live APIs, cached snapshots | Rich feature model and route generation artifacts. | Useful baseline to convert into repair testbed. | Not implemented | Not implemented | Not implemented | Multi-objective route objective exists, but not repair lexicographic stages | Partial; route oracle currently falls back to geodesic solver | Limited dashboards/explanations | E7 | A |
| 26 | Proposed method | Ownership-aware minimal itinerary repair | How can a user-owned accepted plan be minimally repaired after disruptions? | Typed edit model + progressive neighborhoods + lexicographic optimization + independent certification | Parent plans, disruptions, route/weather/hotel snapshots, validation artifacts | Treats preservation as first-class objective and explains changes from evidence. | Central contribution. | Yes | Yes | Yes | Yes | Yes | Yes | Target E4/E5 | A |

## Gap Summary

The strongest empty matrix cells are the intersection of:

- Itinerary repair after localized disruption.
- Explicit parent-child plan lineage.
- Ownership-aware locked/booked/strong/weak/flexible commitments.
- Typed edit costs across move, replace, delete, add, reorder, hotel change, route change, and time-shift operations.
- Progressive local repair neighborhoods before broader reoptimization.
- Sequential lexicographic objectives that put preservation before utility.
- Independent route-valid certification before a plan is displayed or counted in final evaluation.
- Evidence-grounded explanations tied to diff, solver, route, weather, hotel, and counterfactual artifacts.

This gap is methodological and application-oriented. It is not a claim that no one has studied itinerary modification or disruption response; iTIMO and TripTide make that claim unsafe. The defensible contribution is the combination of ownership-aware repair semantics, optimization stages, certification, and explanation evidence in one itinerary repair framework.

## Reference Key

- Borras, J., Moreno, A., and Valls, A. (2014). Intelligent tourism recommender systems: A survey. Expert Systems with Applications, 41(16), 7370-7389. DOI: 10.1016/j.eswa.2014.06.007
- Braunhofer, M., Elahi, M., Ge, M., Ricci, F., and Schievenin, T. (2013). STS: Design of Weather-Aware Mobile Recommender Systems in Tourism. AI*HCI@AI*IA / CEUR Workshop Proceedings.
- Chen, H. et al. (2025). OptiChat: Bridging Optimization Models and Practitioners with Large Language Models. INFORMS Journal on Data Science. DOI: 10.1287/ijds.2025.0074
- Chen, W. et al. (2026). TravelEval: A Comprehensive Benchmarking Framework for Evaluating LLM-Powered Travel Planning Agents. arXiv:2606.01046.
- de la Rosa, T., Gopalakrishnan, S., Pozanco, A., Zeng, Z., and Borrajo, D. (2024). TRIP-PAL: Travel Planning with Guarantees by Combining Large Language Models and Automated Planners. arXiv:2406.10196.
- Gama, R., and Fernandes, H. L. (2021). A reinforcement learning approach to the orienteering problem with time windows. Computers & Operations Research, 133, 105357. DOI: 10.1016/j.cor.2021.105357
- Gunawan, A., Lau, H. C., and Vansteenwegen, P. (2016). Orienteering Problem: A survey of recent variants, solution approaches and applications. European Journal of Operational Research, 255(2), 315-332. DOI: 10.1016/j.ejor.2016.04.059
- Halder, S., Lim, K. H., Chan, J., and Zhang, X. (2024). A survey on personalized itinerary recommendation: From optimisation to deep learning. Applied Soft Computing, 152, 111200. DOI: 10.1016/j.asoc.2023.111200
- Horvitz, E. (1999). Principles of mixed-initiative user interfaces. CHI 1999. DOI: 10.1145/302979.303030
- Huang, Z., Ma, Y., Zhang, H., Ma, H., and Sun, Z. (2026). iTIMO: An LLM-empowered Synthesis Dataset for Travel Itinerary Modification. arXiv:2601.10609.
- Ju, D. et al. (2024). To the Globe (TTG): Towards Language-Driven Guaranteed Travel Planning. EMNLP 2024 System Demonstrations, 240-249.
- Karmakar, P., Chaudhuri, S., Mallick, S., Gupta, M., Jana, A., and Ghosh, S. (2025). TripTide: A Benchmark for Adaptive Travel Planning under Disruptions. arXiv:2510.21329.
- Kool, W., van Hoof, H., and Welling, M. (2019). Attention, Learn to Solve Routing Problems! ICLR 2019.
- Lim, K. H., Chan, J., Karunasekera, S., and Leckie, C. (2017). Personalized Itinerary Recommendation with Queuing Time Awareness. SIGIR 2017, 325-334. DOI: 10.1145/3077136.3080778
- Nazari, M., Oroojlooy, A., Snyder, L., and Takac, M. (2018). Reinforcement Learning for Solving the Vehicle Routing Problem. NeurIPS 2018.
- Otto, D., Kurtz, J., and Birbil, S. I. (2025). Coherent Local Explanations for Mathematical Optimization. arXiv:2502.04840.
- Pillac, V., Gendreau, M., Gueret, C., and Medaglia, A. L. (2013). A review of dynamic vehicle routing problems. European Journal of Operational Research, 225(1), 1-11. DOI: 10.1016/j.ejor.2012.08.015
- Psaraftis, H. N., Wen, M., and Kontovas, C. A. (2016). Dynamic vehicle routing problems: Three decades and counting. Networks, 67(1), 3-31. DOI: 10.1002/net.21628
- Qu, Y., Xiao, H., Li, F., Zhou, H., and Dai, X. (2025). TripScore: Benchmarking and rewarding real-world travel planning with fine-grained evaluation. arXiv:2510.09011.
- Quercia, D., Schifanella, R., and Aiello, L. M. (2014). The Shortest Path to Happiness: Recommending Beautiful, Quiet, and Happy Routes in the City. HT 2014. DOI: 10.1145/2631775.2631799
- Ruiz-Meza, J. L., and Montoya-Torres, J. R. (2022). A systematic literature review for the tourist trip design problem: Extensions, solution techniques and future research lines. Operations Research Perspectives. DOI: 10.1016/j.orp.2022.100228
- Tang, J. et al. (2024). ITINERA: Integrating Spatial Optimization with LLMs for Open-domain Urban Itinerary Planning. EMNLP 2024 Industry Track.
- TravelPlanner: Xie, J. et al. (2024). TravelPlanner: A Benchmark for Real-World Planning with Language Agents. arXiv:2402.01622.
- Vansteenwegen, P., Souffriau, W., and Van Oudheusden, D. (2011). The orienteering problem: A survey. European Journal of Operational Research, 209(1), 1-10. DOI: 10.1016/j.ejor.2010.03.045
