# Literature Search Log: Ownership-Aware Itinerary Repair

Search date: 2026-07-04

Research question: What literature supports or limits a contribution framed as ownership-aware minimal itinerary repair under travel disruptions?

Review type: scoping literature synthesis with matrix-based gap analysis. This is not a full PRISMA systematic review because the Codex environment did not provide exportable Scopus, Web of Science, or Google Scholar result sets, and hit counts were not exposed for several web searches.

## Inclusion and Exclusion Criteria

Inclusion criteria:

- Tourist trip design, orienteering, itinerary recommendation, personalized travel planning, or context-aware tourism recommendation.
- Dynamic routing, adaptive routing, RL routing, or disruption response with relevance to itinerary repair.
- LLM travel planning, LLM itinerary modification, travel planning benchmarks, or planner/solver hybrids.
- Explainable optimization, mixed-initiative interaction, or recommender explanation relevant to repair explanations.
- Peer-reviewed paper, major conference paper, recognized workshop paper, publisher page, arXiv paper, or official project page.

Exclusion criteria:

- Travel blog, product page, or nontechnical itinerary generator with no research artifact.
- Papers about unrelated routing domains where no transfer to itinerary repair was clear.
- Citation metadata that could not be verified and was not needed for the matrix.
- Claims of "first" that came only from a preprint abstract and were not needed for the project's novelty framing.

## Search Surfaces and Queries

The web-search tool did not expose exact hit counts. The "hits" column therefore records "not exposed" unless a database page itself returned count information.

| # | Source surface | Query string | Filters | Hits | Result |
|---|---|---|---|---|---|
| 1 | ScienceDirect / Elsevier | `"The orienteering problem: A survey" DOI EJOR 2011` | Publisher page preferred | Not exposed | Verified Vansteenwegen et al. 2011 and DOI 10.1016/j.ejor.2010.03.045. |
| 2 | ScienceDirect / RePEc / SMU | `"Orienteering Problem: A survey of recent variants" DOI 2016` | Publisher or institutional page | Not exposed | Verified Gunawan et al. 2016 and DOI 10.1016/j.ejor.2016.04.059. |
| 3 | ScienceDirect / institutional | `"tourist trip design problem" systematic literature review 2022 DOI` | Publisher or university page | Not exposed | Verified Ruiz-Meza and Montoya-Torres 2022 and DOI 10.1016/j.orp.2022.100228. |
| 4 | ScienceDirect / ACM mirror | `"A survey on personalized itinerary recommendation: From optimisation to deep learning" DOI` | Publisher page preferred | Not exposed | Verified Halder et al. 2024 and DOI 10.1016/j.asoc.2023.111200. |
| 5 | ACM / IR Anthology | `"Personalized Itinerary Recommendation with Queuing Time Awareness" SIGIR 2017 DOI` | ACM or anthology page | Not exposed | Verified Lim et al. 2017, SIGIR, DOI 10.1145/3077136.3080778. |
| 6 | CEUR / DBLP | `"STS: Design of Weather-Aware Mobile Recommender Systems in Tourism" CEUR` | Workshop PDF or DBLP | Not exposed | Verified Braunhofer et al. 2013 workshop record. |
| 7 | ACM / arXiv | `"The Shortest Path to Happiness" "Beautiful, Quiet, and Happy Routes" DOI` | ACM and arXiv | Not exposed | Verified Quercia et al. 2014 and DOI 10.1145/2631775.2631799. |
| 8 | ScienceDirect / RePEc | `"A review of dynamic vehicle routing problems" DOI Pillac Gendreau Gueret Medaglia 2013` | Publisher or RePEc | Not exposed | Verified Pillac et al. 2013 and DOI 10.1016/j.ejor.2012.08.015. |
| 9 | Wiley / institutional pages | `"Dynamic vehicle routing problems: Three decades and counting" DOI` | Publisher page preferred | Not exposed | Verified Psaraftis et al. 2016 and DOI 10.1002/net.21628. |
| 10 | NeurIPS / arXiv | `"Reinforcement Learning for Solving the Vehicle Routing Problem" NeurIPS 2018 Nazari` | Proceedings page preferred | Not exposed | Verified NeurIPS 2018 paper and authors. |
| 11 | OpenReview / arXiv / DBLP | `"Attention, Learn to Solve Routing Problems" ICLR 2019` | OpenReview/DBLP | Not exposed | Verified ICLR 2019 routing paper. |
| 12 | ScienceDirect / arXiv | `"A reinforcement learning approach to the orienteering problem with time windows" DOI` | Publisher page preferred | Not exposed | Verified Gama and Fernandes 2021 and DOI 10.1016/j.cor.2021.105357. |
| 13 | arXiv / project page | `"TravelPlanner" "A Benchmark for Real-World Planning with Language Agents" arXiv 2402.01622` | arXiv and project page | Not exposed | Verified TravelPlanner metadata and constraints benchmark scope. |
| 14 | ACL Anthology / arXiv | `"To the Globe" "Language-Driven Guaranteed Travel Planning" EMNLP 2024` | ACL page preferred | Not exposed | Verified TTG EMNLP Demo 2024 citation. |
| 15 | arXiv | `"TRIP-PAL" "Travel Planning with Guarantees" LLM automated planners` | arXiv PDF/abstract | Not exposed | Verified TRIP-PAL arXiv:2406.10196. |
| 16 | ACL Anthology / arXiv | `"ITINERA" "Integrating Spatial Optimization with LLMs" EMNLP Industry 2024` | ACL or arXiv | Not exposed | Verified ITINERA paper title and venue stream. |
| 17 | arXiv | `"iTIMO" "An LLM-empowered Synthesis Dataset" travel itinerary modification` | arXiv | Not exposed | Verified iTIMO arXiv:2601.10609 and edit operations ADD/DELETE/REPLACE. |
| 18 | arXiv / ACL PDF | `"TripTide" itinerary modification disruption benchmark arXiv 2510.21329` | arXiv | Not exposed | Verified TripTide arXiv:2510.21329 and disruption-adaptation metrics. |
| 19 | arXiv / OpenReview | `"TripScore" travel itinerary evaluation arXiv 2510.09011` | arXiv/OpenReview | Not exposed | Verified TripScore arXiv:2510.09011. |
| 20 | arXiv | `"TravelEval" "travel itinerary" arXiv 2606.01046` | arXiv | Not exposed | Verified TravelEval arXiv:2606.01046. |
| 21 | INFORMS / arXiv | `"OptiChat" "Bridging Optimization Models and Practitioners" DOI` | Publisher preferred | Not exposed | Verified OptiChat and DOI 10.1287/ijds.2025.0074. |
| 22 | arXiv / Semantic Scholar | `"Coherent Local Explanations for Mathematical Optimization" arXiv 2502.04840` | arXiv | Not exposed | Verified CLEMO arXiv:2502.04840. |
| 23 | ACM | `"Principles of mixed-initiative user interfaces" Horvitz DOI` | ACM | Not exposed | Verified Horvitz 1999 and DOI 10.1145/302979.303030. |
| 24 | Local repository docs | `rg -n "repair|novelty|literature|score|weather|utility"` | Repo documents and code | Local search | Used to audit existing claims, scoring formulas, and technical gaps. |

## Screening Counts

Because the search tool did not expose database-level hit counts, counts below refer to records manually screened from returned search results, local literature docs, and citation chaining.

| Stage | Count | Notes |
|---|---:|---|
| Records/candidates screened | 46 | Across OP/TTDP, context/weather recommendation, dynamic/RL routing, LLM travel, explainable optimization, and local docs. |
| Duplicate or less relevant records removed | 10 | Mostly mirrors, PDFs, secondary pages, or broad adjacent routing papers. |
| Records retained for matrix or citation support | 31 | Includes verified core papers plus recent arXiv benchmark papers. |
| Core papers emphasized in review | 23 | Used directly in synthesis and references. |
| Items marked secondary/unverified | 5 | Kept as watchlist only; not used for strong claims. |

## Included Core Papers

| Stream | Included papers |
|---|---|
| OP/TTDP and itinerary recommendation | Vansteenwegen et al. 2011; Gunawan et al. 2016; Ruiz-Meza and Montoya-Torres 2022; Halder et al. 2024; Borras et al. 2014; Lim et al. 2017. |
| Context, weather, scenic, queueing | Braunhofer et al. 2013; Quercia et al. 2014; Lim et al. 2017; Porras et al. 2022 was noted but not central. |
| Dynamic/RL routing | Pillac et al. 2013; Psaraftis et al. 2016; Nazari et al. 2018; Kool et al. 2019; Gama and Fernandes 2021. |
| LLM planning, modification, evaluation | TravelPlanner/Xie et al. 2024; TTG/Ju et al. 2024; TRIP-PAL/de la Rosa et al. 2024; ITINERA/Tang et al. 2024; iTIMO/Huang et al. 2026; TripTide/Karmakar et al. 2025; TripScore/Qu et al. 2025; TravelEval/Chen et al. 2026. |
| Explainability and mixed initiative | Horvitz 1999; OptiChat/Chen et al. 2025; CLEMO/Otto et al. 2025; recommender explanation and critiquing surveys from local bibliography were used as background. |

## Excluded or Watchlist Items

| Item | Decision | Reason |
|---|---|---|
| Broad travel assistant demos with no solver/evaluation artifact | Excluded | Not useful for defensible literature gap. |
| Recent LLM travel benchmarks not directly about modification or constraints | Watchlist | Could help future related work but not central. |
| Product/service pages for itinerary planning | Excluded | Not peer-reviewed and no reproducible method. |
| Routing papers without user-owned plans or disruption repair semantics | Mostly excluded | Relevant to algorithms but too far from research gap. |
| Local project claims in older docs | Used cautiously | Good for internal continuity, not external evidence. |

## Citation Verification Notes

Verified DOI or primary record:

- Vansteenwegen et al. 2011: 10.1016/j.ejor.2010.03.045.
- Gunawan et al. 2016: 10.1016/j.ejor.2016.04.059.
- Ruiz-Meza and Montoya-Torres 2022: 10.1016/j.orp.2022.100228.
- Halder et al. 2024: 10.1016/j.asoc.2023.111200.
- Borras et al. 2014: 10.1016/j.eswa.2014.06.007.
- Lim et al. 2017: 10.1145/3077136.3080778.
- Quercia et al. 2014: 10.1145/2631775.2631799.
- Pillac et al. 2013: 10.1016/j.ejor.2012.08.015.
- Psaraftis et al. 2016: 10.1002/net.21628.
- Gama and Fernandes 2021: 10.1016/j.cor.2021.105357.
- Horvitz 1999: 10.1145/302979.303030.
- OptiChat 2025: 10.1287/ijds.2025.0074.

Verified arXiv/primary record but no journal DOI asserted:

- TravelPlanner: arXiv:2402.01622; ICML metadata found through Semantic Scholar.
- TRIP-PAL: arXiv:2406.10196.
- TTG: ACL Anthology EMNLP 2024 System Demonstrations page; arXiv:2410.16456.
- iTIMO: arXiv:2601.10609.
- TripTide: arXiv:2510.21329.
- TripScore: arXiv:2510.09011.
- TravelEval: arXiv:2606.01046.
- CLEMO: arXiv:2502.04840.

Unverified or secondary only:

- Some newest LLM travel benchmark venues after June 2026 were not deeply checked.
- Local bibliography entries for recommender explanation surveys were used only as background unless citation metadata was already present in local docs.

## Search Limitations

- The web-search tool does not provide reproducible hit counts.
- Scopus, Web of Science, and Google Scholar exports were not available.
- Publisher pages sometimes expose metadata but not full text.
- Recent 2025-2026 LLM travel papers are moving targets; arXiv versions and conference status may change.
- The search was English-focused.

## Reproducible Query Blocks for Future Update

Use these blocks in Scopus/Web of Science/Google Scholar if available:

```text
("tourist trip design problem" OR "orienteering problem" OR "itinerary recommendation")
AND (repair OR reoptimization OR disruption OR modification OR "plan revision")
```

```text
("travel itinerary" OR "tourist itinerary")
AND ("large language model" OR LLM OR "language agent")
AND (modification OR disruption OR repair OR benchmark OR evaluation)
```

```text
("dynamic vehicle routing" OR "adaptive routing" OR "reinforcement learning routing")
AND (repair OR reoptimization OR disruption OR "local search")
```

```text
("explainable optimization" OR "optimization explanation" OR "mixed-initiative")
AND (routing OR itinerary OR travel OR "vehicle routing")
```

```text
("weather-aware" OR "context-aware" OR scenic OR queueing)
AND ("tourism recommender" OR "itinerary recommendation" OR "route recommendation")
```
