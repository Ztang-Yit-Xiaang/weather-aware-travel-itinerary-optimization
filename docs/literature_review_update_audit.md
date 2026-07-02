# Literature Review Update Audit

**Date:** 2026-07-02  
**Review type:** scoped narrative literature review update, not a formal PRISMA systematic review.  
**Active skills followed:** `literature-review`, `literature-review-lean`, plus `imagegen` and `pdf` where the review workflow required visual/PDF artifacts.

## Review Objective

Update the project literature review so the publication framing is no longer "LLM plus optimizer" and is instead grounded as:

> Temporal, user-specific, evidence-conflict-aware, counterfactual minimal-change repair for weather-sensitive multi-day itineraries.

The update must document the search scope, cite the closest novelty threats, synthesize them thematically, and identify what claims remain unsupported without experiments.

## Search Strategy

### Concepts Searched

1. LLM travel planning and language-to-symbolic optimization.
2. Itinerary modification, disruption revision, and minimal-change repair.
3. Whole-trip evaluation and tool-use travel benchmarks.
4. Conflict-aware retrieval and evidence-grounded planning.
5. Counterfactual and user-controllable recommendation.
6. Social-media/UGC travel information and route planning.
7. Exact-phrase checks for counterfactual/minimal-change itinerary repair.

### Sources Checked

| Source type | Coverage | Status |
| --- | --- | --- |
| Local PDF corpus | Existing `reference/*.pdf` files and generated summaries | Checked |
| arXiv | Current 2024-2026 travel-planning, RAG, and agent-planning preprints | Checked by targeted web opens/search |
| DOI/publisher metadata | From Stay to Play DOI and PDF metadata | Checked |
| Local review docs | `related_work_outline`, `core_paper_reading_cards`, deep report, addendum, evidence matrix | Checked |

### Inclusion Criteria

- English-language peer-reviewed papers or preprints.
- Direct relevance to travel planning, itinerary recommendation, route optimization, travel-agent evaluation, disruption revision, evidence conflict, or counterfactual user control.
- Papers that threaten at least one broad novelty claim or support one component of the revised repair framing.

### Exclusion Criteria

- Generic recommender or routing papers without travel/itinerary relevance.
- Product pages or blog posts without an associated paper.
- General LLM prompting papers without planning, retrieval, optimization, or travel relevance.

## Citation Verification Log

| Citation | Verification evidence | Use in review | Status |
| --- | --- | --- | --- |
| iTIMO | arXiv:2601.10609, title and abstract verified 2026-07-02 | Itinerary modification; ADD/DELETE/REPLACE novelty threat | Verified |
| VeriTrip | arXiv:2605.28683, title and abstract verified 2026-07-02 | Multi-source contradiction and evidence-grounded travel-agent benchmark | Verified |
| TP-RAG | arXiv:2504.08694, title, EMNLP reference, and abstract verified 2026-07-02 | Retrieval-augmented spatiotemporal travel planning; noisy/conflicting references | Verified |
| RAG with Conflicting Evidence | arXiv:2504.13079, title and abstract verified 2026-07-02 | Generic conflicting-evidence RAG novelty threat | Verified |
| CONFACT | arXiv:2505.17762, title and abstract verified 2026-07-02 | Generic conflicting-evidence fact-checking/RAG novelty threat | Verified |
| COMPASS | arXiv:2510.07043, title and abstract verified 2026-07-02 | Multi-turn constrained optimization in travel-planning agents | Verified |
| CostBench | arXiv:2511.02734, title and abstract verified 2026-07-02 | Cost-optimal planning and dynamic adaptation in travel-planning tool-use agents | Verified |
| TravelBench | arXiv:2512.22673, title, ACL 2026 note, and abstract verified 2026-07-02 | Multi-turn, tool-use, and unsolvable travel tasks | Verified |
| TripCraft | arXiv:2502.20508, title and abstract verified 2026-07-02 | Spatiotemporally fine-grained travel-planning evaluation | Verified |
| ITINERA | arXiv:2402.07204, title, related EMNLP DOI, and abstract verified 2026-07-02 | LLM plus spatial optimization novelty threat | Verified |
| From Stay to Play | Local PDF metadata and DOI `10.1016/j.apgeog.2016.10.002` verified 2026-07-02 | UGC/social-media travel decision-support background | Verified |
| Braunhofer weather-aware tourism citation | Existing deep-read note says source could not be reliably verified | Removed from active citation backbone | Quarantined |

## Updated Artifacts

| Artifact | Update |
| --- | --- |
| `docs/related_work_outline.md` | Added search methodology, inclusion/exclusion criteria, Mermaid gap map, AI schematic reference, and a new repair/evidence-conflict subsection |
| `docs/project_literature_evidence_matrix.md` | Added rows for minimal-change repair operations, evidence-conflict handling, counterfactual repair explanations, and UGC/social-media travel evidence |
| `docs/recent_papers_2023_2026_addendum.md` | Added iTIMO, VeriTrip, TP-RAG, ITINERA, TripCraft, TravelBench, COMPASS, CostBench, and From Stay to Play |
| `docs/novelty_claim_verification.md` | Updated novelty-threat audit and safe contribution statement |
| `docs/limitation_driven_itinerary_repair_method.md` | Reclassified From Stay to Play as verified background, not a core novelty citation |
| `docs/chi_oriented_literature_review.md` | Replaced the unverified Braunhofer citation with verified From Stay to Play |
| `docs/literature_review_slides.tex` | Replaced the unverified Braunhofer slide citation with Zhou et al. 2017 |
| `reference/1-s2.0-S0143622816306051-main_summary.md` | Added the missing paper-reading summary for From Stay to Play |
| `reference/paper_summary_index.md` | Added From Stay to Play to the summary index |
| `scripts/render_literature_review_audit_pdf.py` | Added a reproducible reportlab renderer for the audit PDF |
| `output/pdf/literature_review_update_audit.pdf` | Generated the visual PDF artifact from this audit |

## Visual Artifact

The literature-review skill requires at least one visual schematic. The current session does not expose the separate `scientific-schematics` skill, so the available `imagegen` skill was used for an AI-generated schematic and the Mermaid source remains the exact reproducible diagram.

- AI schematic: `docs/figures/literature_repair_gap_schematic.png`
- Consuming document: `docs/related_work_outline.md`
- Reproducible text version: Mermaid block in `docs/related_work_outline.md`

## Thematic Synthesis

### Theme 1: Hybrid LLM + Optimizer Planning Is Established

TTG, TRIP-PAL, LLMAP, and ITINERA show that language-to-symbolic, language-to-graph-search, and LLM-plus-spatial-optimization travel planning are already active research areas. Therefore, the project must not claim novelty for combining an LLM with an optimizer.

### Theme 2: Modification and Disruption Revision Are Established, but Mostly as Benchmarks

iTIMO formalizes itinerary modification through ADD, DELETE, and REPLACE perturbations. TripTide evaluates disruption revision and preservation/adaptability behavior. These papers make modification and disruption response central, but they do not provide the proposed solver/evaluator-backed, evidence-conflict-aware, counterfactual repair mechanism.

### Theme 3: Evaluation Is Moving Toward Complete Plans

TravelEval, TripScore, TripCraft, TravelBench, COMPASS, and CostBench all push beyond surface plausibility. They motivate whole-trip metrics, tool-use boundaries, dynamic adaptation, and constrained optimization, but they do not replace the need for a repair method that reports hard feasibility, preservation, and counterfactual tradeoffs for a repaired route.

### Theme 4: Evidence Conflict Is a Real Planning-Agent Problem

DRAGged into Conflicts, RAG with Conflicting Evidence, CONFACT, TP-RAG, and VeriTrip show that noisy, contradictory, and multi-source evidence cannot be hidden behind fluent outputs. The project should treat evidence-conflict handling as a required design element, not as a novelty claim by itself.

### Theme 5: Counterfactual Control Exists in Recommendation, but Not as Itinerary Repair

User-controllable counterfactual recommendation and critique-based recommenders motivate "what must change" explanations. The project extends this idea to constrained multi-day route repair: what threshold must relax, which stop must move, or which substitution preserves feasibility.

## Claim Boundary

**Safe claim after this update:**

> We introduce a limitation-driven itinerary repair formulation that converts confirmed user intent and frozen disruption evidence into auditable ADD/DELETE/REPLACE/MOVE/RELAX/KEEP operations, explicitly labels conflicting evidence, optimizes for minimal change under temporal and weather-sensitive constraints, and reports whole-trip feasibility plus counterfactual explanations.

**Claims still requiring experiments:**

- Repair preserves more original intent than full replanning or heuristic replacement.
- Evidence-conflict handling improves route choices under contradictory weather/closure sources.
- Counterfactual explanations improve user comprehension, control, or error detection.
- The method outperforms baselines on feasibility, risk reduction, utility retention, runtime, or lodging consistency.

**Claims to avoid:**

- LLM plus optimizer is novel.
- UGC/social-media travel data plus routing is novel.
- Evidence-conflict handling is new in general.
- The system guarantees real-world itinerary validity.
- The system is an autonomous travel agent.

## Quality Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Search methodology documented | `related_work_outline.md` and this audit | Complete |
| Inclusion/exclusion criteria stated | `related_work_outline.md` and this audit | Complete |
| Thematic synthesis rather than only study-by-study notes | `related_work_outline.md`, evidence matrix, and this audit | Complete |
| Citation verification performed | Citation verification log above; source links retained in review docs | Complete for newly added claims |
| Unverified citation quarantined | Braunhofer weather-aware tourism citation removed from active backbone | Complete |
| Visual schematic included | `docs/figures/literature_repair_gap_schematic.png` referenced in outline | Complete with available tool |
| PDF generation attempted | Beamer slide rebuild failed due missing `beamer.cls`; non-LaTeX audit PDF generated separately | Partially complete |

## Remaining Caveat

The slide PDF could not be rebuilt in the local TeX installation because `beamer.cls` is missing. The slide source was updated and failed-build byproducts were restored. A separate audit PDF is generated from this report using the bundled Python/reportlab runtime so the literature-review update still has a PDF artifact.
