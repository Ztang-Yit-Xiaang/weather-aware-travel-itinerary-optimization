# Tintarev and Masthoff (2007) - A Survey of Explanations in Recommender Systems Summary

- Local PDF: `reference/A_Survey_of_Explanations_in_Recommender_Systems.pdf`
- Pages: 10
- Reading date: 2026-06-26
- Source handling: generated from the local PDF text plus the existing deep-read evidence bank; long verbatim extraction requested by the supplied skill is replaced with paraphrased snapshots.

## 1. PAPER SNAPSHOT (paraphrased multi-section extraction)

### Abstract
- The paper frames explanations are often discussed without clear goals.
- The abstract-level contribution is The paper gives a vocabulary for deciding what route explanations are supposed to achieve.

### Introduction
- The introduction motivates the work through an explanation that persuades may not help users make better decisions.
- The stated research question is What purposes can explanations serve in recommender systems, and how can they be evaluated?

### Method
- The method is survey and taxonomy of explanation aims, presentation types, and interaction modes.
- The work differentiates itself through seven explanation aims and evaluation distinctions.

### Conclusion
- The main takeaway is explanation design must be evaluated according to its intended aim, such as transparency, scrutability, trust, effectiveness, persuasiveness, efficiency, or satisfaction.
- For this project, the usable lesson is how the paper informs weather-aware itinerary optimization.
- The limit to preserve is Project-team inference - the survey predates modern explainable AI, LLMs, and interactive map systems, and much of its evidence is example-based.

## 2. 150-WORD SUMMARY

Tintarev and Masthoff (2007) - A Survey of Explanations in Recommender Systems examines explanation goals in recommender systems. Its central question is What purposes can explanations serve in recommender systems, and how can they be evaluated? The problem matters because an explanation that persuades may not help users make better decisions. The authors use survey and taxonomy of explanation aims, presentation types, and interaction modes. Its main contribution is seven explanation aims and evaluation distinctions. The conclusion is explanation design must be evaluated according to its intended aim, such as transparency, scrutability, trust, effectiveness, persuasiveness, efficiency, or satisfaction. The main caution is Project-team inference - the survey predates modern explainable AI, LLMs, and interactive map systems, and much of its evidence is example-based. It helps separate established background from the contribution the project can credibly claim. It also identifies where route feasibility, evidence, user control, or evaluation must be made explicit.

Word count: 151

## 3. NOVELTY & CONTEXT

### Key Prior Works
- expert systems and recommenders used explanations.

### Core Contribution & Differentiation
- Core contribution: seven explanation aims and evaluation distinctions.
- Differentiation: Project-team inference - the survey predates modern explainable AI, LLMs, and interactive map systems, and much of its evidence is example-based.
- Defensible project use: Foundational explanation citation for separating transparency, scrutability, effectiveness, and trust-related goals. It cannot demonstrate that the proposed route explanations achieve those goals.

### Accessibility
- None / Not Accessible. No paper-produced public code, dataset, or demo asset was found in the local PDF text or the web verification pass.

## 4. FINDINGS GROUNDED IN EXAMPLES

### Finding 1
- Claim: explanation design must be evaluated according to its intended aim, such as transparency, scrutability, trust, effectiveness, persuasiveness, efficiency, or satisfaction.
- Example: survey and taxonomy of explanation aims, presentation types, and interaction modes.
- How the example makes it tangible: it points to the paper's concrete evidence rather than leaving the contribution at the level of a broad claim.

### Finding 2
- Claim: Foundational explanation citation for separating transparency, scrutability, effectiveness, and trust-related goals. It cannot demonstrate that the proposed route explanations achieve those goals.
- Example: Give each explanation panel a declared user task and metric: understand selection, diagnose skipping, identify a feasible change, or judge uncertainty.
- How the example makes it tangible: it translates the reading into a specific design, evaluation, or citation decision for the weather-aware itinerary project.

### Project Status Note
- partially implemented - basic why-selected text and nature-region reason codes exist; generalized why-skipped and what-would-change explanations are incomplete.
