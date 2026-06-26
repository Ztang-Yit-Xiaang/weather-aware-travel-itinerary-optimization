# Sacha et al. (2017) - What You See Is What You Can Change Summary

- Local PDF: `reference/what_you_see_is_what_can_change.pdf`
- Pages: 33
- Reading date: 2026-06-26
- Source handling: generated from the local PDF text plus the existing deep-read evidence bank; long verbatim extraction requested by the supplied skill is replaced with paraphrased snapshots.

## 1. PAPER SNAPSHOT (paraphrased multi-section extraction)

### Abstract
- The paper frames ML systems often hide what users need to inspect or change.
- The abstract-level contribution is The paper argues that users can only meaningfully control model behavior when the relevant model state is visible.

### Introduction
- The introduction motivates the work through hidden model assumptions reduce effective human control.
- The stated research question is How can interactive visualization let humans inspect and modify ML processes?

### Method
- The method is conceptual framework and literature synthesis of VA/ML interactions.
- The work differentiates itself through framework connecting ML pipeline stages, visualization, and human-centered interaction.

### Conclusion
- The main takeaway is visual interfaces are the lens between analysts and ML models; useful systems let users observe, interpret, validate, and refine model behavior.
- For this project, the usable lesson is how the paper informs weather-aware itinerary optimization.
- The limit to preserve is Project-team inference - the principle is developed for machine-learning visual analytics and is conceptual rather than validated for constrained route optimization.

## 2. 150-WORD SUMMARY

Sacha et al. (2017) - What You See Is What You Can Change examines human-centered machine learning through interactive visualization. Its central question is How can interactive visualization let humans inspect and modify ML processes? The problem matters because hidden model assumptions reduce effective human control. The authors use conceptual framework and literature synthesis of VA/ML interactions. Its main contribution is framework connecting ML pipeline stages, visualization, and human-centered interaction. The conclusion is visual interfaces are the lens between analysts and ML models; useful systems let users observe, interpret, validate, and refine model behavior. The main caution is Project-team inference - the principle is developed for machine-learning visual analytics and is conceptual rather than validated for constrained route optimization. It helps separate established background from the contribution the project can credibly claim. It also identifies where route feasibility, evidence, user control, or evaluation must be made explicit.

Word count: 150

## 3. NOVELTY & CONTEXT

### Key Prior Works
- VA and interactive ML both include feedback loops.

### Core Contribution & Differentiation
- Core contribution: framework connecting ML pipeline stages, visualization, and human-centered interaction.
- Differentiation: Project-team inference - the principle is developed for machine-learning visual analytics and is conceptual rather than validated for constrained route optimization.
- Defensible project use: Supporting design citation for making route factors both visible and changeable. It cannot support claims about the usability or correctness of the project's controls.

### Accessibility
- None / Not Accessible. No paper-produced public code, dataset, or demo asset was found in the local PDF text or the web verification pass.

## 4. FINDINGS GROUNDED IN EXAMPLES

### Finding 1
- Claim: visual interfaces are the lens between analysts and ML models; useful systems let users observe, interpret, validate, and refine model behavior.
- Example: conceptual framework and literature synthesis of VA/ML interactions.
- How the example makes it tangible: it points to the paper's concrete evidence rather than leaving the contribution at the level of a broad claim.

### Finding 2
- Claim: Supporting design citation for making route factors both visible and changeable. It cannot support claims about the usability or correctness of the project's controls.
- Example: Ensure every adjustable preference has a corresponding visible route metric and every displayed constraint exposes whether the user can change it.
- How the example makes it tangible: it translates the reading into a specific design, evaluation, or citation decision for the weather-aware itinerary project.

### Project Status Note
- partially implemented - interest and method controls exist, but several visible route factors cannot trigger a saved solver-backed revision.
