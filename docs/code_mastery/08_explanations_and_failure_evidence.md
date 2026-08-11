# Explanations and Failure Evidence

## `ExplanationEvidenceBuilder`

1. **Why:** create claims whose references can be inspected.
2. **Category:** stateless evidence service.
3. **Called by:** pipeline repair adapters and explanation helpers.
4. **Inputs:** `PlanDiff`, certificate, route evidence, counterfactual records.
5. **Outputs:** `ExplanationEvidence` with claims and evidence records.
6. **State ownership:** serialized under the run's `explanations/`.
7. **Invariants:** numerical/causal claims cite allowed, existing evidence IDs.
8. **Failure:** unsupported claims are marked unsafe/omitted from publication.
9. **Tests:** `tests/explanation/test_evidence.py`,
   `test_evidence_builder.py`.
10. **Gate:** E3/G7.
11. **State:** current.
12. **Read next:** `explanation/counterfactual.py`.

## `CounterfactualRunner`

- **Why/category:** sandbox service for why-not/what-if evidence.
- **Caller:** explanation and optional interaction flows.
- **Inputs/outputs:** parent plus forced/changed request and injected repair
  executor in; counterfactual run record out.
- **State:** sandbox child request; parent must remain unchanged.
- **Failure:** solver error or parent mutation becomes failure evidence.
- **Tests:** `tests/explanation/test_counterfactual.py`.

## `DeterministicTemplateVerbalizer`

- **Why/category:** deterministic compatibility-safe presentation service.
- **Caller:** benchmark-default explanation rendering.
- **Inputs/outputs:** structured supported evidence in; rendered text plus
  claim-to-evidence map out.
- **Invariant:** hides unsupported claims; no free-form evidence invention.
- **Tests/gate/state:** counterfactual tests; E3; current.

## Failure Is Evidence

Failed methods stay visible with requested/executed IDs, status, refusal or
physical reason, route lineage where applicable, and absence of a certificate.
The UI must not turn failure into a missing row or successful alternative.

See [explanation grounding](diagrams/explanation_grounding.md).

> **Beginner note / 初学者提示:** An explanation is not grounded because it
> sounds reasonable. It is grounded when each claim points to stored evidence
> that supports it.

