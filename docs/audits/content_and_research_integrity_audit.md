# Itinerary Repair Copilot Content and Research-Integrity Baseline Audit

**Audit date:** 2026-08-03  
**Audit gate:** AUD-0  
**Disposition:** **BLOCKING — product claims exceed current evidence**  
**Auditor role:** Independent read-only content/research-integrity audit.

## Scope and evidence rules

This audit evaluates interface language, action semantics, artifact ownership,
provider claims, evidence presentation, and the product/research boundary.
Findings are grounded in the current product source and the recorded 2026-08-02
browser audit. Historical v6 dashboard verification is treated as evidence for
the frozen v6 artifact only, not for `/app`.

## Blocking findings

### CONTENT-B01 — Product verification status is overstated

**Evidence:** Source-confirmed.

- `docs/current/current_problem_manifest.md` records CP-010 as “implemented and
  locally verified; user visual sign-off pending.”
- The current Copilot plan repeats that only user visual sign-off remains.
- The current product lacks the geographic map, general edit execution, live
  provider, dedicated mobile flows, PWA, repository acceptance, and the complete
  black-box journey.

**Required wording:** CP-010 is `in-progress`. Preserve the distinction among
`implemented`, `tested`, `verified`, and `accepted`.

### CONTENT-B02 — “Map” language can conceal a route schematic

**Evidence:** Source-confirmed.

- The UI labels the route as Map and exposes Map edit.
- CSS separately discloses `ROUTE CONTEXT · SCHEMATIC`.
- No geographic library is initialized in the product application.

**Risk:** Users can reasonably infer that spatial positions and route lines are
geographic even though they are index-based.

**Required wording/behavior:** Until the geographic renderer exists, use
“route schematic” consistently and do not imply geographic validation from the
display. Once implemented, map claims must follow artifact route evidence.

### CONTENT-B03 — Compare conflates methods and distinct alternatives

**Evidence:** Source-confirmed.

- `renderCompare()` consumes `state.workspace.alternatives`, whose rows are method-
  oriented.
- Cards call themselves “Artifact-supported options,” and eligible cards say
  “Inspect option,” but the current fixture exposes one registered child.

**Risk:** Multiple methods can appear to be multiple plan choices even when they
refer to the same child or incomplete method output.

**Required wording/behavior:** A user alternative must have a distinct immutable
plan ID and content hash. Methods belong under technical evidence. Missing and
ineligible outputs remain unavailable and unranked.

### CONTENT-B04 — Inspect and Keep original have conflicting semantics

**Evidence:** Source-confirmed.

- An eligible baseline card says “Inspect option.”
- Its handler posts to `/keep-original` and the server appends a durable decision.

**Risk:** A user can create a decision record while believing they are only
reviewing details.

**Required wording/behavior:** “Inspect” must never mutate. “Keep original” requires
a separate, explicit, confirmed decision action.

### CONTENT-B05 — Copilot capability language overpromises the fixture

**Evidence:** Source-confirmed.

- The welcome message says the fixture can “help you draft a typed map change.”
- Only keyword interpretation exists; most draft operations cannot produce an
  evaluated repair.
- The provider is a deterministic local fixture, not a live LLM.

**Required wording/behavior:** Until W4, describe the adapter as a deterministic
fixture that can select the registered repair or request clarification. After W4,
show provider identity, transmitted context, permission boundary, and explicit
failure states.

### CONTENT-B06 — Preview language exceeds actual execution

**Evidence:** Source-confirmed.

- Route-feedback-only preview returns the pre-registered child.
- Other operations return `operation_not_executable_in_frozen_fixture_universe`.

**Required wording/behavior:** Separate “review registered repair” from “compile and
evaluate this draft.” Do not label the former as a newly solved result.

### CONTENT-B07 — Evidence hierarchy is technical before decision-useful

**Evidence:** Source-confirmed.

- Evidence cards surface certificate state, raw parent/child lineage, method IDs,
  route/source text, and a download.
- Monospace identifiers are visually prominent.

**Required hierarchy:** decision summary; eligibility and limitations; modeled
checks; route/context freshness; lineage/methods/metrics; technical hashes and
downloads.

## Research-integrity boundary

The following language remains mandatory:

- Product work is a parallel, non-publication track.
- The frozen v6 `dashboard_product/` may remain `verified` for E3.UX0–E3.UX4.
- `/app` work cannot advance E3.1, E3.3, E4, E3.UX5, or E5.
- “Certificate” means modeled eligibility under the recorded artifacts; it is not
  a real-world safety guarantee.
- Synthetic or frozen weather is not live weather.
- Geodesic or absent route evidence is never called road validated.
- Planner-owned values are not presented as independent evaluator judgments.
- Null and unavailable are not zero.
- Ineligible plans are inspectable but neither rankable nor acceptable.

## Truthful terminology contract

| Current or risky term | Required contract |
|---|---|
| Map (for current SVG) | Route schematic until geographic rendering exists |
| Repair generated | Registered repair loaded, unless the current draft was executed |
| Safer | Lower modeled contextual risk, with scope and evidence date |
| Alternative | Distinct plan ID and content hash |
| Inspect | Read-only presentation only |
| Keep original | Explicit durable decision |
| Copilot | Provider identity and experimental state visible |
| Evidence passed | Named modeled checks passed; limitations remain visible |
| Accepted plan | Current local application pointer, not a rewritten parent artifact |
| Verified | All named gate evidence passed, not merely source presence or HTTP 200 |

## Content defects and strengths

| ID | Severity | Finding |
|---|---|---|
| CONTENT-H01 | High | README says the new app reads `dashboard_product/`, while the service reads canonical run-root artifacts through the dashboard adapter |
| CONTENT-H02 | High | “Acceptance decisions persist locally” is true for decision files, but current acceptance is not `PlanRepository`-backed |
| CONTENT-M01 | Medium | Raw method IDs and hashes dominate Evidence readability |
| CONTENT-M02 | Medium | Missing-provider, Atlas, offline, transcript retention, and delete controls have no product copy because those behaviors are absent |
| CONTENT-M03 | Medium | The current plan’s “no live provider” and “no PWA” decisions conflict with the approved corrective direction |

Strengths to preserve:

- The interface labels itself an experimental demo.
- The About dialog states that it is not E5 participant-study evidence.
- Unsupported advanced map tools are disabled and labeled prototype-only.
- The current preview path fails closed rather than fabricating an eligible child.
- The parent/child immutability note describes the intended boundary.

## Required gate reviews

| Gate | Content review |
|---|---|
| G0 | Status and README truth; research/product separation |
| G1 | Readiness, credential, and recovery wording |
| G4 | Provider identity, permissions, transmission, errors, deterministic mode |
| G5 | Alternative identity, Evidence hierarchy, action semantics, acceptance |
| G6 | Mobile/PWA/offline/transcript content |
| G8 | Final claims match test and user-journey evidence |

## Gate recommendation

Do not retain any statement that `/app` is locally verified with only visual sign-
off pending. The correct aggregate state is `in-progress`. The frozen v6 status
must remain separate and unchanged.
