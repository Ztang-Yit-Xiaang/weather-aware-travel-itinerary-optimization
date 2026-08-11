# Itinerary Repair Copilot AUD-0 Synthesis

**Synthesis date:** 2026-08-03  
**Gate:** AUD-0 baseline  
**Aggregate disposition:** **BLOCKING — CP-010 must remain `in-progress`**

## Decision

The `/app` codebase contains a substantive local product shell: pinned registry,
FastAPI boundary, token/revision checks, fixture Copilot, typed draft records,
fail-closed preview behavior, evidence ZIP allow-listing, local decision files,
and an atomic pointer. Those foundations are worth preserving.

They do not establish product verification. The three independent baseline audits
agree that the geographic map, general repair execution, real Copilot lifecycle,
distinct Compare alternatives, repository-backed acceptance, dedicated mobile/PWA
flows, and replacement-level verification remain incomplete. The existing
“implemented and locally verified; user visual sign-off pending” claim is not
supported.

## Audit provenance

| Report | Role | Disposition |
|---|---|---|
| `uiux_audit_report.md` | UI/UX and mockup correspondence | Blocking |
| `content_and_research_integrity_audit.md` | Truthfulness and research boundary | Blocking |
| `web_frontend_implementation_audit.md` | Source, service, state, handlers, tests | Blocking |
| This synthesis | Cross-audit reconciliation and gate routing | Blocking |

Evidence classes are kept separate:

- Recorded browser observations from the 2026-08-02 audit.
- Source/artifact inspection performed on 2026-08-03.
- Behaviors explicitly covered by focused tests.

No audit treats source presence, HTTP 200, or a screenshot as proof of a complete
user journey.

## Cross-audit blocking register

| Finding | Severity | Agreement | Required gate |
|---|---|---|---|
| SYN-B01: `/app` uses a schematic rather than a geographic map | Critical | UIUX, Content, Web | G2 |
| SYN-B02: core edits do not generally execute through repair/evaluation | Critical | UIUX, Web | G3 |
| SYN-B03: fixture-only Copilot lacks provider lifecycle | High | UIUX, Content, Web | G4 |
| SYN-B04: Compare conflates methods/options and Inspect mutates | Critical | UIUX, Content, Web | G5 |
| SYN-B05: acceptance does not use `PlanRepository` | Critical | Content, Web | G5 |
| SYN-B06: Evidence hierarchy is too technical and incomplete | High | UIUX, Content, Web | G5 |
| SYN-B07: required mobile screens and PWA are absent | Critical | UIUX, Web | G6 |
| SYN-B08: accessibility has semantic and interaction defects | High | UIUX, Web | G6 |
| SYN-B09: health/launcher readiness is shallow | High | Web | G1/G7 |
| SYN-B10: current tests cannot prove replacement verification | Critical | All | G8 |
| SYN-B11: CP-010 and the current plan overstate verification | Critical | Content, Web | G0 |

## Correct aggregate status

| Capability | AUD-0 status |
|---|---|
| Service shell, `/app`, registry | Implemented; smoke-tested |
| Mutation token/revision boundary | Implemented; focused-test covered |
| Desktop shell | In progress |
| Geographic map | Not implemented |
| Controlled draft editing | Partially implemented |
| General repair preview | Not implemented |
| Deterministic Copilot | Implemented; demo/test-only |
| OpenAI Copilot | Not implemented |
| Map–Copilot synchronization | Partially implemented |
| Compare | Partially implemented; critical action bug |
| Evidence | Partially implemented |
| Local decision/pointer | Partially implemented |
| `PlanRepository` acceptance | Not implemented |
| Required mobile routes | Not implemented |
| PWA | Not implemented |
| Accessibility verification | Not verified |
| Windows black-box journey | Not verified |
| Product replacement | Not accepted |

The frozen v6 E3.UX artifact remains a separate verified compatibility fixture.
No product finding changes E3.1, E3.3, E4, E3.UX5, or E5.

## Specialist baseline routing

The following is a **routing register**, not a claim that additional independent
specialist audits have already passed. Each specialist must independently inspect
the implementation at the assigned gate.

| Specialist audit | AUD-0 baseline | Next required review |
|---|---|---|
| Design System | Partial token/layout system; dense evidence text, shared orange states, sub-44px controls | G2 and G6 |
| Map/Artifact Integrity | Coordinates and road geometry exist; `/app` does not render them | G2, G3, G5 |
| Copilot Privacy/Security | No external transmission today; provider, transcript, TTL, CSP, and cache boundaries absent | G1, G4, G6 |
| Acceptance/Persistence | Decision file and atomic pointer exist; `PlanRepository`, full re-verification, persistent sessions/drafts absent | G3 and G5 |
| Mobile/PWA | Responsive shell exists; dedicated screens, bottom sheet, manifest, service worker absent | G6/G7 |
| Accessibility | Semantic foundations exist; button roles, SVG map model, names, target sizes, and journey evidence need correction | G3/G6/G8 |
| Visual Fidelity | Recognizable shell but map and mobile panels fail the approved mockup | G2/G6/G8 |
| User Journey | Several local interactions observed; full clean launch-to-restart decision journey not run | G7/G8 |
| Phase/Gate Status | Product status is inflated; research and product tracks must remain non-propagating | G0 and every gate |

## What is necessary now: AUD-0/W0 only

1. Preserve the four baseline/synthesis reports and nine independent specialist
   reports as the AUD-0 evidence package.
2. Change CP-010 to `in-progress` in the status-authority document.
3. Update only the active product plan, current execution plan, gate map, indexes,
   and inaccurate README claims needed to reflect the approved direction.
4. Record D1 as the next research action and W0 as the next product action.
5. Add the non-propagation rule: product gates cannot advance E3/E4/E5.
6. Run documentation links, status consistency, encoding, focused pytest, Ruff,
   and `git diff --check`.
7. Stop at G0 and report evidence before feature work.

AUD-0/W0 must not install Atlas, integrate OpenAI, change product code, replace the
map, change acceptance, build mobile/PWA features, or advance research gates.

## Gate routing

| Gate | Blocking findings to close | Required independent review |
|---|---|---|
| G0 — Truth | SYN-B11 | Content, Web, Phase/Gate Status |
| G1 — Runtime | SYN-B09 plus configuration/security boundary | Web, Copilot Privacy/Security, Phase/Gate Status |
| G2 — Geographic | SYN-B01 | UIUX, Map/Artifact, Design System, Visual Fidelity |
| G3 — Draft | SYN-B02 plus persistent draft/accessibility concerns | UIUX, Web, Map/Artifact, Acceptance/Persistence |
| G4 — Copilot | SYN-B03 | Content, Web, Copilot Privacy/Security, Phase/Gate Status |
| G5 — Decision | SYN-B04, SYN-B05, SYN-B06 | Content, Web, Map/Artifact, Acceptance/Persistence |
| G6 — Mobile/PWA | SYN-B07, SYN-B08 | UIUX, Mobile/PWA, Accessibility, Visual Fidelity |
| G7 — Launch | SYN-B09 and clean-user recovery | Web, Mobile/PWA, User Journey, Phase/Gate Status |
| G8 — Replacement | SYN-B10 and every residual finding | Entire audit pool, Integration Review, user visual approval |

A gate fails on any blocking finding; scores and auditor votes are not averaged.
An implementation agent cannot independently approve its own work.

## Historical AUD-0 requirement traceability matrix

This matrix is the frozen pre-W1/W2 AUD-0 baseline and is superseded as a live
control surface by `docs/planning/current_execution_plan.md` and the canonical
Copilot implementation plan. Its planned/not-implemented findings are retained
for traceability and must not be read as current W2 status. At baseline, `Planned` meant no
implementation evidence exists yet; `In progress` means baseline work or
implementation exists but the named gate has not received all independent
sign-offs; `Blocked` names an external or predecessor dependency. A row becomes
`Verified` only after the planned evidence is recorded and every named auditor
signs the owning gate.

| Requirement ID | Requirement / accountable implementation owner | Code, files, or product area | Planned tests and evidence | Independent auditor | Gate | Current status |
|---|---|---|---|---|---|---|
| AUDIT-001 | Materialize the baseline and specialist audit package / `EXECUTION_ORCHESTRATOR` | `docs/audits/`; evidence manifest | Report presence, provenance, finding IDs, manifest schema/JSON validation | Phase/Gate Status Auditor | G0 | Verified at G0; downstream audit findings remain open at their routed gates |
| AGENT-001 | Enforce multi-agent ownership, handoffs, and independent approvals / `EXECUTION_ORCHESTRATOR` | Canonical Copilot plan, audit routing, handoff records | Owner-to-file audit, handoff log, no self-approval check | Phase/Gate Status Auditor, Integration Review | G0 and every gate | Verified at G0; re-audit required at every later gate |
| STATUS-001 | Keep written status aligned with evidence / `EXECUTION_ORCHESTRATOR` | Current problem manifest, execution plan, gate map, Copilot plan | Cross-document status scan, research/product non-propagation check | Content Integrity Auditor, Phase/Gate Status Auditor | G0 and every gate | In progress; CP-010 must remain `in-progress` |
| MAP-001 | Render an Atlas-backed geographic map / `MAP_ENGINEER` | Map configuration/API, Mapbox GL JS module, Atlas loopback service | Style/assets load, WebGL initialization, range-request and recovery tests; desktop/mobile captures | Map/Artifact Integrity Auditor, Visual Fidelity Auditor | G1/G2 | Planned; live exit blocked by Atlas entitlement and Docker preflight |
| MAP-002 | Preserve validated stop and route geometry, coordinate order, and provenance / `MAP_ENGINEER` | Run artifacts, workspace GeoJSON adapter, route/stop layers | Artifact-to-GeoJSON coordinate/lineage assertions, original-vs-child geometry comparison, no fabricated route check | Map/Artifact Integrity Auditor | G2 | Planned; current `/app` discards geographic geometry |
| MAP-003 | Support controlled typed edits, undo, and real preview / `MAP_ENGINEER` with `PRODUCT_FRONTEND_AGENT` | Draft models/store/compiler, map actions, repair/evaluator service | Operation schema/target tests, revision conflict, undo/restart, solver/evaluator integration, certificate invalidation | Map/Artifact Integrity Auditor, Acceptance/Persistence Auditor | G3 | In progress; only partial draft behavior exists |
| COPILOT-001 | Add a provider-neutral OpenAI Responses adapter and explicit deterministic double / `COPILOT_ENGINEER` | Copilot protocol/provider module, API lifecycle, provider configuration | Structured-output, timeout, 429/5xx/auth/cancel, unsupported request, no silent fallback tests | Copilot Privacy/Security Auditor, Web Implementation Auditor | G4 | Planned; deterministic fixture only |
| COPILOT-002 | Persist, expire, and delete local transcripts without exposing them as research evidence / `COPILOT_ENGINEER` | Conversation store, retention job, delete APIs/settings | 30-day TTL, delete-one/delete-all, cache/log/evidence exclusion tests | Copilot Privacy/Security Auditor, Content Integrity Auditor | G4/G6 | Planned |
| SYNC-001 | Synchronize authoritative map, Copilot, Compare, and Evidence context / `PRODUCT_FRONTEND_AGENT` | Session context, selection API, client state/router | Selection propagation, close/reopen, run switch, stale revision, focus restoration E2E | User Journey Auditor, Accessibility Auditor | G3/G4/G5 | In progress; map-Copilot sync is partial |
| COMPARE-001 | Compare a baseline and two distinct artifact-backed child plans / `PRODUCT_FRONTEND_AGENT` | Alternative service, Compare route/cards/map preview | Unique plan ID/hash, eligibility, null preservation, no method-as-alternative, inspect-is-read-only tests | Map/Artifact Integrity Auditor, Content Integrity Auditor | G5 | In progress; current alternatives are semantically defective |
| EVIDENCE-001 | Present progressive, truthful decision evidence / `CONTENT_IMPLEMENTATION_AGENT` | Evidence route, evaluation/certificate/lineage adapters, bundle endpoint | Eligible/ineligible/missing/stale/hash-mismatch/fallback states, allow-list download, hierarchy/a11y E2E | Content Integrity Auditor, Map/Artifact Integrity Auditor, Accessibility Auditor | G5 | In progress; technical-first incomplete view exists |
| ACCEPT-001 | Save immutable children, append decisions, and atomically CAS the accepted pointer / `PERSISTENCE_ENGINEER` | `PlanRepository`, acceptance transaction, workspace pointer, decision ledger | Hash/lineage/fresh-certificate checks, concurrent CAS, crash/restart, rollback, parent immutability | Acceptance/Persistence Auditor, Integration Review | G5 | In progress; current local pointer/decision path is insufficient |
| MOBILE-001 | Supply dedicated Overview, Day Details, Map, Copilot, Compare, and Evidence experiences / `MOBILE_PWA_AGENT` | Responsive routes, bottom navigation, map sheet, safe-area composer | 430/390/360 journeys, overflow, touch target, virtual-keyboard, long-label captures | Mobile/PWA Auditor, Accessibility Auditor, Visual Fidelity Auditor | G6 | Planned; responsive hiding is not a mobile implementation |
| PWA-001 | Provide an installable shell with truthful backend/provider/offline states / `MOBILE_PWA_AGENT` | Web manifest, service worker, icons, cache policy | Installability, update, API/state deny-list, backend-down and internet-down tests | Mobile/PWA Auditor, Copilot Privacy/Security Auditor | G6/G7 | Planned |
| ACCESS-001 | Meet keyboard, semantics, focus, announcement, contrast, and textual-map requirements / `DESIGN_SYSTEM_AGENT` plus feature owners | Semantic HTML, CSS tokens, dialogs/dock, DOM map alternative | Automated accessibility scan, manual keyboard/screen-reader rubric, contrast/target-size checks | Accessibility Auditor | G3/G6/G8 | In progress; blocking semantic and interaction defects recorded |
| ENTRY-001 | Launch the validated default workspace at stable `/app` / `WINDOWS_LAUNCH_AGENT` with `RUNTIME_BACKEND_ENGINEER` readiness support | Launcher, health/readiness API, registry, state-store preflight | Clean PowerShell launch, identity/port collision, default workspace load, degraded/failed recovery | User Journey Auditor, Web Implementation Auditor | G1/G7 | In progress; shell works but readiness is shallow |
| LEGACY-001 | Keep frozen dashboards debug-only with no fallback / `EXECUTION_ORCHESTRATOR` | Legacy route flag and labels, launcher/router | Disabled-by-default route tests, explicit-enable label test, failed-product no-fallback journey | User Journey Auditor, Content Integrity Auditor | G7 | Planned |
| VISUAL-001 | Match the approved eight-panel hierarchy truthfully / `DESIGN_SYSTEM_AGENT` with UI owners | Desktop/mobile shell, tokens, map, Copilot, Compare, Evidence | 1440/1280/1024/768/430/390/360 rubric and deterministic screenshots; user approval | Visual Fidelity Auditor | G2/G6/G8 | Not verified; baseline scored 13/32 |
| JOURNEY-001 | Complete the clean launch-to-restart decision journey / `TEST_AUTOMATION_AGENT` with `QA_ACCESSIBILITY_AGENT` | Entire local product, disposable workspace, browser/Windows harness | Documented 28-step black-box run, console/network/overflow/offline/shutdown evidence | User Journey Auditor, Integration Review | G7/G8 | Not verified |
| SECURITY-001 | Enforce loopback, Host/Origin, mutation, path, cache, CSP, secret, and size boundaries / feature owners coordinated by `EXECUTION_ORCHESTRATOR` | FastAPI middleware/API, Atlas/PWA/provider/transcript/evidence boundaries | Host/Origin negatives, token/revision auth, traversal, size/rate/cache/CSP/log-redaction tests | Copilot Privacy/Security Auditor | G1/G4/G6/G8 | In progress; multiple blocking gaps recorded |
| RESEARCH-001 | Prevent product evidence from advancing E3.1/E3.3/E4/E5 / `EXECUTION_ORCHESTRATOR` | Research gate map, status authority, product/research manifests | Cross-track status diff and non-propagation audit at every gate | Content Integrity Auditor, Phase/Gate Status Auditor | G0 and every gate | Enforced as policy; research statuses unchanged |

## Preserved foundations

The correction should reuse rather than discard:

- Safe pinned run registry.
- FastAPI loopback application boundary.
- Per-session mutation token and revision conflict.
- Explicit experimental/fixture labeling.
- Fail-closed unsupported preview behavior.
- Safe evidence archive path handling.
- Append-only decision creation and atomic pointer replacement.
- Frozen v6 compatibility fixture and its regression evidence.

## External blockers

- Mapbox Atlas license/token and supported Docker environment block live G1/G2.
- OpenAI credentials and model access block live G4 verification.
- A new parent plus two distinct independently evaluated children block complete
  G2/G5.
- User visual approval blocks G8.

These blockers do not justify misleading substitutes or premature verification.

## AUD-0 / G0 exit statement

AUD-0/W0 and G0 are **verified**. Baseline corrections, three broad audits, all
nine specialist reports, the detailed traceability matrix, final evidence
records, and independent Content, Web, and Phase/Gate Status sign-offs pass.
CP-010 remains `in-progress`; W1 is ready for deterministic work while live G1
exit remains Atlas/Docker-gated. No research status has changed.
