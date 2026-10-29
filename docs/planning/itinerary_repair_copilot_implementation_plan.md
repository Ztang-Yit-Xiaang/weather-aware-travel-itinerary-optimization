# Itinerary Repair Copilot Corrective Multi-Agent Implementation Plan

**Track:** Parallel non-publication product prototype  
**Product:** Itinerary Repair Copilot  
**Entrypoint:** `OPEN_ITINERARY_COPILOT.cmd` -> `/app`  
**Current product action:** implement and independently verify the mandatory E2E-1 vertical-slice gate, then run G5; keep W6 closed until both pass
**Research boundary:** Product work cannot advance E3.1, E3.3, E4, E3.UX5, or E5.

**Current checkpoint (October 7, 2026):** the fresh real website GPT-to-repair/evaluator/map journey and exact-output replay pass; the complete E2E-1 gate remains in progress. See [the concise status and R01–R24 register](../reports/runtime_coherence_status_20261007.md). Full regression, the fresh aggregate after today's shared-test repair, cross-family composition, native mobile-keyboard evidence and phase audit remain open. Historical claims below are source-bound; G5 and W6-W8 have not advanced.

## 1. Executive Summary

Replace the partially functional `/app` prototype with a verified local product
matching the approved eight-panel workflow. The target includes a locally hosted MapLibre/PMTiles
geographic map, controlled typed edits, real repair/evaluation preview,
provider-backed Copilot plus a deterministic test adapter, distinct evaluated
alternatives, progressive Evidence, repository-backed acceptance, complete
mobile routes, an installable PWA, and a stable Windows launcher.

Implementation is multi-agent by contract. The root Execution Orchestrator
coordinates shared contracts and integration; bounded implementation packages
run through non-overlapping agents; independent audit agents remain read-only.
No feature work starts until AUD-0/W0 closes the truth gate G0.

## 2. Historical Acceptance-Failure Baseline (Pre-W2)

The bullets below preserve the acceptance baseline recorded before W2. They are
historical traceability, not current implementation claims. W2 has since added
the local geographic renderer, distinct evaluated alternatives, non-mutating
baseline presentation, and selected-plan Evidence. Those v1 verdicts remain historical; corrected-v2 G2/G3 have now passed independent revalidation.

- The primary map is a non-geographic SVG schematic.
- Original/repaired routes are not meaningfully separated.
- Map selection did not reliably update session state in user-like testing.
- Most typed drafts cannot execute through repair and independent evaluation.
- Copilot is a keyword fixture, not an OpenAI provider integration.
- Compare may present methods as distinct plan alternatives.
- A Keep-original Inspect action has mutation semantics.
- Evidence leads with technical identifiers instead of decision evidence.
- Required mobile Overview, Day Details, map sheet, Copilot, and Evidence flows
  are absent or incomplete.
- Browser, accessibility, PWA, and clean Windows journeys are not verified.

The FastAPI shell and partial product UI exist; CP-010 is therefore
`in-progress`, not `verified`.

## 3. Corrected Statuses and Current Gate Progress

| Area | Status |
|---|---|
| FastAPI, `/app`, fixed-port launcher | Implemented; deterministic and HTTP smoke evidence pass |
| Registry/runtime readiness | W1 implemented; W1M/G1 verified with the live local MapLibre/PMTiles package |
| Desktop shell | Corrected-v2 W2/G2 verified |
| Geographic map | Continuous v2 MapLibre/PMTiles route verified |
| Controlled edits | W4R implemented: server-owned snap previews, confirmed waypoints, registered POI replacement, and typed stop/route edits; G4R verified |
| Repair preview | W4R implemented for the registered operation boundary, including evaluated typed edits and exact route evidence; G4R verified |
| Fixture Copilot | Implemented; explicit demo/test only |
| OpenAI Copilot | W4 implemented; August v6 corpus evidence is historical. October 7's bounded `gpt-5.4-mini` real website journey and exact-output replay independently pass; this does not prove arbitrary prompts or close E2E-1 |
| Compare / Evidence | W4R implemented with exact selected-option resolution, synchronized dual map controllers, active-preview evidence, and user-oriented defaults; G4R verified |
| `PlanRepository` acceptance | W5 implemented; confirmed Accept/Keep/rollback, immutable repository, ledger, pointer CAS, recovery and idempotency are present. Continued acceptance has September browser evidence; the complete E2E-1 gate and G5 remain open |
| Mobile / PWA | Responsive web and touch journeys have bounded browser evidence. Native OS keyboard remains untested; installable PWA work remains planned, not implemented |
| AUD-0 | Verified: three broad and nine specialist audits are durable |
| W0 / G0 | Verified: traceability, evidence metadata, and independent sign-offs pass |
| W1 / W1M / G1 | W1 implemented; W1M/G1 verified by live, offline-local, browser-shell, regression, and six-auditor evidence |
| W2 / G2 | Corrected-v2 W2/G2 verified by seven independent audit roles |
| W3 / G3 | Corrected-v2 W3/G3 verified by preview, persistence, integrity, and user-journey evidence |
| W4 | Implemented offline; not provider-verified |
| W4R / G4R | W4R implemented; G4R verified after independent remediation re-audit |
| G4 | Current-source v6 verified / passed after three independent read-only reviews |
| W5 | Implemented; decision/persistence mechanics remain subject to G5 |
| E2E-1 | In progress; six deterministic journeys pass; bounded billed website-chat journey authorized and pending before G5 |
| G5 | Blocked behind E2E-1 |
| W6-W8 | Planned and closed behind E2E-1/G5-G7 |
| CP-010 | In progress until G8 |

Research status is unchanged: E1, E2, E3.0, E3.M, E3.2, E3.C1-C4, and
E3.UX0-UX4 remain verified; E3.1 is blocked on D1; E3.3 and E4 remain blocked;
E3.UX5 and E5 remain deferred. D1 is the next research action. Resolution and
Historical v1 G2/G3 verdicts remain preserved. The v2 continuity correction has passed independent revalidation. W3 provides persistent controlled drafts,
revision-safe undo, compilation into typed repair requests, independent
repair/evaluator preview, and exact proposal geography/evidence while the
accepted parent remains immutable. The W4 implementation-ready phase plan was
approved and W4 deterministic implementation passed independent audits on 2026-08-05. Historical G4 aggregates failed provenance audit; v4 closed recomputability, and the authorized v5 exact-I/O/transport run plus three independent audits verified its frozen source. The authorized 2026-08-13 current-source v6 run passes the same fixed-24 comparison at low and medium reasoning with exact file/LLM I/O; three independent read-only reviews pass after the audit-binding defect was repaired.

The v2 correction record is
[`../audits/w2_route_continuity_correction_report.md`](../audits/w2_route_continuity_correction_report.md).
The independent corrected-v2 gate record is
[`../audits/w2_v2_g2_g3_revalidation_report.md`](../audits/w2_v2_g2_g3_revalidation_report.md).

## 4. Confirmed Decisions

| Decision | Contract |
|---|---|
| Product interpretation | Functional implementation of all eight mockup panels |
| Default demo | California Coast parent plus two distinct evaluated children |
| Map | Local MapLibre GL JS + PMTiles primary; Mapbox Atlas deferred explicit backup |
| Editing | Constraint-grounded direct manipulation: explicit edit mode, catalog-stop replacement semantics, movable custom waypoints, exploratory pins, route-policy edits, and server-authoritative snapping/routing; no freehand road geometry |
| Copilot | OpenAI Responses API plus deterministic test adapter |
| Model | Configurable GPT-5.6 Terra default |
| Transcripts | Local full transcript, 30-day retention, delete-one/delete-all |
| Drafts | Persistent local typed drafts |
| Acceptance | `PlanRepository`, append-only ledger, atomic current pointer |
| Mobile | Dedicated 430/390/360px routes |
| PWA | Installable shell; local PMTiles assets supply offline map data |
| Legacy | Debug-only; never fallback |
| Execution | Mandatory coordinated multi-agent implementation and specialist audits |

## 5. Repository Findings

- `src/itinerary_system/product_app/` contains the partial service and UI.
- `static/js/app.js` concentrates most browser behavior.
- W2 now adapts run coordinates and pinned road-validated route cells into
  browser GeoJSON for the local MapLibre renderer.
- Frozen v6 remains the read-only compatibility artifact.
- `PlanRepository` provides immutable storage/hash verification but no accepted
  pointer or decision ledger.
- Preview substitutes a registered child for a narrow fixture request rather
  than compiling general draft operations.
- Focused W1/W1M coverage is split across registry, runtime/map-provider, persistence,
  security/API, launcher, frontend-contract, and product-journey modules.
- Provider-neutral configuration and the local MapLibre static-service boundary are verified with
  pinned assets and a live loopback service. The W2 browser renderer is implemented;
  Atlas is retained only as an opt-in backup awaiting entitlement.
  The OpenAI SDK/transport, web manifest, service worker, and PWA icons remain absent.

## 6. Current Architecture

```text
immutable run -> ProductRunRegistry -> ProductService -> ProductRuntime -> FastAPI -> static /app
                                                     |              |              |
                                           default preload   security boundary   MapLibre WebGL map
LocalStateLayout -> layout metadata + readiness probe
MapLibre/PMTiles static boundary -> MapConfigurationV2 -> artifact-backed W2 renderer
fixture adapter -> API; sessions and drafts remain memory-only
```

Required connections are validated geometry to the map, drafts to repair and
evaluation, OpenAI to the adapter boundary, evaluated children to repository
acceptance, and shared desktop state to dedicated mobile/PWA routes.

## 7. Historical Root Causes and Disposition

| Failure | Root cause |
|---|---|
| Non-geographic map | Addressed in W2 with local MapLibre/PMTiles and validated GeoJSON |
| Non-executable drafts | No complete draft-to-repair compiler |
| Fixture-only Copilot | Provider integration never crossed the adapter boundary |
| Duplicate alternatives | Addressed in W2 with two hash-distinct evaluated children |
| Unsafe Inspect semantics | Addressed in W2; baseline is a disabled reference and Inspect is non-mutating |
| Incomplete acceptance | Session state substitutes for repository authority |
| Missing mobile flows | Desktop panels are hidden without mobile replacements |
| Inflated status | Presence/HTTP/unit checks were treated as acceptance evidence |

## 8. Scope

AUD-0/W0 truth correction; local MapLibre/PMTiles runtime; California demo; geographic desktop
and mobile map; typed drafts; real preview; OpenAI adapter; transcripts; shared
context; Compare; Evidence; repository acceptance; PWA; stable launcher; and
automated, specialist-audit, and black-box verification.

## 9. Non-Goals

Arbitrary coordinates/dragging, live booking/pricing/weather/safety feeds,
multi-user/cloud state, native apps, automatic credential purchasing, silent
fixture fallback, public OSM offline caching, publication claims, or any change
to the D1/E3.1/E3.3/E4/E3.UX5/E5 research sequence.

## 10. Assumptions

The product is loopback-only and single-user. Unrelated dirty-tree changes are
preserved. Credentials are supplied only at their gates. A bounded ODbL-compatible
California Coast PMTiles extract is sufficient for V1. Docker uses a supported Linux engine. The
PWA requires FastAPI for mutations. Transcripts are local application state,
never research evidence. Auditors are read-only.

## 11. Blocking Dependencies

| Dependency | Gate |
|---|---|
| Explicit W0 authorization | Satisfied on 2026-08-03; AUD-0/W0 verified |
| Pinned MapLibre/PMTiles/style assets, archive provenance/coverage, Linux Docker engine, digest/disk/port preflight | Satisfied at W1M/G1 |
| Atlas entitlement/license | Optional backup provider gate only; not on the product critical path |
| OpenAI key/model access | Available locally; every fresh billed G4 run still requires explicit user authorization |
| Three-plan route-validated demo | W2-W5 |
| Frozen shared contracts | W2-W5 |
| User visual approval | G8 |

Credentials are not required for W0, schemas, deterministic tests, or explicit
unavailable states; their absence prevents the affected phase from becoming
verified.

## 12. Product Information Architecture

Routes are `/app`, `/app/itinerary`, `/app/day/{day}`, `/app/map`,
`/app/copilot`, `/app/repairs`, `/app/compare`, and `/app/evidence`. Desktop
adds a resizable Copilot dock, map layers, edit dialogs, alerts/bookings
drawers, transcript settings, and recovery panels. URLs never contain tokens,
transcripts, drafts, permissions, or secrets.

## 13. Mockup Mapping

| Panel | Delivery |
|---|---|
| 1 Main Review | Itinerary, geographic map, repair panel, metrics, decisions |
| 2 Desktop Copilot | Resizable dock, context, provider/proposal lifecycle |
| 3 Map Editing | Controlled actions, draft, undo, preview |
| 4 Compare | Baseline plus two distinct evaluated children |
| 5 Evidence | Progressive checks, route, lineage, metrics, download |
| 6 Mobile Overview | Issue, affected day, repair, metrics, bottom nav |
| 7 Mobile Copilot | Full-screen chat, permission, proposal, compare/apply |
| 8 Mobile Evidence | Accessible accordions and downloads |

## 14. Desktop Specification

At >=1280px use an approximately 80px navigation rail, 300-340px itinerary
rail, flexible map, 340-400px repair panel, and 360-520px Copilot dock. At
1024px use drawers; at 768px use explicit Itinerary/Map/Repair/Copilot tabs.
Accept repair is the sole primary action. Required modules never disappear
without an accessible replacement.

## 15. Mobile Specification

At 430/390/360px provide Overview, Day Details, full-screen map with snap-point
edit sheet, full-screen Copilot with safe-area composer, stacked Compare,
accordion Evidence, and five-item bottom navigation. Require no horizontal
overflow, 44px targets, wrapping names, and virtual-keyboard reachability.

## 16. Geographic Map Architecture

MapLibre GL JS and the PMTiles browser protocol consume local styles, glyphs, sprites, and a bounded archive through loopback
Nginx on port 8080 with range-request support. Separate GeoJSON sources/layers
represent original, accepted, draft, and alternative routes; stops and
disruption annotations are artifact-derived. Browser code never fabricates
road validation. WebGL/local-map failure renders an explicit textual route view with visible source attribution.

With internet disconnected but FastAPI and the local asset service running, local map and
decision flows remain available; OpenAI reports unavailable. With FastAPI
stopped, the installed PWA shows a cached unavailable shell only.

## 17. Map Editing Architecture

Allowed operations are `select_stop`, `select_segment`, `keep_stop`,
`lock_stop`, `mark_flexible`, `move_day`, `route_feedback`,
`replace_nearby`, and `add_candidate`. Every mutation validates token,
revision, and target universe, appends to a persistent ordered draft, and
updates overlays/text. Preview compiles the draft, creates immutable candidate
children, and invokes independent evaluation. New drafts invalidate old
certificates. Ineligible plans remain inspectable but cannot be ranked or
accepted.

## 18. Copilot Lifecycle

`OpenAICopilotAdapter` and `DeterministicCopilotAdapter` share a strict typed
contract. Production uses the Responses API, configurable Terra default,
server-side key, 30-second timeout, and one retry for transient 429/5xx only.
Model output never directly mutates constraints or accepts plans.

Lifecycle: Idle -> Sending -> Interpreting -> Clarification/Permission/Proposal
-> Solving -> Evaluating -> Eligible/Ineligible/Failed. Explicit failure states
cover configuration, timeout, rate limit, authentication, invalid output,
unsupported request, stale revision, solver, and evaluator failures. Fixture
mode is explicit and never a silent runtime fallback.

## 19. Map-Copilot Synchronization

`WorkspaceContextV1` holds run, session, revision, accepted plan, selected day,
stop, segment, candidate and alternative, draft, proposal, and conversation.
Map selection updates Copilot chips; proposals highlight exact entities;
Compare updates map and Evidence. Run switching clears scoped state. Every
mutation carries `expected_revision`; stale requests return 409 and reload.

## 20. Compare Architecture

Compare contains Keep original, the recommended eligible repair, and one
distinct low-driving/preservation child. Alternatives require distinct plan
IDs and content hashes. Methods producing the same child remain method evidence.
Inspect is non-mutating. Null stays `Unavailable`; ineligible plans are not
ranked or accepted.

## 21. Evidence Architecture

Progressive order is decision/eligibility, checks, route/freshness,
parent-child lineage, requested/executed method, evaluator metrics,
explanations, then technical IDs/downloads. Missing certificate, mismatch,
stale context, partial artifact, fallback route, exact incompleteness, and
missing metrics are explicit. Downloads are manifest allow-listed.

## 22. Acceptance and Persistence

Accept verifies token, revision, current proposal, child hash, eligibility,
fresh certificate, diff, and lineage; saves the immutable child through
`PlanRepository`; appends `AcceptanceDecisionV1`; atomically replaces the
workspace pointer; advances revision; and clears obsolete state.

`.product_app_state/` contains the application plan repository, per-trip
pointer and decision JSONL, sessions/drafts, and conversations. Keep original
requires confirmation and records a distinct decision. Rollback appends a new
decision pointing to a prior eligible plan; history is never rewritten.

## 23. State Model

Server authority owns registry, accepted pointer, revision/token, selection,
drafts, proposals, permissions, evaluations, transcripts, and decisions.
Client state is limited to route, panel width, camera, scroll, hover, and focus.
Drafts survive refresh; transcripts expire after 30 days; decisions persist;
run switching cannot leak state.

## 24. Backend Contracts

Required types include `WorkspaceContextV1`, `DraftOperationV1`,
`DraftSnapshotV1`, `CopilotTurnV1`, `CopilotProposalV1`,
`AlternativeSummaryV1`, `AcceptanceDecisionV1`, `WorkspacePointerV1`, and
`MapConfigurationV2`.

Required API coverage includes deep health, map configuration, geographic
workspace data, session/selection, draft append/undo/preview, Copilot messages,
permissions, alternatives, accept, Keep original, conversation read/delete,
delete-all, and allow-listed evidence download. Schemas reject unknown types,
invalid IDs, traversal, stale revisions, invalid tokens, oversized input,
nonfinite metrics, duplicate alternatives, ineligible acceptance, and lineage
mismatch.

## 25. Stable Entrypoint

The launcher resolves paths, owns fixed port 8127, reuses only a matching
healthy product, validates the default workspace and state writability, probes
the explicitly selected local map provider, reports OpenAI configuration without secrets, opens `/app`, never falls
back to legacy, and shuts down with Ctrl+C. Health reports `ready`, `degraded`,
or `failed` per component. Final acceptance requires the selected local map provider to be ready; Atlas is not required.

## 26. Legacy and Planning Migration

Frozen v6, Folium, and modular artifacts remain unchanged. Explicit
`--enable-legacy` routes are labeled research/debug artifacts and excluded from
navigation. W0 corrects CP-010, replaces obsolete product decisions, adds the
product branch/gates, repairs only inaccurate README statements, and writes
audit evidence without rewriting historical reports.

## 27. Accessibility

Use semantic landmarks/headings, real list semantics, stable accessible names,
44px targets, DOM equivalents for map controls, focus restoration, modal-only
focus traps, live status announcements, non-color state cues, reduced motion,
AA contrast, textual map alternatives, and virtual-keyboard testing.

## 28. Security and Privacy

Bind FastAPI, the local map asset service, and OSRM to loopback. Validate Host, Origin, token,
revision, schema, and path. Keep credentials outside Git. Restrict CSP/CORS;
escape artifact/model/transcript text; limit sizes; never log keys, licenses,
full transcripts, or host paths. Apply 30-day transcript TTL and deletion.
Exclude transcripts and sensitive APIs from evidence and service-worker cache.

## 29. Testing Strategy

Add unit, API, repair/evaluator integration, deterministic OpenAI transport,
local-map range/provenance/offline, browser interaction, PWA, accessibility, visual, Windows
black-box, full pytest, Ruff, project-validator, and legacy-regression layers.
Live OpenAI smoke is credential-gated and never the deterministic CI oracle.
HTTP 200 or element presence never proves a user journey.

## 30. Black-Box Journey

From clean PowerShell: start the selected local map service; run only the documented launcher; verify
identity/readiness/redirect/no legacy; select Day 3, a geographic stop and
segment; use Copilot and reopen it; create/undo/recreate/preview a draft; wait
for evaluation; Compare distinct map alternatives; inspect Evidence; accept in
a disposable workspace; restart and verify pointer; test Keep original in a
second workspace; delete transcripts; disconnect internet while local services
continue; exercise desktop/mobile/PWA; inspect console/network/overflow; shut
down cleanly.

## 31. Visual Verification

Capture 1440, 1280, 1024, 768, 430, 390, and 360px. Audit all eight panels for
module presence, hierarchy, map realism, state/action clarity, Copilot,
Compare/Evidence, mobile intentionality, typography, spacing, color, icons,
density, and truthfulness. Only dynamic basemap pixels may be masked. User
approval is the final visual gate.

## 32. Detailed Phases

| Phase | Status | Work | Exit |
|---|---|---|---|
| AUD-0 | Verified | Materialize independent baseline audits/synthesis | Three broad and nine attributed specialist reports plus evidence manifest pass |
| W0 | Verified | Correct authority/status, traceability, baseline evidence | G0 truth gate passed |
| W1 | Implemented | Historical shared contracts, readiness, security/storage boundaries | Preserved W1 evidence |
| W1M | Verified | MapLibre-primary/provider-neutral v2 contracts and static boundary | G1 passed with closed assets, live/offline-local checks, and six audits |
| W2 | Verified | Continuous v2 three-plan demo and geographic map | Corrected G2 passed |
| W3 | Verified | Persistent controlled drafts and real preview inside the registered operation/candidate boundary | Corrected G3 passed; preserve its evidence |
| W4 | Implemented offline | OpenAI adapter, versioned prompt/eval contract, transcripts, synchronization | Preserve implementation evidence; G4 is separately verified by the v5 exact-I/O gate |
| W4R | Implemented | Direct map/list/timeline interaction, complete route-leg accounting, runtime snap/route overlay, route-aware POIs, stop/route editing, contextual Copilot proposals, exact Compare/Evidence selection, and stability/coherence remediation | Preserve the implementation report as history and the later G4R remediation/re-audit as the current gate verdict |
| G4R | Verified / passed | Preserve `../audits/g4r_remediation_final_reaudit_20260811.md` and the full-size timestamped evidence | Every required audit passed; findings were not averaged |
| G4 | Current-source v6 verified / passed | Preserve the v6 exact-I/O evidence, repaired five-file audit binding, and three independent PASS reviews; no additional provider call is required | W5 entry satisfied |
| W5 | Implemented | Distinct Compare/Evidence, fresh proposal revalidation, repository/ledger/pointer acceptance, Keep original, rollback, restart/concurrency/idempotency, and confirmed UI | E2E-1 mandatory runtime gate |
| E2E-1 | In progress; E2E-1L live chat pending before G5 | Preserve the six deterministic journeys and add the authorized actual website chatbox -> OpenAI -> typed proposal -> Copilot draft -> preview -> evaluated child/map path with exact I/O, attempts, usage, cost, and privacy evidence | Independent E2E-1 pass; then G5 |
| G5 | Blocked behind E2E-1 | Independently verify the complete runtime slice plus W5 decision/persistence integrity | G5 decision gate |
| W6 | Planned | Mobile, PWA, accessibility | G6 mobile gate |
| W7 | Planned | Launcher, legacy isolation, documentation | G7 launch gate |
| W8 | Planned | Full regressions, audits, black-box, visual approval | G8 replacement gate |

Each phase starts only after its entry gate. W4R follows the completed offline W4
implementation, G4R independently verifies W4R, and G4 then closes the provider
evidence boundary. G4 is verified and W5 is implemented. E2E-1 must verify the complete browser-to-artifact-to-map slice before G5 may verify W5; W6 remains closed until both pass.

## 33. Multi-Agent Assignments

### Implementation agents

| Agent | Exclusive responsibility |
|---|---|
| `EXECUTION_ORCHESTRATOR` | Wave briefs, shared contracts, scheduling, integration, status authority |
| `PLANNING_STATUS_AGENT` | Audits/planning/status/index corrections |
| `RUNTIME_BACKEND_ENGINEER` | Models, API, registry, readiness, repair orchestration |
| `PERSISTENCE_ENGINEER` | Sessions, drafts, transcripts, repository, ledger, pointer |
| `DEMO_ARTIFACT_ENGINEER` | California demo, manifests, route/evaluation evidence |
| `DESIGN_SYSTEM_AGENT` | Tokens, primitives, responsive rules |
| `MAP_ENGINEER` | MapLibre/PMTiles, optional Atlas backup, GeoJSON layers/interactions/fallback |
| `COPILOT_ENGINEER` | Provider adapters, schemas, lifecycle |
| `PRODUCT_FRONTEND_AGENT` | Shell, router/API/state, desktop modules |
| `CONTENT_IMPLEMENTATION_AGENT` | Truthful labels, errors, Evidence hierarchy |
| `MOBILE_PWA_AGENT` | Mobile routes/sheets, manifest, service worker |
| `WINDOWS_LAUNCH_AGENT` | Launcher, wrappers, diagnostics, shutdown |
| `TEST_AUTOMATION_AGENT` | Unit/API/integration/browser/PWA/Windows tests |
| `QA_ACCESSIBILITY_AGENT` | Accessibility, viewport, screenshots, rubrics |
| `INTEGRATION_REVIEW_AGENT` | Read-only cross-module/final integrity review |

### Independent auditors

`UIUX_AUDITOR`, `CONTENT_AUDITOR`, `WEB_IMPLEMENTATION_AUDITOR`,
`DESIGN_SYSTEM_AUDITOR`, `MAP_ARTIFACT_INTEGRITY_AUDITOR`,
`COPILOT_PRIVACY_SECURITY_AUDITOR`, `ACCEPTANCE_PERSISTENCE_AUDITOR`,
`MOBILE_PWA_AUDITOR`, `ACCESSIBILITY_AUDITOR`,
`VISUAL_FIDELITY_AUDITOR`, `USER_JOURNEY_BLACK_BOX_AUDITOR`, and
`PHASE_GATE_STATUS_AUDITOR` remain read-only.

The Orchestrator publishes each agent's requirement IDs, input contracts,
owned/prohibited paths, outputs, tests, rollback boundary, and handoff. Maximum
concurrency is the root plus three agents. Shared files are sequentially owned.
An implementer cannot audit its own work. Any blocking audit finding fails the
gate; findings are not averaged.

## 34. File Ownership

Planning/Status owns `docs/audits/` and active authority documents. Runtime
Backend owns product server models/API/registry/service. Persistence owns
workspace/drafts and focused transcript/acceptance modules. Demo owns the
generator and generated package. Design owns tokens/shared component CSS. Map
owns local map deployment and map-specific modules/styles. Copilot owns provider
modules. Product Frontend owns bootstrap/router/API/client state and desktop
modules. Mobile/PWA owns mobile modules/styles/manifest/service worker. Windows
Launch owns the Python launcher and `.cmd` wrappers. Test/QA own their evidence
modules. Licensed assets, credentials, application state, transcripts, and
browser profiles remain untracked.

## 35. Dependencies and Critical Path

```text
research: D1 -> E3.1 -> E3.3 -> E4 -> E5
product:  AUD-0 -> W0/G0 -> W1 -> W1M/G1 -> W2/G2 -> W3/G3
                                      -> W4 implemented -> W4R -> G4R
                                      -> G4 -> W5 -> E2E-1 -> G5 -> W6/G6
                                      -> W7/G7 -> W8/G8 + user approval
```

There is no product-to-research status-propagation edge.

## 36. Parallel Work

Audits run in batches of at most three. W4R contract/routing/map work may proceed
in exclusive-path batches; stop/POI/route workflows follow the shared contract
freeze; contextual Copilot/explanation and exact Compare work follow their
respective schema handoffs. G4R audits are independent. W5 persistence/decision
work cannot begin before G4. Later W6/W7 work retains exclusive path ownership.
Late contract changes reopen dependent gates.

## 37. Gates

| Gate | Evidence | Independent review |
|---|---|---|
| G0 Truth | Reports, CP-010 correction, consistent authority/checks | Content, Web, Status |
| G1 Runtime | Schemas, readiness, security/config/storage boundaries | Web Implementation; Copilot Privacy/Security; Map/Artifact Integrity; Content; User Journey Black-Box; Phase/Gate Status |
| G2 Geographic | MapLibre map, authoritative geometry, attribution, fallback/screenshots | UIUX, Map, Design, Visual |
| G3 Draft | Operations, persistence, undo, real preview, immutability | UIUX, Web, Map, Acceptance |
| G4R Interaction | Typed direct actions, required-leg coverage/gaps, route-aware POIs, exact Compare/Evidence, responsive equivalents | Map Journey, Route Artifact, Accessibility, UIUX, Content, Visual, Security, Status, Black-Box |
| G4 Copilot | Post-fix provider smoke, fixed-24 evaluation, reasoning comparison, schemas, permission, context, retention | Provider, Prompt Quality, Content, Security, Web, Status |
| E2E-1 Runtime truth | Real browser action, exact child/diff/evaluation/geography identities, panel/map agreement, map acknowledgement, stale-response rejection, accepted refresh, exact API/file/LLM I/O | Integration, Map Journey, Artifact, Acceptance, Content, Security, Black-Box |
| G5 Decision | E2E-1 PASS plus distinct plans, Evidence, certificate, repository/CAS/restart | Artifact, Acceptance, Content, Web |
| G6 Mobile | Screens, keyboard, PWA/cache, accessibility | UIUX, Mobile, Accessibility, Visual |
| G7 Launch | Clean Windows, deep readiness, no fallback, shutdown | Web, Journey, Status |
| G8 Replacement | Full regressions, audits, black-box, user visual approval | Entire audit pool + Integration Review |

Only the Orchestrator/Lead Planner updates status after a gate passes.

## 38. Risks

Primary risks are agent collision, nominal audits, inflated status, unavailable
unavailable local map assets/OpenAI credentials, duplicate alternatives, stale certificates,
transcript leakage, sensitive service-worker caching, prototype/research
confusion, and superficial visual checks. Mitigation is exclusive ownership,
read-only auditors, strict gates, explicit degraded states, hash/revision checks,
privacy/cache boundaries, non-propagation, and real black-box journeys.

## 39. Rollback

Preserve v6 and legacy artifacts. Feature flags disable map editing, OpenAI,
acceptance, and PWA independently. Map failure uses text fallback. Drafts can
be discarded. Accepted-plan rollback appends a new decision. Immutable plans
and history are never rewritten. Failed phases stop at their gate. Agents do
not stage, commit, or push without separate authorization.

## 40. Deliverables

Audit reports/synthesis; corrected authority documents; local MapLibre/PMTiles deployment;
three-plan demo; geographic map; executable drafts; OpenAI/test Copilot;
transcript controls; synchronization; Compare; Evidence; repository acceptance;
mobile/PWA; stable launcher; automated test suites; accessibility, security,
artifact, persistence, visual, and black-box evidence; user approval record.

## 41. Acceptance Criteria

AUD-0 and G0-G8 have evidence; multi-agent ownership/handoffs are recorded; no
blocking audit remains; clean `/app` launch works; map geometry is
artifact-grounded; all operations produce truthful drafts/previews; Copilot
never directly mutates; alternatives are distinct; Evidence is truthful;
acceptance persists through repository/ledger/pointer and restart; mobile/PWA,
accessibility, security, regressions, black-box, and visual review pass; CP-010
is not verified before G8; research statuses remain unchanged.

## 42. Traceability

Requirement groups are AUDIT/AGENT/STATUS, MAP, COPILOT/SYNC,
COMPARE/EVIDENCE, ACCEPT, MOBILE/PWA/ACCESS, ENTRY/LEGACY, VISUAL/JOURNEY,
SECURITY, and RESEARCH. Every wave brief maps each requirement to an owner,
files, tests, auditor, and gate. The detailed live matrix resides in the
cross-audit synthesis and cannot mark a requirement complete without evidence.

## 43. Known External Gates

No design decisions remain. Local map assets/provenance, OpenAI credentials,
the validated three-plan demo, and final user visual approval are known gates.
Unavailable credentials permit deterministic implementation but not live
verification or misleading substitution.

## 44. Historical v1 Result: W3/G3 Controlled Drafts and Real Preview

The v1 W2/G2 and W3/G3 execution was verified before the continuity defect was found. The preserved historical execution was:

1. Freeze W3 draft, persistence, revision, preview, and error contracts; assign implementation and independent audit paths.
2. Activate the versioned session/draft store under `.product_app_state/sessions/` using the W1 process-safe lock and atomic replacement boundary.
3. Implement the canonical operations `keep_stop`, `lock_stop`, `mark_flexible`, `move_day`, `route_feedback`, `replace_nearby`, and `add_candidate` against the registered candidate universe.
4. Require the session mutation token and `expected_revision` for every draft mutation; return stable 403, 409, 413, and 422 failures without partial writes.
5. Implement ordered undo and restart-safe draft restoration while preserving all legacy state bytes and the accepted parent.
6. Compile only executable ordered operations into existing typed repair requests; fail closed for unsupported targets or evidence gaps.
7. Execute the existing repair pipeline, independently evaluate each immutable candidate, and expose eligible/ineligible preview states without updating the accepted pointer.
8. Synchronize draft/proposal highlights with the MapLibre workspace and invalidate any previously visible certificate after draft changes.
9. Test lifecycle, concurrency, restart, invalid target, stale revision, route evidence, evaluator failure, and parent immutability through API and browser journeys.
10. Require independent Draft/State, Solver/Evaluator, Security/Integrity, Web/Accessibility, Content/Truth, User-Journey, and Phase/Gate audits before G3 may pass.

That historical G3 verdict did not verify v2. Corrected-v2 G2/G3 have since
passed independent revalidation, so W4 is now ready. Atlas remains an explicit unverified
backup and is not on the W3 critical path. The frozen v6 artifact, CP-010
`in-progress` status, D1/E3.1, E3.3, E4, E3.UX5, and E5 remain unchanged.

## 45. Approved W4R Direct-Interaction Remediation

W4R started with the status/contract correction in this document and is now
implemented with root-run stability/coherence evidence. That evidence does not
prove G4R by itself; the later remediation and fresh independent closeout verified G4R.

Current blocking defects:

- direct map editing exposes too few meaningful operations and has no complete
  catalog-drag, custom-waypoint-drag, or empty-map confirmation flow;
- displayed route connections must be audited for every ordered adjacent stop
  pair and any missing required leg must become an explicit unvalidated gap;
- the current Compare path needs exact selected-option state, active evaluated
  preview support, dual instance-owned map controllers, and exact Evidence
  resolution; and
- Accept is available only for a current eligible evaluated proposal and Keep
  original is available as an explicit confirmed durable decision; both remain
  fail-closed on stale, malformed, unauthorized, or conflicting state.

W4R implementation order:

1. W4R.0 - freeze contracts and add the required-pair route diagnostic.
2. W4R.1 - add runtime routing, route overlays, and explicit gap states.
3. W4R.2 - add direct map/list interaction, snapping, confirmation, and history.
4. W4R.3 - add Place/Stop editing and route-aware POI candidates.
5. W4R.4 - add route policies, timeline, slack, and failure diagnosis.
6. W4R.5 - extend contextual Copilot proposals without direct mutation.
7. W4R.6 - correct Compare/Evidence and close responsive behavior.
8. G4R - run independent interaction audits.
9. G4 - current-source v6 is verified after the statewide-POI/name-aware change and three independent read-only reviews.
10. W5 - implemented; preserve the decision/persistence mechanics.
11. E2E-1 - implement and independently verify the mandatory vertical slice in `e2e_1_vertical_slice_integration_runtime_truth_phase_plan.md`.
12. G5 - run only after E2E-1 passes; keep W6 closed until G5 passes.

The research path remains separately blocked in the order
`D1 -> E3.1 -> E3.3 -> E4`; W4R/G4R cannot advance it.
