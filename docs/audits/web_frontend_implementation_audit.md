# Itinerary Repair Copilot Web Implementation Baseline Audit

**Audit date:** 2026-08-03  
**Audit gate:** AUD-0  
**Disposition:** **BLOCKING — substantive shell, incomplete product system**  
**Auditor role:** Independent read-only frontend/backend implementation audit.

## Method and evidence

The audit traced controls through browser handlers, HTTP routes, service logic,
session state, files, and focused tests. It distinguishes:

- **Observed:** recorded 2026-08-02 user-like browser manipulation.
- **Source-confirmed:** current repository inspection on 2026-08-03.
- **Test-covered:** behavior explicitly asserted in
  `tests/product_app/test_product_app.py`.

Eight focused tests passed in the recorded audit. That result establishes the
tested contracts below, not full product verification.

## Implementation classification

| Subsystem | Classification | Evidence |
|---|---|---|
| FastAPI shell and route fallback | Substantive | `/`, `/app`, client routes, static assets, CSP |
| Registry | Substantive | Safe relative paths, one pinned default, manifest ID/hash |
| Session mutation token/revision | Substantive | Authentication and `409 stale_session_revision` tests |
| Geographic map | Absent | Inline SVG schematic; no map library/runtime |
| Draft persistence | Partial | Drafts are in-memory session objects; decisions/pointer use disk |
| Draft execution | Partial | Route feedback maps to one registered child; other edits fail closed |
| Copilot | Demo-only | Deterministic keyword adapter, no provider transport |
| Compare | Partial/unsafe | Method-oriented rows; Inspect baseline mutates Keep original |
| Evidence | Partial | Allow-listed ZIP exists; progressive and failure-state coverage incomplete |
| Acceptance | Partial local implementation | Decision file and atomic pointer; no `PlanRepository` child save |
| Mobile | Responsive shell only | Desktop rails hidden; dedicated product routes absent |
| PWA | Absent | No manifest or service-worker registration |
| Automated browser/a11y/visual suite | Insufficient | No current product browser suite in `tests/product_app/` |
| Windows black-box | Not verified | Launcher code exists; complete clean-user journey lacks durable evidence |

## Blocking implementation findings

### WEB-B01 — `/api/health` reports ready before workspace validation

**Evidence:** Source-confirmed.

`api.py:70-78` returns `ready: true` from registry construction alone. The default
workspace is not loaded, the state root is not tested for writability, and map or
provider components are not represented. The launcher opens `/app` when this
shallow check passes.

**Required closure:** Deep component health with `ready`, `degraded`, and `failed`
states; load the default workspace before core readiness.

### WEB-B02 — Product ignores available geographic evidence

**Evidence:** Source-confirmed.

- Both registered plans contain latitude/longitude values.
- `route_matrix_68ab535465b06808.json` contains cached OSRM geometry marked
  `road_validated: true`.
- `product_dashboard_view_models.py:481-532` already constructs a map view.
- `static/js/app.js:132-176` discards geography and renders index-based SVG points.

**Required closure:** Expose validated GeoJSON and initialize the approved map
runtime. Do not synthesize road geometry in browser code.

### WEB-B03 — Draft model and approved operation set are incomplete

**Evidence:** Source-confirmed and test-covered.

- `WorkspaceStore.ALLOWED_OPERATIONS` permits only `move_day`, `mark_locked`,
  `mark_flexible`, and `route_feedback`.
- Selection and draft operations mutate only the in-memory `ProductSession`.
- The test suite confirms custom Move Day preview is ineligible.
- Nearby replacement, add candidate, selected candidate context, persistent draft
  restoration, and general compilation are absent.

**Required closure:** Complete the typed union, persist ordered drafts, compile
operations into repair requests, and independently evaluate every produced child.

### WEB-B04 — Copilot has no real provider boundary implementation

**Evidence:** Source-confirmed.

- `ProductService.__init__` always constructs `FixtureCopilotAdapter`.
- `FixtureCopilotAdapter.interpret` is synchronous keyword matching.
- No OpenAI import, transport abstraction, strict response schema, timeout, retry,
  cancellation, transcript storage, or delete API exists.

**Required closure:** Add the approved provider adapter without changing the
deterministic adapter’s role as the CI oracle. Never silently fall back.

### WEB-B05 — Copilot bypasses the draft lifecycle for its registered repair

**Evidence:** Source-confirmed.

`api.py:167-170` turns a fixture interpretation directly into
`registered_proposal()`, which sets the frozen child as the current proposal. It
does not create a typed draft or execute the repair pipeline.

**Required closure:** Interpretation yields clarification, permission, or typed
proposal input. Drafting, preview, evaluation, and acceptance remain separate.

### WEB-B06 — Acceptance is not backed by `PlanRepository`

**Evidence:** Source-confirmed.

- `WorkspaceStore.accept()` appends a decision JSON and atomically replaces
  `workspace_pointer.json`.
- It checks session proposal state and detects an intervening pointer.
- No `PlanRepository` reference exists under `product_app/`, and no child is saved
  or re-verified through that repository during acceptance.

**Impact:** The current local acceptance boundary is useful but does not satisfy
the approved durable acceptance contract.

**Required closure:** Verify hash/certificate/diff lineage, save the immutable
child through `PlanRepository`, append the ledger event, and compare-and-swap the
current pointer.

### WEB-B07 — Compare Inspect invokes a decision mutation

**Evidence:** Source-confirmed.

`renderCompare()` labels the baseline card “Inspect option,” then calls
`keepOriginal()`, which posts to `/keep-original`. This is a critical handler-
semantics bug.

### WEB-B08 — Required mobile and PWA application boundaries are absent

**Evidence:** Source-confirmed.

- Client routing has no `/app/day/{day}` or `/app/copilot` mobile route.
- There is no map editing bottom sheet.
- No `manifest.webmanifest`, service worker, or service-worker registration exists.

### WEB-B09 — Test coverage cannot support the verification claim

**Evidence:** Source-confirmed.

The focused suite covers registry, shell/security headers, fixture interpretation,
one route-feedback acceptance path, one ineligible Move Day path, Keep original,
safe evidence ZIP names, and pointer conflict. It does not cover:

- A real browser map runtime or geographic correctness.
- Map marker/segment selection in a browser.
- All control handlers.
- Distinct alternative plans.
- Provider failure states.
- Transcript retention/deletion.
- Session/draft restoration after process restart.
- `PlanRepository` acceptance.
- Required mobile routes, virtual keyboard, PWA, or accessibility automation.
- A clean Windows launch-to-decision journey.

## Security and integrity baseline

Implemented strengths:

- Loopback-only launcher host choice.
- Mutation token and expected-revision checks.
- Non-loopback Origin rejection for mutations.
- CSP, `nosniff`, frame denial, and no-referrer policy.
- Registry traversal/absolute-path rejection.
- Evidence ZIP paths resolved through the safe run-relative-path helper.
- Append-only decision creation and atomic pointer replacement.
- Pointer conflict test.
- Parent artifact hash test around acceptance.

Remaining gaps:

- Host header validation is not evident.
- CSP has no reviewed Atlas/worker boundary because Atlas/PWA are absent.
- Session state is memory-only and has no expiry/restore policy.
- Full transcripts, TTL, deletion, and log-redaction behavior are absent.
- Acceptance trusts the loaded bundle fields instead of repository/hash/certificate
  re-verification at the transaction boundary.
- `GET /api/sessions/{id}/alternatives` reads session state without a token. This is
  loopback-limited but should be assessed against the final session privacy model.

## Recorded user-like manipulation

The 2026-08-02 audit recorded:

- Fixture Copilot request: worked.
- Day-card selection: worked.
- Map-marker selection: did not visibly update in the tested path.
- Route-feedback draft: worked.
- Undo: worked.
- Compare and Evidence: opened.
- 360/390/430 document overflow: not observed.
- Acceptance: deliberately not executed against the user’s real state.

These observations prove neither the full journey nor restart persistence.

## Gate recommendation

Treat the product as `in-progress`. Preserve the working service, registry,
security, fail-closed preview, and pointer-conflict foundations. Close geographic
mapping at G2, general drafts at G3, Copilot at G4, repository-backed decisions at
G5, mobile/PWA at G6, launch at G7, and replacement verification at G8.
