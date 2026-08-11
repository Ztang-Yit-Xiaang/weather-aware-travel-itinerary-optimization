# W4 Map–Copilot Context Synchronization Audit

**Audit date:** 2026-08-06  
**Auditor:** `MAP_CONTEXT_SYNC_AUDITOR` (`/root/audit_map_context_sync`)  
**Scope:** W4 context construction, map/itinerary selection propagation, provider-output highlights, stale revisions, and advisory-only behavior  
**Initial offline verdict:** **BLOCKED**  
**Re-audit offline verdict:** **PASS**  
**G4 status:** **Blocked on remaining G4 evidence; W5 must remain planned**

## Independence statement

This review was performed in a separate audit-agent context. The auditor did
not implement or modify W4 production code and did not audit its own work. No
production source file was changed by this audit. Findings were derived from
independent source tracing, adversarial counterexamples, and focused automated
tests.

## Acceptance criteria audited

1. Copilot context is derived from the authenticated current session and the
   registered artifact/candidate universe.
2. Day, stop, segment, candidate, and alternative selections propagate
   truthfully between the map/workspace and Copilot.
3. Provider intents and highlights containing invented identifiers fail closed.
4. A stale session revision cannot append an apparently current proposal.
5. Copilot output remains advisory: it cannot directly mutate drafts,
   permissions, proposals, accepted-plan state, or acceptance decisions.
6. Copilot highlighting selects validated entities only and does not fabricate
   route geometry.

## Evidence and commands

### Focused automated suite

```powershell
$auditTemp = Join-Path ([System.IO.Path]::GetTempPath()) `
  ('w4-map-context-audit-' + [guid]::NewGuid().ToString('N'))
python -m pytest `
  tests\product_app\test_copilot_api.py `
  tests\product_app\test_openai_copilot.py `
  tests\product_app\test_copilot_prompt.py `
  tests\product_app\test_geography.py `
  tests\product_app\test_geographic_frontend_contract.py `
  -q --basetemp $auditTemp
```

Result: **58 passed, 1 warning in 33.16s**. The warning is the existing
Starlette `TestClient`/`httpx` deprecation warning.

### Source trace

- `ProductService.copilot_context()` constructs the provider context from the
  current authenticated session and artifact-derived stop, candidate, and day
  universes.
- `validate_copilot_interpretation()` revalidates intent stop IDs, candidate
  IDs, destination days, and highlight stop/candidate/day IDs.
- The service authenticates and checks the expected revision before the
  provider call and repeats both checks after it returns, before appending the
  turn.
- The browser revalidates highlight targets before calling the ordinary
  selection endpoint. It creates no route geometry from model output.
- Adding an interpreted edit requires a separate user confirmation and then
  calls the ordinary typed-draft endpoint. Preview requires a second explicit
  confirmation. Compare and Evidence only navigate.
- W5 Accept and Keep-original endpoints remain fail-closed with
  `acceptance_not_enabled_until_w5`.

## Findings

### MCS-001 — Resolved on 2026-08-06: selected segment and alternative artifact membership

`CopilotContextV1` includes `selected_segment_id` and
`selected_alternative_id`, and `ProductService.copilot_context()` copies both
from the persisted session. However, the context includes allowlists only for
stops, candidates, and days. `_validate_context()` therefore checks a selected
segment or alternative only for non-empty string shape, not membership in the
current artifact universe.

The normal selection API does validate these IDs when the user first selects
them. That is useful but insufficient at the provider boundary: a stale session
snapshot after a registry/artifact change, or corrupted local state, can carry
an identifier no longer present in the current run and have it serialized into
the provider prompt as workspace context. This contradicts the W4/G4 rule that
Copilot context uses artifact-valid identifiers.

Required correction before G4:

- add artifact-derived `allowed_segment_ids` and `allowed_alternative_ids` to
  the context contract, or revalidate both selected fields directly in
  `ProductService.copilot_context()` immediately before provider dispatch;
- add tests for stale/forged persisted selections and artifact-universe changes;
- keep invalid values out of both provider prompts and persisted turns.

Re-audit result: **resolved**. `CopilotContextV1` now carries
`allowed_segment_ids` and `allowed_alternative_ids`.
`ProductService.copilot_context()` derives both universes from the registered
run, and `_validate_context()` rejects a selected segment or alternative that
is absent from its current universe. Highlighted segments are validated against
the complete artifact-derived segment universe. Regression coverage explicitly
constructs stale/forged segment and alternative selections and requires
`CopilotContractError`.

### MCS-002 — Resolved on 2026-08-06: complete selection synchronization and visibility

The map-edit candidate dropdown is populated from registered candidate choices,
but changing it does not call the session selection endpoint. It is used only
when a later draft action is submitted. Consequently, merely selecting a map
candidate does not update `selected_candidate_id` and the Copilot cannot receive
that selection as current context.

In addition, `renderContext()` displays only run, day, stop, and revision. It
does not render the current candidate, segment, or alternative from the server
session. Compare can set `selected_alternative_id`, and Copilot highlighting can
set a candidate/segment, but the user cannot verify those synchronized values in
the Copilot context chips. This contradicts the phase contract that map-to-
Copilot chips reflect current server session context.

Required correction before G4:

- synchronize registered-candidate selection through `selectContext()` without
  creating a draft;
- render accessible context chips for selected candidate, segment, and
  alternative, with truthful labels resolved from validated workspace data;
- add a browser journey covering map candidate selection, compare alternative
  selection, Copilot close/reopen, and Ctrl+R persistence.

Re-audit result: **resolved for the W4 implementation boundary**. The registered
candidate control now submits `selected_candidate_id` through the ordinary
revision-checked selection endpoint on change. Reopening map edit restores the
selected candidate. Copilot context chips now expose selected candidate,
segment, and alternative values in addition to run, day, stop, and revision.
Static regression checks cover the event binding and all three chip categories;
the existing browser journey continues to cover context restoration across
Copilot close/reopen and Ctrl+R without console errors.

### MCS-003 — No blocking defect found: advisory-only and stale-revision boundaries

The focused suite and source trace support the following:

- invalid intent/highlight stop, candidate, and day IDs fail closed;
- a revision change during provider processing returns
  `stale_session_revision` and appends no turn;
- successful Copilot interpretation leaves the session revision, draft,
  evaluated proposal, accepted plan, and permission decisions unchanged;
- adding a suggestion to a draft is a distinct confirmed action through the
  typed W3 endpoint;
- provider output does not create route geometry; Show-on-map submits only
  revalidated selections;
- W5 decision endpoints remain disabled.

This evidence does not prove absence of every mutation defect; it covers the
specified W4 paths and adversarial cases exercised by the focused tests.

## Severity summary

| ID | Severity | Blocking G4 | Result |
|---|---|---:|---|
| MCS-001 | High | Resolved | Artifact-derived segment/alternative universes and rejection tests added |
| MCS-002 | High | Resolved | Candidate selection and complete context-chip synchronization added |
| MCS-003 | Informational | No | Advisory-only and stale-revision controls passed focused verification |

## Verdict and gate recommendation

The W4 advisory execution boundary and repaired Map–Copilot synchronization
contract are materially sound in the tested paths. MCS-001 and MCS-002 are
resolved. The independent re-audit offline verdict is **PASS** with no remaining
Map–Copilot blocker found.

- Do not mark G4 verified.
- Do not mark W5 ready or begin W5.
- W4 may be `implemented` under the orchestrator's implementation-evidence
  rules. G4 remains blocked until live-provider evidence and every other
  required independent audit pass.
- Research gates E3.1, E3.3, E4, E3.UX5, and E5 remain unchanged.

## Remaining external limitation

No live OpenAI request was authorized or executed during this audit. Even after
the two synchronization findings were repaired, live provider smoke/evaluation
and the other required independent audits remain separate G4 requirements.

## Re-audit evidence — 2026-08-06

The auditor independently re-read the changed context model, service context
builder, provider-neutral validator, selection event binding, map-edit restore
logic, Copilot context rendering, and regression tests. No production file was
edited during re-audit.

```powershell
$auditTemp = Join-Path ([System.IO.Path]::GetTempPath()) `
  ('w4-map-context-reaudit-' + [guid]::NewGuid().ToString('N'))
python -m pytest `
  tests\product_app\test_copilot_contracts.py `
  tests\product_app\test_copilot_api.py `
  tests\product_app\test_openai_copilot.py `
  tests\product_app\test_copilot_prompt.py `
  tests\product_app\test_geography.py `
  tests\product_app\test_geographic_frontend_contract.py `
  tests\product_app\browser\test_w4_copilot_journey.py `
  -q --basetemp $auditTemp
```

Result: **85 passed, 1 warning in 52.35s**. The warning is the existing
Starlette `TestClient`/`httpx` deprecation warning.

```powershell
python -m ruff check src\itinerary_system\product_app tests\product_app
```

Result: **passed**.
