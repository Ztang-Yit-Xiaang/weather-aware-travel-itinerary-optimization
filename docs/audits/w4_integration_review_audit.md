# W4 Independent Integration Review Audit

**Audit date:** 2026-08-06

**Audit role:** `INTEGRATION_REVIEW_AGENT`

**Independence:** This review was performed after the implementation and the
eight specialist audits. The reviewer did not implement the W4 provider,
prompt, conversation, API, frontend, persistence, or status changes. This
report is the reviewer's only repository edit. No OpenAI request was made and
the local API-key file was not read.

**Offline integration verdict:** **PASS**

**Recommended W4 status:** **`implemented`**

**G4 verdict:** **`blocked`**

**W5 readiness:** **`planned` / not ready**

## Scope and acceptance boundary

This audit reviewed the final W4 offline integration across:

- OpenAI Responses transport construction, destination pinning, retry,
  cancellation, strict schema, and sanitized errors;
- prompt construction, fixed evaluation scopes, safety/quality thresholds,
  and injection handling;
- application-owned canonicalization of every provider-authored free-text
  field before persistence or API exposure;
- artifact-derived Copilot context allowlists and post-provider validation;
- conversation ownership, bounded persistence, deletion compensation, and
  exclusion from health/evidence;
- frontend provider disclosure, failure lifecycle, mobile focus containment,
  persistence, deletion, and safe text rendering;
- advisory-only Copilot behavior and the W5 fail-closed decision boundary;
- secret handling and the local environment-loader boundary; and
- phase/gate truth and the non-propagation boundary for research statuses.

The audit intentionally did not test credential authentication, live access to
`gpt-5.6-terra`, live model quality, latency, token use, refusal behavior, or
the planned low-versus-medium reasoning comparison. Those are G4 live-evidence
requirements and require explicit user authorization.

## Specialist reports reviewed

All eight required W4 audit reports were read as evidence, not treated as
automatic proof:

| Specialist audit | Current offline verdict | Integration conclusion |
| --- | --- | --- |
| Provider/schema | Pass | Strict Responses contract and fail-closed transport pass; live access remains unproved |
| Privacy/security | Pass after re-audit | Ambient `OPENAI_BASE_URL` redirect was repaired and independently re-tested |
| Transcript retention | Pass after re-audit | Delete-current failure restores the owning session binding; additional low-level failure injection remains hardening work |
| Map/context synchronization | Pass after re-audit | Selected stop/day/segment/candidate/alternative values are artifact-bound and advisory-only |
| Accessibility/content | Pass after re-audit | Persistent failure state, mobile focus containment, Escape restoration, and current Copilot target sizing pass |
| User-journey black box | Pass | Served deterministic desktop/mobile journey passes; this is not live-provider evidence |
| Prompt quality/injection | Pass after three re-audits | Fixed scopes, explicit thresholds, and canonical local copy close the prior prompt-evidence and provider-prose findings |
| Phase/gate/status | W4 implemented; G4 blocked | Status documents must be updated by the orchestrator without opening W5 or changing research gates |

No specialist report has a remaining offline W4 blocker. Their verdicts are
consistent with the source and tests reviewed here.

## Cross-module review findings

### Provider destination and secret boundary

`OpenAICopilotAdapter._require_client()` pins
`https://api.openai.com/v1`, disables SDK retries, and supplies the API key only
to the server-side SDK constructor. A hostile ambient `OPENAI_BASE_URL` is
covered by a regression test and cannot redirect the configured client.

The request uses `store=False`, `tools=[]`, a bounded prompt, a bounded output,
one application-owned transient retry, and an outer deadline. Raw provider
exceptions, refusals, and output text do not enter user-facing errors. The key
is absent from object representations and public health/configuration payloads.
`.env.local` is ignored and untracked; this audit checked only ignore/tracking
metadata and did not read the file.

### Provider free text and typed interpretation

The provider schema has only two free-text fields: `assistant_message` and an
optional `clarification_question`. After strict Pydantic parsing, the adapter
replaces `assistant_message` with state-derived application copy and replaces
the clarification question for `request_clarification` with application-owned
copy. A clarification field on any other intent is rejected by the local intent
contract. Enums, days, stop IDs, candidate IDs, segment highlights, and state /
permission combinations are all revalidated locally.

As a result, provider-authored claims such as booking/payment completion do not
reach the transcript or API even when supplied in both provider text fields.
This is stronger than phrase filtering because no provider prose is retained.

### Prompt evaluation scope and thresholds

The evaluator requires exactly all 72 corpus cases for `offline-fake` and
exactly the fixed 24-case stratified subset for any live model label. Duplicate,
missing, extra, unsafe-field, invalid-reference, and unsafe model-label inputs
fail closed. The fixed live subset spans all nine categories.

The emitted summary includes an explicit `thresholds_passed` result and rejects
evidence below these gates: 100% strict schema, zero unsafe automatic-action
cases, zero invalid identifiers, 100% external-action and injection fail-safe
rates, and at least 95% exact state/intent and clarification rates. The 72/72
offline result is correctly treated as evaluator/harness evidence, not live
model-quality evidence.

### Artifact context and mutation boundary

Provider context is built from a fixed `CopilotContextV1` allowlist. Stop IDs,
route segment IDs, candidates, alternatives, and days are derived from the
validated workspace/geography artifacts. The current draft is reduced to typed
operation data and the evaluated proposal is reduced to a decision-safe field
allowlist. Mutation tokens, salts/verifiers, raw artifact files, evidence-file
contents, filesystem paths, and acceptance state are not included.

The model result is advisory. It cannot directly invoke a draft, preview,
permission, acceptance, booking, filesystem, or map-geometry operation. Draft
actions remain explicit W3 operations. Both Accept and Keep original continue
to return `409 acceptance_not_enabled_until_w5` after authentication and
revision checks.

### Conversation deletion and frontend lifecycle

Delete-current first detaches the session and then deletes the owned file. If
the repository deletion raises `ConversationError`, the service restores the
same conversation ID before re-raising the stable failure. The API regression
proves the original conversation ID and turns remain reachable after the
injected failure. Normal persistence, retention, restart, ownership, idempotency,
size/count, and delete-all controls are separately covered.

The browser uses `textContent` for transcript and context rendering. A failed
provider request remains visibly failed after transcript restoration, retains
the user's composer text, and does not become a false idle/success state. On
mobile, the open dock becomes a modal dialog with a focus loop; Escape closes it
and restores the opener. Reduced-height 390 x 360 checks keep the composer and
Send action inside the viewport. Desktop semantics revert to a complementary
region.

## Verification performed

### Corrected focused cross-module suite

```powershell
$auditTemp = Join-Path ([System.IO.Path]::GetTempPath()) `
  ('w4-integration-audit-' + [guid]::NewGuid().ToString('N'))
python -m pytest `
  tests\product_app\test_openai_copilot.py `
  tests\product_app\test_copilot_prompt.py `
  tests\product_app\test_copilot_contracts.py `
  tests\product_app\test_copilot_api.py `
  tests\product_app\test_copilot_privacy.py `
  tests\product_app\test_conversations.py `
  tests\product_app\test_geography.py `
  tests\product_app\test_w3_preview.py `
  tests\product_app\test_copilot_frontend_contract.py `
  tests\product_app\browser\test_w4_copilot_journey.py `
  -q --basetemp $auditTemp
```

Result: **121 passed**, one existing Starlette/TestClient deprecation warning,
no failures, in 133.33 seconds. This post-fix run includes provider, prompt,
API, privacy, conversation, artifact geography, W3 preview, frontend contract,
and served-browser coverage.

An initial invocation named a nonexistent
`tests\product_app\test_compiler_integration.py`; pytest correctly exited with
`file or directory not found` and ran no tests. The command was corrected to
the existing `test_w3_preview.py` and rerun as shown above. The failed command
is not counted as evidence.

### Static analysis

```powershell
python -m ruff check `
  src\itinerary_system\product_app `
  tests\product_app `
  scripts\run_product_app.py `
  scripts\evaluate_copilot_prompt.py
```

Result: **PASS** (`All checks passed!`).

### Scoped diff and secret metadata checks

```powershell
git diff --check -- .gitignore README.md pyproject.toml `
  scripts/run_product_app.py scripts/evaluate_copilot_prompt.py `
  src/itinerary_system/product_app tests/product_app `
  docs/planning/w4_copilot_provider_transcript_phase_plan.md
git check-ignore -v -- .env.local
git ls-files -- .env.local
```

Result: **PASS**. `git diff --check` reported only informational LF-to-CRLF
working-copy warnings. `.env.local` is ignored by `.gitignore:68` and is not
tracked. Its contents were not read.

A secret-pattern scan over the reviewed W4 source, tests, scripts, audit
reports, and secret-free example configuration found no credential-shaped
value. The only `sk-` match was the deliberate invalid-model-label test value
`sk-secret-shaped-value`.

## Findings and residual risks

### W4-INT-001 - No offline integration blocker found

**Severity:** Gate conclusion

**Status:** Pass.

The reviewed production boundaries, eight independent reports, and 121-test
post-fix integration run did not reveal a remaining deterministic W4 blocker.
A passing suite is evidence rather than proof; the residual items below remain
explicit.

### W4-INT-002 - Live provider evidence is absent

**Severity:** Blocking G4 verification

**Status:** Open.

No authorized request has established authentication, current project/model
access, one real structured response, live error/refusal behavior, or browser
lifecycle against the provider. The fixed 24-case live evaluation and the
planned low-versus-medium reasoning comparison have not run. These actions send
bounded itinerary context externally and may incur cost, so they require the
user's explicit authorization. A saved key or `openai_configured` health code
is not evidence for this gate.

### W4-INT-003 - Final full-suite evidence is not atomic after the last fix

**Severity:** Procedural G4 blocker

**Status:** Open.

The prior full project run passed 615 tests, but the final canonical-copy change
occurred after that run collected tests. The post-fix 121-test suite covers the
changed boundary and supports the W4 offline implementation verdict. Before G4
verification, the orchestrator must run one stable full project-check cycle
after W4 files stop changing and record it in the evidence manifest.

### W4-INT-004 - Non-blocking hardening and later-wave evidence

**Severity:** Low / later scope

**Status:** Open, non-blocking for offline W4.

- HTTP-date-form `Retry-After` is not parsed; the provider still performs at
  most one retry inside the deadline, but may retry immediately instead of
  respecting that form.
- Additional low-level unlink/delete-all failure injection and concurrent
  delete/rebind testing would strengthen transcript compensation evidence.
- A real screen-reader pass, complete automated accessibility scan, and
  real-device virtual-keyboard evidence remain W6/final-verification work.
- Deterministic responsive checks establish functional geometry, not final
  mockup visual sign-off.

### W4-INT-005 - Product status documents lag the evidence

**Severity:** Medium status-integrity finding

**Status:** Open for the orchestrator.

The active documents still say W4 `in-progress` and G4 `planned`. After this
offline integration sign-off, the truthful transition is W4 `implemented`, G4
`blocked`, and W5 `planned` / not ready. This review intentionally did not edit
status authority files.

## Recommended phase and gate statuses

| Item | Recommended status | Reason |
| --- | --- | --- |
| W4 | `implemented` | Eight specialist audits and this post-fix integration review found no offline implementation blocker |
| G4 | `blocked` | Authorized live smoke, fixed 24-case evaluation, reasoning comparison, final stable full checks, and sanitized evidence package are absent |
| W5 | `planned` / not ready | G4 is not verified; Accept and Keep original remain fail closed |
| CP-010 | `in-progress` | The corrective product track continues through G8 |
| E3.1, E3.3, E4 | unchanged: `blocked` | Product work is not research/publication evidence |
| E3.UX5, E5 | unchanged: `deferred` | Product Copilot work cannot satisfy interaction-research or ethics gates |

The orchestrator should update only the product-track status language. W5 must
not start until G4 is verified. The next gate-producing action is to obtain
explicit user authorization for the bounded live smoke and fixed evaluation;
without that authorization, stop at W4 `implemented` / G4 `blocked`.

## Final verdict

The W4 deterministic/offline integration is **PASS** and supports marking W4
`implemented`. G4 remains **BLOCKED**, not verified, because the required live
provider and model-quality evidence does not exist. W5 remains planned and
closed. No research phase or gate may advance from this product evidence.
