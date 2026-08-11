# W4 / G4 Verification Report

**Generated:** 2026-08-06T20:43:16-05:00  
**Track:** Local non-publication Itinerary Repair Copilot  
**Verdict:** W4 `implemented`; G4 `blocked`; W5 `planned` / not ready

## Outcome

The deterministic W4 implementation is complete and has no remaining offline
blocking finding. The application now provides a provider-neutral typed Copilot,
an OpenAI Responses adapter, fixed/versioned prompt controls, artifact-validated
context, durable local conversations with deletion and retention, explicit
privacy disclosures, and browser-accessible desktop/mobile Copilot behavior.

G4 is not verified. The user authorized one bounded live smoke request on
2026-08-06. Exactly one request using a synthetic, allow-listed itinerary
context reached OpenAI with model `gpt-5.6-terra`, but the adapter failed closed
with `openai_invalid_structured_output` (application HTTP status 502). The
current sanitizer intentionally collapses provider 400/404/409/422 errors,
provider parse/validation failures, and missing parsed output into that stable
application code, so this run cannot distinguish model/request incompatibility
from malformed structured output without another request using improved safe
diagnostics. No second external request was made, no application state changed,
and no raw prompt, response, transcript, credential, or provider exception was
recorded. W5 therefore remains closed.

## Post-smoke corrective diagnostics

The adapter now retains only bounded, non-textual diagnostic metadata on its
local exception object: a fixed diagnostic category, numeric provider status,
and a provider error code only when it matches a strict 64-character safe
pattern. It distinguishes invalid requests, model/endpoint absence, response
schema failures, incomplete output, content filtering, authentication, quota
exhaustion, ordinary rate limiting, and transport failures. Raw provider error
messages and bodies remain discarded and are not added to API responses or
conversation records.

Focused verification passed after this change: 49 provider/privacy/API tests,
then a final 42-test provider/API run, plus Ruff and `git diff --check`. The
existing Starlette TestClient deprecation warning remains informational. These
changes cannot retroactively classify the consumed smoke request; one newly
authorized request is required to capture the safe category.

## Diagnostic smoke rerun

The user separately authorized one diagnostic smoke request on 2026-08-06.
Exactly one request reached OpenAI and failed closed with provider HTTP 400,
provider code `unsupported_parameter`, and local diagnostic category
`provider_request_invalid`. Authentication, model availability, quota, rate
limits, transport, and model-output schema parsing were therefore not the
failure class. No retry or state mutation occurred, and no raw provider data
was recorded.

The safe diagnostics did not yet retain the provider's parameter name. Current
official GPT-5.6 guidance supports `reasoning.context=current_turn` and
documents verbosity under `text.verbosity`. The installed SDK accepts a
top-level `verbosity` convenience argument and transmits it as a top-level
request field. That field is the leading compatibility suspect, but this is an
inference until a reviewed payload correction and newly authorized smoke pass.

## Corrective request-payload patch

The adapter now sends `text={"verbosity": "low"}` and no longer emits a
top-level `verbosity` request field. This matches the phase contract and current
official GPT-5.6 guidance while leaving `reasoning.context=current_turn`, strict
structured output, `store=false`, no tools, and all prompt limits unchanged.

Offline verification passed: 56 focused provider, prompt, privacy, and API
tests; Ruff; and the complete 306-test product-app suite. One existing Starlette
TestClient deprecation warning remains informational. The first product-suite
attempt reached approximately 87% without a failure but exceeded its
three-minute command limit; the clean rerun completed with all 306 tests
passing. No OpenAI request was made during this corrective step.

## Stable verification evidence

- Repository project checks passed on the stable post-fix tree:
  - Ruff: pass.
  - Context snapshot: 5 passed.
  - Full pytest: 616 passed, 1 existing Starlette TestClient deprecation warning.
- Product suite: 296 passed on the integrated W4 tree; later focused prompt,
  provider, status, accessibility, and browser checks passed after the final
  canonical-copy changes.
- Browser W4 journey: 8 passed, including 390x360 reduced-height behavior,
  provider-failure visibility, mobile focus containment, Escape close, and
  focus restoration.
- Nine independent reports are durable: the eight required provider, privacy,
  transcript, map-context, accessibility/content, user-journey, prompt, and
  phase/status audits, plus an additional cross-module integration audit.
- `.env.local` is ignored and untracked. The key value was never written to an
  audit report, test output, or evidence bundle.

## Closed audit findings

- The OpenAI client destination is pinned to `https://api.openai.com/v1`; an
  ambient `OPENAI_BASE_URL` cannot redirect the key or Copilot context.
- Provider-authored assistant and clarification prose is discarded before
  validation, persistence, and API presentation. Local canonical copy is used.
- Prompt evidence requires either the complete 72-case offline corpus or the
  exact fixed 24-case live subset spanning all nine categories, with explicit
  thresholds.
- Segment, candidate, and alternative context is checked against artifact
  inventories and synchronized into visible Copilot context.
- A failed transcript deletion restores the session binding so retained bytes
  remain reachable and retryable.
- Provider failures remain visibly failed; mobile keyboard focus stays inside
  the full-screen Copilot and all relevant controls meet the 44px target.
- Copilot output remains advisory. Accept and Keep-original return the W5
  fail-closed response and do not mutate a plan or pointer.

## Remaining G4 evidence gate

Both one-request authorizations have been consumed. A new explicit
authorization is required before any further billed/external request. The
request-parameter compatibility patch is implemented offline; next run:

1. One bounded live smoke using the visible itinerary context, the submitted
   test message, and no more than 12 recent messages / 12,000 history characters.
2. The exact fixed 24-case stratified evaluation without storing raw provider
   prompts, responses, transcripts, or credentials.
3. The planned low-versus-medium reasoning comparison on the same approved
   subset and record aggregate latency/token/quality results.
4. Re-run stable project checks and update the sanitized manifest with live
   result metadata only.

Until those steps pass, W4 must not be described as verified, G4 must remain
blocked, and W5 must remain planned/not ready. CP-010 stays `in-progress`.
E3.1, E3.3, and E4 remain `blocked`; E3.UX5 and E5 remain `deferred`.
