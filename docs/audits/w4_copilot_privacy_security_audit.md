# W4 Copilot Privacy and Security Audit

**Audit date:** 2026-08-06  
**Audit role:** `COPILOT_PRIVACY_SECURITY_AUDITOR`  
**Independence:** Read-only review performed independently of the W4 implementation agents. The auditor made no production-code, configuration, state, or provider changes and made no live OpenAI request.  
**Offline implementation verdict:** **PASS**  
**G4 verification verdict:** **BLOCKED**

## Scope

The audit covered the W4 OpenAI transport, prompt construction and injection
boundary, typed interpretation validation, local conversation retention and
deletion, FastAPI Host/Origin/cache/CSP controls, `.env.local` loading, frontend
rendering, evidence/log exclusions, provider fallback behavior, and W5
fail-closed boundaries.

The audit did not read or print `.env.local`, did not inspect its value, and did
not send any request to OpenAI. Presence, Git tracking, and ignore status were
checked as metadata only.

## Evidence and commands

| Evidence | Result |
|---|---|
| `git check-ignore -v -- .env.local` and `git ls-files -- .env.local` | File is ignored by `.gitignore:68`, present, and untracked. Its contents were not read. |
| Focused deterministic suite: `python -m pytest tests\product_app\test_copilot_privacy.py tests\product_app\test_openai_copilot.py tests\product_app\test_copilot_prompt.py tests\product_app\test_copilot_contracts.py tests\product_app\test_conversations.py tests\product_app\test_security_api.py tests\product_app\test_launcher.py -q --basetemp <unique-temp>` | **101 passed**, one upstream Starlette deprecation warning, no failures. |
| `python -m ruff check src\itinerary_system\product_app tests\product_app scripts\run_product_app.py scripts\evaluate_copilot_prompt.py` | Passed. |
| `git diff --check --` on the W4 application, tests, launcher, prompt evaluator, ignore rule, and W4 phase plan | Passed; Git reported only the existing LF-to-CRLF working-copy warning for `.gitignore`. |
| Offline provider-routing probe with a sentinel key and `OPENAI_BASE_URL=http://127.0.0.1:9999/v1` | `AsyncOpenAI` resolved its base URL to `http://127.0.0.1:9999/v1/`; no request was sent. |
| W4-SEC-001 re-audit: source inspection plus `python -m pytest tests\product_app\test_openai_copilot.py tests\product_app\test_copilot_api.py -q --basetemp <unique-temp>` | Production pins `base_url="https://api.openai.com/v1"`; the hostile-environment regression test passed; **32 passed**, one upstream Starlette deprecation warning. |
| Re-audit Ruff: `python -m ruff check src\itinerary_system\product_app\openai_copilot.py tests\product_app\test_openai_copilot.py tests\product_app\test_copilot_api.py` | Passed. |

## Controls verified

- Provider-bound context is constructed from a fixed public
  `CopilotContextV1` allow-list and bounded recent conversation history. It
  excludes mutation tokens, token salts/verifiers, filesystem paths, raw
  artifacts, and evidence-file contents.
- The code-owned prompt labels workspace context, recent history, and the user
  request as untrusted data and explicitly rejects instruction disclosure,
  invented identifiers, unsupported facts, and automatic execution.
- Provider output uses strict Pydantic structured output and then passes a
  second application-owned validator. Intent types, target stop IDs, candidate
  IDs, days, highlights, permissions, and state combinations fail closed when
  outside the current workspace universe.
- The Responses request uses `store=False`, an empty tools list, bounded input,
  bounded output, a timeout, one transient-only retry, cancellation
  propagation, and sanitized stable errors.
- Selecting `openai` without a key does not fall back to the deterministic
  adapter. Deterministic mode is explicit, network-free, and identified in the
  API and UI.
- Conversations are local JSON records with validated IDs, bounded file/turn
  counts, a 30-day sliding expiry, atomic replacement under the shared process
  lock, symlink/non-file rejection, ownership checks, delete-one, and
  delete-all.
- Delete-all requires a valid session ID/token, matching session revision,
  exact confirmation text, and an exact allowed Origin. Host/Origin failures
  remain no-store and do not add permissive CORS headers.
- API successes and errors receive `Cache-Control: no-store`; the app shell is
  `no-cache`. CSP is self/validated-loopback scoped, framing and object loading
  are denied, and the browser renders user/model transcript text with
  `textContent` rather than HTML.
- Captured tests demonstrate that keys, session tokens, paths, provider error
  payloads, and transcript sentinel text do not enter health, evidence ZIPs,
  API errors, or captured logs.
- W4 advisory responses do not change drafts or accepted plans automatically;
  frontend draft addition requires explicit confirmation. W5 Accept and Keep
  original endpoints remain fail closed with
  `acceptance_not_enabled_until_w5`.
- The bounded `.env.local` loader rejects symlinks, paths outside the
  repository, oversized/non-UTF-8/malformed files, duplicate or unknown keys,
  shell syntax, and does not overwrite an already-defined process variable.

## Findings

### W4-SEC-001 — High — provider destination could be redirected by ambient environment

**Status:** **Resolved and independently re-verified on 2026-08-06.**

`OpenAICopilotAdapter._require_client()` constructs `AsyncOpenAI` with the API
key, timeout, and retry policy but does not provide or validate `base_url`.
The official SDK therefore honors an ambient `OPENAI_BASE_URL`. An offline
sentinel probe confirmed that setting this variable to
`http://127.0.0.1:9999/v1` changes the client's destination to that URL. If a
Copilot request were then made, the Bearer key and bounded trip/transcript
payload would be sent to the substituted endpoint.

This variable is not accepted by the repository's `.env.local` loader, which
is a useful control, but it can still be inherited from the launching process.
The current tests verify outbound *payload fields* but do not assert the
outbound *destination*.

**Closure evidence:** `OpenAICopilotAdapter._require_client()` now passes the
literal approved endpoint `https://api.openai.com/v1` to `AsyncOpenAI` while
retaining `max_retries=0` and the configured timeout. The regression test sets
a hostile `OPENAI_BASE_URL=http://127.0.0.1:9999/v1/` and asserts that the SDK
factory still receives the pinned HTTPS OpenAI endpoint. The focused provider
and API suite passed 32/32 and Ruff passed. No request was sent and no key or
payload was printed. W4-SEC-001 is closed.

### W4-G4-001 — Gate evidence — live provider smoke is not yet authorized/executed

**Status:** Blocking G4 verification only; not an offline implementation defect.

No authorized live OpenAI smoke test was performed in this audit. Consequently
there is no durable evidence yet for credential authentication, access to the
configured `gpt-5.6-terra` model, one real structured Responses result,
provider refusal/error behavior against the live service, or the live
no-fallback user journey.

**Required closure:** After W4-SEC-001 is fixed and the user authorizes the
bounded billed request, run the plan's secret-safe live smoke and record only
model/prompt/schema versions, stable outcome codes, latency/token aggregates,
and pass/fail evidence. Never record the key, raw prompt, raw response, or
transcript. The larger prompt-evaluation batch still requires its separate
cost preflight and user approval.

## Gate conclusion

The deterministic W4 privacy implementation now passes this independent audit.
The previously blocking provider-destination redirect is fixed and
independently re-verified. No offline privacy/security blocker remains in this
audit scope.

G4 nevertheless remains **BLOCKED solely on the missing authorized live-provider
evidence described in W4-G4-001**. W4/G4 must not be marked verified and W5 must
not be opened until that bounded live smoke is authorized, completed, recorded
without sensitive data, and accepted by the remaining required auditors.
