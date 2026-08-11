# W4 Provider and Schema Audit

**Audit role:** `PROVIDER_SCHEMA_AUDITOR`  
**Audit date:** 2026-08-06  
**Implementation independence:** The auditor did not implement the reviewed provider, prompt, service, runtime, API, configuration, or test files. The only repository edit made by this audit is this report.  
**Offline implementation verdict:** **PASS**  
**G4 gate verdict:** **BLOCKED** pending the separately authorized live OpenAI smoke/evaluation evidence and the other required independent G4 sign-offs.

## Scope

The audit compared the W4 provider, prompt, validation, and error contracts in
`docs/planning/w4_copilot_provider_transcript_phase_plan.md` with:

- `src/itinerary_system/product_app/openai_copilot.py`
- `src/itinerary_system/product_app/copilot.py`
- `src/itinerary_system/product_app/copilot_prompt.py`
- `src/itinerary_system/product_app/service.py`
- `src/itinerary_system/product_app/api.py`
- `src/itinerary_system/product_app/runtime.py`
- `src/itinerary_system/product_app/config.py`
- `scripts/run_product_app.py`
- `pyproject.toml`
- the focused provider, contract, prompt, API, and runtime tests under
  `tests/product_app/`

No live provider request was made. The audit used current official OpenAI
documentation for [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[GPT-5.6 model guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.6-terra),
and the [official Python SDK](https://github.com/openai/openai-python), plus the
installed SDK's callable signature and generated JSON Schema.

## Commands and results

| Check | Result |
| --- | --- |
| `python -m pytest tests\product_app\test_openai_copilot.py tests\product_app\test_copilot_contracts.py tests\product_app\test_copilot_prompt.py tests\product_app\test_copilot_api.py -q --basetemp <temporary-directory>` | **PASS** — 57 passed; one unrelated Starlette/httpx deprecation warning |
| `python -m pytest tests\product_app\test_config_runtime.py -q -k "openai or local_environment" --basetemp <temporary-directory>` | **PASS** — 12 passed, 46 deselected |
| Focused `python -m ruff check` over the reviewed production and test files | **PASS** |
| Installed-SDK signature and generated-schema assertion | **PASS** — `openai==2.53.0`; every used `responses.parse` keyword is accepted; strict output schema has `additionalProperties: false` and every property is required at every object level |
| Secret-pattern scan over `src/itinerary_system/product_app`, `tests/product_app`, and `pyproject.toml` | **PASS** — no `sk-`/`sk-proj-` value found; only explicit fake values occur in environment-loading tests |
| Provider/fallback scan | **PASS** — `store=False`, `tools=[]`, no `previous_response_id`, no provider conversation identifier, and no silent deterministic fallback |
| Focused `git diff --check` | **PASS** with only an informational LF-to-CRLF working-copy warning for `pyproject.toml` |

## Contract assessment

### Responses API and SDK compatibility

- The adapter calls `AsyncOpenAI.responses.parse`, not Chat Completions.
- The installed SDK accepts the exact request arguments used: `model`,
  `instructions`, `input`, `text_format`, `reasoning`, `verbosity`,
  `max_output_tokens`, `store`, `tools`, and `timeout`.
- The Pydantic output models use strict validation and forbid extra fields.
  The SDK-generated JSON Schema is strict and satisfies OpenAI's requirement
  that all fields be required; nullable values are represented as unions with
  `null`.
- Refusals are detected separately from parsed structured output and raw
  provider refusal text is not returned to the application.
- Unparsed, malformed, length-limited, content-filtered, and invalid responses
  fail closed without activating the deterministic adapter.

### Provider boundaries

- The default configured model is `gpt-5.6-terra`; the runtime permits an
  explicit nonempty override as planned.
- `store=False` is explicit. Provider-side conversation IDs and
  `previous_response_id` are absent.
- Tools are explicitly empty. No browsing, file input, remote MCP, background
  mode, or external action is enabled.
- The prompt contains only the allow-listed workspace context, current message,
  and bounded recent history. History is capped again by the prompt builder at
  12 messages and 12,000 characters. Output is capped at 1,200 tokens.
- Static instructions identify context/history/user text as untrusted data,
  prohibit invented itinerary facts, and prohibit mutation or claims of
  execution.
- The server revalidates parsed stop IDs, candidate IDs, day IDs, highlights,
  intent/field combinations, permission/state combinations, and output size
  before any result is accepted.

### Retry, timeout, cancellation, and errors

- SDK automatic retries are disabled with `max_retries=0`.
- The application performs at most one retry, only for connection/timeouts,
  429, and 5xx conditions.
- A single outer `asyncio.timeout` bounds the complete request/retry sequence.
- Cancellation is propagated and is not converted into or persisted as a
  fabricated provider result.
- Authentication/permission, rate limit, timeout, provider unavailable,
  invalid structured output, and local interpretation failures map to stable,
  secret-free codes. Raw provider messages and exceptions are not returned.
- A post-provider session revision check prevents a stale response from being
  persisted as the current result.

### Configuration and secret handling

- `OPENAI_API_KEY` is server-side, excluded from configuration repr/public
  health, and never placed in provider turns or responses.
- `.env.local` is ignored by Git. The launcher parser accepts only an
  allow-list of relevant variables, rejects links/oversized or malformed files,
  and does not overwrite an already-set environment variable.
- `openai>=2.45,<3` and `pydantic>=2.7,<3` are declared; the audited installed
  versions satisfy those bounds.
- Health code `openai_configured` truthfully means local adapter construction,
  not verified network or model access, as the W4 plan specifies.

## Findings by severity

### Critical

None.

### High

None.

### Medium

None.

### Low — W4-PSA-001: HTTP-date `Retry-After` is ignored

`_retry_after_seconds` accepts a numeric delay and caps it at two seconds, but
does not parse the other standards-valid `Retry-After` HTTP-date form. This does
not expand the retry count or deadline and therefore is not a security or G4
offline blocker; it can cause the one retry to occur immediately instead of
honoring a short provider delay. A future hardening patch should either parse
HTTP-date safely and cap it at two seconds or explicitly document that only the
delta-seconds form is honored.

### Informational — W4-PSA-002: live provider evidence is intentionally absent

The deterministic transport suite proves request construction, parsing,
validation, retries, cancellation, and error mapping without making a billed
request. It cannot prove that the selected project currently has access to
`gpt-5.6-terra`, that the saved credential authenticates, or that live model
behavior satisfies the prompt-evaluation thresholds.

## Blockers and verdict

There is no blocker to calling the **offline W4 provider implementation
implemented**. The reviewed code satisfies the provider/schema contract without
silent fallback or state mutation.

**G4 remains BLOCKED.** The W4 plan explicitly requires a credential-gated live
OpenAI smoke test and live prompt-evaluation evidence before G4 can be verified.
Those billed calls require the user's explicit consent and must remain bounded;
this audit did not make them. G4 also remains dependent on the other required
independent audit verdicts. W5 must not be promoted from this offline provider
audit alone.
