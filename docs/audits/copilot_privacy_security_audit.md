# Itinerary Repair Copilot Copilot Privacy and Security Audit

**Audit date:** 2026-08-03

**Audit role:** Independent read-only Copilot privacy/security specialist

**Scope:** FastAPI boundary, fixture adapter, session mutations, evidence export,
planned OpenAI/Atlas/PWA boundaries, and local transcript requirements.

**Gate verdict:** **BLOCKING — security portions of G1, G4, G6, and G8 fail.**

## Method

The auditor traced `api.py`, `copilot.py`, `models.py`, `registry.py`,
`workspace.py`, launcher behavior, CSP, client rendering, and focused tests.
Threats were evaluated at trust boundaries; future provider behavior was not
inferred from the current fixture-only implementation.

## Existing controls worth preserving

- Launcher binds to loopback and the current browser app makes no provider call.
- Registry paths are constrained to run-relative locations.
- Mutations use a per-session token, constant-time comparison, and revisions.
- CSP, `nosniff`, HTML escaping, and `textContent` reduce common injection risk.

## Findings

| ID | Severity | Gates | Evidence | Required closure |
|---|---|---|---|---|
| SEC-001 | Critical | G1/G8 | No complete Host allow-list and exact Origin policy protects all mutations. | Reject unexpected Host/Origin values and test hostile cases. |
| SEC-002 | High | G1/G6 | Sensitive API, transcript, and decision responses lack a consistent `Cache-Control: no-store` contract. | Apply and test no-store headers; deny service-worker caching. |
| SEC-003 | High | G1/G4 | Message, operation, session, transcript, and evidence resource budgets are incomplete or bypassable. | Enforce schema and aggregate limits with deterministic 4xx responses. |
| SEC-004 | High | G1/G5 | Alternatives/session-derived reads can be obtained without the same session authentication boundary as mutations. | Authenticate session-private reads and avoid token exposure. |
| SEC-005 | Critical | G4 | No OpenAI transport boundary or strict typed response schema exists; current interpretations contain loose dictionaries. | Add server-only provider config, strict schema validation, timeout/retry/cancel rules, and no silent fallback. |
| SEC-006 | Critical | G4/G8 | Transcript persistence, 30-day TTL, deletion APIs, and evidence exclusion are absent. | Implement retention/deletion and privacy tests before recording conversations. |
| SEC-007 | High | G1/G6 | CSP/connect/worker policy for loopback Atlas and the service worker is not designed or tested. | Pin exact loopback origins and deny broad worker/network access. |
| SEC-008 | High | G5/G8 | Evidence export lacks a complete authentication, size-budget, and cache policy. | Bind export to session/run authority and manifest allow-lists; set no-store. |
| SEC-009 | High | G4/G8 | Log redaction for keys, paths, provider payloads, and full transcripts is unproven. | Add structured redaction and captured-log tests. |
| SEC-010 | High | G3/G4 | Permission records are untyped and not strongly bound to proposal, scope, revision, and expiry. | Use typed, append-only, scoped permission decisions. |
| SEC-011 | High | G1 | Application factories/manual server paths do not independently enforce the launcher host restriction. | Enforce loopback at every supported entrypoint. |
| SEC-012 | Medium | G4 | Fixture/live provider identity is hard-coded rather than derived from validated configuration. | Return explicit adapter provenance in health, turns, and UI. |

## Closure evidence

Each affected gate needs adversarial API tests, captured headers/logs, and an
independent security rerun. A live key is not required for deterministic G4
tests, but live-provider verification cannot pass without approved credentials.
