# G4 External-Provider Preflight

Date: 2026-08-11

Verdict: **BLOCKED; no external provider request was made.**

## Preconditions

- W4 offline provider implementation: implemented.
- W4R product interaction/stability implementation: implemented.
- G4R independent final-state audit: failed; remediation and re-audit required.
- `OPENAI_API_KEY`: absent.
- `PRODUCT_COPILOT_PROVIDER`: unset.
- `PRODUCT_COPILOT_MODEL`: unset.
- Explicit authorization for a new bounded billed run: not recorded for this run.

## Offline verification

The following provider-boundary, prompt, privacy, and runtime suites were run without network access or provider calls:

```text
tests/product_app/test_copilot_contracts.py
tests/product_app/test_copilot_prompt.py
tests/product_app/test_openai_copilot.py
tests/product_app/test_copilot_privacy.py
tests/product_app/test_config_runtime.py
```

Result: **133 passed**, with one known Starlette TestClient/httpx deprecation warning.

The prompt evaluator CLI was also inspected and remains deliberately offline-only: it aggregates sanitized result records and does not make provider calls.

## Decision

G4 cannot start yet. The next required product action is remediation of `g4r_final_state_independent_audit_20260811.md` followed by a fresh independent re-audit. If G4R passes, a separate bounded run may begin only after explicit authorization and local secret/model configuration. That later run must include the documented one-request smoke, fixed 24-case evaluation, low-versus-medium reasoning comparison, sanitized evidence, and independent provider audits.

W5 remains closed; Accept and Keep-original remain disabled.
