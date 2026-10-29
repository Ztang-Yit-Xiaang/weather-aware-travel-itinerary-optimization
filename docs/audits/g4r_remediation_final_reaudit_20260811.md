# G4R Remediation Final Independent Re-audit

Date: 2026-08-11 (America/Chicago)  
Verdict: **PASS**  
Scope: W4R interaction/stability gate only. This does not pass G4 or open W5.

## Independent slice verdicts

| Slice | Verdict | Evidence |
| --- | --- | --- |
| Map lifecycle / black-box | PASS | Positive tile-source readiness, fresh/reload/module/Copilot/Evidence behavior, bounded desktop/mobile maps, and fresh post-resize readiness timestamps |
| Product / accessibility / content | PASS | Five-item navigation, selected Compare preview action, named text-route legs, user-facing Copilot context, responsive state, keyboard/focus basics |
| Integrity / security / status | PASS | Fail-closed readiness adversaries, exact artifact/state boundaries, Host/Origin/token/revision/no-store controls, W5 disabled |

The initial failed audit remains preserved in `g4r_final_state_independent_audit_20260811.md`. Its blockers were remediated and then independently re-audited under the one-blocker rule; no unresolved G4R blocker remains.

## Final evidence

- Executable map-readiness adversaries passed, including missing APIs, absent/partial sources, `areTilesLoaded() === false`, and explicit readiness invalidation.
- Focused remediation matrix: **64 passed**.
- Independent integrity/security matrices: **130 passed**.
- Independent product/accessibility/Compare matrix: **18 passed**.
- Independent final resize/map contract matrix: **14 passed**.
- JavaScript syntax, Ruff, coherent-fixture build check, and scoped diff checks passed.
- The clean live run records 12 scenarios, zero console errors, and zero network failures.

Full-size evidence and hashes are recorded in [`live_evidence.json`](../../results/product_app/g4r_remediation_20260811/live_evidence.json). Required viewport captures exist at 1440×900, 1280×800, 1024×768, and 390×844.

The resize acceptance path now invalidates the previous ready flag, timestamp, and coverage on a real container size change; waits for a fresh MapLibre `idle`; reruns positive tile-source and canvas coverage checks; and only then publishes a new timestamp. Final resize evidence:

- Evidence-close ready: `2026-08-12T04:21:40.645Z`.
- 1024×768 ready: `2026-08-12T04:21:41.394Z`; capture: `2026-08-12T04:21:41.415941Z`; coverage: `1.0`.
- Returned 1440×900 ready: `2026-08-12T04:21:42.025Z`; capture: `2026-08-12T04:21:42.046766Z`; coverage: `1.0`.

## Gate disposition

- W4R: **implemented**.
- G4R: **verified / passed**.
- G4: **blocked** pending its separately authorized billed-provider smoke, fixed-24 evaluation, prescribed reasoning comparison, and provider-specific independent audits.
- W5: **closed / planned** until G4 passes; Accept and Keep-original remain disabled.
- Research statuses are unchanged.

## Recorded limitations

- The Codex in-app Browser runtime failed before discovery because of the Windows sandbox helper. Independent live product checks used the installed local Chromium, while the map slice separately corroborated source, executable adversaries, timestamps, hashes, and rendered pixels.
- Screen-reader, axe, forced-colors, and physical-device checks remain later accessibility risk, not an observed G4R blocker.
- A broader test command still has five coherent-fixture migration-test failures in evaluated-edit/POI API fixtures. They are tracked regression debt and were not hidden by weakening product contracts.
