# G4R Final-State Independent Audit

> Historical failed audit. Its findings were remediated and the final independent re-audit passed; see `g4r_remediation_final_reaudit_20260811.md`.

Date: 2026-08-11

Verdict: **FAIL — remediation and a fresh independent re-audit are required.**

No auditor edited implementation, product state, gate/status files, or acceptance state. No auditor called OpenAI. Accept and Keep-original remained disabled.

## Audit slices

### Live map and browser black-box — FAIL

- The required in-app Browser could not start because the Windows sandbox helper failed during browser discovery.
- `renderedCanvasCoverage()` measures canvas/container rectangle overlap, not rendered tile pixels. A correctly sized gray canvas can therefore report full coverage.
- Map readiness fails open when `areTilesLoaded()` is unavailable.
- The stored 390x844 evidence does not demonstrate a mobile map workspace.
- Evidence-open/close captures and capture-after-ready telemetry are absent.

### Product, accessibility, content, and user journeys — FAIL

Fresh live Chromium checks passed coherent fixture ordering, Compare default selection and synchronized metrics, selection persistence, bounded map heights, no horizontal overflow, and basic focus/target behavior. Four blockers remain:

1. Global navigation still exposes `Settings` in addition to the required five destinations.
2. Compare shows `Choose option (not enabled)`, so the required Choose journey is incomplete.
3. Text route legs expose `Stop unavailable` endpoints rather than clear place names.
4. Copilot exposes a raw plan ID outside Advanced Evidence.

### Route, artifact, state, security, and phase gate — FAIL

Positive independent evidence:

- Fixture/route/map tests: 48 passed.
- API/security/W5 tests: 29 passed.
- Compare/state/privacy tests: 28 passed.
- Total focused tests: 105 passed.
- Coherent product builder, Ruff, Node syntax, and scoped diff checks passed.
- Manifest/registry hash matched; 16/16 declared artifacts were present with no mismatch.
- Route matrix contained 19/19 road-validated cached-OSRM cells with no fallback or invalid geometry; each plan declared 15 required legs.
- Host/Origin, request size, schema, token/revision, no-store, sanitized-error, and W5 fail-closed boundaries passed inspection.

Blocking counterexample:

```js
const tilesLoaded = typeof nextMap.areTilesLoaded !== "function"
  || nextMap.areTilesLoaded();
```

If the tile-state API is missing, readiness treats tiles as loaded. The current test checks source-string presence but does not execute this fail-open branch.

## Required remediation before re-audit

1. Make missing or false tile-state evidence fail closed.
2. Replace rectangle-overlap coverage with an executable rendered-tile/pixel readiness invariant and adversarial tests.
3. Produce a real 390x844 mobile map journey and Evidence open/close captures with post-ready telemetry.
4. Remove the extra global Settings destination from the consolidated navigation.
5. Complete or truthfully rescope the Compare Choose action without weakening the W5 acceptance boundary.
6. Resolve route endpoint labels in the text alternative.
7. Remove raw plan identity from ordinary Copilot UI.
8. Repeat the three independent final-state audit slices.

## Gate decision

- W4R: remains **implemented**, with G4R remediation now required.
- G4R: **failed / blocked**.
- G4: remains **blocked**; no billed provider evaluation may start.
- W5: remains **closed/planned**.
- CP-010: remains **in progress**.
- Research-track statuses remain unchanged.
