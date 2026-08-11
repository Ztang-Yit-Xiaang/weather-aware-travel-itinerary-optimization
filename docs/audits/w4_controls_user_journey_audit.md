# W4 Copilot, Map Edit, and Repair-Control User-Journey Audit

**Original audit:** 2026-08-07

**Independent re-audit:** 2026-08-07

**Audit role:** Independent controls/user-journey auditor
**Re-audit verdict:** **PASS for the requested deterministic Copilot and Map-edit journeys.**

## Independence and test boundary

The auditor did not implement the reviewed frontend, API, draft, Copilot, map,
or launcher code. No production or status file was changed. Browser work used a
fresh application process on a free loopback port, a unique temporary state
root, a fresh headless Chromium context, and the deterministic Copilot adapter.
The audit terminated only its own temporary application processes.

The local MapLibre service is configured to permit browser requests from
`http://127.0.0.1:8127`. A disposable application origin therefore cannot render
the live map without changing or restarting that shared service. The re-audit
did not change it. Repair-preview selection was exercised through the Day 7
workspace state; actual marker-click dispatch remains an environment-limited
coverage gap, not an observed product failure.

## Browser evidence

### Copilot: send, add to draft, preview

Journey:

1. Select Day 3.
2. Open Copilot.
3. Submit `Mark this stop flexible`.
4. Add the typed proposal to the draft.
5. Preview the repair from the same proposal card.

Observed state transitions:

| Checkpoint | Observed result |
|---|---|
| Before Add | `Add to draft` enabled; Preview disabled |
| After Add | Label changed to `Added to draft`; Add disabled; Preview enabled; workspace showed `1 draft change` |
| After Preview | Repair status became `Evaluated eligible proposal` |
| Diagnostics | No console errors and no HTTP failure responses |

This closes the earlier stale Copilot-action defect. A successful mutation now
rerenders the proposal actions, prevents duplicate Add, and exposes Preview.

### Map edit: add, undo, re-add, preview without reopening

Journey:

1. Select an accepted-plan stop.
2. Open Map edit.
3. Add `Mark flexible`.
4. Use Undo while the same dialog remains open.
5. Re-add the operation.
6. Preview the repair.

Observed state transitions:

| Checkpoint | Dialog open | Undo | Preview | Workspace |
|---|---:|---:|---:|---|
| After Add | Yes | Enabled | Enabled | `1 draft change` |
| After Undo | Yes | Disabled | Disabled | `0 draft changes` |
| After re-add and Preview | Closed by completed Preview | N/A | N/A | `Evaluated eligible proposal` |

No dialog close/reopen workaround was required. No console errors or failed HTTP
responses occurred. This closes the earlier stale Map-edit-control defect.

### Candidate compatibility and preview-stop safeguards

The browser matrix inspected all seven displayed days:

- Accepted stops without a registered replacement showed Replace disabled.
- The candidate selector may still list registered candidates because it is
  shared with the independently allowed Add-candidate operation.
- Add candidate remaining enabled is intentional and does not authorize an
  incompatible replacement.
- The replacement path filters with
  `candidate.replaces_stop_id === selected.id` before enabling Replace.
- The API/compiler rejects a candidate paired with the wrong parent target.

For the displayed Day 7 repair-preview stop, the dialog stated:

> The selected marker belongs to a repair preview and cannot be edited in
> place. Select an original accepted-plan stop, or add a registered candidate.

Keep, Move Day, Lock, Mark flexible, and Replace were disabled. Add candidate
remained enabled as a separate registered-candidate operation. This is truthful:
the preview stop itself cannot be mutated, while a new artifact-backed candidate
may still be drafted.

## Automated regression evidence

Command:

```powershell
python -m pytest tests\product_app\test_w3_preview.py `
  tests\product_app\test_geographic_frontend_contract.py `
  tests\product_app\browser\test_w4_copilot_journey.py -q
```

Result: **47 passed in 74.67 seconds**.

The focused set covers typed-preview compilation and rejection cases,
geographic frontend contracts, and deterministic Copilot browser behavior.

## Earlier findings: re-audit disposition

| Finding | Re-audit disposition |
|---|---|
| W4-UJ-002: Map-edit controls stale after draft mutation | **Closed** by the same-dialog Add/Undo/re-add/Preview journey |
| W4-UJ-003: Copilot proposal actions stale after Add | **Closed** by the Add-label/disable and Preview-enable journey |
| W4-UJ-004: selection race with Map edit | **Closed for the exercised path**; Map edit waits for the pending selection request |
| W4-UJ-005: raw mutation errors | **Improved**; stable draft errors have user-facing copy, and no raw error surfaced in either journey |
| W4-UJ-001: reuse of an older API-incompatible process | Not part of the disposable-current-server re-audit; retain as a launcher compatibility concern until its separate black-box test passes |
| W4-UJ-006: workflow-header progress | Outside this focused rerun; remains presentation debt |
| W4-UJ-007: raw draft identifiers in Repairs | Outside this focused rerun; remains low-priority content debt |

## Final verdict and remaining limitation

No blocking defect was reproduced in either requested deterministic workflow.
Draft state, action availability, Undo, and Preview now remain coherent without
manual reload or dialog reopening. Candidate replacement remains fail-closed,
and repair-preview stops are explicitly explained and protected from parent-stop
edits.

The only unresolved audit limitation is a live click on the repair-preview map
marker from a disposable origin. Completing that exact check requires a
disposable map-service origin or a controlled audit on the configured 8127
origin. It does not overturn the current PASS because the same preview stop,
same dialog logic, source contract, and server mismatch protections were
independently exercised.
