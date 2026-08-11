# Itinerary Repair Copilot User-Journey Black-Box Audit

**Audit date:** 2026-08-03

**Audit role:** Independent read-only black-box journey specialist

**Scope:** Live launch identity, desktop/mobile workflows, state persistence,
decision effects, reload/restart behavior, recovery, console, and accessibility
signals.

**Gate verdict:** **G7 FAIL; G8 FAIL.**

## Method and isolation

The auditor launched a disposable product server at `127.0.0.1:8138` with a
disposable state root, manipulated the live browser at desktop/mobile viewports,
reloaded the page, restarted the server, and inspected generated decision/pointer
state. No repository files were modified and disposable processes/state were
cleaned up. HTTP success alone was not considered journey success.

## Findings

| ID | Severity | Gates | Observed evidence | Required closure |
|---|---|---|---|---|
| JRN-001 | Blocker | G8 | The principal map remains a schematic rather than a geographic map. | Complete geographic stop/route/edit journeys with textual fallback. |
| JRN-002 | Blocker | G8 | Copilot is fixture-only and its conversation does not persist across a true reload/restart transcript lifecycle. | Verify explicit adapter identity, persisted transcript, TTL, and delete controls. |
| JRN-003 | High | G8 | Draft interactions are partial and session-memory-only. | Persist, restore, undo, compile, evaluate, and invalidate drafts end to end. |
| JRN-004 | Critical | G8 | `Inspect option` on Keep original records a durable keep decision. | Make inspection side-effect free and require explicit confirmed decisions. |
| JRN-005 | Blocker | G8 | Compare does not prove distinct immutable plan alternatives. | Show baseline plus two distinct evaluated children with unique IDs/hashes/geometries. |
| JRN-006 | Blocker | G8 | Acceptance survives only as current pointer/decision state, not as verified `PlanRepository` authority. | Verify repository child, ledger, CAS pointer, reload, and restart as one transaction. |
| JRN-007 | High | G8 | Evidence can remain internally confusing; proposal identifiers remain visible after Keep original. | Clear obsolete proposal context and lead with decision-useful evidence. |
| JRN-008 | Blocker | G8 | Required mobile Overview, Day Details, geographic Map/bottom sheet, Copilot, and Evidence journeys are missing. | Pass complete 430/390/360 px route journeys. |
| JRN-009 | Blocker | G7 | Launch identity is correct, but health readiness is superficial and does not prove default workspace/state/map readiness. | Add deep component readiness and clean-launch recovery tests. |
| JRN-010 | High | G7/G8 | The exercised console was clean, but port collision, unavailable Atlas/provider/backend, offline, and corrupt-state recovery remain unproven. | Record deterministic recovery journeys and expected user-facing states. |
| JRN-011 | High | G8 | Current semantics obstruct reliable keyboard/screen-reader journeys. | Close the accessibility audit and rerun the journey without pointer-only input. |

## Required 28-step closure journey

The replacement black-box record must execute and preserve evidence for all of
the following in one clean-user protocol:

1. Start from a clean repository-root Windows PowerShell process.
2. Start the documented local Atlas dependency and verify range-request health.
3. Run only `OPEN_ITINERARY_COPILOT.cmd`.
4. Verify product identity, fixed port, validated default run, state-store
   writability, and component readiness.
5. Verify `/` redirects to `/app` and no legacy framing/fallback appears.
6. Verify the geographic basemap, original route, repaired route, and stops load.
7. Select Day 3 and confirm itinerary, map, and server context agree.
8. Select a geographic stop and segment and confirm accessible selected state.
9. Open desktop Copilot and verify exact context chips.
10. Submit a live-provider request when credentials are approved, or an explicitly
    labeled deterministic request in the deterministic protocol.
11. Verify interpreting, clarification/permission, and proposal lifecycle states.
12. Close and reopen Copilot and verify focus, context, and transcript continuity.
13. Enter Map edit and append a controlled Move Day operation.
14. Undo it and verify the revision and visible route/draft state.
15. Recreate the Move Day operation and add route feedback.
16. Preview and wait for real repair execution plus independent evaluation.
17. Verify ineligible states cannot be accepted and stale revisions return 409.
18. Open Compare and verify baseline plus two distinct plan IDs/hashes/geometries.
19. Inspect Keep original and verify that inspection writes no decision.
20. Open Evidence and verify eligibility, checks, routes, freshness, lineage,
    methods, metrics, nulls, and technical hashes.
21. Accept an eligible child in a disposable workspace and verify repository
    content, ledger entry, pointer CAS, and parent immutability.
22. Reload, restart the application, and verify the accepted child is authoritative.
23. In a second disposable workspace, explicitly confirm Keep original and verify
    obsolete draft/proposal context is cleared.
24. Delete one conversation, then delete all transcripts, and inspect storage.
25. Exercise Overview, Day Details, Map bottom sheet, Copilot, Compare, Evidence,
    and bottom navigation at 430, 390, and 360 px.
26. Install/launch the PWA; disconnect internet while FastAPI/Atlas remain local;
    then separately stop the backend and verify truthful unavailable behavior.
27. Test port collision, missing provider, Atlas failure, corrupt state, console,
    network errors, overflow, keyboard, screen reader, and virtual keyboard.
28. Shut down cleanly and preserve commands, versions, viewport evidence, state
    hashes, screenshots, console/network logs, auditor identity, and verdict.

## Gate interpretation

G7 remains failed until clean Windows launch, deep readiness, recovery, and clean
shutdown are recorded. G8 remains failed until every required product journey,
full regression/audit pool, and user visual approval pass. This audit does not
change any E3/E4/E5 research status.
