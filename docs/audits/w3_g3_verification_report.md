# W3 Persistent Draft / G3 Verification Report

**Generated:** 2026-08-05 (America/Chicago)  
**Track:** CP-010 corrective local product  
**Default run:** `california_coast_product_demo_v1`  
**Historical v1 gate result at generation time:** W3/G3 `verified`; W4 `ready`  
**Current authority:** see `docs/planning/current_execution_plan.md`

## Scope and non-claims

W3 adds authenticated, restart-safe local sessions; ordered typed drafts; revision
compare-and-swap; undo; compilation of all seven canonical operations into the
existing deterministic repair pipeline; independent evaluation; exact evaluated-
child geography and evidence; and truthful eligible/ineligible browser states.
The accepted parent, frozen run artifacts, legacy decisions, and W5 pointer remain
unchanged.

This gate does not implement OpenAI transport or transcripts (W4), repository-
backed acceptance (W5), PWA/mobile completion (W6), or replacement verification
(W7-W8). It does not advance E3.1, E3.3, E4, E3.UX5, or E5.

## Implemented contracts

- `workspace-context-v1` sessions persist beneath `.product_app_state/sessions/`.
- The raw mutation token is returned once; only a salted SHA-256 verifier is stored.
- Disk-backed locking and expected-revision CAS protect selection, draft, undo,
  proposal, and permission changes across processes.
- `keep_stop`, `lock_stop`, `mark_flexible`, `move_day`, `route_feedback`,
  `replace_nearby`, and `add_candidate` use strict target/parameter validation.
- Semantic request identity excludes session-local operation IDs and sources, so
  identical ordered intent yields identical request, child, and content hashes.
- Preview calls the existing deterministic repair executor and independent
  evaluator. Ineligible results remain inspectable and cannot be accepted.
- Eligible proposals bind the parent, child, diff, certificate, route validation,
  draft hash, expected revision, exact GeoJSON, and session-local Evidence.
- W5 Accept and Keep original endpoints remain fail-closed with
  `acceptance_not_enabled_until_w5`.

## User-like verification

A clean controlled browser opened `http://127.0.0.1:8127/app` against the restarted
product service and ready local MapLibre/PMTiles runtime.

1. Selected Day 3 / Griffith Observatory.
2. Added `keep_stop`; preview truthfully returned `ineligible` with
   `no_feasible_evaluated_child` and `Unavailable` repair metrics.
3. Undid the operation, added `route_feedback`, and previewed again.
4. The independent evaluator returned eligible child `plan_c12aa2c626b69fc3`,
   certificate `cert_2224e335ad68a6fc`, and 16/16 road-validated legs.
5. Map and Evidence used that exact evaluated child; they did not substitute a
   registered alternative. Preview evidence was correctly session-local and not
   downloadable before W5.
6. Reload restored selection, draft, evaluated proposal, metrics, geography, and
   evidence through authenticated session restoration.
7. Accept and Keep original remained disabled. Console warning/error count was zero.
8. At 390 x 844, document client and scroll widths were both 390 pixels, all mobile
   navigation labels were visible, and no horizontal document overflow occurred.

The independent user-journey auditor separately verified add/undo, ineligible
`lock_stop`, eligible `replace_nearby`, stale-revision 409 behavior, authenticated
restoration, W5 fail-closed responses, and unchanged run-artifact digests.

## Automated verification

- Focused product plus `PlanRepository`: **189 passed**, one existing
  Starlette/httpx TestClient deprecation warning.
- Repository-wide check: Ruff passed; context snapshot **5 passed**; full pytest
  **500 passed**, with the same warning.
- JavaScript syntax checks and `git diff --check` passed.
- The Windows long-path regression passed after shortening same-directory atomic
  temporary filenames; the prior 262-character session-write failure is closed.
- Security audit rehashed 74 protected run files after a complete preview journey:
  zero changed and zero added.
- Artifact audit rehashed the 27-file California demo package: zero changes.

## Independent audit verdicts

| Audit category | Independent agent | Result |
|---|---|---|
| Draft/State and Security/Integrity | `w3_security_auditor` | PASS |
| Solver/Evaluator and Artifact Integrity | `w3_artifact_integrity_auditor` | PASS |
| Web/Accessibility | `w3_ui_accessibility_auditor` | PASS |
| Content/Truth | `w3_content_truth_auditor` | PASS |
| User Journey / Black Box | `w3_user_journey_auditor` | PASS |
| Phase/Gate Status | `w3_phase_gate_auditor` | PASS |

Blocking findings were not averaged away. Auditors found and verified fixes for
candidate-selection forgery, ineligible-state masking, registered-metric leakage,
session-dependent preview hashes, evaluated-child route/evidence substitution,
Fixture Copilot state mixing, inaccurate W3/W5 copy, dialog naming, and 44-pixel
control sizing.

## Gate decision

W3/G3 are verified and W4 is ready for its own implementation-ready phase plan.
CP-010 remains `in-progress`; W5-W8 remain planned; Atlas remains a deferred backup;
and all research statuses remain unchanged.
