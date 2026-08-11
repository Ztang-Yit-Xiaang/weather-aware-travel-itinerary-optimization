# W4 Phase, Gate, and Status Truth Audit

**Audit date:** 2026-08-06  
**Audit role:** `PHASE_GATE_STATUS_AUDITOR`  
**Independence:** This auditor did not implement the W4 provider, prompt,
conversation, frontend, persistence, map-context, or test changes. The auditor
made no production-code or status-document changes and made no OpenAI request.
The only repository edit made by this audit is this report.  
**Offline W4 status verdict:** **IMPLEMENTED**  
**G4 gate verdict:** **BLOCKED**  
**W5 readiness verdict:** **PLANNED / NOT READY**

## Scope

This audit treated planning and status documents as claims rather than
implementation evidence. It independently checked:

- the W4 transition rules and G4 acceptance criteria;
- all seven W4 specialist audit reports and their post-fix verdicts;
- current deterministic tests, static checks, and browser evidence;
- the fixed 72-case offline and fixed 24-case live-evaluation contracts;
- the W5 fail-closed boundary;
- CP-010 status; and
- the E3.1, E3.3, E4, E3.UX5, and E5 non-propagation boundary.

No live OpenAI request was authorized or executed during this audit. A saved or
configured key is not live-provider evidence and cannot promote G4.

## Status rules applied

The W4 phase plan defines these relevant transitions:

| Observed condition | W4 | G4 | W5 |
| --- | --- | --- | --- |
| Deterministic implementation passes but live evidence is absent | `implemented` | `blocked` | `planned` |
| Live smoke, fixed evaluation, all audits, and durable evidence pass | `verified` | `verified` | `ready` |
| Any blocking provider/privacy/audit defect remains | `implemented` or `in-progress` | `blocked` | `planned` |

The current worktree satisfies the first row, not the second. All seven
specialist audits now pass their offline scopes after independent re-audit. The
last prompt-quality blockers were closed by requiring the exact evidence scope
and replacing both provider-authored free-text fields with application-owned
copy before validation, persistence, and API exposure. The required live model
evidence has not been produced.

## Commands and evidence

### Planning, status, and boundary inspection

```powershell
rg -n "CP-010|W4|G4|W5|E3\.1|E3\.3|E3\.UX5|E4|E5" `
  docs\planning docs\current README.md docs\README.md
rg -n "acceptance_not_enabled_until_w5|/accept|/keep-original" `
  src\itinerary_system\product_app tests\product_app
```

Observed result:

- the active documents consistently keep CP-010 `in-progress`;
- E3.1, E3.3, and E4 remain `blocked`;
- E3.UX5 and E5 remain `deferred`;
- no product status is used as publication or participant-study evidence; and
- both Accept and Keep original still fail closed with
  `acceptance_not_enabled_until_w5`.

The active documents still describe W4 as `in-progress` and G4 as `planned`.
Those labels are now stale relative to the completed offline implementation and
must be changed to W4 `implemented` and G4 `blocked`. W5 must remain `planned`.

### Current focused deterministic verification

```powershell
python -m pytest `
  tests\product_app\test_copilot_prompt.py `
  tests\product_app\test_openai_copilot.py `
  tests\product_app\test_copilot_contracts.py `
  tests\product_app\test_copilot_api.py `
  -q --basetemp <unique-temporary-directory>
```

Result: **66 passed**, one pre-existing Starlette/TestClient deprecation
warning, no failures.

```powershell
python -m ruff check `
  src\itinerary_system\product_app `
  scripts\evaluate_copilot_prompt.py `
  tests\product_app\test_copilot_prompt.py `
  tests\product_app\test_openai_copilot.py `
  tests\product_app\test_copilot_contracts.py `
  tests\product_app\test_copilot_api.py
```

Result: **PASS**.

```powershell
git diff --check -- `
  src\itinerary_system\product_app `
  scripts\evaluate_copilot_prompt.py `
  tests\product_app `
  docs\audits\w4_prompt_quality_injection_audit.md
```

Result: **PASS**.

An additional current focused run covering prompt, provider, contract, API,
privacy, and status assertions passed **71 tests** before the last canonical-copy
repair. The 66-test run above was executed after that repair and is the relevant
post-fix evidence.

### Full-suite evidence qualification

A full repository run completed with **615 passed** and one existing warning.
The last canonical-copy production and regression-test edits occurred after
that run had already collected its tests. A subsequent collection found **616
tests**. The 615-test result is therefore useful regression evidence but is not
claimed as an atomic post-fix full-suite pass. The post-fix 66-test suite and
independent prompt re-audit cover the changed boundary. Before G4 verification,
the orchestrator should run one stable full project-check cycle after all W4
files stop changing and record it in the G4 evidence manifest.

### Specialist audit matrix

| Independent audit | Offline verdict | Remaining gate effect |
| --- | --- | --- |
| Provider/schema | Pass | Live provider evidence still required |
| Privacy/security | Pass after re-audit | No offline blocker |
| Transcript retention | Pass after re-audit | Non-blocking hardening suggestions only |
| Map/context synchronization | Pass after re-audit | No offline blocker |
| Accessibility/content | Pass after re-audit | W6 real-device work remains later scope |
| User-journey black box | Pass | Deterministic journey only; live journey absent |
| Prompt quality/injection | Pass after three audit passes | Fixed live evaluation still required |

The prompt identity recorded by the implementation and prompt audit is:

- prompt version: `copilot-prompt-v1`;
- prompt SHA-256:
  `4cb96797b5b93cc0530d631b4aa293ef4e8fdee7bb3fdf3fa702f0616ad0713f`.

The evaluator now requires exactly all 72 cases for `offline-fake` or the fixed
24-case stratified set for a live model label, rejects duplicate/incomplete
sets, and emits explicit threshold results. Application-owned canonical copy
prevents provider prose from reaching the persisted/API interpretation.

## Findings

### W4-STATUS-001 - Active W4/G4 labels lag the evidence

**Severity:** Medium status-integrity defect  
**Status:** Open until the orchestrator updates the authority documents.

The current authority documents still say W4 `in-progress` and G4 `planned`.
That was truthful during implementation, but the seven specialist offline
audits and post-fix focused checks now support W4 `implemented`. Because live
evidence is missing, G4 is no longer merely future/planned; it is actively
`blocked` on an explicit gate condition.

**Required correction:** change only the product-track status text to W4
`implemented`, G4 `blocked`, and W5 `planned`/not ready. Do not promote W4 or G4
to `verified`.

### W4-STATUS-002 - Live provider and quality evidence is absent

**Severity:** Blocking G4 verification  
**Status:** Open.

No authorized live request has established credential/model access, one real
strict Responses result, or browser/provider lifecycle behavior. The fixed
24-case live set has not measured state/intent/reference quality, clarification
behavior, external-action safety, injection behavior, token use, or latency.
The planned low-versus-medium reasoning comparison has also not been recorded.

This is the only remaining substantive G4 blocker identified by the eight W4
audits. It requires explicit user authorization because it sends bounded trip
context to OpenAI and may incur cost. There is no silent fixture substitute.

### W4-STATUS-003 - Final G4 evidence package cannot yet exist

**Severity:** Procedural G4 blocker  
**Status:** Expected consequence of W4-STATUS-002.

`w4_g4_verification_report.md` and its machine-readable evidence manifest are
not present. They must be created only after a stable final regression/project
check and the authorized live evidence exist. Their absence is not a separate
implementation defect and must not be filled with planned or synthetic claims.

### W4-STATUS-004 - W5 remains correctly closed

**Severity:** Boundary confirmation  
**Status:** Pass.

W5 has not been implemented or verified. Accept and Keep original remain
disabled/fail closed in both API and workspace code, and regression tests assert
that the workspace does not change. W5 must remain `planned` and cannot become
`ready` until G4 is verified.

### W4-STATUS-005 - No research-status propagation occurred

**Severity:** Boundary confirmation  
**Status:** Pass.

The product work did not alter the research dependency chain:

- CP-010 remains `in-progress` until G8;
- E3.1 remains `blocked` on D1;
- E3.3 remains `blocked` on E3.1;
- E4 remains `blocked` on E3.3;
- E3.UX5 remains `deferred` and E5-dependent; and
- E5 remains `deferred` until E4 and its independent ethics/research gates.

No W4 artifact, prompt result, browser journey, or product audit is E3, E4,
E3.UX5, E5, publication, or participant-study evidence.

## Exact recommended status text

Use this text consistently in the W4 plan, planning index, current execution
plan, current problem manifest, gate map, implementation plan, and documentation
indexes:

> W4 is `implemented`. Seven independent specialist audits and the independent
> phase/gate/status audit found no remaining offline W4 implementation blocker
> after re-audited provider, privacy, retention, map-context, accessibility,
> browser, and prompt-evaluation fixes. G4 is `blocked` pending explicit user
> authorization and successful completion of the bounded live OpenAI smoke and
> fixed 24-case evaluation, including the planned low-versus-medium reasoning
> comparison, followed by a stable final project-check run and durable G4
> report/evidence manifest. W5 remains `planned` and not ready. CP-010 remains
> `in-progress`. E3.1, E3.3, and E4 remain `blocked`; E3.UX5 and E5 remain
> `deferred`.

Recommended status table row:

| Phase / gate | Exact status | Next action |
| --- | --- | --- |
| W4 / G4 | W4 `implemented`; G4 `blocked` | Obtain explicit user authorization; run the bounded live smoke and fixed 24-case evaluation; compare low versus medium; run stable final checks; write the sanitized G4 evidence package |
| W5 / G5 | `planned`; not ready | Do not start until G4 is `verified` |

## Verdict

The W4 implementation is complete at the deterministic/offline boundary and
may truthfully move from `in-progress` to `implemented`. G4 must move from
`planned` to `blocked`, not to `verified`. W5 remains `planned` and not ready.
No deterministic specialist blocker remains after the final prompt re-audit;
the remaining substantive blocker is the explicitly authorized live OpenAI
smoke/fixed-24 quality evaluation and its planned reasoning comparison. Final
stable checks and the sanitized report/manifest are required evidence-producing
steps, not substitutes for the live gate.

