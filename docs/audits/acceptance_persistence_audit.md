# Itinerary Repair Copilot Acceptance and Persistence Audit

**Audit date:** 2026-08-03

**Audit role:** Independent read-only persistence/transaction specialist

**Scope:** Sessions, drafts, proposals, permissions, decision ledger, pointer,
`PlanRepository`, restart behavior, and concurrent acceptance.

**Gate verdict:** **BLOCKING — G3, G5, G7, and G8 fail.**

## Method

The auditor traced `workspace.py`, service/API acceptance paths, the existing
`PlanRepository` implementation, client post-accept state handling, and focused
tests. Sequential, concurrent, crash, and restart semantics were considered
separately.

## Existing controls worth preserving

- Mutation tokens and sequential stale-revision rejection exist.
- Accept/keep paths use an in-process `RLock`.
- Decision files use exclusive creation, and pointer replacement uses
  `os.replace`.
- A focused test checks that the source parent bytes are unchanged.

## Findings

| ID | Severity | Gates | Evidence | Required closure |
|---|---|---|---|---|
| ACC-C01 | Critical | G5 | Accept bypasses `PlanRepository` and does not reverify child hash, eligibility, certificate freshness, diff, and lineage in one transaction. | Implement the approved repository/ledger/pointer transaction. |
| ACC-C02 | Critical | G5/G8 | Pointer authority is global and compare-and-swap is not safe across processes. | Scope pointers by workspace/trip and use a process-safe lock/CAS protocol. |
| ACC-C03 | Critical | G5/G7 | Reload does not reliably establish the accepted pointer as the next active parent. | Restore and validate pointer state at session/workspace creation. |
| ACC-C04 | Critical | G5 | `Inspect option` for Keep original records a durable keep decision. | Make inspection read-only and require explicit confirmation for decisions. |
| ACC-H01 | High | G3/G7 | Sessions and typed drafts are memory-only. | Persist ordered drafts and recover them with revision integrity. |
| ACC-H02 | High | G3/G5 | Normal selection/draft/session mutations do not share a complete locking discipline. | Define one mutation transaction boundary and concurrency tests. |
| ACC-H03 | High | G3/G5 | A proposal is not preserved as an immutable snapshot of draft, plan, evaluation, and certificate. | Persist immutable proposal references and invalidate on draft changes. |
| ACC-H04 | High | G4/G5 | Permission state is unscoped. | Bind permissions to session, proposal, operation, revision, and expiry. |
| ACC-H05 | High | G4/G7 | Transcript persistence, retention, and deletion are absent. | Add the approved local transcript store and deletion controls. |
| ACC-H06 | High | G5/G8 | Decision append and pointer replace are not crash-consistent as a unit. | Define recovery markers/order and test failures at each write boundary. |
| ACC-H07 | High | G5 | Rollback-by-new-decision is absent. | Implement rollback only to a previously eligible repository plan. |
| ACC-H08 | High | G5/G8 | Existing `PlanRepository` has no proven concurrent-writer contract. | Add locking/idempotency and concurrency tests before product use. |
| ACC-M01 | Medium | G5/G7 | Pointer and ledger integrity are not fully validated on load. | Validate schemas, hashes, monotonic revisions, and referenced plans. |
| ACC-M02 | Medium | G5 | Client code synthesizes canonical accepted state after the response. | Return and render the server-authoritative summary. |
| ACC-H09 | High | G8 | Current tests do not cover restart, crash boundaries, multiprocess races, rollback, or tampering. | Add black-box and fault-injection persistence coverage. |

## Closure evidence

G5 requires repository hash/lineage tests, concurrent CAS tests, crash recovery,
and a restart-persistent accepted pointer. G7/G8 require the same behavior from a
clean launcher journey, not only direct unit calls.
