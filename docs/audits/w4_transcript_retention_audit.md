# W4 Transcript Retention Audit

## Audit identity and independence

- Role: `TRANSCRIPT_RETENTION_AUDITOR`
- Date: 2026-08-06
- Scope: W4 conversation persistence, shared locking, workspace ownership binding,
  transcript APIs, retention/deletion behavior, and transcript exclusion boundaries.
- Independence: this auditor did not implement the audited W4 modules and made no
  production-code changes. The only file written by this audit is this report.
- Live OpenAI transport: outside this audit's implementation verdict. G4 may still
  remain blocked by the live-provider gate and other independent audits.

## Verdict

**PASS**

Re-audit completed after the TR-001 correction. Delete-current now restores the
session-to-conversation binding when repository deletion fails, and the new API
regression proves that the returned `conversation_delete_failed` response leaves
the same conversation ID and turns reachable. The durable conversation codec,
30-day retention, bounded schema, restart behavior, UTF-8 serialization, shared
locking, ownership checks, deletion controls, and evidence/health exclusion meet
the W4 transcript-retention gate. The remaining failure-injection suggestions are
defense-in-depth coverage and are not G4 blockers.

## Commands and results

### Focused deterministic suite

```powershell
$testBase = Join-Path ([System.IO.Path]::GetTempPath()) `
  ('w4-transcript-audit-' + [guid]::NewGuid().ToString('N'))
python -m pytest tests\product_app\test_conversations.py `
  tests\product_app\test_copilot_api.py `
  tests\product_app\test_copilot_privacy.py `
  tests\product_app\test_persistence.py `
  tests\product_app\test_workspace_w3.py -q --basetemp $testBase
```

Initial result: **32 passed**, 1 unrelated Starlette/httpx deprecation warning,
in 57.58 seconds.

Re-audit result after TR-001 correction: **33 passed**, 1 unrelated
Starlette/httpx deprecation warning, in 66.84 seconds.

### Adversarial restart and write-failure probe

An isolated temporary state root was used to:

1. write and reload a multilingual UTF-8 turn through a newly constructed
   `ConversationRepository`;
2. inject an `OSError` at `os.replace` while appending a second turn; and
3. compare the conversation bytes before and after the failed replace.

Result: **PASS** — restart restored the typed conversation and a failed replace
preserved the prior valid bytes exactly.

### Static boundary inspection

```powershell
rg -n "conversation|evidence-bundle|health|delete_all" `
  src\itinerary_system\product_app tests\product_app -g "*.py"
```

Result: conversation files are stored only below the configured state root;
evidence downloads continue to use manifest-declared immutable run artifacts.
The focused privacy test also confirmed that transcript content is absent from
health and evidence responses.

## Findings

### TR-001 — Delete-current failure recovery

- Original severity: **High / G4 blocker**
- Re-audit status: **Resolved**
- Correction inspected:
  - `ProductService.delete_conversation` still detaches before repository
    deletion, but now catches `ConversationError`, restores the exact prior
    conversation binding through `WorkspaceStore.bind_conversation_id`, and
    re-raises the stable deletion error.
  - The new API regression injects `conversation_delete_failed`, verifies HTTP
    503 with the stable error code, then reads the conversation again and verifies
    that both its ID and turns match the pre-failure record.
- Re-audit result: the normal storage-failure path is truthful, retryable, and no
  longer leaves retained transcript bytes orphaned from the owning session.

### TR-002 — Additional deletion failure-injection coverage

- Severity: **Medium hardening suggestion / non-blocking**
- Current coverage: delete-current now has service/API-level injected repository
  failure coverage and proves binding recovery. Normal delete-one/delete-all,
  malformed and oversized recovery files, capacity/turn limits, ownership, path
  validation, TTL, and message idempotency also pass.
- Remaining suggestion: add lower-level deterministic injections for an `unlink`
  failure, a directory-`fsync` failure after unlink, a binding-restore failure, and
  a partial delete-all failure. These double-failure and partial-operation cases
  should retain stable error codes and document their recovery procedure.
- Gate effect: none for G4. The implemented repository error normalization plus
  the new API reachability regression resolves the original blocker.

## Controls that passed inspection

- Stable `product-conversation-v1` and `copilot-turn-v1` schemas reject unknown
  fields and unsupported schema values.
- Conversation, session, turn, and client-message IDs are strict and never derive
  paths from user text.
- The conversation directory and entries reject symlinks and non-regular files;
  direct paths must remain immediate children of the configured conversation root.
- Files are stable sorted UTF-8 JSON with a terminal newline and a 1 MiB limit.
- Writes use a same-directory exclusive temporary file, flush, file `fsync`,
  validation, `os.replace`, directory `fsync`, and post-write validation.
- The shared `.product_app_state/locks/layout.lock` serializes conversation and
  session state across threads/processes.
- Ownership is enforced by `session_id`; cross-session reads/deletes fail closed.
- Retention is exactly 30 days from `updated_at`, cleanup runs at initialization
  and no more than hourly afterward, and expired files are deleted.
- Limits are enforced at 200 turns, 1 MiB per file, and 128 active conversation
  files; active data is not silently truncated or evicted.
- Duplicate client-message IDs are idempotent only for identical user-message
  content; conflicting reuse fails closed.
- Session binding persists through the W3 snapshot store without advancing the
  itinerary/draft revision.
- Conversation read/delete routes require the authenticated session token;
  delete-all additionally requires a session ID, exact confirmation, expected
  revision, and an Origin-bearing unsafe request.
- Transcript content does not enter health JSON, evidence ZIPs, immutable run
  artifacts, or the inspected error payloads.

## Blockers and re-audit conclusion

No transcript-retention blocker remains for G4.

- TR-001 is resolved and independently re-tested.
- TR-002 is retained as a non-blocking hardening recommendation.
- Live-provider verification and verdicts from the other independent W4 auditors
  remain separate G4 conditions and are not implied by this PASS.
