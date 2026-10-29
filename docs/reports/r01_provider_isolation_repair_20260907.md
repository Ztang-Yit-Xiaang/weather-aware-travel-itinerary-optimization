# R01 repair: deterministic provider isolation

Date: September 7, 2026, America/Chicago. Runtime receipts use September 8 UTC timestamps.

## Outcome and boundary

**R01 is implemented and verified for the current application-server Python dispatch path.** The deterministic runner forces the actual deterministic adapter, disables the child key without editing the user's configuration, blocks external HTTPX/socket dispatch before transmission, and requires a finalized, independently checked isolation receipt before reporting verified zero provider calls.

**This does not close the complete E2E-1/stability phase.** The first real browser run failed map acknowledgement after accepted-plan reload. Its failure is preserved. A second run passed with only an additional test diagnostic field; therefore the intermittent reload problem is not fixed or dismissed. R02-R24 remain open.

There were no real OpenAI calls in this repair. The production application, optimizer, evaluator, user's saved trip state, real key configuration, and historical evidence bundles were not edited. Both browser runs used disposable sessions and test state. The owned test server shut down; local map/routing services were left running.

## Changes and why they were needed

| File | Focused change |
| --- | --- |
| `scripts/e2e1_provider_isolation.py` | New disposable bootstrap because an environment-only override cannot observe or block dispatch inside the application process. Installs HTTPX sync/async and socket-connect guards before app imports; verifies concrete adapter identity; writes a run/PID-bound receipt; finalizes only after clean server shutdown. |
| `scripts/run_e2e1_vertical_slice.py` | Overrides inherited adapter/key for both subprocess environments; launches the guarded bootstrap; rejects incorrect/malformed adapter health before browser execution; replaces unconditional zero-call reporting with receipt-backed reporting; retains failures and forces a nonzero result after runner errors. |
| `scripts/e2e1_source_universe.py` | Includes the new bootstrap in the deterministic source inventory. |
| `scripts/verify_e2e1_vertical_slice.py` | Independently checks manifest v3, exact receipt binding, adapter, lifecycle, run/PID, timestamps, counters, blocked attempts, artifact inventory and reproducible report. |
| `tests/product_app/test_e2e1_provider_isolation.py` | New adversarial boundary tests; no real external transport. |
| `tests/product_app/test_e2e1_runner.py` | Reproduces poisoned inheritance, preflight rejection, invalid telemetry and post-pytest output-write failure; preserves the primary failure. |
| `tests/product_app/test_e2e1_evidence.py` | Adds strict witness/forgery cases and binds complete-bundle tests to the new v2 evidence. |
| `tests/product_app/browser/test_e2e1_vertical_slice_live.py` | Adds only `map_status_code` to existing accepted-reload diagnostics. No waits or assertions were weakened. |

The opt-in authorized live/billed branch remains separate. This repair did not change its model, retry policy, or credentials.

## Test process and retained results

| Step | Exact observed result | Meaning |
| --- | --- | --- |
| Old-runner reproductions | `2 failed, 6 deselected in 0.55s` | The child inherited OpenAI mode and readiness accepted it. Expected red result; preserved in `results/repairs/r01_provider_isolation_20260907/red_tests.xml`. |
| Independent guard tests, first pass | `1 failed, 28 passed in 0.74s` | Exposed an unstructured error for a malformed runtime. Fixed, not waived. |
| Independent guard recheck | `29 passed in 0.54s` | External fake transport never receives the blocked request; local dispatch remains allowed; actual adapter and cleanup checks hold. |
| Final runner/guard checks | `49 passed in 2.06s` | Includes blocked/incomplete receipts and passing pytest followed by an injected output-save failure. |
| Independent pure witness/forgery cases | `62 passed, 34 deselected in 0.33s` | Rejects missing, unfinished, mismatched, boolean/string-valued and coherently rehashed invalid receipts. |
| Real browser v1 | `1 failed, 6 passed in 175.26s (0:02:55)` | Provider isolation passed; accepted-reload map failed. Overall runner result was correctly false/nonzero. |
| Focused checks against incomplete v1 | `8 failed, 137 passed in 10.59s` | Full-bundle/mutation checks could not accept the failed, incomplete browser fixture. The failed JUnit evidence is retained, not relabeled. |
| Real browser v2 | `7 passed in 145.28s (0:02:25)` | Same production source; only additional test diagnostics. Full current-source independent verifier passes. |
| Final complete focused checks against v2 | `145 passed in 22.40s` | Runner, guard and evidence-verifier modules all pass. |
| Static checks | `All checks passed!` | Focused Ruff checks; E402 is the pre-existing standalone-script import bootstrap exception. |

Three independent subagents reviewed transport safety, adversarial guard tests, and evidence verification. The safety review also found and closed a case where saving test output could fail after pytest returned zero. This now records the original pytest result and nonzero runner result separately in failure evidence.

## Exact execution inputs

Both real browser invocations deliberately used these **synthetic parent settings**, local to their command processes:

```powershell
$env:PRODUCT_COPILOT_ADAPTER='openai'
$env:OPENAI_API_KEY='synthetic-test-key-not-a-credential'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
python scripts/run_e2e1_vertical_slice.py --output-dir results/e2e/r01_provider_isolation_20260907_v1
```

The second invocation used the same settings and `--output-dir results/e2e/r01_provider_isolation_20260907_v2`. Inside the application and pytest subprocesses the adapter was forced to `deterministic` and `OPENAI_API_KEY` was set to an empty string. The parent mapping was not mutated. An explicitly empty key is intentional: the local configuration loader cannot overwrite that present-but-empty value from `.env.local`.

Final independent verification and focused tests:

```powershell
python scripts/verify_e2e1_vertical_slice.py --output-dir results/e2e/r01_provider_isolation_20260907_v2
python -m pytest tests/product_app/test_e2e1_runner.py tests/product_app/test_e2e1_provider_isolation.py tests/product_app/test_e2e1_evidence.py -q -p no:cacheprovider --basetemp tmp/r01_full_verification_20260907_b --junitxml results/repairs/r01_provider_isolation_20260907/final_v2_verification_tests.xml
python -m ruff check scripts/e2e1_provider_isolation.py scripts/run_e2e1_vertical_slice.py scripts/e2e1_source_universe.py scripts/verify_e2e1_vertical_slice.py tests/product_app/test_e2e1_runner.py tests/product_app/test_e2e1_provider_isolation.py tests/product_app/test_e2e1_evidence.py tests/product_app/browser/test_e2e1_vertical_slice_live.py --ignore E402
```

These are validation commands actually executed, not proof that all repository tests or all interactions pass.

## Exact final output and runtime witness

Independent verifier output:

```json
{
  "artifact_files": 98,
  "llm_records": 7,
  "manifest_content_hash": "5c97ec2c75c5c7fa4b7a51272442ba979cc474ed948c5047672fcb6a385aab36",
  "manifest_sha256": "69ed7942e2b2df4fbf3f0f4b631d111ef3f07d782e109d32cc7d28b0ebbd4dcd",
  "provider_calls": 0,
  "report_sha256": "ab28fff6e25b3591b1fe9f5e62ac14757582e84e0f065e1f9f3d0988aa7611fe",
  "scenarios": 6,
  "source_inputs": 169,
  "verified": true
}
```

The [complete receipt](../../results/e2e/r01_provider_isolation_20260907_v2/provider_isolation.json) records run `e2e1_vertical_slice_20260908T033037425749Z`, PID `22832`, actual `DeterministicCopilotAdapter`, disabled key, finalized lifecycle, **633 allowed local socket connections, zero blocked attempts and zero sent provider calls**. Installation was `2026-09-08T03:30:38.079721+00:00`; finalization was `2026-09-08T03:33:08.010777+00:00`.

Zero sent calls is established by the enforced deny policy and finalized application-process instrumentation. It is not an independent provider-billing audit or an OS/browser/descendant-process firewall. Any blocked attempt fails this deterministic gate; a missing or incomplete witness yields unknown/null, not a fabricated zero.

## Real browser-to-artifact evidence

The seven browser tests exercise registered replacement, Copilot one-click preview, route-aware candidate discovery/replacement, unsupported removal, ineligible duration, stale-response protection and accepted-plan reload. They use actual browser controls and the real local application/optimizer/evaluator. This is not a claim of coverage for every advertised operation, a true concurrent A/B race, or all viewport/pixel-coverage requirements.

The exact current fixture is **Bixby Creek Bridge → Pfeiffer Big Sur State Park**, not Pfeiffer Beach. The [registered trace](../../results/e2e/r01_provider_isolation_20260907_v2/registered_replacement/trace.json) records:

```json
{
  "request_id": "w3_preview_3c21d335a49ac208",
  "parent_plan_id": "plan_359968d255ff9288",
  "child_plan_id": "plan_c9a1dffb7461d665",
  "child_plan_content_hash": "e0c7162763e71edd",
  "certificate_id": "cert_05e7851692e958f9",
  "diff_id": "diff_30f0e2174d16b5a4",
  "comparison_eligibility": "eligible",
  "geometry_hash": "9163802f28fe20da03c40e1a022ab5fdad2d0524fe423844255dd8321056d680",
  "rendered_map_plan_id": "plan_c9a1dffb7461d665",
  "assertions_passed": true
}
```

This excerpt is explicitly a projection; the linked trace preserves every field and marker ID. The [accepted-reload trace](../../results/e2e/r01_provider_isolation_20260907_v2/accepted_reload/trace.json) confirms the same child/hash/geometry and markers after reload. Full-size [reload screenshot](../../results/e2e/r01_provider_isolation_20260907_v2/accepted_reload/screenshot_after_reload.png) and [render diagnostics](../../results/e2e/r01_provider_isolation_20260907_v2/accepted_reload/reload_rendered_diagnostics.json) are preserved. Structural acknowledgement passed; no independent visible-pixel coverage claim is made.

## Exact file and LLM input/output

- [Full exact-I/O report](../../results/e2e/r01_provider_isolation_20260907_v2/exact_io_report.md): every captured file's role, byte size, SHA-256 and exact content location, plus all seven LLM-boundary records. Large payloads are linked in full, not summarized away.
- [Manifest](../../results/e2e/r01_provider_isolation_20260907_v2/file_io_manifest.json): 98 captured artifacts and 169 exact source/fixture inputs. This is the defined source universe, not a complete dependency/browser/OSRM-dataset environment attestation; R24 remains open.
- [Accepted-reload chat wire I/O](../../results/e2e/r01_provider_isolation_20260907_v2/accepted_reload/copilot_message_http_io.json): exact browser request body and raw SSE response. The user message is `Review the current itinerary without changing it.` This is a real website chat request to the **deterministic local adapter**, not a GPT response.
- [Exact preview request](../../results/e2e/r01_provider_isolation_20260907_v2/registered_replacement/preview_request.json), [response](../../results/e2e/r01_provider_isolation_20260907_v2/registered_replacement/preview_response.json), diff/evaluation/map files and server/browser logs are all bound by the manifest.
- [Final focused-test JUnit](../../results/repairs/r01_provider_isolation_20260907/final_v2_verification_tests.xml) preserves per-test results. [Earlier failed full check](../../results/repairs/r01_provider_isolation_20260907/final_verification_tests.xml) remains separately preserved.

Exact external-LLM boundary for accepted reload:

```json
{
  "llm_input": null,
  "llm_output": null,
  "provider_call_count": 0,
  "reason": "Direct repair and acceptance with a deterministic local Copilot transcript; exact chat I/O is in copilot_message_http_io.json and no external provider is invoked."
}
```

Null means no external LLM input/output exists, not omitted evidence. Fake-transport unit cases remain explicitly synthetic in their test source; their injected requests/responses are not claimed as paid provider observations. Credentials and authentication headers are intentionally not persisted in reports.

## Retained failure and next issue

V1 remains at [its exact-I/O report](../../results/e2e/r01_provider_isolation_20260907_v1/exact_io_report.md), [failed browser output](../../results/e2e/r01_provider_isolation_20260907_v1/pytest_output.txt) and [reload diagnostics](../../results/e2e/r01_provider_isolation_20260907_v1/accepted_reload/reload_failure_diagnostics.json). Its original manifest hash is `a4a114a0a660c340780a4d92d1e309c0ce09146e3683058a27d8123c5bb2468e`. The independent verifier correctly rejected overall completion while separately validating its finalized isolation witness: zero sent, zero blocked, 648 local connections.

Reload returned the same accepted session and retained Pfeiffer/Day 5. A network-free Node check of the saved accept/reload geography through the frontend plan validator passed. The v1 test did not record the map status code, so the precise cause is **unresolved**. A transient map configuration/readiness or asset-runtime failure is a hypothesis, not a diagnosed cause. V2 adds the missing code and records `geographic_map_v2_ready` on its passing reload.

Next priority before a stability claim: capture exact reload health/map-configuration responses and failure code under repeated/failure-injected initialization, then implement and verify the corresponding recovery without weakening plan identity or map-readiness requirements. Keep this follow-up separate from R01. The next planned semantic repair remains R02: distinguish missing route evidence/not-evaluated from independently evaluated ineligibility. No R02 change was made in this repair.
