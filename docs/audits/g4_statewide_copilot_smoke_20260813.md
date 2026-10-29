# Current-source statewide Copilot live smoke

Date: 2026-08-13  
Scope: one synthetic product/API message using the new statewide California POI context.  
Model: `gpt-5.4-mini` with low reasoning; provider resolved snapshot `gpt-5.4-mini-2026-03-17`.  
Status: **live smoke passed; full current-source G4 fixed evaluation remains pending**. W5 remains closed and Accept/Keep remain disabled.

## Result

- One final evidence-bound provider attempt completed successfully.
- The typed result was `clarification_required` with one `request_clarification` intent.
- The safe server-owned response named Redwood National and State Parks, disclosed that road access and routing were not validated, requested add/replace and day context, and stated that nothing changed.
- Protected itinerary state was byte-equivalent before and after the Copilot turn.
- The provider response reported 4,238 input tokens and 266 output tokens, including 109 reasoning tokens.
- At the published `gpt-5.4-mini` text-token rates, the evidence-bound final call cost is estimated at approximately USD 0.0044.

## Exact input and output

The exact synthetic product input/output and exact LLM HTTP request/response bodies are preserved in:

- `results/product_app/g4_live_20260813/statewide_smoke_gpt54mini/statewide_copilot_exact_io_report.md`
- `results/product_app/g4_live_20260813/statewide_smoke_gpt54mini/statewide_copilot_exact_io.json`
- `results/product_app/g4_live_20260813/statewide_smoke_gpt54mini/file_io_manifest.json`

SHA-256:

- evidence JSON: `4a6b94a142bcf3ad973d0acae808058270a87481eb4f7fdf76b262326bb4e084`
- human exact-I/O report: `ffb053f206c3fd2e465d55e68809f83d1f8d91c2ef2886a8f7e318bfcd1432c4`
- file-I/O manifest: `25c17a99701661b05102eecd997cd4b9f745a03d92bc64fc29d52f6a3715945e`

The exact provider witness stores request and response bodies intentionally for this synthetic audit. It does not store the API key, Authorization or cookie headers, the product session mutation token, or raw exception traces.

## Verification

Independent offline recomputation passed all checks: evidence and manifest content hashes; exact request/response body hashes; one SDK/provider attempt; one HTTP transport record; requested/resolved model identity; `store=false`; `tools=[]`; low reasoning; HTTP 200/completed status; strict Redwood semantic result; current source/file bindings; protected-state invariance; and credential/header exclusion.

## Failed pre-evidence attempts

Two earlier authorized billed calls reached the provider but were not evidence-bound because the new harness failed after the response: first on an incorrect outer-turn state assertion, then on cross-event-loop HTTP-client cleanup. Both harness defects were corrected and offline-tested before the final call. Their exact token usage and bodies were not persisted, so this report does not estimate or reconstruct them. The evidence above covers only the final successful one-attempt run.

## Gate boundary

This single current-source live smoke establishes connection, strict product interpretation, safe local canonicalization, and non-mutation for the statewide Redwood case. It does not replace the fixed-24 reasoning evaluation or independently reverify the full current-source G4 gate. Historical v5 G4 evidence remains preserved for its frozen source snapshot; current-source G4 revalidation remains pending and requires separate authorization if a billed multi-case run is desired.
