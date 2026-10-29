# G4 current-source live provider evidence

Date: 2026-08-13  
Model: `gpt-5.4-mini`  
Status: **verified / passed after independent re-audit**. This does not open W5; Accept and Keep remain disabled.

## Result

The authorized current-source evaluation ran the same fixed 24 synthetic cases at low and medium reasoning:

| Reasoning | Cases | Schema | Exact state/intent | External safety | Injection safety | Clarification | Invalid IDs | Automatic actions | Input tokens | Output tokens | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 | 36,324 | 4,049 | 32,592.472 ms |
| Medium | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 | 36,324 | 5,960 | 60,956.607 ms |

Both efforts passed every threshold. Low is recommended because quality is equal while it used 1,911 fewer output tokens and completed about 28.4 seconds faster.

The run recorded exactly 48 logical cases, 48 SDK/provider attempts, and 48 HTTP transport attempts. No retry or extra provider attempt occurred. Exact-token cost for the full run is estimated at USD 0.0995265 using the published `gpt-5.4-mini` text-token rates.

## Exact file and LLM input/output

The human-review report contains every exact synthetic request body, provider response body, and file input/output record:

- `docs/audits/g4_exact_file_and_llm_io_20260813_gpt54mini.md`
  - SHA-256: `37000873e06a244329de818f6bb0edd62bbd289d7155d0404a4eac57ea5bf132`
  - size: 1,071,873 bytes

Machine evidence:

- `results/product_app/g4_live_20260813/provenance_bound_v6_gpt54mini/fixed24_reasoning_comparison/low_exact_llm_io.json`
  - SHA-256: `cde01941f27bae930d1786a41078a903fae5604591afdd653a328691a46b5e99`
- `results/product_app/g4_live_20260813/provenance_bound_v6_gpt54mini/fixed24_reasoning_comparison/medium_exact_llm_io.json`
  - SHA-256: `e231c4041dd6bf1504361b9422e6face4c3dd77577e2683e952345531732eb8e`
- `results/product_app/g4_live_20260813/provenance_bound_v6_gpt54mini/fixed24_reasoning_comparison/file_io_manifest.json`
  - SHA-256: `4b0f7958b11514b6073bdc1de0dcc648d4ba22501abaddfe07809212b2c2af1f`
- `results/product_app/g4_live_20260813/provenance_bound_v6_gpt54mini/g4_current_source_manifest_v1.json`
  - SHA-256: `56b5007879f507eab50456692364294d0167bd29b2edd3b3f19b238cf7ded394`
  - canonical content hash: `ee5ddb1119b9bb2ed120a1e51afec66fd547d8184f645c3054df751f3302e3ba`

## Independent offline verification

`scripts/verify_live_copilot_prompt_evidence_v4.py` independently replayed all 48 exact request/response pairs through the strict response schema, current server canonicalization, local contract, corpus expectations, source bindings, summaries, and recommendation. It returned `verified=true`, 48 logical/provider/transport attempts, and low recommended.

The post-run evidence tests passed 17/17, including coherent request, response-projection, and transport-count tampering. The root manifest binds the verifier, deterministic report renderer, and all three test files used by that 17-test command. Ruff and Python compilation passed. A separate privacy/file audit confirmed 48 unique provider response IDs, HTTP 200/completed for every case, exact request/response hashes, current input/output file bindings, `store=false`, `tools=[]`, and no persisted API key, Authorization header, bearer token, cookie, or mutation token.

## Independent re-audit outcome

Three separate read-only audit contexts reviewed the frozen current-source bundle:

- Provider/prompt/evaluation: PASS. All 48 requests were reconstructed from the current prompt, strict schema, synthetic context, and corpus; all 48 responses passed strict projection and local-contract replay. Coherent request, response, control, source, transport-count, and file-binding tampering failed closed.
- Security/privacy/memory: PASS. A 203-test focused suite passed with one known Starlette/httpx deprecation warning; no secret, Authorization/Bearer, cookie, mutation token, non-synthetic context/history, tool execution, automatic state mutation, Accept, or Keep action was found.
- Evidence gate/status: initially FAIL because the root manifest bound only one of the three files used by the claimed 17-test command and omitted the deterministic report renderer. The root manifest was repaired to bind all five audit files, regenerated, and re-audited. The final verdict is PASS with 17/17 bound tests, exact report rerender, closed inventory, coherent chronology, and W5/Accept/Keep still closed.

Final root manifest SHA-256: `56b5007879f507eab50456692364294d0167bd29b2edd3b3f19b238cf7ded394`. Final canonical content hash: `ee5ddb1119b9bb2ed120a1e51afec66fd547d8184f645c3054df751f3302e3ba`.
## Boundary

This evidence covers the current Copilot prompt/provider/source boundary and the fixed synthetic 24-case corpus. The separate one-call statewide Redwood product smoke is recorded in `g4_statewide_copilot_smoke_20260813.md`. Three independent read-only reviews now pass after the initial audit-binding defect was repaired and rebound. G4 is verified for this v6 source snapshot. W5 remains closed throughout.
