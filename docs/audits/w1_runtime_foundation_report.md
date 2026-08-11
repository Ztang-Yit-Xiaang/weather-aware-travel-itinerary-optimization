# W1 Runtime Foundation Implementation Evidence

**Recorded:** 2026-08-03  
**Scope:** W1 runtime foundation only  
**Product status:** implemented  
**G1 status:** blocked by live Atlas prerequisites

## Outcome

The deterministic W1 boundary is implemented. The application now validates its pinned default workspace before declaring core readiness, reports versioned component health, enforces loopback/configuration/security boundaries, initializes a versioned local-state layout without importing legacy decisions, exposes a no-store Atlas map contract, and launches only after a matching product-health response.

This report is implementation evidence, not G1 verification. No geographic map, OpenAI transport, persistent draft/session store, repository-backed acceptance, mobile redesign, or PWA was implemented in W1.

## Deterministic verification

| Check | Result |
|---|---|
| Focused W1 and `PlanRepository` suite | 89 passed; one Starlette deprecation warning |
| Repository project checks | Passed |
| Full pytest through project checks | 400 passed; one Starlette deprecation warning |
| Ruff | Passed |
| Atlas Compose static configuration | Passed; Docker client-config access warning only |
| JavaScript syntax | Passed |
| Diff whitespace check | Passed; existing line-ending notices only |
| Disposable HTTP health smoke | `degraded`, `core_ready=true`; registry/workspace/state ready; Atlas unavailable; deterministic adapter explicit |
| Legacy-state preservation | Existing pointer and three decision hashes unchanged |

Commands:

```powershell
python -m pytest tests\product_app tests\plans\test_repository.py -q -p no:cacheprovider --basetemp <disposable-root>
python -m ruff check src\itinerary_system\product_app tests\product_app scripts\run_product_app.py
python scripts\run_project_checks.py
docker compose --env-file docker\atlas\.env.example -f docker\atlas\docker-compose.yml config --quiet
node --check src\itinerary_system\product_app\static\js\app.js
git diff --check
```

## Readiness snapshot

The disposable local service returned `product-health-v1` with:

- `product_id = itinerary-repair-copilot`
- `status = degraded`
- `core_ready = true`
- `registry = registry_ready`
- `default_workspace = workspace_ready`
- `state_store = state_store_ready`
- `atlas = atlas_unavailable`
- `openai = deterministic_adapter_selected`

The degraded result is expected and truthful. The browser map is not initialized in W1.

Audit remediation also established that configured OpenAI remains degraded until W4 transport exists; acceptance and Keep original are disabled until W5; legacy pointers are preserved but not trusted; Atlas redirects and exact CORS origin are validated; and oversized streams are rejected before their chunks are copied into the request buffer.

The user-journey repair additionally gives an occupied fixed port a bounded eight-second matching-health retry. A process-level regression expires the Atlas health cache, invokes a second launcher, verifies exit 0 reuse, terminates the server, and verifies the port is released.

## Independent audit closeout

| Auditor | Scope | Verdict | Independent/read-only |
|---|---|---|---|
| `w1_web_auditor` | Web implementation and recovery contracts | PASS after remediation | Yes |
| `w1_security_auditor` | Copilot privacy and runtime security | PASS after remediation | Yes |
| `w1_map_integrity_auditor` | Map/Atlas and artifact integrity | PASS after remediation | Yes |
| `w1_content_auditor` | Content truthfulness and authority copy | PASS after remediation | Yes |
| `w1_user_journey_auditor` | Deterministic process/API user journey | PASS after launcher remediation | Yes |
| `w1_phase_gate_auditor` | Phase, gate, dependency, and research status | PASS after synchronized closeout | Yes |

## External preflight

| Dependency | Observed state | Gate consequence |
|---|---|---|
| Docker CLI | 29.6.1 | Available |
| Docker Compose | 5.3.0 | Available |
| Docker server | 29.6.1 running with active Linux engine; `hello-world` passed | Verified from user PowerShell evidence |
| WSL/Linux engine | WSL 2.7.11.0; kernel 6.18.33.2; Ubuntu and `docker-desktop` running as version 2 | Verified from user PowerShell evidence |
| F drive free space | 41.15 GiB | Meets the documented development minimum |
| Atlas installer token | Absent | Blocks licensed asset acquisition |
| Atlas runtime license | Absent | Blocks live browser map contract |
| Nginx image digest | Deliberately unresolved | Must be resolved from an approved live image |
| Atlas assets | Absent | Blocks style/PMTiles verification |
| Port 8080 | Not accepting connections | Atlas not running |
| OpenAI key | Absent | Recorded for W4; does not block deterministic W1 |

Stable live blocker codes:

- `atlas_credentials_unavailable`
- `atlas_image_digest_unresolved`
- `atlas_assets_unavailable`
- `atlas_live_preflight_not_run`

The image digest was not fabricated, credentials were not requested or logged, and licensed assets were not placed in Git.

## User-like browser check

After the initial W1 closeout, a user opened `static/index.html` directly through a `file:///` URL. That source page cannot load server-relative CSS, JavaScript, or `/api/health`, so it remained on the initial readiness text. The repair adds source-page guidance naming the supported root launcher and HTTP URL, hides that guidance on a supported server-hosted load, and documents the distinction in the root README.

The canonical root launcher was then started on port 8127. The in-app browser loaded `http://127.0.0.1:8127/app`, initialized the California workspace, displayed the truthful degraded Atlas banner, and kept the direct-file guidance hidden. A read-only independent user-journey re-audit passed. Clean `.cmd`/Chrome and user-issued `Ctrl+C` verification remains owned by W7; G1 remains blocked independently by live Atlas prerequisites.

## Boundary and status decision

- W1 may be marked `implemented` only after every required independent W1 audit closes without a blocking finding of any category.
- G1 must remain `blocked` until Atlas entitlement/license, licensed assets, digest pinning, live HTTP 206 range behavior, and offline-local checks pass.
- W2 must not start.
- CP-010 remains `in-progress`.
- E3.1, E3.3, E4, E3.UX5, and E5 remain unchanged.
