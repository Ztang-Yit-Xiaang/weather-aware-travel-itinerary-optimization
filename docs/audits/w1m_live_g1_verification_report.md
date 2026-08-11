# W1M Live G1 Verification Report

**Generated:** 2026-08-04 (America/Chicago)  
**Product track:** CP-010 corrective local product  
**Provider decision:** MapLibre GL JS + local PMTiles primary; Atlas explicit licensed backup  
**Current verdict:** W1M/G1 verified; six independent auditors passed  

## Scope and non-claims

This evidence closes the live-runtime work owned by W1/W1M: pinned local assets, provenance, real PMTiles coverage, loopback HTTP serving, range/CORS/security behavior, product readiness, recovery, and a no-egress replay. It does not implement or verify the W2 geographic renderer, W3 repair execution, W4 OpenAI provider, W5 acceptance, W6 mobile/PWA, or W7/W8 replacement journey. It does not advance E3.1, E3.3, E4, E3.UX5, or E5.

## Staged local package

The package is stored in the ignored `docker/maplibre/assets/` directory and is not part of Git.

| Item | Verified value |
|---|---|
| MapLibre GL JS | 5.24.0, pinned npm integrity and commit |
| PMTiles JS | 4.4.1, pinned npm integrity and commit |
| Protomaps basemap style | 5.7.2, pinned npm integrity and commit |
| PMTiles CLI | 1.30.0, commit `39722fde57646c59c23e67a8cf739a59fc657590` |
| Source build | `https://build.protomaps.com/20260804.pmtiles` |
| Retained source BLAKE3 | `665c468f538ffaebd6fdf5b1185982b105056577a32f2fdc15c87fddd4b24b95` |
| Extracted archive SHA-256 | `28f8ab0d2e58437bd945944c66594b9531fb17b41aac8a4f4e9531edbe6842c3` |
| Archive size | 796,263,584 bytes |
| Bounds / maximum zoom | `[-123.0, 33.5, -117.5, 38.5]` / 15 |
| Registered workspace bounds | `[-122.478528, 34.101542, -118.289934, 37.8199]` |
| Representative real tile reads | 139 at zooms 8 and 12 |
| Closed runtime assets | 19 |
| License families | BSD-3-Clause, CC0-1.0, MIT, ODbL-1.0, OFL-1.1, Protomaps data notices |

The SHA-512 digests recomputed from all three downloaded npm tarballs exactly matched their registry integrity values. The PMTiles CLI ZIP SHA-256 exactly matched the GitHub release digest `8a983059b1e7a24f6930f677f5b7aa6d01e125eb2e15cb0f34bd11247e82df35` before extraction.

The generated style uses absolute loopback URLs for the PMTiles archive, sprites, and glyphs. The unsupported `landcover` layer was removed. Three concrete Noto Sans font stacks and both sprite densities are staged. Visible attribution is `© OpenStreetMap contributors`, linked to the copyright page. Runtime and preflight validation require the exact safe anchor, source build, source BLAKE3, package commits, npm integrity strings, and PMTiles CLI release digest.

## Live service and security checks

The local service uses the digest-pinned Nginx image `sha256:97d490c12ba55b4946b01546d1c3ed324e8d41ab1c9fcb2a616aa470620e5b46` and publishes only `127.0.0.1:8080`.

- Container health passed.
- Root filesystem is read-only.
- Asset and configuration mounts are read-only.
- `no-new-privileges:true` is active.
- The host and mounted-container archive hashes are identical.
- Every manifested small asset was fetched and matched its declared SHA-256.
- PMTiles returned HTTP 206 with exactly 1,024 bytes, `Content-Length: 1024`, `Accept-Ranges: bytes`, and `Content-Range: bytes 0-1023/796263584`.
- CORS returned only `http://127.0.0.1:8127`; wildcard and foreign-origin readability checks failed closed.
- Unsafe methods, unsatisfiable ranges, and encoded traversal paths were rejected.
- A secret-pattern scan found zero tracked-file matches and no secret-like value in container logs. Only environment-variable names were recorded.

The first traversal probe exposed that Nginx normalized `%2e%2e` to a root path. The configuration was repaired to reject encoded dot, slash, and backslash segments. An audit also found that `POST /healthz` returned 200 and that unexpected API exceptions did not receive the normal security headers. A global safe-method allow-list and a sanitized API exception response repaired both defects. The complete HTTP and API suites passed afterward. Failed probe attempts are not counted as passing evidence.

## Product readiness and recovery

With the map service available:

- `/api/health` returned `product-health-v2`, product version `0.2.0`, and `core_ready=true`.
- The map component returned `ready / maplibre_ready`.
- The deterministic Copilot adapter remained explicit; OpenAI was disabled rather than silently replaced.
- `/api/map/config` returned `product-map-configuration-v2`, provider `maplibre_pmtiles`, loopback-only URLs, and `runtime_license=null`.
- API responses retained `Cache-Control: no-store` and a CSP limited to self plus the validated loopback map origin.

When the map container was deliberately stopped, core readiness stayed true and map readiness became `degraded / maplibre_unavailable`. After restart and cache expiry, it returned to `ready / maplibre_ready` without selecting Atlas.

## No-egress replay

The service and a fresh test client were moved temporarily to a Docker network created with `--internal`. Both external-network probes failed. The fresh client then loaded local health and style resources and received the expected PMTiles HTTP 206, content range, and length headers. The normal loopback-published service was recreated afterward and returned healthy.

This proves the W1 local static-service contract without changing the host's network. The real browser renderer and its fresh-profile offline journey remain W2/G2 work because W1 deliberately does not initialize MapLibre in the browser.

## Browser and automated verification

The in-app browser loaded `http://127.0.0.1:8127/app`, displayed the product shell and validated workspace, navigated to `/app/map`, and reported no console warnings/errors or framework overlay. The page truthfully still labels the current canvas as a schematic; replacing it with the geographic renderer is W2. Screenshot evidence is `g1_live_app_browser.png` with SHA-256 `4ba91ecc426699ee901629698dda7007f8ec0354481a04c63520ac5b88a931ec`.

- Focused product and `PlanRepository` checks: 133 passed, one existing Starlette deprecation warning.
- Full project gate: Ruff passed, context snapshot 5 passed, full pytest 444 passed with the same warning.
- Docker Compose configuration and `git diff --check` passed.

The prescribed standalone `agent-browser` executable was unavailable. The supported Codex in-app browser controller was used instead and this substitution is recorded rather than hidden.

## Independent audit closeout

The Web Implementation, Copilot Privacy/Security, Map/Artifact Integrity,
Content, User Journey Black-Box, and Phase/Gate Status auditors each returned
PASS against the frozen pre-transition manifest SHA-256
`85876265fc5b72e4f552213254e7772238afb7526b24769d1f9186408e54cd60`
and report SHA-256
`515398940651f9e8d446d2280528a76392879a5bc58079b3d61161a25d1a9c2e`.
Implementation agents did not audit their own work. Blocking findings for
encoded traversal, unsafe map methods, unexpected API error headers, loose
upstream pinning, stale evidence hashes, and non-linked attribution were
repaired and independently reverified before sign-off.

## Gate transition

W1M and G1 are verified. W2 is ready but not started. CP-010 remains in
progress. E3.1, E3.3, and E4 remain blocked; E3.UX5 and E5 remain deferred.
The real geographic renderer, fresh-browser offline journey, three-plan demo,
route/stop layers, and G2 screenshots remain W2/G2 work.
