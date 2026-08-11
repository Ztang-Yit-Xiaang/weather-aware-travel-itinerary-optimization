# W2 Geographic Workspace / G2 Verification Report

**Generated:** 2026-08-04 (America/Chicago)  
**Track:** CP-010 corrective local product  
**Candidate artifact:** `california_coast_product_demo_v1`  
**Historical v1 gate result at generation time:** W2/G2 `verified`; W3 `ready`  
**Current authority:** see `docs/planning/current_execution_plan.md`

## Scope and non-claims

This report records the W2 read-only geographic workspace: the deterministic
parent-plus-two-child demo, artifact-derived MapLibre routes and stops, selected
alternative synchronization, Compare, and progressive Evidence. It does not
implement W3 drafts, W4 OpenAI, W5 acceptance, W6 PWA/mobile routes, or W7/W8
replacement verification. It does not advance E3.1, E3.3, E4, E3.UX5, or E5.

## Frozen demo evidence

| Item | Verified value |
|---|---|
| Package manifest SHA-256 | `56728434b2f53a796202d7cf10e7e519dd652a96dc2aadf3a626ede10ee585ad` |
| Package inventory | 26 declared nested files; closed and SHA-256 pinned |
| Parent | `plan_e1c4f803691e3188` / `20b540fdc5ed5cc9` |
| Recommended child | `plan_3f341b2c989837d2` / `c434706b88fc3d41` |
| Low-driving child | `plan_37018c91aacc87b3` / `928619fdc7f34210` |
| Recommended certificate | `cert_bf2c713be313c39c`; eligible; 16/16 road-validated; zero fallback |
| Low-driving certificate | `cert_2d7ef2f6fe5b6572`; eligible; 16/16 road-validated; zero fallback |
| Strict route totals | 539.8 and 484.693333 minutes |
| Evidence scope | Parent road-validated; evaluated children certified |

Every rendered leg is an exact coordinate-order conversion of a pinned
RouteMatrix cell. The original has no evaluation certificate and is therefore
never labeled certified. Generated redacted configurations contain repository-
relative source paths; a raw Windows drive-path scan found no package match.

## Final browser journey

A fresh controlled browser loaded `http://127.0.0.1:8127/app` at 1280×900,
rendered the MapLibre WebGL map, opened Compare, selected the low-driving child,
and opened Evidence. Evidence displayed the selected child's 16/16 certificate,
484.69-minute route, policy/modeled checks, Santa Barbara Sea Center route text,
nine selected-plan source hashes, methods, diff, and technical lineage. The
download link was bound to `plan_37018c91aacc87b3`.

Both mobile evidence PNGs are physically 390×844. The capture viewport reported
`innerWidth=390`, `innerHeight=846`, `devicePixelRatio=1.0`,
`clientWidth=390`, and `scrollWidth=390`; no horizontal document overflow was
present. The primary capture shows every toolbar label, the expanded selected-
alternative route through the Sea Center, linked OpenStreetMap attribution, and
the complete bottom navigation without right-edge clipping. The detail capture
shows the Sea Center plus Golden Gate Bridge labeled `Original only; removed by
selected alternative`. W3 editing and W5 decision controls stayed disabled.
Browser console diagnostics returned zero entries.

The browser DOM asset inventory contained only:

- `http://127.0.0.1:8127` for the application module and stylesheet.
- `http://127.0.0.1:8080` for MapLibre GL JS/CSS and PMTiles JS.

Map configuration, CSP, style URLs, glyphs, sprites, and PMTiles sources are
validated as exact loopback origins. The earlier W1M Docker-internal replay
separately proved local map assets operate without external egress. This W2
journey did not disable the host network globally.

## Screenshots

| View | File | Bytes | SHA-256 |
|---|---|---:|---|
| Selected low-driving Evidence, 1280×900 | `w2_g2_selected_evidence_1280.png` | 126002 | `dc0b3d906833bc05aa3e760428805dcfa4c9aef55aecd64dd9773d642dbfcb16` |
| Selected alternative map/text, 390×844 | `w2_g2_map_390.png` | 54430 | `e38431b1a0a5e0a253009fb8cc524b885dc409884999bbede63c80cdef41fce8` |
| Sea Center and original-only Golden Gate detail, 390×844 | `w2_g2_map_text_390.png` | 52207 | `48d7bc274392113ef9b448b92f2dcf6ea22e18fe1ec839ef5341c10841cb748a` |

## Automated verification

- Focused product and `PlanRepository` gate: **154 passed**, one existing
  Starlette deprecation warning.
- Repository-wide gate: Ruff passed, context snapshot **5 passed**, full pytest
  **465 passed**, with the same warning.
- Node syntax checks and `git diff --check` passed.
- Selected-plan ZIP regression verifies the low-driving plan/certificate are
  present and recommended-plan evidence is absent.
- Adversarial tests reject modified inventory bytes, extra files, matrix/hash
  tampering, false certificate scopes, missing edit metrics, absolute host
  paths, invalid alternatives, and unbound multi-plan downloads.

## Independent audit state

- Map/Artifact Integrity: PASS after two adversarial repair cycles.
- Web/Accessibility: PASS after responsive, retry, W3-scope, and map-selection repairs.
- Content/Truth/User Journey: PASS after the live non-ranking baseline repair.
- Security/Runtime: PASS after selected-plan download, path-redaction, and live-service checks.
- Phase/Gate Status: PASS after authority-chain corrections.
- Visual/Responsive Evidence: PASS after exact-width, unclipped mobile recapture.

W2/G2 are `verified`; W3 is `ready`. CP-010 remains `in-progress`, W4-W8
remain `planned`, and no research status changed.
