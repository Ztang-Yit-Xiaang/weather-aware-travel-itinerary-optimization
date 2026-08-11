# Itinerary Repair Copilot Mobile and PWA Audit

**Audit date:** 2026-08-03

**Audit role:** Independent read-only mobile/PWA specialist

**Scope:** Routes, responsive layouts, bottom navigation, map editing, keyboard
behavior, manifest/service worker, installability, and offline truthfulness.

**Gate verdict:** **BLOCKING — G6, G7, and G8 fail.**

## Method

The auditor inspected the route table, HTML/CSS/JavaScript, and exercised the
running app in disposable narrow viewports. Manifest and service-worker URLs were
requested directly. Desktop functionality was not counted as a mobile
replacement merely because it reflowed.

## Findings

| ID | Severity | Gates | Evidence | Required closure |
|---|---|---|---|---|
| MOB-001 | Critical | G6 | `/app/day/3` and `/app/copilot` resolve to the same generic schematic shell; dedicated mobile route composition is absent. | Add Overview, Day Details, and full-screen Copilot route states. |
| MOB-002 | Critical | G6/G8 | Responsive CSS hides itinerary/repair context instead of creating the approved mobile Overview. | Build issue, affected-day, repair, metrics, and bottom-nav modules. |
| MOB-003 | Critical | G6 | Mobile Map is neither geographic nor paired with a snap-point editing bottom sheet. | Implement geographic layers and keyboard-equivalent sheet actions. |
| MOB-004 | Critical | G6/G7 | No web manifest or service worker is present; both expected endpoints return 404. | Add reviewed icons, manifest, registration, update, and cache-policy tests. |
| MOB-005 | High | G6 | Fixed bottom navigation can cover content because safe-area/content padding is incomplete. | Reserve layout space and test short/tall/zoomed viewports. |
| MOB-006 | High | G6 | Bottom navigation information architecture does not match the required routes and lacks reliable current-state semantics. | Implement the specified five destinations and `aria-current`. |
| MOB-007 | High | G6 | Mobile toolbar actions use symbols without stable visible/accessibility labels. | Add labels/names and 44x44 px targets. |
| MOB-008 | High | G6/G8 | Real virtual-keyboard behavior for the Copilot composer is unverified. | Test keyboard-open reachability and safe-area positioning on target widths. |
| MOB-009 | Critical | G6/G7 | No truthful offline shell/backend/provider behavior exists. | Cache only non-sensitive shell assets; show explicit unavailable states. |
| MOB-010 | High | G8 | No current mobile/PWA browser suite or approved 430/390/360 baselines exist. | Add route journeys, installability, cache, overflow, and screenshot evidence. |

## Gate interpretation

G6 cannot pass by hiding desktop panels or by returning HTTP 200 for client
routes. G7 additionally requires clean-launch and backend/provider unavailable
recovery. G8 requires device-width journeys and user visual approval.
