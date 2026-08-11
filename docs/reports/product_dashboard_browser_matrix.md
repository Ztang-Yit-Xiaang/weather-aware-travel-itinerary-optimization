# Product Dashboard Browser Matrix

**Artifact:** `runs/e3ux-weather-repair-demo-v6/dashboard_product/index.html`  
**Date:** 2026-07-29  
**Surface:** Codex in-app browser through the repository HTTP server

| Viewport | Document width | Clipped controls | Map placement/height | Result |
|---|---:|---:|---|---|
| 1440×900 | 1425px | 0 | Three-region layout; top 575px; 540px high | Pass |
| 1024×900 | 1009px | 0 | After evidence; top 4262px; 260px high | Pass |
| 768×900 | 753px | 0 | After evidence; top 4711px; 260px high | Pass |
| 430×844 | 415px | 0 | After evidence; top 5301px; 260px high | Pass |
| 390×844 | 375px | 0 | After evidence; top 5636px; 260px high | Pass |
| 360×844 | 345px | 0 | After evidence; top 5944px; 260px high | Pass |

At every width the document matched the scrollbar-adjusted viewport width, 21
visible interactive controls were inside the viewport, the document was
complete, the map had seven Leaflet panes, and no image remained pending.
Product-page console warnings/errors were empty.

## Read-only flow evidence

- Selecting day 7 preserved `aria-current`, changed the map label to Day 7, and
  announced the two route markers and synthetic affected-day weather evidence.
- Selecting a day also restyled original/repaired route segments: selected
  segments use full opacity and greater weight while other-day segments are
  subdued; affected segments retain the warning color.
- Switching to research mode preserved day 7 and exposed the certificate,
  canonical hashes, method lineage, and
  `complete_candidate_limit_exceeded:50000`.
- No “Accept repair” control appeared.
- Returning to customer mode and activating Review evidence moved focus to the
  certificate/evidence heading.
- Visible focus measured as a solid blue outline with offset.
- Every visible button, summary, and Leaflet zoom control measured at least
  44×44px.

## Browser-found defect and correction

The first product export relied on the remote Leaflet stylesheet. When that CSS
was unavailable, Leaflet panes entered normal flow and expanded the page.
Product-owned essential Leaflet layout/control CSS now preserves the map even
when the remote stylesheet is absent. The final six-width matrix above is from
the corrected v6 artifact.
