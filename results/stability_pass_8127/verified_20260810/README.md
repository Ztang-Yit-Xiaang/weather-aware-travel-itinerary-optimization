# Verified stability evidence — 2026-08-10

This directory is the authoritative post-fix browser evidence set. It supersedes
the older `results/stability_pass_8127/stability_pass_report.json`, which was
captured before the map-readiness and layout corrections.

The browser marked a map ready only after MapLibre emitted `idle`,
`areTilesLoaded()` returned true, and the canvas covered at least 95% of the
visible map container. The exact measurements and timestamps are in
`stability_records.json`.

Verified results:

- fresh load, reload, Compare → Trip, and Copilot open/close retained complete
  map rendering;
- 1440 → 1024 → 1440 resizing retained complete map rendering;
- measured map coverage ranged from 99.789% to 99.969%;
- the main desktop Trip map rendered at 629.6px high at 1440×900 and 558.4px
  inside a 560px bounded stage at 1024×768;
- Compare opened with Recommended repair selected and two ready map panes;
- the text-route capture exposes day, stop order, travel legs and duration,
  disruption, original/proposed route, booking impact, and eligibility;
- console errors and recorded network failures were empty;
- all PNGs are full-size viewport captures; the poster is a derived summary and
  is not a replacement for those files.

The poster was assembled deterministically from six named screenshots by
`results/stability_pass_8127/build_verified_poster.py`. An independent agent
verified exact source-to-poster pixel correspondence and found no invented UI.

Known product limitation: the verified demo is a coherent northbound
Los Angeles-to-San Francisco itinerary, not the preferred future seven-day
San Francisco-to-Los Angeles mockup fixture. Synthetic provenance remains
available in advanced Evidence; it is not presented as a user-facing trip fact.
