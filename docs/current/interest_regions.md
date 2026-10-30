# Persistent interest regions

The map opens the interest layer automatically and refreshes it after a day or stop selection. Explicitly hiding the layer is remembered. Radius, stated preference, model mass, and park search-area choice are saved per trip in the browser. Derived estimates are requested again against the current displayed plan; old geometry is not treated as current evidence.

The same browser profile and app address resume the saved workspace across tab/browser closure. The browser stores the workspace's existing session credential and scopes unfinished-action journals and unsent text to that session. Pending edits retain the existing confirmation/revision checks. Reopening does not automatically accept a plan or replay an unknown action.

Workspace authentication now expires after 90 days without activity. Saved session snapshots are retained for 365 days without activity; this is separate from the existing 30-day conversation retention. Clearing site data, using a private browser profile, changing the app's address, or deleting the server state directory prevents automatic resume.

## Place coverage and search centers

The local catalog includes the original POIs, California State Parks entries, official Presidio places, and a cached, attributed OpenStreetMap snapshot around all eight stops of the default seven-day itinerary. No live third-party search is required to reopen the layer. The snapshot is dated, and does not certify current opening conditions, public access or route feasibility.

At a 2 km straight-line radius, the 2026-10-29 verification found 18 places around Golden Gate Bridge, 66 around Stanford Main Quad, 14 around Yosemite Valley, 16 around Giant Forest, 4 around Grant Grove, 3 around Bixby, 72 around Stearns Wharf, and 12 around Griffith Observatory. Counts describe this catalog's locations, not completeness of all nearby places.

Broad park coordinates are unsuitable for a small visitor search. For those specific park stop IDs, the default center is an explicitly labelled visitor area using published NPS coordinates. The Search area selector retains an Original itinerary pin option. Accepted itinerary coordinates and routes are unchanged.

- [Yosemite visitor information](https://www.nps.gov/yose/planyourvisit/basicinfo.htm)
- [Sequoia and Kings Canyon visitor centers](https://www.nps.gov/seki/planyourvisit/visitorcenters.htm)
- [Grant Grove](https://www.nps.gov/seki/grant-grove.htm)
- [OpenStreetMap attribution and license](https://www.openstreetmap.org/copyright)

Source records, retrieval times and hashes are in `configs/interest_regions/itinerary_places.json` and `exploration_manifest.json`. `scripts/prepare_interest_region_catalog.py` rebuilds the merged, deduplicated catalog from these checked-in inputs. Runtime build identity includes the interest configuration files.

## Online lookup and durable place cache

The interactive interest endpoint now queries OpenStreetMap's Overpass API around the displayed search center and radius. Successful results are stored under `<state-root>/nearby_places/`, independently of the browser and the derived probability surface. The cache key includes the geographic query and its categories, so another day or radius cannot inherit the wrong area's results. Model preference changes reuse the same source data.

Fresh saved responses are reused for seven days. A cache miss or an expired response attempts one online lookup; a timeout or outage falls back to the last good response, and then to the bundled catalog if no online response has ever been saved for that search. Older data is labelled. Valid empty online results are cached and distinguished from failed requests. Partial responses, malformed coordinates and oversized results do not replace good cached data.

The map shows Online places, Saved online places, Older saved online places, or Saved place catalog, with the original fetch time where available. Refresh online places in the place panel requests an update; Refresh estimates recalculates interest using the normal cache policy. Refreshing places does not stage a stop or establish road access, opening status or itinerary eligibility.

Requests are serialized across local processes, use a socket timeout and response-size limits, and wait at least ten seconds between public API requests. Failures pause requests for five minutes; provider Retry-After values are respected up to one hour. A broad query exceeding the result limit retains saved data and allows a smaller search after the normal request interval. Repeated clicks and restarts do not bypass these limits. The source response is saved atomically before it is reported as persisted.

The public endpoint is used for this bounded local research prototype. Its [usage policy](https://dev.overpass-api.de/overpass-doc/en/preface/commons.html) calls for restrained shared-resource use and a separate instance for a public application's backend. Attribution and source-object links remain visible in the place details.

## Local service and hosted deployments

The existing local launcher is `results/repairs/copilot_user_recovery_20261021/run_app.py`. It resumes that workspace's existing `user_state` directory, starts its recorded stopped map container, and can recreate the old auto-removed container using the same pinned image, loopback port, read-only mounts and no-download policy. The map is stopped when the app exits and now remains available for the next launch. Docker Desktop must be running.

The generic launcher also supports `PRODUCT_STATE_ROOT` (or the existing explicit `--state-root`) for choosing a durable server state directory. Explicit arguments take precedence. Restarting against a different empty directory creates different state.

The source, catalog, browser preferences and saved server state provide local persistence. The local server must still be launched before its localhost address can be opened. A separate hosted launcher now prepares Railway/FastAPI, private Supabase PostgreSQL state and owner sign-in, ORS routing, and a gated GitHub Pages frontend. See the [deployment guide](../deployment/railway_supabase_ors.md). No cloud deployment has been completed. Hosted interest exploration uses labelled saved places; the public Overpass lookup remains disabled there until a suitable discovery service is supplied.

## Verification

The 2026-10-29 repair record is `results/repairs/interest_durable_20261029/`. It contains source diffs, state backups, source provenance, per-stop API coverage, automated test results, restart receipts and actual Day 2/Day 3 screenshots. Saved accepted plans, existing drafts and conversations are preserved; selecting days changes only the expected session selection metadata.

The density and 90% model intervals remain uncalibrated interest estimates. The 50/80/95% boundaries describe integrated model interest mass. Darkest must-go markers express confirmed constraints, not a model probability of 100%.
