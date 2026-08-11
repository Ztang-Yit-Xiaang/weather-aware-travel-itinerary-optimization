# Local, Pinned OSRM

This directory provides the publication-oriented local routing path required by ROUTE-003. It follows the official OSRM Docker MLD sequence: `osrm-extract`, `osrm-partition`, `osrm-customize`, then `osrm-routed --algorithm mld`.

## Safety and provenance rules

- Never use `latest` or a mutable image tag. `OSRM_IMAGE` must use `registry@sha256:digest` form.
- Never accept an unverified map extract. Record and check `OSM_PBF_SHA256` before preprocessing.
- Do not commit `.osm.pbf`, `.osrm*`, or generated graph files; `data/.gitignore` excludes them.
- The bundled `/opt/car.lua` belongs to the digest-pinned image, so the profile and engine version move together.
- The service binds only to `127.0.0.1` by default.
- Keep `data/source-provenance.json` with the run evidence, but do not treat it as sufficient until its source URL, checksum, acquisition terms, and image digest have been reviewed.

## Setup

1. Install Docker with Compose support.
2. Copy `.env.example` to `.env`.
3. Replace the image digest, extract URL, and extract checksum with approved values.
4. Run `sh scripts/download_extract.sh`, or place an independently verified `${OSRM_DATASET}.osm.pbf` in `data/`.
5. Run `sh scripts/preprocess.sh`.
6. Start the service with `docker compose up -d osrm`.
7. Run `sh scripts/healthcheck.sh`.
8. Build project route evidence:

   ```bash
   python scripts/build_road_route_cache.py \
     --output-dir results/outputs \
     --cache-dir results/cache \
     --fetch-missing \
     --osrm-base-url http://127.0.0.1:5000 \
     --require-complete
   ```

9. Copy `source-provenance.example.json` to a run-evidence location and replace every placeholder with the reviewed image digest, extract URL/checksum, routing profile, and license/terms.
10. Freeze and require the evidence bundle:

   ```bash
   python scripts/freeze_route_evidence_bundle.py \
     --output-dir results/outputs \
     --provider-provenance path/to/source-provenance.json \
     --expected-request-count 42 \
     --require-publication-ready
   ```

11. Run strict Phase 0 validation. A healthy server alone does not close E2; every requested leg must be road validated, endpoint-snap validated, fallback free, fresh, provenance-backed, and consumed consistently by optimizer and evaluator.

The OSRM image and preprocessing commands are based on the official Project OSRM quick start: https://github.com/Project-OSRM/osrm-backend

