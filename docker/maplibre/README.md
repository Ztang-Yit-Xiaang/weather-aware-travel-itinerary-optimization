# Local MapLibre and PMTiles Service

This is the primary local map boundary. It uses open-source MapLibre GL JS and the PMTiles browser protocol with a bounded, locally served OpenStreetMap-derived Protomaps basemap. Mapbox Atlas remains an optional, explicitly selected backup under `docker/atlas/`; it is never an automatic fallback.

## Required external assets

Keep the asset package outside Git and use this exact layout:

```text
MAP_ASSET_ROOT/
  maplibre/maplibre-gl.js
  maplibre/maplibre-gl.css
  pmtiles/pmtiles.js
  styles/protomaps-light.json
  data/california-coast-v1.pmtiles
  sprites/...
  fonts/...
  licenses/...
  provenance.json
```

The reviewed baseline pins MapLibre GL JS 5.24.0, PMTiles JS 4.4.1, `@protomaps/basemaps` 5.7.2, and PMTiles CLI 1.30.0. Preserve each distribution's license notice. The map data is an ODbL Produced Work and the product must visibly show `© OpenStreetMap contributors` linked to `https://www.openstreetmap.org/copyright`. Do not download or cache `tile.openstreetmap.org` for offline use. Every font stack used by a finalized symbol layer must include its local `0-255.pbf` glyph range and be listed in the closed asset manifest.

Create a bounded California Coast archive from a retained Protomaps daily build, recording the source build before extraction:

```powershell
pmtiles extract <retained-build-url> california-coast-v1.pmtiles --bbox=-123.0,33.5,-117.5,38.5 --maxzoom=15
pmtiles verify california-coast-v1.pmtiles
pmtiles show california-coast-v1.pmtiles --header-json
pmtiles show california-coast-v1.pmtiles --metadata
Get-FileHash california-coast-v1.pmtiles -Algorithm SHA256
```

Retain the licenses for all included data, fonts, sprites, and software. Generate the closed provenance manifest from the staged files and the pinned PMTiles CLI, then validate it:

Before starting the service, cryptographically bind the manifest to the archive and every local asset, validate the PMTiles v3 header/bounds/zoom, and prove that the archive covers every registered parent/child route and stop:

```powershell
python scripts\finalize_local_map_assets.py --asset-root "C:\path\to\itinerary-map-assets" --pmtiles-cli "C:\path\to\pmtiles.exe"
python scripts\validate_local_map_assets.py --asset-root "C:\path\to\itinerary-map-assets" --pmtiles-cli "C:\path\to\pmtiles.exe" --output "C:\path\to\itinerary-map-assets\preflight.json"
```

The validator invokes the pinned CLI's `show --header-json`, `show --metadata`, and `verify` commands and requires their actual output hashes to match `provenance.json`. It also validates every registered workspace's route/stop coverage. G1 cannot pass unless this command succeeds against the real asset package. A range response alone is not content or coverage evidence.

## Start and verify

Copy `.env.example` to the ignored `.env`, resolve the Nginx image to a platform-specific repository digest, and set the external asset root. Then:

```powershell
docker compose --env-file docker\maplibre\.env -f docker\maplibre\docker-compose.yml config
docker compose --env-file docker\maplibre\.env -f docker\maplibre\docker-compose.yml up -d
curl.exe -sS http://127.0.0.1:8080/healthz
curl.exe -sS -H "Origin: http://127.0.0.1:8127" http://127.0.0.1:8080/styles/protomaps-light.json
curl.exe -sS -D - -o NUL -H "Origin: http://127.0.0.1:8127" -H "Range: bytes=0-1023" http://127.0.0.1:8080/data/california-coast-v1.pmtiles
```

The range request must return HTTP 206, `Accept-Ranges: bytes`, the exact allowed origin, and a valid `Content-Range`. Repeat the service, asset, and range checks with the internet disconnected before G1 can pass. The real geographic browser renderer remains W2 work.

For a reproducible no-egress replay without changing the host's network, attach the map container and a fresh Nginx test client to a temporary Docker network created with `--internal`, mount `offline_probe.sh` read-only into the client, and run it with `/bin/sh`. The probe requires local health, style containment, and PMTiles 206 headers while both containers have no external route. Restore the normal Compose service afterward.

Primary references: [MapLibre GL JS](https://maplibre.org/maplibre-gl-js/docs), [PMTiles with MapLibre](https://docs.protomaps.com/pmtiles/maplibre), [Protomaps basemap licensing](https://github.com/protomaps/basemaps), and [OpenStreetMap attribution](https://www.openstreetmap.org/copyright/attribution-guide/).
