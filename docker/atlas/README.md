# Local Mapbox Atlas Asset Service

This boundary serves a separately licensed Mapbox Atlas v3 installation to the local Itinerary Repair Copilot. It does not acquire Atlas, contain licensed data, or store an Atlas token or browser license. The service binds only to `127.0.0.1:8080` and accepts browser requests only from the exact configured product origin.

## Required external inputs

- A Mapbox Atlas entitlement and installer token with `atlas:read` for acquisition.
- A Docker Desktop Linux engine with Docker Compose.
- At least 5 GB free for the Atlas development data and deployment overhead.
- An installer-produced asset directory outside this repository containing:

  ```text
  data/
    styles/
    ...
  mapbox-gl-js/
    mapbox-gl.js
    mapbox-gl.css
    ...
  ```

The acquisition token is installer-only. Do not add `MAPBOX_ATLAS_TOKEN` to `.env`, Compose, Nginx, application responses, logs, or Git. Configure the separate read-only browser license through the product application runtime when W2 consumes the map configuration API.

## Resolve and pin the Nginx image

The Compose file refuses to start without `ATLAS_NGINX_IMAGE`. Resolve the image only while the Docker Linux engine is running, review the resulting platform image, and record its immutable repository digest:

```powershell
docker pull nginx:stable-alpine
docker image inspect nginx:stable-alpine --format '{{join .RepoDigests "`n"}}'
```

Select the reviewed `nginx@sha256:...` entry for the active Linux platform. Copy `.env.example` to the ignored `.env`, replace the placeholder with that complete digest, and set `ATLAS_ASSET_ROOT` to the absolute external installer directory. Do not commit `.env`. A mutable tag is acceptable only for resolving a digest; it is never accepted in the runtime configuration.

## Validate and start

From the repository root:

```powershell
docker compose --env-file docker/atlas/.env -f docker/atlas/docker-compose.yml config
docker compose --env-file docker/atlas/.env -f docker/atlas/docker-compose.yml up -d
docker compose --env-file docker/atlas/.env -f docker/atlas/docker-compose.yml ps
```

The bind mount is read-only. Nginx itself uses a read-only root filesystem with temporary files limited to container `tmpfs` mounts. The local development endpoint is fixed at `http://127.0.0.1:8080`; do not publish it on all interfaces.

## Live preflight

Verify the service and local Atlas runtime before G1 can pass:

```powershell
curl.exe --fail --silent --show-error http://127.0.0.1:8080/healthz
curl.exe --fail --silent --show-error --output NUL http://127.0.0.1:8080/mapbox-gl-js/mapbox-gl.js
curl.exe --fail --silent --show-error --output NUL http://127.0.0.1:8080/mapbox-gl-js/mapbox-gl.css
curl.exe --fail --silent --show-error --output NUL http://127.0.0.1:8080/data/styles/mapbox/streets-v12.json
```

Find the PMTiles file referenced by the installed Streets style, convert its path below `/data` into an HTTP path, and make a bounded range request:

```powershell
curl.exe --silent --show-error --dump-header - --output NUL --range 0-1023 http://127.0.0.1:8080/data/REPLACE_WITH_STYLE_PM_TILES_PATH.pmtiles
```

The range request must report all of the following:

- `HTTP/1.1 206 Partial Content`
- `Accept-Ranges: bytes`
- `Content-Range: bytes 0-1023/...`
- A bounded `Content-Length`

HTTP 200 for that range request fails Atlas readiness. Nginx static-file serving generates `Content-Range` and `Content-Length`; the configuration exposes those headers and disables gzip so byte offsets remain meaningful.

Then disconnect internet access while leaving FastAPI and this container running and repeat the health, GL JS/CSS, style, and PMTiles range checks. The local checks must still pass. Record only versions, response codes, response headers, and file hashes; never record credential values or licensed asset contents.

## CORS and licensed-data boundaries

- `PRODUCT_APP_ORIGIN` defaults to the exact `http://127.0.0.1:8127` origin. Change it only to the product runtime's validated loopback origin.
- Requests receive no wildcard CORS policy. `Range` is the only non-simple request header allowed, and range-related response headers are explicitly exposed.
- Licensed Atlas assets must remain outside Git. `docker/atlas/assets/`, local `.env`, and local license files are ignored as a defense in depth measure.
- This service is not a public deployment and must not be exposed through port forwarding, a reverse proxy, or a non-loopback bind.

## Stop and remove the container

```powershell
docker compose --env-file docker/atlas/.env -f docker/atlas/docker-compose.yml down
```

This does not alter or remove the external Atlas asset directory.
