#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

if [ ! -f .env ]; then
  echo "Missing docker/osrm/.env; copy .env.example first." >&2
  exit 2
fi

set -a
. ./.env
set +a

case "${OSRM_IMAGE:-}" in
  *@sha256:REPLACE*|""|*:latest)
    echo "OSRM_IMAGE must use an approved immutable sha256 digest." >&2
    exit 2
    ;;
  *@sha256:*) ;;
  *)
    echo "OSRM_IMAGE must use registry@sha256:digest form." >&2
    exit 2
    ;;
esac

PBF="data/${OSRM_DATASET}.osm.pbf"
if [ ! -f "$PBF" ]; then
  echo "Missing $PBF; run scripts/download_extract.sh or place a checksum-verified extract there." >&2
  exit 2
fi

ACTUAL_SHA256=$(sha256sum "$PBF" | awk '{print $1}')
if [ "$ACTUAL_SHA256" != "$OSM_PBF_SHA256" ]; then
  echo "PBF checksum mismatch: expected $OSM_PBF_SHA256, got $ACTUAL_SHA256" >&2
  exit 1
fi

docker run --rm -t -v "$ROOT/data:/data" "$OSRM_IMAGE" \
  osrm-extract -p /opt/car.lua "/data/${OSRM_DATASET}.osm.pbf"
docker run --rm -t -v "$ROOT/data:/data" "$OSRM_IMAGE" \
  osrm-partition "/data/${OSRM_DATASET}.osrm"
docker run --rm -t -v "$ROOT/data:/data" "$OSRM_IMAGE" \
  osrm-customize "/data/${OSRM_DATASET}.osrm"

echo "Prepared MLD graph for ${OSRM_DATASET} with ${OSRM_IMAGE}"

