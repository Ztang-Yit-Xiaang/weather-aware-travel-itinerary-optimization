#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

if [ ! -f .env ]; then
  echo "Missing docker/osrm/.env; copy .env.example and pin every REPLACE value." >&2
  exit 2
fi

set -a
. ./.env
set +a

case "${OSM_PBF_URL:-} ${OSM_PBF_SHA256:-}" in
  *REPLACE*|" ")
    echo "OSM_PBF_URL and OSM_PBF_SHA256 must be explicitly approved and pinned." >&2
    exit 2
    ;;
esac

mkdir -p data
PBF="data/${OSRM_DATASET}.osm.pbf"
TMP="${PBF}.partial"
curl --fail --location --retry 3 --output "$TMP" "$OSM_PBF_URL"
ACTUAL_SHA256=$(sha256sum "$TMP" | awk '{print $1}')
if [ "$ACTUAL_SHA256" != "$OSM_PBF_SHA256" ]; then
  echo "PBF checksum mismatch: expected $OSM_PBF_SHA256, got $ACTUAL_SHA256" >&2
  rm -f "$TMP"
  exit 1
fi
mv "$TMP" "$PBF"

cat > data/source-provenance.json <<EOF
{
  "dataset": "${OSRM_DATASET}",
  "source_url": "${OSM_PBF_URL}",
  "sha256": "${ACTUAL_SHA256}",
  "osrm_image": "${OSRM_IMAGE}",
  "profile": "/opt/car.lua",
  "algorithm": "mld"
}
EOF

echo "Verified $PBF"

