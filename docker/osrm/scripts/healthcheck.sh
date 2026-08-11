#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

PORT=${OSRM_PORT:-5000}
COORDS=${OSRM_TEST_COORDS:--122.4783,37.8199;-122.1697,37.4275}
URL="http://127.0.0.1:${PORT}/route/v1/driving/${COORDS}?overview=false"
BODY=$(curl --fail --silent --show-error "$URL")
printf '%s' "$BODY" | grep -q '"code":"Ok"'
echo "PASSED local OSRM healthcheck: $URL"

