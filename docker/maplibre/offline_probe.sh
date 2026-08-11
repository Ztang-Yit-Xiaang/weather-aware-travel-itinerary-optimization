#!/bin/sh
set -eu

if wget -T 3 -q -O - https://example.com >/dev/null 2>&1; then
    echo "client_external_egress_available" >&2
    exit 1
fi

test "$(wget -q -O - http://map-service/healthz)" = "ok"
wget -q -O /tmp/style.json http://map-service/styles/protomaps-light.json
grep -q "pmtiles://http://127.0.0.1:8080/data/california-coast-v1.pmtiles" /tmp/style.json

printf "GET /data/california-coast-v1.pmtiles HTTP/1.1\r\nHost: map-service\r\nOrigin: http://127.0.0.1:8127\r\nRange: bytes=0-1023\r\nConnection: close\r\n\r\n" \
    | nc map-service 80 > /tmp/range.response
grep -q "HTTP/1.1 206 Partial Content" /tmp/range.response
grep -qi "Content-Range: bytes 0-1023/796263584" /tmp/range.response
grep -qi "Content-Length: 1024" /tmp/range.response

echo "offline_local_checks=passed"
