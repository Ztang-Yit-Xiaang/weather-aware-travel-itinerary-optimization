"""Adversarially verify the loopback MapLibre/PMTiles HTTP service."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote


class LiveMapVerificationError(ValueError):
    pass


def request(
    url: str,
    *,
    method: str = "GET",
    origin: str | None = None,
    byte_range: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    headers = {}
    if origin is not None:
        headers["Origin"] = origin
    if byte_range is not None:
        headers["Range"] = byte_range
    prepared = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(prepared, timeout=5) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()


def header(headers: dict[str, str], name: str) -> str | None:
    return next((value for key, value in headers.items() if key.lower() == name.lower()), None)


def verify(asset_root: Path, base_url: str, allowed_origin: str) -> dict[str, Any]:
    root = asset_root.resolve()
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    status, headers, body = request(f"{base_url}/healthz", origin=allowed_origin)
    if status != 200 or body != b"ok\n":
        raise LiveMapVerificationError("healthz_failed")
    if header(headers, "Access-Control-Allow-Origin") != allowed_origin:
        raise LiveMapVerificationError("allowed_origin_missing")

    mime_types = {
        ".js": "application/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".png": "image/png",
        ".pbf": "application/octet-stream",
        ".txt": "text/plain",
        ".md": "application/octet-stream",
    }
    verified_assets = 0
    for asset in provenance["assets"]:
        relative = asset["path"]
        url = f"{base_url}/{quote(relative, safe='/@')}"
        status, headers, body = request(url, origin=allowed_origin)
        if status != 200:
            raise LiveMapVerificationError(f"asset_status_failed:{relative}:{status}")
        if hashlib.sha256(body).hexdigest() != asset["sha256"]:
            raise LiveMapVerificationError(f"served_asset_hash_mismatch:{relative}")
        if header(headers, "Access-Control-Allow-Origin") != allowed_origin:
            raise LiveMapVerificationError(f"asset_cors_failed:{relative}")
        if header(headers, "X-Content-Type-Options") != "nosniff":
            raise LiveMapVerificationError(f"asset_nosniff_missing:{relative}")
        expected_mime = mime_types[Path(relative).suffix]
        if header(headers, "Content-Type") != expected_mime:
            raise LiveMapVerificationError(f"asset_mime_failed:{relative}")
        verified_assets += 1

    status, headers, body = request(f"{base_url}/provenance.json", origin=allowed_origin)
    if (
        status != 200
        or hashlib.sha256(body).hexdigest() != hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    ):
        raise LiveMapVerificationError("served_provenance_mismatch")

    archive_size = provenance["artifact_size"]
    archive_url = f"{base_url}{provenance['artifact_url']}"
    status, headers, body = request(
        archive_url,
        origin=allowed_origin,
        byte_range="bytes=0-1023",
    )
    expected_range = f"bytes 0-1023/{archive_size}"
    if (
        status != 206
        or len(body) != 1024
        or body[:8] != b"PMTiles\x03"
        or header(headers, "Content-Length") != "1024"
        or header(headers, "Accept-Ranges") != "bytes"
        or header(headers, "Content-Range") != expected_range
        or header(headers, "Access-Control-Allow-Origin") != allowed_origin
    ):
        raise LiveMapVerificationError("pmtiles_range_contract_failed")

    foreign_origin = "http://example.invalid"
    status, foreign_headers, _ = request(archive_url, origin=foreign_origin, byte_range="bytes=0-0")
    if status != 206 or header(foreign_headers, "Access-Control-Allow-Origin") == foreign_origin:
        raise LiveMapVerificationError("foreign_origin_readable")
    if header(foreign_headers, "Access-Control-Allow-Origin") == "*":
        raise LiveMapVerificationError("wildcard_cors_present")

    status, options_headers, _ = request(
        f"{base_url}/styles/protomaps-light.json",
        method="OPTIONS",
        origin=allowed_origin,
    )
    if status != 204 or header(options_headers, "Access-Control-Allow-Origin") != allowed_origin:
        raise LiveMapVerificationError("options_contract_failed")
    unsafe_targets = (
        "/healthz",
        "/styles/protomaps-light.json",
        "/provenance.json",
        provenance["artifact_url"],
    )
    for target in unsafe_targets:
        for method in ("POST", "PUT", "DELETE"):
            status, _, _ = request(f"{base_url}{target}", method=method)
            if status != 405:
                raise LiveMapVerificationError(
                    f"unsafe_method_allowed:{target}:{method}:{status}"
                )
    status, _, _ = request(
        archive_url,
        origin=allowed_origin,
        byte_range=f"bytes={archive_size}-{archive_size + 100}",
    )
    if status != 416:
        raise LiveMapVerificationError("unsatisfiable_range_not_rejected")
    for path in ("/%2e%2e/provenance.json", "/fonts/%2e%2e/provenance.json"):
        status, _, _ = request(f"{base_url}{path}")
        if status not in {400, 404}:
            raise LiveMapVerificationError(f"encoded_traversal_allowed:{status}")

    return {
        "schema_version": "live-map-http-verification-v1",
        "status": "passed",
        "base_url": base_url,
        "allowed_origin": allowed_origin,
        "asset_count": verified_assets,
        "archive_size": archive_size,
        "range_status": 206,
        "range_length": 1024,
        "content_range": expected_range,
        "foreign_origin_readable": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--origin", default="http://127.0.0.1:8127")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.asset_root, args.base_url.rstrip("/"), args.origin)
    except (LiveMapVerificationError, OSError, KeyError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
