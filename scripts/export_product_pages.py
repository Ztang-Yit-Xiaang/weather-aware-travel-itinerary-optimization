"""Export the current interface only after the matching hosted backend is ready."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itinerary_system.product_app.hosted import _https_origin, public_supabase_key
from itinerary_system.product_app.runtime import PRODUCT_ID, product_build_id


def export_pages(
    *, root: Path, output: Path, api_origin: str, supabase_url: str, publishable_key: str, opener=None
) -> dict:
    api = _https_origin(api_origin, "hosted_api_origin_invalid")
    auth = _https_origin(supabase_url, "supabase_url_invalid")
    if not urllib.parse.urlsplit(auth).hostname.endswith(".supabase.co"):
        raise ValueError("supabase_url_invalid")
    key = public_supabase_key(publishable_key)
    build = product_build_id(root)
    request = urllib.request.Request(api + "/healthz", headers={"Accept": "application/json"})

    # Never follow a redirect to another origin during a release check.
    class RejectRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args, **_kwargs):
            raise ValueError("backend_health_redirect_rejected")

    with (opener or urllib.request.build_opener(RejectRedirect()).open)(request, timeout=20) as response:
        raw = response.read(65_537)
        if response.status != 200 or len(raw) > 65_536:
            raise ValueError("backend_health_unavailable")
        health = json.loads(raw)
    if not isinstance(health, dict) or any(
        health.get(k) != v
        for k, v in {
            "status": "ready",
            "product_id": PRODUCT_ID,
            "build_id": build,
        }.items()
    ):
        raise ValueError("matching_backend_build_required")
    if output.exists() and any(output.iterdir()):
        raise ValueError("export_directory_must_be_empty")
    static = root / "src/itinerary_system/product_app/static"
    shutil.copytree(static, output / "static")
    config = {"api_origin": api, "supabase_url": auth, "supabase_publishable_key": key, "static_frontend": True}
    (output / "static/js/deployment.js").write_text(
        "window.PRODUCT_DEPLOYMENT = " + json.dumps(config) + ";\n", encoding="utf-8"
    )
    html = (static / "index.html").read_text(encoding="utf-8")
    html = html.replace("__PRODUCT_BUILD_ID__", build).replace('"/static/', '"./static/')
    policy = (
        f"default-src 'self'; script-src 'self' {api}; style-src 'self' {api}; "
        f"connect-src 'self' {api} {auth} https://tile.openstreetmap.org; "
        f"img-src 'self' {api} data: blob: https://tile.openstreetmap.org https://www.nps.gov; "
        f"font-src 'self' {api}; worker-src 'self' blob:; object-src 'none'; base-uri 'none'; form-action 'self'"
    )
    html = html.replace(
        "<head>",
        '<head>\n<meta name="referrer" content="strict-origin-when-cross-origin">\n'
        + '<meta http-equiv="Content-Security-Policy" content="'
        + policy
        + '">',
    )
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "static/index.html").unlink()
    (output / ".nojekyll").touch()
    receipt = {
        "schema_version": "hosted-pages-release-v1",
        "product_id": PRODUCT_ID,
        "build_id": build,
        "api_origin": api,
        "backend_health": "ready",
    }
    (output / "release.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-origin", required=True)
    parser.add_argument("--supabase-url", required=True)
    parser.add_argument("--publishable-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = export_pages(
            root=Path(__file__).resolve().parents[1],
            output=args.output,
            api_origin=args.api_origin,
            supabase_url=args.supabase_url,
            publishable_key=args.publishable_key,
        )
    except Exception:
        print(
            "Export stopped: public settings, an empty output directory, and a matching ready backend are required.",
            file=sys.stderr,
        )
        return 3
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
