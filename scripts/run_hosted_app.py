"""Start the explicitly configured Railway backend; never load local secrets."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

from itinerary_system.product_app.config import ProductConfigError
from itinerary_system.product_app.hosted import HostedSettings, create_hosted_app


def main() -> int:
    try:
        port = int(os.environ.get("PORT", "8080"))
        if not 1 <= port <= 65535:
            raise ProductConfigError("hosted_port_invalid")
        if os.environ.get("WEB_CONCURRENCY", "1") != "1":
            raise ProductConfigError("hosted_single_worker_required")
        settings = HostedSettings.from_environment()
        if "--check" in sys.argv:
            print("Hosted configuration is syntactically valid; no resources or provider calls were made.")
            return 0
        app = create_hosted_app(settings=settings)
    except ProductConfigError as exc:
        print(f"Hosted startup stopped: {exc}.", file=sys.stderr)
        return 3
    except Exception:
        print("Hosted startup stopped: database, migration, or runtime configuration unavailable.", file=sys.stderr)
        return 3
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, proxy_headers=False, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
