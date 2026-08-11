from __future__ import annotations

import pytest

from itinerary_system.product_app.models import ComponentHealthV1
from itinerary_system.product_app.runtime import ProductRuntime


@pytest.fixture(autouse=True)
def deterministic_product_routing_health(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep product tests independent of a developer's local OSRM process.

    RuntimeRoutingService has focused transport/contract tests.  Tests that
    construct the whole FastAPI application should not wait on or infer
    readiness from port 5000 unless they explicitly exercise that boundary.
    """

    monkeypatch.setattr(
        ProductRuntime,
        "_probe_routing",
        lambda self: ComponentHealthV1(
            "routing", "ready", False, "runtime_osrm_ready"
        ),
    )
