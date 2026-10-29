from __future__ import annotations

import gc
from collections.abc import Callable, Iterator

import pytest
from fastapi.dependencies import models as dependency_models

from itinerary_system.product_app.models import ComponentHealthV1
from itinerary_system.product_app.runtime import ProductRuntime


@pytest.fixture(autouse=True)
def fastapi_dependency_cache_cleanup() -> Iterator[Callable[[], None]]:
    """Release closed test apps without changing any running app's artifacts."""

    def clear() -> None:
        # FastAPI's classification keys retain handler closures and their runtimes.
        # These are derived booleans, not app state; live handlers recompute them.
        for name in (
            "_is_gen_callable_cached",
            "_is_async_gen_callable_cached",
            "_is_coroutine_callable_cached",
        ):
            cache_clear = getattr(getattr(dependency_models, name, None), "cache_clear", None)
            if callable(cache_clear):
                cache_clear()
        gc.collect()

    yield clear
    clear()


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
        lambda self: ComponentHealthV1("routing", "ready", False, "runtime_osrm_ready"),
    )
