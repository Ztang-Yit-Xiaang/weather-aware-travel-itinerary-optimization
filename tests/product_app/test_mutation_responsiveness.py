"""Controlled endpoint I/O pauses must not starve the recovery event loop."""

import asyncio
import threading

import pytest

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.runtime import ProductRuntime
from itinerary_system.product_app.security import ProductSecurityBoundary
from itinerary_system.product_app.workspace import WorkspaceError


@pytest.mark.parametrize("endpoint_name", ["add_operation", "draft_impact_preview", "mutation_status"])
def test_mutation_io_yields_for_recovery_tasks(tmp_path, monkeypatch, endpoint_name):
    # No app data/provider is loaded: exercise the actual endpoint with controlled I/O.
    monkeypatch.setattr(ProductRuntime, "initialize", lambda self: None)
    app = create_product_app(repository_root=tmp_path, state_root=tmp_path / "state")
    runtime = app.state.product_runtime
    endpoint = next(route.endpoint for route in app.routes if route.name == endpoint_name)
    entered, heartbeat = threading.Event(), threading.Event()
    released = []

    def paused_service():
        entered.set()
        released.append(heartbeat.wait(0.5))
        raise WorkspaceError("controlled_io_complete", 409)

    monkeypatch.setattr(runtime, "require_service", paused_service)
    payload = {"expected_revision": 0}
    if endpoint_name == "add_operation":
        payload.update(type="keep_stop", target="test_stop", parameters={})

    async def read_json(self, request):
        return payload

    # The endpoint retains its real body validators; replace only transport parsing.
    monkeypatch.setattr(ProductSecurityBoundary, "read_bounded_json", read_json)

    async def journey():
        kwargs = {"session_id": "session_" + "a" * 32, "x_session_token": "synthetic-token"}
        if endpoint_name == "mutation_status":
            kwargs["mutation_id"] = "mutation_" + "b" * 32
        else:
            kwargs["request"] = object()
        task = asyncio.create_task(endpoint(**kwargs))
        while not entered.is_set() and not task.done():
            await asyncio.sleep(0.001)
        heartbeat.set()
        with pytest.raises(WorkspaceError, match="controlled_io_complete"):
            await task

    asyncio.run(journey())
    assert released == [True], "Blocking mutation I/O starved other recovery tasks"
