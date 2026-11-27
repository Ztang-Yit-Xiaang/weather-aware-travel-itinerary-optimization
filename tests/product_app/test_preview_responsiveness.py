"""Event-loop responsiveness while a real compiler execution is in flight."""

import asyncio
import json
import threading
import time
from pathlib import Path

import httpx

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter

ROOT = Path(__file__).resolve().parents[2]
RUN = "california_coast_coherent_demo_v1"


def test_preview_does_not_block_an_independent_health_request(tmp_path, monkeypatch):
    app = create_product_app(
        repository_root=ROOT,
        registry_path=ROOT / "configs/product_app_registry.json",
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
        copilot=DeterministicCopilotAdapter(),
    )

    async def journey():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client,
        ):
            created = (await client.post("/api/sessions", json={"run_id": RUN})).json()
            session = created["session"]
            headers = {"X-Session-Token": created["mutation_token"]}
            url = f"/api/sessions/{session['session_id']}"
            staged = await client.post(
                f"{url}/draft/operations",
                headers=headers,
                json={
                    "expected_revision": 0,
                    "type": "replace_nearby",
                    "target": "bixby_creek_bridge_viewpoint",
                    "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
                    "source": "map",
                },
            )
            assert staged.status_code == 200
            compiler = app.state.product_service._draft_compilers[RUN]
            original = compiler.compile
            entered = threading.Event()
            release = threading.Event()

            def delayed(*args, **kwargs):
                entered.set()
                assert release.wait(5), "controlled compiler delay was not released"
                return original(*args, **kwargs)

            monkeypatch.setattr(compiler, "compile", delayed)
            preview_task = asyncio.create_task(
                client.post(f"{url}/preview", headers=headers, json={"expected_revision": 1})
            )
            try:
                compiler_entered_before_health = await asyncio.to_thread(entered.wait, 5)
                warm_started = time.monotonic()
                warm_health = await client.get("/api/health")
                warm_elapsed = time.monotonic() - warm_started
                started = time.monotonic()
                health = await client.get("/api/health")
                elapsed = time.monotonic() - started
                health_completed_while_compiler_paused = (
                    compiler_entered_before_health and not release.is_set() and not preview_task.done()
                )
            finally:
                release.set()
                preview = await preview_task
            (tmp_path / "exact_responsiveness_io.json").write_text(
                json.dumps(
                    {
                        "injected_fault": "local compiler held until health finishes, with the existing five-second safety bound; internal execution unchanged",
                        "warm_health_elapsed_seconds": warm_elapsed,
                        "warm_health_response": warm_health.json(),
                        "health_elapsed_seconds": elapsed,
                        "health_response": health.json(),
                        "preview_response": preview.json(),
                        "compiler_entered": entered.is_set(),
                        "compiler_entered_before_health": compiler_entered_before_health,
                        "health_completed_while_compiler_paused": health_completed_while_compiler_paused,
                        "latency_scope": "cached health after a real health/probe request; both requests finish before compiler release",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            assert compiler_entered_before_health
            assert health_completed_while_compiler_paused
            assert warm_health.status_code == 200
            assert health.status_code == 200
            assert preview.status_code == 200, preview.text
            assert preview.json()["proposal"]["state"] == "eligible"
            assert elapsed < 0.8, "cached health request waited while the compiler was paused"

    asyncio.run(journey())
