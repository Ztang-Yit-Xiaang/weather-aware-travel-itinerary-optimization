"""R22 local timing faults around real compiler/evaluator; no provider calls."""

import asyncio
import inspect
import json
import threading
import time
from pathlib import Path

import httpx
import pytest

from itinerary_system.product_app import api
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler

ROOT = Path(__file__).resolve().parents[2]
RUN = "california_coast_coherent_demo_v1"


@pytest.fixture
def app(tmp_path):
    return api.create_product_app(
        repository_root=ROOT, registry_path=ROOT / "configs/product_app_registry.json",
        state_root=tmp_path / "state", additional_allowed_authorities=("testserver",),
        copilot=DeterministicCopilotAdapter(),
    )


class CompilerPause:
    """Pause only timing; all optimizer and evaluator results remain real."""

    def __init__(self, monkeypatch):
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.entered = 0
        self.finished = 0
        self.inputs = []
        self.events = []
        original = FrozenDraftCompiler.compile

        def delayed(compiler, operations, **kwargs):
            with self.lock:
                self.entered += 1
                self.events.append({"event": "compiler_entered", "monotonic": time.monotonic(),
                    "thread": threading.current_thread().name, "entered": self.entered})
            try:
                released = self.release.wait(15)
                with self.lock:
                    self.events.append({"event": "controlled_pause_finished", "monotonic": time.monotonic(),
                        "thread": threading.current_thread().name, "released": released})
                assert released, "controlled compiler pause was not released"
                self.inputs.append([operation.as_dict() for operation in operations])
                return original(compiler, operations, **kwargs)
            finally:
                with self.lock:
                    self.finished += 1
                    self.events.append({"event": "compiler_finished", "monotonic": time.monotonic(),
                        "thread": threading.current_thread().name, "finished": self.finished})

        monkeypatch.setattr(FrozenDraftCompiler, "compile", delayed)

    def snapshot(self):
        with self.lock:
            return {"entered": self.entered, "finished": self.finished, "release_set": self.release.is_set(),
                "events": [dict(event) for event in self.events]}

    async def wait_for(self, field, count, *, timeout_seconds=15):
        started = time.monotonic()
        deadline = started + timeout_seconds
        while getattr(self, field) < count:
            current = time.monotonic()
            if current >= deadline:
                raise AssertionError(json.dumps({"failure": f"compiler {field} did not reach {count}",
                    "wait_started_monotonic": started, "deadline_monotonic": deadline,
                    "observed_monotonic": current, "elapsed_seconds": current - started,
                    "compiler_pause": self.snapshot()}, sort_keys=True))
            await asyncio.sleep(0.01)


async def staged_session(client):
    created = await client.post("/api/sessions", json={"run_id": RUN})
    assert created.status_code == 200, created.text
    body = created.json()
    url = f"/api/sessions/{body['session']['session_id']}"
    headers = {"X-Session-Token": body["mutation_token"]}
    staged = await client.post(f"{url}/draft/operations", headers=headers, json={
        "expected_revision": 0, "type": "replace_nearby", "target": "bixby_creek_bridge_viewpoint",
        "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"}, "source": "map",
    })
    assert staged.status_code == 200, staged.text
    return url, headers


def preview(client, session):
    url, headers = session
    return client.post(f"{url}/preview", headers=headers, json={"expected_revision": 1})


def record(tmp_path, **details):
    (tmp_path / "exact_concurrency_io.json").write_text(json.dumps({
        "fault_injection": "local compiler timing only; real internal compiler and independent evaluator",
        **details,
    }, indent=2), encoding="utf-8")


def test_two_running_previews_reject_third_without_blocking_health(app, tmp_path, monkeypatch):
    pause = CompilerPause(monkeypatch)

    async def journey():
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver",
        ) as client:
            sessions = [await staged_session(client) for _ in range(3)]
            tasks = [asyncio.create_task(preview(client, session)) for session in sessions[:2]]
            try:
                await pause.wait_for("entered", 2)
                started = time.monotonic()
                health = await client.get("/api/health")
                busy = await preview(client, sessions[2])
                elapsed = time.monotonic() - started
                assert health.status_code == 200
                assert busy.status_code == 503, busy.text
                assert busy.json() == {"detail": "preview_capacity_reached"}
                assert busy.headers["Retry-After"] == "1"
                assert elapsed < 0.8
                assert pause.entered == 2
            except BaseException as error:
                record(tmp_path, assertions_passed=False, error_type=type(error).__name__,
                    error=str(error), compiler_pause=pause.snapshot(),
                    request_targets=[url + "/preview" for url, _ in sessions],
                    request_input={"expected_revision": 1},
                    pending_preview_tasks=[{"done": task.done(), "cancelled": task.cancelled(),
                        "exception_type": type(task.exception()).__name__ if task.done() and not task.cancelled()
                            and task.exception() is not None else None,
                        "response_status": task.result().status_code if task.done() and not task.cancelled()
                            and task.exception() is None else None,
                        "response_body_utf8": task.result().text if task.done() and not task.cancelled()
                            and task.exception() is None else None} for task in tasks])
                raise
            finally:
                pause.release.set()
                results = await asyncio.gather(*tasks)
            assert all(result.status_code == 200 for result in results), [r.text for r in results]
            assert all(result.json()["proposal"]["state"] == "eligible" for result in results)
            after_capacity = await preview(client, sessions[2])
            assert after_capacity.status_code == 200, after_capacity.text
            record(tmp_path, health_elapsed_seconds=elapsed, health=health.json(), busy=busy.json(),
                   previews=[result.json() for result in results], after_capacity=after_capacity.json(),
                   compiler_pause=pause.snapshot())

    asyncio.run(journey())


def test_compiler_pause_timeout_reports_counts_and_monotonic_interval(monkeypatch):
    pause = CompilerPause(monkeypatch)
    with pytest.raises(AssertionError) as failure:
        asyncio.run(pause.wait_for("entered", 2, timeout_seconds=0))
    diagnostic = json.loads(str(failure.value))
    assert diagnostic["failure"] == "compiler entered did not reach 2"
    assert diagnostic["observed_monotonic"] >= diagnostic["deadline_monotonic"]
    assert diagnostic["elapsed_seconds"] >= 0
    assert diagnostic["compiler_pause"] == {"entered": 0, "finished": 0, "release_set": False, "events": []}


@pytest.mark.parametrize("preflight", ["runtime_readiness", "accepted_baseline", "executor_submission", "completed_baseline", "completed_publication"])
def test_preview_preflight_io_yields_to_other_event_loop_tasks(app, tmp_path, monkeypatch, preflight):
    """An I/O pause may delay this preview, never every coroutine on its loop."""
    entered, heartbeat = threading.Event(), threading.Event()
    observations = []

    async def journey():
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver",
        ) as client:
            session = await staged_session(client)
            runtime = app.state.product_runtime
            if preflight == "runtime_readiness":
                owner, name = runtime, "require_service"
            elif preflight == "executor_submission":
                endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/api/sessions/{session_id}/preview")
                owner, name = inspect.getclosurevars(endpoint).nonlocals["preview_executor"], "submit"
            elif preflight == "completed_publication":
                owner, name = app.state.product_service.workspace, "set_proposal"
            else:
                owner, name = app.state.product_service, "ensure_editable_baseline"
            original = getattr(owner, name)
            event_loop_thread = threading.get_ident()
            invocations = 0

            def observed_io(*args, **kwargs):
                nonlocal invocations
                invocations += 1
                if invocations == (2 if preflight == "completed_baseline" else 1):
                    entered.set()
                    started = time.monotonic()
                    yielded = heartbeat.wait(0.5)
                    observations.append({"preflight": preflight, "event_loop_thread": event_loop_thread,
                        "observed_thread": threading.get_ident(), "started_monotonic": started,
                        "finished_monotonic": time.monotonic(), "heartbeat_during_io": yielded})
                return original(*args, **kwargs)

            monkeypatch.setattr(owner, name, observed_io)

            async def concurrent_loop_task():
                while not entered.is_set():
                    await asyncio.sleep(0.005)
                heartbeat.set()

            observer = asyncio.create_task(concurrent_loop_task())
            try:
                response = await preview(client, session)
            finally:
                observer.cancel()
                await asyncio.gather(observer, return_exceptions=True)
            record(tmp_path, fault_injection="local preflight I/O timing only; original readiness, baseline, compiler and evaluator",
                input={"url": session[0] + "/preview", "body": {"expected_revision": 1}},
                response_status=response.status_code, response_body_utf8=response.text, observations=observations)
            assert response.status_code == 200, response.text
            assert response.json()["proposal"]["state"] == "eligible"
            assert len(observations) == 1
            assert observations[0]["heartbeat_during_io"] is True, observations

    asyncio.run(journey())


@pytest.mark.parametrize("interruption", ["cancel", "undo", "disconnect"])
def test_completed_preview_revalidates_before_publication(app, tmp_path, monkeypatch, interruption):
    entered, release, finished, disconnected = (threading.Event() for _ in range(4))
    service = app.state.product_service
    original = service.ensure_editable_baseline
    invocations = 0
    waits = []

    def delayed_completion(session):
        nonlocal invocations
        invocations += 1
        if invocations != 2:
            return original(session)
        entered.set()
        try:
            waits.append(release.wait(10))
            return original(session)
        finally:
            finished.set()

    async def observed_disconnect(request):
        return request.url.path.endswith("/preview") and disconnected.is_set()

    async def journey():
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver",
        ) as client:
            session = await staged_session(client)
            url, headers = session
            monkeypatch.setattr(service, "ensure_editable_baseline", delayed_completion)
            if interruption == "disconnect":
                monkeypatch.setattr(api.Request, "is_disconnected", observed_disconnect)
            task = asyncio.create_task(preview(client, session))
            mutation = None
            try:
                assert await asyncio.to_thread(entered.wait, 15), "completed preview did not reach publication preparation"
                assert not task.done(), "request published before the interruption could run"
                if interruption == "cancel":
                    task.cancel()
                    result = await asyncio.gather(task, return_exceptions=True)
                    assert isinstance(result[0], asyncio.CancelledError)
                elif interruption == "undo":
                    mutation = await client.post(f"{url}/draft/undo", headers=headers, json={"expected_revision": 1})
                    assert mutation.status_code == 200, mutation.text
                else:
                    disconnected.set()
            finally:
                release.set()
                results = await asyncio.gather(task, return_exceptions=True)
                assert await asyncio.to_thread(finished.wait, 15), "completed preparation did not settle"
            result = results[0]
            after = (await client.get(url, headers=headers)).json()
            record(tmp_path, fault_injection="local post-compiler baseline I/O pause only; real completed evaluation",
                interruption=interruption, request_input={"expected_revision": 1}, request_url=url + "/preview",
                response_status=result.status_code if isinstance(result, httpx.Response) else None,
                response_body_utf8=result.text if isinstance(result, httpx.Response) else None,
                exception_type=type(result).__name__ if isinstance(result, BaseException) else None,
                mutation_input={"expected_revision": 1} if mutation is not None else None,
                mutation_response=mutation.json() if mutation is not None else None,
                pause_released=waits, session_after=after)
            assert waits == [True]
            if interruption != "cancel":
                expected = (409, "stale_session_revision") if interruption == "undo" else (499, "preview_client_disconnected")
                assert result.status_code == expected[0], result.text
                assert result.json() == {"detail": expected[1]}
            assert after["session"]["proposal"] is None
            assert after["session"]["revision"] == (2 if interruption == "undo" else 1)
            assert len(after["session"]["draft"]) == (0 if interruption == "undo" else 1)

    asyncio.run(journey())


def test_draft_edit_during_execution_rejects_old_snapshot_without_publishing(app, tmp_path, monkeypatch):
    pause = CompilerPause(monkeypatch)

    async def journey():
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver",
        ) as client:
            session = await staged_session(client)
            url, headers = session
            before = (await client.get(url, headers=headers)).json()
            task = asyncio.create_task(preview(client, session))
            try:
                await pause.wait_for("entered", 1)
                undo = await client.post(f"{url}/draft/undo", headers=headers, json={"expected_revision": 1})
                assert undo.status_code == 200, undo.text
            finally:
                pause.release.set()
                result = await task
            after = (await client.get(url, headers=headers)).json()
            record(tmp_path, before=before, undo=undo.json(), preview=result.json(), after=after,
                   compiler_inputs=pause.inputs)
            assert result.status_code == 409, result.text
            assert result.json()["detail"] == "stale_session_revision"
            assert len(pause.inputs[0]) == 1, "worker must compile the immutable original draft"
            assert pause.inputs[0][0]["type"] == "replace_nearby"
            assert after["session"]["draft"] == []
            assert after["session"]["proposal"] is None
            assert after["session"]["revision"] == 2
            assert after["session"]["accepted_plan_id"] == before["session"]["accepted_plan_id"]

    asyncio.run(journey())


@pytest.mark.parametrize("abandonment", ["cancel", "timeout"])
def test_abandoned_requests_keep_running_slots_and_cannot_publish(app, tmp_path, monkeypatch, abandonment):
    pause = CompilerPause(monkeypatch)
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/api/sessions/{session_id}/preview")
    executor = inspect.getclosurevars(endpoint).nonlocals["preview_executor"]
    worker_futures = []
    phases = []
    observing = False

    def observe(owner, name):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            enabled = observing
            if enabled:
                phases.append({"phase":name, "edge":"start", "at":time.monotonic(), "thread":threading.get_ident()})
            try:
                value = original(*args, **kwargs)
                if name == "submit":
                    worker_futures.append(value)
                return value
            finally:
                if enabled:
                    phases.append({"phase":name, "edge":"finish", "at":time.monotonic(), "thread":threading.get_ident()})
        monkeypatch.setattr(owner, name, wrapped)

    runtime = app.state.product_runtime
    for owner, name in ((runtime, "require_current_build"), (runtime, "require_service"),
                        (app.state.product_service.workspace, "authenticate"),
                        (app.state.product_service, "ensure_editable_baseline"), (executor, "submit")):
        observe(owner, name)
    if abandonment == "timeout":
        monkeypatch.setattr(api, "PREVIEW_TIMEOUT_SECONDS", 0.15)

    async def journey():
        nonlocal observing
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver",
        ) as client:
            sessions = [await staged_session(client) for _ in range(3)]
            observing = True
            loop_thread = threading.get_ident()
            heartbeat = []
            async def observe_loop():
                previous = time.monotonic()
                while True:
                    await asyncio.sleep(0.01)
                    current = time.monotonic()
                    if current - previous > 0.1:
                        heartbeat.append({"at":current, "delay_seconds":current-previous})
                    previous = current
            observer = asyncio.create_task(observe_loop())
            tasks = [asyncio.create_task(preview(client, session)) for session in sessions[:2]]
            cancellation_issued = False
            try:
                await pause.wait_for("entered", 2)
                if abandonment == "cancel":
                    cancellation_issued = True
                    for task in tasks:
                        task.cancel()
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    assert all(isinstance(result, asyncio.CancelledError) for result in results)
                    outcomes = ["CancelledError" for _ in results]
                else:
                    results = await asyncio.gather(*tasks)
                    assert all(result.status_code == 504 for result in results)
                    outcomes = [result.json() for result in results]
                    assert outcomes == [{"detail": "preview_execution_timeout"}] * 2
                busy = await preview(client, sessions[2])
                assert busy.status_code == 503, busy.text
                assert pause.entered == 2
                assert pause.finished == 0, "slots must still belong to live worker threads"
            except BaseException as error:
                record(tmp_path, assertions_passed=False, abandonment=abandonment,
                       cancellation_issued=cancellation_issued, error_type=type(error).__name__, error=str(error),
                       compiler_pause=pause.snapshot(), phases=phases, event_loop_thread=loop_thread,
                       heartbeat_delays=heartbeat, request_input={"expected_revision":1},
                       request_targets=[url + "/preview" for url, _ in sessions],
                       request_tasks=[{"done":task.done(), "cancelled":task.cancelled()} for task in tasks],
                       full_workers=[{"done":future.done(), "cancelled":future.cancelled()} for future in worker_futures])
                raise
            finally:
                pause.release.set()
                await asyncio.gather(*tasks, return_exceptions=True)
                # These callbacks are attached after the API's slot-release callbacks.
                # Compiler completion alone does not mean geography/worker completion.
                await asyncio.gather(*(asyncio.wrap_future(future) for future in worker_futures), return_exceptions=True)
                observer.cancel()
                await asyncio.gather(observer, return_exceptions=True)
                (tmp_path / "cleanup_observation.json").write_text(json.dumps({"compiler_pause":pause.snapshot(),
                    "phases":phases, "event_loop_thread":loop_thread, "heartbeat_delays":heartbeat,
                    "full_workers":[{"done":future.done(), "cancelled":future.cancelled()} for future in worker_futures]}, indent=2), encoding="utf-8")
            after = [(await client.get(url, headers=headers)).json() for url, headers in sessions[:2]]
            assert all(state["session"]["proposal"] is None for state in after)
            assert all(state["session"]["revision"] == 1 for state in after)
            monkeypatch.setattr(api, "PREVIEW_TIMEOUT_SECONDS", 60.0)
            retry = await preview(client, sessions[2])
            assert retry.status_code == 200, retry.text
            assert retry.json()["proposal"]["state"] == "eligible"
            record(tmp_path, abandonment=abandonment, outcomes=outcomes, busy=busy.json(),
                   abandoned_sessions_after_worker_completion=after, retry=retry.json(),
                   cancellation_issued=cancellation_issued, compiler_pause=pause.snapshot(), phases=phases,
                   event_loop_thread=loop_thread, heartbeat_delays=heartbeat,
                   full_workers=[{"done":future.done(), "cancelled":future.cancelled()} for future in worker_futures])

    asyncio.run(journey())


@pytest.mark.parametrize("submission_outcome", ["worker_runs", "submission_fails"])
def test_cancel_during_submission_retains_slots_until_actual_workers_finish(app, tmp_path, monkeypatch, submission_outcome):
    pause = CompilerPause(monkeypatch)
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/api/sessions/{session_id}/preview")
    closure = inspect.getclosurevars(endpoint).nonlocals
    executor, slots = closure["preview_executor"], closure["preview_slots"]
    submit_release = threading.Event()
    entered, released_slots = [], []
    submit_waits = []
    original_submit, original_release = executor.submit, slots.release

    def delayed_submit(*args, **kwargs):
        entered.append({"at":time.monotonic(), "thread":threading.get_ident()})
        if len(entered) <= 2:
            released = submit_release.wait(15)
            submit_waits.append({"at":time.monotonic(), "released":released, "thread":threading.get_ident()})
            assert released, "controlled submission pause was not explicitly released"
            if submission_outcome == "submission_fails":
                raise RuntimeError("controlled executor submission failure")
        return original_submit(*args, **kwargs)

    def observed_release(*args, **kwargs):
        result = original_release(*args, **kwargs)
        released_slots.append({"at":time.monotonic(), "thread":threading.get_ident()})
        return result

    monkeypatch.setattr(executor, "submit", delayed_submit)
    monkeypatch.setattr(slots, "release", observed_release)

    async def wait_until(predicate):
        deadline = time.monotonic() + 15
        while not predicate():
            assert time.monotonic() < deadline, "submission/worker lifecycle did not settle"
            await asyncio.sleep(0.005)

    async def journey():
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver",
        ) as client:
            sessions = [await staged_session(client) for _ in range(3)]
            loop_thread = threading.get_ident()
            tasks = [asyncio.create_task(preview(client, session)) for session in sessions[:2]]
            cancellation_issued = False
            busy = None
            try:
                await wait_until(lambda:len(entered) == 2)
                assert submit_waits == [], "event loop could not cancel before native submission returned"
                assert pause.entered == 0 and released_slots == []
                cancellation_issued = True
                for task in tasks:
                    task.cancel()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                assert all(isinstance(result, asyncio.CancelledError) for result in results)
                if submission_outcome == "worker_runs":
                    submit_release.set()
                busy = await preview(client, sessions[2])
                assert busy.status_code == 503 and busy.json() == {"detail":"preview_capacity_reached"}
                assert released_slots == [], "cancellation released slots while submissions still owned them"
            except BaseException as error:
                record(tmp_path, assertions_passed=False, error_type=type(error).__name__, error=str(error),
                    cancellation_issued=cancellation_issued, submission_entered=entered, submission_waits=submit_waits,
                    slot_releases=released_slots, event_loop_thread=loop_thread, compiler_pause=pause.snapshot(),
                    request_input={"expected_revision":1}, request_targets=[url + "/preview" for url, _ in sessions])
                raise
            finally:
                submit_release.set()
                pause.release.set()
                await asyncio.gather(*tasks, return_exceptions=True)
                await wait_until(lambda:len(released_slots) == 2)
            after = [(await client.get(url, headers=headers)).json() for url, headers in sessions[:2]]
            assert submit_waits and all(row["released"] for row in submit_waits)
            assert all(row["thread"] != loop_thread for row in entered)
            assert all(state["session"]["proposal"] is None and state["session"]["revision"] == 1 for state in after)
            retry = await preview(client, sessions[2])
            assert retry.status_code == 200 and retry.json()["proposal"]["state"] == "eligible"
            record(tmp_path, assertions_passed=True, cancellation_issued=cancellation_issued,
                submission_outcome=submission_outcome,
                submission_entered=entered, submission_waits=submit_waits, slot_releases=released_slots,
                event_loop_thread=loop_thread, compiler_pause=pause.snapshot(),
                request_input={"expected_revision":1}, request_targets=[url + "/preview" for url, _ in sessions],
                busy=busy.json(), abandoned_sessions_after_worker_completion=after, retry=retry.json())

    asyncio.run(journey())


def test_disconnected_client_discards_completed_real_preview(app, tmp_path, monkeypatch):
    async def injected_disconnect(request):
        return request.url.path.endswith("/preview")

    monkeypatch.setattr(api.Request, "is_disconnected", injected_disconnect)

    async def journey():
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver",
        ) as client:
            session = await staged_session(client)
            result = await preview(client, session)
            url, headers = session
            after = (await client.get(url, headers=headers)).json()
            record(tmp_path, fault_injection="ASGI client disconnect indication only; real repair path",
                   response=result.json(), session_after=after)
            assert result.status_code == 499, result.text
            assert result.json()["detail"] == "preview_client_disconnected"
            assert after["session"]["proposal"] is None
            assert after["session"]["revision"] == 1

    asyncio.run(journey())


def test_cached_preview_cannot_bypass_revision_guard_after_concurrent_selection(app, tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    service = app.state.product_service
    original = service.preview

    def delayed_response(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(15), "controlled cached-result pause was not released"
        return result

    async def journey():
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver",
        ) as client:
            session = await staged_session(client)
            first = await preview(client, session)
            assert first.status_code == 200, first.text
            repeated = await preview(client, session)
            assert repeated.status_code == 200, repeated.text
            assert repeated.json()["proposal"] == first.json()["proposal"]
            assert repeated.json()["session"]["revision"] == first.json()["session"]["revision"]
            monkeypatch.setattr(service, "preview", delayed_response)
            task = asyncio.create_task(preview(client, session))
            url, headers = session
            try:
                deadline = time.monotonic() + 15
                while not entered.is_set():
                    assert time.monotonic() < deadline
                    await asyncio.sleep(0.01)
                selection = await client.post(f"{url}/selection", headers=headers, json={
                    "expected_revision": 2, "selected_day": 3,
                })
                assert selection.status_code == 200, selection.text
            finally:
                release.set()
                result = await task
            after = (await client.get(url, headers=headers)).json()
            record(tmp_path, first=first.json(), selection=selection.json(), delayed_response=result.json(),
                   session_after=after)
            assert result.status_code == 409, result.text
            assert result.json()["detail"] == "stale_session_revision"
            assert after["session"]["revision"] == 3
            assert after["session"]["selected_day"] == 3
            assert after["session"]["proposal"] == {
                **first.json()["proposal"], "expected_revision": 2, "session_revision": 3,
            }

    asyncio.run(journey())
