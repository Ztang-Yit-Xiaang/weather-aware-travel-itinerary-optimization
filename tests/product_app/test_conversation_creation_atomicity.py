"""Real repository creation/binding races; no browser, provider, or fake outcome."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("after_delete", [False, True])
def test_concurrent_first_reads_share_one_bound_conversation(tmp_path, after_delete):
    service = ProductService(ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"), tmp_path)
    bundle, _ = service.load(service.registry.default.run_id)
    owner, token = service.workspace.create_session(
        service.registry.default.run_id, bundle.parent_plan["plan_id"], 5, trip_id=service.registry.default.trip_id
    )
    if after_delete:
        service.read_conversation(owner)
        assert service.delete_conversation(owner, owner.revision)
    assert owner.conversation_id is None
    first_created, second_started, second_created, release = (threading.Event() for _ in range(4))
    real = service.conversations.get_or_create
    calls, trace = [], []

    def before_bind(session):
        conversation = real(session)
        calls.append(conversation.as_dict())
        if not first_created.is_set():
            first_created.set()
            assert release.wait(5)
        else:
            second_created.set()
        return conversation

    def read(second=False):
        if second:
            second_started.set()
        try:
            session = service.workspace.authenticate(owner.session_id, token)
            conversation = service.read_conversation(session)
            value = {"conversation": conversation.as_dict(), "session": session.public_dict()}
            trace.append({"result": "ok", "output": value})
            return value
        except Exception as exc:
            trace.append({"result": "error", "type": type(exc).__name__, "code": str(exc)})
            raise

    with patch.object(service.conversations, "get_or_create", before_bind):
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(read)
            assert first_created.wait(5)
            # Verify the boundary excludes another authenticator; then exercise
            # the actual second reader instead of replacing its result.
            unlocked = service.workspace._lock.acquire(blocking=False)
            if unlocked:
                service.workspace._lock.release()
            second = pool.submit(read, True)
            assert second_started.wait(5)
            overlapped = second_created.wait(0.2)
            release.set()
            try:
                results = [first.result(timeout=5), second.result(timeout=5)]
            finally:
                (tmp_path / "creation_binding_io.json").write_text(
                    json.dumps(
                        {
                            "after_delete": after_delete,
                            "repository_outputs": calls,
                            "reader_results": trace,
                            "boundary_unlocked": unlocked,
                            "second_repository_overlapped": overlapped,
                            "instrumentation": "Pause only after the actual first repository result",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    assert not unlocked and not overlapped
    current = service.workspace.authenticate(owner.session_id, token)
    assert {row["conversation"]["conversation_id"] for row in results} == {current.conversation_id}
    assert all(row["session"]["conversation_id"] == current.conversation_id for row in results)
    paths = list(service.conversations.root.glob("conversation_*.json"))
    assert len(paths) == 1
    assert json.loads(paths[0].read_text(encoding="utf-8"))["conversation_id"] == current.conversation_id


def test_evidence_observer_does_not_create_a_deleted_conversation(tmp_path):
    import importlib.util
    import sys
    from types import SimpleNamespace

    browser = ROOT / "tests/product_app/browser"
    sys.path.insert(0, str(browser))
    try:
        spec = importlib.util.spec_from_file_location(
            "conversation_fixture_under_test", browser / "copilot_conversation_fixture.py"
        )
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
    finally:
        sys.path.remove(str(browser))
    service = ProductService(ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"), tmp_path / "state")
    owner, _, _ = service.create_session(service.registry.default.run_id)
    assert service.delete_conversation(owner, owner.revision)
    before = owner.public_dict()
    context = fixture.observe_context(SimpleNamespace(service=service), owner.session_id, tmp_path, "deleted")
    assert owner.public_dict() == before
    assert owner.conversation_id is None
    assert not list(service.conversations.root.glob("conversation_*.json"))
    assert context["conversation_memory"]["proposed_preferences"] == []
