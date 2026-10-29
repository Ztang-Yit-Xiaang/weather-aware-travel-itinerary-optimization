from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER_V3 = ROOT / "scripts" / "run_live_copilot_prompt_evaluation_v3.py"
RUNNER = ROOT / "scripts" / "run_live_copilot_prompt_evaluation_v2.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("exact_io_runner", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp_exact_test",
                "created_at": 1_786_570_000,
                "model": "gpt-5.6-terra",
                "status": "completed",
                "output": [],
            },
            request=request,
        )


def test_transport_captures_exact_bodies_without_headers() -> None:
    runner = load_runner()
    transport = runner.ExactIOTransport(FakeTransport())
    transport.begin_case(effort="low", case_id="case_1")

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": "Bearer secret-must-not-persist"},
                json={"input": "exact request", "store": False, "tools": []},
            )
            assert response.status_code == 200

    asyncio.run(exercise())

    assert len(transport.records) == 1
    record = transport.records[0]
    assert json.loads(record["request_body_utf8"]) == {
        "input": "exact request",
        "store": False,
        "tools": [],
    }
    assert json.loads(record["response_body_utf8"])["id"] == "resp_exact_test"
    assert record["request_body_sha256"] == hashlib.sha256(record["request_body_utf8"].encode()).hexdigest()
    assert record["response_body_sha256"] == hashlib.sha256(record["response_body_utf8"].encode()).hexdigest()
    assert record["request_headers_persisted"] is False
    assert record["response_headers_persisted"] is False
    assert record["authorization_persisted"] is False
    assert "secret-must-not-persist" not in json.dumps(record)


def test_exact_io_document_is_hash_bound_and_explicit() -> None:
    runner = load_runner()
    source = {"content_hash": "a" * 64}
    controls = {"content_hash": "b" * 64}
    document = runner.exact_io_document(
        effort="low",
        model="gpt-5.6-terra",
        source_snapshot=source,
        request_controls=controls,
        records=[],
    )
    unsigned = dict(document)
    content_hash = unsigned.pop("content_hash")

    assert content_hash == runner.canonical_sha256(unsigned)
    assert document["exact_request_and_response_bodies_persisted"] is True
    assert document["authorization_persisted"] is False
    assert document["api_key_persisted"] is False


def test_transport_requires_explicit_case_binding() -> None:
    runner = load_runner()
    transport = runner.ExactIOTransport(FakeTransport())
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
        json={"input": "unbound"},
    )

    with pytest.raises(RuntimeError, match="transport_case_not_bound"):
        asyncio.run(transport.handle_async_request(request))


def test_future_runner_normalizes_output_before_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("exact_io_runner_v3", RUNNER_V3)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    relative = Path("relative-evidence")
    captured: dict[str, Path] = {}

    async def fake_main(args) -> int:
        captured["output_dir"] = args.output_dir
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.v2, "parse_args", lambda: type("Args", (), {"output_dir": relative})())
    monkeypatch.setattr(module.v2, "_main", fake_main)
    original_builder = module.v2.build_source_snapshot

    assert asyncio.run(module.main_async()) == 0
    assert captured["output_dir"] == (tmp_path / relative).resolve()
    assert module.v2.build_source_snapshot is original_builder
    assert "scripts/run_live_copilot_prompt_evaluation_v3.py" in module.SOURCE_INPUTS
