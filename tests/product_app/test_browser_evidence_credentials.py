from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

_PATH = Path(__file__).parent / "browser" / "test_e2e1_live_copilot_chat.py"
_SPEC = importlib.util.spec_from_file_location("browser_capture_contract", _PATH)
assert _SPEC and _SPEC.loader
capture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(capture)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_vertical_slice as verifier  # noqa: E402


def test_exact_nonsecret_body_is_not_reserialized() -> None:
    raw = '{ "plan_id": "plan_123", "nested": [1, 2] }\n'
    assert capture._credential_safe_body(raw) == (raw, [])


def test_session_response_redacts_credentials_and_discloses_original_hash() -> None:
    raw = '{"session_id":"session_synthetic","mutation_token":"synthetic-private-token"}'
    response = SimpleNamespace(request=SimpleNamespace(post_data=None, method="POST"),
                               url="http://127.0.0.1:8127/api/sessions", status=201)
    record = capture._http_record(response, response_body=raw)
    assert "synthetic-private-token" not in json.dumps(record)
    assert json.loads(record["response_body_utf8"])["mutation_token"] == "OMITTED_SECRET"
    assert record["response_body_sha256"] == capture._hash_text(record["response_body_utf8"])
    assert record["response_original_body_sha256"] == capture._hash_text(raw)
    assert record["body_capture"] == "credential_redacted"
    assert record["credential_redactions"] == {"request": [], "response": ["/mutation_token"]}
    assert record["session_token_persisted"] is False


def test_nested_credentials_are_not_mistaken_for_plan_content() -> None:
    body, removed = capture._credential_safe_body(
        '{"a":[{"session_token":"secret","token_verifier":{"algorithm":"sha256","digest":"private"},'
        '"plan_content_hash":"abc"}]}'
    )
    assert json.loads(body)["a"][0] == {
        "session_token": "OMITTED_SECRET",
        "token_verifier": "OMITTED_SECRET",
        "plan_content_hash": "abc",
    }
    assert removed == ["/a/0/session_token", "/a/0/token_verifier"]


def test_evidence_security_scan_rejects_unredacted_token_verifier(tmp_path: Path) -> None:
    evidence = tmp_path / "session.json"
    evidence.write_text(json.dumps({"token_verifier": {"algorithm": "sha256", "digest": "private"}}), encoding="utf-8")
    with pytest.raises(verifier.E2E1VerificationError):
        verifier._verify_text_security(tmp_path)
    evidence.write_text(json.dumps({"token_verifier": "OMITTED_SECRET"}), encoding="utf-8")
    verifier._verify_text_security(tmp_path)


@pytest.mark.parametrize("filename,index_name", [
    ("test_direct_edit_journeys_live.py", "responses"),
    ("test_mobile_chat_journey_live.py", "http"),
])
def test_nested_response_callback_does_not_overwrite_exact_io(tmp_path, filename, index_name):
    source = (_PATH.parent / filename).read_text(encoding="utf-8")
    callback = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == "record")
    index = []
    namespace = {"evidence":tmp_path,index_name:index,"urlsplit":urlsplit,"_write_json":capture._write_json}

    def read_body(response, *, response_body=None):
        if response.url.endswith("/outer"):
            namespace["record"](SimpleNamespace(url="http://127.0.0.1:8127/api/inner"))
        return {"url":response.url,"method":"GET","response_status_code":200}

    namespace["_http_record"] = read_body
    exec(compile(ast.Module(body=[callback], type_ignores=[]), filename, "exec"), namespace)
    namespace["record"](SimpleNamespace(url="http://127.0.0.1:8127/api/outer"))
    assert len(index) == len({row["file"] for row in index}) == 2
    assert len(list(tmp_path.glob("http_*.json"))) == 2
    for row in index:
        assert json.loads((tmp_path / row["file"]).read_text(encoding="utf-8"))["url"] == row["url"]
