"""Two real guarded servers over a private copy of product inputs, not API mocks."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from e2e1_provider_isolation import deterministic_environment
from run_e2e1_vertical_slice import _isolation_passed, _port_is_available, _wait_until_ready


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_state(root: Path):
    return {path.relative_to(root).as_posix(): {
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes":path.stat().st_size,
    } for path in sorted(root.rglob("*")) if path.is_file()}


def copy_product_inputs(destination: Path, evidence: Path):
    registration = json.loads((ROOT / "configs/product_app_registry.json").read_text(encoding="utf-8"))
    delta = json.loads((ROOT / "configs/product_route_delta.json").read_text(encoding="utf-8"))
    delta_directory = (ROOT / delta["descriptor_path"]).resolve().parent
    assert delta_directory.is_relative_to((ROOT / "results/route_deltas").resolve())
    paths = {ROOT / "scripts/run_product_app.py", ROOT / "scripts/e2e1_provider_isolation.py"}
    for directory in (ROOT / "src", ROOT / "configs", delta_directory):
        paths.update(path for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    for run in registration["runs"]:
        directory = (ROOT / run["relative_path"]).resolve()
        assert directory.is_relative_to((ROOT / "runs").resolve())
        if run["default"]:
            paths.update(path for path in directory.rglob("*") if path.is_file())
        else:
            paths.add(directory / "manifest.json")
    inventory = []
    longest_destination = max(len(str(destination / source.relative_to(ROOT))) for source in paths)
    assert os.name != "nt" or longest_destination < 260, "Private input copy exceeds the Windows path limit; choose a shorter explicit root"
    for source in sorted(paths):
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        body = source.read_bytes()
        target.write_bytes(body)
        inventory.append({"path":relative.as_posix(), "sha256":hashlib.sha256(body).hexdigest(), "size_bytes":len(body),
            "copied_sha256":hashlib.sha256(target.read_bytes()).hexdigest()})
    assert all(hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"] for row in inventory), "Source changed while copying private fixture"
    assert all(row["sha256"] == row["copied_sha256"] for row in inventory), "Private inputs must exactly match the recorded production inputs"
    write_json(evidence / "private_fixture_inputs.json", {
        "schema_version":"build-coherence-private-inputs-v1", "files":inventory,
        "copied_at":datetime.now(UTC).isoformat(),
        "private_repository":str(destination.resolve()),
        "maximum_destination_path_characters":longest_destination,
        "production_source_modified":False, "api_response_mocking":False,
        "fixture_scope":"Exact source/configuration/default evidence copy; nondefault runs are not selected",
    })


def change_registry_label(repository: Path, evidence: Path, generation: str):
    path = repository / "configs/product_app_registry.json"
    before = path.read_bytes()
    value = json.loads(before)
    selected = next(row for row in value["runs"] if row["default"])
    selected["label"] = f"California Coast build-coherence fixture {generation}"
    after = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(after)
    write_json(evidence / f"registry_change_{generation}.json", {
        "path":"configs/product_app_registry.json", "purpose":"Isolated valid build generation change; plan/evidence IDs untouched",
        "before_utf8":before.decode(), "after_utf8":after.decode(),
        "before_sha256":hashlib.sha256(before).hexdigest(), "after_sha256":hashlib.sha256(after).hexdigest(),
    })
    return before


@contextmanager
def guarded_generation(repository: Path, state: Path, evidence: Path, base_url: str, run_id: str, generation: str):
    from urllib.parse import urlsplit

    port = urlsplit(base_url).port
    assert port is not None and _port_is_available(port), "The disposable test port must be unused"
    generation_dir = evidence / generation
    generation_dir.mkdir()
    environment = deterministic_environment(os.environ)
    environment.update(PRODUCT_APP_ORIGIN=base_url, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "scripts/e2e1_provider_isolation.py", "--port", str(port),
        "--state-root", str(state), "--output", str(generation_dir / "provider_isolation.json"),
        "--run-id", f"{run_id}_{generation}"]
    write_json(generation_dir / "server_command.json", {"command":command, "cwd_scope":"private_product_copy",
        "generation":generation,"run_id":f"{run_id}_{generation}"})
    with (generation_dir / "server.stdout.log").open("w", encoding="utf-8") as stdout, (generation_dir / "server.stderr.log").open("w", encoding="utf-8") as stderr:
        started_at = datetime.now(UTC).isoformat()
        process = subprocess.Popen(command, cwd=repository, env=environment, stdout=stdout,
            stderr=stderr, stdin=subprocess.PIPE, text=True)
        ready_at = None
        failed = False
        try:
            health = _wait_until_ready(process, base_url)
            ready_at = datetime.now(UTC).isoformat()
            write_json(generation_dir / "health.json", health)
            yield health
        except BaseException:
            failed = True
            raise
        finally:
            if process.poll() is None:
                try:
                    process.communicate("stop\n", timeout=20)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait(timeout=5)
            receipt_path = generation_dir / "provider_isolation.json"
            receipt_bytes = receipt_path.read_bytes() if receipt_path.is_file() else None
            try:
                receipt = json.loads(receipt_bytes) if receipt_bytes is not None else None
            except ValueError:
                receipt = None
            verified = _isolation_passed({"artifact":"provider_isolation.json", "record":receipt,
                "run_id":f"{run_id}_{generation}", "server_pid":process.pid})
            write_json(generation_dir / "server_lifecycle.json", {"pid":process.pid, "exit_code":process.returncode,
                "generation":generation,"run_id":f"{run_id}_{generation}",
                "process_started_at":started_at,"ready_at":ready_at,"process_stopped_at":datetime.now(UTC).isoformat(),
                "receipt_sha256":hashlib.sha256(receipt_bytes).hexdigest() if receipt_bytes is not None else None,
                "isolation_verified":verified, "provider_call_count":receipt.get("sent_provider_calls") if verified else None})
            if not failed:
                assert process.returncode == 0 and verified, "Guard lifecycle must finalize independently for each generation"


def generation_evidence_bindings(evidence: Path, run_id: str):
    """Join each real capture to its own finalized application process witness."""
    bindings = []
    for generation in ("generation_a", "generation_b"):
        directory = evidence / generation
        lifecycle = json.loads((directory / "server_lifecycle.json").read_text(encoding="utf-8"))
        receipt_path = directory / "provider_isolation.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        identity = f"{run_id}_{generation}"
        assert lifecycle["isolation_verified"] and lifecycle["exit_code"] == 0
        assert lifecycle["run_id"] == receipt["run_id"] == identity
        assert receipt["process_id"] == lifecycle["pid"]
        assert lifecycle["receipt_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        installed, finalized = map(datetime.fromisoformat, (receipt["installed_at"], receipt["finalized_at"]))
        assert datetime.fromisoformat(lifecycle["process_started_at"]) <= installed
        assert installed <= datetime.fromisoformat(lifecycle["ready_at"]) <= finalized
        assert finalized <= datetime.fromisoformat(lifecycle["process_stopped_at"])
        captures = []
        referenced = set()
        for path in sorted(directory.glob("*.readiness.json")):
            readiness = json.loads(path.read_text(encoding="utf-8"))
            runtime_binding = readiness["runtime_environment"]
            runtime_path = directory / runtime_binding["artifact"]
            assert runtime_path.resolve().parent == directory.resolve()
            record = json.loads(runtime_path.read_text(encoding="utf-8"))
            assert record["run_id"] == identity
            assert runtime_binding["sha256"] == hashlib.sha256(runtime_path.read_bytes()).hexdigest()
            assert installed <= datetime.fromisoformat(record["observed_at"]) <= datetime.fromisoformat(record["completed_at"])
            assert datetime.fromisoformat(record["completed_at"]) <= datetime.fromisoformat(readiness["screenshot_started_at"])
            assert datetime.fromisoformat(readiness["screenshot_started_at"]) <= datetime.fromisoformat(readiness["screenshot_completed_at"]) <= finalized
            referenced.add(runtime_path.name)
            captures.append(path.relative_to(evidence).as_posix())
        assert captures and referenced == {path.name for path in directory.glob("runtime_environment_*.json")}
        bindings.append({"generation":generation,"run_id":identity,"pid":lifecycle["pid"],
            "build_id":json.loads((directory / "health.json").read_text(encoding="utf-8"))["build_id"],
            "provider_receipt":receipt_path.relative_to(evidence).as_posix(),
            "provider_receipt_sha256":lifecycle["receipt_sha256"],
            "installed_at":receipt["installed_at"],"finalized_at":receipt["finalized_at"],
            "readiness_artifacts":captures,"provider_call_count":receipt["sent_provider_calls"]})
    assert datetime.fromisoformat(bindings[0]["finalized_at"]) <= datetime.fromisoformat(bindings[1]["installed_at"])
    assert bindings[0]["build_id"] != bindings[1]["build_id"]
    return bindings
