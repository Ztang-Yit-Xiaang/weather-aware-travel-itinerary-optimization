"""Run repository validation checks with workspace-local temporary directories."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    """Result of one local validation command."""

    name: str
    command: list[str]
    exit_code: int
    duration_seconds: float
    passed: bool
    stdout_excerpt: str
    stderr_excerpt: str
    failure_class: str
    timed_out: bool = False


def configure_workspace_temp(repo_root: Path) -> dict[str, str]:
    """Return environment overrides that keep Python temp files inside the repo."""

    temp_root = repo_root / ".codex_tmp_pytest" / "pytest"
    temp_root.mkdir(parents=True, exist_ok=True)
    probe_path = temp_root / ".run_project_checks_probe"
    try:
        probe_path.write_text("ok", encoding="utf-8")
        if probe_path.read_text(encoding="utf-8") != "ok":
            raise RuntimeError("workspace temp probe readback failed")
    except OSError as exc:
        raise RuntimeError(f"workspace temp root is not writable: {temp_root}") from exc

    env = os.environ.copy()
    temp_value = str(temp_root)
    env["TEMP"] = temp_value
    env["TMP"] = temp_value
    env["TMPDIR"] = temp_value
    env["RUFF_NO_CACHE"] = "1"
    tempfile.tempdir = temp_value
    return env


def classify_validation_failure(result: CheckResult) -> str:
    """Classify a failed check into a compact troubleshooting category."""

    if result.passed:
        return "none"
    combined = f"{result.stdout_excerpt}\n{result.stderr_excerpt}".lower()
    if result.timed_out or result.exit_code == 124 or "timed out" in combined:
        return "timeout"
    if "permissionerror" in combined or "access is denied" in combined:
        return "environment"
    if "pytestcachewarning" in combined or "temporarydirectory" in combined or "tempfile" in combined:
        return "environment"
    if "modulenotfounderror" in combined or "importerror" in combined or "fixture" in combined:
        return "test_fixture"
    if "assertionerror" in combined or "failed" in combined or "error collecting" in combined:
        return "product_code"
    return "unknown"


def run_check(command: list[str], env: Mapping[str, str], *, name: str | None = None, timeout_seconds: int = 900) -> CheckResult:
    """Run one validation command and return a summarized result."""

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            env=dict(env),
            text=True,
            timeout=timeout_seconds,
        )
        duration = time.perf_counter() - started
        result = CheckResult(
            name=name or " ".join(command),
            command=command,
            exit_code=int(completed.returncode),
            duration_seconds=float(duration),
            passed=completed.returncode == 0,
            stdout_excerpt=_excerpt(completed.stdout),
            stderr_excerpt=_excerpt(completed.stderr),
            failure_class="none",
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        result = CheckResult(
            name=name or " ".join(command),
            command=command,
            exit_code=124,
            duration_seconds=float(duration),
            passed=False,
            stdout_excerpt=_excerpt(_decode_timeout_output(exc.stdout)),
            stderr_excerpt=_excerpt(_decode_timeout_output(exc.stderr)),
            failure_class="timeout",
            timed_out=True,
        )
    if result.failure_class == "none":
        return CheckResult(**{**asdict(result), "failure_class": classify_validation_failure(result)})
    return result


def write_summary(results: list[CheckResult], output_path: Path) -> Path:
    """Write a compact JSON summary for local inspection."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary_version": "project-check-summary-v1",
        "passed": all(result.passed for result in results),
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _decode_timeout_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _excerpt(value: str, *, max_chars: int = 4000) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    env = configure_workspace_temp(repo_root)
    full_pytest_base = repo_root / "tmp_test" / f"project_checks_full_{time.time_ns()}"
    checks = [
        ("ruff", [sys.executable, "-m", "ruff", "check", "--no-cache", "src", "tests", "scripts"]),
        ("context_snapshot_pytest", [sys.executable, "-m", "pytest", "tests/data/test_context_snapshot.py"]),
        (
            "full_pytest",
            [sys.executable, "-m", "pytest", "--basetemp", str(full_pytest_base)],
        ),
    ]
    results = [run_check(command, env, name=name) for name, command in checks]
    summary_path = write_summary(results, repo_root / "results" / "quality" / "project_check_summary.json")
    for result in results:
        status = "PASS" if result.passed else f"FAIL:{result.failure_class}"
        print(f"{status} {result.name} ({result.duration_seconds:.1f}s)")
    print(f"Summary: {summary_path}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
