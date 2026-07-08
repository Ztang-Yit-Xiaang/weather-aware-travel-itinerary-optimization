import unittest
from pathlib import Path

from scripts.run_project_checks import CheckResult, classify_validation_failure, configure_workspace_temp


def check_result(stdout: str = "", stderr: str = "", *, exit_code: int = 1, timed_out: bool = False) -> CheckResult:
    return CheckResult(
        name="unit",
        command=["python", "-m", "pytest"],
        exit_code=exit_code,
        duration_seconds=0.1,
        passed=False,
        stdout_excerpt=stdout,
        stderr_excerpt=stderr,
        failure_class="unknown",
        timed_out=timed_out,
    )


class ProjectCheckTests(unittest.TestCase):
    def test_permission_error_classifies_as_environment(self):
        result = check_result(stderr="PermissionError: [WinError 5] Access is denied")

        self.assertEqual(classify_validation_failure(result), "environment")

    def test_assertion_failure_classifies_as_product_code(self):
        result = check_result(stdout="FAILED tests/test_example.py::test_case - AssertionError")

        self.assertEqual(classify_validation_failure(result), "product_code")

    def test_timeout_classifies_as_timeout(self):
        result = check_result(stderr="command timed out after 600 seconds", exit_code=124, timed_out=True)

        self.assertEqual(classify_validation_failure(result), "timeout")

    def test_workspace_temp_disables_ruff_cache(self):
        repo_root = Path.cwd()

        env = configure_workspace_temp(repo_root)

        self.assertEqual(env["RUFF_NO_CACHE"], "1")
        self.assertEqual(env["TEMP"], str(repo_root / ".codex_tmp_pytest" / "pytest"))


if __name__ == "__main__":
    unittest.main()
