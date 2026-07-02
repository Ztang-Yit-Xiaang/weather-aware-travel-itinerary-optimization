import subprocess
import unittest
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from unittest import mock

from itinerary_system.repository_state import (
    ENV_COMMIT_SHA,
    ENV_DIRTY,
    ENV_PACKAGE_VERSION,
    RepositoryStateUnavailable,
    capture_repository_state,
)


def fixed_clock() -> datetime:
    return datetime(2026, 6, 30, 17, 8, tzinfo=UTC)


def completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["git"], returncode=0, stdout=stdout, stderr="")


class RepositoryStateTests(unittest.TestCase):
    def test_capture_clean_git_commit(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            self.assertEqual(kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")
            if args[-2:] == ["rev-parse", "HEAD"]:
                return completed("abc123\n")
            if args[-3:] == ["status", "--porcelain", "--untracked-files=all"]:
                return completed("")
            raise AssertionError(f"unexpected git command: {args}")

        with (
            mock.patch("itinerary_system.repository_state.subprocess.run", side_effect=fake_run),
            mock.patch("itinerary_system.repository_state.version", return_value="0.1.0"),
        ):
            state = capture_repository_state(repo_root="/tmp/repo", env={}, clock=fixed_clock)

        self.assertEqual(state.commit_sha, "abc123")
        self.assertFalse(state.dirty)
        self.assertEqual(state.package_version, "0.1.0")
        self.assertEqual(state.captured_at, "2026-06-30T17:08:00+00:00")
        self.assertEqual(len(calls), 2)

    def test_capture_dirty_git_worktree(self):
        def fake_run(args, **kwargs):
            if args[-2:] == ["rev-parse", "HEAD"]:
                return completed("abc123\n")
            if args[-3:] == ["status", "--porcelain", "--untracked-files=all"]:
                return completed(" M src/example.py\n?? new.txt\n")
            raise AssertionError(f"unexpected git command: {args}")

        with (
            mock.patch("itinerary_system.repository_state.subprocess.run", side_effect=fake_run),
            mock.patch("itinerary_system.repository_state.version", return_value="0.1.0"),
        ):
            state = capture_repository_state(repo_root="/tmp/repo", env={}, clock=fixed_clock)

        self.assertEqual(state.commit_sha, "abc123")
        self.assertTrue(state.dirty)

    def test_environment_override_does_not_require_git(self):
        env = {
            ENV_COMMIT_SHA: "override-sha",
            ENV_DIRTY: "false",
            ENV_PACKAGE_VERSION: "9.9.9",
        }
        with mock.patch("itinerary_system.repository_state.subprocess.run") as run:
            state = capture_repository_state(repo_root="/missing", env=env, clock=fixed_clock)

        run.assert_not_called()
        self.assertEqual(state.commit_sha, "override-sha")
        self.assertFalse(state.dirty)
        self.assertEqual(state.package_version, "9.9.9")

    def test_permissive_unknown_repository_is_explicit_and_dirty(self):
        with (
            mock.patch(
                "itinerary_system.repository_state.subprocess.run",
                side_effect=subprocess.CalledProcessError(128, ["git"], stderr="not a repo"),
            ),
            mock.patch("itinerary_system.repository_state.version", return_value="0.1.0"),
        ):
            state = capture_repository_state(repo_root="/missing", env={}, clock=fixed_clock)

        self.assertEqual(state.commit_sha, "unknown")
        self.assertTrue(state.dirty)
        self.assertEqual(state.package_version, "0.1.0")

    def test_strict_unknown_repository_raises(self):
        with (
            mock.patch(
                "itinerary_system.repository_state.subprocess.run",
                side_effect=subprocess.CalledProcessError(128, ["git"], stderr="not a repo"),
            ),
            mock.patch("itinerary_system.repository_state.version", return_value="0.1.0"),
        ):
            with self.assertRaises(RepositoryStateUnavailable):
                capture_repository_state(repo_root="/missing", strict=True, env={}, clock=fixed_clock)

    def test_package_version_fallback_is_unknown(self):
        env = {ENV_COMMIT_SHA: "override-sha", ENV_DIRTY: "clean"}
        with mock.patch("itinerary_system.repository_state.version", side_effect=PackageNotFoundError):
            state = capture_repository_state(repo_root="/missing", env=env, clock=fixed_clock)

        self.assertEqual(state.package_version, "unknown")


if __name__ == "__main__":
    unittest.main()
