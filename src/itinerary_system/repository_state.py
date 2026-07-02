"""Repository identity capture for reproducible research artifacts."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGE_DISTRIBUTION_NAME = "weather-aware-travel-itinerary-optimization"
ENV_COMMIT_SHA = "ITINERARY_SYSTEM_COMMIT_SHA"
ENV_DIRTY = "ITINERARY_SYSTEM_DIRTY"
ENV_PACKAGE_VERSION = "ITINERARY_SYSTEM_PACKAGE_VERSION"


@dataclass(frozen=True)
class RepositoryState:
    """Exact code state used to create a research artifact."""

    commit_sha: str
    dirty: bool
    package_version: str
    captured_at: str

    def to_record(self) -> dict[str, str | bool]:
        """Return a JSON-serializable repository-state record."""

        return asdict(self)


class RepositoryStateUnavailable(RuntimeError):
    """Raised when strict repository-state capture cannot resolve code identity."""


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _timestamp(clock: Callable[[], datetime | str] | None) -> str:
    value = clock() if clock is not None else datetime.now(UTC)
    if isinstance(value, datetime):
        timestamp = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return timestamp.isoformat()
    return str(value)


def _package_version(env: Mapping[str, str]) -> str:
    override = str(env.get(ENV_PACKAGE_VERSION, "")).strip()
    if override:
        return override
    try:
        return version(PACKAGE_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return "unknown"


def _parse_dirty(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "dirty"}:
        return True
    if text in {"0", "false", "no", "n", "clean"}:
        return False
    return default


def _git_env(env: Mapping[str, str]) -> dict[str, str]:
    run_env = dict(os.environ)
    run_env.update({str(key): str(value) for key, value in env.items()})
    run_env["GIT_OPTIONAL_LOCKS"] = "0"
    return run_env


def _run_git(repo_root: Path, args: list[str], env: Mapping[str, str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        check=True,
        env=_git_env(env),
        text=True,
        timeout=8,
    )
    return completed.stdout.strip()


def capture_repository_state(
    repo_root: str | Path | None = None,
    *,
    strict: bool = False,
    env: Mapping[str, str] | None = None,
    clock: Callable[[], datetime | str] | None = None,
) -> RepositoryState:
    """Capture commit, dirty flag, package version, and timestamp.

    Args:
        repo_root: Repository root to inspect. Defaults to the package's repo.
        strict: Raise if Git identity cannot be resolved.
        env: Optional environment mapping for deterministic tests or CI overrides.
        clock: Optional timestamp provider.

    Returns:
        RepositoryState with explicit unknown values when permissive capture fails.

    Raises:
        RepositoryStateUnavailable: If strict mode cannot resolve the Git state.
    """

    env_values: Mapping[str, str] = os.environ if env is None else env
    package_version = _package_version(env_values)
    captured_at = _timestamp(clock)

    commit_override = str(env_values.get(ENV_COMMIT_SHA, "")).strip()
    if commit_override:
        return RepositoryState(
            commit_sha=commit_override,
            dirty=_parse_dirty(env_values.get(ENV_DIRTY), default=False),
            package_version=package_version,
            captured_at=captured_at,
        )

    resolved_root = Path(repo_root) if repo_root is not None else _default_repo_root()
    try:
        commit_sha = _run_git(resolved_root, ["rev-parse", "HEAD"], env_values)
        status = _run_git(resolved_root, ["status", "--porcelain", "--untracked-files=all"], env_values)
        return RepositoryState(
            commit_sha=commit_sha,
            dirty=bool(status.strip()),
            package_version=package_version,
            captured_at=captured_at,
        )
    except Exception as exc:
        if strict:
            raise RepositoryStateUnavailable(f"Cannot capture repository state for {resolved_root}: {exc}") from exc
        return RepositoryState(
            commit_sha="unknown",
            dirty=True,
            package_version=package_version,
            captured_at=captured_at,
        )
