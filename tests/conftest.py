"""Pytest workspace setup for managed Windows runs."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path


def _configure_workspace_temp() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    temp_root = repo_root / ".codex_tmp_pytest" / "pytest"
    temp_root.mkdir(parents=True, exist_ok=True)

    probe_file = temp_root / ".write_probe"
    try:
        probe_file.write_text("ok", encoding="utf-8")
        if probe_file.read_text(encoding="utf-8") != "ok":
            raise RuntimeError("workspace temp probe readback failed")
    except OSError as exc:
        raise RuntimeError(f"workspace pytest temp root is not writable: {temp_root}") from exc

    temp_value = str(temp_root)
    os.environ["TEMP"] = temp_value
    os.environ["TMP"] = temp_value
    os.environ["TMPDIR"] = temp_value
    tempfile.tempdir = temp_value
    return temp_root


PYTEST_WORKSPACE_TEMP = _configure_workspace_temp()


class _WorkspaceTemporaryDirectory:
    def __init__(
        self,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | os.PathLike[str] | None = None,
        ignore_cleanup_errors: bool = False,
        delete: bool = True,
    ) -> None:
        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._delete = delete
        base_dir = Path(dir) if dir is not None else PYTEST_WORKSPACE_TEMP
        prefix_value = prefix or "tmp"
        suffix_value = suffix or ""
        self.name = str(base_dir / f"{prefix_value}{uuid.uuid4().hex}{suffix_value}")
        Path(self.name).mkdir(parents=True, exist_ok=False)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.cleanup()
        return False

    def cleanup(self) -> None:
        # The managed workspace may deny deletion of files created during tests.
        # The temp root is ignored, so cleanup is intentionally best-effort.
        return None


tempfile.TemporaryDirectory = _WorkspaceTemporaryDirectory
