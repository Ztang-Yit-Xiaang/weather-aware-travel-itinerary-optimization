"""Compatibility imports for the existing notebook helper modules."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_notebook_path() -> Path:
    notebook_dir = _project_root() / "notebook"
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))
    return notebook_dir


def import_legacy_module(module_name: str):
    ensure_notebook_path()
    return importlib.import_module(module_name)
