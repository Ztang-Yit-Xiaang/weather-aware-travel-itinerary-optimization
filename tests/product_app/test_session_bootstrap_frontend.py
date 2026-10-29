"""Executable regressions for returning users and unsent local text."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("filename", ["session_bootstrap_contract.test.mjs", "accepted_day_order_contract.test.mjs"])
def test_session_bootstrap_and_unsent_composer_contract(filename):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the frontend contract")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name(filename))],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
