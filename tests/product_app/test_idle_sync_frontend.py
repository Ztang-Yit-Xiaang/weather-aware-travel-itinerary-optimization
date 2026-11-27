import os
import subprocess
from pathlib import Path


def test_idle_sync_frontend_contract():
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, IDLE_SYNC_CANDIDATE=str(root))
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("idle_sync_contract.test.mjs"))],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
