import subprocess
from pathlib import Path


def test_pending_resolution_frontend_contract() -> None:
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("pending_resolution_contract.test.mjs"))],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
