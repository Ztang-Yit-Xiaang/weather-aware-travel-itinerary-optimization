import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize('script', [
    'mutation_recovery_frontend_contract.test.mjs',
    'mutation_ui_contract.test.mjs',
])
def test_pending_mutation_recovery_contract(script: str) -> None:
    target = Path(__file__).with_name(script)
    result = subprocess.run(
        ['node', '--test', str(target)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
