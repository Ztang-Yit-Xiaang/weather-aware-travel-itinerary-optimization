import subprocess
from pathlib import Path


def test_destination_overlay_layout_and_keyboard_contracts():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("destination_overlay_layout.test.mjs"))],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
