"""Keep the real frontend reply-order regression in normal nonbrowser pytest."""

import subprocess
from pathlib import Path


def test_selection_conversation_generation_contract():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("selection_conversation_generation_contract.test.mjs"))],
        cwd=root,
        check=True,
    )
