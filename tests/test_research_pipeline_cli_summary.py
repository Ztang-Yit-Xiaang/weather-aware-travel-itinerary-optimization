from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_research_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_research_pipeline_cli", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_main_prints_pipeline_run_record(monkeypatch, capsys):
    run_type = type("Run", (), {"to_record": lambda self: {"run_id": "run_test", "status": "completed"}})
    monkeypatch.setattr(MODULE, "parse_args", lambda: object())
    monkeypatch.setattr(MODULE, "run_from_args", lambda parsed: run_type())

    assert MODULE.main() == 0
    assert '"run_id": "run_test"' in capsys.readouterr().out
