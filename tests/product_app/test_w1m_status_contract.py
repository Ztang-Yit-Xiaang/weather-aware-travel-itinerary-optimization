from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECISION = ROOT / "docs" / "current" / "map_runtime_substitution_decision.md"
EXECUTION = ROOT / "docs" / "planning" / "current_execution_plan.md"
GATE_MAP = ROOT / "docs" / "planning" / "research_pipeline_and_gate_map.md"
PLANNING_README = ROOT / "docs" / "planning" / "README.md"
PROBLEMS = ROOT / "docs" / "current" / "current_problem_manifest.md"


def test_w1m_authority_chain_is_strict_utf8_and_has_one_status_story() -> None:
    documents = [path.read_bytes().decode("utf-8", errors="strict") for path in (
        DECISION,
        EXECUTION,
        GATE_MAP,
        PLANNING_README,
        PROBLEMS,
    )]
    combined = "\n".join(documents)

    assert documents[0].startswith("# MAP-DEC-002 - Local Map Runtime Substitution")
    assert "MapLibre GL JS plus PMTiles the primary runtime" in combined
    assert "Atlas a deferred explicit backup" in combined
    assert "W1M/G1 are `verified`" in combined
    assert "| W2 / G2 | `verified` |" in documents[1]
    assert "| W3 / G3 | `verified` |" in documents[1]
    assert "| W4 / G4 | W4 `implemented`; G4 `blocked` |" in documents[1]
    assert "corrected-v2 W2/G2" in documents[2]
    assert "corrected-v2 W3/G3 are `verified`" in documents[2]
    assert "W4 is `implemented`" in documents[2]
    assert "W1M/G1 and corrected-v2 W2/G2 and W3/G3 are verified" in documents[3]
    assert "W4 is implemented" in documents[3]
    assert "G2 and G3 are `blocked` pending independent v2 revalidation" not in combined
    assert "G2/G3 are blocked pending" not in combined
    assert "W4 is therefore `blocked`" not in combined
    assert "ready_not_started" not in combined
    assert "planned_not_ready" not in combined
    assert "G1 remains `blocked`" not in combined
    assert "W2 is `planned` but not ready" not in combined
    assert "W2 `implemented`; G2 `in-progress`" not in combined
    assert "G2 independent verification is `in-progress`" not in combined
    assert "CP-010" in combined and "`in-progress`" in combined
    assert "E3.1" in combined and "E5" in combined


def test_w1m_authority_chain_contains_no_known_mojibake_sequences() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (DECISION, EXECUTION, GATE_MAP, PLANNING_README, PROBLEMS)
    )
    forbidden = ("\ufffd", "鈥", "锟", "漏 OpenStreetMap", "бк", "иC")
    assert all(value not in combined for value in forbidden)
