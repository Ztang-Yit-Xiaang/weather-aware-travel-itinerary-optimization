"""Deterministic semantic-candidate providers with no live model dependency."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

from ..research_artifacts import PlanArtifactV2, stable_content_hash
from .models import SemanticInterpretationCandidate


class SemanticCandidateProvider(Protocol):
    def candidates(
        self,
        *,
        parent: PlanArtifactV2,
        user_edit: str,
        repair_session_id: str,
        evidence_refs: tuple[str, ...],
    ) -> tuple[SemanticInterpretationCandidate, ...]: ...


class FrozenSemanticCandidateProvider:
    """Load replayable candidates from a JSON object, list, or JSONL fixture."""

    source_name = "frozen_fixture"

    def __init__(self, fixture_path: str | Path) -> None:
        self.fixture_path = Path(fixture_path)

    def candidates(
        self,
        *,
        parent: PlanArtifactV2,
        user_edit: str,
        repair_session_id: str,
        evidence_refs: tuple[str, ...],
    ) -> tuple[SemanticInterpretationCandidate, ...]:
        del parent
        records = _read_fixture(self.fixture_path)
        text_hash = stable_content_hash({"user_edit": user_edit})
        candidates: list[SemanticInterpretationCandidate] = []
        for index, record in enumerate(records, start=1):
            interpretation_type = str(record.get("interpretation_type", "")).strip()
            candidate_seed = {
                "session": repair_session_id,
                "index": index,
                "type": interpretation_type,
                "targets": record.get("target_ids", ()),
                "parameters": record.get("normalized_parameters", {}),
            }
            candidates.append(
                SemanticInterpretationCandidate(
                    interpretation_id=str(
                        record.get("interpretation_id") or f"interpretation_{stable_content_hash(candidate_seed)}"
                    ),
                    repair_session_id=repair_session_id,
                    user_text_hash=text_hash,
                    target_ids=tuple(str(value) for value in record.get("target_ids", ())),
                    interpretation_type=interpretation_type,
                    normalized_parameters=dict(record.get("normalized_parameters", {})),
                    support_score=_optional_float(record.get("support_score")),
                    evidence_refs=tuple(str(value) for value in record.get("evidence_refs", evidence_refs)),
                    confirmed=bool(record.get("confirmed", False)),
                    redacted_excerpt=_optional_text(record.get("redacted_excerpt")),
                )
            )
        return tuple(candidates)


class RuleBasedSemanticCandidateProvider:
    """Small transparent vocabulary for the initial research baseline."""

    source_name = "rule_based_v1"

    def candidates(
        self,
        *,
        parent: PlanArtifactV2,
        user_edit: str,
        repair_session_id: str,
        evidence_refs: tuple[str, ...],
    ) -> tuple[SemanticInterpretationCandidate, ...]:
        normalized = " ".join(str(user_edit).lower().split())
        targets = _named_targets(parent, normalized)
        interpretations: list[tuple[str, dict[str, object]]] = []
        if "less walking" in normalized or "easier" in normalized:
            interpretations.append(("reduce_walking_burden", {"ordinal_reduction": 1}))
        if "less driving" in normalized or "easier" in normalized:
            interpretations.append(("reduce_driving_burden", {"target_minutes_reduction": 30.0}))
        if "earlier" in normalized and any(token in normalized for token in ("hotel", "lodging", "arrival")):
            interpretations.append(("require_earlier_lodging_arrival", {"arrival_shift_minutes": 60.0}))
        if "preserve" in normalized or "keep" in normalized:
            interpretations.append(("preserve_named_stop", {}))
        if "less busy" in normalized or "fewer stops" in normalized:
            interpretations.append(("adjust_ordinal_pace", {"ordinal_reduction": 1}))
        if not interpretations:
            return ()

        text_hash = stable_content_hash({"user_edit": user_edit})
        result: list[SemanticInterpretationCandidate] = []
        for index, (interpretation_type, parameters) in enumerate(interpretations, start=1):
            seed = {
                "session": repair_session_id,
                "index": index,
                "type": interpretation_type,
                "targets": targets,
                "parameters": parameters,
            }
            result.append(
                SemanticInterpretationCandidate(
                    interpretation_id=f"interpretation_{stable_content_hash(seed)}",
                    repair_session_id=repair_session_id,
                    user_text_hash=text_hash,
                    target_ids=targets,
                    interpretation_type=interpretation_type,
                    normalized_parameters=parameters,
                    support_score=None,
                    evidence_refs=evidence_refs,
                    confirmed=False,
                )
            )
        return tuple(result)


def _read_fixture(path: Path) -> tuple[dict[str, object], ...]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return tuple(json.loads(line) for line in text.splitlines() if line.strip())
    raw = json.loads(text)
    if isinstance(raw, dict):
        raw = raw.get("semantic_candidates", raw.get("candidates", ()))
    if not isinstance(raw, list):
        raise ValueError("semantic candidate fixture must contain a list of candidate objects")
    return tuple(dict(record) for record in raw)


def _named_targets(parent: PlanArtifactV2, normalized_text: str) -> tuple[str, ...]:
    matches: list[str] = []
    for index, stop in enumerate(parent.selected_stops, start=1):
        stop_id = str(stop.get("stop_id") or stop.get("poi_id") or f"stop_{index}")
        name = str(stop.get("name") or stop.get("stop_name") or stop_id).lower().strip()
        if name and re.search(rf"\b{re.escape(name)}\b", normalized_text):
            matches.append(stop_id)
    return tuple(matches)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
