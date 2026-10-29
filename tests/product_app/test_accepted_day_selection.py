"""Synthetic endpoint contract: authoritative accepted days supersede registry count.

This stubs the validated accepted-baseline boundary, not acceptance or route lineage.
The real retained ten-day artifact regression is separate local audit evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter

ROOT = Path(__file__).resolve().parents[2]


def test_selection_uses_current_accepted_days_without_preview(tmp_path: Path, monkeypatch) -> None:
    app = create_product_app(
        repository_root=ROOT, state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",), copilot=DeterministicCopilotAdapter(),
    )
    exchanges = []
    failures = []
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={})
        assert created.status_code == 200, created.text
        data = created.json()
        service = app.state.product_service
        owner = service.workspace.get(data["session"]["session_id"])
        pointer, parent, snapshot = service._current_baseline(owner.run_id)
        accepted_domain = {**parent, "ordered_days": [{"day": day, "stop_ids": []} for day in range(1, 11)]}
        monkeypatch.setattr(service, "_current_baseline", lambda run_id: (pointer, accepted_domain, snapshot))
        assert service.load(owner.run_id)[1]["trip"]["day_count"] == 7
        assert owner.proposal is None and not owner.draft
        headers = {"X-Session-Token": data["mutation_token"]}
        for day, expected_status, expected_detail in (
            (8, 200, None), (9, 200, None), (10, 200, None),
            (11, 400, "invalid_selected_day"), (0, 400, "invalid_selected_day"),
            (-1, 400, "invalid_selected_day"), (True, 422, "request_validation_failed"),
            ("8", 422, "request_validation_failed"), (8.0, 422, "request_validation_failed"),
        ):
            before = owner.public_dict()
            body = {"expected_revision": owner.revision, "selected_day": day}
            response = client.post(f"/api/sessions/{owner.session_id}/selection", headers=headers, json=body)
            output = response.json()
            exchanges.append({"request": body, "expected_status": expected_status,
                              "status": response.status_code, "response": output})
            (tmp_path / "selection_io.json").write_text(json.dumps(exchanges, indent=2), encoding="utf-8")
            if response.status_code != expected_status:
                failures.append(f"day={day!r}: expected {expected_status}, got {response.status_code} {output}")
            elif expected_status == 200:
                assert output["session"]["selected_day"] == day
                assert output["session"]["accepted_plan_id"] == parent["plan_id"]
                assert owner.proposal is None and not owner.draft
            else:
                assert output["detail"] == expected_detail
                assert owner.public_dict() == before
        assert not failures, "\n".join(failures)
