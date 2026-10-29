"""Real API -> compiler -> evaluator -> production JS outcome contracts."""

import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler

ROOT = Path(__file__).resolve().parents[2]
RUN = "california_coast_coherent_demo_v1"
TARGET = "bixby_creek_bridge_viewpoint"
CANDIDATE = "pfeiffer_big_sur_state_park"

NODE = r"""
const fs = require('node:fs'); const assert = require('node:assert/strict');
const row = JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));
const source = fs.readFileSync('src/itinerary_system/product_app/static/js/app.js', 'utf8');
const part = source.slice(source.indexOf('const TYPED_EDIT_OPERATIONS'), source.indexOf('function safeExternalLink'));
const bind = workspace => new Function('state', part + '\nreturn normalizeEvaluatedPreviewResponse;')({session:row.before,workspace});
const normalize = bind(row.workspace);
const expected = {...row.before, parent_plan_content_hash:row.response.proposal.parent_plan_content_hash};
(async () => {
  const result = await normalize(structuredClone(row.response), expected);
  assert.ok(result, 'actual API response must pass production frontend contract');
  assert.equal(result.proposal.state, row.expected_state);
  if (row.expected_state !== 'blocked') {
    assert.equal(await bind({timeline:[]})(structuredClone(row.response), expected), null,
      'missing accepted-plan workspace must remain blocked, not bypass production provenance checks');
  }
  if (row.expected_state === 'blocked') {
    const forged = structuredClone(row.response);
    forged.proposal.state = 'ineligible'; forged.proposal.eligibility = 'ineligible';
    forged.session.proposal = structuredClone(forged.proposal);
    assert.equal(await normalize(forged, expected), null, 'no evaluator means no evaluated-ineligible claim');
  }
  console.log(JSON.stringify({state:result.proposal.state,request_id:result.proposal.runtime_execution.request_id,
    child_plan_id:result.proposal.child_plan_id,production_normalizer_passed:true}));
})().catch(error => {console.error(error);process.exitCode=1;});
"""


@pytest.mark.parametrize("case,expected_state", [
    ("missing_route", "blocked"), ("keep_only", "ineligible"), ("keep_replace", "eligible"),
])
def test_actual_api_outcome_passes_strict_website_contract(tmp_path, case, expected_state):
    app = create_product_app(repository_root=ROOT, registry_path=ROOT / "configs/product_app_registry.json",
                             state_root=tmp_path / "state", additional_allowed_authorities=("testserver",),
                             copilot=DeterministicCopilotAdapter())
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={"run_id": RUN})
        assert created.status_code == 200, created.text
        data = created.json()
        session = data["session"]
        headers = {"X-Session-Token": data["mutation_token"]}
        url = f"/api/sessions/{session['session_id']}"
        if case == "missing_route":
            service = app.state.product_service
            compiler = service._draft_compilers[RUN]
            bundles = {}
            for key, bundle in service._evidence_bundles[RUN].items():
                matrix = deepcopy(bundle.route_matrix)
                matrix["cells"] = [cell for cell in matrix["cells"] if cell["destination_id"] != CANDIDATE]
                bundles[key] = replace(bundle, route_matrix=matrix)
            service._draft_compilers[RUN] = FrozenDraftCompiler(compiler.parent_plan, bundles, repository_root=ROOT)
        operations = []
        if case != "missing_route":
            operations.append({"type": "keep_stop", "target": "golden_gate_bridge", "parameters": {}})
        if case != "keep_only":
            operations.append({"type": "replace_nearby", "target": TARGET, "parameters": {"candidate_id": CANDIDATE}})
        exact_requests = []
        for operation in operations:
            request = {"expected_revision": session["revision"], **operation, "source": "map"}
            staged = client.post(f"{url}/draft/operations", headers=headers, json=request)
            assert staged.status_code == 200, staged.text
            exact_requests.append({"request": request, "response": staged.json()})
            session = staged.json()["session"]
        request = {"expected_revision": session["revision"]}
        preview = client.post(f"{url}/preview", headers=headers, json=request)
        assert preview.status_code == 200, preview.text
        response = preview.json()
        row = {"before": session, "workspace": data["workspace"], "draft_io": exact_requests, "preview_request": request,
               "response": response, "expected_state": expected_state}
        path = tmp_path / "exact_api_io.json"
        path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        assert response["proposal"]["state"] == expected_state
        if expected_state == "blocked":
            assert response["proposal"]["eligibility"] == "not_evaluated"
            assert response["proposal"]["runtime_execution"]["evaluation_executed"] is False
            assert response["proposal"]["blocking_findings"][0]["code"] == "missing_route_matrix_cell"
            assert response["session"]["accepted_plan_id"] == session["accepted_plan_id"]
        result = subprocess.run(["node", "-e", NODE, str(path)], cwd=ROOT, capture_output=True, text=True, check=False)
        (tmp_path / "frontend_assertions.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        assert result.returncode == 0, result.stdout + result.stderr
