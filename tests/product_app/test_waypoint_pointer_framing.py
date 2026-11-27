"""Execute the pointer test's real aiming function on offline DOM rectangles.

These are geometry/hit-test counterexamples, not browser acceptance evidence.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

BROWSER = Path(__file__).parent / "browser" / "test_waypoint_pointer_live.py"


def _aim(fixture):
    source = ast.parse(BROWSER.read_text(encoding="utf-8"))
    script = next(
        ast.literal_eval(node.value)
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "CONTEXT_RELEASE_GEOMETRY" for target in node.targets)
    )
    node = shutil.which("node")
    assert node, "Node is required to execute the actual browser aiming function"
    harness = r"""
      const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
      const rect=([left,top,right,bottom])=>({left,top,right,bottom,width:right-left,height:bottom-top});
      const within=(x,y,r)=>x>=r.left&&x<r.right&&y>=r.top&&y<r.bottom;
      const canvasRect=rect(input.fixture.canvas),panelRect=rect(input.fixture.panel);
      const canvas={tagName:'CANVAS',id:'',outerHTML:'<canvas></canvas>',closest:()=>null,getBoundingClientRect:()=>canvasRect};
      const panel={tagName:'SECTION',id:'context',outerHTML:'<section class="map-context-card"></section>',
        closest:()=>panel,getBoundingClientRect:()=>panelRect};
      const cover={tagName:'HEADER',id:'overlay',outerHTML:'<header></header>',closest:()=>null};
      const width=input.fixture.width||1440,height=input.fixture.height||900;
      const document={documentElement:{clientWidth:width},
        querySelector:selector=>selector==='.map-context-card'?panel:{open:input.fixture.text_open!==false},
        elementFromPoint:(x,y)=>{
          if(x<0||y<0||x>=width||y>=height)return null;
          if(input.fixture.cover_all||input.fixture.cover_panel&&within(x,y,panelRect))return cover;
          if(within(x,y,panelRect))return panel;
          return within(x,y,canvasRect)?canvas:null;
        }};
      const map={getCanvas:()=>canvas,getContainer:()=>({getBoundingClientRect:()=>canvasRect})};
      const run=new Function('map','document','innerHeight','scrollX','scrollY','return ('+input.script+')(map)');
      process.stdout.write(JSON.stringify(run(map,document,height,0,input.fixture.scroll_y||0)));
    """
    completed = subprocess.run(
        [node, "-e", harness],
        input=json.dumps({"script": script, "fixture": fixture}),
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _save(tmp_path, fixture, output):
    (tmp_path / "exact_framing_io.json").write_text(
        json.dumps(
            {"scope": "offline_dom_geometry_fixture", "real_browser_run": False, "input": fixture, "output": output},
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "fixture",
    [
        {"canvas": [320, 160, 1120, 790], "panel": [850, 210, 1100, 410]},
        {"canvas": [320, -450, 1120, 180], "panel": [850, -60, 1100, 120], "scroll_y": 610},
        {"canvas": [320, 160, 1120, 790], "panel": [850, 700, 1100, 990], "height": 768},
    ],
)
def test_native_targets_are_both_visible_and_match_real_element_hits(tmp_path, fixture):
    observed = _aim(fixture)
    assert observed["start"] and observed["end"] and observed["endpoint_is_context"] is True
    for key, samples, hit in (
        ("start", "visible_canvas_samples", "hit_is_canvas"),
        ("end", "visible_panel_samples", "hit_is_context"),
    ):
        point = observed[key]
        assert 0 <= point["x"] < observed["viewport"]["right"] and 0 <= point["y"] < observed["viewport"]["bottom"]
        matches = [sample for sample in observed[samples] if sample["x"] == point["x"] and sample["y"] == point["y"]]
        assert len(matches) == 1 and matches[0][hit] is True
    assert observed["native_text_route_open"] is True
    _save(tmp_path, fixture, observed)


def test_scrolled_partial_map_reproduces_legacy_sampler_miss_without_null_dereference(tmp_path):
    fixture = {"canvas": [320, -450, 1120, 180], "panel": [850, -60, 1100, 120], "scroll_y": 610}
    observed = _aim(fixture)
    assert len(observed["legacy_full_container_samples"]) == 16
    assert not any(row["hit_is_canvas"] for row in observed["legacy_full_container_samples"])
    assert observed["start"] and observed["end"]
    assert observed["visible_canvas"]["top"] == observed["visible_panel"]["top"] == 0
    assert observed["scroll"]["y"] == 610
    _save(tmp_path, fixture, observed)


@pytest.mark.parametrize(
    "fixture",
    [
        {"canvas": [320, -540, 1120, 90], "panel": [850, -500, 1100, -280], "scroll_y": 700},
        {"canvas": [320, -750, 1120, -120], "panel": [850, -700, 1100, -480]},
        {"canvas": [320, 160, 1120, 790], "panel": [850, 210, 1100, 410], "cover_all": True},
        {"canvas": [320, 160, 1120, 790], "panel": [850, 210, 1100, 410], "cover_panel": True},
        {"canvas": [1000, 160, 1600, 790], "panel": [1450, 210, 1580, 410], "width": 1024},
    ],
)
def test_no_visible_canvas_to_panel_path_returns_diagnostics_not_a_fabricated_gesture(tmp_path, fixture):
    observed = _aim(fixture)
    assert observed["start"] is None and observed["end"] is None and observed["endpoint_is_context"] is False
    assert set(observed) >= {
        "viewport",
        "scroll",
        "canvas",
        "panel",
        "visible_canvas",
        "visible_panel",
        "legacy_full_container_samples",
    }
    assert len(observed["legacy_full_container_samples"]) == 16
    _save(tmp_path, fixture, observed)


def test_framing_does_not_replace_or_weaken_native_capture_and_bubble_assertions():
    source = BROWSER.read_text(encoding="utf-8")
    section = source[
        source.index("def release_pan_over_context(") : source.index("def aim_marker_release_over_context(")
    ]
    assert section.index("scroll_into_view_if_needed()") < section.index("page.mouse.down()")
    assert section.index("window.__e2eLastContextRelease=value") < section.index('assert gesture["start"]')
    assert 'for phase in ("capture", "bubble")' in section
    assert 'released[0]["target_is_canvas"] and released[0]["hit_is_context"]' in section
    assert 'assert len(released) == 1 and released[0]["trusted"]' in section
    assert 'assert all(event in events for event in ("dragstart", "dragend", "moveend", "idle"))' in section
    assert "page.mouse.up()" in section and "finally:" in section
    for forbidden in (".flyTo(", ".easeTo(", ".jumpTo(", ".panTo(", ".panBy(", "dispatchEvent("):
        assert forbidden not in section


def test_text_route_closes_natively_only_after_overlay_release_evidence():
    source = BROWSER.read_text(encoding="utf-8")
    completed = source.index('assert capture("canvas_overlay_release") == parent')
    close = source.index('text_route.locator("summary").click()', completed)
    focus = source.index("pixels = focus_with_pointer(page, coordinate)", close)
    assert completed < close < focus
    transition = source[completed:focus]
    assert 'assert text_route.evaluate("element=>element.open") is True' in transition
    assert 'framing["click"]["trusted"] is True' in transition
    assert 'framing["after"]["native_text_route_open"] is False' in transition
    assert "assert _map_ack(page) == parent" in transition
    assert '_write_json(evidence / "canvas_focus_transition.json", framing)' in transition
    assert "dispatchEvent" not in transition and ".open =" not in transition


@pytest.mark.parametrize("failure", [None, "before", "after", "changed_during_screenshot"])
def test_actual_capture_function_preserves_overlay_before_each_assertion(tmp_path, failure):
    tree = ast.parse(BROWSER.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "capture")
    before = {
        "plan_id": "parent",
        "plan_content_hash": "parent-hash",
        "geometry_hash": "geometry",
        "accepted_marker_ids": ["stop"],
        "draft_routes": {"features": [{"route_leg_id": "old-route"}]},
    }
    after = deepcopy(before)
    if failure == "before":
        before["invalid"] = True
    elif failure == "after":
        after["invalid"] = True
        after["draft_routes"]["features"] = [{"route_leg_id": "wrong-after-route"}]
    elif failure == "changed_during_screenshot":
        after["draft_routes"]["features"] = [{"route_leg_id": "changed-route"}]
    operation = {
        "operation_id": "expected-operation",
        "parameters": {"affected_route_legs": [{"route_leg_id": "expected-route"}]},
    }
    ack = {
        "active_plan_id": "parent",
        "rendered_map_plan_id": "parent",
        "active_plan_content_hash": "parent-hash",
        "rendered_geometry_hash": "geometry",
        "rendered_marker_ids": ["stop"],
    }
    queue, writes, events = [before, after], [], []

    def write(path, value):
        writes.append(deepcopy(value))
        events.append("write_after" if value["after"] is not None else "write_before")

    def check(value, expected):
        assert expected == operation
        assert writes and writes[-1]["expected_operation"] == operation
        assert value == writes[-1]["before" if len(queue) else "after"]
        events.append("assert")
        assert not value.get("invalid"), "deliberate overlay mismatch"

    namespace = {
        "overlay_observation": lambda *args: queue.pop(0),
        "assert_overlay": check,
        "_write_json": write,
        "_capture_ready": lambda *args: events.append("screenshot"),
        "_map_ack": lambda page: ack,
        "evidence": tmp_path,
        "checkpoints": {},
        "page": SimpleNamespace(locator=lambda selector: SimpleNamespace(inner_text=lambda: "draft context")),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(BROWSER), "exec"), namespace)
    if failure:
        with pytest.raises(AssertionError):
            namespace["capture"]("undo", operation)
        assert writes[-1]["assertions_passed"] is False
    else:
        assert namespace["capture"]("undo", operation) == ack
        assert writes[-1]["assertions_passed"] is True
    assert writes[0]["before"] == before and writes[0]["after"] is None
    assert events[:2] == ["write_before", "assert"]
    if failure != "before":
        assert writes[-1]["after"] == after
        assert events[2:5] == ["screenshot", "write_after", "assert"]
    (tmp_path / "exact_capture_retention_io.json").write_text(
        json.dumps(
            {
                "scope": "unit_actual_extracted_capture_function_not_browser_evidence",
                "provider_calls": 0,
                "input": {"failure": failure, "operation": operation, "before": before, "after": after},
                "output": {"events": events, "retained_writes": writes},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_undo_waits_for_exact_server_impact_revision_and_visible_draft_before_map_capture():
    source = BROWSER.read_text(encoding="utf-8")
    block = source[source.index("for undo_index,") : source.index('phase = "reload_after_undo"')]
    assert 'response.url.endswith("/draft/impact-preview")' in block
    assert 'response.request.post_data_json["expected_revision"] == expected_revision' in block
    assert block.index("as impact:") < block.index('page.locator("#inline-undo").click()')
    assert 'payload["session"]["draft"] == ([remaining] if remaining else [])' in block
    assert 'impact.value.json()["session_revision"] == expected_revision' in block
    acknowledgement = re.search(r'to_have_text\(\s*"1 draft change" if remaining else "0 draft changes"', block)
    assert acknowledgement is not None
    assert acknowledgement.start() < block.index("assert capture(label, remaining) == parent")
    assert "wait_for_timeout" not in block
