"""Portable synthetic receipt tests, not real browser or manifest certification."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_e2e1_runtime_gate_coverage import bundle, gates

pytest_plugins = ["recorded_runtime_fixture"]


def _artifacts(context, fault):
    plan = 'child' if context == 'accepted_child' else 'parent'
    identity = {'active_plan_id': plan, 'active_plan_content_hash': 'plan_hash',
        'rendered_map_plan_id': plan, 'rendered_geometry_hash': 'geometry', 'rendered_marker_ids': ['wharf']}
    session = {'session_id': 'owner', 'revision': 7, 'accepted_plan_id': plan,
        'draft': [], 'proposal': None, 'conversation_id': 'old'}
    turns, records, clicks = [], [], []
    for index in range(11):
        request = {'expected_revision': 7, 'client_message_id': 'client_message_' + f'{index:032x}',
            'message': 'Synthetic message ' + str(index)}
        current = {**session, 'conversation_id': 'new' if index == 10 else 'old'}
        turn = {'client_message_id': request['client_message_id'], 'user_message': request['message'], 'context_revision': 7}
        events = [{'schema_version': 'copilot-stream-v1', 'type': kind, 'sequence': offset,
            'client_message_id': request['client_message_id'], 'request_id': 'copilot_request_' + f'{index:032x}'}
            for offset, kind in enumerate(('message.accepted', 'assistant.status', 'turn.completed'), 1)]
        events[1].update(state='interpreting', message='Reading the current itinerary and validating a typed suggestion.')
        events[2]['payload'] = {'turn': turn, 'session': current}
        turns.append({'input': request, 'events': events, 'output': events[-1]})
        records.append({'method': 'POST', 'url': 'http://127.0.0.1:8127/api/sessions/owner/copilot/messages',
            'response_status_code': 200, 'response_capture_complete': True, 'request_body_utf8': json.dumps(request),
            'response_body_utf8': ''.join('data: ' + json.dumps(event) + '\n\n' for event in events)})
        clicks.append({'id': 'copilot-submit', 'trusted': True, 'aria_label': 'Send message', 'at': index * 10})
    deletion = {'method': 'DELETE', 'url': 'http://127.0.0.1:8127/api/sessions/owner/conversation',
        'response_status_code': 200, 'response_capture_complete': True,
        'request_body_utf8': json.dumps({'expected_revision': 7}),
        'response_body_utf8': json.dumps({'deleted': True, 'session': {**session, 'conversation_id': None}})}
    records.insert(10, deletion)
    clicks.insert(10, {'id': 'delete-current-conversation', 'trusted': True, 'aria_label': None, 'at': 95})
    before = {'accepted_plan_id': plan, 'accepted_plan_content_hash': 'plan_hash', 'draft_operations': [],
        'conversation_memory': {'proposed_preferences': [{'preference': 'preserve_experience'}], 'pending_clarification': None}}
    after = deepcopy(before)
    after['conversation_memory']['proposed_preferences'] = []
    if fault == 'untrusted_send':
        clicks[0]['trusted'] = False
    elif fault == 'untrusted_delete':
        clicks[10]['trusted'] = False
    elif fault == 'missing_send':
        clicks.pop(0)
    elif fault == 'stream_changed':
        turns[0]['output']['payload']['turn']['user_message'] = 'Different'
    elif fault == 'failed_delete':
        deletion['response_body_utf8'] = json.dumps({'deleted': False, 'session': {**session, 'conversation_id': None}})
    elif fault == 'foreign_owner':
        deletion['url'] = deletion['url'].replace('/owner/', '/foreign/')
    elif fault == 'wrong_revision':
        deletion['request_body_utf8'] = json.dumps({'expected_revision': 6})
    elif fault == 'wrong_reset_plan':
        after['accepted_plan_id'] = 'other'
    elif fault == 'memory_retained':
        after['conversation_memory']['proposed_preferences'] = [{'preference': 'preserve_experience'}]
    elif fault == 'bad_native_order':
        clicks[10]['at'] = 105
    artifacts = {'trace.json': {'context': context, 'initial': identity, 'final': identity,
        'assertions_passed': True, 'chat_turn_count': 11}, 'exact_chat_io.json': turns, 'control_clicks.json': clicks,
        'context_corrected.json': before, 'context_reset.json': after,
        'http_index.json': [{'file': f'http_{index}.json'} for index in range(len(records))],
        **{f'http_{index}.json': row for index, row in enumerate(records)}}
    if context == 'accepted_child':
        artifacts.update({
            'accepted_baseline_artifacts.json': {'context': context, 'assertions_passed': True,
                'accepted_map': identity, 'original': {'active_plan_id': 'parent'}},
            'accepted_setup_accept_http.json': {'response_body_utf8': json.dumps({'result': {'decision': {
                'selected_plan_id': plan, 'selected_plan_content_hash': 'plan_hash', 'decision_id': 'decision'}}})},
            'accepted_setup_restored_http.json': {'response_body_utf8': json.dumps({'session': session,
                'workspace': {'accepted_plan_pointer': {'decision_id': 'decision'}}})},
        })
    base = 'conversation/preference_memory' + ('_accepted_child' if context == 'accepted_child' else '') + '/'
    return {base + name: value for name, value in artifacts.items()}


@pytest.mark.parametrize('context', ['original', 'accepted_child'])
@pytest.mark.parametrize('fault', [None, 'untrusted_send', 'untrusted_delete', 'missing_send', 'stream_changed',
    'failed_delete', 'foreign_owner', 'wrong_revision', 'wrong_reset_plan', 'memory_retained', 'bad_native_order'])
def test_existing_conversation_submit_and_current_deletion_require_exact_control_results(tmp_path, context, fault):
    suites = [bundle(tmp_path, 'conversation', _artifacts(context, fault))]
    proofs, unavailable = gates.existing_control_proofs(gates.Receipts(tmp_path, suites))
    bindings = ('chat_submit', 'transcript_delete_current')
    if fault is None:
        for binding in bindings:
            assert (binding, context) in proofs
            assert proofs[(binding, context)]['detail']['old_conversation_id'] == 'old'
            assert proofs[(binding, context)]['detail']['new_conversation_id'] == 'new'
            assert proofs[(binding, context)]['evidence']
            assert (binding, 'accepted_child' if context == 'original' else 'original') not in proofs
    else:
        assert not any((binding, context) in proofs for binding in bindings)
        assert any(row['context'] == context and set(row['bindings']) == set(bindings) for row in unavailable)


@pytest.mark.parametrize('context,stage_first', [('original', True), ('original', False),
    ('accepted_child', True), ('accepted_child', False)])
def test_recorded_combined_controls_bind_both_entry_paths_and_plan_contexts(context, stage_first, recorded_runtime_inputs):
    from verify_e2e1_self_managed import combined_chat_control_receipt

    name = 'combined_chat' + ('' if stage_first else '_one_click') + ('_accepted_child' if context == 'accepted_child' else '')
    run = 'combined_continuation_probe_20261007_v4' if context == 'accepted_child' else 'combined_continuation_probe_20261007_v3'
    root = recorded_runtime_inputs / 'results/e2e' / run / 'conversation' / name
    proof = combined_chat_control_receipt(lambda filename: json.loads((root / filename).read_text()), name=name)
    assert proof['context'] == context
    assert set(proof['variants']) == ({'proposal_combined_add', 'proposal_combined_preview_staged', 'proposal_combined_added_disabled'}
        if stage_first else {'proposal_combined_preview_empty', 'proposal_combined_added_disabled'})


@pytest.mark.parametrize('context', ['original', 'accepted_child'])
@pytest.mark.parametrize('fault', ['untrusted', 'partial_batch', 'wrong_context', 'wrong_parent', 'stale_revision',
    'changed_stage_map', 'wrong_child', 'enabled_added', 'extra_mutation', 'wrong_acceptance', 'missing_reload', 'changed_duration'])
def test_recorded_combined_controls_reject_disconnected_artifacts(context, fault, recorded_runtime_inputs):
    # Recorded semantic counterexamples only; not current-source browser proof.
    import verify_e2e1_self_managed as verifier

    name = 'combined_chat' + ('_one_click_accepted_child' if context == 'accepted_child' else '')
    run = 'combined_continuation_probe_20261007_v4' if context == 'accepted_child' else 'combined_continuation_probe_20261007_v3'
    root = recorded_runtime_inputs / 'results/e2e' / run / 'conversation' / name

    def read(filename):
        value = json.loads((root / filename).read_text())
        if fault == 'untrusted' and filename == 'combined_controls_before_reload.json':
            next(row for row in value['clicks'] if row['label'] == 'Preview combined repair')['trusted'] = False
        elif fault == 'partial_batch' and filename == 'staging_io.json':
            value['input']['operations'].pop()
        elif fault == 'wrong_context' and filename == 'context_combined_before_stage.json':
            value['allowed_stop_ids'] = []
        elif fault == 'wrong_parent' and filename == 'trace.json':
            value['initial']['active_plan_id'] = 'foreign_parent'
        elif fault == 'stale_revision' and filename == 'before_preview_observation.json':
            value['request']['expected_revision'] -= 1
        elif fault == 'changed_stage_map' and filename == 'before_preview_observation.json':
            value['active_plan_id'] = 'foreign_child'
        elif fault == 'wrong_child' and filename == 'trace.json':
            value['rendered']['active_plan_id'] = value['initial']['active_plan_id']
        elif fault == 'enabled_added' and filename == 'combined_added_disabled.json':
            value['disabled'] = False
        elif fault == 'extra_mutation' and filename == 'http_index.json':
            value.append(next(row for row in value if row['url'].endswith('/draft/operations')))
        elif fault == 'wrong_acceptance' and filename == 'acceptance_response.json':
            value['session']['accepted_plan_id'] = 'foreign_child'
        elif fault == 'missing_reload' and filename == 'http_index.json':
            value = [row for row in value if not (row['method'] == 'GET' and '/api/sessions/' in row['url']
                and row['url'].rsplit('/', 1)[-1].startswith('session_'))]
        elif fault == 'changed_duration' and filename == 'preview_response.json':
            next(stop for stop in value['proposal']['runtime_execution']['child_plan']['selected_stops']
                if stop['stop_id'] == ('pfeiffer_big_sur_state_park' if context == 'accepted_child' else 'griffith_observatory'))['visit_duration_minutes'] = 91
        return value

    with pytest.raises(verifier.core.E2E1VerificationError):
        verifier.combined_chat_control_receipt(read, name=name)


def test_combined_control_variants_are_required_not_optional_registry_credit():
    registry = json.loads((Path(__file__).resolve().parents[2] / 'configs/e2e1_interaction_coverage.json').read_text())
    binding = next(row for row in registry['source_bindings'] if row['binding_id'] == 'chat_generated_action')
    assert set(binding['required_journey_variants']) >= {'proposal_combined_add', 'proposal_combined_preview_staged',
        'proposal_combined_preview_empty', 'proposal_combined_added_disabled'}


def test_recorded_selection_generation_rejects_changed_sources(recorded_runtime_inputs):
    """Exercise the real case verifier, not certification of an old whole bundle."""
    import verify_e2e1_self_managed as verifier
    from verify_e2e1_selection_generation import verify_selection_generation

    case = recorded_runtime_inputs / 'results/e2e/combined_continuation_browser_20261007_v2/conversation/selection_conversation_generation'
    identities = json.loads((case / 'source_identity_before.json').read_text())
    trace = json.loads((case / 'trace.json').read_text())
    turns = json.loads((case / 'exact_chat_io.json').read_text())
    records = verifier._http(case, 'http://127.0.0.1:8127')
    current = all(verifier.core._sha256(verifier.core.ROOT / path) == digest for path, digest in identities.items())
    if current:
        assert verify_selection_generation(case, trace, records, turns) == ('before', 'before_late_selection', 'after_late_selection')
    else:
        with pytest.raises(verifier.core.E2E1VerificationError, match='selection_generation_source_changed'):
            verify_selection_generation(case, trace, records, turns)
