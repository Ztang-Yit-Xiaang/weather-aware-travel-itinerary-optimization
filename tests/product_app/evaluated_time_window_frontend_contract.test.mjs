import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const contractPath = process.env.ACTUAL_EVALUATED_TIME_WINDOW_CONTRACT_PATH;
assert.ok(contractPath, "ACTUAL_EVALUATED_TIME_WINDOW_CONTRACT_PATH is required");
const actual = JSON.parse(fs.readFileSync(contractPath, "utf8"));
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const appPath = path.join(root, "src", "itinerary_system", "product_app", "static", "js", "app.js");
const source = fs.readFileSync(appPath, "utf8");
const contractCode = source.slice(
  source.indexOf("const TYPED_EDIT_OPERATIONS"),
  source.indexOf("function safeExternalLink"),
);
const state = { workspace: actual.workspace, session: actual.eligible.before_session };
const normalizers = new Function(
  "state",
  `${contractCode}\nreturn {
    normalizeTypedEditCapabilities, normalizeDraftImpactPreview, normalizeEvaluatedPreviewResponse,
    normalizeRouteValidation, normalizeTimeWindowPlanDiff, normalizeTimeWindowScheduleImpact,
    normalizeTimeWindowCertificateScheduleEvidence, normalizeTimeWindowRouteEvidence,
    normalizeTimeWindowScheduleAccounting, normalizeTimeWindowTrace,
    normalizePreviewRepair, normalizePreviewGeography, durationRouteEvidenceUnchanged,
    previewProposalBinding, exactKeys, sameJson, canonicalJson, computedDraftContentHash,
    computedStableContentHash, draftHashRecords,
  };`,
)(state);

assert.ok(normalizers.normalizeTypedEditCapabilities(
  structuredClone(actual.workspace.typed_edit_capabilities),
));
assert.ok(normalizers.normalizeDraftImpactPreview(structuredClone(actual.eligible.impact)));

const proposalKeys = [
  "schema_version", "state", "reason", "eligibility", "provenance", "execution_mode",
  "expected_revision", "session_revision", "draft_content_hash", "draft_operation_ids",
  "message", "parent_plan_id", "parent_plan_content_hash", "child_plan_id",
  "child_plan_content_hash", "certificate_id", "certificate_content_hash", "diff_id",
  "diff_content_hash", "diff_identity", "evaluation_status", "certificate_identity",
  "route_validation", "route_validation_identity", "evidence", "geography_plan",
  "compiled_request", "repair", "plan_diff", "schedule_impact", "schedule_impact_identity",
  "certificate_schedule_evidence", "decision_eligible", "ranking_eligible",
  "acceptance_eligible", "acceptance_blocking_code",
];

const normalizeCase = async (row, label) => {
  const proposal = row.preview.proposal;
  assert.ok(normalizers.exactKeys(proposal, proposalKeys), `${label} proposal keys`);
  assert.ok(normalizers.exactKeys(row.preview.session.proposal, proposalKeys), `${label} session proposal keys`);
  assert.equal(proposal.draft_content_hash, await normalizers.computedDraftContentHash(row.expected.draft));
  assert.equal(
    normalizers.canonicalJson(proposal.compiled_request.operations),
    normalizers.canonicalJson(normalizers.draftHashRecords(row.expected.draft)),
    `${label} compiled request`,
  );
  const routeValidation = normalizers.normalizeRouteValidation(proposal.route_validation);
  assert.ok(routeValidation, `${label} route validation`);
  const planDiff = await normalizers.normalizeTimeWindowPlanDiff(
    proposal.plan_diff, proposal, row.expected.draft,
  );
  assert.ok(planDiff, `${label} PlanDiff v3`);
  assert.ok(normalizers.normalizeTimeWindowRouteEvidence(
    proposal.schedule_impact.route_leg_evidence, routeValidation,
  ), `${label} exact route-leg evidence`);
  const commonAccounting = {
    evaluatorVersion: proposal.schedule_impact.evaluator_version,
    routeMatrixId: proposal.schedule_impact.route_matrix_id,
    routeValidation,
    configuredDefaultVisitMinutes: proposal.schedule_impact.configured_default_visit_minutes,
    configuredDayStartMinute: proposal.schedule_impact.configured_day_start_minute,
    routeLegEvidence: proposal.schedule_impact.route_leg_evidence,
  };
  assert.ok(await normalizers.normalizeTimeWindowScheduleAccounting(
    proposal.schedule_impact.parent,
    { ...commonAccounting, planId: proposal.parent_plan_id, planContentHash: proposal.parent_plan_content_hash },
  ), `${label} parent accounting`);
  assert.ok(await normalizers.normalizeTimeWindowScheduleAccounting(
    proposal.schedule_impact.child,
    { ...commonAccounting, planId: proposal.child_plan_id, planContentHash: proposal.child_plan_content_hash },
  ), `${label} child accounting`);
  const schedule = await normalizers.normalizeTimeWindowScheduleImpact(
    proposal.schedule_impact, proposal.schedule_impact_identity,
    proposal, planDiff, routeValidation,
  );
  assert.ok(schedule, `${label} schedule impact`);
  assert.ok(await normalizers.normalizeTimeWindowCertificateScheduleEvidence(
    proposal.certificate_schedule_evidence, proposal, schedule,
  ), `${label} certificate schedule evidence`);
  assert.ok(normalizers.normalizePreviewRepair(proposal.repair, proposal), `${label} repair`);
  assert.ok(normalizers.normalizePreviewGeography(
    proposal.geography_plan, proposal, routeValidation, row.expected.draft,
  ), `${label} geography`);
  assert.ok(normalizers.durationRouteEvidenceUnchanged(proposal.geography_plan), `${label} unchanged routes`);
  assert.deepEqual(
    normalizers.previewProposalBinding(row.preview.session.proposal),
    normalizers.previewProposalBinding(proposal),
    `${label} session proposal binding`,
  );
  const full = await normalizers.normalizeEvaluatedPreviewResponse(
    structuredClone(row.preview), structuredClone(row.expected),
  );
  assert.ok(full, `${label} full response`);
  assert.equal(full.proposal.ranking_eligible, false, `${label} has no ranking evidence`);
  assert.equal(full.proposal.acceptance_eligible, false, `${label} W5 acceptance disabled`);
  assert.equal(full.proposal.acceptance_blocking_code, "acceptance_not_enabled_until_w5");
  assert.equal(full.proposal.repair.tradeoffs.utility_retained, null);
  assert.equal(full.proposal.repair.tradeoffs.weather_risk_delta, null);
  return full;
};

const eligible = await normalizeCase(actual.eligible, "eligible earliest window");
assert.equal(eligible.proposal.state, "eligible");
assert.equal(eligible.proposal.decision_eligible, true);
const eligibleTarget = eligible.proposal.schedule_impact.child.stop_traces.find(
  (trace) => trace.stop_id === "stearns_wharf",
);
assert.equal(eligibleTarget.time_window_constraint.earliest_arrival, "10:00");
assert.equal(eligibleTarget.time_window_constraint.latest_departure, null);
assert.equal(eligibleTarget.time_window_source, "trip_specific_user_constraint");
assert.ok(eligibleTarget.required_window_wait_minutes >= 0);
assert.equal(eligibleTarget.latest_departure_status, "not_set");
assert.equal(eligibleTarget.latest_departure_overrun_minutes, null);

const latest = await normalizeCase(actual.latest_ineligible, "latest-departure ineligible");
assert.equal(latest.proposal.state, "ineligible");
assert.equal(latest.proposal.decision_eligible, false);
assert.deepEqual(latest.proposal.schedule_impact.child.blocking_codes, [
  "stop_time_window_latest_departure_exceeded",
]);
const latestTarget = latest.proposal.schedule_impact.child.stop_traces.find(
  (trace) => trace.stop_id === "stearns_wharf",
);
assert.equal(latestTarget.latest_departure_status, "violated");
assert.ok(latestTarget.latest_departure_overrun_minutes > 0);
assert.deepEqual(latestTarget.failure_codes, ["stop_time_window_latest_departure_exceeded"]);

const dayOverrun = await normalizeCase(actual.day_overrun, "earliest-wait day overrun");
assert.equal(dayOverrun.proposal.state, "ineligible");
assert.ok(dayOverrun.proposal.schedule_impact.child.blocking_codes.includes("day_time_exceeded"));
assert.ok(dayOverrun.proposal.schedule_impact.child.metrics.day_4_overrun_minutes > 0);

const mixedState = { workspace: actual.workspace, session: actual.mixed.session };
const mixedNormalizer = new Function(
  "state", `${contractCode}\nreturn {normalizeDraftImpactPreview};`,
)(mixedState);
assert.ok(mixedNormalizer.normalizeDraftImpactPreview(structuredClone(actual.mixed.impact)));
assert.equal(actual.mixed.impact.summary.can_run_evaluated_preview, false);
assert.ok(actual.mixed.impact.summary.blocking_codes.includes(
  "draft_evaluated_operation_combination_unsupported",
));

const contentPayload = (value) => {
  const payload = structuredClone(value);
  delete payload.content_hash;
  return payload;
};
const rehashPlanDiff = async (proposal) => {
  const hash = await normalizers.computedStableContentHash(proposal.plan_diff);
  proposal.diff_content_hash = hash;
  proposal.diff_identity.content_hash = hash;
};
const rehashSchedule = async (proposal) => {
  for (const side of ["parent", "child"]) {
    proposal.schedule_impact[side].content_hash = await normalizers.computedStableContentHash(
      contentPayload(proposal.schedule_impact[side]),
    );
  }
  const hash = await normalizers.computedStableContentHash(contentPayload(proposal.schedule_impact));
  proposal.schedule_impact.content_hash = hash;
  proposal.schedule_impact_identity.content_hash = hash;
};
const rehashCertificateSchedule = async (proposal) => {
  proposal.certificate_schedule_evidence.content_hash = await normalizers.computedStableContentHash(
    contentPayload(proposal.certificate_schedule_evidence),
  );
};
const forge = (base, mutate) => {
  const payload = structuredClone(base);
  mutate(payload.proposal);
  payload.session.proposal = structuredClone(payload.proposal);
  return payload;
};
const coherentForge = async (base, mutate, rehash) => {
  const payload = structuredClone(base);
  await mutate(payload.proposal);
  await rehash(payload.proposal);
  payload.session.proposal = structuredClone(payload.proposal);
  return payload;
};
const change = (proposal) => proposal.plan_diff.time_window_changes[0];
const targetTrace = (proposal, side = "child") => proposal.schedule_impact[side].stop_traces.find(
  (trace) => trace.stop_id === "stearns_wharf",
);

const ordinaryForgeries = [
  ["v2_time_window_smuggling", (proposal) => { proposal.plan_diff.schema_version = "plan-diff-v2"; }],
  ["time_shift_smuggling", (proposal) => { proposal.plan_diff.time_shifts.push({ forged: true }); }],
  ["duration_change_smuggling", (proposal) => { proposal.plan_diff.duration_changes = []; }],
  ["raw_bound_mismatch", (proposal) => { change(proposal).to_earliest_arrival = "11:00"; }],
  ["constraint_semantics", (proposal) => {
    change(proposal).to_constraint.early_arrival_policy = "road_arrival_not_before";
  }],
  ["compiled_request", (proposal) => {
    proposal.compiled_request.operations[0].parameters.earliest_arrival = "11:00";
  }],
  ["schedule_hash", (proposal) => { proposal.schedule_impact.content_hash = "0".repeat(16); }],
  ["schedule_identity", (proposal) => {
    proposal.schedule_impact_identity.route_matrix_id = `route_matrix_${"1".repeat(16)}`;
  }],
  ["missing_risk_as_zero", (proposal) => { proposal.repair.tradeoffs.weather_risk_delta = 0; }],
  ["child_route_duration", (proposal) => {
    proposal.geography_plan.validated_legs.features[0].properties.duration_s += 1;
  }],
  ["ranking", (proposal) => { proposal.ranking_eligible = true; }],
  ["acceptance", (proposal) => { proposal.acceptance_eligible = true; }],
];
for (const [name, mutate] of ordinaryForgeries) {
  assert.equal(
    await normalizers.normalizeEvaluatedPreviewResponse(
      forge(actual.eligible.preview, mutate), actual.eligible.expected,
    ),
    null,
    name,
  );
}

const rehashBoth = async (proposal) => {
  await rehashSchedule(proposal);
  await rehashCertificateSchedule(proposal);
};
const coherentForgeries = [
  ["rehashed_bounds", actual.eligible, (proposal) => {
    change(proposal).to_constraint.earliest_arrival = "11:00";
    change(proposal).to_earliest_arrival = "11:00";
  }, rehashPlanDiff],
  ["rehashed_constraint_source", actual.eligible, (proposal) => {
    targetTrace(proposal).time_window_source = "place_opening_hours";
  }, rehashSchedule],
  ["rehashed_owner_cost", actual.eligible, (proposal) => {
    change(proposal).owner_strength = "soft";
    change(proposal).cost = 2.5;
    proposal.plan_diff.weighted_edit_cost = 2.5;
  }, rehashPlanDiff],
  ["rehashed_day_start", actual.eligible, (proposal) => {
    proposal.schedule_impact.configured_day_start_minute = 541;
  }, rehashSchedule],
  ["rehashed_day_start_and_all_arrivals", actual.eligible, (proposal) => {
    proposal.schedule_impact.configured_day_start_minute = 541;
    for (const side of ["parent", "child"]) {
      for (const trace of proposal.schedule_impact[side].stop_traces) {
        trace.road_arrival_minute += 1;
        trace.service_start_minute += 1;
        trace.departure_minute += 1;
      }
    }
  }, rehashSchedule],
  ["rehashed_route_duration", actual.eligible, (proposal) => {
    const leg = proposal.schedule_impact.route_leg_evidence[0];
    leg.duration_minutes += 1;
    for (const side of ["parent", "child"]) {
      const trace = proposal.schedule_impact[side].stop_traces.find(
        (item) => item.day === leg.day && item.stop_id === leg.destination_id,
      );
      trace.incoming_travel_minutes += 1;
    }
  }, rehashSchedule],
  ["rehashed_shifted_arrivals", actual.eligible, (proposal) => {
    for (const side of ["parent", "child"]) {
      for (const trace of proposal.schedule_impact[side].stop_traces.filter((item) => item.day === 1)) {
        trace.road_arrival_minute += 1;
        trace.service_start_minute += 1;
        trace.departure_minute += 1;
      }
    }
  }, rehashSchedule],
  ["rehashed_invented_opening_evidence", actual.eligible, (proposal) => {
    for (const side of ["parent", "child"]) {
      const trace = proposal.schedule_impact[side].stop_traces[0];
      trace.opening_start_minute = trace.road_arrival_minute - 1;
      trace.opening_end_minute = 1439;
      trace.opening_wait_minutes = 0;
      trace.opening_window_source = "plan_stop_fields:opening_start:opening_end";
      proposal.schedule_impact[side].missing_opening_window_stop_ids = proposal.schedule_impact[
        side
      ].missing_opening_window_stop_ids.filter((stopId) => stopId !== trace.stop_id);
    }
    proposal.certificate_schedule_evidence.schedule_metrics.schedule_missing_opening_window_count -= 1;
  }, rehashBoth],
  ["rehashed_required_wait", actual.eligible, (proposal) => {
    const trace = targetTrace(proposal);
    trace.required_window_wait_minutes += 1;
    trace.service_start_minute += 1;
    trace.departure_minute += 1;
  }, rehashSchedule],
  ["rehashed_latest_overrun", actual.latest_ineligible, (proposal) => {
    const trace = targetTrace(proposal);
    trace.latest_departure_status = "satisfied";
    trace.latest_departure_overrun_minutes = 0;
    trace.failure_codes = [];
  }, rehashSchedule],
  ["rehashed_warning_code", actual.eligible, (proposal) => {
    proposal.certificate_schedule_evidence.warning_codes = ["forged_warning"];
  }, rehashCertificateSchedule],
  ["rehashed_failure_code", actual.latest_ineligible, (proposal) => {
    proposal.certificate_schedule_evidence.failure_codes = ["forged_failure"];
  }, rehashCertificateSchedule],
  ["rehashed_status", actual.latest_ineligible, (proposal) => {
    proposal.schedule_impact.child.status = "passed";
  }, rehashSchedule],
  ["rehashed_missing_trace", actual.eligible, (proposal) => {
    proposal.schedule_impact.child.stop_traces.pop();
  }, rehashSchedule],
  ["rehashed_visit_source", actual.eligible, (proposal) => {
    targetTrace(proposal).visit_duration_source = "visit_duration_minutes";
  }, rehashSchedule],
];
for (const [name, row, mutate, rehash] of coherentForgeries) {
  assert.equal(
    await normalizers.normalizeEvaluatedPreviewResponse(
      await coherentForge(row.preview, mutate, rehash), row.expected,
    ),
    null,
    name,
  );
}

console.log("evaluated time-window frontend contract and forgeries passed");
