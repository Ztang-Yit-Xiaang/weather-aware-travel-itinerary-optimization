import {
  disposeGeographicMap,
  normalizePoiCandidateResponse,
  reduceConfirmedMapDraftOperations,
  renderGeographicMap,
} from "./map.js?v=20260810-stability5";
import { createCopilotController } from "./copilot.js?v=20260805-w4-copilot1";
import {
  disposeCompareWorkspace,
  renderCompareWorkspace,
  resolveCompareSelection,
} from "./compare.js?v=20260810-stability5";

const state = {
  health: null,
  mapConfig: null,
  workspace: null,
  session: null,
  token: null,
  registry: null,
  route: normalizeRoute(location.pathname),
  mapInteraction: {
    mode: "select",
    selectionKind: null,
    selectedStopPlanId: null,
    selectedRouteLeg: null,
    selectedWaypointId: null,
    exploratoryPin: null,
    snapPreview: null,
    snapState: "idle",
  },
  poiDiscovery: {
    status: "idle",
    response: null,
    selectedCandidateId: null,
    replacementTargetId: null,
    showLayer: true,
    role: "",
    maxDetourMinutes: 60,
    limit: 5,
    error: null,
  },
  typedImpact: {
    status: "idle",
    response: null,
    error: null,
  },
};
let pendingSelection = null;

const SESSION_STORAGE_KEY = "itinerary-repair-copilot.session.v1";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const copilot = createCopilotController({
  getState: () => state,
  api,
  selectContext,
  addDraft: addCopilotDraft,
  previewDraft,
  navigate,
  toast,
  renderAll,
});

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function unavailable(value, formatter = (item) => String(item)) {
  return value === null || value === undefined || value === "" ? "Unavailable" : formatter(value);
}

const TYPED_EDIT_OPERATIONS = [
  "set_stop_role",
  "set_stop_day",
  "set_stop_order",
  "set_stop_duration",
  "set_stop_time_window",
  "set_stop_commitment",
  "set_attribute_constraint",
  "change_route_preference",
  "report_route_issue",
];
const TYPED_EDIT_VOCABULARIES = {
  stop_roles: ["attraction", "activity", "meal", "lodging", "transport_hub", "rest_stop", "scenic_stop", "route_waypoint", "origin", "destination"],
  duration_modes: ["exact", "preferred", "minimum", "maximum", "range"],
  commitment_strengths: ["optional", "prefer_keep", "strong_preference", "must_keep", "booked", "excluded"],
  scope_lifetimes: ["current_draft_only", "current_repair_session", "remainder_of_trip"],
  attributes: ["existence", "day", "duration", "sequence_order", "role"],
  route_issue_types: ["suspected_closure", "access_problem", "route_quality"],
};

function exactKeys(value, expected) {
  return value && typeof value === "object" && !Array.isArray(value)
    && Object.keys(value).sort().join("|") === [...expected].sort().join("|");
}

function exactStringArray(value, expected) {
  return Array.isArray(value)
    && value.length === expected.length
    && value.every((item, index) => item === expected[index]);
}

function normalizeTypedEditCapabilities(value) {
  if (!exactKeys(value, ["schema_version", "vocabularies", "operations"])
      || value.schema_version !== "product-typed-edit-capabilities-v1"
      || !exactKeys(value.vocabularies, Object.keys(TYPED_EDIT_VOCABULARIES))) return null;
  if (!Object.entries(TYPED_EDIT_VOCABULARIES).every(
    ([name, expected]) => exactStringArray(value.vocabularies[name], expected),
  )) return null;
  if (!exactKeys(value.operations, TYPED_EDIT_OPERATIONS)) return null;
  for (const [name, capability] of Object.entries(value.operations)) {
    const protectedOperation = ["set_stop_commitment", "set_attribute_constraint"].includes(name);
    const orderOperation = name === "set_stop_order";
    const roleOperation = name === "set_stop_role";
    const durationOperation = name === "set_stop_duration";
    const timeWindowOperation = name === "set_stop_time_window";
    const expectedKeys = [
      "enabled", "feedback_tier", "preview_executable", "evaluated_repair", "blocking_code",
      ...(protectedOperation ? ["protected_strengths", "protected_strengths_blocking_code"] : []),
      ...(orderOperation ? ["supported_scope", "sequence_index_base"] : []),
      ...(roleOperation ? [
        "supported_evaluated_roles", "draft_only_roles", "unsupported_role_blocking_code",
        "typed_plan_field", "typed_source_field", "typed_source_value",
        "combinable_operation_types",
      ] : []),
      ...(durationOperation ? [
        "supported_evaluated_modes", "draft_only_modes", "unsupported_mode_blocking_code",
        "scalar_plan_field", "typed_plan_field",
      ] : []),
      ...(timeWindowOperation ? [
        "typed_plan_field", "constraint_schema_version", "early_arrival_policy",
        "latest_departure_semantics", "combinable_operation_types",
      ] : []),
    ];
    if (!exactKeys(capability, expectedKeys)
        || typeof capability.enabled !== "boolean"
        || !["evaluated", "draft_only", "conditional"].includes(capability.feedback_tier)
        || typeof capability.preview_executable !== "boolean"
        || typeof capability.evaluated_repair !== "boolean"
        || !(capability.blocking_code === null || typeof capability.blocking_code === "string")) return null;
    const expectedBlockingCode = name === "change_route_preference"
      ? "route_preference_not_supported"
      : "full_evaluation_not_supported";
    const expectedSemantics = ["set_stop_day", "set_stop_order", "set_stop_time_window"].includes(name)
      ? [true, "evaluated", true, true, null]
      : (durationOperation || roleOperation)
        ? [true, "conditional", true, true, null]
        : [name !== "change_route_preference", "draft_only", false, false, expectedBlockingCode];
    if ([
      capability.enabled,
      capability.feedback_tier,
      capability.preview_executable,
      capability.evaluated_repair,
      capability.blocking_code,
    ].some((item, index) => item !== expectedSemantics[index])) return null;
    if (protectedOperation
        && (!exactStringArray(capability.protected_strengths, ["must_keep", "booked"])
          || capability.protected_strengths_blocking_code !== "commitment_permission_required")) return null;
    if (orderOperation
        && (capability.supported_scope !== "same_day"
          || capability.sequence_index_base !== 0)) return null;
    if (roleOperation
        && (!exactStringArray(capability.supported_evaluated_roles, [
          "attraction", "activity", "meal", "rest_stop", "scenic_stop",
        ])
          || !exactStringArray(capability.draft_only_roles, [
            "lodging", "transport_hub", "route_waypoint", "origin", "destination",
          ])
          || capability.unsupported_role_blocking_code !== "stop_role_evaluation_not_supported"
          || capability.typed_plan_field !== "itinerary_role"
          || capability.typed_source_field !== "itinerary_role_source"
          || capability.typed_source_value !== "user_declared_itinerary_role"
          || !exactStringArray(capability.combinable_operation_types, ["set_stop_role"]))) return null;
    if (durationOperation
        && (!exactStringArray(capability.supported_evaluated_modes, ["exact"])
          || !exactStringArray(capability.draft_only_modes, ["preferred", "minimum", "maximum", "range"])
          || capability.unsupported_mode_blocking_code !== "duration_mode_evaluation_not_supported"
          || capability.scalar_plan_field !== "visit_duration_minutes"
          || capability.typed_plan_field !== "duration_constraint")) return null;
    if (timeWindowOperation
        && (capability.typed_plan_field !== "time_window_constraint"
          || capability.constraint_schema_version !== "stop-time-window-constraint-v1"
          || capability.early_arrival_policy !== "wait_until_earliest_arrival"
          || capability.latest_departure_semantics !== "departure_after_visit"
          || !exactStringArray(capability.combinable_operation_types, ["set_stop_time_window"]))) return null;
  }
  return value;
}

function normalizeDraftImpactPreview(value) {
  const topKeys = [
    "schema_version", "session_id", "session_revision", "parent_plan_id", "parent_plan_content_hash",
    "certification_status", "is_certified", "operations", "summary",
  ];
  if (!exactKeys(value, topKeys)
      || value.schema_version !== "product-draft-impact-preview-v1"
      || value.session_id !== state.session?.session_id
      || value.session_revision !== state.session?.revision
      || value.parent_plan_id !== state.session?.accepted_plan_id
      || typeof value.parent_plan_content_hash !== "string"
      || !/^[0-9a-f]{16}$/.test(value.parent_plan_content_hash)
      || value.certification_status !== "not_certified"
      || value.is_certified !== false
      || !Array.isArray(value.operations)
      || !exactKeys(value.summary, [
        "operation_count", "evaluated_executable_count", "draft_only_count",
        "can_run_evaluated_preview", "blocking_codes",
      ])) return null;
  const draft = state.session?.draft;
  const capabilities = normalizeTypedEditCapabilities(state.workspace?.typed_edit_capabilities);
  if (!Array.isArray(draft) || !capabilities
      || !Number.isInteger(value.summary.operation_count)
      || value.summary.operation_count !== value.operations.length
      || value.operations.length !== draft.length
      || !Number.isInteger(value.summary.evaluated_executable_count)
      || !Number.isInteger(value.summary.draft_only_count)
      || typeof value.summary.can_run_evaluated_preview !== "boolean"
      || !Array.isArray(value.summary.blocking_codes)
      || !value.summary.blocking_codes.every((code) => typeof code === "string")) return null;
  const changedAttributes = {
    set_stop_role: ["itinerary_role"],
    set_stop_day: ["day"],
    set_stop_order: ["day", "sequence_order"],
    set_stop_duration: ["duration"],
    set_stop_time_window: ["time_window"],
    set_stop_commitment: ["commitment"],
    report_route_issue: ["user_route_report"],
    change_route_preference: ["route_preference"],
  };
  for (const [index, row] of value.operations.entries()) {
    const draftOperation = draft[index];
    if (!exactKeys(row, [
      "operation_id", "type", "target", "feedback_tier", "preview_executable",
      "evaluated_repair", "changed_attributes", "blocking_codes",
    ])
        || typeof row.operation_id !== "string"
        || typeof row.type !== "string"
        || typeof row.target !== "string"
        || row.operation_id !== draftOperation?.operation_id
        || row.type !== draftOperation?.type
        || row.target !== draftOperation?.target
        || !["evaluated", "draft_only"].includes(row.feedback_tier)
        || typeof row.preview_executable !== "boolean"
        || typeof row.evaluated_repair !== "boolean"
        || !Array.isArray(row.changed_attributes)
        || !row.changed_attributes.every((attribute) => typeof attribute === "string")
        || !Array.isArray(row.blocking_codes)
        || !row.blocking_codes.every((code) => typeof code === "string")) return null;
    const typedCapability = capabilities.operations[row.type];
    const legacyCapability = state.workspace?.draft_capabilities?.operations?.[row.type];
    const mapCapability = state.workspace?.map_edit_capabilities?.operations?.[row.type];
    const durationMode = draftOperation?.parameters?.duration?.mode;
    const requestedRole = draftOperation?.parameters?.role;
    const typedRoleSemantics = row.type === "set_stop_role" && typedCapability
      ? typedCapability.supported_evaluated_roles.includes(requestedRole)
        ? { feedback_tier: "evaluated", preview_executable: true, evaluated_repair: true, blocking_codes: [] }
        : typedCapability.draft_only_roles.includes(requestedRole)
          ? {
            feedback_tier: "draft_only",
            preview_executable: false,
            evaluated_repair: false,
            blocking_codes: [typedCapability.unsupported_role_blocking_code],
          }
          : null
      : null;
    const typedDurationSemantics = row.type === "set_stop_duration" && typedCapability
      ? typedCapability.supported_evaluated_modes.includes(durationMode)
        ? { feedback_tier: "evaluated", preview_executable: true, evaluated_repair: true, blocking_codes: [] }
        : typedCapability.draft_only_modes.includes(durationMode)
          ? {
            feedback_tier: "draft_only",
            preview_executable: false,
            evaluated_repair: false,
            blocking_codes: [typedCapability.unsupported_mode_blocking_code],
          }
          : null
      : null;
    const expected = typedDurationSemantics || typedRoleSemantics || (typedCapability
      && !["set_stop_duration", "set_stop_role"].includes(row.type)
      ? {
        feedback_tier: typedCapability.feedback_tier,
        preview_executable: typedCapability.preview_executable,
        evaluated_repair: typedCapability.evaluated_repair,
        blocking_codes: typedCapability.blocking_code ? [typedCapability.blocking_code] : [],
      }
      : legacyCapability?.enabled === true && legacyCapability.preview_executable === true
        ? { feedback_tier: "evaluated", preview_executable: true, evaluated_repair: true, blocking_codes: [] }
        : mapCapability?.enabled === true && mapCapability.preview_executable === false
          ? {
            feedback_tier: "draft_only",
            preview_executable: false,
            evaluated_repair: false,
            blocking_codes: ["full_evaluation_not_supported"],
          }
          : null);
    const expectedChangedAttributes = row.type === "set_attribute_constraint"
      ? [String(draftOperation.parameters?.attribute || "")]
      : changedAttributes[row.type] || [];
    if (!expected
        || row.feedback_tier !== expected.feedback_tier
        || row.preview_executable !== expected.preview_executable
        || row.evaluated_repair !== expected.evaluated_repair
        || !exactStringArray(row.blocking_codes, expected.blocking_codes)
        || !exactStringArray(row.changed_attributes, expectedChangedAttributes)) return null;
  }
  const evaluatedCount = value.operations.filter((row) => row.preview_executable).length;
  const draftOnlyCount = value.operations.length - evaluatedCount;
  const blockingCodes = [...new Set(value.operations.flatMap((row) => row.blocking_codes))];
  const directTypes = new Set([
    "set_stop_order", "set_stop_role", "set_stop_duration", "set_stop_time_window",
  ]);
  const presentDirectTypes = new Set(draft.map((operation) => operation.type).filter((type) => directTypes.has(type)));
  if (presentDirectTypes.size > 0 && (
    presentDirectTypes.size !== 1
      || draft.some((operation) => !presentDirectTypes.has(operation.type))
  )) {
    blockingCodes.push("draft_evaluated_operation_combination_unsupported");
  }
  if (value.summary.evaluated_executable_count !== evaluatedCount
      || value.summary.draft_only_count !== draftOnlyCount
      || value.summary.can_run_evaluated_preview !== (
        value.operations.length > 0
          && evaluatedCount === value.operations.length
          && blockingCodes.length === 0
      )
      || !exactStringArray(value.summary.blocking_codes, blockingCodes)) return null;
  return value;
}

function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.keys(value).sort().map(
    (key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`,
  ).join(",")}}`;
}

function sameJson(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function draftHashRecords(draft) {
  return draft.map((operation) => ({
    operation_id: operation.operation_id,
    type: operation.type,
    target: operation.target,
    parameters: operation.parameters,
    source: operation.source,
    evidence_refs: operation.evidence_refs,
  }));
}

async function computedDraftContentHash(draft) {
  if (!globalThis.crypto?.subtle || typeof TextEncoder !== "function") return null;
  const bytes = new TextEncoder().encode(canonicalJson({ operations: draftHashRecords(draft) }));
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((item) => item.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

const STABLE_FLOAT_FIELDS = new Set([
  "from_minutes", "to_minutes", "accounting_from_minutes", "accounting_to_minutes",
  "delta_minutes", "cost", "weighted_edit_cost", "configured_default_visit_minutes",
  "configured_day_start_minute", "duration_minutes", "incoming_travel_minutes",
  "base_change_cost", "road_arrival_minute", "opening_start_minute", "opening_end_minute",
  "opening_wait_minutes", "required_window_wait_minutes", "service_start_minute",
  "visit_minutes", "departure_minute", "latest_departure_overrun_minutes",
  "distance_m", "preservation_rate",
]);
const STABLE_FLOAT_OBJECTS = new Set(["metrics", "constraint_strength_weights", "schedule_metrics"]);

function pythonStableJson(value, parentKey = null) {
  if (value === null) return "null";
  if (typeof value === "boolean" || typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return null;
    if (STABLE_FLOAT_OBJECTS.has(parentKey) || STABLE_FLOAT_FIELDS.has(parentKey)) {
      return Number.isInteger(value) ? `${value}.0` : String(value);
    }
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    const items = value.map((item) => pythonStableJson(item, parentKey));
    return items.includes(null) ? null : `[${items.join(",")}]`;
  }
  if (!value || typeof value !== "object") return null;
  const fields = [];
  for (const key of Object.keys(value).sort()) {
    const serialized = pythonStableJson(
      value[key],
      STABLE_FLOAT_OBJECTS.has(parentKey) ? parentKey : key,
    );
    if (serialized === null) return null;
    fields.push(`${JSON.stringify(key)}:${serialized}`);
  }
  return `{${fields.join(",")}}`;
}

async function computedStableContentHash(value) {
  if (!globalThis.crypto?.subtle || typeof TextEncoder !== "function") return null;
  const serialized = pythonStableJson(value);
  if (serialized === null) return null;
  const bytes = new TextEncoder().encode(serialized);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((item) => item.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

async function computedRoleProposalIntegrityDigest(value) {
  if (!globalThis.crypto?.subtle || typeof TextEncoder !== "function"
      || typeof state.token !== "string" || !state.token) return null;
  const payload = { ...value };
  delete payload.role_proposal_integrity;
  const serialized = pythonStableJson(payload);
  if (serialized === null) return null;
  const encoder = new TextEncoder();
  const key = await globalThis.crypto.subtle.importKey(
    "raw",
    encoder.encode(state.token),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const digest = await globalThis.crypto.subtle.sign(
    "HMAC", key, encoder.encode(serialized),
  );
  return [...new Uint8Array(digest)]
    .map((item) => item.toString(16).padStart(2, "0"))
    .join("");
}

async function normalizeRoleProposalIntegrity(value, proposal) {
  if (!exactKeys(value, ["schema_version", "algorithm", "digest"])
      || value.schema_version !== "evaluated-role-proposal-integrity-v1"
      || value.algorithm !== "hmac-sha256-session-token-v1"
      || typeof value.digest !== "string" || !/^[0-9a-f]{64}$/.test(value.digest)) return null;
  const digest = await computedRoleProposalIntegrityDigest(proposal);
  return digest !== null && digest === value.digest ? value : null;
}

function canonicalExactDuration(value) {
  if (!exactKeys(value, ["mode", "preferred_minutes", "minimum_minutes", "maximum_minutes"])) return null;
  const minutes = value.preferred_minutes;
  if (value.mode !== "exact" || !Number.isInteger(minutes) || minutes < 15 || minutes > 480
      || value.minimum_minutes !== minutes || value.maximum_minutes !== minutes) return null;
  return { mode: "exact", preferred_minutes: minutes, minimum_minutes: minutes, maximum_minutes: minutes };
}

function validLocalClock(value) {
  return value === null || (typeof value === "string" && /^(?:[01][0-9]|2[0-3]):[0-5][0-9]$/.test(value));
}

function localClockMinute(value) {
  if (!validLocalClock(value) || value === null) return null;
  const [hour, minute] = value.split(":").map(Number);
  return hour * 60 + minute;
}

function canonicalTimeWindowParameters(value) {
  if (!exactKeys(value, ["earliest_arrival", "latest_departure"])) return null;
  const earliest = value.earliest_arrival;
  const latest = value.latest_departure;
  if (!validLocalClock(earliest) || !validLocalClock(latest)
      || (earliest === null && latest === null)
      || (earliest !== null && latest !== null
        && localClockMinute(earliest) > localClockMinute(latest))) return null;
  return { earliest_arrival: earliest, latest_departure: latest };
}

function canonicalTimeWindowConstraint(value) {
  if (!exactKeys(value, [
    "schema_version", "earliest_arrival", "latest_departure",
    "early_arrival_policy", "latest_departure_semantics",
  ])) return null;
  const parameters = canonicalTimeWindowParameters({
    earliest_arrival: value.earliest_arrival,
    latest_departure: value.latest_departure,
  });
  if (!parameters
      || value.schema_version !== "stop-time-window-constraint-v1"
      || value.early_arrival_policy !== "wait_until_earliest_arrival"
      || value.latest_departure_semantics !== "departure_after_visit") return null;
  return { schema_version: value.schema_version, ...parameters,
    early_arrival_policy: value.early_arrival_policy,
    latest_departure_semantics: value.latest_departure_semantics };
}

function timeWindowConstraintFromParameters(value) {
  const parameters = canonicalTimeWindowParameters(value);
  return parameters ? {
    schema_version: "stop-time-window-constraint-v1",
    ...parameters,
    early_arrival_policy: "wait_until_earliest_arrival",
    latest_departure_semantics: "departure_after_visit",
  } : null;
}

async function normalizeDurationPlanDiff(value, proposal, draft) {
  if (!exactKeys(value, [
    "diff_id", "parent_plan_id", "child_plan_id", "added_stops", "deleted_stops",
    "day_moves", "time_shifts", "reorder_changes", "lodging_changes", "road_changes",
    "unchanged_days", "weighted_edit_cost", "schema_version", "duration_changes",
  ])
      || value.schema_version !== "plan-diff-v2"
      || value.diff_id !== proposal.diff_id
      || value.parent_plan_id !== proposal.parent_plan_id
      || value.child_plan_id !== proposal.child_plan_id
      || !["added_stops", "deleted_stops", "day_moves", "time_shifts", "reorder_changes", "lodging_changes", "road_changes"]
        .every((key) => Array.isArray(value[key]) && value[key].length === 0)
      || !Array.isArray(value.duration_changes)
      || value.duration_changes.length !== draft.length
      || !Number.isFinite(value.weighted_edit_cost) || value.weighted_edit_cost < 0) return null;
  const draftByTarget = new Map();
  for (const operation of draft) {
    const requested = canonicalExactDuration(operation.parameters?.duration);
    if (operation.type !== "set_stop_duration" || !requested || draftByTarget.has(operation.target)) return null;
    draftByTarget.set(operation.target, requested);
  }
  const affectedDays = new Set();
  let totalCost = 0;
  for (const change of value.duration_changes) {
    if (!exactKeys(change, [
      "stop_id", "day", "from_constraint", "to_constraint", "from_minutes", "to_minutes",
      "from_source", "to_source", "accounting_from_minutes", "accounting_to_minutes",
      "accounting_from_source", "accounting_to_source", "delta_minutes", "owner_strength", "cost",
    ])) return null;
    const requested = draftByTarget.get(change.stop_id);
    const fromConstraint = change.from_constraint === null
      ? null
      : canonicalExactDuration(change.from_constraint);
    const acceptedDay = state.workspace?.timeline?.find(
      (day) => day.stops.some((stop) => stop.id === change.stop_id),
    )?.day;
    const acceptedOwnership = state.workspace?.geography?.plans?.find(
      (plan) => plan.plan_id === state.session?.accepted_plan_id,
    )?.stops?.features?.find(
      (feature) => feature.properties?.stop_id === change.stop_id,
    )?.properties?.ownership_strength;
    if (!requested || change.day !== acceptedDay || !sameJson(change.to_constraint, requested)
        || (change.from_constraint !== null && !fromConstraint)
        || !(change.from_minutes === null || (
          Number.isInteger(change.from_minutes) && change.from_minutes >= 15 && change.from_minutes <= 480
        ))
        || change.to_minutes !== requested.preferred_minutes
        || !["unavailable", "visit_duration_minutes", "duration_minutes", "service_minutes", "dwell_minutes"].includes(change.from_source)
        || change.to_source !== "visit_duration_minutes"
        || !Number.isFinite(change.accounting_from_minutes)
        || !Number.isFinite(change.accounting_to_minutes)
        || change.accounting_to_minutes !== requested.preferred_minutes
        || !["configured_evaluator_default", "visit_duration_minutes", "duration_minutes", "service_minutes", "dwell_minutes"].includes(change.accounting_from_source)
        || change.accounting_to_source !== "visit_duration_minutes"
        || !Number.isFinite(change.delta_minutes)
        || typeof acceptedOwnership !== "string" || change.owner_strength !== acceptedOwnership
        || !Number.isFinite(change.cost) || change.cost < 0) return null;
    const rawUnavailable = change.from_source === "unavailable";
    if ((rawUnavailable && (change.from_minutes !== null || change.from_constraint !== null))
        || (!rawUnavailable && !Number.isFinite(change.from_minutes))
        || (fromConstraint && fromConstraint.preferred_minutes !== change.from_minutes)
        || (change.accounting_from_source === "configured_evaluator_default" && !rawUnavailable)
        || (change.accounting_from_source !== "configured_evaluator_default" && (
          rawUnavailable
            || change.accounting_from_source !== change.from_source
            || change.accounting_from_minutes !== change.from_minutes
        ))) return null;
    if (Math.abs(
      change.accounting_to_minutes - change.accounting_from_minutes - change.delta_minutes,
    ) > 1e-9) return null;
    affectedDays.add(change.day);
    totalCost += change.cost;
    draftByTarget.delete(change.stop_id);
  }
  const allDays = (state.workspace?.timeline || []).map((day) => day.day);
  const expectedUnchanged = allDays.filter((day) => !affectedDays.has(day));
  if (draftByTarget.size !== 0 || !exactStringArray(value.unchanged_days, expectedUnchanged)
      || Math.abs(totalCost - value.weighted_edit_cost) > 1e-9) return null;
  const hash = await computedStableContentHash(value);
  if (hash === null || hash !== proposal.diff_content_hash
      || proposal.diff_identity.content_hash !== hash) return null;
  return value;
}

async function normalizeTimeWindowPlanDiff(value, proposal, draft) {
  if (!exactKeys(value, [
    "diff_id", "parent_plan_id", "child_plan_id", "added_stops", "deleted_stops",
    "day_moves", "time_shifts", "reorder_changes", "lodging_changes", "road_changes",
    "unchanged_days", "weighted_edit_cost", "schema_version", "time_window_changes",
  ])
      || value.schema_version !== "plan-diff-v3"
      || value.diff_id !== proposal.diff_id
      || value.parent_plan_id !== proposal.parent_plan_id
      || value.child_plan_id !== proposal.child_plan_id
      || !["added_stops", "deleted_stops", "day_moves", "time_shifts", "reorder_changes", "lodging_changes", "road_changes"]
        .every((key) => Array.isArray(value[key]) && value[key].length === 0)
      || !Array.isArray(value.time_window_changes)
      || value.time_window_changes.length !== draft.length
      || !Number.isFinite(value.weighted_edit_cost) || value.weighted_edit_cost < 0) return null;
  const draftByTarget = new Map();
  for (const operation of draft) {
    const requested = timeWindowConstraintFromParameters(operation.parameters);
    if (operation.type !== "set_stop_time_window" || !requested || draftByTarget.has(operation.target)) return null;
    draftByTarget.set(operation.target, requested);
  }
  const parentPlan = state.workspace?.geography?.plans?.find(
    (plan) => plan.plan_id === state.session?.accepted_plan_id,
  );
  const affectedDays = new Set();
  let totalCost = 0;
  for (const change of value.time_window_changes) {
    if (!exactKeys(change, [
      "stop_id", "day", "from_constraint", "to_constraint", "from_earliest_arrival",
      "to_earliest_arrival", "from_latest_departure", "to_latest_departure",
      "owner_strength", "cost",
    ])) return null;
    const requested = draftByTarget.get(change.stop_id);
    const fromConstraint = change.from_constraint === null
      ? null
      : canonicalTimeWindowConstraint(change.from_constraint);
    const toConstraint = canonicalTimeWindowConstraint(change.to_constraint);
    const acceptedDay = state.workspace?.timeline?.find(
      (day) => day.stops.some((stop) => stop.id === change.stop_id),
    )?.day;
    const acceptedOwnership = parentPlan?.stops?.features?.find(
      (feature) => feature.properties?.stop_id === change.stop_id,
    )?.properties?.ownership_strength;
    if (!requested || !toConstraint
        || change.day !== acceptedDay
        || canonicalJson(toConstraint) !== canonicalJson(requested)
        || (change.from_constraint !== null && !fromConstraint)
        || change.from_earliest_arrival !== (fromConstraint?.earliest_arrival ?? null)
        || change.from_latest_departure !== (fromConstraint?.latest_departure ?? null)
        || change.to_earliest_arrival !== requested.earliest_arrival
        || change.to_latest_departure !== requested.latest_departure
        || typeof acceptedOwnership !== "string" || change.owner_strength !== acceptedOwnership
        || !Number.isFinite(change.cost) || change.cost < 0) return null;
    affectedDays.add(change.day);
    totalCost += change.cost;
    draftByTarget.delete(change.stop_id);
  }
  const allDays = (state.workspace?.timeline || []).map((day) => day.day);
  const expectedUnchanged = allDays.filter((day) => !affectedDays.has(day));
  if (draftByTarget.size !== 0 || !exactStringArray(value.unchanged_days, expectedUnchanged)
      || Math.abs(totalCost - value.weighted_edit_cost) > 1e-9) return null;
  const hash = await computedStableContentHash(value);
  if (hash === null || hash !== proposal.diff_content_hash
      || proposal.diff_identity.content_hash !== hash) return null;
  return value;
}

const EVALUATED_STOP_ROLES = ["attraction", "activity", "meal", "rest_stop", "scenic_stop"];
const ROLE_SEMANTIC_SCOPE = {
  itinerary_role_semantics: "trip_specific_user_declared_use",
  place_identity_unchanged: true,
  place_categories_unchanged: true,
  route_schedule_effect: "none_for_supported_visit_roles",
  semantic_fit_claim: "unavailable",
  recommendation_claim: "unavailable",
};

async function normalizeRoleConstraintEvidence(value) {
  if (!exactKeys(value, [
    "schema_version", "parent_plan_id", "parent_plan_content_hash", "constraints", "content_hash",
  ])
      || value.schema_version !== "evaluated-role-constraint-evidence-v1"
      || value.parent_plan_id !== state.session?.accepted_plan_id
      || value.parent_plan_content_hash !== state.workspace?.geography?.plans?.find(
        (plan) => plan.plan_id === state.session?.accepted_plan_id,
      )?.content_hash
      || !Array.isArray(value.constraints)) return null;
  const validStrengths = new Set([
    "locked", "booked", "hard", "strong", "soft", "weak", "preference",
  ]);
  const validRelations = new Set([
    "role", "itinerary_role", "role_equals", "set_role", "attribute:role",
  ]);
  const validRelaxation = new Set(["never", "explicit_only", "auto_with_penalty", "always"]);
  const acceptedStops = new Set((acceptedPlanTimeWindowTraceStops() || []).map((stop) => stop.stop_id));
  const seenIds = new Set();
  const roleByTarget = new Map();
  let priorSortKey = null;
  for (const constraint of value.constraints) {
    if (!exactKeys(constraint, [
      "constraint_id", "target_stop_id", "required_role", "strength", "scope", "relation",
      "relaxation_policy", "permission_semantics",
    ])
        || typeof constraint.constraint_id !== "string" || !constraint.constraint_id
        || constraint.constraint_id.length > 128 || seenIds.has(constraint.constraint_id)
        || !acceptedStops.has(constraint.target_stop_id)
        || !TYPED_EDIT_VOCABULARIES.stop_roles.includes(constraint.required_role)
        || !validStrengths.has(constraint.strength)
        || !["stop", "poi"].includes(constraint.scope)
        || !validRelations.has(constraint.relation)
        || !validRelaxation.has(constraint.relaxation_policy)) return null;
    const explicit = ["locked", "booked", "hard"].includes(constraint.strength)
      || ["never", "explicit_only"].includes(constraint.relaxation_policy);
    if (constraint.permission_semantics !== (
      explicit ? "explicit_permission_required_for_mismatch" : "weighted_mismatch_allowed"
    )) return null;
    const priorRole = roleByTarget.get(constraint.target_stop_id);
    const sortKey = `${constraint.target_stop_id}\u0000${constraint.constraint_id}`;
    if ((priorRole !== undefined && priorRole !== constraint.required_role)
        || (priorSortKey !== null && sortKey <= priorSortKey)) return null;
    roleByTarget.set(constraint.target_stop_id, constraint.required_role);
    seenIds.add(constraint.constraint_id);
    priorSortKey = sortKey;
  }
  const payload = { ...value };
  delete payload.content_hash;
  const hash = await computedStableContentHash(payload);
  if (hash === null || hash !== value.content_hash) return null;
  return value;
}

function roleConstraintStrength(evidence, stopId, costPolicy) {
  let result = "";
  for (const constraint of evidence.constraints.filter((row) => row.target_stop_id === stopId)) {
    if (costPolicy.constraint_strength_weights[constraint.strength]
        >= costPolicy.constraint_strength_weights[result]) result = constraint.strength;
  }
  return result;
}

async function normalizeRolePlanDiff(value, proposal, draft) {
  if (!exactKeys(value, [
    "diff_id", "parent_plan_id", "child_plan_id", "added_stops", "deleted_stops",
    "day_moves", "time_shifts", "reorder_changes", "lodging_changes", "road_changes",
    "unchanged_days", "weighted_edit_cost", "schema_version", "role_changes",
  ])
      || value.schema_version !== "plan-diff-v4"
      || value.diff_id !== proposal.diff_id
      || value.parent_plan_id !== proposal.parent_plan_id
      || value.child_plan_id !== proposal.child_plan_id
      || !["added_stops", "deleted_stops", "day_moves", "time_shifts", "reorder_changes", "lodging_changes", "road_changes"]
        .every((key) => Array.isArray(value[key]) && value[key].length === 0)
      || !Array.isArray(value.role_changes) || value.role_changes.length !== draft.length
      || !Number.isFinite(value.weighted_edit_cost) || value.weighted_edit_cost < 0) return null;
  const requestedByTarget = new Map();
  for (const operation of draft) {
    const role = operation.parameters?.role;
    if (operation.type !== "set_stop_role" || !EVALUATED_STOP_ROLES.includes(role)
        || requestedByTarget.has(operation.target)) return null;
    requestedByTarget.set(operation.target, role);
  }
  const acceptedPlan = state.workspace?.geography?.plans?.find(
    (plan) => plan.plan_id === state.session?.accepted_plan_id,
  );
  const acceptedStops = acceptedPlanTimeWindowTraceStops();
  const roleConstraintEvidence = await normalizeRoleConstraintEvidence(
    state.workspace?.role_constraint_evidence,
  );
  const roleCostPolicy = normalizeRoleCostPolicy({
    base_change_cost: 0.25,
    formula: "base_times_role_scoped_strength_weight",
    constraint_strength_weights: {
      "": 1, locked: 1000, booked: 500, hard: 100, strong: 50,
      soft: 10, weak: 5, preference: 1, test_only: 0,
    },
  });
  if (!roleConstraintEvidence || !roleCostPolicy) return null;
  const dayByStop = new Map((acceptedStops || []).map((stop) => [stop.stop_id, stop.day]));
  const affectedDays = new Set();
  let totalCost = 0;
  for (const change of value.role_changes) {
    if (!exactKeys(change, [
      "stop_id", "day", "from_role", "to_role", "from_source", "to_source",
      "owner_strength", "cost",
    ])) return null;
    const requested = requestedByTarget.get(change.stop_id);
    const feature = acceptedPlan?.stops?.features?.find(
      (candidate) => candidate.properties?.stop_id === change.stop_id,
    );
    const parentRole = feature?.properties?.itinerary_role ?? null;
    const parentSource = feature?.properties?.itinerary_role_source;
    if (!requested || change.day !== dayByStop.get(change.stop_id)
        || change.from_role !== parentRole || change.to_role !== requested
        || change.from_source !== (parentRole === null ? "unavailable" : "user_declared_itinerary_role")
        || parentSource !== change.from_source
        || change.to_source !== "user_declared_itinerary_role") return null;
    const ownerStrength = roleConstraintStrength(
      roleConstraintEvidence, change.stop_id, roleCostPolicy,
    );
    const targetConstraints = roleConstraintEvidence.constraints.filter(
      (constraint) => constraint.target_stop_id === change.stop_id,
    );
    if (targetConstraints.some((constraint) => (
      constraint.permission_semantics === "explicit_permission_required_for_mismatch"
        && constraint.required_role !== change.to_role
    ))
        || change.owner_strength !== ownerStrength
        || change.cost !== roleCostPolicy.base_change_cost
          * roleCostPolicy.constraint_strength_weights[ownerStrength]) return null;
    affectedDays.add(change.day);
    totalCost += change.cost;
    requestedByTarget.delete(change.stop_id);
  }
  const allDays = [...new Set((acceptedStops || []).map((stop) => stop.day))];
  const unchangedDays = allDays.filter((day) => !affectedDays.has(day));
  if (requestedByTarget.size !== 0 || !exactStringArray(value.unchanged_days, unchangedDays)
      || Math.abs(value.weighted_edit_cost - totalCost) > 1e-9) return null;
  const hash = await computedStableContentHash(value);
  if (hash === null || hash !== proposal.diff_content_hash
      || proposal.diff_identity.content_hash !== hash) return null;
  return value;
}

function durationScheduleMetricKeys() {
  const keys = [
    "route_required_leg_count", "route_present_leg_count", "route_road_validated_leg_count",
    "route_fallback_leg_count", "route_missing_leg_count",
  ];
  for (const { day } of state.workspace?.timeline || []) {
    for (const metric of ["travel", "visit", "waiting", "total", "limit", "slack", "overrun"]) {
      keys.push(`day_${day}_${metric}_minutes`);
    }
  }
  return keys;
}

function normalizeDurationCostPolicy(value) {
  const expectedWeights = {
    "": 1,
    locked: 1000,
    booked: 500,
    hard: 100,
    strong: 50,
    soft: 10,
    weak: 5,
    preference: 1,
    test_only: 0,
  };
  if (!exactKeys(value, ["base_change_cost", "formula", "constraint_strength_weights"])
      || value.base_change_cost !== 0.25
      || value.formula !== "base_times_strength_weight_times_max_1_abs_delta_minutes_over_60"
      || !exactKeys(value.constraint_strength_weights, Object.keys(expectedWeights))
      || !Object.entries(expectedWeights).every(
        ([name, weight]) => value.constraint_strength_weights[name] === weight,
      )) return null;
  return value;
}

function durationDiffMatchesCostPolicy(planDiff, policy) {
  if (!policy) return false;
  let total = 0;
  for (const change of planDiff.duration_changes) {
    const weight = policy.constraint_strength_weights[change.owner_strength];
    if (!Number.isFinite(weight)) return false;
    const expectedCost = policy.base_change_cost * weight
      * Math.max(1, Math.abs(change.delta_minutes) / 60);
    if (Math.abs(change.cost - expectedCost) > 1e-9) return false;
    total += expectedCost;
  }
  return Math.abs(total - planDiff.weighted_edit_cost) <= 1e-9;
}

async function normalizeScheduleAccounting(
  value, { evaluatorVersion, routeMatrixId, planId, planContentHash, routeValidation },
) {
  const allowedBlockingCodes = new Set(["opening_window_missed", "day_time_exceeded"]);
  if (!exactKeys(value, [
    "schema_version", "evaluator_version", "plan_id", "plan_content_hash", "route_matrix_id",
    "status", "blocking_codes", "missing_opening_window_stop_ids", "modeled_components",
    "unavailable_components", "metrics", "content_hash",
  ])
      || value.schema_version !== "evaluated-schedule-accounting-v1"
      || value.evaluator_version !== evaluatorVersion
      || value.plan_id !== planId || value.plan_content_hash !== planContentHash
      || value.route_matrix_id !== routeMatrixId
      || !["failed", "incomplete_evidence", "passed"].includes(value.status)
      || !Array.isArray(value.blocking_codes)
      || !value.blocking_codes.every((code) => allowedBlockingCodes.has(code))
      || new Set(value.blocking_codes).size !== value.blocking_codes.length
      || !Array.isArray(value.missing_opening_window_stop_ids)
      || !value.missing_opening_window_stop_ids.every((stopId) => typeof stopId === "string" && stopId)
      || !exactStringArray(
        value.missing_opening_window_stop_ids,
        [...value.missing_opening_window_stop_ids].sort(),
      )
      || !exactStringArray(value.modeled_components, [
        "road_travel", "visit_duration", "opening_wait_when_window_available", "day_limit",
      ])
      || !exactStringArray(value.unavailable_components, [
        "parking_dropoff", "walking_transfer", "queue_wait", "service_buffer",
      ])
      || !exactKeys(value.metrics, durationScheduleMetricKeys())
      || !Object.values(value.metrics).every((metric) => Number.isFinite(metric) && metric >= 0)
      || value.metrics.route_required_leg_count !== routeValidation.required_leg_count
      || value.metrics.route_present_leg_count !== routeValidation.present_leg_count
      || value.metrics.route_road_validated_leg_count !== routeValidation.road_validated_leg_count
      || value.metrics.route_fallback_leg_count !== routeValidation.fallback_leg_count
      || value.metrics.route_missing_leg_count !== routeValidation.missing_leg_count) return null;
  const expectedStatus = value.blocking_codes.length
    ? "failed"
    : value.missing_opening_window_stop_ids.length
      ? "incomplete_evidence"
      : "passed";
  if (value.status !== expectedStatus) return null;
  for (const { day } of state.workspace?.timeline || []) {
    const travel = value.metrics[`day_${day}_travel_minutes`];
    const visit = value.metrics[`day_${day}_visit_minutes`];
    const waiting = value.metrics[`day_${day}_waiting_minutes`];
    const total = value.metrics[`day_${day}_total_minutes`];
    const limit = value.metrics[`day_${day}_limit_minutes`];
    const slack = value.metrics[`day_${day}_slack_minutes`];
    const overrun = value.metrics[`day_${day}_overrun_minutes`];
    if (Math.abs(total - travel - visit - waiting) > 1e-8
        || Math.abs(slack - Math.max(0, limit - total)) > 1e-8
        || Math.abs(overrun - Math.max(0, total - limit)) > 1e-8) return null;
  }
  const payload = { ...value };
  delete payload.content_hash;
  const hash = await computedStableContentHash(payload);
  if (hash === null || hash !== value.content_hash) return null;
  return value;
}

async function normalizeDurationScheduleImpact(value, identity, proposal, planDiff, routeValidation) {
  if (!exactKeys(value, [
    "schema_version", "evaluator_version", "route_matrix_id", "configured_default_visit_minutes",
    "duration_cost_policy", "affected_days", "target_stop_ids", "parent", "child", "content_hash",
  ])
      || value.schema_version !== "evaluated-duration-schedule-impact-v1"
      || typeof value.evaluator_version !== "string" || !value.evaluator_version
      || value.route_matrix_id !== routeValidation.matrix_id
      || value.configured_default_visit_minutes !== 45
      || !normalizeDurationCostPolicy(value.duration_cost_policy)
      || !exactKeys(identity, [
        "content_hash", "evaluator_version", "route_matrix_id", "parent_plan_id",
        "parent_plan_content_hash", "child_plan_id", "child_plan_content_hash",
        "certificate_id", "certificate_content_hash",
      ])) return null;
  const changes = planDiff.duration_changes;
  const affectedDays = [...new Set(changes.map((change) => change.day))].sort((left, right) => left - right);
  const targetStopIds = changes.map((change) => change.stop_id).sort();
  if (!exactStringArray(value.affected_days, affectedDays)
      || !exactStringArray(value.target_stop_ids, targetStopIds)) return null;
  for (const change of changes) {
    if (change.accounting_from_source === "configured_evaluator_default"
        && change.accounting_from_minutes !== value.configured_default_visit_minutes) return null;
  }
  if (!durationDiffMatchesCostPolicy(planDiff, value.duration_cost_policy)) return null;
  const common = {
    evaluatorVersion: value.evaluator_version,
    routeMatrixId: value.route_matrix_id,
    routeValidation,
  };
  const parent = await normalizeScheduleAccounting(value.parent, {
    ...common, planId: proposal.parent_plan_id, planContentHash: proposal.parent_plan_content_hash,
  });
  const child = await normalizeScheduleAccounting(value.child, {
    ...common, planId: proposal.child_plan_id, planContentHash: proposal.child_plan_content_hash,
  });
  if (!parent || !child
      || !exactStringArray(parent.missing_opening_window_stop_ids, child.missing_opening_window_stop_ids)) return null;
  const deltaByDay = new Map();
  for (const change of changes) {
    deltaByDay.set(change.day, (deltaByDay.get(change.day) || 0) + change.delta_minutes);
  }
  for (const { day } of state.workspace?.timeline || []) {
    const parentMetrics = parent.metrics;
    const childMetrics = child.metrics;
    const delta = deltaByDay.get(day) || 0;
    for (const name of ["travel", "limit"]) {
      if (childMetrics[`day_${day}_${name}_minutes`] !== parentMetrics[`day_${day}_${name}_minutes`]) return null;
    }
    const waitingDelta = childMetrics[`day_${day}_waiting_minutes`]
      - parentMetrics[`day_${day}_waiting_minutes`];
    if (Math.abs(
      childMetrics[`day_${day}_visit_minutes`] - parentMetrics[`day_${day}_visit_minutes`] - delta,
    ) > 1e-8 || Math.abs(
      childMetrics[`day_${day}_total_minutes`] - parentMetrics[`day_${day}_total_minutes`]
        - delta - waitingDelta,
    ) > 1e-8) return null;
  }
  for (const key of durationScheduleMetricKeys().filter((name) => name.startsWith("route_"))) {
    if (parent.metrics[key] !== child.metrics[key]) return null;
  }
  const payload = { ...value };
  delete payload.content_hash;
  const hash = await computedStableContentHash(payload);
  if (hash === null || hash !== value.content_hash
      || identity.content_hash !== hash
      || identity.evaluator_version !== value.evaluator_version
      || identity.route_matrix_id !== value.route_matrix_id
      || identity.parent_plan_id !== proposal.parent_plan_id
      || identity.parent_plan_content_hash !== proposal.parent_plan_content_hash
      || identity.child_plan_id !== proposal.child_plan_id
      || identity.child_plan_content_hash !== proposal.child_plan_content_hash
      || identity.certificate_id !== proposal.certificate_id
      || identity.certificate_content_hash !== proposal.certificate_content_hash) return null;
  return value;
}

async function normalizeCertificateScheduleEvidence(value, proposal, scheduleImpact) {
  if (!exactKeys(value, [
    "schema_version", "certificate_id", "certificate_content_hash", "plan_id",
    "plan_content_hash", "evaluator_version", "evaluation_status", "comparison_eligibility",
    "warning_codes", "failure_codes", "schedule_metrics", "content_hash",
  ])
      || value.schema_version !== "evaluated-duration-certificate-schedule-evidence-v1"
      || value.certificate_id !== proposal.certificate_id
      || value.certificate_content_hash !== proposal.certificate_content_hash
      || value.plan_id !== proposal.child_plan_id
      || value.plan_content_hash !== proposal.child_plan_content_hash
      || value.evaluator_version !== scheduleImpact.evaluator_version
      || value.evaluation_status !== proposal.evaluation_status
      || value.comparison_eligibility !== proposal.eligibility
      || !Array.isArray(value.warning_codes)
      || !value.warning_codes.every((code) => typeof code === "string" && code)
      || !Array.isArray(value.failure_codes)
      || !value.failure_codes.every((code) => typeof code === "string" && code)
      || !exactKeys(value.schedule_metrics, [
        ...durationScheduleMetricKeys(), "schedule_missing_opening_window_count",
      ])
      || !Object.values(value.schedule_metrics).every((metric) => Number.isFinite(metric) && metric >= 0)) return null;
  const child = scheduleImpact.child;
  for (const [name, metric] of Object.entries(child.metrics)) {
    if (value.schedule_metrics[name] !== metric) return null;
  }
  const missingCount = child.missing_opening_window_stop_ids.length;
  const expectedWarnings = missingCount > 0 ? ["opening_window_evidence_missing"] : [];
  const expectedFailures = child.blocking_codes;
  const expectedEvaluation = child.status === "failed"
    ? "FAILED"
    : child.status === "incomplete_evidence"
      ? "PASSED_WITH_WARNINGS"
      : "PASSED";
  const expectedEligibility = child.status === "failed" ? "ineligible" : "eligible";
  if (value.schedule_metrics.schedule_missing_opening_window_count !== missingCount
      || !exactStringArray(value.warning_codes, expectedWarnings)
      || !exactStringArray(value.failure_codes, expectedFailures)
      || value.evaluation_status !== expectedEvaluation
      || value.comparison_eligibility !== expectedEligibility
      || proposal.state !== expectedEligibility
      || proposal.eligibility !== expectedEligibility
      || proposal.decision_eligible !== (expectedEligibility === "eligible")
      || proposal.ranking_eligible !== false
      || proposal.acceptance_eligible !== false
      || proposal.acceptance_blocking_code !== "acceptance_not_enabled_until_w5"
      || proposal.repair.certificate.failure_count !== value.failure_codes.length) return null;
  const payload = { ...value };
  delete payload.content_hash;
  const hash = await computedStableContentHash(payload);
  if (hash === null || hash !== value.content_hash) return null;
  return value;
}

function normalizeTimeWindowCostPolicy(value) {
  const expectedWeights = {
    "": 1,
    locked: 1000,
    booked: 500,
    hard: 100,
    strong: 50,
    soft: 10,
    weak: 5,
    preference: 1,
    test_only: 0,
  };
  if (!exactKeys(value, ["base_change_cost", "formula", "constraint_strength_weights"])
      || value.base_change_cost !== 0.25
      || value.formula !== "base_times_strength_weight"
      || !exactKeys(value.constraint_strength_weights, Object.keys(expectedWeights))
      || !Object.entries(expectedWeights).every(
        ([name, weight]) => value.constraint_strength_weights[name] === weight,
      )) return null;
  return value;
}

function timeWindowDiffMatchesCostPolicy(planDiff, policy) {
  if (!policy) return false;
  let total = 0;
  for (const change of planDiff.time_window_changes) {
    const weight = policy.constraint_strength_weights[change.owner_strength];
    if (!Number.isFinite(weight)) return false;
    const expectedCost = policy.base_change_cost * weight;
    if (Math.abs(change.cost - expectedCost) > 1e-9) return false;
    total += expectedCost;
  }
  return Math.abs(total - planDiff.weighted_edit_cost) <= 1e-9;
}

function normalizeTimeWindowRouteEvidence(value, routeValidation) {
  const parentPlan = state.workspace?.geography?.plans?.find(
    (plan) => plan.plan_id === state.session?.accepted_plan_id,
  );
  const geographicLegs = [...(parentPlan?.validated_legs?.features || [])].sort(
    (left, right) => left.properties?.leg_index - right.properties?.leg_index,
  );
  if (!Array.isArray(value)
      || value.length !== routeValidation.required_leg_count
      || geographicLegs.length !== routeValidation.required_leg_count) return null;
  const seen = new Set();
  for (const [index, row] of value.entries()) {
    const properties = geographicLegs[index]?.properties;
    const key = `${row?.day}:${row?.origin_id}:${row?.destination_id}`;
    if (!exactKeys(row, [
      "day", "origin_id", "destination_id", "duration_minutes", "query_hash",
      "road_validated", "fallback_used",
    ])
        || !Number.isInteger(row.day) || row.day <= 0
        || typeof row.origin_id !== "string" || !row.origin_id
        || typeof row.destination_id !== "string" || !row.destination_id
        || !Number.isFinite(row.duration_minutes) || row.duration_minutes <= 0
        || !/^[0-9a-f]{16}$/.test(row.query_hash)
        || row.road_validated !== true || row.fallback_used !== false
        || seen.has(key)
        || properties?.day !== row.day
        || properties?.origin_id !== row.origin_id
        || properties?.destination_id !== row.destination_id
        || properties?.road_validated !== true || properties?.fallback_used !== false
        || !Number.isFinite(properties?.duration_s)
        || Math.abs(row.duration_minutes - properties.duration_s / 60) > 1e-8) return null;
    seen.add(key);
  }
  return value;
}

function acceptedPlanTimeWindowTraceStops() {
  const parentPlan = state.workspace?.geography?.plans?.find(
    (plan) => plan.plan_id === state.session?.accepted_plan_id,
  );
  const path = [...(parentPlan?.route_path?.features || [])].sort(
    (left, right) => left.properties?.occurrence_index - right.properties?.occurrence_index,
  );
  const sequenceByDay = new Map();
  const stops = [];
  for (const feature of path) {
    const properties = feature?.properties;
    if (properties?.selected_stop !== true) continue;
    const day = properties.arrival_day;
    if (typeof properties.node_id !== "string" || !properties.node_id
        || !Number.isInteger(day) || day <= 0) return null;
    const sequenceIndex = sequenceByDay.get(day) || 0;
    stops.push({ stop_id: properties.node_id, day, sequence_index: sequenceIndex });
    sequenceByDay.set(day, sequenceIndex + 1);
  }
  const stopUniverse = new Set((parentPlan?.stops?.features || []).map(
    (feature) => feature.properties?.stop_id,
  ));
  if (!stops.length || stopUniverse.size !== stops.length
      || stops.some((stop) => !stopUniverse.has(stop.stop_id))) return null;
  return stops;
}

function normalizeTimeWindowTrace(
  value, configuredDefaultVisitMinutes, { expected, incoming, expectedRoadArrival } = {},
) {
  if (!exactKeys(value, [
    "stop_id", "day", "sequence_index", "incoming_origin_id", "incoming_travel_minutes",
    "incoming_route_query_hash", "road_arrival_minute", "opening_start_minute",
    "opening_end_minute", "opening_wait_minutes", "opening_window_source", "time_window_constraint",
    "time_window_source", "required_window_wait_minutes", "service_start_minute",
    "visit_minutes", "visit_duration_source", "departure_minute",
    "latest_departure_status", "latest_departure_overrun_minutes", "failure_codes",
  ])
      || typeof value.stop_id !== "string" || !value.stop_id
      || !Number.isInteger(value.day) || value.day <= 0
      || !Number.isInteger(value.sequence_index) || value.sequence_index < 0
      || typeof value.incoming_origin_id !== "string" || !value.incoming_origin_id
      || !Number.isFinite(value.incoming_travel_minutes) || value.incoming_travel_minutes <= 0
      || !/^[0-9a-f]{16}$/.test(value.incoming_route_query_hash)
      || !Number.isFinite(value.road_arrival_minute) || value.road_arrival_minute < 0
      || !validFiniteOrNull(value.opening_start_minute)
      || !validFiniteOrNull(value.opening_end_minute)
      || !validFiniteOrNull(value.opening_wait_minutes)
      || typeof value.opening_window_source !== "string" || !value.opening_window_source
      || !validFiniteOrNull(value.required_window_wait_minutes)
      || !Number.isFinite(value.service_start_minute) || value.service_start_minute < 0
      || !Number.isFinite(value.visit_minutes) || value.visit_minutes < 15 || value.visit_minutes > 480
      || value.visit_duration_source !== "configured_evaluator_default"
      || value.visit_minutes !== configuredDefaultVisitMinutes
      || !Number.isFinite(value.departure_minute) || value.departure_minute < 0
      || !["not_set", "satisfied", "violated"].includes(value.latest_departure_status)
      || !validFiniteOrNull(value.latest_departure_overrun_minutes)
      || !Array.isArray(value.failure_codes)) return null;
  const openingMissing = value.opening_start_minute === null && value.opening_end_minute === null;
  if ((value.opening_start_minute !== null
      && (value.opening_start_minute < 0 || value.opening_start_minute >= 1440))
      || (value.opening_end_minute !== null
        && (value.opening_end_minute < 0 || value.opening_end_minute >= 1440))
      || (value.opening_start_minute !== null && value.opening_end_minute !== null
        && value.opening_start_minute > value.opening_end_minute)
      || (openingMissing && value.opening_wait_minutes !== null)
      || (!openingMissing && (
        !Number.isFinite(value.opening_wait_minutes) || value.opening_wait_minutes < 0
      ))
      || (openingMissing && value.opening_window_source !== "unavailable")
      || (!openingMissing && !value.opening_window_source.startsWith("plan_stop_fields:"))) return null;
  if (expected && (
    value.stop_id !== expected.stop_id
      || value.day !== expected.day
      || value.sequence_index !== expected.sequence_index
  )) return null;
  if (incoming && (
    value.incoming_origin_id !== incoming.origin_id
      || value.incoming_travel_minutes !== incoming.duration_minutes
      || value.incoming_route_query_hash !== incoming.query_hash
  )) return null;
  if (Number.isFinite(expectedRoadArrival)
      && Math.abs(value.road_arrival_minute - expectedRoadArrival) > 1e-8) return null;
  const expectedOpeningWait = value.opening_start_minute === null
    ? (openingMissing ? null : 0)
    : Math.max(0, value.opening_start_minute - value.road_arrival_minute);
  if (value.opening_wait_minutes !== expectedOpeningWait) return null;
  const constraint = value.time_window_constraint === null
    ? null
    : canonicalTimeWindowConstraint(value.time_window_constraint);
  if ((value.time_window_constraint !== null && !constraint)
      || value.time_window_source !== (constraint ? "trip_specific_user_constraint" : null)) return null;
  const afterOpening = value.road_arrival_minute + (value.opening_wait_minutes ?? 0);
  const earliestMinute = constraint ? localClockMinute(constraint.earliest_arrival) : null;
  const expectedRequiredWait = earliestMinute === null ? null : Math.max(0, earliestMinute - afterOpening);
  if (value.required_window_wait_minutes !== expectedRequiredWait
      || Math.abs(
        value.service_start_minute - afterOpening - (value.required_window_wait_minutes ?? 0),
      ) > 1e-8
      || Math.abs(value.departure_minute - value.service_start_minute - value.visit_minutes) > 1e-8) return null;
  const latestMinute = constraint ? localClockMinute(constraint.latest_departure) : null;
  const expectedOverrun = latestMinute === null
    ? null
    : Math.max(0, value.departure_minute - latestMinute);
  const expectedLatestStatus = latestMinute === null
    ? "not_set"
    : expectedOverrun > 0 ? "violated" : "satisfied";
  const expectedFailures = [];
  if (value.opening_end_minute !== null && value.service_start_minute > value.opening_end_minute) {
    expectedFailures.push("opening_window_missed");
  }
  if (expectedOverrun > 0) expectedFailures.push("stop_time_window_latest_departure_exceeded");
  if (value.latest_departure_status !== expectedLatestStatus
      || value.latest_departure_overrun_minutes !== expectedOverrun
      || !exactStringArray(value.failure_codes, expectedFailures)) return null;
  return value;
}

async function normalizeTimeWindowScheduleAccounting(
  value, {
    evaluatorVersion, routeMatrixId, planId, planContentHash, routeValidation,
    configuredDefaultVisitMinutes, configuredDayStartMinute, routeLegEvidence,
  },
) {
  const allowedBlockingCodes = new Set([
    "opening_window_missed", "stop_time_window_latest_departure_exceeded", "day_time_exceeded",
  ]);
  if (!exactKeys(value, [
    "schema_version", "evaluator_version", "plan_id", "plan_content_hash", "route_matrix_id",
    "status", "modeled_components", "unavailable_components", "blocking_codes",
    "missing_opening_window_stop_ids", "metrics", "stop_traces", "content_hash",
  ])
      || value.schema_version !== "evaluated-time-window-schedule-accounting-v1"
      || value.evaluator_version !== evaluatorVersion
      || value.plan_id !== planId || value.plan_content_hash !== planContentHash
      || value.route_matrix_id !== routeMatrixId
      || !["failed", "incomplete_evidence", "passed"].includes(value.status)
      || !exactStringArray(value.modeled_components, [
        "road_travel", "visit_duration", "opening_wait_when_window_available",
        "required_time_window_wait", "required_latest_departure", "day_limit",
      ])
      || !exactStringArray(value.unavailable_components, [
        "parking_dropoff", "walking_transfer", "queue_wait", "service_buffer",
      ])
      || !Array.isArray(value.blocking_codes)
      || !value.blocking_codes.every((code) => allowedBlockingCodes.has(code))
      || new Set(value.blocking_codes).size !== value.blocking_codes.length
      || !Array.isArray(value.missing_opening_window_stop_ids)
      || !value.missing_opening_window_stop_ids.every((stopId) => typeof stopId === "string" && stopId)
      || !exactStringArray(
        value.missing_opening_window_stop_ids,
        [...value.missing_opening_window_stop_ids].sort(),
      )
      || !exactKeys(value.metrics, durationScheduleMetricKeys())
      || !Object.values(value.metrics).every((metric) => Number.isFinite(metric) && metric >= 0)
      || value.metrics.route_required_leg_count !== routeValidation.required_leg_count
      || value.metrics.route_present_leg_count !== routeValidation.present_leg_count
      || value.metrics.route_road_validated_leg_count !== routeValidation.road_validated_leg_count
      || value.metrics.route_fallback_leg_count !== routeValidation.fallback_leg_count
      || value.metrics.route_missing_leg_count !== routeValidation.missing_leg_count
      || !Array.isArray(value.stop_traces)) return null;
  const expectedStatus = value.blocking_codes.length
    ? "failed"
    : value.missing_opening_window_stop_ids.length ? "incomplete_evidence" : "passed";
  if (value.status !== expectedStatus) return null;
  const timelineStops = acceptedPlanTimeWindowTraceStops();
  if (!timelineStops) return null;
  if (value.stop_traces.length !== timelineStops.length) return null;
  const incomingByStop = new Map();
  const routeMinutesByDay = new Map();
  for (const leg of routeLegEvidence) {
    const key = `${leg.day}:${leg.destination_id}`;
    if (incomingByStop.has(key)) return null;
    incomingByStop.set(key, leg);
    routeMinutesByDay.set(
      leg.day, (routeMinutesByDay.get(leg.day) || 0) + leg.duration_minutes,
    );
  }
  const tracesByDay = new Map();
  const priorDepartureByDay = new Map();
  for (const [index, rawTrace] of value.stop_traces.entries()) {
    const expected = timelineStops[index];
    const incoming = incomingByStop.get(`${expected.day}:${expected.stop_id}`);
    const expectedOrigin = expected.sequence_index === 0
      ? incoming?.origin_id
      : timelineStops[index - 1]?.stop_id;
    const expectedRoadArrival = (
      priorDepartureByDay.get(expected.day) ?? configuredDayStartMinute
    ) + (incoming?.duration_minutes ?? Number.NaN);
    if (!incoming || incoming.origin_id !== expectedOrigin) return null;
    const trace = normalizeTimeWindowTrace(rawTrace, configuredDefaultVisitMinutes, {
      expected, incoming, expectedRoadArrival,
    });
    if (!trace) return null;
    priorDepartureByDay.set(trace.day, trace.departure_minute);
    if (!tracesByDay.has(trace.day)) tracesByDay.set(trace.day, []);
    tracesByDay.get(trace.day).push(trace);
  }
  const missingOpeningFromTraces = value.stop_traces
    .filter((trace) => trace.opening_start_minute === null && trace.opening_end_minute === null)
    .map((trace) => trace.stop_id)
    .sort();
  if (missingOpeningFromTraces.length !== timelineStops.length
      || !exactStringArray(value.missing_opening_window_stop_ids, missingOpeningFromTraces)) return null;
  for (const { day } of state.workspace?.timeline || []) {
    const travel = value.metrics[`day_${day}_travel_minutes`];
    const visit = value.metrics[`day_${day}_visit_minutes`];
    const waiting = value.metrics[`day_${day}_waiting_minutes`];
    const total = value.metrics[`day_${day}_total_minutes`];
    const limit = value.metrics[`day_${day}_limit_minutes`];
    const slack = value.metrics[`day_${day}_slack_minutes`];
    const overrun = value.metrics[`day_${day}_overrun_minutes`];
    const traces = tracesByDay.get(day) || [];
    const traceVisit = traces.reduce((sum, trace) => sum + trace.visit_minutes, 0);
    const traceWaiting = traces.reduce(
      (sum, trace) => sum + (trace.opening_wait_minutes ?? 0)
        + (trace.required_window_wait_minutes ?? 0),
      0,
    );
    if (Math.abs(travel - (routeMinutesByDay.get(day) || 0)) > 1e-8
        || Math.abs(total - travel - visit - waiting) > 1e-8
        || Math.abs(slack - Math.max(0, limit - total)) > 1e-8
        || Math.abs(overrun - Math.max(0, total - limit)) > 1e-8
        || Math.abs(visit - traceVisit) > 1e-8
        || Math.abs(waiting - traceWaiting) > 1e-8) return null;
  }
  const expectedBlockingCodes = [];
  for (const { day } of state.workspace?.timeline || []) {
    for (const trace of tracesByDay.get(day) || []) {
      for (const code of trace.failure_codes) {
        if (!expectedBlockingCodes.includes(code)) expectedBlockingCodes.push(code);
      }
    }
    if (value.metrics[`day_${day}_overrun_minutes`] > 0
        && !expectedBlockingCodes.includes("day_time_exceeded")) {
      expectedBlockingCodes.push("day_time_exceeded");
    }
  }
  if (!exactStringArray(value.blocking_codes, expectedBlockingCodes)) return null;
  const payload = { ...value };
  delete payload.content_hash;
  const hash = await computedStableContentHash(payload);
  if (hash === null || hash !== value.content_hash) return null;
  return value;
}

async function normalizeTimeWindowScheduleImpact(value, identity, proposal, planDiff, routeValidation) {
  if (!exactKeys(value, [
    "schema_version", "evaluator_version", "route_matrix_id", "configured_default_visit_minutes",
    "configured_day_start_minute", "route_leg_evidence", "time_window_cost_policy",
    "affected_days", "target_stop_ids", "parent", "child", "content_hash",
  ])
      || value.schema_version !== "evaluated-time-window-schedule-impact-v1"
      || typeof value.evaluator_version !== "string" || !value.evaluator_version
      || value.route_matrix_id !== routeValidation.matrix_id
      || value.configured_default_visit_minutes !== 45
      || value.configured_day_start_minute !== 540
      || !normalizeTimeWindowCostPolicy(value.time_window_cost_policy)
      || !exactKeys(identity, [
        "content_hash", "evaluator_version", "route_matrix_id", "parent_plan_id",
        "parent_plan_content_hash", "child_plan_id", "child_plan_content_hash",
        "certificate_id", "certificate_content_hash",
      ])) return null;
  const changes = planDiff.time_window_changes;
  const affectedDays = [...new Set(changes.map((change) => change.day))].sort((left, right) => left - right);
  const targetStopIds = changes.map((change) => change.stop_id).sort();
  if (!exactStringArray(value.affected_days, affectedDays)
      || !exactStringArray(value.target_stop_ids, targetStopIds)
      || !timeWindowDiffMatchesCostPolicy(planDiff, value.time_window_cost_policy)) return null;
  const routeLegEvidence = normalizeTimeWindowRouteEvidence(
    value.route_leg_evidence, routeValidation,
  );
  if (!routeLegEvidence) return null;
  const common = {
    evaluatorVersion: value.evaluator_version,
    routeMatrixId: value.route_matrix_id,
    routeValidation,
    configuredDefaultVisitMinutes: value.configured_default_visit_minutes,
    configuredDayStartMinute: value.configured_day_start_minute,
    routeLegEvidence,
  };
  const parent = await normalizeTimeWindowScheduleAccounting(value.parent, {
    ...common, planId: proposal.parent_plan_id, planContentHash: proposal.parent_plan_content_hash,
  });
  const child = await normalizeTimeWindowScheduleAccounting(value.child, {
    ...common, planId: proposal.child_plan_id, planContentHash: proposal.child_plan_content_hash,
  });
  if (!parent || !child
      || !exactStringArray(parent.missing_opening_window_stop_ids, child.missing_opening_window_stop_ids)) return null;
  const changeByTarget = new Map(changes.map((change) => [change.stop_id, change]));
  for (const [index, parentTrace] of parent.stop_traces.entries()) {
    const childTrace = child.stop_traces[index];
    const change = changeByTarget.get(parentTrace.stop_id);
    if (parentTrace.stop_id !== childTrace.stop_id
        || parentTrace.day !== childTrace.day
        || parentTrace.sequence_index !== childTrace.sequence_index
        || parentTrace.time_window_constraint !== null
        || parentTrace.time_window_source !== null
        || parentTrace.required_window_wait_minutes !== null
        || parentTrace.latest_departure_status !== "not_set"
        || parentTrace.latest_departure_overrun_minutes !== null
        || parentTrace.opening_start_minute !== childTrace.opening_start_minute
        || parentTrace.opening_end_minute !== childTrace.opening_end_minute
        || parentTrace.opening_window_source !== childTrace.opening_window_source
        || parentTrace.incoming_origin_id !== childTrace.incoming_origin_id
        || parentTrace.incoming_travel_minutes !== childTrace.incoming_travel_minutes
        || parentTrace.incoming_route_query_hash !== childTrace.incoming_route_query_hash
        || parentTrace.visit_minutes !== childTrace.visit_minutes
        || parentTrace.visit_duration_source !== childTrace.visit_duration_source
        || canonicalJson(parentTrace.time_window_constraint)
          !== canonicalJson(change ? change.from_constraint : childTrace.time_window_constraint)
        || canonicalJson(childTrace.time_window_constraint)
          !== canonicalJson(change ? change.to_constraint : parentTrace.time_window_constraint)) return null;
    if (parentTrace.sequence_index === 0) {
      if (parentTrace.road_arrival_minute !== childTrace.road_arrival_minute) return null;
    } else {
      const parentPrevious = parent.stop_traces[index - 1];
      const childPrevious = child.stop_traces[index - 1];
      if (Math.abs(
        childTrace.road_arrival_minute - parentTrace.road_arrival_minute
          - childPrevious.departure_minute + parentPrevious.departure_minute,
      ) > 1e-8) return null;
    }
  }
  for (const { day } of state.workspace?.timeline || []) {
    const parentMetrics = parent.metrics;
    const childMetrics = child.metrics;
    for (const name of ["travel", "visit", "limit"]) {
      if (childMetrics[`day_${day}_${name}_minutes`] !== parentMetrics[`day_${day}_${name}_minutes`]) return null;
    }
    const waitingDelta = childMetrics[`day_${day}_waiting_minutes`]
      - parentMetrics[`day_${day}_waiting_minutes`];
    if (Math.abs(
      childMetrics[`day_${day}_total_minutes`] - parentMetrics[`day_${day}_total_minutes`]
        - waitingDelta,
    ) > 1e-8) return null;
  }
  for (const key of durationScheduleMetricKeys().filter((name) => name.startsWith("route_"))) {
    if (parent.metrics[key] !== child.metrics[key]) return null;
  }
  const payload = { ...value };
  delete payload.content_hash;
  const hash = await computedStableContentHash(payload);
  if (hash === null || hash !== value.content_hash
      || identity.content_hash !== hash
      || identity.evaluator_version !== value.evaluator_version
      || identity.route_matrix_id !== value.route_matrix_id
      || identity.parent_plan_id !== proposal.parent_plan_id
      || identity.parent_plan_content_hash !== proposal.parent_plan_content_hash
      || identity.child_plan_id !== proposal.child_plan_id
      || identity.child_plan_content_hash !== proposal.child_plan_content_hash
      || identity.certificate_id !== proposal.certificate_id
      || identity.certificate_content_hash !== proposal.certificate_content_hash) return null;
  return value;
}

async function normalizeTimeWindowCertificateScheduleEvidence(value, proposal, scheduleImpact) {
  if (!exactKeys(value, [
    "schema_version", "certificate_id", "certificate_content_hash", "plan_id",
    "plan_content_hash", "evaluator_version", "evaluation_status", "comparison_eligibility",
    "warning_codes", "failure_codes", "schedule_metrics", "content_hash",
  ])
      || value.schema_version !== "evaluated-time-window-certificate-schedule-evidence-v1"
      || value.certificate_id !== proposal.certificate_id
      || value.certificate_content_hash !== proposal.certificate_content_hash
      || value.plan_id !== proposal.child_plan_id
      || value.plan_content_hash !== proposal.child_plan_content_hash
      || value.evaluator_version !== scheduleImpact.evaluator_version
      || value.evaluation_status !== proposal.evaluation_status
      || value.comparison_eligibility !== proposal.eligibility
      || !exactKeys(value.schedule_metrics, [
        ...durationScheduleMetricKeys(), "schedule_missing_opening_window_count",
      ])
      || !Object.values(value.schedule_metrics).every((metric) => Number.isFinite(metric) && metric >= 0)) return null;
  const child = scheduleImpact.child;
  for (const [name, metric] of Object.entries(child.metrics)) {
    if (value.schedule_metrics[name] !== metric) return null;
  }
  const missingCount = child.missing_opening_window_stop_ids.length;
  const expectedWarnings = missingCount > 0 ? ["opening_window_evidence_missing"] : [];
  const expectedEvaluation = child.status === "failed"
    ? "FAILED"
    : child.status === "incomplete_evidence" ? "PASSED_WITH_WARNINGS" : "PASSED";
  const expectedEligibility = child.status === "failed" ? "ineligible" : "eligible";
  if (value.schedule_metrics.schedule_missing_opening_window_count !== missingCount
      || !exactStringArray(value.warning_codes, expectedWarnings)
      || !exactStringArray(value.failure_codes, child.blocking_codes)
      || value.evaluation_status !== expectedEvaluation
      || value.comparison_eligibility !== expectedEligibility
      || proposal.state !== expectedEligibility || proposal.eligibility !== expectedEligibility
      || proposal.decision_eligible !== (expectedEligibility === "eligible")
      || proposal.ranking_eligible !== false
      || proposal.acceptance_eligible !== false
      || proposal.acceptance_blocking_code !== "acceptance_not_enabled_until_w5"
      || proposal.repair.certificate.failure_count !== value.failure_codes.length) return null;
  const payload = { ...value };
  delete payload.content_hash;
  const hash = await computedStableContentHash(payload);
  if (hash === null || hash !== value.content_hash) return null;
  return value;
}

function normalizeRoleCostPolicy(value) {
  const expectedWeights = {
    "": 1,
    locked: 1000,
    booked: 500,
    hard: 100,
    strong: 50,
    soft: 10,
    weak: 5,
    preference: 1,
    test_only: 0,
  };
  if (!exactKeys(value, ["base_change_cost", "formula", "constraint_strength_weights"])
      || value.base_change_cost !== 0.25
      || value.formula !== "base_times_role_scoped_strength_weight"
      || !exactKeys(value.constraint_strength_weights, Object.keys(expectedWeights))
      || !Object.entries(expectedWeights).every(
        ([strength, weight]) => value.constraint_strength_weights[strength] === weight,
      )) return null;
  return value;
}

function normalizeRoleStates(value, planDiff, child) {
  const acceptedStops = acceptedPlanTimeWindowTraceStops();
  const acceptedPlan = state.workspace?.geography?.plans?.find(
    (plan) => plan.plan_id === state.session?.accepted_plan_id,
  );
  if (!acceptedStops || !acceptedPlan || !Array.isArray(value)
      || value.length !== acceptedStops.length) return null;
  const acceptedFeatures = new Map(acceptedPlan.stops.features.map(
    (feature) => [feature.properties?.stop_id, feature],
  ));
  const changes = new Map(planDiff.role_changes.map((change) => [change.stop_id, change]));
  for (const [index, roleState] of value.entries()) {
    const expected = acceptedStops[index];
    const feature = acceptedFeatures.get(expected.stop_id);
    const parentRole = feature?.properties?.itinerary_role ?? null;
    const parentSource = feature?.properties?.itinerary_role_source;
    const change = changes.get(expected.stop_id);
    const expectedRole = child && change ? change.to_role : parentRole;
    const expectedSource = child && change ? change.to_source : parentSource;
    if (!exactKeys(roleState, [
      "stop_id", "day", "itinerary_role", "itinerary_role_source",
    ])
        || roleState.stop_id !== expected.stop_id || roleState.day !== expected.day
        || roleState.itinerary_role !== expectedRole
        || roleState.itinerary_role_source !== expectedSource
        || (roleState.itinerary_role === null && roleState.itinerary_role_source !== "unavailable")
        || (roleState.itinerary_role !== null
          && (!TYPED_EDIT_VOCABULARIES.stop_roles.includes(roleState.itinerary_role)
            || roleState.itinerary_role_source !== "user_declared_itinerary_role"))) return null;
  }
  return value;
}

function scheduleAccountingWithoutIdentity(value) {
  const copy = structuredClone(value);
  delete copy.plan_id;
  delete copy.plan_content_hash;
  delete copy.content_hash;
  return copy;
}

function normalizeRoleGeography(value, planDiff, roleImpact) {
  const parentPlan = state.workspace?.geography?.plans?.find(
    (plan) => plan.plan_id === state.session?.accepted_plan_id,
  );
  if (!parentPlan || !durationRouteEvidenceUnchanged(value)) return null;
  const parentByStop = new Map(parentPlan.stops.features.map(
    (feature) => [feature.properties?.stop_id, feature],
  ));
  const childByStop = new Map(value.stops.features.map(
    (feature) => [feature.properties?.stop_id, feature],
  ));
  const parentStates = new Map(roleImpact.parent_role_states.map((row) => [row.stop_id, row]));
  const childStates = new Map(roleImpact.child_role_states.map((row) => [row.stop_id, row]));
  const changes = new Map(planDiff.role_changes.map((change) => [change.stop_id, change]));
  if (parentByStop.size !== childByStop.size || parentByStop.size !== parentStates.size) return null;
  for (const [stopId, parentFeature] of parentByStop.entries()) {
    const childFeature = childByStop.get(stopId);
    const parentState = parentStates.get(stopId);
    const childState = childStates.get(stopId);
    const change = changes.get(stopId);
    if (!childFeature || !parentState || !childState
        || !sameJson(parentFeature.geometry, childFeature.geometry)
        || parentFeature.properties?.role !== "original"
        || childFeature.properties?.role !== "draft_preview"
        || parentFeature.properties?.itinerary_role !== parentState.itinerary_role
        || parentFeature.properties?.itinerary_role_source !== parentState.itinerary_role_source
        || childFeature.properties?.itinerary_role !== childState.itinerary_role
        || childFeature.properties?.itinerary_role_source !== childState.itinerary_role_source
        || (change && (childState.itinerary_role !== change.to_role
          || childState.itinerary_role_source !== change.to_source))
        || (!change && !sameJson(parentState, childState))) return null;
    const stableProperties = (properties) => {
      const copy = structuredClone(properties);
      for (const field of [
        "plan_id", "content_hash", "role", "itinerary_role", "itinerary_role_source",
      ]) delete copy[field];
      return copy;
    };
    if (!sameJson(stableProperties(parentFeature.properties), stableProperties(childFeature.properties))) {
      return null;
    }
  }
  return value;
}

async function normalizeRoleImpact(value, identity, proposal, planDiff, routeValidation) {
  if (!exactKeys(value, [
    "schema_version", "evaluator_version", "route_matrix_id", "configured_default_visit_minutes",
    "configured_day_start_minute", "supported_evaluated_roles", "semantic_scope",
    "role_constraint_evidence", "route_leg_evidence", "role_cost_policy", "target_stop_ids",
    "affected_days", "parent_role_states", "child_role_states", "parent_schedule",
    "child_schedule", "invariance", "content_hash",
  ])
      || value.schema_version !== "evaluated-role-impact-v1"
      || typeof value.evaluator_version !== "string" || !value.evaluator_version
      || value.route_matrix_id !== routeValidation.matrix_id
      || value.configured_default_visit_minutes !== 45
      || value.configured_day_start_minute !== 540
      || !exactStringArray(value.supported_evaluated_roles, EVALUATED_STOP_ROLES)
      || !sameJson(value.semantic_scope, ROLE_SEMANTIC_SCOPE)
      || !exactKeys(identity, [
        "content_hash", "evaluator_version", "route_matrix_id", "parent_plan_id",
        "parent_plan_content_hash", "child_plan_id", "child_plan_content_hash",
        "certificate_id", "certificate_content_hash", "role_constraint_evidence_content_hash",
      ])) return null;
  const workspaceConstraintEvidence = await normalizeRoleConstraintEvidence(
    state.workspace?.role_constraint_evidence,
  );
  const impactConstraintEvidence = await normalizeRoleConstraintEvidence(value.role_constraint_evidence);
  const costPolicy = normalizeRoleCostPolicy(value.role_cost_policy);
  if (!workspaceConstraintEvidence || !impactConstraintEvidence || !costPolicy
      || !sameJson(workspaceConstraintEvidence, impactConstraintEvidence)) return null;
  for (const change of planDiff.role_changes) {
    const strength = roleConstraintStrength(impactConstraintEvidence, change.stop_id, costPolicy);
    if (change.owner_strength !== strength
        || change.cost !== costPolicy.base_change_cost
          * costPolicy.constraint_strength_weights[strength]) return null;
  }
  const affectedDays = [...new Set(planDiff.role_changes.map((change) => change.day))]
    .sort((left, right) => left - right);
  const targetStopIds = planDiff.role_changes.map((change) => change.stop_id).sort();
  if (!exactStringArray(value.affected_days, affectedDays)
      || !exactStringArray(value.target_stop_ids, targetStopIds)
      || !normalizeRoleStates(value.parent_role_states, planDiff, false)
      || !normalizeRoleStates(value.child_role_states, planDiff, true)
      || !exactKeys(value.invariance, [
        "sequence_unchanged", "ordered_days_unchanged", "route_ids_by_day_unchanged",
        "route_legs_unchanged", "schedule_accounting_unchanged",
      ])
      || !Object.values(value.invariance).every((item) => item === true)) return null;
  const routeLegEvidence = normalizeTimeWindowRouteEvidence(value.route_leg_evidence, routeValidation);
  if (!routeLegEvidence) return null;
  const common = {
    evaluatorVersion: value.evaluator_version,
    routeMatrixId: value.route_matrix_id,
    routeValidation,
    configuredDefaultVisitMinutes: value.configured_default_visit_minutes,
    configuredDayStartMinute: value.configured_day_start_minute,
    routeLegEvidence,
  };
  const parentSchedule = await normalizeTimeWindowScheduleAccounting(value.parent_schedule, {
    ...common, planId: proposal.parent_plan_id, planContentHash: proposal.parent_plan_content_hash,
  });
  const childSchedule = await normalizeTimeWindowScheduleAccounting(value.child_schedule, {
    ...common, planId: proposal.child_plan_id, planContentHash: proposal.child_plan_content_hash,
  });
  if (!parentSchedule || !childSchedule
      || !sameJson(
        scheduleAccountingWithoutIdentity(parentSchedule),
        scheduleAccountingWithoutIdentity(childSchedule),
      )) return null;
  const payload = { ...value };
  delete payload.content_hash;
  const hash = await computedStableContentHash(payload);
  if (hash === null || hash !== value.content_hash
      || identity.content_hash !== hash
      || identity.evaluator_version !== value.evaluator_version
      || identity.route_matrix_id !== value.route_matrix_id
      || identity.parent_plan_id !== proposal.parent_plan_id
      || identity.parent_plan_content_hash !== proposal.parent_plan_content_hash
      || identity.child_plan_id !== proposal.child_plan_id
      || identity.child_plan_content_hash !== proposal.child_plan_content_hash
      || identity.certificate_id !== proposal.certificate_id
      || identity.certificate_content_hash !== proposal.certificate_content_hash
      || identity.role_constraint_evidence_content_hash
        !== impactConstraintEvidence.content_hash) return null;
  return value;
}

async function normalizeRoleCertificateRecord(
  value, proposal, roleImpact, expectedWarningRecords, expectedFailures,
) {
  if (!exactKeys(value, [
    "certificate_id", "evaluation_id", "plan_id", "source_run_id", "plan_content_hash",
    "evaluator_version", "artifact_grounding_status", "hard_feasibility_status",
    "evaluation_status", "comparison_eligibility", "eligible", "reward_eligibility",
    "failures", "warnings", "failure_count", "warning_count", "blocking_warning_count",
    "nonblocking_warning_count", "metrics", "route_validation", "evaluated_at",
    "schema_version", "content_hash",
  ])
      || value.schema_version !== "plan-evaluation-certificate-v1"
      || value.certificate_id !== proposal.certificate_id
      || value.evaluation_id !== proposal.certificate_id
      || value.plan_id !== proposal.child_plan_id
      || value.plan_content_hash !== proposal.child_plan_content_hash
      || !/^w4r_role_run_[0-9a-f]{16}$/.test(value.source_run_id)
      || value.evaluator_version !== roleImpact.evaluator_version
      || value.artifact_grounding_status !== proposal.evidence.artifact_grounding_status
      || value.hard_feasibility_status !== proposal.evidence.hard_feasibility_status
      || value.evaluation_status !== proposal.evaluation_status
      || value.comparison_eligibility !== proposal.eligibility
      || value.eligible !== (proposal.eligibility === "eligible")
      || value.reward_eligibility !== value.eligible
      || !Array.isArray(value.failures) || !Array.isArray(value.warnings)
      || value.failure_count !== value.failures.length
      || value.warning_count !== value.warnings.length
      || value.blocking_warning_count !== value.failure_count
      || value.nonblocking_warning_count !== value.warning_count
      || !exactStringArray(value.failures.map((finding) => finding?.code), expectedFailures)
      || !sameJson(value.warnings, expectedWarningRecords)
      || !exactKeys(value.metrics, [
        "selected_stop_count", "sequence_stop_count", "preserved_stop_count", "preservation_rate",
        ...durationScheduleMetricKeys(), "schedule_missing_opening_window_count",
      ])
      || !Object.values(value.metrics).every((metric) => Number.isFinite(metric) && metric >= 0)
      || value.metrics.selected_stop_count !== roleImpact.child_role_states.length
      || value.metrics.sequence_stop_count !== roleImpact.child_role_states.length
      || value.metrics.preserved_stop_count !== roleImpact.child_role_states.length
      || value.metrics.preservation_rate !== 1
      || !sameJson(value.route_validation, proposal.route_validation)
      || typeof value.evaluated_at !== "string"
      || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(value.evaluated_at)) return null;
  for (const failure of value.failures) {
    if (!exactKeys(failure, ["code", "message", "severity", "category", "evidence_refs"])
        || typeof failure.message !== "string" || !failure.message
        || failure.severity !== "failure"
        || typeof failure.category !== "string" || !failure.category
        || !Array.isArray(failure.evidence_refs)
        || !failure.evidence_refs.every((ref) => typeof ref === "string" && ref)) return null;
  }
  const payload = { ...value };
  delete payload.content_hash;
  const hash = await computedStableContentHash(payload);
  if (hash === null || hash !== value.content_hash
      || hash !== proposal.certificate_content_hash
      || hash !== proposal.certificate_identity.content_hash) return null;
  return value;
}

async function normalizeRoleCertificateEvidence(value, proposal, roleImpact, planDiff) {
  if (!exactKeys(value, [
    "schema_version", "certificate_id", "certificate_content_hash", "plan_id",
    "plan_content_hash", "evaluator_version", "evaluation_status", "comparison_eligibility",
    "warning_codes", "failure_codes", "schedule_metrics", "semantic_scope",
    "role_constraint_evidence_content_hash", "certificate_record", "content_hash",
  ])
      || value.schema_version !== "evaluated-role-certificate-evidence-v1"
      || value.certificate_id !== proposal.certificate_id
      || value.certificate_content_hash !== proposal.certificate_content_hash
      || value.plan_id !== proposal.child_plan_id
      || value.plan_content_hash !== proposal.child_plan_content_hash
      || value.evaluator_version !== roleImpact.evaluator_version
      || value.evaluation_status !== proposal.evaluation_status
      || value.comparison_eligibility !== proposal.eligibility
      || !sameJson(value.semantic_scope, ROLE_SEMANTIC_SCOPE)
      || value.role_constraint_evidence_content_hash
        !== roleImpact.role_constraint_evidence.content_hash
      || !exactKeys(value.schedule_metrics, [
        ...durationScheduleMetricKeys(), "schedule_missing_opening_window_count",
      ])
      || !Object.values(value.schedule_metrics).every(
        (metric) => Number.isFinite(metric) && metric >= 0,
      )) return null;
  const child = roleImpact.child_schedule;
  for (const [name, metric] of Object.entries(child.metrics)) {
    if (value.schedule_metrics[name] !== metric) return null;
  }
  const changeByTarget = new Map(planDiff.role_changes.map((change) => [change.stop_id, change]));
  const weightedMismatches = roleImpact.role_constraint_evidence.constraints.filter((constraint) => {
    const change = changeByTarget.get(constraint.target_stop_id);
    return change && constraint.required_role !== change.to_role
      && constraint.permission_semantics === "weighted_mismatch_allowed";
  });
  const missingCount = child.missing_opening_window_stop_ids.length;
  const expectedWarnings = [
    ...weightedMismatches.map(() => "owned_role_constraint_unsatisfied"),
    ...(missingCount ? ["opening_window_evidence_missing"] : []),
  ];
  const expectedWarningRecords = [
    ...weightedMismatches.map((constraint) => ({
      code: "owned_role_constraint_unsatisfied",
      message: `Owned itinerary-role constraint '${constraint.constraint_id}' is not satisfied.`,
      severity: "warning",
      category: "soft",
      evidence_refs: [constraint.constraint_id],
    })),
    ...(missingCount ? [{
      code: "opening_window_evidence_missing",
      message: "Opening-window evidence is unavailable for one or more evaluated stops.",
      severity: "warning",
      category: "soft",
      evidence_refs: child.missing_opening_window_stop_ids,
    }] : []),
  ];
  const expectedFailures = child.blocking_codes;
  const expectedEvaluation = expectedFailures.length
    ? "FAILED"
    : expectedWarnings.length ? "PASSED_WITH_WARNINGS" : "PASSED";
  const expectedEligibility = expectedFailures.length ? "ineligible" : "eligible";
  if (value.schedule_metrics.schedule_missing_opening_window_count !== missingCount
      || !exactStringArray(value.warning_codes, expectedWarnings)
      || !exactStringArray(value.failure_codes, expectedFailures)
      || value.evaluation_status !== expectedEvaluation
      || value.comparison_eligibility !== expectedEligibility
      || proposal.state !== expectedEligibility || proposal.eligibility !== expectedEligibility
      || proposal.decision_eligible !== (expectedEligibility === "eligible")
      || proposal.ranking_eligible !== false
      || proposal.acceptance_eligible !== false
      || proposal.acceptance_blocking_code !== "acceptance_not_enabled_until_w5"
      || proposal.repair.certificate.failure_count !== expectedFailures.length) return null;
  const certificateRecord = await normalizeRoleCertificateRecord(
    value.certificate_record, proposal, roleImpact, expectedWarningRecords, expectedFailures,
  );
  if (!certificateRecord) return null;
  for (const [name, metric] of Object.entries(value.schedule_metrics)) {
    if (certificateRecord.metrics[name] !== metric) return null;
  }
  const payload = { ...value };
  delete payload.content_hash;
  const hash = await computedStableContentHash(payload);
  if (hash === null || hash !== value.content_hash) return null;
  return value;
}

function validFiniteOrNull(value) {
  return value === null || (typeof value === "number" && Number.isFinite(value));
}

function normalizeRouteValidation(value) {
  if (!exactKeys(value, [
    "matrix_id", "context_snapshot_id", "required_leg_count", "present_leg_count",
    "road_validated_leg_count", "fallback_leg_count", "invalid_value_count",
    "missing_leg_count", "publication_ready", "source_bundle_id",
    "source_content_sha256", "errors", "warnings",
  ])) return null;
  const counts = [
    value.required_leg_count, value.present_leg_count, value.road_validated_leg_count,
    value.fallback_leg_count, value.invalid_value_count, value.missing_leg_count,
  ];
  if (!/^route_matrix_[0-9a-f]{16}$/.test(value.matrix_id)
      || typeof value.context_snapshot_id !== "string"
      || !value.context_snapshot_id || value.context_snapshot_id.length > 128
      || !/^route_bundle_[0-9a-f]{16}$/.test(value.source_bundle_id)
      || !/^[0-9a-f]{64}$/.test(value.source_content_sha256)
      || !counts.every((count) => Number.isInteger(count) && count >= 0)
      || value.required_leg_count === 0
      || value.present_leg_count !== value.required_leg_count
      || value.road_validated_leg_count !== value.required_leg_count
      || value.fallback_leg_count !== 0
      || value.invalid_value_count !== 0
      || value.missing_leg_count !== 0
      || value.publication_ready !== true
      || !Array.isArray(value.errors) || value.errors.length !== 0
      || !Array.isArray(value.warnings)
      || !value.warnings.every((item) => typeof item === "string")) return null;
  return value;
}

function normalizePreviewRepair(value, proposal) {
  if (!exactKeys(value, [
    "status", "status_state", "result", "changed", "unchanged", "permissions",
    "tradeoffs", "accepted_radius", "certificate", "primary_action",
  ])
      || !exactKeys(value.changed, [
        "added", "removed", "affected_day_count", "road_change_count", "lodging_change_count",
      ])
      || !exactKeys(value.unchanged, ["days", "day_count"])
      || !exactKeys(value.permissions, [
        "booked_change_count", "locked_change_count", "permission_required", "message",
      ])
      || !exactKeys(value.tradeoffs, [
        "weighted_edit_cost", "preservation_rate", "utility_retained", "weather_risk_delta",
      ])
      || !exactKeys(value.certificate, [
        "id", "eligible", "evaluation_status", "failure_count", "failures",
      ])) return null;
  const integers = [
    value.changed.affected_day_count, value.changed.road_change_count,
    value.changed.lodging_change_count, value.unchanged.day_count,
    value.permissions.booked_change_count, value.permissions.locked_change_count,
    value.certificate.failure_count,
  ];
  if (!Array.isArray(value.changed.added) || !Array.isArray(value.changed.removed)
      || !value.changed.added.every((item) => typeof item === "string")
      || !value.changed.removed.every((item) => typeof item === "string")
      || !Array.isArray(value.unchanged.days)
      || !value.unchanged.days.every((day) => Number.isInteger(day) && day > 0)
      || !integers.every((item) => Number.isInteger(item) && item >= 0)
      || value.unchanged.day_count !== value.unchanged.days.length
      || typeof value.permissions.permission_required !== "boolean"
      || typeof value.permissions.message !== "string"
      || !Object.values(value.tradeoffs).every(validFiniteOrNull)
      || !Array.isArray(value.certificate.failures)
      || value.certificate.failure_count !== value.certificate.failures.length
      || value.certificate.id !== proposal.certificate_id
      || value.certificate.evaluation_status !== proposal.evaluation_status
      || value.certificate.eligible !== (proposal.state === "eligible")
      || typeof value.result !== "string"
      || value.primary_action !== "Review evidence") return null;
  return value;
}

function validCoordinate(value) {
  return Array.isArray(value) && value.length === 2
    && Number.isFinite(value[0]) && value[0] >= -180 && value[0] <= 180
    && Number.isFinite(value[1]) && value[1] >= -90 && value[1] <= 90;
}

function sameCoordinate(left, right) {
  return validCoordinate(left) && validCoordinate(right)
    && Math.abs(left[0] - right[0]) <= 1e-6
    && Math.abs(left[1] - right[1]) <= 1e-6;
}

function lineDistanceMeters(coordinates) {
  let total = 0;
  const radians = (value) => value * Math.PI / 180;
  for (let index = 1; index < coordinates.length; index += 1) {
    const left = coordinates[index - 1];
    const right = coordinates[index];
    if (!validCoordinate(left) || !validCoordinate(right)) return null;
    const latitudeDelta = radians(right[1] - left[1]);
    const longitudeDelta = radians(right[0] - left[0]);
    const leftLatitude = radians(left[1]);
    const rightLatitude = radians(right[1]);
    const haversine = Math.sin(latitudeDelta / 2) ** 2
      + Math.cos(leftLatitude) * Math.cos(rightLatitude) * Math.sin(longitudeDelta / 2) ** 2;
    total += 6371008.8 * 2 * Math.atan2(Math.sqrt(haversine), Math.sqrt(1 - haversine));
  }
  return total;
}

function routeChainMatchesEvidence(value, routeValidation) {
  const path = [...value.route_path.features].sort(
    (left, right) => left.properties?.occurrence_index - right.properties?.occurrence_index,
  );
  const legs = [...value.validated_legs.features].sort(
    (left, right) => left.properties?.leg_index - right.properties?.leg_index,
  );
  if (path.length !== routeValidation.required_leg_count + 1
      || legs.length !== routeValidation.required_leg_count
      || value.gaps.features.length !== 0) return false;
  const pathPropertyKeys = [
    "plan_id", "content_hash", "role", "occurrence_index", "node_id", "selected_stop",
    "selected_sequence_index", "route_anchor", "arrival_day", "departure_day", "coordinate_source",
  ];
  const legPropertyKeys = [
    "plan_id", "content_hash", "role", "leg_index", "requirement_id", "day", "from_day",
    "to_day", "cross_day", "travel_mode", "evidence_scope", "origin_id", "destination_id",
    "validation_status", "route_leg_id", "road_validated", "fallback_used", "geometry_source",
    "distance_m", "duration_s",
  ];
  for (const [index, feature] of path.entries()) {
    if (!exactKeys(feature, ["type", "id", "geometry", "properties"])
        || feature.type !== "Feature"
        || !exactKeys(feature.properties, pathPropertyKeys)
        || feature.properties.occurrence_index !== index
        || typeof feature.properties.node_id !== "string" || !feature.properties.node_id
        || feature.id !== `${value.plan_id}:route-path-node:${index}`
        || !exactKeys(feature.geometry, ["type", "coordinates"])
        || feature.geometry.type !== "Point" || !validCoordinate(feature.geometry.coordinates)) return false;
  }
  for (const [index, feature] of legs.entries()) {
    const properties = feature.properties;
    const geometry = feature.geometry;
    const origin = path[index];
    const destination = path[index + 1];
    if (!exactKeys(feature, ["type", "id", "geometry", "properties"])
        || feature.type !== "Feature"
        || !exactKeys(properties, legPropertyKeys)
        || properties.leg_index !== index + 1
        || feature.id !== `${value.plan_id}:leg:${index + 1}`
        || properties.requirement_id !== `${value.plan_id}:required-leg:${index + 1}`
        || properties.origin_id !== origin.properties.node_id
        || properties.destination_id !== destination.properties.node_id
        || properties.route_leg_id !== `leg_${properties.origin_id}_${properties.destination_id}`
        || properties.travel_mode !== "driving"
        || typeof properties.evidence_scope !== "string" || !properties.evidence_scope
        || properties.validation_status !== "road_validated"
        || properties.road_validated !== true || properties.fallback_used !== false
        || properties.geometry_source !== "cached_osrm_route_geometry"
        || typeof properties.cross_day !== "boolean"
        || !Number.isInteger(properties.day) || !Number.isInteger(properties.from_day)
        || !Number.isInteger(properties.to_day)
        || !Number.isFinite(properties.distance_m) || properties.distance_m <= 0
        || !Number.isFinite(properties.duration_s) || properties.duration_s <= 0
        || !exactKeys(geometry, ["type", "coordinates"])
        || geometry.type !== "LineString" || !Array.isArray(geometry.coordinates)
        || geometry.coordinates.length < 2
        || !sameCoordinate(geometry.coordinates[0], origin.geometry.coordinates)
        || !sameCoordinate(geometry.coordinates.at(-1), destination.geometry.coordinates)) return false;
    const geometryDistance = lineDistanceMeters(geometry.coordinates);
    if (geometryDistance === null
        || Math.abs(geometryDistance - properties.distance_m) / properties.distance_m > 0.08) return false;
  }
  return true;
}

function durationRouteEvidenceUnchanged(childPlan) {
  const parentPlan = state.workspace?.geography?.plans?.find(
    (plan) => plan.plan_id === state.session?.accepted_plan_id,
  );
  if (!parentPlan) return false;
  const pathProjection = (plan) => [...plan.route_path.features]
    .sort((left, right) => left.properties.occurrence_index - right.properties.occurrence_index)
    .map((feature) => ({
      geometry: feature.geometry,
      occurrence_index: feature.properties.occurrence_index,
      node_id: feature.properties.node_id,
      selected_stop: feature.properties.selected_stop,
      selected_sequence_index: feature.properties.selected_sequence_index,
      route_anchor: feature.properties.route_anchor,
      arrival_day: feature.properties.arrival_day,
      departure_day: feature.properties.departure_day,
      coordinate_source: feature.properties.coordinate_source,
    }));
  const legProjection = (plan) => [...plan.validated_legs.features]
    .sort((left, right) => left.properties.leg_index - right.properties.leg_index)
    .map((feature) => ({
      geometry: feature.geometry,
      leg_index: feature.properties.leg_index,
      day: feature.properties.day,
      from_day: feature.properties.from_day,
      to_day: feature.properties.to_day,
      cross_day: feature.properties.cross_day,
      travel_mode: feature.properties.travel_mode,
      origin_id: feature.properties.origin_id,
      destination_id: feature.properties.destination_id,
      validation_status: feature.properties.validation_status,
      route_leg_id: feature.properties.route_leg_id,
      road_validated: feature.properties.road_validated,
      fallback_used: feature.properties.fallback_used,
      geometry_source: feature.properties.geometry_source,
      distance_m: feature.properties.distance_m,
      duration_s: feature.properties.duration_s,
    }));
  return sameJson(pathProjection(childPlan), pathProjection(parentPlan))
    && sameJson(legProjection(childPlan), legProjection(parentPlan))
    && childPlan.gaps.features.length === 0
    && parentPlan.gaps.features.length === 0;
}

function normalizePreviewGeography(value, proposal, routeValidation, draft) {
  if (!exactKeys(value, [
    "plan_id", "content_hash", "role", "label", "status", "coverage",
    "route_path", "stops", "validated_legs", "gaps",
  ])
      || value.plan_id !== proposal.child_plan_id
      || value.content_hash !== proposal.child_plan_content_hash
      || value.role !== "draft_preview"
      || value.label !== "Evaluated draft preview"
      || value.status !== "ready"
      || !exactKeys(value.coverage, [
        "schema_version", "status", "route_path_node_count", "required_leg_count",
        "road_validated_leg_count", "gap_count", "itinerary_sequence_accounted", "complete",
      ])
      || value.coverage.status !== "complete"
      || value.coverage.complete !== true
      || value.coverage.itinerary_sequence_accounted !== true
      || value.coverage.required_leg_count !== routeValidation.required_leg_count
      || value.coverage.road_validated_leg_count !== routeValidation.road_validated_leg_count
      || value.coverage.gap_count !== 0
      || !routeChainMatchesEvidence(value, routeValidation)) return null;
  for (const name of ["route_path", "stops", "validated_legs", "gaps"]) {
    const collection = value[name];
    if (!exactKeys(collection, ["type", "features"])
        || collection.type !== "FeatureCollection" || !Array.isArray(collection.features)) return null;
    for (const feature of collection.features) {
      if (!feature || typeof feature !== "object" || Array.isArray(feature)
          || !feature.properties || feature.properties.plan_id !== value.plan_id
          || feature.properties.content_hash !== value.content_hash
          || feature.properties.role !== "draft_preview") return null;
    }
  }
  const acceptedByDay = new Map((state.workspace?.timeline || []).map(
    (day) => [day.day, day.stops.map((stop) => stop.id)],
  ));
  const reorderedDays = new Set();
  for (const operation of draft) {
    if (operation.type !== "set_stop_order") continue;
    const order = acceptedByDay.get(operation.parameters.day);
    const current = order?.indexOf(operation.target) ?? -1;
    if (!order || current < 0 || operation.parameters.sequence_index >= order.length) return null;
    order.splice(current, 1);
    order.splice(operation.parameters.sequence_index, 0, operation.target);
    reorderedDays.add(operation.parameters.day);
  }
  for (const day of reorderedDays) {
    const expectedOrder = acceptedByDay.get(day);
    const actualOrder = value.stops.features
      .filter((feature) => feature.properties.day === day)
      .sort((left, right) => left.properties.sequence_index - right.properties.sequence_index)
      .map((feature) => feature.properties.stop_id);
    if (!exactStringArray(actualOrder, expectedOrder)) return null;
  }
  return value;
}

function previewProposalBinding(value) {
  return {
    schema_version: value?.schema_version,
    state: value?.state,
    reason: value?.reason,
    eligibility: value?.eligibility,
    provenance: value?.provenance,
    execution_mode: value?.execution_mode,
    message: value?.message,
    expected_revision: value?.expected_revision,
    session_revision: value?.session_revision,
    draft_content_hash: value?.draft_content_hash,
    draft_operation_ids: value?.draft_operation_ids,
    parent_plan_id: value?.parent_plan_id,
    parent_plan_content_hash: value?.parent_plan_content_hash,
    child_plan_id: value?.child_plan_id,
    child_plan_content_hash: value?.child_plan_content_hash,
    certificate_id: value?.certificate_id,
    certificate_content_hash: value?.certificate_content_hash,
    diff_id: value?.diff_id,
    diff_content_hash: value?.diff_content_hash,
    diff_identity: value?.diff_identity,
    evaluation_status: value?.evaluation_status,
    certificate_identity: value?.certificate_identity,
    route_validation_identity: value?.route_validation_identity,
    evidence_status: value?.evidence ? {
      artifact_grounding_status: value.evidence.artifact_grounding_status,
      hard_feasibility_status: value.evidence.hard_feasibility_status,
      evaluation_status: value.evidence.evaluation_status,
      comparison_eligibility: value.evidence.comparison_eligibility,
      requested_method: value.evidence.requested_method,
      executed_method: value.evidence.executed_method,
    } : null,
    compiled_request: value?.compiled_request,
    repair_certificate: value?.repair?.certificate ?? null,
    plan_diff: value?.plan_diff ?? null,
    schedule_impact: value?.schedule_impact ?? null,
    schedule_impact_identity: value?.schedule_impact_identity ?? null,
    certificate_schedule_evidence: value?.certificate_schedule_evidence ?? null,
    role_impact: value?.role_impact ?? null,
    role_impact_identity: value?.role_impact_identity ?? null,
    certificate_role_evidence: value?.certificate_role_evidence ?? null,
    role_proposal_integrity: value?.role_proposal_integrity ?? null,
    decision_eligible: value?.decision_eligible ?? null,
    ranking_eligible: value?.ranking_eligible ?? null,
    acceptance_eligible: value?.acceptance_eligible ?? null,
    acceptance_blocking_code: value?.acceptance_blocking_code ?? null,
    geography_identity: value?.geography_plan ? {
      plan_id: value.geography_plan.plan_id,
      content_hash: value.geography_plan.content_hash,
      role: value.geography_plan.role,
      status: value.geography_plan.status,
      coverage: value.geography_plan.coverage,
    } : null,
  };
}

async function normalizeEvaluatedPreviewResponse(value, expected) {
  if (!exactKeys(value, ["proposal", "session"])) return null;
  const { proposal, session } = value;
  const orderOnly = expected.draft.length > 0
    && expected.draft.every((operation) => operation.type === "set_stop_order");
  const durationOnly = expected.draft.length > 0
    && expected.draft.every((operation) => operation.type === "set_stop_duration");
  const timeWindowOnly = expected.draft.length > 0
    && expected.draft.every((operation) => operation.type === "set_stop_time_window");
  const roleOnly = expected.draft.length > 0
    && expected.draft.every((operation) => operation.type === "set_stop_role");
  if (!exactKeys(session, [
    "schema_version", "session_id", "trip_id", "run_id", "revision", "accepted_plan_id",
    "selected_day", "selected_stop_id", "selected_segment_id", "selected_candidate_id",
    "selected_alternative_id", "draft", "proposal", "permission_decisions", "conversation_id",
  ])
      || session.schema_version !== "workspace-context-v1"
      || session.session_id !== expected.session_id
      || session.trip_id !== expected.trip_id || session.run_id !== expected.run_id
      || session.accepted_plan_id !== expected.accepted_plan_id
      || session.revision !== expected.revision + 1
      || canonicalJson(session.draft) !== canonicalJson(expected.draft)
      || !sameJson(session.permission_decisions, expected.permission_decisions)
      || session.conversation_id !== expected.conversation_id) return null;
  const proposalKeys = [
    "schema_version", "state", "reason", "eligibility", "provenance", "execution_mode",
    "expected_revision", "session_revision", "draft_content_hash", "draft_operation_ids",
    "message", "parent_plan_id", "parent_plan_content_hash", "child_plan_id",
    "child_plan_content_hash", "certificate_id", "certificate_content_hash", "diff_id",
    "diff_content_hash", "diff_identity", "evaluation_status", "certificate_identity",
    "route_validation", "route_validation_identity", "evidence", "geography_plan",
    "compiled_request", "repair",
    ...((durationOnly || timeWindowOnly) ? [
      "plan_diff", "schedule_impact", "schedule_impact_identity", "certificate_schedule_evidence",
      "decision_eligible", "ranking_eligible", "acceptance_eligible", "acceptance_blocking_code",
    ] : []),
    ...(roleOnly ? [
      "plan_diff", "role_impact", "role_impact_identity", "certificate_role_evidence",
      "role_proposal_integrity",
      "decision_eligible", "ranking_eligible", "acceptance_eligible", "acceptance_blocking_code",
    ] : []),
  ];
  const draftHash = await computedDraftContentHash(expected.draft);
  const operationIds = expected.draft.map((operation) => operation.operation_id);
  if (!exactKeys(proposal, proposalKeys)
      || proposal.schema_version !== "draft-preview-v1"
      || !["eligible", "ineligible"].includes(proposal.state)
      || proposal.eligibility !== proposal.state
      || !(proposal.reason === null || typeof proposal.reason === "string")
      || !["deterministic_repair_pipeline", "independent_evaluated_direct_edit"].includes(proposal.provenance)
      || !["deterministic_repair_pipeline", "typed_direct_edit_independent_evaluation"].includes(proposal.execution_mode)
      || proposal.expected_revision !== expected.revision
      || proposal.session_revision !== session.revision
      || draftHash === null || proposal.draft_content_hash !== draftHash
      || !exactStringArray(proposal.draft_operation_ids, operationIds)
      || typeof proposal.message !== "string"
      || proposal.parent_plan_id !== expected.accepted_plan_id
      || proposal.parent_plan_content_hash !== expected.parent_plan_content_hash
      || !/^[0-9a-f]{16}$/.test(proposal.parent_plan_content_hash)
      || !exactKeys(proposal.compiled_request, ["schema_version", "source_request_id", "operations"])
      || proposal.compiled_request.schema_version !== "typed-repair-request-v1"
      || typeof proposal.compiled_request.source_request_id !== "string"
      || !proposal.compiled_request.source_request_id
      || proposal.compiled_request.source_request_id.length > 128
      || canonicalJson(proposal.compiled_request.operations)
        !== canonicalJson(draftHashRecords(expected.draft))) return null;
  if (!exactKeys(session.proposal, proposalKeys)
      || !sameJson(previewProposalBinding(session.proposal), previewProposalBinding(proposal))) return null;
  const roleProposalIntegrity = roleOnly
    ? await normalizeRoleProposalIntegrity(proposal.role_proposal_integrity, proposal)
    : null;
  if (roleOnly && !roleProposalIntegrity) return null;
  const directEditOnly = orderOnly || durationOnly || timeWindowOnly || roleOnly;
  if ((directEditOnly && (
    proposal.provenance !== "independent_evaluated_direct_edit"
      || proposal.execution_mode !== "typed_direct_edit_independent_evaluation"
  )) || (!directEditOnly && (
    proposal.provenance !== "deterministic_repair_pipeline"
      || proposal.execution_mode !== "deterministic_repair_pipeline"
  ))) return null;
  const identityKeys = ["matrix_id", "context_snapshot_id", "source_bundle_id", "source_content_sha256"];
  if (!exactKeys(proposal.route_validation_identity, identityKeys)
      || !exactKeys(proposal.diff_identity, ["diff_id", "parent_plan_id", "child_plan_id", "content_hash"])
      || !exactKeys(proposal.certificate_identity, [
        "certificate_id", "plan_id", "plan_content_hash", "content_hash",
      ])
      || !exactKeys(proposal.evidence, [
        "artifact_grounding_status", "hard_feasibility_status", "evaluation_status",
        "comparison_eligibility", "route_validation", "requested_method", "executed_method",
      ])) return null;
  const hasChild = proposal.child_plan_id !== null;
  if (!hasChild) {
    if (proposal.state !== "ineligible" || proposal.reason === null
        || proposal.child_plan_content_hash !== null || proposal.certificate_id !== null
        || proposal.certificate_content_hash !== null || proposal.diff_id !== null
        || proposal.diff_content_hash !== null || proposal.evaluation_status !== null
        || proposal.geography_plan !== null || proposal.repair !== null
        || Object.keys(proposal.route_validation).length !== 0
        || Object.values(proposal.diff_identity).some((item) => item !== null)
        || Object.values(proposal.certificate_identity).some((item) => item !== null)
        || Object.values(proposal.route_validation_identity).some((item) => item !== null)) return null;
    return { proposal, session: { ...session, proposal } };
  }
  if (!/^plan_[0-9a-f]{16}$/.test(proposal.child_plan_id)
      || !/^[0-9a-f]{16}$/.test(proposal.child_plan_content_hash)
      || !/^diff_[0-9a-f]{16}$/.test(proposal.diff_id)
      || !/^[0-9a-f]{16}$/.test(proposal.diff_content_hash)
      || !/^cert_[0-9a-f]{16}$/.test(proposal.certificate_id)
      || !/^[0-9a-f]{16}$/.test(proposal.certificate_content_hash)
      || proposal.diff_identity.diff_id !== proposal.diff_id
      || proposal.diff_identity.parent_plan_id !== proposal.parent_plan_id
      || proposal.diff_identity.child_plan_id !== proposal.child_plan_id
      || proposal.diff_identity.content_hash !== proposal.diff_content_hash
      || proposal.certificate_identity.certificate_id !== proposal.certificate_id
      || proposal.certificate_identity.plan_id !== proposal.child_plan_id
      || proposal.certificate_identity.plan_content_hash !== proposal.child_plan_content_hash
      || proposal.certificate_identity.content_hash !== proposal.certificate_content_hash) return null;
  const routeValidation = normalizeRouteValidation(proposal.route_validation);
  const planDiff = durationOnly
    ? await normalizeDurationPlanDiff(proposal.plan_diff, proposal, expected.draft)
    : timeWindowOnly
      ? await normalizeTimeWindowPlanDiff(proposal.plan_diff, proposal, expected.draft)
      : roleOnly
        ? await normalizeRolePlanDiff(proposal.plan_diff, proposal, expected.draft)
        : null;
  const scheduleImpact = planDiff && (durationOnly || timeWindowOnly)
    ? await (durationOnly ? normalizeDurationScheduleImpact : normalizeTimeWindowScheduleImpact)(
      proposal.schedule_impact,
      proposal.schedule_impact_identity,
      proposal,
      planDiff,
      routeValidation,
    ) : null;
  const certificateScheduleEvidence = scheduleImpact && (durationOnly || timeWindowOnly)
    ? await (durationOnly
      ? normalizeCertificateScheduleEvidence
      : normalizeTimeWindowCertificateScheduleEvidence)(
      proposal.certificate_schedule_evidence, proposal, scheduleImpact,
    ) : null;
  const roleImpact = planDiff && roleOnly
    ? await normalizeRoleImpact(
      proposal.role_impact, proposal.role_impact_identity,
      proposal, planDiff, routeValidation,
    ) : null;
  const certificateRoleEvidence = roleImpact && roleOnly
    ? await normalizeRoleCertificateEvidence(
      proposal.certificate_role_evidence, proposal, roleImpact, planDiff,
    ) : null;
  if (!routeValidation
      || !identityKeys.every((key) => proposal.route_validation_identity[key] === routeValidation[key])
      || !sameJson(proposal.evidence.route_validation, routeValidation)
      || proposal.evidence.evaluation_status !== proposal.evaluation_status
      || proposal.evidence.comparison_eligibility !== proposal.eligibility
      || !["PASSED", "PASSED_WITH_WARNINGS", "FAILED"].includes(proposal.evaluation_status)
      || !normalizePreviewRepair(proposal.repair, proposal)
      || !normalizePreviewGeography(
        proposal.geography_plan, proposal, routeValidation, expected.draft,
      )
      || (durationOnly && (
        !planDiff
          || !scheduleImpact
          || !certificateScheduleEvidence
          || !durationRouteEvidenceUnchanged(proposal.geography_plan)
          || proposal.repair.changed.added.length !== 0
          || proposal.repair.changed.removed.length !== 0
          || proposal.repair.changed.affected_day_count !== scheduleImpact.affected_days.length
          || proposal.repair.changed.road_change_count !== 0
          || proposal.repair.changed.lodging_change_count !== 0
          || proposal.repair.tradeoffs.weighted_edit_cost !== planDiff.weighted_edit_cost
          || proposal.repair.tradeoffs.preservation_rate !== 1
          || proposal.repair.tradeoffs.utility_retained !== null
          || proposal.repair.tradeoffs.weather_risk_delta !== null
          || proposal.evaluation_status !== (
            scheduleImpact.child.status === "failed"
              ? "FAILED"
              : scheduleImpact.child.status === "incomplete_evidence"
                ? "PASSED_WITH_WARNINGS"
                : "PASSED"
          )
          || proposal.evidence.hard_feasibility_status !== (
            scheduleImpact.child.status === "failed" ? "FAILED" : "PASSED"
          )
          || proposal.state !== (scheduleImpact.child.status === "failed" ? "ineligible" : "eligible")
          || proposal.reason !== (
            scheduleImpact.child.status === "failed" ? "independent_evaluation_failed" : null
          )
      ))) return null;
  if (timeWindowOnly && (
    !planDiff
      || !scheduleImpact
      || !certificateScheduleEvidence
      || !durationRouteEvidenceUnchanged(proposal.geography_plan)
      || proposal.repair.changed.added.length !== 0
      || proposal.repair.changed.removed.length !== 0
      || proposal.repair.changed.affected_day_count !== scheduleImpact.affected_days.length
      || proposal.repair.changed.road_change_count !== 0
      || proposal.repair.changed.lodging_change_count !== 0
      || proposal.repair.tradeoffs.weighted_edit_cost !== planDiff.weighted_edit_cost
      || proposal.repair.tradeoffs.preservation_rate !== 1
      || proposal.repair.tradeoffs.utility_retained !== null
      || proposal.repair.tradeoffs.weather_risk_delta !== null
      || proposal.evaluation_status !== (
        scheduleImpact.child.status === "failed"
          ? "FAILED"
          : scheduleImpact.child.status === "incomplete_evidence"
            ? "PASSED_WITH_WARNINGS"
            : "PASSED"
      )
      || proposal.evidence.hard_feasibility_status !== (
        scheduleImpact.child.status === "failed" ? "FAILED" : "PASSED"
      )
      || proposal.state !== (scheduleImpact.child.status === "failed" ? "ineligible" : "eligible")
      || proposal.reason !== (
        scheduleImpact.child.status === "failed" ? "independent_evaluation_failed" : null
      )
  )) return null;
  if (roleOnly && (
    !planDiff
      || !roleImpact
      || !certificateRoleEvidence
      || !normalizeRoleGeography(proposal.geography_plan, planDiff, roleImpact)
      || proposal.repair.changed.added.length !== 0
      || proposal.repair.changed.removed.length !== 0
      || proposal.repair.changed.affected_day_count !== roleImpact.affected_days.length
      || proposal.repair.changed.road_change_count !== 0
      || proposal.repair.changed.lodging_change_count !== 0
      || proposal.repair.tradeoffs.weighted_edit_cost !== planDiff.weighted_edit_cost
      || proposal.repair.tradeoffs.preservation_rate !== 1
      || proposal.repair.tradeoffs.utility_retained !== null
      || proposal.repair.tradeoffs.weather_risk_delta !== null
      || proposal.evaluation_status !== (
        roleImpact.child_schedule.status === "failed"
          ? "FAILED"
          : certificateRoleEvidence.warning_codes.length
            ? "PASSED_WITH_WARNINGS"
            : "PASSED"
      )
      || proposal.evidence.hard_feasibility_status !== (
        roleImpact.child_schedule.status === "failed" ? "FAILED" : "PASSED"
      )
      || proposal.state !== (roleImpact.child_schedule.status === "failed" ? "ineligible" : "eligible")
      || proposal.reason !== (
        roleImpact.child_schedule.status === "failed" ? "independent_evaluation_failed" : null
      )
  )) return null;
  return { proposal, session: { ...session, proposal } };
}

function typedEditContract() {
  return normalizeTypedEditCapabilities(state.workspace?.typed_edit_capabilities);
}

function typedEditCapability(type) {
  return typedEditContract()?.operations?.[type] || null;
}

function safeExternalLink(value, label) {
  if (typeof value !== "string") return "Unavailable";
  try {
    const parsed = new URL(value);
    if (!["http:", "https:"].includes(parsed.protocol)
        || parsed.username || parsed.password || !parsed.hostname) return "Unavailable";
    return `<a href="${escapeHtml(parsed.href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)} · ${escapeHtml(parsed.hostname)}</a>`;
  } catch {
    return "Unavailable";
  }
}

function candidateStateLabel(burden) {
  const labels = {
    unavailable: "Unavailable / insufficient evidence",
    nearby: "Nearby",
    route_near: "Route-near",
    likely_feasible: "Likely feasible",
    evaluated_feasible: "Evaluated feasible",
    recommended: "Recommended",
  };
  if (burden?.candidate_state === "recommended"
      && !(burden.recommended === true && burden.ranking_eligible === true)) {
    return labels.unavailable;
  }
  return labels[burden?.candidate_state] || labels.unavailable;
}

function normalizeRoute(path) {
  if (path === "/app" || path === "/app/") return "/app/itinerary";
  const allowed = ["/app/itinerary", "/app/map", "/app/repairs", "/app/compare", "/app/evidence"];
  return allowed.includes(path) ? path : "/app/itinerary";
}

const API_ERROR_COPY = {
  request_validation_failed: "The browser and local service use different API versions. Stop the service, restart it, then refresh this page.",
  draft_candidate_target_mismatch: "That candidate cannot replace the selected stop. Choose a compatible candidate.",
  draft_duplicate_operation: "That change is already in the draft.",
  stale_session_revision: "The workspace changed in another action. Refresh the page and try again.",
  invalid_draft_target: "Select an editable stop from the accepted plan, not a repair-preview marker.",
  routing_timeout: "The local road-access check timed out. The exploratory point was not saved.",
  routing_unavailable: "The local routing service is unavailable. The exploratory point remains non-executable.",
  routing_http_error: "The local routing service rejected the road-access check. No draft was created.",
  routing_response_invalid: "The local routing service returned invalid evidence. No draft was created.",
  routing_transport_failure: "The browser could not complete the local road-access check. No draft was created.",
  snap_preview_not_found: "That road-check preview expired. Check road access again.",
  stale_snap_preview: "The draft changed after this road check. Check road access again.",
  snap_preview_expired: "That road-check preview expired. Check road access again.",
  snap_preview_already_consumed: "That road-check preview was already confirmed.",
  snap_warning_acknowledgement_required: "Review and acknowledge the long snap before confirming this waypoint.",
  waypoint_insertion_context_mismatch: "This waypoint's route insertion context changed. Restore its selected route and try again.",
  commitment_permission_required: "This protected commitment needs an explicit permission flow that is not enabled yet. No draft was added.",
  draft_no_effect: "Choose a value that differs from the accepted stop. No draft was added.",
  draft_conflicting_day_moves: "This stop already has a conflicting day change in the draft. Undo or revise that change first.",
  draft_conflicting_day_order: "The selected order conflicts with this stop's drafted day. Use the same day or revise the day change first.",
  draft_contains_non_executable_operation: "This draft contains route- or schedule-impact edits that are not enabled for full evaluation. Undo or revise those edits before running evaluated repair.",
  invalid_route_issue: "Choose a supported route issue and keep the note within 500 characters.",
  invalid_draft_impact_preview: "The local service returned an invalid draft-impact contract. Restart the product before editing.",
  invalid_evaluated_preview: "The evaluated repair response did not match this exact draft, parent, route evidence, or certificate. Nothing was shown as evaluated.",
  draft_evaluated_operation_combination_unsupported: "An evaluated direct edit must run alone. Keep only same-day reorder, exact-duration, time-window, or itinerary-role operations before previewing.",
  draft_order_index_invalid: "Choose a valid position within the stop's accepted day.",
  duration_mode_evaluation_not_supported: "Preferred, minimum, maximum, and range duration rules remain draft only. Choose Exact to run an evaluated preview.",
  invalid_stop_duration: "Enter an exact whole-minute duration from 15 through 480. No draft was added.",
  invalid_stop_time_window: "Provide at least one zero-padded local-time bound, with earliest service admission no later than latest departure. Cross-midnight windows are not supported.",
  stop_role_evaluation_not_supported: "Structural itinerary roles remain draft only and cannot run an evaluated preview in this slice.",
  invalid_stop_role: "Choose a supported trip-specific itinerary role. No draft was added.",
};

function apiErrorMessage(detail, status) {
  return API_ERROR_COPY[detail] || detail || `Request failed (${status})`;
}

async function api(path, { method = "GET", body, headers: extraHeaders = {} } = {}) {
  const headers = { Accept: "application/json", ...extraHeaders };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (state.token) headers["X-Session-Token"] = state.token;
  const response = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) throw new Error(apiErrorMessage(payload?.detail, response.status));
  return payload;
}

async function boot() {
  try {
    const health = await api("/api/health");
    if (health?.schema_version !== "product-health-v2" || health?.product_id !== "itinerary-repair-copilot") {
      throw new Error("invalid_product_health");
    }
    state.health = health;
    if (!health.core_ready) {
      renderCoreRecovery(health);
      return;
    }

    try {
      state.mapConfig = await api("/api/map/config");
      if (state.mapConfig?.schema_version !== "product-map-configuration-v2") {
        state.mapConfig = null;
      }
    } catch {
      state.mapConfig = null;
    }

    const payload = await restoreOrCreateSession();
    state.workspace = payload.workspace;
    state.session = payload.session;
    if (payload.mutation_token) state.token = payload.mutation_token;
    state.registry = payload.registry;
    await loadDraftImpactPreview({ render: false });
    bindEvents();
    applyRuntimeAvailability();
    $("#loading-screen").hidden = true;
    $("#app-shell").hidden = false;
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const defaultCompareId = state.route === "/app/compare"
      && !state.session.selected_alternative_id ? recommendedAlternativeId() : null;
    if (defaultCompareId) await selectContext({ selected_alternative_id: defaultCompareId });
    else renderAll();
    await copilot.restore();
  } catch (error) {
    renderStartupFailure(error);
  }
}

async function restoreOrCreateSession() {
  let saved = null;
  try {
    saved = JSON.parse(sessionStorage.getItem(SESSION_STORAGE_KEY) || "null");
  } catch {
    sessionStorage.removeItem(SESSION_STORAGE_KEY);
  }
  if (saved?.session_id && saved?.mutation_token) {
    state.token = saved.mutation_token;
    try {
      return await api(`/api/sessions/${encodeURIComponent(saved.session_id)}`);
    } catch {
      state.token = null;
      sessionStorage.removeItem(SESSION_STORAGE_KEY);
    }
  }
  const payload = await api("/api/sessions", { method: "POST", body: {} });
  state.token = payload.mutation_token;
  sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify({
    session_id: payload.session.session_id,
    mutation_token: payload.mutation_token,
  }));
  return payload;
}

const COMPONENT_COPY = {
  registry: "The run registry could not be validated.",
  default_workspace: "The pinned default workspace could not be loaded.",
  state_store: "Local product state is not writable or could not be verified.",
  map: "The selected local map service is unavailable. The textual route remains available.",
  openai: "The selected Copilot provider is not configured. No deterministic fallback was selected.",
};

const STATUS_CODE_COPY = {
  legacy_state_deferred: "Existing local decisions were preserved for validated review in W5; no automatic import occurs.",
  openai_not_configured: "The selected OpenAI provider is not configured. No deterministic fallback was used.",
  openai_transport_unavailable: "The selected OpenAI provider could not be initialized. No deterministic fallback was used.",
};

function componentMessage(component) {
  return STATUS_CODE_COPY[component.code] || COMPONENT_COPY[component.name] || component.code;
}

function componentEntries(health, statuses) {
  return Object.values(health?.components || {}).filter((component) => statuses.includes(component.status));
}

function renderCoreRecovery(health) {
  const failures = componentEntries(health, ["failed"]);
  const items = failures.length ? failures : [{ name: "runtime", code: "product_core_not_ready" }];
  $("#loading-screen").classList.add("recovery-screen");
  $("#loading-screen").innerHTML = `<section class="recovery-card" aria-labelledby="recovery-title">
    <span class="recovery-mark" aria-hidden="true">!</span>
    <p class="eyebrow">Local product recovery</p>
    <h1 id="recovery-title">The workspace is not ready</h1>
    <p>No session was created and all mutation controls remain unavailable.</p>
    <ul class="component-list">${items.map((component) => `<li><strong>${escapeHtml(component.name.replaceAll("_", " "))}</strong><span>${escapeHtml(componentMessage(component) || "A required runtime component failed validation.")}</span><code>${escapeHtml(component.code)}</code></li>`).join("")}</ul>
    <button class="primary" type="button" id="retry-runtime">Retry readiness check</button>
  </section>`;
  $("#retry-runtime").addEventListener("click", () => location.reload());
}

function renderStartupFailure(error) {
  const code = error?.message === "invalid_product_health" ? "invalid_product_health" : "health_unavailable";
  $("#loading-screen").classList.add("recovery-screen");
  $("#loading-screen").innerHTML = `<section class="recovery-card" aria-labelledby="recovery-title">
    <span class="recovery-mark" aria-hidden="true">!</span>
    <p class="eyebrow">Local product recovery</p>
    <h1 id="recovery-title">Copilot could not start</h1>
    <p>The product health endpoint did not return a valid, matching readiness contract.</p>
    <code>${code}</code>
    <button class="primary" type="button" id="retry-runtime">Retry readiness check</button>
  </section>`;
  $("#retry-runtime").addEventListener("click", () => location.reload());
}

function component(name) {
  return state.health?.components?.[name] || null;
}

function applyRuntimeAvailability() {
  const unavailable = componentEntries(state.health, ["degraded", "failed"]);
  if (component("map")?.status === "ready" && state.mapConfig?.status !== "ready") {
    unavailable.push({ name: "map", code: "map_configuration_unavailable" });
  }
  const banner = $("#runtime-banner");
  if (unavailable.length) {
    banner.hidden = false;
    $("#app-shell").classList.add("has-runtime-banner");
    banner.innerHTML = `<strong>Limited local runtime</strong><span>${unavailable.map((item) => escapeHtml(componentMessage(item))).join(" ")}</span>`;
  }

  const mapReady = component("map")?.status === "ready" && state.mapConfig?.status === "ready";
  $("#layers-button").disabled = !mapReady;
  $("#layers-button").title = mapReady ? "Show route layers" : "Local map layers are unavailable";

  copilot.update();
}

function bindEvents() {
  $$("[data-route]").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.route)));
  copilot.bind();
  $("#map-select-button").addEventListener("click", () => setMapInteractionMode("select"));
  $("#map-edit-button").addEventListener("click", () => setMapInteractionMode("edit"));
  $("#close-map-edit").addEventListener("click", () => $("#map-edit-dialog").close());
  $("#layers-button").addEventListener("click", () => {
    if (state.poiDiscovery.response?.candidates?.length) {
      state.poiDiscovery.showLayer = !state.poiDiscovery.showLayer;
      renderMap();
      toast(`Candidate layer ${state.poiDiscovery.showLayer ? "shown" : "hidden"}. It remains an inspectable discovery layer, not route-valid itinerary evidence.`);
      return;
    }
    toast("This pane shows the original route and the active option only. Find route-aware candidates from a selected route leg to add a candidate layer.");
  });
  $("#help-button").addEventListener("click", () => $("#info-dialog").showModal());
  $("#settings-button").addEventListener("click", () => $("#info-dialog").showModal());
  $("#draft-move").addEventListener("click", () => addDraft("move_day", { day: Number($("#destination-day").value) }));
  $("#draft-keep").addEventListener("click", () => addDraft("keep_stop", {}));
  $("#draft-lock").addEventListener("click", () => addDraft("lock_stop", {}));
  $("#draft-flexible").addEventListener("click", () => addDraft("mark_flexible", {}));
  $("#draft-feedback").addEventListener("click", () => addDraft(
    "route_feedback",
    { preference: "reduce_contextual_risk" },
    "selected_route",
  ));
  $("#registered-candidate").addEventListener("change", (event) => {
    void selectContext({ selected_candidate_id: event.currentTarget.value || null });
  });
  $("#draft-replace").addEventListener("click", () => addDraft("replace_nearby", { candidate_id: $("#registered-candidate").value }));
  $("#draft-add").addEventListener("click", () => addDraft("add_candidate", { day: Number($("#destination-day").value) }, $("#registered-candidate").value));
  $("#draft-undo").addEventListener("click", undoDraft);
  $("#draft-preview").addEventListener("click", previewDraft);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape"
        && state.mapInteraction.selectionKind === "exploratory"
        && !document.querySelector("dialog[open]")) {
      event.preventDefault();
      cancelExploratoryPin();
    }
  });
  window.addEventListener("popstate", () => {
    state.route = normalizeRoute(location.pathname);
    renderAll();
  });
  setupDockResize();
}

function navigate(route) {
  const normalized = normalizeRoute(route);
  if (location.pathname !== normalized) history.pushState({}, "", normalized);
  state.route = normalized;
  if (normalized === "/app/compare" && !state.session.selected_alternative_id) {
    const recommendedId = recommendedAlternativeId();
    if (recommendedId) {
      void selectContext({ selected_alternative_id: recommendedId });
      return;
    }
  }
  renderAll();
  $("#main-workspace").focus();
}

function recommendedAlternativeId() {
  return (state.workspace?.alternatives || []).find((option) => (
    option.role === "recommended"
    && option.ranking_eligible === true
    && (option.status === "eligible" || option.display_status === "Eligible")
  ))?.plan_id || null;
}

function renderAll() {
  renderTrip();
  renderWorkspace();
  renderContextPanel();
  renderMetrics();
  copilot.update();
  updateActiveNav();
  const editButton = $("#map-edit-button");
  const selectButton = $("#map-select-button");
  editButton.classList.toggle("is-active", state.mapInteraction.mode === "edit");
  editButton.setAttribute("aria-pressed", String(state.mapInteraction.mode === "edit"));
  editButton.querySelector("span").textContent = state.mapInteraction.mode === "edit" ? "Editing" : "Edit";
  selectButton.classList.toggle("is-active", state.mapInteraction.mode === "select");
  selectButton.setAttribute("aria-pressed", String(state.mapInteraction.mode === "select"));
  const flow = $(".flow");
  flow.hidden = state.route !== "/app/repairs"
    && state.session.draft.length === 0
    && !state.session.proposal;
}

function activeAlternativeId() {
  return state.session.proposal?.state === "eligible" && state.session.proposal?.child_plan_id
    ? state.session.proposal.child_plan_id
    : state.session.selected_alternative_id;
}

function activeProposalPlan() {
  const proposal = state.session.proposal;
  return proposal?.schema_version === "draft-preview-v1"
    && proposal?.state === "eligible"
    && proposal.geography_plan?.plan_id === proposal.child_plan_id
    ? proposal.geography_plan
    : null;
}

function activeGeography() {
  const geography = state.workspace.geography;
  const proposalPlan = activeProposalPlan();
  if (!proposalPlan || !Array.isArray(geography?.plans)) return geography;
  return {
    ...geography,
    plans: [...geography.plans.filter((plan) => plan.plan_id !== proposalPlan.plan_id), proposalPlan],
  };
}

function renderTrip() {
  const { trip, timeline } = state.workspace;
  const originalPlan = state.workspace.geography?.plans?.find((plan) => plan.role === "original");
  const routeNodes = [...(originalPlan?.route_path?.features || [])]
    .sort((left, right) => Number(left.properties?.occurrence_index) - Number(right.properties?.occurrence_index));
  const routeStart = readableRouteAnchor(routeNodes[0]?.properties?.node_id);
  const routeEnd = readableRouteAnchor(routeNodes.at(-1)?.properties?.node_id);
  $("#trip-title").textContent = routeNodes.length
    ? `California Coast · ${routeStart} to ${routeEnd}` : trip.title;
  $("#run-label").textContent = state.registry.label;
  $("#trip-metrics").innerHTML = `
    <div><b>${trip.day_count}</b><span>Days</span></div>
    <div><b>${trip.stop_count}</b><span>Stops</span></div>
    <div><b>${trip.affected_days.length}</b><span>Affected</span></div>`;
  $("#day-list").innerHTML = timeline.map((day) => {
    const selected = day.day === state.session.selected_day;
    const affected = day.states.includes("affected");
    const names = day.stops.map((stop) => stop.name).join(" · ") || "No recorded stop";
    return `<button type="button" class="day-card ${selected ? "is-selected" : ""} ${affected ? "is-affected" : ""}" data-day="${day.day}" data-stop="${escapeHtml(day.stops[0]?.id || "")}" aria-pressed="${selected}">
      <span class="day-number">Day ${day.day}</span>
      <span class="day-copy"><strong>${escapeHtml(names)}</strong><span>${day.stops.length} recorded stop${day.stops.length === 1 ? "" : "s"}</span></span>
      <span class="day-state" aria-label="${affected ? "Affected day" : "Unchanged day"}">${affected ? "⚠" : "✓"}</span>
    </button>`;
  }).join("");
  $$(".day-card").forEach((button) => button.addEventListener("click", () => selectMapStop({
    selected_day: Number(button.dataset.day),
    selected_stop_id: button.dataset.stop || null,
  })));
}

function renderWorkspace() {
  if (state.route === "/app/compare") { disposeGeographicMap(); return renderCompare(); }
  disposeCompareWorkspace();
  if (state.route === "/app/evidence") { disposeGeographicMap(); return renderEvidence(); }
  if (state.route === "/app/repairs") { disposeGeographicMap(); return renderRepairHistory(); }
  renderMap();
}

function routeSummaryForPlan(plan) {
  const byDay = new Map();
  (plan?.stops?.features || []).forEach((feature) => {
    const stop = feature.properties || {};
    const day = Number(stop.day);
    if (!Number.isInteger(day)) return;
    if (!byDay.has(day)) byDay.set(day, []);
    byDay.get(day).push({
      name: stop.name || stop.stop_id,
      order: Number(stop.stop_order ?? Number.MAX_SAFE_INTEGER),
    });
  });
  return [...byDay.entries()]
    .sort(([left], [right]) => left - right)
    .map(([day, stops]) => `Day ${day}: ${stops.sort((left, right) => left.order - right.order).map((stop) => stop.name).join(", ")}.`)
    .join(" ");
}

function validatedRouteFeatures(plan) {
  return plan?.validated_legs?.features || plan?.routes?.features || [];
}

function routeGapFeatures(plan) {
  return plan?.gaps?.features || plan?.route_gaps?.features || [];
}

function readableRouteAnchor(value) {
  return String(value || "Unknown anchor").split("_").map((word) => (
    word.toLowerCase() === "la"
      ? "LA"
      : `${word.slice(0, 1).toUpperCase()}${word.slice(1)}`
  )).join(" ");
}

function textualRouteEvidence(plan, daySummary) {
  const routes = [...validatedRouteFeatures(plan)]
    .sort((left, right) => Number(left.properties?.leg_index) - Number(right.properties?.leg_index));
  const gaps = routeGapFeatures(plan);
  const relocation = routes.find((feature) => Number(feature.properties?.day) === 4);
  const previousDayStop = [...(plan?.stops?.features || [])]
    .filter((feature) => Number(feature.properties?.day) === 3)
    .sort((left, right) => Number(left.properties?.stop_order) - Number(right.properties?.stop_order))
    .at(-1)?.properties?.name;
  const continuity = relocation
    ? `Route continuity: after Day 3 at ${previousDayStop || "the last recorded stop"}, Day 4 continues from ${readableRouteAnchor(relocation.properties?.origin_id)} to ${readableRouteAnchor(relocation.properties?.destination_id)} on a road-validated leg.`
    : "Route continuity: the next inter-day relocation is unavailable.";
  const validatedCount = routes.filter((feature) => (
    feature.properties?.road_validated === true && feature.properties?.fallback_used === false
  )).length;
  const alternative = (state.workspace.alternatives || []).find((row) => row.plan_id === plan?.plan_id);
  const proposalMatches = state.session.proposal?.child_plan_id === plan?.plan_id;
  const evidence = proposalMatches
    ? state.session.proposal?.evidence
    : state.workspace.alternative_evidence?.[plan?.plan_id]?.certificate;
  const eligibility = proposalMatches
    ? state.session.proposal?.eligibility
    : alternative?.display_status || evidence?.comparison_eligibility || state.workspace.repair?.status || "Unavailable";
  const permissions = proposalMatches
    ? state.session.proposal?.repair?.permissions
    : plan?.role === "registered_repair" ? state.workspace.repair?.permissions : null;
  const bookingImpact = permissions
    ? `${permissions.booked_change_count ?? "Unavailable"} booked changes and ${permissions.locked_change_count ?? "Unavailable"} locked changes recorded.`
    : "No plan-specific booking-change metric is available.";
  const requiredCount = Number(plan?.coverage?.required_leg_count) || routes.length + gaps.length;
  const gapSummary = gaps.length
    ? `${gaps.length} required connection${gaps.length === 1 ? " is" : "s are"} an explicit unvalidated route gap.`
    : "No required route gaps are hidden.";
  return `${daySummary} ${continuity} Eligibility: ${eligibility}; ${validatedCount}/${requiredCount} required route legs are road validated with no fallback. ${gapSummary} Booking impact: ${bookingImpact}`;
}

function textRoutePanelHtml(plan, originalPlan) {
  const selectedDay = Number(state.session.selected_day);
  const stops = (plan?.stops?.features || [])
    .filter((feature) => Number(feature.properties?.day) === selectedDay)
    .sort((left, right) => Number(left.properties?.stop_order) - Number(right.properties?.stop_order));
  const legs = validatedRouteFeatures(plan)
    .filter((feature) => Number(feature.properties?.day) === selectedDay)
    .sort((left, right) => Number(left.properties?.leg_index) - Number(right.properties?.leg_index));
  const durations = legs.map((feature) => feature.properties?.duration_minutes);
  const duration = durations.length && durations.every(
    (value) => typeof value === "number" && Number.isFinite(value) && value > 0,
  ) ? `${formatMetric(durations.reduce((total, value) => total + value, 0))} minutes` : "Unavailable";
  const alternative = (state.workspace.alternatives || []).find((row) => row.plan_id === plan?.plan_id);
  const proposalMatches = state.session.proposal?.child_plan_id === plan?.plan_id;
  const eligibility = proposalMatches
    ? state.session.proposal?.eligibility
    : alternative?.display_status || state.workspace.repair?.status || "Unavailable";
  const permissions = proposalMatches
    ? state.session.proposal?.repair?.permissions
    : plan?.role === "registered_repair" ? state.workspace.repair?.permissions : null;
  const bookingImpact = permissions
    ? `${permissions.booked_change_count ?? "Unavailable"} booked; ${permissions.locked_change_count ?? "Unavailable"} locked changes`
    : "Unavailable";
  const stopOrder = stops.length
    ? stops.map((feature, index) => `${index + 1}. ${feature.properties?.name || feature.properties?.stop_id}`).join(" → ")
    : "Unavailable";
  const travelLegs = legs.length ? `<ol>${legs.map((feature) => {
    const properties = feature.properties || {};
    const names = routeNamePair(properties, plan);
    const minutes = typeof properties.duration_minutes === "number" && properties.duration_minutes > 0
      ? `${formatMetric(properties.duration_minutes)} minutes` : "duration Unavailable";
    return `<li>${escapeHtml(names.origin)} → ${escapeHtml(names.destination)} · ${escapeHtml(minutes)} · road validated</li>`;
  }).join("")}</ol>` : "<p>Travel legs are Unavailable for this day.</p>";
  const originalDay = routeSummaryForPlan(originalPlan).split(/(?=Day \d+:)/)
    .find((row) => row.startsWith(`Day ${selectedDay}:`)) || "Unavailable";
  const proposedDay = routeSummaryForPlan(plan).split(/(?=Day \d+:)/)
    .find((row) => row.startsWith(`Day ${selectedDay}:`)) || "Unavailable";
  return `<section class="text-route-summary" aria-label="Selected day route summary">
    <h3>Day ${escapeHtml(selectedDay)} route</h3>
    <dl><dt>Stop order</dt><dd>${escapeHtml(stopOrder)}</dd><dt>Travel duration</dt><dd>${escapeHtml(duration)}</dd><dt>Disruption</dt><dd>${escapeHtml(state.workspace.issue.summary)}</dd><dt>Original route</dt><dd>${escapeHtml(originalDay)}</dd><dt>Proposed route</dt><dd>${escapeHtml(proposedDay)}</dd><dt>Booking impact</dt><dd>${escapeHtml(bookingImpact)}</dd><dt>Eligibility</dt><dd>${escapeHtml(eligibility)}</dd></dl>
    <h3>Travel legs</h3>${travelLegs}
  </section>`;
}

function activeDisplayedPlan(geography) {
  const activeId = activeAlternativeId();
  return geography?.plans?.find((plan) => plan.plan_id === activeId)
    || geography?.plans?.find((plan) => plan.role === "registered_repair")
    || geography?.plans?.find((plan) => plan.role === "original")
    || null;
}

function findRouteLeg(plan, routeLegId) {
  return validatedRouteFeatures(plan).find(
    (feature) => feature.properties?.route_leg_id === routeLegId,
  ) || null;
}

const MAP_DRAFT_OPERATION_TYPES = new Set([
  "add_custom_waypoint",
  "add_route_waypoint",
  "relocate_custom_waypoint",
]);

function mapEditCapability(type) {
  const capabilities = state.workspace?.map_edit_capabilities;
  if (capabilities?.schema_version !== "map-edit-capabilities-v1") return null;
  const capability = capabilities.operations?.[type];
  return capability?.enabled === true
    && capability.feedback_tier === "route_checked"
    && capability.preview_executable === false
    && capability.evaluated_repair === false ? capability : null;
}

function confirmedMapDraftState() {
  return reduceConfirmedMapDraftOperations(
    state.session?.draft || [],
    state.workspace?.timeline?.length || 0,
  );
}

function selectedDraftWaypoint() {
  return confirmedMapDraftState().waypoints.find(
    (waypoint) => waypoint.waypoint_id === state.mapInteraction.selectedWaypointId,
  ) || null;
}

function hasRouteCheckedOnlyDraft() {
  return state.session?.draft?.some((operation) => MAP_DRAFT_OPERATION_TYPES.has(operation.type)) === true;
}

function durationInputRows(mode) {
  const preferred = ["exact", "preferred"].includes(mode);
  const minimum = ["minimum", "range"].includes(mode);
  const maximum = ["maximum", "range"].includes(mode);
  const row = (id, label, enabled, required) => `<label class="duration-field" ${enabled ? "" : "hidden"} for="${id}">${label}
    <input id="${id}" type="number" min="15" max="480" step="15" ${enabled ? "" : "disabled"} ${required ? "required" : ""}>
  </label>`;
  return `${row("waypoint-duration-preferred", mode === "exact" ? "Visit duration (minutes)" : "Preferred duration (minutes)", preferred, preferred)}
    ${row("waypoint-duration-minimum", "Minimum duration (minutes)", minimum, mode !== "preferred" && minimum)}
    ${row("waypoint-duration-maximum", "Maximum duration (minutes)", maximum, mode !== "preferred" && maximum)}`;
}

function roadCheckedPreviewCanConfirm(preview) {
  return preview?.schema_version === "map-snap-preview-v1"
    && preview.persisted === false
    && preview.validation_state === "route_checked"
    && preview.draft_append_allowed === true
    && preview.selected_access_point?.road_validated === true
    && preview.selected_access_point?.source === "runtime_osrm"
    && Array.isArray(preview.affected_route_legs)
    && preview.affected_route_legs.length > 0
    && preview.affected_route_legs.every((leg) => leg?.road_validated === true
      && leg.validation_status === "road_validated"
      && leg.fallback_used === false
      && leg.provider === "runtime_osrm"
      && /^[0-9a-f]{64}$/.test(leg.query_hash || ""));
}

function selectedPoiCandidate() {
  const candidates = state.poiDiscovery.response?.candidates || [];
  return candidates.find(
    (candidate) => candidate.candidate_id === state.poiDiscovery.selectedCandidateId,
  ) || null;
}

function candidateCoordinates(candidate) {
  const display = candidate.place?.display_coordinate;
  const access = candidate.selected_access_point?.coordinate;
  const coordinate = (value) => value
    ? `${Number(value.latitude).toFixed(5)}, ${Number(value.longitude).toFixed(5)}`
    : "Unavailable";
  return { display: coordinate(display), access: coordinate(access) };
}

function candidateCardHtml(candidate) {
  const place = candidate.place || {};
  const burden = candidate.burden || {};
  const precheck = candidate.precheck || {};
  const coordinates = candidateCoordinates(candidate);
  const replacement = candidate.registered_replacement;
  const canReplace = replacement?.draft_type === "replace_nearby"
    && replacement.candidate_id === candidate.candidate_id
    && replacement.candidate_id === place.place_id
    && replacement.target_stop_id === state.poiDiscovery.response?.context?.replacement_target_id
    && operationEnabled("replace_nearby");
  const minutes = (value) => unavailable(value, (item) => `${Number(item).toFixed(1)} min`);
  const signedMinutes = (value) => unavailable(value, (item) => {
    const amount = Number(item);
    if (amount < 0) return `${Math.abs(amount).toFixed(1)} min less travel`;
    if (amount > 0) return `${amount.toFixed(1)} min more travel`;
    return "No travel-time change";
  });
  const signedMiles = (value) => unavailable(value, (item) => {
    const amount = Number(item) / 1609.344;
    if (amount < 0) return `${Math.abs(amount).toFixed(1)} mi less`;
    if (amount > 0) return `${amount.toFixed(1)} mi more`;
    return "No distance change";
  });
  const infoLink = place.informational_urls?.length
    ? safeExternalLink(place.informational_urls[0], "Informational source") : "Unavailable";
  const sourceLink = candidate.sources?.length
    ? safeExternalLink(candidate.sources[0].source_url, candidate.sources[0].source_type || "Catalog source")
    : "Unavailable";
  return `<article class="poi-candidate-card" aria-labelledby="poi-candidate-title">
    <span class="status-pill ${burden.recommended === true ? "success" : ""}">${escapeHtml(candidateStateLabel(burden))}</span>
    <h3 id="poi-candidate-title">${escapeHtml(place.name)}</h3>
    <p>${escapeHtml(place.description || "Description unavailable")}</p>
    <dl>
      <dt>Category</dt><dd>${escapeHtml(place.place_categories?.join(", ") || "Unavailable")}</dd>
      <dt>Display point</dt><dd>${escapeHtml(coordinates.display)}</dd>
      <dt>Route access</dt><dd>${escapeHtml(coordinates.access)} · ${escapeHtml(unavailable(candidate.selected_access_point?.access_type))}</dd>
      <dt>Access evidence</dt><dd>${candidate.selected_access_point?.road_validated === true ? "Road-validated access point" : "Not road validated"} · ${escapeHtml(unavailable(candidate.selected_access_point?.access_confidence))}</dd>
      <dt>Marginal travel</dt><dd>${escapeHtml(signedMinutes(burden.marginal_travel_minutes))} · ${escapeHtml(signedMiles(burden.marginal_travel_distance_m))}</dd>
      <dt>Total insertion</dt><dd>${escapeHtml(minutes(burden.total_insertion_minutes))}</dd>
      <dt>Predicted arrival</dt><dd>${escapeHtml(unavailable(precheck.predicted_arrival))}</dd>
      <dt>Open at arrival</dt><dd>${escapeHtml(precheck.open_at_arrival === null ? "Unavailable" : precheck.open_at_arrival ? "Open" : "Closed")}</dd>
      <dt>Visit duration</dt><dd>${escapeHtml(minutes(place.recommended_visit_minutes ?? burden.visit_minutes))}</dd>
      <dt>Parking</dt><dd>${escapeHtml(minutes(burden.parking_minutes))}</dd>
      <dt>Walking</dt><dd>${escapeHtml(minutes(burden.walking_minutes))}</dd>
      <dt>Waiting</dt><dd>${escapeHtml(minutes(burden.waiting_minutes))}</dd>
      <dt>Weather suitability</dt><dd>${escapeHtml(unavailable(place.weather_suitability))}</dd>
      <dt>Schedule precheck</dt><dd>${escapeHtml(unavailable(precheck.status))}</dd>
      <dt>Evaluator state</dt><dd>${burden.evaluated_feasible === true ? "Evaluated feasible" : "Not evaluated feasible"}; ranking ${burden.ranking_eligible === true ? "eligible" : "unavailable"}</dd>
      <dt>Freshness</dt><dd>${escapeHtml(unavailable(place.source_freshness))}</dd>
      <dt>Source IDs</dt><dd>${escapeHtml(place.source_refs?.join(", ") || "Unavailable")}</dd>
    </dl>
    <p class="candidate-links">Official: ${safeExternalLink(place.official_url, "Official site")}<br>Information: ${infoLink}<br>Catalog source: ${sourceLink}</p>
    <p><strong>Blocking evidence:</strong> ${escapeHtml(burden.blocking_codes?.join(", ") || "None recorded")}</p>
    <button type="button" id="use-candidate-replacement" class="primary" ${canReplace ? "" : "disabled title=\"Only an exact server-registered replacement for the selected accepted-plan stop can become a draft\""}>Use as replacement</button>
    <p class="truth-note">Candidate coordinates are display/access evidence only. This card does not make the candidate part of the itinerary or certify feasibility.</p>
  </article>`;
}

function candidateExplorerHtml() {
  const discovery = state.poiDiscovery;
  if (discovery.status === "idle") return "";
  const response = discovery.response;
  const candidates = response?.candidates || [];
  const selected = selectedPoiCandidate();
  const context = response?.context;
  const list = candidates.map((candidate) => {
    const selectedRow = candidate.candidate_id === discovery.selectedCandidateId;
    return `<li><button type="button" data-poi-candidate="${escapeHtml(candidate.candidate_id)}" aria-pressed="${selectedRow}">
      <strong>${escapeHtml(candidate.place?.name || "Unavailable")}</strong>
      <span>${escapeHtml(candidateStateLabel(candidate.burden))}</span>
      <small>${escapeHtml(unavailable(candidate.burden?.marginal_travel_minutes, (value) => Number(value) < 0 ? `${Math.abs(Number(value)).toFixed(1)} min less travel` : `${Number(value).toFixed(1)} min more travel`))}</small>
    </button></li>`;
  }).join("");
  return `<aside class="poi-candidate-drawer" aria-labelledby="poi-candidate-heading">
    <header><div><span class="eyebrow">Route-aware discovery</span><h2 id="poi-candidate-heading">Candidates for this exact route slot</h2></div>
      <button type="button" id="close-poi-candidates" aria-label="Close candidate discovery">×</button></header>
    <p class="candidate-context">${context
      ? `Day ${escapeHtml(context.day)} · ${escapeHtml(context.predecessor_id)} → ${escapeHtml(context.successor_id)} · ${escapeHtml(context.kind)}`
      : "Waiting for exact route context."}</p>
    <form id="poi-candidate-filter" class="poi-candidate-filter">
      <label for="poi-role-filter">Role/category <select id="poi-role-filter"><option value="">Any supported role</option>${["attraction", "activity", "meal", "rest_stop", "scenic_stop"].map((role) => `<option value="${role}" ${discovery.role === role ? "selected" : ""}>${escapeHtml(role.replaceAll("_", " "))}</option>`).join("")}</select></label>
      <label for="poi-detour-filter">Maximum detour <select id="poi-detour-filter">${[15, 30, 60, 120].map((value) => `<option value="${value}" ${discovery.maxDetourMinutes === value ? "selected" : ""}>${value} min</option>`).join("")}</select></label>
      <label for="poi-limit-filter">Results <select id="poi-limit-filter">${[3, 5, 10].map((value) => `<option value="${value}" ${discovery.limit === value ? "selected" : ""}>Top ${value}</option>`).join("")}</select></label>
      <button type="submit">Refresh candidates</button>
    </form>
    <label class="candidate-layer-toggle"><input id="poi-layer-toggle" type="checkbox" ${discovery.showLayer ? "checked" : ""} ${candidates.length ? "" : "disabled"}> Show candidate layer (${candidates.length}/10 maximum)</label>
    <div class="candidate-results" role="status">${discovery.status === "loading" ? "Loading route-aware candidate evidence…" : discovery.status === "failed" ? `Candidate discovery unavailable: ${escapeHtml(discovery.error || "Unavailable")}` : candidates.length ? `${candidates.length} candidate${candidates.length === 1 ? "" : "s"}; no automatic recommendation is inferred.` : "No candidates are available for this exact route context."}</div>
    ${list ? `<ol class="poi-candidate-list" aria-label="Route-aware candidates">${list}</ol>` : ""}
    ${selected ? candidateCardHtml(selected) : candidates.length ? "<p>Select a candidate marker or list item to inspect exact evidence.</p>" : ""}
  </aside>`;
}

function humanLabel(value) {
  return String(value || "Unavailable").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function stopNameForId(stopId, plan = null) {
  if (!stopId) return "Stop unavailable";
  const planFeature = plan?.stops?.features?.find(
    (feature) => feature.properties?.stop_id === stopId,
  );
  if (planFeature?.properties?.name) return planFeature.properties.name;
  for (const day of state.workspace?.timeline || []) {
    const stop = day.stops.find((candidate) => candidate.id === stopId);
    if (stop?.name) return stop.name;
  }
  const geographyFeature = state.workspace?.geography?.plans
    ?.flatMap((row) => row.stops?.features || [])
    .find((feature) => feature.properties?.stop_id === stopId);
  return geographyFeature?.properties?.name || "Stop unavailable";
}

function acceptedPlan(geography) {
  return geography?.plans?.find((plan) => plan.plan_id === state.session?.accepted_plan_id) || null;
}

function selectedAcceptedStopFeature(geography) {
  if (!state.session?.selected_stop_id) return null;
  const selectedPlanId = state.mapInteraction.selectedStopPlanId;
  if (selectedPlanId && selectedPlanId !== state.session.accepted_plan_id) return null;
  return acceptedPlan(geography)?.stops?.features?.find(
    (feature) => feature.properties?.stop_id === state.session.selected_stop_id,
  ) || null;
}

function selectedAcceptedRouteFeature(geography) {
  const selected = state.mapInteraction.selectedRouteLeg;
  if (!selected?.route_leg_id || selected.plan_id !== state.session?.accepted_plan_id) return null;
  return findRouteLeg(acceptedPlan(geography), selected.route_leg_id);
}

function formatDurationConstraint(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return "Unavailable";
  const mode = value.mode;
  if (!TYPED_EDIT_VOCABULARIES.duration_modes.includes(mode)) return "Unavailable";
  const parts = [];
  if (Number.isInteger(value.preferred_minutes)) parts.push(`${value.preferred_minutes} min preferred`);
  if (Number.isInteger(value.minimum_minutes)) parts.push(`${value.minimum_minutes} min minimum`);
  if (Number.isInteger(value.maximum_minutes)) parts.push(`${value.maximum_minutes} min maximum`);
  return parts.length ? `${humanLabel(mode)} · ${parts.join(" · ")}` : "Unavailable";
}

function stopDraftOperations(stopId) {
  return (state.session?.draft || []).filter((operation) => operation.target === stopId);
}

function routeDraftOperations(routeLegId) {
  return (state.session?.draft || []).filter(
    (operation) => operation.target === routeLegId
      || operation.parameters?.route_leg_id === routeLegId,
  );
}

function impactRow(operationId) {
  return state.typedImpact.response?.operations?.find(
    (row) => row.operation_id === operationId,
  ) || null;
}

function operationDisplayName(type) {
  const names = {
    set_stop_role: "Role change",
    set_stop_day: "Day change",
    set_stop_order: "Order change",
    set_stop_duration: "Duration rule",
    set_stop_time_window: "Time window",
    set_stop_commitment: "Commitment",
    set_attribute_constraint: "Attribute constraint",
    change_route_preference: "Route preference",
    report_route_issue: "Route issue",
    move_day: "Day change",
    keep_stop: "Keep stop",
    lock_stop: "Lock stop",
    mark_flexible: "Flexible stop",
    route_feedback: "Route feedback",
    replace_nearby: "Replacement",
    add_candidate: "Added candidate",
    add_custom_waypoint: "Custom stop",
    add_route_waypoint: "Route waypoint",
    relocate_custom_waypoint: "Waypoint relocation",
  };
  return names[type] || "Typed draft change";
}

function draftStateHtml(operations) {
  if (!operations.length) return '<span class="status-pill">Matches accepted parent</span><p>No active draft changes target this selection.</p>';
  const rows = operations.map((operation) => {
    const impact = impactRow(operation.operation_id);
    const tier = impact?.feedback_tier === "evaluated" ? "Evaluated-preview capable" : "Draft only; not evaluated";
    const attributes = impact?.changed_attributes?.length
      ? impact.changed_attributes.map(humanLabel).join(", ") : "Impact details unavailable";
    return `<li><strong>${escapeHtml(operationDisplayName(operation.type))}</strong><span>${escapeHtml(tier)} · ${escapeHtml(attributes)}</span></li>`;
  }).join("");
  return `<span class="status-pill warning">Draft differs from parent</span><ul class="selection-draft-list">${rows}</ul>`;
}

function routeNamePair(properties, plan) {
  return {
    origin: stopNameForId(properties?.origin_id, plan),
    destination: stopNameForId(properties?.destination_id, plan),
  };
}

function routeDurationLabel(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? `${Math.round(value / 60)} min`
    : "Unavailable";
}

function routeDistanceLabel(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? `${(value / 1609.344).toFixed(1)} mi`
    : "Unavailable";
}

function draftTargetLabel(operation) {
  const geography = activeGeography();
  const parentPlan = acceptedPlan(geography);
  if (["report_route_issue", "route_feedback"].includes(operation.type)) {
    const route = findRouteLeg(parentPlan, operation.target);
    if (!route) return "Selected itinerary route";
    const names = routeNamePair(route.properties, parentPlan);
    return `${names.origin} to ${names.destination}`;
  }
  const waypoint = confirmedMapDraftState().waypoints.find(
    (candidate) => candidate.waypoint_id === operation.target,
  );
  if (waypoint) return waypoint.name;
  const candidate = (state.workspace?.draft_capabilities?.candidate_choices || []).find(
    (choice) => choice.candidate_id === operation.target,
  );
  if (candidate) return candidate.label || candidate.name || "Registered candidate";
  return stopNameForId(operation.target, parentPlan);
}

function sourceFreshnessHtml(properties) {
  const source = properties?.source_type || properties?.source || properties?.provider || null;
  const freshness = properties?.source_freshness || properties?.freshness || properties?.checked_at || null;
  return `${escapeHtml(unavailable(source, humanLabel))} · ${escapeHtml(unavailable(freshness))}`;
}

function mapInspectorHtml(geography, displayedPlan) {
  const interaction = state.mapInteraction;
  if (interaction.selectionKind === "custom_waypoint") {
    const waypoint = selectedDraftWaypoint();
    if (!waypoint) return "";
    const relocateEnabled = interaction.mode === "edit"
      && Boolean(mapEditCapability("relocate_custom_waypoint"));
    return `<section class="map-context-card draft-waypoint" aria-labelledby="map-context-title">
      <span class="status-pill route-checked">Route-checked draft</span>
      <h3 id="map-context-title">${escapeHtml(waypoint.name)}</h3>
      <dl><dt>Role</dt><dd>${escapeHtml(waypoint.role)}</dd>
        <dt>Day</dt><dd>${escapeHtml(waypoint.day ?? "Unavailable")}</dd>
        <dt>Insertion</dt><dd>${escapeHtml(stopNameForId(waypoint.insertion?.predecessor_id, acceptedPlan(geography)))} → ${escapeHtml(stopNameForId(waypoint.insertion?.successor_id, acceptedPlan(geography)))}</dd>
        <dt>State</dt><dd>Confirmed session draft; not evaluated</dd></dl>
      <p>This waypoint and its affected road legs are server-owned draft evidence. The accepted parent is unchanged.</p>
      <button type="button" id="relocate-custom-waypoint" ${relocateEnabled ? "" : "disabled title=\"Switch to Edit mode to relocate this custom waypoint\""}>Relocate this waypoint</button>
    </section>`;
  }
  if (interaction.selectionKind === "stop") {
    const stop = findSelectedStop();
    if (!stop) return "";
    const acceptedPlanId = state.session.accepted_plan_id;
    const selectedPlanId = interaction.selectedStopPlanId || acceptedPlanId;
    const stopPlan = geography?.plans?.find(
      (plan) => plan.plan_id === selectedPlanId,
    );
    const feature = stopPlan?.stops?.features?.find(
      (candidate) => candidate.properties?.stop_id === stop.id,
    );
    const properties = feature?.properties || {};
    const acceptedTarget = selectedPlanId === acceptedPlanId
      && (state.workspace.draft_capabilities?.parent_targets || []).includes(stop.id);
    const duration = properties.duration || properties.duration_constraint || null;
    const access = properties.selected_access_point || properties.route_access_point || null;
    const attributeConstraints = Array.isArray(properties.attribute_constraints)
      ? properties.attribute_constraints.map((row) => humanLabel(row.attribute || row)).join(", ")
      : "Unavailable";
    const placeLink = properties.official_url
      ? safeExternalLink(properties.official_url, "Official source")
      : safeExternalLink(properties.informational_url, "Informational source");
    const activeDraft = stopDraftOperations(stop.id);
    return `<section class="map-context-card" aria-labelledby="map-context-title">
      <span class="status-pill">Selected stop</span>
      <h3 id="map-context-title">${escapeHtml(stop.name)}</h3>
      <p>${escapeHtml(unavailable(properties.description, (value) => value))}</p>
      <dl><dt>Itinerary role</dt><dd>${escapeHtml(unavailable(properties.itinerary_role, humanLabel))}</dd>
        <dt>Role source</dt><dd>${escapeHtml(unavailable(properties.itinerary_role_source, humanLabel))}</dd>
        <dt>Place categories</dt><dd>${escapeHtml(unavailable(
    Array.isArray(properties.place_categories) && properties.place_categories.length
      ? properties.place_categories.map(humanLabel).join(", ")
      : null,
  ))}</dd>
        <dt>Day and order</dt><dd>Day ${escapeHtml(properties.day ?? stop.day ?? "Unavailable")} · ${escapeHtml(unavailable(properties.sequence_index, (value) => `position ${Number(value) + 1}`))}</dd>
        <dt>Arrival</dt><dd>${escapeHtml(unavailable(properties.arrival_time || properties.expected_arrival))}</dd>
        <dt>Departure</dt><dd>${escapeHtml(unavailable(properties.departure_time || properties.expected_departure))}</dd>
        <dt>Duration rule</dt><dd>${escapeHtml(formatDurationConstraint(duration))}</dd>
        <dt>Commitment</dt><dd>${escapeHtml(unavailable(properties.commitment_strength || properties.ownership_strength, humanLabel))}</dd>
        <dt>Attribute constraints</dt><dd>${escapeHtml(attributeConstraints)}</dd>
        <dt>Weather / context</dt><dd>${escapeHtml(unavailable(properties.context_summary || properties.weather_context))}</dd>
        <dt>Visit guidance</dt><dd>${escapeHtml(unavailable(properties.visit_guidance))}</dd>
        <dt>Route access</dt><dd>${access?.road_validated === true ? `${escapeHtml(humanLabel(access.access_type))} · road validated` : "Unavailable"}</dd>
        <dt>Source / freshness</dt><dd>${sourceFreshnessHtml(properties)}</dd>
        <dt>Place link</dt><dd>${placeLink}</dd></dl>
      <div class="selection-draft-state">${draftStateHtml(activeDraft)}</div>
      <p>This is a trip stop, not a movable catalog place. Editing it creates a typed draft and never moves or rewrites the catalog place.</p>
      <div class="map-context-actions"><button type="button" id="inspect-edit-stop" ${acceptedTarget ? "" : "disabled title=\"Only an accepted-parent stop can be edited\""}>Open typed stop edits</button>
        <button type="button" id="prepare-replacement-candidates" ${acceptedTarget ? "" : "disabled title=\"Replacement discovery requires an accepted-parent stop\""}>Find a replacement</button></div>
    </section>`;
  }
  if (interaction.selectionKind === "route_leg") {
    const selected = interaction.selectedRouteLeg;
    const routePlan = geography?.plans?.find((plan) => plan.plan_id === selected?.plan_id)
      || displayedPlan;
    const feature = findRouteLeg(routePlan, selected?.route_leg_id);
    const properties = feature?.properties || selected || {};
    const minutes = routeDurationLabel(properties.duration_s);
    const miles = routeDistanceLabel(properties.distance_m);
    const names = routeNamePair(properties, routePlan);
    const activeDraft = routeDraftOperations(properties.route_leg_id);
    const issueCapability = typedEditCapability("report_route_issue");
    const preferenceCapability = typedEditCapability("change_route_preference");
    const acceptedRouteTarget = selectedAcceptedRouteFeature(geography) !== null;
    return `<section class="map-context-card" aria-labelledby="map-context-title">
      <span class="status-pill success">Road-validated route leg</span>
      <h3 id="map-context-title">${escapeHtml(names.origin)} → ${escapeHtml(names.destination)}</h3>
      <dl><dt>Day</dt><dd>${escapeHtml(properties.day ?? "Unavailable")}</dd>
        <dt>Mode</dt><dd>${escapeHtml(properties.travel_mode || "Unavailable")}</dd>
        <dt>Distance</dt><dd>${escapeHtml(miles)}</dd><dt>Duration</dt><dd>${escapeHtml(minutes)}</dd>
        <dt>Route source</dt><dd>${sourceFreshnessHtml(properties)}</dd>
        <dt>Closure / weather</dt><dd>${escapeHtml(unavailable(properties.context_summary || properties.closure_weather_context))}</dd>
        <dt>Validation</dt><dd>${properties.road_validated === true && properties.fallback_used === false ? "Road validated · no fallback" : "Unvalidated"}</dd></dl>
      <div class="selection-draft-state">${draftStateHtml(activeDraft)}</div>
      <p>Clicking a road line selects router evidence; it does not make the polyline freely draggable.</p>
      ${state.poiDiscovery.replacementTargetId ? `<p><strong>Replacement target:</strong> ${escapeHtml(stopNameForId(state.poiDiscovery.replacementTargetId, routePlan))}. The server will require this leg to touch that exact stop.</p>` : ""}
      <div class="map-context-actions"><button type="button" id="find-route-candidates" ${interaction.mode === "edit" && acceptedRouteTarget ? "" : "disabled title=\"Select an accepted-plan route leg in Edit mode\""}>Find route-aware candidates</button>
        <button type="button" id="inspect-route-edit" ${issueCapability?.enabled === true && acceptedRouteTarget ? "" : `disabled title=\"${escapeHtml(acceptedRouteTarget ? issueCapability?.blocking_code || "Route issue reporting is unavailable" : "Only an accepted-plan route leg can be edited")}\"`}>Add supported route feedback</button>
        <button type="button" disabled title="${escapeHtml(preferenceCapability?.blocking_code || "Route preference editing is unavailable")}">Change route preference</button></div>
    </section>`;
  }
  if (interaction.selectionKind === "exploratory" && interaction.exploratoryPin) {
    const pin = interaction.exploratoryPin;
    const preview = interaction.snapPreview;
    const isRelocation = Boolean(pin.targetWaypointId);
    const operationType = preview?.operation_intent || snapOperationIntent(pin.intent, pin);
    const confirmable = roadCheckedPreviewCanConfirm(preview)
      && Boolean(mapEditCapability(operationType));
    const snapped = preview?.snapped_coordinate || preview?.snapped || null;
    const snappedLabel = snapped && Number.isFinite(Number(snapped.longitude)) && Number.isFinite(Number(snapped.latitude))
      ? `${Number(snapped.longitude).toFixed(5)}, ${Number(snapped.latitude).toFixed(5)}`
      : null;
    const snapDescription = interaction.snapState === "checking"
      ? "Checking local road access…"
      : interaction.snapState === "failed"
        ? "The local road-access check failed. The raw pin remains non-executable and can be corrected or cancelled."
      : preview
        ? `Road-check result: ${escapeHtml(preview.validation_state || preview.status || "returned")}. ${snappedLabel ? `Snapped to ${escapeHtml(snappedLabel)}.` : "No authoritative snapped coordinate was returned."}${preview.validation_state === "route_checked" ? " Affected route legs were checked by the local router." : " This point is not executable without affected route evidence."}`
        : "No road-access check has run. This point cannot become an executable draft.";
    const customForm = pin.intent === "possible_stop" ? `<label for="waypoint-role">Itinerary role</label>
      <select id="waypoint-role" required>
        ${["attraction", "activity", "meal", "rest_stop", "scenic_stop"].map(
          (role) => `<option value="${role}">${escapeHtml(role.replaceAll("_", " "))}</option>`,
        ).join("")}
      </select>
      <label for="waypoint-duration-mode">Visit duration rule</label>
      <select id="waypoint-duration-mode" required>
        ${["exact", "preferred", "minimum", "maximum", "range"].map(
          (mode) => `<option value="${mode}">${escapeHtml(mode)}</option>`,
        ).join("")}
      </select>
      <div id="waypoint-duration-fields">${durationInputRows("exact")}</div>` : "";
    const confirmForm = confirmable ? `<form id="confirm-map-waypoint-form">
      <p><strong>Insertion context:</strong> Day ${escapeHtml(preview.selected_day ?? state.session.selected_day)} · selected road-validated route slot.</p>
      ${isRelocation ? `<p><strong>Relocating:</strong> ${escapeHtml(pin.retainedName || pin.targetWaypointId)}. Its name, role, duration, and day remain unchanged.</p>` : `<label for="waypoint-name">${pin.intent === "route_waypoint" ? "Waypoint label" : "Stop name"}</label>
        <input id="waypoint-name" name="waypoint-name" type="text" minlength="1" maxlength="120" required>
        ${customForm}`}
      ${preview.confirmation_required ? `<label class="snap-warning-ack"><input id="snap-warning-ack" type="checkbox" required> I reviewed the snap distance of ${escapeHtml(Math.round(Number(preview.snap_distance_m)))} m and want to continue.</label>` : ""}
      <button type="submit" class="primary">${isRelocation ? "Confirm relocation" : pin.intent === "route_waypoint" ? "Add route waypoint" : "Add custom stop"}</button>
    </form>` : `<p class="truth-note">${preview && preview.validation_state !== "route_checked"
      ? "This preview is not executable. Select an exact route leg and check road access again."
      : pin.intent === "explore_only" ? "Exploration never appends an itinerary operation."
      : "A route-checked server preview is required before confirmation."}</p>`;
    return `<section class="map-context-card exploratory" aria-labelledby="map-context-title">
      <span class="status-pill ${confirmable ? "route-checked" : "warning"}">${confirmable ? "Route-checked preview" : "Exploratory only"}</span>
      <h3 id="map-context-title">${isRelocation ? "Unconfirmed relocation" : "Unconfirmed map point"}</h3>
      <p>Raw coordinate: ${pin.longitude.toFixed(5)}, ${pin.latitude.toFixed(5)}. A raw point is not a road-valid stop.</p>
      ${isRelocation ? "" : `<label for="exploratory-meaning">What should this point mean?</label>
      <select id="exploratory-meaning">
        <option value="explore_only" ${pin.intent === "explore_only" ? "selected" : ""}>Explore this area only</option>
        <option value="possible_stop" ${pin.intent === "possible_stop" ? "selected" : ""}>Consider a possible stop</option>
        <option value="route_waypoint" ${pin.intent === "route_waypoint" ? "selected" : ""}>Consider a route-only waypoint</option>
      </select>`}
      <p class="snap-preview-state" role="status">${snapDescription}</p>
      <div class="map-context-actions"><button type="button" id="check-road-access" ${interaction.snapState === "checking" ? "disabled" : ""}>Check road access</button>
        <button type="button" id="cancel-exploratory">Cancel point</button></div>
      ${confirmForm}
    </section>`;
  }
  return `<section class="map-context-card guidance" aria-labelledby="map-context-title">
    <span class="status-pill">${state.mapInteraction.mode === "edit" ? "Edit mode" : "Select mode"}</span>
    <h3 id="map-context-title">${state.mapInteraction.mode === "edit" ? "Choose a map target" : "Inspect the itinerary"}</h3>
    <p>${state.mapInteraction.mode === "edit"
      ? "Select a stop or route leg, or click empty map space to place a non-executable exploratory pin."
      : "Select a stop or validated route leg. Switch to Edit mode before placing an exploratory pin."}</p>
  </section>`;
}

function renderMap() {
  const geography = activeGeography();
  const mapDraft = confirmedMapDraftState();
  const geographyReady = geography?.schema_version === "product-geography-v2"
    && ["ready", "ready_with_gaps"].includes(geography.status);
  const original = geographyReady ? geography.plans?.find((plan) => plan.role === "original") : null;
  const repaired = geographyReady ? geography.plans?.find((plan) => plan.role === "registered_repair") : null;
  const proposalEligible = state.session.proposal?.schema_version === "draft-preview-v1"
    && state.session.proposal?.state === "eligible"
    && Boolean(state.session.proposal?.child_plan_id);
  const selectedAlternative = geographyReady ? geography.plans?.find(
    (plan) => ["alternative", "draft_preview"].includes(plan.role)
      && plan.plan_id === activeAlternativeId(),
  ) : null;
  const displayed = proposalEligible ? selectedAlternative : selectedAlternative || repaired || original;
  const displayedStopIds = new Set(displayed?.stops?.features?.map((feature) => feature.properties?.stop_id) || []);
  const accessibleStops = [
    ...(displayed?.stops?.features || []).map((feature) => ({ feature, removed: false })),
    ...(original?.stops?.features || [])
      .filter((feature) => !displayedStopIds.has(feature.properties?.stop_id))
      .map((feature) => ({ feature, removed: true })),
  ];
  const dayRouteDescription = selectedAlternative
    ? routeSummaryForPlan(selectedAlternative)
    : state.workspace.map_alternative;
  const routeDescription = textualRouteEvidence(displayed, dayRouteDescription);
  const routeLegs = validatedRouteFeatures(displayed);
  const routeGaps = routeGapFeatures(displayed);
  const removedByLabel = selectedAlternative ? "selected alternative" : "registered repair";
  const stopButtons = accessibleStops.map(({ feature, removed }) => {
    const stop = feature.properties || {};
    const selected = stop.stop_id === state.session.selected_stop_id;
    return `<li><button type="button" class="${removed ? "is-removed" : ""}" data-geo-stop="${escapeHtml(stop.stop_id)}" data-geo-day="${escapeHtml(stop.day)}" data-geo-plan="${escapeHtml(stop.plan_id || "")}" aria-pressed="${selected}">
      <span>Day ${escapeHtml(stop.day)}</span><strong>${escapeHtml(stop.name || stop.stop_id)}</strong><small>${escapeHtml(stop.city || "City unavailable")}${removed ? ` · Original only; removed by ${removedByLabel}` : ""}</small>
    </button></li>`;
  }).join("") || "";
  const routeButtons = routeLegs.map((feature) => {
    const leg = feature.properties || {};
    const selected = leg.route_leg_id === state.session.selected_segment_id;
    const names = routeNamePair(leg, displayed);
    return `<li><button type="button" data-geo-route-leg="${escapeHtml(leg.route_leg_id)}" aria-pressed="${selected}">
      <strong>${escapeHtml(names.origin)} → ${escapeHtml(names.destination)}</strong>
      <small>Day ${escapeHtml(leg.day ?? "Unavailable")} · ${escapeHtml(leg.travel_mode || "Unavailable")} · road validated</small>
    </button></li>`;
  }).join("");
  const gapItems = routeGaps.map((feature) => {
    const gap = feature.properties || {};
    return `<li><strong>Unvalidated route gap:</strong> ${escapeHtml(stopNameForId(gap.origin_id, displayed))} → ${escapeHtml(stopNameForId(gap.destination_id, displayed))} (${escapeHtml(humanLabel(gap.failure_code || "evidence unavailable"))})</li>`;
  }).join("");
  const customWaypointItems = mapDraft.waypoints.map((waypoint) => {
    const selected = waypoint.waypoint_id === state.mapInteraction.selectedWaypointId;
    const relocateEnabled = state.mapInteraction.mode === "edit"
      && Boolean(mapEditCapability("relocate_custom_waypoint"));
    return `<li class="accessible-draft-waypoint">
      <button type="button" data-draft-waypoint="${escapeHtml(waypoint.waypoint_id)}" aria-pressed="${selected}">
        <strong>${escapeHtml(waypoint.name)}</strong>
        <small>Day ${escapeHtml(waypoint.day ?? "Unavailable")} · ${escapeHtml(waypoint.role)} · route-checked draft</small>
      </button>
      <button type="button" data-relocate-waypoint="${escapeHtml(waypoint.waypoint_id)}" ${relocateEnabled ? "" : "disabled"} aria-label="Relocate ${escapeHtml(waypoint.name)}">Relocate</button>
    </li>`;
  }).join("");
  const draftCount = state.session.draft.length;
  const routeCheckedOnly = hasRouteCheckedOnlyDraft();
  const impactSummary = state.typedImpact.response?.summary;
  const impactCurrent = state.typedImpact.response?.session_revision === state.session.revision;
  const evaluatedPreviewReady = impactCurrent && impactSummary?.can_run_evaluated_preview === true;
  const impactBlockerLabel = impactSummary?.blocking_codes?.length
    ? `Preview blocked: ${impactSummary.blocking_codes.map(humanLabel).join(", ")}.`
    : null;
  $("#workspace-content").innerHTML = `<div class="map-stage">
    <div id="geographic-map" class="geographic-map" aria-hidden="true"></div>
    <div id="map-render-status" class="map-render-status" role="status">
      ${geographyReady ? "Loading verified local geography…" : "Geographic evidence is unavailable. Use the text route description."}
    </div>
    <div class="map-mode-switch" role="group" aria-label="Map interaction mode">
      <button type="button" data-map-mode="select" aria-pressed="${state.mapInteraction.mode === "select"}">Select</button>
      <button type="button" data-map-mode="edit" aria-pressed="${state.mapInteraction.mode === "edit"}">Edit</button>
    </div>
    <div class="map-note"><strong>Day ${escapeHtml(state.session.selected_day)} highlighted</strong><br>${escapeHtml(state.workspace.issue.label)} · ${escapeHtml(state.workspace.issue.summary)}</div>
    ${mapInspectorHtml(geography, displayed)}
    ${candidateExplorerHtml()}
    <details class="textual-map-alternative">
      <summary>Text route description</summary>
      <p>${escapeHtml(routeDescription)}</p>
      ${textRoutePanelHtml(displayed, original)}
      ${stopButtons ? `<ol class="accessible-stop-list">${stopButtons}</ol>` : ""}
      ${routeButtons ? `<h3>Validated route legs</h3><ol class="accessible-route-list">${routeButtons}</ol>` : ""}
      ${customWaypointItems ? `<h3>Confirmed custom draft waypoints</h3><ol class="accessible-draft-waypoint-list">${customWaypointItems}</ol>` : ""}
      ${gapItems ? `<div class="route-gap-summary" role="alert"><h3>Required route gaps</h3><ul>${gapItems}</ul></div>` : ""}
    </details>
    <div class="map-legend" aria-label="Map legend">
      <span><i class="line-swatch original"></i>Original</span>
      <span><i class="line-swatch repair"></i>Registered repair</span>
      <span><i class="line-swatch selected-day"></i>Selected Day ${escapeHtml(state.session.selected_day)}</span>
      ${selectedAlternative ? `<span><i class="line-swatch alternative"></i>Selected alternative</span>` : ""}
      ${routeGaps.length ? `<span><i class="marker-swatch gap"></i>Unvalidated gap</span>` : ""}
      ${mapDraft.waypoints.length ? `<span><i class="marker-swatch draft-waypoint"></i>Route-checked draft waypoint</span><span><i class="line-swatch draft-route"></i>Affected draft route</span>` : ""}
      ${state.poiDiscovery.response?.candidates?.length ? `<span><i class="marker-swatch poi-candidate"></i>POI candidate · discovery only</span>` : ""}
      <span><i class="marker-swatch changed"></i>Changed</span>
    </div>
    ${mapAttribution(geography)}
    <div class="draft-bar">
      <strong>${draftCount} draft ${draftCount === 1 ? "change" : "changes"}</strong>
      <span>${routeCheckedOnly ? "Route-checked draft; evaluated repair is not enabled for arbitrary waypoints yet." : proposalEligible ? `Evaluated ${escapeHtml(state.session.proposal.evaluation_status || state.session.proposal.eligibility || "proposal")} certificate is current for this draft.` : draftCount && impactCurrent ? impactBlockerLabel ? escapeHtml(impactBlockerLabel) : `${impactSummary.evaluated_executable_count} evaluated-preview capable · ${impactSummary.draft_only_count} draft only · not certified.` : draftCount ? "Checking server-owned draft impact; evaluation is disabled until it resolves." : state.mapInteraction.mode === "edit" ? "Map clicks remain exploratory until road checking and explicit confirmation." : "Select a stop or road-validated route leg."}</span>
      <button type="button" id="inline-undo" ${draftCount ? "" : "disabled"}>Undo</button>
      <button type="button" id="inline-preview" class="primary" ${draftCount && evaluatedPreviewReady ? "" : `disabled title="${draftCount ? escapeHtml(impactBlockerLabel || "This draft includes non-executable or not-yet-classified operations") : "Add a draft before preview"}"`}>Preview repair</button>
    </div>
  </div>`;
  $$('[data-geo-stop]').forEach((button) => button.addEventListener("click", () => selectMapStop({
    plan_id: button.dataset.geoPlan || null,
    selected_day: Number(button.dataset.geoDay),
    selected_stop_id: button.dataset.geoStop,
  })));
  $$('[data-geo-route-leg]').forEach((button) => button.addEventListener("click", () => {
    const feature = routeLegs.find(
      (candidate) => candidate.properties?.route_leg_id === button.dataset.geoRouteLeg,
    );
    selectRouteLeg(feature?.properties || null);
  }));
  $$('[data-draft-waypoint]').forEach((button) => button.addEventListener("click", () => {
    selectCustomWaypoint(button.dataset.draftWaypoint);
  }));
  $$('[data-relocate-waypoint]').forEach((button) => button.addEventListener("click", () => {
    beginCustomWaypointRelocation(button.dataset.relocateWaypoint);
  }));
  $$('[data-map-mode]').forEach((button) => button.addEventListener("click", () => {
    setMapInteractionMode(button.dataset.mapMode);
  }));
  $("#inspect-edit-stop")?.addEventListener("click", openMapEdit);
  $("#prepare-replacement-candidates")?.addEventListener("click", prepareReplacementDiscovery);
  $("#find-route-candidates")?.addEventListener("click", fetchPoiCandidates);
  $("#relocate-custom-waypoint")?.addEventListener("click", () => {
    beginCustomWaypointRelocation(state.mapInteraction.selectedWaypointId);
  });
  $("#inspect-route-edit")?.addEventListener("click", openMapEdit);
  $("#exploratory-meaning")?.addEventListener("change", (event) => {
    state.mapInteraction.exploratoryPin.intent = event.currentTarget.value;
    state.mapInteraction.snapPreview = null;
    state.mapInteraction.snapState = "idle";
    renderMap();
  });
  $("#check-road-access")?.addEventListener("click", requestExploratorySnapPreview);
  $("#cancel-exploratory")?.addEventListener("click", cancelExploratoryPin);
  $("#waypoint-duration-mode")?.addEventListener("change", (event) => {
    $("#waypoint-duration-fields").innerHTML = durationInputRows(event.currentTarget.value);
  });
  $("#confirm-map-waypoint-form")?.addEventListener("submit", confirmMapWaypoint);
  $("#close-poi-candidates")?.addEventListener("click", closePoiDiscovery);
  $("#poi-candidate-filter")?.addEventListener("submit", refreshPoiCandidateFilters);
  $("#poi-layer-toggle")?.addEventListener("change", (event) => {
    state.poiDiscovery.showLayer = event.currentTarget.checked;
    renderMap();
  });
  $$('[data-poi-candidate]').forEach((button) => button.addEventListener("click", () => {
    void selectPoiCandidate(button.dataset.poiCandidate);
  }));
  $("#use-candidate-replacement")?.addEventListener("click", useSelectedCandidateReplacement);
  $("#inline-undo").addEventListener("click", undoDraft);
  $("#inline-preview").addEventListener("click", previewDraft);

  if (proposalEligible && !selectedAlternative) {
    updateGeographicMapStatus({ state: "failed", code: "evaluated_preview_geography_unavailable" });
    return;
  }
  renderGeographicMap({
    container: $("#geographic-map"),
    geography,
    mapConfig: state.mapConfig,
    selectedStopId: state.session.selected_stop_id,
    selectedAlternativeId: activeAlternativeId(),
    selectedDay: state.session.selected_day,
    selectedRouteLegId: state.session.selected_segment_id,
    interactionMode: state.mapInteraction.mode,
    exploratoryPin: state.mapInteraction.exploratoryPin,
    draftWaypoints: mapDraft.waypoints,
    draftRouteLegs: mapDraft.routeLegs,
    selectedWaypointId: state.mapInteraction.selectedWaypointId,
    poiCandidates: state.poiDiscovery.response?.candidates || [],
    selectedCandidateId: state.poiDiscovery.selectedCandidateId,
    showPoiCandidates: state.poiDiscovery.showLayer,
    onSelectStop: selectMapStop,
    onSelectRouteLeg: selectRouteLeg,
    onSelectCandidate: (candidate) => selectPoiCandidate(candidate.candidate_id),
    onSelectCustomWaypoint: (waypoint) => selectCustomWaypoint(waypoint.waypoint_id),
    onCustomWaypointDrag: beginCustomWaypointDragPreview,
    onEmptyMapClick: createExploratoryPin,
    onStatus: updateGeographicMapStatus,
  });
}

function mapAttribution(geography) {
  const attribution = geography?.attribution || {};
  const label = attribution.label || state.mapConfig?.attribution;
  const url = attribution.url || state.mapConfig?.attribution_url;
  if (!label) return "";
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "https:") throw new Error("invalid_attribution_url");
    return `<div class="map-attribution"><a href="${escapeHtml(parsed.href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a></div>`;
  } catch {
    return `<div class="map-attribution">${escapeHtml(label)}</div>`;
  }
}

function updateGeographicMapStatus(status) {
  const element = $("#map-render-status");
  if (!element) return;
  element.classList.remove("is-warning", "is-error");
  if (["ready", "ready_with_gaps"].includes(status.state)) {
    const notices = [];
    if (Number(status.visiblePlanCount) < 2) notices.push("Only the accepted-plan route is available in this pane.");
    if (status.gapCount) notices.push(`${status.gapCount} required route connection${status.gapCount === 1 ? " is" : "s are"} shown as an explicit unvalidated gap.`);
    if (!notices.length) {
      element.textContent = "Map ready with loaded local tiles and road-validated route layers.";
      element.hidden = true;
    } else {
      element.hidden = false;
      element.classList.add("is-warning");
      element.textContent = notices.join(" ");
    }
    return;
  }
  element.hidden = false;
  element.classList.add("is-error");
  element.textContent = "The geographic map could not be verified. The text route description remains available.";
}

function timeWindowLabel(value) {
  if (!value) return "Unavailable";
  const parts = [];
  if (value.earliest_arrival) parts.push(`earliest service admission ${value.earliest_arrival}`);
  if (value.latest_departure) parts.push(`latest departure after visit ${value.latest_departure}`);
  return parts.join(" · ") || "Unavailable";
}

function scheduleClockLabel(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "Unavailable";
  const wholeSeconds = Math.round(value * 60);
  const hours = Math.floor(wholeSeconds / 3600);
  const minutes = Math.floor((wholeSeconds % 3600) / 60);
  const seconds = wholeSeconds % 60;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}${seconds ? `:${String(seconds).padStart(2, "0")}` : ""}`;
}

function evaluatedEditEvidenceHtml(proposal) {
  if (proposal?.schema_version !== "draft-preview-v1"
      || !["eligible", "ineligible"].includes(proposal.state)) return "";
  const operations = proposal.compiled_request?.operations || [];
  const operationRows = operations.map((operation) => {
    const target = stopNameForId(operation.target, acceptedPlan(activeGeography()));
    const request = operation.type === "set_stop_order"
      ? `Day ${operation.parameters.day} · position ${operation.parameters.sequence_index + 1}`
      : operation.type === "set_stop_duration"
        ? `${operation.parameters.duration.preferred_minutes} min · exact visit duration`
        : operation.type === "set_stop_time_window"
          ? timeWindowLabel(operation.parameters)
          : operation.type === "set_stop_role"
            ? `${humanLabel(operation.parameters.role)} · trip-specific user-declared use`
            : operationDisplayName(operation.type);
    return `<li><strong>${escapeHtml(operationDisplayName(operation.type))}</strong><span>${escapeHtml(target)} · ${escapeHtml(request)}</span></li>`;
  }).join("");
  const route = proposal.route_validation || {};
  const repair = proposal.repair;
  const roleEvidence = proposal.plan_diff?.schema_version === "plan-diff-v4";
  const schedule = proposal.schedule_impact || proposal.role_impact;
  const scheduleAccounting = roleEvidence ? schedule?.child_schedule : schedule?.child;
  const timeWindowEvidence = proposal.plan_diff?.schema_version === "plan-diff-v3";
  const scheduleStatus = schedule
    ? `${humanLabel(scheduleAccounting.status)} · evaluator-owned modeled accounting`
    : proposal.evidence?.hard_feasibility_status === "PASSED"
      ? "Hard feasibility passed; stop-by-stop schedule accounting is Unavailable in this response."
      : "Unavailable";
  const durationChanges = proposal.plan_diff?.duration_changes || [];
  const durationRows = durationChanges.map((change) => {
    const explicitFrom = change.from_minutes == null
      ? "Explicit parent duration Unavailable"
      : `${formatMetric(change.from_minutes)} min explicit parent duration`;
    const accountingFrom = change.accounting_from_minutes == null
      ? "accounting baseline Unavailable"
      : `${formatMetric(change.accounting_from_minutes)} min accounting baseline (${humanLabel(change.accounting_from_source)})`;
    return `<li><strong>${escapeHtml(stopNameForId(change.stop_id, acceptedPlan(activeGeography())))}</strong><span>${escapeHtml(explicitFrom)} · ${escapeHtml(accountingFrom)} → ${escapeHtml(formatMetric(change.to_minutes))} min exact · ${escapeHtml(formatMetric(change.delta_minutes))} min modeled change</span></li>`;
  }).join("");
  const timeWindowChanges = proposal.plan_diff?.time_window_changes || [];
  const timeWindowRows = timeWindowChanges.map((change) => `
    <li><strong>${escapeHtml(stopNameForId(change.stop_id, acceptedPlan(activeGeography())))}</strong>
      <span>${escapeHtml(timeWindowLabel(change.from_constraint))} → ${escapeHtml(timeWindowLabel(change.to_constraint))} · trip-specific user constraint</span></li>`).join("");
  const roleChanges = proposal.plan_diff?.role_changes || [];
  const roleRows = roleChanges.map((change) => `
    <li><strong>${escapeHtml(stopNameForId(change.stop_id, acceptedPlan(activeGeography())))}</strong>
      <span>${escapeHtml(unavailable(change.from_role, humanLabel))} → ${escapeHtml(humanLabel(change.to_role))} · ${escapeHtml(humanLabel(change.to_source))}</span></li>`).join("");
  const targetStopIds = new Set(schedule?.target_stop_ids || []);
  const traceRows = timeWindowEvidence ? (schedule?.child?.stop_traces || [])
    .filter((trace) => targetStopIds.has(trace.stop_id))
    .map((trace) => {
      const visitSource = trace.visit_duration_source === "configured_evaluator_default"
        ? "configured evaluator fallback; not source-observed"
        : humanLabel(trace.visit_duration_source);
      return `<li><strong>${escapeHtml(stopNameForId(trace.stop_id, acceptedPlan(activeGeography())))}</strong><span>
        Incoming road leg from ${escapeHtml(trace.incoming_origin_id)} · ${escapeHtml(formatMetric(trace.incoming_travel_minutes))} min · route query <code>${escapeHtml(trace.incoming_route_query_hash)}</code> · raw road arrival ${escapeHtml(scheduleClockLabel(trace.road_arrival_minute))} · opening-hours evidence ${escapeHtml(humanLabel(trace.opening_window_source))} · opening-hours wait ${escapeHtml(formatMetric(trace.opening_wait_minutes))} min · required-window wait ${escapeHtml(formatMetric(trace.required_window_wait_minutes))} min · service starts ${escapeHtml(scheduleClockLabel(trace.service_start_minute))} · visit ${escapeHtml(formatMetric(trace.visit_minutes))} min (${escapeHtml(visitSource)}) · departs ${escapeHtml(scheduleClockLabel(trace.departure_minute))} · latest-departure status ${escapeHtml(humanLabel(trace.latest_departure_status))} · overrun ${escapeHtml(formatMetric(trace.latest_departure_overrun_minutes))} min</span></li>`;
    }).join("") : "";
  const modeledSchedule = schedule ? schedule.affected_days.map((day) => {
    const metrics = scheduleAccounting.metrics;
    return `<li><strong>Day ${escapeHtml(day)}</strong><span>Road travel ${escapeHtml(formatMetric(metrics[`day_${day}_travel_minutes`]))} min · visit ${escapeHtml(formatMetric(metrics[`day_${day}_visit_minutes`]))} min · modeled wait ${escapeHtml(formatMetric(metrics[`day_${day}_waiting_minutes`]))} min · modeled total ${escapeHtml(formatMetric(metrics[`day_${day}_total_minutes`]))} min · limit ${escapeHtml(formatMetric(metrics[`day_${day}_limit_minutes`]))} min · slack ${escapeHtml(formatMetric(metrics[`day_${day}_slack_minutes`]))} min · overrun ${escapeHtml(formatMetric(metrics[`day_${day}_overrun_minutes`]))} min</span></li>`;
  }).join("") : "";
  const scheduleDisclosure = schedule ? `<p><strong>Modeled components:</strong> ${timeWindowEvidence ? "road travel, visit duration, opening wait when window evidence exists, required-window wait, required latest departure, and day limit" : "road travel, visit duration, opening wait when window evidence exists, and day limit"}.</p>
    <p><strong>Unavailable components:</strong> parking/drop-off, walking transfer, queue wait, and service buffer.</p>
    ${timeWindowEvidence ? `<p><strong>Window semantics:</strong> Each day starts at the configured ${escapeHtml(scheduleClockLabel(schedule.configured_day_start_minute))} accounting anchor. Raw road arrival is reconstructed from the exact validated incoming route leg and prior departure; it may be earlier than the trip-specific bound. Service waits until earliest arrival. Latest departure is checked after the visit. These constraints are separate from place opening hours.</p>` : ""}
    <p><strong>Opening-window evidence:</strong> ${scheduleAccounting.missing_opening_window_stop_ids.length ? `${escapeHtml(scheduleAccounting.missing_opening_window_stop_ids.length)} stops unavailable; modeled accounting is incomplete evidence.` : "Available for all modeled stops."}</p>
    ${roleEvidence ? '<p><strong>Role semantics:</strong> This is a trip-specific user-declared use. Place identity and categories are unchanged; semantic fit and recommendation are Unavailable. Supported visit roles have no route or schedule effect.</p>' : ""}` : "";
  const outcome = proposal.state === "eligible"
    ? '<span class="status-pill success">Evaluated eligible</span>'
    : `<span class="status-pill warning">Evaluated ineligible</span><p>Blocking reason: ${escapeHtml(humanLabel(proposal.reason || "Unavailable"))}</p>`;
  return `<details class="evaluated-edit-evidence" open>
    <summary>Evaluated edit evidence</summary>
    <div class="evaluated-edit-evidence-grid">
      <section><h4>Requested draft</h4><ol>${operationRows || "<li>Unavailable</li>"}</ol><p>Draft hash <code>${escapeHtml(proposal.draft_content_hash)}</code></p></section>
      <section><h4>Child and diff</h4><dl>
        <dt>Parent</dt><dd>${escapeHtml(proposal.parent_plan_id)}</dd>
        <dt>In-memory child</dt><dd>${escapeHtml(unavailable(proposal.child_plan_id))}</dd>
        <dt>Diff</dt><dd>${escapeHtml(unavailable(proposal.diff_id))}</dd>
        <dt>Affected days</dt><dd>${escapeHtml(unavailable(repair?.changed?.affected_day_count))}</dd>
        <dt>Road changes</dt><dd>${escapeHtml(unavailable(repair?.changed?.road_change_count))}</dd>
        <dt>Weighted edit cost</dt><dd>${escapeHtml(formatMetric(repair?.tradeoffs?.weighted_edit_cost))}</dd>
      </dl></section>
      <section><h4>Schedule and route</h4><dl>
        <dt>Schedule detail</dt><dd>${escapeHtml(scheduleStatus)}</dd>
        <dt>Road-validated legs</dt><dd>${route.road_validated_leg_count == null || route.required_leg_count == null ? "Unavailable" : `${escapeHtml(route.road_validated_leg_count)}/${escapeHtml(route.required_leg_count)}`}</dd>
        <dt>Fallback legs</dt><dd>${escapeHtml(unavailable(route.fallback_leg_count))}</dd>
        <dt>Route matrix</dt><dd>${escapeHtml(unavailable(proposal.route_validation_identity?.matrix_id))}</dd>
      </dl>${scheduleDisclosure}${modeledSchedule ? `<ol>${modeledSchedule}</ol>` : ""}</section>
      ${durationRows ? `<section><h4>Typed duration diff</h4><ol>${durationRows}</ol><p>Duration changes are PlanDiff v2 evidence; they are not time shifts or road changes.</p></section>` : ""}
      ${timeWindowRows ? `<section><h4>Typed time-window diff</h4><ol>${timeWindowRows}</ol><p>Time-window changes are PlanDiff v3 trip constraints. They are not place opening hours, latest-start bounds, or time shifts.</p></section>` : ""}
      ${roleRows ? `<section><h4>Typed itinerary-role diff</h4><ol>${roleRows}</ol><p>Role changes are PlanDiff v4 trip-use evidence. They do not change PlaceEntity identity, place categories, road geometry, or modeled schedule.</p></section>` : ""}
      ${traceRows ? `<section><h4>Required-window schedule trace</h4><ol>${traceRows}</ol></section>` : ""}
      <section><h4>Independent certificate</h4>${outcome}<dl>
        <dt>Certificate</dt><dd>${escapeHtml(unavailable(proposal.certificate_id))}</dd>
        <dt>Evaluation</dt><dd>${escapeHtml(unavailable(proposal.evaluation_status))}</dd>
        <dt>Artifact grounding</dt><dd>${escapeHtml(unavailable(proposal.evidence?.artifact_grounding_status))}</dd>
      </dl></section>
    </div>
    <p class="truth-note">This is a route-validated, independently evaluated session preview. The accepted parent is unchanged; acceptance remains disabled until W5.</p>
  </details>`;
}

function renderRepair() {
  const issue = state.workspace.issue;
  const proposal = state.session.proposal;
  const evaluated = proposal?.schema_version === "draft-preview-v1" && ["eligible", "ineligible"].includes(proposal.state);
  const eligible = evaluated && proposal.state === "eligible";
  const registered = proposal?.state === "eligible" && proposal?.provenance === "registered_fixture_artifacts";
  const repair = evaluated || registered ? proposal.repair : state.workspace.repair;
  const statusLabel = eligible ? "Evaluated eligible proposal" : evaluated ? "Evaluated ineligible proposal" : registered ? "Registered fixture proposal" : "Preview required";
  const repairTitle = eligible ? "Evaluated draft repair" : evaluated ? "No eligible draft repair" : registered ? "Registered repair fixture" : "Registered repair";
  const repairBody = repair ? `
      <p>${escapeHtml(repair.result)}</p>
      <ul class="change-list">
        <li>${repair.changed.affected_day_count} affected day</li>
        <li>${repair.permissions.booked_change_count} booked changes</li>
        <li>${repair.permissions.locked_change_count} locked changes</li>
        <li>Contextual risk delta: ${formatMetric(repair.tradeoffs.weather_risk_delta)}</li>
      </ul>` : `<p>The evaluated draft produced no eligible child. Reason: ${escapeHtml(proposal?.reason || "Unavailable")}</p>`;
  $("#repair-panel").innerHTML = `
    <span class="eyebrow">Repair review</span>
    <h2 id="repair-heading">Decision workspace</h2>
    <span class="status-pill ${eligible || registered ? "success" : "warning"}">${escapeHtml(statusLabel)}</span>
    <article class="issue-card">
      <h3>Issue detected · Registered weather disruption</h3>
      <p>${escapeHtml(issue.summary)}</p>
    </article>
    <article class="repair-card">
      <h3>${repairTitle}</h3>
      ${proposal?.message ? `<p class="truth-note">${escapeHtml(proposal.message)}</p>` : ""}
      ${repairBody}
      ${evaluatedEditEvidenceHtml(proposal)}
    </article>
    <div class="action-stack">
      <button class="primary" id="accept-repair" disabled>Accept evaluated repair</button>
      <button id="compare-repair">Compare alternatives</button>
      <button id="review-evidence">Review evidence</button>
      <button id="keep-original" disabled>Keep original</button>
    </div>
    <p class="truth-note">The parent and child artifacts are immutable. Acceptance and Keep original remain disabled until the W5 repository transaction is implemented. This prototype does not verify E5.</p>`;
  $("#compare-repair").addEventListener("click", () => navigate("/app/compare"));
  $("#review-evidence").addEventListener("click", () => navigate("/app/evidence"));
}

function renderContextPanel() {
  if (state.route === "/app/map") return renderMapContextPanel();
  if (state.route === "/app/repairs") return renderDraftContextPanel();
  if (state.route === "/app/compare") return renderCompareContextPanel();
  if (state.route === "/app/evidence") return renderEvidenceContextPanel();
  return renderRepair();
}

function renderMapContextPanel() {
  const stop = findSelectedStop();
  const leg = state.mapInteraction.selectedRouteLeg;
  const selected = stop
    ? `<span class="status-pill success">Selected stop</span><h3>${escapeHtml(stop.name)}</h3><dl><dt>Day</dt><dd>${escapeHtml(stop.day ?? "Unavailable")}</dd><dt>Map mode</dt><dd>${escapeHtml(humanLabel(state.mapInteraction.mode))}</dd></dl>`
    : leg
      ? `<span class="status-pill success">Selected route</span><h3>${escapeHtml(stopNameForId(leg.origin_id))} → ${escapeHtml(stopNameForId(leg.destination_id))}</h3><dl><dt>Day</dt><dd>${escapeHtml(leg.day ?? "Unavailable")}</dd><dt>Evidence</dt><dd>Road validated</dd></dl>`
      : '<span class="status-pill">No map selection</span><p>Select a stop or road segment. The timeline and this panel will follow the same session selection.</p>';
  $("#repair-panel").innerHTML = `<span class="eyebrow">Map context</span><h2 id="repair-heading">Selection</h2>
    <article class="repair-card">${selected}</article>
    <div class="action-stack"><button id="context-select-mode" type="button">Select mode</button><button id="context-edit-mode" type="button">Edit selected item</button></div>`;
  $("#context-select-mode").addEventListener("click", () => setMapInteractionMode("select"));
  $("#context-edit-mode").addEventListener("click", () => setMapInteractionMode("edit"));
}

function renderDraftContextPanel() {
  const impact = state.typedImpact.response?.summary;
  const operations = state.session.draft.map((operation) => `<li><strong>${escapeHtml(operationDisplayName(operation.type))}</strong><span>${escapeHtml(stopNameForId(operation.target))}</span></li>`).join("");
  $("#repair-panel").innerHTML = `<span class="eyebrow">Repair draft</span><h2 id="repair-heading">Validation</h2>
    <span class="status-pill ${impact?.can_run_evaluated_preview ? "success" : "warning"}">${impact?.can_run_evaluated_preview ? "Ready to preview" : "Preview blocked or unavailable"}</span>
    <article class="repair-card"><h3>${state.session.draft.length} draft ${state.session.draft.length === 1 ? "operation" : "operations"}</h3>${operations ? `<ul class="selection-draft-list">${operations}</ul>` : "<p>No draft operations yet. Select a map item and switch to Edit.</p>"}</article>
    <div class="action-stack"><button id="context-undo" type="button" ${state.session.draft.length ? "" : "disabled"}>Undo latest</button><button id="context-preview" class="primary" type="button" ${canRunEvaluatedPreview() ? "" : "disabled"}>Preview repair</button></div>`;
  $("#context-undo").addEventListener("click", undoDraft);
  $("#context-preview").addEventListener("click", previewDraft);
}

function renderCompareContextPanel() {
  const resolution = resolveCompareSelection(state.workspace, state.session);
  const option = resolution.state === "selected" ? resolution.option : null;
  $("#repair-panel").innerHTML = `<span class="eyebrow">Selected alternative</span><h2 id="repair-heading">Tradeoff</h2>
    ${option ? `<span class="status-pill success">${escapeHtml(option.display_status || option.status || "Selected")}</span><article class="repair-card"><h3>${escapeHtml(option.method_label || "Selected repair")}</h3><dl><dt>Route time</dt><dd>${escapeHtml(option.route_total_minutes == null ? "Unavailable" : `${formatMetric(option.route_total_minutes)} minutes`)}</dd><dt>Edit cost</dt><dd>${escapeHtml(formatMetric(option.weighted_edit_cost))}</dd><dt>Road route</dt><dd>${option.route_validated ? "Validated" : "Unavailable"}</dd></dl><p>The maps, metrics, changes, and Evidence button refer to this same selected repair.</p></article>` : '<article class="repair-card"><h3>Choose a repair</h3><p>Select one available option in Compare.</p></article>'}
    <div class="action-stack"><button id="context-choose" class="primary" type="button" disabled title="Acceptance remains disabled until W5">Choose option (not enabled)</button><button id="context-evidence" type="button" ${resolution.evidenceExact ? "" : "disabled"}>View advanced Evidence</button></div>`;
  $("#context-evidence").addEventListener("click", () => navigate("/app/evidence"));
}

function renderEvidenceContextPanel() {
  const resolution = resolveCompareSelection(state.workspace, state.session);
  const proposal = state.session.proposal;
  const evaluated = proposal?.schema_version === "draft-preview-v1"
    && ["eligible", "ineligible"].includes(proposal.state);
  const exact = resolution.state === "selected" && resolution.evidenceExact;
  $("#repair-panel").innerHTML = `<span class="eyebrow">Evidence summary</span><h2 id="repair-heading">Eligibility checks</h2>
    <span class="status-pill ${evaluated || exact ? "success" : "warning"}">${evaluated ? escapeHtml(humanLabel(proposal.state)) : exact ? "Exact registered evidence" : "Selection required"}</span>
    <article class="repair-card"><h3>Current scope</h3><ul class="change-list"><li>Exact selected option: ${exact ? "verified" : "Unavailable"}</li><li>Road validation: ${exact || evaluated ? "checked" : "Unavailable"}</li><li>Acceptance: disabled until W5</li></ul><p>Technical plan, hash, diff, method, and certificate details are shown in the main Evidence view.</p></article>
    <div class="action-stack"><button id="context-back-compare" type="button">Back to Compare</button></div>`;
  $("#context-back-compare").addEventListener("click", () => navigate("/app/compare"));
}

function renderMetrics() {
  const proposal = state.session.proposal;
  const evaluated = proposal?.schema_version === "draft-preview-v1" && ["eligible", "ineligible"].includes(proposal.state);
  const registered = proposal?.state === "eligible" && proposal?.provenance === "registered_fixture_artifacts";
  const r = evaluated ? proposal.repair : state.workspace.repair;
  const preservation = r?.tradeoffs?.preservation_rate;
  const metrics = [
    ["Contextual risk", r ? formatMetric(r.tradeoffs.weather_risk_delta) : "Unavailable", "Evaluator-owned delta"],
    ["Preservation", preservation == null ? "Unavailable" : `${Math.round(preservation * 100)}%`, "Artifact-backed"],
    ["Edit cost", r ? formatMetric(r.tradeoffs.weighted_edit_cost) : "Unavailable", "Lower is better"],
    ["Certificate", evaluated ? proposal.evaluation_status || proposal.eligibility || "Unavailable" : registered ? proposal.certificate_id || "Registered fixture" : "Preview required", evaluated ? "Current evaluated draft certificate" : "Draft changes invalidate evaluated state"],
  ];
  $("#metric-strip").innerHTML = metrics.map(([label, value, note]) => `<div class="metric"><span>${label}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(note)}</small></div>`).join("");
}

function renderCompare() {
  renderCompareWorkspace({
    container: $("#workspace-content"),
    workspace: state.workspace,
    session: state.session,
    mapConfig: state.mapConfig,
    onBack: () => navigate("/app/itinerary"),
    onSelectAlternative: (planId) => selectContext({ selected_alternative_id: planId }),
    onOpenEvidence: (planId) => {
      if (state.session.selected_alternative_id === planId) navigate("/app/evidence");
    },
  });
}

function comparisonTable() {
  const alternatives = state.workspace.alternatives;
  const rows = [
    ["Independent eligibility", "required", "Unevaluated reference", ...alternatives.map((row) => row.display_status)],
    ["Strict route time", "lower", "Unavailable", ...alternatives.map((row) => row.route_total_minutes == null ? "Unavailable" : `${formatMetric(row.route_total_minutes)} minutes`)],
    ["Ownership-weighted edit cost", "lower", "0 (reference)", ...alternatives.map((row) => formatMetric(row.weighted_edit_cost))],
    ["Road-validated route", "required", "Unavailable", ...alternatives.map((row) => row.route_validated ? "Validated" : "Unavailable")],
    ["Certificate", "required", "Unavailable", ...alternatives.map((row) => row.certificate_id || "Unavailable")],
  ];
  const headings = ["Keep original", ...alternatives.map((row) => row.method_label)];
  return `<table class="comparison-table"><thead><tr><th>Metric</th><th>Direction</th>${headings.map((label) => `<th>${escapeHtml(label)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>
    <th scope="row">${escapeHtml(row[0])}</th><td>${escapeHtml(row[1])}</td>${row.slice(2).map((value) => `<td class="${value === "Unavailable" ? "unavailable" : ""}">${escapeHtml(value)}</td>`).join("")}
  </tr>`).join("")}</tbody></table>`;
}

function renderEvidence() {
  const proposal = state.session.proposal;
  const registeredSelection = resolveCompareSelection(state.workspace, state.session);
  const proposalMatches = !state.session.selected_alternative_id
    && proposal?.schema_version === "draft-preview-v1"
    && proposal?.state === "eligible"
    && proposal?.child_plan_id;
  const selectedAlternative = proposalMatches ? {
    plan_id: proposal.child_plan_id,
    method_label: "Evaluated draft repair",
  } : registeredSelection.state === "selected" && registeredSelection.evidenceExact
    ? registeredSelection.option
    : null;
  if (!selectedAlternative) {
    const code = state.session.selected_alternative_id
      ? registeredSelection.code
      : "evidence_option_not_selected";
    $("#workspace-content").innerHTML = `<div class="route-view">
      <header class="route-heading"><div><span class="eyebrow">Structured evidence</span><h2>Evidence unavailable</h2><p>Evidence is shown only for the exact option selected in Compare. No first-option fallback was used.</p></div><button class="secondary" id="evidence-back">Back to Compare</button></header>
      <article class="evidence-card"><h3>Select an artifact-backed option</h3><p>Open Compare, select one exact plan/hash, then use its Evidence button.</p><code>${escapeHtml(code)}</code></article>
    </div>`;
    $("#evidence-back").addEventListener("click", () => navigate("/app/compare"));
    return;
  }
  const selectedEvidence = proposalMatches ? {
    parent_plan_id: proposal.parent_plan_id,
    plan_id: proposal.child_plan_id,
    plan_content_hash: proposal.child_plan_content_hash,
    diff_id: proposal.diff_id,
    certificate_id: proposal.certificate_id,
    source_hashes: {
      child: proposal.child_plan_content_hash,
      diff: proposal.diff_content_hash,
      certificate: proposal.certificate_content_hash,
    },
    methods: {
      requested: [proposal.evidence?.requested_method || "typed_draft_compiler_v1"],
      executed: [proposal.evidence?.executed_method || proposal.execution_mode],
    },
    certificate: proposal.evidence,
  } : registeredSelection.evidence;
  const selectedPlan = proposalMatches ? activeProposalPlan() : state.workspace.geography?.plans?.find((plan) => plan.plan_id === selectedAlternative?.plan_id);
  const lineage = selectedEvidence ? {
    parent_plan_id: selectedEvidence.parent_plan_id,
    child_plan_id: selectedEvidence.plan_id,
    child_content_hash: selectedEvidence.plan_content_hash,
    diff_id: selectedEvidence.diff_id,
    certificate_id: selectedEvidence.certificate_id,
  } : {};
  const methods = selectedEvidence?.methods || {};
  const certificate = selectedEvidence?.certificate;
  const certVisible = Boolean(certificate);
  const checks = certVisible ? {
    artifact_grounding: certificate.artifact_grounding_status,
    modeled_feasibility: certificate.hard_feasibility_status,
    independent_evaluation: certificate.evaluation_status,
    route_publication_ready: certificate.route_validation?.publication_ready === true ? "Passed" : "Unavailable",
  } : {};
  const routeDescription = selectedPlan ? routeSummaryForPlan(selectedPlan) : "Unavailable";
  const sourceHashCount = Object.keys(selectedEvidence?.source_hashes || {}).length;
  $("#workspace-content").innerHTML = `<div class="route-view">
    <header class="route-heading"><div><span class="eyebrow">Structured evidence</span><h2>Evidence and lineage</h2><p>Missing and stale states are displayed, never coerced to success.</p></div><button class="secondary" id="evidence-back">Back to trip</button></header>
    <div class="evidence-grid">
      <article class="evidence-card"><h3>1 · Eligibility and certificate</h3>
        <p><strong>${escapeHtml(selectedAlternative?.method_label || "Registered repair")}</strong></p>
        <span class="status-pill ${certVisible ? "success" : "warning"}">${certVisible ? escapeHtml(certificate.evaluation_status) : "Certificate unavailable"}</span>
        <p>${certVisible ? `This plan is ${escapeHtml(certificate.comparison_eligibility)} with ${escapeHtml(certificate.route_validation?.road_validated_leg_count ?? "Unavailable")}/${escapeHtml(certificate.route_validation?.required_leg_count ?? "Unavailable")} road-validated legs.` : "No exact certificate is available for this selected plan and content hash."}</p>
      </article>
      <article class="evidence-card"><h3>2 · Policy and modeled-feasibility checks</h3><dl>${objectDl(checks)}</dl></article>
      <article class="evidence-card"><h3>3 · Route and source evidence</h3>
        <p>${escapeHtml(routeDescription)}</p>
        <p>${sourceHashCount || "No"} selected-plan source hashes are available.</p>
        ${proposalMatches ? '<p class="truth-note">Preview evidence is session-local and cannot be downloaded until W5 acceptance persists the child.</p>' : `<a class="download-link" href="/api/runs/${encodeURIComponent(state.session.run_id)}/evidence-bundle?plan_id=${encodeURIComponent(selectedAlternative.plan_id)}">Download selected-plan evidence</a>`}
      </article>
      <article class="evidence-card"><h3>4 · Requested and executed methods</h3>
        <details open><summary>Requested methods</summary><p>${escapeHtml((methods.requested || []).join(", ") || "Unavailable")}</p></details>
        <details><summary>Executed methods</summary><p>${escapeHtml((methods.executed || []).join(", ") || "Unavailable")}</p></details>
      </article>
      <article class="evidence-card"><h3>5 · Technical lineage</h3><dl>${objectDl(lineage)}</dl></article>
    </div>
    ${comparisonTable()}
  </div>`;
  $("#evidence-back").addEventListener("click", () => navigate("/app/itinerary"));
}

function renderRepairHistory() {
  const proposal = state.session.proposal;
  $("#workspace-content").innerHTML = `<div class="route-view">
    <header class="route-heading"><div><span class="eyebrow">Session state</span><h2>Repairs and drafts</h2><p>Local session revision ${state.session.revision}</p></div><button class="secondary" id="repairs-back">Back to trip</button></header>
    <div class="evidence-grid">
      <article class="evidence-card"><h3>Validated session baseline</h3><p><code>${escapeHtml(state.session.accepted_plan_id)}</code></p></article>
      <article class="evidence-card"><h3>Current proposal</h3><p>${escapeHtml(proposal?.message || "No evaluated proposal in this session.")}</p></article>
      <article class="evidence-card"><h3>Typed draft operations</h3>${state.session.draft.length ? `<ol>${state.session.draft.map((op) => `<li><strong>${escapeHtml(operationDisplayName(op.type))}</strong><span>${escapeHtml(draftTargetLabel(op))}</span></li>`).join("")}</ol>` : "<p>No active draft operations.</p>"}</article>
      <article class="evidence-card"><h3>Decision boundary</h3><p>Legacy decision files are preserved but not trusted or extended. Acceptance activates in W5. Copilot transcripts are stored only in local product state and expire after 30 days.</p></article>
    </div>
  </div>`;
  $("#repairs-back").addEventListener("click", () => navigate("/app/itinerary"));
}

function optionsHtml(values, selected = null, { protectedValues = [], labels = {} } = {}) {
  return values.map((value) => {
    const isProtected = protectedValues.includes(value);
    const label = labels[value] || humanLabel(value);
    return `<option value="${escapeHtml(value)}" ${value === selected ? "selected" : ""} ${isProtected ? "disabled" : ""}>${escapeHtml(label)}${isProtected ? " · permission required" : ""}</option>`;
  }).join("");
}

function durationModeSemantics(mode) {
  const capability = typedEditCapability("set_stop_duration");
  if (!capability
      || !Array.isArray(capability.supported_evaluated_modes)
      || !Array.isArray(capability.draft_only_modes)
      || !TYPED_EDIT_VOCABULARIES.duration_modes.includes(mode)) return null;
  if (capability.supported_evaluated_modes.includes(mode)) {
    return { feedback_tier: "evaluated", blocking_code: null };
  }
  if (capability.draft_only_modes.includes(mode)) {
    return {
      feedback_tier: "draft_only",
      blocking_code: capability.unsupported_mode_blocking_code,
    };
  }
  return null;
}

function durationModeOptionsHtml(selected = "exact") {
  const labels = Object.fromEntries(TYPED_EDIT_VOCABULARIES.duration_modes.map((mode) => {
    const semantics = durationModeSemantics(mode);
    const suffix = semantics?.feedback_tier === "evaluated" ? "evaluated preview" : "draft only";
    return [mode, `${humanLabel(mode)} · ${suffix}`];
  }));
  return optionsHtml(TYPED_EDIT_VOCABULARIES.duration_modes, selected, { labels });
}

function durationCapabilityNote(mode) {
  const semantics = durationModeSemantics(mode);
  if (!semantics) {
    return '<p class="typed-capability-note is-blocked" data-duration-capability-note>Duration capability unavailable. Restart the local product.</p>';
  }
  if (semantics.feedback_tier === "evaluated") {
    return '<p class="typed-capability-note is-evaluated" data-duration-capability-note>Exact duration is evaluated-preview capable after server validation.</p>';
  }
  return `<p class="typed-capability-note is-draft-only" data-duration-capability-note>This duration mode remains draft only; full evaluation is unavailable (${escapeHtml(humanLabel(semantics.blocking_code))}).</p>`;
}

function roleSemantics(role) {
  const capability = typedEditCapability("set_stop_role");
  if (!capability || !TYPED_EDIT_VOCABULARIES.stop_roles.includes(role)) return null;
  if (capability.supported_evaluated_roles.includes(role)) {
    return { feedback_tier: "evaluated", blocking_code: null };
  }
  if (capability.draft_only_roles.includes(role)) {
    return { feedback_tier: "draft_only", blocking_code: capability.unsupported_role_blocking_code };
  }
  return null;
}

function roleOptionsHtml(selected = null) {
  const placeholder = '<option value="" disabled selected>Unavailable — choose a trip role</option>';
  const labels = Object.fromEntries(TYPED_EDIT_VOCABULARIES.stop_roles.map((role) => [
    role,
    `${humanLabel(role)} · ${roleSemantics(role)?.feedback_tier === "evaluated" ? "evaluated preview" : "draft only"}`,
  ]));
  return `${selected === null ? placeholder : ""}${optionsHtml(
    TYPED_EDIT_VOCABULARIES.stop_roles, selected, { labels },
  )}`;
}

function roleCapabilityNote(role) {
  const semantics = roleSemantics(role);
  if (role === "") {
    return '<p class="typed-capability-note is-blocked" data-role-capability-note>Choose a trip role explicitly. The accepted role is Unavailable and is not coerced to a category.</p>';
  }
  if (!semantics) {
    return '<p class="typed-capability-note is-blocked" data-role-capability-note>Role capability unavailable. Restart the local product.</p>';
  }
  if (semantics.feedback_tier === "evaluated") {
    return '<p class="typed-capability-note is-evaluated" data-role-capability-note>This trip-specific user-declared use is evaluated-preview capable. Place identity and categories stay unchanged; semantic fit and recommendation remain Unavailable.</p>';
  }
  return `<p class="typed-capability-note is-draft-only" data-role-capability-note>This structural role remains draft only; evaluated repair is unavailable (${escapeHtml(humanLabel(semantics.blocking_code))}).</p>`;
}

function typedCapabilityNote(type) {
  const capability = typedEditCapability(type);
  if (!capability) return '<p class="typed-capability-note is-blocked">Capability contract unavailable. Restart the local product.</p>';
  if (!capability.enabled) {
    return `<p class="typed-capability-note is-blocked">Unavailable: ${escapeHtml(humanLabel(capability.blocking_code))}.</p>`;
  }
  if (capability.feedback_tier === "evaluated"
      && capability.preview_executable === true
      && capability.evaluated_repair === true) {
    return '<p class="typed-capability-note is-evaluated">Evaluated-preview capable after server validation.</p>';
  }
  return `<p class="typed-capability-note is-draft-only">Draft only; full evaluation is not supported for this edit${capability.blocking_code ? ` (${escapeHtml(humanLabel(capability.blocking_code))})` : ""}.</p>`;
}

function typedDurationFields(prefix, mode = "exact") {
  const enabled = {
    preferred: ["exact", "preferred"].includes(mode),
    minimum: ["minimum", "range"].includes(mode),
    maximum: ["maximum", "range"].includes(mode),
  };
  const row = (field, label) => `<label data-duration-field="${field}" ${enabled[field] ? "" : "hidden"}>${label}
    <input name="${prefix}-${field}" type="number" min="15" max="480" step="15" ${enabled[field] ? "required" : "disabled"}>
  </label>`;
  return `${row("preferred", mode === "exact" ? "Exact visit minutes" : "Preferred visit minutes")}
    ${row("minimum", "Minimum visit minutes")}${row("maximum", "Maximum visit minutes")}`;
}

function typedImpactSummaryHtml() {
  if (state.typedImpact.status === "loading") return '<section class="typed-impact" role="status"><strong>Checking draft impact…</strong></section>';
  const response = state.typedImpact.response;
  if (!response) {
    return `<section class="typed-impact is-blocked" role="status"><strong>Draft impact unavailable</strong><p>${escapeHtml(state.typedImpact.error || "The server-owned impact preview has not loaded.")}</p></section>`;
  }
  const summary = response.summary;
  const rows = response.operations.map((row) => `<li><strong>${escapeHtml(operationDisplayName(row.type))}</strong>
    <span>${row.feedback_tier === "evaluated" ? "Evaluated-preview capable" : "Draft only; not evaluated"} · ${escapeHtml(row.changed_attributes.map(humanLabel).join(", ") || "No changed attributes reported")}</span></li>`).join("");
  const blockers = summary.blocking_codes.length
    ? `<p class="typed-impact-blocker"><strong>Preview blocked:</strong> ${escapeHtml(summary.blocking_codes.map(humanLabel).join(", "))}. Undo or revise the conflicting draft before evaluation.</p>`
    : "";
  return `<section class="typed-impact" aria-labelledby="typed-impact-title">
    <h3 id="typed-impact-title">Server-owned draft impact</h3>
    <p><strong>${summary.operation_count}</strong> operation${summary.operation_count === 1 ? "" : "s"} · ${summary.evaluated_executable_count} evaluated-preview capable · ${summary.draft_only_count} draft only.</p>
    ${rows ? `<ul>${rows}</ul>` : "<p>The active draft is empty.</p>"}
    ${blockers}
    <p class="truth-note">Not certified. The accepted parent remains unchanged.</p>
  </section>`;
}

function stopTypedEditorHtml(stop, properties) {
  const contract = typedEditContract();
  if (!contract) return '<p class="typed-editor-recovery">Typed editing is unavailable because the capability contract is invalid. Restart the local product.</p>';
  const vocab = contract.vocabularies;
  const currentDay = Number(properties?.day || stop.day || state.session.selected_day || 1);
  const currentOrder = Number.isInteger(properties?.sequence_index)
    ? Math.max(0, properties.sequence_index - 1)
    : 0;
  const currentDayStopCount = state.workspace.timeline.find(
    (day) => day.day === currentDay,
  )?.stops?.length || 0;
  const commitmentCapability = typedEditCapability("set_stop_commitment");
  const attributeCapability = typedEditCapability("set_attribute_constraint");
  return `<div class="typed-editor" data-editor-kind="stop">
    <header class="typed-editor-heading"><div><span class="eyebrow">Accepted-plan stop</span><h3>${escapeHtml(stop.name)}</h3></div><span class="status-pill warning">Draft, not accepted</span></header>
    <p>Each save appends exactly one server-validated operation. Draft-only edits cannot be sent to evaluated repair.</p>
    <details open><summary>Role, day, and order</summary>
      <form class="typed-operation-form" data-typed-operation="set_stop_role">
        <label>Itinerary role<select name="role" data-role-select required>${roleOptionsHtml(properties?.itinerary_role ?? null)}</select></label>
        <p class="field-help">This role records the stop's trip-specific user-declared use. It does not change the PlaceEntity identity or categories, infer semantic fit, create a recommendation, or affect the route and schedule for supported visit roles.</p>
        ${roleCapabilityNote(properties?.itinerary_role ?? "")}<button type="submit">Save role draft</button>
      </form>
      <form class="typed-operation-form" data-typed-operation="set_stop_day" data-current-day="${currentDay}">
        <label>Move to day<select name="day">${state.workspace.timeline.map((day) => `<option value="${day.day}" ${day.day === currentDay ? "selected" : ""}>Day ${day.day}</option>`).join("")}</select></label>
        ${typedCapabilityNote("set_stop_day")}<button type="submit">Save day draft</button>
      </form>
      <form class="typed-operation-form" data-typed-operation="set_stop_order">
        <input name="day" type="hidden" value="${currentDay}">
        <div class="typed-inline-fields"><span class="typed-readonly-value"><small>Accepted day</small><strong>Day ${currentDay}</strong></span>
          <label>Position in Day ${currentDay}<select name="sequence_position">${Array.from(
    { length: currentDayStopCount },
    (_, index) => `<option value="${index + 1}" ${index === currentOrder ? "selected" : ""}>${index + 1}${index === 0 ? " · first" : ""}</option>`,
  ).join("")}</select><small>Positions are shown starting at 1. This reorders only within the accepted day.</small></label></div>
        ${typedCapabilityNote("set_stop_order")}<button type="submit" ${currentDayStopCount > 1 ? "" : 'disabled title="This accepted day has only one stop, so no different order exists"'}>Save order draft</button>
      </form>
    </details>
    <details><summary>Duration and time window</summary>
      <form class="typed-operation-form" data-typed-operation="set_stop_duration">
        <label>Duration rule<select name="duration_mode" data-duration-mode>${durationModeOptionsHtml("exact")}</select></label>
        <div data-duration-fields>${typedDurationFields("duration", "exact")}</div>
        ${durationCapabilityNote("exact")}<button type="submit">Save duration draft</button>
      </form>
      <form class="typed-operation-form" data-typed-operation="set_stop_time_window">
        <div class="typed-inline-fields"><label>Earliest service admission<input name="earliest_arrival" type="time" aria-describedby="time-window-help"></label>
          <label>Latest departure after visit<input name="latest_departure" type="time" aria-describedby="time-window-help"></label></div>
        <p class="field-help" id="time-window-help">Provide one or both local-time bounds. Raw road arrival may be earlier; the visit waits until the earliest bound before service starts. Latest departure is checked after the visit. This trip-specific window is separate from place opening hours and is not a latest-start rule. Cross-midnight windows are not supported.</p>
        ${typedCapabilityNote("set_stop_time_window")}<button type="submit">Save time-window draft</button>
      </form>
    </details>
    <details><summary>Commitment and attribute constraints</summary>
      <form class="typed-operation-form" data-typed-operation="set_stop_commitment">
        <label>Commitment strength<select name="strength">${optionsHtml(vocab.commitment_strengths, "prefer_keep", { protectedValues: commitmentCapability.protected_strengths })}</select></label>
        <label>Scope lifetime<select name="scope_lifetime">${optionsHtml(vocab.scope_lifetimes, "current_repair_session")}</select></label>
        <p class="field-help">Must keep and Booked require a later explicit permission flow and cannot be selected here.</p>
        ${typedCapabilityNote("set_stop_commitment")}<button type="submit">Save commitment draft</button>
      </form>
      <form class="typed-operation-form" data-typed-operation="set_attribute_constraint">
        <label>Attribute<select name="attribute" data-attribute-kind>${optionsHtml(vocab.attributes, "existence")}</select></label>
        <div data-attribute-value></div>
        <label>Strength<select name="strength">${optionsHtml(vocab.commitment_strengths, "prefer_keep", { protectedValues: attributeCapability.protected_strengths })}</select></label>
        <label>Scope lifetime<select name="scope_lifetime">${optionsHtml(vocab.scope_lifetimes, "current_repair_session")}</select></label>
        ${typedCapabilityNote("set_attribute_constraint")}<button type="submit">Save attribute constraint</button>
      </form>
    </details>
    ${typedImpactSummaryHtml()}
  </div>`;
}

function routeTypedEditorHtml(routeProperties, routePlan) {
  const contract = typedEditContract();
  if (!contract) return '<p class="typed-editor-recovery">Typed route editing is unavailable because the capability contract is invalid. Restart the local product.</p>';
  const names = routeNamePair(routeProperties, routePlan);
  const issueCapability = typedEditCapability("report_route_issue");
  const preferenceCapability = typedEditCapability("change_route_preference");
  return `<div class="typed-editor" data-editor-kind="route">
    <header class="typed-editor-heading"><div><span class="eyebrow">Router-owned segment</span><h3>${escapeHtml(names.origin)} → ${escapeHtml(names.destination)}</h3></div><span class="status-pill success">Road evidence</span></header>
    <p>The route geometry is not freehand-editable. A report is a typed draft instruction and does not itself claim a closure or a feasible reroute.</p>
    <form class="typed-operation-form" data-typed-operation="report_route_issue">
      <label>Issue type<select name="issue_type">${optionsHtml(contract.vocabularies.route_issue_types, "suspected_closure")}</select></label>
      <label>Optional note<textarea name="note" maxlength="500" rows="3" placeholder="Describe what you observed; do not paste credentials or private data."></textarea></label>
      ${typedCapabilityNote("report_route_issue")}<button type="submit" ${issueCapability.enabled ? "" : "disabled"}>Add route issue draft</button>
    </form>
    <section class="disabled-capability" aria-label="Unavailable route preference editing">
      <h4>Route preference</h4><p>${escapeHtml(humanLabel(preferenceCapability.blocking_code))}. No simulated scenic, toll, highway, or mode change is applied.</p>
      <button type="button" disabled>Change route preference</button>
    </section>
    ${typedImpactSummaryHtml()}
  </div>`;
}

function typedEditSurfaceHtml() {
  const geography = activeGeography();
  if (state.mapInteraction.selectionKind === "route_leg") {
    const routePlan = acceptedPlan(geography);
    const routeFeature = selectedAcceptedRouteFeature(geography);
    return routeFeature ? routeTypedEditorHtml(routeFeature.properties, routePlan)
      : '<p class="typed-editor-recovery">Select a road-validated route leg from the accepted plan. Repair-preview route legs cannot be edited in place.</p>';
  }
  const stop = findSelectedStop();
  const parentTargets = new Set(state.workspace.draft_capabilities?.parent_targets || []);
  const feature = selectedAcceptedStopFeature(geography);
  if (stop && feature && parentTargets.has(stop.id)) {
    return stopTypedEditorHtml(stop, feature.properties || {});
  }
  return '<p class="typed-editor-recovery">Select an accepted-plan stop or a road-validated route leg. Repair-preview markers cannot be edited in place.</p>';
}

function renderAttributeValueEditor(form) {
  const container = form.querySelector("[data-attribute-value]");
  const attribute = form.elements.attribute.value;
  const dayOptions = state.workspace.timeline.map(
    (day) => `<option value="${day.day}">Day ${day.day}</option>`,
  ).join("");
  if (attribute === "existence") {
    container.innerHTML = '<label class="typed-checkbox"><input name="attribute_existence" type="checkbox" checked> Keep this stop in the itinerary</label>';
  } else if (attribute === "day") {
    container.innerHTML = `<label>Required day<select name="attribute_day">${dayOptions}</select></label>`;
  } else if (attribute === "duration") {
    container.innerHTML = `<label>Duration rule<select name="attribute_duration_mode" data-duration-mode>${optionsHtml(TYPED_EDIT_VOCABULARIES.duration_modes, "exact")}</select></label>
      <div data-duration-fields>${typedDurationFields("attribute-duration", "exact")}</div>`;
  } else if (attribute === "sequence_order") {
    container.innerHTML = `<div class="typed-inline-fields"><label>Day<select name="attribute_order_day">${dayOptions}</select></label>
      <label>Position in day<input name="attribute_sequence_index" type="number" min="0" max="63" value="0" required></label></div>`;
  } else if (attribute === "role") {
    container.innerHTML = `<label>Required role<select name="attribute_role">${optionsHtml(TYPED_EDIT_VOCABULARIES.stop_roles, "attraction")}</select></label>`;
  } else {
    container.innerHTML = '<p class="typed-editor-recovery">Unsupported attribute. No draft can be added.</p>';
  }
  bindDurationModeSelects(form);
}

function bindDurationModeSelects(root) {
  root.querySelectorAll("[data-duration-mode]").forEach((select) => {
    if (select.dataset.bound === "true") return;
    select.dataset.bound = "true";
    select.addEventListener("change", () => {
      const fields = select.parentElement.nextElementSibling;
      const prefix = select.name === "attribute_duration_mode" ? "attribute-duration" : "duration";
      fields.innerHTML = typedDurationFields(prefix, select.value);
      if (select.name === "duration_mode") {
        const note = select.form?.querySelector("[data-duration-capability-note]");
        if (note) note.outerHTML = durationCapabilityNote(select.value);
      }
    });
  });
}

function bindRoleSelects(root) {
  root.querySelectorAll("[data-role-select]").forEach((select) => {
    if (select.dataset.bound === "true") return;
    select.dataset.bound = "true";
    select.addEventListener("change", () => {
      const note = select.form?.querySelector("[data-role-capability-note]");
      if (note) note.outerHTML = roleCapabilityNote(select.value);
    });
  });
}

function durationPayload(form, modeName, prefix) {
  const mode = form.elements[modeName]?.value;
  if (!TYPED_EDIT_VOCABULARIES.duration_modes.includes(mode)) return null;
  const numberOrNull = (name) => {
    const input = form.elements[name];
    if (!input || input.disabled || input.value === "") return null;
    const value = Number(input.value);
    return Number.isInteger(value) && value >= 15 && value <= 480 ? value : Number.NaN;
  };
  const preferred = numberOrNull(`${prefix}-preferred`);
  const minimum = numberOrNull(`${prefix}-minimum`);
  const maximum = numberOrNull(`${prefix}-maximum`);
  if ([preferred, minimum, maximum].some(Number.isNaN)) return null;
  if (["exact", "preferred"].includes(mode) && preferred === null) return null;
  if (mode === "minimum" && minimum === null) return null;
  if (mode === "maximum" && maximum === null) return null;
  if (mode === "range" && (minimum === null || maximum === null || minimum > maximum)) return null;
  if (mode === "exact") {
    return {
      mode,
      preferred_minutes: preferred,
      minimum_minutes: preferred,
      maximum_minutes: preferred,
    };
  }
  return {
    mode,
    preferred_minutes: preferred,
    minimum_minutes: minimum,
    maximum_minutes: maximum,
  };
}

function typedOperationParameters(form, type) {
  if (type === "set_stop_role") {
    const role = form.elements.role.value;
    return TYPED_EDIT_VOCABULARIES.stop_roles.includes(role) ? { role } : null;
  }
  if (type === "set_stop_day") return { day: Number(form.elements.day.value) };
  if (type === "set_stop_order") return {
    day: Number(form.elements.day.value),
    sequence_index: Number(form.elements.sequence_position.value) - 1,
  };
  if (type === "set_stop_duration") {
    const duration = durationPayload(form, "duration_mode", "duration");
    return duration ? { duration } : null;
  }
  if (type === "set_stop_time_window") {
    const earliest = form.elements.earliest_arrival.value || null;
    const latest = form.elements.latest_departure.value || null;
    return canonicalTimeWindowParameters({ earliest_arrival: earliest, latest_departure: latest });
  }
  if (type === "set_stop_commitment") return {
    strength: form.elements.strength.value,
    scope_lifetime: form.elements.scope_lifetime.value,
  };
  if (type === "set_attribute_constraint") {
    const attribute = form.elements.attribute.value;
    let value = null;
    if (attribute === "existence") value = form.elements.attribute_existence.checked;
    if (attribute === "day") value = Number(form.elements.attribute_day.value);
    if (attribute === "duration") value = durationPayload(form, "attribute_duration_mode", "attribute-duration");
    if (attribute === "sequence_order") value = {
      day: Number(form.elements.attribute_order_day.value),
      sequence_index: Number(form.elements.attribute_sequence_index.value),
    };
    if (attribute === "role") value = form.elements.attribute_role.value;
    if (value === null) return null;
    return {
      attribute,
      strength: form.elements.strength.value,
      value,
      scope_lifetime: form.elements.scope_lifetime.value,
    };
  }
  if (type === "report_route_issue") {
    const routeLegId = state.mapInteraction.selectedRouteLeg?.route_leg_id;
    if (!routeLegId) return null;
    return {
      route_leg_id: routeLegId,
      issue_type: form.elements.issue_type.value,
      note: form.elements.note.value.trim() || null,
    };
  }
  return null;
}

async function submitTypedOperation(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const type = form.dataset.typedOperation;
  const capability = typedEditCapability(type);
  if (!capability?.enabled) {
    toast(`This edit is unavailable: ${humanLabel(capability?.blocking_code)}.`, true);
    return;
  }
  const geography = activeGeography();
  if (type === "report_route_issue" && !selectedAcceptedRouteFeature(geography)) {
    toast("Select a route leg from the accepted plan. No draft was added.", true);
    return;
  }
  if (type !== "report_route_issue" && !selectedAcceptedStopFeature(geography)) {
    toast("Select a stop from the accepted plan. No draft was added.", true);
    return;
  }
  const parameters = typedOperationParameters(form, type);
  if (!parameters) {
    toast("Review the typed values. No draft was added.", true);
    return;
  }
  if (type === "set_stop_day" && parameters.day === Number(form.dataset.currentDay)) {
    toast("Choose a different day. No draft was added.", true);
    return;
  }
  const target = type === "report_route_issue"
    ? state.mapInteraction.selectedRouteLeg?.route_leg_id
    : state.session.selected_stop_id;
  await addDraft(type, parameters, target, "map");
}

function bindTypedEditSurface() {
  const surface = $("#typed-edit-surface");
  if (!surface) return;
  surface.querySelectorAll(".typed-operation-form").forEach(
    (form) => form.addEventListener("submit", submitTypedOperation),
  );
  const attributeForm = surface.querySelector('[data-typed-operation="set_attribute_constraint"]');
  if (attributeForm) {
    renderAttributeValueEditor(attributeForm);
    attributeForm.elements.attribute.addEventListener("change", () => renderAttributeValueEditor(attributeForm));
  }
  bindDurationModeSelects(surface);
  bindRoleSelects(surface);
}

async function updateSelection(patch) {
  try {
    const stopExplicit = Object.hasOwn(patch, "selected_stop_id");
    const candidateExplicit = Object.hasOwn(patch, "selected_candidate_id");
    const selectedStopId = stopExplicit ? patch.selected_stop_id : state.session.selected_stop_id;
    let selectedCandidateId = candidateExplicit
      ? patch.selected_candidate_id
      : state.session.selected_candidate_id;
    if (stopExplicit && !candidateExplicit && selectedCandidateId) {
      const selectedCandidate = (state.workspace.draft_capabilities?.candidate_choices || [])
        .find((candidate) => candidate.candidate_id === selectedCandidateId);
      if (
        selectedCandidate?.replaces_stop_id
        && selectedCandidate.replaces_stop_id !== selectedStopId
      ) selectedCandidateId = null;
    }

    const payload = await api(`/api/sessions/${state.session.session_id}/selection`, {
      method: "POST",
      body: {
        expected_revision: state.session.revision,
        selected_day: Object.hasOwn(patch, "selected_day") ? patch.selected_day : state.session.selected_day,
        selected_stop_id: selectedStopId,
        selected_segment_id: Object.hasOwn(patch, "selected_segment_id") ? patch.selected_segment_id : state.session.selected_segment_id,
        selected_candidate_id: selectedCandidateId,
        selected_alternative_id: Object.hasOwn(patch, "selected_alternative_id") ? patch.selected_alternative_id : state.session.selected_alternative_id,
      },
    });
    state.session = payload.session;
    await loadDraftImpactPreview({ render: false });
    renderAll();
    return true;
  } catch (error) { toast(error.message, true); return false; }
}

function operationEnabled(type) {
  return state.workspace.draft_capabilities?.operations?.[type]?.enabled === true;
}

async function loadDraftImpactPreview({ render = true } = {}) {
  if (!state.session?.session_id) return null;
  state.typedImpact.status = "loading";
  state.typedImpact.error = null;
  try {
    const payload = await api(`/api/sessions/${state.session.session_id}/draft/impact-preview`, {
      method: "POST",
      body: { expected_revision: state.session.revision },
    });
    const normalized = normalizeDraftImpactPreview(payload);
    if (!normalized) throw new Error("invalid_draft_impact_preview");
    state.typedImpact.status = "ready";
    state.typedImpact.response = normalized;
    if (render) renderAll();
    return normalized;
  } catch (error) {
    state.typedImpact.status = "failed";
    state.typedImpact.response = null;
    state.typedImpact.error = error.message;
    if (render) renderAll();
    return null;
  }
}

function canRunEvaluatedPreview() {
  const response = state.typedImpact.response;
  return response?.session_revision === state.session?.revision
    && response?.summary?.can_run_evaluated_preview === true;
}

function selectContext(patch) {
  const request = updateSelection(patch);
  pendingSelection = request;
  return request.finally(() => {
    if (pendingSelection === request) pendingSelection = null;
  });
}

function resetPoiDiscovery({ keepReplacementTarget = true } = {}) {
  state.poiDiscovery.status = "idle";
  state.poiDiscovery.response = null;
  state.poiDiscovery.selectedCandidateId = null;
  state.poiDiscovery.showLayer = true;
  state.poiDiscovery.error = null;
  if (!keepReplacementTarget) state.poiDiscovery.replacementTargetId = null;
}

function prepareReplacementDiscovery() {
  const stop = findSelectedStop();
  const parentTargets = new Set(state.workspace.draft_capabilities?.parent_targets || []);
  if (!stop || !selectedAcceptedStopFeature(activeGeography()) || !parentTargets.has(stop.id)) {
    toast("Select an accepted-parent stop before starting replacement discovery.", true);
    return;
  }
  resetPoiDiscovery({ keepReplacementTarget: false });
  state.poiDiscovery.replacementTargetId = stop.id;
  toast(`Replacement target set to ${stop.name}. Now select an exact route leg touching this stop, then choose Find route-aware candidates.`);
  renderMap();
  requestAnimationFrame(() => $("[data-geo-route-leg]")?.focus());
}

async function fetchPoiCandidates(options = {}) {
  const preserveCandidateId = typeof options === "object"
    && !(options instanceof Event) ? options.preserveCandidateId || null : null;
  const quiet = typeof options === "object" && !(options instanceof Event) && options.quiet === true;
  const routeLeg = state.mapInteraction.selectedRouteLeg;
  const routeLegId = routeLeg?.route_leg_id;
  const day = Number(routeLeg?.day ?? state.session.selected_day);
  if (state.mapInteraction.mode !== "edit" || !routeLegId || !Number.isInteger(day) || day < 1) {
    toast("Switch to Edit mode and select one exact road-validated route leg first.", true);
    return false;
  }
  const expected = {
    sessionId: state.session.session_id,
    revision: state.session.revision,
    day,
    routeLegId,
    replacementTargetId: state.poiDiscovery.replacementTargetId,
  };
  state.poiDiscovery.status = "loading";
  state.poiDiscovery.response = null;
  state.poiDiscovery.selectedCandidateId = null;
  state.poiDiscovery.error = null;
  renderMap();
  const query = new URLSearchParams({
    expected_revision: String(expected.revision),
    day: String(day),
    route_leg_id: routeLegId,
    max_detour_minutes: String(Math.max(0, Math.min(240, state.poiDiscovery.maxDetourMinutes))),
    limit: String(Math.max(1, Math.min(10, state.poiDiscovery.limit))),
  });
  if (state.poiDiscovery.replacementTargetId) {
    query.set("replacement_target_id", state.poiDiscovery.replacementTargetId);
  }
  if (state.poiDiscovery.role) query.set("role", state.poiDiscovery.role);
  try {
    const payload = await api(`/api/sessions/${state.session.session_id}/map/poi-candidates?${query.toString()}`);
    const normalized = normalizePoiCandidateResponse(payload, expected);
    state.poiDiscovery.response = normalized;
    state.poiDiscovery.status = "ready";
    state.poiDiscovery.selectedCandidateId = normalized.candidates.some(
      (candidate) => candidate.candidate_id === preserveCandidateId,
    ) ? preserveCandidateId : null;
    state.poiDiscovery.showLayer = true;
    renderMap();
    if (!quiet) toast(`${normalized.candidates.length} route-context candidate${normalized.candidates.length === 1 ? "" : "s"} loaded. Inspect before creating any draft.`);
    return true;
  } catch (error) {
    state.poiDiscovery.status = "failed";
    state.poiDiscovery.response = null;
    state.poiDiscovery.selectedCandidateId = null;
    state.poiDiscovery.error = error.message;
    renderMap();
    toast(error.message, true);
    return false;
  }
}

function refreshPoiCandidateFilters(event) {
  event.preventDefault();
  state.poiDiscovery.role = $("#poi-role-filter")?.value || "";
  state.poiDiscovery.maxDetourMinutes = Math.max(
    0, Math.min(240, Number($("#poi-detour-filter")?.value) || 60),
  );
  state.poiDiscovery.limit = Math.max(
    1, Math.min(10, Number($("#poi-limit-filter")?.value) || 5),
  );
  void fetchPoiCandidates();
}

async function selectPoiCandidate(candidateId) {
  const candidate = (state.poiDiscovery.response?.candidates || []).find(
    (row) => row.candidate_id === candidateId,
  );
  if (!candidate) {
    toast("That candidate is not part of the current exact route context.", true);
    return false;
  }
  const selected = await selectContext({ selected_candidate_id: candidate.candidate_id });
  if (!selected) return false;
  return fetchPoiCandidates({ preserveCandidateId: candidate.candidate_id, quiet: true });
}

async function useSelectedCandidateReplacement() {
  const candidate = selectedPoiCandidate();
  const context = state.poiDiscovery.response?.context;
  const mapping = candidate?.registered_replacement;
  const parentTargets = new Set(state.workspace.draft_capabilities?.parent_targets || []);
  if (!candidate || !context || mapping?.draft_type !== "replace_nearby"
      || mapping.candidate_id !== candidate.candidate_id
      || mapping.candidate_id !== candidate.place?.place_id
      || mapping.target_stop_id !== context.replacement_target_id
      || !parentTargets.has(mapping.target_stop_id)
      || !operationEnabled("replace_nearby")) {
    toast("This candidate has no exact server-registered replacement mapping for the selected parent stop.", true);
    return false;
  }
  const selected = await selectContext({
    selected_day: context.day,
    selected_stop_id: mapping.target_stop_id,
    selected_segment_id: context.route_leg_id,
    selected_candidate_id: candidate.candidate_id,
  });
  if (!selected) return false;
  const added = await addDraft(
    "replace_nearby",
    { candidate_id: mapping.candidate_id },
    mapping.target_stop_id,
    "map",
  );
  return added;
}

async function closePoiDiscovery() {
  resetPoiDiscovery({ keepReplacementTarget: false });
  if (state.session.selected_candidate_id) {
    await selectContext({ selected_candidate_id: null });
  } else {
    renderMap();
  }
  requestAnimationFrame(() => $("#find-route-candidates")?.focus());
}

function setMapInteractionMode(mode) {
  if (!["select", "edit"].includes(mode)) return;
  state.mapInteraction.mode = mode;
  if (mode === "select" && state.mapInteraction.selectionKind === "exploratory") {
    cancelExploratoryPin(false);
  }
  if (state.route !== "/app/map" && state.route !== "/app/itinerary") {
    navigate("/app/map");
    return;
  }
  renderAll();
  requestAnimationFrame(() => $("[data-map-mode][aria-pressed='true']")?.focus());
}

function selectMapStop(selection) {
  const stopId = selection?.selected_stop_id || selection?.stop_id;
  if (!stopId) return Promise.resolve(false);
  resetPoiDiscovery({ keepReplacementTarget: false });
  state.mapInteraction.selectionKind = "stop";
  state.mapInteraction.selectedStopPlanId = selection.plan_id || null;
  state.mapInteraction.selectedRouteLeg = null;
  state.mapInteraction.selectedWaypointId = null;
  state.mapInteraction.exploratoryPin = null;
  state.mapInteraction.snapPreview = null;
  state.mapInteraction.snapState = "idle";
  return selectContext({
    selected_day: selection.selected_day ?? selection.day ?? state.session.selected_day,
    selected_stop_id: stopId,
    selected_segment_id: null,
  });
}

function selectRouteLeg(selection) {
  if (!selection?.route_leg_id) return Promise.resolve(false);
  resetPoiDiscovery({ keepReplacementTarget: true });
  state.mapInteraction.selectionKind = "route_leg";
  state.mapInteraction.selectedStopPlanId = null;
  state.mapInteraction.selectedRouteLeg = { ...selection };
  state.mapInteraction.selectedWaypointId = null;
  state.mapInteraction.exploratoryPin = null;
  state.mapInteraction.snapPreview = null;
  state.mapInteraction.snapState = "idle";
  return selectContext({
    selected_day: selection.day ?? state.session.selected_day,
    selected_stop_id: null,
    selected_segment_id: selection.route_leg_id,
  });
}

function selectCustomWaypoint(waypointId) {
  const waypoint = confirmedMapDraftState().waypoints.find(
    (candidate) => candidate.waypoint_id === waypointId,
  );
  if (!waypoint) return false;
  state.mapInteraction.selectionKind = "custom_waypoint";
  state.mapInteraction.selectedStopPlanId = null;
  state.mapInteraction.selectedRouteLeg = null;
  state.mapInteraction.selectedWaypointId = waypointId;
  state.mapInteraction.exploratoryPin = null;
  state.mapInteraction.snapPreview = null;
  state.mapInteraction.snapState = "idle";
  renderMap();
  requestAnimationFrame(() => $("#relocate-custom-waypoint")?.focus());
  return true;
}

function beginCustomWaypointRelocation(waypointId) {
  if (!mapEditCapability("relocate_custom_waypoint")) {
    toast("Custom waypoint relocation is unavailable for this workspace.", true);
    return false;
  }
  if (state.mapInteraction.mode !== "edit") {
    toast("Switch to Edit mode before relocating a custom waypoint.", true);
    return false;
  }
  const selected = selectCustomWaypoint(waypointId);
  if (selected) toast("Drag the selected custom waypoint marker to propose a new raw location.");
  return selected;
}

function beginCustomWaypointDragPreview(raw) {
  if (state.mapInteraction.mode !== "edit"
      || !mapEditCapability("relocate_custom_waypoint")) return;
  const waypoint = confirmedMapDraftState().waypoints.find(
    (candidate) => candidate.waypoint_id === raw?.waypoint_id,
  );
  if (!waypoint || !Number.isFinite(Number(raw.longitude))
      || !Number.isFinite(Number(raw.latitude))) return;
  state.mapInteraction.selectionKind = "exploratory";
  state.mapInteraction.selectedWaypointId = waypoint.waypoint_id;
  state.mapInteraction.exploratoryPin = {
    longitude: Number(raw.longitude),
    latitude: Number(raw.latitude),
    intent: waypoint.kind === "route_waypoint" ? "route_waypoint" : "possible_stop",
    targetWaypointId: waypoint.waypoint_id,
    retainedName: waypoint.name,
    retainedRole: waypoint.role,
    day: waypoint.day,
    insertion: waypoint.insertion,
    validation_status: "raw_coordinate_only",
  };
  state.mapInteraction.snapPreview = null;
  state.mapInteraction.snapState = "idle";
  renderMap();
  requestExploratorySnapPreview();
}

function createExploratoryPin(raw) {
  if (state.mapInteraction.mode !== "edit") {
    toast("Switch to Edit mode before placing an exploratory point.", true);
    return;
  }
  const longitude = Number(raw?.longitude);
  const latitude = Number(raw?.latitude);
  if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) {
    toast("The map did not return a valid raw coordinate.", true);
    return;
  }
  state.mapInteraction.selectionKind = "exploratory";
  state.mapInteraction.selectedStopPlanId = null;
  state.mapInteraction.selectedRouteLeg = null;
  state.mapInteraction.selectedWaypointId = null;
  state.mapInteraction.exploratoryPin = {
    longitude,
    latitude,
    intent: "explore_only",
    validation_status: "raw_coordinate_only",
  };
  state.mapInteraction.snapPreview = null;
  state.mapInteraction.snapState = "idle";
  renderMap();
  requestAnimationFrame(() => $("#exploratory-meaning")?.focus());
}

function cancelExploratoryPin(render = true) {
  const targetWaypointId = state.mapInteraction.exploratoryPin?.targetWaypointId || null;
  state.mapInteraction.exploratoryPin = null;
  state.mapInteraction.snapPreview = null;
  state.mapInteraction.snapState = "idle";
  state.mapInteraction.selectionKind = targetWaypointId ? "custom_waypoint" : null;
  state.mapInteraction.selectedWaypointId = targetWaypointId;
  state.mapInteraction.selectedStopPlanId = null;
  if (render) {
    renderMap();
    requestAnimationFrame(() => {
      const target = targetWaypointId
        ? $("#relocate-custom-waypoint")
        : $("[data-map-mode='edit']");
      target?.focus();
    });
  }
}

function snapOperationIntent(intent, pin = state.mapInteraction.exploratoryPin) {
  if (pin?.targetWaypointId) return "relocate_custom_waypoint";
  if (intent === "possible_stop") return "add_custom_waypoint";
  if (intent === "route_waypoint") return "add_route_waypoint";
  return "explore_only";
}

async function requestExploratorySnapPreview() {
  const pin = state.mapInteraction.exploratoryPin;
  if (!pin || state.mapInteraction.mode !== "edit") return;
  state.mapInteraction.snapState = "checking";
  renderMap();
  try {
    const preview = await api(`/api/sessions/${state.session.session_id}/map/snap-preview`, {
      method: "POST",
      body: {
        expected_revision: state.session.revision,
        longitude: pin.longitude,
        latitude: pin.latitude,
        operation_intent: snapOperationIntent(pin.intent),
        selected_day: pin.targetWaypointId ? pin.day : state.session.selected_day,
        selected_route_segment_id: pin.targetWaypointId
          ? pin.insertion?.route_leg_id || null
          : state.session.selected_segment_id,
        travel_mode: "driving",
        ...(pin.targetWaypointId ? { target_waypoint_id: pin.targetWaypointId } : {}),
      },
    });
    if (preview?.schema_version !== "map-snap-preview-v1" || preview.persisted !== false) {
      throw new Error("The road-access service returned an incompatible preview.");
    }
    state.mapInteraction.snapPreview = preview;
    state.mapInteraction.snapState = "checked";
    renderMap();
    if (["rejected", "unavailable"].includes(preview.validation_state)) {
      toast("The point did not produce an executable road-access preview.", true);
    } else {
      toast("Road access checked. This remains a non-persisted preview; no draft was appended.");
    }
  } catch (error) {
    state.mapInteraction.snapPreview = null;
    state.mapInteraction.snapState = "failed";
    renderMap();
    toast(error.message, true);
  }
}

function durationFromConfirmationForm() {
  const mode = $("#waypoint-duration-mode")?.value;
  const number = (selector) => {
    const field = $(selector);
    if (!field || field.disabled || field.value === "") return null;
    const value = Number(field.value);
    return Number.isInteger(value) && value >= 15 && value <= 480 ? value : Number.NaN;
  };
  const preferred = number("#waypoint-duration-preferred");
  const minimum = number("#waypoint-duration-minimum");
  const maximum = number("#waypoint-duration-maximum");
  if (mode === "exact" && Number.isFinite(preferred)) {
    return { mode, preferred_minutes: preferred, minimum_minutes: preferred, maximum_minutes: preferred };
  }
  if (mode === "preferred" && Number.isFinite(preferred)) {
    return { mode, preferred_minutes: preferred, minimum_minutes: null, maximum_minutes: null };
  }
  if (mode === "minimum" && Number.isFinite(minimum)) {
    return { mode, preferred_minutes: null, minimum_minutes: minimum, maximum_minutes: null };
  }
  if (mode === "maximum" && Number.isFinite(maximum)) {
    return { mode, preferred_minutes: null, minimum_minutes: null, maximum_minutes: maximum };
  }
  if (mode === "range" && Number.isFinite(minimum) && Number.isFinite(maximum)
      && minimum <= maximum) {
    return { mode, preferred_minutes: null, minimum_minutes: minimum, maximum_minutes: maximum };
  }
  return null;
}

async function confirmMapWaypoint(event) {
  event.preventDefault();
  const preview = state.mapInteraction.snapPreview;
  const pin = state.mapInteraction.exploratoryPin;
  if (!pin || !roadCheckedPreviewCanConfirm(preview)
      || !mapEditCapability(preview.operation_intent)) {
    toast("A current route-checked server preview is required.", true);
    return;
  }
  const warningAcknowledged = preview.confirmation_required
    ? $("#snap-warning-ack")?.checked === true : false;
  if (preview.confirmation_required && !warningAcknowledged) {
    toast("Review and acknowledge the snap warning before confirming.", true);
    return;
  }
  const relocation = preview.operation_intent === "relocate_custom_waypoint";
  const body = {
    expected_revision: state.session.revision,
    warning_acknowledged: warningAcknowledged,
  };
  if (!relocation) {
    const name = $("#waypoint-name")?.value.trim() || "";
    const routeWaypoint = preview.operation_intent === "add_route_waypoint";
    const duration = routeWaypoint ? null : durationFromConfirmationForm();
    if (!name || name.length > 120 || (!routeWaypoint && duration === null)) {
      toast("Enter a valid name and duration between 15 and 480 minutes.", true);
      return;
    }
    Object.assign(body, {
      name,
      role: routeWaypoint ? "route_waypoint" : $("#waypoint-role")?.value,
      duration,
    });
  }
  try {
    const payload = await api(`/api/sessions/${state.session.session_id}/map/snap-previews/${encodeURIComponent(preview.snap_preview_id)}/confirm`, {
      method: "POST",
      body,
    });
    state.session = payload.session;
    state.mapInteraction.selectionKind = "custom_waypoint";
    state.mapInteraction.selectedWaypointId = payload.operation?.target || pin.targetWaypointId || null;
    state.mapInteraction.exploratoryPin = null;
    state.mapInteraction.snapPreview = null;
    state.mapInteraction.snapState = "idle";
    toast(relocation
      ? "Custom waypoint relocated as one route-checked draft operation."
      : "Custom waypoint added as one route-checked draft operation.");
    renderAll();
    requestAnimationFrame(() => {
      $("#relocate-custom-waypoint")?.focus();
    });
  } catch (error) {
    toast(error.message, true);
  }
}

function refreshMapEditDialog() {
  $("#typed-edit-surface").innerHTML = typedEditSurfaceHtml();
  bindTypedEditSurface();
  const select = $("#destination-day");
  select.innerHTML = state.workspace.timeline.map((day) => `<option value="${day.day}">Day ${day.day}</option>`).join("");
  select.value = String(state.session.selected_day || 1);
  select.disabled = !operationEnabled("move_day") && !operationEnabled("add_candidate");
  const selected = findSelectedStop();
  const parentTargets = new Set(state.workspace.draft_capabilities?.parent_targets || []);
  const selectedParent = selected
    && selectedAcceptedStopFeature(activeGeography())
    && parentTargets.has(selected.id);
  const allCandidates = state.workspace.draft_capabilities?.candidate_choices || [];
  const compatibleCandidates = selectedParent
    ? allCandidates.filter((candidate) => candidate.replaces_stop_id === selected.id)
    : [];
  const candidates = compatibleCandidates.length ? compatibleCandidates : allCandidates;
  const candidateSelect = $("#registered-candidate");
  candidateSelect.innerHTML = candidates.map((candidate) => {
    const id = candidate.candidate_id || candidate.stop_id || candidate.id;
    return `<option value="${escapeHtml(id)}">${escapeHtml(candidate.label || candidate.name || id)}</option>`;
  }).join("");
  if (candidates.some((candidate) => (candidate.candidate_id || candidate.stop_id || candidate.id) === state.session.selected_candidate_id)) {
    candidateSelect.value = state.session.selected_candidate_id;
  }
  candidateSelect.disabled = candidates.length === 0;
  $("#draft-keep").disabled = !selectedParent || !operationEnabled("keep_stop");
  $("#draft-move").disabled = !selectedParent || !operationEnabled("move_day");
  $("#draft-lock").disabled = !selectedParent || !operationEnabled("lock_stop");
  $("#draft-flexible").disabled = !selectedParent || !operationEnabled("mark_flexible");
  $("#draft-feedback").disabled = !operationEnabled("route_feedback");
  $("#draft-replace").disabled = !selectedParent || compatibleCandidates.length === 0 || !operationEnabled("replace_nearby");
  $("#draft-add").disabled = candidates.length === 0 || !operationEnabled("add_candidate");
  $("#draft-undo").disabled = state.session.draft.length === 0;
  $("#draft-preview").disabled = state.session.draft.length === 0 || hasRouteCheckedOnlyDraft() || !canRunEvaluatedPreview();
  $("#draft-preview").title = state.session.draft.length && !canRunEvaluatedPreview()
    ? "This draft includes changes that are not enabled for evaluated repair" : "";
  if (selectedParent) {
    $("#edit-target").textContent = `Editing accepted-plan stop: ${selected.name}. Changes remain drafts until Preview repair finishes.`;
  } else if (state.session.selected_stop_id) {
    $("#edit-target").textContent = "The selected marker belongs to a repair preview and cannot be edited in place. Select an original accepted-plan stop, or add a registered candidate.";
  } else {
    $("#edit-target").textContent = "Select an original accepted-plan stop. Route feedback and Add candidate remain available.";
  }
}

async function openMapEdit() {
  if (pendingSelection) await pendingSelection;
  refreshMapEditDialog();
  $("#map-edit-dialog").showModal();
}

async function addDraft(type, parameters, explicitTarget, source = "map") {
  if (pendingSelection) await pendingSelection;
  const target = explicitTarget || state.session.selected_stop_id;
  if (!target && type !== "route_feedback") return toast("Select a stop before adding this draft.", true);
  try {
    const payload = await api(`/api/sessions/${state.session.session_id}/draft/operations`, {
      method: "POST",
      body: { expected_revision: state.session.revision, type, target: target || "selected_route", parameters, source },
    });
    state.session = payload.session;
    resetPoiDiscovery({ keepReplacementTarget: false });
    await loadDraftImpactPreview({ render: false });
    toast("Draft operation added. The parent plan is unchanged.");
    renderAll();
    if ($("#map-edit-dialog").open) refreshMapEditDialog();
    return true;
  } catch (error) { toast(error.message, true); return false; }
}

async function addCopilotDraft(intent) {
  const target = intent.target_stop_id || state.session.selected_stop_id;
  if (intent.type === "route_feedback") {
    return addDraft(intent.type, { preference: intent.preference || "reduce_contextual_risk" }, "selected_route", "copilot");
  }
  if (intent.type === "move_day") {
    return addDraft(intent.type, { day: intent.destination_day }, target, "copilot");
  }
  if (intent.type === "replace_nearby") {
    return addDraft(intent.type, { candidate_id: intent.candidate_id }, target, "copilot");
  }
  if (intent.type === "add_candidate") {
    return addDraft(intent.type, { day: intent.destination_day || state.session.selected_day }, intent.candidate_id, "copilot");
  }
  if (["keep_stop", "lock_stop", "mark_flexible"].includes(intent.type)) {
    return addDraft(intent.type, {}, target, "copilot");
  }
  toast("This Copilot suggestion is advisory and cannot be added as a typed draft.", true);
}

async function undoDraft() {
  try {
    const payload = await api(`/api/sessions/${state.session.session_id}/draft/undo`, {
      method: "POST", body: { expected_revision: state.session.revision },
    });
    state.session = payload.session;
    resetPoiDiscovery({ keepReplacementTarget: false });
    if (state.mapInteraction.selectedWaypointId && !selectedDraftWaypoint()) {
      state.mapInteraction.selectionKind = null;
      state.mapInteraction.selectedWaypointId = null;
    }
    await loadDraftImpactPreview({ render: false });
    toast("Latest draft operation removed.");
    renderAll();
    if ($("#map-edit-dialog").open) refreshMapEditDialog();
  } catch (error) { toast(error.message, true); }
}

async function previewDraft() {
  if (hasRouteCheckedOnlyDraft()) {
    toast("This arbitrary waypoint draft is route-checked, but evaluated repair is not enabled yet.", true);
    return;
  }
  if (!canRunEvaluatedPreview()) {
    toast("This draft contains changes that are not enabled for evaluated repair. Inspect the server-owned draft impact, then undo or revise those changes.", true);
    return;
  }
  try {
    const impact = state.typedImpact.response;
    const expected = {
      session_id: state.session.session_id,
      trip_id: state.session.trip_id,
      run_id: state.session.run_id,
      revision: state.session.revision,
      accepted_plan_id: state.session.accepted_plan_id,
      parent_plan_content_hash: impact?.parent_plan_content_hash,
      draft: structuredClone(state.session.draft),
      permission_decisions: structuredClone(state.session.permission_decisions),
      conversation_id: state.session.conversation_id,
    };
    const payload = await api(`/api/sessions/${state.session.session_id}/preview`, {
      method: "POST", body: { expected_revision: state.session.revision },
    });
    const normalized = await normalizeEvaluatedPreviewResponse(payload, expected);
    if (!normalized) throw new Error(apiErrorMessage("invalid_evaluated_preview"));
    state.session = normalized.session;
    await loadDraftImpactPreview({ render: false });
    $("#map-edit-dialog").close();
    toast(normalized.proposal.message, normalized.proposal.state !== "eligible");
    renderAll();
  } catch (error) { toast(error.message, true); }
}

function setupDockResize() {
  const dock = $("#copilot-dock");
  const handle = $("#dock-resizer");
  const setWidth = (width) => {
    const bounded = Math.max(320, Math.min(560, width));
    dock.style.width = `${bounded}px`;
    handle.setAttribute("aria-valuenow", String(Math.round(bounded)));
  };
  const resize = (clientX) => setWidth(innerWidth - clientX);
  handle.addEventListener("pointerdown", (event) => {
    handle.setPointerCapture(event.pointerId);
    const move = (next) => resize(next.clientX);
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", () => handle.removeEventListener("pointermove", move), { once: true });
  });
  handle.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      const current = dock.getBoundingClientRect().width;
      setWidth(current + (event.key === "ArrowLeft" ? 20 : -20));
    }
  });
}

function updateActiveNav() {
  $$("[data-route]").forEach((button) => {
    const active = normalizeRoute(button.dataset.route) === state.route;
    button.classList.toggle("is-active", active);
    if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  });
}

function findSelectedStop() {
  for (const day of state.workspace.timeline) {
    const found = day.stops.find((stop) => stop.id === state.session.selected_stop_id);
    if (found) return found;
  }
  const geography = activeGeography();
  const selectedPlan = geography?.plans?.find(
    (plan) => plan.plan_id === state.mapInteraction.selectedStopPlanId,
  );
  const searchPlans = selectedPlan
    ? [selectedPlan, ...(geography?.plans || []).filter((plan) => plan !== selectedPlan)]
    : geography?.plans || [];
  const feature = searchPlans
    .flatMap((plan) => plan?.stops?.features || [])
    .find((candidate) => candidate.properties?.stop_id === state.session.selected_stop_id);
  if (feature) {
    return {
      id: feature.properties.stop_id,
      name: feature.properties.name || feature.properties.stop_id,
      day: feature.properties.day,
    };
  }
  return null;
}

function metricCell(cell, unit) {
  if (cell.state === "unavailable" || cell.value === null || cell.value === undefined) return "Unavailable";
  return `${escapeHtml(formatMetric(cell.value))}${unit ? ` ${escapeHtml(unit)}` : ""}`;
}

function objectDl(value) {
  return Object.entries(value || {}).map(([key, item]) => `<dt>${escapeHtml(key.replaceAll("_", " "))}</dt><dd>${escapeHtml(Array.isArray(item) ? item.join(", ") : item ?? "Unavailable")}</dd>`).join("");
}

function formatMetric(value) {
  if (value === null || value === undefined) return "Unavailable";
  return typeof value === "number" ? Number(value.toFixed(2)).toString() : String(value);
}

function toast(message, error = false) {
  const item = document.createElement("div");
  item.className = `toast${error ? " error" : ""}`;
  item.textContent = message;
  $("#toast-region").append(item);
  setTimeout(() => item.remove(), 5000);
}

boot();
