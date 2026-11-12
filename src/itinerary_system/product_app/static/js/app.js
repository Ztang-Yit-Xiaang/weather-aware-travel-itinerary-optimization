import {
  disposeGeographicMap,
  normalizePoiCandidateResponse,
  reduceConfirmedMapDraftOperations,
  renderGeographicMap,
} from "./map.js?v=20260817-e2e3";
import {
  copilotOperationForIntent,
  createCopilotController,
  sameCopilotDraftOperation,
} from "./copilot.js?v=20260904-hci1";
import {
  disposeCompareWorkspace,
  renderCompareWorkspace,
  resolveCompareSelection,
} from "./compare.js?v=20260813-w5-accept1";

import { createMutationRecovery } from "./mutation-recovery.js";
import { interestPlacesInRegion, normalizeInterestResponse } from "./interest-regions.js";

import { createWorkspaceStorage, restoreInterestPreferences } from "./workspace-storage.js";
import { productFetch as fetch, currentAppPath, appRouteUrl, ensureHostedLogin, signOutHosted } from "./deployment-client.js";

const state = {
  clientBuildId: document.querySelector('meta[name="product-build-id"]')?.content || "",
  buildRecovery: null,
  health: null,
  mapConfig: null,
  workspace: null,
  session: null,
  token: null,
  undoPending: false,
  workspaceConflict: null,
  workspaceSyncPending: false,
  sessionRecoveryNotice: null,
  mutationRecovery: { status: "idle", blocked: false, message: "" },
  registry: null,
  route: normalizeRoute(currentAppPath()),
  tripPreviewPlanId: null,
  acknowledgedPreviewPlanId: null,
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
  interestRegions: {
    enabled: false, status: "idle", response: null, error: null,
    naturePreference: 0, massLevel: 0.8, selectedId: null, generation: 0, controller: null,
    placeSearch: "", placeScope: "all", radiusKm: 5, focusStopId: null, searchArea: "visitor_area",
  },
  typedImpact: {
    request: null,
    status: "idle",
    response: null,
    error: null,
  },
  previewRuntime: {
    epoch: 0,
    status: "PARENT_RENDERED",
    requestId: null,
  },
};
let pendingSelection = null;
let workspaceChannel = null;
let workspaceRefreshQueued = false;
let workspaceRefreshTimer = null;
let workspaceNoticeEpoch = 0;

const SESSION_STORAGE_KEY = "itinerary-repair-copilot.session.v1";
const PENDING_MUTATION_KEY = "itinerary-repair-copilot.pending-mutation.v1";
const PENDING_INPUT_KEY = "itinerary-repair-copilot.pending-input.v1";

const workspaceStorage = createWorkspaceStorage({
  persistent: localStorage, tab: sessionStorage, sessionKey: SESSION_STORAGE_KEY,
  scopedKeys: [PENDING_MUTATION_KEY, PENDING_INPUT_KEY],
});
let interestRefreshRunning = false;
let interestRefreshTimer = null;
const interestPreferencesKey = () => `itinerary-repair-copilot.interest.v1:${state.session.trip_id}`;
function saveInterestPreferences() {
  const {enabled, radiusKm, naturePreference, massLevel, searchArea} = state.interestRegions;
  try { workspaceStorage.setItem(interestPreferencesKey(), JSON.stringify({enabled, radiusKm, naturePreference, massLevel, searchArea})); }
  catch { toast("Interest settings could not be saved for reopening.", true); }
}
function scheduleInterestRefresh() {
  if (!state.interestRegions.enabled || state.interestRegions.status !== "idle"
      || interestRefreshRunning || pendingSelection || state.route !== "/app/map") return;
  clearTimeout(interestRefreshTimer);
  interestRefreshTimer = setTimeout(() => { void fetchInterestRegions(); }, 0);
}

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const copilot = createCopilotController({
  getState: () => state,
  api,
  streamMessage: streamCopilotMessage,
  selectContext,
  addDraft: addCopilotDraft,
  previewDraft,
  getPreviewState: copilotPreviewState,
  getSelectedStop: findSelectedStop,
  navigate,
  toast,
  renderAll,
  syncPreviewRuntime: syncPreviewRuntimeFromSession,
  onIdle: flushQueuedWorkspaceRefresh,
});

const mutationRecovery = createMutationRecovery({
  api: (path, options = {}) => api(path, {...options, pendingMutation: true}),
  storage: {
    getItem: key => workspaceStorage.getItem(key),
    setItem: (key, value) => workspaceStorage.setItem(key, value),
    removeItem: key => workspaceStorage.removeItem(key),
  },
  key: PENDING_MUTATION_KEY,
  getSession: () => state.session,
  onChange: value => {
    state.mutationRecovery = value;
    if (value.blocked) savePendingMutationInput();
    if (state.session && !$("#app-shell").hidden) renderAll();
  },
});

function savePendingMutationInput() {
  if (!state.session) return;
  try {
    workspaceStorage.setItem(PENDING_INPUT_KEY, JSON.stringify({
      session_id: state.session.session_id, text: $("#copilot-input")?.value || "",
    }));
  } catch {
    toast("This unsent message could not be saved for reload.", true);
  }
}

function restorePendingMutationInput() {
  try {
    const saved = JSON.parse(workspaceStorage.getItem(PENDING_INPUT_KEY) || "null");
    if (saved?.session_id === state.session.session_id && typeof saved.text === "string" && saved.text.length <= 20_000) {
      $("#copilot-input").value = saved.text;
    }
  } catch {
    toast("The unsent message could not be restored.", true);
  }
}

function restoreMutationControls() {
  $$("[data-pending-mutation-disabled]").forEach(control => {
    control.disabled = false;
    delete control.dataset.pendingMutationDisabled;
  });
}

function renderMutationRecovery() {
  let banner = $("#mutation-recovery-banner");
  if (!banner) {
    banner = document.createElement("section");
    banner.id = "mutation-recovery-banner";
    banner.className = "runtime-banner";
    banner.setAttribute("role", "status");
    $("#runtime-banner").after(banner);
    new ResizeObserver(updateMutationRecoveryLayout).observe(banner);
  }
  const recovery = state.mutationRecovery;
  const blocked = recovery.blocked || Boolean(state.workspaceConflict) || state.workspaceSyncPending;
  banner.hidden = !blocked;
  $("#app-shell").classList.toggle("has-mutation-recovery", blocked);
  updateMutationRecoveryLayout();
  if (!blocked) return;
  const checking = recovery.status === "checking";
  if (!recovery.blocked && (state.workspaceConflict || state.workspaceSyncPending)) {
    const conflict = state.workspaceConflict || { refreshing: true, message: "The trip changed in another tab. Waiting for the current action to finish before loading the saved workspace." };
    banner.innerHTML = `<strong>${conflict.refreshing ? "Refreshing current workspace" : "Workspace changed"}</strong>
      <span>${escapeHtml(conflict.message)}</span>
      <button type="button" class="secondary" data-refresh-current-workspace ${conflict.refreshing ? "disabled" : ""}>Refresh current workspace</button>`;
    banner.querySelector("[data-refresh-current-workspace]").addEventListener("click", () => refreshStaleWorkspace());
  } else if (recovery.unreadable) {
    banner.innerHTML = `<strong>${checking ? "Refreshing protected workspace" : "Saved action unavailable"}</strong>
      <span>${escapeHtml(recovery.message)} Already saved changes will be kept. A protected refresh blocks unfinished edits based on the previous workspace version; their original outcome may remain unknown.</span>
      <button type="button" class="secondary" data-repair-pending-mutation ${checking ? "disabled" : ""}>Refresh protected workspace</button>`;
    banner.querySelector("[data-repair-pending-mutation]").addEventListener("click", () => recoverPendingMutation(false, false, true));
  } else {
    banner.innerHTML = `<strong>${checking ? "Confirming saved action" : "Action awaiting confirmation"}</strong>
      <span>${escapeHtml(recovery.message || "Check the saved action before making another edit. Your message has not been sent.")} Cancellation leaves any already saved changes in place.</span>
      <button type="button" class="secondary" data-check-pending-mutation ${checking ? "disabled" : ""}>Check saved action</button>
      <button type="button" class="secondary" data-retry-pending-mutation ${checking ? "disabled" : ""}>Retry saved action</button>
      <button type="button" class="secondary" data-cancel-pending-mutation ${checking ? "disabled" : ""}>Cancel pending action</button>`;
    banner.querySelector("[data-check-pending-mutation]").addEventListener("click", () => recoverPendingMutation(false));
    banner.querySelector("[data-retry-pending-mutation]").addEventListener("click", () => recoverPendingMutation(true));
    banner.querySelector("[data-cancel-pending-mutation]").addEventListener("click", () => recoverPendingMutation(false, true));
  }
  updateMutationRecoveryLayout();
  $$(".flow button, #map-edit-dialog button, #typed-edit-surface button, .copilot-proposal-actions button, [data-accept-action], #inline-undo, #inline-preview, #accept-repair, #copilot-submit[data-mode=send]")
    .filter(control => !control.disabled).forEach(control => {
      control.dataset.pendingMutationDisabled = "true";
      control.disabled = true;
    });
}

function updateMutationRecoveryLayout() {
  const banner = $("#mutation-recovery-banner");
  if (banner && !banner.hidden) {
    $("#app-shell").style.setProperty("--mutation-recovery-bottom", `${banner.getBoundingClientRect().bottom}px`);
  } else {
    $("#app-shell").style.removeProperty("--mutation-recovery-bottom");
  }
}

async function recoverPendingMutation(retry = false, cancel = false, repair = false) {
  try {
    const payload = await (repair ? mutationRecovery.repair() : cancel ? mutationRecovery.cancel() : retry ? mutationRecovery.retry() : mutationRecovery.check());
    installDraftMutationSession(payload.session, payload);
    syncPreviewRuntimeFromSession();
    state.tripPreviewPlanId = null;
    state.acknowledgedPreviewPlanId = null;
    resetPoiDiscovery({ keepReplacementTarget: false });
    if (state.mapInteraction.selectedWaypointId && !selectedDraftWaypoint()) {
      state.mapInteraction.selectionKind = null;
      state.mapInteraction.selectedWaypointId = null;
    }
    savePendingMutationInput();
    const impact = loadDraftImpactPreview({ render: false });
    renderAll();
    toast(payload.recovery_outcome === "refreshed"
      ? "Protected workspace refreshed. Saved changes were kept and unfinished edits from the previous workspace version are blocked. Review the current draft before continuing."
      : payload.recovery_outcome === "applied"
      ? (state.session.draft?.length && !state.session.proposal
        ? "Saved action confirmed. Your draft is staged; select Preview repair to evaluate it and update the map."
        : "Saved action confirmed. The current workspace has been refreshed.")
      : "That action was not applied. Saved draft changes were kept; review the refreshed workspace before continuing.");
    await impact;
    renderAll();
    await copilot.restore();
    return true;
  } catch (error) {
    toast(error.message, true);
    renderAll();
    return false;
  }
}

function flushQueuedWorkspaceRefresh() {
  if (!workspaceRefreshQueued || workspaceRefreshTimer !== null) return;
  workspaceRefreshTimer = setTimeout(async () => {
    workspaceRefreshTimer = null;
    if (!state.session || state.mutationRecovery.blocked || state.workspaceConflict
        || state.undoPending || pendingSelection || copilot.isBusy()
        || state.previewRuntime.status === "PREVIEW_RUNNING") return;
    workspaceRefreshQueued = false;
    await refreshStaleWorkspace(null, true);
  }, 0);
}

function requestWorkspaceRefresh() {
  workspaceRefreshQueued = true;
  flushQueuedWorkspaceRefresh();
}

function bindWorkspaceSynchronization() {
  const receive = ({ data }) => {
    if (!data || typeof data !== "object" || Array.isArray(data)
        || Object.keys(data).sort().join("|") !== "conversation_id|revision|schema_version|session_id"
        || data.schema_version !== "product-workspace-invalidation-v1"
        || data.session_id !== state.session?.session_id
        || !Number.isSafeInteger(data.revision) || data.revision < state.session.revision
        || !(data.conversation_id === null || typeof data.conversation_id === "string")) return;
    if (data.revision > state.session.revision || data.conversation_id !== state.session.conversation_id) {
      workspaceNoticeEpoch += 1;
      state.workspaceSyncPending = true;
      renderAll();
      requestWorkspaceRefresh();
    }
  };
  try {
    if (typeof BroadcastChannel === "function") {
      workspaceChannel = new BroadcastChannel("itinerary-repair-copilot.workspace.v1");
      workspaceChannel.addEventListener("message", receive);
    }
  } catch { workspaceChannel = null; }
  window.addEventListener("focus", requestWorkspaceRefresh);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") requestWorkspaceRefresh();
  });
}

function announceWorkspaceChange(payload) {
  const session = payload?.session;
  if (!session || typeof session.session_id !== "string" || !Number.isSafeInteger(session.revision)) return;
  try {
    workspaceChannel?.postMessage({ schema_version: "product-workspace-invalidation-v1",
      session_id: session.session_id, revision: session.revision, conversation_id: session.conversation_id ?? null });
  } catch { workspaceChannel = null; }
}

async function refreshStaleWorkspace(error = null, automatic = false) {
  if (error && (error.status !== 409 || error.code !== "stale_session_revision")) return false;
  if (state.mutationRecovery.blocked || !state.session) return false;
  if (state.workspaceConflict?.refreshing) return true;
  automatic ||= state.workspaceConflict?.automatic === true;
  const noticeEpoch = workspaceNoticeEpoch;
  const origin = { ...state.session };
  const invalidatePreview = () => {
    state.previewRuntime = { epoch: state.previewRuntime.epoch + 1, status: "PREVIEW_FAILED", requestId: null, message: "The saved workspace needs to be checked before this preview can be used." };
    state.tripPreviewPlanId = null;
    state.acknowledgedPreviewPlanId = null;
  };
  state.workspaceConflict = { refreshing: true, automatic, message: automatic
    ? "Checking the saved workspace for changes. Your message draft will be kept."
    : "The trip changed in another action. Reading the saved workspace; your rejected action will not be retried automatically." };
  if (!automatic) { state.workspaceSyncPending = true; invalidatePreview(); }
  if (!automatic || state.workspaceSyncPending) {
    renderAll();
    copilot.previewStateChanged();
  }
  const controller = new AbortController();
  let deadline;
  try {
    const expired = new Promise((_resolve, reject) => {
      deadline = setTimeout(() => {
        reject(new Error("The saved workspace did not respond in time."));
        controller.abort();
      }, 20000);
    });
    const payload = await Promise.race([api(`/api/sessions/${encodeURIComponent(origin.session_id)}`, { signal: controller.signal }), expired]);
    clearTimeout(deadline);
    const owns = session => session && ["session_id", "trip_id", "run_id"].every(key => session[key] === origin[key]);
    if (!owns(state.session) || !owns(payload?.session)
        || !Number.isSafeInteger(payload.session.revision) || payload.session.revision < state.session.revision
        || !payload.workspace || !payload.registry) throw new Error("The refreshed workspace did not match the current trip.");
    if (workspaceNoticeEpoch !== noticeEpoch) {
      state.workspaceConflict = null;
      renderAll();
      return true;
    }
    if (automatic && !state.workspaceSyncPending && ["revision", "accepted_plan_id", "conversation_id"].every(key => payload.session[key] === state.session[key])) {
      state.workspaceConflict = null;
      state.workspaceSyncPending = false;
      return true;
    }
    if (automatic) invalidatePreview();
    installDraftMutationSession(payload.session, payload);
    syncPreviewRuntimeFromSession();
    resetPoiDiscovery({ keepReplacementTarget: false });
    if (state.mapInteraction.selectedWaypointId && !selectedDraftWaypoint()) {
      state.mapInteraction.selectionKind = null;
      state.mapInteraction.selectedWaypointId = null;
    }
    state.workspaceConflict = null;
    state.workspaceSyncPending = false;
    const impact = loadDraftImpactPreview({ render: false });
    renderAll();
    toast(automatic ? "Workspace updated from another action. Review the current draft before continuing."
      : "Workspace refreshed. The rejected action was not applied. Review the current draft before trying again.");
    await impact;
    renderAll();
    await copilot.restore();
  } catch (refreshError) {
    if (state.session?.session_id === origin.session_id) {
      invalidatePreview();
      state.workspaceSyncPending = true;
      state.workspaceConflict = { refreshing: false, automatic, message: "The current workspace could not be loaded. Check the connection and refresh before making another change." };
      renderAll();
      toast(state.workspaceConflict.message, true);
    }
  } finally {
    clearTimeout(deadline);
  }
  return true;
}

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
    const combinable = state.workspace?.draft_capabilities?.executable_combinations
      ?.find((combination) => combination.operations.includes(name))?.operations || [name];
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
    const optimizerMappingUnsupported = name === "set_stop_day"
      && capability.blocking_code === "unsupported_optimizer_mapping";
    const acceptedReadOnly = capability.blocking_code === "accepted_plan_requires_new_repair_baseline";
    const expectedSemantics = acceptedReadOnly
      ? [false, capability.feedback_tier, false, false, "accepted_plan_requires_new_repair_baseline"]
      : optimizerMappingUnsupported
      ? [false, "draft_only", false, false, "unsupported_optimizer_mapping"]
      : ["set_stop_day", "set_stop_order", "set_stop_time_window"].includes(name)
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
          || !exactStringArray(capability.combinable_operation_types, combinable))) return null;
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
          || !exactStringArray(capability.combinable_operation_types, combinable))) return null;
  }
  return value;
}

function draftCombinationBlocker(draft) {
  const directTypes = new Set([
    "set_stop_order", "set_stop_role", "set_stop_duration", "set_stop_time_window",
  ]);
  const present = new Set(draft.map(operation => operation.type).filter(type => directTypes.has(type)));
  const modes = new Set(draft.filter(operation => operation.type === "set_stop_duration")
    .map(operation => operation.parameters?.duration?.mode).filter(Boolean));
  const mixed = present.size > 0 && (present.size !== 1 || draft.some(operation => !present.has(operation.type)));
  const combination = state.workspace?.draft_capabilities?.executable_combinations?.some(
    (row) => draft.length <= row.cardinality && draft.every((operation) => row.operations.includes(operation.type)),
  );
  return (mixed && (!combination || [...modes].some((mode) => mode !== "exact")))
    || (modes.has("exact") && modes.size > 1)
    ? "draft_evaluated_operation_combination_unsupported" : null;
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
    const additionReadiness = row.type === "add_candidate"
      ? legacyCapability?.candidate_day_readiness?.find(item => item.candidate_id === row.target
        && item.day === draftOperation?.parameters?.day
        && item.parent_plan_id === value.parent_plan_id
        && item.parent_plan_content_hash === value.parent_plan_content_hash) : null;
    const additionSemantics = additionReadiness ? {
      feedback_tier: additionReadiness.preview_executable ? "evaluated" : "draft_only",
      preview_executable: additionReadiness.preview_executable,
      evaluated_repair: additionReadiness.preview_executable,
      blocking_codes: additionReadiness.preview_executable ? [] : [additionReadiness.status],
    } : null;
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
    const expected = additionSemantics || typedDurationSemantics || typedRoleSemantics || (typedCapability
      && !["set_stop_duration", "set_stop_role"].includes(row.type)
      ? {
        feedback_tier: typedCapability.feedback_tier,
        preview_executable: typedCapability.preview_executable,
        evaluated_repair: typedCapability.evaluated_repair,
        blocking_codes: typedCapability.blocking_code ? [typedCapability.blocking_code] : [],
      }
      : legacyCapability?.enabled === true && legacyCapability.preview_executable === true
        ? { feedback_tier: "evaluated", preview_executable: true, evaluated_repair: true, blocking_codes: [] }
        : legacyCapability?.enabled === false && legacyCapability.status === "unsupported_optimizer_mapping"
          ? { feedback_tier: "draft_only", preview_executable: false, evaluated_repair: false,
            blocking_codes: [legacyCapability.status] }
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
  const combinationBlocker = draftCombinationBlocker(draft);
  if (combinationBlocker) blockingCodes.push(combinationBlocker);
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
  return canonicalJson(left) === canonicalJson(right);
}

function previewStateForSession(session = state.session) {
  const proposal = session?.proposal;
  if (proposal?.schema_version === "draft-preview-v1" && proposal.state === "eligible") {
    return "PREVIEW_DATA_READY";
  }
  if (proposal?.schema_version === "draft-preview-v1" && proposal.state === "ineligible") {
    return "PREVIEW_INELIGIBLE";
  }
  if (proposal?.schema_version === "draft-preview-v1") {
    if (proposal.state === "blocked") return "PREVIEW_BLOCKED";
    if (proposal.state === "not_evaluated") return "PREVIEW_NOT_EVALUATED";
    if (proposal.state === "failed") return "PREVIEW_FAILED";
  }
  return "PARENT_RENDERED";
}

function syncPreviewRuntimeFromSession({ invalidate = true } = {}) {
  if (invalidate) state.previewRuntime.epoch += 1;
  state.previewRuntime.status = previewStateForSession();
  state.previewRuntime.requestId = state.session?.proposal?.compiled_request?.source_request_id || null;
  state.previewRuntime.message = null;
}

function previewRequestIsCurrent(epoch, expected) {
  return epoch === state.previewRuntime.epoch
    && state.session?.session_id === expected.session_id
    && state.session?.trip_id === expected.trip_id
    && state.session?.run_id === expected.run_id
    && state.session?.revision === expected.revision
    && state.session?.accepted_plan_id === expected.accepted_plan_id
    && sameJson(state.session?.draft, expected.draft)
    && sameJson(state.session?.permission_decisions, expected.permission_decisions)
    && state.session?.conversation_id === expected.conversation_id;
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
  "visit_minutes", "visit_duration_minutes", "departure_minute", "latest_departure_overrun_minutes",
  "distance_m", "duration_s", "preservation_rate", "weather_risk_delta", "weather_risk", "source_confidence",
  "base_change_weight", "strength_multiplier", "delta_scale", "default_visit_minutes",
]);
const STABLE_FLOAT_OBJECTS = new Set(["metrics", "constraint_strength_weights", "change_type_weights", "schedule_metrics"]);

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
    // Guide-reference durations are integer metadata, unlike itinerary-stop durations.
    const scopeKey = parentKey === "destination_places" && key === "visit_duration_minutes"
      ? null : STABLE_FLOAT_OBJECTS.has(parentKey) ? parentKey : key;
    const serialized = pythonStableJson(value[key], scopeKey);
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

async function normalizeDirectCertificateRecord(record, proposal, evidence = null) {
  // Order edits have no schedule envelope; their full record is still hash/identity bound.
  evidence ||= {evaluator_version: record?.evaluator_version,
    failure_codes: repairCertificateFailureCodes(proposal),
    warning_codes: (record?.warnings || []).map(finding => finding?.code), schedule_metrics: {}};
  if (!exactKeys(record, [
    "certificate_id", "evaluation_id", "plan_id", "source_run_id", "plan_content_hash",
    "evaluator_version", "artifact_grounding_status", "hard_feasibility_status",
    "evaluation_status", "comparison_eligibility", "eligible", "reward_eligibility",
    "failures", "warnings", "failure_count", "warning_count", "blocking_warning_count",
    "nonblocking_warning_count", "metrics", "route_validation", "evaluated_at",
    "schema_version", "content_hash",
  ]) || record.schema_version !== "plan-evaluation-certificate-v1"
      || record.certificate_id !== proposal.certificate_id
      || record.evaluation_id !== proposal.certificate_id
      || record.plan_id !== proposal.child_plan_id
      || record.plan_content_hash !== proposal.child_plan_content_hash
      || record.content_hash !== proposal.certificate_content_hash
      || record.content_hash !== proposal.certificate_identity.content_hash
      || record.evaluator_version !== evidence.evaluator_version
      || record.artifact_grounding_status !== proposal.evidence.artifact_grounding_status
      || record.hard_feasibility_status !== proposal.evidence.hard_feasibility_status
      || record.evaluation_status !== proposal.evaluation_status
      || record.comparison_eligibility !== proposal.eligibility
      || record.eligible !== (proposal.eligibility === "eligible")
      || record.reward_eligibility !== record.eligible
      || typeof record.source_run_id !== "string" || !record.source_run_id
      || !Array.isArray(record.failures) || !Array.isArray(record.warnings)
      || record.failure_count !== record.failures.length || record.warning_count !== record.warnings.length
      || record.blocking_warning_count !== record.failure_count
      || record.nonblocking_warning_count !== record.warning_count
      || !exactStringArray(record.failures.map((finding) => finding?.code), evidence.failure_codes)
      || !exactStringArray(record.warnings.map((finding) => finding?.code), evidence.warning_codes)
      || !record.metrics || !Object.values(record.metrics).every(Number.isFinite)
      || Object.entries(evidence.schedule_metrics).some(([key, value]) => record.metrics[key] !== value)
      || !sameJson(record.route_validation, proposal.route_validation)
      || typeof record.evaluated_at !== "string" || !Number.isFinite(Date.parse(record.evaluated_at))) return null;
  const payload = {...record};
  delete payload.content_hash;
  return await computedStableContentHash(payload) === record.content_hash ? record : null;
}

async function normalizeCertificateScheduleEvidence(value, proposal, scheduleImpact) {
  if (!exactKeys(value, [
    "schema_version", "certificate_id", "certificate_content_hash", "plan_id",
    "plan_content_hash", "evaluator_version", "evaluation_status", "comparison_eligibility",
    "warning_codes", "failure_codes", "schedule_metrics", "certificate_record", "content_hash",
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
  if (!await normalizeDirectCertificateRecord(value.certificate_record, proposal, value)) return null;
  const child = scheduleImpact.child;
  for (const [name, metric] of Object.entries(child.metrics)) {
    if (value.schedule_metrics[name] !== metric) return null;
  }
  const missingCount = child.missing_opening_window_stop_ids.length;
  const expectedWarnings = missingCount > 0 ? ["opening_window_evidence_missing"] : [];
  const expectedFailures = repairCertificateFailureCodes(proposal);
  if (expectedFailures === null
      || !child.blocking_codes.every((code) => expectedFailures.includes(code))) return null;
  const expectedEvaluation = expectedFailures.length
    ? "FAILED"
    : child.status === "incomplete_evidence"
      ? "PASSED_WITH_WARNINGS"
      : "PASSED";
  const expectedEligibility = expectedFailures.length ? "ineligible" : "eligible";
  if (value.schedule_metrics.schedule_missing_opening_window_count !== missingCount
      || !exactStringArray(value.warning_codes, expectedWarnings)
      || !exactStringArray(value.failure_codes, expectedFailures)
      || value.evaluation_status !== expectedEvaluation
      || value.comparison_eligibility !== expectedEligibility
      || proposal.state !== expectedEligibility
      || proposal.eligibility !== expectedEligibility
      || proposal.decision_eligible !== (expectedEligibility === "eligible")
      || proposal.ranking_eligible !== false
      || proposal.acceptance_eligible !== (expectedEligibility === "eligible")
      || proposal.acceptance_blocking_code !== (expectedEligibility === "eligible" ? null : "proposal_not_eligible")
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
  const stopFeatures = new Map((parentPlan?.stops?.features || []).map(
    (feature) => [feature.properties?.stop_id, feature],
  ));
  const sequenceByDay = new Map();
  const stops = [];
  for (const feature of path) {
    const properties = feature?.properties;
    if (properties?.selected_stop !== true) continue;
    const day = properties.arrival_day;
    if (typeof properties.node_id !== "string" || !properties.node_id
        || !Number.isInteger(day) || day <= 0) return null;
    const stopProperties = stopFeatures.get(properties.node_id)?.properties;
    const visitMinutes = stopProperties?.visit_duration_minutes;
    const visitSource = stopProperties?.visit_duration_source;
    const visitMissing = visitMinutes === null;
    if ((!visitMissing && (
      !Number.isFinite(visitMinutes) || visitMinutes < 15 || visitMinutes > 480
        || !["visit_duration_minutes", "duration_minutes", "service_minutes", "dwell_minutes"].includes(visitSource)
    )) || (visitMissing && visitSource !== "unavailable")) return null;
    const sequenceIndex = sequenceByDay.get(day) || 0;
    stops.push({
      stop_id: properties.node_id,
      day,
      sequence_index: sequenceIndex,
      visit_duration_minutes: visitMinutes,
      visit_duration_source: visitSource,
    });
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
  const expectedVisitMinutes = expected?.visit_duration_minutes === null
    ? configuredDefaultVisitMinutes
    : expected?.visit_duration_minutes;
  const expectedVisitSource = expected?.visit_duration_minutes === null
    ? "configured_evaluator_default"
    : expected?.visit_duration_source;
  if (expected && (
    value.stop_id !== expected.stop_id
      || value.day !== expected.day
      || value.sequence_index !== expected.sequence_index
      || value.visit_minutes !== expectedVisitMinutes
      || value.visit_duration_source !== expectedVisitSource
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
    "warning_codes", "failure_codes", "schedule_metrics", "certificate_record", "content_hash",
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
  if (!await normalizeDirectCertificateRecord(value.certificate_record, proposal, value)) return null;
  const child = scheduleImpact.child;
  for (const [name, metric] of Object.entries(child.metrics)) {
    if (value.schedule_metrics[name] !== metric) return null;
  }
  const missingCount = child.missing_opening_window_stop_ids.length;
  const expectedWarnings = missingCount > 0 ? ["opening_window_evidence_missing"] : [];
  const expectedFailures = repairCertificateFailureCodes(proposal);
  if (expectedFailures === null
      || !child.blocking_codes.every((code) => expectedFailures.includes(code))) return null;
  const expectedEvaluation = expectedFailures.length
    ? "FAILED"
    : child.status === "incomplete_evidence" ? "PASSED_WITH_WARNINGS" : "PASSED";
  const expectedEligibility = expectedFailures.length ? "ineligible" : "eligible";
  if (value.schedule_metrics.schedule_missing_opening_window_count !== missingCount
      || !exactStringArray(value.warning_codes, expectedWarnings)
      || !exactStringArray(value.failure_codes, expectedFailures)
      || value.evaluation_status !== expectedEvaluation
      || value.comparison_eligibility !== expectedEligibility
      || proposal.state !== expectedEligibility || proposal.eligibility !== expectedEligibility
      || proposal.decision_eligible !== (expectedEligibility === "eligible")
      || proposal.ranking_eligible !== false
      || proposal.acceptance_eligible !== (expectedEligibility === "eligible")
      || proposal.acceptance_blocking_code !== (expectedEligibility === "eligible" ? null : "proposal_not_eligible")
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
  if (!parentPlan || !["original", "accepted"].includes(parentPlan.role)
      || !durationRouteEvidenceUnchanged(value)) return null;
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
        || parentFeature.properties?.role !== parentPlan.role
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
        ...(Object.hasOwn(value.metrics || {}, "weather_risk_delta") ? ["weather_risk_delta"] : []),
        ...durationScheduleMetricKeys(),
        "schedule_missing_opening_window_count",
      ])
      || !Object.values(value.metrics).every((metric) => Number.isFinite(metric) && metric >= 0)
      || value.metrics.selected_stop_count !== roleImpact.child_role_states.length
      || value.metrics.sequence_stop_count !== roleImpact.child_role_states.length
      || value.metrics.preserved_stop_count !== roleImpact.child_role_states.length
      || value.metrics.preservation_rate !== 1
      || (value.metrics.weather_risk_delta !== undefined && value.metrics.weather_risk_delta !== 0)
      || proposal.repair.tradeoffs.weather_risk_delta !== (value.metrics.weather_risk_delta ?? null)
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
  const expectedFailures = repairCertificateFailureCodes(proposal);
  if (expectedFailures === null
      || !child.blocking_codes.every((code) => expectedFailures.includes(code))) return null;
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
      || proposal.acceptance_eligible !== (expectedEligibility === "eligible")
      || proposal.acceptance_blocking_code !== (expectedEligibility === "eligible" ? null : "proposal_not_eligible")
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
  const originalIdentity = /^route_matrix_[0-9a-f]{16}$/.test(value.matrix_id)
    && /^route_bundle_[0-9a-f]{16}$/.test(value.source_bundle_id);
  const deltaPrefix = /^[0-9a-f]{64}$/.test(value.source_content_sha256)
    ? value.source_content_sha256.slice(0, 16) : null;
  const compositeIdentity = deltaPrefix !== null
    && value.matrix_id === `route_matrix_delta_${deltaPrefix}`
    && value.source_bundle_id === `route_bundle_composite_${deltaPrefix}`;
  if ((!originalIdentity && !compositeIdentity)
      || typeof value.context_snapshot_id !== "string"
      || !value.context_snapshot_id || value.context_snapshot_id.length > 128
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

function repairCertificateFailureCodes(proposal) {
  const failures = proposal?.repair?.certificate?.failures;
  if (!Array.isArray(failures)
      || failures.some((finding) => !exactKeys(finding, [
        "code", "message", "severity", "category", "evidence_refs",
      ])
        || typeof finding.code !== "string" || !finding.code
        || typeof finding.message !== "string" || !finding.message
        || finding.severity !== "failure"
        || typeof finding.category !== "string" || !finding.category
        || !Array.isArray(finding.evidence_refs)
        || !finding.evidence_refs.every((ref) => typeof ref === "string" && ref))) return null;
  return failures.map((finding) => finding.code);
}

function diffChangedDays(diff) {
  if (!diff || typeof diff !== "object") return null;
  const days = new Set();
  for (const key of ["added_stops", "deleted_stops", "day_moves", "time_shifts", "reorder_changes",
    "lodging_changes", "road_changes", "duration_changes", "time_window_changes", "role_changes"]) {
    const changes = diff[key] ?? [];
    if (!Array.isArray(changes)) return null;
    for (const change of changes) {
      if (!change || typeof change !== "object") return null;
      for (const field of ["day", "from_day", "to_day"]) {
        const day = change[field];
        if (day === undefined || day === null) continue;
        if (!Number.isInteger(day) || day <= 0) return null;
        days.add(day);
      }
    }
  }
  return [...days].sort((left, right) => left - right);
}

function planningCanonicalText(value) {
  if (value === null || typeof value === "boolean" || typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return null;
    const buffer = new ArrayBuffer(8);
    new DataView(buffer).setFloat64(0, value, false);
    return "#" + [...new Uint8Array(buffer)].map(byte => byte.toString(16).padStart(2, "0")).join("");
  }
  if (Array.isArray(value)) {
    const parts = value.map(planningCanonicalText);
    return parts.includes(null) ? null : `[${parts.join(",")}]`;
  }
  if (!value || typeof value !== "object") return null;
  const parts = Object.keys(value).sort().map(key => {
    const part = planningCanonicalText(value[key]);
    return part === null ? null : `${JSON.stringify(key)}:${part}`;
  });
  return parts.includes(null) ? null : `{${parts.join(",")}}`;
}

async function normalizeTripPlanningExecution(value, proposal, draft) {
  const bundle = value?.planning_bundle;
  if (value?.schema_version !== "trip-planning-execution-v1" || !bundle
      || bundle.schema_version !== "trip-planning-bundle-v1"
      || draft.length !== 1 || draft[0].type !== "replan_trip"
      || !sameJson(bundle.parameters, draft[0].parameters)
      || bundle.parent_plan_id !== proposal.parent_plan_id
      || bundle.parent_content_hash !== proposal.parent_plan_content_hash
      || bundle.plan?.plan_id !== proposal.child_plan_id
      || bundle.plan?.content_hash !== proposal.child_plan_content_hash
      || !["OPTIMAL", "FEASIBLE"].includes(bundle.solver?.status)
      || bundle.plan?.parent_plan_id !== proposal.parent_plan_id
      || value.plan_diff?.parent_plan_id !== proposal.parent_plan_id
      || value.plan_diff?.child_plan_id !== proposal.child_plan_id
      || await computedStableContentHash(value.plan_diff) !== proposal.diff_content_hash
      || !await normalizeDirectCertificateRecord(value.evaluation_certificate, proposal)
      || !Array.isArray(bundle.official_sources)) return null;
  const sources = new Map([
    ["big_sur_park", "https://www.parks.ca.gov/?page_id=570"],
    ["highway_1", "https://roads.dot.ca.gov/roadscell.php?roadnumber=1"],
    ["hearst_castle", "https://hearstcastle.org/plan/faqs/"],
  ]);
  if (bundle.plan.sequence.includes("redwood_national_park")) {
    sources.set("redwood_lodging", "https://www.nps.gov/redw/planyourvisit/lodging.htm");
    sources.set("redwood_walk", "https://www.nps.gov/thingstodo/lbjgrove.htm");
    sources.set("redwood_conditions", "https://www.nps.gov/redw/planyourvisit/conditions.htm");
    const stops = bundle.plan.selected_stops.filter(s => (s.poi_id || s.stop_id) === "redwood_national_park");
    const conditions = bundle.official_sources.find(s => s.source_id === "redwood_conditions");
    if (stops.length !== 1 || stops[0].visit_duration_minutes !== 90
        || stops[0].latitude !== 41.304183 || stops[0].longitude !== -124.018552
        || !["open", "open_with_hazards"].includes(conditions?.park_access)) return null;
  }
  if (bundle.parameters.additional_yosemite_days !== undefined) {
    if (bundle.parameters.additional_yosemite_days !== 3 || bundle.days?.length !== 10) return null;
    const groups = [["yosemite_lower_fall","yosemite_cooks_meadow"],["yosemite_glacier_point","yosemite_sentinel_dome"],["yosemite_mariposa_grove","yosemite_wawona_history"]];
    const urls = YOSEMITE_SOURCE_URLS;
    Object.entries(urls).forEach(([id,url])=>sources.set(id,url));
    const original = bundle.plan.selected_stops.find(s=>s.poi_id==="yosemite_national_park");
    if (!original || bundle.official_sources.find(s=>s.source_id==="yosemite_conditions")?.park_access!=="open_with_hazards") return null;
    for (const [offset,ids] of groups.entries()) {
      const day=original.day+offset+1;
      const stops=bundle.plan.selected_stops.filter(s=>ids.includes(s.poi_id));
      if (stops.length!==2 || new Set(stops.map(s=>s.poi_id)).size!==2 || stops.some(s=>s.day!==day || !validYosemiteGuide(s.guide))) return null;
      const route=bundle.days.find(d=>d.day===day);
      if (!route || !sameJson([...route.stop_ids].sort(),[...ids].sort())) return null;
    }
  }
  if (bundle.parameters.total_days !== undefined) {
    const count = bundle.parameters.total_days;
    const effective = bundle.effective_parameters || bundle.parameters;
    const limit = {relaxed: 2, balanced: 3, active: 4}[effective.pace];
    if (!Number.isInteger(count) || count < 1 || count > 21 || !limit
        || effective.total_days !== count
        || (bundle.parameters.profile_updates || []).some(key => !sameJson(effective[key], bundle.parameters[key]))
        || bundle.days?.length !== count
        || bundle.days.some((day, index) => day.day !== index + 1 || day.stop_ids.length > limit)
        || new Set(bundle.plan.selected_stops.map(stop => stop.day)).size !== count) return null;
  }
  const hasIsland = bundle.plan.sequence.includes("channel_islands");
  if (hasIsland) {
    sources.set("channel_islands_nps", "https://www.nps.gov/chis/planyourvisit/directions.htm");
    sources.set("channel_islands_operator", "https://www.islandpackers.com/santa-cruz-island/");
    const stop = bundle.plan.selected_stops.find(s => (s.poi_id || s.stop_id) === "channel_islands");
    const excursion = stop?.excursion;
    const routes = proposal.geography_plan?.marine_routes?.features;
    const islands = proposal.geography_plan?.offshore_stops?.features;
    if (!excursion || stop.visit_duration_minutes !== 480 || excursion.planning_minutes !== 480
        || excursion.travel_mode !== "passenger_ferry" || excursion.round_trip !== true
        || excursion.geometry_kind !== "schematic_not_navigation"
        || excursion.schedule_status !== "not_verified" || excursion.reservation_status !== "not_verified"
        || !sameJson(excursion.departure_coordinates, [stop.longitude, stop.latitude])
        || !sameJson(excursion.destination_coordinates, [-119.5562112, 34.0493266])
        || !sameJson(excursion.time_allowance, {check_in_minutes:45, boat_each_way_minutes:90, island_minutes:210, contingency_minutes:45})
        || routes?.length !== 1 || islands?.length !== 1
        || !sameJson(routes[0].geometry, {type:"LineString", coordinates:[excursion.departure_coordinates, excursion.destination_coordinates]})
        || !sameJson(islands[0].geometry, {type:"Point", coordinates:excursion.destination_coordinates})
        || [routes[0], islands[0]].some(f => f.properties?.plan_id !== bundle.plan.plan_id
          || f.properties?.content_hash !== bundle.plan.content_hash || f.properties?.day !== stop.day
          || f.properties?.source_stop_id !== "channel_islands" || f.properties?.road_validated !== false
          || f.properties?.geometry_source !== "schematic_port_to_island"
          || f.properties?.schedule_verified !== false || f.properties?.reservation_verified !== false
          || f.properties?.travel_mode !== "passenger_ferry" || f.properties?.round_trip !== true)) return null;
  } else if ((proposal.geography_plan?.marine_routes?.features || []).length
      || (proposal.geography_plan?.offshore_stops?.features || []).length) return null;
  if (bundle.official_sources.length !== sources.size
      || new Set(bundle.official_sources.map(row => row.source_id)).size !== sources.size
      || bundle.official_sources.some(row => sources.get(row.source_id) !== row.url
        || row.status !== "retrieved" || !Number.isFinite(Date.parse(row.retrieved_at))
        || typeof row.excerpt !== "string")) return null;
  const bundlePayload = { ...bundle };
  delete bundlePayload.content_hash;
  for (const [payload, expected] of [[bundlePayload, bundle.content_hash],
    [bundle.route_matrix?.cells, bundle.route_matrix?.source_content_sha256]]) {
    const serialized = planningCanonicalText(payload);
    if (serialized === null || typeof expected !== "string" || !/^[0-9a-f]{64}$/.test(expected)) return null;
    const bytes = new TextEncoder().encode(serialized);
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const actual = [...new Uint8Array(digest)].map(item => item.toString(16).padStart(2, "0")).join("");
    if (actual !== expected) return null;
  }
  const cells = new Map(bundle.route_matrix.cells.map(cell => [JSON.stringify([cell.origin_id, cell.destination_id]), cell]));
  for (const feature of proposal.geography_plan?.validated_legs?.features || []) {
    const props = feature.properties;
    const cell = cells.get(JSON.stringify([props.origin_id, props.destination_id]));
    if (!cell || props.distance_m !== cell.distance_m || props.duration_s !== cell.duration_s
        || props.geometry_source !== cell.geometry_source
        || !sameJson(feature.geometry.coordinates, cell.geometry.map(([lat, lon]) => [lon, lat]))) return null;
  }
  if (bundle.parameters.required_place_ids.some(id => !bundle.plan.sequence.includes(id))
      || bundle.parameters.excluded_place_ids.some(id => bundle.plan.sequence.includes(id))) return null;
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
      || repairCertificateFailureCodes({ repair: value }) === null
      || value.certificate.failure_count !== value.certificate.failures.length
      || value.certificate.id !== proposal.certificate_id
      || value.certificate.evaluation_status !== proposal.evaluation_status
      || value.certificate.eligible !== (proposal.state === "eligible")
      || typeof value.result !== "string"
      || value.primary_action !== "Review evidence") return null;
  const diff = proposal.plan_diff || proposal.runtime_execution?.plan_diff;
  if (diff) {
    const changedDays = diffChangedDays(diff);
    if (!changedDays || value.changed.affected_day_count !== changedDays.length
        || !sameJson(value.unchanged.days, diff.unchanged_days)) return null;
  }
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

function routeChainMatchesEvidence(value, routeValidation, planning = false) {
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
        || !["cached_osrm_route_geometry", ...(planning ? ["runtime_osrm_geojson"] : [])].includes(properties.geometry_source)
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

function acceptedStopOrderByDay() {
  const plan = state.workspace?.geography?.plans?.find(row => row.plan_id === state.session?.accepted_plan_id);
  if (!plan?.stops?.features) return null;
  const days = new Map();
  for (const feature of [...plan.stops.features].sort((a, b) => a.properties.sequence_index - b.properties.sequence_index)) {
    const {day, stop_id: id} = feature.properties;
    if (!Number.isInteger(day) || day <= 0 || day > 21 || typeof id !== "string"
        || [...days.values()].some(ids => ids.includes(id))) return null;
    if (!days.has(day)) days.set(day, []);
    days.get(day).push(id);
  }
  return days;
}

function normalizePreviewGeography(value, proposal, routeValidation, draft) {
  const marineKeys = proposal.execution_mode === "trip_replan_independent_evaluation"
    && proposal.runtime_execution?.planning_bundle?.plan?.sequence?.includes("channel_islands")
    ? ["marine_routes", "offshore_stops"] : [];
  if (!exactKeys(value, [
    "plan_id", "content_hash", "role", "label", "status", "coverage",
    "route_path", "stops", "validated_legs", "gaps", ...marineKeys,
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
      || !routeChainMatchesEvidence(value, routeValidation, proposal.execution_mode === "trip_replan_independent_evaluation")) return null;
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
  const acceptedByDay = acceptedStopOrderByDay();
  if (!acceptedByDay) return null;
  const reorderedDays = new Set();
  for (const operation of draft) {
    if (operation.type === "replace_nearby") {
      const order = [...acceptedByDay.values()].find((ids) => ids.includes(operation.target));
      if (!order) return null;
      order.splice(order.indexOf(operation.target), 1, operation.parameters.candidate_id);
    } else if (operation.type === "add_candidate") {
      const order = acceptedByDay.get(operation.parameters.day);
      if (!order) return null;
      order.push(operation.target);
    }
  }
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
    runtime_execution: value?.runtime_execution ?? null,
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

function normalizeRuntimeExecution(value, proposal) {
  if (!exactKeys(value, [
    "schema_version", "execution_id", "started_at", "completed_at", "request_id",
    "request_content_hash", "parent_plan_id", "parent_plan_content_hash", "planner_run",
    "repair_request", "planner_trace", "child_plan_id", "child_plan_content_hash",
    "evaluation_id", "evaluation_content_hash", "child_plan", "plan_diff",
    "evaluation_certificate", "evaluation_executed", "evaluator",
  ]) || value.schema_version !== "runtime-repair-execution-v1"
      || !/^execution_[0-9a-f]{32}$/.test(value.execution_id)
      || !/^[0-9a-f]{16}$/.test(value.request_content_hash)
      || value.request_id !== proposal.compiled_request.source_request_id
      || value.parent_plan_id !== proposal.parent_plan_id
      || value.parent_plan_content_hash !== proposal.parent_plan_content_hash
      || value.child_plan_id !== proposal.child_plan_id
      || value.child_plan_content_hash !== proposal.child_plan_content_hash
      || value.evaluation_id !== proposal.certificate_id
      || value.evaluation_content_hash !== proposal.certificate_content_hash
      || value.repair_request?.request_id !== value.request_id
      || value.repair_request?.parent_plan_id !== value.parent_plan_id
      || value.planner_run?.planning_request_id !== value.request_id
      || value.planner_trace?.request_id !== value.request_id
      || value.planner_trace?.parent_plan_id !== value.parent_plan_id
      || !sameJson(value.planner_trace?.planner_run, value.planner_run)) return null;
  const started = Date.parse(value.started_at);
  const completed = Date.parse(value.completed_at);
  if (!Number.isFinite(started) || !Number.isFinite(completed) || completed < started) return null;
  const constraints = value.repair_request.confirmed_constraints;
  if (!constraints || !Array.isArray(constraints.must_include)
      || !Array.isArray(constraints.must_delete)) return null;
  for (const operation of proposal.compiled_request.operations) {
    if (operation.type === "replace_nearby"
        && (!constraints.must_delete.includes(operation.target)
          || !constraints.must_include.includes(operation.parameters.candidate_id))) return null;
    if (operation.type === "add_candidate"
        && !constraints.must_include.includes(operation.target)) return null;
    if (operation.type === "keep_stop" && !constraints.must_include.includes(operation.target)
        && !constraints.preserved_stop_ids?.includes(operation.target)) return null;
  }
  if (proposal.child_plan_id === null) {
    return value.child_plan === null && value.plan_diff === null
      && value.evaluation_certificate === null && value.evaluation_executed === false ? value : null;
  }
  const certificate = value.evaluation_certificate;
  const evaluated = Date.parse(certificate?.evaluated_at);
  if (value.evaluation_executed !== true || value.evaluator !== "PlanEvaluator.evaluate_final_plan"
      || value.planner_run.result_plan_id !== proposal.child_plan_id
      || value.planner_run.execution_status !== "COMPLETED"
      || value.planner_run.method_executed !== proposal.evidence.executed_method
      || value.planner_run.method_requested !== proposal.evidence.requested_method
      || value.child_plan?.plan_id !== proposal.child_plan_id
      || value.child_plan?.content_hash !== proposal.child_plan_content_hash
      || value.child_plan?.parent_plan_id !== proposal.parent_plan_id
      || value.child_plan?.planning_request_id !== value.request_id
      || value.child_plan?.source_run_id !== value.planner_run.run_id
      || value.child_plan?.context_snapshot_id !== value.planner_run.context_snapshot_id
      || value.child_plan?.catalog_snapshot_id !== value.planner_run.catalog_snapshot_id
      || value.plan_diff?.diff_id !== proposal.diff_id
      || value.plan_diff?.parent_plan_id !== proposal.parent_plan_id
      || value.plan_diff?.child_plan_id !== proposal.child_plan_id
      || certificate?.certificate_id !== proposal.certificate_id
      || certificate?.plan_id !== proposal.child_plan_id
      || certificate?.plan_content_hash !== proposal.child_plan_content_hash
      || certificate?.content_hash !== proposal.certificate_content_hash
      || certificate?.source_run_id !== value.planner_run.run_id
      || !sameJson(certificate?.route_validation, proposal.route_validation)
      || proposal.route_validation?.context_snapshot_id !== value.child_plan.context_snapshot_id
      || certificate?.evaluation_status !== proposal.evaluation_status
      || !Number.isFinite(evaluated) || evaluated < started || evaluated > completed) return null;
  return value;
}

async function normalizeCombinedAttributeOwnership(diff, proposal, child) {
  const context = state.workspace?.attribute_diff_context;
  const evidence = diff.attribute_ownership_evidence;
  const mixed = diff.schema_version === "plan-diff-v7";
  if (!exactKeys(context, ["schema_version", "parent_plan_id", "parent_plan_content_hash",
    "owned_constraints", "attribute_constraints", "stops", "route_ids_by_day",
    "default_visit_minutes", "ownership_policy", "content_hash"])
      || context.schema_version !== "accepted-attribute-diff-context-v1"
      || context.parent_plan_id !== proposal.parent_plan_id
      || context.parent_plan_content_hash !== proposal.parent_plan_content_hash
      || !Array.isArray(context.stops) || !Array.isArray(context.attribute_constraints)
      || !Array.isArray(context.owned_constraints) || !Array.isArray(child.owned_constraints)
      || !sameJson([...child.owned_constraints].sort((a, b) => a.constraint_id.localeCompare(b.constraint_id)),
        [...context.owned_constraints].sort((a, b) => a.constraint_id.localeCompare(b.constraint_id)))
      || !exactKeys(evidence, ["schema_version", "parent_plan_id", "parent_plan_content_hash",
        "semantic_scope", "ownership_policy", "constraints", "changes", "content_hash"])
      || evidence.schema_version !== `plan-diff-attribute-ownership-evidence-v${mixed ? 2 : 1}`
      || evidence.parent_plan_id !== context.parent_plan_id
      || evidence.parent_plan_content_hash !== context.parent_plan_content_hash
      || !sameJson(evidence.ownership_policy, context.ownership_policy)
      || !sameJson(evidence.semantic_scope, {
        cost_semantics: "attribute_sensitivity", permission_claim: "not_evaluated",
        feasibility_claim: "not_evaluated", cost_coverage: "all_listed_diff_changes",
        relation_specific_attributes: ["day", "duration", "end_time", "existence", "order", ...(mixed ? ["role"] : []), "start_time", "time_window"],
        lodging_and_road_relations: "not_migrated",
      }) || !Array.isArray(evidence.constraints) || !evidence.constraints.length
      || !Array.isArray(evidence.changes) || !Array.isArray(child.selected_stops)) return null;
  for (const record of [context, evidence]) {
    const payload = {...record};
    delete payload.content_hash;
    if (await computedStableContentHash(payload) !== record.content_hash) return null;
  }
  const policy = context.ownership_policy;
  if (!exactKeys(policy, ["change_type_weights", "constraint_strength_weights"])
      || [policy.change_type_weights, policy.constraint_strength_weights].some(weights => !weights
        || typeof weights !== "object" || Array.isArray(weights)
        || Object.values(weights).some(weight => !Number.isFinite(weight) || weight < 0))) return null;
  const parents = new Map(context.stops.map(stop => [stop.stop_id, stop]));
  const children = new Map(child.selected_stops.map(stop => [stop.stop_id || stop.poi_id, stop]));
  if (parents.size !== context.stops.length || children.size !== child.selected_stops.length) return null;
  const expected = [];
  const consumed = new Set();
  const families = {added_stops: ["added_stop", "existence"], deleted_stops: ["deleted_stop", "existence"],
    duration_changes: ["duration_change", "duration"], time_window_changes: ["time_window_change", "time_window"],
    role_changes: ["role_change", "role"], reorder_changes: ["order_change", "order"], road_changes: ["road_change", "road"]};
  for (const [group, [kind, attribute]] of Object.entries(families)) {
    for (const row of diff[group] || []) {
      const id = attribute === "road" ? `route:${row.day}` : row.stop_id;
      const parent = parents.get(id), stop = children.get(id);
      let before, after, scale = 1;
      if (attribute === "existence") {
        before = parents.has(id); after = children.has(id);
        if (before === after || (kind === "added_stop") !== after) return null;
      } else if (attribute === "road") {
        before = context.route_ids_by_day[row.day] ?? null;
        after = child.route_ids_by_day?.[row.day] ?? null;
        if (row.from_route_id !== before || row.to_route_id !== after) return null;
      } else {
        if (!parent || !stop) return null;
        if (attribute === "duration") {
          const durationValue = (constraint, minutes) => constraint ?? (Number.isInteger(minutes) ? {
            mode: "exact", preferred_minutes: minutes, minimum_minutes: minutes, maximum_minutes: minutes,
          } : null);
          before = durationValue(parent.duration_constraint, parent.duration_minutes);
          after = durationValue(stop.duration_constraint, stop.visit_duration_minutes);
          const accountingFrom = parent.duration_minutes ?? context.default_visit_minutes;
          const accountingTo = stop.visit_duration_minutes ?? context.default_visit_minutes;
          if (!sameJson(row.from_constraint, parent.duration_constraint)
              || row.from_minutes !== parent.duration_minutes || row.from_source !== parent.duration_source
              || row.accounting_from_minutes !== accountingFrom || row.accounting_to_minutes !== accountingTo
              || row.delta_minutes !== accountingTo - accountingFrom) return null;
          scale = Math.max(1, Math.abs(row.delta_minutes) / 60);
        } else if (attribute === "time_window") {
          before = parent.time_window_constraint; after = stop.time_window_constraint ?? null;
          if (!sameJson(row.from_constraint, before)
              || row.from_earliest_arrival !== (before?.earliest_arrival ?? null)
              || row.from_latest_departure !== (before?.latest_departure ?? null)) return null;
        } else if (attribute === "role") {
          before = parent.itinerary_role; after = stop.itinerary_role ?? null;
          if (row.from_role !== before || row.from_source !== parent.itinerary_role_source) return null;
        } else {
          before = parent.order; after = stop.stop_order;
          if (row.from_order !== before || row.to_order !== after) return null;
        }
      }
      const rules = context.attribute_constraints.filter(rule => rule.target_stop_id === id && rule.attribute === attribute);
      if (proposal.eligibility === "eligible" && rules.some(rule => !sameJson(after, rule.canonical_value)
          && (["locked", "booked", "hard"].includes(rule.strength)
            || !["auto_with_penalty", "always"].includes(rule.relaxation_policy)))) return null;
      for (const rule of rules) consumed.add(rule.constraint_id);
      const weight = strength => strength ? (policy.constraint_strength_weights[strength] ?? 1) : 1;
      const ranked = [...rules].sort((a, b) => weight(b.strength) - weight(a.strength)
        || (a.strength < b.strength ? 1 : a.strength > b.strength ? -1 : 0));
      const owner = ranked[0]?.strength ?? "";
      const multiplier = weight(owner), base = policy.change_type_weights[kind] ?? 1;
      const cost = base * multiplier * scale;
      if (row.owner_strength !== owner || row.cost !== cost || before === undefined || after === undefined) return null;
      const satisfaction = actual => actual === null ? "unavailable" : null;
      expected.push({change_kind: kind, target_stop_id: id, attributes: [attribute],
        applicable_constraint_ids: rules.map(rule => rule.constraint_id).sort(),
        selected_constraint_ids: rules.filter(rule => weight(rule.strength) === multiplier).map(rule => rule.constraint_id).sort(),
        before_value: {[attribute]: before}, after_value: {[attribute]: after},
        constraint_satisfaction: rules.map(rule => ({constraint_id: rule.constraint_id, attribute,
          canonical_value: rule.canonical_value,
          satisfaction_before: satisfaction(before) || (sameJson(before, rule.canonical_value) ? "satisfied" : "unsatisfied"),
          satisfaction_after: satisfaction(after) || (sameJson(after, rule.canonical_value) ? "satisfied" : "unsatisfied"),
        })), owner_strength: owner, base_change_weight: base, strength_multiplier: multiplier,
        delta_scale: scale, cost});
    }
  }
  expected.sort((a, b) => a.change_kind.localeCompare(b.change_kind) || a.target_stop_id.localeCompare(b.target_stop_id));
  const constraints = context.attribute_constraints.filter(rule => consumed.has(rule.constraint_id));
  return sameJson(evidence.changes, expected) && sameJson(evidence.constraints, constraints) ? evidence : null;
}

async function normalizeCombinedRuntimeExecution(value, proposal, draft) {
  if (!exactKeys(value, [
    "schema_version", "execution_id", "started_at", "completed_at", "request_id",
    "parent_plan_id", "parent_plan_content_hash", "operations", "repair_request",
    "request_content_hash", "optimizer_execution", "planner_run", "child_plan", "plan_diff",
    "evaluation_certificate", "child_plan_id", "child_plan_content_hash", "evaluation_id",
    "evaluation_content_hash", "evaluation_executed", "evaluator",
  ]) || value.schema_version !== "atomic-combined-repair-execution-v1"
      || !/^execution_[0-9a-f]{32}$/.test(value.execution_id)
      || value.request_id !== proposal.compiled_request.source_request_id
      || value.parent_plan_id !== proposal.parent_plan_id
      || value.parent_plan_content_hash !== proposal.parent_plan_content_hash
      || value.child_plan_id !== proposal.child_plan_id
      || value.child_plan_content_hash !== proposal.child_plan_content_hash
      || value.evaluation_id !== proposal.certificate_id
      || value.evaluation_content_hash !== proposal.certificate_content_hash
      || value.evaluator !== "PlanEvaluator.evaluate_final_plan"
      || !sameJson(value.operations, draftHashRecords(draft))) return null;
  const started = Date.parse(value.started_at);
  const completed = Date.parse(value.completed_at);
  const request = value.repair_request;
  if (!Number.isFinite(started) || !Number.isFinite(completed) || completed < started
      || !exactKeys(request, ["schema_version", "request_id", "parent_plan_id", "request_family",
        "user_intent", "operations", "confirmed_constraints"])
      || request.schema_version !== "atomic-combined-repair-request-v1"
      || request.request_id !== value.request_id || request.parent_plan_id !== value.parent_plan_id
      || request.request_family !== "combined_typed_edit"
      || request.user_intent !== "Review the complete staged draft as one independently evaluated child."
      || !sameJson(request.operations, value.operations) || !exactKeys(request.confirmed_constraints, [])
      || await computedStableContentHash(request) !== value.request_content_hash) return null;
  const selection = draft.filter(operation => ["keep_stop", "replace_nearby", "add_candidate"].includes(operation.type));
  const optimizer = value.optimizer_execution;
  if (selection.length) {
    const certificate = optimizer?.evaluation_certificate;
    if (!optimizer || !normalizeRuntimeExecution(optimizer, {
      compiled_request: {source_request_id: optimizer.request_id, operations: selection},
      parent_plan_id: optimizer.parent_plan_id, parent_plan_content_hash: optimizer.parent_plan_content_hash,
      child_plan_id: optimizer.child_plan_id, child_plan_content_hash: optimizer.child_plan_content_hash,
      certificate_id: optimizer.evaluation_id, certificate_content_hash: optimizer.evaluation_content_hash,
      diff_id: optimizer.plan_diff?.diff_id,
      route_validation: certificate?.route_validation,
      evaluation_status: certificate?.evaluation_status,
      evidence: {requested_method: optimizer.planner_run?.method_requested,
        executed_method: optimizer.planner_run?.method_executed},
    }) || Date.parse(optimizer.started_at) < started || Date.parse(optimizer.completed_at) > completed) return null;
  } else if (optimizer !== null) return null;
  if (proposal.child_plan_id === null) {
    return value.evaluation_executed === false && value.child_plan === null
      && value.plan_diff === null && value.evaluation_certificate === null && value.planner_run === null ? value : null;
  }
  const child = value.child_plan;
  const run = value.planner_run;
  const certificate = value.evaluation_certificate;
  if (value.evaluation_executed !== true
      || child?.plan_id !== proposal.child_plan_id || child.content_hash !== proposal.child_plan_content_hash
      || child.parent_plan_id !== proposal.parent_plan_id || child.planning_request_id !== value.request_id
      || child.source_run_id !== run?.run_id || child.context_snapshot_id !== run?.context_snapshot_id
      || child.catalog_snapshot_id !== run?.catalog_snapshot_id
      || run?.planning_request_id !== value.request_id || run.result_plan_id !== child.plan_id
      || run.execution_status !== "COMPLETED"
      || run.method_requested !== "atomic_combined_repair_v1" || run.method_executed !== run.method_requested
      || run.method_requested !== proposal.evidence.requested_method || run.method_executed !== proposal.evidence.executed_method
      || run.solver_backend !== (selection.length ? "deterministic_context_aware_heuristic" : "typed_direct_edits")
      || !await normalizeDirectCertificateRecord(certificate, proposal)
      || certificate.source_run_id !== run.run_id
      || certificate.route_validation.context_snapshot_id !== child.context_snapshot_id
      || Date.parse(certificate.evaluated_at) < started || Date.parse(certificate.evaluated_at) > completed) return null;
  const diff = value.plan_diff;
  const groups = ["added_stops", "deleted_stops", "day_moves", "time_shifts", "reorder_changes", "lodging_changes", "road_changes"];
  const typedGroups = ["duration_changes", "time_window_changes", "role_changes"];
  const typed = typedGroups.filter(key => diff?.[key]?.length);
  const owned = ["plan-diff-v5", "plan-diff-v7"].includes(diff?.schema_version);
  const ownershipGroups = {added_stops: "existence", deleted_stops: "existence",
    reorder_changes: "order", duration_changes: "duration", time_window_changes: "time_window"};
  const requiresOwnedEvidence = (state.workspace?.attribute_diff_context?.attribute_constraints || []).some(rule =>
    Object.entries(ownershipGroups).some(([key, attribute]) => rule.attribute === attribute
      && (diff?.[key] || []).some(row => row.stop_id === rule.target_stop_id)));
  if (requiresOwnedEvidence && !owned) return null;
  const schema = owned ? diff.schema_version : typed.length > 1 ? "plan-diff-v6" : typed.includes("role_changes") ? "plan-diff-v4"
    : typed.includes("time_window_changes") ? "plan-diff-v3" : typed.includes("duration_changes") ? "plan-diff-v2" : "plan-diff-v1";
  // v6 serializes only the non-empty attribute families, not all three.
  const typedKeys = typed;
  if (!exactKeys(diff, ["diff_id", "parent_plan_id", "child_plan_id", ...groups,
    "unchanged_days", "weighted_edit_cost", "schema_version", ...typedKeys,
    ...(owned ? ["attribute_ownership_evidence"] : [])])
      || diff.schema_version !== schema || diff.diff_id !== proposal.diff_id
      || diff.parent_plan_id !== proposal.parent_plan_id || diff.child_plan_id !== child.plan_id
      || ![...groups, ...typedKeys].every(key => Array.isArray(diff[key]))
      || diff.day_moves.length || diff.time_shifts.length || diff.lodging_changes.length
      || !Number.isFinite(diff.weighted_edit_cost)
      || await computedStableContentHash(diff) !== proposal.diff_content_hash) return null;
  if (owned && (certificate.evaluator_version !== "verify-001-plan-evaluator-v3"
      || (diff.schema_version === "plan-diff-v7" && typed.length < 2 && !typed.includes("role_changes"))
      || (diff.schema_version === "plan-diff-v5" && (typed.length > 1 || typed.includes("role_changes")))
      || !await normalizeCombinedAttributeOwnership(diff, proposal, child))) return null;
  const costRows = [...groups, ...typedKeys].flatMap(key => diff[key]);
  if (costRows.some(row => !Number.isFinite(row.cost) || row.cost < 0)
      || Math.abs(costRows.reduce((sum, row) => sum + row.cost, 0) - diff.weighted_edit_cost) > 1e-8) return null;
  const diffSeed = {parent_plan_id: diff.parent_plan_id, child_plan_id: diff.child_plan_id,
    added: diff.added_stops, deleted: diff.deleted_stops, day_moves: diff.day_moves, time_shifts: diff.time_shifts,
    reorder: diff.reorder_changes, lodging: diff.lodging_changes, road: diff.road_changes,
    unchanged_days: diff.unchanged_days, weighted_edit_cost: diff.weighted_edit_cost};
  for (const key of typed) diffSeed[key] = diff[key];
  if (owned) diffSeed.attribute_ownership_evidence = diff.attribute_ownership_evidence;
  if (`diff_${await computedStableContentHash(diffSeed)}` !== diff.diff_id) return null;
  const expectedDays = acceptedStopOrderByDay();
  if (!expectedDays) return null;
  for (const operation of selection) {
    if (operation.type === "replace_nearby") {
      const order = [...expectedDays.values()].find(ids => ids.includes(operation.target));
      if (!order) return null;
      order.splice(order.indexOf(operation.target), 1, operation.parameters.candidate_id);
    } else if (operation.type === "add_candidate") {
      const order = expectedDays.get(operation.parameters.day);
      if (!order || order.includes(operation.target)) return null;
      order.push(operation.target);
    }
  }
  for (const operation of draft.filter(row => row.type === "set_stop_order")) {
    const order = expectedDays.get(operation.parameters.day);
    const index = order?.indexOf(operation.target) ?? -1;
    if (index < 0 || operation.parameters.sequence_index >= order.length) return null;
    order.splice(index, 1);
    order.splice(operation.parameters.sequence_index, 0, operation.target);
  }
  const orderedDays = [...expectedDays].map(([day, stop_ids]) => ({day, stop_ids}));
  const sequence = orderedDays.flatMap(day => day.stop_ids);
  const assignments = Object.fromEntries(orderedDays.flatMap(day => day.stop_ids.map(id => [id, day.day])));
  if (!sameJson(child.ordered_days, orderedDays) || !sameJson(child.sequence, sequence)
      || !sameJson(child.day_assignments, assignments) || !Array.isArray(child.selected_stops)
      || child.selected_stops.length !== sequence.length) return null;
  const stops = new Map(child.selected_stops.map(stop => [stop.stop_id || stop.poi_id, stop]));
  const geography = proposal.geography_plan?.stops?.features;
  if (stops.size !== sequence.length || !Array.isArray(geography) || geography.length !== sequence.length) return null;
  for (const [index, id] of sequence.entries()) {
    const stop = stops.get(id);
    const feature = geography.find(row => row.properties.stop_id === id);
    if (!stop || !feature || stop.day !== assignments[id] || feature.properties.day !== stop.day
        || feature.properties.sequence_index !== index + 1 || feature.properties.stop_order !== (stop.stop_order ?? index + 1)
        || !sameCoordinate(feature.geometry?.coordinates, [stop.longitude, stop.latitude])
        || feature.properties.itinerary_role !== (stop.itinerary_role ?? null)
        || feature.properties.itinerary_role_source !== (stop.itinerary_role_source || "unavailable")
        || feature.properties.visit_duration_minutes !== (stop.visit_duration_minutes ?? null)
        || !sameJson(feature.properties.inspector_metadata?.duration_constraint, stop.duration_constraint ?? null)) return null;
  }
  const operationGroups = {set_stop_duration: "duration_changes", set_stop_time_window: "time_window_changes", set_stop_role: "role_changes"};
  for (const [type, key] of Object.entries(operationGroups)) {
    const operations = draft.filter(row => row.type === type);
    const changes = diff[key] || [];
    if (changes.length !== operations.length || new Set(changes.map(row => row.stop_id)).size !== changes.length) return null;
    for (const operation of operations) {
      const stop = stops.get(operation.target);
      const change = changes.find(row => row.stop_id === operation.target);
      if (!stop || !change || change.day !== stop.day) return null;
      if (type === "set_stop_duration") {
        const requested = canonicalExactDuration(operation.parameters.duration);
        if (!requested || !sameJson(stop.duration_constraint, requested)
            || stop.visit_duration_minutes !== requested.preferred_minutes
            || !sameJson(change.to_constraint, requested) || change.to_minutes !== stop.visit_duration_minutes) return null;
      } else if (type === "set_stop_time_window") {
        const requested = timeWindowConstraintFromParameters(operation.parameters);
        if (!requested || !sameJson(stop.time_window_constraint, requested)
            || !sameJson(change.to_constraint, requested)
            || change.to_earliest_arrival !== requested.earliest_arrival
            || change.to_latest_departure !== requested.latest_departure) return null;
      } else if (stop.itinerary_role !== operation.parameters.role
          || stop.itinerary_role_source !== "user_declared_itinerary_role"
          || change.to_role !== stop.itinerary_role || change.to_source !== stop.itinerary_role_source) return null;
    }
  }
  const added = selection.filter(row => row.type !== "keep_stop").map(row => row.type === "replace_nearby" ? row.parameters.candidate_id : row.target).sort();
  const deleted = selection.filter(row => row.type === "replace_nearby").map(row => row.target).sort();
  const changedDays = diffChangedDays(diff);
  if (!sameJson(diff.added_stops.map(row => row.stop_id).sort(), added)
      || !sameJson(diff.deleted_stops.map(row => row.stop_id).sort(), deleted)
      || selection.some(row => row.type === "keep_stop" && !stops.has(row.target))
      || !changedDays || !sameJson(diff.unchanged_days, [...expectedDays.keys()].filter(day => !changedDays.includes(day)))) return null;
  return value;
}

async function normalizeEvaluatedPreviewResponse(value, expected) {
  if (!exactKeys(value, ["proposal", "session"])) return null;
  const { proposal, session } = value;
  const orderOnly = expected.draft.length > 0
    && expected.draft.every((operation) => operation.type === "set_stop_order");
  const durationDraftOnly = expected.draft.length > 0
    && expected.draft.every((operation) => operation.type === "set_stop_duration");
  const timeWindowDraftOnly = expected.draft.length > 0
    && expected.draft.every((operation) => operation.type === "set_stop_time_window");
  const ownedScheduleEdit = (durationDraftOnly || timeWindowDraftOnly)
    && (state.workspace?.attribute_diff_context?.attribute_constraints || []).some(rule =>
      rule.attribute === (durationDraftOnly ? "duration" : "time_window")
        && expected.draft.some(operation => operation.target === rule.target_stop_id));
  const durationOnly = durationDraftOnly && !ownedScheduleEdit;
  const timeWindowOnly = timeWindowDraftOnly && !ownedScheduleEdit;
  const roleOnly = expected.draft.length > 0
    && expected.draft.every((operation) => operation.type === "set_stop_role");
  const directEditOnly = orderOnly || durationOnly || timeWindowOnly || roleOnly;
  const combined = ownedScheduleEdit || (!directEditOnly && expected.draft.some(operation => [
    "set_stop_order", "set_stop_duration", "set_stop_time_window", "set_stop_role",
  ].includes(operation.type)));
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
    ...(proposal?.child_plan_id === null ? ["blocking_findings"] : []),
    ...(["deterministic_repair_pipeline", "atomic_combined_independent_evaluation", "trip_replan_independent_evaluation"].includes(proposal?.execution_mode) ? ["runtime_execution"] : []),
    "decision_eligible", "ranking_eligible", "acceptance_eligible", "acceptance_blocking_code",
    ...(orderOnly ? ["evaluation_certificate", "plan_diff"] : []),
    ...((durationOnly || timeWindowOnly) ? [
      "plan_diff", "schedule_impact", "schedule_impact_identity", "certificate_schedule_evidence",
    ] : []),
    ...(roleOnly ? [
      "plan_diff", "role_impact", "role_impact_identity", "certificate_role_evidence",
      "role_proposal_integrity",
    ] : []),
  ];
  const draftHash = await computedDraftContentHash(expected.draft);
  const operationIds = expected.draft.map((operation) => operation.operation_id);
  if (!exactKeys(proposal, proposalKeys)
      || proposal.schema_version !== "draft-preview-v1"
      || !["eligible", "ineligible", "blocked", "failed", "not_evaluated"].includes(proposal.state)
      || proposal.eligibility !== (["eligible", "ineligible"].includes(proposal.state)
        ? proposal.state : "not_evaluated")
      || !(proposal.reason === null || typeof proposal.reason === "string")
      || !["deterministic_repair_pipeline", "independent_evaluated_direct_edit", "independent_evaluated_combined_edit"].includes(proposal.provenance)
      || !["deterministic_repair_pipeline", "typed_direct_edit_independent_evaluation", "atomic_combined_independent_evaluation", "trip_replan_independent_evaluation"].includes(proposal.execution_mode)
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
        !== canonicalJson(draftHashRecords(expected.draft))
      || proposal.decision_eligible !== (proposal.state === "eligible")
      || proposal.ranking_eligible !== false
      || proposal.acceptance_eligible !== (proposal.state === "eligible")
      || proposal.acceptance_blocking_code !== (proposal.state === "eligible" ? null : "proposal_not_eligible")) return null;
  if (!exactKeys(session.proposal, proposalKeys)
      || !sameJson(previewProposalBinding(session.proposal), previewProposalBinding(proposal))) return null;
  if (proposal.execution_mode === "deterministic_repair_pipeline"
      && !normalizeRuntimeExecution(proposal.runtime_execution, proposal)) return null;
  if (proposal.execution_mode === "atomic_combined_independent_evaluation"
      && !await normalizeCombinedRuntimeExecution(proposal.runtime_execution, proposal, expected.draft)) return null;
  if (proposal.execution_mode === "trip_replan_independent_evaluation"
      && !await normalizeTripPlanningExecution(proposal.runtime_execution, proposal, expected.draft)) return null;
  const roleProposalIntegrity = roleOnly
    ? await normalizeRoleProposalIntegrity(proposal.role_proposal_integrity, proposal)
    : null;
  if (roleOnly && !roleProposalIntegrity) return null;
  const deterministicPipeline = proposal.provenance === "deterministic_repair_pipeline"
    && proposal.execution_mode === "deterministic_repair_pipeline";
  if ((directEditOnly && (
    proposal.provenance !== "independent_evaluated_direct_edit"
      || proposal.execution_mode !== "typed_direct_edit_independent_evaluation"
  )) || (combined && (proposal.provenance !== "independent_evaluated_combined_edit"
    || proposal.execution_mode !== "atomic_combined_independent_evaluation"))
    || (!directEditOnly && !combined && !deterministicPipeline
      && !(proposal.execution_mode === "trip_replan_independent_evaluation"
        && expected.draft.length === 1 && expected.draft[0].type === "replan_trip"))) return null;
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
    if (!["blocked", "failed", "not_evaluated"].includes(proposal.state) || proposal.reason === null
        || !Array.isArray(proposal.blocking_findings)
        || proposal.blocking_findings.some((finding) => !finding || typeof finding.code !== "string"
          || typeof finding.corrective_action !== "string")
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
  if (!["eligible", "ineligible"].includes(proposal.state)
      || !/^plan_[0-9a-f]{16}$/.test(proposal.child_plan_id)
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
  if (orderOnly && (
    !await normalizeDirectCertificateRecord(proposal.evaluation_certificate, proposal)
    || proposal.plan_diff?.diff_id !== proposal.diff_id
    || proposal.plan_diff?.parent_plan_id !== proposal.parent_plan_id
    || proposal.plan_diff?.child_plan_id !== proposal.child_plan_id
    || await computedStableContentHash(proposal.plan_diff) !== proposal.diff_content_hash
  )) return null;
  const routeValidation = normalizeRouteValidation(proposal.route_validation);
  if (!routeValidation) return null;
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
          || proposal.repair.changed.road_change_count !== 0
          || proposal.repair.changed.lodging_change_count !== 0
          || proposal.repair.tradeoffs.weighted_edit_cost !== planDiff.weighted_edit_cost
          || proposal.repair.tradeoffs.preservation_rate !== 1
          || proposal.repair.tradeoffs.utility_retained !== null
          || (certificateScheduleEvidence.certificate_record.metrics.weather_risk_delta !== undefined
            && certificateScheduleEvidence.certificate_record.metrics.weather_risk_delta !== 0)
          || proposal.repair.tradeoffs.weather_risk_delta
            !== (certificateScheduleEvidence.certificate_record.metrics.weather_risk_delta ?? null)
          || proposal.evaluation_status !== certificateScheduleEvidence.evaluation_status
          || proposal.evidence.hard_feasibility_status !== (
            certificateScheduleEvidence.failure_codes.length ? "FAILED" : "PASSED"
          )
          || proposal.state !== (
            certificateScheduleEvidence.failure_codes.length ? "ineligible" : "eligible"
          )
          || proposal.reason !== (
            certificateScheduleEvidence.failure_codes.length ? "independent_evaluation_failed" : null
          )
      ))) return null;
  if (timeWindowOnly && (
    !planDiff
      || !scheduleImpact
      || !certificateScheduleEvidence
      || !durationRouteEvidenceUnchanged(proposal.geography_plan)
      || proposal.repair.changed.added.length !== 0
      || proposal.repair.changed.removed.length !== 0
      || proposal.repair.changed.road_change_count !== 0
      || proposal.repair.changed.lodging_change_count !== 0
      || proposal.repair.tradeoffs.weighted_edit_cost !== planDiff.weighted_edit_cost
      || proposal.repair.tradeoffs.preservation_rate !== 1
      || proposal.repair.tradeoffs.utility_retained !== null
      || (certificateScheduleEvidence.certificate_record.metrics.weather_risk_delta !== undefined
        && certificateScheduleEvidence.certificate_record.metrics.weather_risk_delta !== 0)
      || proposal.repair.tradeoffs.weather_risk_delta
        !== (certificateScheduleEvidence.certificate_record.metrics.weather_risk_delta ?? null)
      || proposal.evaluation_status !== certificateScheduleEvidence.evaluation_status
      || proposal.evidence.hard_feasibility_status !== (
        certificateScheduleEvidence.failure_codes.length ? "FAILED" : "PASSED"
      )
      || proposal.state !== (
        certificateScheduleEvidence.failure_codes.length ? "ineligible" : "eligible"
      )
      || proposal.reason !== (
        certificateScheduleEvidence.failure_codes.length ? "independent_evaluation_failed" : null
      )
  )) return null;
  if (roleOnly && (
    !planDiff
      || !roleImpact
      || !certificateRoleEvidence
      || !normalizeRoleGeography(proposal.geography_plan, planDiff, roleImpact)
      || proposal.repair.changed.added.length !== 0
      || proposal.repair.changed.removed.length !== 0
      || proposal.repair.changed.road_change_count !== 0
      || proposal.repair.changed.lodging_change_count !== 0
      || proposal.repair.tradeoffs.weighted_edit_cost !== planDiff.weighted_edit_cost
      || proposal.repair.tradeoffs.preservation_rate !== 1
      || proposal.repair.tradeoffs.utility_retained !== null
      || proposal.repair.tradeoffs.weather_risk_delta
        !== (certificateRoleEvidence.certificate_record.metrics.weather_risk_delta ?? null)
      || proposal.evaluation_status !== certificateRoleEvidence.evaluation_status
      || proposal.evidence.hard_feasibility_status !== (
        certificateRoleEvidence.failure_codes.length ? "FAILED" : "PASSED"
      )
      || proposal.state !== (
        certificateRoleEvidence.failure_codes.length ? "ineligible" : "eligible"
      )
      || proposal.reason !== (
        certificateRoleEvidence.failure_codes.length ? "independent_evaluation_failed" : null
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
  owner_sign_in_required: "Please sign in again to open your private workspace.",
  hosted_auth_unavailable: "Sign-in is temporarily unavailable. Your workspace has been kept. Please try again.",
  hosted_state_unavailable: "Your workspace is temporarily unavailable. Saved changes are protected; please try again after the service restarts.",
  invalid_copilot_stream: "The reply could not be read. Check the saved conversation before trying again.",
  copilot_stream_incomplete: "The connection closed before the reply finished. Check its saved status before sending it again.",
  copilot_outcome_unknown: "The provider outcome is unknown. This message will not be sent again automatically; checking it only reconciles saved status.",
  copilot_result_context_stale: "The itinerary changed while the reply was running. The reply is retained in history but cannot stage changes for this newer itinerary.",
  conversation_turn_limit_reached: "This conversation is full. Start a new conversation before sending another message. No provider call was made.",
  conversation_size_limit_reached: "This conversation has reached its storage limit. Start a new conversation. No provider call was made.",
  draft_pipeline_execution_timed_out: "The repair timed out before evaluation. Your accepted itinerary and staged draft are unchanged.",
  preview_capacity_reached: "The local repair engine is busy. Your draft is unchanged. Wait for a running preview to finish, then try again.",
  preview_execution_timeout: "The repair did not finish in time. No result was applied; your accepted itinerary and staged draft are unchanged.",
  preview_client_disconnected: "The preview connection closed before its result could be applied. Your accepted itinerary is unchanged.",
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
  unsupported_optimizer_mapping: "This action does not yet have an implemented optimizer mapping. No evaluated repair was run.",
  draft_optimizer_mapping_unsupported: "This draft contains an action without an implemented optimizer mapping. Remove or revise that action before previewing.",
  draft_evaluated_operation_combination_unsupported: "An evaluated direct edit must run alone. Keep only same-day reorder, exact-duration, time-window, or itinerary-role operations before previewing.",
  draft_order_index_invalid: "Choose a valid position within the stop's accepted day.",
  duration_mode_evaluation_not_supported: "Preferred, minimum, maximum, and range duration rules remain draft only. Choose Exact to run an evaluated preview.",
  invalid_stop_duration: "Enter an exact whole-minute duration from 15 through 480. No draft was added.",
  invalid_stop_time_window: "Provide at least one zero-padded local-time bound, with earliest service admission no later than latest departure. Cross-midnight windows are not supported.",
  stop_role_evaluation_not_supported: "Structural itinerary roles remain draft only and cannot run an evaluated preview in this slice.",
  invalid_stop_role: "Choose a supported trip-specific itinerary role. No draft was added.",
  acceptance_confirmation_required: "Confirm the durable decision before continuing.",
  acceptance_proposal_required: "Run and review an eligible evaluated preview before accepting.",
  acceptance_proposal_invalid: "The evaluated proposal no longer matches the immutable artifacts. Run Preview again.",
  acceptance_pointer_conflict: "Another decision changed the accepted plan. Reload before deciding again.",
  acceptance_idempotency_conflict: "This decision key was already used for a different request.",
  accepted_plan_requires_new_repair_baseline: "This accepted plan is durable. Start a new repair baseline before adding more edits.",
};

function apiErrorMessage(detail, status) {
  return API_ERROR_COPY[detail] || detail || `Request failed (${status})`;
}

function isBuildMismatch(code) {
  return ["product_client_build_mismatch", "product_server_build_changed"].includes(code);
}

function renderBuildRecovery(code) {
  state.buildRecovery = code;
  state.previewRuntime.epoch += 1;
  $$("dialog[open]").forEach((dialog) => dialog.close());
  $("#app-shell").hidden = true;
  const screen = $("#loading-screen");
  screen.hidden = false;
  screen.classList.add("recovery-screen");
  const restart = code === "product_server_build_changed";
  screen.innerHTML = `<section class="recovery-card" data-build-recovery="${code}">
    <p class="eyebrow">Workspace update</p><h1>${restart ? "Restart the local app, then reload" : "Reload to use the updated app"}</h1>
    <p>${restart ? "The source files changed after this server started. It cannot safely combine the old backend with new website files. Restart the launcher before continuing." : "This page belongs to a different app build. No new edit or chat request will be sent from this page."}</p>
    <p>Your stored session, drafts, accepted itinerary and conversation have not been cleared.</p>
    <button class="primary" id="reload-current-build" type="button">${restart ? "Reload after restarting" : "Reload updated app"}</button>
  </section>`;
  $("#reload-current-build").addEventListener("click", () => location.reload());
  $("#reload-current-build").focus();
}

async function api(path, { method = "GET", body, headers: extraHeaders = {}, signal, pendingMutation = false } = {}) {
  if (state.mutationRecovery?.blocked && method !== "GET" && !path.endsWith("/draft/impact-preview") && !pendingMutation) {
    throw Object.assign(new Error("Resolve the saved action before continuing. Your message has not been sent."), {code:"mutation_outcome_uncertain"});
  }
  if ((state.workspaceConflict || state.workspaceSyncPending) && method !== "GET" && !path.endsWith("/draft/impact-preview")
      && !(pendingMutation && state.mutationRecovery?.blocked)) {
    throw Object.assign(new Error("Refresh the current workspace before making another change."), {code:"workspace_refresh_required"});
  }
  if (state.buildRecovery) throw Object.assign(new Error(state.buildRecovery), {code:state.buildRecovery, status:409});
  const headers = { Accept: "application/json", ...extraHeaders, "X-Product-Build-Id":state.clientBuildId || "" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (state.token) headers["X-Session-Token"] = state.token;
  const response = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    if (isBuildMismatch(payload?.detail)) renderBuildRecovery(payload.detail);
    const error = new Error(apiErrorMessage(payload?.detail, response.status));
    error.code = payload?.detail;
    error.status = response.status;
    throw error;
  }
  if (method !== "GET" && !path.endsWith("/draft/impact-preview")) announceWorkspaceChange(payload);
  return payload;
}

function copilotStreamError(code, status = 500, retryable = false) {
  const error = new Error(apiErrorMessage(code, status));
  error.code = code;
  error.status = status;
  error.retryable = retryable;
  return error;
}

function copilotStreamFrame(block) {
  const data = block.split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");
  if (!data) return null;
  try {
    return JSON.parse(data);
  } catch {
    throw copilotStreamError("invalid_copilot_stream", 502, true);
  }
}

async function streamCopilotMessage(
  path,
  { body, signal, onEvent = () => {} },
) {
  if (state.workspaceConflict || state.workspaceSyncPending) throw copilotStreamError("workspace_refresh_required", 409, false);
  if (state.mutationRecovery?.blocked) throw copilotStreamError("mutation_outcome_uncertain", 409, false);
  if (state.buildRecovery) throw copilotStreamError(state.buildRecovery, 409, false);
  const headers = {
    Accept: "text/event-stream",
    "Content-Type": "application/json",
    "X-Product-Build-Id": state.clientBuildId || "",
  };
  if (state.token) headers["X-Session-Token"] = state.token;
  const response = await fetch(path, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal,
  });
  const contentType = response.headers.get("content-type") || "";
  if (!response.ok) {
    const payload = contentType.includes("application/json") ? await response.json() : null;
    if (isBuildMismatch(payload?.detail)) renderBuildRecovery(payload.detail);
    throw copilotStreamError(payload?.detail, response.status, response.status >= 500 && !isBuildMismatch(payload?.detail));
  }
  if (contentType.includes("application/json")) return response.json();
  if (!contentType.includes("text/event-stream")) {
    throw copilotStreamError("invalid_copilot_stream", 502, true);
  }

  let buffer = "";
  let completed = null;
  let terminal = false;
  let lastSequence = 0;
  const consume = (block) => {
    const event = copilotStreamFrame(block);
    if (!event) return;
    if (
      event.schema_version !== "copilot-stream-v1"
      || event.client_message_id !== body.client_message_id
      || !Number.isInteger(event.sequence)
      || event.sequence <= lastSequence
      || typeof event.request_id !== "string"
    ) {
      throw copilotStreamError("invalid_copilot_stream", 502, true);
    }
    lastSequence = event.sequence;
    onEvent(event);
    if (event.type === "turn.completed") {
      if (terminal || !event.payload) throw copilotStreamError("invalid_copilot_stream", 502, true);
      terminal = true;
      completed = event.payload;
    } else if (event.type === "turn.failed") {
      if (terminal) throw copilotStreamError("invalid_copilot_stream", 502, true);
      terminal = true;
      throw copilotStreamError(event.code, event.status, event.retryable === true);
    } else if (!["message.accepted", "assistant.status"].includes(event.type)) {
      throw copilotStreamError("invalid_copilot_stream", 502, true);
    }
  };

  if (response.body?.getReader) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() || "";
      frames.forEach(consume);
      if (done) break;
    }
  } else {
    buffer = await response.text();
  }
  if (buffer.trim()) consume(buffer);
  if (!terminal || !completed) throw copilotStreamError("copilot_stream_incomplete", 502, true);
  return completed;
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

    await refreshMapRuntime();

    const payload = await restoreOrCreateSession();
    state.workspace = payload.workspace;
    state.session = payload.session;
    syncPreviewRuntimeFromSession({ invalidate: false });
    if (payload.mutation_token) state.token = payload.mutation_token;
    state.registry = payload.registry;
    Object.assign(state.interestRegions, restoreInterestPreferences(workspaceStorage, interestPreferencesKey()));
    restorePendingMutationInput();
    try { await mutationRecovery.restore(); } catch { /* The recovery panel retains an unreadable record. */ }
    await loadDraftImpactPreview({ render: false });
    bindEvents();
    applyRuntimeAvailability();
    $("#loading-screen").hidden = true;
    $("#app-shell").hidden = false;
    if (findSelectedStop()) {
      state.mapInteraction.selectionKind = "stop";
      state.mapInteraction.selectedStopPlanId = findSelectedStop().plan_id;
    }
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const defaultCompareId = state.route === "/app/compare"
      && !state.session.selected_alternative_id
      && !activeProposalPlan() ? recommendedAlternativeId() : null;
    if (defaultCompareId) await selectContext({ selected_alternative_id: defaultCompareId });
    else renderAll();
    if (mutationRecovery.pending) await recoverPendingMutation(false);
    else await copilot.restore();
    if (state.sessionRecoveryNotice) toast(state.sessionRecoveryNotice, true);
  } catch (error) {
    renderStartupFailure(error);
  }
}

async function refreshMapRuntime() {
  if (state.mapRecoveryPending) return false;
  state.mapRecoveryPending = true;
  state.mapConfig = null;
  try {
    for (const delay of [0, 1000, 2000]) {
      if (delay) await new Promise((resolve) => setTimeout(resolve, delay));
      try {
        const health = await api("/api/health");
        if (health?.schema_version !== "product-health-v2"
            || health.product_id !== "itinerary-repair-copilot" || !health.core_ready) continue;
        state.health = health;
        const config = await api("/api/map/config");
        state.mapConfig = config?.schema_version === "product-map-configuration-v2" ? config : null;
        if (state.mapConfig?.status === "ready" && component("map")?.status === "ready") return true;
      } catch {
        state.mapConfig = null;
      }
    }
    state.mapConfig = null;
    return false;
  } finally {
    state.mapRecoveryPending = false;
  }
}

async function restoreOrCreateSession() {
  let saved = null;
  try {
    saved = JSON.parse(workspaceStorage.getItem(SESSION_STORAGE_KEY) || "null");
  } catch {
    workspaceStorage.removeItem(SESSION_STORAGE_KEY);
  }
  if (saved?.session_id && saved?.mutation_token) {
    state.token = saved.mutation_token;
    try {
      return await api(`/api/sessions/${encodeURIComponent(saved.session_id)}`);
    } catch (error) {
      if (isBuildMismatch(error?.code) || workspaceStorage.getItem(PENDING_MUTATION_KEY)
          || error.status !== 404 || !["unknown_session", "session_not_found", "session_expired"].includes(error.code)) throw error;
      if (error.code === "session_expired") {
        state.sessionRecoveryNotice = "Your editing session expired. Your accepted itinerary is current. Earlier drafts are retained for recovery and have not been reapplied.";
      }
      state.token = null;
      workspaceStorage.removeItem(SESSION_STORAGE_KEY);
    }
  }
  if (workspaceStorage.getItem(PENDING_MUTATION_KEY)) {
    throw new Error("The pending action's workspace credentials could not be restored. Its saved record has been preserved.");
  }
  const payload = await api("/api/sessions", { method: "POST", body: {} });
  state.token = payload.mutation_token;
  workspaceStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify({
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
  if ($("#loading-screen")?.querySelector?.(".hosted-login")) return;
  if (isBuildMismatch(error?.code)) return renderBuildRecovery(error.code);
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
  const mapReady = component("map")?.status === "ready" && state.mapConfig?.status === "ready";
  banner.hidden = unavailable.length === 0;
  $("#app-shell").classList.toggle("has-runtime-banner", unavailable.length > 0);
  if (unavailable.length) {
    banner.innerHTML = `<strong>Limited local runtime</strong><span>${unavailable.map((item) => escapeHtml(componentMessage(item))).join(" ")}</span>${mapReady ? "" : '<button type="button" id="retry-map-runtime">Retry map</button>'}`;
    $("#retry-map-runtime")?.addEventListener("click", async () => {
      if (state.mapRecoveryPending) return;
      $("#retry-map-runtime").disabled = true;
      $("#retry-map-runtime").textContent = "Checking map…";
      await refreshMapRuntime();
      applyRuntimeAvailability();
      renderAll();
    });
  } else {
    banner.innerHTML = "";
  }

  $("#layers-button").disabled = !mapReady;
  $("#layers-button").title = mapReady ? "Show route layers" : "Local map layers are unavailable";

  copilot.update();
}

function bindEvents() {
  const signOut = $("#hosted-sign-out");
  signOut.hidden = window.PRODUCT_DEPLOYMENT?.static_frontend !== true;
  if (!signOut.hidden) signOut.addEventListener("click", () => {
    signOutHosted();
    location.reload();
  });
  $$("[data-route]").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.route)));
  copilot.bind();
  bindWorkspaceSynchronization();
  $("#copilot-input").addEventListener("input", savePendingMutationInput);
  window.addEventListener("pagehide", savePendingMutationInput);
  $("#map-select-button").addEventListener("click", () => setMapInteractionMode("select"));
  $("#map-edit-button").addEventListener("click", () => setMapInteractionMode("edit"));
  $("#map-interest-button").addEventListener("click", () => {
    if (state.interestRegions.enabled) {
      resetInterestRegions({hide: true});
      renderMap();
    } else {
      void fetchInterestRegions();
    }
  });
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
  $("#destination-day").addEventListener("change", refreshCandidateAdditionAvailability);
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
    state.route = normalizeRoute(currentAppPath());
    renderAll();
  });
  setupDockResize();
}

function navigate(route) {
  const normalized = normalizeRoute(route);
  if (currentAppPath() !== normalized) history.pushState({}, "", appRouteUrl(normalized));
  state.route = normalized;
  if (normalized === "/app/compare" && !state.session.selected_alternative_id && !activeProposalPlan()) {
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
  restoreMutationControls();
  renderTrip();
  renderWorkspace();
  renderContextPanel();
  renderMetrics();
  copilot.update();
  updateActiveNav();
  $("#map-interest-button").hidden = state.route !== "/app/map";
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
  renderMutationRecovery();
}

function activeAlternativeId() {
  return activeRenderPlan()?.plan_id || null;
}

function activeProposalPlan() {
  if (state.workspaceSyncPending) return null;
  const proposal = state.session.proposal;
  const plan = proposal?.geography_plan;
  const routeValidation = normalizeRouteValidation(proposal?.route_validation);
  if (proposal?.schema_version !== "draft-preview-v1"
      || proposal.execution_mode === "registered_independent_evaluation"
      || proposal.state !== "eligible"
      || plan?.plan_id !== proposal.child_plan_id
      || plan?.content_hash !== proposal.child_plan_content_hash
      || plan?.role !== "draft_preview"
      || plan?.status !== "ready"
      || plan?.coverage?.status !== "complete"
      || plan.coverage.complete !== true
      || plan.coverage.itinerary_sequence_accounted !== true
      || !Number.isInteger(plan.coverage.route_path_node_count)
      || !Number.isInteger(plan.coverage.required_leg_count)
      || !Number.isInteger(plan.coverage.road_validated_leg_count)
      || !routeValidation
      || routeValidation?.required_leg_count !== plan.coverage.required_leg_count
      || routeValidation?.road_validated_leg_count !== plan.coverage.road_validated_leg_count
      || plan.coverage.required_leg_count !== plan.coverage.road_validated_leg_count
      || plan.coverage.gap_count !== 0) return null;
  for (const name of ["route_path", "stops", "validated_legs", "gaps"]) {
    if (plan[name]?.type !== "FeatureCollection"
        || !Array.isArray(plan[name].features)
        || (["route_path", "stops"].includes(name) && plan[name].features.length === 0)
        || plan[name].features.some((feature) => (
          feature?.properties?.plan_id !== proposal.child_plan_id
          || feature?.properties?.content_hash !== proposal.child_plan_content_hash
          || feature?.properties?.role !== "draft_preview"
        ))) return null;
  }
  if (plan.coverage.route_path_node_count !== plan.route_path.features.length
      || plan.coverage.required_leg_count !== plan.validated_legs.features.length
      || plan.route_path.features.length !== plan.coverage.required_leg_count + 1
      || plan.gaps.features.length !== 0) return null;
  return plan;
}

function copilotPreviewState() {
  const plan = activeProposalPlan();
  const mapFailureMessage = {
    MAP_RENDER_FAILED: "The map could not be rendered or verified. Acceptance remains disabled. Retry the map before accepting this preview.",
    MAP_IDENTITY_FAILED: "The map does not match the expected itinerary identity. Acceptance remains disabled. Retry the map before accepting this preview.",
  }[state.previewRuntime.status];
  return {
    status: state.previewRuntime.status,
    planId: plan?.plan_id || state.session?.proposal?.child_plan_id || null,
    contentHash: plan?.content_hash || state.session?.proposal?.child_plan_content_hash || null,
    mapPlanId: state.acknowledgedPreviewPlanId,
    message: mapFailureMessage || state.previewRuntime.message || state.session?.proposal?.message || null,
  };
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

function activeRenderPlan(geography = activeGeography()) {
  const proposalPlan = activeProposalPlan();
  if (proposalPlan) return proposalPlan;
  const explicitTripPreview = geography?.plans?.find(
    (plan) => plan.plan_id === state.tripPreviewPlanId,
  );
  return explicitTripPreview
    || acceptedPlan(geography)
    || geography?.plans?.find((plan) => plan.role === "original")
    || null;
}

function timelineForPlan(plan) {
  const accepted = acceptedPlan(state.workspace.geography)
    || state.workspace.geography?.plans?.find((candidate) => candidate.role === "original");
  const proposal = state.session.proposal;
  const previewDays = proposal?.state === "eligible" && proposal.child_plan_id === plan?.plan_id
    && proposal.child_plan_content_hash === plan?.content_hash
    ? diffChangedDays(proposal.plan_diff || proposal.runtime_execution?.plan_diff) || [] : [];
  const templateByDay = new Map(
    (state.workspace.timeline || []).map((day) => [Number(day.day), day]),
  );
  const stopsByDay = new Map();
  (plan?.stops?.features || []).forEach((feature) => {
    const stop = feature.properties || {};
    const day = Number(stop.day);
    if (!Number.isInteger(day)) return;
    if (!stopsByDay.has(day)) stopsByDay.set(day, []);
    stopsByDay.get(day).push({
      id: stop.stop_id,
      name: stop.name || stop.stop_id,
      city: stop.city || null,
      order: Number(stop.stop_order),
    });
  });
  const acceptedByDay = new Map();
  (accepted?.stops?.features || []).forEach((feature) => {
    const stop = feature.properties || {};
    const day = Number(stop.day);
    if (!acceptedByDay.has(day)) acceptedByDay.set(day, []);
    acceptedByDay.get(day).push(`${stop.stop_id}:${Number(stop.stop_order)}`);
  });
  const dayCount = Math.max(Number(state.workspace.trip?.day_count) || templateByDay.size, ...stopsByDay.keys());
  return Array.from({ length: dayCount }, (_, index) => {
    const day = index + 1;
    const template = templateByDay.get(day) || {};
    const stops = (stopsByDay.get(day) || []).sort((left, right) => left.order - right.order);
    const activeIdentity = stops.map((stop) => `${stop.id}:${stop.order}`);
    const acceptedIdentity = (acceptedByDay.get(day) || []).sort();
    const changed = previewDays.includes(day)
      || [...activeIdentity].sort().join("|") !== acceptedIdentity.join("|");
    return {
      ...template,
      day,
      stops: stops.map((stop) => ({ ...stop, changed })),
      states: [
        ...(plan?.role === "original" && state.workspace.trip?.affected_days?.includes(day) ? ["affected"] : []),
        ...(changed ? ["changed"] : []),
      ],
    };
  });
}
function renderTrip() {
  const { trip } = state.workspace;
  const renderedPlan = activeRenderPlan();
  const timeline = timelineForPlan(renderedPlan);
  const routeNodes = [...(renderedPlan?.route_path?.features || [])]
    .sort((left, right) => Number(left.properties?.occurrence_index) - Number(right.properties?.occurrence_index));
  const routeStart = readableRouteAnchor(routeNodes[0]?.properties?.node_id);
  const routeEnd = readableRouteAnchor(routeNodes.at(-1)?.properties?.node_id);
  $("#trip-title").textContent = routeNodes.length
    ? `California Coast · ${routeStart} to ${routeEnd}` : trip.title;
  $("#run-label").textContent = state.registry.label;
  $("#trip-metrics").innerHTML = `
    <div><b>${timeline.length}</b><span>Days</span></div>
    <div><b>${renderedPlan?.stops?.features?.length ?? trip.stop_count}</b><span>Stops</span></div>
    <div><b>${trip.affected_days.length}</b><span>Original issue days</span></div>`;
  $("#day-list").innerHTML = timeline.map((day) => {
    const selected = day.day === state.session.selected_day;
    const affected = day.states.includes("affected");
    const changed = day.states.includes("changed");
    const names = day.stops.map((stop) => stop.name).join(" · ") || "No recorded stop";
    return `<button type="button" class="day-card ${selected ? "is-selected" : ""} ${affected ? "is-affected" : ""}" data-day="${day.day}" data-stop="${escapeHtml(day.stops[0]?.id || "")}" aria-pressed="${selected}">
      <span class="day-number">Day ${day.day}</span>
      <span class="day-copy"><strong>${escapeHtml(names)}</strong><span>${day.stops.length} recorded stop${day.stops.length === 1 ? "" : "s"}</span></span>
      <span class="day-state" aria-label="${changed ? "Changed from accepted itinerary" : affected ? "Original disruption day" : "Recorded day"}">${changed ? "Δ" : affected ? "⚠" : "·"}</span>
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

function proposalCertificate(proposal) {
  const record = proposal?.runtime_execution?.evaluation_certificate
    || proposal?.evaluation_certificate
    || proposal?.certificate_schedule_evidence?.certificate_record
    || proposal?.certificate_role_evidence?.certificate_record;
  return record?.plan_id === proposal?.child_plan_id
    && record?.plan_content_hash === proposal?.child_plan_content_hash
    && record?.certificate_id === proposal?.certificate_id
    && record?.content_hash === proposal?.certificate_content_hash ? record : null;
}

function planPresentationEvidence(plan) {
  const proposal = state.session.proposal;
  const proposalExact = Boolean(plan?.plan_id && plan?.content_hash)
    && proposal?.child_plan_id === plan?.plan_id
    && proposal?.child_plan_content_hash === plan?.content_hash
    && proposal?.certificate_identity?.plan_id === plan?.plan_id
    && proposal?.certificate_identity?.plan_content_hash === plan?.content_hash;
  if (proposalExact) return {
    eligibility: proposal.eligibility || "Not evaluated",
    certificateId: proposal.certificate_id,
    permissions: proposal.repair?.permissions || null,
    repair: proposal.repair || null,
    metrics: proposalCertificate(proposal)?.metrics || {},
    weightedEditCost: proposal.repair?.tradeoffs?.weighted_edit_cost ?? null,
  };
  const record = state.workspace.alternative_evidence?.[plan?.plan_id];
  const certificate = record?.certificate;
  const exact = Boolean(plan?.plan_id && plan?.content_hash)
    && record?.plan_id === plan?.plan_id && record?.plan_content_hash === plan?.content_hash
    && certificate?.plan_id === plan?.plan_id && certificate?.plan_content_hash === plan?.content_hash;
  return {
    eligibility: exact ? certificate.comparison_eligibility || "Not evaluated" : "Not evaluated",
    certificateId: exact ? certificate.certificate_id : null,
    permissions: exact && plan?.role === "registered_repair" ? state.workspace.repair?.permissions : null,
    repair: exact && plan?.role === "registered_repair" ? state.workspace.repair : null,
    metrics: exact ? certificate.metrics || {} : {},
    weightedEditCost: exact ? record.diff?.weighted_edit_cost ?? null : null,
  };
}

function routeLegMinutes(feature) {
  const seconds = feature?.properties?.duration_s;
  return typeof seconds === "number" && Number.isFinite(seconds) && seconds >= 0 ? seconds / 60 : null;
}

function planRouteMinutes(plan) {
  const legs = validatedRouteFeatures(plan);
  if (!legs.length || plan?.coverage?.complete !== true || routeGapFeatures(plan).length
      || plan.coverage.gap_count !== 0 || legs.length !== plan.coverage.required_leg_count
      || legs.length !== plan.coverage.road_validated_leg_count
      || legs.some((leg) => leg.properties?.plan_id !== plan.plan_id
        || leg.properties?.road_validated !== true || leg.properties?.fallback_used !== false)) return null;
  const minutes = legs.map(routeLegMinutes);
  return minutes.every((value) => value !== null) ? minutes.reduce((total, value) => total + value, 0) : null;
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
  const { eligibility, permissions } = planPresentationEvidence(plan);
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
  const durations = legs.map(routeLegMinutes);
  const duration = planRouteMinutes(plan) !== null && durations.length && durations.every(
    (value) => typeof value === "number" && Number.isFinite(value) && value >= 0,
  ) ? `${formatMetric(durations.reduce((total, value) => total + value, 0))} minutes` : "Unavailable";
  const { eligibility, permissions } = planPresentationEvidence(plan);
  const bookingImpact = permissions
    ? `${permissions.booked_change_count ?? "Unavailable"} booked; ${permissions.locked_change_count ?? "Unavailable"} locked changes`
    : "Unavailable";
  const stopOrder = stops.length
    ? stops.map((feature, index) => `${index + 1}. ${feature.properties?.name || feature.properties?.stop_id}`).join(" → ")
    : "Unavailable";
  const travelLegs = legs.length ? `<ol>${legs.map((feature) => {
    const properties = feature.properties || {};
    const names = routeNamePair(properties, plan);
    const durationMinutes = routeLegMinutes(feature);
    const minutes = durationMinutes !== null
      ? `${formatMetric(durationMinutes)} minutes` : "duration Unavailable";
    return `<li>${escapeHtml(names.origin)} → ${escapeHtml(names.destination)} · ${escapeHtml(minutes)} · road validated</li>`;
  }).join("")}</ol>` : "<p>Travel legs are Unavailable for this day.</p>";
  const originalDay = routeSummaryForPlan(originalPlan).split(/(?=Day \d+:)/)
    .find((row) => row.startsWith(`Day ${selectedDay}:`)) || "Unavailable";
  const proposedDay = routeSummaryForPlan(plan).split(/(?=Day \d+:)/)
    .find((row) => row.startsWith(`Day ${selectedDay}:`)) || "Unavailable";
  return `<section class="text-route-summary" data-plan-id="${escapeHtml(plan?.plan_id || "")}" data-plan-content-hash="${escapeHtml(plan?.content_hash || "")}" aria-label="Selected day route summary">
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

function currentInterestResponse() {
  const value = state.interestRegions.response;
  const parent = acceptedPlan(activeGeography());
  const displayed = activeRenderPlan();
  return state.interestRegions.enabled && value?.session_id === state.session?.session_id
    && value.session_revision === state.session.revision
    && value.accepted_plan_id === state.session.accepted_plan_id
    && value.accepted_plan_content_hash === parent?.content_hash
    && value.exploration_plan_id === displayed?.plan_id
    && value.exploration_plan_content_hash === displayed?.content_hash
    && value.context.day === state.session.selected_day
    && value.search_context?.area_mode === state.interestRegions.searchArea
    && value.search_context?.radius_km === state.interestRegions.radiusKm
    && value.search_context?.stop_id === state.interestRegions.focusStopId
    && value.context.route_leg_id === state.session.selected_segment_id ? value : null;
}

function selectInterestPlace(placeId) {
  state.interestRegions.selectedId = placeId;
  renderMap();
  $("#geographic-map")?.scrollIntoView({block: "center"});
}

function interestPlaceListHtml(response) {
  if (!response) return "";
  const view = state.interestRegions;
  const query = view.placeSearch.trim().toLocaleLowerCase();
  const regionPoints = interestPlacesInRegion(response, view.massLevel);
  const inside = new Set(regionPoints.map(point => point.place_id));
  const points = (view.placeScope === "region" ? regionPoints : response.poi_predictions)
    .filter(point => `${point.name} ${(point.place_categories || []).join(" ")}`.toLocaleLowerCase().includes(query))
    .slice().sort((a, b) => (a.distance_from_focus_km || 0) - (b.distance_from_focus_km || 0)
      || b.interest_probability_mean - a.interest_probability_mean || a.name.localeCompare(b.name));
  const hasRegion = response.surface.status === "ready";
  return `<p role="status">${points.length} matching places · ${response.poi_predictions.length} nearby in this catalog.</p>
    ${points.length ? `<ol class="interest-place-list">${points.map(point => `<li><button type="button" data-interest-place="${escapeHtml(point.place_id)}" aria-pressed="${point.place_id === view.selectedId}">
      <strong>${escapeHtml(point.name)}</strong><span>${Math.round(point.interest_probability_mean * 100)}% estimated interest${Number.isFinite(point.distance_from_focus_km) ? ` · ${point.distance_from_focus_km.toFixed(1)} km away` : ""}</span>
      <small>${escapeHtml((point.place_categories || []).map(humanLabel).join(", ") || "Category unavailable")} · ${hasRegion ? inside.has(point.place_id) ? "Inside selected boundary" : "Outside selected boundary" : "Boundary unavailable"}</small>
      <small>90% model interval: ${point.interest_probability_interval.map(value => `${Math.round(value * 100)}%`).join("–")} · ${escapeHtml(humanLabel(point.feasibility_status))} · Locate on map</small>
    </button></li>`).join("")}</ol>` : response.poi_predictions.length ? "<p>No catalog places match this filter. Try All nearby places or another search.</p>" : "<p>No nearby places are recorded for this route yet. Choose another day or route to explore.</p>"}`;
}

function interestPlaceDataHtml(data) {
  if (!data) return "";
  const label = {live: "Online places", fresh_cache: "Saved online places", stale_cache: "Older saved online places", bundled_fallback: "Saved place catalog"}[data.status] || "Place data";
  const date = Date.parse(data.fetched_at || data.bundled_snapshot_at || "");
  const updated = Number.isFinite(date) ? ` · ${data.fetched_at ? "fetched" : "map snapshot"} ${new Date(date).toLocaleString()}` : "";
  const note = data.error_code === "nearby_cache_write_failed" ? " These online results could not be saved for reopening."
    : data.error_code === "nearby_response_limit_exceeded" ? " Online search was too broad; try a smaller radius."
    : ["nearby_refresh_cooldown", "nearby_refresh_in_progress"].includes(data.error_code) ? " Online refresh is temporarily paused; saved places remain available."
    : data.error_code ? " Online refresh is unavailable; showing saved places." : "";
  return `<small class="interest-place-source" role="status">${escapeHtml(label + updated + note)}</small>`;
}

function interestMapSummaryHtml(response) {
  const view = state.interestRegions;
  if (!view.enabled) return "";
  if (!response) {
    const failed = view.status === "failed";
    const message = view.error === "poi_route_matrix_invalid"
      ? "The route evidence could not be read. Retry to check again."
      : view.error || "Finding nearby places for this route…";
    return `<aside class="interest-map-summary" aria-label="Interest places">
      <strong>${failed ? "Interest region unavailable" : "Loading interest region…"}</strong>
      <span role="${failed ? "alert" : "status"}">${escapeHtml(message)}</span>
      ${failed ? '<button type="button" id="retry-interest-regions">Retry interest region</button>' : ""}</aside>`;
  }
  const selected = response.poi_predictions.find(point => point.place_id === view.selectedId);
  const required = response.required_points.find(point => point.place_id === view.selectedId);
  const search = response.search_context;
  return `<aside class="interest-map-summary" aria-label="Interest places"><strong>${response.poi_predictions.length} nearby ${response.poi_predictions.length === 1 ? "place" : "places"}</strong>
    <span>Around ${escapeHtml(search.name)}</span>
    <label>Search radius <select id="interest-radius" aria-label="Nearby search radius">${[1, 2, 5, 10].map(radius => `<option value="${radius}" ${radius === search.radius_km ? "selected" : ""}>${radius} km</option>`).join("")}</select></label>
    ${search.visitor_area ? `<label>Search area <select id="interest-search-area" aria-label="Nearby search area"><option value="visitor_area" ${view.searchArea === "visitor_area" ? "selected" : ""}>${escapeHtml(search.visitor_area.name)}</option><option value="itinerary_pin" ${view.searchArea === "itinerary_pin" ? "selected" : ""}>Original itinerary pin</option></select></label>` : ""}
    <small>Straight-line distance from this search center</small>
    ${response.place_data ? interestPlaceDataHtml(response.place_data) : ""}
    ${selected ? `<span>${escapeHtml(selected.name)} · ${selected.distance_from_focus_km.toFixed(1)} km away</span><small>${Math.round(selected.interest_probability_mean * 100)}% estimated interest · 90% interval ${selected.interest_probability_interval.map(value => `${Math.round(value * 100)}%`).join("–")}</small>` : required ? `<span>Must-go: ${escapeHtml(required.name)}</span><small>Confirmed itinerary constraint</small>` : response.poi_predictions.length ? "<span>Click a blue point or browse the place list.</span>" : "<span>No nearby places are recorded within this radius. Try a larger radius.</span>"}
    ${response.poi_predictions.length && response.surface.status !== "ready" ? '<small role="status">Continuous shading is unavailable for this route. You can still inspect the listed places.</small>' : ""}
    <button type="button" id="browse-interest-places">Browse places ↓</button>
    ${view.selectedId ? '<button type="button" id="show-whole-interest-region">Show whole region</button>' : ""}</aside>`;
}

function interestExplorerHtml() {
  const view = state.interestRegions;
  if (!view.enabled) return "";
  const response = currentInterestResponse();
  const points = response?.poi_predictions || [];
  const selected = points.find(point => point.place_id === view.selectedId);
  const required = response?.required_points || [];
  const selectedRequired = required.find(point => point.place_id === view.selectedId);
  const region = response?.surface?.regions_geojson?.features.find(feature => feature.properties.mass_level === view.massLevel);
  const previewExploration = response?.plan_scope === "evaluated_preview";
  const percent = value => `${Math.round(value * 100)}%`;
  return `<section class="interest-region-panel" aria-labelledby="interest-region-heading">
    <header><div><span class="eyebrow">Explore places</span><h2 id="interest-region-heading">Your interest region</h2></div><button type="button" id="close-interest-regions">Hide interest layer</button></header>
    ${response ? `<p>Searching within ${response.search_context.radius_km} km of <strong>${escapeHtml(response.search_context.name)}</strong>. Distances are straight-line distances from the displayed search center. Your itinerary stops are unchanged.</p>` : ""}
    <p>Same-hue shading shows model interest density. The darkest markers are confirmed must-go constraints. Interest does not establish itinerary feasibility.</p>
    ${previewExploration ? '<p class="interest-region-caveat">Exploring the evaluated preview. This itinerary has not been accepted. Place browsing is read only. Road and official-source evidence may have aged; accepting the trip still requires current evidence.</p>' : ""}
    <div class="interest-region-legend" aria-label="Interest layer legend"><span><i class="interest-density-swatch" aria-hidden="true"></i>Lower → higher model density</span><span><i class="interest-required-swatch" aria-hidden="true"></i>Confirmed must-go</span></div>
    <div class="interest-region-controls">
      <label>Stated preference<select id="interest-preference"><option value="1" ${view.naturePreference === 1 ? "selected" : ""}>More nature</option><option value="0" ${view.naturePreference === 0 ? "selected" : ""}>Balanced</option><option value="-1" ${view.naturePreference === -1 ? "selected" : ""}>More culture</option></select></label>
      <label>Model interest mass<select id="interest-mass-level">${[0.5, 0.8, 0.95].map(level => `<option value="${level}" ${view.massLevel === level ? "selected" : ""}>${percent(level)} region</option>`).join("")}</select></label>
      <button type="button" id="refresh-interest-regions" ${view.status === "loading" ? "disabled" : ""}>Refresh estimates</button>
      <button type="button" id="refresh-interest-place-data" ${view.status === "loading" ? "disabled" : ""}>Refresh online places</button>
    </div>
    ${response?.place_data ? interestPlaceDataHtml(response.place_data) : ""}
    <div class="interest-region-status" role="status">${view.status === "loading" ? "Estimating interest for this route context…" : view.error ? escapeHtml(view.error) : response ? `${response.model.evidence_mode === "prior_only" ? "Prior estimate" : "Feedback-based estimate"} · not calibrated · ${points.length} places · ${response.candidate_universe.route_checked_candidate_count} route-checked candidates. ${region ? `Boundary contains ${percent(region.properties.achieved_mass)} of the model distribution.` : points.length ? `Continuous region unavailable: ${humanLabel(response.surface.unavailable_reason)}. Places remain inspectable.` : "No nearby places are recorded for this route yet. Choose another day or route to explore."}` : "Route context changed. Select a validated route leg and refresh."}</div>
    ${response ? `<p class="interest-region-caveat">${escapeHtml(response.surface.support_note || "Geographic support is unavailable.")} ${response.surface.excluded_candidate_count || 0} places excluded from the land-supported density. The boundary percentage describes model mass, not the chance that every place is suitable.</p>` : ""}
    ${points.length ? `<label>Inspect a place<select id="interest-place-selector"><option value="">Choose a place (${points.length})</option>${points.map(point => `<option value="${escapeHtml(point.place_id)}" ${point.place_id === view.selectedId ? "selected" : ""}>${escapeHtml(point.name)}</option>`).join("")}</select></label>` : ""}
    ${selected ? `<article class="interest-place-detail"><h3>${escapeHtml(selected.name)}</h3><p>Estimated interest: <strong>${percent(selected.interest_probability_mean)}</strong> · 90% model uncertainty interval: ${selected.interest_probability_interval.map(percent).join("–")}.</p>
      ${selected.description ? `<p>${escapeHtml(selected.description)}</p>` : ""}
      <p>${escapeHtml((selected.place_categories || []).map(humanLabel).join(", "))} · ${selected.display_coordinate.latitude.toFixed(5)}, ${selected.display_coordinate.longitude.toFixed(5)}</p>
      <p>Sources: ${(selected.source_refs || []).map(source => safeExternalLink(source, source)).join(" · ") || "Unavailable"}</p>
      <p>Route status: ${escapeHtml(humanLabel(selected.feasibility_status))}. ${selected.candidate?.burden?.recommended === true ? "Existing evaluator recommendation is available." : "No automatic recommendation."}</p><button type="button" id="explore-interest-place" ${state.mapInteraction.mode === "edit" && !previewExploration ? "" : 'disabled aria-describedby="interest-edit-hint"'}>Check this place on the map</button>${previewExploration ? '<p id="interest-edit-hint">Accept this evaluated itinerary before creating a stop draft on its routes.</p>' : state.mapInteraction.mode === "edit" ? "" : '<p id="interest-edit-hint">Choose Edit in the map toolbar to check road access for this place.</p>'}<p>Road checking does not establish itinerary feasibility. Custom stops are draft only; evaluated preview and acceptance require a supported itinerary operation.</p></article>` : ""}
    ${selectedRequired ? `<article class="interest-place-detail"><h3>Must-go: ${escapeHtml(selectedRequired.name)}</h3><p>${selectedRequired.conflict_status === "satisfied" ? "Confirmed constraint is satisfied by the accepted itinerary." : `Conflict: ${escapeHtml(humanLabel(selectedRequired.conflict_status))}`}</p><p>Estimated interest: ${selectedRequired.interest_probability_mean == null ? "insufficient model features" : percent(selectedRequired.interest_probability_mean)}. Its darkest color represents the confirmed constraint, not a probability of 100%.</p></article>` : ""}
    <details ${required.length ? "open" : ""}><summary>Confirmed must-go points (${required.length})</summary>${required.length ? `<ul>${required.map(point => `<li><button type="button" data-interest-place="${escapeHtml(point.place_id)}" aria-pressed="${point.place_id === view.selectedId}">Must-go: ${escapeHtml(point.name)}</button> · ${point.conflict_status === "satisfied" ? "present in accepted itinerary" : `Conflict: ${escapeHtml(humanLabel(point.conflict_status))}`}${point.coordinate ? "" : " · coordinate unavailable"}</li>`).join("")}</ul>` : "<p>No confirmed hard existence constraints for this day. Map selection alone does not create one.</p>"}</details>
    ${response ? `<h3>Browse specific places</h3><div class="interest-region-controls">
      <label>Show places<select id="interest-place-scope"><option value="region" ${view.placeScope === "region" ? "selected" : ""}>Inside selected region</option><option value="all" ${view.placeScope === "all" ? "selected" : ""}>All nearby places</option></select></label>
      <label>Search name or category<input id="interest-place-search" type="search" value="${escapeHtml(view.placeSearch)}" placeholder="e.g. beach, museum, Point Lobos"></label></div>
      <p>These are catalog locations. Shading between them does not imply additional known places or confirmed road access.</p>
      <div id="interest-place-results">${interestPlaceListHtml(response)}</div>` : ""}
  </section>`;
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
  return geographyFeature?.properties?.name || humanLabel(stopId);
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
    replan_trip: "Trip replan",
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
  if (!operations.length && state.session.proposal?.execution_mode === "trip_replan_independent_evaluation"
      && state.session.proposal?.state === "eligible") {
    return '<span class="status-pill warning">Trip replan preview</span><p>This selection belongs to the evaluated trip preview. Accepting the new itinerary remains separate.</p>';
  }
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
    const latestDraft = state.session.draft.at(-1);
    const removeEnabled = MAP_DRAFT_OPERATION_TYPES.has(latestDraft?.type)
      && latestDraft?.target === waypoint.waypoint_id;
    return `<section class="map-context-card draft-waypoint" aria-labelledby="map-context-title">
      <span class="status-pill route-checked">Route-checked draft</span>
      <h3 id="map-context-title">${escapeHtml(waypoint.name)}</h3>
      <dl><dt>Role</dt><dd>${escapeHtml(waypoint.role)}</dd>
        <dt>Day</dt><dd>${escapeHtml(waypoint.day ?? "Unavailable")}</dd>
        <dt>Insertion</dt><dd>${escapeHtml(stopNameForId(waypoint.insertion?.predecessor_id, acceptedPlan(geography)))} → ${escapeHtml(stopNameForId(waypoint.insertion?.successor_id, acceptedPlan(geography)))}</dd>
        <dt>State</dt><dd>Confirmed session draft; not evaluated</dd></dl>
      <p>This waypoint and its affected road legs are server-owned draft evidence. The accepted parent is unchanged.</p>
      <button type="button" id="relocate-custom-waypoint" ${relocateEnabled ? "" : "disabled title=\"Switch to Edit mode to relocate this custom waypoint\""}>Relocate this waypoint</button>
      <button type="button" id="remove-custom-waypoint" ${removeEnabled ? "" : "disabled title=\"Undo newer draft changes before removing this waypoint\""}>Remove this draft waypoint</button>
      ${removeEnabled ? "" : "<p>Only the latest draft operation can be undone. Undo newer changes first.</p>"}
    </section>`;
  }
  if (interaction.selectionKind === "stop") {
    const stop = findSelectedStop();
    if (!stop) return "";
    const acceptedPlanId = state.session.accepted_plan_id;
    const selectedPlanId = stop.plan_id;
    const stopPlan = geography?.plans?.find(
      (plan) => plan.plan_id === selectedPlanId,
    );
    const feature = stopPlan?.stops?.features?.find(
      (candidate) => candidate.properties?.stop_id === stop.id,
    );
    const canonicalPlan = state.session.proposal?.runtime_execution?.planning_bundle?.plan;
    const canonicalStop = canonicalPlan?.plan_id === selectedPlanId && canonicalPlan?.content_hash === stopPlan?.content_hash
      ? canonicalPlan.selected_stops?.find(row => (row.stop_id || row.poi_id) === stop.id) : null;
    const properties = {...canonicalStop, ...feature?.properties};
    const reference = geography?.plans?.flatMap(plan => plan.stops?.features || []).find(row =>
      row.properties?.stop_id === stop.id && Array.isArray(row.properties.destination_places));
    if (!properties.destination_places && reference) properties.destination_places = reference.properties.destination_places;
    const acceptedTarget = selectedPlanId === acceptedPlanId
      && (state.workspace.draft_capabilities?.parent_targets || []).includes(stop.id);
    const metadata = properties.inspector_metadata?.schema_version === "plan-stop-inspector-v1"
      ? properties.inspector_metadata : {};
    const duration = metadata.duration_constraint || properties.duration_constraint || null;
    const dayStops = (stopPlan?.stops?.features || [])
      .filter(item => item.properties?.day === properties.day)
      .sort((left, right) => left.properties.sequence_index - right.properties.sequence_index);
    const dayPosition = dayStops.findIndex(item => item.properties?.stop_id === stop.id) + 1;
    const visitDuration = Number.isFinite(properties.visit_duration_minutes)
      ? `${formatMetric(properties.visit_duration_minutes)} min · plan value` : "Unavailable";
    const recordedContext = [
      typeof metadata.weather_feasible === "boolean"
        ? `Recorded weather feasibility: ${metadata.weather_feasible ? "feasible" : "not feasible"}` : null,
      Number.isFinite(metadata.weather_risk) ? `Recorded contextual risk: ${formatMetric(metadata.weather_risk)}` : null,
    ].filter(Boolean).join(" · ");
    const recordedSource = typeof metadata.source_list === "string" ? metadata.source_list
      : Array.isArray(metadata.source_list) && metadata.source_list.every(item => typeof item === "string")
        ? metadata.source_list.join(", ") : null;
    const sourceText = recordedSource
      ? `${humanLabel(recordedSource)}${Number.isFinite(metadata.source_confidence) ? ` · confidence ${formatMetric(metadata.source_confidence)}` : ""} · freshness unavailable`
      : null;
    const access = properties.selected_access_point || properties.route_access_point || null;
    const attributeConstraints = Array.isArray(properties.attribute_constraints)
      ? properties.attribute_constraints.map((row) => humanLabel(row.attribute || row)).join(", ")
      : "Unavailable";
    const placeLink = properties.official_url
      ? safeExternalLink(properties.official_url, "Official source")
      : safeExternalLink(properties.informational_url, "Informational source");
    const activeDraft = stopDraftOperations(stop.id);
    const selectionKey = `${selectedPlanId}:${stopPlan?.content_hash}:${stop.id}`;
    if (state.destinationDetails?.selectionKey !== selectionKey) {
      state.destinationDetails = {selectionKey, tab: "overview", expanded: false};
    }
    const information = destinationInformationHtml(properties, stop.name);
    const panels = {
      overview: `<p>${escapeHtml(properties.description || properties.guide?.intro || "A description has not been recorded for this destination.")}</p>${information.overview}`,
      visit: `<dl><dt>Itinerary role</dt><dd>${escapeHtml(unavailable(properties.itinerary_role, humanLabel))}</dd>
        <dt>Role source</dt><dd>${escapeHtml(unavailable(properties.itinerary_role_source, humanLabel))}</dd>
        <dt>Place categories</dt><dd>${escapeHtml(unavailable(
    Array.isArray(properties.place_categories) && properties.place_categories.length
      ? properties.place_categories.map(humanLabel).join(", ")
      : null,
  ))}</dd>
        <dt>Day and order</dt><dd>Day ${escapeHtml(properties.day ?? stop.day ?? "Unavailable")} · ${dayPosition > 0 ? `position ${dayPosition} in day` : "Unavailable"}</dd>
        <dt>Arrival</dt><dd>${escapeHtml(unavailable(properties.arrival_time || properties.expected_arrival))}</dd>
        <dt>Departure</dt><dd>${escapeHtml(unavailable(properties.departure_time || properties.expected_departure))}</dd>
        <dt>Visit duration</dt><dd>${escapeHtml(visitDuration)}</dd>
        <dt>Duration rule</dt><dd>${duration ? escapeHtml(formatDurationConstraint(duration)) : "Not specified"}</dd>
        <dt>Commitment</dt><dd>${escapeHtml(unavailable(properties.commitment_strength || properties.ownership_strength, humanLabel))}</dd>
        <dt>Attribute constraints</dt><dd>${escapeHtml(attributeConstraints)}</dd>
        <dt>Weather / context</dt><dd>${recordedContext ? `${escapeHtml(recordedContext)} · recorded plan context, not a live forecast` : "Unavailable"}</dd>
        <dt>Visit guidance</dt><dd>${escapeHtml(unavailable(properties.visit_guidance || properties.access_note || properties.guide?.access_name))}</dd>
        <dt>Route access</dt><dd>${access?.road_validated === true ? `${escapeHtml(humanLabel(access.access_type))} · road validated` : "Unavailable"}</dd></dl>
      <div class="selection-draft-state">${draftStateHtml(activeDraft)}</div>
      <p>Editing this stop creates a draft for review.</p>
      <div class="map-context-actions"><button type="button" id="inspect-edit-stop" ${acceptedTarget ? "" : "disabled title=\"Only an accepted-parent stop can be edited\""}>Open typed stop edits</button>
        <button type="button" id="prepare-replacement-candidates" ${acceptedTarget ? "" : "disabled title=\"Replacement discovery requires an accepted-parent stop\""}>Find a replacement</button></div>`,
      photos: information.photos,
      reviews: information.reviews,
      sources: `<dl><dt>Source / freshness</dt><dd>${sourceText ? escapeHtml(sourceText) : sourceFreshnessHtml(properties)}</dd>
        <dt>Place link</dt><dd>${placeLink}</dd></dl>${information.sources}`,
    };
    const labels = {overview: "Overview", visit: "Visit details", photos: "Photos", reviews: "Reviews", sources: "Sources"};
    return `<section class="map-context-card destination-details ${state.destinationDetails.expanded ? "is-expanded" : ""}" aria-labelledby="map-context-title">
      <header class="destination-heading"><div>
      <span class="status-pill">${selectedPlanId === displayedPlan?.plan_id ? "Selected stop" : "Original plan stop"}</span>
      <h3 id="map-context-title">${escapeHtml(stop.name)}</h3></div>
      <button type="button" id="expand-destination-details" aria-expanded="${state.destinationDetails.expanded}" aria-controls="destination-content">${state.destinationDetails.expanded ? "Compact view" : "Expand details"}</button></header>
      <div class="destination-tabs" role="tablist" aria-label="Destination details">${Object.keys(panels).map(key => `<button type="button" role="tab" id="destination-tab-${key}" data-destination-tab="${key}" aria-controls="destination-panel-${key}" aria-selected="${state.destinationDetails.tab === key}" tabindex="${state.destinationDetails.tab === key ? "0" : "-1"}">${labels[key]}</button>`).join("")}</div>
      <div class="destination-content" id="destination-content">${Object.entries(panels).map(([key, html]) => `<section role="tabpanel" id="destination-panel-${key}" aria-labelledby="destination-tab-${key}" tabindex="0" ${state.destinationDetails.tab === key ? "" : "hidden"}>${html}</section>`).join("")}</div>
    </section>`;
  }
  if (interaction.selectionKind === "route_leg") {
    if (state.interestRegions.enabled) return "";
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
        <button type="button" id="explore-interest-regions" ${selected?.plan_id === displayedPlan?.plan_id ? "" : "disabled"}>Explore interest region</button>
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

const YOSEMITE_SOURCE_URLS = {"yosemite_lower_fall": "https://www.nps.gov/yose/planyourvisit/lowerfalltrail.htm", "yosemite_cooks_meadow": "https://www.nps.gov/yose/planyourvisit/cooksmeadowtrail.htm", "yosemite_glacier_point": "https://www.nps.gov/yose/planyourvisit/glacierpoint.htm", "yosemite_sentinel_dome": "https://www.nps.gov/places/000/sentinel-dome-and-taft-point-trailhead.htm", "yosemite_mariposa_grove": "https://www.nps.gov/yose/planyourvisit/mg.htm", "yosemite_wawona_history": "https://www.nps.gov/yose/learn/historyculture/historycenter.htm", "yosemite_conditions": "https://www.nps.gov/yose/planyourvisit/conditions.htm"};
function validYosemiteGuide(guide) {
  if (guide?.schema_version!=="yosemite-stop-guide-v1" || typeof guide.intro!=="string" || !guide.intro
      || !Array.isArray(guide.visitor_reviews) || !guide.photo?.alt || !guide.photo?.credit) return false;
  try {
    const image=new URL(guide.photo.url);
    if (image.origin!=="https://www.nps.gov" || !["/yose/","/common/uploads/"].some(p=>image.pathname.startsWith(p))) return false;
    return guide.visitor_reviews.every(r=>typeof r.summary==="string" && r.author && new URL(r.source_url).origin==="https://www.reddit.com");
  } catch {return false;}
}
function destinationInformationHtml(properties, name) {
  const regional = Array.isArray(properties.destination_places) ? properties.destination_places : [];
  const places = (regional.length ? regional : [{attraction_name: name, guide: properties.guide, official_url: properties.official_url}])
    .filter(place => validYosemiteGuide(place.guide));
  const overview = regional.length && places.length ? `<h4>Places within this destination</h4><p class="truth-note">Reference guide only. These places have not been added to your itinerary. Opening hours and seasonal access need confirmation.</p>${places.map(place => `<details><summary>${escapeHtml(place.attraction_name)}</summary><p>${escapeHtml(place.guide.intro)}</p><p>Access: ${escapeHtml(place.guide.access_name)}</p>${safeExternalLink(place.official_url, "Official place information")}</details>`).join("")}` : "";
  const photos = places.map(place => {
    const photo = place.guide.photo;
    return `<figure><h4>${escapeHtml(place.attraction_name)}</h4><img loading="lazy" src="${escapeHtml(photo.url)}" alt="${escapeHtml(photo.alt)}" referrerpolicy="no-referrer"><figcaption>${escapeHtml(photo.caption_and_credit || photo.alt)}<br>${escapeHtml(photo.credit)} · ${safeExternalLink(photo.source_url, "Photo source")}</figcaption></figure>`;
  }).join("") || "<p>No sourced photos are available for this destination.</p>";
  const reviews = places.map(place => `<article><h4>${escapeHtml(place.attraction_name)}</h4>${place.guide.visitor_reviews.length ? place.guide.visitor_reviews.map(review => `<blockquote><p>${escapeHtml(review.summary)}</p><footer>${escapeHtml(review.author)} · ${escapeHtml(review.date_label)} · ${safeExternalLink(review.source_url, "Read visitor report")}</footer></blockquote>`).join("") : "<p>No verified visitor reports are available for this place.</p>"}<p class="truth-note">${escapeHtml(place.guide.review_disclosure)}</p></article>`).join("") || "<p>No verified visitor reports are available for this destination.</p>";
  const sources = places.map(place => `<p><strong>${escapeHtml(place.attraction_name)}</strong> · ${safeExternalLink(place.official_url, "Official place information")}</p>`).join("");
  return {overview, photos, reviews, sources};
}

function yosemiteGuideHtml(planId) {
  const bundle=state.session.proposal?.runtime_execution?.planning_bundle;
  if (bundle?.parameters?.additional_yosemite_days!==3 || bundle.plan.plan_id!==planId) return "";
  const guides=bundle.plan.selected_stops.filter(s=>s.guide);
  if (!guides.every(s=>validYosemiteGuide(s.guide))) return "";
  return `<section class="yosemite-guide" id="yosemite-guide" aria-label="新增三天優勝美地圖文"><h2>新增三天 · 優勝美地圖文導覽</h2><p>三天使用不同景點與主題；共同進出道路可能重疊。照片與個人遊記均附來源，飯店、季節道路和接駁須按實際日期再確認。</p><div class="yosemite-guide-grid">${guides.map(stop=>{const g=stop.guide;return `<article class="yosemite-guide-card" data-yosemite-stop="${escapeHtml(stop.poi_id)}"><p class="eyebrow">Day ${stop.day} · ${escapeHtml(g.theme)}</p><h3>${escapeHtml(stop.attraction_name)}</h3><figure><img src="${escapeHtml(g.photo.url)}" alt="${escapeHtml(g.photo.alt)}" referrerpolicy="no-referrer"><figcaption>${escapeHtml(g.photo.caption_and_credit || g.photo.alt)}<br>${escapeHtml(g.photo.credit)} · ${safeExternalLink(g.photo.source_url,"照片原頁")}</figcaption></figure><p>${escapeHtml(g.intro)}</p><p><strong>預留 ${stop.visit_duration_minutes} 分鐘</strong> · 車程終點：${escapeHtml(g.access_name)}</p><h4>遊客分享</h4>${g.visitor_reviews.length?g.visitor_reviews.map(r=>`<blockquote><p>${escapeHtml(r.summary)}</p><footer>${escapeHtml(r.author)} · ${escapeHtml(r.date_label)} · ${safeExternalLink(r.source_url,"閱讀原始遊記")}</footer></blockquote>`).join(""):"<p>目前未取得可核實的遊客評論。</p>"}<p class="truth-note">${escapeHtml(g.review_disclosure)}</p><p>${safeExternalLink(stop.official_url,"NPS 官方簡介")}</p><button type="button" data-yosemite-day="${stop.day}" data-yosemite-stop-id="${escapeHtml(stop.poi_id)}">在地圖查看 ${escapeHtml(stop.attraction_name)}</button></article>`}).join("")}</div></section>`;
}

function renderMap() {
  scheduleInterestRefresh();
  const interestButton = $("#map-interest-button");
  if (interestButton) {
    interestButton.classList.toggle("is-active", state.interestRegions.enabled);
    interestButton.setAttribute("aria-pressed", String(state.interestRegions.enabled));
    interestButton.title = "Explore nearby places for the displayed itinerary and day. Viewing does not create a draft.";
  }
  const textRouteOpen = $(".textual-map-alternative")?.open === true;
  const geography = activeGeography();
  const mapDraft = confirmedMapDraftState();
  const geographyReady = geography?.schema_version === "product-geography-v2"
    && ["ready", "ready_with_gaps"].includes(geography.status);
  const original = geographyReady ? geography.plans?.find((plan) => plan.role === "original") : null;
  const accepted = geographyReady ? acceptedPlan(geography) : null;
  const proposalEligible = state.session.proposal?.schema_version === "draft-preview-v1"
    && state.session.proposal?.state === "eligible"
    && Boolean(state.session.proposal?.child_plan_id);
  const selectedAlternative = geographyReady ? geography.plans?.find(
    (plan) => plan.plan_id === activeAlternativeId() && plan.plan_id !== accepted?.plan_id,
  ) : null;
  const displayed = selectedAlternative || accepted || original;
  const displayedStopIds = new Set(displayed?.stops?.features?.map((feature) => feature.properties?.stop_id) || []);
  const accessibleStops = [
    ...(displayed?.stops?.features || []).map((feature) => ({ feature, removed: false })),
    ...(original?.stops?.features || [])
      .filter((feature) => !displayedStopIds.has(feature.properties?.stop_id))
      .map((feature) => ({ feature, removed: true })),
  ];
  const dayRouteDescription = routeSummaryForPlan(displayed);
  const routeDescription = textualRouteEvidence(displayed, dayRouteDescription);
  const routeLegs = validatedRouteFeatures(displayed);
  const routeGaps = routeGapFeatures(displayed);
  const islandTrips = displayed?.marine_routes?.features || [];
  const islandNote = islandTrips.length ? `<p class="truth-note"><strong>Channel Islands · Day ${escapeHtml(islandTrips[0].properties.day)}</strong>: Drive to Ventura Harbor · Island Packers (1691 Spinnaker Dr), then passenger ferry to Santa Cruz Island / Scorpion Anchorage and back. Purple dashed line = schematic sea route. 480-minute allowance; sailing schedule, tickets and landing conditions remain unverified.</p>` : "";
  const yosemiteGuide = yosemiteGuideHtml(displayed?.plan_id);
  const removedByLabel = selectedAlternative ? "selected preview" : "active itinerary";
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
  const interest = currentInterestResponse();
  const routeCheckedOnly = hasRouteCheckedOnlyDraft();
  const impactSummary = state.typedImpact.response?.summary;
  const impactCurrent = state.typedImpact.response?.session_revision === state.session.revision;
  const evaluatedPreviewReady = impactCurrent && impactSummary?.can_run_evaluated_preview === true;
  const impactBlockerLabel = impactSummary?.blocking_codes?.length
    ? `Preview blocked: ${impactSummary.blocking_codes.map(humanLabel).join(", ")}.`
    : null;
  $("#workspace-content").innerHTML = `${yosemiteGuide ? '<button type="button" id="open-yosemite-guide">查看新增三天圖文</button>' : ""}<div class="map-stage" data-preview-state="${escapeHtml(state.previewRuntime.status)}" data-preview-request-epoch="${escapeHtml(state.previewRuntime.epoch)}" data-preview-request-id="${escapeHtml(state.previewRuntime.requestId || "")}">
    <div id="geographic-map" class="geographic-map" aria-hidden="true"></div>
    <div class="map-note">${islandTrips.length ? `<strong>Ventura ↔ Channel Islands · Day ${escapeHtml(islandTrips[0].properties.day)}</strong><br>Purple dashed sea route · schematic; sailing unverified.<br>` : ""}<strong>Day ${escapeHtml(state.session.selected_day)} highlighted</strong><br>${displayed?.plan_id === original?.plan_id ? `${escapeHtml(state.workspace.issue.label)} · ${escapeHtml(state.workspace.issue.summary)}` : "Current evaluated route shown with the original route for reference."}</div>
    ${mapInspectorHtml(geography, displayed)}
    ${candidateExplorerHtml()}
    ${interestMapSummaryHtml(interest)}
    <div class="map-accessibility-panel">
    <div id="map-render-status" class="map-render-status" role="status">
      ${geographyReady ? "Loading verified local geography…" : "Geographic evidence is unavailable. Use the text route description."}
    </div>
    <details class="textual-map-alternative" ${textRouteOpen ? "open" : ""}>
      <summary>Text route description</summary>
      ${textRoutePanelHtml(displayed, original)}
      <p>${escapeHtml(routeDescription)}</p>
      ${islandNote}
      ${stopButtons ? `<ol class="accessible-stop-list">${stopButtons}</ol>` : ""}
      ${routeButtons ? `<h3>Validated route legs</h3><ol class="accessible-route-list">${routeButtons}</ol>` : ""}
      ${customWaypointItems ? `<h3>Confirmed custom draft waypoints</h3><ol class="accessible-draft-waypoint-list">${customWaypointItems}</ol>` : ""}
      ${gapItems ? `<div class="route-gap-summary" role="alert"><h3>Required route gaps</h3><ul>${gapItems}</ul></div>` : ""}
    </details>
    </div>
    <div class="map-legend" aria-label="Map legend">
      <span><i class="line-swatch original"></i>Original</span>
      ${islandTrips.length ? `<span style="color:#7445b0;border-top:3px dashed #7445b0">Ferry round trip · schematic</span>` : ""}
      ${displayed?.plan_id && displayed.plan_id !== original?.plan_id ? `<span><i class="line-swatch repair"></i>${displayed.plan_id === accepted?.plan_id ? "Accepted route" : displayed.role === "draft_preview" ? "Evaluated preview" : "Selected alternative"}</span>` : ""}
      <span><i class="line-swatch selected-day"></i>Selected Day ${escapeHtml(state.session.selected_day)}</span>
      ${routeGaps.length ? `<span><i class="marker-swatch gap"></i>Unvalidated gap</span>` : ""}
      ${mapDraft.waypoints.length ? `<span><i class="marker-swatch draft-waypoint"></i>Route-checked draft waypoint</span><span><i class="line-swatch draft-route"></i>Affected draft route</span>` : ""}
      ${state.poiDiscovery.response?.candidates?.length ? `<span><i class="marker-swatch poi-candidate"></i>POI candidate · discovery only</span>` : ""}
      <span><i class="marker-swatch changed"></i>Selected-day stop</span>
    </div>
    ${mapAttribution(geography)}
    <div class="draft-bar">
      <strong>${draftCount} draft ${draftCount === 1 ? "change" : "changes"}</strong>
      <span>${state.previewRuntime.status === "PREVIEW_RUNNING" ? "Preview running. The accepted parent remains rendered until an eligible child and matching map acknowledgement are ready." : routeCheckedOnly ? "Route-checked draft; evaluated repair is not enabled for arbitrary waypoints yet." : proposalEligible ? `Evaluated ${escapeHtml(state.session.proposal.evaluation_status || state.session.proposal.eligibility || "proposal")} certificate is current for this draft.` : draftCount && impactCurrent ? impactBlockerLabel ? escapeHtml(impactBlockerLabel) : `${impactSummary.evaluated_executable_count} evaluated-preview capable · ${impactSummary.draft_only_count} draft only · not certified.` : draftCount ? "Checking server-owned draft impact; evaluation is disabled until it resolves." : state.mapInteraction.mode === "edit" ? "Map clicks remain exploratory until road checking and explicit confirmation." : "Select a stop or road-validated route leg."}</span>
      <button type="button" id="inline-undo" ${draftCount ? "" : "disabled"}>Undo latest draft</button>
      <button type="button" id="inline-preview" class="primary" ${draftCount && evaluatedPreviewReady && state.previewRuntime.status !== "PREVIEW_RUNNING" ? "" : `disabled title="${state.previewRuntime.status === "PREVIEW_RUNNING" ? "Preview request in progress" : draftCount ? escapeHtml(impactBlockerLabel || "This draft includes non-executable or not-yet-classified operations") : "Add a draft before preview"}"`}>${state.previewRuntime.status === "PREVIEW_RUNNING" ? "Preview running…" : "Preview repair"}</button>
    </div>
  </div>${yosemiteGuide}${interestExplorerHtml()}`;
  $("#open-yosemite-guide")?.addEventListener("click",()=>$("#yosemite-guide")?.scrollIntoView({block:"start"}));
  $$('[data-yosemite-day]').forEach(button=>button.addEventListener("click",async()=>{await selectMapStop({plan_id:displayed.plan_id,selected_day:Number(button.dataset.yosemiteDay),selected_stop_id:button.dataset.yosemiteStopId}); $("#workspace-content").scrollTop=0;}));
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
  $("#inspect-edit-stop")?.addEventListener("click", openMapEdit);
  const destinationTabs = $$("[data-destination-tab]");
  const selectDestinationTab = key => {
    state.destinationDetails.tab = key;
    destinationTabs.forEach(tab => {
      const selected = tab.dataset.destinationTab === key;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      $(`#destination-panel-${tab.dataset.destinationTab}`).hidden = !selected;
    });
    $("#destination-content").scrollTop = 0;
  };
  destinationTabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectDestinationTab(tab.dataset.destinationTab));
    tab.addEventListener("keydown", event => {
      const next = event.key === "Home" ? 0 : event.key === "End" ? destinationTabs.length - 1
        : event.key === "ArrowRight" ? (index + 1) % destinationTabs.length
        : event.key === "ArrowLeft" ? (index + destinationTabs.length - 1) % destinationTabs.length : null;
      if (next === null) return;
      event.preventDefault();
      selectDestinationTab(destinationTabs[next].dataset.destinationTab);
      destinationTabs[next].focus();
    });
  });
  $("#expand-destination-details")?.addEventListener("click", event => {
    state.destinationDetails.expanded = !state.destinationDetails.expanded;
    $(".destination-details").classList.toggle("is-expanded", state.destinationDetails.expanded);
    event.currentTarget.setAttribute("aria-expanded", String(state.destinationDetails.expanded));
    event.currentTarget.textContent = state.destinationDetails.expanded ? "Compact view" : "Expand details";
  });
  $("#prepare-replacement-candidates")?.addEventListener("click", prepareReplacementDiscovery);
  $("#find-route-candidates")?.addEventListener("click", fetchPoiCandidates);
  $("#explore-interest-regions")?.addEventListener("click", fetchInterestRegions);
  $("#refresh-interest-regions")?.addEventListener("click", fetchInterestRegions);
  $("#refresh-interest-place-data")?.addEventListener("click", () => fetchInterestRegions({refreshPlaces: true}));
  $("#retry-interest-regions")?.addEventListener("click", fetchInterestRegions);
  $("#interest-search-area")?.addEventListener("change", event => {
    state.interestRegions.searchArea = event.currentTarget.value;
    void fetchInterestRegions();
  });
  $("#interest-radius")?.addEventListener("change", event => {
    state.interestRegions.radiusKm = Number(event.currentTarget.value);
    void fetchInterestRegions();
  });
  $("#browse-interest-places")?.addEventListener("click", () => {
    $(".interest-region-panel")?.scrollIntoView({block: "start"});
    $("#interest-place-search")?.focus({preventScroll: true});
  });
  $("#show-whole-interest-region")?.addEventListener("click", () => selectInterestPlace(null));
  $("#interest-place-search")?.addEventListener("input", event => {
    state.interestRegions.placeSearch = event.currentTarget.value;
    $("#interest-place-results").innerHTML = interestPlaceListHtml(currentInterestResponse());
  });
  $("#interest-place-scope")?.addEventListener("change", event => {
    state.interestRegions.placeScope = event.currentTarget.value;
    $("#interest-place-results").innerHTML = interestPlaceListHtml(currentInterestResponse());
  });
  $(".interest-region-panel")?.addEventListener("click", event => {
    const button = event.target.closest("[data-interest-place]");
    if (!button) return;
    selectInterestPlace(button.dataset.interestPlace);
  });
  $("#close-interest-regions")?.addEventListener("click", () => { resetInterestRegions({hide: true}); renderMap(); });
  $("#interest-preference")?.addEventListener("change", event => {
    state.interestRegions.naturePreference = Number(event.currentTarget.value);
    void fetchInterestRegions();
  });
  $("#interest-mass-level")?.addEventListener("change", event => {
    state.interestRegions.massLevel = Number(event.currentTarget.value);
    saveInterestPreferences();
    renderMap();
  });
  $("#interest-place-selector")?.addEventListener("change", event => {
    selectInterestPlace(event.currentTarget.value || null);
  });
  $("#explore-interest-place")?.addEventListener("click", () => {
    const point = currentInterestResponse()?.poi_predictions.find(row => row.place_id === state.interestRegions.selectedId);
    if (point) createExploratoryPin(point.display_coordinate);
  });
  $("#relocate-custom-waypoint")?.addEventListener("click", () => {
    beginCustomWaypointRelocation(state.mapInteraction.selectedWaypointId);
  });
  $("#remove-custom-waypoint")?.addEventListener("click", () => {
    if (!window.confirm("Remove this latest staged waypoint? The accepted itinerary will remain unchanged.")) return;
    void undoDraft();
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
    interestRegions: currentInterestResponse(),
    interestMassLevel: state.interestRegions.massLevel,
    selectedInterestId: state.interestRegions.selectedId,
    onSelectInterest: selectInterestPlace,
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
  element.dataset.mapStatusCode = status.code || "unknown";
  element.classList.remove("is-warning", "is-error");
  const renderedPlan = activeRenderPlan();
  const stage = element.closest(".map-stage");
  if (stage) {
    if (renderedPlan) {
      stage.dataset.activePlanId = renderedPlan.plan_id;
      stage.dataset.activePlanContentHash = renderedPlan.content_hash;
    } else {
      delete stage.dataset.activePlanId;
      delete stage.dataset.activePlanContentHash;
    }
  }
  if (["ready", "ready_with_gaps"].includes(status.state)) {
    const expectedMarkerIds = (renderedPlan?.stops?.features || [])
      .map((feature) => feature.properties?.stop_id)
      .filter(Boolean)
      .sort();
    const acknowledgedMarkerIds = Array.isArray(status.activeMarkerIds)
      ? [...status.activeMarkerIds].sort() : [];
    const identityExact = Boolean(
      renderedPlan
      && status.activePlanId === renderedPlan.plan_id
      && status.activePlanContentHash === renderedPlan.content_hash
      && typeof status.activeGeometryHash === "string"
      && /^[0-9a-f]{64}$/.test(status.activeGeometryHash)
      && JSON.stringify(acknowledgedMarkerIds) === JSON.stringify(expectedMarkerIds),
    );
    if (!identityExact) {
      state.acknowledgedPreviewPlanId = null;
      state.previewRuntime.status = "MAP_IDENTITY_FAILED";
      if (stage) {
        stage.dataset.previewState = state.previewRuntime.status;
        delete stage.dataset.renderedMapPlanId;
        delete stage.dataset.renderedGeometryHash;
        delete stage.dataset.renderedMarkerIds;
      }
      element.hidden = false;
      element.classList.add("is-error");
      element.textContent = "The map rendered, but its plan identity could not be verified. The accepted itinerary remains authoritative.";
      copilot.previewStateChanged();
      renderContextPanel();
      return;
    }
    if (stage) {
      stage.dataset.renderedMapPlanId = status.activePlanId;
      stage.dataset.renderedGeometryHash = status.activeGeometryHash;
      stage.dataset.renderedMarkerIds = JSON.stringify(acknowledgedMarkerIds);
      if (activeProposalPlan()) state.previewRuntime.status = "PREVIEW_RENDERED";
      else if (!["PREVIEW_RUNNING", "PREVIEW_FAILED", "PREVIEW_INELIGIBLE", "PREVIEW_BLOCKED", "PREVIEW_NOT_EVALUATED"].includes(
        state.previewRuntime.status,
      )) state.previewRuntime.status = "PARENT_RENDERED";
      stage.dataset.previewState = state.previewRuntime.status;
      stage.dataset.previewRequestEpoch = String(state.previewRuntime.epoch);
      stage.dataset.previewRequestId = state.previewRuntime.requestId || "";
    }
    if (activeProposalPlan() && state.acknowledgedPreviewPlanId !== renderedPlan.plan_id) {
      state.acknowledgedPreviewPlanId = renderedPlan.plan_id;
      toast("Preview ready. The itinerary and map now show the same evaluated child plan.");
      copilot.previewStateChanged();
    }
    renderContextPanel();
    const notices = [];
    if (status.basemapOverviewOnly) notices.push("Overview basemap: this route extends beyond local detailed map coverage. Roads and stop positions are verified; local street detail is limited.");
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
  state.acknowledgedPreviewPlanId = null;
  if (activeProposalPlan() && state.previewRuntime.status !== "PREVIEW_RUNNING") {
    state.previewRuntime.status = status.state === "loading" ? "PREVIEW_DATA_READY" : "MAP_RENDER_FAILED";
  }
  if (stage) {
    stage.dataset.previewState = state.previewRuntime.status;
    delete stage.dataset.renderedMapPlanId;
    delete stage.dataset.renderedGeometryHash;
    delete stage.dataset.renderedMarkerIds;
  }
  element.hidden = false;
  element.classList.add(status.state === "loading" ? "is-warning" : "is-error");
  element.textContent = status.state === "loading"
    ? "Loading the current itinerary map…"
    : "The geographic map could not be verified. The text route description remains available.";
  copilot.previewStateChanged();
  renderContextPanel();
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
  const diff = proposal.plan_diff || proposal.runtime_execution?.plan_diff;
  const roleEvidence = diff?.schema_version === "plan-diff-v4";
  const schedule = proposal.schedule_impact || proposal.role_impact;
  const scheduleAccounting = roleEvidence ? schedule?.child_schedule : schedule?.child;
  const timeWindowEvidence = diff?.schema_version === "plan-diff-v3";
  const scheduleStatus = schedule
    ? `${humanLabel(scheduleAccounting.status)} · evaluator-owned modeled accounting`
    : proposal.evidence?.hard_feasibility_status === "PASSED"
      ? "Hard feasibility passed; stop-by-stop schedule accounting is Unavailable in this response."
      : "Unavailable";
  const durationChanges = diff?.duration_changes || [];
  const durationRows = durationChanges.map((change) => {
    const explicitFrom = change.from_minutes == null
      ? "Explicit parent duration Unavailable"
      : `${formatMetric(change.from_minutes)} min explicit parent duration`;
    const accountingFrom = change.accounting_from_minutes == null
      ? "accounting baseline Unavailable"
      : `${formatMetric(change.accounting_from_minutes)} min accounting baseline (${humanLabel(change.accounting_from_source)})`;
    return `<li><strong>${escapeHtml(stopNameForId(change.stop_id, acceptedPlan(activeGeography())))}</strong><span>${escapeHtml(explicitFrom)} · ${escapeHtml(accountingFrom)} → ${escapeHtml(formatMetric(change.to_minutes))} min exact · ${escapeHtml(formatMetric(change.delta_minutes))} min modeled change</span></li>`;
  }).join("");
  const timeWindowChanges = diff?.time_window_changes || [];
  const timeWindowRows = timeWindowChanges.map((change) => `
    <li><strong>${escapeHtml(stopNameForId(change.stop_id, acceptedPlan(activeGeography())))}</strong>
      <span>${escapeHtml(timeWindowLabel(change.from_constraint))} → ${escapeHtml(timeWindowLabel(change.to_constraint))} · trip-specific user constraint</span></li>`).join("");
  const roleChanges = diff?.role_changes || [];
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
  return `<details class="evaluated-edit-evidence">
    <summary>Evaluated edit evidence (advanced)</summary>
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
      ${durationRows ? `<section><h4>Typed duration diff</h4><ol>${durationRows}</ol><p>Duration changes are ${escapeHtml(diff.schema_version)} evidence; they are not time shifts or road changes.</p></section>` : ""}
      ${timeWindowRows ? `<section><h4>Typed time-window diff</h4><ol>${timeWindowRows}</ol><p>Time-window changes are ${escapeHtml(diff.schema_version)} trip constraints. They are not place opening hours, latest-start bounds, or time shifts.</p></section>` : ""}
      ${roleRows ? `<section><h4>Typed itinerary-role diff</h4><ol>${roleRows}</ol><p>Role changes are ${escapeHtml(diff.schema_version)} trip-use evidence. They do not change PlaceEntity identity or place categories; other edits in the draft may change road geometry or schedule.</p></section>` : ""}
      ${traceRows ? `<section><h4>Required-window schedule trace</h4><ol>${traceRows}</ol></section>` : ""}
      ${proposal.execution_mode === "trip_replan_independent_evaluation" ? tripPlanningEvidenceHtml(proposal) : ""}
      <section><h4>Independent certificate</h4>${outcome}<dl>
        <dt>Certificate</dt><dd>${escapeHtml(unavailable(proposal.certificate_id))}</dd>
        <dt>Evaluation</dt><dd>${escapeHtml(unavailable(proposal.evaluation_status))}</dd>
        <dt>Artifact grounding</dt><dd>${escapeHtml(unavailable(proposal.evidence?.artifact_grounding_status))}</dd>
      </dl></section>
    </div>
    <p class="truth-note">This is a route-validated, independently evaluated session preview. Accepting it requires explicit confirmation and a fresh server-side revalidation against the immutable parent, diff, route evidence, and certificate.</p>
  </details>`;
}

function tripPlanningEvidenceHtml(proposal) {
  const value = proposal.runtime_execution;
  const solver = value?.solver || {};
  const sources = (value?.official_sources || []).map(row => `<li><a href="${escapeHtml(row.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(row.source_id)}</a><p>Retrieved ${escapeHtml(row.retrieved_at)} · ${escapeHtml(row.park_access || row.status)}</p><p>${escapeHtml(row.excerpt)}</p></li>`).join("");
  const notices = (value?.notices || []).map(text => `<li>${escapeHtml(text)}</li>`).join("");
  return `<section><h4>Trip planning result</h4><p>${solver.status === "OPTIMAL" ? "Optimal within this finite candidate model" : "Feasible plan; optimality not proven"} · ${escapeHtml(solver.objective)}.</p><p>${escapeHtml(solver.scope)}. Gap: ${escapeHtml(solver.optimality_gap == null ? "Unavailable" : solver.optimality_gap)}.</p><h4>Official travel sources</h4><ul>${sources}</ul><ul>${notices}</ul></section>`;
}

function normalizeAcceptanceAction(value, expectedDecisionType) {
  const pointerKeys = [
    "schema_version", "trip_id", "run_id", "original_plan_id", "original_plan_content_hash",
    "accepted_plan_id", "accepted_plan_content_hash", "decision_id", "revision", "updated_at", "content_hash",
  ];
  const decisionKeys = [
    "schema_version", "decision_id", "idempotency_key", "decision_type", "trip_id", "run_id",
    "session_id", "session_revision_before", "session_revision_after", "pointer_revision_before",
    "pointer_revision_after", "previous_plan_id", "previous_plan_content_hash", "selected_plan_id",
    "selected_plan_content_hash", "proposal_id", "draft_content_hash", "diff_id", "diff_content_hash",
    "certificate_id", "certificate_content_hash", "snapshot_content_hash", "created_at",
    "request_fingerprint", "content_hash",
  ];
  const sessionKeys = [
    "schema_version", "session_id", "trip_id", "run_id", "revision", "accepted_plan_id",
    "selected_day", "selected_stop_id", "selected_segment_id", "selected_candidate_id",
    "selected_alternative_id", "draft", "proposal", "permission_decisions", "conversation_id",
  ];
  if (!exactKeys(value, ["schema_version", "result", "session", "workspace"])
      || value.schema_version !== "acceptance-action-v1"
      || !value.workspace || typeof value.workspace !== "object"
      || value.workspace.accepted_plan_pointer?.accepted_plan_id !== value.session?.accepted_plan_id
      || !exactKeys(value.result, ["schema_version", "decision", "pointer", "recovery_performed"])
      || value.result.schema_version !== "acceptance-result-v1"
      || typeof value.result.recovery_performed !== "boolean"
      || !exactKeys(value.result.pointer, pointerKeys)
      || !exactKeys(value.result.decision, decisionKeys)
      || !exactKeys(value.session, sessionKeys)) return null;
  const { decision, pointer } = value.result;
  if (pointer.schema_version !== "workspace-pointer-v1"
      || decision.schema_version !== "acceptance-decision-v1"
      || decision.decision_type !== expectedDecisionType
      || !/^decision_[0-9a-f]{32}$/.test(decision.decision_id)
      || pointer.decision_id !== decision.decision_id
      || decision.session_id !== value.session.session_id
      || decision.trip_id !== value.session.trip_id
      || decision.run_id !== value.session.run_id
      || decision.session_revision_after !== decision.session_revision_before + 1
      || decision.pointer_revision_after !== decision.pointer_revision_before + 1
      || pointer.revision !== decision.pointer_revision_after
      || pointer.accepted_plan_id !== decision.selected_plan_id
      || pointer.accepted_plan_content_hash !== decision.selected_plan_content_hash
      || value.session.revision !== decision.session_revision_after
      || value.session.accepted_plan_id !== pointer.accepted_plan_id
      || !Array.isArray(value.session.draft) || value.session.draft.length !== 0
      || value.session.proposal !== null
      || !Array.isArray(value.session.permission_decisions)
      || value.session.permission_decisions.length !== 0
      || !/^[0-9a-f]{64}$/.test(pointer.content_hash)
      || !/^[0-9a-f]{64}$/.test(decision.content_hash)
      || !/^[0-9a-f]{64}$/.test(decision.request_fingerprint)) return null;
  return value;
}

function previewCanBeAccepted() {
  const proposal = state.session?.proposal;
  return !state.workspaceSyncPending && !state.workspaceConflict && proposal?.state === "eligible" && proposal.acceptance_eligible === true
    && state.previewRuntime.status === "PREVIEW_RENDERED"
    && state.acknowledgedPreviewPlanId === proposal.child_plan_id;
}

async function submitAcceptance(decisionType) {
  if (pendingSelection && !(await pendingSelection)) return toast("Selection could not be updated. No acceptance was sent.", true);
  const acceptRepair = decisionType === "accept_repair";
  if (acceptRepair && !previewCanBeAccepted()) {
    toast("Wait for the eligible preview to finish rendering before accepting it.", true);
    return;
  }
  const dialog = $("#acceptance-dialog");
  if (dialog.open || dialog.dataset.submitting) return;
  const sessionId = state.session.session_id;
  const revision = state.session.revision;
  const parentPlanId = state.session.accepted_plan_id;
  const conversationId = state.session.conversation_id;
  const childPlanId = state.session.proposal?.child_plan_id;
  const childContentHash = state.session.proposal?.child_plan_content_hash;
  $("#acceptance-message").textContent = acceptRepair
    ? "Accept this independently evaluated repair as the durable trip plan?"
    : "Keep the immutable original plan as the durable trip plan?";
  $("#acceptance-confirm").textContent = acceptRepair ? "Accept repair" : "Keep original";
  dialog.returnValue = "cancel";
  dialog.dataset.submitting = "true";
  try {
    const confirmed = await new Promise((resolve) => {
      dialog.addEventListener("close", () => resolve(dialog.returnValue === "confirm"), { once: true });
      dialog.showModal();
    });
    if (!confirmed) return;
    if (state.session.session_id !== sessionId || state.session.revision !== revision
        || state.session.accepted_plan_id !== parentPlanId
        || (acceptRepair && (!previewCanBeAccepted()
          || state.session.proposal.child_plan_id !== childPlanId
          || state.session.proposal.child_plan_content_hash !== childContentHash))) {
      toast("The trip or preview changed while confirmation was open. Review it and confirm again.", true);
      return;
    }
    const confirmation = acceptRepair ? "accept_evaluated_repair" : "keep_original_plan";
    const endpoint = acceptRepair ? "accept" : "keep-original";
    const payload = await api(`/api/sessions/${encodeURIComponent(sessionId)}/${endpoint}`, {
      method: "POST",
      body: {
        expected_revision: revision,
        confirmation,
        idempotency_key: `${endpoint}:${crypto.randomUUID()}`,
      },
    });
    const normalized = normalizeAcceptanceAction(payload, decisionType);
    if (!normalized) throw new Error("invalid_acceptance_response");
    if (state.session?.session_id !== sessionId || state.session.revision !== revision
        || state.session.accepted_plan_id !== parentPlanId || state.session.conversation_id !== conversationId) {
      toast("The trip changed while the decision was saving. Reload that trip to review its saved decision.", true);
      return;
    }
    state.session = normalized.session;
    state.workspace = normalized.workspace;
    syncPreviewRuntimeFromSession();
    state.typedImpact = { status: "idle", response: null, error: null };
    renderAll();
    toast(acceptRepair
      ? "Evaluated repair accepted. The immutable plan pointer and decision ledger were updated."
      : "Original plan kept. The durable decision was recorded.");
  } catch (error) {
    if (!(error.status === 409 && error.code === "stale_session_revision" && await refreshStaleWorkspace(error))) toast(error.message, true);
  } finally {
    delete dialog.dataset.submitting;
  }
}
function renderRepair() {
  const issue = state.workspace.issue;
  const proposal = state.workspaceSyncPending ? null : state.session.proposal;
  const evaluated = proposal?.schema_version === "draft-preview-v1" && ["eligible", "ineligible"].includes(proposal.state);
  const eligible = evaluated && proposal.state === "eligible";
  const registered = proposal?.state === "eligible" && proposal?.provenance === "registered_fixture_artifacts";
  const unevaluated = proposal?.schema_version === "draft-preview-v1" && !evaluated && !registered;
  const acceptedBaseline = acceptedPlan(state.workspace.geography);
  const hasAcceptedRepair = state.workspace.accepted_plan_pointer?.decision_id
    && acceptedBaseline?.content_hash === state.workspace.accepted_plan_pointer.accepted_plan_content_hash
    && planPresentationEvidence(acceptedBaseline).certificateId;
  const acceptedChild = !proposal && hasAcceptedRepair && activeRenderPlan()?.plan_id === acceptedBaseline.plan_id;
  const displayed = activeRenderPlan();
  const hasRenderedRepair = hasAcceptedRepair || (eligible && displayed?.plan_id === proposal.child_plan_id
    && displayed?.content_hash === proposal.child_plan_content_hash);
  const canAccept = eligible && previewCanBeAccepted();
  const repair = evaluated || registered ? proposal.repair : state.workspace.repair;
  const statusLabel = eligible ? "Evaluated eligible proposal" : evaluated ? "Evaluated ineligible proposal" : unevaluated ? "No independent evaluation" : acceptedChild ? "Accepted evaluated itinerary" : registered ? "Registered fixture proposal" : "Preview required";
  const repairTitle = eligible ? "Evaluated draft repair" : evaluated ? "No eligible draft repair" : unevaluated ? "Preview did not produce an evaluated child" : acceptedChild ? "Accepted repair is displayed" : registered ? "Registered repair fixture" : "Recommendation — not applied";
  const repairBody = unevaluated
    ? `<ul>${(proposal.blocking_findings || []).map((finding) => `<li>${escapeHtml(humanLabel(finding.code))}: ${escapeHtml(finding.origin_id || "")} ${finding.destination_id ? `→ ${escapeHtml(finding.destination_id)}` : ""}. ${escapeHtml(finding.corrective_action)}</li>`).join("")}</ul><p>The staged draft is retained. The accepted itinerary and map have not been replaced.</p>`
    : acceptedChild ? `<p>The saved accepted child is the displayed itinerary. Its metrics and certificate come from the immutable acceptance snapshot.</p>`
    : !evaluated && !registered
    ? `<p>This registered recommendation is not the displayed itinerary. Run Preview repair to generate and evaluate your draft before accepting any changes.</p>`
    : repair ? `
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
      <h3>${hasRenderedRepair ? "Original repair context" : "Issue detected · Registered weather disruption"}</h3>
      <p>${escapeHtml(issue.summary)}</p>
    </article>
    <article class="repair-card">
      <h3>${repairTitle}</h3>
      ${proposal?.message ? `<p class="truth-note">${escapeHtml(proposal.message)}</p>` : ""}
      ${repairBody}
      ${evaluatedEditEvidenceHtml(proposal)}
    </article>
    <div class="action-stack">
      <button class="primary" id="accept-repair" ${canAccept ? "" : "disabled"}>Accept evaluated repair</button>
      <button id="compare-repair">Compare alternatives</button>
      <button id="review-evidence">Review evidence</button>
      <button id="keep-original">Keep original</button>
    </div>
    <p class="truth-note">Accept and Keep original require explicit confirmation. Decisions append to the durable ledger and atomically update the accepted-plan pointer; immutable plan artifacts are never overwritten. Rollback remains a separate audited action.</p>`;
  $("#compare-repair").addEventListener("click", () => navigate("/app/compare"));
  $("#review-evidence").addEventListener("click", () => navigate("/app/evidence"));
  $("#accept-repair").addEventListener("click", () => submitAcceptance("accept_repair"));
  $("#keep-original").addEventListener("click", () => submitAcceptance("keep_original"));
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
  const context = evaluatedProposalCompareContext();
  const resolution = resolveCompareSelection(
    context?.workspace || state.workspace,
    context?.session || state.session,
  );
  const option = resolution.state === "selected" ? resolution.option : null;
  $("#repair-panel").innerHTML = `<span class="eyebrow">Selected alternative</span><h2 id="repair-heading">Tradeoff</h2>
    ${option ? `<span class="status-pill success">${escapeHtml(option.display_status || option.status || "Selected")}</span><article class="repair-card"><h3>${escapeHtml(option.method_label || "Selected repair")}</h3><dl><dt>Route time</dt><dd>${escapeHtml(option.route_total_minutes == null ? "Unavailable" : `${formatMetric(option.route_total_minutes)} minutes`)}</dd><dt>Edit cost</dt><dd>${escapeHtml(formatMetric(option.weighted_edit_cost))}</dd><dt>Road route</dt><dd>${option.route_validated ? "Validated" : "Unavailable"}</dd></dl><p>The maps, metrics, changes, and Evidence button refer to this same selected repair.</p></article>` : '<article class="repair-card"><h3>Choose a repair</h3><p>Select one available option in Compare.</p></article>'}
    <div class="action-stack"><button id="context-choose" class="primary" type="button" ${option ? "" : "disabled"} title="Selects this option for Trip preview; acceptance requires an eligible evaluated preview and confirmation">Choose option for Trip preview</button><button id="context-evidence" type="button" ${resolution.evidenceExact ? "" : "disabled"}>View advanced Evidence</button></div>`;
  $("#context-choose").addEventListener("click", () => {
    if (!option) return;
    state.tripPreviewPlanId = option.plan_id;
    navigate("/app/itinerary");
  });
  $("#context-evidence").addEventListener("click", () => navigate("/app/evidence"));
}

function renderEvidenceContextPanel() {
  const context = evidencePresentationContext();
  const resolution = resolveCompareSelection(
    context?.workspace || state.workspace, context?.session || state.session,
  );
  const proposal = state.session.proposal;
  const evaluated = proposal?.schema_version === "draft-preview-v1"
    && ["eligible", "ineligible"].includes(proposal.state);
  const exact = resolution.state === "selected" && resolution.evidenceExact;
  $("#repair-panel").innerHTML = `<span class="eyebrow">Evidence summary</span><h2 id="repair-heading">Eligibility checks</h2>
    <span class="status-pill ${evaluated || exact ? "success" : "warning"}">${evaluated ? escapeHtml(humanLabel(proposal.state)) : exact ? context?.scope === "accepted" ? "Exact accepted-plan evidence" : "Exact selected evidence" : "Selection required"}</span>
    <article class="repair-card"><h3>Current scope</h3><ul class="change-list"><li>Exact selected option: ${exact ? "verified" : "Unavailable"}</li><li>Road validation: ${(evaluated ? proposal.route_validation?.publication_ready : exact) ? "checked" : "Unavailable"}</li><li>Acceptance requires an eligible rendered preview and explicit confirmation in Trip.</li></ul><p>Technical plan, hash, diff, method, and certificate details are shown in the main Evidence view.</p></article>
    <div class="action-stack"><button id="context-back-compare" type="button">Back to Compare</button></div>`;
  $("#context-back-compare").addEventListener("click", () => navigate("/app/compare"));
}

function renderMetrics() {
  const plan = activeRenderPlan();
  const facts = planPresentationEvidence(plan);
  const preservation = facts.repair?.tradeoffs?.preservation_rate ?? facts.metrics.preservation_rate;
  const risk = facts.repair?.tradeoffs?.weather_risk_delta ?? facts.metrics.weather_risk_delta;
  const routeMinutes = planRouteMinutes(plan);
  const scope = facts.certificateId ? "Exact displayed-plan evidence" : "Not evaluated for this plan";
  const metrics = [
    ["Route time", routeMinutes === null ? "Unavailable" : `${formatMetric(routeMinutes)} min`, "Exact road-evidence duration"],
    ["Contextual risk", risk == null ? "Unavailable" : formatMetric(risk), scope],
    ["Preservation", preservation == null ? "Unavailable" : `${Math.round(preservation * 100)}%`, scope],
    ["Edit cost", facts.weightedEditCost == null ? "Unavailable" : formatMetric(facts.weightedEditCost), scope],
    ["Certificate", facts.eligibility, scope],
  ];
  const strip = $("#metric-strip");
  // Compare and Evidence render their own selected-artifact summaries. Keep
  // the Trip summary from appearing as an unlabeled competing plan there.
  strip.hidden = state.route === "/app/compare" || state.route === "/app/evidence";
  strip.dataset.planId = plan?.plan_id || "";
  strip.dataset.planContentHash = plan?.content_hash || "";
  strip.dataset.evaluationId = facts.certificateId || "";
  strip.innerHTML = metrics.map(([label, value, note]) => `<div class="metric" data-plan-id="${escapeHtml(plan?.plan_id || "")}" data-plan-content-hash="${escapeHtml(plan?.content_hash || "")}"><span>${label}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(note)}</small></div>`).join("");
}

function evaluatedProposalCompareContext() {
  const proposal = state.session.proposal;
  const plan = activeProposalPlan();
  if (!plan || proposal?.child_plan_id !== plan.plan_id) return null;
  const option = {
    plan_id: proposal.child_plan_id,
    plan_content_hash: proposal.child_plan_content_hash,
    role: "evaluated_preview",
    status: proposal.state,
    display_status: proposal.state === "eligible" ? "Eligible" : "Ineligible",
    ranking_eligible: proposal.ranking_eligible === true,
    method_label: "Evaluated draft repair",
    route_total_minutes: planRouteMinutes(plan),
    weighted_edit_cost: proposal.repair?.tradeoffs?.weighted_edit_cost ?? null,
    route_validated: proposal.route_validation?.publication_ready === true,
  };
  const evidence = {
    parent_plan_id: proposal.parent_plan_id,
    plan_id: proposal.child_plan_id,
    plan_content_hash: proposal.child_plan_content_hash,
    diff_id: proposal.diff_id,
    certificate_id: proposal.certificate_id,
    certificate: proposalCertificate(proposal),
    diff: proposal.runtime_execution?.plan_diff || proposal.plan_diff || null,
    source_hashes: {child: proposal.child_plan_content_hash, diff: proposal.diff_content_hash,
      certificate: proposal.certificate_content_hash},
    methods: {requested: [proposal.evidence?.requested_method || "Unavailable"],
      executed: [proposal.evidence?.executed_method || proposal.execution_mode]},
  };
  return {
    scope: "preview",
    workspace: {
      ...state.workspace,
      alternatives: [option],
      alternative_evidence: { [proposal.child_plan_id]: evidence },
      geography: activeGeography(),
    },
    session: { ...state.session, selected_alternative_id: proposal.child_plan_id },
  };
}

function evidencePresentationContext() {
  const preview = evaluatedProposalCompareContext();
  if (preview) return preview;
  // Explicit Compare selection wins; otherwise inspect the exact accepted snapshot.
  if (state.session.selected_alternative_id) return null;
  const pointer = state.workspace.accepted_plan_pointer;
  const id = state.session.accepted_plan_id;
  const plans = activeGeography()?.plans?.filter(plan => plan.plan_id === id) || [];
  const evidence = state.workspace.alternative_evidence?.[id];
  const certificate = evidence?.certificate;
  if (!pointer?.decision_id || pointer.accepted_plan_id !== id || plans.length !== 1
      || plans[0].content_hash !== pointer.accepted_plan_content_hash
      || evidence?.plan_id !== id || evidence.plan_content_hash !== plans[0].content_hash
      || certificate?.plan_id !== id || certificate.plan_content_hash !== plans[0].content_hash
      || certificate.certificate_id !== evidence.certificate_id
      || evidence.diff?.child_plan_id !== id || evidence.diff.diff_id !== evidence.diff_id
      || evidence.diff.parent_plan_id !== evidence.parent_plan_id) return null;
  return {
    scope: "accepted",
    workspace: { ...state.workspace, geography: activeGeography(), alternatives: [{
      plan_id: id, plan_content_hash: plans[0].content_hash, role: "accepted",
      method_label: "Accepted evaluated itinerary", status: certificate.comparison_eligibility,
      display_status: "Accepted", ranking_eligible: false,
    }] },
    session: { ...state.session, selected_alternative_id: id },
  };
}

function renderCompare() {
  const context = evaluatedProposalCompareContext();
  renderCompareWorkspace({
    container: $("#workspace-content"),
    workspace: context?.workspace || state.workspace,
    session: context?.session || state.session,
    mapConfig: state.mapConfig,
    onBack: () => navigate("/app/itinerary"),
    onSelectAlternative: context
      ? () => {}
      : (planId) => selectContext({ selected_alternative_id: planId }),
    onChooseAlternative: (planId) => { state.tripPreviewPlanId = planId; },
    onOpenEvidence: (planId) => {
      if (context?.session.selected_alternative_id === planId
          || state.session.selected_alternative_id === planId) navigate("/app/evidence");
    },
  });
}

async function downloadEvidenceBundle(runId, planId, button) {
  if (button.disabled) return;
  button.disabled = true;
  let objectUrl;
  let download;
  try {
    const headers = {Accept: "application/zip", "X-Product-Build-Id": state.clientBuildId || ""};
    if (state.token) headers["X-Session-Token"] = state.token;
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/evidence-bundle?plan_id=${encodeURIComponent(planId)}`, {headers});
    if (!response.ok) {
      const payload = (response.headers.get("content-type") || "").includes("application/json") ? await response.json() : null;
      if (isBuildMismatch(payload?.detail)) renderBuildRecovery(payload.detail);
      throw new Error("evidence_download_failed");
    }
    objectUrl = URL.createObjectURL(await response.blob());
    download = document.createElement("a");
    download.href = objectUrl;
    download.download = `${runId}-${planId}-evidence.zip`.replace(/[^\w.-]/g, "_");
    document.body.append(download);
    download.click();
  } catch {
    toast("Evidence could not be downloaded. Please try again.", true);
  } finally {
    download?.remove();
    if (objectUrl) setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    button.disabled = false;
  }
}

function renderEvidence() {
  const context = evidencePresentationContext();
  const resolution = resolveCompareSelection(context?.workspace || state.workspace, context?.session || state.session);
  const proposalMatches = context?.scope === "preview";
  const selectedAlternative = resolution.state === "selected" && resolution.evidenceExact
    ? resolution.option
    : null;
  if (!selectedAlternative) {
    const code = state.session.selected_alternative_id
      ? resolution.code
      : "evidence_option_not_selected";
    $("#workspace-content").innerHTML = `<div class="route-view">
      <header class="route-heading"><div><span class="eyebrow">Structured evidence</span><h2>Evidence unavailable</h2><p>Evidence is shown only for the exact option selected in Compare. No first-option fallback was used.</p></div><button class="secondary" id="evidence-back">Back to Compare</button></header>
      <article class="evidence-card"><h3>Select an artifact-backed option</h3><p>Open Compare, select one exact plan/hash, then use its Evidence button.</p><code>${escapeHtml(code)}</code></article>
    </div>`;
    $("#evidence-back").addEventListener("click", () => navigate("/app/compare"));
    return;
  }
  const selectedEvidence = resolution.evidence;
  const selectedPlan = resolution.plan;
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
  $("#workspace-content").innerHTML = `<div class="route-view" data-evidence-scope="${context?.scope || "selected_option"}" data-evidence-plan-id="${escapeHtml(selectedPlan.plan_id)}" data-plan-content-hash="${escapeHtml(selectedPlan.content_hash)}" data-evaluation-id="${escapeHtml(certificate?.certificate_id || "")}">
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
        ${proposalMatches ? '<p class="truth-note">This is evaluated preview evidence, not an accepted change. Explicit acceptance persists the child and its lineage.</p>' : context?.scope === "accepted" ? '<p class="truth-note">This is the persisted accepted itinerary, with its exact inspected certificate and parent-child diff. It is not a new preview.</p>' : '<button class="download-link" id="evidence-download" type="button">Download selected-plan evidence</button>'}
      </article>
      <article class="evidence-card"><h3>4 · Requested and executed methods</h3>
        <details open><summary>Requested methods</summary><p>${escapeHtml((methods.requested || []).join(", ") || "Unavailable")}</p></details>
        <details><summary>Executed methods</summary><p>${escapeHtml((methods.executed || []).join(", ") || "Unavailable")}</p></details>
      </article>
      <article class="evidence-card"><h3>5 · Technical lineage</h3><dl>${objectDl(lineage)}</dl></article>
      <article class="evidence-card"><h3>6 · Exact selected-plan metrics</h3><dl>${objectDl(certificate?.metrics || {})}</dl><p>Values come from the selected plan's evaluator certificate; compare against the accepted baseline in Compare.</p></article>
    </div>
  </div>`;
  $("#evidence-back").addEventListener("click", () => navigate("/app/itinerary"));
  const runId = state.session.run_id;
  $("#evidence-download")?.addEventListener("click", event => downloadEvidenceBundle(runId, selectedAlternative.plan_id, event.currentTarget));
}

function renderRepairHistory() {
  const proposal = state.session.proposal;
  $("#workspace-content").innerHTML = `<div class="route-view">
    <header class="route-heading"><div><span class="eyebrow">Session state</span><h2>Repairs and drafts</h2><p>Local session revision ${state.session.revision}</p></div><button class="secondary" id="repairs-back">Back to trip</button></header>
    <div class="evidence-grid">
      <article class="evidence-card"><h3>Validated session baseline</h3><p><code>${escapeHtml(state.session.accepted_plan_id)}</code></p></article>
      <article class="evidence-card"><h3>Current proposal</h3><p>${escapeHtml(proposal?.message || "No evaluated proposal in this session.")}</p></article>
      <article class="evidence-card"><h3>Typed draft operations</h3>${state.session.draft.length ? `<ol>${state.session.draft.map((op) => `<li><strong>${escapeHtml(operationDisplayName(op.type))}</strong><span>${escapeHtml(draftTargetLabel(op))}</span></li>`).join("")}</ol>` : "<p>No active draft operations.</p>"}</article>
      <article class="evidence-card"><h3>Decision boundary</h3><p>Legacy decision files are preserved but not trusted or extended. Eligible evaluated repairs require explicit confirmation and are recorded in the durable W5 ledger. Copilot transcripts expire after 30 days without a new turn; separately saved test evidence is retained independently.</p></article>
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
    return `<section class="typed-impact is-blocked" role="status"><strong>Draft impact unavailable</strong><p>${escapeHtml(state.typedImpact.error || "The server-owned impact preview has not loaded.")}</p><button type="button" class="secondary" data-retry-draft-impact>Retry draft impact check</button></section>`;
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
    <p class="truth-note" data-combination-policy>Preview repair evaluates the complete supported draft as one proposed itinerary. Conflicting or draft-only changes block preview; protected rules still apply.</p>
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
  surface.querySelector("[data-retry-draft-impact]")?.addEventListener("click", async () => {
    if (state.typedImpact.status === "loading") return;
    const pending = loadDraftImpactPreview({ render: false });
    const request = state.typedImpact.request;
    renderAll();
    refreshMapEditDialog();
    await pending;
    if (!draftImpactRequestIsCurrent(request)) return;
    renderAll();
    if ($("#map-edit-dialog").open) refreshMapEditDialog();
  });
  surface.querySelectorAll(".typed-operation-form").forEach((form) => {
    form.addEventListener("submit", submitTypedOperation);
    const type = form.dataset.typedOperation;
    const submit = form.querySelector('button[type="submit"]');
    if (!submit) return;
    const intrinsicDisabled = submit.disabled || !typedEditCapability(type)?.enabled;
    const originalTitle = submit.title;
    const updateCombination = () => {
      const parameters = type === "set_stop_duration" ? { duration: { mode: form.elements.duration_mode?.value } } : {};
      const blocker = draftCombinationBlocker([...state.session.draft, { type, parameters }]);
      submit.disabled = intrinsicDisabled || Boolean(blocker);
      submit.title = blocker ? "This edit cannot be combined with the staged draft. Preview and accept, or undo, the current edit first."
        : originalTitle;
      if (blocker) form.dataset.combinationBlockingCode = blocker;
      else delete form.dataset.combinationBlockingCode;
    };
    form.addEventListener("change", updateCombination);
    updateCombination();
  });
  const attributeForm = surface.querySelector('[data-typed-operation="set_attribute_constraint"]');
  if (attributeForm) {
    renderAttributeValueEditor(attributeForm);
    attributeForm.elements.attribute.addEventListener("change", () => renderAttributeValueEditor(attributeForm));
  }
  bindDurationModeSelects(surface);
  bindRoleSelects(surface);
}

async function updateSelection(patch) {
  if (state.workspaceConflict || state.workspaceSyncPending) return false;
  const origin = { ...state.session };
  const ownsSession = (session) => session
    && ["session_id", "trip_id", "run_id"].every((field) => session[field] === origin[field])
    && Number.isSafeInteger(session.revision) && session.revision >= 0;
  let saved = false;
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

    let payload = await mutationRecovery.submit("selection", {
        expected_revision: state.session.revision,
        selected_day: Object.hasOwn(patch, "selected_day") ? patch.selected_day : state.session.selected_day,
        selected_stop_id: selectedStopId,
        selected_segment_id: Object.hasOwn(patch, "selected_segment_id") ? patch.selected_segment_id : state.session.selected_segment_id,
        selected_candidate_id: selectedCandidateId,
        selected_alternative_id: Object.hasOwn(patch, "selected_alternative_id") ? patch.selected_alternative_id : state.session.selected_alternative_id,
    });
    if (!ownsSession(payload?.session)) throw new Error("The selection response did not match its workspace.");
    saved = true;
    if (!ownsSession(state.session)) throw new Error("The visible workspace changed while saving selection.");
    if (state.session.conversation_id !== origin.conversation_id
        || payload.session.revision < state.session.revision) {
      // Conversation deletion/rebinding does not advance itinerary revision.
      // Read the committed selection and current conversation together; never
      // merge a stale response's trip fields with guessed conversation metadata.
      let reconciled = false;
      for (let attempt = 0; attempt < 3; attempt += 1) {
        const beforeRead = { ...state.session };
        payload = await api(`/api/sessions/${encodeURIComponent(origin.session_id)}`);
        if (!ownsSession(state.session) || !ownsSession(payload?.session)) {
          throw new Error("The selection refresh did not match its workspace.");
        }
        if (state.session.revision !== beforeRead.revision
            || state.session.conversation_id !== beforeRead.conversation_id
            || payload.session.revision < state.session.revision) continue;
        reconciled = true;
        break;
      }
      if (!reconciled) throw new Error("The workspace kept changing. Refresh to read the saved selection.");
      if (payload.workspace) state.workspace = payload.workspace;
      if (payload.registry) state.registry = payload.registry;
    }
    if (payload.workspace) state.workspace = payload.workspace;
    if (payload.registry) state.registry = payload.registry;
    state.session = payload.session;
    syncPreviewRuntimeFromSession();
    await loadDraftImpactPreview({ render: false });
    renderAll();
    return true;
  } catch (error) {
    if (!saved && error.status === 409 && error.code === "stale_session_revision" && await refreshStaleWorkspace(error)) return false;
    toast(saved ? `Selection was saved, but its current workspace could not be refreshed. ${error.message}` : error.message, true);
    return false;
  }
}

function operationEnabled(type) {
  return state.workspace.draft_capabilities?.operations?.[type]?.enabled === true;
}

function draftImpactRequestIsCurrent(request) {
  return Boolean(request && state.typedImpact.request === request
    && Object.entries(request).every(([key, value]) => state.session?.[key] === value));
}

async function loadDraftImpactPreview({ render = true } = {}) {
  if (!state.session?.session_id) return null;
  const request = Object.fromEntries(
    ["session_id", "trip_id", "run_id", "revision", "accepted_plan_id"].map(key => [key, state.session[key]]),
  );
  state.typedImpact.request = request;
  state.typedImpact.status = "loading";
  state.typedImpact.response = null;
  state.typedImpact.error = null;
  const controller = new AbortController();
  let deadline;
  try {
    const expired = new Promise((_resolve, reject) => {
      deadline = setTimeout(() => {
        reject(new Error("Draft impact check timed out. Your saved draft is unchanged. Retry the check when the server responds."));
        controller.abort();
      }, 20000);
    });
    const payload = await Promise.race([api(`/api/sessions/${state.session.session_id}/draft/impact-preview`, {
      method: "POST", signal: controller.signal,
      body: { expected_revision: state.session.revision },
    }), expired]);
    if (!draftImpactRequestIsCurrent(request)) return null;
    const normalized = normalizeDraftImpactPreview(payload);
    if (!normalized) throw new Error("invalid_draft_impact_preview");
    state.typedImpact.status = "ready";
    state.typedImpact.response = normalized;
    if (render) renderAll();
    return normalized;
  } catch (error) {
    if (!draftImpactRequestIsCurrent(request)) return null;
    state.typedImpact.status = "failed";
    state.typedImpact.response = null;
    state.typedImpact.error = error.message;
    if (render) renderAll();
    return null;
  } finally {
    clearTimeout(deadline);
  }
}

function canRunEvaluatedPreview() {
  const response = state.typedImpact.response;
  return response?.session_revision === state.session?.revision
    && response?.summary?.can_run_evaluated_preview === true;
}

function selectContext(patch) {
  const sessionId = state.session?.session_id;
  const previous = pendingSelection || Promise.resolve();
  // Resolve defaults and the revision only after earlier selection writes finish.
  const request = previous.catch(() => false).then(() => state.session?.session_id === sessionId
    ? updateSelection({ ...patch }) : false);
  pendingSelection = request;
  return request.finally(() => {
    if (pendingSelection === request) pendingSelection = null;
    scheduleInterestRefresh();
  });
}

function resetPoiDiscovery({ keepReplacementTarget = true } = {}) {
  resetInterestRegions();
  state.poiDiscovery.status = "idle";
  state.poiDiscovery.response = null;
  state.poiDiscovery.selectedCandidateId = null;
  state.poiDiscovery.showLayer = true;
  state.poiDiscovery.error = null;
  if (!keepReplacementTarget) state.poiDiscovery.replacementTargetId = null;
}

function resetInterestRegions({hide = false} = {}) {
  state.interestRegions.controller?.abort();
  state.interestRegions.controller = null;
  state.interestRegions.generation += 1;
  if (hide) { state.interestRegions.enabled = false; saveInterestPreferences(); }
  state.interestRegions.status = "idle";
  state.interestRegions.response = null;
  state.interestRegions.selectedId = null;
  state.interestRegions.focusStopId = null;
  state.interestRegions.error = null;
}

async function fetchInterestRegions(options = {}) {
  if (interestRefreshRunning) return false;
  interestRefreshRunning = true;
  state.interestRegions.enabled = true;
  const originSession = state.session.session_id;
  const originRevision = state.session.revision;
  clearTimeout(interestRefreshTimer);
  try {
    const loaded = await loadInterestRegions({refreshPlaces: options?.refreshPlaces === true});
    if (!loaded && state.interestRegions.enabled && ["idle", "loading"].includes(state.interestRegions.status)) {
      const changed = state.session.session_id !== originSession || state.session.revision !== originRevision;
      state.interestRegions.status = changed ? "idle" : "failed";
      state.interestRegions.error = changed ? null : "Choose an available day or retry the interest region.";
      renderMap();
    }
    return loaded;
  }
  finally { interestRefreshRunning = false; scheduleInterestRefresh(); }
}

async function loadInterestRegions({refreshPlaces = false} = {}) {
  if (pendingSelection && !(await pendingSelection)) {
    toast("Selection could not be updated. Interest estimates were not requested.", true);
    return false;
  }
  const displayed = activeRenderPlan();
  const parent = acceptedPlan(activeGeography());
  if (!displayed || !parent) return false;
  const stops = displayed.stops?.features?.filter(feature => feature.properties.day === state.session.selected_day) || [];
  const focusedStopId = stops.find(feature => feature.properties.stop_id === state.session.selected_stop_id)?.properties.stop_id
    || stops.find(feature => feature.properties.stop_id === state.interestRegions.focusStopId)?.properties.stop_id;
  let route = state.mapInteraction.selectedRouteLeg;
  const legs = displayed.validated_legs?.features || [];
  if (!route) {
    const savedLeg = legs.find(feature => feature.properties?.route_leg_id === state.session.selected_segment_id
      && feature.properties?.day === state.session.selected_day);
    if (savedLeg) {
      route = {...savedLeg.properties, plan_id: displayed.plan_id};
      state.mapInteraction.selectedRouteLeg = route;
    }
  }
  if (route?.plan_id !== displayed.plan_id || route.day !== state.session.selected_day
      || !legs.some(feature => feature.properties?.route_leg_id === route.route_leg_id)) {
    const leg = legs.find(feature => feature.properties?.day === state.session.selected_day);
    if (!leg) {
      toast("No road-validated route is available for this day. Choose another day to explore interest.", true);
      return false;
    }
    const selected = await selectRouteLeg({...leg.properties, plan_id: displayed.plan_id});
    if (!selected || activeRenderPlan()?.content_hash !== displayed.content_hash) return false;
    route = state.mapInteraction.selectedRouteLeg;
  }
  if (!route?.route_leg_id) {
    return false;
  }
  const view = state.interestRegions;
  if (!view.enabled) return false;
  view.radiusKm ??= 5;
  view.focusStopId = focusedStopId || stops.find(feature => feature.properties.stop_id === route.destination_id)?.properties.stop_id
    || stops[0]?.properties.stop_id || null;
  view.controller?.abort();
  const controller = new AbortController();
  const generation = ++view.generation;
  const expected = {sessionId: state.session.session_id, revision: state.session.revision,
    planId: parent.plan_id, planHash: parent.content_hash, day: route.day,
    explorationPlanId: displayed.plan_id, explorationPlanHash: displayed.content_hash,
    routeLegId: route.route_leg_id, naturePreference: view.naturePreference,
    focusStopId: view.focusStopId, radiusKm: view.radiusKm, searchArea: view.searchArea,
    replacementTargetId: displayed.plan_id === state.session.accepted_plan_id ? state.poiDiscovery.replacementTargetId : null};
  view.controller = controller; view.status = "loading"; view.response = null; view.error = null;
  saveInterestPreferences();
  const query = new URLSearchParams({expected_revision: String(expected.revision), day: String(expected.day),
    route_leg_id: expected.routeLegId, nature_preference: String(expected.naturePreference),
    plan_id: expected.explorationPlanId,
    max_detour_minutes: String(state.poiDiscovery.maxDetourMinutes)});
  query.set("radius_km", String(expected.radiusKm));
  query.set("search_area", expected.searchArea);
  if (refreshPlaces) query.set("refresh_places", "true");
  if (expected.focusStopId) query.set("focus_stop_id", expected.focusStopId);
  if (expected.replacementTargetId) query.set("replacement_target_id", expected.replacementTargetId);
  renderMap();
  const timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const raw = await api(`/api/sessions/${expected.sessionId}/map/interest-regions?${query}`, {signal: controller.signal});
    if (generation !== view.generation || !view.enabled || state.session.revision !== expected.revision
        || state.session.session_id !== expected.sessionId || state.session.selected_day !== expected.day
        || activeRenderPlan()?.plan_id !== expected.explorationPlanId
        || activeRenderPlan()?.content_hash !== expected.explorationPlanHash
        || state.session.selected_segment_id !== expected.routeLegId) return false;
    view.response = normalizeInterestResponse(raw, expected);
    view.status = "ready";
    view.selectedId = null;
    renderMap();
    return true;
  } catch (error) {
    if (generation !== view.generation || !view.enabled) return false;
    view.status = "failed";
    view.error = controller.signal.aborted ? "Interest request cancelled or timed out. The accepted itinerary is unchanged." : error.message;
    view.response = null;
    renderMap();
    return false;
  } finally {
    clearTimeout(timeout);
    if (view.controller === controller) view.controller = null;
  }
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
  if (mode === "edit") {
    const textRoute = $(".textual-map-alternative");
    if (textRoute) textRoute.open = false;
  }
  if (mode === "select" && state.mapInteraction.selectionKind === "exploratory") {
    cancelExploratoryPin(false);
  }
  if (state.route !== "/app/map" && state.route !== "/app/itinerary") {
    navigate("/app/map");
    return;
  }
  renderAll();
  requestAnimationFrame(() => $(state.mapInteraction.mode === "edit" ? "#map-edit-button" : "#map-select-button")?.focus());
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
  const textRoute = $(".textual-map-alternative");
  if (textRoute) textRoute.open = false;
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
  const textRoute = $(".textual-map-alternative");
  if (textRoute) textRoute.open = false;
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
        : $("#map-edit-button");
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
    installDraftMutationSession(payload.session);
    syncPreviewRuntimeFromSession();
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
  const selectedCandidateAvailable = candidates.some(
    (candidate) => (candidate.candidate_id || candidate.stop_id || candidate.id) === state.session.selected_candidate_id,
  );
  const requireAcceptedChildChoice = Boolean(state.workspace.accepted_plan_pointer?.decision_id)
    && !selectedCandidateAvailable;
  candidateSelect.innerHTML = (requireAcceptedChildChoice
    ? `<option value="">Choose a registered candidate</option>` : "") + candidates.map((candidate) => {
    const id = candidate.candidate_id || candidate.stop_id || candidate.id;
    return `<option value="${escapeHtml(id)}">${escapeHtml(candidate.label || candidate.name || id)}</option>`;
  }).join("");
  if (selectedCandidateAvailable) {
    candidateSelect.value = state.session.selected_candidate_id;
  } else if (requireAcceptedChildChoice) {
    candidateSelect.value = "";
  }
  candidateSelect.disabled = candidates.length === 0;
  $("#draft-keep").disabled = !selectedParent || !operationEnabled("keep_stop");
  $("#draft-move").disabled = !selectedParent || !operationEnabled("move_day");
  $("#draft-lock").disabled = !selectedParent || !operationEnabled("lock_stop");
  $("#draft-flexible").disabled = !selectedParent || !operationEnabled("mark_flexible");
  $("#draft-feedback").disabled = !operationEnabled("route_feedback");
  $("#draft-replace").disabled = !selectedParent || !candidateSelect.value
    || compatibleCandidates.length === 0 || !operationEnabled("replace_nearby");
  refreshCandidateAdditionAvailability();
  for (const [id, type] of Object.entries({ "draft-keep":"keep_stop", "draft-move":"move_day",
    "draft-lock":"lock_stop", "draft-flexible":"mark_flexible", "draft-feedback":"route_feedback",
    "draft-replace":"replace_nearby", "draft-add":"add_candidate" })) {
    if (draftCombinationBlocker([...state.session.draft, { type }])) {
      $(`#${id}`).disabled = true;
      $(`#${id}`).title = "This action cannot be combined with the staged direct edit. Preview and accept, or undo, that edit first.";
    }
  }
  $("#draft-undo").disabled = state.session.draft.length === 0;
  $("#draft-preview").disabled = state.session.draft.length === 0 || hasRouteCheckedOnlyDraft() || !canRunEvaluatedPreview();
  $("#draft-preview").title = state.session.draft.length && !canRunEvaluatedPreview()
    ? "This draft includes changes that are not enabled for evaluated repair" : "";
  if (selectedParent) {
    $("#edit-target").textContent = `Editing accepted-plan stop: ${selected.name}. Changes remain drafts until Preview repair finishes.`;
  } else if (state.session.selected_stop_id) {
    $("#edit-target").textContent = "The selected marker belongs to a repair preview and cannot be edited in place. Select an original accepted-plan stop, or add a registered candidate.";
  } else {
    $("#edit-target").textContent = "Select an accepted-plan stop. Only actions with an implemented optimizer mapping and the required route evidence are enabled.";
  }
}

function refreshCandidateAdditionAvailability() {
  const candidate = $("#registered-candidate")?.value;
  const day = Number($("#destination-day")?.value);
  const additionCapability = state.workspace.draft_capabilities?.operations?.add_candidate;
  const selectedAddition = additionCapability?.candidate_day_readiness?.find((row) => (
    row.candidate_id === candidate
      && row.day === day
      && row.parent_plan_id === state.session.accepted_plan_id
      && row.parent_plan_content_hash === state.workspace.accepted_plan_pointer?.accepted_plan_content_hash
  ));
  const control = $("#draft-add");
  const combinationBlocked = Boolean(draftCombinationBlocker([
    ...state.session.draft, { type: "add_candidate", target: candidate, parameters: { day } },
  ]));
  control.disabled = selectedAddition?.preview_executable !== true || combinationBlocked;
  control.title = combinationBlocked
    ? "This action cannot be combined with the staged direct edit. Preview and accept, or undo, that edit first."
    : selectedAddition?.preview_executable === true
      ? ""
      : "This candidate/day pair does not have complete route evidence for an evaluated preview.";
}

async function openMapEdit() {
  if (pendingSelection && !(await pendingSelection)) return toast("Selection could not be updated. The editor was not opened.", true);
  refreshMapEditDialog();
  $("#map-edit-dialog").showModal();
}

function previewExpectedContext() {
  return {
    session_id: state.session.session_id,
    trip_id: state.session.trip_id,
    run_id: state.session.run_id,
    revision: state.session.revision,
    accepted_plan_id: state.session.accepted_plan_id,
    parent_plan_content_hash: state.typedImpact.response?.parent_plan_content_hash,
    draft: structuredClone(state.session.draft),
    permission_decisions: structuredClone(state.session.permission_decisions),
    conversation_id: state.session.conversation_id,
  };
}

async function submitDraftMutation(action, body = {}) {
  if (state.workspaceConflict || state.workspaceSyncPending) throw new Error("Refresh the current workspace before making another change.");
  if (!["operations", "undo"].includes(action)) throw new Error("unsupported_draft_mutation");
  const expected = previewExpectedContext();
  const supersedesPreview = state.previewRuntime.status === "PREVIEW_RUNNING";
  if (supersedesPreview) state.previewRuntime.epoch += 1;
  const epoch = state.previewRuntime.epoch;
  let completed = false;
  try {
    let payload;
    try {
      payload = await mutationRecovery.submit(action, { ...body, expected_revision: expected.revision });
    } catch (error) {
      if (!supersedesPreview || error.status !== 409 || error.code !== "stale_session_revision"
          || !previewRequestIsCurrent(epoch, expected)) throw error;
      const current = await api(`/api/sessions/${expected.session_id}`);
      const normalized = await normalizeEvaluatedPreviewResponse({
        proposal: current?.session?.proposal, session: current?.session,
      }, expected);
      // Only an independently checked preview of this exact draft may advance
      // our revision. Never install that older proposal or retry another action.
      if (!normalized || !previewRequestIsCurrent(epoch, expected)) throw error;
      payload = await mutationRecovery.submit(action, { ...body, expected_revision: normalized.session.revision });
    }
    if (!previewRequestIsCurrent(epoch, expected)) throw new Error(apiErrorMessage("stale_session_revision"));
    completed = true;
    return payload;
  } finally {
    if (!completed && supersedesPreview && epoch === state.previewRuntime.epoch) {
      syncPreviewRuntimeFromSession();
      renderAll();
      copilot.previewStateChanged();
    }
  }
}

function installDraftMutationSession(nextSession, payload = null) {
  const previous = state.session;
  const proposal = previous?.proposal;
  const withdrewPreview = proposal?.state === "eligible"
    && typeof proposal.child_plan_id === "string" && proposal.child_plan_id.trim()
    && proposal.geography_plan?.plan_id === proposal.child_plan_id
    && proposal.geography_plan?.content_hash === proposal.child_plan_content_hash
    && nextSession.proposal === null
    && nextSession.session_id === previous.session_id
    && nextSession.accepted_plan_id === previous.accepted_plan_id;
  if (payload?.workspace) state.workspace = payload.workspace;
  if (payload?.registry) state.registry = payload.registry;
  state.session = nextSession;
  if (!withdrewPreview) return;
  // Server selection is authoritative; local inspector hints must follow the
  // same preview-to-accepted transition without overriding ordinary inspection.
  const interaction = state.mapInteraction;
  if (interaction.selectionKind === "custom_waypoint" && selectedDraftWaypoint()) return;
  const accepted = acceptedPlan(state.workspace.geography);
  const stop = accepted?.stops?.features?.find(
    feature => feature.properties?.stop_id === nextSession.selected_stop_id,
  );
  const leg = findRouteLeg(accepted, nextSession.selected_segment_id);
  interaction.selectedStopPlanId = stop ? accepted.plan_id : null;
  interaction.selectedRouteLeg = leg ? { ...leg.properties } : null;
  interaction.selectionKind = stop ? "stop" : leg ? "route_leg" : null;
}

async function addDraft(type, parameters, explicitTarget, source = "map") {
  if (pendingSelection && !(await pendingSelection)) {
    toast("Selection could not be updated. No draft was added.", true);
    return false;
  }
  const target = explicitTarget || state.session.selected_stop_id;
  const operations = Array.isArray(type) ? type : [{ type, target: target || "selected_route", parameters }];
  if (Array.isArray(type) ? operations.some(operation => !operation?.target) : !target && type !== "route_feedback") return toast("Select a stop before adding this draft.", true);
  if (draftCombinationBlocker([...state.session.draft, ...operations])) {
    toast("This combination cannot be evaluated. Nothing was added. Preview and accept, or undo, the current draft before adding a different edit family.", true);
    return false;
  }
  try {
    const payload = await submitDraftMutation("operations", Array.isArray(type) ? {
      operations: operations.map(operation => ({ ...operation, source })),
    } : {
      type, target: target || "selected_route", parameters, source,
    });
    installDraftMutationSession(payload.session, payload);
    syncPreviewRuntimeFromSession();
    state.tripPreviewPlanId = null;
    state.acknowledgedPreviewPlanId = null;
    resetPoiDiscovery({ keepReplacementTarget: false });
    const impactPending = loadDraftImpactPreview({ render: false });
    const impactRequest = state.typedImpact.request;
    toast(source === "copilot"
      ? "Copilot suggestion staged. Run Preview repair to compute and display any map change."
      : "Draft operation added. The parent plan is unchanged.");
    renderAll();
    if ($("#map-edit-dialog").open) refreshMapEditDialog();
    await impactPending;
    if (!draftImpactRequestIsCurrent(impactRequest)) return true;
    renderAll();
    if ($("#map-edit-dialog").open) refreshMapEditDialog();
    return true;
  } catch (error) { if (!(error.status === 409 && error.code === "stale_session_revision" && await refreshStaleWorkspace(error))) toast(error.message, true); return false; }
}

async function addCopilotDraft(intent) {
  if (Array.isArray(intent)) {
    const operations = intent.map(item => copilotOperationForIntent(item));
    if (operations.some(operation => !operation?.target)) return false;
    return addDraft(operations, null, null, "copilot");
  }
  const operation = copilotOperationForIntent(
    intent,
    state.session.selected_stop_id,
    state.session.selected_day,
  );
  if (operation?.target) {
    return addDraft(
      operation.type,
      operation.parameters,
      operation.target,
      "copilot",
    );
  }
  toast("This Copilot suggestion is advisory and cannot be added as a typed draft.", true);
  return false;
}

async function undoDraft() {
  if (state.undoPending) return;
  state.undoPending = true;
  copilot.update();
  try {
    if (pendingSelection && !(await pendingSelection)) return toast("Selection could not be updated. No undo was sent.", true);
    const payload = await submitDraftMutation("undo");
    installDraftMutationSession(payload.session, payload);
    syncPreviewRuntimeFromSession();
    state.tripPreviewPlanId = null;
    state.acknowledgedPreviewPlanId = null;
    resetPoiDiscovery({ keepReplacementTarget: false });
    if (state.mapInteraction.selectedWaypointId && !selectedDraftWaypoint()) {
      state.mapInteraction.selectionKind = null;
      state.mapInteraction.selectedWaypointId = null;
    }
    const impactPending = loadDraftImpactPreview({ render: false });
    const impactRequest = state.typedImpact.request;
    toast("Latest draft operation removed.");
    renderAll();
    if ($("#map-edit-dialog").open) refreshMapEditDialog();
    await impactPending;
    if (!draftImpactRequestIsCurrent(impactRequest)) return;
    renderAll();
    if ($("#map-edit-dialog").open) refreshMapEditDialog();
  } catch (error) { if (!(error.status === 409 && error.code === "stale_session_revision" && await refreshStaleWorkspace(error))) toast(error.message, true); } finally {
    state.undoPending = false;
    copilot.update();
  }
}

async function previewDraft(options = null) {
  // DOM click handlers pass an Event; only an explicit Copilot option narrows scope.
  const scopedToCopilot = options && (Object.hasOwn(options, "expectedCopilotOperation") || Object.hasOwn(options, "expectedCopilotOperations"));
  const expectedCopilotOperations = scopedToCopilot
    ? structuredClone(options.expectedCopilotOperations || [options.expectedCopilotOperation]) : null;
  const copilotOrigin = scopedToCopilot ? Object.fromEntries(
    ["session_id", "trip_id", "run_id", "accepted_plan_id", "conversation_id"].map(key => [key, state.session[key]]),
  ) : null;
  if (pendingSelection && !(await pendingSelection)) {
    toast("Selection could not be updated. No preview was sent.", true);
    return false;
  }
  if (scopedToCopilot && !Object.entries(copilotOrigin).every(([key, value]) => state.session[key] === value)) {
    toast("This queued Copilot preview belongs to an older workspace or conversation. No preview was sent; review the preserved draft in Repairs.", true);
    return false;
  }
  if (hasRouteCheckedOnlyDraft()) {
    toast("This arbitrary waypoint draft is route-checked, but evaluated repair is not enabled yet.", true);
    return false;
  }
  if (!canRunEvaluatedPreview()) {
    toast("This draft contains changes that are not enabled for evaluated repair. Inspect the server-owned draft impact, then undo or revise those changes.", true);
    return false;
  }
  const previewEpoch = state.previewRuntime.epoch + 1;
  state.previewRuntime = {
    epoch: previewEpoch,
    status: "PREVIEW_RUNNING",
    requestId: null,
  };
  renderAll();
  copilot.previewStateChanged();
  let expected = null;
  try {
    if (scopedToCopilot && (state.session.draft.length !== expectedCopilotOperations.length
        || !state.session.draft.every((operation, index) => sameCopilotDraftOperation(operation, expectedCopilotOperations[index])))) {
      throw new Error("The draft no longer contains only these exact staged suggestions. No preview was sent; review the current draft in Repairs.");
    }
    expected = previewExpectedContext();
    const payload = await api(`/api/sessions/${state.session.session_id}/preview`, {
      method: "POST", body: { expected_revision: state.session.revision },
    });
    if (!previewRequestIsCurrent(previewEpoch, expected)) {
      toast("An older preview response was ignored because the workspace changed.");
      return false;
    }
    const normalized = await normalizeEvaluatedPreviewResponse(payload, expected);
    if (!normalized) throw new Error(apiErrorMessage("invalid_evaluated_preview"));
    if (!previewRequestIsCurrent(previewEpoch, expected)) {
      toast("An older preview response was ignored because the workspace changed.");
      return false;
    }
    state.session = normalized.session;
    if (normalized.proposal.state === "eligible") {
      state.mapInteraction.selectedStopPlanId = normalized.proposal.child_plan_id;
      state.mapInteraction.selectedRouteLeg = null;
      state.mapInteraction.selectionKind = state.session.selected_stop_id ? "stop" : null;
    }
    state.previewRuntime.status = previewStateForSession(normalized.session);
    state.previewRuntime.requestId = normalized.proposal.compiled_request?.source_request_id || null;
    state.tripPreviewPlanId = null;
    state.acknowledgedPreviewPlanId = null;
    const installedContext = previewExpectedContext();
    const impactPending = loadDraftImpactPreview({ render: false });
    const impactRequest = state.typedImpact.request;
    await impactPending;
    if (!previewRequestIsCurrent(previewEpoch, installedContext)
        || !draftImpactRequestIsCurrent(impactRequest)) return false;
    $("#map-edit-dialog").close();
    toast(normalized.proposal.state === "eligible"
      ? "Preview evaluated. Rendering the exact child itinerary and map…"
      : normalized.proposal.message, normalized.proposal.state !== "eligible");
    renderAll();
    copilot.previewStateChanged();
    return true;
  } catch (error) {
    if (previewEpoch !== state.previewRuntime.epoch) return false;
    if (error.status === 409 && error.code === "stale_session_revision" && await refreshStaleWorkspace(error)) return false;
    state.previewRuntime.status = "PREVIEW_FAILED";
    state.previewRuntime.requestId = null;
    state.previewRuntime.message = error.message;
    renderAll();
    copilot.previewStateChanged();
    toast(error.message, true);
    return false;
  }
}
function setupDockResize() {
  const dock = $("#copilot-dock");
  const handle = $("#dock-resizer");
  const setWidth = (width) => {
    const bounded = Math.max(320, Math.min(560, width));
    $("#app-shell").style.setProperty("--copilot-width", `${bounded}px`);
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
      event.preventDefault();
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
  const geography = activeGeography();
  const explicitPlan = geography?.plans?.find(
    (plan) => plan.plan_id === state.mapInteraction.selectedStopPlanId,
  );
  const hasSelectedStop = (plan) => plan?.stops?.features?.some(
    (feature) => feature.properties?.stop_id === state.session.selected_stop_id,
  );
  const selectedPlan = hasSelectedStop(explicitPlan) ? explicitPlan : activeRenderPlan(geography);
  const feature = selectedPlan?.stops?.features?.find(
    (candidate) => candidate.properties?.stop_id === state.session.selected_stop_id,
  );
  if (feature) {
    return {
      id: feature.properties.stop_id,
      name: feature.properties.name || feature.properties.stop_id,
      day: feature.properties.day,
      plan_id: selectedPlan.plan_id,
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
  const region = $("#toast-region");
  // A newer status supersedes old progress, while errors retain their own lifetime.
  region.querySelectorAll(".toast:not(.error)").forEach((previous) => previous.remove());
  region.append(item);
  setTimeout(() => item.remove(), 5000);
}

ensureHostedLogin().then(boot).catch(renderStartupFailure);
