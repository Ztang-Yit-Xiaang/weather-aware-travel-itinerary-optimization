const TYPED_EDIT_INTENT_TYPES = new Set(["set_stop_duration", "set_stop_order", "set_stop_time_window", "set_stop_role"]);

export function copilotCapacityBlock(error) {
  if (error?.status !== 422) return null;
  if (error.code === "conversation_turn_limit_reached") {
    return "This conversation has reached its message limit. This message was blocked before any new provider attempt. Your trip and draft are unchanged. Open Conversation settings to delete this local conversation before sending a new message.";
  }
  if (error.code === "conversation_size_limit_reached") {
    return "This conversation has reached its storage limit. This message was blocked before any new provider attempt. Your trip and draft are unchanged. Open Conversation settings to delete this local conversation before sending a new message.";
  }
  return null;
}

const EDIT_INTENT_TYPES = new Set([
  ...TYPED_EDIT_INTENT_TYPES,
  "replan_trip",
  "keep_stop",
  "lock_stop",
  "mark_flexible",
  "move_day",
  "route_feedback",
  "replace_nearby",
  "add_candidate",
]);

const PARENT_TARGET_INTENT_TYPES = new Set([
  ...TYPED_EDIT_INTENT_TYPES,
  "keep_stop",
  "lock_stop",
  "mark_flexible",
  "move_day",
]);

export function copilotDraftability(intent, workspace) {
  const capabilities = workspace?.draft_capabilities;
  const typed = TYPED_EDIT_INTENT_TYPES.has(intent?.type);
  const operation = (typed ? workspace?.typed_edit_capabilities : capabilities)?.operations?.[intent?.type];
  if (typed && (operation?.preview_executable !== true || !typedEditParameters(intent))) {
    return { allowed: false, reason: "This typed stop edit has invalid parameters or is not enabled for evaluated preview." };
  }
  if (intent?.type === "add_candidate" && Array.isArray(operation?.candidate_day_readiness)) {
    const readiness = operation.candidate_day_readiness.find(row => row.candidate_id === intent.candidate_id
      && row.day === intent.destination_day && row.parent_plan_id === workspace.accepted_plan_pointer?.accepted_plan_id
      && row.parent_plan_content_hash === workspace.accepted_plan_pointer?.accepted_plan_content_hash);
    if (!readiness?.preview_executable) {
      const missing = readiness?.blocking_findings?.[0];
      return { allowed: false, reason: missing
        ? `Addition is blocked for Day ${intent.destination_day}: ${missing.origin_id} → ${missing.destination_id} needs road evidence. ${missing.corrective_action}`
        : "Addition has no verified route evidence for this candidate, day, and accepted itinerary. No evaluated repair can run yet." };
    }
  }
  if (!EDIT_INTENT_TYPES.has(intent?.type) || operation?.enabled !== true) {
    return {
      allowed: false,
      reason: "This suggestion is advisory because this draft operation is not enabled for the current itinerary.",
    };
  }

  const parentTargets = new Set(capabilities.parent_targets || []);
  if (PARENT_TARGET_INTENT_TYPES.has(intent.type) && !parentTargets.has(intent.target_stop_id)) {
    return {
      allowed: false,
      reason: "This suggestion targets a repair-preview or discovery-only stop. Choose an accepted-plan stop, or use Compare to preview the registered repair.",
    };
  }

  const candidates = capabilities.candidate_choices || [];
  if (intent.type === "replace_nearby") {
    const compatible = candidates.some((candidate) => (
      candidate.candidate_id === intent.candidate_id
      && candidate.replaces_stop_id === intent.target_stop_id
      && parentTargets.has(intent.target_stop_id)
    ));
    if (!compatible) {
      return {
        allowed: false,
        reason: "This registered candidate cannot replace the referenced accepted-plan stop.",
      };
    }
  }
  if (intent.type === "add_candidate" && !candidates.some(
    (candidate) => candidate.candidate_id === intent.candidate_id,
  )) {
    return {
      allowed: false,
      reason: "This candidate is not registered for the current itinerary.",
    };
  }
  return { allowed: true, reason: null };
}

export function copilotTurnIsCurrent(turn, session) {
  return Number.isInteger(turn?.context_revision) && turn.context_revision === session?.revision;
}

export function copilotOperationForIntent(intent, selectedStopId = null, selectedDay = null) {
  if (intent?.type === "replan_trip") return { type: "replan_trip", target: "trip", parameters: intent.parameters };
  const target = intent?.target_stop_id || selectedStopId;
  if (TYPED_EDIT_INTENT_TYPES.has(intent?.type)) {
    const parameters = typedEditParameters(intent);
    return target && parameters ? { type: intent.type, target, parameters } : null;
  }
  if (intent?.type === "route_feedback") {
    return {
      type: intent.type,
      target: "selected_route",
      parameters: { preference: intent.preference || "reduce_contextual_risk" },
    };
  }
  if (intent?.type === "move_day") {
    return { type: intent.type, target, parameters: { day: intent.destination_day } };
  }
  if (intent?.type === "replace_nearby") {
    return { type: intent.type, target, parameters: { candidate_id: intent.candidate_id } };
  }
  if (intent?.type === "add_candidate") {
    return {
      type: intent.type,
      target: intent.candidate_id,
      parameters: { day: intent.destination_day || selectedDay },
    };
  }
  if (["keep_stop", "lock_stop", "mark_flexible"].includes(intent?.type)) {
    return { type: intent.type, target, parameters: {} };
  }
  return null;
}

function typedEditParameters(intent) {
  const parameters = intent?.parameters;
  const fields = (value, expected) => value && typeof value === "object" && !Array.isArray(value)
    && Object.keys(value).sort().join(",") === [...expected].sort().join(",");
  if (intent.type === "set_stop_duration") {
    const duration = parameters?.duration;
    if (!fields(parameters, ["duration"]) || !fields(duration, ["mode", "preferred_minutes", "minimum_minutes", "maximum_minutes"])
        || duration.mode !== "exact" || !Number.isInteger(duration.preferred_minutes)
        || duration.preferred_minutes < 15 || duration.preferred_minutes > 480
        || duration.minimum_minutes !== duration.preferred_minutes || duration.maximum_minutes !== duration.preferred_minutes) return null;
  } else if (intent.type === "set_stop_order") {
    if (!fields(parameters, ["day", "sequence_index"]) || !Number.isInteger(parameters.day)
        || parameters.day < 1 || parameters.day > 21 || !Number.isInteger(parameters.sequence_index)
        || parameters.sequence_index < 0 || parameters.sequence_index > 63) return null;
  } else if (intent.type === "set_stop_time_window") {
    const time = value => value === null || typeof value === "string" && /^(?:[01][0-9]|2[0-3]):[0-5][0-9]$/.test(value);
    if (!fields(parameters, ["earliest_arrival", "latest_departure"]) || !time(parameters.earliest_arrival)
        || !time(parameters.latest_departure) || parameters.earliest_arrival === null && parameters.latest_departure === null
        || parameters.earliest_arrival !== null && parameters.latest_departure !== null && parameters.earliest_arrival > parameters.latest_departure) return null;
  } else if (!fields(parameters, ["role"]) || !["attraction", "activity", "meal", "rest_stop", "scenic_stop"].includes(parameters.role)) return null;
  return JSON.parse(JSON.stringify(parameters));
}

function canonicalParameters(parameters) {
  if (!parameters || typeof parameters !== "object" || Array.isArray(parameters)) return null;
  const sorted = value => Array.isArray(value) ? value.map(sorted) : value && typeof value === "object"
    ? Object.fromEntries(Object.entries(value).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => [key, sorted(item)])) : value;
  return JSON.stringify(sorted(parameters));
}

export function sameCopilotDraftOperation(left, right) {
  return Boolean(
    left
    && right
    && left.type === right.type
    && left.target === right.target
    && canonicalParameters(left.parameters) !== null
    && canonicalParameters(left.parameters) === canonicalParameters(right.parameters),
  );
}

export function copilotOneClickEligibility(intent, workspace, session) {
  const actionability = copilotDraftability(intent, workspace);
  if (!actionability.allowed) return { ...actionability, alreadyStaged: false };
  const operation = copilotOperationForIntent(
    intent,
    session?.selected_stop_id,
    session?.selected_day,
  );
  if (!operation?.target) {
    return {
      allowed: false,
      alreadyStaged: false,
      reason: "Select the referenced stop or day before previewing this suggestion.",
    };
  }
  const draft = Array.isArray(session?.draft) ? session.draft : [];
  const alreadyStaged = draft.some((candidate) => sameCopilotDraftOperation(candidate, operation));
  if (draft.length === 0 || (draft.length === 1 && alreadyStaged)) {
    return { allowed: true, alreadyStaged, reason: null };
  }
  return {
    allowed: false,
    alreadyStaged,
    reason: "Review the existing draft first. Previewing this suggestion must not silently include unrelated or conflicting changes.",
  };
}

export function copilotCombinedEligibility(intents, workspace, session) {
  const blocked = reason => ({ allowed: false, alreadyStaged: false, reason });
  if (!Array.isArray(intents) || intents.length < 2 || intents.length > 4) return blocked("A combined suggestion must contain two to four complete edits.");
  for (const intent of intents) {
    const actionability = copilotDraftability(intent, workspace);
    if (!actionability.allowed) return blocked(actionability.reason);
  }
  const combinations = workspace?.draft_capabilities?.executable_combinations;
  const combination = Array.isArray(combinations) && combinations.some(row =>
    Array.isArray(row?.operations) && Number.isInteger(row.cardinality) && row.cardinality <= 32
    && intents.length <= row.cardinality && intents.every(intent => row.operations.includes(intent.type)));
  if (!combination) return blocked("This complete combination is not enabled for evaluated preview. No part will be staged.");
  const operations = intents.map(intent => copilotOperationForIntent(intent));
  if (operations.some(operation => !operation?.target)) return blocked("Every edit needs an explicit validated target.");
  const draft = Array.isArray(session?.draft) ? session.draft : [];
  const alreadyStaged = draft.length === operations.length && draft.every((operation, index) => sameCopilotDraftOperation(operation, operations[index]));
  return draft.length === 0 || alreadyStaged ? { allowed: true, alreadyStaged, reason: null }
    : blocked("Review the existing draft first. A combined suggestion must not silently include unrelated or partial changes.");
}

const LIFECYCLE_COPY = {
  sending: "Sending",
  interpreting: "Interpreting",
  cancelling: "Stopping",
  cancelled: "Stopped",
  blocked: "Message blocked",
  preview_running: "Preview running",
  preview_rendering: "Rendering preview",
  preview_rendered: "Preview ready",
  preview_ineligible: "Preview not eligible",
  clarification_required: "Clarification required",
  permission_required: "Permission required",
  proposal_ready: "Proposal ready",
  awaiting_confirmation: "Awaiting confirmation",
  refused: "Refused",
  failed: "Failed",
};

export function createCopilotController({
  getState,
  api,
  streamMessage,
  selectContext,
  addDraft,
  previewDraft,
  getPreviewState,
  getSelectedStop,
  navigate,
  toast,
  renderAll,
  syncPreviewRuntime,
  onIdle = () => {},
}) {
  let requestActive = false;
  let dockOpener = null;
  let settingsOpener = null;
  let lastInterpretation = null;
  let lastConversation = null;
  let ephemeralTurn = null;
  let activeRequest = null;
  let requestEpoch = 0;
  let restoreEpoch = 0;
  let conversationEpoch = 0;
  let conversationSessionId = null;
  let conversationId = null;
  let pendingConversationReset = null;
  let previewActionActive = false;
  let stagedLifecycle = null;

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  function interaction() {
    return getState().workspace?.interaction || {};
  }

  function providerName() {
    return interaction().provider === "openai" ? "OpenAI Copilot" : "Deterministic demo";
  }

  function available() {
    return interaction().enabled === true;
  }

  function adoptSession(incoming, projection = null) {
    const state = getState();
    const current = state.session;
    if (!incoming || incoming.session_id !== current?.session_id
        || incoming.trip_id !== current?.trip_id || incoming.run_id !== current?.run_id
        || !Number.isSafeInteger(incoming.revision) || incoming.revision < 0
        || !Number.isSafeInteger(current?.revision)) {
      throw new Error("The conversation response did not match the current session.");
    }
    // Chat reads and replies can finish after a newer draft/selection mutation.
    if (incoming.revision < current.revision) return false;
    const acceptedChanged = incoming.accepted_plan_id !== current.accepted_plan_id;
    if (acceptedChanged) {
      const workspace = projection?.workspace;
      const pointer = workspace?.accepted_plan_pointer;
      const plan = workspace?.geography?.plans?.find(row => row.plan_id === incoming.accepted_plan_id);
      if (workspace?.schema_version !== "product-dashboard-data-v1"
          || pointer?.accepted_plan_id !== incoming.accepted_plan_id
          || !pointer.accepted_plan_content_hash || plan?.content_hash !== pointer.accepted_plan_content_hash) {
        throw new Error("The saved trip needs its matching map before the conversation can be refreshed.");
      }
      state.workspace = workspace;
      state.registry = projection.registry;
    }
    state.session = incoming;
    if (acceptedChanged) {
      syncPreviewRuntime?.();
      renderAll();
    }
    return true;
  }

  function resetConversation() {
    const waiting = Boolean(activeRequest);
    conversationEpoch += 1;
    requestEpoch += 1;
    restoreEpoch += 1;
    activeRequest?.controller.abort(); // Stop this browser's wait, not provider processing.
    activeRequest = null;
    pendingConversationReset = null;
    ephemeralTurn = null;
    lastConversation = null;
    lastInterpretation = null;
    conversationSessionId = getState().session?.session_id ?? null;
    conversationId = getState().session?.conversation_id ?? null;
    setSubmitting(false);
    renderConversation(null);
    setLifecycle("idle", waiting
      ? "Local wait stopped. Provider processing or usage may already have occurred."
      : "Previous local conversation cleared");
  }

  function conversationOrigin() {
    const session = getState().session;
    const nextId = session?.conversation_id ?? null;
    if (conversationSessionId !== null && (conversationSessionId !== session?.session_id
        || conversationId !== null && conversationId !== nextId)) resetConversation();
    conversationSessionId = session?.session_id ?? null;
    conversationId = nextId;
    return { sessionId: conversationSessionId, epoch: conversationEpoch,
      tripId: session?.trip_id, runId: session?.run_id, acceptedPlanId: session?.accepted_plan_id };
  }

  function originIsCurrent(origin) {
    const current = conversationOrigin();
    return ["sessionId", "epoch", "tripId", "runId", "acceptedPlanId"].every(field => current[field] === origin[field]);
  }

  function requestIsCurrent(epoch) {
    conversationOrigin();
    return activeRequest?.epoch === epoch && requestEpoch === epoch
      && originIsCurrent(activeRequest.origin);
  }

  function validConversation(payload) {
    const incoming = payload?.session, conversation = payload?.conversation;
    return conversation?.schema_version === "product-conversation-v1"
      && typeof conversation.conversation_id === "string" && conversation.conversation_id.length > 0
      && conversation.conversation_id === incoming?.conversation_id
      && ["session_id", "trip_id", "run_id"].every(field => typeof conversation[field] === "string"
        && conversation[field].length > 0 && conversation[field] === incoming?.[field])
      && Number.isSafeInteger(conversation.revision) && conversation.revision >= 0
      && Array.isArray(conversation.turns);
  }

  function setLifecycle(state, detail = "") {
    stagedLifecycle = null;
    const region = $("#copilot-lifecycle");
    if (!region) return;
    const label = LIFECYCLE_COPY[state] || state;
    region.textContent = detail ? `${label}: ${detail}` : label;
    region.dataset.state = state;
    delete region.dataset.previewState;
  }

  function setSubmitting(active) {
    requestActive = active;
    const submit = $("#copilot-submit");
    if (submit) {
      submit.disabled = previewActionActive || !available() || Boolean(pendingConversationReset) || (!active && Boolean((getState().undoPending || getState().mutationRecovery?.blocked || getState().workspaceConflict || getState().workspaceSyncPending)));
      submit.dataset.mode = active ? "stop" : "send";
      submit.setAttribute("aria-label", active ? "Stop waiting for Copilot" : "Send message");
      submit.textContent = active ? "■" : "→";
    }
    const input = $("#copilot-input");
    if (input) input.setAttribute("aria-busy", String(active || previewActionActive || Boolean(pendingConversationReset) || Boolean((getState().undoPending || getState().mutationRecovery?.blocked || getState().workspaceConflict || getState().workspaceSyncPending))));
    if (!active && !previewActionActive && !pendingConversationReset) onIdle();
    $$(".quick-prompts button").forEach((control) => { control.disabled = active || previewActionActive || !available() || Boolean(pendingConversationReset) || Boolean((getState().undoPending || getState().mutationRecovery?.blocked || getState().workspaceConflict || getState().workspaceSyncPending)); });
  }

  function appendMessageText(paragraph, text) {
    const links = [...text.matchAll(/\[([^\]\n]{1,200})\]\((https?:\/\/[^\s<>()]{1,2048})\)/g)]
      .map(match => {
        try {
          const url = new URL(match[2]);
          return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password
            ? { match, url } : null;
        } catch { return null; }
      }).filter(Boolean);
    if (!links.length) { paragraph.textContent = text; return; }
    let cursor = 0;
    for (const { match, url } of links) {
      paragraph.append(text.slice(cursor, match.index));
      const link = document.createElement("a");
      link.href = url.href;
      link.textContent = match[1];
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      paragraph.append(link);
      cursor = match.index + match[0].length;
    }
    paragraph.append(text.slice(cursor));
  }

  function appendMessage(
    role,
    text,
    label,
    turn = null,
    { interactive = false, pending = false } = {},
  ) {
    const article = document.createElement("article");
    article.className = `message message-${role}`;
    if (pending) article.classList.add("message-pending");
    if (turn?.client_message_id) article.dataset.clientMessageId = turn.client_message_id;
    if (turn?.state) article.dataset.deliveryState = turn.state;
    const avatar = document.createElement("span");
    avatar.className = "avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = role === "user" ? "Y" : "C";
    const copy = document.createElement("div");
    const strong = document.createElement("strong");
    strong.textContent = label;
    const paragraph = document.createElement("p");
    appendMessageText(paragraph, text);
    copy.append(strong, paragraph);
    if (interactive && turn?.interpretation?.state === "proposal_ready") {
      copy.append(buildProposalSummary(turn.interpretation));
      copy.append(buildProposalActions(turn));
    }
    if (interactive && ["failed", "cancelled"].includes(turn?.state)) {
      copy.append(buildRecoveryActions(turn));
    }
    if (interactive && turn?.state === "blocked") {
      const actions = document.createElement("div");
      actions.className = "copilot-recovery-actions";
      actions.setAttribute("aria-label", "Conversation capacity actions");
      actions.append(
        button("Conversation settings", () => openSettings($("#transcript-settings-button"))),
        button("Edit message", () => editMessage(turn.user_message)),
      );
      copy.append(actions);
    }
    article.append(avatar, copy);
    $("#conversation").append(article);
  }

  function renderEmptyConversation() {
    const provider = providerName();
    const message = interaction().provider === "openai"
      ? "I can interpret this itinerary and return typed, reviewable suggestions. I cannot change the trip without your explicit action."
      : "I can demonstrate typed itinerary requests locally. No message is sent to an external provider, and I cannot change the trip without your explicit action.";
    appendMessage("assistant", message, provider);
  }

  function renderConversation(conversation) {
    const container = $("#conversation");
    const wasNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 80;
    const previousScrollTop = container.scrollTop;
    container.replaceChildren();
    lastConversation = conversation;
    lastInterpretation = null;
    const turns = conversation?.turns || [];
    let interactiveIndex = -1;
    for (let index = turns.length - 1; index >= 0; index -= 1) {
      const turn = turns[index];
      if (turn.state === "failed" || turn.interpretation?.state === "proposal_ready") {
        interactiveIndex = index;
        break;
      }
    }
    turns.forEach((turn, index) => {
      appendMessage("user", turn.user_message, "You", turn);
      appendMessage(
        "assistant",
        turn.assistant_message,
        turn.provider === "openai" ? "OpenAI Copilot" : "Deterministic demo",
        turn,
        { interactive: index === interactiveIndex },
      );
      if (turn.interpretation) lastInterpretation = turn.interpretation;
    });
    if (ephemeralTurn) {
      appendMessage("user", ephemeralTurn.user_message, "You", ephemeralTurn, { pending: true });
      appendMessage(
        "assistant",
        ephemeralTurn.assistant_message,
        providerName(),
        ephemeralTurn,
        { interactive: true, pending: true },
      );
    }
    if (!turns.length && !ephemeralTurn) renderEmptyConversation();
    container.scrollTop = wasNearBottom ? container.scrollHeight : previousScrollTop;
  }

  function button(label, action, disabled = false) {
    const control = document.createElement("button");
    control.type = "button";
    control.textContent = label;
    control.disabled = disabled || Boolean(getState().mutationRecovery?.blocked);
    control.addEventListener("click", action);
    return control;
  }

  function buildRecoveryActions(turn) {
    const origin = conversationOrigin();
    const guarded = action => () => originIsCurrent(origin) ? action()
      : toast("This request belongs to a previous conversation. Send a new message for the current conversation.", true);
    const actions = document.createElement("div");
    actions.className = "copilot-recovery-actions";
    actions.setAttribute("aria-label", "Message recovery actions");
    actions.append(
      button("Retry", guarded(() => startMessage(turn.user_message, {
        clientMessageId: turn.client_message_id,
      }))),
      button("Edit message", guarded(() => editMessage(turn.user_message))),
    );
    return actions;
  }

  function stopName(stopId) {
    const workspace = getState().workspace;
    const timelineStop = workspace.timeline
      .flatMap((day) => day.stops)
      .find((stop) => stop.id === stopId);
    if (timelineStop?.name) return timelineStop.name;
    const geographyStop = (workspace.geography?.plans || [])
      .flatMap((plan) => plan?.stops?.features || [])
      .find((feature) => feature?.properties?.stop_id === stopId);
    return geographyStop?.properties?.name || stopId || "the selected stop";
  }

  function candidateName(candidateId) {
    const candidate = (getState().workspace.draft_capabilities?.candidate_choices || [])
      .find((row) => row.candidate_id === candidateId);
    return candidate?.label || candidate?.name || (candidateId ? stopName(candidateId) : "the selected candidate");
  }

  function intentDescription(intent) {
    if (intent.type === "replace_nearby") {
      return `Replace ${stopName(intent.target_stop_id)} with ${candidateName(intent.candidate_id)}.`;
    }
    if (intent.type === "add_candidate") {
      return `Add ${candidateName(intent.candidate_id)} to Day ${intent.destination_day || getState().session.selected_day}.`;
    }
    if (intent.type === "move_day") {
      return `Move ${stopName(intent.target_stop_id)} to Day ${intent.destination_day}.`;
    }
    if (intent.type === "keep_stop") return `Keep ${stopName(intent.target_stop_id)} in the itinerary.`;
    if (intent.type === "lock_stop") return `Preserve ${stopName(intent.target_stop_id)} as a locked commitment.`;
    if (intent.type === "mark_flexible") return `Mark ${stopName(intent.target_stop_id)} as flexible.`;
    if (intent.type === "set_stop_duration") {
      return `Set ${stopName(intent.target_stop_id)} to exactly ${intent.parameters.duration.preferred_minutes} minutes.`;
    }
    if (intent.type === "set_stop_order") {
      return `Place ${stopName(intent.target_stop_id)} at position ${intent.parameters.sequence_index + 1} on Day ${intent.parameters.day}.`;
    }
    if (intent.type === "set_stop_role") {
      return `Use ${stopName(intent.target_stop_id)} as a ${intent.parameters.role.replaceAll("_", " ")}.`;
    }
    if (intent.type === "set_stop_time_window") {
      const { earliest_arrival: arrival, latest_departure: departure } = intent.parameters;
      const bounds = [arrival && `arrive no earlier than ${arrival}`, departure && `depart no later than ${departure}`].filter(Boolean);
      return `At ${stopName(intent.target_stop_id)}, ${bounds.join(" and ")}.`;
    }
    if (intent.type === "replan_trip") {
      const p = intent.parameters;
      if (Number.isInteger(p?.total_days)) {
        const updates = p.profile_updates || [];
        const pace = updates.includes("pace") ? p.pace : "keep accepted pace (balanced for the first plan)";
        const driving = updates.includes("max_daily_drive_minutes") ? `${p.max_daily_drive_minutes} minutes` : "keep accepted limit (720 minutes for the first plan)";
        const placeNames = {
          hearst_castle: "Hearst Castle", channel_islands: "Channel Islands",
          redwood_national_park: "Redwood National Park", kings_canyon_national_park: "Kings Canyon National Park",
          yosemite_national_park: "Yosemite National Park", sequoia_national_park: "Sequoia National Park",
          bixby_creek_bridge_viewpoint: "Bixby Creek Bridge Viewpoint", point_lobos_state_natural_reserve: "Point Lobos State Natural Reserve",
          pfeiffer_big_sur_state_park: "Pfeiffer Big Sur State Park", golden_gate_bridge: "Golden Gate Bridge",
          stanford_university_main_quad: "Stanford University Main Quad", stearns_wharf: "Stearns Wharf",
          griffith_observatory: "Griffith Observatory", hollywood_walk_of_fame: "Hollywood Walk of Fame",
          tcl_chinese_theatre: "TCL Chinese Theatre", santa_barbara_county_courthouse: "Santa Barbara County Courthouse",
        };
        const required = p.required_place_ids.map(id => placeNames[id] || candidateName(id)).join(", ") || "preserve existing visits";
        const excluded = p.excluded_place_ids.map(id => placeNames[id] || candidateName(id)).join(", ") || "no new exclusions";
        return `Plan ${p.total_days} days · Pace: ${pace} · Daily driving limit: ${driving} · Interests: ${p.interests.join(", ") || "keep accepted interests"}. Required: ${required}; Excluded: ${excluded}. ${p.coastal ? "Prefer coastal stops. " : ""}Preview will check road evidence and the full itinerary; nothing has been applied.`;
      }
      return "Replan the trip with these start, required-place, avoidance and coastal constraints. Preview checks official sources and road evidence before evaluation.";
    }
    if (intent.type === "route_feedback") return "Evaluate the current route using the requested travel preference.";
    return "Review this advisory suggestion.";
  }

  function buildProposalSummary(interpretation) {
    const group = document.createElement("section");
    group.className = "copilot-proposal-list";
    group.setAttribute("aria-label", "Typed Copilot suggestions");
    (interpretation.intents || []).forEach((intent, index) => {
      if (!EDIT_INTENT_TYPES.has(intent.type)) return;
      const card = document.createElement("article");
      card.className = "copilot-proposal-card";
      card.dataset.copilotIntentIndex = String(index);
      card.dataset.copilotIntentType = intent.type;
      const eyebrow = document.createElement("span");
      eyebrow.className = "copilot-card-eyebrow";
      eyebrow.textContent = "Typed suggestion";
      const description = document.createElement("p");
      description.textContent = intentDescription(intent);
      const status = document.createElement("small");
      status.textContent = "Suggestion only · no itinerary change has been applied.";
      card.append(eyebrow, description, status);
      group.append(card);
    });
    return group;
  }

  function intentAlreadyDrafted(intent) {
    const state = getState();
    const operation = copilotOperationForIntent(
      intent,
      state.session.selected_stop_id,
      state.session.selected_day,
    );
    return Boolean(operation) && state.session.draft.some(
      (candidate) => sameCopilotDraftOperation(candidate, operation),
    );
  }

  function buildProposalActions(turn) {
    const interpretation = turn.interpretation;
    const origin = conversationOrigin();
    const current = () => originIsCurrent(origin) && copilotTurnIsCurrent(turn, getState().session);
    const staleReason = "This suggestion belongs to an older workspace state. Send a new message for the current itinerary; existing drafts can still be reviewed in Repairs.";
    const guarded = (action) => (event) => current() ? action(event) : toast(staleReason, true);
    const actions = document.createElement("div");
    actions.className = "copilot-proposal-actions";
    actions.setAttribute("aria-label", "Copilot proposal actions");
    const highlights = interpretation.highlights || {};
    const hasHighlights = ["day_ids", "stop_ids", "segment_ids", "candidate_ids"]
      .some((field) => Array.isArray(highlights[field]) && highlights[field].length > 0);
    if (hasHighlights) actions.append(button("Show on map", guarded(() => showOnMap(highlights)), previewActionActive || !current()));

    const editIntents = (interpretation.intents || [])
      .map((intent, index) => ({
        intent,
        index,
        actionability: copilotDraftability(intent, getState().workspace),
      }))
      .filter(({ intent }) => EDIT_INTENT_TYPES.has(intent.type));
    if (editIntents.length > 1) {
      const intents = interpretation.intents;
      const eligibility = copilotCombinedEligibility(intents, getState().workspace, getState().session);
      const canUse = current() || eligibility.alreadyStaged;
      const preview = button("Preview combined repair", () => {
        if (!originIsCurrent(origin) || (!current() && !copilotCombinedEligibility(intents, getState().workspace, getState().session).alreadyStaged)) return toast(staleReason, true);
        return previewIntentFromCopilot(intents);
      }, previewActionActive || !canUse || !eligibility.allowed);
      preview.classList.add("copilot-primary-action");
      const add = button(eligibility.alreadyStaged ? "All changes added to draft" : "Add all to draft",
        guarded(() => addIntentToDraft(intents)), previewActionActive || !current() || !eligibility.allowed || eligibility.alreadyStaged);
      actions.append(preview, add);
      const note = document.createElement("p");
      note.className = "copilot-proposal-note";
      note.textContent = !canUse ? staleReason : eligibility.reason || (eligibility.alreadyStaged
        ? "All requested changes are staged. Preview evaluates this exact combined draft."
        : "All changes will be staged together or none will be staged. The map changes only after an eligible preview renders.");
      actions.append(note);
    } else editIntents.forEach(({ intent, index, actionability }) => {
      const row = document.createElement("div");
      row.className = "copilot-proposal-action-row";
      row.dataset.copilotIntentIndex = String(index);
      row.dataset.copilotIntentType = intent.type;
      const actionTarget = intentDescription(intent);
      const actionContext = intent.type === "replan_trip" ? "trip plan" : actionTarget;
      const actionLabel = document.createElement("strong");
      actionLabel.className = "copilot-action-target";
      actionLabel.textContent = actionTarget;
      row.append(actionLabel);
      const previewEligibility = copilotOneClickEligibility(intent, getState().workspace, getState().session);
      // Staging increments the revision. Preview consumes that exact server-owned
      // draft; it does not replay a stale suggestion or authorize another Add.
      if (actionability.allowed && (current() || (previewEligibility.allowed && previewEligibility.alreadyStaged))) {
        const preview = button(
          "Preview this repair",
          (event) => {
            if (!originIsCurrent(origin)) return toast(staleReason, true);
            const latest = copilotOneClickEligibility(intent, getState().workspace, getState().session);
            if (!current() && !(latest.allowed && latest.alreadyStaged)) return toast(staleReason, true);
            return previewIntentFromCopilot(intent, event.currentTarget);
          },
          previewActionActive || !previewEligibility.allowed,
        );
        preview.classList.add("copilot-primary-action");
        preview.setAttribute("aria-label", `Preview this repair: ${actionContext}`);
        row.append(preview);
        const alreadyAdded = intentAlreadyDrafted(intent);
        const add = button(
          alreadyAdded ? "Added to draft" : "Add to draft",
          guarded(() => addIntentToDraft(intent)),
          previewActionActive || alreadyAdded || !current(),
        );
        add.setAttribute("aria-label", `${alreadyAdded ? "Added to draft" : "Add to draft"}: ${actionContext}`);
        row.append(add);
        if (!previewEligibility.allowed) {
          const note = document.createElement("p");
          note.className = "copilot-proposal-note";
          note.textContent = previewEligibility.reason;
          row.append(note);
        } else if (alreadyAdded) {
          const note = document.createElement("p");
          note.className = "copilot-proposal-note";
          note.textContent = "This exact suggestion is staged. Preview will run the deterministic repair and independent evaluation.";
          row.append(note);
        }
      } else {
        const unavailable = button("Cannot add to draft", () => {}, true);
        unavailable.setAttribute("aria-label", `Cannot add to draft: ${actionContext}`);
        row.append(unavailable);
        const note = document.createElement("p");
        note.className = "copilot-proposal-note";
        note.textContent = current() ? actionability.reason : staleReason;
        row.append(note);
      }
      actions.append(row);
    });

    actions.append(
      button("Compare", () => navigate("/app/compare"), previewActionActive),
      button("Review evidence", () => navigate("/app/evidence"), previewActionActive),
    );
    const unrelatedTrip = editIntents.some(({intent}) => intent.type === "replan_trip" && !intentAlreadyDrafted(intent));
    const previewCard = unrelatedTrip ? null : buildPreviewResult();
    if (previewCard) actions.append(previewCard);
    return actions;
  }

  function validSet(field) {
    const state = getState();
    if (field === "stop_ids") return new Set(state.workspace.timeline.flatMap((day) => day.stops.map((stop) => stop.id)));
    if (field === "day_ids") return new Set(state.workspace.timeline.map((day) => day.day));
    if (field === "segment_ids") {
      return new Set(state.session.selected_segment_id ? [state.session.selected_segment_id] : []);
    }
    if (field === "candidate_ids") return new Set((state.workspace.draft_capabilities?.candidate_choices || []).map((candidate) => candidate.candidate_id));
    return new Set();
  }

  function firstValidated(highlights, field) {
    const allowed = validSet(field);
    return (highlights[field] || []).find((value) => allowed.has(value)) ?? null;
  }

  function acceptedDayForStop(stopId) {
    if (stopId === null) return null;
    return getState().workspace.timeline.find(
      (day) => day.stops.some((stop) => stop.id === stopId),
    )?.day ?? null;
  }

  async function showOnMap(highlights) {
    const patch = {};
    const stop = firstValidated(highlights, "stop_ids");
    const day = firstValidated(highlights, "day_ids") ?? acceptedDayForStop(stop);
    const segment = firstValidated(highlights, "segment_ids");
    const candidate = firstValidated(highlights, "candidate_ids");
    if (day !== null) patch.selected_day = day;
    if (stop !== null) patch.selected_stop_id = stop;
    if (segment !== null) patch.selected_segment_id = segment;
    if (candidate !== null) patch.selected_candidate_id = candidate;
    if (!Object.keys(patch).length) return toast("This proposal has no validated map target.", true);
    if (await selectContext(patch)) {
      navigate("/app/map");
      close();
    }
  }

  function buildPreviewResult() {
    const preview = getPreviewState();
    if (!preview || preview.status === "PARENT_RENDERED") return null;
    const card = document.createElement("section");
    card.className = "copilot-preview-result";
    card.dataset.previewState = preview.status;
    if (preview.planId) card.dataset.previewPlanId = preview.planId;
    if (preview.mapPlanId) card.dataset.renderedMapPlanId = preview.mapPlanId;
    const title = document.createElement("strong");
    const description = document.createElement("p");
    if (preview.status === "PREVIEW_RUNNING") {
      title.textContent = "Running evaluated preview";
      description.textContent = "The accepted itinerary remains active while the repair is generated and checked.";
    } else if (preview.status === "PREVIEW_DATA_READY") {
      title.textContent = "Evaluation complete";
      description.textContent = "The eligible child itinerary is ready. Waiting for the map to acknowledge the same plan.";
    } else if (preview.status === "PREVIEW_RENDERED" && preview.planId === preview.mapPlanId) {
      title.textContent = "Preview ready";
      description.textContent = "The itinerary and map are displaying the same independently evaluated child plan. Nothing has been applied.";
    } else if (preview.status === "PREVIEW_INELIGIBLE") {
      title.textContent = "Preview not eligible";
      description.textContent = preview.message || "The evaluated child did not pass the required checks. The accepted itinerary remains active.";
    } else if (["PREVIEW_BLOCKED", "PREVIEW_NOT_EVALUATED"].includes(preview.status)) {
      title.textContent = preview.status === "PREVIEW_BLOCKED" ? "Preview blocked — not evaluated" : "No evaluated preview";
      description.textContent = preview.message || "No child was evaluated. The accepted itinerary remains active.";
    } else {
      title.textContent = "Preview could not be verified";
      description.textContent = preview.message || "The accepted itinerary remains authoritative. Review the visible error before retrying.";
      card.classList.add("is-error");
    }
    card.append(title, description);
    return card;
  }

  function syncPreviewLifecycle() {
    if (getState().mutationRecovery?.blocked) {
      setLifecycle("awaiting_confirmation", "The saved action has not been confirmed. Use Check saved action, Retry saved action or Cancel pending action before making another edit.");
      return;
    }
    const preview = getPreviewState();
    if (!preview) return;
    const latest = lastConversation?.turns?.at(-1)?.interpretation;
    if (!requestActive && !previewActionActive && latest?.state === "proposal_ready"
        && latest.intents?.some(intent => intent.type === "replan_trip" && !intentAlreadyDrafted(intent))) {
      setLifecycle("proposal_ready", getState().session?.draft?.length
        ? "This trip proposal has not been previewed. The map still shows the previous plan. Review the existing draft before previewing this proposal."
        : "This trip proposal has not been previewed. Choose Preview this repair to solve and check the itinerary and map.");
      return;
    }
    const region = $("#copilot-lifecycle");
    if (region?.dataset.state === "awaiting_confirmation") {
      setLifecycle(getState().session?.draft?.length ? "proposal_ready" : "idle",
        "Saved action checked. Review the current draft before continuing.");
      if (region) region.dataset.previewState = preview.status;
    }
    if (preview.status === "PARENT_RENDERED") {
      // Draft mutations withdraw the evaluated preview even when the parent
      // map is already ready and emits no new map acknowledgement.
      const stagedChanged = stagedLifecycle && (!originIsCurrent(stagedLifecycle.origin)
        || !stagedLifecycle.intents.every(intentAlreadyDrafted));
      if ((!region?.dataset.previewState && !stagedChanged) || requestActive || previewActionActive) return;
      setLifecycle(getState().session?.draft?.length ? "proposal_ready" : "idle",
        getState().session?.draft?.length
          ? "Draft changed. Run Preview repair to update the itinerary and map."
          : "Showing the accepted itinerary. No evaluated preview is active.");
    } else if (preview.status === "PREVIEW_RUNNING") {
      setLifecycle("preview_running", "Generating and independently evaluating the child itinerary");
    } else if (preview.status === "PREVIEW_DATA_READY") {
      setLifecycle("preview_rendering", "Waiting for matching itinerary and map acknowledgement");
    } else if (preview.status === "PREVIEW_RENDERED" && preview.planId === preview.mapPlanId) {
      setLifecycle("preview_rendered", "The itinerary and map show the same evaluated child");
    } else if (preview.status === "PREVIEW_INELIGIBLE") {
      setLifecycle("preview_ineligible", preview.message || "The accepted itinerary remains active");
    } else if (["PREVIEW_BLOCKED", "PREVIEW_NOT_EVALUATED"].includes(preview.status)) {
      setLifecycle("failed", preview.message || "No child was evaluated; the accepted itinerary remains active");
    } else if (["PREVIEW_FAILED", "MAP_IDENTITY_FAILED", "MAP_RENDER_FAILED"].includes(preview.status)) {
      setLifecycle("failed", preview.message || "The preview could not be verified; the accepted itinerary remains authoritative");
    }
    if (region) region.dataset.previewState = preview.status;
  }

  function previewStateChanged() {
    conversationOrigin();
    renderConversation(lastConversation);
    syncPreviewLifecycle();
  }

  async function addIntentToDraft(intent) {
    if (previewActionActive || requestActive) return;
    if ((getState().undoPending || getState().mutationRecovery?.blocked || getState().workspaceConflict || getState().workspaceSyncPending)) return;
    const origin = conversationOrigin();
    const actionability = Array.isArray(intent)
      ? copilotCombinedEligibility(intent, getState().workspace, getState().session)
      : copilotDraftability(intent, getState().workspace);
    if (!actionability.allowed) {
      return toast(actionability.reason, true);
    }
    previewActionActive = true;
    setSubmitting(requestActive);
    $$(".copilot-proposal-actions button").forEach((buttonControl) => {
      buttonControl.disabled = true;
    });
    try {
      const added = await addDraft(intent);
      if (!originIsCurrent(origin)) return;
      if (added) {
        const stagedRevision = getState().session.revision;
        const restored = await restore({ announce: false });
        if (!originIsCurrent(origin)) return;
        const stillStaged = (Array.isArray(intent) ? intent : [intent]).every(intentAlreadyDrafted);
        if (restored && stillStaged && getState().session.revision === stagedRevision) {
          setLifecycle("proposal_ready", "Draft staged. The accepted itinerary and map are unchanged");
          stagedLifecycle = { origin, intents: structuredClone(Array.isArray(intent) ? intent : [intent]) };
        }
      } else {
        if (getState().mutationRecovery?.blocked) syncPreviewLifecycle();
        else setLifecycle("failed", "The suggestion was not staged");
      }
    } finally {
      previewActionActive = false;
      setSubmitting(requestActive);
      renderConversation(lastConversation);
      syncPreviewLifecycle();
    }
  }

  async function previewIntentFromCopilot(intent, control) {
    if (previewActionActive || requestActive) return;
    if ((getState().undoPending || getState().mutationRecovery?.blocked || getState().workspaceConflict || getState().workspaceSyncPending)) return;
    const origin = conversationOrigin();
    const eligibility = () => Array.isArray(intent)
      ? copilotCombinedEligibility(intent, getState().workspace, getState().session)
      : copilotOneClickEligibility(intent, getState().workspace, getState().session);
    const initial = eligibility();
    if (!initial.allowed) return toast(initial.reason, true);
    previewActionActive = true;
    setSubmitting(requestActive);
    $$(".copilot-proposal-actions button").forEach((buttonControl) => {
      buttonControl.disabled = true;
    });
    setLifecycle("preview_running", initial.alreadyStaged
      ? "Running the exact staged suggestion"
      : "Staging this suggestion before evaluation");
    try {
      if (!initial.alreadyStaged) {
        const added = await addDraft(intent);
        if (!originIsCurrent(origin)) return;
        if (!added) {
          setLifecycle("failed", "The suggestion was not staged, so preview did not run");
          return;
        }
      }
      if (!originIsCurrent(origin)) return;
      const current = eligibility();
      if (!current.allowed || !current.alreadyStaged) {
        setLifecycle("failed", "The staged operation no longer matches this suggestion");
        return;
      }
      const previewed = await previewDraft(Array.isArray(intent) ? { expectedCopilotOperations: intent.map(item => copilotOperationForIntent(item)) } : { expectedCopilotOperation: copilotOperationForIntent(
        intent, getState().session.selected_stop_id, getState().session.selected_day,
      ) });
      if (!originIsCurrent(origin)) return;
      if (!previewed) {
        setLifecycle("failed", "The draft remains staged, but preview did not complete");
        return;
      }
      await restore({ announce: false });
      if (!originIsCurrent(origin)) return;
      syncPreviewLifecycle();
    } finally {
      previewActionActive = false;
      setSubmitting(requestActive);
      renderConversation(lastConversation);
    }
  }

  function editMessage(message) {
    const input = $("#copilot-input");
    input.value = message;
    resizeInput();
    input.focus();
  }

  function updateProviderPresentation() {
    const provider = providerName();
    $("#copilot-provider-label").textContent = provider;
    $("#copilot-title").textContent = provider;
    const disclosure = $("#copilot-disclosure");
    if (interaction().provider === "openai") {
      disclosure.hidden = false;
      disclosure.textContent = "Before you send: visible trip context, your message, and a bounded recent conversation window are sent to OpenAI. Local conversations expire after 30 days without a new turn. Delete here removes the conversation and dispatch journal; separately saved test/evidence reports are not deleted.";
    } else {
      disclosure.hidden = false;
      disclosure.textContent = "Deterministic demo: chat requests stay on this computer. Local conversations expire after 30 days without a new turn. Delete here removes the conversation and dispatch journal; separately saved test/evidence reports are not deleted.";
    }
  }

  function renderContext() {
    const state = getState();
    const selected = getSelectedStop();
    const selectedCandidate = (state.workspace.draft_capabilities?.candidate_choices || [])
      .find((candidate) => (candidate.candidate_id || candidate.stop_id || candidate.id) === state.session.selected_candidate_id);
    const selectedAlternative = (state.workspace.alternatives || [])
      .find((alternative) => (alternative.plan_id || alternative.id) === state.session.selected_alternative_id);
    const values = [
      `Run: ${state.registry.label}`,
      state.session.selected_day ? `Day ${state.session.selected_day}` : "No day selected",
      selected?.name || "No stop selected",
      ...(state.session.selected_segment_id ? [`Segment: ${state.session.selected_segment_id}`] : []),
      ...(state.session.selected_candidate_id ? [`Candidate: ${selectedCandidate?.label || selectedCandidate?.name || state.session.selected_candidate_id}`] : []),
      ...(state.session.selected_alternative_id ? [`Alternative: ${selectedAlternative?.method_label || selectedAlternative?.label || selectedAlternative?.name || "Selected repair"}`] : []),
      `Revision ${state.session.revision}`,
    ];
    const chips = $("#context-chips");
    chips.replaceChildren(...values.map((value) => {
      const chip = document.createElement("span");
      chip.textContent = value;
      return chip;
    }));
  }

  async function restoreConversation({ announce = true } = {}) {
    const origin = conversationOrigin();
    const epoch = ++restoreEpoch;
    updateProviderPresentation();
    let received = false;
    try {
      const payload = await api(`/api/sessions/${encodeURIComponent(origin.sessionId)}/conversation`);
      if (epoch !== restoreEpoch || !originIsCurrent(origin)) return { status: "superseded" };
      received = true;
      if (!validConversation(payload)) throw new Error("The conversation response did not match its session owner.");
      let projection = null;
      if (payload.session.accepted_plan_id !== getState().session.accepted_plan_id) {
        const beforeRead = { ...getState().session };
        projection = await api(`/api/sessions/${encodeURIComponent(origin.sessionId)}`);
        if (epoch !== restoreEpoch || !originIsCurrent(origin)
            || getState().session.revision !== beforeRead.revision
            || getState().session.conversation_id !== beforeRead.conversation_id) return { status: "superseded" };
        if (!["session_id", "trip_id", "run_id", "revision", "conversation_id", "accepted_plan_id"]
          .every(field => projection?.session?.[field] === payload.session[field])) {
          throw new Error("The saved trip changed during recovery. Refresh to read its current conversation and map.");
        }
      }
      if (!adoptSession(payload.session, projection)) return { status: "superseded" };
      conversationOrigin();
      renderConversation(payload.conversation);
      if (announce) setLifecycle("idle", "Conversation restored");
      return { status: "restored" };
    } catch (error) {
      if (epoch !== restoreEpoch || !originIsCurrent(origin)) return { status: "superseded" };
      if (!lastConversation) renderConversation(null);
      if (announce) setLifecycle("failed", "Conversation could not be restored");
      toast(error.message, true);
      return { status: received || Number.isInteger(error.status) || error.name === "SyntaxError" ? "invalid" : "failed" };
    }
  }

  async function restore(options) {
    return (await restoreConversation(options)).status === "restored";
  }

  async function restoreAfterDeletion() {
    resetConversation();
    const pending = {};
    pendingConversationReset = pending;
    setSubmitting(false);
    try { await restore(); }
    finally {
      if (pendingConversationReset === pending) {
        pendingConversationReset = null;
        setSubmitting(false);
      }
    }
  }

  function resizeInput() {
    const input = $("#copilot-input");
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
  }

  function appendCanonicalTurnLocally(turn) {
    const turns = [...(lastConversation?.turns || [])]
      .filter((candidate) => candidate.client_message_id !== turn.client_message_id);
    renderConversation({ ...(lastConversation || {}), turns: [...turns, turn] });
  }

  function cancelActiveRequest() {
    if (!activeRequest) return;
    setLifecycle("cancelling", "Stopping local wait");
    activeRequest.controller.abort();
  }

  function handleStreamEvent(event, epoch) {
    if (!requestIsCurrent(epoch) || !ephemeralTurn) return;
    if (event.type === "message.accepted") {
      ephemeralTurn.assistant_message = "Request accepted. Reading the current itinerary…";
      ephemeralTurn.state = "sending";
      setLifecycle("sending", "Request accepted");
    } else if (event.type === "assistant.status") {
      ephemeralTurn.assistant_message = event.message;
      ephemeralTurn.state = event.state;
      setLifecycle(event.state, event.message);
    }
    renderConversation(lastConversation);
  }

  async function startMessage(rawMessage, { clientMessageId = null } = {}) {
    const origin = conversationOrigin();
    if (previewActionActive || (getState().undoPending || getState().mutationRecovery?.blocked || getState().workspaceConflict || getState().workspaceSyncPending)) return toast("Wait for the draft action to finish. Your message has not been sent.", true);
    if (pendingConversationReset) return toast("Wait for this conversation to finish resetting. Your message has not been sent.", true);
    if (requestActive || !available()) return;
    const message = rawMessage.trim();
    if (!message) return;
    const state = getState();
    const envelope = {
      expected_revision: state.session.revision,
      client_message_id: clientMessageId || `client_message_${crypto.randomUUID().replaceAll("-", "")}`,
      message,
    };
    const epoch = requestEpoch + 1;
    requestEpoch = epoch;
    const controller = new AbortController();
    activeRequest = { controller, envelope, epoch, origin };
    ephemeralTurn = {
      client_message_id: envelope.client_message_id,
      user_message: message,
      assistant_message: "Preparing the visible trip context…",
      state: "sending",
    };
    const input = $("#copilot-input");
    input.value = "";
    resizeInput();
    setSubmitting(true);
    setLifecycle("sending");
    renderConversation(lastConversation);
    $("#conversation").scrollTop = $("#conversation").scrollHeight;
    try {
      const payload = await streamMessage(
        `/api/sessions/${state.session.session_id}/copilot/messages`,
        {
          body: envelope,
          signal: controller.signal,
          onEvent: (event) => handleStreamEvent(event, epoch),
        },
      );
      if (!requestIsCurrent(epoch)) return;
      if (
        payload?.turn?.client_message_id !== envelope.client_message_id
        || payload.turn.user_message !== message
      ) {
        const error = new Error("The Copilot response did not match this message.");
        error.code = "invalid_copilot_stream";
        throw error;
      }
      adoptSession(payload.session);
      if (!requestIsCurrent(epoch)) return;
      ephemeralTurn = null;
      const refreshed = await restoreConversation({ announce: false });
      if (!requestIsCurrent(epoch) || refreshed.status === "superseded") return;
      if (refreshed.status === "invalid") {
        setLifecycle("failed", "The completed response could not be bound to the current conversation");
        return;
      }
      if (refreshed.status === "failed") {
        appendCanonicalTurnLocally(payload.turn);
        toast("The response completed, but the transcript could not be refreshed. No retry was sent.", true);
      }
      const lifecycle = payload.turn.interpretation?.state || payload.turn.state;
      setLifecycle(lifecycle, payload.turn.assistant_message);
      renderAll();
    } catch (error) {
      if (!requestIsCurrent(epoch)) return;
      const capacityBlock = copilotCapacityBlock(error);
      if (capacityBlock) {
        ephemeralTurn.state = "blocked";
        ephemeralTurn.assistant_message = capacityBlock;
        setLifecycle("blocked", capacityBlock);
        renderConversation(lastConversation);
      } else if (error.name === "AbortError") {
        ephemeralTurn.state = "cancelled";
        ephemeralTurn.assistant_message = "Stopped waiting locally. No itinerary change was made. Provider processing or usage may already have occurred.";
        setLifecycle("cancelled", "You can retry this exact request without creating a new message identity");
        renderConversation(lastConversation);
      } else {
        const refreshed = await restoreConversation({ announce: false });
        if (!requestIsCurrent(epoch) || refreshed.status === "superseded") return;
        const restored = refreshed.status === "restored";
        const recoveredTurn = restored ? (lastConversation?.turns || []).find(
          (turn) => turn.client_message_id === envelope.client_message_id,
        ) : null;
        if (!recoveredTurn) {
          ephemeralTurn.state = "cancelled";
          ephemeralTurn.assistant_message = `The result was not confirmed (${error.message}). Retry will reconcile this same request identity without another provider attempt. Edit and send a new message only if you want a new request.`;
          renderConversation(lastConversation);
          setLifecycle("cancelled", error.message);
        } else {
          ephemeralTurn = null;
          renderConversation(lastConversation);
          const lifecycle = recoveredTurn.interpretation?.state || recoveredTurn.state;
          setLifecycle(lifecycle, recoveredTurn.assistant_message);
        }
      }
    } finally {
      if (activeRequest?.epoch === epoch) {
        if (ephemeralTurn?.client_message_id === envelope.client_message_id
            && !["blocked", "cancelled"].includes(ephemeralTurn.state)) {
          ephemeralTurn = null;
          renderConversation(lastConversation);
          setLifecycle("idle", "The trip changed. Review the current conversation before sending again.");
        }
        activeRequest = null;
        setSubmitting(false);
        resizeInput();
        input.focus();
      }
    }
  }

  async function submit(event) {
    event.preventDefault();
    if (requestActive) {
      cancelActiveRequest();
      return;
    }
    const input = $("#copilot-input");
    await startMessage(input.value);
  }

  function open(opener) {
    if (!available()) return toast("The selected Copilot provider is unavailable.", true);
    dockOpener = opener;
    const dock = $("#copilot-dock");
    dock.hidden = false;
    updateDockSemantics();
    $("#copilot-input").focus();
  }

  function close() {
    const dock = $("#copilot-dock");
    dock.hidden = true;
    dock.setAttribute("role", "complementary");
    dock.removeAttribute("aria-modal");
    dockOpener?.focus();
  }

  function updateDockSemantics() {
    const dock = $("#copilot-dock");
    const mobile = window.matchMedia("(max-width: 820px)").matches;
    dock.setAttribute("role", mobile ? "dialog" : "complementary");
    if (mobile) dock.setAttribute("aria-modal", "true");
    else dock.removeAttribute("aria-modal");
  }

  function handleDockKeydown(event) {
    const dock = $("#copilot-dock");
    if (dock.hidden) return;
    if (event.key === "Escape") {
      event.preventDefault();
      close();
      return;
    }
    if (event.key !== "Tab" || !window.matchMedia("(max-width: 820px)").matches) return;
    const focusable = $$(
      "button:not([disabled]), textarea:not([disabled]), [href], [tabindex]:not([tabindex='-1'])",
      dock,
    ).filter((control) => !control.hidden && control.getClientRects().length > 0);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable.at(-1);
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function openSettings(opener) {
    settingsOpener = opener;
    $("#transcript-settings-dialog").showModal();
    $("#delete-current-conversation").focus();
  }

  function closeSettings() {
    $("#transcript-settings-dialog").close();
    settingsOpener?.focus();
  }

  async function deleteCurrent() {
    if (!window.confirm("Delete this local conversation? This cannot be undone.")) return;
    const origin = conversationOrigin();
    const state = getState();
    try {
      const payload = await api(`/api/sessions/${state.session.session_id}/conversation`, {
        method: "DELETE",
        body: { expected_revision: state.session.revision },
      });
      if (!originIsCurrent(origin)) return;
      adoptSession(payload.session);
      closeSettings();
      await restoreAfterDeletion();
      if (getState().session?.session_id !== origin.sessionId) return;
      toast("This conversation was deleted. Your trip and draft were preserved.");
    } catch (error) { if (originIsCurrent(origin)) toast(error.message, true); }
  }

  async function deleteAll() {
    if (!window.confirm("Delete every local Copilot conversation? This cannot be undone.")) return;
    const origin = conversationOrigin();
    const state = getState();
    try {
      const payload = await api("/api/conversations", {
        method: "DELETE",
        headers: { "X-Session-Id": state.session.session_id },
        body: { expected_revision: state.session.revision, confirmation: "delete_all_conversations" },
      });
      if (!originIsCurrent(origin)) return;
      adoptSession(payload.session);
      closeSettings();
      await restoreAfterDeletion();
      if (getState().session?.session_id !== origin.sessionId) return;
      toast(`${payload.deleted_count} local conversation${payload.deleted_count === 1 ? " was" : "s were"} deleted.`);
    } catch (error) { if (originIsCurrent(origin)) toast(error.message, true); }
  }

  function bind() {
    $("#open-copilot").addEventListener("click", (event) => open(event.currentTarget));
    $("#close-copilot").addEventListener("click", close);
    $("#copilot-dock").addEventListener("keydown", handleDockKeydown);
    window.addEventListener("resize", () => {
      if (!$("#copilot-dock").hidden) updateDockSemantics();
    });
    $("#copilot-form").addEventListener("submit", submit);
    $("#copilot-input").addEventListener("input", resizeInput);
    $("#copilot-input").addEventListener("keydown", (event) => {
      if (
        event.key === "Enter"
        && !event.shiftKey
        && !event.isComposing
        && !requestActive
      ) {
        event.preventDefault();
        $("#copilot-form").requestSubmit();
      }
    });
    $$(".quick-prompts button").forEach((control) => control.addEventListener("click", () => {
      $("#copilot-input").value = control.dataset.prompt;
      resizeInput();
      $("#copilot-input").focus();
    }));
    $("#transcript-settings-button").addEventListener("click", (event) => openSettings(event.currentTarget));
    $("#close-transcript-settings").addEventListener("click", closeSettings);
    $("#delete-current-conversation").addEventListener("click", deleteCurrent);
    $("#delete-all-conversations").addEventListener("click", deleteAll);
    $("#transcript-settings-dialog").addEventListener("close", () => settingsOpener?.focus());
  }

  function update() {
    conversationOrigin();
    updateProviderPresentation();
    renderContext();
    setSubmitting(requestActive);
    $("#open-copilot").disabled = !available();
    $("#copilot-input").disabled = !available();
    $$(".quick-prompts button").forEach((control) => {
      control.disabled = requestActive || previewActionActive || !available() || Boolean(pendingConversationReset) || Boolean((getState().undoPending || getState().mutationRecovery?.blocked || getState().workspaceConflict || getState().workspaceSyncPending));
    });
    if (getState().mutationRecovery?.blocked) {
      $$(".copilot-proposal-actions button").forEach(control => { control.disabled = true; });
    }
    if (getState().mutationRecovery?.blocked || getPreviewState()?.status === "PARENT_RENDERED") syncPreviewLifecycle();
  }

  return { available, bind, close, open, previewStateChanged, restore, update,
    isBusy: () => requestActive || previewActionActive || Boolean(pendingConversationReset) };
}
