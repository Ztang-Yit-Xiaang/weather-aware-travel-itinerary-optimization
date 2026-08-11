const EDIT_INTENT_TYPES = new Set([
  "keep_stop",
  "lock_stop",
  "mark_flexible",
  "move_day",
  "route_feedback",
  "replace_nearby",
  "add_candidate",
]);

const LIFECYCLE_COPY = {
  sending: "Sending",
  interpreting: "Interpreting",
  clarification_required: "Clarification required",
  permission_required: "Permission required",
  proposal_ready: "Proposal ready",
  refused: "Refused",
  failed: "Failed",
};

export function createCopilotController({
  getState,
  api,
  selectContext,
  addDraft,
  previewDraft,
  navigate,
  toast,
  renderAll,
}) {
  let requestActive = false;
  let dockOpener = null;
  let settingsOpener = null;
  let lastInterpretation = null;

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

  function setLifecycle(state, detail = "") {
    const region = $("#copilot-lifecycle");
    if (!region) return;
    const label = LIFECYCLE_COPY[state] || state;
    region.textContent = detail ? `${label}: ${detail}` : label;
    region.dataset.state = state;
  }

  function setSubmitting(active) {
    requestActive = active;
    const submit = $("#copilot-form button[type='submit']");
    if (submit) submit.disabled = active || !available();
    const input = $("#copilot-input");
    if (input) input.setAttribute("aria-busy", String(active));
  }

  function appendMessage(role, text, label, turn = null) {
    const article = document.createElement("article");
    article.className = `message message-${role}`;
    const avatar = document.createElement("span");
    avatar.className = "avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = role === "user" ? "Y" : "C";
    const copy = document.createElement("div");
    const strong = document.createElement("strong");
    strong.textContent = label;
    const paragraph = document.createElement("p");
    paragraph.textContent = text;
    copy.append(strong, paragraph);
    if (turn?.interpretation?.state === "proposal_ready") {
      copy.append(buildProposalActions(turn.interpretation));
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
    container.replaceChildren();
    lastInterpretation = null;
    for (const turn of conversation?.turns || []) {
      appendMessage("user", turn.user_message, "You");
      appendMessage("assistant", turn.assistant_message, turn.provider === "openai" ? "OpenAI Copilot" : "Deterministic demo", turn);
      if (turn.interpretation) lastInterpretation = turn.interpretation;
    }
    if (!conversation?.turns?.length) renderEmptyConversation();
    container.scrollTop = container.scrollHeight;
  }

  function button(label, action, disabled = false) {
    const control = document.createElement("button");
    control.type = "button";
    control.textContent = label;
    control.disabled = disabled;
    control.addEventListener("click", action);
    return control;
  }

  function intentTarget(intent) {
    if (intent.type === "route_feedback") return "selected_route";
    if (intent.type === "add_candidate") return intent.candidate_id || null;
    return intent.target_stop_id || getState().session.selected_stop_id || null;
  }

  function intentAlreadyDrafted(intent) {
    const target = intentTarget(intent);
    return Boolean(target) && getState().session.draft.some(
      (operation) => operation.type === intent.type && operation.target === target,
    );
  }

  function buildProposalActions(interpretation) {
    const actions = document.createElement("div");
    actions.className = "copilot-proposal-actions";
    actions.setAttribute("aria-label", "Copilot proposal actions");
    const highlights = interpretation.highlights || {};
    const hasHighlights = ["day_ids", "stop_ids", "segment_ids", "candidate_ids"]
      .some((field) => Array.isArray(highlights[field]) && highlights[field].length > 0);
    if (hasHighlights) actions.append(button("Show on map", () => showOnMap(highlights)));

    const editIntent = (interpretation.intents || []).find((intent) => EDIT_INTENT_TYPES.has(intent.type));
    if (editIntent) {
      const alreadyAdded = intentAlreadyDrafted(editIntent);
      actions.append(button(
        alreadyAdded ? "Added to draft" : "Add to draft",
        (event) => addIntentToDraft(editIntent, event.currentTarget),
        alreadyAdded,
      ));
    }

    actions.append(
      button("Preview repair", previewFromCopilot, getState().session.draft.length === 0),
      button("Compare", () => navigate("/app/compare")),
      button("Review evidence", () => navigate("/app/evidence")),
    );
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

  async function showOnMap(highlights) {
    const patch = {};
    const day = firstValidated(highlights, "day_ids");
    const stop = firstValidated(highlights, "stop_ids");
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

  async function addIntentToDraft(intent, control) {
    if (!window.confirm("Add this typed Copilot suggestion to your local draft? The accepted plan will remain unchanged.")) return;
    control.disabled = true;
    const added = await addDraft(intent);
    if (!added) control.disabled = false;
    if (added) {
      await restore({ announce: false });
    }
  }

  async function previewFromCopilot() {
    if (!getState().session.draft.length) return toast("Add a typed draft operation before previewing a repair.", true);
    if (!window.confirm("Run the deterministic repair and independent evaluation for the current draft?")) return;
    await previewDraft();
  }

  function updateProviderPresentation() {
    const provider = providerName();
    $("#copilot-provider-label").textContent = provider;
    $("#copilot-title").textContent = provider;
    const disclosure = $("#copilot-disclosure");
    if (interaction().provider === "openai") {
      disclosure.hidden = false;
      disclosure.textContent = "Before you send: visible trip context, your message, and a bounded recent conversation window are sent to OpenAI. Full transcripts are stored locally for 30 days and can be deleted here.";
    } else {
      disclosure.hidden = false;
      disclosure.textContent = "Deterministic demo: requests stay on this computer. Full transcripts are stored locally for 30 days and can be deleted here.";
    }
  }

  function renderContext() {
    const state = getState();
    const selected = state.workspace.timeline.flatMap((day) => day.stops)
      .find((stop) => stop.id === state.session.selected_stop_id);
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
      ...(state.session.selected_alternative_id ? [`Alternative: ${selectedAlternative?.label || state.session.selected_alternative_id}`] : []),
      `Revision ${state.session.revision}`,
    ];
    const chips = $("#context-chips");
    chips.replaceChildren(...values.map((value) => {
      const chip = document.createElement("span");
      chip.textContent = value;
      return chip;
    }));
  }

  async function restore({ announce = true } = {}) {
    updateProviderPresentation();
    const state = getState();
    try {
      const payload = await api(`/api/sessions/${encodeURIComponent(state.session.session_id)}/conversation`);
      state.session = payload.session;
      renderConversation(payload.conversation);
      if (announce) setLifecycle("idle", "Conversation restored");
    } catch (error) {
      renderConversation(null);
      if (announce) setLifecycle("failed", "Conversation could not be restored");
      toast(error.message, true);
    }
  }

  async function submit(event) {
    event.preventDefault();
    if (requestActive || !available()) return;
    const input = $("#copilot-input");
    const message = input.value.trim();
    if (!message) return;
    const clientMessageId = `client_message_${crypto.randomUUID().replaceAll("-", "")}`;
    setSubmitting(true);
    setLifecycle("sending");
    try {
      setLifecycle("interpreting");
      const state = getState();
      const payload = await api(`/api/sessions/${state.session.session_id}/copilot/messages`, {
        method: "POST",
        body: { expected_revision: state.session.revision, client_message_id: clientMessageId, message },
      });
      state.session = payload.session;
      input.value = "";
      await restore();
      const lifecycle = payload.turn.interpretation?.state || payload.turn.state;
      setLifecycle(lifecycle, payload.turn.assistant_message);
      renderAll();
    } catch (error) {
      input.value = message;
      try { await restore({ announce: false }); } catch { /* restore already reports safely */ }
      setLifecycle("failed", error.message);
    } finally {
      setSubmitting(false);
      input.focus();
    }
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
    const state = getState();
    try {
      const payload = await api(`/api/sessions/${state.session.session_id}/conversation`, {
        method: "DELETE",
        body: { expected_revision: state.session.revision },
      });
      state.session = payload.session;
      closeSettings();
      await restore();
      toast("This conversation was deleted. Your trip and draft were preserved.");
    } catch (error) { toast(error.message, true); }
  }

  async function deleteAll() {
    if (!window.confirm("Delete every local Copilot conversation? This cannot be undone.")) return;
    const state = getState();
    try {
      const payload = await api("/api/conversations", {
        method: "DELETE",
        headers: { "X-Session-Id": state.session.session_id },
        body: { expected_revision: state.session.revision, confirmation: "delete_all_conversations" },
      });
      state.session = payload.session;
      closeSettings();
      await restore();
      toast(`${payload.deleted_count} local conversation${payload.deleted_count === 1 ? " was" : "s were"} deleted.`);
    } catch (error) { toast(error.message, true); }
  }

  function bind() {
    $("#open-copilot").addEventListener("click", (event) => open(event.currentTarget));
    $("#mobile-copilot")?.addEventListener("click", (event) => open(event.currentTarget));
    $("#close-copilot").addEventListener("click", close);
    $("#copilot-dock").addEventListener("keydown", handleDockKeydown);
    window.addEventListener("resize", () => {
      if (!$("#copilot-dock").hidden) updateDockSemantics();
    });
    $("#copilot-form").addEventListener("submit", submit);
    $$(".quick-prompts button").forEach((control) => control.addEventListener("click", () => {
      $("#copilot-input").value = control.dataset.prompt;
      $("#copilot-input").focus();
    }));
    $("#transcript-settings-button").addEventListener("click", (event) => openSettings(event.currentTarget));
    $("#close-transcript-settings").addEventListener("click", closeSettings);
    $("#delete-current-conversation").addEventListener("click", deleteCurrent);
    $("#delete-all-conversations").addEventListener("click", deleteAll);
    $("#transcript-settings-dialog").addEventListener("close", () => settingsOpener?.focus());
  }

  function update() {
    updateProviderPresentation();
    renderContext();
    setSubmitting(requestActive);
    $("#open-copilot").disabled = !available();
    if ($("#mobile-copilot")) $("#mobile-copilot").disabled = !available();
    $("#copilot-input").disabled = !available();
    $$(".quick-prompts button").forEach((control) => { control.disabled = !available(); });
  }

  return { available, bind, close, open, restore, update };
}
