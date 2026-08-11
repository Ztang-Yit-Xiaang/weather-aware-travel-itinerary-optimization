import {
  createGeographicMapController,
  synchronizeMapCameras,
} from "./map-controller-v2.js?v=20260810-stability5";

let activeCompare = null;
let compareGeneration = 0;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function displayMetric(value, suffix = "") {
  if (value === null || value === undefined) return "Unavailable";
  const formatted = typeof value === "number"
    ? Number(value.toFixed(2)).toString()
    : String(value);
  return `${formatted}${suffix}`;
}

function exactBaseline(workspace, session) {
  const originals = (workspace?.geography?.plans || []).filter((plan) => plan.role === "original");
  if (originals.length !== 1) return null;
  return originals[0].plan_id === session?.accepted_plan_id ? originals[0] : null;
}

/** Resolve only the explicitly selected, exact alternative. Never choose a fallback option. */
export function resolveCompareSelection(workspace, session) {
  const selectedId = session?.selected_alternative_id;
  if (!selectedId) {
    return Object.freeze({ state: "unavailable", code: "compare_option_not_selected" });
  }
  const matchingOptions = (workspace?.alternatives || []).filter(
    (option) => option.plan_id === selectedId,
  );
  if (matchingOptions.length !== 1) {
    return Object.freeze({ state: "unavailable", code: "compare_option_not_found" });
  }
  const option = matchingOptions[0];
  const matchingPlans = (workspace?.geography?.plans || []).filter(
    (plan) => plan.plan_id === selectedId,
  );
  if (matchingPlans.length !== 1) {
    return Object.freeze({ state: "unavailable", code: "compare_option_geography_unavailable", option });
  }
  const plan = matchingPlans[0];
  if (!option.plan_content_hash || plan.content_hash !== option.plan_content_hash) {
    return Object.freeze({ state: "unavailable", code: "compare_option_hash_mismatch", option });
  }
  const evidence = workspace?.alternative_evidence?.[selectedId] || null;
  const evidenceExact = Boolean(
    evidence
    && evidence.plan_id === selectedId
    && evidence.plan_content_hash === option.plan_content_hash,
  );
  return Object.freeze({
    state: "selected",
    code: evidenceExact ? "compare_option_ready" : "compare_option_evidence_unavailable",
    option,
    plan,
    evidence: evidenceExact ? evidence : null,
    evidenceExact,
  });
}

function optionCard(option, selectedId) {
  const selected = option.plan_id === selectedId;
  const eligible = option.status === "eligible" || option.display_status === "Eligible";
  const rankingEligible = option.ranking_eligible === true;
  const status = option.display_status || option.status || "Unavailable";
  const description = eligible
    ? `${displayMetric(option.route_total_minutes, " route minutes")} · ${displayMetric(option.weighted_edit_cost, " edit cost")}`
    : option.failure_reason || "This option is inspectable but is not decision eligible.";
  return `<article class="alternative-card ${option.role === "recommended" && rankingEligible ? "recommended" : ""} ${eligible ? "" : "ineligible"}" data-option-card="${escapeHtml(option.plan_id)}">
    <span class="status-pill ${eligible ? "success" : "warning"}">${escapeHtml(status)}</span>
    <h3>${escapeHtml(option.method_label || option.plan_id)}</h3>
    <p>${escapeHtml(description)}</p>
    <button type="button" data-compare-option="${escapeHtml(option.plan_id)}" aria-pressed="${selected}">${selected ? "Previewing on maps" : "Preview this option"}</button>
  </article>`;
}

function selectedMetricsTable(baseline, resolution) {
  const option = resolution.state === "selected" ? resolution.option : null;
  const baselineRouteState = baseline?.coverage?.status === "complete" ? "Validated" : "Unavailable";
  const rows = [
    ["Independent eligibility", "required", "Current accepted baseline", option?.display_status || option?.status || "Unavailable"],
    ["Strict route time", "lower", "Unavailable", displayMetric(option?.route_total_minutes, option?.route_total_minutes == null ? "" : " minutes")],
    ["Ownership-weighted edit cost", "lower", "0 (reference)", displayMetric(option?.weighted_edit_cost)],
    ["Road-validated route", "required", baselineRouteState, option ? (option.route_validated ? "Validated" : "Unavailable") : "Unavailable"],
    ["Primary tradeoff", "review", "Keep the current trip", option ? `${displayMetric(option.route_total_minutes, " route minutes")} with ${displayMetric(option.weighted_edit_cost, " edit cost")}` : "Unavailable"],
  ];
  return `<table class="comparison-table"><thead><tr><th>Metric</th><th>Direction</th><th>Accepted baseline</th><th>Selected option</th></tr></thead><tbody>${rows.map((row) => `<tr>
    <th scope="row">${escapeHtml(row[0])}</th><td>${escapeHtml(row[1])}</td>${row.slice(2).map((value) => `<td class="${value === "Unavailable" ? "unavailable" : ""}">${escapeHtml(value)}</td>`).join("")}
  </tr>`).join("")}</tbody></table>`;
}

function diffSummary(resolution) {
  if (resolution.state !== "selected") {
    return `<article class="evidence-card"><h3>Choose an option</h3><p>Select one available repair above to preview its route, metrics, and changes alongside the current trip.</p></article>`;
  }
  const diff = resolution.evidence?.diff;
  if (!diff) {
    return `<article class="evidence-card"><h3>Diff unavailable</h3><p>The exact option map is inspectable, but its diff/evidence identity could not be verified.</p><code>${escapeHtml(resolution.code)}</code></article>`;
  }
  return `<article class="evidence-card"><h3>What changes</h3>
    <p>${diff.added_stops?.length ?? 0} added · ${diff.deleted_stops?.length ?? 0} removed · ${diff.day_moves?.length ?? 0} day moves · ${diff.road_changes?.length ?? 0} route changes.</p>
    <p>Technical plan, diff, hash, method, and certificate identities are available in Advanced Evidence.</p>
  </article>`;
}

function textFallbackMarkup(payload) {
  const legs = payload?.route_legs || [];
  const gaps = payload?.gaps || [];
  return `<p>${escapeHtml(payload?.summary || "Text route evidence is unavailable.")}</p>
    ${legs.length ? `<ol>${legs.map((leg) => `<li>${escapeHtml(leg.text)}</li>`).join("")}</ol>` : ""}
    ${gaps.length ? `<ul>${gaps.map((gap) => `<li>${escapeHtml(gap.text)}</li>`).join("")}</ul>` : ""}`;
}

function setPaneFallback(pane, payload) {
  const output = pane.querySelector("[data-compare-text]");
  if (output) output.innerHTML = textFallbackMarkup(payload);
}

function setPaneStatus(pane, payload) {
  const status = pane.querySelector("[data-compare-map-status]");
  if (!status) return;
  status.textContent = payload.state === "failed"
    ? `Map unavailable (${payload.code}). Use the text route evidence in this pane.`
    : payload.state === "ready_with_gaps"
      ? "Map ready with explicit unvalidated route gaps."
      : "Map ready with road-validated route evidence.";
  status.setAttribute("role", payload.state === "failed" ? "alert" : "status");
  if (payload.state === "ready" || payload.state === "ready_with_gaps") {
    requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
  }
}

function mapPane(kind, title, plan, unavailableCode = null) {
  const identity = plan
    ? `<span class="status-pill success">Route evidence available</span>`
    : `<span class="status-pill warning">Choose an available option</span>`;
  return `<article class="evidence-card" data-compare-map-pane="${kind}">
    <h3>${escapeHtml(title)}</h3>
    <p>${identity}</p>
    <p data-compare-map-status role="status">${plan ? "Loading map tiles and route layers…" : "No selected route is available for this pane."}</p>
    <div class="map-stage compare-map-stage" ${plan ? "" : "hidden"}>
      <div class="compare-map-canvas" data-compare-map-canvas></div>
    </div>
    <details><summary>Text route evidence</summary><div data-compare-text><p>${plan ? "Preparing text route evidence…" : "Unavailable"}</p></div></details>
  </article>`;
}

function configureMobileToggle(root, baselinePane, optionPane, initialView) {
  const media = window.matchMedia("(max-width: 820px)");
  const buttons = [...root.querySelectorAll("[data-compare-mobile-view]")];
  let active = initialView;
  const update = () => {
    const mobile = media.matches;
    root.querySelector("[data-compare-mobile-toggle]").hidden = !mobile;
    baselinePane.hidden = mobile && active !== "baseline";
    optionPane.hidden = mobile && active !== "option";
    buttons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.compareMobileView === active)));
  };
  buttons.forEach((button) => button.addEventListener("click", () => {
    active = button.dataset.compareMobileView;
    update();
    requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
  }));
  media.addEventListener("change", update);
  update();
  return () => media.removeEventListener("change", update);
}

function startCompareMaps(root, workspace, mapConfig, baseline, resolution) {
  const baselinePane = root.querySelector('[data-compare-map-pane="baseline"]');
  const optionPane = root.querySelector('[data-compare-map-pane="option"]');
  const disposeMobile = configureMobileToggle(
    root,
    baselinePane,
    optionPane,
    resolution.state === "selected" ? "option" : "baseline",
  );
  if (!baseline) {
    return { dispose: disposeMobile };
  }
  const readyPanes = new Set();
  let optionController = null;
  let stopCameraSync = null;
  let disposed = false;
  const recordReady = (paneId, payload) => {
    if (payload.state !== "ready" && payload.state !== "ready_with_gaps") return;
    readyPanes.add(paneId);
    if (readyPanes.size !== 2 || stopCameraSync || disposed || !optionController) return;
    requestAnimationFrame(() => {
      if (disposed || stopCameraSync || !optionController) return;
      stopCameraSync = synchronizeMapCameras(baselineController, optionController);
    });
  };
  const baselineController = createGeographicMapController({
    container: baselinePane.querySelector("[data-compare-map-canvas]"),
    mapConfig,
    paneId: "compare-baseline",
    onTextFallback: (payload) => setPaneFallback(baselinePane, payload),
    onStatus: (payload) => {
      setPaneStatus(baselinePane, payload);
      recordReady("baseline", payload);
    },
  });
  baselineController.render({
    geography: workspace.geography,
    activePlanId: baseline.plan_id,
  });
  if (resolution.state === "selected") {
    optionController = createGeographicMapController({
      container: optionPane.querySelector("[data-compare-map-canvas]"),
      mapConfig,
      paneId: "compare-option",
      onTextFallback: (payload) => setPaneFallback(optionPane, payload),
      onStatus: (payload) => {
        setPaneStatus(optionPane, payload);
        recordReady("option", payload);
      },
    });
    optionController.render({
      geography: workspace.geography,
      activePlanId: resolution.plan.plan_id,
    });
  }
  return {
    dispose() {
      disposed = true;
      disposeMobile();
      stopCameraSync?.();
      baselineController.dispose();
      optionController?.dispose();
    },
  };
}

export function disposeCompareWorkspace() {
  compareGeneration += 1;
  activeCompare?.dispose();
  activeCompare = null;
}

/** Render Compare without mutating a decision or navigating away on option selection. */
export function renderCompareWorkspace({
  container,
  workspace,
  session,
  mapConfig,
  onBack,
  onSelectAlternative,
  onOpenEvidence,
}) {
  disposeCompareWorkspace();
  const baseline = exactBaseline(workspace, session);
  const resolution = resolveCompareSelection(workspace, session);
  const selectedId = resolution.state === "selected" ? resolution.option.plan_id : null;
  const baselineUnavailable = baseline ? null : "compare_baseline_identity_mismatch";
  container.innerHTML = `<div class="route-view" data-compare-workspace>
    <header class="route-heading"><div><span class="eyebrow">Trip choices</span><h2>Compare repairs</h2><p>Review the current trip and one repair side by side. The recommended eligible repair is selected automatically when Compare opens. Ineligible options remain inspectable but cannot be chosen.</p></div><button class="secondary" id="compare-back" type="button">Back to trip</button></header>
    <div class="action-stack" data-compare-mobile-toggle hidden aria-label="Compare map view">
      <button type="button" data-compare-mobile-view="baseline" aria-pressed="false">Baseline map</button>
      <button type="button" data-compare-mobile-view="option" aria-pressed="true">Option map</button>
    </div>
    <div class="evidence-grid" data-compare-map-grid>
      ${mapPane("baseline", "Original route", baseline, baselineUnavailable)}
      ${mapPane("option", "Selected repair route", resolution.state === "selected" ? resolution.plan : null, resolution.code)}
    </div>
    <div class="alternative-grid">
      <article class="alternative-card"><span class="status-pill">Current trip</span><h3>Keep original</h3><p>The current accepted itinerary, shown as the reference route.</p><button type="button" disabled>Baseline map shown</button></article>
      ${(workspace?.alternatives || []).map((option) => optionCard(option, selectedId)).join("")}
    </div>
    <div class="action-stack compare-actions">
      <button type="button" id="compare-choose" disabled title="Acceptance remains disabled until W5">Choose option (not enabled)</button>
      <button type="button" id="compare-evidence" ${resolution.evidenceExact ? "" : "disabled"}>View advanced Evidence</button>
    </div>
    ${diffSummary(resolution)}
    ${selectedMetricsTable(baseline, resolution)}
  </div>`;
  container.querySelector("#compare-back").addEventListener("click", onBack);
  container.querySelectorAll("[data-compare-option]").forEach((button) => {
    button.addEventListener("click", () => onSelectAlternative(button.dataset.compareOption));
  });
  const evidenceButton = container.querySelector("#compare-evidence");
  evidenceButton.addEventListener("click", () => {
    if (resolution.evidenceExact) onOpenEvidence(resolution.option.plan_id);
  });
  const root = rootOrContainer(container);
  const generation = compareGeneration;
  requestAnimationFrame(() => {
    if (generation !== compareGeneration || !root.isConnected) return;
    activeCompare = startCompareMaps(root, workspace, mapConfig, baseline, resolution);
    requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
  });
  return resolution;
}

function rootOrContainer(container) {
  return container.querySelector("[data-compare-workspace]") || container;
}
