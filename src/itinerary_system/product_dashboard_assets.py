"""Static HTML, CSS, and browser runtimes for the product dashboard."""

from __future__ import annotations


def product_dashboard_html() -> str:
    """Return the semantic product dashboard shell."""

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="color-scheme" content="light" />
  <title>Itinerary Repair Review</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIINfQ3ynh9tOeOH9CHf9Zy0YTHUklFQ4VY="
        crossorigin="" />
  <link rel="stylesheet" href="assets/product.css" />
</head>
<body data-product-mode="customer">
  <a class="skip-link" href="#product-main">Skip to itinerary review</a>
  <header class="product-header">
    <div>
      <p class="eyebrow">Read-only artifact review</p>
      <h1>Itinerary Repair Review</h1>
      <p id="trip-subtitle" class="header-subtitle">Loading immutable run artifacts…</p>
    </div>
    <div class="header-actions" aria-label="Dashboard mode">
      <div class="mode-switch" role="group" aria-label="Presentation mode">
        <button type="button" data-mode="customer" aria-pressed="true">Trip view</button>
        <button type="button" data-mode="research" aria-pressed="false">Evidence view</button>
      </div>
      <span class="read-only-badge">Read-only</span>
    </div>
  </header>
  <div id="product-status" class="sr-status" role="status" aria-live="polite">
    Loading product dashboard.
  </div>
  <main id="product-main" class="product-grid">
    <section id="issue-region" class="panel issue-region" aria-labelledby="issue-title">
      <div class="section-heading">
        <div>
          <p class="section-kicker">What happened</p>
          <h2 id="issue-title">Issue and result</h2>
        </div>
        <div id="truth-state-chips" class="state-chips" aria-label="Artifact states"></div>
      </div>
      <div id="issue-content"></div>
    </section>

    <section id="timeline-region" class="panel timeline-region" aria-labelledby="timeline-title">
      <div class="section-heading">
        <div>
          <p class="section-kicker">Accepted itinerary</p>
          <h2 id="timeline-title">Day by day</h2>
        </div>
        <span id="day-count" class="metric-badge"></span>
      </div>
      <div id="timeline-list" class="timeline-list" aria-label="Itinerary days"></div>
    </section>

    <section id="repair-region" class="panel repair-region" aria-labelledby="repair-title">
      <div class="section-heading">
        <div>
          <p class="section-kicker">Recommended result</p>
          <h2 id="repair-title">Repair and tradeoffs</h2>
        </div>
      </div>
      <div id="repair-content"></div>
    </section>

    <section id="comparison-region" class="panel comparison-region" aria-labelledby="comparison-title">
      <div class="section-heading">
        <div>
          <p class="section-kicker">What changed</p>
          <h2 id="comparison-title">Original and repaired plan</h2>
        </div>
        <span class="direction-note">Directions are shown per metric</span>
      </div>
      <div id="comparison-content"></div>
      <div id="method-landscape" class="method-landscape"></div>
    </section>

    <section id="evidence-region" class="panel evidence-region" aria-labelledby="evidence-title">
      <div class="section-heading">
        <div>
          <p class="section-kicker">Why trust it</p>
          <h2 id="evidence-title">Certificate and evidence</h2>
        </div>
      </div>
      <div id="evidence-content"></div>
    </section>

    <section id="map-region" class="panel map-region" aria-labelledby="map-title">
      <div class="section-heading">
        <div>
          <p class="section-kicker">Route context</p>
          <h2 id="map-title">Selected day map</h2>
        </div>
        <span id="map-day-label" class="metric-badge"></span>
      </div>
      <div id="product-map" role="img" aria-label="Original and repaired itinerary route map"></div>
      <p id="map-status" class="map-status" role="status"></p>
      <details class="map-alternative">
        <summary>Text alternative for the map</summary>
        <p id="map-alternative-text"></p>
      </details>
    </section>

    <section id="research-region" class="panel research-region" data-research-only
             aria-labelledby="research-title" hidden>
      <div class="section-heading">
        <div>
          <p class="section-kicker">Artifact provenance</p>
          <h2 id="research-title">Research evidence</h2>
        </div>
      </div>
      <div id="research-content"></div>
    </section>
  </main>
  <footer>
    This product artifact presents existing evidence. It does not execute, accept,
    persist, or grant permission for a repair.
  </footer>
  <script src="assets/product_data.js"></script>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
          crossorigin=""></script>
  <script src="assets/product_ui.js"></script>
  <script src="assets/product_map.js"></script>
</body>
</html>
"""


def product_dashboard_stylesheet() -> str:
    """Return mobile-first product styles."""

    return r""":root {
  --ink: #17211b;
  --muted: #59645c;
  --paper: #f4f1e8;
  --surface: #fffdf7;
  --surface-strong: #ffffff;
  --line: #d7d2c5;
  --line-strong: #aaa596;
  --teal: #0b6f68;
  --teal-soft: #d9eeea;
  --amber: #854306;
  --amber-soft: #f7e5cb;
  --red: #a63a32;
  --red-soft: #f5ded9;
  --blue: #245f87;
  --blue-soft: #dceaf4;
  --neutral-soft: #ebe8df;
  --focus: #0b5fff;
  --shadow: 0 10px 30px rgba(31, 42, 34, 0.08);
  --radius-sm: 6px;
  --radius-md: 12px;
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --content: 1520px;
  --z-skip: 100;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 16px;
  line-height: 1.55;
}
html, body { min-width: 0; overflow-x: clip; }
button, input, select { font: inherit; }
button { min-height: 44px; }
button, summary, a { -webkit-tap-highlight-color: transparent; }
button:focus-visible, summary:focus-visible, a:focus-visible, [tabindex]:focus-visible {
  outline: 3px solid var(--focus);
  outline-offset: 3px;
}
.skip-link {
  position: fixed;
  left: var(--space-3);
  top: var(--space-3);
  z-index: var(--z-skip);
  padding: var(--space-3) var(--space-4);
  background: var(--ink);
  color: white;
  transform: translateY(-160%);
}
.skip-link:focus { transform: translateY(0); }
.product-header {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
  max-width: var(--content);
  margin: 0 auto;
  padding: var(--space-5) var(--space-4) var(--space-4);
  border-bottom: 1px solid var(--line-strong);
}
.eyebrow, .section-kicker {
  margin: 0 0 var(--space-1);
  color: var(--teal);
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
h1, h2, h3, p { margin-top: 0; }
h1 {
  margin-bottom: var(--space-1);
  font-family: ui-serif, Georgia, Cambria, "Times New Roman", serif;
  font-size: clamp(2rem, 7vw, 3.25rem);
  line-height: 1.05;
  letter-spacing: -0.03em;
}
h2 {
  margin-bottom: 0;
  font-family: ui-serif, Georgia, Cambria, "Times New Roman", serif;
  font-size: clamp(1.35rem, 4vw, 1.75rem);
  line-height: 1.15;
}
h3 { font-size: 1rem; line-height: 1.25; }
.header-subtitle, .direction-note, .map-status, footer { color: var(--muted); }
.header-subtitle { margin-bottom: 0; max-width: 66ch; }
.header-actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-3); }
.mode-switch {
  display: inline-grid;
  grid-template-columns: 1fr 1fr;
  padding: 3px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-sm);
  background: var(--surface);
}
.mode-switch button {
  border: 0;
  border-radius: 4px;
  background: transparent;
  color: var(--muted);
  padding: var(--space-2) var(--space-3);
  cursor: pointer;
}
.mode-switch button[aria-pressed="true"] {
  background: var(--ink);
  color: white;
}
.read-only-badge, .metric-badge, .state-chip, .day-tag, .direction-pill {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 0.78rem;
  font-weight: 800;
  line-height: 1.2;
}
.read-only-badge { background: var(--blue-soft); color: var(--blue); }
.metric-badge { background: var(--neutral-soft); color: var(--ink); }
.product-grid {
  display: grid;
  gap: var(--space-4);
  max-width: var(--content);
  margin: 0 auto;
  padding: var(--space-4);
}
.panel {
  min-width: 0;
  padding: var(--space-4);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  background: var(--surface);
  box-shadow: var(--shadow);
}
.section-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-3);
  padding-bottom: var(--space-3);
  margin-bottom: var(--space-4);
  border-bottom: 1px solid var(--line);
}
.state-chips { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: var(--space-2); }
.state-chip[data-tone="success"] { background: var(--teal-soft); color: var(--teal); }
.state-chip[data-tone="warning"] { background: var(--amber-soft); color: var(--amber); }
.state-chip[data-tone="danger"] { background: var(--red-soft); color: var(--red); }
.state-chip[data-tone="info"] { background: var(--blue-soft); color: var(--blue); }
.state-chip[data-tone="neutral"] { background: var(--neutral-soft); color: var(--muted); }
.issue-layout { display: grid; gap: var(--space-4); }
.issue-callout {
  padding-left: var(--space-4);
  border-left: 4px solid var(--amber);
}
.issue-callout p:last-child { margin-bottom: 0; }
.fact-list, .repair-facts, .evidence-list, .research-grid {
  display: grid;
  gap: var(--space-3);
  margin: 0;
}
.fact-row, .repair-fact {
  display: grid;
  grid-template-columns: minmax(110px, 0.8fr) minmax(0, 1.4fr);
  gap: var(--space-3);
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--line);
}
.fact-row dt, .repair-fact dt { color: var(--muted); }
.fact-row dd, .repair-fact dd { margin: 0; font-weight: 700; overflow-wrap: anywhere; }
.timeline-list { display: grid; gap: var(--space-2); }
.day-card {
  width: 100%;
  padding: var(--space-3);
  border: 1px solid var(--line);
  border-left: 4px solid transparent;
  border-radius: var(--radius-sm);
  background: var(--surface-strong);
  color: var(--ink);
  text-align: left;
  cursor: pointer;
}
.day-card[aria-current="true"] { border-left-color: var(--teal); background: #f3fbf8; }
.day-card[data-affected="true"] { border-color: var(--amber); }
.day-card-title { display: flex; justify-content: space-between; gap: var(--space-3); font-weight: 850; }
.day-stops { margin: var(--space-2) 0 0; color: var(--muted); font-size: 0.9rem; }
.day-tags { display: flex; flex-wrap: wrap; gap: var(--space-1); margin-top: var(--space-2); }
.day-tag { min-height: 24px; background: var(--neutral-soft); color: var(--muted); }
.day-tag[data-state="changed"], .day-tag[data-state="affected"] {
  background: var(--amber-soft);
  color: var(--amber);
}
.result-banner {
  padding: var(--space-4);
  margin-bottom: var(--space-4);
  border-left: 5px solid var(--teal);
  background: var(--teal-soft);
}
.result-banner[data-state="ineligible_repair"] { border-color: var(--red); background: var(--red-soft); }
.result-banner strong { display: block; font-size: 1.08rem; }
.result-banner p { margin: var(--space-1) 0 0; }
.primary-action {
  width: 100%;
  margin-top: var(--space-4);
  border: 1px solid var(--ink);
  border-radius: var(--radius-sm);
  background: var(--ink);
  color: white;
  cursor: pointer;
}
.primary-action:hover { background: #2b3a30; }
.comparison-region, #comparison-content, .comparison-table-wrap,
.method-landscape, .method-cards, .method-card, .method-card > * {
  min-width: 0;
}
.comparison-table-wrap {
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
  overscroll-behavior-inline: contain;
}
.comparison-table {
  width: 100%;
  min-width: 680px;
  border-collapse: collapse;
}
.comparison-table th, .comparison-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}
.comparison-table th { color: var(--muted); font-size: 0.8rem; letter-spacing: 0.03em; }
.metric-name { font-weight: 800; }
.metric-owner { display: block; color: var(--muted); font-size: 0.75rem; font-weight: 500; }
.metric-unavailable { color: var(--muted); font-style: italic; }
.direction-pill { min-height: 24px; background: var(--neutral-soft); color: var(--muted); }
.method-landscape { margin-top: var(--space-5); }
.method-cards { display: grid; gap: var(--space-2); }
.method-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--space-3);
  padding: var(--space-3);
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface-strong);
}
.method-card[data-status="failed"] { border-left: 4px solid var(--red); }
.method-card[data-status="completed"] { border-left: 4px solid var(--teal); }
.method-card p {
  margin: var(--space-1) 0 0;
  color: var(--muted);
  font-size: 0.88rem;
  overflow-wrap: anywhere;
}
.evidence-card {
  padding: var(--space-3);
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface-strong);
}
.evidence-card + .evidence-card { margin-top: var(--space-2); }
.evidence-card p { margin-bottom: var(--space-2); }
.evidence-card code, .research-value code {
  overflow-wrap: anywhere;
  font-size: 0.78rem;
}
details { border-top: 1px solid var(--line); padding-top: var(--space-3); }
summary { min-height: 44px; cursor: pointer; font-weight: 800; }
#product-map {
  position: relative;
  overflow: hidden;
  min-height: 260px;
  height: 260px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-sm);
  background: #e8ece7;
}
.leaflet-container {
  position: relative;
  overflow: hidden;
  background: #ddd;
  outline-offset: 1px;
  -webkit-tap-highlight-color: transparent;
}
.leaflet-pane,
.leaflet-tile,
.leaflet-marker-icon,
.leaflet-marker-shadow,
.leaflet-tile-container,
.leaflet-pane > svg,
.leaflet-pane > canvas,
.leaflet-zoom-box,
.leaflet-image-layer {
  position: absolute;
  left: 0;
  top: 0;
}
.leaflet-pane { z-index: 400; }
.leaflet-tile-pane { z-index: 200; }
.leaflet-overlay-pane { z-index: 400; }
.leaflet-shadow-pane { z-index: 500; }
.leaflet-marker-pane { z-index: 600; }
.leaflet-tooltip-pane { z-index: 650; }
.leaflet-popup-pane { z-index: 700; }
.leaflet-map-pane canvas { z-index: 100; }
.leaflet-map-pane svg { z-index: 200; }
.leaflet-tile { visibility: hidden; }
.leaflet-tile-loaded { visibility: inherit; }
.leaflet-zoom-animated { transform-origin: 0 0; }
.leaflet-control {
  position: relative;
  z-index: 800;
  pointer-events: auto;
}
.leaflet-top, .leaflet-bottom {
  position: absolute;
  z-index: 1000;
  pointer-events: none;
}
.leaflet-top { top: 0; }
.leaflet-right { right: 0; }
.leaflet-bottom { bottom: 0; }
.leaflet-left { left: 0; }
.leaflet-control { float: left; clear: both; }
.leaflet-right .leaflet-control { float: right; }
.leaflet-top .leaflet-control { margin-top: 10px; }
.leaflet-bottom .leaflet-control { margin-bottom: 10px; }
.leaflet-left .leaflet-control { margin-left: 10px; }
.leaflet-right .leaflet-control { margin-right: 10px; }
.leaflet-bar {
  border: 2px solid rgba(0, 0, 0, 0.2);
  border-radius: 4px;
  background-clip: padding-box;
}
.leaflet-bar a {
  display: block;
  width: 44px;
  height: 44px;
  border-bottom: 1px solid #ccc;
  background: #fff;
  color: #17211b;
  font: bold 20px/44px Arial, Helvetica, sans-serif;
  text-align: center;
  text-decoration: none;
}
.leaflet-bar a:last-child { border-bottom: 0; }
.leaflet-control-attribution {
  margin: 0;
  padding: 0 5px;
  background: rgba(255, 255, 255, 0.82);
  color: #333;
  font-size: 11px;
}
.leaflet-control-attribution a { color: #245f87; }
.leaflet-marker-icon, .leaflet-marker-shadow, .leaflet-pane > svg path {
  pointer-events: auto;
}
.map-status { margin: var(--space-2) 0; font-size: 0.88rem; }
.map-alternative { margin-top: var(--space-3); }
.research-grid { grid-template-columns: 1fr; }
.research-block { min-width: 0; padding: var(--space-3); border: 1px solid var(--line); }
.research-block h3 { margin-bottom: var(--space-2); }
.research-row { display: grid; gap: var(--space-1); padding: var(--space-2) 0; border-bottom: 1px solid var(--line); }
.research-label { color: var(--muted); font-size: 0.8rem; }
.research-value { overflow-wrap: anywhere; font-weight: 700; }
.source-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.source-table th, .source-table td { padding: 8px; border-bottom: 1px solid var(--line); text-align: left; overflow-wrap: anywhere; }
.empty-state, .error-state {
  padding: var(--space-5);
  border: 1px dashed var(--line-strong);
  background: var(--neutral-soft);
}
.error-state { border-color: var(--red); background: var(--red-soft); color: var(--red); }
.sr-status {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  clip-path: inset(50%);
  white-space: nowrap;
}
footer {
  max-width: var(--content);
  margin: 0 auto;
  padding: var(--space-4) var(--space-4) var(--space-6);
  font-size: 0.88rem;
}
[hidden] { display: none !important; }

@media (min-width: 720px) {
  .product-header { flex-direction: row; align-items: center; justify-content: space-between; padding-inline: var(--space-5); }
  .product-grid { padding: var(--space-5); }
  .issue-layout { grid-template-columns: 1.4fr 1fr; }
  .method-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .research-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (min-width: 1080px) {
  .product-grid {
    grid-template-columns: minmax(280px, 0.85fr) minmax(420px, 1.5fr) minmax(320px, 0.95fr);
    grid-template-areas:
      "issue issue issue"
      "timeline map repair"
      "timeline map repair"
      "comparison comparison repair"
      "evidence evidence repair"
      "research research research";
    align-items: start;
  }
  .issue-region { grid-area: issue; }
  .timeline-region { grid-area: timeline; }
  .map-region { grid-area: map; position: sticky; top: var(--space-4); }
  .repair-region { grid-area: repair; position: sticky; top: var(--space-4); }
  .comparison-region { grid-area: comparison; }
  .evidence-region { grid-area: evidence; }
  .research-region { grid-area: research; }
  #product-map { min-height: 540px; height: 540px; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; }
}
"""


def product_dashboard_ui_script() -> str:
    """Return the read-only presentation controller."""

    return r"""(() => {
  'use strict';

  const data = window.PRODUCT_DASHBOARD_DATA;
  const status = document.getElementById('product-status');
  const byId = id => document.getElementById(id);
  const text = value => value === null || value === undefined || value === '' ? 'Unavailable' : String(value);
  const node = (tag, className, content) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (content !== undefined) element.textContent = content;
    return element;
  };
  const append = (parent, ...children) => children.filter(Boolean).forEach(child => parent.appendChild(child));
  const formatNumber = (value, unit = '') => {
    if (value === null || value === undefined) return 'Unavailable';
    if (typeof value !== 'number') return String(value);
    const formatted = Number.isInteger(value) ? String(value) : value.toFixed(Math.abs(value) < 10 ? 2 : 1);
    return unit ? `${formatted} ${unit}` : formatted;
  };

  function fail(message) {
    const main = byId('product-main');
    main.replaceChildren();
    const panel = node('section', 'error-state');
    panel.setAttribute('role', 'alert');
    append(panel, node('h2', '', 'Dashboard artifact could not be rendered'), node('p', '', message));
    main.appendChild(panel);
    status.textContent = `Malformed artifact: ${message}`;
  }

  function setMode(mode) {
    const next = mode === 'research' ? 'research' : 'customer';
    document.body.dataset.productMode = next;
    document.querySelectorAll('[data-mode]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.mode === next));
    });
    document.querySelectorAll('[data-research-only]').forEach(element => {
      element.hidden = next !== 'research';
    });
    status.textContent = next === 'research'
      ? 'Evidence view is active.'
      : 'Trip view is active.';
  }

  function renderStateChips(states) {
    const target = byId('truth-state-chips');
    target.replaceChildren();
    states.filter(state => ['eligible_repair', 'ineligible_repair', 'partial_run', 'certificate_mismatch'].includes(state.id))
      .forEach(state => {
        const chip = node('span', 'state-chip', state.label);
        chip.dataset.tone = state.tone;
        target.appendChild(chip);
      });
  }

  function factList(rows) {
    const list = node('dl', 'fact-list');
    rows.forEach(([label, value]) => {
      const row = node('div', 'fact-row');
      append(row, node('dt', '', label), node('dd', '', text(value)));
      list.appendChild(row);
    });
    return list;
  }

  function renderIssue() {
    const issue = data.issue;
    const repair = data.repair;
    const layout = node('div', 'issue-layout');
    const callout = node('div', 'issue-callout');
    append(
      callout,
      node('h3', '', issue.label),
      node('p', '', issue.summary),
      node('p', '', `${issue.source_status}. ${issue.targets.length ? `Affected commitment: ${issue.targets.join(', ')}.` : 'No target commitment is named.'}`)
    );
    const facts = factList([
      ['Result', repair.status],
      ['Affected days', issue.affected_days.length ? issue.affected_days.join(', ') : 'Unavailable'],
      ['Changed days', repair.changed.affected_day_count],
      ['Unchanged days', repair.unchanged.day_count],
    ]);
    append(layout, callout, facts);
    byId('issue-content').replaceChildren(layout);
  }

  function renderTimeline(selectedDay) {
    const target = byId('timeline-list');
    target.replaceChildren();
    data.timeline.forEach(day => {
      const button = node('button', 'day-card');
      button.type = 'button';
      button.dataset.day = String(day.day);
      button.dataset.affected = String(day.states.includes('affected'));
      button.setAttribute('aria-current', String(day.day === selectedDay));
      button.setAttribute('aria-label', `Day ${day.day}: ${day.child_stop_names.join(', ') || 'no recorded stops'}`);
      const title = node('div', 'day-card-title');
      append(title, node('span', '', `Day ${day.day}`), node('span', '', `${day.stops.length} stop${day.stops.length === 1 ? '' : 's'}`));
      const stopLine = node('p', 'day-stops', day.child_stop_names.join(' · ') || 'No recorded stops');
      const tags = node('div', 'day-tags');
      day.states.forEach(state => {
        const tag = node('span', 'day-tag', state);
        tag.dataset.state = state;
        tags.appendChild(tag);
      });
      append(button, title, stopLine, tags);
      button.addEventListener('click', () => selectDay(day.day));
      target.appendChild(button);
    });
  }

  function selectDay(day) {
    data.trip.selected_day = day;
    document.querySelectorAll('.day-card').forEach(button => {
      button.setAttribute('aria-current', String(Number(button.dataset.day) === day));
    });
    byId('map-day-label').textContent = `Day ${day}`;
    status.textContent = `Day ${day} selected.`;
    window.dispatchEvent(new CustomEvent('product-day-selected', { detail: { day } }));
  }

  function repairFacts(repair) {
    return factList([
      ['What changed', repair.result],
      ['What stayed the same', `${repair.unchanged.day_count} day${repair.unchanged.day_count === 1 ? '' : 's'}`],
      ['Booked changes', repair.permissions.booked_change_count],
      ['Locked changes', repair.permissions.locked_change_count],
      ['Edit cost', formatNumber(repair.tradeoffs.weighted_edit_cost)],
      ['Utility retained', formatNumber(repair.tradeoffs.utility_retained)],
      ['Accepted radius', repair.accepted_radius],
    ]);
  }

  function renderRepair() {
    const repair = data.repair;
    const target = byId('repair-content');
    const banner = node('div', 'result-banner');
    banner.dataset.state = repair.status_state;
    append(banner, node('strong', '', repair.status), node('p', '', repair.result));
    const permission = node('p', '', repair.permissions.message);
    const action = node('button', 'primary-action', repair.primary_action);
    action.type = 'button';
    action.addEventListener('click', () => {
      byId('evidence-region').scrollIntoView({ block: 'start' });
      byId('evidence-title').setAttribute('tabindex', '-1');
      byId('evidence-title').focus();
      status.textContent = 'Evidence section opened.';
    });
    append(target, banner, repairFacts(repair), permission, action);
  }

  function metricCell(side) {
    const cell = node('td');
    if (side.state !== 'available') {
      cell.appendChild(node('span', 'metric-unavailable', side.note || 'Unavailable'));
      return cell;
    }
    append(cell, node('strong', '', formatNumber(side.value)), side.note ? node('small', 'metric-owner', side.note) : null);
    return cell;
  }

  function renderComparison() {
    const wrap = node('div', 'comparison-table-wrap');
    const table = node('table', 'comparison-table');
    const caption = node('caption', 'sr-status', 'Original and repaired plan metric comparison');
    const head = node('thead');
    const headRow = node('tr');
    ['Metric', 'Original', 'Repaired', 'Direction'].forEach(label => headRow.appendChild(node('th', '', label)));
    head.appendChild(headRow);
    const body = node('tbody');
    data.comparison.forEach(metric => {
      const row = node('tr');
      const name = node('td');
      append(name, node('span', 'metric-name', metric.label), node('span', 'metric-owner', `Owner: ${metric.owner}`));
      const direction = node('span', 'direction-pill', metric.direction);
      append(row, name, metricCell(metric.parent), metricCell(metric.child), node('td'));
      row.lastElementChild.appendChild(direction);
      body.appendChild(row);
    });
    append(table, caption, head, body);
    wrap.appendChild(table);
    byId('comparison-content').replaceChildren(wrap);
    renderAlternatives();
  }

  function renderAlternatives() {
    const target = byId('method-landscape');
    target.replaceChildren(node('h3', '', 'Method outcomes for this scenario'));
    if (!data.alternatives.length) {
      target.appendChild(node('p', 'empty-state', 'No comparison rows were declared for this run.'));
      return;
    }
    const cards = node('div', 'method-cards');
    data.alternatives.forEach(method => {
      const card = node('article', 'method-card');
      card.dataset.status = method.status;
      const copy = node('div');
      append(
        copy,
        node('strong', '', method.method_label),
        node('p', '', method.failure_reason || (method.ranking_eligible ? 'Completed and independently eligible.' : method.display_status))
      );
      const chip = node('span', 'state-chip', method.display_status);
      chip.dataset.tone = method.status === 'failed' ? (method.exact_search_incomplete ? 'warning' : 'danger') : method.ranking_eligible ? 'success' : 'neutral';
      append(card, copy, chip);
      cards.appendChild(card);
    });
    target.appendChild(cards);
  }

  function renderEvidence() {
    const target = byId('evidence-content');
    target.replaceChildren();
    const certificate = data.repair.certificate;
    target.appendChild(factList([
      ['Certificate', certificate.id],
      ['Evaluation', certificate.evaluation_status],
      ['Eligibility', certificate.eligible === true ? 'Eligible' : certificate.eligible === false ? 'Ineligible' : 'Unavailable'],
      ['Evaluator failures', certificate.failure_count],
    ]));
    const claimsTitle = node('h3', '', 'Evidence-linked explanation');
    claimsTitle.style.marginTop = '24px';
    target.appendChild(claimsTitle);
    if (!data.evidence.claims.length) {
      target.appendChild(node('p', 'empty-state', 'No grounded explanation was declared.'));
    }
    data.evidence.claims.forEach(claim => {
      const card = node('article', 'evidence-card');
      append(card, node('p', '', claim.text), node('small', 'metric-owner', `${claim.type} · ${claim.confidence} · ${claim.supported ? 'supported' : 'unsupported'}`));
      const details = node('details');
      append(details, node('summary', '', 'Evidence references'));
      claim.evidence_refs.forEach(ref => details.appendChild(node('code', '', ref)));
      card.appendChild(details);
      target.appendChild(card);
    });
  }

  function researchBlock(title, rows) {
    const block = node('section', 'research-block');
    append(block, node('h3', '', title));
    rows.forEach(([label, value]) => {
      const row = node('div', 'research-row');
      append(row, node('span', 'research-label', label), node('span', 'research-value', text(value)));
      block.appendChild(row);
    });
    return block;
  }

  function renderResearch() {
    const research = data.research;
    const target = byId('research-content');
    const grid = node('div', 'research-grid');
    append(
      grid,
      researchBlock('Lineage', Object.entries(research.lineage)),
      researchBlock('Method identity', [
        ['Requested', research.methods.requested.join(', ')],
        ['Executed', research.methods.executed.join(', ')],
        ['Planner attempts', research.methods.planner_runs.length],
      ]),
      researchBlock('Certificate', [
        ['Status', research.certificate.evaluation_status],
        ['Eligibility', research.certificate.comparison_eligibility],
        ['Evaluator version', research.certificate.evaluator_version],
        ['Route matrix', research.certificate.route_validation?.matrix_id],
      ]),
      researchBlock('Diff', [
        ['Diff ID', research.diff?.diff_id],
        ['Weighted edit cost', research.diff?.weighted_edit_cost],
        ['Unchanged days', research.diff?.unchanged_days?.join(', ')],
      ])
    );
    const sourceBlock = node('section', 'research-block');
    sourceBlock.style.gridColumn = '1 / -1';
    sourceBlock.appendChild(node('h3', '', 'Canonical source hashes'));
    const table = node('table', 'source-table');
    const head = node('thead');
    const row = node('tr');
    append(row, node('th', '', 'Run-relative artifact'), node('th', '', 'SHA-256'));
    head.appendChild(row);
    const body = node('tbody');
    Object.entries(research.source_hashes).forEach(([path, hash]) => {
      const tr = node('tr');
      append(tr, node('td', '', path), node('td', '', hash));
      body.appendChild(tr);
    });
    append(table, head, body);
    sourceBlock.appendChild(table);
    grid.appendChild(sourceBlock);
    target.replaceChildren(grid);
  }

  function init() {
    if (!data || data.schema_version !== 'product-dashboard-data-v1') {
      fail('The product data asset is missing or has an unsupported schema.');
      return;
    }
    byId('trip-subtitle').textContent = `${data.trip.day_count}-day review · Run ${data.run.run_id}`;
    byId('day-count').textContent = `${data.trip.day_count} days`;
    byId('map-day-label').textContent = `Day ${data.trip.selected_day}`;
    byId('map-alternative-text').textContent = data.map_alternative;
    renderStateChips(data.truth_states);
    renderIssue();
    renderTimeline(data.trip.selected_day);
    renderRepair();
    renderComparison();
    renderEvidence();
    renderResearch();
    document.querySelectorAll('[data-mode]').forEach(button => {
      button.addEventListener('click', () => setMode(button.dataset.mode));
    });
    setMode('customer');
    status.textContent = 'Product dashboard loaded from validated canonical artifacts.';
    window.productSelectDay = selectDay;
    window.dispatchEvent(new CustomEvent('product-dashboard-ready'));
  }

  try {
    init();
  } catch (error) {
    fail(error instanceof Error ? error.message : 'Unknown rendering error.');
  }
})();
"""


def product_dashboard_map_script() -> str:
    """Return the Leaflet synchronization controller."""

    return r"""(() => {
  'use strict';

  const data = window.PRODUCT_DASHBOARD_DATA;
  const mapElement = document.getElementById('product-map');
  const status = document.getElementById('map-status');
  if (!data || !mapElement || !status) return;
  if (!window.L) {
    status.textContent = 'Interactive map unavailable. Use the text alternative below.';
    return;
  }

  const map = L.map(mapElement, { zoomControl: true, scrollWheelZoom: false });
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);

  const parentLayer = L.layerGroup().addTo(map);
  const childLayer = L.layerGroup().addTo(map);
  const markerLayer = L.layerGroup().addTo(map);
  const routeLines = [];
  const affectedDays = new Set(data.map.affected_days || []);
  const routeStopDays = new Map();
  [data.map.parent, data.map.child].filter(Boolean).forEach(route => {
    route.stops.forEach(stop => routeStopDays.set(stop.id, Number(stop.day)));
  });

  function popupNode(stop, routeLabel) {
    const wrapper = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = stop.name;
    const detail = document.createElement('div');
    detail.textContent = `${routeLabel} · Day ${stop.day} · ${stop.city || 'City unavailable'}`;
    wrapper.append(title, detail);
    return wrapper;
  }

  function routeStyle(kind, day, selectedDay) {
    const affected = affectedDays.has(day);
    const selected = selectedDay === undefined || day === Number(selectedDay);
    const style = kind === 'parent'
      ? { color: affected ? '#854306' : '#59645c', weight: affected ? 5 : 3, opacity: affected ? 0.9 : 0.5, dashArray: '8 7' }
      : { color: affected ? '#854306' : '#0b6f68', weight: affected ? 6 : 4, opacity: affected ? 1 : 0.82 };
    if (!selected) return { ...style, weight: 2, opacity: affected ? 0.28 : 0.16 };
    return { ...style, weight: style.weight + 2, opacity: 1 };
  }

  function drawRoute(route, kind, layer) {
    if (!route) return;
    route.segments.forEach(segment => {
      const day = routeStopDays.get(segment.destination_id) || 0;
      const line = L.polyline(segment.coordinates, routeStyle(kind, day));
      line.productDay = day;
      line.productKind = kind;
      line.bindTooltip(`${kind === 'parent' ? 'Original' : 'Repaired'} route · ${segment.origin_id} to ${segment.destination_id}`);
      line.addTo(layer);
      routeLines.push(line);
    });
    route.stops.forEach(stop => {
      if (!Number.isFinite(Number(stop.latitude)) || !Number.isFinite(Number(stop.longitude))) return;
      const affected = affectedDays.has(Number(stop.day));
      const marker = L.circleMarker([stop.latitude, stop.longitude], {
        radius: affected ? 8 : 6,
        color: affected ? '#854306' : kind === 'parent' ? '#59645c' : '#0b6f68',
        fillColor: '#fffdf7',
        fillOpacity: 1,
        weight: affected ? 4 : 2,
        opacity: kind === 'parent' ? 0.65 : 1,
      });
      marker.productDay = Number(stop.day);
      marker.productKind = kind;
      marker.productStrength = stop.ownership_strength || '';
      marker.bindPopup(popupNode(stop, kind === 'parent' ? 'Original plan' : 'Repaired plan'));
      marker.addTo(markerLayer);
    });
  }

  drawRoute(data.map.parent, 'parent', parentLayer);
  drawRoute(data.map.child, 'child', childLayer);
  L.control.layers(
    {},
    { 'Original route': parentLayer, 'Repaired route': childLayer, 'Stops': markerLayer },
    { collapsed: true, position: 'topright' }
  ).addTo(map);

  const allCoordinates = [];
  [data.map.parent, data.map.child].filter(Boolean).forEach(route => {
    route.stops.forEach(stop => {
      if (Number.isFinite(Number(stop.latitude)) && Number.isFinite(Number(stop.longitude))) {
        allCoordinates.push([stop.latitude, stop.longitude]);
      }
    });
  });
  if (allCoordinates.length) map.fitBounds(allCoordinates, { padding: [24, 24] });
  else map.setView([36.4, -119.7], 5);

  function selectDay(day) {
    const selectedCoordinates = [];
    routeLines.forEach(line => {
      line.setStyle(routeStyle(line.productKind, line.productDay, day));
    });
    markerLayer.eachLayer(marker => {
      const selected = marker.productDay === Number(day);
      const protectedStop = marker.productStrength === 'booked' || marker.productStrength === 'locked';
      marker.setStyle({
        radius: selected ? (protectedStop ? 11 : 9) : (protectedStop ? 7 : 5),
        weight: protectedStop ? 5 : selected ? 3 : 2,
        opacity: selected ? 1 : 0.42,
        fillOpacity: selected ? 1 : 0.55,
      });
      if (selected) selectedCoordinates.push(marker.getLatLng());
    });
    if (selectedCoordinates.length) {
      map.fitBounds(selectedCoordinates, { padding: [36, 36], maxZoom: 11 });
    }
    const evidence = affectedDays.has(Number(day))
      ? ` Affected-day evidence: ${data.map.evidence.label} (${data.map.evidence.source_status}).`
      : '';
    status.textContent = selectedCoordinates.length
      ? `Showing ${selectedCoordinates.length} route stop markers for day ${day}.${evidence}`
      : `No mappable stop coordinates are recorded for day ${day}.`;
  }

  window.addEventListener('product-day-selected', event => selectDay(event.detail.day));
  window.addEventListener('product-dashboard-ready', () => {
    map.invalidateSize();
    selectDay(data.trip.selected_day);
  });
  selectDay(data.trip.selected_day);
  window.productDashboardMap = map;
})();
"""
