(() => {
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
