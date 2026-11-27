import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {test} from "node:test";
import {createDeploymentClient} from "../../src/itinerary_system/product_app/static/js/deployment-client.js";

const source = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8").replace(/\r\n/g, "\n");
const html = readFileSync(new URL("../../src/itinerary_system/product_app/static/index.html", import.meta.url), "utf8");
const config = {api_origin: "https://api.example.test", supabase_url: "https://project.supabase.co",
  supabase_publishable_key: "sb_publishable_test", static_frontend: true};
const accessToken = [{alg: "HS256"}, {role: "authenticated", sub: "owner", iss: `${config.supabase_url}/auth/v1`, exp: 1900000000}, "signature"]
  .map(value => Buffer.from(typeof value === "string" ? value : JSON.stringify(value)).toString("base64url")).join(".");

function extract(name) {
  const start = source.search(new RegExp(`(?:async )?function ${name}\\(`));
  assert.ok(start >= 0, name);
  const tail = source.slice(start);
  const next = tail.slice(1).search(/\n(?:async )?function \w+\(/);
  return next < 0 ? tail : tail.slice(0, next + 1);
}

function load(name, context) {
  return new Function(...Object.keys(context), `${extract(name)};return ${name};`)(...Object.values(context));
}

test("Sign out is initially hidden, hosted-only, and calls the auth helper before reload", () => {
  assert.match(html, /<button[^>]*id="hosted-sign-out"[^>]*hidden[^>]*>Sign out<\/button>/);
  assert.match(source, /import \{[^}]*productFetch as fetch[^}]*signOutHosted[^}]*\} from "\.\/deployment-client\.js"/);
  const start = source.indexOf("function bindEvents()");
  const end = source.indexOf('  $$("[data-route]")', start);
  assert.ok(start >= 0 && end > start);
  const binding = `${source.slice(start, end)}\n};return bindEvents;`;
  for (const deployment of [undefined, null, config]) {
    const events = [];
    const button = {hidden: true, addEventListener(event, callback) { assert.equal(event, "click"); this.click = callback; }};
    const bind = new Function("window", "$", "signOutHosted", "location", binding)(
      {PRODUCT_DEPLOYMENT: deployment}, selector => { assert.equal(selector, "#hosted-sign-out"); return button; },
      () => events.push("clear hosted auth"), {reload: () => events.push("reload")});
    bind();
    assert.equal(button.hidden, deployment == null);
    if (deployment) { button.click(); assert.deepEqual(events, ["clear hosted auth", "reload"]); }
    else { assert.equal(button.click, undefined); assert.deepEqual(events, []); }
  }
});

for (const hosted of [false, true]) {
  test(`${hosted ? "hosted" : "local"} evidence download fetches the exact selected ZIP and cleans up the blob URL`, async () => {
    const calls = [], notices = [], urls = [], revoked = [], timers = [], downloads = [];
    const state = {clientBuildId: "build-id", token: "workspace-token", session: {draft: ["protected"]}};
    const client = createDeploymentClient({config: hosted ? config : null, location: new URL("https://owner.github.io/repo/"),
      now: () => 1800000000000,
      storage: {getItem: () => JSON.stringify({access_token: accessToken, refresh_token: "refresh"})},
      fetch: async (input, options) => { calls.push({input, options}); return new Response("zip-bytes", {headers: {"Content-Type": "application/zip"}}); }});
    const anchor = {click() { downloads.push({href: this.href, filename: this.download}); }, remove() { this.removed = true; }};
    const download = load("downloadEvidenceBundle", {state, fetch: client.productFetch,
      URL: {createObjectURL(blob) { urls.push(blob); return "blob:download"; }, revokeObjectURL: value => revoked.push(value)},
      document: {createElement(tag) { assert.equal(tag, "a"); return anchor; }, body: {append: node => assert.equal(node, anchor)}},
      setTimeout: (callback, delay) => timers.push({callback, delay}),
      toast: (...args) => notices.push(args), isBuildMismatch: () => false, renderBuildRecovery() { assert.fail(); }});
    const button = {disabled: false};
    const promise = download("run one", "plan&two", button);
    assert.equal(button.disabled, true);
    await download("run one", "plan&two", button);
    await promise;
    assert.equal(calls.length, 1);
    assert.equal(calls[0].input, `${hosted ? config.api_origin : ""}/api/runs/run%20one/evidence-bundle?plan_id=plan%26two`);
    const headers = new Headers(calls[0].options.headers);
    assert.equal(headers.get("x-product-build-id"), "build-id");
    assert.equal(headers.get("x-session-token"), "workspace-token");
    assert.equal(headers.get("authorization"), hosted ? `Bearer ${accessToken}` : null);
    assert.equal(await urls[0].text(), "zip-bytes");
    assert.deepEqual(downloads, [{href: "blob:download", filename: "run_one-plan_two-evidence.zip"}]);
    assert.equal(anchor.removed, true);
    assert.equal(button.disabled, false);
    assert.deepEqual(notices, []);
    assert.equal(timers.length, 1);
    timers[0].callback();
    assert.deepEqual(revoked, ["blob:download"]);
    assert.deepEqual(state.session.draft, ["protected"]);
  });
}

test("failed evidence downloads never produce a file or retry and retain workspace state", async () => {
  for (const failure of [401, 403, 409, 500, "network", "body"]) {
    let requests = 0, recovery = null;
    const notices = [];
    const state = {clientBuildId: "build", token: "workspace", session: {draft: ["protected"]}};
    const download = load("downloadEvidenceBundle", {state, fetch: async () => {
      requests++;
      if (failure === "network") throw new Error("private diagnostic");
      if (failure === "body") return {ok: true, blob() { throw new Error("read failed"); }};
      return new Response(JSON.stringify({detail: "product_client_build_mismatch"}), {status: failure, headers: {"content-type": "application/json"}});
    }, URL: {createObjectURL() { assert.fail("Do not download an error"); }}, document: {},
    toast: (...args) => notices.push(args), isBuildMismatch: code => failure === 409 && code === "product_client_build_mismatch",
    renderBuildRecovery: code => { recovery = code; }});
    const button = {disabled: false};
    await download("run", "plan", button);
    assert.equal(requests, 1);
    assert.equal(button.disabled, false);
    assert.equal(notices.length, 1);
    assert.doesNotMatch(notices[0][0], /private diagnostic|read failed/);
    assert.equal(recovery, failure === 409 ? "product_client_build_mismatch" : null);
    assert.deepEqual(state, {clientBuildId: "build", token: "workspace", session: {draft: ["protected"]}});
  }
});

test("Evidence binds the displayed selected artifact to a button; previews and accepted evidence keep their existing flow", async () => {
  for (const scope of ["selected_option", "preview", "accepted", "unavailable"]) {
    const listeners = new Map(), calls = [];
    const content = {innerHTML: ""};
    const button = {addEventListener: (event, callback) => listeners.set(event, callback)};
    const state = {session: {run_id: "displayed-run", selected_alternative_id: "selected"}, workspace: {}};
    const render = load("renderEvidence", {state, evidencePresentationContext: () => scope === "selected_option" ? null : {scope},
      resolveCompareSelection: () => ({state: scope === "unavailable" ? "unavailable" : "selected", evidenceExact: true,
        option: {plan_id: "displayed-plan"}, plan: {plan_id: "displayed-plan", content_hash: "hash"}, evidence: {}}),
      routeSummaryForPlan: () => "Route evidence", objectDl: () => "", escapeHtml: String, navigate() {},
      $: selector => selector === "#workspace-content" ? content : selector === "#evidence-download"
        ? content.innerHTML.includes('id="evidence-download"') ? button : null : {addEventListener() {}},
      downloadEvidenceBundle: (...args) => calls.push(args)});
    render();
    assert.doesNotMatch(content.innerHTML, /href=["']\/api\//);
    if (scope === "selected_option") {
      assert.match(content.innerHTML, /<button[^>]*id="evidence-download"[^>]*type="button"/);
      state.session.run_id = "new-run";
      await listeners.get("click")({currentTarget: button});
      assert.deepEqual(calls, [["displayed-run", "displayed-plan", button]]);
    } else assert.equal(listeners.size, 0);
  }
});

test("startup failure preserves the accessible sign-in form already mounted by an auth denial", () => {
  const screen = {querySelector: selector => { assert.equal(selector, ".hosted-login"); return {}; },
    set innerHTML(_) { assert.fail("The auth form must remain usable"); }};
  const recover = load("renderStartupFailure", {$: () => screen});
  recover(Object.assign(new Error("Please sign in again"), {status: 401}));
});

const messageSource = source.slice(source.indexOf("const API_ERROR_COPY ="), source.indexOf("function isBuildMismatch("));
const apiErrorMessage = new Function(`${messageSource};return apiErrorMessage;`)();
function stream(fetch, recover = () => {}) {
  return new Function("state", "fetch", "apiErrorMessage", "isBuildMismatch", "renderBuildRecovery",
    `${extract("copilotStreamError")}\n${extract("copilotStreamFrame")}\n${extract("streamCopilotMessage")}\nreturn streamCopilotMessage;`)(
    {clientBuildId: "build", token: "workspace"}, fetch, apiErrorMessage,
    code => code === "product_server_build_changed", recover);
}

test("streamed replies retain event order across chunks and issue only one request", async () => {
  let requests = 0;
  const frames = [
    {type: "message.accepted", sequence: 1}, {type: "assistant.status", sequence: 2},
    {type: "turn.completed", sequence: 3, payload: {reply: "Saved response"}},
  ].map(event => ({schema_version: "copilot-stream-v1", client_message_id: "message-one", request_id: "request-one", ...event}));
  const text = frames.map(event => `data: ${JSON.stringify(event)}\n\n`).join("");
  const events = [];
  const send = stream(async (_path, options) => {
    requests++;
    assert.equal(options.headers["X-Product-Build-Id"], "build");
    assert.equal(options.headers["X-Session-Token"], "workspace");
    const bytes = new TextEncoder().encode(text);
    return new Response(new ReadableStream({start(controller) {
      controller.enqueue(bytes.slice(0, 23)); controller.enqueue(bytes.slice(23)); controller.close();
    }}), {headers: {"Content-Type": "text/event-stream"}});
  });
  assert.deepEqual(await send("/api/chat", {body: {client_message_id: "message-one"}, onEvent: event => events.push(event)}), {reply: "Saved response"});
  assert.deepEqual(events, frames);
  assert.equal(requests, 1);
});

test("stream HTTP auth and restart errors stay visible without automatic replay", async () => {
  for (const [status, code, copy] of [[401, "owner_sign_in_required", /sign in again/],
    [503, "hosted_auth_unavailable", /temporarily unavailable/],
    [503, "hosted_state_unavailable", /^Your workspace is temporarily unavailable\. Saved changes are protected; please try again after the service restarts\.$/],
    [409, "product_server_build_changed", /product_server_build_changed/]]) {
    let requests = 0, recovered = null;
    const send = stream(async () => { requests++; return new Response(JSON.stringify({detail: code}), {status, headers: {"Content-Type": "application/json"}}); }, value => { recovered = value; });
    await assert.rejects(send("/api/chat", {body: {client_message_id: "message-one"}}), error => {
      assert.equal(error.status, status);
      assert.equal(error.code, code);
      assert.match(error.message, copy);
      if (status !== 503) assert.equal(error.retryable, false);
      return true;
    });
    assert.equal(requests, 1);
    assert.equal(recovered, status === 409 ? code : null);
  }
});

test("an interrupted streamed reply asks to check saved status and does not resend", async () => {
  let requests = 0;
  const send = stream(async () => { requests++; return new Response("", {headers: {"Content-Type": "text/event-stream"}}); });
  await assert.rejects(send("/api/chat", {body: {client_message_id: "message-one"}}), error => {
    assert.equal(error.code, "copilot_stream_incomplete");
    assert.match(error.message, /Check its saved status/);
    return true;
  });
  assert.equal(requests, 1);
});
