import assert from "node:assert/strict";
import {test} from "node:test";
import {
  createDeploymentClient, productFetch, currentAppPath, appRouteUrl, ensureHostedLogin, signOutHosted,
} from "../../src/itinerary_system/product_app/static/js/deployment-client.js";

const CONFIG = Object.freeze({
  api_origin: "https://api.example.test",
  supabase_url: "https://project.supabase.co",
  supabase_publishable_key: "sb_publishable_test_public",
  static_frontend: true,
});
const PAGE = "https://owner.github.io/repo/?keep=1#main-workspace";
const TIME = 1800000000000;
const jwt = payload => [
  {alg: "HS256", typ: "JWT"}, payload, "test-signature",
].map(value => Buffer.from(typeof value === "string" ? value : JSON.stringify(value)).toString("base64url")).join(".");
const authSession = (overrides = {}, authOrigin = CONFIG.supabase_url) => ({
  access_token: jwt({role: "authenticated", sub: "owner-uid", iss: `${authOrigin}/auth/v1`, exp: TIME / 1000 + 3600}),
  refresh_token: "refresh-one", token_type: "bearer", expires_in: 3600, ...overrides,
});
const json = (body, status = 200) => new Response(JSON.stringify(body), {status});

function memory() {
  const data = new Map();
  const removed = [];
  return {
    data, removed,
    getItem: key => data.get(key) ?? null,
    setItem: (key, value) => data.set(key, value),
    removeItem(key) { removed.push(key); data.delete(key); },
    clear() { assert.fail("Must never clear workspace storage"); },
  };
}

// Only the DOM operations used by the login are provided; HTML/style injection is forbidden.
function loginDocument() {
  const document = {focused: null};
  class Element {
    constructor(tagName) {
      this.tagName = tagName;
      this.children = [];
      this.attributes = new Map();
      this.listeners = new Map();
      this.value = "";
      this.hidden = false;
      this.disabled = false;
      this.text = "";
    }
    set innerHTML(_) { assert.fail("Login must use safe DOM text"); }
    set textContent(value) { this.text = value; this.children = []; }
    get textContent() { return this.text + this.children.map(child => child.textContent).join(" "); }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.text = ""; this.children = children; }
    setAttribute(name, value) {
      assert.ok(name !== "style" && !name.startsWith("on"));
      this.attributes.set(name, value);
    }
    getAttribute(name) { return this.attributes.get(name) ?? null; }
    removeAttribute(name) { this.attributes.delete(name); }
    addEventListener(name, listener) { this.listeners.set(name, listener); }
    focus() { document.focused = this; }
    reportValidity() { return descendants(this).filter(el => el.required).every(el => el.value); }
    submit() { return this.listeners.get("submit")({preventDefault() {}}); }
  }
  const descendants = node => [node, ...node.children.flatMap(descendants)];
  document.createElement = tag => new Element(tag);
  const screen = document.createElement("div");
  screen.id = "loading-screen";
  screen.setAttribute("role", "status");
  const localCopy = document.createElement("p");
  localCopy.textContent = "Validating local product readiness…";
  const helper = document.createElement("div");
  helper.className = "direct-file-help";
  helper.textContent = "OPEN_ITINERARY_COPILOT.cmd";
  screen.append(localCopy, helper);
  const shell = document.createElement("div");
  shell.id = "app-shell";
  document.getElementById = id => [shell, ...descendants(screen)].find(el => el.id === id) ?? null;
  return {document, screen, shell, descendants};
}

function harness({config = CONFIG, href = PAGE, storage = memory(), respond, now = () => TIME, dom = loginDocument()} = {}) {
  const location = new URL(href);
  const calls = [];
  const fetch = (input, options) => {
    calls.push({input, options});
    if (respond) return respond(input, options, calls.length);
    return Promise.resolve(String(input).includes("/auth/v1/token?") ? json(authSession()) : json({ok: true}));
  };
  const client = createDeploymentClient({config, location, fetch, storage, document: dom.document, now});
  return {client, config, location, storage, calls, ...dom};
}

function fillLogin(h, {email = "owner@example.test", password = "test-password"} = {}) {
  h.document.getElementById("hosted-sign-in-email").value = email;
  h.document.getElementById("hosted-sign-in-password").value = password;
  return h.screen.children[0];
}

async function signIn(h) {
  const ready = h.client.ensureHostedLogin();
  await fillLogin(h).submit();
  await ready;
}

const savedKey = h => [...h.storage.data.keys()].find(key => key.startsWith("product:hosted-auth:v1:"));

test("absent/null config preserves native local fetch arguments, result identity and pathname routing", async () => {
  const input = {opaque: true};
  const options = {method: "POST", headers: {"X-Session-Token": "legacy"}};
  const response = Promise.resolve({local: true});
  for (const config of [null, undefined]) {
    const client = createDeploymentClient({config, location: {pathname: "/app/map"}, fetch: (actual, init) => {
      assert.equal(actual, input);
      assert.equal(init, options);
      return response;
    }});
    assert.equal(client.productFetch(input, options), response);
    assert.equal(client.currentAppPath(), "/app/map");
    assert.equal(client.appRouteUrl("/app/repairs"), "/app/repairs");
    assert.equal(await client.ensureHostedLogin(), undefined);
    assert.equal(client.signOutHosted(), undefined);
  }
});

test("default exports read window config lazily and local fetch needs no document or storage", async t => {
  const previous = globalThis.window;
  t.after(() => { if (previous === undefined) delete globalThis.window; else globalThis.window = previous; });
  const result = Promise.resolve("native");
  globalThis.window = {
    location: {pathname: "/app/evidence"},
    fetch(input, options) { assert.equal(this, window); assert.equal(input, "/api/state"); assert.equal(options, undefined); return result; },
    get sessionStorage() { assert.fail("Local fetch must not read storage"); },
  };
  assert.equal(productFetch("/api/state"), result);
  assert.equal(signOutHosted(), undefined);
  assert.equal(currentAppPath(), "/app/evidence");
  assert.equal(appRouteUrl("/app/map"), "/app/map");
  await ensureHostedLogin();
  const hosted = harness();
  globalThis.window = {
    PRODUCT_DEPLOYMENT: CONFIG, location: hosted.location, sessionStorage: hosted.storage,
    document: hosted.document, fetch: () => Promise.resolve(json(authSession({access_token: jwt({
      role: "authenticated", sub: "owner-uid", iss: `${CONFIG.supabase_url}/auth/v1`, exp: Math.floor(Date.now() / 1000) + 3600,
    })}))),
  };
  const login = ensureHostedLogin();
  await fillLogin(hosted).submit();
  await login;
  assert.equal(currentAppPath(), "/app/itinerary");
  assert.equal(appRouteUrl("/app/map"), "/repo/?keep=1&view=map#main-workspace");
  const key = savedKey(hosted);
  assert.ok(key);
  signOutHosted();
  assert.equal(hosted.storage.getItem(key), null);
});

test("rejects invalid config, noncanonical origins, remote HTTP, lookalike auth hosts and secret keys", () => {
  const invalid = [
    false, [], {}, {...CONFIG, static_frontend: false},
    ...[
      "https://api.example.test/", "https://api.example.test/path", "https://api.example.test?x=1",
      "https://api.example.test#", "https://user:password@api.example.test", "https://API.example.test",
      "https://api.example.test:443", "https://api.example.test/%2e%2e", "http://api.example.test",
      "http://127.0.0.1:8127", "https://localhost:8127", "//api.example.test", "javascript:alert(1)",
    ].map(api_origin => ({...CONFIG, api_origin})),
    ...[
      "https://supabase.co", "https://project.supabase.co.evil.test", "https://nested.project.supabase.co",
      "https://project.supabase.co:8443", "http://project.supabase.co", "https://project.supabase.co/",
    ].map(supabase_url => ({...CONFIG, supabase_url})),
    ...["sb_secret_hidden", "service_role", "sb_publishable_service_role", "not-a-key", jwt({role: "service_role"})]
      .map(supabase_publishable_key => ({...CONFIG, supabase_publishable_key})),
    {...CONFIG, SUPABASE_SERVICE_ROLE_KEY: "private"}, {...CONFIG, OPENAI_API_KEY: "private"},
  ];
  for (const config of invalid) assert.throws(() => harness({config}), /configuration is invalid/);
  assert.doesNotThrow(() => harness({config: {...CONFIG, supabase_publishable_key: jwt({role: "anon"})}}));
  assert.throws(() => harness({href: "file:///repo/index.html"}), /configuration is invalid/);
  assert.throws(() => harness({href: "http://owner.example.test/repo/"}), /configuration is invalid/);
});

test("HTTP exceptions require the current page and each HTTP backend to be loopback", () => {
  for (const host of ["localhost", "127.0.0.1", "127.2.3.4", "[::1]"]) {
    assert.doesNotThrow(() => harness({href: `http://${host}:8000/repo/`, config: {
      ...CONFIG, api_origin: `http://${host}:8127`, supabase_url: `http://${host}:54321`,
    }}));
  }
  assert.throws(() => harness({href: "https://localhost/repo/", config: {...CONFIG, api_origin: "http://localhost:8127"}}), /invalid/);
  assert.throws(() => harness({href: "http://localhost:8000/", config: {...CONFIG, api_origin: "http://remote.example.test"}}), /invalid/);
  assert.throws(() => harness({href: "https://owner.github.io/", config: {...CONFIG, supabase_url: "http://localhost:54321"}}), /invalid/);
});

test("Pages query routing retains project/index path, other parameters and skip-link hash", () => {
  for (const path of ["/repo/", "/repo/index.html"]) {
    const h = harness({href: `https://owner.github.io${path}?keep=1&view=map#main-workspace`});
    assert.equal(h.client.currentAppPath(), "/app/map");
    for (const view of ["map", "repairs", "compare", "evidence", "settings"]) {
      const target = h.client.appRouteUrl(`/app/${view}`);
      assert.equal(target, `${path}?keep=1&view=${view}#main-workspace`);
      h.location.href = new URL(target, h.location).href;
      assert.equal(h.client.currentAppPath(), `/app/${view}`);
      assert.equal(harness({href: h.location.href}).client.currentAppPath(), `/app/${view}`);
    }
    for (const route of ["/app", "/app/", "/app/itinerary", "/unknown", "//evil.test", "https://evil.test/app/map", "/app/../map", "/app/map?view=evidence"]) {
      assert.equal(h.client.appRouteUrl(route), `${path}?keep=1#main-workspace`);
    }
    h.location.href = `https://owner.github.io${path}?view=unknown#main-workspace`;
    assert.equal(h.client.currentAppPath(), "/app/itinerary");
  }
});

test("query config overrides never change the configured API or auth origins", async () => {
  const h = harness({href: `${PAGE.split("#")[0]}&api_origin=https://evil.test&supabase_url=https://evil.test#main-workspace`});
  await signIn(h);
  await h.client.productFetch("/api/state");
  assert.equal(h.calls[0].input, `${CONFIG.supabase_url}/auth/v1/token?grant_type=password`);
  assert.equal(h.calls[1].input, `${CONFIG.api_origin}/api/state`);
});

test("login renders a labeled accessible form using existing classes and keeps passwords out of storage", async () => {
  const h = harness();
  const ready = h.client.ensureHostedLogin();
  const form = fillLogin(h);
  assert.equal(h.screen.children.length, 1);
  assert.equal(form.tagName, "form");
  assert.match(form.className, /recovery-card dialog-card/);
  assert.ok(form.className.split(" ").includes("hosted-login"));
  assert.equal(form.getAttribute("aria-labelledby"), "hosted-sign-in-title");
  assert.equal(h.screen.getAttribute("role"), null);
  assert.equal(h.shell.hidden, true);
  assert.doesNotMatch(h.screen.textContent, /local|launcher|Supabase|Railway|API|OPEN_ITINERARY/i);
  const nodes = h.descendants(form);
  const inputs = nodes.filter(node => node.tagName === "input");
  assert.deepEqual(inputs.map(node => [node.type, node.autocomplete, node.required]), [["email", "username", true], ["password", "current-password", true]]);
  for (const input of inputs) assert.ok(nodes.some(node => node.tagName === "label" && node.htmlFor === input.id && node.textContent));
  assert.equal(nodes.find(node => node.tagName === "button").textContent, "Sign in");
  await form.submit();
  await ready;
  const {input, options} = h.calls[0];
  assert.equal(input, `${CONFIG.supabase_url}/auth/v1/token?grant_type=password`);
  assert.equal(options.headers.apikey, CONFIG.supabase_publishable_key);
  assert.equal(options.credentials, "omit");
  assert.equal(options.redirect, "error");
  assert.deepEqual(JSON.parse(options.body), {email: "owner@example.test", password: "test-password"});
  assert.equal(inputs[1].value, "");
  assert.deepEqual(Object.keys(JSON.parse(h.storage.data.get(savedKey(h)))).sort(), ["access_token", "refresh_token"]);
  assert.doesNotMatch([...h.storage.data.values()].join(""), /test-password|owner@example.test/);
  assert.equal(h.screen.textContent, "Opening your workspace…");
});

test("empty submission does not call auth; concurrent login calls/submissions share one sign-in", async () => {
  let release;
  const h = harness({respond: () => new Promise(resolve => { release = resolve; })});
  const first = h.client.ensureHostedLogin();
  const second = h.client.ensureHostedLogin();
  const form = h.screen.children[0];
  await form.submit();
  assert.equal(h.calls.length, 0);
  fillLogin(h);
  const submission = form.submit();
  await form.submit();
  assert.equal(h.calls.length, 1);
  assert.equal(form.getAttribute("aria-busy"), "true");
  release(json(authSession()));
  await Promise.all([first, second, submission]);
});

test("API qualification preserves caller headers, body, method, signal and options without mutating them", async () => {
  const h = harness();
  await signIn(h);
  for (const original of [
    {"X-Product-Build-Id": "build", "X-Session-Token": "workspace", Authorization: "old"},
    new Headers({"X-Product-Build-Id": "build", "X-Session-Token": "workspace", Authorization: "old"}),
    [["X-Product-Build-Id", "build"], ["X-Session-Token", "workspace"], ["Authorization", "old"]],
  ]) {
    const signal = new AbortController().signal;
    const options = {method: "POST", body: '{"action":"once"}', headers: original, signal, keepalive: true, redirect: "follow", credentials: "include"};
    const snapshot = [...new Headers(original)];
    await h.client.productFetch("/api/workspaces/edit?x=1", options);
    const call = h.calls.at(-1);
    assert.equal(call.input, `${CONFIG.api_origin}/api/workspaces/edit?x=1`);
    assert.equal(call.options.headers.get("authorization"), `Bearer ${authSession().access_token}`);
    assert.equal(call.options.headers.get("x-product-build-id"), "build");
    assert.equal(call.options.headers.get("x-session-token"), "workspace");
    assert.equal(call.options.body, options.body);
    assert.equal(call.options.method, "POST");
    assert.equal(call.options.signal, signal);
    assert.equal(call.options.keepalive, true);
    assert.equal(call.options.redirect, "error");
    assert.equal(call.options.credentials, "omit");
    assert.deepEqual([...new Headers(original)], snapshot);
    assert.equal(options.redirect, "follow");
  }
  for (const input of [new URL("/api/state", PAGE), `${CONFIG.api_origin}/api/state`]) {
    await h.client.productFetch(input);
    assert.equal(h.calls.at(-1).input, `${CONFIG.api_origin}/api/state`);
  }
});

test("Request input retains streamed body, headers, method and abort signal when moved to API origin", async () => {
  const h = harness();
  await signIn(h);
  const abort = new AbortController();
  const request = new Request(new URL("/api/edit", PAGE), {method: "PATCH", body: "one-edit", headers: {"X-Session-Token": "workspace", "X-Product-Build-Id": "build"}, signal: abort.signal});
  await h.client.productFetch(request);
  const call = h.calls.at(-1);
  const effective = new Request(call.input, call.options);
  assert.equal(effective.url, `${CONFIG.api_origin}/api/edit`);
  assert.equal(effective.method, "PATCH");
  assert.equal(await effective.text(), "one-edit");
  assert.equal(effective.headers.get("x-session-token"), "workspace");
  assert.equal(effective.headers.get("x-product-build-id"), "build");
  assert.match(effective.headers.get("authorization"), /^Bearer /);
  abort.abort();
  assert.equal(effective.signal.aborted, true);
});

test("same-origin non-API requests pass through unchanged; cross-origin and traversal targets fail before auth/network", async () => {
  const h = harness();
  const options = {headers: {Accept: "text/css"}};
  await h.client.productFetch("./static/app.css", options);
  assert.equal(h.calls[0].input, "./static/app.css");
  assert.equal(h.calls[0].options, options);
  for (const target of [
    "https://evil.test/api/state", "https://evil.test/asset", `${CONFIG.api_origin}/non-api`,
    "//evil.test/api/state", "//owner.github.io/api/state", "/api/../non-api", "/api/./state",
    "/api/%2e%2e/asset", "/api/%252e%252e/asset", "/api/a%2fb", "/api/a%5cb", "/api/a\\b",
    "/api//state", "/api/state#fragment", " https://evil.test", "\n/api/state",
    "https://username@api.example.test/api/state", "data:text/plain,example", "javascript:alert(1)",
  ]) await assert.rejects(h.client.productFetch(target), /not allowed/);
  assert.equal(h.calls.length, 1);
  assert.equal(h.screen.children[0].tagName, "p");
});

test("signed-out API denial sends nothing and preserves old workspace credentials, drafts and other namespaces", async () => {
  const h = harness();
  h.storage.setItem("product:session", "legacy-credential");
  h.storage.setItem("pending:workspace", "protected-draft");
  h.storage.setItem("product:hosted-auth:v1:another-config", "another-owner");
  const before = [...h.storage.data];
  await assert.rejects(h.client.productFetch("/api/paid-request", {method: "POST"}), /sign in again/);
  assert.equal(h.calls.length, 0);
  assert.deepEqual([...h.storage.data], before);
  assert.match(h.document.getElementById("hosted-sign-in-error").textContent, /sign in again/);
});

test("auth failures clear only this auth namespace, show generic text, and allow explicit sign-in again", async () => {
  for (const failure of [
    () => json({msg: "<img src=x onerror=alert(1)> secret infrastructure"}, 400),
    () => { throw new Error("private provider error"); },
    () => new Response("not-json"),
    () => json(authSession({access_token: jwt({role: "service_role"})})),
    () => json(authSession({access_token: jwt({role: "authenticated", sub: "owner", iss: `${CONFIG.supabase_url}/auth/v1`, exp: TIME / 1000 - 1})})),
    () => json(authSession({refresh_token: ""})),
  ]) {
    let fail = true;
    const h = harness({respond: () => fail ? failure() : json(authSession())});
    h.storage.setItem("draft", "preserved");
    const ready = h.client.ensureHostedLogin();
    const form = fillLogin(h);
    await form.submit();
    const error = h.document.getElementById("hosted-sign-in-error");
    assert.equal(error.hidden, false);
    assert.match(error.textContent, /couldn't sign you in/);
    assert.doesNotMatch(h.screen.textContent, /img|infrastructure|provider/);
    assert.equal(h.document.getElementById("hosted-sign-in-password").value, "");
    assert.equal(h.storage.getItem("draft"), "preserved");
    assert.equal(savedKey(h), undefined);
    assert.equal(h.calls.length, 1);
    fail = false;
    fillLogin(h);
    await form.submit();
    await ready;
    assert.equal(h.calls.length, 2);
  }
});

test("tokens restore only for the same config and frontend project; malformed stored auth is removed narrowly", async () => {
  const h = harness();
  await signIn(h);
  const storage = h.storage;
  storage.setItem("draft", "keep");
  const restored = harness({storage});
  await restored.client.ensureHostedLogin();
  await restored.client.productFetch("/api/state");
  assert.equal(restored.calls.length, 1);
  assert.equal(restored.calls[0].input, `${CONFIG.api_origin}/api/state`);
  for (const variation of [
    {config: {...CONFIG, api_origin: "https://other.example.test"}},
    {config: {...CONFIG, supabase_url: "https://other.supabase.co"}},
    {config: {...CONFIG, supabase_publishable_key: "sb_publishable_another"}},
    {href: "https://owner.github.io/other-project/"},
  ]) {
    const isolated = harness({storage, ...variation});
    await assert.rejects(isolated.client.productFetch("/api/state"), /sign in again/);
    assert.equal(isolated.calls.length, 0);
    assert.ok(storage.getItem(savedKey(h)));
  }
  const key = savedKey(h);
  for (const invalid of ["not-json", "null", '{"access_token":"invalid","refresh_token":"refresh"}']) {
    storage.setItem(key, invalid);
    const corrupted = harness({storage});
    await assert.rejects(corrupted.client.productFetch("/api/state"), /sign in again/);
    assert.equal(storage.getItem(key), null);
    assert.equal(storage.getItem("draft"), "keep");
  }
});

test("expired tokens refresh once before concurrent requests; replacement tokens are stored without other response data", async () => {
  let time = TIME;
  let refresh;
  const initial = authSession();
  const next = authSession({access_token: jwt({role: "authenticated", sub: "owner-uid", iss: `${CONFIG.supabase_url}/auth/v1`, exp: TIME / 1000 + 7200}), refresh_token: "refresh-two", user: {email: "private@example.test"}});
  const h = harness({now: () => time, respond: (input, options) => {
    if (String(input).includes("grant_type=password")) return json(initial);
    if (String(input).includes("grant_type=refresh_token")) {
      assert.deepEqual(JSON.parse(options.body), {refresh_token: "refresh-one"});
      return new Promise(resolve => { refresh = resolve; });
    }
    return json({ok: true});
  }});
  await signIn(h);
  time += 3600 * 1000;
  const first = h.client.productFetch("/api/edit", {method: "POST", body: "once"});
  const second = h.client.productFetch("/api/state");
  assert.equal(h.calls.length, 2);
  refresh(json(next));
  await Promise.all([first, second]);
  assert.equal(h.calls.length, 4);
  for (const call of h.calls.slice(2)) assert.equal(call.options.headers.get("authorization"), `Bearer ${next.access_token}`);
  assert.deepEqual(JSON.parse(h.storage.getItem(savedKey(h))), {access_token: next.access_token, refresh_token: "refresh-two"});
});

test("refresh rejection clears auth, blocks the queued mutation, and offers sign-in without replay", async () => {
  let time = TIME;
  const h = harness({now: () => time, respond: input => String(input).includes("grant_type=password") ? json(authSession()) : json({error: "refresh expired"}, 400)});
  await signIn(h);
  const key = savedKey(h);
  h.storage.setItem("pending", "uncertain-edit");
  time += 3600 * 1000;
  await assert.rejects(h.client.productFetch("/api/paid-request", {method: "POST"}), /sign in again/);
  assert.equal(h.calls.length, 2);
  assert.equal(h.storage.getItem(key), null);
  assert.equal(h.storage.getItem("pending"), "uncertain-edit");
  assert.match(h.document.getElementById("hosted-sign-in-error").textContent, /sign in again/);
  await assert.rejects(h.client.productFetch("/api/paid-request", {method: "POST"}), /sign in again/);
  assert.equal(h.calls.length, 2);
});

test("API 401/403, server errors and network failures never retry/replay a mutation or paid request", async () => {
  for (const outcome of [401, 403, 500, "network"]) {
    const h = harness({respond: input => {
      if (String(input).includes("/auth/v1/")) return json(authSession());
      if (outcome === "network") throw new Error("connection lost");
      return json({detail: "denied"}, outcome);
    }});
    await signIn(h);
    const key = savedKey(h);
    h.storage.setItem("pending", "protected");
    if (outcome === "network") await assert.rejects(h.client.productFetch("/api/paid-request", {method: "POST"}), /connection lost/);
    else assert.equal((await h.client.productFetch("/api/paid-request", {method: "POST"})).status, outcome);
    assert.equal(h.calls.length, 2);
    assert.equal(h.storage.getItem("pending"), "protected");
    if ([401, 403].includes(outcome)) {
      assert.equal(h.storage.getItem(key), null);
      assert.equal(h.shell.hidden, true);
      assert.match(h.document.getElementById("hosted-sign-in-error").textContent, /sign in again/);
    } else assert.ok(h.storage.getItem(key));
  }
});

test("blocked sessionStorage keeps auth in memory and does not prevent sign-in or touch workspace storage", async () => {
  const storage = {getItem() { throw new Error("denied"); }, setItem() { throw new Error("denied"); }, removeItem() { throw new Error("denied"); }};
  const h = harness({storage});
  await signIn(h);
  await h.client.productFetch("/api/state");
  assert.equal(h.calls.length, 2);
  assert.match(h.calls[1].options.headers.get("authorization"), /^Bearer /);
});

test("hosted sign-out clears only owner auth and preserves workspace credentials, drafts, journals and preferences", async () => {
  const h = harness();
  await signIn(h);
  const key = savedKey(h);
  const unrelated = [
    ["product:session", '{"session_id":"workspace","mutation_token":"protected"}'],
    ["draft:workspace", '{"edits":["unsaved"]}'],
    ["pending:workspace", '{"mutation_id":"uncertain","body":"original"}'],
    ["input:workspace", "Unsent message"],
    ["interest:trip", '{"enabled":false,"radiusKm":2}'],
    ["product:hosted-auth:v1:another-config", "another-project-auth"],
  ];
  for (const [name, value] of unrelated) h.storage.setItem(name, value);
  const removedBefore = h.storage.removed.length;
  assert.equal(h.client.signOutHosted(), undefined);
  assert.equal(h.calls.length, 1, "sign-out must not call a remote endpoint");
  assert.equal(h.storage.getItem(key), null);
  assert.deepEqual([...h.storage.data], unrelated);
  assert.deepEqual(h.storage.removed.slice(removedBefore), [key]);
  await assert.rejects(h.client.productFetch("/api/edit", {method: "POST"}), /sign in again/);
  assert.equal(h.calls.length, 1, "the in-memory access token must also be removed");
  const reloaded = harness({storage: h.storage});
  await assert.rejects(reloaded.client.productFetch("/api/state"), /sign in again/);
  assert.equal(reloaded.calls.length, 0);
  assert.deepEqual([...h.storage.data], unrelated);
});

test("sign-out prevents a late refresh from restoring auth or dispatching the queued mutation", async () => {
  let time = TIME;
  let release;
  const h = harness({now: () => time, respond: input => {
    if (String(input).includes("grant_type=password")) return json(authSession());
    return new Promise(resolve => { release = resolve; });
  }});
  await signIn(h);
  const key = savedKey(h);
  h.storage.setItem("pending:workspace", "protected");
  time += 3600 * 1000;
  const pending = h.client.productFetch("/api/edit", {method: "POST", body: "only-once"});
  h.client.signOutHosted();
  release(json(authSession({access_token: jwt({
    role: "authenticated", sub: "owner-uid", iss: `${CONFIG.supabase_url}/auth/v1`, exp: TIME / 1000 + 7200,
  }), refresh_token: "late-refresh"})));
  await assert.rejects(pending, /sign in again/);
  assert.equal(h.storage.getItem(key), null);
  assert.equal(h.storage.getItem("pending:workspace"), "protected");
  assert.equal(h.calls.length, 2, "only sign-in and the already-started refresh may run");
  await assert.rejects(h.client.productFetch("/api/state"), /sign in again/);
  assert.equal(h.calls.length, 2);
});

test("sign-out also blocks an API call queued with a still-valid in-memory token", async () => {
  const h = harness();
  await signIn(h);
  const pending = h.client.productFetch("/api/edit", {method: "POST"});
  h.client.signOutHosted();
  await assert.rejects(pending, /sign in again/);
  assert.equal(h.calls.length, 1);
});

test("explicit sign-in after an API auth denial reloads the workspace without replaying the denied action", async () => {
  let reloads = 0;
  const h = harness({respond: input => String(input).includes("/auth/v1/token?") ? json(authSession()) : json({detail: "owner_sign_in_required"}, 401)});
  h.location.reload = () => { reloads++; };
  await signIn(h);
  assert.equal(reloads, 0);
  await h.client.productFetch("/api/paid-request", {method: "POST"});
  assert.equal(reloads, 0);
  await fillLogin(h).submit();
  assert.equal(reloads, 1);
  assert.equal(h.calls.length, 3, "only the initial sign-in, denied action, and explicit second sign-in run");
  assert.ok(h.storage.getItem(savedKey(h)));
});
