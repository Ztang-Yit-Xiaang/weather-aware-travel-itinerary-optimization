const VIEWS = new Set(["itinerary", "map", "repairs", "compare", "evidence", "settings"]);
const SIGN_IN_ERROR = "We couldn't sign you in. Please check your details and try again.";
const SESSION_ERROR = "Please sign in again to open your private workspace.";

function claims(token) {
  try {
    if (typeof token !== "string" || !/^[\w-]+\.[\w-]+\.[\w-]+$/.test(token)) return null;
    const payload = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(atob(payload.padEnd(Math.ceil(payload.length / 4) * 4, "=")));
  } catch {
    return null;
  }
}

function loopback(url) {
  return url.hostname === "localhost" || url.hostname === "[::1]" || /^127\.\d+\.\d+\.\d+$/.test(url.hostname);
}

function deploymentConfig(config, page) {
  const invalid = () => { throw new TypeError("This workspace's sign-in configuration is invalid."); };
  const fields = ["api_origin", "supabase_url", "supabase_publishable_key", "static_frontend"];
  if (!config || typeof config !== "object" || Array.isArray(config)
      || Object.keys(config).some(key => !fields.includes(key)) || config.static_frontend !== true) invalid();
  if (page.protocol !== "https:" && !(page.protocol === "http:" && loopback(page))) invalid();
  const origin = (value, auth = false) => {
    let url;
    try { url = new URL(value); } catch { invalid(); }
    if (typeof value !== "string" || value !== url.origin) invalid();
    const local = loopback(page) && loopback(url);
    if (loopback(url) && !local) invalid();
    if (url.protocol !== "https:" && !(local && page.protocol === "http:" && url.protocol === "http:")) invalid();
    if (auth && !local && (url.port || !/^[a-z0-9-]+\.supabase\.co$/.test(url.hostname))) invalid();
    return url.origin;
  };
  const apiOrigin = origin(config.api_origin);
  const authOrigin = origin(config.supabase_url, true);
  const key = config.supabase_publishable_key;
  if (typeof key !== "string" || /sb_secret|service_role/i.test(key)
      || (!/^sb_publishable_[\w-]+$/.test(key) && claims(key)?.role !== "anon")) invalid();
  return {apiOrigin, authOrigin, key};
}

/** Dependencies are injected so hosted behavior can be tested without a browser or network. */
export function createDeploymentClient({config = null, location, fetch, storage, document, now = Date.now} = {}) {
  if (config == null) {
    return {
      productFetch: (input, options) => fetch(input, options),
      currentAppPath: () => location.pathname,
      appRouteUrl: route => route,
      ensureHostedLogin: async () => {},
      signOutHosted: () => {},
    };
  }

  const page = new URL(location.href);
  const {apiOrigin, authOrigin, key} = deploymentConfig(config, page);
  const storageKey = `product:hosted-auth:v1:${JSON.stringify([page.origin, page.pathname, apiOrigin, authOrigin, key])}`;
  let session = null;
  let authEpoch = 0;
  let refreshPromise = null;
  let loginPromise = null;

  function clearSession() {
    authEpoch += 1;
    session = null;
    try { storage?.removeItem(storageKey); } catch { /* Storage may be disabled. */ }
  }

  function readSession(value) {
    const token = claims(value?.access_token);
    // This checks shape and expiry only. The API must verify the signature and owner UID.
    if (!token || token.role !== "authenticated" || typeof token.sub !== "string" || !token.sub
        || token.iss !== `${authOrigin}/auth/v1` || !Number.isFinite(token.exp)
        || typeof value.refresh_token !== "string" || !value.refresh_token || /\s/.test(value.refresh_token)) {
      throw new Error(SESSION_ERROR);
    }
    return {access_token: value.access_token, refresh_token: value.refresh_token};
  }

  try {
    const saved = storage?.getItem(storageKey);
    if (saved) session = readSession(JSON.parse(saved));
  } catch {
    clearSession();
  }

  async function requestToken(grant, body) {
    const epoch = authEpoch;
    try {
      const response = await fetch(`${authOrigin}/auth/v1/token?grant_type=${grant}`, {
        method: "POST", headers: {apikey: key, "Content-Type": "application/json"},
        body: JSON.stringify(body), credentials: "omit", redirect: "error", cache: "no-store",
      });
      if (!response.ok) throw new Error(SIGN_IN_ERROR);
      const next = readSession(await response.json());
      if (epoch !== authEpoch || claims(next.access_token).exp * 1000 <= now()) throw new Error(SESSION_ERROR);
      session = next;
      try { storage?.setItem(storageKey, JSON.stringify(next)); } catch { /* Keep only the in-memory session. */ }
      return next.access_token;
    } catch {
      if (epoch === authEpoch) clearSession();
      throw new Error(grant === "password" ? SIGN_IN_ERROR : SESSION_ERROR);
    }
  }

  async function accessToken() {
    if (!session) throw new Error(SESSION_ERROR);
    if (claims(session.access_token).exp * 1000 > now() + 30000) return session.access_token;
    // Rotating refresh tokens must only be exchanged once, including concurrent API calls.
    if (!refreshPromise) {
      refreshPromise = requestToken("refresh_token", {refresh_token: session.refresh_token})
        .finally(() => { refreshPromise = null; });
    }
    return refreshPromise;
  }

  function mountLogin(message = "") {
    if (loginPromise) return loginPromise;
    const screen = document?.getElementById("loading-screen");
    if (!screen) throw new Error(message || SESSION_ERROR);
    const shell = document.getElementById("app-shell");
    if (shell) shell.hidden = true;
    screen.hidden = false;
    screen.removeAttribute("role");
    const form = document.createElement("form");
    form.className = "recovery-card dialog-card hosted-login";
    form.setAttribute("aria-labelledby", "hosted-sign-in-title");
    const heading = document.createElement("h1");
    heading.id = "hosted-sign-in-title";
    heading.textContent = "Sign in to your workspace";
    const copy = document.createElement("p");
    copy.textContent = "This is a private workspace. Sign in to continue.";
    form.append(heading, copy);
    const field = (name, label, autocomplete) => {
      const caption = document.createElement("label");
      const input = document.createElement("input");
      input.id = `hosted-sign-in-${name}`;
      input.name = name;
      input.type = name;
      input.autocomplete = autocomplete;
      input.required = true;
      caption.htmlFor = input.id;
      caption.textContent = label;
      form.append(caption, input);
      return input;
    };
    const email = field("email", "Email", "username");
    const password = field("password", "Password", "current-password");
    const error = document.createElement("p");
    error.id = "hosted-sign-in-error";
    error.setAttribute("role", "alert");
    error.textContent = message;
    error.hidden = !message;
    form.setAttribute("aria-describedby", error.id);
    const submit = document.createElement("button");
    submit.type = "submit";
    submit.className = "primary";
    submit.textContent = "Sign in";
    form.append(error, submit);
    // Replace the local loading copy and direct-file helper; never interpolate config or errors as HTML.
    screen.replaceChildren(form);
    loginPromise = new Promise(resolve => {
      form.addEventListener("submit", async event => {
        event.preventDefault();
        if (submit.disabled || !form.reportValidity()) return;
        submit.disabled = true;
        form.setAttribute("aria-busy", "true");
        error.hidden = true;
        const credentials = {email: email.value.trim(), password: password.value};
        password.value = "";
        try {
          await requestToken("password", credentials);
          if (message && typeof location.reload === "function") {
            location.reload();
            return;
          }
          const loading = document.createElement("p");
          loading.textContent = "Opening your workspace…";
          screen.replaceChildren(loading);
          screen.setAttribute("role", "status");
          loginPromise = null;
          resolve();
        } catch {
          error.textContent = SIGN_IN_ERROR;
          error.hidden = false;
          password.focus();
        } finally {
          credentials.password = "";
          submit.disabled = false;
          form.removeAttribute("aria-busy");
        }
      });
    });
    email.focus();
    return loginPromise;
  }

  async function ensureHostedLogin() {
    if (session) {
      const epoch = authEpoch;
      try {
        await accessToken();
        if (epoch !== authEpoch) throw new Error(SESSION_ERROR);
        return;
      } catch { return mountLogin(SESSION_ERROR); }
    }
    return mountLogin();
  }

  async function productFetch(input, options) {
    const raw = typeof input === "string" ? input : input instanceof URL ? input.href : input?.url;
    if (typeof raw !== "string" || /[\s\\\u0000-\u001f\u007f]/.test(raw) || raw.startsWith("//")) {
      throw new TypeError("This request address is not allowed.");
    }
    const path = raw.split(/[?#]/)[0];
    if (/(?:^|\/)\.\.(?:\/|$)|\/\.(?:\/|$)|%(?:2e|2f|5c|25)/i.test(path)) {
      throw new TypeError("This request address is not allowed.");
    }
    const url = new URL(raw, location.href);
    const api = url.pathname.startsWith("/api/");
    if (url.username || url.password || ![page.origin, apiOrigin].includes(url.origin)
        || (!api && url.origin !== page.origin) || (api && (url.hash || url.pathname.includes("//")))) {
      throw new TypeError("This request address is not allowed.");
    }
    if (!api) return fetch(input, options);
    const epoch = authEpoch;
    let token;
    try {
      token = await accessToken();
      if (epoch !== authEpoch) throw new Error(SESSION_ERROR);
    } catch {
      mountLogin(SESSION_ERROR);
      throw new Error(SESSION_ERROR);
    }
    const headers = new Headers(options?.headers ?? input?.headers);
    headers.set("Authorization", `Bearer ${token}`);
    const target = `${apiOrigin}${url.pathname}${url.search}`;
    const request = typeof Request !== "undefined" && input instanceof Request ? new Request(target, input) : target;
    const response = await fetch(request, {...options, headers, credentials: "omit", redirect: "error", mode: "cors"});
    if (epoch === authEpoch && (response.status === 401 || response.status === 403)) {
      clearSession();
      mountLogin(SESSION_ERROR);
    }
    return response;
  }

  function currentAppPath() {
    const view = new URL(location.href).searchParams.get("view");
    return `/app/${VIEWS.has(view) ? view : "itinerary"}`;
  }

  function appRouteUrl(route) {
    const view = typeof route === "string" && route.startsWith("/app/") ? route.slice(5) : "itinerary";
    const url = new URL(location.href);
    url.searchParams.delete("view");
    if (VIEWS.has(view) && view !== "itinerary") url.searchParams.set("view", view);
    return `${url.pathname}${url.search}${url.hash}`;
  }

  return {productFetch, currentAppPath, appRouteUrl, ensureHostedLogin, signOutHosted: clearSession};
}

let defaultClient;
let defaultWindow;
let defaultConfig;

function browserClient() {
  const browser = window;
  if (!defaultClient || defaultWindow !== browser || defaultConfig !== browser.PRODUCT_DEPLOYMENT) {
    let storage;
    if (browser.PRODUCT_DEPLOYMENT != null) {
      try { storage = browser.sessionStorage; } catch { /* Storage may be disabled. */ }
    }
    defaultClient = createDeploymentClient({
      config: browser.PRODUCT_DEPLOYMENT, location: browser.location,
      fetch: (input, options) => browser.fetch(input, options), storage, document: browser.document,
    });
    defaultWindow = browser;
    defaultConfig = browser.PRODUCT_DEPLOYMENT;
  }
  return defaultClient;
}

export function productFetch(input, options) {
  if (window.PRODUCT_DEPLOYMENT == null) return window.fetch(input, options);
  return browserClient().productFetch(input, options);
}
export function currentAppPath() { return browserClient().currentAppPath(); }
export function appRouteUrl(route) { return browserClient().appRouteUrl(route); }
export function ensureHostedLogin() { return browserClient().ensureHostedLogin(); }
export function signOutHosted() {
  if (window.PRODUCT_DEPLOYMENT == null) return;
  return browserClient().signOutHosted();
}
