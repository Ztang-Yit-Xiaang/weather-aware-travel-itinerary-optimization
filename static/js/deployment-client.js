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
  const fields = ["api_origin", "supabase_url", "supabase_publishable_key", "static_frontend",
    "allow_signup", "allow_password_recovery"];
  if (!config || typeof config !== "object" || Array.isArray(config)
      || Object.keys(config).some(key => !fields.includes(key)) || config.static_frontend !== true) invalid();
  if (["allow_signup", "allow_password_recovery"].some(
    key => Object.hasOwn(config, key) && typeof config[key] !== "boolean",
  )) invalid();
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
export function createDeploymentClient({
  config = null, location, fetch, storage, document, recoveryStorage, history, now = Date.now,
} = {}) {
  if (config == null) {
    return {
      productFetch: (input, options) => fetch(input, options),
      currentAppPath: () => location.pathname,
      appRouteUrl: route => route,
      ensureHostedLogin: async () => {},
      signOutHosted: () => {},
      workspaceAccountScope: () => "",
    };
  }

  const page = new URL(location.href);
  const {apiOrigin, authOrigin, key} = deploymentConfig(config, page);
  const storageKey = `product:hosted-auth:v1:${JSON.stringify([page.origin, page.pathname, apiOrigin, authOrigin, key])}`;
  const emailFlows = {signup: config.allow_signup === true, recovery: config.allow_password_recovery === true};
  const authReturn = new URL(page.href);
  authReturn.search = "";
  authReturn.hash = "";
  const pendingEmailKey = mode => `product:hosted-email:v1:${authReturn.href}:${authOrigin}:${mode}`;
  let callbackPending = page.searchParams.has("auth");
  let callbackLogin = null;
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

  async function authJson(path, body, {method = "POST", accessToken: bearer} = {}) {
    const headers = {apikey: key, "Content-Type": "application/json"};
    if (bearer) headers.Authorization = `Bearer ${bearer}`;
    const response = await fetch(`${authOrigin}/auth/v1/${path}`, {
      method, headers, body: JSON.stringify(body), credentials: "omit", redirect: "error", cache: "no-store",
    });
    if (!response.ok) throw new Error("auth_email_flow_failed");
    return response;
  }

  async function sendEmailLink(mode, credentials) {
    const verifier = Array.from(crypto.getRandomValues(new Uint8Array(32)), byte => byte.toString(16).padStart(2, "0")).join("");
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
    const challenge = btoa(String.fromCharCode(...new Uint8Array(digest)))
      .replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
    const pendingKey = pendingEmailKey(mode);
    if (!recoveryStorage) throw new Error("auth_email_flow_storage_unavailable");
    const redirect = new URL(authReturn);
    redirect.searchParams.set("auth", mode);
    recoveryStorage.setItem(pendingKey, JSON.stringify({verifier, expires: now() + 3600000}));
    try {
      const action = mode === "signup" ? "signup" : "recover";
      await authJson(`${action}?redirect_to=${encodeURIComponent(redirect.href)}`, {
        ...credentials, code_challenge: challenge, code_challenge_method: "s256",
      });
    } catch (error) {
      recoveryStorage.removeItem(pendingKey);
      throw error;
    }
  }

  async function consumeEmailLink() {
    callbackPending = false;
    const mode = page.searchParams.get("auth");
    const code = page.searchParams.get("code");
    if (!history?.replaceState) throw new Error("auth_email_flow_history_unavailable");
    // Remove the one-time code from the address before any request leaves the page.
    history.replaceState(null, "", authReturn.href);
    if (!Object.hasOwn(emailFlows, mode) || !emailFlows[mode]
        || page.searchParams.getAll("auth").length !== 1 || page.searchParams.getAll("code").length !== 1
        || !/^[\w-]{16,256}$/.test(code ?? "")) {
      throw new Error("auth_email_flow_invalid");
    }
    const pendingKey = pendingEmailKey(mode);
    const pending = JSON.parse(recoveryStorage?.getItem(pendingKey) || "null");
    recoveryStorage?.removeItem(pendingKey);
    if (!/^[0-9a-f]{64}$/.test(pending?.verifier ?? "") || !(pending.expires > now())) {
      throw new Error("auth_email_flow_invalid");
    }
    const response = await authJson("token?grant_type=pkce", {auth_code: code, code_verifier: pending.verifier});
    const recovered = readSession(await response.json());
    if (claims(recovered.access_token).exp * 1000 <= now()) throw new Error("auth_email_flow_invalid");
    clearSession();
    return {mode, token: recovered.access_token};
  }

  function mountLogin(message = "", recoveryToken = null) {
    if (loginPromise) return loginPromise;
    const screen = document?.getElementById("loading-screen");
    if (!screen) throw new Error(message || SESSION_ERROR);
    const shell = document.getElementById("app-shell");
    if (shell) shell.hidden = true;
    screen.hidden = false;
    screen.removeAttribute("role");
    let resolveLogin;
    loginPromise = new Promise(resolve => { resolveLogin = resolve; });

    function render(mode, notice = "", successful = false) {
      const form = document.createElement("form");
      form.className = "recovery-card dialog-card hosted-login";
      form.setAttribute("aria-labelledby", "hosted-sign-in-title");
      const heading = document.createElement("h1");
      heading.id = "hosted-sign-in-title";
      heading.textContent = {
        signin: "Sign in to your workspace", signup: "Create an account",
        recovery: "Reset your password", reset: "Choose a new password",
      }[mode];
      const copy = document.createElement("p");
      copy.textContent = mode === "signin" ? "Sign in to open your workspace." : mode === "reset"
        ? "Choose a password with at least eight characters."
        : "Open the email link in this browser to continue.";
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
      const email = mode === "reset" ? null : field("email", "Email", "username");
      const password = mode === "recovery" ? null : field(
        "password", mode === "reset" ? "New password" : "Password",
        mode === "signin" ? "current-password" : "new-password",
      );
      if (password && mode !== "signin") password.minLength = 8;
      const error = document.createElement("p");
      error.id = "hosted-sign-in-error";
      error.setAttribute("role", successful ? "status" : "alert");
      error.textContent = notice;
      error.hidden = !notice;
      form.setAttribute("aria-describedby", error.id);
      const submit = document.createElement("button");
      submit.type = "submit";
      submit.className = "primary";
      submit.textContent = {signin: "Sign in", signup: "Send confirmation email",
        recovery: "Send reset link", reset: "Update password"}[mode];
      form.append(error, submit);
      const navigate = (label, next) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.addEventListener("click", () => {
          if (submit.disabled) return;
          if (mode === "reset") recoveryToken = null;
          render(next);
        });
        form.append(button);
      };
      if (mode === "signin") {
        if (emailFlows.recovery) navigate("Forgot password?", "recovery");
        if (emailFlows.signup) navigate("Create an account", "signup");
      } else navigate("Back to sign in", "signin");
      // Replace the local loading copy and direct-file helper; never interpolate config or errors as HTML.
      screen.replaceChildren(form);
      form.addEventListener("submit", async event => {
        event.preventDefault();
        if (submit.disabled || !form.reportValidity()) return;
        submit.disabled = true;
        form.setAttribute("aria-busy", "true");
        error.hidden = true;
        const credentials = {};
        if (email) credentials.email = email.value.trim();
        if (password) {
          credentials.password = password.value;
          password.value = "";
        }
        try {
          if (mode === "signup" || mode === "recovery") {
            await sendEmailLink(mode, credentials);
            render("signin", mode === "signup"
              ? "Check your email for a confirmation link."
              : "If an account matches this address, check your email for a reset link.", true);
            return;
          }
          if (mode === "reset") {
            await authJson("user", credentials, {method: "PUT", accessToken: recoveryToken});
            recoveryToken = null;
            clearSession();
            render("signin", "Password updated. Sign in with your new password.", true);
            return;
          }
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
          resolveLogin();
        } catch {
          error.textContent = mode === "signin" ? SIGN_IN_ERROR : mode === "reset"
            ? "Could not update your password. Check the password requirements or request a new link."
            : "Could not send the email. Please try again later.";
          error.hidden = false;
          (password ?? email).focus();
        } finally {
          credentials.password = "";
          submit.disabled = false;
          form.removeAttribute("aria-busy");
        }
      });
      (email ?? password).focus();
    }
    render(recoveryToken ? "reset" : "signin", message);
    return loginPromise;
  }

  async function ensureHostedLogin() {
    if (callbackLogin) return callbackLogin;
    if (callbackPending) {
      callbackLogin = consumeEmailLink().then(
        callback => callback.mode === "recovery" ? mountLogin("", callback.token)
          : mountLogin("Email confirmed. Sign in to continue."),
        () => mountLogin("This link is invalid or expired. Request a new link in this browser."),
      );
      return callbackLogin;
    }
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

  function workspaceAccountScope() {
    return emailFlows.signup ? claims(session?.access_token)?.sub ?? null : "";
  }

  return {productFetch, currentAppPath, appRouteUrl, ensureHostedLogin, signOutHosted: clearSession, workspaceAccountScope};
}

let defaultClient;
let defaultWindow;
let defaultConfig;

function browserClient() {
  const browser = window;
  if (!defaultClient || defaultWindow !== browser || defaultConfig !== browser.PRODUCT_DEPLOYMENT) {
    let storage, recoveryStorage;
    if (browser.PRODUCT_DEPLOYMENT != null) {
      try { storage = browser.sessionStorage; } catch { /* Storage may be disabled. */ }
      if (browser.PRODUCT_DEPLOYMENT.allow_signup || browser.PRODUCT_DEPLOYMENT.allow_password_recovery) {
        try { recoveryStorage = browser.localStorage; } catch { /* Email links need shared browser storage. */ }
      }
    }
    defaultClient = createDeploymentClient({
      config: browser.PRODUCT_DEPLOYMENT, location: browser.location,
      fetch: (input, options) => browser.fetch(input, options), storage, document: browser.document,
      recoveryStorage, history: browser.history,
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
export function workspaceAccountScope() { return browserClient().workspaceAccountScope(); }
export function signOutHosted() {
  if (window.PRODUCT_DEPLOYMENT == null) return;
  return browserClient().signOutHosted();
}
