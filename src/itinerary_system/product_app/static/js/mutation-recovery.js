const ACTIONS = new Set(["selection", "operations", "undo"]);
const RECORD_KEYS = ["schema_version", "session_id", "trip_id", "run_id", "accepted_plan_id", "action", "mutation_id", "expected_revision", "body"];
const RECEIPT_KEYS = ["mutation_id", "action", "request_sha256", "expected_revision", "committed_revision"];
const ID = /^mutation_[0-9a-f]{32}$/;
const exactKeys = (value, keys) => value && typeof value === "object" && !Array.isArray(value)
  && Object.keys(value).sort().join("|") === [...keys].sort().join("|");
const revision = value => Number.isSafeInteger(value) && value >= 0;
const failure = (code, message) => Object.assign(new Error(message), {code});

function decodeRecord(value) {
  const recovery = value?.action === "recovery";
  if (!exactKeys(value, recovery ? [...RECORD_KEYS, "unreadable_record"] : RECORD_KEYS)
      || value.schema_version !== (recovery ? "product-pending-recovery-v1" : "product-pending-mutation-v1")
      || (recovery ? typeof value.unreadable_record !== "string" : !ACTIONS.has(value.action)) || !ID.test(value.mutation_id)
      || !revision(value.expected_revision) || typeof value.body !== "string"
      || !["session_id", "trip_id", "run_id", "accepted_plan_id"].every(key => typeof value[key] === "string" && value[key])) {
    throw failure("invalid_pending_mutation", "The saved pending action could not be read. Its outcome remains unresolved.");
  }
  const body = JSON.parse(value.body);
  if (!body || Array.isArray(body) || typeof body !== "object"
      || (recovery && !exactKeys(body, ["expected_revision", "mutation_id"]))
      || body.mutation_id !== value.mutation_id || body.expected_revision !== value.expected_revision) {
    throw failure("invalid_pending_mutation", "The saved pending action does not match its request.");
  }
  return Object.freeze({...value});
}

// One durable request per browser tab covers selection, single/batch Add and Undo.
// The application supplies its authenticated API and installs only the returned
// normal session refresh; historical receipt results never replace current state.
export function createMutationRecovery({api, storage, key, getSession, onChange = () => {}}) {
  let pending = null;
  let corrupt = false;
  let active = false;
  let status = "idle";
  let message = "";
  const publish = (next, text = "") => {
    status = next;
    message = text;
    onChange({status, message, blocked: Boolean(pending || corrupt || active), action: pending?.action || null, unreadable: corrupt || pending?.action === "recovery"});
  };
  const sameOwner = (record, session) => session
    && ["session_id", "trip_id", "run_id"].every(field => record[field] === session[field])
    && revision(session.revision);
  const assertOwner = record => {
    if (!sameOwner(record, getSession())) throw failure("mutation_workspace_mismatch", "The pending action belongs to a different workspace. It has not been retried.");
  };
  const pathFor = record => `/api/sessions/${encodeURIComponent(record.session_id)}/${record.action === "selection" ? "selection" : `draft/${record.action}`}`;
  const bounded = async (path, options = {}) => {
    const controller = new AbortController();
    let timer;
    const timeout = new Promise((_, reject) => {
      timer = setTimeout(() => {
        reject(failure("mutation_request_timeout", "The request did not finish in time. Check the saved action before making another edit."));
        controller.abort();
      }, 20_000);
    });
    try {
      return await Promise.race([api(path, {...options, signal: controller.signal}), timeout]);
    } finally {
      clearTimeout(timer);
    }
  };
  const validateReceipt = (record, receipt) => {
    if (!exactKeys(receipt, RECEIPT_KEYS) || receipt.mutation_id !== record.mutation_id
        || receipt.action !== record.action || receipt.expected_revision !== record.expected_revision
        || receipt.committed_revision !== record.expected_revision + 1
        || typeof receipt.request_sha256 !== "string" || !/^[0-9a-f]{64}$/.test(receipt.request_sha256)) {
      throw failure("invalid_mutation_receipt", "The saved-action confirmation did not match the pending request.");
    }
    return receipt;
  };
  const readStatus = async record => {
    assertOwner(record);
    const value = await bounded(`/api/sessions/${encodeURIComponent(record.session_id)}/mutations/${record.mutation_id}`);
    if (!exactKeys(value, ["schema_version", "state", "mutation_id", "mutation_receipt", "result"])
        || value.schema_version !== "product-mutation-status-v1" || value.mutation_id !== record.mutation_id
        || !["applied", "unseen"].includes(value.state)
        || (value.state === "unseen" && (value.mutation_receipt !== null || value.result !== null))) {
      throw failure("invalid_mutation_status", "The saved-action status was incompatible. The action remains unresolved.");
    }
    if (value.state === "applied") {
      validateReceipt(record, value.mutation_receipt);
      if (!value.result || typeof value.result !== "object" || Array.isArray(value.result)) {
        throw failure("invalid_mutation_status", "The saved-action result was incompatible.");
      }
    }
    return value;
  };
  const readWorkspace = async record => {
    assertOwner(record);
    const before = getSession();
    const payload = await bounded(`/api/sessions/${encodeURIComponent(record.session_id)}`);
    assertOwner(record);
    if (!sameOwner(record, payload?.session) || payload.session.revision < before.revision
        || payload.session.revision < getSession().revision
        || !Array.isArray(payload.session.draft) || !payload.workspace
        || payload.workspace.accepted_plan_pointer?.accepted_plan_id !== payload.session.accepted_plan_id
        || !payload.registry) {
      throw failure("invalid_mutation_workspace", "The current workspace could not be verified. The pending action is retained.");
    }
    return payload;
  };
  const clear = () => {
    storage.removeItem(key);
    pending = null;
    corrupt = false;
    publish("idle");
  };
  const confirmed = async (record, proof, current = null) => {
    validateReceipt(record, proof.mutation_receipt);
    current ||= await readWorkspace(record);
    if (current.session.revision < proof.mutation_receipt.committed_revision) current = await readWorkspace(record);
    if (current.session.revision < proof.mutation_receipt.committed_revision) {
      throw failure("invalid_mutation_workspace", "The saved action is confirmed, but its current workspace is not yet available.");
    }
    assertOwner(record);
    clear();
    return {...proof.result, ...current, mutation_receipt: proof.mutation_receipt, recovery_outcome: "applied"};
  };
  const execute = async (record, firstAttempt) => {
    assertOwner(record);
    let payload;
    try {
      payload = await bounded(pathFor(record), {method: "POST", body: JSON.parse(record.body)});
    } catch (error) {
      // These initial-request rejections occur before the new ID can commit.
      // Once an attempt times out, a later rejection cannot settle that older
      // attempt: the explicit status/refresh sequence must resolve it instead.
      if (firstAttempt && error.status === 409 && ["stale_session_revision", "mutation_receipt_limit_reached", "mutation_workspace_changed"].includes(error.code)) clear();
      if (firstAttempt && record.action === "selection" && [400, 422].includes(error.status)
          && ["invalid_selected_day", "invalid_selected_stop", "invalid_selected_segment", "invalid_selected_candidate", "invalid_selected_alternative"].includes(error.code)) clear();
      throw error;
    }
    validateReceipt(record, payload?.mutation_receipt);
    assertOwner(record);
    if (!sameOwner(record, payload?.session) || !Array.isArray(payload.session.draft)
        || payload.session.revision < payload.mutation_receipt.committed_revision) {
      throw failure("invalid_mutation_workspace", "The edit confirmation did not contain a matching current workspace.");
    }
    // A normal keyed response already passes the server's current-draft and
    // proposal validation. Recovery retries still require the full normal GET.
    if (firstAttempt && payload.session.revision === payload.mutation_receipt.committed_revision
        && payload.session.revision >= getSession().revision
        && payload.session.accepted_plan_id === record.accepted_plan_id) {
      clear();
      return {...payload, recovery_outcome: "applied"};
    }
    return confirmed(record, {mutation_receipt: payload.mutation_receipt, result: payload});
  };
  const perform = async task => {
    if (active) throw failure("mutation_recovery_busy", "A saved-action check is already running.");
    active = true;
    publish("checking", "Checking the saved action…");
    try {
      return await task();
    } catch (error) {
      if (pending || corrupt) publish("uncertain", error.message);
      throw error;
    } finally {
      active = false;
      if (!pending && !corrupt) publish("idle");
      else if (status !== "uncertain") publish("uncertain", message || "This action's outcome is not confirmed. Check it before making another edit.");
    }
  };

  const resolve = async record => {
    const first = await readStatus(record);
    if (first.state === "applied") return confirmed(record, first);
    const proof = await bounded(`/api/sessions/${encodeURIComponent(record.session_id)}/mutations/${record.mutation_id}/cancel`, {
      method: "POST", body: {action: record.action === "recovery" ? "undo" : record.action, request: JSON.parse(record.body)},
    });
    assertOwner(record);
    if (!exactKeys(proof, ["schema_version", "state", "mutation_id", "action", "expected_revision", "resolved_revision", "mutation_receipt", "result"])
        || proof.schema_version !== "product-mutation-resolution-v1" || proof.mutation_id !== record.mutation_id
        || proof.action !== (record.action === "recovery" ? "undo" : record.action) || proof.expected_revision !== record.expected_revision
        || !revision(proof.resolved_revision) || proof.resolved_revision <= record.expected_revision
        || !["applied", "not_applied"].includes(proof.state)
        || (record.action === "recovery" && proof.state !== "not_applied")
        || (proof.state === "not_applied" && (proof.mutation_receipt !== null || proof.result !== null))) {
      throw failure("invalid_mutation_resolution", "The pending action could not be safely resolved. Its saved record is retained.");
    }
    if (proof.state === "applied") {
      validateReceipt(record, proof.mutation_receipt);
      if (!proof.result || typeof proof.result !== "object" || Array.isArray(proof.result)) {
        throw failure("invalid_mutation_resolution", "The saved-action result was incompatible.");
      }
    }
    let current = await readWorkspace(record);
    if (current.session.revision < proof.resolved_revision) current = await readWorkspace(record);
    if (current.session.revision < proof.resolved_revision) {
      throw failure("invalid_mutation_workspace", "The resolution is confirmed, but its current workspace is not yet available.");
    }
    if (proof.state === "applied") return confirmed(record, proof, current);
    assertOwner(record);
    clear();
    return {...current, recovery_outcome: record.action === "recovery" ? "refreshed" : "not_applied"};
  };

  return {
    get blocked() { return Boolean(pending || corrupt || active); },
    get pending() { return pending ? {...pending} : null; },
    async restore() {
      try {
        const saved = storage.getItem(key);
        if (!saved) {
          if (pending || corrupt) throw failure("invalid_pending_mutation", "The saved action is unavailable. Refresh the protected workspace before editing.");
          return null;
        }
        pending = decodeRecord(JSON.parse(saved));
        corrupt = false;
        assertOwner(pending);
        publish("uncertain", "A previous action is awaiting confirmation. Check its saved status before continuing.");
        return {...pending};
      } catch (error) {
        corrupt = true;
        const problem = error.code ? error : failure("invalid_pending_mutation", "The saved action could not be read. Refresh the protected workspace before editing.");
        publish("uncertain", problem.message);
        throw problem;
      }
    },
    async submit(action, body) {
      if (pending || corrupt || active) throw failure("mutation_outcome_uncertain", "Resolve the saved action before making another edit.");
      const session = getSession();
      if (!ACTIONS.has(action) || !revision(body?.expected_revision) || body.expected_revision !== session?.revision
          || Object.hasOwn(body, "mutation_id")) throw failure("invalid_mutation_request", "The edit does not match the current workspace.");
      const mutationId = `mutation_${crypto.randomUUID().replaceAll("-", "")}`;
      const record = decodeRecord({schema_version: "product-pending-mutation-v1", ...Object.fromEntries(
        ["session_id", "trip_id", "run_id", "accepted_plan_id"].map(field => [field, session[field]])),
        action, mutation_id: mutationId, expected_revision: body.expected_revision,
        body: JSON.stringify({...body, mutation_id: mutationId}, (_key, value) => {
          if (typeof value === "number" && !Number.isFinite(value)) throw failure("invalid_mutation_request", "The edit contains an invalid numeric value. Nothing was sent.");
          return value;
        })});
      try {
        storage.setItem(key, JSON.stringify(record));
      } catch {
        throw failure("mutation_journal_unavailable", "The browser could not save this pending action. Nothing was sent. Make storage available before retrying.");
      }
      pending = record;
      return perform(() => execute(record, true));
    },
    async retry() {
      if (!pending || corrupt) throw failure("invalid_pending_mutation", "No readable pending action is available to retry.");
      return perform(() => pending.action === "recovery" ? resolve(pending) : execute(pending, false));
    },
    async cancel() {
      if (!pending || corrupt) throw failure("invalid_pending_mutation", "No readable pending action is available to cancel.");
      return perform(() => resolve(pending));
    },
    async repair() {
      if (pending?.action === "recovery" && !corrupt) return perform(() => resolve(pending));
      if (!corrupt) throw failure("invalid_pending_mutation", "No unreadable pending action needs a protected refresh.");
      return perform(async () => {
        let saved;
        try { saved = storage.getItem(key); }
        catch { throw failure("mutation_journal_unavailable", "Browser storage is unavailable. Restore access, then refresh the protected workspace again. Nothing was sent."); }
        let readable = null;
        try { if (saved) readable = decodeRecord(JSON.parse(saved)); } catch { /* Preserve unreadable bytes in the recovery record. */ }
        if (readable) {
          assertOwner(readable);
          pending = readable;
          corrupt = false;
          return resolve(readable);
        }
        const owner = getSession();
        const current = await readWorkspace(owner);
        const mutationId = `mutation_${crypto.randomUUID().replaceAll("-", "")}`;
        const record = decodeRecord({schema_version: "product-pending-recovery-v1", ...Object.fromEntries(
          ["session_id", "trip_id", "run_id", "accepted_plan_id"].map(field => [field, current.session[field]])),
          action: "recovery", mutation_id: mutationId, expected_revision: current.session.revision,
          unreadable_record: saved || "", body: JSON.stringify({expected_revision: current.session.revision, mutation_id: mutationId})});
        try {
          if (storage.getItem(key) !== saved) throw failure("mutation_journal_changed", "The saved action changed during refresh. Read it again before continuing.");
          storage.setItem(key, JSON.stringify(record));
        } catch (error) {
          if (error.code === "mutation_journal_changed") throw error;
          throw failure("mutation_journal_unavailable", "The browser could not save the protected refresh. The unreadable record was kept. Nothing was sent.");
        }
        pending = record;
        corrupt = false;
        publish("checking", "Refreshing the protected workspace. Already saved changes will be kept.");
        return resolve(record);
      });
    },
    async check() {
      if (!pending || corrupt) throw failure("invalid_pending_mutation", "No readable pending action is available to check.");
      return perform(async () => {
        const record = pending;
        const first = await readStatus(record);
        if (first.state === "applied") return confirmed(record, first);
        const current = await readWorkspace(record);
        const last = await readStatus(record);
        if (last.state === "applied") return confirmed(record, last, current);
        // A later persisted revision, followed by an unseen receipt read, proves
        // an older expected-revision request can no longer commit. Unseen alone
        // never authorizes clearing a pending request or sending a fresh edit.
        if (current.session.revision > record.expected_revision) {
          clear();
          return {...current, recovery_outcome: record.action === "recovery" ? "refreshed" : "not_applied"};
        }
        throw failure("mutation_outcome_uncertain", "This action is not confirmed yet. It may still finish. Check again or retry the exact saved action.");
      });
    },
  };
}
