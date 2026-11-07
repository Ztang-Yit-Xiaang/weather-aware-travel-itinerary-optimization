// Keep the workspace across browser restarts; unfinished actions belong to one session.
export function createWorkspaceStorage({persistent, tab, sessionKey, scopedKeys = []}) {
  const read = (storage, key) => { try { return storage.getItem(key); } catch { return null; } };
  const owner = () => {
    try { return JSON.parse(read(tab, sessionKey) || read(persistent, sessionKey) || "null")?.session_id; }
    catch { return null; }
  };
  const scoped = key => scopedKeys.includes(key) ? `${key}:${owner() || "unbound"}` : key;
  return {
    getItem(key) {
      const legacy = read(tab, key);
      if (legacy !== null) {
        try { persistent.setItem(scoped(key), legacy); } catch { /* Tab storage remains available. */ }
        return legacy;
      }
      const value = read(persistent, scoped(key));
      if (key === sessionKey && value !== null) {
        try { tab.setItem(key, value); } catch { /* Persistent storage is still available. */ }
      }
      return value;
    },
    setItem(key, value) {
      let saved = false;
      try { persistent.setItem(scoped(key), value); saved = true; } catch { /* Try tab storage. */ }
      try { tab.setItem(key, value); saved = true; } catch { /* Persistent storage may be available. */ }
      if (!saved) throw new Error("Workspace storage is unavailable. Enable browser storage before editing.");
    },
    removeItem(key) {
      const target = scoped(key);
      // Closing an older workspace must not remove another tab's newer resume link.
      if (key !== sessionKey || read(persistent, key) === read(tab, key)) persistent.removeItem(target);
      tab.removeItem(key);
    },
  };
}

export function restoreInterestPreferences(storage, key) {
  const defaults = {enabled: true, radiusKm: 5, naturePreference: 0, massLevel: 0.8, searchArea: "visitor_area"};
  try {
    const saved = JSON.parse(storage.getItem(key) || "null");
    if (!saved || typeof saved.enabled !== "boolean" || ![1, 2, 5, 10].includes(saved.radiusKm)
        || ![-1, 0, 1].includes(saved.naturePreference) || ![0.5, 0.8, 0.95].includes(saved.massLevel)
        || !["visitor_area", "itinerary_pin"].includes(saved.searchArea)) return defaults;
    return {...defaults, ...Object.fromEntries(Object.keys(defaults).map(key => [key, saved[key]]))};
  } catch { return defaults; }
}
