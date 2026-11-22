import { refreshDrop, t, toast } from "/studio/ui.js";
import { preference, storePreference } from "/studio/session.js";

// IndexedDB can clone a user-selected File, including its filename and MIME type.
// Only the current selection for each input is retained. Restoring it is local.
export function savedFiles() {
  let db, scope;
  const edits = new Map();
  const inputs = [...document.querySelectorAll('input[type="file"][id]')];
  const page = Math.random().toString(36).slice(2);
  let sequence = 0;
  const marker = input => "rvc-studio-file-edit:" + JSON.stringify([scope, input.id]);
  function forget(key, revision) {
    try { if (localStorage.getItem(key) === revision) localStorage.removeItem(key); } catch {}
  }
  function show(input, files) {
    if (!files.length && !input.files.length) return;
    if (files.length) {
      const transfer = new DataTransfer();
      for (const file of files) transfer.items.add(file);
      input.files = transfer.files;
    } else input.value = "";
    const zone = document.querySelector(`[data-drop-for="${input.id}"]`);
    if (zone) refreshDrop(zone, true);
  }
  function save(input, revision) {
    const tx = db.transaction("files", "readwrite");
    const store = tx.objectStore("files");
    const key = [scope, input.id];
    const intent = marker(input);
    const marked = preference(intent, null) === revision;
    if (input.files.length) store.put({ revision, files: [...input.files] }, key);
    else store.delete(key);
    tx.oncomplete = () => forget(intent, revision);
    tx.onabort = () => {
      if (!db || edits.get(input) !== revision || (marked && preference(intent, null) !== revision)) return;
      const cleanup = db.transaction("files", "readwrite");
      cleanup.objectStore("files").delete(key);
      cleanup.oncomplete = () => forget(intent, revision);
      toast(t("fileNotSaved"), true);
    };
    tx.commit?.();
  }
  function changed(input) {
    const revision = (input.files.length ? "file:" : "clear:") + page + ":" + ++sequence;
    edits.set(input, revision);
    if (scope !== undefined) storePreference(marker(input), revision);
    if (db) save(input, revision);
  }
  for (const input of inputs) input.addEventListener("change", () => changed(input));
  return async workspace => {
    scope = workspace;
    for (const [input, revision] of edits) storePreference(marker(input), revision);
    try {
      db = await new Promise((resolve, reject) => {
        const req = indexedDB.open("nvc-studio-files", 1);
        req.onupgradeneeded = () => req.result.createObjectStore("files");
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
      db.onversionchange = () => { db.close(); db = null; };
      for (const input of inputs) {
        if (edits.has(input)) { save(input, edits.get(input)); continue; }
        const intent = marker(input);
        const pending = preference(intent, null);
        if (!pending && input.files.length) { changed(input); continue; }
        // A tab can close after Clear but before its IndexedDB deletion commits.
        const tx = db.transaction("files", pending?.startsWith("clear:") ? "readwrite" : "readonly");
        const store = tx.objectStore("files");
        const key = [scope, input.id];
        const req = store.get(key);
        req.onsuccess = () => {
          if (edits.has(input)) return;
          const latest = preference(intent, null);
          const value = req.result;
          if (latest && latest !== value?.revision) {
            show(input, []);
            if (latest === pending && tx.mode === "readwrite") {
              store.delete(key);
              tx.oncomplete = () => forget(intent, latest);
            }
            return;
          }
          if (latest) forget(intent, latest);
          // Studio 0.2.3 stored an array of Files directly.
          show(input, Array.isArray(value) ? value : value?.files || []);
        };
      }
    } catch {
      // Browsers that disable site storage can still use the regular file picker.
      for (const input of inputs) input.addEventListener("change", () => {
        if (input.files.length) toast(t("fileNotSaved"), true);
      }, { once: true });
    }
  };
}
