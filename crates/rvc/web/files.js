import { refreshDrop, t, toast } from "/studio/ui.js";

// IndexedDB can clone a user-selected File, including its filename and MIME type.
// Only the current selection for each input is retained. Restoring it is local.
export function savedFiles() {
  let db, scope;
  const edits = new Map();
  const inputs = [...document.querySelectorAll('input[type="file"][id]')];
  function save(input) {
    const revision = edits.get(input);
    const tx = db.transaction("files", "readwrite");
    const store = tx.objectStore("files");
    const key = [scope, input.id];
    if (input.files.length) store.put([...input.files], key);
    else store.delete(key);
    tx.onabort = () => {
      // Never show a previous selection in place of a new file that exceeded the quota.
      if (!db || edits.get(input) !== revision) return;
      const cleanup = db.transaction("files", "readwrite");
      cleanup.objectStore("files").delete(key);
      toast(t("fileNotSaved"), true);
    };
  }
  for (const input of inputs) input.addEventListener("change", () => {
    edits.set(input, (edits.get(input) || 0) + 1);
    if (db) save(input);
  });
  return async workspace => {
    scope = workspace;
    try {
      db = await new Promise((resolve, reject) => {
        const req = indexedDB.open("nvc-studio-files", 1);
        req.onupgradeneeded = () => req.result.createObjectStore("files");
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
      db.onversionchange = () => { db.close(); db = null; };
      for (const input of inputs) {
        if (edits.has(input) || input.files.length) { save(input); continue; }
        const req = db.transaction("files").objectStore("files").get([scope, input.id]);
        req.onsuccess = () => {
          if (edits.has(input) || !req.result?.length) return;
          const transfer = new DataTransfer();
          for (const file of req.result) transfer.items.add(file);
          input.files = transfer.files;
          const zone = document.querySelector(`[data-drop-for="${input.id}"]`);
          if (zone) refreshDrop(zone, true);
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
