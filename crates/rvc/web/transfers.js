import { api, t, toast, upload } from "/studio/ui.js";

const endpoint = id => "/api/studio/job/" + encodeURIComponent(id);

async function revokeOnce(id, dataset) {
  // A proxy can finish forwarding the body just as the browser aborts. Studio
  // uses the ticket as the native job ID, so this DELETE covers both cases.
  try {
    const reply = await api(endpoint(id), { method: "DELETE", keepalive: true });
    return reply?.upload_cancelled || dataset ? { uploadCancelled: true } : { id };
  } catch (error) {
    if (error.status !== 404) throw error;
    if (!dataset) {
      try { await api(endpoint(id) + "?cursor=0"); return { id }; }
      catch (lookup) { if (lookup.status !== 404) throw lookup; }
    }
    return { uploadCancelled: true };
  }
}

async function revoke(id, dataset) {
  for (let attempt = 1; ; attempt++) {
    try { return await revokeOnce(id, dataset); }
    catch (error) {
      if (error.status && error.status < 500) throw error;
      if (attempt === 1) toast(t("cancelUploadRetry"));
      await new Promise(resolve => setTimeout(resolve, Math.min(attempt * 500, 5000)));
    }
  }
}

export function audioUpload(button, cancelling) {
  const controller = new AbortController();
  const { signal } = controller;
  const idleKey = button.dataset.i18n;
  const label = key => { button.dataset.i18n = key; button.textContent = t(key); };
  const cancel = () => {
    if (signal.aborted) return;
    button.disabled = true;
    label("cancellingUpload");
    cancelling();
    controller.abort();
  };
  label("cancelUpload");
  button.disabled = false;
  button.hidden = false;
  button.onclick = cancel;

  async function send(url, settings, files, progress) {
    let onAbort;
    try {
      let ticket;
      try { ticket = await api("/api/studio/conversion-ticket", { method: "POST" }); }
      catch (error) { if (signal.aborted) return { uploadCancelled: true }; throw error; }
      const { id } = ticket;
      let cancellation;
      onAbort = () => {
        // Record a rejection immediately even if XHR has not delivered its abort event yet.
        cancellation = revoke(id, url === "/api/studio/dataset").then(value => ({ value }), error => ({ error }));
      };
      signal.addEventListener("abort", onAbort, { once: true });
      if (signal.aborted) onAbort();
      else {
        try {
          const result = await upload(url + "?ticket=" + encodeURIComponent(id), settings, files, progress, signal);
          if (!signal.aborted) return result;
        } catch (error) { if (!signal.aborted) throw error; }
      }
      const { value, error } = await cancellation;
      if (error) throw error;
      return value;
    } finally {
      if (onAbort) signal.removeEventListener("abort", onAbort);
      if (button.onclick === cancel) {
        button.onclick = null;
        button.hidden = true;
        button.disabled = false;
        label(idleKey);
      }
    }
  }
  return { cancel, send };
}
