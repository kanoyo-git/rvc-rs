import { api, upload, log, t, toast, facts } from "/studio/ui.js";

const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const path = id => "/api/studio/job/" + encodeURIComponent(id);
const byId = id => document.getElementById(id);
let snapshotKey = "rvc-studio-output-v1/";

// Short, local actions (e.g. importing a dataset) have no server job ID.
// Remember their text or fact rows; job transcripts and audio stay on the server.
function readOutput(id) {
  try { return JSON.parse(localStorage.getItem(snapshotKey + id)); } catch { return null; }
}
export function rememberOutput(node) {
  const rows = node.matches("dl") ? [...node.querySelectorAll("dt")].map(term => [term.textContent, term.nextElementSibling?.textContent || ""]) : null;
  const note = { after: node.dataset.jobId || null, text: rows ? "" : node.textContent, error: node.classList.contains("is-error"), rows };
  try { localStorage.setItem(snapshotKey + node.id, JSON.stringify(note)); } catch { /* Storage can be disabled. */ }
}
function restoreNotes() {
  document.querySelectorAll(".log[id], #insInfo").forEach(node => {
    const note = readOutput(node.id);
    if (note?.rows && node.matches("dl")) facts(node, note.rows);
    else if (typeof note?.text === "string") log(node, note.text, note.error);
  });
}

function saveReadingPositions() {
  for (const node of document.querySelectorAll(".log[data-job-id]")) {
    if (!node.clientHeight) continue;
    const top = node.scrollTop + node.clientHeight >= node.scrollHeight - 4 ? null : node.scrollTop;
    try { localStorage.setItem(snapshotKey + "scroll:" + node.id, JSON.stringify({ id: node.dataset.jobId, top })); } catch {}
  }
}

function restoreReadingPosition(output, id) {
  const position = readOutput("scroll:" + output.id);
  if (position?.id !== id || !Number.isFinite(position.top)) return;
  const apply = () => { if (output.dataset.jobId === id) output.scrollTop = position.top; };
  if (output.clientHeight) { apply(); return; }
  const observer = new ResizeObserver(() => {
    if (output.clientHeight || output.dataset.jobId !== id) { observer.disconnect(); apply(); }
  });
  observer.observe(output);
}

export function studioJobs({ tally, progressBar, displayLine, completed, available, explain }) {
  let active;
  let syncing;
  let restored = false;
  const showing = new Map();

  function attach(task, job) {
    if (active && active !== task) return;
    active = task;
    if (task.stop) return;
    tally.start(job.operation, job.started_at || Date.now());
    const stop = byId(job.studio.stop);
    const start = stop.previousElementSibling;
    task.stop = () => {
      stop.onclick = null;
      stop.hidden = true;
      if (start.dataset.jobLock === task.id) {
        delete start.dataset.jobLock;
        start.disabled = !available(start);
      }
      task.progress("");
      tally.stop();
    };
    start.dataset.jobLock = task.id;
    start.disabled = true;
    stop.hidden = false;
    stop.disabled = false;
    stop.onclick = async () => {
      stop.disabled = true;
      try { await api(path(task.id), { method: "DELETE" }); }
      catch (error) { explain(error); stop.disabled = false; }
    };
  }

  async function follow(task, output, meta, original) {
    const { id } = task;
    task.output = output.id;
    showing.set(output.id, id);
    output.dataset.jobId = id;
    task.progress = progressBar(output);
    let cursor = 0;
    let retries = 0;
    let running = original || meta.status === "running";
    let text;
    let restoredPosition = original;
    let keepNote = !running && readOutput(output.id)?.after === id;
    // Preserve a newer local action in this pane, such as a dataset import after preprocessing.
    if (!keepNote) log(output, t("working"));
    if (meta.studio && running) attach(task, meta);
    for (;;) {
      if (showing.get(output.id) !== id) return;
      let job;
      try {
        job = await api(path(id) + "?cursor=" + cursor, { cache: "no-store" });
        if (!job?.studio) throw new Error(t("jobUnavailable"));
        retries = 0;
      } catch (error) {
        if (error.status === 404 || (error.status && error.status < 500) || error.message === t("jobUnavailable")) {
          if (!keepNote) { output.append("\n" + t("jobUnavailable")); output.classList.add("is-error"); }
          return;
        }
        if (++retries === 1) toast(t("jobReconnect"));
        await pause(Math.min(retries * 1000, 5000));
        continue;
      }
      if (showing.get(output.id) !== id) return;
      if (job.status === "running") { running = true; keepNote = false; attach(task, job); }
      if (running && task.stop) task.progress(job.status === "running" ? job.message : "");
      if (!keepNote && job.log.length) {
        if (!text) { text = document.createTextNode(""); output.replaceChildren(text); }
        const atBottom = output.scrollTop + output.clientHeight >= output.scrollHeight - 4;
        text.appendData((text.length ? "\n" : "") + job.log.map(displayLine).join("\n"));
        if (atBottom) output.scrollTop = output.scrollHeight;
      }
      cursor = job.next_cursor;
      if (job.more) continue;
      if (!restoredPosition) { restoreReadingPosition(output, id); restoredPosition = true; }
      if (job.status === "running") { await pause(600); continue; }
      if (!keepNote) {
        if (!text) output.replaceChildren();
        const end = job.status === "complete" ? t("done") : job.status === "cancelled" ? t("cancelled") : job.message;
        output.append((text ? "\n" : "") + end);
        output.classList.toggle("is-error", job.status === "failed");
      }
      let pending = false;
      try { pending = localStorage.getItem(snapshotKey + "pending:" + output.id) === id; } catch {}
      if (job.status === "complete") {
        await completed(job, output, running || pending, keepNote);
        if (running) toast(t("doneShort") + " · " + t("op_" + job.operation));
      }
      if (pending) { try { localStorage.removeItem(snapshotKey + "pending:" + output.id); } catch {} }
      return;
    }
  }

  function track(task, output, meta, original = false) {
    return follow(task, output, meta, original).finally(() => {
      if (active === task) { task.stop?.(); active = null; }
    });
  }

  async function run(settings, files, output) {
    if (active) throw new Error(t("busyTask"));
    const task = { output: output.id };
    active = task;
    showing.delete(output.id);
    try {
      log(output, t("working"));
      const reply = await upload("/api/studio/job", settings, files, files.length ? pct => log(output, `${t("uploading")} · ${pct}%`) : null);
      task.id = reply.id;
      try { localStorage.setItem(snapshotKey + "pending:" + output.id, task.id); } catch {}
      await track(task, output, { status: "running" }, true);
    } finally { if (active === task) active = null; }
  }

  async function discover() {
    const prior = new Map(showing);
    const records = await api("/api/studio/jobs", { cache: "no-store" });
    for (const job of records) {
      const output = byId(job.studio.output);
      if (!output || showing.get(output.id) === job.id || (active?.output === output.id && active.id !== job.id)
          || showing.get(output.id) !== prior.get(output.id)) continue;
      track({ id: job.id }, output, job).catch(error => explain(error));
    }
    if (!active) {
      // Jobs submitted through the CLI-compatible web API still share the compute slot.
      const current = await api("/api/job");
      if (current?.status === "running" && !active && !records.some(job => job.id === current.id)) {
        // A Studio job can start between the list and status requests.
        const latest = (await api("/api/studio/jobs", { cache: "no-store" })).find(job => job.id === current.id);
        if (active) return;
        if (latest) {
          const output = byId(latest.studio.output);
          if (output && showing.get(output.id) !== latest.id) track({ id: latest.id }, output, latest).catch(error => explain(error));
          return;
        }
        const foreign = { id: current.id };
        active = foreign;
        tally.start(current.operation, current.started_at || Date.now());
        const poll = async () => {
          if (active !== foreign) return;
          const job = await api("/api/job").catch(() => null);
          if (!job || (job.id === foreign.id && job.status === "running")) setTimeout(poll, 1500);
          else { active = null; tally.stop(); refresh().catch(error => explain(error)); }
        };
        setTimeout(poll, 1500);
      }
    }
  }
  function refresh() {
    if (!syncing) syncing = discover().finally(() => { syncing = null; });
    return syncing;
  }
  function restore(scope) {
    if (!restored) {
      restored = true;
      snapshotKey += encodeURIComponent(scope) + "/";
      restoreNotes();
      document.addEventListener("visibilitychange", () => {
        if (document.hidden) saveReadingPositions();
        else refresh().catch(error => explain(error));
      });
      window.addEventListener("pagehide", saveReadingPositions);
      window.addEventListener("online", () => refresh().catch(error => explain(error)));
    }
    return refresh();
  }
  return { run, restore, busy: () => Boolean(active) };
}
