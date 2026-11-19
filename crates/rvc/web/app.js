"use strict";

const find = (selector, node = document) => node.querySelector(selector);
const all = (selector, node = document) => [...node.querySelectorAll(selector)];
const modelTemplate = find("#model-fields");
let liveProfileLoaded = false;
let liveStatus;

for (const field of all(".model-fields")) field.append(modelTemplate.content.cloneNode(true));
const liveForm = find("#live-form");
liveForm.elements.index_rate.value = "0";
liveForm.elements.rms_mix_rate.value = "0";
liveForm.elements.resample_sr.closest("label").hidden = true;
liveForm.elements.protect.closest("label").hidden = true;
liveForm.elements.pitch.step = "0.1";

function showPanel(id) {
  for (const button of all("nav [data-panel]")) {
    if (button.dataset.panel === id) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  for (const panel of all("main > section")) panel.hidden = panel.id !== id;
  if (id === "realtime" && !liveProfileLoaded) {
    loadLiveProfile().catch(error => status(find("#live-status"), error.message, true));
  }
}

for (const button of all("nav [data-panel]")) {
  button.addEventListener("click", () => showPanel(button.dataset.panel));
}

async function request(url, init) {
  const response = await fetch(url, init);
  if (!response.ok) {
    const message = response.headers.get("content-type")?.includes("application/json")
      ? (await response.json()).error
      : await response.text();
    throw new Error(message || response.statusText);
  }
  return response;
}

const sendJson = (url, body, method = "POST") => request(url, {
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

function status(node, message, error = false) {
  node.hidden = !message;
  node.classList.toggle("error", error);
  node.textContent = message;
}

function fill(select, entries, emptyOption) {
  const previous = select.value;
  const elements = entries.map(([value, label]) => new Option(label, value));
  if (emptyOption) elements.unshift(new Option(emptyOption, ""));
  select.replaceChildren(...elements);
  if (elements.some(option => option.value === previous)) select.value = previous;
}

async function configure() {
  const options = await (await request("/api/options")).json();
  setModels(options.models);
  const labels = {
    "dereverb-less-aggressive-18.8050": "Remove reverb",
    "dereverb-anvuew-19.1729": "Remove reverb (aggressive)",
    "vocals-bs-roformer-368": "Separate vocals and accompaniment",
    "vocals-bs-roformer-317": "Separate vocals and accompaniment (aggressive)",
    "karaoke-mel-roformer-10.1956": "Main and backing vocals",
  };
  fill(find("#presets"), options.presets.map(preset => [preset.id, labels[preset.id] || preset.label]));
  translate();
  find("#presets").value = "vocals-bs-roformer-368";
  find("#live-nav").hidden = !options.realtime;
  find("#index-form").hidden = !options.index;
  find("#train-all").hidden = !options.index;
  if (options.realtime) pollLive();
  const panel = new URLSearchParams(location.search).get("panel");
  if (all("nav [data-panel]").some(button => button.dataset.panel === panel && !button.hidden)) showPanel(panel);
}

function setModels(models) {
  for (const select of all(".models")) {
    fill(select, models.map(model => [model, model]), t(models.length ? "Select a model" : "No models in assets/weights"));
  }
}

async function refreshModels() {
  setModels((await (await request("/api/options")).json()).models);
}

for (const select of all(".models")) {
  select.addEventListener("change", async () => {
    const model = select.value;
    const speakers = select.closest("form")?.elements.speaker_id;
    const details = select.closest("#model-info") ? find("#model-details") : null;
    if (speakers) fill(speakers, [], t("Automatic"));
    if (details) details.replaceChildren();
    if (!model || (!speakers && !details)) return;
    try {
      const info = await (await request("/api/model/" + encodeURIComponent(model))).json();
      if (model !== select.value) return;
      const names = Object.entries(info.speakers);
      if (speakers) {
        const entries = names.length ? names : Array.from({ length: Math.min(info.speaker_count, 4096) }, (_, id) => [id, String(id)]);
        fill(speakers, entries, t("Automatic"));
      }
      if (details) {
        const fields = [
          ["Version", info.version],
          ["Sample rate", info.sample_rate ? String(info.sample_rate) + " Hz" : "Unknown"],
          ["Pitch guidance", info.uses_f0 ? "Yes" : "No"],
          ["Speakers", names.length ? names.map(([id, name]) => name + " (" + id + ")").join(", ") : String(info.speaker_count) + " IDs"],
          ["Description", info.info || "None"],
        ];
        for (const [key, value] of fields) {
          const dt = document.createElement("dt");
          const dd = document.createElement("dd");
          dt.textContent = key;
          dd.textContent = value;
          details.append(dt, dd);
        }
      }
    } catch (error) {
      status(find("#startup"), error.message, true);
    }
  });
}

function values(form) {
  const result = {};
  for (const field of form.elements) {
    if (!field.name || field.disabled || field.type === "file" || field.type === "submit") continue;
    if (field.type === "checkbox") result[field.name] = field.checked;
    else if (field.type === "number") result[field.name] = field.value === "" ? null : Number(field.value);
    else result[field.name] = field.value.trim() || null;
  }
  return result;
}

function conversion(formValues) {
  const { model, speaker_id, index, device, pitch, f0_method, index_rate, resample_sr, rms_mix_rate, protect, formant } = formValues;
  return {
    model, speaker_id: speaker_id === null ? null : Number(speaker_id), index, device: device || "auto",
    settings: { pitch, f0_method, index_rate, resample_sr, rms_mix_rate, protect, formant: formant || 0 },
  };
}

let audioUrl;
find("#convert-form").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const message = find(".message", form);
  const model = conversion(values(form));
  if (!model.model) {
    status(message, "Select a voice model.", true);
    return;
  }
  const button = find("button[type=submit]", form);
  const data = new FormData();
  data.append("settings", JSON.stringify(model));
  data.append("audio", form.elements.audio.files[0]);
  button.disabled = true;
  status(message, t("Converting audio…"));
  try {
    const result = await (await request("/api/convert", { method: "POST", body: data })).blob();
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = URL.createObjectURL(result);
    find("#preview").src = audioUrl;
    find("#preview").hidden = false;
    find("#download").href = audioUrl;
    find("#download").hidden = false;
    status(message, "");
  } catch (error) {
    status(message, error.message, true);
  } finally {
    button.disabled = false;
  }
});

find("#release-voice").addEventListener("click", async () => {
  try {
    await request("/api/voice", { method: "DELETE" });
    const model = find("#convert-form .models");
    model.value = "";
    model.dispatchEvent(new Event("change"));
  } catch (error) {
    status(find("#convert-form .message"), error.message, true);
  }
});

function showJob(job) {
  find("#job").hidden = job.status === "idle";
  find("#stop-job").hidden = job.status !== "running";
  const names = {
    batch: "Batch conversion", separate: "Separation", preprocess: "Dataset preparation",
    extract: "Feature extraction", index: "Index training", train: "Voice model training", train_all: "Training pipeline",
  };
  const label = names[job.operation] ?? "Background task";
  const states = { running: "Running", complete: "Complete", failed: "Failed", cancelled: "Cancelled" };
  find("#job-title").textContent = [label, states[job.status]].filter(Boolean).map(t).join(" · ");
  status(find("#job-message"), job.message, job.status === "failed");
  find("#job-result").hidden = job.result === null;
  find("#job-result").textContent = job.result === null ? "" : JSON.stringify(job.result, null, 2);
}

for (const form of all(".task-form")) {
  form.addEventListener("submit", async event => {
    event.preventDefault();
    const input = values(form);
    const operation = event.submitter?.dataset.operation || form.dataset.operation;
    const data = { operation, ...input };
    if (operation === "batch") {
      if (!input.model) { status(find("#startup"), "Select a voice model.", true); return; }
      Object.assign(data, conversion(input));
    }
    if (typeof input.sample_rate === "string") data.sample_rate = Number(input.sample_rate);
    if ("gpus" in input) data.gpus = input.gpus ? input.gpus.split(",").map(Number) : [];
    if (operation === "train" || operation === "train_all") {
      const selected = find("#training-precision").value;
      data.fp16 = selected ? selected === "true" : null;
    }
    try {
      await sendJson("/api/job", data);
      status(find("#startup"), "");
      await pollJob();
    } catch (error) {
      status(find("#startup"), error.message, true);
    }
  });
}

find("#stop-job").addEventListener("click", async () => {
  try {
    await request("/api/job", { method: "DELETE" });
    status(find("#job-message"), t("Cancellation requested…"));
  } catch (error) {
    status(find("#job-message"), error.message, true);
  }
});

async function pollJob() {
  const job = await (await request("/api/job")).json();
  const changed = find("#job").dataset.status !== job.status;
  find("#job").dataset.status = job.status;
  showJob(job);
  if (changed && job.status === "complete" && ["train", "train_all"].includes(job.operation)) await refreshModels();
}

for (const form of all(".model-action")) {
  form.addEventListener("submit", async event => {
    event.preventDefault();
    const data = { operation: form.dataset.operation, ...values(form) };
    data.description ||= "";
    if (data.sample_rate) data.sample_rate = Number(data.sample_rate);
    const message = find(".message", form);
    try {
      const result = await (await sendJson("/api/model", data)).json();
      status(message, result.path);
      await refreshModels();
    } catch (error) {
      status(message, error.message, true);
    }
  });
}

find("#add-speaker").addEventListener("click", () => {
  const row = find("#speaker-row").content.firstElementChild.cloneNode(true);
  find(".remove-speaker", row).addEventListener("click", () => row.remove());
  find("#speaker-rows").append(row);
});

find("#manifest-form").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const rows = all(".speaker-row").map(row => Object.fromEntries(all("[data-field]", row)
    .map(input => [input.dataset.field, input.value])));
  const output = find(".message", form);
  try {
    const result = await (await sendJson("/api/manifest", { ...values(form), rows: rows.length ? rows : null })).json();
    const skipped = result.invalid_rows?.length ? " · Invalid rows " + result.invalid_rows.join(", ") : "";
    status(output, result.path + skipped);
  } catch (error) {
    status(output, error.message, true);
  }
});

async function loadDevices(host) {
  const url = host ? "/api/devices?host=" + encodeURIComponent(host) : "/api/devices";
  const catalog = await (await request(url)).json();
  fill(liveForm.elements.host, catalog.available_hosts.map(name => [name, name]));
  liveForm.elements.host.value = catalog.host;
  for (const [field, devices] of [["input_device", catalog.inputs], ["output_device", catalog.outputs]]) {
    fill(liveForm.elements[field], devices.map(device => [device.id, device.name]), "System default");
  }
  liveForm.elements.wasapi_exclusive.closest("label").hidden = !catalog.host.toLowerCase().includes("wasapi");
  return catalog;
}

async function loadLiveProfile() {
  liveProfileLoaded = true;
  const saved = await (await request("/api/realtime/profile")).json();
  for (const key of ["block_ms", "crossfade_ms", "extra_ms", "threshold_db", "device", "index"]) {
    if (saved[key] != null) liveForm.elements[key].value = saved[key];
  }
  for (const key of ["pitch", "f0_method", "index_rate", "formant"]) {
    if (saved.settings[key] != null) liveForm.elements[key].value = saved.settings[key];
  }
  for (const key of ["monitor", "wasapi_exclusive", "input_noise_reduction", "output_noise_reduction"]) {
    liveForm.elements[key].checked = saved[key];
  }
  liveForm.elements.rms_mix_rate.value = saved.rms_mix;
  liveForm.elements.rate_choice.value = saved.model_rate ? "model" : String(saved.sample_rate || "device");
  if (saved.model) {
    const name = saved.model.split(/[/\\]/).pop();
    const model = liveForm.elements.model;
    if ([...model.options].some(entry => entry.value === name)) {
      model.value = name;
      model.dispatchEvent(new Event("change"));
      if (saved.speaker_id != null) {
        const speakers = liveForm.elements.speaker_id;
        const observer = new MutationObserver(() => {
          if (model.value === name) speakers.value = String(saved.speaker_id);
          observer.disconnect();
        });
        observer.observe(speakers, { childList: true });
      }
    } else {
      liveForm.elements.model_path.value = saved.model;
    }
  }
  const defaults = await loadDevices();
  const host = defaults.available_hosts.find(name => name.toLowerCase() === saved.host?.toLowerCase());
  const available = host && host !== defaults.host ? await loadDevices(host) : defaults;
  for (const [field, devices] of [["input_device", available.inputs], ["output_device", available.outputs]]) {
    const device = devices.find(device => device.id === saved[field] || device.name === saved[field]);
    if (device) liveForm.elements[field].value = device.id;
  }
}

liveForm.elements.host.addEventListener("change", () => {
  loadDevices(liveForm.elements.host.value).catch(error => status(find("#live-status"), error.message, true));
});

liveForm.addEventListener("submit", async event => {
  event.preventDefault();
  const data = values(liveForm);
  const { rate_choice } = data;
  const convert = conversion(data);
  if (data.model_path) convert.model = data.model_path;
  if (!data.monitor && !convert.model) {
    status(find("#live-status"), "Select a voice model.", true);
    return;
  }
  data.sample_rate = rate_choice === "model" || rate_choice === "device" ? null : Number(rate_choice);
  data.model_rate = rate_choice === "model";
  data.rms_mix = convert.settings.rms_mix_rate;
  Object.assign(data, convert);
  try {
    await sendJson("/api/realtime", data);
    await readLive();
  } catch (error) {
    status(find("#live-status"), error.message, true);
  }
});

const liveFields = new Set(["pitch", "formant", "f0_method", "index_rate", "rms_mix_rate", "threshold_db", "monitor", "input_noise_reduction", "output_noise_reduction"]);
liveForm.addEventListener("change", async event => {
  if (liveStatus !== "running" || !liveFields.has(event.target.name) || !event.target.checkValidity()) return;
  const data = values(liveForm);
  const { threshold_db, rms_mix_rate, monitor, input_noise_reduction, output_noise_reduction } = data;
  try {
    await sendJson("/api/realtime", { threshold_db, rms_mix: rms_mix_rate, monitor,
      input_noise_reduction, output_noise_reduction, settings: conversion(data).settings }, "PATCH");
  } catch (error) {
    status(find("#live-status"), error.message, true);
  }
});

find("#stop-live").addEventListener("click", async () => {
  try {
    await request("/api/realtime", { method: "DELETE" });
  } catch (error) {
    status(find("#live-status"), error.message, true);
  }
});

async function readLive() {
  const info = await (await request("/api/realtime")).json();
  liveStatus = info.status;
  const button = find("#live-form button[type=submit]");
  button.disabled = ["starting", "running"].includes(info.status);
  find("#stop-live").disabled = !button.disabled;
  for (const field of liveForm.elements) {
    if (field.name && !liveFields.has(field.name)) field.disabled = button.disabled;
  }
  if (info.status === "running") {
    const { sample_rate, block_frames } = info.format;
    const { device_xruns, worker } = info.stats;
    const detail = sample_rate + " Hz · " + Math.round(block_frames * 1000 / sample_rate) +
      " ms per block · " + Math.round(worker.last_inference_us / 1000) + " ms processing · " + device_xruns +
      " device dropouts · " + (worker.processed_blocks ?? 0) + " blocks";
    status(find("#live-status"), info.message ? detail + " · " + info.message : detail, Boolean(info.message));
  } else {
    status(find("#live-status"), info.message || (info.status === "starting" ? t("Loading model and audio devices…") : ""), info.status === "failed");
  }
}

function pollLive() {
  const poll = () => readLive().catch(() => {}).finally(() => setTimeout(poll, 750));
  poll();
}

const showStartupError = error => status(find("#startup"), error.message, true);
interfaceLanguage(showStartupError).catch(showStartupError);
configure().catch(showStartupError);
setInterval(() => pollJob().catch(() => {}), 1250);
