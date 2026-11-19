"use strict";

const byId = id => document.getElementById(id);
const state = { catalog: [], model: null, busy: false, limits: {}, urls: [] };

function announce(message, error = false) {
  const node = byId("status");
  node.textContent = message;
  node.toggleAttribute("data-error", error);
}

async function api(path, method = "GET", body, type = "application/json") {
  const headers = new Headers();
  const key = byId("api-key").value.trim();
  if (key) headers.set("Authorization", `Bearer ${key}`);
  if (body !== undefined) headers.set("Content-Type", type);
  const payload = body !== undefined && type === "application/json" ? JSON.stringify(body) : body;
  const response = await fetch(path, { method, headers, body: payload });
  if (!response.ok) {
    if (response.status === 401) byId("connection").open = true;
    const message = await response.json().catch(() => ({}));
    throw new Error(message.error?.message || message.detail || `Server returned HTTP ${response.status}`);
  }
  return response;
}

function selection() {
  return state.catalog.find(entry => entry.id === byId("catalog").value);
}

function buttons() {
  const entry = selection();
  byId("load").disabled = state.busy || !entry?.pymss.supported;
  byId("download").disabled = state.busy || !entry;
  byId("separate").disabled = state.busy || !state.model || !byId("audio-file").files.length;
  byId("save-source").disabled = state.busy;
  byId("refresh").disabled = state.busy;
}

async function perform(message, action, done) {
  if (state.busy) return;
  state.busy = true;
  buttons();
  announce(message);
  try {
    await action();
    announce(done);
  } catch (error) {
    announce(error.message, true);
  } finally {
    state.busy = false;
    buttons();
  }
}

function renderDetails() {
  const entry = selection();
  const description = byId("model-detail");
  if (!entry) description.textContent = "No models match the filters.";
  else {
    const model = entry.pymss;
    const availability = model.local.complete ? "Downloaded" : `${model.local.missing_count} file(s) needed`;
    const size = model.size_bytes > 0 ? `${(model.size_bytes / 1048576).toFixed(1)} MiB` : "";
    description.textContent = [model.architecture, model.target_stem, size, availability,
      model.supported ? "" : model.unsupported_reason].filter(Boolean).join(" · ");
  }
  buttons();
}

function renderCatalog() {
  const list = byId("catalog");
  const previouslySelected = list.value;
  const query = byId("search").value.trim().toLowerCase();
  const category = byId("category").value;
  const matches = state.catalog.filter(({ id, pymss: model }) =>
    (model.supported || byId("show-unsupported").checked) &&
    (!byId("local-only").checked || model.local.complete) &&
    (!category || category === model.primary_category) &&
    (!query || [id, model.architecture, model.target_stem, ...model.aliases].some(value =>
      String(value || "").toLowerCase().includes(query))));
  list.replaceChildren(...matches.map(({ id, pymss: model }) => new Option(
    `${id}${model.local.complete ? " (downloaded)" : ""}`, id)));
  list.value = matches.some(entry => entry.id === previouslySelected) ? previouslySelected : matches[0]?.id || "";
  renderDetails();
}

function renderActive(model) {
  state.model = model;
  const details = model?.pymss;
  byId("active-model").textContent = details ? `${model.id} · ${details.sample_rate} Hz` : "No model loaded.";
  const controls = (details?.instruments || []).map(stem => {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = stem;
    checkbox.checked = true;
    label.append(checkbox, ` ${stem}`);
    return label;
  });
  byId("stems").replaceChildren(...controls);
  const parameters = details ? Object.values(details.supported_parameters || {}).flat() : [];
  byId("parameters").textContent = parameters.length ? `Supported by the loaded model: ${parameters.join(", ")}` : "";
  buttons();
}

async function refresh() {
  const [server, catalog] = await Promise.all([
    api("/v1/server/info").then(response => response.json()),
    api("/v1/catalog/models?supported=all").then(response => response.json()),
  ]);
  state.catalog = catalog.data;
  state.limits = server.limits;
  byId("source").value = server.download_source.source;
  byId("endpoint").value = server.download_source.endpoint || "";
  const category = byId("category");
  const selected = category.value;
  const categories = [...new Set(state.catalog.map(row => row.pymss.primary_category).filter(Boolean))].sort();
  category.replaceChildren(new Option("All categories", ""), ...categories.map(value => new Option(value, value)));
  if (categories.includes(selected)) category.value = selected;
  renderCatalog();
  renderActive(server.model);
}

async function convertInput(file, rate) {
  const context = new OfflineAudioContext(1, 1, rate);
  let audio = await context.decodeAudioData(await file.arrayBuffer());
  if (state.limits.max_audio_seconds > 0 && audio.duration > state.limits.max_audio_seconds) {
    throw new Error(`Audio exceeds ${state.limits.max_audio_seconds} seconds.`);
  }
  if (audio.numberOfChannels > 2) {
    const stereo = new OfflineAudioContext(2, audio.length, rate);
    const source = stereo.createBufferSource();
    source.buffer = audio;
    source.connect(stereo.destination);
    source.start();
    audio = await stereo.startRendering();
  }
  const channels = Array.from({ length: audio.numberOfChannels }, (_, index) => audio.getChannelData(index));
  const bytes = audio.length * channels.length * 4;
  if (state.limits.max_request_bytes > 0 && bytes > state.limits.max_request_bytes) {
    throw new Error("Decoded audio exceeds the server request limit.");
  }
  const pcm = new DataView(new ArrayBuffer(bytes));
  let offset = 0;
  for (let frame = 0; frame < audio.length; frame++) {
    for (const channel of channels) {
      pcm.setFloat32(offset, channel[frame], true);
      offset += 4;
    }
  }
  return { bytes: pcm.buffer, channels: channels.length, rate };
}

function wavBlob(audio) {
  const binary = atob(audio.data);
  const pcm = Uint8Array.from(binary, char => char.charCodeAt(0));
  const header = new DataView(new ArrayBuffer(56));
  for (const [offset, word] of [[0, "RIFF"], [8, "WAVE"], [12, "fmt "], [36, "fact"], [48, "data"]]) {
    for (let i = 0; i < word.length; i++) header.setUint8(offset + i, word.charCodeAt(i));
  }
  const frameSize = audio.channels * 4;
  header.setUint32(4, pcm.length + 48, true);
  header.setUint32(16, 16, true);
  header.setUint16(20, 3, true);
  header.setUint16(22, audio.channels, true);
  header.setUint32(24, audio.sample_rate, true);
  header.setUint32(28, audio.sample_rate * frameSize, true);
  header.setUint16(32, frameSize, true);
  header.setUint16(34, 32, true);
  header.setUint32(40, 4, true);
  header.setUint32(44, pcm.length / frameSize, true);
  header.setUint32(52, pcm.length, true);
  return new Blob([header.buffer, pcm], { type: "audio/wav" });
}

function downloadLink(blob, filename, label) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  state.urls.push(link.href);
  link.download = filename;
  link.textContent = label;
  return link;
}

function showResult(data, filename, preview) {
  const results = byId("results");
  state.urls.forEach(url => URL.revokeObjectURL(url));
  state.urls = [];
  results.replaceChildren();
  if (!preview) {
    const row = document.createElement("p");
    row.append(downloadLink(data, `${filename}_stems.zip`, "Download ZIP archive"));
    results.append(row);
    return;
  }
  for (const { stem, audio } of data.outputs) {
    const item = document.createElement("div");
    item.className = "stem";
    const heading = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = stem;
    const link = downloadLink(wavBlob(audio), `${filename}_${stem.replace(/[\\/]/g, "_")}.wav`, "Download WAV");
    heading.append(name, link);
    const player = document.createElement("audio");
    player.controls = true;
    player.preload = "metadata";
    player.src = link.href;
    item.append(heading, player);
    results.append(item);
  }
}

byId("source").replaceChildren(new Option("ModelScope", "modelscope"), new Option("Hugging Face", "huggingface"), new Option("HF Mirror", "hf-mirror"));
for (const id of ["search", "category", "local-only", "show-unsupported"]) byId(id).addEventListener("input", renderCatalog);
byId("catalog").addEventListener("change", renderDetails);
byId("audio-file").addEventListener("change", buttons);
byId("refresh").addEventListener("click", () => perform("Connecting…", refresh, "Connected."));
byId("save-source").addEventListener("click", () => perform("Updating download source…", async () => {
  await api("/v1/download-source", "POST", { source: byId("source").value, endpoint: byId("endpoint").value.trim() || null });
  await refresh();
}, "Download source updated."));
byId("download").addEventListener("click", () => perform("Downloading model files…", async () => {
  await api("/v1/models/download", "POST", { model: selection().id, timeout_seconds: 1800 });
  await refresh();
}, "Model files are ready."));
byId("load").addEventListener("click", () => perform("Loading model…", async () => {
  const text = byId("params").value.trim();
  const inference_params = text ? JSON.parse(text) : {};
  if (inference_params === null || Array.isArray(inference_params) || typeof inference_params !== "object") {
    throw new Error("Inference overrides must be a JSON object.");
  }
  let response;
  try {
    response = await api("/v1/models/load", "POST", { model: selection().id, inference_params });
  } catch (error) {
    const info = await api("/v1/server/info").then(result => result.json()).catch(() => null);
    renderActive(info?.model ?? null);
    throw error;
  }
  renderActive((await response.json()).model);
  await refresh();
}, "Model loaded."));
byId("separation").addEventListener("submit", event => {
  event.preventDefault();
  perform("Separating audio…", async () => {
    const file = byId("audio-file").files[0];
    if (!file || !state.model) throw new Error("Choose a model and an audio file first.");
    const stems = [...byId("stems").querySelectorAll("input:checked")].map(input => input.value);
    if (byId("stems").children.length && !stems.length) throw new Error("Choose at least one stem.");
    const input = await convertInput(file, state.model.pymss.sample_rate);
    const output = byId("output-format").value;
    const preview = output === "preview";
    const query = new URLSearchParams({ model: state.model.id, format: "pcm_f32le", sample_rate: input.rate,
      channels: input.channels, response_format: preview ? "json" : "zip", output_audio_format: preview ? "pcm_f32le" : output });
    if (stems.length) query.set("stems", stems.join(","));
    const response = await api(`/v1/audio/separations?${query}`, "POST", input.bytes, "application/octet-stream");
    showResult(preview ? await response.json() : await response.blob(), file.name.replace(/\.[^.]+$/, ""), preview);
  }, "Separation complete.");
});

perform("Connecting…", refresh, "Connected.");
