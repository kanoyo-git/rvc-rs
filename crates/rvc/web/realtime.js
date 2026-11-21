import { $, api, send, fill, t, log, toast } from "/studio/ui.js";

const editable = new Set(["pitch", "formant", "f0_method", "index_rate", "rms_mix_rate",
  "threshold_db", "monitor", "input_noise_reduction", "output_noise_reduction"]);

export function realtime(options) {
  const form = $("#live-form");
  const fields = form.elements;
  const output = $("#live-status");
  const start = $('button[type="submit"]', form);
  const stop = $("#stop-live");
  let open = false;
  let loaded;
  let timer;
  let status = "idle";
  let speakerRevision = 0;
  let requestError;
  let stopping = false;

  const stats = (values) => {
    ["lsRate", "lsBlock", "lsProc", "lsDrops"].forEach((id, i) => { $("#" + id).textContent = values?.[i] ?? "—"; });
  };
  const error = (error) => {
    requestError = error.message;
    log(output, error.message, true);
    toast(error.message, true);
  };
  function controls(busy) {
    start.disabled = busy;
    stop.disabled = !busy || stopping;
    for (const field of fields) {
      if (field.name && !editable.has(field.name)) field.disabled = busy;
    }
  }

  async function loadSpeakers(wanted) {
    const revision = ++speakerRevision;
    const name = fields.model.value;
    if (!name) {
      fill(fields.speaker_id, []);
      return;
    }
    const info = await api("/api/model/" + encodeURIComponent(name));
    if (revision !== speakerRevision) return;
    const named = Object.entries(info.speakers);
    const speakers = named.length
      ? named.map(([id, name]) => ({ value: id, label: `${name} · ${id}` }))
      : Array.from({ length: Math.min(1000, info.speaker_count) }, (_, i) => String(i));
    fill(fields.speaker_id, speakers, wanted == null ? "" : String(wanted));
  }

  function modelsChanged(models) {
    fill(fields.model, [{ value: "", key: "noVoices" }, ...models]);
    if (loaded && open) loadSpeakers(fields.speaker_id.value).catch(error);
  }

  async function loadDevices(host, saved = {}) {
    const query = host ? "?" + new URLSearchParams({ host }) : "";
    const devices = await api("/api/devices" + query);
    fill(fields.host, devices.available_hosts, devices.host);
    for (const [field, list] of [["input_device", devices.inputs], ["output_device", devices.outputs]]) {
      const selected = list.find(device => device.id === saved[field] || device.name === saved[field]);
      fill(fields[field], [{ value: "", key: "systemDefault" }, ...list.map(device => ({
        value: device.id, label: device.name,
      }))], selected?.id || "");
    }
    fields.wasapi_exclusive.closest("label").hidden = !devices.host.toLowerCase().includes("wasapi");
  }

  async function setup() {
    const saved = await api("/api/realtime/profile");
    for (const field of ["block_ms", "crossfade_ms", "extra_ms", "threshold_db", "index"]) {
      if (saved[field] != null) fields[field].value = saved[field];
    }
    for (const field of ["pitch", "index_rate", "formant"]) {
      fields[field].value = saved.settings[field];
      fields[field].dispatchEvent(new Event("input"));
    }
    for (const field of ["monitor", "wasapi_exclusive", "input_noise_reduction", "output_noise_reduction"]) {
      fields[field].checked = Boolean(saved[field]);
    }
    for (const method of fields.f0_method.options) method.disabled = !options.pitch[method.value];
    fields.f0_method.value = options.pitch[saved.settings.f0_method] ? saved.settings.f0_method : "pm";
    fields.rms_mix_rate.value = saved.rms_mix;
    fields.rms_mix_rate.dispatchEvent(new Event("input"));
    fields.rate_choice.value = saved.model_rate ? "model" : String(saved.sample_rate || "device");
    if (!fields.rate_choice.value) {
      fields.rate_choice.add(new Option(saved.sample_rate, saved.sample_rate));
      fields.rate_choice.value = String(saved.sample_rate);
    }
    modelsChanged(options.models);
    if (saved.model) {
      const name = saved.model.split(/[\\/]/).pop();
      if (options.models.includes(name)) fields.model.value = name;
      else fields.model_path.value = saved.model;
    }
    await loadSpeakers(saved.speaker_id);
    const available = await api("/api/devices");
    const host = available.available_hosts.find(name => name.toLowerCase() === saved.host?.toLowerCase());
    await loadDevices(host, saved);
  }

  function liveSettings() {
    const asNumber = (name) => fields[name].valueAsNumber;
    const rms = asNumber("rms_mix_rate");
    return {
      threshold_db: asNumber("threshold_db"), rms_mix: rms, monitor: fields.monitor.checked,
      input_noise_reduction: fields.input_noise_reduction.checked,
      output_noise_reduction: fields.output_noise_reduction.checked,
      settings: {
        pitch: asNumber("pitch"), formant: asNumber("formant"), index_rate: asNumber("index_rate"),
        f0_method: fields.f0_method.value, rms_mix_rate: rms,
      },
    };
  }

  fields.model.addEventListener("change", () => {
    fields.model_path.value = "";
    loadSpeakers().catch(error);
  });
  fields.host.addEventListener("change", () => loadDevices(fields.host.value).catch(error));
  form.addEventListener("submit", async event => {
    event.preventDefault();
    const model = fields.model_path.value.trim() || fields.model.value || null;
    if (!model && !fields.monitor.checked) {
      error(new Error(t("chooseVoice")));
      return;
    }
    const rate = fields.rate_choice.value;
    const data = {
      ...liveSettings(), model, speaker_id: !fields.model_path.value.trim() && fields.speaker_id.value ? Number(fields.speaker_id.value) : null,
      index: fields.index.value.trim() || null,
      host: fields.host.value || null, input_device: fields.input_device.value || null,
      output_device: fields.output_device.value || null, wasapi_exclusive: fields.wasapi_exclusive.checked,
      model_rate: rate === "model", sample_rate: Number(rate) || null,
      block_ms: fields.block_ms.valueAsNumber, crossfade_ms: fields.crossfade_ms.valueAsNumber, extra_ms: fields.extra_ms.valueAsNumber,
    };
    controls(true);
    status = "starting";
    requestError = null;
    log(output, t("liveStarting"));
    try {
      await send("/api/realtime", data);
    } catch (failure) {
      status = "idle";
      controls(false);
      error(failure);
    }
  });
  form.addEventListener("change", event => {
    if (status === "running" && editable.has(event.target.name) && event.target.checkValidity()) {
      send("/api/realtime", liveSettings(), "PATCH").catch(error);
    }
  });
  stop.addEventListener("click", async () => {
    stopping = true;
    stop.disabled = true;
    try {
      await api("/api/realtime", { method: "DELETE" });
    } catch (failure) {
      stopping = false;
      error(failure);
      stop.disabled = false;
    }
  });

  async function refresh() {
    const info = await api("/api/realtime");
    status = info.status;
    const busy = status === "starting" || status === "running";
    if (!busy) stopping = false;
    controls(busy);
    if (status === "running") {
      const { sample_rate, block_frames } = info.format;
      const { device_xruns, worker } = info.stats;
      const block = Math.round(block_frames * 1000 / sample_rate);
      const inference = Math.round(worker.last_inference_us / 1000);
      stats([`${(sample_rate / 1000).toFixed(1)} kHz`, `${block} ms`, `${inference} ms`, String(device_xruns)]);
      $("#lsProc").classList.toggle("is-over", inference > block);
      const detail = [info.active_f0_method && `${t("f0Method")} · ${info.active_f0_method}`, info.message];
      log(output, detail.filter(Boolean).join("\n"), Boolean(info.message));
    } else {
      stats(null);
      const message = info.message || requestError || (status === "starting" ? t("liveStarting") : "");
      log(output, message, status === "failed" || Boolean(requestError));
    }
  }

  async function poll() {
    try {
      await refresh();
    } catch (failure) {
      log(output, failure.message, true);
    } finally {
      if (open) timer = setTimeout(poll, status === "idle" ? 2000 : 750);
    }
  }
  return {
    modelsChanged,
    get active() { return open; },
    refresh: () => refresh().catch(error),
    activate(active) {
      if (active === open) return;
      open = active;
      clearTimeout(timer);
      if (!active) return;
      loaded ??= setup().catch(failure => {
        loaded = undefined;
        error(failure);
      });
      loaded.then(() => { if (open) poll(); });
    },
  };
}
