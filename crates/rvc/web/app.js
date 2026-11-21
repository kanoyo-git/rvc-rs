import {
  $, $$, api, send, upload, value, checked, radio, t, log, fill, facts, toast, clock,
  bindRanges, syncRange, bindDrops, refreshDrop, enhanceAudioPlayers, repaintPlayers,
} from "/studio/ui.js";
import { realtime } from "/studio/realtime.js";

let options;
let live;
let activeJob;
let metrics = {};
let datasetPath;
let experiments = { runs: [], resumable: [], all: [] };
const fileList = (id) => [...document.getElementById(id).files];
const fileName = (path) => String(path).split(/[\\/]/).pop();
const number = (id) => Number(value(id));
const optionalNumber = (id) => value(id) === "" ? null : number(id);
// Native errors can carry a whole libtorch backtrace: keep the part a person can act on.
const tidy = (message) => String(message).split(/\n(?:Exception raised from|frame #\d)/)[0].trim();
const explain = (error, target) => {
  const message = tidy(error.message);
  if (target) log(target, message, true);
  toast(message.length > 240 ? message.slice(0, 237) + "…" : message, true);
};

function requireValue(value, key) {
  if (!value) throw new Error(t(key));
  return value;
}

function listen(selector, event, handler) {
  $$(selector).forEach(node => node.addEventListener(event, async (event) => {
    try {
      await handler(event);
    } catch (error) {
      explain(error);
    }
  }));
}

function action(id, output, run) {
  listen("#" + id, "click", async event => {
    const button = event.currentTarget;
    const target = $("#" + output);
    button.disabled = true;
    try {
      log(target, "");
      await run(target);
    } catch (error) {
      explain(error, target);
    } finally {
      button.disabled = false;
    }
  });
}

/* Language and theme */
const prefs = {
  theme: localStorage.getItem("rvc-studio-theme") === "light" ? "light" : "dark",
  lang: localStorage.getItem("rvc-studio-lang") || (navigator.language.startsWith("ru") ? "ru" : "en"),
};
function localize() {
  const root = document.documentElement;
  root.lang = prefs.lang === "ru" ? "ru" : "en";
  root.dataset.theme = prefs.theme;
  $('meta[name="theme-color"]').content = prefs.theme === "light" ? "#f4f4f5" : "#0f0f10";
  $$("[data-i18n]").forEach(node => {
    if (["true", undefined].includes(node.dataset.empty)) node.textContent = t(node.dataset.i18n);
  });
  $$("[data-i18n-attr]").forEach(node => {
    for (const pair of node.dataset.i18nAttr.split(";")) {
      const [attr, key] = pair.split(":");
      node.setAttribute(attr, t(key));
    }
  });
  $("#themeToggle").setAttribute("aria-label", t(prefs.theme === "light" ? "themeToDark" : "themeToLight"));
  $("#langSelect").value = root.lang;
  document.title = "NVC Studio · " + t("tab" + capital(currentPanel));
  $$("[data-drop-for]").forEach(zone => refreshDrop(zone));
  renderSpeakers();
  syncSeparationModel();
  syncGate();
  tally.render();
  if (Object.keys(metrics).length) renderMetrics();
  repaintPlayers();
  if (live?.active) live.refresh();
}
const capital = (name) => ({ infer: "Infer", cover: "Cover", realtime: "Realtime", separate: "Separate", train: "Train",
  layers: "Layers", metrics: "Metrics", speakers: "Speakers", library: "Library", ckpt: "Ckpt", faq: "Faq" })[name] || "Infer";
listen("#langSelect", "change", () => {
  prefs.lang = value("langSelect");
  localStorage.setItem("rvc-studio-lang", prefs.lang);
  localize();
});
listen("#themeToggle", "click", () => {
  prefs.theme = prefs.theme === "light" ? "dark" : "light";
  localStorage.setItem("rvc-studio-theme", prefs.theme);
  localize();
});

/* Navigation */
let currentPanel = "infer";
const panelAliases = { convert: "infer", training: "train", separation: "separate", models: "ckpt", tensorboard: "metrics", helper: "speakers" };
function showPanel(name, updateLocation = false) {
  name = panelAliases[name] || name;
  if (!$$(".nav-item").some(tab => tab.dataset.tab === name && !tab.hidden)) name = "infer";
  currentPanel = name;
  for (const tab of $$(".nav-item")) {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("is-active", active);
    if (active) tab.setAttribute("aria-current", "page");
    else tab.removeAttribute("aria-current");
  }
  for (const panel of $$(".panel")) panel.classList.toggle("is-active", panel.dataset.panel === name);
  for (const audio of $$("audio")) if (!audio.closest(".panel.is-active")) audio.pause();
  $("#workspace").scrollTop = 0;
  document.title = "NVC Studio · " + t("tab" + capital(name));
  if (updateLocation) {
    const url = new URL(location.href);
    if (name === "infer") url.searchParams.delete("panel");
    else url.searchParams.set("panel", name);
    if (url.href !== location.href) history.pushState(null, "", url);
  }
  live?.activate(name === "realtime");
  if (!options) return;
  if (name === "metrics") refreshRuns().then(loadMetrics).catch(error => explain(error, $("#tbStatus")));
  if (name === "train") Promise.all([loadPretrained(), refreshRuns()]).catch(error => explain(error, $("#trainLog")));
  if (name === "layers") refreshRuns().catch(error => explain(error, $("#laLog")));
  requestAnimationFrame(repaintPlayers);
}
listen(".nav-item", "click", event => showPanel(event.currentTarget.dataset.tab, true));
window.addEventListener("popstate", () => showPanel(new URLSearchParams(location.search).get("panel")));

/* Signature: the tally lamp shows the one task the studio runs */
const tally = {
  operation: null, started: 0, timer: null, percent: null,
  start(operation, started = Date.now()) {
    this.operation = operation;
    this.started = started;
    clearInterval(this.timer);
    this.timer = setInterval(() => this.render(), 1000);
    this.render();
  },
  stop() {
    this.operation = null;
    this.percent = null;
    clearInterval(this.timer);
    this.render();
  },
  render() {
    const live = Boolean(this.operation);
    $("#tally").classList.toggle("is-live", live);
    $("#tallyText").textContent = live ? t("op_" + this.operation) : t("idle");
    $("#tallyTime").textContent = !live ? "" : this.percent != null ? Math.round(this.percent) + "%" : clock((Date.now() - this.started) / 1000);
  },
};

// A bar above a task's log for any step that reports a percentage (downloads, mostly).
function progressBar(output) {
  let node = output.previousElementSibling;
  if (!node?.classList.contains("progress")) {
    node = document.createElement("div");
    node.className = "progress";
    node.hidden = true;
    node.innerHTML = '<span class="progress-text"></span><span class="progress-track"><span class="progress-fill"></span></span>';
    output.before(node);
  }
  return (message) => {
    const match = /(\d+(?:\.\d+)?)%/.exec(message || "");
    node.hidden = !match;
    tally.percent = match ? Number(match[1]) : null;
    tally.render();
    if (!match) return;
    $(".progress-text", node).textContent = message;
    $(".progress-fill", node).style.transform = `scaleX(${Math.min(100, Number(match[1])) / 100})`;
  };
}

async function runJob(settings, files, output, stopId) {
  if (activeJob) throw new Error(t("busyTask"));
  const showProgress = progressBar(output);
  const stop = stopId ? $("#" + stopId) : null;
  const task = { url: null };
  activeJob = task;
  if (stop) {
    stop.disabled = false;
    stop.onclick = async () => {
      stop.disabled = true;
      try {
        await api(task.url, { method: "DELETE" });
      } catch (error) {
        explain(error, output);
        stop.disabled = false;
      }
    };
  }
  try {
    log(output, t("working"));
    const { id } = await upload("/api/studio/job", settings, files, files.length ? (pct) => {
      log(output, `${t("uploading")} · ${pct}%`);
    } : null);
    task.url = "/api/studio/job/" + id;
    tally.start(settings.operation);
    if (stop) stop.hidden = false;
    let previous = "";
    for (;;) {
      const job = await api(task.url);
      const text = job.log.join("\n");
      showProgress(job.status === "running" ? job.message : "");
      if (job.status === "complete") {
        log(output, text ? text + "\n" + t("done") : t("done"));
        toast(t("doneShort") + " · " + t("op_" + settings.operation));
        return job.result;
      }
      if (job.status === "cancelled") {
        log(output, t("cancelled") + (text ? "\n" + text : ""));
        return null;
      }
      if (job.status === "failed") throw new Error(job.message);
      if (text && text !== previous) {
        log(output, text);
        output.scrollTop = output.scrollHeight;
        previous = text;
      }
      await new Promise(resolve => setTimeout(resolve, 600));
    }
  } finally {
    activeJob = null;
    showProgress("");
    tally.stop();
    if (stop) {
      stop.onclick = null;
      stop.hidden = true;
    }
  }
}

async function watchForeignJob() {
  const job = await api("/api/job").catch(() => null);
  if (!job || job.status !== "running" || activeJob) return;
  tally.start(job.operation);
  const timer = setInterval(async () => {
    const now = await api("/api/job").catch(() => null);
    if (activeJob) return clearInterval(timer);
    if (!now || now.status !== "running") {
      clearInterval(timer);
      tally.stop();
    }
  }, 2000);
}

/* Voices */
const embedderLabels = { hubert_base: "HuBERT Base", contentvec: "ContentVec", spin: "Spin", "spin-v2": "Spin-v2", mhubert147: "mHuBERT-147", w2v_ru: "wav2vec 2.0 ru" };

// Indexes inside an experiment folder get the experiment name, to tell copies apart.
function indexLabel(path) {
  const parts = String(path).split(/[\\/]/);
  const experiments = String(options?.experiments || "").replace(/[\\/]+$/, "");
  return String(path).startsWith(experiments + "/") ? `${parts.pop()} · ${parts.pop()}` : parts.pop();
}

function voicePicker(prefix, onChange) {
  const el = (id) => $("#" + prefix + id);
  let revision = 0;
  let data = null;
  const sync = () => {
    el("IndexPathWrap").hidden = el("Index").value !== "path";
    onChange?.();
  };
  async function load(speakerChanged = false) {
    const current = ++revision;
    const model = el("Model").value;
    const speaker = speakerChanged ? el("Speaker").value : "";
    const previousIndex = speakerChanged ? el("Index").value : null;
    data = null;
    el("VoiceMeta").textContent = "";
    if (!model) {
      el("SpeakerWrap").hidden = true;
      fill(el("Index"), [{ value: "", key: "noIndex" }, { value: "path", key: "indexFile" }], "");
      sync();
      return;
    }
    const query = new URLSearchParams();
    if (speaker) query.set("speaker", speaker);
    try {
      const result = await api("/api/studio/voice/" + encodeURIComponent(model) + "?" + query);
      if (current !== revision) return;
      data = result;
      const info = result.info;
      // Only multi-speaker models name their speakers; the rest always use speaker 0.
      const named = Object.entries(info.speakers);
      const speakers = named.length
        ? named.map(([id, name]) => ({ value: id, label: `${name} · ${id}` }))
        : [{ value: String(result.speaker), label: String(result.speaker) }];
      el("SpeakerWrap").hidden = named.length <= 1;
      fill(el("Speaker"), speakers, String(result.speaker));
      const choices = [
        { value: "", key: "noIndex" },
        ...(result.index ? [{ value: "suggested", label: `${t("autoIndex")} · ${fileName(result.index)}` }] : []),
        ...(result.indexes || []).filter(path => path !== result.index)
          .map(path => ({ value: "file:" + path, label: indexLabel(path) })),
        { value: "path", key: "indexFile" },
      ];
      const keep = previousIndex !== null && choices.some(choice => choice.value === previousIndex);
      fill(el("Index"), choices, keep ? previousIndex : (result.index ? "suggested" : ""));
      el("VoiceMeta").textContent = [
        info.version, info.sample_rate ? info.sample_rate / 1000 + " kHz" : null,
        embedderLabels[info.embedder] || info.embedder, info.uses_f0 ? "f0" : t("noPitch"),
      ].filter(Boolean).join(" · ");
    } catch (error) {
      if (current === revision) explain(error);
    }
    sync();
  }
  el("Model").addEventListener("change", () => load());
  el("Speaker").addEventListener("change", () => load(true));
  el("Index").addEventListener("change", sync);
  return {
    load,
    get data() { return data; },
    setModels(models) {
      fill(el("Model"), models.length ? models : [{ value: "", key: "noVoices" }]);
      return load();
    },
    usesIndex: () => Boolean(el("Index").value),
    request() {
      const voice = requireValue(data, "chooseVoice");
      const use = el("Index").value;
      return {
        model: el("Model").value, speaker_id: voice.speaker,
        index: use === "path" ? requireValue(el("IndexPath").value.trim(), "chooseIndex")
          : use.startsWith("file:") ? use.slice(5) : (use ? voice.index : null),
      };
    },
  };
}

const inferVoice = voicePicker("i", () => {
  $("#iIndexRate").disabled = !inferVoice?.usesIndex();
  const voice = inferVoice?.data;
  $("#iAutoRegister").disabled = !voice?.info.uses_f0;
  syncProfile();
  syncGate();
});
const coverVoice = voicePicker("c", () => {
  $("#cIndexRate").disabled = !coverVoice?.usesIndex();
  syncGate();
});

function syncGate() {
  for (const [prefix, picker] of [["i", inferVoice], ["c", coverVoice]]) {
    const available = Boolean(picker?.data?.gate);
    $("#" + prefix + "Gate").disabled = !available;
    $("#" + prefix + "GateHint").textContent = t(available ? "voiceGateHint" : "voiceGateMissing");
  }
}

async function updateModels() {
  const { models } = await api("/api/options");
  options.models = models;
  $("#modelNames").replaceChildren(...models.map(model => new Option(model)));
  live?.modelsChanged(models);
  renderVoiceList();
  await Promise.all([inferVoice.setModels(models), coverVoice.setModels(models)]);
}
listen("[data-refresh-models]", "click", updateModels);
listen("#unloadModel", "click", async () => {
  await api("/api/voice", { method: "DELETE" });
  toast(t("voiceReleased"));
});

/* Inference */
function syncMode() {
  const mode = radio("iMode");
  for (const node of $$("[data-panel=infer] [data-mode]")) {
    node.hidden = node.dataset.mode !== mode;
    if (node.hidden) $$("audio", node).forEach(audio => audio.pause());
  }
}
listen('input[name="iMode"]', "change", syncMode);

function conversion() {
  const useIndex = inferVoice.usesIndex();
  const voice = inferVoice.request();
  return {
    ...voice,
    settings: {
      pitch: number("iPitch"), f0_method: radio("iF0"), resample_sr: number("iResample"),
      rms_mix_rate: number("iRms"), protect: number("iProtect"), index_rate: useIndex ? number("iIndexRate") : 0,
      formant: number("iFormant"), aperiodic: optionalNumber("iAperiodic"), level_guard: checked("iLevelGuard"),
      voice_gate: inferVoice.data?.gate ? number("iGate") : 0, precision: value("iPrecision") || "auto",
    },
    auto_register: inferVoice.data.info.uses_f0 && checked("iAutoRegister"),
  };
}

function syncProfile() {
  $("#iProfile").hidden = $("#iAutoRegister").disabled || !checked("iAutoRegister");
}
listen("#iAutoRegister", "change", syncProfile);
action("iBuildProfile", "iProfileLog", async output => {
  requireValue(inferVoice.data, "chooseVoice");
  const result = await runJob({
    operation: "pitch_profile", model: value("iModel"),
    input: value("iProfileDataset") || null, method: radio("iF0"),
  }, fileList("iProfileFile"), output, "iStopProfile");
  if (result) log(output, `${result.path}\n${t("voicedFrames")} · ${result.profile.voiced_frames}`);
});

function setAudio(audio, media) {
  audio.dataset.downloadName = media.name;
  audio.src = media.url;
  audio.hidden = false;
}

function renderTracks(id, groups) {
  const root = $("#" + id);
  root.replaceChildren();
  root.hidden = groups.length === 0;
  for (const group of groups) {
    const row = document.createElement("div");
    row.className = "result-group";
    if (group.name) {
      const name = document.createElement("span");
      name.className = "result-name";
      name.textContent = group.name;
      row.append(name);
    }
    group.tracks.forEach((track, index) => {
      const suffix = group.suffixes?.[index];
      if (suffix) {
        const label = document.createElement("span");
        label.className = "result-label";
        const key = "stem_" + suffix;
        const text = t(key);
        label.textContent = text === key ? suffix : text;
        row.append(label);
      }
      const audio = document.createElement("audio");
      audio.preload = "none";
      audio.dataset.downloadFormats = "wav,flac,mp3,m4a";
      if (suffix === "cover" || (groups.length === 1 && group.tracks.length === 1)) {
        audio.dataset.wave = "eager";
        audio.dataset.accent = "true";
      }
      setAudio(audio, track);
      row.append(audio);
    });
    root.append(row);
  }
  enhanceAudioPlayers(root);
}

action("runSingle", "sLog", async output => {
  const files = fileList("sAudio");
  requireValue(files[0], "chooseAudio");
  const result = await runJob({ operation: "single", ...conversion() }, files, output, "stopSingle");
  if (result) {
    setAudio($("#sOut"), result);
    $("#sResult").hidden = false;
  }
});
action("runBatch", "bLog", async output => {
  const input = value("bIn");
  const files = input ? [] : fileList("bFiles");
  if (!input) requireValue(files.length, "chooseAudio");
  const result = await runJob({
    operation: "batch", ...conversion(), input,
    output: requireValue(value("bOut"), "chooseOutput"), format: radio("bFmt"),
    recursive: checked("bRecursive"), overwrite: checked("bOverwrite"),
  }, files, output, "stopBatch");
  if (result) renderTracks("bResults", result.map(track => ({ name: track.name, tracks: [track] })));
});

/* Cover */
function syncCover() {
  const vocals = radio("cSource") === "vocals";
  $("#cInstrumental").hidden = !vocals;
  $('[data-drop-for="cFiles"] .drop-title').textContent = t(vocals ? "dropVocals" : "dropSongs");
  $('[data-drop-for="cFiles"] .drop-title').dataset.i18n = vocals ? "dropVocals" : "dropSongs";
}
listen('input[name="cSource"]', "change", syncCover);
action("runCover", "cLog", async output => {
  if (!options.cover) throw new Error(t("coverMissing"));
  const vocals = radio("cSource") === "vocals";
  const input = value("cIn");
  const files = input ? [] : fileList("cFiles");
  if (!input) requireValue(files.length, "chooseAudio");
  const instFile = vocals && !input ? fileList("cInstFile")[0] : null;
  if (instFile) {
    if (files.length !== 1) throw new Error(t("oneVocalForInstrumental"));
    files.push(instFile);
  }
  const result = await runJob({
    operation: "cover", ...coverVoice.request(),
    index_rate: coverVoice.usesIndex() ? number("cIndexRate") : 0,
    pitch: number("cPitch"), f0_method: value("cF0"), vocals_only: vocals, backing: radio("cBacking"), dry: !checked("cReverb"),
    vocal_gain: number("cVocalGain"), aperiodic: optionalNumber("cAperiodic"),
    voice_gate: coverVoice.data?.gate ? number("cGate") : 0, format: radio("cFmt"), device: value("cDevice") || "auto",
    input: input || null, output: value("cOut") || null,
    instrumental: vocals && !instFile ? value("cInstPath") || null : null, instrumental_upload: Boolean(instFile),
  }, files, output, "stopCover");
  if (result) renderTracks("cResults", result);
});

/* Separation */
const presetLabels = {
  "dereverb-less-aggressive-18.8050": "sepDereverb", "dereverb-anvuew-19.1729": "sepDereverbAggressive",
  "vocals-bs-roformer-368": "sepVocals", "vocals-bs-roformer-317": "sepVocalsAggressive",
  "karaoke-mel-roformer-10.1956": "sepMainVocal", "vocals-bs-roformer-ft1-anvuew": "sepVocalsFt1",
  "karaoke-bs-roformer-anvuew": "sepLeadBacking", "dereverb-bs-roformer-anvuew-22.5050": "sepDereverbBs",
};
function syncSeparationModel() {
  if (!options) return;
  const keep = value("pModel") || options.presets.find(preset => preset.installed)?.id;
  fill($("#pModel"), options.presets.map(preset => ({
    value: preset.id,
    label: (presetLabels[preset.id] ? t(presetLabels[preset.id]) : preset.label) + (preset.installed ? "" : " · " + t("downloadsOnUse")),
  })), keep);
  const preset = options.presets.find(item => item.id === value("pModel"));
  $("#pInfo").textContent = preset ? preset.checkpoint + (preset.installed ? "" : " · " + t("downloadsOnUse")) : "";
}
listen("#pModel", "change", syncSeparationModel);
action("runSep", "pLog", async output => {
  const input = value("pIn");
  const files = input ? [] : fileList("pFiles");
  if (!input) requireValue(files.length, "chooseAudio");
  const result = await runJob({
    operation: "separate", input, preset: value("pModel"),
    output: requireValue(value("pVocal"), "chooseOutput"), secondary_output: value("pResidual") || null,
    format: radio("pFmt"), overwrite: checked("sepOverwrite"),
  }, files, output, "stopSep");
  if (result) renderTracks("pResults", result);
});

/* Library */
function syncLibrary() {
  const pair = radio("libKind") === "pretrained";
  $("#libRateWrap").hidden = !pair;
  $("#libPretrainedFields").hidden = !pair;
  $("#libUrlWrap").hidden = pair;
  $("#libFileWrap").hidden = pair;
}
listen('input[name="libKind"]', "change", syncLibrary);
action("runLib", "libLog", async output => {
  const pair = radio("libKind") === "pretrained";
  const name = value("libName");
  const settings = pair ? {
    operation: "pretrained", name: requireValue(name, "chooseImportName"), sample_rate: number("libRate"),
    generator: requireValue(value("libGUrl"), "chooseUrl"), discriminator: requireValue(value("libDUrl"), "chooseUrl"),
  } : { operation: "import", name: name || null, url: value("libUrl") || null };
  const result = await runJob(settings, pair ? [] : fileList("libFile"), output, "stopLib");
  if (result) {
    log(output, result.paths.join("\n"));
    await (pair ? loadPretrained() : updateModels());
  }
});
function renderVoiceList() {
  const models = options.models;
  $("#voiceCount").textContent = String(models.length);
  $("#voiceList").replaceChildren(...(models.length ? models : [t("noVoices")]).map(name => {
    const item = document.createElement("li");
    const label = document.createElement("span");
    label.className = "voice-name";
    label.textContent = name;
    item.append(label);
    if (models.length) item.append(exportLink(name, "icon-btn"));
    return item;
  }));
}

// The voice and its index as one .zip, built by the server on request.
function exportLink(model, className) {
  const link = document.createElement("a");
  link.className = className;
  link.href = "/api/studio/export/" + encodeURIComponent(model);
  link.download = model.replace(/\.pth$/i, "") + ".zip";
  link.title = t("downloadZip");
  link.setAttribute("aria-label", `${t("downloadZip")} · ${model}`);
  link.innerHTML = '<svg class="icon" aria-hidden="true"><use href="#i-download" /></svg>';
  if (className === "btn") link.append(t("downloadZip"));
  return link;
}

/* Training */
function experiment(name) {
  if (!name || name === "." || name === ".." || /[/\\:]/.test(name)) throw new Error(t("chooseExperiment"));
  return options.experiments.replace(/[/\\]+$/, "") + "/" + name;
}

function configTraining() {
  const name = value("expName");
  return {
    name, experiment: experiment(name), version: value("trainVersion"),
    sample_rate: parseInt(radio("sr"), 10) * 1000, f0: radio("ifF0") === "1",
    embedder: value("trainEmbedder") || null,
  };
}

function prepParams() {
  const source = radio("mode");
  return {
    source, input: source === "manifest" ? null : requireValue(value("trainset"), "chooseDataset"),
    workers: number("cpuWorkers"), segment_seconds: number("segmentSeconds"),
    denoise: checked("noiseReduce") ? number("reduceStrength") : 0,
  };
}

function pretrainedPath(id) {
  const selected = value(id);
  return selected === "custom" ? requireValue(value(id + "Path"), "pretrainedPathMissing") : selected || null;
}

function trainingPrecision(id) {
  const precision = value(id);
  return {
    fp16: { fp16: true, fp32: false }[precision] ?? null,
    bf16: precision ? precision === "bf16" : null,
  };
}

const checkpointing = (id) => value(id) ? value(id) === "true" : null;
const list = (text) => text.split(",").map(item => item.trim()).filter(Boolean);

function trainParams() {
  const config = configTraining();
  const timbre = value("timbreMode");
  const pool = timbre === "pool" ? requireValue(value("timbrePool"), "chooseFeaturePool") : null;
  const gpus = list(value("trainGpus")).map(Number);
  if (gpus.some(gpu => !Number.isInteger(gpu) || gpu < 0)) throw new Error(t("badGpus"));
  return {
    ...config, speaker_id: number("spkId"), multi: radio("mode") !== "single",
    epochs: number("totalEpoch"), save_every: number("saveEvery"), batch_size: number("batchSize"),
    latest_only: checked("latestOnly"), save_weights: checked("saveSmall"),
    cache_data_in_gpu: checked("cacheGpu"), tensorboard: checked("writeMetrics"),
    ...trainingPrecision("trainPrecision"),
    gradient_checkpointing: checkpointing("gradientCheckpointing"), gpus,
    timbre_shift_pool: timbre === "off" ? "" : pool,
    timbre_shift_exclude: pool ? list(value("timbreExclude")) : [],
    multiscale_mel: checked("multiScaleMel"),
    ema_decay: checked("emaWeights") ? 0.999 : 0,
    mrd: checked("mrdDisc"),
    unvoiced_f0: value("unvoicedF0") || null,
    stft_loss: Math.max(0, number("stftLoss") || 0),
    phonetic_loss: Math.max(0, number("phoneticLoss") || 0),
    phonetic_encoder: value("phoneticEncoder") || "mhubert147",
    hf_bands: checked("hfBands"), dec_aa: checked("decAa"),
    layer_aug: value("layerAug") ? experiment(value("layerAug")) : null,
    layer_aug_prob: number("layerAugProb"),
    train_only: list(value("trainOnly")),
    fresh: checked("freshStart"),
    pretrained_g: pretrainedPath("preG"), pretrained_d: pretrainedPath("preD"), random_weights: true,
  };
}

function syncTrainingControls() {
  const source = radio("mode");
  $("#multiHelp").hidden = source !== "multi_folder";
  $("#spkWrap").hidden = source !== "single";
  $("#trainsetWrap").hidden = source === "manifest";
  $("[data-drop-for=trainsetZip]").hidden = source === "manifest";
  $("#reduceStrengthWrap").hidden = !checked("noiseReduce");
  const cuda = options?.device?.startsWith("cuda");
  $("#cacheGpu").disabled = !cuda;
  if (!cuda) $("#cacheGpu").checked = false;
  const explicitPool = value("timbreMode") === "pool";
  $("#timbrePoolWrap").hidden = !explicitPool;
  $("#timbreExcludeWrap").hidden = !explicitPool;
  $("#downloadEmbedder").disabled = !value("trainEmbedder");
  $("#layerAugProbWrap").hidden = !value("layerAug");
}

function embedderOptions(automatic) {
  return [
    ...(automatic ? [{ value: "", key: "automatic" }] : []),
    ...options.embedders.map(item => ({
      value: item.name,
      label: (embedderLabels[item.name] || item.name) + (item.installed ? "" : " · " + t(item.name === "w2v_ru" ? "notInstalled" : "downloadsOnUse")),
    })),
  ];
}

let datasetRevision = 0;
listen("#trainsetZip", "change", async () => {
  const revision = ++datasetRevision;
  const files = fileList("trainsetZip");
  if (!files.length) {
    if (value("trainset") === datasetPath) $("#trainset").value = "";
    return;
  }
  const output = $("#preLog");
  const original = value("trainset");
  try {
    const data = await upload("/api/studio/dataset", null, files, pct => {
      if (revision === datasetRevision) log(output, `${t("uploading")} · ${pct}%`);
    });
    if (revision !== datasetRevision) return;
    datasetPath = data.path;
    if (value("trainset") === original) $("#trainset").value = datasetPath;
    log(output, `${t("datasetReady")} ${datasetPath}`);
  } catch (error) {
    if (revision === datasetRevision) explain(error, output);
  }
});

listen('#trainEmbedder, #timbreMode, #layerAug, #trainVersion, #noiseReduce, input[name="mode"]', "change", syncTrainingControls);
listen('#trainVersion, input[name="sr"], input[name="ifF0"]', "change", () => loadPretrained());
action("runPre", "preLog", async output => {
  const config = configTraining();
  await runJob({ operation: "preprocess", experiment: config.experiment, sample_rate: config.sample_rate, ...prepParams() }, [], output, "stopPre");
  await refreshRuns();
});
action("runExt", "extLog", async output => {
  const config = configTraining();
  await runJob({
    operation: "extract", experiment: config.experiment, version: config.version,
    f0_method: config.f0 ? radio("tf0") : null, embedder: config.embedder,
    overwrite: checked("overwriteFeatures"), unvoice_fricatives: checked("unvoiceFricatives"),
  }, [], output, "stopExt");
});
action("downloadEmbedder", "extLog", async output => {
  const result = await runJob({
    operation: "install_embedder", embedder: requireValue(value("trainEmbedder"), "chooseEmbedder"),
  }, [], output, "stopEmbedder");
  if (result) {
    log(output, t("done") + " " + result.path);
    const item = options.embedders.find(entry => entry.name === value("trainEmbedder"));
    if (item) item.installed = true;
    renderEmbedders();
  }
});
action("buildFeaturePool", "trainLog", async output => {
  const result = await runJob({
    operation: "build_feature_pool", spec: requireValue(value("poolSpec"), "choosePoolSpec"),
    output: requireValue(value("poolOutput"), "chooseOutput"), max_minutes: number("poolMinutes"),
    embedder: value("trainEmbedder") || "hubert_base",
  }, [], output, "stopFeaturePool");
  if (result) {
    $("#timbreMode").value = "pool";
    $("#timbrePool").value = result.path;
    syncTrainingControls();
    const speakers = Object.entries(result.frames).map(([speaker, frames]) => speaker + " · " + frames);
    log(output, [t("done") + " " + result.path, ...speakers].join("\n"));
  }
});
action("runTrain", "trainLog", async output => {
  const result = await runJob({ operation: "train", ...trainParams() }, [], output, "stopTrain");
  if (result) await afterTraining(result, output);
});
action("runIndex", "trainLog", async output => {
  await runJob({
    operation: "index", ...configTraining(), mode: radio("mode") === "single" ? "single" : "multi", workers: number("cpuWorkers"),
  }, [], output, "stopIndex");
});
action("runAll", "trainLog", async output => {
  const result = await runJob({
    operation: "train_all", ...trainParams(), ...prepParams(), f0_method: radio("tf0"),
  }, [], output, "stopAll");
  if (result) await afterTraining(result.training, output);
});

let pretrainedRevision = 0;
let loadedPretrainedQuery;
function syncPretrainedPaths() {
  for (const id of ["preG", "preD"]) $("#" + id + "PathWrap").hidden = value(id) !== "custom";
}
listen("#preG, #preD", "change", syncPretrainedPaths);
async function loadPretrained(preferred) {
  const revision = ++pretrainedRevision;
  const query = new URLSearchParams({
    version: value("trainVersion"), sample_rate: parseInt(radio("sr"), 10) * 1000,
    f0: radio("ifF0") === "1",
  });
  const data = await api("/api/studio/pretrained?" + query);
  if (revision !== pretrainedRevision) return;
  const keep = query.toString() === loadedPretrainedQuery;
  loadedPretrainedQuery = query.toString();
  const installed = new Set(data.installed || []);
  const makeOptions = paths => [
    { value: "", key: "noPretrained" },
    ...paths.map(path => ({
      value: path,
      label: path.split(/[\\/]/).slice(-2).join("/") + (installed.has(path) ? "" : " · " + t("downloadsOnTrain")),
    })),
    { value: "custom", key: "pretrainedPath" },
  ];
  fill($("#preG"), makeOptions(data.generator), preferred?.[0] ?? (keep ? value("preG") : data.default_generator) ?? "");
  fill($("#preD"), makeOptions(data.discriminator), preferred?.[1] ?? (keep ? value("preD") : data.default_discriminator) ?? "");
  fill($("#preCatalog"), data.catalog.length ? data.catalog.map(item => ({
    value: item.name, label: item.name + (item.installed ? " · " + t("installed") : ""),
  })) : [{ value: "", key: "catalogEmpty" }]);
  $("#downloadPretrained").disabled = !data.catalog.length;
  syncPretrainedPaths();
  syncNvcPretrained();
}
// The full recipe NVC-Pretrained voices were checked with (preprocess, extract and train).
// Anything not listed there ran with the CLI defaults, set here explicitly.
const NVC_RECIPE = {
  segmentSeconds: 3, noiseReduce: false,
  trainEmbedder: "mhubert147", tf0: "rmvpe", overwriteFeatures: false, unvoiceFricatives: false,
  totalEpoch: 250, saveEvery: 50, batchSize: 8, trainPrecision: "", gradientCheckpointing: "",
  latestOnly: false, saveSmall: true, cacheGpu: false,
  unvoicedF0: "", layerAug: "", layerAugProb: 0.5, hfBands: true, emaWeights: false,
  multiScaleMel: false, mrdDisc: false, decAa: false, freshStart: false,
  stftLoss: 0, phoneticLoss: 0, phoneticEncoder: "mhubert147", trainOnly: "", timbreMode: "off",
};
function applyNvcRecipe() {
  for (const [id, setting] of Object.entries(NVC_RECIPE)) {
    const radios = $$(`input[type=radio][name="${id}"]`);
    if (radios.length) {
      const match = radios.find(input => input.value === setting && !input.disabled);
      if (match) match.checked = true;
      continue;
    }
    const input = $("#" + id);
    if (!input) continue;
    if (input.type === "checkbox") input.checked = setting && !input.disabled;
    else input.value = setting;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }
  syncTrainingControls();
}
// Applied when NVC-Pretrained becomes the selected pretrain, so later edits stay.
let nvcSelected = false;
function syncNvcPretrained() {
  const nvc = /NVC-Pretrained/.test(value("preG"));
  $("#nvcPretrainedHint").hidden = !nvc;
  if (nvc && !nvcSelected) applyNvcRecipe();
  nvcSelected = nvc;
}
listen("#preG", "change", syncNvcPretrained);
action("downloadPretrained", "trainLog", async output => {
  const result = await runJob({
    operation: "pretrained", name: requireValue(value("preCatalog"), "noPretrained"),
    sample_rate: parseInt(radio("sr"), 10) * 1000,
  }, [], output, "stopPretrained");
  if (result) await loadPretrained(result.paths);
});
listen("#refreshPretrained", "click", () => loadPretrained());

async function afterTraining(result, output) {
  log(output, `${t("done")} ${result.model} · ${t("completedEpoch")} ${result.epochs_completed}`);
  await Promise.all([updateModels(), refreshRuns()]);
  $("#trainExport").replaceChildren(exportLink(fileName(result.model), "btn"));
}

async function refreshRuns() {
  experiments = await api("/api/studio/experiments");
  const { runs, resumable, all } = experiments;
  fill($("#metricRun"), runs.length ? runs : [{ value: "", key: "metricsEmpty" }]);
  const current = value("resumeExp");
  fill($("#resumeExp"), resumable.length ? resumable : [{ value: "", key: "resumeEmpty" }]);
  $("#runResume").disabled = !resumable.length;
  fill($("#layerAug"), [{ value: "", key: "off" }, ...all]);
  fill($("#laSource"), all.length ? all : [{ value: "", key: "noExperiments" }]);
  syncTrainingControls();
  if (current !== value("resumeExp") || !$("#resumeInfo").children.length) await showResume();
}
let resumeRevision = 0;
async function showResume() {
  const revision = ++resumeRevision;
  const name = value("resumeExp");
  if (!name) {
    facts($("#resumeInfo"), []);
    return;
  }
  const total = $("#resumeTotal");
  const previous = total.value;
  const info = await api("/api/studio/resume?" + new URLSearchParams({ name }));
  if (revision !== resumeRevision || name !== value("resumeExp")) return;
  total.min = info.epoch + 1;
  if (total.value === previous) total.value = info.epoch + 10;
  facts($("#resumeInfo"), [
    [t("model"), `RVC ${info.version} · ${info.sample_rate / 1000} kHz · ${embedderLabels[info.embedder] || info.embedder}`],
    [t("completedEpoch"), String(info.epoch)],
    [t("trainPrecision"), info.bf16 ? "BF16" : t("automatic")],
    [t("gradientCheckpointing"), t(info.gradient_checkpointing ? "yes" : "no")],
    [t("checkpoints"), fileName(info.generator) + " + " + fileName(info.discriminator)],
  ]);
}
listen("#refreshResume", "click", refreshRuns);
listen("#resumeExp", "change", showResume);
action("runResume", "resumeLog", async output => {
  const result = await runJob({
    operation: "resume", name: requireValue(value("resumeExp"), "resumeEmpty"), epochs: number("resumeTotal"),
    ...trainingPrecision("resumePrecision"), gradient_checkpointing: checkpointing("resumeCheckpointing"),
  }, [], output, "stopResume");
  if (result) {
    await afterTraining(result, output);
    await showResume();
  }
});

/* Layers and effects */
function syncLayers() {
  $("#laFxProfileWrap").hidden = !checked("laFx");
  $("#laExtractFields").hidden = !checked("laExtract");
}
listen("#laFx, #laExtract", "change", syncLayers);
listen("#laSource", "change", () => {
  if (!value("laOutput") && value("laSource")) $("#laOutput").value = value("laSource") + "_layers";
});
action("runLayers", "laLog", async output => {
  const source = requireValue(value("laSource"), "chooseExperiment");
  const target = requireValue(value("laOutput"), "chooseOutputName");
  const result = await runJob({
    operation: "layer_aug", experiment: source, output: target, sample_rate: Number(radio("laSr")),
    seed: number("laSeed"), layers: checked("laLayers"), fx: checked("laFx"),
    fx_profile: checked("laFx") ? value("laFxProfile") || null : null, bleed: value("laBleed") || null,
    extract: checked("laExtract") ? {
      version: value("trainVersion"), f0_method: radio("laF0"), embedder: value("laEmbedder") || null,
      unvoice_fricatives: checked("unvoiceFricatives"),
    } : null,
  }, [], output, "stopLayers");
  if (result) {
    const r = result.report;
    log(output, [result.path, t("layersReport").replace("%clips", r.clips).replace("%doubled", r.doubled)
      .replace("%backed", r.backed).replace("%reverb", r.reverberated).replace("%fx", r.processed).replace("%bleed", r.bled)].join("\n"));
    await refreshRuns();
    $("#layerAug").value = target;
    syncTrainingControls();
  }
});
action("runFx", "fxLog", async output => {
  const result = await runJob({
    operation: "fx_profile", input: requireValue(value("fxIn"), "chooseInput"), output: requireValue(value("fxOut"), "chooseOutput"),
  }, [], output, "stopFx");
  if (result) {
    log(output, `${result.path} · ${t("filesRead").replace("%s", result.files)}`);
    $("#laFx").checked = true;
    $("#laFxProfile").value = value("fxOut");
    syncLayers();
  }
});

/* Metrics */
async function loadMetrics() {
  const name = value("metricRun");
  const data = name ? await api("/api/studio/metrics?" + new URLSearchParams({ name })) : {};
  if (name !== value("metricRun")) return;
  metrics = data;
  const names = Object.keys(metrics);
  fill($("#metricName"), names.length ? names : [{ value: "", key: "metricsEmpty" }], value("metricName") || "loss/g/total");
  renderMetrics();
}
const shortNumber = (v) => {
  const a = Math.abs(v);
  return a !== 0 && (a < 0.001 || a >= 100000) ? v.toExponential(2) : Number(v.toPrecision(4)).toString();
};
function renderMetrics() {
  const svg = $("#metricChart");
  const values = metrics[value("metricName")] || [];
  svg.replaceChildren();
  svg.toggleAttribute("hidden", values.length === 0);
  $("#metricFigures").hidden = values.length === 0;
  log($("#tbStatus"), values.length ? "" : t("metricsEmpty"));
  if (!values.length) return;
  const ys = values.map(([, v]) => v);
  $("#mLast").textContent = shortNumber(ys.at(-1));
  $("#mBest").textContent = shortNumber(Math.min(...ys));
  $("#mSteps").textContent = String(values.at(-1)[0]);
  const width = Math.max(320, Math.round(svg.getBoundingClientRect().width || 960));
  const height = 320;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("aria-label", value("metricName"));
  const left = 64, right = width - 16, top = 12, bottom = height - 28;
  const half = Math.max(0, Math.round(values.length / 40));
  const smooth = ys.map((_, i) => {
    const from = Math.max(0, i - half), to = Math.min(ys.length, i + half + 1);
    let sum = 0;
    for (let j = from; j < to; j++) sum += ys[j];
    return sum / (to - from);
  });
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const first = values[0][0], last = values.at(-1)[0];
  const x = step => left + (right - left) * (step - first) / Math.max(1, last - first);
  const y = v => hi === lo ? (top + bottom) / 2 : bottom - (bottom - top) * (v - lo) / (hi - lo);
  const node = (type, attrs, text) => {
    const el = document.createElementNS("http://www.w3.org/2000/svg", type);
    for (const [key, val] of Object.entries(attrs)) el.setAttribute(key, val);
    if (text !== undefined) el.textContent = text;
    svg.append(el);
    return el;
  };
  const defs = node("defs", {});
  const fade = document.createElementNS("http://www.w3.org/2000/svg", "linearGradient");
  for (const [key, val] of Object.entries({ id: "chartFade", x1: 0, y1: 0, x2: 0, y2: 1 })) fade.setAttribute(key, val);
  fade.innerHTML = '<stop offset="0" class="fade-top" /><stop offset="1" class="fade-bottom" />';
  defs.append(fade);
  for (let i = 0; i <= 4; i++) {
    const gy = top + i * (bottom - top) / 4;
    node("line", { class: "grid-line", x1: left, x2: right, y1: gy, y2: gy });
    node("text", { class: "axis", x: left - 10, y: gy + 4, "text-anchor": "end" }, shortNumber(hi - (hi - lo) * i / 4));
  }
  node("text", { class: "axis", x: left, y: height - 6 }, String(first));
  node("text", { class: "axis", x: right, y: height - 6, "text-anchor": "end" }, String(last));
  const raw = values.map(([step, v]) => `${x(step).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const line = values.map(([step], i) => `${x(step).toFixed(1)},${y(smooth[i]).toFixed(1)}`);
  node("polygon", { class: "area", points: `${x(first)},${bottom} ${line.join(" ")} ${x(last)},${bottom}` });
  if (values.length > 12) node("polyline", { class: "raw", points: raw });
  node("polyline", { class: "line", points: line.join(" ") });
  node("circle", { class: "dot", cx: x(last), cy: y(smooth.at(-1)), r: 4.5 });
}
listen("#metricRun", "change", loadMetrics);
listen("#metricName", "change", renderMetrics);
listen("#tbReload", "click", async () => {
  await refreshRuns();
  await loadMetrics();
});
new ResizeObserver(() => { if (currentPanel === "metrics" && Object.keys(metrics).length) renderMetrics(); }).observe($("#metricChart").parentElement);

/* Speakers */
const speakers = [newSpeaker(0), newSpeaker(1)];
const pageSize = 10;
let speakerPage = 0;
function newSpeaker(id) { return { path: "", speaker_name: "", speaker_id: String(id), repeat: "1" }; }
function renderSpeakers() {
  const pages = Math.ceil(speakers.length / pageSize);
  speakerPage = Math.max(0, Math.min(speakerPage, pages - 1));
  $("#hPage").textContent = `${speakerPage + 1} / ${pages} · ${speakers.length}`;
  $("#hPrev").disabled = speakerPage === 0;
  $("#hNext").disabled = speakerPage === pages - 1;
  $("#hDel").disabled = speakers.length === 1;
  $("#hAdd").disabled = speakers.length >= 110;
  $("#hRows").replaceChildren(...speakers.slice(speakerPage * pageSize, speakerPage * pageSize + pageSize).map(row => {
    const wrap = document.createElement("div");
    wrap.className = "row-item";
    for (const [key, label, mono] of [["path", "path"], ["speaker_name", "name"], ["speaker_id", "id", true], ["repeat", "repeat", true]]) {
      const input = document.createElement("input");
      input.type = "text";
      input.spellcheck = false;
      if (mono) input.classList.add("mono");
      input.setAttribute("aria-label", t(label));
      input.value = row[key];
      input.addEventListener("input", () => { row[key] = input.value; });
      wrap.append(input);
    }
    return wrap;
  }));
}
listen("#hPrev", "click", () => { speakerPage--; renderSpeakers(); });
listen("#hNext", "click", () => { speakerPage++; renderSpeakers(); });
listen("#hAdd", "click", () => {
  speakers.push(newSpeaker(speakers.length));
  speakerPage = Math.floor((speakers.length - 1) / pageSize);
  renderSpeakers();
});
listen("#hDel", "click", () => { speakers.pop(); renderSpeakers(); });
action("hSubmit", "hStatus", async output => {
  const rows = speakers.filter(row => row.path.trim() || row.speaker_name.trim());
  requireValue(rows.length, "fillSpeaker");
  const result = await send("/api/manifest", { experiment: experiment(value("hExp")), rows });
  const invalid = result.invalid_rows.length ? `\n${t("invalidRows")} · ${result.invalid_rows.join(", ")}` : "";
  log(output, result.path + invalid);
});

/* Checkpoints */
function checkpointName(id) {
  const name = requireValue(value(id), "chooseOutputName");
  return /\.pth$/i.test(name) ? name : name + ".pth";
}
async function editModel(output, settings) {
  const result = await send("/api/model", { ...settings, overwrite: false });
  log(output, result.path);
  await updateModels();
}
action("runMerge", "mLog", output => editModel(output, {
  operation: "merge", first: requireValue(value("mA"), "chooseVoice"), second: requireValue(value("mB"), "chooseVoice"),
  output: checkpointName("mName"), description: value("mInfo"), alpha: number("mAlpha"),
}));
action("runMod", "modLog", output => editModel(output, {
  operation: "describe", model: requireValue(value("modPath"), "chooseVoice"), output: checkpointName("modName"), description: value("modInfo"),
}));
action("runShow", "insInfo", async output => {
  const path = requireValue(value("insPath"), "chooseVoice");
  const info = await api("/api/studio/inspect?" + new URLSearchParams({ path }));
  const speakers = Object.entries(info.speakers || {}).map(([id, name]) => `${name} · ${id}`).join(", ");
  facts(output, [
    [t("model"), `RVC ${info.version}${info.sample_rate ? " · " + info.sample_rate / 1000 + " kHz" : ""}`],
    [t("contentEmbedder"), embedderLabels[info.embedder] || info.embedder],
    [t("pitchGuide"), t(info.uses_f0 ? "yes" : "no")],
    [t("unvoicedF0"), String(info.f0_unvoiced ?? "")],
    [t("speakers"), speakers || String(info.speaker_count)],
    ...(info.dec_aa ? [[t("decAa"), t("yes")]] : []),
    ...(info.info ? [[t("modelInfo"), info.info]] : []),
  ]);
});
action("runExtCkpt", "eLog", output => editModel(output, {
  operation: "extract", checkpoint: requireValue(value("ePath"), "chooseVoice"), output: checkpointName("eName"),
  version: value("exportVersion"), sample_rate: parseInt(radio("eSr"), 10) * 1000, f0: radio("eF0") === "1",
  description: value("eInfo"),
}));

function renderEmbedders() {
  fill($("#trainEmbedder"), embedderOptions(true));
  fill($("#laEmbedder"), embedderOptions(true));
  fill($("#phoneticEncoder"), embedderOptions(false), value("phoneticEncoder") || "mhubert147");
  syncTrainingControls();
}

async function start() {
  bindRanges();
  bindDrops();
  enhanceAudioPlayers();
  localize();
  options = await api("/api/studio/options");
  $("#deviceName").textContent = options.device;
  $("#cpuWorkers").value = options.workers;
  syncRange($("#cpuWorkers"));
  for (const name of ["iF0", "tf0", "laF0"]) {
    for (const input of $$(`input[name="${name}"]`)) {
      input.disabled = !options.pitch[input.value];
      input.checked = input.value === (options.pitch.rmvpe ? "rmvpe" : "pm");
    }
  }
  const allPitch = Object.values(options.pitch).every(Boolean);
  for (const option of $("#cF0").options) option.disabled = !options.pitch[option.value];
  $("#cF0").value = options.pitch.rmvpe ? "rmvpe" : "pm";
  $$(".pitch-availability").forEach(node => { node.hidden = allPitch; });
  for (const id of ["runIndex", "runAll"]) $("#" + id).hidden = !options.index;
  $("#coverMissing").hidden = options.cover;
  $("#runCover").disabled = !options.cover;
  if (options.realtime) {
    $('[data-tab="realtime"]').hidden = false;
    live = realtime(options);
  }
  renderEmbedders();
  syncLibrary();
  syncMode();
  syncCover();
  syncLayers();
  syncSeparationModel();
  showPanel(new URLSearchParams(location.search).get("panel"));
  await Promise.all([updateModels(), refreshRuns(), watchForeignJob()]);
}
start().catch(error => {
  $("#startup").hidden = false;
  explain(error, $("#startup"));
});
