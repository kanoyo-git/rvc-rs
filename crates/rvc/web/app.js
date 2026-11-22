import {
  $, $$, api, send, upload, value, checked, radio, t, log, fill, facts, toast, clock,
  bindRanges, syncRange, bindDrops, refreshDrop, enhanceAudioPlayers, repaintPlayers,
} from "/studio/ui.js";
import { savedFiles } from "/studio/files.js";
import { studioJobs, rememberOutput } from "/studio/jobs.js";
import { realtime } from "/studio/realtime.js";
import * as session from "/studio/session.js";

let options;
let live;
let metrics = {};
let datasetPath;
let datasetUploading = false;
let datasetPaths;
let experiments = { runs: [], resumable: [], all: [] };
const fileList = (id) => [...document.getElementById(id).files];
const fileName = (path) => String(path).split(/[\\/]/).pop();
const number = (id) => {
  const input = document.getElementById(id);
  if (input.matches(".range-row input, #resumeTotal") && !input.reportValidity()) {
    input.focus();
    throw new Error(`${input.labels[0].textContent.trim()} · ${input.validationMessage}`);
  }
  return Number(input.value);
};
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

function action(id, output, run, available = () => true) {
  listen("#" + id, "click", async event => {
    const button = event.currentTarget;
    const target = $("#" + output);
    if (jobs.busy() && target.dataset.jobId && $(".btn[data-job-lock]")) { toast(t("busyTask"), true); return; }
    const previousJob = target.dataset.jobId;
    button.disabled = true;
    try {
      log(target, "");
      await run(target);
    } catch (error) {
      explain(error, target);
    } finally {
      if (target.dataset.jobId === previousJob) rememberOutput(target);
      button.disabled = !available() || Boolean(button.dataset.jobLock);
    }
  });
}

/* Language and theme */
const prefs = {
  theme: session.preference("rvc-studio-theme", "dark") === "light" ? "light" : "dark",
  lang: session.preference("rvc-studio-lang", navigator.language.startsWith("ru") ? "ru" : "en"),
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
  renderEmbedders();
  if (pretrainedChoices && loadedPretrainedQuery === pretrainedQuery()) renderPretrained(pretrainedChoices, loadedPretrainedQuery);
  tally.render();
  if (Object.keys(metrics).length) renderMetrics();
  repaintPlayers();
  if (live?.active) live.refresh();
}
const capital = (name) => ({ infer: "Infer", cover: "Cover", realtime: "Realtime", separate: "Separate", train: "Train",
  layers: "Layers", metrics: "Metrics", speakers: "Speakers", library: "Library", ckpt: "Ckpt", faq: "Faq" })[name] || "Infer";
listen("#langSelect", "change", () => {
  prefs.lang = value("langSelect");
  session.storePreference("rvc-studio-lang", prefs.lang);
  localize();
});
listen("#themeToggle", "click", () => {
  prefs.theme = prefs.theme === "light" ? "dark" : "light";
  session.storePreference("rvc-studio-theme", prefs.theme);
  localize();
});

/* Navigation */
let currentPanel = "infer";
function addressPanel() {
  const query = new URLSearchParams(location.search);
  return query.get("panel") || query.get("tab");
}
const panelAliases = { convert: "infer", training: "train", separation: "separate", models: "ckpt", tensorboard: "metrics", helper: "speakers" };
function showPanel(name, updateLocation = false) {
  name = panelAliases[name] || name;
  if (!$$(".nav-item").some(tab => tab.dataset.tab === name && !tab.hidden)) name = "infer";
  currentPanel = name;
  session.remember("panel", name);
  for (const tab of $$(".nav-item")) {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("is-active", active);
    if (active) tab.setAttribute("aria-current", "page");
    else tab.removeAttribute("aria-current");
  }
  for (const panel of $$(".panel")) panel.classList.toggle("is-active", panel.dataset.panel === name);
  for (const audio of $$("audio")) if (!audio.closest(".panel.is-active")) audio.pause();
  $("#workspace").scrollTop = Number(session.saved("scroll:" + name)) || 0;
  document.title = "NVC Studio · " + t("tab" + capital(name));
  if (updateLocation) {
    const url = new URL(location.href);
    const key = url.searchParams.has("tab") && !url.searchParams.has("panel") ? "tab" : "panel";
    url.searchParams.delete("panel");
    url.searchParams.delete("tab");
    if (name !== "infer") url.searchParams.set(key, name);
    if (url.href !== location.href) history.pushState({ studioPanel: name }, "", url);
  } else if (history.state?.studioPanel !== name) {
    history.replaceState({ ...history.state, studioPanel: name }, "");
  }
  live?.activate(name === "realtime");
  requestAnimationFrame(repaintPlayers);
  if (!options) return;
  if (name === "metrics") return refreshRuns().then(loadMetrics).catch(error => explain(error, $("#tbStatus")));
  if (name === "train") return Promise.all([loadPretrained(), refreshRuns()]).catch(error => explain(error, $("#trainLog")));
  if (name === "layers") return refreshRuns().catch(error => explain(error, $("#laLog")));
}
listen(".nav-item", "click", event => showPanel(event.currentTarget.dataset.tab, true));
window.addEventListener("popstate", () => showPanel(addressPanel() || history.state?.studioPanel || "infer"));
$("#workspace").addEventListener("scroll", event => session.remember("scroll:" + currentPanel, event.target.scrollTop), { passive: true });

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

function displayJobLine(line) {
  const match = /^Epoch (\d+\/\d+) completed in (\d+(?:\.\d+)?) s(?: · ETA (\d+ (?:d|h|min|s)(?: \d+ (?:d|h|min|s))*))?$/.exec(line);
  if (!match) return line;
  const seconds = Number(match[2]).toLocaleString(document.documentElement.lang, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const duration = t("epochDuration").replace("{epoch}", match[1]).replace("{seconds}", seconds);
  if (!match[3]) return duration;
  const eta = match[3].replace(/\b(d|h|min|s)\b/g, unit => t("etaUnit_" + unit));
  return duration + " · " + t("epochEta").replace("{time}", eta);
}

const jobs = studioJobs({ tally, progressBar, displayLine: displayJobLine, completed: presentResult, explain });
const runJob = jobs.run;

/* Voices */
const embedderLabels = { hubert_base: "ContentVec (RVC)", contentvec: "ContentVec (Applio)", spin: "Spin", "spin-v2": "Spin-v2", mhubert147: "mHuBERT-147", w2v_ru: "wav2vec 2.0 ru" };

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
      const path = "/api/studio/voice/" + encodeURIComponent(model) + "?";
      let result = await api(path + query);
      if (current !== revision) return;
      const storedSpeaker = session.waiting("field:" + prefix + "Speaker");
      if (!speakerChanged && storedSpeaker && storedSpeaker !== String(result.speaker) && Object.hasOwn(result.info.speakers, storedSpeaker)) {
        query.set("speaker", storedSpeaker);
        result = await api(path + query);
        if (current !== revision) return;
      }
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
  el("Model").addEventListener("change", () => {
    for (const field of ["Speaker", "Index"]) session.remember("field:" + prefix + field, null);
    load();
  });
  el("Speaker").addEventListener("change", () => load(true));
  el("Index").addEventListener("change", sync);
  return {
    load,
    get data() { return data; },
    setModels(models) {
      fill(el("Model"), models.length ? models : [{ value: "", key: "noVoices" }]);
      if (session.saved("field:" + prefix + "Model") !== el("Model").value) {
        for (const field of ["Speaker", "Index"]) session.take("field:" + prefix + field);
      }
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
  await runJob({
    operation: "pitch_profile", model: value("iModel"),
    input: value("iProfileDataset") || null, method: radio("iF0"),
  }, fileList("iProfileFile"), output);
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
  await runJob({ operation: "single", ...conversion() }, files, output);
});
action("runBatch", "bLog", async output => {
  const input = value("bIn");
  const files = input ? [] : fileList("bFiles");
  if (!input) requireValue(files.length, "chooseAudio");
  await runJob({
    operation: "batch", ...conversion(), input,
    output: requireValue(value("bOut"), "chooseOutput"), format: radio("bFmt"),
    recursive: checked("bRecursive"), overwrite: checked("bOverwrite"),
  }, files, output);
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
  await runJob({
    operation: "cover", ...coverVoice.request(),
    index_rate: coverVoice.usesIndex() ? number("cIndexRate") : 0,
    pitch: number("cPitch"), f0_method: value("cF0"), vocals_only: vocals, backing: radio("cBacking"), dry: !checked("cReverb"),
    vocal_gain: number("cVocalGain"), aperiodic: optionalNumber("cAperiodic"),
    voice_gate: coverVoice.data?.gate ? number("cGate") : 0, format: radio("cFmt"), device: value("cDevice") || "auto",
    input: input || null, output: value("cOut") || null,
    instrumental: vocals && !instFile ? value("cInstPath") || null : null, instrumental_upload: Boolean(instFile),
  }, files, output);
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
  await runJob({
    operation: "separate", input, preset: value("pModel"),
    output: requireValue(value("pVocal"), "chooseOutput"), secondary_output: value("pResidual") || null,
    format: radio("pFmt"), overwrite: checked("sepOverwrite"),
  }, files, output);
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
    generator: value("libGUrl").trim() || null, discriminator: value("libDUrl").trim() || null,
  } : { operation: "import", name: name || null, url: value("libUrl") || null };
  if (pair && !settings.generator && !settings.discriminator) throw new Error(t("chooseUrl"));
  await runJob(settings, pair ? [] : fileList("libFile"), output);
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
  if (source !== "manifest" && datasetUploading) throw new Error(t("waitForDataset"));
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
  if (loadedPretrainedQuery !== pretrainedQuery()) throw new Error(t("loadingPretrained"));
  const config = configTraining();
  const timbre = value("timbreMode");
  const pool = timbre === "pool" ? requireValue(value("timbrePool"), "chooseFeaturePool") : null;
  const phoneticLoss = Math.max(0, number("phoneticLoss") || 0);
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
    phonetic_loss: phoneticLoss,
    phonetic_encoder: phoneticLoss > 0 ? value("phoneticEncoder") || null : null,
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
  if (datasetPaths && source !== "manifest" && value("trainset") === datasetPath) {
    datasetPath = datasetPaths[source];
    $("#trainset").value = datasetPath;
    session.remember("field:trainset", datasetPath);
  }
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
  const phonetic = $("#phoneticEncoder");
  const enabled = number("phoneticLoss") > 0;
  phonetic.disabled = !enabled;
  phonetic.hidden = !enabled;
  $("#phoneticEncoderOff").hidden = enabled;
  if (phonetic.options.length) {
    const primary = value("trainEmbedder");
    phonetic.options[0].text = primary ? `${t("sameContentEmbedder")} · ${embedderLabels[primary] || primary}` : t("sameContentEmbedder");
  }
}

function embedderOptions() {
  // A catalog selection can arrive before the server replies with local install status.
  const choices = options?.embedders ?? Object.keys(embedderLabels).map(name => ({ name }));
  return choices.map(item => ({
    value: item.name,
    label: (embedderLabels[item.name] || item.name) + (item.installed === false ? " · " + t(item.name === "w2v_ru" ? "notInstalled" : "downloadsOnUse") : ""),
  }));
}

let datasetRevision = 0;
listen("#trainsetZip", "change", async () => {
  const revision = ++datasetRevision;
  const files = fileList("trainsetZip");
  if (value("trainset") === datasetPath) {
    $("#trainset").value = "";
    session.remember("field:trainset", "");
  }
  session.remember("dataset", null);
  datasetPath = undefined;
  datasetPaths = undefined;
  datasetUploading = false;
  if (!files.length) return;
  const output = $("#preLog");
  const original = value("trainset");
  const archive = files.length === 1 && /\.zip$/i.test(files[0].name);
  const mode = radio("mode");
  try {
    if (!archive && files.some(file => /\.zip$/i.test(file.name))) throw new Error(t("chooseDatasetUpload"));
    datasetUploading = true;
    const data = await upload("/api/studio/dataset", { mode }, files, pct => {
      if (revision === datasetRevision) log(output, `${t("uploading")} · ${pct}%`);
    });
    if (revision !== datasetRevision) return;
    datasetPaths = data;
    datasetPath = data[radio("mode")] || data.path;
    if (value("trainset") === original) {
      $("#trainset").value = datasetPath;
      session.remember("field:trainset", datasetPath);
      session.remember("dataset", session.safeFields(data));
    }
    log(output, `${t("datasetReady")} ${datasetPath}`);
  } catch (error) {
    if (revision === datasetRevision) explain(error, output);
  } finally {
    if (revision === datasetRevision) { datasetUploading = false; rememberOutput(output); }
  }
});

listen("#phoneticLoss", "input", syncTrainingControls);
listen('#trainEmbedder, #timbreMode, #layerAug, #trainVersion, #noiseReduce, input[name="mode"]', "change", syncTrainingControls);
listen('#trainVersion, input[name="sr"], input[name="ifF0"]', "change", () => loadPretrained());
action("runPre", "preLog", async output => {
  const config = configTraining();
  await runJob({ operation: "preprocess", experiment: config.experiment, sample_rate: config.sample_rate, ...prepParams() }, [], output);
});
action("runExt", "extLog", async output => {
  const config = configTraining();
  await runJob({
    operation: "extract", experiment: config.experiment, version: config.version,
    f0_method: config.f0 ? radio("tf0") : null, embedder: config.embedder,
    overwrite: checked("overwriteFeatures"), unvoice_fricatives: checked("unvoiceFricatives"),
  }, [], output);
});
action("downloadEmbedder", "extLog", async output => {
  await runJob({
    operation: "install_embedder", embedder: requireValue(value("trainEmbedder"), "chooseEmbedder"),
  }, [], output);
});
action("buildFeaturePool", "trainLog", async output => {
  await runJob({
    operation: "build_feature_pool", spec: requireValue(value("poolSpec"), "choosePoolSpec"),
    output: requireValue(value("poolOutput"), "chooseOutput"), max_minutes: number("poolMinutes"),
    embedder: value("trainEmbedder") || "hubert_base",
  }, [], output);
});
action("runTrain", "trainLog", async output => {
  const settings = trainParams();
  await runJob({ operation: "train", ...settings }, [], output);
});
action("runIndex", "trainLog", async output => {
  await runJob({
    operation: "index", ...configTraining(), mode: radio("mode") === "single" ? "single" : "multi", workers: number("cpuWorkers"),
  }, [], output);
});
action("runAll", "trainLog", async output => {
  const settings = trainParams();
  await runJob({
    operation: "train_all", ...settings, ...prepParams(), f0_method: radio("tf0"),
  }, [], output);
});

const CONTENTVEC_NVC_DIRECTORY = "NVC-Pretrained-Rage-ContentVec";
function pretrainedFolderLabel(folder) {
  if (folder === CONTENTVEC_NVC_DIRECTORY) return "NVC-Pretrained";
  if (folder === "NVC-Pretrained") return t("legacyNvcPretrained");
  return folder;
}

let pretrainedRevision = 0;
let pretrainedChoices;
let loadedPretrainedQuery;
let pretrainedSets = [];
const selectedSet = () => pretrainedSets.find(item => item.id === value("preCatalog"));
const pretrainedQuery = () => new URLSearchParams({
  version: value("trainVersion"), sample_rate: parseInt(radio("sr"), 10) * 1000, f0: radio("ifF0") === "1",
}).toString();

function syncPretrained() {
  for (const id of ["preG", "preD"]) $("#" + id + "PathWrap").hidden = value(id) !== "custom";
  const pair = pretrainedSets.find(item => item.paths[0] === value("preG") && item.paths[1] === value("preD"));
  $("#preCatalog").value = pair?.id ?? (value("preG") || value("preD") ? "manual" : "none");
  $("#downloadPretrained").disabled = !pair?.downloadable || pair.installed;
  syncNvcPretrained();
}

async function loadPretrained() {
  const revision = ++pretrainedRevision;
  const query = pretrainedQuery();
  const data = await api("/api/studio/pretrained?" + query);
  if (revision === pretrainedRevision) renderPretrained(data, query);
}

function renderPretrained(data, query) {
  pretrainedChoices = data;
  const old = selectedSet();
  const sameQuery = query === loadedPretrainedQuery;
  const installed = new Set(data.installed);
  pretrainedSets = data.catalog.map(item => ({
    ...item, id: item.name, downloadable: true,
  }));
  const discriminators = new Set(data.discriminator);
  for (const g of data.generator) {
    const d = g.replace(/([\\/])(f0)?G(\d+k\.pth)$/i, "$1$2D$3");
    if (!discriminators.has(d) || pretrainedSets.some(item => item.paths[0] === g)) continue;
    const folder = g.split(/[\\/]/).at(-2);
    const label = g === data.default_generator ? "RVC " + value("trainVersion")
      : pretrainedFolderLabel(folder);
    pretrainedSets.push({ id: "local:" + g, name: label, paths: [g, d],
      installed: installed.has(g) && installed.has(d), downloadable: false });
  }
  const selectPaths = sameQuery ? [value("preG"), value("preD")]
    : (old?.downloadable && pretrainedSets.find(item => item.id === old.id)?.paths)
      || [data.default_generator, data.default_discriminator];
  const options = paths => [
    { value: "", key: "noPretrained" },
    ...paths.map(path => {
      const [folder, file] = path.split(/[\\/]/).slice(-2);
      return { value: path, label: `${pretrainedFolderLabel(folder)}/${file}` + (installed.has(path) ? "" : " · " + t("downloadsOnTrain")) };
    }),
    { value: "custom", key: "pretrainedPath" },
  ];
  fill($("#preG"), options(data.generator), selectPaths[0] || "");
  fill($("#preD"), options(data.discriminator), selectPaths[1] || "");
  fill($("#preCatalog"), [
    ...pretrainedSets.map(item => ({
      value: item.id,
      label: (item.name === CONTENTVEC_NVC_DIRECTORY ? "NVC-Pretrained" : item.name) + (item.installed ? " · " + t("installed") : ""),
    })),
    { value: "none", key: "noPretrained" }, { value: "manual", key: "manualPretrained" },
  ]);
  loadedPretrainedQuery = query;
  syncPretrained();
}
listen("#preCatalog", "change", () => {
  const pair = selectedSet();
  const fallback = value("preCatalog") === "manual" ? "custom" : "";
  $("#preG").value = pair?.paths[0] ?? fallback;
  $("#preD").value = pair?.paths[1] ?? fallback;
  syncPretrained();
});
listen("#preG", "change", () => {
  if (selectedSet()) {
    const pair = pretrainedSets.find(item => item.paths[0] === value("preG"));
    if (pair) $("#preD").value = pair.paths[1];
  }
  syncPretrained();
});
listen("#preGPath, #preD, #preDPath", "change", syncPretrained);
// The full recipe NVC-Pretrained voices were checked with (preprocess, extract and train).
// Anything not listed there ran with the CLI defaults, set here explicitly.
const NVC_RECIPE = {
  segmentSeconds: 3, noiseReduce: false,
  trainEmbedder: "mhubert147", tf0: "rmvpe", overwriteFeatures: false, unvoiceFricatives: false,
  totalEpoch: 250, saveEvery: 50, batchSize: 8, trainPrecision: "", gradientCheckpointing: "",
  latestOnly: false, saveSmall: true, cacheGpu: false,
  unvoicedF0: "", layerAug: "", layerAugProb: 0.5, hfBands: true, emaWeights: false,
  multiScaleMel: false, mrdDisc: false, decAa: false, freshStart: false,
  stftLoss: 0, phoneticLoss: 0, trainOnly: "", timbreMode: "off",
};
const CONTENTVEC_RECIPE = { ...NVC_RECIPE, trainEmbedder: "hubert_base", hfBands: false, unvoicedF0: "zero", totalEpoch: 60, saveEvery: 60 };
const NVC_PRETRAINS = new Map([
  ["NVC-Pretrained", { settings: NVC_RECIPE, hint: "nvcPretrainedHint" }],
  [CONTENTVEC_NVC_DIRECTORY, { settings: CONTENTVEC_RECIPE, hint: "contentvecNvcPretrainedHint" }],
]);

function readRecipeSettings() {
  return Object.fromEntries(Object.keys(NVC_RECIPE).map(id => {
    const input = $("#" + id);
    return [id, input ? (input.type === "checkbox" ? input.checked : input.value) : radio(id)];
  }));
}

function applyNvcRecipe(settings) {
  for (const [id, setting] of Object.entries(settings)) {
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
let activeRecipe;
let beforeRecipe;
let appliedRecipe;
function syncNvcPretrained() {
  const path = value("preG") === "custom" ? value("preGPath") : value("preG");
  const folder = path.split(/[\\/]/).at(-2);
  const recipe = NVC_PRETRAINS.get(folder);
  const hint = $("#nvcPretrainedHint");
  hint.hidden = !recipe;
  if (recipe) {
    hint.dataset.i18n = recipe.hint;
    hint.textContent = t(recipe.hint);
  }
  if (recipe === activeRecipe) return;
  const remembered = session.take("recipe");
  if (recipe && remembered?.folder === folder && remembered.before && remembered.applied) {
    const fields = Object.keys(NVC_RECIPE);
    beforeRecipe = Object.fromEntries(fields.filter(id => Object.hasOwn(remembered.before, id)).map(id => [id, remembered.before[id]]));
    appliedRecipe = remembered.applied;
    activeRecipe = recipe;
    return;
  }
  if (activeRecipe) {
    const current = readRecipeSettings();
    const restore = Object.entries(beforeRecipe).filter(([id]) => current[id] === appliedRecipe[id]);
    applyNvcRecipe(Object.fromEntries(restore));
  }
  if (recipe) {
    beforeRecipe = readRecipeSettings();
    applyNvcRecipe(recipe.settings);
    appliedRecipe = readRecipeSettings();
  }
  activeRecipe = recipe;
}
listen("#preCatalog, #preG, #preD, #preGPath, #preDPath", "change", () => {
  for (const id of ["preG", "preD", "preCatalog"]) session.remember("field:" + id, value(id));
  for (const [id, v] of Object.entries(session.safeFields(readRecipeSettings()))) {
    session.remember(($("#" + id) ? "field:" : "radio:") + id, v);
  }
  const generator = value("preG") === "custom" ? value("preGPath") : value("preG");
  session.remember("recipe", activeRecipe ? { folder: generator.split(/[\\/]/).at(-2),
    before: session.safeFields(beforeRecipe), applied: session.safeFields(appliedRecipe) } : null);
});
listen("#downloadPretrained", "click", async () => {
  const output = $("#trainLog");
  const pair = selectedSet();
  if (!pair?.downloadable) return;
  $("#downloadPretrained").disabled = true;
  log(output, "");
  try {
    await runJob({ operation: "pretrained", name: pair.name, sample_rate: parseInt(radio("sr"), 10) * 1000 },
      [], output);
  } catch (error) {
    explain(error, output);
  } finally {
    await loadPretrained();
  }
});
listen("#refreshPretrained", "click", () => loadPretrained());

async function presentResult(job, output, refresh, keepNote) {
  const { result, operation, studio } = job;
  const append = text => { if (!keepNote) output.append("\n" + text); };
  switch (operation) {
    case "train": case "train_all": case "resume":
      return afterTraining(operation === "train_all" ? result.training : result, output, job, refresh, keepNote);
    case "single":
      setAudio($("#sOut"), result);
      $("#sResult").hidden = false;
      break;
    case "batch":
      renderTracks("bResults", result.map(track => ({ name: track.name, tracks: [track] })));
      break;
    case "cover": case "separate":
      renderTracks(operation === "cover" ? "cResults" : "pResults", result);
      break;
    case "pitch_profile":
      append(`${result.path}\n${t("voicedFrames")} · ${result.profile.voiced_frames}`);
      break;
    case "pretrained": case "import":
      if (studio.stop === "stopLib") append(result.paths.join("\n"));
      if (refresh) await (operation === "pretrained" ? loadPretrained() : updateModels());
      break;
    case "preprocess":
      if (refresh) await refreshRuns();
      break;
    case "install_embedder": {
      append(result.path);
      const item = options.embedders.find(entry => entry.name === studio.detail);
      if (item) item.installed = true;
      if (refresh) renderEmbedders();
      break;
    }
    case "build_feature_pool":
      append([result.path, ...Object.entries(result.frames).map(([speaker, frames]) => speaker + " · " + frames)].join("\n"));
      if (refresh) {
        $("#timbreMode").value = "pool";
        $("#timbrePool").value = result.path;
        syncTrainingControls();
      }
      break;
    case "layer_aug": {
      const r = result.report;
      append([result.path, t("layersReport").replace("%clips", r.clips).replace("%doubled", r.doubled)
        .replace("%backed", r.backed).replace("%reverb", r.reverberated).replace("%fx", r.processed).replace("%bleed", r.bled)].join("\n"));
      if (refresh) {
        await refreshRuns();
        $("#layerAug").value = studio.detail;
        syncTrainingControls();
      }
      break;
    }
    case "fx_profile":
      append(`${result.path} · ${t("filesRead").replace("%s", result.files)}`);
      if (refresh) {
        $("#laFx").checked = true;
        $("#laFxProfile").value = studio.detail;
        syncLayers();
      }
      break;
  }
}

let exportedTrainingStarted = 0;
async function afterTraining(result, output, job, refresh, keepNote) {
  const experimentName = job.studio.name;
  const link = document.createElement("a");
  link.href = "/api/studio/train-log/" + encodeURIComponent(experimentName);
  link.download = experimentName + "-train.log";
  link.dataset.i18n = "downloadTrainingLog";
  link.textContent = t("downloadTrainingLog");
  if (!keepNote) output.append(` ${result.model} · ${t("completedEpoch")} ${result.epochs_completed}\n`, link);
  if (job.started_at >= exportedTrainingStarted) {
    exportedTrainingStarted = job.started_at;
    $("#trainExport").replaceChildren(exportLink(fileName(result.model), "btn"));
  }
  if (refresh) await Promise.all([updateModels(), refreshRuns(true)]);
}

async function refreshRuns(refreshResume = false) {
  experiments = await api("/api/studio/experiments");
  const { runs, resumable, all } = experiments;
  fill($("#metricRun"), runs.length ? runs : [{ value: "", key: "metricsEmpty" }]);
  const current = value("resumeExp");
  fill($("#resumeExp"), resumable.length ? resumable : [{ value: "", key: "resumeEmpty" }]);
  fill($("#layerAug"), [{ value: "", key: "off" }, ...all]);
  fill($("#laSource"), all.length ? all : [{ value: "", key: "noExperiments" }]);
  syncTrainingControls();
  if (refreshResume || current !== value("resumeExp") || !$("#resumeInfo").children.length) await showResume();
}
let resumeRevision = 0;
listen("#resumeTotal", "input", event => {
  if (value("resumeExp")) session.remember("resume:" + value("resumeExp"), event.currentTarget.value);
});
async function showResume() {
  const revision = ++resumeRevision;
  const name = value("resumeExp");
  const total = $("#resumeTotal");
  const run = $("#runResume");
  run.disabled = true;
  total.disabled = true;
  total.value = "";
  $("#resumeLimit").hidden = true;
  if (!name) {
    facts($("#resumeInfo"), []);
    return;
  }
  const info = await api("/api/studio/resume?" + new URLSearchParams({ name }));
  if (revision !== resumeRevision || name !== value("resumeExp")) return;
  const maximum = Number(total.max);
  const atLimit = info.epoch >= maximum;
  total.min = Math.min(info.epoch + 1, maximum);
  total.disabled = atLimit;
  run.disabled = atLimit;
  $("#resumeLimit").hidden = !atLimit;
  if (!atLimit) {
    const wanted = Number(session.saved("resume:" + name));
    total.value = Number.isInteger(wanted) && wanted >= Number(total.min) && wanted <= maximum ? wanted : Math.min(info.epoch + 10, maximum);
  }
  facts($("#resumeInfo"), [
    [t("model"), `RVC ${info.version} · ${info.sample_rate / 1000} kHz · ${embedderLabels[info.embedder] || info.embedder}`],
    [t("completedEpoch"), String(info.epoch)],
    [t("trainPrecision"), info.bf16 ? "BF16" : t("automatic")],
    [t("gradientCheckpointing"), t(info.gradient_checkpointing ? "yes" : "no")],
    [t("checkpoints"), fileName(info.generator) + " + " + fileName(info.discriminator)],
  ]);
}
listen("#refreshResume", "click", () => refreshRuns(true));
listen("#resumeExp", "change", showResume);
action("runResume", "resumeLog", async output => {
  const name = requireValue(value("resumeExp"), "resumeEmpty");
  await runJob({
    operation: "resume", name, epochs: number("resumeTotal"),
    ...trainingPrecision("resumePrecision"), gradient_checkpointing: checkpointing("resumeCheckpointing"),
  }, [], output);
}, () => !$("#resumeTotal").disabled);

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
  await runJob({
    operation: "layer_aug", experiment: source, output: target, sample_rate: Number(radio("laSr")),
    seed: number("laSeed"), layers: checked("laLayers"), fx: checked("laFx"),
    fx_profile: checked("laFx") ? value("laFxProfile") || null : null, bleed: value("laBleed") || null,
    extract: checked("laExtract") ? {
      version: value("trainVersion"), f0_method: radio("laF0"), embedder: value("laEmbedder") || null,
      unvoice_fricatives: checked("unvoiceFricatives"),
    } : null,
  }, [], output);
});
action("runFx", "fxLog", async output => {
  await runJob({
    operation: "fx_profile", input: requireValue(value("fxIn"), "chooseInput"), output: requireValue(value("fxOut"), "chooseOutput"),
  }, [], output);
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
  const sums = [0];
  for (const v of ys) sums.push(sums.at(-1) + v);
  const smooth = ys.map((_, i) => {
    const from = Math.max(0, i - half), to = Math.min(ys.length, i + half + 1);
    return (sums[to] - sums[from]) / (to - from);
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
function saveSpeakerEditor() {
  session.remember("speakerEditor", { page: speakerPage, rows: speakers.map(row => session.safeFields(row)) });
}
function restoreSpeakerEditor() {
  const editor = session.saved("speakerEditor");
  if (!Array.isArray(editor?.rows) || !editor.rows.length) return;
  speakers.splice(0, speakers.length, ...editor.rows.slice(0, 110).map((row, i) => {
    const fields = newSpeaker(i);
    for (const key of Object.keys(fields)) if (typeof row?.[key] === "string") fields[key] = row[key];
    return fields;
  }));
  speakerPage = Number.isInteger(editor.page) ? editor.page : 0;
  renderSpeakers();
}
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
      input.addEventListener("input", () => { row[key] = input.value; saveSpeakerEditor(); });
      wrap.append(input);
    }
    return wrap;
  }));
}
listen("#hPrev", "click", () => { speakerPage--; renderSpeakers(); saveSpeakerEditor(); });
listen("#hNext", "click", () => { speakerPage++; renderSpeakers(); saveSpeakerEditor(); });
listen("#hAdd", "click", () => {
  speakers.push(newSpeaker(speakers.length));
  speakerPage = Math.floor((speakers.length - 1) / pageSize);
  renderSpeakers();
  saveSpeakerEditor();
});
listen("#hDel", "click", () => { speakers.pop(); renderSpeakers(); saveSpeakerEditor(); });
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
  const choices = embedderOptions();
  fill($("#trainEmbedder"), [{ value: "", key: "experimentEmbedder" }, ...choices]);
  fill($("#laEmbedder"), [{ value: "", key: "automatic" }, ...choices]);
  fill($("#phoneticEncoder"), [{ value: "", key: "sameContentEmbedder" }, ...choices]);
  syncTrainingControls();
}

async function start() {
  bindRanges();
  bindDrops();
  const restoreFiles = savedFiles();
  enhanceAudioPlayers();
  localize();
  options = await api("/api/studio/options");
  session.openDraft(options.experiments);
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
  session.restoreFields();
  restoreSpeakerEditor();
  const dataset = session.saved("dataset");
  if (dataset?.single && dataset?.multi_folder && [dataset.single, dataset.multi_folder].includes(value("trainset"))) {
    datasetPaths = dataset;
    datasetPath = value("trainset");
  }
  $$(".range-row input[type=number]").forEach(syncRange);
  syncTrainingControls();
  syncProfile();
  syncLibrary();
  syncMode();
  syncCover();
  syncLayers();
  syncSeparationModel();
  void restoreFiles(options.experiments);
  const panelReady = showPanel(addressPanel() || session.saved("panel") || "infer");
  await Promise.all([updateModels(), panelReady ?? refreshRuns()]);
  await jobs.restore(options.experiments);
}
start().catch(error => {
  $("#startup").hidden = false;
  explain(error, $("#startup"));
});
