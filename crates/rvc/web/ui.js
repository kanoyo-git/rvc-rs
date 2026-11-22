import { restoredSelection } from "/studio/session.js";
import { messages } from "/studio/i18n.js";

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
export const t = (key) => (messages[document.documentElement.lang] || messages.en)[key] ?? messages.en[key] ?? key;
export const value = (id) => document.getElementById(id).value.trim();
export const checked = (id) => document.getElementById(id).checked;
export const radio = (name) => $(`input[name="${name}"]:checked`)?.value;
export const icon = (name) => `<svg class="icon" aria-hidden="true"><use href="#${name}" /></svg>`;
export const log = (node, message, error = false) => {
  node.textContent = message;
  node.classList.toggle("is-error", error);
};

export function fill(select, items, keep = select.value) {
  const options = items.map(item => {
    const { value, label, key, disabled } = typeof item === "string" ? { value: item, label: item } : item;
    const option = new Option(key ? t(key) : label, value);
    if (key) option.dataset.i18n = key;
    if (disabled) option.disabled = true;
    return option;
  });
  select.replaceChildren(...options);
  keep = restoredSelection(select, options, keep);
  if (options.some(option => option.value === keep && !option.disabled)) select.value = keep;
}

export function facts(node, rows) {
  node.replaceChildren(...rows.flatMap(([term, detail]) => {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = detail;
    return [dt, dd];
  }));
}

function responseError(status, text) {
  let message;
  try { message = JSON.parse(text).error; } catch { /* Plain-text HTTP error. */ }
  const error = new Error(message || `HTTP ${status}${text ? " · " + text.slice(0, 180) : ""}`);
  error.status = status;
  return error;
}

export async function api(url, init) {
  let response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new Error(t("networkError"));
  }
  if (!response.ok) throw responseError(response.status, await response.text());
  if (response.status === 204 || !response.headers.get("content-type")?.includes("json")) return;
  return response.json();
}

export const send = (url, body, method = "POST") => api(url, {
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export function upload(url, settings, files, onProgress, signal) {
  const data = new FormData();
  if (settings) data.set("settings", JSON.stringify(settings));
  for (const file of files) data.append("files", file);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    if (signal?.aborted) { reject(new DOMException("Upload aborted", "AbortError")); return; }
    const abort = () => xhr.abort();
    xhr.open("POST", url);
    signal?.addEventListener("abort", abort, { once: true });
    xhr.addEventListener("loadend", () => signal?.removeEventListener("abort", abort), { once: true });
    xhr.addEventListener("abort", () => reject(new DOMException("Upload aborted", "AbortError")), { once: true });
    if (onProgress) xhr.upload.addEventListener("progress", event => {
      if (!signal?.aborted && event.lengthComputable) onProgress(Math.round(100 * event.loaded / event.total));
    });
    xhr.addEventListener("load", () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(responseError(xhr.status, xhr.responseText));
        return;
      }
      try {
        resolve(JSON.parse(xhr.responseText));
      } catch (error) {
        reject(error);
      }
    });
    xhr.addEventListener("error", () => reject(new Error(t("networkError"))));
    xhr.send(data);
  });
}

export function toast(message, error = false) {
  const host = $("#toasts");
  const item = document.createElement("div");
  item.className = "toast" + (error ? " is-error" : "");
  item.setAttribute("role", error ? "alert" : "status");
  item.innerHTML = icon(error ? "i-alert" : "i-check");
  const body = document.createElement("p");
  body.textContent = message;
  const close = document.createElement("button");
  close.type = "button";
  close.className = "icon-btn";
  close.innerHTML = icon("i-x");
  close.setAttribute("aria-label", t("close"));
  item.append(body, close);
  host.append(item);
  if (host.children.length > 4) host.firstElementChild.remove();
  const dismiss = () => {
    clearTimeout(timer);
    item.classList.add("is-leaving");
    const exit = item.getAnimations?.().find(animation => animation.animationName === "toast-out");
    if (exit) exit.finished.then(() => item.remove(), () => item.remove());
    else item.remove();
  };
  const timer = setTimeout(dismiss, error ? 9000 : 4500);
  close.addEventListener("click", dismiss, { once: true });
}

export function syncRange(number) {
  const slider = number.previousElementSibling;
  for (const key of ["min", "max", "step", "disabled"]) slider[key] = number[key];
  if (Number.isFinite(number.valueAsNumber) && number.validity.valid) slider.value = number.value;
  const fraction = (slider.valueAsNumber - Number(slider.min)) / (Number(slider.max) - Number(slider.min));
  slider.style.setProperty("--val", `${100 * fraction}%`);
}

export function bindRanges(root = document) {
  const bounds = new MutationObserver(changes => changes.forEach(({ target }) => syncRange(target)));
  $$(".range-row input[type=number]", root).forEach(number => {
    const slider = number.previousElementSibling;
    slider.addEventListener("input", () => {
      number.value = slider.value;
      number.dispatchEvent(new Event("input", { bubbles: true }));
    });
    slider.addEventListener("change", () => number.dispatchEvent(new Event("change", { bubbles: true })));
    number.addEventListener("input", () => syncRange(number));
    number.addEventListener("keydown", event => {
      if (event.key === "Enter") {
        event.preventDefault();
        if (number.reportValidity()) number.blur();
      }
    });
    bounds.observe(number, { attributes: true, attributeFilter: ["min", "max", "step", "disabled"] });
    syncRange(number);
  });
}

export function refreshDrop(zone, refreshPreview = false) {
  const input = document.getElementById(zone.dataset.dropFor);
  const files = [...input.files];
  const name = $(".drop-name", zone);
  name.dataset.empty = String(!files.length);
  name.textContent = files.length > 1 ? t("filesSelected").replace("%s", files.length)
    : files[0]?.name || t("noFile");
  const clear = $(".drop-clear", zone);
  clear.hidden = files.length === 0;
  clear.setAttribute("aria-label", t(files.length > 1 ? "clearFiles" : "clearFile"));
  if (!refreshPreview || !zone.dataset.previewFor) return;
  const audio = document.getElementById(zone.dataset.previewFor);
  const file = files.find(file => file.type.startsWith("audio/") || /\.(aac|flac|m4a|mp3|ogg|opus|wav)$/i.test(file.name));
  audio.pause();
  if (zone.previewUrl) URL.revokeObjectURL(zone.previewUrl);
  zone.previewUrl = file ? URL.createObjectURL(file) : "";
  if (file) {
    audio.dataset.downloadName = file.name;
    audio.src = zone.previewUrl;
  } else {
    audio.removeAttribute("src");
  }
  audio.hidden = !file;
  audio.load();
}

export function bindDrops() {
  $$("[data-drop-for]").forEach(zone => {
    const input = document.getElementById(zone.dataset.dropFor);
    const change = () => input.dispatchEvent(new Event("change", { bubbles: true }));
    zone.tabIndex = 0;
    zone.setAttribute("role", "button");
    zone.addEventListener("click", event => {
      if (event.target.closest("button, input")) return;
      event.preventDefault();
      input.click();
    });
    zone.addEventListener("keydown", event => {
      if (event.target === zone && (event.key === "Enter" || event.key === " ")) {
        event.preventDefault();
        input.click();
      }
    });
    zone.addEventListener("dragover", event => {
      event.preventDefault();
      zone.classList.add("is-over");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("is-over"));
    zone.addEventListener("drop", event => {
      event.preventDefault();
      zone.classList.remove("is-over");
      if (!event.dataTransfer.files.length) return;
      if (input.multiple) {
        input.files = event.dataTransfer.files;
      } else {
        const only = new DataTransfer();
        only.items.add(event.dataTransfer.files[0]);
        input.files = only.files;
      }
      change();
    });
    $(".drop-clear", zone).addEventListener("click", event => {
      event.preventDefault();
      event.stopPropagation();
      input.value = "";
      change();
    });
    input.addEventListener("change", () => refreshDrop(zone, true));
    refreshDrop(zone);
  });
}

export const clock = (seconds) => {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
  const min = Math.floor(seconds / 60);
  const sec = Math.floor(seconds % 60);
  return `${min}:${String(sec).padStart(2, "0")}`;
};

/* Waveform: peaks of the decoded audio, played part in the accent for results. */
let audioContext;
const peakCache = new Map();
async function peaks(src) {
  if (peakCache.has(src)) return peakCache.get(src);
  const job = (async () => {
    const response = await fetch(src);
    if (!response.ok) throw new Error("HTTP " + response.status);
    const bytes = await response.arrayBuffer();
    audioContext ??= new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, 1, 44100);
    const buffer = await audioContext.decodeAudioData(bytes);
    const columns = 1200;
    const data = buffer.getChannelData(0);
    const other = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : null;
    const step = Math.max(1, Math.floor(data.length / columns));
    const result = new Float32Array(columns);
    let top = 0;
    for (let c = 0; c < columns; c++) {
      let peak = 0;
      const end = Math.min(data.length, (c + 1) * step);
      for (let i = c * step; i < end; i += 4) {
        const v = Math.abs(other ? (data[i] + other[i]) / 2 : data[i]);
        if (v > peak) peak = v;
      }
      result[c] = peak;
      if (peak > top) top = peak;
    }
    if (top > 0) for (let c = 0; c < columns; c++) result[c] /= top;
    return result;
  })();
  peakCache.set(src, job);
  job.catch(() => peakCache.delete(src));
  if (peakCache.size > 24) peakCache.delete(peakCache.keys().next().value);
  return job;
}

function drawWave(canvas, data, ratio, accent) {
  const box = canvas.getBoundingClientRect();
  const scale = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(box.width * scale));
  const height = Math.max(1, Math.round(box.height * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  const style = getComputedStyle(canvas);
  const played = accent ? style.getPropertyValue("--accent").trim() : style.getPropertyValue("--text-2").trim();
  const muted = accent ? style.getPropertyValue("--wave-accent").trim() : style.getPropertyValue("--wave").trim() || "#4a4a4f";
  ctx.clearRect(0, 0, width, height);
  const bar = 2 * scale, gap = 1 * scale, mid = height / 2;
  const bars = Math.floor(width / (bar + gap));
  const split = ratio * bars;
  for (let b = 0; b < bars; b++) {
    let level = 0.02;
    if (data) {
      const from = Math.floor(b * data.length / bars), to = Math.max(from + 1, Math.floor((b + 1) * data.length / bars));
      for (let i = from; i < to; i++) if (data[i] > level) level = data[i];
    }
    const h = Math.max(scale, level * (height - 2 * scale));
    ctx.fillStyle = b < split ? played : muted;
    ctx.fillRect(b * (bar + gap), mid - h / 2, bar, h);
  }
}

export function enhanceAudioPlayers(root = document) {
  $$("audio", root).forEach(audio => {
    if (audio.dataset.playerReady) return;
    audio.dataset.playerReady = "true";
    const accent = audio.dataset.accent === "true";
    const wrap = document.createElement("div");
    wrap.className = "player";
    wrap.hidden = audio.hidden;
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "player-toggle";
    toggle.setAttribute("aria-label", t("play"));
    toggle.innerHTML = '<svg class="icon i-play" aria-hidden="true"><use href="#i-play" /></svg><svg class="icon i-pause" aria-hidden="true"><use href="#i-pause" /></svg>';
    const wave = document.createElement("div");
    wave.className = "player-wave";
    wave.setAttribute("role", "slider");
    wave.setAttribute("aria-label", t("seek"));
    wave.setAttribute("aria-valuemin", "0");
    wave.setAttribute("aria-valuemax", "100");
    wave.tabIndex = 0;
    const canvas = document.createElement("canvas");
    wave.append(canvas);
    const side = document.createElement("div");
    side.className = "player-side";
    const time = document.createElement("span");
    time.className = "player-time";
    time.textContent = "0:00";
    const download = document.createElement("a");
    download.className = "player-download";
    download.setAttribute("aria-label", t("download"));
    download.title = t("download");
    download.hidden = true;
    download.innerHTML = icon("i-download");
    const formats = (audio.dataset.downloadFormats || "").split(",").map(f => f.trim()).filter(Boolean);
    let formatSelect = null;
    if (formats.length) {
      formatSelect = document.createElement("select");
      formatSelect.className = "player-format";
      formatSelect.setAttribute("aria-label", t("downloadFormat"));
      formatSelect.hidden = true;
      for (const format of formats) formatSelect.add(new Option(format, format));
    }
    side.append(time, ...(formatSelect ? [formatSelect] : []), download);
    audio.parentNode.insertBefore(wrap, audio);
    wrap.append(toggle, wave, side, audio);

    let data = null;
    let loadingFor = "";
    const ratio = () => Number.isFinite(audio.duration) && audio.duration > 0 ? audio.currentTime / audio.duration : 0;
    const paint = () => drawWave(canvas, data, ratio(), accent);
    const loadPeaks = () => {
      const src = audio.getAttribute("src");
      if (!src || src === loadingFor) return;
      loadingFor = src;
      data = null;
      paint();
      peaks(src).then(result => {
        if (audio.getAttribute("src") === src) {
          data = result;
          paint();
        }
      }).catch(() => {});
    };
    const eager = audio.dataset.wave === "eager" || audio.getAttribute("src")?.startsWith("blob:");

    const downloadName = (src) => {
      if (audio.dataset.downloadName) return audio.dataset.downloadName;
      try {
        const base = new URL(src, location.href).pathname.split(/[\\/]/).pop();
        if (base) return base;
      } catch {}
      return "audio.wav";
    };
    const syncDownload = () => {
      const src = audio.getAttribute("src");
      download.hidden = !src;
      if (formatSelect) formatSelect.hidden = !src || src.startsWith("blob:");
      if (!src) return;
      let href = src;
      let name = downloadName(src);
      const format = formatSelect && !src.startsWith("blob:") ? formatSelect.value : "";
      if (format) {
        const url = new URL(src, location.href);
        url.searchParams.set("format", format);
        url.searchParams.set("download", "true");
        href = url.pathname + url.search;
        name = name.replace(/\.[^.]+$/, "") + "." + format;
      }
      download.href = href;
      download.setAttribute("download", name);
    };
    if (formatSelect) {
      const preferred = audio.dataset.downloadName?.split(".").pop()?.toLowerCase();
      if (formats.includes(preferred)) formatSelect.value = preferred;
      formatSelect.addEventListener("change", syncDownload);
    }

    const sync = () => {
      const dur = audio.duration;
      wave.setAttribute("aria-valuenow", String(Math.round(ratio() * 100)));
      time.textContent = Number.isFinite(dur) ? `${clock(audio.currentTime)} / ${clock(dur)}` : clock(audio.currentTime);
      paint();
    };
    audio.addEventListener("timeupdate", sync);
    audio.addEventListener("durationchange", sync);
    audio.addEventListener("loadedmetadata", () => { sync(); syncDownload(); });
    new MutationObserver(() => {
      syncDownload();
      if (eager || audio.dataset.wave === "eager" || audio.getAttribute("src")?.startsWith("blob:")) loadPeaks();
      else { data = null; loadingFor = ""; paint(); }
    }).observe(audio, { attributes: true, attributeFilter: ["src"] });
    syncDownload();
    audio.addEventListener("play", () => {
      wrap.classList.add("is-playing");
      toggle.setAttribute("aria-label", t("pause"));
      loadPeaks();
    });
    const onStop = () => {
      wrap.classList.remove("is-playing");
      toggle.setAttribute("aria-label", t("play"));
    };
    audio.addEventListener("pause", onStop);
    audio.addEventListener("ended", onStop);
    toggle.addEventListener("click", () => {
      if (audio.paused) {
        for (const other of $$("audio")) if (other !== audio) other.pause();
        audio.play().catch(error => toast(error.message, true));
      } else audio.pause();
    });
    const seek = (event) => {
      if (!Number.isFinite(audio.duration) || !audio.duration) return;
      const rect = wave.getBoundingClientRect();
      audio.currentTime = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)) * audio.duration;
    };
    wave.addEventListener("pointerdown", event => {
      wave.setPointerCapture(event.pointerId);
      seek(event);
      const move = (e) => seek(e);
      wave.addEventListener("pointermove", move);
      wave.addEventListener("lostpointercapture", () => wave.removeEventListener("pointermove", move), { once: true });
    });
    wave.addEventListener("keydown", event => {
      if (!Number.isFinite(audio.duration) || !audio.duration) return;
      const step = event.key === "ArrowRight" ? 5 : event.key === "ArrowLeft" ? -5 : 0;
      if (!step) return;
      event.preventDefault();
      audio.currentTime = Math.min(audio.duration, Math.max(0, audio.currentTime + step));
    });
    new ResizeObserver(paint).observe(wave);
    canvas.addEventListener("repaint", paint);
    new MutationObserver(() => {
      wrap.hidden = audio.hidden;
      if (audio.hidden && !audio.paused) audio.pause();
      if (!audio.hidden) requestAnimationFrame(paint);
    }).observe(audio, { attributes: true, attributeFilter: ["hidden"] });
    if (eager) loadPeaks();
    sync();
  });
}

export function repaintPlayers() {
  for (const canvas of $$(".player-wave canvas")) canvas.dispatchEvent(new Event("repaint"));
}
