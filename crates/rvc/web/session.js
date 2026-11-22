// Local form drafts are scoped to the Studio workspace behind this browser origin.
// Only changed fields are written, so a second tab saving another field does not overwrite it.
const pending = new Map();
const draft = new Map();
const modified = new Map();
let storageKey;
let timer;

export function preference(key, fallback) {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
}
export function storePreference(key, value) {
  try { localStorage.setItem(key, value); } catch { /* The form also works without storage. */ }
}
function readDraft() {
  try {
    const text = localStorage.getItem(storageKey);
    if (!text || text.length > 512 * 1024) return {};
    const value = JSON.parse(text);
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  } catch { return {}; }
}
const credential = /-----BEGIN .*PRIVATE KEY|\b(?:Bearer\s+|gh[pousr]_|github_pat_|hf_[a-z0-9]{15,}|KGAT_|sk-[A-Za-z0-9])|(?:^|[?&\s])(?:token|signature|api[_-]?key|password)=|^[a-f0-9]{64}$/i;
const sensitiveParameter = /auth|token|secret|passw|credential|signature|(?:^|[-_])sig(?:$|[-_])|(?:^|[_-])key(?:$|[_-])|resourcekey|policy|expires|^x-(?:amz|goog|ms)-|^googleaccessid$|^(?:jwt|session|code|hmac|keypairid)$/i;
function publicLink(text) {
  try {
    const url = new URL(text);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || !url.host) return false;
    for (const [name, value] of url.searchParams) {
      if (sensitiveParameter.test(name) || credential.test(value) || /^eyJ[\w-]+\.[\w-]+\./.test(value)) return false;
    }
    if (url.hash && (/(?:^|\.)mega\.(?:nz|co\.nz|io)$/i.test(url.hostname) || !/^#[a-z][\w.-]{0,31}$/i.test(url.hash))) return false;
    return !credential.test(decodeURIComponent(url.pathname));
  } catch { return false; }
}
function safeText(value) {
  if (typeof value !== 'string' || value.length > 4096 || /[\u0000]/.test(value) || credential.test(value)) return false;
  if (/^https?:\/\//i.test(value)) return publicLink(value);
  return !/(?:[a-z][a-z0-9+.-]*:\/\/|\b(?:data|blob):)/i.test(value);
}
export function safeFields(entries) {
  return Object.fromEntries(Object.entries(entries).filter(([, v]) => typeof v === 'boolean' || safeText(v)));
}
function flush() {
  clearTimeout(timer);
  if (!storageKey || !modified.size) return;
  const state = readDraft();
  for (const [key, value] of modified) {
    if (value == null) delete state[key]; else state[key] = value;
  }
  modified.clear();
  storePreference(storageKey, JSON.stringify(state));
}
export function openDraft(workspace) {
  storageKey = 'rvc-studio-draft-v1:' + location.pathname + ':' + workspace;
  for (const [key, value] of Object.entries(readDraft())) {
    if (!modified.has(key)) { draft.set(key, value); pending.set(key, value); }
  }
  flush();
}
export function remember(key, value) {
  pending.delete(key);
  if (value == null) draft.delete(key); else draft.set(key, value);
  modified.set(key, value);
  clearTimeout(timer);
  timer = setTimeout(flush, 60);
}
export const saved = key => draft.get(key);
export function take(key) {
  const value = pending.get(key);
  pending.delete(key);
  return value;
}

const texts = new Set(('iIndexPath iProfileDataset bIn bOut cIndexPath cIn cInstPath cOut pVocal pResidual pIn expName trainset trainGpus preGPath preDPath trainOnly timbrePool timbreExclude poolSpec poolOutput laOutput laFxProfile laBleed fxIn fxOut hExp libName libUrl libGUrl libDUrl mA mB mName mInfo ePath eName eInfo modPath modName modInfo insPath').split(' '));
function controlKey(el) {
  if (!el?.matches?.('input, select, textarea') || !el.closest('.panel') || el.closest('#live-form') || el.id === 'resumeTotal') return;
  if (el.type === 'radio') return el.name && 'radio:' + el.name;
  if (!el.id || (!el.matches('select, input[type=number], input[type=checkbox]') && !texts.has(el.id))) return;
  return 'field:' + el.id;
}
function changed(event) {
  const el = event.target;
  if (el.closest?.('#live-form') && el.name && el.matches('input, select') && !['file', 'url', 'password', 'range'].includes(el.type)) {
    const value = el.type === 'checkbox' ? el.checked : el.value;
    remember('live:' + el.name, typeof value === 'boolean' || safeText(value) ? value : null);
    return;
  }
  const key = controlKey(el);
  if (!key || (el.type === 'radio' && !el.checked)) return;
  const value = el.type === 'checkbox' ? el.checked : el.value;
  remember(key, typeof value === 'boolean' || safeText(value) ? value : null);
  // Entering a directory manually replaces a previous upload.
  if (el.id === 'trainset') remember('dataset', null);
}
document.addEventListener('input', changed);
document.addEventListener('change', changed);
window.addEventListener('pagehide', flush);
document.addEventListener('visibilitychange', () => { if (document.hidden) flush(); });

// Lists of voices, pretrained pairs and experiment metrics can arrive after the static form.
export function restoredSelection(select, options, fallback) {
  const key = controlKey(select);
  const wanted = pending.get(key);
  if (wanted === undefined) return fallback;
  if (options.some(item => item.value === wanted && !item.disabled)) {
    pending.delete(key);
    return wanted;
  }
  if (options.some(item => item.value && !item.disabled)) pending.delete(key);
  return fallback;
}
export function restoreFields() {
  for (const el of document.querySelectorAll('.panel input, .panel select, .panel textarea')) {
    const key = controlKey(el);
    if (!pending.has(key)) continue;
    if (el.type === 'radio') {
      if (pending.get(key) === el.value && !el.disabled) { el.checked = true; pending.delete(key); }
      continue;
    }
    if (el.tagName === 'SELECT') {
      if (el.options.length) el.value = restoredSelection(el, [...el.options], el.value);
      continue;
    }
    const value = take(key);
    if (el.type === 'checkbox') { if (typeof value === 'boolean') el.checked = value; }
    else if (safeText(value)) el.value = value;
  }
  for (const el of document.querySelectorAll('.panel details')) {
    const value = take(detailKey(el));
    if (typeof value === 'boolean') el.open = value;
  }
}
function detailKey(el) {
  const label = el.querySelector('summary');
  return 'details:' + el.closest('.panel').dataset.panel + ':' + (label.dataset.i18n || label.textContent.trim());
}
document.addEventListener('toggle', event => {
  if (event.target.matches?.('.panel details')) remember(detailKey(event.target), event.target.open);
}, true);
export const waiting = key => pending.get(key);

export function liveValues(form) {
  return Object.fromEntries([...form.elements].filter(el => el.name && draft.has('live:' + el.name))
    .map(el => [el.name, draft.get('live:' + el.name)]));
}
