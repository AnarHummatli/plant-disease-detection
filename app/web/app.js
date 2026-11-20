// Browser demo: photo -> centred square -> TensorFlow-exact resize -> ONNX model -> verdict,
// top-3 and Grad-CAM (D-023). Same answers as plantdx.predict / app/gradio/app.py.
"use strict";

const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
const EXAMPLES = [
  "apple_cedar_apple_rust", "grape_black_rot", "pepper_bell_healthy", "potato_early_blight",
  "strawberry_leaf_scorch", "tomato_early_blight_part_of_leaf", "tomato_yellow_leaf_curl_virus",
];
const MAX_EXACT = 512; // squares up to this size are resized exactly like training
const LANG_KEY = "plantdx.lang";

const $ = (id) => document.getElementById(id);
const state = { meta: null, session: null, busy: false, ready: false, lang: "en", last: null };
window.plantdx = state; // inspected by the automated browser check

// ------------------------------------------------------------------------------ language
function pickLanguage() {
  const fromUrl = new URLSearchParams(location.search).get("lang");
  if (fromUrl === "az" || fromUrl === "en") return fromUrl;
  try {
    const saved = localStorage.getItem(LANG_KEY);
    if (saved === "az" || saved === "en") return saved;
  } catch { /* storage unavailable: use the default */ }
  return "en"; // English unless the visitor chose Azerbaijani
}

function t(key, vars = {}) {
  const text = i18n.I18N[state.lang][key] ?? i18n.I18N.en[key] ?? key;
  return text.replace(/\{(\w+)\}/g, (_, k) => (k in vars ? vars[k] : `{${k}}`));
}

const pct = (p, digits = 1) => `${(100 * p).toFixed(digits)}%`;
const threshold = () => pct(state.meta ? state.meta.threshold : 0.947);

// class -> { crop, condition, en } in the current language
function names(i) {
  const c = state.meta.classes[i];
  if (state.lang === "az") {
    return { crop: i18n.CROPS_AZ[c.crop] || c.crop, condition: i18n.CONDITIONS_AZ[i], en: c.display };
  }
  return { crop: c.crop, condition: c.healthy ? "healthy" : c.condition, en: null };
}

function applyLanguage(lang) {
  state.lang = lang;
  try { localStorage.setItem(LANG_KEY, lang); } catch { /* not persisted */ }
  document.documentElement.lang = lang;
  document.title = t("doc.title");
  const vars = { thr: threshold() };
  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n, vars);
  for (const b of document.querySelectorAll(".lang button")) {
    b.setAttribute("aria-pressed", String(b.dataset.lang === lang));
  }
  if (state.meta) {
    $("crops").textContent = state.meta.crops.map((c) => (lang === "az" ? i18n.CROPS_AZ[c] || c : c)).join(", ");
  }
  if (state.status) setStatus(...state.status);
  if (state.last) renderResult(state.last);
}

// --------------------------------------------------------------------------------- status
function setStatus(key, vars = {}, kind = "") {
  state.status = [key, vars, kind];
  $("status-text").textContent = t(key, vars);
  $("status").className = `status ${kind}`;
  if ("p" in vars) $("progress").style.width = `${vars.p}%`;
}

async function fetchWithProgress(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url}: HTTP ${resp.status}`);
  const total = Number(resp.headers.get("content-length")) || state.meta.onnx_bytes || 0;
  if (!resp.body || !total) return new Uint8Array(await resp.arrayBuffer());
  const reader = resp.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    setStatus("status.loading", { p: Math.min(100, Math.round((100 * got) / total)) });
  }
  const buf = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) { buf.set(c, at); at += c.length; }
  return buf;
}

// ------------------------------------------------------------------------------- startup
async function init() {
  state.lang = pickLanguage();
  applyLanguage(state.lang);
  setStatus("status.loading", { p: 0 });
  try {
    state.meta = await (await fetch("meta.json")).json();
    applyLanguage(state.lang);
    buildExamples();
    ort.env.wasm.wasmPaths = ORT_DIST;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    const bytes = await fetchWithProgress("model.onnx");
    setStatus("status.preparing", { p: 100 });
    state.session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    await predict(new Uint8Array(224 * 224 * 3).fill(128)); // warm-up
    setStatus("status.ready", {}, "ready");
    state.ready = true;
  } catch (err) {
    console.error(err);
    setStatus("status.error", { msg: err.message }, "error");
  }
}

function buildExamples() {
  const box = $("examples");
  for (const name of EXAMPLES) {
    const b = document.createElement("button");
    b.type = "button";
    b.title = name.replaceAll("_", " ");
    b.setAttribute("aria-label", b.title);
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<img src="examples/${name}.jpg" alt="" loading="lazy" width="96" height="96">`;
    b.addEventListener("click", async () => {
      const blob = await (await fetch(`examples/${name}.jpg`)).blob();
      handle(blob, b);
    });
    box.appendChild(b);
  }
}

// ------------------------------------------------------------------------------ the model
// Photo -> uint8 RGB [224, 224, 3], like plantdx.predict.prepare_image.
async function prepare(blob) {
  const opts = { imageOrientation: "from-image", colorSpaceConversion: "none", premultiplyAlpha: "none" };
  const bmp = await createImageBitmap(blob, opts);
  const { s, top, left } = imaging.centerSquare(bmp.height, bmp.width);
  // large photos: the browser halves the square to 448 px first (like the decoder downscale in
  // Python), then the exact TensorFlow resize to 224
  const side = s > MAX_EXACT ? 2 * 224 : s;
  const square = await createImageBitmap(bmp, left, top, s, s, {
    ...opts, resizeWidth: side, resizeHeight: side, resizeQuality: "high",
  });
  const canvas = new OffscreenCanvas(side, side);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(square, 0, 0);
  const rgba = ctx.getImageData(0, 0, side, side).data;
  const rgb = new Uint8Array(side * side * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i]; rgb[j + 1] = rgba[i + 1]; rgb[j + 2] = rgba[i + 2];
  }
  bmp.close(); square.close();
  return side === 224 ? rgb : imaging.resizeAntialias(rgb, side, side, 224);
}

async function predict(rgb) {
  const x = new Float32Array(rgb.length);
  for (let i = 0; i < rgb.length; i++) x[i] = rgb[i];
  const out = await state.session.run({ image: new ort.Tensor("float32", x, [1, 224, 224, 3]) });
  const [ch, cw] = out.cam.dims.slice(1);
  return { probs: Array.from(out.probs.data), cam: new Float32Array(out.cam.data), ch, cw };
}

// ------------------------------------------------------------------------------ the label
function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

function renderResult(r) {
  const m = state.meta;
  const order = r.probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  const [conf, top] = order[0];
  const n = names(top);
  const unsure = conf < m.threshold;
  const healthy = m.classes[top].healthy;
  const kind = unsure ? "unsure" : healthy ? "ok" : "bad";
  const headline = unsure ? t("verdict.unsure") : healthy ? t("verdict.healthy") : n.condition;
  const enLine = !unsure && n.en ? `<span class="en" lang="en">${esc(n.en)}</span>` : "";
  const note = unsure
    ? t("note.unsure", { name: `${n.crop}: ${n.condition}`, pct: pct(conf), thr: threshold() })
    : t(healthy ? "note.healthy" : "note.disease");
  const bars = order.slice(0, 3).map(([p, i]) => {
    const k = names(i);
    const label = `${k.crop}: ${m.classes[i].healthy ? t("verdict.healthy").toLowerCase() : k.condition}`;
    const shown = p >= 0.995 || p < 0.005 ? pct(p, 0) : pct(p);
    return `<li><span>${esc(label)}</span><span class="pct">${shown}</span>
      ${k.en ? `<span class="sub" lang="en">${esc(k.en)}</span>` : ""}
      <span class="track"><span class="fill" style="width:${(100 * p).toFixed(1)}%"></span></span></li>`;
  }).join("");
  $("label-no").textContent = new Date().toLocaleDateString(state.lang === "az" ? "az-Latn-AZ" : "en-GB");
  $("result").innerHTML = `
    <p class="verdict ${kind}">${esc(headline)}${enLine}</p>
    <p class="note ${kind}">${esc(note)}</p>
    <dl class="fields">
      <dt>${t("label.crop")}</dt><dd>${esc(n.crop)}</dd>
      <dt>${t("label.condition")}</dt><dd>${esc(unsure ? "—" : healthy ? t("verdict.healthy") : n.condition)}</dd>
      <dt>${t("label.confidence")}</dt><dd>${pct(conf)}</dd>
      <dt>${t("label.by")}</dt><dd>${esc(t("label.model", { v: m.model_version }))}</dd>
    </dl>
    <div class="analysis">
      <h3>${t("top.title")}</h3>
      <ol class="bars">${bars}</ol>
      <h3>${t("cam.title")}</h3>
      <div class="pair">
        <figure><canvas id="seen" width="224" height="224"></canvas><figcaption>${t("cam.seen")}</figcaption></figure>
        <figure><canvas id="heat" width="224" height="224"></canvas><figcaption>${t("cam.heat")}</figcaption></figure>
      </div>
      <p class="timing" id="info">${t("timing", { a: r.prepMs, b: r.runMs })}</p>
    </div>`;
  const img = new ImageData(224, 224);
  for (let i = 0, j = 0; j < r.rgb.length; i += 4, j += 3) {
    img.data[i] = r.rgb[j]; img.data[i + 1] = r.rgb[j + 1]; img.data[i + 2] = r.rgb[j + 2]; img.data[i + 3] = 255;
  }
  $("seen").getContext("2d").putImageData(img, 0, 0);
  const heat = imaging.upsampleCam(r.cam, r.ch, r.cw, 224);
  $("heat").getContext("2d").putImageData(new ImageData(imaging.overlay(r.rgb, heat, 224), 224, 224), 0, 0);
}

function renderError() {
  $("result").innerHTML = `<p class="note unsure">${esc(t("note.unreadable"))}</p>`;
}

// ------------------------------------------------------------------------------ the flow
// `example` is the example button that was clicked, or null for the visitor's own photo
async function handle(blob, example = null) {
  if (!state.ready || state.busy || !blob) return;
  state.busy = true;
  for (const o of $("examples").children) o.setAttribute("aria-pressed", String(o === example));
  const mount = $("mount");
  mount.classList.add("busy");
  try {
    const url = URL.createObjectURL(blob);
    const img = $("preview");
    if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
    img.dataset.url = url;
    img.src = url;
    img.hidden = false;
    $("empty").hidden = true;
    const t0 = performance.now();
    const rgb = await prepare(blob);
    const t1 = performance.now();
    const out = await predict(rgb);
    const t2 = performance.now();
    state.last = { ...out, rgb, prepMs: Math.round(t1 - t0), runMs: Math.round(t2 - t1) };
    renderResult(state.last);
    Object.assign(state, { lastInput: rgb, lastProbs: out.probs, lastCam: out.cam });
    if (window.matchMedia("(max-width: 860px)").matches) {
      document.querySelector(".label").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (err) {
    console.error(err);
    state.last = null;
    renderError();
  } finally {
    mount.classList.remove("busy");
    state.busy = false;
  }
}

function wire() {
  for (const id of ["file", "camera"]) {
    $(id).addEventListener("change", (e) => { handle(e.target.files[0]); e.target.value = ""; });
  }
  if (window.matchMedia("(pointer: coarse)").matches) $("camera-btn").hidden = false;
  const mount = $("mount");
  mount.addEventListener("dragover", (e) => { e.preventDefault(); mount.classList.add("over"); });
  mount.addEventListener("dragleave", () => mount.classList.remove("over"));
  mount.addEventListener("drop", (e) => {
    e.preventDefault();
    mount.classList.remove("over");
    handle(e.dataTransfer.files[0]);
  });
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) handle(item.getAsFile());
  });
  $("paste").addEventListener("click", async () => {
    try {
      for (const item of await navigator.clipboard.read()) {
        const type = item.types.find((x) => x.startsWith("image/"));
        if (type) return handle(await item.getType(type));
      }
      setStatus("paste.none", {}, state.ready ? "ready" : "");
    } catch {
      setStatus("paste.blocked", {}, state.ready ? "ready" : "");
    }
  });
  for (const b of document.querySelectorAll(".lang button")) {
    b.addEventListener("click", () => applyLanguage(b.dataset.lang));
  }
}

wire();
init();
