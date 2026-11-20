// Browser demo: photo -> centred square -> TensorFlow-exact resize -> ONNX model -> verdict,
// top-3 and Grad-CAM (D-023). Same answers as plantdx.predict / app/gradio/app.py.
"use strict";

const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
const EXAMPLES = [
  "apple_cedar_apple_rust", "grape_black_rot", "pepper_bell_healthy", "potato_early_blight",
  "strawberry_leaf_scorch", "tomato_early_blight_part_of_leaf", "tomato_yellow_leaf_curl_virus",
];
const MAX_EXACT = 512; // squares up to this size are resized exactly like training

const $ = (id) => document.getElementById(id);
const state = { meta: null, session: null, busy: false };
window.plantdx = state; // inspected by the automated browser check

function setStatus(text, error = false) {
  $("status").textContent = text;
  $("status").className = "status" + (error ? " error" : "");
}

async function fetchWithProgress(url, label) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url}: HTTP ${resp.status}`);
  const total = Number(resp.headers.get("content-length")) || 0;
  if (!resp.body || !total) return new Uint8Array(await resp.arrayBuffer());
  const reader = resp.body.getReader();
  const buf = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, got);
    got += value.length;
    setStatus(`${label} ${Math.round((100 * got) / total)}%`);
  }
  return buf.subarray(0, got);
}

async function init() {
  try {
    state.meta = await (await fetch("meta.json")).json();
    const m = state.meta;
    $("crops").innerHTML = `<b>Supported crops (${m.crops.length}):</b> ${m.crops.join(", ")}.`;
    $("thr").textContent = `${(100 * m.threshold).toFixed(1)}%`;
    $("mb").textContent = `${(m.onnx_bytes / 1e6).toFixed(1)} MB`;
    $("ver").textContent = m.model_version;
    buildExamples();
    ort.env.wasm.wasmPaths = ORT_DIST;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    const bytes = await fetchWithProgress("model.onnx", "Loading the model (once):");
    setStatus("Preparing the model…");
    state.session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    await predict(new Uint8Array(224 * 224 * 3).fill(128)); // warm-up
    setStatus("Model ready. It runs on this device; photos are not uploaded.");
    state.ready = true;
  } catch (err) {
    console.error(err);
    setStatus(`Could not load the model: ${err.message}`, true);
  }
}

function buildExamples() {
  const box = $("examples");
  for (const name of EXAMPLES) {
    const b = document.createElement("button");
    b.type = "button";
    b.title = name.replaceAll("_", " ");
    b.innerHTML = `<img src="examples/${name}.jpg" alt="${b.title}" loading="lazy"><span>${b.title}</span>`;
    b.addEventListener("click", async () => {
      const blob = await (await fetch(`examples/${name}.jpg`)).blob();
      handle(blob);
    });
    box.appendChild(b);
  }
}

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
  const probs = Array.from(out.probs.data);
  const cam = new Float32Array(out.cam.data);
  const [ch, cw] = out.cam.dims.slice(1);
  return { probs, cam, ch, cw };
}

function verdictHTML(top, conf, m) {
  const cls = m.classes[top];
  const pct = `${(100 * conf).toFixed(1)}%`;
  if (conf < m.threshold) {
    return ["unsure", `<h3>🟡 Not sure</h3><p>The best guess is <i>${cls.display}</i>, but with
      confidence <b>${pct}</b> (below ${(100 * m.threshold).toFixed(1)}%). Please take a clearer
      photo: one whole leaf, in focus, filling most of the frame.</p>`];
  }
  if (cls.healthy) return ["ok", `<h3>🟢 ${cls.crop}: healthy</h3><p>confidence <b>${pct}</b></p>`];
  return ["bad", `<h3>🔴 ${cls.crop}: ${cls.condition}</h3><p>confidence <b>${pct}</b>. A
    preliminary result: confirm it with an agronomist before treating.</p>`];
}

function drawRGB(canvas, rgb) {
  const img = new ImageData(224, 224);
  for (let i = 0, j = 0; j < rgb.length; i += 4, j += 3) {
    img.data[i] = rgb[j]; img.data[i + 1] = rgb[j + 1]; img.data[i + 2] = rgb[j + 2]; img.data[i + 3] = 255;
  }
  canvas.getContext("2d").putImageData(img, 0, 0);
}

async function handle(blob) {
  if (!state.ready || state.busy || !blob) return;
  state.busy = true;
  try {
    const url = URL.createObjectURL(blob);
    $("preview").src = url;
    $("preview").classList.remove("hidden");
    $("hint").classList.add("hidden");
    const t0 = performance.now();
    const rgb = await prepare(blob);
    const t1 = performance.now();
    const { probs, cam, ch, cw } = await predict(rgb);
    const t2 = performance.now();
    const m = state.meta;
    const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    const [conf, top] = order[0];
    const [kind, html] = verdictHTML(top, conf, m);
    $("verdict").className = `verdict ${kind}`;
    $("verdict").innerHTML = html;
    $("bars").innerHTML = order.slice(0, 3).map(([p, i]) => `<div class="bar"><div class="label">
      <span>${m.classes[i].display}</span><span>${(100 * p).toFixed(p >= 0.995 || p < 0.005 ? 0 : 1)}%</span>
      </div><div class="track"><div class="fill" style="width:${(100 * p).toFixed(1)}%"></div></div></div>`).join("");
    const heat = imaging.upsampleCam(cam, ch, cw, 224);
    drawRGB($("seen"), rgb);
    $("heat").getContext("2d").putImageData(new ImageData(imaging.overlay(rgb, heat, 224), 224, 224), 0, 0);
    $("heatcap").textContent = `Heat map for: ${m.classes[top].display}`;
    $("info").textContent = `Photo prepared in ${Math.round(t1 - t0)} ms, model and Grad-CAM in ${Math.round(t2 - t1)} ms, on this device.`;
    $("details").classList.remove("hidden");
    Object.assign(state, { lastInput: rgb, lastProbs: probs, lastCam: cam, lastTop: top });
  } catch (err) {
    console.error(err);
    $("verdict").className = "verdict unsure";
    $("verdict").innerHTML = `<p>Could not read this file as an image (${err.message}).</p>`;
  } finally {
    state.busy = false;
  }
}

$("file").addEventListener("change", (e) => { handle(e.target.files[0]); e.target.value = ""; });
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("over");
  handle(e.dataTransfer.files[0]);
});
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) handle(item.getAsFile());
});
$("paste").addEventListener("click", async () => {
  try {
    for (const item of await navigator.clipboard.read()) {
      const type = item.types.find((t) => t.startsWith("image/"));
      if (type) return handle(await item.getType(type));
    }
    setStatus("No image in the clipboard.");
  } catch {
    setStatus("Clipboard not available here: use Ctrl+V / ⌘V on the page instead.");
  }
});

init();
