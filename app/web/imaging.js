// Image maths of the browser demo, written to match TensorFlow (D-023).
// Plain functions on typed arrays: used by app.js in the browser and by the Node check
// (tests/web/check_imaging.mjs) against tf.image.resize.
//
// rgb images: Uint8Array of length h * w * 3 (row-major, channels last), like a decoded photo.

// tf.image.resize(method="bilinear", antialias=True) -> round -> clip -> uint8.
// A port of TensorFlow's ScaleAndTranslate (kernels/image/scale_and_translate_op.cc): triangle
// kernel widened by 1/scale when downscaling, rows gathered first, then columns, float32
// arithmetic in the same order. Checked against TensorFlow in tests/web/check_imaging.js.
function spans(inSize, outSize) {
  const f = Math.fround;
  const scale = f(outSize / inSize);
  const invScale = f(1 / scale);
  const kScale = f(Math.max(invScale, 1));
  const oneOverK = f(1 / kScale);
  const starts = new Int32Array(outSize);
  const weights = [];
  for (let x = 0; x < outSize; x++) {
    const sample = f(f(x + 0.5) * invScale);
    let start = Math.ceil(f(f(sample - kScale) - 0.5));
    let end = Math.floor(f(f(sample + kScale) - 0.5));
    start = Math.min(Math.max(start, 0), inSize - 1);
    end = Math.min(Math.max(end, 0), inSize - 1) + 1;
    const w = new Float32Array(end - start);
    let total = 0;
    for (let s = start; s < end; s++) {
      const pos = f(Math.abs(f(f(f(s + 0.5) - sample) * oneOverK)));
      const k = pos < 1 ? f(1 - pos) : 0;
      w[s - start] = k;
      total = f(total + k);
    }
    const inv = f(1 / total);
    for (let i = 0; i < w.length; i++) w[i] = f(w[i] * inv);
    starts[x] = start;
    weights.push(w);
  }
  return { starts, weights };
}

// tf.round: halves go to the even neighbour (2.5 -> 2, 3.5 -> 4).
function roundHalfEven(v) {
  const r = Math.round(v);
  return r - v === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

function resizeAntialias(rgb, h, w, size) {
  const f = Math.fround;
  const rows = spans(h, size);
  const cols = spans(w, size);
  const rowLen = w * 3;
  const tmp = new Float32Array(size * rowLen); // rows first: [size, w, 3]
  for (let y = 0; y < size; y++) {
    const wt = rows.weights[y];
    const out = y * rowLen;
    for (let i = 0; i < wt.length; i++) {
      const src = (rows.starts[y] + i) * rowLen;
      for (let k = 0; k < rowLen; k++) tmp[out + k] = f(tmp[out + k] + f(wt[i] * rgb[src + k]));
    }
  }
  const out = new Uint8Array(size * size * 3);
  const acc = new Float32Array(3);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const wt = cols.weights[x];
      acc.fill(0);
      for (let i = 0; i < wt.length; i++) {
        const src = y * rowLen + (cols.starts[x] + i) * 3;
        for (let c = 0; c < 3; c++) acc[c] = f(acc[c] + f(wt[i] * tmp[src + c]));
      }
      for (let c = 0; c < 3; c++) {
        out[(y * size + x) * 3 + c] = Math.min(255, Math.max(0, roundHalfEven(acc[c])));
      }
    }
  }
  return out;
}

// Largest centred square of an [h, w, 3] image (crop offsets), like predict.center_square.
function centerSquare(h, w) {
  const s = Math.min(h, w);
  return { s, top: Math.floor((h - s) / 2), left: Math.floor((w - s) / 2) };
}

// tf.image.resize(method="bilinear") of a small float map, no antialiasing (upsampling the
// 7 x 7 Grad-CAM map to the photo), half-pixel centres; then scaled to max 1.
function upsampleCam(cam, ch, cw, size) {
  const out = new Float32Array(size * size);
  let max = 0;
  for (let y = 0; y < size; y++) {
    const fy = Math.max((y + 0.5) * (ch / size) - 0.5, 0);
    const y0 = Math.min(Math.floor(fy), ch - 1);
    const y1 = Math.min(y0 + 1, ch - 1);
    const dy = fy - y0;
    for (let x = 0; x < size; x++) {
      const fx = Math.max((x + 0.5) * (cw / size) - 0.5, 0);
      const x0 = Math.min(Math.floor(fx), cw - 1);
      const x1 = Math.min(x0 + 1, cw - 1);
      const dx = fx - x0;
      const top = cam[y0 * cw + x0] + (cam[y0 * cw + x1] - cam[y0 * cw + x0]) * dx;
      const bot = cam[y1 * cw + x0] + (cam[y1 * cw + x1] - cam[y1 * cw + x0]) * dx;
      const v = top + (bot - top) * dy;
      out[y * size + x] = v;
      if (v > max) max = v;
    }
  }
  if (max > 0) for (let i = 0; i < out.length; i++) out[i] /= max;
  return out;
}

// matplotlib's "jet" colour map (the Python demo's overlay), 256 entries.
const JET = (() => {
  const seg = {
    r: [[0, 0], [0.35, 0], [0.66, 1], [0.89, 1], [1, 0.5]],
    g: [[0, 0], [0.125, 0], [0.375, 1], [0.64, 1], [0.91, 0], [1, 0]],
    b: [[0, 0.5], [0.11, 1], [0.34, 1], [0.65, 0], [1, 0]],
  };
  const at = (pts, x) => {
    for (let i = 1; i < pts.length; i++) {
      if (x <= pts[i][0]) {
        const [x0, y0] = pts[i - 1];
        const [x1, y1] = pts[i];
        return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
      }
    }
    return pts[pts.length - 1][1];
  };
  const lut = new Float32Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    lut[i * 3] = at(seg.r, x);
    lut[i * 3 + 1] = at(seg.g, x);
    lut[i * 3 + 2] = at(seg.b, x);
  }
  return lut;
})();

// RGBA pixels of the photo with the heat map blended on top (alpha 0.45, like gradcam.overlay).
function overlay(rgb, cam, size, alpha = 0.45) {
  const out = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    const j = Math.min(255, Math.max(0, Math.floor(cam[i] * 256))) * 3;
    for (let c = 0; c < 3; c++) {
      const v = (1 - alpha) * (rgb[i * 3 + c] / 255) + alpha * JET[j + c];
      out[i * 4 + c] = Math.round(Math.min(1, Math.max(0, v)) * 255);
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}

const imaging = { spans, roundHalfEven, resizeAntialias, centerSquare, upsampleCam, overlay, JET };
if (typeof module !== "undefined") module.exports = imaging;
if (typeof window !== "undefined") window.imaging = imaging;
