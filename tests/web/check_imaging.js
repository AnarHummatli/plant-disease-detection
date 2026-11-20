// Run by tests/test_web.py: compares app/web/imaging.js with TensorFlow outputs written by Python.
// usage: node check_imaging.js <folder>   (folder has cases.json and the raw arrays)
const fs = require("fs");
const path = require("path");
const im = require(path.join(__dirname, "..", "..", "app", "web", "imaging.js"));
const dir = process.argv[2];
const cases = JSON.parse(fs.readFileSync(path.join(dir, "cases.json")));
const u8 = (f) => new Uint8Array(fs.readFileSync(path.join(dir, f)));
const f32 = (f) => { const b = fs.readFileSync(path.join(dir, f)); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
const out = { resize: [], cam: [] };
for (const c of cases.resize) {
  const got = im.resizeAntialias(u8(c.input), c.h, c.w, c.size);
  const ref = u8(c.output);
  let differ = 0;
  for (let i = 0; i < ref.length; i++) if (got[i] !== ref[i]) differ++;
  out.resize.push(differ);
}
for (const c of cases.cam) {
  const got = im.upsampleCam(f32(c.input), c.h, c.w, c.size);
  const ref = f32(c.output);
  let d = 0;
  for (let i = 0; i < ref.length; i++) d = Math.max(d, Math.abs(got[i] - ref[i]));
  out.cam.push(d);
}
console.log(JSON.stringify(out));
