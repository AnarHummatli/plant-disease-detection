"""Browser demo (D-023): the JavaScript image maths and the ONNX export match TensorFlow.

The JavaScript check needs Node.js, the export check tf2onnx and onnxruntime (`.[web]`);
each is skipped when its tools are missing.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

import keras  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "app" / "gradio" / "examples"


def test_javascript_resize_is_bit_identical_to_tensorflow(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js not installed")
    from plantdx.predict import load_rgb

    photos = [load_rgb(p) for p in sorted(EXAMPLES.glob("*.jpg"))]  # 256 x 256 dataset photos
    rng = np.random.default_rng(0)
    for n in (448, 300, 513, 100):  # other square sizes the page can meet
        photos.append(rng.integers(0, 256, (n, n, 3), dtype=np.uint8))
    cases = {"resize": [], "cam": []}
    for i, rgb in enumerate(photos):
        ref = tf.image.resize(rgb, (224, 224), method="bilinear", antialias=True)
        ref = tf.cast(tf.round(tf.clip_by_value(ref, 0.0, 255.0)), tf.uint8).numpy()
        rgb.tofile(tmp_path / f"in{i}.raw")
        ref.tofile(tmp_path / f"out{i}.raw")
        h, w = rgb.shape[:2]
        cases["resize"].append(
            {"input": f"in{i}.raw", "output": f"out{i}.raw", "h": h, "w": w, "size": 224}
        )
    for i in range(3):
        cam = rng.random((7, 7)).astype(np.float32)
        up = tf.image.resize(cam[None, ..., None], (224, 224), method="bilinear")[0, ..., 0]
        up = (up / tf.reduce_max(up)).numpy().astype(np.float32)
        cam.tofile(tmp_path / f"cam{i}.raw")
        up.tofile(tmp_path / f"camup{i}.raw")
        cases["cam"].append(
            {"input": f"cam{i}.raw", "output": f"camup{i}.raw", "h": 7, "w": 7, "size": 224}
        )
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    script = ROOT / "tests" / "web" / "check_imaging.js"
    out = json.loads(
        subprocess.run(
            [node, str(script), str(tmp_path)], check=True, capture_output=True, text=True
        ).stdout
    )
    assert out["resize"] == [0] * len(photos)  # every value identical
    assert max(out["cam"]) < 1e-5


def test_translations_are_complete():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js not installed")
    from plantdx.data.source import CLASSES, CROPS

    script = (
        f"const i = require({json.dumps(str(ROOT / 'app' / 'web' / 'i18n.js'))});"
        "console.log(JSON.stringify({en: Object.keys(i.I18N.en), az: Object.keys(i.I18N.az),"
        " crops: Object.keys(i.CROPS_AZ), conditions: i.CONDITIONS_AZ}));"
    )
    out = json.loads(
        subprocess.run([node, "-e", script], check=True, capture_output=True, text=True).stdout
    )
    assert sorted(out["en"]) == sorted(out["az"])  # every text in both languages
    assert set(out["crops"]) == set(CROPS)
    assert len(out["conditions"]) == len(CLASSES)
    healthy = [i for i, c in enumerate(CLASSES) if c.endswith("___healthy")]
    assert all(out["conditions"][i] == "Sağlam" for i in healthy)  # aligned with the class order
    html = (ROOT / "app" / "web" / "index.html").read_text(encoding="utf-8")
    keys = set(re.findall(r'data-i18n="([^"]+)"', html))
    assert keys <= set(out["en"]), keys - set(out["en"])


def test_onnx_export_matches_keras_and_gradcam(tmp_path):
    pytest.importorskip("tf2onnx")
    pytest.importorskip("onnxruntime")
    from test_models import tiny_head_model

    from plantdx import web_export as W

    keras.utils.set_random_seed(1)
    model = tiny_head_model(num_classes=38, size=32)
    images = np.random.default_rng(2).uniform(0, 255, (4, 32, 32, 3)).astype(np.float32)
    path = W.export_onnx(model, tmp_path / "model.onnx")
    check = W.check_onnx(model, path, images)
    assert check["same_top_class"] and check["probs"] < 1e-5 and check["cam"] < 1e-4
    meta = W.meta(path)
    assert len(meta["classes"]) == 38 and meta["threshold"] == 0.947
    assert meta["classes"][19]["display"] == "Pepper, bell · healthy"
    assert W.ORT_VERSION in (ROOT / "app" / "web" / "index.html").read_text(encoding="utf-8")
    assert W.ORT_VERSION in (ROOT / "app" / "web" / "app.js").read_text(encoding="utf-8")
    site = W.build_site(tmp_path / "site", model, check_images=images)
    assert site["same_top_class"]
    names = {p.name for p in (tmp_path / "site").iterdir()}
    assert {"index.html", "app.js", "imaging.js", "i18n.js", "fonts.css", "favicon.svg"} <= names
    assert {"model.onnx", "meta.json", "README.md", "fonts"} <= names
    css = (tmp_path / "site" / "fonts.css").read_text(encoding="utf-8")
    for font in re.findall(r"url\(fonts/([^)]+)\)", css):  # every font the page asks for exists
        assert (tmp_path / "site" / "fonts" / font).is_file(), font
    assert len(list((tmp_path / "site" / "examples").glob("*.jpg"))) == 7
    assert "sdk: static" in (tmp_path / "site" / "README.md").read_text(encoding="utf-8")


def test_export_refuses_another_head():
    pytest.importorskip("tf2onnx")
    from plantdx import web_export as W
    from plantdx.models.cnn import build_cnn

    with pytest.raises(ValueError, match="backbone"):
        W.head_weights(build_cnn(image_size=32, num_classes=4, filters=(4,), convs_per_block=1))
