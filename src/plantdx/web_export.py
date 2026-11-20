"""Build the browser demo: the final model in the visitor's browser (plan step 6.1, D-022).

    pip install -e ".[tf,web]"
    python -m plantdx.web_export --out build/web

The site is static (no server): ONNX Runtime Web runs `model.onnx` on the visitor's device, so
it can be hosted for free (a static Hugging Face Space, GitHub Pages) and photos never leave the
device. Written to `--out`:

* `model.onnx`: the release model with two outputs, the 38 probabilities and the 7 x 7 Grad-CAM
  map of the top class, so the browser needs no gradients;
* `meta.json`: classes, "not sure" threshold, image size, checksums;
* the page from `app/web/` (`index.html`, `app.js`, `imaging.js`, `i18n.js`, self-hosted fonts,
  icon, preview image) and `examples/` (from `app/gradio/examples/`);
* `README.md`: the Hugging Face Space card (`sdk: static`).

Grad-CAM in closed form. The head after the backbone is global average pooling, dropout (a no-op
at inference) and a softmax layer with weights W [1280, 38]. The gradient of the top-class logit
with respect to every position of the feature map is W[:, c] / (h w) (D-022), so Grad-CAM is
relu(sum_k W[k, c] f_k) / (h w) and the browser needs no backpropagation. The export checks that
the result equals `gradcam.gradcam_fn` on the same images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from plantdx.data.source import CLASSES, CROPS, crop_name, display_name, is_healthy
from plantdx.paths import repo_root
from plantdx.predict import IMAGE_SIZE, MODEL_SHA256, MODEL_VERSION, NOT_SURE_THRESHOLD

ORT_VERSION = "1.31.1"  # onnxruntime-web loaded by app/web/index.html; keep the two in step
SITE_FILES = ("index.html", "app.js", "imaging.js", "i18n.js", "fonts.css", "favicon.svg", "og.png")
HEAD = ("gap", "dropout", "probs")  # layers after "backbone" in every MobileNetV2 of this project


def head_weights(model) -> np.ndarray:
    """Softmax weights W [features, classes]; refuses a head that is not GAP -> dropout -> dense."""
    names = [layer.name for layer in model.layers if layer.__class__.__name__ != "InputLayer"]
    if "backbone" not in names or tuple(names[names.index("backbone") + 1 :]) != HEAD:
        raise ValueError(f"expected layers ... backbone, {', '.join(HEAD)}; got {names}")
    dense = model.get_layer("probs")
    if dense.activation.__name__ != "softmax":
        raise ValueError("the last layer must be a softmax")
    return np.asarray(dense.get_weights()[0], np.float32)


def export_function(model):
    """tf.function image [n, s, s, 3] (0-255) -> {"probs": [n, 38], "cam": [n, h, w]}."""
    import tensorflow as tf

    from plantdx.gradcam import split_model

    w_np = head_weights(model)  # [k, classes]; a constant inside the graph, not a captured input
    features, head = split_model(model, "backbone")
    size = int(model.input_shape[1])

    @tf.function(input_signature=[tf.TensorSpec([None, size, size, 3], tf.float32, name="image")])
    def run(image):
        w = tf.constant(w_np)
        f = features(image)  # [n, h, w, k]
        p = head(f)  # [n, classes]
        w_top = tf.gather(tf.transpose(w), tf.argmax(p, axis=1))  # [n, k]
        hw = tf.cast(tf.shape(f)[1] * tf.shape(f)[2], tf.float32)
        cam = tf.nn.relu(tf.einsum("nhwk,nk->nhw", f, w_top / hw))  # d logit / d f = W[:, c] / hw
        return {"probs": p, "cam": cam}

    return run


def export_onnx(model, path: str | Path, opset: int = 17) -> Path:
    import tf2onnx

    fn = export_function(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_function(
        fn, input_signature=fn.input_signature, opset=opset, output_path=str(path)
    )
    return path


def check_onnx(model, path: str | Path, images: np.ndarray) -> dict[str, float]:
    """Largest differences between the ONNX model and Keras / the demo's Grad-CAM."""
    import onnxruntime as ort
    import tensorflow as tf

    from plantdx.gradcam import gradcam_fn

    images = np.asarray(images, np.float32)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    out = dict(zip(names, session.run(None, {"image": images}), strict=True))
    cams_ref, probs_ref = gradcam_fn(model)(tf.constant(images))
    h, w = images.shape[1:3]
    cam = tf.image.resize(out["cam"][..., None], (h, w), method="bilinear")[..., 0]
    cam = tf.math.divide_no_nan(cam, tf.reduce_max(cam, axis=(1, 2), keepdims=True)).numpy()
    return {
        "probs": float(np.abs(out["probs"] - probs_ref.numpy()).max()),
        "cam": float(np.abs(cam - cams_ref.numpy()).max()),
        "same_top_class": bool((out["probs"].argmax(1) == probs_ref.numpy().argmax(1)).all()),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meta(onnx_path: Path) -> dict:
    return {
        "model_version": MODEL_VERSION,
        "keras_sha256": MODEL_SHA256,
        "onnx_sha256": _sha256(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "image_size": IMAGE_SIZE,
        "threshold": NOT_SURE_THRESHOLD,
        "ort_version": ORT_VERSION,
        "crops": list(CROPS),
        "classes": [
            {
                "name": c,
                "display": display_name(c),
                "crop": crop_name(c),
                "condition": display_name(c).split(" · ", 1)[1],
                "healthy": is_healthy(c),
            }
            for c in CLASSES
        ],
    }


def build_site(out: str | Path, model=None, check_images: np.ndarray | None = None) -> dict:
    """Write the static site to `out`; returns the export check (raises if it fails)."""
    import keras

    from plantdx.predict import Predictor, prepare_image

    out = Path(out)
    root = repo_root()
    web, examples = root / "app" / "web", root / "app" / "gradio" / "examples"
    if not isinstance(model, keras.Model):
        model = Predictor(model).model  # the verified release model (or a local path)
    onnx_path = export_onnx(model, out / "model.onnx")
    if check_images is None:
        check_images = np.stack([prepare_image(p) for p in sorted(examples.glob("*.jpg"))])
    check = check_onnx(model, onnx_path, check_images)
    if not (check["same_top_class"] and check["probs"] < 1e-4 and check["cam"] < 1e-3):
        raise RuntimeError(f"ONNX export differs from the Keras model: {check}")
    (out / "meta.json").write_text(json.dumps(meta(onnx_path), indent=1), encoding="utf-8")
    for name in SITE_FILES:
        if (web / name).is_file():  # og.png is optional
            shutil.copyfile(web / name, out / name)
    shutil.copytree(web / "fonts", out / "fonts", dirs_exist_ok=True)
    shutil.copyfile(web / "SPACE.md", out / "README.md")
    shutil.copytree(examples, out / "examples", dirs_exist_ok=True)
    return check


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="build/web", help="output folder (default build/web)")
    parser.add_argument("--model", help=f"local inference.keras (default: release {MODEL_VERSION})")
    args = parser.parse_args(argv)
    check = build_site(args.out, args.model)
    print(f"ONNX vs Keras: probabilities {check['probs']:.1e}, Grad-CAM {check['cam']:.1e}")
    print(f"site written to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
