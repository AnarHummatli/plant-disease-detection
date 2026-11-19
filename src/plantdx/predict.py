"""Diagnose one leaf photo with the final model: the core of the demo (plan step 6.1, D-021).

    python -m plantdx.predict leaf.jpg [more.jpg ...]   # top-3 and the verdict
    python -m plantdx.predict --benchmark               # CPU latency per image

The final model (`mobilenetv2_ft_all`, D-017) is downloaded once from the GitHub release v1.0
and checked against its SHA-256; `PLANTDX_MODEL=<path>` or `--model` uses a local copy instead.
A photo is prepared like the training images: decoded by TensorFlow, EXIF rotation applied, cut
to its centred square (training photos are square), bilinear resize with antialiasing to
224 x 224, rounded to uint8; the model takes raw 0-255 pixels and scales them itself (D-008).
Below a top-class probability of 0.947, chosen on the validation split (D-019), the answer is
"not sure" instead of a diagnosis. Grad-CAM shows which part of the photo drove the answer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plantdx.data.source import CLASSES, crop_name, display_name, is_healthy
from plantdx.paths import artifacts_dir

MODEL_VERSION = "v1.0"
MODEL_URL = (
    "https://github.com/AnarHummatli/plant-disease-detection/releases/download/"
    f"{MODEL_VERSION}/inference.keras"
)
MODEL_SHA256 = "aad4ba57880dac9047f6baf3864dd7413c0f11fa8c3f197b303b21ed892afcad"
ENV_MODEL = "PLANTDX_MODEL"
NOT_SURE_THRESHOLD = 0.947  # D-019: smallest threshold with >= 99.5% answered accuracy on val
IMAGE_SIZE = 224


# ---------------------------------------------------------------------------------- model file
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(dest: str | Path, url: str = MODEL_URL, sha256: str = MODEL_SHA256) -> Path:
    """Download `url` to `dest` through a temporary file; refuse a file with another hash."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=120) as resp:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
        got = sha256_file(tmp)
        if got != sha256:
            raise RuntimeError(f"model download from {url} has SHA-256 {got}, expected {sha256}")
        os.chmod(tmp, 0o644)  # mkstemp creates owner-only files
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def model_path(path: str | Path | None = None, download: bool = True) -> Path:
    """`path`, else $PLANTDX_MODEL, else the verified cached release file (downloaded once)."""
    explicit = path or os.environ.get(ENV_MODEL)
    if explicit:
        explicit = Path(explicit).expanduser()
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    cached = artifacts_dir() / "models" / MODEL_VERSION / "inference.keras"
    if cached.is_file() and sha256_file(cached) == MODEL_SHA256:
        return cached
    if not download:
        raise FileNotFoundError(f"{cached} (run once with download=True)")
    print(f"downloading the final model ({MODEL_VERSION}, ~10 MB) to {cached} ...")
    return download_model(cached)


MIN_KERAS = (3, 13)  # the release model was saved by Keras 3.13+; 3.12 cannot read it (D-021)


def check_keras(version: str | None = None) -> None:
    if version is None:
        import keras

        version = keras.__version__
    parts = tuple(int(p) for p in version.split(".")[:2] if p.isdigit())
    if parts < MIN_KERAS:
        need = ".".join(map(str, MIN_KERAS))
        raise RuntimeError(
            f"the final model needs Keras >= {need} (Python >= 3.11); found Keras {version}. "
            f'Run: pip install -U "keras>={need}"'
        )


# --------------------------------------------------------------------------------- the photo
# EXIF orientation -> numpy transform of an [H, W, C] array (same result as PIL's exif_transpose)
_ORIENT = {
    2: lambda a: a[:, ::-1],
    3: lambda a: a[::-1, ::-1],
    4: lambda a: a[::-1],
    5: lambda a: a.transpose(1, 0, 2),
    6: lambda a: np.rot90(a, -1),
    7: lambda a: a[::-1, ::-1].transpose(1, 0, 2),
    8: lambda a: np.rot90(a, 1),
}


def _orientation(data: bytes) -> int:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as im:
            return int(im.getexif().get(0x0112, 1))  # read from the header, pixels not decoded
    except (UnidentifiedImageError, OSError, ValueError):
        return 1


def load_rgb(image) -> np.ndarray:
    """uint8 RGB [H, W, 3] from a path, bytes, a PIL image or an array.

    Files and bytes are decoded by TensorFlow like the training images (`pipeline._decode`):
    PIL's JPEG decoder gives slightly different pixels (up to 4 of 255 on PlantVillage), and the
    demo should see exactly what the model saw in training. The EXIF rotation of phone photos is
    applied afterwards; formats TensorFlow cannot read (e.g. WebP) fall back to PIL.
    """
    from PIL import Image, ImageOps

    if isinstance(image, Image.Image):
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            raise ValueError(f"expected an RGB image array, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr[..., :3])
    data = bytes(image) if isinstance(image, bytes | bytearray) else Path(image).read_bytes()
    try:
        import tensorflow as tf

        rgb = tf.io.decode_image(data, channels=3, expand_animations=False).numpy()
    except Exception:  # not a BMP/GIF/JPEG/PNG: let PIL try (it raises if it cannot either)
        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    fix = _ORIENT.get(_orientation(data))
    return np.ascontiguousarray(fix(rgb)) if fix else rgb


def center_square(rgb: np.ndarray) -> np.ndarray:
    """The largest centred square: every training photo is square (256 x 256), so a 4:3 phone
    photo is cropped rather than squeezed, which would stretch the leaf."""
    h, w = rgb.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return rgb[top : top + s, left : left + s]


def prepare_image(image, size: int = IMAGE_SIZE) -> np.ndarray:
    """uint8 [size, size, 3]: centred square, then resized exactly like `pipeline._decode`.

    For a square photo (all of PlantVillage) the result equals the training input pixel for pixel.
    """
    import tensorflow as tf

    rgb = center_square(load_rgb(image))
    img = tf.image.resize(rgb, (size, size), method="bilinear", antialias=True)
    return tf.cast(tf.round(tf.clip_by_value(img, 0.0, 255.0)), tf.uint8).numpy()


# ----------------------------------------------------------------------------- the prediction
@dataclass
class Prediction:
    image: np.ndarray  # the prepared uint8 photo the model saw
    probs: np.ndarray  # [38] class probabilities
    cam: np.ndarray | None  # Grad-CAM [size, size] in [0, 1] for the top class
    threshold: float = NOT_SURE_THRESHOLD

    @property
    def index(self) -> int:
        return int(np.argmax(self.probs))

    @property
    def label(self) -> str:
        return CLASSES[self.index]

    @property
    def confidence(self) -> float:
        return float(self.probs[self.index])

    @property
    def not_sure(self) -> bool:
        return self.confidence < self.threshold

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        order = np.argsort(-self.probs)[:k]
        return [(CLASSES[i], float(self.probs[i])) for i in order]

    def overlay(self, alpha: float = 0.45) -> np.ndarray | None:
        from plantdx.gradcam import overlay

        return None if self.cam is None else overlay(self.image, self.cam, alpha)

    def verdict(self) -> str:
        """One line for people: the diagnosis, or "not sure" below the threshold."""
        conf = f"confidence {self.confidence:.1%}"
        if self.not_sure:
            return (
                f"Not sure ({conf}, below {self.threshold:.1%}): please take a clearer photo of "
                "a single leaf of a supported crop."
            )
        crop = crop_name(self.label)
        if is_healthy(self.label):
            return f"{crop}: healthy ({conf})."
        return f"{crop}: {display_name(self.label).split(' · ', 1)[1]} ({conf})."

    def to_dict(self, k: int = 3) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "not_sure": self.not_sure,
            "threshold": self.threshold,
            "top": [{"class": c, "probability": p} for c, p in self.top(k)],
            "verdict": self.verdict(),
        }


class Predictor:
    """The final model behind compiled functions: one call per photo, no per-layer Python."""

    def __init__(self, model=None, threshold: float = NOT_SURE_THRESHOLD):
        import keras
        import tensorflow as tf

        from plantdx.gradcam import gradcam_fn

        if not isinstance(model, keras.Model):
            check_keras()
            model = keras.models.load_model(model_path(model), compile=False)
        if model.output_shape[-1] != len(CLASSES):
            raise ValueError(f"model has {model.output_shape[-1]} outputs, expected {len(CLASSES)}")
        self.model = model
        self.size = int(model.input_shape[1])
        self.threshold = threshold
        spec = tf.TensorSpec([None, self.size, self.size, 3], tf.float32)
        self._forward = tf.function(lambda x: model(x, training=False), input_signature=[spec])
        self._explain = gradcam_fn(model)

    def predict(self, image, explain: bool = True) -> Prediction:
        rgb = prepare_image(image, self.size)
        x = rgb[None].astype(np.float32)
        if explain:
            cam, probs = self._explain(x)
            cam = cam.numpy()[0]
        else:
            probs, cam = self._forward(x), None
        return Prediction(rgb, np.asarray(probs)[0].astype(np.float32), cam, self.threshold)

    def warmup(self) -> Predictor:
        """Trace both functions once, so the first photo is not slow."""
        blank = np.full((self.size, self.size, 3), 128, np.uint8)
        self.predict(blank, explain=False)
        self.predict(blank, explain=True)
        return self


# ---------------------------------------------------------------------------------- latency
def benchmark(predictor: Predictor, n: int = 50, warmup: int = 5) -> dict[str, float]:
    """Median CPU milliseconds per photo (batch 1) for each way of running the model."""
    import tensorflow as tf
    from PIL import Image

    from plantdx.evaluate import median_ms

    rng = np.random.default_rng(0)
    x = tf.constant(rng.uniform(0, 255, (1, predictor.size, predictor.size, 3)), tf.float32)
    photo = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
    model = predictor.model
    with tf.device("/CPU:0"):
        timings = {
            # the protocol of the one-time test evaluation (results.csv, cpu_ms): eager Keras
            "eager_keras_ms": median_ms(lambda: model(x, training=False), n, warmup),
            "compiled_ms": median_ms(lambda: predictor._forward(x).numpy(), n, warmup),
            "compiled_gradcam_ms": median_ms(lambda: predictor._explain(x)[1].numpy(), n, warmup),
            "photo_to_answer_ms": median_ms(lambda: predictor.predict(photo), n, warmup),
        }
    timings["cpus"] = float(os.cpu_count() or 0)
    return {k: round(v, 1) for k, v in timings.items()}


# -------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("images", nargs="*", help="leaf photos (JPG / PNG)")
    parser.add_argument("--model", help=f"local inference.keras (default: release {MODEL_VERSION})")
    parser.add_argument("--no-cam", action="store_true", help="skip Grad-CAM")
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--benchmark", action="store_true", help="measure CPU latency")
    args = parser.parse_args(argv)
    if not args.images and not args.benchmark:
        parser.error("give at least one image or --benchmark")

    t0 = time.perf_counter()
    predictor = Predictor(args.model).warmup()
    print(f"model ready in {time.perf_counter() - t0:.1f} s")
    for path in args.images:
        pred = predictor.predict(path, explain=not args.no_cam)
        if args.json:
            print(json.dumps({"image": path, **pred.to_dict()}))
            continue
        print(f"\n{path}\n  {pred.verdict()}")
        for name, p in pred.top():
            print(f"  {p:7.2%}  {display_name(name)}")
    if args.benchmark:
        print(json.dumps(benchmark(predictor), indent=2))


if __name__ == "__main__":
    main()
