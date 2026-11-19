"""Demo core and app (plan step 6.1, D-021). The first half needs no TensorFlow."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from plantdx import predict as P
from plantdx.data.source import CLASSES, CROPS, crop_name, display_name, is_healthy

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "gradio" / "app.py"


def load_app():
    spec = importlib.util.spec_from_file_location("demo_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probs_for(index: int, p: float) -> np.ndarray:
    probs = np.full(len(CLASSES), (1 - p) / (len(CLASSES) - 1), np.float32)
    probs[index] = p
    return probs


# ------------------------------------------------------------------------------ no TensorFlow
def test_readable_class_names():
    assert display_name("Corn_(maize)___Common_rust_") == "Corn · Common rust"
    assert display_name("Pepper,_bell___healthy") == "Pepper, bell · healthy"
    assert crop_name("Cherry_(including_sour)___Powdery_mildew") == "Cherry"
    assert is_healthy("Apple___healthy") and not is_healthy("Apple___Apple_scab")
    assert len(CROPS) == 14 and CROPS[0] == "Apple" and CROPS[-1] == "Tomato"


def test_threshold_is_the_one_chosen_on_validation():
    analysis = json.loads((ROOT / "reports" / "analysis.json").read_text())
    assert analysis["not_sure"]["val"]["threshold"] == P.NOT_SURE_THRESHOLD
    final = json.loads((ROOT / "reports" / "final_model.json").read_text())
    assert final["run_id"] == "mobilenetv2_ft_all"
    assert len(P.MODEL_SHA256) == 64 and P.MODEL_VERSION in P.MODEL_URL


def test_prediction_answers_or_abstains():
    sure = P.Prediction(np.zeros((4, 4, 3), np.uint8), probs_for(11, 0.99), None)
    assert sure.label == "Grape___Black_rot" and not sure.not_sure
    assert sure.verdict() == "Grape: Black rot (confidence 99.0%)."
    assert sure.top(3)[0] == ("Grape___Black_rot", pytest.approx(0.99))
    assert sure.overlay() is None
    healthy = P.Prediction(np.zeros((4, 4, 3), np.uint8), probs_for(19, 0.98), None)
    assert healthy.verdict() == "Pepper, bell: healthy (confidence 98.0%)."
    unsure = P.Prediction(np.zeros((4, 4, 3), np.uint8), probs_for(29, 0.5), None)
    assert unsure.not_sure and unsure.verdict().startswith("Not sure (confidence 50.0%")
    d = unsure.to_dict()
    assert d["not_sure"] and len(d["top"]) == 3 and d["label"] == "Tomato___Early_blight"
    # exactly at the threshold the model answers
    edge = P.Prediction(np.zeros((4, 4, 3), np.uint8), probs_for(0, P.NOT_SURE_THRESHOLD), None)
    assert not edge.not_sure


def test_center_square():
    a = np.arange(4 * 6 * 3).reshape(4, 6, 3)
    assert np.array_equal(P.center_square(a), a[:, 1:5])
    assert np.array_equal(P.center_square(a.transpose(1, 0, 2)), a.transpose(1, 0, 2)[1:5])
    sq = np.zeros((5, 5, 3))
    assert P.center_square(sq) is sq or np.array_equal(P.center_square(sq), sq)


def test_download_checks_the_hash(tmp_path):
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    dest = tmp_path / "cache" / "inference.keras"
    assert P.download_model(dest, src.as_uri(), good) == dest
    assert dest.read_bytes() == b"weights" and P.sha256_file(dest) == good
    bad = tmp_path / "cache2" / "inference.keras"
    with pytest.raises(RuntimeError, match="SHA-256"):
        P.download_model(bad, src.as_uri(), "0" * 64)
    assert not bad.exists() and list(bad.parent.iterdir()) == []  # no partial file left


def test_model_path_order(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANTDX_ARTIFACTS", str(tmp_path / "art"))
    monkeypatch.delenv(P.ENV_MODEL, raising=False)
    with pytest.raises(FileNotFoundError):
        P.model_path(download=False)  # nothing cached
    with pytest.raises(FileNotFoundError):
        P.model_path(tmp_path / "missing.keras")
    local = tmp_path / "local.keras"
    local.write_bytes(b"x")
    assert P.model_path(local) == local
    monkeypatch.setenv(P.ENV_MODEL, str(local))
    assert P.model_path() == local
    monkeypatch.delenv(P.ENV_MODEL)
    cached = tmp_path / "art" / "models" / P.MODEL_VERSION / "inference.keras"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"release")
    monkeypatch.setattr(P, "MODEL_SHA256", hashlib.sha256(b"release").hexdigest())
    assert P.model_path(download=False) == cached
    cached.write_bytes(b"corrupt")  # a damaged cache is not used
    with pytest.raises(FileNotFoundError):
        P.model_path(download=False)


def test_app_texts_without_a_model():
    app = load_app()

    class FakePredictor:
        def predict(self, image):
            assert image == "leaf.jpg"
            cam = np.ones((4, 4), np.float32)
            return P.Prediction(np.zeros((4, 4, 3), np.uint8), probs_for(29, 0.6), cam)

    verdict, top, overlay, info = app.respond(FakePredictor(), "leaf.jpg")
    assert verdict.startswith("### 🟡 Not sure") and "Tomato · Early blight" in verdict
    assert list(top)[0] == "Tomato · Early blight" and len(top) == 3
    assert overlay.shape == (4, 4, 3) and overlay.dtype == np.uint8
    assert "ms" in info
    assert app.respond(FakePredictor(), None)[1] is None
    sure = P.Prediction(np.zeros((4, 4, 3), np.uint8), probs_for(11, 0.99), None)
    assert app.verdict_markdown(sure).startswith("### 🔴 Grape: Black rot")
    healthy = P.Prediction(np.zeros((4, 4, 3), np.uint8), probs_for(3, 0.99), None)
    assert app.verdict_markdown(healthy).startswith("### 🟢 Apple: healthy")


def test_examples_are_small_attributed_test_photos():
    app = load_app()
    assert len(app.EXAMPLES) == 7
    attribution = (APP.parent / "examples" / "README.md").read_text(encoding="utf-8")
    test_paths = (ROOT / "splits" / "test.csv").read_text(encoding="utf-8")
    for path in app.EXAMPLES:
        assert path.stat().st_size < 30_000
        assert path.name in attribution
    for line in attribution.splitlines():
        if line.startswith("| `") and "/" in line:
            source = line.split("`")[3]
            assert source in test_paths  # every example comes from the test split


# --------------------------------------------------------------------------------- TensorFlow
def needs_tf():
    return pytest.importorskip("tensorflow")


def jpeg_bytes(rgb: np.ndarray, orientation: int = 1, fmt: str = "JPEG") -> bytes:
    im = Image.fromarray(rgb)
    exif = im.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    im.save(buf, fmt, exif=exif.tobytes(), **({"quality": 90} if fmt == "JPEG" else {}))
    return buf.getvalue()


@pytest.mark.parametrize("fmt, tolerance", [("PNG", 0), ("JPEG", 8)])
def test_exif_rotation_matches_pil(fmt, tolerance):
    needs_tf()
    # random noise: a wrong rotation or flip would differ by ~85 grey levels on average
    rgb = np.random.default_rng(0).integers(0, 256, (24, 40, 3), dtype=np.uint8)
    for orientation in range(1, 9):
        data = jpeg_bytes(rgb, orientation, fmt)
        expected = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"))
        got = P.load_rgb(data)  # decoded by TensorFlow; PIL's JPEG decoder differs slightly
        assert got.shape == expected.shape == ((24, 40, 3) if orientation < 5 else (40, 24, 3))
        assert np.abs(got.astype(int) - expected).max() <= tolerance, orientation


def test_prepared_photo_equals_the_training_input(tmp_path):
    needs_tf()
    from plantdx.data.pipeline import _decode

    rgb = np.random.default_rng(1).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    path = tmp_path / "leaf.JPG"
    Image.fromarray(rgb).save(path, quality=92)
    assert np.array_equal(P.prepare_image(path), _decode(str(path), 224).numpy())
    assert np.array_equal(P.prepare_image(path.read_bytes()), P.prepare_image(str(path)))
    wide = np.random.default_rng(2).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    assert P.prepare_image(wide).shape == (224, 224, 3)
    webp = io.BytesIO()
    Image.fromarray(rgb).save(webp, "WEBP", lossless=True)  # TensorFlow cannot decode WebP
    assert np.array_equal(P.load_rgb(webp.getvalue()), rgb)
    with pytest.raises(Exception):  # noqa: B017 - any decoder error is fine
        P.load_rgb(b"not an image")


def small_model():
    from plantdx.models.cnn import build_cnn

    return build_cnn(image_size=32, num_classes=len(CLASSES), filters=(4, 8), convs_per_block=1)


def test_predictor_matches_the_model_and_gradcam():
    needs_tf()
    from plantdx.gradcam import gradcam

    model = small_model()
    predictor = P.Predictor(model).warmup()
    photo = np.random.default_rng(3).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    pred = predictor.predict(photo)
    x = pred.image[None].astype(np.float32)
    np.testing.assert_allclose(pred.probs, model(x, training=False).numpy()[0], atol=1e-6)
    cams, probs = gradcam(model, x)  # the eager stage-5 implementation
    np.testing.assert_allclose(pred.cam, cams[0], atol=1e-5)
    assert pred.image.shape == (32, 32, 3) and pred.overlay().shape == (32, 32, 3)
    quick = predictor.predict(photo, explain=False)
    assert quick.cam is None and np.allclose(quick.probs, pred.probs, atol=1e-6)
    timings = P.benchmark(predictor, n=2, warmup=1)
    assert {"eager_keras_ms", "compiled_ms", "compiled_gradcam_ms", "photo_to_answer_ms"} <= set(
        timings
    )


def test_predictor_rejects_a_model_with_other_classes():
    needs_tf()
    from plantdx.models.cnn import build_cnn

    with pytest.raises(ValueError, match="outputs"):
        P.Predictor(build_cnn(image_size=32, num_classes=4, filters=(4,), convs_per_block=1))


def test_cli_prints_the_verdict(tmp_path, capsys):
    needs_tf()
    path = tmp_path / "model.keras"
    small_model().save(path)
    photo = tmp_path / "leaf.png"
    Image.fromarray(np.full((40, 40, 3), 120, np.uint8)).save(photo)
    P.main([str(photo), "--model", str(path), "--json"])
    line = capsys.readouterr().out.strip().splitlines()[-1]
    out = json.loads(line)
    assert out["image"] == str(photo) and len(out["top"]) == 3


def test_gradio_app_builds():
    needs_tf()
    pytest.importorskip("gradio")
    app = load_app()
    demo = app.build(P.Predictor(small_model()))
    assert demo.title == app.TITLE


def test_keras_version_check():
    P.check_keras("3.13.0")
    P.check_keras("3.15.1")
    with pytest.raises(RuntimeError, match="Keras >= 3.13"):
        P.check_keras("3.12.4")
