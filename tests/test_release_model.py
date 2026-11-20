"""The released final model on the seven demo examples (D-021). Downloads ~10 MB once.

Unit tests use toy models; this one loads the real `inference.keras` from release v1.0, so it
fails if a dependency change breaks loading the file (e.g. Keras 3.12 cannot read it) or if a
change to the photo preparation alters its answers. Skipped without TensorFlow or network;
`PLANTDX_MODEL=<path>` uses a local copy.
"""

import os
import urllib.error
from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from plantdx import predict as P  # noqa: E402
from plantdx import reproduce as REPRO  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "app" / "gradio" / "examples"
EXPECTED = REPRO.EXAMPLES  # one list for this test and the reproduction check
NOT_SURE = REPRO.EXAMPLE_NOT_SURE


@pytest.fixture(scope="module")
def predictor(request, tmp_path_factory):
    path = os.environ.get(P.ENV_MODEL)
    if not path:
        cache = getattr(request.config, "cache", None)  # .pytest_cache: download once
        folder = cache.mkdir("release-model") if cache else tmp_path_factory.mktemp("model")
        path = Path(folder) / f"inference-{P.MODEL_VERSION}.keras"
        if not (path.is_file() and P.sha256_file(path) == P.MODEL_SHA256):
            try:
                P.download_model(path)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
                pytest.skip(f"release model not reachable: {err}")
    return P.Predictor(path)


def test_release_model_answers_the_examples(predictor):
    assert sorted(EXPECTED) == sorted(p.name for p in EXAMPLES.glob("*.jpg"))
    for name, label in EXPECTED.items():
        pred = predictor.predict(EXAMPLES / name, explain=False)
        assert pred.label == label, name
        if name == NOT_SURE:
            assert pred.not_sure and 0.45 < pred.confidence < 0.6, pred.confidence
        else:
            assert not pred.not_sure and pred.confidence > 0.99, (name, pred.confidence)


def test_release_model_heat_map_lies_on_the_leaf(predictor):
    pred = predictor.predict(EXAMPLES / "grape_black_rot.jpg")
    leaf = pred.image.max(axis=-1).astype(int) - pred.image.min(axis=-1) > 40  # green, not grey
    on_leaf = float((pred.cam * leaf).sum() / pred.cam.sum())
    assert on_leaf > leaf.mean()  # more heat on the leaf than an even spread would give
