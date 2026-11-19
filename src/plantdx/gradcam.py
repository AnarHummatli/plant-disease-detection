"""Grad-CAM: where the model looks (plan step 5.5, D-013, D-020; demo: D-021).

Grad-CAM (Selvaraju et al., 2017) weights the feature maps of a late convolutional layer by the
gradient of the class score and keeps the positive part: a coarse heat map of the image regions
that raised that class's probability.

Works for every model in plantdx.models, because their inference versions (inference.keras) are
plain chains of layers: the model is split at `layer` into a feature part and a head part.

* MobileNetV2: layer "backbone" (7 x 7 x 1280 feature maps for a 224 px input);
* CNN from scratch: layer "block4_relu2" (14 x 14 x 256 before the last pooling).

The heat map is computed at feature resolution and resized to the image, so it marks regions of
about 32 px (MobileNetV2), not single pixels.
"""

from __future__ import annotations

import numpy as np

DEFAULT_LAYERS = ("backbone", "block4_relu2")


def default_layer(model) -> str:
    """A known feature layer, else the last layer with a 4-D (feature map) output."""
    names = [layer.name for layer in model.layers]
    for name in DEFAULT_LAYERS:
        if name in names:
            return name
    for layer in reversed(model.layers):
        shape = getattr(layer, "output", None)
        if shape is not None and len(shape.shape) == 4 and layer.__class__.__name__ != "InputLayer":
            if not layer.__class__.__name__.startswith(("MaxPooling", "Rescaling")):
                return layer.name
    raise ValueError("no feature-map layer found; pass layer=")


def split_model(model, layer: str):
    """(features, head): two callables with head(features(x)) == model(x) for chain models."""
    layers = [lyr for lyr in model.layers if lyr.__class__.__name__ != "InputLayer"]
    names = [lyr.name for lyr in layers]
    if layer not in names:
        raise ValueError(f"layer {layer!r} not in model")
    k = names.index(layer)

    def features(x):
        for lyr in layers[: k + 1]:
            x = lyr(x, training=False)
        return x

    def head(f):
        for lyr in layers[k + 1 :]:
            f = lyr(f, training=False)
        return f

    return features, head


def gradcam(
    model, images: np.ndarray, layer: str | None = None, class_index=None, batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Heat maps [n, H, W] in [0, 1] and probabilities [n, classes] for float images 0-255.

    `class_index`: None = the predicted class of each image; an int or an array of length n
    explains that class instead (e.g. the true class of a misclassified image).
    """
    import tensorflow as tf

    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 4:
        raise ValueError(f"expected [n, H, W, 3], got {images.shape}")
    n, h, w = images.shape[:3]
    features, head = split_model(model, layer or default_layer(model))
    if class_index is not None:
        class_index = np.broadcast_to(np.asarray(class_index, dtype=np.int64), (n,))
    cams, probs = [], []
    for s in range(0, n, batch_size):
        x = tf.constant(images[s : s + batch_size])
        with tf.GradientTape() as tape:
            f = features(x)
            tape.watch(f)
            p = head(f)
            idx = (
                tf.argmax(p, axis=1)
                if class_index is None
                else tf.constant(class_index[s : s + batch_size])
            )
            score = tf.gather(p, idx, axis=1, batch_dims=1)
        cam = _cam_from(f, tape.gradient(score, f), h, w)
        cams.append(cam.numpy())
        probs.append(p.numpy())
    return np.concatenate(cams), np.concatenate(probs)


def _cam_from(f, grads, h: int, w: int):
    """Grad-CAM maps [n, h, w] in [0, 1] from feature maps and the class-score gradients."""
    import tensorflow as tf

    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * f, axis=-1))[..., None]
    cam = tf.image.resize(cam, (h, w), method="bilinear")[..., 0]
    # scale each map to max 1; a map with no positive evidence stays all zero
    return tf.math.divide_no_nan(cam, tf.reduce_max(cam, axis=(1, 2), keepdims=True))


def gradcam_fn(model, layer: str | None = None):
    """Compiled `x -> (cams, probs)` for the predicted class (the demo's fast path, D-021).

    Same maths as `gradcam`, traced once with `tf.function`, so a single photo costs one
    forward and one backward pass instead of a Python loop over the layers.
    """
    import tensorflow as tf

    features, head = split_model(model, layer or default_layer(model))
    h, w = model.input_shape[1:3]

    @tf.function(input_signature=[tf.TensorSpec([None, h, w, 3], tf.float32)])
    def run(x):
        with tf.GradientTape() as tape:
            f = features(x)
            tape.watch(f)
            p = head(f)
            score = tf.gather(p, tf.argmax(p, axis=1), axis=1, batch_dims=1)
        return _cam_from(f, tape.gradient(score, f), h, w), p

    return run


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """uint8 RGB image with the heat map (jet colours) blended on top."""
    import matplotlib

    heat = matplotlib.colormaps["jet"](np.clip(cam, 0, 1))[..., :3]
    out = (1 - alpha) * (np.asarray(rgb, np.float32) / 255.0) + alpha * heat
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def leaf_mask(segmented_rgb: np.ndarray, threshold: int = 25) -> np.ndarray:
    """Leaf pixels of a PlantVillage "segmented" image (background is near black)."""
    return np.asarray(segmented_rgb).max(axis=-1) > threshold


def cam_on_mask(cam: np.ndarray, mask: np.ndarray) -> float:
    """Share of the heat-map mass that falls on the mask (1.0 = all on the leaf)."""
    total = float(cam.sum())
    return float((cam * mask).sum() / total) if total > 0 else float("nan")
