"""MobileNetV2 transfer learning (plan steps 4.1-4.2, D-016).

    input 0-255 -> [augmentation, training only] -> Rescaling to [-1, 1]
    -> MobileNetV2 backbone (ImageNet, include_top=False), always called with training=False
    -> GlobalAveragePooling -> Dropout -> Dense(num_classes, softmax, float32)

* Head only (4.1): the backbone is frozen; only the 48,678 head parameters train.
* Fine-tune (4.2): layers from `fine_tune_from` (default block_13_expand, the last 38 layers)
  become trainable. BatchNorm layers stay frozen and the backbone always runs in inference
  mode (training=False), so their ImageNet statistics are not overwritten by small batches.

The scaling x / 127.5 - 1 is keras.applications.mobilenet_v2.preprocess_input written as a
Rescaling layer, so it is saved with the model and needs no custom code to load (D-008).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

BACKBONE = "backbone"
DEFAULT_FINE_TUNE_FROM = "block_13_expand"


def set_backbone_trainable(backbone, fine_tune_from: str | None) -> int:
    """Freeze the whole backbone (None) or unfreeze from layer `fine_tune_from` on.

    BatchNorm layers stay frozen in both cases. Returns the number of trainable layers.
    """
    import keras

    if fine_tune_from is None:
        backbone.trainable = False
        return 0
    names = [layer.name for layer in backbone.layers]
    if fine_tune_from not in names:
        raise ValueError(f"layer {fine_tune_from!r} is not in the backbone")
    start = names.index(fine_tune_from)
    backbone.trainable = True
    count = 0
    for i, layer in enumerate(backbone.layers):
        trainable = i >= start and not isinstance(layer, keras.layers.BatchNormalization)
        layer.trainable = trainable
        count += bool(trainable and layer.weights)
    return count


def build_mobilenetv2(
    image_size: int = 224,
    num_classes: int = 38,
    dropout: float = 0.2,
    fine_tune_from: str | None = None,
    weights: str | None = "imagenet",
    augmentation=None,
    name: str = "mobilenetv2",
):
    import keras
    from keras import layers

    inputs = keras.Input((image_size, image_size, 3), name="image")
    x = augmentation(inputs) if augmentation is not None else inputs
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(x)
    backbone = keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=weights,
        name=BACKBONE,
    )
    set_backbone_trainable(backbone, fine_tune_from)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="probs")(x)
    return keras.Model(inputs, outputs, name=name)


def from_config(cfg: Mapping[str, Any], augmentation=None):
    m, data = cfg.get("model", {}), cfg.get("data", {})
    return build_mobilenetv2(
        image_size=data.get("image_size", 224),
        num_classes=data.get("num_classes", 38),
        dropout=m.get("dropout", 0.2),
        fine_tune_from=m.get("fine_tune_from"),
        weights=m.get("weights", "imagenet"),
        augmentation=augmentation,
    )
