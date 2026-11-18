"""CNN trained from scratch (plan step 3.3).

    input 0-255 -> [augmentation, training only] -> Rescaling(1/255)
    -> 4 blocks of [Conv3x3 -> BN -> ReLU] x convs_per_block -> MaxPool2
       (filters 32, 64, 128, 256)
    -> GlobalAveragePooling -> Dropout -> Dense(num_classes, softmax)

Two convolutions per block (VGG style) instead of one: the receptive field after four blocks
grows from 46 to 76 pixels, enough to see a whole lesion with its surroundings at 224 px, for
1.18 M parameters instead of 0.40 M (D-012). The model takes raw 0-255 pixels, like every
model in this project (D-008).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_cnn(
    image_size: int = 224,
    num_classes: int = 38,
    filters: Sequence[int] = (32, 64, 128, 256),
    convs_per_block: int = 2,
    dropout: float = 0.3,
    augmentation=None,
    name: str = "cnn_scratch",
):
    import keras
    from keras import layers

    inputs = keras.Input((image_size, image_size, 3), name="image")
    x = augmentation(inputs) if augmentation is not None else inputs
    x = layers.Rescaling(1.0 / 255, name="rescale")(x)
    for b, f in enumerate(filters, start=1):
        for c in range(1, convs_per_block + 1):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"block{b}_conv{c}")(x)
            x = layers.BatchNormalization(name=f"block{b}_bn{c}")(x)
            x = layers.ReLU(name=f"block{b}_relu{c}")(x)
        x = layers.MaxPooling2D(2, name=f"block{b}_pool")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    # float32 output even under mixed precision, for a stable softmax / loss
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="probs")(x)
    return keras.Model(inputs, outputs, name=name)


def from_config(cfg: Mapping[str, Any], augmentation=None):
    m, data = cfg.get("model", {}), cfg.get("data", {})
    return build_cnn(
        image_size=data.get("image_size", 224),
        num_classes=data.get("num_classes", 38),
        filters=tuple(m.get("filters", (32, 64, 128, 256))),
        convs_per_block=m.get("convs_per_block", 2),
        dropout=m.get("dropout", 0.3),
        augmentation=augmentation,
    )
