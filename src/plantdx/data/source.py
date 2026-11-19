"""Access to the official PlantVillage release on the Hugging Face Hub.

Why not `datasets.load_dataset("mohanty/PlantVillage", "color")`?
The repo is built around a Python loading script (plant_village.py), and
`datasets` 4.0+ no longer runs loading scripts. So we read the same files the
script reads, directly via `huggingface_hub`:

    data.zip                      images, laid out as raw/<variant>/<class>/<file>
    splits/<variant>_train.txt    official leaf-grouped 80/20 split, one path per line
    splits/<variant>_test.txt
    leaf_grouping/leaf-map.json   image identifier -> physical leaf id(s)

`leaf_identifier` and `resolve_leaf_id` reproduce the script's leaf_id logic, so
our splits keep the same leakage guarantee as the official release.
See docs/decisions.md (D-001).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HF_REPO_ID = "mohanty/PlantVillage"
HF_REPO_TYPE = "dataset"
VARIANTS = ("color", "grayscale", "segmented")
LEAF_MAP_FILE = "leaf_grouping/leaf-map.json"
DATA_ZIP_FILE = "data.zip"

# Sizes stated on the dataset card for the `color` variant; checked in step 2.1.
EXPECTED_SIZES = {"color": {"train": 43_596, "test": 10_709}}

# The 38 classes in the canonical (ASCII-sorted) order. Indices never change:
# training, evaluation, the demo and every exported model use this order.
CLASSES: tuple[str, ...] = (
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}


def crop_name(name: str) -> str:
    """Readable crop of a class: 'Pepper,_bell___healthy' -> 'Pepper, bell'."""
    return name.partition("___")[0].split("_(")[0].replace("_", " ")


def display_name(name: str) -> str:
    """Readable class name: 'Corn_(maize)___Common_rust_' -> 'Corn · Common rust'."""
    condition = name.partition("___")[2].replace("_", " ").strip()
    return f"{crop_name(name)} · {condition}"


def is_healthy(name: str) -> bool:
    return name.endswith("___healthy")


CROPS: tuple[str, ...] = tuple(dict.fromkeys(crop_name(c) for c in CLASSES))


@dataclass(frozen=True)
class Record:
    """One image entry from an official split file."""

    path: str  # relative path inside data.zip, e.g. raw/color/Apple___healthy/x.JPG
    label: str
    crop: str
    disease: str

    @property
    def label_index(self) -> int:
        return CLASS_TO_INDEX[self.label]


def split_file(variant: str, split: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if split not in ("train", "test"):
        raise ValueError(f"unknown split {split!r}; expected 'train' or 'test'")
    return f"splits/{variant}_{split}.txt"


def parse_path(rel_path: str) -> Record | None:
    """Turn 'raw/color/<class>/<file>' into a Record; None for malformed lines."""
    rel_path = rel_path.strip()
    parts = rel_path.split("/")
    if len(parts) < 4:
        return None
    label = parts[2]
    crop, _, disease = label.partition("___")
    return Record(path=rel_path, label=label, crop=crop, disease=disease or "unknown")


def relative_path(member_path: str) -> str:
    """'raw/color/<class>/<file>' -> '<class>/<file>' (the form stored in splits/*.csv)."""
    return member_path.split("/", 2)[2]


def zip_member(variant: str, rel_path: str) -> str:
    """Inverse of `relative_path`: the member name inside data.zip."""
    return f"raw/{variant}/{rel_path}"


def read_split_index(path: str | Path) -> list[Record]:
    """Read an official split file, skipping blank and malformed lines."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rec = parse_path(line)
        if rec is not None:
            records.append(rec)
    return records


def leaf_identifier(filename: str) -> str:
    """Image identifier used as the key in leaf-map.json.

    '6c2e...___Mary_HL 6318.JPG' -> 'mary_hl 6318'
    """
    name = Path(filename).name.replace("_final_masked", "")
    if "___" in name:
        name = name.rsplit("___", 1)[1]
    name = name.split("copy", 1)[0]
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        name = name.replace(ext, "")
    return name.strip().lower()


def load_leaf_map(path: str | Path) -> dict[str, list[str]]:
    """Load leaf-map.json. Accepts a dict, a list of one-key dicts, or JSON lines."""
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line.rstrip(",")) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        return data
    merged: dict[str, list[str]] = {}
    for item in data:
        merged.update(item)
    return merged


def resolve_leaf_id(identifier: str, label: str, leaf_map: dict[str, list[str]]) -> str:
    """Physical leaf id for one image, mirroring the official loading script.

    Images with no entry in the map get a unique 'fallback_<identifier>' id, i.e.
    they are treated as their own leaf. The share of fallbacks is reported in EDA.
    """
    candidates = leaf_map.get(identifier, [])
    if len(candidates) == 1:
        return candidates[0]
    for cand in candidates:
        if label in cand:
            return cand
    return f"fallback_{identifier}"


def fetch_index_files(variant: str = "color", cache_dir: str | Path | None = None) -> dict:
    """Download the small index files (split lists + leaf map), not the images.

    Returns local paths: {"train": ..., "test": ..., "leaf_map": ...}.
    """
    from huggingface_hub import hf_hub_download

    def get(filename: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                filename=filename,
                cache_dir=cache_dir,
            )
        )

    return {
        "train": get(split_file(variant, "train")),
        "test": get(split_file(variant, "test")),
        "leaf_map": get(LEAF_MAP_FILE),
    }
