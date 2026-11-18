import json

import pytest

from plantdx.data.source import (
    CLASS_TO_INDEX,
    CLASSES,
    leaf_identifier,
    load_leaf_map,
    parse_path,
    read_split_index,
    resolve_leaf_id,
    split_file,
)

# Real lines from splits/color_test.txt
LINE = "raw/color/Raspberry___healthy/6c2e049f-38c9-42c5-864f-613e6e12d59e___Mary_HL 6318.JPG"


def test_classes_are_38_unique_and_sorted():
    assert len(CLASSES) == 38
    assert len(set(CLASSES)) == 38
    assert list(CLASSES) == sorted(CLASSES)
    assert CLASS_TO_INDEX["Apple___Apple_scab"] == 0
    assert CLASS_TO_INDEX["Tomato___healthy"] == 37


def test_parse_path_extracts_label_crop_disease():
    rec = parse_path(LINE)
    assert rec.label == "Raspberry___healthy"
    assert rec.crop == "Raspberry"
    assert rec.disease == "healthy"
    assert rec.label_index == CLASS_TO_INDEX["Raspberry___healthy"]


@pytest.mark.parametrize("bad", ["", "   ", "raw/color/only_three"])
def test_parse_path_rejects_malformed(bad):
    assert parse_path(bad) is None


def test_read_split_index_skips_bad_lines(tmp_path):
    f = tmp_path / "color_test.txt"
    f.write_text(f"{LINE}\n\nraw/bad\n{LINE}\n", encoding="utf-8")
    assert len(read_split_index(f)) == 2


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("6c2e049f-38c9___Mary_HL 6318.JPG", "mary_hl 6318"),
        ("abc___RS_Early.B 7557.JPG", "rs_early.b 7557"),
        ("abc___RS_Early.B 7557 copy.JPG", "rs_early.b 7557"),
        ("abc___UF.Citrus_HLB_Lab 1658_final_masked.jpg", "uf.citrus_hlb_lab 1658"),
        ("no_separator 12.png", "no_separator 12"),
    ],
)
def test_leaf_identifier(filename, expected):
    assert leaf_identifier(filename) == expected


def test_resolve_leaf_id_single_multiple_and_missing():
    leaf_map = {
        "rs_early.b 7557": ["Potato___Early_blight:::115.0"],
        "dup 1": ["Apple___healthy:::3.0", "Tomato___healthy:::9.0"],
    }
    assert resolve_leaf_id("rs_early.b 7557", "Potato___Early_blight", leaf_map) == (
        "Potato___Early_blight:::115.0"
    )
    assert resolve_leaf_id("dup 1", "Tomato___healthy", leaf_map) == "Tomato___healthy:::9.0"
    assert resolve_leaf_id("dup 1", "Grape___healthy", leaf_map) == "fallback_dup 1"
    assert resolve_leaf_id("unknown 5", "Grape___healthy", leaf_map) == "fallback_unknown 5"


def test_load_leaf_map_accepts_dict_and_list(tmp_path):
    as_dict = tmp_path / "a.json"
    as_dict.write_text(json.dumps({"k 1": ["A:::1.0"]}))
    as_list = tmp_path / "b.json"
    as_list.write_text(json.dumps([{"k 1": ["A:::1.0"]}, {"k 2": ["B:::2.0"]}]))
    assert load_leaf_map(as_dict) == {"k 1": ["A:::1.0"]}
    assert load_leaf_map(as_list) == {"k 1": ["A:::1.0"], "k 2": ["B:::2.0"]}


def test_split_file_validates_arguments():
    assert split_file("color", "train") == "splits/color_train.txt"
    with pytest.raises(ValueError):
        split_file("colour", "train")
    with pytest.raises(ValueError):
        split_file("color", "val")
