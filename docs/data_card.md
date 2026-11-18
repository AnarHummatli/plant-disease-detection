# Data card · PlantVillage (colour) as used in this project

**Version:** 1.0 · 18 November 2026 · splits frozen at commit after D-006
**Owner:** Yarpaq Həkimləri (ASOIU Neural Networks project, topic 06)

## 1. Summary

| | |
|---|---|
| Dataset | PlantVillage, colour variant |
| Task | Single-label classification of one leaf photo into 38 classes |
| Classes | 38 = 14 crop species × condition (26 disease classes, 12 healthy classes) |
| Images | 54,305 in the release; 54,015 used (train 36,963 · val 6,147 · test 10,709), 290 + 196 removed from train |
| Image format | RGB JPEG; PlantVillage distributes them at 256 × 256; the pipeline resizes to 224 × 224 |
| Licence | CC BY-SA 3.0 |
| Where the images are | Downloaded by code from the Hugging Face Hub; never stored in this repository |

## 2. Source and citation

- Hugging Face dataset [`mohanty/PlantVillage`](https://huggingface.co/datasets/mohanty/PlantVillage):
  `data.zip` (images), `splits/color_{train,test}.txt` (official split),
  `leaf_grouping/leaf-map.json` (physical-leaf ids). Loaded with `plantdx.data.source` (D-001).
- D. P. Hughes and M. Salathé, "An open access repository of images on plant health to enable the
  development of mobile disease diagnostics," arXiv:1511.08060, 2015.
- S. P. Mohanty, D. P. Hughes and M. Salathé, "Using deep learning for image-based plant disease
  detection," *Frontiers in Plant Science*, 7:1419, 2016.

**Licence obligations (CC BY-SA 3.0).** The dataset is credited above and in the README. The files in
`splits/` list image names and our grouping; they are derived metadata and are shared under the same
licence. Images are not redistributed.

## 3. How the data was collected (what that implies)

Each photo shows **one detached leaf on a plain background** under controlled, lab-like conditions,
and most leaves were photographed several times. Consequences for this project:

- A model can reach very high accuracy here and still do much worse on field photos (cluttered
  backgrounds, several leaves, other lighting). Field performance is out of scope for part A and is
  measured later on PlantDoc (plan step 10.1).
- Background colour differs between groups of classes (grey, purple-grey, pink-brown; some images
  look pre-segmented on black). A model could learn the background instead of the symptoms; this is
  checked with Grad-CAM (plan step 5.5) and, in part B, with the segmented variant.
- Several photos of one leaf must never be split across train and test (section 5).

## 4. Composition

Class sizes range from 152 images (Potato healthy) to about 5,500 (Orange Haunglongbing), a ratio
of about 36 : 1. Because of this imbalance **macro-F1 is the main metric**; accuracy is reported but
not used for model selection. The smallest classes have only 16 validation and 32 test images, so
their per-class scores are noisy and are always reported together with the image count.

"No leaf id" is the share of images that have no entry in the official leaf map; their leaf is
inferred from the file name ("fallback", D-004). Eight classes have no verified leaf ids at all
(all four Corn classes, Grape healthy, Squash, Tomato Target Spot, Tomato mosaic virus). "Removed" are
train images taken out by the grouping rules of section 5; **Corn healthy** is hit hardest (220 of
1,162 images, 19%), because many of its file-name leaf ids are shared with the official test split.

<!-- generated:class-table:start -->
<!-- Do not edit by hand: `python -m plantdx.data.card` rewrites this block. -->

| # | Crop | Condition | Train | Val | Test | Removed | Total | No leaf id |
|--:|---|---|--:|--:|--:|--:|--:|--:|
| 0 | Apple | Apple scab | 446 | 57 | 127 | 0 | 630 | 0% |
| 1 | Apple | Black rot | 391 | 84 | 146 | 0 | 621 | 0% |
| 2 | Apple | Cedar apple rust | 199 | 24 | 52 | 0 | 275 | 0% |
| 3 | Apple | healthy | 1,145 | 214 | 286 | 0 | 1,645 | 0% |
| 4 | Blueberry | healthy | 1,011 | 158 | 324 | 9 | 1,502 | 0% |
| 5 | Cherry (including sour) | Powdery mildew | 708 | 125 | 219 | 0 | 1,052 | 0% |
| 6 | Cherry (including sour) | healthy | 621 | 110 | 123 | 0 | 854 | 0% |
| 7 | Corn (maize) | Cercospora leaf spot Gray leaf spot | 333 | 51 | 107 | 22 | 513 | 100% |
| 8 | Corn (maize) | Common rust | 833 | 142 | 217 | 0 | 1,192 | 100% |
| 9 | Corn (maize) | Northern Leaf Blight | 625 | 103 | 220 | 37 | 985 | 100% |
| 10 | Corn (maize) | healthy | 581 | 111 | 250 | 220 | 1,162 | 100% |
| 11 | Grape | Black rot | 876 | 136 | 168 | 0 | 1,180 | 0% |
| 12 | Grape | Esca (Black Measles) | 944 | 155 | 284 | 0 | 1,383 | 0% |
| 13 | Grape | Leaf blight (Isariopsis Leaf Spot) | 736 | 136 | 204 | 0 | 1,076 | 0% |
| 14 | Grape | healthy | 277 | 46 | 100 | 0 | 423 | 100% |
| 15 | Orange | Haunglongbing (Citrus greening) | 3,844 | 627 | 980 | 56 | 5,507 | 0% |
| 16 | Peach | Bacterial spot | 1,517 | 238 | 542 | 0 | 2,297 | 0% |
| 17 | Peach | healthy | 249 | 40 | 71 | 0 | 360 | 0% |
| 18 | Pepper bell | Bacterial spot | 736 | 95 | 166 | 0 | 997 | 0% |
| 19 | Pepper bell | healthy | 1,027 | 168 | 283 | 0 | 1,478 | 0% |
| 20 | Potato | Early blight | 724 | 112 | 164 | 0 | 1,000 | 0% |
| 21 | Potato | Late blight | 668 | 120 | 212 | 0 | 1,000 | 0% |
| 22 | Potato | healthy | 104 | 16 | 32 | 0 | 152 | 0% |
| 23 | Raspberry | healthy | 175 | 34 | 162 | 0 | 371 | 0% |
| 24 | Soybean | healthy | 3,433 | 603 | 973 | 81 | 5,090 | 0% |
| 25 | Squash | Powdery mildew | 1,244 | 207 | 378 | 6 | 1,835 | 100% |
| 26 | Strawberry | Leaf scorch | 782 | 127 | 200 | 0 | 1,109 | 30% |
| 27 | Strawberry | healthy | 320 | 56 | 80 | 0 | 456 | 0% |
| 28 | Tomato | Bacterial spot | 1,415 | 240 | 436 | 36 | 2,127 | 0% |
| 29 | Tomato | Early blight | 677 | 108 | 215 | 0 | 1,000 | 0% |
| 30 | Tomato | Late blight | 1,320 | 208 | 375 | 6 | 1,909 | 52% |
| 31 | Tomato | Leaf Mold | 664 | 116 | 172 | 0 | 952 | 0% |
| 32 | Tomato | Septoria leaf spot | 1,233 | 201 | 337 | 0 | 1,771 | 44% |
| 33 | Tomato | Spider mites Two-spotted spider mite | 1,164 | 192 | 320 | 0 | 1,676 | 0% |
| 34 | Tomato | Target Spot | 955 | 161 | 288 | 0 | 1,404 | 100% |
| 35 | Tomato | Tomato Yellow Leaf Curl Virus | 3,657 | 598 | 1,100 | 2 | 5,357 | 49% |
| 36 | Tomato | Tomato mosaic virus | 262 | 45 | 66 | 0 | 373 | 100% |
| 37 | Tomato | healthy | 1,067 | 183 | 330 | 11 | 1,591 | 37% |
| | **All 38 classes** | | **36,963** | **6,147** | **10,709** | **486** | **54,305** | **24%** |

<!-- generated:class-table:end -->

## 5. Splits

| Split | Images | Leaves | Groups | Leaf id verified | How it is made |
|---|--:|--:|--:|--:|---|
| train | 36,963 | 13,586 | 13,567 | 76% | official train minus removed groups, minus val |
| val | 6,147 | 2,265 | 2,261 | 76% | one fold of `StratifiedGroupKFold` (7 folds, `group_id`, seed 42) |
| test | 10,709 | 4,118 | 4,112 | 75% | official test, **unchanged** image for image |

Rules (details: `docs/decisions.md`):

1. **Leaf grouping (D-004).** Images of one physical leaf share a `leaf_id`. The official split shares
   227 file-name leaf ids between train and test; their 290 train images are removed.
2. **Near-identical images (D-006).** A correlation search over all images (8 rotations/flips,
   32 → 96 px) found re-shots of the same leaf under different leaf ids, including in classes whose
   ids come from the official leaf map. 84 pairs with r ≥ 0.92 join their leaves into one `group_id`
   (27 merged groups); groups that touch test are removed from train (196 more images).
   Before: 43 near-identical pairs across splits. After: 0.
3. **Validation** is drawn by `group_id`, stratified by class; class shares in val differ from
   train + val by at most 0.45 percentage points.
4. `tests/test_splits.py` fails if any leaf, group or saved near-identical pair crosses splits, if the
   test split differs from the official one, or if a class is missing from a split.

Files: `splits/{train,val,test}.csv` (`path, label, label_index, leaf_id, group_id, leaf_source`),
`splits/classes.json` (index → class name, fixed forever), `splits/dropped_train_overlap.csv`,
`splits/duplicate_pairs.csv`, `splits/duplicates_meta.json`, `splits/summary.json`.

## 6. Preprocessing and augmentation (`plantdx.data.pipeline`, D-008)

- Decode as RGB → resize to 224 × 224 (bilinear, antialiased) → float32 in 0–255. Model-specific
  scaling happens inside each model.
- Training only: random horizontal and vertical flip, rotation up to ±20°, zoom ±15%, brightness
  and contrast ±15% (`configs/base.yaml`, `augment:`), applied on the GPU as the first layers of
  the model. Example: `reports/figures/aug_examples.jpg`.
- Decoded images are cached as uint8 (val/test in memory, train on local disk).
- Labels are one-hot over the 38 classes in `classes.json` order.

## 7. Known limitations

| Limitation | Effect | What we do |
|---|---|---|
| Lab photos, plain backgrounds | Optimistic accuracy; weak transfer to field photos | State it; field test in part B (PlantDoc) |
| Class-correlated backgrounds | Possible shortcut learning | Grad-CAM check; segmented variant in part B |
| 24.3% of images without verified leaf ids | Some photos of one leaf may still sit in two splits | Grouping + duplicate search; report test metrics also on the verified subset (8,033 images) |
| Re-shots below r = 0.92 and other-angle photos of one leaf are not detected | Small residual leakage may remain | Stated in the report; conservative cut-off |
| Strong class imbalance (36 : 1) | Accuracy hides weak classes | Macro-F1 for selection; per-class scores with counts |
| Only 14 crops, 26 diseases | Other crops or diseases are misclassified, not rejected | "Not sure" threshold (plan 5.6); OOD detection in part B |
| Labels from the dataset authors | Label noise unknown | Failure cases are inspected by eye (plan 5.4) |

## 8. Intended use

Teaching and research on leaf-image classification, and a preliminary-information demo. **Not** for
diagnosing crops in the field without an agronomist, and not for safety-critical decisions.

## 9. Reproducing

```bash
python -m plantdx.data.prepare      # rebuild splits/ from the official index files + duplicate pairs
python -m plantdx.data.card         # refresh the class table above from splits/
pytest tests/test_splits.py tests/test_data_card.py
```

The images themselves are needed only for notebooks 01–03 (EDA, duplicate search, pipeline check).
