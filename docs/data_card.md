# Data card · PlantVillage (colour) as used in this project

**Version:** 1.6 · 20 November 2026 · splits frozen after D-010 (1.2–1.6: text and results only)
**Owner:** Yarpaq Həkimləri (ASOIU Neural Networks project, topic 06)

## 1. Summary

| | |
|---|---|
| Dataset | PlantVillage, colour variant |
| Task | Single-label classification of one leaf photo into 38 classes |
| Classes | 38 = 14 crop species × condition (26 disease classes, 12 healthy classes) |
| Images | 54,305 in the release; 53,609 used (train 36,779 · val 6,121 · test 10,709), 696 removed from train (290 + 196 + 210, section 5) |
| Image format | All 256 × 256: 54,304 RGB JPEG and 1 RGBA PNG (`reports/image_sizes.json`); decoded as 3-channel RGB and resized to 224 × 224 |
| Licence | CC BY-SA 3.0 |
| Where the images are | Downloaded by code from the Hugging Face Hub; not stored in this repository, except seven test photos as demo examples (`app/gradio/examples/`, D-021) |

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
  backgrounds, several leaves, other lighting). Field performance is out of scope and is left as
  future work (e.g. on PlantDoc; section 7).
- Background colour differs between groups of classes (grey, purple-grey, pink-brown; some images
  look pre-segmented on black). A model could learn the background instead of the symptoms. This was
  measured with Grad-CAM and the leaf masks of the segmented variant (D-020): the final model puts a
  median 70% of its heat map on the leaf, which covers 47% of the image (section 7; D-022).
- Several photos of one leaf must never be split across train and test (section 5).

## 4. Composition

Class sizes range from 152 images (Potato healthy) to about 5,500 (Orange Haunglongbing), a ratio
of about 36 : 1. Because of this imbalance **macro-F1 is the main metric**; accuracy is reported but
not used for model selection. The smallest class (Potato healthy) has only 12 validation and 32 test images, so
their per-class scores are noisy and are always reported together with the image count.

"No leaf id" is the share of images that have no entry in the official leaf map; their leaf is
inferred from the file name ("fallback", D-004). Eight classes have no verified leaf ids at all
(all four Corn classes, Grape healthy, Squash, Tomato Target Spot, Tomato mosaic virus). "Removed" are
train images taken out by the grouping rules of section 5; **Corn healthy** is hit hardest (222 of
1,162 images, 19%), because many of its file-name leaf ids are shared with the official test split,
followed by **Tomato Yellow Leaf Curl Virus** (159 of 5,357, 3%, mostly same-leaf pairs from D-010).

<!-- generated:class-table:start -->
<!-- Do not edit by hand: `python -m plantdx.data.card` rewrites this block. -->

| # | Crop | Condition | Train | Val | Test | Removed | Total | No leaf id |
|--:|---|---|--:|--:|--:|--:|--:|--:|
| 0 | Apple | Apple scab | 429 | 74 | 127 | 0 | 630 | 0% |
| 1 | Apple | Black rot | 404 | 71 | 146 | 0 | 621 | 0% |
| 2 | Apple | Cedar apple rust | 199 | 24 | 52 | 0 | 275 | 0% |
| 3 | Apple | healthy | 1,140 | 219 | 286 | 0 | 1,645 | 0% |
| 4 | Blueberry | healthy | 1,025 | 144 | 324 | 9 | 1,502 | 0% |
| 5 | Cherry (including sour) | Powdery mildew | 749 | 84 | 219 | 0 | 1,052 | 0% |
| 6 | Cherry (including sour) | healthy | 621 | 110 | 123 | 0 | 854 | 0% |
| 7 | Corn (maize) | Cercospora leaf spot Gray leaf spot | 329 | 55 | 107 | 22 | 513 | 100% |
| 8 | Corn (maize) | Common rust | 832 | 143 | 217 | 0 | 1,192 | 100% |
| 9 | Corn (maize) | Northern Leaf Blight | 618 | 110 | 220 | 37 | 985 | 100% |
| 10 | Corn (maize) | healthy | 586 | 104 | 250 | 222 | 1,162 | 100% |
| 11 | Grape | Black rot | 860 | 152 | 168 | 0 | 1,180 | 0% |
| 12 | Grape | Esca (Black Measles) | 940 | 159 | 284 | 0 | 1,383 | 0% |
| 13 | Grape | Leaf blight (Isariopsis Leaf Spot) | 752 | 120 | 204 | 0 | 1,076 | 0% |
| 14 | Grape | healthy | 272 | 41 | 100 | 10 | 423 | 100% |
| 15 | Orange | Haunglongbing (Citrus greening) | 3,879 | 592 | 980 | 56 | 5,507 | 0% |
| 16 | Peach | Bacterial spot | 1,510 | 245 | 542 | 0 | 2,297 | 0% |
| 17 | Peach | healthy | 255 | 34 | 71 | 0 | 360 | 0% |
| 18 | Pepper bell | Bacterial spot | 717 | 114 | 166 | 0 | 997 | 0% |
| 19 | Pepper bell | healthy | 1,020 | 175 | 283 | 0 | 1,478 | 0% |
| 20 | Potato | Early blight | 704 | 132 | 164 | 0 | 1,000 | 0% |
| 21 | Potato | Late blight | 680 | 108 | 212 | 0 | 1,000 | 0% |
| 22 | Potato | healthy | 108 | 12 | 32 | 0 | 152 | 0% |
| 23 | Raspberry | healthy | 174 | 35 | 162 | 0 | 371 | 0% |
| 24 | Soybean | healthy | 3,438 | 598 | 973 | 81 | 5,090 | 0% |
| 25 | Squash | Powdery mildew | 1,251 | 200 | 378 | 6 | 1,835 | 100% |
| 26 | Strawberry | Leaf scorch | 741 | 168 | 200 | 0 | 1,109 | 30% |
| 27 | Strawberry | healthy | 336 | 40 | 80 | 0 | 456 | 0% |
| 28 | Tomato | Bacterial spot | 1,408 | 247 | 436 | 36 | 2,127 | 0% |
| 29 | Tomato | Early blight | 681 | 104 | 215 | 0 | 1,000 | 0% |
| 30 | Tomato | Late blight | 1,286 | 232 | 375 | 16 | 1,909 | 52% |
| 31 | Tomato | Leaf Mold | 660 | 120 | 172 | 0 | 952 | 0% |
| 32 | Tomato | Septoria leaf spot | 1,233 | 201 | 337 | 0 | 1,771 | 44% |
| 33 | Tomato | Spider mites Two-spotted spider mite | 1,164 | 192 | 320 | 0 | 1,676 | 0% |
| 34 | Tomato | Target Spot | 955 | 161 | 288 | 0 | 1,404 | 100% |
| 35 | Tomato | Tomato Yellow Leaf Curl Virus | 3,523 | 575 | 1,100 | 159 | 5,357 | 49% |
| 36 | Tomato | Tomato mosaic virus | 255 | 49 | 66 | 3 | 373 | 100% |
| 37 | Tomato | healthy | 1,045 | 177 | 330 | 39 | 1,591 | 37% |
| | **All 38 classes** | | **36,779** | **6,121** | **10,709** | **696** | **54,305** | **24%** |

<!-- generated:class-table:end -->

### Images per leaf

PlantVillage photographed most leaves several times. Verified leaves have a median of **4 images**;
file-name ids almost always contain **one** image, i.e. for those images the other photos of the
same leaf are not grouped by the id (`reports/figures/eda_images_per_leaf.png`). D-010 recovers only
the clearest of these groups (single-image share of file-name groups 96.8% → 94.9%); most of the
hidden groups remain, see section 7.

<!-- generated:leaf-stats:start -->
<!-- Do not edit by hand: `python -m plantdx.data.card` rewrites this block. -->

| Leaf ids | Unit | Count | Images | Mean | Median | Max | With 1 image |
|---|---|--:|--:|--:|--:|--:|--:|
| verified (leaf map) | leaf id | 7,561 | 40,929 | 5.41 | 4 | 33 | 0.2% |
| verified (leaf map) | group | 7,543 | 40,929 | 5.43 | 4 | 44 | 0.2% |
| file name only | leaf id | 12,199 | 12,680 | 1.04 | 1 | 5 | 96.8% |
| file name only | group | 11,921 | 12,680 | 1.06 | 1 | 7 | 94.9% |

<!-- generated:leaf-stats:end -->

## 5. Splits

| Split | Images | Leaves | Groups | Leaf id verified | How it is made |
|---|--:|--:|--:|--:|---|
| train | 36,779 | 13,406 | 13,183 | 77% | official train minus removed groups, minus val |
| val | 6,121 | 2,236 | 2,196 | 77% | one fold of `StratifiedGroupKFold` (7 folds, `group_id`, seed 42) |
| test | 10,709 | 4,118 | 4,085 | 75% | official test, **unchanged** image for image |

Rules (details: `docs/decisions.md`):

1. **Leaf grouping (D-004).** Images of one physical leaf share a `leaf_id`. The official split shares
   227 file-name leaf ids between train and test; their 290 train images are removed.
2. **Near-identical images (D-006).** A correlation search over all images (8 rotations/flips,
   32 → 96 px) found re-shots of the same leaf under different leaf ids, including in classes whose
   ids come from the official leaf map. 84 pairs with r ≥ 0.92 join their leaves into one `group_id`;
   groups that touch test are removed from train (196 more images).
   Before: 43 near-identical pairs across splits. After: 0.
3. **Same-leaf pairs (D-010).** MobileNetV2 (ImageNet) embeddings, nearest neighbours inside each
   class, cut-off calibrated on verified leaves (cosine ≥ 0.94: precision 0.96, recall 0.004).
   565 pairs that involve a file-name image join their groups (210 more train images removed).
   Before: 253 such pairs across splits. After: 0. Rules 2 and 3 together merge 241 groups.
4. **Validation** is drawn by `group_id`, stratified by class; class shares in val differ from
   train + val by at most 0.75 percentage points.
5. `tests/test_splits.py` fails if any leaf, group or saved grouping pair crosses splits, if the
   test split differs from the official one, or if a class is missing from a split.

Files: `splits/{train,val,test}.csv` (`path, label, label_index, leaf_id, group_id, leaf_source`),
`splits/classes.json` (index → class name, fixed forever), `splits/dropped_train_overlap.csv`,
`splits/duplicate_pairs.csv`, `splits/duplicates_meta.json`, `splits/sameleaf_pairs.csv`,
`splits/sameleaf_meta.json`, `splits/summary.json`.

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
| Lab photos, plain backgrounds | Optimistic accuracy; weak transfer to field photos | State it; field test (PlantDoc) is future work |
| Class-correlated backgrounds | Possible shortcut learning | Measured with Grad-CAM and the segmented variant's leaf masks (D-020): median 0.70 of the heat map on the leaf, which covers 0.47 of the image; 97% of the images above their leaf area (D-022). The model relies on the leaf |
| 24.3% of images without verified leaf ids; most of their other photos cannot be found (D-010 recall 0.4%) | Photos of one leaf can still sit in train and test, mainly in the 8 classes without verified ids and in Tomato classes; the full-test score may be optimistic | Test metrics reported twice: full test and leaf-verified subset (8,033 images, 30 classes, leakage excluded by construction). Result: the neural models score no lower on the verified subset (final model macro-F1 0.985 vs 0.982 on the full test), so there is no sign of inflated scores; the subset lacks 8 classes, so this is indicative, not a proof |
| Near-identical re-shots below r = 0.92 and same-leaf pairs below cosine 0.94 are not grouped | Same as above | Conservative cut-offs chosen to avoid wrong merges; stated in the report |
| Strong class imbalance (36 : 1) | Accuracy hides weak classes | Macro-F1 for selection; per-class scores with counts |
| Only 14 crops, 26 diseases | Other crops or diseases are misclassified, not rejected | "Not sure" below 0.947, chosen on val (D-019): on test 96.8% answered, 99.5% of them correct; OOD detection is future work |
| Labels from the dataset authors | Label noise unknown | Six failure cases inspected with Grad-CAM (`docs/error_analysis.md`); one (apple scab predicted healthy) shows no visible symptoms and may be an early infection or a doubtful label |

## 8. Intended use

Teaching and research on leaf-image classification, and a preliminary-information demo. **Not** for
diagnosing crops in the field without an agronomist, and not for safety-critical decisions.

## 9. Reproducing

```bash
python -m plantdx.data.prepare      # rebuild splits/ from the official index files + grouping pairs
python -m plantdx.data.card         # refresh the generated tables above from splits/
pytest tests/test_splits.py tests/test_data_card.py
```

The images themselves are needed only for notebooks 01–04 (EDA, duplicate search, pipeline check,
same-leaf groups).
