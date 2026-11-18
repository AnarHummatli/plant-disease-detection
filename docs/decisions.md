# Decision log

Short records of technical decisions that differ from the obvious default or from the
original plan (v1.0, 18 Nov 2026). Each entry: context → decision → consequence.

## D-001 · Read PlantVillage files directly instead of `datasets.load_dataset`

- **Context.** The official Hugging Face release (`mohanty/PlantVillage`) is driven by a loading
  script, `plant_village.py`. Since `datasets` 4.0, loading scripts are no longer executed
  (`trust_remote_code` was removed), so `load_dataset("mohanty/PlantVillage", "color")` fails on a
  current Colab runtime. Pinning `datasets<4` would work, but would lock the project to an old
  library and its dependency tree.
- **Decision.** Download the same files the script uses with `huggingface_hub.hf_hub_download`
  (`data.zip`, `splits/color_{train,test}.txt`, `leaf_grouping/leaf-map.json`) and reimplement the
  script's leaf-id logic in `plantdx.data.source` (`leaf_identifier`, `resolve_leaf_id`), covered
  by unit tests.
- **Consequence.** Same official leaf-grouped split, no dependency on deprecated features, and
  full control over how leaf ids are derived. The share of images without a known leaf id
  ("fallback") is measured and reported in EDA.

## D-002 · Python 3.12 and the TensorFlow that Colab provides

- **Context.** Plan v1.0 said Python 3.11 and TensorFlow 2.21. Colab's default runtime moved to
  Python 3.12 in 2025, and Colab ships its own CUDA-matched TensorFlow build. TF 2.21 supports
  Python 3.10–3.13.
- **Decision.** Target Python ≥ 3.10 (CI runs 3.12). TensorFlow is an optional extra
  (`pip install -e ".[tf]"`) for local machines and is never reinstalled on Colab/Kaggle.
  The exact runtime versions are recorded by `python -m plantdx.env` at the start of each session
  and frozen into `requirements.lock` after the first full run (step 7.2).
- **Consequence.** No CUDA/TensorFlow mismatch on Colab; versions used for every reported result
  are still documented.

## D-003 · Repository language

- **Decision.** Code, README and `docs/` are written in English, because the repository is also a
  public portfolio piece and the assignment itself is in English. Presentation and Q&A follow the
  lecturer's language.

## D-004 · Official split is not fully leaf-grouped: handling of "fallback" leaves

- **Context.** The release ships `leaf_grouping/leaf-map.json`, but it covers only part of the
  images. Measured on the real `color` variant (54,305 images): 13,194 images (24.3%) have no
  entry in the map, so their leaf id is guessed from the file name ("fallback" ids). Whole
  classes are affected (e.g. Squash Powdery_mildew, Tomato Target_Spot, Corn Common_rust,
  Corn healthy and Corn Northern_Leaf_Blight are 100% fallback). Among the official train and
  test splits, 227 leaf ids occur in both; **all 227 are fallback ids** (290 train and 243 test
  images). Examples: `GH_HL Leaf 220.JPG` appears in train and test under different UUID
  prefixes. Images with a leaf-map entry show no overlap at all.
- **Decision.**
  1. Keep the official test split unchanged.
  2. Remove the 290 overlapping images from train (0.7% of train); they are listed in
     `splits/dropped_train_overlap.csv`.
  3. Carve validation out of the rest with `StratifiedGroupKFold` (7 folds, one used for
     validation: 6,184 images, 14.3% of the remaining train), grouped by leaf id, seed 42.
     Final sizes: train 37,122 / val 6,184 / test 10,709 images.
  4. Store `leaf_source` (`map` / `fallback`) per row so results can be reported separately for
     the leaf-verified subset.
  5. Measure residual duplicate leakage; first with a perceptual hash (below), then with the
     correlation search of D-006, which also changes the splits.
- **First audit (notebook `01_data_audit_eda`, dHash) — superseded, see D-006.** Every val image
  was compared with all same-class train images and every test image with all same-class
  train + val images, using a 256-bit difference hash (Hamming distance 0 = identical,
  unrelated images ≈ 128).
  - It reported 0.0% of val/test images with a same-class neighbour within distance 20.
  - A positive control on the 290 removed train images found 5 byte-identical copies
    (distance 0), so the code worked; the other ~285 shared file names are different photos.
  - **But the closest pairs, inspected by eye, were plainly the same photo at distance 30–34**
    (Tomato Late_blight and Yellow_Leaf_Curl_Virus). On flat grey backgrounds a slight blur or
    re-compression flips many hash bits, so the ≤ 20 threshold missed real copies. The 0% result
    is therefore **withdrawn**. Nearest-neighbour distances were also a little smaller for
    `fallback` than for `map` images (1st percentile 47 vs 58), consistent with such copies.
  - Files kept as a record: `reports/audit_summary.csv`, `reports/audit_distance_quantiles.csv`,
    `reports/audit_positive_control.csv`, `reports/figures/audit_*.jpg`.
- **Consequence.** The plan's assumption that the official split is fully leaf-grouped is
  replaced by explicit grouping (this entry) plus duplicate grouping (D-006). The report states
  the 24.3% fallback share, the removed images, the withdrawn first audit and the
  verified-subset metrics openly.

## D-005 · Repository size: large derived files stay out of git

- **Decision.** Photo grids are saved as JPEG (the 38-image sample grid went from 4.2 MB PNG to
  about 0.5 MB). Per-image result tables (`audit_near_duplicates.csv`, ~3 MB) are written to
  Drive and ignored by git; only small summaries are committed. `.gitattributes` forces LF
  line endings so commits from Windows do not create whitespace-only diffs.

## D-006 · Copies of the same photo are found by correlation and grouped before splitting

- **Context.** D-004's dHash audit missed copies of the same photo that differ by blur or
  re-compression (distance 30–34). A threshold high enough to catch them would also catch many
  unrelated look-alikes, so a different method is needed.
- **Decision.**
  1. Search **all** images (all splits, plus the removed ones), per class, for copies of the
     same photo: Pearson correlation of greyscale thumbnails, taking the best of the 8
     rotations/flips (D4). Stage 1 compares every same-class pair at 32×32 (candidates ≥ 0.90);
     stage 2 re-checks candidates at 96×96. Code: `plantdx.data.duplicates`, tests in
     `tests/test_duplicates.py`.
  2. The stage-2 cut-off is chosen by **looking at random example pairs in correlation bands**
     (notebook `02_duplicates_and_resplit`, section 5), not assumed; it is stored in
     `splits/duplicates_meta.json`.
  3. Confirmed pairs that join different leaf ids are saved in `splits/duplicate_pairs.csv`.
     The split builder joins their leaves into one `group_id` (union-find). Then, as in D-004,
     train groups that touch test are removed and validation is drawn by `group_id`.
     The official test split is still unchanged; validation is re-drawn, so some images move
     between train and val.
  4. `tests/test_splits.py` fails if any leaf, group or confirmed duplicate pair crosses splits.
- **Results.** *To be filled in from the notebook 02 run* (number of confirmed copies, how many
  crossed the old splits, images removed, new split sizes).
- **Limitation.** This finds copies of one photo (re-saved, blurred, resized, flipped, rotated),
  not a *different photo of the same leaf*. Test metrics are therefore also reported on the
  leaf-verified (`map`) test subset.

## Notes for the data card (step 2.6)

- **Class imbalance:** from 152 images (Potato healthy) to about 5,500 (Orange Haunglongbing);
  this is why macro-F1, not accuracy, is the main metric.
- **"copy" files:** names like `RS_NLB 4044 copy 2.jpg` are near-copies of `RS_NLB 4044.JPG`;
  `leaf_identifier` strips "copy …", so both get the same leaf id and stay in one split.
- **Background shortcut risk:** backgrounds differ between classes (grey, purple-grey, black on
  segmented-looking images). A model could learn the background instead of the symptoms; this is
  checked later with Grad-CAM and, in part B, with the segmented variant.
