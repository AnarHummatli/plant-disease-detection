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
  prefixes. Images with a leaf-map entry share no leaf id between train and test (they can
  still share re-shots under different ids; see D-006).
- **Decision.**
  1. Keep the official test split unchanged.
  2. Remove the 290 overlapping images from train (0.7% of train); they are listed in
     `splits/dropped_train_overlap.csv`.
  3. Carve validation out of the rest with `StratifiedGroupKFold` (7 folds, one used for
     validation, about 14.3% of the remaining train), grouped by leaf id, seed 42.
  4. Store `leaf_source` (`map` / `fallback`) per row so results can be reported separately for
     the leaf-verified subset.
  5. Measure residual duplicate leakage; first with a perceptual hash (below), then with the
     correlation search of D-006, which also changes the splits (final sizes are in D-006).
- **First audit (notebook `01_data_audit_eda`, dHash) — superseded, see D-006.** Every val image
  was compared with all same-class train images and every test image with all same-class
  train + val images, using a 256-bit difference hash (Hamming distance 0 = identical,
  unrelated images ≈ 128).
  - It reported 0.0% of val/test images with a same-class neighbour within distance 20.
  - A positive control on the 290 removed train images found 5 byte-identical copies
    (distance 0), so the code worked; the other ~285 shared file names are different photos.
  - **But the closest pairs, inspected by eye, looked like the same leaf at distance 30–34**
    (Tomato Late_blight and Yellow_Leaf_Curl_Virus; D-006 later showed these are re-shots). On flat grey backgrounds a slight blur or
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

## D-006 · Near-identical images are found by correlation and grouped before splitting

- **Context.** D-004's dHash audit missed near-identical images (distance 30–34). A threshold high
  enough to catch them would also catch many unrelated look-alikes, so a different method is
  needed.
- **Method** (notebook `02_duplicates_and_resplit`, code `plantdx.data.duplicates`, tests in
  `tests/test_duplicates.py`). All 54,305 images (every split plus the removed ones) are searched
  per class: Pearson correlation of greyscale thumbnails, best of the 8 rotations/flips (D4).
  Stage 1 compares every same-class pair at 32×32 (candidates ≥ 0.90); stage 2 re-checks the
  candidates at 96×96. 26,513 candidate pairs.
- **What inspection showed** (all 28 pairs ≥ 0.93 and the 12 highest below were checked by eye).
  - No pair with different leaf ids reaches 0.97 (maximum 0.954): there are **no plain copies**
    of one photo under different leaf ids. Exact copies (r ≈ 1.0) exist only *within* a leaf id,
    e.g. the "copy" files.
  - What does cross the splits are **re-shots of the same leaf** (same pose, slightly different
    light or position): about 16 of the 28 pairs ≥ 0.93, mostly Tomato healthy,
    Yellow_Leaf_Curl_Virus and Late_blight, and some just below 0.93.
  - Different leaves that merely look alike reach up to 0.954 (Orange, Soybean,
    Bacterial_spot). **No cut-off separates the two cleanly.**
  - Some re-shots are in Soybean healthy, a class whose leaf ids all come from the official leaf
    map: the official leaf map is not perfect either.
- **Choosing the cut-off.** Merging two different leaves by mistake is harmless (they end up on
  the same side); missing a re-shot is leakage. So the cut-off is the **lowest one before
  look-alikes chain together** through union-find (`reports/duplicates_threshold_cost.csv`):

  | cut-off | pairs (different leaf ids) | across splits | train/val images to remove | largest group |
  |---:|---:|---:|---:|---:|
  | 0.90 | 851 | 388 | 1,743 | 1,024 |
  | 0.91 | 298 | 129 | 687 | 236 |
  | **0.92** | **84** | **43** | **196** | **63** |
  | 0.93 | 28 | 11 | 44 | 41 |

  At 0.91 and below, look-alike Orange and Soybean leaves chain into groups of hundreds of
  images. **0.92** keeps the largest group within about twice the largest single leaf (33
  images) and removes 0.45% of train/val. Stored in `splits/duplicates_meta.json`.
- **Decision.** The 84 pairs ≥ 0.92 that join different leaf ids are saved in
  `splits/duplicate_pairs.csv`; the split builder joins their leaves into one `group_id`
  (27 merged groups). As in D-004, train groups that touch test are removed and validation is
  drawn by `group_id`. `tests/test_splits.py` fails if any leaf, group or saved pair crosses
  splits; the notebook also checks all 176 confirmed pairs (including same-leaf ones).
- **Results.**
  - Before the fix, confirmed pairs crossed the splits 43 times: train/test 29, val/test 2,
    train/val 12 (plus 7 pairs between the already-removed images and test). After: **0**.
  - Removed from train in total: 486 images (the earlier 290 + 196), 304 fallback and 182 map.
  - Final sizes: **train 36,963 / val 6,147 / test 10,709** (test unchanged, image for image).
  - Validation is one fold of `StratifiedGroupKFold`, so changing the groups re-draws it:
    10,574 images swapped between train and val. No model had been trained yet, so nothing
    is affected.
- **Limitation.** Re-shots below 0.92 and different photos of the same leaf taken from another
  angle are not found. Test metrics are therefore also reported on the leaf-verified (`map`)
  test subset (8,033 images), and the report states that a small residual leakage may remain.
- **Figures.** `reports/figures/dup_corr_hist.png` (all candidates and the cut-off),
  `reports/figures/dup_boundary_pairs.jpg` (pairs at the cut-off: mostly look-alikes, merged on
  purpose). A figure of the clearest leaks (`dup_leak_examples.jpg`) is produced when the
  notebook is run again.

## Notes for the data card (step 2.6)

Moved into `docs/data_card.md` (class balance, "copy" files, background shortcut risk, smallest
classes), where they are kept next to a class table generated from the splits.

## D-007 · Tooling: pinned formatter, self-checking bootstrap

- **Decision.** `ruff` is pinned to one version (0.17.5) in the dev extra and in pre-commit, so
  CI, pre-commit and local runs format code identically and CI cannot break on a new ruff
  release. The bootstrap notebook now checks the committed splits too (test split equals the
  official one; no leaf, group or near-identical pair is shared) and reports the official
  split's 227 shared leaf ids as information instead of a failed check.

## D-008 · Input pipeline: raw pixels out, augmentation on batches, uint8 cache

- **Decision** (`plantdx.data.pipeline`, settings in `configs/base.yaml`).
  1. Images leave the pipeline as float32 in **0–255**; each model scales its own input inside the
     model. The demo and exported models can then take raw pixels, and a preprocessing mismatch
     between training and serving is impossible.
  2. Labels are **one-hot** (CategoricalCrossentropy; label smoothing possible later).
  3. **Augmentation only for training**, applied to whole batches with seeded Keras layers: flips
     (horizontal and vertical: leaf photos have no natural "up"), rotation ±20°, zoom ±15%,
     brightness and contrast ±15%. Rotation and zoom fill corners by reflection. Colour hue is not
     changed, because yellowing and colour of lesions carry the diagnosis.
  4. Decoded images are **cached as uint8**: val + test take about 2.5 GB of RAM (float32 would be
     about 10 GB). The training set (about 5.5 GB) is not cached by default; it can be cached to a
     local file (`data.cache_train`).
  5. File names are shuffled every epoch over the whole training split; evaluation keeps the CSV
     order, so predictions line up with `splits/*.csv`.
- **Checks.** `tests/test_pipeline.py` (15 tests: shapes, value range, one-hot, order, reshuffling,
  augmentation only in training, grayscale/PNG decoding, caches, config). CI now installs
  CPU-only TensorFlow (`.[dev,cpu]`) so these tests run there. Notebook `03_pipeline_check` checks
  the real data, saves augmentation examples and measures speed and memory.
