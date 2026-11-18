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
  5. Measure residual near-duplicate leakage with a perceptual hash (see "Audit results").
- **Audit results** (notebook `01_data_audit_eda`, 18 Nov 2026). Every val image was compared
  with all same-class train images, every test image with all same-class train + val images,
  using a 256-bit difference hash (dHash; Hamming distance 0 = identical, unrelated images
  ≈ 128).
  - **Across the final splits: no near-duplicates.** 0.0% of val and test images have a
    same-class neighbour within distance 20, for both `map` (12,762 images) and `fallback`
    (4,131 images). The closest cross-split pair is 30 bits apart.
  - **Positive control.** The same search on the 290 removed train images against test finds
    5 near-identical copies (1.7% at distance 0; 2.1% within 20). So the audit does detect
    duplicates when they exist, and removing these images was necessary. It also shows that
    the other ~285 "shared" file names are *different photos* (same name, different capture),
    not copies.
  - **Weak signal in the tail.** Nearest-neighbour distances are a little smaller for
    `fallback` than for `map` (1st percentile 47 vs 58, 5th percentile 63 vs 68; medians 87
    vs 84). This may mean some photos of the same leaf remain across splits among fallback
    images; it is small and is reported, not hidden.
  - **Limitation.** A perceptual hash recognises re-saved or slightly resized copies, not a
    different photo of the same leaf. "0% near-duplicates" therefore means *no duplicate
    images across splits*, not *no shared leaves*. Results are reported on the full test set
    and on its leaf-verified (`map`) subset (8,033 of 10,709 images).
  - Files: `reports/audit_summary.csv`, `reports/audit_distance_quantiles.csv`,
    `reports/audit_positive_control.csv`, `reports/figures/audit_*.png`. The per-image table
    (~3 MB) is kept on Drive (`artifacts/reports_step2/`), not in git.
- **Consequence.** The claim in the report is: "no leaf with a known id appears in two splits,
  and no duplicate image appears in two splits", with the 24.3% fallback share, the removed
  overlap, the weak tail signal and the verified-subset metrics stated openly. This replaces
  the plan's assumption that the official split is fully leaf-grouped.

## D-005 · Repository size: large derived files stay out of git

- **Decision.** Photo grids are saved as JPEG (the 38-image sample grid went from 4.2 MB PNG to
  about 0.5 MB). Per-image result tables (`audit_near_duplicates.csv`, ~3 MB) are written to
  Drive and ignored by git; only small summaries are committed. `.gitattributes` forces LF
  line endings so commits from Windows do not create whitespace-only diffs.
