# Decision log

Short records of technical decisions that differ from the obvious default or from the
original plan (v1.0, 18 Nov 2026). Each entry: context → decision → consequence. References such
as "plan 4.2" or "step 7.2" point to that plan's stages: 1 foundation, 2 data, 3 baselines and CNN,
4 transfer learning and test, 5 error analysis, 6 demo, 7 reproducibility, 8 report.

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

## D-002 · Python 3.12–3.13 and the TensorFlow that Colab provides

- **Context.** Plan v1.0 said Python 3.11 and TensorFlow 2.21. Colab's default runtime moved to
  Python 3.12 in 2025 and to 3.13 by November 2026 (the stage 3 runs recorded 3.13.16 and
  3.13.18), and Colab ships its own CUDA-matched TensorFlow build. TF 2.21 supports
  Python 3.10–3.13.
- **Decision.** Target Python ≥ 3.10; CI runs 3.12 and 3.13 (the Colab version). TensorFlow is an optional extra
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
    Superseded by D-010, which adds same-leaf pairs.
  - Validation is one fold of `StratifiedGroupKFold`, so changing the groups re-draws it:
    10,574 images swapped between train and val. No model had been trained yet, so nothing
    is affected.
- **Limitation.** Re-shots below 0.92 and different photos of the same leaf taken from another
  angle are not found. Test metrics are therefore also reported on the leaf-verified (`map`)
  test subset (8,033 images), and the report states that a small residual leakage may remain.
- **Figures.** `reports/figures/dup_corr_hist.png` (all candidates and the cut-off),
  `reports/figures/dup_boundary_pairs.jpg` (pairs at the cut-off: mostly look-alikes, merged on
  purpose). A figure of the clearest leaks (`dup_leak_examples.jpg`) needs the images and is
  produced in the error-analysis notebook (stage 5).

## Notes for the data card

Moved into `docs/data_card.md` (class balance, "copy" files, background shortcut risk, smallest
classes), where they are kept next to a class table generated from the splits.

## D-007 · Tooling: pinned formatter, self-checking bootstrap

- **Decision.** `ruff` is pinned to one version (0.17.5) in the dev extra and in pre-commit, so
  CI, pre-commit and local runs format code identically and CI cannot break on a new ruff
  release. The bootstrap notebook now checks the committed splits too (test split equals the
  official one; no leaf, group or near-identical pair is shared) and reports the official
  split's 227 shared leaf ids as information instead of a failed check.

## D-008 · Input pipeline: raw pixels out, augmentation on the GPU, uint8 caches

- **Decision** (`plantdx.data.pipeline`, settings in `configs/base.yaml`).
  1. Images leave the pipeline as float32 in **0–255**; each model scales its own input inside the
     model. The demo and exported models can then take raw pixels, and a preprocessing mismatch
     between training and serving is impossible.
  2. Labels are **one-hot** (CategoricalCrossentropy; label smoothing possible later).
  3. **Augmentation only for training**, with seeded Keras layers: flips (horizontal and
     vertical: leaf photos have no natural "up"), rotation ±20°, zoom ±15%, brightness and
     contrast ±15%. Hue is not changed, because yellowing and lesion colour carry the
     diagnosis. Rotation/zoom corners are filled by reflection; this sometimes mirrors a piece of
     the leaf at the edge (`reports/figures/aug_examples.jpg`), which we accept: black corners
     would add a new background cue, and edge stretching creates streaks.
  4. **Augmentation runs on the GPU as the first model layers** (`train.augment_on: model`;
     `augmentation_from_config` builds them, Keras turns them off at inference). First
     measurement on Colab (T4, 2 CPU cores) with augmentation inside tf.data: **67.5 images/s**,
     i.e. about 9 minutes of input work per training epoch, while decoding alone ran at about
     317 images/s. The tf.data option (`augment_on: pipeline`) is kept for CPU-only use.
  5. **Caches hold decoded uint8 images**: val + test in memory (about 2.5 GB; float32 would be
     4×), train (about 5.5 GB) as a file on the VM's local disk (`cache_train: disk`,
     `PLANTDX_CACHE`, `/content/cache` on Colab). Cache names contain the image size and a hash of
     the file list, so a changed split cannot reuse a stale cache. Val from memory: 6,147 images in
     1.6 s instead of 19.4 s.
  6. **Order.** The CSVs are sorted by class. Without a cache the whole file list is reshuffled
     every epoch. With a cache the file list is shuffled once with the seed **before** decoding,
     then a 4,096-image buffer reshuffles every epoch. (A first version shuffled only the cached,
     class-sorted stream with a small buffer, which would have produced batches of one or two
     classes; caught while reviewing the speed fix, now covered by a regression test.)
     Evaluation keeps the CSV order, so predictions line up with `splits/*.csv`.
- **Measured after the change** (Colab T4, 2 CPU cores, `reports/pipeline_check.json`):

  | | before | after |
  |---|--:|--:|
  | training input, images/s | 67.5 | 935.8 (from disk cache) |
  | augmentation, images/s | (on CPU, inside the 67.5) | 4,744 (on GPU) |
  | input time per training epoch | ≈ 548 s | **≈ 39.5 s** (about 14× faster) |
  | first epoch on a fresh VM (decode + write cache) | – | 140 s |
  | val pass (6,147 images) | 22.9 s | 0.9 s from memory cache |

  Disk cache 5.6 GB (112 GB free), val cache 0.9 GB of 13.6 GB RAM. The input pipeline now
  delivers about 940 images/s, more than a T4 is expected to train MobileNetV2 at 224 × 224, so the
  GPU, not the data, should set the training speed.
- **Checks.** `tests/test_pipeline.py` (18 tests: shapes, value range, one-hot, order,
  reshuffling, class mixing with a cache, augmentation only in training or only in the model,
  grayscale/PNG decoding, memory and disk caches, cache naming, config). CI installs CPU-only
  TensorFlow (`.[dev,cpu]`) so they run there. Notebook `03_pipeline_check` checks the real data,
  saves augmentation examples and measures speed (results: `reports/pipeline_check.json`).

## D-009 · Colab notebooks force-sync the repository

- **Context.** After a push, notebook 03 kept importing old code: an earlier run had written
  `reports/figures/aug_examples.jpg` into the Colab copy of the repo, the new commit added the
  same file, and `git pull` refused to overwrite it. The error was easy to miss.
- **Decision.** Every notebook's setup cell now runs `git fetch` + `git reset --hard origin/main`
  (the Colab copy is disposable; results are downloaded or written to Drive) and stops with an
  error if the checked-out commit is not the latest one. After pushing new code to an open Colab
  session, use *Runtime → Restart session* so Python re-imports the package.

## D-010 · Recovering the leaf groups of images without a verified leaf id

- **Context.** Verified leaves have a median of 4 photos (mean 5.4); 96.9% of file-name
  ("fallback") leaf ids contain a single image (`docs/data_card.md`, images per leaf). So for
  12,890 images (24%, including 8 whole classes) the other photos of the same leaf are not
  grouped and can sit in train and test. D-006 does not catch them: photos of one leaf are
  usually different poses, not near-copies (only about 1,300 same-leaf pairs among all verified
  leaves reached the D-006 candidate level). This also means the "verified" test subset alone
  cannot serve as the only clean benchmark: it lacks the 8 fallback-only classes.
- **Method** (`plantdx.data.sameleaf`, notebook `04_sameleaf_groups`, tests in
  `tests/test_sameleaf.py`).
  1. Embed all 54,305 images with ImageNet-pretrained MobileNetV2 (global average pooling),
     plain and averaged over the 8 rotations/flips (D4), so that the pose matters less.
  2. Take the 10 nearest neighbours of every image inside its class (cosine similarity),
     optionally keeping only mutual neighbours.
  3. **Calibrate on verified leaves**, where the answer is known: precision and recall of
     "similarity ≥ t" on pairs of two verified images. The suggested cut-off is the lowest one
     at which precision stays ≥ 0.95 for every higher cut-off.
  4. Look at pairs with a fallback image around the cut-off, and measure what each cut-off does
     to the splits (pairs, train images removed, largest group, fallback group sizes).
  5. Pairs above the chosen cut-off that involve a fallback image go to
     `splits/sameleaf_pairs.csv`; the split builder joins them like D-006 pairs
     (`load_group_pairs` reads both files; `tests/test_splits.py` checks both).
- **Results** (Colab T4, 18 Nov 2026; `reports/sameleaf_*.csv`, `reports/figures/sameleaf_calibration.png`).
  - Calibration: ImageNet features separate "this leaf" from "another leaf of the same class"
    poorly. Precision stays ≥ 0.95 only at the top: plain embeddings at cosine ≥ 0.94 give precision
    0.963 and recall **0.004**; D4-averaged ones need ≥ 0.98 (precision 0.987, recall 0.002). At
    10–15% recall precision is only 0.28–0.55, depending on the variant. (On synthetic data D4 had won; on the real
    images the plain variant reaches the bar earlier.) Likely reason: all leaves of a class were
    shot on the same background under the same light, and other photos of a leaf often show
    another side or pose.
  - Visual check of pairs with a file-name image: above 0.94 every sampled pair is the same leaf;
    in 0.88–0.94 similar but different leaves appear.
  - Cost per cut-off (with the D-006 pairs): 0.94 → 562 pairs, 207 train/val images to remove;
    0.92 → 2,442 and 1,044; 0.88 → 14,202 and 3,474, largest group 1,037 (chaining). Even at 0.88
    the single-image share of file-name groups only falls from 95% to 86%.
- **Decision.** Plain embeddings, cut-off **0.94** (`splits/sameleaf_meta.json`). 565 pairs, 381
  of them in Tomato Yellow Leaf Curl Virus. Lower cut-offs would merge different leaves and remove
  thousands of images without closing the gap.
- **Effect.** 253 of these pairs crossed the splits before, 0 after; 210 more train images removed
  (696 in total: 290 leaf ids + 196 near-identical + 210 same-leaf). Final sizes: **train 36,779 /
  val 6,121 / test 10,709** (test unchanged). Single-image share of file-name groups: 96.8% → 94.9%.
- **Conclusion: a documented negative result.** Hidden same-leaf pairs among file-name images
  cannot be recovered reliably with generic CNN features, so some leakage may remain. It is
  handled in evaluation (stage 4): every test metric is reported on the full official test set and
  on the leaf-verified subset (8,033 images, 30 classes), where it is excluded by construction;
  the gap estimates its effect. Local-feature matching (keypoints + geometric check) could find
  more re-shots, but not other-side photos; it is left as possible future work.

## D-011 · Classical baseline: HOG reduced with PCA, colour and LBP kept

- **Context** (plan 3.1–3.2). The baseline features are an HSV histogram (96 values), HOG
  (1,764) and a uniform LBP histogram (10) at 128 × 128, 1,870 values per image. On 36,779 train
  images, one logistic-regression fit on all 1,870 standardised features did not finish within
  10 minutes on a 2-core machine, so a grid of 4 C values × 2 models would take hours on Colab.
- **Decision.** HOG alone is standardised and reduced to **256 PCA components** (randomised SVD,
  seed 42); the colour and LBP histograms are passed through unchanged, then everything is
  standardised again (`plantdx.baselines.classical.make_model`). PCA over all features would be
  dominated by the 1,764 HOG values and could drop the colour histogram, which carries most of the
  symptom colour. Measured on Colab (CPU runtime, 19 Nov 2026): feature extraction 9.5 min
  (train) + 1.3 min (val), then one fit per C: logistic regression 35–46 s; linear SVM 101 s
  (C = 0.01) to 351 s (C = 1), and C = 10 did not converge in 5,000 iterations (1,396 s). Neither
  model selected C = 10, so the result is unaffected; the grid is kept as it ran.
- **Protocol.** `class_weight="balanced"`; C ∈ {0.01, 0.1, 1, 10} chosen on val macro-F1, ties to
  the smaller C; fit on train only; test untouched until the final evaluation. Features are cached
  on Drive as .npz (name = split, size, feature version, hash of the file list).

## D-012 · CNN from scratch and the training protocol

- **Architecture.** The plan had one Conv3×3 per block (4 blocks, 32–256 filters). With one
  convolution the receptive field after the last pooling is 46 px, a fifth of the 224 px image;
  two convolutions per block (VGG style) give 76 px, enough to see a lesion with its surroundings,
  for 1.18 M parameters instead of 0.40 M. Conv layers have no bias (BatchNorm follows), GAP,
  Dropout 0.3, softmax in float32. `model.convs_per_block` keeps the plan's variant available.
- **Selection vs stopping.** The checkpoint keeps the epoch with the best **val macro-F1** (the
  selection metric, computed in-graph with `keras.metrics.F1Score`); EarlyStopping (patience 5)
  and ReduceLROnPlateau (×0.5, patience 2, min 1e-6) watch **val loss**, which is far smoother on a
  validation set whose smallest class has 12 images. After training the checkpoint is re-scored
  with scikit-learn (exact macro-F1, per-class scores) and its val probabilities are saved for
  threshold selection and error analysis.
- **Mixed precision** (`train.mixed_precision: true` in `cnn.yaml`): float16 compute on the T4's
  tensor cores; the output layer stays float32. The speed-up was not measured separately: the
  run took 244 s per epoch (425 ms per batch of 64), slower than the 1–2 min estimated
  beforehand. The two 32-filter convolutions at full 224 px resolution are the most likely
  cost. A saved checkpoint keeps the float16 layers; the float32 inference model is D-014.
- **Results table.** Every model writes one row per (run, split) to `<artifacts>/reports/results.csv`
  (Drive), the same columns for classical and neural models; notebooks export it with the small
  run files (`plantdx.results.export_for_repo`) so it can be committed under `reports/`.

## D-013 · Grad-CAM and the "not sure" answer are required, not optional

- **Context.** Plan v1.0 listed Grad-CAM and the "not sure" threshold as optional (5.5), to be
  dropped first if time ran short, while the problem statement (success criteria 4–5) and the
  README already promised both.
- **Decision.** Both are part of the minimum, in a light form: Grad-CAM for the final model on the
  failure cases and in the demo; one confidence threshold chosen on val from the
  coverage–accuracy curve. Together they take a few hours and answer the main data risk named in
  the data card (class-correlated backgrounds): whether the model looks at the lesions or at the
  background. Still optional: Hugging Face Spaces, a second fine-tuning variant.

## D-014 · Inference model and the reported model size

- **Context.** The first CNN run reported `size_mb` = 23.84: the size of `best.keras`, the
  training checkpoint, which also stores the Adam state (two extra copies of every weight) and,
  under mixed precision, float16 compute layers. That is not the model a user would deploy.
- **Decision.** After training, `plantdx.train` rebuilds the network from its config in float32
  without the augmentation layers, copies the weights layer by layer by name and saves it
  uncompiled as `inference.keras` (`plantdx.export`). `size_mb` in the results table and CPU
  timing in the evaluation use this file; the demo loads it. A test checks that it predicts like
  the mixed-precision model (max difference 2e-3).
- **Earlier run.** `cnn_scratch` was trained before this change; `python -m plantdx.export
  --run-id cnn_scratch` creates its inference model (about 4.9 MB for 1.18 M float32 parameters),
  and its size in the results table is corrected in the final evaluation (step 4.4).

## D-015 · Test protocol: select on validation, then score every family once

- **Rule.** The test split is not used for any choice. On validation macro-F1 (ties go to the
  smaller model) `plantdx.evaluate` picks the best run of every neural family (CNN from scratch,
  MobileNetV2) and the **final model = the best of those winners**, whichever family it comes
  from (changed on 19 Nov 2026 from "MobileNetV2 runs only", see D-017). The classical models
  were already chosen on validation in stage 3 (C). Every family winner and the classical models
  are then scored on test, each exactly once; the comparison table in the report needs test
  numbers for all of them, and no decision depends on them. A second evaluation of the same run is
  refused (`metrics_test.json` exists) unless `--force` is passed to repeat a run that failed
  midway.
- **Two test reports.** Full test split (10,709 images, 38 classes) and its leaf-verified part
  (`leaf_source == "map"`: 8,033 images, 30 classes), where leaf grouping rules out same-leaf
  leakage by construction (D-004). Macro-F1 on the verified part averages over its 30 classes. A
  clearly lower score on the verified part would point to residual leakage in the rest.
- **Also measured once per run:** model size (inference model or the joblib file) and CPU time
  per image at batch size 1 (median of 50 after 5 warm-up calls; for the classical models this
  includes feature extraction from a decoded image). Both are written to every row of the run in
  the results table, which also corrects the size of `cnn_scratch` (D-014).
- **Outcome** (20 Nov 2026, Colab CPU runtime: the GPU quota was used up, and no training
  happens in this step). Test macro-F1 / verified macro-F1: final MobileNetV2 0.9816 / 0.9849,
  CNN 0.9671 / 0.9708, logistic regression 0.8839 / 0.8807, linear SVM 0.8731 / 0.8692; the
  ranking is the same as on validation. The first session was interrupted after scoring the
  final model; the second completed the other three runs, keeping the first result (since then
  `select_and_evaluate` resumes this way by itself).
- **CPU time caveat.** Measured with eager Keras calls at batch size 1 on Colab's shared 2-core
  CPU: MobileNetV2 239 ms, CNN 147 ms, classical 19 ms per image. MobileNetV2 needs about 9×
  fewer operations than the CNN, so its higher time reflects TensorFlow's slow CPU path for
  depthwise convolutions and per-call overhead, not the model's cost. Latency is measured again
  for the demo with a compiled graph and TFLite; this is not a test metric, so re-measuring it
  does not touch the one-time test rule.

## D-016 · MobileNetV2 transfer learning

- **Model** (`plantdx.models.mobilenet`). Input 0–255 → augmentation (training only) →
  `Rescaling(1/127.5, offset=-1)` (identical to `mobilenet_v2.preprocess_input`, saved with the
  model) → MobileNetV2 (ImageNet, `include_top=False`, named `backbone`), always called with
  `training=False` → GAP → Dropout 0.2 → Dense(38, softmax, float32). 2,306,662 parameters.
- **Head** (`mobilenetv2_head.yaml`, plan 4.1): backbone frozen, 48,678 trainable parameters,
  Adam 1e-3, at most 10 epochs, mixed precision.
- **Fine-tuning** (`mobilenetv2_ft.yaml`, plan 4.2): starts from the best head checkpoint
  (`train.init_from`); layers from `block_13_expand` on are trainable. That is the plan's "last
  ~30 layers": 38 layers, of which 26 have weights, but because MobileNetV2 keeps most of its
  parameters in the last blocks it is 1.66 M of the backbone's 2.26 M parameters (1,712,038
  trainable with the head). BatchNorm layers stay frozen and in inference mode, so small batches
  do not overwrite the ImageNet statistics. Adam 1e-5 (min 1e-7), at most 10 epochs. Variant
  `mobilenetv2_ft_lr3e-5.yaml`: same layers, Adam 3e-5. Both variants are trained; the choice
  between head, fine-tune and variant is made on validation (D-015).
- **Weight transfer.** Checkpoint weights are copied by variable path (`block_13_expand/kernel`),
  not by position: paths are unique and do not depend on which layers are frozen, so a copy
  cannot silently shuffle weights between layers. Tests check head → fine-tune and the float32
  export.

## D-017 · MobileNetV2 under-trained by the plan's settings: two more variants

- **Observation** (validation, 19 Nov 2026). The plan's MobileNetV2 settings ended below the CNN
  from scratch (val macro-F1 0.9732): head 0.9158 (best epoch 8/10), fine-tuning with Adam 1e-5
  0.9456 and with 3e-5 0.9551, both with their **best epoch at the last of 10** and val
  macro-F1 still rising. Raising the learning rate from 1e-5 to 3e-5 added about one point. The
  CNN used the same pipeline and reached 0.973, so a data or preprocessing fault is unlikely
  (a test also checks the preprocessing against Keras' `preprocess_input`). The reading:
  under-training, not a weak method.
- **Decision.** Two more variants, each testing one explanation, both starting from the same
  head checkpoint and allowed up to 30 epochs (early stopping on val loss, patience 5):
  **A** `mobilenetv2_ft_lr1e-4` (same 38 layers, Adam 1e-4: too little training) and
  **B** `mobilenetv2_ft_all` (whole backbone from `Conv1`, BatchNorm still frozen, 2,238,438
  trainable parameters, Adam 3e-5: the lower layers must adapt to leaf textures too).
- **Selection** stays on validation only, and the final model is now chosen across neural
  families (D-015): if the CNN from scratch stays ahead, it is the final model and that is
  reported as a finding.
- **Tuning budget, stated for a fair comparison:** CNN from scratch 1 configuration;
  MobileNetV2 5 (head, 3 fine-tuning learning rates / epoch limits, whole backbone).
- **Outcome** (validation, 19 Nov 2026). Both explanations hold: A (Adam 1e-4) reached val
  macro-F1 0.97722 (best epoch 12, stopped at 17), B (whole backbone) 0.97738 (best epoch 14,
  stopped at 18), both above the CNN from scratch (0.97321). A and B are tied within noise
  (difference 0.00016; A has the higher accuracy, 0.9863 vs 0.9858). The selection rule fixed
  before training, highest val macro-F1, picks **B, `mobilenetv2_ft_all`, as the final model**;
  it is not changed after seeing the numbers. Both have the same architecture and inference cost.

## D-018 · Reproducibility on the GPU

- **Context.** Seeds are fixed (`seed: 42` for Python, NumPy, TensorFlow and the data order), but
  cuDNN kernels on the GPU add floating-point numbers in a varying order, so two runs of the same
  config differ slightly. `tf.config.experimental.enable_op_determinism()` would remove this, at a
  large speed cost on the T4 and with some operations unsupported.
- **Decision.** Not enabled. Every run records its config, environment and commit
  (`reports/runs/<run_id>/`), and the trained models are published (release v1.0), so the
  reported numbers can be checked against the exact model rather than by retraining.
- **Consequence.** A re-run of a config can differ from the reported validation scores by a
  fraction of a point; the comparisons in this project (e.g. MobileNetV2 vs CNN: 0.982 vs 0.967
  test macro-F1) are far larger than that. The two final candidates (val macro-F1 0.97738 vs
  0.97722) are within this noise, which is why they are reported as tied (D-017).

## D-019 · The "not sure" threshold

- **Rule** (fixed on 20 Nov 2026, before computing it). The demo answers "not sure" when the
  final model's highest probability is below a threshold *t*. *t* is the **smallest** threshold
  for which the answered **validation** images are at least **99.5%** correct; the smallest such
  *t* keeps the most images answered. If 99.5% is never reached, the threshold with the best
  answered accuracy is used. Thresholds are searched on a grid that is finest near 1 (steps of
  0.0001 above 0.99), because a confident model puts most of its probabilities there.
- **Then** *t* is applied unchanged to test, which shows whether the coverage and accuracy hold
  on unseen images. Reported: share answered, accuracy of the answered images, errors caught,
  correct answers held back (`plantdx.analysis`).
- **Limit.** A confidence threshold catches uncertain in-distribution images; it is not an
  out-of-distribution detector, and a photo of something that is not a leaf can still receive a
  confident answer (out of scope, problem statement section 5).
- **Outcome** (20 Nov 2026). *t* = **0.947**. Validation: 96.4% answered, 99.51% of them correct.
  Test, unchanged: 96.8% answered, 99.51% correct, 100 of 151 errors caught, 245 correct answers
  (2.3%) held back. The 51 errors above the threshold are confident ones (probability ≈ 1), which
  calls for calibration (temperature scaling) as future work (`docs/error_analysis.md`).

## D-020 · Does the model look at the leaf? Grad-CAM measured with leaf masks

- **Question.** Backgrounds differ between groups of classes (data card, section 3), so a model
  could classify by background. Grad-CAM pictures alone answer this only for a few images.
- **Method** (`plantdx.analyze`, `plantdx.gradcam`). Grad-CAM of the final model on the
  backbone's last feature maps (7 × 7, upsampled to 224 × 224), for the predicted class, on up to
  15 random test images per class (seed 42). The dataset's *segmented* variant has the same
  photos with the background set to black, which gives a leaf mask for every image (pixel = leaf
  if its brightest channel is above 25). Per image: the share of the heat-map mass on the leaf,
  compared with the leaf's share of the image (what a heat map spread evenly would give).
- **Reading.** Values well above the leaf's area mean the model concentrates on the leaf. The
  heat map is coarse (cells of about 32 px), so mass near the leaf edge partly spills onto the
  background; the masks come from the dataset's own segmentation and contain some errors.
- **Outcome** (20 Nov 2026, 555 test images; 15 of the 570 sampled had no segmented copy). Median
  Grad-CAM mass on the leaf **0.69** against a median leaf area of 0.47; for 95% of the images the
  mass on the leaf exceeds the leaf's area, for 10% it is below one half. Correct and wrong
  predictions: 0.69 and 0.62; healthy and disease classes: both 0.69. Lowest class medians:
  peach healthy 0.42, tomato mosaic virus 0.54. The model relies on the leaf, not on the
  background; for healthy leaves part of the evidence is the leaf outline.
