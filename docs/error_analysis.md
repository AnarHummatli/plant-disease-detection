# Error analysis of the final model

**Model:** MobileNetV2, whole backbone fine-tuned (`mobilenetv2_ft_all`, release v1.0)
**Data:** the test split, scored once (D-015); this analysis reuses the saved predictions.
**Generated tables and figures:** `reports/results.md`, `reports/analysis/`, `reports/figures/`
(`python -m plantdx.analyze`, notebook 08). Version 1.0 · 20 November 2026.

## 1. Summary

- Test macro-F1 **0.982**, accuracy **0.986**: 151 of 10,709 test images are misclassified.
- The errors are concentrated in a few pairs of **visually similar diseases of the same crop**
  (tomato early blight / target spot / late blight, the two corn leaf-spot diseases). The 21% of
  errors that cross crops are mostly healthy leaves of similar shape, or tomato vs potato (same
  plant family; late blight of both is the same pathogen).
- The model **looks at the leaf**: on 555 test images, a median 69% of the Grad-CAM mass lies on
  the leaf, which covers a median 47% of the image (D-020).
- **No sign of leakage inflating the score**: on the same 30 classes, leaf-verified test images
  (leakage impossible) score as high as the full test (macro-F1 0.9849 vs 0.9848).
- A **"not sure" threshold** of 0.947, chosen on validation, holds on test: 96.8% of the images
  are answered with 99.5% accuracy, and two thirds of the errors are caught (D-019).
- **Weakness:** the model is over-confident on its hardest errors. All six failure cases below
  were predicted with probability ≥ 0.98, so the threshold cannot catch them.

## 2. Where the errors are

Most confused pairs (test, `reports/analysis/top_confusions.csv`):

| True class | Predicted | Images | Share of the true class |
|---|---|--:|--:|
| Tomato · early blight | Tomato · target spot | 21 | 9.8% |
| Tomato · yellow leaf curl virus | Tomato · spider mites | 10 | 0.9% |
| Tomato · early blight | Tomato · late blight | 10 | 4.7% |
| Corn · northern leaf blight | Corn · Cercospora (grey) leaf spot | 10 | 4.5% |
| Corn · Cercospora (grey) leaf spot | Corn · northern leaf blight | 8 | 7.5% |

Weakest classes (F1): tomato early blight 0.885 (recall 0.805), corn Cercospora leaf spot 0.912,
tomato target spot 0.935 (precision 0.883: it absorbs early-blight images), potato healthy 0.952
(32 test images, 2 errors), corn northern leaf blight 0.954.

**Tomato early blight** is the clear weak point: one in five of its images goes elsewhere, half of
them to target spot. Both diseases make brown spots with concentric rings on tomato leaves, and
agronomists also tell them apart mainly by lesion size and pattern rather than a single feature.
The two corn diseases are confused in both directions: both make elongated tan lesions between
the leaf veins, and many corn photos are close-ups of one part of a leaf, which hides the lesion
length that separates them.

**Across crops:** 32 of the 151 errors (21%) predict another crop. The largest group are healthy
leaves of similar shape (raspberry → cherry healthy 7, soybean → apple healthy 2, potato →
soybean healthy 2). The other recurring group is tomato vs potato, two crops of the same plant
family with similar leaves and similar diseases: tomato early blight → potato late blight (4),
potato late blight → tomato late blight (3), tomato late blight → potato late blight (1). Late
blight of tomato and potato is the same pathogen (*Phytophthora infestans*), so in those 4 cases
the disease is right and only the crop is wrong. A user knows the crop, so a demo could restrict
the answer to the crop the user names (possible future work).

## 3. Failure cases (`reports/figures/failures.jpg`)

The most confident error of each of the five top confusion pairs, plus the most confident other
error; left the photo, right the Grad-CAM for the predicted class.

| # | True → predicted (probability) | What the image and the heat map show |
|---|---|---|
| 1 | Tomato early blight → target spot (1.00) | A leaf with only a few small brown spots near its base and a strong shadow. The heat map sits on those spots. With so few lesions, the concentric-ring pattern shared by both diseases is all there is to go on. |
| 2 | Tomato yellow leaf curl virus → spider mites (1.00) | Mottled, slightly yellow leaf surface without the typical curling. The heat map covers the speckled centre of the leaf; fine yellow stippling is also what spider-mite feeding looks like. |
| 3 | Tomato early blight → late blight (1.00) | A late-stage leaf: large dry, brown, curled areas. At this stage both blights destroy large parts of the leaf and look alike; the heat map is on the necrotic tissue. |
| 4 | Corn northern leaf blight → Cercospora leaf spot (1.00) | A close-up of a corn leaf with many small, short tan spots; the long "cigar-shaped" lesions typical of northern leaf blight are not in the picture. The heat map follows the spot clusters. |
| 5 | Corn Cercospora leaf spot → northern leaf blight (0.99) | The opposite case: one long continuous lesion along the midrib, the shape typical of northern leaf blight. The heat map lies on that lesion. |
| 6 | Apple scab → apple healthy (1.00) | The leaf shows almost no visible symptoms; the heat map is on the leaf blade. Probably a very early infection, or a label the photo cannot support. |

In all six cases the model looked at the leaf and at the lesions (not at the background), and the
error matches a real visual similarity between diseases. What it lacks is uncertainty: it answers
these cases with probability ≥ 0.98.

## 4. Does the model look at the leaf? (D-020)

`reports/figures/gradcam_on_leaf.png`, `reports/figures/gradcam_examples.jpg`,
`reports/analysis/gradcam_on_leaf.csv`.

| | Median |
|---|--:|
| Leaf share of the image (leaf masks from the segmented variant) | 0.47 |
| Grad-CAM mass on the leaf, all 555 images | **0.69** |
| … correctly classified (544) / misclassified (11) | 0.69 / 0.62 |
| … healthy classes / disease classes | 0.69 / 0.69 |

For 95% of the images the heat map is more concentrated on the leaf than its area alone would
give; for 10% less than half of it is on the leaf. The lowest class medians are peach healthy
(0.42), tomato mosaic virus (0.54) and tomato spider mites (0.55). In the examples, disease classes
light up on lesions and discoloured tissue, while healthy leaves without lesions light up on the
leaf margin and outline (e.g. tomato healthy), where the coarse 7 × 7 map partly spills onto the
background. So the background is not what the model relies on, but for healthy leaves part of the
evidence is the leaf's shape at its edge.

## 5. Residual leakage (D-004, D-010)

| Test part | Images | Classes | Macro-F1 | Accuracy |
|---|--:|--:|--:|--:|
| Full test | 10,709 | 38 | 0.9816 | 0.9859 |
| Full test, the 30 classes with verified leaves | 9,083 | 30 | 0.9848 | 0.9862 |
| Leaf-verified images (leakage impossible) | 8,033 | 30 | 0.9849 | 0.9867 |
| Unverified images of those 30 classes | 1,050 | 6 | — | 0.9829 |

Rows 2 and 3 have the same classes and differ only in whether the leaf is verified: same score,
so photos of leaves that might also be in train do not make the test easier. The unverified images
(the only ones where leakage is possible) are even classified slightly less accurately (0.983)
than the verified ones (0.987). The difference between rows 1 and 2 (0.9816 vs 0.9848) comes from
the 8 classes that have no verified leaves, two of which (corn Cercospora leaf spot, tomato target
spot) are among the hardest. Conclusion: no measurable inflation from residual leakage. The
evidence is indirect for the 8 classes without verified leaves, where no such comparison is
possible.

## 6. The "not sure" answer (D-019)

`reports/figures/coverage_not_sure.png`. Threshold **0.947** (smallest with ≥ 99.5% accuracy on the
answered validation images).

| Split | Answered | Accuracy (all) | Accuracy (answered) | Errors caught | Correct answers held back |
|---|--:|--:|--:|--:|--:|
| val | 96.4% | 0.9858 | 0.9951 | 58 of 87 | 162 |
| test | 96.8% | 0.9859 | 0.9951 | 100 of 151 | 245 |

The validation choice transfers to test almost exactly. The cost is 245 correct answers (2.3% of
the test images) replaced by "not sure". For reference only (test numbers, not used for any
choice): a stricter threshold would trade coverage for accuracy (0.99: 94.3% answered, 99.8%
accurate; 0.999: 89.8% answered, 99.9% accurate). The threshold stays the one chosen on validation.
The 51 errors that remain above the threshold are the over-confident ones of section 3.

## 7. Limitations and what would help

- **Over-confidence.** Errors with probability ≈ 1 are not caught by any threshold. Temperature
  scaling on the validation set or label smoothing during training would calibrate the
  probabilities; both are future work.
- **Lab photos.** All of this holds for PlantVillage's single leaves on plain backgrounds; field
  photos would need their own test (problem statement, section 5).
- **Grad-CAM is coarse** (7 × 7 cells of about 32 px) and the leaf masks come from the dataset's
  own segmentation, which has some errors; 15 of the 570 sampled images had no segmented copy.
- **Small classes.** Potato healthy has 32 test images; one error changes its F1 by about 0.015.
