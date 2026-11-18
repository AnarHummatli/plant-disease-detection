# Problem statement

**Project:** Plant Disease Detection Using Neural Networks (topic 06)
**Team:** Yarpaq Həkimləri · Neural Networks group project · ASOIU, Fall 2026
**Version:** 1.0 · 18 November 2026

## 1. Problem

Plant diseases are easiest to contain when they are recognised early, from the first visible
symptoms on the leaves. In practice, many growers cannot get an agronomist's opinion at the
moment they notice a suspicious leaf. A system that gives a quick, preliminary diagnosis from a
phone photo of a leaf can narrow that gap: it tells the grower what the symptoms most likely
indicate and when it is worth asking an expert.

## 2. Task definition

| | |
|---|---|
| **Input** | One RGB photograph of a single leaf (JPG or PNG), resized to 224 × 224 inside the model. |
| **Output** | One of 38 classes (14 crop species; each class is a crop + disease or crop + healthy), shown as the top-3 classes with probabilities. If the top probability is below a threshold chosen on the validation set, the system answers *"not sure — please take a clearer photo"* instead of a diagnosis. |
| **Task type** | Single-label, multi-class image classification. |
| **Data** | PlantVillage (color variant), official leaf-grouped train/test split; see `docs/data_card.md` (step 2.6). |

## 3. Target users

- **Small and medium farmers** who want a first opinion before spending money on treatment or an expert visit.
- **Agronomists and extension workers** who can use the top-3 output and the Grad-CAM heat map as a quick screening aid.
- **Home gardeners and students** learning to recognise common diseases.

## 4. Success criteria

1. **Main metric:** macro-F1 ≥ 0.97 on the official test split, evaluated once, on the final model only.
   Macro-F1 is the main metric because the classes are imbalanced and every disease matters equally.
2. **Honest comparison:** the neural networks (CNN from scratch, MobileNetV2 transfer learning) are compared with a
   classical baseline (colour/HOG/LBP features + logistic regression / linear SVM) under the same split and protocol.
   Reported: accuracy, macro-F1, per-class F1, top-3 accuracy, confusion matrix, model size and CPU inference time.
3. **No leakage:** no physical leaf appears in more than one of train / validation / test; checked by automated tests.
4. **Understanding failures:** at least 3 (target 6) misclassified examples explained with Grad-CAM.
5. **Working demo:** an upload interface that shows the top-3 prediction, the Grad-CAM overlay and the "not sure" state.
6. **Reproducibility:** a clean Colab session reproduces split → training → evaluation → demo from the README.

## 5. Out of scope

- Detecting or localising **several leaves or several diseases** in one photo (no object detection or segmentation).
- **Crops and diseases outside the 38 PlantVillage classes**; such inputs may be misclassified (out-of-distribution
  detection is planned for the portfolio phase, step 10.5).
- **Severity estimation**, treatment plans, pesticide names or doses.
- Any claim of **field-ready or safety-critical** use. PlantVillage photos were taken under lab conditions on plain
  backgrounds; accuracy on real field photos is expected to be lower and is measured separately later (PlantDoc, step 10.1).

## 6. Responsible use

The system is an educational, preliminary information tool. It does not replace an agronomist's diagnosis, and this is
stated in the demo and in the report. All data is openly licensed (PlantVillage: CC BY-SA 3.0); images are downloaded
by code and never committed to the repository. Any photos taken by the team contain no faces or personal items and are
taken only with the owner's permission.
