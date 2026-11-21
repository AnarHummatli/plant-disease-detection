# Model card · Plant disease classifier (MobileNetV2, release v1.0)

A short, standard summary of the released model: what it is for, how well it works and where it
should not be trusted. Details: [technical report](report.pdf), [data card](data_card.md),
[decision log](decisions.md).

## Model

| | |
|---|---|
| Task | Single-label classification of one leaf photo into 38 classes (14 crops; 26 diseases, 12 healthy) |
| Architecture | MobileNetV2, ImageNet initialisation, whole backbone fine-tuned; global average pooling, dropout 0.2, 38-way softmax; batch normalisation frozen (D-016, D-017) |
| Parameters | 2,306,662 (2,238,438 trainable during fine-tuning) |
| Input / output | RGB, centred square, 224 × 224, values 0–255 (scaling inside the model) / 38 probabilities |
| Training | Adam 3e-5, batch 64, mixed precision, best of 18 epochs by validation macro-F1 (epoch 14), 39 min on an NVIDIA T4; TensorFlow 2.21, Keras 3.13 |
| Files | `inference.keras` (9.8 MB, SHA-256 `aad4ba57…92afcad`) in release [v1.0](https://github.com/AnarHummatli/plant-disease-detection/releases/tag/v1.0); an ONNX export with Grad-CAM is built by `plantdx.web_export` for the browser demo |
| Author, licence | Anar Hummatli, 2026; code MIT, training data CC BY-SA 3.0 |

## Intended use

- **For:** a first opinion on a photo of a single leaf of one of the 14 supported crops, with the
  top three classes, a Grad-CAM heat map and a "not sure" answer; teaching and demonstrating
  image classification and its evaluation.
- **Not for:** treatment decisions without an expert, crops or diseases outside the 38 classes,
  photos of whole plants or several leaves, or any safety-critical or commercial use. It is an
  educational prototype and does not replace an agronomist.

## Data

PlantVillage, colour variant (Hughes & Salathé, 2015): 54,305 photos of detached leaves on plain
backgrounds. Leaf-grouped split rebuilt to remove same-leaf leakage: train 36,779 / validation
6,121 / test 10,709 (official test split unchanged). The leaf-verified test subset (8,033 images,
30 classes) excludes leakage by construction.

## Performance (test split, evaluated once)

| Measure | Value |
|---|---|
| Macro-F1 / accuracy, all 38 classes | 0.9816 / 0.9859 |
| Macro-F1 / accuracy, leaf-verified subset | 0.9849 / 0.9867 |
| Top-3 accuracy | 0.9996 |
| With "not sure" below 0.947 (chosen on validation) | 96.8% answered, 99.5% of them correct |
| Weakest classes (F1) | tomato early blight 0.885, corn Cercospora leaf spot 0.912, tomato target spot 0.935 |
| Grad-CAM mass on the leaf (555 test images) | median 0.70; the leaf covers a median 0.47 of the image |

Baselines on the same split: CNN from scratch 0.9671, colour/HOG/LBP with logistic regression
0.8839 (test macro-F1). A clean-session check reproduced the test results exactly (D-024).

## Limitations and risks

- **Lab images only.** Trained and tested on single leaves against plain backgrounds; accuracy on
  field photos was not measured and is expected to be much lower.
- **Closed set.** Inputs outside the 38 classes get the nearest class, often with high
  confidence; "not sure" is not an out-of-distribution detector.
- **Over-confidence.** 51 of the 151 test errors still pass the 0.947 threshold, many with
  probabilities near 1; the confidence is not calibrated.
- **Residual leakage.** For 24.3% of the images the true leaf is unknown; the verified subset shows
  no inflation but covers 30 of the 38 classes.
- **Privacy.** In the browser demo the model runs on the user's device; photos are not uploaded.

## How to use

```python
from plantdx.predict import Predictor

predictor = Predictor()  # downloads inference.keras from release v1.0 once, checks SHA-256
result = predictor.predict("leaf.jpg")
print(result.verdict())  # e.g. "Grape: Black rot (confidence 100.0%)." or "Not sure (...)"
print(result.top(3))  # [(class, probability), ...]
```

Command line: `python -m plantdx.predict leaf.jpg`. Browser:
<https://huggingface.co/spaces/AnarHummatli/plant-disease-detection>.
