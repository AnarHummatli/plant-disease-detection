---
title: Plant Disease Detection
emoji: 🌿
colorFrom: green
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: Leaf photo to plant disease, with Grad-CAM and "not sure"
tags:
  - image-classification
  - plant-disease
  - mobilenetv2
  - transfer-learning
  - grad-cam
  - onnx
  - plantvillage
---

# Plant disease detection · Yarpaq Həkimləri

Upload a photo of one leaf of a supported crop (14 crops, 38 classes). The page gives the
diagnosis, or "not sure" below a confidence of 0.947, the three most likely classes and a
Grad-CAM heat map of the region that drove the answer. The model runs in your browser with
ONNX Runtime Web: photos are not uploaded anywhere.

- **Model:** MobileNetV2 fine-tuned on PlantVillage; test macro-F1 0.982 on a leaf-grouped,
  leakage-checked split, scored once.
- **Same answers as the Python model:** the photo is resized with a port of TensorFlow's resize
  (bit-identical on the checked images) and the ONNX export is checked against Keras.
- **Code, data checks and the full evaluation:**
  <https://github.com/AnarHummatli/plant-disease-detection> (this Space is built by
  `python -m plantdx.web_export` from the release model).
- **Limits:** trained on single leaves photographed on plain lab backgrounds; field photos and
  plants outside the 38 classes are not handled reliably.

Educational prototype: it gives preliminary information only and does not replace an
agronomist's diagnosis. Example photos: PlantVillage (Hughes & Salathé, 2015), CC BY-SA 3.0,
see `examples/README.md`.
