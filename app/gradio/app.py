"""Gradio demo: upload a leaf photo, get the diagnosis, the top-3 and a Grad-CAM heat map.

    python app/gradio/app.py             # http://127.0.0.1:7860
    python app/gradio/app.py --share     # temporary public link (e.g. from Colab)
    make demo

The model is the final MobileNetV2 of release v1.0, downloaded once and checked against its
SHA-256 (`plantdx.predict`, D-021). Below a confidence of 0.947 (chosen on validation, D-019)
the answer is "not sure" instead of a diagnosis.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from plantdx.data.source import CROPS, crop_name, display_name, is_healthy
from plantdx.predict import MODEL_VERSION, NOT_SURE_THRESHOLD, Prediction, Predictor

HERE = Path(__file__).resolve().parent
EXAMPLES = sorted((HERE / "examples").glob("*.jpg"))
REPO = "https://github.com/AnarHummatli/plant-disease-detection"

TITLE = "Plant disease detection · Yarpaq Həkimləri"
INTRO = f"""# 🌿 Plant disease detection
Upload a photo of **one leaf** of a supported crop. The model names the disease (or "healthy"),
shows its three most likely classes, and highlights the part of the photo that drove the answer.

**Supported crops (14):** {", ".join(CROPS)}.
"""
ABOUT = f"""
**Model.** MobileNetV2 (ImageNet weights) with the whole backbone fine-tuned on PlantVillage,
38 classes. On the official test split (10,709 lab photos, scored once): **macro-F1 0.982,
accuracy 98.6%**, top-3 accuracy 99.96%. The split keeps every leaf on one side, so photos of a
test leaf were never seen in training.

**"Not sure".** If the top class has a probability below {NOT_SURE_THRESHOLD:.1%} (a threshold
chosen on the validation split), the demo does not give a diagnosis. On the test split this
happens for 3.2% of the photos; the answered ones are 99.5% correct.

**Heat map (Grad-CAM).** Red areas raised the score of the top class the most. The map is
coarse (about 1/7 of the photo per cell): it shows a region, not single spots.

**Limits.** The training photos are single leaves on plain lab backgrounds. Field photos
(several leaves, soil, strong shadows) are harder and were not evaluated systematically. Crops
and diseases outside the 38 classes are not recognised as unknown: they get the closest known
class, often with high confidence. Details: [repository]({REPO}) · model release {MODEL_VERSION}.
"""
DISCLAIMER = (
    "⚠️ Educational prototype. It gives preliminary information only and does not replace an "
    "agronomist's diagnosis."
)
CSS = ".verdict h3 {margin: 0.2em 0;} footer {display: none !important;}"


def verdict_markdown(pred: Prediction) -> str:
    """The answer for people: diagnosis, healthy, or "not sure"."""
    conf = f"confidence **{pred.confidence:.1%}**"
    if pred.not_sure:
        return (
            f"### 🟡 Not sure\nThe best guess is *{display_name(pred.label)}*, but with {conf} "
            f"(below {pred.threshold:.1%}). Please take a clearer photo: one whole leaf, in focus, "
            "filling most of the frame."
        )
    crop = crop_name(pred.label)
    if is_healthy(pred.label):
        return f"### 🟢 {crop}: healthy\n{conf}"
    disease = display_name(pred.label).split(" · ", 1)[1]
    return (
        f"### 🔴 {crop}: {disease}\n{conf}. A preliminary result: confirm it with an "
        "agronomist before treating."
    )


def respond(predictor: Predictor, image) -> tuple[str, dict | None, object, str]:
    """(verdict, top-3 for gr.Label, Grad-CAM overlay, timing line) for one uploaded photo."""
    if image is None:
        return "Upload a leaf photo or pick an example.", None, None, ""
    t0 = time.perf_counter()
    pred = predictor.predict(image)
    ms = (time.perf_counter() - t0) * 1000
    top = {display_name(name): p for name, p in pred.top(3)}
    info = (
        f"Answered in {ms:.0f} ms (photo → prepared 224 × 224 image → model → Grad-CAM). "
        f"Heat map for: {display_name(pred.label)}."
    )
    return verdict_markdown(pred), top, pred.overlay(), info


def build(predictor: Predictor):
    import gradio as gr

    def diagnose(image):
        try:
            return respond(predictor, image)
        except Exception as err:  # unreadable upload: say so instead of a stack trace
            raise gr.Error(f"Could not read this file as an image ({type(err).__name__}).") from err

    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(INTRO)
        with gr.Row(equal_height=False):
            with gr.Column():
                photo = gr.Image(
                    type="filepath",
                    sources=["upload", "webcam", "clipboard"],
                    label="Leaf photo",
                    height=360,
                )
                gr.Examples(
                    examples=[[str(p)] for p in EXAMPLES],
                    inputs=photo,
                    example_labels=[p.stem.replace("_", " ") for p in EXAMPLES],
                    label="Examples (test photos the model never saw in training)",
                )
            with gr.Column():
                verdict = gr.Markdown(elem_classes="verdict")
                top = gr.Label(num_top_classes=3, label="Top-3 classes", show_heading=False)
                cam = gr.Image(
                    label="Grad-CAM: where the model looked", interactive=False, height=300
                )
                info = gr.Markdown()
        with gr.Accordion("About the model and its limits", open=False):
            gr.Markdown(ABOUT)
        gr.Markdown(DISCLAIMER)
        photo.change(diagnose, inputs=photo, outputs=[verdict, top, cam, info])
    return demo


def main(argv: list[str] | None = None) -> None:
    import gradio as gr

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--share", action="store_true", help="create a temporary public link")
    parser.add_argument("--port", type=int, default=None, help="default 7860")
    parser.add_argument("--model", help=f"local inference.keras (default: release {MODEL_VERSION})")
    args = parser.parse_args(argv)

    predictor = Predictor(args.model).warmup()
    build(predictor).queue(default_concurrency_limit=2).launch(
        share=args.share,
        server_port=args.port,
        max_file_size="15mb",
        theme=gr.themes.Soft(primary_hue="green"),
        css=CSS,
    )


if __name__ == "__main__":
    main()
