import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import urllib.request
import gradio as gr
from PIL import Image
from app.models import get_model_names, get_target_layer_names, get_device
from app.explainers import generate_cam, get_prediction, get_top_predictions, CAM_METHODS
from app.metrics import compute_metrics

IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
)

def load_imagenet_labels() -> list[str]:
    with urllib.request.urlopen(IMAGENET_LABELS_URL) as response:
        return json.loads(response.read().decode())

LABELS = load_imagenet_labels()
METHOD_NAMES = list(CAM_METHODS.keys())

METRICS_HELP = (
    "**Avg Drop ↓** — how much confidence drops when only highlighted regions are kept. "
    "Lower = heatmap correctly found what matters.  \n"
    "**Confidence ↑** — did masking the background actually *boost* confidence? "
    "Yes = the method cleanly isolated the object.  \n"
    "**Time (ms)** — generation time. Gradient-free methods (ScoreCAM, AblationCAM) are much slower."
)

def on_model_change(model_name: str):
    layers = get_target_layer_names(model_name)
    default = [l for l in layers if "default" in l]
    return gr.update(choices=layers, value=default[0] if default else layers[-1])

def on_image_upload(image: Image.Image, model_name: str):
    if image is None:
        return gr.update(choices=[], value=None), ""

    image = image.convert("RGB")
    preds = get_top_predictions(image, model_name, k=5)
    choices = [f"{LABELS[idx]} ({conf * 100:.1f}%)" for idx, conf in preds]
    return gr.update(choices=choices, value=choices[0]), choices[0]

def analyze(image: Image.Image, model_name: str, target_class_str: str, layer_name: str):
    if image is None:
        raise gr.Error("Upload an image first.")

    image = image.convert("RGB")

    if target_class_str:
        class_name_from_dropdown = target_class_str.split(" (")[0]
        target_class = LABELS.index(class_name_from_dropdown) if class_name_from_dropdown in LABELS else None
    else:
        target_class = None

    if target_class is not None:
        import torch
        from app.models import get_model, preprocess, get_device as gd
        model = get_model(model_name)
        input_tensor = preprocess(image).unsqueeze(0).to(gd())
        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1)
            confidence = probs[0, target_class].item()
        class_idx = target_class
        class_name = LABELS[class_idx]
    else:
        class_idx, confidence = get_prediction(image, model_name)
        class_name = LABELS[class_idx]

    prediction_text = f"{class_name} ({confidence * 100:.1f}%)"

    cam_results = generate_cam(image, model_name, target_class=class_idx, layer_name=layer_name)
    overlays = []
    table_rows = []

    for method_name in METHOD_NAMES:
        result = cam_results[method_name]
        overlays.append(result["overlay"])

        metrics = compute_metrics(
            image=image,
            heatmap=result["heatmap"],
            model_name=model_name,
            class_idx=class_idx,
            original_confidence=confidence,
            time_ms=result["time_ms"],
        )

        table_rows.append([
            method_name,
            f"{metrics['avg_drop']:.1f}%",
            "Yes" if metrics["increase_in_conf"] else "No",
            f"{metrics['time_ms']:.1f}",
        ])

    return [prediction_text] + overlays + [table_rows]

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="XAI Comparison",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(f"# XAI_comp\nCompare 8 Class Activation Mapping methods side by side. Device: **{get_device()}**")

        with gr.Row():
            image_input = gr.Image(
                type="pil",
                label="Upload image",
                sources=["upload", "clipboard"],
            )
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    choices=get_model_names(),
                    value="ResNet50",
                    label="Model",
                )
                target_class_dropdown = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Target class (top-5 predictions)",
                    interactive=True,
                )
                layer_dropdown = gr.Dropdown(
                    choices=get_target_layer_names("ResNet50"),
                    value="layer4 (default)",
                    label="Target layer",
                )
                run_button = gr.Button("Analyze", variant="primary", size="lg")
                prediction_label = gr.Textbox(label="Analyzing as", interactive=False)

        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[layer_dropdown],
        )

        image_input.change(
            fn=on_image_upload,
            inputs=[image_input, model_dropdown],
            outputs=[target_class_dropdown, prediction_label],
        )

        gr.Markdown("## Heatmaps")
        cam_outputs = []
        with gr.Row(equal_height=True):
            for name in METHOD_NAMES[:4]:
                cam_outputs.append(gr.Image(label=name, type="numpy"))
        with gr.Row(equal_height=True):
            for name in METHOD_NAMES[4:]:
                cam_outputs.append(gr.Image(label=name, type="numpy"))

        gr.Markdown("## Metrics")
        gr.Markdown(METRICS_HELP)
        metrics_table = gr.Dataframe(
            headers=["Method", "Avg Drop ↓", "Confidence ↑", "Time (ms)"],
            label="Metrics",
            interactive=False,
        )

        run_button.click(
            fn=analyze,
            inputs=[image_input, model_dropdown, target_class_dropdown, layer_dropdown],
            outputs=[prediction_label] + cam_outputs + [metrics_table],
        )

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()