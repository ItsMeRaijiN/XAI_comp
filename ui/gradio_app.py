import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr
from PIL import Image

from app.labels import get_labels
from app.models import get_model_names, get_target_layer_names, get_device
from app.explainers import (
    generate_cam,
    get_prediction,
    get_top_predictions,
    get_class_confidence,
    CAM_METHODS,
)
from app.metrics import compute_metrics, compute_agreement

LABELS = get_labels()
METHOD_NAMES = list(CAM_METHODS.keys())

METRICS_HELP = (
    "**Avg Drop ↓** — confidence loss when only salient regions kept. Lower = better localization.  \n"
    "**Confidence ↑** — did masking background boost confidence? Yes = clean isolation.  \n"
    "**DAUC ↓** — Deletion AUC: progressively remove important pixels. Lower = better faithfulness.  \n"
    "**IAUC ↑** — Insertion AUC: progressively reveal important pixels. Higher = better faithfulness.  \n"
    "**Sparsity ↓** — % of heatmap active. Lower = more focused.  \n"
    "**Entropy ↓** — Shannon entropy of heatmap distribution. Lower = more concentrated.  \n"
    "**Agreement** — avg correlation with other methods. High = consensus, low = outlier.  \n"
    "**Time (ms)** — generation time."
)

SINGLE_HEADERS = [
    "Method", "Avg Drop ↓", "Conf ↑", "DAUC ↓", "IAUC ↑",
    "Sparsity ↓", "Entropy ↓", "Agreement", "Time (ms)",
]

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

def _parse_target_class(target_class_str: str | None) -> int | None:
    if not target_class_str:
        return None
    class_name = target_class_str.split(" (")[0]
    return LABELS.index(class_name) if class_name in LABELS else None

def analyze(image: Image.Image, model_name: str, target_class_str: str, layer_name: str):
    if image is None:
        raise gr.Error("Upload an image first.")

    image = image.convert("RGB")
    target_class = _parse_target_class(target_class_str)

    if target_class is not None:
        class_idx = target_class
        confidence = get_class_confidence(image, model_name, class_idx)
    else:
        class_idx, confidence = get_prediction(image, model_name)

    class_name = LABELS[class_idx]
    prediction_text = f"{class_name} ({confidence * 100:.1f}%)"
    cam_results = generate_cam(image, model_name, target_class=class_idx, layer_name=layer_name)
    heatmaps_dict = {name: cam_results[name]["heatmap"] for name in METHOD_NAMES}
    agreement = compute_agreement(heatmaps_dict)
    overlays = [cam_results[name]["overlay"] for name in METHOD_NAMES]
    yield [prediction_text] + overlays + [[]]
    table_rows = []
    for method_name in METHOD_NAMES:
        result = cam_results[method_name]
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
            f"{metrics['deletion_auc']:.3f}",
            f"{metrics['insertion_auc']:.3f}",
            f"{metrics['sparsity']:.1f}%",
            f"{metrics['entropy']:.1f}",
            f"{agreement[method_name]:.2f}",
            f"{metrics['time_ms']:.1f}",
        ])
        yield [prediction_text] + overlays + [list(table_rows)]

def run_batch_ui(folder_path: str, selected_models, progress=gr.Progress()):
    if not folder_path or not folder_path.strip():
        raise gr.Error("Enter a folder path.")
    if not selected_models:
        raise gr.Error("Select at least one model.")

    folder = Path(folder_path.strip())
    if not folder.exists():
        raise gr.Error(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise gr.Error(f"Not a folder: {folder}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted(f for f in folder.iterdir() if f.suffix.lower() in extensions)

    if not image_files:
        raise gr.Error(f"No images found in {folder}")

    model_names = selected_models
    rows = []
    total = len(image_files) * len(model_names)
    done = 0

    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")
        img_name = img_path.name

        for model_name in model_names:
            progress(done / total, desc=f"{img_name} — {model_name}")
            class_idx, confidence = get_prediction(image, model_name)
            class_name = LABELS[class_idx] if class_idx < len(LABELS) else str(class_idx)
            cam_results = generate_cam(image, model_name, target_class=class_idx)

            heatmaps_dict = {name: cam_results[name]["heatmap"] for name in METHOD_NAMES}
            agreement = compute_agreement(heatmaps_dict)

            for method_name in METHOD_NAMES:
                result = cam_results[method_name]
                metrics = compute_metrics(
                    image=image,
                    heatmap=result["heatmap"],
                    model_name=model_name,
                    class_idx=class_idx,
                    original_confidence=confidence,
                    time_ms=result["time_ms"],
                )
                rows.append({
                    "image": img_name,
                    "model": model_name,
                    "method": method_name,
                    "predicted_class": class_name,
                    "confidence": round(confidence * 100, 2),
                    "avg_drop": metrics["avg_drop"],
                    "increase_in_conf": metrics["increase_in_conf"],
                    "deletion_auc": metrics["deletion_auc"],
                    "insertion_auc": metrics["insertion_auc"],
                    "sparsity": metrics["sparsity"],
                    "entropy": metrics["entropy"],
                    "agreement": agreement[method_name],
                    "time_ms": metrics["time_ms"],
                })

            done += 1

    csv_path = Path(tempfile.mkdtemp()) / "results.csv"
    fieldnames = [
        "image", "model", "method", "predicted_class", "confidence",
        "avg_drop", "increase_in_conf", "deletion_auc", "insertion_auc",
        "sparsity", "entropy", "agreement", "time_ms",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    chart_paths = _generate_charts(rows, METHOD_NAMES, model_names)

    summary_rows = []
    for method in METHOD_NAMES:
        for model in model_names:
            mr = [r for r in rows if r["method"] == method and r["model"] == model]
            if mr:
                summary_rows.append([
                    method, model,
                    f"{np.mean([r['avg_drop'] for r in mr]):.1f}%",
                    f"{sum(1 for r in mr if r['increase_in_conf']) / len(mr) * 100:.0f}%",
                    f"{np.mean([r['deletion_auc'] for r in mr]):.3f}",
                    f"{np.mean([r['insertion_auc'] for r in mr]):.3f}",
                    f"{np.mean([r['sparsity'] for r in mr]):.1f}%",
                    f"{np.mean([r['entropy'] for r in mr]):.1f}",
                    f"{np.mean([r['agreement'] for r in mr]):.2f}",
                    f"{np.mean([r['time_ms'] for r in mr]):.0f}",
                ])

    return (
        str(csv_path),
        summary_rows,
        chart_paths[0], chart_paths[1], chart_paths[2],
        chart_paths[3], chart_paths[4], chart_paths[5],
        f"Analyzed {len(rows)} results",
    )

def _make_bar_chart(rows, method_names, model_names, metric_key, ylabel, title, log_scale=False):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(method_names))
    width = 0.8 / len(model_names)

    for i, model in enumerate(model_names):
        model_rows = [r for r in rows if r["model"] == model]
        if callable(metric_key):
            vals = [metric_key(model_rows, m) for m in method_names]
        else:
            vals = [np.mean([r[metric_key] for r in model_rows if r["method"] == m]) for m in method_names]
        ax.bar(x + i * width, vals, width, label=model)

    ax.set_xlabel("CAM Method")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(method_names, rotation=45, ha="right")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = Path(tempfile.mkdtemp()) / f"{ylabel.lower().replace(' ', '_')}.png"
    plt.savefig(p, dpi=150)
    plt.close()
    return str(p)

def _generate_charts(rows, method_names, model_names):
    def ioc_rate(model_rows, m):
        mr = [r for r in model_rows if r["method"] == m]
        return sum(1 for r in mr if r["increase_in_conf"]) / len(mr) * 100 if mr else 0

    return [
        _make_bar_chart(rows, method_names, model_names,
                        "avg_drop", "Average Drop (%)", "Average Drop by Method and Model (lower is better)"),
        _make_bar_chart(rows, method_names, model_names,
                        "time_ms", "Time (ms)", "Inference Time by Method and Model", log_scale=True),
        _make_bar_chart(rows, method_names, model_names,
                        ioc_rate, "Confidence Increase (%)", "Percentage of Images Where Masking Boosted Confidence"),
        _make_bar_chart(rows, method_names, model_names,
                        "deletion_auc", "Deletion AUC", "Deletion AUC by Method and Model (lower is better)"),
        _make_bar_chart(rows, method_names, model_names,
                        "insertion_auc", "Insertion AUC", "Insertion AUC by Method and Model (higher is better)"),
        _make_bar_chart(rows, method_names, model_names,
                        "sparsity", "Sparsity (%)", "Heatmap Sparsity by Method and Model (lower = more focused)"),
    ]

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="XAI Comparison",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(f"# XAI_comp\nCompare 8 Class Activation Mapping methods. Device: **{get_device()}**")

        with gr.Tabs():
            with gr.TabItem("Single Image"):
                with gr.Row():
                    image_input = gr.Image(
                        type="pil",
                        label="Upload image",
                        sources=["upload", "clipboard"],
                    )
                    with gr.Column(variant="panel"):
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
                with gr.Accordion("Jak interpretować metryki?", open=False):
                    gr.Markdown(METRICS_HELP)
                metrics_table = gr.Dataframe(
                    headers=SINGLE_HEADERS,
                    label="Metrics",
                    interactive=False,
                )

                run_button.click(
                    fn=analyze,
                    inputs=[image_input, model_dropdown, target_class_dropdown, layer_dropdown],
                    outputs=[prediction_label] + cam_outputs + [metrics_table],
                )

            with gr.TabItem("Batch Analysis"):
                gr.Markdown("Point to a folder with images to compare all CAM methods across models.")

                with gr.Row():
                    batch_folder = gr.Textbox(
                        label="Image folder path",
                        placeholder="data/samples",
                        value="data/samples",
                    )
                    with gr.Column():
                        batch_models = gr.CheckboxGroup(
                            choices=get_model_names(),
                            value=get_model_names(),
                            label="Models to analyze",
                        )
                        batch_button = gr.Button("Run Batch Analysis", variant="primary", size="lg")
                        batch_status = gr.Textbox(label="Status", interactive=False)

                batch_csv = gr.File(label="Download CSV results")

                gr.Markdown("## Summary")
                batch_summary = gr.Dataframe(
                    headers=[
                        "Method", "Model", "Avg Drop ↓", "Conf ↑ %",
                        "DAUC ↓", "IAUC ↑", "Sparsity ↓", "Entropy ↓",
                        "Agreement", "Time (ms)",
                    ],
                    label="Aggregated metrics",
                    interactive=False,
                )

                gr.Markdown("## Charts")
                with gr.Row():
                    chart_drop = gr.Image(label="Average Drop", type="filepath")
                    chart_time = gr.Image(label="Inference Time", type="filepath")
                with gr.Row():
                    chart_conf = gr.Image(label="Confidence Increase", type="filepath")
                    chart_dauc = gr.Image(label="Deletion AUC", type="filepath")
                with gr.Row():
                    chart_iauc = gr.Image(label="Insertion AUC", type="filepath")
                    chart_sparsity = gr.Image(label="Sparsity", type="filepath")

                batch_button.click(
                    fn=run_batch_ui,
                    inputs=[batch_folder, batch_models],
                    outputs=[
                        batch_csv, batch_summary,
                        chart_drop, chart_time, chart_conf,
                        chart_dauc, chart_iauc, chart_sparsity,
                        batch_status,
                    ],
                )
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(theme=gr.themes.Soft(), ssr_mode=False)