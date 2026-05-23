import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import get_model_names, get_device
from app.explainers import generate_cam, get_prediction, CAM_METHODS

def load_imagenet_labels() -> list[str]:
    import json
    import urllib.request
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode())

def run_batch(
    image_dir: str,
    output_dir: str,
    models: list[str] | None = None,
) -> str:
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = get_model_names()

    labels = load_imagenet_labels()
    method_names = list(CAM_METHODS.keys())

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in extensions)

    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    print(f"Device: {get_device()}")
    print(f"Found {len(images)} images, {len(models)} models, {len(method_names)} methods")
    print(f"Total analyses: {len(images) * len(models) * len(method_names)}")
    print()

    csv_path = output_dir / "results.csv"
    rows = []

    for img_idx, img_path in enumerate(images):
        image = Image.open(img_path).convert("RGB")
        print(f"[{img_idx + 1}/{len(images)}] {img_path.name}")

        for model_name in models:
            class_idx, confidence = get_prediction(image, model_name)
            class_name = labels[class_idx] if class_idx < len(labels) else str(class_idx)

            cam_results = generate_cam(image, model_name, target_class=class_idx)

            for method_name in method_names:
                result = cam_results[method_name]
                heatmap = result["heatmap"]

                from app.metrics import compute_metrics
                metrics = compute_metrics(
                    image=image,
                    heatmap=heatmap,
                    model_name=model_name,
                    class_idx=class_idx,
                    original_confidence=confidence,
                    time_ms=result["time_ms"],
                )

                rows.append({
                    "image": img_path.name,
                    "model": model_name,
                    "method": method_name,
                    "predicted_class": class_name,
                    "confidence": round(confidence * 100, 2),
                    "avg_drop": metrics["avg_drop"],
                    "increase_in_conf": metrics["increase_in_conf"],
                    "time_ms": metrics["time_ms"],
                })

            print(f"  {model_name}: {class_name} ({confidence * 100:.1f}%) ✓")

    fieldnames = ["image", "model", "method", "predicted_class", "confidence",
                  "avg_drop", "increase_in_conf", "time_ms"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {csv_path}")

    _generate_charts(rows, method_names, models, output_dir)

    return str(csv_path)

def _generate_charts(
    rows: list[dict],
    method_names: list[str],
    model_names: list[str],
    output_dir: Path,
):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(method_names))
    width = 0.8 / len(model_names)

    for i, model in enumerate(model_names):
        model_rows = [r for r in rows if r["model"] == model]
        avg_drops = []
        for method in method_names:
            method_rows = [r for r in model_rows if r["method"] == method]
            avg_drops.append(np.mean([r["avg_drop"] for r in method_rows]) if method_rows else 0)

        ax.bar(x + i * width, avg_drops, width, label=model)

    ax.set_xlabel("CAM Method")
    ax.set_ylabel("Average Drop (%)")
    ax.set_title("Average Drop by Method and Model (lower is better)")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(method_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "avg_drop_comparison.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(model_names):
        model_rows = [r for r in rows if r["model"] == model]
        avg_times = []
        for method in method_names:
            method_rows = [r for r in model_rows if r["method"] == method]
            avg_times.append(np.mean([r["time_ms"] for r in method_rows]) if method_rows else 0)

        ax.bar(x + i * width, avg_times, width, label=model)

    ax.set_xlabel("CAM Method")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Inference Time by Method and Model")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(method_names, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "time_comparison.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(model_names):
        model_rows = [r for r in rows if r["model"] == model]
        ioc_rates = []
        for method in method_names:
            method_rows = [r for r in model_rows if r["method"] == method]
            if method_rows:
                ioc_rates.append(sum(1 for r in method_rows if r["increase_in_conf"]) / len(method_rows) * 100)
            else:
                ioc_rates.append(0)

        ax.bar(x + i * width, ioc_rates, width, label=model)

    ax.set_xlabel("CAM Method")
    ax.set_ylabel("Increase in Confidence (%)")
    ax.set_title("Percentage of Images Where Masking Boosted Confidence")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(method_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_increase.png", dpi=150)
    plt.close()

    print(f"Charts saved to {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch CAM analysis")
    parser.add_argument("--images", type=str, required=True, help="Path to image folder")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Models to use")
    args = parser.parse_args()

    run_batch(args.images, args.output, args.models)