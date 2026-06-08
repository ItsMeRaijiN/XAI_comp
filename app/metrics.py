import numpy as np
import torch
from PIL import Image, ImageFilter

from app.models import get_model, preprocess, get_device
from app.explainers import resize_heatmap


def _get_confidence(model, input_tensor, class_idx: int) -> float:
    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)
    return probs[0, class_idx].item()


def _masked_confidence(
    image: Image.Image,
    heatmap: np.ndarray,
    model_name: str,
    class_idx: int,
) -> float:
    model = get_model(model_name)
    img_array = np.array(
        image.resize((224, 224), Image.Resampling.LANCZOS)
    ).astype(np.float32) / 255.0

    hm = resize_heatmap(heatmap, (224, 224))
    masked = (img_array * np.stack([hm] * 3, axis=-1) * 255).astype(np.uint8)

    tensor = preprocess(Image.fromarray(masked)).unsqueeze(0).to(get_device())
    return _get_confidence(model, tensor, class_idx)

def _auc(
    image: Image.Image,
    heatmap: np.ndarray,
    model_name: str,
    class_idx: int,
    steps: int = 20,
    mode: str = "deletion",
) -> float:
    model = get_model(model_name)
    device = get_device()

    img = np.array(
        image.resize((224, 224), Image.Resampling.LANCZOS)
    ).astype(np.float32) / 255.0

    hm = resize_heatmap(heatmap, (224, 224))
    order = np.argsort(hm.flatten())[::-1]
    pixels_per_step = len(order) // steps

    if mode == "insertion":
        blurred = (
            image.resize((224, 224), Image.Resampling.LANCZOS)
            .filter(ImageFilter.GaussianBlur(radius=10))
        )
        current = np.array(blurred).astype(np.float32) / 255.0
    else:
        current = img.copy()

    tensors: list[torch.Tensor] = []
    for step in range(steps + 1):
        pil = Image.fromarray((current * 255).astype(np.uint8))
        tensors.append(preprocess(pil))

        if step < steps:
            start = step * pixels_per_step
            end = min(start + pixels_per_step, len(order))
            idx = order[start:end]
            rows, cols = idx // 224, idx % 224
            if mode == "deletion":
                current[rows, cols, :] = 0.0
            else:
                current[rows, cols, :] = img[rows, cols, :]

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        confs = torch.softmax(model(batch), dim=1)[:, class_idx].cpu().tolist()

    return round(float(np.trapezoid(confs, dx=1.0 / steps)), 4)


def sparsity(heatmap: np.ndarray, threshold: float = 0.15) -> float:
    return round(np.sum(heatmap > threshold) / heatmap.size * 100, 1)


def entropy(heatmap: np.ndarray) -> float:
    flat = heatmap.flatten().astype(np.float64) + 1e-10
    flat /= flat.sum()
    return round(float(-np.sum(flat * np.log2(flat))), 2)


def compute_metrics(
    image: Image.Image,
    heatmap: np.ndarray,
    model_name: str,
    class_idx: int,
    original_confidence: float,
    time_ms: float,
) -> dict:
    masked_conf = _masked_confidence(image, heatmap, model_name, class_idx)
    drop = max(0.0, original_confidence - masked_conf) / original_confidence * 100

    return {
        "avg_drop": round(drop, 2),
        "increase_in_conf": masked_conf > original_confidence,
        "deletion_auc": _auc(image, heatmap, model_name, class_idx, mode="deletion"),
        "insertion_auc": _auc(image, heatmap, model_name, class_idx, mode="insertion"),
        "sparsity": sparsity(heatmap),
        "entropy": entropy(heatmap),
        "time_ms": time_ms,
    }


def compute_agreement(heatmaps: dict[str, np.ndarray]) -> dict[str, float]:
    names = list(heatmaps.keys())
    flat = {n: heatmaps[n].flatten().astype(np.float64) for n in names}

    result: dict[str, float] = {}
    for i, ni in enumerate(names):
        corrs = [
            (lambda c: c if not np.isnan(c) else 0.0)(
                np.corrcoef(flat[ni], flat[nj])[0, 1]
            )
            for j, nj in enumerate(names) if i != j
        ]
        result[ni] = round(np.mean(corrs), 3) if corrs else 0.0
    return result