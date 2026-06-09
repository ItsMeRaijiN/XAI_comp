import numpy as np
import torch
from PIL import Image, ImageFilter

from app.models import get_model, get_device, IMAGENET_MEAN, IMAGENET_STD
from app.explainers import resize_heatmap

INPUT_SIZE = 224
AUC_STEPS = 20

try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz

_MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

def _to_normalized_batch(images: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    t = (t - _MEAN) / _STD
    return t.to(device)


def _prepare_image(image: Image.Image) -> np.ndarray:
    return np.asarray(
        image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
    ).astype(np.float32) / 255.0

def _build_auc_frames(
    img: np.ndarray,
    order: np.ndarray,
    mode: str,
    blurred: np.ndarray | None,
    steps: int,
) -> np.ndarray:
    current = (blurred if mode == "insertion" else img).copy()
    frames = np.empty((steps + 1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    pixels_per_step = len(order) // steps

    for step in range(steps + 1):
        frames[step] = current
        if step < steps:
            idx = order[step * pixels_per_step:(step + 1) * pixels_per_step]
            rows, cols = np.divmod(idx, INPUT_SIZE)
            if mode == "deletion":
                current[rows, cols, :] = 0.0
            else:
                current[rows, cols, :] = img[rows, cols, :]

    return frames

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
    model = get_model(model_name)
    device = get_device()

    img = _prepare_image(image)
    hm = resize_heatmap(heatmap, (INPUT_SIZE, INPUT_SIZE))
    blurred = np.asarray(
        image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
        .filter(ImageFilter.GaussianBlur(radius=10))
    ).astype(np.float32) / 255.0

    order = np.argsort(hm.flatten())[::-1]

    masked = img * hm[..., None]
    frames = np.concatenate([
        masked[None],
        _build_auc_frames(img, order, "deletion", None, AUC_STEPS),
        _build_auc_frames(img, order, "insertion", blurred, AUC_STEPS),
    ])

    batch = _to_normalized_batch(frames, device)
    with torch.no_grad():
        confs = torch.softmax(model(batch), dim=1)[:, class_idx].cpu().numpy()

    masked_conf = float(confs[0])
    deletion_confs = confs[1:AUC_STEPS + 2]
    insertion_confs = confs[AUC_STEPS + 2:]

    drop = max(0.0, original_confidence - masked_conf) / original_confidence * 100

    return {
        "avg_drop": round(drop, 2),
        "increase_in_conf": masked_conf > original_confidence,
        "deletion_auc": round(float(_trapezoid(deletion_confs, dx=1.0 / AUC_STEPS)), 4),
        "insertion_auc": round(float(_trapezoid(insertion_confs, dx=1.0 / AUC_STEPS)), 4),
        "sparsity": sparsity(heatmap),
        "entropy": entropy(heatmap),
        "time_ms": time_ms,
    }

def compute_agreement(heatmaps: dict[str, np.ndarray]) -> dict[str, float]:
    names = list(heatmaps.keys())
    flat = np.stack([heatmaps[n].flatten().astype(np.float64) for n in names])

    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(flat), nan=0.0)

    result: dict[str, float] = {}
    for i, name in enumerate(names):
        others = np.delete(corr[i], i)
        result[name] = round(float(others.mean()), 3) if others.size else 0.0
    return result