import numpy as np
import torch
from PIL import Image
from app.models import get_model, preprocess

def _resize_heatmap_to_image(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    pil = pil.resize(size, Image.Resampling.BILINEAR)
    return np.array(pil).astype(np.float32) / 255.0

def _masked_confidence(
    image: Image.Image,
    heatmap: np.ndarray,
    model_name: str,
    class_idx: int,
) -> float:
    model = get_model(model_name)
    img_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized).astype(np.float32) / 255.0

    heatmap_resized = _resize_heatmap_to_image(heatmap, (224, 224))
    mask_3ch = np.stack([heatmap_resized] * 3, axis=-1)

    masked = (img_array * mask_3ch * 255).astype(np.uint8)
    masked_pil = Image.fromarray(masked)

    input_tensor = preprocess(masked_pil).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)

    return probs[0, class_idx].item()

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
        "time_ms": time_ms,
    }