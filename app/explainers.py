import time
import numpy as np
from PIL import Image
from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    XGradCAM,
    ScoreCAM,
    AblationCAM,
    LayerCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from app.models import get_model, get_target_layer, preprocess, get_device

DISPLAY_SIZE = (640, 640)

CAM_METHODS = {
    "GradCAM": GradCAM,
    "GradCAM++": GradCAMPlusPlus,
    "XGradCAM": XGradCAM,
    "ScoreCAM": ScoreCAM,
    "AblationCAM": AblationCAM,
    "LayerCAM": LayerCAM,
    "EigenCAM": EigenCAM,
    "FullGrad": FullGrad,
}

def _resize_heatmap(heatmap: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    pil = pil.resize((w, h), Image.Resampling.BILINEAR)
    return np.array(pil).astype(np.float32) / 255.0

def generate_cam(
    image: Image.Image,
    model_name: str,
    target_class: int | None = None,
    layer_name: str | None = None,
) -> dict[str, dict]:
    model = get_model(model_name)
    target_layer = get_target_layer(model_name, layer_name)
    input_tensor = preprocess(image).unsqueeze(0).to(get_device())

    display_image = image.resize(DISPLAY_SIZE, Image.Resampling.LANCZOS)
    rgb_display = np.array(display_image).astype(np.float32) / 255.0
    display_hw = (DISPLAY_SIZE[1], DISPLAY_SIZE[0])

    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

    results = {}

    for method_name, cam_class in CAM_METHODS.items():
        cam = cam_class(model=model, target_layers=[target_layer])
        start = time.perf_counter()
        heatmap = cam(input_tensor=input_tensor, targets=targets)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        heatmap_large = _resize_heatmap(heatmap, display_hw)
        overlay = show_cam_on_image(rgb_display, heatmap_large, use_rgb=True)

        results[method_name] = {
            "heatmap": heatmap,
            "overlay": overlay,
            "time_ms": round(elapsed_ms, 1),
        }

    return results

def get_prediction(image: Image.Image, model_name: str) -> tuple[int, float]:
    import torch
    model = get_model(model_name)
    input_tensor = preprocess(image).unsqueeze(0).to(get_device())

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, class_idx = probs.max(dim=1)

    return class_idx.item(), confidence.item()

def get_top_predictions(image: Image.Image, model_name: str, k: int = 5) -> list[tuple[int, str, float]]:
    import torch
    model = get_model(model_name)
    input_tensor = preprocess(image).unsqueeze(0).to(get_device())

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = probs.topk(k, dim=1)

    return [(idx.item(), top_probs[0, i].item()) for i, idx in enumerate(top_indices[0])]