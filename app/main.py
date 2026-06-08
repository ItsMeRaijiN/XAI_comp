import base64
import io

import numpy as np
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from PIL import Image

from app.labels import get_labels
from app.models import get_model_names, get_target_layer_names
from app.explainers import generate_cam, get_prediction, CAM_METHODS
from app.metrics import compute_metrics

app = FastAPI(title="XAI Comparison API")


def _numpy_to_base64(img: np.ndarray) -> str:
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@app.get("/models")
def list_models():
    return {"models": get_model_names()}

@app.get("/layers/{model_name}")
def list_layers(model_name: str):
    return {"layers": get_target_layer_names(model_name)}

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    model_name: str = Query(default="ResNet50"),
    target_class: int | None = Query(default=None),
    layer_name: str | None = Query(default=None),
):
    labels = get_labels()
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    class_idx, confidence = get_prediction(image, model_name)
    if target_class is not None:
        class_idx = target_class
    class_name = labels[class_idx] if labels else str(class_idx)

    cam_results = generate_cam(image, model_name, target_class=class_idx, layer_name=layer_name)

    methods = []
    for method_name in CAM_METHODS:
        result = cam_results[method_name]
        metrics = compute_metrics(
            image=image,
            heatmap=result["heatmap"],
            model_name=model_name,
            class_idx=class_idx,
            original_confidence=confidence,
            time_ms=result["time_ms"],
        )
        methods.append({
            "name": method_name,
            "overlay": _numpy_to_base64(result["overlay"]),
            "metrics": metrics,
        })

    return JSONResponse({
        "prediction": {
            "class_name": class_name,
            "class_idx": class_idx,
            "confidence": round(confidence, 4),
        },
        "methods": methods,
    })