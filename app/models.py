import torch
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

MODEL_REGISTRY = {
    "ResNet50": {
        "factory": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        "target_layer": lambda m: m.layer4[-1],
    },
    "EfficientNet-B0": {
        "factory": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
        "target_layer": lambda m: m.features[-1],
    },
    "ConvNeXt-Tiny": {
        "factory": lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT),
        "target_layer": lambda m: m.features[-1],
    },
}
_model_cache: dict[str, torch.nn.Module] = {}

def get_model(name: str) -> torch.nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(MODEL_REGISTRY.keys())}")

    if name not in _model_cache:
        model = MODEL_REGISTRY[name]["factory"]()
        model.eval()
        _model_cache[name] = model

    return _model_cache[name]

def get_target_layer(name: str) -> torch.nn.Module:
    model = get_model(name)
    return MODEL_REGISTRY[name]["target_layer"](model)

def get_model_names() -> list[str]:
    return list(MODEL_REGISTRY.keys())