import torch
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

MODEL_REGISTRY = {
    "ResNet50": {
        "factory": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        "target_layer": lambda m: m.layer4[-1],
        "all_target_layers": lambda m: {
            "layer1 (shallow)": m.layer1[-1],
            "layer2": m.layer2[-1],
            "layer3": m.layer3[-1],
            "layer4 (default)": m.layer4[-1],
        },
    },
    "EfficientNet-B0": {
        "factory": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
        "target_layer": lambda m: m.features[-1],
        "all_target_layers": lambda m: {
            "features[2] (shallow)": m.features[2],
            "features[4]": m.features[4],
            "features[6]": m.features[6],
            "features[8] (default)": m.features[8],
        },
    },
    "ConvNeXt-Tiny": {
        "factory": lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT),
        "target_layer": lambda m: m.features[-1],
        "all_target_layers": lambda m: {
            "features[1] (shallow)": m.features[1],
            "features[3]": m.features[3],
            "features[5]": m.features[5],
            "features[7] (default)": m.features[7],
        },
    },
}

_model_cache: dict[str, torch.nn.Module] = {}

def get_device() -> torch.device:
    return DEVICE

def get_model(name: str) -> torch.nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(MODEL_REGISTRY.keys())}")

    if name not in _model_cache:
        model = MODEL_REGISTRY[name]["factory"]()
        model.eval()
        model.to(DEVICE)
        _model_cache[name] = model

    return _model_cache[name]

def get_target_layer(name: str, layer_name: str | None = None) -> torch.nn.Module:
    model = get_model(name)
    if layer_name is None or layer_name == "":
        return MODEL_REGISTRY[name]["target_layer"](model)

    all_layers = MODEL_REGISTRY[name]["all_target_layers"](model)
    if layer_name not in all_layers:
        raise ValueError(f"Unknown layer: {layer_name}. Choose from: {list(all_layers.keys())}")
    return all_layers[layer_name]

def get_target_layer_names(name: str) -> list[str]:
    model = get_model(name)
    return list(MODEL_REGISTRY[name]["all_target_layers"](model).keys())

def get_model_names() -> list[str]:
    return list(MODEL_REGISTRY.keys())