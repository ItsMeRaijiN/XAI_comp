# XAI_comp
 
Quantitative comparison of Class Activation Mapping methods for convolutional neural networks.
 
## Methods
 
Eight CAM methods grouped by approach:
 
**Gradient-based**
- GradCAM — weighted activation maps using gradient signal
- GradCAM++ — improved localization for multiple object instances
- XGradCAM — axiom-based extension with sensitivity and conservation
**Gradient-free**
- ScoreCAM — perturbation-based, uses forward passes instead of gradients
- AblationCAM — systematic occlusion of activation channels
**Other**
- LayerCAM — pixel-level weighted combination, works on shallow layers
- EigenCAM — PCA of activation maps, no class-specific gradients needed
- FullGrad — combines input gradients with bias gradients across all layers
## Evaluation Metrics
 
| Metric                     | What it measures                                              |
|----------------------------|---------------------------------------------------------------|
| Average Drop (%)           | Confidence loss when keeping only salient regions             |
| Increase in Confidence (%) | Cases where masking irrelevant regions boosts prediction      |
| Pointing Game              | Whether the heatmap maximum falls inside the ground-truth box |
| Inference Time (ms)        | Computational cost per method                                 |
 
## Tech Stack
 
- **PyTorch + torchvision** — pretrained models (ResNet50, VGG16, EfficientNet-B0)
- **pytorch-grad-cam** — unified API for all CAM methods
- **FastAPI** — REST API serving explanations
- **Gradio** — interactive UI for side-by-side comparison
- **Matplotlib / Plotly** — metric visualizations
