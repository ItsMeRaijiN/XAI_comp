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

| Metric                   | What it measures                                               | Better  |
|--------------------------|----------------------------------------------------------------|---------|
| Average Drop (%)         | Confidence loss when keeping only salient regions              | Lower   |
| Increase in Confidence   | Whether masking background boosts prediction                   | Higher  |
| Deletion AUC             | Confidence curve when progressively removing important pixels  | Lower   |
| Insertion AUC            | Confidence curve when progressively revealing important pixels | Higher  |
| Sparsity (%)             | How focused the heatmap is (% of active pixels)                | Lower   |
| Shannon Entropy          | Information spread in the heatmap distribution                 | Lower   |
| Inter-method correlation | Average pairwise Pearson correlation with other methods        | Context |
| Inference Time (ms)      | Computational cost per method                                  | Lower   |
 
## Tech Stack
 
- **PyTorch + torchvision** — pretrained models (ResNet50, EfficientNet-B0, ConvNeXt-Tiny)
- **pytorch-grad-cam** — unified API for all CAM methods
- **FastAPI** — REST API serving explanations
- **Gradio** — interactive UI for single image and batch comparison
- **Matplotlib** — metric visualizations and charts
