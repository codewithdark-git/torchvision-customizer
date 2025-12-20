<p align="center">
  <img src="https://raw.githubusercontent.com/codewithdark-git/torchvision-customizer/main/assets/logo.png" alt="torchvision-customizer" width="400"/>
</p>

<h1 align="center">torchvision-customizer</h1>

<p align="center">
  <strong>Build, customize, and fine-tune CNN architectures with unprecedented flexibility</strong>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.5+-ee4c2c.svg" alt="PyTorch 2.5+"></a>
  <a href="https://pypi.org/project/torchvision-customizer/"><img src="https://img.shields.io/badge/version-2.1.0-green.svg" alt="Version"></a>
</p>

<p align="center">
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-v21-hybrid-models">v2.1 Hybrid</a> •
  <a href="#-documentation">Docs</a>
</p>

---

## ✨ What is torchvision-customizer?

**torchvision-customizer** is a production-ready Python package that empowers researchers and developers to:

- 🔧 **Build custom CNNs** from scratch with a fluent 3-tier API
- 🔀 **Customize pre-trained models** by injecting attention, replacing blocks, and modifying architectures
- ⚡ **Fine-tune efficiently** with smart weight loading and selective freezing
- 📝 **Define architectures declaratively** using YAML recipes with macros and inheritance

Whether you're prototyping a new architecture or fine-tuning ResNet with CBAM attention, this library makes it effortless.

---

## 📦 Installation

### From GitHub (Latest)
```bash
pip install git+https://github.com/codewithdark-git/torchvision-customizer.git
```

### From Source (Development)
```bash
git clone https://github.com/codewithdark-git/torchvision-customizer.git
cd torchvision-customizer
pip install -e ".[dev]"
```

---

## 🚀 Quick Start

### Option 1: Customize a Pre-trained Model (v2.1 🆕)

```python
from torchvision_customizer import HybridBuilder

# Load ResNet50 with ImageNet weights, add attention, customize head
model = HybridBuilder().from_torchvision(
    "resnet50",
    weights="IMAGENET1K_V2",
    patches={
        "layer3": {"wrap": "se"},           # Add SE attention
        "layer4": {"wrap": "cbam_block"},   # Add CBAM attention
    },
    num_classes=100,
    freeze_backbone=True,
    unfreeze_stages=[3],  # Only train last stage + head
)

# Ready for fine-tuning!
print(f"Trainable params: {model.count_parameters(trainable_only=True):,}")
```

### Option 2: Build from Scratch with Composer API

```python
from torchvision_customizer import Stem, Stage, Head

model = (
    Stem(64, kernel=7, stride=2)
    >> Stage(64, blocks=3, pattern='residual')
    >> Stage(128, blocks=4, pattern='residual+se', downsample=True)
    >> Stage(256, blocks=6, pattern='residual+se', downsample=True)
    >> Stage(512, blocks=3, pattern='residual', downsample=True)
    >> Head(num_classes=1000)
)

print(model.explain())
```

### Option 3: Use Declarative Recipes

```python
from torchvision_customizer import Recipe, build_recipe

recipe = Recipe(
    stem="conv(64, k=7, s=2)",
    stages=[
        "residual(64) x 3",
        "residual(128) x 4 | downsample",
        "residual(256) x 6 | downsample",
        "residual(512) x 3 | downsample",
    ],
    head="linear(1000)"
)
model = build_recipe(recipe)
```

### Option 4: Load from YAML (v2.1 🆕)

```yaml
# model.yaml
backbone:
  name: resnet50
  weights: IMAGENET1K_V2
  patches:
    layer4: {wrap: cbam_block}
head:
  num_classes: 10
  dropout: 0.3
```

```python
from torchvision_customizer.recipe import load_yaml_recipe
model = load_yaml_recipe("model.yaml")
```

### Option 5: Use the CLI (v2.1 🆕)

```bash
# Build and save model
tvc build --yaml model.yaml --output model.pt

# Benchmark performance
tvc benchmark --yaml model.yaml --device cuda --batch-size 32

# Export to ONNX
tvc export --yaml model.yaml --format onnx --output model.onnx
```

---

## 🌟 Key Features

### 🔧 3-Tier API Architecture

| Tier | Name | Description | Use Case |
|------|------|-------------|----------|
| 1 | **Registry** | Centralized block discovery | `registry.get('residual')` |
| 2 | **Recipes** | Declarative string definitions | Config-driven experiments |
| 3 | **Composer** | Fluent `>>` operator API | Programmatic construction |

### 🔀 v2.1: Hybrid Models (NEW!)

Customize **any torchvision pre-trained model** while preserving weights:

```python
from torchvision_customizer import HybridBuilder

builder = HybridBuilder()

# Supported: resnet18-152, efficientnet_b0-b7, convnext, mobilenet_v2/v3, vgg, densenet, vit, swin...
model = builder.from_torchvision(
    "efficientnet_b4",
    weights="IMAGENET1K_V1",
    patches={"features.5": {"wrap": "eca"}},  # Add ECA attention
    num_classes=200,
)
```

### 🧱 40+ Building Blocks

| Category | Blocks |
|----------|--------|
| **Convolution** | `ConvBlock`, `DepthwiseBlock`, `MBConv`, `FusedMBConv`, `CoordConv` |
| **Residual** | `ResidualBlock`, `Bottleneck`, `WideBottleneck`, `GroupedBottleneck` |
| **Attention** | `SEBlock`, `CBAMBlock`, `ECABlock`, `ChannelAttention`, `SpatialAttention` |
| **Regularization** | `DropPath`, `Dropout`, `DropBlock` |
| **Activation** | `Mish`, `Swish`, `GELU`, and all PyTorch activations |
| **Pooling** | `GeM`, `AdaptiveAvgPool`, `MaxPool` |
| **Architecture** | `InceptionModule`, `DenseBlock` |

### 📝 Enhanced YAML Recipes (v2.1)

```yaml
# Macros for reuse
macros:
  attention: se
  dropout: 0.3

# Inherit from templates
extends: resnet_base

# Or use hybrid backbone
backbone:
  name: resnet50
  weights: IMAGENET1K_V2
  patches:
    layer3: {wrap: "@attention"}  # Macro expansion!
```

### ⚡ Weight Utilities (v2.1)

```python
from torchvision_customizer import partial_load, transfer_weights

# Load with mismatch tolerance
report = partial_load(model, checkpoint, ignore_mismatch=True)
print(f"Loaded {report.load_ratio:.1%} of weights")

# Transfer specific layers
transfer_weights(source, target, exclude_patterns=['fc', 'classifier'])
```

### 🖥️ CLI Tools (v2.1)

```bash
tvc build --yaml model.yaml              # Build model
tvc benchmark --yaml model.yaml          # Benchmark speed
tvc validate --yaml model.yaml           # Validate config
tvc export --yaml model.yaml --format onnx  # Export
tvc list-backbones                       # List 40+ backbones
tvc list-blocks                          # List all blocks
tvc create-recipe --template hybrid_resnet_se  # Create from template
```

---

## 🔀 v2.1: Hybrid Models

The star feature of v2.1 - **customize pre-trained torchvision models**:

### Supported Backbones (40+)

| Family | Models |
|--------|--------|
| **ResNet** | resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet50_2, resnext50_32x4d |
| **EfficientNet** | efficientnet_b0-b7, efficientnet_v2_s/m/l |
| **ConvNeXt** | convnext_tiny, convnext_small, convnext_base, convnext_large |
| **MobileNet** | mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large |
| **VGG** | vgg11, vgg13, vgg16, vgg19 (with/without BN) |
| **DenseNet** | densenet121, densenet169, densenet201, densenet161 |
| **Vision Transformer** | vit_b_16, vit_b_32, vit_l_16, vit_l_32 |
| **Swin Transformer** | swin_t, swin_s, swin_b |

### Patch Operations

```python
patches = {
    "layer3": {"wrap": "se"},           # Wrap with SE attention
    "layer4": {"wrap": "cbam_block"},   # Wrap with CBAM
    "layer2": {"inject": "eca"},        # Inject after layer
    "layer1": {"replace": "custom"},    # Replace entirely
}
```

### Fine-tuning Workflow

```python
from torchvision_customizer import HybridBuilder

# 1. Create hybrid model
model = HybridBuilder().from_torchvision(
    "resnet50",
    weights="IMAGENET1K_V2",
    patches={"layer4": {"wrap": "cbam_block"}},
    num_classes=10,
)

# 2. Freeze backbone (keep last stage trainable)
model.freeze_backbone(unfreeze_stages=[3])

# 3. Train only ~2M params instead of 25M
print(f"Training {model.count_parameters(trainable_only=True):,} parameters")

# 4. Later, unfreeze for full fine-tuning
model.unfreeze_all()
```

---

## 📊 Comparison

| Feature | torchvision | timm | torchvision-customizer |
|---------|-------------|------|------------------------|
| Pre-trained models | ✅ | ✅ | ✅ (via hybrid) |
| Custom architectures | ❌ | ⚠️ | ✅ |
| Block-level customization | ❌ | ⚠️ | ✅ |
| Inject attention | ❌ | ❌ | ✅ |
| YAML recipes | ❌ | ❌ | ✅ |
| Fluent `>>` API | ❌ | ❌ | ✅ |
| Weight utilities | ⚠️ | ⚠️ | ✅ |
| CLI tools | ❌ | ❌ | ✅ |

---

## 📚 Documentation

| Resource | Link |
|----------|------|
| 📖 Full Documentation | [torchvision-customizer.readthedocs.io](https://torchvision-customizer.readthedocs.io/) |
| 🎓 Tutorials | [docs/examples/](docs/examples/) |
| 📋 API Reference | [docs/api/](docs/api/) |
| 📝 Release Notes | [RELEASE_NOTES_v2.1.md](RELEASE_NOTES_v2.1.md) |

---

## 🛠️ Examples

### Custom ResNet with SE Attention

```python
from torchvision_customizer import Stem, Stage, Head

model = (
    Stem(64, kernel=7, stride=2)
    >> Stage(64, blocks=2, pattern='residual+se')
    >> Stage(128, blocks=2, pattern='residual+se', downsample=True)
    >> Stage(256, blocks=2, pattern='residual+se', downsample=True)
    >> Stage(512, blocks=2, pattern='residual+se', downsample=True)
    >> Head(num_classes=100)
)
```

### EfficientNet-style MBConv Network

```python
from torchvision_customizer import Stem, Stage, Head

model = (
    Stem(32, kernel=3, stride=2)
    >> Stage(16, blocks=1, pattern='mbconv')
    >> Stage(24, blocks=2, pattern='mbconv', downsample=True)
    >> Stage(40, blocks=2, pattern='mbconv', downsample=True)
    >> Stage(80, blocks=3, pattern='mbconv', downsample=True)
    >> Head(num_classes=1000, dropout=0.2)
)
```

### Fine-tune ConvNeXt for Medical Imaging

```python
from torchvision_customizer import HybridBuilder

model = HybridBuilder().from_torchvision(
    "convnext_base",
    weights="IMAGENET1K_V1",
    num_classes=5,  # 5 disease classes
    dropout=0.4,
    freeze_backbone=True,
    unfreeze_stages=[2, 3],  # Train last 2 stages
)
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/codewithdark-git/torchvision-customizer.git
cd torchvision-customizer
pip install -e ".[dev]"
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 📄 Citation

If you use torchvision-customizer in your research, please cite:

```bibtex
@software{torchvision_customizer_2025,
  title={torchvision-customizer: Flexible CNN Architecture Builder},
  author={Ahsan Umar},
  year={2025},
  version={2.1.0},
  url={https://github.com/codewithdark-git/torchvision-customizer}
}
```

---

## 🙏 Acknowledgments

Built with 💙 for the computer vision and deep learning research community.

Special thanks to the PyTorch and torchvision teams for their foundational work.

---

<p align="center">
  <a href="https://github.com/codewithdark-git/torchvision-customizer">⭐ Star us on GitHub</a> •
  <a href="https://github.com/codewithdark-git/torchvision-customizer/issues">🐛 Report Bug</a> •
  <a href="https://github.com/codewithdark-git/torchvision-customizer/discussions">💬 Discussions</a>
</p>
