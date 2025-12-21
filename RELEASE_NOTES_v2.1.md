# Release Notes v2.1.0

## 🚀 Major Release: Hybrid Models & Pre-trained Customization

Version 2.1.0 introduces **Hybrid Models** - the ability to load torchvision pre-trained models and customize them with your own blocks, attention mechanisms, and architectural modifications. This release also adds new building blocks, enhanced YAML recipes, and a CLI for rapid prototyping.

---

### ✨ Key Features

#### 1. Hybrid Builder 🔧
Load any torchvision pre-trained model and customize it while preserving maximum weights.

```python
from torchvision_customizer import HybridBuilder

builder = HybridBuilder()
model = builder.from_torchvision(
    "resnet50",
    weights="IMAGENET1K_V2",
    patches={
        "layer3": {"wrap": "se"},           # Add SE attention
        "layer4": {"wrap": "cbam_block"},   # Add CBAM
    },
    num_classes=100,
    dropout=0.3,
)

# Freeze backbone for fine-tuning
model.freeze_backbone(unfreeze_stages=[3])  # Keep last stage trainable
```

**Supported Backbones:**
- ResNet family (18, 34, 50, 101, 152)
- EfficientNet (B0-B7, V2)
- ConvNeXt (Tiny, Small, Base, Large)
- MobileNet (V2, V3)
- VGG (11, 13, 16, 19)
- DenseNet (121, 169, 201)
- Vision Transformer (ViT)
- Swin Transformer

#### 2. Weight Utilities 📦
Smart weight loading with mismatch tolerance and detailed reports.

```python
from torchvision_customizer import partial_load, transfer_weights

# Load weights with mismatch tolerance
report = partial_load(model, state_dict, ignore_mismatch=True)
print(report.summary())
# Output:
# Loaded:           95% of parameters
# Shape Mismatch:   12 parameters
# Newly Initialized: 5 parameters

# Transfer weights between models
transfer_weights(pretrained, custom, exclude_patterns=['fc', 'classifier'])
```

#### 3. New Building Blocks 🧱

| Block | Description | Use Case |
|-------|-------------|----------|
| `CBAMBlock` | Convolutional Block Attention Module | Feature recalibration |
| `ECABlock` | Efficient Channel Attention | Lightweight attention |
| `DropPath` | Stochastic Depth regularization | Training deeper networks |
| `Mish` | Self-regularizing activation | Smoother gradients |
| `GeM` | Generalized Mean Pooling | Image retrieval |
| `MBConv` | Mobile Inverted Bottleneck | EfficientNet-style blocks |
| `FusedMBConv` | Fused MBConv (EfficientNetV2) | Faster training |
| `LayerScale` | Per-channel scaling | Vision Transformers |

```python
from torchvision_customizer import CBAMBlock, ECABlock, DropPath

# Apply CBAM attention
cbam = CBAMBlock(channels=256, reduction=16)

# Efficient channel attention
eca = ECABlock(channels=512)

# Stochastic depth for residual
drop_path = DropPath(drop_prob=0.2)
out = x + drop_path(residual)
```

#### 4. Enhanced YAML Recipes 📝
JSONSchema validation, inheritance, and macro expansion.

```yaml
# my_model.yaml
name: ResNet50-SE-Custom
version: "1.0.0"

# Macros for reuse
macros:
  attention: se
  dropout: 0.3

# Use pretrained backbone
backbone:
  name: resnet50
  weights: IMAGENET1K_V2
  patches:
    layer3:
      wrap:
        type: "@attention"
        params:
          reduction: 16
    layer4:
      wrap: cbam_block

# Custom head
head:
  num_classes: 100
  dropout: "@dropout"
```

**Recipe Inheritance:**
```yaml
# my_custom.yaml
extends: resnet_base  # Built-in template or file path
stages:
  - pattern: residual
    channels: 128
    blocks: 4
    attention: se  # Add attention to stages
```

#### 5. CLI Prototyper 🖥️
Command-line tools for rapid model development.

```bash
# Build model from recipe
tvc build --yaml my_model.yaml --output model.pt

# Benchmark performance
tvc benchmark --yaml my_model.yaml --device cuda --batch-size 32

# Validate recipe
tvc validate --yaml my_model.yaml --strict

# Export to ONNX
tvc export --yaml my_model.yaml --format onnx --output model.onnx

# List available backbones
tvc list-backbones

# List available blocks
tvc list-blocks --category attention

# Create recipe from template
tvc create-recipe --template hybrid_resnet_se --output custom.yaml
```

---

### 🆕 New Modules

| Module | Purpose |
|--------|---------|
| `torchvision_customizer.hybrid` | Hybrid model building |
| `torchvision_customizer.hybrid.weight_utils` | Weight loading utilities |
| `torchvision_customizer.hybrid.extractor` | Backbone structure extraction |
| `torchvision_customizer.recipe.schema` | JSONSchema validation |
| `torchvision_customizer.recipe.yaml_loader` | Enhanced YAML loading |
| `torchvision_customizer.cli` | Command-line interface |

---

### 🛠️ API Additions

```python
# New top-level imports
from torchvision_customizer import (
    # Hybrid
    HybridBuilder,
    partial_load,
    transfer_weights,
    extract_tiers,
    get_backbone_info,
    
    # New blocks
    CBAMBlock,
    ECABlock,
    DropPath,
    Mish,
    GeM,
    MBConv,
    FusedMBConv,
)

# Recipe enhancements
from torchvision_customizer.recipe import (
    load_yaml_recipe,
    load_yaml_config,
    save_yaml_recipe,
    validate_recipe_config,
    expand_macros,
    list_templates,
    create_recipe_from_template,
)
```

---

### 📊 Quick Example: Fine-tune ResNet50 with Attention

```python
from torchvision_customizer import HybridBuilder

# Create hybrid model
builder = HybridBuilder()
model = builder.from_torchvision(
    "resnet50",
    weights="IMAGENET1K_V2",
    patches={
        "layer3": {"wrap": "se"},
        "layer4": {"wrap": "cbam_block"},
    },
    num_classes=10,
    freeze_backbone=True,
    unfreeze_stages=[3],  # Only train last stage + head
)

# Print model info
print(model.explain())

# Training loop
for images, labels in dataloader:
    outputs = model(images)
    loss = criterion(outputs, labels)
    # ...
```

---

### 📦 Dependencies

New optional dependencies for v2.1:
- `pyyaml>=6.0` (for YAML recipes)
- `jsonschema>=4.0` (optional, for strict validation)

---

### 🔄 Migration from v2.0

v2.1 is **fully backward compatible** with v2.0. All existing code will continue to work:

```python
# v2.0 code still works
from torchvision_customizer import Stem, Stage, Head
model = Stem(64) >> Stage(128, blocks=3) >> Head(10)

# v2.0 templates still work  
from torchvision_customizer import resnet
model = resnet(layers=50, num_classes=1000)

# v2.0 recipes still work
from torchvision_customizer import Recipe, build_recipe
recipe = Recipe(stem="conv(64)", stages=["residual(128) x 3"], head="linear(10)")
model = build_recipe(recipe)
```

---

### 🔗 Contributors
- @codewithdark-git (Architecture & Implementation)

---

### 📈 What's Next (v2.2 Roadmap)

- [ ] ONNX optimization and quantization
- [ ] AutoML architecture search integration
- [ ] Multi-GPU training utilities
- [ ] Pre-built fine-tuning pipelines
- [ ] Hugging Face Hub integration

---

### 📚 Full Changelog

**Added:**
- `HybridBuilder` for pre-trained model customization
- Weight utilities (`partial_load`, `transfer_weights`)
- 12 new building blocks (CBAM, ECA, DropPath, Mish, GeM, MBConv, etc.)
- YAML recipe schema validation
- Recipe inheritance and macros
- CLI (`tvc` command)
- Backbone extraction utilities

**Improved:**
- Registry now includes 30+ blocks
- Better error messages for validation failures
- Documentation updates

**Fixed:**
- Hybrid Builder: Smart parameter detection for wrapped blocks (channels vs in_channels)
- Stage patterns: Added `mbconv` and `fused_mbconv` patterns for EfficientNet-style blocks
- Recipe parser: Added parameter shortcuts (`k` → `kernel_size`, `s` → `stride`, `p` → `padding`, etc.)
- Minor type hint corrections
- Windows compatibility improvements

