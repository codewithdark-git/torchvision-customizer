# Changelog

All notable changes to torchvision-customizer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.1.1] Pre-Release - 2025-12-21

### Added
- **Training API** - Built-in training utilities for quick experimentation
  - `Trainer` class with `fit()`, `fit_mnist()`, `fit_cifar10()`, `fit_cifar100()` methods
  - `quick_train()` function for one-liner training
  - `TrainingMetrics` class for tracking training history with summaries
  - Support for Adam, AdamW, SGD optimizers
  - Cosine, Step, OneCycle learning rate schedulers
  - Automatic device selection (CPU/CUDA)
- Training example scripts: `train_mnist.py`, `train_cifar10.py`, `quick_train_example.py`
- Tests for trainer module (`test_trainer.py`)

### Fixed
- **Hybrid Builder**: Smart parameter detection for wrapped blocks (channels vs in_channels)
- **Stage patterns**: Added `mbconv` and `fused_mbconv` patterns for EfficientNet-style blocks
- **Recipe parser**: Added parameter shortcuts (`k` → `kernel_size`, `s` → `stride`, `p` → `padding`, etc.)
- **Stem class**: Added `kernel_size` alias for `kernel` parameter

---

## [2.1.0] - 2025-12-20

### Added
- **Hybrid Builder** - Load and customize pre-trained torchvision models
  - `HybridBuilder.from_torchvision()` for loading any torchvision model
  - Patch operations: `wrap`, `inject`, `replace` for modifying layers
  - `freeze_backbone()` and `unfreeze_all()` for fine-tuning control
  - Support for 40+ backbone architectures (ResNet, EfficientNet, ConvNeXt, MobileNet, VGG, DenseNet, ViT, Swin)

- **Weight Utilities**
  - `partial_load()` for loading weights with mismatch tolerance
  - `transfer_weights()` for transferring weights between models
  - `match_state_dict()` for finding compatible parameters
  - `WeightLoadingReport` for detailed loading statistics

- **New Building Blocks** (12 new blocks)
  - `CBAMBlock` - Convolutional Block Attention Module
  - `ECABlock` - Efficient Channel Attention
  - `DropPath` - Stochastic Depth regularization
  - `Mish` - Self-regularizing activation function
  - `GeM` - Generalized Mean Pooling
  - `MBConv` - Mobile Inverted Bottleneck
  - `FusedMBConv` - Fused MBConv (EfficientNetV2)
  - `LayerScale` - Per-channel scaling for ViT
  - `CoordConv` - Coordinate Convolution
  - `ConvBNAct` - Conv + BatchNorm + Activation combo
  - `SqueezeExcitation` - Enhanced SE block
  - `GELUActivation` - GELU with approximate mode

- **Enhanced YAML Recipes**
  - JSONSchema validation with `validate_recipe_config()`
  - Recipe inheritance with `extends` keyword
  - Macro expansion with `@macro_name` syntax
  - Template system with `list_templates()` and `create_recipe_from_template()`

- **CLI Prototyper** (`tvc` command)
  - `tvc build` - Build model from YAML recipe
  - `tvc benchmark` - Benchmark model performance
  - `tvc validate` - Validate recipe configuration
  - `tvc export` - Export to ONNX/TorchScript
  - `tvc list-backbones` - List available backbones
  - `tvc list-blocks` - List available blocks
  - `tvc create-recipe` - Create recipe from template

- **New Modules**
  - `torchvision_customizer.hybrid` - Hybrid model building
  - `torchvision_customizer.hybrid.weight_utils` - Weight utilities
  - `torchvision_customizer.hybrid.extractor` - Backbone extraction
  - `torchvision_customizer.recipe.schema` - JSONSchema validation
  - `torchvision_customizer.recipe.yaml_loader` - Enhanced YAML loading
  - `torchvision_customizer.cli` - Command-line interface

### Improved
- Registry now includes 40+ registered blocks
- Better error messages for validation failures
- Comprehensive documentation updates

---

## [2.0.0] - 2025-12-07

### Added
- **3-Tier API Architecture**
  - Tier 1: Component Registry for block discovery
  - Tier 2: Recipe system for declarative model definitions
  - Tier 3: Composer API with `>>` operator for fluent model building

- **Compose System**
  - `Stem` - Network entry point builder
  - `Stage` - Network body builder with pattern support
  - `Head` - Classification head builder
  - `VisionModel` - Complete model wrapper

- **Recipe System**
  - `Recipe` class for declarative definitions
  - `build_recipe()` for converting recipes to models
  - String-based component definitions

- **Templates** (Parametric Architectures)
  - `resnet()` - ResNet family (18, 34, 50, 101, 152)
  - `vgg()` - VGG family (11, 13, 16, 19)
  - `mobilenet()` - MobileNet architectures
  - `densenet()` - DenseNet family
  - `efficientnet()` - EfficientNet family

- **Building Blocks**
  - `ConvBlock` - Standard convolution block
  - `ResidualBlock` - Residual connection block
  - `SEBlock` - Squeeze-and-Excitation
  - `DepthwiseBlock` - Depthwise separable convolution
  - `InceptionModule` - Inception-style block
  - `DenseConnectionBlock` - Dense connection block
  - `StandardBottleneck`, `WideBottleneck`, `GroupedBottleneck`

- **Utilities**
  - `print_model_summary()` - Model architecture summary
  - `validate_architecture()` - Architecture validation
  - `get_model_flops()` - FLOP computation

### Changed
- Complete architecture redesign from v1.x
- New modular design philosophy

---

## [1.0.0] - 2025-11-17

### Added
- Initial release
- Basic CNN building blocks
- Simple model construction API
- PyTorch integration

---

[2.1.1]: https://github.com/codewithdark-git/torchvision-customizer/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/codewithdark-git/torchvision-customizer/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/codewithdark-git/torchvision-customizer/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/codewithdark-git/torchvision-customizer/releases/tag/v1.0.0

