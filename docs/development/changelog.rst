===========
Changelog
===========

All notable changes to this project are documented here.

Version 2.1.0 (2025)
====================

**🚀 Major Release: Hybrid Models & Pre-trained Customization**

**New Features:**

- ✨ **Hybrid Builder**: Customize pre-trained torchvision models with patches
- ✨ **Weight Utilities**: Smart partial loading and weight transfer
- ✨ **12 New Blocks**: CBAM, ECA, DropPath, Mish, GeM, MBConv, and more
- ✨ **Enhanced YAML Recipes**: Macros, inheritance, and JSONSchema validation
- ✨ **CLI Tools**: Build, benchmark, validate, and export from command line
- ✨ **40+ Supported Backbones**: ResNet, EfficientNet, ConvNeXt, MobileNet, ViT, Swin

**New Modules:**

- ``torchvision_customizer.hybrid`` - Hybrid model building
- ``torchvision_customizer.hybrid.weight_utils`` - Weight loading utilities
- ``torchvision_customizer.hybrid.extractor`` - Backbone structure extraction
- ``torchvision_customizer.recipe.schema`` - JSONSchema validation
- ``torchvision_customizer.recipe.yaml_loader`` - Enhanced YAML loading
- ``torchvision_customizer.cli`` - Command-line interface

**New Blocks:**

- CBAMBlock - Convolutional Block Attention Module
- ECABlock - Efficient Channel Attention
- DropPath - Stochastic Depth regularization
- Mish - Self-regularizing activation
- GeM - Generalized Mean Pooling
- MBConv - Mobile Inverted Bottleneck
- FusedMBConv - Fused MBConv (EfficientNetV2)
- LayerScale - Per-channel scaling
- CoordConv - Coordinate Convolution
- SqueezeExcitation - Enhanced SE block
- ConvBNAct - Conv + BN + Activation combo
- GELUActivation - GELU with approximate mode

**CLI Commands:**

.. code-block:: bash

   tvc build --yaml model.yaml
   tvc benchmark --yaml model.yaml
   tvc validate --yaml model.yaml
   tvc export --yaml model.yaml --format onnx
   tvc list-backbones
   tvc list-blocks
   tvc create-recipe --template hybrid_resnet_se

**Improvements:**

- 📚 Complete documentation rewrite
- 🎨 Better README with visual design
- ✅ Comprehensive test suite for hybrid module
- 🔧 Registry now includes 30+ blocks

Version 2.0.0 (2025)
====================

**🔄 Major Release: 3-Tier API Architecture**

Complete paradigm shift from monolithic ``CustomCNN`` to component-based architecture.

**Breaking Changes:**

- ❌ Removed ``CustomCNN`` class
- ❌ Removed ``torchvision_customizer.models`` module
- ⚠️ Minimum Python version: 3.9 (previously 3.13)

**New Features:**

- ✨ **Tier 1: Component Registry** - Centralized block discovery
- ✨ **Tier 2: Architecture Recipes** - Declarative model definitions
- ✨ **Tier 3: Model Composer** - Fluent API with ``>>`` operators
- ✨ **Parametric Templates** - ResNet, VGG, MobileNet, DenseNet, EfficientNet

**API Examples:**

.. code-block:: python

   # Registry (Tier 1)
   block = registry.get('residual')(64, 64)
   
   # Recipes (Tier 2)
   recipe = Recipe(stem="conv(64)", stages=["residual(64) x 2"], head="linear(10)")
   model = build_recipe(recipe)
   
   # Composer (Tier 3)
   model = Stem(64) >> Stage(128, blocks=2) >> Head(10)

Version 1.0.0 (2025-01-17)
============================

**Initial Release**

**Features:**

- ✨ 90+ pre-built components (blocks, layers, models)
- ✨ 16 modules for comprehensive CNN building
- ✨ Multiple APIs (Simple, Builder, Advanced)
- ✨ Configuration-based model definition (YAML/JSON)
- ✨ Model analysis utilities (FLOPs, memory, summary)
- ✨ 8 complete development steps with full documentation
- ✨ 113 working examples
- ✨ 100% type hints coverage
- ✨ Comprehensive API documentation

**Testing:**

- ✅ 382 tests with 100% pass rate
- ✅ 100% code coverage
- ✅ Cross-platform compatibility
- ✅ GPU and CPU support

Roadmap
=======

**Version 2.2.0 (Planned)**

- ONNX optimization and INT8 quantization
- AutoML architecture search integration
- Multi-GPU training utilities (DDP wrappers)
- Pre-built fine-tuning pipelines
- Hugging Face Hub integration

**Version 2.3.0 (Planned)**

- Knowledge distillation utilities
- Pruning and model compression
- TensorRT export support
- Mobile deployment tools

**Version 3.0.0 (Future)**

- 3D networks for video/medical imaging
- Graph neural networks
- Multi-modal architectures
- Advanced quantization-aware training

Release Timeline
================

- **v1.0.0**: 2025-01-17 ✅
- **v2.0.0**: 2025 ✅
- **v2.1.0**: 2025 ✅ (Current)
- **v2.2.0**: Q2 2025
- **v2.3.0**: Q3 2025
- **v3.0.0**: Q4 2025+

Breaking Changes Policy
=======================

- All breaking changes happen in major versions (v2.0.0, v3.0.0, etc.)
- Deprecations are announced 1 major version in advance
- Minor versions (v2.1.0, v2.2.0, etc.) are backward compatible
- Patch versions (v2.1.1, v2.1.2, etc.) fix bugs without API changes

Migration Guides
================

**v1.x → v2.0**

.. code-block:: python

   # Old (v1.x)
   model = CustomCNN(layers=50, architecture='resnet')
   
   # New (v2.0+)
   model = resnet(layers=50, num_classes=1000)
   # or
   model = Stem(64) >> Stage(128, blocks=2) >> Head(1000)

**v2.0 → v2.1**

v2.1 is fully backward compatible with v2.0. All v2.0 code works unchanged.

New v2.1 features are additive:

.. code-block:: python

   # v2.0 code still works
   model = Stem(64) >> Stage(128, blocks=2) >> Head(10)
   
   # v2.1 adds hybrid models
   model = HybridBuilder().from_torchvision("resnet50", ...)

Contributing
============

Contributions welcome! See :doc:`contributing` for guidelines.

License
=======

MIT License - See LICENSE file for details.
