.. torchvision-customizer documentation master file

========================================
torchvision-customizer Documentation
========================================

**Version 2.1.0** | `GitHub <https://github.com/codewithdark-git/torchvision-customizer>`_ | `PyPI <https://pypi.org/project/torchvision-customizer/>`_

**Build, customize, and fine-tune CNN architectures with unprecedented flexibility.**

A production-ready Python package that empowers researchers and developers to create flexible, modular CNNs with fine-grained control over every architectural decision.

.. note::
   **v2.1.0 Release** - Now with Hybrid Models! Customize pre-trained torchvision models with attention injection, block replacement, and smart weight loading.

What's New in v2.1
==================

🔀 **Hybrid Models**
   Customize any torchvision pre-trained model (ResNet, EfficientNet, ConvNeXt, etc.) while preserving weights.

🧱 **12 New Blocks**
   CBAM, ECA, DropPath, Mish, GeM, MBConv, and more.

📝 **Enhanced Recipes**
   YAML recipes with macros, inheritance, and validation.

🖥️ **CLI Tools**
   Build, benchmark, and export models from the command line.

Quick Example
=============

**Customize a Pre-trained Model (v2.1)**

.. code-block:: python

   from torchvision_customizer import HybridBuilder

   model = HybridBuilder().from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       patches={
           "layer3": {"wrap": "se"},
           "layer4": {"wrap": "cbam_block"},
       },
       num_classes=100,
       freeze_backbone=True,
   )

**Build from Scratch**

.. code-block:: python

   from torchvision_customizer import Stem, Stage, Head

   model = (
       Stem(64)
       >> Stage(64, blocks=2, pattern='residual')
       >> Stage(128, blocks=2, pattern='residual+se', downsample=True)
       >> Head(num_classes=10)
   )

**Use the CLI**

.. code-block:: bash

   tvc build --yaml model.yaml --output model.pt
   tvc benchmark --yaml model.yaml --device cuda
   tvc list-backbones

Key Concepts
============

**3-Tier API Architecture**

1. **Component Registry (Tier 1)**: Centralized management of all building blocks.
2. **Architecture Recipes (Tier 2)**: Declarative, string-based model definitions.
3. **Model Composer (Tier 3)**: Fluent, operator-based API for programmatic construction.
4. **Hybrid Builder (v2.1)**: Pre-trained model customization with patches.
5. **Templates**: Parametric implementations of standard architectures (ResNet, VGG, etc.).

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   introduction
   installation
   quick_start

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user_guide/basics
   user_guide/hybrid
   user_guide/blocks
   user_guide/layers
   user_guide/templates
   user_guide/advanced

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/hybrid
   api/blocks
   api/recipe
   api/compose
   api/templates
   api/registry
   api/cli
   api/layers
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/basic_usage
   examples/hybrid_models
   examples/custom_architectures
   examples/transfer_learning
   examples/advanced_patterns

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   development/contributing
   development/changelog
   development/testing
   development/performance

Features
========

* **Hybrid Models**: Customize pre-trained torchvision models with attention injection and block replacement.
* **40+ Building Blocks**: SE, CBAM, ECA, MBConv, Inception, DenseBlock, and more.
* **Granular Control**: Customize every aspect of the architecture (depth, width, attention, activation).
* **Pattern Mixing**: Mix different block types in the same stage (e.g., ``residual+se``).
* **YAML Recipes**: Declarative model definitions with macros and inheritance.
* **CLI Tools**: Build, benchmark, validate, and export from the command line.
* **Weight Utilities**: Smart partial loading and weight transfer.
* **Introspection**: Built-in ``model.explain()`` for human-readable summaries.

Installation
============

.. code-block:: bash

   pip install torchvision-customizer

For development:

.. code-block:: bash

   pip install git+https://github.com/codewithdark-git/torchvision-customizer.git

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
