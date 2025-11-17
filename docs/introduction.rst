==========================
Introduction
==========================

What is torchvision-customizer?
===============================

**torchvision-customizer** is a comprehensive Python library for building, customizing, and deploying 
convolutional neural networks (CNNs) with PyTorch. It provides a collection of modular components, utilities, 
and architectural patterns that enable researchers and practitioners to:

- Create flexible CNN architectures with fine-grained control
- Quickly prototype models with intuitive APIs
- Share model configurations via YAML/JSON files
- Analyze and optimize neural network architectures
- Deploy models efficiently in production environments

Key Design Principles
=====================

Our library is built on three core principles:

1. **Modularity** - Every component (layers, blocks, models) is independently testable and reusable
2. **Flexibility** - Multiple APIs provide different levels of abstraction for different use cases
3. **Clarity** - Type hints, docstrings, and examples make the codebase accessible to everyone

Architecture Overview
=====================

The library is organized into four main modules:

.. code-block:: text

    torchvision_customizer/
    ├── blocks/          # Composite building blocks (Conv, Residual, Inception, etc.)
    ├── layers/          # Low-level components (Activations, Pooling, Attention, etc.)
    ├── models/          # High-level model classes (CustomCNN, etc.)
    └── utils/           # Utilities (Architecture search, Model summary, Validation)

Each module is designed to work both independently and together, giving you maximum flexibility.

Core Components
===============

**Blocks** - Composite Modules
------------------------------

Blocks combine multiple layers into reusable units:

- **ConvBlock**: Convolution with normalization and activation
- **ResidualBlock**: Skip connection for gradient flow
- **BottleneckBlock**: Multi-scale feature extraction
- **DepthwiseBlock**: Efficient depth-wise convolution
- **InceptionModule**: Multi-branch architecture
- **SEBlock**: Squeeze-and-Excitation attention
- **Super Layers**: Advanced convolution and linear layers

**Layers** - Atomic Components
------------------------------

Low-level building blocks for maximum control:

- **Activations**: ReLU, LeakyReLU, GELU, Mish, Swish, SiLU, etc.
- **Normalization**: BatchNorm, LayerNorm, GroupNorm, InstanceNorm
- **Pooling**: MaxPool, AvgPool, Adaptive, Stochastic, LP-norm
- **Attention**: Channel attention, Spatial attention, Multi-head attention

**Models** - High-Level Interfaces
----------------------------------

Ready-to-use model classes:

- **CustomCNN**: Simple sequential models
- **ResidualArchitecture**: ResNet-style networks
- **DenseArchitecture**: DenseNet-style networks
- **AdvancedArchitecture**: Custom multi-branch networks

**Utilities** - Analysis & Optimization
----------------------------------------

Tools for model analysis and optimization:

- **Model Summary**: Parameter counts, FLOPs, memory usage
- **Architecture Search**: Find optimal configurations
- **Validators**: Configuration validation and error checking

Use Cases
=========

Research & Development
~~~~~~~~~~~~~~~~~~~~~~

Perfect for researchers exploring new architectures:

.. code-block:: python

    # Easy experimentation with different configurations
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=4,
        channels=[64, 128, 256, 512],
        activation='gelu'
    )

Transfer Learning
~~~~~~~~~~~~~~~~~

Build on pre-trained models:

.. code-block:: python

    # Load pre-trained backbone
    backbone = ResNet50()
    
    # Add custom head
    model = CustomCNN(
        backbone=backbone,
        num_classes=10,
        dropout=0.3
    )

Production Deployment
~~~~~~~~~~~~~~~~~~~~~

Reproducible model configurations:

.. code-block:: yaml

    # config.yaml
    input_shape: [3, 224, 224]
    num_classes: 1000
    blocks:
      - type: conv
        channels: 64
        kernel_size: 7
      - type: residual
        channels: 128
        num_layers: 2

Performance & Scalability
~~~~~~~~~~~~~~~~~~~~~~~~~~

Efficient implementations for various scales:

.. code-block:: python

    # Mobile-friendly model
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=10,
        num_conv_blocks=3,
        channels=[32, 64, 128],
        depthwise=True,  # Lightweight
        dropout=0.2
    )

Comparison with Alternatives
=============================

.. csv-table::
   :header: "Feature", "torchvision-customizer", "timm", "torchvision", "PyTorch"
   :widths: 30, 20, 20, 20, 20

   "Modularity", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐⭐"
   "Pre-built Models", "⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐"
   "Configuration-Based", "⭐⭐⭐⭐⭐", "⭐⭐", "⭐", "⭐"
   "Custom Blocks", "⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐", "⭐⭐⭐⭐⭐"
   "Model Analysis", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐", "⭐⭐"
   "Documentation", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐"

Project Statistics
==================

.. rst-class:: statistics

+----------------------------+----------------------------+
| Metric                     | Value                      |
+============================+============================+
| **Tests**                  | 382 (100% passing)         |
+----------------------------+----------------------------+
| **Examples**               | 113 working examples       |
+----------------------------+----------------------------+
| **Code Size**              | 7,750+ lines               |
+----------------------------+----------------------------+
| **Components**             | 90+ classes/functions      |
+----------------------------+----------------------------+
| **Modules**                | 16 (blocks, layers, etc.)  |
+----------------------------+----------------------------+
| **Type Hints**             | 100% coverage              |
+----------------------------+----------------------------+
| **Documentation**          | Comprehensive              |
+----------------------------+----------------------------+
| **License**                | MIT (Open Source)          |
+----------------------------+----------------------------+

What's Next?
============

Ready to get started? 

- Head to :doc:`Installation <installation>` to set up the library
- Check out :doc:`Quick Start <quick_start>` for your first model
- Explore :doc:`Examples <examples/basic_usage>` for common patterns
- Dive into :doc:`API Reference <api/blocks>` for detailed documentation

Questions or Issues?
====================

- 📚 **Documentation**: Browse this comprehensive guide
- 💡 **Examples**: Check out 113 working examples
- 🐛 **Issues**: Report bugs on `GitHub <https://github.com/codewithdark-git/torchvision-customizer/issues>`_
- 💬 **Discussions**: Join our community discussions

Citation
========

If you use torchvision-customizer in your research, please cite:

.. code-block:: bibtex

    @software{torchvision_customizer_2025,
        title={torchvision-customizer: Flexible CNN Architecture Builder},
        author={Umar, Ahsan},
        year={2025},
        doi={10.5281/zenodo.17633293},
        url={https://github.com/codewithdark-git/torchvision-customizer}
    }

Citation Formats
----------------

**APA Format:**

Umar, A. (2025). torchvision-customizer: Flexible CNN Architecture Builder [Software]. 
Retrieved from https://github.com/codewithdark-git/torchvision-customizer

**Chicago Format:**

Umar, Ahsan. 2025. "torchvision-customizer: Flexible CNN Architecture Builder." 
Accessed November 17, 2025. https://github.com/codewithdark-git/torchvision-customizer.

**Harvard Format:**

Umar, A., 2025. torchvision-customizer: Flexible CNN Architecture Builder. 
Available at: https://github.com/codewithdark-git/torchvision-customizer.

**MLA Format:**

Umar, Ahsan. "torchvision-customizer: Flexible CNN Architecture Builder." 
GitHub, 2025, https://github.com/codewithdark-git/torchvision-customizer.

Share Your Research
-------------------

If you've published research using torchvision-customizer, we'd love to hear about it! Please:

1. Add your paper to our `Research Papers Wiki <https://github.com/codewithdark-git/torchvision-customizer/wiki/Research-Papers>`_
2. Open an issue on GitHub with the tag ``[publication]``
3. Share in `GitHub Discussions <https://github.com/codewithdark-git/torchvision-customizer/discussions>`_

Tracking Citation Metrics
--------------------------

Monitor the project's academic impact:

- **GitHub Repository**: https://github.com/codewithdark-git/torchvision-customizer
- **Documentation**: https://torchvision-customizer.readthedocs.io
- **PyPI Package**: https://pypi.org/project/torchvision-customizer
- **GitHub Stars**: Show your support on our `Stargazers page <https://github.com/codewithdark-git/torchvision-customizer/stargazers>`_
