.. torchvision-customizer documentation master file, created by
   sphinx-quickstart on 2025-01-01.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========================================
torchvision-customizer Documentation
========================================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/badge/python-3.13+-blue.svg
    :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/pytorch-2.9+-red.svg
    :target: https://pytorch.org/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

**Build highly customizable convolutional neural networks from scratch with an intuitive Python API.**

A production-ready Python package that empowers researchers and developers to create flexible, modular CNNs 
with fine-grained control over every architectural decision while maintaining full compatibility with the 
PyTorch ecosystem.

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
   user_guide/layers
   user_guide/blocks
   user_guide/models
   user_guide/advanced

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/blocks
   api/layers
   api/models
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/basic_usage
   examples/advanced_patterns
   examples/transfer_learning
   examples/custom_architectures

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   development/contributing
   development/testing
   development/performance
   development/changelog

Quick Links
===========

* :ref:`Quick Start <quick_start>`
* :ref:`API Reference <api_reference>`
* :ref:`Examples <examples>`
* `GitHub Repository <https://github.com/codewithdark-git/torchvision-customizer>`_
* `Report Issues <https://github.com/codewithdark-git/torchvision-customizer/issues>`_

Key Features
============

🔧 **Granular Control**
   Customize network depth, channels, activations, normalization, pooling, and more

🧩 **Modular Blocks**
   Pre-built components (ConvBlock, ResidualBlock, InceptionBlock, SEBlock, Bottleneck, etc.)

🏗️ **Multiple APIs**
   Simple interface for quick prototyping, builder pattern for advanced users

📊 **Model Introspection**
   Get parameter counts, FLOPs, memory footprint, and architecture summaries

⚙️ **Configuration-Based**
   Define models via YAML/JSON for reproducibility and sharing

🚀 **Production-Ready**
   Full type hints, comprehensive tests (382 tests, 100% passing), and CI/CD integration

📈 **Architecture Patterns**
   Sequential, ResNet, DenseNet, Inception, MobileNet-style blocks

🎓 **Well-Documented**
   Comprehensive examples (113 examples) and extensive API documentation

Quick Example
=============

Create a simple CNN in just a few lines:

.. code-block:: python

    from torchvision_customizer import CustomCNN
    
    # Create a simple 4-layer CNN
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=4,
        channels=[64, 128, 256, 512],
        activation='relu'
    )
    
    # Inspect your model
    print(model.summary())
    print(f"Total parameters: {model.count_parameters():,}")

Why torchvision-customizer?
============================

Traditional deep learning frameworks often force you to choose between:

- **Simplicity vs. Flexibility** - High-level APIs are easy but inflexible
- **Speed vs. Control** - Quick prototyping comes at the cost of architectural control
- **Modularity vs. Performance** - Modular designs often sacrifice efficiency

**torchvision-customizer** bridges these gaps by providing:

✨ An intuitive, Pythonic API that feels natural to use  
🎯 Fine-grained control over every network component  
⚡ Near-native PyTorch performance  
🔄 Seamless integration with existing PyTorch workflows  
📦 Pre-built, battle-tested components  
🧪 Comprehensive testing and validation  

Project Statistics
==================

* **382 Tests** - Comprehensive test suite with 100% pass rate
* **113 Examples** - Real-world usage examples for all features
* **7,750+ Lines** - Production-quality code with full type hints
* **90+ Components** - Pre-built classes and functions
* **8 Steps** - Complete development journey documented

Installation
============

Install from source:

.. code-block:: bash

    git clone https://github.com/codewithdark-git/torchvision-customizer.git
    cd torchvision-customizer
    pip install -e ".[dev]"

Requirements:

* Python 3.13 or higher
* PyTorch 2.9.0 or higher
* torchvision 0.24.1 or higher

Support
=======

Need help? Check out:

* **Documentation**: Comprehensive guides and API reference
* **Examples**: 113 working examples for all use cases
* **GitHub Issues**: Report bugs or request features
* **Discussions**: Ask questions and share ideas

Contributing
============

We welcome contributions! See our :doc:`Contributing Guide <development/contributing>` for details.

License
=======

This project is licensed under the MIT License - see the LICENSE file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
