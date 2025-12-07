.. torchvision-customizer documentation master file

========================================
torchvision-customizer Documentation
========================================

**Build highly customizable convolutional neural networks from scratch with a 3-tier API.**

A production-ready Python package that empowers researchers and developers to create flexible, modular CNNs with fine-grained control over every architectural decision.

Key Concepts
============

1. **Component Registry (Tier 1)**: Centralized management of all building blocks.
2. **Architecture Recipes (Tier 2)**: Declarative, string-based model definitions.
3. **Model Composer (Tier 3)**: Fluent, operator-based API for programmatic construction.
4. **Templates**: Parametric implementations of standard architectures (ResNet, VGG, etc.).

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   introduction
   installation
   quick_start

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/blocks
   api/recipe
   api/compose
   api/templates

Examples
========

**Composer API**

.. code-block:: python

    from torchvision_customizer import Stem, Stage, Head

    model = (
        Stem(64)
        >> Stage(64, blocks=2, pattern='residual')
        >> Stage(128, blocks=2, pattern='bottleneck', downsample=True)
        >> Head(10)
    )

**Recipe API**

.. code-block:: python

    from torchvision_customizer import Recipe, build_recipe

    recipe = Recipe(
        stem="conv(64)",
        stages=["residual(64) x 2", "residual(128) x 2 | downsample"],
        head="linear(10)"
    )
    model = build_recipe(recipe)

Features
========

* **Granular Control**: Customize every aspect of the architecture (depth, width, attention, activation).
* **Pattern Mixing**: Mix different block types in the same stage (e.g., ``residual+se``).
* **Introspection**: Built-in ``model.explain()`` for human-readable summaries.
* **No Pre-trained Weights**: Focused purely on architecture definition and instantiation.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
