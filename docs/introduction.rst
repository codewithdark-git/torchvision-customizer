Introduction
============

What is torchvision-customizer?
-------------------------------

**torchvision-customizer** is a production-ready Python package that empowers researchers and developers to build, customize, and fine-tune CNN architectures with unprecedented flexibility.

The Problem
-----------

Deep learning research often involves tweaking model architectures—changing normalization, adding attention, modifying block types. Standard libraries like `torchvision` offer pre-built models (e.g., ``resnet50``), but:

* **Modifying them** requires monkey-patching or copy-pasting code
* **Adding attention** means understanding internal structure
* **Loading partial weights** when architecture changes is error-prone
* **Experimenting** with new block types is tedious

Our Solution
------------

**torchvision-customizer** solves this with:

1. **Hybrid Builder (v2.1)**: Load any pre-trained torchvision model and customize it with patches

   .. code-block:: python

      model = HybridBuilder().from_torchvision(
          "resnet50",
          weights="IMAGENET1K_V2",
          patches={"layer4": {"wrap": "se"}},
          num_classes=100,
      )

2. **Registry**: All blocks (Conv, Residual, SE, CBAM) are discoverable components

   .. code-block:: python

      block = registry.get('cbam_block')(channels=256)

3. **Recipes**: Architectures defined by declarative strings or YAML

   .. code-block:: python

      model = build_recipe(Recipe(
          stem="conv(64)",
          stages=["residual(64) x 2"],
          head="linear(10)"
      ))

4. **Composition**: Operators (``>>``) make it easy to chain stages

   .. code-block:: python

      model = Stem(64) >> Stage(128, blocks=2) >> Head(10)

Philosophy
----------

*   **Flexibility First**: Customize anything without monkey-patching
*   **Preserve Weights**: Smart loading preserves pre-trained knowledge
*   **Components are Modules**: All components are standard ``nn.Module`` subclasses
*   **Declarative & Programmatic**: Use YAML configs or Python code
*   **Explicit > Implicit**: Channel dimensions are handled explicitly where needed

Who Should Use This?
--------------------

* **Researchers** experimenting with attention mechanisms
* **Engineers** fine-tuning models for production
* **Students** learning about CNN architectures
* **Anyone** who wants more control than ``torchvision.models``

Version History
---------------

* **v2.1.0** (Current): Hybrid Models, 12 new blocks, CLI, YAML recipes
* **v2.0.0**: 3-Tier API (Registry, Recipes, Composer)
* **v1.x**: Legacy CustomCNN class (deprecated)

Getting Started
---------------

.. code-block:: bash

   pip install torchvision-customizer

.. code-block:: python

   from torchvision_customizer import HybridBuilder
   
   model = HybridBuilder().from_torchvision(
       "resnet50", 
       weights="IMAGENET1K_V2",
       num_classes=10
   )

See :doc:`quick_start` for more examples.
