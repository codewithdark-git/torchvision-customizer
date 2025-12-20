Quick Start
===========

This guide will get you up and running with torchvision-customizer in 5 minutes.

Installation
------------

.. code-block:: bash

   pip install torchvision-customizer

Hello World: Customize a Pre-trained Model (v2.1)
-------------------------------------------------

The fastest way to get started is with the Hybrid Builder:

.. code-block:: python

   from torchvision_customizer import HybridBuilder

   # Load ResNet50 with ImageNet weights
   # Add SE attention to layer3
   # Customize for 10 classes
   model = HybridBuilder().from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       patches={"layer3": {"wrap": "se"}},
       num_classes=10,
   )

   # Test it
   import torch
   x = torch.randn(1, 3, 224, 224)
   output = model(x)
   print(f"Output shape: {output.shape}")  # torch.Size([1, 10])

Hello World: Build from Scratch
-------------------------------

Use the Composer API to build custom architectures:

.. code-block:: python

   from torchvision_customizer import Stem, Stage, Head

   model = (
       Stem(64)
       >> Stage(64, blocks=2, pattern='residual')
       >> Stage(128, blocks=2, pattern='residual', downsample=True)
       >> Head(num_classes=10)
   )

   print(model.explain())

Hello World: Use a Recipe
-------------------------

Define architectures declaratively:

.. code-block:: python

   from torchvision_customizer import Recipe, build_recipe

   recipe = Recipe(
       stem="conv(64, k=7, s=2)",
       stages=[
           "residual(64) x 2",
           "residual(128) x 2 | downsample",
       ],
       head="linear(10)"
   )
   model = build_recipe(recipe)

Hello World: CLI
----------------

Use the command-line interface:

.. code-block:: bash

   # Create a recipe from template
   tvc create-recipe --template hybrid_resnet_se --output model.yaml

   # Build and benchmark
   tvc build --yaml model.yaml --output model.pt
   tvc benchmark --yaml model.yaml --device cuda

   # List available options
   tvc list-backbones
   tvc list-blocks

Next Steps
----------

* :doc:`user_guide/hybrid` - Learn about hybrid model customization
* :doc:`user_guide/basics` - Understand the 3-tier API
* :doc:`examples/hybrid_models` - See more hybrid examples
* :doc:`api/blocks` - Explore all building blocks

Common Patterns
---------------

Fine-tune with Frozen Backbone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = HybridBuilder().from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       num_classes=10,
       freeze_backbone=True,
       unfreeze_stages=[3],  # Only train last stage
   )

   # Only ~2M trainable params instead of 25M
   print(f"Trainable: {model.count_parameters(trainable_only=True):,}")

Add Attention to Multiple Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = HybridBuilder().from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       patches={
           "layer2": {"wrap": "se"},
           "layer3": {"wrap": "cbam_block"},
           "layer4": {"wrap": "eca"},
       },
       num_classes=100,
   )

Build an EfficientNet-style Network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from torchvision_customizer import Stem, Stage, Head

   model = (
       Stem(32, kernel=3, stride=2)
       >> Stage(16, blocks=1, pattern='mbconv')
       >> Stage(24, blocks=2, pattern='mbconv', downsample=True)
       >> Stage(40, blocks=2, pattern='mbconv', downsample=True)
       >> Head(num_classes=1000, dropout=0.2)
   )

Load from YAML
^^^^^^^^^^^^^^

.. code-block:: yaml

   # model.yaml
   backbone:
     name: efficientnet_b4
     weights: IMAGENET1K_V1
   head:
     num_classes: 200
     dropout: 0.4

.. code-block:: python

   from torchvision_customizer.recipe import load_yaml_recipe
   model = load_yaml_recipe("model.yaml")
