The 3-Tier System
=================

torchvision-customizer is built around three tiers of abstraction, designed to flexible for different needs.

Tier 1: Component Registry
--------------------------

The foundation of the library. It manages all building blocks.

.. code-block:: python

    from torchvision_customizer import registry
    
    # Get a block class
    ConvBlock = registry.get('conv')
    block = ConvBlock(64, 128)

Tier 2: Architecture Recipes
----------------------------

Declarative blueprints. Ideal for saving configurations and rapid experimentation.

.. code-block:: python

    from torchvision_customizer import Recipe, build_recipe
    
    recipe = Recipe(
        stem="conv(64)",
        stages=["residual(64) x 2", "residual(128) x 2 | downsample"],
        head="linear(10)"
    )
    model = build_recipe(recipe)

Tier 3: Model Composer
----------------------

Fluent, imperative API. Best for readability and writing custom model scripts.

.. code-block:: python

    from torchvision_customizer import Stem, Stage, Head
    
    model = (
        Stem(64)
        >> Stage(64, blocks=2, pattern='residual')
        >> Stage(128, blocks=2, pattern='residual+se', downsample=True)
        >> Head(10)
    )

Parametric Templates
--------------------

For standard architectures with modern tweaks.

.. code-block:: python

    from torchvision_customizer import resnet, vgg
    
    # Standard ResNet-50
    model = resnet(layers=50)
