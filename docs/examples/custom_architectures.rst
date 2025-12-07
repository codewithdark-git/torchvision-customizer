Custom Architectures
====================

How to build a complex, non-standard architecture using Recipes.

.. code-block:: python

    from torchvision_customizer import Recipe, build_recipe
    
    # Define a recipe for a network with mixed block types and irregular stages/channels
    recipe = Recipe(
        name="ExperimentalNet",
        stem="conv(32, k=3, s=2)",
        stages=[
            "depthwise(32) x 1",
            "residual(48) x 3 | downsample",
            "residual+se(96) x 4 | downsample",
            "dense(128, growth_rate=32) | downsample", # Single dense block
        ],
        head="linear(100)"
    )
    
    model = build_recipe(recipe)
    print(model.explain())
