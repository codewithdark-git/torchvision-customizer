Advanced Usage
==============

Mixing Patterns
---------------

You can mix different block patterns within a single stage using ``+``.

.. code-block:: python

    # Residual blocks with SE attention
    Stage(128, blocks=3, pattern='residual+se')

Customizing Recipes
-------------------

Recipes can be serialized to dictionaries or YAML, making them perfect for config-driven experiments.

.. code-block:: python

    config = {
        'stem': 'conv(64, k=7, s=2)',
        'stages': [
            'residual(64) x 3',
            'bottleneck(128) x 4 | downsample'
        ],
        'head': {'type': 'linear', 'num_classes': 100}
    }
    
    from torchvision_customizer.recipe import build_from_config
    model = build_from_config(config)

Model Introspection
-------------------

Use ``model.explain()`` to see a human-readable summary of your custom architecture.

.. code-block:: python

    print(model.explain())
