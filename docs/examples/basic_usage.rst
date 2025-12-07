Basic Usage
===========

This example demonstrates how to build a model using the default component registry and the simple composer API.

.. code-block:: python

    import torch
    from torchvision_customizer import Stem, Stage, Head
    
    # 1. Define the model
    model = (
        Stem(channels=64, kernel=7, stride=2)
        >> Stage(channels=64, blocks=2, pattern='residual', in_channels=64)
        >> Stage(channels=128, blocks=2, pattern='residual', downsample=True, in_channels=64)
        >> Head(num_classes=10)
    )
    
    # 2. Forward pass
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    
    print(f"Output shape: {y.shape}") # torch.Size([1, 10])
    print(model.explain())
