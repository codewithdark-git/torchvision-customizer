=======================
API Reference: Models
=======================

.. py:module:: torchvision_customizer.models

High-level model classes for rapid prototyping.

CustomCNN
=========

.. py:class:: CustomCNN

    Simple sequential CNN for quick model creation.
    
    **Parameters:**
    
    - ``input_shape`` (tuple): Input dimensions (C, H, W)
    - ``num_classes`` (int): Number of output classes
    - ``num_conv_blocks`` (int): Number of convolutional blocks
    - ``channels`` (list): Channel dimensions per block
    - ``activation`` (str): Activation function ('relu', 'gelu', etc.)
    - ``normalization`` (str): Normalization type ('batch', 'layer', 'group')
    - ``pooling_type`` (str): Pooling type ('max', 'avg', 'adaptive')
    - ``dropout`` (float): Dropout rate (0.0-1.0)
    - ``depthwise`` (bool): Use depthwise convolutions
    
    **Example:**
    
    .. code-block:: python
    
        from torchvision_customizer import CustomCNN
        import torch
        
        model = CustomCNN(
            input_shape=(3, 224, 224),
            num_classes=10,
            num_conv_blocks=4,
            channels=[64, 128, 256, 512],
            activation='relu',
            dropout=0.3
        )
        
        x = torch.randn(32, 3, 224, 224)
        output = model(x)  # Shape: (32, 10)

CNNBuilder
==========

.. py:class:: CNNBuilder

    Builder pattern for flexible model construction.
    
    **Methods:**
    
    - ``add_conv_block(...)``: Add convolutional block
    - ``add_residual_block(...)``: Add residual block
    - ``add_pooling(...)``: Add pooling layer
    - ``add_global_pooling(...)``: Add global pooling
    - ``add_classifier(...)``: Add classification head
    - ``build()``: Create the model
    
    **Example:**
    
    .. code-block:: python
    
        from torchvision_customizer import CNNBuilder
        
        model = (CNNBuilder(input_shape=(3, 224, 224))
            .add_conv_block(64, kernel_size=7, stride=2)
            .add_pooling('max', kernel_size=3, stride=2)
            .add_residual_block(128, num_convs=2)
            .add_residual_block(256, num_convs=2, downsample=True)
            .add_global_pooling('adaptive')
            .add_classifier([512], num_classes=10)
            .build()
        )

add_conv_block()
~~~~~~~~~~~~~~~~

Add a convolutional block:

.. code-block:: python

    builder.add_conv_block(
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        activation='relu',
        normalization='batch'
    )

add_residual_block()
~~~~~~~~~~~~~~~~~~~~

Add a residual block:

.. code-block:: python

    builder.add_residual_block(
        out_channels=128,
        num_convs=2,
        downsample=False,
        activation='relu'
    )

add_pooling()
~~~~~~~~~~~~~

Add pooling layer:

.. code-block:: python

    builder.add_pooling(
        pooling_type='max',
        kernel_size=2,
        stride=2
    )

add_global_pooling()
~~~~~~~~~~~~~~~~~~~~

Add global pooling:

.. code-block:: python

    builder.add_global_pooling(pooling_type='adaptive')

add_classifier()
~~~~~~~~~~~~~~~~

Add classification head:

.. code-block:: python

    builder.add_classifier(
        hidden_dims=[512, 256],
        num_classes=10,
        dropout=0.3
    )

build()
~~~~~~~

Create the final model:

.. code-block:: python

    model = builder.build()

Model Methods
=============

All models inherit common methods:

.. code-block:: python

    # Forward pass
    output = model(x)
    
    # Get model summary
    summary = model.get_summary()
    
    # Count parameters
    num_params = model.count_parameters()
    
    # Get FLOPs
    flops = model.get_flops(input_size=(1, 3, 224, 224))
    
    # Save/load
    torch.save(model.state_dict(), 'model.pth')
    model.load_state_dict(torch.load('model.pth'))

Configuration Loading
=====================

Load from Dictionary
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = {
        'input_shape': (3, 224, 224),
        'num_classes': 10,
        'num_conv_blocks': 4,
        'channels': [64, 128, 256, 512],
        'activation': 'relu'
    }
    
    model = CustomCNN(**config)

Load from YAML
~~~~~~~~~~~~~~

.. code-block:: python

    import yaml
    from torchvision_customizer import CustomCNN
    
    with open('model_config.yaml') as f:
        config = yaml.safe_load(f)
    
    model = CustomCNN(**config)

Common Model Configurations
============================

Mobile-Friendly
~~~~~~~~~~~~~~~

.. code-block:: python

    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=10,
        num_conv_blocks=3,
        channels=[32, 64, 128],
        depthwise=True,
        dropout=0.2
    )

Standard (ResNet-50 Like)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=4,
        channels=[64, 128, 256, 512]
    )

High-Capacity
~~~~~~~~~~~~~~

.. code-block:: python

    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=5,
        channels=[64, 128, 256, 512, 1024],
        dropout=0.3
    )

Lightweight (MobileNet-Like)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=6,
        channels=[32, 32, 64, 64, 128, 128],
        depthwise=True
    )

Model Utilities
===============

get_model()
~~~~~~~~~~~

Factory function to get models by name:

.. code-block:: python

    from torchvision_customizer.models import get_model
    
    model = get_model('custom_cnn', input_shape=(3, 224, 224), num_classes=10)

Complete Training Example
==========================

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from torch.optim import Adam
    from torchvision_customizer import CustomCNN
    
    # Create model
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=10
    )
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(100):
        # Create dummy batch
        x = torch.randn(32, 3, 224, 224).to(device)
        y = torch.randint(0, 10, (32,)).to(device)
        
        # Forward pass
        output = model(x)
        loss = F.cross_entropy(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        x_test = torch.randn(10, 3, 224, 224).to(device)
        predictions = model(x_test)
        print(f"Output shape: {predictions.shape}")

Next Steps
==========

- :doc:`../api/blocks` - Learn about blocks
- :doc:`../api/layers` - Learn about layers
- :doc:`../examples/basic_usage` - See working examples
- :doc:`../user_guide/advanced` - Advanced patterns
