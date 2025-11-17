====================
Quick Start Guide
====================

Your First CNN Model
====================

Create a basic CNN in just 3 lines:

.. code-block:: python

    from torchvision_customizer import CustomCNN
    
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000
    )

That's it! You now have a working CNN. Let's train it:

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from torch.optim import Adam
    
    # Create dummy data
    x = torch.randn(32, 3, 224, 224)  # Batch of 32 images
    y = torch.randint(0, 1000, (32,))  # Labels
    
    # Define optimizer and loss
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = F.cross_entropy
    
    # Training loop
    model.train()
    for i in range(100):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            print(f"Step {i+1:3d}: Loss = {loss.item():.4f}")

Inspecting Your Model
=====================

Get model statistics:

.. code-block:: python

    from torchvision_customizer.utils import get_model_summary
    
    # Print summary
    summary = get_model_summary(model)
    print(summary)
    
    # Get individual metrics
    print(f"Parameters: {summary['total_params']:,}")
    print(f"FLOPs: {summary['total_flops']:,}")
    print(f"Memory: {summary['total_memory_mb']:.2f} MB")

Using the Builder Pattern
=========================

For more control, use the builder:

.. code-block:: python

    from torchvision_customizer import CNNBuilder
    
    model = (CNNBuilder(input_shape=(3, 224, 224))
        .add_conv_block(64, kernel_size=7, stride=2)
        .add_pooling('max', kernel_size=3, stride=2)
        .add_residual_block(128, num_convs=2)
        .add_residual_block(256, num_convs=2, downsample=True)
        .add_global_pooling('adaptive')
        .add_classifier([512, 256], num_classes=1000)
        .build()
    )

Each method returns the builder, so you can chain them:

.. code-block:: python

    # Step-by-step building
    builder = CNNBuilder(input_shape=(3, 224, 224))
    
    # Add layers one by one
    builder.add_conv_block(64)
    builder.add_pooling('max')
    builder.add_residual_block(128)
    
    # Build when ready
    model = builder.build()

Working with Blocks
===================

Use pre-built blocks for specific architectures:

.. code-block:: python

    from torchvision_customizer.blocks import (
        ConvBlock, ResidualBlock, BottleneckBlock, InceptionModule
    )
    
    # Convolutional block
    conv = ConvBlock(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        padding=1,
        activation='relu',
        normalization='batch'
    )
    
    # Residual block with skip connection
    residual = ResidualBlock(
        in_channels=64,
        out_channels=64,
        num_convs=2,
        downsample=False,
        activation='relu'
    )
    
    # Bottleneck block (ResNet-style)
    bottleneck = BottleneckBlock(
        in_channels=256,
        bottleneck_channels=64,
        expansion=4,
        stride=1
    )
    
    # Inception module (multi-branch)
    inception = InceptionModule(
        in_channels=192,
        out_1x1=64,
        red_3x3=96,
        out_3x3=128,
        red_5x5=16,
        out_5x5=32,
        out_pool=32
    )

Using Different Activations
============================

Customize activation functions:

.. code-block:: python

    from torchvision_customizer.layers import get_activation
    
    # Available activations
    activations = [
        'relu',      # ReLU
        'leaky_relu',# LeakyReLU with negative slope
        'gelu',      # Gaussian Error Linear Unit
        'mish',      # Mish activation
        'swish',     # Swish/SiLU
        'elu',       # Exponential Linear Unit
        'selu',      # Scaled ELU
        'sigmoid',   # Sigmoid
        'tanh',      # Hyperbolic tangent
    ]
    
    # Create activation layer
    relu = get_activation('relu')
    gelu = get_activation('gelu')
    
    # Use in model
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        activation='gelu'  # Use GELU
    )

Pooling Options
===============

Choose different pooling strategies:

.. code-block:: python

    from torchvision_customizer.layers import get_pooling
    
    # Different pooling methods
    max_pool = get_pooling('max', kernel_size=2, stride=2)
    avg_pool = get_pooling('avg', kernel_size=2, stride=2)
    adaptive = get_pooling('adaptive_max', output_size=7)
    stochastic = get_pooling('stochastic', kernel_size=2, stride=2)
    
    # Use in model
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        pooling_type='adaptive'

Transfer Learning
=================

Fine-tune for your task:

.. code-block:: python

    from torchvision_customizer import CustomCNN
    from torch.optim import SGD
    from torch.optim.lr_scheduler import StepLR
    import torch.nn.functional as F
    
    # 1. Create a pre-trained base model
    base_model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000
    )
    # (In practice, load pre-trained weights)
    
    # 2. Modify for your task
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=10,  # Your dataset classes
        num_conv_blocks=4
    )
    
    # 3. Freeze early layers
    for name, param in model.named_parameters():
        if 'block1' in name or 'block2' in name:
            param.requires_grad = False
    
    # 4. Train with lower learning rate
    optimizer = SGD(
        model.parameters(),
        lr=0.001,  # Lower LR for fine-tuning
        momentum=0.9
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 5. Train your model
    model.train()
    for epoch in range(100):
        # Training code here
        pass
        scheduler.step()

Configuration Files
===================

Define models with YAML:

.. code-block:: yaml

    # model_config.yaml
    input_shape: [3, 224, 224]
    num_classes: 1000
    num_conv_blocks: 4
    channels: [64, 128, 256, 512]
    activation: relu
    normalization: batch
    pooling_type: max
    dropout: 0.2

Load and use:

.. code-block:: python

    import yaml
    from torchvision_customizer import CustomCNN
    
    # Load from YAML
    with open('model_config.yaml') as f:
        config = yaml.safe_load(f)
    
    model = CustomCNN(**config)
    print(model)

Or define as dictionary:

.. code-block:: python

    config = {
        'input_shape': (3, 224, 224),
        'num_classes': 10,
        'num_conv_blocks': 3,
        'channels': [32, 64, 128],
        'activation': 'gelu',
        'dropout': 0.1
    }
    
    model = CustomCNN(**config)

Common Patterns
===============

Image Classification
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torchvision_customizer import CustomCNN
    
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=10,  # Number of classes
        num_conv_blocks=4,
        dropout=0.3  # Regularization
    )

Mobile-Friendly Model
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=10,
        num_conv_blocks=3,
        channels=[32, 64, 128],  # Smaller
        depthwise=True  # Efficient
    )

High-Capacity Model
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=5,
        channels=[64, 128, 256, 512, 1024],  # Larger
        expansion_factor=2  # More capacity
    )

Residual Architecture
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torchvision_customizer.blocks import ResidualArchitecture
    
    model = ResidualArchitecture(
        input_shape=(3, 224, 224),
        num_classes=1000,
        depth=50,  # ResNet-50 style
        bottleneck=True
    )

Next Steps
==========

Now that you have the basics:

1. **Explore Blocks**: :doc:`user_guide/blocks`
2. **Learn Layers**: :doc:`user_guide/layers`
3. **Advanced Patterns**: :doc:`user_guide/advanced`
4. **See Examples**: :doc:`examples/basic_usage`
5. **API Reference**: :doc:`api/blocks`

Tips & Tricks
=============

✅ **Always check model summary** before training to catch architecture issues  
✅ **Use validation** to verify your configurations are correct  
✅ **Start small** and scale up if needed  
✅ **Experiment with activations** for performance improvements  
✅ **Use dropouts** for regularization in large models  
✅ **Save configurations** to YAML for reproducibility  
✅ **Monitor memory usage** with model summary before training  

Common Issues
=============

"Shape mismatch error"
~~~~~~~~~~~~~~~~~~~~~~

If you get shape mismatch errors:

.. code-block:: python

    # Check input/output shapes
    x = torch.randn(1, 3, 224, 224)
    summary = get_model_summary(model)
    print(summary)  # Shows all layer shapes

"Out of memory (OOM)"
~~~~~~~~~~~~~~~~~~~~~

If you run out of GPU memory:

.. code-block:: python

    # Reduce batch size
    batch_size = 16  # Instead of 32
    
    # Or use smaller model
    model = CustomCNN(
        num_conv_blocks=2,  # Fewer blocks
        channels=[32, 64]   # Smaller channels
    )

"Slow training"
~~~~~~~~~~~~~~~

Speed up training:

.. code-block:: python

    # Use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use mixed precision
    from torch.cuda.amp import autocast
    with autocast():
        output = model(x)

Questions?
==========

Check out the :doc:`examples/basic_usage` section for more patterns!
