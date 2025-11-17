================
User Guide: Basics
================

Understanding the Architecture
===============================

torchvision-customizer is organized into a hierarchy:

.. code-block:: text

    Model (High-level)
       ↓
    Blocks (Composite)
       ↓
    Layers (Atomic)
       ↓
    PyTorch Modules

Each level provides different levels of abstraction and control.

Core Concepts
=============

Channels
--------

The number of feature maps in each layer:

.. code-block:: python

    model = CustomCNN(
        channels=[64, 128, 256, 512],  # Channel progression
        input_shape=(3, 224, 224)
    )

- **Input**: 3 channels (RGB)
- **After Block 1**: 64 channels
- **After Block 2**: 128 channels
- etc.

Kernel Size
-----------

Receptive field of each convolution:

.. code-block:: python

    block = ConvBlock(
        in_channels=64,
        out_channels=128,
        kernel_size=3,  # 3x3 kernel
        padding=1       # Keep spatial size
    )

- **Small kernel** (3x3): Local features, efficient
- **Large kernel** (7x7): Wider receptive field, more compute

Stride
------

How much to move the kernel:

.. code-block:: python

    block = ConvBlock(
        kernel_size=3,
        stride=1,  # Move by 1 (no downsampling)
        padding=1
    )
    
    # vs.
    
    block = ConvBlock(
        kernel_size=3,
        stride=2,  # Move by 2 (downsample 2x)
        padding=1
    )

Stride effects:
- **stride=1**: Output same spatial size as input
- **stride=2**: Output is 1/2 size in each dimension
- **stride=4**: Output is 1/4 size in each dimension

Padding
-------

Add pixels around input:

.. code-block:: python

    # No padding (valid convolution)
    block = ConvBlock(kernel_size=3, padding=0)
    # Output size = (input_size - 3 + 1) = input_size - 2
    
    # Same padding
    block = ConvBlock(kernel_size=3, padding=1)
    # Output size = input_size
    
    # Full padding
    block = ConvBlock(kernel_size=3, padding=2)
    # Output size = input_size + 2

Creating Models
===============

Simple Sequential
-----------------

Stack layers sequentially:

.. code-block:: python

    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=10,
        num_conv_blocks=4,
        channels=[64, 128, 256, 512]
    )

This creates 4 convolutional blocks plus a classifier.

With Builder
------------

More control:

.. code-block:: python

    model = (CNNBuilder(input_shape=(3, 224, 224))
        .add_conv_block(64, kernel_size=7, stride=2)
        .add_pooling('max')
        .add_conv_block(128)
        .add_conv_block(256)
        .add_global_pooling('adaptive')
        .add_classifier([512], num_classes=10)
        .build()
    )

With Blocks Directly
--------------------

Maximum control:

.. code-block:: python

    import torch.nn as nn
    from torchvision_customizer.blocks import ConvBlock, ResidualBlock
    
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2)
            self.pool1 = nn.MaxPool2d(3, stride=2)
            self.res1 = ResidualBlock(64, 64, num_convs=2)
            self.res2 = ResidualBlock(64, 128, num_convs=2, downsample=True)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.res1(x)
            x = self.res2(x)
            x = self.gap(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x

Input Shapes
============

Understanding input/output shapes:

.. code-block:: python

    import torch
    from torchvision_customizer.utils import get_model_summary
    
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=10
    )
    
    # Print detailed shapes
    summary = get_model_summary(model, input_shape=(1, 3, 224, 224))
    for layer, shapes in summary['layer_shapes'].items():
        print(f"{layer}: {shapes}")

Batch Dimensions
----------------

Always include batch dimension:

.. code-block:: python

    # WRONG - missing batch dimension
    x = torch.randn(3, 224, 224)
    # RuntimeError: Expected 4-D input (got 3-D input)
    
    # CORRECT - include batch of 1
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    
    # Or batch of 32
    x = torch.randn(32, 3, 224, 224)
    output = model(x)  # Shape: (32, 10)

Computing Output Size
~~~~~~~~~~~~~~~~~~~~~

Formula for convolutional layers:

.. code-block:: text

    output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1

Examples:

.. code-block:: python

    # Input: 224x224, Kernel: 3, Padding: 1, Stride: 1
    output = (224 + 2*1 - 3) / 1 + 1 = 224
    
    # Input: 224x224, Kernel: 3, Padding: 1, Stride: 2
    output = (224 + 2*1 - 3) / 2 + 1 = 112

Normalization Strategies
========================

Batch Normalization
-------------------

Normalize within a batch:

.. code-block:: python

    block = ConvBlock(
        in_channels=64,
        out_channels=128,
        normalization='batch'
    )

Best for:
- Large batch sizes (≥32)
- Training with many samples
- Stable training dynamics

Layer Normalization
-------------------

Normalize per sample:

.. code-block:: python

    block = ConvBlock(
        normalization='layer'
    )

Best for:
- Small batch sizes
- Variable-length sequences
- Transformers

Group Normalization
--------------------

Normalize within groups:

.. code-block:: python

    block = ConvBlock(
        normalization='group',
        num_groups=32  # Number of groups
    )

Best for:
- Small batch sizes with groups
- Fine-grained control
- Hybrid approaches

None (No Normalization)
-----------------------

.. code-block:: python

    block = ConvBlock(
        normalization=None
    )

Manual Gradient Control
=======================

Freezing Layers
---------------

Prevent updates to specific layers:

.. code-block:: python

    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=10
    )
    
    # Freeze early layers
    for name, param in model.named_parameters():
        if 'block1' in name or 'block2' in name:
            param.requires_grad = False
    
    # Check frozen status
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

Unfreezing Layers
------------------

.. code-block:: python

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    # Unfreeze only specific layers
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True

Layer Groups
~~~~~~~~~~~~

Create layer groups for differential learning rates:

.. code-block:: python

    # Group layers
    early_layers = []
    late_layers = []
    
    for name, param in model.named_parameters():
        if 'block1' in name or 'block2' in name:
            early_layers.append(param)
        else:
            late_layers.append(param)
    
    # Different learning rates
    optimizer = torch.optim.Adam([
        {'params': early_layers, 'lr': 1e-4},
        {'params': late_layers, 'lr': 1e-3}
    ])

Device Handling
===============

CPU vs GPU
----------

.. code-block:: python

    import torch
    
    # Check availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Move data to device
    x = x.to(device)
    y = y.to(device)

Multi-GPU
---------

.. code-block:: python

    import torch.nn as nn
    
    # Wrap model for multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)

Model State Management
======================

Saving Models
-------------

.. code-block:: python

    # Save only weights
    torch.save(model.state_dict(), 'model.pth')
    
    # Load weights
    model = CustomCNN(input_shape=(3, 224, 224), num_classes=10)
    model.load_state_dict(torch.load('model.pth'))
    
    # Save entire model (not recommended)
    torch.save(model, 'model_full.pth')

Checkpointing
-------------

.. code-block:: python

    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_acc': best_acc
    }
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
    
    # Load checkpoint
    checkpoint = torch.load('checkpoint_epoch_10.pth')
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

Mode Switching
==============

Training Mode
-------------

.. code-block:: python

    model.train()  # Enable dropout, batch norm updates, etc.

Evaluation Mode
---------------

.. code-block:: python

    model.eval()  # Disable dropout, use running batch norm stats

Inference
---------

.. code-block:: python

    model.eval()
    with torch.no_grad():  # Disable gradient computation
        output = model(x)

Debugging Tips
==============

Print Model Architecture
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    print(model)
    # Or with more detail
    from torchvision_customizer.utils import get_model_summary
    print(get_model_summary(model))

Check Tensor Shapes
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    x = torch.randn(1, 3, 224, 224)
    print(f"Input: {x.shape}")
    
    x = model.conv1(x)
    print(f"After conv1: {x.shape}")
    
    x = model.res1(x)
    print(f"After res1: {x.shape}")

Monitor Gradients
~~~~~~~~~~~~~~~~~

.. code-block:: python

    model.train()
    x = torch.randn(1, 3, 224, 224)
    y = torch.randint(0, 10, (1,))
    
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()
    
    # Check gradient flow
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_mean={param.grad.mean():.6f}, grad_std={param.grad.std():.6f}")

Next Steps
==========

- :doc:`../user_guide/layers` - Work with individual layers
- :doc:`../user_guide/blocks` - Use building blocks
- :doc:`../user_guide/models` - Create complete models
- :doc:`../examples/basic_usage` - See working examples
