=======================
API Reference: Blocks
=======================

.. py:module:: torchvision_customizer.blocks

All building blocks available in torchvision-customizer.

ConvBlock
=========

.. py:class:: ConvBlock

    Composite block combining convolution, normalization, and activation.
    
    **Example:**
    
    .. code-block:: python
    
        from torchvision_customizer.blocks import ConvBlock
        
        block = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            padding=1,
            stride=1,
            activation='relu',
            normalization='batch'
        )
        
        x = torch.randn(1, 3, 224, 224)
        y = block(x)  # Output: (1, 64, 224, 224)

ResidualBlock
=============

.. py:class:: ResidualBlock

    Block with skip connections for gradient flow.
    
    **Example:**
    
    .. code-block:: python
    
        from torchvision_customizer.blocks import ResidualBlock
        
        block = ResidualBlock(
            in_channels=64,
            out_channels=64,
            num_convs=2,
            downsample=False,
            activation='relu'
        )
        
        x = torch.randn(1, 64, 224, 224)
        y = block(x)  # Output: (1, 64, 224, 224)

BottleneckBlock
===============

.. py:class:: BottleneckBlock

    ResNet-style bottleneck block for efficient multi-scale processing.
    
    **Features:**
    
    - 1x1 reduction convolution
    - 3x3 main convolution
    - 1x1 expansion convolution
    - Skip connection
    
    **Example:**
    
    .. code-block:: python
    
        from torchvision_customizer.blocks import BottleneckBlock
        
        block = BottleneckBlock(
            in_channels=256,
            bottleneck_channels=64,
            expansion=4,
            stride=1
        )

InceptionModule
===============

.. py:class:: InceptionModule

    Multi-branch module for parallel feature extraction.
    
    **Branches:**
    
    - 1x1 convolution
    - 1x1 → 3x3 convolution
    - 1x1 → 5x5 convolution
    - Max pooling → 1x1 convolution
    
    **Example:**
    
    .. code-block:: python
    
        from torchvision_customizer.blocks import InceptionModule
        
        block = InceptionModule(
            in_channels=192,
            out_1x1=64,
            red_3x3=96,
            out_3x3=128,
            red_5x5=16,
            out_5x5=32,
            out_pool=32
        )

DepthwiseBlock
==============

.. py:class:: DepthwiseBlock

    Efficient depthwise separable convolution.
    
    **Components:**
    
    - Depthwise convolution (channel-wise)
    - Pointwise convolution (1x1)
    
    **Example:**
    
    .. code-block:: python
    
        from torchvision_customizer.blocks import DepthwiseBlock
        
        block = DepthwiseBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1
        )

SEBlock
=======

.. py:class:: SEBlock

    Squeeze-and-Excitation attention block.
    
    **Components:**
    
    - Global average pooling (squeeze)
    - Fully connected layers (excitation)
    - Channel scaling
    
    **Example:**
    
    .. code-block:: python
    
        from torchvision_customizer.blocks import SEBlock
        
        block = SEBlock(
            channels=128,
            reduction_ratio=16
        )

Conv3DBlock
===========

.. py:class:: Conv3DBlock

    3D convolution block for volumetric data.
    
    **Example:**
    
    .. code-block:: python
    
        from torchvision_customizer.blocks import Conv3DBlock
        
        block = Conv3DBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        x = torch.randn(1, 3, 16, 224, 224)  # (B, C, D, H, W)
        y = block(x)  # Output: (1, 64, 16, 224, 224)

SuperConv2d
===========

.. py:class:: SuperConv2d

    Advanced 2D convolution with extra features.
    
    **Features:**
    
    - Configurable kernel initialization
    - Optional bias
    - Padding modes
    - Groups/depth-wise
    
    **Example:**
    
    .. code-block:: python
    
        from torchvision_customizer.blocks import SuperConv2d
        
        conv = SuperConv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=True
        )

SuperLinear
===========

.. py:class:: SuperLinear

    Advanced linear layer with extra features.
    
    **Example:**
    
    .. code-block:: python
    
        from torchvision_customizer.blocks import SuperLinear
        
        linear = SuperLinear(
            in_features=1024,
            out_features=512,
            bias=True
        )

ResidualArchitecture
====================

.. py:class:: ResidualArchitecture

    Complete ResNet-style architecture.
    
    **Parameters:**
    
    - input_shape: Input dimensions
    - num_classes: Number of output classes
    - depth: Network depth (16, 34, 50, 101, 152)
    - bottleneck: Use bottleneck blocks
    
    **Example:**
    
    .. code-block:: python
    
        from torchvision_customizer.blocks import ResidualArchitecture
        
        model = ResidualArchitecture(
            input_shape=(3, 224, 224),
            num_classes=1000,
            depth=50,  # ResNet-50
            bottleneck=True
        )

AdvancedArchitecture
====================

.. py:class:: AdvancedArchitecture

    Flexible architecture combining different block types.
    
    **Composition Options:**
    
    - Sequential
    - Parallel (multi-branch)
    - Hierarchical
    - Custom patterns
    
    **Example:**
    
    .. code-block:: python
    
        from torchvision_customizer.blocks import AdvancedArchitecture
        
        model = AdvancedArchitecture(
            input_shape=(3, 224, 224),
            num_classes=1000,
            composition='mixed'
        )

Factory Functions
=================

get_block()
-----------

.. py:function:: get_block(block_type: str, **kwargs) -> nn.Module

    Get a block by name.
    
    **Example:**
    
    .. code-block:: python
    
        from torchvision_customizer.blocks import get_block
        
        block = get_block(
            'conv',
            in_channels=64,
            out_channels=128,
            kernel_size=3
        )

Supported types: 'conv', 'residual', 'bottleneck', 'inception', 'depthwise', 'se', 'conv3d'

Complete Block Reference
========================

For complete API details, see:

- :doc:`../api_reference`
- Source code: `torchvision_customizer/blocks/`
- Examples: :doc:`../examples/basic_usage`

Working with Blocks
===================

Basic Usage
-----------

.. code-block:: python

    from torchvision_customizer.blocks import ConvBlock, ResidualBlock
    import torch
    
    # Create blocks
    conv = ConvBlock(3, 64, kernel_size=7, stride=2)
    res = ResidualBlock(64, 64, num_convs=2)
    
    # Forward pass
    x = torch.randn(1, 3, 224, 224)
    x = conv(x)  # (1, 64, 112, 112)
    x = res(x)   # (1, 64, 112, 112)

Chaining Blocks
---------------

.. code-block:: python

    import torch.nn as nn
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2)
            self.pool1 = nn.MaxPool2d(3, stride=2)
            self.res1 = ResidualBlock(64, 64)
            self.res2 = ResidualBlock(64, 128, downsample=True)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, 1000)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.res1(x)
            x = self.res2(x)
            x = self.gap(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x

Next Steps
==========

- :doc:`../api/layers` - Learn about layers
- :doc:`../api/models` - Explore model classes
- :doc:`../examples/basic_usage` - See working examples
