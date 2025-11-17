=======================
API Reference: Layers
=======================

.. py:module:: torchvision_customizer.layers

Individual atomic components and utilities for building networks.

Activation Functions
====================

Module: ``torchvision_customizer.layers.activations``

Available Activations
~~~~~~~~~~~~~~~~~~~~~~

- **ReLU**: Rectified Linear Unit (max(0, x))
- **LeakyReLU**: ReLU with small negative slope
- **GELU**: Gaussian Error Linear Unit
- **Mish**: Self-regularizing activation
- **Swish/SiLU**: Sigmoid-weighted linear unit
- **ELU**: Exponential Linear Unit
- **SELU**: Scaled ELU
- **Sigmoid**: Smooth step function
- **Tanh**: Hyperbolic tangent

get_activation()
~~~~~~~~~~~~~~~~

.. code-block:: python

    from torchvision_customizer.layers import get_activation
    
    # Get activation by name
    relu = get_activation('relu')
    gelu = get_activation('gelu')
    mish = get_activation('mish')
    
    # Use in model
    x = torch.randn(1, 64, 224, 224)
    y = relu(x)

Normalization Layers
====================

Module: ``torchvision_customizer.layers.normalizations``

Batch Normalization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torchvision_customizer.layers import get_normalization
    
    batch_norm = get_normalization('batch', num_features=64)
    
    # Properties:
    # - Normalizes across batch dimension
    # - Tracks running mean/variance
    # - Best for larger batches

Layer Normalization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    layer_norm = get_normalization('layer', num_features=64)
    
    # Properties:
    # - Normalizes per sample
    # - No running statistics
    # - Better for small batches

Group Normalization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    group_norm = get_normalization('group', num_channels=64, num_groups=8)
    
    # Properties:
    # - Normalizes within groups
    # - No batch statistics
    # - Good balance

Instance Normalization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    instance_norm = get_normalization('instance', num_features=64)
    
    # Properties:
    # - Normalizes per sample/channel
    # - Used in style transfer
    # - No batch statistics

Pooling Operations
==================

Module: ``torchvision_customizer.layers.pooling``

get_pooling()
~~~~~~~~~~~~~

Factory function for pooling layers:

.. code-block:: python

    from torchvision_customizer.layers import get_pooling
    
    # Max pooling
    max_pool = get_pooling('max', kernel_size=2, stride=2)
    
    # Average pooling
    avg_pool = get_pooling('avg', kernel_size=2, stride=2)
    
    # Adaptive pooling
    adaptive = get_pooling('adaptive_max', output_size=7)
    
    # Stochastic pooling
    stochastic = get_pooling('stochastic', kernel_size=2, stride=2)
    
    # LP-norm pooling
    lp_pool = get_pooling('lp', kernel_size=2, stride=2, norm_type=2)

PoolingBlock
~~~~~~~~~~~~

Pooling with optional dropout:

.. code-block:: python

    from torchvision_customizer.layers import PoolingBlock
    
    pool = PoolingBlock(
        pooling_type='max',
        kernel_size=2,
        stride=2,
        dropout_rate=0.1
    )

StochasticPool2d
~~~~~~~~~~~~~~~~

Probabilistic pooling for regularization:

.. code-block:: python

    from torchvision_customizer.layers import StochasticPool2d
    
    pool = StochasticPool2d(kernel_size=2, stride=2)
    
    # During training: random pooling
    # During inference: deterministic

LPPool2d
~~~~~~~~

L-p norm pooling (RMS pooling):

.. code-block:: python

    from torchvision_customizer.layers import LPPool2d
    
    # RMS pooling (p=2)
    lp_pool = LPPool2d(
        kernel_size=2,
        stride=2,
        norm_type=2
    )

Attention Mechanisms
====================

Module: ``torchvision_customizer.layers.attention``

Channel Attention
~~~~~~~~~~~~~~~~~

Focus on important channels:

.. code-block:: python

    from torchvision_customizer.layers import ChannelAttention
    
    attn = ChannelAttention(
        channels=128,
        reduction_ratio=16
    )

Spatial Attention
~~~~~~~~~~~~~~~~~

Focus on important spatial regions:

.. code-block:: python

    from torchvision_customizer.layers import SpatialAttention
    
    attn = SpatialAttention(kernel_size=7)

Multi-Head Attention
~~~~~~~~~~~~~~~~~~~~~

Self-attention with multiple heads:

.. code-block:: python

    from torchvision_customizer.layers import MultiHeadAttention
    
    attn = MultiHeadAttention(
        channels=128,
        num_heads=8
    )

Utility Functions
=================

Output Size Calculation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torchvision_customizer.layers import calculate_pooling_output_size
    
    output_size = calculate_pooling_output_size(
        input_size=224,
        kernel_size=2,
        stride=2,
        padding=0
    )
    # Returns: 112

Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torchvision_customizer.layers import validate_pooling_config
    
    config = {
        'pooling_type': 'max',
        'kernel_size': 2,
        'stride': 2,
        'padding': 0
    }
    
    is_valid = validate_pooling_config(config)

Layer Factory
=============

Generic layer creation:

.. code-block:: python

    from torchvision_customizer.layers import get_layer
    
    # Get by type
    layer = get_layer('activation', type='relu')
    layer = get_layer('normalization', type='batch', num_features=64)
    layer = get_layer('pooling', type='max', kernel_size=2)

Complete Layer Usage Example
=============================

.. code-block:: python

    import torch
    import torch.nn as nn
    from torchvision_customizer.layers import (
        get_activation,
        get_normalization,
        get_pooling
    )
    
    class SimpleLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.norm = get_normalization('batch', 64)
            self.activation = get_activation('relu')
            self.pool = get_pooling('max', kernel_size=2, stride=2)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)
            x = self.pool(x)
            return x
    
    model = SimpleLayer()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)  # Output: (1, 64, 112, 112)

Next Steps
==========

- :doc:`../api/blocks` - Learn about composite blocks
- :doc:`../api/models` - Explore model classes
- :doc:`../examples/basic_usage` - See working examples
