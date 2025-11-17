"""Layers module: Custom layers (activations, normalizations, pooling).

This module provides utility functions and classes for working with different
layer types in neural networks, including activation functions, normalization
techniques, and pooling operations.

Submodules:
    - activations: Activation function factory and utilities
    - normalizations: Normalization layer factory and utilities
    - pooling: Pooling layer factory and utilities

Example:
    >>> from torchvision_customizer.layers import (
    ...     get_activation,
    ...     get_normalization,
    ...     get_pooling
    ... )
    >>> relu = get_activation('relu')
    >>> batch_norm = get_normalization('batch', num_channels=64)
    >>> max_pool = get_pooling('max', kernel_size=2)
"""

from .activations import (
    get_activation,
    is_activation_supported,
    get_supported_activations,
    ActivationFactory,
    ACTIVATION_REGISTRY,
    ACTIVATION_DEFAULTS,
)

from .normalizations import (
    get_normalization,
    is_normalization_supported,
    get_supported_normalizations,
    NormalizationFactory,
    NORMALIZATION_REGISTRY,
    NORMALIZATION_DEFAULTS,
)

from .pooling import (
    get_pooling,
    is_pooling_supported,
    get_supported_pooling,
    PoolingFactory,
    PoolingBlock,
    StochasticPool2d,
    LPPool2d,
    calculate_pooling_output_size,
    validate_pooling_config,
    POOLING_REGISTRY,
    POOLING_DEFAULTS,
)

from .attention import (
    ChannelAttention,
    SpatialAttention,
    ChannelSpatialAttention,
    MultiHeadAttention,
    PositionalEncoding,
    AttentionBlock,
    create_attention_map,
    apply_attention,
    normalize_attention,
)

__all__ = [
    # Activation utilities
    'get_activation',
    'is_activation_supported',
    'get_supported_activations',
    'ActivationFactory',
    'ACTIVATION_REGISTRY',
    'ACTIVATION_DEFAULTS',
    
    # Normalization utilities
    'get_normalization',
    'is_normalization_supported',
    'get_supported_normalizations',
    'NormalizationFactory',
    'NORMALIZATION_REGISTRY',
    'NORMALIZATION_DEFAULTS',
    
    # Pooling utilities
    'get_pooling',
    'is_pooling_supported',
    'get_supported_pooling',
    'PoolingFactory',
    'PoolingBlock',
    'StochasticPool2d',
    'LPPool2d',
    'calculate_pooling_output_size',
    'validate_pooling_config',
    'POOLING_REGISTRY',
    'POOLING_DEFAULTS',
    
    # Step 6: Attention Mechanisms
    'ChannelAttention',
    'SpatialAttention',
    'ChannelSpatialAttention',
    'MultiHeadAttention',
    'PositionalEncoding',
    'AttentionBlock',
    'create_attention_map',
    'apply_attention',
    'normalize_attention',
]

