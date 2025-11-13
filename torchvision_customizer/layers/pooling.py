"""Pooling layer utilities for torchvision-customizer.

This module provides flexible pooling options for building neural networks,
supporting multiple pooling techniques including MaxPool, AvgPool, and
Adaptive variants.

Supported Pooling Types:
    - max: MaxPool2d
    - avg: AvgPool2d
    - adaptive_max: AdaptiveMaxPool2d
    - adaptive_avg: AdaptiveAvgPool2d
    - none: Identity (no pooling)

Example:
    >>> from torchvision_customizer.layers import get_pooling
    >>> pool = get_pooling('max', kernel_size=2, stride=2)
    >>> adaptive_pool = get_pooling('adaptive_avg', output_size=(1, 1))
"""

import torch.nn as nn
from typing import Dict, Optional, Type, Any, Tuple, Union


# Registry of available pooling functions
POOLING_REGISTRY: Dict[str, Type[nn.Module]] = {
    'max': nn.MaxPool2d,
    'avg': nn.AvgPool2d,
    'adaptive_max': nn.AdaptiveMaxPool2d,
    'adaptive_avg': nn.AdaptiveAvgPool2d,
    'none': nn.Identity,
}

# Default parameters for each pooling type
POOLING_DEFAULTS: Dict[str, Dict[str, Any]] = {
    'max': {'kernel_size': 2, 'stride': 2},
    'avg': {'kernel_size': 2, 'stride': 2},
    'adaptive_max': {'output_size': (1, 1)},
    'adaptive_avg': {'output_size': (1, 1)},
    'none': {},
}


def get_pooling(
    pool_type: str,
    kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    output_size: Optional[Union[int, Tuple[int, int]]] = None,
    **kwargs
) -> nn.Module:
    """Create a pooling layer by type.

    Factory function that returns a configured pooling module based on
    the provided type. Supports multiple pooling techniques with
    automatic parameter configuration.

    Args:
        pool_type: Type of pooling layer. Case-insensitive.
            Supported values: 'max', 'avg', 'adaptive_max', 'adaptive_avg', 'none'
        kernel_size: Kernel size for non-adaptive pooling. Default is 2.
        stride: Stride for non-adaptive pooling. Default is same as kernel_size.
        output_size: Output size for adaptive pooling. Default is (1, 1).
        **kwargs: Additional keyword arguments for the pooling layer.

    Returns:
        An instantiated nn.Module pooling layer.

    Raises:
        ValueError: If the pooling type is not supported.
        TypeError: If invalid keyword arguments are provided.

    Examples:
        >>> # MaxPool with default parameters
        >>> pool = get_pooling('max')

        >>> # AvgPool with custom kernel size
        >>> pool = get_pooling('avg', kernel_size=3, stride=2)

        >>> # AdaptiveMaxPool for global pooling
        >>> pool = get_pooling('adaptive_max', output_size=(1, 1))

        >>> # No pooling
        >>> pool = get_pooling('none')
    """
    normalized_type = pool_type.lower().strip()

    if normalized_type not in POOLING_REGISTRY:
        supported = ', '.join(sorted(POOLING_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported pooling type: '{pool_type}'\n"
            f"Supported types: {supported}"
        )

    pool_class = POOLING_REGISTRY[normalized_type]

    # Get default parameters
    default_params = POOLING_DEFAULTS[normalized_type].copy()

    # Override with provided parameters
    if kernel_size is not None and normalized_type in ['max', 'avg']:
        default_params['kernel_size'] = kernel_size
    if stride is not None and normalized_type in ['max', 'avg']:
        default_params['stride'] = stride
    if output_size is not None and normalized_type in ['adaptive_max', 'adaptive_avg']:
        default_params['output_size'] = output_size

    # Override with additional keyword arguments
    default_params.update(kwargs)

    if normalized_type == 'none':
        return nn.Identity()

    try:
        return pool_class(**default_params)
    except TypeError as e:
        raise TypeError(
            f"Invalid parameters for {normalized_type}: {str(e)}"
        ) from e


def is_pooling_supported(pool_type: str) -> bool:
    """Check if a pooling type is supported.

    Args:
        pool_type: Type of pooling to check.

    Returns:
        True if the pooling type is supported, False otherwise.

    Example:
        >>> is_pooling_supported('max')
        True
        >>> is_pooling_supported('unsupported')
        False
    """
    return pool_type.lower().strip() in POOLING_REGISTRY


def get_supported_pooling() -> list[str]:
    """Get list of all supported pooling types.

    Returns:
        A sorted list of supported pooling type names.

    Example:
        >>> pooling = get_supported_pooling()
        >>> print(pooling)
        ['adaptive_avg', 'adaptive_max', 'avg', 'max', 'none']
    """
    return sorted(POOLING_REGISTRY.keys())


class PoolingFactory:
    """Factory class for creating and managing pooling layers.

    Provides a stateful interface for creating pooling layers with
    configuration management.

    Example:
        >>> factory = PoolingFactory()
        >>> pool = factory.create('max', kernel_size=2, stride=2)
        >>> adaptive_pool = factory.create('adaptive_avg', output_size=(1, 1))
    """

    @staticmethod
    def create(
        pool_type: str,
        kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        output_size: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs
    ) -> nn.Module:
        """Create a pooling layer.

        Args:
            pool_type: Type of pooling layer.
            kernel_size: Kernel size for non-adaptive pooling.
            stride: Stride for non-adaptive pooling.
            output_size: Output size for adaptive pooling.
            **kwargs: Additional keyword arguments for the pooling.

        Returns:
            The created pooling layer.

        Raises:
            ValueError: If pooling type is not supported.
        """
        return get_pooling(pool_type, kernel_size, stride, output_size, **kwargs)

    @staticmethod
    def is_supported(pool_type: str) -> bool:
        """Check if a pooling type is supported.

        Args:
            pool_type: Type of pooling to check.

        Returns:
            True if supported, False otherwise.
        """
        return is_pooling_supported(pool_type)

    @staticmethod
    def supported_pooling() -> list[str]:
        """Get list of supported pooling types.

        Returns:
            List of supported pooling type names.
        """
        return get_supported_pooling()

    @staticmethod
    def get_defaults(pool_type: str) -> Dict[str, Any]:
        """Get default parameters for a pooling type.

        Args:
            pool_type: Type of pooling.

        Returns:
            Dictionary of default parameters.

        Raises:
            ValueError: If pooling type is not supported.
        """
        normalized_type = pool_type.lower().strip()
        if normalized_type not in POOLING_REGISTRY:
            supported = ', '.join(sorted(POOLING_REGISTRY.keys()))
            raise ValueError(
                f"Unsupported pooling type: '{pool_type}'\n"
                f"Supported types: {supported}"
            )
        return POOLING_DEFAULTS[normalized_type].copy()
