"""Normalization layer utilities for torchvision-customizer.

This module provides flexible normalization options for building neural networks,
supporting multiple normalization techniques including BatchNorm, GroupNorm,
LayerNorm, InstanceNorm, and custom variants.

Supported Normalization Types:
    - batch: BatchNorm2d (default)
    - group: GroupNorm with configurable groups
    - instance: InstanceNorm2d
    - layer: LayerNorm
    - none: Identity (no normalization)

Example:
    >>> from torchvision_customizer.layers import get_normalization
    >>> norm = get_normalization('batch', num_channels=64)
    >>> norm_group = get_normalization('group', num_channels=64, num_groups=32)
"""

import torch.nn as nn
from typing import Dict, Optional, Type, Any


# Registry of available normalization functions
NORMALIZATION_REGISTRY: Dict[str, Type[nn.Module]] = {
    'batch': nn.BatchNorm2d,
    'group': nn.GroupNorm,
    'instance': nn.InstanceNorm2d,
    'layer': nn.LayerNorm,
    'none': nn.Identity,
}

# Default parameters for each normalization type
NORMALIZATION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    'batch': {},
    'group': {'num_groups': 32},
    'instance': {},
    'layer': {},
    'none': {},
}


def get_normalization(
    norm_type: str,
    num_channels: int,
    **kwargs
) -> nn.Module:
    """Create a normalization layer by type.

    Factory function that returns a configured normalization module based on
    the provided type. Supports multiple normalization techniques with
    automatic parameter configuration.

    Args:
        norm_type: Type of normalization layer. Case-insensitive.
            Supported values: 'batch', 'group', 'instance', 'layer', 'none'
        num_channels: Number of channels for the layer.
        **kwargs: Additional keyword arguments to pass to the normalization
            layer constructor. For GroupNorm, 'num_groups' can be specified.

    Returns:
        An instantiated nn.Module normalization layer.

    Raises:
        ValueError: If the normalization type is not supported.
        TypeError: If invalid keyword arguments are provided.

    Examples:
        >>> # BatchNorm (default)
        >>> norm = get_normalization('batch', num_channels=64)

        >>> # GroupNorm with custom groups
        >>> norm = get_normalization('group', num_channels=64, num_groups=32)

        >>> # InstanceNorm
        >>> norm = get_normalization('instance', num_channels=64)

        >>> # No normalization
        >>> norm = get_normalization('none', num_channels=64)
    """
    normalized_type = norm_type.lower().strip()

    if normalized_type not in NORMALIZATION_REGISTRY:
        supported = ', '.join(sorted(NORMALIZATION_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported normalization type: '{norm_type}'\n"
            f"Supported types: {supported}"
        )

    norm_class = NORMALIZATION_REGISTRY[normalized_type]

    # Get default parameters
    default_params = NORMALIZATION_DEFAULTS[normalized_type].copy()

    # Handle special cases
    if normalized_type == 'group':
        default_params['num_channels'] = num_channels
    elif normalized_type in ['batch', 'instance']:
        default_params['num_features'] = num_channels
    elif normalized_type == 'layer':
        # LayerNorm expects normalized_shape, not num_channels
        default_params['normalized_shape'] = (num_channels,)
    elif normalized_type == 'none':
        # Identity doesn't take any arguments
        return nn.Identity()

    # Override with provided keyword arguments
    default_params.update(kwargs)

    try:
        return norm_class(**default_params)
    except TypeError as e:
        raise TypeError(
            f"Invalid parameters for {normalized_type}: {str(e)}"
        ) from e


def is_normalization_supported(norm_type: str) -> bool:
    """Check if a normalization type is supported.

    Args:
        norm_type: Type of normalization to check.

    Returns:
        True if the normalization type is supported, False otherwise.

    Example:
        >>> is_normalization_supported('batch')
        True
        >>> is_normalization_supported('unsupported')
        False
    """
    return norm_type.lower().strip() in NORMALIZATION_REGISTRY


def get_supported_normalizations() -> list[str]:
    """Get list of all supported normalization types.

    Returns:
        A sorted list of supported normalization type names.

    Example:
        >>> normalizations = get_supported_normalizations()
        >>> print(normalizations)
        ['batch', 'group', 'instance', 'layer', 'none']
    """
    return sorted(NORMALIZATION_REGISTRY.keys())


class NormalizationFactory:
    """Factory class for creating and managing normalization layers.

    Provides a stateful interface for creating normalization layers with
    configuration management.

    Example:
        >>> factory = NormalizationFactory()
        >>> norm = factory.create('batch', num_channels=64)
        >>> group_norm = factory.create('group', num_channels=64, num_groups=16)
    """

    @staticmethod
    def create(norm_type: str, num_channels: int, **kwargs) -> nn.Module:
        """Create a normalization layer.

        Args:
            norm_type: Type of normalization layer.
            num_channels: Number of channels.
            **kwargs: Additional keyword arguments for the normalization.

        Returns:
            The created normalization layer.

        Raises:
            ValueError: If normalization type is not supported.
        """
        return get_normalization(norm_type, num_channels, **kwargs)

    @staticmethod
    def is_supported(norm_type: str) -> bool:
        """Check if a normalization type is supported.

        Args:
            norm_type: Type of normalization to check.

        Returns:
            True if supported, False otherwise.
        """
        return is_normalization_supported(norm_type)

    @staticmethod
    def supported_normalizations() -> list[str]:
        """Get list of supported normalization types.

        Returns:
            List of supported normalization type names.
        """
        return get_supported_normalizations()

    @staticmethod
    def get_defaults(norm_type: str) -> Dict[str, Any]:
        """Get default parameters for a normalization type.

        Args:
            norm_type: Type of normalization.

        Returns:
            Dictionary of default parameters.

        Raises:
            ValueError: If normalization type is not supported.
        """
        normalized_type = norm_type.lower().strip()
        if normalized_type not in NORMALIZATION_REGISTRY:
            supported = ', '.join(sorted(NORMALIZATION_REGISTRY.keys()))
            raise ValueError(
                f"Unsupported normalization type: '{norm_type}'\n"
                f"Supported types: {supported}"
            )
        return NORMALIZATION_DEFAULTS[normalized_type].copy()
