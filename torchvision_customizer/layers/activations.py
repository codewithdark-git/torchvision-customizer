"""Activation function utilities for torchvision-customizer.

This module provides a flexible interface for working with various activation
functions in PyTorch. It includes a registry-based system for managing different
activation types and a factory function for creating activation modules.

Supported Activation Functions:
    - relu: Rectified Linear Unit
    - leaky_relu: Leaky ReLU with configurable negative slope
    - prelu: Parametric ReLU with learnable negative slope
    - gelu: Gaussian Error Linear Unit
    - silu: Sigmoid Linear Unit (Swish)
    - sigmoid: Sigmoid activation
    - tanh: Hyperbolic tangent
    - elu: Exponential Linear Unit
    - selu: Scaled Exponential Linear Unit

Example:
    >>> import torch
    >>> from torchvision_customizer.layers import get_activation
    >>>
    >>> # Create a ReLU activation
    >>> relu = get_activation('relu')
    >>> x = torch.randn(2, 64, 32, 32)
    >>> output = relu(x)
    >>>
    >>> # Create a Leaky ReLU with custom slope
    >>> leaky_relu = get_activation('leaky_relu', negative_slope=0.2)
    >>> output = leaky_relu(x)
    >>>
    >>> # Case-insensitive
    >>> gelu = get_activation('GELU')
    >>> output = gelu(x)
"""

import torch.nn as nn
from typing import Dict, Optional, Type, Any


# Registry of available activation functions
ACTIVATION_REGISTRY: Dict[str, Type[nn.Module]] = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'prelu': nn.PReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'elu': nn.ELU,
    'selu': nn.SELU,
}

# Default parameters for each activation function
ACTIVATION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    'relu': {},
    'leaky_relu': {'negative_slope': 0.01},
    'prelu': {'num_parameters': 1},
    'gelu': {},
    'silu': {},
    'sigmoid': {},
    'tanh': {},
    'elu': {'alpha': 1.0},
    'selu': {},
}


def get_activation(
    name: str,
    **kwargs
) -> nn.Module:
    """Create an activation function module by name.
    
    Factory function that returns a configured activation module based on the
    provided name. Supports case-insensitive activation names and optional
    keyword arguments for fine-tuning activation parameters.
    
    Args:
        name: Name of the activation function. Case-insensitive.
            Supported values: 'relu', 'leaky_relu', 'prelu', 'gelu', 'silu',
            'sigmoid', 'tanh', 'elu', 'selu'
        **kwargs: Additional keyword arguments to pass to the activation
            function constructor. These override default parameters.
            
    Returns:
        An instantiated nn.Module activation function.
        
    Raises:
        ValueError: If the activation function name is not supported.
        TypeError: If invalid keyword arguments are provided for the
            activation function.
            
    Examples:
        >>> # Basic usage with default parameters
        >>> relu = get_activation('relu')
        >>> leaky_relu = get_activation('leaky_relu')
        
        >>> # Custom parameters
        >>> leaky_relu = get_activation('leaky_relu', negative_slope=0.2)
        >>> elu = get_activation('elu', alpha=0.5)
        
        >>> # Case-insensitive
        >>> gelu = get_activation('GELU')
        >>> silu = get_activation('SiLU')
        
        >>> # Use in a model
        >>> import torch
        >>> model = nn.Sequential(
        ...     nn.Conv2d(3, 64, 3),
        ...     nn.BatchNorm2d(64),
        ...     get_activation('relu')
        ... )
        >>> x = torch.randn(2, 3, 32, 32)
        >>> output = model(x)
    """
    # Normalize name to lowercase for case-insensitive lookup
    normalized_name = name.lower().strip()
    
    # Check if activation is supported
    if normalized_name not in ACTIVATION_REGISTRY:
        supported = ', '.join(sorted(ACTIVATION_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported activation function: '{name}'\n"
            f"Supported activations: {supported}"
        )
    
    # Get the activation class
    activation_class = ACTIVATION_REGISTRY[normalized_name]
    
    # Get default parameters
    default_params = ACTIVATION_DEFAULTS[normalized_name].copy()
    
    # Override with provided keyword arguments
    default_params.update(kwargs)
    
    try:
        # Create and return the activation module
        return activation_class(**default_params)
    except TypeError as e:
        raise TypeError(
            f"Invalid parameters for {normalized_name}: {str(e)}\n"
            f"Default parameters: {ACTIVATION_DEFAULTS[normalized_name]}"
        ) from e


def is_activation_supported(name: str) -> bool:
    """Check if an activation function is supported.
    
    Args:
        name: Name of the activation function to check.
        
    Returns:
        True if the activation function is supported, False otherwise.
        
    Example:
        >>> is_activation_supported('relu')
        True
        >>> is_activation_supported('unsupported_activation')
        False
    """
    return name.lower().strip() in ACTIVATION_REGISTRY


def get_supported_activations() -> list[str]:
    """Get list of all supported activation functions.
    
    Returns:
        A sorted list of supported activation function names.
        
    Example:
        >>> activations = get_supported_activations()
        >>> print(activations)
        ['elu', 'gelu', 'leaky_relu', 'prelu', 'relu', 'selu', 'sigmoid', 'silu', 'tanh']
    """
    return sorted(ACTIVATION_REGISTRY.keys())


# Alias for backward compatibility
get_supported_activations.__doc__ = """Get list of all supported activation functions.
    
    Returns:
        A sorted list of supported activation function names.
    """


class ActivationFactory:
    """Factory class for creating and managing activation functions.
    
    Provides a stateful interface for creating activation functions with
    configuration management.
    
    Example:
        >>> factory = ActivationFactory()
        >>> relu = factory.create('relu')
        >>> gelu = factory.create('gelu')
        >>> supported = factory.supported_activations()
    """
    
    @staticmethod
    def create(name: str, **kwargs) -> nn.Module:
        """Create an activation function module.
        
        Args:
            name: Name of the activation function.
            **kwargs: Additional keyword arguments for the activation.
            
        Returns:
            The created activation module.
            
        Raises:
            ValueError: If activation name is not supported.
        """
        return get_activation(name, **kwargs)
    
    @staticmethod
    def is_supported(name: str) -> bool:
        """Check if an activation function is supported.
        
        Args:
            name: Name of the activation function.
            
        Returns:
            True if supported, False otherwise.
        """
        return is_activation_supported(name)
    
    @staticmethod
    def supported_activations() -> list[str]:
        """Get list of supported activation functions.
        
        Returns:
            List of supported activation names.
        """
        return get_supported_activations()
    
    @staticmethod
    def get_defaults(name: str) -> Dict[str, Any]:
        """Get default parameters for an activation function.
        
        Args:
            name: Name of the activation function.
            
        Returns:
            Dictionary of default parameters.
            
        Raises:
            ValueError: If activation name is not supported.
        """
        normalized_name = name.lower().strip()
        if normalized_name not in ACTIVATION_REGISTRY:
            supported = ', '.join(sorted(ACTIVATION_REGISTRY.keys()))
            raise ValueError(
                f"Unsupported activation function: '{name}'\n"
                f"Supported activations: {supported}"
            )
        return ACTIVATION_DEFAULTS[normalized_name].copy()
