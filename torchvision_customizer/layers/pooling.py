"""Pooling layer utilities for torchvision-customizer.

This module provides flexible pooling options for building neural networks,
supporting multiple pooling techniques including MaxPool, AvgPool, and
Adaptive variants.

Supported Pooling Types:
    - max: MaxPool2d
    - avg: AvgPool2d
    - adaptive_max: AdaptiveMaxPool2d
    - adaptive_avg: AdaptiveAvgPool2d
    - stochastic: StochasticPool2d (random sampling during training)
    - none: Identity (no pooling)

Example:
    >>> from torchvision_customizer.layers import get_pooling, PoolingBlock
    >>> pool = get_pooling('max', kernel_size=2, stride=2)
    >>> block = PoolingBlock('avg', kernel_size=3, dropout_rate=0.1)
    >>> adaptive_pool = get_pooling('adaptive_avg', output_size=(1, 1))
"""

import torch
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


class PoolingBlock(nn.Module):
    """Flexible pooling block with optional dropout.

    Combines pooling operation with optional dropout for regularization.
    Simplifies integration of pooling into network architectures.

    Args:
        pool_type: Type of pooling layer. Default is 'max'.
        kernel_size: Kernel size for pooling. Default is 2.
        stride: Stride for pooling. Default is kernel_size.
        padding: Padding for pooling. Default is 0.
        dropout_rate: Dropout probability after pooling. Default is 0.0 (no dropout).
        output_size: Output size for adaptive pooling. Default is (1, 1).

    Example:
        >>> block = PoolingBlock('max', kernel_size=2, dropout_rate=0.1)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([2, 64, 16, 16])
    """

    def __init__(
        self,
        pool_type: str = 'max',
        kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: int = 0,
        dropout_rate: float = 0.0,
        output_size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> None:
        """Initialize PoolingBlock."""
        super().__init__()

        self.pool_type = pool_type.lower().strip()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.output_size = output_size

        # Create pooling layer with defaults
        if kernel_size is None and self.pool_type in ['max', 'avg']:
            kernel_size = 2
        if stride is None and self.pool_type in ['max', 'avg']:
            stride = kernel_size
        if output_size is None and self.pool_type in ['adaptive_max', 'adaptive_avg']:
            output_size = (1, 1)

        # Build parameters
        pool_kwargs = {}
        if self.pool_type in ['max', 'avg']:
            pool_kwargs['kernel_size'] = kernel_size
            pool_kwargs['stride'] = stride
            pool_kwargs['padding'] = padding
        elif self.pool_type in ['adaptive_max', 'adaptive_avg']:
            pool_kwargs['output_size'] = output_size

        # Create pooling layer
        self.pool = get_pooling(self.pool_type, **pool_kwargs)

        # Optional dropout
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=dropout_rate)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pooling and optional dropout.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H', W').
        """
        out = self.pool(x)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

    def calculate_output_size(
        self, input_height: int, input_width: int
    ) -> Tuple[int, int]:
        """Calculate output spatial dimensions.

        Args:
            input_height: Input height.
            input_width: Input width.

        Returns:
            Tuple of (output_height, output_width).
        """
        if self.pool_type in ['adaptive_max', 'adaptive_avg']:
            # Adaptive pooling returns fixed size
            if isinstance(self.output_size, int):
                return (self.output_size, self.output_size)
            else:
                return self.output_size

        # Standard pooling formula
        kernel_size = self.kernel_size if self.kernel_size is not None else 2
        stride = self.stride if self.stride is not None else kernel_size

        h_out = (input_height + 2 * self.padding - kernel_size) // stride + 1
        w_out = (input_width + 2 * self.padding - kernel_size) // stride + 1

        return (h_out, w_out)


class StochasticPool2d(nn.Module):
    """Stochastic pooling layer.

    During training, randomly picks values from the pooling region according to
    a multinomial distribution weighted by activation values. During evaluation,
    uses max pooling for stability.

    Args:
        kernel_size: Size of the pooling kernel. Default is 2.
        stride: Stride of the pooling. Default is kernel_size.
        padding: Padding. Default is 0.

    Reference:
        "Stochastic Pooling for Regularization of Deep Convolutional Neural Networks"
        https://arxiv.org/abs/1301.3557

    Example:
        >>> pool = StochasticPool2d(kernel_size=2)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = pool(x)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: int = 0,
    ) -> None:
        """Initialize StochasticPool2d."""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        
        # For evaluation, use max pooling
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=self.stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic or max pooling based on training mode.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Pooled tensor.
        """
        if self.training:
            return self._stochastic_pool(x)
        else:
            return self.max_pool(x)

    def _stochastic_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Perform stochastic pooling operation."""
        batch_size, channels, height, width = x.shape

        # Normalize kernel_size and stride to tuples
        if isinstance(self.kernel_size, int):
            k_h, k_w = self.kernel_size, self.kernel_size
        else:
            k_h, k_w = self.kernel_size

        if isinstance(self.stride, int):
            s_h, s_w = self.stride, self.stride
        else:
            s_h, s_w = self.stride

        # Calculate output dimensions
        out_h = (height + 2 * self.padding - k_h) // s_h + 1
        out_w = (width + 2 * self.padding - k_w) // s_w + 1

        output = torch.zeros(
            batch_size, channels, out_h, out_w, device=x.device, dtype=x.dtype
        )

        # Apply stochastic pooling
        for i in range(out_h):
            for j in range(out_w):
                h_start = max(0, i * s_h - self.padding)
                h_end = min(height, h_start + k_h)
                w_start = max(0, j * s_w - self.padding)
                w_end = min(width, w_start + k_w)

                pool_region = x[:, :, h_start:h_end, w_start:w_end]
                batch, chans, pool_h, pool_w = pool_region.shape
                pool_region = pool_region.reshape(batch, chans, -1)

                # Create probability distribution from values
                # Use softmax to weight by activation magnitude
                probs = torch.softmax(pool_region, dim=2)

                # Sample indices according to probabilities
                indices = torch.multinomial(
                    probs.reshape(batch * chans, -1), num_samples=1
                )
                indices = indices.view(batch, chans)

                # Gather values
                for b in range(batch):
                    for c in range(chans):
                        idx = indices[b, c].item()
                        output[b, c, i, j] = pool_region[b, c, idx]

        return output


class LPPool2d(nn.Module):
    """L-p norm pooling layer.

    Computes the L-p norm over pooling regions. When p=2, becomes RMS pooling.
    When p=1, becomes average pooling.

    Args:
        norm_type: Norm order (typically 2 for RMS). Default is 2.
        kernel_size: Size of pooling kernel. Default is 2.
        stride: Stride. Default is kernel_size.

    Example:
        >>> pool = LPPool2d(norm_type=2, kernel_size=2)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = pool(x)
    """

    def __init__(
        self,
        norm_type: float = 2.0,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> None:
        """Initialize LPPool2d."""
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

        # Use built-in LPPool2d (note: no padding parameter)
        self.pool = nn.LPPool2d(
            norm_type=norm_type,
            kernel_size=kernel_size,
            stride=self.stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply L-p norm pooling.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Pooled tensor.
        """
        return self.pool(x)


def calculate_pooling_output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """Calculate output size after pooling.

    Args:
        input_size: Input spatial dimension.
        kernel_size: Pooling kernel size.
        stride: Pooling stride.
        padding: Pooling padding. Default is 0.
        dilation: Pooling dilation. Default is 1.

    Returns:
        Output spatial dimension.

    Formula:
        output_size = floor((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    """
    return (
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    )


def validate_pooling_config(
    pool_type: str,
    kernel_size: int,
    stride: int,
    input_height: int,
    input_width: int,
) -> Tuple[bool, str]:
    """Validate pooling configuration.

    Checks if pooling configuration will produce valid output dimensions.

    Args:
        pool_type: Type of pooling ('max', 'avg', etc.).
        kernel_size: Kernel size.
        stride: Stride.
        input_height: Input height.
        input_width: Input width.

    Returns:
        Tuple of (is_valid, message).

    Example:
        >>> is_valid, msg = validate_pooling_config('max', 2, 2, 32, 32)
        >>> print(is_valid, msg)
        (True, 'Valid')
    """
    # Check kernel size
    if kernel_size <= 0:
        return False, "Kernel size must be positive"

    if kernel_size > input_height or kernel_size > input_width:
        return False, (
            f"Kernel size ({kernel_size}) larger than input "
            f"({input_height}x{input_width})"
        )

    # Check stride
    if stride <= 0:
        return False, "Stride must be positive"

    # Calculate output size
    out_h = calculate_pooling_output_size(input_height, kernel_size, stride)
    out_w = calculate_pooling_output_size(input_width, kernel_size, stride)

    if out_h <= 0 or out_w <= 0:
        return False, f"Output size would be invalid: {out_h}x{out_w}"

    if out_h < 1 or out_w < 1:
        return False, "Output spatial dimensions too small"

    return True, "Valid"
