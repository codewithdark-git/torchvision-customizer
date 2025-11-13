"""Depthwise separable convolution block.

Implements depthwise separable convolutions for efficient feature extraction.
Separates spatial and channel-wise convolutions for parameter efficiency.

Example:
    >>> from torchvision_customizer.blocks import DepthwiseBlock
    >>> block = DepthwiseBlock(in_channels=64, out_channels=128)
    >>> output = block(x)
"""

import torch
import torch.nn as nn
from torchvision_customizer.layers import get_activation, get_normalization


class DepthwiseBlock(nn.Module):
    """Depthwise separable convolution block.

    Implements depthwise separable convolutions combining:
    1. Depthwise convolution (per-channel spatial convolution)
    2. Pointwise convolution (1x1 convolution for channel mixing)

    This decomposition reduces parameters while maintaining expressiveness.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for depthwise convolution. Default is 3.
        stride: Stride for convolution. Default is 1.
        padding: Padding for convolution. Default is 1.
        activation: Activation function. Default is 'relu'.
        norm_type: Normalization type. Default is 'batch'.
        add_residual: Whether to add residual connection. Default is False.

    Example:
        >>> # Basic depthwise block
        >>> block = DepthwiseBlock(in_channels=64, out_channels=128)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([2, 128, 32, 32])

        >>> # With residual connection (same channels)
        >>> block = DepthwiseBlock(
        ...     in_channels=64, out_channels=64,
        ...     kernel_size=5, add_residual=True
        ... )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = 'relu',
        norm_type: str = 'batch',
        add_residual: bool = False,
    ) -> None:
        """Initialize DepthwiseBlock."""
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_residual = add_residual and (in_channels == out_channels)
        self.activation_fn = get_activation(activation)

        # Depthwise convolution (groups=in_channels)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = get_normalization(norm_type, in_channels)

        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = get_normalization(norm_type, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise separable convolution.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H', W').
        """
        identity = x

        # Depthwise convolution
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.activation_fn(out)

        # Pointwise convolution
        out = self.pointwise(out)
        out = self.bn2(out)

        # Residual connection
        if self.add_residual:
            out = out + identity

        out = self.activation_fn(out)

        return out
