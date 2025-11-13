"""Enhanced 2D convolutional layer with advanced options.

Implements a flexible 2D convolution with support for grouped convolutions,
dilated convolutions, and other advanced configurations.

Example:
    >>> from torchvision_customizer.blocks import SuperConv2d
    >>> conv = SuperConv2d(in_channels=64, out_channels=128)
    >>> output = conv(x)
"""

import torch
import torch.nn as nn
from torchvision_customizer.layers import get_activation, get_normalization


class SuperConv2d(nn.Module):
    """Enhanced 2D convolutional layer.

    Provides flexible 2D convolution with support for:
    - Grouped convolutions for depthwise operations
    - Dilated convolutions for receptive field expansion
    - Optional batch normalization
    - Optional activation functions

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of convolutional kernel. Default is 3.
        stride: Stride of convolution. Default is 1.
        padding: Padding for convolution. Default is 1.
        dilation: Dilation rate for convolution. Default is 1.
        groups: Number of groups for grouped convolution. Default is 1.
        activation: Activation function. Default is 'relu'.
        norm_type: Normalization type. Default is 'batch'.
        use_activation: Whether to apply activation. Default is True.
        use_normalization: Whether to apply normalization. Default is True.

    Example:
        >>> # Basic convolution
        >>> conv = SuperConv2d(in_channels=64, out_channels=128)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = conv(x)
        >>> print(output.shape)
        torch.Size([2, 128, 32, 32])

        >>> # Depthwise convolution (groups=in_channels)
        >>> conv = SuperConv2d(
        ...     in_channels=64, out_channels=64,
        ...     groups=64, kernel_size=3
        ... )

        >>> # Dilated convolution
        >>> conv = SuperConv2d(
        ...     in_channels=64, out_channels=128,
        ...     kernel_size=3, dilation=2, padding=2
        ... )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activation: str = 'relu',
        norm_type: str = 'batch',
        use_activation: bool = True,
        use_normalization: bool = True,
    ) -> None:
        """Initialize SuperConv2d."""
        super().__init__()

        self.use_activation = use_activation
        self.use_normalization = use_normalization

        # Convolutional layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=not use_normalization,
        )

        # Normalization
        if use_normalization:
            self.norm = get_normalization(norm_type, out_channels)
        else:
            self.norm = nn.Identity()

        # Activation
        if use_activation:
            self.activation = get_activation(activation)
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply enhanced convolution.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H', W').
        """
        out = self.conv(x)

        if self.use_normalization:
            out = self.norm(out)

        if self.use_activation:
            out = self.activation(out)

        return out
