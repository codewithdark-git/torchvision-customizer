"""Residual block with skip connections.

Implements residual learning as in ResNet. Supports both basic and
bottleneck architectures with optional channel dimension matching.

Example:
    >>> from torchvision_customizer.blocks import ResidualBlock
    >>> res_block = ResidualBlock(in_channels=64, out_channels=64)
    >>> output = res_block(x)
"""

import torch
import torch.nn as nn
from torchvision_customizer.layers import get_activation, get_normalization
from torchvision_customizer.blocks.se_block import SEBlock


class ResidualBlock(nn.Module):
    """Residual block with skip connections.

    Implements residual learning as in ResNet. Supports both basic and
    bottleneck architectures with optional SE blocks for channel attention
    and flexible normalization options.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for convolution. Default is 1.
        downsample: Whether to downsample. Default is False.
        activation: Activation function. Default is 'relu'.
        norm_type: Normalization type. Default is 'batch'.
        use_se: Whether to use SE block. Default is False.
        se_reduction: SE reduction ratio. Default is 16.
        bottleneck: Whether to use bottleneck architecture. Default is False.

    Example:
        >>> # Basic residual block
        >>> res_block = ResidualBlock(in_channels=64, out_channels=64)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = res_block(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])

        >>> # With SE block and downsample
        >>> res_block = ResidualBlock(
        ...     in_channels=64, out_channels=128,
        ...     stride=2, downsample=True, use_se=True
        ... )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: bool = False,
        activation: str = 'relu',
        norm_type: str = 'batch',
        use_se: bool = False,
        se_reduction: int = 16,
        bottleneck: bool = False,
    ) -> None:
        """Initialize ResidualBlock."""
        super().__init__()

        self.stride = stride
        self.downsample_layer = None
        self.activation_fn = get_activation(activation)
        self.use_se = use_se

        if bottleneck:
            # 1x1 -> 3x3 -> 1x1 pattern
            bottleneck_channels = max(1, out_channels // 4)

            self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            self.bn1 = get_normalization(norm_type, bottleneck_channels)

            self.conv2 = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.bn2 = get_normalization(norm_type, bottleneck_channels)

            self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
            self.bn3 = get_normalization(norm_type, out_channels)
        else:
            # 3x3 -> 3x3 pattern
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.bn1 = get_normalization(norm_type, out_channels)

            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = get_normalization(norm_type, out_channels)

        # Dimension matching
        if stride != 1 or in_channels != out_channels or downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_normalization(norm_type, out_channels),
            )

        # SE block
        if use_se:
            self.se_block = SEBlock(out_channels, reduction=se_reduction, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C', H', W').
        """
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, 'conv3'):  # Bottleneck
            out = self.activation_fn(out)
            out = self.conv3(out)
            out = self.bn3(out)

        # Apply SE block if enabled
        if self.use_se:
            out = self.se_block(out)

        # Skip connection
        if self.downsample_layer is not None:
            identity = self.downsample_layer(x)

        out = out + identity
        out = self.activation_fn(out)

        return out
