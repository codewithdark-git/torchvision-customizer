"""Bottleneck blocks with advanced residual patterns.

Implements various bottleneck architectures for building efficient deep networks:
- Standard bottleneck (1x1 -> 3x3 -> 1x1)
- Wide bottleneck (wider mid-channels)
- Grouped convolution bottleneck
- Multi-scale bottleneck
- Asymmetric bottleneck

Example:
    >>> from torchvision_customizer.blocks import StandardBottleneck, WideBott leneck
    >>> bottleneck = StandardBottleneck(in_channels=256, out_channels=256)
    >>> output = bottleneck(x)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from torchvision_customizer.layers import get_activation, get_normalization


class StandardBottleneck(nn.Module):
    """Standard bottleneck block (1x1 -> 3x3 -> 1x1).

    Reduces channel dimension with 1x1 conv, applies 3x3 conv,
    then expands back to output channels.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the 3x3 convolution. Default is 1.
        expansion: Expansion ratio for bottleneck. Default is 4.
        activation: Activation function. Default is 'relu'.
        norm_type: Normalization type. Default is 'batch'.
        use_downsample: Whether to use downsample shortcut. Default is False.

    Example:
        >>> block = StandardBottleneck(in_channels=256, out_channels=256)
        >>> x = torch.randn(2, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        activation: str = 'relu',
        norm_type: str = 'batch',
        use_downsample: bool = False,
    ) -> None:
        """Initialize StandardBottleneck."""
        super().__init__()

        self.expansion = expansion
        self.activation_fn = get_activation(activation)
        bottleneck_channels = max(1, out_channels // expansion)

        # 1x1 reduce
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = get_normalization(norm_type, bottleneck_channels)

        # 3x3 main
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = get_normalization(norm_type, bottleneck_channels)

        # 1x1 expand
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = get_normalization(norm_type, out_channels)

        # Downsample if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels or use_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_normalization(norm_type, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck block."""
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation_fn(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.activation_fn(out)

        return out


class WideBottleneck(nn.Module):
    """Wide bottleneck block with expanded mid channels.

    Uses wider intermediate channels to increase model capacity
    while maintaining parameter efficiency.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        width_multiplier: Multiplier for bottleneck channels. Default is 1.0.
        stride: Stride for the 3x3 convolution. Default is 1.
        expansion: Base expansion ratio. Default is 4.
        activation: Activation function. Default is 'relu'.
        norm_type: Normalization type. Default is 'batch'.
        use_downsample: Whether to use downsample shortcut. Default is False.

    Example:
        >>> block = WideBottleneck(
        ...     in_channels=256, out_channels=256, width_multiplier=1.5
        ... )
        >>> x = torch.randn(2, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width_multiplier: float = 1.0,
        stride: int = 1,
        expansion: int = 4,
        activation: str = 'relu',
        norm_type: str = 'batch',
        use_downsample: bool = False,
    ) -> None:
        """Initialize WideBottleneck."""
        super().__init__()

        self.expansion = expansion
        self.activation_fn = get_activation(activation)
        bottleneck_channels = max(1, int(out_channels / expansion * width_multiplier))

        # 1x1 reduce
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = get_normalization(norm_type, bottleneck_channels)

        # 3x3 main
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = get_normalization(norm_type, bottleneck_channels)

        # 1x1 expand
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = get_normalization(norm_type, out_channels)

        # Downsample if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels or use_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_normalization(norm_type, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply wide bottleneck block."""
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation_fn(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.activation_fn(out)

        return out


class GroupedBottleneck(nn.Module):
    """Bottleneck with grouped convolutions (ShuffleNet style).

    Uses grouped convolutions to improve efficiency while
    maintaining model capacity.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_groups: Number of groups for grouped conv. Default is 2.
        stride: Stride for the 3x3 convolution. Default is 1.
        expansion: Expansion ratio. Default is 4.
        activation: Activation function. Default is 'relu'.
        norm_type: Normalization type. Default is 'batch'.
        use_downsample: Whether to use downsample shortcut. Default is False.

    Example:
        >>> block = GroupedBottleneck(
        ...     in_channels=256, out_channels=256, num_groups=4
        ... )
        >>> x = torch.randn(2, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 2,
        stride: int = 1,
        expansion: int = 4,
        activation: str = 'relu',
        norm_type: str = 'batch',
        use_downsample: bool = False,
    ) -> None:
        """Initialize GroupedBottleneck."""
        super().__init__()

        self.expansion = expansion
        self.num_groups = num_groups
        self.activation_fn = get_activation(activation)
        bottleneck_channels = max(1, out_channels // expansion)

        # Ensure grouped conv is valid
        bottleneck_channels = (bottleneck_channels // num_groups) * num_groups
        if bottleneck_channels == 0:
            bottleneck_channels = num_groups

        # 1x1 reduce
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = get_normalization(norm_type, bottleneck_channels)

        # 3x3 grouped convolution
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=min(num_groups, bottleneck_channels),
            bias=False,
        )
        self.bn2 = get_normalization(norm_type, bottleneck_channels)

        # 1x1 expand
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = get_normalization(norm_type, out_channels)

        # Downsample if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels or use_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_normalization(norm_type, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grouped bottleneck block."""
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation_fn(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.activation_fn(out)

        return out


class MultiScaleBottleneck(nn.Module):
    """Multi-scale bottleneck with parallel branches.

    Combines multiple 3x3 kernels and dilations in parallel
    to capture features at different scales.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for convolutions. Default is 1.
        expansion: Expansion ratio. Default is 4.
        use_dilation: Whether to use dilated convolutions. Default is True.
        activation: Activation function. Default is 'relu'.
        norm_type: Normalization type. Default is 'batch'.
        use_downsample: Whether to use downsample shortcut. Default is False.

    Example:
        >>> block = MultiScaleBottleneck(
        ...     in_channels=256, out_channels=256, use_dilation=True
        ... )
        >>> x = torch.randn(2, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        use_dilation: bool = True,
        activation: str = 'relu',
        norm_type: str = 'batch',
        use_downsample: bool = False,
    ) -> None:
        """Initialize MultiScaleBottleneck."""
        super().__init__()

        self.expansion = expansion
        self.activation_fn = get_activation(activation)
        bottleneck_channels = max(1, out_channels // expansion)
        scale_channels = bottleneck_channels // 2

        # 1x1 reduce
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = get_normalization(norm_type, bottleneck_channels)

        # Multi-scale 3x3 branches
        if use_dilation:
            # Standard and dilated
            self.conv2a = nn.Conv2d(
                bottleneck_channels,
                scale_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.conv2b = nn.Conv2d(
                bottleneck_channels,
                scale_channels,
                kernel_size=3,
                stride=stride,
                padding=2,
                dilation=2,
                bias=False,
            )
        else:
            # Standard and larger kernel
            self.conv2a = nn.Conv2d(
                bottleneck_channels,
                scale_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.conv2b = nn.Conv2d(
                bottleneck_channels,
                scale_channels,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,
            )

        self.bn2a = get_normalization(norm_type, scale_channels)
        self.bn2b = get_normalization(norm_type, scale_channels)

        # 1x1 expand
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = get_normalization(norm_type, out_channels)

        # Downsample if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels or use_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_normalization(norm_type, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale bottleneck block."""
        identity = x

        # 1x1 reduce
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fn(out)

        # Multi-scale branches
        out_a = self.conv2a(out)
        out_a = self.bn2a(out_a)
        out_a = self.activation_fn(out_a)

        out_b = self.conv2b(out)
        out_b = self.bn2b(out_b)
        out_b = self.activation_fn(out_b)

        # Combine branches
        out = torch.cat([out_a, out_b], dim=1)

        # 1x1 expand
        out = self.conv3(out)
        out = self.bn3(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.activation_fn(out)

        return out


class AsymmetricBottleneck(nn.Module):
    """Asymmetric bottleneck with rectangular kernels.

    Uses asymmetric kernels (1x5, 5x1) to capture directional features
    efficiently.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for convolutions. Default is 1.
        expansion: Expansion ratio. Default is 4.
        activation: Activation function. Default is 'relu'.
        norm_type: Normalization type. Default is 'batch'.
        use_downsample: Whether to use downsample shortcut. Default is False.

    Example:
        >>> block = AsymmetricBottleneck(
        ...     in_channels=256, out_channels=256
        ... )
        >>> x = torch.randn(2, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        activation: str = 'relu',
        norm_type: str = 'batch',
        use_downsample: bool = False,
    ) -> None:
        """Initialize AsymmetricBottleneck."""
        super().__init__()

        self.expansion = expansion
        self.activation_fn = get_activation(activation)
        bottleneck_channels = max(1, out_channels // expansion)

        # 1x1 reduce
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = get_normalization(norm_type, bottleneck_channels)

        # Asymmetric convolutions (1x5 and 5x1)
        self.conv2a = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=(1, 5),
            stride=(1, stride),
            padding=(0, 2),
            bias=False,
        )
        self.bn2a = get_normalization(norm_type, bottleneck_channels)

        self.conv2b = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=(5, 1),
            stride=(stride, 1),
            padding=(2, 0),
            bias=False,
        )
        self.bn2b = get_normalization(norm_type, bottleneck_channels)

        # 1x1 expand
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = get_normalization(norm_type, out_channels)

        # Downsample if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels or use_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_normalization(norm_type, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply asymmetric bottleneck block."""
        identity = x

        # 1x1 reduce
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fn(out)

        # Asymmetric convolutions
        out = self.conv2a(out)
        out = self.bn2a(out)
        out = self.activation_fn(out)

        out = self.conv2b(out)
        out = self.bn2b(out)
        out = self.activation_fn(out)

        # 1x1 expand
        out = self.conv3(out)
        out = self.bn3(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.activation_fn(out)

        return out


def create_bottleneck(
    bottleneck_type: str,
    in_channels: int,
    out_channels: int,
    **kwargs: Any,
) -> nn.Module:
    """Factory function to create bottleneck blocks.

    Args:
        bottleneck_type: Type of bottleneck ('standard', 'wide', 'grouped',
                         'multi_scale', 'asymmetric').
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        **kwargs: Additional keyword arguments for the bottleneck.

    Returns:
        Initialized bottleneck block.

    Raises:
        ValueError: If bottleneck_type is not supported.

    Example:
        >>> block = create_bottleneck(
        ...     'wide', in_channels=256, out_channels=256,
        ...     width_multiplier=1.5
        ... )
    """
    bottleneck_types = {
        'standard': StandardBottleneck,
        'wide': WideBottleneck,
        'grouped': GroupedBottleneck,
        'multi_scale': MultiScaleBottleneck,
        'asymmetric': AsymmetricBottleneck,
    }

    if bottleneck_type not in bottleneck_types:
        raise ValueError(
            f"Bottleneck type '{bottleneck_type}' not supported. "
            f"Choose from: {list(bottleneck_types.keys())}"
        )

    return bottleneck_types[bottleneck_type](in_channels, out_channels, **kwargs)
