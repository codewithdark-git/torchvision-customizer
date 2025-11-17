"""Residual architecture patterns and builders.

Implements various residual network architectures and patterns for
efficient model building:
- ResidualStage: Multi-block stages
- ResidualSequence: Sequential residual patterns
- ResidualBottleneckStage: Bottleneck-based stages
- ResidualArchitectureBuilder: Fluent builder for residual networks

Example:
    >>> from torchvision_customizer.blocks import ResidualStage
    >>> stage = ResidualStage(
    ...     in_channels=64, out_channels=64, num_blocks=3
    ... )
    >>> output = stage(x)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from torchvision_customizer.blocks import ResidualBlock
from torchvision_customizer.blocks.bottleneck_block import (
    create_bottleneck,
    StandardBottleneck,
    WideBottleneck,
    GroupedBottleneck,
    MultiScaleBottleneck,
    AsymmetricBottleneck,
)
from torchvision_customizer.layers import get_activation


@dataclass
class ResidualStageConfig:
    """Configuration for a residual stage."""

    in_channels: int
    out_channels: int
    num_blocks: int
    stride: int = 1
    expansion: int = 4
    use_bottleneck: bool = True
    bottleneck_type: str = 'standard'
    activation: str = 'relu'
    norm_type: str = 'batch'
    use_se: bool = False
    se_reduction: int = 16
    downsample_first: bool = True


class ResidualStage(nn.Module):
    """Sequential residual blocks forming a stage.

    Groups multiple residual blocks into a stage with consistent
    architecture and optional downsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_blocks: Number of residual blocks in stage.
        stride: Stride for first block. Default is 1.
        expansion: Channel expansion ratio. Default is 4.
        use_bottleneck: Whether to use bottleneck blocks. Default is True.
        bottleneck_type: Type of bottleneck. Default is 'standard'.
        activation: Activation function. Default is 'relu'.
        norm_type: Normalization type. Default is 'batch'.
        use_se: Whether to use SE blocks. Default is False.
        se_reduction: SE reduction ratio. Default is 16.
        downsample_first: Downsample in first block. Default is True.

    Example:
        >>> stage = ResidualStage(
        ...     in_channels=64, out_channels=128, num_blocks=3, stride=2
        ... )
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = stage(x)
        >>> print(output.shape)
        torch.Size([2, 128, 16, 16])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        expansion: int = 4,
        use_bottleneck: bool = True,
        bottleneck_type: str = 'standard',
        activation: str = 'relu',
        norm_type: str = 'batch',
        use_se: bool = False,
        se_reduction: int = 16,
        downsample_first: bool = True,
    ) -> None:
        """Initialize ResidualStage."""
        super().__init__()

        self.num_blocks = num_blocks
        self.use_bottleneck = use_bottleneck
        blocks = []

        for i in range(num_blocks):
            # First block handles stride and dimension change
            if i == 0:
                block_stride = stride if downsample_first else 1
                block_in = in_channels
                block_out = out_channels
                needs_downsample = stride != 1 or in_channels != out_channels
            else:
                block_stride = 1
                block_in = out_channels
                block_out = out_channels
                needs_downsample = False

            if use_bottleneck:
                block = create_bottleneck(
                    bottleneck_type,
                    in_channels=block_in,
                    out_channels=block_out,
                    stride=block_stride,
                    expansion=expansion,
                    activation=activation,
                    norm_type=norm_type,
                    use_downsample=needs_downsample,
                )
            else:
                block = ResidualBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    stride=block_stride,
                    downsample=needs_downsample,
                    activation=activation,
                    norm_type=norm_type,
                    use_se=use_se,
                    se_reduction=se_reduction,
                    bottleneck=False,
                )

            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual stage.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Output tensor of shape (B, out_channels, H', W').
        """
        return self.blocks(x)


class ResidualSequence(nn.Module):
    """Sequence of residual stages with controlled downsampling.

    Builds a full residual network by stacking multiple stages
    with optional position-specific downsampling.

    Args:
        in_channels: Initial number of channels.
        num_stages: Number of stages to create.
        blocks_per_stage: Blocks in each stage. Can be int or list.
        channels_per_stage: Output channels per stage. Can be int or list.
        stride_schedule: Where to apply stride=2. Default is 'later'.
        expansion: Bottleneck expansion. Default is 4.
        use_bottleneck: Use bottleneck blocks. Default is True.
        activation: Activation function. Default is 'relu'.
        norm_type: Normalization type. Default is 'batch'.
        use_se: Use SE blocks. Default is False.

    Example:
        >>> seq = ResidualSequence(
        ...     in_channels=3, num_stages=4, blocks_per_stage=3,
        ...     channels_per_stage=[64, 128, 256, 512]
        ... )
        >>> x = torch.randn(2, 3, 224, 224)
        >>> output = seq(x)
        >>> print(output.shape)
        torch.Size([2, 512, 14, 14])
    """

    def __init__(
        self,
        in_channels: int,
        num_stages: int,
        blocks_per_stage: int | List[int],
        channels_per_stage: int | List[int],
        stride_schedule: str = 'later',
        expansion: int = 4,
        use_bottleneck: bool = True,
        activation: str = 'relu',
        norm_type: str = 'batch',
        use_se: bool = False,
    ) -> None:
        """Initialize ResidualSequence."""
        super().__init__()

        # Handle list vs scalar inputs
        if isinstance(blocks_per_stage, int):
            blocks_per_stage = [blocks_per_stage] * num_stages
        if isinstance(channels_per_stage, int):
            channels_per_stage = [channels_per_stage] * num_stages

        assert len(blocks_per_stage) == num_stages
        assert len(channels_per_stage) == num_stages

        # Determine stride schedule
        if stride_schedule == 'later':
            strides = [1] + [2] * (num_stages - 1)
        elif stride_schedule == 'all':
            strides = [2] * num_stages
        elif stride_schedule == 'first':
            strides = [2] + [1] * (num_stages - 1)
        else:
            strides = [1] * num_stages

        stages = []
        current_channels = in_channels

        for i in range(num_stages):
            stage = ResidualStage(
                in_channels=current_channels,
                out_channels=channels_per_stage[i],
                num_blocks=blocks_per_stage[i],
                stride=strides[i],
                expansion=expansion,
                use_bottleneck=use_bottleneck,
                activation=activation,
                norm_type=norm_type,
                use_se=use_se,
                downsample_first=True,
            )
            stages.append(stage)
            current_channels = channels_per_stage[i]

        self.stages = nn.Sequential(*stages)
        self.out_channels = channels_per_stage[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual sequence."""
        return self.stages(x)


class ResidualBottleneckStage(nn.Module):
    """Stage with mixed bottleneck types for advanced architectures.

    Allows using different bottleneck types in the same stage for
    flexible architecture design.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_blocks: Number of blocks in stage.
        bottleneck_types: Bottleneck types per block (str or list).
        stride: Stride for first block. Default is 1.
        expansion: Channel expansion ratio. Default is 4.
        activation: Activation function. Default is 'relu'.
        norm_type: Normalization type. Default is 'batch'.

    Example:
        >>> stage = ResidualBottleneckStage(
        ...     in_channels=64, out_channels=128, num_blocks=3,
        ...     bottleneck_types=['standard', 'wide', 'grouped']
        ... )
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = stage(x)
        >>> print(output.shape)
        torch.Size([2, 128, 16, 16])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        bottleneck_types: str | List[str] = 'standard',
        stride: int = 1,
        expansion: int = 4,
        activation: str = 'relu',
        norm_type: str = 'batch',
    ) -> None:
        """Initialize ResidualBottleneckStage."""
        super().__init__()

        if isinstance(bottleneck_types, str):
            bottleneck_types = [bottleneck_types] * num_blocks

        assert len(bottleneck_types) == num_blocks

        blocks = []
        for i in range(num_blocks):
            if i == 0:
                block_stride = stride
                block_in = in_channels
                block_out = out_channels
                needs_downsample = stride != 1 or in_channels != out_channels
            else:
                block_stride = 1
                block_in = out_channels
                block_out = out_channels
                needs_downsample = False

            block = create_bottleneck(
                bottleneck_types[i],
                in_channels=block_in,
                out_channels=block_out,
                stride=block_stride,
                expansion=expansion,
                activation=activation,
                norm_type=norm_type,
                use_downsample=needs_downsample,
            )
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual bottleneck stage."""
        return self.blocks(x)


class ResidualArchitectureBuilder:
    """Fluent builder for constructing residual networks.

    Provides a chainable interface for building complex residual
    architectures with various configurations.

    Example:
        >>> model = (ResidualArchitectureBuilder()
        ...     .add_initial_conv(3, 64)
        ...     .add_stage(64, 64, 3)
        ...     .add_stage(64, 128, 4, stride=2, use_bottleneck=True)
        ...     .add_stage(128, 256, 6, stride=2, use_bottleneck=True)
        ...     .build())
        >>> x = torch.randn(2, 3, 224, 224)
        >>> output = model(x)
    """

    def __init__(self) -> None:
        """Initialize ResidualArchitectureBuilder."""
        self.layers: List[Tuple[str, Dict[str, Any]]] = []
        self.current_channels: Optional[int] = None

    def add_initial_conv(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        padding: int = 3,
        norm_type: str = 'batch',
        activation: str = 'relu',
    ) -> 'ResidualArchitectureBuilder':
        """Add initial convolutional layer.

        Args:
            in_channels: Number of input channels (typically 3 for RGB).
            out_channels: Number of output channels.
            kernel_size: Kernel size. Default is 7.
            stride: Stride. Default is 2.
            padding: Padding. Default is 3.
            norm_type: Normalization type. Default is 'batch'.
            activation: Activation function. Default is 'relu'.

        Returns:
            Self for chaining.
        """
        self.layers.append(
            (
                'initial_conv',
                {
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding,
                    'norm_type': norm_type,
                    'activation': activation,
                },
            )
        )
        self.current_channels = out_channels
        return self

    def add_max_pool(self, kernel_size: int = 3, stride: int = 2, padding: int = 1) -> 'ResidualArchitectureBuilder':
        """Add max pooling layer.

        Args:
            kernel_size: Kernel size. Default is 3.
            stride: Stride. Default is 2.
            padding: Padding. Default is 1.

        Returns:
            Self for chaining.
        """
        self.layers.append(
            (
                'max_pool',
                {'kernel_size': kernel_size, 'stride': stride, 'padding': padding},
            )
        )
        return self

    def add_stage(
        self,
        in_channels: Optional[int],
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        expansion: int = 4,
        use_bottleneck: bool = True,
        bottleneck_type: str = 'standard',
        activation: str = 'relu',
        norm_type: str = 'batch',
        use_se: bool = False,
    ) -> 'ResidualArchitectureBuilder':
        """Add residual stage.

        Args:
            in_channels: Number of input channels (None to use current).
            out_channels: Number of output channels.
            num_blocks: Number of blocks in stage.
            stride: Stride for first block. Default is 1.
            expansion: Channel expansion. Default is 4.
            use_bottleneck: Use bottleneck blocks. Default is True.
            bottleneck_type: Bottleneck type. Default is 'standard'.
            activation: Activation function. Default is 'relu'.
            norm_type: Normalization type. Default is 'batch'.
            use_se: Use SE blocks. Default is False.

        Returns:
            Self for chaining.
        """
        if in_channels is None:
            if self.current_channels is None:
                raise ValueError('in_channels must be specified for first stage')
            in_channels = self.current_channels

        self.layers.append(
            (
                'stage',
                {
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'num_blocks': num_blocks,
                    'stride': stride,
                    'expansion': expansion,
                    'use_bottleneck': use_bottleneck,
                    'bottleneck_type': bottleneck_type,
                    'activation': activation,
                    'norm_type': norm_type,
                    'use_se': use_se,
                },
            )
        )
        self.current_channels = out_channels
        return self

    def add_mixed_bottleneck_stage(
        self,
        in_channels: Optional[int],
        out_channels: int,
        num_blocks: int,
        bottleneck_types: List[str],
        stride: int = 1,
        expansion: int = 4,
        activation: str = 'relu',
        norm_type: str = 'batch',
    ) -> 'ResidualArchitectureBuilder':
        """Add stage with mixed bottleneck types.

        Args:
            in_channels: Number of input channels (None to use current).
            out_channels: Number of output channels.
            num_blocks: Number of blocks.
            bottleneck_types: Bottleneck type per block.
            stride: Stride for first block. Default is 1.
            expansion: Channel expansion. Default is 4.
            activation: Activation function. Default is 'relu'.
            norm_type: Normalization type. Default is 'batch'.

        Returns:
            Self for chaining.
        """
        if in_channels is None:
            if self.current_channels is None:
                raise ValueError('in_channels must be specified for first stage')
            in_channels = self.current_channels

        self.layers.append(
            (
                'mixed_bottleneck_stage',
                {
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'num_blocks': num_blocks,
                    'bottleneck_types': bottleneck_types,
                    'stride': stride,
                    'expansion': expansion,
                    'activation': activation,
                    'norm_type': norm_type,
                },
            )
        )
        self.current_channels = out_channels
        return self

    def add_global_avg_pool(self) -> 'ResidualArchitectureBuilder':
        """Add global average pooling layer.

        Returns:
            Self for chaining.
        """
        self.layers.append(('global_avg_pool', {}))
        return self

    def build(self) -> nn.Module:
        """Build the residual network.

        Returns:
            Constructed nn.Module.
        """
        if not self.layers:
            raise ValueError('No layers added to builder')

        return _ResidualArchitecture(self.layers)


class _ResidualArchitecture(nn.Module):
    """Internal class for built residual architectures."""

    def __init__(self, layers: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Initialize _ResidualArchitecture."""
        super().__init__()
        self.layer_sequence = nn.ModuleList()

        for layer_type, config in layers:
            if layer_type == 'initial_conv':
                layer = self._create_initial_conv(**config)
            elif layer_type == 'max_pool':
                layer = nn.MaxPool2d(**config)
            elif layer_type == 'stage':
                layer = ResidualStage(**config)
            elif layer_type == 'mixed_bottleneck_stage':
                layer = ResidualBottleneckStage(**config)
            elif layer_type == 'global_avg_pool':
                layer = nn.AdaptiveAvgPool2d((1, 1))
            else:
                raise ValueError(f'Unknown layer type: {layer_type}')

            self.layer_sequence.append(layer)

    @staticmethod
    def _create_initial_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        norm_type: str,
        activation: str,
    ) -> nn.Sequential:
        """Create initial convolutional layer."""
        from torchvision_customizer.layers import get_normalization

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            get_normalization(norm_type, out_channels),
            get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual architecture."""
        for layer in self.layer_sequence:
            x = layer(x)
        return x
