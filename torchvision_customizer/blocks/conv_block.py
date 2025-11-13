"""Convolutional block implementation for building CNN architectures."""

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from torchvision_customizer.layers import get_activation


class ConvBlock(nn.Module):
    """A configurable convolutional building block for neural networks.

    This block combines a convolutional layer with optional batch normalization,
    activation functions, dropout, and pooling operations. It serves as a
    fundamental building unit for constructing custom CNN architectures.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int | Tuple[int, int]): Size of the convolutional kernel.
        stride (int | Tuple[int, int]): Stride of the convolutional layer.
        padding (int | Tuple[int, int]): Padding of the convolutional layer.
        activation (str | None): Type of activation function to use.
        use_batchnorm (bool): Whether to use batch normalization.
        dropout_rate (float): Dropout probability (0 to 1).
        pooling_type (str | None): Type of pooling operation.
        pooling_kernel_size (int | Tuple[int, int]): Kernel size for pooling.
        pooling_stride (int | Tuple[int, int]): Stride for pooling.

    Example:
        >>> # Create a basic convolutional block
        >>> block = ConvBlock(
        ...     in_channels=3,
        ...     out_channels=64,
        ...     kernel_size=3,
        ...     activation='relu',
        ...     use_batchnorm=True,
        ...     dropout_rate=0.1,
        ...     pooling_type='max'
        ... )
        >>> input_tensor = torch.randn(4, 3, 224, 224)
        >>> output = block(input_tensor)
        >>> print(output.shape)
        torch.Size([4, 64, 111, 111])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 1,
        activation: Literal["relu", "leaky_relu", "gelu", "elu", "selu", "sigmoid", "tanh", "prelu", "silu"] | None = "relu",
        use_batchnorm: bool = True,
        dropout_rate: float = 0.0,
        pooling_type: Literal["max", "avg", "adaptive_avg"] | None = None,
        pooling_kernel_size: int | Tuple[int, int] = 2,
        pooling_stride: int | Tuple[int, int] = 2,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize a ConvBlock.

        Args:
            in_channels: Number of channels in the input image. Must be positive.
            out_channels: Number of channels produced by the convolution. Must be positive.
            kernel_size: Size of the convolutional kernel. Default is 3.
                Can be int (square kernel) or tuple (height, width).
            stride: Stride of the convolution. Default is 1.
                Can be int or tuple for different dimensions.
            padding: Padding added to input. Default is 1.
                Can be int or tuple for different dimensions.
            activation: Type of activation function to apply after convolution.
                Options: 'relu', 'leaky_relu', 'gelu', 'elu', 'selu', 'sigmoid', 
                'tanh', 'prelu', 'silu'.
                If None, no activation is applied. Default is 'relu'.
            use_batchnorm: Whether to apply batch normalization after convolution.
                Default is True.
            dropout_rate: Dropout probability (between 0 and 1). Default is 0.0 (no dropout).
            pooling_type: Type of pooling to apply after the block.
                Options: 'max', 'avg', 'adaptive_avg'.
                If None, no pooling is applied. Default is None.
            pooling_kernel_size: Kernel size for pooling operation. Default is 2.
            pooling_stride: Stride for pooling operation. Default is 2.
            dilation: Spacing between kernel elements. Default is 1.
            groups: Number of groups for grouped convolution. Default is 1.
            bias: Whether to use bias in convolutional layer. Default is True.
                Usually set to False when using batch normalization.

        Raises:
            ValueError: If any parameter is invalid (e.g., negative channels, invalid activation).
        """
        super().__init__()

        # Validate inputs
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.activation_type = activation
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.pooling_type = pooling_type
        self.pooling_kernel_size = pooling_kernel_size if isinstance(pooling_kernel_size, tuple) else (pooling_kernel_size, pooling_kernel_size)
        self.pooling_stride = pooling_stride if isinstance(pooling_stride, tuple) else (pooling_stride, pooling_stride)

        # Build convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Build batch normalization layer
        self.bn: nn.Module | None = None
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        # Build activation layer
        self.activation: nn.Module | None = None
        if activation is not None:
            self.activation = self._get_activation(activation)

        # Build dropout layer
        self.dropout: nn.Module | None = None
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout2d(p=dropout_rate)

        # Build pooling layer
        self.pooling: nn.Module | None = None
        if pooling_type is not None:
            self.pooling = self._get_pooling(
                pooling_type,
                self.pooling_kernel_size,
                self.pooling_stride,
            )

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module by name.

        Uses the activation factory from torchvision_customizer.layers
        to create the activation function.

        Args:
            activation: Name of the activation function.
                Supported: 'relu', 'leaky_relu', 'gelu', 'elu', 'selu',
                'sigmoid', 'tanh', 'prelu', 'silu'

        Returns:
            Activation function module.

        Raises:
            ValueError: If activation name is not recognized.
        """
        try:
            return get_activation(activation)
        except ValueError as e:
            raise ValueError(
                f"Unknown activation function: {activation}. {str(e)}"
            ) from e

    def _get_pooling(
        self,
        pooling_type: str,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
    ) -> nn.Module:
        """Get pooling layer module by type.

        Args:
            pooling_type: Type of pooling ('max', 'avg', 'adaptive_avg').
            kernel_size: Kernel size for pooling.
            stride: Stride for pooling.

        Returns:
            Pooling layer module.

        Raises:
            ValueError: If pooling type is not recognized.
        """
        if pooling_type == "max":
            return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        elif pooling_type == "avg":
            return nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        elif pooling_type == "adaptive_avg":
            return nn.AdaptiveAvgPool2d(output_size=kernel_size)
        else:
            raise ValueError(
                f"Unknown pooling type: {pooling_type}. "
                f"Available options: ['max', 'avg', 'adaptive_avg']"
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the convolutional block.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Output tensor of shape (batch_size, out_channels, height', width'),
            where height' and width' depend on kernel_size, stride, padding,
            and pooling operations.
        """
        # Convolutional layer
        x = self.conv(x)

        # Batch normalization
        if self.bn is not None:
            x = self.bn(x)

        # Activation function
        if self.activation is not None:
            x = self.activation(x)

        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # Pooling
        if self.pooling is not None:
            x = self.pooling(x)

        return x

    @property
    def get_output_channels(self) -> int:
        """Get the number of output channels.

        Returns:
            Number of output channels produced by this block.
        """
        return self.out_channels

    def calculate_output_shape(
        self,
        input_height: int,
        input_width: int,
    ) -> Tuple[int, int]:
        """Calculate output spatial dimensions for given input size.

        This method computes the output height and width after applying
        convolution, and if applicable, pooling operations.

        Args:
            input_height: Height of the input feature map.
            input_width: Width of the input feature map.

        Returns:
            Tuple of (output_height, output_width).

        Example:
            >>> block = ConvBlock(3, 64, kernel_size=3, padding=1, stride=1, pooling_type='max')
            >>> height, width = block.calculate_output_shape(224, 224)
            >>> print(height, width)
            112 112
        """
        # Calculate output size after convolution
        # Formula: out = (in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
        conv_height = (
            input_height
            + 2 * self.padding[0]
            - (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        conv_width = (
            input_width
            + 2 * self.padding[1]
            - (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1

        # Calculate output size after pooling (if applicable)
        if self.pooling is not None:
            if self.pooling_type == "adaptive_avg":
                # Adaptive pooling outputs fixed size
                pool_height, pool_width = self.pooling_kernel_size
            else:
                # Regular pooling calculation
                pool_height = (conv_height - self.pooling_kernel_size[0]) // self.pooling_stride[0] + 1
                pool_width = (conv_width - self.pooling_kernel_size[1]) // self.pooling_stride[1] + 1
            return pool_height, pool_width

        return conv_height, conv_width

    def __repr__(self) -> str:
        """String representation of the ConvBlock."""
        config = (
            f"ConvBlock("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"activation={self.activation_type}, "
            f"use_batchnorm={self.use_batchnorm}, "
            f"dropout_rate={self.dropout_rate}, "
            f"pooling_type={self.pooling_type}"
            f")"
        )
        return config
