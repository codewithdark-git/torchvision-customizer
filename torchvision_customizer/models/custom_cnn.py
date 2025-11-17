"""Custom CNN model with flexible architecture.

Implements a configurable sequential CNN model for image classification
and feature extraction tasks.

Example:
    >>> from torchvision_customizer.models import CustomCNN
    >>> model = CustomCNN(input_shape=(3, 224, 224), num_classes=1000)
    >>> x = torch.randn(2, 3, 224, 224)
    >>> output = model(x)
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torchvision_customizer.blocks import ConvBlock
from torchvision_customizer.layers import get_pooling


class CustomCNN(nn.Module):
    """Flexible sequential CNN model for image classification.

    A highly configurable CNN model that allows specification of architecture
    at initialization. Supports auto channel progression or manual channel
    specification.

    Args:
        input_shape: Tuple of (channels, height, width) for input images.
        num_classes: Number of output classes for classification.
        num_conv_blocks: Number of convolutional blocks. Default is 4.
        channels: Channel progression. Either 'auto' to generate [64, 128, 256, ...]
                 or a list of integers specifying channels per block. Default is 'auto'.
        activation: Activation function name. Default is 'relu'.
        use_batchnorm: Whether to use batch normalization. Default is True.
        dropout_rate: Dropout probability. Default is 0.0.
        pooling_type: Type of pooling ('max', 'avg', 'adaptive_max', 'adaptive_avg').
                     Default is 'max'.
        use_pooling: Whether to use pooling after each block. Default is True.
        pooling_kernel_size: Kernel size for pooling. Default is 2.
        pooling_stride: Stride for pooling. Default is 2.

    Example:
        >>> # Auto channel progression
        >>> model = CustomCNN(
        ...     input_shape=(3, 224, 224),
        ...     num_classes=1000,
        ...     num_conv_blocks=5,
        ...     channels='auto'
        ... )
        
        >>> # Custom channel specification
        >>> model = CustomCNN(
        ...     input_shape=(3, 32, 32),
        ...     num_classes=10,
        ...     num_conv_blocks=4,
        ...     channels=[32, 64, 128, 256],
        ...     activation='leaky_relu',
        ...     use_pooling=True
        ... )
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        num_conv_blocks: int = 4,
        channels: Union[str, List[int]] = 'auto',
        activation: str = 'relu',
        use_batchnorm: bool = True,
        dropout_rate: float = 0.0,
        pooling_type: str = 'max',
        use_pooling: bool = True,
        pooling_kernel_size: int = 2,
        pooling_stride: int = 2,
    ) -> None:
        """Initialize CustomCNN."""
        super().__init__()

        # Validate input shape
        if (
            not isinstance(input_shape, (tuple, list))
            or len(input_shape) != 3
            or not all(isinstance(x, int) and x > 0 for x in input_shape)
        ):
            raise ValueError(
                f"input_shape must be a tuple of 3 positive integers, got {input_shape}"
            )

        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}")

        if not isinstance(num_conv_blocks, int) or num_conv_blocks <= 0:
            raise ValueError(f"num_conv_blocks must be a positive integer, got {num_conv_blocks}")

        if not isinstance(dropout_rate, (int, float)) or not (0 <= dropout_rate < 1):
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_conv_blocks = num_conv_blocks
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.pooling_type = pooling_type
        self.use_pooling = use_pooling
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_stride = pooling_stride

        # Generate or validate channels
        self.channels = self._validate_and_get_channels(channels)

        # Build feature extractor and classifier
        self.features = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _validate_and_get_channels(self, channels: Union[str, List[int]]) -> List[int]:
        """Generate or validate channel progression.

        Args:
            channels: Either 'auto' or a list of channel sizes.

        Returns:
            List of channel sizes for each block.

        Raises:
            ValueError: If channels is invalid or list length doesn't match num_conv_blocks.
        """
        if isinstance(channels, str):
            if channels != 'auto':
                raise ValueError(f"Unsupported channel mode: {channels}. Use 'auto' or a list.")
            # Generate auto progression: 64, 128, 256, 512, ...
            return [64 * (2**i) for i in range(self.num_conv_blocks)]

        elif isinstance(channels, (list, tuple)):
            channels = list(channels)
            if len(channels) != self.num_conv_blocks:
                raise ValueError(
                    f"Length of channels ({len(channels)}) must match "
                    f"num_conv_blocks ({self.num_conv_blocks})"
                )
            if not all(isinstance(c, int) and c > 0 for c in channels):
                raise ValueError("All channel values must be positive integers")
            return channels

        else:
            raise ValueError(f"channels must be 'auto' or a list, got {type(channels)}")

    def _make_feature_extractor(self) -> nn.Sequential:
        """Build the feature extraction layers.

        Returns:
            nn.Sequential containing convolutional blocks and pooling layers.
        """
        layers: List[nn.Module] = []
        in_channels = self.input_shape[0]

        for i, out_channels in enumerate(self.channels):
            # Convolutional block
            layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    activation=self.activation,
                    use_batchnorm=self.use_batchnorm,
                    dropout_rate=self.dropout_rate,
                    pooling_type=self.pooling_type if self.use_pooling else None,
                )
            )

            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_classifier(self) -> nn.Sequential:
        """Build the classification head.

        Returns:
            nn.Sequential containing global pooling, flatten, and linear layer.
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.channels[-1], self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        # Validate input shape at runtime
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got {x.dim()}D")
        if x.shape[1] != self.input_shape[0]:
            raise ValueError(
                f"Expected input channels {self.input_shape[0]}, got {x.shape[1]}"
            )

        x = self.features(x)
        x = self.classifier(x)
        return x

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count the number of parameters.

        Args:
            trainable_only: If True, count only trainable parameters. Default is False.

        Returns:
            Total number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> dict:
        """Get model configuration as a dictionary.

        Returns:
            Dictionary containing all model configuration parameters.
        """
        return {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'num_conv_blocks': self.num_conv_blocks,
            'channels': self.channels,
            'activation': self.activation,
            'use_batchnorm': self.use_batchnorm,
            'dropout_rate': self.dropout_rate,
            'pooling_type': self.pooling_type,
            'use_pooling': self.use_pooling,
            'pooling_kernel_size': self.pooling_kernel_size,
            'pooling_stride': self.pooling_stride,
        }

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_shape={self.input_shape},\n"
            f"  num_classes={self.num_classes},\n"
            f"  num_conv_blocks={self.num_conv_blocks},\n"
            f"  channels={self.channels},\n"
            f"  activation={self.activation},\n"
            f"  use_batchnorm={self.use_batchnorm},\n"
            f"  dropout_rate={self.dropout_rate},\n"
            f"  pooling_type={self.pooling_type}\n"
            f")"
        )
