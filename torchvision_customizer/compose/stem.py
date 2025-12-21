"""Stem builder for network entry point.

The Stem is the initial feature extraction stage of a network,
typically consisting of a large convolution followed by pooling.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from torchvision_customizer.compose.operators import ComposableModule
from torchvision_customizer.blocks import ConvBlock
from torchvision_customizer.layers import get_activation, get_normalization, get_pooling


class Stem(ComposableModule):
    """Network stem (entry point) builder.
    
    Creates the initial feature extraction layers, typically:
    - Large kernel convolution (e.g., 7x7 with stride 2)
    - Batch normalization
    - Activation
    - Optional pooling
    
    Args:
        in_channels: Input channels (default 3 for RGB)
        channels: Output channels
        kernel: Kernel size (default 7)
        stride: Stride (default 2)
        padding: Padding (default auto-calculated)
        activation: Activation function name
        norm: Normalization type ('batch', 'layer', 'group', 'instance', None)
        pool: Pooling type ('max', 'avg', None)
        pool_kernel: Pooling kernel size
        pool_stride: Pooling stride
        
    Example:
        >>> stem = Stem(channels=64, kernel=7, stride=2)
        >>> # Creates: Conv(3->64, 7x7, s2) -> BN -> ReLU -> MaxPool(3x3, s2)
        
        >>> stem = Stem(channels=64, kernel=3, stride=1, pool=None)
        >>> # Creates: Conv(3->64, 3x3, s1) -> BN -> ReLU
    """
    
    def __init__(
        self,
        channels: int,
        in_channels: int = 3,
        kernel: int = 7,
        stride: int = 2,
        padding: Optional[int] = None,
        activation: str = 'relu',
        norm: str = 'batch',
        pool: Optional[str] = 'max',
        pool_kernel: int = 3,
        pool_stride: int = 2,
        pool_padding: int = 1,
        # Aliases for convenience
        kernel_size: Optional[int] = None,
    ):
        super().__init__()
        
        # Handle kernel_size alias
        if kernel_size is not None:
            kernel = kernel_size
        
        # Auto-calculate padding for same output spatial dim (before stride)
        if padding is None:
            padding = kernel // 2
        
        self.in_channels = in_channels
        self._out_channels = channels
        
        # Build stem layers
        layers = []
        
        # Main convolution
        layers.append(nn.Conv2d(
            in_channels, channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=(norm is None)  # No bias if using normalization
        ))
        
        # Normalization
        if norm:
            layers.append(get_normalization(norm, channels))
        
        # Activation
        if activation:
            layers.append(get_activation(activation))
        
        # Pooling
        if pool:
            if pool == 'max':
                layers.append(nn.MaxPool2d(
                    kernel_size=pool_kernel,
                    stride=pool_stride,
                    padding=pool_padding
                ))
            elif pool == 'avg':
                layers.append(nn.AvgPool2d(
                    kernel_size=pool_kernel,
                    stride=pool_stride,
                    padding=pool_padding
                ))
        
        self.stem = nn.Sequential(*layers)
        
        # Store config for repr
        self._config = {
            'in_channels': in_channels,
            'channels': channels,
            'kernel': kernel,
            'stride': stride,
            'activation': activation,
            'norm': norm,
            'pool': pool,
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)
    
    def __repr__(self) -> str:
        parts = [f"Conv({self._config['in_channels']}->{self._config['channels']}, "
                 f"{self._config['kernel']}x{self._config['kernel']}, s{self._config['stride']})"]
        
        if self._config['norm']:
            parts.append(self._config['norm'].upper())
        if self._config['activation']:
            parts.append(self._config['activation'].upper())
        if self._config['pool']:
            parts.append(f"{self._config['pool'].upper()}Pool")
        
        return f"Stem({' -> '.join(parts)})"


class SimpleStem(ComposableModule):
    """Simple stem for small images (e.g., CIFAR).
    
    Uses a single 3x3 convolution without pooling.
    
    Args:
        channels: Output channels
        in_channels: Input channels (default 3)
        activation: Activation function
        norm: Normalization type
    """
    
    def __init__(
        self,
        channels: int,
        in_channels: int = 3,
        activation: str = 'relu',
        norm: str = 'batch',
    ):
        super().__init__()
        
        self._out_channels = channels
        
        layers = [
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=(norm is None))
        ]
        
        if norm:
            layers.append(get_normalization(norm, channels))
        if activation:
            layers.append(get_activation(activation))
        
        self.stem = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)
    
    def __repr__(self) -> str:
        return f"SimpleStem({self._out_channels})"


class PatchStem(ComposableModule):
    """Patch embedding stem for Vision Transformer style models.
    
    Divides input into patches and projects them to embedding dimension.
    
    Args:
        embed_dim: Embedding dimension
        patch_size: Size of each patch
        in_channels: Input channels (default 3)
    """
    
    def __init__(
        self,
        embed_dim: int,
        patch_size: int = 16,
        in_channels: int = 3,
    ):
        super().__init__()
        
        self._out_channels = embed_dim
        self.patch_size = patch_size
        
        # Patch embedding via convolution
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, E, H/P, W/P)
        return self.proj(x)
    
    def __repr__(self) -> str:
        return f"PatchStem(dim={self._out_channels}, patch={self.patch_size})"
