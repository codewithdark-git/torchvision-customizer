"""
Advanced Architecture Features Module

This module provides sophisticated neural network components for advanced architectures:
- Residual connections with per-layer configuration
- Multiple skip connection patterns
- Dense connections (DenseNet-style)
- Mixed architecture combinations

Author: torchvision-customizer
License: MIT
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision_customizer.blocks.conv_block import ConvBlock


class ResidualConnector(nn.Module):
    """
    Configurable residual connection module.
    
    Supports different skip connection patterns and projection strategies.
    Can be used to add residual connections to any two layers.
    
    Attributes:
        skip_type (str): Type of skip connection ('identity', 'projection', 'bottle')
        stride (int): Stride for the main path
        in_channels (int): Input channels
        out_channels (int): Output channels
        
    Examples:
        >>> # Identity skip (same dimensions)
        >>> rc = ResidualConnector('identity', in_channels=64, out_channels=64)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> skip = torch.randn(2, 64, 32, 32)
        >>> out = rc(x, skip)
        
        >>> # Projection skip (different dimensions)
        >>> rc = ResidualConnector('projection', in_channels=64, out_channels=128, stride=2)
        >>> x = torch.randn(2, 128, 16, 16)
        >>> skip = torch.randn(2, 64, 32, 32)
        >>> out = rc(x, skip)
    """
    
    def __init__(
        self,
        skip_type: str = 'identity',
        in_channels: int = 64,
        out_channels: int = 64,
        stride: int = 1,
        activation: str = 'relu',
        use_batchnorm: bool = True,
    ):
        """
        Initialize ResidualConnector.
        
        Args:
            skip_type: Type of skip connection ('identity', 'projection', 'bottle')
            in_channels: Number of input channels for skip path
            out_channels: Number of output channels
            stride: Stride for skip path if needed
            activation: Activation function name
            use_batchnorm: Whether to use batch normalization
            
        Raises:
            ValueError: If skip_type is not recognized
        """
        super().__init__()
        
        valid_types = ['identity', 'projection', 'bottle']
        if skip_type not in valid_types:
            raise ValueError(f"skip_type must be one of {valid_types}, got {skip_type}")
        
        self.skip_type = skip_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_batchnorm = use_batchnorm
        
        # Get activation function
        from torchvision_customizer.layers.activations import get_activation
        self.activation = get_activation(activation)
        
        # Build skip path
        if skip_type == 'identity':
            self.skip = nn.Identity()
        elif skip_type == 'projection':
            self.skip = self._build_projection()
        elif skip_type == 'bottle':
            self.skip = self._build_bottle()
    
    def _build_projection(self) -> nn.Module:
        """Build projection skip connection."""
        layers = [
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=self.stride,
                bias=not self.use_batchnorm
            )
        ]
        
        if self.use_batchnorm:
            layers.append(nn.BatchNorm2d(self.out_channels))
        
        return nn.Sequential(*layers)
    
    def _build_bottle(self) -> nn.Module:
        """Build bottleneck skip connection."""
        hidden_channels = max(1, self.out_channels // 4)
        layers = [
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                self.out_channels,
                kernel_size=1,
                stride=self.stride,
                bias=not self.use_batchnorm
            )
        ]
        
        if self.use_batchnorm:
            layers.append(nn.BatchNorm2d(self.out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Main path output tensor
            skip: Skip connection input tensor
            
        Returns:
            Residual connection output with activation
        """
        skip_out = self.skip(skip)
        out = x + skip_out
        return self.activation(out)


class SkipConnectionBuilder(nn.Module):
    """
    Builder for different skip connection patterns.
    
    Supports various skip patterns:
    - dense: All previous layers connected
    - residual: Only previous layer
    - hierarchical: Skip connections at multiple scales
    - dense_residual: Combination of dense and residual
    
    Examples:
        >>> builder = SkipConnectionBuilder('residual', num_blocks=5, in_channels=64)
        >>> features = [torch.randn(2, 64, 32, 32) for _ in range(5)]
        >>> output = builder(features)
    """
    
    def __init__(
        self,
        pattern: str = 'residual',
        num_blocks: int = 4,
        in_channels: int = 64,
        activation: str = 'relu',
        use_batchnorm: bool = True,
    ):
        """
        Initialize SkipConnectionBuilder.
        
        Args:
            pattern: Type of skip pattern ('residual', 'dense', 'hierarchical', 'dense_residual')
            num_blocks: Number of blocks to connect
            in_channels: Input channels
            activation: Activation function name
            use_batchnorm: Whether to use batch normalization
            
        Raises:
            ValueError: If pattern is not recognized
        """
        super().__init__()
        
        valid_patterns = ['residual', 'dense', 'hierarchical', 'dense_residual']
        if pattern not in valid_patterns:
            raise ValueError(f"pattern must be one of {valid_patterns}, got {pattern}")
        
        self.pattern = pattern
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        
        # Create projections for connecting features
        if pattern in ['dense', 'dense_residual']:
            self._create_dense_projections()
    
    def _create_dense_projections(self):
        """Create 1x1 convolutions for dense connections."""
        self.projections = nn.ModuleList()
        for i in range(self.num_blocks):
            # Each layer needs projections to all subsequent layers
            proj_dict = nn.ModuleDict()
            for j in range(i + 1, self.num_blocks):
                proj = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.in_channels, 1, bias=False),
                    nn.BatchNorm2d(self.in_channels) if self.use_batchnorm else nn.Identity()
                )
                proj_dict[f'to_{j}'] = proj
            self.projections.append(proj_dict)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply skip connection pattern.
        
        Args:
            features: List of feature tensors from blocks
            
        Returns:
            Combined feature tensor
        """
        if self.pattern == 'residual':
            return self._residual_pattern(features)
        elif self.pattern == 'dense':
            return self._dense_pattern(features)
        elif self.pattern == 'hierarchical':
            return self._hierarchical_pattern(features)
        elif self.pattern == 'dense_residual':
            return self._dense_residual_pattern(features)
    
    def _residual_pattern(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Residual: current + previous."""
        output = features[0]
        for i in range(1, len(features)):
            output = output + features[i]
        return output
    
    def _dense_pattern(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Dense: concatenate all with projections."""
        outputs = [features[0]]
        for i in range(1, len(features)):
            # Project all previous features and concatenate
            projected = []
            for j in range(i):
                proj = self.projections[j][f'to_{i}']
                projected.append(proj(features[j]))
            combined = torch.cat(projected + [features[i]], dim=1)
            outputs.append(combined)
        return outputs[-1]
    
    def _hierarchical_pattern(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Hierarchical: skip at multiple scales."""
        # Connect every 2nd layer
        output = features[0]
        for i in range(2, len(features), 2):
            if i < len(features):
                output = output + features[i]
        output = output + features[-1]
        return output
    
    def _dense_residual_pattern(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Dense-Residual: combination of dense and residual."""
        # Residual within blocks, dense across stages
        output = features[0]
        for i in range(1, len(features)):
            # Local residual
            local_sum = output + features[i]
            # Global connection (every other layer)
            if i % 2 == 0:
                output = local_sum + features[0]
            else:
                output = local_sum
        return output


class DenseConnectionBlock(nn.Module):
    """
    DenseNet-style dense connection block.
    
    All layers are connected to each other with learned transformations.
    Features are concatenated rather than added.
    
    Attributes:
        num_layers (int): Number of dense layers
        growth_rate (int): Number of new channels per layer
        bottleneck_ratio (int): Bottleneck reduction ratio
        
    Examples:
        >>> block = DenseConnectionBlock(
        ...     num_layers=4,
        ...     in_channels=64,
        ...     growth_rate=32,
        ...     kernel_size=3
        ... )
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = block(x)
        >>> # Output channels = 64 + (4 * 32) = 192
    """
    
    def __init__(
        self,
        num_layers: int = 4,
        in_channels: int = 64,
        growth_rate: int = 32,
        kernel_size: int = 3,
        bottleneck_ratio: int = 4,
        activation: str = 'relu',
        use_batchnorm: bool = True,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize DenseConnectionBlock.
        
        Args:
            num_layers: Number of dense layers
            in_channels: Input channels
            growth_rate: Number of new channels per layer
            kernel_size: Convolution kernel size
            bottleneck_ratio: Bottleneck reduction ratio
            activation: Activation function name
            use_batchnorm: Whether to use batch normalization
            dropout_rate: Dropout probability
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if growth_rate < 1:
            raise ValueError(f"growth_rate must be >= 1, got {growth_rate}")
        
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate
        
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(num_layers):
            layer = self._build_dense_layer(
                current_channels,
                growth_rate,
                kernel_size,
                bottleneck_ratio,
                activation,
                use_batchnorm,
                dropout_rate
            )
            self.layers.append(layer)
            current_channels += growth_rate
        
        self.out_channels = current_channels
    
    def _build_dense_layer(
        self,
        in_channels: int,
        growth_rate: int,
        kernel_size: int,
        bottleneck_ratio: int,
        activation: str,
        use_batchnorm: bool,
        dropout_rate: float,
    ) -> nn.Module:
        """Build a single dense layer."""
        # Bottleneck design: 1x1 conv reduces channels, 3x3 conv creates new features
        bottleneck_channels = bottleneck_ratio * growth_rate
        
        return nn.Sequential(
            nn.BatchNorm2d(in_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                bottleneck_channels,
                kernel_size=1,
                bias=not use_batchnorm
            ),
            nn.BatchNorm2d(bottleneck_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bottleneck_channels,
                growth_rate,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=not use_batchnorm
            ),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dense connections.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with all features concatenated
        """
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)


class MixedArchitectureBlock(nn.Module):
    """
    Block combining multiple architecture patterns.
    
    Supports mixing residual, dense, and standard connections
    with optional attention mechanisms.
    
    Examples:
        >>> block = MixedArchitectureBlock(
        ...     num_layers=4,
        ...     in_channels=64,
        ...     mixed_patterns=['residual', 'dense', 'residual']
        ... )
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = block(x)
    """
    
    def __init__(
        self,
        num_layers: int = 4,
        in_channels: int = 64,
        out_channels: int = 64,
        mixed_patterns: Optional[List[str]] = None,
        kernel_size: int = 3,
        activation: str = 'relu',
        use_batchnorm: bool = True,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize MixedArchitectureBlock.
        
        Args:
            num_layers: Number of layers
            in_channels: Input channels
            out_channels: Output channels
            mixed_patterns: List of patterns per layer or None for auto
            kernel_size: Convolution kernel size
            activation: Activation function name
            use_batchnorm: Whether to use batch normalization
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Default pattern cycling
        if mixed_patterns is None:
            mixed_patterns = ['standard', 'residual', 'dense', 'standard'] * (
                (num_layers // 4) + 1
            )
            mixed_patterns = mixed_patterns[:num_layers]
        
        if len(mixed_patterns) != num_layers:
            raise ValueError(
                f"mixed_patterns length ({len(mixed_patterns)}) "
                f"must match num_layers ({num_layers})"
            )
        
        self.patterns = mixed_patterns
        self.conv_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            block = ConvBlock(
                in_channels=in_channels if i == 0 else in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                activation=activation,
                use_batchnorm=use_batchnorm,
                dropout_rate=dropout_rate,
            )
            self.conv_blocks.append(block)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed patterns."""
        features = [x]
        
        for i, (block, pattern) in enumerate(zip(self.conv_blocks, self.patterns)):
            current = block(features[-1])
            
            if pattern == 'residual' and i > 0:
                current = current + features[-1]
            elif pattern == 'dense' and i > 0:
                current = torch.cat([features[-1], current], dim=1)
            
            features.append(current)
        
        return features[-1]


# Utility functions

def create_skip_connections(
    features: List[torch.Tensor],
    pattern: str = 'residual',
) -> torch.Tensor:
    """
    Create skip connections between features.
    
    Args:
        features: List of feature tensors
        pattern: Skip pattern type
        
    Returns:
        Combined feature tensor
    """
    if pattern == 'residual':
        result = features[0]
        for feat in features[1:]:
            result = result + feat
        return result
    elif pattern == 'concatenate':
        return torch.cat(features, dim=1)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def validate_architecture_compatibility(
    features: List[torch.Tensor],
    skip_pattern: str,
) -> bool:
    """
    Validate that features are compatible with skip pattern.
    
    Args:
        features: List of feature tensors
        skip_pattern: Skip pattern type
        
    Returns:
        True if compatible
        
    Raises:
        ValueError: If incompatible
    """
    if len(features) == 0:
        raise ValueError("features list cannot be empty")
    
    if skip_pattern == 'residual':
        # All features must have same spatial and channel dimensions
        ref_shape = features[0].shape
        for feat in features[1:]:
            if feat.shape != ref_shape:
                raise ValueError(
                    f"All features must have same shape for residual pattern. "
                    f"Got {feat.shape} vs {ref_shape}"
                )
    
    return True
