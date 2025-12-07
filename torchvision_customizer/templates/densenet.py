"""DenseNet Template: Parametric DenseNet implementation.

Implements DenseNet variants based on the layers parameter:
- layers=121: DenseNet-121 (6, 12, 24, 16 blocks)
- layers=169: DenseNet-169 (6, 12, 32, 32 blocks)
- layers=201: DenseNet-201 (6, 12, 48, 32 blocks)
- layers=264: DenseNet-264 (6, 12, 64, 48 blocks)

All implementations are from scratch using the package's building blocks.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision_customizer.templates.base import Template
from torchvision_customizer.layers import get_activation, get_normalization


# DenseNet configurations: layers -> block_counts
DENSENET_CONFIGS = {
    121: [6, 12, 24, 16],
    169: [6, 12, 32, 32],
    201: [6, 12, 48, 32],
    264: [6, 12, 64, 48],
}


class DenseLayer(nn.Module):
    """Single dense layer (BN -> ReLU -> Conv)."""
    
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        mid_channels = bn_size * growth_rate
        
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.act1 = get_activation(activation)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(mid_channels)
        self.act2 = get_activation(activation)
        self.conv2 = nn.Conv2d(mid_channels, growth_rate, 3, padding=1, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))
        if self.dropout is not None:
            out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    """Dense block containing multiple dense layers."""
    
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size,
                activation,
                dropout,
            ))
        
        self.layers = nn.Sequential(*layers)
        self.out_channels = in_channels + num_layers * growth_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Transition(nn.Module):
    """Transition layer between dense blocks."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = 'relu',
    ):
        super().__init__()
        
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = get_activation(activation)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.act(self.norm(x)))
        x = self.pool(x)
        return x


class DenseNetModel(nn.Module):
    """Complete DenseNet model built from blocks."""
    
    def __init__(
        self,
        layers: int,
        num_classes: int = 1000,
        growth_rate: int = 32,
        bn_size: int = 4,
        compression: float = 0.5,
        activation: str = 'relu',
        dropout: float = 0.0,
        num_init_features: int = 64,
    ):
        super().__init__()
        
        if layers not in DENSENET_CONFIGS:
            raise ValueError(f"Unsupported layers={layers}. Choose from {list(DENSENET_CONFIGS.keys())}")
        
        block_counts = DENSENET_CONFIGS[layers]
        self.layers_config = layers
        self.growth_rate = growth_rate
        
        # Initial convolution
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, num_init_features, 7, stride=2, padding=3, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', get_activation(activation))
        self.features.add_module('pool0', nn.MaxPool2d(3, stride=2, padding=1))
        
        # Dense blocks + transitions
        num_features = num_init_features
        for i, num_layers in enumerate(block_counts):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                activation=activation,
                dropout=dropout,
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = block.out_channels
            
            # Add transition (except after last block)
            if i != len(block_counts) - 1:
                out_features = int(num_features * compression)
                trans = Transition(num_features, out_features, activation)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = out_features
        
        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        self.features.add_module('relu_final', get_activation(activation))
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_features, num_classes)
        self.num_features = num_features
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def explain(self) -> str:
        block_counts = DENSENET_CONFIGS[self.layers_config]
        
        lines = [
            "+" + "-" * 60 + "+",
            "|" + f" DenseNet-{self.layers_config} ".center(60, "-") + "|",
            "+" + "-" * 60 + "+",
            f"| Stem: Conv(7x7, s2) -> BN -> ReLU -> MaxPool(3x3, s2)".ljust(61) + "|",
            "|" + " " * 60 + "|",
        ]
        
        for i, count in enumerate(block_counts):
            lines.append(f"| DenseBlock {i+1}: {count} layers (growth={self.growth_rate})".ljust(61) + "|")
            if i < len(block_counts) - 1:
                lines.append(f"|   -> Transition (compress + avgpool)".ljust(61) + "|")
        
        lines.extend([
            "|" + " " * 60 + "|",
            f"| Head: BN -> ReLU -> AdaptiveAvgPool -> Linear".ljust(61) + "|",
            "+" + "-" * 60 + "+",
            f"| Parameters: {self.count_parameters():,}".ljust(61) + "|",
            "+" + "-" * 60 + "+",
        ])
        
        return "\n".join(lines)


@Template.register('densenet')
class DenseNetTemplate(Template):
    """DenseNet architecture template."""
    
    def __init__(self, layers: int = 121, **kwargs):
        if layers not in DENSENET_CONFIGS:
            raise ValueError(f"layers must be one of {list(DENSENET_CONFIGS.keys())}, got {layers}")
        
        config = {
            'name': f'DenseNet-{layers}',
            'layers': layers,
            'growth_rate': kwargs.get('growth_rate', 32),
            'bn_size': kwargs.get('bn_size', 4),
            'compression': kwargs.get('compression', 0.5),
            'activation': kwargs.get('activation', 'relu'),
            'dropout': kwargs.get('dropout', 0.0),
        }
        super().__init__(config)
    
    def build(self, num_classes: int = 1000, **kwargs) -> nn.Module:
        config = self.get_config()
        
        return DenseNetModel(
            layers=config['layers'],
            num_classes=num_classes,
            growth_rate=config.get('growth_rate', 32),
            bn_size=config.get('bn_size', 4),
            compression=config.get('compression', 0.5),
            activation=config.get('activation', 'relu'),
            dropout=config.get('dropout', 0.0),
            **kwargs,
        )


def densenet(
    layers: int = 121,
    num_classes: int = 1000,
    **kwargs,
) -> nn.Module:
    """Create a DenseNet model.
    
    Args:
        layers: Number of layers (121, 169, 201, 264)
        num_classes: Number of output classes
        **kwargs: Additional configuration (growth_rate, compression, etc.)
        
    Returns:
        DenseNet model
        
    Example:
        >>> model = densenet(layers=121, num_classes=10)
        >>> model = densenet(layers=201, growth_rate=48)
    """
    return DenseNetModel(layers=layers, num_classes=num_classes, **kwargs)
