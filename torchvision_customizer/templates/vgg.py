"""VGG Template: Parametric VGG implementation.

Implements VGG variants based on the layers parameter:
- layers=11: VGG-11 (8 conv layers)
- layers=13: VGG-13 (10 conv layers)
- layers=16: VGG-16 (13 conv layers)
- layers=19: VGG-19 (16 conv layers)

All implementations are from scratch using the package's building blocks.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from torchvision_customizer.templates.base import Template
from torchvision_customizer.layers import get_activation, get_normalization


# VGG configurations: layers -> list of (num_convs, channels)
# 'M' represents max pooling
VGG_CONFIGS = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGModel(nn.Module):
    """Complete VGG model built from blocks."""
    
    def __init__(
        self,
        layers: int,
        num_classes: int = 1000,
        in_channels: int = 3,
        activation: str = 'relu',
        use_batchnorm: bool = True,
        dropout: float = 0.5,
        channel_scale: float = 1.0,
    ):
        super().__init__()
        
        if layers not in VGG_CONFIGS:
            raise ValueError(f"Unsupported layers={layers}. Choose from {list(VGG_CONFIGS.keys())}")
        
        self.layers = layers
        self.use_batchnorm = use_batchnorm
        
        # Build features
        config = VGG_CONFIGS[layers]
        self.features = self._make_features(
            config, in_channels, activation, use_batchnorm, channel_scale
        )
        
        # Classifier
        final_channels = int(512 * channel_scale)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(final_channels * 7 * 7, 4096),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _make_features(
        self,
        config: List,
        in_channels: int,
        activation: str,
        use_batchnorm: bool,
        channel_scale: float,
    ) -> nn.Sequential:
        """Build the feature extraction layers."""
        layers = []
        
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = int(v * channel_scale)
                layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(get_activation(activation))
                in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def explain(self) -> str:
        """Human-readable model description."""
        config = VGG_CONFIGS[self.layers]
        conv_count = sum(1 for v in config if v != 'M')
        bn_str = " + BatchNorm" if self.use_batchnorm else ""
        
        lines = [
            "+" + "-" * 60 + "+",
            "|" + f" VGG-{self.layers}{bn_str} ".center(60, "-") + "|",
            "+" + "-" * 60 + "+",
            f"| Features: {conv_count} Conv3x3 layers with 5 MaxPool".ljust(61) + "|",
            "|" + " " * 60 + "|",
        ]
        
        # Show stage structure
        current_stage = []
        stage_num = 1
        for v in config:
            if v == 'M':
                lines.append(f"| Stage {stage_num}: {' -> '.join(current_stage)} -> MaxPool".ljust(61) + "|")
                current_stage = []
                stage_num += 1
            else:
                current_stage.append(f"Conv({v})")
        
        lines.extend([
            "|" + " " * 60 + "|",
            f"| Classifier: FC(4096) -> FC(4096) -> FC(classes)".ljust(61) + "|",
            "+" + "-" * 60 + "+",
            f"| Parameters: {self.count_parameters():,}".ljust(61) + "|",
            "+" + "-" * 60 + "+",
        ])
        
        return "\n".join(lines)


@Template.register('vgg')
class VGGTemplate(Template):
    """VGG architecture template.
    
    Example:
        >>> template = VGGTemplate(layers=16)
        >>> template.use_batchnorm(True)
        >>> model = template.build(num_classes=100)
    """
    
    def __init__(self, layers: int = 16, **kwargs):
        if layers not in VGG_CONFIGS:
            raise ValueError(f"layers must be one of {list(VGG_CONFIGS.keys())}, got {layers}")
        
        config = {
            'name': f'VGG-{layers}',
            'layers': layers,
            'activation': kwargs.get('activation', 'relu'),
            'use_batchnorm': kwargs.get('use_batchnorm', True),
            'dropout': kwargs.get('dropout', 0.5),
            'channel_scale': kwargs.get('channel_scale', 1.0),
        }
        super().__init__(config)
    
    def use_batchnorm(self, enabled: bool = True) -> 'VGGTemplate':
        """Enable or disable batch normalization."""
        self._config['use_batchnorm'] = enabled
        return self
    
    def build(self, num_classes: int = 1000, **kwargs) -> nn.Module:
        """Build VGG model from template configuration."""
        config = self.get_config()
        
        return VGGModel(
            layers=config['layers'],
            num_classes=num_classes,
            activation=config.get('activation', 'relu'),
            use_batchnorm=config.get('use_batchnorm', True),
            dropout=config.get('dropout', 0.5),
            channel_scale=config.get('channel_scale', 1.0),
            **kwargs,
        )


def vgg(
    layers: int = 16,
    num_classes: int = 1000,
    **kwargs,
) -> nn.Module:
    """Create a VGG model.
    
    Parametric factory function for VGG architectures.
    
    Args:
        layers: Number of layers (11, 13, 16, 19)
        num_classes: Number of output classes
        **kwargs: Additional configuration (activation, use_batchnorm, dropout, etc.)
        
    Returns:
        VGG model
        
    Example:
        >>> model = vgg(layers=16, num_classes=10)
        >>> model = vgg(layers=19, num_classes=1000, use_batchnorm=True)
    """
    return VGGModel(layers=layers, num_classes=num_classes, **kwargs)
