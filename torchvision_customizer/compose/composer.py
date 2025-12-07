"""Model Composer: High-level composition utilities.

Provides the Compose class for flexible model assembly and
utilities for common architecture patterns.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from torchvision_customizer.compose.operators import ComposableModule, ComposedModule
from torchvision_customizer.compose.stem import Stem
from torchvision_customizer.compose.stage import Stage
from torchvision_customizer.compose.head import Head


class Compose(ComposableModule):
    """High-level model composer.
    
    Flexible way to compose models from layers, with automatic
    channel tracking and configuration.
    
    Args:
        input_shape: Input tensor shape (C, H, W)
        layers: List of layers/modules to compose
        
    Example:
        >>> model = Compose(
        ...     input_shape=(3, 224, 224),
        ...     layers=[
        ...         Stem(64),
        ...         Stage(128, blocks=3, pattern='residual'),
        ...         Stage(256, blocks=3, pattern='residual', downsample=True),
        ...         Head(num_classes=1000)
        ...     ]
        ... )
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        layers: List[nn.Module],
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.layers = nn.ModuleList(layers)
        
        # Track output channels
        if layers and hasattr(layers[-1], 'out_channels'):
            self._out_channels = layers[-1].out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def explain(self) -> str:
        """Generate human-readable model explanation."""
        lines = ["+" + "-" * 60 + "+"]
        lines.append("|" + "Model Architecture".center(60) + "|")
        lines.append("+" + "-" * 60 + "+")
        
        for i, layer in enumerate(self.layers):
            layer_str = repr(layer)
            if len(layer_str) > 58:
                layer_str = layer_str[:55] + "..."
            lines.append("| " + layer_str.ljust(58) + " |")
        
        lines.append("+" + "-" * 60 + "+")
        
        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        lines.append("| " + f"Parameters: {total_params:,} ({trainable:,} trainable)".ljust(58) + " |")
        lines.append("+" + "-" * 60 + "+")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        layer_types = [type(l).__name__ for l in self.layers]
        return f"Compose([{', '.join(layer_types)}])"


class Sequential(ComposableModule):
    """Simple sequential model wrapper with operator support.
    
    Like nn.Sequential but with >> operator support.
    
    Example:
        >>> model = Sequential(conv1, conv2, conv3)
        >>> model = model >> head  # Add more layers
    """
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        
        if layers and hasattr(layers[-1], 'out_channels'):
            self._out_channels = layers[-1].out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    def __rshift__(self, other: nn.Module) -> 'Sequential':
        """Add layer with >> operator."""
        new_layers = list(self.layers) + [other]
        return Sequential(*new_layers)
    
    def __repr__(self) -> str:
        return f"Sequential({len(self.layers)} layers)"


class VisionModel(ComposableModule):
    """Base class for vision models with structure.
    
    Provides a standard structure with stem, stages, and head,
    plus introspection and modification methods.
    
    Args:
        stem: Network entry point
        stages: List of body stages
        head: Classification/output head
    """
    
    def __init__(
        self,
        stem: nn.Module,
        stages: List[nn.Module],
        head: nn.Module,
    ):
        super().__init__()
        
        self.stem = stem
        self.stages = nn.ModuleList(stages)
        self.head = head
        
        if hasattr(head, 'num_classes'):
            self._out_channels = head.num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without head (feature extraction)."""
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x
    
    def get_stage_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get intermediate outputs from each stage."""
        outputs = []
        
        x = self.stem(x)
        outputs.append(x)
        
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        
        return outputs
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def explain(self) -> str:
        """Generate human-readable model explanation."""
        lines = []
        lines.append("+" + "-" * 60 + "+")
        lines.append("|" + f" {type(self).__name__} ".center(60, "-") + "|")
        lines.append("+" + "-" * 60 + "+")
        
        # Stem
        lines.append("| Stem: " + repr(self.stem).ljust(52) + " |")
        lines.append("|" + " " * 60 + "|")
        
        # Stages
        for i, stage in enumerate(self.stages):
            stage_str = repr(stage)
            if len(stage_str) > 50:
                stage_str = stage_str[:47] + "..."
            lines.append(f"| Stage {i+1}: {stage_str}".ljust(61) + "|")
        
        lines.append("|" + " " * 60 + "|")
        
        # Head
        lines.append("| Head: " + repr(self.head).ljust(52) + " |")
        
        lines.append("+" + "-" * 60 + "+")
        
        # Stats
        params = self.count_parameters()
        trainable = self.count_parameters(trainable_only=True)
        lines.append(f"| Parameters: {params:,} | Trainable: {trainable:,}".ljust(61) + "|")
        
        lines.append("+" + "-" * 60 + "+")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (f"{type(self).__name__}("
                f"stem={type(self.stem).__name__}, "
                f"stages={len(self.stages)}, "
                f"head={type(self.head).__name__})")


def build_vision_model(
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    num_classes: int = 1000,
    stem_channels: int = 64,
    stage_channels: List[int] = [64, 128, 256, 512],
    blocks_per_stage: Union[int, List[int]] = 2,
    pattern: str = 'residual',
    **kwargs,
) -> VisionModel:
    """Build a vision model with standard structure.
    
    Args:
        input_shape: Input tensor shape (C, H, W)
        num_classes: Number of output classes
        stem_channels: Stem output channels
        stage_channels: Channels for each stage
        blocks_per_stage: Blocks per stage (int or list)
        pattern: Block pattern for stages
        
    Returns:
        VisionModel instance
    """
    # Stem
    stem = Stem(channels=stem_channels, in_channels=input_shape[0])
    
    # Stages
    if isinstance(blocks_per_stage, int):
        blocks_per_stage = [blocks_per_stage] * len(stage_channels)
    
    stages = []
    in_channels = stem_channels
    
    for i, (channels, num_blocks) in enumerate(zip(stage_channels, blocks_per_stage)):
        downsample = (i > 0)  # Downsample after first stage
        stages.append(Stage(
            channels=channels,
            blocks=num_blocks,
            pattern=pattern,
            in_channels=in_channels,
            downsample=downsample,
            **kwargs,
        ))
        in_channels = channels
    
    # Head
    head = Head(num_classes=num_classes)
    
    return VisionModel(stem, stages, head)
