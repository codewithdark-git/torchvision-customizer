"""Head builder for network output.

The Head is the final classification/regression part of a network,
typically consisting of global pooling, flattening, and linear layers.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn

from torchvision_customizer.compose.operators import ComposableModule
from torchvision_customizer.layers import get_activation


class Head(ComposableModule):
    """Network head (classifier) builder.
    
    Creates the classification head with configurable layers.
    
    Args:
        num_classes: Number of output classes
        in_features: Input features (auto-detected if None)
        hidden: List of hidden layer sizes (optional)
        dropout: Dropout probability
        pool: Pooling type ('adaptive_avg', 'adaptive_max', 'flatten', None)
        activation: Activation for hidden layers
        
    Example:
        >>> head = Head(num_classes=1000)
        >>> # Creates: AdaptiveAvgPool -> Flatten -> Linear(->1000)
        
        >>> head = Head(num_classes=100, hidden=[512, 256], dropout=0.5)
        >>> # Creates: Pool -> Flatten -> Linear(->512) -> ReLU -> Dropout
        >>>          # -> Linear(->256) -> ReLU -> Dropout -> Linear(->100)
    """
    
    def __init__(
        self,
        num_classes: int,
        in_features: Optional[int] = None,
        hidden: Optional[List[int]] = None,
        dropout: float = 0.0,
        pool: str = 'adaptive_avg',
        activation: str = 'relu',
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.in_features = in_features
        self._out_channels = num_classes
        
        layers = []
        
        # Pooling
        if pool == 'adaptive_avg':
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif pool == 'adaptive_max':
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        
        # Flatten
        layers.append(nn.Flatten())
        
        # Store for lazy initialization if in_features not provided
        self._hidden = hidden or []
        self._dropout = dropout
        self._activation = activation
        self._pool_layers = nn.Sequential(*layers)
        self._classifier = None
        self._in_features = in_features
        
        # Build classifier if in_features is known
        if in_features is not None:
            self._classifier = self._build_classifier(in_features)
    
    def _build_classifier(self, in_features: int) -> nn.Sequential:
        """Build the classifier layers."""
        layers = []
        current_features = in_features
        
        # Hidden layers
        for hidden_size in self._hidden:
            layers.append(nn.Linear(current_features, hidden_size))
            layers.append(get_activation(self._activation))
            if self._dropout > 0:
                layers.append(nn.Dropout(self._dropout))
            current_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_features, self.num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply pooling and flatten
        x = self._pool_layers(x)
        
        # Lazy initialization of classifier
        if self._classifier is None:
            in_features = x.shape[1]
            self._classifier = self._build_classifier(in_features)
            self._classifier = self._classifier.to(x.device)
        
        return self._classifier(x)
    
    def __repr__(self) -> str:
        parts = []
        if self._hidden:
            parts.append(f"hidden={self._hidden}")
        parts.append(f"classes={self.num_classes}")
        if self._dropout > 0:
            parts.append(f"dropout={self._dropout}")
        return f"Head({', '.join(parts)})"


class SegmentationHead(ComposableModule):
    """Segmentation head for dense prediction tasks.
    
    Args:
        num_classes: Number of segmentation classes
        in_channels: Input channels
        hidden_channels: Hidden layer channels
        upsample_factor: Upsampling factor to original resolution
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: Optional[int] = None,
        hidden_channels: int = 256,
        upsample_factor: int = 1,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self._out_channels = num_classes
        self._in_channels = in_channels
        self._hidden_channels = hidden_channels
        self._upsample_factor = upsample_factor
        self._head = None
        
        if in_channels is not None:
            self._head = self._build_head(in_channels)
    
    def _build_head(self, in_channels: int) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_channels, self._hidden_channels, 3, padding=1),
            nn.BatchNorm2d(self._hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self._hidden_channels, self.num_classes, 1),
        ]
        
        if self._upsample_factor > 1:
            layers.append(nn.Upsample(
                scale_factor=self._upsample_factor,
                mode='bilinear',
                align_corners=False
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._head is None:
            self._head = self._build_head(x.shape[1])
            self._head = self._head.to(x.device)
        return self._head(x)
    
    def __repr__(self) -> str:
        return f"SegmentationHead(classes={self.num_classes})"


class DetectionHead(ComposableModule):
    """Detection head for object detection tasks.
    
    Args:
        num_classes: Number of detection classes
        num_anchors: Number of anchor boxes per position
        in_channels: Input channels
    """
    
    def __init__(
        self,
        num_classes: int,
        num_anchors: int = 9,
        in_channels: Optional[int] = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self._in_channels = in_channels
        
        # Detection head outputs: (class scores, bbox regression)
        # Per anchor: num_classes + 4 (x, y, w, h)
        self._out_per_anchor = num_classes + 4
        self._head = None
        
        if in_channels is not None:
            self._head = self._build_head(in_channels)
    
    def _build_head(self, in_channels: int) -> nn.Module:
        return nn.Conv2d(
            in_channels,
            self.num_anchors * self._out_per_anchor,
            kernel_size=3,
            padding=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._head is None:
            self._head = self._build_head(x.shape[1])
            self._head = self._head.to(x.device)
        
        # Output: (B, num_anchors * (num_classes + 4), H, W)
        return self._head(x)
    
    def __repr__(self) -> str:
        return f"DetectionHead(classes={self.num_classes}, anchors={self.num_anchors})"
