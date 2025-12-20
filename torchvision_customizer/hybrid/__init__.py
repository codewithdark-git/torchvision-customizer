"""Hybrid Module: Pre-trained model customization.

Load torchvision pre-trained models and customize them with
custom blocks, attention mechanisms, and architectural modifications.

Example:
    >>> from torchvision_customizer.hybrid import HybridBuilder
    >>> builder = HybridBuilder()
    >>> model = builder.from_torchvision(
    ...     "resnet50",
    ...     weights="IMAGENET1K_V2",
    ...     patches={"layer3": {"wrap": "SEBlock"}},
    ...     num_classes=100
    ... )
"""

from torchvision_customizer.hybrid.builder import HybridBuilder
from torchvision_customizer.hybrid.weight_utils import (
    partial_load,
    match_state_dict,
    transfer_weights,
)
from torchvision_customizer.hybrid.extractor import (
    extract_tiers,
    get_backbone_info,
)

__all__ = [
    "HybridBuilder",
    "partial_load",
    "match_state_dict",
    "transfer_weights",
    "extract_tiers",
    "get_backbone_info",
]

