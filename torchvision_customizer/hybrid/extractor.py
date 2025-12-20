"""Backbone Extractor: Extract and analyze torchvision model structures.

Provides utilities to decompose pre-trained models into their
constituent parts (stem, stages, head) for customization.

Example:
    >>> from torchvision.models import resnet50
    >>> from torchvision_customizer.hybrid import extract_tiers
    >>> 
    >>> model = resnet50()
    >>> tiers = extract_tiers(model)
    >>> print(tiers.keys())  # ['stem', 'stages', 'head']
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch
import torch.nn as nn


# Backbone configurations for supported architectures
BACKBONE_CONFIGS = {
    # ResNet family
    "resnet18": {"stages": ["layer1", "layer2", "layer3", "layer4"], "stem": ["conv1", "bn1", "relu", "maxpool"], "head": ["avgpool", "fc"]},
    "resnet34": {"stages": ["layer1", "layer2", "layer3", "layer4"], "stem": ["conv1", "bn1", "relu", "maxpool"], "head": ["avgpool", "fc"]},
    "resnet50": {"stages": ["layer1", "layer2", "layer3", "layer4"], "stem": ["conv1", "bn1", "relu", "maxpool"], "head": ["avgpool", "fc"]},
    "resnet101": {"stages": ["layer1", "layer2", "layer3", "layer4"], "stem": ["conv1", "bn1", "relu", "maxpool"], "head": ["avgpool", "fc"]},
    "resnet152": {"stages": ["layer1", "layer2", "layer3", "layer4"], "stem": ["conv1", "bn1", "relu", "maxpool"], "head": ["avgpool", "fc"]},
    
    # Wide ResNet
    "wide_resnet50_2": {"stages": ["layer1", "layer2", "layer3", "layer4"], "stem": ["conv1", "bn1", "relu", "maxpool"], "head": ["avgpool", "fc"]},
    "wide_resnet101_2": {"stages": ["layer1", "layer2", "layer3", "layer4"], "stem": ["conv1", "bn1", "relu", "maxpool"], "head": ["avgpool", "fc"]},
    
    # ResNeXt
    "resnext50_32x4d": {"stages": ["layer1", "layer2", "layer3", "layer4"], "stem": ["conv1", "bn1", "relu", "maxpool"], "head": ["avgpool", "fc"]},
    "resnext101_32x8d": {"stages": ["layer1", "layer2", "layer3", "layer4"], "stem": ["conv1", "bn1", "relu", "maxpool"], "head": ["avgpool", "fc"]},
    "resnext101_64x4d": {"stages": ["layer1", "layer2", "layer3", "layer4"], "stem": ["conv1", "bn1", "relu", "maxpool"], "head": ["avgpool", "fc"]},
    
    # EfficientNet family
    "efficientnet_b0": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "efficientnet_b1": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "efficientnet_b2": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "efficientnet_b3": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "efficientnet_b4": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "efficientnet_b5": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "efficientnet_b6": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "efficientnet_b7": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "efficientnet_v2_s": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "efficientnet_v2_m": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "efficientnet_v2_l": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    
    # VGG family
    "vgg11": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "vgg13": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "vgg16": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "vgg19": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "vgg11_bn": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "vgg13_bn": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "vgg16_bn": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "vgg19_bn": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    
    # DenseNet family
    "densenet121": {"stages": ["features"], "stem": [], "head": ["classifier"]},
    "densenet169": {"stages": ["features"], "stem": [], "head": ["classifier"]},
    "densenet201": {"stages": ["features"], "stem": [], "head": ["classifier"]},
    "densenet161": {"stages": ["features"], "stem": [], "head": ["classifier"]},
    
    # MobileNet family
    "mobilenet_v2": {"stages": ["features"], "stem": [], "head": ["classifier"]},
    "mobilenet_v3_small": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "mobilenet_v3_large": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    
    # ConvNeXt family  
    "convnext_tiny": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "convnext_small": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "convnext_base": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    "convnext_large": {"stages": ["features"], "stem": [], "head": ["avgpool", "classifier"]},
    
    # Vision Transformer (partial support)
    "vit_b_16": {"stages": ["encoder"], "stem": ["conv_proj", "class_token", "encoder.pos_embedding"], "head": ["heads"]},
    "vit_b_32": {"stages": ["encoder"], "stem": ["conv_proj", "class_token", "encoder.pos_embedding"], "head": ["heads"]},
    "vit_l_16": {"stages": ["encoder"], "stem": ["conv_proj", "class_token", "encoder.pos_embedding"], "head": ["heads"]},
    "vit_l_32": {"stages": ["encoder"], "stem": ["conv_proj", "class_token", "encoder.pos_embedding"], "head": ["heads"]},
    
    # Swin Transformer
    "swin_t": {"stages": ["features"], "stem": [], "head": ["avgpool", "head"]},
    "swin_s": {"stages": ["features"], "stem": [], "head": ["avgpool", "head"]},
    "swin_b": {"stages": ["features"], "stem": [], "head": ["avgpool", "head"]},
}


class BackboneInfo:
    """Information about a backbone architecture."""
    
    def __init__(
        self,
        name: str,
        model: nn.Module,
        stem_parts: List[str],
        stage_parts: List[str],
        head_parts: List[str],
    ):
        self.name = name
        self.model = model
        self.stem_parts = stem_parts
        self.stage_parts = stage_parts
        self.head_parts = head_parts
        
        # Analyze model structure
        self._analyze_model()
    
    def _analyze_model(self) -> None:
        """Analyze the model structure."""
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        # Get output channels from stages
        self.stage_channels = []
        for stage_name in self.stage_parts:
            if hasattr(self.model, stage_name):
                stage = getattr(self.model, stage_name)
                self.stage_channels.append(_get_out_channels(stage))
    
    def __repr__(self) -> str:
        return (f"BackboneInfo(name='{self.name}', "
                f"params={self.total_params:,}, "
                f"stages={len(self.stage_parts)})")
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Backbone: {self.name}",
            "-" * 40,
            f"Stem:   {', '.join(self.stem_parts) or 'N/A'}",
            f"Stages: {', '.join(self.stage_parts)}",
            f"Head:   {', '.join(self.head_parts)}",
            "-" * 40,
            f"Parameters: {self.total_params:,}",
            f"Stage Channels: {self.stage_channels}",
        ]
        return "\n".join(lines)


def _get_out_channels(module: nn.Module) -> Optional[int]:
    """Try to determine output channels of a module."""
    # Check direct attribute
    if hasattr(module, 'out_channels'):
        return module.out_channels
    
    # Check last child
    children = list(module.children())
    if children:
        last = children[-1]
        if hasattr(last, 'out_channels'):
            return last.out_channels
        # Recurse
        return _get_out_channels(last)
    
    # Check for Conv2d
    if isinstance(module, nn.Conv2d):
        return module.out_channels
    
    return None


def get_backbone_info(
    model: nn.Module,
    backbone_name: Optional[str] = None,
) -> BackboneInfo:
    """Get structural information about a backbone model.
    
    Args:
        model: The backbone model
        backbone_name: Optional name hint for lookup
        
    Returns:
        BackboneInfo with structural details
    """
    # Try to determine backbone type
    if backbone_name is None:
        backbone_name = _infer_backbone_name(model)
    
    if backbone_name and backbone_name in BACKBONE_CONFIGS:
        config = BACKBONE_CONFIGS[backbone_name]
        return BackboneInfo(
            name=backbone_name,
            model=model,
            stem_parts=config.get("stem", []),
            stage_parts=config.get("stages", []),
            head_parts=config.get("head", []),
        )
    
    # Fallback: auto-detect structure
    return _auto_detect_structure(model, backbone_name or "unknown")


def _infer_backbone_name(model: nn.Module) -> Optional[str]:
    """Try to infer backbone name from model class."""
    class_name = type(model).__name__.lower()
    
    # Check common patterns
    for name in BACKBONE_CONFIGS:
        if name.replace("_", "") in class_name.replace("_", ""):
            return name
    
    return None


def _auto_detect_structure(model: nn.Module, name: str) -> BackboneInfo:
    """Auto-detect model structure when not in config."""
    stem_parts = []
    stage_parts = []
    head_parts = []
    
    for child_name, child in model.named_children():
        child_name_lower = child_name.lower()
        
        # Classify by name
        if any(s in child_name_lower for s in ['stem', 'conv1', 'bn1']):
            stem_parts.append(child_name)
        elif any(s in child_name_lower for s in ['layer', 'stage', 'features', 'blocks', 'encoder']):
            stage_parts.append(child_name)
        elif any(s in child_name_lower for s in ['fc', 'classifier', 'head', 'avgpool', 'pool']):
            head_parts.append(child_name)
        else:
            # Check by position and type
            if isinstance(child, (nn.Linear, nn.AdaptiveAvgPool2d)):
                head_parts.append(child_name)
            else:
                stage_parts.append(child_name)
    
    return BackboneInfo(
        name=name,
        model=model,
        stem_parts=stem_parts,
        stage_parts=stage_parts,
        head_parts=head_parts,
    )


def extract_tiers(
    model: nn.Module,
    backbone_name: Optional[str] = None,
) -> Dict[str, Union[nn.Module, List[nn.Module]]]:
    """Extract model into stem, stages, and head tiers.
    
    Args:
        model: The model to decompose
        backbone_name: Optional name for lookup
        
    Returns:
        Dictionary with 'stem', 'stages', and 'head' keys
        
    Example:
        >>> from torchvision.models import resnet50
        >>> model = resnet50()
        >>> tiers = extract_tiers(model)
        >>> print(type(tiers['stem']))  # nn.Sequential
        >>> print(len(tiers['stages']))  # 4
    """
    info = get_backbone_info(model, backbone_name)
    
    # Build stem
    stem_modules = []
    for part_name in info.stem_parts:
        if hasattr(model, part_name):
            stem_modules.append(getattr(model, part_name))
    stem = nn.Sequential(*stem_modules) if stem_modules else nn.Identity()
    
    # Build stages (can be single module or list)
    stages = []
    for part_name in info.stage_parts:
        if hasattr(model, part_name):
            stage = getattr(model, part_name)
            # Check if it's already a Sequential of stages
            if isinstance(stage, nn.Sequential) and part_name == "features":
                # EfficientNet/VGG style - features contains everything
                stages.append(stage)
            else:
                stages.append(stage)
    
    # Build head
    head_modules = []
    for part_name in info.head_parts:
        if hasattr(model, part_name):
            head_modules.append(getattr(model, part_name))
    head = nn.Sequential(*head_modules) if head_modules else nn.Identity()
    
    return {
        "stem": stem,
        "stages": stages,
        "head": head,
        "info": info,
    }


def get_stage_channels(model: nn.Module, backbone_name: Optional[str] = None) -> List[int]:
    """Get output channels for each stage.
    
    Args:
        model: The backbone model
        backbone_name: Optional name hint
        
    Returns:
        List of output channels per stage
    """
    info = get_backbone_info(model, backbone_name)
    return info.stage_channels


def freeze_backbone(
    model: nn.Module,
    freeze_stem: bool = True,
    freeze_stages: Union[bool, List[int]] = True,
    freeze_head: bool = False,
    backbone_name: Optional[str] = None,
) -> nn.Module:
    """Freeze parts of the backbone for fine-tuning.
    
    Args:
        model: The model to freeze
        freeze_stem: Whether to freeze stem
        freeze_stages: True to freeze all, or list of stage indices
        freeze_head: Whether to freeze head
        backbone_name: Optional name hint
        
    Returns:
        The model with frozen parameters
        
    Example:
        >>> model = resnet50(weights='IMAGENET1K_V2')
        >>> # Freeze all but last stage and head
        >>> freeze_backbone(model, freeze_stages=[0, 1, 2])
    """
    tiers = extract_tiers(model, backbone_name)
    
    # Freeze stem
    if freeze_stem:
        for param in tiers['stem'].parameters():
            param.requires_grad = False
    
    # Freeze stages
    if freeze_stages is True:
        for stage in tiers['stages']:
            for param in stage.parameters():
                param.requires_grad = False
    elif isinstance(freeze_stages, list):
        for i, stage in enumerate(tiers['stages']):
            if i in freeze_stages:
                for param in stage.parameters():
                    param.requires_grad = False
    
    # Freeze head
    if freeze_head:
        for param in tiers['head'].parameters():
            param.requires_grad = False
    
    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """Unfreeze all parameters in the model.
    
    Args:
        model: The model to unfreeze
        
    Returns:
        The model with all parameters unfrozen
    """
    for param in model.parameters():
        param.requires_grad = True
    return model

