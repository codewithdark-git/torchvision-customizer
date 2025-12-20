"""Hybrid Builder: Build customized models from pre-trained backbones.

The HybridBuilder allows you to:
1. Load any torchvision pre-trained model
2. Apply patches (replace, wrap, inject blocks)
3. Modify the head for your task
4. Preserve maximum weights during customization

Example:
    >>> builder = HybridBuilder()
    >>> model = builder.from_torchvision(
    ...     "resnet50",
    ...     weights="IMAGENET1K_V2",
    ...     patches={
    ...         "layer3": {"wrap": {"type": "SEBlock", "params": {"reduction": 16}}},
    ...         "layer4": {"inject": {"type": "CBAMBlock"}},
    ...     },
    ...     num_classes=100,
    ... )
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import copy
import logging
import torch
import torch.nn as nn

from torchvision_customizer.hybrid.extractor import (
    extract_tiers,
    get_backbone_info,
    BACKBONE_CONFIGS,
)
from torchvision_customizer.hybrid.weight_utils import (
    partial_load,
    WeightLoadingReport,
)
from torchvision_customizer import registry

logger = logging.getLogger(__name__)


class PatchSpec:
    """Specification for a patch operation."""
    
    def __init__(
        self,
        operation: str,  # 'replace', 'wrap', 'inject', 'inject_after'
        block_type: str,
        params: Dict[str, Any] = None,
        position: str = 'after',  # 'before', 'after', 'replace'
    ):
        self.operation = operation
        self.block_type = block_type
        self.params = params or {}
        self.position = position
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PatchSpec':
        """Create PatchSpec from dictionary."""
        for op in ['replace', 'wrap', 'inject', 'inject_after']:
            if op in d:
                spec = d[op]
                if isinstance(spec, str):
                    # Simple form: {"wrap": "SEBlock"}
                    return cls(op, spec)
                elif isinstance(spec, dict):
                    # Full form: {"wrap": {"type": "SEBlock", "params": {...}}}
                    return cls(
                        op,
                        spec.get('type', spec.get('block', '')),
                        spec.get('params', {}),
                        spec.get('position', 'after'),
                    )
        
        raise ValueError(f"Invalid patch specification: {d}")


class HybridModel(nn.Module):
    """A hybrid model composed from a pre-trained backbone with custom modifications."""
    
    def __init__(
        self,
        stem: nn.Module,
        stages: nn.ModuleList,
        head: nn.Module,
        backbone_name: str,
        modifications: List[str] = None,
    ):
        super().__init__()
        self.stem = stem
        self.stages = stages
        self.head = head
        self.backbone_name = backbone_name
        self.modifications = modifications or []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classification head."""
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x
    
    def get_stage_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get intermediate outputs from each stage (useful for FPN, etc.)."""
        outputs = []
        x = self.stem(x)
        outputs.append(x)
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs
    
    def freeze_backbone(self, unfreeze_stages: List[int] = None) -> 'HybridModel':
        """Freeze backbone, optionally leaving some stages unfrozen.
        
        Args:
            unfreeze_stages: List of stage indices to keep trainable
            
        Returns:
            Self for chaining
        """
        # Freeze stem
        for param in self.stem.parameters():
            param.requires_grad = False
        
        # Freeze/unfreeze stages
        unfreeze_set = set(unfreeze_stages or [])
        for i, stage in enumerate(self.stages):
            freeze = i not in unfreeze_set
            for param in stage.parameters():
                param.requires_grad = not freeze
        
        return self
    
    def unfreeze_all(self) -> 'HybridModel':
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        return self
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def explain(self) -> str:
        """Generate human-readable model description."""
        lines = [
            "+" + "=" * 60 + "+",
            "|" + f" HybridModel ({self.backbone_name}) ".center(60) + "|",
            "+" + "=" * 60 + "+",
        ]
        
        # Stem
        stem_params = sum(p.numel() for p in self.stem.parameters())
        lines.append(f"| Stem: {stem_params:,} params".ljust(61) + "|")
        
        # Stages
        lines.append("|" + " Stages: ".ljust(60) + "|")
        for i, stage in enumerate(self.stages):
            stage_params = sum(p.numel() for p in stage.parameters())
            lines.append(f"|   Stage {i+1}: {stage_params:,} params".ljust(61) + "|")
        
        # Head
        head_params = sum(p.numel() for p in self.head.parameters())
        lines.append(f"| Head: {head_params:,} params".ljust(61) + "|")
        
        lines.append("+" + "-" * 60 + "+")
        
        # Modifications
        if self.modifications:
            lines.append("| Modifications:".ljust(61) + "|")
            for mod in self.modifications[:5]:
                lines.append(f"|   - {mod}".ljust(61) + "|")
            if len(self.modifications) > 5:
                lines.append(f"|   ... and {len(self.modifications) - 5} more".ljust(61) + "|")
            lines.append("+" + "-" * 60 + "+")
        
        # Totals
        total = self.count_parameters()
        trainable = self.count_parameters(trainable_only=True)
        lines.append(f"| Total: {total:,} | Trainable: {trainable:,}".ljust(61) + "|")
        lines.append("+" + "=" * 60 + "+")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (f"HybridModel(backbone='{self.backbone_name}', "
                f"stages={len(self.stages)}, "
                f"mods={len(self.modifications)})")


class HybridBuilder:
    """Builder for creating hybrid models from pre-trained backbones.
    
    The HybridBuilder provides a fluent API for:
    - Loading pre-trained torchvision models
    - Extracting and customizing model tiers
    - Applying patches (attention, custom blocks)
    - Replacing the classification head
    
    Example:
        >>> builder = HybridBuilder()
        >>> 
        >>> # Quick build
        >>> model = builder.from_torchvision(
        ...     "resnet50", 
        ...     weights="IMAGENET1K_V2",
        ...     num_classes=100
        ... )
        >>> 
        >>> # With patches
        >>> model = builder.from_torchvision(
        ...     "efficientnet_b4",
        ...     weights="IMAGENET1K_V1", 
        ...     patches={
        ...         "features.5": {"wrap": "SEBlock"},
        ...     },
        ...     num_classes=10,
        ...     dropout=0.3,
        ... )
    """
    
    # Supported backbone names
    SUPPORTED_BACKBONES = list(BACKBONE_CONFIGS.keys())
    
    def __init__(self):
        """Initialize HybridBuilder."""
        self._base_model: Optional[nn.Module] = None
        self._backbone_name: Optional[str] = None
        self._tiers: Optional[Dict] = None
        self._modifications: List[str] = []
    
    @property
    def supported_backbones(self) -> List[str]:
        """List of supported backbone names."""
        return self.SUPPORTED_BACKBONES
    
    def from_torchvision(
        self,
        backbone_name: str,
        weights: Optional[str] = "DEFAULT",
        patches: Optional[Dict[str, Dict]] = None,
        num_classes: int = 1000,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        unfreeze_stages: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> HybridModel:
        """Create a hybrid model from a torchvision backbone.
        
        Args:
            backbone_name: Name of torchvision model (e.g., 'resnet50')
            weights: Weights to load ('DEFAULT', 'IMAGENET1K_V1', 'IMAGENET1K_V2', None)
            patches: Dictionary of patches to apply
            num_classes: Number of output classes
            dropout: Dropout rate for classifier
            freeze_backbone: Whether to freeze backbone weights
            unfreeze_stages: If freezing, which stages to keep trainable
            verbose: Print loading information
            
        Returns:
            HybridModel instance
            
        Example:
            >>> model = builder.from_torchvision(
            ...     "resnet50",
            ...     weights="IMAGENET1K_V2",
            ...     patches={"layer3": {"wrap": "SEBlock"}},
            ...     num_classes=100,
            ... )
        """
        # Import torchvision
        try:
            from torchvision import models
        except ImportError:
            raise ImportError("torchvision is required for hybrid models")
        
        # Validate backbone name
        if not hasattr(models, backbone_name):
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Supported: {', '.join(self.SUPPORTED_BACKBONES[:10])}..."
            )
        
        # Load base model
        model_fn = getattr(models, backbone_name)
        
        if weights:
            # Try to get weights enum
            weights_enum = self._resolve_weights(backbone_name, weights)
            self._base_model = model_fn(weights=weights_enum)
            if verbose:
                print(f"Loaded {backbone_name} with {weights} weights")
        else:
            self._base_model = model_fn(weights=None)
            if verbose:
                print(f"Loaded {backbone_name} (random initialization)")
        
        self._backbone_name = backbone_name
        
        # Extract tiers
        self._tiers = extract_tiers(self._base_model, backbone_name)
        
        # Apply patches
        if patches:
            self._apply_patches(patches)
        
        # Build hybrid model
        model = self._build_hybrid_model(num_classes, dropout)
        
        # Freeze if requested
        if freeze_backbone:
            model.freeze_backbone(unfreeze_stages)
        
        return model
    
    def _resolve_weights(self, backbone_name: str, weights: str) -> Any:
        """Resolve weight string to torchvision weights enum."""
        from torchvision import models
        
        if weights == "DEFAULT":
            return "DEFAULT"
        
        # Try to find weights enum
        weights_name = f"{backbone_name.title().replace('_', '')}Weights" 
        
        # Handle special cases
        weights_map = {
            "resnet18": "ResNet18_Weights",
            "resnet34": "ResNet34_Weights",
            "resnet50": "ResNet50_Weights",
            "resnet101": "ResNet101_Weights",
            "resnet152": "ResNet152_Weights",
            "efficientnet_b0": "EfficientNet_B0_Weights",
            "efficientnet_b1": "EfficientNet_B1_Weights",
            "efficientnet_b2": "EfficientNet_B2_Weights",
            "efficientnet_b3": "EfficientNet_B3_Weights",
            "efficientnet_b4": "EfficientNet_B4_Weights",
            "efficientnet_b5": "EfficientNet_B5_Weights",
            "efficientnet_b6": "EfficientNet_B6_Weights",
            "efficientnet_b7": "EfficientNet_B7_Weights",
            "vgg16": "VGG16_Weights",
            "vgg19": "VGG19_Weights",
            "mobilenet_v2": "MobileNet_V2_Weights",
            "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
            "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
            "convnext_tiny": "ConvNeXt_Tiny_Weights",
            "convnext_small": "ConvNeXt_Small_Weights",
            "convnext_base": "ConvNeXt_Base_Weights",
            "convnext_large": "ConvNeXt_Large_Weights",
        }
        
        if backbone_name in weights_map:
            weights_enum_name = weights_map[backbone_name]
            if hasattr(models, weights_enum_name):
                weights_enum = getattr(models, weights_enum_name)
                if hasattr(weights_enum, weights):
                    return getattr(weights_enum, weights)
        
        # Fallback to string
        return weights
    
    def _apply_patches(self, patches: Dict[str, Dict]) -> None:
        """Apply patches to the extracted tiers."""
        for target, patch_dict in patches.items():
            patch_spec = PatchSpec.from_dict(patch_dict)
            self._apply_single_patch(target, patch_spec)
    
    def _apply_single_patch(self, target: str, patch: PatchSpec) -> None:
        """Apply a single patch to a target layer."""
        # Find the target module
        target_module, parent, attr_name = self._find_module(target)
        
        if target_module is None:
            logger.warning(f"Target '{target}' not found, skipping patch")
            return
        
        # Get the block class from registry
        try:
            BlockClass = registry.get(patch.block_type)
        except ValueError:
            logger.warning(f"Block type '{patch.block_type}' not in registry, skipping")
            return
        
        # Apply based on operation
        if patch.operation == 'replace':
            # Replace the module entirely
            new_module = self._create_replacement(target_module, BlockClass, patch.params)
            setattr(parent, attr_name, new_module)
            self._modifications.append(f"Replaced {target} with {patch.block_type}")
            
        elif patch.operation == 'wrap':
            # Wrap the module with attention/block
            new_module = self._wrap_module(target_module, BlockClass, patch.params)
            setattr(parent, attr_name, new_module)
            self._modifications.append(f"Wrapped {target} with {patch.block_type}")
            
        elif patch.operation in ['inject', 'inject_after']:
            # Inject block after the target
            new_module = self._inject_after(target_module, BlockClass, patch.params)
            setattr(parent, attr_name, new_module)
            self._modifications.append(f"Injected {patch.block_type} after {target}")
    
    def _find_module(self, target: str) -> Tuple[Optional[nn.Module], Optional[nn.Module], str]:
        """Find a module by dot-separated path."""
        parts = target.split('.')
        current = self._base_model
        parent = None
        attr_name = parts[-1]
        
        for i, part in enumerate(parts):
            if hasattr(current, part):
                parent = current
                current = getattr(current, part)
                attr_name = part
            elif part.isdigit() and isinstance(current, (nn.Sequential, nn.ModuleList)):
                idx = int(part)
                if idx < len(current):
                    parent = current
                    current = current[idx]
                    attr_name = str(idx)
                else:
                    return None, None, ""
            else:
                return None, None, ""
        
        return current, parent, attr_name
    
    def _create_replacement(
        self,
        original: nn.Module,
        BlockClass: Type[nn.Module],
        params: Dict[str, Any],
    ) -> nn.Module:
        """Create a replacement module with inferred channels."""
        # Try to infer channels from original
        in_channels = self._get_channels(original, 'in')
        out_channels = self._get_channels(original, 'out')
        
        # Build params
        init_params = dict(params)
        if 'in_channels' not in init_params and in_channels:
            init_params['in_channels'] = in_channels
        if 'out_channels' not in init_params and out_channels:
            init_params['out_channels'] = out_channels
        if 'channels' not in init_params and out_channels:
            init_params['channels'] = out_channels
        
        return BlockClass(**init_params)
    
    def _wrap_module(
        self,
        original: nn.Module,
        BlockClass: Type[nn.Module],
        params: Dict[str, Any],
    ) -> nn.Module:
        """Wrap a module with an attention/block layer."""
        out_channels = self._get_channels(original, 'out')
        
        # Build params for wrapper
        init_params = dict(params)
        if 'channels' not in init_params and out_channels:
            init_params['channels'] = out_channels
        if 'in_channels' not in init_params and out_channels:
            init_params['in_channels'] = out_channels
        
        wrapper = BlockClass(**init_params)
        
        # Return sequential: original -> wrapper
        return nn.Sequential(original, wrapper)
    
    def _inject_after(
        self,
        original: nn.Module,
        BlockClass: Type[nn.Module],
        params: Dict[str, Any],
    ) -> nn.Module:
        """Inject a block after the original module."""
        # Same as wrap for now
        return self._wrap_module(original, BlockClass, params)
    
    def _get_channels(self, module: nn.Module, which: str = 'out') -> Optional[int]:
        """Get input or output channels from a module."""
        if which == 'out':
            if hasattr(module, 'out_channels'):
                return module.out_channels
            # Check last child
            children = list(module.children())
            for child in reversed(children):
                ch = self._get_channels(child, 'out')
                if ch:
                    return ch
            if isinstance(module, nn.Conv2d):
                return module.out_channels
            if isinstance(module, nn.BatchNorm2d):
                return module.num_features
        else:  # 'in'
            if hasattr(module, 'in_channels'):
                return module.in_channels
            children = list(module.children())
            for child in children:
                ch = self._get_channels(child, 'in')
                if ch:
                    return ch
            if isinstance(module, nn.Conv2d):
                return module.in_channels
        
        return None
    
    def _build_hybrid_model(self, num_classes: int, dropout: float) -> HybridModel:
        """Build the final HybridModel."""
        # Get tiers
        stem = self._tiers['stem']
        stages = self._tiers['stages']
        original_head = self._tiers['head']
        
        # Determine feature dimension for new head
        feature_dim = self._infer_feature_dim()
        
        # Build new head
        head = self._build_head(feature_dim, num_classes, dropout, original_head)
        
        return HybridModel(
            stem=stem,
            stages=nn.ModuleList(stages),
            head=head,
            backbone_name=self._backbone_name,
            modifications=self._modifications.copy(),
        )
    
    def _infer_feature_dim(self) -> int:
        """Infer feature dimension from the model."""
        # Try to get from last stage
        stages = self._tiers['stages']
        if stages:
            last_stage = stages[-1]
            ch = self._get_channels(last_stage, 'out')
            if ch:
                return ch
        
        # Try from original head
        original_head = self._tiers['head']
        for module in original_head.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        
        # Default fallback
        return 2048
    
    def _build_head(
        self,
        feature_dim: int,
        num_classes: int,
        dropout: float,
        original_head: nn.Module,
    ) -> nn.Module:
        """Build classification head."""
        # Check if original head has pooling
        has_pool = any(
            isinstance(m, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AvgPool2d))
            for m in original_head.modules()
        )
        
        layers = []
        
        if has_pool:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        layers.append(nn.Flatten(1))
        
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        
        layers.append(nn.Linear(feature_dim, num_classes))
        
        return nn.Sequential(*layers)
    
    def from_checkpoint(
        self,
        checkpoint_path: str,
        backbone_name: str,
        patches: Optional[Dict[str, Dict]] = None,
        num_classes: int = 1000,
        dropout: float = 0.0,
        strict: bool = False,
    ) -> HybridModel:
        """Create hybrid model from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            backbone_name: Backbone architecture name
            patches: Patches to apply
            num_classes: Number of output classes
            dropout: Dropout rate
            strict: Whether to strictly match checkpoint keys
            
        Returns:
            HybridModel loaded from checkpoint
        """
        # First create model structure
        model = self.from_torchvision(
            backbone_name,
            weights=None,
            patches=patches,
            num_classes=num_classes,
            dropout=dropout,
            verbose=False,
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Clean up state dict keys (remove 'module.' prefix if present)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            cleaned_state_dict[k] = v
        
        # Load with partial matching
        partial_load(model, cleaned_state_dict, ignore_mismatch=not strict)
        
        return model
    
    @classmethod
    def list_backbones(cls) -> List[str]:
        """List all supported backbone architectures."""
        return cls.SUPPORTED_BACKBONES


# Convenience function
def create_hybrid(
    backbone: str,
    num_classes: int = 1000,
    weights: str = "DEFAULT",
    patches: Dict[str, Dict] = None,
    **kwargs,
) -> HybridModel:
    """Convenience function to create a hybrid model.
    
    Args:
        backbone: Backbone name (e.g., 'resnet50')
        num_classes: Number of output classes
        weights: Pretrained weights to load
        patches: Patches to apply
        **kwargs: Additional arguments for HybridBuilder
        
    Returns:
        HybridModel instance
        
    Example:
        >>> model = create_hybrid(
        ...     "resnet50",
        ...     num_classes=100,
        ...     patches={"layer3": {"wrap": "se"}}
        ... )
    """
    builder = HybridBuilder()
    return builder.from_torchvision(
        backbone,
        weights=weights,
        patches=patches,
        num_classes=num_classes,
        **kwargs,
    )

