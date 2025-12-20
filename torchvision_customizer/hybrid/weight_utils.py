"""Weight Utilities for Hybrid Models.

Provides utilities for partial weight loading, shape matching,
and weight transfer between models.

Features:
    - partial_load: Load weights with mismatch tolerance
    - match_state_dict: Find matching parameters between models
    - transfer_weights: Copy weights from source to target

Example:
    >>> partial_load(model, state_dict, ignore_mismatch=True)
    >>> matched = match_state_dict(source_dict, target_dict)
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class WeightLoadingReport:
    """Report of weight loading operation."""
    
    def __init__(self):
        self.loaded: List[str] = []
        self.skipped_shape_mismatch: List[Tuple[str, Tuple, Tuple]] = []
        self.skipped_missing: List[str] = []
        self.skipped_unexpected: List[str] = []
        self.newly_initialized: List[str] = []
    
    @property
    def total_loaded(self) -> int:
        return len(self.loaded)
    
    @property
    def total_skipped(self) -> int:
        return (len(self.skipped_shape_mismatch) + 
                len(self.skipped_missing) + 
                len(self.skipped_unexpected))
    
    @property
    def load_ratio(self) -> float:
        total = self.total_loaded + self.total_skipped
        return self.total_loaded / total if total > 0 else 0.0
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "Weight Loading Report",
            "=" * 60,
            f"Loaded:           {self.total_loaded} parameters",
            f"Shape Mismatch:   {len(self.skipped_shape_mismatch)} parameters",
            f"Missing (target): {len(self.skipped_missing)} parameters", 
            f"Unexpected:       {len(self.skipped_unexpected)} parameters",
            f"Newly Initialized:{len(self.newly_initialized)} parameters",
            "-" * 60,
            f"Load Ratio:       {self.load_ratio:.1%}",
            "=" * 60,
        ]
        
        if self.skipped_shape_mismatch:
            lines.append("\nShape Mismatches:")
            for name, src_shape, tgt_shape in self.skipped_shape_mismatch[:5]:
                lines.append(f"  {name}: {src_shape} -> {tgt_shape}")
            if len(self.skipped_shape_mismatch) > 5:
                lines.append(f"  ... and {len(self.skipped_shape_mismatch) - 5} more")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"WeightLoadingReport(loaded={self.total_loaded}, skipped={self.total_skipped})"


def partial_load(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    ignore_mismatch: bool = True,
    strict: bool = False,
    verbose: bool = True,
    init_new_layers: str = "kaiming",
) -> WeightLoadingReport:
    """Load weights with tolerance for mismatches.
    
    Handles shape mismatches, missing keys, and unexpected keys gracefully.
    New layers (not in state_dict) are initialized with specified method.
    
    Args:
        model: Target model to load weights into
        state_dict: Source state dictionary
        ignore_mismatch: If True, skip mismatched shapes instead of error
        strict: If True, raise error on any mismatch
        verbose: If True, print loading report
        init_new_layers: Initialization for new layers ('kaiming', 'xavier', 'zero')
        
    Returns:
        WeightLoadingReport with loading statistics
        
    Example:
        >>> from torchvision.models import resnet50, ResNet50_Weights
        >>> base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        >>> custom_model = MyCustomResNet()
        >>> report = partial_load(custom_model, base.state_dict())
        >>> print(report.summary())
    """
    report = WeightLoadingReport()
    model_state = model.state_dict()
    
    # Keys in both
    common_keys = set(state_dict.keys()) & set(model_state.keys())
    
    # Missing in source (new layers in target)
    missing_keys = set(model_state.keys()) - set(state_dict.keys())
    report.skipped_missing = list(missing_keys)
    
    # Unexpected in source
    unexpected_keys = set(state_dict.keys()) - set(model_state.keys())
    report.skipped_unexpected = list(unexpected_keys)
    
    # Load matching weights
    new_state_dict = {}
    
    for key in common_keys:
        src_tensor = state_dict[key]
        tgt_tensor = model_state[key]
        
        if src_tensor.shape == tgt_tensor.shape:
            new_state_dict[key] = src_tensor
            report.loaded.append(key)
        else:
            if ignore_mismatch:
                report.skipped_shape_mismatch.append(
                    (key, tuple(src_tensor.shape), tuple(tgt_tensor.shape))
                )
            elif strict:
                raise RuntimeError(
                    f"Shape mismatch for '{key}': "
                    f"source {src_tensor.shape} vs target {tgt_tensor.shape}"
                )
    
    # Load the matched weights
    model.load_state_dict(new_state_dict, strict=False)
    
    # Initialize new layers
    if missing_keys:
        _init_new_parameters(model, missing_keys, init_new_layers)
        report.newly_initialized = list(missing_keys)
    
    if verbose:
        print(report.summary())
    
    return report


def _init_new_parameters(
    model: nn.Module,
    param_names: Set[str],
    method: str = "kaiming"
) -> None:
    """Initialize new parameters that weren't loaded from checkpoint."""
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{name}.{param_name}" if name else param_name
            
            if full_name in param_names:
                if method == "kaiming":
                    if param.dim() >= 2:
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    else:
                        nn.init.zeros_(param)
                elif method == "xavier":
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.zeros_(param)
                elif method == "zero":
                    nn.init.zeros_(param)
                else:
                    logger.warning(f"Unknown init method '{method}', skipping {full_name}")


def match_state_dict(
    source_dict: Dict[str, torch.Tensor],
    target_dict: Dict[str, torch.Tensor],
    fuzzy_match: bool = False,
) -> Dict[str, str]:
    """Find matching parameters between two state dicts.
    
    Args:
        source_dict: Source state dictionary
        target_dict: Target state dictionary
        fuzzy_match: If True, attempt to match by shape when names differ
        
    Returns:
        Mapping from source keys to target keys
        
    Example:
        >>> source = pretrained_model.state_dict()
        >>> target = custom_model.state_dict()
        >>> mapping = match_state_dict(source, target)
    """
    mapping = {}
    
    # Exact name matches
    for src_key in source_dict:
        if src_key in target_dict:
            if source_dict[src_key].shape == target_dict[src_key].shape:
                mapping[src_key] = src_key
    
    if not fuzzy_match:
        return mapping
    
    # Fuzzy matching by shape (for unmatched keys)
    unmatched_source = set(source_dict.keys()) - set(mapping.keys())
    unmatched_target = set(target_dict.keys()) - set(mapping.values())
    
    # Group by shape
    target_by_shape: Dict[Tuple, List[str]] = {}
    for key in unmatched_target:
        shape = tuple(target_dict[key].shape)
        if shape not in target_by_shape:
            target_by_shape[shape] = []
        target_by_shape[shape].append(key)
    
    # Try to match by shape and name similarity
    for src_key in unmatched_source:
        shape = tuple(source_dict[src_key].shape)
        if shape in target_by_shape and target_by_shape[shape]:
            # Find best matching name
            candidates = target_by_shape[shape]
            best_match = _find_best_name_match(src_key, candidates)
            if best_match:
                mapping[src_key] = best_match
                target_by_shape[shape].remove(best_match)
    
    return mapping


def _find_best_name_match(src_key: str, candidates: List[str]) -> Optional[str]:
    """Find the best matching candidate based on name similarity."""
    src_parts = src_key.split('.')
    
    best_score = 0
    best_match = None
    
    for candidate in candidates:
        tgt_parts = candidate.split('.')
        
        # Count common parts
        score = sum(1 for p in src_parts if p in tgt_parts)
        
        if score > best_score:
            best_score = score
            best_match = candidate
    
    return best_match if best_score > 0 else None


def transfer_weights(
    source: nn.Module,
    target: nn.Module,
    layer_mapping: Optional[Dict[str, str]] = None,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> WeightLoadingReport:
    """Transfer weights from source model to target model.
    
    Args:
        source: Source model (e.g., pretrained)
        target: Target model (e.g., customized)
        layer_mapping: Optional mapping from source layer names to target
        include_patterns: Only transfer layers matching these patterns
        exclude_patterns: Skip layers matching these patterns
        
    Returns:
        WeightLoadingReport with transfer statistics
        
    Example:
        >>> pretrained = resnet50(weights='IMAGENET1K_V2')
        >>> custom = CustomResNet()
        >>> report = transfer_weights(pretrained, custom, 
        ...     exclude_patterns=['fc', 'classifier'])
    """
    import re
    
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    
    # Filter source keys
    filtered_dict = {}
    for key, value in source_dict.items():
        # Apply include patterns
        if include_patterns:
            if not any(re.search(p, key) for p in include_patterns):
                continue
        
        # Apply exclude patterns
        if exclude_patterns:
            if any(re.search(p, key) for p in exclude_patterns):
                continue
        
        # Apply layer mapping
        target_key = key
        if layer_mapping and key in layer_mapping:
            target_key = layer_mapping[key]
        
        if target_key in target_dict:
            filtered_dict[target_key] = value
    
    # Use partial_load for the actual transfer
    return partial_load(target, filtered_dict, verbose=False)


def get_layer_shapes(model: nn.Module) -> Dict[str, Tuple[int, ...]]:
    """Get shapes of all parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary mapping parameter names to shapes
    """
    return {name: tuple(param.shape) for name, param in model.named_parameters()}


def count_matching_params(
    source: nn.Module,
    target: nn.Module,
) -> Tuple[int, int, float]:
    """Count how many parameters can be transferred between models.
    
    Args:
        source: Source model
        target: Target model
        
    Returns:
        Tuple of (matching_params, total_source_params, match_ratio)
    """
    source_shapes = get_layer_shapes(source)
    target_shapes = get_layer_shapes(target)
    
    matching = 0
    total = 0
    
    for name, shape in source_shapes.items():
        total += 1
        if name in target_shapes and target_shapes[name] == shape:
            matching += 1
    
    ratio = matching / total if total > 0 else 0.0
    return matching, total, ratio

