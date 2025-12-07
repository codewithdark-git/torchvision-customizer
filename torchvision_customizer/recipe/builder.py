"""Recipe builder.

Builds PyTorch models from parsed recipes.
"""

from typing import Any, Dict, List, Optional, Union
import torch.nn as nn

from torchvision_customizer.recipe.definition import Recipe
from torchvision_customizer.recipe.parser import parse_recipe
from torchvision_customizer.compose import (
    Stem,
    Stage,
    Head,
    Compose,
    VisionModel,
)


def build_recipe(recipe: Recipe) -> VisionModel:
    """Build a model from a Recipe object.
    
    Args:
        recipe: Recipe definition
        
    Returns:
        Built VisionModel
    """
    config = parse_recipe(recipe)
    
    # 1. Build Stem
    stem_conf = config['stem']
    stem_type = stem_conf.pop('type', 'conv') # 'conv' maps to Stem
    
    if stem_type in ['conv', 'stem', 'standard']:
        # Ensure input channels from recipe
        if 'in_channels' not in stem_conf:
            stem_conf['in_channels'] = config['input_shape'][0]
        stem = Stem(**stem_conf)
    else:
        # Fallback for other stem types if implemented (e.g., SimpleStem)
        # For now, map everything to Stem or raise error
        if stem_type == 'simple':
            from torchvision_customizer.compose.stem import SimpleStem
            stem = SimpleStem(**stem_conf)
        else:
             stem = Stem(**stem_conf) # Default fallback
            
    # 2. Build Stages
    stages = []
    current_in_channels = stem.out_channels
    
    for i, stage_conf in enumerate(config['stages']):
        stage_type = stage_conf.pop('type', 'residual')
        
        # If type is a pattern name, use it as pattern
        if 'pattern' not in stage_conf:
            stage_conf['pattern'] = stage_type
            
        # Ensure in_channels (Stage tracks it, but good to be explicit)
        if 'in_channels' not in stage_conf:
            stage_conf['in_channels'] = current_in_channels
            
        stage = Stage(**stage_conf)
        stages.append(stage)
        current_in_channels = stage.out_channels
        
    # 3. Build Head
    head_conf = config['head']
    head_type = head_conf.pop('type', 'linear')
    
    # Pass input features from last stage if not specified
    if 'in_features' not in head_conf:
        head_conf['in_features'] = current_in_channels
        
    head = Head(**head_conf)
    
    return VisionModel(stem, stages, head)


def build_from_config(config: Dict[str, Any]) -> VisionModel:
    """Build a model from a raw configuration dictionary."""
    recipe = Recipe.from_dict(config)
    return build_recipe(recipe)
