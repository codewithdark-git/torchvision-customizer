"""Enhanced YAML Recipe Loader.

v2.1 features:
- Recipe inheritance (extends)
- Macro expansion (@macro)
- Hybrid backbone support
- JSONSchema validation

Example:
    >>> from torchvision_customizer.recipe import load_yaml_recipe
    >>> 
    >>> # Load and build model
    >>> model = load_yaml_recipe("my_recipe.yaml")
    >>> 
    >>> # Or get config only
    >>> config = load_yaml_config("my_recipe.yaml")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os
import torch.nn as nn

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from torchvision_customizer.recipe.schema import (
    validate_recipe_config,
    expand_macros,
    merge_recipes,
    get_template,
    RECIPE_TEMPLATES,
    ValidationError,
)


class RecipeLoadError(Exception):
    """Error loading recipe file."""
    pass


def load_yaml_config(
    path: Union[str, Path],
    validate: bool = True,
    expand: bool = True,
) -> Dict[str, Any]:
    """Load and process a YAML recipe configuration.
    
    Args:
        path: Path to YAML file
        validate: Whether to validate against schema
        expand: Whether to expand macros
        
    Returns:
        Processed recipe configuration
        
    Raises:
        RecipeLoadError: If file cannot be loaded
        ValidationError: If validation fails
    """
    if not HAS_YAML:
        raise ImportError("PyYAML is required for YAML recipes. Install with: pip install pyyaml")
    
    path = Path(path)
    if not path.exists():
        raise RecipeLoadError(f"Recipe file not found: {path}")
    
    # Load YAML
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RecipeLoadError(f"Invalid YAML in {path}: {e}")
    
    if config is None:
        raise RecipeLoadError(f"Empty recipe file: {path}")
    
    # Handle inheritance
    if 'extends' in config:
        config = _resolve_inheritance(config, path.parent)
    
    # Expand macros
    if expand and 'macros' in config:
        config = expand_macros(config)
    
    # Validate
    if validate:
        warnings = validate_recipe_config(config, strict=True)
        for warning in warnings:
            print(f"Warning: {warning}")
    
    return config


def _resolve_inheritance(
    config: Dict[str, Any],
    base_dir: Path,
) -> Dict[str, Any]:
    """Resolve recipe inheritance chain."""
    extends = config.get('extends')
    
    if extends is None:
        return config
    
    # Check if it's a built-in template
    if extends in RECIPE_TEMPLATES:
        base = get_template(extends)
    else:
        # Load from file
        base_path = base_dir / extends
        if not base_path.suffix:
            base_path = base_path.with_suffix('.yaml')
        
        if not base_path.exists():
            raise RecipeLoadError(f"Cannot find base recipe: {extends}")
        
        base = load_yaml_config(base_path, validate=False, expand=False)
    
    # Merge base with override
    return merge_recipes(base, config)


def load_yaml_recipe(
    path: Union[str, Path],
    num_classes: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Load a YAML recipe and build the model.
    
    Args:
        path: Path to YAML recipe file
        num_classes: Override number of classes
        **kwargs: Additional build arguments
        
    Returns:
        Built PyTorch model
        
    Example:
        >>> model = load_yaml_recipe("resnet_cifar.yaml", num_classes=10)
    """
    config = load_yaml_config(path)
    
    # Override num_classes if provided
    if num_classes is not None:
        if 'head' in config:
            if isinstance(config['head'], dict):
                config['head']['num_classes'] = num_classes
            else:
                config['head'] = {'num_classes': num_classes}
        elif 'backbone' in config:
            config['num_classes'] = num_classes
    
    # Build model based on type
    if 'backbone' in config:
        return _build_hybrid_model(config, **kwargs)
    else:
        return _build_recipe_model(config, **kwargs)


def _build_hybrid_model(config: Dict[str, Any], **kwargs) -> nn.Module:
    """Build a hybrid model from config."""
    from torchvision_customizer.hybrid import HybridBuilder
    
    backbone_config = config['backbone']
    
    if isinstance(backbone_config, str):
        # Parse string form: "resnet50(weights=IMAGENET1K_V2)"
        import re
        match = re.match(r'(\w+)(?:\((.+)\))?', backbone_config)
        if match:
            name = match.group(1)
            params_str = match.group(2)
            backbone_config = {'name': name}
            if params_str:
                # Parse simple key=value pairs
                for pair in params_str.split(','):
                    if '=' in pair:
                        k, v = pair.split('=', 1)
                        backbone_config[k.strip()] = v.strip()
    
    builder = HybridBuilder()
    
    return builder.from_torchvision(
        backbone_name=backbone_config['name'],
        weights=backbone_config.get('weights', 'DEFAULT'),
        patches=backbone_config.get('patches'),
        num_classes=config.get('num_classes', config.get('head', {}).get('num_classes', 1000)),
        dropout=config.get('head', {}).get('dropout', 0.0),
        **kwargs,
    )


def _build_recipe_model(config: Dict[str, Any], **kwargs) -> nn.Module:
    """Build a model from stem/stages/head config."""
    from torchvision_customizer.recipe import Recipe, build_recipe
    
    recipe = Recipe(
        stem=config.get('stem', 'conv(64)'),
        stages=config.get('stages', []),
        head=config.get('head', 'linear(1000)'),
        input_shape=tuple(config.get('input_shape', [3, 224, 224])),
    )
    
    return build_recipe(recipe)


def save_yaml_recipe(
    config: Dict[str, Any],
    path: Union[str, Path],
    include_metadata: bool = True,
) -> None:
    """Save a recipe configuration to YAML file.
    
    Args:
        config: Recipe configuration
        path: Output file path
        include_metadata: Whether to include schema version comment
    """
    if not HAS_YAML:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    content = ""
    if include_metadata:
        content = "# torchvision-customizer Recipe v2.1\n"
        content += "# Schema: https://github.com/codewithdark-git/torchvision-customizer#recipes\n\n"
    
    content += yaml.dump(config, default_flow_style=False, sort_keys=False)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def list_templates() -> List[str]:
    """List available recipe templates."""
    return list(RECIPE_TEMPLATES.keys())


def create_recipe_from_template(
    template_name: str,
    output_path: Optional[Union[str, Path]] = None,
    **overrides,
) -> Dict[str, Any]:
    """Create a new recipe from a template.
    
    Args:
        template_name: Name of the template
        output_path: Optional path to save the recipe
        **overrides: Values to override in the template
        
    Returns:
        Recipe configuration
    """
    config = get_template(template_name)
    
    # Apply overrides
    for key, value in overrides.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    
    if output_path:
        save_yaml_recipe(config, output_path)
    
    return config


# Example YAML recipe content for documentation
EXAMPLE_RECIPE_YAML = """# Example: Custom ResNet with Attention
# =====================================
# This recipe creates a ResNet-50 backbone with SE attention injected
# and a custom head for 100-class classification.

name: ResNet50-SE-Custom
version: "1.0.0"
description: ResNet50 with SE attention for custom classification

# Macros for reusable values
macros:
  attention: se
  activation: relu
  dropout: 0.3

# Use pretrained backbone
backbone:
  name: resnet50
  weights: IMAGENET1K_V2
  patches:
    layer3:
      wrap:
        type: "@attention"
        params:
          reduction: 16
    layer4:
      wrap:
        type: cbam_block

# Custom head
head:
  num_classes: 100
  dropout: "@dropout"

# Training hints (optional)
training:
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.01
  epochs: 100
  batch_size: 32
"""


def create_example_recipe(output_path: Union[str, Path]) -> None:
    """Create an example recipe file for reference.
    
    Args:
        output_path: Where to save the example
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(EXAMPLE_RECIPE_YAML)
    
    print(f"Created example recipe at: {path}")

