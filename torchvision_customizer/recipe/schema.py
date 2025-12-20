"""YAML Recipe Schema and Validation.

Provides JSONSchema-based validation for YAML recipe files,
along with utilities for schema generation and error reporting.

v2.1 Features:
- JSONSchema validation with helpful error messages
- Recipe inheritance (extend other recipes)
- Macro expansion (@attention, @block, etc.)
- Hybrid backbone support in recipes

Example:
    >>> from torchvision_customizer.recipe.schema import validate_recipe_config
    >>> config = yaml.safe_load(open("my_recipe.yaml"))
    >>> validate_recipe_config(config)
"""

from typing import Any, Dict, List, Optional, Union
import copy
import re


# JSONSchema for v2.1 recipes
RECIPE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "torchvision-customizer Recipe Schema v2.1",
    "type": "object",
    "properties": {
        # Metadata
        "name": {
            "type": "string",
            "description": "Recipe name for identification"
        },
        "version": {
            "type": "string",
            "pattern": r"^\d+\.\d+(\.\d+)?$",
            "description": "Recipe version (semver)"
        },
        "description": {
            "type": "string",
            "description": "Recipe description"
        },
        "extends": {
            "type": "string",
            "description": "Parent recipe to extend"
        },
        
        # Input configuration
        "input_shape": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 3,
            "maxItems": 3,
            "default": [3, 224, 224],
            "description": "Input tensor shape (C, H, W)"
        },
        
        # Macros for reuse
        "macros": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Macro definitions for @macro substitution"
        },
        
        # v2.1: Hybrid backbone support
        "backbone": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Torchvision backbone name (e.g., resnet50)"
                },
                "weights": {
                    "type": "string",
                    "description": "Pretrained weights (e.g., IMAGENET1K_V2)"
                },
                "patches": {
                    "type": "object",
                    "description": "Layer patches to apply"
                }
            },
            "required": ["name"]
        },
        
        # Architecture components (used when not using backbone)
        "stem": {
            "oneOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "channels": {"type": "integer", "minimum": 1},
                        "kernel_size": {"type": "integer", "minimum": 1},
                        "stride": {"type": "integer", "minimum": 1},
                        "activation": {"type": "string"},
                        "norm_type": {"type": "string"}
                    }
                }
            ],
            "description": "Stem configuration (string or object)"
        },
        
        "stages": {
            "type": "array",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "pattern": {"type": "string"},
                            "channels": {"type": "integer", "minimum": 1},
                            "blocks": {"type": "integer", "minimum": 1},
                            "downsample": {"type": "boolean"},
                            "activation": {"type": "string"},
                            "attention": {"type": "string"},
                            "drop_path": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                ]
            },
            "description": "List of stage configurations"
        },
        
        "head": {
            "oneOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "num_classes": {"type": "integer", "minimum": 1},
                        "dropout": {"type": "number", "minimum": 0, "maximum": 1},
                        "hidden_dims": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    }
                }
            ],
            "description": "Head/classifier configuration"
        },
        
        # Training hints (optional)
        "training": {
            "type": "object",
            "properties": {
                "optimizer": {"type": "string"},
                "learning_rate": {"type": "number"},
                "weight_decay": {"type": "number"},
                "epochs": {"type": "integer"},
                "batch_size": {"type": "integer"}
            }
        }
    },
    
    # Either backbone OR stem/stages/head
    "oneOf": [
        {"required": ["backbone"]},
        {"required": ["stem", "stages", "head"]}
    ]
}


class ValidationError(Exception):
    """Recipe validation error with detailed information."""
    
    def __init__(self, message: str, path: str = "", suggestion: str = ""):
        self.message = message
        self.path = path
        self.suggestion = suggestion
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        msg = f"Validation Error: {self.message}"
        if self.path:
            msg += f"\n  At: {self.path}"
        if self.suggestion:
            msg += f"\n  Suggestion: {self.suggestion}"
        return msg


def validate_recipe_config(config: Dict[str, Any], strict: bool = True) -> List[str]:
    """Validate a recipe configuration against the schema.
    
    Args:
        config: Recipe configuration dictionary
        strict: If True, raise on errors; if False, return warnings
        
    Returns:
        List of warning messages (empty if all OK)
        
    Raises:
        ValidationError: If strict and validation fails
    """
    warnings = []
    
    # Check required structure
    has_backbone = 'backbone' in config
    has_components = all(k in config for k in ['stem', 'stages', 'head'])
    
    if not has_backbone and not has_components:
        error = ValidationError(
            "Recipe must have either 'backbone' OR 'stem', 'stages', and 'head'",
            path="root",
            suggestion="Add backbone: {name: resnet50} or define stem/stages/head"
        )
        if strict:
            raise error
        warnings.append(str(error))
    
    # Validate backbone
    if has_backbone:
        backbone = config['backbone']
        if isinstance(backbone, str):
            # Simple string form: "resnet50(weights=IMAGENET1K_V2)"
            pass
        elif isinstance(backbone, dict):
            if 'name' not in backbone:
                error = ValidationError(
                    "Backbone must have 'name' field",
                    path="backbone",
                    suggestion="Add name: resnet50"
                )
                if strict:
                    raise error
                warnings.append(str(error))
    
    # Validate stages
    if 'stages' in config:
        stages = config['stages']
        if not isinstance(stages, list):
            error = ValidationError(
                "stages must be a list",
                path="stages",
                suggestion="stages should be [stage1, stage2, ...]"
            )
            if strict:
                raise error
            warnings.append(str(error))
        else:
            for i, stage in enumerate(stages):
                stage_warnings = _validate_stage(stage, i, strict)
                warnings.extend(stage_warnings)
    
    # Validate input_shape
    if 'input_shape' in config:
        shape = config['input_shape']
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            error = ValidationError(
                "input_shape must be [C, H, W]",
                path="input_shape",
                suggestion="Use input_shape: [3, 224, 224]"
            )
            if strict:
                raise error
            warnings.append(str(error))
    
    # Validate macros
    if 'macros' in config:
        if not isinstance(config['macros'], dict):
            error = ValidationError(
                "macros must be a dictionary",
                path="macros",
                suggestion="macros: {attention: SEBlock, block: residual}"
            )
            if strict:
                raise error
            warnings.append(str(error))
    
    return warnings


def _validate_stage(stage: Union[str, Dict], index: int, strict: bool) -> List[str]:
    """Validate a single stage configuration."""
    warnings = []
    path = f"stages[{index}]"
    
    if isinstance(stage, str):
        # String definition - will be parsed later
        return warnings
    
    if not isinstance(stage, dict):
        error = ValidationError(
            f"Stage must be string or dict, got {type(stage).__name__}",
            path=path
        )
        if strict:
            raise error
        warnings.append(str(error))
        return warnings
    
    # Check channels
    if 'channels' in stage:
        ch = stage['channels']
        if not isinstance(ch, int) or ch < 1:
            error = ValidationError(
                "channels must be positive integer",
                path=f"{path}.channels"
            )
            if strict:
                raise error
            warnings.append(str(error))
    
    # Check blocks
    if 'blocks' in stage:
        blocks = stage['blocks']
        if not isinstance(blocks, int) or blocks < 1:
            error = ValidationError(
                "blocks must be positive integer",
                path=f"{path}.blocks"
            )
            if strict:
                raise error
            warnings.append(str(error))
    
    return warnings


def expand_macros(config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand macros in recipe configuration.
    
    Replaces @macro patterns with their definitions.
    
    Args:
        config: Recipe configuration with macros
        
    Returns:
        Configuration with macros expanded
        
    Example:
        >>> config = {
        ...     'macros': {'attention': 'SEBlock'},
        ...     'stages': [{'attention': '@attention'}]
        ... }
        >>> expanded = expand_macros(config)
        >>> # expanded['stages'][0]['attention'] == 'SEBlock'
    """
    if 'macros' not in config:
        return config
    
    macros = config['macros']
    result = copy.deepcopy(config)
    
    def replace_in_value(value):
        if isinstance(value, str):
            # Check for @macro pattern
            if value.startswith('@'):
                macro_name = value[1:]
                if macro_name in macros:
                    return macros[macro_name]
            return value
        elif isinstance(value, dict):
            return {k: replace_in_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [replace_in_value(item) for item in value]
        return value
    
    # Expand macros in all fields except 'macros' itself
    for key, value in result.items():
        if key != 'macros':
            result[key] = replace_in_value(value)
    
    return result


def merge_recipes(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two recipe configurations (for inheritance).
    
    Args:
        base: Base recipe configuration
        override: Recipe to merge on top
        
    Returns:
        Merged configuration
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key == 'extends':
            # Don't include extends in result
            continue
        elif key == 'stages' and key in result:
            # For stages, replace or extend based on index
            if isinstance(value, list):
                result[key] = value  # Replace stages
        elif key == 'macros' and key in result:
            # Merge macros
            result[key] = {**result[key], **value}
        elif isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Deep merge dictionaries
            result[key] = merge_recipes(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def generate_schema_docs() -> str:
    """Generate human-readable documentation from schema."""
    lines = [
        "# Recipe Schema Documentation",
        "",
        "## Root Properties",
        ""
    ]
    
    props = RECIPE_SCHEMA.get('properties', {})
    for name, spec in props.items():
        desc = spec.get('description', 'No description')
        ptype = spec.get('type', spec.get('oneOf', 'mixed'))
        lines.append(f"### `{name}`")
        lines.append(f"- Type: `{ptype}`")
        lines.append(f"- Description: {desc}")
        if 'default' in spec:
            lines.append(f"- Default: `{spec['default']}`")
        lines.append("")
    
    return "\n".join(lines)


# Common recipe templates
RECIPE_TEMPLATES = {
    "resnet_base": {
        "name": "ResNet Base",
        "input_shape": [3, 224, 224],
        "stem": {"type": "conv", "channels": 64, "kernel_size": 7, "stride": 2},
        "stages": [
            {"pattern": "residual", "channels": 64, "blocks": 2},
            {"pattern": "residual", "channels": 128, "blocks": 2, "downsample": True},
            {"pattern": "residual", "channels": 256, "blocks": 2, "downsample": True},
            {"pattern": "residual", "channels": 512, "blocks": 2, "downsample": True},
        ],
        "head": {"type": "linear", "num_classes": 1000}
    },
    
    "efficientnet_base": {
        "name": "EfficientNet Base",
        "input_shape": [3, 224, 224],
        "macros": {"attention": "se", "activation": "swish"},
        "stem": {"type": "conv", "channels": 32, "kernel_size": 3, "stride": 2, "activation": "@activation"},
        "stages": [
            {"pattern": "mbconv", "channels": 16, "blocks": 1, "attention": "@attention"},
            {"pattern": "mbconv", "channels": 24, "blocks": 2, "downsample": True, "attention": "@attention"},
            {"pattern": "mbconv", "channels": 40, "blocks": 2, "downsample": True, "attention": "@attention"},
            {"pattern": "mbconv", "channels": 80, "blocks": 3, "downsample": True, "attention": "@attention"},
            {"pattern": "mbconv", "channels": 112, "blocks": 3, "attention": "@attention"},
            {"pattern": "mbconv", "channels": 192, "blocks": 4, "downsample": True, "attention": "@attention"},
            {"pattern": "mbconv", "channels": 320, "blocks": 1, "attention": "@attention"},
        ],
        "head": {"type": "linear", "num_classes": 1000, "dropout": 0.2}
    },
    
    "hybrid_resnet_se": {
        "name": "Hybrid ResNet with SE",
        "backbone": {
            "name": "resnet50",
            "weights": "IMAGENET1K_V2",
            "patches": {
                "layer3": {"wrap": "se"},
                "layer4": {"wrap": "cbam"}
            }
        },
        "head": {"num_classes": 100, "dropout": 0.3}
    }
}


def get_template(name: str) -> Dict[str, Any]:
    """Get a recipe template by name.
    
    Args:
        name: Template name
        
    Returns:
        Template configuration dictionary
    """
    if name not in RECIPE_TEMPLATES:
        raise ValueError(f"Unknown template '{name}'. Available: {list(RECIPE_TEMPLATES.keys())}")
    return copy.deepcopy(RECIPE_TEMPLATES[name])

