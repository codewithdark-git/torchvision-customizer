"""Architecture Recipes: Declarative model definitions.

Recipes allow defining model architectures using human-readable strings
and dictionaries, similar to a "cooking recipe".

v2.1 Features:
    - YAML recipe loading with validation
    - Recipe inheritance (extends)
    - Macro expansion (@macro)
    - Hybrid backbone support

Example:
    >>> from torchvision_customizer.recipe import Recipe, build_recipe
    
    >>> my_recipe = Recipe(
    ...     stem="conv(64, kernel=7, stride=2)",
    ...     stages=[
    ...         "residual(64) x 2",
    ...         "residual(128) x 2 | downsample",
    ...         "residual(256) x 2 | downsample",
    ...     ],
    ...     head="linear(1000)"
    ... )
    
    >>> model = build_recipe(my_recipe)
    
    # v2.1: Load from YAML
    >>> from torchvision_customizer.recipe import load_yaml_recipe
    >>> model = load_yaml_recipe("my_model.yaml")
"""

from torchvision_customizer.recipe.definition import Recipe
from torchvision_customizer.recipe.parser import parse_recipe, parse_definition
from torchvision_customizer.recipe.builder import build_recipe, build_from_config

# v2.1: Schema and validation
from torchvision_customizer.recipe.schema import (
    validate_recipe_config,
    expand_macros,
    merge_recipes,
    get_template,
    RECIPE_SCHEMA,
    ValidationError,
)

# v2.1: YAML loading
from torchvision_customizer.recipe.yaml_loader import (
    load_yaml_config,
    load_yaml_recipe,
    save_yaml_recipe,
    list_templates,
    create_recipe_from_template,
    create_example_recipe,
)

__all__ = [
    # Core
    'Recipe',
    'build_recipe',
    'build_from_config',
    'parse_recipe',
    'parse_definition',
    
    # v2.1: Schema and validation
    'validate_recipe_config',
    'expand_macros',
    'merge_recipes',
    'get_template',
    'RECIPE_SCHEMA',
    'ValidationError',
    
    # v2.1: YAML loading
    'load_yaml_config',
    'load_yaml_recipe',
    'save_yaml_recipe',
    'list_templates',
    'create_recipe_from_template',
    'create_example_recipe',
]
