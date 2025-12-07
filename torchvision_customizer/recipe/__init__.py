"""Architecture Recipes: Declarative model definitions.

Recipes allow defining model architectures using human-readable strings
and dictionaries, similar to a "cooking recipe".

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
"""

from torchvision_customizer.recipe.definition import Recipe
from torchvision_customizer.recipe.parser import parse_recipe, parse_definition
from torchvision_customizer.recipe.builder import build_recipe, build_from_config

__all__ = [
    'Recipe',
    'build_recipe',
    'build_from_config',
    'parse_recipe',
    'parse_definition',
]
