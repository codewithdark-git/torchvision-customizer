"""Component Registry: Discover and access all building blocks.

The registry provides a unified interface to discover, inspect, and
instantiate any component in the torchvision-customizer package.

Example:
    >>> from torchvision_customizer import registry
    >>> registry.list()  # List all components
    ['conv', 'residual', 'se', 'inception', ...]
    
    >>> registry.list(category='attention')
    ['channel', 'spatial', 'cbam', 'multihead']
    
    >>> Conv = registry.get('conv')
    >>> block = Conv(in_channels=64, out_channels=128)
"""

from torchvision_customizer.registry.component_registry import (
    ComponentRegistry,
    register,
    get,
    list_components,
    info,
    categories,
)

# Global registry instance
_registry = ComponentRegistry()

# Re-export registry methods at module level
def list(category: str = None) -> list:
    """List all registered components, optionally filtered by category."""
    return _registry.list(category)

def get(name: str, **kwargs):
    """Get a component class or instance by name."""
    return _registry.get(name, **kwargs)

def info(name: str) -> dict:
    """Get detailed information about a component."""
    return _registry.info(name)

def categories() -> list:
    """List all available component categories."""
    return _registry.categories()

def register(name: str, category: str = 'other'):
    """Decorator to register a new component."""
    return _registry.register(name, category)

__all__ = [
    'ComponentRegistry',
    'register',
    'get', 
    'list',
    'info',
    'categories',
    '_registry',
]
