"""Component Registry implementation.

Provides a centralized registry for all neural network components,
enabling discovery, inspection, and instantiation.
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch.nn as nn


class ComponentInfo:
    """Information about a registered component."""
    
    def __init__(
        self,
        name: str,
        cls: Type[nn.Module],
        category: str,
        description: str = "",
        aliases: List[str] = None,
    ):
        self.name = name
        self.cls = cls
        self.category = category
        self.description = description or cls.__doc__ or ""
        self.aliases = aliases or []
    
    def __repr__(self) -> str:
        return f"ComponentInfo(name='{self.name}', category='{self.category}')"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        import inspect
        sig = inspect.signature(self.cls.__init__)
        params = {
            name: str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any'
            for name, param in sig.parameters.items()
            if name != 'self'
        }
        
        return {
            'name': self.name,
            'category': self.category,
            'description': self.description.split('\n')[0] if self.description else "",
            'parameters': params,
            'aliases': self.aliases,
        }


class ComponentRegistry:
    """Central registry for all neural network components.
    
    The registry allows:
    - Component discovery via list() and categories()
    - Component inspection via info()
    - Component instantiation via get()
    - Custom component registration via register()
    
    Example:
        >>> registry = ComponentRegistry()
        >>> registry.list()
        ['conv', 'residual', 'se', ...]
        
        >>> Conv = registry.get('conv')
        >>> block = Conv(64, 128)
    """
    
    def __init__(self):
        self._components: Dict[str, ComponentInfo] = {}
        self._aliases: Dict[str, str] = {}
        self._categories: Dict[str, List[str]] = {}
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization - register all built-in components."""
        if self._initialized:
            return
        self._initialized = True
        self._register_builtin_components()
    
    def _register_builtin_components(self):
        """Register all built-in components from blocks and layers."""
        # Import blocks
        from torchvision_customizer.blocks import (
            ConvBlock,
            ResidualBlock,
            SEBlock,
            DepthwiseBlock,
            InceptionModule,
            Conv3DBlock,
            SuperConv2d,
            SuperLinear,
            DenseConnectionBlock,
            StandardBottleneck,
            WideBottleneck,
            GroupedBottleneck,
            ResidualStage,
            ResidualSequence,
        )
        
        # Import layers
        from torchvision_customizer.layers import (
            ChannelAttention,
            SpatialAttention,
            ChannelSpatialAttention,
            MultiHeadAttention,
            AttentionBlock,
        )
        
        # Register blocks
        self._register_component('conv', ConvBlock, 'block', 
                                aliases=['conv2d', 'convblock'])
        self._register_component('conv3d', Conv3DBlock, 'block')
        self._register_component('residual', ResidualBlock, 'block',
                                aliases=['res', 'resblock'])
        self._register_component('se', SEBlock, 'block',
                                aliases=['squeeze_excitation', 'seblock'])
        self._register_component('depthwise', DepthwiseBlock, 'block',
                                aliases=['dw', 'depthwise_separable'])
        self._register_component('inception', InceptionModule, 'block',
                                aliases=['inception_module'])
        self._register_component('dense', DenseConnectionBlock, 'block',
                                aliases=['dense_block', 'densenet'])
        self._register_component('super_conv', SuperConv2d, 'block')
        self._register_component('super_linear', SuperLinear, 'block')
        
        # Register bottleneck variants
        self._register_component('bottleneck', StandardBottleneck, 'bottleneck',
                                aliases=['standard_bottleneck'])
        self._register_component('wide_bottleneck', WideBottleneck, 'bottleneck')
        self._register_component('grouped_bottleneck', GroupedBottleneck, 'bottleneck')
        
        # Register architecture components
        self._register_component('residual_stage', ResidualStage, 'architecture')
        self._register_component('residual_sequence', ResidualSequence, 'architecture')
        
        # Register attention components
        self._register_component('channel_attention', ChannelAttention, 'attention',
                                aliases=['ca', 'channel'])
        self._register_component('spatial_attention', SpatialAttention, 'attention',
                                aliases=['sa', 'spatial'])
        self._register_component('cbam', ChannelSpatialAttention, 'attention',
                                aliases=['channel_spatial'])
        self._register_component('multihead_attention', MultiHeadAttention, 'attention',
                                aliases=['mha', 'multihead'])
        self._register_component('attention_block', AttentionBlock, 'attention')
    
    def _register_component(
        self,
        name: str,
        cls: Type[nn.Module],
        category: str,
        description: str = "",
        aliases: List[str] = None,
    ):
        """Internal method to register a component."""
        info = ComponentInfo(name, cls, category, description, aliases)
        self._components[name] = info
        
        # Register in category
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)
        
        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name
    
    def register(self, name: str, category: str = 'custom', aliases: List[str] = None):
        """Decorator to register a custom component.
        
        Example:
            >>> @registry.register('my_block', category='block')
            ... class MyBlock(nn.Module):
            ...     def __init__(self, channels):
            ...         super().__init__()
            ...         self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        """
        def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
            self._ensure_initialized()
            self._register_component(name, cls, category, aliases=aliases)
            return cls
        return decorator
    
    def get(self, name: str, **kwargs) -> Union[Type[nn.Module], nn.Module]:
        """Get a component class by name, or instantiate if kwargs provided.
        
        Args:
            name: Component name or alias
            **kwargs: If provided, instantiate the component with these args
            
        Returns:
            Component class if no kwargs, instantiated module if kwargs provided
            
        Example:
            >>> Conv = registry.get('conv')  # Get class
            >>> block = registry.get('conv', in_channels=64, out_channels=128)  # Get instance
        """
        self._ensure_initialized()
        
        # Resolve alias
        if name in self._aliases:
            name = self._aliases[name]
        
        if name not in self._components:
            available = ', '.join(sorted(self._components.keys()))
            raise ValueError(f"Unknown component '{name}'. Available: {available}")
        
        cls = self._components[name].cls
        
        if kwargs:
            return cls(**kwargs)
        return cls
    
    def list(self, category: str = None) -> List[str]:
        """List all registered component names.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of component names
        """
        self._ensure_initialized()
        
        if category:
            if category not in self._categories:
                return []
            return sorted(self._categories[category])
        
        return sorted(self._components.keys())
    
    def categories(self) -> List[str]:
        """List all available categories."""
        self._ensure_initialized()
        return sorted(self._categories.keys())
    
    def info(self, name: str) -> dict:
        """Get detailed information about a component.
        
        Args:
            name: Component name or alias
            
        Returns:
            Dictionary with component information
        """
        self._ensure_initialized()
        
        # Resolve alias
        if name in self._aliases:
            name = self._aliases[name]
        
        if name not in self._components:
            raise ValueError(f"Unknown component '{name}'")
        
        return self._components[name].to_dict()
    
    def __repr__(self) -> str:
        self._ensure_initialized()
        return f"ComponentRegistry({len(self._components)} components in {len(self._categories)} categories)"


# Create default instance for module-level access
_default_registry = ComponentRegistry()

def register(name: str, category: str = 'custom', aliases: List[str] = None):
    """Decorator to register a component with the default registry."""
    return _default_registry.register(name, category, aliases)

def get(name: str, **kwargs):
    """Get component from default registry."""
    return _default_registry.get(name, **kwargs)

def list_components(category: str = None) -> List[str]:
    """List components from default registry."""
    return _default_registry.list(category)

def info(name: str) -> dict:
    """Get component info from default registry."""
    return _default_registry.info(name)

def categories() -> List[str]:
    """List categories from default registry."""
    return _default_registry.categories()
