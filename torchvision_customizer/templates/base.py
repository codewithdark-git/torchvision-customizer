"""Base Template class for architecture customization.

Provides the foundation for creating customizable model templates
that can be modified before building.
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union
from copy import deepcopy
import torch.nn as nn


class Template:
    """Base class for architecture templates with customization.
    
    Templates allow you to start from a known architecture and
    customize it before building.
    
    Example:
        >>> template = Template.resnet(layers=50)
        >>> template.replace_activation('gelu')
        >>> template.add_attention('se', after='conv')
        >>> model = template.build(num_classes=100)
    """
    
    # Registry of available templates
    _templates: Dict[str, Type['Template']] = {}
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize template with configuration.
        
        Args:
            config: Architecture configuration dictionary
        """
        self._config = deepcopy(config)
        self._modifications = []
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a template class."""
        def decorator(template_cls: Type['Template']) -> Type['Template']:
            cls._templates[name] = template_cls
            return template_cls
        return decorator
    
    @classmethod
    def from_name(cls, name: str, **kwargs) -> 'Template':
        """Create a template by name.
        
        Args:
            name: Template name (e.g., 'resnet', 'vgg')
            **kwargs: Template-specific arguments
            
        Returns:
            Template instance
        """
        if name not in cls._templates:
            available = ', '.join(sorted(cls._templates.keys()))
            raise ValueError(f"Unknown template '{name}'. Available: {available}")
        
        return cls._templates[name](**kwargs)
    
    @classmethod
    def resnet(cls, layers: int = 50, **kwargs) -> 'Template':
        """Create a ResNet template."""
        from torchvision_customizer.templates.resnet import ResNetTemplate
        return ResNetTemplate(layers=layers, **kwargs)
    
    @classmethod
    def vgg(cls, layers: int = 16, **kwargs) -> 'Template':
        """Create a VGG template."""
        from torchvision_customizer.templates.vgg import VGGTemplate
        return VGGTemplate(layers=layers, **kwargs)
    
    @classmethod
    def mobilenet(cls, version: int = 2, **kwargs) -> 'Template':
        """Create a MobileNet template."""
        from torchvision_customizer.templates.mobilenet import MobileNetTemplate
        return MobileNetTemplate(version=version, **kwargs)
    
    @classmethod
    def densenet(cls, layers: int = 121, **kwargs) -> 'Template':
        """Create a DenseNet template."""
        from torchvision_customizer.templates.densenet import DenseNetTemplate
        return DenseNetTemplate(layers=layers, **kwargs)
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """List available template names."""
        return sorted(cls._templates.keys())
    
    # Customization methods
    
    def replace_activation(self, activation: str) -> 'Template':
        """Replace all activation functions.
        
        Args:
            activation: New activation function name
            
        Returns:
            Self for chaining
        """
        self._config['activation'] = activation
        self._modifications.append(('activation', activation))
        return self
    
    def replace_norm(self, norm_type: str) -> 'Template':
        """Replace all normalization layers.
        
        Args:
            norm_type: New normalization type ('batch', 'layer', 'group', 'instance')
            
        Returns:
            Self for chaining
        """
        self._config['norm_type'] = norm_type
        self._modifications.append(('norm', norm_type))
        return self
    
    def add_attention(self, attention_type: str = 'se', after: str = 'block') -> 'Template':
        """Add attention mechanism.
        
        Args:
            attention_type: Type of attention ('se', 'cbam', 'channel', 'spatial')
            after: Where to add ('block', 'conv', 'stage')
            
        Returns:
            Self for chaining
        """
        if 'attention' not in self._config:
            self._config['attention'] = {}
        self._config['attention']['type'] = attention_type
        self._config['attention']['position'] = after
        self._modifications.append(('attention', f'{attention_type} after {after}'))
        return self
    
    def modify_stage(self, stage_idx: int, **kwargs) -> 'Template':
        """Modify a specific stage.
        
        Args:
            stage_idx: Stage index (0-based)
            **kwargs: Stage modifications (blocks, channels, etc.)
            
        Returns:
            Self for chaining
        """
        if 'stage_modifications' not in self._config:
            self._config['stage_modifications'] = {}
        self._config['stage_modifications'][stage_idx] = kwargs
        self._modifications.append(('stage', f'stage {stage_idx}: {kwargs}'))
        return self
    
    def set_stem(self, **kwargs) -> 'Template':
        """Configure the stem.
        
        Args:
            **kwargs: Stem configuration (channels, kernel, stride, etc.)
            
        Returns:
            Self for chaining
        """
        if 'stem' not in self._config:
            self._config['stem'] = {}
        self._config['stem'].update(kwargs)
        self._modifications.append(('stem', str(kwargs)))
        return self
    
    def set_head(self, **kwargs) -> 'Template':
        """Configure the head.
        
        Args:
            **kwargs: Head configuration (hidden, dropout, etc.)
            
        Returns:
            Self for chaining
        """
        if 'head' not in self._config:
            self._config['head'] = {}
        self._config['head'].update(kwargs)
        return self
    
    def use_dropout(self, rate: float) -> 'Template':
        """Set dropout rate.
        
        Args:
            rate: Dropout probability
            
        Returns:
            Self for chaining
        """
        self._config['dropout'] = rate
        return self
    
    def scale_channels(self, factor: float) -> 'Template':
        """Scale all channel counts by a factor.
        
        Args:
            factor: Scaling factor (e.g., 0.5 for half, 2.0 for double)
            
        Returns:
            Self for chaining
        """
        self._config['channel_scale'] = factor
        self._modifications.append(('scale', f'{factor}x channels'))
        return self
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return deepcopy(self._config)
    
    def describe(self) -> str:
        """Get a human-readable description of the template."""
        lines = [f"Template: {self._config.get('name', 'Custom')}"]
        lines.append("-" * 40)
        
        for key, value in self._config.items():
            if key not in ['name', 'stage_modifications']:
                lines.append(f"  {key}: {value}")
        
        if self._modifications:
            lines.append("\nModifications:")
            for mod_type, mod_value in self._modifications:
                lines.append(f"  • {mod_type}: {mod_value}")
        
        return "\n".join(lines)
    
    def build(self, num_classes: int = 1000, **kwargs) -> nn.Module:
        """Build the model from template.
        
        Args:
            num_classes: Number of output classes
            **kwargs: Additional build arguments
            
        Returns:
            Built nn.Module
        """
        # Override in subclasses
        raise NotImplementedError("Subclasses must implement build()")
    
    def __repr__(self) -> str:
        name = self._config.get('name', 'Template')
        mods = len(self._modifications)
        return f"{name}(modifications={mods})"
