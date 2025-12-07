"""Recipe definitions.

Defines the structure of an architecture recipe.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Recipe:
    """Architecture Recipe definition.
    
    A declarative blueprint for a neural network architecture.
    
    Args:
        stem: Stem definition string or dict (e.g., "conv(64, k=7, s=2)")
        stages: List of stage definitions (e.g., ["residual(64) x 3"])
        head: Head definition string or dict (e.g., "linear(1000)")
        name: Optional name for the architecture
        input_shape: Expected input shape (C, H, W)
    """
    stem: Union[str, Dict[str, Any]]
    stages: List[Union[str, Dict[str, Any]]]
    head: Union[str, Dict[str, Any]]
    name: str = "CustomModel"
    input_shape: tuple = (3, 224, 224)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'input_shape': self.input_shape,
            'stem': self.stem,
            'stages': self.stages,
            'head': self.head,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recipe':
        """Create from dictionary."""
        return cls(
            stem=data.get('stem', "conv(64)"),
            stages=data.get('stages', []),
            head=data.get('head', "linear(1000)"),
            name=data.get('name', "CustomModel"),
            input_shape=tuple(data.get('input_shape', (3, 224, 224))),
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Recipe':
        """Load from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
