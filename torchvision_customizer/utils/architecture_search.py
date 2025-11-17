"""
Architecture Search Utilities

Neural Architecture Search (NAS) utilities for exploring and generating
model architectures programmatically:
- Grid search over architecture hyperparameters
- Random search for architecture discovery
- Architecture factory for quick model generation
- Architecture validation and scoring

Author: torchvision-customizer
License: MIT
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import random
import itertools
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from enum import Enum


class ArchitecturePattern(Enum):
    """Predefined architecture patterns."""
    SEQUENTIAL = 'sequential'
    RESIDUAL = 'residual'
    DENSE = 'dense'
    INCEPTION = 'inception'
    MIXED = 'mixed'


@dataclass
class ArchitectureConfig:
    """
    Architecture configuration dataclass.
    
    Attributes:
        input_shape: Tuple of (channels, height, width)
        num_classes: Number of output classes
        num_conv_blocks: Number of convolutional blocks
        channels: List of channel sizes or 'auto'
        kernel_sizes: List of kernel sizes or single value
        strides: List of strides or single value
        activations: List of activation names or single name
        dropout_rates: List of dropout rates or single value
        use_batchnorm: Whether to use batch normalization
        pattern: Architecture pattern
        use_attention: Whether to use attention mechanisms
        use_residual: Whether to use residual connections
        use_dense: Whether to use dense connections
    """
    input_shape: Tuple[int, int, int]
    num_classes: int
    num_conv_blocks: int = 4
    channels: Union[str, List[int]] = 'auto'
    kernel_sizes: Union[int, List[int]] = 3
    strides: Union[int, List[int]] = 1
    activations: Union[str, List[str]] = 'relu'
    dropout_rates: Union[float, List[float]] = 0.0
    use_batchnorm: bool = True
    pattern: str = 'sequential'
    use_attention: bool = False
    use_residual: bool = False
    use_dense: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArchitectureConfig':
        """Create from dictionary."""
        return cls(**data)
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if valid
            
        Raises:
            ValueError: If invalid
        """
        if len(self.input_shape) != 3:
            raise ValueError(f"input_shape must have 3 elements, got {self.input_shape}")
        
        if self.num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {self.num_classes}")
        
        if self.num_conv_blocks < 1:
            raise ValueError(f"num_conv_blocks must be >= 1, got {self.num_conv_blocks}")
        
        return True


class GridSearch:
    """
    Grid search over architecture hyperparameters.
    
    Systematically explores all combinations of provided parameters.
    
    Examples:
        >>> search_space = {
        ...     'num_conv_blocks': [3, 4, 5],
        ...     'channels': [['auto'], [[64, 128, 256], [32, 64, 128, 256], [64, 128, 256, 512]]],
        ...     'activation': ['relu', 'gelu'],
        ... }
        >>> searcher = GridSearch(search_space)
        >>> configs = list(searcher.generate())
        >>> len(configs)  # 3 * 3 * 2 = 18
    """
    
    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        base_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize GridSearch.
        
        Args:
            search_space: Dictionary mapping parameter names to lists of values
            base_config: Base configuration to update
        """
        self.search_space = search_space
        self.base_config = base_config or {}
        self.num_combinations = self._count_combinations()
    
    def _count_combinations(self) -> int:
        """Count total combinations."""
        count = 1
        for values in self.search_space.values():
            count *= len(values)
        return count
    
    def generate(self):
        """
        Generate all configurations.
        
        Yields:
            ArchitectureConfig objects
        """
        keys = self.search_space.keys()
        value_lists = self.search_space.values()
        
        for combination in itertools.product(*value_lists):
            config_dict = self.base_config.copy()
            config_dict.update(zip(keys, combination))
            yield ArchitectureConfig(**config_dict)
    
    def generate_with_index(self):
        """
        Generate configurations with their index.
        
        Yields:
            Tuple of (index, ArchitectureConfig)
        """
        for idx, config in enumerate(self.generate(), 1):
            yield idx, config


class RandomSearch:
    """
    Random search over architecture hyperparameters.
    
    Randomly samples from provided parameter distributions.
    
    Examples:
        >>> search_space = {
        ...     'num_conv_blocks': [3, 4, 5, 6],
        ...     'activation': ['relu', 'gelu', 'leaky_relu'],
        ...     'dropout_rate': [0.0, 0.1, 0.2, 0.3],
        ... }
        >>> searcher = RandomSearch(search_space, num_samples=100)
        >>> configs = list(searcher.generate())
    """
    
    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        num_samples: int = 10,
        base_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize RandomSearch.
        
        Args:
            search_space: Dictionary mapping parameter names to lists of values
            num_samples: Number of random samples to generate
            base_config: Base configuration to update
            seed: Random seed for reproducibility
        """
        self.search_space = search_space
        self.num_samples = num_samples
        self.base_config = base_config or {}
        
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
    
    def generate(self):
        """
        Generate random configurations.
        
        Yields:
            ArchitectureConfig objects
        """
        for _ in range(self.num_samples):
            config_dict = self.base_config.copy()
            
            for key, values in self.search_space.items():
                config_dict[key] = random.choice(values)
            
            yield ArchitectureConfig(**config_dict)
    
    def generate_with_index(self):
        """
        Generate random configurations with their index.
        
        Yields:
            Tuple of (index, ArchitectureConfig)
        """
        for idx, config in enumerate(self.generate(), 1):
            yield idx, config


class ArchitectureFactory:
    """
    Factory for generating architectures from patterns and configurations.
    
    Simplifies creating models from predefined patterns.
    
    Examples:
        >>> factory = ArchitectureFactory()
        >>> config = ArchitectureConfig(
        ...     input_shape=(3, 224, 224),
        ...     num_classes=1000,
        ...     pattern='residual'
        ... )
        >>> architecture_dict = factory.create(config)
    """
    
    def __init__(self):
        """Initialize ArchitectureFactory."""
        self.patterns = {}
        self._register_default_patterns()
    
    def _register_default_patterns(self):
        """Register default architecture patterns."""
        self.patterns['sequential'] = self._create_sequential
        self.patterns['residual'] = self._create_residual
        self.patterns['dense'] = self._create_dense
        self.patterns['inception'] = self._create_inception
        self.patterns['mixed'] = self._create_mixed
    
    def register_pattern(
        self,
        name: str,
        builder: Callable[[ArchitectureConfig], Dict[str, Any]],
    ):
        """
        Register custom architecture pattern.
        
        Args:
            name: Pattern name
            builder: Function that creates architecture from config
        """
        self.patterns[name] = builder
    
    def create(self, config: ArchitectureConfig) -> Dict[str, Any]:
        """
        Create architecture from configuration.
        
        Args:
            config: Architecture configuration
            
        Returns:
            Dictionary with architecture information
            
        Raises:
            ValueError: If pattern is not recognized
        """
        config.validate()
        
        pattern = config.pattern
        if pattern not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        builder = self.patterns[pattern]
        return builder(config)
    
    def _create_sequential(self, config: ArchitectureConfig) -> Dict[str, Any]:
        """Create sequential architecture."""
        return {
            'type': 'sequential',
            'num_blocks': config.num_conv_blocks,
            'channels': config.channels,
            'kernel_sizes': config.kernel_sizes,
            'activations': config.activations,
            'dropout_rates': config.dropout_rates,
            'use_batchnorm': config.use_batchnorm,
        }
    
    def _create_residual(self, config: ArchitectureConfig) -> Dict[str, Any]:
        """Create residual architecture."""
        return {
            'type': 'residual',
            'num_blocks': config.num_conv_blocks,
            'channels': config.channels,
            'kernel_sizes': config.kernel_sizes,
            'activations': config.activations,
            'use_residual': True,
            'skip_pattern': 'residual',
        }
    
    def _create_dense(self, config: ArchitectureConfig) -> Dict[str, Any]:
        """Create dense architecture."""
        return {
            'type': 'dense',
            'num_blocks': config.num_conv_blocks,
            'growth_rate': 32,
            'compression': 0.5,
            'use_dense': True,
        }
    
    def _create_inception(self, config: ArchitectureConfig) -> Dict[str, Any]:
        """Create inception architecture."""
        return {
            'type': 'inception',
            'num_blocks': config.num_conv_blocks,
            'branch_ratios': [0.5, 0.25, 0.125, 0.125],
        }
    
    def _create_mixed(self, config: ArchitectureConfig) -> Dict[str, Any]:
        """Create mixed architecture."""
        return {
            'type': 'mixed',
            'num_blocks': config.num_conv_blocks,
            'patterns': ['sequential', 'residual', 'dense', 'sequential'],
        }


class ArchitectureScorer:
    """
    Score architectures based on various metrics.
    
    Evaluates architectures by parameter count, FLOPs, memory usage, etc.
    
    Examples:
        >>> scorer = ArchitectureScorer()
        >>> config = ArchitectureConfig(
        ...     input_shape=(3, 224, 224),
        ...     num_classes=1000,
        ...     num_conv_blocks=4
        ... )
        >>> score = scorer.score_config(config)
    """
    
    def __init__(
        self,
        max_parameters: Optional[int] = None,
        max_memory_mb: Optional[float] = None,
        target_flops: Optional[float] = None,
    ):
        """
        Initialize ArchitectureScorer.
        
        Args:
            max_parameters: Maximum allowed parameters
            max_memory_mb: Maximum allowed memory in MB
            target_flops: Target FLOPs (for optimization)
        """
        self.max_parameters = max_parameters or 100_000_000  # 100M default
        self.max_memory_mb = max_memory_mb or 1000.0  # 1GB default
        self.target_flops = target_flops
    
    def score_config(self, config: ArchitectureConfig) -> float:
        """
        Score architecture configuration.
        
        Args:
            config: Architecture configuration
            
        Returns:
            Score (higher is better)
        """
        config.validate()
        
        # Estimate parameters
        params = self._estimate_parameters(config)
        
        # Check constraints
        if params > self.max_parameters:
            return 0.0
        
        # Calculate score (normalize parameters)
        score = 1.0 - (params / self.max_parameters)
        
        return score
    
    def _estimate_parameters(self, config: ArchitectureConfig) -> int:
        """Estimate number of parameters."""
        # Simplified estimation
        in_channels = config.input_shape[0]
        params = 0
        
        if isinstance(config.channels, str):
            # Auto channel generation
            channels = [64 * (2 ** i) for i in range(config.num_conv_blocks)]
        else:
            channels = config.channels
        
        for ch in channels:
            kernel_size = config.kernel_sizes if isinstance(config.kernel_sizes, int) else config.kernel_sizes[0]
            params += in_channels * ch * kernel_size * kernel_size
            in_channels = ch
        
        # Classifier
        params += in_channels * config.num_classes
        
        return int(params)


class ArchitectureComparator:
    """
    Compare multiple architectures.
    
    Examples:
        >>> configs = [
        ...     ArchitectureConfig(...),
        ...     ArchitectureConfig(...),
        ... ]
        >>> comparator = ArchitectureComparator(configs)
        >>> best = comparator.get_best()
    """
    
    def __init__(
        self,
        configs: List[ArchitectureConfig],
        scorer: Optional[ArchitectureScorer] = None,
    ):
        """
        Initialize ArchitectureComparator.
        
        Args:
            configs: List of configurations to compare
            scorer: Architecture scorer (uses default if None)
        """
        self.configs = configs
        self.scorer = scorer or ArchitectureScorer()
        self.scores = [self.scorer.score_config(c) for c in configs]
    
    def get_best(self) -> Tuple[ArchitectureConfig, float]:
        """Get best configuration."""
        best_idx = max(range(len(self.scores)), key=lambda i: self.scores[i])
        return self.configs[best_idx], self.scores[best_idx]
    
    def get_worst(self) -> Tuple[ArchitectureConfig, float]:
        """Get worst configuration."""
        worst_idx = min(range(len(self.scores)), key=lambda i: self.scores[i])
        return self.configs[worst_idx], self.scores[worst_idx]
    
    def get_top_k(self, k: int = 3) -> List[Tuple[ArchitectureConfig, float]]:
        """Get top k configurations."""
        sorted_indices = sorted(range(len(self.scores)), key=lambda i: self.scores[i], reverse=True)
        return [(self.configs[i], self.scores[i]) for i in sorted_indices[:k]]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get score statistics."""
        return {
            'mean': sum(self.scores) / len(self.scores),
            'min': min(self.scores),
            'max': max(self.scores),
            'std': (sum((s - (sum(self.scores) / len(self.scores))) ** 2 for s in self.scores) / len(self.scores)) ** 0.5,
        }


# Utility functions

def expand_config_list(
    value: Union[Any, List[Any]],
    length: int,
) -> List[Any]:
    """
    Expand single value or list to specified length.
    
    Args:
        value: Single value or list
        length: Target length
        
    Returns:
        List of specified length
        
    Raises:
        ValueError: If list length doesn't match target
    """
    if isinstance(value, list):
        if len(value) != length:
            raise ValueError(f"List length {len(value)} doesn't match target {length}")
        return value
    else:
        return [value] * length


def sample_architecture(
    num_conv_blocks: int,
    min_channels: int = 32,
    max_channels: int = 512,
    min_kernel: int = 3,
    max_kernel: int = 7,
) -> ArchitectureConfig:
    """
    Generate random architecture.
    
    Args:
        num_conv_blocks: Number of conv blocks
        min_channels: Minimum channels
        max_channels: Maximum channels
        min_kernel: Minimum kernel size
        max_kernel: Maximum kernel size
        
    Returns:
        Random ArchitectureConfig
    """
    channels = [random.randint(min_channels, max_channels) for _ in range(num_conv_blocks)]
    kernel_sizes = [random.randint(min_kernel, max_kernel) for _ in range(num_conv_blocks)]
    activations = [random.choice(['relu', 'gelu', 'leaky_relu']) for _ in range(num_conv_blocks)]
    
    return ArchitectureConfig(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=num_conv_blocks,
        channels=channels,
        kernel_sizes=kernel_sizes,
        activations=activations,
    )
