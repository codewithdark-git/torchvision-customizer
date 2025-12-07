"""Composable module base class with operator overloading.

Provides the foundation for the >> (compose), + (sequential),
* (repeat), and | (branch) operators.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn


class ComposableModule(nn.Module):
    """Base class for composable neural network modules.
    
    Supports operator overloading for intuitive model composition:
    - >> : Compose (sequential flow)
    - +  : Add (sequential combination)
    - *  : Repeat (multiply instances)
    - |  : Branch (parallel paths)
    
    Example:
        >>> block1 = ComposableModule(...)
        >>> block2 = ComposableModule(...)
        >>> model = block1 >> block2  # Sequential composition
        >>> model = block1 + block2   # Same as >>
        >>> model = block1 * 3        # Repeat 3 times
        >>> model = block1 | block2   # Parallel branches
    """
    
    def __init__(self):
        super().__init__()
        self._out_channels: Optional[int] = None
        self._out_shape: Optional[Tuple[int, ...]] = None
    
    @property
    def out_channels(self) -> Optional[int]:
        """Output channels of this module (if applicable)."""
        return self._out_channels
    
    @property
    def out_shape(self) -> Optional[Tuple[int, ...]]:
        """Output shape of this module (if known)."""
        return self._out_shape
    
    def __rshift__(self, other: 'ComposableModule') -> 'ComposedModule':
        """Compose with >> operator: self >> other."""
        return ComposedModule([self, other])
    
    def __add__(self, other: 'ComposableModule') -> 'ComposedModule':
        """Sequential combination with + operator."""
        return ComposedModule([self, other])
    
    def __mul__(self, n: int) -> 'ComposedModule':
        """Repeat module n times with * operator."""
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"Repeat count must be positive integer, got {n}")
        # Note: Creates separate instances, not weight sharing
        return ComposedModule([self] * n)
    
    def __rmul__(self, n: int) -> 'ComposedModule':
        """Support n * module syntax."""
        return self.__mul__(n)
    
    def __or__(self, other: 'ComposableModule') -> 'BranchedModule':
        """Parallel branching with | operator."""
        return BranchedModule([self, other])


class ComposedModule(ComposableModule):
    """Sequential composition of multiple modules.
    
    Created automatically when using >> or + operators.
    """
    
    def __init__(self, modules: List[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for module in modules:
            if isinstance(module, ComposedModule):
                # Flatten nested compositions
                self.layers.extend(module.layers)
            else:
                self.layers.append(module)
        
        # Track output channels from last layer
        if self.layers and hasattr(self.layers[-1], 'out_channels'):
            self._out_channels = self.layers[-1].out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __rshift__(self, other: nn.Module) -> 'ComposedModule':
        """Allow chaining: (a >> b) >> c."""
        if isinstance(other, ComposedModule):
            return ComposedModule(list(self.layers) + list(other.layers))
        return ComposedModule(list(self.layers) + [other])
    
    def __add__(self, other: nn.Module) -> 'ComposedModule':
        """Allow chaining with +."""
        return self.__rshift__(other)
    
    def __repr__(self) -> str:
        layer_names = [type(l).__name__ for l in self.layers]
        return f"ComposedModule({' >> '.join(layer_names)})"

    def explain(self) -> str:
        """Generate human-readable model explanation."""
        lines = ["+" + "-" * 60 + "+"]
        lines.append("|" + "Composed Module".center(60) + "|")
        lines.append("+" + "-" * 60 + "+")
        
        for i, layer in enumerate(self.layers):
            layer_str = repr(layer)
            if len(layer_str) > 58:
                layer_str = layer_str[:55] + "..."
            lines.append("| " + layer_str.ljust(58) + " |")
        
        lines.append("+" + "-" * 60 + "+")
        
        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        lines.append("| " + f"Parameters: {total_params:,} ({trainable:,} trainable)".ljust(58) + " |")
        lines.append("+" + "-" * 60 + "+")
        
        return "\n".join(lines)


class BranchedModule(ComposableModule):
    """Parallel branching of multiple modules.
    
    Applies input to all branches and concatenates outputs.
    Created automatically when using | operator.
    """
    
    def __init__(self, modules: List[nn.Module], concat_dim: int = 1):
        super().__init__()
        self.branches = nn.ModuleList()
        self.concat_dim = concat_dim
        
        for module in modules:
            if isinstance(module, BranchedModule):
                # Flatten nested branches
                self.branches.extend(module.branches)
            else:
                self.branches.append(module)
        
        # Track combined output channels
        total_channels = 0
        for branch in self.branches:
            if hasattr(branch, 'out_channels') and branch.out_channels:
                total_channels += branch.out_channels
        if total_channels > 0:
            self._out_channels = total_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=self.concat_dim)
    
    def __or__(self, other: nn.Module) -> 'BranchedModule':
        """Allow chaining: (a | b) | c."""
        if isinstance(other, BranchedModule):
            return BranchedModule(list(self.branches) + list(other.branches))
        return BranchedModule(list(self.branches) + [other])
    
    def __repr__(self) -> str:
        branch_names = [type(b).__name__ for b in self.branches]
        return f"BranchedModule({' | '.join(branch_names)})"


def compose(*modules: nn.Module) -> ComposedModule:
    """Compose multiple modules sequentially.
    
    Alternative to >> operator for explicit composition.
    
    Example:
        >>> model = compose(conv1, conv2, conv3, head)
    """
    return ComposedModule(list(modules))


def repeat(module: nn.Module, n: int) -> ComposedModule:
    """Repeat a module n times.
    
    Note: Creates n separate instances with independent weights.
    For weight sharing, use a loop in forward.
    
    Example:
        >>> stage = repeat(ResidualBlock(64), 3)
    """
    return module * n


def branch(*modules: nn.Module, concat_dim: int = 1) -> BranchedModule:
    """Create parallel branches that concatenate outputs.
    
    Alternative to | operator for explicit branching.
    
    Example:
        >>> inception = branch(conv1x1, conv3x3, conv5x5, pool)
    """
    return BranchedModule(list(modules), concat_dim=concat_dim)


class ResidualWrapper(ComposableModule):
    """Wrapper to add residual connection around any module."""
    
    def __init__(self, module: nn.Module, projection: nn.Module = None):
        super().__init__()
        self.module = module
        self.projection = projection
        
        if hasattr(module, 'out_channels'):
            self._out_channels = module.out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.projection is not None:
            identity = self.projection(x)
        return self.module(x) + identity
    
    def __repr__(self) -> str:
        return f"Residual({self.module})"


def residual(module: nn.Module, projection: nn.Module = None) -> ResidualWrapper:
    """Wrap a module with residual connection.
    
    Example:
        >>> block = residual(ConvBlock(64, 64))
    """
    return ResidualWrapper(module, projection)
