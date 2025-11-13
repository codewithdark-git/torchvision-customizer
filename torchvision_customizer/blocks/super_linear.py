"""Enhanced linear layer with activation and dropout.

Implements a flexible linear layer with integrated activation functions
and dropout for convenient classifier head construction.

Example:
    >>> from torchvision_customizer.blocks import SuperLinear
    >>> linear = SuperLinear(in_features=2048, out_features=1000)
    >>> output = linear(x)
"""

import torch
import torch.nn as nn
from torchvision_customizer.layers import get_activation


class SuperLinear(nn.Module):
    """Enhanced linear layer with activation and dropout.

    Provides a flexible linear layer with integrated:
    - Batch normalization
    - Activation functions
    - Dropout for regularization

    Useful for building classifier heads and dense layers in networks.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        activation: Activation function. Default is 'relu'.
        dropout_rate: Dropout probability. Default is 0.0.
        use_activation: Whether to apply activation. Default is True.
        use_dropout: Whether to apply dropout. Default is True if dropout_rate > 0.
        bias: Whether to use bias. Default is True.

    Example:
        >>> # Basic linear layer
        >>> linear = SuperLinear(in_features=2048, out_features=1000)
        >>> x = torch.randn(2, 2048)
        >>> output = linear(x)
        >>> print(output.shape)
        torch.Size([2, 1000])

        >>> # Linear layer with dropout
        >>> linear = SuperLinear(
        ...     in_features=512, out_features=256,
        ...     activation='relu', dropout_rate=0.5
        ... )

        >>> # No activation (for final output layer)
        >>> linear = SuperLinear(
        ...     in_features=256, out_features=10,
        ...     use_activation=False, use_dropout=False
        ... )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        use_activation: bool = True,
        use_dropout: bool | None = None,
        bias: bool = True,
    ) -> None:
        """Initialize SuperLinear."""
        super().__init__()

        # Determine if dropout should be used
        if use_dropout is None:
            use_dropout = dropout_rate > 0.0

        self.use_activation = use_activation
        self.use_dropout = use_dropout

        # Linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Activation
        if use_activation:
            self.activation = get_activation(activation)
        else:
            self.activation = nn.Identity()

        # Dropout
        if use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply enhanced linear layer.

        Args:
            x: Input tensor of shape (B, in_features) or (B, ..., in_features).

        Returns:
            Output tensor of shape (B, out_features) or (B, ..., out_features).
        """
        out = self.linear(x)

        if self.use_activation:
            out = self.activation(out)

        if self.use_dropout:
            out = self.dropout(out)

        return out
