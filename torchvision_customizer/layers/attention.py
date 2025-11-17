"""
Attention Mechanisms Module

Comprehensive attention mechanisms for neural networks:
- Channel Attention (Squeeze-and-Excitation style)
- Spatial Attention
- Multi-Head Attention
- Positional Encoding for transformers

Author: torchvision-customizer
License: MIT
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Mechanism (Squeeze-and-Excitation style).
    
    Recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels.
    
    Attributes:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for bottleneck
        
    Examples:
        >>> attn = ChannelAttention(channels=64, reduction=16)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = attn(x)  # Same shape as input
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        activation: str = 'relu',
    ):
        """
        Initialize ChannelAttention.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for bottleneck layer
            activation: Activation function name
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        if reduction < 1:
            raise ValueError(f"reduction must be >= 1, got {reduction}")
        
        self.channels = channels
        self.reduction = reduction
        reduced_channels = max(1, channels // reduction)
        
        # Squeeze and Excitation
        self.fc1 = nn.Linear(channels, reduced_channels, bias=True)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.fc2 = nn.Linear(reduced_channels, channels, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-scaled tensor of same shape
        """
        batch, channels, height, width = x.size()
        
        # Squeeze: Global average pooling
        squeeze = F.adaptive_avg_pool2d(x, 1).view(batch, channels)
        
        # Excitation: FC layers
        excitation = self.fc1(squeeze)
        excitation = self.activation(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(batch, channels, 1, 1)
        
        # Scale
        return x * excitation


class SpatialAttention(nn.Module):
    """
    Spatial Attention Mechanism.
    
    Generates attention maps along the spatial dimension.
    Focuses on "where" to pay attention.
    
    Attributes:
        kernel_size (int): Kernel size for convolution
        
    Examples:
        >>> attn = SpatialAttention(kernel_size=7)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = attn(x)  # Same shape as input
    """
    
    def __init__(
        self,
        kernel_size: int = 7,
    ):
        """
        Initialize SpatialAttention.
        
        Args:
            kernel_size: Kernel size for convolution (must be odd)
            
        Raises:
            ValueError: If kernel_size is even
        """
        super().__init__()
        
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-scaled tensor of same shape
        """
        # Channel statistics
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(concat)
        attention = self.sigmoid(attention)
        
        # Scale
        return x * attention


class ChannelSpatialAttention(nn.Module):
    """
    Combined Channel and Spatial Attention.
    
    Sequentially applies channel and spatial attention for comprehensive
    feature recalibration.
    
    Examples:
        >>> attn = ChannelSpatialAttention(channels=64)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = attn(x)
    """
    
    def __init__(
        self,
        channels: int,
        channel_reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        """
        Initialize ChannelSpatialAttention.
        
        Args:
            channels: Number of input channels
            channel_reduction: Channel attention reduction ratio
            spatial_kernel: Spatial attention kernel size
        """
        super().__init__()
        self.channel_attn = ChannelAttention(channels, channel_reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sequential attention."""
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism (scaled dot-product).
    
    Allows the model to attend to information from different representation
    subspaces at different positions.
    
    Attributes:
        embed_dim (int): Total embedding dimension
        num_heads (int): Number of attention heads
        
    Examples:
        >>> attn = MultiHeadAttention(embed_dim=256, num_heads=8)
        >>> x = torch.randn(2, 32, 256)  # (batch, seq_len, embed_dim)
        >>> out = attn(x)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        """
        Initialize MultiHeadAttention.
        
        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of parallel attention heads
            dropout: Dropout probability
            bias: Whether to use bias
            batch_first: Whether batch dimension is first
            
        Raises:
            ValueError: If embed_dim is not divisible by num_heads
        """
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        
        # Linear transformations
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: Query tensor of shape (B, L, E) or (L, B, E)
            key: Key tensor (defaults to query if None)
            value: Value tensor (defaults to query if None)
            attention_mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, seq_len, _ = query.shape if self.batch_first else (query.shape[1], query.shape[0], query.shape[2])
        
        # Linear projections
        Q = self.q_proj(query)  # (B, L, E)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        
        # Apply mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (B, H, L, D)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.embed_dim)
        
        # Final linear projection
        output = self.out_proj(context)
        
        return output, attn_weights


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer models.
    
    Provides position information to the model using sinusoidal functions.
    
    Examples:
        >>> pos_enc = PositionalEncoding(d_model=256, max_len=512)
        >>> x = torch.randn(2, 32, 256)
        >>> x = x + pos_enc(x)
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        """
        Initialize PositionalEncoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, L, D)
            
        Returns:
            Positional encoding for the sequence
        """
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]
        return self.dropout(pe)


class AttentionBlock(nn.Module):
    """
    Complete attention block combining multiple attention mechanisms.
    
    Can be used as a building block in neural networks.
    
    Examples:
        >>> block = AttentionBlock(
        ...     channels=64,
        ...     use_channel=True,
        ...     use_spatial=True,
        ...     use_cross=False
        ... )
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = block(x)
    """
    
    def __init__(
        self,
        channels: int,
        use_channel: bool = True,
        use_spatial: bool = True,
        use_cross: bool = False,
        channel_reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        """
        Initialize AttentionBlock.
        
        Args:
            channels: Number of input channels
            use_channel: Whether to use channel attention
            use_spatial: Whether to use spatial attention
            use_cross: Whether to use cross-channel spatial attention
            channel_reduction: Channel attention reduction ratio
            spatial_kernel: Spatial attention kernel size
        """
        super().__init__()
        
        self.use_channel = use_channel
        self.use_spatial = use_spatial
        self.use_cross = use_cross
        
        if use_channel:
            self.channel_attn = ChannelAttention(channels, channel_reduction)
        
        if use_spatial:
            self.spatial_attn = SpatialAttention(spatial_kernel)
        
        if use_cross:
            # Cross-channel spatial attention
            self.cross_attn = nn.Sequential(
                nn.Conv2d(channels, channels // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 2, channels, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with selected attention mechanisms."""
        if self.use_channel:
            x = self.channel_attn(x)
        
        if self.use_spatial:
            x = self.spatial_attn(x)
        
        if self.use_cross:
            x = x * self.cross_attn(x)
        
        return x


# Utility functions

def create_attention_map(
    x: torch.Tensor,
    attention_type: str = 'channel',
) -> torch.Tensor:
    """
    Create attention map from input.
    
    Args:
        x: Input tensor
        attention_type: Type of attention ('channel', 'spatial')
        
    Returns:
        Attention map
    """
    if attention_type == 'channel':
        return torch.mean(x, dim=(2, 3), keepdim=True)
    elif attention_type == 'spatial':
        return torch.mean(x, dim=1, keepdim=True)
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")


def apply_attention(
    x: torch.Tensor,
    attention: torch.Tensor,
) -> torch.Tensor:
    """
    Apply attention map to input.
    
    Args:
        x: Input tensor
        attention: Attention map
        
    Returns:
        Attention-weighted tensor
    """
    return x * attention


def normalize_attention(
    attention: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize attention to sum to 1.
    
    Args:
        attention: Attention map
        
    Returns:
        Normalized attention
    """
    batch_size = attention.size(0)
    return attention.view(batch_size, -1).softmax(dim=1).view_as(attention)
