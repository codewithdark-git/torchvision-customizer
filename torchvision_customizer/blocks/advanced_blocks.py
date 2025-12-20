"""Advanced Building Blocks for v2.1.

New blocks for modern architectures:
- CBAMBlock: Convolutional Block Attention Module
- ECABlock: Efficient Channel Attention
- DropPath: Stochastic Depth
- Mish: Self-regularizing activation
- GeM: Generalized Mean Pooling
- CoordConv: Coordinate Convolution

These blocks are designed for use in hybrid models and custom architectures.
"""

from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (CBAM).
    
    Applies channel attention followed by spatial attention for
    comprehensive feature recalibration.
    
    Reference:
        "CBAM: Convolutional Block Attention Module" (ECCV 2018)
        https://arxiv.org/abs/1807.06521
        
    Args:
        channels: Number of input/output channels
        reduction: Channel reduction ratio (default: 16)
        spatial_kernel: Spatial attention kernel size (default: 7)
        
    Example:
        >>> block = CBAMBlock(channels=256, reduction=16)
        >>> x = torch.randn(2, 256, 32, 32)
        >>> out = block(x)  # Same shape
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__()
        
        self.channels = channels
        reduced = max(1, channels // reduction)
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
        )
        
        # Spatial attention
        padding = spatial_kernel // 2
        self.spatial_conv = nn.Conv2d(
            2, 1, 
            kernel_size=spatial_kernel, 
            padding=padding, 
            bias=False
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        b, c, h, w = x.size()
        
        avg_out = self.channel_fc(self.avg_pool(x).view(b, c))
        max_out = self.channel_fc(self.max_pool(x).view(b, c))
        channel_attn = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        x = x * channel_attn
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.sigmoid(
            self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        )
        
        return x * spatial_attn


class ECABlock(nn.Module):
    """Efficient Channel Attention (ECA) Module.
    
    Captures cross-channel interaction using 1D convolution,
    avoiding dimensionality reduction for better efficiency.
    
    Reference:
        "ECA-Net: Efficient Channel Attention for Deep CNNs" (CVPR 2020)
        https://arxiv.org/abs/1910.03151
        
    Args:
        channels: Number of input channels
        kernel_size: 1D conv kernel size (auto-calculated if None)
        gamma: Coefficient for adaptive kernel calculation
        beta: Coefficient for adaptive kernel calculation
        
    Example:
        >>> block = ECABlock(channels=512)
        >>> x = torch.randn(2, 512, 16, 16)
        >>> out = block(x)
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: Optional[int] = None,
        gamma: int = 2,
        beta: int = 1,
    ):
        super().__init__()
        
        self.channels = channels
        
        # Adaptive kernel size
        if kernel_size is None:
            t = int(abs((math.log2(channels) + beta) / gamma))
            kernel_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, 
            kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2, 
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling
        y = self.avg_pool(x)  # (B, C, 1, 1)
        
        # 1D convolution for channel attention
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        
        return x * self.sigmoid(y)


class DropPath(nn.Module):
    """Drop Path (Stochastic Depth) regularization.
    
    Randomly drops entire residual branches during training,
    effectively creating an ensemble of networks.
    
    Reference:
        "Deep Networks with Stochastic Depth" (ECCV 2016)
        https://arxiv.org/abs/1603.09382
        
    Args:
        drop_prob: Probability of dropping the path (default: 0.0)
        scale_by_keep: Whether to scale by keep probability (default: True)
        
    Example:
        >>> drop_path = DropPath(drop_prob=0.2)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = x + drop_path(residual)
    """
    
    def __init__(
        self,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        
        # Create random tensor for batch dimension only
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        
        if self.scale_by_keep:
            random_tensor.div_(keep_prob)
        
        return x * random_tensor
    
    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class Mish(nn.Module):
    """Mish Activation Function.
    
    Self-regularizing non-monotonic activation function.
    Mish(x) = x * tanh(softplus(x))
    
    Reference:
        "Mish: A Self Regularized Non-Monotonic Activation Function"
        https://arxiv.org/abs/1908.08681
        
    Example:
        >>> mish = Mish()
        >>> x = torch.randn(2, 64)
        >>> out = mish(x)
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class GeM(nn.Module):
    """Generalized Mean Pooling.
    
    Generalizes average and max pooling through a learnable
    pooling parameter p.
    
    Reference:
        "Fine-tuning CNN Image Retrieval with No Human Annotation" (TPAMI 2018)
        https://arxiv.org/abs/1711.02512
        
    Args:
        p: Initial pooling power (default: 3.0)
        eps: Numerical stability epsilon
        learnable: Whether p is learnable (default: True)
        
    Example:
        >>> gem = GeM(p=3.0)
        >>> x = torch.randn(2, 2048, 7, 7)
        >>> out = gem(x)  # (2, 2048, 1, 1)
    """
    
    def __init__(
        self,
        p: float = 3.0,
        eps: float = 1e-6,
        learnable: bool = True,
    ):
        super().__init__()
        
        self.eps = eps
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer('p', torch.tensor([p]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            output_size=1
        ).pow(1.0 / self.p)
    
    def extra_repr(self) -> str:
        return f"p={self.p.item():.2f}, eps={self.eps}"


class CoordConv(nn.Module):
    """Coordinate Convolution.
    
    Adds coordinate channels to the input before convolution,
    allowing the network to learn spatial relationships.
    
    Reference:
        "An Intriguing Failing of Convolutional Neural Networks..." (NeurIPS 2018)
        https://arxiv.org/abs/1807.03247
        
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        with_r: Include radial coordinate (default: False)
        **kwargs: Additional Conv2d arguments
        
    Example:
        >>> conv = CoordConv(64, 128, kernel_size=3, padding=1)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = conv(x)  # (2, 128, 32, 32)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        with_r: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.with_r = with_r
        extra_channels = 3 if with_r else 2
        
        self.conv = nn.Conv2d(
            in_channels + extra_channels,
            out_channels,
            kernel_size,
            **kwargs,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        device = x.device
        dtype = x.dtype
        
        # Create coordinate channels
        xx_channel = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        xx_channel = xx_channel.view(1, 1, 1, w).expand(b, 1, h, w)
        
        yy_channel = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        yy_channel = yy_channel.view(1, 1, h, 1).expand(b, 1, h, w)
        
        coords = [x, xx_channel, yy_channel]
        
        if self.with_r:
            rr_channel = torch.sqrt(xx_channel ** 2 + yy_channel ** 2)
            coords.append(rr_channel)
        
        return self.conv(torch.cat(coords, dim=1))


class GELUActivation(nn.Module):
    """GELU Activation with optional approximate mode.
    
    Gaussian Error Linear Unit - smoother than ReLU.
    
    Args:
        approximate: Use tanh approximation (default: 'none')
        
    Example:
        >>> gelu = GELUActivation()
        >>> x = torch.randn(2, 64)
        >>> out = gelu(x)
    """
    
    def __init__(self, approximate: str = 'none'):
        super().__init__()
        self.approximate = approximate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation Module (Enhanced).
    
    Enhanced SE block with more options for activation and normalization.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio (default: 16)
        activation: Activation function ('relu', 'swish', 'gelu')
        gate_activation: Gate activation ('sigmoid', 'hardsigmoid')
        use_bn: Use batch normalization in FC layers
        
    Example:
        >>> se = SqueezeExcitation(256, reduction=8, activation='swish')
        >>> x = torch.randn(2, 256, 16, 16)
        >>> out = se(x)
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        activation: str = 'relu',
        gate_activation: str = 'sigmoid',
        use_bn: bool = False,
    ):
        super().__init__()
        
        self.channels = channels
        reduced = max(1, channels // reduction)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        layers = [nn.Linear(channels, reduced, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm1d(reduced))
        
        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'swish':
            layers.append(nn.SiLU(inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        else:
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Linear(reduced, channels, bias=True))
        
        # Gate activation
        if gate_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif gate_activation == 'hardsigmoid':
            layers.append(nn.Hardsigmoid(inplace=True))
        else:
            layers.append(nn.Sigmoid())
        
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LayerScale(nn.Module):
    """Layer Scale for Vision Transformers.
    
    Learnable per-channel scaling factor, initialized small.
    
    Reference:
        "Going deeper with Image Transformers" (ICCV 2021)
        https://arxiv.org/abs/2103.17239
        
    Args:
        dim: Number of channels
        init_value: Initial scale value (default: 1e-5)
        
    Example:
        >>> ls = LayerScale(768, init_value=1e-5)
        >>> x = torch.randn(2, 196, 768)
        >>> out = ls(x)
    """
    
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation combo block.
    
    Commonly used building block with configurable components.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding (auto if None)
        groups: Convolution groups
        activation: Activation type ('relu', 'swish', 'gelu', 'mish', None)
        use_bn: Use batch normalization
        
    Example:
        >>> block = ConvBNAct(64, 128, kernel_size=3, activation='swish')
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = block(x)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation: Optional[str] = 'relu',
        use_bn: bool = True,
    ):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=not use_bn
        )
        
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # Activation
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'swish':
            self.act = nn.SiLU(inplace=True)
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'mish':
            self.act = Mish()
        elif activation is None:
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU(inplace=True)
        
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution (MBConv).
    
    Used in EfficientNet and MobileNetV2.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        expansion: Expansion ratio for hidden channels
        stride: Depthwise conv stride
        kernel_size: Depthwise conv kernel size
        use_se: Use squeeze-excitation
        se_reduction: SE reduction ratio
        drop_path: Drop path probability
        
    Example:
        >>> block = MBConv(32, 64, expansion=4, stride=2)
        >>> x = torch.randn(2, 32, 56, 56)
        >>> out = block(x)  # (2, 64, 28, 28)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 4,
        stride: int = 1,
        kernel_size: int = 3,
        use_se: bool = True,
        se_reduction: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        hidden_channels = in_channels * expansion
        
        # Expansion
        self.expand = nn.Identity()
        if expansion != 1:
            self.expand = ConvBNAct(
                in_channels, hidden_channels, 
                kernel_size=1, activation='swish'
            )
        
        # Depthwise
        padding = kernel_size // 2
        self.depthwise = ConvBNAct(
            hidden_channels, hidden_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=hidden_channels, activation='swish'
        )
        
        # SE
        self.se = nn.Identity()
        if use_se:
            self.se = SqueezeExcitation(
                hidden_channels, 
                reduction=expansion * se_reduction,
                activation='swish',
            )
        
        # Projection
        self.project = ConvBNAct(
            hidden_channels, out_channels,
            kernel_size=1, activation=None
        )
        
        # Skip connection
        self.use_skip = stride == 1 and in_channels == out_channels
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        
        if self.use_skip:
            x = identity + self.drop_path(x)
        
        return x


class FusedMBConv(nn.Module):
    """Fused Mobile Inverted Bottleneck (FusedMBConv).
    
    Used in EfficientNetV2, replaces depthwise+pointwise with regular conv.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        expansion: Expansion ratio
        stride: Convolution stride
        kernel_size: Convolution kernel size
        use_se: Use squeeze-excitation
        drop_path: Drop path probability
        
    Example:
        >>> block = FusedMBConv(32, 64, expansion=4, stride=2)
        >>> x = torch.randn(2, 32, 56, 56)
        >>> out = block(x)  # (2, 64, 28, 28)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 4,
        stride: int = 1,
        kernel_size: int = 3,
        use_se: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        hidden_channels = in_channels * expansion
        
        # Fused expand + depthwise
        self.expand = ConvBNAct(
            in_channels, hidden_channels,
            kernel_size=kernel_size, stride=stride,
            activation='swish'
        )
        
        # SE
        self.se = nn.Identity()
        if use_se:
            self.se = SqueezeExcitation(
                hidden_channels,
                reduction=expansion * 4,
                activation='swish',
            )
        
        # Projection
        self.project = ConvBNAct(
            hidden_channels, out_channels,
            kernel_size=1, activation=None
        )
        
        # Skip connection
        self.use_skip = stride == 1 and in_channels == out_channels
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        x = self.expand(x)
        x = self.se(x)
        x = self.project(x)
        
        if self.use_skip:
            x = identity + self.drop_path(x)
        
        return x

