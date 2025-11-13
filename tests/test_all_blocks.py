"""Quick test of all block implementations."""

import torch
from torchvision_customizer.blocks import (
    ConvBlock, SEBlock, ResidualBlock, DepthwiseBlock,
    InceptionModule, Conv3DBlock, SuperConv2d, SuperLinear
)

print("Testing all blocks...")
x = torch.randn(2, 3, 32, 32)

# Test ConvBlock
block = ConvBlock(in_channels=3, out_channels=64)
out = block(x)
print(f"✓ ConvBlock: {x.shape} → {out.shape}")

# Test SEBlock
se = SEBlock(channels=64)
se_out = se(out)
print(f"✓ SEBlock: {out.shape} → {se_out.shape}")

# Test ResidualBlock
res = ResidualBlock(in_channels=64, out_channels=64)
res_out = res(out)
print(f"✓ ResidualBlock: {out.shape} → {res_out.shape}")

# Test DepthwiseBlock
dw = DepthwiseBlock(in_channels=64, out_channels=128)
dw_out = dw(out)
print(f"✓ DepthwiseBlock: {out.shape} → {dw_out.shape}")

# Test InceptionModule
inception = InceptionModule(in_channels=64, out_1x1=32, out_3x3=32, out_5x5=16, out_pool=16)
inc_out = inception(out)
print(f"✓ InceptionModule: {out.shape} → {inc_out.shape}")

# Test Conv3DBlock
x3d = torch.randn(2, 3, 8, 32, 32)
conv3d = Conv3DBlock(in_channels=3, out_channels=64)
conv3d_out = conv3d(x3d)
print(f"✓ Conv3DBlock: {x3d.shape} → {conv3d_out.shape}")

# Test SuperConv2d
super_conv = SuperConv2d(in_channels=64, out_channels=128)
super_conv_out = super_conv(out)
print(f"✓ SuperConv2d: {out.shape} → {super_conv_out.shape}")

# Test SuperLinear
x_flat = torch.randn(2, 2048)
linear = SuperLinear(in_features=2048, out_features=1000)
linear_out = linear(x_flat)
print(f"✓ SuperLinear: {x_flat.shape} → {linear_out.shape}")

print("\n✓ All blocks working correctly!")
