"""v2.1 Hybrid Model Examples.

This file demonstrates how to use the new HybridBuilder to customize
pre-trained torchvision models with custom blocks and attention mechanisms.
"""

import torch

# ============================================================================
# Example 1: Basic Hybrid Model
# ============================================================================

def example_basic_hybrid():
    """Create a basic hybrid model from ResNet50."""
    from torchvision_customizer import HybridBuilder
    
    print("=" * 60)
    print("Example 1: Basic Hybrid Model")
    print("=" * 60)
    
    builder = HybridBuilder()
    
    # Load ResNet50 with ImageNet weights, customize for 100 classes
    model = builder.from_torchvision(
        "resnet50",
        weights="IMAGENET1K_V2",
        num_classes=100,
        dropout=0.3,
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {model.count_parameters():,}")
    
    return model


# ============================================================================
# Example 2: Hybrid with Attention Patches
# ============================================================================

def example_hybrid_with_attention():
    """Add attention mechanisms to a pre-trained backbone."""
    from torchvision_customizer import HybridBuilder
    
    print("\n" + "=" * 60)
    print("Example 2: Hybrid with Attention Patches")
    print("=" * 60)
    
    builder = HybridBuilder()
    
    # Add SE attention to layer3, CBAM to layer4
    model = builder.from_torchvision(
        "resnet50",
        weights="IMAGENET1K_V2",
        patches={
            "layer3": {"wrap": "se"},           # Squeeze-Excitation
            "layer4": {"wrap": "cbam_block"},   # CBAM attention
        },
        num_classes=10,
    )
    
    print(f"\nModifications applied:")
    for mod in model.modifications:
        print(f"  - {mod}")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"\nOutput shape: {out.shape}")
    
    return model


# ============================================================================
# Example 3: Fine-tuning with Frozen Backbone
# ============================================================================

def example_finetune_frozen():
    """Fine-tune with frozen backbone (only train head + last stage)."""
    from torchvision_customizer import HybridBuilder
    
    print("\n" + "=" * 60)
    print("Example 3: Fine-tuning with Frozen Backbone")
    print("=" * 60)
    
    builder = HybridBuilder()
    
    model = builder.from_torchvision(
        "resnet50",
        weights="IMAGENET1K_V2",
        num_classes=10,
        freeze_backbone=True,
        unfreeze_stages=[3],  # Only train stage 4 (index 3)
    )
    
    trainable = model.count_parameters(trainable_only=True)
    total = model.count_parameters()
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen: {(total - trainable):,} ({100*(total-trainable)/total:.1f}%)")
    
    return model


# ============================================================================
# Example 4: EfficientNet with Custom Head
# ============================================================================

def example_efficientnet_custom():
    """Customize EfficientNet for a different task."""
    from torchvision_customizer import HybridBuilder
    
    print("\n" + "=" * 60)
    print("Example 4: EfficientNet with Custom Configuration")
    print("=" * 60)
    
    builder = HybridBuilder()
    
    model = builder.from_torchvision(
        "efficientnet_b4",
        weights="IMAGENET1K_V1",
        num_classes=5,
        dropout=0.4,
    )
    
    x = torch.randn(1, 3, 380, 380)  # EfficientNet-B4 resolution
    out = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(model.explain())
    
    return model


# ============================================================================
# Example 5: Weight Transfer Utilities
# ============================================================================

def example_weight_utils():
    """Demonstrate weight loading utilities."""
    from torchvision_customizer import partial_load, transfer_weights
    from torchvision.models import resnet18
    import torch.nn as nn
    
    print("\n" + "=" * 60)
    print("Example 5: Weight Transfer Utilities")
    print("=" * 60)
    
    # Create source and target models
    source = resnet18(weights="IMAGENET1K_V1")
    
    # Custom model with different classifier
    target = resnet18(weights=None)
    target.fc = nn.Linear(512, 50)  # Different number of classes
    
    # Transfer weights with partial loading
    report = partial_load(
        target,
        source.state_dict(),
        ignore_mismatch=True,
        init_new_layers="kaiming",
    )
    
    print(f"Load ratio: {report.load_ratio:.1%}")
    
    return report


# ============================================================================
# Example 6: Using New Building Blocks
# ============================================================================

def example_new_blocks():
    """Demonstrate new v2.1 building blocks."""
    from torchvision_customizer import (
        CBAMBlock, ECABlock, DropPath, Mish, GeM, MBConv
    )
    
    print("\n" + "=" * 60)
    print("Example 6: New Building Blocks")
    print("=" * 60)
    
    # CBAM Block
    cbam = CBAMBlock(channels=256, reduction=16, spatial_kernel=7)
    x = torch.randn(2, 256, 32, 32)
    out = cbam(x)
    print(f"CBAM: {x.shape} -> {out.shape}")
    
    # ECA Block
    eca = ECABlock(channels=512)
    x = torch.randn(2, 512, 16, 16)
    out = eca(x)
    print(f"ECA: {x.shape} -> {out.shape}")
    
    # DropPath (Stochastic Depth)
    drop_path = DropPath(drop_prob=0.2)
    residual = torch.randn(2, 64, 32, 32)
    out = residual + drop_path(residual)  # Apply to residual branch
    print(f"DropPath: applied with p=0.2")
    
    # Mish Activation
    mish = Mish()
    x = torch.randn(2, 64)
    out = mish(x)
    print(f"Mish: {x.shape} -> {out.shape}")
    
    # GeM Pooling
    gem = GeM(p=3.0, learnable=True)
    x = torch.randn(2, 2048, 7, 7).abs() + 0.1
    out = gem(x)
    print(f"GeM: {x.shape} -> {out.shape}")
    
    # MBConv (EfficientNet style)
    mbconv = MBConv(in_channels=32, out_channels=64, expansion=4, stride=2)
    x = torch.randn(2, 32, 56, 56)
    out = mbconv(x)
    print(f"MBConv: {x.shape} -> {out.shape}")


# ============================================================================
# Example 7: Recipe with Macros
# ============================================================================

def example_recipe_macros():
    """Use YAML-style recipe with macros."""
    from torchvision_customizer.recipe import (
        expand_macros, validate_recipe_config, get_template
    )
    
    print("\n" + "=" * 60)
    print("Example 7: Recipe with Macros")
    print("=" * 60)
    
    # Define recipe with macros
    config = {
        "name": "Custom ResNet",
        "macros": {
            "attention": "se",
            "activation": "relu",
            "dropout": 0.3,
        },
        "backbone": {
            "name": "resnet50",
            "weights": "IMAGENET1K_V2",
            "patches": {
                "layer3": {"wrap": "@attention"},
                "layer4": {"wrap": "@attention"},
            }
        },
        "head": {
            "num_classes": 100,
            "dropout": "@dropout"
        }
    }
    
    # Expand macros
    expanded = expand_macros(config)
    
    print("Original patches:")
    print(f"  layer3: {config['backbone']['patches']['layer3']}")
    print("\nExpanded patches:")
    print(f"  layer3: {expanded['backbone']['patches']['layer3']}")
    
    # Validate
    warnings = validate_recipe_config(expanded, strict=False)
    print(f"\nValidation: {'Passed' if not warnings else 'Warnings found'}")
    
    return expanded


# ============================================================================
# Example 8: Backbone Extraction
# ============================================================================

def example_backbone_extraction():
    """Extract and analyze backbone structure."""
    from torchvision.models import resnet50
    from torchvision_customizer import extract_tiers, get_backbone_info
    
    print("\n" + "=" * 60)
    print("Example 8: Backbone Extraction")
    print("=" * 60)
    
    model = resnet50(weights=None)
    
    # Get backbone info
    info = get_backbone_info(model, "resnet50")
    print(info.summary())
    
    # Extract tiers
    tiers = extract_tiers(model, "resnet50")
    print(f"\nExtracted Tiers:")
    print(f"  Stem: {type(tiers['stem']).__name__}")
    print(f"  Stages: {len(tiers['stages'])} stages")
    print(f"  Head: {type(tiers['head']).__name__}")
    
    return tiers


# ============================================================================
# Run all examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("torchvision-customizer v2.1 Examples")
    print("=" * 60 + "\n")
    
    # Run examples (some may be slow due to model loading)
    example_basic_hybrid()
    example_hybrid_with_attention()
    example_finetune_frozen()
    example_weight_utils()
    example_new_blocks()
    example_recipe_macros()
    example_backbone_extraction()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

