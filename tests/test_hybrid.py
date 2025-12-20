"""Tests for v2.1 Hybrid Module.

Tests the HybridBuilder, weight utilities, and backbone extraction.
"""

import pytest
import torch
import torch.nn as nn

# Skip all tests if torchvision is not available
pytest.importorskip("torchvision")


class TestWeightUtils:
    """Test weight utilities."""
    
    def test_weight_loading_report(self):
        """Test WeightLoadingReport class."""
        from torchvision_customizer.hybrid.weight_utils import WeightLoadingReport
        
        report = WeightLoadingReport()
        report.loaded = ['conv1.weight', 'conv1.bias']
        report.skipped_shape_mismatch = [('fc.weight', (100, 512), (10, 512))]
        report.skipped_missing = ['new_layer.weight']
        
        assert report.total_loaded == 2
        assert report.total_skipped == 2
        assert report.load_ratio == 0.5
        
        summary = report.summary()
        assert 'Weight Loading Report' in summary
        assert 'Loaded:' in summary
    
    def test_partial_load_matching_shapes(self):
        """Test partial_load with matching shapes."""
        from torchvision_customizer.hybrid.weight_utils import partial_load
        
        # Create simple models with same structure
        model1 = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
        model2 = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
        
        # Load weights from model1 to model2
        report = partial_load(model2, model1.state_dict(), verbose=False)
        
        assert report.total_loaded == 4  # weights and biases for 2 layers
        assert report.load_ratio == 1.0
    
    def test_partial_load_shape_mismatch(self):
        """Test partial_load with shape mismatches."""
        from torchvision_customizer.hybrid.weight_utils import partial_load
        
        model1 = nn.Linear(10, 20)
        model2 = nn.Linear(10, 30)  # Different output size
        
        report = partial_load(model2, model1.state_dict(), ignore_mismatch=True, verbose=False)
        
        assert len(report.skipped_shape_mismatch) == 2  # weight and bias
    
    def test_match_state_dict(self):
        """Test state dict matching."""
        from torchvision_customizer.hybrid.weight_utils import match_state_dict
        
        source = {'layer1.weight': torch.randn(10, 5), 'layer2.weight': torch.randn(20, 10)}
        target = {'layer1.weight': torch.randn(10, 5), 'layer3.weight': torch.randn(20, 10)}
        
        mapping = match_state_dict(source, target)
        
        assert 'layer1.weight' in mapping
        assert mapping['layer1.weight'] == 'layer1.weight'
    
    def test_get_layer_shapes(self):
        """Test getting layer shapes."""
        from torchvision_customizer.hybrid.weight_utils import get_layer_shapes
        
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.Linear(100, 10)
        )
        
        shapes = get_layer_shapes(model)
        
        assert '0.weight' in shapes
        assert shapes['0.weight'] == (64, 3, 3, 3)


class TestBackboneExtractor:
    """Test backbone extraction utilities."""
    
    def test_backbone_info(self):
        """Test BackboneInfo class."""
        from torchvision_customizer.hybrid.extractor import BackboneInfo
        
        model = nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU())
        info = BackboneInfo(
            name="test_model",
            model=model,
            stem_parts=["0"],
            stage_parts=[],
            head_parts=["1"],
        )
        
        assert info.name == "test_model"
        assert info.total_params > 0
    
    def test_extract_tiers_resnet(self):
        """Test tier extraction from ResNet."""
        from torchvision.models import resnet18
        from torchvision_customizer.hybrid.extractor import extract_tiers
        
        model = resnet18(weights=None)
        tiers = extract_tiers(model, "resnet18")
        
        assert 'stem' in tiers
        assert 'stages' in tiers
        assert 'head' in tiers
        assert len(tiers['stages']) == 4  # layer1-4
    
    def test_get_backbone_info_resnet(self):
        """Test getting backbone info for ResNet."""
        from torchvision.models import resnet50
        from torchvision_customizer.hybrid.extractor import get_backbone_info
        
        model = resnet50(weights=None)
        info = get_backbone_info(model, "resnet50")
        
        assert info.name == "resnet50"
        assert len(info.stage_parts) == 4
        assert info.total_params > 0
    
    def test_freeze_backbone(self):
        """Test freezing backbone layers."""
        from torchvision.models import resnet18
        from torchvision_customizer.hybrid.extractor import freeze_backbone
        
        model = resnet18(weights=None)
        freeze_backbone(model, freeze_stem=True, freeze_stages=True)
        
        # Check stem is frozen
        assert not model.conv1.weight.requires_grad
        
        # Check stages are frozen
        assert not model.layer1[0].conv1.weight.requires_grad


class TestHybridBuilder:
    """Test HybridBuilder class."""
    
    def test_list_backbones(self):
        """Test listing supported backbones."""
        from torchvision_customizer.hybrid import HybridBuilder
        
        backbones = HybridBuilder.list_backbones()
        
        assert 'resnet50' in backbones
        assert 'efficientnet_b0' in backbones
        assert len(backbones) > 20
    
    def test_from_torchvision_resnet(self):
        """Test loading ResNet from torchvision."""
        from torchvision_customizer.hybrid import HybridBuilder
        
        builder = HybridBuilder()
        model = builder.from_torchvision(
            "resnet18",
            weights=None,  # No pretrained for faster test
            num_classes=10,
            verbose=False,
        )
        
        # Check it's a valid model
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        
        assert out.shape == (2, 10)
    
    def test_from_torchvision_with_patches(self):
        """Test loading with patches."""
        from torchvision_customizer.hybrid import HybridBuilder
        
        builder = HybridBuilder()
        model = builder.from_torchvision(
            "resnet18",
            weights=None,
            patches={
                "layer3": {"wrap": "se"},
            },
            num_classes=10,
            verbose=False,
        )
        
        assert len(model.modifications) == 1
        assert "layer3" in model.modifications[0]
        
        # Check forward pass works
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 10)
    
    def test_hybrid_model_methods(self):
        """Test HybridModel methods."""
        from torchvision_customizer.hybrid import HybridBuilder
        
        builder = HybridBuilder()
        model = builder.from_torchvision(
            "resnet18",
            weights=None,
            num_classes=10,
            verbose=False,
        )
        
        # Test count_parameters
        params = model.count_parameters()
        assert params > 0
        
        # Test explain
        explanation = model.explain()
        assert "HybridModel" in explanation
        
        # Test forward_features
        x = torch.randn(1, 3, 224, 224)
        features = model.forward_features(x)
        assert features.shape[0] == 1
        
        # Test freeze_backbone
        model.freeze_backbone()
        assert not model.stem[0].weight.requires_grad
        
        # Test unfreeze
        model.unfreeze_all()
        assert model.stem[0].weight.requires_grad


class TestAdvancedBlocks:
    """Test new v2.1 building blocks."""
    
    def test_cbam_block(self):
        """Test CBAMBlock."""
        from torchvision_customizer.blocks import CBAMBlock
        
        block = CBAMBlock(channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        
        assert out.shape == x.shape
    
    def test_eca_block(self):
        """Test ECABlock."""
        from torchvision_customizer.blocks import ECABlock
        
        block = ECABlock(channels=256)
        x = torch.randn(2, 256, 16, 16)
        out = block(x)
        
        assert out.shape == x.shape
    
    def test_drop_path(self):
        """Test DropPath."""
        from torchvision_customizer.blocks import DropPath
        
        drop = DropPath(drop_prob=0.5)
        x = torch.randn(4, 64, 8, 8)
        
        # Training mode
        drop.train()
        out = drop(x)
        assert out.shape == x.shape
        
        # Eval mode (no drop)
        drop.eval()
        out = drop(x)
        assert torch.allclose(out, x)
    
    def test_mish(self):
        """Test Mish activation."""
        from torchvision_customizer.blocks import Mish
        
        mish = Mish()
        x = torch.randn(2, 64)
        out = mish(x)
        
        assert out.shape == x.shape
        # Mish should be different from input for non-zero values
        assert not torch.allclose(out, x)
    
    def test_gem_pooling(self):
        """Test GeM pooling."""
        from torchvision_customizer.blocks import GeM
        
        gem = GeM(p=3.0)
        x = torch.randn(2, 512, 7, 7).abs() + 0.1  # Ensure positive
        out = gem(x)
        
        assert out.shape == (2, 512, 1, 1)
    
    def test_mbconv(self):
        """Test MBConv block."""
        from torchvision_customizer.blocks import MBConv
        
        block = MBConv(in_channels=32, out_channels=64, expansion=4, stride=2)
        x = torch.randn(2, 32, 56, 56)
        out = block(x)
        
        assert out.shape == (2, 64, 28, 28)
    
    def test_fused_mbconv(self):
        """Test FusedMBConv block."""
        from torchvision_customizer.blocks import FusedMBConv
        
        block = FusedMBConv(in_channels=32, out_channels=64, expansion=4, stride=1)
        x = torch.randn(2, 32, 56, 56)
        out = block(x)
        
        assert out.shape == (2, 64, 56, 56)


class TestRecipeEnhancements:
    """Test v2.1 recipe enhancements."""
    
    def test_validate_recipe_config(self):
        """Test recipe validation."""
        from torchvision_customizer.recipe.schema import validate_recipe_config
        
        valid_config = {
            "backbone": {"name": "resnet50", "weights": "DEFAULT"},
            "head": {"num_classes": 100}
        }
        
        warnings = validate_recipe_config(valid_config, strict=False)
        assert len(warnings) == 0
    
    def test_expand_macros(self):
        """Test macro expansion."""
        from torchvision_customizer.recipe.schema import expand_macros
        
        config = {
            "macros": {"attention": "SEBlock", "classes": 100},
            "stages": [{"attention": "@attention"}],
            "head": {"num_classes": "@classes"}
        }
        
        expanded = expand_macros(config)
        
        assert expanded["stages"][0]["attention"] == "SEBlock"
    
    def test_merge_recipes(self):
        """Test recipe merging."""
        from torchvision_customizer.recipe.schema import merge_recipes
        
        base = {
            "stem": {"channels": 64},
            "stages": [{"channels": 128}],
            "head": {"num_classes": 1000}
        }
        
        override = {
            "head": {"num_classes": 100}
        }
        
        merged = merge_recipes(base, override)
        
        assert merged["head"]["num_classes"] == 100
        assert merged["stem"]["channels"] == 64
    
    def test_get_template(self):
        """Test getting recipe templates."""
        from torchvision_customizer.recipe.schema import get_template
        
        template = get_template("resnet_base")
        
        assert "stem" in template
        assert "stages" in template
        assert len(template["stages"]) == 4


class TestRegistry:
    """Test registry updates for v2.1."""
    
    def test_new_blocks_registered(self):
        """Test that new blocks are registered."""
        from torchvision_customizer import registry
        
        blocks = registry.list_components()
        
        assert 'cbam_block' in blocks
        assert 'eca' in blocks
        assert 'drop_path' in blocks
        assert 'mish' in blocks
        assert 'gem' in blocks
        assert 'mbconv' in blocks
    
    def test_get_new_blocks(self):
        """Test getting new blocks from registry."""
        from torchvision_customizer import registry
        
        CBAMBlock = registry.get('cbam_block')
        ECABlock = registry.get('eca')
        
        assert CBAMBlock is not None
        assert ECABlock is not None

