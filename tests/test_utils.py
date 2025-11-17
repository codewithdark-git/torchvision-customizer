"""Tests for model summary and validation utilities.

Tests cover:
- Output shape calculation
- Layer summaries
- Parameter analysis
- Memory and FLOPs estimation
- Architecture validation
"""

import pytest
import torch
import torch.nn as nn

from torchvision_customizer import CustomCNN
from torchvision_customizer.utils import (
    calculate_output_shape,
    count_parameters_by_type,
    estimate_model_size,
    get_layer_summary,
    get_memory_usage,
    get_model_flops,
    predict_memory_usage,
    print_model_summary,
    validate_architecture,
    validate_channel_progression,
    validate_spatial_dimensions,
)


class TestCalculateOutputShape:
    """Test output shape calculation."""

    def test_basic_output_shape(self) -> None:
        """Test basic output shape calculation."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        result = calculate_output_shape(model, input_shape=(2, 3, 32, 32))
        assert result["success"] is True
        assert result["output_shape"] == (2, 10)

    def test_large_input_output_shape(self) -> None:
        """Test output shape with large input."""
        model = CustomCNN(input_shape=(3, 224, 224), num_classes=1000)
        result = calculate_output_shape(model, input_shape=(1, 3, 224, 224))
        assert result["success"] is True
        assert result["output_shape"][0] == 1
        assert result["output_shape"][1] == 1000

    def test_batch_size_variations(self) -> None:
        """Test output shape with different batch sizes."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        for batch_size in [1, 2, 8, 16]:
            result = calculate_output_shape(model, input_shape=(batch_size, 3, 32, 32))
            assert result["success"] is True
            assert result["output_shape"][0] == batch_size

    def test_invalid_input_shape(self) -> None:
        """Test with invalid input shape."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        with pytest.raises(ValueError):
            calculate_output_shape(model, input_shape=(3, 32))


class TestLayerSummary:
    """Test layer-by-layer summary."""

    def test_basic_layer_summary(self) -> None:
        """Test basic layer summary generation."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        layers = get_layer_summary(model, input_shape=(2, 3, 32, 32))
        assert len(layers) > 0
        assert all("name" in layer for layer in layers)
        assert all("type" in layer for layer in layers)
        assert all("output_shape" in layer for layer in layers)

    def test_layer_names(self) -> None:
        """Test that layer names are recorded."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        layers = get_layer_summary(model, input_shape=(2, 3, 32, 32))
        names = [layer["name"] for layer in layers]
        assert len(names) > 0
        assert all(isinstance(name, str) for name in names)

    def test_layer_types(self) -> None:
        """Test that layer types are correctly identified."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        layers = get_layer_summary(model, input_shape=(2, 3, 32, 32))
        types = [layer["type"] for layer in layers]
        assert len(types) > 0
        assert any(t in types for t in ["Conv2d", "BatchNorm2d", "ReLU", "Linear"])

    def test_invalid_input_shape_layer_summary(self) -> None:
        """Test layer summary with invalid input shape."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        with pytest.raises(ValueError):
            get_layer_summary(model, input_shape=(3, 32))


class TestParameterCounting:
    """Test parameter counting utilities."""

    def test_basic_parameter_count(self) -> None:
        """Test basic parameter counting."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        param_count = count_parameters_by_type(model)
        assert isinstance(param_count, dict)
        assert all("total" in v for v in param_count.values())
        assert all("trainable" in v for v in param_count.values())
        assert all("count" in v for v in param_count.values())

    def test_conv2d_parameters(self) -> None:
        """Test Conv2d parameter counting."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        param_count = count_parameters_by_type(model)
        assert "Conv2d" in param_count
        assert param_count["Conv2d"]["count"] > 0

    def test_linear_parameters(self) -> None:
        """Test Linear layer parameter counting."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        param_count = count_parameters_by_type(model)
        assert "Linear" in param_count
        assert param_count["Linear"]["count"] > 0

    def test_parameter_consistency(self) -> None:
        """Test consistency between parameter counts."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        param_count = count_parameters_by_type(model)
        total = sum(v["total"] for v in param_count.values())
        model_total = sum(p.numel() for p in model.parameters())
        assert total == model_total


class TestMemoryUsage:
    """Test memory usage estimation."""

    def test_basic_memory_usage(self) -> None:
        """Test basic memory usage calculation."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        memory = get_memory_usage(model, input_shape=(2, 3, 32, 32))
        assert "parameter_memory_mb" in memory
        assert "activation_memory_mb" in memory
        assert "total_memory_mb" in memory
        assert memory["parameter_memory_mb"] > 0

    def test_memory_values_positive(self) -> None:
        """Test that memory values are positive."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        memory = get_memory_usage(model, input_shape=(2, 3, 32, 32))
        assert memory["parameter_memory_mb"] > 0
        assert memory["activation_memory_mb"] > 0
        assert memory["total_memory_mb"] > 0

    def test_memory_ordering(self) -> None:
        """Test that memory values are properly ordered."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        memory = get_memory_usage(model, input_shape=(2, 3, 32, 32))
        total = memory["total_memory_mb"]
        param = memory["parameter_memory_mb"]
        assert total >= param

    def test_invalid_input_shape_memory(self) -> None:
        """Test memory with invalid input shape."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        with pytest.raises(ValueError):
            get_memory_usage(model, input_shape=(3, 32))


class TestFLOPsEstimation:
    """Test FLOPs estimation."""

    def test_basic_flops_estimation(self) -> None:
        """Test basic FLOPs estimation."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        flops = get_model_flops(model, input_shape=(2, 3, 32, 32))
        assert "total_flops" in flops
        assert "total_flops_in_billions" in flops
        assert flops["success"] is True
        assert flops["total_flops"] > 0

    def test_flops_positive(self) -> None:
        """Test that FLOPs value is positive."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        flops = get_model_flops(model, input_shape=(2, 3, 32, 32))
        assert flops["total_flops"] > 0
        assert flops["total_flops_in_billions"] > 0

    def test_flops_scaling_with_batch_size(self) -> None:
        """Test that FLOPs scale with batch size."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        flops_b1 = get_model_flops(model, input_shape=(1, 3, 32, 32))
        flops_b4 = get_model_flops(model, input_shape=(4, 3, 32, 32))
        # FLOPs should scale roughly linearly with batch size
        assert flops_b4["total_flops"] > flops_b1["total_flops"]

    def test_invalid_input_shape_flops(self) -> None:
        """Test FLOPs with invalid input shape."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        with pytest.raises(ValueError):
            get_model_flops(model, input_shape=(3, 32))


class TestModelSummary:
    """Test complete model summary."""

    def test_print_model_summary_structure(self) -> None:
        """Test that model summary has correct structure."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        summary = print_model_summary(model, input_shape=(2, 3, 32, 32), verbose=False)
        assert "model_name" in summary
        assert "input_shape" in summary
        assert "output_shape" in summary
        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert "memory" in summary
        assert "flops" in summary

    def test_summary_parameter_values(self) -> None:
        """Test that summary contains reasonable parameter values."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        summary = print_model_summary(model, input_shape=(2, 3, 32, 32), verbose=False)
        assert summary["total_parameters"] > 0
        assert summary["trainable_parameters"] > 0
        assert summary["trainable_parameters"] <= summary["total_parameters"]

    def test_summary_with_verbose(self) -> None:
        """Test summary with verbose output."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        summary = print_model_summary(model, input_shape=(2, 3, 32, 32), verbose=True)
        assert summary is not None


class TestValidateSpatialDimensions:
    """Test spatial dimension validation."""

    def test_valid_spatial_dimensions(self) -> None:
        """Test validation of valid spatial dimensions."""
        result = validate_spatial_dimensions(
            input_shape=(3, 224, 224),
            num_conv_blocks=4,
        )
        assert "valid" in result
        assert "input_spatial" in result
        assert "final_spatial" in result

    def test_invalid_input_shape_validation(self) -> None:
        """Test validation with invalid input shape."""
        result = validate_spatial_dimensions(
            input_shape=(3, 224),
            num_conv_blocks=4,
        )
        assert result["valid"] is False
        assert "error" in result

    def test_too_many_pooling_operations(self) -> None:
        """Test that warning is issued for too many pooling ops."""
        result = validate_spatial_dimensions(
            input_shape=(3, 16, 16),
            num_conv_blocks=8,  # Many pooling ops
        )
        assert len(result.get("warnings", [])) > 0

    def test_spatial_dimension_calculation(self) -> None:
        """Test that spatial dimensions are calculated correctly."""
        result = validate_spatial_dimensions(
            input_shape=(3, 64, 64),
            num_conv_blocks=2,
            pooling_kernel_size=2,
            pooling_stride=2,
        )
        assert result["input_spatial"] == (64, 64)
        # After 2 pooling ops: 64 -> 32 -> 16
        assert result["final_spatial"] == (16, 16)


class TestValidateChannelProgression:
    """Test channel progression validation."""

    def test_valid_channel_progression(self) -> None:
        """Test validation of valid channel progression."""
        result = validate_channel_progression(
            channels=[64, 128, 256, 512],
            num_conv_blocks=4,
        )
        assert result["valid"] is True
        assert "warnings" in result

    def test_channel_length_mismatch(self) -> None:
        """Test validation with mismatched channel length."""
        result = validate_channel_progression(
            channels=[64, 128, 256],
            num_conv_blocks=4,
        )
        assert result["valid"] is False
        assert "error" in result

    def test_growth_rate_calculation(self) -> None:
        """Test growth rate calculation."""
        result = validate_channel_progression(
            channels=[64, 128, 256, 512],
            num_conv_blocks=4,
        )
        assert "growth_rate" in result
        assert result["growth_rate"] > 1  # Channels increase

    def test_high_growth_rate_warning(self) -> None:
        """Test warning for high growth rate."""
        result = validate_channel_progression(
            channels=[64, 512, 4096, 32768],
            num_conv_blocks=4,
        )
        assert len(result.get("warnings", [])) > 0


class TestValidateArchitecture:
    """Test complete architecture validation."""

    def test_valid_architecture(self) -> None:
        """Test validation of valid architecture."""
        result = validate_architecture(
            input_shape=(3, 224, 224),
            num_classes=1000,
            num_conv_blocks=5,
            channels=[64, 128, 256, 512, 512],
        )
        assert result["valid"] is True
        assert "checks" in result

    def test_invalid_input_shape_architecture(self) -> None:
        """Test architecture validation with invalid input shape."""
        result = validate_architecture(
            input_shape=(3, 224),
            num_classes=1000,
            num_conv_blocks=5,
        )
        assert result["valid"] is False

    def test_invalid_num_classes(self) -> None:
        """Test architecture validation with invalid num_classes."""
        result = validate_architecture(
            input_shape=(3, 224, 224),
            num_classes=-1,
            num_conv_blocks=5,
        )
        assert result["valid"] is False

    def test_architecture_checks_structure(self) -> None:
        """Test that all checks are performed."""
        result = validate_architecture(
            input_shape=(3, 224, 224),
            num_classes=1000,
            num_conv_blocks=5,
            channels=[64, 128, 256, 512, 512],
        )
        assert "spatial_dimensions" in result["checks"]
        assert "channels" in result["checks"]


class TestEstimateModelSize:
    """Test model size estimation."""

    def test_basic_size_estimation(self) -> None:
        """Test basic model size estimation."""
        estimate = estimate_model_size(
            num_conv_blocks=4,
            channels=[64, 128, 256, 512],
            num_classes=1000,
        )
        assert "conv_params" in estimate
        assert "linear_params" in estimate
        assert "total_params" in estimate
        assert estimate["total_params"] > 0

    def test_size_estimation_consistency(self) -> None:
        """Test consistency of size estimation."""
        estimate = estimate_model_size(
            num_conv_blocks=4,
            channels=[64, 128, 256, 512],
            num_classes=1000,
        )
        expected_total = estimate["conv_params"] + estimate["bn_params"] + estimate["linear_params"]
        assert estimate["total_params"] == expected_total

    def test_without_batch_norm(self) -> None:
        """Test estimation without batch norm."""
        with_bn = estimate_model_size(
            num_conv_blocks=4,
            channels=[64, 128, 256, 512],
            num_classes=1000,
            include_batch_norm=True,
        )
        without_bn = estimate_model_size(
            num_conv_blocks=4,
            channels=[64, 128, 256, 512],
            num_classes=1000,
            include_batch_norm=False,
        )
        assert with_bn["total_params"] > without_bn["total_params"]


class TestPredictMemoryUsage:
    """Test memory usage prediction."""

    def test_basic_memory_prediction(self) -> None:
        """Test basic memory usage prediction."""
        memory = predict_memory_usage(
            num_conv_blocks=4,
            channels=[64, 128, 256, 512],
            num_classes=1000,
        )
        assert "parameter_memory_mb" in memory
        assert "activation_memory_mb" in memory
        assert "total_training_memory_mb" in memory
        assert "inference_memory_mb" in memory

    def test_memory_values_positive(self) -> None:
        """Test that memory predictions are positive."""
        memory = predict_memory_usage(
            num_conv_blocks=4,
            channels=[64, 128, 256, 512],
            num_classes=1000,
        )
        assert all(v > 0 for v in [
            memory["parameter_memory_mb"],
            memory["activation_memory_mb"],
            memory["total_training_memory_mb"],
        ])

    def test_training_memory_larger_than_inference(self) -> None:
        """Test that training memory is larger than inference."""
        memory = predict_memory_usage(
            num_conv_blocks=4,
            channels=[64, 128, 256, 512],
            num_classes=1000,
        )
        assert memory["total_training_memory_mb"] > memory["inference_memory_mb"]

    def test_batch_size_scaling(self) -> None:
        """Test that memory scales with batch size."""
        memory_b1 = predict_memory_usage(
            num_conv_blocks=4,
            channels=[64, 128, 256, 512],
            num_classes=1000,
            batch_size=1,
        )
        memory_b32 = predict_memory_usage(
            num_conv_blocks=4,
            channels=[64, 128, 256, 512],
            num_classes=1000,
            batch_size=32,
        )
        assert memory_b32["total_training_memory_mb"] > memory_b1["total_training_memory_mb"]
