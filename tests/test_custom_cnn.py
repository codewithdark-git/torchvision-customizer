"""Tests for CustomCNN model."""

import pytest
import torch
import torch.nn as nn
from torchvision_customizer.models import CustomCNN


class TestCustomCNNInitialization:
    """Test CustomCNN initialization."""

    def test_basic_initialization(self) -> None:
        """Test basic initialization with default parameters."""
        model = CustomCNN(input_shape=(3, 224, 224), num_classes=1000)
        assert isinstance(model, nn.Module)
        assert model.num_classes == 1000
        assert model.num_conv_blocks == 4
        assert model.activation == 'relu'

    def test_custom_num_conv_blocks(self) -> None:
        """Test with custom number of conv blocks."""
        model = CustomCNN(
            input_shape=(3, 224, 224),
            num_classes=10,
            num_conv_blocks=6,
        )
        assert model.num_conv_blocks == 6
        assert len(model.channels) == 6

    def test_auto_channels_generation(self) -> None:
        """Test automatic channel progression generation."""
        model = CustomCNN(
            input_shape=(3, 32, 32),
            num_classes=10,
            num_conv_blocks=5,
            channels='auto',
        )
        expected_channels = [64, 128, 256, 512, 1024]
        assert model.channels == expected_channels

    def test_custom_channels_specification(self) -> None:
        """Test with manually specified channels."""
        custom_channels = [32, 64, 128, 256]
        model = CustomCNN(
            input_shape=(3, 224, 224),
            num_classes=100,
            num_conv_blocks=4,
            channels=custom_channels,
        )
        assert model.channels == custom_channels

    def test_different_activations(self) -> None:
        """Test with different activation functions."""
        for activation in ['relu', 'leaky_relu', 'gelu', 'silu']:
            model = CustomCNN(
                input_shape=(3, 224, 224),
                num_classes=10,
                activation=activation,
            )
            assert model.activation == activation

    def test_without_batchnorm(self) -> None:
        """Test model without batch normalization."""
        model = CustomCNN(
            input_shape=(3, 224, 224),
            num_classes=10,
            use_batchnorm=False,
        )
        assert model.use_batchnorm is False

    def test_with_dropout(self) -> None:
        """Test model with dropout."""
        model = CustomCNN(
            input_shape=(3, 224, 224),
            num_classes=10,
            dropout_rate=0.5,
        )
        assert model.dropout_rate == 0.5

    def test_without_pooling(self) -> None:
        """Test model without pooling."""
        model = CustomCNN(
            input_shape=(3, 224, 224),
            num_classes=10,
            use_pooling=False,
        )
        assert model.use_pooling is False

    def test_different_pooling_types(self) -> None:
        """Test with different pooling types."""
        for pool_type in ['max', 'avg', 'adaptive_avg']:
            model = CustomCNN(
                input_shape=(3, 224, 224),
                num_classes=10,
                pooling_type=pool_type,
            )
            assert model.pooling_type == pool_type

    def test_invalid_input_shape_2d(self) -> None:
        """Test that 2D input shape raises error."""
        with pytest.raises(ValueError):
            CustomCNN(input_shape=(224, 224), num_classes=10)

    def test_invalid_input_shape_4d(self) -> None:
        """Test that 4D input shape raises error."""
        with pytest.raises(ValueError):
            CustomCNN(input_shape=(3, 224, 224, 3), num_classes=10)

    def test_invalid_input_shape_negative(self) -> None:
        """Test that negative dimensions raise error."""
        with pytest.raises(ValueError):
            CustomCNN(input_shape=(3, -224, 224), num_classes=10)

    def test_invalid_num_classes(self) -> None:
        """Test that invalid num_classes raises error."""
        with pytest.raises(ValueError):
            CustomCNN(input_shape=(3, 224, 224), num_classes=-10)

    def test_invalid_num_conv_blocks(self) -> None:
        """Test that invalid num_conv_blocks raises error."""
        with pytest.raises(ValueError):
            CustomCNN(input_shape=(3, 224, 224), num_classes=10, num_conv_blocks=0)

    def test_invalid_dropout_rate(self) -> None:
        """Test that invalid dropout_rate raises error."""
        with pytest.raises(ValueError):
            CustomCNN(input_shape=(3, 224, 224), num_classes=10, dropout_rate=1.5)

    def test_channels_list_length_mismatch(self) -> None:
        """Test that mismatched channels list length raises error."""
        with pytest.raises(ValueError):
            CustomCNN(
                input_shape=(3, 224, 224),
                num_classes=10,
                num_conv_blocks=4,
                channels=[64, 128, 256],  # Only 3 channels for 4 blocks
            )

    def test_channels_invalid_values(self) -> None:
        """Test that invalid channel values raise error."""
        with pytest.raises(ValueError):
            CustomCNN(
                input_shape=(3, 224, 224),
                num_classes=10,
                channels=[64, -128, 256, 512],
            )

    def test_invalid_channels_mode(self) -> None:
        """Test that invalid channels mode raises error."""
        with pytest.raises(ValueError):
            CustomCNN(
                input_shape=(3, 224, 224),
                num_classes=10,
                channels='invalid_mode',
            )


class TestCustomCNNForward:
    """Test CustomCNN forward pass."""

    def test_forward_basic(self) -> None:
        """Test basic forward pass."""
        model = CustomCNN(input_shape=(3, 224, 224), num_classes=1000)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == torch.Size([2, 1000])

    def test_forward_batch_size_1(self) -> None:
        """Test forward pass with batch size 1."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        assert output.shape == torch.Size([1, 10])

    def test_forward_large_batch(self) -> None:
        """Test forward pass with large batch size."""
        model = CustomCNN(input_shape=(3, 64, 64), num_classes=100)
        x = torch.randn(128, 3, 64, 64)
        output = model(x)
        assert output.shape == torch.Size([128, 100])

    def test_forward_different_input_sizes(self) -> None:
        """Test forward pass with various input sizes."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        for h, w in [(32, 32), (64, 64), (16, 16)]:
            x = torch.randn(2, 3, h, w)
            output = model(x)
            assert output.shape == torch.Size([2, 10])

    def test_forward_single_channel(self) -> None:
        """Test forward pass with single channel input."""
        model = CustomCNN(input_shape=(1, 28, 28), num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        assert output.shape == torch.Size([2, 10])

    def test_forward_wrong_channel_count(self) -> None:
        """Test that wrong channel count raises error."""
        model = CustomCNN(input_shape=(3, 224, 224), num_classes=10)
        x = torch.randn(2, 4, 224, 224)  # Wrong channel count
        with pytest.raises(ValueError):
            model(x)

    def test_forward_wrong_input_dims(self) -> None:
        """Test that wrong input dimensions raise error."""
        model = CustomCNN(input_shape=(3, 224, 224), num_classes=10)
        x = torch.randn(2, 3, 224)  # 3D instead of 4D
        with pytest.raises(ValueError):
            model(x)

    def test_forward_gradient_flow(self) -> None:
        """Test gradient flow through the model."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_forward_training_eval_mode(self) -> None:
        """Test model in training and evaluation modes."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        x = torch.randn(2, 3, 32, 32)

        # Training mode
        model.train()
        output_train1 = model(x)

        # Eval mode (should be deterministic with BatchNorm)
        model.eval()
        with torch.no_grad():
            output_eval = model(x)

        # Training mode again
        model.train()
        output_train2 = model(x)

        # Outputs should have correct shape
        assert output_train1.shape == torch.Size([2, 10])
        assert output_eval.shape == torch.Size([2, 10])
        assert output_train2.shape == torch.Size([2, 10])


class TestCustomCNNMethods:
    """Test CustomCNN utility methods."""

    def test_count_parameters_basic(self) -> None:
        """Test parameter counting."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10, num_conv_blocks=2)
        params = model.count_parameters()
        assert params > 0

    def test_count_parameters_trainable_only(self) -> None:
        """Test trainable parameter counting."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        trainable = model.count_parameters(trainable_only=True)
        total = model.count_parameters(trainable_only=False)
        assert trainable <= total
        assert trainable > 0

    def test_count_parameters_larger_model(self) -> None:
        """Test parameter counting for larger model."""
        model = CustomCNN(
            input_shape=(3, 224, 224),
            num_classes=1000,
            num_conv_blocks=6,
        )
        params = model.count_parameters()
        assert params > 1000  # Should have millions of parameters

    def test_get_config(self) -> None:
        """Test getting model configuration."""
        config_input = {
            'input_shape': (3, 32, 32),
            'num_classes': 10,
            'num_conv_blocks': 3,
            'activation': 'leaky_relu',
            'use_batchnorm': False,
            'dropout_rate': 0.5,
        }
        model = CustomCNN(**config_input)
        config = model.get_config()

        assert config['input_shape'] == (3, 32, 32)
        assert config['num_classes'] == 10
        assert config['num_conv_blocks'] == 3
        assert config['activation'] == 'leaky_relu'
        assert config['use_batchnorm'] is False
        assert config['dropout_rate'] == 0.5

    def test_repr(self) -> None:
        """Test string representation."""
        model = CustomCNN(input_shape=(3, 224, 224), num_classes=1000)
        repr_str = repr(model)
        assert 'CustomCNN' in repr_str
        assert '3' in repr_str
        assert '1000' in repr_str


class TestCustomCNNArchitectures:
    """Test different model architectures."""

    def test_small_model_cifar10(self) -> None:
        """Test small model configuration for CIFAR-10."""
        model = CustomCNN(
            input_shape=(3, 32, 32),
            num_classes=10,
            num_conv_blocks=3,
            channels=[32, 64, 128],
            use_pooling=True,
        )
        x = torch.randn(32, 3, 32, 32)
        output = model(x)
        assert output.shape == torch.Size([32, 10])

    def test_medium_model_imagenet(self) -> None:
        """Test medium model configuration for ImageNet."""
        model = CustomCNN(
            input_shape=(3, 224, 224),
            num_classes=1000,
            num_conv_blocks=5,
            channels=[64, 128, 256, 512, 512],
            activation='relu',
            use_pooling=True,
        )
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == torch.Size([2, 1000])

    def test_model_with_dropout_cifar100(self) -> None:
        """Test model with dropout for CIFAR-100."""
        model = CustomCNN(
            input_shape=(3, 32, 32),
            num_classes=100,
            num_conv_blocks=4,
            channels='auto',
            dropout_rate=0.3,
            use_batchnorm=True,
        )
        x = torch.randn(64, 3, 32, 32)
        output = model(x)
        assert output.shape == torch.Size([64, 100])

    def test_model_without_pooling(self) -> None:
        """Test model without pooling layers."""
        model = CustomCNN(
            input_shape=(3, 32, 32),
            num_classes=10,
            num_conv_blocks=2,
            use_pooling=False,
        )
        x = torch.randn(4, 3, 32, 32)
        output = model(x)
        assert output.shape == torch.Size([4, 10])


class TestCustomCNNEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_conv_block(self) -> None:
        """Test model with only one conv block."""
        model = CustomCNN(
            input_shape=(3, 224, 224),
            num_classes=10,
            num_conv_blocks=1,
        )
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == torch.Size([2, 10])

    def test_many_conv_blocks(self) -> None:
        """Test model with many conv blocks."""
        model = CustomCNN(
            input_shape=(3, 64, 64),
            num_classes=10,
            num_conv_blocks=4,
            channels=[32, 64, 128, 128],
        )
        model.eval()  # Set to eval mode to avoid batch norm issues with batch_size=1
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        assert output.shape == torch.Size([2, 10])

    def test_very_small_input(self) -> None:
        """Test with very small input."""
        model = CustomCNN(
            input_shape=(1, 8, 8),
            num_classes=5,
            num_conv_blocks=2,
        )
        x = torch.randn(2, 1, 8, 8)
        output = model(x)
        assert output.shape == torch.Size([2, 5])

    def test_many_output_classes(self) -> None:
        """Test with many output classes."""
        model = CustomCNN(
            input_shape=(3, 32, 32),
            num_classes=10000,
            num_conv_blocks=3,
        )
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == torch.Size([2, 10000])

    def test_model_device_movement(self) -> None:
        """Test moving model to different devices."""
        model = CustomCNN(input_shape=(3, 32, 32), num_classes=10)
        x = torch.randn(2, 3, 32, 32)

        # Test on CPU (default)
        output = model(x)
        assert output.shape == torch.Size([2, 10])

        # Test model can be placed on CPU explicitly
        model = model.cpu()
        output = model(x)
        assert output.shape == torch.Size([2, 10])

    def test_different_pooling_kernels(self) -> None:
        """Test with different pooling kernel sizes."""
        for kernel_size in [2, 3, 4]:
            model = CustomCNN(
                input_shape=(3, 64, 64),
                num_classes=10,
                num_conv_blocks=3,
                pooling_kernel_size=kernel_size,
                pooling_stride=kernel_size,
            )
            x = torch.randn(2, 3, 64, 64)
            output = model(x)
            assert output.shape == torch.Size([2, 10])
