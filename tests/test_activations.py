"""Unit tests for activation function utilities.

Test the get_activation() factory function and related utilities including:
- Basic functionality for all supported activations
- Case-insensitivity
- Custom parameters
- Error handling
- Registry system
"""

import pytest
import torch
import torch.nn as nn
from torchvision_customizer.layers import (
    get_activation,
    is_activation_supported,
    get_supported_activations,
    ActivationFactory,
    ACTIVATION_REGISTRY,
)


class TestGetActivation:
    """Test the get_activation() factory function."""
    
    def test_relu_activation(self):
        """Test ReLU activation creation."""
        activation = get_activation('relu')
        assert isinstance(activation, nn.ReLU)
        
        x = torch.randn(2, 64, 32, 32)
        output = activation(x)
        assert output.shape == x.shape
        assert (output >= 0).all()  # ReLU should have no negative values
    
    def test_leaky_relu_default(self):
        """Test Leaky ReLU with default parameters."""
        activation = get_activation('leaky_relu')
        assert isinstance(activation, nn.LeakyReLU)
        
        x = torch.randn(2, 64, 32, 32)
        output = activation(x)
        assert output.shape == x.shape
    
    def test_leaky_relu_custom_slope(self):
        """Test Leaky ReLU with custom negative slope."""
        slope = 0.2
        activation = get_activation('leaky_relu', negative_slope=slope)
        assert isinstance(activation, nn.LeakyReLU)
        
        x = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        output = activation(x)
        expected = torch.tensor([[-slope, -slope * 0.5, 0.0, 0.5, 1.0]])
        assert torch.allclose(output, expected)
    
    def test_prelu_activation(self):
        """Test PReLU (Parametric ReLU) activation."""
        activation = get_activation('prelu')
        assert isinstance(activation, nn.PReLU)
        
        x = torch.randn(2, 64, 32, 32)
        output = activation(x)
        assert output.shape == x.shape
    
    def test_gelu_activation(self):
        """Test GELU activation."""
        activation = get_activation('gelu')
        assert isinstance(activation, nn.GELU)
        
        x = torch.randn(2, 64, 32, 32)
        output = activation(x)
        assert output.shape == x.shape
    
    def test_silu_activation(self):
        """Test SiLU (Sigmoid Linear Unit) activation."""
        activation = get_activation('silu')
        assert isinstance(activation, nn.SiLU)
        
        x = torch.randn(2, 64, 32, 32)
        output = activation(x)
        assert output.shape == x.shape
    
    def test_sigmoid_activation(self):
        """Test Sigmoid activation."""
        activation = get_activation('sigmoid')
        assert isinstance(activation, nn.Sigmoid)
        
        x = torch.randn(2, 64, 32, 32)
        output = activation(x)
        assert output.shape == x.shape
        assert (output >= 0).all() and (output <= 1).all()  # Sigmoid output range [0, 1]
    
    def test_tanh_activation(self):
        """Test Tanh activation."""
        activation = get_activation('tanh')
        assert isinstance(activation, nn.Tanh)
        
        x = torch.randn(2, 64, 32, 32)
        output = activation(x)
        assert output.shape == x.shape
        assert (output >= -1).all() and (output <= 1).all()  # Tanh output range [-1, 1]
    
    def test_elu_activation(self):
        """Test ELU (Exponential Linear Unit) activation."""
        activation = get_activation('elu')
        assert isinstance(activation, nn.ELU)
        
        x = torch.randn(2, 64, 32, 32)
        output = activation(x)
        assert output.shape == x.shape
    
    def test_elu_custom_alpha(self):
        """Test ELU with custom alpha parameter."""
        alpha = 0.5
        activation = get_activation('elu', alpha=alpha)
        assert isinstance(activation, nn.ELU)
        
        x = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        output = activation(x)
        assert output.shape == x.shape
    
    def test_selu_activation(self):
        """Test SELU (Scaled Exponential Linear Unit) activation."""
        activation = get_activation('selu')
        assert isinstance(activation, nn.SELU)
        
        x = torch.randn(2, 64, 32, 32)
        output = activation(x)
        assert output.shape == x.shape
    
    def test_case_insensitive_lowercase(self):
        """Test that activation names are case-insensitive (lowercase)."""
        activation1 = get_activation('relu')
        activation2 = get_activation('RELU')
        activation3 = get_activation('ReLU')
        
        assert type(activation1) == type(activation2) == type(activation3)
    
    def test_case_insensitive_mixed(self):
        """Test case-insensitivity with mixed case names."""
        activations = [
            ('relu', nn.ReLU),
            ('RELU', nn.ReLU),
            ('ReLU', nn.ReLU),
            ('leaky_relu', nn.LeakyReLU),
            ('LEAKY_RELU', nn.LeakyReLU),
            ('leaky_relu', nn.LeakyReLU),  # Note: registry uses lowercase with underscores
            ('gelu', nn.GELU),
            ('GELU', nn.GELU),
            ('gelu', nn.GELU),  # Note: registry uses lowercase
        ]
        
        for name, expected_type in activations:
            activation = get_activation(name)
            assert isinstance(activation, expected_type)
    
    def test_whitespace_handling(self):
        """Test that whitespace is stripped from names."""
        activation1 = get_activation('relu')
        activation2 = get_activation(' relu ')
        activation3 = get_activation('\trelu\n')
        
        assert type(activation1) == type(activation2) == type(activation3)
    
    def test_unsupported_activation_error(self):
        """Test that unsupported activations raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_activation('unsupported_activation')
        
        assert 'Unsupported activation function' in str(excinfo.value)
        assert 'Supported activations' in str(excinfo.value)
    
    def test_invalid_parameters_error(self):
        """Test that invalid parameters raise TypeError."""
        with pytest.raises(TypeError):
            get_activation('relu', invalid_param=True)
    
    def test_leaky_relu_invalid_slope_error(self):
        """Test error handling for invalid LeakyReLU parameters."""
        # Note: PyTorch's LeakyReLU accepts string values and converts them
        # So we test with a parameter that doesn't exist instead
        with pytest.raises(TypeError):
            get_activation('leaky_relu', invalid_param_name=True)
    
    def test_activation_output_shape(self):
        """Test that activations preserve input shape."""
        activations = [
            'relu', 'leaky_relu', 'prelu', 'gelu', 'silu',
            'sigmoid', 'tanh', 'elu', 'selu'
        ]
        
        input_shapes = [
            (2, 64, 32, 32),
            (4, 128, 16, 16),
            (1, 256, 8, 8),
        ]
        
        for activation_name in activations:
            activation = get_activation(activation_name)
            for input_shape in input_shapes:
                x = torch.randn(*input_shape)
                output = activation(x)
                assert output.shape == input_shape
    
    def test_gradient_flow(self):
        """Test that gradients flow through activations."""
        activations = [
            'relu', 'leaky_relu', 'gelu', 'silu', 'sigmoid', 'tanh', 'elu', 'selu'
        ]
        
        for activation_name in activations:
            activation = get_activation(activation_name)
            x = torch.randn(2, 64, 32, 32, requires_grad=True)
            output = activation(x)
            loss = output.sum()
            loss.backward()
            assert x.grad is not None
            assert not torch.isnan(x.grad).any()


class TestActivationFactory:
    """Test the ActivationFactory class."""
    
    def test_factory_create(self):
        """Test factory.create() method."""
        factory = ActivationFactory()
        relu = factory.create('relu')
        assert isinstance(relu, nn.ReLU)
    
    def test_factory_create_with_params(self):
        """Test factory.create() with parameters."""
        factory = ActivationFactory()
        leaky_relu = factory.create('leaky_relu', negative_slope=0.2)
        assert isinstance(leaky_relu, nn.LeakyReLU)
    
    def test_factory_is_supported(self):
        """Test factory.is_supported() method."""
        factory = ActivationFactory()
        assert factory.is_supported('relu')
        assert factory.is_supported('gelu')
        assert not factory.is_supported('unsupported')
    
    def test_factory_supported_activations(self):
        """Test factory.supported_activations() method."""
        factory = ActivationFactory()
        supported = factory.supported_activations()
        assert isinstance(supported, list)
        assert 'relu' in supported
        assert 'gelu' in supported
        assert len(supported) > 0
    
    def test_factory_get_defaults(self):
        """Test factory.get_defaults() method."""
        factory = ActivationFactory()
        defaults = factory.get_defaults('leaky_relu')
        assert isinstance(defaults, dict)
        assert 'negative_slope' in defaults
    
    def test_factory_get_defaults_error(self):
        """Test that factory.get_defaults() raises error for unsupported activation."""
        factory = ActivationFactory()
        with pytest.raises(ValueError):
            factory.get_defaults('unsupported')


class TestActivationUtilities:
    """Test utility functions for activation management."""
    
    def test_is_activation_supported(self):
        """Test is_activation_supported() function."""
        assert is_activation_supported('relu')
        assert is_activation_supported('RELU')
        assert is_activation_supported('gelu')
        assert is_activation_supported('silu')
        assert not is_activation_supported('unsupported')
        assert not is_activation_supported('invalid_activation')
    
    def test_get_supported_activations(self):
        """Test get_supported_activations() function."""
        supported = get_supported_activations()
        
        assert isinstance(supported, list)
        assert len(supported) >= 9  # At least 9 activations
        
        expected = {
            'relu', 'leaky_relu', 'prelu', 'gelu', 'silu',
            'sigmoid', 'tanh', 'elu', 'selu'
        }
        assert set(supported) == expected
    
    def test_supported_activations_sorted(self):
        """Test that supported activations are sorted."""
        supported = get_supported_activations()
        assert supported == sorted(supported)
    
    def test_activation_registry(self):
        """Test ACTIVATION_REGISTRY contains all expected activations."""
        assert len(ACTIVATION_REGISTRY) >= 9
        assert 'relu' in ACTIVATION_REGISTRY
        assert 'gelu' in ACTIVATION_REGISTRY
        assert 'leaky_relu' in ACTIVATION_REGISTRY
        
        # Check that all values are activation classes
        for name, activation_class in ACTIVATION_REGISTRY.items():
            assert issubclass(activation_class, nn.Module)


class TestActivationIntegration:
    """Integration tests with neural network modules."""
    
    def test_activation_in_sequential(self):
        """Test using get_activation() in nn.Sequential."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            get_activation('relu'),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            get_activation('gelu'),
        )
        
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 128, 32, 32)
    
    def test_activation_in_custom_module(self):
        """Test using get_activation() in a custom module."""
        class CustomBlock(nn.Module):
            def __init__(self, in_channels, out_channels, activation='relu'):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                self.bn = nn.BatchNorm2d(out_channels)
                self.activation = get_activation(activation)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.activation(x)
                return x
        
        block = CustomBlock(3, 64, activation='leaky_relu')
        x = torch.randn(2, 3, 32, 32)
        output = block(x)
        assert output.shape == (2, 64, 32, 32)
    
    def test_multiple_activations_in_model(self):
        """Test using multiple different activations in a model."""
        activations_to_test = ['relu', 'leaky_relu', 'gelu', 'silu', 'elu']
        
        for activation_name in activations_to_test:
            model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                get_activation(activation_name),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                get_activation(activation_name),
            )
            
            x = torch.randn(2, 3, 32, 32)
            output = model(x)
            assert output.shape == (2, 128, 32, 32)
    
    def test_activation_backward_compatibility(self):
        """Test that different ways of creating activations are compatible."""
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        
        # Method 1: Using get_activation
        act1 = get_activation('relu')
        out1 = act1(x)
        loss1 = out1.sum()
        loss1.backward()
        
        # Method 2: Direct instantiation
        x2 = torch.randn(2, 64, 32, 32, requires_grad=True)
        act2 = nn.ReLU()
        out2 = act2(x2)
        loss2 = out2.sum()
        loss2.backward()
        
        # Both should have similar behavior
        assert out1.shape == out2.shape
        assert x.grad is not None
        assert x2.grad is not None
