=======================
Examples: Basic Usage
=======================

Complete working examples for common use cases.

Example 1: Simple Image Classifier
===================================

Create a basic CNN for image classification:

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from torch.optim import Adam
    from torchvision_customizer import CustomCNN
    
    # Create model
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=10,
        num_conv_blocks=4,
        channels=[64, 128, 256, 512],
        dropout=0.2
    )
    
    # Create dummy dataset
    x_train = torch.randn(100, 3, 224, 224)
    y_train = torch.randint(0, 10, (100,))
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        
        x, y = x_train.to(device), y_train.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1:2d}: Loss = {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        x_test = torch.randn(10, 3, 224, 224).to(device)
        logits = model(x_test)
        predictions = logits.argmax(dim=1)
        print(f"Predictions shape: {predictions.shape}")

Example 2: Transfer Learning
=============================

Fine-tune for a new task:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch.optim import SGD
    from torch.optim.lr_scheduler import StepLR
    import torch.nn.functional as F
    from torchvision_customizer import CustomCNN
    
    # Step 1: Create base model
    base_model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=4
    )
    # (In practice, load pre-trained weights)
    
    # Step 2: Create new model for task
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=5,  # New task
        num_conv_blocks=4
    )
    
    # Step 3: Freeze early layers
    for name, param in model.named_parameters():
        if 'block1' in name or 'block2' in name:
            param.requires_grad = False
    
    # Step 4: Setup training
    optimizer = SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        momentum=0.9
    )
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Step 5: Training with lower learning rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    x_train = torch.randn(50, 3, 224, 224)
    y_train = torch.randint(0, 5, (50,))
    
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        
        x, y = x_train.to(device), y_train.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}: Loss = {loss.item():.4f}")

Example 3: Custom Architecture with Builder
============================================

Build a custom architecture step by step:

.. code-block:: python

    import torch
    from torchvision_customizer import CNNBuilder
    
    # Create custom architecture
    model = (CNNBuilder(input_shape=(3, 224, 224))
        # Initial convolution - extract low-level features
        .add_conv_block(64, kernel_size=7, stride=2, activation='relu')
        .add_pooling('max', kernel_size=3, stride=2)
        
        # Residual blocks - build deeper features
        .add_residual_block(128, num_convs=2, activation='relu')
        .add_residual_block(128, num_convs=2, activation='relu')
        .add_residual_block(256, num_convs=2, downsample=True, activation='relu')
        
        # More residual blocks
        .add_residual_block(256, num_convs=2, activation='relu')
        .add_residual_block(512, num_convs=2, downsample=True, activation='relu')
        
        # Global pooling and classifier
        .add_global_pooling('adaptive')
        .add_classifier([512, 256], num_classes=1000, dropout=0.3)
        
        .build()
    )
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")

Example 4: Comparing Different Activations
============================================

See performance differences:

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from torch.optim import Adam
    from torchvision_customizer import CustomCNN
    
    activations = ['relu', 'gelu', 'mish', 'swish']
    results = {}
    
    for activation in activations:
        model = CustomCNN(
            input_shape=(3, 224, 224),
            num_classes=10,
            num_conv_blocks=3,
            activation=activation
        )
        
        x = torch.randn(32, 3, 224, 224)
        y = torch.randint(0, 10, (32,))
        
        optimizer = Adam(model.parameters(), lr=0.001)
        losses = []
        
        model.train()
        for _ in range(100):
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        results[activation] = {
            'final_loss': losses[-1],
            'avg_loss': sum(losses) / len(losses)
        }
    
    print("=== Activation Comparison ===")
    for act, stats in results.items():
        print(f"{act:8s}: Final Loss = {stats['final_loss']:.4f}, "
              f"Avg Loss = {stats['avg_loss']:.4f}")

Example 5: Model Analysis
==========================

Analyze model properties:

.. code-block:: python

    import torch
    from torchvision_customizer import CustomCNN
    from torchvision_customizer.utils import (
        get_model_summary,
        calculate_flops,
        calculate_memory,
        profile_model
    )
    
    # Create model
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=4
    )
    
    # Get summary
    print("=== Model Summary ===")
    summary = get_model_summary(model, input_size=(1, 3, 224, 224))
    print(f"Total Parameters: {summary['total_params']:,}")
    print(f"Trainable Parameters: {summary['trainable_params']:,}")
    print(f"Model Size: {summary['total_memory_mb']:.2f} MB")
    
    # Calculate FLOPs
    print("\n=== Computational Complexity ===")
    flops = calculate_flops(model, input_size=(1, 3, 224, 224))
    print(f"FLOPs: {flops:,}")
    print(f"GFLOPs (for batch size 1): {flops / 1e9:.2f}")
    
    # Profile execution
    print("\n=== Performance Profile ===")
    stats = profile_model(model, input_size=(1, 3, 224, 224), num_iterations=100)
    print(f"Average Forward Time: {stats['forward_time_ms']:.2f} ms")
    print(f"Memory Peak: {stats['memory_peak_mb']:.2f} MB")

Example 6: Data Augmentation Integration
========================================

Use with torchvision transforms:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import transforms
    from torchvision_customizer import CustomCNN
    import torch.nn.functional as F
    from torch.optim import Adam
    
    # Create augmentation pipeline
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])
    
    # Create dummy dataset
    x = torch.randn(100, 3, 224, 224)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=10
    )
    
    optimizer = Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Training with augmentation
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            # Apply augmentation
            batch_x = augmentation(batch_x)
            
            # Forward pass
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(dataloader):.4f}")

Example 7: Configuration-Based Models
======================================

Define and load models from config:

.. code-block:: yaml

    # configs/mobilenet_config.yaml
    input_shape: [3, 224, 224]
    num_classes: 1000
    num_conv_blocks: 6
    channels: [32, 32, 64, 64, 128, 128]
    activation: relu
    normalization: batch
    pooling_type: adaptive
    depthwise: true
    dropout: 0.2

.. code-block:: python

    import yaml
    from torchvision_customizer import CustomCNN
    
    # Load configuration
    with open('configs/mobilenet_config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Create model from config
    model = CustomCNN(**config)
    
    print(model)
    print(f"Model created with config: {config}")

Example 8: Ensemble Models
===========================

Combine multiple models:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torchvision_customizer import CustomCNN
    
    class EnsembleModel(nn.Module):
        def __init__(self, num_models=3):
            super().__init__()
            self.models = nn.ModuleList([
                CustomCNN(
                    input_shape=(3, 224, 224),
                    num_classes=10,
                    num_conv_blocks=3
                )
                for _ in range(num_models)
            ])
        
        def forward(self, x):
            outputs = []
            for model in self.models:
                output = model(x)
                outputs.append(output)
            
            # Average predictions
            ensemble_output = torch.stack(outputs, dim=0).mean(dim=0)
            return ensemble_output
    
    # Create ensemble
    ensemble = EnsembleModel(num_models=3)
    
    # Test
    x = torch.randn(1, 3, 224, 224)
    output = ensemble(x)
    print(f"Ensemble output shape: {output.shape}")

Next Steps
==========

- :doc:`../examples/advanced_patterns` - Advanced techniques
- :doc:`../examples/transfer_learning` - Transfer learning
- :doc:`../api/models` - Model API reference
