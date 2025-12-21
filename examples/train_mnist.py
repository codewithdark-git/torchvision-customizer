#!/usr/bin/env python
"""Train a Hybrid Model on MNIST.

This example demonstrates how to use the Trainer API to train a
customized ResNet model on the MNIST dataset.

Usage:
    python examples/train_mnist.py
"""

import torch
from torchvision_customizer.hybrid import HybridBuilder, Trainer


def main():
    print("=" * 60)
    print("Training Hybrid ResNet-18 on MNIST")
    print("=" * 60)
    
    # Build a customized ResNet-18 model
    # - Add SE attention to layer3
    # - Use 10 classes for MNIST digits
    builder = HybridBuilder()
    model = builder.from_torchvision(
        "resnet18",
        weights=None,  # Train from scratch for MNIST
        patches={
            "layer2": {"wrap": "se"},
            "layer3": {"wrap": "se"},
        },
        num_classes=10,
        verbose=True,
    )
    
    # Print model info
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer and train
    trainer = Trainer(model, device='auto')
    
    print("\nStarting training...")
    metrics = trainer.fit_mnist(
        epochs=3,
        batch_size=64,
        lr=0.001,
        optimizer='adam',
    )
    
    print("\n" + metrics.summary())
    
    # Final evaluation
    print(f"\nFinal test accuracy: {metrics.val_accs[-1]:.2%}")


if __name__ == "__main__":
    main()

