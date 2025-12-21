#!/usr/bin/env python
"""Quick training examples using the simple API.

This example shows the simplest way to train models using quick_train().

Usage:
    python examples/quick_train_example.py
"""

from torchvision_customizer.hybrid import HybridBuilder, quick_train


def example_mnist():
    """Quick train on MNIST."""
    print("\n" + "=" * 60)
    print("Quick Training on MNIST")
    print("=" * 60)
    
    model = HybridBuilder().from_torchvision(
        "resnet18",
        weights=None,
        num_classes=10,
    )
    
    # One-liner training!
    metrics = quick_train(model, dataset='mnist', epochs=2)
    print(f"Test accuracy: {metrics.val_accs[-1]:.2%}")


def example_cifar10():
    """Quick train on CIFAR-10."""
    print("\n" + "=" * 60)
    print("Quick Training on CIFAR-10")
    print("=" * 60)
    
    model = HybridBuilder().from_torchvision(
        "resnet18",
        weights=None,
        patches={"layer3": {"wrap": "se"}},
        num_classes=10,
    )
    
    metrics = quick_train(model, dataset='cifar10', epochs=3, lr=0.001)
    print(f"Test accuracy: {metrics.val_accs[-1]:.2%}")


def example_efficientnet():
    """Quick train EfficientNet on CIFAR-10."""
    print("\n" + "=" * 60)
    print("Quick Training EfficientNet-B0 on CIFAR-10")
    print("=" * 60)
    
    model = HybridBuilder().from_torchvision(
        "efficientnet_b0",
        weights="IMAGENET1K_V1",
        num_classes=10,
        freeze_backbone=True,
        unfreeze_stages=[6, 7],  # Unfreeze last 2 stages
    )
    
    metrics = quick_train(model, dataset='cifar10', epochs=3, lr=0.0001)
    print(f"Test accuracy: {metrics.val_accs[-1]:.2%}")


if __name__ == "__main__":
    example_mnist()
    example_cifar10()

