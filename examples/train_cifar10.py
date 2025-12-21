#!/usr/bin/env python
"""Train a Hybrid Model on CIFAR-10.

This example demonstrates how to use the Trainer API to fine-tune a
pre-trained ResNet model on CIFAR-10.

Usage:
    python examples/train_cifar10.py
"""

import torch
from torchvision_customizer.hybrid import HybridBuilder, Trainer


def main():
    print("=" * 60)
    print("Fine-tuning Hybrid ResNet-18 on CIFAR-10")
    print("=" * 60)
    
    # Build a customized ResNet-18 with attention mechanisms
    builder = HybridBuilder()
    model = builder.from_torchvision(
        "resnet18",
        weights="IMAGENET1K_V1",  # Use pre-trained weights
        patches={
            "layer2": {"wrap": "eca"},          # Efficient Channel Attention
            "layer3": {"wrap": "cbam_block"},   # CBAM attention
            "layer4": {"wrap": "se"},           # SE attention
        },
        num_classes=10,
        dropout=0.2,
        freeze_backbone=False,  # Fine-tune entire model
        verbose=True,
    )
    
    # Print model info
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer
    trainer = Trainer(model, device='auto')
    
    print("\nStarting training...")
    metrics = trainer.fit_cifar10(
        epochs=5,
        batch_size=128,
        lr=0.001,
        optimizer='adamw',
        scheduler='cosine',
        augment=True,
    )
    
    print("\n" + metrics.summary())


def train_from_scratch():
    """Alternative: Train a model from scratch."""
    print("=" * 60)
    print("Training ResNet-18 from scratch on CIFAR-10")
    print("=" * 60)
    
    builder = HybridBuilder()
    model = builder.from_torchvision(
        "resnet18",
        weights=None,  # No pre-training
        patches={
            "layer3": {"wrap": "se"},
            "layer4": {"wrap": "se"},
        },
        num_classes=10,
    )
    
    trainer = Trainer(model)
    metrics = trainer.fit_cifar10(
        epochs=10,
        batch_size=128,
        lr=0.01,
        optimizer='sgd',
        scheduler='cosine',
    )
    
    return metrics


if __name__ == "__main__":
    main()

