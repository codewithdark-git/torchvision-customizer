"""Tests for Hybrid Trainer.

Tests the training utilities for MNIST, CIFAR-10, and custom datasets.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Skip if torchvision not available
pytest.importorskip("torchvision")


class TestTrainingMetrics:
    """Test TrainingMetrics class."""
    
    def test_metrics_update(self):
        """Test updating metrics."""
        from torchvision_customizer.hybrid.trainer import TrainingMetrics
        
        metrics = TrainingMetrics()
        metrics.update(0, 1.0, 0.5, 0.8, 0.6, 10.0)
        metrics.update(1, 0.5, 0.7, 0.4, 0.8, 10.0)
        
        assert len(metrics.train_losses) == 2
        assert metrics.best_val_acc == 0.8
        assert metrics.best_epoch == 1
    
    def test_metrics_summary(self):
        """Test metrics summary generation."""
        from torchvision_customizer.hybrid.trainer import TrainingMetrics
        
        metrics = TrainingMetrics()
        metrics.update(0, 1.0, 0.5, 0.8, 0.6, 10.0)
        
        summary = metrics.summary()
        assert "Training Summary" in summary
        assert "Best validation accuracy" in summary


class TestTrainer:
    """Test Trainer class."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        from torchvision_customizer.hybrid.trainer import Trainer
        
        model = nn.Linear(10, 5)
        trainer = Trainer(model, device='cpu')
        
        assert trainer.device == torch.device('cpu')
    
    def test_trainer_fit_custom_data(self):
        """Test training on custom data loaders."""
        from torchvision_customizer.hybrid.trainer import Trainer
        
        # Create simple model and data
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )
        
        # Create dummy data
        X = torch.randn(100, 3, 8, 8)
        y = torch.randint(0, 5, (100,))
        dataset = TensorDataset(X, y)
        
        train_loader = DataLoader(dataset, batch_size=20, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=20, shuffle=False)
        
        trainer = Trainer(model, device='cpu')
        metrics = trainer.fit(
            train_loader,
            val_loader,
            epochs=2,
            lr=0.01,
            verbose=False,
        )
        
        assert len(metrics.train_losses) == 2
        assert metrics.train_accs[-1] > 0  # Some accuracy
    
    def test_trainer_evaluate(self):
        """Test evaluation method."""
        from torchvision_customizer.hybrid.trainer import Trainer
        
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 5),
        )
        
        X = torch.randn(50, 3, 8, 8)
        y = torch.randint(0, 5, (50,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)
        
        trainer = Trainer(model, device='cpu')
        loss, acc = trainer.evaluate(loader)
        
        assert loss >= 0
        assert 0 <= acc <= 1
    
    def test_create_optimizers(self):
        """Test optimizer creation."""
        from torchvision_customizer.hybrid.trainer import Trainer
        
        model = nn.Linear(10, 5)
        trainer = Trainer(model, device='cpu')
        
        adam = trainer._create_optimizer('adam', lr=0.001)
        assert isinstance(adam, torch.optim.Adam)
        
        sgd = trainer._create_optimizer('sgd', lr=0.01)
        assert isinstance(sgd, torch.optim.SGD)
        
        adamw = trainer._create_optimizer('adamw', lr=0.001)
        assert isinstance(adamw, torch.optim.AdamW)
    
    def test_create_scheduler(self):
        """Test scheduler creation."""
        from torchvision_customizer.hybrid.trainer import Trainer
        
        model = nn.Linear(10, 5)
        trainer = Trainer(model, device='cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        cosine = trainer._create_scheduler('cosine', optimizer, epochs=10, steps_per_epoch=100)
        assert cosine is not None
        
        step = trainer._create_scheduler('step', optimizer, epochs=10, steps_per_epoch=100)
        assert step is not None
        
        none_sched = trainer._create_scheduler(None, optimizer, epochs=10, steps_per_epoch=100)
        assert none_sched is None


class TestQuickTrain:
    """Test quick_train function."""
    
    def test_quick_train_invalid_dataset(self):
        """Test quick_train with invalid dataset."""
        from torchvision_customizer.hybrid.trainer import quick_train
        
        model = nn.Linear(10, 5)
        
        with pytest.raises(ValueError, match="Unknown dataset"):
            quick_train(model, dataset='invalid')


class TestTrainerWithHybridModel:
    """Test Trainer with actual Hybrid models."""
    
    def test_trainer_with_hybrid_builder(self):
        """Test training a HybridBuilder model."""
        from torchvision_customizer.hybrid import HybridBuilder, Trainer
        
        # Build a simple hybrid model
        model = HybridBuilder().from_torchvision(
            "resnet18",
            weights=None,
            num_classes=5,
            verbose=False,
        )
        
        # Create dummy data
        X = torch.randn(20, 3, 32, 32)
        y = torch.randint(0, 5, (20,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)
        
        trainer = Trainer(model, device='cpu')
        metrics = trainer.fit(
            loader, loader,
            epochs=1,
            lr=0.001,
            verbose=False,
        )
        
        assert len(metrics.train_losses) == 1
    
    def test_trainer_with_frozen_backbone(self):
        """Test training with frozen backbone."""
        from torchvision_customizer.hybrid import HybridBuilder, Trainer
        
        model = HybridBuilder().from_torchvision(
            "resnet18",
            weights=None,
            num_classes=5,
            freeze_backbone=True,
            verbose=False,
        )
        
        # Count trainable params (should be just the head)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        assert trainable < total  # Some params should be frozen
        
        # Create dummy data
        X = torch.randn(10, 3, 32, 32)
        y = torch.randint(0, 5, (10,))
        loader = DataLoader(TensorDataset(X, y), batch_size=5)
        
        trainer = Trainer(model, device='cpu')
        metrics = trainer.fit(loader, epochs=1, verbose=False)
        
        assert len(metrics.train_losses) == 1

