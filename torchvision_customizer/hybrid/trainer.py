"""Training utilities for Hybrid Models.

Provides a simple training API for quick experimentation with
MNIST, CIFAR-10, CIFAR-100, and custom datasets.

Example:
    >>> from torchvision_customizer.hybrid import HybridBuilder, Trainer
    >>> 
    >>> model = HybridBuilder().from_torchvision("resnet18", num_classes=10)
    >>> trainer = Trainer(model)
    >>> trainer.fit_cifar10(epochs=5)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class TrainingMetrics:
    """Container for training metrics."""
    
    def __init__(self):
        self.train_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_losses: List[float] = []
        self.val_accs: List[float] = []
        self.epoch_times: List[float] = []
        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        epoch_time: float,
    ):
        """Update metrics for an epoch."""
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.epoch_times.append(epoch_time)
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
    
    def summary(self) -> str:
        """Return training summary."""
        lines = [
            "=" * 50,
            "Training Summary",
            "=" * 50,
            f"Total epochs: {len(self.train_losses)}",
            f"Best validation accuracy: {self.best_val_acc:.2%} (epoch {self.best_epoch + 1})",
            f"Final train accuracy: {self.train_accs[-1]:.2%}",
            f"Final validation accuracy: {self.val_accs[-1]:.2%}",
            f"Total training time: {sum(self.epoch_times):.1f}s",
            f"Average epoch time: {sum(self.epoch_times) / len(self.epoch_times):.1f}s",
            "=" * 50,
        ]
        return "\n".join(lines)


class Trainer:
    """Simple trainer for Hybrid Models.
    
    Provides convenient methods for training on common datasets
    like MNIST, CIFAR-10, and CIFAR-100.
    
    Args:
        model: The model to train
        device: Device to use ('cuda', 'cpu', or 'auto')
        
    Example:
        >>> model = HybridBuilder().from_torchvision("resnet18", num_classes=10)
        >>> trainer = Trainer(model)
        >>> metrics = trainer.fit_cifar10(epochs=10, lr=0.001)
        >>> print(metrics.summary())
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
    ):
        self.model = model
        
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        lr: float = 0.001,
        optimizer: Optional[str] = 'adam',
        scheduler: Optional[str] = 'cosine',
        criterion: Optional[nn.Module] = None,
        verbose: bool = True,
    ) -> TrainingMetrics:
        """Train the model on custom data loaders.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            lr: Learning rate
            optimizer: Optimizer type ('adam', 'sgd', 'adamw')
            scheduler: Learning rate scheduler ('cosine', 'step', None)
            criterion: Loss function (default: CrossEntropyLoss)
            verbose: Print progress
            
        Returns:
            TrainingMetrics with training history
        """
        # Setup
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        opt = self._create_optimizer(optimizer, lr)
        sched = self._create_scheduler(scheduler, opt, epochs, len(train_loader))
        
        metrics = TrainingMetrics()
        
        if verbose:
            print(f"Training on {self.device}")
            print(f"Epochs: {epochs}, LR: {lr}, Optimizer: {optimizer}")
            print("-" * 60)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, opt, sched
            )
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader, criterion)
            else:
                val_loss, val_acc = 0.0, 0.0
            
            epoch_time = time.time() - epoch_start
            
            metrics.update(epoch, train_loss, train_acc, val_loss, val_acc, epoch_time)
            
            if verbose:
                val_str = f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}" if val_loader else ""
                print(f"Epoch {epoch + 1:3d}/{epochs} | "
                      f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%} | "
                      f"{val_str} | {epoch_time:.1f}s")
        
        if verbose:
            print(metrics.summary())
        
        return metrics
    
    def fit_mnist(
        self,
        epochs: int = 5,
        batch_size: int = 64,
        lr: float = 0.001,
        data_dir: str = './data',
        **kwargs,
    ) -> TrainingMetrics:
        """Train on MNIST dataset.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            data_dir: Directory to download/load data
            **kwargs: Additional arguments passed to fit()
            
        Returns:
            TrainingMetrics with training history
        """
        train_loader, val_loader = self._get_mnist_loaders(batch_size, data_dir)
        return self.fit(train_loader, val_loader, epochs=epochs, lr=lr, **kwargs)
    
    def fit_cifar10(
        self,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 0.001,
        data_dir: str = './data',
        augment: bool = True,
        **kwargs,
    ) -> TrainingMetrics:
        """Train on CIFAR-10 dataset.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            data_dir: Directory to download/load data
            augment: Use data augmentation
            **kwargs: Additional arguments passed to fit()
            
        Returns:
            TrainingMetrics with training history
        """
        train_loader, val_loader = self._get_cifar10_loaders(
            batch_size, data_dir, augment
        )
        return self.fit(train_loader, val_loader, epochs=epochs, lr=lr, **kwargs)
    
    def fit_cifar100(
        self,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 0.001,
        data_dir: str = './data',
        augment: bool = True,
        **kwargs,
    ) -> TrainingMetrics:
        """Train on CIFAR-100 dataset.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            data_dir: Directory to download/load data
            augment: Use data augmentation
            **kwargs: Additional arguments passed to fit()
            
        Returns:
            TrainingMetrics with training history
        """
        train_loader, val_loader = self._get_cifar100_loaders(
            batch_size, data_dir, augment
        )
        return self.fit(train_loader, val_loader, epochs=epochs, lr=lr, **kwargs)
    
    def evaluate(
        self,
        data_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> Tuple[float, float]:
        """Evaluate the model.
        
        Args:
            data_loader: Data loader for evaluation
            criterion: Loss function
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        return self._validate(data_loader, criterion)
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
    ) -> Tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _create_optimizer(
        self,
        optimizer_name: str,
        lr: float,
    ) -> optim.Optimizer:
        """Create optimizer."""
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if optimizer_name.lower() == 'adam':
            return optim.Adam(params, lr=lr)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=0.01)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(
        self,
        scheduler_name: Optional[str],
        optimizer: optim.Optimizer,
        epochs: int,
        steps_per_epoch: int,
    ) -> Optional[Any]:
        """Create learning rate scheduler."""
        if scheduler_name is None:
            return None
        
        total_steps = epochs * steps_per_epoch
        
        if scheduler_name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps
            )
        elif scheduler_name.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=steps_per_epoch * (epochs // 3), gamma=0.1
            )
        elif scheduler_name.lower() == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=optimizer.param_groups[0]['lr'] * 10,
                total_steps=total_steps
            )
        else:
            return None
    
    def _get_mnist_loaders(
        self,
        batch_size: int,
        data_dir: str,
    ) -> Tuple[DataLoader, DataLoader]:
        """Get MNIST data loaders."""
        try:
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError("torchvision is required for MNIST dataset")
        
        # MNIST needs to be resized for models expecting larger inputs
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
            transforms.ToTensor(),
            transforms.Normalize((0.1307,) * 3, (0.3081,) * 3),
        ])
        
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        val_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        return train_loader, val_loader
    
    def _get_cifar10_loaders(
        self,
        batch_size: int,
        data_dir: str,
        augment: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """Get CIFAR-10 data loaders."""
        try:
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError("torchvision is required for CIFAR-10 dataset")
        
        # Normalization for CIFAR-10
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        )
        
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=val_transform
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        return train_loader, val_loader
    
    def _get_cifar100_loaders(
        self,
        batch_size: int,
        data_dir: str,
        augment: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """Get CIFAR-100 data loaders."""
        try:
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError("torchvision is required for CIFAR-100 dataset")
        
        # Normalization for CIFAR-100
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )
        
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        train_dataset = datasets.CIFAR100(
            data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = datasets.CIFAR100(
            data_dir, train=False, download=True, transform=val_transform
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        return train_loader, val_loader


def quick_train(
    model: nn.Module,
    dataset: str = 'cifar10',
    epochs: int = 5,
    lr: float = 0.001,
    batch_size: int = 128,
    device: str = 'auto',
    verbose: bool = True,
) -> TrainingMetrics:
    """Quick training function for common datasets.
    
    A convenience function for fast experimentation.
    
    Args:
        model: Model to train
        dataset: Dataset name ('mnist', 'cifar10', 'cifar100')
        epochs: Number of epochs
        lr: Learning rate
        batch_size: Batch size
        device: Device to use
        verbose: Print progress
        
    Returns:
        TrainingMetrics with training history
        
    Example:
        >>> from torchvision_customizer.hybrid import HybridBuilder, quick_train
        >>> model = HybridBuilder().from_torchvision("resnet18", num_classes=10)
        >>> metrics = quick_train(model, dataset='cifar10', epochs=5)
    """
    trainer = Trainer(model, device=device)
    
    dataset = dataset.lower()
    
    if dataset == 'mnist':
        return trainer.fit_mnist(epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'cifar10':
        return trainer.fit_cifar10(epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'cifar100':
        return trainer.fit_cifar100(epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Available: mnist, cifar10, cifar100")

