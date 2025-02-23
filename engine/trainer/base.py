"""Base classes for trainer implementations."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable
import torch
from torch.utils.data import DataLoader

from engine.trainer.managers.progress_manager import ProgressManager
from engine.config.training_config import TrainingConfig


class BaseTrainer(ABC):
    """Abstract base class for model trainers."""

    @abstractmethod
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model on validation set."""
        pass

    @abstractmethod
    def _train_step(self, batch: Tuple[torch.Tensor, torch.Tensor], return_esr: bool = False) -> Tuple[float, float]:
        """Execute a single training step."""
        pass

    @abstractmethod
    def check_early_stopping(self, current_loss: float) -> bool:
        """Check if training should be stopped."""
        pass

    @abstractmethod
    def set_progress_manager(self, progress_manager: Optional['ProgressManager']):
        """Set the progress manager for updating progress bars."""
        pass

    @property
    @abstractmethod
    def model(self) -> torch.nn.Module:
        """Get the model being trained."""
        pass

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer being used."""
        pass

    @property
    @abstractmethod
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Get the learning rate scheduler being used."""
        pass

    @property
    @abstractmethod
    def config(self) -> TrainingConfig:
        """Get the training configuration."""
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get the device being used for training."""
        pass

    @property
    @abstractmethod
    def current_epoch(self) -> int:
        """Get the current epoch number."""
        pass

    @current_epoch.setter
    @abstractmethod
    def current_epoch(self, value: int):
        """Set the current epoch number."""
        pass
