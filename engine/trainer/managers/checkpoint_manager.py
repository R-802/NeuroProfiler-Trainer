"""Checkpoint management module for model saving and loading."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from engine.trainer.managers.progress_manager import TrainingMetrics


class CheckpointManager:
    """Manages model checkpoints and early stopping."""

    def __init__(self, save_dir: str):
        """Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.best_esr = float('inf')

    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer,
             metrics: TrainingMetrics, epoch: int, config: Dict[str, Any]) -> bool:
        """Save model checkpoint and return True if this is the best model so far.

        Args:
            model: The model to save
            optimizer: The optimizer to save
            metrics: Current training metrics
            epoch: Current epoch number
            config: Training configuration

        Returns:
            bool: True if this checkpoint has the best ESR so far
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metrics.val_loss,
            'esr': metrics.val_esr,
            'epoch': epoch,
            'config': config
        }

        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest.pt')

        # Save best checkpoint if this is the best ESR
        if metrics.val_esr < self.best_esr:
            torch.save(checkpoint, self.save_dir / 'best.pt')
            self.best_esr = metrics.val_esr
            return True

        return False

    def load(self, path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Load a checkpoint from the given path.

        Args:
            path: Path to the checkpoint file

        Returns:
            Tuple containing model state dict, optimizer state dict, and metadata
        """
        checkpoint = torch.load(path)
        model_state = checkpoint['model_state_dict']
        optimizer_state = checkpoint['optimizer_state_dict']
        metadata = {
            'loss': checkpoint['loss'],
            'esr': checkpoint['esr'],
            'epoch': checkpoint['epoch'],
            'config': checkpoint['config']
        }
        return model_state, optimizer_state, metadata

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get the path to the best checkpoint if it exists.

        Returns:
            Path to best checkpoint or None if it doesn't exist
        """
        best_path = self.save_dir / 'best.pt'
        return best_path if best_path.exists() else None

    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get the path to the latest checkpoint if it exists.

        Returns:
            Path to latest checkpoint or None if it doesn't exist
        """
        latest_path = self.save_dir / 'latest.pt'
        return latest_path if latest_path.exists() else None
