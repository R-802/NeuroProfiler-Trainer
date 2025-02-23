"""Training management module for coordinating the training process. """

import logging
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from engine.utils import logger
from engine.config.training_config import TrainingConfig
from engine.models.lstm import LSTMProfiler
from engine.trainer.trainer import ModelTrainer
from engine.trainer.factories.optimizer_factory import OptimizerFactory, SchedulerFactory
from engine.trainer.managers.checkpoint_manager import CheckpointManager
from engine.trainer.managers.progress_manager import ProgressManager, TrainingMetrics
from engine.trainer.utils.tensor_validation import (
    validate_batch_tensors,
    validate_and_trim_sequences,
    validate_hidden_state
)


class TrainingManager:
    """Manages the training process, including checkpointing and early stopping."""

    def __init__(self, trainer: ModelTrainer, save_dir: str = "checkpoints",
                 display_config: bool = False, show_progress: bool = True):
        """Initialize training manager.

        Args:
            trainer: The model trainer instance
            save_dir: Directory to save checkpoints
            display_config: Whether to display configuration at start
            show_progress: Whether to show progress bars
        """
        self.trainer = trainer
        self.checkpoint_manager = CheckpointManager(save_dir)
        self.display_config = display_config
        self.show_progress = show_progress
        self.writer = None
        self.early_stop = False

        if self.show_progress:
            self._log_hardware_info()
            self._log_model_summary()

        if display_config and self.show_progress:
            print("\nTraining Configuration:")
            print("-" * 50)
            self._log_formatted_config(self.trainer.config.to_dict())

        print("\nLoss Function Configuration:")
        print("-" * 50)
        loss_config = self.trainer.config.to_dict().get("loss")
        if loss_config is not None:
            self._log_formatted_config(loss_config)

    def _log_hardware_info(self):
        """Log available hardware information."""
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        current_device = torch.cuda.current_device() if cuda_available else None

        print("\nHardware Information:")
        print("-" * 50)
        if self.trainer.config.hardware.mixed_precision:
            logger.info("FP16 (mixed precision) training enabled")
        else:
            logger.info("FP32 training enabled (training might be slower)")

        logger.info(
            f"GPU available: {cuda_available} (cuda), used: {cuda_available}")
        if cuda_available:
            logger.info(
                f"LOCAL_RANK: {current_device} - CUDA_VISIBLE_DEVICES: {list(range(cuda_device_count))}")
            logger.info(
                f"GPU Device: {torch.cuda.get_device_name(current_device)}")

    def _log_model_summary(self):
        """Log model architecture summary."""
        model = self.trainer.model

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        # Count modules in train/eval mode
        train_modules = sum(1 for m in model.modules() if m.training)
        eval_modules = sum(1 for m in model.modules() if not m.training)

        # Calculate model size in KB
        param_size = sum(p.numel() * p.element_size()
                         for p in model.parameters()) / 1024

        # Print summary
        print("\nModel Summary:")
        print("-" * 50)
        logger.info(f"{trainable_params/1000:.1f} K     Trainable params")
        logger.info(
            f"{non_trainable_params/1000:.1f} K     Non-trainable params")
        logger.info(f"{total_params/1000:.1f} K     Total params")
        logger.info(f"{param_size:.1f} KB   Total estimated model params size")
        logger.info(f"{train_modules}         Modules in train mode")
        logger.info(f"{eval_modules}         Modules in eval mode")

    def _log_formatted_config(self, config: Dict[str, Any], indent: int = 0):
        """Format and log configuration with proper indentation."""
        for key, value in config.items():
            if isinstance(value, dict):
                logger.info(" " * indent + f"{key}:")
                self._log_formatted_config(value, indent + 2)
            elif isinstance(value, list):
                logger.info(" " * indent + f"{key}:")
                for item in value:
                    if isinstance(item, (dict, list)):
                        logger.info(" " * (indent + 2) + "- ")
                        self._log_formatted_value(item, indent + 4)
                    else:
                        formatted = self._format_value(item)
                        logger.info(" " * (indent + 2) + f"- {formatted}")
            else:
                formatted = self._format_value(value)
                logger.info(" " * indent + f"{key}: {formatted}")

    def _log_formatted_value(self, value: Any, indent: int):
        """Helper to handle nested structures within lists."""
        if isinstance(value, dict):
            for k, v in value.items():
                formatted = self._format_value(v)
                if isinstance(v, (dict, list)):
                    logger.info(" " * indent + f"{k}:")
                    self._log_formatted_value(v, indent + 2)
                else:
                    logger.info(" " * indent + f"{k}: {formatted}")
        elif isinstance(value, list):
            for item in value:
                formatted = self._format_value(item)
                if isinstance(item, (dict, list)):
                    logger.info(" " * indent + "- ")
                    self._log_formatted_value(item, indent + 2)
                else:
                    logger.info(" " * indent + f"- {formatted}")

    def _format_value(self, value: Any) -> str:
        """Consistent value formatting for different types."""
        if value is None:
            return "None"
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return f"{value}" if isinstance(value, int) else f"{value:.6g}"
        else:
            return str(value)

    def _handle_training_interrupt(self, metrics: TrainingMetrics, val_loader: Optional[DataLoader] = None):
        """Handle training interruption gracefully."""
        # Attempt to compute validation metrics if a validation loader is provided
        if val_loader is not None:
            try:
                metrics.val_loss, metrics.val_esr = self.trainer.validate(
                    val_loader)
            except Exception as e:
                logger.warning(f"Validation during interrupt failed: {e}")

        self.checkpoint_manager.save(
            self.trainer.model,
            self.trainer.optimizer,
            metrics,
            self.trainer.current_epoch,
            self.trainer.config.to_dict()
        )

    def _log_metrics(self, metrics: TrainingMetrics, epoch: int):
        """Log metrics to tensorboard."""
        if self.writer is None:
            return

        # Log losses
        self.writer.add_scalars('Loss', {
            'Train': metrics.train_loss,
            'Validation': metrics.val_loss
        }, epoch)

        # Log ESR in a combined plot
        self.writer.add_scalars('ESR', {
            'Train': metrics.train_esr,
            'Validation': metrics.val_esr
        }, epoch)

        # Log learning rate separately
        self.writer.add_scalar('Learning Rate', metrics.learning_rate, epoch)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              open_tensorboard: bool = True, tensorboard_logging: bool = True):
        """Run the complete training loop with centralized error handling."""
        metrics = TrainingMetrics()

        try:
            if tensorboard_logging:
                self._setup_tensorboard(
                    train_loader, open_tensorboard=open_tensorboard)
            else:
                print("\nTensorBoard logging is disabled.")

            print("\nTraining:")
            print("=" * 50)

            start_time = time.time()
            epoch_times: list[float] = []

            self.progress_manager = ProgressManager(
                self.show_progress,
                self.trainer.config.epochs,
                len(train_loader),
                len(val_loader)
            )

            # Set the progress manager in the trainer
            self.trainer.set_progress_manager(self.progress_manager)

            for epoch in self.progress_manager.epoch_bar:
                self.trainer.current_epoch = epoch + 1
                self.progress_manager.update_epoch(
                    epoch + 1, self.trainer.config.epochs)

                # Training phase
                metrics = self.trainer.train_epoch(train_loader)

                # Validation phase
                val_metrics = self.trainer.validate_epoch(val_loader)
                metrics.val_loss = val_metrics.val_loss
                metrics.val_esr = val_metrics.val_esr
                metrics.learning_rate = val_metrics.learning_rate

                # Calculate ETA
                epoch_time = time.time() - start_time
                epoch_times.append(epoch_time)
                avg_epoch_time = np.mean(epoch_times)
                remaining_epochs = self.trainer.config.epochs - (epoch + 1)
                eta_seconds = avg_epoch_time * remaining_epochs
                eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))

                self.progress_manager.update_metrics(metrics, eta_str)

                # Log metrics if tensorboard logging is enabled
                if tensorboard_logging:
                    self._log_metrics(metrics, epoch)

                # Save checkpoint and check if it's the best model so far
                is_best = self.checkpoint_manager.save(
                    self.trainer.model,
                    self.trainer.optimizer,
                    metrics,
                    epoch,
                    self.trainer.config.to_dict()
                )

                # Check for early stopping
                if self.trainer.check_early_stopping(metrics.val_loss):
                    logger.info(
                        f"Early stopping triggered: No significant improvement for {self.trainer.config.patience} epochs")
                    self.early_stop = True
                    break

                start_time = time.time()

        except KeyboardInterrupt:
            self._handle_training_interrupt(metrics, val_loader)
            sys.exit(0)
        except Exception as e:
            logger.exception("Unhandled exception in training loop")
            self._handle_training_interrupt(metrics, val_loader)
            sys.exit(1)
        finally:
            # Remove the progress manager from the trainer
            self.trainer.set_progress_manager(None)
            if self.writer is not None:
                self.writer.close()
            self.progress_manager.cleanup(
                metrics, self.trainer.current_epoch, self.trainer.config.epochs)

            # Save final checkpoint
            final_checkpoint_path = Path(
                self.checkpoint_manager.save_dir) / 'final.pt'
            torch.save({
                'model_state_dict': self.trainer.model.state_dict(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'loss': metrics.val_loss,
                'esr': metrics.val_esr,
                'epoch': self.trainer.current_epoch,
                'config': self.trainer.config.to_dict()
            }, final_checkpoint_path)

            # Print checkpoint information
            print("\nCheckpoint Information:")
            print("-" * 50)
            logger.info(f"Final checkpoint saved to: {final_checkpoint_path}")
            if best_path := self.checkpoint_manager.get_best_checkpoint_path():
                logger.info(
                    f"Best model checkpoint: {best_path} (ESR: {self.checkpoint_manager.best_esr:.6f})")
            logger.info(
                f"Latest checkpoint: {self.checkpoint_manager.get_latest_checkpoint_path()}")

            sys.exit(0)

    def _setup_tensorboard(self, train_loader: DataLoader, open_tensorboard: bool = True):
        """Setup TensorBoard logging."""
        self.writer = SummaryWriter(log_dir="logs")
        logger.info("TensorBoard:")
        logger.info("-" * 50)
        try:
            example_batch = next(iter(train_loader))
            example_inputs = example_batch[0].to(self.trainer.device)
            batch_size = example_inputs.shape[0]
            model = self.trainer.model
            hidden_state = (
                model._initial_hidden.repeat(1, batch_size, 1).detach(),
                model._initial_cell.repeat(1, batch_size, 1).detach()
            )
            self.writer.add_graph(model, (example_inputs, hidden_state))
            tensorboard_process = subprocess.Popen([
                "tensorboard", "--logdir=logs", "--port=6006", "--reload_interval=5"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("TensorBoard server started at http://localhost:6006")
            if open_tensorboard:
                logger.info("\033[92m✓ Opening TensorBoard in browser\033[0m")
                webbrowser.open("http://localhost:6006")
        except Exception as e:
            logger.warning(
                "Failed to auto-start TensorBoard. Please run: tensorboard --logdir=logs manually.")
