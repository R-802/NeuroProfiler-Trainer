"""Training module for model training and management."""

# Standard library imports
from typing import Any, Optional, Tuple, Callable

# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

# Local imports
from engine.config.training_config import TrainingConfig
from engine.models.loss import esr
from engine.models.losses import create_loss_from_config
from engine.models.lstm import LSTMProfiler
from engine.trainer.factories.optimizer_factory import OptimizerFactory, SchedulerFactory
from engine.trainer.utils.tensor_validation import (
    validate_batch_tensors,
    validate_and_trim_sequences,
    validate_hidden_state
)
from engine.trainer.managers.progress_manager import ProgressManager, TrainingMetrics
from engine.trainer.base import BaseTrainer

logger = logging.getLogger(__name__)


class ModelTrainer(BaseTrainer):
    """Handles model training and validation following NAM's approach."""

    def __init__(self,
                 model: LSTMProfiler,
                 config: TrainingConfig,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 loss_fn: Optional[Callable] = None):
        """Initialize trainer with model, configuration, and optional dependencies."""
        self._model = model
        self._config = config
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._progress_manager = None
        self._current_epoch = 1
        self._batch_count = 0

        # Initialize model and hardware settings
        self._initialize_hardware_settings()
        self._initialize_model()

        # Initialize training components
        self._initialize_training_components(optimizer, scheduler, loss_fn)

        # Initialize training state
        self._initialize_training_state()

    def _initialize_hardware_settings(self):
        """Initialize hardware-specific settings."""
        self._config.hardware.mixed_precision = self._config.hardware.mixed_precision
        self._config.hardware.accumulation_steps = self._config.hardware.accumulation_steps
        self.scaler = torch.amp.GradScaler(
            enabled=self._config.hardware.mixed_precision)

    def _initialize_model(self):
        """Initialize model and move to device."""
        self._model.to(self._device)

    def _initialize_training_components(self, optimizer, scheduler, loss_fn):
        """Initialize loss function, optimizer and scheduler."""
        # Initialize loss function
        self.loss_fn = loss_fn if loss_fn is not None else create_loss_from_config(
            self._config.loss, self._device)

        # Initialize optimizer
        self._optimizer = optimizer if optimizer is not None else OptimizerFactory.create(
            self._model.parameters(), self._config.optimizer)

        # Initialize scheduler
        self._scheduler = scheduler if scheduler is not None else SchedulerFactory.create(
            self._optimizer, self._config.scheduler)

    def _initialize_training_state(self):
        """Initialize training state variables."""
        self.best_loss = float('inf')
        self.min_required_length = 2048  # Minimum length for MRSTFT computation
        self.improvement_threshold = self._config.improvement_threshold
        self.epochs_without_improvement = 0

    # Properties from BaseTrainer
    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self._scheduler

    @property
    def config(self) -> TrainingConfig:
        return self._config

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value: int):
        self._current_epoch = value

    @property
    def progress_manager(self) -> Optional[ProgressManager]:
        return self._progress_manager

    def set_progress_manager(self, progress_manager: Optional[ProgressManager]):
        self._progress_manager = progress_manager

    def train_epoch(self, train_loader: DataLoader) -> TrainingMetrics:
        """Run a single training epoch."""
        metrics = TrainingMetrics()
        self._model.train()

        if self._progress_manager:
            self._progress_manager.train_bar.reset()

        for batch_idx, (x, y) in enumerate(train_loader):
            loss, batch_esr = self._train_step((x, y), return_esr=True)
            metrics.train_loss += loss
            metrics.train_esr += batch_esr

            if self._progress_manager:
                self._progress_manager.update_train_progress()

        # Average the metrics
        metrics.train_loss /= len(train_loader)
        metrics.train_esr /= len(train_loader)

        return metrics

    def validate_epoch(self, val_loader: DataLoader) -> TrainingMetrics:
        """Run a single validation epoch."""
        metrics = TrainingMetrics()

        if self._progress_manager:
            self._progress_manager.val_bar.reset(total=len(val_loader))
            self._progress_manager.val_bar.refresh()

        # Pass the progress manager to validate
        metrics.val_loss, metrics.val_esr = self.validate(
            val_loader, self._progress_manager)

        # Update learning rate
        self._update_learning_rate(metrics.val_loss)
        metrics.learning_rate = self._optimizer.param_groups[0]['lr']

        return metrics

    def _update_learning_rate(self, val_loss: float):
        """Update learning rate based on scheduler type."""
        if self._scheduler.__class__.__name__ == "ReduceLROnPlateau":
            self._scheduler.step(val_loss)
        else:
            self._scheduler.step()

    def validate(self, val_loader: DataLoader, progress_manager: Optional[ProgressManager] = None) -> Tuple[float, float]:
        """Validate model on validation set."""
        self._model.eval()
        total_loss = 0.0
        total_esr = 0.0
        valid_batches = 0
        hidden_state = None
        hidden_size = self._model.lstm_hidden

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                batch_loss, batch_esr = self._validate_batch(
                    x, y, hidden_state, hidden_size)

                # Only accumulate metrics if batch was valid
                if batch_loss is not None:
                    total_loss += batch_loss
                    total_esr += batch_esr
                    valid_batches += 1

                # Update progress bar
                if progress_manager and progress_manager.show_progress:
                    progress_manager.val_bar.update(1)
                    progress_manager.val_bar.refresh()

                # If we've skipped too many batches, warn about it
                if batch_loss is None and batch_idx < len(val_loader) - 1:
                    logger.warning(
                        f"Validation batch {batch_idx + 1}/{len(val_loader)} was skipped due to validation failure")

            if valid_batches == 0:
                return float('inf'), float('inf')

            return total_loss / valid_batches, total_esr / valid_batches

    def _validate_batch(self, x: torch.Tensor, y: torch.Tensor,
                        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
                        hidden_size: int) -> Tuple[Optional[float], Optional[float]]:
        """Process a single validation batch."""
        x, y = x.to(self._device), y.to(self._device)

        if not validate_batch_tensors(x, y):
            logger.warning(
                f"Batch validation failed: x shape={x.shape}, y shape={y.shape}")
            return None, None

        if x.shape[1] < self._model.receptive_field:
            logger.warning(
                f"Sequence length {x.shape[1]} less than receptive field {self._model.receptive_field}")
            return None, None

        if not validate_hidden_state(hidden_state, x.size(0), hidden_size):
            logger.warning(
                f"Hidden state validation failed for batch size {x.size(0)} and hidden size {hidden_size}")
            hidden_state = None

        with torch.amp.autocast(device_type=self._device.type, enabled=self._config.hardware.mixed_precision):
            output, hidden_state = self._model(x, hidden_state)
            output, y = validate_and_trim_sequences(
                output, y,
                burn_in=self._config.model.train_burn_in,
                min_required_length=self.min_required_length
            )

            if output is None or y is None:
                logger.warning(
                    f"Sequence validation failed after burn-in and trimming")
                return None, None

            loss = self.loss_fn(output, y)
            esr_val = esr(output, y)

            return loss.item(), esr_val.item()

    def _train_step(self, batch: Tuple[torch.Tensor, torch.Tensor], return_esr: bool = False) -> Tuple[float, float]:
        """Execute a single training step with proper burn-in and masking."""
        x, y = batch
        x, y = x.to(self._device), y.to(self._device)

        if not validate_batch_tensors(x, y):
            return 0.0, 0.0

        if x.size(1) < self._model.receptive_field:
            return 0.0, 0.0

        loss, batch_esr = self._compute_loss_and_esr(x, y, return_esr)
        if loss is None:
            return 0.0, 0.0

        self._backward_pass(loss)

        return loss.item(), batch_esr

    def _compute_loss_and_esr(self, x: torch.Tensor, y: torch.Tensor, return_esr: bool) -> Tuple[Optional[torch.Tensor], float]:
        """Compute loss and ESR for a batch."""
        self._optimizer.zero_grad(set_to_none=True)
        device_type = 'cuda' if self._device.type == 'cuda' else 'cpu'

        with torch.amp.autocast(device_type=device_type, enabled=self._config.hardware.mixed_precision):
            output, _ = self._model(x)
            output, y = validate_and_trim_sequences(
                output, y,
                burn_in=self._config.model.train_burn_in,
                min_required_length=self.min_required_length
            )

            if output is None or y is None:
                return None, 0.0

            loss = self.loss_fn(output, y)
            batch_esr = esr(output, y).item() if return_esr else 0.0

            return loss, batch_esr

    def _backward_pass(self, loss: torch.Tensor):
        """Perform backward pass with gradient scaling."""
        self.scaler.scale(loss).backward()
        if self._config.hardware.grad_clip > 0:
            self.scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(), self._config.hardware.grad_clip)
        self.scaler.step(self._optimizer)
        self.scaler.update()

    def check_early_stopping(self, current_loss: float) -> bool:
        """Check if training should be stopped based on improvement threshold."""
        if self._current_epoch < self._config.min_epochs:
            return False

        if current_loss < self.best_loss * (1 - self.improvement_threshold):
            self.best_loss = current_loss
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= self._config.patience
