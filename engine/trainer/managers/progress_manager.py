"""Progress management module for training visualization."""

from dataclasses import dataclass
from tqdm import tqdm
from typing import Optional


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_loss: float = 0.0
    train_esr: float = 0.0
    val_loss: float = 0.0
    val_esr: float = 0.0
    learning_rate: float = 0.0


class DummyTqdm:
    """Dummy progress bar for when progress display is disabled."""

    def __init__(self, *args, **kwargs):
        self.n = 0
        self.total = kwargs.get('total', 100)

    def update(self, *args, **kwargs): pass
    def set_description_str(self, *args, **kwargs): pass
    def set_postfix(self, *args, **kwargs): pass
    def clear(self, *args, **kwargs): pass
    def close(self, *args, **kwargs): pass
    def reset(self, total: Optional[int] = None, *args, **kwargs): pass
    def refresh(self, *args, **kwargs): pass
    def __iter__(self): return iter(range(self.total))


class ProgressManager:
    """Manages training progress bars and status displays."""

    def __init__(self, show_progress: bool, total_epochs: int, train_steps: int, val_steps: int):
        """Initialize progress manager.

        Args:
            show_progress: Whether to show progress bars
            total_epochs: Total number of epochs
            train_steps: Number of training steps per epoch
            val_steps: Number of validation steps per epoch
        """
        self.show_progress = show_progress
        self.train_steps = train_steps
        self.val_steps = val_steps

        if not show_progress:
            self._create_dummy_bars(total_epochs, train_steps, val_steps)
            return

        # Create progress bars with fixed positions
        self.epoch_counter = tqdm(
            bar_format='{desc}', desc=f"Epoch: 0/{total_epochs}", position=0, leave=False)
        self.train_esr_bar = tqdm(
            bar_format='{desc}', desc="Train ESR: waiting...", position=1, leave=False)
        self.val_esr_bar = tqdm(
            bar_format='{desc}', desc="Val ESR: waiting...", position=2, leave=False)
        self.spacer_bar = tqdm(
            bar_format='{desc}', desc="", position=3, leave=False)
        self.epoch_bar = tqdm(
            range(total_epochs), desc="Progress", position=4, leave=False, unit="epoch")
        self.train_bar = tqdm(
            total=train_steps, desc="Training", position=5, leave=False)
        self.val_bar = tqdm(
            total=val_steps, desc="Validation", position=6, leave=False)

    def _create_dummy_bars(self, total_epochs: int, train_steps: int, val_steps: int):
        """Create dummy progress bars when show_progress is False."""
        self.epoch_counter = DummyTqdm()
        self.train_esr_bar = DummyTqdm()
        self.val_esr_bar = DummyTqdm()
        self.spacer_bar = DummyTqdm()
        self.epoch_bar = DummyTqdm(total=total_epochs)
        self.train_bar = DummyTqdm(total=train_steps)
        self.val_bar = DummyTqdm(total=val_steps)

    def update_epoch(self, current_epoch: int, total_epochs: int):
        """Update epoch counter display."""
        if self.show_progress:
            self.epoch_counter.set_description_str(
                f"Epoch: {current_epoch}/{total_epochs}")

    def update_train_progress(self):
        """Update training progress display."""
        if self.show_progress:
            self.train_bar.update(1)

    def update_val_progress(self):
        """Update validation progress."""
        if self.show_progress:
            self.val_bar.update(1)

    def reset_val_bar(self):
        """Reset validation progress bar."""
        if self.show_progress:
            self.val_bar.reset(total=self.val_steps)
            self.val_bar.refresh()

    def update_metrics(self, metrics: TrainingMetrics, eta_str: str):
        """Update all metric displays."""
        if self.show_progress:
            self.train_esr_bar.set_description_str(
                f"Train ESR: {metrics.train_esr:.6f}".rstrip())
            self.val_esr_bar.set_description_str(
                f"Val ESR: {metrics.val_esr:.6f}".rstrip())
            self.epoch_bar.set_postfix({
                'train_loss': f'{metrics.train_loss:.4f}',
                'val_loss': f'{metrics.val_loss:.4f}',
                'lr': f'{metrics.learning_rate:.6f}',
                'ETA': eta_str
            })

    def cleanup(self, metrics: TrainingMetrics, current_epoch: int, total_epochs: int):
        """Clean up all progress bars."""
        if not self.show_progress:
            return

        # Clear and close all progress bars without printing any final status
        for bar in [self.train_bar, self.val_bar, self.epoch_bar,
                    self.spacer_bar, self.epoch_counter,
                    self.train_esr_bar, self.val_esr_bar]:
            bar.clear()
            bar.close()

        print(f"Epoch: {current_epoch}/{total_epochs}")
        print(f"Train ESR: {metrics.train_esr:.6f}")
        print(f"Val ESR: {metrics.val_esr:.6f}")
