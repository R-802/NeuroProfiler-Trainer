"""Training configuration module."""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import multiprocessing
from typing import Optional, Dict, Any
from engine.config.constants import SAMPLE_RATE
import torch
from engine.utils import logger


@dataclass
class AudioConfig:
    """Audio processing and alignment configuration."""
    # Core audio parameters (copy and pasted from NAM)
    sample_rate: float = SAMPLE_RATE
    lookahead: int = 100
    lookback: int = 10000
    abs_threshold: float = 0.0003
    rel_threshold: float = 0.001
    safety_factor: int = 0
    max_delay_disagreement: int = 20

    # V3 timing constants (in samples)
    validation1_end: int = int(9 * SAMPLE_RATE)      # 0:09
    validation1_start: int = 0                       # 0:00
    silence1_end: int = int(10 * SAMPLE_RATE)        # 0:10
    blip1_time: int = int(10.5 * SAMPLE_RATE)        # 0:10.5
    blip2_time: int = int(11.5 * SAMPLE_RATE)        # 0:11.5
    noise_start: int = int(10.25 * SAMPLE_RATE)      # 10.25s
    noise_end: int = int(10.375 * SAMPLE_RATE)       # 10.375s
    train_start: int = int(17 * SAMPLE_RATE)         # 0:17
    train_end: int = int(180.5 * SAMPLE_RATE)        # 3:00.5
    silence2_end: int = int(181 * SAMPLE_RATE)       # 3:01
    validation2_end: int = int(189 * SAMPLE_RATE)    # 3:10

    # Validation parameters
    esr_threshold: float = 0.01
    silence_threshold: float = 0.001
    t_blips: int = 96000

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AudioConfig':
        """Create config from dictionary, using defaults for missing values."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})

    @property
    def validation_length(self) -> int:
        """Length of validation segments in samples"""
        return self.validation1_end - self.validation1_start

    @property
    def blip_locations(self) -> tuple[int, int]:
        """Return the blip locations as a tuple"""
        return (self.blip1_time, self.blip2_time)

    @property
    def noise_interval(self) -> tuple[int, int]:
        """Return the noise interval for calibration"""
        return (self.noise_start, self.noise_end)


@dataclass
class ModelArchConfig:
    """Model architecture configuration."""
    name: str = "LSTMProfiler"
    input_dim: int = 1

    # WG-WaveNet specific parameters
    channels: Optional[int] = None

    # Common parameters
    num_layers: Optional[int] = None

    # LSTM specific parameters
    lstm_hidden: Optional[int] = None
    conv_filters: Optional[int] = None
    conv_kernel: Optional[int] = None
    conv_stride: Optional[int] = None
    dropout: float = 0.0
    train_burn_in: Optional[int] = None
    train_truncate: Optional[int] = None
    receptive_field: Optional[int] = None

    def __post_init__(self):
        """Validate configuration based on model type."""
        if self.name == "LSTMProfiler":
            if self.lstm_hidden is None:
                raise ValueError("lstm_hidden is required for LSTMProfiler")
        elif self.name == "WGWaveNet":
            if self.channels is None:
                raise ValueError(
                    "channels parameter is required for WGWaveNet")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ModelArchConfig':
        """Create config from dictionary, using defaults for missing values."""
        # Extract the model name first
        name = config.get("name", cls.name)
        # Get the config dict, defaulting to empty dict if not present
        config_dict = config.get("config", {})
        # Add the name to the config dict
        config_dict["name"] = name
        return cls(**config_dict)


@dataclass
class DataConfig:
    """Data processing configuration."""
    batch_size: Optional[int] = None
    segment_length: Optional[int] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DataConfig':
        return cls(**config)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    class_name: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OptimizerConfig':
        if "learning_rate" in config:
            lr = config.pop("learning_rate")
            if "kwargs" in config and isinstance(config["kwargs"], dict):
                config["kwargs"]["lr"] = lr
            else:
                config["kwargs"] = {"lr": lr}
        elif "kwargs" in config and "learning_rate" in config["kwargs"]:
            config["kwargs"]["lr"] = config["kwargs"].pop("learning_rate")
        return cls(**config)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    class_name: str = "ExponentialLR"
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SchedulerConfig':
        return cls(**config)


@dataclass
class LossConfig:
    """Loss function configuration for training and validation."""
    description: Optional[str] = None
    weights: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    validation_loss_function: Optional[str] = None
    mask_first: Optional[int] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LossConfig':
        """Create a LossConfig from a dictionary, handling both old and new formats."""
        if "weights" in config:
            # New format with structured weights
            return cls(
                description=config.get("description"),
                weights=config.get("weights", {}),
                validation_loss_function=config.get(
                    "validation_loss_function"),
                mask_first=config.get("mask_first")
            )
        else:
            # Old format - convert to new structure
            weights = {}
            weight_mapping = {
                "mse": "mse_weight",
                "mse_fft": "mse_fft_weight",
                "esr": "esr_weight",
                "dc": "dc_weight",
                "mrstft": "mrstft_weight",
                "pre_emph": {
                    "value": "pre_emph_weight",
                    "coefficient": "pre_emph_coef"
                },
                "pre_emph_mrstft": {
                    "value": "pre_emph_mrstft_weight",
                    "coefficient": "pre_emph_mrstft_coef"
                }
            }

            for new_key, old_key in weight_mapping.items():
                if isinstance(old_key, dict):
                    if old_key["value"] in config and config[old_key["value"]] is not None:
                        weights[new_key] = {
                            "value": config[old_key["value"]],
                            "coefficient": config.get(old_key["coefficient"])
                        }
                else:
                    if old_key in config and config[old_key] is not None:
                        weights[new_key] = {"value": config[old_key]}

            return cls(
                weights=weights,
                validation_loss_function=config.get(
                    "validation_loss_function"),
                mask_first=config.get("mask_first")
            )

    def get_weight(self, name: str) -> Optional[float]:
        """
        Get the weight value for a given loss component.

        If a key matching the expected name is not found but a similar key (e.g. one ending
        with '_weight') exists that would match after stripping that suffix, then print a warning.
        Otherwise, if no key is present (and other keys do exist), warn that the provided name
        does not match any known key.
        Do not warn if no weight was entered at all.
        """
        if name in self.weights:
            return self.weights[name].get("value")

        # Try to find a candidate key that ends with '_weight'
        candidate = None
        for key in self.weights:
            if key.lower().endswith("_weight") and key.lower().replace("_weight", "") == name.lower():
                candidate = key
                break

        if candidate:
            logger.warning(
                f"Loss weight '{name}' not found in configuration; found '{candidate}' instead. "
                f"Please update your configuration or use the correct key name."
            )
            return self.weights[candidate].get("value")
        return None

    def get_coefficient(self, name: str) -> Optional[float]:
        """
        Get the coefficient for a given loss component if it exists.

        If a key matching the expected name is not found but a similar key (e.g. one ending
        with '_coef') exists that would match after stripping that suffix, then print a warning.
        Otherwise, if no key is present (and other keys do exist), warn that the provided name
        does not match any known key.
        Do not warn if the coefficient was simply not provided.
        """
        if name in self.weights:
            return self.weights[name].get("coefficient")

        candidate = None
        for key in self.weights:
            if key.lower().endswith("_coef") and key.lower().replace("_coef", "") == name.lower():
                candidate = key
                break

        if candidate:
            logger.warning(
                f"Loss coefficient for '{name}' not found in configuration; found '{candidate}' instead. "
                "Please update your configuration or use the correct key name."
            )
            return self.weights[candidate].get("coefficient")
        return None


@dataclass
class HardwareConfig:
    """Hardware optimization configuration."""
    num_workers: int = multiprocessing.cpu_count()
    mixed_precision: bool = True
    grad_clip: Optional[float] = 1.0
    accumulation_steps: Optional[int] = 2

    def __post_init__(self):
        """Validate and adjust hardware settings."""
        # Ensure num_workers is non-negative
        self.num_workers = max(0, self.num_workers)

        # Disable mixed precision if CUDA is not available
        if self.mixed_precision and not torch.cuda.is_available():
            logger.warning(
                "\033[91mMixed precision disabled: CUDA not available\033[0m")
            self.mixed_precision = False

        # Validate grad_clip
        if self.grad_clip is not None:
            self.grad_clip = max(0.0, float(self.grad_clip))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'HardwareConfig':
        return cls(**config)


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    model: ModelArchConfig
    data: DataConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    scheduler: SchedulerConfig
    hardware: HardwareConfig
    audio: AudioConfig
    epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    patience: Optional[int] = None
    improvement_threshold: Optional[float] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from a structured dictionary."""
        return cls(
            model=ModelArchConfig.from_config(config_dict.get("model", {})),
            data=DataConfig.from_config(config_dict.get("data", {})),
            optimizer=OptimizerConfig.from_config(
                config_dict.get("optimizer", {})),
            loss=LossConfig.from_config(config_dict.get("loss", {})),
            scheduler=SchedulerConfig.from_config(
                config_dict.get("lr_scheduler", {})),
            hardware=HardwareConfig.from_config(
                config_dict.get("hardware", {})),
            audio=AudioConfig.from_config(config_dict.get("audio", {})),
            epochs=config_dict.get("epochs", cls.epochs),
            min_epochs=config_dict.get("min_epochs", cls.min_epochs),
            patience=config_dict.get("patience", cls.patience),
            improvement_threshold=config_dict.get(
                "improvement_threshold", cls.improvement_threshold),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a structured dictionary."""
        return {
            "model": {
                "name": self.model.name,
                "config": {
                    "input_dim": self.model.input_dim,
                    "channels": self.model.channels,
                    "num_layers": self.model.num_layers,
                    "conv_filters": self.model.conv_filters,
                    "conv_kernel": self.model.conv_kernel,
                    "conv_stride": self.model.conv_stride,
                    "lstm_hidden": self.model.lstm_hidden,
                    "dropout": self.model.dropout,
                    "train_burn_in": self.model.train_burn_in,
                    "train_truncate": self.model.train_truncate,
                }
            },
            "data": {
                "batch_size": self.data.batch_size,
                "segment_length": self.data.segment_length,
            },
            "optimizer": {
                "class_name": self.optimizer.class_name,
                "kwargs": self.optimizer.kwargs,
            },
            "lr_scheduler": {
                "class_name": self.scheduler.class_name,
                "kwargs": self.scheduler.kwargs,
            },
            "loss": {
                "description": self.loss.description,
                "weights": self.loss.weights,
                "validation_loss_function": self.loss.validation_loss_function,
                "mask_first": self.loss.mask_first,
            },
            "hardware": {
                "num_workers": self.hardware.num_workers,
                "mixed_precision": self.hardware.mixed_precision,
                "grad_clip": self.hardware.grad_clip,
                "accumulation_steps": self.hardware.accumulation_steps,
            },
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "lookahead": self.audio.lookahead,
                "lookback": self.audio.lookback,
                "abs_threshold": self.audio.abs_threshold,
                "rel_threshold": self.audio.rel_threshold,
                "safety_factor": self.audio.safety_factor,
                "max_delay_disagreement": self.audio.max_delay_disagreement,
                "t_blips": self.audio.t_blips,
                "silence_threshold": self.audio.silence_threshold,
                "esr_threshold": self.audio.esr_threshold,
            },
            "epochs": self.epochs,
            "min_epochs": self.min_epochs,
            "patience": self.patience,
            "improvement_threshold": self.improvement_threshold,
        }
