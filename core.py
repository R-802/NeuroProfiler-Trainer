"""
NeuroProfiler Audio Processing Pipeline

A modular pipeline for training LSTM models on aligned audio signals.
Handles data loading, preprocessing, model training, and visualization.
"""

import json
import os
import random
from datetime import datetime
import multiprocessing
import copy

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

# Local application imports
from engine.utils import logger
from engine.models.lstm import LSTMProfiler
from engine.audio.processor import AudioProcessor, V3Dataset
from engine.trainer import ModelTrainer, TrainingManager
from engine.config.training_config import (
    TrainingConfig, ModelArchConfig, DataConfig, AudioConfig,
    OptimizerConfig, LossConfig, SchedulerConfig, HardwareConfig
)
from engine.audio.alignment import AlignmentConfig
from engine.config.constants import SAMPLE_RATE

OPTIMIZE = False


def set_seed(seed: int = 42):
    """Set random seeds with optimized cuDNN settings"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def create_dataloaders(
    model,
    config: TrainingConfig,
    dry_train: torch.Tensor,
    proc_train: torch.Tensor,
    dry_val: torch.Tensor,
    proc_val: torch.Tensor
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders following NAM's approach."""

    receptive_field = model.receptive_field

    # Create datasets
    train_dataset = V3Dataset(
        dry=dry_train,
        proc=proc_train,
        config=config.audio,
        segment_length=config.data.segment_length,
        receptive_field=receptive_field
    )

    val_dataset = V3Dataset(
        dry=dry_val,
        proc=proc_val,
        config=config.audio,
        segment_length=config.data.segment_length,
        receptive_field=receptive_field
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.hardware.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def create_model(config: TrainingConfig) -> LSTMProfiler:
    """Create model instance with configuration."""
    return LSTMProfiler(
        input_dim=config.model.input_dim,
        lstm_hidden=config.model.lstm_hidden,
        conv_filters=config.model.conv_filters,
        conv_kernel=config.model.conv_kernel,
        conv_stride=config.model.conv_stride,
        dropout=config.model.dropout,
        train_burn_in=config.model.train_burn_in,
        train_truncate=config.model.train_truncate,
        num_layers=config.model.num_layers
    )


def prepare_pipeline(config):
    """Prepare audio processing, model creation, and dataloader setup."""
    # Create alignment config from audio config
    alignment_config = AlignmentConfig(config.audio)

    # Initialize audio processor
    processor = AudioProcessor(alignment_config)

    # Load input and target wav files
    dry, proc, _ = processor.load_wav_pair(
        input_path="data/input.wav", target_path="data/target.wav")

    # Validate structure of loaded audio
    processor.validate_structure(dry, proc)

    # Align input and target signals
    dry, proc = processor.align_signals(dry, proc, alignment_config)

    # Validate alignment was successful
    if not processor.validate_alignment(dry, proc, alignment_config, plot_debug=False):
        raise RuntimeError("Alignment validation failed")

    # Split into train/validation sets
    dry_train, proc_train, dry_val, proc_val = processor.slice_v3_waves(
        dry, proc)

    # Create model from config
    model = create_model(config)

    # Create train/val dataloaders
    train_loader, val_loader = create_dataloaders(
        model, config, dry_train, proc_train, dry_val, proc_val)

    return model, train_loader, val_loader


def run_training_pipeline(config):
    """Run the full training pipeline using the provided configuration."""
    # Prepare model and data
    model, train_loader, val_loader = prepare_pipeline(config)

    # Initialize trainer
    trainer = ModelTrainer(model, config)

    # Create training manager
    manager = TrainingManager(
        trainer, show_progress=True, display_config=False)

    # Run training
    manager.train(train_loader, val_loader,
                  open_tensorboard=False, tensorboard_logging=False)


def preprocess_data(config):
    """Perform data preprocessing (alignment, slicing) and return preprocessed train/val data."""
    # Create alignment config
    alignment_config = AlignmentConfig(config.audio)

    # Initialize audio processor
    processor = AudioProcessor(alignment_config)

    # Load input and target wav files
    dry, proc, _ = processor.load_wav_pair(
        input_path="data/input.wav", target_path="data/NAM_Audio/NAM_EXAMPLE_target.wav")

    # Validate structure of loaded audio
    processor.validate_structure(dry, proc)

    # Align input and target signals
    dry, proc = processor.align_signals(dry, proc, alignment_config)

    # Validate alignment was successful
    if not processor.validate_alignment(dry, proc, alignment_config, plot_debug=False):
        raise RuntimeError("Alignment validation failed")

    # Split into train/validation sets and return
    return processor.slice_v3_waves(dry, proc)


class HyperparameterSpace:
    """Defines the search space for hyperparameter optimization."""

    def __init__(self):
        # Model architecture parameters - only optimize the most impactful ones
        self.model_params = {
            "lstm_hidden": {"type": "int", "low": 16, "high": 64, "step": 16},
            "num_layers": {"type": "int", "low": 1, "high": 3, "step": 1},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
        }

        # Fixed model parameters that don't need optimization
        self.fixed_model_params = {
            "lstm_hidden": 18,
            "num_layers": 3,
            "dropout": 0.0,
            "train_burn_in": 8192,
            "train_truncate": None
        }

        # Optimizer parameters - only optimize learning rate
        self.optimizer_params = {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        }

        # Fixed optimizer parameters with proven good defaults
        self.fixed_optimizer_params = {
            "optimizer_type": "Adam",
            "learning_rate": 0.008,
        }

        # Scheduler parameters - only optimize key parameters
        self.scheduler_params = {
            # For ExponentialLR
            "gamma": {"type": "float", "low": 0.9, "high": 0.999},
        }

        # Fixed scheduler parameters
        self.fixed_scheduler_params = {
            "scheduler_type": "ExponentialLR",  # Simple and effective
        }

        # Training parameters - only optimize batch size
        self.training_params = {
            "batch_size": {"type": "int", "step": 16},
        }

        # Fixed training parameters
        self.fixed_training_params = {
            "segment_length": 32768,    # Standard for audio
            "epochs": 100,              # Standard training length
            "min_epochs": 10,           # Reasonable minimum
            "patience": 10,             # Standard early stopping
            "improvement_threshold": 1e-6,  # Standard threshold
        }

        # Fixed hardware parameters - these don't affect accuracy
        self.fixed_hardware_params = {
            "num_workers": multiprocessing.cpu_count(),
            "mixed_precision": True,
            "grad_clip": 1.0,
            "accumulation_steps": 1,
        }

        # Loss parameters
        # The search space for each loss component is defined here.
        self.loss_params = {
            "mse": {"type": "float", "low": 0.5, "high": 2.0},
            "esr": {"type": "float", "low": 0.0, "high": 1.0},
            "dc": {"type": "float", "low": 0.0, "high": 1.0},
            "pre_emph_mrstft": {
                "value": {"type": "float", "low": 0.0, "high": 0.01},
                "coefficient": {"type": "float", "low": 0.5, "high": 1.0},
            },
            "mask_first": {"type": "int", "low": 0, "high": 8192, "step": 1024},
        }

        # Which parameter groups to optimize (enable loss tuning by setting to True)
        self.optimize_groups = {
            "model": True,        # Only optimize core model params
            "optimizer": True,    # Only optimize learning rate
            "scheduler": True,    # Only optimize gamma
            "training": True,     # Only optimize batch size
            "hardware": False,    # Don't optimize hardware params
            "loss": True,         # Allow users to adjust loss weights
        }

    def suggest_params(self, trial: optuna.Trial) -> dict:
        """Suggest parameters based on the defined search space."""
        params = {}

        # Start with fixed parameters
        params.update(self.fixed_model_params)

        # Model parameters
        if self.optimize_groups["model"]:
            for name, spec in self.model_params.items():
                if spec["type"] == "int":
                    params[name] = trial.suggest_int(
                        name, spec["low"], spec["high"], step=spec.get("step", 1))
                elif spec["type"] == "float":
                    params[name] = trial.suggest_float(
                        name, spec["low"], spec["high"], step=spec.get("step"))

        # Optimizer parameters
        if self.optimize_groups["optimizer"]:
            params["optimizer"] = {
                "class_name": self.fixed_optimizer_params["optimizer_type"],
                "kwargs": {
                    "lr": trial.suggest_float(
                        "learning_rate",
                        self.optimizer_params["learning_rate"]["low"],
                        self.optimizer_params["learning_rate"]["high"],
                        log=self.optimizer_params["learning_rate"].get("log", False)),
                }
            }

        # Scheduler parameters
        if self.optimize_groups["scheduler"]:
            params["scheduler"] = {
                "class_name": self.fixed_scheduler_params["scheduler_type"],
                "kwargs": {
                    "gamma": trial.suggest_float(
                        "gamma",
                        self.scheduler_params["gamma"]["low"],
                        self.scheduler_params["gamma"]["high"])
                }
            }

        # Training parameters
        if self.optimize_groups["training"]:
            params["batch_size"] = trial.suggest_int(
                "batch_size",
                self.training_params["batch_size"]["low"],
                self.training_params["batch_size"]["high"],
                step=self.training_params["batch_size"]["step"]
            )
            params.update(self.fixed_training_params)

        # Hardware parameters (fixed)
        params["hardware"] = self.fixed_hardware_params

        # Loss parameters - allow adjusting them if specified
        if self.optimize_groups["loss"]:
            loss_params = {}
            for loss_key, spec in self.loss_params.items():
                if "type" in spec:
                    # Single value parameter
                    if spec["type"] == "int":
                        loss_params[loss_key] = trial.suggest_int(
                            loss_key, spec["low"], spec["high"], step=spec.get("step", 1))
                    elif spec["type"] == "float":
                        loss_params[loss_key] = trial.suggest_float(
                            loss_key, spec["low"], spec["high"], step=spec.get("step"))
                else:
                    # Nested parameters (e.g., pre_emph, pre_emph_mrstft)
                    nested_params = {}
                    for nested_key, nested_spec in spec.items():
                        param_name = f"{loss_key}_{nested_key}"
                        if nested_spec["type"] == "int":
                            nested_params[nested_key] = trial.suggest_int(
                                param_name, nested_spec["low"], nested_spec["high"], step=nested_spec.get("step", 1))
                        elif nested_spec["type"] == "float":
                            nested_params[nested_key] = trial.suggest_float(
                                param_name, nested_spec["low"], nested_spec["high"], step=nested_spec.get("step"))
                    loss_params[loss_key] = nested_params
            params["loss"] = {"weights": loss_params}
        else:
            # Default loss weights if not optimizing loss
            params["loss"] = {"weights": {
                "mse": 1.0,
                "mse_fft": 0.0,
                "esr": 0.35,
                "dc": 0.0,
                "mrstft": 0.0,
                "pre_emph": {"value": 0.0, "coefficient": 0.95},
                "pre_emph_mrstft": {"value": 0.002, "coefficient": 0.85},
                "mask_first": 0,
            }}

        return params


def main():
    try:
        # Load default configuration
        with open("engine/config/config.json", "r") as f:
            default_config = json.load(f)

        if OPTIMIZE:
            # Initialize hyperparameter search space
            hparam_space = HyperparameterSpace()

            # Configure which parameter groups to optimize
            hparam_space.optimize_groups.update({
                "model": True,        # Optimize model architecture
                "optimizer": True,    # Optimize optimizer settings
                "scheduler": False,   # Don't optimize scheduler
                "training": False,
                "hardware": False,    # Don't optimize hardware params
                "loss": True,         # Allow users to adjust loss weights
            })

            # Preprocess data only once using the baseline config
            baseline_config = TrainingConfig.from_dict(default_config)
            preprocessed_data = {}
            preprocessed_data['dry_train'], preprocessed_data['proc_train'], \
                preprocessed_data['dry_val'], preprocessed_data['proc_val'] = preprocess_data(
                    baseline_config)

            def objective(trial):
                # Create a deep copy of the default configuration
                config_trial = copy.deepcopy(default_config)

                # Get suggested parameters from the search space
                suggested_params = hparam_space.suggest_params(trial)

                # Update configuration with suggested parameters
                if "model" in suggested_params:
                    config_trial["model"]["config"].update(
                        suggested_params["model"])
                if "optimizer" in suggested_params:
                    config_trial["optimizer"].update(
                        suggested_params["optimizer"])
                if "scheduler" in suggested_params:
                    config_trial["lr_scheduler"].update(
                        suggested_params["scheduler"])
                if "training" in suggested_params:
                    config_trial["data"].update(suggested_params["training"])
                if "loss" in suggested_params:
                    config_trial["loss"]["weights"].update(
                        suggested_params["loss"])

                # For faster optimization, train for one epoch
                config_trial["epochs"] = 1
                config_trial["min_epochs"] = 1

                # Create training configuration
                config_obj = TrainingConfig.from_dict(config_trial)

                # Create model and dataloaders using preprocessed data
                model = create_model(config_obj)
                train_loader, val_loader = create_dataloaders(
                    model, config_obj,
                    preprocessed_data['dry_train'], preprocessed_data['proc_train'],
                    preprocessed_data['dry_val'], preprocessed_data['proc_val']
                )

                # Initialize trainer and run one epoch
                trainer = ModelTrainer(model, config_obj)
                trainer.model.train()
                for batch in train_loader:
                    trainer._train_step(batch)

                # Run validation and return the loss as the objective
                val_loss, _ = trainer.validate(val_loader)
                return val_loss

            # Callback function to save trial results after each trial.
            # This will overwrite the JSON file (./trials/trials.json) with the current study details.
            def trial_callback(study, trial):
                trials_list = []
                for t in study.trials:
                    trial_dict = {
                        "number": t.number,
                        "value": t.value,
                        "params": t.params,
                        "state": t.state.name,
                        "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
                        "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
                    }
                    trials_list.append(trial_dict)

                output_dir = "./trials"
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, "trials.json")
                with open(output_file, "w") as f:
                    json.dump(trials_list, f, indent=4)

            print("\nHyperparameter Optimization:")
            print("-" * 50)
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=1, callbacks=[trial_callback])

            # Get the configuration from the best trial and save it
            best_config = copy.deepcopy(default_config)
            best_params = study.best_params

            def safe_update(target: dict, target_key: str, source: dict, source_key: str = None):
                """
                Update the target dictionary at target_key with the value from source[source_key]
                if source_key exists; otherwise log a warning.
                """
                key = source_key or target_key
                if key in source:
                    target[target_key] = source[key]
                else:
                    logger.warning(f"Parameter '{key}' not found in best trial parameters; "
                                   f"'{target_key}' not updated.")

            # Update model configuration parameters
            for param in ["lstm_hidden", "num_layers", "dropout"]:
                safe_update(best_config["model"]["config"], param, best_params)

            # Update optimizer configuration
            safe_update(best_config["optimizer"]["kwargs"],
                        "lr", best_params, "learning_rate")

            # Update scheduler configuration only if available
            if "gamma" in best_params:
                safe_update(best_config["lr_scheduler"]
                            ["kwargs"], "gamma", best_params)
            else:
                logger.warning("Parameter 'gamma' not found in best trial parameters; "
                               "scheduler parameters remain unchanged.")

            # Update loss configuration (flat keys)
            for param in ["mse", "esr", "dc", "mask_first"]:
                safe_update(best_config["loss"]["weights"], param, best_params)

            # Update nested loss configuration for pre_emph_mrstft
            if "pre_emph_mrstft_value" in best_params and "pre_emph_mrstft_coefficient" in best_params:
                best_config["loss"]["weights"]["pre_emph_mrstft"] = {
                    "value": best_params["pre_emph_mrstft_value"],
                    "coefficient": best_params["pre_emph_mrstft_coefficient"]
                }
            else:
                logger.warning("Parameters 'pre_emph_mrstft_value' and/or "
                               "'pre_emph_mrstft_coefficient' not found in best trial parameters; "
                               "pre_emph_mrstft not updated.")

            # Finalize epochs (or any other non-optimized configs)
            best_config["epochs"] = default_config["epochs"]
            best_config["min_epochs"] = default_config["min_epochs"]

            # Save the best configuration
            best_config_path = "./trials/best_config.json"
            with open(best_config_path, "w") as f:
                json.dump(best_config, f, indent=4)

            logger.info(
                f"Best configuration has been saved to {best_config_path}")
        else:
            logger.info("Starting training pipeline...")
            config = TrainingConfig.from_dict(default_config)
            run_training_pipeline(config)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise


if __name__ == "__main__":
    main()
