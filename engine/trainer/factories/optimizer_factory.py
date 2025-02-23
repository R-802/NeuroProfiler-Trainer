"""Factory classes for creating optimizers and schedulers."""

import torch
from typing import Dict, Any, Type

from engine.config.training_config import OptimizerConfig, SchedulerConfig


class OptimizerFactory:
    """Factory class for creating optimizers."""

    OPTIMIZER_MAP: Dict[str, Type[torch.optim.Optimizer]] = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD
    }

    @classmethod
    def create(cls, model_params, optimizer_config: OptimizerConfig) -> torch.optim.Optimizer:
        """Create an optimizer instance based on configuration.

        Args:
            model_params: Model parameters to optimize
            optimizer_config: Optimizer configuration object

        Returns:
            torch.optim.Optimizer: Configured optimizer instance

        Raises:
            ValueError: If optimizer class is not supported
        """
        if optimizer_config.class_name not in cls.OPTIMIZER_MAP:
            raise ValueError(
                f"Unsupported optimizer class: {optimizer_config.class_name}")

        optimizer_kwargs = optimizer_config.kwargs
        return cls.OPTIMIZER_MAP[optimizer_config.class_name](model_params, **optimizer_kwargs)


class SchedulerFactory:
    """Factory class for creating learning rate schedulers."""

    SCHEDULER_MAP: Dict[str, Type[torch.optim.lr_scheduler._LRScheduler]] = {
        "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau
    }

    @classmethod
    def create(cls, optimizer: torch.optim.Optimizer, scheduler_config: SchedulerConfig) -> torch.optim.lr_scheduler._LRScheduler:
        """Create a scheduler instance based on configuration.

        Args:
            optimizer: The optimizer to schedule
            scheduler_config: Scheduler configuration object

        Returns:
            torch.optim.lr_scheduler._LRScheduler: Configured scheduler instance

        Raises:
            ValueError: If scheduler class is not supported
        """
        if scheduler_config.class_name not in cls.SCHEDULER_MAP:
            raise ValueError(
                f"Unsupported scheduler class: {scheduler_config.class_name}")

        scheduler_kwargs = scheduler_config.kwargs
        return cls.SCHEDULER_MAP[scheduler_config.class_name](optimizer, **scheduler_kwargs)
