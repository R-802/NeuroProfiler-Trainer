import torch
from typing import Optional
from engine.config.training_config import LossConfig
from engine.models.loss import CombinedLoss


def create_loss_from_config(config: LossConfig, device: Optional[torch.device] = None) -> CombinedLoss:
    """
    Create a CombinedLoss instance from a LossConfig.

    :param config: Loss configuration
    :param device: Optional device to place the loss function on
    :return: Configured CombinedLoss instance
    """
    return CombinedLoss(
        mse_weight=config.get_weight('mse'),
        mse_fft_weight=config.get_weight('mse_fft'),
        pre_emph_weight=config.get_weight('pre_emph'),
        pre_emph_coef=config.get_coefficient('pre_emph'),
        mrstft_weight=config.get_weight('mrstft'),
        pre_emph_mrstft_weight=config.get_weight('pre_emph_mrstft'),
        pre_emph_mrstft_coef=config.get_coefficient('pre_emph_mrstft'),
        esr_weight=config.get_weight('esr'),
        dc_weight=config.get_weight('dc'),
        mask_first=config.mask_first,
        device=device
    )
