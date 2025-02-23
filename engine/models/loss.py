import torch.nn as nn
import torch
from typing import Optional
from auraloss.freq import MultiResolutionSTFTLoss
from engine.config.constants import SAMPLE_RATE
from engine.utils import logger


def apply_pre_emphasis_filter(x: torch.Tensor, coef: float) -> torch.Tensor:
    """
    Apply first-order pre-emphasis filter

    :param x: (*, L)
    :param coef: The coefficient

    :return: (*, L-1)
    """
    return x[..., 1:] - coef * x[..., :-1]


def esr(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate Error-to-Signal Ratio with automatic dimension handling"""
    if predictions.dim() == 3:
        predictions = predictions.squeeze(-1)
    if target.dim() == 3:
        target = target.squeeze(-1)

    if predictions.dim() != 2:
        raise ValueError(
            f"Expect 2D predictions (batch_size, num_samples). Got {predictions.shape}"
        )

    return torch.mean((predictions - target) ** 2) / torch.mean(target ** 2)


def multi_resolution_stft_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    loss_func: Optional[MultiResolutionSTFTLoss] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Experimental Multi Resolution Short Time Fourier Transform Loss using auraloss implementation.
    B: Batch size
    L: Sequence length

    :param preds: (B,L) or (B,L,1)
    :param targets: (B,L) or (B,L,1)
    :param loss_func: A pre-initialized instance of the loss function module. Providing
        this saves time.
    :param device: If provided, send the preds and targets to the provided device.
    :return: ()
    """
    # Ensure tensors are 2D (batch_size, sequence_length)
    if preds.dim() == 3:
        preds = preds.squeeze(-1)
    if targets.dim() == 3:
        targets = targets.squeeze(-1)

    # Ensure we have non-empty tensors with valid dimensions
    if preds.size(1) == 0 or targets.size(1) == 0:
        # Return zero loss if either tensor is empty
        return torch.tensor(0.0, device=preds.device)

    # Initialize loss function if not provided
    loss_func = MultiResolutionSTFTLoss() if loss_func is None else loss_func

    # Move tensors to specified device if provided
    if device is not None:
        preds, targets = preds.to(device), targets.to(device)
        loss_func = loss_func.to(device)

    return loss_func(preds, targets)


def mse_fft(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate MSE loss with frequency-dependent weighting.

    Args:
        preds: Predicted outputs (B, L).
        targets: Target outputs (B, L).

    Returns:
        MSE loss with frequency-dependent weighting.
    """
    # Ensure inputs are in float32 for FFT computation
    preds_fp = preds.float() if preds.dtype != torch.float32 else preds
    targets_fp = targets.float() if targets.dtype != torch.float32 else targets

    # Compute FFT in float32 precision
    fp = torch.fft.fft(preds_fp)
    ft = torch.fft.fft(targets_fp)

    # Create frequency-dependent weighting
    freqs = torch.fft.fftfreq(preds_fp.shape[-1], d=1.0/SAMPLE_RATE)
    # Emphasize frequencies above 3kHz with a gradual increase
    weights = torch.ones_like(freqs)
    high_freq_mask = torch.abs(freqs) > 3000
    # Linear increase up to 10kHz
    weights[high_freq_mask] = 1.0 + 2.0 * \
        (torch.abs(freqs[high_freq_mask]) - 3000) / 7000
    weights = weights.unsqueeze(0).unsqueeze(0)  # Match dimensions

    error = fp - ft
    weighted_error = error * weights.to(error.device)

    # Compute weighted mean squared error
    return torch.mean(torch.square(torch.abs(weighted_error)))


def mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate MSE loss with shape validation.

    Args:
        preds: Predictions tensor
        targets: Target tensor
    """
    # Ensure consistent dimensions
    if preds.dim() != targets.dim():
        if preds.dim() == 2 and targets.dim() == 3:
            preds = preds.unsqueeze(-1)
        elif targets.dim() == 2 and preds.dim() == 3:
            targets = targets.unsqueeze(-1)

    # Log shapes for debugging
    logger.debug(
        f"MSE Loss shapes - Preds: {preds.shape}, Targets: {targets.shape}")

    return nn.MSELoss()(preds, targets)


class CombinedLoss(nn.Module):
    def __init__(
        self,
        mse_weight: Optional[float] = 1.0,
        mse_fft_weight: Optional[float] = None,
        pre_emph_weight: Optional[float] = None,
        pre_emph_coef: Optional[float] = None,
        mrstft_weight: Optional[float] = None,
        pre_emph_mrstft_weight: Optional[float] = None,
        pre_emph_mrstft_coef: Optional[float] = None,
        esr_weight: Optional[float] = None,
        dc_weight: Optional[float] = None,
        mask_first: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Combined loss function for signal processing models.

        :param mse_weight: Weight for the MSE loss.
        :param mse_fft_weight: Weight for the Fourier (MSE_FFT) loss.
        :param pre_emph_weight: Weight for pre-emphasised MSE loss.
        :param pre_emph_coef: Coefficient for pre-emphasis filter.
        :param mrstft_weight: Weight for Multi-Resolution STFT loss.
        :param pre_emph_mrstft_weight: Weight for pre-emphasised MRSTFT loss.
        :param pre_emph_mrstft_coef: Coefficient for pre-emphasis filter for MRSTFT.
        :param esr_weight: Weight for Error Signal Ratio (ESR) loss.
        :param dc_weight: Weight for DC loss (mean difference).
        :param device: Optional device for loss computation.
        :param mask_first: Number of initial time steps/samples to ignore.
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.mse_fft_weight = mse_fft_weight
        self.pre_emph_weight = pre_emph_weight
        self.pre_emph_coef = pre_emph_coef
        self.mrstft_weight = mrstft_weight
        self.pre_emph_mrstft_weight = pre_emph_mrstft_weight
        self.pre_emph_mrstft_coef = pre_emph_mrstft_coef
        self.esr_weight = esr_weight
        self.dc_weight = dc_weight
        self.mask_first = mask_first
        self.device = device

        if mrstft_weight is not None or pre_emph_mrstft_weight is not None:
            self.mrstft_loss_func = MultiResolutionSTFTLoss().to(device)
        else:
            self.mrstft_loss_func = None

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined loss.

        :param preds: Predicted outputs (B, L).
        :param targets: Ground truth (B, L).
        :return: Combined loss scalar.
        """
        # Apply the mask_first: ignore the first mask_first samples if provided
        if self.mask_first is not None:
            preds = preds[..., self.mask_first:]
            targets = targets[..., self.mask_first:]

        total_loss = 0.0

        # MSE Loss
        if self.mse_weight is not None:
            total_loss += self.mse_weight * mse(preds, targets)

        # Fourier (MSE_FFT) Loss
        if self.mse_fft_weight is not None:
            total_loss += self.mse_fft_weight * mse_fft(preds, targets)

        # Pre-emphasised MSE Loss
        if self.pre_emph_weight is not None and self.pre_emph_coef is not None:
            preds_pre = apply_pre_emphasis_filter(preds, self.pre_emph_coef)
            targets_pre = apply_pre_emphasis_filter(
                targets, self.pre_emph_coef)
            total_loss += self.pre_emph_weight * mse(preds_pre, targets_pre)

        # MRSTFT Loss
        if self.mrstft_weight is not None and self.mrstft_loss_func is not None:
            total_loss += self.mrstft_weight * multi_resolution_stft_loss(
                preds, targets, loss_func=self.mrstft_loss_func, device=self.device
            )

        # Pre-emphasised MRSTFT Loss
        if (
            self.pre_emph_mrstft_weight is not None
            and self.pre_emph_mrstft_coef is not None
            and self.mrstft_loss_func is not None
        ):
            preds_pre = apply_pre_emphasis_filter(
                preds, self.pre_emph_mrstft_coef)
            targets_pre = apply_pre_emphasis_filter(
                targets, self.pre_emph_mrstft_coef)
            total_loss += self.pre_emph_mrstft_weight * multi_resolution_stft_loss(
                preds_pre, targets_pre, loss_func=self.mrstft_loss_func, device=self.device
            )

        # ESR Loss
        if self.esr_weight is not None:
            total_loss += self.esr_weight * esr(preds, targets)

        # DC Loss
        if self.dc_weight is not None:
            mean_preds = preds.mean(dim=1)  # Mean along the sequence dimension
            mean_targets = targets.mean(dim=1)
            dc_loss = mse(mean_preds, mean_targets)
            total_loss += self.dc_weight * dc_loss

        return total_loss
