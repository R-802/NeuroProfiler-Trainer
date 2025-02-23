"""Utility functions for tensor validation and preprocessing."""

import torch
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def validate_batch_tensors(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Validate input and target tensors.

    Args:
        x: Input tensor
        y: Target tensor

    Returns:
        bool: True if tensors are valid, False otherwise
    """
    if x.numel() == 0 or y.numel() == 0:
        logger.warning("Skipping batch with empty tensors")
        return False

    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        logger.warning("Skipping batch with non-finite values")
        return False

    return True


def validate_and_trim_sequences(
    output: torch.Tensor,
    target: torch.Tensor,
    burn_in: int = 0,
    min_required_length: int = 2048
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Validate and trim output and target sequences.

    Args:
        output: Model output tensor
        target: Target tensor
        burn_in: Number of samples to remove from start
        min_required_length: Minimum required sequence length

    Returns:
        Tuple of processed output and target tensors, or (None, None) if invalid
    """
    # Remove singleton dimensions
    output = output.squeeze(-1)
    target = target.squeeze(-1)

    # Apply burn-in
    if burn_in > 0:
        if output.shape[1] <= burn_in + min_required_length:
            logger.warning(
                f"Output length {output.shape[1]} too short after burn-in "
                f"(need at least {min_required_length} additional samples)"
            )
            return None, None
        output = output[:, burn_in:]
        target = target[:, burn_in:]

    # Ensure same length
    min_length = min(output.shape[1], target.shape[1])
    if min_length < min_required_length:
        logger.warning(
            f"Sequence too short after trimming: {min_length} samples")
        return None, None

    output = output[:, :min_length]
    target = target[:, :min_length]

    # Final shape check
    if output.shape != target.shape:
        logger.warning(
            f"Shape mismatch: output {output.shape} vs target {target.shape}")
        return None, None

    return output, target


def validate_hidden_state(
    hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    hidden_size: int
) -> bool:
    """Validate hidden state dimensions.

    Args:
        hidden_state: Tuple of (hidden, cell) states
        batch_size: Expected batch size
        hidden_size: Expected hidden size

    Returns:
        bool: True if hidden state is valid, False otherwise
    """
    if hidden_state is None:
        return True

    expected_size = (1, batch_size, hidden_size)
    return hidden_state[0].size() == expected_size
