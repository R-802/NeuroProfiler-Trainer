import torch
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def normalize_waveform(data: torch.Tensor) -> torch.Tensor:
    """
    Normalize waveform to have values in the range [-1, 1] based on the maximum absolute value.
    Handles PyTorch tensors only.
    """
    data_max = torch.max(data.abs()) + \
        1e-8  # Add a small epsilon to avoid division by zero
    return data / data_max


def plot_waveforms(dry_train, proc_train, dry_val, proc_val):
    """Plot the waveforms of training and validation sets."""
    plt.figure(figsize=(12, 8))

    # Plot training waveforms
    plt.subplot(2, 2, 1)
    plt.plot(dry_train.numpy().flatten(), label='Dry Train', color='blue')
    plt.title('Dry Training Waveform')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(proc_train.numpy().flatten(),
             label='Processed Train', color='orange')
    plt.title('Processed Training Waveform')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot validation waveforms
    plt.subplot(2, 2, 3)
    plt.plot(dry_val.numpy().flatten(), label='Dry Validation', color='green')
    plt.title('Dry Validation Waveform')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(proc_val.numpy().flatten(),
             label='Processed Validation', color='red')
    plt.title('Processed Validation Waveform')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()


def preprocess_waveform(waveform: torch.Tensor, coeff: float = 0.97, target_rms: float = 0.1, max_value: float = 1.0) -> torch.Tensor:
    """
    Preprocess the waveform for guitar amp/effect pedal emulation.
    Applies pre-emphasis, RMS normalization, and optional peak clipping.
    """
    # Step 1: Apply pre-emphasis
    waveform = torch.cat([waveform[:1], waveform[1:] - coeff * waveform[:-1]])

    # Step 2: Normalize to target RMS
    rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
    waveform = waveform * (target_rms / rms)

    # Step 3: Optional peak clipping
    waveform = torch.clamp(waveform, -max_value, max_value)

    return waveform


def segment_waveform(waveform, segment_length, hop_length):
    return [
        waveform[i: i + segment_length]
        for i in range(0, len(waveform) - segment_length + 1, hop_length)
    ]


def slice_waveforms(dry_wave, proc_wave, train_start, validation_length):
    n = len(dry_wave)
    validation_start = n - validation_length

    if n < train_start + validation_length:
        raise ValueError(
            f"Waveform too short! Need at least {train_start + validation_length} samples, got {n}."
        )

    dry_train, proc_train = dry_wave[train_start:
                                     validation_start], proc_wave[train_start:validation_start]
    dry_val, proc_val = dry_wave[validation_start:], proc_wave[validation_start:]
    return dry_train, proc_train, dry_val, proc_val


def check_audio_normalization(data: torch.Tensor, name: str = "Audio") -> bool:
    """
    Check if audio data is normalized between [-1, 1]
    Returns True if normalized, False otherwise
    """
    max_abs = torch.max(torch.abs(data))
    is_normalized = max_abs <= 1.0

    if is_normalized:
        logger.info(
            f"{name} is properly normalized between [-1, 1] (max abs: {max_abs:.6f})")
    else:
        logger.warning(
            f"{name} is not normalized! Max absolute value: {max_abs:.6f}")

    return is_normalized
