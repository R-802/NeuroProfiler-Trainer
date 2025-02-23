"""
Core alignment module implementing NAM's exact approach for audio signal alignment.
Handles calibration, alignment, and validation of audio signals.
"""

import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from engine.config.constants import SAMPLE_RATE
from engine.utils import logger
from dataclasses import dataclass
from engine.config.training_config import AudioConfig
from engine.models.loss import esr


@dataclass
class AlignmentConfig:
    """Configuration for V3 data format alignment"""
    audio_config: AudioConfig = None

    def __post_init__(self):
        self.audio_config = self.audio_config or AudioConfig()

    # Forward all properties to audio_config
    def __getattr__(self, name):
        if hasattr(self.audio_config, name):
            return getattr(self.audio_config, name)
        raise AttributeError(f"'AlignmentConfig' has no attribute '{name}'")


def validate_input_signals(dry: np.ndarray, processed: np.ndarray, config: AlignmentConfig) -> None:
    """Validate input signals before alignment"""
    # Input validation
    if dry is None or processed is None:
        raise ValueError("Input signals cannot be None")

    if len(dry) != len(processed):
        raise ValueError(
            f"Signal lengths must match: dry ({len(dry)}) != processed ({len(processed)})"
        )

    # Check minimum length requirement
    min_required_length = config.silence1_end + config.t_blips
    if len(dry) < min_required_length:
        raise ValueError(
            f"Signals too short: {len(dry)} samples < {min_required_length} required")


def align_signals(dry: np.ndarray, processed: np.ndarray, config: AudioConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align signals using NAM's v3 approach.

    Args:
        dry: Input signal
        processed: Processed signal
        config: Audio configuration

    Returns:
        Tuple of (aligned_dry, aligned_proc)
    """
    # First validate gear consistency
    if not validate_gear_consistency(processed, config):
        logger.warning(
            "Proceeding with alignment despite gear consistency issues")

    # Calculate delay using NAM's method
    delay = calibrate_latency(processed, config)
    logger.info(
        f"Detected delay: {delay} samples ({delay/config.sample_rate}s)")

    # Apply alignment
    if delay > 0:
        # Processed signal is delayed
        aligned_proc = processed[delay:]
        aligned_dry = dry[:len(aligned_proc)]
    else:
        # Dry signal is delayed
        delay = abs(delay)
        aligned_dry = dry[delay:]
        aligned_proc = processed[:len(aligned_dry)]

    return aligned_dry, aligned_proc


def calibrate_latency(
    processed: np.ndarray,
    config: AlignmentConfig
) -> int:
    """
    Enhanced latency calibration using NAM's exact averaged blip detection approach.
    """
    # Extract the blip region
    y = processed[config.silence1_end:config.silence1_end + config.t_blips]

    # Calculate background level from noise interval
    background_level = np.max(np.abs(y[
        config.noise_interval[0] - config.silence1_end:
        config.noise_interval[1] - config.silence1_end
    ]))

    # Set trigger threshold using NAM's exact values
    trigger_threshold = max(
        background_level + 0.0003,  # NAM's absolute threshold
        (1.0 + 0.001) * background_level  # NAM's relative threshold
    )

    # Collect scans for both blips
    y_scans = []
    for blip_time in [config.blip1_time, config.blip2_time]:
        i_rel = blip_time - config.silence1_end
        start_looking = i_rel - config.lookahead
        stop_looking = i_rel + config.lookback
        y_scans.append(y[start_looking:stop_looking])

    # Average scans before threshold detection (NAM's approach)
    y_scan_average = np.mean(np.stack(y_scans), axis=0)
    triggered = np.where(np.abs(y_scan_average) > trigger_threshold)[0]

    if len(triggered) == 0:
        logger.error("No trigger point found in averaged scan")
        return 0

    # Calculate delay from first trigger point
    j = triggered[0]
    i_rel = config.blip1_time - config.silence1_end
    delay = j - config.lookahead

    # Apply NAM's exact safety factor
    return delay - 1  # NAM's safety factor


def calibrate_latency_nam_v3(processed: np.ndarray, config: AudioConfig) -> int:
    """
    Calibrate latency using NAM's v3 approach of averaging blip scans.

    Args:
        processed: The processed signal
        config: Audio configuration

    Returns:
        Recommended delay in samples
    """
    # Extract the blip region
    y = processed[config.silence1_end:config.silence1_end + config.t_blips]

    # Calculate background level from noise interval
    background_level = np.max(np.abs(y[
        config.noise_start - config.silence1_end:
        config.noise_end - config.silence1_end
    ]))

    # Set trigger threshold
    trigger_threshold = max(
        background_level + config.abs_threshold,
        (1.0 + config.rel_threshold) * background_level
    )

    # Collect scans for both blips
    y_scans = []
    for blip_time in [config.blip1_time, config.blip2_time]:
        i_rel = blip_time - config.silence1_end
        start_looking = i_rel - config.lookahead
        stop_looking = i_rel + config.lookback
        y_scans.append(y[start_looking:stop_looking])

    # Average scans before threshold detection (NAM's approach)
    y_scan_average = np.mean(np.stack(y_scans), axis=0)
    triggered = np.where(np.abs(y_scan_average) > trigger_threshold)[0]

    if len(triggered) == 0:
        logger.error("No trigger point found in averaged scan")
        return 0

    # Calculate delay from first trigger point
    j = triggered[0]
    i_rel = config.blip1_time - config.silence1_end
    delay = j - config.lookahead

    # Apply safety factor
    return delay - config.safety_factor


def validate_alignment(
    dry: torch.Tensor,
    processed: torch.Tensor,
    alignment_config: AlignmentConfig = AlignmentConfig(),
    plot_debug: bool = False,
    wav_info: Optional[Dict] = None
) -> bool:
    """
    Validate alignment using NAM's exact approach:
    1. Compare ESR between blip pairs
    2. Compare ESR between validation segments
    3. Verify silence periods
    4. Check length tolerances
    """

    # Convert inputs to tensors if they're numpy arrays
    if isinstance(dry, np.ndarray):
        dry = torch.from_numpy(dry).float()
    if isinstance(processed, np.ndarray):
        processed = torch.from_numpy(processed).float()

    # Step 1: Extract and validate blips
    blips = []
    for blip_time in [alignment_config.blip1_time, alignment_config.blip2_time]:
        window_start = blip_time - alignment_config.lookahead
        window_end = blip_time + alignment_config.lookback
        blip = processed[window_start:window_end]
        blips.append(blip.unsqueeze(0))

    if len(blips) >= 2:
        # Calculate ESR between blip pairs
        esr_blips = esr(blips[1], blips[0]).item()
        logger.info(f"Blip pair ESR: {esr_blips:.6f}")

        if esr_blips > alignment_config.audio_config.esr_threshold:
            logger.error(
                f"\033[91m⚠ Blip alignment validation failed: ESR = {esr_blips:.6f} > {alignment_config.audio_config.esr_threshold}\033[0m")
            return False

    # Step 2: Validate validation segments
    validation1 = processed[:alignment_config.validation1_end]
    validation2 = processed[alignment_config.silence2_end:alignment_config.validation2_end]

    # Ensure equal lengths for validation segments
    min_length = min(len(validation1), len(validation2))
    validation1 = validation1[:min_length]
    validation2 = validation2[:min_length]

    # Calculate ESR between validation segments
    esr_replicate = esr(validation2.unsqueeze(
        0), validation1.unsqueeze(0)).item()
    logger.info(f"Validation segment ESR: {esr_replicate:.8f}")

    if esr_replicate > alignment_config.audio_config.esr_threshold:
        logger.error(
            f"\033[91m⚠ Validation segment consistency failed: ESR = {esr_replicate:.6f} > {alignment_config.audio_config.esr_threshold}\033[0m")
        return False

    # Step 3: Validate silence periods
    SILENCE_THRESHOLD = alignment_config.audio_config.silence_threshold

    # Check first silence period (0:09-0:10)
    silence1_start = alignment_config.validation1_end
    silence1 = processed[silence1_start:alignment_config.silence1_end]
    silence1_level = float(torch.max(torch.abs(silence1)).item())

    # Check second silence period (3:00.5-3:01)
    silence2_start = alignment_config.train_end
    silence2_end = alignment_config.silence2_end
    silence2 = processed[silence2_start:silence2_end]
    silence2_level = float(torch.max(torch.abs(silence2)).item())

    if silence1_level > SILENCE_THRESHOLD or silence2_level > SILENCE_THRESHOLD:
        logger.error(
            f"\033[91m⚠ Silence validation failed: levels {silence1_level:.6f}, {silence2_level:.6f} > {SILENCE_THRESHOLD}\033[0m")
        return False

    # Step 4: Length tolerance validation
    length_diff = abs(len(dry) - len(processed))
    max_diff_samples = int(
        0.1 * alignment_config.sample_rate)  # 100ms tolerance
    if length_diff > max_diff_samples:
        logger.error(
            f"\033[91m⚠ Length difference {length_diff/alignment_config.sample_rate:.3f}s exceeds tolerance\033[0m")
        return False

    return True


def detect_blips(audio_signal: np.ndarray, config: AlignmentConfig) -> List[int]:
    """
    Detect blips in the audio signal using NAM's averaged scan method.
    This implementation exactly matches NAM's approach of averaging scans before threshold detection.

    Args:
        audio_signal: The audio signal as a numpy array
        config: Alignment configuration

    Returns:
        List of sample indices where blips are detected
    """
    # If stereo, use the mean of both channels
    if audio_signal.ndim > 1 and audio_signal.shape[0] > 1:
        detection_signal = np.mean(audio_signal, axis=0)
    else:
        detection_signal = np.squeeze(audio_signal)

    # Extract the blip region
    y = detection_signal[config.silence1_end:config.silence1_end + config.t_blips]

    # Calculate background level from noise interval
    background_level = np.max(np.abs(y[
        config.noise_start - config.silence1_end:
        config.noise_end - config.silence1_end
    ]))

    # Set trigger threshold based on noise floor
    trigger_threshold = max(
        background_level + config.abs_threshold,
        (1.0 + config.rel_threshold) * background_level
    )

    # Collect scans for all blips
    y_scans = []
    for blip_time in [config.blip1_time, config.blip2_time]:
        i_rel = blip_time - config.silence1_end
        start_looking = i_rel - config.lookahead
        stop_looking = i_rel + config.lookback
        y_scans.append(y[start_looking:stop_looking])

    # Average scans before threshold detection (NAM's approach)
    y_scan_average = np.mean(np.stack(y_scans), axis=0)
    triggered = np.where(np.abs(y_scan_average) > trigger_threshold)[0]

    if len(triggered) == 0:
        logger.warning("No blips detected in averaged scan")
        return []

    # Calculate blip positions from the averaged scan trigger
    detected_indices = []
    j = triggered[0]
    for i, blip_time in enumerate([config.blip1_time, config.blip2_time]):
        i_rel = blip_time - config.silence1_end
        delay = j - config.lookahead
        blip_index = config.silence1_end + i_rel + delay
        detected_indices.append(blip_index)

    return detected_indices


def _plot_debug(dry_scans: List[np.ndarray], proc_scans: List[np.ndarray], threshold: float, alignment_config: AlignmentConfig):
    """Plot debug information for blip detection with zoomed view"""
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Calculate time array centered around blip
    t = np.arange(-alignment_config.lookahead,
                  alignment_config.lookback) / alignment_config.sample_rate

    # Full view plot (left subplot)
    for dry_scan in dry_scans:
        ax1.plot(t, dry_scan, alpha=0.2, color='blue')
    for proc_scan in proc_scans:
        ax1.plot(t, proc_scan, alpha=0.2, color='orange')

    if dry_scans:
        avg_dry = np.mean(np.stack(dry_scans), axis=0)
        ax1.plot(t, avg_dry, color='blue',
                 label='Dry signal average', linewidth=2)
    if proc_scans:
        avg_proc = np.mean(np.stack(proc_scans), axis=0)
        ax1.plot(t, avg_proc, color='orange',
                 label='Processed signal average', linewidth=2)

    ax1.axhline(y=threshold, color='k', linestyle='--', label='Threshold')
    ax1.axhline(y=-threshold, color='k', linestyle='--')
    ax1.axvline(x=0, color='r', linestyle='--', label='Expected blip')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Full View')
    ax1.legend()
    ax1.grid(True)

    # Zoomed view plot (right subplot)
    zoom_range = 0.005  # ±5ms around blip
    zoom_mask = np.abs(t) <= zoom_range

    for dry_scan in dry_scans:
        ax2.plot(t[zoom_mask], dry_scan[zoom_mask], alpha=0.2, color='blue')
    for proc_scan in proc_scans:
        ax2.plot(t[zoom_mask], proc_scan[zoom_mask], alpha=0.2, color='orange')

    if dry_scans:
        ax2.plot(t[zoom_mask], avg_dry[zoom_mask], color='blue',
                 label='Dry signal average', linewidth=2)
    if proc_scans:
        ax2.plot(t[zoom_mask], avg_proc[zoom_mask], color='orange',
                 label='Processed signal average', linewidth=2)

    ax2.axhline(y=threshold, color='k', linestyle='--', label='Threshold')
    ax2.axhline(y=-threshold, color='k', linestyle='--')
    ax2.axvline(x=0, color='r', linestyle='--', label='Expected blip')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Zoomed View (±5ms)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def validate_silence_periods(signal: np.ndarray, config: AlignmentConfig) -> bool:
    """Validate silence periods in V3 format audio."""
    # Check first silence period (0:09-0:10)
    silence1_start = config.validation1_end
    silence1 = signal[silence1_start:config.silence1_end]
    silence1_level = np.max(np.abs(silence1))

    # Check second silence period (3:00.5-3:01)
    silence2_start = config.train_end
    silence2_end = config.silence2_end
    silence2 = signal[silence2_start:silence2_end]
    silence2_level = np.max(np.abs(silence2))

    # Define silence threshold (can be made configurable)
    SILENCE_THRESHOLD = 0.001

    if silence1_level > SILENCE_THRESHOLD or silence2_level > SILENCE_THRESHOLD:
        logger.warning(
            f"Silence validation failed: levels {silence1_level:.6f}, {silence2_level:.6f}")
        return False

    return True


def validate_length_tolerances(dry: np.ndarray, processed: np.ndarray,
                               config: AlignmentConfig) -> bool:
    """Validate length differences within configurable tolerances."""
    length_diff = abs(len(dry) - len(processed))
    max_diff_samples = int(0.1 * config.sample_rate)  # 100ms tolerance

    if length_diff > max_diff_samples:
        logger.warning(
            f"Length difference {length_diff/config.sample_rate:.3f}s exceeds tolerance")
        return False

    return True


def validate_gear_consistency(processed: np.ndarray, config: AudioConfig) -> bool:
    """
    Validate gear consistency using NAM's exact ESR validation approach.
    Compares ESR between validation segments and verifies blip consistency.

    Args:
        processed: The processed signal
        config: Audio configuration

    Returns:
        True if gear is consistent, False otherwise
    """
    # Extract validation segments
    validation1 = processed[:config.validation1_end]
    validation2 = processed[config.silence2_end:config.validation2_end]

    # Ensure equal lengths by truncating to shorter segment
    min_length = min(len(validation1), len(validation2))
    validation1 = validation1[:min_length]
    validation2 = validation2[:min_length]

    # Convert to torch tensors for ESR calculation
    val1_tensor = torch.from_numpy(validation1).unsqueeze(0)
    val2_tensor = torch.from_numpy(validation2).unsqueeze(0)

    # Calculate ESR between validation segments
    esr_replicate = esr(val2_tensor, val1_tensor).item()

    # Extract and compare blips
    blips = []
    for blip_time in [config.blip1_time, config.blip2_time]:
        window_start = blip_time - config.lookahead
        window_end = blip_time + config.lookback
        blip = processed[window_start:window_end]
        blips.append(torch.from_numpy(blip).unsqueeze(0))

    # Calculate ESR between blip pairs
    if len(blips) >= 2:
        blip_esr = esr(blips[1], blips[0]).item()
        if blip_esr > config.esr_threshold:
            logger.warning(
                f"Blip consistency check failed: ESR = {blip_esr:.6f}")
            return False

    if esr_replicate > config.esr_threshold:
        logger.warning(
            f"Validation segment consistency check failed: ESR = {esr_replicate:.6f}")
        return False

    return True
