"""Audio processing module for handling audio data operations."""

# Standard library imports
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TypeVar

# Third-party imports
import numpy as np
import torch
import wavio
from torch.utils.data import Dataset

# Local imports
from engine.audio.alignment import (
    AlignmentConfig,
    align_signals,
    validate_alignment
)
from engine.config.constants import SAMPLE_RATE
from engine.config.training_config import AudioConfig
from engine.models.loss import esr
from engine.utils import logger

# Type aliases
AudioTensor = TypeVar('AudioTensor', torch.Tensor, np.ndarray)

# Constants
ONE_GB = 1024 * 1024 * 1024
V3_DATA_INFO = {
    'validation1_end': int(9 * SAMPLE_RATE),      # 0:09
    'silence1_end': int(10 * SAMPLE_RATE),        # 0:10
    'blips_end': int(12 * SAMPLE_RATE),          # 0:12
    'chirps_end': int(15 * SAMPLE_RATE),         # 0:15
    'noise_end': int(17 * SAMPLE_RATE),          # 0:17
    'train_start': int(17 * SAMPLE_RATE),        # 0:17
    'train_end': int(180.5 * SAMPLE_RATE),       # 3:00.5
    'silence2_end': int(181 * SAMPLE_RATE),      # 3:01
    'validation2_end': int(189 * SAMPLE_RATE),   # 3:10
}


class V3Dataset(Dataset):
    """Dataset implementation following NAM's V3 format requirements."""

    def __init__(self, dry: torch.Tensor, proc: torch.Tensor, config: AudioConfig,
                 segment_length: int, receptive_field: int):
        """
        Initialize dataset with tensors.

        Args:
            dry: Input signal tensor
            proc: Processed signal tensor
            config: Audio configuration
            segment_length: Length of each segment
            receptive_field: Model's receptive field size
        """
        # Ensure inputs are tensors
        self.dry = torch.as_tensor(dry, dtype=torch.float32)
        self.proc = torch.as_tensor(proc, dtype=torch.float32)
        self.config = config
        self.segment_length = segment_length
        self.receptive_field = receptive_field

        # Calculate effective length considering receptive field
        self.total_length = self.dry.shape[0]
        effective_length = self.total_length - self.receptive_field

        # Calculate segments with 50% overlap (NAM's approach)
        segment_stride = self.segment_length // 2
        self.num_segments = max(
            0, (effective_length - self.segment_length) // segment_stride + 1)

        self._validate_inputs()

    def _validate_inputs(self):
        if self.dry.shape != self.proc.shape:
            raise ValueError(
                f"Signal shapes must match: dry {self.dry.shape} != processed {self.proc.shape}")
        if self.dry.shape[0] < self.segment_length + self.receptive_field:
            raise ValueError("Signal too short for required segment length")

    def __len__(self) -> int:
        return self.num_segments

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate segment boundaries with NAM's 50% overlap
        segment_stride = self.segment_length // 2
        segment_start = idx * segment_stride

        # Calculate input boundaries including receptive field
        input_start = max(0, segment_start - self.receptive_field)
        input_end = min(self.dry.shape[0], segment_start + self.segment_length)
        target_end = min(self.proc.shape[0],
                         segment_start + self.segment_length)

        # Get input with context and target
        x = self.dry[input_start:input_end]
        y = self.proc[segment_start:target_end]

        # Ensure consistent channel dimension
        x = x.view(-1, 1) if x.dim() == 1 else x
        y = y.view(-1, 1) if y.dim() == 1 else y

        # Pad input if needed at the start
        if input_start == 0 and segment_start > 0:
            x = torch.nn.functional.pad(
                x, (0, 0, segment_start, 0))  # Pad time dimension

        # Pad input if needed at the end
        if x.shape[0] < self.segment_length + self.receptive_field:
            pad_size = self.segment_length + self.receptive_field - x.shape[0]
            x = torch.nn.functional.pad(
                x, (0, 0, 0, pad_size))  # Pad time dimension

        # Pad target if needed
        if y.shape[0] < self.segment_length:
            pad_size = self.segment_length - y.shape[0]
            y = torch.nn.functional.pad(
                y, (0, 0, 0, pad_size))  # Pad time dimension

        # Ensure exact sizes
        # [time, channels]
        x = x[:(self.segment_length + self.receptive_field), :]
        y = y[:self.segment_length, :]  # [time, channels]

        # Add assertions to catch any inconsistencies
        assert x.shape == (self.segment_length + self.receptive_field, 1), \
            f"Input shape mismatch: got {x.shape}, expected {(self.segment_length + self.receptive_field, 1)}"
        assert y.shape == (self.segment_length, 1), \
            f"Target shape mismatch: got {y.shape}, expected {(self.segment_length, 1)}"

        return x, y


@dataclass
class WavInfo:
    """WAV file information container."""
    sampwidth: int
    rate: int


class AudioError(Exception):
    """Base class for audio-related exceptions."""
    pass


class AudioShapeMismatchError(AudioError):
    """Raised when audio shapes don't match."""

    def __init__(self, expected_shape: tuple, actual_shape: tuple, message: str):
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        super().__init__(message)


class AudioLoadError(AudioError):
    """Raised when there are issues loading audio files."""
    pass


class AudioProcessor:
    """Handles audio loading, preprocessing, and alignment following NAM's approach."""

    def __init__(self, alignment_config: AlignmentConfig):
        """Initialize AudioProcessor with default alignment configuration."""
        self.alignment_config = alignment_config

    @staticmethod
    def _check_file_size(filepath: Union[str, Path]) -> None:
        """
        Check if file size is within acceptable limits.

        Args:
            filepath: Path to the audio file

        Raises:
            AudioLoadError: If file size exceeds MAX_FILE_SIZE
        """
        file_size = os.path.getsize(filepath)
        if file_size > ONE_GB:
            raise AudioLoadError(
                f"File too large: {file_size/1024/1024:.1f}MB exceeds {ONE_GB/1024/1024:.1f}MB limit"
            )

    @staticmethod
    def wav_to_np(
        filename: Union[str, Path],
        info: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, WavInfo]]:
        """
        Load WAV file to numpy array following NAM's approach.

        Args:
            filename: Path to WAV file
            info: Whether to return WavInfo along with array

        Returns:
            Numpy array or tuple of (array, WavInfo)

        Raises:
            AudioLoadError: For various audio loading/validation issues
        """
        try:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"File not found: {filename}")

            AudioProcessor._check_file_size(filename)
            x_wav = wavio.read(str(filename))
            data = x_wav.data

            # Check if the loaded audio has zero channels
            if data.ndim == 2 and data.shape[1] == 0:
                raise AudioLoadError(
                    f"Loaded audio file '{filename}' has zero channels")

            # Convert to mono if needed
            if data.ndim == 2 and data.shape[1] > 1:
                logger.warning(f"Converting {data.shape[1]} channels to mono")
                data = np.mean(data, axis=1)  # TODO Stereo?

            # Verify sample rate
            if SAMPLE_RATE is not None and x_wav.rate != SAMPLE_RATE:
                raise AudioLoadError(
                    f"Expected sample rate of {SAMPLE_RATE}, but found {x_wav.rate}"
                )

            # NAM's normalization method
            arr = data / (2.0 ** (8 * x_wav.sampwidth - 1))

            if not np.isfinite(arr).all():
                raise AudioLoadError("Audio contains NaN or Inf values")

            return arr if not info else (arr, WavInfo(x_wav.sampwidth, x_wav.rate))

        except Exception as e:
            if isinstance(e, (FileNotFoundError, AudioLoadError)):
                raise
            raise AudioLoadError(f"Error loading {filename}: {str(e)}") from e

    @staticmethod
    def wav_to_tensor(
        filename: Union[str, Path],
        info: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, WavInfo]]:
        """
        Convert WAV file to PyTorch tensor.

        Args:
            filename: Path to WAV file
            info: Whether to return WavInfo

        Returns:
            Tensor or tuple of (tensor, WavInfo)
        """
        out = AudioProcessor.wav_to_np(filename, info=info)
        if info:
            arr, info = out
            return torch.from_numpy(arr).float(), info
        else:
            arr = out
            return torch.from_numpy(arr).float()

    @staticmethod
    def load_wav_pair(
        input_path: Union[str, Path],
        target_path: Union[str, Path],
        info: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, WavInfo]:
        """
        Load and validate a pair of input/target WAV files.

        Args:
            input_path: Path to input WAV file
            target_path: Path to target WAV file
            info: Whether to return WavInfo

        Returns:
            Tuple of (input_tensor, target_tensor, wav_info)

        Raises:
            AudioLoadError: For various loading/validation issues
        """
        try:
            print("\nTensor Conversion:")
            print("-" * 50)
            # Load input file first to get reference WAV info
            input_tensor, wav_info = AudioProcessor.wav_to_tensor(
                input_path, info=info)

            # Load target file
            target_tensor, target_info = AudioProcessor.wav_to_tensor(
                target_path, info=info)

            # Verify matching sample rates and lengths
            if wav_info.rate != target_info.rate:
                raise AudioLoadError(
                    f"Sample rate mismatch: {wav_info.rate} vs {target_info.rate}"
                )
            if len(input_tensor) != len(target_tensor):
                raise AudioLoadError(
                    f"Length mismatch: {len(input_tensor)} vs {len(target_tensor)}"
                )

            # Validate tensors
            AudioProcessor._validate_tensor_pair(input_tensor, target_tensor)

            logger.info(
                f"Loaded audio lengths - Input: {len(input_tensor)} samples ({len(input_tensor)/SAMPLE_RATE:.2f}s), Target: {len(target_tensor)} samples ({len(target_tensor)/SAMPLE_RATE:.2f}s)")

            return input_tensor, target_tensor, wav_info

        except Exception as e:
            if isinstance(e, (FileNotFoundError, AudioLoadError)):
                raise
            raise AudioLoadError(f"Error loading audio pair: {str(e)}") from e

    @staticmethod
    def _validate_tensor_pair(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> None:
        """
        Validate a pair of audio tensors.
        """
        logger.info(f"Input shape: {input_tensor.shape}")
        logger.info(f"Target shape: {target_tensor.shape}")

        if input_tensor.numel() == 0 or target_tensor.numel() == 0:
            raise AudioLoadError("Empty audio data")

        if torch.isnan(input_tensor).any() or torch.isnan(target_tensor).any():
            raise AudioLoadError("Audio contains NaN values")

        if torch.isinf(input_tensor).any() or torch.isinf(target_tensor).any():
            raise AudioLoadError("Audio contains Inf values")

    @staticmethod
    def validate_structure(dry: torch.Tensor, proc: torch.Tensor) -> None:
        """
        Validate and log the V3 data structure timings.

        Args:
            dry: Dry signal tensor
            proc: Processed signal tensor

        Raises:
            ValueError: If data doesn't match V3 structure requirements
        """
        # Add debug prints at the start
        print(f"\nValidating V3 structure:")
        print("-" * 50)
        logger.info(f"Dry tensor shape: {dry.shape}, dtype: {dry.dtype}")
        logger.info(
            f"Processed tensor shape: {proc.shape}, dtype: {proc.dtype}")

        # Check for empty tensors
        if dry.numel() == 0 or proc.numel() == 0:
            raise ValueError("Empty tensor detected during V3 validation")

        # Get total length and ensure it's a 1D tensor
        if dry.dim() > 1:
            logger.warning(
                f"Input tensor has {dry.dim()} dimensions, flattening to 1D")
            dry = dry.flatten()
            proc = proc.flatten()

        total_length = dry.size(-1)
        info = V3_DATA_INFO

        # Validate total length
        if total_length < info['validation2_end']:
            raise ValueError(
                f"Data too short for V3 format. Expected {info['validation2_end']/SAMPLE_RATE:.1f}s "
                f"but got {total_length/SAMPLE_RATE:.1f}s"
            )

        # Log segment information
        print("\nV3 Data Structure Information:")
        print("-" * 50)
        logger.info("Validation 1 (0:00-0:09):  %6.2fs (%d samples)",
                    info['validation1_end']/SAMPLE_RATE, info['validation1_end'])
        logger.info("Silence 1 (0:09-0:10):     %6.2fs (%d samples)",
                    (info['silence1_end'] - info['validation1_end'])/SAMPLE_RATE,
                    info['silence1_end'] - info['validation1_end'])
        logger.info("Blips (0:10-0:12):         %6.2fs (%d samples)",
                    (info['blips_end'] - info['silence1_end'])/SAMPLE_RATE,
                    info['blips_end'] - info['silence1_end'])
        logger.info("Training (0:17-3:00.5):    %6.2fs (%d samples)",
                    (info['train_end'] - info['train_start'])/SAMPLE_RATE,
                    info['train_end'] - info['train_start'])
        logger.info("Validation 2 (3:01-3:10):  %6.2fs (%d samples)",
                    (info['validation2_end'] -
                     info['silence2_end'])/SAMPLE_RATE,
                    info['validation2_end'] - info['silence2_end'])
        logger.info("-" * 50)
        logger.info("Total Length:              %6.2fs (%d samples)",
                    total_length/SAMPLE_RATE, total_length)

    @staticmethod
    def slice_v3_waves(dry: torch.Tensor, proc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Slice aligned waves according to NAM V3 specification.

        Training data: From 0:17 to 3:00.5
        Validation data: From 0:00 to 0:09 + From 3:01 to 3:10

        Returns:
            Tuple of (dry_train, proc_train, dry_val, proc_val)
        """
        info = V3_DATA_INFO

        # Extract training data (0:17-3:00.5)
        dry_train = dry[info['train_start']:info['train_end']]
        proc_train = proc[info['train_start']:info['train_end']]

        # Extract validation data (0:00-0:09 + 3:01-3:10)
        dry_val = torch.cat([
            dry[:info['validation1_end']],
            dry[info['silence2_end']:info['validation2_end']]
        ])

        proc_val = torch.cat([
            proc[:info['validation1_end']],
            proc[info['silence2_end']:info['validation2_end']]
        ])

        return dry_train, proc_train, dry_val, proc_val

    def align_signals(
        self,
        dry: torch.Tensor,
        proc: torch.Tensor,
        alignment_config: AlignmentConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align signals using NAM's approach.

        Args:
            dry: Input signal as torch tensor
            proc: Processed signal as torch tensor
            alignment_config: Configuration for alignment

        Returns:
            Tuple of (aligned_dry, aligned_proc) as torch tensors

        Raises:
            AudioError: If alignment fails
        """
        try:
            print(f"\nAlignment:")
            print("-" * 50)
            logger.info(
                f"Pre-alignment lengths - Dry: {len(dry)} samples, Proc: {len(proc)} samples")

            # Convert tensors to numpy arrays and perform alignment
            aligned_dry_np, aligned_proc_np = align_signals(
                dry.cpu().numpy(), proc.cpu().numpy(), alignment_config
            )

            # Convert back to torch tensors
            return torch.from_numpy(aligned_dry_np).float(), torch.from_numpy(aligned_proc_np).float()
        except Exception as e:
            raise AudioError(f"Signal alignment failed: {str(e)}") from e

    def validate_alignment(
        self,
        dry: torch.Tensor,
        proc: torch.Tensor,
        alignment_config: AlignmentConfig,
        plot_debug: bool = False
    ) -> bool:
        """
        Validate alignment between two signals.

        Args:
            dry: Input signal tensor
            proc: Processed signal tensor
            alignment_config: Alignment configuration
            plot_debug: Whether to plot blip information

        Returns:
            bool: True if alignment is valid
        """
        dry_np = dry.cpu().numpy()
        proc_np = proc.cpu().numpy()
        return validate_alignment(dry_np, proc_np, alignment_config, plot_debug)
