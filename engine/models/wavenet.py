import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Sequence, Dict
import numpy as np


class Conv1d(nn.Conv1d):
    """Custom Conv1d with weight export/import for better weight management."""

    def export_weights(self) -> torch.Tensor:
        tensors = []
        if self.weight is not None:
            tensors.append(self.weight.data.flatten())
        if self.bias is not None:
            tensors.append(self.bias.data.flatten())
        return torch.cat(tensors) if tensors else torch.zeros(0)

    def import_weights(self, weights: torch.Tensor, start_idx: int) -> int:
        if self.weight is not None:
            n = self.weight.numel()
            self.weight.data = weights[start_idx:start_idx +
                                       n].reshape(self.weight.shape).to(self.weight.device)
            start_idx += n
        if self.bias is not None:
            n = self.bias.numel()
            self.bias.data = weights[start_idx:start_idx +
                                     n].reshape(self.bias.shape).to(self.bias.device)
            start_idx += n
        return start_idx


class WaveNetLayer(nn.Module):
    def __init__(self, input_channels, residual_channels, skip_channels, kernel_size, dilation, use_gated_activation=True):
        """
        A single layer of WaveNet with optional gated activation.

        Parameters:
        - input_channels: Number of input channels.
        - residual_channels: Number of residual channels.
        - skip_channels: Number of skip connection channels.
        - kernel_size: Size of the convolution kernel.
        - dilation: Dilation factor for the convolution.
        - use_gated_activation: Whether to use gated activation.
        """
        super().__init__()
        self.use_gated_activation = use_gated_activation

        # Calculate padding to ensure output has the same length as input
        padding = (kernel_size - 1) * dilation

        self.dilated_conv = Conv1d(
            residual_channels,
            residual_channels * (2 if use_gated_activation else 1),
            kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.residual_conv = Conv1d(
            residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = Conv1d(
            residual_channels, skip_channels, kernel_size=1)

    def forward(self, x, skip_connection):
        """
        Forward pass for a single layer.

        Parameters:
        - x: Input tensor of shape (batch_size, residual_channels, seq_len).
        - skip_connection: Accumulated skip connection tensor.

        Returns:
        - Updated residual and skip tensors.
        """
        # Ensure x has the correct dimensions
        if x.dim() != 3:
            raise ValueError(
                f"Expected input to have 3 dimensions (batch_size, channels, seq_len), but got {x.shape}"
            )

        # Apply dilated convolution
        conv_output = self.dilated_conv(x)

        # Ensure output and input sequence lengths match
        if conv_output.size(2) > x.size(2):
            conv_output = conv_output[:, :, :x.size(2)]

        if self.use_gated_activation:
            t, s = torch.chunk(conv_output, 2, dim=1)
            h = torch.tanh(t) * torch.sigmoid(s)
        else:
            h = torch.tanh(conv_output)

        skip_out = self.skip_conv(h)
        residual_out = self.residual_conv(h) + x

        if skip_connection is None:
            skip_connection = skip_out
        else:
            skip_connection = skip_connection[:,
                                              :, -skip_out.shape[2]:] + skip_out

        return residual_out, skip_connection


class WaveNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        residual_channels: int = 64,
        skip_channels: int = 128,
        dilation_channels: int = 64,
        kernel_size: int = 2,
        num_layers: int = 10,
        num_stacks: int = 2,
        output_channels: int = 1,
        use_gated_activation: bool = True,
    ):
        """
        A modular WaveNet implementation.

        Parameters:
        - input_channels: Number of input channels.
        - residual_channels: Number of residual channels.
        - skip_channels: Number of skip connection channels.
        - dilation_channels: Number of dilation channels.
        - kernel_size: Kernel size for convolutions.
        - num_layers: Number of layers per stack.
        - num_stacks: Number of repeated stacks.
        - output_channels: Number of output channels.
        - use_gated_activation: Whether to use gated activations.
        """
        super().__init__()
        self.input_conv = Conv1d(
            input_channels, residual_channels, kernel_size=1)

        self.layers = nn.ModuleList()
        dilations = [2 ** i for i in range(num_layers)]
        for _ in range(num_stacks):
            for dilation in dilations:
                self.layers.append(
                    WaveNetLayer(
                        input_channels,
                        residual_channels,
                        skip_channels,
                        kernel_size,
                        dilation,
                        use_gated_activation=use_gated_activation
                    )
                )

        self.output_conv1 = Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.output_conv2 = Conv1d(
            skip_channels, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_conv(x)
        skip_connection = None

        for i, layer in enumerate(self.layers):
            x, skip_connection = layer(x, skip_connection)

        x = torch.relu(self.output_conv1(skip_connection))
        x = self.output_conv2(x)
        return x
