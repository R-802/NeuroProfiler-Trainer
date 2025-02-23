import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InvertibleConv1x1(nn.Module):
    """Invertible 1x1 Convolution Layer from Glow."""

    def __init__(self, channels):
        super().__init__()
        W = torch.linalg.qr(torch.randn(channels, channels))[
            0]  # Random orthogonal matrix
        self.W = nn.Parameter(W)

    def forward(self, x):
        return F.conv1d(x, self.W.unsqueeze(-1))

    def inverse(self, x):
        return F.conv1d(x, self.W.inverse().unsqueeze(-1))


class AffineCouplingLayer(nn.Module):
    """Affine Coupling Layer used in WaveGlow."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels // 2, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Splitting input tensor into two equal parts
        x_a, x_b = x.chunk(2, dim=1)
        net_out = self.net(x_a)
        # Ensure log_s and t have the same shape as x_b
        log_s, t = net_out.chunk(2, dim=1)
        s = torch.exp(log_s)  # Convert log-scale to multiplicative scale
        x_b = s * x_b + t  # Affine transformation
        return torch.cat([x_a, x_b], dim=1)

    def inverse(self, x):
        x_a, x_b = x.chunk(2, dim=1)
        log_s, t = self.net(x_a).chunk(2, dim=1)
        s = torch.exp(-log_s)
        x_b = (x_b - t) * s
        return torch.cat([x_a, x_b], dim=1)


class CompressedWaveGlow(nn.Module):
    """Compressed WaveGlow Model with parameter sharing."""

    def __init__(self, channels, num_layers):
        super().__init__()
        self.coupling_layer = AffineCouplingLayer(
            channels)  # Shared across layers
        self.convs = nn.ModuleList(
            [InvertibleConv1x1(channels) for _ in range(num_layers)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
            x = self.coupling_layer(x)
        return x

    def inverse(self, x):
        for conv in reversed(self.convs):
            x = self.coupling_layer.inverse(x)
            x = conv.inverse(x)
        return x


class WaveNetPostFilter(nn.Module):
    """WaveNet-based post-filter to improve waveform quality."""

    def __init__(self, channels, num_layers=7, dilation_base=2):
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=3,
                      dilation=dilation_base ** i, padding=dilation_base ** i)
            for i in range(num_layers)
        ])
        self.final_conv = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        for conv in self.dilated_convs:
            x = F.relu(conv(x))
        return self.final_conv(x)


class WGWaveNet(nn.Module):
    """WaveNet-based post-filter to improve waveform quality."""

    def __init__(self, channels: int = 64, num_layers: int = 4, dilation_base: int = 2):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.dilation_base = dilation_base

        # Input projection to match channel dimensions
        self.input_conv = nn.Conv1d(1, channels, kernel_size=1)

        # Dilated convolutions
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=3,
                      dilation=dilation_base ** i, padding=dilation_base ** i)
            for i in range(num_layers)
        ])

        # Output projection back to 1 channel
        self.final_conv = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        # Input shape: [batch_size, time_steps] or [batch_size, time_steps, 1]
        if x.dim() == 3:
            x = x.squeeze(-1)  # Remove channel dimension if present

        # Add channel dimension and transpose: [batch_size, 1, time_steps]
        x = x.unsqueeze(1)

        # Project to correct channel dimension
        x = self.input_conv(x)  # [batch_size, channels, time_steps]

        # Apply dilated convolutions
        skip_connections = []
        for conv in self.dilated_convs:
            residual = x
            x = F.relu(conv(x))
            x = x + residual  # Add residual connection
            skip_connections.append(x)

        # Combine skip connections
        x = torch.stack(skip_connections).sum(0)

        # Final projection
        x = self.final_conv(x)  # [batch_size, 1, time_steps]

        # Remove channel dimension and add back if needed
        x = x.squeeze(1).unsqueeze(-1)  # [batch_size, time_steps, 1]

        return x, None  # Return None as second value to match LSTM interface

    @property
    def receptive_field(self) -> int:
        """Calculate the model's receptive field."""
        return 2 ** (self.num_layers - 1) * 3  # kernel_size is fixed at 3


# Test the model with dummy data
if __name__ == "__main__":
    model = WGWaveNet()
    dummy_input = torch.randn(1, 22050)  # [batch_size, time_steps]
    output, _ = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
