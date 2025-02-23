import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LSTMProfiler(nn.Module):
    def __init__(
        self,
        input_dim=1,
        lstm_hidden=32,
        conv_filters=None,
        conv_kernel=None,
        conv_stride=None,
        dropout=0.0,
        train_burn_in=None,
        train_truncate=None,
        block_size=65535,  # Fixed block size as per reference
        num_layers=1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.train_burn_in = train_burn_in
        self.train_truncate = train_truncate
        self.block_size = block_size
        self.num_layers = num_layers

        # Determine if we should use convolutional preprocessing
        self.use_conv = all(x is not None for x in [
                            conv_filters, conv_kernel, conv_stride])

        if self.use_conv:
            # Convolutional preprocessing layer
            padding = (conv_kernel - 1) // 2
            self.conv = nn.Conv1d(
                input_dim, conv_filters,
                kernel_size=conv_kernel, stride=conv_stride,
                padding=padding, padding_mode='replicate'
            )
            self.bn = nn.BatchNorm1d(conv_filters)
            lstm_input_size = conv_filters
            self._conv_kernel = conv_kernel
            self._conv_stride = conv_stride
        else:
            self.conv = None
            self.bn = None
            lstm_input_size = input_dim
            self._conv_kernel = 1
            self._conv_stride = 1

        # Store for receptive field calculation
        self._lstm_input_size = lstm_input_size

        # LSTM core with multiple layers and improved forget gate initialization
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Optimize forget gate initialization (similar to reference)
        for layer in range(num_layers):
            # Get layer parameter names based on LSTM implementation
            bias_ih = getattr(self.lstm, f'bias_ih_l{layer}')
            bias_hh = getattr(self.lstm, f'bias_hh_l{layer}')

            # Modify input and forget gate biases
            # Format is (input_gate, forget_gate, cell_gate, output_gate)
            value = 2.0
            for bias in [bias_ih, bias_hh]:
                input_size = lstm_hidden
                # Balance input and forget gates
                bias.data[0:input_size] -= value  # input gate
                bias.data[input_size:2*input_size] += value  # forget gate

        # Initial hidden and cell states for each layer (learnable parameters)
        self._initial_hidden = nn.Parameter(
            torch.zeros((num_layers, 1, lstm_hidden))
        )
        self._initial_cell = nn.Parameter(
            torch.zeros((num_layers, 1, lstm_hidden))
        )

        # Fully connected output layer
        self.output_fc = nn.Linear(lstm_hidden, 1)

    @property
    def receptive_field(self) -> int:
        """
        Calculate the receptive field of the model.
        For LSTM-based models, this is technically 1 as it can use all past information.
        The convolutional layer's receptive field is only relevant for preprocessing.
        """
        # LSTM can theoretically use all past information
        return 1

    def _initial_state(self, batch_size: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state with learnable parameters"""
        if batch_size is None or batch_size == 1:
            return (self._initial_hidden, self._initial_cell)
        return (
            torch.tile(self._initial_hidden, (1, batch_size, 1)),
            torch.tile(self._initial_cell, (1, batch_size, 1))
        )

    def reset_hidden(self, hidden_state: Optional[tuple[torch.Tensor, torch.Tensor]],
                     batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset hidden state between sequences while preserving some memory"""
        if hidden_state is None:
            return self._initial_state(batch_size)
        h, c = hidden_state
        # Partially reset states (maintain some memory)
        mask = torch.bernoulli(torch.ones_like(h) * 0.1)  # 10% retention
        h = h * mask
        c = c * mask
        return (h, c)

    def _process_in_blocks(self, x: torch.Tensor, hidden_state=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Process long sequences in blocks to avoid memory issues."""
        outputs = []
        curr_state = hidden_state if hidden_state is not None else self._initial_state(
            x.size(0))

        for i in range(0, x.size(1), self.block_size):
            block = x[:, i:i + self.block_size, :]
            out, curr_state = self.lstm(block, curr_state)
            outputs.append(out)

        return torch.cat(outputs, dim=1), curr_state

    def forward(self, x: torch.Tensor, hidden_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the LSTM model.

        Args:
            x: Input tensor of shape [batch_size, seq_len] or [batch_size, seq_len, input_dim]
            hidden_state: Optional tuple of (hidden_state, cell_state) from previous step

        Returns:
            tuple: (output, hidden_state)
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Add channel dimension if not present
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # [batch, seq_len, 1]

        if self.use_conv:
            # Now safe to permute since we ensured 3D input
            x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
            x = F.relu(self.bn(self.conv(x)))
            x = x.permute(0, 2, 1)  # [batch, seq_len, conv_filters]

        # Process through LSTM with provided hidden state or initial state
        if hidden_state is None:
            hidden_state = self._initial_state(batch_size)

        output, hidden_state = self._process_in_blocks(x, hidden_state)

        # Final output layer
        output = self.output_fc(output)

        return output, hidden_state

    def process_sample(self, x: torch.Tensor, state: Optional[tuple[torch.Tensor, torch.Tensor]] = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Process a single audio sample with state management."""
        with torch.no_grad():
            return self.forward(x, state)
