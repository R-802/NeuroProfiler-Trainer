import torch
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from engine.config.constants import SAMPLE_RATE
from engine.config.training_config import TrainingConfig
from engine.audio.alignment import AlignmentConfig
from engine.audio.processor import V3Dataset, AudioProcessor, V3_DATA_INFO
from engine.models.loss import esr
import soundfile as sf
from tqdm import tqdm
import logging
import pyloudnorm as pyln
import webbrowser
import os

# Suppress verbose logging from Pillow
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("PIL.ImageFile").setLevel(logging.ERROR)

# Suppress verbose logging from matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def load_config():
    with open('engine/config/config.json', 'r') as f:
        return json.load(f)


def evaluate_model(model_path: str, dry: torch.Tensor, proc: torch.Tensor):
    # Load configuration
    config = load_config()

    # Print the configuration to the terminal
    print("\nModel Configuration:")
    print("-"*50)
    print(json.dumps(config['model'], indent=4))

    model_name = config['model'].get('name', 'LSTMProfiler').upper()

    if model_name == "LSTMPROFILER":
        from engine.models.lstm import LSTMProfiler
        model = LSTMProfiler(
            lstm_hidden=config['model']['config'].get('lstm_hidden', 32),
            num_layers=config['model']['config'].get('num_layers', 1),
            train_burn_in=config['model']['config'].get('train_burn_in', 0)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Ensure the model is on the right device and in evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Increase efficiency on CUDA
    torch.backends.cudnn.benchmark = True

    # Ensure tensors are 1D (remove extra dimensions)
    if dry.dim() > 1:
        dry = dry.squeeze()
    if proc.dim() > 1:
        proc = proc.squeeze()

    # Create dataset and DataLoader
    segment_length = config['data']['segment_length']
    dataset = V3Dataset(
        dry=dry,
        proc=proc,
        config=config,
        segment_length=segment_length,
        receptive_field=model.receptive_field
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=True
    )

    # Initialise output and target buffers for reconstruction
    output_buffer = torch.zeros_like(dry)
    target_buffer = torch.zeros_like(dry)
    overlap_counts = torch.zeros_like(dry)
    current_position = 0

    burn_in = config['model']['config'].get('train_burn_in', 0)

    with torch.inference_mode(), torch.amp.autocast(device_type=device.type):
        hidden_state = None
        prev_batch_size = None

        print("\nInference:")
        print("-"*50)
        for x, y in tqdm(dataloader, desc="Progress", total=len(dataloader)):
            x, y = x.to(device, non_blocking=True), y.to(
                device, non_blocking=True)
            y = y.squeeze(-1)

            # Reset hidden state if the batch size changes
            batch_size = x.size(0)
            if prev_batch_size != batch_size:
                hidden_state = None
                prev_batch_size = batch_size

            # Forward pass
            output, hidden_state = model(x, hidden_state)
            output = output.squeeze(-1)

            # Handle burn-in period
            if burn_in:
                output = output[:, burn_in:]
                y = y[:, burn_in:]

            # Ensure output and target have the same length
            min_length = min(output.shape[1], y.shape[1])
            output = output[:, :min_length]
            y = y[:, :min_length]

            curr_length = output.shape[1]

            for b in range(batch_size):
                pos = current_position + (b * segment_length // 2)
                if pos >= len(dry):
                    break

                # Store both output and target
                output_buffer[pos:pos + curr_length] += output[b].cpu()
                target_buffer[pos:pos + curr_length] += y[b].cpu()
                overlap_counts[pos:pos + curr_length] += 1

            current_position += (batch_size * segment_length // 2)
            if current_position >= len(dry):
                break

    # Average overlapped regions
    mask = overlap_counts > 0
    output_buffer[mask] /= overlap_counts[mask]
    target_buffer[mask] /= overlap_counts[mask]

    # Calculate ESR on the full signal
    output_buffer = output_buffer.unsqueeze(0)
    target_buffer = target_buffer.unsqueeze(0)
    esr_value = esr(output_buffer, target_buffer)
    print(f"Model ESR: {esr_value:.6f}")

    # Convert to numpy for plotting
    output_np = output_buffer.squeeze(0).numpy()
    target_np = target_buffer.squeeze(0).numpy()

    # Save predicted waveform as WAV file
    sf.write('data/predicted.wav', output_np, samplerate=int(SAMPLE_RATE))
    print("Saved predicted waveform to data/predicted.wav")

    # Decimate data for the overview plot (reduce number of points)
    def decimate_signal(signal, target_points=50000):
        if len(signal) > target_points:
            stride = len(signal) // target_points
            return signal[::stride]
        return signal

    # Create decimated versions for the overview
    output_decimated = decimate_signal(output_np)
    target_decimated = decimate_signal(target_np)
    x_decimated = np.arange(0, len(output_np), len(
        output_np)//len(output_decimated))

    # Create an interactive plot with both full view and zoomed view using plotly
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Model Prediction vs Target (Full View, ESR: {esr_value:.6f})',
            f'Zoomed View (Samples 40000 to 50000)'
        ),
        vertical_spacing=0.15
    )

    # Full view plot (decimated)
    fig.add_trace(
        go.Scattergl(
            x=x_decimated,
            y=target_decimated,
            name='Target',
            opacity=0.7,
            line=dict()
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scattergl(
            x=x_decimated,
            y=output_decimated,
            name='Predicted',
            opacity=0.7,
            line=dict()
        ),
        row=1, col=1
    )

    # Zoomed view plot
    zoom_start = 40000
    zoom_end = 45000
    x_zoom = np.arange(zoom_start, zoom_end)

    fig.add_trace(
        go.Scattergl(
            x=x_zoom,
            y=target_np[zoom_start:zoom_end],
            name='Target',
            opacity=0.7,
            showlegend=False,
            line=dict()
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scattergl(
            x=x_zoom,
            y=output_np[zoom_start:zoom_end],
            name='Predicted',
            opacity=0.7,
            showlegend=False,
            line=dict()
        ),
        row=2, col=1
    )

    # Update layout with optimized settings
    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',  # More efficient hover mode
        uirevision=True,  # Preserve UI state on updates
        margin=dict(l=100, r=100, t=100, b=100),  # Reduce margins
        autosize=True,  # Enable auto-sizing
    )

    # Update axes labels and configure for better performance
    for row in [1, 2]:
        fig.update_xaxes(
            title_text='Sample',
            row=row,
            col=1,
            # Disable rangeslider for better performance
            rangeslider=dict(visible=False),
            spikesnap='cursor',  # Improve hover performance
        )
        fig.update_yaxes(
            title_text='Amplitude',
            row=row,
            col=1,
            spikesnap='cursor',  # Improve hover performance
        )

    # Configure the plot for better performance and responsiveness
    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        # Remove unused buttons
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'responsive': True,
        'autosizable': True,
        'fillFrame': True  # Make the plot fill the frame
    }

    # Save the interactive plot with responsive HTML wrapper
    html_content = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                }}
                #plot-container {{
                    width: 100vw;
                    height: 100vh;
                    position: absolute;
                }}
            </style>
        </head>
        <body>
            <div id="plot-container">
                {fig.to_html(config=config, include_plotlyjs=True, full_html=False)}
            </div>
        </body>
    </html>
    """

    with open('data/evaluation_plot.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    plot_path = os.path.abspath('data/evaluation_plot.html')
    print("Saved interactive evaluation plot to data/evaluation_plot.html")
    print("Opening evaluation plot in browser...")
    webbrowser.open('file://' + plot_path)
    print(f"Audio length: {len(output_np)/SAMPLE_RATE:.2f} seconds")

    # Perform Null audio test (difference between target and predicted)
    null_signal = target_np - output_np

    # Save null test result as WAV
    sf.write('data/null_test.wav', null_signal, samplerate=int(SAMPLE_RATE))

    # Calculate LUFS of the null signal
    # Initialize loudness meter
    meter = pyln.Meter(int(SAMPLE_RATE))

    # Ensure the signal is in the correct range for LUFS measurement
    null_signal_normalized = null_signal / np.max(np.abs(null_signal))

    # Measure LUFS (Integrated)
    null_lufs = meter.integrated_loudness(null_signal_normalized)

    print("\nNull Test Results:")
    print("-"*50)
    print(f"Null Test LUFS: {null_lufs:.2f} LUFS")
    print("Saved null test audio to data/null_test.wav")

    return esr_value, null_lufs


if __name__ == "__main__":
    model_path = "checkpoints/best.pt"
    input_file = "data/input.wav"
    target_file = "data/target.wav"
    predicted_file = "data/predicted.wav"

    with open("engine/config/config.json", "r") as f:
        default_config = json.load(f)

    config = TrainingConfig.from_dict(default_config)
    alignment_config = AlignmentConfig(config.audio)
    processor = AudioProcessor(alignment_config)
    dry, proc, _ = processor.load_wav_pair(
        input_path=input_file,
        target_path=target_file
    )

    esr_val, null_lufs = evaluate_model(model_path, dry, proc)
