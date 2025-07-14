import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients


def explain_duet_channels(duet_model, input_data, device='cpu', nsamples=100):
    """
    Compute Integrated Gradients attribution for each channel in the DUET model.

    Args:
        duet_model: Trained DUET model instance
        input_data: DataFrame of shape (seq_len, n_channels) - input time series
        device: Device to run computation on ('cpu' or 'cuda')
        nsamples: Number of samples for attribution (not used in IG)

    Returns:
        attribution_matrix: Array of shape (horizon, n_channels) - attribution scores
    """
    # Set model to evaluation mode and move to device
    duet_model.model.eval()
    duet_model.model.to(device)

    # Prepare input tensor: (1, seq_len, n_channels)
    input_tensor = torch.tensor(input_data.values, dtype=torch.float32).unsqueeze(0).to(device)
    baseline = torch.zeros_like(input_tensor)

    # Create wrapper for model output
    class DUETModelWrapper(torch.nn.Module):
        def __init__(self, duet_model):
            super().__init__()
            self.duet_model = duet_model

        def forward(self, x):
            output, _ = self.duet_model.model(x)
            return output

    wrapped_model = DUETModelWrapper(duet_model).to(device).eval()

    # Get a sample output shape
    with torch.no_grad():
        sample_output = wrapped_model(input_tensor)

    output_shape = sample_output.shape  # e.g., (1, horizon) or (1, horizon, d)

    # Initialize IG
    ig = IntegratedGradients(wrapped_model)

    attr_steps = []
    horizon = output_shape[1]
    n_channels = input_data.shape[1]

    for t in range(horizon):
        if len(output_shape) == 2:
            target = t
        elif len(output_shape) == 3:
            target = (t, 0)  # Use the first output dimension if multi-dimensional
        else:
            raise ValueError(f"Unexpected model output shape: {output_shape}")

        attributions = ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target
        )
        attributions = attributions.squeeze(0).cpu().numpy()  # (seq_len, n_channels)
        step_attr = np.mean(attributions, axis=0)  # Mean across time steps
        attr_steps.append(step_attr)

    attribution_matrix = np.stack(attr_steps, axis=0)  # (horizon, n_channels)
    return attribution_matrix


def plot_attribution_summary(attribution_values, input_data, save_path=None):
    """
    Plot Integrated Gradients attribution summary for channel contributions.

    Args:
        attribution_values: Array of attribution scores (horizon, n_channels)
        input_data: DataFrame containing the input data with column names
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    channel_names = input_data.columns
    mean_attributions = np.mean(attribution_values, axis=0)

    bars = plt.bar(range(len(channel_names)), mean_attributions,
                   color='skyblue', edgecolor='navy', alpha=0.7)

    plt.xlabel('Channels', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Attribution Score', fontsize=12, fontweight='bold')
    plt.title('Channel Attribution Analysis (Integrated Gradients)',
              fontsize=14, fontweight='bold', pad=20)

    for bar, value in zip(bars, mean_attributions):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(range(len(channel_names)), channel_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Attribution plot saved to: {save_path}")

    plt.close()


def channel_correlation(predictions, columns=None):
    """
    Compute correlation matrix between channels in the predictions.

    Args:
        predictions: Array of shape (horizon, n_channels) - model predictions
        columns: Optional list of column names for the DataFrame

    Returns:
        corr_matrix: DataFrame correlation matrix between channels
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    df = pd.DataFrame(predictions, columns=columns)
    return df.corr()


def plot_correlation_heatmap(corr_matrix, save_path=None):
    """
    Plot correlation heatmap between channels.

    Args:
        corr_matrix: DataFrame correlation matrix
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})

    plt.title('Channel Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Correlation heatmap saved to: {save_path}")

    plt.close()
