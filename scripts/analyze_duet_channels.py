import argparse
import os
import numpy as np
import pandas as pd
import torch
from ts_benchmark.baselines.duet.duet import DUET
from ts_benchmark.baselines.duet.utils import channel_analysis


def main(model_path, input_csv, output_dir, device='cpu', nsamples=100):
    """
    Analyze DUET model channel contributions and correlations.
    
    Args:
        model_path: Path to trained DUET model .pth file
        input_csv: Path to input CSV file (seq_len x n_channels)
        output_dir: Directory to save analysis results
        device: Device to use ('cpu' or 'cuda')
        nsamples: Number of samples for attribution (kept for compatibility)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input data
    print(f"Loading input data from: {input_csv}")
    input_data = pd.read_csv(input_csv, index_col=0)
    
    # Load model configuration and weights
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint['config']
    
    # Initialize and load DUET model
    duet = DUET(**config_dict)
    duet.model.load_state_dict(checkpoint['model_state_dict'])
    duet.model.to(device)
    duet.model.eval()
    
    # Run prediction to get model outputs
    print("Running model prediction...")
    with torch.no_grad():
        input_tensor = torch.tensor(input_data.values, dtype=torch.float32).unsqueeze(0).to(device)
        output, _ = duet.model(input_tensor)
        predictions = output.cpu().numpy().reshape(-1, input_data.shape[1])
    
    # Perform Integrated Gradients attribution analysis
    print("Computing channel attributions using Integrated Gradients...")
    attribution_vals = channel_analysis.explain_duet_channels(
        duet, input_data, device=device, nsamples=nsamples
    )
    
    # Save attribution results
    attribution_path = os.path.join(output_dir, 'channel_attributions.npy')
    np.save(attribution_path, attribution_vals)
    print(f"Attribution values saved to: {attribution_path}")
    
    # Generate and save attribution plot
    attribution_plot_path = os.path.join(output_dir, 'channel_attribution.png')
    channel_analysis.plot_attribution_summary(
        attribution_vals, input_data, save_path=attribution_plot_path
    )
    
    # Perform channel correlation analysis
    print("Computing channel correlations...")
    corr_matrix = channel_analysis.channel_correlation(
        predictions, columns=input_data.columns
    )
    
    # Save correlation results
    correlation_path = os.path.join(output_dir, 'channel_correlation.csv')
    corr_matrix.to_csv(correlation_path)
    print(f"Correlation matrix saved to: {correlation_path}")
    
    # Generate and save correlation heatmap
    correlation_plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
    channel_analysis.plot_correlation_heatmap(
        corr_matrix, save_path=correlation_plot_path
    )
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Files generated:")
    print(f"  - channel_attributions.npy: Raw attribution scores")
    print(f"  - channel_attribution.png: Attribution visualization")
    print(f"  - channel_correlation.csv: Correlation matrix")
    print(f"  - correlation_heatmap.png: Correlation visualization")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze DUET channel contributions and correlations using Integrated Gradients.'
    )
    parser.add_argument(
        '--model', type=str, required=True, 
        help='Path to trained DUET model .pth file'
    )
    parser.add_argument(
        '--input', type=str, required=True, 
        help='Path to input CSV file (seq_len x n_channels)'
    )
    parser.add_argument(
        '--output', type=str, required=True, 
        help='Directory to save analysis results'
    )
    parser.add_argument(
        '--device', type=str, default='cpu', 
        help='Device to use (cpu or cuda)'
    )
    parser.add_argument(
        '--nsamples', type=int, default=100, 
        help='Number of samples for attribution (kept for compatibility)'
    )
    
    args = parser.parse_args()
    main(args.model, args.input, args.output, device=args.device, nsamples=args.nsamples) 