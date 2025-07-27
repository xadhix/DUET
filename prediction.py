import pandas as pd
import numpy as np
import torch
import os
from ts_benchmark.baselines.duet.duet import DUET
from ts_benchmark.baselines.duet.utils.tools import EarlyStopping
from ts_benchmark.models.model_base import BatchMaker

# === Configuration ===
latest_model_path = "C:\\Duet\\DUET\\result\\duet_model_h96_20250726_231950.pth"  # UPDATE this path
test_data_path = "dataset/forecasting/ETTh2.csv"  # UPDATE this path
output_file = "duet_predictions.csv"
tail_rows = 1000  # How many latest rows to use from test data

def _create_duet_instance(horizon=96, seq_len=96):
    """
    Create and initialize a DUET instance.
    """
    duet_instance = DUET(
        seq_len=seq_len,
        horizon=horizon,
        norm=True,
        batch_size=256,
        num_epochs=100
    ) 
    duet_instance.early_stopping = EarlyStopping(patience=5)
    return duet_instance

def load_model_and_predict(model_path, test_data, horizon=96, seq_len=96):
    """
    Load DUET model and forecast using test data.
    """
    print(f"\n=== Loading Model and Making Predictions ===")
    duet = _create_duet_instance(horizon=horizon, seq_len=seq_len)

    if not duet.load_model(model_path):
        print("❌ Failed to load model")
        return None, None

    print("✓ Model loaded successfully")

    # Transform and forecast
    # Create batch maker from test_data
    batch_maker = BatchMaker.make_batch(
        series=test_data,
        horizon=horizon,
        seq_len=seq_len,
        step_size=horizon  # Slide by horizon to avoid overlap
    )

    # Use batch forecast
    predictions = duet.batch_forecast(horizon=horizon, batch_maker=batch_maker)

    if predictions is not None:
        print(f"✓ Predictions completed: shape = {predictions.shape}")
        print(f"Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        return predictions, duet
    else:
        print("❌ Forecasting returned None")
        return None, None

def save_predictions_to_csv(predictions, duet_model, original_data, output_file="predictions.csv"):
    """
    Save predictions to CSV with datetime index.
    """
    try:
        freq = duet_model.config.freq.upper()
        pred_df = pd.DataFrame(
            predictions,
            columns=original_data.columns,
            index=pd.date_range(
                start=original_data.index[-1],
                periods=len(predictions) + 1,
                freq=freq
            )[1:]
        )
        pred_df.to_csv(output_file)
        print(f"✓ Saved predictions to: {output_file}")
        return pred_df
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")
        return None

def display_prediction_results(pred_df, num_samples=5):
    """
    Display summary of predictions.
    """
    print("\n=== Prediction Results ===")
    print(f"Shape: {pred_df.shape}")
    print(f"\nFirst {num_samples} rows:\n", pred_df.head(num_samples))
    print(f"\nLast {num_samples} rows:\n", pred_df.tail(num_samples))
    print("\nDescriptive Stats:\n", pred_df.describe())

# === Main execution ===
if __name__ == "__main__":
    print("=" * 60)
    print("📦 DUET MODEL INFERENCE")
    print("=" * 60)

    # Step 1: Load test data
    print(f"Loading test data from: {test_data_path}")
    test_data = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
    test_data = test_data.tail(tail_rows)
    print(f"✓ Test data loaded: shape = {test_data.shape}")

    # Step 2: Load model metadata (seq_len, horizon, freq)
    if not os.path.exists(latest_model_path):
        raise FileNotFoundError(f"Model file not found: {latest_model_path}")
    checkpoint = torch.load(latest_model_path, map_location='cpu')
    seq_len = checkpoint.get("seq_len", 96)
    horizon = checkpoint.get("horizon", 96)

    # Step 3: Forecast
    predictions, duet_model = load_model_and_predict(
        model_path=latest_model_path,
        test_data=test_data,
        horizon=horizon,
        seq_len=seq_len
    )

    # Step 4: Save and display
    if predictions is not None and duet_model is not None:
        pred_df = save_predictions_to_csv(
            predictions=predictions,
            duet_model=duet_model,
            original_data=test_data,
            output_file=output_file
        )
        if pred_df is not None:
            display_prediction_results(pred_df)
            print("\n🎉 All done!")
        else:
            print("❌ Failed to save predictions")
    else:
        print("❌ Prediction failed")

    print("\n✅ Script complete.")
