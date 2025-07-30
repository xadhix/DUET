import numpy as np
import matplotlib.pyplot as plt

# Replace 'seriesname' with your actual series name
series_name = "ETTh1.csv"  # e.g., "ETTh1.csv"

# Load the data
# all_predicts = np.load(r'C:\Duet\DUET\all_predicts_ETTh2.csv_exclude_0_20250728_133126.npy')
# targets = np.load(r'C:\Duet\DUET\targets_ETTh2.csv_exclude_0_20250728_133126.npy')
scaler = np.load(r'C:\Duet\DUET\ETTh1.csv_scaler_20250730_165452.npy')

print("scaler mean:", scaler[0])
print("scaler var:", scaler[1])

# print(f"Loaded data for {series_name}")
# print(f"Predictions shape: {all_predicts.shape}")
# print(f"Targets shape: {targets.shape}")

# Access specific samples
# first_prediction = all_predicts[0]  # First prediction
# first_target = targets[0]          # First target

# # Calculate metrics (example)
# mse_overall = np.mean((all_predicts - targets) ** 2)
# print(f"Overall Mean Squared Error: {mse_overall}")

# # Calculate MSE for each channel separately
# num_channels = all_predicts.shape[2]  # Assuming shape is (samples, time_steps, channels)
# print(f"\nMSE for each channel (averaged across all samples and time steps):")
# mse_per_channel = np.mean((all_predicts - targets) ** 2, axis=(0, 1))  # Average over samples and time steps
# for channel, mse_val in enumerate(mse_per_channel):
#     print(f"Channel {channel}: {mse_val:.6f}")



# # Plot comparison (if you want to visualize)
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(targets[0])  # First target sequence
# plt.title('Target')
# plt.subplot(1, 2, 2)
# plt.plot(all_predicts[0])  # First prediction sequence
# plt.title('Prediction')
# plt.show()