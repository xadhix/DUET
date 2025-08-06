import numpy as np
import os

def find_and_load_npy_files():
    """
    Find all prediction-target pairs in the DUET directory and calculate channel-wise MSE.
    """
    base_path = r"C:\Duet\DUET"  ### Change this to your DUET directory path ###
    
    # Find all .npy files
    all_files = [f for f in os.listdir(base_path) if f.endswith('.npy')]
    
    # Group prediction and target files
    pairs = {}
    for file in all_files:
        if 'all_predicts_' in file:
            key = file.replace('all_predicts_', '').replace('.npy', '')
            if key not in pairs:
                pairs[key] = {}
            pairs[key]['predictions'] = os.path.join(base_path, file)
        elif 'targets_' in file:
            key = file.replace('targets_', '').replace('.npy', '')
            if key not in pairs:
                pairs[key] = {}
            pairs[key]['targets'] = os.path.join(base_path, file)
        elif 'scaler_' in file:
            scaler = os.path.join(base_path, file)
    
    print(f"Found {len(all_files)} .npy files total")
    print(f"Found {len(pairs)} prediction-target pairs")
    
    return pairs , scaler

def calculate_channel_wise_mse_table():
    """
    Calculate channel-wise MSE for all pairs and display as a table.
    """
    pairs , scaler = find_and_load_npy_files()

    # ETTh2 channel names
    channel_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    
    # Store results
    results = []
    
    print("\n" + "=" * 120)
    print("LOADING AND ANALYZING ALL PAIRS")
    print("=" * 120)

    scaler = np.load(scaler)
    mean, std = scaler[0], scaler[1]

    # print(f"Scaler files loaded: {scaler}")
    # print(f"here they are :- Scaler mean: {mean}, std: {std}")
    # print(f"Scaler shape: {mean.shape}, {std.shape}")

    def transform(x):
        # print("mean of x: ", np.mean(x))
        # print("std of x: ", np.std(x))
        # print("mean of scaler: ", mean)
        # print("std of scaler: ", std)
        xminusmean = (x - mean)
        # print("xminusmean: ", np.sum(xminusmean))
        return (x - mean) / std
    def transform_channel(x,ch):
        # print("mean of x: ", np.mean(x))
        # print("std of x: ", np.std(x))
        # print("mean of channel: ", mean[ch])
        # print("std of channel: ", std[ch])
        xminusmean = (x - mean[ch])
        # print("xminusmean: ", np.sum(xminusmean))
        check = xminusmean / (std[ch]+ 1e-8)
        # print("np.sum(check): ", np.sum(check))
        if  (np.sum(check) == 0):
            print(f"Warning: Channel {ch} has zero variance in the scaler, skipping transformation.")
            exit(1)

        return check
    
    for key, files in pairs.items():
        if 'predictions' not in files or 'targets' not in files:
            print(f"Skipping {key} - missing prediction or target file")
            exit(1)
        
        try:
            # Load arrays
            predictions = np.load(files['predictions'])
            targets = np.load(files['targets'])
            
            print(f"\nLoaded {key}: Shape {predictions.shape}")

            
            # Calculate overall MSE
            difference = transform(predictions) - transform(targets)
            overall_mse = np.mean((difference) ** 2)
            
            # Calculate channel-wise MSE
            if len(predictions.shape) == 3:
                num_channels = predictions.shape[2]
                channel_mses = []
                
                for ch in range(num_channels):
                    # print(f"\nProcessing channel {ch} for {key}")
                    # print("transforming predictions")
                    # print("sum of predictions: ", np.sum(predictions[:, :, ch]))
                    transformed_predictions = transform_channel(predictions[:, :, ch],ch)
                    # print("transforming targets")
                    # print("sum of targets: ", np.sum(targets[:, :, ch]))
                    transformed_targets = transform_channel(targets[:, :, ch],ch)
                    difference = transformed_predictions - transformed_targets
                    # print(" sum of difference: ", np.sum(difference))
                    # print("difference squared: ", np.sum(difference ** 2))

                    ch_mse = np.mean((difference) ** 2)
                    # print(f"Channel {ch} MSE: {ch_mse}")
                    channel_mses.append(ch_mse)
                
                results.append({
                    'name': key,
                    'overall_mse': overall_mse,
                    'channel_mses': channel_mses
                })
            else:
                print(f"{key} is not 3D - skipping channel analysis")
        
        except Exception as e:
            print(f"Error loading {key}: {e}")
    
    # Display results table in markdown format
    if results:
        print("\n" + "=" * 120)
        print("CHANNEL-WISE MSE TABLE (MARKDOWN FORMAT)")
        print("=" * 120)
        
        # Create markdown header
        header_names = ['Pair Name', 'Overall MSE'] + channel_names[:len(results[0]['channel_mses'])]
        markdown_header = "| " + " | ".join(f"{name:>10s}" for name in header_names) + " |"
        markdown_separator = "|" + "|".join(":" + "-" * 10 + ":" for _ in header_names) + "|"
        
        print(markdown_header)
        print(markdown_separator)
        
        # Print data rows in markdown format
        for result in results:
            row_data = [f"{result['name']:>10s}", f"{result['overall_mse']}"]
            for ch_mse in result['channel_mses']:
                row_data.append(f"{ch_mse}")
            markdown_row = "| " + " | ".join(row_data) + " |"
            print(markdown_row)
        
        # Calculate averages in markdown format
        avg_overall = np.mean([r['overall_mse'] for r in results])
        avg_data = [f"{'AVERAGE':>10s}", f"{avg_overall}"]
        
        num_channels = len(results[0]['channel_mses'])
        for ch in range(num_channels):
            ch_avg = np.mean([r['channel_mses'][ch] for r in results])
            avg_data.append(f"{ch_avg}")
        avg_markdown_row = "| " + " | ".join(avg_data) + " |"
        # print(avg_markdown_row)
        
        # Calculate standard deviations in markdown format
        std_overall = np.std([r['overall_mse'] for r in results])
        std_data = [f"{'STD DEV':>10s}", f"{std_overall}"]
        
        for ch in range(num_channels):
            ch_std = np.std([r['channel_mses'][ch] for r in results])
            std_data.append(f"{ch_std}")
        std_markdown_row = "| " + " | ".join(std_data) + " |"
        # print(std_markdown_row)
        
        # Find best and worst channels overall in markdown format
        print("\n## CHANNEL PERFORMANCE SUMMARY")
        print("\n| Rank | Channel | Name | Average MSE |")
        print("|:----:|:-------:|:----:|:-----------:|")
        
        channel_avgs = []
        for ch in range(num_channels):
            ch_avg = np.mean([r['channel_mses'][ch] for r in results])
            channel_avgs.append((ch, channel_names[ch], ch_avg))
        
        # Sort by MSE
        channel_avgs.sort(key=lambda x: x[2])
        
        for i, (ch_idx, ch_name, avg_mse) in enumerate(channel_avgs):
            print(f"| {i+1:2d} | {ch_idx:7d} | {ch_name:4s} | {avg_mse} |")

if __name__ == "__main__":
    calculate_channel_wise_mse_table()