import sys
import pandas as pd
import numpy as np

def process_channel(accel_signal, timestamps, event_indices):
    # Get indices of all events
    peak_values = [] 
    trough_values = []

    # Iterate through each event to define the window
    for event_idx in event_indices:
        
        # define window: -2.5/+2 seconds from event marker timestamp
        event_time = timestamps[event_idx]
        start_time = event_time - 2.5
        end_time = event_time + 2
        
        # Find the nearest indices for the window boundaries and get data
        start_idx = np.searchsorted(timestamps, start_time, side='left')
        end_idx = np.searchsorted(timestamps, end_time, side='right')
        trial_data = accel_signal[start_idx:end_idx]
        
        # Find peak in this trial
        if len(trial_data) > 0:
            peak = np.max(trial_data)
            trough = np.min(trial_data)
            peak_values.append(peak)
            trough_values.append(trough)
        else:
            peak_values.append(np.nan)
            trough_values.append(np.nan)

    return peak_values, trough_values

def process_raw_data(file1_path, output_path):
    print("Reading input files...")
    
    # Process raw accel file to get peak feature values
    df = pd.read_csv(file1_path)
    df.columns = df.columns.str.strip()  # Clean up any hidden whitespace
    df = df.sort_values(by='timestamp').reset_index(drop=True) # sort timestamps

    # Get info from DataFrame
    channels = [col for col in df.columns if col.startswith('accel')]
    timestamps = df['timestamp'].values
    event_column = df['event'].values
    event_indices = np.where(event_column == 1)[0]

    # Run feature extraction
    all_features = {'event_timestamp': timestamps[event_indices]}
    for channel in channels:
        accel_signal = df[channel].values
        peaks, troughs = process_channel(accel_signal, timestamps, event_indices)
        all_features[f'{channel}_max'] = peaks
        all_features[f'{channel}_min'] = troughs

        # Get 5 minimum and 5 maximum values (ignoring NaNs)
        valid_peaks = [p for p in peaks if not np.isnan(p)]
        valid_troughs = [t for t in troughs if not np.isnan(t)]
        top_5_max = sorted(valid_peaks, reverse=True)[:10]
        bottom_5_min = sorted(valid_troughs)[:10]

        print(f"\n--- Channel: {channel} ---")
        print(f"  10 Maximum Trial Peaks:     {[round(float(x), 4) for x in top_5_max]}")
        print(f"  10 Minimum Trial Troughs: {[round(float(x), 4) for x in bottom_5_min]}")

    final_df = pd.DataFrame(all_features)
    final_df.to_csv(output_path, index=False)
    print(f"Processing complete! Output saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3: # Ensure all required file paths are passed via CLI arguments
        print("Error: Missing arguments.\n"
            "Usage: python process_data.py <file1_path> <output_path>")
    else:
        process_raw_data(sys.argv[1], sys.argv[2])