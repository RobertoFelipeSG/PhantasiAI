import sys
import pandas as pd


def process_raw_data(file1_path, file2_path, output_path):
    print("Reading input files...")
    
    # 1: Process raw emg file to get trial timestamps
    df1 = pd.read_csv(file1_path)
    df1.columns = df1.columns.str.strip()  # Clean up any hidden whitespace

    trials = []
    current_trial = {}
    trial_counter = 1

    # State tracking for the trial -> e_stim -> event pipeline
    state = "TRIAL" # first timestamp we will gather is the first trial marker (skip the first event+e_stim marker)

    for _, row in df1.iterrows():
       
        # iterate through rows to extract and rename sequential timestamps for each trial
        if state == "TRIAL" and row["trial"] == 1:
            current_trial["trial_num"] = trial_counter
            current_trial["trial_start"] = row["timestamp"]
            state = "STIM"

        if state == "STIM" and row["e_stim"] == 1:
            current_trial["stim_start"] = row["timestamp"]
            state = "EVENT"

        if state == "EVENT" and row["event"] == 1:
            current_trial["dorsi_start"] = row["timestamp"]
            trials.append(current_trial)

            # Reset for the next sequence cycle
            current_trial = {}
            trial_counter += 1
            state = "TRIAL"

    # Convert the processed trial list into a DataFrame
    df_trials = pd.DataFrame(trials)

    # 2: Process the peak features file
    df2 = pd.read_csv(file2_path)
    df2.columns = df2.columns.str.strip()

    # Disregard the first row of data entirely
    df2_filtered = df2.iloc[1:].copy()

    # Extract and rename the target metrics
    df_peaks = (
        df2_filtered[["timestamp", "max_amplitude"]]
        .rename(columns={"timestamp": "peak_time", "max_amplitude": "peak_amp"})
        .reset_index(drop=True)
    )

    # 3: Merge data and export

    final_df = pd.concat([df_trials, df_peaks], axis=1)
    final_df.to_csv(output_path, index=False)
    print(f"Processing complete! Output saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4: # Ensure all required file paths are passed via CLI arguments
        print("Error: Missing arguments.\n"
            "Usage: python process_data.py <file1_path> <file2_path> <output_path>")
    else:
        process_raw_data(sys.argv[1], sys.argv[2], sys.argv[3])