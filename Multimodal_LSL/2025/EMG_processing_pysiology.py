import numpy as np
import pandas as pd
import time
from scipy.signal import welch
import neurokit2 as nk
from pathlib import Path
from scipy.io import loadmat
import scipy.signal

# Save original welch function
_original_welch = scipy.signal.welch

# Patch welch to replace 'hanning' with 'hann'
def patched_welch(*args, **kwargs):
    if kwargs.get("window") == "hanning":
        kwargs["window"] = "hann"
    return _original_welch(*args, **kwargs)

# Apply patch
scipy.signal.welch = patched_welch


# Use Pysiology to compute the features
from pysiology.electromyography import (
    getMAV, getRMS, getWL, getZC, getIEMG, getWAMP,
    getVAR, getLOG, getPSD, getMNF, getMDF
)


# Preprocess EMG signal 

def preprocess_signal(signal, sampling_rate=2000):

     # Filter the signal (default bandpass 20–450 Hz)
    clean = nk.emg_clean(signal, sampling_rate=sampling_rate)
    # Rectified signal
    rect = np.abs(clean)
    # Smoothed envelope of rectified signal
    envelope = nk.signal_smooth(rect)
    # Signal amplitude
    amplitude = nk.emg_amplitude(clean)

    # Build DF
    df = pd.DataFrame({
        "EMG_Raw": signal,
        "EMG_Clean": clean,
        "EMG_Rect": rect,
        "EMG_Envelope": envelope,
        "EMG_Amplitude": amplitude,
    })

    return df


# Feature extraction per EMG segment using pysiology

def extract_emg_features_per_segment(signal, num_segments=10, fs=2000, threshold=1e-4):
    signal = np.array(signal)
    segment_length = len(signal) // num_segments

    features_per_segment = {
        key: [] for key in [
            "MAV", "RMS", "MeanFreq", "MedianFreq",
            "WL", "ZC", "IEMG", "WAMP", "VAR", "LogD"
        ]
    }

    for i in range(num_segments):
        segment = signal[i * segment_length:(i + 1) * segment_length]
        segment_list = list(segment)  # pysiology expects lists

        # Time-domain features
        features_per_segment["MAV"].append(getMAV(segment_list))
        features_per_segment["RMS"].append(getRMS(segment_list))
        features_per_segment["WL"].append(getWL(segment_list))
        features_per_segment["ZC"].append(getZC(segment_list, threshold))
        features_per_segment["IEMG"].append(getIEMG(segment_list))
        features_per_segment["WAMP"].append(getWAMP(segment_list, threshold))
        features_per_segment["VAR"].append(getVAR(segment_list))
        features_per_segment["LogD"].append(getLOG(segment_list))

        # Frequency-domain features
        psd, freqs = getPSD(segment_list, fs)
        features_per_segment["MeanFreq"].append(getMNF(psd, freqs))
        features_per_segment["MedianFreq"].append(getMDF(psd, freqs))

        # Optional preview
        print(f"Segment {i+1}/{num_segments}:")
        print(f"  MAV = {features_per_segment['MAV'][-1]:.4f}")
        print(f"  RMS = {features_per_segment['RMS'][-1]:.4f}")
        print(f"  WL = {features_per_segment['WL'][-1]:.4f}")
        print(f"  ZC = {features_per_segment['ZC'][-1]}")
        print(f"  IEMG = {features_per_segment['IEMG'][-1]:.4f}")
        print(f"  WAMP = {features_per_segment['WAMP'][-1]}")
        print(f"  VAR = {features_per_segment['VAR'][-1]:.4f}")
        print(f"  LogD = {features_per_segment['LogD'][-1]:.4f}")
        print(f"  MeanFreq = {features_per_segment['MeanFreq'][-1]:.2f}")
        print(f"  MedianFreq = {features_per_segment['MedianFreq'][-1]:.2f}\n")

    return features_per_segment


# Load last minute EMG data

def load_last_minute_emg_data(file_path, selected_features, num_segments=10, fs=2000):
    
    # Load EMG + Timestamp data
    data = pd.read_csv(file_path, sep="\t")

    # Compute minute column  
    data["minute"] = (data["Timestamp"] // 60).astype(int)

    # Filter rows corresponding to the last minute of recording
    last_minute = data["minute"].max()
    last_minute_data = data[data["minute"] == last_minute]["EMG1"].values
    print("Raw last minute EMG1 min/max:", last_minute_data.min(), last_minute_data.max())

    # Scale signal to microvolts
    last_minute_data *= 1e6  

    # Process the raw signal
    processed_data = preprocess_signal(last_minute_data)

    # Keep only the filtered signal
    filtered_signal = processed_data["EMG_Clean"].values

    # Extract features for that last minute
    segmented_features = extract_emg_features_per_segment(filtered_signal, num_segments, fs)
    
    # Keep only the selected features
    filtered_features = {key: segmented_features[key] for key in selected_features}
    
    return filtered_features

# Saving functions

def save_subject_in_file(file, vector, n_suj, n_dim, n_val, interaction_scale, interaction_sign):
    file.write(str(n_suj) + ";")
    for i_dim in range(n_dim):
        for i_val in range(n_val):
            file.write(f"{round(vector[i_dim][i_val], 2)} ")
        file.write(";")
    file.write(f"{interaction_scale};{interaction_sign}\n")

def save_data_in_file(vector, titles, n_suj, n_dim, n_val, interaction_sign, interaction_scale, output_path):
    with open(output_path, "w") as file:
        file.write("Sujet;")
        for i_dim in range(n_dim):
            file.write(titles[i_dim] + ";")
        file.write("Interaction_Scale;Interaction_Sign\n")
        for i_suj in range(n_suj):
            save_subject_in_file(file, vector[i_suj], i_suj, n_dim, n_val, interaction_scale, interaction_sign)





def main():

    ## ADDED PROCESS FOR DATA FROM NINAPRO

    # Load .mat file and save to txt
    data = loadmat("S1_D1_T1.mat")
    print(data.keys())
    # Extract EMG and time arrays
    emg = data["emg"]
    time_arr = data["time"]
    print("EMG shape:", emg.shape)
    print("Time shape:", time_arr.shape)
    # Generate timestamp vector based on sampling rate (2000 Hz)
    fs = 2000
    num_samples = emg.shape[0]
    timestamp = np.arange(num_samples) / fs

    # Build DF with EMG + timestamp
    df = pd.DataFrame(emg)
    df["Timestamp"] = timestamp
    emg_cols = [f"EMG{i+1}" for i in range(emg.shape[1])]
    df.columns = emg_cols + ["Timestamp"]
    df.to_csv("emg_with_timestamps.txt", sep="\t", index=False)
    print("Saved as emg_with_timestamps.txt")

    # Select only EMG channel 1 and Timestamp columns
    df_subset = df[["EMG1", "Timestamp"]]
    df_subset.to_csv("emg1_with_timestamps.txt", sep="\t", index=False)
    print("Saved as emg1_with_timestamps.txt")

    # END --------------

    ## PROCESS WITH EXPERIMENTAL DATA FROM .TXT

    # Configurable parameters 
    file_path = "emg1_with_timestamps.txt"
    output_path = "processed_trial.txt"
    n_suj = 1

    # Features we want to extract
    selected_features = ["MAV", "RMS", "MeanFreq", "MedianFreq", "WL", "ZC", "IEMG", "WAMP", "VAR", "LogD"]
    #selected_features = ["MAV", "RMS"]

    units = {
        "MAV": "µV",
        "RMS": "µV",
        "MeanFreq": "Hz",
        "MedianFreq": "Hz",
        "WL": "µV",
        "ZC": "count",
        "IEMG": "µV",
        "WAMP": "count",
        "VAR": "µV^2",
        "LogD": "µV"
    }

    n_dim = len(selected_features)
    n_val = 10
    interaction_sign = 1
    interaction_scale = 0.5
    titles = [f"{feat} ({units[feat]})" for feat in selected_features]

    # Call load_last_minute_emg_data() to get features
    features_per_minute = load_last_minute_emg_data(file_path, selected_features, num_segments=n_val)

    if features_per_minute:
        # Convert feature dict to vector
        vector = [list(features_per_minute.values())]

        # Preview data
        print("Processed data:")
        for sublist in vector[0]:
            print([round(float(val), 2) for val in sublist])

        # Save features in file
        save_data_in_file(vector, titles, n_suj, n_dim, n_val, interaction_sign, interaction_scale, output_path)
        print("File saved.")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\n Total execution time: {end - start:.4f} seconds")