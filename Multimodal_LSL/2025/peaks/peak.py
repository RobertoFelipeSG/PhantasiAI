import numpy as np
from matplotlib import pyplot as plt
from mne.io import read_raw_fif
from mne.viz import set_browser_backend
from scipy.signal import find_peaks
import pandas as pd
from mne import create_info
from mne.io import RawArray
import os
import glob
from pathlib import Path

from mne_lsl.datasets import sample


def get_latest_emg_file(folder_path="../emg-recordings"):
    """
    Get the latest CSV file from the emg-recordings folder.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing EMG recordings
        
    Returns:
    --------
    str : Path to the latest CSV file
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' not found")
    
    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{folder_path}'")
    
    # Get the latest file based on modification time
    latest_file = max(csv_files, key=os.path.getmtime)
    return latest_file


def analyze_emg_peaks(csv_file_path=None, 
                     sampling_rate=220,
                     height_percentile=98,
                     min_distance=3,
                     show_plots=True,
                     save_results=True):
    """
    Analyze EMG signal and detect peaks.
    
    Parameters:
    -----------
    csv_file_path : str, optional
        Path to the CSV file. If None, will use the latest file from emg-recordings folder
    sampling_rate : int
        Sampling rate in Hz
    height_percentile : float
        Percentile threshold for peak detection
    min_distance : float
        Minimum distance between peaks in seconds
    show_plots : bool
        Whether to display plots
    save_results : bool
        Whether to save results to a text file
        
    Returns:
    --------
    dict : Dictionary containing analysis results
    """
    
    # Get the CSV file path
    if csv_file_path is None:
        csv_file_path = get_latest_emg_file()
        print(f"Using latest EMG file: {csv_file_path}")
    
    # Load EMG Data
    df_emg = pd.read_csv(csv_file_path)
    
    # Extract and convert EMG signal from mV to µV
    emg_signal = df_emg['emg'].dropna().values * 1e3  # mV → µV
    time_vector = df_emg['timestamp'].iloc[:len(emg_signal)].values
    
    # Create MNE Raw Object
    info = create_info(ch_names=['EMG'], sfreq=sampling_rate, ch_types=['emg'])
    raw = RawArray(emg_signal.reshape(1, -1).astype(np.float64), info)
    
    # Debug information
    print(f"First 10 values from RawArray: {raw.get_data()[0, :10]}")
    print(f"Signal duration: {raw.times[-1]:.2f} seconds")
    
    # Plot raw EMG data 
    if show_plots:
        set_browser_backend("matplotlib")
        raw.plot(scalings=dict(emg=1000), show_scrollbars=False, duration=130)
        plt.show()
    
    # Lowpass Filtering 
    raw_lowpassed = raw.copy()
    _ = raw_lowpassed.filter(None, 100, method="iir", phase="forward", picks="emg")
    
    # Define signal duration 
    signal_start = 0
    signal_end = raw.times[-1]
    
    # Plot lowpassed signal
    if show_plots:
        start = int(signal_start * raw.info["sfreq"])
        stop = int(signal_end * raw.info["sfreq"])
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), layout="constrained")
        data, times = raw_lowpassed[:, start:stop] 
        data -= data.mean()  # detrend
        ax.plot(times, data.squeeze(), label='Lowpassed Signal')
        ax.legend()
        plt.show()
    
    # Detect and mark peaks (using the lowpassed signal)
    start = int(signal_start * raw.info["sfreq"])
    stop = int(signal_end * raw.info["sfreq"])
    data, times = raw_lowpassed[:, start:stop]
    data -= data.mean()  # detrend
    
    peaks = find_peaks(
        data.squeeze(),
        height=np.percentile(data.squeeze(), height_percentile),
        distance=min_distance * raw_lowpassed.info["sfreq"],
    )[0]
    
    # Plot peaks
    if show_plots:
        fig, ax = plt.subplots(1, 1, layout="constrained")
        ax.plot(times, data.squeeze(), label='Filtered EMG Signal')
        for peak in peaks:
            ax.axvline(times[peak], color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'EMG Signal with {len(peaks)} Detected Peaks')
        ax.legend()
        plt.show()
    
    # Save results
    if save_results:
        base_filename = Path(csv_file_path).stem
        output_filename = f'peaks_count_of_{base_filename}.txt'
        with open(output_filename, 'w') as f:
            f.write(f"EMG File: {csv_file_path}\n")
            f.write(f"Number of peaks: {len(peaks)}\n")
            f.write(f"Height percentile: {height_percentile}\n")
            f.write(f"Min distance: {min_distance} seconds\n")
            f.write(f"Sampling rate: {sampling_rate} Hz\n")
            f.write(f"Signal duration: {signal_end:.2f} seconds\n")
        print(f"Results saved to: {output_filename}")
    
    print(f"Number of peaks detected: {len(peaks)}")
    
    # Return results dictionary
    results = {
        'csv_file_path': csv_file_path,
        'num_peaks': len(peaks),
        'peaks_indices': peaks,
        'peak_times': times[peaks] if len(peaks) > 0 else [],
        'signal_duration': signal_end,
        'sampling_rate': sampling_rate,
        'raw_data': raw,
        'filtered_data': raw_lowpassed
    }
    
    return results


if __name__ == "__main__":
    results = analyze_emg_peaks()
    
 