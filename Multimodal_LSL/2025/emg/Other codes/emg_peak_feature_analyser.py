import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne.io import RawArray
from mne import create_info
from mne.viz import set_browser_backend
from scipy.signal import find_peaks
from pathlib import Path
import neurokit2 as nk
import scipy.signal
from utils.scipy_patch import patch_scipy_welch

# Patch: Replace deprecated 'hanning' window with 'hann' in scipy.signal.welch
scipy.signal.welch = patch_scipy_welch

# Import custom EMG feature functions from pysiology
from pysiology.electromyography import (
    getMAV, getRMS, getWL, getZC, getIEMG, getWAMP,
    getVAR, getLOG, getPSD, getMNF, getMDF
)



"""
peak.py — EMG Peak Detection and Feature Extraction Module
---------------------------------------------------------

This module defines the `EMGPeakAnalyzer` class, which detects peaks in EMG recordings
and extracts features from signal segments around each peak using pysiology functions.

Functionality:
--------------
The module:
- Reads a timestamped EMG signal from a CSV file
- Converts units if needed (mV → µV)
- Applies optional filtering (low-pass)
- Detects peaks above a specified amplitude percentile
- Enforces a minimum time between peaks
- Extracts EMG features from segments around each peak using pysiology
- Optionally shows plots
- Saves the analysis result as `peaks.txt` next to the CSV
- Saves extracted features as `features.txt`
"""

class EMGPeakAnalyzer:
    def __init__(self, csv_path, sampling_rate=220, height_percentile=98, min_distance=3):
        """
        Initialize the analyzer with path and parameters.

        Parameters:
        -----------
        csv_path : str or Path
            Path to the EMG CSV file
        sampling_rate : int
            Sampling rate of EMG data (Hz)
        height_percentile : float
            Threshold for peak detection (percentile of signal amplitude)
        min_distance : float
            Minimum time (in seconds) between peaks
        """
        self.csv_path = Path(csv_path)
        self.sampling_rate = sampling_rate
        self.height_percentile = height_percentile
        self.min_distance = min_distance

        self.df_emg = None
        self.raw = None
        self.raw_filtered = None
        self.peaks = None
        self.times = None
        self.features = None

    def _load_data(self):
        self.df_emg = pd.read_csv(self.csv_path)

        if 'emg' in self.df_emg.columns:
            emg_signal = self.df_emg['emg'].dropna().values * 1e3
        elif 'ch1 (µV)' in self.df_emg.columns:
            emg_signal = self.df_emg['ch1 (µV)'].dropna().values
        elif any(col.startswith('ch') and '(µV)' in col for col in self.df_emg.columns):
            ch_cols = [col for col in self.df_emg.columns if col.startswith('ch') and '(µV)' in col]
            emg_signal = self.df_emg[ch_cols[0]].dropna().values
        else:
            raise ValueError(f"No EMG column found. Available columns: {self.df_emg.columns.tolist()}")

        self.time_vector = self.df_emg['timestamp'].iloc[:len(emg_signal)].values
        info = create_info(ch_names=['EMG'], sfreq=self.sampling_rate, ch_types=['emg'])
        self.raw = RawArray(emg_signal.reshape(1, -1).astype(np.float64), info)

    def _filter_data(self):
        self.raw_filtered = self.raw.copy()
        self.raw_filtered.filter(None, 100, method="iir", phase="forward", picks="emg")

    def _detect_peaks(self):
        start_idx = 0
        stop_idx = self.raw_filtered.n_times
        data, self.times = self.raw_filtered[:, start_idx:stop_idx]
        data -= data.mean()

        self.peaks = find_peaks(
            data.squeeze(),
            height=np.percentile(data.squeeze(), self.height_percentile),
            distance=self.min_distance * self.sampling_rate
        )[0]

    def _extract_features(self, signal, threshold=1e-4):
        """
        Extract multiple EMG features from the signal split into segments around peaks
        using pysiology functions. Number of segments equals number of peaks found.
        ** Recommended to use epochs nor shorter to 125ms and not longer than 2s
        """
        if len(self.peaks) == 0:
            print("No peaks found - can't extract features")
            return None

        # Create segments centered around each peak
        seg_len = int(2 * self.sampling_rate)  # 2 second segments
        segments = []
        for peak in self.peaks:
            start = max(0, peak - seg_len//2)
            end = min(len(signal), peak + seg_len//2)
            segments.append(signal[start:end])

        features = {
            key: [] for key in [
                "MAV", "RMS", "MeanFreq", "MedianFreq",
                "WL", "ZC", "IEMG", "WAMP", "VAR"
            ]
        }

        for i, seg in enumerate(segments):
            seg_list = list(seg)

            # Time-domain features using pysiology
            features["MAV"].append(getMAV(seg_list))
            features["RMS"].append(getRMS(seg_list))
            features["WL"].append(getWL(seg_list))
            features["ZC"].append(getZC(seg_list, threshold))
            features["IEMG"].append(getIEMG(seg_list))
            features["WAMP"].append(getWAMP(seg_list, threshold))
            features["VAR"].append(getVAR(seg_list))


            # Frequency-domain features using pysiology
            psd, freqs = getPSD(seg_list, self.sampling_rate)
            features["MeanFreq"].append(getMNF(psd, freqs))
            features["MedianFreq"].append(getMDF(psd, freqs))

        return features
    
    def _unit(self, feature_name):
        """
        Return appropriate unit for a given EMG feature.
        """
        return {
            "MAV": "uV", "RMS": "uV", "MeanFreq": "Hz", "MedianFreq": "Hz",
            "WL": "uV", "ZC": "count", "IEMG": "uV", "WAMP": "count",
            "VAR": "uV^2"
        }.get(feature_name, "")

    def _save_features_to_file(self, features, selected_features, output_path):
        """
        Save extracted features to a CSV file with features as columns and segments as rows
        """
        import csv
        
        # Prepare headers with feature names and units
        headers = ["Segment"] + [f"{feature} ({self._unit(feature)})" for feature in selected_features]
        
        # Prepare data rows - one row per segment
        rows = []
        for i in range(len(self.peaks)):
            row = [f"Segment_{i+1}"]
            for feature in selected_features:
                row.append(f"{features[feature][i]:.2f}")
            rows.append(row)
        
        # Write to CSV file
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(headers)
            writer.writerows(rows)

    def _save_results(self):
        output_file = self.csv_path.with_name("peaks.txt")
        features_file = self.csv_path.with_name("features.csv")

        # Save peak detection results
        with open(output_file, 'w') as f:
            f.write(f"EMG File: {self.csv_path}\n")
            f.write(f"Number of peaks: {len(self.peaks)}\n")
            f.write(f"Height percentile: {self.height_percentile}\n")
            f.write(f"Min distance: {self.min_distance} seconds\n")
            f.write(f"Sampling rate: {self.sampling_rate} Hz\n")
            f.write(f"Signal duration: {self.raw.times[-1]:.2f} seconds\n")

        # Save features using the reference format
        if self.features is not None:
            all_features = [
                "MAV", "RMS", "MeanFreq", "MedianFreq",
                "WL", "ZC", "IEMG", "WAMP", "VAR"
            ]
            self._save_features_to_file(self.features, all_features, features_file)

        print(f"[PeakAnalyzer] Results saved to: {output_file}")
        if self.features is not None:
            print(f"[PeakAnalyzer] Features saved to: {features_file}")



    def _plot(self):
        fig, ax = plt.subplots(1, 1, layout="constrained")
        ax.plot(self.times, self.raw_filtered.get_data().squeeze() - self.raw_filtered.get_data().mean(), label='Filtered EMG')
        for peak in self.peaks:
            ax.axvline(self.times[peak], color='red', linestyle='--', alpha=0.6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title(f"EMG Signal with {len(self.peaks)} Peaks")
        ax.legend()
        plt.show()


    def run(self, show_plots=False, save_results=True):
        """
        Run the full peak detection and feature extraction pipeline.

        Parameters:
        -----------
        show_plots : bool
            Whether to display the signal + peak plot
        save_results : bool
            Whether to write results to files

        Returns:
        --------
        dict : results containing metadata, peak info, and features
        """
        print(f"[PeakAnalyzer] Analyzing: {self.csv_path}")
        self._load_data()
        self._filter_data()
        self._detect_peaks()
        
        # Extract features from segments around each peak using pysiology
        if len(self.peaks) > 0:
            self.features = self._extract_features(self.raw_filtered.get_data().squeeze())

        if save_results:
            self._save_results()

        if show_plots:
            set_browser_backend("matplotlib")
            self._plot()

        return {
            'csv_file_path': str(self.csv_path),
            'num_peaks': len(self.peaks),
            'peak_times': self.times[self.peaks] if len(self.peaks) > 0 else [],
            'sampling_rate': self.sampling_rate,
            'signal_duration': self.raw.times[-1],
            'raw_data': self.raw,
            'filtered_data': self.raw_filtered,
            'features': self.features
        }
    
    