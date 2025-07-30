import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne.io import RawArray
from mne import create_info
from mne.viz import set_browser_backend
from scipy.signal import find_peaks
from pathlib import Path

class EMGPeakAnalyzer:
    def __init__(self, csv_path, sampling_rate=220, height_percentile=98, min_distance=3):
        """
        Initialize the EMGPeakAnalyzer class.
        """
        self.csv_path = Path(csv_path)
        self.sampling_rate = sampling_rate  # Hz
        self.height_percentile = height_percentile  # Peak height threshold percentile
        self.min_distance = min_distance  # Minimum distance between peaks in seconds

        self.df_emg = None  # Will hold the raw CSV data
        self.raw = None  # Unfiltered EMG signal in MNE format
        self.raw_filtered = None  # Filtered EMG signal
        self.peaks = None  # Indices of detected peaks
        self.times = None  # Time vector for the signal

    def _load_data(self):
        """
        Load EMG data (assumed in uV) from CSV and wrap in an MNE RawArray.
        """
        self.df_emg = pd.read_csv(self.csv_path)

        # Pick the first column that looks like EMG
        emg_col = None
        for c in self.df_emg.columns:
            if 'emg' in c.lower() or c.lower().startswith('ch'):
                emg_col = c
                break
        if emg_col is None:
            raise ValueError(f"No EMG column found. Available: {self.df_emg.columns.tolist()}")

        emg_uv = self.df_emg[emg_col].dropna().astype(np.float64).values
        self.times = self.df_emg['timestamp'].values[:len(emg_uv)]

        # Convert µV to V for MNE
        info = create_info(ch_names=['EMG (uV)'], sfreq=self.sampling_rate, ch_types=['emg'])
        self.raw = RawArray((emg_uv * 1e-6).reshape(1, -1), info)

    def _filter_data(self):
        """
        Apply a low-pass filter to remove high-frequency noise above 100 Hz.
        """
        self.raw_filtered = self.raw.copy()
        self.raw_filtered.filter(None, 100, method="iir", phase="forward", picks="emg")

    def _detect_peaks(self):
        """
        Detect peaks on the filtered signal, but do all math in uV.
        """
        # get filtered data (in V) and times
        data_v, _ = self.raw_filtered[:, :]
        data_uv = data_v.squeeze() * 1e6          # V → µV
        data_uv -= data_uv.mean()                # center around zero

        threshold = np.percentile(data_uv, self.height_percentile)
        min_dist_samples = int(self.min_distance * self.sampling_rate)

        self.peaks = find_peaks(
            data_uv,
            height=threshold,
            distance=min_dist_samples
        )[0]

    def _analyze_peaks_per_minute(self):
        """
        Analyze peaks in 6-second windows to summarize peak locations.
        """
        # 1) grab the filtered data (in volts) and convert to µV
        data_v = self.raw_filtered.get_data().squeeze()
        data_uv = data_v * 1e6          # V → µV
        times = self.times

        duration = times[-1]  # Total duration of the signal in seconds
        samples_per_6s = int(6 * self.sampling_rate)
        results = []

        # Split the signal into 6-second segments
        num_segments = int(np.ceil(duration / 6))
        for i in range(num_segments):
            start_idx = i * samples_per_6s
            end_idx = min((i + 1) * samples_per_6s, len(data_uv))

            segment_data = data_uv[start_idx:end_idx]
            segment_times = times[start_idx:end_idx]

            if len(segment_data) == 0:
                continue

            # Identify the maximum peak within the segment
            peak_idx = np.argmax(segment_data)
            peak_value = segment_data[peak_idx]    # already in µV
            peak_time = segment_times[peak_idx]

            results.append({
                'segment':    i + 1,
                'start_time': segment_times[0],
                'end_time':   segment_times[-1],
                'peak_time':  peak_time,
                'peak_value': peak_value            # µV
            })

        return results

    def _save_results(self):
        """
        Save summary of detected peaks and segment analysis to a text file.
        """
        output_file = self.csv_path.with_name("peaks.txt")

        # Try to make file paths relative to current working directory
        try:
            relative_csv_path = self.csv_path.relative_to(Path.cwd())
        except ValueError:
            relative_csv_path = self.csv_path

        try:
            relative_output_path = output_file.relative_to(Path.cwd())
        except ValueError:
            relative_output_path = output_file

        segment_results = self._analyze_peaks_per_minute()

        # Write results to file
        with open(output_file, 'w') as f:
            f.write(f"EMG File: {relative_csv_path}\n")
            f.write(f"Output File: {relative_output_path.name}\n")
            f.write(f"Number of peaks: {len(self.peaks)}\n")
            f.write(f"Height percentile: {self.height_percentile}\n")
            f.write(f"Min distance: {self.min_distance} seconds\n")
            f.write(f"Sampling rate: {self.sampling_rate} Hz\n")
            f.write(f"Signal duration: {self.raw.times[-1]:.2f} seconds\n\n")

            f.write("Segmented Peak Summary (6s windows):\n")
            for result in segment_results:
                seg = result['segment']
                start = result['start_time']
                end = result['end_time']
                peak_t = result['peak_time']
                peak_v = result['peak_value']
                f.write(f"  Segment {seg} ({start:.2f} to {end:.2f}s): "
                        f"Peak at {peak_t:.2f}s = {peak_v:.2f} uV\n")

        print(f"[PeakAnalyzer] Results saved to: {relative_output_path}")

    def _plot(self):
        """
        Plot the filtered EMG signal and highlight detected peaks.
        """
        fig, ax = plt.subplots(1, 1, layout="constrained")

        # Subtract mean for visual clarity (in µV)
        data_v = self.raw_filtered.get_data().squeeze()
        data_uv = data_v * 1e6
        ax.plot(self.times, data_uv - data_uv.mean(), label='Filtered EMG')

        # Mark detected peak locations
        for peak in self.peaks:
            ax.axvline(self.times[peak], linestyle='--', alpha=0.6)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title(f"EMG Signal with {len(self.peaks)} Peaks")
        ax.legend()
        plt.show()

    def run(self, show_plots=False, save_results=True):
        """
        Main method to execute the full peak analysis pipeline.
        """
        print(f"[PeakAnalyzer] Analyzing: {self.csv_path}")

        self._load_data()       # Load and format EMG data
        self._filter_data()     # Apply filtering
        self._detect_peaks()    # Detect peaks in the signal

        if save_results:
            self._save_results()  # Save analysis results to a text file

        if show_plots:
            set_browser_backend("matplotlib")
            self._plot()          # Plot EMG signal with detected peaks

        # Return useful summary and data for further processing or debugging
        return {
            'csv_file_path': str(self.csv_path),
            'num_peaks': len(self.peaks),
            'peak_times': self.times[self.peaks] if len(self.peaks) > 0 else [],
            'sampling_rate': self.sampling_rate,
            'signal_duration': self.raw.times[-1],
            'raw_data': self.raw,
            'filtered_data': self.raw_filtered
        }
