#!/usr/bin/env python3
"""
Fixed EMG Peak Analyzer for Combined Dataset
============================================

This is a corrected version of the EMGPeakAnalyzer that can properly handle
the combined_emg_dorsiflex.csv dataset format with columns:
- Time, fwEMG 3, Subject, MVC, Trial

The original analyzer was designed for different CSV formats and was causing
incorrect peak detection.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne.io import RawArray
from mne import create_info
from mne.viz import set_browser_backend
from scipy.signal import find_peaks
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EMGPeakAnalyzerFixed:
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
        self.emg_signal = None

    def _load_data(self):
        """Load data from CSV with proper column handling."""
        self.df_emg = pd.read_csv(self.csv_path)
        
        print(f"Available columns: {self.df_emg.columns.tolist()}")
        
        # Handle different possible column names for EMG data
        if 'fwEMG 3' in self.df_emg.columns:
            # Combined dataset format
            emg_signal = self.df_emg['fwEMG 3'].dropna().values
            time_vector = self.df_emg['Time'].iloc[:len(emg_signal)].values
            print(f"Using 'fwEMG 3' column for EMG data")
        elif 'emg' in self.df_emg.columns:
            # Legacy format
            emg_signal = self.df_emg['emg'].dropna().values * 1e3  # Convert mV to µV
            time_vector = self.df_emg['timestamp'].iloc[:len(emg_signal)].values
            print(f"Using 'emg' column for EMG data (converted from mV to µV)")
        elif 'ch1 (µV)' in self.df_emg.columns:
            # New format
            emg_signal = self.df_emg['ch1 (µV)'].dropna().values
            time_vector = self.df_emg['timestamp'].iloc[:len(emg_signal)].values
            print(f"Using 'ch1 (µV)' column for EMG data")
        elif any(col.startswith('ch') and '(µV)' in col for col in self.df_emg.columns):
            # Multi-channel format
            ch_cols = [col for col in self.df_emg.columns if col.startswith('ch') and '(µV)' in col]
            emg_signal = self.df_emg[ch_cols[0]].dropna().values
            time_vector = self.df_emg['timestamp'].iloc[:len(emg_signal)].values
            print(f"Using '{ch_cols[0]}' column for EMG data")
        else:
            raise ValueError(f"No EMG column found. Available columns: {self.df_emg.columns.tolist()}")

        # Store the EMG signal and time vector
        self.emg_signal = emg_signal
        self.time_vector = time_vector
        
        print(f"EMG signal shape: {emg_signal.shape}")
        print(f"Time vector shape: {time_vector.shape}")
        print(f"EMG signal range: {emg_signal.min():.4f} to {emg_signal.max():.4f} µV")
        print(f"Time range: {time_vector.min():.2f} to {time_vector.max():.2f} seconds")
        
        # Create MNE Raw object
        info = create_info(ch_names=['EMG'], sfreq=self.sampling_rate, ch_types=['emg'])
        self.raw = RawArray(emg_signal.reshape(1, -1).astype(np.float64), info)

    def _filter_data(self):
        """Apply low-pass filter to the EMG signal."""
        self.raw_filtered = self.raw.copy()
        self.raw_filtered.filter(None, 100, method="iir", phase="forward", picks="emg")
        print(f"Applied low-pass filter (100 Hz cutoff)")

    def _detect_peaks(self):
        """Detect peaks in the filtered EMG signal."""
        start_idx = 0
        stop_idx = self.raw_filtered.n_times
        data, mne_times = self.raw_filtered[:, start_idx:stop_idx]
        
        # Use the original time vector from the CSV file, not MNE's internal time
        self.times = self.time_vector
        
        # Remove DC offset
        data_centered = data.squeeze() - data.squeeze().mean()
        
        # Extract envelope using Hilbert transform
        from scipy.signal import hilbert
        analytic_signal = hilbert(data_centered)
        envelope = np.abs(analytic_signal)
        print(f"Extracted signal envelope using Hilbert transform")
        
        # Resample envelope to 200Hz
        from scipy import signal
        target_sampling_rate = 200
        current_sampling_rate = self.sampling_rate
        
        # Calculate resampling factor
        resample_factor = target_sampling_rate / current_sampling_rate
        new_length = int(len(envelope) * resample_factor)
        
        # Resample the envelope
        envelope_resampled = signal.resample(envelope, new_length)
        
        # Create new time vector for resampled signal
        original_duration = self.times[-1] - self.times[0]
        resampled_times = np.linspace(self.times[0], self.times[-1], new_length)
        
        print(f"Resampled envelope from {current_sampling_rate}Hz to {target_sampling_rate}Hz")
        print(f"Original length: {len(envelope)}, Resampled length: {len(envelope_resampled)}")
        
        # Store resampled data for later use
        self.envelope_resampled = envelope_resampled
        self.resampled_times = resampled_times
        self.resampled_sampling_rate = target_sampling_rate
        
        # Calculate height threshold on resampled envelope
        height_threshold = np.percentile(envelope_resampled, self.height_percentile)
        print(f"Height threshold (percentile {self.height_percentile}): {height_threshold:.4f} µV")
        
        # Detect peaks in the resampled envelope
        self.peaks, properties = find_peaks(
            envelope_resampled,  # Use resampled envelope for peak detection
            height=height_threshold,
            distance=int(self.min_distance * target_sampling_rate)  # Adjust distance for new sampling rate
        )
        
        print(f"Detected {len(self.peaks)} peaks in resampled envelope")
        if len(self.peaks) > 0:
            peak_amplitudes = envelope_resampled[self.peaks]  # Use envelope values
            print(f"Peak amplitudes range: {peak_amplitudes.min():.4f} to {peak_amplitudes.max():.4f} µV")

    def _save_results(self):
        """Save peak detection results."""
        output_file = self.csv_path.with_name("peaks.txt")

        # Get relative paths for display
        try:
            relative_csv_path = self.csv_path.relative_to(Path.cwd())
        except ValueError:
            relative_csv_path = self.csv_path

        try:
            relative_output_path = output_file.relative_to(Path.cwd())
        except ValueError:
            relative_output_path = output_file

        with open(output_file, 'w') as f:
            f.write(f"EMG File: {relative_csv_path}\n")
            f.write(f"Output File: {relative_output_path.name}\n")
            f.write(f"Number of peaks: {len(self.peaks)}\n")
            f.write(f"Height percentile: {self.height_percentile}\n")
            f.write(f"Min distance: {self.min_distance} seconds\n")
            f.write(f"Sampling rate: {self.sampling_rate} Hz\n")
            f.write(f"Signal duration: {self.raw.times[-1]:.2f} seconds\n")
            
            if len(self.peaks) > 0:
                f.write(f"\nPeak details:\n")
                for i, peak_idx in enumerate(self.peaks):
                    peak_time = self.resampled_times[peak_idx]
                    peak_amplitude = self.envelope_resampled[peak_idx]
                    f.write(f"Peak {i+1}: Time={peak_time:.3f}s, Amplitude={peak_amplitude:.4f}µV\n")
        
        print(f"[PeakAnalyzer] Results saved to: {relative_output_path}")

    def _plot(self):
        """Plot the EMG signal with detected peaks."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), layout="constrained")
        
        # Plot original signal
        ax1.plot(self.time_vector, self.emg_signal, label='Original EMG Signal', alpha=0.7)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude (µV)")
        ax1.set_title("Original EMG Signal")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot filtered and centered signal
        data_centered = self.raw_filtered.get_data().squeeze() - self.raw_filtered.get_data().squeeze().mean()
        ax2.plot(self.times, data_centered, label='Filtered EMG (centered)', alpha=0.7)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude (µV)")
        ax2.set_title("Filtered and Centered EMG Signal")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot resampled envelope with peaks
        ax3.plot(self.resampled_times, self.envelope_resampled, label='Resampled Envelope (200Hz)', alpha=0.7, color='orange')
        
        if len(self.peaks) > 0:
            peak_times = self.resampled_times[self.peaks]
            peak_amplitudes = self.envelope_resampled[self.peaks]
            
            # Plot all peaks
            ax3.scatter(peak_times, peak_amplitudes, color='red', s=50, alpha=0.8, label='Detected Peaks')
            
            # Highlight the highest peak
            max_peak_idx = np.argmax(peak_amplitudes)
            ax3.scatter(peak_times[max_peak_idx], peak_amplitudes[max_peak_idx], 
                       color='green', s=100, marker='*', edgecolors='black', linewidth=2,
                       label=f'Highest Peak: {peak_amplitudes[max_peak_idx]:.3f}µV')
            
            # Add peak count to title
            title = f"Resampled Envelope with {len(self.peaks)} Peaks"
        else:
            title = "Resampled Envelope (No Peaks Detected)"
        
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Amplitude (µV)")
        ax3.set_title(title)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def run(self, show_plots=False, save_results=True):
        """
        Run the full peak detection pipeline.

        Parameters:
        -----------
        show_plots : bool
            Whether to display the signal + peak plot
        save_results : bool
            Whether to write a summary to peaks.txt

        Returns:
        --------
        dict : results containing metadata and peak info
        """
        print(f"[PeakAnalyzer] Analyzing: {self.csv_path}")
        self._load_data()
        self._filter_data()
        self._detect_peaks()

        if save_results:
            self._save_results()

        if show_plots:
            set_browser_backend("matplotlib")
            self._plot()

        # Return results with peak times and amplitudes
        if len(self.peaks) > 0:
            peak_times = self.resampled_times[self.peaks]
            peak_amplitudes = self.envelope_resampled[self.peaks]
            
            # Find the highest peak
            max_peak_idx = np.argmax(peak_amplitudes)
            highest_peak_time = peak_times[max_peak_idx]
            highest_peak_amplitude = peak_amplitudes[max_peak_idx]
        else:
            peak_times = []
            peak_amplitudes = []
            highest_peak_time = None
            highest_peak_amplitude = None

        return {
            'csv_file_path': str(self.csv_path),
            'num_peaks': len(self.peaks),
            'peak_times': peak_times,
            'peak_amplitudes': peak_amplitudes,
            'highest_peak_time': highest_peak_time,
            'highest_peak_amplitude': highest_peak_amplitude,
            'sampling_rate': self.sampling_rate,
            'signal_duration': self.raw.times[-1],
            'raw_data': self.raw,
            'filtered_data': self.raw_filtered
        }

def test_with_sample_data():
    """Test the fixed analyzer with a sample from the combined dataset."""
    from pathlib import Path
    
    # Load a small sample from the combined dataset
    dataset_path = Path(__file__).parent.parent / "dataset" / "combined_emg_dorsiflex.csv"
    
    print("Testing with sample data...")
    
    # Read a small sample (first 1000 rows)
    df_sample = pd.read_csv(dataset_path, nrows=1000)
    
    # Save as temporary file
    temp_file = Path("temp_sample.csv")
    df_sample.to_csv(temp_file, index=False)
    
    try:
        # Test the analyzer
        analyzer = EMGPeakAnalyzerFixed(
            csv_path=temp_file,
            sampling_rate=220,
            height_percentile=95,  # Lower threshold for testing
            min_distance=1
        )
        
        results = analyzer.run(show_plots=True, save_results=True)
        
        print(f"\nTest Results:")
        print(f"Number of peaks: {results['num_peaks']}")
        if results['num_peaks'] > 0:
            print(f"Highest peak time: {results['highest_peak_time']:.3f}s")
            print(f"Highest peak amplitude: {results['highest_peak_amplitude']:.4f}µV")
        
    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()

if __name__ == "__main__":
    test_with_sample_data()
