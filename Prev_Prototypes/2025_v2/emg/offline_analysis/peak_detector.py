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
import os
import sys
from io import StringIO
warnings.filterwarnings('ignore')

# Suppress specific MNE warnings about channel info
warnings.filterwarnings('ignore', message='No data channels found')
warnings.filterwarnings('ignore', message='The highpass and lowpass values in the measurement info will not be updated')
warnings.filterwarnings('ignore', category=UserWarning, module='mne')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mne')

class PeakDetector:
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
        if 'EMG' in self.df_emg.columns:
            # New master dataset format
            emg_signal = self.df_emg['EMG'].dropna().values
            time_vector = self.df_emg['Time'].iloc[:len(emg_signal)].values
            print(f"Using 'EMG' column for EMG data")
        elif 'fwEMG 3' in self.df_emg.columns:
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
        #self.events = self.df_emg['event']
        
        print(f"EMG signal shape: {emg_signal.shape}")
        print(f"Time vector shape: {time_vector.shape}")
        print(f"EMG signal range: {emg_signal.min():.4f} to {emg_signal.max():.4f} µV")
        print(f"Time range: {time_vector.min():.2f} to {time_vector.max():.2f} seconds")
        
        # Create MNE Raw object
        info = create_info(ch_names=['EMG'], sfreq=self.sampling_rate, ch_types=['emg'])
        # Suppress MNE warnings and print statements during RawArray creation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Temporarily redirect stdout to suppress print statements
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                self.raw = RawArray(emg_signal.reshape(1, -1).astype(np.float64), info)
            finally:
                sys.stdout = old_stdout

    def _filter_data(self):
        """Apply low-pass filter to the EMG signal."""
        self.raw_filtered = self.raw.copy()
        # Use 80Hz cutoff for 200Hz sampling rate (well below Nyquist frequency of 100Hz)
        lowpass_freq = min(80, self.sampling_rate // 2 - 10)  # Ensure it's below Nyquist
        # Suppress MNE warnings and print statements during filtering
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Temporarily redirect stdout to suppress print statements
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                self.raw_filtered.filter(None, lowpass_freq, method="iir", phase="forward", picks="emg")
            finally:
                sys.stdout = old_stdout
        print(f"Applied low-pass filter ({lowpass_freq} Hz cutoff)")

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
        
        # Store envelope data for later use
        self.envelope = envelope
        
        # Calculate height threshold on original envelope
        height_threshold = np.percentile(envelope, self.height_percentile)
        print(f"Height threshold (percentile {self.height_percentile}): {height_threshold:.4f} µV")
        
        # Find the  event timestamp (where event = 1)
        event_indices = self._find_event_windows()
        
        if len(event_indices) < 2:
            print("Less than 2 event, can't create windows")
            
        # Detect peaks in each window
        self.peaks = self._detect_peaks_in_windows(envelope, event_indices)
        
        print(f"Detected {len(self.peaks)} peaks in {len(event_indices - 1)} windows")

        if len(self.peaks) > 0:
            peak_amplitudes = envelope[self.peaks]  # Use envelope values
            print(f"Peak amplitudes range: {peak_amplitudes.min():.4f} to {peak_amplitudes.max():.4f} µV")
               
        
        
        # Create trials 
        
        # Get EMG signal
        #data, times = self.raw_filtered[:, :]
        #emg_signal = data.squeeze()
        
        # Get markers
        #marker_times = times[self.peaks] 
        
        # Use window centered around each marker
        #window_duration = 6  # seconds
        #seg_len = int(window_duration * self.sampling_rate)  # Convert to samples
        
        #print(f"Using {window_duration*1000:.0f}ms window ({seg_len} samples) from markers {self.sampling_rate}Hz")
        
       
        #for peak_time in peak_times:
            # Convert time to sample index
        #    peak_idx = int(peak_time * self.sampling_rate)
            
            # Extract segment around peak (3 s window)
        #    start_idx = max(0, peak_idx - seg_len // 2)
        #    end_idx = min(len(emg_signal), peak_idx + seg_len // 2)
        #    segment = emg_signal[start_idx:end_idx]
                
        #for epoch in segments:
        
        
        # Detect peaks in the original envelope
        #self.peaks, properties = find_peaks(
        #    envelope,  # Use original envelope for peak detection
        #    height=height_threshold,
        #    distance=int(self.min_distance * self.sampling_rate)  # Use original sampling rate
        #)
    
    def _find_event_windows(self):
        """Find indices where colum events == 1"""
        
        # look for the event column
        event_col = None
        
        for col in self.df_emg.columns:
            if col.lower() in ['event','events']:
                event_col = col
                break 
        if event_col is None:
            raise ValueError("No event column founded on the dataset")
            
        event_mask = self.df_emg[event_col] == 1
        event_indices = np.where(event_mask)[0] 
        print(f"Found {len(event_indices)} events")
        
        if len(event_indices) > 0:
            event_times = self.time_vector[event_indices]
            print(f"Event times: {event_times[:5]}..." if len(event_times) > 5 else "Event times: {event_times}")
        
        return event_indices
    
    
    def _detect_peaks_in_windows(self, envelope, event_indices):
        """Detect highest peak in each window between consecutive events."""
        all_peaks = []
        
        # Create windows between consecutive events
        for i in range(len(event_indices) - 1):
            start_idx = event_indices[i]
            end_idx = event_indices[i + 1]
            
            # Extract window data
            window_envelope = envelope[start_idx:end_idx]
            window_times = self.times[start_idx:end_idx]
            
            if len(window_envelope) < 10:  # Skip very short windows
                continue
            
            # Find all peaks in this window with a lower threshold
            window_peaks, _ = find_peaks(
                window_envelope,
                height=np.percentile(window_envelope, 80),  # Lower threshold for window
                distance=int(self.min_distance * self.sampling_rate)  
            )
            
            if len(window_peaks) > 0:
                # Find the highest peak in this window
                window_peak_amplitudes = window_envelope[window_peaks]
                highest_peak_idx = window_peaks[np.argmax(window_peak_amplitudes)]
                
                # Convert back to global index
                global_peak_idx = start_idx + highest_peak_idx
                all_peaks.append(global_peak_idx)
                
                peak_time = window_times[highest_peak_idx]
                peak_amplitude = window_envelope[highest_peak_idx]
                print(f"Window {i+1}: Highest peak at {peak_time:.3f}s, amplitude {peak_amplitude:.4f}µV")
        
        return np.array(all_peaks)
            
            

    
            
        

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

        # Get event information
        event_indices = self._find_event_windows()
        
        with open(output_file, 'w') as f:
            f.write(f"EMG File: {relative_csv_path}\n")
            f.write(f"Output File: {relative_output_path.name}\n")
            f.write(f"Number of events: {len(event_indices)}\n")
            f.write(f"Number of event windows: {len(event_indices)-1 if len(event_indices) > 1 else 0}\n")
            f.write(f"Number of peaks detected: {len(self.peaks)}\n")
            f.write(f"Height percentile: {self.height_percentile}\n")
            f.write(f"Min distance: {self.min_distance} seconds\n")
            f.write(f"Sampling rate: {self.sampling_rate} Hz\n")
            f.write(f"Signal duration: {self.raw.times[-1]:.2f} seconds\n")
            
            if len(event_indices) > 0:
                f.write(f"\nEvent details:\n")
                for i, event_idx in enumerate(event_indices):
                    event_time = self.time_vector[event_idx]
                    f.write(f"Event {i+1}: Time={event_time:.3f}s\n")
            
            if len(self.peaks) > 0:
                f.write(f"\nPeak details (highest peak per event window):\n")
                for i, peak_idx in enumerate(self.peaks):
                    peak_time = self.times[peak_idx]
                    peak_amplitude = self.envelope[peak_idx]
                    f.write(f"Peak {i+1}: Time={peak_time:.3f}s, Amplitude={peak_amplitude:.4f}µV\n")
        
        print(f"[PeakAnalyzer] Results saved to: {relative_output_path}")
        
        
        

    def _plot(self):
        """Plot the EMG signal with detected peaks and event windows."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), layout="constrained")
        
        # Plot original signal with event markers
        ax1.plot(self.time_vector, self.emg_signal, label='Original EMG Signal', alpha=0.7)
        
        # Add event markers
        event_indices = self._find_event_windows()
        if len(event_indices) > 0:
            event_times = self.time_vector[event_indices]
            ax1.axvline(x=event_times[0], color='red', linestyle='--', alpha=0.7, label='Event Markers')
            for event_time in event_times[1:]:
                ax1.axvline(x=event_time, color='red', linestyle='--', alpha=0.7)
        
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude (µV)")
        ax1.set_title("Original EMG Signal with Event Markers")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot filtered and centered signal
        data_centered = self.raw_filtered.get_data().squeeze() - self.raw_filtered.get_data().squeeze().mean()
        ax2.plot(self.times, data_centered, label='Filtered EMG (centered)', alpha=0.7)
        
        # Add event markers
        if len(event_indices) > 0:
            event_times = self.time_vector[event_indices]
            for event_time in event_times:
                ax2.axvline(x=event_time, color='red', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude (µV)")
        ax2.set_title("Filtered and Centered EMG Signal with Event Windows")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot envelope with peaks and event windows
        ax3.plot(self.times, self.envelope, label='Signal Envelope', alpha=0.7, color='orange')
        
        # Add event windows (shaded regions)
        if len(event_indices) > 1:
            for i in range(len(event_indices) - 1):
                start_time = self.time_vector[event_indices[i]]
                end_time = self.time_vector[event_indices[i + 1]]
                ax3.axvspan(start_time, end_time, alpha=0.2, color='blue', 
                           label='Event Windows' if i == 0 else "")
        
        # Add event markers
        if len(event_indices) > 0:
            event_times = self.time_vector[event_indices]
            for event_time in event_times:
                ax3.axvline(x=event_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        if len(self.peaks) > 0:
            peak_times = self.times[self.peaks]
            peak_amplitudes = self.envelope[self.peaks]
            
            # Plot all peaks
            ax3.scatter(peak_times, peak_amplitudes, color='red', s=80, alpha=0.9, 
                       label=f'Detected Peaks ({len(self.peaks)})', zorder=5)
            
            # Highlight the highest peak
            max_peak_idx = np.argmax(peak_amplitudes)
            ax3.scatter(peak_times[max_peak_idx], peak_amplitudes[max_peak_idx], 
                       color='green', s=120, marker='*', edgecolors='black', linewidth=2,
                       label=f'Highest Peak: {peak_amplitudes[max_peak_idx]:.3f}µV', zorder=6)
            
            # Add peak count to title
            title = f"Signal Envelope with {len(self.peaks)} Peaks in Event Windows"
        else:
            title = "Signal Envelope (No Peaks Detected in Event Windows)"
        
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
            peak_times = self.times[self.peaks]
            peak_amplitudes = self.envelope[self.peaks]
            
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
