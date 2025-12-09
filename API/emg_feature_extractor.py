import pandas as pd
import numpy as np
import neurokit2 as nk
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from mne import create_info
from mne.io import RawArray
from scipy.signal import find_peaks, hilbert
from connection_manager import logging

warnings.filterwarnings('ignore')

class FeatureExtractor:
    """
    Clean and organized EMG feature extraction pipeline.
    
    Handles:
    - Data loading (requires 'event' column)
    - Signal preprocessing (filtering, DC offset removal, envelope extraction)
    - Event-based peak detection (requires at least 2 events)
    - Feature extraction (amplitude and frequency features using NeuroKit2)
    - Optional tangential acceleration calculation from accelerometer data
    """
    
    def __init__(self, sampling_rate=200, height_percentile=98.0, min_distance=3.0, lowpass_cutoff=80.0, window_duration=1.0):
        self.sampling_rate = sampling_rate
        self.height_percentile = height_percentile
        self.min_distance = min_distance # safety for minimum distance between peaks (To-do: Figure out why hardcoded to 3s)
        self.lowpass_cutoff = min(lowpass_cutoff, sampling_rate // 2 - 10)  # Ensure below Nyquist
        self.window_duration = window_duration # window of feature extraction (changed from 3 to 1 minutes)
        self.window_samples = int(window_duration * sampling_rate)
    
    def load_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Validate required columns
        if 'timestamp' not in df.columns:
            raise ValueError("'timestamp' column is required in CSV file for event-based peak detection")
        
        if 'event' not in df.columns:
            raise ValueError("'event' column is required in CSV file for event-based peak detection")
        
        emg_columns = [col for col in df.columns if col.startswith('ch') and 'µV' in col]
        if len(emg_columns) == 0:
            raise ValueError("No EMG channels found. Expected columns like 'ch1 (µV)', 'ch2 (µV)', etc.")
        
        return df
    
    def preprocess_signal(self, emg_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess EMG signal: filtering, DC offset removal, and envelope extraction.
        """
        # Create MNE info object
        info = create_info(ch_names=['EMG'], sfreq=self.sampling_rate, ch_types=['emg'])
        
        # Create MNE Raw object
        raw = RawArray(emg_signal.reshape(1, -1), info, verbose=False)
        
        # Apply low-pass filter
        raw_filtered = raw.copy()
        raw_filtered.filter(l_freq=None, h_freq=self.lowpass_cutoff, picks='emg', verbose=False)
        
        # Extract filtered data
        data, _ = raw_filtered[:, :]
        data = data.squeeze()
        
        # Remove DC offset
        data_centered = data - data.mean()
        
        # Extract envelope using Hilbert transform
        analytic_signal = hilbert(data_centered)
        envelope = np.abs(analytic_signal)
        
        return data_centered, envelope
    
    def detect_peaks(self, envelope: np.ndarray, event_column: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Event-based peak detection: find highest peak in each trial between events.
        """
        event_indices = np.where(event_column == 1)[0]
        peak_indices = []
        peak_amplitudes = []
        
        # Find highest peak in each trial between consecutive events
        for i in range(len(event_indices) - 1):
            start_idx = event_indices[i]
            end_idx = event_indices[i + 1]
            trial = envelope[start_idx:end_idx]
            
            if len(trial) > 0:
                # Find peak in this trial
                # To-do: Figure out why we are doing this extra check of 3 second window (which would give 2 local peaks for each trial)
                local_peaks, _ = find_peaks(trial, height=np.percentile(trial, self.height_percentile), distance=int(self.min_distance * self.sampling_rate))
                
                if len(local_peaks) > 0: # Get the highest peak in this trial
                    local_peak_amplitudes = trial[local_peaks]
                    highest_local_peak_idx = local_peaks[np.argmax(local_peak_amplitudes)]
                    global_peak_idx = start_idx + highest_local_peak_idx
                    
                    peak_indices.append(global_peak_idx)
                    peak_amplitudes.append(envelope[global_peak_idx])
        
        return np.array(peak_indices), np.array(peak_amplitudes)
    
    def calculate_frequency_features(self, signal_segment: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate mean and median frequency from a signal segment using NeuroKit2.
        
        """
        if len(signal_segment) == 0:
            return np.nan, np.nan
        
        try:
            # Use NeuroKit2 for frequency analysis
            psd = nk.signal_psd(signal_segment, sampling_rate=self.sampling_rate)
            
            if isinstance(psd, pd.DataFrame) and len(psd) > 0:
                if 'Frequency' in psd.columns and 'Power' in psd.columns:
                    frequencies = psd['Frequency'].values
                    power = psd['Power'].values
                    
                    # Calculate mean frequency
                    mean_freq = np.sum(frequencies * power) / np.sum(power) if np.sum(power) > 0 else np.nan
                    
                    # Calculate median frequency
                    cumsum_power = np.cumsum(power)
                    if cumsum_power[-1] > 0:
                        median_idx = np.argmin(np.abs(cumsum_power - 0.5 * cumsum_power[-1]))
                        median_freq = frequencies[median_idx]
                    else:
                        median_freq = np.nan
                    
                    return mean_freq, median_freq
                else:
                    raise ValueError("NeuroKit2 PSD output missing required columns 'Frequency' or 'Power'")
            else:
                raise ValueError("NeuroKit2 PSD returned invalid format")
            
        except Exception as e:
            logging.error(f"Error calculating frequency features with NeuroKit2: {e}")
            return np.nan, np.nan
    
    def extract_features(self, emg_signal: np.ndarray, peak_indices: np.ndarray, peak_amplitudes: np.ndarray, timestamps: np.ndarray
    ) -> List[Dict[str, float]]:
        """
        Extract features from EMG signal for each detected peak. 
        """
        if len(peak_indices) == 0:
            return []
        
        # Calculate min peak amplitude (same for all peaks in this trial)
        min_peak_amplitude = np.min(peak_amplitudes) if len(peak_amplitudes) > 0 else np.nan
        
        # Extract features for each peak
        peak_features = []
        
        for i, (peak_idx, peak_amplitude) in enumerate(zip(peak_indices, peak_amplitudes)):
            # Get timestamp for this peak
            peak_timestamp = timestamps[peak_idx] if peak_idx < len(timestamps) else np.nan
            
            # Extract window centered around peak
            start_idx = max(0, peak_idx - self.window_samples // 2)
            end_idx = min(len(emg_signal), peak_idx + self.window_samples // 2)
            segment = emg_signal[start_idx:end_idx]
            
            # Calculate frequency features for this peak
            if len(segment) > 0:
                mean_freq, median_freq = self.calculate_frequency_features(segment)
            else:
                mean_freq, median_freq = np.nan, np.nan
            
            # Create feature dictionary for this peak
            peak_feature = {
                'peak_id': i + 1,
                'timestamp': float(peak_timestamp),
                'amplitude': float(peak_amplitude),
                'min_amplitude': float(min_peak_amplitude),
                'mean_frequency': float(mean_freq) if not np.isnan(mean_freq) else np.nan,
                'median_frequency': float(median_freq) if not np.isnan(median_freq) else np.nan
            }
            
            peak_features.append(peak_feature)
        
        return peak_features
    
    def calculate_tangential_acceleration(self, accel_x: np.ndarray, accel_y: np.ndarray, accel_z: np.ndarray
    ) -> float:
        """
        Calculate mean tangential acceleration from accelerometer data.
        """
        if len(accel_x) == 0 or len(accel_y) == 0 or len(accel_z) == 0:
            return np.nan
        
        # Calculate magnitude: sqrt(x^2 + y^2 + z^2)
        magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # Return mean tangential acceleration
        return float(np.nanmean(magnitude))
    
    def process_channel(self, emg_signal: np.ndarray, timestamps: np.ndarray, event_column: np.ndarray, channel_name: str = "unknown", 
                        accel_x: Optional[np.ndarray] = None, accel_y: Optional[np.ndarray] = None, accel_z: Optional[np.ndarray] = None
    ) -> List[Dict[str, float]]:
        """
        Process a single EMG channel and extract features for each peak.
        """
        # Preprocess signal
        filtered_signal, envelope = self.preprocess_signal(emg_signal)
        
        # Detect peaks using event-based detection
        peak_indices, peak_amplitudes = self.detect_peaks(envelope, event_column)
        
        # Extract features for each peak
        peak_features = self.extract_features(filtered_signal, peak_indices, peak_amplitudes, timestamps)
        
        # Add channel name and tangential acceleration to each peak feature
        tangential_acc = self.calculate_tangential_acceleration(accel_x, accel_y, accel_z)
        
        for peak_feature in peak_features:
            peak_feature['channel'] = channel_name
            peak_feature['tangential_acceleration'] = float(tangential_acc)
        
        return peak_features
    
    def run(self, analysis_df: pd.DataFrame, channels: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Process all EMG channels from a DataFrame and extract features for each peak.
        
        Returns: pd.DataFrame containing features for all detected peaks (one row per peak)
        """
        # Load data
        df = self.load_data(analysis_df)
        
        # Get EMG channels
        if channels is None:
            channels = [col for col in df.columns if col.startswith('ch') and 'µV' in col]
        
        # Get info from DataFrame
        timestamps = df['timestamp'].values
        accel_x = df['accel_x'].values 
        accel_y = df['accel_y'].values 
        accel_z = df['accel_z'].values
        event_column = df['event'].values
        
        num_events = np.sum(event_column == 1)
        if num_events < 2:
            logging.warning(f"At least 2 events required for peak detection. Found {num_events} events.")
            return None
        logging.info(f"Starting peak detection: {num_events} events detected")
        
        # Process each channel and collect all peak features
        all_peak_features = []
        
        for channel in channels:
            if channel not in df.columns:
                logging.warning(f"Channel {channel} not found in data. Skipping.")
                continue
            
            logging.info(f"Processing {channel}...")
            emg_signal = df[channel].values
            
            # Process channel
            peak_features = self.process_channel(emg_signal, timestamps, event_column, channel, accel_x, accel_y, accel_z)
            
            all_peak_features.extend(peak_features)
            
            # Print summary
            if peak_features:
                logging.info(f"Detected {len(peak_features)} peaks")
        
        # Create DataFrame from all peak features
        if not all_peak_features:
            logging.warning("No peaks detected in any channel")
            return None
        
        features_df = pd.DataFrame(all_peak_features)
        
        logging.info(f"Total peaks: {len(features_df)}")
        
        return features_df