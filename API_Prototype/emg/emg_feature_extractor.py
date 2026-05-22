import pandas as pd
import numpy as np
import neurokit2 as nk
import warnings
import os
from time import time
from typing import Dict, List, Optional, Tuple
from mne import create_info
from mne.io import RawArray
from pathlib import Path
from scipy.signal import find_peaks, hilbert
from config.connection_manager import logging
from config.config_manager import load_config

CONFIG = load_config()

warnings.filterwarnings('ignore')

class FeatureExtractor:
    """
    Clean and organized EMG feature extraction pipeline.
    
    Handles:
    - Data loading (requires 'event' column)
    - Signal preprocessing (filtering, DC offset removal, envelope extraction)
    - Event-based peak detection (event occurs in the middle of a trial)    
    - Feature extraction (amplitude and frequency features using NeuroKit2)
    - Optional tangential acceleration calculation from accelerometer data
    """
    
    def __init__(self, sample_rate, profiler, single_trial_analysis, output_path: Path):
        self.sampling_rate = sample_rate
        self.profiler = profiler
        self.output_path = output_path
        self.single_trial_analysis = single_trial_analysis
        
        self.height_percentile = CONFIG.get("height_percentile")
        self.min_distance = CONFIG.get("min_distance") # safety for minimum distance between peaks (To-do: Figure out why hardcoded to 3s)
        self.lowpass_cutoff = min(CONFIG.get("feature_cutoff_freq"), self.sampling_rate // 2 - 10)  # Ensure below Nyquist
        self.window_duration = CONFIG.get("feature_window") # window of feature extraction (currently 5s)
        self.window_samples = int(self.window_duration * self.sampling_rate) 

        # if we are doing ONE dorsiflexion per analysis/extraction, CSV logic is slightly different (for space efficiency)
        if self.single_trial_analysis:
            self.output_file = self.output_path / f"all_peak_features.csv"    
    
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
    
    def detect_peaks(self, envelope: np.ndarray, event_column: np.ndarray, timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Event-based peak detection: find highest peak in each trial between events.
        """
        
        # Get indices of all events
        event_indices = np.where(event_column == 1)[0]
        peak_indices = []
        peak_amplitudes = [] 

        # Iterate through each event to define the window
        for event_idx in event_indices:
            
            # define window: -2/+3 seconds from event marker timestamp
            event_time = timestamps[event_idx]
            start_time = event_time - (self.window_duration // 2)
            end_time = event_time + (self.window_duration // 2 + 1)
            
            # Find the nearest indices for the window boundaries
            start_idx = np.searchsorted(timestamps, start_time, side='left')
            end_idx = np.searchsorted(timestamps, end_time, side='right')
            
            # Ensure the slice is within the bounds of the envelope array
            start_idx = max(0, start_idx)
            end_idx = min(len(envelope), end_idx)
            
            trial_data = envelope[start_idx:end_idx]
            
            # Find peak in this trial
            if len(trial_data) > 0:
                local_peaks, _ = find_peaks(trial_data, height=np.percentile(trial_data, self.height_percentile), distance=int(self.min_distance * self.sampling_rate))
                # To-do: Figure out why we are doing this extra check of 3 second window (which would give 2 local peaks for each trial)
                
            if len(local_peaks) > 0: # Get the highest peak in this trial
                local_peak_amplitudes = trial_data[local_peaks]
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
            logging.error(f"[Extractor] Error calculating frequency features with NeuroKit2: {e}")
            return np.nan, np.nan
    
    def extract_features(self, curr_trial: int, emg_signal: np.ndarray, peak_indices: np.ndarray, peak_amplitudes: np.ndarray, timestamps: np.ndarray
    ) -> List[Dict[str, float]]:
        """
        Extract features from EMG signal for each detected peak. 
        """
        if len(peak_indices) == 0:
            return []
        
        # Extract features for each peak
        peak_features = []
        
        for i, (peak_idx, peak_amplitude) in enumerate(zip(peak_indices, peak_amplitudes)):
            # Get timestamp for this peak
            peak_timestamp = timestamps[peak_idx] if peak_idx < len(timestamps) else np.nan
            
            # Extract window centered around peak
            start_idx = max(0, peak_idx - self.window_samples // 2)
            end_idx = min(len(emg_signal), peak_idx + self.window_samples // 2)
            segment = emg_signal[start_idx:end_idx]
            
            '''# Calculate frequency features for this peak
            if len(segment) > 0:
                mean_freq, median_freq = self.calculate_frequency_features(segment)
            else:
                mean_freq, median_freq = np.nan, np.nan'''
            
            # Create feature dictionary for this peak
            peak_feature = {
                'peak_id': i + 1,
                'timestamp': float(peak_timestamp),
                'max_amplitude': float(peak_amplitude)
                #'mean_frequency': float(mean_freq) if not np.isnan(mean_freq) else np.nan,
                #'median_frequency': float(median_freq) if not np.isnan(median_freq) else np.nan
            }

            # Add peak information to profiler (subtract 1 because first trial is 'trial 0', no electrical stimulation)
            if (curr_trial - 1 != 0):
                self.profiler.log_metric((curr_trial - 1), "peak_time", float(peak_timestamp))
                self.profiler.log_metric((curr_trial - 1), "peak_amp", float(peak_amplitude))
            
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
    
    def process_channel(self, curr_trial: int, emg_signal: np.ndarray, timestamps: np.ndarray, event_column: np.ndarray, channel_name: str = "unknown", 
                        accel_x: Optional[np.ndarray] = None, accel_y: Optional[np.ndarray] = None, accel_z: Optional[np.ndarray] = None
    ) -> List[Dict[str, float]]:
        """
        Process a single EMG channel and extract features for each peak.
        """
        # Preprocess signal
        filtered_signal, envelope = self.preprocess_signal(emg_signal)
        
        # Detect peaks using event-based detection
        peak_indices, peak_amplitudes = self.detect_peaks(envelope, event_column, timestamps)        
        
        # Extract features for each peak
        peak_features = self.extract_features(curr_trial, filtered_signal, peak_indices, peak_amplitudes, timestamps)
        
        # Add tangential acceleration to each peak feature
        #tangential_acc = self.calculate_tangential_acceleration(accel_x, accel_y, accel_z)
        
        # Add channel name
        for peak_feature in peak_features:
            peak_feature['channel'] = channel_name
            #peak_feature['tangential_acceleration'] = float(tangential_acc)
        
        return peak_features
    
    def _save_results_txt(self, output_path: Path, peak_features: List[Dict[str, float]]):
        """
        Save classification results to TXT file with specified structure
        """

        output_file = output_path / "features.txt"

        # Extract vectors from all results
        amplitude_vector = [peak_feature['max_amplitude'] for peak_feature in peak_features]
        #mean_frequency_vector = [peak_feature['mean_frequency'] for peak_feature in peak_features]
        #median_frequency_vector = [peak_feature['median_frequency'] for peak_feature in peak_features]

        # Write header
        with open(output_file, 'w') as f:
            f.write("sujet;max_amplitude\n") #;mean_frequency;median_frequency\n")

            # Write data as row vectors
            sujet = 0  # Always 0 as specified
            
            # Format vectors as strings with proper precision
            amplitude_str = ",".join([f"{val:.4f}" for val in amplitude_vector])
            #mean_frequency_str = ",".join([f"{val:.2f}" for val in mean_frequency_vector])
            #median_frequency_str = ",".join([f"{val:.2f}" for val in median_frequency_vector])
            
            f.write(f"{sujet};{amplitude_str}\n") #;{mean_frequency_str};{median_frequency_str}\n")
        
        #logging.info(f"[Extractor] Peak feature results saved to: {output_file}")
    
    def _run_feature_extraction(self, curr_trial: int, analysis_df: pd.DataFrame, output_path: Path, curr_timestamp: int, channels: Optional[List[str]] = None
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
        '''accel_x = df['accel_x'].values 
        accel_y = df['accel_y'].values 
        accel_z = df['accel_z'].values'''
        event_column = df['event'].values
        
        num_events = np.sum(event_column == 1)
        if num_events < 1:
            logging.warning(f"[Extractor] At least 1 event required for peak detection. Found {num_events} events.")
            return None
        
        # Process each channel and collect all peak features
        all_peak_features = []
        
        for channel in channels:
            if channel not in df.columns:
                logging.warning(f"[Extractor] Channel {channel} not found in data. Skipping.")
                continue
            
            #logging.info(f"[Extractor] Processing {channel}...")
            emg_signal = df[channel].values
            
            # Process channel
            peak_features = self.process_channel(curr_trial, emg_signal, timestamps, event_column, channel) #, accel_x, accel_y, accel_z)
            
            all_peak_features.extend(peak_features)
            
            # Print summary
            #if peak_features:
                #logging.info(f"[Extractor] Detected {len(peak_features)} peaks in {channel}")
        
        # Create DataFrame from all peak features
        if not all_peak_features:
            logging.warning("[Extractor] No peaks detected in any channel")
            return None # optimization + stimulation will be skipped for this trial
        features_df = pd.DataFrame(all_peak_features)
        
        # Save to features.txt (triggers GPBO)
        self._save_results_txt(self.output_path, all_peak_features)

        # Save to CSV 
        if self.single_trial_analysis: # append single row from dataframe to same CSV file each extraction
            file_exists = os.path.isfile(self.output_file)
            features_df.to_csv(self.output_file, mode='a', index=False, header=not file_exists)
        else: # save features dataframe as a new CSV file each extraction 
            output_file = self.output_path / f"{curr_timestamp}peak_features.csv"
            features_df.to_csv(output_file, index=False)
        
        return features_df

    def run(self, curr_trial, analysis_df: pd.DataFrame, curr_timestamp: int, channels: Optional[List[str]] = None
    ) -> pd.DataFrame:
        start_time = time()
        
        features_df = self._run_feature_extraction(curr_trial, analysis_df, curr_timestamp, channels)
        # message = f"[Extractor] Feature extractor completed. Total peaks: {len(features_df)}. Duration: {time() - start_time:.2f} seconds."
        # logging.info(message)

        # add to log if feature extraction successful
        if features_df is not None:
            duration = time() - start_time
            self.profiler.log_metric(curr_trial, "feat_extract", duration)

        return features_df