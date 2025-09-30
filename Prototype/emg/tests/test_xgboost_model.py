#!/usr/bin/env python3
"""
Test XGBoost Model with Test Data
=================================================

This script loads the trained XGBoost model and tests it with test_set_ch1_200Hz.csv data.
It processes the raw EMG data to extract the same features used during training and makes predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
import time
from scipy import signal
import neurokit2 as nk
warnings.filterwarnings('ignore')

class XGBoostTester:
    """Test the trained XGBoost model with new data."""

    def __init__(self, model_path=None):
        """Initialize the tester with a trained model."""
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = ['Peak_Amplitude', 'Min_Peak_Amplitude', 'Mean_Frequency', 'Median_Frequency']
        
        if model_path is None:
            model_path = Path(__file__).parent / "xgboost_model.pkl"
        
        if Path(model_path).exists():
            self.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model not found at: {model_path}")

    def load_model(self, model_path):
        """Load the trained XGBoost model."""
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded successfully!")
        print(f"Feature names: {self.feature_names}")
        print(f"Available classes: {self.label_encoder.classes_}")

    def extract_features_from_raw_data(self, df, window_duration=3.0):
        """
        Extract features from raw EMG data using the same preprocessing pipeline as training.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw EMG data with columns: Time, EMG, Subject, MVC, Trial
        window_duration : float
            Duration of window around peaks for frequency analysis (seconds)
            
        Returns:
        --------
        list : List of feature dictionaries for each trial
        """
        features_list = []
        sampling_rate = 200  # Hz
        
        # Group by Subject, MVC, and Trial
        for (subject, mvc, trial), group in df.groupby(['Subject', 'MVC', 'Trial']):
            if len(group) == 0:
                continue
                
            print(f"Processing: Subject={subject}, MVC={mvc}, Trial={trial}")
            
            # Get EMG signal and time
            emg_signal = group['EMG'].values
            time_signal = group['Time'].values
            
            # Step 1: Apply low-pass filter (80Hz cutoff)
            from scipy.signal import butter, filtfilt
            nyquist = sampling_rate / 2
            lowpass_freq = min(80, nyquist - 10)  # 80Hz cutoff, ensure below Nyquist
            b, a = butter(4, lowpass_freq / nyquist, btype='low')
            filtered_signal = filtfilt(b, a, emg_signal)
            print(f"  Applied low-pass filter ({lowpass_freq} Hz cutoff)")
            
            # Step 2: Remove DC offset (center the signal)
            filtered_centered = filtered_signal - filtered_signal.mean()
            
            # Step 3: Extract envelope using Hilbert transform
            from scipy.signal import hilbert
            analytic_signal = hilbert(filtered_centered)
            envelope = np.abs(analytic_signal)
            print(f"  Extracted signal envelope using Hilbert transform")
            
            # Step 4: Resample envelope to 200Hz (if needed)
            target_sampling_rate = 200
            current_sampling_rate = sampling_rate
            
            if current_sampling_rate != target_sampling_rate:
                resample_factor = target_sampling_rate / current_sampling_rate
                new_length = int(len(envelope) * resample_factor)
                envelope_resampled = signal.resample(envelope, new_length)
                resampled_times = np.linspace(time_signal[0], time_signal[-1], new_length)
                print(f"  Resampled envelope from {current_sampling_rate}Hz to {target_sampling_rate}Hz")
            else:
                envelope_resampled = envelope
                resampled_times = time_signal
                print(f"  No resampling needed (already at {target_sampling_rate}Hz)")
            
            # Step 5: Detect peaks in the resampled envelope
            height_threshold = np.percentile(envelope_resampled, 95)  # 95th percentile threshold (same as training)
            min_distance_samples = int(2 * target_sampling_rate)  # 2 seconds minimum distance (same as training)
            
            peaks, properties = signal.find_peaks(
                envelope_resampled,
                height=height_threshold,
                distance=min_distance_samples
            )
            
            print(f"  Detected {len(peaks)} peaks in resampled envelope")
            
            if len(peaks) == 0:
                # No peaks found, use maximum value
                max_idx = np.argmax(envelope_resampled)
                peak_amplitudes = [envelope_resampled[max_idx]]
                peak_times = [resampled_times[max_idx]]
            else:
                peak_amplitudes = envelope_resampled[peaks]
                peak_times = resampled_times[peaks]
            
            # Extract features
            peak_amplitude = np.max(peak_amplitudes) if len(peak_amplitudes) > 0 else np.nan
            min_peak_amplitude = np.min(peak_amplitudes) if len(peak_amplitudes) > 0 else np.nan
            
            # Extract frequency features from original filtered signal (not envelope)
            mean_freqs = []
            median_freqs = []
            
            seg_len = int(window_duration * sampling_rate)
            
            for peak_time in peak_times:
                # Convert time to sample index in original signal
                peak_idx = int((peak_time - time_signal[0]) * sampling_rate)
                
                # Extract segment around peak from the original filtered signal
                start_idx = max(0, peak_idx - seg_len // 2)
                end_idx = min(len(filtered_centered), peak_idx + seg_len // 2)
                segment = filtered_centered[start_idx:end_idx]
                
                if len(segment) > 0:
                    try:
                        # Use NeuroKit2 for frequency analysis
                        psd = nk.signal_psd(segment, sampling_rate=sampling_rate)
                        
                        if isinstance(psd, pd.DataFrame) and len(psd) > 0 and 'Frequency' in psd.columns and 'Power' in psd.columns:
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
                            
                            if not np.isnan(mean_freq) and not np.isnan(median_freq):
                                mean_freqs.append(mean_freq)
                                median_freqs.append(median_freq)
                                
                    except Exception as e:
                        print(f"  Frequency analysis failed for peak at {peak_time:.3f}s: {e}")
                        continue
            
            # Calculate average frequency features
            mean_frequency = np.mean(mean_freqs) if len(mean_freqs) > 0 else 50.0  # Default value
            median_frequency = np.mean(median_freqs) if len(median_freqs) > 0 else 45.0  # Default value
            
            # Create feature vector
            feature_vector = [
                peak_amplitude,
                min_peak_amplitude,
                mean_frequency,
                median_frequency
            ]
            
            features_dict = {
                'Subject': subject,
                'MVC': mvc,
                'Trial': trial,
                'features': feature_vector,
                'peak_amplitude': peak_amplitude,
                'min_peak_amplitude': min_peak_amplitude,
                'mean_frequency': mean_frequency,
                'median_frequency': median_frequency,
                'num_peaks': len(peaks)
            }
            
            features_list.append(features_dict)
            
            print(f"  Extracted features: Peak={peak_amplitude:.4f}, Min={min_peak_amplitude:.4f}, "
                  f"MeanFreq={mean_frequency:.2f}Hz, MedianFreq={median_frequency:.2f}Hz")
        
        return features_list

    def preprocess_features(self, features_list):
        """Preprocess features for prediction."""
        if not features_list:
            return np.array([])
        
        # Extract feature vectors
        X = np.array([f['features'] for f in features_list])
        
        # Handle NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"Found {nan_count} NaN values in features")
            
            for i, feature_name in enumerate(self.feature_names):
                if 'amplitude' in feature_name.lower():
                    # For amplitude features, use 0.0 as default
                    X[np.isnan(X[:, i]), i] = 0.0
                elif 'frequency' in feature_name.lower():
                    # For frequency features, use typical EMG frequency values
                    if 'mean' in feature_name.lower():
                        X[np.isnan(X[:, i]), i] = 50.0
                    elif 'median' in feature_name.lower():
                        X[np.isnan(X[:, i]), i] = 45.0
                    else:
                        X[np.isnan(X[:, i]), i] = 50.0
                else:
                    X[np.isnan(X[:, i]), i] = 0.0
        
        # Check for infinite values
        inf_count = np.isinf(X).sum()
        if inf_count > 0:
            print(f"Replacing {inf_count} infinite values...")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X

    def predict(self, X):
        """Make predictions on preprocessed features."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Convert back to original labels
        predictions_original = self.label_encoder.inverse_transform(predictions)
        
        return predictions_original, probabilities

    def test_with_data(self, test_data_path):
        """Test the model with the provided test data."""
        print("=" * 60)
        print("Testing XGBoost Model")
        print("=" * 60)
        
        # Load test data
        print(f"Loading test data from: {test_data_path}")
        df = pd.read_csv(test_data_path)
        print(f"Test data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Extract features
        print("\nExtracting features from test data...")
        features_list = self.extract_features_from_raw_data(df)
        
        if not features_list:
            print("No features extracted. Check the test data format.")
            return
        
        print(f"Extracted features for {len(features_list)} trials")
        
        # Preprocess features
        X = self.preprocess_features(features_list)
        
        if len(X) == 0:
            print("No valid features after preprocessing.")
            return
        
        print(f"Feature matrix shape: {X.shape}")
        
        # Make predictions
        print("\nMaking predictions...")
        predictions, probabilities = self.predict(X)
        
        # Create results DataFrame
        results_data = []
        for i, features_dict in enumerate(features_list):
            results_data.append({
                'Subject': features_dict['Subject'],
                'MVC': features_dict['MVC'],
                'Trial': features_dict['Trial'],
                'Predicted_MVC': predictions[i],
                'Peak_Amplitude': features_dict['peak_amplitude'],
                'Min_Peak_Amplitude': features_dict['min_peak_amplitude'],
                'Mean_Frequency': features_dict['mean_frequency'],
                'Median_Frequency': features_dict['median_frequency'],
                'Num_Peaks': features_dict['num_peaks'],
                'Probability_10': probabilities[i, 0] if len(probabilities[i]) > 0 else np.nan,
                'Probability_25': probabilities[i, 1] if len(probabilities[i]) > 1 else np.nan,
                'Probability_50': probabilities[i, 2] if len(probabilities[i]) > 2 else np.nan
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Save results
        output_path = Path(__file__).parent / "test_results_xgboost.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("PREDICTION SUMMARY")
        print("=" * 60)
        
        # Count predictions by class
        prediction_counts = results_df['Predicted_MVC'].value_counts().sort_index()
        print("\nPrediction Distribution:")
        for mvc, count in prediction_counts.items():
            print(f"  MVC {mvc}: {count} predictions")
        
        # Show some example predictions
        print("\nExample Predictions:")
        print(results_df[['Subject', 'MVC', 'Trial', 'Predicted_MVC', 'Probability_10', 'Probability_25', 'Probability_50']].head(10))
        
        # Calculate accuracy if true labels are available
        if 'MVC' in results_df.columns:
            correct_predictions = (results_df['MVC'] == results_df['Predicted_MVC']).sum()
            total_predictions = len(results_df)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            print(f"\nAccuracy: {correct_predictions}/{total_predictions} = {accuracy:.4f}")
            
            # Confusion matrix
            print("\nConfusion Matrix:")
            confusion = pd.crosstab(results_df['MVC'], results_df['Predicted_MVC'], margins=True)
            print(confusion)
        
        return results_df

def main():
    """Main function to test the XGBoost model."""
    # Initialize tester
    tester = XGBoostTester()
    
    # Test with the provided data
    test_data_path = Path(__file__).parent.parent / "dataset" / "test_set_ch1_200Hz.csv"
    
    if not test_data_path.exists():
        print(f"Test data not found at: {test_data_path}")
        return
    
    # Run the test
    results = tester.test_with_data(test_data_path)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
