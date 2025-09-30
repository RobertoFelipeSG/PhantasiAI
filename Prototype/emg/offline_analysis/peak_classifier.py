#!/usr/bin/env python3
"""
EMG Peak Classifier with XGBoost
================================

This module extends the EMG peak analyzer to include XGBoost classification
of detected peaks. It processes recorded EMG data, detects peaks, and
classifies them using a pre-trained XGBoost model.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Check if NeuroKit2 is available
try:
    import neurokit2 as nk
    NEUROKIT2_AVAILABLE = True
    print("NeuroKit2 available for frequency analysis")
except ImportError:
    NEUROKIT2_AVAILABLE = False
    print("NeuroKit2 not available. Please install with: pip install neurokit2")

# Check if SciPy is available for fallback
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
    print("SciPy available for fallback frequency analysis")
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available. Please install with: pip install scipy")

from .peak_detector import PeakDetector
from .xgboost_classifier import XGBoostClassifier

class PeakClassifier(PeakDetector):
    """
    EMG peak analyzer with XGBoost classification capabilities.
    
    This class extends the basic peak analyzer to classify detected peaks
    using a pre-trained XGBoost model.
    """
    
    def __init__(self, csv_path, model_path=None, sampling_rate=200, 
                 height_percentile=95, min_distance=2):
        """
        Initialize the EMG peak classifier with XGBoost.
        
        Parameters:
        -----------
        csv_path : str or Path
            Path to the EMG CSV file
        model_path : str or Path, optional
            Path to the pre-trained XGBoost model (.pkl file)
        sampling_rate : int
            Sampling rate of EMG data (Hz) - default 200Hz
        height_percentile : float
            Threshold for peak detection (percentile of signal amplitude) - default 95
        min_distance : float
            Minimum time (in seconds) between peaks - default 2
        """
        super().__init__(csv_path, sampling_rate, height_percentile, min_distance)
        
        # Initialize XGBoost classifier
        self.classifier = None
        self.classification_results = []
        
        if model_path:
            self.load_model(model_path)
        else:
            # Try to find the default model in multiple locations
            possible_model_paths = [
                Path(__file__).parent / "xgboost_model.pkl",  # offline_analysis directory
                Path(__file__).parent.parent / "models" / "xgboost_model.pkl",  # models directory
                Path(__file__).parent.parent / "xgboost_model.pkl",  # emg root directory
                Path(__file__).parent.parent.parent / "emg" / "models" / "xgboost_model.pkl",  # parent emg/models
                Path(__file__).parent.parent.parent / "emg" / "xgboost_model.pkl",  # parent emg root
            ]
            
            model_found = False
            for model_path in possible_model_paths:
                if model_path.exists():
                    self.load_model(model_path)
                    model_found = True
                    break
            
            if not model_found:
                print("Warning: No XGBoost model found. Classification will be skipped.")
                print("Searched in the following locations:")
                for path in possible_model_paths:
                    print(f"  - {path}")
    
    def load_model(self, model_path):
        """
        Load a pre-trained XGBoost model.
        
        Parameters:
        -----------
        model_path : str or Path
            Path to the .pkl file containing the trained model
        """
        try:
            self.classifier = XGBoostClassifier()
            self.classifier.load_model(model_path)
            print(f"XGBoost model loaded successfully from: {model_path}")
            print(f"Available classes: {self.classifier.label_encoder.classes_}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.classifier = None
            return False
    
    def extract_peak_features(self, peak_times, peak_amplitudes):
        """
        Extract enhanced features for each peak including frequency features.
        Uses a 20ms window centered around each peak and PySiology toolkit for feature calculation.
        Uses the same approach as the batch analysis for consistency.
        
        Parameters:
        -----------
        peak_times : array-like
            Times of detected peaks
        peak_amplitudes : array-like
            Amplitudes of detected peaks
            
        Returns:
        --------
        array : Feature matrix with shape (n_peaks, n_features)
        """
        if len(peak_times) == 0:
            return np.array([])
        
        # Get EMG signal
        data, times = self.raw_filtered[:, :]
        emg_signal = data.squeeze()
        
        # Extract minimum peak amplitude (same for all peaks in this trial)
        min_peak_amplitude = np.min(peak_amplitudes) if len(peak_amplitudes) > 0 else np.nan
        
        # Extract frequency features from segments around peaks
        mean_freqs = []
        median_freqs = []
        
        # Use window centered around each peak
        window_duration = 3  # 3 seconds
        seg_len = int(window_duration * self.sampling_rate)  # Convert to samples
        
        print(f"Using {window_duration*1000:.0f}ms window ({seg_len} samples) centered around peaks at {self.sampling_rate}Hz")
        
        # Use the global NeuroKit2 and SciPy availability flags
        
        for peak_time in peak_times:
            # Convert time to sample index
            peak_idx = int(peak_time * self.sampling_rate)
            
            # Extract segment around peak (3 s window)
            start_idx = max(0, peak_idx - seg_len // 2)
            end_idx = min(len(emg_signal), peak_idx + seg_len // 2)
            segment = emg_signal[start_idx:end_idx]
            
            if len(segment) > 0:
                try:
                    # Try NeuroKit2 first for enhanced frequency analysis
                    if NEUROKIT2_AVAILABLE:
                        try:
                            psd = nk.signal_psd(segment, sampling_rate=self.sampling_rate)
                            
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
                                    print(f"NeuroKit2 features - Mean: {mean_freq:.2f} Hz, Median: {median_freq:.2f} Hz")
                                else:
                                    raise ValueError("NeuroKit2 returned NaN values")
                            else:
                                raise ValueError("NeuroKit2 PSD returned invalid format")
                        except Exception as nk_error:
                            print(f"NeuroKit2 failed for peak at {peak_time:.3f}s: {nk_error}")
                            # Fall back to SciPy
                            if SCIPY_AVAILABLE:
                                print(f"Falling back to SciPy for peak at {peak_time:.3f}s")
                                # Use SciPy for frequency analysis
                                frequencies, power = signal.welch(segment, fs=self.sampling_rate, nperseg=min(len(segment), 256))
                                
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
                                    print(f"SciPy fallback features - Mean: {mean_freq:.2f} Hz, Median: {median_freq:.2f} Hz")
                                else:
                                    raise ValueError("SciPy also returned NaN values")
                            else:
                                print("ERROR: Neither NeuroKit2 nor SciPy available for frequency analysis")
                                raise ValueError("No frequency analysis method available")
                    else:
                        # NeuroKit2 not available, try SciPy
                        if SCIPY_AVAILABLE:
                            print(f"Using SciPy for peak at {peak_time:.3f}s (NeuroKit2 not available)")
                            # Use SciPy for frequency analysis
                            frequencies, power = signal.welch(segment, fs=self.sampling_rate, nperseg=min(len(segment), 256))
                            
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
                                print(f"SciPy features - Mean: {mean_freq:.2f} Hz, Median: {median_freq:.2f} Hz")
                            else:
                                raise ValueError("SciPy returned NaN values")
                        else:
                            print("ERROR: Neither NeuroKit2 nor SciPy available for frequency analysis")
                            raise ValueError("No frequency analysis method available")
                            
                except Exception as e:
                    print(f"Warning: All frequency analysis methods failed for peak at {peak_time:.3f}s: {e}")
                    # Only use defaults if all methods fail
                    mean_freqs.append(np.nan)
                    median_freqs.append(np.nan)
            else:
                print(f"Warning: Empty segment for peak at {peak_time:.3f}s")
                # Empty segment - no frequency analysis possible
                mean_freqs.append(np.nan)
                median_freqs.append(np.nan)
        
        # Calculate average frequency features across all segments (same for all peaks)
        mean_frequency = np.nanmean(mean_freqs) if mean_freqs else np.nan
        median_frequency = np.nanmean(median_freqs) if median_freqs else np.nan
        
        # Check if we have any valid frequency calculations
        if np.isnan(mean_frequency) or np.isnan(median_frequency):
            print("Warning: All frequency calculations failed for this trial.")
            print("This may indicate issues with the signal quality or segment length.")
            # Keep NaN values - let the LDA classifier handle them
        
        print(f"Final frequency features - Mean: {mean_frequency:.2f} Hz, Median: {median_frequency:.2f} Hz")
        
        # Create feature vectors for each peak (all peaks get the same trial-level features)
        features = []
        for peak_amplitude in peak_amplitudes:
            feature_vector = [
                peak_amplitude,  # Peak amplitude (different for each peak)
                min_peak_amplitude,  # Min peak amplitude (same for all peaks)
                mean_frequency,  # Mean frequency (same for all peaks)
                median_frequency  # Median frequency (same for all peaks)
            ]
            features.append(feature_vector)
        
        return np.array(features)

    def classify_peaks(self):
        """
        Classify detected peaks using the XGBoost model.
        
        Returns:
        --------
        list : List of classification results
        """
        if self.classifier is None:
            print("No XGBoost model available for classification")
            return []
        
        if len(self.peaks) == 0:
            print("No peaks detected for classification")
            return []
        
        # Get peak amplitudes
        data, times = self.raw_filtered[:, :]
        peak_amplitudes = data.squeeze()[self.peaks]
        peak_times = times[self.peaks]
        
        # Extract features for classification
        features = self.extract_peak_features(peak_times, peak_amplitudes)
        
        # Classify peaks
        try:
            predictions = self.classifier.predict(features)
            probabilities = self.classifier.predict_proba(features)
            
            # Create classification results
            self.classification_results = []
            for i, (time, amplitude, pred, prob, feature_vec) in enumerate(zip(peak_times, peak_amplitudes, predictions, probabilities, features)):
                result = {
                    'peak_id': i + 1,
                    'timestamp': time,
                    'amplitude': amplitude,
                    'min_amplitude': feature_vec[1] if len(feature_vec) > 1 else np.nan,
                    'mean_frequency': feature_vec[2] if len(feature_vec) > 2 else np.nan,
                    'median_frequency': feature_vec[3] if len(feature_vec) > 3 else np.nan,
                    'predicted_class': pred,
                    'probabilities': prob,
                    'confidence': max(prob),
                    'class_names': self.classifier.label_encoder.classes_
                }
                self.classification_results.append(result)
            
            print(f"Classified {len(self.classification_results)} peaks")
            return self.classification_results
            
        except Exception as e:
            print(f"Error during classification: {e}")
            return []
    
    def _save_classification_results(self):
        """Save classification results to CSV file."""
        if not self.classification_results:
            return
        
        # Create output file path
        output_file = self.csv_path.with_name("peak_classificat.csv")
        
        # Prepare data for CSV
        data = []
        for result in self.classification_results:
            row = {
                'peak_id': result['peak_id'],
                'timestamp': result['timestamp'],
                'amplitude': result['amplitude'],
                'min_amplitude': result['min_amplitude'],
                'mean_frequency': result['mean_frequency'],
                'median_frequency': result['median_frequency'],
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence']
            }
            
            # Add probability columns
            for i, class_name in enumerate(result['class_names']):
                row[f'prob_{class_name}'] = result['probabilities'][i]
            
            data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        # Show relative path for display
        try:
            relative_path = output_file.relative_to(Path.cwd())
        except ValueError:
            relative_path = output_file
        
        print(f"Classification results saved to: {relative_path}")
        
        # Also save in .txt format with the specified structure
        self._save_classification_results_txt()
    
    def _save_classification_results_txt(self):
        """Save classification results to /temp TXT file with specified structure."""
        if not self.classification_results:
            return
            
        # Extract vectors from all results
        #amplitude_vec = np.round(np.array([result['amplitude'] for result in self.classification_results]),2)
        #min_amplitude_vec = np.round(np.array([result['min_amplitude'] for result in self.classification_results]),2)
        #mean_frequency_vec = np.round(np.array([result['mean_frequency'] for result in self.classification_results]),2)
        #median_frequency_vec = np.round(np.array([result['median_frequency'] for result in self.classification_results]),2)
        
        # Format vectors as strings with proper precision
        #sujet = 0
        #amplitude_str = ",".join([f"{val:.2f}" for val in amplitude_vec])
        #min_amplitude_str = ",".join([f"{val:.2f}" for val in min_amplitude_vec])
        #mean_frequency_str = ",".join([f"{val:.2f}" for val in mean_frequency_vec])
        #median_frequency_str = ",".join([f"{val:.2f}" for val in median_frequency_vec])
        
        #amplitude_str = np.array([f"{val:.2f}" for val in amplitude_vec])
        #print(self.classification_results('amplitude'))
        #print(amplitude_vec)
        #print(len(amplitude_str))
        #print(amplitude_str)
        #min_amplitude_str = [f"{val:.2f}" for val in min_amplitude_vec]
        #mean_frequency_str = [f"{val:.2f}" for val in mean_frequency_vec]
        #median_frequency_str = [f"{val:.2f}" for val in median_frequency_vec]
        
        headers = "Subject;amplitude;min_amplitude;mean_frequency;median_frequency;Interaction_Scale;Interaction_Sign\n"
        #line_to_write = (f"{sujet};{amplitude_vec};{min_amplitude_vec};{mean_frequency_vec};{median_frequency_vec}\n")
        results = ['Subject','amplitude','min_amplitude','mean_frequency','median_frequency','Interaction_Scale','Interaction_Sign']
                
        # Create output file path
        output_file = self.csv_path.with_name("peak_classificat.txt")
        relative_path = output_file.relative_to(Path.cwd())
        print(relative_path)
        
        # Write header
        write_headers = not os.path.isfile(relative_path) #or os.path.getsize("peak_classificat.txt") == 0
        with open(output_file, 'w') as f: #"a" append to save all the data rows
            #if write_headers:
            f.write(headers + '\n')
        #try:
            for r in results:
                if r=="Subject":
                    f.write('0'+ ' ' + ';')
                    #print(r)
                elif r=='Interaction_Scale':
                    f.write('0.5'+ ' ' + ';')
                elif r=='Interaction_Sign':  
                    f.write('1')  
                #f.write(line_to_write + "\n")
                    #print(r)
                else:
                    vec = [f"{result[r]:.2f}" for result in self.classification_results]
                    for v in vec:
                        #print(v)
                        f.write(v+ ' ')
                    f.write(';')
            f.write('\n')    

        #if os.path.getsize("peak_classificat.txt") > len(line_to_write) + 1:
        #    print(f"Appended new data to peak_classifications.txt")
        #else:
        #    print(f"Created new file peak_classifications.txt")
            
        #except IOerror as e:
        #    print(f"An error ocurred: {e}")
                
        
        # Show relative path for display
        try:
            relative_path = output_file.relative_to(Path.cwd())
        except ValueError:
            relative_path = output_file
        
        print(f"Classification results (TXT format) saved to: {relative_path}")
    
    def _save_results(self):
        """Override parent method to include classification results."""
        # Call parent method first
        super()._save_results()
        
        # Add classification results if available
        if self.classification_results:
            output_file = self.csv_path.with_name("classfication_results.txt")
                       
            with open(output_file, 'a') as f:
                f.write(f"\n--- Classification Results ---\n")
                f.write(f"Model used: XGBoost Classifier\n")
                f.write(f"Available classes: {list(self.classifier.label_encoder.classes_)}\n")
                f.write(f"Classified peaks: {len(self.classification_results)}\n")
                
                # Summary statistics
                classes = [r['predicted_class'] for r in self.classification_results]
                class_counts = {}
                for cls in classes:
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                f.write(f"Class distribution: {class_counts}\n")
                
                # Individual peak results
                f.write(f"\nPeak Details:\n")
                for result in self.classification_results:
                    f.write(f"  Peak {result['peak_id']}: {result['timestamp']:.3f}s - "
                           f"{result['predicted_class']}% MVC "
                           f"(Amplitude: {result['amplitude']:.1f}µV, "
                           f"Confidence: {result['confidence']:.3f})\n")
    
    def run(self, show_plots=False, save_results=True, classify_peaks=True):
        """
        Run the full peak detection and classification pipeline.
        
        Parameters:
        -----------
        show_plots : bool
            Whether to display the signal + peak plot
        save_results : bool
            Whether to write results to files
        classify_peaks : bool
            Whether to classify detected peaks
            
        Returns:
        --------
        dict : results containing metadata, peak info, and classifications
        """
        print(f"[PeakClassifier] Analyzing: {self.csv_path}")
        
        # Run peak detection (parent method)
        peak_results = super().run(show_plots=False, save_results=False)
        
        # Classify peaks if requested and model is available
        if classify_peaks and self.classifier is not None:
            print("Classifying detected peaks...")
            classifications = self.classify_peaks()
            
            if classifications:
                # Save classification results
                self._save_classification_results()
                
                # Print summary
                print("\nClassification Summary:")
                class_counts = {}
                for result in classifications:
                    cls = result['predicted_class']
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                for cls, count in sorted(class_counts.items()):
                    print(f"  {cls}% MVC: {count} peaks")
                
                # Add classifications to results
                peak_results['classifications'] = classifications
                peak_results['classifier_available'] = True
            else:
                peak_results['classifications'] = []
                peak_results['classifier_available'] = False
        else:
            peak_results['classifications'] = []
            peak_results['classifier_available'] = self.classifier is not None
        
        # Save all results
        if save_results:
            self._save_results()
        
        # Show plots if requested
        if show_plots:
            self._plot()
        
        return peak_results


