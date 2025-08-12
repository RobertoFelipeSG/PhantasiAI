#!/usr/bin/env python3
"""
EMG Peak Classifier
==================

This module extends the EMG peak analyzer to include LDA classification
of detected peaks. It processes recorded EMG data, detects peaks, and
classifies them using a pre-trained LDA model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .emg_peak_analyzer import EMGPeakAnalyzer
from .emg_LDA_classifier import EMGLDAClassifier

class EMGPeakClassifier(EMGPeakAnalyzer):
    """
    EMG peak analyzer with LDA classification capabilities.
    
    This class extends the basic peak analyzer to classify detected peaks
    using a pre-trained LDA model.
    """
    
    def __init__(self, csv_path, model_path=None, sampling_rate=220, 
                 height_percentile=98, min_distance=3):
        """
        Initialize the EMG peak classifier.
        
        Parameters:
        -----------
        csv_path : str or Path
            Path to the EMG CSV file
        model_path : str or Path, optional
            Path to the pre-trained LDA model (.pkl file)
        sampling_rate : int
            Sampling rate of EMG data (Hz)
        height_percentile : float
            Threshold for peak detection (percentile of signal amplitude)
        min_distance : float
            Minimum time (in seconds) between peaks
        """
        super().__init__(csv_path, sampling_rate, height_percentile, min_distance)
        
        # Initialize LDA classifier
        self.classifier = None
        self.classification_results = []
        
        if model_path:
            self.load_model(model_path)
        else:
            # Try to find the default model
            default_model = Path(__file__).parent / "lda_model.pkl"
            if default_model.exists():
                self.load_model(default_model)
            else:
                print("Warning: No LDA model found. Classification will be skipped.")
    
    def load_model(self, model_path):
        """
        Load a pre-trained LDA model.
        
        Parameters:
        -----------
        model_path : str or Path
            Path to the .pkl file containing the trained model
        """
        try:
            self.classifier = EMGLDAClassifier()
            self.classifier.load_model(model_path)
            print(f"LDA model loaded successfully from: {model_path}")
            print(f"Available classes: {self.classifier.label_encoder.classes_}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.classifier = None
            return False
    
    def extract_peak_features(self, peak_times, peak_amplitudes):
        """
        Extract enhanced features for each peak including frequency features.
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
        
        try:
            # Import scipy for frequency analysis
            from scipy import signal
            
            # Create segments centered around each peak
            seg_len = int(2 * self.sampling_rate)  # 2 second segments
            
            for peak_time in peak_times:
                # Convert time to sample index
                peak_idx = int(peak_time * self.sampling_rate)
                
                # Extract segment around peak
                start_idx = max(0, peak_idx - seg_len // 2)
                end_idx = min(len(emg_signal), peak_idx + seg_len // 2)
                segment = emg_signal[start_idx:end_idx]
                
                if len(segment) > 0:
                    try:
                        # Calculate frequency features using SciPy
                        # Use power spectral density analysis
                        f, Pxx = signal.welch(segment, self.sampling_rate)
                        
                        # Calculate mean frequency (frequency weighted by power)
                        mean_freq = np.sum(f * Pxx) / np.sum(Pxx) if np.sum(Pxx) > 0 else np.nan
                        
                        # Calculate median frequency (frequency where cumulative power is 50%)
                        cumsum_power = np.cumsum(Pxx)
                        if cumsum_power[-1] > 0:
                            median_idx = np.argmin(np.abs(cumsum_power - 0.5 * cumsum_power[-1]))
                            median_freq = f[median_idx]
                        else:
                            median_freq = np.nan
                        
                        mean_freqs.append(mean_freq)
                        median_freqs.append(median_freq)
                    except Exception as e:
                        print(f"Warning: Error calculating frequency features: {e}")
                        mean_freqs.append(np.nan)
                        median_freqs.append(np.nan)
                else:
                    mean_freqs.append(np.nan)
                    median_freqs.append(np.nan)
                    
        except ImportError:
            print("Warning: SciPy not available. Using NaN for frequency features.")
            mean_freqs = [np.nan] * len(peak_times)
            median_freqs = [np.nan] * len(peak_times)
        
        # Calculate average frequency features across all segments (same for all peaks)
        mean_frequency = np.nanmean(mean_freqs) if mean_freqs else np.nan
        median_frequency = np.nanmean(median_freqs) if median_freqs else np.nan
        
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
        Classify detected peaks using the LDA model.
        
        Returns:
        --------
        list : List of classification results
        """
        if self.classifier is None:
            print("No LDA model available for classification")
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
        output_file = self.csv_path.with_name("peak_classifications.csv")
        
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
    
    def _save_results(self):
        """Override parent method to include classification results."""
        # Call parent method first
        super()._save_results()
        
        # Add classification results if available
        if self.classification_results:
            output_file = self.csv_path.with_name("peaks.txt")
            
            with open(output_file, 'a') as f:
                f.write(f"\n--- Classification Results ---\n")
                f.write(f"Model used: LDA Classifier\n")
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

def classify_emg_recording(csv_path, model_path=None, show_plots=False):
    """
    Convenience function to classify an EMG recording.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the EMG CSV file
    model_path : str or Path, optional
        Path to the LDA model
    show_plots : bool
        Whether to show plots
        
    Returns:
    --------
    dict : Analysis results
    """
    classifier = EMGPeakClassifier(csv_path, model_path)
    return classifier.run(show_plots=show_plots, save_results=True, classify_peaks=True)

def main():
    """Demo function to test the peak classifier."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python emg_peak_classifier.py <emg_csv_file> [model_path]")
        return
    
    csv_file = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("=" * 60)
    print("EMG Peak Classifier")
    print("=" * 60)
    
    results = classify_emg_recording(csv_file, model_path, show_plots=False)
    
    print(f"\nAnalysis completed!")
    print(f"Peaks detected: {results['num_peaks']}")
    print(f"Classifications: {len(results['classifications'])}")
    
    if results['classifications']:
        print(f"Class distribution: {dict(pd.Series([r['predicted_class'] for r in results['classifications']]).value_counts())}")

if __name__ == "__main__":
    main()
