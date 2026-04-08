
#!/usr/bin/env python3
"""
Batch Peak Analysis for Combined EMG Dataset
====================================================

This script processes the combined_emg_dorsiflex.csv dataset using 
EMGPeakAnalyzerFixed (version compatible with database structure)
to properly detect peaks in each trial and extract enhanced features.

The script will:
1. Load the combined dataset
2. Extract individual trials (Subject, MVC, Trial combinations)
3. Run peak detection on each trial using the fixed analyzer
4. Extract features including:
   - Maximum peak amplitude 
   - Minimum peak amplitude
   - Mean frequency 
   - Median frequency
5. Save results in the correct format
"""

import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from datetime import datetime
import warnings
import subprocess
import sys
warnings.filterwarnings('ignore')

# Import the fixed peak analyzer
from emg_peak_analyzer_fixed import EMGPeakAnalyzerFixed

# Import NeuroKit2 for enhanced frequency analysis
try:
    import neurokit2 as nk
    NEUROKIT2_AVAILABLE = True
    print("NeuroKit2 available for enhanced frequency analysis")
except ImportError:
    print("ERROR: NeuroKit2 is required for frequency analysis. Please install it with: pip install neurokit2")
    NEUROKIT2_AVAILABLE = False

# Import SciPy for fallback frequency analysis
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
    print("SciPy available for fallback frequency analysis")
except ImportError:
    print("WARNING: SciPy not available for fallback. Please install it with: pip install scipy")
    SCIPY_AVAILABLE = False



class BatchPeakAnalyzer:
    def __init__(self, dataset_path, sampling_rate=10000, height_percentile=98, min_distance=3):
        """
        Initialize the batch analyzer.
        
        Parameters:
        -----------
        dataset_path : str or Path
            Path to the combined_emg_dorsiflex.csv file
        sampling_rate : int
            Sampling rate of EMG data (Hz)
        height_percentile : float
            Threshold for peak detection (percentile of signal amplitude)
        min_distance : float
            Minimum time (in seconds) between peaks
        """
        self.dataset_path = Path(dataset_path)
        self.sampling_rate = sampling_rate
        self.height_percentile = height_percentile
        self.min_distance = min_distance
        
        self.df = None
        self.output_dir = None
        
    def load_data(self):
        """Load the combined dataset."""
        print(f"Loading dataset: {self.dataset_path}")

        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        

        # Check data structure
        subjects = sorted(self.df['Subject'].unique())
        mvcs = sorted(self.df['MVC'].unique())
        trials = sorted(self.df['Trial'].unique())
        
        print(f"Unique subjects: {subjects}")
        print(f"MVC levels: {mvcs}")
        print(f"Trials: {trials}")
        
        # Verify expected structure
        expected_base_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S07']
        expected_mvcs = [10, 25, 50]
        expected_trials = [1, 2, 3]
        
        # Extract base subject names (remove channel info)
        base_subjects = sorted(list(set([s.split('_')[0] for s in subjects if '_' in s])))
        print(f"Base subjects (without channels): {base_subjects}")
        
        # Check if we have multiple channels per subject
        channels_per_subject = {}
        for subject in subjects:
            if '_' in subject:
                base_subject = subject.split('_')[0]
                channel = subject.split('_')[1]
                if base_subject not in channels_per_subject:
                    channels_per_subject[base_subject] = []
                channels_per_subject[base_subject].append(channel)
        
        print(f"Channels per subject: {channels_per_subject}")
        
        if base_subjects != expected_base_subjects:
            print(f"Warning: Unexpected base subjects. Expected: {expected_base_subjects}, Got: {base_subjects}")
        
        if mvcs != expected_mvcs:
            print(f"Warning: Unexpected MVC levels. Expected: {expected_mvcs}, Got: {mvcs}")
            
        if trials != expected_trials:
            print(f"Warning: Unexpected trials. Expected: {expected_trials}, Got: {trials}")
    
    def create_output_directory(self):
        """Create output directory for results."""
        # Use the existing dataset directory inside 2025
        dataset_dir = self.dataset_path.parent
        dataset_dir.mkdir(exist_ok=True)
        
        # Create timestamped output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = dataset_dir / f"batch_peak_analysis_{timestamp}"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
    
    def extract_trial_data(self, subject, mvc, trial):
        """Extract data for a specific trial."""

        # Use the full subject name (including channel) as provided
        trial_data = self.df[
            (self.df['Subject'] == subject) & 
            (self.df['MVC'] == mvc) & 
            (self.df['Trial'] == trial)
        ].copy()
        
        if trial_data.empty:
            print(f"No data found for {subject}, MVC={mvc}%, Trial={trial}")
            return None

        
        # Store the original start time before resetting
        original_start_time = trial_data['Time'].iloc[0]
        
        # Reset time to start from 0 for this trial
        trial_data = trial_data.reset_index(drop=True)
        trial_data['Time'] = trial_data['Time'] - original_start_time
        
        # Store the original start time for later use
        trial_data.attrs['original_start_time'] = original_start_time
        
        return trial_data
    
    def save_temporary_csv(self, trial_data, subject, mvc, trial):
        """Save trial data to a temporary CSV file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        trial_data.to_csv(temp_file.name, index=False)
        return temp_file.name
    
    def analyze_trial(self, subject, mvc, trial):
        """Analyze a single trial using the fixed peak analyzer."""

        print(f"Analyzing {subject}, MVC={mvc}%, Trial={trial}...")
        
        # Extract trial data
        trial_data = self.extract_trial_data(subject, mvc, trial)
        if trial_data is None:
            return None

        
        # Save to temporary CSV
        temp_csv_path = self.save_temporary_csv(trial_data, subject, mvc, trial)
        
        try:
            # Run peak analysis
            analyzer = EMGPeakAnalyzerFixed(

                csv_path=temp_csv_path,
                sampling_rate=self.sampling_rate,
                height_percentile=self.height_percentile,
                min_distance=self.min_distance
            )
            
            results = analyzer.run(show_plots=False, save_results=False)
            
            # Extract enhanced features
            enhanced_features = self.extract_enhanced_features(
                trial_data, 
                results['peak_times'], 
                results['peak_amplitudes']
            )

            # Clean up temporary file
            os.unlink(temp_csv_path)
            
            return {
                'Subject': subject,
                'MVC': mvc,
                'Trial': trial,
                'num_peaks': results['num_peaks'],
                'peak_times': results['peak_times'],
                'peak_amplitudes': results['peak_amplitudes'],
                'highest_peak_time': results['highest_peak_time'],
                'highest_peak_amplitude': results['highest_peak_amplitude'],
                'signal_duration': results['signal_duration'],
                'trial_data': trial_data,  # Keep original trial data for verification
                'min_peak_amplitude': enhanced_features['min_peak_amplitude'],
                'mean_frequency': enhanced_features['mean_frequency'],
                'median_frequency': enhanced_features['median_frequency']
            }
            
        except Exception as e:
            print(f"Error analyzing {subject}, MVC={mvc}%, Trial={trial}: {e}")
            # Clean up temporary file
            if os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
            return None
    
    def save_summary_results(self, all_results):
        """Save the highest peak for each trial in the required format."""
        print("Saving summary results...")
        
        summary_data = []
        
        for result in all_results:
            if result is not None and result['num_peaks'] > 0:
                # Get the highest peak
                highest_peak_time = result['highest_peak_time']
                highest_peak_amplitude = result['highest_peak_amplitude']
                
                # The peak_time is in the trial's relative coordinate system (0-11s)
                # We need to convert it back to the original absolute time
                original_start_time = result['trial_data'].attrs['original_start_time']
                absolute_peak_time = original_start_time + highest_peak_time
                
                summary_data.append({
                    'Time': absolute_peak_time,
                    'EMG': highest_peak_amplitude,
                    'Min_Peak_Amplitude': result['min_peak_amplitude'],
                    'Mean_Frequency': result['mean_frequency'],
                    'Median_Frequency': result['median_frequency'],
                    'Subject': result['Subject'],
                    'MVC': result['MVC'],
                    'Trial': result['Trial']
                })
            else:
                # No peaks detected - use the maximum value from the trial
                trial_data = result['trial_data']
                max_idx = trial_data['EMG'].idxmax()
                # Convert back to absolute time
                original_start_time = trial_data.attrs['original_start_time']
                relative_max_time = trial_data.loc[max_idx, 'Time']
                absolute_max_time = original_start_time + relative_max_time
                max_amplitude = trial_data.loc[max_idx, 'EMG']
                
                summary_data.append({
                    'Time': absolute_max_time,
                    'EMG': max_amplitude,
                    'Min_Peak_Amplitude': result['min_peak_amplitude'],
                    'Mean_Frequency': result['mean_frequency'],
                    'Median_Frequency': result['median_frequency'],
                    'Subject': result['Subject'],
                    'MVC': result['MVC'],
                    'Trial': result['Trial']
                })
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "peak_analysis_results.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Summary results saved to: {summary_path}")
        print(f"Total trials processed: {len(summary_data)}")
        
        return summary_df
    
    def save_detailed_results(self, all_results):
        """Save detailed results including all peaks for each trial."""
        print("Saving detailed results...")
        
        detailed_data = []
        
        for result in all_results:
            if result is not None:
                subject = result['Subject']
                mvc = result['MVC']
                trial = result['Trial']
                
                if result['num_peaks'] > 0:
                    # Add all detected peaks
                    for i, (peak_time, peak_amplitude) in enumerate(zip(result['peak_times'], result['peak_amplitudes'])):
                        # Convert back to absolute time
                        original_start_time = result['trial_data'].attrs['original_start_time']
                        absolute_peak_time = original_start_time + peak_time
                        
                        detailed_data.append({
                            'Time': absolute_peak_time,
                            'EMG': peak_amplitude,
                            'Subject': subject,
                            'MVC': mvc,
                            'Trial': trial,
                            'Peak_Number': i + 1,
                            'Is_Highest': (i == np.argmax(result['peak_amplitudes']))
                        })
                else:
                    # No peaks detected - use the maximum value
                    trial_data = result['trial_data']
                    max_idx = trial_data['EMG'].idxmax()
                    # Convert back to absolute time
                    original_start_time = trial_data.attrs['original_start_time']
                    relative_max_time = trial_data.loc[max_idx, 'Time']
                    absolute_max_time = original_start_time + relative_max_time
                    max_amplitude = trial_data.loc[max_idx, 'EMG']
                    
                    detailed_data.append({
                        'Time': absolute_max_time,
                        'EMG': max_amplitude,
                        'Subject': subject,
                        'MVC': mvc,
                        'Trial': trial,
                        'Peak_Number': 1,
                        'Is_Highest': True
                    })
        
        # Create DataFrame and save
        detailed_df = pd.DataFrame(detailed_data)
        detailed_path = self.output_dir / "detailed_peak_analysis.csv"
        detailed_df.to_csv(detailed_path, index=False)
        
        print(f"Detailed results saved to: {detailed_path}")
        
        return detailed_df
    
    def extract_enhanced_features(self, trial_data, peak_times, peak_amplitudes):
        """
        Extract enhanced features from EMG signal segments around peaks.
        Uses a 20ms window centered around each peak and PySiology toolkit for feature calculation.
        
        Parameters:
        -----------
        trial_data : pd.DataFrame
            Trial data containing EMG signal
        peak_times : list
            Times of detected peaks
        peak_amplitudes : list
            Amplitudes of detected peaks
            
        Returns:
        --------
        dict : Dictionary containing extracted features
        """
        if len(peak_times) == 0:
            return {
                'min_peak_amplitude': np.nan,
                'mean_frequency': np.nan,
                'median_frequency': np.nan
            }
        
        # Get EMG signal
        emg_signal = trial_data['EMG'].values
        
        # Extract minimum peak amplitude
        min_peak_amplitude = np.min(peak_amplitudes) if len(peak_amplitudes) > 0 else np.nan
        
        # Extract frequency features from segments around peaks
        mean_freqs = []
        median_freqs = []
        
        # Use 3s window centered around each peak
        window_duration = 3  # 3 seconds
        # Use resampled sampling rate (200Hz) for feature extraction
        resampled_sampling_rate = 200
        seg_len = int(window_duration * resampled_sampling_rate)  # Convert to samples
        
        print(f"Using {window_duration*1000:.0f}ms window ({seg_len} samples) centered around peaks at {resampled_sampling_rate}Hz")
        
        for peak_time in peak_times:
            # Convert time to sample index using resampled sampling rate
            peak_idx = int(peak_time * resampled_sampling_rate)
            
            # For feature extraction, we need to get the resampled envelope from the analyzer
            # Since we're working with the original trial data, we'll need to resample it too
            # Extract segment around peak from the original signal and then resample
            original_peak_idx = int(peak_time * self.sampling_rate)
            original_start_idx = max(0, original_peak_idx - int(seg_len * self.sampling_rate / resampled_sampling_rate // 2))
            original_end_idx = min(len(emg_signal), original_peak_idx + int(seg_len * self.sampling_rate / resampled_sampling_rate // 2))
            original_segment = emg_signal[original_start_idx:original_end_idx]
            
            # Resample the segment to 200Hz
            from scipy import signal
            target_length = int(len(original_segment) * resampled_sampling_rate / self.sampling_rate)
            if target_length > 0:
                segment = signal.resample(original_segment, target_length)
            else:
                segment = np.array([])
            
            if len(segment) > 0:
                try:
                    # Try NeuroKit2 first for enhanced frequency analysis
                    if NEUROKIT2_AVAILABLE:
                        try:
                            psd = nk.signal_psd(segment, sampling_rate=resampled_sampling_rate)
                            
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
                                frequencies, power = signal.welch(segment, fs=resampled_sampling_rate, nperseg=min(len(segment), 256))
                                
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
                            frequencies, power = signal.welch(segment, fs=resampled_sampling_rate, nperseg=min(len(segment), 256))
                            
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
        
        # Calculate average frequency features across all segments
        mean_frequency = np.nanmean(mean_freqs) if mean_freqs else np.nan
        median_frequency = np.nanmean(median_freqs) if median_freqs else np.nan
        
        # Check if we have any valid frequency calculations
        if np.isnan(mean_frequency) or np.isnan(median_frequency):
            print("Warning: All frequency calculations failed for this trial.")
            print("This may indicate issues with the signal quality or segment length.")
            # Keep NaN values - let the LDA classifier handle them
        
        print(f"Final frequency features - Mean: {mean_frequency:.2f} Hz, Median: {median_frequency:.2f} Hz")
        
        return {
            'min_peak_amplitude': min_peak_amplitude,
            'mean_frequency': mean_frequency,
            'median_frequency': median_frequency
        }
    
    def run_lda_classification(self):
        """Run LDA classification on the peak analysis results."""
        try:
            print("=" * 60)
            print("Running LDA Classification")
            print("=" * 60)
            
            # Import the LDA classifier
            from emg_LDA_classifier import EMGLDAClassifier
            
            # Path to the peak analysis results
            peak_results_file = self.output_dir / "peak_analysis_results.csv"
            
            if not peak_results_file.exists():
                print(f"Peak analysis results not found: {peak_results_file}")
                return
            
            print(f"Loading peak analysis results: {peak_results_file}")
            
            # Initialize and train the LDA classifier
            classifier = EMGLDAClassifier()
            results = classifier.train(data_path=str(peak_results_file))
            
            print(f"LDA training completed!")
            print(f"Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"CV Test Accuracy: {results['cv_test_accuracy']:.4f}")
            
            # Load the peak analysis results for classification
            df = pd.read_csv(peak_results_file)
            
            # Create feature vectors using all 4 features
            feature_vectors = []
            for _, row in df.iterrows():
                feature_vector = [
                    row['EMG'],  # Peak amplitude
                    row['Min_Peak_Amplitude'],  # Min peak amplitude
                    row['Mean_Frequency'],  # Mean frequency
                    row['Median_Frequency']  # Median frequency
                ]
                feature_vectors.append(feature_vector)
            
            feature_vectors = np.array(feature_vectors)
            print(f"Created feature vectors with shape: {feature_vectors.shape}")
            
            # Classify using all features
            classifications = classifier.predict(feature_vectors)
            probabilities = classifier.predict_proba(feature_vectors)
            
            # Create results DataFrame
            results_df = df.copy()
            results_df['Predicted_MVC'] = classifications
            
            # Add probability columns
            class_names = classifier.label_encoder.classes_
            for i, class_name in enumerate(class_names):
                results_df[f'Prob_{class_name}'] = probabilities[:, i]
            
            # Save LDA results
            lda_results_file = self.output_dir / "lda_classification_results.csv"
            results_df.to_csv(lda_results_file, index=False)
            
            # Save the trained model
            model_file = self.output_dir / "lda_model.pkl"
            classifier.save_model(str(model_file))
            
            print(f"LDA results saved to: {lda_results_file}")
            print(f"Trained model saved to: {model_file}")
            
            # Display summary
            print(f"\nLDA Classification Summary:")
            print(f"   Total samples: {len(results_df)}")
            print(f"   Predicted classes: {sorted(np.unique(classifications))}")
            
            # Show some examples
            print(f"\nSample classifications:")
            for i in range(min(5, len(results_df))):
                row = results_df.iloc[i]
                print(f"   Sample {i+1}: Subject={row['Subject']}, MVC={row['MVC']}, "
                      f"Peak={row['EMG']:.3f}, Predicted={row['Predicted_MVC']}")
            
            print("=" * 60)
            print("LDA Classification completed successfully!")
            
        except Exception as e:
            print(f"Error running LDA classification: {e}")
            print("Continuing without LDA classification...")
    
    def run(self):
        """Run the complete batch analysis."""
        print("=" * 60)
        print("Fixed Batch Peak Analysis")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Create output directory
        self.create_output_directory()
        
        # Get all unique combinations for ALL channels
        combinations = self.df.groupby(['Subject', 'MVC', 'Trial']).size().reset_index()
        print(f"\nFound {len(combinations)} unique trial combinations (ALL channels)")
        
        # Analyze each trial
        all_results = []
        
        for _, row in combinations.iterrows():
            subject_full, mvc, trial = row['Subject'], row['MVC'], row['Trial']
            # Extract base subject name and channel
            if '_' in subject_full:
                subject = subject_full.split('_')[0]
                channel = subject_full.split('_')[1]
            else:
                subject = subject_full
                channel = 'CH1'  # Default for backward compatibility
            
            print(f"Processing {subject_full} (Subject: {subject}, Channel: {channel})")
            result = self.analyze_trial(subject_full, mvc, trial)  # Pass full subject name
            all_results.append(result)
        
        # Save results
        summary_df = self.save_summary_results(all_results)
        detailed_df = self.save_detailed_results(all_results)
        
        # Print summary statistics
        print(f"\n=== Analysis Summary ===")
        print(f"Total trials: {len(all_results)}")
        successful_analyses = sum(1 for r in all_results if r is not None)
        print(f"Successful analyses: {successful_analyses}")
        
        if successful_analyses > 0:
            total_peaks = sum(r['num_peaks'] for r in all_results if r is not None)
            print(f"Total peaks detected: {total_peaks}")
            avg_peaks_per_trial = total_peaks / successful_analyses
            print(f"Average peaks per trial: {avg_peaks_per_trial:.2f}")
        
        print(f"\nResults saved to: {self.output_dir}")
        print("=" * 60)
        
        # Run LDA classification after peak analysis
        print("\nRunning LDA Classification...")
        self.run_lda_classification()
        
        return {
            'summary_df': summary_df,
            'detailed_df': detailed_df,
            'all_results': all_results,
            'output_dir': self.output_dir
        }

def main():
    """Main function to run the batch analysis."""
    # Path to the combined dataset
    dataset_path = Path(__file__).parent.parent / "dataset" / "combined_emg_dorsiflex_master.csv"
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    # Create and run the batch analyzer
    analyzer = BatchPeakAnalyzer(
        dataset_path=dataset_path,
        sampling_rate=10000,
        height_percentile=95,  # Slightly lower threshold for better detection
        min_distance=2  # Minimum 2 seconds between peaks
    )
    
    results = analyzer.run()
    
    print(f"\nBatch analysis complete!")
    print(f"Check the results in: {results['output_dir']}")


if __name__ == "__main__":
    main()
