#!/usr/bin/env python3
"""
Fixed Batch Peak Analysis for Combined EMG Dataset
=================================================

This script processes the combined_emg_dorsiflex.csv dataset using the corrected
EMGPeakAnalyzerFixed to properly detect peaks in each trial.

The script will:
1. Load the combined dataset
2. Extract individual trials (Subject, MVC, Trial combinations)
3. Run peak detection on each trial using the fixed analyzer
4. Save results in the correct format
"""

import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the fixed peak analyzer
from emg_peak_analyzer_fixed import EMGPeakAnalyzerFixed

class BatchPeakAnalyzer:
    def __init__(self, dataset_path, sampling_rate=220, height_percentile=98, min_distance=3):
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
        
        print(f"Subjects: {subjects}")
        print(f"MVC levels: {mvcs}")
        print(f"Trials: {trials}")
        
        # Verify expected structure
        expected_subjects = [f"S0{i}" for i in range(1, 8)]
        expected_mvcs = [10, 25, 50]
        expected_trials = [1, 2, 3]
        
        if subjects != expected_subjects:
            print(f"Warning: Unexpected subjects. Expected: {expected_subjects}, Got: {subjects}")
        
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
                'trial_data': trial_data  # Keep original trial data for verification
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
                
                # Convert back to absolute time (add the original trial start time)
                original_start_time = result['trial_data'].attrs['original_start_time']
                absolute_peak_time = original_start_time + highest_peak_time
                
                summary_data.append({
                    'Time': absolute_peak_time,
                    'fwEMG 3': highest_peak_amplitude,
                    'Subject': result['Subject'],
                    'MVC': result['MVC'],
                    'Trial': result['Trial']
                })
            else:
                # No peaks detected - use the maximum value from the trial
                trial_data = result['trial_data']
                max_idx = trial_data['fwEMG 3'].idxmax()
                # Convert back to absolute time
                original_start_time = trial_data.attrs['original_start_time']
                relative_max_time = trial_data.loc[max_idx, 'Time']
                absolute_max_time = original_start_time + relative_max_time
                max_amplitude = trial_data.loc[max_idx, 'fwEMG 3']
                
                summary_data.append({
                    'Time': absolute_max_time,
                    'fwEMG 3': max_amplitude,
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
                            'fwEMG 3': peak_amplitude,
                            'Subject': subject,
                            'MVC': mvc,
                            'Trial': trial,
                            'Peak_Number': i + 1,
                            'Is_Highest': (i == np.argmax(result['peak_amplitudes']))
                        })
                else:
                    # No peaks detected - use the maximum value
                    trial_data = result['trial_data']
                    max_idx = trial_data['fwEMG 3'].idxmax()
                    # Convert back to absolute time
                    original_start_time = trial_data.attrs['original_start_time']
                    relative_max_time = trial_data.loc[max_idx, 'Time']
                    absolute_max_time = original_start_time + relative_max_time
                    max_amplitude = trial_data.loc[max_idx, 'fwEMG 3']
                    
                    detailed_data.append({
                        'Time': absolute_max_time,
                        'fwEMG 3': max_amplitude,
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
    
    def run(self):
        """Run the complete batch analysis."""
        print("=" * 60)
        print("Fixed Batch Peak Analysis")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Create output directory
        self.create_output_directory()
        
        # Get all unique combinations
        combinations = self.df.groupby(['Subject', 'MVC', 'Trial']).size().reset_index()
        print(f"\nFound {len(combinations)} unique trial combinations")
        
        # Analyze each trial
        all_results = []
        
        for _, row in combinations.iterrows():
            subject, mvc, trial = row['Subject'], row['MVC'], row['Trial']
            result = self.analyze_trial(subject, mvc, trial)
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
        
        return {
            'summary_df': summary_df,
            'detailed_df': detailed_df,
            'all_results': all_results,
            'output_dir': self.output_dir
        }

def main():
    """Main function to run the batch analysis."""
    # Path to the combined dataset
    dataset_path = Path(__file__).parent.parent / "dataset" / "combined_emg_dorsiflex.csv"
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    # Create and run the batch analyzer
    analyzer = BatchPeakAnalyzer(
        dataset_path=dataset_path,
        sampling_rate=220,
        height_percentile=95,  # Slightly lower threshold for better detection
        min_distance=2  # Minimum 2 seconds between peaks
    )
    
    results = analyzer.run()
    
    print(f"\nBatch analysis complete!")
    print(f"Check the results in: {results['output_dir']}")

if __name__ == "__main__":
    main()
