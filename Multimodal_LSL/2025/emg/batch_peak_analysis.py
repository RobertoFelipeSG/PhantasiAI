import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from emg_peak_analyzer import EMGPeakAnalyzer
import tempfile
import json
from datetime import datetime

class BatchPeakAnalyzer:
    def __init__(self, dataset_path, output_dir=None):
        """
        Initialize the batch peak analyzer.
        
        Parameters:
        -----------
        dataset_path : str
            Path to the combined EMG dataset CSV file
        output_dir : str, optional
            Directory to save results. If None, creates a timestamped directory
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Define the expected data structure
        self.subjects = [f"S0{i}" for i in range(1, 8)]
        self.mvcs = [10, 25, 50]
        self.trials = [1, 2, 3]
        
        # Analysis parameters
        self.sampling_rate = 220
        self.height_percentile = 98
        self.min_distance = 3
        
        # Results storage
        self.results = {}
        
    def load_dataset(self):
        """Load the combined dataset."""
        print(f"Loading dataset from: {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        
        # Verify expected columns
        expected_columns = ['Time', 'fwEMG 3', 'Subject', 'MVC', 'Trial']
        missing_columns = [col for col in expected_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing expected columns: {missing_columns}")
            
        print("Dataset loaded successfully!")
        
    def create_output_directory(self):
        """Create output directory for results."""
        if self.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create output in a 'dataset' folder outside the current directory
            # Go up two levels from emg folder to get to 2025, then create dataset folder
            parent_dir = Path(__file__).parent.parent.parent  # Go up to 2025's parent
            dataset_dir = parent_dir / "dataset"
            dataset_dir.mkdir(exist_ok=True)  # Create dataset folder if it doesn't exist
            
            self.output_dir = dataset_dir / f"batch_peak_analysis_{timestamp}"
        
        self.output_dir.mkdir(exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}")
        
    def extract_trial_data(self, subject, mvc, trial):
        """
        Extract data for a specific trial.
        
        Parameters:
        -----------
        subject : str
            Subject identifier (e.g., 'S01')
        mvc : int
            MVC percentage (10, 25, or 50)
        trial : int
            Trial number (1, 2, or 3)
            
        Returns:
        --------
        pandas.DataFrame : Trial data with 'timestamp' and 'emg' columns
        """
        # Filter data for specific trial
        trial_data = self.df[
            (self.df['Subject'] == subject) & 
            (self.df['MVC'] == mvc) & 
            (self.df['Trial'] == trial)
        ].copy()
        
        if trial_data.empty:
            print(f"No data found for {subject}, MVC={mvc}%, Trial={trial}")
            return None
            
        # Rename columns to match EMGPeakAnalyzer expectations
        trial_data = trial_data.rename(columns={
            'Time': 'timestamp',
            'fwEMG 3': 'emg'
        })
        
        # Select only the required columns
        trial_data = trial_data[['timestamp', 'emg']].reset_index(drop=True)
        
        return trial_data
        
    def save_trial_csv(self, trial_data, subject, mvc, trial):
        """
        Save trial data to a temporary CSV file for analysis.
        
        Returns:
        --------
        str : Path to the temporary CSV file
        """
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.csv', 
            delete=False,
            dir=self.output_dir
        )
        
        # Save trial data
        trial_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name
        
    def analyze_trial(self, subject, mvc, trial):
        """
        Analyze a single trial.
        
        Returns:
        --------
        dict : Analysis results or None if no data
        """
        print(f"Analyzing {subject}, MVC={mvc}%, Trial={trial}...")
        
        # Extract trial data
        trial_data = self.extract_trial_data(subject, mvc, trial)
        if trial_data is None:
            return None
            
        # Save to temporary CSV
        temp_csv_path = self.save_trial_csv(trial_data, subject, mvc, trial)
        
        try:
            # Run peak analysis
            analyzer = EMGPeakAnalyzer(
                csv_path=temp_csv_path,
                sampling_rate=self.sampling_rate,
                height_percentile=self.height_percentile,
                min_distance=self.min_distance
            )
            
            results = analyzer.run(show_plots=False, save_results=False)
            
            # Add metadata
            results.update({
                'subject': subject,
                'mvc': mvc,
                'trial': trial,
                'temp_csv_path': temp_csv_path
            })
            
            print(f"  Found {results['num_peaks']} peaks in {results['signal_duration']:.2f}s")
            return results
            
        except Exception as e:
            print(f"  Error analyzing {subject}, MVC={mvc}%, Trial={trial}: {str(e)}")
            return None
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_csv_path)
            except:
                pass
                
    def run_batch_analysis(self):
        """
        Run peak analysis for all trials in the dataset.
        """
        print("Starting batch peak analysis...")
        print(f"Subjects: {self.subjects}")
        print(f"MVC levels: {self.mvcs}")
        print(f"Trials: {self.trials}")
        print(f"Total combinations: {len(self.subjects) * len(self.mvcs) * len(self.trials)}")
        print("-" * 50)
        
        # Create output directory
        self.create_output_directory()
        
        # Initialize results
        self.results = {
            'analysis_parameters': {
                'sampling_rate': self.sampling_rate,
                'height_percentile': self.height_percentile,
                'min_distance': self.min_distance
            },
            'trials': {}
        }
        
        # Analyze each trial
        for subject in self.subjects:
            self.results['trials'][subject] = {}
            
            for mvc in self.mvcs:
                self.results['trials'][subject][mvc] = {}
                
                for trial in self.trials:
                    trial_key = f"{subject}_MVC{mvc}_Trial{trial}"
                    result = self.analyze_trial(subject, mvc, trial)
                    
                    if result:
                        self.results['trials'][subject][mvc][trial] = result
                    else:
                        self.results['trials'][subject][mvc][trial] = None
                        
        # Save summary results
        self.save_summary_results()
        
        print("\n" + "=" * 50)
        print("BATCH ANALYSIS COMPLETE")
        print("=" * 50)
        self.print_summary()
        
    def save_summary_results(self):
        """Save summary results to JSON file and CSV file with same structure as input."""
        # Create summary without raw data objects
        summary = {
            'analysis_parameters': self.results['analysis_parameters'],
            'summary_stats': {},
            'trial_results': {}
        }
        
        total_peaks = 0
        total_duration = 0
        successful_trials = 0
        
        # Prepare data for CSV output
        csv_data = []
        
        for subject in self.subjects:
            summary['trial_results'][subject] = {}
            
            for mvc in self.mvcs:
                summary['trial_results'][subject][mvc] = {}
                
                for trial in self.trials:
                    trial_result = self.results['trials'][subject][mvc][trial]
                    
                    if trial_result:
                        summary['trial_results'][subject][mvc][trial] = {
                            'num_peaks': trial_result['num_peaks'],
                            'signal_duration': trial_result['signal_duration'],
                            'peak_times': trial_result['peak_times'].tolist() if len(trial_result['peak_times']) > 0 else []
                        }
                        
                        # Add to CSV data - keep only the highest peak per trial
                        if len(trial_result['peak_times']) > 0:
                            # Get the original trial data to find peak amplitudes
                            trial_data = self.extract_trial_data(subject, mvc, trial)
                            if trial_data is not None:
                                # Find the highest peak amplitude
                                peak_amplitudes = []
                                for peak_time in trial_result['peak_times']:
                                    # Find the closest timestamp in the original data
                                    time_diff = np.abs(trial_data['timestamp'] - peak_time)
                                    closest_idx = np.argmin(time_diff)
                                    peak_amplitude = trial_data.iloc[closest_idx]['emg']
                                    peak_amplitudes.append(peak_amplitude)
                                
                                # Find the highest peak
                                max_peak_idx = np.argmax(peak_amplitudes)
                                max_peak_time = trial_result['peak_times'][max_peak_idx]
                                max_peak_amplitude = peak_amplitudes[max_peak_idx]
                                
                                csv_data.append({
                                    'Time': max_peak_time,           # Timestamp of highest peak
                                    'fwEMG 3': max_peak_amplitude,   # Amplitude of highest peak
                                    'Subject': subject,
                                    'MVC': mvc,
                                    'Trial': trial
                                })
                            else:
                                # Fallback: use first peak if we can't get original data
                                csv_data.append({
                                    'Time': trial_result['peak_times'][0],
                                    'fwEMG 3': 1,  # Default amplitude
                                    'Subject': subject,
                                    'MVC': mvc,
                                    'Trial': trial
                                })
                        else:
                            # No peaks detected in this trial
                            csv_data.append({
                                'Time': 0.0,
                                'fwEMG 3': 0,  # No peaks detected
                                'Subject': subject,
                                'MVC': mvc,
                                'Trial': trial
                            })
                        
                        total_peaks += trial_result['num_peaks']
                        total_duration += trial_result['signal_duration']
                        successful_trials += 1
                    else:
                        summary['trial_results'][subject][mvc][trial] = None
                        
                        # Add row for failed trial with NaN values
                        csv_data.append({
                            'Time': 0.0,
                            'fwEMG 3': 0,  # No peaks detected
                            'Subject': subject,
                            'MVC': mvc,
                            'Trial': trial
                        })
                        
        summary['summary_stats'] = {
            'total_trials_analyzed': successful_trials,
            'total_peaks_detected': total_peaks,
            'total_duration_analyzed': total_duration,
            'average_peaks_per_trial': total_peaks / successful_trials if successful_trials > 0 else 0,
            'average_duration_per_trial': total_duration / successful_trials if successful_trials > 0 else 0
        }
        
        # Save to JSON file
        summary_file = self.output_dir / "batch_analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Summary saved to: {summary_file}")
        
        # Save to CSV file with same structure as input
        csv_df = pd.DataFrame(csv_data)
        csv_file = self.output_dir / "peak_analysis_results.csv"
        csv_df.to_csv(csv_file, index=False)
        
        print(f"CSV results saved to: {csv_file}")
        print(f"CSV shape: {csv_df.shape}")
        print(f"CSV columns: {csv_df.columns.tolist()}")
        print(f"CSV dtypes:\n{csv_df.dtypes}")
        
    def print_summary(self):
        """Print a summary of the analysis results."""
        successful_trials = 0
        total_peaks = 0
        total_duration = 0
        
        for subject in self.subjects:
            for mvc in self.mvcs:
                for trial in self.trials:
                    result = self.results['trials'][subject][mvc][trial]
                    if result:
                        successful_trials += 1
                        total_peaks += result['num_peaks']
                        total_duration += result['signal_duration']
                        
        print(f"Successful trials: {successful_trials}")
        print(f"Total peaks detected: {total_peaks}")
        print(f"Total duration analyzed: {total_duration:.2f} seconds")
        print(f"Average peaks per trial: {total_peaks / successful_trials:.1f}" if successful_trials > 0 else "No successful trials")
        print(f"Results saved to: {self.output_dir}")


def main():
    """Main function to run batch analysis."""
    # Default dataset path
    default_dataset_path = Path(__file__).parent.parent / "dataset" / "combined_emg_dorsiflex.csv"
    
    if not default_dataset_path.exists():
        print(f"Default dataset not found at: {default_dataset_path}")
        print("Please provide the path to your dataset.")
        return
        
    # Initialize and run batch analysis
    batch_analyzer = BatchPeakAnalyzer(default_dataset_path)
    batch_analyzer.load_dataset()
    batch_analyzer.run_batch_analysis()


if __name__ == "__main__":
    main()
