#!/usr/bin/env python3
"""
EMG Peak Detection Verification Script
=====================================

This script loads the original combined_emg_dorsiflex.csv dataset and the peak_analysis_results.csv
file, then plots each trial with the detected peaks overlaid to verify that the peak detection
is working correctly.

The script will:
1. Load both the original dataset and peak analysis results
2. Plot each trial (Subject, MVC, Trial combination)
3. Overlay the detected peaks on the original EMG signal
4. Save plots for verification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def find_latest_peak_results():
    """Find the most recent peak_analysis_results.csv file in the dataset directory."""
    dataset_dir = Path(__file__).parent.parent / "dataset"
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Look for batch_peak_analysis folders
    peak_folders = list(dataset_dir.glob("batch_peak_analysis_*"))
    
    if not peak_folders:
        raise FileNotFoundError(f"No batch_peak_analysis folders found in {dataset_dir}")
    
    # Get the most recent folder
    latest_folder = max(peak_folders, key=lambda x: x.stat().st_mtime)
    
    # Look for peak_analysis_results.csv in the latest folder
    results_file = latest_folder / "peak_analysis_results.csv"
    
    if not results_file.exists():
        raise FileNotFoundError(f"peak_analysis_results.csv not found in {latest_folder}")
    
    return results_file

def load_datasets():
    """Load both the original dataset and peak analysis results."""
    # Load original dataset
    original_file = Path(__file__).parent.parent / "dataset" / "combined_emg_dorsiflex.csv"
    if not original_file.exists():
        raise FileNotFoundError(f"Original dataset not found: {original_file}")
    
    print(f"Loading original dataset: {original_file}")
    original_df = pd.read_csv(original_file)
    print(f"Original dataset shape: {original_df.shape}")
    
    # Load peak analysis results
    peak_file = find_latest_peak_results()
    print(f"Loading peak results: {peak_file}")
    peak_df = pd.read_csv(peak_file)
    print(f"Peak results shape: {peak_df.shape}")
    
    return original_df, peak_df

def plot_trial_with_peaks(original_df, peak_df, subject, mvc, trial, save_dir):
    """Plot a single trial with detected peaks overlaid."""
    # Filter original data for this trial
    trial_data = original_df[
        (original_df['Subject'] == subject) & 
        (original_df['MVC'] == mvc) & 
        (original_df['Trial'] == trial)
    ].copy()
    
    if trial_data.empty:
        print(f"No data found for {subject}, MVC={mvc}%, Trial={trial}")
        return False
    
    # Filter peak data for this trial
    trial_peaks = peak_df[
        (peak_df['Subject'] == subject) & 
        (peak_df['MVC'] == mvc) & 
        (peak_df['Trial'] == trial)
    ].copy()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot original EMG signal
    ax.plot(trial_data['Time'], trial_data['fwEMG 3'], 
            color='blue', alpha=0.7, linewidth=1, label='Original EMG Signal')
    
    # Plot detected peaks
    if not trial_peaks.empty:
        # Get the highest peak for this trial
        highest_peak_idx = trial_peaks['fwEMG 3'].idxmax()
        highest_peak = trial_peaks.loc[highest_peak_idx]
        
        # Plot all detected peaks
        ax.scatter(trial_peaks['Time'], trial_peaks['fwEMG 3'], 
                  color='red', s=50, alpha=0.8, label='All Detected Peaks')
        
        # Highlight the highest peak
        ax.scatter(highest_peak['Time'], highest_peak['fwEMG 3'], 
                  color='green', s=100, marker='*', edgecolors='black', linewidth=2,
                  label=f'Highest Peak: {highest_peak["fwEMG 3"]:.2f}')
        
        # Add text annotation for highest peak
        ax.annotate(f'Highest: {highest_peak["fwEMG 3"]:.2f}', 
                   xy=(highest_peak['Time'], highest_peak['fwEMG 3']),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        peak_info = f"Peaks detected: {len(trial_peaks)}"
    else:
        peak_info = "No peaks detected"
    
    # Customize the plot
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('EMG Amplitude (µV)')
    ax.set_title(f'EMG Signal with Peak Detection\n{subject}, MVC={mvc}%, Trial={trial}\n{peak_info}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"""Signal Statistics:
Duration: {trial_data['Time'].max() - trial_data['Time'].min():.2f}s
Mean: {trial_data['fwEMG 3'].mean():.2f}
Max: {trial_data['fwEMG 3'].max():.2f}
Min: {trial_data['fwEMG 3'].min():.2f}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"{subject}_MVC{mvc}_Trial{trial}_peaks.png"
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

def create_summary_plot(original_df, peak_df, save_dir):
    """Create a summary plot showing peak detection across all trials."""
    # Get unique combinations
    combinations = original_df.groupby(['Subject', 'MVC', 'Trial']).size().reset_index()
    
    # Create a large subplot grid
    n_combinations = len(combinations)
    n_cols = 3
    n_rows = (n_combinations + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for idx, (_, row) in enumerate(combinations.iterrows()):
        subject, mvc, trial = row['Subject'], row['MVC'], row['Trial']
        ax = axes_flat[idx]
        
        # Get trial data
        trial_data = original_df[
            (original_df['Subject'] == subject) & 
            (original_df['MVC'] == mvc) & 
            (original_df['Trial'] == trial)
        ]
        
        # Get peak data
        trial_peaks = peak_df[
            (peak_df['Subject'] == subject) & 
            (peak_df['MVC'] == mvc) & 
            (peak_df['Trial'] == trial)
        ]
        
        if not trial_data.empty:
            # Plot EMG signal
            ax.plot(trial_data['Time'], trial_data['fwEMG 3'], 
                   color='blue', alpha=0.6, linewidth=0.8)
            
            # Plot peaks
            if not trial_peaks.empty:
                ax.scatter(trial_peaks['Time'], trial_peaks['fwEMG 3'], 
                          color='red', s=20, alpha=0.8)
                
                # Highlight highest peak
                highest_peak_idx = trial_peaks['fwEMG 3'].idxmax()
                highest_peak = trial_peaks.loc[highest_peak_idx]
                ax.scatter(highest_peak['Time'], highest_peak['fwEMG 3'], 
                          color='green', s=50, marker='*', edgecolors='black')
            
            ax.set_title(f'{subject}, MVC={mvc}%, T{trial}')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Hide empty subplots
    for idx in range(n_combinations, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle('EMG Peak Detection Verification - All Trials', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save the summary plot
    summary_path = save_dir / "peak_detection_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return summary_path

def analyze_peak_accuracy(original_df, peak_df):
    """Analyze the accuracy of peak detection."""
    print("\n=== Peak Detection Analysis ===")
    
    # Group by trial
    trial_groups = original_df.groupby(['Subject', 'MVC', 'Trial'])
    
    accuracy_results = []
    
    for (subject, mvc, trial), trial_data in trial_groups:
        # Get peaks for this trial
        trial_peaks = peak_df[
            (peak_df['Subject'] == subject) & 
            (peak_df['MVC'] == mvc) & 
            (peak_df['Trial'] == trial)
        ]
        
        if not trial_peaks.empty:
            # Get the highest peak
            highest_peak_idx = trial_peaks['fwEMG 3'].idxmax()
            highest_peak = trial_peaks.loc[highest_peak_idx]
            
            # Find the corresponding value in original data
            # Find the closest time point
            time_diff = np.abs(trial_data['Time'] - highest_peak['Time'])
            closest_idx = time_diff.idxmin()
            original_value = trial_data.loc[closest_idx, 'fwEMG 3']
            
            # Check if the peak value matches
            is_correct = abs(highest_peak['fwEMG 3'] - original_value) < 0.01  # Small tolerance
            
            accuracy_results.append({
                'Subject': subject,
                'MVC': mvc,
                'Trial': trial,
                'Peak_Time': highest_peak['Time'],
                'Peak_Value': highest_peak['fwEMG 3'],
                'Original_Value': original_value,
                'Difference': abs(highest_peak['fwEMG 3'] - original_value),
                'Is_Correct': is_correct
            })
    
    accuracy_df = pd.DataFrame(accuracy_results)
    
    if not accuracy_df.empty:
        print(f"Total trials analyzed: {len(accuracy_df)}")
        print(f"Correct peak detections: {accuracy_df['Is_Correct'].sum()}")
        print(f"Accuracy: {accuracy_df['Is_Correct'].mean():.2%}")
        
        print(f"\nPeak detection statistics:")
        print(accuracy_df['Difference'].describe())
        
        # Show problematic detections
        problematic = accuracy_df[accuracy_df['Difference'] > 1.0]
        if not problematic.empty:
            print(f"\nTrials with large differences (>1.0):")
            print(problematic[['Subject', 'MVC', 'Trial', 'Peak_Value', 'Original_Value', 'Difference']])
    
    return accuracy_df

def main():
    """Main function to verify peak detection."""
    print("=" * 60)
    print("EMG Peak Detection Verification")
    print("=" * 60)
    
    try:
        # Load datasets
        original_df, peak_df = load_datasets()
        
        # Create output directory
        output_dir = Path("peak_verification_plots")
        output_dir.mkdir(exist_ok=True)
        print(f"\nSaving plots to: {output_dir}")
        
        # Get unique combinations
        combinations = original_df.groupby(['Subject', 'MVC', 'Trial']).size().reset_index()
        print(f"\nFound {len(combinations)} unique trial combinations")
        
        # Plot individual trials
        print("\n=== Generating Individual Trial Plots ===")
        successful_plots = 0
        
        for _, row in combinations.iterrows():
            subject, mvc, trial = row['Subject'], row['MVC'], row['Trial']
            print(f"Plotting {subject}, MVC={mvc}%, Trial={trial}...")
            
            if plot_trial_with_peaks(original_df, peak_df, subject, mvc, trial, output_dir):
                successful_plots += 1
        
        print(f"Successfully created {successful_plots} individual trial plots")
        
        # Create summary plot
        print("\n=== Generating Summary Plot ===")
        summary_path = create_summary_plot(original_df, peak_df, output_dir)
        print(f"Summary plot saved to: {summary_path}")
        
        # Analyze peak accuracy
        accuracy_df = analyze_peak_accuracy(original_df, peak_df)
        
        # Save accuracy results
        accuracy_path = output_dir / "peak_accuracy_analysis.csv"
        accuracy_df.to_csv(accuracy_path, index=False)
        print(f"Accuracy analysis saved to: {accuracy_path}")
        
        # Create accuracy summary plot
        if not accuracy_df.empty:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            accuracy_df['Difference'].hist(bins=20, alpha=0.7)
            plt.xlabel('Difference between Peak and Original Value')
            plt.ylabel('Frequency')
            plt.title('Distribution of Peak Detection Differences')
            
            plt.subplot(2, 2, 2)
            accuracy_df.boxplot(column='Difference', by='MVC', ax=plt.gca())
            plt.title('Peak Detection Differences by MVC Level')
            plt.suptitle('')
            
            plt.subplot(2, 2, 3)
            accuracy_df.groupby('Subject')['Is_Correct'].mean().plot(kind='bar')
            plt.title('Peak Detection Accuracy by Subject')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 4)
            accuracy_df.groupby('MVC')['Is_Correct'].mean().plot(kind='bar')
            plt.title('Peak Detection Accuracy by MVC Level')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            accuracy_plot_path = output_dir / "peak_accuracy_summary.png"
            plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Accuracy summary plot saved to: {accuracy_plot_path}")
        
        print(f"\n=== Verification Complete ===")
        print(f"All plots saved to: {output_dir}")
        print(f"Check the individual trial plots to verify peak detection accuracy")
        print(f"Review the summary plot for an overview of all trials")
        print(f"Examine the accuracy analysis for quantitative verification")
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
