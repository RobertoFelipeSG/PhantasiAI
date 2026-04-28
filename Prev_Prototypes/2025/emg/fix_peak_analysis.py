#!/usr/bin/env python3
"""
Fix Peak Analysis - Use Actual Maximum Values
============================================

This script fixes the peak analysis by using the actual maximum values from the original signal
instead of relying on the peak detection algorithm's time conversion which seems to be incorrect.
"""

import pandas as pd
import numpy as np
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
    
    return results_file, latest_folder

def fix_peak_analysis():
    """Fix the peak analysis by using actual maximum values from the original signal."""
    print("=" * 60)
    print("Fixing Peak Analysis - Using Actual Maximum Values")
    print("=" * 60)
    
    # Load original dataset
    original_file = Path(__file__).parent.parent / "dataset" / "combined_emg_dorsiflex.csv"
    if not original_file.exists():
        raise FileNotFoundError(f"Original dataset not found: {original_file}")
    
    print(f"Loading original dataset: {original_file}")
    original_df = pd.read_csv(original_file)
    print(f"Original dataset shape: {original_df.shape}")
    
    # Load current peak analysis results
    peak_file, peak_folder = find_latest_peak_results()
    print(f"Loading current peak results: {peak_file}")
    peak_df = pd.read_csv(peak_file)
    print(f"Current peak results shape: {peak_df.shape}")
    
    # Create fixed results
    fixed_data = []
    
    for _, peak_row in peak_df.iterrows():
        subject = peak_row['Subject']
        mvc = peak_row['MVC']
        trial = peak_row['Trial']
        
        # Get the actual trial data from original dataset
        trial_data = original_df[
            (original_df['Subject'] == subject) & 
            (original_df['MVC'] == mvc) & 
            (original_df['Trial'] == trial)
        ]
        
        if not trial_data.empty:
            # Find actual maximum and minimum values
            max_idx = trial_data['fwEMG 3'].idxmax()
            min_idx = trial_data['fwEMG 3'].idxmin()
            
            actual_max_amplitude = trial_data.loc[max_idx, 'fwEMG 3']
            actual_max_time = trial_data.loc[max_idx, 'Time']
            actual_min_amplitude = trial_data.loc[min_idx, 'fwEMG 3']
            
            # Keep the frequency features from the original analysis
            # (these should be correct since they're calculated from the signal)
            mean_frequency = peak_row['Mean_Frequency']
            median_frequency = peak_row['Median_Frequency']
            
            fixed_data.append({
                'Time': actual_max_time,
                'fwEMG 3': actual_max_amplitude,
                'Min_Peak_Amplitude': actual_min_amplitude,  # Use actual minimum
                'Mean_Frequency': mean_frequency,
                'Median_Frequency': median_frequency,
                'Subject': subject,
                'MVC': mvc,
                'Trial': trial
            })
            
            print(f"{subject}, MVC={mvc}%, Trial={trial}:")
            print(f"  Original detected: {peak_row['fwEMG 3']:.4f} at {peak_row['Time']:.2f}s")
            print(f"  Fixed actual:      {actual_max_amplitude:.4f} at {actual_max_time:.2f}s")
            print(f"  Min amplitude:     {actual_min_amplitude:.4f}")
            print()
        else:
            print(f"Warning: No data found for {subject}, MVC={mvc}%, Trial={trial}")
    
    # Create fixed DataFrame
    fixed_df = pd.DataFrame(fixed_data)
    
    # Save fixed results
    fixed_file = peak_folder / "peak_analysis_results_fixed.csv"
    fixed_df.to_csv(fixed_file, index=False)
    
    print(f"Fixed results saved to: {fixed_file}")
    print(f"Total trials processed: {len(fixed_data)}")
    
    # Verify the fix
    print("\n=== Verification ===")
    correct_count = 0
    for _, row in fixed_df.iterrows():
        subject = row['Subject']
        mvc = row['MVC']
        trial = row['Trial']
        fixed_peak = row['fwEMG 3']
        fixed_time = row['Time']
        
        # Get original data
        trial_data = original_df[
            (original_df['Subject'] == subject) & 
            (original_df['MVC'] == mvc) & 
            (original_df['Trial'] == trial)
        ]
        
        if not trial_data.empty:
            actual_max = trial_data['fwEMG 3'].max()
            actual_max_time = trial_data.loc[trial_data['fwEMG 3'].idxmax(), 'Time']
            
            if abs(fixed_peak - actual_max) < 0.001:  # Small tolerance
                correct_count += 1
    
    accuracy = correct_count / len(fixed_df) * 100
    print(f"Accuracy after fix: {accuracy:.1f}% ({correct_count}/{len(fixed_df)} correct)")
    
    return fixed_df, fixed_file

def run_lda_with_fixed_data(fixed_file):
    """Run LDA classification with the fixed data."""
    print("\n" + "=" * 60)
    print("Running LDA Classification with Fixed Data")
    print("=" * 60)
    
    try:
        # Import the LDA classifier
        from emg_LDA_classifier import EMGLDAClassifier
        
        # Initialize and train the LDA classifier with fixed data
        classifier = EMGLDAClassifier()
        results = classifier.train(data_path=str(fixed_file))
        
        print(f"LDA training completed!")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"CV Test Accuracy: {results['cv_test_accuracy']:.4f}")
        
        # Load the fixed peak analysis results for classification
        df = pd.read_csv(fixed_file)
        
        # Create feature vectors using all 4 features
        feature_vectors = []
        for _, row in df.iterrows():
            feature_vector = [
                row['fwEMG 3'],  # Peak amplitude (now correct)
                row['Min_Peak_Amplitude'],  # Min peak amplitude (now correct)
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
        output_dir = fixed_file.parent
        lda_results_file = output_dir / "lda_classification_results_fixed.csv"
        results_df.to_csv(lda_results_file, index=False)
        
        # Save the trained model
        model_file = output_dir / "lda_model_fixed.pkl"
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
                  f"Peak={row['fwEMG 3']:.3f}, Predicted={row['Predicted_MVC']}")
        
        print("=" * 60)
        print("LDA Classification with fixed data completed successfully!")
        
    except Exception as e:
        print(f"Error running LDA classification: {e}")
        print("Continuing without LDA classification...")

def main():
    """Main function to fix peak analysis and run LDA."""
    try:
        # Fix the peak analysis
        fixed_df, fixed_file = fix_peak_analysis()
        
        # Run LDA with fixed data
        run_lda_with_fixed_data(fixed_file)
        
        print(f"\n=== Fix Complete ===")
        print(f"Fixed peak analysis results: {fixed_file}")
        print(f"Check the LDA results to see if performance improved")
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
