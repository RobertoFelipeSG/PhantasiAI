#!/usr/bin/env python3
"""
Run LDA Classification After Peak Analysis
==========================================

This script runs the LDA classifier directly after the peak analyzer,
using the actual peak analysis results to classify the data.

Usage:
1. First run: python3 batch_peak_analysis.py
2. Then run: python3 run_lda_after_peak_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the LDA classifier
from emg_LDA_classifier import EMGLDAClassifier

def find_latest_peak_results():
    """Find the most recent peak_analysis_results.csv file."""
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

def run_lda_classification():
    """Run LDA classification on the latest peak analysis results."""
    print("=" * 60)
    print("Running LDA Classification After Peak Analysis")
    print("=" * 60)
    
    # Find the latest peak analysis results
    try:
        peak_results_file = find_latest_peak_results()
        print(f"Found peak analysis results: {peak_results_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run batch_peak_analysis.py first to generate peak analysis results.")
        return None
    
    # Load the peak analysis results
    df = pd.read_csv(peak_results_file)
    print(f"Loaded {len(df)} peak analysis results")
    print(f"Columns: {df.columns.tolist()}")
    
    # Initialize and train the LDA classifier
    print("\nTraining LDA classifier...")
    classifier = EMGLDAClassifier()
    results = classifier.train(data_path=peak_results_file)
    
    print(f"LDA training completed!")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"CV Test Accuracy: {results['cv_test_accuracy']:.4f}")
    
    # Get the peak amplitudes from the results for classification
    print("\n" + "=" * 60)
    print("Classifying Peak Analysis Results")
    print("=" * 60)
    
    # Extract peak amplitudes from the results
    peak_amplitudes = df['fwEMG 3'].values
    print(f"Classifying {len(peak_amplitudes)} peak amplitudes...")
    
    # Classify all peak amplitudes
    classifications = classifier.predict(peak_amplitudes)
    probabilities = classifier.predict_proba(peak_amplitudes)
    
    # Create results DataFrame
    results_df = df.copy()
    results_df['Predicted_MVC'] = classifications
    
    # Add probability columns
    class_names = classifier.label_encoder.classes_
    for i, class_name in enumerate(class_names):
        results_df[f'Prob_{class_name}'] = probabilities[:, i]
    
    # Display results
    print(f"\nClassification Results:")
    print(f"Total samples: {len(results_df)}")
    print(f"Predicted classes: {sorted(np.unique(classifications))}")
    
    # Show some examples
    print(f"\nSample classifications:")
    for i in range(min(10, len(results_df))):
        row = results_df.iloc[i]
        print(f"  Sample {i+1}: Subject={row['Subject']}, MVC={row['MVC']}, "
              f"Peak={row['fwEMG 3']:.3f}, Predicted={row['Predicted_MVC']}")
    
    # Save results
    output_file = peak_results_file.parent / "lda_classification_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Save the trained model
    model_file = peak_results_file.parent / "lda_model.pkl"
    classifier.save_model(model_file)
    print(f"Trained model saved to: {model_file}")
    
    # Summary statistics
    print(f"\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    y_true = df['MVC'].values
    y_pred = classifications
    
    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=class_names))
    
    # Accuracy by class
    print(f"\nAccuracy by MVC level:")
    for mvc in sorted(np.unique(y_true)):
        mask = y_true == mvc
        correct = (y_true[mask] == y_pred[mask]).sum()
        total = mask.sum()
        accuracy = correct / total if total > 0 else 0
        print(f"  MVC {mvc}%: {correct}/{total} correct ({accuracy:.3f})")
    
    return results_df, classifier

def classify_new_data(classifier, peak_amplitudes):
    """
    Classify new peak amplitudes using a trained classifier.
    
    Parameters:
    -----------
    classifier : EMGLDAClassifier
        Trained LDA classifier
    peak_amplitudes : array-like
        Array of peak amplitudes to classify
        
    Returns:
    --------
    tuple : (classifications, probabilities)
    """
    if classifier is None:
        print("Error: No trained classifier provided")
        return None, None
    
    # Classify the new data
    classifications = classifier.predict(peak_amplitudes)
    probabilities = classifier.predict_proba(peak_amplitudes)
    
    return classifications, probabilities

def main():
    """Main function to run LDA classification after peak analysis."""
    print("=" * 60)
    print("LDA Classification Pipeline")
    print("=" * 60)
    
    # Run the classification
    results_df, classifier = run_lda_classification()
    

if __name__ == "__main__":
    main()
