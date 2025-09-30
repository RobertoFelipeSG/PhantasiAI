#!/usr/bin/env python3
"""
Run XGBoost Classification After Peak Analysis
==============================================

This script runs the XGBoost classifier directly after the peak analyzer,
using the actual peak analysis results to classify the data.

Usage:
1. First run: python3 batch_peak_analysis_200hz.py
2. Then run: python3 run_xgboost_after_peak_analysis.py
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the XGBoost classifier
try:
    from ..offline_analysis.xgboost_classifier import XGBoostClassifier
except ImportError:
    # For direct execution, add parent directories to path
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from offline_analysis.xgboost_classifier import XGBoostClassifier

def find_latest_peak_results():
    """Find the most recent peak_analysis_results_200hz.csv file."""
    dataset_dir = Path(__file__).parent.parent.parent / "dataset"
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Look for batch_peak_analysis folders
    peak_folders = list(dataset_dir.glob("batch_peak_analysis_*"))
    
    if not peak_folders:
        raise FileNotFoundError(f"No batch_peak_analysis folders found in {dataset_dir}")
    
    # Get the most recent folder
    latest_folder = max(peak_folders, key=lambda x: x.stat().st_mtime)
    
    # Look for peak_analysis_results_200hz.csv in the latest folder
    results_file = latest_folder / "peak_analysis_results_200hz.csv"
    
    if not results_file.exists():
        raise FileNotFoundError(f"peak_analysis_results_200hz.csv not found in {latest_folder}")
    
    return results_file

def run_xgboost_classification():
    """Run XGBoost classification on the latest peak analysis results."""
    print("=" * 60)
    print("Running XGBoost Classification After Peak Analysis")
    print("=" * 60)
    
    # Find the latest peak analysis results
    try:
        peak_results_file = find_latest_peak_results()
        print(f"Found peak analysis results: {peak_results_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run batch_peak_analysis_200hz.py first to generate peak analysis results.")
        return None
    
    # Load the peak analysis results
    df = pd.read_csv(peak_results_file)
    print(f"Loaded {len(df)} peak analysis results")
    print(f"Columns: {df.columns.tolist()}")
    
    # Initialize and train the XGBoost classifier
    print("\nTraining XGBoost classifier...")
    classifier = XGBoostClassifier()
    results = classifier.train(data_path=peak_results_file)
    
    print(f"XGBoost training completed!")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"CV Test Accuracy: {results['cv_test_accuracy']:.4f}")
    
    # Get the peak amplitudes from the results for classification
    print("\n" + "=" * 60)
    print("Classifying Peak Analysis Results")
    print("=" * 60)
    
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
    print(f"Classifying {len(feature_vectors)} feature vectors with shape: {feature_vectors.shape}")
    
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
    
    # Display results
    print(f"\nClassification Results:")
    print(f"Total samples: {len(results_df)}")
    print(f"Predicted classes: {sorted(np.unique(classifications))}")
    
    # Show some examples
    print(f"\nSample classifications:")
    for i in range(min(10, len(results_df))):
        row = results_df.iloc[i]
        print(f"  Sample {i+1}: Subject={row['Subject']}, MVC={row['MVC']}, "
              f"Peak={row['EMG']:.3f}, Min={row['Min_Peak_Amplitude']:.3f}, "
              f"MeanFreq={row['Mean_Frequency']:.1f}Hz, MedianFreq={row['Median_Frequency']:.1f}Hz, "
              f"Predicted={row['Predicted_MVC']}")
    
    # Save results
    output_file = peak_results_file.parent / "xgboost_classification_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Save the trained model
    model_file = peak_results_file.parent / "xgboost_model.pkl"
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

def classify_new_data(classifier, feature_vectors):
    """
    Classify new feature vectors using a trained XGBoost classifier.
    
    Parameters:
    -----------
    classifier : EMGXGBoostClassifier
        Trained XGBoost classifier
    feature_vectors : array-like
        Array of feature vectors to classify (4 features: peak_amplitude, min_peak_amplitude, mean_frequency, median_frequency)
        
    Returns:
    --------
    tuple : (classifications, probabilities)
    """
    if classifier is None:
        print("Error: No trained classifier provided")
        return None, None
    
    # Ensure feature vectors have the correct shape
    feature_vectors = np.array(feature_vectors)
    if feature_vectors.ndim == 1:
        feature_vectors = feature_vectors.reshape(1, -1)
    
    # Classify the new data
    classifications = classifier.predict(feature_vectors)
    probabilities = classifier.predict_proba(feature_vectors)
    
    return classifications, probabilities

def load_trained_model(model_path=None):
    """
    Load a trained XGBoost model for classification.
    
    Parameters:
    -----------
    model_path : str or Path, optional
        Path to the trained model file. If None, will look for the latest model.
        
    Returns:
    --------
    EMGXGBoostClassifier : Trained classifier or None if not found
    """
    if model_path is None:
        # Look for the latest model in the dataset directory
        dataset_dir = Path(__file__).parent.parent.parent / "dataset"
        peak_folders = list(dataset_dir.glob("batch_peak_analysis_*"))
        
        if not peak_folders:
            print("No batch_peak_analysis folders found")
            return None
        
        latest_folder = max(peak_folders, key=lambda x: x.stat().st_mtime)
        model_path = latest_folder / "xgboost_model.pkl"
    
    if not Path(model_path).exists():
        print(f"Model not found at: {model_path}")
        return None
    
    try:
        classifier = EMGXGBoostClassifier(model_path=model_path)
        print(f"Successfully loaded trained model from: {model_path}")
        return classifier
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def main():
    """Main function to run XGBoost classification after peak analysis."""
    print("=" * 60)
    print("XGBoost Classification Pipeline")
    print("=" * 60)
    
    # Run the classification
    results_df, classifier = run_xgboost_classification()
    
    if results_df is not None:
        print(f"\n" + "=" * 60)
        print("Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"Results saved to: {results_df.shape[0]} samples")
        print(f"Model saved and ready for new data classification")
    else:
        print(f"\nPipeline failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
