#!/usr/bin/env python3
"""
Simple LDA Classifier Function
==============================

This module provides a simple function to classify peak amplitudes using LDA.
It automatically loads the latest trained model and returns classification arrays.
"""

import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the LDA classifier
import sys
sys.path.append('.')
from emg_LDA_classifier import EMGLDAClassifier

def get_classification_array(peak_amplitudes, model_path=None):
    """
    Get classification array for peak amplitudes using LDA.
    
    Parameters:
    -----------
    peak_amplitudes : array-like
        Array of peak amplitudes to classify
    model_path : str or Path, optional
        Path to trained model. If None, uses the latest model.
        
    Returns:
    --------
    tuple : (classifications, probabilities)
        - classifications: array of predicted MVC classes (10, 25, 50)
        - probabilities: array of class probabilities
    """
    try:
        # Load the classifier
        if model_path is None:
            # Find the latest model
            dataset_dir = Path(__file__).parent.parent / "dataset"
            peak_folders = list(dataset_dir.glob("batch_peak_analysis_*"))
            
            if not peak_folders:
                raise FileNotFoundError("No batch_peak_analysis folders found")
            
            latest_folder = max(peak_folders, key=lambda x: x.stat().st_mtime)
            model_path = latest_folder / "lda_model.pkl"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the trained classifier
        classifier = EMGLDAClassifier(str(model_path))
        
        # Convert input to numpy array
        peak_amplitudes = np.array(peak_amplitudes)
        
        # Classify the data
        classifications = classifier.predict(peak_amplitudes)
        probabilities = classifier.predict_proba(peak_amplitudes)
        
        return classifications, probabilities
        
    except Exception as e:
        print(f"Error in classification: {e}")
        return None, None

def classify_single_peak(peak_amplitude, model_path=None):
    """
    Classify a single peak amplitude.
    
    Parameters:
    -----------
    peak_amplitude : float
        Single peak amplitude to classify
    model_path : str or Path, optional
        Path to trained model. If None, uses the latest model.
        
    Returns:
    --------
    tuple : (classification, probabilities)
        - classification: predicted MVC class (10, 25, 50)
        - probabilities: array of class probabilities
    """
    classifications, probabilities = get_classification_array([peak_amplitude], model_path)
    
    if classifications is not None:
        return classifications[0], probabilities[0]
    else:
        return None, None

def classify_multiple_peaks(peak_amplitudes, model_path=None):
    """
    Classify multiple peak amplitudes.
    
    Parameters:
    -----------
    peak_amplitudes : array-like
        Array of peak amplitudes to classify
    model_path : str or Path, optional
        Path to trained model. If None, uses the latest model.
        
    Returns:
    --------
    tuple : (classifications, probabilities)
        - classifications: array of predicted MVC classes
        - probabilities: array of class probabilities
    """
    return get_classification_array(peak_amplitudes, model_path)

# Example usage functions
def example_usage():
    """Example of how to use the classification functions."""
    print("=" * 60)
    print("LDA Classification Example")
    print("=" * 60)
    
    # Example 1: Classify a single peak
    print("\n1. Classifying a single peak amplitude:")
    single_peak = 1.5
    classification, probabilities = classify_single_peak(single_peak)
    
    if classification is not None:
        print(f"Peak amplitude: {single_peak}")
        print(f"Predicted class: {classification}")
        print(f"Probabilities: {probabilities}")
    
    # Example 2: Classify multiple peaks
    print("\n2. Classifying multiple peak amplitudes:")
    multiple_peaks = [0.5, 1.2, 2.1, 0.8, 1.8]
    classifications, probabilities = classify_multiple_peaks(multiple_peaks)
    
    if classifications is not None:
        print(f"Peak amplitudes: {multiple_peaks}")
        print(f"Classifications: {classifications}")
        print(f"Probabilities shape: {probabilities.shape}")
        
        # Show detailed results
        for i, (peak, classification, prob) in enumerate(zip(multiple_peaks, classifications, probabilities)):
            print(f"  Peak {i+1}: {peak:.2f} -> Class {classification} (Probs: {prob})")
    
    # Example 3: Classify array of peaks
    print("\n3. Classifying array of peaks:")
    peak_array = np.array([0.3, 0.7, 1.1, 1.5, 1.9, 2.3])
    classifications, probabilities = classify_multiple_peaks(peak_array)
    
    if classifications is not None:
        print(f"Peak array: {peak_array}")
        print(f"Classification array: {classifications}")
        print(f"All classifications: {list(classifications)}")

if __name__ == "__main__":
    example_usage()
