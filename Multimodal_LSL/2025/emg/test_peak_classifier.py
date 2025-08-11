#!/usr/bin/env python3
"""
Test EMG Peak Classifier
=======================

This script tests the EMG peak classifier to ensure it works correctly
with the pre-trained LDA model.
"""

import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add the parent directory to sys.path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from emg.emg_peak_classifier import EMGPeakClassifier, classify_emg_recording

def create_test_emg_data():
    """Create synthetic EMG data for testing."""
    print("Creating synthetic EMG data...")
    
    # Generate test EMG signal
    sample_rate = 220
    duration = 10  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create baseline EMG signal
    emg_signal = np.random.normal(0, 5, len(t))
    
    # Add peaks at different amplitudes (simulating different MVC levels)
    peak_times = [2, 4, 6, 8]  # seconds
    peak_amplitudes = [30, 80, 120, 180]  # µV - different MVC levels
    
    for time_idx, amplitude in zip(peak_times, peak_amplitudes):
        idx = int(time_idx * sample_rate)
        if idx < len(emg_signal):
            # Add a peak with some width
            width = int(0.1 * sample_rate)  # 100ms width
            for i in range(max(0, idx-width), min(len(emg_signal), idx+width)):
                emg_signal[i] = amplitude * np.exp(-((i - idx) / (width/3))**2)
    
    return t, emg_signal

def create_test_csv(t, emg_signal, temp_dir):
    """Create a test CSV file with EMG data."""
    csv_path = temp_dir / "test_emg.csv"
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': t,
        'ch1 (µV)': emg_signal,
        'event': [0] * len(t)
    })
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Test CSV created: {csv_path}")
    
    return csv_path

def test_peak_classifier():
    """Test the peak classifier with synthetic data."""
    print("=" * 60)
    print("Testing EMG Peak Classifier")
    print("=" * 60)
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        t, emg_signal = create_test_emg_data()
        csv_path = create_test_csv(t, emg_signal, temp_path)
        
        # Test the classifier
        print("\nTesting peak classification...")
        try:
            results = classify_emg_recording(csv_path, show_plots=False)
            
            print(f"\nTest Results:")
            print(f"Peaks detected: {results['num_peaks']}")
            print(f"Classifications: {len(results['classifications'])}")
            print(f"Classifier available: {results['classifier_available']}")
            
            if results['classifications']:
                print(f"\nClassification Details:")
                for result in results['classifications']:
                    print(f"  Peak {result['peak_id']}: {result['timestamp']:.3f}s - "
                          f"{result['predicted_class']}% MVC "
                          f"(Amplitude: {result['amplitude']:.1f}µV, "
                          f"Confidence: {result['confidence']:.3f})")
                
                # Check output files
                peaks_file = csv_path.with_name("peaks.txt")
                classifications_file = csv_path.with_name("peak_classifications.csv")
                
                print(f"\nOutput files created:")
                print(f"  Peaks file: {peaks_file.exists()}")
                print(f"  Classifications file: {classifications_file.exists()}")
                
                if classifications_file.exists():
                    # Load and display classification results
                    df_class = pd.read_csv(classifications_file)
                    print(f"\nClassification CSV contents:")
                    print(df_class.to_string(index=False))
                
                return True
            else:
                print("No classifications performed (likely no LDA model available)")
                return False
                
        except Exception as e:
            print(f"Error during testing: {e}")
            return False

def test_model_loading():
    """Test loading the LDA model."""
    print("\n" + "=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    # Check if model file exists
    model_path = Path(__file__).parent / "lda_model.pkl"
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        print("Please train the LDA model first:")
        print("1. python run_peak_analysis.py")
        print("2. python run_lda_after_peak_analysis.py")
        return False
    
    print(f"Model file found: {model_path}")
    
    # Try to load the model
    try:
        from emg.emg_LDA_classifier import EMGLDAClassifier
        classifier = EMGLDAClassifier()
        classifier.load_model(model_path)
        print("Model loaded successfully")
        print(f"Available classes: {classifier.label_encoder.classes_}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def test_standalone_classifier():
    """Test the classifier class directly."""
    print("\n" + "=" * 60)
    print("Testing Standalone Classifier")
    print("=" * 60)
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        t, emg_signal = create_test_emg_data()
        csv_path = create_test_csv(t, emg_signal, temp_path)
        
        # Test the classifier class
        try:
            classifier = EMGPeakClassifier(csv_path)
            
            if classifier.classifier is None:
                print("No LDA model available for classification")
                return False
            
            print("Peak classifier initialized successfully")
            
            # Run classification
            results = classifier.run(show_plots=False, save_results=True, classify_peaks=True)
            
            print(f"Analysis completed:")
            print(f"  Peaks detected: {results['num_peaks']}")
            print(f"  Classifications: {len(results['classifications'])}")
            
            return len(results['classifications']) > 0
            
        except Exception as e:
            print(f"Error testing standalone classifier: {e}")
            return False

if __name__ == "__main__":
    print("EMG Peak Classifier Test")
    print("=" * 60)
    
    # Test model loading first
    if not test_model_loading():
        print("\nModel loading test failed. Please ensure you have a trained LDA model.")
        sys.exit(1)
    
    # Test standalone classifier
    if not test_standalone_classifier():
        print("\nStandalone classifier test failed.")
        sys.exit(1)
    
    # Test full classification pipeline
    if test_peak_classifier():
        print("\nAll tests passed!")
    else:
        print("\nPeak classifier test failed.")
        sys.exit(1)
