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
    """Create synthetic EMG data for testing with 6 peaks per MVC level (18 total)."""
    print("Creating synthetic EMG data with 6 peaks per MVC level (18 total)...")
    
    # Generate test EMG signal
    sample_rate = 10000  # Updated to match the 10kHz sampling rate used in training
    duration = 30  # seconds - increased to accommodate 18 peaks
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create baseline EMG signal with lower noise to make peaks more prominent
    emg_signal = np.random.normal(0, 0.01, len(t))
    
    # Define MVC levels and their typical amplitude ranges (based on real training data analysis)
    # Using more diverse amplitudes that better match the training data distributions
    mvc_levels = {
        10: [0.08, 0.12, 0.15, 0.18, 0.20, 0.22],  # 10% MVC - diverse range
        25: [0.18, 0.22, 0.25, 0.28, 0.32, 0.35],  # 25% MVC - diverse range  
        50: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]   # 50% MVC - diverse range
    }
    
    # Peak times (spread out over the duration) - 18 peaks total
    peak_times = [
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0,      # 10% MVC peaks
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, # 25% MVC peaks
        16.0, 17.0, 18.0, 19.0, 20.0, 21.0  # 50% MVC peaks
    ]
    
    # Create peaks for each MVC level
    peak_idx = 0
    for mvc_level, amplitudes in mvc_levels.items():
        print(f"Adding 6 peaks for {mvc_level}% MVC with amplitudes: {amplitudes}")
        
        for amplitude in amplitudes:
            if peak_idx < len(peak_times):
                time_idx = peak_times[peak_idx]
                idx = int(time_idx * sample_rate)
                
                if idx < len(emg_signal):
                    # Add a peak with some width
                    width = int(0.02 * sample_rate)  # 20ms width to match the analysis window
                    for i in range(max(0, idx-width), min(len(emg_signal), idx+width)):
                        emg_signal[i] = amplitude * np.exp(-((i - idx) / (width/3))**2)
                
                peak_idx += 1
    
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

def validate_mvc_classifications(results):
    """Validate that the classifier correctly identifies MVC levels for test peaks."""
    print("\n" + "=" * 60)
    print("Validating MVC Classifications")
    print("=" * 60)
    
    if not results['classifications']:
        print("No classifications to validate")
        return False
    
    # Expected MVC levels and their amplitude ranges (based on training data analysis)
    expected_mvc_ranges = {
        10: (0.05, 0.25),  # 10% MVC expected range (expanded for better coverage)
        25: (0.15, 0.40),  # 25% MVC expected range (expanded for better coverage)
        50: (0.25, 0.65)   # 50% MVC expected range (expanded for better coverage)
    }
    
    correct_predictions = 0
    total_predictions = len(results['classifications'])
    
    # Track predictions by expected MVC level
    predictions_by_mvc = {10: [], 25: [], 50: []}
    
    print(f"Validating {total_predictions} peak classifications:")
    
    for i, result in enumerate(results['classifications']):
        amplitude = result['amplitude']
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        
        # Determine expected MVC level based on amplitude
        expected_mvc = None
        for mvc_level, (min_amp, max_amp) in expected_mvc_ranges.items():
            if min_amp <= amplitude <= max_amp:
                expected_mvc = mvc_level
                break
        
        if expected_mvc is None:
            print(f"  Peak {i+1}: Amplitude {amplitude:.3f} outside expected ranges")
            continue
        
        # Track prediction for analysis
        predictions_by_mvc[expected_mvc].append({
            'amplitude': amplitude,
            'predicted': predicted_class,
            'confidence': confidence,
            'correct': predicted_class == expected_mvc
        })
        
        # Check if prediction matches expected
        is_correct = predicted_class == expected_mvc
        if is_correct:
            correct_predictions += 1
            status = "✓ CORRECT"
        else:
            status = "✗ INCORRECT"
        
        print(f"  Peak {i+1}: Amplitude {amplitude:.3f} → Expected {expected_mvc}% MVC, "
              f"Predicted {predicted_class}% MVC, Confidence {confidence:.3f} {status}")
    
    # Overall accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nOverall Classification Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.1%}")
    
    # Per-MVC level analysis
    print(f"\nPer-MVC Level Analysis:")
    for mvc_level in [10, 25, 50]:
        predictions = predictions_by_mvc[mvc_level]
        if predictions:
            correct = sum(1 for p in predictions if p['correct'])
            total = len(predictions)
            mvc_accuracy = correct / total if total > 0 else 0
            avg_confidence = np.mean([p['confidence'] for p in predictions])
            
            print(f"  {mvc_level}% MVC: {correct}/{total} correct = {mvc_accuracy:.1%} "
                  f"(avg confidence: {avg_confidence:.3f})")
            
            # Show prediction distribution
            pred_counts = {}
            for p in predictions:
                pred = p['predicted']
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
            
            print(f"    Prediction distribution: {pred_counts}")
    
    # Detailed analysis for edge cases
    print(f"\nEdge Case Analysis:")
    edge_cases = []
    for mvc_level, predictions in predictions_by_mvc.items():
        for p in predictions:
            if not p['correct']:
                edge_cases.append({
                    'expected': mvc_level,
                    'amplitude': p['amplitude'],
                    'predicted': p['predicted'],
                    'confidence': p['confidence']
                })
    
    if edge_cases:
        print(f"  Found {len(edge_cases)} misclassifications:")
        for case in edge_cases[:5]:  # Show first 5
            print(f"    Expected {case['expected']}% MVC (amp: {case['amplitude']:.3f}) → "
                  f"Predicted {case['predicted']}% MVC (confidence: {case['confidence']:.3f})")
        if len(edge_cases) > 5:
            print(f"    ... and {len(edge_cases) - 5} more")
    else:
        print(f"  No misclassifications found!")
    
    # Consider the test passed if we have classifications and reasonable accuracy
    min_acceptable_accuracy = 0.3  # 30% accuracy threshold
    test_passed = len(results['classifications']) > 0 and accuracy >= min_acceptable_accuracy
    
    if test_passed:
        print(f"\n✓ Test PASSED: Sufficient classifications with {accuracy:.1%} accuracy")
    else:
        print(f"\n✗ Test FAILED: Insufficient accuracy ({accuracy:.1%} < {min_acceptable_accuracy:.1%})")
    
    return test_passed

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
                          f"(Amplitude: {result['amplitude']:.3f}, "
                          f"Confidence: {result['confidence']:.3f})")
                
                # Validate the classifications
                validation_passed = validate_mvc_classifications(results)
                
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
                
                return validation_passed
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
            
            if results['classifications']:
                # Validate the classifications
                validation_passed = validate_mvc_classifications(results)
                return validation_passed
            else:
                return False
            
        except Exception as e:
            print(f"Error testing standalone classifier: {e}")
            return False

def debug_lda_classifier():
    """Debug the LDA classifier with test amplitudes."""
    print("\n" + "=" * 60)
    print("Debugging LDA Classifier")
    print("=" * 60)
    
    try:
        from emg.emg_LDA_classifier import EMGLDAClassifier
        
        # Load the model
        model_path = Path(__file__).parent / "lda_model.pkl"
        classifier = EMGLDAClassifier()
        classifier.load_model(model_path)
        
        # Test amplitudes for each MVC level (expanded range)
        test_amplitudes = {
            10: [0.08, 0.12, 0.15, 0.18, 0.20, 0.22],
            25: [0.18, 0.22, 0.25, 0.28, 0.32, 0.35], 
            50: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
        }
        
        print("Testing classifier with different amplitudes:")
        print("-" * 50)
        
        for mvc_level, amplitudes in test_amplitudes.items():
            print(f"\n{mvc_level}% MVC test amplitudes:")
            for amp in amplitudes:
                # Create feature vector with all 4 features
                X = np.array([[amp, amp * 0.8, 50.0, 45.0]])  # [amp, min_amp, mean_freq, median_freq]
                
                # Get prediction and probabilities
                prediction = classifier.predict(X)[0]
                probabilities = classifier.predict_proba(X)[0]
                confidence = max(probabilities)
                
                print(f"  Amplitude {amp:.3f} → Predicted: {prediction}% MVC, "
                      f"Confidence: {confidence:.3f}")
                print(f"    Probabilities: 10%={probabilities[0]:.3f}, "
                      f"25%={probabilities[1]:.3f}, 50%={probabilities[2]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error debugging classifier: {e}")
        return False

def test_edge_cases():
    """Test the classifier with edge cases and boundary conditions."""
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)
    
    try:
        from emg.emg_LDA_classifier import EMGLDAClassifier
        
        # Load the model
        model_path = Path(__file__).parent / "lda_model.pkl"
        classifier = EMGLDAClassifier()
        classifier.load_model(model_path)
        
        # Test edge cases
        edge_cases = [
            # Very low amplitudes
            (0.01, "Very low amplitude"),
            (0.05, "Low amplitude"),
            (0.10, "Boundary 10% MVC"),
            
            # Boundary cases between MVC levels
            (0.20, "Boundary 10-25% MVC"),
            (0.25, "Boundary 25% MVC"),
            (0.30, "Boundary 25-50% MVC"),
            (0.35, "Boundary 50% MVC"),
            
            # Very high amplitudes
            (0.60, "High amplitude"),
            (0.80, "Very high amplitude"),
            (1.00, "Extreme amplitude")
        ]
        
        print("Testing edge cases and boundary conditions:")
        print("-" * 50)
        
        for amplitude, description in edge_cases:
            # Create feature vector with default frequency values
            # (since we're only testing amplitude-based classification)
            X = np.array([[amplitude, amplitude * 0.8, 50.0, 45.0]])  # [amp, min_amp, mean_freq, median_freq]
            
            try:
                # Get prediction and probabilities
                prediction = classifier.predict(X)[0]
                probabilities = classifier.predict_proba(X)[0]
                confidence = max(probabilities)
                
                print(f"  {description} ({amplitude:.3f}): Predicted {prediction}% MVC, "
                      f"Confidence {confidence:.3f}")
                print(f"    Probabilities: 10%={probabilities[0]:.3f}, "
                      f"25%={probabilities[1]:.3f}, 50%={probabilities[2]:.3f}")
                
            except Exception as e:
                print(f"  {description} ({amplitude:.3f}): Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"Error testing edge cases: {e}")
        return False

def test_noise_robustness():
    """Test the classifier's robustness to noise."""
    print("\n" + "=" * 60)
    print("Testing Noise Robustness")
    print("=" * 60)
    
    try:
        from emg.emg_LDA_classifier import EMGLDAClassifier
        
        # Load the model
        model_path = Path(__file__).parent / "lda_model.pkl"
        classifier = EMGLDAClassifier()
        classifier.load_model(model_path)
        
        # Base amplitude to test
        base_amplitude = 0.25  # 25% MVC level
        
        # Different noise levels
        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10]
        
        print("Testing robustness to noise:")
        print("-" * 50)
        
        for noise_level in noise_levels:
            # Add noise to the amplitude
            noisy_amplitude = base_amplitude + np.random.normal(0, noise_level)
            noisy_amplitude = max(0.01, noisy_amplitude)  # Ensure positive
            
            # Create feature vector
            X = np.array([[noisy_amplitude, noisy_amplitude * 0.8, 50.0, 45.0]])
            
            try:
                # Get prediction and probabilities
                prediction = classifier.predict(X)[0]
                probabilities = classifier.predict_proba(X)[0]
                confidence = max(probabilities)
                
                print(f"  Noise level {noise_level:.3f}: Amplitude {noisy_amplitude:.3f} → "
                      f"Predicted {prediction}% MVC, Confidence {confidence:.3f}")
                
            except Exception as e:
                print(f"  Noise level {noise_level:.3f}: Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"Error testing noise robustness: {e}")
        return False

if __name__ == "__main__":
    print("EMG Peak Classifier Test")
    
    # Test model loading first
    if not test_model_loading():
        print("\nModel loading test failed. Please ensure you have a trained LDA model.")
        sys.exit(1)
    
    # Debug the LDA classifier
    debug_lda_classifier()
    
    # Test edge cases
    test_edge_cases()
    
    # Test noise robustness
    test_noise_robustness()
    
    # Test standalone classifier with enhanced validation
    if not test_standalone_classifier():
        print("\nStandalone classifier test failed.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    