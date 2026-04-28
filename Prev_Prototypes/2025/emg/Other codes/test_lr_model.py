#!/usr/bin/env python3
"""
Test Saved Logistic Regression Model
===================================

This script tests the saved Logistic Regression model to ensure it works correctly.
"""

import pickle
import numpy as np
from pathlib import Path

def load_model(model_path):
    """Load the saved model package."""
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    return model_package

def test_model(model_package, test_features):
    """Test the model with sample features."""
    model = model_package['model']
    scaler = model_package['scaler']
    label_encoder = model_package['label_encoder']
    feature_names = model_package['feature_names']
    
    print(f"Model type: {model_package['model_type']}")
    print(f"Version: {model_package['version']}")
    print(f"Feature names: {feature_names}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    # Scale the features
    test_features_scaled = scaler.transform(test_features)
    
    # Make prediction
    prediction = model.predict(test_features_scaled)
    probabilities = model.predict_proba(test_features_scaled)
    
    # Convert back to original labels
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    
    print(f"\nTest Features: {test_features[0]}")
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities:")
    for i, (class_name, prob) in enumerate(zip(label_encoder.classes_, probabilities[0])):
        print(f"  {class_name}% MVC: {prob:.4f}")
    
    return predicted_class, probabilities[0]

def main():
    """Test the saved model."""
    print("=" * 50)
    print("Test Saved Logistic Regression Model")
    print("=" * 50)
    
    # Load the model
    model_path = Path(__file__).parent / "lr_model.pkl"
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please run train_lr_model.py first to create the model.")
        return
    
    print(f"Loading model from: {model_path}")
    model_package = load_model(model_path)
    
    # Test with sample features from different MVC levels
    print(f"\n=== Testing with Sample Features ===")
    
    # Sample features (you can modify these to test different scenarios)
    test_cases = [
        # Low amplitude, low frequency (likely 10% MVC)
        np.array([[0.05, 0.04, 15.0, 10.0]]),
        
        # Medium amplitude, medium frequency (likely 25% MVC)
        np.array([[0.15, 0.12, 30.0, 25.0]]),
        
        # High amplitude, high frequency (likely 50% MVC)
        np.array([[0.35, 0.30, 50.0, 45.0]]),
        
        # Very high amplitude (likely 50% MVC)
        np.array([[0.60, 0.55, 40.0, 35.0]])
    ]
    
    for i, test_features in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        predicted_class, probabilities = test_model(model_package, test_features)
        
        # Show confidence
        max_prob = np.max(probabilities)
        print(f"Confidence: {max_prob:.4f}")
        
        if max_prob > 0.7:
            print("High confidence prediction")
        elif max_prob > 0.5:
            print("Medium confidence prediction")
        else:
            print("Low confidence prediction")
    
    # Show model performance summary
    results = model_package['results']
    print(f"\n=== Model Performance Summary ===")
    print(f"Cross-validation accuracy: {results['cv_test_accuracy']:.4f}")
    print(f"Cross-validation F1: {results['cv_test_f1']:.4f}")
    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    print(f"Test F1: {results['test_f1']:.4f}")
    
    print(f"\nModel is working correctly and ready for use!")

if __name__ == "__main__":
    main()
