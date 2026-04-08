#!/usr/bin/env python3
"""
Train and Save Logistic Regression Model for EMG Classification
==============================================================

This script trains a Logistic Regression model on the peak analysis results
and saves it in a format compatible with the test_peak_classifier.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pickle
import time
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

def create_features(df):
    """Create enhanced features from the peak analysis results."""
    features = []
    labels = []
    
    # Group by Subject, MVC, and Trial to get one sample per trial
    for (subject, mvc, trial), group in df.groupby(['Subject', 'MVC', 'Trial']):
        if len(group) > 0:
            # Extract enhanced features
            peak_amplitude = group['EMG'].max()  # Maximum peak amplitude
            min_peak_amplitude = group['Min_Peak_Amplitude'].iloc[0]  # Minimum peak amplitude
            mean_frequency = group['Mean_Frequency'].iloc[0]  # Mean frequency
            median_frequency = group['Median_Frequency'].iloc[0]  # Median frequency
            
            # Create feature vector with all 4 features
            feature_vector = [
                peak_amplitude,
                min_peak_amplitude,
                mean_frequency,
                median_frequency
            ]
            
            features.append(feature_vector)
            labels.append(mvc)
    
    return np.array(features), np.array(labels)

def train_logistic_regression_model(X, y, feature_names):
    """Train a Logistic Regression model with cross-validation."""
    print("Training Logistic Regression model...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Cross-validation
    print("Performing cross-validation...")
    cv_scores = cross_validate(
        LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        X_train_scaled,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=['accuracy', 'f1_macro'],
        return_train_score=True
    )
    
    print(f"CV Train Accuracy: {cv_scores['train_accuracy'].mean():.4f}")
    print(f"CV Test Accuracy:  {cv_scores['test_accuracy'].mean():.4f}")
    print(f"CV Train F1:       {cv_scores['train_f1_macro'].mean():.4f}")
    print(f"CV Test F1:        {cv_scores['test_f1_macro'].mean():.4f}")
    
    # Train final model on full training set
    print("Training final model on full training set...")
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    
    # Feature importance
    coefficients = np.abs(model.coef_[0])
    print(f"\nFeature Importance (absolute coefficients):")
    for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
        print(f"  {name}: {coef:.4f}")
    
    # Confusion matrix
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test_original, y_pred_original, labels=label_encoder.classes_)
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test_original, y_pred_original))
    
    return model, scaler, label_encoder, {
        'cv_train_accuracy': cv_scores['train_accuracy'].mean(),
        'cv_test_accuracy': cv_scores['test_accuracy'].mean(),
        'cv_train_f1': cv_scores['train_f1_macro'].mean(),
        'cv_test_f1': cv_scores['test_f1_macro'].mean(),
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'training_time': training_time,
        'confusion_matrix': cm
    }

def save_model(model, scaler, label_encoder, results, feature_names, output_path):
    """Save the trained model and preprocessing components."""
    print(f"\nSaving model to: {output_path}")
    
    # Create a model package
    model_package = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'results': results,
        'model_type': 'LogisticRegression',
        'version': '1.0'
    }
    
    # Save the model package
    with open(output_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"Model saved successfully!")
    
    # Also save a summary file
    summary_path = output_path.with_suffix('.txt')
    with open(summary_path, 'w') as f:
        f.write("Logistic Regression Model Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model Type: Logistic Regression\n")
        f.write(f"Features: {feature_names}\n")
        f.write(f"Classes: {list(label_encoder.classes_)}\n\n")
        f.write(f"Cross-validation Results:\n")
        f.write(f"  Train Accuracy: {results['cv_train_accuracy']:.4f}\n")
        f.write(f"  Test Accuracy:  {results['cv_test_accuracy']:.4f}\n")
        f.write(f"  Train F1:       {results['cv_train_f1']:.4f}\n")
        f.write(f"  Test F1:        {results['cv_test_f1']:.4f}\n\n")
        f.write(f"Final Test Results:\n")
        f.write(f"  Test Accuracy: {results['test_accuracy']:.4f}\n")
        f.write(f"  Test F1:       {results['test_f1']:.4f}\n")
        f.write(f"  Training Time: {results['training_time']:.2f}s\n\n")
        f.write(f"Feature Importance:\n")
        coefficients = np.abs(model.coef_[0])
        for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
            f.write(f"  {name}: {coef:.4f}\n")
    
    print(f"Model summary saved to: {summary_path}")

def main():
    """Main function to train and save the Logistic Regression model."""
    print("=" * 60)
    print("Train and Save Logistic Regression Model")
    print("=" * 60)
    
    # Load peak analysis results
    try:
        peak_results_file = find_latest_peak_results()
        print(f"Loading peak results from: {peak_results_file}")
        
        df = pd.read_csv(peak_results_file)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the batch peak analysis first to generate the results file.")
        return
    
    # Data overview
    print(f"\n=== Data Overview ===")
    print(f"Total samples: {len(df)}")
    print(f"Subjects: {sorted(df['Subject'].unique())}")
    print(f"MVC levels: {sorted(df['MVC'].unique())}")
    print(f"Trials: {sorted(df['Trial'].unique())}")
    
    # Create features and labels
    print(f"\n=== Feature Engineering ===")
    X, y = create_features(df)
    feature_names = ['Peak_Amplitude', 'Min_Peak_Amplitude', 'Mean_Frequency', 'Median_Frequency']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique labels: {sorted(np.unique(y))}")
    print(f"Feature names: {feature_names}")
    
    # Data quality check
    print(f"\n=== Data Quality Check ===")
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"NaN values: {nan_count}")
    print(f"Infinite values: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("Cleaning data...")
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
        X = X[valid_mask]
        y = y[valid_mask]
        print(f"After cleaning: {X.shape[0]} samples remaining")
    
    # Train the model
    print(f"\n=== Model Training ===")
    model, scaler, label_encoder, results = train_logistic_regression_model(X, y, feature_names)
    
    # Save the model
    print(f"\n=== Model Saving ===")
    output_path = Path(__file__).parent / "lr_model.pkl"
    save_model(model, scaler, label_encoder, results, feature_names, output_path)
    
    # Final summary
    print(f"\n=== Final Summary ===")
    print(f"Model saved to: {output_path}")
    print(f"Best CV Test Accuracy: {results['cv_test_accuracy']:.4f}")
    print(f"Best CV Test F1: {results['cv_test_f1']:.4f}")
    print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Final Test F1: {results['test_f1']:.4f}")
    print(f"Model is ready for use with test_peak_classifier!")

if __name__ == "__main__":
    main()
