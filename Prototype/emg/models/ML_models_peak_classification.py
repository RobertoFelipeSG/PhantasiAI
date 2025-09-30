#!/usr/bin/env python3
"""
EMG Peak Classification from Peak Analysis Results
=================================================

This script processes peak_analysis_results.csv (which contains the highest peak values for each trial) 
and classifies the data using multiple ML models with cross-validation for model selection.

The peak_analysis_results.csv file contains:
- Time: Timestamp of the highest peak in each trial
- EMG: Amplitude of the highest peak
- Min_Peak_Amplitude: Minimum peak amplitude in the trial
- Mean_Frequency: Mean frequency of the EMG signal
- Median_Frequency: Median frequency of the EMG signal
- Subject: Subject identifier (S01-S07)
- MVC: MVC percentage (10, 25, 50)
- Trial: Trial number (1, 2, 3)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")
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
    results_file = latest_folder / "peak_analysis_results_200hz.csv"
    
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

def cross_validate_model(model, X_train_scaled, y_train, model_name):
    """Perform cross-validation for a single model."""
    print(f"\n{model_name} Cross-Validation Results:")
    
    # Perform cross-validation
    cv_scores = cross_validate(
        model, 
        X_train_scaled, 
        y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=['accuracy', 'f1_macro'],
        return_train_score=True,
        return_estimator=False
    )
    
    # Calculate metrics
    train_accuracy = cv_scores['train_accuracy'].mean()
    test_accuracy = cv_scores['test_accuracy'].mean()
    train_f1 = cv_scores['train_f1_macro'].mean()
    test_f1 = cv_scores['test_f1_macro'].mean()
    avg_fit_time = cv_scores['fit_time'].mean()
    
    # Print results
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy:  {test_accuracy:.4f}")
    print(f"  Train F1:       {train_f1:.4f}")
    print(f"  Test F1:        {test_f1:.4f}")
    print(f"  Avg Fit Time:   {avg_fit_time:.2f}s")
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'avg_fit_time': avg_fit_time
    }

def analyze_feature_importance(model, feature_names, model_name):
    """Analyze feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        print(f"\n{model_name} Feature Importance:")
        for i, (name, importance) in enumerate(zip(feature_names, importances)):
            print(f"  {name}: {importance:.4f}")
        return importances
    elif hasattr(model, 'coef_'):
        coefficients = np.abs(model.coef_[0])  # Take absolute values
        print(f"\n{model_name} Feature Coefficients (absolute):")
        for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
            print(f"  {name}: {coef:.4f}")
        return coefficients
    else:
        print(f"\n{model_name} does not provide feature importance information.")
        return None

def evaluate_best_model_on_test_set(best_model, X_train_scaled, X_test_scaled, y_train, y_test, model_name, label_encoder, feature_names):
    """Evaluate the best model on the test set."""
    print(f"\n* Final Evaluation on Test Set *")
    
    # Train the model on full training set
    start_time = time.time()
    best_model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Predict on test set
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    
    # Print results
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    
    # Analyze feature importance
    analyze_feature_importance(best_model, feature_names, model_name)
    
    # Convert predictions back to original labels for reporting
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    
    # Print confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test_original, y_pred_original, labels=label_encoder.classes_)
    print(cm)
    
    # Print detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test_original, y_pred_original))
    
    return {
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'training_time': training_time,
        'predictions': y_pred,
        'confusion_matrix': cm
    }

def main():
    """Main function to run the peak-based classification."""
    print("=" * 60)
    print("EMG Peak Classification from Peak Analysis Results")
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
    
    # Data exploration and preprocessing
    print("\n=== Data Overview ===")
    print(f"Total samples: {len(df)}")
    print(f"Subjects: {sorted(df['Subject'].unique())}")
    print(f"MVC levels: {sorted(df['MVC'].unique())}")
    print(f"Trials: {sorted(df['Trial'].unique())}")
    
    # Show feature statistics
    print(f"\n=== Feature Statistics ===")
    feature_cols = ['EMG', 'Min_Peak_Amplitude', 'Mean_Frequency', 'Median_Frequency']
    for col in feature_cols:
        if col in df.columns:
            print(f"{col}:")
            print(f"  Mean: {df[col].mean():.4f}")
            print(f"  Std:  {df[col].std():.4f}")
            print(f"  Min:  {df[col].min():.4f}")
            print(f"  Max:  {df[col].max():.4f}")
    
    # Show class distribution
    print(f"\n=== Class Distribution ===")
    class_counts = df['MVC'].value_counts().sort_index()
    for mvc, count in class_counts.items():
        print(f"MVC {mvc}%: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Create features and labels
    print("\n=== Feature Engineering ===")
    X, y = create_features(df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique labels: {sorted(np.unique(y))}")
    
    feature_names = ['Peak_Amplitude', 'Min_Peak_Amplitude', 'Mean_Frequency', 'Median_Frequency']
    print(f"Feature names: {feature_names}")
    
    # Check for NaN values in features
    print(f"\n=== Data Quality Check ===")
    nan_count = np.isnan(X).sum()
    print(f"NaN values in features: {nan_count}")
    if nan_count > 0:
        print("Warning: NaN values detected in features!")
        print("NaN values per feature:")
        for i, name in enumerate(feature_names):
            nan_feature = np.isnan(X[:, i]).sum()
            if nan_feature > 0:
                print(f"  {name}: {nan_feature} NaN values")
    
    # Remove any remaining NaN values
    if nan_count > 0:
        print("Removing samples with NaN values...")
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        print(f"After removing NaN: {X.shape[0]} samples remaining")
    
    # Check for infinite values
    inf_count = np.isinf(X).sum()
    print(f"Infinite values in features: {inf_count}")
    if inf_count > 0:
        print("Warning: Infinite values detected in features!")
        # Replace infinite values with large finite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        print("Replaced infinite values with finite bounds")
    
    # Encode labels to consecutive integers (required for XGBoost)
    print("\n=== Label Encoding ===")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Original labels: {sorted(np.unique(y))}")
    print(f"Encoded labels: {sorted(np.unique(y_encoded))}")
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Train/test split
    print("\n=== Data Split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"\nTraining set class distribution:")
    print(pd.Series(y_train).value_counts().sort_index())
    print(f"\nTest set class distribution:")
    print(pd.Series(y_test).value_counts().sort_index())
    
    # Define models with optimized parameters
    models = {
        "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "LDA": LinearDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights='distance'),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    else:
        print("XGBoost not available - skipping XGBoost model")
    
    # Cross-validation for model selection
    print("\n" + "=" * 60)
    print("Cross-Validation for Model Selection")
    print("=" * 60)
    
    cv_results = {}
    for name, model in models.items():
        cv_results[name] = cross_validate_model(model, X_train_scaled, y_train, name)
    
    # Select best model based on CV test accuracy
    best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['test_accuracy'])
    best_model = models[best_model_name]
    
    print(f"\nSelected best model based on CV: {best_model_name}")
    print(f"CV Test Accuracy: {cv_results[best_model_name]['test_accuracy']:.4f}")
    print(f"CV Test F1: {cv_results[best_model_name]['test_f1']:.4f}")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print(f"Final Evaluation: {best_model_name}")
    print("=" * 60)
    
    final_results = evaluate_best_model_on_test_set(
        best_model, X_train_scaled, X_test_scaled, y_train, y_test, best_model_name, label_encoder, feature_names
    )
    
    # Model comparison summary
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    print(f"{'Model':<20} {'CV Acc':<10} {'CV F1':<10} {'Test Acc':<10} {'Test F1':<10}")
    print("-" * 60)
    
    for name, results in cv_results.items():
        test_acc = final_results['test_accuracy'] if name == best_model_name else "N/A"
        test_f1 = final_results['test_f1'] if name == best_model_name else "N/A"
        print(f"{name:<20} {results['test_accuracy']:<10.4f} {results['test_f1']:<10.4f} {test_acc:<10} {test_f1:<10}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Dataset: {len(df)} trials from {len(df['Subject'].unique())} subjects")
    print(f"Features: {feature_names}")
    print(f"Best model: {best_model_name}")
    print(f"Final test accuracy: {final_results['test_accuracy']:.4f}")
    print(f"Final test F1: {final_results['test_f1']:.4f}")
    
    # Save results
    print(f"\n=== Results Summary ===")
    print(f"Best performing model: {best_model_name}")
    print(f"Cross-validation accuracy: {cv_results[best_model_name]['test_accuracy']:.4f}")
    print(f"Cross-validation F1: {cv_results[best_model_name]['test_f1']:.4f}")
    print(f"Test set accuracy: {final_results['test_accuracy']:.4f}")
    print(f"Test set F1: {final_results['test_f1']:.4f}")

if __name__ == "__main__":
    main()
