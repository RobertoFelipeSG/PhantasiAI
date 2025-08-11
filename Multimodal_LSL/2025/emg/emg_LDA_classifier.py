#!/usr/bin/env python3
"""
EMG LDA Classifier
=================================================

This script processes peak_analysis_results.csv and trains an LDA model using the same features
as the classification file. It outputs classification arrays for new data.

The peak_analysis_results.csv file contains:
- Time: Timestamp of the highest peak in each trial
- fwEMG 3: Amplitude of the highest peak
- Subject: Subject identifier (S01-S07)
- MVC: MVC percentage (10, 25, 50)
- Trial: Trial number (1, 2, 3)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pathlib import Path
import warnings
import pickle
import time
warnings.filterwarnings('ignore')

class EMGLDAClassifier:
    """LDA classifier for EMG peak classification."""

    def __init__(self, model_path=None):
        """Initialize the LDA classifier."""
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = ['Peak_Amplitude']

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def find_latest_peak_results(self):
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

    def create_features(self, df):
        """Create features from the peak analysis results (same as classification file)."""
        features = []
        labels = []

        # Group by Subject, MVC, and Trial to get one sample per trial
        for (subject, mvc, trial), group in df.groupby(['Subject', 'MVC', 'Trial']):
            if len(group) > 0:
                # Use only the highest peak amplitude as the feature
                peak_amplitude = group['fwEMG 3'].max()

                # Create feature vector (only peak amplitude)
                feature_vector = [peak_amplitude]

                features.append(feature_vector)
                labels.append(mvc)

        return np.array(features), np.array(labels)

    def preprocess_data(self, X, y):
        """Preprocess the data (same as classification file)."""
        # Check for NaN values in features
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"Removing {nan_count} samples with NaN values...")
            valid_mask = ~np.isnan(X).any(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]

        # Check for infinite values
        inf_count = np.isinf(X).sum()
        if inf_count > 0:
            print(f"Replacing {inf_count} infinite values...")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        return X, y_encoded

    def train(self, data_path=None, test_size=0.2, random_state=42):
        """Train the LDA model using the same approach as the classification file."""
        print("=" * 60)
        print("Training LDA Classifier")
        print("=" * 60)

        # Load data
        if data_path is None:
            data_path = self.find_latest_peak_results()

        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Dataset shape: {df.shape}")

        # Create features and labels
        X, y = self.create_features(df)
        print(f"Feature matrix shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Unique labels: {sorted(np.unique(y))}")

        # Preprocess data
        X, y_encoded = self.preprocess_data(X, y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, stratify=y_encoded, test_size=test_size, random_state=random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Train LDA model
        print("\nTraining LDA model...")
        self.model = LinearDiscriminantAnalysis()

        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time

        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='macro')

        print(f"Training completed in {training_time:.2f}s")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")

        # Cross-validation
        print("\nPerforming cross-validation...")
        cv_scores = cross_validate(
            self.model,
            X_train_scaled,
            y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring=['accuracy', 'f1_macro'],
            return_train_score=True
        )

        cv_test_accuracy = cv_scores['test_accuracy'].mean()
        cv_test_f1 = cv_scores['test_f1_macro'].mean()

        print(f"CV Test Accuracy: {cv_test_accuracy:.4f}")
        print(f"CV Test F1: {cv_test_f1:.4f}")

        return {
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'cv_test_accuracy': cv_test_accuracy,
            'cv_test_f1': cv_test_f1,
            'training_time': training_time
        }

    def predict(self, X):
        """Predict classes for new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Convert to numpy array if needed
        X = np.array(X)

        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Reshape to (n_samples, 1) for single feature

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)

        # Convert back to original labels
        predictions_original = self.label_encoder.inverse_transform(predictions)

        return predictions_original

    def predict_proba(self, X):
        """Predict class probabilities for new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Convert to numpy array if needed
        X = np.array(X)

        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Reshape to (n_samples, 1) for single feature

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict probabilities
        probabilities = self.model.predict_proba(X_scaled)

        return probabilities

    def save_model(self, model_path):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {model_path}")

    def load_model(self, model_path):
        """Load a trained model."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']

        print(f"Model loaded from: {model_path}")

def main():
    """Main function to demonstrate the LDA classifier."""
    print("=" * 60)
    print("EMG LDA Classifier")
    print("=" * 60)

    # Initialize classifier
    classifier = EMGLDAClassifier()

    # Train the model
    results = classifier.train()

    # Save the model
    model_path = Path(__file__).parent / "lda_model.pkl"
    classifier.save_model(model_path)

    # Example: Predict on some sample data
    print("\n" + "=" * 60)
    print("Example Predictions")
    print("=" * 60)

    # Sample peak amplitudes (you would replace these with your actual data)
    sample_amplitudes = np.array([0.5, 1.2, 2.1, 0.8, 1.8])

    print(f"Sample peak amplitudes: {sample_amplitudes}")

    # Predict classes
    predictions = classifier.predict(sample_amplitudes)
    print(f"Predicted classes: {predictions}")

    # Predict probabilities
    probabilities = classifier.predict_proba(sample_amplitudes)
    print(f"Class probabilities:")
    for i, (amp, pred, prob) in enumerate(zip(sample_amplitudes, predictions, probabilities)):
        print(f"  Sample {i+1}: Amplitude={amp:.2f}, Class={pred}, Probs={prob}")

    print("\n" + "=" * 60)
    print("Usage:")
    print("1. Train: classifier = EMGLDAClassifier(); classifier.train()")
    print("2. Predict: predictions = classifier.predict(peak_amplitudes)")
    print("3. Save: classifier.save_model('model.pkl')")
    print("4. Load: classifier = EMGLDAClassifier('model.pkl')")
    print("=" * 60)

if __name__ == "__main__":
    main()
