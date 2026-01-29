import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from typing import List, Dict
from config.connection_manager import logging

warnings.filterwarnings('ignore')

class PeakClassifier:
    """
    Clean EMG peak classifier that loads features and classifies using XGBoost.
    
    This class:
    1. Loads features from CSV (created by emg_feature_extractor.py)
    2. Loads pre-trained XGBoost model
    3. Classifies peaks
    4. Saves results in the same format as the original peak_classifier.py
    """
    
    def __init__(self, base_path):
        """
        Initialize the peak classifier
        """
        model_path = base_path / "models" / "xgboost_model.pkl"
        self.model_path = str(model_path)
        
        # Model components (loaded from pickle)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.emg_features = []
        self.tangential_features = []
        self.additional_features = []
        
        self.features_df = None
        self.classification_results = []

        # Load model
        self._load_model(self.model_path)
    
    def _load_model(self, model_path: str):
        """
        Load a pre-trained XGBoost model directly from pickle file
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model data from pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract model components
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            self.emg_features = model_data.get('emg_features', [])
            self.tangential_features = model_data.get('tangential_features', [])
            self.additional_features = model_data.get('additional_features', [])
            
            logging.info(f"[Classifier] XGBoost model loaded successfully from: {model_path}")
            return True
        
        except Exception as e:
            logging.error(f"[Classifier] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.scaler = None
            self.label_encoder = None
            self.feature_names = None
            return False
        
    def _load_features(self, df: pd.DataFrame):
        """
        Load and validate features directly from features dataframe
        """
        self.features_df = df

        # Validate required columns
        required_columns = ['peak_id', 'timestamp', 'amplitude', 'min_amplitude', 
                          'mean_frequency', 'median_frequency']
        missing_columns = [col for col in required_columns if col not in self.features_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in features CSV: {missing_columns}")
        
        logging.info(f"[Classifier] Loaded {len(self.features_df)} peak features")
    
    def _handle_nan_values(self, X: np.ndarray) -> np.ndarray:
        """
        Handle NaN values in feature data using the same strategy as training
        """
        if self.feature_names is None:
            raise ValueError("Model not loaded. Feature names unknown.")
        
        X_processed = X.copy()
        
        for i, feature_name in enumerate(self.feature_names):
            if np.isnan(X_processed[:, i]).any():
                if 'frequency' in feature_name.lower():
                    # For frequency features, use median of non-NaN values
                    median_val = np.nanmedian(X_processed[:, i])
                    X_processed[np.isnan(X_processed[:, i]), i] = median_val if not np.isnan(median_val) else 0
                elif 'amplitude' in feature_name.lower():
                    # For amplitude features, use 0
                    X_processed[np.isnan(X_processed[:, i]), i] = 0
                elif 'tangential' in feature_name.lower():
                    # For tangential acceleration features, use 0
                    X_processed[np.isnan(X_processed[:, i]), i] = 0
                elif 'rms' in feature_name.lower() or 'energy' in feature_name.lower():
                    # For RMS/energy features, use 0
                    X_processed[np.isnan(X_processed[:, i]), i] = 0
                elif 'spectral' in feature_name.lower() or 'centroid' in feature_name.lower():
                    # For spectral features, use median
                    median_val = np.nanmedian(X_processed[:, i])
                    X_processed[np.isnan(X_processed[:, i]), i] = median_val if not np.isnan(median_val) else 0
                else:
                    # Default: use median for unknown features
                    median_val = np.nanmedian(X_processed[:, i])
                    X_processed[np.isnan(X_processed[:, i]), i] = median_val if not np.isnan(median_val) else 0
        
        return X_processed
    
    def _validate_and_prepare_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate and prepare input data for prediction
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to numpy array if needed
        X = np.array(X)
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Ensure X has the correct number of features
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Input has {X.shape[1]} features, but model expects {len(self.feature_names)} features: {self.feature_names}"
            )
        
        # Handle NaN values
        X_processed = self._handle_nan_values(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_processed)
        
        return X_scaled
        
    def _prepare_features_for_classification(self) -> np.ndarray:
        """
        Prepare feature matrix from DataFrame for classification
        """
        if self.features_df is None:
            raise ValueError("Features not loaded. Call load_features() first.")
        
        if self.feature_names is None:
            raise ValueError("Model not loaded. Cannot prepare features without knowing expected feature names.")
        
        # Map CSV column names to model feature names
        column_mapping = {
            'amplitude': 'Peak_Amplitude',
            'min_amplitude': 'Min_Peak_Amplitude',
            'mean_frequency': 'Mean_Frequency',
            'median_frequency': 'Median_Frequency',
            'tangential_acceleration': 'tangential_acc',
        }
        
        # Reverse mapping: model feature name -> CSV column name
        reverse_mapping = {v: k for k, v in column_mapping.items()}
        
        # Extract feature columns based on what the model expects
        feature_vectors = []
        
        for _, row in self.features_df.iterrows():
            feature_vector = []
            
            for feature_name in self.feature_names:
                # Try direct match first
                if feature_name in self.features_df.columns:
                    feature_vector.append(row[feature_name])
                # Try reverse mapping
                elif feature_name in reverse_mapping:
                    csv_col = reverse_mapping[feature_name]
                    if csv_col in self.features_df.columns:
                        feature_vector.append(row[csv_col])
                    else:
                        # Feature missing, will be handled by NaN handling
                        feature_vector.append(np.nan)
                else:
                    # Feature not found, will be handled by NaN handling
                    feature_vector.append(np.nan)
            
            feature_vectors.append(feature_vector)
        
        return np.array(feature_vectors)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for new data
        """
        X_scaled = self._validate_and_prepare_input(X)
        
        # Predict
        predictions_encoded = self.model.predict(X_scaled)
        
        # Convert back to original labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for new data.
        """
        X_scaled = self._validate_and_prepare_input(X)
    
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def _classify_peaks(self) -> List[Dict]:
        """
        Classify detected peaks using the XGBoost model
        """
        if self.model is None:
            logging.warning("[Classifier] No XGBoost model available for classification")
            return []
        
        if self.features_df is None or len(self.features_df) == 0:
            logging.warning("[Classifier] No features available for classification")
            return []
        
        # Prepare features for classification
        feature_matrix = self._prepare_features_for_classification()
        
        # Classify peaks
        try:
            predictions = self._predict(feature_matrix)
            probabilities = self._predict_proba(feature_matrix)
            
            # Create classification results
            self.classification_results = []
            for i, (_, row) in enumerate(self.features_df.iterrows()):
                result = {
                    'peak_id': int(row['peak_id']),
                    'timestamp': float(row['timestamp']),
                    'amplitude': float(row['amplitude']),
                    'min_amplitude': float(row['min_amplitude']),
                    'mean_frequency': float(row['mean_frequency']),
                    'median_frequency': float(row['median_frequency']),
                    'predicted_class': predictions[i],
                    'probabilities': probabilities[i],
                    'confidence': float(np.max(probabilities[i])),
                    'class_names': self.label_encoder.classes_
                }
                
                # Add channel if available
                if 'channel' in row:
                    result['channel'] = row['channel']
                
                # Add tangential acceleration if available
                if 'tangential_acceleration' in row:
                    result['tangential_acceleration'] = float(row['tangential_acceleration'])
                
                self.classification_results.append(result)
            
            logging.info(f"[Classifier] Classified {len(self.classification_results)} peaks")
            return self.classification_results
            
        except Exception as e:
            logging.error(f"[Classifier] Error during classification: {e}")
            import traceback
            traceback.print_exc()
            return []
        
    def _save_classification_results_csv(self, output_path: Path, curr_timestamp: int):
        """
        Save classification results to CSV
        """
        if not self.classification_results:
            logging.warning("[Classifier] No classification results to save")
            return
        
        output_file = output_path / f"{curr_timestamp}peak_classification.csv"
        
        # Prepare data for CSV
        data = []
        for result in self.classification_results:
            row = {
                'peak_id': result['peak_id'],
                'timestamp': result['timestamp'],
                'amplitude': result['amplitude'],
                'min_amplitude': result['min_amplitude'],
                'mean_frequency': result['mean_frequency'],
                'median_frequency': result['median_frequency'],
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence']
            }
            
            # Add channel if available
            if 'channel' in result:
                row['channel'] = result['channel']
            
            # Add probability columns
            for i, class_name in enumerate(result['class_names']):
                row[f'prob_{class_name}'] = result['probabilities'][i]
            
            data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

    def _save_classification_results_txt(self, output_path: Path):

        """
        Save classification results to TXT file with specified structure
        """

        output_file = output_path / "peak_classification.txt"

        # Extract vectors from all results
        amplitude_vector = [result['amplitude'] for result in self.classification_results]
        min_amplitude_vector = [result['min_amplitude'] for result in self.classification_results]
        mean_frequency_vector = [result['mean_frequency'] for result in self.classification_results]
        median_frequency_vector = [result['median_frequency'] for result in self.classification_results]

        # Write header
        with open(output_file, 'w') as f:
            f.write("sujet;amplitude;min_amplitude;mean_frequency;median_frequency\n")

            # Write data as row vectors
            sujet = 0  # Always 0 as specified
            
            # Format vectors as strings with proper precision
            amplitude_str = ",".join([f"{val:.4f}" for val in amplitude_vector])
            min_amplitude_str = ",".join([f"{val:.4f}" for val in min_amplitude_vector])
            mean_frequency_str = ",".join([f"{val:.2f}" for val in mean_frequency_vector])
            median_frequency_str = ",".join([f"{val:.2f}" for val in median_frequency_vector])
            
            f.write(f"{sujet};{amplitude_str};{min_amplitude_str};{mean_frequency_str};{median_frequency_str}\n")
        
        logging.info(f"[Classifier] Classification results (TXT format) saved to: {output_file}")
    
    def run(self, features_df: pd.DataFrame, output_path: Path, curr_timestamp: int) -> Dict:
        """
        Run the full classification pipeline.
        
        Returns: Dict (dictionary containing classification results and metadata)
        """
        
        # Load features
        self._load_features(features_df)
        
        # Classify peaks
        classifications = self._classify_peaks()
        
        if not classifications:
            logging.warning("[Classifier] No classifications generated")
            return {
                'classifications': [],
                'classifier_available': self.model is not None
            }
        
        # Save results
        self._save_classification_results_csv(output_path, curr_timestamp)
        self._save_classification_results_txt(output_path)
        
        # Print summary
        class_counts = {}
        for result in classifications:
            cls = result['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        for cls, count in sorted(class_counts.items()):
            logging.info(f"[Classifier] {cls}% MVC: {count} peaks\n")
        
        return {
            'classifications': classifications,
            'classifier_available': True,
            'class_counts': class_counts
        }