"""
Real-time EMG Peak Analyzer
===========================

This module provides real-time peak analysis capabilities for live EMG data.
It wraps the offline analysis components to provide immediate feedback
during live recording sessions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
from typing import Dict, List, Optional, Tuple

from ..offline_analysis.peak_classifier import PeakClassifier


class RealTimePeakAnalyzer:
    """
    Real-time EMG peak analyzer for live data streams.
    
    This class provides methods to analyze EMG data in real-time,
    detecting peaks and providing immediate classification feedback.
    """
    
    def __init__(self, sampling_rate: int = 220, 
                 height_percentile: float = 95,
                 min_distance: float = 2.0,
                 model_path: Optional[str] = None):
        """
        Initialize the real-time peak analyzer.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate of the EMG data (Hz)
        height_percentile : float
            Threshold for peak detection (percentile of signal amplitude)
        min_distance : float
            Minimum time (in seconds) between peaks
        model_path : str, optional
            Path to the pre-trained XGBoost model
        """
        self.sampling_rate = sampling_rate
        self.height_percentile = height_percentile
        self.min_distance = min_distance
        self.model_path = model_path
        
        # Initialize the offline classifier for analysis
        self.classifier = None
        if model_path:
            self._load_classifier()
    
    def _load_classifier(self):
        """Load the peak classifier with the specified model."""
        try:
            # Create a temporary CSV file to initialize the classifier
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("timestamp,ch1 (µV),event\n0.0,0.0,0\n")
                temp_csv = f.name
            
            self.classifier = PeakClassifier(
                csv_path=temp_csv,
                model_path=self.model_path,
                sampling_rate=self.sampling_rate,
                height_percentile=self.height_percentile,
                min_distance=self.min_distance
            )
            
            # Clean up temporary file
            os.unlink(temp_csv)
            print("Real-time peak analyzer initialized successfully")
            
        except Exception as e:
            print(f"Error initializing real-time peak analyzer: {e}")
            self.classifier = None
    
    def analyze_buffer(self, timestamps: np.ndarray, 
                      emg_data: np.ndarray) -> Dict:
        """
        Analyze a buffer of EMG data for peaks.
        
        Parameters:
        -----------
        timestamps : np.ndarray
            Array of timestamps corresponding to the EMG data
        emg_data : np.ndarray
            Array of EMG values (can be multi-channel)
            
        Returns:
        --------
        dict
            Analysis results containing peak information and classifications
        """
        if self.classifier is None:
            return {"error": "Classifier not initialized"}
        
        try:
            # Create a temporary CSV file with the buffer data
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                # Write header
                if emg_data.ndim == 1:
                    f.write("timestamp,ch1 (µV),event\n")
                else:
                    channels = [f"ch{i+1} (µV)" for i in range(emg_data.shape[1])]
                    f.write(f"timestamp,{','.join(channels)},event\n")
                
                # Write data
                for i, t in enumerate(timestamps):
                    if emg_data.ndim == 1:
                        f.write(f"{t:.8f},{emg_data[i]:.5f},0\n")
                    else:
                        values = [f"{v:.5f}" for v in emg_data[i]]
                        f.write(f"{t:.8f},{','.join(values)},0\n")
                
                temp_csv = f.name
            
            # Update the classifier's CSV path and run analysis
            self.classifier.csv_path = temp_csv
            results = self.classifier.run(show_plots=False, save_results=False, classify_peaks=True)
            
            # Clean up temporary file
            os.unlink(temp_csv)
            
            return results
            
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
    
    def detect_peaks_simple(self, emg_data: np.ndarray, 
                          threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple peak detection without classification.
        
        Parameters:
        -----------
        emg_data : np.ndarray
            Array of EMG values
        threshold : float, optional
            Peak detection threshold. If None, uses height_percentile
            
        Returns:
        --------
        tuple
            (peak_indices, peak_amplitudes)
        """
        if threshold is None:
            threshold = np.percentile(emg_data, self.height_percentile)
        
        # Simple peak detection
        peaks = []
        amplitudes = []
        
        for i in range(1, len(emg_data) - 1):
            if (emg_data[i] > threshold and 
                emg_data[i] > emg_data[i-1] and 
                emg_data[i] > emg_data[i+1]):
                peaks.append(i)
                amplitudes.append(emg_data[i])
        
        return np.array(peaks), np.array(amplitudes)
    
    def get_analysis_summary(self, results: Dict) -> str:
        """
        Generate a human-readable summary of analysis results.
        
        Parameters:
        -----------
        results : dict
            Results from analyze_buffer()
            
        Returns:
        --------
        str
            Formatted summary string
        """
        if "error" in results:
            return f"Analysis Error: {results['error']}"
        
        summary = []
        summary.append(f"Peaks detected: {results.get('num_peaks', 0)}")
        
        if results.get('classifications'):
            class_counts = {}
            for result in results['classifications']:
                cls = result['predicted_class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            summary.append("Classifications:")
            for cls, count in sorted(class_counts.items()):
                summary.append(f"  {cls}% MVC: {count} peaks")
        
        return "\n".join(summary)
