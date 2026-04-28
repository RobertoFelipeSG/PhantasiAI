"""
Offline EMG analysis module.

This module contains components for analyzing recorded EMG data,
including peak detection, feature extraction, and classification.
"""

from .peak_detector import PeakDetector
from .peak_classifier import PeakClassifier
#from .batch_analyzer import BatchAnalyzer
from .xgboost_classifier import XGBoostClassifier

__all__ = ['PeakDetector', 'PeakClassifier', 'BatchAnalyzer', 'XGBoostClassifier']
