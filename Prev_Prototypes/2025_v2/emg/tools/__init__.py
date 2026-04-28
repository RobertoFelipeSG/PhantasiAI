"""
EMG analysis tools module.

This module contains standalone tools and scripts for running
EMG analysis workflows.
"""

from .offline_analysis_tool import run_offline_analysis
from .xgboost_analysis_tool import run_xgboost_classification

__all__ = ['run_offline_analysis', 'run_xgboost_classification']
