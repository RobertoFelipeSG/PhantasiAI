#!/usr/bin/env python3
"""
Launcher script for XGBoost EMG analysis.
This script can be run directly from the emg directory.
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the XGBoost analysis tool
from Previous_Prototypes.Prototype.emg.tools.xgboost_analysis_tool import run_xgboost_classification

if __name__ == "__main__":
    run_xgboost_classification()
