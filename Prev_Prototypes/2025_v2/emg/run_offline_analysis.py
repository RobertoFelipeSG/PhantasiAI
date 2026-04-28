#!/usr/bin/env python3
"""
Launcher script for offline EMG analysis.
This script can be run directly from the emg directory.
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the offline analysis tool
from Previous_Prototypes.Prototype.emg.tools.offline_analysis_tool import run_offline_analysis

if __name__ == "__main__":
    run_offline_analysis()
