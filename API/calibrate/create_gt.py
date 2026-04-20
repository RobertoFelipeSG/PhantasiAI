import pandas as pd
import numpy as np
import os
from pathlib import Path

from config.connection_manager import logging

class GTGenerator:
    def __init__(self, mode="individual"):
        self.mode = mode
        self.generated = False
        
        self.base_path = Path(__file__).parent
        self.calibrations_dir = self.base_path / "calibrations"
        self.gt_path = self.base_path / "ground_truth.csv"
        
    def _generate_gt(self):
        if not self.calibrations_dir.exists():
            logging.error("[GTGenerator] Calibration directory not found")
            return False

        csv_files = list(self.calibrations_dir.glob("*.csv"))
        if not csv_files:
            logging.error("[GTGenerator] No CSV files found in calibrations folder")
            return False

        if self.mode == "accumulated": # get data from ALL files in calibrations/
            calb_files = csv_files 
        else: # get data from most recent calibration session
            calb_files = [max(csv_files, key=os.path.getmtime)] 
            logging.info(f"[GTGenerator] Using {str(calb_files)} for ground truth")

        try:
            # prepare calibration data
            calb_df = [pd.read_csv(file) for file in calb_files]
            raw_df = pd.concat(calb_df, ignore_index=True)
            
            gt_df = raw_df.groupby(['dutycycle', 'frequency'])['max_amplitude'].apply(lambda x: np.sqrt(np.mean(x**2))).reset_index()

            # save as CSV
            gt_df.to_csv(self.gt_path, index=False)
            logging.info(f"[GTGenerator] Ground truth table generated")
            return True
        
        except Exception as e:
            logging.error(f"[GTGenerator] Error creating ground truth table: {e}")
            return False
        
    def run(self):
        return self._generate_gt()