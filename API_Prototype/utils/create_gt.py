import pandas as pd
import numpy as np
from pathlib import Path

class GTGenerator:
    def __init__(self, mode="individual"):
        self.mode = mode
        self.generated = False
        
base_path = Path(__file__).parent
calibration_path = Path(__file__).parent.parent
calibrations_dir = base_path / "calibrate"
gt_path = base_path / "ground_truth.csv"
        
def generate_gt():

    try:
        # prepare calibration data
        file = base_path / "mock_calb.csv"
        calb_df = pd.read_csv(file)
        raw_df = pd.concat(calb_df, ignore_index=True)
        
        gt_df = raw_df.groupby(['dutycycle', 'frequency'])['max_amplitude'].apply(lambda x: np.sqrt(np.mean(x**2))).reset_index()

        # save as CSV
        gt_df.to_csv(gt_path, index=False)
        print(f"[GTGenerator] Ground truth table generated")
    
    except Exception as e:
        print(f"[GTGenerator] Error creating ground truth table: {e}")
    
if __name__ == "__main__":
    generate_gt()