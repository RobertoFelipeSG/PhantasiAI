import time
import csv
from datetime import datetime
from pathlib import Path
from config.connection_manager import logging

class SessionProfiler: 
    def __init__(self):
        self.trials = {}
        self.current_trial = 0
        self.timings_saved = False

    def start_trial(self, trial_num):
        '''Initialize new trial dictionary for current trial'''
        self.current_trial = trial_num
        self.trials[trial_num] =  {
            "overall_process": None,
            "file_create": None,
            "feat_extract": None,
            "opt_iter": None,
            "stim": None,
            "_trial_end_time": time.time() 
        }

    def log_metric(self, trial_num, metric_name, duration):
        """Saves a specific timing metric to trial dictionary"""
        if trial_num in self.trials:
            self.trials[trial_num][metric_name] = duration

    def mark_process_complete(self, trial_num):
        """
        Store overall time from when a trial ended to when stimulation completed
        (One entire iteration process)
        """
        if trial_num in self.trials:
            start_time = self.trials[trial_num]["_trial_end_time"]
            self.trials[trial_num]["overall_process"] = time.time() - start_time

    def save_as_csv(self, base_path: Path):
        """Converts dict to CSV file at end of the session"""
        if self.timings_saved:
            return # timings have already been saved
        
        if not self.trials:
            return # Nothing to export
            
        filepath = base_path / "timings.csv"
        
        fieldnames = [
            "trial_num",
            "overall_process",
            "file_create",
            "feat_extract",
            "opt_iter",
            "stim",
        ]

        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for t_num, metrics in self.trials.items():
                    row = {"trial_num": t_num}
                    for field in fieldnames[1:]: 
                        row[field] = metrics.get(field, "ERROR") # if no time, process unsuccesful
                    writer.writerow(row)
            self.timings_saved = True
        except OSError as e:
            logging.info(f"[Profiler] Could not save session timing metrics: {e}")