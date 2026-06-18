import os
import sys
import gpiod
import numpy as np
import pandas as pd
import time as tm
from time import sleep, time
from pathlib import Path
from gpiod.line import Direction, Value

from config.connection_manager import logging
from config.config_manager import load_config

CONFIG = load_config()

class Calibrator:
    def __init__(self, calibrate_voltage, stimulator, profiler, stim_flag, stim_state, n_reps, session_dir, folder_name, on_complete=None, on_stim_fail=None):
        self.calibrate_voltage = calibrate_voltage # boolean to determine type of calibration 
        self.stimulator = stimulator
        self.profiler = profiler
        self.stim_flag = stim_flag
        self.stim_state = stim_state
        self.n_reps = n_reps
        self.session_dir = session_dir
        self.folder_name = folder_name
        self.on_complete = on_complete
        self.on_stim_fail = on_stim_fail

        logging.info(f"[Calibrator] Initialized for {n_reps} reps per parameter combination")

        # define parameter combinations and stimulation array
        if self.calibrate_voltage:
            self.no_stim_reps = 0
        else:
            self.no_stim_reps = CONFIG.get("no_stim_reps")
        self.parameter_names = CONFIG.get("parameters")
        dutycycles = CONFIG.get("dutycycles")
        frequencies = CONFIG.get("frequencies")
        if self.calibrate_voltage:
            self.total_reps = CONFIG.get("n_calb_combos") * self.n_reps
        else:
            self.total_reps = len(dutycycles) * len(frequencies) * self.n_reps + self.no_stim_reps
        self.param_combos = [[dc, freq] for dc in dutycycles for freq in frequencies for _ in range(n_reps)]
        self.curr_reps = 0
        self.selected_params = None
        self.stim_success = False
        self.calb_complete = False
        
        # initialize calibration data log and CSV files
        self.calb_log = []
        self.data_saved = False
        self.calb_data = os.path.join(self.session_dir, "calibration.csv") # CSV in session data folder
        
        base_path = base_path = Path(__file__).parent
        self.calb_dir = os.path.join(str(base_path), "calibrations")
        os.makedirs(self.calb_dir, exist_ok=True)
        timestamp = tm.strftime("%Y-%m-%d_%Hh%Mm%S", tm.localtime())
        if self.folder_name is None:
            folder_name = timestamp
        else:
            folder_name = f"{self.folder_name}_{timestamp}"
        self.gen_calb_data = os.path.join(self.calb_dir, f"{folder_name}.csv") # CSV in general calibrations data folder

    def _log_response(self):
        ''' 
        Gets the selected feature values from features.txt of the previous dorsiflexion
        and logs the data + the respective stimulation parameters

        Current Setup:
        - features = max amplitude
        - n trials = 1
        '''
        data = pd.read_csv(self.file_path, sep=';') # Rows split by semicolon ;

        # validate features.txt
        if data.empty:
            logging.error("[Calibrator] File is empty or no subjects found")
            raise ValueError("File is empty or no subjects found")
        
        resp_features = data['max_amplitude']
        feature_val = resp_features.iloc[0] # all values are stored in the first row

        # add to data log
        rep_data = {
            'rep': self.curr_reps + 1,
            'dutycycle': float(self.selected_params[0]),
            'frequency': float(self.selected_params[1]),
            'max_amplitude': float(feature_val)
            }

        self.calb_log.append(rep_data)
    
    def _run_calibration(self, file_path, curr_trial):
        ''' 
        Runs one calibration rep once signal processing complete (feature extraction)
        Triggered by WatchDogCalb as soon as features are written to features.txt -> sets up GPIO 
        Waits until dorsiflexion period begins to stimulate 
        '''  
        self.stim_flag.clear() # reset dorsiflexion flag to trigger stimulation
        self.file_path = file_path
        
        # check if calibration is completed (all reps complete and data saved to CSV)
        if self.calb_complete:
            if self.on_complete:
                logging.info("[Calibrator] Triggering auto-stop from main.py...")
                self.on_complete()
                return
        
        # log data from last stimulation
        if (self.selected_params is not None) and self.stim_success: 
            self._log_response()
            self.curr_reps += 1
            self.stim_success = False # reset success state
        
        # save to CSV if all reps for all stimulation combinations completed
        if self.curr_reps >= self.total_reps:
            df = pd.DataFrame(self.calb_log)
            df.to_csv(self.calb_data, index=False)
            df.to_csv(self.gen_calb_data, index=False)       
            self.data_saved = True
            
            self.calb_complete = True # update state
            
            return 
        
        # run next stimulation trial
        if self.calibrate_voltage: # keep selected frequency and dutycycle when we are calibrating the voltage
            voltage_param_combos = [
                self.param_combos[0],     # first duty, first freq
                [self.param_combos[0][0], self.param_combos[-1][1]],  # first duty, last freq
                [self.param_combos[-1][0], self.param_combos[0][1]],  # last duty, first freq
                self.param_combos[-1]     # last duty, last freq
             ]
                
            combo_index = self.curr_reps // self.n_reps
            self.selected_params = voltage_param_combos[combo_index]
            best_params = {
                self.parameter_names[0]: float(self.selected_params[0]),
                self.parameter_names[1]: float(self.selected_params[1])
            }
        elif self.curr_reps < self.no_stim_reps: # no stim reps to test baseline MVC
            self.selected_params = [0.0, 0.0]
        else: # stimulation reps
            self.selected_params = self.param_combos[self.curr_reps - self.no_stim_reps]
            best_params = {
                self.parameter_names[0]: float(self.selected_params[0]),
                self.parameter_names[1]: float(self.selected_params[1])
            }
        
        self.stim_flag.wait() # wait until event marker flag is raised to signal start of dorsiflexion

        ready_duration = self.stim_state["ready_duration"] # get duration of prep period to pass to stimulator 

        self.stim_flag.clear() # clear the flag as soon as dorsiflexion begins
        try:
            if self.selected_params != [0.0, 0.0]:
                self.stimulator.run(best_params, ready_duration, curr_trial)
            self.stim_success = True
            self.profiler.mark_process_complete(curr_trial)
        except (OSError, FileNotFoundError) as e:
            logging.error(f"[Calibrator] Hardware Stim Error: Cannot access GPIO chip. {e}")
            if self.on_stim_fail: self.on_stim_fail()
        except KeyError as e:
            logging.error(f"[Calibrator] Parameter Stim Error: Missing key in best_params: {e}")
            if self.on_stim_fail: self.on_stim_fail()
        except Exception as e:
            logging.error(f"[Calibrator] Unexpected error during stimulation: {type(e).__name__}: {e}")
            if self.on_stim_fail: self.on_stim_fail()

    def run(self, file_path, curr_trial):
        start_time = time()
        
        self._run_calibration(file_path, curr_trial)

        duration = time() - start_time
        self.profiler.log_metric(curr_trial, "calib", duration)

    def handle_stop(self):
        ''' 
        Backup to save optimization data for current optimization run 
        (in case of sudden stop) 
        '''
        # add to CSV with stimulation data
        if not self.data_saved:
            df = pd.DataFrame(self.calb_log)
            backup_file = os.path.join(self.session_dir, "calibration_backup.csv") 
            df.to_csv(backup_file, index=False)
            df.to_csv(self.gen_calb_data, index=False)
            self.data_saved = True