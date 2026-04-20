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
    def __init__(self, profiler, dorsi_flag, n_reps, session_dir, folder_name, on_complete=None, on_stim_fail=None):
        self.profiler = profiler
        self.dorsi_flag = dorsi_flag
        self.n_reps = n_reps
        self.session_dir = session_dir
        self.folder_name = folder_name
        self.on_complete = on_complete
        self.on_stim_fail = on_stim_fail

        # define gpio variables 
        self.gpio_chip = CONFIG.get("gpio_chip")
        self.gpio_pin = CONFIG.get("gpio_pin")
        self.duration = CONFIG.get("duration")

        # define parameter combinations and stimulation array
        dutycycles = CONFIG.get("dutycycles")
        frequencies = CONFIG.get("frequencies")
        self.total_reps = len(dutycycles) * len(frequencies) * self.n_reps
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
    
    def _run_stimulation(self):
        ''' Runs parameter generation for GPIO square stimulation '''   
        # Define GPIO pin (BCM numbering)
        GPIO_CHIP = self.gpio_chip

        # Open GPIO chip and request the line
        chip = gpiod.Chip(GPIO_CHIP)
        gpio_pin = self.gpio_pin
        duration = self.duration
        
        # Define frequency (Hz) and duty cycle (0-1)
        dutycycle, frequency = self.param_combos[self.curr_reps]
        self.selected_params = [dutycycle, frequency]
        
        total_cycles = int(duration*frequency)
        period = 1/frequency
        on_time = period * (dutycycle)
        off_time = period - on_time
        #print(f"[Square]: {chip}")
    
        self.dorsi_flag.wait() # wait until event marker flag is raised to signal start of dorsiflexion

        self.dorsi_flag.clear() # clear the flag as soon as dorsiflexion begins
        logging.info(f"[Calibrator] Stimulating at frequency: {frequency}, dutycycle: {dutycycle}")
        
        line_request = gpiod.request_lines(
            GPIO_CHIP,
            consumer="pwm_generator",
            config={
                gpio_pin: gpiod.LineSettings(
                    direction=Direction.OUTPUT,
                    output_value=Value.INACTIVE
                )
            }
        )
        line_request.set_value(gpio_pin, Value.ACTIVE)
        
        try:
            for cycle in range(total_cycles):
                # Set pin HIGH
                if on_time > 0:  # Only set high if duty cycle > 0
                    line_request.set_value(gpio_pin, Value.ACTIVE)
                    sleep(on_time)
                
                # Set pin LOW
                if off_time > 0:  # Only set low if duty cycle < 100
                    line_request.set_value(gpio_pin, Value.INACTIVE)
                    sleep(off_time)
        
        finally:
            line_request.set_value(gpio_pin, Value.INACTIVE) # Ensure pin is set LOW at the end
            line_request.release() # Release the GPIO line
    
    def _run_calibration(self, file_path, curr_trial):
        ''' 
        Runs one calibration rep once signal processing complete (feature extraction)
        Triggered by WatchDogCalb as soon as features are written to features.txt -> sets up GPIO 
        Waits until dorsiflexion period begins to stimulate 
        '''  
        self.dorsi_flag.clear() # reset dorsiflexion flag to trigger stimulation
        self.file_path = file_path
        
        # check if calibration is completed (all reps complete and data saved to CSV)
        if self.calb_complete:
            if self.on_complete:
                logging.info("[Calibrator] Triggering auto-stop from main.py...")
                self.on_complete()
                return
        
        # log data from last stimulation
        if (self.selected_params is not None) and self.stim_success: 
            self._log_response() # TO:DO
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
        try:
            self._run_stimulation()
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
        self.profiler.log_metric(curr_trial, "stim", duration)

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