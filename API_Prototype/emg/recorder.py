import os
import csv
import json
import numpy as np
import pandas as pd
import time
from collections import deque
from pathlib import Path

from config.connection_manager import logging
from config.config_manager import load_config
from utils.create_isi import generate_intervals

CONFIG = load_config()

# ----- Real Time EMG Recorder: CSV Storing & Analysis Files ---- #
class RealTimeRecorder:
    def __init__(self, sample_rate, profiler, dorsi_flag, isi_type, base_path, folder_name=None):
        self.recording = False
        self.csv_file = None
        self.csv_writer = None
        
        self.dorsi_flag = dorsi_flag
        self.isi_type = isi_type
        self.profiler = profiler
        self.base_path = base_path
        self.filename = None
        self._index = 0

        # Create session recording folder
        timestamp = time.strftime("%Y-%m-%d_%Hh%Mm%S", time.localtime())
        if folder_name is None:
            folder_name = timestamp
        else:
            folder_name = f"{folder_name}_{timestamp}"
        self.session_dir = os.path.join(str(self.base_path), folder_name)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Create features folder within session folder
        self.features_dir = os.path.join(self.session_dir, "features")
        os.makedirs(self.features_dir, exist_ok=True)
        
        self.min_time = 0.0 # starting timestamp pointer for analysis DataFrame 
        self._sample_rate = sample_rate
        self._buffer_seconds = CONFIG.get("recorder_buffer_seconds")
        self._buffer_len = self._sample_rate * self._buffer_seconds # max data points per session         
        self._buffer = deque(maxlen=self._buffer_len)
        self._buffer_header = None

        self.emg_channel_count = CONFIG.get("num_emg_ch") 
        self.accel_channel_count = CONFIG.get("num_accel_ch") 

        self.event_times_buffer = [] # buffer to store event times for frontend push
        self._event_times = [] # store event times for entire session

        self.trial_times_buffer = [] # buffer to store trial times for frontend push (timestamp for the start of a new trial)
        self._trial_times = [] # store trial times for entire session
        
        # define inter-stimulus interval and stimulus timing
        if self.isi_type == 'dynamic':
            isi_path = Path(__file__).parent.parent / "utils" / "dynamic_intervals.json"
            
            if os.path.exists(isi_path):
                with open(isi_path, "r") as f:
                    self.dynamic_intervals = json.load(f)["intervals"] # read from array of dynamic ISIs; 400 total
            else: 
                logging.warning("[Recorder] Could not load dynamic ISI file")
                self.dynamic_intervals = generate_intervals() 
        
        self.inter_stim_interval = CONFIG.get("static_isi") # initial ISI always static
        self.stim_duration = CONFIG.get("static_stim_duration")
        
        # setup event/trial marker logic
        self.total_events = 0 # stores how many events completed
        self.total_trials = 0 # stores how many trials completed
        self.marker_interval = self.inter_stim_interval + self.stim_duration # marker occurs every (ISI + stimulus) seconds; this value does not change if ISI = static
        self.next_event_time = self.inter_stim_interval # first event marker occurs at the end of first isi (i.e. when first stim presented)
        self.next_trial_time = self.marker_interval # first trial marker occurs at the end of first stimulus 
        
        logging.info(f"[Recorder] ({self.isi_type}) initialized for {self.emg_channel_count} EMG and {self.accel_channel_count} Accel channels") 
    
    def _mark_event(self, timestamp):
        event_flag = 0
        if timestamp >= self.next_event_time:
            logging.info(f"[Recorder] Event detected")
            self.dorsi_flag.set()
            
            event_flag = 1
            self.total_events += 1
            
            self._event_times.append(timestamp)
            self.event_times_buffer.append(timestamp)

            # Add event timestamp and isi to profiler (to mark beginning of dorsiflexion)
            if self.total_trials != 0:
                self.profiler.log_metric(self.total_trials, "event", timestamp)
                self.profiler.log_metric(self.total_trials, "isi", self.inter_stim_interval)
            
            # Advance next event marker time
            if (self.isi_type == 'dynamic') and (self.total_trials < 400): # safety to avoid index error if trials exceed 400
                # redefine marker interval time for dynamic
                self.inter_stim_interval = self.dynamic_intervals[self.total_trials] # first ISI indexed at 0 
                self.marker_interval = self.stim_duration + self.inter_stim_interval 

            self.next_event_time += float(self.marker_interval)
            #logging.info(f"[Recorder] Interval event marked at {timestamp}s. Next marker due at {self.next_event_time:.4f}s")
        
        return event_flag

    def _mark_trial(self, timestamp):
        trial_flag = 0
        if timestamp >= self.next_trial_time:
            trial_flag = 1
            self.total_trials += 1

            self._trial_times.append(timestamp)
            self.trial_times_buffer.append(timestamp)

            # Advance next trial time
            self.next_trial_time += float(self.marker_interval)

        return trial_flag
        
    def record_data_point(self, timestamp, emg_values, accel_values):
        '''
        Records single row of EMG and Accel data in csv file
        Returns if event occured
        '''
        if not self.recording or not self.csv_writer:
            return False

        try:
            # Ensure emg_values and accel_values are a list of floats
            if isinstance(emg_values, (np.ndarray, list)):
                emg_values = [float(v) for v in emg_values]
            else:
                emg_values = [float(emg_values)]

            if isinstance(accel_values, (np.ndarray, list)):
                accel_values = [float(v) for v in accel_values]
            else:
                accel_values = [float(accel_values)]
            
            # Get event flag and trial flag
            event_flag = self._mark_event(timestamp)
            trial_flag = self._mark_trial(timestamp)
            
            # safety filters
            emg_values = emg_values[:self.emg_channel_count]
            accel_values = accel_values[:self.accel_channel_count]

            # Numeric row for buffers
            numeric_row = [timestamp] + emg_values + accel_values + [event_flag] + [trial_flag]
            self._buffer.append(numeric_row)
            
            # Format Row: Timestamp | Ch1 | Ch2 ... | AccelX | ... | EventFlag
            formatted_emg = [f"{v:.2f}" for v in emg_values]
            formatted_accel = [f"{v:.2f}" for v in accel_values]
            row = [f"{timestamp}"] + formatted_emg + formatted_accel + [int(event_flag)] + [int(trial_flag)]
            
            self.csv_writer.writerow(row)

            # Flush periodically 
            if self._index % 10 == 0:
                self.csv_file.flush()
            self._index += 1

            return bool(event_flag), bool(trial_flag)            

        except Exception as e:
            logging.warning(f"[Recorder] Failed to record data point: {e}")
            return False, False

    def create_analysis_file(self, max_time, trial): 
        """
        Create an analysis file for data of a single trial
        Return: DataFrame (df of analysis data) 
        """

        if not self.recording or not self._buffer: return None

        try:
            start_time = time.time()
            
            df = pd.DataFrame(self._buffer, columns=self._buffer_header)
            df['timestamp'] = df['timestamp'].astype(float)

            # Calculate analysis window and get data
            analysis_data = df[(df['timestamp'] >= self.min_time) & (df['timestamp'] <= max_time)]

            last_min = self.min_time
            self.min_time = max_time # Advance minimum timestamp pointer

            if analysis_data.empty:
                logging.error(f"[Recorder] Could not create analysis file")
                return None

            '''# Save df as csv
            filename = f"{max_time:.1f}.csv"
            output_path = os.path.join(self.analyses_dir, filename)
            analysis_data.to_csv(output_path, index=False)
            '''
        
            duration = time.time() - start_time
            self.profiler.log_metric(trial, "file_create", duration)

            return analysis_data

        except Exception as e:
            logging.error(f"[Recorder] Failed to create temp analysis file: {e}")
            return None
    
    def start_recording(self):
        ''' Initialize main CSV file '''
        
        if self.recording:
            return

        self.filename = os.path.join(self.session_dir, "emg_accel.csv")

        try:
            self.csv_file = open(self.filename, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)

            emg_headers = [f'ch{i+1} (µV)' for i in range(self.emg_channel_count)]
            accel_headers = [f'accel_{axis}' for axis in ['x', 'y', 'z']]
            header = ['timestamp'] + emg_headers + accel_headers + ['event'] + ['trial']
            
            self._buffer_header = header
            self.csv_writer.writerow(header)
            self.csv_file.flush()

            self.recording = True

            logging.info(f"\n[Recorder] Recording started: {self.filename}")
        
        except Exception as e:
            logging.error(f"[Recorder] Failed to start recording: {e}")
            self.recording = False
    
    def stop_recording(self):
        if not self.recording:
            return
        
        try:
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
            self.csv_writer = None
            self.recording = False
            self.next_event_time = 0.0 # Reset marker time on stop

        except Exception as e:
            logging.error(f"[Recorder] Failed to stop recording: {e}")