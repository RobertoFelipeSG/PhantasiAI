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
    def __init__(self, sample_rate, session_size, profiler, stim_flag, stim_state, isi_type, calibrate_voltage, base_path, folder_name=None):
        self.recording = False
        self.csv_file = None
        self.csv_writer = None
        
        self.stim_flag = stim_flag
        self.stim_state = stim_state
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

        self.ready_times_buffer = [] # buffer to store 'ready' times for frontend push (timestamp for the start of electrical stimulation period)
        self._ready_times = [] # store 'ready' times for entire session
        
        # within session-break setup
        self.session_size = session_size
        self.in_break = False
        self.break_end_time = 0.0
        if calibrate_voltage:
            self._trial_break_interval = 10
            self._break_duration = 10
        else:
            self._trial_break_interval = CONFIG.get("trial_break_interval")
            self._break_duration = CONFIG.get("break_duration")
        self.next_break_block = self._trial_break_interval + 1 # first trial in a session = "mock" trial
        
        # define inter-stimulus interval and stimulus timing
        if self.isi_type == 'dynamic':
            isi_path = Path(__file__).parent.parent / "utils" / "dynamic_intervals.json"
            
            if os.path.exists(isi_path):
                with open(isi_path, "r") as f:
                    self.dynamic_intervals = json.load(f)["intervals"] # read from array of dynamic ISIs; 400 total
                    self.total_intervals = len(self.dynamic_intervals)
                    logging.info(f"[Recorder] Loaded {self.total_intervals} inter-stimulus intervals")
            else: 
                logging.warning("[Recorder] Could not load dynamic ISI file")
                self.dynamic_intervals = generate_intervals() 
        
        # setup event/trial marker logic
        self.total_events = 0 # stores how many events completed (start of a dorsiflexion)
        self.total_trials = 0 # stores how many trials completed (end of a dorsiflexion)

        self.rest_duration = CONFIG.get("static_rest_duration")
        self.ready_duration = CONFIG.get("static_ready_duration")
        self.go_duration = CONFIG.get("static_go_duration")
 
        self.inter_stim_interval = self.rest_duration + self.ready_duration # initial ISI always static (first trial is always 6s)
        self.marker_interval = self.inter_stim_interval + self.go_duration # marker occurs every (ISI + GO) seconds; this value does not change if ISI = static

        self.next_ready_time = self.rest_duration # first "ready" marker happens right after rest period
        self.next_event_time = self.inter_stim_interval # first event marker occurs at the end of first rest+ready period (start of first GO period)
        self.next_trial_time = self.marker_interval # first trial marker occurs at the end of first GO period 
        
        logging.info(f"[Recorder] ({self.isi_type}) initialized for {self.emg_channel_count} EMG and {self.accel_channel_count} Accel channels") 
    
    def _mark_ready(self, timestamp):
        ready_flag = 0
        if timestamp >= self.next_ready_time:
            ready_flag = 1

            self._ready_times.append(timestamp)
            self.ready_times_buffer.append(timestamp)

            if self.total_trials > 0:
                logging.info(f"[Recorder] Ready state detected; Duration = {self.ready_duration}")

                # notify stimulator to begin stimulation based on ready duration of CURRENT trial
                self.stim_state["ready_duration"] = self.ready_duration
                self.stim_flag.set()
            
            # Add ready timestamp (to mark beginning of electrical stimulation)
            if self.total_trials != 0:
                self.profiler.log_metric(self.total_trials, "stim_start", timestamp)
                self.profiler.log_metric(self.total_trials, "isi", self.stim_state["ready_duration"]) 
            
            # Advance next ready time
            self.next_ready_time += float(self.marker_interval)

        return ready_flag 
    
    def _mark_event(self, timestamp):
        event_flag = 0
        if timestamp >= self.next_event_time:
            logging.info(f"[Recorder] Event detected")
            
            event_flag = 1
            self.total_events += 1
            
            self._event_times.append(timestamp)
            self.event_times_buffer.append(timestamp)

            # Advance next event marker time
            if (self.isi_type == 'dynamic') and (self.total_trials < self.total_intervals): # safety to avoid index error if trials exceed dynamic intervals array
                self.ready_duration = self.dynamic_intervals[self.total_trials] # first ISI indexed at 0 
                
                # redefine marker interval time for dynamic
                self.inter_stim_interval = self.rest_duration + self.ready_duration 
                self.marker_interval = self.go_duration + self.inter_stim_interval

            # Add event timestamp to profiler (to mark beginning of dorsiflexion)
            if self.total_trials != 0:
                self.profiler.log_metric(self.total_trials, "dorsi_start", timestamp)

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

            # Add trial timestamp (to mark beginning of trial)
            self.profiler.start_trial(self.total_trials)
            self.profiler.log_metric(self.total_trials, "trial_start", timestamp)
            logging.info(f"[Recorder] Trial {self.total_trials+1} starting...")

            # Advance next trial time
            self.next_trial_time += float(self.marker_interval)

        return trial_flag
        
    def _update_break_state(self, timestamp):
        '''
        Checks if break should start/stop
            Start: trial number has crossed the trial threshold
            Stop: timestamp has crossed the time threshold
        '''
        
        if self.in_break:
            if timestamp >= self.break_end_time:
                self.in_break = False
                logging.info(f"[Recorder] Break complete at {timestamp:.2f}s")
                
                return "break_ended"
            
            return "in_break"
        
        if timestamp >= self.next_trial_time:
            if self.total_trials > 0 and self.total_trials < self.session_size and ((self.total_trials + 1) == self.next_break_block):
                # Initialize break and advance pointers
                self.in_break = True
                self.break_end_time = timestamp + self._break_duration
                self.next_break_block += self._trial_break_interval

                logging.info(f"[Recorder] Initializing break period at {timestamp:.2f}s until {self.break_end_time:.2f}s")

                # Shift internal markers forward by break duration
                self.next_trial_time += float(self._break_duration)
                self.next_ready_time += float(self._break_duration)
                self.next_event_time += float(self._break_duration)

                return "break_started"
            
        return "no_break"
    
    def record_data_point(self, timestamp, emg_values, accel_values):
        '''
        Records single row of EMG and Accel data in csv file
        Returns if event occured
        '''
        if not self.recording or not self.csv_writer:
            return "no_break", False, False

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
            
            # Evaluate break state BEFORE markers (must check before trial starts)
            break_status = self._update_break_state(timestamp)
            
            # Get markers (mark start of ready phase, event (dorsiflexion), and trial)
            if break_status in ["in_break", "break_started"]:
                ready_flag = 0
                event_flag = 0
                trial_flag = 0
                break_flag = 1
            else: # no_break or break_ended
                ready_flag = self._mark_ready(timestamp)
                event_flag = self._mark_event(timestamp)
                trial_flag = self._mark_trial(timestamp)
                break_flag = 0
            
            # safety filters
            emg_values = emg_values[:self.emg_channel_count]
            accel_values = accel_values[:self.accel_channel_count]

            # Numeric row for buffers
            numeric_row = [timestamp] + emg_values + accel_values + [event_flag] + [trial_flag] + [ready_flag] + [break_flag]
            self._buffer.append(numeric_row)
            
            # Format Row: Timestamp | Ch1 | Ch2 ... | AccelX | ... | EventFlag
            formatted_emg = [f"{v:.2f}" for v in emg_values]
            formatted_accel = [f"{v:.2f}" for v in accel_values]
            row = [f"{timestamp}"] + formatted_emg + formatted_accel + [int(event_flag)] + [int(trial_flag)] + [int(ready_flag)] + [int(break_flag)]
            
            self.csv_writer.writerow(row)

            # Flush periodically 
            if self._index % 10 == 0:
                self.csv_file.flush()
            self._index += 1

            return break_status,bool(event_flag), bool(trial_flag)            

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
            header = ['timestamp'] + emg_headers + accel_headers + ['event'] + ['trial'] + ['e_stim'] + ['break']
            
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