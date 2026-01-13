import os
import time
import csv
import numpy as np
import pandas as pd
from collections import deque

from config.connection_manager import logging
from config.config_manager import load_config

config = load_config()

# ----- Real Time EMG Recorder: CSV Storing & Analysis Files ---- #
class RealTimeRecorder:
    def __init__(self, sample_rate, base_path):
        self.recording = False
        self.csv_file = None
        self.csv_writer = None
        
        self.base_path = base_path
        self.filename = None
        self._index = 0

        # Create session recording folder
        timestamp = time.strftime("%Y-%m-%d_%Hh%Mm%S", time.localtime())
        self.session_dir = os.path.join(str(self.base_path), "emg-recordings", timestamp)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Create minute analyses and classification folders within session folder
        self.analyses_dir = os.path.join(self.session_dir, "minute_analyses")
        os.makedirs(self.analyses_dir, exist_ok=True)
        self.classification_dir = os.path.join(self.session_dir, "classifications")
        os.makedirs(self.classification_dir, exist_ok=True)
        
        self._sample_rate = sample_rate
        self._buffer_seconds = config.get("recorder_buffer_seconds")
        self._buffer_len = self._sample_rate * self._buffer_seconds # max data points per session         
        self._buffer = deque(maxlen=self._buffer_len)
        self._buffer_header = None

        self.emg_channel_count = config.get("num_emg_ch") 
        self.accel_channel_count = config.get("num_accel_ch") 
        
        self.marker_interval = config.get("marker_interval") # event every x seconds
        self.next_event_time = self.marker_interval // 2 # first marker occurs at half of an interval 
        self.min_time = 0.0 # starting timestamp pointer for analysis DataFrame
        
        self.event_times_buffer = [] # buffer to store event times for frontend push
        self._event_times = [] # store event times for entire session 
        
        logging.info(f"[Recorder] initialized for {self.emg_channel_count} EMG and {self.accel_channel_count} Accel channels")

    def _mark_event(self, timestamp):
        event_flag = 0
        if timestamp >= self.next_event_time:
            event_flag = 1
            self._event_times.append(timestamp)
            self.event_times_buffer.append(timestamp)
            
            # Advance the marker time
            self.next_event_time += float(self.marker_interval)
            logging.info(f"[Recorder] Interval event marked at {timestamp}s. Next marker due at {self.next_event_time:.4f}s")
        
        return event_flag
        
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
            
            # Get event flag
            event_flag = self._mark_event(timestamp)
            
            # safety filters
            emg_values = emg_values[:self.emg_channel_count]
            accel_values = accel_values[:self.accel_channel_count]

            # Numeric row for buffers
            numeric_row = [timestamp] + emg_values + accel_values + [event_flag]
            self._buffer.append(numeric_row)
            
            # Format Row: Timestamp | Ch1 | Ch2 ... | AccelX | ... | EventFlag
            formatted_emg = [f"{v:.2f}" for v in emg_values]
            formatted_accel = [f"{v:.2f}" for v in accel_values]
            row = [f"{timestamp}"] + formatted_emg + formatted_accel + [int(event_flag)]
            
            self.csv_writer.writerow(row)

            # Flush periodically 
            if self._index % 10 == 0:
                self.csv_file.flush()
            self._index += 1

            return bool(event_flag)            

        except Exception as e:
            logging.warning(f"[Recorder] Failed to record data point: {e}")
            return False

    def create_analysis_file(self, max_time, trial): 
        """
        Create an analysis file with X number of trials and saves file as CSV 
        Return: DataFrame (df of analysis data) 
        """

        if not self.recording or not self._buffer: return None

        try:
            df = pd.DataFrame(self._buffer, columns=self._buffer_header)
            df['timestamp'] = df['timestamp'].astype(float)

            # Calculate analysis window and get data
            analysis_data = df[(df['timestamp'] > self.min_time) & (df['timestamp'] < max_time)]

            self.min_time = self._event_times[trial - 1] # Advance minimum timestamp pointer

            if analysis_data.empty:
                return None

            # Save df as csv
            filename = f"{max_time:.1f}.csv"
            output_path = os.path.join(self.analyses_dir, filename)
            analysis_data.to_csv(output_path, index=False)

            return analysis_data

        except Exception as e:
            print(f"[Recorder] Failed to create temp analysis file: {e}")
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
            header = ['timestamp'] + emg_headers + accel_headers + ['event']
            
            self._buffer_header = header
            self.csv_writer.writerow(header)
            self.csv_file.flush()

            self.recording = True

            logging.info(f"[Recorder] Recording started: {self.filename}")
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

            logging.info("[Recorder] Recording stopped")
        except Exception as e:
            logging.error(f"[Recorder] Failed to stop recording: {e}")