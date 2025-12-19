import os
import time
import threading
import asyncio
import json
import csv
import numpy as np
import pandas as pd
from collections import deque

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
from pylsl import StreamInfo, StreamOutlet

from connection_manager import manager, logging
from emg_feature_extractor import FeatureExtractor
from emg_peak_classifier import PeakClassifier

# ----- Synthetic EMG Recorder: CSV Storing & Analysis Files ---- #
class SyntheticRecorder:
    def __init__(self, num_emg_channels=4, num_accel_channels=3):
        self.recording = False
        self.csv_file = None
        self.csv_writer = None
        self.filename = None
        self.session_dir = None
        self.analyses_dir = None
        self.classification_dir = None
        self._index = 0
        
        self._sample_rate = 250
        self._buffer_seconds = 120
        self._buffer_len = self._sample_rate * self._buffer_seconds # max data points per session=24000 (~15 trials MAX per session) 
        self._buffer = deque(maxlen=self._buffer_len)
        self._buffer_header = None

        self.emg_channel_count = num_emg_channels # auto: all 4
        self.accel_channel_count = num_accel_channels # auto: all 3
        
        self.marker_interval = 6.0 # event every 6 seconds
        self.next_event_time = self.marker_interval // 2 # first marker occurs at 3 seconds
        self.min_time = 0.0 # minimum timestamp pointer for analysis DataFrame
        
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

            if analysis_data.empty:
                return None

            # Save df as csv
            filename = f"{max_time:.1f}.csv"
            output_path = os.path.join(self.analyses_dir, filename)
            analysis_data.to_csv(output_path, index=False)

            self.min_time = self._event_times[trial - 1] # Advance minimum timestamp pointer

            return analysis_data

        except Exception as e:
            print(f"[Recorder] Failed to create temp analysis file: {e}")
            return None
    
    def start_recording(self):
        ''' Initialize main CSV file '''
        
        if self.recording:
            return
        
        # Create session recording folder
        timestamp = time.strftime("%Y-%m-%d_%Hh%Mm%S", time.localtime())
        base_dir = os.getcwd() # create recordings folder in current directory
        self.session_dir = os.path.join(base_dir, "synthetic-emg-recordings", timestamp)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Create minute analyses, features, and classification folder within session folder
        self.analyses_dir = os.path.join(self.session_dir, "minute_analyses")
        os.makedirs(self.analyses_dir, exist_ok=True)
        self.classification_dir = os.path.join(self.session_dir, "classifications")
        os.makedirs(self.classification_dir, exist_ok=True)

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

# ----- EMG Logic: Initialize board, thread and stream data ----- #
ganglion_instance = None

class SyntheticGanglionData:
    def __init__(self, num_trials=10, mac_address=None, channel_list=None, sample_rate=250, buffer_seconds=2):
        '''Initialize Ganglion board system'''
        
        self.serial_port = "COM4"
        self.mac_address = mac_address
        self._selected_channels = channel_list or [0] # default EMG channel 1 + accel channels
        self._sample_rate = sample_rate
        self._buffer_seconds = buffer_seconds
        self._buffer_len = sample_rate * buffer_seconds
        self._num_points = 20 # number of data points processed at once
        
        self._stop_event = threading.Event() # create threading flag for EMG stream
        self._emg_thread = None
        self._session_start_time = None
        
        self._num_trials = int(num_trials) # trials per session
        self.next_trial_block = int(num_trials)
        self._total_events = 0
        self.marker_broadcast_counter = 0

        self.recorder = SyntheticRecorder()
        self.feature_extractor = FeatureExtractor(sampling_rate=250, height_percentile=98, min_distance=3.0)
        self.peak_classifier = PeakClassifier(model_path="./models/xgboost_model.pkl")
        
        self._outlets = {} # LSL StreamOutlets for each channel
        self._accel_outlet = None # LSL StreamOutlets for accel channel

        self.board_shim = None
        self.board_id = BoardIds.SYNTHETIC_BOARD

    
    def _process_data(self, loop):
        # Read and remove current data chunk
        data = self.board_shim.get_board_data() 
        if data.size == 0: return False

        timestamps = data[self.timestamp_channel] # get raw timestamps

        # Set session start time and calculate relative timestamps
        if self._session_start_time is None:
            self._session_start_time = timestamps[0]
            logging.info(f"Recording reference time set: {self._session_start_time}")
        start_ref = self._session_start_time
        relative_timestamps = timestamps - start_ref
        
        # Transpose and write raw data to CSV files
        for i in range(len(timestamps)):
            curr_emg_ch = data[self.emg_channels, i]
            curr_accel_ch = data[self.accel_channels, i]
            curr_timestamp = relative_timestamps[i] 

            has_event = self.recorder.record_data_point(curr_timestamp, curr_emg_ch, curr_accel_ch)
            
            if has_event: 
                self._total_events += 1
                logging.info(f"Recorded {self._total_events} events")

            # Perform analysis on selected trials
            curr_trials = self._total_events - 1
            if curr_trials == self.next_trial_block:
                logging.info(f"Creating analysis file for {curr_trials} trials, from {self.recorder.min_time} to {curr_timestamp} seconds")
                analysis_df = self.recorder.create_analysis_file(curr_timestamp, curr_trials)
                
                # Signal processing of analysis file
                if analysis_df is None:
                    logging.error("Skipping feature extraction because analysis_df is None.")
                else:
                    # Peak detection
                    features_df = self.feature_extractor.run(analysis_df=analysis_df, channels=['ch1 (µV)'])
                    
                    # Peak Classification
                    class_filename = f"{int(curr_timestamp)}_classification.csv"
                    class_filepath = os.path.join(self.recorder.classification_dir, class_filename)
                    classification_results = self.peak_classifier.run(
                        features_df=features_df, output_path=class_filepath)
                
                # Advance to next trial block
                while curr_trials == self.next_trial_block:
                    self.next_trial_block += self._num_trials
                
        # Broadcast event markers
        if self.recorder.event_times_buffer: 
            new_event_times = self.recorder.event_times_buffer[:]
            self.recorder.event_times_buffer.clear() # Clear event times after capturing

            event_data = json.dumps({
                "type": "event_times",
                "timestamps": new_event_times
            })
            asyncio.run_coroutine_threadsafe(manager.broadcast(event_data), loop)
        
        # Send marker interval countdown to frontend
        if self.recorder.marker_interval > 0.0:
            self.marker_broadcast_counter += len(relative_timestamps)
            
            # Only broadcast remaining time every 40 samples (~200ms at 200Hz)
            if self.marker_broadcast_counter >= self._num_points:
                self.marker_broadcast_counter = 0 
                
                next_event_time = self.recorder.next_event_time
                curr_timestamp = relative_timestamps[-1] 
                time_remaining = max(0.0, (next_event_time - curr_timestamp)) 
                
                marker_data = json.dumps({
                    "type": "marker_countdown", 
                    "time_remaining": float(time_remaining)
                })
                asyncio.run_coroutine_threadsafe(manager.broadcast(marker_data), loop)

        # Accel channel processing
        accel_data = data[self.accel_channels]

        self._accel_outlet.push_chunk(accel_data.T.tolist()) # push to LSL
        json_accel = json.dumps({
            "type": "accel_data",
            "timestamp": relative_timestamps.tolist(), 
            "value": accel_data.tolist()
        })
        asyncio.run_coroutine_threadsafe(manager.broadcast(json_accel), loop)
        
        # EMG channel processing
        for ch in self._selected_channels:
            
            if ch < len(self.emg_channels):
                emg_channel = self.emg_channels[ch]
                
                if emg_channel < data.shape[0]:
                    emg_data = data[emg_channel] 
                    
                    # Apply 50Hz low pass filter to remove high freq noise
                    DataFilter.perform_lowpass(emg_data, self.actual_sample_rate, 50.0, 4, 
                                            FilterTypes.BUTTERWORTH.value, 0)

                    # Store in buffers
                    self._buffers[ch].extend(emg_data.tolist())
                    self._timestamps[ch].extend(relative_timestamps.tolist())

                    # Push to LSL
                    if emg_channel in self._outlets:
                        chunk = [[float(val)] for val in emg_data]
                        self._outlets[emg_channel].push_chunk(chunk)

                    # Push to Frontend
                    json_data = json.dumps({
                        "type": "emg_data",
                        "channel_index": ch,
                        "timestamp": relative_timestamps.tolist(), 
                        "value": emg_data.tolist()})
                    # safely place async task onto event loop's queue
                    asyncio.run_coroutine_threadsafe(manager.broadcast(json_data), loop)
        
        return True
    
    def _stream_emg_thread(self, loop: asyncio.AbstractEventLoop):
        '''
        Connect to Ganglion board and initialize EMG stream
        Background thread to acquire data, stream to LSL and display on frontend
        '''
        BoardShim.enable_dev_board_logger()
        
        # Ganglion set-up and initialization

        # For synthetic data
        params = BrainFlowInputParams()
        self.board_shim = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)

        self.board_shim.prepare_session()
        self.board_shim.config_board("n") # Turn on accelerometer
        self.board_shim.start_stream()
        logging.info("Streaming EMG and Accel data...")

        self.emg_channels = BoardShim.get_exg_channels(self.board_id)
        self.accel_channels = BoardShim.get_accel_channels(self.board_id)

        self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
        self.actual_sample_rate = BoardShim.get_sampling_rate(self.board_id)
        self._all_channels = self.emg_channels + self.accel_channels
        
        self._buffers = {ch: deque(maxlen=self._buffer_len) for ch in self._selected_channels}
        self._timestamps = {ch: deque(maxlen=self._buffer_len) for ch in self._selected_channels}

        # LSL initialization for each channel
        for ch in self._selected_channels:
                info = StreamInfo(name=f"EMG_Channel_{ch+1}", type="EMG", channel_count=1,
                    nominal_srate=self.actual_sample_rate, channel_format='float32',
                    source_id=f"ganglion_ch_{ch}"
                )
                self._outlets[ch] = StreamOutlet(info)

        # LSL initialization for accelerometer (handles multiple channels at once)
        accel_info = StreamInfo(name="Ganglion_Accel", type="MOCAP", channel_count=3,
                                nominal_srate=self.actual_sample_rate, channel_format='float32',
                                source_id="ganglion_accel")
        self._accel_outlet = StreamOutlet(accel_info)
        
        try:
            # For real data

            if self.recorder:
                self.recorder.start_recording()
            
            while not self._stop_event.is_set(): # ensure EMG threading flag is down
                
                # Wait for data to accumulate
                if self.board_shim.get_board_data_count() < self._num_points:
                    time.sleep(0.01)
                    continue
                
                # Processing: Filtering, Buffering, Writing to CSV, Pushing to LSL & Frontend
                if self.recorder is not None and self.recorder.recording:
                    processed = self._process_data(loop)
                    if not processed:
                        continue
                else: 
                    logging.warning("Waiting for recorder to be initialized...")

        except Exception as e:
            logging.error(f"Error in EMG thread: {e}", exc_info=True)
        
        finally:
            logging.info("Stopping EMG stream...")
            if self.recorder and self.recorder.recording:
                self.recorder.stop_recording()
            
            if self.board_shim and self.board_shim.is_prepared():
                try:
                    self.board_shim.stop_stream()
                    self.board_shim.release_session()
                except Exception as e:
                    logging.warning(f"Error during board release: {e}")
            logging.info("Session released.")
    
    def start(self, loop):
        if getattr(self, "_emg_thread", None) is not None and self._emg_thread.is_alive():
            logging.warning("Attempted to start EMG thread but one is already running.")
            return
        
        self._stop_event.clear()
        self._emg_thread = threading.Thread(target=self._stream_emg_thread, args=(loop,))
        self._emg_thread.start()

    def stop(self):
        self._stop_event.set()
        if hasattr(self, '_emg_thread') and self._emg_thread.is_alive():
            self._emg_thread.join()