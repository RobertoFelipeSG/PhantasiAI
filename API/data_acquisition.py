import os
import time
import threading
import asyncio
import json
import csv
import numpy as np
from collections import deque
from pathlib import Path

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pylsl import StreamInfo, StreamOutlet

from connection_manager import manager, logging

# ----- Real Time EMG Recorder: CSV Storing & Analysis Files ---- #
class RealTimeRecorder:
    def __init__(self, num_emg_channels=1, num_accel_channels=3):
        self.recording = False
        self.csv_file = None
        self.csv_writer = None
        self.event_times = []
        self.filename = None
        self.session_dir = None
        self._index = 0
        self.emg_channel_count = num_emg_channels # auto: 1 channel
        self.accel_channel_count = num_accel_channels # auto: all 3
        self.last_event_time = 0 # track the last event time for automatic marking
        self._marked_events = set() # track which events had been written to the csv
        logging.info(f"[Recorder] initialized for {self.emg_channel_count} EMG and {self.accel_channel_count} Accel channels")

    def start_recording(self):
        ''' Initializes CSV files'''
        
        if self.recording:
            return
        
        timestamp = time.strftime("%Y-%m-%d_%Hh%Mm%S", time.localtime())
        base_dir = os.getcwd() # create recordings folder in current directory
        self.session_dir = os.path.join(base_dir, "emg-recordings", timestamp)
        os.makedirs(self.session_dir, exist_ok=True)

        self.filename = os.path.join(self.session_dir, "emg_accel.csv")

        try:
            self.csv_file = open(self.filename, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)

            emg_headers = [f'ch{i+1} (µV)' for i in range(self.emg_channel_count)]
            accel_headers = [f'accel_{axis}' for axis in ['x', 'y', 'z']]

            header = ['timestamp'] + emg_headers + accel_headers + ['event']
            self.csv_writer.writerow(header)
            self.csv_file.flush()

            self.recording = True
            self._marked_events.clear() # reset marked events for new recording 
            logging.info(f"[Recorder] Recording started: {self.filename}")
        except Exception as e:
            logging.error(f"[Recorder] Failed to start recording: {e}")
            self.recording = False
        
    def record_data_point(self, timestamp, emg_values, accel_values):
        '''Records single row of EMG and Accel data in csv file'''
        if not self.recording or not self.csv_writer:
            return

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
            
            # Event flag logic
            event_flag = 0
            for event_time in self.event_times:
                
                # Check if current timestamp is within 10ms of any event time
                if abs(timestamp - event_time) < 0.01:
                    
                    # Only set flag for unmarked events
                    if event_time not in self._marked_events:
                        event_flag = 1
                        self._marked_events.add(event_time)
                        logging.info(f"[Recorder] Event marked at {timestamp:.3f}s")
                    break
            
            # safety filters
            emg_values = emg_values[:self.emg_channel_count]
            accel_values = accel_values[:self.accel_channel_count]
            
            # Format Row: Timestamp | Ch1 | Ch2 ... | AccelX | ... | EventFlag
            formatted_emg = [f"{v:.2f}" for v in emg_values]
            formatted_accel = [f"{v:.2f}" for v in accel_values]
            row = [f"{timestamp:.4f}"] + formatted_emg + formatted_accel + [int(event_flag)]
            self.csv_writer.writerow(row)

            # Flush periodically 
            if self._index % 10 == 0:
                self.csv_file.flush()
            self._index += 1

        except Exception as e:
            logging.warning(f"[Recorder] Failed to record data point: {e}")

    def stop_recording(self):
        if not self.recording:
            return
        
        try:
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
            self.csv_writer = None
            self.recording = False
            logging.info("[Recorder] Recording stopped")
        except Exception as e:
            logging.error(f"[Recorder] Failed to stop recording: {e}")

# ----- EMG Logic: Initialize board, thread and stream data ----- #
ganglion_instance = None

class GanglionData:
    def __init__(self, serial_port="COM4", mac_address=None, channel_list=None, sample_rate=200, buffer_seconds=2):
        '''Initialize Ganglion board system'''
        
        self.serial_port = serial_port
        self.mac_address = mac_address
        self._selected_channels = channel_list or [0] # default EMG channel 1 + accel channels
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.buffer_len = sample_rate * buffer_seconds

        self.board_shim = None
        self.board_id = BoardIds.GANGLION_BOARD.value

        self.emg_channels = BoardShim.get_exg_channels(self.board_id)
        self.accel_channels = BoardShim.get_accel_channels(self.board_id)

        if not self.emg_channels or not self.accel_channels:
             logging.error("BrainFlow did not return EMG or ACCEL channels!")

        self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
        self.actual_sample_rate = BoardShim.get_sampling_rate(self.board_id)

        self._all_channels = self.emg_channels + self.accel_channels
        
        self._buffers = {ch: deque(maxlen=self.buffer_len) for ch in self._selected_channels}
        self._timestamps = {ch: deque(maxlen=self.buffer_len) for ch in self._selected_channels}

        self._outlets = {} # LSL StreamOutlets for each channel
        self._accel_outlet = None # LSL StreamOutlets for accel channel
        
        self._stop_event = threading.Event() # create threading flag for EMG stream
        self._emg_thread = None
        self._start_time = None

        self.recorder = RealTimeRecorder(
            num_emg_channels=len(self.emg_channels),
            num_accel_channels=len(self.accel_channels))
    
    def stream_emg_thread(self, loop: asyncio.AbstractEventLoop):
        '''
        Connect to Ganglion board and initialize EMG stream
        Background thread to acquire data, stream to LSL and display on frontend
        '''
        BoardShim.enable_dev_board_logger()
        
        # Ganglion set-up and initialization
        params = BrainFlowInputParams()
        params.serial_port = self.serial_port
        if self.mac_address:
            params.mac_address = self.mac_address
        self.board_shim = BoardShim(self.board_id, params)

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
            self.board_shim.prepare_session()
            self.board_shim.config_board("n") # Turn on accelerometer
            self.board_shim.start_stream()
            logging.info("Streaming EMG and Accel data...")

            self.recorder.start_recording() # Start recording in CSV file
            
            num_points = 20
            while not self._stop_event.is_set(): # ensure EMG threading flag is down
                
                # Wait for data to accumulate
                if self.board_shim.get_board_data_count() < num_points:
                    time.sleep(0.01)
                    continue

                # Read and remove current data chunk
                data = self.board_shim.get_board_data() 
                if data.size == 0:
                    continue
                
                # Record raw CSV data
                if hasattr(self, 'recorder') and self.recorder.recording:
                    rec_timestamps = data[self.timestamp_channel]

                    # Start time edge case
                    rec_start_time = self._start_time if self._start_time else rec_timestamps[0]
                    rec_relative_times = rec_timestamps - rec_start_time

                    # Transpose and write to CSV files
                    for i in range(len(rec_timestamps)):
                        curr_emg_ch = data[self.emg_channels, i]
                        curr_accel_ch = data[self.accel_channels, i]

                        self.recorder.record_data_point(rec_relative_times[i], 
                                                        curr_emg_ch, curr_accel_ch)

                # Channel processing: Filtering, Buffering, Pushing to LSL & Frontend
                timestamps = data[self.timestamp_channel] 
                if self._start_time is None:
                    self._start_time = timestamps[0]
                relative_times = timestamps - self._start_time

                # Accel channel processing
                accel_data = data[self.accel_channels]

                self._accel_outlet.push_chunk(accel_data.T.tolist()) # push to LSL
                json_accel = json.dumps({
                    "type": "accel_data",
                    "timestamp": relative_times.tolist(),
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
                            self._timestamps[ch].extend(relative_times.tolist())

                            # Push to LSL
                            if emg_channel in self._outlets:
                                chunk = [[float(val)] for val in emg_data]
                                self._outlets[emg_channel].push_chunk(chunk)

                            # Push to Frontend
                            json_data = json.dumps({
                                "type": "raw_data",
                                "channel_index": ch,
                                "timestamp": relative_times.tolist(),
                                "value": emg_data.tolist()})
                            # safely place async task onto event loop's queue
                            asyncio.run_coroutine_threadsafe(manager.broadcast(json_data), loop)

        except Exception as e:
            logging.error(f"Error in EMG thread: {e}", exc_info=True)
        
        finally:
            logging.info("Stopping EMG stream...")
            if self.board_shim and self.board_shim.is_prepared():
                try:
                    if hasattr(self, 'recorder') and self.recorder.recording:
                        self.recorder.stop_recording()
                    self.board_shim.stop_stream()
                    self.board_shim.release_session()
                except Exception as e:
                    logging.warning(f"Error during board release: {e}")
            logging.info("Session released.")

    def start(self, loop):
        self._stop_event.clear()
        self._emg_thread = threading.Thread(target=self.stream_emg_thread, args=(loop,))
        self._emg_thread.start()

    def stop(self):
        self._stop_event.set()
        if hasattr(self, '_emg_thread') and self._emg_thread.is_alive():
            self._emg_thread.join()