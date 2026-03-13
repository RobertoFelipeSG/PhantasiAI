import time
import threading
import asyncio
import json
import queue
import numpy as np
from collections import deque
from pathlib import Path

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
from pylsl import StreamInfo, StreamOutlet

from config.connection_manager import manager, logging
from config.config_manager import load_config
from emg.recorder import RealTimeRecorder
from emg.emg_feature_extractor import FeatureExtractor
#from emg.emg_peak_classifier import PeakClassifier

CONFIG = load_config() # load config settings

# ----- EMG Logic: Initialize board, thread and stream data ----- #

class GanglionData:
    def __init__(self, websocket, serial_port="serial_port_A", sample_rate=200, num_trials=1, folder_name=None):
        '''Initialize Ganglion board system'''

        self.websocket = websocket
        self.base_path = Path(__file__).resolve().parent.parent / "data"
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.serial_port = CONFIG.get(serial_port)
        self.mac_address = CONFIG.get("mac_address")
        self._selected_channels = CONFIG.get("selected_channels")
        self._buffer_seconds = CONFIG.get("data_acq_buffer_seconds")
        self._num_points = CONFIG.get("num_points") # minimum chunk size

        self._stop_event = threading.Event() # create threading flag for EMG stream
        self._emg_thread = None
        self._session_start_time = None
        self.connection_status = "pending"
        
        self._num_trials = int(num_trials) # trials per session (inputed by user)
        self.next_trial_block = int(num_trials)
        self._total_events = 0
        self._total_trials = 0
        self.marker_broadcast_counter = 0
        
        # Data containers
        self._sample_rate = sample_rate
        self._buffer_len = self._sample_rate * self._buffer_seconds
        self._buffers = {ch: deque(maxlen=self._buffer_len) for ch in self._selected_channels}
        self._timestamps = {ch: deque(maxlen=self._buffer_len) for ch in self._selected_channels}

        # LSL StreamOutlets
        self._outlets = {} # EMG
        self._accel_outlet = None 

        self.board_shim = None
        self.board_id = BoardIds.GANGLION_BOARD.value
        self.emg_channels = BoardShim.get_exg_channels(self.board_id)
        self.accel_channels = BoardShim.get_accel_channels(self.board_id)

        self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
        self.actual_sample_rate = BoardShim.get_sampling_rate(self.board_id)
        self._all_channels = self.emg_channels + self.accel_channels

        self.recorder = RealTimeRecorder(sample_rate=self._sample_rate, base_path=self.base_path, folder_name=folder_name)
        self.feature_extractor = FeatureExtractor(self._sample_rate, self._num_trials == 1, output_path=Path(self.recorder.features_dir))
        #self.peak_classifier = PeakClassifier(self.base_path)

    def _handle_analysis(self, curr_timestamp: float):
        ''' Helper to handle analysis: peak extraction + classification'''
        
        # Create analysis DataFrame
        #logging.info(f"[Ganglion] Creating analysis file for {self._total_trials} trials, from {self.recorder.min_time} to {curr_timestamp} seconds")
        analysis_df = self.recorder.create_analysis_file(curr_timestamp, self._total_trials)
        
        # Signal processing of analysis file
        if analysis_df is None:
            logging.error("[Ganglion] Skipping feature extraction because analysis_df is None.")
        else:
            # Peak detection
            channels = [f"ch{ch + 1} (µV)" for ch in CONFIG.get("selected_channels", [])]
            self.feature_extractor.run(analysis_df=analysis_df,
                                        curr_timestamp=int(curr_timestamp),
                                        channels=channels)
            
            '''
            # Peak Classification
            self.peak_classifier.run(features_df=features_df, 
                                     output_path=Path(self.recorder.classification_dir), 
                                     curr_timestamp=int(curr_timestamp))
            '''
        
        # Advance to next trial block
        while self._total_trials == self.next_trial_block:
            self.next_trial_block += self._num_trials
    
    def _broadcast_events(self, loop):
        '''Helper to broadcast event marker times to frontend'''
        new_event_times = self.recorder.event_times_buffer[:]
        self.recorder.event_times_buffer.clear() # Clear event times after capturing

        event_data = json.dumps({
            "type": "event_times",
            "timestamps": new_event_times
        })
        asyncio.run_coroutine_threadsafe(self.websocket.send_text(event_data), loop)

    def _broadcast_countdown(self, loop):
        '''Helper to broadcast countdown until next event to frontend'''
        
        next_event_time = self.recorder.next_event_time
        
        marker_data = json.dumps({
            "type": "marker_target_time", 
            "target_timestamp": float(next_event_time)
        })
        asyncio.run_coroutine_threadsafe(self.websocket.send_text(marker_data), loop)
            
    def _process_data(self, loop):
        '''Complete data processing for current chunk '''
        
        # Read and remove current data chunk
        data = self.board_shim.get_board_data() 
        if data.size == 0: return False

        timestamps = data[self.timestamp_channel] # get raw timestamps
        
        # Set session start time and calculate relative timestamps
        if self._session_start_time is None:
            self._session_start_time = timestamps[0]
        relative_timestamps = timestamps - self._session_start_time
        
        # Transpose and write raw data to CSV files
        for i in range(len(timestamps)):
            curr_emg_ch = data[self.emg_channels, i]
            curr_accel_ch = data[self.accel_channels, i]
            curr_timestamp = relative_timestamps[i]

            has_event, end_trial = self.recorder.record_data_point(curr_timestamp, curr_emg_ch, curr_accel_ch)
            
            if has_event: 
                self._total_events += 1
                logging.info(f"[Ganglion] Recorded {self._total_events} events")
            if end_trial:
                self._total_trials += 1
                logging.info(f"[Ganglion] Trial {self._total_trials} complete")
            
            # Perform analysis on selected trials
            if self._total_trials == self.next_trial_block:
                self._handle_analysis(curr_timestamp)
            
        # Broadcast event markers
        if self.recorder.event_times_buffer: 
            self._broadcast_events(loop)
        
        # Broadcast marker interval countdown to frontend
        self.marker_broadcast_counter += len(relative_timestamps)
        if self.marker_broadcast_counter >= self._num_points:
            self._broadcast_countdown(loop) 
            self.marker_broadcast_counter = 0

        # Accel channel processing
        accel_data = data[self.accel_channels]

        self._accel_outlet.push_chunk(accel_data.T.tolist()) # push to LSL
        json_accel = json.dumps({
            "type": "accel_data",
            "timestamp": relative_timestamps.tolist(),
            "value": accel_data.tolist()
        })
        asyncio.run_coroutine_threadsafe(self.websocket.send_text(json_accel), loop)
        
        # EMG channel processing
        for ch in self._selected_channels:
            
            if ch < len(self.emg_channels):
                emg_channel = self.emg_channels[ch]
                
                if emg_channel < data.shape[0]:
                    emg_data = data[emg_channel] 
                    
                    # Apply 50Hz low pass filter to remove high freq noise
                    DataFilter.perform_lowpass(emg_data, self.actual_sample_rate, 
                                               CONFIG.get("cutoff_freq"), CONFIG.get("cutoff_order"),
                                               FilterTypes.BUTTERWORTH.value, 0)

                    # Store in buffers
                    self._buffers[ch].extend(emg_data.tolist())
                    self._timestamps[ch].extend(relative_timestamps.tolist())

                    # Push to LSL
                    if emg_channel in self._outlets:
                        chunk = [[float(val)] for val in emg_data]
                        self._outlets[emg_channel].push_chunk(chunk)

                    # Push to Frontend
                    json_emg = json.dumps({
                        "type": "emg_data",
                        "channel_index": ch,
                        "timestamp": relative_timestamps.tolist(),
                        "value": emg_data.tolist()})
                    # safely place async task onto event loop's queue
                    asyncio.run_coroutine_threadsafe(self.websocket.send_text(json_emg), loop)
        
        return True
    
    def _stream_emg_thread(self, loop: asyncio.AbstractEventLoop):
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
                info = StreamInfo(name=f"EMG_Channel_{ch+1}", type="EMG", channel_count=CONFIG.get("num_emg_ch"), 
                    nominal_srate=self.actual_sample_rate, channel_format='float32',
                    source_id=f"ganglion_ch_{ch}"
                )
                self._outlets[ch] = StreamOutlet(info)

        # LSL initialization for accelerometer (handles multiple channels at once)
        accel_info = StreamInfo(name="Ganglion_Accel", type="MOCAP", channel_count=CONFIG.get("num_accel_ch"),
                                nominal_srate=self.actual_sample_rate, channel_format='float32',
                                source_id="ganglion_accel")
        self._accel_outlet = StreamOutlet(accel_info)
        
        try:
            self.board_shim.prepare_session()
            self.board_shim.config_board("n") # Turn on accelerometer
            self.board_shim.start_stream()
            self.connection_status = "connected"
            logging.info("[Ganglion] Streaming EMG and Accel data...")

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
                    logging.warning("[Ganglion] Waiting for recorder to be initialized...")

        except Exception as e:
            logging.error(f"[Ganglion] Error in EMG thread: {e}", exc_info=True)
            self.connection_status = "failed"
        
        finally:
            logging.info("[Ganglion] Stopping EMG stream...")
            if self.recorder and self.recorder.recording:
                self.recorder.stop_recording()
                self.recorder = None
            if self.feature_extractor: self.feature_extractor = None
            #if self.classifier: self.classifier =  None
            
            if self.board_shim and self.board_shim.is_prepared():
                try:
                    self.board_shim.stop_stream()
                    self.board_shim.release_session()
                except Exception as e:
                    logging.warning(f"[Ganglion] Error during board release: {e}")
            logging.info("[Ganglion] Session released.")
    
    def start(self, loop):
        if getattr(self, "_emg_thread", None) is not None and self._emg_thread.is_alive():
            logging.warning("[Ganglion] Attempted to start EMG thread but one is already running.")
            return
        
        self.connection_status = "pending"
        self._stop_event.clear()
        self._emg_thread = threading.Thread(target=self._stream_emg_thread, args=(loop,))
        self._emg_thread.start()        

    def stop(self):
        self._stop_event.set()
        if hasattr(self, '_emg_thread') and self._emg_thread.is_alive():
            self._emg_thread.join()