import time
import threading
import asyncio
import json
from collections import deque
from pathlib import Path

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
from pylsl import StreamInfo, StreamOutlet

from config.connection_manager import manager, logging
from config.config_manager import load_config
from emg.recorder import RealTimeRecorder
from emg.emg_feature_extractor import FeatureExtractor
from emg.emg_peak_classifier import PeakClassifier

config = load_config() # load config settings

# ----- EMG Logic: Initialize board, thread and stream data ----- #
ganglion_instance = None

class GanglionData:
    def __init__(self, num_trials=5):
        '''Initialize Ganglion board system'''

        self.base_path = Path(__file__).resolve().parent
        
        self.serial_port = config.get("serial_port")
        self.mac_address = config.get("mac_address")
        self._selected_channels = config.get("selected_channels")
        self._sample_rate = config.get("sample_rate")
        self._buffer_seconds = config.get("data_acq_buffer_seconds")
        self._buffer_len = self._sample_rate * self._buffer_seconds
        self._num_points = config.get("num_points") # number of data points processed at once

        self._stop_event = threading.Event() # create threading flag for EMG stream
        self._emg_thread = None
        self._session_start_time = None
        
        self._num_trials = int(num_trials) # trials per session (inputed by user)
        self.next_trial_block = int(num_trials)
        self._total_events = 0
        self.marker_broadcast_counter = 0
        
        self._outlets = {} # LSL StreamOutlets for each channel
        self._accel_outlet = None # LSL StreamOutlets for accel channel

        self.board_shim = None
        self.board_id = BoardIds.GANGLION_BOARD.value
        self.emg_channels = BoardShim.get_exg_channels(self.board_id)
        self.accel_channels = BoardShim.get_accel_channels(self.board_id)

        self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
        self.actual_sample_rate = BoardShim.get_sampling_rate(self.board_id)
        self._all_channels = self.emg_channels + self.accel_channels
        
        self._buffers = {ch: deque(maxlen=self._buffer_len) for ch in self._selected_channels}
        self._timestamps = {ch: deque(maxlen=self._buffer_len) for ch in self._selected_channels}

        self.recorder = RealTimeRecorder(self.base_path)
        self.feature_extractor = FeatureExtractor()
        self.peak_classifier = PeakClassifier(self.base_path)
            
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
                    channels = [f"ch{ch + 1} (µV)" for ch in config.get("selected_channels", [])]
                    features_df = self.feature_extractor.run(analysis_df=analysis_df, channels=channels)
                    
                    # Peak Classification
                    class_folder = f"{int(curr_timestamp)}_classification"
                    output_path = Path(self.recorder.classification_dir) / class_folder
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    self.peak_classifier.run(features_df=features_df, output_path=output_path)

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
            self.marker_broadcast_counter += len(timestamps)
            
            # Only broadcast remaining time every x samples
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
                    DataFilter.perform_lowpass(emg_data, self.actual_sample_rate, 
                                               config.get("cutoff_freq"), config.get("cutoff_order"),
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
        params = BrainFlowInputParams()
        params.serial_port = self.serial_port
        if self.mac_address:
            params.mac_address = self.mac_address
        self.board_shim = BoardShim(self.board_id, params)

        # LSL initialization for each channel
        for ch in self._selected_channels:
                info = StreamInfo(name=f"EMG_Channel_{ch+1}", type="EMG", channel_count=config.get("num_emg_ch"), 
                    nominal_srate=self.actual_sample_rate, channel_format='float32',
                    source_id=f"ganglion_ch_{ch}"
                )
                self._outlets[ch] = StreamOutlet(info)

        # LSL initialization for accelerometer (handles multiple channels at once)
        accel_info = StreamInfo(name="Ganglion_Accel", type="MOCAP", channel_count=config.get("num_accel_ch"),
                                nominal_srate=self.actual_sample_rate, channel_format='float32',
                                source_id="ganglion_accel")
        self._accel_outlet = StreamOutlet(accel_info)
        
        try:
            self.board_shim.prepare_session()
            self.board_shim.config_board("n") # Turn on accelerometer
            self.board_shim.start_stream()
            logging.info("Streaming EMG and Accel data...")

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