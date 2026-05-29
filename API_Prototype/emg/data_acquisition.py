import os
import signal
import time
import threading
import asyncio
import json
import queue
import numpy as np
from collections import deque
from pathlib import Path

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError, BrainFlowExitCodes
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
    def __init__(self, websocket, profiler, stim_flag, stim_state, isi_type='static', serial_port="serial_port_A", sample_rate=200, 
                 num_trials=1, session_size=400, folder_name=None, on_error=None):
        '''Initialize Ganglion board system'''

        self.websocket = websocket
        self.profiler = profiler
        self.stim_flag = stim_flag
        self.stim_state = stim_state
        self.on_error = on_error
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
        self._last_active = None
        self.connection_status = "pending"
        self.stream_lost = False
        self.in_break = False
        
        self._num_trials = int(num_trials) # trials per analysis session (default is 1; single trial analysis)
        self.next_trial_block = int(num_trials)
        self._total_events = 0
        self._total_trials = 0
        self._broadcast_counter = 0
        
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

        self.recorder = RealTimeRecorder(self._sample_rate, session_size, self.profiler, self.stim_flag, self.stim_state, isi_type=isi_type,
                                         base_path=self.base_path, folder_name=folder_name)
        self.feature_extractor = FeatureExtractor(self._sample_rate, self.profiler, self._num_trials == 1, 
                                                  output_path=Path(self.recorder.features_dir))
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
            features_df = self.feature_extractor.run(self._total_trials,
                                        analysis_df=analysis_df,
                                        curr_timestamp=int(curr_timestamp),
                                        channels=channels)
            if features_df is None:
                logging.error("[Ganglion] Skipping optimization + stimulation because no features detected.")
            
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
            "event_timestamps": new_event_times
        })
        asyncio.run_coroutine_threadsafe(self.websocket.send_text(event_data), loop)

    def _broadcast_event_countdown(self, loop):
        '''Helper to broadcast event countdown until next event to frontend'''
        
        next_event_time = self.recorder.next_event_time
        
        marker_data = json.dumps({
            "type": "event_target_time", 
            "event_target_time": float(next_event_time)
        })
        asyncio.run_coroutine_threadsafe(self.websocket.send_text(marker_data), loop)

    def _broadcast_trials(self, loop):
        '''Helper to broadcast trial times to frontend'''
        new_trial_times = self.recorder.trial_times_buffer[:]
        self.recorder.trial_times_buffer.clear() # Clear event times after capturing

        trial_data = json.dumps({
            "type": "trial_times",
            "trial_timestamps": new_trial_times # first timestamp of the new trial
        })
        asyncio.run_coroutine_threadsafe(self.websocket.send_text(trial_data), loop)
    
    def _broadcast_trial_countdown(self, loop):
        '''Helper to broadcast trial countdown until next event to frontend'''
        
        next_trial_time = self.recorder.next_trial_time
        
        marker_data = json.dumps({
            "type": "trial_target_time", 
            "trial_target_time": float(next_trial_time)
        })
        asyncio.run_coroutine_threadsafe(self.websocket.send_text(marker_data), loop)

    def _broadcast_trial_completion(self, loop):
        '''Helper to broadcast trial completion to frontend'''
        curr_total_trials = float(self._total_trials)
        
        trial_data = json.dumps({
            "type": "trial_completion",
            "total_trials": curr_total_trials
        })
        asyncio.run_coroutine_threadsafe(self.websocket.send_text(trial_data), loop)

    def _broadcast_ready_countdown(self, loop):
        '''Helper to broadcast event countdown until next 'ready' phase to frontend
            Note: Ready phase = start of electrical stimulation'''
        
        next_ready_time = self.recorder.next_ready_time
        
        phase_data = json.dumps({
            "type": "ready_target_time", 
            "ready_target_time": float(next_ready_time)
        })
        asyncio.run_coroutine_threadsafe(self.websocket.send_text(phase_data), loop)

    def _broadcast_break_countdown(self, loop):
        '''Helper to broadcast time until break is over to frontend'''
         
        break_end_time = self.recorder.break_end_time
        break_data = json.dumps({
            "type": "break_end_time", 
            "break_end_time": float(break_end_time)
        })
        asyncio.run_coroutine_threadsafe(self.websocket.send_text(break_data), loop)
            
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
            
            # record data point
            break_status, has_event, new_trial = self.recorder.record_data_point(
                curr_timestamp, curr_emg_ch, curr_accel_ch)
            
            # broadcast break state to frontend
            if break_status == "break_started":
                self.in_break = True
                break_start_msg = json.dumps({"type": "break_status", "status": "started"})
                asyncio.run_coroutine_threadsafe(self.websocket.send_text(break_start_msg), loop)
            elif break_status == "break_ended":
                self.in_break = False
                break_end_msg = json.dumps({"type": "break_status", "status": "ended"})
                asyncio.run_coroutine_threadsafe(self.websocket.send_text(break_end_msg), loop)
            
            # signal processing ONLY if not in break
            if not self.in_break:
                if has_event: 
                    self._total_events = self.recorder.total_events
                    #logging.info(f"[Ganglion] Recorded {self._total_events} events")
                
                if new_trial:
                    self._total_trials = self.recorder.total_trials
                    self._broadcast_trial_completion(loop)

                # Perform analysis on selected trials
                if self._total_trials == self.next_trial_block:
                    self._handle_analysis(curr_timestamp)
            
        if not self.in_break: # Send markers for real-time graphs
            # Broadcast event markers
            if self.recorder.event_times_buffer: 
                self._broadcast_events(loop)

            # Broadcast trial markers
            if self.recorder.trial_times_buffer:
                self._broadcast_trials(loop)
        
        # Broadcast marker interval and trial countdown, and ready countdown to frontend
        self._broadcast_counter += len(relative_timestamps)
        if self._broadcast_counter >= self._num_points:
            if not self.in_break: # Send countdowns for real-time instructions
                self._broadcast_event_countdown(loop) 
                self._broadcast_trial_countdown(loop) 
                self._broadcast_ready_countdown(loop) # currently the benchmark for live animation updates
            else: self._broadcast_break_countdown(loop)
            
            self._broadcast_counter = 0

        # Accel channel processing
        accel_data = data[self.accel_channels]

        self._accel_outlet.push_chunk(accel_data.T.tolist()) # push to LSL
        '''json_accel = json.dumps({
            "type": "accel_data",
            "timestamp": relative_timestamps.tolist(),
            "value": accel_data.tolist()
        })
        asyncio.run_coroutine_threadsafe(self.websocket.send_text(json_accel), loop)'''
        
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
            self._last_active = time.time() # initial time for data stream check
            logging.info("[Ganglion] Streaming EMG and Accel data...")

            if self.recorder:
                self.recorder.start_recording()
            
            while not self._stop_event.is_set(): # ensure EMG threading flag is down
                
                # check if data is still being processed
                if time.time() - self._last_active > CONFIG.get('board_data_timeout'):
                    self.stream_lost = True
                    raise RuntimeError(f"[Ganglion] No EMG data received for {CONFIG.get('board_data_timeout')}s")
                
                # Wait for data to accumulate
                if self.board_shim.get_board_data_count() < self._num_points:
                    time.sleep(0.01)
                    continue

                self._last_active = time.time() # update time for data stream check
                
                # Processing: Filtering, Buffering, Writing to CSV, Pushing to LSL & Frontend
                if self.recorder is not None and self.recorder.recording:
                    processed = self._process_data(loop)
                    if not processed:
                        continue
                else: 
                    logging.warning("[Ganglion] Waiting for recorder to be initialized...")

        except BrainFlowError as e:
            self.connection_status = "failed"
            logging.error(f"[Ganglion] Brainflow error: {e}")
            error_type = "general_brainflow_error"
            message = f"Brainflow error: {e}"

            if e.exit_code == BrainFlowExitCodes.ANOTHER_BOARD_IS_CREATED_ERROR.value:
                error_type = "board_already_in_use"
            elif e.exit_code == BrainFlowExitCodes.PORT_ALREADY_OPEN_ERROR.value:
                error_type = "port_already_in_use"

            error_status = {"status": "error", 
                            "type": error_type, 
                            "message": message}
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_json(error_status), loop)

            if error_type == "board_already_in_use" or error_type == "port_already_in_use":
                logging.warning(f"[Ganglion] Restart board due to {error_type} error")
                time.sleep(1.0) # allow time to notify user
                os.kill(os.getpid(), signal.SIGTERM) # restart main application
        
        except Exception as e:
            self.connection_status = "failed"
            logging.error(f"[Ganglion] Error in EMG thread: {e}") #, exc_info=True)

            # if stream was lost, no data received for X seconds
            if self.stream_lost: 
                if hasattr(self, 'websocket') and hasattr(self, 'on_error'):
                    error_status = {"status": "error", "type": "data_timeout"}
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send_json(error_status), loop)
    
                    if self.on_error:
                        loop.call_soon_threadsafe(self.on_error)

            else:
                message = f"EMG thread error: {e}"
                error_status = {"status": "error", 
                                "type": "general_EMG_error",
                                "message": message}
                asyncio.run_coroutine_threadsafe(
                        self.websocket.send_json(error_status), loop)
        
        finally:
            if self.recorder and self.recorder.recording:
                self.recorder.stop_recording()
                self.recorder = None
            if self.feature_extractor: self.feature_extractor = None
            #if self.classifier: self.classifier =  None
            
            if self.board_shim and self.board_shim.is_prepared():   
                try:
                    if not self.stream_lost:
                        self.board_shim.stop_stream()
                        self.board_shim.release_session()
                        logging.info("[Ganglion] Stopped stream and released session.")
                except Exception as e:
                    logging.error(f"[Ganglion] Error during board release: {e}")
    
    def start(self, loop):
        if getattr(self, "_emg_thread", None) is not None and self._emg_thread.is_alive():
            logging.warning("[Ganglion] Attempted to start EMG thread but one is already running.")
            return
        
        self.connection_status = "pending"
        self._stop_event.clear()
        self._emg_thread = threading.Thread(target=self._stream_emg_thread, args=(loop,))
        self._emg_thread.start()        

    def stop(self):
        self.connection_status = "pending"
        self._stop_event.set()

        if getattr(self, "_emg_thread", None) and self._emg_thread.is_alive():
            self._emg_thread.join(timeout=5.0)
            if self._emg_thread.is_alive():
                logging.error("[Ganglion] EMG thread failed to exit cleanly within timeout.")