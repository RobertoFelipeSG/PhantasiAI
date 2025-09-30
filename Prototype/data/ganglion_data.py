import time
import threading
from collections import deque
import numpy as np
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from pylsl import StreamInfo, StreamOutlet

from config.config_manager import load_config

class GanglionData:
    """
    Manages live data acquisition from a Ganglion board using BrainFlow
    """

    def __init__(self, port=None, channel_list=None, sample_rate=200, buffer_seconds=2):
        """
        Initialize the Ganglion board acquisition system.

        Parameters:
        - port: Serial port of the Ganglion board (e.g., "/dev/ttyUSB0").
                If None, loaded from config file.
        - channels: List of channel indices to use (0-3 for Ganglion).
        - sample_rate: Number of samples per second (Ganglion default is 200Hz).
        - buffer_seconds: Number of seconds to keep in memory.
        """
        # If no port provided, fetch from config
        if port is None:
            cfg = load_config()
            port = cfg.get("ganglion_port")

        self.port = port
        self._selected_channels = channel_list or [0,1,2,3]
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.buffer_len = sample_rate * buffer_seconds

        self.board = None
        self.board_id = BoardIds.GANGLION_BOARD.value #BoardIds.GANGLION_NATIVE_BOARD.value
        
        # Get channel information from BrainFlow
        self.emg_channels = BoardShim.get_exg_channels(self.board_id)
        self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
        self.actual_sample_rate = BoardShim.get_sampling_rate(self.board_id)

        # Buffers to store recent values and timestamps for each channel
        self._buffers = {ch: deque(maxlen=self.buffer_len) for ch in self._selected_channels}
        self._timestamps = {ch: deque(maxlen=self.buffer_len) for ch in self._selected_channels}

        self._outlets = {}  # LSL StreamOutlets for each channel
        self._stop_event = threading.Event()
        self._acquisition_thread = None
        self._start_time = None

    def connect(self):
        """
        Connect to the Ganglion board and initialize EMG streams.

        Returns:
        - True if connection is successful.
        - False if the board cannot be connected.
        """
        try:
            # Enable BrainFlow logging
            BoardShim.enable_dev_board_logger()
            
            # Create BrainFlow parameters
            params = BrainFlowInputParams()
            params.mac_address = "F5:75:72:A1:77:03"
            if self.port:
                params.serial_port = self.port
            
            # Initialize board
            self.board = BoardShim(self.board_id, params)
            
            # Prepare session and start stream
            self.board.prepare_session()
            self.board.start_stream()
            
            # Create LSL streams for each channel
            for ch in self._selected_channels:
                info = StreamInfo(
                    name=f"EMG_Channel_{ch+1}",
                    type="EMG",
                    channel_count=1,
                    nominal_srate=self.actual_sample_rate,
                    channel_format='float32',
                    source_id=f"ganglion_ch_{ch}"
                )
                self._outlets[ch] = StreamOutlet(info)

            return True
        except Exception as e:
            print(f"Failed to connect to Ganglion board: {e}")
            return False

    def _acquire_data(self):
        """
        Background thread function to acquire data from the Ganglion board.
        """
        while not self._stop_event.is_set():
            try:
                # Wait for data to accumulate
                while self.board.get_board_data_count() < 50:
                    time.sleep(0.01)
                    if self._stop_event.is_set():
                        return

                # Get current board data
                data = self.board.get_current_board_data(50)
                
                if data.size == 0:
                    continue

                # Apply low pass filter to remove high frequency noise
                for ch in self._selected_channels:
                    if ch < len(self.emg_channels):
                        emg_channel = self.emg_channels[ch]
                        if emg_channel < data.shape[0]:
                            # Apply 50Hz low pass filter
                            DataFilter.perform_lowpass(
                                data[emg_channel], 
                                self.actual_sample_rate, 
                                50.0, 4, 
                                FilterTypes.BUTTERWORTH.value, 0
                            )

                # Process each channel
                for ch in self._selected_channels:
                    if ch < len(self.emg_channels):
                        emg_channel = self.emg_channels[ch]
                        if emg_channel < data.shape[0]:
                            emg_data = data[emg_channel]
                            timestamps = data[self.timestamp_channel]
                            
                            # Convert timestamps to relative time
                            if self._start_time is None:
                                self._start_time = timestamps[0]
                            
                            relative_times = timestamps - self._start_time
                            
                            # Store in buffers
                            for i, (t, val) in enumerate(zip(relative_times, emg_data)):
                                self._buffers[ch].append(float(val))
                                self._timestamps[ch].append(float(t))
                                
                                # Push to LSL
                                if ch in self._outlets:
                                    self._outlets[ch].push_sample([float(val)])

            except Exception as e:
                print(f"Error in data acquisition: {e}")
                time.sleep(0.1)

    def start(self):
        """
        Start data acquisition thread.
        """
        if not self.board:
            raise RuntimeError("Must call connect() before start()")

        self._stop_event.clear()
        self._acquisition_thread = threading.Thread(target=self._acquire_data, daemon=True)
        self._acquisition_thread.start()

    def stop(self):
        """
        Stop data acquisition thread cleanly.
        """
        self._stop_event.set()
        if self._acquisition_thread:
            self._acquisition_thread.join()
            self._acquisition_thread = None

    def read_latest(self, selected_indices=None):
        """
        Return the most recent value from each channel buffer
        """
        values = []
        for ch in self._selected_channels:
            buf = self._buffers[ch]
            values.append(buf[-1] if buf else 0.0)

        if selected_indices is not None:
            return [values[i] for i in selected_indices]

        return values

    def close(self):
        """
        stop acquisition and disconnect from the Ganglion board
        """
        self.stop()
        if self.board:
            try:
                self.board.stop_stream()
                self.board.release_session()
            except:
                pass
            self.board = None

    @property
    def channels(self):
        """Return the list of available channels"""
        return list(range(len(self.emg_channels)))

    def get_channel_count(self):
        """Return the number of available channels"""
        return len(self.emg_channels)
